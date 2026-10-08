"""Independent P4.2R3 verification; no recommendation model fitting.

Only the new-stage verification receipt is written. Old reconstruction,
calibration and matching helpers below are pure/read-only; old stage runners
and verification entry points are never invoked.
"""
from __future__ import annotations

import gc
import subprocess
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit

from .p41a_contract import check_identity, read_json, write_json
from .p41a_data import identities, parquet
from .p42_contract import FEATURE_SPEC, TAUS, WINDOWS, guard_cutoff
from .p42_propensity import MODEL_PARAMS, THREAD_LIMIT
from .p42_verify import _close, _prediction_list, _verify_calibration
from .p42r_contract import LOG1P_COUNTS
from .p42r_verify import _raw, _predict, independent_qw_matrix
from .p42r2_verify import (
    _verify_shared_reuse, independent_calibration_gate,
    independent_promotion, verify_admission_bundle,
)

INVARIANTS = [
    "P4_2_P4_2R_P4_2R2_history_preserved", "branch_main_not_Warm",
    "final_week_fail_closed", "W0_exact_parity", "M4_B0_frozen_parity",
    "qC_exact_P4_2R_repaired_lineage", "qW_feature_contract_exact_P4_2R2",
    "R1_cleanup_exact_rule", "global_R2_exact_19_counts",
    "training_only_transform_order_unchanged", "solver_LBFGS",
    "C_remains_1", "tol_remains_1e_7", "only_qW_max_iter_1200",
    "no_class_weight", "no_negative_sampling",
    "four_qW_chains_new_refits_under_1200",
    "calibration_only_after_four_solver_success",
    "utility_unchanged", "taus_unchanged", "exact_matching_unchanged",
    "no_hard_max1", "exactly12_unique_final_items",
    "no_edge_exact_W0", "all_trusted_SHA_comparisons_pass",
]
FAILURES = {"engineering_failure", "qW_solver_boundary_repair_failure", "propensity_calibration_failure"}


def verify_solver_sequence(attempts, calibration, admission, decision):
    """Check the actual captured optimizer result, never substitute gradient."""
    assert len(attempts) <= 4
    assert [a["window"] for a in attempts] == list(WINDOWS)[:len(attempts)]
    accepted = []
    for i, a in enumerate(attempts):
        assert a["side"] == "qW" and a["repair"] == "R2"
        solver = a["solver_result"]
        assert isinstance(solver["success"], bool)
        assert isinstance(solver["status"], int) and isinstance(solver["message"], str)
        assert 0 <= solver["nit"] <= 1200
        success = (solver["success"] and solver["status"] == 0
                   and solver["nit"] < 1200 and not a["convergence_warnings"])
        assert (a["status"] == "converged") == success
        accepted.append(success)
        if not success:
            assert i == len(attempts)-1
            assert decision == "qW_solver_boundary_repair_failure"
    complete = len(attempts) == 4 and all(accepted)
    if calibration or admission:
        assert complete, "outer computation before four actual solver successes"
    if decision == "qW_solver_boundary_repair_failure":
        assert attempts and not accepted[-1]
        assert not calibration and not admission
    return {"pass": True, "attempts": len(attempts), "solver_successes": sum(accepted),
            "all_four_converged": complete, "all_attempted_qW_new_budget": 1200,
            "gradient_does_not_override_solver_status": True}


def same_preprocessing(a, b):
    """Ignore provenance timestamps, not statistical preprocessing values."""
    return {k: v for k, v in a.items() if k != "lineage"} == {
        k: v for k, v in b.items() if k != "lineage"}


def independent_parameter_delta(frames, old_model, new_model, old_prep, new_prep):
    """Compute paired saved-model diagnostics on the identical training rows."""
    assert same_preprocessing(old_prep, new_prep)
    assert {k: v for k, v in old_model["params"].items() if k != "max_iter"} == {
        k: v for k, v in new_model["params"].items() if k != "max_iter"}
    old_coef = np.asarray(old_model["coefficient"], dtype=float)
    new_coef = np.asarray(new_model["coefficient"], dtype=float)
    assert old_coef.shape == new_coef.shape
    predictions, labels, old_scores, new_scores = [], [], [], []
    old_gradient = np.zeros(len(old_coef)+1)
    new_gradient = np.zeros(len(old_coef)+1)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=THREAD_LIMIT):
        for frame in frames:
            for start in range(0, len(frame), 16384):
                part = frame.iloc[start:start+16384]
                x = independent_qw_matrix(part, new_prep)
                old_s = x @ old_coef + old_model["intercept"]
                new_s = x @ new_coef + new_model["intercept"]
                old_scores.append(old_s)
                new_scores.append(new_s)
                predictions.append(np.abs(expit(new_s)-expit(old_s)))
                y_part = part.target.to_numpy(dtype=float)
                labels.append(y_part)
                for accumulator, score in ((old_gradient, old_s), (new_gradient, new_s)):
                    residual = expit(score)-y_part
                    accumulator[:-1] += x.T @ residual
                    accumulator[-1] += residual.sum()
    y = np.concatenate(labels)
    old_s, new_s, absolute = map(np.concatenate, (old_scores, new_scores, predictions))
    delta = new_coef-old_coef
    old_gradient = old_gradient/len(y) + np.r_[old_coef/len(y), 0.]
    new_gradient = new_gradient/len(y) + np.r_[new_coef/len(y), 0.]
    old_norm, new_norm = float(np.abs(old_gradient).max()), float(np.abs(new_gradient).max())
    old_nit, new_nit = old_model["n_iter"][0], new_model["n_iter"][0]
    return {
        "rows": len(y), "preprocessing_exact_except_lineage": True,
        "coefficient_order_exact": True,
        "old_iterations": old_nit, "new_iterations": new_nit,
        "additional_iterations_beyond_old_cap": max(0, new_nit-1000),
        "iteration_delta_vs_old": new_nit-old_nit,
        "coefficient_delta_l2": float(np.linalg.norm(delta)),
        "coefficient_delta_max_abs": float(np.abs(delta).max()),
        "intercept_delta": float(new_model["intercept"]-old_model["intercept"]),
        "old_training_log_loss": float(np.mean(np.logaddexp(0, old_s)-y*old_s)),
        "new_training_log_loss": float(np.mean(np.logaddexp(0, new_s)-y*new_s)),
        "training_log_loss_delta": float(np.mean(np.logaddexp(0, new_s)-y*new_s)-np.mean(np.logaddexp(0, old_s)-y*old_s)),
        "old_gradient_infinity_norm": old_norm,
        "new_gradient_infinity_norm": new_norm,
        "gradient_infinity_norm_delta": new_norm-old_norm,
        "training_prediction_abs_delta": {"mean": float(absolute.mean()),
            "p95": float(np.quantile(absolute, .95)), "max": float(absolute.max()),
            "quantile_method": "numpy linear"},
        "exact_coefficient_intercept_parity": bool(np.array_equal(old_coef, new_coef)
            and old_model["intercept"] == new_model["intercept"]),
        "exact_training_prediction_parity": bool(not absolute.any()),
        "diagnostic_only": True, "new_optimizer_calls": 0,
    }


def independent_qw_training(frames, prep, audit, model, spec=FEATURE_SPEC["qW"], expected_params=None):
    """Recompute all train-only medians, exact cleanup, scaler and gradient."""
    expected_params = dict(expected_params or {**MODEL_PARAMS, "max_iter": 1200})
    numeric, binary = spec["numeric"], spec["binary"]
    columns = numeric + binary + [name+"_available" for name in numeric]
    n = sum(len(frame) for frame in frames)
    assert n > 0 and prep["numeric"] == numeric and prep["binary"] == binary
    assert prep["input_columns"] == columns and prep["repair"] in ("R1", "R2")
    assert n == prep["training_rows"] == audit["training_rows"] == audit["fitted_rows"]
    assert audit["positives"] == sum(int(f.target.sum()) for f in frames)
    assert audit["base_rate"] == audit["positives"]/n
    assert model["params"] == expected_params
    # One bounded logical matrix. No duplicate n-by-p weighted Hessian copy.
    x = np.empty((n, len(columns)), dtype=np.float64)
    missing = []
    for j, name in enumerate(numeric):
        values = np.concatenate([_raw(f, name) for f in frames])
        finite = np.isfinite(values)
        median = float(np.median(values[finite])) if finite.any() else 0.
        assert prep["median"][name] == median, name
        assert prep["missing_training_rows"][name] == int((~finite).sum()), name
        if not finite.any():
            missing.append(name)
        if prep["repair"] == "R2" and name in LOG1P_COUNTS:
            assert not (values[finite] < 0).any(), name
        x[:, j] = np.where(finite, values, median)
        x[:, len(numeric)+len(binary)+j] = finite
    assert prep["all_missing_numeric"] == missing
    for j, name in enumerate(binary):
        values = np.concatenate([_raw(f, name) for f in frames])
        finite = np.isfinite(values)
        assert np.isin(values[finite], [0, 1]).all(), name
        x[:, len(numeric)+j] = np.where(finite, values, 0)
    # Independent exhaustive candidate comparison (range filter only, no sums).
    ranges = [(float(x[:, j].min()), float(x[:, j].max())) for j in range(len(columns))]
    constants = {columns[j]: lo for j, (lo, hi) in enumerate(ranges) if lo == hi}
    active = sorted((j for j, name in enumerate(columns) if name not in constants),
                    key=lambda j: (columns[j].endswith("_available"), columns[j]))
    survivors, duplicates = [], {}
    for j in active:
        equal = next((i for i in survivors if ranges[i] == ranges[j] and np.array_equal(x[:, i], x[:, j])), None)
        if equal is None:
            survivors.append(j)
        else:
            duplicates[columns[j]] = columns[equal]
    keep = sorted(survivors)
    cleanup = prep["cleanup"]
    assert cleanup == audit["cleanup"]
    assert {r["feature"]: r["value"] for r in cleanup["constant_dropped"]} == constants
    assert all(r["reason"] == "constant_on_training_matrix" for r in cleanup["constant_dropped"])
    assert {r["feature"]: r["survivor"] for r in cleanup["exact_duplicate_dropped"]} == duplicates
    assert cleanup["constant_dropped_count"] == len(constants)
    assert cleanup["exact_duplicate_dropped_count"] == len(duplicates)
    assert cleanup["dims_before"] == len(columns) and cleanup["dims_after"] == len(keep)
    assert cleanup["kept_indices"] == prep["kept_indices"] == keep
    assert cleanup["kept_columns"] == prep["columns"] == [columns[j] for j in keep]
    assert not cleanup["labels_used"] and cleanup["high_correlation_columns_dropped"] == 0
    assert prep["scaled_numeric"] == [name for name in prep["columns"] if name in numeric]
    applied = [name for name in prep["columns"] if prep["repair"] == "R2" and name in LOG1P_COUNTS]
    assert prep["log1p_applied_features"] == applied
    assert prep["r2_count_features"] == (LOG1P_COUNTS if prep["repair"] == "R2" else [])
    assert prep["scaler"]["columns"] == prep["scaled_numeric"]
    assert prep["scaler"]["n_samples_seen"] == n
    bound = max(1e-12, 16*n*np.finfo(float).eps)
    # Verify moments columnwise, then retain only projected matrix for diagnostics.
    for k, name in enumerate(prep["scaled_numeric"]):
        v = x[:, columns.index(name)]
        if name in applied:
            v = np.log1p(v)
        mean, var = float(v.mean()), float(v.var())
        eps = np.finfo(float).eps
        scale = 1. if var <= n*eps*var+(n*mean*eps)**2 else float(np.sqrt(var))
        for field, expected in (("mean", mean), ("var", var), ("scale", scale)):
            np.testing.assert_allclose(prep["scaler"][field][k], expected, rtol=bound, atol=1e-12)
    selected = np.ascontiguousarray(x[:, keep])
    del x
    for j, name in enumerate(prep["columns"]):
        if name in applied:
            selected[:, j] = np.log1p(selected[:, j])
        if name in numeric:
            k = prep["scaled_numeric"].index(name)
            selected[:, j] = (selected[:, j]-prep["scaler"]["mean"][k])/prep["scaler"]["scale"][k]
    coefficient = np.asarray(model["coefficient"], dtype=np.float64)
    assert len(coefficient) == selected.shape[1] and model["classes"] == [0, 1]
    labels = np.concatenate([f.target.to_numpy(dtype=np.float64) for f in frames])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=THREAD_LIMIT):
        q = expit(selected @ coefficient+model["intercept"])
        residual = q-labels
        gradient = np.r_[selected.T@residual/n+coefficient/(expected_params["C"]*n), residual.mean()]
        p = selected.shape[1]
        hessian = np.zeros((p+1, p+1))
        weights = q*(1-q)
        # Different chunk size from fitting diagnostics; tolerate summation roundoff.
        for start in range(0, n, 16384):
            block = np.column_stack([selected[start:start+16384], np.ones(min(16384, n-start))])
            hessian += block.T@(block*weights[start:start+16384, None])/n
        hessian[np.arange(p), np.arange(p)] += 1/(expected_params["C"]*n)
        eigenvalues = np.linalg.eigvalsh(hessian)
    diag = audit["diagnostics"]
    np.testing.assert_allclose(diag["gradient_infinity_norm"], np.abs(gradient).max(), rtol=1e-8, atol=1e-12)
    maxima = [float(np.abs(selected[:, prep["columns"].index(name)]).max()) for name in prep["scaled_numeric"]]
    assert diag["max_standardized_absolute_value"] == (max(maxima) if maxima else None)
    assert np.isfinite(diag["local_regularized_hessian_min_eigenvalue"])
    assert np.isfinite(diag["local_regularized_hessian_max_eigenvalue"])
    np.testing.assert_allclose(diag["local_regularized_hessian_min_eigenvalue"], eigenvalues[0], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(diag["local_regularized_hessian_max_eigenvalue"], eigenvalues[-1], rtol=1e-8, atol=1e-12)
    if diag["local_regularized_hessian_min_eigenvalue"] > 0:
        assert diag["local_regularized_hessian_condition_number"] == diag["local_regularized_hessian_max_eigenvalue"]/diag["local_regularized_hessian_min_eigenvalue"]
    assert not audit["negative_sampling"] and not audit["oversampling"]
    assert audit["class_weight"] is None and audit["sample_weight"] is None
    return {"pass": True, "rows": n, "positives": int(labels.sum()), "independent_medians": len(numeric),
            "constant_dropped": len(constants), "duplicates_dropped": len(duplicates), "retained_features": len(keep),
            "gradient_infinity_norm": float(np.abs(gradient).max()), "moment_rtol": bound,
            "negative_finite_count_check": "pass" if prep["repair"] == "R2" else "not_run",
            "no_estimator_fit": True}


def verify(repo):
    """Verify completed new-stage evidence only; never retrain any estimator."""
    repo = Path(repo).resolve()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch != "main":
        raise ValueError(f"P4.2R3 verification requires main, observed {branch!r}")
    started = time.perf_counter()
    report = repo/"reports/phase4"
    checks = [{"number": i+1, "name": name, "status": "not_run"} for i, name in enumerate(INVARIANTS)]
    out = {"stage": "P4.2R3", "status": "fail", "branch": branch, "checks": checks,
        "training": {}, "qc_reuse": {}, "shared_population": {}, "boundary_comparison": {},
        "calibration": {}, "admission": {}, "recommendation_model_fits": 0, "final_week": "not_run"}
    pending = []
    def passed(*numbers):
        for number in numbers:
            checks[number-1]["status"] = "pass"
    passed(2)
    try:
        c = read_json(report/"P4_2R3_EXPERIMENT_CONTRACT.json")
        m = read_json(report/"P4_2R3_metrics.json")
        previous_c = read_json(report/"P4_2R2_EXPERIMENT_CONTRACT.json")
        prior = read_json(Path(c["prior_P4_2R_metrics"]["path"]))
        previous_run = read_json(Path(c["prior_P4_2R2_metrics"]["path"]))
        root = repo/"artifacts/phase4"/c["run_id"]
        prior_root = Path(c["prior_run_root"])
        out["run_id"], out["experiment_status"] = c["run_id"], m["decision"]
        pending = [3]
        assert m["final_week"] == c["final_week"] == "not_run"
        assert not m["Warm_v2_integrated"] and not m["P4_3_started"]
        for cutoff in c["inputs"]:
            guard_cutoff(cutoff)
        for cutoff in ("2020-09-16", "2020-09-17", "2020-09-10"):
            try:
                guard_cutoff(cutoff)
            except ValueError:
                continue
            raise AssertionError("final week or overlapping labels accepted")
        passed(3)
        pending = [1, 25]
        execution = read_json(root/"EXECUTION_START.json")
        assert c["status"] == "preregistered_before_formal_computation"
        assert datetime.fromisoformat(c["created_at_utc"]) < datetime.fromisoformat(execution["started_at_utc"])
        records = [execution["contract"], c["prior_P4_2R_metrics"], c["prior_P4_2R2_metrics"]]
        for key, decision in (("original_P4_2_manifest", "engineering_failure"),
                              ("prior_P4_2R_manifest", "engineering_failure"),
                              ("prior_P4_2R2_manifest", "qW_global_R2_convergence_failure")):
            records.append(c[key])
            manifest = read_json(Path(c[key]["path"]))
            assert manifest["decision"] == decision
            records.extend(record for group in ("reports", "sources", "artifacts") for record in manifest[group])
        closure = {"p42r3_verify.py", "p42r3_report.py"}
        records.extend(r for r in execution["implementation_before_formal"] if Path(r["path"]).name not in closure)
        out["postrun_only_sources_not_claimed_frozen_before_computation"] = [
            r["path"] for r in execution["implementation_before_formal"] if Path(r["path"]).name in closure]
        receipt = read_json(root/"input-verification.json")
        records.extend(receipt["trusted_comparisons"])
        records.extend({**r, "path": r["snapshot_path"]} for r in receipt["authority_snapshots"].values())
        records.extend(identities(m["artifacts"]))
        for attempt in m["attempts"]:
            records.extend([attempt["model"], attempt["preprocessing"]])
        for prepared in c["prepared_reuse"].values():
            records.extend(prepared["features"].values())
        unique = {}
        for record in records:
            path = str(Path(record["path"]).resolve())
            if path in unique:
                assert unique[path]["sha256"] == record["sha256"] and unique[path]["bytes"] == record["bytes"], path
            unique[path] = record
        for record in unique.values():
            check_identity(record)
        out["trusted_SHA_comparisons"] = len(unique)
        passed(1, 25)
        pending = [6, 7, 11, 12, 13, 14, 15, 16, 20]
        assert c["stage"] == "P4.2R3" and c["windows"] == WINDOWS
        for key in ("population", "historical_pools", "excluded_cutoffs", "feature_spec", "epsilon", "utility",
                    "taus", "pairing", "calibration", "metrics_contract", "gates", "selection",
                    "qC_reuse", "prepared_reuse", "model", "software_versions", "budget"):
            assert c[key] == previous_c[key], key
        for key in ("R1", "R2", "all_chains_repair"):
            assert c["qW_repair"][key] == previous_c["qW_repair"][key], key
        assert c["feature_spec"] == FEATURE_SPEC
        assert {k: c["model"][k] for k in MODEL_PARAMS} == MODEL_PARAMS
        expected_params = {**MODEL_PARAMS, "max_iter": 1200}
        assert c["qW_model_params"] == c["qW_repair"]["model_params"] == expected_params
        assert c["taus"] == TAUS and c["epsilon"] == 1e-6
        assert c["qW_repair"]["R2"]["log1p_columns"] == LOG1P_COUNTS
        assert c["qW_repair"]["all_chains_repair"] == m["selected_qW_repair"] == "R2"
        assert c["qC_reuse"]["new_fits"] == m["resources"]["qC_new_fits"] == 0
        assert read_json(report/"P4_2R_VERIFICATION.json")["status"] == "pass"
        assert read_json(report/"P4_2R2_VERIFICATION.json")["status"] == "pass"
        for window in WINDOWS:
            source = prior["training"][window]["qC"]
            assert c["qC_reuse"]["windows"][window] == m["training"][window]["qC"] == source
            reuse = m["qc_reuse"][window]
            assert reuse["status"] == "reused_exact" and not reuse["new_fit"]
            assert reuse["source_stage"] == "P4.2R" and reuse["source_status"] == source["status"] == "converged"
            for key in ("model", "preprocessing", "training_cutoffs"):
                assert reuse[key] == source[key]
            assert read_json(Path(source["model"]["path"]))["params"] == MODEL_PARAMS
            out["qc_reuse"][window] = {"pass": True, "new_fits": 0, "exact_old_model_preprocessing_cohort": True}
        passed(6, 7, 11, 12, 13, 14, 15, 16, 20)
        pending = [4, 5]
        out["shared_population"] = _verify_shared_reuse(m["prepared"], prior["prepared"], prior_root)
        assert c["prepared_reuse"] == prior["prepared"]
        for window, pool in c["historical_pools"].items():
            assert pool["qC"] == pool["qW"]
            for cutoff in pool["qW"]:
                assert cutoff != "2019-11-27"
                assert datetime.fromisoformat(cutoff)+timedelta(days=7) < datetime.fromisoformat(WINDOWS[window])
                lineage = c["inputs"][cutoff]["b0_lineage"]
                assert lineage["available"] and lineage["model_label_end"] <= cutoff
        passed(4, 5)
        pending = [8, 9, 10, 17]
        sequence = verify_solver_sequence(m["attempts"], m["calibration"], m["windows"], m["decision"])
        out["attempt_sequence"] = sequence
        old_warm = {a["window"]: a for a in previous_run["attempts"]}
        for attempt in m["attempts"]:
            window = attempt["window"]
            assert attempt == m["training"][window]["qW"]
            assert attempt["training_cutoffs"] == c["historical_pools"][window]["qW"]
            assert attempt["latest_training_label_end"] < WINDOWS[window]
            assert attempt["train_population"] == "H_t" and attempt["preprocessing_training_only"]
            assert datetime.fromisoformat(attempt["started_at_utc"]) >= datetime.fromisoformat(execution["started_at_utc"])
            model, prep = [read_json(Path(attempt[key]["path"])) for key in ("model", "preprocessing")]
            for key in ("model", "preprocessing"):
                assert Path(attempt[key]["path"]).resolve().is_relative_to(root.resolve())
            assert model["params"] == expected_params and prep["repair"] == "R2"
            solver = attempt["solver_result"]
            assert model["n_iter"] == [solver["nit"]] and solver["max_iter_argument"] == 1200
            assert solver["solver"] == "lbfgs" and solver["returned_parameters_match_estimator"]
            assert not solver["observer_changes_solver_options"] and not solver["observer_changes_result"]
            assert solver["original_check_called_unchanged"]
            np.testing.assert_array_equal(solver["parameter_vector"], np.r_[model["coefficient"], model["intercept"]])
            np.testing.assert_allclose(solver["jac_infinity_norm"], attempt["diagnostics"]["gradient_infinity_norm"], rtol=1e-8, atol=1e-12)
            frames = [parquet(c["prepared_reuse"][t]["features"]["qW"]["path"]) for t in attempt["training_cutoffs"]]
            out["training"][window] = independent_qw_training(frames, prep, attempt, model, expected_params=expected_params)
            previous = old_warm[window]
            assert previous["cleanup"] == attempt["cleanup"]
            assert (previous["training_rows"], previous["positives"]) == (attempt["training_rows"], attempt["positives"])
            old_model, old_prep = [read_json(Path(previous[key]["path"])) for key in ("model", "preprocessing")]
            assert old_model["params"] == MODEL_PARAMS and same_preprocessing(old_prep, prep)
            delta = independent_parameter_delta(frames, old_model, model, old_prep, prep)
            comparison = m["boundary_comparison"][window]
            _close(comparison["parameter_delta"], delta)
            _close(comparison, {"old_stage": "P4.2R2", "new_stage": "P4.2R3",
                "old_iterations": old_model["n_iter"][0], "new_iterations": model["n_iter"][0],
                "additional_iterations_beyond_old_cap": max(0, model["n_iter"][0]-1000),
                "old_diagnostics": previous["diagnostics"], "new_diagnostics": attempt["diagnostics"],
                "solver_result": solver}, exact=True)
            parity = {"preprocessing": same_preprocessing(old_prep, prep),
                      "coefficients": old_model["coefficient"] == model["coefficient"],
                      "intercept": old_model["intercept"] == model["intercept"],
                      "n_iter": old_model["n_iter"] == model["n_iter"]}
            _close(comparison["early_window_exact_parity"], parity, exact=True)
            if window != list(WINDOWS)[-1]:
                assert all(parity.values()), "early-window deterministic parity broken"
            _close(solver["fun"], delta["new_training_log_loss"]+
                   float(np.square(model["coefficient"]).sum())/(2*len(np.concatenate([f.target for f in frames]))))
            out["boundary_comparison"][window] = {"pass": True, "parameter_delta": delta, "parity": parity}
            del frames
            gc.collect()
        if m["attempts"]:
            passed(8, 9, 10)
        if len(m["attempts"]) == 4:
            passed(17)
        pending = [18]
        if sequence["all_four_converged"]:
            gate = read_json(root/"QW_CONVERGENCE_PASSED_BEFORE_CALIBRATION.json")
            assert gate == m["qW_convergence_gate"] and gate["pass"]
            assert gate["checked_before_any_outer_prediction"]
            assert max(a["finished_at_utc"] for a in m["attempts"]) <= gate["checked_at_utc"]
            passed(18)
        else:
            assert not m["calibration"] and not m["windows"]
            assert not (root/"QW_CONVERGENCE_PASSED_BEFORE_CALIBRATION.json").exists()
            assert not list((root/"outer").glob("**/*-predictions.parquet"))
            out["convergence_stop_enforcement"] = "pass; outer not_run because four solver successes not achieved"
        pending = []
        for window, saved in m["calibration"].items():
            folder = root/"outer"/window
            cold, warm = [parquet(folder/f"{side}-predictions.parquet") for side in ("qC", "qW")]
            truth = parquet(folder/"truth.parquet")
            truthsets = {u: set(g.article_id) for u, g in truth.groupby("customer_id", sort=False)}
            h = set(cold.customer_id)
            assert set(warm.customer_id) == set(truthsets)
            out["calibration"][window] = {}
            with duckdb.connect() as con:
                original = con.execute("SELECT customer_id,article_id,warm_rank FROM read_parquet(?) WHERE warm_rank<=12 ORDER BY customer_id,warm_rank", [c["inputs"][WINDOWS[window]]["warm150"]["path"]]).fetchdf()
                pd.testing.assert_frame_equal(warm[list(original)], original, check_dtype=False, check_exact=True)
                original_cold = con.execute("SELECT * FROM read_parquet(?) ORDER BY customer_id,cold_rank,article_id", [c["inputs"][WINDOWS[window]]["cold50"]["path"]]).fetchdf()
                pd.testing.assert_frame_equal(cold[list(original_cold)], original_cold, check_dtype=False, check_exact=True)
            for side, frame in (("qC", cold), ("qW", warm)):
                source = m["training"][window][side]
                model, prep = [read_json(Path(source[key]["path"])) for key in ("model", "preprocessing")]
                q = _predict(frame, model, prep, side)
                np.testing.assert_array_equal(q, frame.propensity)
                np.testing.assert_array_equal(frame.target, [int(i in truthsets[u]) for u, i in zip(frame.customer_id, frame.article_id)])
                mask = np.ones(len(frame), dtype=bool) if side == "qC" else frame.customer_id.isin(h).to_numpy()
                _verify_calibration(frame.target.to_numpy()[mask], q[mask], saved[side])
                assert saved[side]["training_base_rate"] == source["base_rate"]
                if side == "qW":
                    _verify_calibration(frame.target.to_numpy(), q, m["full_calibration"][window])
                out["calibration"][window][side] = {"pass": True, "primary_H_rows": int(mask.sum()), "complete_prediction_rows": len(frame)}
        if set(m["calibration"]) == set(WINDOWS):
            gate = independent_calibration_gate(m["calibration"])
            _close(m["calibration_gate"], gate, exact=True)
            out["calibration_gate"] = gate
            if not gate["pass"]:
                assert m["decision"] == "propensity_calibration_failure" and not m["windows"]
        if m["windows"]:
            pending = [19, 21, 22, 23, 24]
            assert set(m["calibration"]) == set(WINDOWS) and m["calibration_gate"]["pass"]
            assert read_json(root/"CALIBRATION_PASSED_BEFORE_ADMISSION.json") == m["calibration_gate"]
            from .p42_data import load_cutoff
            derived = {}
            for window, metrics in m["windows"].items():
                folder = root/"outer"/window
                data = load_cutoff(c, WINDOWS[window])
                cold, warm = [parquet(folder/f"{side}-predictions.parquet") for side in ("qC", "qW")]
                recommendations = {v: _prediction_list(parquet(folder/f"{v}-top12.parquet"), data["users"]) for v in ("W0", *TAUS)}
                derived[window], out["admission"][window] = verify_admission_bundle(data, cold, warm,
                    parquet(folder/"eligible_cold.parquet"), parquet(folder/"executed.parquet"), parquet(folder/"user_audit.parquet"),
                    np.load(folder/"pair-utility.float64.npy", allow_pickle=False), recommendations, metrics)
                assert derived[window]["variants"]["W0"]["map12"] == c["inputs"][WINDOWS[window]]["w0_map"]
            passed(19, 21, 22, 23, 24)
            if set(derived) == set(WINDOWS):
                _close(m, independent_promotion(derived, c), exact=True)
                out["machine_decision_independently_recomputed"] = True
        if not m["windows"]:
            assert not list((root/"outer").glob("**/*-top12.parquet"))
            assert not list((root/"outer").glob("**/pair-utility.float64.npy"))
        if m["decision"] in FAILURES:
            failure = read_json(root/"FAILURE.json")
            assert failure["decision"] == m["decision"] and m["selected_variant"] == "W0"
            assert failure["failed_at_utc"] >= execution["started_at_utc"]
        else:
            assert set(m["windows"]) == set(WINDOWS)
        out["status"] = "pass"
    except Exception as exc:
        out["status"], out["error"], out["traceback"] = "fail", str(exc), traceback.format_exc()
        for number in pending:
            if checks[number-1]["status"] == "not_run":
                checks[number-1]["status"] = "fail"
        out["failed_check_scope"] = pending
    out["seconds"] = time.perf_counter()-started
    out["summary"] = {name: sum(r["status"] == name for r in checks) for name in ("pass", "not_run", "fail")}
    out["interpretation"] = "Verification validates recorded evidence and enforced boundaries, not hypothetical unrun metrics or automatic promotion."
    write_json(report/"P4_2R3_VERIFICATION.json", out)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    result = verify(parser.parse_args().repo)
    print({"status": result["status"], "summary": result["summary"], "seconds": result["seconds"], "error": result.get("error")})
    if result["status"] != "pass":
        raise SystemExit(2)
