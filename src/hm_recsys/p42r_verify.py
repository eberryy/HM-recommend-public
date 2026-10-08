"""Read-only P4.2R evidence verification, including honest stopped runs.

No recommendation estimator is fitted. qW statistics, cleanup and replay are
implemented independently of p42r_propensity; runtime-not-reached invariants
remain not_run even when the frozen contract specifies their future behavior.
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
from .p42_verify import _assignment, _close, _metrics_from_lists, _replay_matrix, _verify_calibration, _verify_preprocessing
from .p42r_contract import LOG1P_COUNTS, RUN_ID, REPAIR_FAILURES


INVARIANTS = [
    "original_P4_2_report_contract_failure_evidence_preserved", "branch_main_not_Warm",
    "final_week_fail_closed", "frozen_W0_exact_parity", "frozen_B0_M4_parity",
    "no_future_B0", "candidate_generation_before_future_truth_join",
    "no_cold_truth_conditioned_user_eligibility", "shared_H_t_actual_candidate_user_parity",
    "no_safe_B0_cutoff_excluded_from_both_sides", "qC_feature_model_unchanged",
    "qW_R1_constant_exact_duplicate_only", "deterministic_duplicate_survivor",
    "no_high_correlation_only_drop", "R2_only_after_winter_R1_failure",
    "R2_exact_log1p_count_list", "R2_no_negative_finite_counts",
    "solver_C_tol_max_iter_unchanged", "training_only_preprocessing",
    "three_taus_unchanged", "utility_unchanged", "exact_matching_unchanged",
    "no_hard_max1", "exactly12_unique_final_items", "no_edge_exact_W0",
    "all_recorded_SHA_comparisons_pass",
]


def _raw(frame, name):
    return frame[name].to_numpy(dtype=np.float64, na_value=np.nan)


def independent_qw_matrix(frame, prep):
    """Saved-transform replay without calling the production matrix builder."""
    original = prep["numeric"] + prep["binary"] + [x+"_available" for x in prep["numeric"]]
    assert original == prep["input_columns"]
    assert [original[j] for j in prep["kept_indices"]] == prep["columns"]
    result = np.empty((len(frame), len(prep["columns"])), dtype=np.float64)
    for name in prep["numeric"]:
        if prep["repair"] == "R2" and name in LOG1P_COUNTS:
            values = _raw(frame, name)
            assert not (values[np.isfinite(values)] < 0).any(), name
    for j, name in enumerate(prep["columns"]):
        if name in prep["numeric"]:
            values = _raw(frame, name)
            values = np.where(np.isfinite(values), values, prep["median"][name])
            if name in prep["log1p_applied_features"]:
                assert (values >= 0).all(), name
                values = np.log1p(values)
            k = prep["scaled_numeric"].index(name)
            result[:, j] = (values-prep["scaler"]["mean"][k])/prep["scaler"]["scale"][k]
        elif name in prep["binary"]:
            values = _raw(frame, name)
            assert np.isin(values[np.isfinite(values)], [0, 1]).all(), name
            result[:, j] = np.where(np.isfinite(values), values, 0)
        else:
            assert name.endswith("_available") and name[:-10] in prep["numeric"]
            result[:, j] = np.isfinite(_raw(frame, name[:-10]))
    assert np.isfinite(result).all()
    return result


def independent_qw_training(frames, prep, audit, model, spec=FEATURE_SPEC["qW"]):
    """Recompute all train-only medians, exact cleanup, scaler and gradient."""
    numeric, binary = spec["numeric"], spec["binary"]
    columns = numeric + binary + [name+"_available" for name in numeric]
    n = sum(len(frame) for frame in frames)
    assert n > 0 and prep["numeric"] == numeric and prep["binary"] == binary
    assert prep["input_columns"] == columns and prep["repair"] in ("R1", "R2")
    assert n == prep["training_rows"] == audit["training_rows"] == audit["fitted_rows"]
    assert audit["positives"] == sum(int(f.target.sum()) for f in frames)
    assert audit["base_rate"] == audit["positives"]/n
    assert model["params"] == MODEL_PARAMS
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
        gradient = np.r_[selected.T@residual/n+coefficient/(MODEL_PARAMS["C"]*n), residual.mean()]
        p = selected.shape[1]
        hessian = np.zeros((p+1, p+1))
        weights = q*(1-q)
        # Different chunk size from fitting diagnostics; tolerate summation roundoff.
        for start in range(0, n, 16384):
            block = np.column_stack([selected[start:start+16384], np.ones(min(16384, n-start))])
            hessian += block.T@(block*weights[start:start+16384, None])/n
        hessian[np.arange(p), np.arange(p)] += 1/(MODEL_PARAMS["C"]*n)
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


def validate_attempt_sequence(attempts, decision, selected):
    """Check no hidden retry, per-window repair choice, or post-failure fitting."""
    keys = [(a["window"], a["side"], a["repair"]) for a in attempts]
    assert len(keys) == len(set(keys)) and len(keys) <= 9
    winter = next(iter(WINDOWS))
    r2 = [a for a in attempts if a["side"] == "qW" and a["repair"] == "R2"]
    r1_winter = next((a for a in attempts if (a["window"], a["side"], a["repair"]) == (winter, "qW", "R1")), None)
    if r2:
        assert r1_winter is not None and r1_winter["status"] == "non_converged" and r1_winter["convergence_warnings"]
        first_r2 = next(i for i, a in enumerate(attempts) if a["side"] == "qW" and a["repair"] == "R2")
        assert attempts[first_r2]["window"] == winter
        assert keys.index((winter, "qW", "R1")) < first_r2
    # The recorded fits must be a prefix of the frozen sequential plan. Early
    # engineering failure may occur between calls, but it cannot reorder them.
    position = 0
    for window in WINDOWS:
        if position == len(attempts):
            break
        assert keys[position] == (window, "qC", "original")
        cold_attempt = attempts[position]
        position += 1
        if cold_attempt["status"] == "non_converged":
            assert position == len(attempts)
            break
        if position == len(attempts):
            break
        repair = "R1" if window == winter else selected
        assert keys[position] == (window, "qW", repair)
        warm_attempt = attempts[position]
        position += 1
        if window == winter and warm_attempt["status"] == "non_converged" and position < len(attempts):
            assert keys[position] == (window, "qW", "R2")
            warm_attempt = attempts[position]
            position += 1
        if warm_attempt["status"] == "non_converged":
            assert position == len(attempts)
            break
    for a in attempts:
        assert a["side"] in ("qC", "qW") and a["window"] in WINDOWS
        assert a["status"] in ("converged", "non_converged")
        assert (a["status"] == "converged") == (not a["convergence_warnings"])
        if a["side"] == "qC":
            assert a["repair"] == "original"
        elif a["window"] != winter:
            assert a["repair"] == selected
    if decision == "qW_R1_and_R2_convergence_failure":
        assert attempts[-1]["window"] == winter and attempts[-1]["repair"] == "R2"
        assert attempts[-1]["status"] == "non_converged" and selected is None
    if decision == "qC_population_repair_failure" and attempts and attempts[-1]["status"] == "non_converged":
        assert attempts[-1]["side"] == "qC"
    return {"pass": True, "attempts": len(attempts), "R2_attempts": len(r2)}


def _verify_population(repo, root, cutoff, entry, contract):
    """Independent S/last20 mapped-history H, row labels and frozen W0 checks."""
    audit, source = entry["audit"], contract["inputs"][cutoff]
    folder = root/"prepared"/cutoff
    roster = parquet(folder/"shared-population-roster.parquet")
    s = set(roster.customer_id.astype(str))
    h = set(roster.loc[roster.has_mapped_history.astype(bool), "customer_id"].astype(str))
    cold, warm = [parquet(entry["features"][side]["path"]) for side in ("qC", "qW")]
    assert set(cold.customer_id.astype(str)) == set(warm.customer_id.astype(str)) == h
    assert cold.groupby("customer_id").size().eq(50).all()
    assert warm.groupby("customer_id").size().eq(12).all()
    for side, frame in (("qC", cold), ("qW", warm)):
        assert not frame.duplicated(["customer_id", "article_id"]).any()
        assert frame.target_cutoff.eq(cutoff).all()
        assert len(frame) == entry["features"][side]["row_count"]
        assert int(frame.target.sum()) == entry["features"][side]["positive_rows"]
    catalog = pd.read_csv(contract["catalog"]["path"], usecols=["article_id"], dtype={"article_id": str})
    with duckdb.connect() as con:
        con.execute("SET threads=4")
        con.execute("SET memory_limit='4GB'")
        con.register("roster", roster)
        con.register("catalog", catalog)
        transactions = contract["transactions"]["path"]
        observed_s = set(con.execute("SELECT DISTINCT customer_id FROM read_parquet(?) WHERE t_dat>=CAST(? AS DATE) AND t_dat<CAST(? AS DATE)+INTERVAL 7 DAY AND hash(customer_id)%1000000<100000", [transactions, cutoff, cutoff]).fetchnumpy()["customer_id"])
        assert s == observed_s
        observed_h = set(con.execute("""WITH latest AS (
            SELECT t.customer_id,t.article_id,max(t_dat) AS last_day FROM read_parquet(?) t JOIN roster USING(customer_id)
            WHERE t_dat<CAST(? AS DATE) GROUP BY t.customer_id,t.article_id), ranked AS (
            SELECT *, row_number() OVER(PARTITION BY customer_id ORDER BY last_day DESC,article_id) AS rn FROM latest)
            SELECT DISTINCT customer_id FROM ranked JOIN catalog USING(article_id) WHERE rn<=20""", [transactions, cutoff]).fetchnumpy()["customer_id"])
        assert h == observed_h
        frozen_warm = con.execute("SELECT customer_id,article_id,warm_rank FROM read_parquet(?) WHERE warm_rank<=12 ORDER BY customer_id,warm_rank", [source["warm150"]["path"]]).fetchdf()
        assert set(frozen_warm.customer_id) == s
        pd.testing.assert_frame_equal(warm[list(frozen_warm)].reset_index(drop=True), frozen_warm.loc[frozen_warm.customer_id.isin(h)].reset_index(drop=True), check_dtype=False, check_exact=True)
        for frame in (cold, warm):
            con.register("candidate", frame[["customer_id", "article_id", "target"]])
            differences = con.execute("""WITH truth AS (SELECT DISTINCT customer_id,article_id FROM read_parquet(?)
              WHERE t_dat>=CAST(? AS DATE) AND t_dat<CAST(? AS DATE)+INTERVAL 7 DAY)
              SELECT count(*) FROM candidate c LEFT JOIN truth t USING(customer_id,article_id)
              WHERE c.target != CASE WHEN t.article_id IS NULL THEN 0 ELSE 1 END""", [transactions, cutoff, cutoff]).fetchone()[0]
            assert differences == 0
    assert audit["S_t_users"] == len(s) and audit["H_t_users"] == len(h)
    assert audit["S_t_without_mapped_history"] == len(s-h)
    assert audit["exact_user_set_parity"] and audit["no_cold_truth_conditioned_eligibility"]
    assert source["b0_lineage"]["available"] and source["b0_lineage"]["model_label_end"] <= cutoff
    fence = read_json(folder/"SELECTION_BEFORE_LABEL_JOIN.json")
    assert not fence["future_item_labels_read"]
    events = {row["event"]: row for row in audit["timeline"]}
    assert datetime.fromisoformat(events["complete_Cold50_persisted_before_label_join"]["at_utc"]) <= datetime.fromisoformat(events["future_truth_join_completed"]["started_at_utc"])
    selection = parquet(folder/"cold50-selection-before-labels.parquet")
    assert "target" not in selection and set(selection.customer_id) == h
    assert list(zip(selection.customer_id, selection.article_id)) == list(zip(cold.customer_id, cold.article_id))
    old = parquet(source["cold50"]["path"])
    reused = old.loc[old.customer_id.isin(h)].sort_values(["customer_id", "cold_rank", "article_id"], ignore_index=True)
    current = selection.loc[selection.customer_id.isin(set(reused.customer_id))].reset_index(drop=True)
    pd.testing.assert_frame_equal(current[list(old)], reused, check_dtype=False, check_exact=True)
    generated = folder/"new-m4-top200-unlabelled.npz"
    if generated.exists():
        with np.load(generated, allow_pickle=False) as arrays:
            assert "target" not in arrays.files
            assert (np.bincount(arrays["user_index"]) == 200).all()
            generation_users = parquet(folder/"generation-users.parquet")
            probe_users = set(generation_users.loc[generation_users.is_probe.astype(bool), "customer_id"])
            old_users = pd.read_csv(source["p31_assets"]["users"]["path"], dtype={"customer_id": str}).customer_id.to_numpy()
            observed_users = generation_users.customer_id.to_numpy()[arrays["user_index"]]
            mask = np.isin(observed_users, list(probe_users))
            observed = pd.DataFrame({"customer_id": observed_users[mask], "catalog_row": arrays["catalog_row"][mask],
                "rank": arrays["rank"][mask], "coarse_score": arrays["coarse_score"][mask],
                "b0_rank": arrays["b0_rank"][mask], "b0_score": arrays["b0_score"][mask]})
            with np.load(source["p31_assets"]["candidates"]["path"], allow_pickle=False) as prior:
                expected_users = old_users[prior["user_index"]]
                keep = np.isin(expected_users, list(probe_users))
                expected = pd.DataFrame({"customer_id": expected_users[keep], "catalog_row": prior["catalog_row"][keep],
                    "rank": prior["rank"][keep], "coarse_score": prior["coarse_score"][keep],
                    "b0_rank": np.load(source["b0_lineage"]["rank_artifact"]["path"], allow_pickle=False)[keep],
                    "b0_score": np.load(source["b0_lineage"]["score_artifact"]["path"], allow_pickle=False)[keep]})
            observed = observed.sort_values(["customer_id", "rank"], ignore_index=True)
            expected = expected.sort_values(["customer_id", "rank"], ignore_index=True)
            assert len(expected) == len(probe_users)*200
            for name in ("customer_id", "catalog_row", "rank", "b0_rank"):
                np.testing.assert_array_equal(observed[name], expected[name])
            for name in ("coarse_score", "b0_score"):
                np.testing.assert_allclose(observed[name], expected[name], rtol=0, atol=1e-5)
        assert audit["reconstruction_probe"]["m4_top200_identity_rank_exact"]
        assert audit["reconstruction_probe"]["b0_all200_rank_exact"]
    return {"pass": True, "S_t_users": len(s), "H_t_users": len(h), "qC_rows": len(cold), "qW_rows": len(warm),
            "roster_and_past_history_recomputed": True, "future_labels_recomputed_after_frozen_selection": True,
            "frozen_reused_Cold50_rows_compared": len(reused), "new_candidate_probe": audit["reconstruction_probe"]}


def _predict(frame, model, prep, side):
    x = _replay_matrix(frame, prep) if side == "qC" else independent_qw_matrix(frame, prep)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=THREAD_LIMIT):
        return expit(x@np.asarray(model["coefficient"])+model["intercept"])


def verify(repo):
    """Write one verification receipt; a truthful failed experiment may pass."""
    started = time.perf_counter()
    repo = Path(repo).resolve()
    report = repo/"reports/phase4"
    root = repo/"artifacts/phase4"/RUN_ID
    checks = [{"number": i+1, "name": name, "status": "not_run"} for i, name in enumerate(INVARIANTS)]
    out = {"stage": "P4.2R", "run_id": RUN_ID, "status": "fail", "checks": checks,
           "models": {}, "populations": {}, "calibration": {}, "admission": {},
           "recommendation_model_fits": 0, "final_week": "not_run"}
    def passed(*numbers):
        for number in numbers:
            checks[number-1]["status"] = "pass"
    try:
        c = read_json(report/"P4_2R_EXPERIMENT_CONTRACT.json")
        m = read_json(report/"P4_2R_metrics.json")
        out["experiment_status"] = m["decision"]
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
        assert branch == "main"
        passed(2)
        assert m["final_week"] == "not_run" and not m["Warm_v2_integrated"] and not m["P4_3_started"]
        for bad in ("2020-09-16", "2020-09-17", "2020-09-10"):
            try:
                guard_cutoff(bad)
            except ValueError:
                continue
            raise AssertionError("final-week or overlapping interval accepted")
        passed(3)
        old = read_json(report/"P4_2_EXPERIMENT_CONTRACT.json")
        records = []
        original_manifest = read_json(Path(c["original_P4_2_manifest"]["path"]))
        records.append(c["original_P4_2_manifest"])
        records.extend(r for group in ("reports", "artifacts", "sources") for r in original_manifest[group])
        execution = read_json(root/"EXECUTION_START.json")
        records.append(execution["contract"])
        postrun_only = {"p42r_verify.py", "p42r_report.py", "p42r_close.py"}
        records.extend(r for r in execution["implementation_before_formal"] if Path(r["path"]).name not in postrun_only)
        out["postrun_reporting_verification_sources_not_claimed_frozen_before_computation"] = [r["path"] for r in execution["implementation_before_formal"] if Path(r["path"]).name in postrun_only]
        assert datetime.fromisoformat(c["created_at_utc"]) < datetime.fromisoformat(execution["started_at_utc"])
        receipt = read_json(root/"input-verification.json")
        records.extend(receipt["trusted_comparisons"])
        for record in receipt["authority_snapshots"].values():
            records.append({**record, "path": record["snapshot_path"]})
        for attempt in m["attempts"]:
            records.extend([attempt["model"], attempt["preprocessing"]])
        for entry in m["prepared"].values():
            records.extend(entry["features"].values())
        unique = {(str(Path(r["path"]).resolve()), r["sha256"]): r for r in records if "sha256" in r}
        for record in unique.values():
            check_identity(record)
        out["SHA_comparisons"] = len(unique)
        passed(1, 26)
        for key in ("feature_spec", "epsilon", "taus", "utility", "pairing", "gates", "selection", "metrics_contract"):
            assert c[key] == old[key], key
        assert c["feature_spec"] == FEATURE_SPEC and c["taus"] == TAUS
        assert c["qW_repair"]["R2"]["log1p_columns"] == LOG1P_COUNTS
        assert {k: c["model"][k] for k in MODEL_PARAMS} == c["qW_repair"]["model_params"] == MODEL_PARAMS
        assert not c["population"]["future_cold_truth_conditioned_eligibility"]
        passed(11, 16, 18, 20)
        for window, pools in c["historical_pools"].items():
            assert pools["qC"] == pools["qW"]
            for cutoff in pools["qC"]:
                guard_cutoff(cutoff)
                assert cutoff != "2019-11-27" and c["inputs"][cutoff]["b0_lineage"]["available"]
                assert datetime.fromisoformat(cutoff)+timedelta(days=7) < datetime.fromisoformat(WINDOWS[window])
        passed(10)
        for cutoff, entry in m["prepared"].items():
            out["populations"][cutoff] = _verify_population(repo, root, cutoff, entry, c)
            gc.collect()
        if m["prepared"]:
            passed(4, 5, 6, 7, 8, 9)
        out["attempt_sequence"] = validate_attempt_sequence(m["attempts"], m["decision"], m["selected_qW_repair"])
        if m["attempts"]:
            passed(15)
        for attempt in m["attempts"]:
            side, window = attempt["side"], attempt["window"]
            assert attempt["training_cutoffs"] == c["historical_pools"][window][side]
            assert attempt["latest_training_label_end"] < WINDOWS[window]
            assert attempt["preprocessing_training_only"] and attempt["train_population"] == "H_t"
            model, prep = [read_json(Path(attempt[key]["path"])) for key in ("model", "preprocessing")]
            assert model["params"] == MODEL_PARAMS
            assert len(model["n_iter"]) == 1 and 0 <= model["n_iter"][0] <= MODEL_PARAMS["max_iter"]
            frames = [parquet(m["prepared"][t]["features"][side]["path"]) for t in attempt["training_cutoffs"]]
            if side == "qC":
                checked = _verify_preprocessing(frames, prep, attempt, FEATURE_SPEC[side], require_convergence=False)
            else:
                checked = independent_qw_training(frames, prep, attempt, model)
                passed(12, 13, 14)
                if attempt["repair"] == "R2":
                    passed(17)
            out["models"][f"{window}/{side}/{attempt['repair']}"] = checked
            del frames
            gc.collect()
        if m["attempts"]:
            passed(19)
        for window, saved in m["calibration"].items():
            out["calibration"][window] = {}
            cold = parquet(root/"outer"/window/"qC-predictions.parquet")
            h = set(cold.customer_id)
            for side in ("qC", "qW"):
                frame = cold if side == "qC" else parquet(root/"outer"/window/"qW-predictions.parquet")
                trained = m["training"][window][side]
                model, prep = [read_json(Path(trained[key]["path"])) for key in ("model", "preprocessing")]
                q = _predict(frame, model, prep, side)
                np.testing.assert_array_equal(q, frame.propensity)
                mask = np.ones(len(frame), bool) if side == "qC" else frame.customer_id.isin(h).to_numpy()
                _verify_calibration(frame.target.to_numpy()[mask], q[mask], saved[side])
                out["calibration"][window][side] = {"pass": True, "full_prediction_rows": len(frame), "primary_H_rows": int(mask.sum())}
        if set(m["calibration"]) == set(WINDOWS):
            counts = {}
            for side in ("qC", "qW"):
                rows = [m["calibration"][w][side] for w in WINDOWS]
                counts[side] = {
                    "extreme_rate_ratio_windows": sum(r["predicted_to_observed_rate_ratio"] is not None and not .1 <= r["predicted_to_observed_rate_ratio"] <= 10 for r in rows),
                    "roc_not_above_chance_windows": sum(r["roc_auc"] is not None and r["roc_auc"] <= .5 for r in rows),
                    "constant_prediction_windows": sum("constant_prediction" in r.get("warnings", []) for r in rows),
                }
            assert m["calibration_gate"]["counts"] == counts
            assert m["calibration_gate"]["pass"] == (not any(v >= 3 for side in counts.values() for v in side.values()))
            out["calibration_gate_independently_recomputed"] = True
        if m["windows"]:
            assert set(m["calibration"]) == set(WINDOWS) and m["calibration_gate"]["pass"]
            assert (root/"CALIBRATION_PASSED_BEFORE_ADMISSION.json").exists()
            for window in m["windows"]:
                out["admission"][window] = _verify_admission(root, c, m, window)
            passed(21, 22, 23, 24, 25)
        elif m["decision"] == "propensity_calibration_failure":
            assert set(m["calibration"]) == set(WINDOWS) and not m["calibration_gate"]["pass"]
            assert not (root/"CALIBRATION_PASSED_BEFORE_ADMISSION.json").exists()
        if m["decision"] in REPAIR_FAILURES:
            assert (root/"FAILURE.json").exists()
            failure = read_json(root/"FAILURE.json")
            assert failure["decision"] == m["decision"] and m["selected_variant"] == "W0"
            assert datetime.fromisoformat(failure["failed_at_utc"]) >= datetime.fromisoformat(execution["started_at_utc"])
        bootstrap = root/"bootstrap-attempt-01"
        if bootstrap.exists():
            out["bootstrap_startup_attempt"] = {"path": str(bootstrap), "scope": "startup engineering failure before real fits; distinct from formal numerical repair attempts"}
        if not m["windows"]:
            assert not list((root/"outer").glob("**/*-top12.parquet"))
            assert not list((root/"outer").glob("**/pair-utility.float64.npy"))
        if m["decision"] in ("qC_population_repair_failure", "qW_R1_and_R2_convergence_failure"):
            assert not m["calibration"] and not m["windows"]
        out["status"] = "pass"
    except Exception as exc:
        out["error"] = str(exc)
        out["traceback"] = traceback.format_exc()
        out["status"] = "fail"
    out["seconds"] = time.perf_counter()-started
    out["summary"] = {status: sum(c["status"] == status for c in checks) for status in ("pass", "not_run", "fail")}
    out["interpretation"] = "Verification pass validates recorded evidence and stop boundaries; it does not turn numerical or calibration failure into experiment success."
    write_json(report/"P4_2R_VERIFICATION.json", out)
    return out


def _verify_admission(root, contract, metrics, window):
    """Independent exact assignment, list replay and full-denominator MAP."""
    from .p42_data import load_cutoff
    data = load_cutoff(contract, WINDOWS[window])
    folder = root/"outer"/window
    cold, warm = [parquet(folder/f"{side}-predictions.parquet") for side in ("qC", "qW")]
    users, original = data["users"], data["warm_lists"]
    np.testing.assert_array_equal(warm.article_id.to_numpy().reshape(-1, 12), original)
    np.testing.assert_array_equal(cold.article_id, data["cold"].article_id)
    eligible = parquet(folder/"eligible_cold.parquet")
    q_c = np.clip(cold.propensity.to_numpy(), 1e-6, 1-1e-6)
    q_w = np.clip(warm.propensity.to_numpy().reshape(-1, 12), 1e-6, 1-1e-6)
    indices = eligible.cold_row_index.to_numpy(dtype=np.int64)
    ui = eligible.user_index.to_numpy(dtype=np.int64)
    utility = np.log(q_c[indices]/(1-q_c[indices]))[:, None]-np.log(q_w/(1-q_w))[ui]
    np.testing.assert_array_equal(utility, np.load(folder/"pair-utility.float64.npy", allow_pickle=False))
    reconstructed = {variant: original.copy() for variant in TAUS}
    pairs = []
    for user_idx, rows in eligible.groupby("user_index", sort=False):
        block = utility[rows.index.to_numpy()]
        assert len(rows) <= 50 and block.shape[1] == 12
        for variant, tau in TAUS.items():
            matches = _assignment(block, tau)
            assert len({a for a, b in matches}) == len(matches) == len({b for a, b in matches})
            for ci, wi in matches:
                item = rows.iloc[ci].article_id
                reconstructed[variant][int(user_idx), wi] = item
                pairs.append((variant, int(user_idx), wi+1, item))
    executed = parquet(folder/"executed.parquet")
    observed = [(r.variant, int(r.user_index), int(r.warm_slot_rank), r.cold_article_id) for r in executed.itertuples()]
    assert sorted(pairs) == sorted(observed)
    for variant in ("W0", *TAUS):
        frame = parquet(folder/f"{variant}-top12.parquet")
        np.testing.assert_array_equal(frame.customer_id, np.repeat(users, 12))
        np.testing.assert_array_equal(frame["rank"], np.tile(np.arange(1, 13), len(users)))
        assert not frame.duplicated(["customer_id", "article_id"]).any()
        top = frame.article_id.to_numpy().reshape(-1, 12)
        np.testing.assert_array_equal(top, original if variant == "W0" else reconstructed[variant])
        measured, _, _ = _metrics_from_lists(users, top, data["truth"], original)
        _close(metrics["windows"][window]["variants"][variant], measured, exact=True)
        if variant == "W0":
            assert measured["map12"] == contract["inputs"][WINDOWS[window]]["w0_map"]
        # Includes all S\H users, not only H users with below-threshold edges.
        active = np.isin(np.arange(len(users)), ui)
        np.testing.assert_array_equal(top[~active], original[~active])
    return {"pass": True, "users": len(users), "utility_values": int(utility.size),
            "executed_replacements": len(pairs), "variants": 4, "full_denominator_MAP_recomputed": True}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    result = verify(args.repo)
    print({"status": result["status"], "summary": result["summary"], "seconds": result["seconds"], "error": result.get("error")})
    if result["status"] != "pass":
        raise SystemExit(2)
