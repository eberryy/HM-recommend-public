"""Independent P4.2 evidence replay: no recommendation-model training API.

The verifier reconstructs preprocessing statistics without fitting a scaler,
replays JSON coefficients directly, independently solves every saved assignment,
and uses the original AP implementation on full truth-user denominators.
"""
from __future__ import annotations

import ast
import copy
import gc
import importlib.metadata
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from .metrics import apk
from .p41a_contract import check_identity, read_json, write_json
from .p41a_data import identities, parquet
from .p42_contract import FEATURE_SPEC, RUN_ID, TAUS, WINDOWS, branch_guard, guard_cutoff
from .p42_propensity import MODEL_PARAMS, EPSILON, THREAD_LIMIT, _calibration_line


def _close(actual, expected, *, exact=False):
    if expected is None:
        assert actual is None, (actual, expected)
    elif isinstance(expected, (float, np.floating)):
        assert actual is not None
        np.testing.assert_allclose(actual, expected, rtol=0 if exact else 1e-12,
                                   atol=0 if exact else 1e-14, equal_nan=False)
    elif isinstance(expected, dict):
        for key, value in expected.items():
            assert key in actual, key
            _close(actual[key], value, exact=exact)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _close(a, b, exact=exact)
    else:
        assert actual == expected, (actual, expected)


def _replay_matrix(frame, prep):
    """Deliberately not the preprocessing implementation under test."""
    numeric, binary = prep["numeric"], prep["binary"]
    p, b = len(numeric), len(binary)
    assert prep["columns"] == numeric + binary + [c+"_available" for c in numeric]
    x = np.empty((len(frame), 2*p+b), dtype=np.float64)
    for j, column in enumerate(numeric):
        raw = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
        valid = np.isfinite(raw)
        x[:, j] = np.where(valid, raw, prep["median"][column])
        x[:, j] -= prep["scaler"]["mean"][j]
        x[:, j] /= prep["scaler"]["scale"][j]
        x[:, p+b+j] = valid
    for j, column in enumerate(binary):
        raw = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
        valid = np.isfinite(raw)
        assert np.isin(raw[valid], [0, 1]).all(), column
        x[:, p+j] = np.where(valid, raw, 0.0)
    assert np.isfinite(x).all()
    return x


def _replay(frame, model, prep):
    x = _replay_matrix(frame, prep)
    assert model["classes"] == [0, 1]
    assert len(model["coefficient"]) == x.shape[1]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=THREAD_LIMIT):
        return expit(x @ np.asarray(model["coefficient"]) + model["intercept"])


def _verify_preprocessing(frames, prep, audit, spec, *, require_convergence=True):
    """Independent column-wise train statistics; never instantiate an estimator."""
    assert prep["numeric"] == spec["numeric"] and prep["binary"] == spec["binary"]
    assert prep["columns"] == spec["numeric"] + spec["binary"] + [c+"_available" for c in spec["numeric"]]
    n = sum(len(frame) for frame in frames)
    positives = sum(int(frame.target.sum()) for frame in frames)
    assert n == prep["training_rows"] == prep["scaler"]["n_samples_seen"]
    assert n == audit["training_rows"] == audit["fitted_rows"]
    assert positives == audit["positives"]
    _close(audit["base_rate"], positives/n, exact=True)
    all_missing = []
    # sklearn reduces a strided multi-column array whereas this independent
    # verifier reduces one contiguous column. Float64 accumulation error grows
    # with row count; medians/counts remain exact, moments use an explicit bound.
    moment_rtol = max(1e-12, 16*n*np.finfo(np.float64).eps)
    maximum_moment_scaled_discrepancy = 0.0
    for j, column in enumerate(spec["numeric"]):
        values = np.concatenate([f[column].to_numpy(dtype=np.float64, na_value=np.nan) for f in frames])
        finite = np.isfinite(values)
        median = float(np.median(values[finite])) if finite.any() else 0.0
        assert prep["median"][column] == median, column
        assert prep["missing_training_rows"][column] == int((~finite).sum())
        if not finite.any():
            all_missing.append(column)
        values[~finite] = median
        mean, var = float(np.mean(values)), float(np.var(values))
        np.testing.assert_allclose(prep["scaler"]["mean"][j], mean, rtol=moment_rtol, atol=1e-12)
        np.testing.assert_allclose(prep["scaler"]["var"][j], var, rtol=moment_rtol, atol=1e-12)
        maximum_moment_scaled_discrepancy = max(maximum_moment_scaled_discrepancy,
            abs(prep["scaler"]["mean"][j]-mean)/max(1, abs(mean)),
            abs(prep["scaler"]["var"][j]-var)/max(1, abs(var)))
        # StandardScaler treats numerical constants using its floating-error
        # upper bound, not only exact variance==0; this is a numeric safeguard.
        eps = np.finfo(np.float64).eps
        constant = var <= n*eps*var + (n*mean*eps)**2
        scale = 1.0 if constant else float(np.sqrt(var))
        np.testing.assert_allclose(prep["scaler"]["scale"][j], scale, rtol=moment_rtol, atol=1e-12)
    assert prep["all_missing_numeric"] == all_missing
    for column in spec["binary"]:
        for frame in frames:
            values = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
            assert np.isin(values[np.isfinite(values)], [0, 1]).all(), column
    assert len(prep["scaler"]["mean"]) == len(spec["numeric"])
    assert not audit["negative_sampling"] and not audit["oversampling"]
    assert audit["class_weight"] is None and audit["sample_weight"] is None
    if require_convergence:
        assert audit["status"] == "converged" and not audit["convergence_warnings"]
    return {"rows": n, "positives": positives, "numeric_medians_means_variances_checked": len(spec["numeric"]),
            "binary_unscaled_columns": len(spec["binary"]), "generated_unscaled_availability_columns": len(spec["numeric"]),
            "independent_float64_moment_rtol": moment_rtol,
            "maximum_mean_variance_scaled_discrepancy": maximum_moment_scaled_discrepancy}


def diagnose_stopped_model(repo, window="winter_20200122", side="qW"):
    """Read-only fixed-parameter diagnostics; no optimizer, no model fit."""
    from threadpoolctl import threadpool_limits
    import sklearn.linear_model._logistic as sklearn_logistic
    import sklearn.linear_model._linear_loss as sklearn_loss

    started = time.perf_counter()
    repo = Path(repo).resolve()
    root = repo / "artifacts/phase4" / RUN_ID
    contract = read_json(repo / "reports/phase4/P4_2_EXPERIMENT_CONTRACT.json")
    model = read_json(root / "models" / window / f"{side}-model.json")
    prep = read_json(root / "models" / window / f"{side}-preprocessing.json")
    audit = read_json(root / "models" / window / f"{side}-training-audit.json")
    frames = [parquet(root / "prepared" / cutoff / f"{side}-features.parquet")
              for cutoff in contract["historical_pools"][window][side]]
    statistics = _verify_preprocessing(frames, prep, audit, contract["feature_spec"][side], require_convergence=False)
    data = pd.concat(frames, ignore_index=True)
    del frames
    x = _replay_matrix(data, prep)
    y = data.target.to_numpy(dtype=np.float64)
    coef = np.asarray(model["coefficient"], dtype=np.float64)
    n, p = x.shape
    with threadpool_limits(limits=THREAD_LIMIT):
        logits = x @ coef + model["intercept"]
        q = expit(logits)
        strength = 1.0/(model["params"]["C"]*n)
        residual = q-y
        gradient = np.concatenate([x.T @ residual/n + strength*coef, [residual.mean()]])
        loss = float(np.mean(np.logaddexp(0, logits)-y*logits) + .5*strength*np.dot(coef, coef))
        means, variances = x.mean(axis=0), x.var(axis=0)
        lower, upper = x.min(axis=0), x.max(axis=0)
        constant = lower == upper
        active = np.flatnonzero(~constant)
        centered_gram = x[:, active].T @ x[:, active]/n - np.outer(means[active], means[active])
        corr = centered_gram / np.sqrt(np.outer(variances[active], variances[active]))
        corr = np.clip(corr, -1, 1)
        pairs = []
        for i in range(len(active)):
            for j in range(i+1, len(active)):
                value = float(corr[i, j])
                if abs(value) >= .995:
                    a, b = int(active[i]), int(active[j])
                    pairs.append({"left": prep["columns"][a], "right": prep["columns"][b],
                                  "pearson": value, "exact_duplicate": bool(np.array_equal(x[:, a], x[:, b]))})
        pairs.sort(key=lambda r: (-abs(r["pearson"]), r["left"], r["right"]))
        # Local curvature at the frozen parameters, not a proposed update.
        weights = q*(1-q)
        hessian = np.empty((p+1, p+1), dtype=np.float64)
        hessian[:p, :p] = x.T @ (x*weights[:, None])/n
        hessian[np.arange(p), np.arange(p)] += strength
        hessian[:p, p] = hessian[p, :p] = x.T @ weights/n
        hessian[p, p] = weights.mean()
        eigenvalues = np.linalg.eigvalsh(hessian)
    numeric_n = len(prep["numeric"])
    numeric_absmax = np.maximum(np.abs(lower[:numeric_n]), np.abs(upper[:numeric_n]))
    outlier_order = np.argsort(-numeric_absmax, kind="stable")[:15]
    all_names = prep["columns"] + ["intercept"]
    gradient_order = np.argsort(-np.abs(gradient), kind="stable")[:15]
    constants = [{"feature": prep["columns"][j], "value": float(lower[j])} for j in np.flatnonzero(constant)]
    return {"stage": "P4.2", "kind": "read_only_stopped_parameter_diagnostic", "window": window, "side": side,
        "recommendation_model_fits": 0, "optimizer_calls": 0, "outer_rows_read": 0,
        "training_only_preprocessing": statistics, "status": audit["status"], "n_iter": model["n_iter"],
        "warnings": audit["convergence_warnings"], "rows": n, "features": p,
        "observed_training_positive_rate": float(y.mean()), "mean_fixed_training_prediction": float(q.mean()),
        "fixed_objective": loss, "objective_definition": "mean logistic loss + ||beta||^2/(2*C*n); intercept unpenalized",
        "l2_reg_strength": strength, "gradient_infinity_norm": float(np.abs(gradient).max()),
        "frozen_gtol": model["params"]["tol"], "lbfgs_ftol": float(64*np.finfo(float).eps),
        "gradient_top15": [{"feature": all_names[j], "gradient": float(gradient[j]), "absolute": float(abs(gradient[j]))} for j in gradient_order],
        "constant_columns": constants, "constant_column_count": len(constants),
        "nonconstant_exact_duplicate_pairs": [r for r in pairs if r["exact_duplicate"]],
        "near_collinearity_threshold_abs_pearson": .995, "near_collinear_pair_count": len(pairs), "near_collinear_top30": pairs[:30],
        "standardized_numeric_abs_max_top15": [{"feature": prep["numeric"][j], "abs_max": float(numeric_absmax[j]),
            "rows_abs_over10": int((np.abs(x[:, j]) > 10).sum()), "rows_abs_over100": int((np.abs(x[:, j]) > 100).sum())} for j in outlier_order],
        "local_regularized_hessian_min_eigenvalue": float(eigenvalues[0]), "local_regularized_hessian_max_eigenvalue": float(eigenvalues[-1]),
        "local_regularized_hessian_condition_number": float(eigenvalues[-1]/eigenvalues[0]) if eigenvalues[0] > 0 else None,
        "installed_source_files_inspected": [sklearn_logistic.__file__, sklearn_loss.__file__],
        "interpretation_limit": "gradient and local conditioning explain numerical difficulty candidates, not a proven unique root cause; no feature/solver/tolerance/iteration change or retry performed",
        "seconds": time.perf_counter()-started, "final_week": "not_run"}


def _verify_calibration(labels, probabilities, saved):
    y = np.asarray(labels, dtype=np.int8)
    raw = np.asarray(probabilities, dtype=np.float64)
    assert np.isfinite(raw).all() and np.all((raw >= 0) & (raw <= 1))
    q = np.clip(raw, 1e-6, 1-1e-6)
    assert saved["epsilon"] == 1e-6
    n, positives = len(y), int(y.sum())
    _close(saved, {"rows": n, "positives": positives,
        "observed_positive_rate": float(y.mean()), "mean_predicted_probability": float(q.mean()),
        "unclipped_mean_predicted_probability": float(raw.mean()),
        "brier_score": float(np.mean(np.square(q-y))),
        "roc_auc": float(roc_auc_score(y, q)) if 0 < positives < n else None,
        "pr_auc": float(average_precision_score(y, q)) if 0 < positives < n else None,
        "predicted_to_observed_rate_ratio": float(q.mean()/y.mean()) if positives else None,
        "clipped_low_rows": int((raw < 1e-6).sum()), "clipped_high_rows": int((raw > 1-1e-6).sum())})
    assert len(saved["calibration_bins"]) == 10
    ece = 0.0
    for number, indices in enumerate(np.array_split(np.argsort(q, kind="stable"), 10), 1):
        count = len(indices)
        row = {"bin": number, "rows": count, "positives": int(y[indices].sum()),
               "mean_probability": float(q[indices].mean()) if count else None,
               "observed_positive_rate": float(y[indices].mean()) if count else None,
               "min_probability": float(q[indices].min()) if count else None,
               "max_probability": float(q[indices].max()) if count else None}
        _close(saved["calibration_bins"][number-1], row)
        if count:
            ece += count * abs(row["mean_probability"]-row["observed_positive_rate"])
    _close(saved["ece"], ece/n)
    assert saved["calibration_line"]["diagnostic_only"]
    _close(saved["calibration_intercept"], saved["calibration_line"]["intercept"])
    _close(saved["calibration_slope"], saved["calibration_line"]["slope"])
    # This is only the explicitly allowed evaluation-only two-parameter
    # calibration diagnostic, never a recommendation-model or scaler fit.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=THREAD_LIMIT):
        reference_line = _calibration_line(y, np.log(q)-np.log1p(-q))
    _close(saved["calibration_line"], reference_line)
    assert ("constant_prediction" in saved["warnings"]) == (float(np.ptp(q)) <= 1e-12)
    assert ("roc_auc_not_above_chance" in saved["warnings"]) == (saved["roc_auc"] is not None and saved["roc_auc"] <= .5)
    ratio = saved["predicted_to_observed_rate_ratio"]
    assert ("mean_probability_to_observed_rate_outside_0.25_4" in saved["warnings"]) == (ratio is not None and (ratio < .25 or ratio > 4))
    return {"rows": n, "bins": 10, "full_pool_base_rate_discrimination_brier_ece_recomputed": True,
            "no_outer_calibrator_in_prediction_replay": True}


def _assignment(matrix, tau):
    """Independent canonical assignment construction, not exact_matching()."""
    k, slots = matrix.shape
    assert k <= 50 and slots == 12 and matrix.size <= 600
    assert tau in (0.0, 0.6931471805599453, 1.3862943611198906)
    costs = np.zeros((slots, k+slots), dtype=np.float64)
    for slot in range(slots):
        for cold in range(k):
            costs[slot, cold] = tau-matrix[cold, slot] if matrix[cold, slot] > tau else np.inf
    rows, columns = linear_sum_assignment(costs)
    return [(int(column), int(slot)) for slot, column in zip(rows, columns) if column < k]


def _prediction_list(frame, users):
    assert len(frame) == len(users)*12
    assert not frame.duplicated(["customer_id", "article_id"]).any()
    np.testing.assert_array_equal(frame.customer_id, np.repeat(users, 12))
    np.testing.assert_array_equal(frame["rank"], np.tile(np.arange(1, 13), len(users)))
    return frame.article_id.to_numpy().reshape(-1, 12)


def _metrics_from_lists(users, lists, truth, baseline_lists):
    """Reference apk on the entire appropriate truth-user population."""
    truthsets = {u: set(g.article_id) for u, g in truth.groupby("customer_id", sort=False)}
    assert set(users) == set(truthsets)
    peruser = np.array([apk(list(truthsets[u]), list(row)) for u, row in zip(users, lists)])
    base = np.array([apk(list(truthsets[u]), list(row)) for u, row in zip(users, baseline_lists)])
    counts = truth.interaction_count_before_cutoff.to_numpy()
    selectors = {"warm_21_plus": counts >= 21, "strict_cold": counts == 0,
                 "sparse1_5": (counts >= 1) & (counts <= 5), "all_cold_sparse": counts <= 5}
    segments = {}
    for name, mask in selectors.items():
        part = truth.loc[mask]
        truths = {u: set(g.article_id) for u, g in part.groupby("customer_id", sort=False)}
        indices = [i for i, u in enumerate(users) if u in truths]
        actual = [apk(list(truths[users[i]]), list(lists[i])) for i in indices]
        reference = [apk(list(truths[users[i]]), list(baseline_lists[i])) for i in indices]
        mean = float(np.mean(actual)) if actual else None
        base_mean = float(np.mean(reference)) if reference else None
        segments[name] = {"map12": mean, "delta_vs_w0": mean-base_mean if mean is not None else None,
                          "truth_users": len(indices), "truth_pairs": len(part)}
    return {"map12": float(peruser.mean()), "delta_vs_w0": float(peruser.mean()-base.mean()),
            "segments": segments}, peruser, base


def _verify_window(repo, root, contract, metrics, training, calibration, window):
    cutoff = contract["windows"][window]
    assets = contract["inputs"][cutoff]
    folder = root / "outer" / window
    cold, warm = parquet(folder / "qC-predictions.parquet"), parquet(folder / "qW-predictions.parquet")
    truth, eligible = parquet(folder / "truth.parquet"), parquet(folder / "eligible_cold.parquet")
    executed, user_audit = parquet(folder / "executed.parquet"), parquet(folder / "user_audit.parquet")
    users = warm.customer_id.drop_duplicates().to_numpy()
    original = warm.article_id.to_numpy().reshape(-1, 12)
    assert len(users) == assets["w0_users"] == metrics["users"]
    with duckdb.connect() as con:
        oldwarm = con.execute("SELECT customer_id,article_id,warm_rank FROM read_parquet(?) WHERE warm_rank<=12 ORDER BY customer_id,warm_rank", [str(Path(assets["warm150"]["path"]).resolve())]).fetchdf()
        oldcold = con.execute("SELECT * FROM read_parquet(?) ORDER BY customer_id,cold_rank,article_id", [str(Path(assets["cold50"]["path"]).resolve())]).fetchdf()
        con.register("original_cold", oldcold[["customer_id", "article_id", "cold_rank"]])
        cold_only = con.execute("SELECT w.article_id IS NULL AS cold_only FROM original_cold c LEFT JOIN read_parquet(?) w USING(customer_id,article_id) ORDER BY c.customer_id,c.cold_rank,c.article_id", [str(Path(assets["warm150"]["path"]).resolve())]).fetchnumpy()["cold_only"]
    pd.testing.assert_frame_equal(warm[list(oldwarm)], oldwarm, check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(cold[list(oldcold)], oldcold, check_dtype=False, check_exact=True)
    assert not cold.duplicated(["customer_id", "article_id"]).any()
    assert cold.groupby("customer_id").size().eq(50).all()
    assert cold.b0_score_available.eq(1).all()
    assert cold.interaction_count_before_cutoff.between(0, 5).all()
    np.testing.assert_array_equal(cold.source_branch.eq("cold_only"), cold_only)
    np.testing.assert_array_equal(warm.warm_rank, np.tile(np.arange(1, 13), len(users)))
    assert not truth.duplicated(["customer_id", "article_id"]).any()
    truthpairs = set(zip(truth.customer_id, truth.article_id))
    for side, frame in (("qC", cold), ("qW", warm)):
        model = read_json(Path(training[side]["model"]["path"]))
        prep = read_json(Path(training[side]["preprocessing"]["path"]))
        q = _replay(frame, model, prep)
        np.testing.assert_array_equal(frame.propensity, q)
        actual = [int((u, item) in truthpairs) for u, item in zip(frame.customer_id, frame.article_id)]
        np.testing.assert_array_equal(frame.target, actual)
        _verify_calibration(frame.target, q, calibration[side])
    user_index = {u: i for i, u in enumerate(users)}
    assert set(cold.customer_id) <= set(users)
    expected = cold.copy()
    expected["cold_row_index"] = np.arange(len(cold), dtype=np.int64)
    expected["user_index"] = expected.customer_id.map(user_index).astype(np.int32)
    warmsets = {u: set(row) for u, row in zip(users, original)}
    overlap = np.array([item in warmsets[u] for u, item in zip(cold.customer_id, cold.article_id)])
    expected["qC"] = np.clip(cold.propensity.to_numpy(), 1e-6, 1-1e-6)
    expected = expected.loc[~overlap].sort_values(["user_index", "cold_rank", "article_id"], ignore_index=True)
    # The saved eligible frame was made before propensity was appended to the
    # original qC frame by the driver; all shared identity/evidence columns match.
    pd.testing.assert_frame_equal(eligible, expected[list(eligible)], check_dtype=False, check_exact=True)
    qwarm = np.clip(warm.propensity.to_numpy().reshape(-1, 12), 1e-6, 1-1e-6)
    qcold = expected.qC.to_numpy()
    utility = np.log(qcold/(1-qcold))[:, None] - np.log(qwarm/(1-qwarm))[expected.user_index.to_numpy()]
    stored_utility = np.load(folder / "pair-utility.float64.npy", mmap_mode="r", allow_pickle=False)
    np.testing.assert_array_equal(stored_utility, utility)
    assert stored_utility.dtype == np.float64
    sizes = expected.groupby("user_index", sort=False).size()
    _close(metrics["pair_audit"], {"users": len(users), "users_with_eligible_cold": len(sizes),
        "cold_nodes": len(expected), "warm_nodes": len(warm), "pair_rows": utility.size,
        "overlap_excluded_cold_rows": int(overlap.sum()), "complete_cold_rows": len(cold),
        "maximum_pairs_per_user": int(sizes.max()*12) if len(sizes) else 0})
    derived = copy.deepcopy(metrics)
    reconstruction = {v: original.copy() for v in TAUS}
    independent_pairs = []
    matching_checks = {v: {"edges_above_tau": 0, "matched_edges": 0,
        "same_warm_conflict_count": 0, "same_cold_conflict_count": 0} for v in TAUS}
    for ui, group in expected.groupby("user_index", sort=False):
        ui = int(ui)
        matrix = utility[group.index.to_numpy()]
        items = group.article_id.to_numpy()
        for variant, tau in TAUS.items():
            matches = _assignment(matrix, tau)
            assert len(set(c for c, _ in matches)) == len(matches)
            assert len(set(w for _, w in matches)) == len(matches)
            edges = matrix > tau
            stats = matching_checks[variant]
            stats["edges_above_tau"] += int(edges.sum())
            stats["matched_edges"] += len(matches)
            stats["same_warm_conflict_count"] += int((edges.sum(axis=0) > 1).sum())
            stats["same_cold_conflict_count"] += int((edges.sum(axis=1) > 1).sum())
            for ci, wi in matches:
                assert matrix[ci, wi] > tau
                reconstruction[variant][ui, wi] = items[ci]
                row = group.iloc[ci]
                independent_pairs.append((variant, ui, wi+1, items[ci], original[ui, wi], int(row.cold_row_index)))
    actual_pairs = [(r.variant, int(r.user_index), int(r.warm_slot_rank), r.cold_article_id,
                     r.warm_article_id, int(r.cold_row_index)) for r in executed.itertuples()]
    assert sorted(actual_pairs) == sorted(independent_pairs)
    assert not executed.duplicated(["variant", "customer_id", "cold_article_id"]).any()
    assert not executed.duplicated(["variant", "customer_id", "warm_slot_rank"]).any()
    for variant, stats in matching_checks.items():
        stats["matching_efficiency"] = stats["matched_edges"]/stats["edges_above_tau"] if stats["edges_above_tau"] else 0.0
        _close(metrics["matching"][variant], stats)
    replacement_checks = 0
    warm_rows = warm.set_index(["customer_id", "warm_rank"])
    cold_rows = cold.reset_index(drop=True)
    eligible_positions = {int(row): i for i, row in enumerate(expected.cold_row_index)}
    truthsets = {u: set(g.article_id) for u, g in truth.groupby("customer_id", sort=False)}
    reference_ap = np.array([apk(list(truthsets[u]), list(row)) for u, row in zip(users, original)])
    for r in executed.itertuples():
        cr = cold_rows.iloc[int(r.cold_row_index)]
        wr = warm_rows.loc[(r.customer_id, int(r.warm_slot_rank))]
        assert cr.article_id == r.cold_article_id and wr.article_id == r.warm_article_id
        assert int(cr.target) == r.inserted_positive and int(wr.target) == r.removed_positive
        assert r.net_positive == r.inserted_positive-r.removed_positive
        assert r.cold_only == (cr.source_branch == "cold_only")
        assert r.strict_cold == (cr.interaction_count_before_cutoff == 0)
        assert r.sparse1_5 == (1 <= cr.interaction_count_before_cutoff <= 5)
        assert r.removed_warm21_positive == int(wr.target == 1 and wr.interaction_count_before_cutoff >= 21)
        tau = TAUS[r.variant]
        qc, qw = float(np.clip(cr.propensity, 1e-6, 1-1e-6)), float(np.clip(wr.propensity, 1e-6, 1-1e-6))
        pair_u = float(utility[eligible_positions[int(r.cold_row_index)], int(r.warm_slot_rank)-1])
        _close({"tau": r.tau, "qC": r.qC, "qW": r.qW, "utility": r.utility, "edge_weight": r.edge_weight},
               {"tau": tau, "qC": qc, "qW": qw, "utility": pair_u, "edge_weight": pair_u-tau}, exact=True)
        single = original[int(r.user_index)].copy()
        single[int(r.warm_slot_rank)-1] = r.cold_article_id
        expected_delta = apk(list(truthsets[r.customer_id]), list(single)) - reference_ap[int(r.user_index)]
        np.testing.assert_allclose(r.individual_delta_ap, expected_delta, rtol=0, atol=2e-16)
        replacement_checks += 1
    assert set(metrics["variants"]) == {"W0", *TAUS}
    assert {p.name for p in folder.glob("*-top12.parquet")} == {f"{v}-top12.parquet" for v in ("W0", *TAUS)}
    for variant in ("W0", *TAUS):
        top = _prediction_list(parquet(folder / f"{variant}-top12.parquet"), users)
        np.testing.assert_array_equal(top, original if variant == "W0" else reconstruction[variant])
        actual_metrics, ap, baseline_ap = _metrics_from_lists(users, top, truth, original)
        _close(metrics["variants"][variant], actual_metrics, exact=True)
        if variant == "W0":
            assert actual_metrics["map12"] == assets["w0_map"]
        derived["variants"][variant].update(actual_metrics)
        records = executed.loc[executed.variant.eq(variant)]
        counts = (top != original).sum(axis=1)
        assert counts.max() <= 12
        np.testing.assert_array_equal(top[counts == 0], original[counts == 0])
        ui = records.user_index.to_numpy(dtype=np.int64)
        np.testing.assert_array_equal(counts, np.bincount(ui, minlength=len(users)))
        inserted = np.bincount(ui, weights=records.inserted_positive, minlength=len(users)).astype(np.int64)
        removed = np.bincount(ui, weights=records.removed_positive, minlength=len(users)).astype(np.int64)
        admission = {"users_with_admission": int((counts > 0).sum()), "admission_user_share": float((counts > 0).mean()),
            "total_replacements": int(counts.sum()), "max_replacements": int(counts.max()),
            "mean_replacements_per_admitted_user": float(counts[counts > 0].mean()) if (counts > 0).any() else 0.0,
            "inserted_cold_positive_pairs": int(inserted.sum()), "removed_warm_positive_pairs": int(removed.sum()),
            "net_positive_pairs": int(inserted.sum()-removed.sum()),
            "beneficial_replacements": int(records.net_positive.gt(0).sum()),
            "neutral_replacements": int(records.net_positive.eq(0).sum()),
            "harmful_replacements": int(records.net_positive.lt(0).sum()),
            "removed_warm21_positive_pairs": int(records.removed_warm21_positive.sum()),
            "cold_only_positive_top12": int((records.cold_only.astype(bool) & records.inserted_positive.eq(1)).sum()),
            "cold_only_positive_actually_inserted": int((records.cold_only.astype(bool) & records.inserted_positive.eq(1)).sum()),
            "strict_cold_positive_insertions": int((records.strict_cold.astype(bool) & records.inserted_positive.eq(1)).sum()),
            "sparse1_5_positive_insertions": int((records.sparse1_5.astype(bool) & records.inserted_positive.eq(1)).sum()),
            "joint_beneficial_users": int((ap > baseline_ap).sum()), "joint_harmful_users": int((ap < baseline_ap).sum()),
            "joint_neutral_users": int((ap == baseline_ap).sum())}
        _close(metrics["variants"][variant]["admission"], admission, exact=True)
        derived["variants"][variant]["admission"] = admission
        ua = user_audit.loc[user_audit.variant.eq(variant)]
        np.testing.assert_array_equal(ua.customer_id, users)
        np.testing.assert_array_equal(ua.ap, ap)
        np.testing.assert_array_equal(ua.baseline_ap, baseline_ap)
        np.testing.assert_array_equal(ua.admissions, counts)
        np.testing.assert_array_equal(ua.inserted_positive, inserted)
        np.testing.assert_array_equal(ua.removed_positive, removed)
        for bucket in ("0", "1", "2", "3", "4+"):
            mask = counts >= 4 if bucket == "4+" else counts == int(bucket)
            avg = float(ap[mask].mean()) if mask.any() else None
            ref = float(baseline_ap[mask].mean()) if mask.any() else None
            observed = {"users": int(mask.sum()), "map12": avg, "w0_map12_same_users": ref,
                "delta_vs_same_users_w0": avg-ref if avg is not None else None,
                "map_contribution_full_window": float(ap[mask].sum()/len(users)),
                "delta_contribution_full_window": float((ap[mask]-baseline_ap[mask]).sum()/len(users)),
                "inserted_cold_positives": int(inserted[mask].sum()), "removed_warm_positives": int(removed[mask].sum()),
                "net_positives": int((inserted[mask]-removed[mask]).sum())}
            _close(metrics["variants"][variant]["admission_buckets"][bucket], observed, exact=True)
            derived["variants"][variant]["admission_buckets"][bucket] = observed
    return derived, {"pass": True, "w0_users": len(users), "cold50_rows": len(cold),
        "propensity_rows_replayed": {"qC": len(cold), "qW": len(warm)}, "pair_values_recomputed_exact": int(utility.size),
        "exact_assignment_user_thresholds": len(sizes)*3, "executed_replacements_checked": replacement_checks,
        "reference_ap_users_per_variant": len(users), "variants": 4, "segment_denominators_recomputed": 16,
        "admission_count_buckets_recomputed": 20, "maximum_admissions_observed": max(m["admission"]["max_replacements"] for m in derived["variants"].values())}


def _verify_failure_boundary(repo, root, contract, metrics, training_doc, checks):
    """Verify a truthful stopped run, never describe skipped experiments as pass."""
    assert metrics["decision"] == "engineering_failure" and metrics["selected_variant"] == "W0"
    assert metrics["failed_component"] == "winter_20200122/qW" and metrics["summary"] is None
    assert set(training_doc["windows"]) == {"winter_20200122"}
    assert set(training_doc["windows"]["winter_20200122"]) == {"qC", "qW"}
    warm_audit = training_doc["windows"]["winter_20200122"]["qW"]
    cold_audit = training_doc["windows"]["winter_20200122"]["qC"]
    assert cold_audit["status"] == "converged" and not cold_audit["convergence_warnings"]
    assert warm_audit["status"] == "non_converged" and warm_audit["convergence_warnings"]
    warning = "\n".join(warm_audit["convergence_warnings"])
    assert "1000" in warning and "ITERATIONS REACHED LIMIT" in warning
    model = read_json(Path(warm_audit["model"]["path"]))
    assert model["n_iter"] == [1000] and model["params"]["max_iter"] == 1000
    assert np.isfinite(model["coefficient"]).all() and np.isfinite(model["intercept"])
    assert metrics["resources"]["model_fits_attempted"] == 2
    assert metrics["resources"]["converged_model_fits"] == 1
    assert set(p.name for p in (root/"models").iterdir() if p.is_dir()) == {"winter_20200122"}
    assert not (root/"outer").exists() or not any(p.is_file() for p in (root/"outer").rglob("*"))
    assert not any("-predictions.parquet" in r["path"] or "-top12.parquet" in r["path"] or "pair-utility" in r["path"] for r in metrics["artifacts"])
    for window in WINDOWS:
        assert metrics["windows"][window]["status"] == "not_run"
    for name in ("propensity_calibration", "pair_utility_audit", "matching_audit", "admission_risk", "segment_metrics"):
        doc = read_json(repo / "reports/phase4" / f"p4_2_{name}.json")
        assert doc["status"] == "not_run" and doc["final_week"] == "not_run"
    prepared_checks = {}
    with duckdb.connect() as con:
        con.execute("SET threads=4")
        con.execute("SET memory_limit='4GB'")
        for cutoff in contract["existing_cutoff_sequence"]:
            prepared = training_doc["prepared_cutoffs"][cutoff]
            source = contract["inputs"][cutoff]
            assert prepared["parity"]["w0_identity_order_exact"]
            assert prepared["parity"]["b0_cold50_exact"]
            sides = {}
            for side in ("qC", "qW"):
                entry = prepared["features"][side]
                frame = con.execute("SELECT * FROM read_parquet(?)", [entry["path"]]).fetchdf()
                assert len(frame) == entry["row_count"] and frame.target_cutoff.eq(cutoff).all()
                assert int(frame.target.sum()) == entry["positive_rows"]
                if side == "qW":
                    original = con.execute("SELECT customer_id,article_id,warm_rank,warm_rank_pct FROM read_parquet(?) WHERE warm_rank<=12 ORDER BY customer_id,warm_rank", [str(Path(source["warm150"]["path"]).resolve())]).fetchdf()
                    np.testing.assert_array_equal(frame.warm_rank, np.tile(np.arange(1,13), len(frame)//12))
                    assert not frame.duplicated(["customer_id", "article_id"]).any()
                else:
                    original = con.execute("SELECT customer_id,article_id,b0_rank_pct,m4_coarse_rank_pct,b0_score_available FROM read_parquet(?) ORDER BY customer_id,cold_rank,article_id", [str(Path(source["cold50"]["path"]).resolve())]).fetchdf()
                    assert frame.groupby("customer_id").size().eq(50).all()
                    assert frame.b0_score_available.eq(int(source["b0_lineage"]["available"])).all()
                pd.testing.assert_frame_equal(frame[list(original)], original, check_dtype=False, check_exact=True)
                con.register("prepared_side", frame[["customer_id", "article_id", "target", "interaction_count_before_cutoff"]])
                discrepancies = con.execute("""SELECT count(*) FROM prepared_side p
                    LEFT JOIN read_parquet(?) f USING(customer_id,article_id)
                    WHERE p.target IS DISTINCT FROM f.target
                       OR p.interaction_count_before_cutoff IS DISTINCT FROM f.interaction_count_before_cutoff""",
                    [str(Path(source["features"]["path"]).resolve())]).fetchone()[0]
                assert discrepancies == 0
                sides[side] = {"rows": len(frame), "users": int(frame.customer_id.nunique()),
                               "exact_frozen_id_rank_and_target_count_parity": True}
            if cutoff in WINDOWS.values():
                assert prepared["parity"]["w0_map_bit_exact"]
                assert prepared["parity"]["w0_map"] == source["w0_map"]
                assert prepared["parity"]["all_users"] == source["w0_users"]
            prepared_checks[cutoff] = sides
    names = ["branch_main", "frozen_w0_identity", "frozen_w0_map_parity", "frozen_b0_cold50_identity", "no_future_b0",
             "historical_label_end_strictly_before_outer", "final_week_fail_closed", "training_only_preprocessing",
             "no_negative_sampling", "no_class_weight", "qc_forbidden_features_excluded", "qw_raw_ids_excluded",
             "probability_clip", "utility_formula", "three_fixed_taus", "cold_w0_overlap_excluded", "max600_pairs",
             "strict_threshold_edges", "exact_not_greedy_matching", "unique_cold_match_nodes", "unique_warm_match_slots",
             "no_hard_max1", "unique_final_top12", "no_edge_exact_w0", "all_recorded_artifact_shas"]
    not_run = set(range(13,15)) | set(range(16,25))
    test_checks = [{"number": i, "name": name, "status": "not_run" if i in not_run else "pass",
                   "scope": "formal outer computation skipped at convergence stop" if i in not_run else
                       ("frozen W0 replay from successful preparation, not a P4.2 admission MAP" if i == 3 else "available frozen/prepared/attempted-fit evidence")}
                  for i, name in enumerate(names,1)]
    checks["prepared_frozen_identity_target_and_count_parity"] = prepared_checks
    checks["expected_convergence_stop"] = {"status": "fail", "expected": True, "component": "winter_20200122/qW",
        "n_iter": 1000, "warning": warning, "reason": "this failed experiment gate correctly prevents all outer evaluations"}
    checks["stop_enforcement"] = {"pass": True, "outer_propensity_files": 0, "outer_top12_files": 0,
        "other_outer_models": 0, "retry_model_fits": 0, "all_outer_status": "not_run"}
    return {"experiment_status": "engineering_failure", "verification_interpretation": "pass validates truthful failure closure, not successful P4.2 evaluation",
        "required_checks": test_checks, "formal_checks_passed": sum(x["status"] == "pass" for x in test_checks),
        "formal_checks_not_run": sum(x["status"] == "not_run" for x in test_checks),
        "convergence_gate": "fail_expected_stop", "outer_metrics": "not_run", "machine_decision_recomputed": True,
        "model_fits_verified": 2, "Warm_v2_integrated": False, "selected_variant": "W0"}


def verify(repo):
    repo = Path(repo).resolve()
    # A wrong branch must cause zero writes, including the failure report.
    branch = branch_guard(repo)
    started = time.perf_counter()
    report = repo / "reports/phase4"
    root = repo / "artifacts/phase4" / RUN_ID
    checks = {}
    result = {"stage": "P4.2", "status": "running", "branch": branch, "checks": checks, "final_week": "not_run"}
    try:
        contract = read_json(report / "P4_2_EXPERIMENT_CONTRACT.json")
        metrics = read_json(report / "P4_2_metrics.json")
        training_doc = read_json(report / "p4_2_training_audit.json")
        calibration_doc = read_json(report / "p4_2_propensity_calibration.json")
        receipt = read_json(root / "input-verification.json")
        assert contract["status"] == "preregistered_before_formal_computation"
        assert contract["windows"] == WINDOWS and contract["taus"] == TAUS
        assert contract["epsilon"] == EPSILON == 1e-6
        assert contract["feature_spec"] == FEATURE_SPEC
        assert contract["model"]["thread_limit"] == THREAD_LIMIT
        assert {k: contract["model"][k] for k in MODEL_PARAMS} == MODEL_PARAMS
        for library, version in contract["software_versions"].items():
            assert importlib.metadata.version(library) == version
        assert set(metrics["windows"]) == set(WINDOWS)
        assert all(d["final_week"] == "not_run" for d in (contract, metrics, training_doc, calibration_doc, receipt))
        assert not metrics["historical_boundaries"]["Warm_v2_integrated"]
        assert not metrics["historical_boundaries"]["P4_1B_started"]
        assert not metrics["historical_boundaries"]["p4_1b_allowed"]
        assert not metrics["historical_boundaries"]["P4_3_started"]
        for cutoff in ("2020-09-16", "2020-09-17", "2026-09-09", "2020-09-10"):
            try:
                guard_cutoff(cutoff)
            except ValueError:
                continue
            raise AssertionError("forbidden final/overlap cutoff accepted")
        for cutoff in contract["existing_cutoff_sequence"]:
            guard_cutoff(cutoff)
        forbidden_c = {"b0_score", "b0_delta_vs_m4", "article_id", "target"}
        assert not forbidden_c.intersection(FEATURE_SPEC["qC"]["numeric"]+FEATURE_SPEC["qC"]["binary"])
        assert all(not x.startswith("article_") and "b0_" not in x and "cold_" not in x and x != "target"
                   for x in FEATURE_SPEC["qW"]["numeric"]+FEATURE_SPEC["qW"]["binary"])
        checks["branch_runtime_preregistration_features_final_boundary"] = {"pass": True, "branch": branch, "tau_count": 3}
        seen = {}

        def check_record(record):
            key = str(Path(record["path"]).resolve())
            assert "warm_v2" not in key.lower() and "warm-v2" not in key.lower(), key
            if key in seen:
                assert seen[key] == (record["bytes"], record["sha256"]), key
                return
            check_identity({**record, "path": key})
            seen[key] = (record["bytes"], record["sha256"])

        # Explicit task requirement: trusted frozen receipts and recorded output
        # identities, not an unauthenticated current-file-only hash exercise.
        for record in receipt["trusted_comparisons"]:
            check_record(record)
        for record in identities({"inputs": contract["inputs"], "transactions": contract["transactions"], "catalog": contract["catalog"]}):
            check_record(record)
        frozen_count = len(seen)
        assert frozen_count == receipt["input_files"]
        for record in receipt["authority_snapshots"].values():
            check_record({"path": record["snapshot_path"], "bytes": record["bytes"], "sha256": record["sha256"]})
        check_record(metrics["contract"])
        assert metrics["contract"]["path"] == metrics["execution"]["contract"]["path"]
        assert contract["created_at_utc"] < metrics["execution"]["started_at_utc"]
        computational_sources = []
        closure_sources = []
        for record in metrics["execution"]["implementation_before_formal"]:
            if Path(record["path"]).name in {"p42_report.py", "p42_verify.py"}:
                # These are downstream presentation/verification, not inputs
                # to the completed formal fits or admission computation. Their
                # final identities belong in the closure output manifest.
                closure_sources.append(record["path"])
                continue
            check_record(record)
            computational_sources.append(record["path"])
        checks["formal_computation_source_identity"] = {
            "pass": True, "checked_paths": computational_sources,
            "downstream_closure_paths_deferred_to_final_manifest": closure_sources,
        }
        for record in identities(metrics["artifacts"]):
            check_record(record)
            if record["path"].endswith(".parquet"):
                with duckdb.connect() as con:
                    assert con.execute("SELECT count(*) FROM read_parquet(?)", [str(Path(record["path"]).resolve())]).fetchone()[0] == record["row_count"]
            elif record["path"].endswith(".npy"):
                a = np.load(Path(record["path"]), mmap_mode="r", allow_pickle=False)
                assert list(a.shape) == record["shape"] and len(a) == record["row_count"]
        for record in identities(training_doc):
            check_record(record)
        for cutoff, item in contract["inputs"].items():
            for line, endkey, trainkey, safe in (
                (item["b0_lineage"], "model_label_end", "model_training_cutoffs", "lineage_safe"),
                (item["warm_lineage"], "latest_training_label_end", "training_cutoffs", "safe")):
                assert line[safe]
                if line["available"]:
                    ends = [(date.fromisoformat(c)+timedelta(days=7)).isoformat() for c in line[trainkey]]
                    assert ends and max(ends) == line[endkey] and max(ends) <= cutoff
                    assert all(c < cutoff for c in line[trainkey])
            if item["b0_lineage"]["available"]:
                assert "B0" in Path(item["b0_lineage"]["model"]["path"]).name
                assert "B1" not in Path(item["b0_lineage"]["model"]["path"]).name
        training_checks = {}
        for window, outer in WINDOWS.items():
            if metrics["decision"] == "engineering_failure" and window not in training_doc["windows"]:
                continue
            print(f"P4.2 verify training-only preprocessing: {window}", flush=True)
            training_checks[window] = {}
            for side in ("qC", "qW"):
                audit = training_doc["windows"][window][side]
                allowed = contract["historical_pools"][window][side]
                assert audit["training_cutoffs"] == allowed
                assert [d["cutoff"] for d in audit["cutoff_details"]] == allowed
                assert audit["outer_cutoff"] == outer
                assert contract["created_at_utc"] < audit["fit_started_at_utc"] < audit["fit_finished_at_utc"]
                assert max(d["label_end"] for d in audit["cutoff_details"]) == audit["latest_training_label_end"] < outer
                frames = []
                spec = contract["feature_spec"][side]
                needed = [*spec["numeric"], *spec["binary"], "target", "target_cutoff"]
                if side == "qC":
                    needed.append("b0_score_available")
                for detail in audit["cutoff_details"]:
                    cutoff = detail["cutoff"]
                    assert (date.fromisoformat(cutoff)+timedelta(days=7)).isoformat() == detail["label_end"] < outer
                    expected = training_doc["prepared_cutoffs"][cutoff]["features"][side]
                    assert detail["artifact"] == expected
                    assert Path(expected["path"]).resolve() == (root/"prepared"/cutoff/f"{side}-features.parquet").resolve()
                    with duckdb.connect() as con:
                        frame = con.execute("SELECT " + ",".join('"'+x+'"' for x in needed) + " FROM read_parquet(?)", [expected["path"]]).fetchdf()
                    assert frame.target_cutoff.eq(cutoff).all() and len(frame) == detail["rows"]
                    assert int(frame.target.sum()) == detail["positive_rows"]
                    if side == "qC":
                        assert frame.b0_score_available.eq(1).all()
                    frames.append(frame)
                model = read_json(Path(audit["model"]["path"]))
                prep = read_json(Path(audit["preprocessing"]["path"]))
                assert model["params"] == MODEL_PARAMS and model["classes"] == [0, 1]
                assert all(0 < n <= MODEL_PARAMS["max_iter"] for n in model["n_iter"])
                assert model["lineage"] == prep["lineage"]
                assert model["lineage"]["training_cutoffs"] == allowed
                assert model["lineage"] == audit["model"]["source_lineage"] == audit["preprocessing"]["source_lineage"]
                _close(model["lineage"], {"side": side, "outer_cutoff": outer,
                    "latest_training_label_end": audit["latest_training_label_end"],
                    "fit_started_at_utc": audit["fit_started_at_utc"], "fit_finished_at_utc": audit["fit_finished_at_utc"],
                    "training_only_preprocessing": True})
                training_checks[window][side] = _verify_preprocessing(frames, prep, audit, spec,
                    require_convergence=not (metrics["decision"] == "engineering_failure" and audit["status"] == "non_converged"))
                del frames
                gc.collect()
        checks["training_temporal_full_pool_and_preprocessing"] = training_checks
        checks["frozen_and_output_sha"] = {"pass": True, "frozen_input_files": frozen_count, "total_files": len(seen)}
        if metrics["decision"] == "engineering_failure":
            result.update(_verify_failure_boundary(repo, root, contract, metrics, training_doc, checks))
            result.update({"status": "pass", "all_artifact_sha_pass": True,
                "output_and_input_sha_files": len(seen), "frozen_input_sha_files": frozen_count,
                "seconds": time.perf_counter()-started})
            write_json(report / "P4_2_VERIFICATION.json", result)
            return result
        computed, window_checks = {}, {}
        for window in WINDOWS:
            print(f"P4.2 independent matching and AP replay: {window}", flush=True)
            computed[window], window_checks[window] = _verify_window(repo, root, contract, metrics["windows"][window],
                training_doc["windows"][window], calibration_doc["windows"][window], window)
            gc.collect()
        from .p42_evaluate import aggregate_windows
        aggregate = aggregate_windows(computed, calibration_doc["windows"], contract)
        _close(metrics, aggregate, exact=True)
        checks["windows"] = window_checks
        checks["gate_and_machine_decision_recomputed"] = {"pass": True, "decision": aggregate["decision"]}
        # Source validation confirms that this verification path contains no
        # model fitting call; no success flag merely echoes an execution claim.
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"fit", "fit_transform", "partial_fit"}
        result.update({"status": "pass", "windows": window_checks, "all_artifact_sha_pass": True,
            "output_and_input_sha_files": len(seen), "frozen_input_sha_files": frozen_count,
            "model_fits_verified": 8, "reference_metric_and_exact_matching_replayed": True,
            "machine_decision_recomputed": True, "Warm_v2_integrated": False})
    except Exception as exc:
        result.update({"status": "fail", "reason": str(exc), "traceback": traceback.format_exc()})
        result["seconds"] = time.perf_counter()-started
        write_json(report / "P4_2_VERIFICATION.json", result)
        raise
    result["seconds"] = time.perf_counter()-started
    write_json(report / "P4_2_VERIFICATION.json", result)
    return result
