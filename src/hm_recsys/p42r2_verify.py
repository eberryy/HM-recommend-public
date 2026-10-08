"""Independent P4.2R2 verification; previous-stage evidence is read-only.

The numerical helper recomputes logical input/scaler statistics and local
curvature without estimator fitting. Admission verification separately rebuilds
all assignments, per-user lists, segment denominators and replacement risk.
"""
from __future__ import annotations

import copy
import gc
import subprocess
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .metrics import apk
from .p41a_contract import check_identity, read_json, write_json
from .p41a_data import identities, parquet
from .p42_contract import FEATURE_SPEC, TAUS, WINDOWS, guard_cutoff
from .p42_propensity import MODEL_PARAMS
from .p42_verify import _assignment, _close, _metrics_from_lists, _prediction_list, _verify_calibration, _verify_preprocessing
from .p42r_contract import LOG1P_COUNTS
from .p42r_verify import independent_qw_training, _predict, _verify_population


INVARIANTS = [
    "P4_2_and_P4_2R_history_preserved", "branch_main_not_Warm", "final_week_fail_closed",
    "W0_exact_parity", "M4_B0_frozen_parity", "qC_population_exact_P4_2R",
    "qC_qW_shared_H_t_users", "all_qW_chains_R1_plus_R2", "no_per_window_preprocessing_exception",
    "R1_same_deterministic_cleanup", "R2_exact_19_count_list", "R2_finite_counts_nonnegative",
    "transform_order_exact", "StandardScaler_training_only", "LBFGS_unchanged", "C_unchanged",
    "tol_unchanged", "max_iter_unchanged", "no_class_weight", "no_negative_sampling",
    "all_four_qW_converged_before_calibration", "utility_unchanged", "three_taus_unchanged",
    "exact_matching_unchanged", "no_hard_max1", "exactly12_unique_final_items",
    "no_edge_exact_W0", "all_recorded_SHAs_match",
]
FAILURES = {"qW_global_R2_convergence_failure", "propensity_calibration_failure", "engineering_failure"}


def verify_global_sequence(attempts, calibration, admission, decision):
    """All attempts are the same global R2 recipe, never a window rescue."""
    assert len(attempts) <= 4
    assert [a["window"] for a in attempts] == list(WINDOWS)[:len(attempts)]
    for index, attempt in enumerate(attempts):
        assert attempt["side"] == "qW" and attempt["repair"] == "R2"
        assert attempt["status"] in ("converged", "non_converged")
        assert (attempt["status"] == "converged") == (not attempt["convergence_warnings"])
        if attempt["status"] == "non_converged":
            assert index == len(attempts)-1 and decision == "qW_global_R2_convergence_failure"
    converged = len(attempts) == 4 and all(a["status"] == "converged" for a in attempts)
    if calibration or admission:
        assert converged, "calibration/admission reached before all four qW fits converged"
    if decision == "qW_global_R2_convergence_failure":
        assert attempts and attempts[-1]["status"] == "non_converged"
        assert not calibration and not admission
    return {"pass": True, "attempts": len(attempts), "converged_attempts": sum(a["status"] == "converged" for a in attempts),
            "all_four_converged": converged, "recipe": "R1 cleanup + global R2 log1p"}


def independent_calibration_gate(calibration, window_order=WINDOWS):
    assert set(calibration) == set(window_order)
    counts = {}
    for side in ("qC", "qW"):
        rows = [calibration[w][side] for w in window_order]
        counts[side] = {
            "extreme_rate_ratio_windows": sum(r["predicted_to_observed_rate_ratio"] is not None and not .1 <= r["predicted_to_observed_rate_ratio"] <= 10 for r in rows),
            "roc_not_above_chance_windows": sum(r["roc_auc"] is not None and r["roc_auc"] <= .5 for r in rows),
            "constant_prediction_windows": sum("constant_prediction" in r.get("warnings", []) for r in rows),
        }
    return {"pass": not any(value >= 3 for side in counts.values() for value in side.values()), "counts": counts}


def verify_admission_bundle(data, cold, warm, eligible, executed, user_audit, utility, recommendations, metrics):
    """Pure independent audit of a complete one-window saved result bundle."""
    users = np.asarray(data["users"])
    original = np.asarray(data["warm_lists"])
    assert len(warm) == len(users)*12 == original.size
    np.testing.assert_array_equal(warm.customer_id, np.repeat(users, 12))
    np.testing.assert_array_equal(warm.warm_rank, np.tile(np.arange(1, 13), len(users)))
    np.testing.assert_array_equal(warm.article_id.to_numpy().reshape(-1, 12), original)
    truthsets = {u: set(g.article_id) for u, g in data["truth"].groupby("customer_id", sort=False)}
    assert set(users) == set(truthsets)
    for frame in (cold, warm):
        assert not frame.duplicated(["customer_id", "article_id"]).any()
        np.testing.assert_array_equal(frame.target, [int(i in truthsets[u]) for u, i in zip(frame.customer_id, frame.article_id)])
        probability = frame.propensity.to_numpy(dtype=float)
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    expected = cold.copy()
    expected["cold_row_index"] = np.arange(len(cold), dtype=np.int64)
    user_index = {u: i for i, u in enumerate(users)}
    assert set(cold.customer_id) <= set(users)
    expected["user_index"] = expected.customer_id.map(user_index).astype(np.int32)
    warmsets = {u: set(row) for u, row in zip(users, original)}
    overlap = np.array([i in warmsets[u] for u, i in zip(cold.customer_id, cold.article_id)])
    expected["qC"] = np.clip(cold.propensity.to_numpy(), 1e-6, 1-1e-6)
    expected = expected.loc[~overlap].sort_values(["user_index", "cold_rank", "article_id"], ignore_index=True)
    pd.testing.assert_frame_equal(eligible, expected[list(eligible)], check_dtype=False, check_exact=True)
    sizes = expected.groupby("user_index", sort=False).size()
    assert sizes.le(50).all()
    qw = np.clip(warm.propensity.to_numpy().reshape(-1, 12), 1e-6, 1-1e-6)
    qc = expected.qC.to_numpy()
    reference_utility = np.log(qc/(1-qc))[:, None] - np.log(qw/(1-qw))[expected.user_index.to_numpy()]
    np.testing.assert_array_equal(utility, reference_utility)
    assert utility.dtype == np.float64
    _close(metrics["pair_audit"], {"users": len(users), "users_with_eligible_cold": len(sizes),
        "cold_nodes": len(expected), "warm_nodes": len(warm), "pair_rows": utility.size,
        "overlap_excluded_cold_rows": int(overlap.sum()), "complete_cold_rows": len(cold),
        "maximum_pairs_per_user": int(sizes.max()*12) if len(sizes) else 0})
    reconstructed = {v: original.copy() for v in TAUS}
    expected_pairs = []
    matching = {v: {"edges_above_tau": 0, "matched_edges": 0, "same_warm_conflict_count": 0,
                    "same_cold_conflict_count": 0} for v in TAUS}
    for ui, rows in expected.groupby("user_index", sort=False):
        ui = int(ui)
        block = reference_utility[rows.index.to_numpy()]
        for variant, tau in TAUS.items():
            assigned = _assignment(block, tau)
            assert len({i for i, j in assigned}) == len(assigned) == len({j for i, j in assigned})
            edge = block > tau
            matching[variant]["edges_above_tau"] += int(edge.sum())
            matching[variant]["matched_edges"] += len(assigned)
            matching[variant]["same_warm_conflict_count"] += int((edge.sum(axis=0) > 1).sum())
            matching[variant]["same_cold_conflict_count"] += int((edge.sum(axis=1) > 1).sum())
            for ci, wi in assigned:
                candidate = rows.iloc[ci]
                assert block[ci, wi] > tau
                reconstructed[variant][ui, wi] = candidate.article_id
                expected_pairs.append((variant, ui, wi+1, candidate.article_id, original[ui, wi], int(candidate.cold_row_index)))
    actual_pairs = [(r.variant, int(r.user_index), int(r.warm_slot_rank), r.cold_article_id,
                     r.warm_article_id, int(r.cold_row_index)) for r in executed.itertuples()]
    assert sorted(actual_pairs) == sorted(expected_pairs)
    assert not executed.duplicated(["variant", "customer_id", "cold_article_id"]).any()
    assert not executed.duplicated(["variant", "customer_id", "warm_slot_rank"]).any()
    for variant, stats in matching.items():
        stats["matching_efficiency"] = stats["matched_edges"]/stats["edges_above_tau"] if stats["edges_above_tau"] else 0.
        _close(metrics["matching"][variant], stats)
    base_ap = np.array([apk(list(truthsets[u]), list(items)) for u, items in zip(users, original)])
    warm_lookup = warm.set_index(["customer_id", "warm_rank"])
    location = {int(row): i for i, row in enumerate(expected.cold_row_index)}
    for row in executed.itertuples():
        c = cold.iloc[int(row.cold_row_index)]
        w = warm_lookup.loc[(row.customer_id, int(row.warm_slot_rank))]
        assert c.customer_id == row.customer_id and c.article_id == row.cold_article_id
        assert w.article_id == row.warm_article_id and users[int(row.user_index)] == row.customer_id
        assert int(c.target) == row.inserted_positive and int(w.target) == row.removed_positive
        assert row.net_positive == row.inserted_positive-row.removed_positive
        assert row.cold_only == (c.source_branch == "cold_only")
        assert row.strict_cold == (c.interaction_count_before_cutoff == 0)
        assert row.sparse1_5 == (1 <= c.interaction_count_before_cutoff <= 5)
        assert row.removed_warm21_positive == int(w.target == 1 and w.interaction_count_before_cutoff >= 21)
        tau = TAUS[row.variant]
        value = float(reference_utility[location[int(row.cold_row_index)], int(row.warm_slot_rank)-1])
        _close({"tau": row.tau, "qC": row.qC, "qW": row.qW, "utility": row.utility, "edge_weight": row.edge_weight},
               {"tau": tau, "qC": float(np.clip(c.propensity, 1e-6, 1-1e-6)),
                "qW": float(np.clip(w.propensity, 1e-6, 1-1e-6)), "utility": value, "edge_weight": value-tau}, exact=True)
        single = original[int(row.user_index)].copy()
        single[int(row.warm_slot_rank)-1] = row.cold_article_id
        local_delta = apk(list(truthsets[row.customer_id]), list(single))-base_ap[int(row.user_index)]
        np.testing.assert_allclose(row.individual_delta_ap, local_delta, rtol=0, atol=2e-16)
    assert set(recommendations) == set(metrics["variants"]) == {"W0", *TAUS}
    derived = copy.deepcopy(metrics)
    for variant, top in recommendations.items():
        top = np.asarray(top)
        assert top.shape == original.shape and all(len(set(row)) == 12 for row in top)
        np.testing.assert_array_equal(top, original if variant == "W0" else reconstructed[variant])
        actual, ap, baseline = _metrics_from_lists(users, top, data["truth"], original)
        _close(metrics["variants"][variant], actual, exact=True)
        derived["variants"][variant].update(actual)
        rows = executed.loc[executed.variant.eq(variant)]
        changed = (top != original).sum(axis=1)
        indices = rows.user_index.to_numpy(dtype=np.int64)
        counts = np.bincount(indices, minlength=len(users))
        np.testing.assert_array_equal(changed, counts)
        np.testing.assert_array_equal(top[counts == 0], original[counts == 0])
        assert counts.max() <= 12
        inserted = np.bincount(indices, weights=rows.inserted_positive.to_numpy(dtype=float), minlength=len(users)).astype(np.int64)
        removed = np.bincount(indices, weights=rows.removed_positive.to_numpy(dtype=float), minlength=len(users)).astype(np.int64)
        sums = lambda name: int(rows[name].sum())
        positive = lambda name: int((rows[name].astype(bool)&rows.inserted_positive.eq(1)).sum())
        admission = {"users_with_admission": int((counts > 0).sum()), "admission_user_share": float((counts > 0).mean()),
            "total_replacements": int(counts.sum()), "mean_replacements_per_admitted_user": float(counts[counts > 0].mean()) if (counts > 0).any() else 0.,
            "max_replacements": int(counts.max()), "beneficial_replacements": int(rows.net_positive.gt(0).sum()),
            "neutral_replacements": int(rows.net_positive.eq(0).sum()), "harmful_replacements": int(rows.net_positive.lt(0).sum()),
            "inserted_cold_positive_pairs": int(inserted.sum()), "removed_warm_positive_pairs": int(removed.sum()),
            "removed_warm21_positive_pairs": sums("removed_warm21_positive"), "net_positive_pairs": int(inserted.sum()-removed.sum()),
            "cold_only_positive_top12": positive("cold_only"), "cold_only_positive_actually_inserted": positive("cold_only"),
            "strict_cold_positive_insertions": positive("strict_cold"), "sparse1_5_positive_insertions": positive("sparse1_5"),
            "joint_beneficial_users": int((ap > baseline).sum()), "joint_harmful_users": int((ap < baseline).sum()),
            "joint_neutral_users": int((ap == baseline).sum())}
        _close(metrics["variants"][variant]["admission"], admission, exact=True)
        derived["variants"][variant]["admission"] = admission
        _close(metrics["variants"][variant]["candidate_truth_changes"], {
            "inserted_positives": int(inserted.sum()), "removed_positives": int(removed.sum()), "net_positives": int(inserted.sum()-removed.sum())}, exact=True)
        ua = user_audit.loc[user_audit.variant.eq(variant)]
        for field, values in (("customer_id", users), ("ap", ap), ("baseline_ap", baseline), ("delta", ap-baseline),
                              ("admissions", counts), ("inserted_positive", inserted), ("removed_positive", removed), ("net_positive", inserted-removed)):
            np.testing.assert_array_equal(ua[field], values)
        for bucket in ("0", "1", "2", "3", "4+"):
            mask = counts >= 4 if bucket == "4+" else counts == int(bucket)
            avg = float(ap[mask].mean()) if mask.any() else None
            ref = float(baseline[mask].mean()) if mask.any() else None
            value = {"users": int(mask.sum()), "map12": avg,
                "w0_map12_same_users": ref, "delta_vs_same_users_w0": avg-ref if avg is not None else None,
                "map_contribution_full_window": float(ap[mask].sum()/len(users)),
                "delta_contribution_full_window": float((ap[mask]-baseline[mask]).sum()/len(users)),
                "inserted_cold_positives": int(inserted[mask].sum()), "removed_warm_positives": int(removed[mask].sum()),
                "net_positives": int((inserted[mask]-removed[mask]).sum())}
            _close(metrics["variants"][variant]["admission_buckets"][bucket], value, exact=True)
            derived["variants"][variant]["admission_buckets"][bucket] = value
    return derived, {"pass": True, "users": len(users), "utility_values": int(utility.size),
        "executed_replacements": len(expected_pairs), "variants": 4, "segment_denominators": 16,
        "admission_buckets": 20, "no_edge_and_no_Cold_exact_abstention": True,
        "maximum_admissions": max(v["admission"]["max_replacements"] for v in derived["variants"].values())}


def independent_promotion(windowmetrics, contract):
    """Recompute frozen nine gate predicates and selection without production aggregate."""
    assert set(windowmetrics) == set(contract["windows"])
    windows = [windowmetrics[w] for w in contract["windows"]]
    gate = contract["gates"]
    summary = {}
    for variant in ("W0", *TAUS):
        rows = [w["variants"][variant] for w in windows]
        mean = float(np.mean([r["map12"] for r in rows]))
        baseline = float(np.mean([w["variants"]["W0"]["map12"] for w in windows]))
        deltas = [r["delta_vs_w0"] for r in rows]
        segments = {}
        for name in ("warm_21_plus", "strict_cold", "sparse1_5", "all_cold_sparse"):
            ds = [r["segments"][name]["delta_vs_w0"] for r in rows]
            value = float(np.mean([r["segments"][name]["map12"] for r in rows]))
            reference = float(np.mean([w["variants"]["W0"]["segments"][name]["map12"] for w in windows]))
            segments[name] = {"mean_map12": value, "mean_delta": value-reference,
                              "nondegrade_windows": sum(d >= 0 for d in ds), "worst_delta": min(ds), "deltas": ds}
        pooled = {name: sum(r["admission"][name] for r in rows) for name in (
            "users_with_admission", "total_replacements", "inserted_cold_positive_pairs", "removed_warm_positive_pairs",
            "removed_warm21_positive_pairs", "net_positive_pairs", "cold_only_positive_actually_inserted",
            "strict_cold_positive_insertions", "sparse1_5_positive_insertions")}
        pooled["admission_user_share"] = pooled["users_with_admission"]/sum(w["users"] for w in windows)
        pooled["admission_count_user_distribution"] = {b: sum(r["admission_buckets"][b]["users"] for r in rows) for b in ("0", "1", "2", "3", "4+")}
        cold_windows = sum(r["admission"]["cold_only_positive_actually_inserted"] > 0 for r in rows)
        warm, cold = segments["warm_21_plus"], segments["all_cold_sparse"]
        checks = {
            "overall_mean": mean-baseline >= gate["overall_mean_delta_min"],
            "overall_nondegrade": sum(d >= 0 for d in deltas) >= gate["overall_nondegrade_min"],
            "overall_worst": min(deltas) >= gate["overall_worst_delta_min"],
            "warm_mean": warm["mean_delta"] >= gate["warm_mean_delta_min"],
            "warm_protected_windows": sum(d >= gate["warm_window_delta_min"] for d in warm["deltas"]) >= gate["warm_protected_windows_min"],
            "cold_mean": cold["mean_delta"] > 0,
            "cold_nondegrade": cold["nondegrade_windows"] >= gate["cold_nondegrade_min"],
            "cold_only_positive_windows": cold_windows >= gate["cold_only_positive_windows_min"],
            "replacement_efficiency": pooled["inserted_cold_positive_pairs"] > pooled["removed_warm_positive_pairs"],
        }
        flags = {"checks": checks, "overall": all(checks[k] for k in ("overall_mean", "overall_nondegrade", "overall_worst")),
            "warm": checks["warm_mean"] and checks["warm_protected_windows"],
            "cold": checks["cold_mean"] and checks["cold_nondegrade"] and checks["cold_only_positive_windows"],
            "replacement_efficiency": checks["replacement_efficiency"], "all_pass": all(checks.values())}
        summary[variant] = {"mean_map12": mean, "mean_delta": mean-baseline, "nondegrade_windows": sum(d >= 0 for d in deltas),
            "worst_delta": min(deltas), "segments": segments, "admission_pooled": pooled,
            "cold_only_positive_windows": cold_windows, "gates": flags}
    passing = [v for v in TAUS if summary[v]["gates"]["all_pass"]]
    tolerance = contract["selection"]["near_equal_abs_tolerance"]
    if passing:
        best = max(summary[v]["segments"]["all_cold_sparse"]["mean_map12"] for v in passing)
        finalists = [v for v in passing if abs(summary[v]["segments"]["all_cold_sparse"]["mean_map12"]-best) <= tolerance]
        best = max(summary[v]["mean_map12"] for v in finalists)
        selected = max((v for v in finalists if abs(summary[v]["mean_map12"]-best) <= tolerance), key=lambda v: TAUS[v])
        decision = {"A_tau0": "promote_p4_2_aggressive", "M_tau_ln2": "promote_p4_2_moderate", "C_tau_ln4": "promote_p4_2_conservative"}[selected]
    else:
        selected = "W0"
        cs = [summary[v]["segments"]["all_cold_sparse"]["mean_map12"] for v in TAUS]
        risk = [-summary[v]["segments"]["warm_21_plus"]["mean_delta"] for v in TAUS]
        monotone = lambda values: values[0] >= values[1]-tolerance and values[1] >= values[2]-tolerance and values[0] > values[2]+tolerance
        if monotone(cs) and monotone(risk) and summary["A_tau0"]["segments"]["all_cold_sparse"]["mean_delta"] > 0:
            decision = "pareto_frontier_supported_but_no_safe_operating_point"
        elif any(summary[v]["segments"]["all_cold_sparse"]["mean_delta"] > 0 and not (summary[v]["gates"]["overall"] and summary[v]["gates"]["warm"]) for v in TAUS):
            decision = "warm_risk_uncontrolled"
        else:
            decision = "cold_gain_not_recovered"
    return {"summary": summary, "passing_variants": passing, "selected_variant": selected, "decision": decision}


def _verify_shared_reuse(prepared, previous, prior_root):
    """Read actual row support; old history/label computations are immutable."""
    assert prepared == previous
    result = {}
    with duckdb.connect() as con:
        con.execute("SET threads=4")
        con.execute("SET memory_limit='4GB'")
        for cutoff, entry in prepared.items():
            guard_cutoff(cutoff)
            roster = parquet(prior_root/"prepared"/cutoff/"shared-population-roster.parquet")
            expected = set(roster.loc[roster.has_mapped_history.astype(bool), "customer_id"])
            sides = {}
            for side in ("qC", "qW"):
                artifact = entry["features"][side]
                rows = con.execute("SELECT customer_id,count(*) AS n,sum(target) AS positive FROM read_parquet(?) GROUP BY customer_id", [artifact["path"]]).fetchdf()
                assert set(rows.customer_id) == expected
                assert rows.n.eq(50 if side == "qC" else 12).all()
                assert int(rows.n.sum()) == artifact["row_count"]
                assert int(rows.positive.sum()) == artifact["positive_rows"]
                sides[side] = {"users": len(rows), "rows": int(rows.n.sum()), "positives": int(rows.positive.sum())}
            result[cutoff] = {"pass": True, "H_t_users": len(expected), **sides,
                "evidence": "actual candidate users and row counts reread; cutoff-history/label lineage retained through previously verified unchanged artifacts"}
    return result


def _tracked_count(frame, prep, name="user_item_events_28d"):
    if name not in prep["columns"]:
        return {"feature": name, "status": "absent_after_R1"}
    values = frame[name].to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values)
    maximum = float(values[finite].max()) if finite.any() else None
    values = np.where(finite, values, prep["median"][name])
    logged = name in prep["log1p_applied_features"]
    if logged:
        assert (values >= 0).all()
        values = np.log1p(values)
    index = prep["scaler"]["columns"].index(name)
    z = (values-prep["scaler"]["mean"][index])/prep["scaler"]["scale"][index]
    return {"feature": name, "status": "measured", "raw_finite_max": maximum,
        "log1p_applied": logged, "max_standardized_absolute_value": float(np.abs(z).max())}


def verify(repo):
    """Verify only completed evidence; write no earlier-stage output files."""
    repo = Path(repo).resolve()
    # Wrong-branch requests must not write even a failed verification receipt.
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    if branch != "main":
        raise ValueError(f"P4.2R2 verification requires main, observed {branch!r}")
    started = time.perf_counter()
    report = repo/"reports/phase4"
    checks = [{"number": i+1, "name": name, "status": "not_run"} for i, name in enumerate(INVARIANTS)]
    out = {"stage": "P4.2R2", "status": "fail", "branch": branch, "checks": checks,
        "training": {}, "qc_reuse": {}, "shared_population": {}, "calibration": {}, "admission": {},
        "recommendation_model_fits": 0, "final_week": "not_run"}
    pending = []
    def passed(*numbers):
        for number in numbers:
            checks[number-1]["status"] = "pass"
    passed(2)
    try:
        c = read_json(report/"P4_2R2_EXPERIMENT_CONTRACT.json")
        m = read_json(report/"P4_2R2_metrics.json")
        previous_c = read_json(report/"P4_2R_EXPERIMENT_CONTRACT.json")
        prior = read_json(Path(c["prior_P4_2R_metrics"]["path"]))
        prior_root = Path(c["prior_run_root"])
        root = repo/"artifacts/phase4"/c["run_id"]
        out["run_id"], out["experiment_status"] = c["run_id"], m["decision"]
        pending = [3]
        assert m["final_week"] == c["final_week"] == "not_run"
        assert not m["Warm_v2_integrated"] and not m["P4_3_started"]
        for forbidden in ("2020-09-16", "2020-09-17", "2020-09-10"):
            try:
                guard_cutoff(forbidden)
            except ValueError:
                continue
            raise AssertionError("final-week/overlap accepted")
        for cutoff in c["inputs"]:
            guard_cutoff(cutoff)
        passed(3)
        pending = [1, 28]
        execution = read_json(root/"EXECUTION_START.json")
        assert c["status"] == "preregistered_before_formal_computation"
        assert datetime.fromisoformat(c["created_at_utc"]) < datetime.fromisoformat(execution["started_at_utc"])
        records = [execution["contract"], c["prior_P4_2R_metrics"]]
        for key in ("original_P4_2_manifest", "prior_P4_2R_manifest"):
            records.append(c[key])
            manifest = read_json(Path(c[key]["path"]))
            assert manifest["decision"] == "engineering_failure"
            records.extend(record for group in ("reports", "sources", "artifacts") for record in manifest[group])
        # The driver intentionally excludes downstream report/verification code
        # from the computational capture; any future capture must disclose it.
        closure = {"p42r2_verify.py", "p42r2_report.py"}
        records.extend(r for r in execution["implementation_before_formal"] if Path(r["path"]).name not in closure)
        out["postrun_only_sources_not_claimed_frozen_before_computation"] = [r["path"] for r in execution["implementation_before_formal"] if Path(r["path"]).name in closure]
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
        passed(1, 28)
        pending = [6, 11, 15, 16, 17, 18, 19, 20, 23]
        assert c["stage"] == "P4.2R2" and c["windows"] == WINDOWS and c["feature_spec"] == FEATURE_SPEC
        for key in ("population", "historical_pools", "excluded_cutoffs", "feature_spec", "epsilon", "utility", "taus", "pairing", "calibration", "metrics_contract", "gates", "selection"):
            assert c[key] == previous_c[key], key
        assert c["qW_repair"]["R1"] == previous_c["qW_repair"]["R1"]
        assert c["qW_repair"]["R2"]["log1p_columns"] == LOG1P_COUNTS
        assert c["qW_repair"]["R2"]["allowed_only_if"] == "unconditional for every formal qW chain"
        assert c["qW_repair"]["all_chains_repair"] == m["selected_qW_repair"] == "R2"
        assert {k: c["model"][k] for k in MODEL_PARAMS} == c["qW_repair"]["model_params"] == MODEL_PARAMS
        assert c["taus"] == TAUS and c["epsilon"] == 1e-6
        assert c["qC_reuse"]["new_fits"] == m["resources"]["qC_new_fits"] == 0
        assert read_json(report/"P4_2R_VERIFICATION.json")["status"] == "pass"
        for window in WINDOWS:
            source = prior["training"][window]["qC"]
            assert c["qC_reuse"]["windows"][window] == m["training"][window]["qC"] == source
            reuse = m["qc_reuse"][window]
            assert reuse["status"] == "reused_exact" and not reuse["new_fit"]
            assert reuse["source_stage"] == "P4.2R" and reuse["source_status"] == source["status"] == "converged"
            for key in ("model", "preprocessing", "training_cutoffs"):
                assert reuse[key] == source[key]
            out["qc_reuse"][window] = {"pass": True, "model_preprocessing_and_training_cohort_exact_prior_verified_assets": True, "new_fits": 0}
        passed(6, 11, 15, 16, 17, 18, 19, 20, 23)
        pending = [4, 5, 7]
        out["shared_population"] = _verify_shared_reuse(m["prepared"], prior["prepared"], prior_root)
        assert c["prepared_reuse"] == prior["prepared"]
        for window, pools in c["historical_pools"].items():
            assert pools["qC"] == pools["qW"]
            for cutoff in pools["qW"]:
                assert cutoff != "2019-11-27"
                assert datetime.fromisoformat(cutoff)+timedelta(days=7) < datetime.fromisoformat(WINDOWS[window])
                lineage = c["inputs"][cutoff]["b0_lineage"]
                assert lineage["available"] and lineage["model_label_end"] <= cutoff
        passed(4, 5, 7)
        pending = [8, 9, 10, 12, 13, 14]
        sequence = verify_global_sequence(m["attempts"], m["calibration"], m["windows"], m["decision"])
        out["attempt_sequence"] = sequence
        previous_warm = {a["window"]: a for a in prior["attempts"] if a["side"] == "qW"}
        for attempt in m["attempts"]:
            window = attempt["window"]
            assert attempt == m["training"][window]["qW"]
            assert attempt["training_cutoffs"] == c["historical_pools"][window]["qW"]
            assert attempt["latest_training_label_end"] < WINDOWS[window]
            assert datetime.fromisoformat(attempt["started_at_utc"]) >= datetime.fromisoformat(execution["started_at_utc"])
            assert attempt["train_population"] == "H_t" and attempt["preprocessing_training_only"]
            model, prep = [read_json(Path(attempt[key]["path"])) for key in ("model", "preprocessing")]
            assert model["params"] == MODEL_PARAMS and prep["repair"] == "R2"
            assert len(model["n_iter"]) == 1 and 0 <= model["n_iter"][0] <= 1000
            frames = [parquet(c["prepared_reuse"][cutoff]["features"]["qW"]["path"]) for cutoff in attempt["training_cutoffs"]]
            out["training"][window] = independent_qw_training(frames, prep, attempt, model)
            previous = previous_warm[window]
            assert previous["cleanup"] == attempt["cleanup"]
            assert (previous["training_rows"], previous["positives"]) == (attempt["training_rows"], attempt["positives"])
            assert prep["log1p_count_features_dropped_by_cleanup"] == [name for name in LOG1P_COUNTS if name not in prep["columns"]]
            comparison = m["numerical_comparison"][window]
            assert comparison["same_training_rows_labels_and_R1_cleanup"]
            old_model = read_json(Path(previous["model"]["path"]))
            for name, a, fitted_model in (("R1", previous, old_model), ("R2", attempt, model)):
                _close(comparison[name], {"iterations": fitted_model["n_iter"][0], "status": a["status"],
                    "dims": a["model_features"], **a["diagnostics"]})
            for key in ("iterations", "gradient_infinity_norm", "local_regularized_hessian_condition_number", "max_standardized_absolute_value"):
                a, b = comparison["R1"][key], comparison["R2"][key]
                _close(comparison["relative_change"][key], b/a-1 if a not in (None, 0) and b is not None else None)
            # Single count vector suffices for the specifically requested tail audit.
            count_frame = pd.DataFrame({"user_item_events_28d": np.concatenate([f.user_item_events_28d.to_numpy() for f in frames])})
            old_prep = read_json(Path(previous["preprocessing"]["path"]))
            for name, p in (("R1", old_prep), ("R2", prep)):
                _close(comparison["user_item_events_28d"][name], _tracked_count(count_frame, p), exact=True)
            del frames, count_frame
            gc.collect()
        if m["attempts"]:
            passed(8, 9, 10, 12, 13, 14)
        pending = [21]
        if sequence["all_four_converged"]:
            gate = read_json(root/"QW_CONVERGENCE_PASSED_BEFORE_CALIBRATION.json")
            assert gate == m["qW_convergence_gate"] and gate["pass"] and gate["converged"] == 4
            assert gate["checked_before_any_outer_prediction"]
            assert max(a["finished_at_utc"] for a in m["attempts"]) <= gate["checked_at_utc"]
            passed(21)
        else:
            assert not m["calibration"] and not m["windows"]
            assert not (root/"QW_CONVERGENCE_PASSED_BEFORE_CALIBRATION.json").exists()
            assert not list((root/"outer").glob("**/*-predictions.parquet"))
            out["convergence_stop_enforcement"] = "pass; all-four success criterion not met, outer computation not_run"
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
                np.testing.assert_allclose(q, frame.propensity, rtol=0, atol=0)
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
            pending = [22, 24, 25, 26, 27]
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
            passed(22, 24, 25, 26, 27)
            if set(derived) == set(WINDOWS):
                aggregate = independent_promotion(derived, c)
                _close(m, aggregate, exact=True)
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
    out["interpretation"] = "Verification pass validates artifacts and enforced boundaries, not model convergence or promotion success."
    write_json(report/"P4_2R2_VERIFICATION.json", out)
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    result = verify(args.repo)
    print({"status": result["status"], "summary": result["summary"], "seconds": result["seconds"], "error": result.get("error")})
    if result["status"] != "pass":
        raise SystemExit(2)
