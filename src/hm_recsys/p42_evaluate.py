"""P4.2 fixed-policy evaluation; labels never reach the matching algorithm."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import apk
from .p41a_stats import distribution, replacement_units
from .p42_contract import TAUS, guard_cutoff
from .p42_matching import (
    EPSILON, apply_admissions, clipped_logit, exact_matching, matching_diagnostics,
)


SEGMENTS = {
    "warm_21_plus": lambda n: n >= 21,
    "strict_cold": lambda n: n == 0,
    "sparse1_5": lambda n: (n >= 1) & (n <= 5),
    "all_cold_sparse": lambda n: n <= 5,
}
BUCKETS = ("0", "1", "2", "3", "4+")
EXECUTED_COLUMNS = [
    "variant", "tau", "user_index", "customer_id", "cold_row_index",
    "cold_article_id", "warm_article_id", "warm_slot_rank", "b0_cold_rank",
    "qC", "qW", "utility", "edge_weight", "inserted_positive",
    "removed_positive", "net_positive", "cold_only", "strict_cold",
    "sparse1_5", "cold_interaction_count", "warm_interaction_count",
    "removed_warm21_positive", "individual_delta_ap",
]


def _average(values):
    return float(np.mean(values)) if len(values) else None


def _segment_truth(data):
    output = {}
    truth = data["truth"]
    for name, predicate in SEGMENTS.items():
        segment = truth.loc[predicate(truth.interaction_count_before_cutoff)]
        truthsets = {u: set(g.article_id) for u, g in segment.groupby("customer_id", sort=False)}
        indices = np.array([i for i, u in enumerate(data["users"]) if u in truthsets], dtype=np.int64)
        output[name] = (truthsets, indices, len(segment))
    return output


def _segment_metrics(data, lists, segments, baseline=None):
    result = {}
    for name, (truthsets, indices, pairs) in segments.items():
        values = [apk(list(truthsets[data["users"][i]]), list(lists[i])) for i in indices]
        mean = _average(values)
        reference = baseline[name]["map12"] if baseline is not None else mean
        result[name] = {"map12": mean, "delta_vs_w0": mean - reference if mean is not None else None,
                        "truth_users": len(indices), "truth_pairs": pairs}
    return result


def _empty_matching():
    return {"edges_above_tau": 0, "matched_edges": 0,
            "same_warm_conflict_count": 0, "same_cold_conflict_count": 0,
            "greedy_would_differ_count": 0, "matched_weight_total": 0.,
            "matched_utility_total": 0., "exact_minus_greedy_weight_diagnostic_only": 0.}


def evaluate_window(data, cold_q, warm_q):
    """Evaluate only W0 and the three frozen thresholds on one outer window.

    q arrays align with data['cold'] and data['warm'] original row positions.
    The returned utility matrix has one row per eligible Cold item and twelve
    original Warm slots, so its full pair identity is reconstructible without
    materializing repeated user/article strings for all 600 pairs per user.
    """
    cutoff = guard_cutoff(data["cutoff"])
    users, cold, warm = data["users"], data["cold"], data["warm"]
    n = len(users)
    cold_q, warm_q = np.asarray(cold_q, float), np.asarray(warm_q, float)
    if cold_q.shape != (len(cold),) or warm_q.shape != (len(warm),):
        raise ValueError("propensity row count does not match frozen candidate identity")
    # Validate probabilities before clipping so malformed model outputs fail.
    clipped_logit(cold_q)
    clipped_logit(warm_q)
    cq, wq = np.clip(cold_q, EPSILON, 1-EPSILON), np.clip(warm_q, EPSILON, 1-EPSILON)
    original = np.asarray(data["warm_lists"], dtype=object)
    if original.shape != (n, 12) or len(warm) != n*12:
        raise ValueError("W0 must contain complete Top12 lists for all users")
    index = {u: i for i, u in enumerate(users)}
    if not np.array_equal(warm.customer_id.to_numpy().reshape(n, 12)[:, 0], users):
        raise ValueError("Warm propensity rows are not in canonical user order")
    if not np.array_equal(warm.article_id.to_numpy().reshape(n, 12), original):
        raise ValueError("Warm propensity article order drift")
    expected_rank = np.tile(np.arange(1, 13), n)
    np.testing.assert_array_equal(warm.warm_rank, expected_rank)
    candidates = cold.copy()
    candidates["cold_row_index"] = np.arange(len(cold), dtype=np.int64)
    candidates["user_index"] = candidates.customer_id.map(index)
    if candidates.user_index.isna().any():
        raise ValueError("outer Cold candidates contain users outside the complete W0 population")
    candidates["user_index"] = candidates.user_index.astype(np.int32)
    warmsets = {u: set(items) for u, items in zip(users, original)}
    overlap = np.array([i in warmsets[u] for u, i in zip(candidates.customer_id, candidates.article_id)])
    candidates["qC"] = cq
    eligible = candidates.loc[~overlap].sort_values(["user_index", "cold_rank", "article_id"], ignore_index=True)
    sizes = eligible.groupby("user_index", sort=False).size()
    if sizes.gt(50).any() or eligible.duplicated(["customer_id", "article_id"]).any():
        raise ValueError("Cold50 candidate budget or uniqueness changed")
    utility = clipped_logit(eligible.qC.to_numpy())[:, None] - clipped_logit(wq.reshape(n, 12))[eligible.user_index.to_numpy()]
    recommendations = {"W0": original.copy(), **{v: original.copy() for v in TAUS}}
    matching = {v: _empty_matching() for v in TAUS}
    executed_rows = []
    positive_warm = warm.target.to_numpy(dtype=np.int64).reshape(n, 12)
    warm_count = warm.interaction_count_before_cutoff.to_numpy(dtype=np.int64).reshape(n, 12)
    warm_probs = wq.reshape(n, 12)
    for ui, group in eligible.groupby("user_index", sort=False):
        ui = int(ui)
        ids = group.index.to_numpy()
        matrix = utility[ids]
        if matrix.size > 600:
            raise ValueError("more than 600 enumerated pairs per user")
        articles = group.article_id.tolist()
        rows = group.to_dict("records")
        for variant, tau in TAUS.items():
            matches = exact_matching(matrix, tau)
            diag = matching_diagnostics(matrix, tau, matches)
            summary = matching[variant]
            for key in summary:
                if key == "greedy_would_differ_count":
                    summary[key] += int(diag["greedy_would_differ"])
                else:
                    summary[key] += diag[key]
            recommendations[variant][ui] = apply_admissions(original[ui], articles, matches)
            for ci, wi in matches:
                row = rows[ci]
                ip, rp = int(row["target"]), int(positive_warm[ui, wi])
                count = int(row["interaction_count_before_cutoff"])
                delta = float(replacement_units(positive_warm[ui], ip, wi+1) / data["denominator"][ui])
                executed_rows.append({
                    "variant": variant, "tau": tau, "user_index": ui, "customer_id": users[ui],
                    "cold_row_index": int(row["cold_row_index"]), "cold_article_id": row["article_id"],
                    "warm_article_id": original[ui, wi], "warm_slot_rank": wi+1,
                    "b0_cold_rank": int(row["cold_rank"]), "qC": float(row["qC"]),
                    "qW": float(warm_probs[ui, wi]), "utility": float(matrix[ci, wi]),
                    "edge_weight": float(matrix[ci, wi]-tau), "inserted_positive": ip,
                    "removed_positive": rp, "net_positive": ip-rp,
                    "cold_only": row["source_branch"] == "cold_only", "strict_cold": count == 0,
                    "sparse1_5": 1 <= count <= 5, "cold_interaction_count": count,
                    "warm_interaction_count": int(warm_count[ui, wi]),
                    "removed_warm21_positive": int(rp and warm_count[ui, wi] >= 21),
                    "individual_delta_ap": delta,
                })
    executed = pd.DataFrame(executed_rows, columns=EXECUTED_COLUMNS)
    variants, user_audits = {}, []
    segments = _segment_truth(data)
    baseline_segments = _segment_metrics(data, original, segments)
    baseline = np.asarray(data["baseline_ap"])
    for variant, lists in recommendations.items():
        if any(len(set(items)) != 12 for items in lists):
            raise ValueError("final list is not exactly twelve unique articles")
        ap = np.array([apk(list(data["truthsets"][u]), list(items)) for u, items in zip(users, lists)])
        if variant == "W0":
            np.testing.assert_array_equal(ap, baseline)
        records = executed.loc[executed.variant.eq(variant)]
        counts = np.zeros(n, dtype=np.int64)
        inserted = np.zeros(n, dtype=np.int64)
        removed = np.zeros(n, dtype=np.int64)
        if len(records):
            idx = records.user_index.to_numpy(dtype=np.int64)
            counts = np.bincount(idx, minlength=n)
            inserted = np.bincount(idx, weights=records.inserted_positive, minlength=n).astype(np.int64)
            removed = np.bincount(idx, weights=records.removed_positive, minlength=n).astype(np.int64)
        changed = np.sum(lists != original, axis=1)
        np.testing.assert_array_equal(changed, counts)
        if np.any(counts > 12):
            raise ValueError("matching produced more than twelve admissions")
        np.testing.assert_array_equal(lists[counts == 0], original[counts == 0])
        admitted = counts > 0
        def total(column):
            return int(records[column].sum()) if len(records) else 0
        def positive_where(column):
            return int((records[column].astype(bool) & records.inserted_positive.eq(1)).sum()) if len(records) else 0
        admission = {
            "users_with_admission": int(admitted.sum()), "admission_user_share": float(admitted.mean()),
            "total_replacements": int(counts.sum()),
            "mean_replacements_per_admitted_user": float(counts[admitted].mean()) if admitted.any() else 0.,
            "max_replacements": int(counts.max()) if n else 0,
            "beneficial_replacements": int(records.net_positive.gt(0).sum()),
            "neutral_replacements": int(records.net_positive.eq(0).sum()),
            "harmful_replacements": int(records.net_positive.lt(0).sum()),
            "inserted_cold_positive_pairs": total("inserted_positive"),
            "removed_warm_positive_pairs": total("removed_positive"),
            "removed_warm21_positive_pairs": total("removed_warm21_positive"),
            "net_positive_pairs": total("net_positive"),
            "cold_only_positive_top12": positive_where("cold_only"),
            "cold_only_positive_actually_inserted": positive_where("cold_only"),
            "strict_cold_positive_insertions": positive_where("strict_cold"),
            "sparse1_5_positive_insertions": positive_where("sparse1_5"),
            "joint_beneficial_users": int(np.count_nonzero(ap > baseline)),
            "joint_harmful_users": int(np.count_nonzero(ap < baseline)),
            "joint_neutral_users": int(np.count_nonzero(ap == baseline)),
        }
        buckets = {}
        for name in BUCKETS:
            mask = counts >= 4 if name == "4+" else counts == int(name)
            avg, ref = _average(ap[mask]), _average(baseline[mask])
            buckets[name] = {"users": int(mask.sum()), "map12": avg, "w0_map12_same_users": ref,
                "delta_vs_same_users_w0": avg-ref if avg is not None else None,
                "map_contribution_full_window": float(ap[mask].sum()/n),
                "delta_contribution_full_window": float((ap[mask]-baseline[mask]).sum()/n),
                "inserted_cold_positives": int(inserted[mask].sum()),
                "removed_warm_positives": int(removed[mask].sum()),
                "net_positives": int((inserted[mask]-removed[mask]).sum())}
        variants[variant] = {"map12": float(ap.mean()), "delta_vs_w0": float(ap.mean()-baseline.mean()),
            "segments": _segment_metrics(data, lists, segments, baseline_segments),
            "candidate_truth_changes": {"inserted_positives": int(inserted.sum()), "removed_positives": int(removed.sum()), "net_positives": int((inserted-removed).sum())},
            "admission": admission, "admission_buckets": buckets}
        user_audits.append(pd.DataFrame({"variant": variant, "user_index": np.arange(n), "customer_id": users,
            "admissions": counts, "baseline_ap": baseline, "ap": ap, "delta": ap-baseline,
            "inserted_positive": inserted, "removed_positive": removed, "net_positive": inserted-removed}))
    for diag in matching.values():
        edges = diag["edges_above_tau"]
        diag["matching_efficiency"] = diag["matched_edges"]/edges if edges else 0.
    metrics = {"cutoff": cutoff, "users": n, "variants": variants,
        "pair_audit": {"users": n, "users_with_eligible_cold": len(sizes),
            "cold_nodes": len(eligible), "warm_nodes": n*12, "pair_rows": int(utility.size),
            "overlap_excluded_cold_rows": int(overlap.sum()), "complete_cold_rows": len(cold),
            "maximum_pairs_per_user": int(sizes.max()*12) if len(sizes) else 0,
            "utility_distribution": distribution(utility.ravel())},
        "matching": matching,
        "invariants": {"w0_parity": True, "only_three_fixed_taus": True, "no_greedy_primary": True,
            "no_hard_max1": True, "no_warm_internal_reordering": True, "exactly12_unique": True,
            "no_edge_exact_w0": True, "cold_w0_overlap_excluded": True,
            "full_truth_user_segment_denominators": True, "max600_pairs_per_user": True,
            "final_week_not_run": True}}
    return {"metrics": metrics, "recommendations": recommendations, "executed": executed,
            "utility": utility, "eligible_cold": eligible, "user_audit": pd.concat(user_audits, ignore_index=True)}


def aggregate_windows(windowmetrics, calibrations, contract):
    """Apply preregistered gates/decision ordering after exactly four windows."""
    if set(windowmetrics) != set(contract["windows"]):
        raise ValueError("promotion requires every preregistered outer window exactly once")
    windows = [windowmetrics[w] for w in contract["windows"]]
    g = contract["gates"]
    summary = {}
    for variant in ("W0", *TAUS):
        rows = [w["variants"][variant] for w in windows]
        delta = np.array([r["delta_vs_w0"] for r in rows])
        mean_map = float(np.mean([r["map12"] for r in rows]))
        mean_baseline = float(np.mean([w["variants"]["W0"]["map12"] for w in windows]))
        segment_summary = {}
        for segment in SEGMENTS:
            srows = [r["segments"][segment] for r in rows]
            if any(s["map12"] is None for s in srows):
                raise ValueError("a required segment has no outer truth-user support")
            ds = np.array([s["delta_vs_w0"] for s in srows])
            segment_mean = float(np.mean([s["map12"] for s in srows]))
            segment_baseline = float(np.mean([w["variants"]["W0"]["segments"][segment]["map12"] for w in windows]))
            segment_summary[segment] = {"mean_map12": segment_mean,
                "mean_delta": segment_mean-segment_baseline, "nondegrade_windows": int((ds >= 0).sum()),
                "worst_delta": float(ds.min()), "deltas": ds.tolist()}
        admission = {key: sum(r["admission"][key] for r in rows) for key in (
            "users_with_admission", "total_replacements", "inserted_cold_positive_pairs",
            "removed_warm_positive_pairs", "removed_warm21_positive_pairs", "net_positive_pairs",
            "cold_only_positive_actually_inserted", "strict_cold_positive_insertions", "sparse1_5_positive_insertions")}
        all_users = sum(w["users"] for w in windows)
        admission["admission_user_share"] = admission["users_with_admission"]/all_users
        admission["admission_count_user_distribution"] = {b: sum(r["admission_buckets"][b]["users"] for r in rows) for b in BUCKETS}
        warm = segment_summary["warm_21_plus"]
        cold = segment_summary["all_cold_sparse"]
        cold_windows = sum(r["admission"]["cold_only_positive_actually_inserted"] > 0 for r in rows)
        checks = {
            "overall_mean": mean_map-mean_baseline >= g["overall_mean_delta_min"],
            "overall_nondegrade": int((delta >= 0).sum()) >= g["overall_nondegrade_min"],
            "overall_worst": float(delta.min()) >= g["overall_worst_delta_min"],
            "warm_mean": warm["mean_delta"] >= g["warm_mean_delta_min"],
            "warm_protected_windows": sum(d >= g["warm_window_delta_min"] for d in warm["deltas"]) >= g["warm_protected_windows_min"],
            "cold_mean": cold["mean_delta"] > 0,
            "cold_nondegrade": cold["nondegrade_windows"] >= g["cold_nondegrade_min"],
            "cold_only_positive_windows": cold_windows >= g["cold_only_positive_windows_min"],
            "replacement_efficiency": admission["inserted_cold_positive_pairs"] > admission["removed_warm_positive_pairs"],
        }
        gates = {"checks": checks, "overall": all(checks[k] for k in ("overall_mean", "overall_nondegrade", "overall_worst")),
                 "warm": checks["warm_mean"] and checks["warm_protected_windows"],
                 "cold": all(checks[k] for k in ("cold_mean", "cold_nondegrade", "cold_only_positive_windows")),
                 "replacement_efficiency": checks["replacement_efficiency"], "all_pass": all(checks.values())}
        summary[variant] = {"mean_map12": mean_map,
            "mean_delta": mean_map-mean_baseline, "nondegrade_windows": int((delta >= 0).sum()),
            "worst_delta": float(delta.min()), "segments": segment_summary,
            "admission_pooled": admission, "cold_only_positive_windows": cold_windows, "gates": gates}
    severe = {}
    for side in ("qC", "qW"):
        rows = [calibrations[w][side] for w in contract["windows"]]
        severe[side] = {
            "extreme_rate_ratio_windows": sum(r["predicted_to_observed_rate_ratio"] is not None and
                (r["predicted_to_observed_rate_ratio"] < .1 or r["predicted_to_observed_rate_ratio"] > 10) for r in rows),
            "roc_not_above_chance_windows": sum(r["roc_auc"] is not None and r["roc_auc"] <= .5 for r in rows),
            "constant_prediction_windows": sum("constant_prediction" in r.get("warnings", []) for r in rows),
        }
    severe_failure = any(count >= 3 for side in severe.values() for count in side.values())
    passing = [v for v in TAUS if summary[v]["gates"]["all_pass"]]
    tolerance = contract["selection"]["near_equal_abs_tolerance"]
    selected = "W0"
    if passing:
        best_cold = max(summary[v]["segments"]["all_cold_sparse"]["mean_map12"] for v in passing)
        finalists = [v for v in passing if abs(summary[v]["segments"]["all_cold_sparse"]["mean_map12"]-best_cold) <= tolerance]
        best_overall = max(summary[v]["mean_map12"] for v in finalists)
        finalists = [v for v in finalists if abs(summary[v]["mean_map12"]-best_overall) <= tolerance]
        selected = max(finalists, key=lambda v: TAUS[v])
        decision = {"A_tau0": "promote_p4_2_aggressive", "M_tau_ln2": "promote_p4_2_moderate", "C_tau_ln4": "promote_p4_2_conservative"}[selected]
    elif severe_failure:
        decision = "propensity_calibration_failure"
    else:
        cs = [summary[v]["segments"]["all_cold_sparse"]["mean_map12"] for v in TAUS]
        wd = [-summary[v]["segments"]["warm_21_plus"]["mean_delta"] for v in TAUS]
        monotone = lambda x: x[0] >= x[1]-tolerance and x[1] >= x[2]-tolerance and x[0] > x[2]+tolerance
        if monotone(cs) and monotone(wd) and summary["A_tau0"]["segments"]["all_cold_sparse"]["mean_delta"] > 0:
            decision = "pareto_frontier_supported_but_no_safe_operating_point"
        elif any(summary[v]["segments"]["all_cold_sparse"]["mean_delta"] > 0 and
                 not (summary[v]["gates"]["overall"] and summary[v]["gates"]["warm"]) for v in TAUS):
            decision = "warm_risk_uncontrolled"
        else:
            decision = "cold_gain_not_recovered"
    reasons = (["selected among thresholds passing every preregistered gate"] if passing else
        [f"{v} failed: " + ", ".join(k for k, passed in summary[v]["gates"]["checks"].items() if not passed) for v in TAUS])
    if severe_failure:
        reasons.append("preregistered severe propensity-calibration diagnostic triggered; applied only after passing-tau priority")
    decision_record = {"machine_decision": decision, "selected_variant": selected,
                       "promoted": bool(passing), "reasons": reasons}
    return {"summary": summary, "decision": decision, "decision_detail": decision_record, "selected_variant": selected,
            "passing_variants": passing, "severe_calibration_counts": severe,
            "severe_calibration_failure": severe_failure, "final_week": "not_run"}
