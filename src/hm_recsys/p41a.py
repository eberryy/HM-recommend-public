"""P4.1A audit CLI. No training, model loading/inference, or threshold selection."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .p41a_contract import DIRECTIONS, RUN_ID, WINDOWS, WITHIN, identity, preregister, read_json, write_json
from .p41a_data import load_window, save_frame, verify_inputs
from .p41a_oracle import opportunities, oracle
from .p41a_stats import auc_metrics, distribution, reliability


def confidence_audit(data, opp):
    cold = data["cold"]
    masks = {"all": np.ones(len(cold), bool), "truth": cold.target.eq(1), "unobserved": cold.target.eq(0),
             "strict_cold": cold.strict_cold_flag.eq(1), "sparse1_5": cold.sparse1_5_flag.eq(1)}
    distributions = {var: {group: distribution(cold.loc[mask, var]) for group, mask in masks.items()}
                     for var in ("b0_score", "b0_delta_vs_m4", *WITHIN)}
    for var, groups in distributions.items():
        groups["truth_unobserved_gap"] = {key: groups["truth"][key]-groups["unobserved"][key]
            if groups["truth"][key] is not None and groups["unobserved"][key] is not None else None
            for key in ("mean", "median")}
    separability, bins = {}, {}
    for var, sign in DIRECTIONS.items():
        score = sign*opp[var].to_numpy(dtype=float)
        separability[var] = {"direction": sign, **auc_metrics(score, opp.beneficial)}
        bins[var] = {str(n): reliability(score, opp.beneficial, n) for n in (4, 10)}
    return {"cold50_distributions": distributions,
            "beneficial_primary_raw_score": distribution(opp.loc[opp.beneficial, "b0_score"])}, separability, bins


def risk_audit(data, opp, choices):
    selected = choices.loc[(choices.cell == "top10_slots10_12") & choices.admitted]
    out = {}
    for j in (10, 11, 12):
        warm = data["warm"].loc[data["warm"].warm_rank == j]
        rows = opp.loc[opp.warm_slot_rank == j]
        out[str(j)] = {"slot_rows": len(warm), "slot_future_positive_rows": int(warm.target.sum()),
            "slot_truth_rate": float(warm.target.mean()), "legal_opportunity_rows": len(rows),
            "beneficial_replacement_rows": int(rows.beneficial.sum()),
            "harmful_replacement_rows": int(rows.opportunity_label.eq("harmful").sum()),
            "neutral_rows": int(rows.opportunity_label.eq("neutral").sum()),
            "oracle_selected_replacement_count": int(selected.replaced_warm_rank.eq(j).sum())}
    users = data["userframe"].copy()
    users["has_proposal"] = users.customer_id.isin(opp.customer_id.unique())
    users["has_beneficial"] = users.customer_id.isin(opp.loc[opp.beneficial, "customer_id"].unique())
    assert int(users.has_beneficial.sum()) == len(selected)
    strata = {"all": np.ones(len(users), bool), "recent_active": users.recent_active,
        "inactive_recent": ~users.recent_active, "profile_available": users.profile_available,
        "profile_unavailable": ~users.profile_available}
    state = {}
    for name, mask in strata.items():
        group = users.loc[mask]
        n, p, b = len(group), int(group.has_proposal.sum()), int(group.has_beneficial.sum())
        state[name] = {"users": n, "users_with_proposal": p, "users_with_beneficial_replacement": b,
            "beneficial_opportunity_share_all_stratum_users": b/n if n else None,
            "beneficial_opportunity_share_proposal_users": b/p if p else None,
            "proposal_user_share": p/n if n else None}
    return out, state


def portability(separability, bins):
    out = {}
    for var in DIRECTIONS:
        auc = {w: separability[w][var]["roc_auc"] for w in WINDOWS}
        q = {w: bins[w][var]["4"]["bins"][-1]["lift"] for w in WINDOWS}
        d = {w: bins[w][var]["10"]["bins"][-1]["lift"] for w in WINDOWS}
        gt = lambda values, threshold: sum(v is not None and v > threshold for v in values.values())
        lt = lambda values, threshold: sum(v is not None and v < threshold for v in values.values())
        out[var] = {"auc_by_window": auc, "top_quartile_lift_by_window": q, "top_decile_lift_by_window": d,
            "auc_gt_055_windows": gt(auc, .55), "auc_gt_060_windows": gt(auc, .60),
            "auc_lt_045_windows": lt(auc, .45), "auc_gt_050_windows": gt(auc, .5),
            "top_quartile_lift_gt_1_windows": gt(q, 1), "top_quartile_lift_lt_1_windows": lt(q, 1),
            "top_decile_lift_gt_1_windows": gt(d, 1),
            "direction_consistency": "all4_positive" if gt(auc, .5) == 4 else "all4_negative" if lt(auc, .5) == 4 else "mixed_or_missing"}
    return out


def drift_summary(audits):
    result = {}
    for var in ("b0_score", "b0_delta_vs_m4", *WITHIN):
        stats = {w: a["cold50_distributions"][var]["all"] for w, a in audits.items()}
        result[var] = {"by_window": stats,
            "cross_window_ranges": {key: distribution([d[key] if d[key] is not None else np.nan for d in stats.values()])
                                    for key in ("median", "iqr", "mean", "std")},
            "truth_unobserved_gaps": {w: a["cold50_distributions"][var]["truth_unobserved_gap"] for w, a in audits.items()}}
    raw = result["b0_score"]["by_window"]
    med = [a["median"] for a in raw.values()]
    iqr = [a["iqr"] for a in raw.values()]
    good = [a["beneficial_primary_raw_score"] for a in audits.values()]
    ratio = lambda num, den, bothzero: float(num/den) if den > 0 else (bothzero if num == 0 else 1e30)
    finite = all(v is not None for v in med+iqr+[g["median"] for g in good]+[g["iqr"] for g in good])
    diagnostics = {"all4_finite": finite, "raw_median_span_over_median_iqr": None,
        "raw_max_iqr_over_min_iqr": None, "beneficial_median_span_over_median_iqr": None}
    if finite:
        diagnostics.update(raw_median_span_over_median_iqr=ratio(max(med)-min(med), float(np.median(iqr)), 0),
            raw_max_iqr_over_min_iqr=ratio(max(iqr), min(iqr), 1),
            beneficial_median_span_over_median_iqr=ratio(max(g["median"] for g in good)-min(g["median"] for g in good), float(np.median([g["iqr"] for g in good])), 0))
    return result, diagnostics


def verdicts(primary, portable, raw):
    delta = [p["delta_MAP_vs_W0"] for p in primary.values()]
    assert min(delta) >= 0
    mean = float(np.mean(delta))
    headroom = "supported" if mean >= .0003 and sum(x > 0 for x in delta) >= 3 else "weak" if mean > 0 else "rejected"
    good = [v for v, p in portable.items() if p["auc_gt_055_windows"] >= 3]
    within_good = [v for v in WITHIN if v in good]
    sep = "supported" if len(good) >= 2 and within_good else "weak" if good or sum(p["auc_gt_055_windows"] >= 2 for p in portable.values()) >= 2 else "rejected"
    within = "supported" if any(portable[v]["auc_gt_055_windows"] >= 3 and portable[v]["top_quartile_lift_gt_1_windows"] >= 3 for v in WITHIN) else "rejected" if any(portable[v]["auc_lt_045_windows"] >= 3 and portable[v]["top_quartile_lift_lt_1_windows"] >= 3 for v in WITHIN) else "inconclusive"
    raw_deciles = portable["b0_score"]["top_decile_lift_gt_1_windows"]
    calibration = "inconclusive"
    if raw["all4_finite"]:
        location = raw["raw_median_span_over_median_iqr"]
        scale = raw["raw_max_iqr_over_min_iqr"]
        beneficial = raw["beneficial_median_span_over_median_iqr"]
        if (location > 1 or scale > 2) and beneficial > 1 and 1 <= raw_deciles <= 3 and within == "supported" and any(portable[v]["top_decile_lift_gt_1_windows"] > raw_deciles for v in WITHIN):
            calibration = "unstable"
        elif location <= .5 and scale <= 1.5 and beneficial <= .5 and raw_deciles == 4:
            calibration = "stable"
    return {"constrained_admission_headroom": headroom, "raw_b0_absolute_calibration": calibration,
        "within_user_confidence_portability": within, "admission_signal_separability": sep,
        "p4_1b_allowed": headroom == "supported" and sep == "supported",
        "qualifying_auc_variables": good, "qualifying_within_variables": within_good,
        "final_week": "not_run", "promoted_baseline": "W0", "P4_1B_started": False}


def peak_working_set():
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(k, ctypes.c_size_t) for k in
            ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counter), counter.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counter.PeakWorkingSetSize)


def build_manifest(repo, metrics):
    report = repo / "reports/phase4"
    source = [identity(p) for p in sorted((repo / "src/hm_recsys").glob("p41a*.py"))]
    source += [identity(p) for p in sorted((repo / "tests").glob("test_p41a*.py"))]
    reports = {}
    for path in sorted(report.glob("*")):
        if "p4_1a" in path.name.lower() and path.name != "P4_1A_OUTPUT_MANIFEST.json":
            reports[path.name] = {**identity(path), "row_count": 1, "row_unit": "report_document",
                "cutoff": list(WINDOWS.values()), "source_lineage": metrics["contract_identity"]}
    manifest = {"stage": "P4.1A", "run_id": RUN_ID, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reports": reports, "artifacts": metrics["artifacts"], "audit_source": source,
        "input_contract": metrics["contract_identity"], "final_week": "not_run"}
    write_json(report / "P4_1A_OUTPUT_MANIFEST.json", manifest)
    return manifest


def run(repo, source):
    start = time.perf_counter()
    report = repo / "reports/phase4"
    if (report / "P4_1A_metrics.json").exists():
        raise RuntimeError("P4.1A measured evidence already exists; use verify, do not overwrite")
    contract = preregister(repo, source)
    contract_id = identity(report / "P4_1A_EXPERIMENT_CONTRACT.json")
    inputs_before = verify_inputs(contract)
    primary, frontier, score, separability, bins, risk, state, parity, artifacts = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for window, cutoff in WINDOWS.items():
        print(f"P4.1A {window}: frozen-asset and bounded truth audit", flush=True)
        window_start = time.perf_counter()
        data = load_window(contract, window)
        opp = opportunities(data)
        p, f, choices, lists = oracle(data)
        s, a, b = confidence_audit(data, opp)
        r, u = risk_audit(data, opp, choices)
        primary[window], frontier[window], score[window], separability[window], bins[window] = p, f, s, a, b
        risk[window], state[window], parity[window] = r, u, data["parity"]
        p["opportunity_rows"] = len(opp)
        p["beneficial_rows"] = int(opp.beneficial.sum())
        p["harmful_rows"] = int(opp.opportunity_label.eq("harmful").sum())
        p["neutral_rows"] = int(opp.opportunity_label.eq("neutral").sum())
        p["opportunity_base_rate"] = p["beneficial_rows"]/len(opp) if len(opp) else None
        dest = repo / "artifacts/phase4" / RUN_ID / cutoff
        lineage = {"contract": contract_id, "frozen_window": window,
            "warm": contract["frozen_inputs"][window]["w0_database"], "cold": contract["frozen_inputs"][window]["cold50"],
            "b0_model_label_end": contract["frozen_inputs"][window]["b0_lineage"].get("model_label_end"),
            "no_model_inference": True, "outer_truth_audit_only": True}
        saved = {}
        for key, frame in (("opportunities", opp), ("oracle_selections", choices), ("cold50_confidence", data["cold"]), ("w0_top12", data["warm"]), ("truth", data["truth"])):
            saved[key] = save_frame(frame, dest / f"{key}.parquet", cutoff, lineage)
        npz = dest / "oracle_top12_lists.npz"
        np.savez_compressed(npz, users=data["users"].astype("U64"), w0=data["warm_lists"].astype(np.int64), **lists)
        saved["oracle_top12_lists"] = {**identity(npz), "row_count": len(data["users"])*16,
            "row_unit": "user-policy Top12 lists; 15 cells plus W0", "cutoff": cutoff, "source_lineage": lineage}
        artifacts[window] = saved
        parity[window]["seconds"] = time.perf_counter()-window_start
        print(f"P4.1A {window}: {len(opp)} opportunities; invariants pass", flush=True)
    portable = portability(separability, bins)
    drift, rawdiag = drift_summary(score)
    decision = verdicts(primary, portable, rawdiag)
    summary = {"w0_mean_map@12": float(np.mean([p["w0_map@12"] for p in primary.values()])),
        "primary_mean_map@12": float(np.mean([p["map@12"] for p in primary.values()])),
        "primary_mean_delta": float(np.mean([p["delta_MAP_vs_W0"] for p in primary.values()])),
        "positive_windows": sum(p["delta_MAP_vs_W0"] > 0 for p in primary.values()),
        "nonnegative_windows": sum(p["delta_MAP_vs_W0"] >= 0 for p in primary.values()),
        "segment_mean_overall_delta_contributions": {s: float(np.mean([p["segments"][s]["overall_delta_contribution"] for p in primary.values()])) for s in ("strict_cold", "sparse1_5", "all_cold_sparse")},
        "selected_rank_pooled_counts": {str(j): sum(p["selected_cold_rank_counts"].get(str(j), 0) for p in primary.values()) for j in range(1, 11)},
        "replaced_slot_pooled_counts": {str(j): sum(p["replaced_warm_rank_counts"].get(str(j), 0) for p in primary.values()) for j in (10, 11, 12)}}
    inputs_after = verify_inputs(contract)
    forbidden = sorted(k for k in sys.modules if k.split(".")[0] in {"torch", "lightgbm", "sklearn", "tensorflow"})
    assert not forbidden, forbidden
    metrics = {"stage": "P4.1A", "run_id": RUN_ID, "status": "measured_pending_independent_verification",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "contract_identity": contract_id,
        "summary": summary, "decision": decision, "windows": primary, "portability": portable,
        "raw_calibration_diagnostics": rawdiag, "parity": parity, "artifacts": artifacts,
        "execution": {"seconds": time.perf_counter()-start, "peak_working_set_bytes": peak_working_set(),
            "input_hash_before": inputs_before, "input_hash_after": inputs_after,
            "training": False, "model_inference": False, "forbidden_model_modules_loaded": forbidden,
            "gpu_used": False, "final_week": "not_run", "P4_1B_started": False}, "final_week": "not_run"}
    for name, value in {
        "p4_1a_primary_oracle.json": {"windows": primary, "summary": summary},
        "p4_1a_oracle_frontier.json": {"windows": frontier, "primary_fixed": "top10_slots10_12", "selection_from_frontier": False},
        "p4_1a_score_drift_audit.json": {"windows": score, "cross_window": drift, "raw_diagnostics": rawdiag},
        "p4_1a_admission_separability.json": {"windows": separability, "portability": portable, "decision": decision},
        "p4_1a_reliability_audit.json": {"windows": bins, "no_deployable_thresholds": True},
        "p4_1a_warm_risk_audit.json": {"windows": risk},
        "p4_1a_user_state_audit.json": {"windows": state},
        "P4_1A_metrics.json": metrics,
    }.items():
        write_json(report / name, {"stage": "P4.1A", "run_id": RUN_ID, "final_week": "not_run", **value})
    build_manifest(repo, metrics)
    print("P4.1A measured; independent verification required", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preregister", "run", "verify", "report"])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    if args.command == "preregister":
        preregister(args.repo_root, args.source_root)
    elif args.command == "run":
        run(args.repo_root, args.source_root)
    elif args.command == "verify":
        from .p41a_verify import verify
        print(verify(args.repo_root))
    else:
        from .p41a_report import render
        render(args.repo_root)


if __name__ == "__main__":
    main()
