"""P4.2R preregistration; the original P4.2 evidence is immutable."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from .p41a_contract import identity, read_json, write_json
from .p42_contract import branch_guard, history_cutoffs, WINDOWS, FEATURE_SPEC, TAUS
from .p42_propensity import MODEL_PARAMS

RUN_ID = "p4-2r-v1-population-numerical-repair"
LOG1P_COUNTS = [
    "source_count", "item2vec_seed_support", "item2vec_vocab_count",
    "user_history_events_12w", "user_unique_items_12w", "item_events_7d",
    "item_events_28d", "item_events_12w", "item_unique_customers_28d",
    "user_item_events_12w", "user_item_events_28d",
    "user_product_code_events_28d", "user_product_code_events_12w",
    "user_product_type_events_28d", "user_product_type_events_12w",
    "user_department_events_28d", "user_department_events_12w",
    "user_garment_events_28d", "user_garment_events_12w",
]
REPAIR_FAILURES = ["qC_population_repair_failure", "qW_R1_and_R2_convergence_failure",
                   "shared_population_contract_failure", "propensity_calibration_failure", "engineering_failure"]


def preregister(repo):
    repo = Path(repo).resolve()
    branch_guard(repo)
    report = repo / "reports/phase4"
    target = report / "P4_2R_EXPERIMENT_CONTRACT.json"
    if target.exists():
        c = read_json(target)
        if c["run_id"] != RUN_ID or c["feature_spec"] != FEATURE_SPEC or c["taus"] != TAUS:
            raise ValueError("P4.2R existing contract differs; never overwrite")
        return c
    if (report / "P4_2R_metrics.json").exists():
        raise ValueError("P4.2R results exist before preregistration")
    old = read_json(report / "P4_2_EXPERIMENT_CONTRACT.json")
    assert read_json(report / "P4_2_metrics.json")["decision"] == "engineering_failure"
    c = deepcopy(old)
    available = {t: a["b0_lineage"]["available"] for t, a in c["inputs"].items()}
    shared = {w: history_cutoffs(t, available) for w, t in WINDOWS.items()}
    required = ["docs/ROADMAP_PHASE4.zh-CN.md", "reports/phase4/P4_0_FINAL.md",
        "reports/phase4/P4_1A_FINAL.md", "reports/phase4/P4_2_FINAL.md",
        "reports/phase4/P4_2_EXPERIMENT_CONTRACT.json", "reports/phase4/P4_2_OUTPUT_MANIFEST.json",
        "reports/phase4/P4_2_TRAINING_POPULATION_NOTE.zh-CN.md",
        "reports/phase4/p4_2_training_population_audit.json",
        "reports/phase4/p4_2_convergence_diagnostic.json", "reports/phase3/P3_1_FINAL.md",
        "reports/phase3/P3_1_metrics.json", "reports/phase3/P3_7B_FINAL.md",
        "reports/phase3/P3_7B_EXPERIMENT_CONTRACT.json", "reports/phase3/P3_7B_OUTPUT_MANIFEST.json",
        "reports/phase4/P4_2R_PREFLIGHT.zh-CN.md", "reports/phase4/p4_2r_shared_population_audit.json"]
    c.update({
        "schema_version": "p4.2r-contract-v1", "stage": "P4.2R", "run_id": RUN_ID,
        "status": "preregistered_before_formal_computation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "authoritative_inputs": {s: identity(repo / s) for s in required},
        "original_P4_2_manifest": identity(report / "P4_2_OUTPUT_MANIFEST.json"),
        "original_P4_2_historical_decision": "engineering_failure",
        "old_prepared_root": str(repo / "artifacts/phase4" / old["run_id"] / "prepared"),
        "old_run_root": str(repo / "artifacts/phase4" / old["run_id"]),
        "historical_pools": {w: {"qC": ts, "qW": list(ts)} for w, ts in shared.items()},
        "excluded_cutoffs": [{"cutoff": t, "reason": "no_safe_B0", "excluded_sides": ["qC", "qW"]}
                             for t, safe in available.items() if not safe],
        "population": {
            "approval": "Lyra explicitly approved common history-eligible W0 fitted support, full W0 evaluation, and no-Cold exact abstention, then requested run P4.2R",
            "S_t": "complete frozen W0 user roster; independently check equality to next-7-day any-truth deterministic hash10pct roster",
            "H_t": "S_t intersect users with at least one mapped item among original M4 latest20 distinct pre-cutoff purchased items",
            "training_invariant": "distinct qC candidate-row users == distinct qW candidate-row users == H_t at every included cutoff",
            "outer_denominator": "full S_t, including S_t minus H_t; no Cold candidates means exact original W0 item/order",
            "future_cold_truth_conditioned_eligibility": False,
            "estimand": "next-week candidate truth propensity conditional on candidate policy, formal offline W0 population, and mapped pre-cutoff history availability; not unconditional online probability",
            "generation_sequence": ["load S_t", "pre-cutoff-only mapped latest20 history -> H_t",
                "label-blind M4 Top200 for missing H_t users", "latest eligible earlier frozen B0 -> Cold50",
                "persist unlabeled candidate identity", "join [t,t+7days) truth last"],
            "reuse": "existing same-policy candidates for overlapping H_t users may be reused with exact identity checks; never call the old cold-truth eligibility selector; all newly restored users generated regardless of future cold truth",
            "regeneration_replay_probe": "first32 lexicographically sorted overlapping H_t users per regenerated cutoff, label-blind; M4 Top200 and B0 Cold50 item/rank identity exact, continuous replay scores abs tolerance1e-5 for GPU batch-shape roundoff; no tolerance for rank/identity drift; original reused rows remain exact frozen bytes/values",
            "zero_candidate_users": "not fake negatives, not placeholder candidates; excluded from BOTH fitted populations only by approved history rule; retained in outer evaluation",
        },
        "qW_repair": {
            "R1": {"stage": "training-only finite/median-imputed logical input before StandardScaler",
                "drop": ["constant column", "exact rowwise duplicate column"],
                "survivor_order": ["non *_available before *_available", "lexicographically smaller name"],
                "output_order": "original input feature order restricted to survivors",
                "high_correlation_only_drop": False,
                "exact_columns": "deterministic function of training X only; concrete list saved per outer attempt; no y/coefficients/validation used"},
            "R2": {"allowed_only_if": "winter R1 has a convergence warning under unchanged budget",
                "log1p_columns": LOG1P_COUNTS,
                "pipeline": "validate all finite observed counts nonnegative -> training-only median -> R1 cleanup -> log1p surviving listed counts -> training-only numeric StandardScaler; binary/availability unchanged",
                "negative_finite_counts": "fail closed, no clipping; validate declared count columns even if R1 drops them"},
            "branch": "winter R1 convergence chooses R1 for every later chain; otherwise winter R2 once. If chosen later-chain repair fails, stop as engineering_failure; never unlock per-window alternatives",
            "winter_both_fail": "qW_R1_and_R2_convergence_failure",
            "penalty_boundary": "removing exact duplicates at fixed C changes effective L2 penalty; not mathematically equivalent to old model",
            "attribution_boundary": "training users and time pool also changed (Nov27 excluded); old-to-new conditioning comparisons cannot isolate cleanup as a unique causal mechanism",
            "model_params": MODEL_PARAMS,
        },
        "old_qC_control": {"window": "winter_20200122", "same_outer_cold50_rows": True,
            "model": str(repo / "artifacts/phase4" / old["run_id"] / "models/winter_20200122/qC-model.json"),
            "preprocessing": str(repo / "artifacts/phase4" / old["run_id"] / "models/winter_20200122/qC-preprocessing.json"),
            "other_windows": "not_available: old P4.2 stopped before fitting them; no new selected-population control fits",
            "used_for_utility": False},
        "budget": {"estimated_formal_minutes": [15, 45], "formal_soft_stop_seconds": 5400,
            "gpu": "local frozen M4/B0 inference only", "fitting": "CPU, 4 BLAS threads",
            "order": "lazy historical preparation, winter qC then qW R1/R2 first; stop before later expensive preparation if failed",
            "maximum_formal_propensity_fit_attempts": 9, "no_sweeps": True,
            "fallback": "W0; preserve all attempts; stop after required closure; no P4.3/Warm-v2/final-week"},
        "repair_failure_states": REPAIR_FAILURES,
    })
    c["feature_contract"]["stage"] = "P4.2R"
    c["model"]["fits"] = "one qC and selected-repair qW per outer, plus only winter R2 if winter R1 has a convergence warning; preserve every failed attempt and stop on all other fit failure"
    c["feature_contract"]["run_id"] = RUN_ID
    c["feature_contract"]["qC_population"] = "all B0 Cold50 rows for shared H_t; no sampling/weighting"
    c["feature_contract"]["qW_population"] = "all frozen W0 Top12 rows for the exact same H_t"
    c["feature_contract"]["preprocessing"] = "qC unchanged P4.2; qW deterministic R1/R2 contract below"
    c["calibration"].update({
        "population": "primary both sides H_t: full Cold50 and exact W0 Top12 for the same history-eligible user set; additional qW full-S_t diagnostic is not a different MAP denominator",
        "stage_order": "all four outer calibration records persisted and severe gate checked BEFORE any pair-utility/admission computation",
        "probability": c["population"]["estimand"],
        "severe_failure_action": "propensity_calibration_failure; admission not_run; no promotion interpretation",
        "full_S_t_qW": "additional diagnostic for completeness, not primary common-population calibration gate",
    })
    c["decisions"]["precedence"] = REPAIR_FAILURES + [
        "passing_tau_promotion", "pareto_frontier_supported_but_no_safe_operating_point",
        "warm_risk_uncontrolled", "cold_gain_not_recovered"]
    assert c["feature_spec"] == old["feature_spec"] == FEATURE_SPEC
    assert {k: c["model"][k] for k in MODEL_PARAMS} == MODEL_PARAMS
    for key in ("epsilon", "taus", "utility", "pairing", "gates", "selection", "metrics_contract"):
        assert c[key] == old[key], key
    write_json(target, c)
    return c


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    contract = preregister(args.repo)
    print({"status": contract["status"], "created_at_utc": contract["created_at_utc"],
           "historical_pools": contract["historical_pools"]})
