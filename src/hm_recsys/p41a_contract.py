"""P4.1A preregistration. Deliberately imports no training or model modules."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

RUN_ID = "p4-1a-v1-constrained-admission-oracle"
FINAL_CUTOFF = "2020-09-16"
WINDOWS = {
    "winter_20200122": "2020-01-22", "spring_20200318": "2020-03-18",
    "early_summer_20200624": "2020-06-24", "late_summer_20200819": "2020-08-19",
}
BUDGETS = (1, 5, 10, 20, 50)
SLOT_SETS = {"slot12_only": (12,), "slots10_12": (10, 11, 12), "slots1_12": tuple(range(1, 13))}
WITHIN = ("b0_user_percentile", "b0_user_zscore", "margin_to_rank2", "margin_to_rank5", "margin_to_user_median")
DIRECTIONS = {
    "b0_score": 1, "b0_rank": -1, "b0_rank_pct": -1,
    **{key: 1 for key in WITHIN}, "b0_delta_vs_m4": 1,
    "m4_coarse_rank": -1, "interaction_count_before_cutoff": 1,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def guard_cutoff(cutoff: str) -> str:
    parsed = date.fromisoformat(cutoff)
    if parsed >= date.fromisoformat(FINAL_CUTOFF) or parsed + timedelta(days=7) > date.fromisoformat(FINAL_CUTOFF):
        raise ValueError("P4.1A rejects final-week cutoff or truth overlap")
    return parsed.isoformat()


def check_identity(expected: dict[str, Any]) -> None:
    observed = identity(Path(expected["path"]))
    if any(observed[key] != expected[key] for key in ("bytes", "sha256")):
        raise ValueError(f"input identity drift: {expected['path']}")


def preregister(repo: Path, source: Path) -> dict[str, Any]:
    report = repo / "reports" / "phase4"
    target = report / "P4_1A_EXPERIMENT_CONTRACT.json"
    if target.exists():
        contract = read_json(target)
        if contract["run_id"] != RUN_ID or contract["status"] != "preregistered_before_formal_computation":
            raise ValueError("existing P4.1A contract is not the frozen contract")
        return contract
    authority_paths = [repo / "docs" / "ROADMAP_PHASE4.zh-CN.md"]
    authority_paths += [report / name for name in (
        "P4_0_FINAL.md", "P4_0_metrics.json", "P4_0_EXPERIMENT_CONTRACT.json",
        "p4_0_upstream_lineage_audit.json", "p4_0_cold_truth_funnel.json",
        "p4_0_replacement_accounting.json", "p4_0_cross_source_calibration.json",
        "P4_0_OUTPUT_MANIFEST.json",
    )]
    authority_paths += [repo / "reports" / "phase3" / name for name in (
        "P3_7B_FINAL.md", "P3_7B_metrics.json", "P3_7B_EXPERIMENT_CONTRACT.json",
        "P3_7B_OUTPUT_MANIFEST.json", "P3_1_metrics.json",
    )]
    authority_paths += [repo / "reports" / "m5_4" / "M5_4_metrics.json"]
    p40 = read_json(report / "P4_0_metrics.json")
    p31 = read_json(repo / "reports" / "phase3" / "P3_1_metrics.json")
    m54 = read_json(repo / "reports" / "m5_4" / "M5_4_metrics.json")
    m3_path = source / "reports/m3_3/m3-3-v1-cross-season-adaptive-seasonal/metrics.json"
    m3 = read_json(m3_path)
    authority_paths.append(m3_path)
    proposals: dict[str, Any] = {}
    for window, cutoff in WINDOWS.items():
        guard_cutoff(cutoff)
        asset = p40["cutoffs"][cutoff]
        state_manifest_path = repo / f"artifacts/phase3/phase3-p3.7b-v1-time-aware-hybrid/features-v1/outer_validation/{window}/manifest.json"
        state_manifest = read_json(state_manifest_path)
        lineage = p40["audits"]["upstream_lineage"][cutoff]
        proposals[window] = {
            "cutoff": cutoff, "cold50": asset["cold"]["artifact"],
            "features": asset["features"]["artifact"], "warm150": asset["warm"]["artifact"],
            "b0_lineage": lineage, "warm_lineage": m54["cutoffs"][cutoff]["lineage"]["warm_score"],
            "w0_database": identity(Path(m3["development"][window]["scoring"]["evaluation_db"])),
            "w0_map": p40["windows"][window]["evaluations"]["W0"]["segments"]["overall"]["map@12"],
            "w0_users": p40["windows"][window]["evaluations"]["W0"]["segments"]["overall"]["truth_users"],
            "p31_assets": p31["all_assets"][f"outer_validation:{window}"]["artifacts"],
            "user_state_manifest": identity(state_manifest_path),
            "user_state": state_manifest["artifacts"]["user_state"],
            "user_state_columns": state_manifest["columns"]["user_state"],
        }
    contract = {
        "schema_version": "p4.1a-contract-v1", "stage": "P4.1A", "run_id": RUN_ID,
        "status": "preregistered_before_formal_computation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "source_sha": identity(Path(__file__)),
        "authoritative_inputs": {str(path): identity(path) for path in authority_paths},
        "frozen_inputs": proposals,
        "transactions": m3["inputs"]["transactions"],
        "catalog": identity(repo / "artifacts/m4/m4-v1-supervised-cold-representation/student-v1/static_catalog/catalog_items.csv"),
        "windows": WINDOWS, "reject_cutoff_on_or_after": FINAL_CUTOFF,
        "primary": {"name": "A_primary", "b0_max_rank": 10, "warm_slots": [10, 11, 12], "max_admissions": 1, "no_admission": True, "exclude": "items already in W0 Top12 only; do not exclude other Warm150 overlap"},
        "secondary": {"budgets": list(BUDGETS), "slot_sets": SLOT_SETS, "max_admissions": 1, "no_admission": True, "select_primary_from_frontier": False},
        "replacement": "delete Warm item at slot j; insert Cold at same j; all other eleven slots keep identical items and positions",
        "oracle": {
            "objective": "maximize each user's overall AP@12; nonpositive best -> exact W0",
            "tie_break": ["maximum exact delta", "lowest original B0 rank", "lowest warm slot", "article_id"],
            "sign": "integer AP numerator with LCM(1..12)=27720; no float tolerance for beneficial/neutral/harmful labels",
            "metric": "AP denominator min(12, distinct truth count); MAP mean on every frozen W0 truth user including no proposal/no positive opportunity",
            "segments": "report fixed chosen policy subgroup MAP on users with subgroup truth; also attribute overall delta by selected proposal strict_cold vs sparse1_5; segment attribution adds to total, subgroup MAP does not",
            "positive_opportunity_share_denominator": "all frozen W0 truth users; user state audit also reports proposal-user denominator",
            "warm_positive_removed": "removed positive from W0 branch, regardless of interaction count; separately report >=21-event removed positives",
            "distribution_denominators": "rank distributions over admitted users; delta distribution both all W0 users and admitted users",
        },
        "opportunity_labels": {"beneficial": "delta_AP12 > 0", "neutral": "delta_AP12 == 0", "harmful": "delta_AP12 < 0", "non_beneficial": "neutral or harmful", "audit_only": True},
        "confidence": {
            "directions": DIRECTIONS, "within_user_family": WITHIN,
            "raw_absolute_family": ["b0_score", "b0_delta_vs_m4"],
            "b0_rank_pct": "frozen P4.0 b0_rank / 50; lower is better",
            "b0_user_percentile": "(average ascending rank of raw score among same user's finite available Cold50 scores - 1)/(n-1); ties average; n<2 missing",
            "b0_user_zscore": "(score - mean same-user finite available Cold50)/population std (ddof=0); std<=1e-12 missing",
            "margin_to_rank2": "candidate score minus same-user B0 rank2 score",
            "margin_to_rank5": "candidate score minus same-user B0 rank5 score",
            "margin_to_user_median": "candidate score minus same-user Cold50 median",
            "availability": "explicit per derived variable; missing score/reference never zero-filled as confidence",
            "m4": ["m4_coarse_score", "m4_coarse_rank", "m4_coarse_rank_pct", "interaction_count_before_cutoff", "strict_cold_flag", "sparse1_5_flag"],
            "warm": ["warm_slot_rank", "warm_item_score", "warm_score_available", "warm_rank_pct"],
            "warm_score_source": "frozen PIT warm_model_score and availability, verified training label_end<=cutoff",
            "not_future_feature_contract": True,
        },
        "statistics": {
            "distribution": ["rows", "finite_rows", "mean", "std", "min", "p10", "p25", "median", "p75", "p90", "max", "iqr"],
            "auc": "exact pooled opportunity-row ROC AUC with ties weight 0.5; PR-AUC is non-interpolated average precision over unique score thresholds; missing scores excluded; one-class AUC null",
            "bins": "per-window all finite eligible opportunity scores; signed confidence; np.quantile linear boundaries independent of labels; equal scores stay together via searchsorted(side=left); retain empty bins with null rate/lift",
            "bins_count": [4, 10],
            "replication_note": "candidate confidence repeated for each of 3 replacement slots; opportunity rows are not independent users; correlated rank/percentile aliases are not independent signals",
            "no_deployable_thresholds": True,
        },
        "user_state": {"recent_active": "history_count_0_7 + history_count_8_28 > 0 in existing P3.7B raw user state", "inactive_recent": "complement including no history", "profile_available": "existing recent_vs_older_profile_available==1; no new geometry computation"},
        "verdict_rules": {
            "headroom_supported": "mean delta>=0.000300, mean MAP>W0, all4 delta>=0, >=3 strictly positive windows",
            "headroom_weak": "mean delta>0 but supported criteria fail",
            "headroom_rejected": "mean delta<=0; ANY negative per-user/window oracle delta is engineering failure",
            "separability_supported": ">=2 variables ROC-AUC>0.55 in >=3 windows; >=1 such variable from percentile/zscore/margin",
            "separability_weak": "not supported and (>=1 variable qualifies in3 windows OR >=2 variables qualify in2 windows)",
            "separability_rejected": "neither supported nor weak",
            "within_supported": ">=1 within variable AUC>0.55 and top-quartile lift>1 each in>=3 windows",
            "within_rejected": "none supported and >=1 within variable AUC<0.45 and top-quartile lift<1 each in>=3 windows",
            "within_inconclusive": "otherwise",
            "raw_drift_notice": "score is not probability; location/scale drift is calibration risk, not automatic ranking degradation",
            "raw_unstable": "ALL: (Cold50 median span/median IQR>1 OR max IQR/min IQR>2); beneficial opportunity score median span/median beneficial IQR>1; top-decile lift>1 in1..3 windows; a within variable top-decile lift>1 count exceeds raw and within portability supported",
            "raw_stable": "ALL4 finite: Cold50 median span/median IQR<=0.5; IQR max/min<=1.5; beneficial median span/median beneficial IQR<=0.5; raw top-decile lift>1 in4 windows",
            "raw_inconclusive": "otherwise (including distribution drift without evidence for every unstable condition)",
            "raw_zero_iqr": "zero denominator with nonzero span/scale -> very large risk ratio; both zero ->0 for span or1 for IQR ratio",
            "p4_1b_allowed": "headroom supported AND separability supported; permission flag only, never start P4.1B this turn",
        },
        "execution": {"read_only_sources": True, "training": False, "model_inference": False, "new_ranking_model": False, "hyperparameter_search": False, "cpu_minutes_estimate": [5, 15], "peak_memory_gib_budget": 6, "artifact_gib_estimate": [0.1, 0.5], "stop": "complete P4.1A audit, independent verification and measured docs append", "fallback": "W0 remains baseline regardless of audit verdict"},
        "final_week": "not_run",
    }
    write_json(target, contract)
    return contract
