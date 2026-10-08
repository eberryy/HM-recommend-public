from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m33 import ROLLING_PROTOCOL
from .m4_contract import atomic_json, file_identity
from .m55 import WARM_FEATURES


RUN_ID = "phase4-p4.0-v1-b0-warm-fusion"
K_WARM = 150
K_COLD = 50
K_FINAL = 12
SEED = 20260903
MAX_BOOST_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 20
NEGATIVES_PER_POSITIVE = 30

SOURCE_FEATURES = [
    "warm_present", "cold_present", "both_present", "source_branch",
    "b0_score", "b0_score_available", "b0_rank", "b0_rank_pct",
    "m4_coarse_score", "m4_coarse_rank", "m4_coarse_rank_pct",
    "b0_delta_vs_m4", "b0_delta_available",
    "interaction_count_before_cutoff", "strict_cold_flag", "sparse1_5_flag",
]
FUSION_FEATURES = list(dict.fromkeys(WARM_FEATURES + SOURCE_FEATURES))


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def build_contract(repo_root: Path, source_root: Path, artifact_dir: Path) -> dict[str, Any]:
    report = repo_root / "reports"
    authorities = [
        report / "phase3" / "P3_7A_2X2_DIAGNOSTIC.json",
        report / "phase3" / "P3_7B_metrics.json",
        report / "phase3" / "P3_7B_EXPERIMENT_CONTRACT.json",
        report / "phase3" / "p3_7b_feature_contract.json",
        report / "phase3" / "p3_7b_training_audit.json",
        report / "phase3" / "P3_7B_OUTPUT_MANIFEST.json",
        report / "m5_3" / "M5_3_metrics.json",
        report / "m5_4" / "M5_4_metrics.json",
        report / "m5_5" / "M5_5_metrics.json",
        report / "m5_6" / "metrics.json",
    ]
    for path in authorities:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "schema_version": "phase4-p4.0-experiment-contract-v1",
        "stage": "P4.0",
        "status": "preregistered_before_formal_computation",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "git_commit": _git_head(repo_root),
            "note": "git commit identifies the frozen pre-P4 base; P4 source files are bound by the output manifest",
        },
        "roots": {
            "repo_root": str(repo_root.resolve()),
            "source_root": str(source_root.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
        },
        "authoritative_inputs": {path.name: file_identity(path) for path in authorities},
        "outer_windows": ROLLING_PROTOCOL,
        "temporal_training": {
            "inner_use": "select boosting rounds only",
            "outer_use": "refit on the two preregistered earlier cutoffs",
            "learned_upstream_rule": "source label_end must be <= scoring cutoff",
            "final_week": "not_run",
        },
        "candidate_budgets": {"warm": K_WARM, "cold": K_COLD, "final": K_FINAL},
        "candidate_contract": {
            "warm": "exact frozen M5.4 Warm-v1 Top150",
            "cold": "exact P3.1 M4 Top200 reranked by latest eligible earlier P3.7B B0 model, then Top50",
            "earliest_fallback": "M4 coarse Top50; B0 score/rank missing and availability=0",
            "union_key": ["target_cutoff", "customer_id", "article_id"],
            "branches": ["warm_only", "cold_only", "warm_and_cold"],
        },
        "feature_contract": {
            "features": FUSION_FEATURES,
            "warm_backbone_authority": "hm_recsys.m55.WARM_FEATURES",
            "new_minimal_upstream_features": SOURCE_FEATURES,
            "incremental_materialization": "reuse M5.4 PIT rows by exact user-item identity and SHA-bound input; recompute only new rows under the same PIT functions; verify complete one-to-one coverage",
            "forbidden_b0_internal": [
                "alpha_0_7", "alpha_8_28", "alpha_29_84", "alpha_over_84",
                "alpha_null", "per_age_max_cosine", "per_age_top3_cosine",
                "profile_stability", "B0_relation_hidden_representation",
            ],
        },
        "lightgbm": {
            "objective": "lambdarank", "metric": "None", "learning_rate": 0.05,
            "num_leaves": 31, "min_data_in_leaf": 100, "feature_fraction": 1.0,
            "bagging_fraction": 1.0, "bagging_freq": 0, "seed": SEED,
            "deterministic": True, "force_col_wise": True, "num_threads": 8,
            "lambdarank_truncation_level": 20,
            "max_boost_rounds": MAX_BOOST_ROUNDS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "selection": "inner exact MAP@12 chooses boosting rounds only",
        },
        "variants": {
            "W0": "frozen Warm-v1 Top12; no training",
            "W1": "Warm150-only control with the same PIT backbone and temporal LambdaRank chain",
            "F0": "Warm150 union B0 Cold50; exact hm_recsys.m55.load_training_sample reuse",
            "F1": "same candidates/features/model as F0; boundary-aware negative sampling only",
        },
        "sampling": {
            "common": "retain all positives; exclude zero-positive groups from training; same-user unobserved rows; deterministic fixed hash; without replacement; at most 30 negatives per positive",
            "F0": "exact M5.5 source/rank-stratified implementation reuse",
            "F1_quota_per_cold_capable_positive": {
                "warm_rank_1_75": 12, "warm_rank_76_150": 12, "cold_rank_1_50": 6,
            },
            "F1_quota_per_warm_only_positive": {
                "cold_rank_1_20": 12, "cold_rank_21_50": 6, "warm_rank_1_150": 12,
            },
            "spill_order": [
                "same-source adjacent rank bucket",
                "other-source hard bucket",
                "remaining union negatives by fixed hash",
            ],
            "outer_metrics_may_not_change_quota": True,
        },
        "metrics": {
            "primary": "MAP@12 on the exact Frozen Warm-v1 full truth-user denominator",
            "segments": ["warm_21_plus", "strict_cold", "sparse1_5"],
            "candidate": ["candidate_recall", "oracle_map@12"],
            "cold_truth_funnel": ["union", "Top100", "Top50", "Top20", "Top12", "best", "p25", "median", "p75"],
            "replacement": "relative to W0 per user-item recommendation pairs",
            "cold_sparse_insertion_efficiency": "inserted positive recommendation pairs whose item had <=5 prior events divided by all inserted recommendation pairs whose item had <=5 prior events; denominator zero yields null",
        },
        "promotion_gates": {
            "overall": "F1 mean MAP@12 > W0; non-degrade >=3/4; worst delta >= -0.000300",
            "sparse": "F1 mean sparse1-5 MAP@12 > W0 and non-degrade >=3/4",
            "cold_top12": "cold-only truth reaches Top12 in >=2/4 windows and pooled count >0",
            "warm_protection": "F1 mean warm_21_plus MAP@12 >= W0",
            "boundary_sampling": "F1 mean MAP@12 > F0 and non-degrade vs F0 >=3/4",
            "fallback": "F0 may be selected only if F0 passes the corresponding overall, sparse, cold_top12 and warm-protection gates while F1 does not",
        },
        "allowed_decisions": [
            "promote_fusion_v2_boundary_sampling",
            "promote_fusion_v2_historical_sampling",
            "fusion_improves_map_but_fails_cold_top12_gate",
            "fusion_improves_cold_but_hurts_warm",
            "stop_phase4_p4_0_after_gate_failure",
            "engineering_failure",
        ],
        "failure_fallback": "retain W0, diagnose reachability versus ranking versus warm damage versus sampling/feature non-use, and wait for manual approval; do not start P4.1",
        "final_week": "not_run",
    }


def write_contract(repo_root: Path, source_root: Path, artifact_dir: Path, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "P4_0_EXPERIMENT_CONTRACT.json"
    if path.exists():
        return path
    atomic_json(path, build_contract(repo_root, source_root, artifact_dir))
    return path
