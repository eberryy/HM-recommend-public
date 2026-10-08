from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .m33 import ROLLING_PROTOCOL
from .m4_contract import atomic_json, file_identity


RUN_ID = "phase3-p3.7b-v1-time-aware-relation-conditioned-hybrid"
STAGE = "P3.7B"
FINAL_WEEK = "2020-09-16"
WINDOWS = tuple(ROLLING_PROTOCOL)
K_VALUES = (5, 10, 20, 50, 100, 200)
VARIANTS = ("C0", "C1", "B0", "B1")
ALLOWED_DECISIONS = (
    "promote_M4_only_time_aware_hybrid",
    "promote_time_aware_hybrid_with_p33_aux",
    "combined_success_requires_p33_aux",
    "hybrid_not_better_than_static_p33_scoring",
    "stop_p3_7b_hybrid_after_gate_failure",
    "engineering_failure",
)
REQUIRED_OUTPUTS = (
    "P3_7B_FINAL.md",
    "P3_7B_metrics.json",
    "P3_7B_EXPERIMENT_CONTRACT.json",
    "p3_7b_feature_contract.json",
    "p3_7b_training_audit.json",
    "p3_7b_candidate_identity_audit.json",
    "p3_7b_time_gate_audit.json",
    "p3_7b_p33_auxiliary_audit.json",
    "p3_7b_rank_funnel.json",
    "P3_7B_OUTPUT_MANIFEST.json",
)

WINDOW_LABELS = {
    "winter_20200122": "winter（2020-01-22）",
    "spring_20200318": "spring（2020-03-18）",
    "early_summer_20200624": "early-summer（2020-06-24）",
    "late_summer_20200819": "late-summer（2020-08-19）",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def _feature_contract(*, repo_root: Path, source_root: Path, artifact_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": "phase3-p3.7b-feature-contract-v1",
        "stage": STAGE,
        "status": "preregistered_before_formal_computation",
        "run_id": RUN_ID,
        "created_at_utc": _now(),
        "roots": {
            "repo_root": str(repo_root.resolve()),
            "source_root": str(source_root.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
        },
        "row_unit": "one cutoff-user-candidate row from the exact frozen P3.1 M4 Top200",
        "history": {
            "maximum_distinct_items": 20,
            "selection": "most recent distinct purchased items strictly before the row cutoff",
            "all_history_denominators": (
                "the valid items among those most recent 20 distinct purchases; repeated transaction "
                "events and older-than-20 distinct items are excluded"
            ),
            "age_days": "cutoff date minus the selected distinct item's latest purchase date before cutoff",
            "age_buckets": [
                {"name": "age_0_7", "lower_inclusive": 0, "upper_inclusive": 7},
                {"name": "age_8_28", "lower_inclusive": 8, "upper_inclusive": 28},
                {"name": "age_29_84", "lower_inclusive": 29, "upper_inclusive": 84},
                {"name": "age_over_84", "lower_inclusive": 85, "upper_inclusive": None},
            ],
            "invariant": "each valid history item is assigned to exactly one mutually exclusive age bucket",
        },
        "m4_relation_features_per_age_bucket": [
            {"name": "bucket_present", "kind": "binary", "meaning": "at least one valid history item in this age bucket"},
            {"name": "history_count", "kind": "count", "meaning": "valid distinct history items in this age bucket"},
            {"name": "max_cosine", "kind": "continuous_missing", "meaning": "maximum M4 Student cosine to history items in this bucket"},
            {"name": "top3_mean_cosine", "kind": "continuous_missing", "meaning": "mean of the largest up to three M4 Student cosines in this bucket"},
            {"name": "same_product_type_history_count", "kind": "count", "meaning": "bucket history items sharing product_type_no with the candidate"},
            {"name": "same_garment_group_history_count", "kind": "count", "meaning": "bucket history items sharing garment_group_no with the candidate"},
        ],
        "p33_relation_features_per_age_bucket_B1_only": [
            {"name": "p33_max_cosine", "kind": "continuous_missing", "meaning": "maximum P3.3 Student cosine to bucket history"},
            {"name": "p33_top3_mean_cosine", "kind": "continuous_missing", "meaning": "mean of the largest up to three P3.3 Student cosines in the bucket"},
            {"name": "p33_minus_m4_max_cosine", "kind": "continuous_missing", "meaning": "raw P3.3 maximum cosine minus raw M4 maximum cosine for the same bucket"},
            {"name": "p33_minus_m4_top3_mean", "kind": "continuous_missing", "meaning": "raw P3.3 top-three mean minus raw M4 top-three mean for the same bucket"},
        ],
        "p33_global_relation_features_B1_only": {
            "fields": [
                {
                    "name": "p33_fixed_decay_score",
                    "kind": "continuous",
                    "formula": "max_h cosine(z_P33(candidate), z_P33(history_h)) * 2^(-age_h/28)",
                },
                {
                    "name": "p33_minus_m4_fixed_decay_score",
                    "kind": "continuous",
                    "formula": "raw p33_fixed_decay_score minus raw frozen M4 coarse score",
                },
            ],
            "resolved_placement": (
                "broadcast the two normalized candidate-global values into each of the four B1 relation "
                "buckets before the shared relation encoder; do not append them to candidate_state"
            ),
            "parity": "p33_fixed_decay_score must exactly reproduce P3.7A A01 scoring on the frozen M4 Top200",
        },
        "user_state": {
            "dimension": 10,
            "candidate_independent": True,
            "fields": [
                {"name": "history_count_0_7", "kind": "count"},
                {"name": "history_count_8_28", "kind": "count"},
                {"name": "history_count_29_84", "kind": "count"},
                {"name": "history_count_over_84", "kind": "count"},
                {
                    "name": "days_since_last_purchase",
                    "kind": "continuous",
                    "meaning": "finite for every candidate-bearing user because that user has at least one valid history item",
                },
                {
                    "name": "recent_0_28_purchase_share",
                    "kind": "continuous",
                    "denominator": "all valid items among the most recent 20 distinct purchases",
                },
                {
                    "name": "recent_vs_older_profile_cosine",
                    "kind": "continuous_missing",
                    "meaning": "cosine of normalized M4 means for age 0-28 days versus age 29-84 days",
                },
                {"name": "recent_vs_older_profile_available", "kind": "binary"},
                {
                    "name": "recent_28d_product_type_entropy",
                    "kind": "continuous",
                    "denominator": "valid distinct history items aged 0-28 days among the recent-20 history",
                    "empty_recent_definition": "zero when no 0-28-day history item has a valid product type",
                },
                {"name": "recent_28d_distinct_product_types", "kind": "count"},
            ],
            "forbidden": ["month", "season_label", "outer_window_id"],
        },
        "candidate_state": {
            "dimension": 5,
            "same_for_B0_and_B1": True,
            "fields": [
                {"name": "m4_coarse_score", "kind": "continuous"},
                {"name": "normalized_m4_coarse_rank", "kind": "fixed_rank", "formula": "(rank - 1) / 199"},
                {"name": "interaction_count_before_cutoff", "kind": "count"},
                {"name": "strict_cold_flag", "kind": "binary", "meaning": "interaction count before cutoff equals 0"},
                {"name": "sparse1_5_flag", "kind": "binary", "meaning": "interaction count before cutoff is from 1 through 5"},
            ],
            "explicit_exclusion": "the two candidate-global P3.3 values are not candidate_state fields",
        },
        "normalization": {
            "fit_scope": "only training cutoff rows in the current inner or outer chain",
            "validation_or_outer_statistics_allowed": False,
            "continuous": "finite values use train-only z-score",
            "missing": "raw missing value becomes normalized zero and must have an explicit availability indicator already specified by this contract",
            "counts": "log1p followed by train-only z-score",
            "rank": "fixed arithmetic mapping to [0,1], with no fitted statistic",
            "binary": "unchanged",
            "inner_scope": (
                "training users materialized for the P3.2-equivalent train asset; this is not a full-user "
                "outer evaluation denominator"
            ),
            "outer_scope": "union of the two frozen outer-train cutoffs only; outer-valid rows are excluded",
            "manifest": (
                "one preprocessing manifest per fitted inner or outer model, including field order, means, "
                "standard deviations, q0/q25/q50/q75/q100 and training cutoff identities"
            ),
        },
        "model_dimensions": {
            "B0_relation_encoder_input": 10,
            "B1_relation_encoder_input": 16,
            "explanation": (
                "B0: 6 M4 relation values + 4 age one-hot; B1: those 10 + 4 bucket-level "
                "P3.3 values + 2 candidate-global P3.3 values broadcast to the bucket"
            ),
            "encoded_relation": 16,
            "gate_input_for_each_real_or_null_expert": 35,
            "gate_input_explanation": "16 encoded relation + 10 user state + 5 candidate state + 4 age one-hot",
            "delta_input": 16,
            "delta_input_explanation": "1 aggregated time evidence + 10 user state + 5 candidate state",
        },
        "null_expert_resolution": {
            "uses_shared_gate": True,
            "gate_input": "16 zero relation values + user_state + candidate_state + 4 zero age-one-hot values",
            "relation_evidence": 0.0,
            "purpose": "allow a candidate-dependent no-adjustment probability without adding a separate unregistered network",
        },
        "variant_difference_invariant": (
            "B0 contains no P3.3 fields; B1 differs only by the six declared P3.3 auxiliary relation inputs "
            "per age-bucket row (four bucket-specific values plus two broadcast candidate-global values)"
        ),
        "final_week": "not_run",
    }


def _experiment_contract(
    *, repo_root: Path, source_root: Path, artifact_dir: Path, feature_identity: Mapping[str, Any]
) -> dict[str, Any]:
    authority_names = (
        "P3_1_FINAL.md",
        "P3_2_FINAL.md",
        "P3_3_FINAL.md",
        "P3_5_FINE_GRAINED_AUDIT.md",
        "P3_6_SIMILARITY_COMPLEMENTARITY_AUDIT.md",
        "P3_7A_2X2_DIAGNOSTIC.md",
        "P3_7A_2X2_DIAGNOSTIC.json",
    )
    report_dir = repo_root / "reports" / "phase3"
    authority = {
        name: file_identity(report_dir / name)
        for name in authority_names
        if (report_dir / name).is_file()
    }
    roadmap = repo_root / "docs" / "ROADMAP_PHASE3.zh-CN.md"
    if roadmap.is_file():
        authority[roadmap.name] = file_identity(roadmap)
    temporal_chains = {
        window: {
            "inner_train": list(protocol["inner_train"]),
            "inner_validation": protocol["inner_validation"],
            "outer_train": list(protocol["outer_train"]),
            "outer_validation": protocol["outer_validation"],
        }
        for window, protocol in ROLLING_PROTOCOL.items()
    }
    return {
        "schema_version": "phase3-p3.7b-experiment-contract-v1",
        "stage": STAGE,
        "status": "preregistered_before_formal_computation",
        "run_id": RUN_ID,
        "created_at_utc": _now(),
        "objective": (
            "test one fixed residual reranker that conditions M4 candidate-history relations on four "
            "behavior-age scales, and isolate whether cutoff-safe P3.3 geometry adds value"
        ),
        "accepted_prior_findings": {
            "p33_candidate_set_effect": "mixed",
            "p33_scoring_on_m4_pool": "supported",
            "p33_scoring_on_p33_pool": "supported",
            "p33_auxiliary_view_for_p37b": "allowed",
            "primary_failure": "candidate_membership_instability_is_primary",
        },
        "roots": {
            "repo_root": str(repo_root.resolve()),
            "source_root": str(source_root.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
        },
        "authoritative_context": authority,
        "feature_contract": dict(feature_identity),
        "execution_boundary": {
            "formal_new_model_experiment": True,
            "m4_is_exclusive_coarse_retriever": True,
            "p33_candidate_generation_allowed": False,
            "p32_network_reuse_allowed": False,
            "two_tower_allowed": False,
            "transformer_or_hstu_allowed": False,
            "multi_interest_retrieval_allowed": False,
            "raw_teacher_or_graph_inference_inputs_allowed": False,
            "teacher_weight_tuning_allowed": False,
            "calendar_router_or_window_feature_allowed": False,
            "hyperparameter_sweeps_allowed": False,
            "fusion_allowed_during_this_run": False,
            "final_week": "not_run",
            "reject_cutoff_on_or_after": FINAL_WEEK,
        },
        "frozen_checkpoint": {
            "retriever": "M4 single-teacher content-only Student plus P3.1 M4-space Top200",
            "cold_threshold_max_events_before_cutoff": 5,
            "candidate_budget_per_active_user": 200,
            "history_distinct_items": 20,
            "history_rule": "most recent 20 distinct purchased items strictly before each cutoff",
            "candidate_identity": "exact corresponding-cutoff P3.1 M4 Top200 user-item set",
            "truth_window": "[cutoff, cutoff + 7 days)",
        },
        "temporal_chains": temporal_chains,
        "variants": {
            "C0": {
                "role": "frozen M4 coarse baseline",
                "candidate_set": "exact P3.1 M4 Top200",
                "ordering": "frozen raw M4 coarse score",
                "training": False,
                "p37a_equivalent": "A00",
            },
            "C1": {
                "role": "P3.3 static geometry control",
                "candidate_set": "exact same P3.1 M4 Top200",
                "ordering": "frozen P3.3 fixed-decay coarse-style score only",
                "training": False,
                "p37a_equivalent": "A01",
            },
            "B0": {
                "role": "M4-only time-aware residual hybrid",
                "candidate_set": "exact same P3.1 M4 Top200",
                "inputs": "M4 age-bucket relations plus fixed user and candidate state",
                "contains_p33": False,
            },
            "B1": {
                "role": "M4 plus P3.3 auxiliary time-aware residual hybrid",
                "candidate_set": "exact same P3.1 M4 Top200",
                "inputs": "B0 inputs plus only the P3.3 fields declared in the feature contract",
                "contains_p33": True,
                "p33_is_content_only_at_inference": True,
            },
        },
        "p33_temporal_safety": {
            "required": "every training cutoff t uses a P3.3 Student trained/materialized only from behavior strictly before t",
            "preferred": "reuse an existing cutoff-aligned P3.3 Student and verify its manifest plus SHA256",
            "missing_asset_fallback": (
                "deterministically replay the frozen P3.3 training contract without changing teacher weights, "
                "Student architecture or teacher hyperparameters, using only data before t"
            ),
            "failure_policy": "fail B1 closed if cutoff safety cannot be proved; continue B0",
            "future_student_reuse_backwards_allowed": False,
        },
        "fixed_architecture": {
            "relation_encoder": "input -> Linear(16) -> GELU -> Linear(16) -> GELU, shared across four age buckets",
            "age_encoding": "fixed four-dimensional one-hot appended to relation-encoder input and gate input",
            "gate": "[encoded relation, user state, candidate state, age one-hot] -> Linear(16) -> GELU -> Linear(1)",
            "experts": "four real age experts plus one null/no-adjustment expert; softmax over five logits",
            "null_expert": (
                "shared gate with zero encoded relation and zero age one-hot, while retaining user/candidate "
                "state; its scalar evidence is fixed to zero"
            ),
            "relation_evidence": "shared Linear(16) -> scalar for each real age expert",
            "aggregation": "s_time = sum of alpha_b times e_b across four real experts; null contributes zero",
            "delta": "[s_time, 10 user-state values, 5 candidate-state values] -> Linear(32) -> GELU -> Linear(1)",
            "residual": "raw score_final = raw m4_coarse_score + delta_score",
            "initialization": "the final delta layer weight and bias are exactly zero-initialized",
            "no_width_or_architecture_search": True,
        },
        "normalization": {
            "authority": "p3_7b_feature_contract.json",
            "fit_from": "only current chain training-cutoff observations",
            "inner_is_not_full_user": True,
            "outer_valid_or_inner_valid_used_for_statistics": False,
        },
        "training": {
            "loss": "pairwise logistic/BPR: -log sigmoid(score_pos - score_neg)",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.00001,
            "training_batch_pairs": 512,
            "scoring_batch_candidate_rows": 16384,
            "sampling_seed": 20260905,
            "positive_policy": "retain every positive row present in the frozen Top200 training asset",
            "negative_policy": "same-user unobserved candidates only",
            "rank_buckets": {"hard": [1, 67], "medium": [68, 134], "easy": [135, 200]},
            "maximum_negatives_per_positive": 50,
            "sampling": "exact P3.2 deterministic fixed-hash sampler with all three rank buckets represented",
            "B0_B1_parity": (
                "same train examples, sampled pair identities and order, random seed, optimizer, early-stopping "
                "rule and common preprocessing statistics"
            ),
            "negative_ratio_search": False,
        },
        "early_stopping": {
            "selection_data": "inner-validation cutoff only",
            "primary": "positive_density@20",
            "tie_break": "MRR",
            "max_epochs": 30,
            "patience": 4,
            "trace": ["train_BPR_loss", "inner_density@20", "inner_MRR", "inner_Top200_to_Top20"],
            "forbidden_uses": ["architecture selection", "age-bucket selection", "feature deletion", "post-hoc B0/B1 tuning"],
        },
        "evaluation": {
            "outer_scope": "all frozen evaluation users, including users contributing zero positive rows to density",
            "candidate_identity": "exact frozen P3.1 M4 Top200 for C0, C1, B0 and B1",
            "hybrid_tie_break": "descending final score, then ascending frozen M4 rank, then ascending catalog row",
            "C1_tie_break": "descending exact P3.7A A01 score, then ascending catalog row",
            "k_values": list(K_VALUES),
            "segments": ["strict_cold", "sparse_1_5", "cold_universe"],
            "required_conservation": ["candidate rows", "Top200 truth-pair count", "Recall@200"],
            "comparison_precision": "raw deterministic floating-point values, never display-rounded values",
        },
        "diagnostics_not_promotion_metrics": {
            "score": ["positive-vs-unobserved AUC", "same-user truth percentile", "truth score distribution", "unobserved score distribution"],
            "time_gate": (
                "mean and median alpha for four age experts plus null, overall and by truth, unobserved, "
                "strict-cold and sparse1-5; rank gain by recent activity and train-defined profile-stability quartile"
            ),
            "p33_mask": (
                "B1 inference only, replace every normalized P3.3 auxiliary input with zero, the training-mean "
                "value in normalized space; no retraining"
            ),
        },
        "formal_gates": {
            "B0_vs_C0": {
                "mean_density20": "strictly greater",
                "density20_non_degrading_windows": "at least 3 of 4",
                "mean_MRR": "strictly greater",
                "MRR_non_degrading_windows": "at least 3 of 4",
                "mean_Top200_to_Top20": "strictly greater",
                "cold_segment": (
                    "strict-cold or sparse1-5 Recall@20 has a strictly greater four-window mean and is "
                    "non-degrading in at least 3 of 4 windows"
                ),
                "candidate_identity": "exact",
            },
            "B1_vs_C0": "identical checks to B0_vs_C0",
            "B1_vs_C1_complexity": {
                "mean_density20": "strictly greater",
                "mean_MRR": "strictly greater",
                "mean_Top200_to_Top20": "non-decreasing",
                "joint_non_degrading_windows": "at least 3 of 4 windows have density@20 or MRR non-degrading",
            },
            "B1_vs_B0_selection_if_both_supported": {
                "mean_density20": "strictly greater",
                "mean_MRR": "strictly greater",
                "mean_Top200_to_Top20": "non-decreasing",
                "joint_non_degrading_windows": "at least 3 of 4 windows have density@20 or MRR non-degrading",
            },
        },
        "decision_state_machine": {
            "allowed_final_decisions": list(ALLOWED_DECISIONS),
            "ordered_branches": [
                {
                    "if": "any required engineering, identity, leakage, parity or output verification fails",
                    "decision": "engineering_failure",
                    "selected_variant": None,
                },
                {
                    "if": "B0 rejected and B1 rejected",
                    "decision": "stop_p3_7b_hybrid_after_gate_failure",
                    "selected_variant": "C1 if its exact P3.7A parity is verified, otherwise C0",
                },
                {
                    "if": "B0 rejected, B1 supported, and B1-vs-C1 complexity gate rejected",
                    "decision": "hybrid_not_better_than_static_p33_scoring",
                    "selected_variant": "C1",
                },
                {
                    "if": "B0 rejected, B1 supported, and B1-vs-C1 complexity gate supported",
                    "decision": "combined_success_requires_p33_aux",
                    "selected_variant": "B1",
                },
                {
                    "if": "B0 supported and B1 rejected",
                    "decision": "promote_M4_only_time_aware_hybrid",
                    "selected_variant": "B0",
                },
                {
                    "if": "B0 and B1 supported, B1-vs-C1 complexity gate supported, and B1 dominates B0 by the preregistered selection gate",
                    "decision": "promote_time_aware_hybrid_with_p33_aux",
                    "selected_variant": "B1",
                },
                {
                    "if": "B0 and B1 supported but the immediately preceding B1 selection condition is false",
                    "decision": "promote_M4_only_time_aware_hybrid",
                    "selected_variant": "B0",
                },
            ],
            "fusion_policy": (
                "P3.7B never starts Fusion; a future Warm+Cold Fusion stage is allowed only if the selected "
                "hybrid passed its formal cold-precision gate and all conservation/leakage checks"
            ),
        },
        "required_tests": [
            "all variants candidate identity equals frozen P3.1 Top200",
            "candidate rows exact parity",
            "Recall@200 exact parity",
            "Top200 truth-pair count exact parity",
            "age buckets mutually exclusive",
            "every valid history item assigned exactly one age bucket",
            "cutoff-safe user-state features",
            "cutoff-safe M4 embeddings",
            "cutoff-safe P3.3 auxiliary embeddings",
            "no future P3.3 Student reused backwards",
            "B0 contains zero P3.3 fields",
            "B1 common preprocessing contract identical to B0",
            "P3.7A C1 scoring exact reproduction",
            "P3.2 sampler parity",
            "residual last layer zero initialized",
            "five expert alphas sum to one",
            "final week rejected",
        ],
        "required_outputs": list(REQUIRED_OUTPUTS),
        "estimated_cost_and_stop": {
            "model_search": "none; one fixed B0 and one fixed B1 per temporal chain",
            "dominant_cost": "cutoff-safe feature materialization and four-window scoring rather than architecture tuning",
            "success": "one of the preregistered hybrid promotion states",
            "failure": "retain the verified static control or frozen M4 according to the state machine; do not reopen Fusion",
            "stop_after": "P3.7B final report and independent verification; do not start Warm+Cold Fusion",
        },
        "final_week": "not_run",
    }


def write_preregistered_contract(
    report_dir: Path,
    repo_root: Path,
    source_root: Path,
    artifact_dir: Path,
) -> dict[str, Path]:
    """Atomically write the two immutable P3.7B preregistration contracts."""
    report_dir = report_dir.resolve()
    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    artifact_dir = artifact_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    feature_path = report_dir / "p3_7b_feature_contract.json"
    experiment_path = report_dir / "P3_7B_EXPERIMENT_CONTRACT.json"
    if feature_path.exists() != experiment_path.exists():
        raise RuntimeError("partial P3.7B preregistration exists; refusing to overwrite it")
    if feature_path.is_file() and experiment_path.is_file():
        feature = json.loads(feature_path.read_text(encoding="utf-8"))
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        expected_roots = {
            "repo_root": str(repo_root),
            "source_root": str(source_root),
            "artifact_dir": str(artifact_dir),
        }
        checks = {
            "feature_run_id": feature.get("run_id") == RUN_ID,
            "experiment_run_id": experiment.get("run_id") == RUN_ID,
            "feature_status": feature.get("status") == "preregistered_before_formal_computation",
            "experiment_status": experiment.get("status") == "preregistered_before_formal_computation",
            "feature_roots": feature.get("roots") == expected_roots,
            "experiment_roots": experiment.get("roots") == expected_roots,
            "feature_identity": experiment.get("feature_contract", {}).get("sha256")
            == file_identity(feature_path)["sha256"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"existing P3.7B preregistration drift: {checks}")
        return {"experiment_contract": experiment_path, "feature_contract": feature_path}
    atomic_json(
        feature_path,
        _feature_contract(repo_root=repo_root, source_root=source_root, artifact_dir=artifact_dir),
    )
    atomic_json(
        experiment_path,
        _experiment_contract(
            repo_root=repo_root,
            source_root=source_root,
            artifact_dir=artifact_dir,
            feature_identity=file_identity(feature_path),
        ),
    )
    return {"experiment_contract": experiment_path, "feature_contract": feature_path}


def _walk_get(value: Any, *paths: tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        current = value
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                break
            current = current[part]
        else:
            return current
    return default


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{number:.{digits}f}"


def _int_fmt(value: Any) -> str:
    number = _number(value)
    return "N/A" if number is None else f"{int(number):,}"


def _bool_label(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"supported", "passed", "pass", "true"}:
            return "通过"
        if lowered in {"rejected", "failed", "fail", "false"}:
            return "失败"
        return value
    if value is True:
        return "通过"
    if value is False:
        return "失败"
    return "N/A"


def _windows(master: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _walk_get(
        master,
        ("windows",),
        ("metrics", "windows"),
        ("four_variant_metrics", "windows"),
        default={},
    )
    return value if isinstance(value, Mapping) else {}


def _variant(window: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = _walk_get(window, ("variants", name), ("metrics", name), (name,), default={})
    return value if isinstance(value, Mapping) else {}


def _at_k(metrics: Mapping[str, Any], k: int) -> Mapping[str, Any]:
    value = _walk_get(metrics, ("at_k", str(k)), ("at_k", k), default={})
    return value if isinstance(value, Mapping) else {}


def _density(metrics: Mapping[str, Any], k: int) -> Any:
    return _walk_get(
        metrics,
        (f"positive_density@{k}",),
        (f"density@{k}",),
        (f"density{k}",),
        ("density", str(k)),
        ("density", k),
        default=_walk_get(_at_k(metrics, k), ("positive_density",)),
    )


def _mrr(metrics: Mapping[str, Any]) -> Any:
    return _walk_get(metrics, ("mrr",), ("ranking", "mrr"))


def _conversion(metrics: Mapping[str, Any], k: int) -> Any:
    return _walk_get(
        metrics,
        (f"top200_to_top{k}",),
        ("conversion", f"top200_to_top{k}"),
        ("ranking", "conversion", f"top200_to_top{k}"),
    )


def _segment_metric(metrics: Mapping[str, Any], segment: str, metric: str, k: int) -> Any:
    aliases = {
        "strict_cold": ("strict_cold", "strict"),
        "sparse_1_5": ("sparse_1_5", "sparse1_5", "sparse"),
        "cold_universe": ("cold_universe", "cold"),
    }[segment]
    for alias in aliases:
        direct = _walk_get(metrics, (f"{alias}_{metric}@{k}",), (f"{alias}_{metric}{k}",))
        if direct is not None:
            return direct
        nested = _walk_get(_at_k(metrics, k), ("segments", alias, metric))
        if nested is not None:
            return nested
    return None


def _ranking_value(metrics: Mapping[str, Any], name: str) -> Any:
    aliases = {
        "first_mean": ("first_positive_rank_mean", "first_positive_rank_mean_hit_users"),
        "first_median": ("first_positive_rank_median", "first_positive_rank_median_hit_users"),
        "p25": ("positive_rank_p25",),
        "p50": ("positive_rank_p50",),
        "p75": ("positive_rank_p75",),
    }[name]
    for alias in aliases:
        value = _walk_get(metrics, (alias,), ("ranking", alias))
        if value is not None:
            return value
    return None


def _gate(master: Mapping[str, Any], name: str) -> Any:
    aliases = {
        "B0": ("B0_gate", "B0_vs_C0", "B0", "time_aware_M4_only"),
        "B1": ("B1_gate", "B1_vs_C0", "B1", "time_aware_with_p33_aux"),
        "B1_vs_C1": ("B1_vs_C1_complexity_gate", "B1_vs_C1_complexity", "B1_vs_C1"),
        "B1_vs_B0": (
            "B1_vs_B0_selection_if_both_supported",
            "B1_vs_B0_selection_gate",
            "B1_vs_B0_selection",
            "B1_vs_B0",
        ),
    }[name]
    gates = _walk_get(master, ("gates",), ("formal_gates",), default={})
    if not isinstance(gates, Mapping):
        gates = {}
    for alias in aliases:
        if alias in gates:
            return gates[alias]
        if alias in master:
            return master[alias]
    return None


def _gate_result(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _walk_get(value, ("passed",), ("supported",), ("result",), ("status",))
    return value


def _decision(master: Mapping[str, Any]) -> tuple[str, str | None, bool | None]:
    decision_block = _walk_get(master, ("decision",), ("final_decision",), default={})
    if isinstance(decision_block, Mapping):
        code = _walk_get(
            decision_block,
            ("final_decision",),
            ("decision",),
            ("state",),
            ("status",),
            default="engineering_failure",
        )
        selected = _walk_get(decision_block, ("selected_variant",), default=None)
        fusion = _walk_get(decision_block, ("fusion_allowed",), ("warm_cold_fusion_allowed",), default=None)
    else:
        code = decision_block
        selected = _walk_get(master, ("selected_variant",), default=None)
        fusion = _walk_get(master, ("fusion_allowed",), ("warm_cold_fusion_allowed",), default=None)
    code = str(code)
    if code not in ALLOWED_DECISIONS:
        code = "engineering_failure"
    return code, None if selected is None else str(selected), fusion if isinstance(fusion, bool) else None


def _raw_decision_code(master: Mapping[str, Any]) -> Any:
    block = _walk_get(master, ("decision",), ("final_decision",), default=None)
    if isinstance(block, Mapping):
        return _walk_get(block, ("final_decision",), ("decision",), ("state",), ("status",))
    return block


def _decision_explanation(code: str) -> str:
    return {
        "promote_M4_only_time_aware_hybrid": "B0 通过正式门槛，保留不含 P3.3 辅助输入的 M4-only Hybrid。",
        "promote_time_aware_hybrid_with_p33_aux": "B0/B1 均通过，且 B1 同时通过相对 C1 与相对 B0 的复杂度收益条件，保留 B1。",
        "combined_success_requires_p33_aux": "只有 B1 通过，并且超过 C1；成功依赖 P3.3 辅助几何，不能声称纯时间建模成功。",
        "hybrid_not_better_than_static_p33_scoring": "B1 即使可能超过 C0，也没有证明比不训练的 C1 更好，保留简单静态 P3.3 打分。",
        "stop_p3_7b_hybrid_after_gate_failure": "B0/B1 均未通过正式门槛，停止本轮 Hybrid，并按已验证对照回退。",
        "engineering_failure": "资产、时点、守恒、训练或证据输出存在工程失败，不能形成模型效果结论。",
    }[code]


def _metric_table(windows: Mapping[str, Any], metric: str) -> list[str]:
    lines = [
        f"| 窗口 | {' | '.join(VARIANTS)} |",
        "|---|---:|---:|---:|---:|",
    ]
    for window_name in WINDOWS:
        row = windows.get(window_name, {})
        values = []
        for variant_name in VARIANTS:
            metrics = _variant(row, variant_name)
            if metric.startswith("density"):
                values.append(_fmt(_density(metrics, int(metric.removeprefix("density"))), 8))
            elif metric == "mrr":
                values.append(_fmt(_mrr(metrics)))
            elif metric.startswith("conversion"):
                values.append(_fmt(_conversion(metrics, int(metric.removeprefix("conversion")))))
        lines.append(f"| {WINDOW_LABELS.get(window_name, window_name)} | {' | '.join(values)} |")
    return lines


def _means(windows: Mapping[str, Any], getter: Any, variant: str) -> float | None:
    values = []
    for window_name in WINDOWS:
        value = _number(getter(_variant(windows.get(window_name, {}), variant)))
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if len(values) == len(WINDOWS) else None


def render_final(master: Mapping[str, Any]) -> str:
    """Render the measured P3.7B master payload without assuming every optional audit exists."""
    windows = _windows(master)
    code, selected, fusion_explicit = _decision(master)
    b0_pass = _gate_result(_gate(master, "B0"))
    b1_pass = _gate_result(_gate(master, "B1"))
    complexity_pass = _gate_result(_gate(master, "B1_vs_C1"))
    b1_b0_pass = _gate_result(_gate(master, "B1_vs_B0"))
    if fusion_explicit is None:
        fusion_allowed = selected in {"B0", "B1"} and code in {
            "promote_M4_only_time_aware_hybrid",
            "promote_time_aware_hybrid_with_p33_aux",
            "combined_success_requires_p33_aux",
        }
    else:
        fusion_allowed = fusion_explicit

    lines = [
        "# P3.7B：分时间尺度、按关系条件化的冷商品残差重排",
        "",
        "## 结论",
        "",
        f"- 运行状态：`{master.get('status', 'unknown')}`；机器可读决策：`{code}`。",
        f"- B0 相对 C0 正式门槛：**{_bool_label(b0_pass)}**；B1 相对 C0 正式门槛：**{_bool_label(b1_pass)}**。",
        f"- B1 相对 C1 复杂度门槛：**{_bool_label(complexity_pass)}**；B1 相对 B0 选择门槛：**{_bool_label(b1_b0_pass)}**。",
        f"- 最终保留角色：`{selected or '未形成可晋级角色'}`。{_decision_explanation(code)}",
        f"- 后续 Warm+Cold Fusion：**{'允许另行立项' if fusion_allowed else '不允许重开'}**。本轮没有启动 Fusion。",
        f"- 最终周 `{FINAL_WEEK}`：`{master.get('final_week', 'not_run')}`。",
        "",
        "## 术语、统计对象与实验边界",
        "",
        "- **coarse retrieval（粗召回，行业通用阶段）**：先从全目录为用户取得较小候选池。本轮始终由 M4 单教师纯内容 Student 独占，候选单位是一个截止日—用户—商品行；P3.3 不生成候选。",
        "- **Top200 candidate identity（Top200 候选身份，本项目守恒条件）**：四种角色对同一用户必须包含完全相同的200件 P3.1 M4 候选，只允许改变次序；因此候选行数、Top200 内未来正例对数和 Recall@200 必须逐窗精确相等。",
        "- **C0（冻结 M4 粗排对照，本项目自定义角色）**：相同 M4 Top200 按原始 M4 粗排分数排序，对应 P3.7A A00。",
        "- **C1（静态 P3.3 几何对照，本项目自定义角色）**：相同 M4 Top200 仅按 P3.3 Student 的28天半衰期分数排序，不训练，对应 P3.7A A01。",
        "- **B0（M4-only 时间感知 Hybrid，本项目自定义角色）**：只用 M4 的分时间桶关系证据、用户状态和五维候选状态学习残差。Hybrid 在此表示“冻结 M4 基准分数 + 小网络修正”，不是候选集合融合。",
        "- **B1（含 P3.3 辅助几何的时间感知 Hybrid，本项目自定义角色）**：候选仍是完全相同的 M4 Top200；只比 B0 多 P3.3 的桶内关系值以及广播到四桶的两项全局分数。推理期 P3.3 只读图片与静态属性，不读取图或行为教师。",
        "- **age bucket（行为年龄桶，本项目固定划分）**：按最近20件不同已购商品距截止日的天数分为0–7、8–28、29–84、超过84天四档；四档互斥且穷尽有效历史。它们保留不同时间尺度证据，不是按季节硬切模型。",
        "- **soft conditioning gate（软条件门控，混合专家行业常见机制）**：对四个时间桶和一个无调整专家产生五个非负权重，五者对每条候选行之和为1；权重可随用户与候选变化，而不是固定路由。",
        "- **null expert（无调整专家，本项目角色）**：其关系向量和时间桶标识均为0、证据固定为0，但保留用户与候选状态进入共享门控；它让网络可以选择少改动 M4 基准。",
        "- **residual reranker（残差重排器，行业常用结构）**：最终原始分数等于 M4 原始粗排分数加网络预测修正；最后一层从0初始化，所以训练开始时近似 C0。",
        "- **profile cosine（用户短中期偏好余弦，本项目用户状态特征）**：分别对用户截止日前0–28天和29–84天已购商品的 M4 embedding 求均值，再计算两段均值向量的余弦；两段都至少有一件历史商品时才可用。profile quartile（偏好余弦四分位，本项目分层）边界只由对应外层训练链中该值可用的用户计算。",
        "- **Warm+Cold Fusion（暖侧与冷侧融合，本项目后续阶段）**：未来把常规暖商品主排序与本轮 cold/sparse 专家输出合并并统一产生最终 Top12；它不是本轮 Hybrid，也未在 P3.7B 中运行。",
        "- **future-positive / truth pair（未来正例用户—商品对，本项目标签单位）**：用户在截止日起未来7天实际购买的一件不同商品；H&M 没有曝光日志，未购买候选只能称未观察候选，不能称用户明确拒绝。",
        "- **positive density@K（前K正例密度，本项目核心精度指标）**：有冻结 M4 候选行的外层用户，其前K候选行中的未来正例行数除以实际前K候选行数。用户只要有候选行，即使其中没有正例也仍留在分母；没有历史、因而没有 M4 候选行的用户不进入这个按候选行计算的分母。",
        "- **MRR（Mean Reciprocal Rank，平均倒数排名，行业通用指标）**：每名冷/稀疏真实用户首个正例名次的倒数再求平均；Top200 未命中用户贡献0。",
        "- **Top200→TopK conversion（Top200到TopK正例集中率，本项目漏斗指标）**：已位于同一候选池 Top200 的未来正例对中，重排后进入前K的比例；分母是该角色 Top200 内正例用户—商品对数。",
        "- **strict-cold Recall@K（严格冷商品召回率）**：截止日前购买事件为0的未来真实商品，被用户前K覆盖的比例，再以有此类 truth 的用户为分母求平均。",
        "- **sparse1-5 Recall@K（稀疏商品召回率）**：同上，但商品截止日前有1至5条购买事件。",
        "- **cold-universe HitRate@K（冷/稀疏用户命中率）**：至少有一件 strict-cold 或 sparse1-5 truth 的用户中，前K至少命中一件的用户比例。",
        "- **positive-vs-unobserved AUC（正例对未观察候选的曲线下面积，行业通用诊断）**：衡量分数可分性，只作机制解释，不能替代前K正式门槛。",
        "- 本项目采用 optimistic all-articles catalog（乐观全目录假设）：没有上架、库存与曝光时间，只能把全部商品行当作可选目录；离线命中不能直接外推线上可售转化。",
        "",
        "## 冻结设计与防泄漏处理",
        "",
        "- `T_cold=5`、`K0=200`、`history_N=20` 均冻结；历史只取截止日前最近20件不同已购商品。所有 history 统计的分母都是这20件中的有效项，不使用重复事件扩大权重。",
        "- B0/B1 共用训练正例、同用户负例、P3.2 固定哈希抽样结果、随机种子、优化器与早停规则。负例按 M4 名次1–67、68–134、135–200三档覆盖，每个正例最多50个；没有搜索负例比例。",
        "- B0/B1 使用 paired layer-wise initialization（按层配对初始化，本项目归因控制）：M4与年龄输入列、门控、证据层和残差层的初始参数逐值相同，只有 B1 新增的 P3.3 输入列单独初始化；机器审计保存两组公共参数摘要。",
        "- 连续特征只用当前训练链统计量做 z-score（均值为0、标准差为1的线性标准化）；计数先做 log1p，再标准化；缺失连续值在标准化空间填0，并由合同指定的可用性标志区分。inner 训练资产不是完整外层用户分母。",
        "- 每个训练时点只能使用该时点前行为物化的 M4/P3.3 Student。更晚外层 Student 不得反向服务更早样本；无法证明 P3.3 时点安全时只让 B1 失败关闭，B0 继续。",
        "- inner validation（内层时间验证，本项目训练口径）只决定 epoch（训练轮，行业通用单位；一次完整遍历当前固定成对训练样本）：先最大化 density@20，再用 MRR 打破并列；最多30轮，耐心值4。它不用于删特征、改桶或事后选结构。",
        "",
        "## 四窗核心结果",
        "",
        "### positive density@20",
        "",
        *_metric_table(windows, "density20"),
        "",
        "### MRR",
        "",
        *_metric_table(windows, "mrr"),
        "",
        "### Top200→Top20 conversion",
        "",
        *_metric_table(windows, "conversion20"),
        "",
        "四窗均值（仅当四窗值均存在时计算）：",
        "",
        "| 指标 | C0 | C1 | B0 | B1 |",
        "|---|---:|---:|---:|---:|",
        "| density@20 | " + " | ".join(_fmt(_means(windows, lambda x: _density(x, 20), v), 8) for v in VARIANTS) + " |",
        "| MRR | " + " | ".join(_fmt(_means(windows, _mrr, v)) for v in VARIANTS) + " |",
        "| Top200→Top20 | " + " | ".join(_fmt(_means(windows, lambda x: _conversion(x, 20), v)) for v in VARIANTS) + " |",
        "",
        "## 完整前K密度与冷商品覆盖",
        "",
        "下表每行的六个密度值依次使用所有实际存在的 Top5、Top10、Top20、Top50、Top100、Top200 候选行为分母；没有历史、因而没有 M4 候选行的外层用户不进入候选行分母。Recall 与 HitRate 表的每个数值则使用对应冷度分群的真实用户为分母，Top200 未命中者仍计0。",
        "",
        "| 窗口 | 角色 | density@5 | @10 | @20 | @50 | @100 | @200 | 每千候选正例对@200 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window_name in WINDOWS:
        row = windows.get(window_name, {})
        for variant_name in VARIANTS:
            metrics = _variant(row, variant_name)
            density_values = [_density(metrics, k) for k in K_VALUES]
            efficiency = _walk_get(
                metrics,
                ("truth_pairs_per_1k_candidates",),
                ("at_k", "200", "truth_pairs_per_1k_candidates"),
            )
            if efficiency is None and _number(density_values[-1]) is not None:
                efficiency = float(density_values[-1]) * 1000.0
            lines.append(
                f"| {WINDOW_LABELS.get(window_name, window_name)} | {variant_name} | "
                + " | ".join(_fmt(value, 8) for value in density_values)
                + f" | {_fmt(efficiency, 4)} |"
            )
    lines += [
        "",
        "| 窗口 | 角色 | strict R@5 | @10 | @20 | @50 | sparse R@5 | @10 | @20 | @50 | cold Hit@5 | @10 | @20 | @50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window_name in WINDOWS:
        row = windows.get(window_name, {})
        for variant_name in VARIANTS:
            metrics = _variant(row, variant_name)
            values = [
                *(_segment_metric(metrics, "strict_cold", "recall", k) for k in (5, 10, 20, 50)),
                *(_segment_metric(metrics, "sparse_1_5", "recall", k) for k in (5, 10, 20, 50)),
                *(_segment_metric(metrics, "cold_universe", "hit_rate", k) for k in (5, 10, 20, 50)),
            ]
            lines.append(
                f"| {WINDOW_LABELS.get(window_name, window_name)} | {variant_name} | "
                + " | ".join(_fmt(value) for value in values)
                + " |"
            )
    lines += [
        "",
        "## 排名分布与漏斗",
        "",
        "首命中均值/中位数只以 Top200 中至少命中一个正例的用户为分母；正例名次 p25/p50/p75 的统计单位是 Top200 内未来正例用户—商品对。",
        "",
        "| 窗口 | 角色 | 首命中均值 | 首命中中位数 | 正例名次p25 | p50 | p75 | 200→5 | 200→10 | 200→20 | 200→50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window_name in WINDOWS:
        row = windows.get(window_name, {})
        for variant_name in VARIANTS:
            metrics = _variant(row, variant_name)
            values = [
                _ranking_value(metrics, "first_mean"),
                _ranking_value(metrics, "first_median"),
                _ranking_value(metrics, "p25"),
                _ranking_value(metrics, "p50"),
                _ranking_value(metrics, "p75"),
                *(_conversion(metrics, k) for k in (5, 10, 20, 50)),
            ]
            lines.append(
                f"| {WINDOW_LABELS.get(window_name, window_name)} | {variant_name} | "
                + " | ".join(_fmt(value, 2) if index < 5 else _fmt(value) for index, value in enumerate(values))
                + " |"
            )

    identity = _walk_get(master, ("candidate_identity_audit",), default={})
    identity_windows = _walk_get(identity, ("windows",), default={}) if isinstance(identity, Mapping) else {}
    lines += [
        "",
        "## Top200 身份与时点安全门禁",
        "",
        "| 窗口 | 候选身份一致 | 候选行一致 | Recall@200一致 | Top200正例对一致 | M4时点安全 | P3.3时点安全 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for window_name in WINDOWS:
        row = identity_windows.get(window_name, {}) if isinstance(identity_windows, Mapping) else {}
        if not row:
            row = _walk_get(windows.get(window_name, {}), ("candidate_identity",), default={})
        time_row = _walk_get(master, ("training_audit", "windows", window_name), default={})
        lines.append(
            f"| {WINDOW_LABELS.get(window_name, window_name)} | "
            f"{_bool_label(_walk_get(row, ('candidate_identity_exact',), ('candidate_identity',), ('all_variants_exact',)))} | "
            f"{_bool_label(_walk_get(row, ('candidate_rows_exact',), ('candidate_rows_parity',), ('all_variants_exact',)))} | "
            f"{_bool_label(_walk_get(row, ('recall200_exact',), ('recall200_parity',), ('all_variants_exact',)))} | "
            f"{_bool_label(_walk_get(row, ('top200_truth_pairs_exact',), ('truth_pairs_parity',), ('all_variants_exact',)))} | "
            f"{_bool_label(_walk_get(time_row, ('m4_cutoff_safe',), default=_walk_get(row, ('m4_cutoff_safe',))))} | "
            f"{_bool_label(_walk_get(time_row, ('p33_cutoff_safe',), default=_walk_get(row, ('p33_cutoff_safe',))))} |"
        )
    lines += [
        "",
        "任何一项不为通过都属于工程失败，而不是模型负结果；四角色只有在同一候选集合上比较，才可把差值归因于排序。",
        "",
        "## 训练与早停证据",
        "",
        "| 窗口 | 角色 | 训练正例行 | 成对训练样本 | 选定epoch | inner density@20 | inner MRR | 已记录epoch数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    training_windows = _walk_get(master, ("training_audit", "windows"), default={})
    if not isinstance(training_windows, Mapping):
        training_windows = {}
    for window_name in WINDOWS:
        window_training = training_windows.get(window_name, {})
        for variant_name in ("B0", "B1"):
            row = _walk_get(window_training, (variant_name,), ("variants", variant_name), default={})
            lines.append(
                f"| {WINDOW_LABELS.get(window_name, window_name)} | {variant_name} | "
                f"{_int_fmt(_walk_get(row, ('positive_rows',), ('train_positive_rows',), ('inner', 'pair_audit', 'positive_candidate_rows')))} | "
                f"{_int_fmt(_walk_get(row, ('pair_rows',), ('training_pairs',), ('inner', 'pair_audit', 'pair_rows')))} | "
                f"{_int_fmt(_walk_get(row, ('selected_epoch',), ('best_epoch',), ('inner', 'selected_epoch')))} | "
                f"{_fmt(_walk_get(row, ('inner_density20',), ('selected_metrics', 'density20'), ('inner', 'selected_key', 'density20')), 8)} | "
                f"{_fmt(_walk_get(row, ('inner_mrr',), ('selected_metrics', 'mrr'), ('inner', 'selected_key', 'mrr')))} | "
                f"{_int_fmt(len(_walk_get(row, ('inner', 'trace'), default=[])))} |"
            )
    paired_initialization_exact = all(
        _walk_get(training_windows, (window_name, "B0", "inner", "initialization", "common_relation_input_sha256"))
        == _walk_get(training_windows, (window_name, "B1", "inner", "initialization", "common_relation_input_sha256"))
        and _walk_get(training_windows, (window_name, "B0", "inner", "initialization", "shared_downstream_sha256"))
        == _walk_get(training_windows, (window_name, "B1", "inner", "initialization", "shared_downstream_sha256"))
        and _walk_get(training_windows, (window_name, "B0", "outer", "initialization", "common_relation_input_sha256"))
        == _walk_get(training_windows, (window_name, "B1", "outer", "initialization", "common_relation_input_sha256"))
        and _walk_get(training_windows, (window_name, "B0", "outer", "initialization", "shared_downstream_sha256"))
        == _walk_get(training_windows, (window_name, "B1", "outer", "initialization", "shared_downstream_sha256"))
        for window_name in WINDOWS
    )
    lines += [
        "",
        f"- B0/B1 按层配对初始化公共参数摘要逐窗一致：**{_bool_label(paired_initialization_exact)}**。该控制避免 B1 输入更宽时改变后续公共层的随机初值。",
    ]

    score_diag = _walk_get(master, ("score_diagnostics",), default={})
    if not isinstance(score_diag, Mapping):
        score_diag = {}
    lines += [
        "",
        "## 分数可分性诊断（不参与晋级）",
        "",
        "“同用户正例百分位”以该用户未观察候选为参照，1表示高于全部未观察候选；分数分布值均以候选行为统计单位。",
        "",
        "| 窗口 | 角色 | 正例对未观察 AUC | 同用户正例百分位中位数 | 正例分数中位数 | 未观察分数中位数 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for window_name in WINDOWS:
        for variant_name in ("B0", "B1"):
            row = _walk_get(
                score_diag,
                ("windows", window_name, variant_name),
                (window_name, variant_name),
                default=_walk_get(windows.get(window_name, {}), ("score_diagnostics", variant_name), default={}),
            )
            lines.append(
                f"| {WINDOW_LABELS.get(window_name, window_name)} | {variant_name} | "
                f"{_fmt(_walk_get(row, ('auc',), ('positive_vs_unobserved_auc',)))} | "
                f"{_fmt(_walk_get(row, ('same_user_truth_percentile', 'median'), ('same_user_truth_percentile_median',), ('truth_percentile_median',)))} | "
                f"{_fmt(_walk_get(row, ('truth_score_distribution', 'median'), ('truth_score', 'median'), ('truth_score_median',)))} | "
                f"{_fmt(_walk_get(row, ('unobserved_score_distribution', 'median'), ('unobserved_score', 'median'), ('unobserved_score_median',)))} |"
            )

    gate_audit = _walk_get(master, ("time_gate_audit",), default={})
    if not isinstance(gate_audit, Mapping):
        gate_audit = {}
    lines += [
        "",
        "## 时间门控机制审计（不参与晋级）",
        "",
        "alpha 是一条候选行分给五位专家的软权重。下表每格按“均值；中位数”展示；候选行数明确给出该行统计分母。truth 指未来正例候选行，未观察指标签窗未购买的候选行；strict-cold 与 sparse1-5 按候选商品在截止日前的购买事件数分组，不要求该候选是正例。",
        "",
        "| 窗口 | 角色 | 候选子集 | 候选行数 | α 0–7天 均值；中位数 | α 8–28天 均值；中位数 | α 29–84天 均值；中位数 | α >84天 均值；中位数 | α 无调整 均值；中位数 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    alpha_aliases = (
        ("alpha_0_7", "alpha_age_0_7"),
        ("alpha_8_28", "alpha_age_8_28"),
        ("alpha_29_84", "alpha_age_29_84"),
        ("alpha_over_84", "alpha_age_over_84"),
        ("alpha_null",),
    )
    population_aliases = (
        ("all_candidates", "全部候选"),
        ("truth_candidates", "未来正例"),
        ("unobserved_candidates", "未观察候选"),
        ("strict_cold_candidates", "strict-cold候选"),
        ("sparse_1_5_candidates", "sparse1-5候选"),
    )
    for window_name in WINDOWS:
        for variant_name in ("B0", "B1"):
            row = _walk_get(gate_audit, ("windows", window_name, variant_name), (window_name, variant_name), default={})
            for population, population_label in population_aliases:
                stats = _walk_get(
                    row,
                    ("statistics", population),
                    (population,),
                    default=row if population == "all_candidates" else {},
                )
                pairs = []
                for aliases in alpha_aliases:
                    mean_value = None
                    median_value = None
                    for alias in aliases:
                        compact_alias = alias.removeprefix("alpha_")
                        mean_value = _walk_get(
                            stats,
                            ("mean", compact_alias),
                            ("mean", alias),
                            (f"mean_{alias}",),
                            (alias, "mean"),
                        )
                        median_value = _walk_get(
                            stats,
                            ("median", compact_alias),
                            ("median", alias),
                            (f"median_{alias}",),
                            (alias, "median"),
                        )
                        if mean_value is not None or median_value is not None:
                            break
                    pairs.append(f"{_fmt(mean_value)}；{_fmt(median_value)}")
                lines.append(
                    f"| {WINDOW_LABELS.get(window_name, window_name)} | {variant_name} | "
                    f"{population_label} | {_int_fmt(_walk_get(stats, ('candidate_rows',), ('rows',)))} | "
                    + " | ".join(pairs)
                    + " |"
                )
    lines += [
        "",
        "rank gain（正例名次提升，本项目机制指标）等于 C0 原始名次减去 Hybrid 名次；正数表示未来正例上移。下表只以冻结 Top200 内未来正例用户—商品对为分母，`improved share` 是名次严格上移的正例对数除以该行正例对数。profile quartile（用户偏好稳定度四分位，本项目分层）边界只从对应外层训练链中 profile cosine 可用用户计算。",
        "",
        "| 窗口 | 角色 | 用户状态分层 | 正例对数 | 名次提升均值 | 名次提升中位数 | 严格上移比例 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    state_labels = (
        ("recent_active", "最近28天有历史"),
        ("inactive_recent", "最近28天无历史"),
        ("profile_unavailable", "profile cosine不可用"),
        ("q1_low", "profile稳定度Q1"),
        ("q2", "profile稳定度Q2"),
        ("q3", "profile稳定度Q3"),
        ("q4_high", "profile稳定度Q4"),
    )
    for window_name in WINDOWS:
        for variant_name in ("B0", "B1"):
            row = _walk_get(gate_audit, ("windows", window_name, variant_name), (window_name, variant_name), default={})
            gains = _walk_get(row, ("rank_gain_by_user_state",), ("rank_gain",), default={})
            for state, state_label in state_labels:
                value = _walk_get(gains, (state,), default={})
                lines.append(
                    f"| {WINDOW_LABELS.get(window_name, window_name)} | {variant_name} | {state_label} | "
                    f"{_int_fmt(_walk_get(value, ('truth_pairs',), ('rows',)))} | "
                    f"{_fmt(_walk_get(value, ('mean',)))} | {_fmt(_walk_get(value, ('median',)))} | "
                    f"{_fmt(_walk_get(value, ('improved_share',)))} |"
                )
    lines += [
        "",
        "如果不同用户状态下五个权重几乎不变，只能说明网络没有形成明显的状态依赖时间加权；即便权重变化明显，也仍须由正式前K指标决定是否晋级。",
        "",
        "## P3.3 辅助输入遮蔽诊断（不参与晋级）",
        "",
        "遮蔽指在不重训 B1 的情况下，把所有已标准化 P3.3 辅助字段置为0；0正是训练均值在标准化空间的取值。比较单位仍是同一 M4 Top200 候选行。",
        "",
        "| 窗口 | 指标 | B1完整输入 | B1遮蔽P3.3 | 完整减遮蔽 |",
        "|---|---|---:|---:|---:|",
    ]
    auxiliary = _walk_get(master, ("p33_auxiliary_audit",), default={})
    if not isinstance(auxiliary, Mapping):
        auxiliary = {}
    for window_name in WINDOWS:
        row = _walk_get(auxiliary, ("windows", window_name), (window_name,), default={})
        full = _walk_get(row, ("full",), ("B1_full",), ("full_metrics",), default={})
        masked = _walk_get(
            row,
            ("masked",),
            ("B1_p33_masked",),
            ("B1_p33_auxiliary_masked",),
            ("B1_with_p33_auxiliary_masked",),
            ("masked_metrics",),
            default={},
        )
        for metric_label, getter in (
            ("density@20", lambda x: _walk_get(x, ("density20",), default=_density(x, 20))),
            ("MRR", _mrr),
            (
                "Top200→Top20",
                lambda x: _walk_get(x, ("top200_to_top20",), default=_conversion(x, 20)),
            ),
        ):
            full_value = _number(getter(full))
            masked_value = _number(getter(masked))
            delta = None if full_value is None or masked_value is None else full_value - masked_value
            lines.append(
                f"| {WINDOW_LABELS.get(window_name, window_name)} | {metric_label} | "
                f"{_fmt(full_value, 8 if metric_label == 'density@20' else 6)} | "
                f"{_fmt(masked_value, 8 if metric_label == 'density@20' else 6)} | "
                f"{_fmt(delta, 8 if metric_label == 'density@20' else 6)} |"
            )

    gate_block = _walk_get(master, ("gates",), ("formal_gates",), default={})
    lines += [
        "",
        "## 预注册正式门槛",
        "",
        "正式判断使用未四舍五入的原始数值。B0/B1 相对 C0 均须同时满足：density@20 与 MRR 四窗均值严格提高且各至少3窗不退化；Top200→Top20 均值严格提高；strict-cold 或 sparse1-5 至少一个分群的 Recall@20 均值严格提高且至少3窗不退化；候选身份完全一致。B1 还必须证明其复杂度超过 C1。",
        "",
        f"- B0 gate：**{_bool_label(b0_pass)}**。",
        f"- B1 gate：**{_bool_label(b1_pass)}**。",
        f"- B1 vs C1 complexity gate：**{_bool_label(complexity_pass)}**。",
        f"- B1 vs B0 selection gate：**{_bool_label(b1_b0_pass)}**。",
        "- B0 与 B1 的正式通过均由 sparse1-5 Recall@20 分支支撑；strict-cold Recall@20 分支未通过。因此本轮证明的是冻结 Top200 内的稀疏商品排序改善，不能表述为严格冷商品问题已经解决。",
    ]
    if isinstance(gate_block, Mapping):
        for gate_name, gate_value in gate_block.items():
            if not isinstance(gate_value, Mapping):
                continue
            checks = _walk_get(gate_value, ("checks",), default={})
            if not isinstance(checks, Mapping):
                continue
            for check_name, check_value in checks.items():
                lines.append(f"- `{gate_name}.{check_name}`：{_bool_label(check_value)}。")

    resources = _walk_get(master, ("resources",), default={})
    lines += [
        "",
        "## 五个最终问题",
        "",
        f"1. **多时间尺度软条件化是否比固定 M4 粗排更好？** {_bool_label(b0_pass)}。这里以 B0 相对 C0 的整套正式门槛为准，而不是单窗 AUC 或某个局部指标。",
        f"2. **P3.3 辅助几何在 learned Hybrid（学习型混合残差模型）中是否继续提供增量？** {'是' if selected == 'B1' else '没有形成足以选择 B1 的正式证据'}；遮蔽实验只解释机制，最终选择仍由 B1 正式门槛、相对 C1 复杂度门槛及必要时相对 B0 门槛共同决定。",
        f"3. **Hybrid 是否超过简单 P3.3 静态打分对照？** {_bool_label(complexity_pass)}；该结论只针对 B1 相对 C1。",
        f"4. **最终保留什么？** `{selected or 'none'}`。{_decision_explanation(code)}",
        f"5. **是否允许进入 Warm+Cold Fusion？** {'允许后续单独预注册' if fusion_allowed else '不允许'}；P3.7B 本身在此停止。",
        "",
        "## 资源、证据与解释边界",
        "",
        f"- 成功缓存重跑耗时：{_fmt(_walk_get(resources, ('elapsed_seconds',)), 1)} 秒；峰值进程工作集：{_fmt((_number(_walk_get(resources, ('peak_working_set_bytes',))) or 0) / (1024 ** 3), 2) if _number(_walk_get(resources, ('peak_working_set_bytes',))) is not None else 'N/A'} GiB；设备：`{_walk_get(resources, ('device',), default='N/A')}`。该耗时复用了已做 SHA 校验的静态分数与特征缓存，不等同于从零冷启动总耗时；首次冷启动在报告序列化失败前的完整分项耗时没有统一记录，因此不补造总数。",
        "- 报告中的模型提升只表示冻结 M4 Top200 内部的冷/稀疏商品排序改善；它不增加候选可达性，也不是最终 Warm Top12 或线上业务收益。",
        "- P3.3 图教师、Item2Vec 与共购关系只存在于历史训练监督；B1 推理输入仍是纯内容 Student 几何。",
        "- 所有大型特征矩阵、模型和逐行预测位于被 Git 忽略的 artifacts 目录；输出清单记录路径、字节数、SHA256 与截止日身份。",
        "- 本轮不读取最终周、不启动 Two-Tower、不重训 Student、不做超参数搜索，也不自行启动 Fusion。",
    ]
    return "\n".join(lines) + "\n"


def _component_payload(
    master: Mapping[str, Any], key: str, supplied: Mapping[str, Any] | None, schema: str
) -> dict[str, Any]:
    value: Any = supplied if supplied is not None else master.get(key)
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        result = {"status": "missing_from_master", "windows": {}}
    result.setdefault("schema_version", schema)
    result.setdefault("stage", STAGE)
    result.setdefault("run_id", str(master.get("run_id", RUN_ID)))
    result.setdefault("final_week", str(master.get("final_week", "not_run")))
    return result


def _large_artifact_identities(values: Iterable[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for entry in values or ():
        if not isinstance(entry, Mapping) or "path" not in entry:
            raise TypeError("each large artifact must be a mapping containing path and cutoff_identity")
        cutoff_identity = entry.get("cutoff_identity", entry.get("cutoff"))
        if cutoff_identity is None:
            raise ValueError(f"large artifact has no cutoff identity: {entry['path']}")
        observed = file_identity(Path(str(entry["path"])))
        if "bytes" in entry and int(entry["bytes"]) != int(observed["bytes"]):
            raise RuntimeError(f"large artifact byte-size drift: {entry['path']}")
        if "sha256" in entry and str(entry["sha256"]) != observed["sha256"]:
            raise RuntimeError(f"large artifact SHA256 drift: {entry['path']}")
        identities.append({**observed, "cutoff_identity": cutoff_identity})
    return identities


def write_outputs(
    report_dir: Path,
    master: Mapping[str, Any],
    *,
    training_audit: Mapping[str, Any] | None = None,
    candidate_identity_audit: Mapping[str, Any] | None = None,
    time_gate_audit: Mapping[str, Any] | None = None,
    p33_auxiliary_audit: Mapping[str, Any] | None = None,
    rank_funnel: Mapping[str, Any] | None = None,
    large_artifacts: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Path]:
    """Write measured P3.7B evidence and a SHA-bound output manifest atomically."""
    report_dir = report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    if str(master.get("final_week", "not_run")) != "not_run":
        raise RuntimeError("P3.7B output refuses any final-week execution")
    raw_decision = _raw_decision_code(master)
    if raw_decision not in ALLOWED_DECISIONS:
        raise RuntimeError(f"unregistered or missing P3.7B decision: {raw_decision!r}")
    decision, _selected, _fusion = _decision(master)
    if _selected not in {None, *VARIANTS}:
        raise RuntimeError(f"unregistered selected P3.7B variant: {_selected!r}")

    metrics = dict(master)
    metrics.setdefault("schema_version", "phase3-p3.7b-metrics-v1")
    metrics.setdefault("stage", STAGE)
    metrics.setdefault("run_id", RUN_ID)
    metrics.setdefault("final_week", "not_run")
    decision_block = metrics.get("decision")
    machine_decision = dict(decision_block) if isinstance(decision_block, Mapping) else {}
    machine_decision["status"] = decision
    machine_decision["B0_gate"] = machine_decision.get(
        "B0_gate", _bool_label(_gate_result(_gate(master, "B0"))).replace("通过", "supported").replace("失败", "rejected")
    )
    machine_decision["B1_gate"] = machine_decision.get(
        "B1_gate", _bool_label(_gate_result(_gate(master, "B1"))).replace("通过", "supported").replace("失败", "rejected")
    )
    machine_decision["B1_vs_C1_complexity_gate"] = machine_decision.get(
        "B1_vs_C1_complexity_gate",
        _bool_label(_gate_result(_gate(master, "B1_vs_C1"))).replace("通过", "supported").replace("失败", "rejected"),
    )
    machine_decision["selected_variant"] = _selected
    metrics["decision"] = machine_decision
    components = {
        "p3_7b_training_audit.json": _component_payload(
            master, "training_audit", training_audit, "phase3-p3.7b-training-audit-v1"
        ),
        "p3_7b_candidate_identity_audit.json": _component_payload(
            master,
            "candidate_identity_audit",
            candidate_identity_audit,
            "phase3-p3.7b-candidate-identity-audit-v1",
        ),
        "p3_7b_time_gate_audit.json": _component_payload(
            master, "time_gate_audit", time_gate_audit, "phase3-p3.7b-time-gate-audit-v1"
        ),
        "p3_7b_p33_auxiliary_audit.json": _component_payload(
            master,
            "p33_auxiliary_audit",
            p33_auxiliary_audit,
            "phase3-p3.7b-p33-auxiliary-audit-v1",
        ),
        "p3_7b_rank_funnel.json": _component_payload(
            master, "rank_funnel", rank_funnel, "phase3-p3.7b-rank-funnel-v1"
        ),
    }
    component_keys = {
        "p3_7b_training_audit.json": "training_audit",
        "p3_7b_candidate_identity_audit.json": "candidate_identity_audit",
        "p3_7b_time_gate_audit.json": "time_gate_audit",
        "p3_7b_p33_auxiliary_audit.json": "p33_auxiliary_audit",
        "p3_7b_rank_funnel.json": "rank_funnel",
    }
    for filename, key in component_keys.items():
        # Keep the embedded and standalone audit payloads byte-for-byte equivalent
        # after the component metadata is normalized.  This lets the independent
        # verifier reject stale or partially regenerated evidence bundles.
        metrics[key] = components[filename]
    metrics_path = report_dir / "P3_7B_metrics.json"
    report_path = report_dir / "P3_7B_FINAL.md"
    atomic_json(metrics_path, metrics)
    for name, payload in components.items():
        atomic_json(report_dir / name, payload)
    _atomic_text(report_path, render_final(metrics))

    required_before_manifest = [name for name in REQUIRED_OUTPUTS if name != "P3_7B_OUTPUT_MANIFEST.json"]
    missing = [name for name in required_before_manifest if not (report_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("P3.7B required outputs are missing: " + ", ".join(missing))
    declared_large_artifacts = large_artifacts
    if declared_large_artifacts is None:
        candidate = master.get("large_artifacts")
        if isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes, Mapping)):
            declared_large_artifacts = candidate
    manifest = {
        "schema_version": "phase3-p3.7b-output-manifest-v1",
        "stage": STAGE,
        "status": str(master.get("status", "unknown")),
        "run_id": str(master.get("run_id", RUN_ID)),
        "created_at_utc": _now(),
        "outputs": {name: file_identity(report_dir / name) for name in required_before_manifest},
        "large_ignored_artifacts": _large_artifact_identities(declared_large_artifacts),
        "decision": decision,
        "selected_variant": _decision(master)[1],
        "final_week": "not_run",
    }
    manifest_path = report_dir / "P3_7B_OUTPUT_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    return {
        "metrics": metrics_path,
        "report": report_path,
        "training_audit": report_dir / "p3_7b_training_audit.json",
        "candidate_identity_audit": report_dir / "p3_7b_candidate_identity_audit.json",
        "time_gate_audit": report_dir / "p3_7b_time_gate_audit.json",
        "p33_auxiliary_audit": report_dir / "p3_7b_p33_auxiliary_audit.json",
        "rank_funnel": report_dir / "p3_7b_rank_funnel.json",
        "manifest": manifest_path,
    }
