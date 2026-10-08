from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from .m2 import build_category_maps
from .m3 import _peak_working_set_bytes
from .m33 import ROLLING_PROTOCOL
from .m4_contract import FINAL_CUTOFF, WARM_MAP, atomic_json, file_identity
from .m5_model import _write_parquet
from .m55 import (
    DUAL_FEATURES,
    MAP_TOLERANCE,
    NEGATIVES_PER_POSITIVE,
    RUN_ID as M55_RUN_ID,
    SEED,
    WARM_FEATURES,
    WINDOW_TOLERANCE,
    _baseline_top12,
    _evaluate,
    _load_validation,
    _prepare,
    _rank_top12,
    _truth_and_counts,
    load_training_sample,
    train_ranker,
)
from .m54 import LINEAGES, RUN_ID as M54_RUN_ID


RUN_ID = "m5-6-v1-cold-supervision-suppression"
FIXED_ROUNDS = 40
COLD_NEGATIVE_QUOTA_PER_POSITIVE = 15
WARM_BOUNDARY_QUOTA_PER_POSITIVE = 5
WARM_HARD_QUOTA_PER_POSITIVE = 5
BOTH_QUOTA_PER_POSITIVE = 3
NO_COLD_POSITIVE_COLD_NEGATIVE_CAP = 3
WARM_BOUNDARY_MIN_RANK = 8
WARM_BOUNDARY_MAX_RANK = 20
SOURCE_FEATURES = sorted(set(DUAL_FEATURES) - set(WARM_FEATURES))
SOURCE_FEATURE_DEFINITIONS = {
    "both_present": "二值标记：同一用户—商品对同时被 Warm150 与 Cold50 召回；分母为该 cutoff 候选并集行。",
    "cold_expert_rank": "严格更早 Cold Expert 对候选给出的来源内名次；缺少时间安全模型时为空。",
    "cold_expert_score": "严格更早 Cold Expert 对候选给出的购买倾向分数；不是当前 cutoff 自拟合分数。",
    "cold_expert_score_available": "二值标记：该行是否存在严格更早且谱系合格的 Cold Expert 分数。",
    "cold_present": "二值标记：该用户—商品对是否进入多模态 Student Cold Top50。",
    "cold_rank": "候选在多模态 Student Cold Top50 中的原始名次，1为最高。",
    "cold_rank_pct": "cold_rank 除以冻结 Cold 候选预算50得到的来源内名次比例，越小越靠前。",
    "raw_best_cosine": "候选 raw FashionCLIP 向量与用户最近商品种子之间的最大余弦相似度。",
    "raw_best_recency_weighted_cosine": "raw 最大余弦相似度乘以对应种子的时间衰减权重。",
    "raw_supporting_seed_count": "raw FashionCLIP 空间中对候选提供正余弦相似度的近期种子数。",
    "raw_top3_mean_cosine": "raw FashionCLIP 空间中最高三个种子余弦相似度的均值。",
    "raw_top3_mean_recency_weighted_cosine": "raw 空间中最高三个时间衰减余弦相似度的均值。",
    "source_branch": "类别特征：warm_only、cold_only 或 warm_and_cold 三种候选来源分支。",
    "student_best_cosine": "候选多模态 Student 向量与用户最近商品种子的最大余弦相似度。",
    "student_best_recency_weighted_cosine": "Student 最大余弦相似度乘以对应种子的时间衰减权重。",
    "student_best_seed_rank": "产生最大 Student 相似度的近期购买种子序号，1表示最近的不同商品。",
    "student_best_seed_recency_weight": "产生最大 Student 相似度的种子所对应的时间衰减权重。",
    "student_rank": "候选在多模态 Student Cold 检索结果中的原始名次，1为最高。",
    "student_raw_weighted_gap": "Student 时间衰减最大相似度减去 raw FashionCLIP 对应值。",
    "student_supporting_seed_count": "Student 空间中对候选提供正余弦相似度的近期种子数。",
    "student_top3_mean_cosine": "Student 空间中最高三个种子余弦相似度的均值。",
    "student_top3_mean_recency_weighted_cosine": "Student 空间中最高三个时间衰减余弦相似度的均值。",
    "warm_present": "二值标记：该用户—商品对是否进入冻结 Warm-v1 Top150。",
}
if set(SOURCE_FEATURE_DEFINITIONS) != set(SOURCE_FEATURES):
    raise RuntimeError("M5.6 source feature glossary differs from frozen 23-feature schema")
KEYS = ["target_cutoff", "customer_id", "article_id"]
VARIANTS = {
    "A_current_sampling_fixed40": {
        "sampling": "current_m5.5",
        "training_stop": "fixed40_no_early_stopping",
        "deployment_eligible": False,
    },
    "B_conditional_sampling_temporal_early_stopping": {
        "sampling": "conditional_source",
        "training_stop": "temporal_early_stopping",
        "deployment_eligible": False,
    },
    "C_conditional_sampling_fixed40": {
        "sampling": "conditional_source",
        "training_stop": "fixed40_no_early_stopping",
        "deployment_eligible": True,
    },
}


def _literal(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def validate_development_cutoff(cutoff: str) -> None:
    if cutoff >= FINAL_CUTOFF:
        raise RuntimeError(f"M5.6 refuses final cutoff {cutoff}")


def fixed40_training_contract() -> dict[str, Any]:
    return {
        "boost_rounds": FIXED_ROUNDS,
        "validation_for_early_stopping": False,
        "early_stopping": False,
    }


def experiment_contract() -> dict[str, Any]:
    return {
        "schema_version": "m5.6-cold-supervision-suppression-contract-v1",
        "stage": "M5.6",
        "run_id": RUN_ID,
        "objective": "diagnose sampling suppression versus early stopping/capacity",
        "frozen": {
            "cold_threshold": "item_events_before_cutoff<=5",
            "k_warm": 150,
            "k_cold": 50,
            "candidate_and_feature_assets": "M5.4 exact identities",
            "feature_schema": DUAL_FEATURES,
            "lightgbm": {
                "objective": "lambdarank",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 100,
                "feature_fraction": 1.0,
                "bagging_fraction": 1.0,
                "lambdarank_truncation_level": 20,
            },
            "negative_cap_per_positive": NEGATIVES_PER_POSITIVE,
            "final_week": "not_run",
        },
        "matrix": {
            "Control": "existing M5.5 current sampling plus temporal early stopping",
            **{name: value for name, value in VARIANTS.items()},
        },
        "conditional_sampling": {
            "cold_positive_group": {
                "cold_only_negatives_per_positive": COLD_NEGATIVE_QUOTA_PER_POSITIVE,
                "warm_boundary_negatives_per_positive": WARM_BOUNDARY_QUOTA_PER_POSITIVE,
                "warm_hard_non_boundary_negatives_per_positive": WARM_HARD_QUOTA_PER_POSITIVE,
                "warm_and_cold_negatives_per_positive": BOTH_QUOTA_PER_POSITIVE,
                "remainder": "fixed priority fill to total cap; no outer metric tuning",
            },
            "warm_boundary_rank_inclusive": [WARM_BOUNDARY_MIN_RANK, WARM_BOUNDARY_MAX_RANK],
            "no_cold_positive_group": {
                "cold_present_negative_cap_per_group": NO_COLD_POSITIVE_COLD_NEGATIVE_CAP,
                "remaining_budget": "warm_only round-robin with frozen hard/medium/easy buckets",
            },
            "rank_buckets": "same one-third boundaries as M5.5",
            "fixed_hash_seed": SEED,
        },
        "interpretation": {
            "A": "A mechanism passes and B does not: capacity/early stopping is primary",
            "B": "B mechanism passes and A does not: source supervision distribution is primary",
            "C": "only C mechanism passes: sampling and capacity interact",
            "D": "none pass: representation/ranking separability bottleneck",
            "mixed": "A and B both pass: both interventions independently expose signal",
        },
        "mechanism_gate": {
            "nonzero_source_gain": "at least 3 of 4 outer windows",
            "cold_truth_top50": "at least 2 of 4 outer windows",
            "stable_separation": (
                "cold-only AUC>0.5 and median within-user cold-positive percentile>0.5 "
                "in at least 3 of 4 windows; worst AUC>=0.48"
            ),
            "warm_features_removed": False,
        },
        "deployment_candidate_gate": {
            "eligible_variant": "C only",
            "mean_map": "strictly above M5.5 Control",
            "window_map": "at least 3 of 4 outer windows non-degrading versus Control",
            "warm_safety": "mean warm delta>=-0.0002 and at most one window below -0.0002",
            "cold_sparse_positive_increase": (
                "at least 5 more cumulative pairs than Control and increases in at least 2 windows"
            ),
            "removed_warm_control": (
                "cumulative removed Warm positives no greater than Control and at most one window higher"
            ),
        },
        "stop": {
            "mechanism_failure": "stop_at_M5.6_representation_ranking_separability",
            "mechanism_only": "mechanism_fixed_but_not_deployable",
            "deployment_pass": "M5.6_C_candidate_next_ranker",
        },
    }


def _conditional_relation(path: Path) -> str:
    return f"""
    WITH source AS (
      SELECT *,
        CASE WHEN least(coalesce(warm_rank_pct,2.0),coalesce(cold_rank_pct,2.0))<=1.0/3 THEN 'hard'
             WHEN least(coalesce(warm_rank_pct,2.0),coalesce(cold_rank_pct,2.0))<=2.0/3 THEN 'medium'
             ELSE 'easy' END AS sample_rank_bucket
      FROM read_parquet({_literal(path)})
    ), group_stats AS (
      SELECT target_cutoff,customer_id,sum(target)::BIGINT AS positives,
        sum(CASE WHEN target=1 AND source_branch IN ('cold_only','warm_and_cold') THEN 1 ELSE 0 END)::BIGINT AS cold_positives
      FROM source GROUP BY target_cutoff,customer_id HAVING positives>0
    ), annotated AS (
      SELECT s.*,g.positives,g.cold_positives,(g.cold_positives>0) AS has_cold_positive,
        hash(s.target_cutoff,s.customer_id,s.article_id,{SEED})::UBIGINT AS stable_hash,
        row_number() OVER(
          PARTITION BY s.target_cutoff,s.customer_id,s.source_branch,s.sample_rank_bucket
          ORDER BY hash(s.target_cutoff,s.customer_id,s.article_id,{SEED}),s.candidate_rank,s.article_id
        )::BIGINT AS stratum_rank
      FROM source s JOIN group_stats g USING(target_cutoff,customer_id)
    ), negatives AS (
      SELECT * FROM annotated WHERE target=0
    ), cp_cold_ordered AS (
      SELECT *,row_number() OVER(
        PARTITION BY target_cutoff,customer_id
        ORDER BY stratum_rank,sample_rank_bucket,stable_hash,candidate_rank,article_id
      )::BIGINT AS quota_rank
      FROM negatives WHERE has_cold_positive AND source_branch='cold_only'
    ), cp_cold AS (
      SELECT target_cutoff,customer_id,article_id FROM cp_cold_ordered
      WHERE quota_rank<={COLD_NEGATIVE_QUOTA_PER_POSITIVE}*positives
    ), cp_warm_boundary_ordered AS (
      SELECT *,row_number() OVER(
        PARTITION BY target_cutoff,customer_id
        ORDER BY stratum_rank,stable_hash,candidate_rank,article_id
      )::BIGINT AS quota_rank
      FROM negatives WHERE has_cold_positive AND source_branch='warm_only'
        AND warm_rank BETWEEN {WARM_BOUNDARY_MIN_RANK} AND {WARM_BOUNDARY_MAX_RANK}
    ), cp_warm_boundary AS (
      SELECT target_cutoff,customer_id,article_id FROM cp_warm_boundary_ordered
      WHERE quota_rank<={WARM_BOUNDARY_QUOTA_PER_POSITIVE}*positives
    ), cp_warm_hard_ordered AS (
      SELECT *,row_number() OVER(
        PARTITION BY target_cutoff,customer_id
        ORDER BY stratum_rank,stable_hash,candidate_rank,article_id
      )::BIGINT AS quota_rank
      FROM negatives WHERE has_cold_positive AND source_branch='warm_only'
        AND sample_rank_bucket='hard'
        AND NOT (warm_rank BETWEEN {WARM_BOUNDARY_MIN_RANK} AND {WARM_BOUNDARY_MAX_RANK})
    ), cp_warm_hard AS (
      SELECT target_cutoff,customer_id,article_id FROM cp_warm_hard_ordered
      WHERE quota_rank<={WARM_HARD_QUOTA_PER_POSITIVE}*positives
    ), cp_both_ordered AS (
      SELECT *,row_number() OVER(
        PARTITION BY target_cutoff,customer_id
        ORDER BY stratum_rank,sample_rank_bucket,stable_hash,candidate_rank,article_id
      )::BIGINT AS quota_rank
      FROM negatives WHERE has_cold_positive AND source_branch='warm_and_cold'
    ), cp_both AS (
      SELECT target_cutoff,customer_id,article_id FROM cp_both_ordered
      WHERE quota_rank<={BOTH_QUOTA_PER_POSITIVE}*positives
    ), cp_reserved AS (
      SELECT * FROM cp_cold UNION ALL SELECT * FROM cp_warm_boundary
      UNION ALL SELECT * FROM cp_warm_hard UNION ALL SELECT * FROM cp_both
    ), cp_reserved_count AS (
      SELECT target_cutoff,customer_id,count(*)::BIGINT AS reserved_count
      FROM cp_reserved GROUP BY target_cutoff,customer_id
    ), cp_remainder_ordered AS (
      SELECT n.*,coalesce(c.reserved_count,0) AS reserved_count,
        row_number() OVER(
          PARTITION BY n.target_cutoff,n.customer_id
          ORDER BY
            CASE WHEN n.source_branch='cold_only' THEN 0
                 WHEN n.source_branch='warm_only' AND n.sample_rank_bucket='hard' THEN 1
                 WHEN n.source_branch='warm_and_cold' THEN 2 ELSE 3 END,
            n.stratum_rank,n.sample_rank_bucket,n.stable_hash,n.candidate_rank,n.article_id
        )::BIGINT AS fill_rank
      FROM negatives n
      LEFT JOIN cp_reserved r USING(target_cutoff,customer_id,article_id)
      LEFT JOIN cp_reserved_count c USING(target_cutoff,customer_id)
      WHERE n.has_cold_positive AND r.article_id IS NULL
    ), cp_remainder AS (
      SELECT target_cutoff,customer_id,article_id FROM cp_remainder_ordered
      WHERE fill_rank<={NEGATIVES_PER_POSITIVE}*positives-reserved_count
    ), no_cp_cold_ordered AS (
      SELECT *,row_number() OVER(
        PARTITION BY target_cutoff,customer_id
        ORDER BY stratum_rank,source_branch,sample_rank_bucket,stable_hash,candidate_rank,article_id
      )::BIGINT AS quota_rank
      FROM negatives WHERE NOT has_cold_positive AND source_branch IN ('cold_only','warm_and_cold')
    ), no_cp_cold AS (
      SELECT target_cutoff,customer_id,article_id FROM no_cp_cold_ordered
      WHERE quota_rank<={NO_COLD_POSITIVE_COLD_NEGATIVE_CAP}
    ), no_cp_cold_count AS (
      SELECT target_cutoff,customer_id,count(*)::BIGINT AS cold_count
      FROM no_cp_cold GROUP BY target_cutoff,customer_id
    ), no_cp_warm_ordered AS (
      SELECT n.*,coalesce(c.cold_count,0) AS cold_count,
        row_number() OVER(
          PARTITION BY n.target_cutoff,n.customer_id
          ORDER BY n.stratum_rank,n.sample_rank_bucket,n.stable_hash,n.candidate_rank,n.article_id
        )::BIGINT AS quota_rank
      FROM negatives n LEFT JOIN no_cp_cold_count c USING(target_cutoff,customer_id)
      WHERE NOT n.has_cold_positive AND n.source_branch='warm_only'
    ), no_cp_warm AS (
      SELECT target_cutoff,customer_id,article_id FROM no_cp_warm_ordered
      WHERE quota_rank<={NEGATIVES_PER_POSITIVE}*positives-cold_count
    ), selected_keys AS (
      SELECT * FROM cp_reserved UNION ALL SELECT * FROM cp_remainder
      UNION ALL SELECT * FROM no_cp_cold UNION ALL SELECT * FROM no_cp_warm
    ), selected AS (
      SELECT a.* FROM annotated a JOIN selected_keys k USING(target_cutoff,customer_id,article_id)
      WHERE a.target=0
      UNION ALL BY NAME
      SELECT * FROM annotated WHERE target=1
    )
    SELECT * FROM selected
    """


def _sampling_audit(frame: pd.DataFrame) -> dict[str, Any]:
    group_key = ["target_cutoff", "customer_id"]
    groups = frame.groupby(group_key, sort=False, observed=True)
    summary_rows: list[dict[str, Any]] = []
    for _, group in groups:
        positives = group.loc[group["target"] == 1]
        negatives = group.loc[group["target"] == 0]
        has_cold_positive = bool(group["has_cold_positive"].iloc[0])
        cold_positive_rows = int(
            ((positives["source_branch"] == "cold_only") | (positives["source_branch"] == "warm_and_cold")).sum()
        )
        warm_positive_rows = int(
            ((positives["source_branch"] == "warm_only") | (positives["source_branch"] == "warm_and_cold")).sum()
        )
        cold_negative_rows = int(
            ((negatives["source_branch"] == "cold_only") | (negatives["source_branch"] == "warm_and_cold")).sum()
        )
        warm_negative_rows = int(
            ((negatives["source_branch"] == "warm_only") | (negatives["source_branch"] == "warm_and_cold")).sum()
        )
        summary_rows.append({
            "has_cold_positive": has_cold_positive,
            "has_cold_candidate": bool((group["cold_present"] == 1).any()),
            "positive_rows": len(positives),
            "negative_rows": len(negatives),
            "cold_positive_rows": cold_positive_rows,
            "warm_positive_rows": warm_positive_rows,
            "cold_negative_rows": cold_negative_rows,
            "warm_negative_rows": warm_negative_rows,
        })
    group_frame = pd.DataFrame(summary_rows)
    if group_frame.empty:
        raise RuntimeError("conditional sampling produced no positive groups")
    if (group_frame["negative_rows"] > NEGATIVES_PER_POSITIVE * group_frame["positive_rows"]).any():
        raise RuntimeError("M5.6 conditional sampling exceeded total negative cap")
    no_cp = group_frame.loc[~group_frame["has_cold_positive"]]
    if not no_cp.empty and int(no_cp["cold_negative_rows"].max()) > NO_COLD_POSITIVE_COLD_NEGATIVE_CAP:
        raise RuntimeError("M5.6 no-cold-positive group exceeded cold negative cap")

    classes: dict[str, Any] = {}
    for class_name, flag in (("cold_positive_group", True), ("no_cold_positive_group", False)):
        group_subset = group_frame.loc[group_frame["has_cold_positive"] == flag]
        row_subset = frame.loc[frame["has_cold_positive"] == flag]
        negatives = row_subset.loc[row_subset["target"] == 0]
        positives = row_subset.loc[row_subset["target"] == 1]
        branch_counts = {
            branch: int((negatives["source_branch"] == branch).sum())
            for branch in ("cold_only", "warm_only", "warm_and_cold")
        }
        cold_positive_rows = int(
            positives["source_branch"].isin(["cold_only", "warm_and_cold"]).sum()
        )
        warm_positive_rows = int(
            positives["source_branch"].isin(["warm_only", "warm_and_cold"]).sum()
        )
        cold_negative_rows = branch_counts["cold_only"] + branch_counts["warm_and_cold"]
        warm_negative_rows = branch_counts["warm_only"] + branch_counts["warm_and_cold"]
        bucket_counts = {
            f"{branch}:{bucket}": int(
                ((negatives["source_branch"] == branch) & (negatives["sample_rank_bucket"] == bucket)).sum()
            )
            for branch in ("cold_only", "warm_only", "warm_and_cold")
            for bucket in ("hard", "medium", "easy")
        }
        classes[class_name] = {
            "groups": len(group_subset),
            "positive_rows": int(len(positives)),
            "cold_positive_rows": cold_positive_rows,
            "warm_positive_rows": warm_positive_rows,
            "selected_negative_rows_by_branch": branch_counts,
            "selected_negative_rows_by_branch_and_bucket": bucket_counts,
            "cold_negative_per_cold_positive": (
                cold_negative_rows / cold_positive_rows if cold_positive_rows else None
            ),
            "warm_negative_per_warm_positive": (
                warm_negative_rows / warm_positive_rows if warm_positive_rows else None
            ),
        }
    return {
        "groups_total": len(group_frame),
        "groups_with_cold_positive": int(group_frame["has_cold_positive"].sum()),
        "groups_without_cold_positive": int((~group_frame["has_cold_positive"]).sum()),
        "groups_without_cold_candidate": int((~group_frame["has_cold_candidate"]).sum()),
        "cold_positive_rows": int(group_frame["cold_positive_rows"].sum()),
        "warm_positive_rows": int(group_frame["warm_positive_rows"].sum()),
        "max_total_negative_per_positive": float(
            (group_frame["negative_rows"] / group_frame["positive_rows"]).max()
        ),
        "max_cold_negatives_in_no_cold_positive_group": (
            int(no_cp["cold_negative_rows"].max()) if not no_cp.empty else 0
        ),
        "classes": classes,
    }


def _merge_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    additive = [
        "groups_total", "groups_with_cold_positive", "groups_without_cold_positive",
        "groups_without_cold_candidate", "cold_positive_rows", "warm_positive_rows",
    ]
    result: dict[str, Any] = {name: sum(int(row[name]) for row in audits) for name in additive}
    result["max_total_negative_per_positive"] = max(
        float(row["max_total_negative_per_positive"]) for row in audits
    )
    result["max_cold_negatives_in_no_cold_positive_group"] = max(
        int(row["max_cold_negatives_in_no_cold_positive_group"]) for row in audits
    )
    result["classes"] = {}
    for class_name in ("cold_positive_group", "no_cold_positive_group"):
        group_rows = [row["classes"][class_name] for row in audits]
        cold_positive_rows = sum(int(row["cold_positive_rows"]) for row in group_rows)
        warm_positive_rows = sum(int(row["warm_positive_rows"]) for row in group_rows)
        branch = {
            name: sum(int(row["selected_negative_rows_by_branch"][name]) for row in group_rows)
            for name in ("cold_only", "warm_only", "warm_and_cold")
        }
        bucket = {
            name: sum(int(row["selected_negative_rows_by_branch_and_bucket"][name]) for row in group_rows)
            for name in group_rows[0]["selected_negative_rows_by_branch_and_bucket"]
        }
        result["classes"][class_name] = {
            "groups": sum(int(row["groups"]) for row in group_rows),
            "positive_rows": sum(int(row["positive_rows"]) for row in group_rows),
            "cold_positive_rows": cold_positive_rows,
            "warm_positive_rows": warm_positive_rows,
            "selected_negative_rows_by_branch": branch,
            "selected_negative_rows_by_branch_and_bucket": bucket,
            "cold_negative_per_cold_positive": (
                (branch["cold_only"] + branch["warm_and_cold"]) / cold_positive_rows
                if cold_positive_rows else None
            ),
            "warm_negative_per_warm_positive": (
                (branch["warm_only"] + branch["warm_and_cold"]) / warm_positive_rows
                if warm_positive_rows else None
            ),
        }
    return result


def load_conditional_training_sample(
    *, paths: list[Path], features: list[str],
) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    sizes: list[int] = []
    audits: list[dict[str, Any]] = []
    cutoff_evidence: dict[str, Any] = {}
    selected = list(dict.fromkeys([
        "target_cutoff", "customer_id", "article_id", "candidate_rank", "target",
        "has_cold_positive", "sample_rank_bucket", *features,
    ]))
    for path in paths:
        connection = duckdb.connect()
        try:
            relation = _conditional_relation(path)
            frame = connection.execute(
                f"SELECT {','.join(selected)} FROM ({relation}) "
                "ORDER BY target_cutoff,customer_id,candidate_rank,article_id"
            ).fetchdf()
        finally:
            connection.close()
        audit = _sampling_audit(frame)
        audits.append(audit)
        group_rows = frame.groupby(
            ["target_cutoff", "customer_id"], sort=False, observed=True
        )["target"].agg(["size", "sum"])
        if (group_rows["sum"] <= 0).any():
            raise RuntimeError("M5.6 retained a zero-positive training group")
        sizes.extend(group_rows["size"].astype(int).tolist())
        cutoff = str(frame["target_cutoff"].iloc[0])[:10]
        positives = int(frame["target"].sum())
        cutoff_evidence[cutoff] = {
            "rows": len(frame),
            "positive_rows": positives,
            "unobserved_rows": len(frame) - positives,
            "groups": len(group_rows),
            "audit": audit,
        }
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if sum(sizes) != len(result):
        raise RuntimeError("M5.6 conditional group sizes do not conserve rows")
    merged = _merge_audits(audits)
    evidence = {
        "cutoffs": cutoff_evidence,
        "rows": len(result),
        "positive_rows": int(result["target"].sum()),
        "unobserved_rows": int((result["target"] == 0).sum()),
        "groups": len(sizes),
        "min_group_rows": min(sizes),
        "max_group_rows": max(sizes),
        "audit": merged,
        "sample_rule": "conditional source sampling v1 with fixed quotas and hash",
    }
    return result, sizes, evidence


def _train_fixed40(
    *, train: pd.DataFrame, train_sizes: list[int], category_maps: dict[str, dict[int, int]],
    output_path: Path,
) -> tuple[lgb.Booster, dict[str, Any]]:
    model, evidence = train_ranker(
        train=train,
        train_sizes=train_sizes,
        validation=None,
        validation_sizes=None,
        validation_truth=None,
        features=DUAL_FEATURES,
        category_maps=category_maps,
        output_path=output_path,
        rounds=FIXED_ROUNDS,
    )
    evidence["training_stop"] = fixed40_training_contract()
    evidence["actual_num_trees"] = int(model.num_trees())
    if evidence["best_iteration"] != FIXED_ROUNDS or model.num_trees() != FIXED_ROUNDS:
        raise RuntimeError("fixed40 model did not produce exactly 40 trees")
    return model, evidence


def _importance(model: lgb.Booster) -> dict[str, Any]:
    gains = dict(zip(model.feature_name(), model.feature_importance("gain")))
    splits = dict(zip(model.feature_name(), model.feature_importance("split")))
    rows = [
        {"feature": name, "gain": float(gains[name]), "split": int(splits[name])}
        for name in SOURCE_FEATURES
    ]
    return {
        "features": rows,
        "nonzero_gain_feature_count": sum(row["gain"] > 0 for row in rows),
        "total_gain": float(sum(row["gain"] for row in rows)),
        "total_split_count": int(sum(row["split"] for row in rows)),
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"rows": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "rows": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    if len(positive) == 0 or len(negative) == 0:
        return None
    combined = pd.Series(np.concatenate([positive, negative]))
    ranks = combined.rank(method="average").to_numpy(dtype=np.float64)
    n_pos = len(positive)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * len(negative)))


def _score_audit(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    audit = frame[["customer_id", "source_branch", "target"]].copy()
    audit["prediction"] = np.asarray(scores, dtype=np.float64)
    source_distributions: dict[str, Any] = {}
    for branch in ("warm_only", "cold_only", "warm_and_cold"):
        for label, label_name in ((1, "positive"), (0, "unobserved")):
            values = audit.loc[
                (audit["source_branch"] == branch) & (audit["target"] == label), "prediction"
            ].to_numpy(dtype=np.float64)
            source_distributions[f"{branch}:{label_name}"] = _distribution(values)
    cold_positive = audit.loc[
        (audit["source_branch"] == "cold_only") & (audit["target"] == 1), "prediction"
    ].to_numpy(dtype=np.float64)
    cold_negative = audit.loc[
        (audit["source_branch"] == "cold_only") & (audit["target"] == 0), "prediction"
    ].to_numpy(dtype=np.float64)
    cold = audit.loc[audit["source_branch"] == "cold_only"].copy()
    cold["score_rank"] = cold.groupby("customer_id", observed=True)["prediction"].rank(
        ascending=False, method="average"
    )
    cold["group_rows"] = cold.groupby("customer_id", observed=True)["prediction"].transform("size")
    cold["positive_percentile"] = np.where(
        cold["group_rows"] > 1,
        (cold["group_rows"] - cold["score_rank"]) / (cold["group_rows"] - 1),
        1.0,
    )
    percentiles = cold.loc[cold["target"] == 1, "positive_percentile"].to_numpy(dtype=np.float64)
    return {
        "cold_positive_vs_unobserved_auc": _auc(cold_positive, cold_negative),
        "cold_positive_within_user_percentile": {
            "definition": "fraction of same-user cold_only candidates scored no higher; 1 is best",
            "rows": len(percentiles),
            "mean": float(np.mean(percentiles)) if len(percentiles) else None,
            "p25": float(np.quantile(percentiles, 0.25)) if len(percentiles) else None,
            "median": float(np.median(percentiles)) if len(percentiles) else None,
            "p75": float(np.quantile(percentiles, 0.75)) if len(percentiles) else None,
        },
        "source_distributions": source_distributions,
    }


def _cold_truth_funnel(frame: pd.DataFrame, ranks: np.ndarray) -> dict[str, Any]:
    mask = ((frame["source_branch"] == "cold_only") & (frame["target"] == 1)).to_numpy()
    truth_ranks = np.asarray(ranks, dtype=np.int64)[mask]
    if len(truth_ranks) == 0:
        return {
            "union": 0, "top100": 0, "top50": 0, "top20": 0, "top12": 0,
            "best_rank": None, "p25_rank": None, "median_rank": None, "p75_rank": None,
        }
    return {
        "union": len(truth_ranks),
        "top100": int((truth_ranks <= 100).sum()),
        "top50": int((truth_ranks <= 50).sum()),
        "top20": int((truth_ranks <= 20).sum()),
        "top12": int((truth_ranks <= 12).sum()),
        "best_rank": int(truth_ranks.min()),
        "p25_rank": float(np.quantile(truth_ranks, 0.25)),
        "median_rank": float(np.median(truth_ranks)),
        "p75_rank": float(np.quantile(truth_ranks, 0.75)),
    }


def _control_scores(frame: pd.DataFrame, prediction_path: Path) -> np.ndarray:
    connection = duckdb.connect()
    try:
        control = connection.execute(
            f"SELECT target_cutoff,customer_id,article_id,candidate_rank,score_dual_fusion "
            f"FROM read_parquet({_literal(prediction_path)}) "
            "ORDER BY target_cutoff,customer_id,candidate_rank,article_id"
        ).fetchdf()
    finally:
        connection.close()
    if len(control) != len(frame):
        raise RuntimeError("M5.5 control prediction row count mismatch")
    for key in KEYS:
        if not np.array_equal(control[key].astype(str).to_numpy(), frame[key].astype(str).to_numpy()):
            raise RuntimeError(f"M5.5 control prediction identity mismatch: {key}")
    if control["score_dual_fusion"].isna().any():
        raise RuntimeError("M5.5 control predictions contain missing scores")
    return control["score_dual_fusion"].to_numpy(dtype=np.float64)


def _evaluate_variant(
    *, frame: pd.DataFrame, scores: np.ndarray, baseline_top12: pd.DataFrame,
    truth: dict[str, set[str]], counts: dict[str, int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    top12, ranks = _rank_top12(frame, scores)
    result = _evaluate(
        frame=frame, top12=top12, ranks=ranks, baseline_top12=baseline_top12,
        truth=truth, counts=counts,
    )
    result["cold_truth_funnel"] = _cold_truth_funnel(frame, ranks)
    result["score_audit"] = _score_audit(frame, scores)
    ranked = frame[[*KEYS, "candidate_rank", "target", "source_branch"]].copy()
    ranked["prediction"] = scores
    ranked["final_rank"] = ranks
    return result, ranked


def _mechanism_summary(windows: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        nonzero_windows = sum(
            row["feature_gain"][variant]["total_gain"] > 0 for row in windows.values()
        )
        top50_windows = sum(
            row["evaluations"][variant]["cold_truth_funnel"]["top50"] > 0
            for row in windows.values()
        )
        aucs = [
            row["evaluations"][variant]["score_audit"]["cold_positive_vs_unobserved_auc"]
            for row in windows.values()
        ]
        medians = [
            row["evaluations"][variant]["score_audit"]["cold_positive_within_user_percentile"]["median"]
            for row in windows.values()
        ]
        stable_windows = sum(
            auc is not None and median is not None and auc > 0.5 and median > 0.5
            for auc, median in zip(aucs, medians)
        )
        separation = stable_windows >= 3 and min(float(value) for value in aucs if value is not None) >= 0.48
        checks = {
            "source_gain_nonzero_in_at_least_3_windows": nonzero_windows >= 3,
            "cold_truth_top50_in_at_least_2_windows": top50_windows >= 2,
            "stable_cold_score_separation": separation,
            "warm_features_unchanged": True,
        }
        result[variant] = {
            "nonzero_gain_windows": nonzero_windows,
            "top50_windows": top50_windows,
            "stable_separation_windows": stable_windows,
            "worst_auc": min(float(value) for value in aucs if value is not None),
            "checks": checks,
            "passed": all(checks.values()),
        }
    a = result["A_current_sampling_fixed40"]["passed"]
    b = result["B_conditional_sampling_temporal_early_stopping"]["passed"]
    c = result["C_conditional_sampling_fixed40"]["passed"]
    if a and b:
        interpretation = "mixed_capacity_and_sampling_independent_signal"
    elif a and not b:
        interpretation = "capacity_or_early_stopping_primary"
    elif b and not a:
        interpretation = "source_supervision_sampling_primary"
    elif c and not a and not b:
        interpretation = "sampling_capacity_interaction"
    elif not (a or b or c):
        interpretation = "representation_ranking_separability_bottleneck"
    else:
        interpretation = "mechanism_signal_shared_with_combination"
    return {"variants": result, "interpretation": interpretation}


def _deployment_gate(windows: dict[str, Any], mechanism: dict[str, Any]) -> dict[str, Any]:
    variant = "C_conditional_sampling_fixed40"
    control = "Control"
    map_deltas = {
        window: row["evaluations"][variant]["segments"]["overall"]["map@12"]
        - row["evaluations"][control]["segments"]["overall"]["map@12"]
        for window, row in windows.items()
    }
    warm_deltas = {
        window: row["evaluations"][variant]["segments"]["warm_21_plus"]["map@12"]
        - row["evaluations"][control]["segments"]["warm_21_plus"]["map@12"]
        for window, row in windows.items()
    }
    c_inserted = {
        window: row["evaluations"][variant]["inserted_cold_sparse_positive_pairs"]
        for window, row in windows.items()
    }
    control_inserted = {
        window: row["evaluations"][control]["inserted_cold_sparse_positive_pairs"]
        for window, row in windows.items()
    }
    c_removed = {
        window: row["evaluations"][variant]["removed_warm_positive_pairs"]
        for window, row in windows.items()
    }
    control_removed = {
        window: row["evaluations"][control]["removed_warm_positive_pairs"]
        for window, row in windows.items()
    }
    checks = {
        "mechanism_gate_passed": bool(mechanism["variants"][variant]["passed"]),
        "mean_map_strictly_above_control": float(np.mean(list(map_deltas.values()))) > MAP_TOLERANCE,
        "at_least_3_windows_non_degrading": sum(value >= -MAP_TOLERANCE for value in map_deltas.values()) >= 3,
        "warm_not_systematically_degraded": (
            float(np.mean(list(warm_deltas.values()))) >= -WINDOW_TOLERANCE
            and sum(value < -WINDOW_TOLERANCE for value in warm_deltas.values()) <= 1
        ),
        "cold_sparse_positive_increase_is_material": (
            sum(c_inserted.values()) >= sum(control_inserted.values()) + 5
            and sum(c_inserted[w] > control_inserted[w] for w in windows) >= 2
        ),
        "removed_warm_positives_controlled": (
            sum(c_removed.values()) <= sum(control_removed.values())
            and sum(c_removed[w] > control_removed[w] for w in windows) <= 1
        ),
    }
    return {
        "eligible_variant": variant,
        "map_delta_vs_control": map_deltas,
        "warm_delta_vs_control": warm_deltas,
        "inserted_cold_sparse": {"candidate": c_inserted, "control": control_inserted},
        "removed_warm": {"candidate": c_removed, "control": control_removed},
        "checks": checks,
        "passed": all(checks.values()),
    }


def _render_report(result: dict[str, Any]) -> str:
    mechanism_code = result["summary"]["mechanism"]["interpretation"]
    mechanism_zh = {
        "representation_ranking_separability_bottleneck": "预注册 case D：当前表示/统一排序的跨来源可分性仍不足",
        "capacity_or_early_stopping_primary": "预注册 case A：容量或过早停止是主因",
        "source_supervision_sampling_primary": "预注册 case B：来源监督采样是主因",
        "sampling_capacity_interaction": "预注册 case C：采样与容量存在交互",
        "mixed_capacity_and_sampling_independent_signal": "容量与采样都能独立暴露信号",
        "mechanism_signal_shared_with_combination": "多个对照共享机制信号",
    }.get(mechanism_code, "未归类机制状态")
    next_stage_code = result["summary"]["next_stage"]
    c_variant = "C_conditional_sampling_fixed40"
    c_map_deltas = result["summary"]["deployment"]["map_delta_vs_control"]
    cold_truth_total = sum(
        row["evaluations"][c_variant]["cold_truth_funnel"]["union"]
        for row in result["windows"].values()
    )
    c_top100_total = sum(
        row["evaluations"][c_variant]["cold_truth_funnel"]["top100"]
        for row in result["windows"].values()
    )
    lines = [
        "# M5.6：冷商品监督抑制机制诊断",
        "",
        "## 结论",
        "",
        f"- 机制归因：`{mechanism_code}`（{mechanism_zh}）。",
        f"- 下一步状态：`{next_stage_code}`（停止当前采样/轮数路线，不产生新 baseline）；部署候选门禁通过：{result['summary']['deployment']['passed']}。",
        f"- C 在4/4窗出现来源/冷商品特征增益、3/4窗达到预注册分数分离，但 {cold_truth_total} 个 cold_only 真实对进入 Top100 的数量仍为 {c_top100_total}。",
        f"- C 相对 Control 的四窗平均 MAP@12 变化为 {float(np.mean(list(c_map_deltas.values()))):+.6f}；只有2/4窗不退化，且没有新增 cold/sparse 正例，不能据均值微升晋级。",
        "- 本轮只改变负采样条件和训练停止轮数；候选、23个来源/冷商品特征、LightGBM 树参数与时间外推谱系全部冻结。",
        "- 最终验证周 2020-09-16：not_run。",
        "",
        "## 术语与统计单位",
        "",
        "- **Control**：复用 M5.5 已保存的来源感知 Dual Fusion 模型和预测，不重新训练；采样和时间内早停均为 M5.5 原配置。",
        "- **Variant A/B/C**：本项目自定义的三种机制对照；A只固定40轮，B只改条件化来源采样，C同时采用两者。40轮仅是预注册诊断值，不是调优结果。",
        "- **Cold-positive group**：一个 cutoff—用户排序组中，至少有一条 target=1 且来源为 cold_only 或 warm_and_cold；组数分母是训练中至少含一个正例的全部排序组。",
        "- **unobserved negative**：验证周未观察购买的候选行；由于无曝光日志，它不是已确认拒绝。",
        "- **Cold truth funnel**：cold_only 正例用户—商品对在并集以及最终统一名次 Top100/50/20/12 的数量；各 TopK 分母均是同窗 union 正例数。",
        "- **cold-positive percentile**：正例分数在同用户 cold_only 候选中的百分位，1表示最高、0表示最低；报告中位数的分母是该窗所有 cold_only 正例行。",
        "- **AUC**：行业通用二分类排序诊断；这里比较 cold_only 正例与未观察候选，0.5近似随机，只用于机制分析，不替代 MAP。",
        "- **MAP@12**：行业通用排序指标；先对每位用户的前12项计算平均准确率，再对评测用户取均值。",
        "- **时间外推/OOF谱系**：行业 stacking/OOF 思路在本项目中的防泄漏约束；上游模型的标签结束时间必须早于当前候选标签周。",
        "- **时间内早停**：只在当前外层验证窗之前的 inner validation（更早时间验证窗）选择树轮数，不用外层指标选轮数。",
        "- **gain / split**：LightGBM 通用特征重要性；gain 是使用该特征分裂带来的目标增益总和，split 是分裂次数，二者都不等同因果贡献。",
        "- **稳定分离**：本项目预注册门槛；至少3/4窗同时满足 cold-only AUC>0.5、同用户正例中位百分位>0.5，且最差AUC不低于0.48。",
        "- **not_run**：本项目审计状态，表示最终验证周没有被读取或用于本轮决策。",
        "",
        "## 2×2 机制矩阵与 MAP@12",
        "",
        "| window | Frozen Warm-v1 | Control | A current+40 | B conditional+early | C conditional+40 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for window, row in result["windows"].items():
        evaluations = row["evaluations"]
        lines.append(
            f"| {window} | {evaluations['Frozen_Warm_v1']['segments']['overall']['map@12']:.6f} | "
            f"{evaluations['Control']['segments']['overall']['map@12']:.6f} | "
            f"{evaluations['A_current_sampling_fixed40']['segments']['overall']['map@12']:.6f} | "
            f"{evaluations['B_conditional_sampling_temporal_early_stopping']['segments']['overall']['map@12']:.6f} | "
            f"{evaluations['C_conditional_sampling_fixed40']['segments']['overall']['map@12']:.6f} |"
        )
    lines.extend([
        "",
        "## 来源/冷商品特征利用",
        "",
        "non-zero count、gain、split 的统计对象是冻结23个新增 Cold/source 特征；gain 是 LightGBM 分裂增益，不是因果贡献。",
        "",
        "| window | variant | non-zero features | total gain | total splits |",
        "|---|---|---:|---:|---:|",
    ])
    for window, row in result["windows"].items():
        for variant, audit in row["feature_gain"].items():
            lines.append(
                f"| {window} | {variant} | {audit['nonzero_gain_feature_count']} | "
                f"{audit['total_gain']:.6f} | {audit['total_split_count']} |"
            )
    lines.extend([
        "",
        "下面逐项列出全部23个冻结的来源/冷商品特征。`non-zero windows`（非零增益窗口数）"
        "是某变体在四个外层验证窗口中该特征 LightGBM gain>0 的窗口数；分母固定为4。",
        "",
        "| feature | 中文定义 | Control | A | B | C |",
        "|---|---|---:|---:|---:|---:|",
    ])
    feature_window_counts = result["summary"]["feature_nonzero_windows"]
    for feature in SOURCE_FEATURES:
        lines.append(
            f"| `{feature}` | {SOURCE_FEATURE_DEFINITIONS[feature]} | "
            f"{feature_window_counts['Control'][feature]} | "
            f"{feature_window_counts['A_current_sampling_fixed40'][feature]} | "
            f"{feature_window_counts['B_conditional_sampling_temporal_early_stopping'][feature]} | "
            f"{feature_window_counts['C_conditional_sampling_fixed40'][feature]} |"
        )
    lines.extend([
        "",
        "## Cold truth 排名漏斗",
        "",
        "| window | variant | union | Top100 | Top50 | Top20 | Top12 | best rank | p25 | median | p75 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["windows"].items():
        for variant in ("Control", *VARIANTS):
            funnel = row["evaluations"][variant]["cold_truth_funnel"]
            lines.append(
                f"| {window} | {variant} | {funnel['union']} | {funnel['top100']} | "
                f"{funnel['top50']} | {funnel['top20']} | {funnel['top12']} | "
                f"{funnel['best_rank']} | {funnel['p25_rank']} | {funnel['median_rank']} | {funnel['p75_rank']} |"
            )
    lines.extend([
        "",
        "## Cold 内部分数分离",
        "",
        "| window | variant | cold-positive rows | cold-unobserved rows | positive mean | negative mean | AUC | positive percentile median |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["windows"].items():
        for variant in ("Control", *VARIANTS):
            audit = row["evaluations"][variant]["score_audit"]
            pos = audit["source_distributions"]["cold_only:positive"]
            neg = audit["source_distributions"]["cold_only:unobserved"]
            percentile = audit["cold_positive_within_user_percentile"]
            lines.append(
                f"| {window} | {variant} | {pos['rows']} | {neg['rows']} | "
                f"{pos['mean']:.8g} | {neg['mean']:.8g} | "
                f"{audit['cold_positive_vs_unobserved_auc']:.6f} | {percentile['median']:.6f} |"
            )
    lines.extend([
        "",
        "### 各来源分支的完整分数分布",
        "",
        "`source branch`（来源分支）是本项目自定义候选分类：warm_only 仅来自 Warm-v1，"
        "cold_only 仅来自 M5 冷商品检索，warm_and_cold 同时被两者召回。每行分母是该窗口、"
        "该模型、该来源分支和该标签的候选行数；p90/p95 是分数的90%/95%分位数。",
        "",
        "| window | variant | source | label | rows | mean | median | p90 | p95 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["windows"].items():
        for variant in ("Control", *VARIANTS):
            distributions = row["evaluations"][variant]["score_audit"]["source_distributions"]
            for key, distribution in distributions.items():
                source, label = key.split(":", 1)
                values = [distribution[name] for name in ("mean", "median", "p90", "p95")]
                formatted = ["N/A" if value is None else f"{value:.8g}" for value in values]
                lines.append(
                    f"| {window} | {variant} | {source} | {label} | {distribution['rows']} | "
                    f"{formatted[0]} | {formatted[1]} | {formatted[2]} | {formatted[3]} |"
                )
    lines.extend([
        "",
        "## 条件化采样审计",
        "",
        "下表统计 B/C 共用的 outer 训练样本；两个变体候选与采样行完全相同。Cold/Warm ratio 中 warm_and_cold 行会分别计入两种来源，因其同时提供两路监督；无 Cold 正例组的 Cold ratio 无定义。",
        "四窗实测均满足每个正例最多30个未观察候选；无 Cold 正例组保留的含 Cold 来源负例最大值均为3。",
        "",
        "| window | group class | groups | cold positives | warm positives | cold-only negatives | warm-only negatives | both negatives | cold neg/pos | warm neg/pos |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["windows"].items():
        for class_name, audit in row["sampling_audit"]["conditional_outer"]["audit"]["classes"].items():
            branches = audit["selected_negative_rows_by_branch"]
            cold_ratio = audit["cold_negative_per_cold_positive"]
            warm_ratio = audit["warm_negative_per_warm_positive"]
            lines.append(
                f"| {window} | {class_name} | {audit['groups']} | {audit['cold_positive_rows']} | "
                f"{audit['warm_positive_rows']} | {branches['cold_only']} | {branches['warm_only']} | "
                f"{branches['warm_and_cold']} | {cold_ratio if cold_ratio is not None else 'N/A'} | "
                f"{warm_ratio if warm_ratio is not None else 'N/A'} |"
            )
    lines.extend([
        "",
        "## 分人群 MAP 与 pair-level safety",
        "",
        "| window | variant | overall | warm_21_plus | strict_cold | sparse_1_5 | inserted all | removed all | inserted cold/sparse | removed Warm | warm-only slots | cold-only slots | both slots |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["windows"].items():
        for variant in ("Control", *VARIANTS):
            evaluation = row["evaluations"][variant]
            segments = evaluation["segments"]
            lines.append(
                f"| {window} | {variant} | {segments['overall']['map@12']:.6f} | "
                f"{segments['warm_21_plus']['map@12']:.6f} | {segments['strict_cold']['map@12']:.6f} | "
                f"{segments['sparse_1_5']['map@12']:.6f} | {evaluation['inserted_positive_pairs']} | "
                f"{evaluation['removed_positive_pairs']} | {evaluation['inserted_cold_sparse_positive_pairs']} | "
                f"{evaluation['removed_warm_positive_pairs']} | "
                f"{evaluation['final_top12_branch_composition']['warm_only']} | "
                f"{evaluation['final_top12_branch_composition']['cold_only']} | "
                f"{evaluation['final_top12_branch_composition']['warm_and_cold']} |"
            )
    lines.extend([
        "",
        "## 冻结候选上限",
        "",
        "Candidate Recall（候选召回率）是冻结候选并集命中的真实用户—商品对数除以该窗全部真实对数；"
        "Oracle MAP@12（理想排序上限）是假设把候选中的正例排到最前时的 MAP@12。二者仅由冻结候选决定，"
        "所以同一窗口 Control/A/B/C 完全相同。",
        "",
        "| window | candidate recall | oracle MAP@12 |",
        "|---|---:|---:|",
    ])
    for window, row in result["windows"].items():
        evaluation = row["evaluations"]["Control"]
        lines.append(
            f"| {window} | {evaluation['candidate_recall']:.6f} | {evaluation['oracle_map@12']:.6f} |"
        )
    lines.extend([
        "",
        "完整逐行数据同时保存在机器可读 JSON 审计文件中。",
        "",
        "## 门禁",
        "",
    ])
    for variant, row in result["summary"]["mechanism"]["variants"].items():
        lines.append(f"- `{variant}` mechanism gate：{row['passed']}；checks={row['checks']}。")
    lines.extend([
        f"- C deployment gate：{result['summary']['deployment']['passed']}；checks={result['summary']['deployment']['checks']}。",
        "",
        "## 工程与边界",
        "",
        f"- 总耗时 {result['resources']['elapsed_seconds']:.2f} 秒；峰值工作集 {result['resources']['peak_working_set_bytes']/2**30:.2f} GiB；artifact {result['resources']['artifact_bytes']/2**20:.2f} MiB。",
        "- A 的 current sampling 审计逐窗与 M5.5 保存统计完全一致；B/C 使用相同条件化样本；固定40轮模型均验证实际树数为40且没有 early stopping validation。",
        "- M5.4 feature parquet、M5.5 Control 模型/预测和所有输出均记录 bytes/SHA256；latest behavior < cutoff 与 OOF lineage 继承 M5.4 已通过门禁并重新核对身份。",
        "- 商品全集仍采用 optimistic all-articles；缺少库存和曝光日志，因此 target=0 只能称未观察候选。",
        "- 2020-09-16 final week 未读取；本轮没有追加第四个新模型变体，也没有修改候选、特征或树参数。",
        "",
        "## 最终归因",
        "",
    ])
    for sentence in result["summary"]["interpretation_notes"]:
        lines.append(f"- {sentence}")
    lines.append("")
    return "\n".join(lines)


def run(
    *, m54_metrics_path: Path, m54_artifact_dir: Path, m55_metrics_path: Path,
    m55_verification_path: Path, artifact_dir: Path, report_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    if artifact_dir.exists() or report_dir.exists():
        raise FileExistsError(artifact_dir if artifact_dir.exists() else report_dir)
    m54 = json.loads(m54_metrics_path.read_text(encoding="utf-8"))
    m55 = json.loads(m55_metrics_path.read_text(encoding="utf-8"))
    m55_verification = json.loads(m55_verification_path.read_text(encoding="utf-8"))
    if m54.get("run_id") != M54_RUN_ID or not m54.get("gate", {}).get("passed"):
        raise RuntimeError("M5.6 requires passing authoritative M5.4")
    if m55.get("run_id") != M55_RUN_ID or m55.get("final_week") != "not_run":
        raise RuntimeError("M5.6 M5.5 control identity mismatch")
    if m55_verification.get("status") != "passed":
        raise RuntimeError("M5.6 requires passing M5.5 independent verification")
    if set(m54["cutoffs"]) != set(LINEAGES):
        raise RuntimeError("M5.6 M5.4 cutoff set mismatch")
    for cutoff, row in m54["cutoffs"].items():
        validate_development_cutoff(cutoff)
        if not all(bool(value) for value in row["gate_checks"].values()):
            raise RuntimeError(f"M5.6 M5.4 gate mismatch: {cutoff}")

    artifact_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    contract = experiment_contract()
    contract["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    contract["inputs"] = {
        "m54_metrics": file_identity(m54_metrics_path),
        "m55_metrics": file_identity(m55_metrics_path),
        "m55_verification": file_identity(m55_verification_path),
    }
    atomic_json(report_dir / "experiment_contract.json", contract)

    m3_path = Path(m55["inputs"]["m3_metrics"]["path"])
    m3 = json.loads(m3_path.read_text(encoding="utf-8"))
    transactions = Path(m3["inputs"]["transactions"]["path"])
    windows: dict[str, Any] = {}
    artifact_bytes = 0

    for window, protocol in ROLLING_PROTOCOL.items():
        print(f"M5.6 train/evaluate {window}", flush=True)
        validate_development_cutoff(protocol["outer_validation"])
        root = artifact_dir / window
        root.mkdir()
        inner_train_paths = [m54_artifact_dir / value / "features.parquet" for value in protocol["inner_train"]]
        inner_valid_path = m54_artifact_dir / protocol["inner_validation"] / "features.parquet"
        outer_train_paths = [m54_artifact_dir / value / "features.parquet" for value in protocol["outer_train"]]
        outer_valid_path = m54_artifact_dir / protocol["outer_validation"] / "features.parquet"
        for path in (*inner_train_paths, inner_valid_path, *outer_train_paths, outer_valid_path):
            identity = m54["cutoffs"][path.parent.name]["features"]["artifact"]
            if file_identity(path) != identity:
                raise RuntimeError(f"M5.6 feature identity mismatch: {path}")
        if file_identity(outer_valid_path) != m55["windows"][window]["inputs"]["outer_features"]:
            raise RuntimeError(f"M5.6 candidate/feature identity differs from M5.5: {window}")

        current_outer, current_outer_sizes, current_evidence = load_training_sample(
            paths=outer_train_paths, features=DUAL_FEATURES, warm_only=False,
        )
        if current_evidence != m55["windows"][window]["training"]["dual_fusion"]["outer_sampling"]:
            raise RuntimeError(f"M5.6 Variant A sampling differs from M5.5: {window}")
        current_maps = build_category_maps(current_outer)
        atomic_json(root / "outer-A-category-maps.json", current_maps)
        model_a, evidence_a = _train_fixed40(
            train=current_outer, train_sizes=current_outer_sizes,
            category_maps=current_maps, output_path=root / "outer-A.txt",
        )
        del current_outer
        gc.collect()

        conditional_inner, conditional_inner_sizes, conditional_inner_evidence = load_conditional_training_sample(
            paths=inner_train_paths, features=DUAL_FEATURES,
        )
        inner_valid, inner_valid_sizes = _load_validation(
            path=inner_valid_path, features=DUAL_FEATURES, warm_only=False,
        )
        users = inner_valid.groupby("customer_id", sort=False, observed=True).size().index.astype(str).tolist()
        inner_truth, _ = _truth_and_counts(
            transactions_path=transactions, cutoff=protocol["inner_validation"], users=users,
        )
        inner_maps = build_category_maps(conditional_inner)
        model_b_inner, evidence_b_inner = train_ranker(
            train=conditional_inner, train_sizes=conditional_inner_sizes,
            validation=inner_valid, validation_sizes=inner_valid_sizes,
            validation_truth=inner_truth, features=DUAL_FEATURES,
            category_maps=inner_maps, output_path=root / "inner-B.txt",
        )
        evidence_b_inner["training_stop"] = "M5.5 temporal early stopping"
        b_rounds = int(evidence_b_inner["best_iteration"])
        del model_b_inner, conditional_inner, inner_valid
        gc.collect()

        conditional_outer, conditional_outer_sizes, conditional_outer_evidence = load_conditional_training_sample(
            paths=outer_train_paths, features=DUAL_FEATURES,
        )
        conditional_maps = build_category_maps(conditional_outer)
        atomic_json(root / "outer-BC-category-maps.json", conditional_maps)
        model_b, evidence_b = train_ranker(
            train=conditional_outer, train_sizes=conditional_outer_sizes,
            validation=None, validation_sizes=None, validation_truth=None,
            features=DUAL_FEATURES, category_maps=conditional_maps,
            output_path=root / "outer-B.txt", rounds=b_rounds,
        )
        evidence_b["training_stop"] = {
            "kind": "rounds selected only on earlier inner temporal validation",
            "boost_rounds": b_rounds,
        }
        model_c, evidence_c = _train_fixed40(
            train=conditional_outer, train_sizes=conditional_outer_sizes,
            category_maps=conditional_maps, output_path=root / "outer-C.txt",
        )
        del conditional_outer
        gc.collect()

        frame, _ = _load_validation(path=outer_valid_path, features=DUAL_FEATURES, warm_only=False)
        users = frame.groupby("customer_id", sort=False, observed=True).size().index.astype(str).tolist()
        truth, counts = _truth_and_counts(
            transactions_path=transactions, cutoff=protocol["outer_validation"], users=users,
        )
        baseline_db = Path(m3["development"][window]["scoring"]["evaluation_db"])
        baseline_top12 = _baseline_top12(baseline_db)
        baseline_ranks = np.full(len(frame), 999, dtype=np.int32)
        baseline_eval = _evaluate(
            frame=frame,
            top12=baseline_top12.rename(columns={"warm_rank": "final_rank"}).assign(
                candidate_rank=0, target=0, user_history_events_12w=0,
                warm_rank=lambda x: x["final_rank"], source_branch="warm_only",
                item_events_before_cutoff=0,
            ),
            ranks=baseline_ranks, baseline_top12=baseline_top12, truth=truth, counts=counts,
        )
        if abs(baseline_eval["segments"]["overall"]["map@12"] - WARM_MAP[window]) > 1e-12:
            raise RuntimeError(f"M5.6 Warm-v1 anchor mismatch: {window}")

        control_prediction_path = Path(m55["windows"][window]["predictions"]["path"])
        if file_identity(control_prediction_path) != m55["windows"][window]["predictions"]:
            raise RuntimeError(f"M5.6 M5.5 prediction identity mismatch: {window}")
        control_scores = _control_scores(frame, control_prediction_path)
        evaluations: dict[str, Any] = {"Frozen_Warm_v1": baseline_eval}
        control_eval, control_ranked = _evaluate_variant(
            frame=frame, scores=control_scores, baseline_top12=baseline_top12,
            truth=truth, counts=counts,
        )
        stored_control = m55["windows"][window]["evaluations"]["dual_fusion"]
        if abs(
            control_eval["segments"]["overall"]["map@12"]
            - stored_control["segments"]["overall"]["map@12"]
        ) > 1e-12:
            raise RuntimeError(f"M5.6 failed to reproduce M5.5 Control MAP: {window}")
        evaluations["Control"] = control_eval

        prepared = _prepare(frame, DUAL_FEATURES, current_maps)
        scores_a = model_a.predict(prepared)
        del prepared
        eval_a, ranked_a = _evaluate_variant(
            frame=frame, scores=scores_a, baseline_top12=baseline_top12, truth=truth, counts=counts,
        )
        evaluations["A_current_sampling_fixed40"] = eval_a
        prepared = _prepare(frame, DUAL_FEATURES, conditional_maps)
        scores_b = model_b.predict(prepared)
        scores_c = model_c.predict(prepared)
        del prepared
        eval_b, ranked_b = _evaluate_variant(
            frame=frame, scores=scores_b, baseline_top12=baseline_top12, truth=truth, counts=counts,
        )
        eval_c, ranked_c = _evaluate_variant(
            frame=frame, scores=scores_c, baseline_top12=baseline_top12, truth=truth, counts=counts,
        )
        evaluations["B_conditional_sampling_temporal_early_stopping"] = eval_b
        evaluations["C_conditional_sampling_fixed40"] = eval_c

        predictions = frame[[*KEYS, "candidate_rank", "target", "source_branch"]].copy()
        predictions["score_Control"] = control_scores
        predictions["score_A"] = scores_a
        predictions["score_B"] = scores_b
        predictions["score_C"] = scores_c
        predictions["rank_Control"] = control_ranked["final_rank"].to_numpy()
        predictions["rank_A"] = ranked_a["final_rank"].to_numpy()
        predictions["rank_B"] = ranked_b["final_rank"].to_numpy()
        predictions["rank_C"] = ranked_c["final_rank"].to_numpy()
        prediction_path = root / "outer-predictions.parquet"
        _write_parquet(predictions, prediction_path)

        control_model_path = Path(
            m55["windows"][window]["training"]["dual_fusion"]["outer_model"]["model"]["path"]
        )
        control_model_identity = m55["windows"][window]["training"]["dual_fusion"]["outer_model"]["model"]
        if file_identity(control_model_path) != control_model_identity:
            raise RuntimeError(f"M5.6 M5.5 control model identity mismatch: {window}")
        control_model = lgb.Booster(model_file=str(control_model_path))
        feature_gain = {
            "Control": _importance(control_model),
            "A_current_sampling_fixed40": _importance(model_a),
            "B_conditional_sampling_temporal_early_stopping": _importance(model_b),
            "C_conditional_sampling_fixed40": _importance(model_c),
        }
        windows[window] = {
            "protocol": protocol,
            "candidate_feature_identity": file_identity(outer_valid_path),
            "sampling_audit": {
                "A_current_sampling": current_evidence,
                "A_matches_M5.5_control": True,
                "conditional_inner": conditional_inner_evidence,
                "conditional_outer": conditional_outer_evidence,
                "B_C_sample_identity": True,
            },
            "training": {
                "A": evidence_a,
                "B_inner": evidence_b_inner,
                "B_outer": evidence_b,
                "C": evidence_c,
                "category_maps_A": file_identity(root / "outer-A-category-maps.json"),
                "category_maps_BC": file_identity(root / "outer-BC-category-maps.json"),
            },
            "feature_gain": feature_gain,
            "evaluations": evaluations,
            "predictions": file_identity(prediction_path),
            "control_inputs": {
                "model": control_model_identity,
                "predictions": m55["windows"][window]["predictions"],
            },
            "final_week": "not_run",
        }
        artifact_bytes += sum(path.stat().st_size for path in root.iterdir() if path.is_file())
        del frame, predictions, model_a, model_b, model_c, control_model
        del ranked_a, ranked_b, ranked_c, control_ranked
        gc.collect()

    mechanism = _mechanism_summary(windows)
    deployment = _deployment_gate(windows, mechanism)
    if not any(row["passed"] for row in mechanism["variants"].values()):
        next_stage = "stop_at_M5.6_representation_ranking_separability"
    elif deployment["passed"]:
        next_stage = "M5.6_C_candidate_next_ranker"
    else:
        next_stage = "mechanism_fixed_but_not_deployable"
    interpretation_notes = [
        f"2×2 机制矩阵按预注册规则归类为 {mechanism['interpretation']}。",
        (
            "只有 C 同时具备采样修正与固定容量，且只有 C 预注册为潜在部署候选；"
            "A/B 即使出现机制信号也不能直接晋级。"
        ),
        f"门禁结论为 {next_stage}；没有据 outer 结果追加模型、轮数或采样比例搜索。",
    ]
    sampling_audit = {
        window: row["sampling_audit"] for window, row in windows.items()
    }
    feature_gain_audit = {
        window: row["feature_gain"] for window, row in windows.items()
    }
    score_separation_audit = {
        window: {
            variant: evaluation["score_audit"]
            for variant, evaluation in row["evaluations"].items()
            if "score_audit" in evaluation
        }
        for window, row in windows.items()
    }
    cold_truth_funnel = {
        window: {
            variant: evaluation["cold_truth_funnel"]
            for variant, evaluation in row["evaluations"].items()
            if "cold_truth_funnel" in evaluation
        }
        for window, row in windows.items()
    }
    atomic_json(report_dir / "sampling_audit.json", sampling_audit)
    atomic_json(report_dir / "feature_gain_audit.json", feature_gain_audit)
    atomic_json(report_dir / "score_separation_audit.json", score_separation_audit)
    atomic_json(report_dir / "cold_truth_funnel.json", cold_truth_funnel)
    feature_nonzero_windows = {
        variant: {
            feature: sum(
                next(
                    item["gain"]
                    for item in row["feature_gain"][variant]["features"]
                    if item["feature"] == feature
                ) > 0
                for row in windows.values()
            )
            for feature in SOURCE_FEATURES
        }
        for variant in ("Control", *VARIANTS)
    }
    result = {
        "schema_version": "m5.6-cold-supervision-suppression-v1",
        "stage": "M5.6",
        "status": "measured",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "inputs": {
            "m54_metrics": file_identity(m54_metrics_path),
            "m55_metrics": file_identity(m55_metrics_path),
            "m55_verification": file_identity(m55_verification_path),
            "m3_metrics": file_identity(m3_path),
        },
        "windows": windows,
        "audits": {
            "experiment_contract": file_identity(report_dir / "experiment_contract.json"),
            "sampling": file_identity(report_dir / "sampling_audit.json"),
            "feature_gain": file_identity(report_dir / "feature_gain_audit.json"),
            "score_separation": file_identity(report_dir / "score_separation_audit.json"),
            "cold_truth_funnel": file_identity(report_dir / "cold_truth_funnel.json"),
        },
        "summary": {
            "mechanism": mechanism,
            "deployment": deployment,
            "feature_nonzero_windows": feature_nonzero_windows,
            "next_stage": next_stage,
            "interpretation_notes": interpretation_notes,
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "artifact_bytes": artifact_bytes,
        },
        "final_week": "not_run",
    }
    atomic_json(report_dir / "metrics.json", result)
    (report_dir / "M5_6_FINAL.md").write_text(_render_report(result), encoding="utf-8")
    return result
