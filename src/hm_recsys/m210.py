from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import duckdb
import numpy as np
import pandas as pd

from .m2 import (
    CATEGORICAL_FEATURES,
    M2Config,
    _literal,
    _prepare_frame,
    _sha256,
    _write_json,
    build_category_maps,
)
from .m21 import inactive_fallback_order_expression
from .m26 import (
    PROTOCOL,
    REQUIRED_CUTOFFS,
    _evaluate_models,
    _frozen_reference,
    _save_category_maps,
)
from .m29 import ALL_FEATURES as M29_FEATURES
from .m29 import _candidate_paths as m29_candidate_paths
from .m29 import _attach_popularity_segments, _read_json


SCHEMA_VERSION = "m2.10-target-aware-ranking-protocol-v1"
FEATURE_SCHEMA_VERSION = "m2.10-target-aware-features-v1"
TARGET_AWARE_FEATURES = [
    "user_item_events_28d",
    "user_item_decay_28d_halflife_12w",
    "user_product_code_events_28d",
    "user_product_code_events_12w",
    "user_product_code_days_since",
    "user_product_code_share_12w",
    "user_product_code_decay_28d_halflife_12w",
    "user_product_type_events_28d",
    "user_product_type_days_since",
    "user_product_type_decay_28d_halflife_12w",
    "user_department_events_28d",
    "user_department_days_since",
    "user_department_decay_28d_halflife_12w",
    "user_garment_events_28d",
    "user_garment_events_12w",
    "user_garment_days_since",
    "user_garment_share_12w",
    "user_garment_decay_28d_halflife_12w",
    "user_colour_events_28d",
    "user_colour_events_12w",
    "user_colour_days_since",
    "user_colour_share_12w",
    "user_colour_decay_28d_halflife_12w",
    "user_index_group_events_28d",
    "user_index_group_events_12w",
    "user_index_group_days_since",
    "user_index_group_share_12w",
    "user_index_group_decay_28d_halflife_12w",
]
ALL_FEATURES = list(dict.fromkeys(M29_FEATURES + TARGET_AWARE_FEATURES))
PROTOCOLS = {
    "a_all_groups_full_lambdarank": {
        "sample_mode": "all",
        "objective": "lambdarank",
    },
    "b_positive_groups_full_lambdarank": {
        "sample_mode": "positive_full",
        "objective": "lambdarank",
    },
    "b_stratified_30x_lambdarank": {
        "sample_mode": "stratified_30x",
        "objective": "lambdarank",
    },
    "b_stratified_30x_map_pair": {
        "sample_mode": "stratified_30x",
        "objective": "map_pair_delta_at_12",
    },
}
HARD_NEGATIVE_RATIO = 30
MAP_PAIR_MAX_ESTIMATED_SECONDS = 900.0


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _target_feature_paths(cache_dir: Path, cutoff: str) -> tuple[Path, Path]:
    root = cache_dir / cutoff
    return root / "features.parquet", root / "feature-manifest.json"


def _date(value: str) -> str:
    return "DATE '" + value.replace("'", "''") + "'"


def _validate_target_cache(
    feature_path: Path,
    manifest_path: Path,
    source_identity: dict[str, Any],
    source_manifest_identity: dict[str, Any],
    transaction_identity: dict[str, Any],
    article_identity: dict[str, Any],
) -> dict[str, Any] | None:
    if not feature_path.is_file() or not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"invalid M2.10 feature manifest: {manifest_path}")
    inputs = manifest["inputs"]
    if inputs["m2_9_features"]["sha256"] != source_identity["sha256"]:
        raise ValueError(f"M2.10 source feature drift: {manifest_path}")
    if (
        inputs["m2_9_feature_manifest"]["sha256"]
        != source_manifest_identity["sha256"]
    ):
        raise ValueError(f"M2.10 source manifest drift: {manifest_path}")
    if inputs["transactions"]["sha256"] != transaction_identity["sha256"]:
        raise ValueError(f"M2.10 transaction drift: {manifest_path}")
    if inputs["articles"]["sha256"] != article_identity["sha256"]:
        raise ValueError(f"M2.10 article dimension drift: {manifest_path}")
    actual = _file_identity(feature_path)
    declared = manifest["artifact"]
    if (
        actual["sha256"] != declared["sha256"]
        or actual["bytes"] != int(declared["bytes"])
    ):
        raise ValueError(f"M2.10 target-aware cache drift: {feature_path}")
    return manifest

def _dimension_sql(
    table: str,
    key: str,
    cutoff_sql: str,
) -> str:
    return f"""
    CREATE TABLE {table} AS
    SELECT customer_id,{key},
           count(*) FILTER (
               WHERE t_dat >= {cutoff_sql}-INTERVAL 28 DAY
           )::BIGINT AS events_28d,
           count(*)::BIGINT AS events_12w,
           date_diff('day',max(t_dat),{cutoff_sql})::BIGINT AS days_since,
           sum(exp(-ln(2.0)*date_diff('day',t_dat,{cutoff_sql})/28.0))::DOUBLE
               AS decay_28d_halflife_12w
    FROM ta_history
    GROUP BY customer_id,{key}
    """


def build_target_aware_cache(
    *,
    raw_dir: Path,
    transactions_path: Path,
    m29_cache_dir: Path,
    cache_dir: Path,
    cutoffs: tuple[str, ...] = REQUIRED_CUTOFFS,
) -> dict[str, Any]:
    transaction_identity = _file_identity(transactions_path)
    article_identity = _file_identity(raw_dir / "articles.csv")
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for cutoff in cutoffs:
        started = time.perf_counter()
        _, _, source_path, source_manifest_path = m29_candidate_paths(
            m29_cache_dir, cutoff
        )
        source_identity = _file_identity(source_path)
        source_manifest = _read_json(source_manifest_path)
        source_manifest_identity = _file_identity(source_manifest_path)
        if source_manifest.get("dataset_sha256") != source_identity["sha256"]:
            raise ValueError(f"M2.9 feature identity mismatch: {cutoff}")
        feature_path, manifest_path = _target_feature_paths(cache_dir, cutoff)
        existing = _validate_target_cache(
            feature_path,
            manifest_path,
            source_identity,
            source_manifest_identity,
            transaction_identity,
            article_identity,
        )
        if existing is not None:
            results[cutoff] = existing
            continue
        root = feature_path.parent
        if root.exists():
            raise FileExistsError(f"incomplete M2.10 feature cache: {root}")
        root.mkdir(parents=True)
        database = root / "feature-build.duckdb"
        connection = duckdb.connect(str(database))
        temp_dir = root / "duckdb-temp"
        temp_dir.mkdir()
        connection.execute("SET threads=8")
        connection.execute("SET memory_limit='11GB'")
        connection.execute(f"SET temp_directory={_literal(temp_dir)}")
        cutoff_sql = _date(cutoff)
        try:
            connection.execute(
                f"CREATE VIEW base AS SELECT * FROM read_parquet({_literal(source_path)})"
            )
            connection.execute(
                "CREATE TABLE ta_users AS SELECT DISTINCT customer_id FROM base"
            )
            connection.execute(
                f"""
                CREATE TABLE ta_articles AS
                SELECT article_id,
                       try_cast(product_code AS INTEGER) AS article_product_code,
                       try_cast(product_type_no AS INTEGER) AS article_product_type_no,
                       try_cast(garment_group_no AS INTEGER) AS article_garment_group_no,
                       try_cast(department_no AS INTEGER) AS article_department_no,
                       try_cast(index_group_no AS INTEGER) AS article_index_group_no,
                       try_cast(perceived_colour_master_id AS INTEGER)
                           AS article_colour_master_id
                FROM read_csv_auto({_literal(raw_dir / "articles.csv")},
                    header=true,all_varchar=true)
                """
            )
            connection.execute(
                f"""
                CREATE TABLE ta_history AS
                SELECT t.customer_id,t.article_id,t.t_dat,
                       a.article_product_code,a.article_product_type_no,
                       a.article_garment_group_no,a.article_department_no,
                       a.article_index_group_no,a.article_colour_master_id
                FROM read_parquet({_literal(transactions_path)}) t
                SEMI JOIN ta_users u USING(customer_id)
                JOIN ta_articles a USING(article_id)
                WHERE t.t_dat >= {cutoff_sql}-INTERVAL 12 WEEK
                  AND t.t_dat < {cutoff_sql}
                """
            )
            dimensions = {
                "ta_item": "article_id",
                "ta_product_code": "article_product_code",
                "ta_product_type": "article_product_type_no",
                "ta_department": "article_department_no",
                "ta_garment": "article_garment_group_no",
                "ta_colour": "article_colour_master_id",
                "ta_index_group": "article_index_group_no",
            }
            for table, key in dimensions.items():
                connection.execute(_dimension_sql(table, key, cutoff_sql))
            select_sql = """
                SELECT b.*,
                    coalesce(ui.events_28d,0)::BIGINT AS user_item_events_28d,
                    coalesce(ui.decay_28d_halflife_12w,0.0)
                        AS user_item_decay_28d_halflife_12w,
                    coalesce(pc.events_28d,0)::BIGINT AS user_product_code_events_28d,
                    coalesce(pc.events_12w,0)::BIGINT AS user_product_code_events_12w,
                    pc.days_since AS user_product_code_days_since,
                    coalesce(pc.events_12w,0)/greatest(b.user_history_events_12w,1.0)
                        AS user_product_code_share_12w,
                    coalesce(pc.decay_28d_halflife_12w,0.0)
                        AS user_product_code_decay_28d_halflife_12w,
                    coalesce(pt.events_28d,0)::BIGINT AS user_product_type_events_28d,
                    pt.days_since AS user_product_type_days_since,
                    coalesce(pt.decay_28d_halflife_12w,0.0)
                        AS user_product_type_decay_28d_halflife_12w,
                    coalesce(dp.events_28d,0)::BIGINT AS user_department_events_28d,
                    dp.days_since AS user_department_days_since,
                    coalesce(dp.decay_28d_halflife_12w,0.0)
                        AS user_department_decay_28d_halflife_12w,
                    coalesce(gg.events_28d,0)::BIGINT AS user_garment_events_28d,
                    coalesce(gg.events_12w,0)::BIGINT AS user_garment_events_12w,
                    gg.days_since AS user_garment_days_since,
                    coalesce(gg.events_12w,0)/greatest(b.user_history_events_12w,1.0)
                        AS user_garment_share_12w,
                    coalesce(gg.decay_28d_halflife_12w,0.0)
                        AS user_garment_decay_28d_halflife_12w,
                    coalesce(cl.events_28d,0)::BIGINT AS user_colour_events_28d,
                    coalesce(cl.events_12w,0)::BIGINT AS user_colour_events_12w,
                    cl.days_since AS user_colour_days_since,
                    coalesce(cl.events_12w,0)/greatest(b.user_history_events_12w,1.0)
                        AS user_colour_share_12w,
                    coalesce(cl.decay_28d_halflife_12w,0.0)
                        AS user_colour_decay_28d_halflife_12w,
                    coalesce(ig.events_28d,0)::BIGINT AS user_index_group_events_28d,
                    coalesce(ig.events_12w,0)::BIGINT AS user_index_group_events_12w,
                    ig.days_since AS user_index_group_days_since,
                    coalesce(ig.events_12w,0)/greatest(b.user_history_events_12w,1.0)
                        AS user_index_group_share_12w,
                    coalesce(ig.decay_28d_halflife_12w,0.0)
                        AS user_index_group_decay_28d_halflife_12w
                FROM base b
                LEFT JOIN ta_item ui USING(customer_id,article_id)
                LEFT JOIN ta_product_code pc
                  ON b.customer_id=pc.customer_id
                 AND b.article_product_code=pc.article_product_code
                LEFT JOIN ta_product_type pt
                  ON b.customer_id=pt.customer_id
                 AND b.article_product_type_no=pt.article_product_type_no
                LEFT JOIN ta_department dp
                  ON b.customer_id=dp.customer_id
                 AND b.article_department_no=dp.article_department_no
                LEFT JOIN ta_garment gg
                  ON b.customer_id=gg.customer_id
                 AND b.article_garment_group_no=gg.article_garment_group_no
                LEFT JOIN ta_colour cl
                  ON b.customer_id=cl.customer_id
                 AND b.article_colour_master_id=cl.article_colour_master_id
                LEFT JOIN ta_index_group ig
                  ON b.customer_id=ig.customer_id
                 AND b.article_index_group_no=ig.article_index_group_no
            """
            connection.execute(
                f"COPY ({select_sql} ORDER BY customer_id,candidate_rank,article_id) "
                f"TO {_literal(feature_path)} "
                "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
            )
            identity_cols = "customer_id,article_id,candidate_rank,target"
            audit = connection.execute(
                f"""
                SELECT
                  (SELECT count(*) FROM base),
                  (SELECT count(*) FROM read_parquet({_literal(feature_path)})),
                  (SELECT count(*) FROM (
                    SELECT {identity_cols} FROM base
                    EXCEPT ALL
                    SELECT {identity_cols} FROM read_parquet({_literal(feature_path)})
                  )),
                  (SELECT count(*) FROM (
                    SELECT {identity_cols} FROM read_parquet({_literal(feature_path)})
                    EXCEPT ALL
                    SELECT {identity_cols} FROM base
                  )),
                  (SELECT count(*) FROM read_parquet({_literal(feature_path)})
                    WHERE target NOT IN (0,1)),
                  (SELECT max(t_dat) FROM ta_history)
                """
            ).fetchone()
            if audit is None:
                raise RuntimeError("M2.10 feature audit returned no rows")
            if int(audit[0]) != int(audit[1]) or int(audit[2]) or int(audit[3]) or int(audit[4]):
                raise RuntimeError(f"M2.10 feature conservation failed: {cutoff} {audit}")
            if audit[5] is not None and str(audit[5]) >= cutoff:
                raise RuntimeError(f"M2.10 temporal leakage detected: {cutoff}")
            feature_stats = {}
            for feature in TARGET_AWARE_FEATURES:
                nonzero, nulls = connection.execute(
                    f"SELECT count(*) FILTER(WHERE {feature}<>0),"
                    f"count(*) FILTER(WHERE {feature} IS NULL) "
                    f"FROM read_parquet({_literal(feature_path)})"
                ).fetchone()
                feature_stats[feature] = {
                    "nonzero_rows": int(nonzero),
                    "null_rows": int(nulls),
                }
            manifest = {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "status": "completed",
                "cutoff": cutoff,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract": {
                    "history": "cutoff-12w <= t_dat < cutoff",
                    "events": "raw rows retained including exact duplicates",
                    "decay": "exp(-ln(2)*days/28) over cutoff-safe 12w",
                    "target_aware_features": TARGET_AWARE_FEATURES,
                },
                "inputs": {
                    "m2_9_features": source_identity,
                    "m2_9_feature_manifest": _file_identity(source_manifest_path),
                    "transactions": transaction_identity,
                    "articles": article_identity,
                },
                "audit": {
                    "source_rows": int(audit[0]),
                    "output_rows": int(audit[1]),
                    "identity_except_all_forward": int(audit[2]),
                    "identity_except_all_reverse": int(audit[3]),
                    "invalid_targets": int(audit[4]),
                    "latest_history_date": str(audit[5]),
                    "feature_stats": feature_stats,
                },
                "artifact": _file_identity(feature_path),
                "elapsed_seconds": time.perf_counter()-started,
            }
            _write_json(manifest_path, manifest)
            results[cutoff] = manifest
        finally:
            connection.close()
    return results


def average_precision_at_k(labels: np.ndarray, k: int = 12) -> float:
    values = np.asarray(labels, dtype=np.uint8)
    total = int(values.sum())
    if total == 0:
        return 0.0
    limit = min(k, len(values))
    top = values[:limit]
    cumulative = np.cumsum(top)
    numerator = float(np.sum((cumulative / np.arange(1, limit+1)) * top))
    return numerator / min(total, k)


def map_swap_delta_bruteforce(
    ranked_labels: np.ndarray, positive_rank: int, negative_rank: int, k: int = 12
) -> float:
    labels = np.asarray(ranked_labels, dtype=np.uint8)
    if labels[positive_rank] != 1 or labels[negative_rank] != 0:
        raise ValueError("swap must move a positive and a negative")
    before = average_precision_at_k(labels, k)
    swapped = labels.copy()
    swapped[positive_rank], swapped[negative_rank] = 0, 1
    return average_precision_at_k(swapped, k)-before


def _swap_delta_fast(
    ranked_labels: np.ndarray,
    cumulative: np.ndarray,
    positive_reciprocal_prefix: np.ndarray,
    positive_rank: int,
    negative_rank: int,
    k: int,
) -> float:
    rp = positive_rank
    rn = negative_rank
    total = int(cumulative[-1])
    denominator = min(total, k)
    if denominator == 0:
        return 0.0
    if rp < rn:
        removed = cumulative[rp]/(rp+1) if rp < k else 0.0
        between = (
            positive_reciprocal_prefix[min(rn, k)]
            - positive_reciprocal_prefix[min(rp+1, k)]
        )
        added = cumulative[rn]/(rn+1) if rn < k else 0.0
        return float((-removed-between+added)/denominator)
    added = (cumulative[rn]+1)/(rn+1) if rn < k else 0.0
    between = (
        positive_reciprocal_prefix[min(rp, k)]
        - positive_reciprocal_prefix[min(rn+1, k)]
    )
    removed = cumulative[rp]/(rp+1) if rp < k else 0.0
    return float((added+between-removed)/denominator)


def map_pair_gradients(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_sizes: list[int] | np.ndarray,
    *,
    k: int = 12,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scores = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.uint8)
    sizes = np.asarray(group_sizes, dtype=np.int64)
    if int(sizes.sum()) != len(scores) or len(scores) != len(targets):
        raise ValueError("MAP pair objective group sizes do not match rows")
    gradients = np.zeros(len(scores), dtype=np.float64)
    hessians = np.full(len(scores), 1e-12, dtype=np.float64)
    pair_count = 0
    weighted_groups = 0
    offset = 0
    for size in sizes:
        end = offset+int(size)
        group_scores = scores[offset:end]
        group_targets = targets[offset:end]
        positive_count = int(group_targets.sum())
        if 0 < positive_count < len(group_targets):
            order = np.argsort(-group_scores, kind="mergesort")
            ranked = group_targets[order]
            cumulative = np.cumsum(ranked)
            limit = min(k, len(ranked))
            reciprocal = np.zeros(len(ranked), dtype=np.float64)
            reciprocal[:limit] = (
                ranked[:limit]/np.arange(1, limit+1, dtype=np.float64)
            )
            prefix = np.concatenate(([0.0], np.cumsum(reciprocal)))
            positive_ranks = np.flatnonzero(ranked == 1)
            negative_ranks = np.flatnonzero(ranked == 0)
            records: list[tuple[int, int, float]] = []
            total_weight = 0.0
            for positive_rank in positive_ranks:
                for negative_rank in negative_ranks:
                    if positive_rank >= k and negative_rank >= k:
                        continue
                    weight = abs(
                        _swap_delta_fast(
                            ranked,
                            cumulative,
                            prefix,
                            int(positive_rank),
                            int(negative_rank),
                            k,
                        )
                    )
                    if weight > 0:
                        records.append(
                            (int(order[positive_rank]), int(order[negative_rank]), weight)
                        )
                        total_weight += weight
            if total_weight > 0:
                weighted_groups += 1
                normalization = 1.0/total_weight
                for positive_row, negative_row, raw_weight in records:
                    weight = raw_weight*normalization
                    difference = float(
                        np.clip(
                            sigma*(group_scores[positive_row]-group_scores[negative_row]),
                            -50.0,
                            50.0,
                        )
                    )
                    probability = 1.0/(1.0+math.exp(difference))
                    pair_lambda = sigma*probability*weight
                    pair_hessian = (
                        sigma*sigma*probability*(1.0-probability)*weight
                    )
                    gradients[offset+positive_row] -= pair_lambda
                    gradients[offset+negative_row] += pair_lambda
                    hessians[offset+positive_row] += pair_hessian
                    hessians[offset+negative_row] += pair_hessian
                    pair_count += 1
        offset = end
    if not np.isfinite(gradients).all() or not np.isfinite(hessians).all():
        raise FloatingPointError("non-finite MAP pair gradients")
    return gradients, hessians, {
        "groups": int(len(sizes)),
        "weighted_groups": weighted_groups,
        "pairs": pair_count,
        "gradient_l1": float(np.abs(gradients).sum()),
        "hessian_sum": float(hessians.sum()),
    }


def _map_objective(k: int = 12) -> Callable[[np.ndarray, Any], tuple[np.ndarray, np.ndarray]]:
    def objective(predictions: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
        groups = dataset.get_group()
        if groups is None:
            raise ValueError("MAP pair objective requires ranking groups")
        gradients, hessians, _ = map_pair_gradients(
            predictions, dataset.get_label(), groups, k=k
        )
        return gradients, hessians
    return objective


def _relation_sql(path: Path, sample_mode: str) -> str:
    source = f"read_parquet({_literal(path)})"
    if sample_mode == "all":
        return f"SELECT * FROM {source}"
    if sample_mode == "positive_full":
        return f"""
        WITH positive_groups AS (
            SELECT customer_id
            FROM {source}
            GROUP BY customer_id
            HAVING sum(target)>0
        )
        SELECT f.*
        FROM {source} f
        JOIN positive_groups g USING(customer_id)
        """
    if sample_mode != "stratified_30x":
        raise ValueError(f"unknown training sample mode: {sample_mode}")
    return f"""
    WITH positive_groups AS (
        SELECT customer_id,sum(target)::BIGINT AS positives
        FROM {source}
        GROUP BY customer_id
        HAVING positives>0
    ), positives AS (
        SELECT f.*,NULL::BIGINT AS negative_sample_rank
        FROM {source} f
        JOIN positive_groups g USING(customer_id)
        WHERE f.target=1
    ), negative_stratum AS (
        SELECT f.*,g.positives,
               CASE WHEN f.candidate_rank<=100 THEN 0 ELSE 1 END AS stratum,
               row_number() OVER(
                   PARTITION BY f.customer_id,
                       CASE WHEN f.candidate_rank<=100 THEN 0 ELSE 1 END
                   ORDER BY f.candidate_rank,f.article_id
               ) AS stratum_rank
        FROM {source} f
        JOIN positive_groups g USING(customer_id)
        WHERE f.target=0
    ), negative_ranked AS (
        SELECT *,
               row_number() OVER(
                   PARTITION BY customer_id
                   ORDER BY stratum_rank,stratum,candidate_rank,article_id
               ) AS negative_sample_rank
        FROM negative_stratum
    ), sampled_negatives AS (
        SELECT * EXCLUDE(positives,stratum,stratum_rank)
        FROM negative_ranked
        WHERE negative_sample_rank<={HARD_NEGATIVE_RATIO}*positives
    )
    SELECT * FROM positives
    UNION ALL BY NAME
    SELECT * FROM sampled_negatives
    """


def load_training_data(
    dataset_paths: list[Path],
    features: list[str],
    sample_mode: str,
) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    group_sizes: list[int] = []
    positive_groups = 0
    total_groups = 0
    source_rows = 0
    sampled_rows = 0
    sampled_negatives = 0
    for path in dataset_paths:
        connection = duckdb.connect()
        try:
            relation = _relation_sql(path, sample_mode)
            selected = ",".join(features+["target"])
            frame = connection.execute(
                f"SELECT {selected} FROM ({relation}) "
                "ORDER BY customer_id,candidate_rank,article_id"
            ).fetchdf()
            group_rows = connection.execute(
                f"""
                SELECT count(*) AS rows,sum(target)::BIGINT AS positives
                FROM ({relation})
                GROUP BY customer_id
                ORDER BY customer_id
                """
            ).fetchall()
            source_count = int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet({_literal(path)})"
                ).fetchone()[0]
            )
        finally:
            connection.close()
        frames.append(frame)
        sizes = [int(row[0]) for row in group_rows]
        group_sizes.extend(sizes)
        total_groups += len(group_rows)
        positive_groups += sum(int(row[1]) > 0 for row in group_rows)
        source_rows += source_count
        sampled_rows += len(frame)
        sampled_negatives += int((frame["target"] == 0).sum())
    if not frames:
        raise ValueError("training paths are empty")
    result = pd.concat(frames, ignore_index=True)
    if int(sum(group_sizes)) != len(result):
        raise RuntimeError("sampled feature rows differ from group sizes")
    if not group_sizes or min(group_sizes) < 1 or max(group_sizes) > 300:
        raise ValueError("sampled ranking group size outside 1..300")
    evidence = {
        "sample_mode": sample_mode,
        "source_rows": source_rows,
        "sampled_rows": sampled_rows,
        "row_retention": sampled_rows/source_rows,
        "groups": total_groups,
        "positive_groups": positive_groups,
        "zero_positive_groups": total_groups-positive_groups,
        "min_group_rows": min(group_sizes),
        "max_group_rows": max(group_sizes),
        "mean_group_rows": float(np.mean(group_sizes)),
        "positives": int(result["target"].sum()),
        "sampled_unobserved_items": sampled_negatives,
        "unobserved_per_positive": (
            sampled_negatives/max(int(result["target"].sum()), 1)
        ),
    }
    if sample_mode != "all" and evidence["zero_positive_groups"] != 0:
        raise RuntimeError("positive-group protocol retained a zero-positive group")
    if (
        sample_mode == "stratified_30x"
        and evidence["unobserved_per_positive"] > HARD_NEGATIVE_RATIO
    ):
        raise RuntimeError("stratified sampling exceeded frozen ratio")
    return result, group_sizes, evidence


def train_ranker(
    *,
    frame: pd.DataFrame,
    group_sizes: list[int],
    group_evidence: dict[str, Any],
    features: list[str],
    name: str,
    objective_name: str,
    artifact_dir: Path,
    config: M2Config,
    category_maps: dict[str, dict[int, int]],
) -> tuple[Any, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required; install dependency.txt") from error
    started = time.perf_counter()
    prepared = _prepare_frame(frame, features, category_maps)
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    dataset = lgb.Dataset(
        prepared,
        label=frame["target"].astype(np.uint8),
        group=group_sizes,
        feature_name=features,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    objective: str | Callable[[np.ndarray, Any], tuple[np.ndarray, np.ndarray]]
    objective = (
        "lambdarank"
        if objective_name == "lambdarank"
        else _map_objective(config.metric_k)
    )
    params: dict[str, Any] = {
        "objective": objective,
        "metric": ["map", "ndcg"],
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "seed": config.seed,
        "feature_fraction_seed": config.seed,
        "bagging_seed": config.seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": config.threads,
        "verbosity": -1,
        "eval_at": [config.metric_k],
        "lambdarank_truncation_level": 20,
    }
    evaluations: dict[str, Any] = {}
    model = lgb.train(
        params,
        dataset,
        num_boost_round=config.num_boost_round,
        valid_sets=[dataset],
        valid_names=["train"],
        callbacks=[lgb.record_evaluation(evaluations)],
    )
    model_path = artifact_dir/f"lightgbm-{name}.txt"
    model.save_model(str(model_path))
    importance = sorted(
        (
            {"feature": feature, "gain": float(gain), "split": int(split)}
            for feature, gain, split in zip(
                features,
                model.feature_importance(importance_type="gain"),
                model.feature_importance(importance_type="split"),
                strict=True,
            )
        ),
        key=lambda row: row["gain"],
        reverse=True,
    )
    params_evidence = {
        key: (
            "custom_map_pair_delta_at_12"
            if key == "objective" and callable(value)
            else value
        )
        for key, value in params.items()
    }
    evidence = {
        "name": name,
        "objective": objective_name,
        "features": features,
        "parameters": params_evidence,
        "num_boost_round": config.num_boost_round,
        "group_evidence": group_evidence,
        "train_metrics": {
            metric: float(values[-1])
            for metric, values in evaluations.get("train", {}).items()
        },
        "top_feature_importance": importance[:40],
        "target_aware_feature_importance": [
            row for row in importance if row["feature"] in TARGET_AWARE_FEATURES
        ],
        "model_path": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "elapsed_seconds": time.perf_counter()-started,
    }
    del dataset, prepared
    gc.collect()
    return model, evidence


def benchmark_map_objective(
    frame: pd.DataFrame,
    group_sizes: list[int],
    *,
    rounds: int,
    k: int,
) -> dict[str, Any]:
    sample_groups = min(1000, len(group_sizes))
    sizes = group_sizes[:sample_groups]
    rows = int(sum(sizes))
    labels = frame["target"].to_numpy(dtype=np.uint8)[:rows]
    predictions = np.linspace(0.0, 1.0, rows, dtype=np.float64)
    started = time.perf_counter()
    gradients, hessians, stats = map_pair_gradients(
        predictions, labels, sizes, k=k
    )
    elapsed = time.perf_counter()-started
    estimated = elapsed*(len(group_sizes)/sample_groups)*rounds
    passed = (
        np.isfinite(gradients).all()
        and np.isfinite(hessians).all()
        and stats["gradient_l1"] > 0
        and estimated <= MAP_PAIR_MAX_ESTIMATED_SECONDS
    )
    return {
        **stats,
        "sample_groups": sample_groups,
        "sample_rows": rows,
        "elapsed_seconds": elapsed,
        "estimated_full_training_objective_seconds": estimated,
        "threshold_seconds": MAP_PAIR_MAX_ESTIMATED_SECONDS,
        "passed": bool(passed),
    }


def score_models(
    *,
    dataset_path: Path,
    models: dict[str, tuple[Any, list[str], dict[str, dict[int, int]]]],
    evaluation_db: Path,
    prediction_path: Path,
    config: M2Config,
) -> dict[str, Any]:
    if evaluation_db.exists() or prediction_path.exists():
        raise FileExistsError(
            evaluation_db if evaluation_db.exists() else prediction_path
        )
    started = time.perf_counter()
    source = duckdb.connect()
    target = duckdb.connect(str(evaluation_db))
    temp_dir = evaluation_db.parent/"duckdb-temp"
    temp_dir.mkdir(exist_ok=True)
    target.execute(f"SET threads={config.threads}")
    target.execute("SET memory_limit='11GB'")
    target.execute(f"SET temp_directory={_literal(temp_dir)}")
    identity = [
        "customer_id",
        "article_id",
        "candidate_rank",
        "target",
        "user_history_events_12w",
    ]
    selected_features = list(
        dict.fromkeys(feature for _, features, _ in models.values() for feature in features)
    )
    cursor = source.execute(
        f"SELECT {','.join(identity+selected_features)} "
        f"FROM read_parquet({_literal(dataset_path)})"
    )
    rows = 0
    batches = 0
    initialized = False
    try:
        while True:
            batch = cursor.fetch_df_chunk(config.prediction_chunk_vectors)
            if batch.empty:
                break
            output = batch[identity].copy()
            for name, (model, features, maps) in models.items():
                output[f"score_{name}"] = model.predict(
                    _prepare_frame(batch, features, maps)
                )
            target.register("m210_prediction_batch", output)
            if not initialized:
                target.execute(
                    "CREATE TABLE predictions AS "
                    "SELECT * FROM m210_prediction_batch WHERE FALSE"
                )
                initialized = True
            target.execute(
                "INSERT INTO predictions SELECT * FROM m210_prediction_batch"
            )
            target.unregister("m210_prediction_batch")
            rows += len(output)
            batches += 1
        if not initialized:
            raise RuntimeError("M2.10 validation produced no rows")
        duplicates = int(
            target.execute(
                "SELECT count(*)-count(DISTINCT (customer_id,article_id)) "
                "FROM predictions"
            ).fetchone()[0]
        )
        if duplicates:
            raise RuntimeError("M2.10 predictions are not unique")
        target.execute(
            f"COPY predictions TO {_literal(prediction_path)} "
            "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
        )
    finally:
        source.close()
        target.close()
    return {
        "rows": rows,
        "batches": batches,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_bytes": prediction_path.stat().st_size,
        "prediction_sha256": _sha256(prediction_path),
        "evaluation_db": str(evaluation_db.resolve()),
        "elapsed_seconds": time.perf_counter()-started,
    }


def summarize_results(
    development: dict[str, Any],
    frozen: dict[str, Any],
    m29_metrics: dict[str, Any],
    metric_k: int,
) -> dict[str, Any]:
    metric = f"map@{metric_k}"
    model_names = sorted(
        set.intersection(*[set(result["models"]) for result in development.values()])
    )
    orderings: dict[str, Any] = {}
    for name in model_names:
        fallback = f"{name}__inactive_rrf"
        values = {
            dev: float(
                development[dev]["evaluation"]["orderings"][fallback]
                ["segments"]["overall"][metric]
            )
            for dev in PROTOCOL
        }
        pure = {
            dev: float(
                development[dev]["evaluation"]["orderings"][name]
                ["segments"]["overall"][metric]
            )
            for dev in PROTOCOL
        }
        orderings[fallback] = {
            "window_map@12": values,
            "pure_window_map@12": pure,
            "mean_map@12": float(np.mean(list(values.values()))),
            "min_map@12": float(np.min(list(values.values()))),
        }

    def m29_values(name: str) -> dict[str, float]:
        return {
            dev: float(
                m29_metrics["development"][dev]["evaluation"]["orderings"][name]
                ["segments"]["overall"][metric]
            )
            for dev in PROTOCOL
        }

    frozen_values = {
        dev: float(frozen[dev]["segments"]["overall"][metric])
        for dev in PROTOCOL
    }
    frozen_mean = float(np.mean(list(frozen_values.values())))
    lineage_name = "expanded_full_plus_item2vec__inactive_rrf"
    best_control_name = "expanded_full_no_item2vec__inactive_rrf"
    lineage_values = m29_values(lineage_name)
    best_control_values = m29_values(best_control_name)
    lineage_mean = float(np.mean(list(lineage_values.values())))
    best_control_mean = float(np.mean(list(best_control_values.values())))
    selected = max(
        orderings,
        key=lambda name: (
            orderings[name]["mean_map@12"],
            orderings[name]["min_map@12"],
            name,
        ),
    )
    selected_values = orderings[selected]["window_map@12"]
    frozen_delta = {
        dev: selected_values[dev]-frozen_values[dev] for dev in PROTOCOL
    }
    lineage_delta = {
        dev: selected_values[dev]-lineage_values[dev] for dev in PROTOCOL
    }
    best_control_delta = {
        dev: selected_values[dev]-best_control_values[dev] for dev in PROTOCOL
    }
    gate = {
        "accepted": (
            all(value >= 0 for value in frozen_delta.values())
            and orderings[selected]["mean_map@12"] > frozen_mean
        ),
        "rule": (
            "mean MAP@12 improves over frozen M2.2 and neither development "
            "window regresses"
        ),
        "selected_variant": selected,
        "window_delta_vs_frozen_map@12": frozen_delta,
        "mean_delta_vs_frozen_map@12": (
            orderings[selected]["mean_map@12"]-frozen_mean
        ),
        "window_delta_vs_m2_9_feature_lineage_map@12": lineage_delta,
        "mean_delta_vs_m2_9_feature_lineage_map@12": (
            orderings[selected]["mean_map@12"]-lineage_mean
        ),
        "window_delta_vs_m2_9_best_same_pool_map@12": best_control_delta,
        "mean_delta_vs_m2_9_best_same_pool_map@12": (
            orderings[selected]["mean_map@12"]-best_control_mean
        ),
    }
    candidate_ceiling = {}
    for dev, result in development.items():
        selected_eval = result["evaluation"]["orderings"][selected]["segments"]["overall"]
        candidate_ceiling[dev] = {
            "candidate_recall@expanded_pool": float(
                selected_eval["candidate_recall@expanded_pool"]
            ),
            "oracle_map@12": float(selected_eval[f"oracle_map@{metric_k}"]),
        }
    return {
        "orderings": orderings,
        "frozen_m2_2": {
            "window_map@12": frozen_values,
            "mean_map@12": frozen_mean,
        },
        "m2_9_feature_lineage_control": {
            "ordering": lineage_name,
            "window_map@12": lineage_values,
            "mean_map@12": lineage_mean,
        },
        "m2_9_best_same_pool_control": {
            "ordering": best_control_name,
            "window_map@12": best_control_values,
            "mean_map@12": best_control_mean,
        },
        "selection_gate": gate,
        "candidate_ceiling": candidate_ceiling,
        "final_week": "not_run",
    }

def render_report(result: dict[str, Any]) -> str:
    summary = result["development_summary"]
    gate = summary["selection_gate"]
    frozen = summary["frozen_m2_2"]
    lineage = summary["m2_9_feature_lineage_control"]
    best_control = summary["m2_9_best_same_pool_control"]
    lines = [
        "# M2.10A/B target-aware features and ranking protocols",
        "",
        "## 结论",
        "",
        f"- development gate：{'通过' if gate['accepted'] else '未通过'}；"
        f"最佳 exploratory variant 为 {gate['selected_variant']}。",
        f"- 相对冻结 M2.2 mean MAP@12："
        f"{gate['mean_delta_vs_frozen_map@12']:+.6f}；"
        f"相对 M2.9 feature-lineage control："
        f"{gate['mean_delta_vs_m2_9_feature_lineage_map@12']:+.6f}；"
        f"相对 M2.9 best same-pool control："
        f"{gate['mean_delta_vs_m2_9_best_same_pool_map@12']:+.6f}。",
        "- 候选池、validation rows 与 Recall/Oracle 保持 M2.9 expanded-300；"
        "final week 未运行。",
        "- MAP pair 是按 AP@12 swap delta 加权的 pairwise logistic surrogate，"
        "不是声称对非光滑 MAP 的精确直接优化。",
        "",
        "## MAP@12（exact inactive fallback）",
        "",
        "| ordering | dev-A | dev-B | mean | worst |",
        "|---|---:|---:|---:|---:|",
        f"| frozen M2.2 | {frozen['window_map@12']['dev_a']:.6f} | "
        f"{frozen['window_map@12']['dev_b']:.6f} | "
        f"{frozen['mean_map@12']:.6f} | "
        f"{min(frozen['window_map@12'].values()):.6f} |",
        f"| M2.9 feature-lineage + Item2Vec | "
        f"{lineage['window_map@12']['dev_a']:.6f} | "
        f"{lineage['window_map@12']['dev_b']:.6f} | "
        f"{lineage['mean_map@12']:.6f} | "
        f"{min(lineage['window_map@12'].values()):.6f} |",
        f"| M2.9 best same-pool no-Item2Vec | "
        f"{best_control['window_map@12']['dev_a']:.6f} | "
        f"{best_control['window_map@12']['dev_b']:.6f} | "
        f"{best_control['mean_map@12']:.6f} | "
        f"{min(best_control['window_map@12'].values()):.6f} |",
    ]
    for name, row in summary["orderings"].items():
        lines.append(
            f"| {name} | {row['window_map@12']['dev_a']:.6f} | "
            f"{row['window_map@12']['dev_b']:.6f} | "
            f"{row['mean_map@12']:.6f} | {row['min_map@12']:.6f} |"
        )
    lines.extend(["", "## 训练协议压缩", ""])
    for dev, evidence in result["development"].items():
        lines.append(f"### {dev}")
        lines.append("")
        for name, model in evidence["models"].items():
            group = model["group_evidence"]
            lines.append(
                f"- {name}：rows={group['sampled_rows']:,} "
                f"({group['row_retention']:.2%})，groups={group['groups']:,}，"
                f"positive_groups={group['positive_groups']:,}，"
                f"unobserved/positive={group['unobserved_per_positive']:.2f}，"
                f"train={model['elapsed_seconds']:.2f}s。"
            )
        benchmark = evidence["map_objective_benchmark"]
        lines.append(
            f"- MAP objective gate：{'pass' if benchmark['passed'] else 'fail'}；"
            f"estimated objective time="
            f"{benchmark['estimated_full_training_objective_seconds']:.2f}s。"
        )
        lines.append("")
    lines.extend([
        "## 审计边界",
        "",
        "- 所有 target-aware 聚合只读取 cutoff-12w <= t_dat < cutoff；"
        "原始重复交易保留。",
        "- hard-negative 只缩减训练中的未观测商品；没有曝光日志，"
        "这些行不是经过曝光确认的负反馈。",
        "- validation 始终完整 score 100--300 候选，不补正例、不做 stage pruning。",
        "- optimistic all-articles catalog 与多次查看相同 development windows 的"
        "选择偏差仍存在。",
        "- BPR 因缺少已验证实现与独立协议，本轮标注 deferred；"
        "不得把 Item2Vec source score 称作 BPR。",
        "",
        "## 产物",
        "",
        f"- metrics：{result['artifacts']['metrics']}",
        f"- private artifacts：{result['artifacts']['artifact_dir']}（Git ignored）",
        f"- feature cache：{result['artifacts']['feature_cache_dir']}（Git ignored）",
        "",
    ])
    return "\n".join(lines)

def run_m210(
    *,
    raw_dir: Path,
    transactions_path: Path,
    m21_metrics_path: Path,
    m29_metrics_path: Path,
    m29_cache_dir: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    config.validate()
    if config.evaluation_role != "development":
        raise ValueError("M2.10 is development-only")
    if config.candidate_k != 300:
        raise ValueError("M2.10 candidate_k must be 300")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    m29_metrics = _read_json(m29_metrics_path.resolve())
    if (
        m29_metrics.get("schema_version")
        != "m2.9-item2vec-ranker-development-v1"
        or m29_metrics.get("status") != "measured"
    ):
        raise ValueError("M2.10 requires measured M2.9 v2 evidence")
    for cutoff in REQUIRED_CUTOFFS:
        _, _, source_path, source_manifest_path = m29_candidate_paths(
            m29_cache_dir, cutoff
        )
        actual = _file_identity(source_path)
        expected = m29_metrics["feature_cache"][cutoff]
        if actual["sha256"] != expected["dataset_sha256"]:
            raise ValueError(f"M2.9 cache identity drift: {cutoff}")
        if _read_json(source_manifest_path)["dataset_sha256"] != actual["sha256"]:
            raise ValueError(f"M2.9 feature manifest drift: {cutoff}")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        feature_cache = build_target_aware_cache(
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            m29_cache_dir=m29_cache_dir,
            cache_dir=feature_cache_dir,
        )
        m21 = _read_json(m21_metrics_path.resolve())
        frozen_name, frozen = _frozen_reference(m21)
        development: dict[str, Any] = {}
        for dev, protocol in PROTOCOL.items():
            dev_dir = artifact_dir/dev
            dev_dir.mkdir()
            train_paths = [
                _target_feature_paths(feature_cache_dir, cutoff)[0]
                for cutoff in protocol["train"]
            ]
            models: dict[str, tuple[Any, list[str], dict[str, dict[int, int]]]] = {}
            model_evidence: dict[str, Any] = {}

            full_frame, full_sizes, full_evidence = load_training_data(
                train_paths, ALL_FEATURES, "all"
            )
            category_maps = build_category_maps(full_frame)
            category_evidence = _save_category_maps(
                dev_dir/"category-maps.json", category_maps
            )
            name = "a_all_groups_full_lambdarank"
            model, evidence = train_ranker(
                frame=full_frame,
                group_sizes=full_sizes,
                group_evidence=full_evidence,
                features=ALL_FEATURES,
                name=name,
                objective_name="lambdarank",
                artifact_dir=dev_dir,
                config=config,
                category_maps=category_maps,
            )
            models[name] = (model, ALL_FEATURES, category_maps)
            model_evidence[name] = evidence
            del full_frame, full_sizes
            gc.collect()

            positive_frame, positive_sizes, positive_evidence = load_training_data(
                train_paths, ALL_FEATURES, "positive_full"
            )
            name = "b_positive_groups_full_lambdarank"
            model, evidence = train_ranker(
                frame=positive_frame,
                group_sizes=positive_sizes,
                group_evidence=positive_evidence,
                features=ALL_FEATURES,
                name=name,
                objective_name="lambdarank",
                artifact_dir=dev_dir,
                config=config,
                category_maps=category_maps,
            )
            models[name] = (model, ALL_FEATURES, category_maps)
            model_evidence[name] = evidence
            del positive_frame, positive_sizes
            gc.collect()

            sampled_frame, sampled_sizes, sampled_evidence = load_training_data(
                train_paths, ALL_FEATURES, "stratified_30x"
            )
            name = "b_stratified_30x_lambdarank"
            model, evidence = train_ranker(
                frame=sampled_frame,
                group_sizes=sampled_sizes,
                group_evidence=sampled_evidence,
                features=ALL_FEATURES,
                name=name,
                objective_name="lambdarank",
                artifact_dir=dev_dir,
                config=config,
                category_maps=category_maps,
            )
            models[name] = (model, ALL_FEATURES, category_maps)
            model_evidence[name] = evidence

            benchmark = benchmark_map_objective(
                sampled_frame,
                sampled_sizes,
                rounds=config.num_boost_round,
                k=config.metric_k,
            )
            if benchmark["passed"]:
                name = "b_stratified_30x_map_pair"
                model, evidence = train_ranker(
                    frame=sampled_frame,
                    group_sizes=sampled_sizes,
                    group_evidence=sampled_evidence,
                    features=ALL_FEATURES,
                    name=name,
                    objective_name="map_pair_delta_at_12",
                    artifact_dir=dev_dir,
                    config=config,
                    category_maps=category_maps,
                )
                models[name] = (model, ALL_FEATURES, category_maps)
                model_evidence[name] = evidence
            del sampled_frame, sampled_sizes
            gc.collect()

            validation_cutoff = protocol["validation"]
            validation_path = _target_feature_paths(
                feature_cache_dir, validation_cutoff
            )[0]
            scoring = score_models(
                dataset_path=validation_path,
                models=models,
                evaluation_db=dev_dir/"evaluation.duckdb",
                prediction_path=dev_dir/"validation-predictions.parquet",
                config=config,
            )
            variant_names = list(models)
            del models
            gc.collect()
            evaluation = _evaluate_models(
                evaluation_db=dev_dir/"evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=validation_cutoff,
                metric_k=config.metric_k,
                variant_names=variant_names,
            )
            evaluation = _attach_popularity_segments(
                evaluation=evaluation,
                evaluation_db=dev_dir/"evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=validation_cutoff,
                variant_names=variant_names,
                budget=config.candidate_k,
            )
            development[dev] = {
                "train_cutoffs": protocol["train"],
                "validation_cutoff": validation_cutoff,
                "models": model_evidence,
                "category_encoding": category_evidence,
                "map_objective_benchmark": benchmark,
                "scoring": scoring,
                "evaluation": evaluation,
            }

        summary = summarize_results(
            development, frozen, m29_metrics, config.metric_k
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M2.10A/B",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "development_only_no_final_week",
            "config": asdict(config),
            "contract": {
                "candidate_pool": (
                    "exact M2.9 six-source Top100 plus up to 200 Item2Vec-only"
                ),
                "features": TARGET_AWARE_FEATURES,
                "training_protocols": PROTOCOLS,
                "hard_negative_ratio": HARD_NEGATIVE_RATIO,
                "validation": "complete variable 100..300 candidate pool",
                "frozen_reference": frozen_name,
                "bpr": "deferred_missing_verified_independent_protocol",
            },
            "inputs": {
                "m21_metrics": _file_identity(m21_metrics_path),
                "m29_metrics": _file_identity(m29_metrics_path),
                "transactions": _file_identity(transactions_path),
            },
            "feature_cache": feature_cache,
            "development": development,
            "development_summary": summary,
            "elapsed_seconds": time.perf_counter()-started,
        }
        metrics_path = output_dir/"metrics.json"
        report_path = output_dir/"M2_10_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path.resolve()),
            "report": str(report_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
            "feature_cache_dir": str(feature_cache_dir.resolve()),
        }
        _write_json(metrics_path, result)
        report_path.write_text(
            render_report(result), encoding="utf-8", newline="\n"
        )
        return result
    except Exception as error:
        _write_json(
            output_dir/f"failure-{time.time_ns()}.json",
            {
                "schema_version": "m2.10-failure-v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter()-started,
            },
        )
        raise
