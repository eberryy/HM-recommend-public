from __future__ import annotations

import gc
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m2 import (
    FULL_FEATURES,
    M2Config,
    _literal,
    _prepare_frame,
    _sha256,
    _write_json,
    build_category_maps,
)
from .m21 import inactive_fallback_order_expression
from .m26 import (
    IMAGE_FEATURES,
    PROTOCOL,
    REQUIRED_CUTOFFS,
    _evaluate_models,
    _frozen_reference,
    _save_category_maps,
    realized_oracle_fraction,
)
from .m29 import _attach_popularity_segments, _read_json
from .m210 import (
    TARGET_AWARE_FEATURES,
    _date,
    _dimension_sql,
    _file_identity,
)
from .m211 import SAMPLING_SEED, _load_category_maps, train_lambdarank
from .m212 import (
    DECAY_FEATURES,
    EARLY_STOPPING_ROUNDS,
    INNER_PROTOCOL,
    load_inner_validation,
    train_inner_ranker,
    validate_inner_protocol,
)


SCHEMA_VERSION = "m2.13a-image-ranking-revisit-v1"
FEATURE_SCHEMA_VERSION = "m2.13-image-target-aware-features-v1"
NEGATIVE_RATIO = 30
PRIMARY_MODEL = "image_pool_image_target_no_decay"
IMAGE_CONTROL = "image_pool_image"
BASE_CONTROL = "image_pool_base"
M26_OLD_BEST = "expanded_full_plus_image_semantic__inactive_rrf"

NO_DECAY_TARGET_FEATURES = [
    feature for feature in TARGET_AWARE_FEATURES if feature not in DECAY_FEATURES
]


def feature_sets() -> dict[str, list[str]]:
    sets = {
        BASE_CONTROL: list(FULL_FEATURES),
        IMAGE_CONTROL: list(dict.fromkeys(FULL_FEATURES + IMAGE_FEATURES)),
        PRIMARY_MODEL: list(
            dict.fromkeys(FULL_FEATURES + IMAGE_FEATURES + NO_DECAY_TARGET_FEATURES)
        ),
    }
    if len(NO_DECAY_TARGET_FEATURES) != 21:
        raise RuntimeError("M2.13 expected 21 no-decay target-aware features")
    if set(sets[PRIMARY_MODEL]) - set(sets[IMAGE_CONTROL]) != set(
        NO_DECAY_TARGET_FEATURES
    ):
        raise RuntimeError("M2.13 primary differs by more than target-aware features")
    return sets


def _source_paths(source_cache_dir: Path, cutoff: str) -> tuple[Path, Path, Path]:
    root = source_cache_dir / cutoff
    return (
        root / "features.parquet",
        root / "feature-manifest.json",
        root / "candidate-manifest.json",
    )


def _target_paths(cache_dir: Path, cutoff: str) -> tuple[Path, Path]:
    root = cache_dir / cutoff
    return root / "features.parquet", root / "feature-manifest.json"


def _validate_m26_source(
    source_cache_dir: Path, cutoff: str
) -> tuple[Path, dict[str, Any]]:
    feature_path, feature_manifest_path, candidate_manifest_path = _source_paths(
        source_cache_dir, cutoff
    )
    feature_identity = _file_identity(feature_path)
    feature_manifest = _read_json(feature_manifest_path)
    candidate_manifest = _read_json(candidate_manifest_path)
    if feature_manifest.get("schema_version") != "m2.6-feature-cache-v1":
        raise ValueError(f"invalid M2.6 feature manifest: {feature_manifest_path}")
    if candidate_manifest.get("schema_version") != "m2.6-expanded-cache-v2":
        raise ValueError(f"invalid M2.6 candidate manifest: {candidate_manifest_path}")
    if feature_manifest.get("cutoff") != cutoff or candidate_manifest.get("cutoff") != cutoff:
        raise ValueError(f"M2.6 cutoff mismatch: {cutoff}")
    if feature_manifest.get("dataset_sha256") != feature_identity["sha256"]:
        raise ValueError(f"M2.6 feature SHA mismatch: {cutoff}")
    if int(feature_manifest.get("dataset_bytes", -1)) != feature_identity["bytes"]:
        raise ValueError(f"M2.6 feature bytes mismatch: {cutoff}")
    candidate = candidate_manifest.get("artifact", {})
    candidate_path = Path(candidate.get("path", "")).resolve()
    if (
        not candidate_path.is_file()
        or candidate_path.stat().st_size != int(candidate.get("bytes", -1))
        or _sha256(candidate_path) != candidate.get("sha256")
    ):
        raise ValueError(f"M2.6 candidate artifact drift: {cutoff}")
    if feature_manifest.get("candidate_sha256") != candidate.get("sha256"):
        raise ValueError(f"M2.6 feature/candidate lineage mismatch: {cutoff}")
    return feature_path, {
        "features": feature_identity,
        "feature_manifest": _file_identity(feature_manifest_path),
        "candidate_manifest": _file_identity(candidate_manifest_path),
        "candidate": _file_identity(candidate_path),
        "candidate_audit": candidate_manifest["audit"],
    }


def _validate_target_cache(
    feature_path: Path,
    manifest_path: Path,
    *,
    source: dict[str, Any],
    transactions: dict[str, Any],
    articles: dict[str, Any],
) -> dict[str, Any] | None:
    if not feature_path.is_file() or not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"invalid M2.13 feature manifest: {manifest_path}")
    inputs = manifest.get("inputs", {})
    for key in ("features", "feature_manifest", "candidate_manifest", "candidate"):
        if inputs["m2_6_source"][key]["sha256"] != source[key]["sha256"]:
            raise ValueError(f"M2.13 source drift ({key}): {manifest_path}")
    if inputs["transactions"]["sha256"] != transactions["sha256"]:
        raise ValueError(f"M2.13 transaction drift: {manifest_path}")
    if inputs["articles"]["sha256"] != articles["sha256"]:
        raise ValueError(f"M2.13 article drift: {manifest_path}")
    actual = _file_identity(feature_path)
    if (
        actual["sha256"] != manifest["artifact"]["sha256"]
        or actual["bytes"] != int(manifest["artifact"]["bytes"])
    ):
        raise ValueError(f"M2.13 target-aware artifact drift: {feature_path}")
    return manifest


def _target_select_sql() -> str:
    return """
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


def build_image_target_cache(
    *,
    raw_dir: Path,
    transactions_path: Path,
    source_cache_dir: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    transaction_identity = _file_identity(transactions_path)
    article_identity = _file_identity(raw_dir / "articles.csv")
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for cutoff in REQUIRED_CUTOFFS:
        started = time.perf_counter()
        source_path, source = _validate_m26_source(source_cache_dir, cutoff)
        feature_path, manifest_path = _target_paths(cache_dir, cutoff)
        existing = _validate_target_cache(
            feature_path,
            manifest_path,
            source=source,
            transactions=transaction_identity,
            articles=article_identity,
        )
        if existing is not None:
            results[cutoff] = existing
            continue
        root = feature_path.parent
        if root.exists():
            raise FileExistsError(f"incomplete M2.13 feature cache: {root}")
        root.mkdir(parents=True)
        database_path = root / "feature-build.duckdb"
        connection = duckdb.connect(str(database_path))
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
            connection.execute("CREATE TABLE ta_users AS SELECT DISTINCT customer_id FROM base")
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
                FROM read_csv_auto({_literal(raw_dir / 'articles.csv')},
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
            select_sql = _target_select_sql()
            connection.execute(
                f"COPY ({select_sql} ORDER BY customer_id,candidate_rank,article_id) "
                f"TO {_literal(feature_path)} "
                "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
            )
            identity = "customer_id,article_id,candidate_rank,target"
            audit = connection.execute(
                f"""
                SELECT
                  (SELECT count(*) FROM base),
                  (SELECT count(*) FROM read_parquet({_literal(feature_path)})),
                  (SELECT count(*) FROM (
                    SELECT {identity} FROM base EXCEPT ALL
                    SELECT {identity} FROM read_parquet({_literal(feature_path)})
                  )),
                  (SELECT count(*) FROM (
                    SELECT {identity} FROM read_parquet({_literal(feature_path)}) EXCEPT ALL
                    SELECT {identity} FROM base
                  )),
                  (SELECT count(*) FROM read_parquet({_literal(feature_path)})
                    WHERE target NOT IN (0,1)),
                  (SELECT max(t_dat) FROM ta_history),
                  (SELECT count(*) FROM read_parquet({_literal(feature_path)})
                    WHERE image_is_new=1)
                """
            ).fetchone()
            if audit is None:
                raise RuntimeError("M2.13 feature audit returned no rows")
            if (
                int(audit[0]) != int(audit[1])
                or int(audit[2])
                or int(audit[3])
                or int(audit[4])
            ):
                raise RuntimeError(f"M2.13 feature conservation failed: {cutoff} {audit}")
            if audit[5] is not None and str(audit[5]) >= cutoff:
                raise RuntimeError(f"M2.13 temporal leakage detected: {cutoff}")
            expected_image_rows = int(source["candidate_audit"]["image_only_rows"])
            if int(audit[6]) != expected_image_rows:
                raise RuntimeError(f"M2.13 image-only row drift: {cutoff}")
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
                    "candidate_pool": "M2.6 v4 baseline Top100 plus up to 200 image-only",
                    "history": "cutoff-12w <= t_dat < cutoff",
                    "events": "raw rows retained including exact duplicates",
                    "target_aware_features": TARGET_AWARE_FEATURES,
                },
                "inputs": {
                    "m2_6_source": source,
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
                    "image_only_rows": int(audit[6]),
                    "feature_stats": feature_stats,
                },
                "artifact": _file_identity(feature_path),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(manifest_path, manifest)
            results[cutoff] = manifest
        finally:
            connection.close()
    return results


def _distribution_relation_sql(path: Path, *, seed: int) -> str:
    source = f"read_parquet({_literal(path)})"
    return f"""
    WITH positive_groups AS (
        SELECT customer_id,sum(target)::BIGINT AS positives
        FROM {source} GROUP BY customer_id HAVING positives>0
    ), positives AS (
        SELECT f.*,'positive'::VARCHAR AS _m213_source_layer,
               0::INTEGER AS _m213_rank_bucket,
               1::BIGINT AS _m213_stratum_rows,
               1::BIGINT AS _m213_selected_stratum_rows,
               1.0::DOUBLE AS _m213_inclusion_probability
        FROM {source} f JOIN positive_groups g USING(customer_id)
        WHERE f.target=1
    ), negative_source AS (
        SELECT f.*,g.positives,
               CASE WHEN f.image_is_new=1 THEN 'image_only'
                    ELSE 'baseline_top100' END AS _m213_source_layer
        FROM {source} f JOIN positive_groups g USING(customer_id)
        WHERE f.target=0
    ), negative_bucketed AS (
        SELECT *,ntile(3) OVER(
            PARTITION BY customer_id,_m213_source_layer
            ORDER BY candidate_rank,article_id
        )::INTEGER AS _m213_rank_bucket
        FROM negative_source
    ), negative_strata AS (
        SELECT *,count(*) OVER(
            PARTITION BY customer_id,_m213_source_layer,_m213_rank_bucket
        )::BIGINT AS _m213_stratum_rows,
        row_number() OVER(
            PARTITION BY customer_id,_m213_source_layer,_m213_rank_bucket
            ORDER BY hash(customer_id,article_id,{seed}),candidate_rank,article_id
        ) AS cell_rank
        FROM negative_bucketed
    ), round_robin AS (
        SELECT *,row_number() OVER(
            PARTITION BY customer_id
            ORDER BY cell_rank,_m213_source_layer,_m213_rank_bucket,
                     hash(customer_id,article_id,{seed}),candidate_rank,article_id
        ) AS sample_rank
        FROM negative_strata
    ), selected_pre AS (
        SELECT * FROM round_robin
        WHERE sample_rank<={NEGATIVE_RATIO}*positives
    ), selected_counted AS (
        SELECT *,count(*) OVER(
            PARTITION BY customer_id,_m213_source_layer,_m213_rank_bucket
        )::BIGINT AS _m213_selected_stratum_rows
        FROM selected_pre
    ), sampled_negatives AS (
        SELECT * EXCLUDE(positives,cell_rank,sample_rank),
               _m213_selected_stratum_rows::DOUBLE/_m213_stratum_rows
                   AS _m213_inclusion_probability
        FROM selected_counted
    )
    SELECT * FROM positives
    UNION ALL BY NAME
    SELECT * FROM sampled_negatives
    """


def load_image_distribution_sample(
    dataset_paths: list[Path],
    features: list[str],
    *,
    seed: int = SAMPLING_SEED,
) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    sizes: list[int] = []
    source_rows = 0
    strata: dict[tuple[str, int], dict[str, float]] = {}
    for path in dataset_paths:
        connection = duckdb.connect()
        try:
            relation = _distribution_relation_sql(path, seed=seed)
            helper = [
                "_m213_source_layer",
                "_m213_rank_bucket",
                "_m213_stratum_rows",
                "_m213_selected_stratum_rows",
                "_m213_inclusion_probability",
            ]
            frame = connection.execute(
                f"SELECT {','.join(features + ['target'] + helper)} "
                f"FROM ({relation}) ORDER BY customer_id,candidate_rank,article_id"
            ).fetchdf()
            groups = connection.execute(
                f"SELECT count(*),sum(target)::BIGINT FROM ({relation}) "
                "GROUP BY customer_id ORDER BY customer_id"
            ).fetchall()
            source_rows += int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet({_literal(path)})"
                ).fetchone()[0]
            )
            audit_rows = connection.execute(
                f"""
                SELECT _m213_source_layer,_m213_rank_bucket,
                       sum(_m213_selected_stratum_rows::DOUBLE/
                           _m213_inclusion_probability/
                           _m213_selected_stratum_rows)::BIGINT AS source_rows,
                       count(*)::BIGINT AS selected_rows,
                       min(_m213_inclusion_probability),
                       max(_m213_inclusion_probability),
                       min(candidate_rank),max(candidate_rank)
                FROM ({relation}) WHERE target=0
                GROUP BY _m213_source_layer,_m213_rank_bucket
                ORDER BY _m213_source_layer,_m213_rank_bucket
                """
            ).fetchall()
        finally:
            connection.close()
        frames.append(frame)
        sizes.extend(int(row[0]) for row in groups)
        if any(int(row[1]) <= 0 for row in groups):
            raise RuntimeError("M2.13 sampling retained zero-positive group")
        for layer, bucket, source_count, rows, p_min, p_max, minimum, maximum in audit_rows:
            key = (str(layer), int(bucket))
            entry = strata.setdefault(
                key,
                {
                    "source_rows": 0.0,
                    "selected_rows": 0.0,
                    "min_probability": 1.0,
                    "max_probability": 0.0,
                    "min_candidate_rank": 301.0,
                    "max_candidate_rank": 0.0,
                },
            )
            entry["source_rows"] += float(source_count)
            entry["selected_rows"] += float(rows)
            entry["min_probability"] = min(entry["min_probability"], float(p_min))
            entry["max_probability"] = max(entry["max_probability"], float(p_max))
            entry["min_candidate_rank"] = min(entry["min_candidate_rank"], float(minimum))
            entry["max_candidate_rank"] = max(entry["max_candidate_rank"], float(maximum))
    if not frames:
        raise ValueError("M2.13 training paths are empty")
    result = pd.concat(frames, ignore_index=True)
    if sum(sizes) != len(result):
        raise RuntimeError("M2.13 sampled rows differ from group sizes")
    if not sizes or min(sizes) < 2 or max(sizes) > 300:
        raise ValueError("M2.13 sampled group size outside 2..300")
    positives = int(result["target"].sum())
    negatives = int((result["target"] == 0).sum())
    if negatives / max(positives, 1) > NEGATIVE_RATIO:
        raise RuntimeError("M2.13 sampling exceeded frozen ratio")
    probabilities = result.loc[
        result["target"] == 0, "_m213_inclusion_probability"
    ].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(probabilities).all()
        or (probabilities <= 0).any()
        or (probabilities > 1).any()
    ):
        raise RuntimeError("M2.13 invalid negative inclusion probability")
    stratum_evidence = []
    for (layer, bucket), row in sorted(strata.items()):
        stratum_evidence.append(
            {
                "source_layer": layer,
                "rank_bucket": bucket,
                "source_rows": int(row["source_rows"]),
                "selected_rows": int(row["selected_rows"]),
                "retention": row["selected_rows"] / row["source_rows"],
                "min_probability": row["min_probability"],
                "max_probability": row["max_probability"],
                "min_candidate_rank": int(row["min_candidate_rank"]),
                "max_candidate_rank": int(row["max_candidate_rank"]),
            }
        )
    evidence = {
        "sample_mode": "baseline_image_rank_bucket_hash_30x",
        "seed": seed,
        "source_rows": source_rows,
        "sampled_rows": len(result),
        "row_retention": len(result) / source_rows,
        "groups": len(sizes),
        "positive_groups": len(sizes),
        "zero_positive_groups": 0,
        "min_group_rows": min(sizes),
        "max_group_rows": max(sizes),
        "positives": positives,
        "sampled_unobserved_items": negatives,
        "unobserved_per_positive": negatives / max(positives, 1),
        "negative_probability_min": float(probabilities.min()),
        "negative_probability_max": float(probabilities.max()),
        "strata": stratum_evidence,
    }
    return result, sizes, evidence


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
    temp_dir = evaluation_db.parent / "duckdb-temp"
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
        "image_present",
        "image_is_new",
    ]
    selected_features = list(
        dict.fromkeys(
            feature for _, features, _ in models.values() for feature in features
        )
    )
    selected = list(dict.fromkeys(identity + selected_features))
    cursor = source.execute(
        f"SELECT {','.join(selected)} FROM read_parquet({_literal(dataset_path)})"
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
            target.register("m213_prediction_batch", output)
            if not initialized:
                target.execute(
                    "CREATE TABLE predictions AS "
                    "SELECT * FROM m213_prediction_batch WHERE FALSE"
                )
                initialized = True
            target.execute("INSERT INTO predictions SELECT * FROM m213_prediction_batch")
            target.unregister("m213_prediction_batch")
            rows += len(output)
            batches += 1
        if not initialized:
            raise RuntimeError("M2.13 validation dataset produced no rows")
        audit = target.execute(
            "SELECT count(*)-count(DISTINCT (customer_id,article_id)),"
            "count(*) FILTER(WHERE image_is_new NOT IN (0,1)) FROM predictions"
        ).fetchone()
        if audit is None or int(audit[0]) or int(audit[1]):
            raise RuntimeError(f"M2.13 prediction identity audit failed: {audit}")
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
        "elapsed_seconds": time.perf_counter() - started,
    }


def evaluate_image_conversion(
    *,
    evaluation_db: Path,
    transactions_path: Path,
    cutoff: str,
    variant_names: list[str],
    metric_k: int,
) -> dict[str, Any]:
    connection = duckdb.connect(str(evaluation_db))
    cutoff_sql = _date(cutoff)
    transactions = _literal(transactions_path)
    try:
        connection.execute("SET threads=8")
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE m213_users AS
            SELECT DISTINCT customer_id FROM predictions;
            CREATE OR REPLACE TEMP TABLE m213_warm_catalog AS
            SELECT DISTINCT article_id FROM read_parquet({transactions})
            WHERE t_dat<{cutoff_sql};
            CREATE OR REPLACE TEMP TABLE m213_truth AS
            SELECT q.customer_id,q.article_id,
                   CASE WHEN w.article_id IS NULL THEN 'cold' ELSE 'warm' END
                       AS item_temperature
            FROM (
                SELECT DISTINCT t.customer_id,t.article_id
                FROM read_parquet({transactions}) t
                SEMI JOIN m213_users u USING(customer_id)
                WHERE t.t_dat>={cutoff_sql} AND t.t_dat<{cutoff_sql}+INTERVAL 7 DAY
            ) q
            LEFT JOIN m213_warm_catalog w USING(article_id);
            CREATE OR REPLACE TEMP TABLE m213_truth_counts AS
            SELECT customer_id,count(*)::BIGINT AS truth_count
            FROM m213_truth GROUP BY customer_id;
            """
        )
        truth_users = int(
            connection.execute("SELECT count(*) FROM m213_truth_counts").fetchone()[0]
        )
        results: dict[str, Any] = {}
        for index, name in enumerate(variant_names):
            ordering = inactive_fallback_order_expression(f"score_{name}")
            ranked = f"m213_ranked_{index}"
            top = f"m213_top_{index}"
            connection.execute(
                f"""
                CREATE OR REPLACE TEMP TABLE {ranked} AS
                SELECT p.*,row_number() OVER(
                    PARTITION BY customer_id
                    ORDER BY {ordering},candidate_rank,article_id
                ) AS pred_rank
                FROM predictions p;
                CREATE OR REPLACE TEMP TABLE {top} AS
                WITH joined AS (
                    SELECT r.*,tc.truth_count,
                           (t.article_id IS NOT NULL)::INTEGER AS is_hit,
                           t.item_temperature
                    FROM {ranked} r
                    JOIN m213_truth_counts tc USING(customer_id)
                    LEFT JOIN m213_truth t USING(customer_id,article_id)
                    WHERE r.pred_rank<={metric_k}
                )
                SELECT *,sum(is_hit) OVER(
                    PARTITION BY customer_id ORDER BY pred_rank
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_hits
                FROM joined;
                """
            )
            candidate = connection.execute(
                f"""
                SELECT
                  count(*) FILTER(WHERE r.image_is_new=1),
                  count(*) FILTER(WHERE r.image_is_new=1 AND t.article_id IS NOT NULL),
                  count(DISTINCT r.customer_id) FILTER(
                    WHERE r.image_is_new=1 AND t.article_id IS NOT NULL),
                  count(*) FILTER(WHERE r.image_is_new=1 AND t.item_temperature='warm'),
                  count(*) FILTER(WHERE r.image_is_new=1 AND t.item_temperature='cold')
                FROM {ranked} r
                LEFT JOIN m213_truth t USING(customer_id,article_id)
                """
            ).fetchone()
            converted = connection.execute(
                f"""
                SELECT
                  count(*) FILTER(WHERE image_is_new=1 AND is_hit=1),
                  count(DISTINCT customer_id) FILTER(
                    WHERE image_is_new=1 AND is_hit=1),
                  count(*) FILTER(
                    WHERE image_is_new=1 AND is_hit=1 AND item_temperature='warm'),
                  count(*) FILTER(
                    WHERE image_is_new=1 AND is_hit=1 AND item_temperature='cold'),
                  coalesce(sum(CASE WHEN image_is_new=1 AND is_hit=1 THEN
                    cumulative_hits::DOUBLE/pred_rank/
                    least(truth_count,{metric_k}) ELSE 0 END),0.0)/{truth_users}
                FROM {top}
                """
            ).fetchone()
            if candidate is None or converted is None:
                raise RuntimeError("M2.13 image conversion query returned no rows")
            image_truth = int(candidate[1])
            image_top = int(converted[0])
            results[f"{name}__inactive_rrf"] = {
                "image_only_candidate_rows": int(candidate[0]),
                "image_only_truth_pairs": image_truth,
                "image_only_truth_users": int(candidate[2]),
                "image_only_warm_truth_pairs": int(candidate[3]),
                "image_only_cold_truth_pairs": int(candidate[4]),
                "image_only_top12_truth_pairs": image_top,
                "image_only_top12_truth_users": int(converted[1]),
                "image_only_top12_warm_truth_pairs": int(converted[2]),
                "image_only_top12_cold_truth_pairs": int(converted[3]),
                "image_only_truth_pair_conversion_rate": (
                    image_top / image_truth if image_truth else 0.0
                ),
                "image_only_ap_contribution_to_overall_map@12": float(converted[4]),
            }
        return {
            "truth_users": truth_users,
            "orderings": results,
            "definition": (
                "image_is_new=1 means absent from six-source Top100 and appended by "
                "the frozen M2.6 image source"
            ),
        }
    finally:
        connection.close()


def _ordering_values(
    development: dict[str, Any], name: str, segment: str = "overall"
) -> dict[str, float]:
    return {
        dev: float(
            row["evaluation"]["orderings"][name]["segments"][segment]["map@12"]
        )
        for dev, row in development.items()
    }


def _gate(values: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    delta = {dev: values[dev] - reference[dev] for dev in PROTOCOL}
    return {
        "window_delta_map@12": delta,
        "mean_delta_map@12": float(np.mean(list(delta.values()))),
        "accepted": all(value >= 0 for value in delta.values())
        and float(np.mean(list(delta.values()))) > 0,
    }


def summarize(
    *,
    development: dict[str, Any],
    frozen: dict[str, Any],
    m26_metrics: dict[str, Any],
    m212_metrics: dict[str, Any],
) -> dict[str, Any]:
    names = list(feature_sets())
    orderings = {}
    for name in names:
        fallback = f"{name}__inactive_rrf"
        values = _ordering_values(development, fallback)
        orderings[fallback] = {
            "window_map@12": values,
            "mean_map@12": float(np.mean(list(values.values()))),
            "min_map@12": float(np.min(list(values.values()))),
        }
    frozen_values = {
        dev: float(frozen[dev]["segments"]["overall"]["map@12"])
        for dev in PROTOCOL
    }
    frozen_warm = {
        dev: float(frozen[dev]["segments"]["warm"]["map@12"])
        for dev in PROTOCOL
    }
    old_values = m26_metrics["development_summary"]["orderings"][M26_OLD_BEST][
        "window_map@12"
    ]
    image_name = f"{IMAGE_CONTROL}__inactive_rrf"
    primary_name = f"{PRIMARY_MODEL}__inactive_rrf"
    training_gate = _gate(orderings[image_name]["window_map@12"], old_values)
    target_gate = _gate(
        orderings[primary_name]["window_map@12"],
        orderings[image_name]["window_map@12"],
    )
    overall_gate = _gate(orderings[primary_name]["window_map@12"], frozen_values)
    primary_warm = _ordering_values(development, primary_name, "warm")
    warm_gate = _gate(primary_warm, frozen_warm)
    conversion = {
        dev: row["image_conversion"]["orderings"][primary_name]
        for dev, row in development.items()
    }
    image_hits_both = all(
        row["image_only_top12_truth_pairs"] > 0 for row in conversion.values()
    )
    cold_hits_any = any(
        row["image_only_top12_cold_truth_pairs"] > 0 for row in conversion.values()
    )
    cold_hits_both = all(
        row["image_only_top12_cold_truth_pairs"] > 0 for row in conversion.values()
    )
    deployment_accepted = (
        overall_gate["accepted"]
        and warm_gate["accepted"]
        and image_hits_both
        and cold_hits_any
    )
    realization = {}
    parity = {}
    for dev, row in development.items():
        primary_eval = row["evaluation"]["orderings"][primary_name]["segments"][
            "overall"
        ]
        frozen_eval = frozen[dev]["segments"]["overall"]
        oracle = float(primary_eval["oracle_map@12"])
        frozen_oracle = float(frozen_eval["oracle_map@12"])
        realization[dev] = {
            "expanded_oracle_map@12": oracle,
            "frozen_oracle_map@12": frozen_oracle,
            "realized_fraction": realized_oracle_fraction(
                orderings[primary_name]["window_map@12"][dev],
                frozen_values[dev],
                oracle,
                frozen_oracle,
            ),
            **conversion[dev],
        }
        old_overall = m26_metrics["development"][dev]["evaluation"]["orderings"][
            M26_OLD_BEST
        ]["segments"]["overall"]
        parity[dev] = {
            "candidate_recall_delta_vs_m2_6": float(
                primary_eval["candidate_recall@expanded_pool"]
                - old_overall["candidate_recall@expanded_pool"]
            ),
            "oracle_map_delta_vs_m2_6": oracle - float(old_overall["oracle_map@12"]),
        }
    return {
        "orderings": orderings,
        "frozen_m2_2": {
            "window_map@12": frozen_values,
            "mean_map@12": float(np.mean(list(frozen_values.values()))),
        },
        "m2_6_old_best": {
            "ordering": M26_OLD_BEST,
            "window_map@12": old_values,
            "mean_map@12": float(np.mean(list(old_values.values()))),
        },
        "m2_12_no_image_context_not_a_gate": m212_metrics["development_summary"][
            "orderings"
        ]["no_decay_anchor__inactive_rrf"],
        "new_training_protocol_gate_vs_m2_6": training_gate,
        "target_aware_gate_vs_image_control": target_gate,
        "primary_deployment_gate": {
            "accepted": deployment_accepted,
            "overall_vs_frozen": overall_gate,
            "warm_vs_frozen": warm_gate,
            "image_only_top12_truth_in_both_windows": image_hits_both,
            "cold_image_only_top12_truth_in_any_window": cold_hits_any,
            "cold_image_only_top12_truth_in_both_windows": cold_hits_both,
            "rule": (
                "primary improves overall and warm MAP over frozen M2.2 in both "
                "windows, converts image-only truth in both, and converts cold "
                "image-only truth in at least one window"
            ),
        },
        "oracle_realization": realization,
        "candidate_ceiling_parity": parity,
        "final_week": "not_run",
        "evidence_boundary": (
            "development windows were observed previously; no final-week evidence"
        ),
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["development_summary"]
    deployment = summary["primary_deployment_gate"]
    frozen = summary["frozen_m2_2"]
    lines = [
        "# M2.13A image expanded-pool ranking revisit",
        "",
        "## 结论",
        "",
        f"- primary deployment gate：{'通过' if deployment['accepted'] else '未通过'}。",
        f"- new training protocol gate vs M2.6："
        f"{'通过' if summary['new_training_protocol_gate_vs_m2_6']['accepted'] else '未通过'}。",
        f"- target-aware gate vs same-protocol image control："
        f"{'通过' if summary['target_aware_gate_vs_image_control']['accepted'] else '未通过'}。",
        "- 本阶段只含 six-source + image，不含 Item2Vec；final week 未运行。",
        "",
        "## Outer MAP@12（complete expanded pool + inactive fallback）",
        "",
        "| ordering | dev-A | dev-B | mean | worst |",
        "|---|---:|---:|---:|---:|",
        f"| frozen M2.2 | {frozen['window_map@12']['dev_a']:.6f} | "
        f"{frozen['window_map@12']['dev_b']:.6f} | {frozen['mean_map@12']:.6f} | "
        f"{min(frozen['window_map@12'].values()):.6f} |",
    ]
    old = summary["m2_6_old_best"]
    lines.append(
        f"| M2.6 old best | {old['window_map@12']['dev_a']:.6f} | "
        f"{old['window_map@12']['dev_b']:.6f} | {old['mean_map@12']:.6f} | "
        f"{min(old['window_map@12'].values()):.6f} |"
    )
    for name, row in summary["orderings"].items():
        lines.append(
            f"| {name} | {row['window_map@12']['dev_a']:.6f} | "
            f"{row['window_map@12']['dev_b']:.6f} | {row['mean_map@12']:.6f} | "
            f"{row['min_map@12']:.6f} |"
        )
    lines.extend(["", "## Inner rounds", ""])
    lines.append(
        "Inner MAP 只用于 active + candidate-covered groups 的轮数选择，不能与 outer "
        "overall MAP 直接比较。"
    )
    lines.append("")
    for dev, row in result["development"].items():
        lines.append(
            f"- {dev}: {row['inner_train_cutoffs']} -> "
            f"{row['inner_validation_cutoff']}"
        )
        for name, model in row["inner_models"].items():
            lines.append(
                f"  - {name}: round={model['best_iteration']}, "
                f"inner MAP={model['best_score']:.6f}"
            )
    lines.extend(["", "## Image-only Oracle conversion", ""])
    for dev, row in summary["oracle_realization"].items():
        fraction = row["realized_fraction"]
        fraction_label = "n/a" if fraction is None else f"{fraction:.2%}"
        lines.append(
            f"- {dev}: image-only truth={row['image_only_truth_pairs']}, "
            f"Top12={row['image_only_top12_truth_pairs']}, cold Top12="
            f"{row['image_only_top12_cold_truth_pairs']}, AP contribution="
            f"{row['image_only_ap_contribution_to_overall_map@12']:.6f}, "
            f"expanded Oracle realization={fraction_label}。"
        )
    lines.extend(
        [
            "",
            "## 审计边界",
            "",
            "- M2.6 v4 candidate/features/manifests 均按 path/bytes/SHA fail-closed 复用。",
            "- target-aware 特征重新生成在 image pool 上；身份双向 EXCEPT ALL=0，行为严格早于 cutoff。",
            "- sampled rows 只用于训练；inner/outer validation 都 score 完整 100--300 候选。",
            "- image_is_new=1 才算图片相对 six-source 的 marginal candidate。",
            "- 未新增召回、未调 image TopK/权重/阈值、未混入 Item2Vec、未运行 final week。",
            "- outer development windows 已被旧实验观察，不是 pristine blind test。",
            "",
            "## 产物",
            "",
            f"- metrics：{result['artifacts']['metrics']}",
            f"- private artifacts：{result['artifacts']['artifact_dir']}（Git ignored）",
            f"- image target-aware cache：{result['artifacts']['feature_cache_dir']}（Git ignored）",
            "",
        ]
    )
    return "\n".join(lines)


def run_m213(
    *,
    raw_dir: Path,
    transactions_path: Path,
    m21_metrics_path: Path,
    m26_metrics_path: Path,
    m212_metrics_path: Path,
    source_cache_dir: Path,
    feature_cache_dir: Path,
    m26_artifact_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    config.validate()
    validate_inner_protocol()
    sets = feature_sets()
    if config.evaluation_role != "development" or config.candidate_k != 300:
        raise ValueError("M2.13 requires development-only candidate_k=300")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    m26_metrics = _read_json(m26_metrics_path.resolve())
    m212_metrics = _read_json(m212_metrics_path.resolve())
    if (
        m26_metrics.get("schema_version") != "m2.6-image-ranker-development-v1"
        or m26_metrics.get("status") != "measured"
    ):
        raise ValueError("M2.13 requires measured M2.6 v4 evidence")
    if (
        m212_metrics.get("schema_version") != "m2.12-inner-temporal-selection-v1"
        or m212_metrics.get("status") != "measured"
    ):
        raise ValueError("M2.13 requires measured M2.12 evidence")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        print("M2.13: validating M2.6 source and building image target cache", flush=True)
        cache_started = time.perf_counter()
        feature_cache = build_image_target_cache(
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            source_cache_dir=source_cache_dir,
            cache_dir=feature_cache_dir,
        )
        cache_seconds = time.perf_counter() - cache_started
        m21 = _read_json(m21_metrics_path.resolve())
        frozen_name, frozen = _frozen_reference(m21)
        development: dict[str, Any] = {}
        for dev, protocol in INNER_PROTOCOL.items():
            print(f"M2.13 {dev}: loading inner image-aware sample", flush=True)
            dev_dir = artifact_dir / dev
            dev_dir.mkdir()
            stage_seconds: dict[str, float] = {}
            stage_started = time.perf_counter()
            inner_train_paths = [
                _target_paths(feature_cache_dir, cutoff)[0]
                for cutoff in protocol["inner_train"]
            ]
            inner_train, inner_sizes, inner_sampling = load_image_distribution_sample(
                inner_train_paths, sets[PRIMARY_MODEL], seed=SAMPLING_SEED
            )
            stage_seconds["inner_sample_load"] = time.perf_counter() - stage_started
            inner_maps = build_category_maps(inner_train)
            inner_map_evidence = _save_category_maps(
                dev_dir / "inner-category-maps.json", inner_maps
            )
            stage_started = time.perf_counter()
            (
                inner_validation,
                inner_validation_sizes,
                truth_counts,
                inner_validation_evidence,
            ) = load_inner_validation(
                dataset_path=_target_paths(
                    feature_cache_dir, protocol["inner_validation"]
                )[0],
                transactions_path=transactions_path,
                cutoff=protocol["inner_validation"],
                features=sets[PRIMARY_MODEL],
            )
            stage_seconds["inner_validation_load"] = (
                time.perf_counter() - stage_started
            )
            inner_models: dict[str, Any] = {}
            stage_started = time.perf_counter()
            for name, features in sets.items():
                print(f"M2.13 {dev}: inner fit {name}", flush=True)
                _, evidence = train_inner_ranker(
                    train_frame=inner_train,
                    train_group_sizes=inner_sizes,
                    train_evidence=inner_sampling,
                    validation_frame=inner_validation,
                    validation_group_sizes=inner_validation_sizes,
                    validation_truth_counts=truth_counts,
                    validation_evidence=inner_validation_evidence,
                    features=features,
                    name=name,
                    artifact_dir=dev_dir,
                    config=config,
                    category_maps=inner_maps,
                )
                inner_models[name] = evidence
            stage_seconds["inner_fit_all"] = time.perf_counter() - stage_started
            del inner_train, inner_sizes, inner_validation, inner_validation_sizes
            gc.collect()

            print(f"M2.13 {dev}: loading outer image-aware sample", flush=True)
            stage_started = time.perf_counter()
            outer_paths = [
                _target_paths(feature_cache_dir, cutoff)[0]
                for cutoff in protocol["outer_train"]
            ]
            outer_frame, outer_sizes, outer_sampling = load_image_distribution_sample(
                outer_paths, sets[PRIMARY_MODEL], seed=SAMPLING_SEED
            )
            stage_seconds["outer_sample_load"] = time.perf_counter() - stage_started
            source_maps_path = m26_artifact_dir / dev / "category-maps.json"
            outer_maps = _load_category_maps(source_maps_path)
            outer_map_evidence = _save_category_maps(
                dev_dir / "outer-category-maps.json", outer_maps
            )
            outer_map_evidence["reused_from"] = _file_identity(source_maps_path)
            models: dict[str, tuple[Any, list[str], dict[str, dict[int, int]]]] = {}
            outer_models: dict[str, Any] = {}
            stage_started = time.perf_counter()
            for name, features in sets.items():
                rounds = int(inner_models[name]["best_iteration"])
                print(f"M2.13 {dev}: outer fit {name} rounds={rounds}", flush=True)
                model, evidence = train_lambdarank(
                    frame=outer_frame,
                    group_sizes=outer_sizes,
                    group_evidence=outer_sampling,
                    features=features,
                    name=name,
                    use_ipw=False,
                    artifact_dir=dev_dir,
                    config=replace(config, num_boost_round=rounds),
                    category_maps=outer_maps,
                )
                evidence["selected_rounds_from_inner"] = rounds
                evidence["inner_selection_score"] = inner_models[name]["best_score"]
                models[name] = (model, features, outer_maps)
                outer_models[name] = evidence
            stage_seconds["outer_fit_all"] = time.perf_counter() - stage_started
            del outer_frame, outer_sizes
            gc.collect()

            print(f"M2.13 {dev}: scoring complete image expanded pool", flush=True)
            stage_started = time.perf_counter()
            validation_path = _target_paths(
                feature_cache_dir, protocol["outer_validation"]
            )[0]
            scoring = score_models(
                dataset_path=validation_path,
                models=models,
                evaluation_db=dev_dir / "evaluation.duckdb",
                prediction_path=dev_dir / "outer-validation-predictions.parquet",
                config=config,
            )
            stage_seconds["outer_scoring"] = time.perf_counter() - stage_started
            variant_names = list(models)
            del models
            gc.collect()
            stage_started = time.perf_counter()
            evaluation = _evaluate_models(
                evaluation_db=dev_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                metric_k=config.metric_k,
                variant_names=variant_names,
            )
            stage_seconds["outer_evaluation"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            evaluation = _attach_popularity_segments(
                evaluation=evaluation,
                evaluation_db=dev_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                variant_names=variant_names,
                budget=config.candidate_k,
            )
            stage_seconds["popularity_evaluation"] = time.perf_counter() - stage_started
            stage_started = time.perf_counter()
            image_conversion = evaluate_image_conversion(
                evaluation_db=dev_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                variant_names=variant_names,
                metric_k=config.metric_k,
            )
            stage_seconds["image_conversion_evaluation"] = (
                time.perf_counter() - stage_started
            )
            development[dev] = {
                "inner_train_cutoffs": protocol["inner_train"],
                "inner_validation_cutoff": protocol["inner_validation"],
                "inner_sampling": inner_sampling,
                "inner_models": inner_models,
                "inner_category_encoding": inner_map_evidence,
                "outer_train_cutoffs": protocol["outer_train"],
                "outer_validation_cutoff": protocol["outer_validation"],
                "outer_sampling": outer_sampling,
                "outer_models": outer_models,
                "outer_category_encoding": outer_map_evidence,
                "scoring": scoring,
                "evaluation": evaluation,
                "image_conversion": image_conversion,
                "stage_seconds": stage_seconds,
            }
        summary = summarize(
            development=development,
            frozen=frozen,
            m26_metrics=m26_metrics,
            m212_metrics=m212_metrics,
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M2.13A",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "rolling_development_only_no_final_week",
            "config": asdict(config),
            "contract": {
                "candidate_pool": "exact M2.6 v4 six-source Top100 plus up to 200 image-only",
                "item2vec": "excluded",
                "sampling": "baseline/image-only x rank-tertile hash 30x unweighted",
                "inner_protocol": INNER_PROTOCOL,
                "feature_sets": sets,
                "primary_model": PRIMARY_MODEL,
                "max_boost_round": config.num_boost_round,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "outer_validation": "complete variable 100..300 with inactive RRF fallback",
                "frozen_reference": frozen_name,
                "final_week": "not_run",
            },
            "inputs": {
                "m21_metrics": _file_identity(m21_metrics_path),
                "m26_metrics": _file_identity(m26_metrics_path),
                "m212_metrics": _file_identity(m212_metrics_path),
                "transactions": _file_identity(transactions_path),
            },
            "feature_cache": feature_cache,
            "feature_cache_seconds": cache_seconds,
            "development": development,
            "development_summary": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M2_13_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path.resolve()),
            "report": str(report_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
            "feature_cache_dir": str(feature_cache_dir.resolve()),
        }
        _write_json(metrics_path, result)
        report_path.write_text(render_report(result), encoding="utf-8")
        return result
    except Exception as error:
        _write_json(
            output_dir / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "stage": "M2.13A",
                "status": "failed",
                "run_id": run_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
