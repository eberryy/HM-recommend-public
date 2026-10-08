from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .audit import prepare_tabular_connection
from .m1 import SOURCES
from .m2 import (
    CATEGORICAL_FEATURES,
    FULL_FEATURES,
    M2Config,
    RETRIEVAL_FEATURES,
    _create_static_dimensions,
    _evaluate_ordering,
    _literal,
    _prepare_evaluation_truth,
    _prepare_frame,
    _sha256,
    _write_json,
    build_category_maps,
    build_point_in_time_dataset,
    validate_candidate_artifact,
)
from .m21 import inactive_fallback_order_expression
from .m25_retrieval import _load_neighbors


SCHEMA_VERSION = "m2.6-image-ranker-development-v1"
CACHE_SCHEMA_VERSION = "m2.6-expanded-cache-v2"
IMAGE_RETRIEVAL_FEATURES = [
    "image_present",
    "image_is_new",
    "image_rank",
    "image_score",
    "image_cosine",
    "image_mean_cosine",
    "image_top3_mean_cosine",
    "image_mean_weighted_score",
    "image_top3_mean_weighted_score",
    "best_seed_rank",
    "best_neighbor_rank",
    "seed_support",
    "user_image_seed_count",
]
IMAGE_SEMANTIC_FEATURES = [
    "image_same_product_code",
    "image_same_colour_master_id",
    "image_same_product_different_colour",
]
IMAGE_FEATURES = IMAGE_RETRIEVAL_FEATURES + IMAGE_SEMANTIC_FEATURES
EXPANDED_RETRIEVAL_FEATURES = RETRIEVAL_FEATURES + IMAGE_FEATURES
FEATURE_SETS = {
    "expanded_full_no_image": FULL_FEATURES,
    "expanded_full_plus_image_retrieval": FULL_FEATURES + IMAGE_RETRIEVAL_FEATURES,
    "expanded_full_plus_image_semantic": FULL_FEATURES + IMAGE_FEATURES,
}
REQUIRED_CUTOFFS = ("2020-05-27", "2020-06-24", "2020-07-22", "2020-08-19")
PROTOCOL = {
    "dev_a": {
        "train": ["2020-05-27", "2020-06-24"],
        "validation": "2020-07-22",
    },
    "dev_b": {
        "train": ["2020-06-24", "2020-07-22"],
        "validation": "2020-08-19",
    },
}


@dataclass(frozen=True)
class M26Window:
    cutoff: str
    candidate_path: Path
    manifest_path: Path


def validate_group_sizes(
    sizes: list[int], minimum: int = 100, maximum: int = 300
) -> dict[str, int]:
    if not sizes:
        raise ValueError("ranking groups are empty")
    invalid = [size for size in sizes if size < minimum or size > maximum]
    if invalid:
        raise ValueError(
            f"ranking groups must contain {minimum}..{maximum} rows; "
            f"invalid_groups={len(invalid)}"
        )
    return {
        "groups": len(sizes),
        "rows": int(sum(sizes)),
        "min_group_rows": int(min(sizes)),
        "max_group_rows": int(max(sizes)),
    }


def realized_oracle_fraction(
    model_map: float, frozen_map: float, expanded_oracle: float, frozen_oracle: float
) -> float | None:
    ceiling = expanded_oracle - frozen_oracle
    if ceiling <= 0:
        return None
    return (model_map - frozen_map) / ceiling


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _date(value: str) -> str:
    return "DATE '" + value.replace("'", "''") + "'"


def _build_extended_image_candidates(
    connection: duckdb.DuckDBPyConnection,
    seeds: pd.DataFrame,
    indices: np.ndarray,
    scores: np.ndarray,
    article_ids: np.ndarray,
    max_candidates: int,
    user_chunk: int = 500,
) -> dict[str, Any]:
    connection.execute(
        """
        CREATE TABLE image_candidates (
            customer_id VARCHAR, article_id VARCHAR, image_rank INTEGER,
            image_score DOUBLE, image_cosine DOUBLE,
            image_mean_cosine DOUBLE, image_top3_mean_cosine DOUBLE,
            image_mean_weighted_score DOUBLE,
            image_top3_mean_weighted_score DOUBLE,
            best_seed_rank INTEGER, best_neighbor_rank INTEGER,
            seed_support INTEGER
        )
        """
    )
    users = seeds["customer_id"].drop_duplicates().tolist()
    rows_written = 0
    for chunk_start in range(0, len(users), user_chunk):
        chunk_users = set(users[chunk_start:chunk_start+user_chunk])
        chunk = seeds[seeds["customer_id"].isin(chunk_users)]
        records: list[dict[str, Any]] = []
        for customer_id, group in chunk.groupby("customer_id", sort=True):
            candidates: dict[int, dict[str, Any]] = {}
            for seed in group.itertuples(index=False):
                seed_row = int(seed.row_index)
                seed_rank = int(seed.seed_rank)
                discount = 1.0/(1.0+0.1*(seed_rank-1))
                for neighbor_rank, (neighbor_row, cosine) in enumerate(
                    zip(indices[seed_row], scores[seed_row]), start=1
                ):
                    row = int(neighbor_row)
                    raw = float(cosine)
                    weighted = raw*discount
                    prior = candidates.get(row)
                    if prior is None:
                        candidates[row] = {
                            "best": [weighted, raw, seed_rank, neighbor_rank],
                            "raw": [raw],
                            "weighted": [weighted],
                        }
                    else:
                        prior["raw"].append(raw)
                        prior["weighted"].append(weighted)
                        best = prior["best"]
                        if (weighted, raw, -seed_rank, -neighbor_rank) > (
                            float(best[0]), float(best[1]),
                            -int(best[2]), -int(best[3]),
                        ):
                            prior["best"] = [weighted, raw, seed_rank, neighbor_rank]
            ordered = sorted(
                candidates.items(),
                key=lambda value: (
                    -float(value[1]["best"][0]),
                    -float(value[1]["best"][1]),
                    int(value[1]["best"][2]),
                    int(value[1]["best"][3]),
                    value[0],
                ),
            )[:max_candidates]
            for rank, (row, evidence) in enumerate(ordered, start=1):
                best = evidence["best"]
                raw_values = evidence["raw"]
                weighted_values = evidence["weighted"]
                raw_top = sorted(raw_values, reverse=True)[:3]
                weighted_top = sorted(weighted_values, reverse=True)[:3]
                records.append(
                    {
                        "customer_id": customer_id,
                        "article_id": str(article_ids[row]),
                        "image_rank": rank,
                        "image_score": float(best[0]),
                        "image_cosine": float(best[1]),
                        "image_mean_cosine": float(np.mean(raw_values)),
                        "image_top3_mean_cosine": float(np.mean(raw_top)),
                        "image_mean_weighted_score": float(np.mean(weighted_values)),
                        "image_top3_mean_weighted_score": float(np.mean(weighted_top)),
                        "best_seed_rank": int(best[2]),
                        "best_neighbor_rank": int(best[3]),
                        "seed_support": len(raw_values),
                    }
                )
        frame = pd.DataFrame.from_records(records)
        connection.register("image_chunk", frame)
        connection.execute("INSERT INTO image_candidates SELECT * FROM image_chunk")
        connection.unregister("image_chunk")
        rows_written += len(frame)
        print(
            f"extended image candidates users "
            f"{min(chunk_start+user_chunk,len(users))}/{len(users)}, "
            f"rows={rows_written}",
            flush=True,
        )
    return {"users_with_candidates": len(users), "rows": rows_written}

def _candidate_cache_paths(cache_dir: Path, cutoff: str) -> tuple[Path, Path, Path]:
    window_dir = cache_dir / cutoff
    return (
        window_dir / "expanded-candidates.parquet",
        window_dir / "features.parquet",
        window_dir / "candidate-manifest.json",
    )


def _validate_reused_image_source(
    m25_metrics: dict[str, Any], cutoff: str
) -> tuple[Path | None, dict[str, Any] | None]:
    window = m25_metrics.get("windows", {}).get(cutoff)
    if window is None:
        return None, None
    source = window["source"]
    path = Path(source["artifact"]).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(source["artifact_bytes"])
        or _sha256(path) != source["artifact_sha256"]
    ):
        raise ValueError(f"M2.5 image source identity mismatch for {cutoff}")
    return path, {
        "mode": "reuse_verified_m2.5",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": source["artifact_sha256"],
    }


def _base_retrieval_projection(prefix: str) -> str:
    return ", ".join(f"{prefix}.{name}" for name in RETRIEVAL_FEATURES)


def _empty_retrieval_projection() -> str:
    values = [
        "0.0::DOUBLE AS fused_score",
        "0::INTEGER AS source_count",
    ]
    for source in SOURCES:
        values.extend(
            [
                f"0::INTEGER AS {source}_present",
                f"NULL::INTEGER AS {source}_rank",
                f"NULL::DOUBLE AS {source}_score",
                f"NULL::DOUBLE AS {source}_rrf_contribution",
            ]
        )
    return ", ".join(values)


def _candidate_manifest_is_valid(
    manifest_path: Path,
    candidate_path: Path,
    identity: dict[str, Any],
    neighbor_sha: str,
) -> dict[str, Any] | None:
    if not manifest_path.is_file() or not candidate_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    expected = manifest.get("inputs", {})
    artifact = manifest.get("artifact", {})
    if (
        manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or expected.get("baseline_sha256") != identity["candidate_sha256"]
        or expected.get("neighbor_metrics_sha256") != neighbor_sha
        or artifact.get("sha256") != _sha256(candidate_path)
        or int(artifact.get("bytes", -1)) != candidate_path.stat().st_size
    ):
        raise ValueError(f"stale or mismatched M2.6 candidate cache: {candidate_path}")
    return manifest


def build_expanded_candidate_cache(
    *,
    raw_dir: Path,
    transactions_path: Path,
    neighbor_metrics_path: Path,
    m25_metrics_path: Path,
    windows: list[M26Window],
    cache_dir: Path,
    history_weeks: int = 12,
    seed_k: int = 5,
    image_candidate_k: int = 300,
    append_k: int = 200,
    threads: int = 8,
) -> dict[str, Any]:
    by_cutoff = {window.cutoff: window for window in windows}
    if tuple(sorted(by_cutoff)) != REQUIRED_CUTOFFS:
        raise ValueError(f"M2.6 requires windows {REQUIRED_CUTOFFS}")
    if image_candidate_k != 300 or append_k != 200:
        raise ValueError("M2.6 freezes image_candidate_k=300 and append_k=200")
    cache_dir.mkdir(parents=True, exist_ok=True)
    neighbor_path = neighbor_metrics_path.resolve()
    neighbor_sha = _sha256(neighbor_path)
    _, indices, scores, items = _load_neighbors(neighbor_path)
    article_ids = items.sort_values("row_index")["article_id"].to_numpy()
    m25_metrics = _read_json(m25_metrics_path.resolve())
    if m25_metrics.get("schema_version") != "m2.5-image-retrieval-dev-v1":
        raise ValueError("requires measured M2.5 retrieval metrics")
    results: dict[str, Any] = {}
    for cutoff in REQUIRED_CUTOFFS:
        started = time.perf_counter()
        window = by_cutoff[cutoff]
        identity = validate_candidate_artifact(
            window.candidate_path, window.manifest_path, 100
        )
        if identity["cutoff"] != cutoff or identity["sample_rate"] != 0.1:
            raise ValueError(f"window identity mismatch for {cutoff}")
        candidate_path, _, manifest_path = _candidate_cache_paths(cache_dir, cutoff)
        existing = _candidate_manifest_is_valid(
            manifest_path, candidate_path, identity, neighbor_sha
        )
        if existing is not None:
            results[cutoff] = existing
            continue
        window_dir = candidate_path.parent
        window_dir.mkdir(parents=True, exist_ok=True)
        if candidate_path.exists() or manifest_path.exists():
            raise FileExistsError(f"incomplete M2.6 cache must be preserved: {window_dir}")
        database_path = window_dir / "candidate-build.duckdb"
        if database_path.exists():
            raise FileExistsError(database_path)
        connection = duckdb.connect(str(database_path))
        connection.execute(f"SET threads={threads}")
        connection.execute("SET memory_limit='11GB'")
        temp_dir = window_dir / "duckdb-temp"
        temp_dir.mkdir(exist_ok=True)
        connection.execute(f"SET temp_directory={_literal(temp_dir)}")
        try:
            connection.register("image_items_frame", items)
            connection.execute(
                f"CREATE TABLE baseline AS SELECT * FROM read_parquet({_literal(window.candidate_path)})"
            )
            connection.execute("CREATE TABLE eval_users AS SELECT DISTINCT customer_id FROM baseline")
            connection.execute("CREATE TABLE image_items AS SELECT * FROM image_items_frame")
            connection.unregister("image_items_frame")
            cutoff_sql = _date(cutoff)
            connection.execute(
                f"""
                CREATE TABLE seeds AS
                WITH latest AS (
                    SELECT t.customer_id,t.article_id,max(t.t_dat) AS latest_t_dat
                    FROM read_parquet({_literal(transactions_path)}) t
                    JOIN eval_users u USING(customer_id)
                    JOIN image_items i USING(article_id)
                    WHERE t.t_dat >= {cutoff_sql}-INTERVAL {history_weeks} WEEK
                      AND t.t_dat < {cutoff_sql}
                    GROUP BY t.customer_id,t.article_id
                ), ranked AS (
                    SELECT latest.*,i.row_index,row_number() OVER(
                        PARTITION BY customer_id ORDER BY latest_t_dat DESC,article_id
                    ) AS seed_rank
                    FROM latest JOIN image_items i USING(article_id)
                )
                SELECT * FROM ranked WHERE seed_rank<={seed_k};
                """
            )
            reused_path, m25_source = _validate_reused_image_source(
                m25_metrics, cutoff
            )
            seeds = connection.execute(
                "SELECT customer_id,row_index,seed_rank FROM seeds "
                "ORDER BY customer_id,seed_rank"
            ).fetchdf()
            source_stats = _build_extended_image_candidates(
                connection,
                seeds,
                indices,
                scores,
                article_ids,
                image_candidate_k,
            )
            image_source: dict[str, Any] = {
                "mode": "generated_for_cross_seed_aggregates",
                **source_stats,
            }
            if reused_path is not None:
                core = (
                    "customer_id,article_id,image_rank,image_score,image_cosine,"
                    "best_seed_rank,best_neighbor_rank,seed_support"
                )
                forward = connection.execute(
                    f"SELECT count(*) FROM (SELECT {core} FROM image_candidates "
                    f"EXCEPT ALL SELECT {core} FROM read_parquet({_literal(reused_path)}))"
                ).fetchone()[0]
                reverse = connection.execute(
                    f"SELECT count(*) FROM (SELECT {core} FROM read_parquet({_literal(reused_path)}) "
                    f"EXCEPT ALL SELECT {core} FROM image_candidates)"
                ).fetchone()[0]
                if int(forward) or int(reverse):
                    raise RuntimeError(
                        f"M2.6 regenerated image core differs from M2.5 for {cutoff}: "
                        f"forward={forward}, reverse={reverse}"
                    )
                image_source.update(
                    {
                        "m2.5_reference": m25_source,
                        "m2.5_core_except_all_forward": int(forward),
                        "m2.5_core_except_all_reverse": int(reverse),
                    }
                )
            articles_path = raw_dir.resolve() / "articles.csv"
            connection.execute(
                f"""
                CREATE TABLE article_semantic AS
                SELECT article_id,
                       try_cast(product_code AS BIGINT) AS product_code,
                       try_cast(perceived_colour_master_id AS INTEGER) AS colour_id
                FROM read_csv_auto({_literal(articles_path)}, header=true, all_varchar=true);
                CREATE TABLE image_enriched AS
                WITH seed_counts AS (
                    SELECT customer_id,count(*) AS seed_count FROM seeds GROUP BY customer_id
                )
                SELECT i.*,s.article_id AS best_seed_article_id,
                       coalesce(sc.seed_count,0)::INTEGER AS user_image_seed_count,
                       coalesce((ci.product_code=cs.product_code)::INTEGER,0) AS image_same_product_code,
                       coalesce((ci.colour_id=cs.colour_id)::INTEGER,0) AS image_same_colour_master_id,
                       coalesce((ci.product_code=cs.product_code AND ci.colour_id<>cs.colour_id)::INTEGER,0)
                           AS image_same_product_different_colour
                FROM image_candidates i
                LEFT JOIN seeds s ON i.customer_id=s.customer_id AND i.best_seed_rank=s.seed_rank
                LEFT JOIN seed_counts sc ON i.customer_id=sc.customer_id
                LEFT JOIN article_semantic ci ON i.article_id=ci.article_id
                LEFT JOIN article_semantic cs ON s.article_id=cs.article_id;
                CREATE TABLE image_new AS
                SELECT i.*,row_number() OVER(
                    PARTITION BY i.customer_id ORDER BY i.image_rank,i.article_id
                ) AS new_rank
                FROM image_enriched i
                WHERE NOT EXISTS(
                    SELECT 1 FROM baseline b
                    WHERE b.customer_id=i.customer_id AND b.article_id=i.article_id
                )
                QUALIFY new_rank<={append_k};
                """
            )
            baseline_projection = _base_retrieval_projection("b")
            empty_projection = _empty_retrieval_projection()
            connection.execute(
                f"""
                CREATE TABLE expanded AS
                SELECT b.customer_id,b.article_id,{baseline_projection},
                       (i.article_id IS NOT NULL)::INTEGER AS image_present,
                       0::INTEGER AS image_is_new,
                       i.image_rank,i.image_score,i.image_cosine,
                       i.image_mean_cosine,i.image_top3_mean_cosine,
                       i.image_mean_weighted_score,i.image_top3_mean_weighted_score,
                       i.best_seed_rank,i.best_neighbor_rank,i.seed_support,
                       coalesce(sc.seed_count,0)::INTEGER AS user_image_seed_count,
                       coalesce(i.image_same_product_code,0)::INTEGER AS image_same_product_code,
                       coalesce(i.image_same_colour_master_id,0)::INTEGER AS image_same_colour_master_id,
                       coalesce(i.image_same_product_different_colour,0)::INTEGER AS image_same_product_different_colour
                FROM baseline b
                LEFT JOIN image_enriched i USING(customer_id,article_id)
                LEFT JOIN (SELECT customer_id,count(*) AS seed_count FROM seeds GROUP BY customer_id) sc USING(customer_id)
                UNION ALL
                SELECT n.customer_id,n.article_id,
                       (100+n.new_rank)::INTEGER AS candidate_rank,{empty_projection},
                       1::INTEGER AS image_present,1::INTEGER AS image_is_new,
                       n.image_rank,n.image_score,n.image_cosine,
                       n.image_mean_cosine,n.image_top3_mean_cosine,
                       n.image_mean_weighted_score,n.image_top3_mean_weighted_score,
                       n.best_seed_rank,n.best_neighbor_rank,n.seed_support,
                       n.user_image_seed_count,n.image_same_product_code,
                       n.image_same_colour_master_id,n.image_same_product_different_colour
                FROM image_new n;
                """
            )
            audit = connection.execute(
                """
                WITH groups AS (
                    SELECT customer_id,count(*) AS group_rows,min(candidate_rank) min_rank,
                           max(candidate_rank) max_rank,
                           count(DISTINCT candidate_rank) distinct_ranks
                    FROM expanded GROUP BY customer_id
                )
                SELECT (SELECT count(*) FROM expanded),
                       (SELECT count(DISTINCT (customer_id,article_id)) FROM expanded),
                       count(*),min(group_rows),max(group_rows),
                       count(*) FILTER(WHERE group_rows<100 OR group_rows>300 OR min_rank<>1
                           OR max_rank<>group_rows OR distinct_ranks<>group_rows),
                       (SELECT count(*) FROM expanded WHERE image_is_new=1),
                       count(*) FILTER(WHERE group_rows=100)
                FROM groups
                """
            ).fetchone()
            if audit is None:
                raise RuntimeError("expanded candidate audit returned no rows")
            rows, unique_rows, users, min_rows, max_rows, invalid, new_rows, base_only_users = map(int, audit)
            if rows != unique_rows or invalid:
                raise RuntimeError("expanded candidate identity audit failed")
            connection.execute(
                f"COPY (SELECT * FROM expanded ORDER BY customer_id,candidate_rank,article_id) "
                f"TO {_literal(candidate_path)} "
                "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
            )
            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "completed",
                "cutoff": cutoff,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract": {
                    "baseline_k": 100,
                    "image_candidate_k": image_candidate_k,
                    "append_image_only_k": append_k,
                    "group_rows": "variable_100_to_300_no_padding",
                    "seed_history_weeks": history_weeks,
                    "seed_k": seed_k,
                },
                "inputs": {
                    "baseline": identity,
                    "baseline_sha256": identity["candidate_sha256"],
                    "neighbor_metrics": str(neighbor_path),
                    "neighbor_metrics_sha256": neighbor_sha,
                    "image_source": image_source,
                },
                "audit": {
                    "rows": rows,
                    "unique_rows": unique_rows,
                    "users": users,
                    "min_group_rows": min_rows,
                    "max_group_rows": max_rows,
                    "image_only_rows": new_rows,
                    "base_only_users": base_only_users,
                    "invalid_groups": invalid,
                },
                "artifact": {
                    "path": str(candidate_path.resolve()),
                    "bytes": candidate_path.stat().st_size,
                    "sha256": _sha256(candidate_path),
                },
                "elapsed_seconds": time.perf_counter()-started,
            }
            _write_json(manifest_path, manifest)
            results[cutoff] = manifest
        finally:
            connection.close()
    return results


def build_feature_cache(
    *,
    raw_dir: Path,
    work_dir: Path,
    cache_dir: Path,
    candidate_manifests: dict[str, Any],
    config: M2Config,
) -> dict[str, Any]:
    connection = prepare_tabular_connection(raw_dir, work_dir)
    results: dict[str, Any] = {}
    try:
        _create_static_dimensions(connection)
        for cutoff in REQUIRED_CUTOFFS:
            candidate_path, feature_path, _ = _candidate_cache_paths(cache_dir, cutoff)
            evidence_path = feature_path.with_name("feature-manifest.json")
            source_sha = candidate_manifests[cutoff]["artifact"]["sha256"]
            if feature_path.is_file() and evidence_path.is_file():
                evidence = _read_json(evidence_path)
                if (
                    evidence.get("candidate_sha256") != source_sha
                    or evidence.get("dataset_sha256") != _sha256(feature_path)
                ):
                    raise ValueError(f"stale M2.6 feature cache: {feature_path}")
                results[cutoff] = evidence
                continue
            if feature_path.exists() or evidence_path.exists():
                raise FileExistsError(f"incomplete feature cache: {feature_path.parent}")
            identity = {
                "cutoff": cutoff,
                "declared_rows": candidate_manifests[cutoff]["audit"]["rows"],
            }
            evidence = build_point_in_time_dataset(
                connection,
                candidate_path,
                identity,
                feature_path,
                config,
                retrieval_features=EXPANDED_RETRIEVAL_FEATURES,
                candidate_group_range=(100, 300),
            )
            evidence.update(
                {
                    "schema_version": "m2.6-feature-cache-v1",
                    "candidate_sha256": source_sha,
                    "dataset_sha256": evidence["dataset_sha256"],
                    "image_features": IMAGE_FEATURES,
                }
            )
            _write_json(evidence_path, evidence)
            results[cutoff] = evidence
    finally:
        connection.close()
    return results

def _load_training_data(
    dataset_paths: list[Path], features: list[str]
) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    sizes: list[int] = []
    positive_groups = 0
    for path in dataset_paths:
        connection = duckdb.connect()
        try:
            selected = ",".join(features + ["target"])
            frames.append(
                connection.execute(
                    f"SELECT {selected} FROM read_parquet({_literal(path)})"
                ).fetchdf()
            )
            group_rows = connection.execute(
                f"""
                SELECT count(*) AS group_rows,sum(target) AS positives
                FROM read_parquet({_literal(path)})
                GROUP BY customer_id ORDER BY customer_id
                """
            ).fetchall()
            sizes.extend(int(row[0]) for row in group_rows)
            positive_groups += sum(int(row[1]) > 0 for row in group_rows)
        finally:
            connection.close()
    audit = validate_group_sizes(sizes)
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != audit["rows"]:
        raise RuntimeError("feature rows differ from LambdaRank group sizes")
    audit.update(
        {
            "positive_groups": positive_groups,
            "zero_positive_groups": audit["groups"]-positive_groups,
            "positive_group_rate": positive_groups/audit["groups"],
        }
    )
    return frame, sizes, audit


def _train_lambdarank(
    *,
    frame: pd.DataFrame,
    group_sizes: list[int],
    group_evidence: dict[str, Any],
    features: list[str],
    name: str,
    artifact_dir: Path,
    config: M2Config,
    category_maps: dict[str, dict[int, int]],
) -> tuple[Any, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required; install dependency.txt") from error
    started = time.perf_counter()
    x_train = _prepare_frame(frame, features, category_maps)
    categorical = [name for name in CATEGORICAL_FEATURES if name in features]
    dataset = lgb.Dataset(
        x_train,
        label=frame["target"].astype(np.uint8),
        group=group_sizes,
        feature_name=features,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    params = {
        "objective": "lambdarank",
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
    model_path = artifact_dir / f"lightgbm-{name}.txt"
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
    evidence = {
        "name": name,
        "objective": "lambdarank",
        "features": features,
        "parameters": params,
        "num_boost_round": config.num_boost_round,
        "group_evidence": group_evidence,
        "train_metrics": {
            metric: float(values[-1])
            for metric, values in evaluations.get("train", {}).items()
        },
        "top_feature_importance": importance[:30],
        "model_path": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "elapsed_seconds": time.perf_counter()-started,
    }
    del dataset, x_train
    gc.collect()
    return model, evidence


def _score_models(
    *,
    dataset_path: Path,
    models: dict[str, tuple[Any, list[str], dict[str, dict[int, int]]]],
    evaluation_db: Path,
    prediction_path: Path,
    config: M2Config,
) -> dict[str, Any]:
    if evaluation_db.exists() or prediction_path.exists():
        raise FileExistsError(evaluation_db if evaluation_db.exists() else prediction_path)
    started = time.perf_counter()
    source = duckdb.connect()
    target = duckdb.connect(str(evaluation_db))
    temp_dir = evaluation_db.parent / "duckdb-temp"
    temp_dir.mkdir(exist_ok=True)
    target.execute(f"SET threads={config.threads}")
    target.execute("SET memory_limit='11GB'")
    target.execute(f"SET temp_directory={_literal(temp_dir)}")
    selected = ",".join(
        ["customer_id", "article_id", "candidate_rank", "target", "user_history_events_12w"]
        + list(dict.fromkeys(FULL_FEATURES + IMAGE_FEATURES))
    )
    cursor = source.execute(
        f"SELECT {selected} FROM read_parquet({_literal(dataset_path)})"
    )
    rows = 0
    batches = 0
    initialized = False
    try:
        while True:
            batch = cursor.fetch_df_chunk(config.prediction_chunk_vectors)
            if batch.empty:
                break
            output = batch[
                ["customer_id", "article_id", "candidate_rank", "target", "user_history_events_12w"]
            ].copy()
            for name, (model, features, maps) in models.items():
                output[f"score_{name}"] = model.predict(
                    _prepare_frame(batch, features, maps)
                )
            target.register("m26_prediction_batch", output)
            if not initialized:
                target.execute(
                    "CREATE TABLE predictions AS SELECT * FROM m26_prediction_batch WHERE FALSE"
                )
                initialized = True
            target.execute("INSERT INTO predictions SELECT * FROM m26_prediction_batch")
            target.unregister("m26_prediction_batch")
            rows += len(output)
            batches += 1
        if not initialized:
            raise RuntimeError("validation dataset produced no rows")
        duplicates = target.execute(
            "SELECT count(*)-count(DISTINCT (customer_id,article_id)) FROM predictions"
        ).fetchone()[0]
        if int(duplicates):
            raise RuntimeError("prediction rows are not unique by customer-item")
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


def _label_expanded_pool_metrics(ordering: dict[str, Any]) -> None:
    for segment in ordering["segments"].values():
        segment["candidate_recall@expanded_pool"] = segment["candidate_recall@100"]
        segment["candidate_hit_rate@expanded_pool"] = segment["candidate_hit_rate@100"]
    for segment in ordering["activity_segments"]:
        segment["candidate_recall@expanded_pool"] = segment["candidate_recall@100"]


def _evaluate_models(
    *,
    evaluation_db: Path,
    transactions_path: Path,
    cutoff: str,
    metric_k: int,
    variant_names: list[str],
) -> dict[str, Any]:
    connection = duckdb.connect(str(evaluation_db))
    try:
        connection.execute("SET threads=8")
        truth = _prepare_evaluation_truth(connection, transactions_path, cutoff)
        orderings: dict[str, Any] = {
            "expanded_append_order": _evaluate_ordering(
                connection, "expanded_append_order", "candidate_rank ASC", metric_k
            )
        }
        for name in variant_names:
            score = f"score_{name}"
            orderings[name] = _evaluate_ordering(
                connection, name, f"{score} DESC", metric_k
            )
            fallback = f"{name}__inactive_rrf"
            orderings[fallback] = _evaluate_ordering(
                connection,
                fallback,
                inactive_fallback_order_expression(score),
                metric_k,
            )
        for ordering in orderings.values():
            _label_expanded_pool_metrics(ordering)
        return {"truth": truth, "orderings": orderings}
    finally:
        connection.close()


def _save_category_maps(path: Path, maps: dict[str, dict[int, int]]) -> dict[str, Any]:
    _write_json(
        path,
        {
            feature: {str(raw): encoded for raw, encoded in mapping.items()}
            for feature, mapping in maps.items()
        },
    )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "cardinalities": {feature: len(mapping) for feature, mapping in maps.items()},
    }


def _frozen_reference(m21_metrics: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    selected = m21_metrics["development_summary"]["selected_development_candidate"]
    if selected != "pooled_lambdarank_full__inactive_rrf":
        raise ValueError(f"unexpected frozen M2.2 model form: {selected}")
    return selected, {
        dev: m21_metrics["development"][dev]["evaluation"]["orderings"][selected]
        for dev in PROTOCOL
    }


def _summarize(
    development: dict[str, Any], frozen: dict[str, Any], metric_k: int
) -> dict[str, Any]:
    metric = f"map@{metric_k}"
    variants = list(FEATURE_SETS)
    ordering_names = ["expanded_append_order"] + [
        name for variant in variants for name in (variant, f"{variant}__inactive_rrf")
    ]
    summary: dict[str, Any] = {}
    for name in ordering_names:
        values = {
            dev: float(result["evaluation"]["orderings"][name]["segments"]["overall"][metric])
            for dev, result in development.items()
        }
        summary[name] = {
            "window_map@12": values,
            "mean_map@12": float(np.mean(list(values.values()))),
            "min_map@12": float(np.min(list(values.values()))),
        }
    frozen_values = {
        dev: float(ordering["segments"]["overall"][metric])
        for dev, ordering in frozen.items()
    }
    frozen_mean = float(np.mean(list(frozen_values.values())))
    eligible = [f"{name}__inactive_rrf" for name in variants]
    selected = max(
        eligible,
        key=lambda name: (summary[name]["mean_map@12"], summary[name]["min_map@12"], name),
    )
    selected_values = summary[selected]["window_map@12"]
    window_deltas = {
        dev: selected_values[dev]-frozen_values[dev] for dev in PROTOCOL
    }
    gate = {
        "accepted": all(delta >= 0 for delta in window_deltas.values())
        and summary[selected]["mean_map@12"] > frozen_mean,
        "rule": "mean MAP@12 improves over frozen M2.2 and neither dev window regresses",
        "selected_variant": selected,
        "window_delta_vs_frozen_map@12": window_deltas,
        "mean_delta_vs_frozen_map@12": summary[selected]["mean_map@12"]-frozen_mean,
    }
    realization: dict[str, Any] = {}
    for dev, result in development.items():
        expanded = result["evaluation"]["orderings"][selected]["segments"]["overall"]
        frozen_segment = frozen[dev]["segments"]["overall"]
        realization[dev] = {
            "expanded_oracle_map@12": float(expanded[f"oracle_map@{metric_k}"]),
            "frozen_oracle_map@12": float(frozen_segment[f"oracle_map@{metric_k}"]),
            "realized_fraction": realized_oracle_fraction(
                selected_values[dev],
                frozen_values[dev],
                float(expanded[f"oracle_map@{metric_k}"]),
                float(frozen_segment[f"oracle_map@{metric_k}"]),
            ),
        }
    return {
        "orderings": summary,
        "frozen_m2_2": {"window_map@12": frozen_values, "mean_map@12": frozen_mean},
        "selection_gate": gate,
        "oracle_realization": realization,
        "final_week": "not_run",
    }


def _render_report(result: dict[str, Any]) -> str:
    summary = result["development_summary"]
    gate = summary["selection_gate"]
    lines = [
        "# M2.6 expanded image candidate ranking",
        "",
        "## 结论",
        "",
        f"- development gate：{'通过' if gate['accepted'] else '未通过'}；选择候选为 `{gate['selected_variant']}`。",
        f"- 相对冻结 M2.2 mean MAP@12：{gate['mean_delta_vs_frozen_map@12']:+.6f}。",
        "- 只训练 pooled LambdaRank full；200 rounds、树参数、两窗口协议和 inactive RRF fallback 均冻结。",
        "- final week 未运行；本报告不是 Kaggle leaderboard 分数。",
        "",
        "## MAP@12",
        "",
        "| ordering | dev-A | dev-B | mean | worst |",
        "|---|---:|---:|---:|---:|",
    ]
    frozen = summary["frozen_m2_2"]
    lines.append(
        f"| frozen M2.2 | {frozen['window_map@12']['dev_a']:.6f} | "
        f"{frozen['window_map@12']['dev_b']:.6f} | {frozen['mean_map@12']:.6f} | "
        f"{min(frozen['window_map@12'].values()):.6f} |"
    )
    for name, row in summary["orderings"].items():
        lines.append(
            f"| {name} | {row['window_map@12']['dev_a']:.6f} | "
            f"{row['window_map@12']['dev_b']:.6f} | {row['mean_map@12']:.6f} | "
            f"{row['min_map@12']:.6f} |"
        )
    lines.extend(["", "## Oracle 兑现", ""])
    for dev, value in summary["oracle_realization"].items():
        fraction = value["realized_fraction"]
        label = "n/a" if fraction is None else f"{fraction:.2%}"
        lines.append(
            f"- {dev}：expanded Oracle={value['expanded_oracle_map@12']:.6f}，"
            f"frozen Oracle={value['frozen_oracle_map@12']:.6f}，兑现率={label}。"
        )
    lines.extend(
        [
            "",
            "## 审计边界",
            "",
            "- 05-27、06-24、07-22、08-19 都用各自 cutoff 前 12 周的最近 5 个不同图像种子；验证周从不反向参与训练候选生成。",
            "- 候选组允许 100–300 行；无图像种子的用户不填充伪候选，rank 必须从 1 连续且 user-item 唯一。",
            "- expanded pool 的共享评测函数历史字段名仍为 `candidate_recall@100`；M2.6 额外写出语义正确的 `candidate_recall@expanded_pool`，两者数值相同。",
            "- retrieval 超参数完全继承 M2.5；本阶段的受控变量只有 expanded pool 上的 image retrieval/semantic feature group。",
            "- optimistic_all_articles 假设、未观察负样本偏差和 retrieval 先前在 dev 窗口选择过的非 nested 边界仍然存在。",
            "",
            "## 产物",
            "",
            f"- metrics：{result['artifacts']['metrics']}",
            f"- private artifact dir：{result['artifacts']['artifact_dir']}（Git ignored）",
            "",
        ]
    )
    return "\n".join(lines)


def run_m26(
    *,
    raw_dir: Path,
    work_dir: Path,
    transactions_path: Path,
    neighbor_metrics_path: Path,
    m25_metrics_path: Path,
    m21_metrics_path: Path,
    windows: list[M26Window],
    cache_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    if config.evaluation_role != "development":
        raise ValueError("M2.6 is development-only")
    if config.candidate_k != 300:
        raise ValueError("M2.6 candidate_k records the maximum expanded budget 300")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        candidate_manifests = build_expanded_candidate_cache(
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            neighbor_metrics_path=neighbor_metrics_path,
            m25_metrics_path=m25_metrics_path,
            windows=windows,
            cache_dir=cache_dir,
            history_weeks=config.history_weeks,
            threads=config.threads,
        )
        feature_manifests = build_feature_cache(
            raw_dir=raw_dir,
            work_dir=work_dir,
            cache_dir=cache_dir,
            candidate_manifests=candidate_manifests,
            config=config,
        )
        m21_metrics = _read_json(m21_metrics_path.resolve())
        frozen_name, frozen = _frozen_reference(m21_metrics)
        development: dict[str, Any] = {}
        for dev, protocol in PROTOCOL.items():
            dev_dir = artifact_dir/dev
            dev_dir.mkdir()
            train_paths = [
                _candidate_cache_paths(cache_dir, cutoff)[1]
                for cutoff in protocol["train"]
            ]
            all_features = list(dict.fromkeys(FULL_FEATURES + IMAGE_FEATURES))
            frame, group_sizes, group_evidence = _load_training_data(
                train_paths, all_features
            )
            maps = build_category_maps(frame)
            map_evidence = _save_category_maps(dev_dir/"category-maps.json", maps)
            models: dict[str, tuple[Any, list[str], dict[str, dict[int, int]]]] = {}
            model_evidence: dict[str, Any] = {}
            for name, features in FEATURE_SETS.items():
                model, evidence = _train_lambdarank(
                    frame=frame,
                    group_sizes=group_sizes,
                    group_evidence=group_evidence,
                    features=features,
                    name=name,
                    artifact_dir=dev_dir,
                    config=config,
                    category_maps=maps,
                )
                models[name] = (model, features, maps)
                model_evidence[name] = evidence
            del frame
            gc.collect()
            valid_cutoff = protocol["validation"]
            valid_path = _candidate_cache_paths(cache_dir, valid_cutoff)[1]
            scoring = _score_models(
                dataset_path=valid_path,
                models=models,
                evaluation_db=dev_dir/"evaluation.duckdb",
                prediction_path=dev_dir/"validation-predictions.parquet",
                config=config,
            )
            del models
            gc.collect()
            evaluation = _evaluate_models(
                evaluation_db=dev_dir/"evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=valid_cutoff,
                metric_k=config.metric_k,
                variant_names=list(FEATURE_SETS),
            )
            development[dev] = {
                "train_cutoffs": protocol["train"],
                "validation_cutoff": valid_cutoff,
                "models": model_evidence,
                "category_encoding": map_evidence,
                "scoring": scoring,
                "evaluation": evaluation,
            }
        summary = _summarize(development, frozen, config.metric_k)
        result = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M2.6",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "development_only_no_final_week",
            "config": asdict(config),
            "contract": {
                "candidate_pool": "baseline_top100_plus_up_to_200_image_only",
                "variable_group_rows": [100, 300],
                "negative_sampling": "none_keep_all_candidates",
                "objective": "pooled_lambdarank_full_only",
                "frozen_reference": frozen_name,
                "feature_sets": FEATURE_SETS,
            },
            "candidate_cache": candidate_manifests,
            "feature_cache": feature_manifests,
            "development": development,
            "development_summary": summary,
            "elapsed_seconds": time.perf_counter()-started,
        }
        metrics_path = output_dir/"metrics.json"
        report_path = output_dir/"M2_6_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path.resolve()),
            "report": str(report_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
            "cache_dir": str(cache_dir.resolve()),
        }
        _write_json(metrics_path, result)
        report_path.write_text(_render_report(result), encoding="utf-8")
        return result
    except Exception as error:
        _write_json(
            output_dir/f"failure-{time.time_ns()}.json",
            {
                "schema_version": "m2.6-failure-v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter()-started,
            },
        )
        raise
