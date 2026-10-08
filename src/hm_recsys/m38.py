from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .audit import prepare_tabular_connection
from .m2 import (
    M2Config,
    RETRIEVAL_FEATURES,
    _create_static_dimensions,
    _literal,
    _prepare_frame,
    _sha256,
    _write_json,
    build_category_maps,
    build_point_in_time_dataset,
)
from .m21 import inactive_fallback_order_expression
from .m25_retrieval import _candidate_metrics, _load_neighbors
from .m26 import _save_category_maps
from .m29 import ITEM2VEC_FEATURES, _read_json
from .m210 import _dimension_sql, _file_identity, score_models
from .m211 import NEGATIVE_RATIO, SAMPLING_SEED, train_lambdarank
from .m212 import feature_sets as frozen_feature_sets
from .m212 import train_inner_ranker
from .m213 import _target_select_sql
from .m3 import ANCHOR_NAME
from .m33 import ALL_CUTOFFS, ROLLING_PROTOCOL
from .m35 import OUTER_WINDOWS, build_cold_sources
from .m37 import CONFIG as M37_CONFIG
from .m37 import _build_route_frames, _prepare_context


SCHEMA_VERSION = "m3.8-image-source-factorial-ranking-v1"
CACHE_SCHEMA_VERSION = "m3.8-image-source-cache-v1"
VARIANTS = (
    "base_historical_400",
    "base_soft_shallow_400",
    "base_historical_soft_500",
)
BUDGETS = {
    "base_300": 300,
    "base_historical_400": 400,
    "base_soft_shallow_400": 400,
    "base_historical_soft_500": 500,
}

IMAGE_SOURCE_FEATURES = [
    "base_present",
    "historical_present",
    "historical_rank",
    "historical_personalized_rank",
    "historical_global_rank",
    "historical_score",
    "soft_present",
    "soft_rank",
    "soft_personalized_present",
    "soft_personalized_rank",
    "soft_global_present",
    "soft_global_rank",
    "soft_visual_rank",
    "soft_season_rank",
    "soft_combined_score",
    "soft_visual_score",
    "soft_season_score",
    "soft_best_neighbor_rank",
    "soft_seed_support",
    "historical_soft_overlap",
]


def validate_protocol() -> None:
    if any(cutoff >= "2020-09-16" for cutoff in ALL_CUTOFFS):
        raise RuntimeError("M3.8 must not read final-week interactions")
    if tuple(VARIANTS) != (
        "base_historical_400",
        "base_soft_shallow_400",
        "base_historical_soft_500",
    ):
        raise RuntimeError("M3.8 factorial variants drifted")
    if M37_CONFIG["shallow_neighbor_depth"] != 100:
        raise RuntimeError("M3.8 requires frozen M3.7 shallow Top100")
    if M37_CONFIG["personalized_route_k"] != 50 or M37_CONFIG["global_route_k"] != 50:
        raise RuntimeError("M3.8 requires frozen M3.7 50+50 route split")
    anchor = frozen_feature_sets()[ANCHOR_NAME]
    if len(anchor) != 84 or len(set(anchor)) != 84:
        raise RuntimeError("M3.8 frozen 84-feature anchor drifted")


def _source_paths(cache_dir: Path, cutoff: str) -> tuple[Path, Path, Path]:
    root = cache_dir / cutoff
    return root / "soft-shallow.parquet", root / "context.duckdb", root / "manifest.json"


def _candidate_paths(cache_dir: Path, cutoff: str) -> tuple[Path, Path]:
    root = cache_dir / cutoff
    return root / "union-candidates.parquet", root / "manifest.json"


def _feature_paths(cache_dir: Path, cutoff: str) -> tuple[Path, Path, Path]:
    root = cache_dir / cutoff
    return root / "point-in-time.parquet", root / "target-aware.parquet", root / "manifest.json"


def _variant_path(cache_dir: Path, cutoff: str, variant: str) -> Path:
    return cache_dir / cutoff / f"{variant}.parquet"


def _base_candidates(m29_cache_dir: Path) -> dict[str, tuple[Path, Path]]:
    result: dict[str, tuple[Path, Path]] = {}
    for cutoff in ALL_CUTOFFS:
        root = m29_cache_dir / cutoff
        candidate = root / "expanded-candidates.parquet"
        manifest = root / "candidate-manifest.json"
        if not candidate.is_file() or not manifest.is_file():
            raise FileNotFoundError(candidate if not candidate.is_file() else manifest)
        result[cutoff] = (candidate, manifest)
    return result


def _fake_m35_db(
    *, base_path: Path, historical_path: Path, output_path: Path
) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    connection = duckdb.connect(str(output_path))
    try:
        connection.execute(
            f"""
            CREATE TABLE base AS
              SELECT customer_id,article_id,candidate_rank
              FROM read_parquet({_literal(base_path)});
            CREATE TABLE eval_truth(
              customer_id VARCHAR,article_id VARCHAR,item_temperature VARCHAR
            );
            CREATE TABLE image_source AS
              SELECT * FROM read_parquet({_literal(historical_path)});
            """
        )
    finally:
        connection.close()


def _write_soft_source(
    *, context: dict[str, Any], personal: pd.DataFrame, global_frame: pd.DataFrame, output: Path
) -> dict[str, Any]:
    con: duckdb.DuckDBPyConnection = context["connection"]
    personal = personal.loc[personal["variant"] == "soft_shallow_split"].copy()
    global_frame = global_frame.loc[global_frame["variant"] == "soft_shallow_split"].copy()
    con.register("m38_personal_frame", personal)
    con.register("m38_global_frame", global_frame)
    try:
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE m38_personal AS
              SELECT * FROM m38_personal_frame;
            CREATE OR REPLACE TEMP TABLE m38_global AS
              SELECT * FROM m38_global_frame;
            CREATE OR REPLACE TEMP TABLE m38_soft AS
            WITH global_users AS (
              SELECT u.customer_id,g.* EXCLUDE(variant)
              FROM eval_users u CROSS JOIN m38_global g
            ), merged AS (
              SELECT coalesce(p.customer_id,g.customer_id) AS customer_id,
                     coalesce(p.article_id,g.article_id) AS article_id,
                     p.route_rank AS personalized_rank,
                     g.route_rank AS global_rank,
                     coalesce(p.visual_rank,g.visual_rank) AS visual_rank,
                     coalesce(p.season_rank,g.season_rank) AS season_rank,
                     coalesce(p.combined_score,g.combined_score) AS combined_score,
                     coalesce(p.visual_score,g.visual_score) AS visual_score,
                     coalesce(p.season_score,g.season_score) AS season_score,
                     coalesce(p.best_neighbor_rank,g.best_neighbor_rank) AS best_neighbor_rank,
                     coalesce(p.seed_support,g.seed_support) AS seed_support
              FROM m38_personal p FULL OUTER JOIN global_users g
                USING(customer_id,article_id)
            ), ranked AS (
              SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY
                       CASE WHEN personalized_rank IS NOT NULL THEN 1 ELSE 2 END,
                       coalesce(personalized_rank,global_rank),article_id) AS soft_rank
              FROM merged
            )
            SELECT customer_id,article_id,soft_rank,
                   (personalized_rank IS NOT NULL)::UTINYINT AS soft_personalized_present,
                   personalized_rank AS soft_personalized_rank,
                   (global_rank IS NOT NULL)::UTINYINT AS soft_global_present,
                   global_rank AS soft_global_rank,
                   visual_rank AS soft_visual_rank,season_rank AS soft_season_rank,
                   combined_score AS soft_combined_score,visual_score AS soft_visual_score,
                   season_score AS soft_season_score,best_neighbor_rank AS soft_best_neighbor_rank,
                   seed_support AS soft_seed_support
            FROM ranked WHERE soft_rank<=100;
            """
        )
        exact = con.execute(
            """
            SELECT count(*),count(DISTINCT (customer_id,article_id)),
                   count(DISTINCT customer_id) FROM m38_soft
            """
        ).fetchone()
        groups = con.execute(
            """
            WITH g AS (
              SELECT customer_id,count(*) AS rows,min(soft_rank) AS lo,max(soft_rank) AS hi,
                     count(DISTINCT soft_rank) AS dr FROM m38_soft GROUP BY customer_id)
            SELECT min(rows),max(rows),count(*) FILTER(WHERE lo<>1 OR hi<>rows OR dr<>rows) FROM g
            """
        ).fetchone()
        if int(exact[0]) != int(exact[1]) or int(groups[2]):
            raise RuntimeError("M3.8 soft source identity/rank audit failed")
        con.execute(
            f"COPY (SELECT * FROM m38_soft ORDER BY customer_id,soft_rank) TO {_literal(output)} "
            "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
        )
        return {
            "rows": int(exact[0]),
            "users": int(exact[2]),
            "min_rows": int(groups[0]),
            "max_rows": int(groups[1]),
        }
    finally:
        con.unregister("m38_personal_frame")
        con.unregister("m38_global_frame")


def build_source_cache(
    *,
    raw_dir: Path,
    transactions_path: Path,
    m29_cache_dir: Path,
    neighbor_metrics_path: Path,
    historical_cache_dir: Path,
    soft_cache_dir: Path,
) -> dict[str, Any]:
    base = _base_candidates(m29_cache_dir)
    windows = {f"cutoff_{cutoff.replace('-', '')}": cutoff for cutoff in ALL_CUTOFFS}
    historical = build_cold_sources(
        raw_dir=raw_dir,
        transactions_path=transactions_path,
        base_candidates=base,
        neighbor_metrics_path=neighbor_metrics_path,
        cache_dir=historical_cache_dir,
        windows=windows,
    )
    _, indices, scores, image_items = _load_neighbors(neighbor_metrics_path)
    results: dict[str, Any] = {}
    soft_cache_dir.mkdir(parents=True, exist_ok=True)
    outer_by_cutoff = {cutoff: window for window, cutoff in OUTER_WINDOWS.items()}
    for cutoff in ALL_CUTOFFS:
        started = time.perf_counter()
        output, context_db, manifest_path = _source_paths(soft_cache_dir, cutoff)
        base_path = base[cutoff][0]
        historical_path = Path(historical[cutoff]["artifacts"]["image"]["path"])
        inputs = {
            "base": _file_identity(base_path),
            "historical": _file_identity(historical_path),
            "transactions": _file_identity(transactions_path),
            "articles": _file_identity(raw_dir / "articles.csv"),
            "neighbors": _file_identity(neighbor_metrics_path),
        }
        if output.is_file() and manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
                raise ValueError(f"M3.8 soft source schema drift: {cutoff}")
            if _file_identity(output) != manifest["artifact"]:
                raise ValueError(f"M3.8 soft source artifact drift: {cutoff}")
            results[cutoff] = {"historical": historical[cutoff], "soft": manifest}
            continue
        root = output.parent
        if root.exists():
            raise FileExistsError(f"incomplete M3.8 soft source cache: {root}")
        root.mkdir(parents=True)
        fake_db = root / "m35-input.duckdb"
        _fake_m35_db(base_path=base_path, historical_path=historical_path, output_path=fake_db)
        context = _prepare_context(
            window=f"cutoff_{cutoff.replace('-', '')}",
            cutoff=cutoff,
            m35_db_path=fake_db,
            transactions_path=transactions_path,
            articles_path=raw_dir / "articles.csv",
            image_items=image_items,
            db_path=context_db,
        )
        try:
            query_rows = np.asarray(
                sorted(
                    set(context["seed_frame"]["row_index"].astype(int).tolist())
                    | set(context["global_frame"]["row_index"].astype(int).tolist())
                ),
                dtype=np.int64,
            )
            personal, global_frame, route_audit = _build_route_frames(
                context=context,
                query_rows=query_rows,
                deep_indices=indices[query_rows, :100],
                deep_scores=scores[query_rows, :100],
            )
            source_audit = _write_soft_source(
                context=context, personal=personal, global_frame=global_frame, output=output
            )
            parity: dict[str, Any] = {"checked": False}
            if cutoff in outer_by_cutoff:
                historical_m37 = (
                    Path("artifacts/m3_7/m3-7-v3-deep-visual-soft-season-cpu-prefix")
                    / outer_by_cutoff[cutoff]
                    / "evaluation.duckdb"
                )
                if historical_m37.is_file():
                    other = duckdb.connect(str(historical_m37), read_only=True)
                    try:
                        old = other.execute(
                            "SELECT customer_id,article_id,candidate_rank FROM source_soft_shallow_split_100"
                        ).fetchdf()
                    finally:
                        other.close()
                    current = duckdb.connect().execute(
                        f"SELECT customer_id,article_id,soft_rank AS candidate_rank FROM read_parquet({_literal(output)})"
                    ).fetchdf()
                    old_rows = set(map(tuple, old.itertuples(index=False, name=None)))
                    new_rows = set(map(tuple, current.itertuples(index=False, name=None)))
                    parity = {
                        "checked": True,
                        "old_rows": len(old_rows),
                        "new_rows": len(new_rows),
                        "symmetric_difference": len(old_rows ^ new_rows),
                        "passed": old_rows == new_rows,
                    }
                    if not parity["passed"]:
                        raise RuntimeError(f"M3.8/M3.7 soft-shallow parity failed: {cutoff}")
            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "completed",
                "cutoff": cutoff,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract": {
                    "visual_depth": 100,
                    "season_control": "soft RRF weight 0.5",
                    "routes": "personalized Top50 then global Top50 with de-duplication",
                    "final_week": "not_run",
                },
                "inputs": inputs,
                "route_audit": route_audit,
                "source_audit": source_audit,
                "outer_parity": parity,
                "artifact": _file_identity(output),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(manifest_path, manifest)
            results[cutoff] = {"historical": historical[cutoff], "soft": manifest}
        finally:
            context["connection"].close()
    return results


def _zero_expression(column: str, dtype: str) -> str:
    if column.endswith("_present") or column.endswith("_is_new") or column in {
        "source_count",
        "item2vec_vocab_count",
    }:
        return f"CAST(0 AS {dtype}) AS {column}"
    return f"CAST(NULL AS {dtype}) AS {column}"


def build_candidate_cache(
    *, m29_cache_dir: Path, sources: dict[str, Any], cache_dir: Path
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for cutoff in ALL_CUTOFFS:
        started = time.perf_counter()
        output, manifest_path = _candidate_paths(cache_dir, cutoff)
        base_path = m29_cache_dir / cutoff / "expanded-candidates.parquet"
        historical_path = Path(sources[cutoff]["historical"]["artifacts"]["image"]["path"])
        soft_path = Path(sources[cutoff]["soft"]["artifact"]["path"])
        inputs = {
            "base": _file_identity(base_path),
            "historical": _file_identity(historical_path),
            "soft": _file_identity(soft_path),
        }
        if output.is_file() and manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if manifest.get("schema_version") != "m3.8-union-candidates-v1":
                raise ValueError(f"M3.8 candidate cache schema drift: {cutoff}")
            if _file_identity(output) != manifest["artifact"]:
                raise ValueError(f"M3.8 candidate artifact drift: {cutoff}")
            results[cutoff] = manifest
            continue
        root = output.parent
        if root.exists():
            raise FileExistsError(f"incomplete M3.8 candidate cache: {root}")
        root.mkdir(parents=True)
        con = duckdb.connect()
        try:
            schema = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet({_literal(base_path)})"
            ).fetchall()
            base_columns = [(str(row[0]), str(row[1])) for row in schema]
            payload_columns = [row for row in base_columns if row[0] not in {"customer_id", "article_id", "candidate_rank"}]
            null_payload = ",".join(_zero_expression(name, dtype) for name, dtype in payload_columns)
            base_payload = ",".join(name for name, _ in payload_columns)
            con.execute(
                f"""
                CREATE TABLE base AS SELECT * FROM read_parquet({_literal(base_path)});
                CREATE TABLE historical AS SELECT * FROM read_parquet({_literal(historical_path)});
                CREATE TABLE soft AS SELECT * FROM read_parquet({_literal(soft_path)});
                CREATE TABLE joined AS
                WITH base_sizes AS (SELECT customer_id,count(*) AS base_rows FROM base GROUP BY customer_id),
                hist_new AS (
                  SELECT h.*,row_number() OVER(PARTITION BY h.customer_id ORDER BY h.image_cold_rank,h.article_id) AS new_rank
                  FROM historical h WHERE NOT EXISTS (SELECT 1 FROM base b WHERE b.customer_id=h.customer_id AND b.article_id=h.article_id)),
                soft_new AS (
                  SELECT s.*,row_number() OVER(PARTITION BY s.customer_id ORDER BY s.soft_rank,s.article_id) AS new_rank
                  FROM soft s WHERE NOT EXISTS (SELECT 1 FROM base b WHERE b.customer_id=s.customer_id AND b.article_id=s.article_id)),
                soft_d_new AS (
                  SELECT s.*,row_number() OVER(PARTITION BY s.customer_id ORDER BY s.soft_rank,s.article_id) AS new_rank
                  FROM soft s WHERE NOT EXISTS (SELECT 1 FROM base b WHERE b.customer_id=s.customer_id AND b.article_id=s.article_id)
                    AND NOT EXISTS (SELECT 1 FROM historical h WHERE h.customer_id=s.customer_id AND h.article_id=s.article_id)),
                hist_counts AS (SELECT customer_id,count(*) AS rows FROM hist_new GROUP BY customer_id),
                ids AS (SELECT customer_id,article_id FROM base UNION SELECT customer_id,article_id FROM historical UNION SELECT customer_id,article_id FROM soft),
                e AS (
                  SELECT i.*,bs.base_rows,b.candidate_rank AS base_rank,h.image_cold_rank AS historical_rank,
                    h.personalized_image_rank AS historical_personalized_rank,h.global_visual_rank AS historical_global_rank,
                    h.score AS historical_score,hn.new_rank AS historical_new_rank,s.soft_rank,
                    s.soft_personalized_present,s.soft_personalized_rank,s.soft_global_present,s.soft_global_rank,
                    s.soft_visual_rank,s.soft_season_rank,s.soft_combined_score,s.soft_visual_score,s.soft_season_score,
                    s.soft_best_neighbor_rank,s.soft_seed_support,sn.new_rank AS soft_new_rank,
                    sd.new_rank AS soft_d_new_rank,coalesce(hc.rows,0) AS hist_new_rows
                  FROM ids i JOIN base_sizes bs USING(customer_id) LEFT JOIN base b USING(customer_id,article_id)
                  LEFT JOIN historical h USING(customer_id,article_id) LEFT JOIN hist_new hn USING(customer_id,article_id)
                  LEFT JOIN soft s USING(customer_id,article_id) LEFT JOIN soft_new sn USING(customer_id,article_id)
                  LEFT JOIN soft_d_new sd USING(customer_id,article_id)
                  LEFT JOIN hist_counts hc USING(customer_id)),
                r AS (
                  SELECT *,CASE WHEN base_rank IS NOT NULL THEN base_rank WHEN historical_rank IS NOT NULL THEN base_rows+historical_new_rank END AS rank_b,
                    CASE WHEN base_rank IS NOT NULL THEN base_rank WHEN soft_rank IS NOT NULL THEN base_rows+soft_new_rank END AS rank_c,
                    CASE WHEN base_rank IS NOT NULL THEN base_rank WHEN historical_rank IS NOT NULL THEN base_rows+historical_new_rank
                         ELSE base_rows+hist_new_rows+soft_d_new_rank END AS rank_d FROM e)
                SELECT r.customer_id,r.article_id,r.rank_d AS candidate_rank,
                       {','.join('b.' + name for name, _ in payload_columns)},
                       (r.base_rank IS NOT NULL)::UTINYINT AS base_present,
                       (r.historical_rank IS NOT NULL)::UTINYINT AS historical_present,r.historical_rank,
                       r.historical_personalized_rank,r.historical_global_rank,r.historical_score,
                       (r.soft_rank IS NOT NULL)::UTINYINT AS soft_present,r.soft_rank,
                       coalesce(r.soft_personalized_present,0)::UTINYINT AS soft_personalized_present,
                       r.soft_personalized_rank,coalesce(r.soft_global_present,0)::UTINYINT AS soft_global_present,
                       r.soft_global_rank,r.soft_visual_rank,r.soft_season_rank,r.soft_combined_score,
                       r.soft_visual_score,r.soft_season_score,r.soft_best_neighbor_rank,r.soft_seed_support,
                       (r.historical_rank IS NOT NULL AND r.soft_rank IS NOT NULL)::UTINYINT AS historical_soft_overlap,
                       r.rank_b,r.rank_c,r.rank_d
                FROM r LEFT JOIN base b USING(customer_id,article_id) WHERE r.rank_d IS NOT NULL;
                """
            )
            # Replace NULL base payloads for image-only rows with the frozen missing-value convention.
            select_payload = []
            for name, dtype in payload_columns:
                if name.endswith("_present") or name.endswith("_is_new") or name in {"source_count", "item2vec_vocab_count"}:
                    select_payload.append(f"coalesce({name},0)::${dtype} AS {name}".replace("$", ""))
                else:
                    select_payload.append(name)
            con.execute(
                f"COPY (SELECT customer_id,article_id,candidate_rank,{','.join(select_payload)},"
                + ",".join(IMAGE_SOURCE_FEATURES)
                + ",rank_b,rank_c,rank_d FROM joined ORDER BY customer_id,candidate_rank,article_id) "
                + f"TO {_literal(output)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
            )
            audit = con.execute(
                """
                WITH g AS (SELECT customer_id,count(*) AS rows,min(candidate_rank) AS lo,
                  max(candidate_rank) AS hi,count(DISTINCT candidate_rank) AS dr FROM joined GROUP BY customer_id)
                SELECT (SELECT count(*) FROM joined),(SELECT count(DISTINCT (customer_id,article_id)) FROM joined),
                  count(*),min(rows),max(rows),count(*) FILTER(WHERE lo<>1 OR hi<>rows OR dr<>rows) FROM g
                """
            ).fetchone()
            if int(audit[0]) != int(audit[1]) or int(audit[5]):
                raise RuntimeError(f"M3.8 union candidate audit failed: {cutoff} {audit}")
            manifest = {
                "schema_version": "m3.8-union-candidates-v1",
                "status": "completed",
                "cutoff": cutoff,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": inputs,
                "audit": {
                    "rows": int(audit[0]), "users": int(audit[2]),
                    "min_group_rows": int(audit[3]), "max_group_rows": int(audit[4]),
                },
                "artifact": _file_identity(output),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(manifest_path, manifest)
            results[cutoff] = manifest
        finally:
            con.close()
    return results


def build_feature_cache(
    *, raw_dir: Path, work_dir: Path, transactions_path: Path,
    candidate_cache_dir: Path, candidate_manifests: dict[str, Any], cache_dir: Path,
    config: M2Config,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    connection = prepare_tabular_connection(raw_dir, work_dir)
    results: dict[str, Any] = {}
    retrieval = RETRIEVAL_FEATURES + ITEM2VEC_FEATURES + IMAGE_SOURCE_FEATURES + ["rank_b", "rank_c", "rank_d"]
    try:
        _create_static_dimensions(connection)
        for cutoff in ALL_CUTOFFS:
            started = time.perf_counter()
            point_path, target_path, manifest_path = _feature_paths(cache_dir, cutoff)
            candidate_path = _candidate_paths(candidate_cache_dir, cutoff)[0]
            if target_path.is_file() and manifest_path.is_file():
                manifest = _read_json(manifest_path)
                if manifest.get("schema_version") != "m3.8-feature-cache-v1" or _file_identity(target_path) != manifest["artifact"]:
                    raise ValueError(f"M3.8 feature cache drift: {cutoff}")
                results[cutoff] = manifest
                continue
            root = point_path.parent
            if any(path.exists() for path in (point_path, target_path, manifest_path)):
                raise FileExistsError(f"incomplete M3.8 feature cache: {root}")
            root.mkdir(parents=True, exist_ok=False)
            evidence = build_point_in_time_dataset(
                connection, candidate_path,
                {"cutoff": cutoff, "declared_rows": candidate_manifests[cutoff]["audit"]["rows"]},
                point_path, replace(config, candidate_k=500), retrieval_features=retrieval,
                candidate_group_range=(100, 500),
            )
            con = duckdb.connect()
            try:
                con.execute(f"CREATE VIEW base AS SELECT * FROM read_parquet({_literal(point_path)})")
                con.execute("CREATE TABLE ta_users AS SELECT DISTINCT customer_id FROM base")
                con.execute(
                    f"""
                    CREATE TABLE ta_articles AS SELECT article_id,
                      try_cast(product_code AS INTEGER) AS article_product_code,
                      try_cast(product_type_no AS INTEGER) AS article_product_type_no,
                      try_cast(garment_group_no AS INTEGER) AS article_garment_group_no,
                      try_cast(department_no AS INTEGER) AS article_department_no,
                      try_cast(index_group_no AS INTEGER) AS article_index_group_no,
                      try_cast(perceived_colour_master_id AS INTEGER) AS article_colour_master_id
                    FROM read_csv_auto({_literal(raw_dir / 'articles.csv')},header=true,all_varchar=true);
                    CREATE TABLE ta_history AS SELECT t.customer_id,t.article_id,t.t_dat,
                      a.article_product_code,a.article_product_type_no,a.article_garment_group_no,
                      a.article_department_no,a.article_index_group_no,a.article_colour_master_id
                    FROM read_parquet({_literal(transactions_path)}) t SEMI JOIN ta_users u USING(customer_id)
                    JOIN ta_articles a USING(article_id)
                    WHERE t.t_dat>=DATE '{cutoff}'-INTERVAL 12 WEEK AND t.t_dat<DATE '{cutoff}';
                    """
                )
                for table, key in {
                    "ta_item": "article_id", "ta_product_code": "article_product_code",
                    "ta_product_type": "article_product_type_no", "ta_department": "article_department_no",
                    "ta_garment": "article_garment_group_no", "ta_colour": "article_colour_master_id",
                    "ta_index_group": "article_index_group_no",
                }.items():
                    con.execute(_dimension_sql(table, key, f"DATE '{cutoff}'"))
                con.execute(
                    f"COPY ({_target_select_sql()} ORDER BY customer_id,candidate_rank,article_id) TO {_literal(target_path)} "
                    "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
                )
                audit = con.execute(
                    f"SELECT count(*),count(DISTINCT (customer_id,article_id)),sum(target),max((SELECT max(t_dat) FROM ta_history)) "
                    f"FROM read_parquet({_literal(target_path)})"
                ).fetchone()
                if int(audit[0]) != int(audit[1]) or (audit[3] is not None and str(audit[3]) >= cutoff):
                    raise RuntimeError(f"M3.8 target feature audit failed: {cutoff}")
            finally:
                con.close()
            manifest = {
                "schema_version": "m3.8-feature-cache-v1", "status": "completed", "cutoff": cutoff,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "inputs": {"candidates": _file_identity(candidate_path), "transactions": _file_identity(transactions_path)},
                "point_in_time": evidence, "artifact": _file_identity(target_path),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(manifest_path, manifest)
            results[cutoff] = manifest
    finally:
        connection.close()
    return results


def materialize_variant_cache(*, feature_cache_dir: Path, variant_cache_dir: Path) -> dict[str, Any]:
    variant_cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for cutoff in ALL_CUTOFFS:
        source = _feature_paths(feature_cache_dir, cutoff)[1]
        root = variant_cache_dir / cutoff
        root.mkdir(exist_ok=True)
        results[cutoff] = {}
        for variant in VARIANTS:
            output = _variant_path(variant_cache_dir, cutoff, variant)
            if variant == "base_historical_400":
                predicate, rank, zero_prefix = "rank_b IS NOT NULL", "rank_b", "soft_"
            elif variant == "base_soft_shallow_400":
                predicate, rank, zero_prefix = "rank_c IS NOT NULL", "rank_c", "historical_"
            else:
                predicate, rank, zero_prefix = "rank_d IS NOT NULL", "rank_d", ""
            con = duckdb.connect()
            try:
                if not output.is_file():
                    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet({_literal(source)})").fetchall()
                    columns = [str(row[0]) for row in schema]
                    select = []
                    for column in columns:
                        if column in {"candidate_rank", "rank_b", "rank_c", "rank_d"}:
                            continue
                        if zero_prefix and column.startswith(zero_prefix):
                            select.append(f"0 AS {column}")
                        elif variant == "base_soft_shallow_400" and column == "historical_soft_overlap":
                            select.append("0 AS historical_soft_overlap")
                        elif variant == "base_historical_400" and column == "historical_soft_overlap":
                            select.append("0 AS historical_soft_overlap")
                        else:
                            select.append(column)
                    layer = (
                        "CASE WHEN base_present=1 AND " + rank + "<=100 THEN 'baseline_top100' "
                        "WHEN base_present=1 THEN 'item2vec_only' "
                        "WHEN historical_present=1 AND soft_present=1 THEN 'historical_soft_overlap' "
                        "WHEN historical_present=1 THEN 'historical_only' ELSE 'soft_only' END"
                    )
                    con.execute(
                        f"COPY (SELECT {','.join(select)},{rank} AS candidate_rank,{layer} AS m38_source_layer "
                        f"FROM read_parquet({_literal(source)}) WHERE {predicate} ORDER BY customer_id,{rank},article_id) "
                        f"TO {_literal(output)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
                    )
                budget = BUDGETS[variant]
                audit = con.execute(
                    f"WITH g AS (SELECT customer_id,count(*) AS group_rows,min(candidate_rank) AS lo,max(candidate_rank) AS hi,count(DISTINCT candidate_rank) AS dr "
                    f"FROM read_parquet({_literal(output)}) GROUP BY customer_id) SELECT min(group_rows),max(group_rows),"
                    f"count(*) FILTER(WHERE lo<>1 OR hi<>group_rows OR dr<>group_rows OR group_rows>{budget}) FROM g"
                ).fetchone()
                if int(audit[2]):
                    raise RuntimeError(f"M3.8 variant group audit failed: {cutoff} {variant}")
            finally:
                con.close()
            results[cutoff][variant] = _file_identity(output)
    return results


def _sample_relation(path: Path, seed: int) -> str:
    source = f"read_parquet({_literal(path)})"
    return f"""
    WITH positive_groups AS (
      SELECT customer_id,sum(target)::BIGINT positives FROM {source} GROUP BY customer_id HAVING positives>0
    ), positives AS (
      SELECT f.*,'positive'::VARCHAR AS _m38_layer,0::INTEGER AS _m38_bucket,
             1::BIGINT AS _m38_source_rows,1::BIGINT AS _m38_selected_rows
      FROM {source} f JOIN positive_groups g USING(customer_id) WHERE target=1
    ), negative_bucketed AS (
      SELECT f.*,g.positives,ntile(3) OVER(PARTITION BY f.customer_id,f.m38_source_layer
        ORDER BY candidate_rank,article_id)::INTEGER AS _m38_bucket
      FROM {source} f JOIN positive_groups g USING(customer_id) WHERE target=0
    ), strata AS (
      SELECT *,m38_source_layer AS _m38_layer,
        count(*) OVER(PARTITION BY customer_id,m38_source_layer,_m38_bucket)::BIGINT AS _m38_source_rows,
        row_number() OVER(PARTITION BY customer_id,m38_source_layer,_m38_bucket
          ORDER BY hash(customer_id,article_id,{seed}),candidate_rank,article_id) AS cell_rank
      FROM negative_bucketed
    ), ordered AS (
      SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY cell_rank,_m38_layer,_m38_bucket,
        hash(customer_id,article_id,{seed}),candidate_rank,article_id) AS sample_rank FROM strata
    ), selected AS (
      SELECT * FROM ordered WHERE sample_rank<={NEGATIVE_RATIO}*positives
    ), selected_counted AS (
      SELECT *,count(*) OVER(PARTITION BY customer_id,_m38_layer,_m38_bucket)::BIGINT AS _m38_selected_rows
      FROM selected
    ), negatives AS (
      SELECT * EXCLUDE(positives,cell_rank,sample_rank) FROM selected_counted
    )
    SELECT * FROM positives UNION ALL BY NAME SELECT * FROM negatives
    """


def load_sample(paths: list[Path], features: list[str], *, seed: int = SAMPLING_SEED) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    sizes: list[int] = []
    strata: dict[tuple[str, int], list[int]] = {}
    source_rows = 0
    for path in paths:
        con = duckdb.connect()
        try:
            relation = _sample_relation(path, seed)
            helper = ["_m38_layer", "_m38_bucket", "_m38_source_rows", "_m38_selected_rows"]
            frame = con.execute(
                f"SELECT {','.join(features + ['target'] + helper)} FROM ({relation}) ORDER BY customer_id,candidate_rank,article_id"
            ).fetchdf()
            groups = con.execute(f"SELECT count(*),sum(target) FROM ({relation}) GROUP BY customer_id ORDER BY customer_id").fetchall()
            audit = con.execute(
                f"SELECT _m38_layer,_m38_bucket,max(_m38_source_rows),count(*) FROM ({relation}) WHERE target=0 GROUP BY 1,2"
            ).fetchall()
            source_rows += int(con.execute(f"SELECT count(*) FROM read_parquet({_literal(path)})").fetchone()[0])
        finally:
            con.close()
        if any(int(row[1]) <= 0 for row in groups):
            raise RuntimeError("M3.8 sampling retained zero-positive group")
        frames.append(frame)
        sizes.extend(int(row[0]) for row in groups)
        for layer, bucket, source_count, selected_count in audit:
            entry = strata.setdefault((str(layer), int(bucket)), [0, 0])
            entry[0] += int(source_count)
            entry[1] += int(selected_count)
    result = pd.concat(frames, ignore_index=True)
    if sum(sizes) != len(result) or not sizes or min(sizes) < 2 or max(sizes) > 500:
        raise RuntimeError("M3.8 sampled group-size audit failed")
    positives = int(result["target"].sum())
    negatives = len(result) - positives
    if negatives > NEGATIVE_RATIO * positives:
        raise RuntimeError("M3.8 sampling exceeded 30:1")
    return result, sizes, {
        "mode": "source_layer_rank_tertile_round_robin_30x", "seed": seed,
        "source_rows": source_rows, "sampled_rows": len(result), "groups": len(sizes),
        "positives": positives, "negatives": negatives, "unobserved_per_positive": negatives / max(positives, 1),
        "strata": [
            {"source_layer": key[0], "rank_bucket": key[1], "source_rows": value[0], "selected_rows": value[1]}
            for key, value in sorted(strata.items())
        ],
    }


def load_inner_validation(*, path: Path, transactions_path: Path, cutoff: str, features: list[str], budget: int) -> tuple[pd.DataFrame, list[int], np.ndarray, dict[str, Any]]:
    columns = list(dict.fromkeys(["customer_id", "article_id", *features, "target", "user_history_events_12w"]))
    con = duckdb.connect()
    try:
        frame = con.execute(
            f"""
            WITH eligible AS (SELECT customer_id FROM read_parquet({_literal(path)}) GROUP BY customer_id
              HAVING sum(target)>0 AND max(user_history_events_12w)>0),
            truth AS (SELECT customer_id,count(DISTINCT article_id)::BIGINT truth_count
              FROM read_parquet({_literal(transactions_path)}) WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY GROUP BY customer_id)
            SELECT {','.join('s.' + column for column in columns)},t.truth_count
            FROM read_parquet({_literal(path)}) s JOIN eligible e USING(customer_id) JOIN truth t USING(customer_id)
            ORDER BY s.customer_id,s.candidate_rank,s.article_id
            """
        ).fetchdf()
    finally:
        con.close()
    groups = frame.groupby("customer_id", sort=False).agg(rows=("target", "size"), positives=("target", "sum"), truth=("truth_count", "first"))
    sizes = groups["rows"].astype(int).tolist()
    if not sizes or min(sizes) < 100 or max(sizes) > budget or (groups["positives"] <= 0).any():
        raise RuntimeError("M3.8 inner-validation group audit failed")
    return frame, sizes, groups["truth"].to_numpy(dtype=np.int64), {
        "cutoff": cutoff, "rows": len(frame), "groups": len(sizes), "min_group_rows": min(sizes),
        "max_group_rows": max(sizes), "role": "active candidate-covered round selection only",
    }


def evaluate_variant(*, evaluation_db: Path, dataset_path: Path, transactions_path: Path, cutoff: str, variant: str, budget: int) -> dict[str, Any]:
    con = duckdb.connect(str(evaluation_db))
    try:
        con.execute(
            f"""
            CREATE TABLE m38_truth AS
            WITH users AS (SELECT DISTINCT customer_id FROM predictions),warm AS (
              SELECT DISTINCT article_id FROM read_parquet({_literal(transactions_path)}) WHERE t_dat<DATE '{cutoff}')
            SELECT q.customer_id,q.article_id,CASE WHEN w.article_id IS NULL THEN 'cold' ELSE 'warm' END item_temperature
            FROM (SELECT DISTINCT t.customer_id,t.article_id FROM read_parquet({_literal(transactions_path)}) t
              JOIN users u USING(customer_id) WHERE t.t_dat>=DATE '{cutoff}' AND t.t_dat<DATE '{cutoff}'+INTERVAL 7 DAY) q
            LEFT JOIN warm w USING(article_id);
            CREATE TABLE m38_ranked AS SELECT p.customer_id,p.article_id,
              row_number() OVER(PARTITION BY p.customer_id ORDER BY
                {inactive_fallback_order_expression(f'score_{variant}')},p.candidate_rank,p.article_id) AS candidate_rank
            FROM predictions p;
            CREATE VIEW m38_features AS SELECT customer_id,article_id,base_present,historical_present,soft_present
              FROM read_parquet({_literal(dataset_path)});
            """
        )
        segments = {
            label: _candidate_metrics(con, "m38_ranked", budget, predicate, truth_table="m38_truth")
            for label, predicate in {"overall": "TRUE", "warm": "item_temperature='warm'", "cold": "item_temperature='cold'"}.items()
        }
        conversion = con.execute(
            """
            SELECT count(*) FILTER(WHERE f.base_present=0 AND t.article_id IS NOT NULL),
                   count(*) FILTER(WHERE f.base_present=0 AND t.item_temperature='cold'),
                   count(DISTINCT r.customer_id) FILTER(WHERE f.base_present=0 AND t.article_id IS NOT NULL)
            FROM m38_ranked r JOIN m38_features f USING(customer_id,article_id)
            LEFT JOIN m38_truth t USING(customer_id,article_id) WHERE r.candidate_rank<=12
            """
        ).fetchone()
        return {
            "segments": segments,
            "new_image_truth_in_top12_pairs": int(conversion[0]),
            "new_image_cold_truth_in_top12_pairs": int(conversion[1]),
            "users_with_new_image_truth_in_top12": int(conversion[2]),
        }
    finally:
        con.close()


def _summary(development: dict[str, Any], m33: dict[str, Any]) -> dict[str, Any]:
    anchor_values = {
        window: float(m33["development"][window]["evaluation"]["orderings"]["anchor__inactive_rrf"]["segments"]["overall"]["map@12"])
        for window in ROLLING_PROTOCOL
    }
    anchor_mean = float(np.mean(list(anchor_values.values())))
    rows: dict[str, Any] = {
        "base_300": {"window_map@12": anchor_values, "mean_map@12": anchor_mean, "accepted": False}
    }
    for variant in VARIANTS:
        values = {window: float(development[window][variant]["evaluation"]["segments"]["overall"]["map@12"]) for window in ROLLING_PROTOCOL}
        deltas = {window: values[window] - anchor_values[window] for window in ROLLING_PROTOCOL}
        mean = float(np.mean(list(values.values())))
        rows[variant] = {
            "window_map@12": values, "mean_map@12": mean,
            "window_delta_vs_base": deltas, "mean_delta_vs_base": mean - anchor_mean,
            "accepted": bool(mean > anchor_mean and all(delta >= 0 for delta in deltas.values())),
        }
    accepted = [name for name in VARIANTS if rows[name]["accepted"]]
    selected = None
    if accepted:
        selected = max(accepted, key=lambda name: (rows[name]["mean_map@12"], -BUDGETS[name]))
        best = rows[selected]["mean_map@12"]
        simpler = [name for name in accepted if best - rows[name]["mean_map@12"] <= 1e-5]
        selected = min(simpler, key=lambda name: (BUDGETS[name], name))
    return {
        "orderings": rows, "accepted_variants": accepted, "selected_variant": selected,
        "anchor_binding": "reused measured M3.3 anchor; no numerical recomputation drift",
        "final_week": "not_run",
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    windows = list(ROLLING_PROTOCOL)
    lines = [
        "# M3.8：历史图片源与季节软图片源的 2×2 排序兑现", "", "## 结论", "",
        f"- 通过晋级门槛的方案：{', '.join(summary['accepted_variants']) if summary['accepted_variants'] else '无'}。",
        f"- 冻结选择：{summary['selected_variant'] or '无；继续保留 M3.3 base'}。",
        "- 最终验证周未运行；四窗此前已参与召回诊断，本结果属于开发证据。", "",
        "## 术语", "",
        "- `base-300`（基础候选池）：六路协同/规则召回 Top100，加至多200个 Item2Vec 独有候选。",
        "- `historical`（历史图片源，本项目自定义名）：M3.5 的视觉 Top100、商品类型季节硬门槛、个性化与全局合并后统一 Top100；不是泛指历史交易。",
        "- `soft-shallow`（季节软图片源，本项目自定义名）：视觉近邻 Top100，季节信号仅以0.5权重参加 RRF 排名，个性化 Top50 加全局 Top50。",
        "- `source-aware`（来源感知）：排序器能看到候选来自哪条图片源、源内名次及视觉/季节分数。",
        "- `inactive fallback`（无近期行为回退）：用户截止前12周无交易时不使用模型分数，保持原候选顺序。", "",
        "## Overall MAP@12", "",
        "| candidate/ranker | " + " | ".join(windows) + " | mean | delta vs base | accepted |",
        "|---|" + "---:|" * (len(windows) + 3),
    ]
    for name, row in summary["orderings"].items():
        delta = row.get("mean_delta_vs_base", 0.0)
        lines.append("| " + name + " | " + " | ".join(f"{row['window_map@12'][w]:.6f}" for w in windows) + f" | {row['mean_map@12']:.6f} | {delta:+.6f} | {row['accepted']} |")
    lines.extend(["", "## Warm/cold 与图片 truth 兑现", ""])
    for window in windows:
        lines.append(f"### {window}")
        lines.append("")
        lines.append("| variant | warm MAP@12 | cold MAP@12 | candidate Recall | Oracle MAP@12 | new image truth Top12 | cold subset |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for variant in VARIANTS:
            ev = result["development"][window][variant]["evaluation"]
            lines.append(f"| {variant} | {ev['segments']['warm']['map@12']:.6f} | {ev['segments']['cold']['map@12']:.6f} | {ev['segments']['overall'][f'candidate_recall@{BUDGETS[variant]}']:.6f} | {ev['segments']['overall']['oracle_map@12']:.6f} | {ev['new_image_truth_in_top12_pairs']} | {ev['new_image_cold_truth_in_top12_pairs']} |")
        lines.append("")
    lines.extend([
        "## 审计边界", "",
        "- 所有10个训练/验证 cutoff 都单独重建 cutoff-safe historical 与 soft-shallow 图片源；验证周 truth 不参与召回或特征。",
        "- A 直接绑定已验收 M3.3 anchor；B/C/D 各自独立候选文件、分层负采样、inner 选轮数、outer 重训与评分。",
        "- 30:1 表示每个正例最多抽30个未观测候选，不是正例:负例=30:1。",
        "- optimistic all-articles 仍无法观测真实库存、曝光和上架时间；cold 结果只在该乐观目录假设下成立。",
        f"- 总耗时：{result['elapsed_seconds']:.2f} 秒。", "",
        "机器可读证据：`metrics.json`。", "",
    ])
    return "\n".join(lines)


def run_m38(
    *, raw_dir: Path, work_dir: Path, transactions_path: Path, neighbor_metrics_path: Path,
    m29_cache_dir: Path, historical_cache_dir: Path, soft_cache_dir: Path,
    candidate_cache_dir: Path, feature_cache_dir: Path, variant_cache_dir: Path,
    m33_metrics_path: Path, output_dir: Path, artifact_dir: Path, config: M2Config, run_id: str,
) -> dict[str, Any]:
    validate_protocol()
    config.validate()
    if config.evaluation_role != "development" or config.candidate_k != 500:
        raise ValueError("M3.8 requires development candidate_k=500")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        m33 = _read_json(m33_metrics_path)
        if m33.get("schema_version") != "m3.3-cross-season-adaptive-seasonal-v1" or m33["contract"].get("final_week") != "not_run":
            raise ValueError("M3.8 requires measured M3.3 no-final-week evidence")
        print("M3.8: building/validating ten-cutoff image sources", flush=True)
        sources = build_source_cache(
            raw_dir=raw_dir, transactions_path=transactions_path, m29_cache_dir=m29_cache_dir,
            neighbor_metrics_path=neighbor_metrics_path, historical_cache_dir=historical_cache_dir,
            soft_cache_dir=soft_cache_dir,
        )
        print("M3.8: building/validating union candidates", flush=True)
        candidates = build_candidate_cache(m29_cache_dir=m29_cache_dir, sources=sources, cache_dir=candidate_cache_dir)
        print("M3.8: building/validating common point-in-time features", flush=True)
        features = build_feature_cache(
            raw_dir=raw_dir, work_dir=work_dir, transactions_path=transactions_path,
            candidate_cache_dir=candidate_cache_dir, candidate_manifests=candidates,
            cache_dir=feature_cache_dir, config=config,
        )
        variants_cache = materialize_variant_cache(feature_cache_dir=feature_cache_dir, variant_cache_dir=variant_cache_dir)
        anchor_features = list(frozen_feature_sets()[ANCHOR_NAME])
        expanded_features = list(dict.fromkeys(anchor_features + IMAGE_SOURCE_FEATURES))
        development: dict[str, Any] = {}
        for window, protocol in ROLLING_PROTOCOL.items():
            print(f"M3.8 {window}: inner round selection", flush=True)
            window_dir = artifact_dir / window
            window_dir.mkdir()
            development[window] = {}
            for variant in VARIANTS:
                variant_dir = window_dir / variant
                variant_dir.mkdir()
                train, train_sizes, train_evidence = load_sample(
                    [_variant_path(variant_cache_dir, cutoff, variant) for cutoff in protocol["inner_train"]],
                    expanded_features,
                )
                maps = build_category_maps(train)
                inner_maps = _save_category_maps(variant_dir / "inner-category-maps.json", maps)
                validation, validation_sizes, truth_counts, validation_evidence = load_inner_validation(
                    path=_variant_path(variant_cache_dir, protocol["inner_validation"], variant),
                    transactions_path=transactions_path, cutoff=protocol["inner_validation"],
                    features=expanded_features, budget=BUDGETS[variant],
                )
                _, inner_model = train_inner_ranker(
                    train_frame=train, train_group_sizes=train_sizes, train_evidence=train_evidence,
                    validation_frame=validation, validation_group_sizes=validation_sizes,
                    validation_truth_counts=truth_counts, validation_evidence=validation_evidence,
                    features=expanded_features, name=variant, artifact_dir=variant_dir,
                    config=config, category_maps=maps,
                )
                del train, validation
                gc.collect()
                print(f"M3.8 {window} {variant}: outer refit", flush=True)
                outer, outer_sizes, outer_evidence = load_sample(
                    [_variant_path(variant_cache_dir, cutoff, variant) for cutoff in protocol["outer_train"]],
                    expanded_features,
                )
                outer_maps_obj = build_category_maps(outer)
                outer_maps = _save_category_maps(variant_dir / "outer-category-maps.json", outer_maps_obj)
                rounds = int(inner_model["best_iteration"])
                model, outer_model = train_lambdarank(
                    frame=outer, group_sizes=outer_sizes, group_evidence=outer_evidence,
                    features=expanded_features, name=variant, use_ipw=False, artifact_dir=variant_dir,
                    config=replace(config, num_boost_round=rounds), category_maps=outer_maps_obj,
                )
                outer_model["selected_rounds_from_inner"] = rounds
                del outer
                gc.collect()
                valid_path = _variant_path(variant_cache_dir, protocol["outer_validation"], variant)
                scoring = score_models(
                    dataset_path=valid_path, models={variant: (model, expanded_features, outer_maps_obj)},
                    evaluation_db=variant_dir / "evaluation.duckdb", prediction_path=variant_dir / "predictions.parquet",
                    config=config,
                )
                del model
                gc.collect()
                evaluation = evaluate_variant(
                    evaluation_db=variant_dir / "evaluation.duckdb", dataset_path=valid_path,
                    transactions_path=transactions_path, cutoff=protocol["outer_validation"],
                    variant=variant, budget=BUDGETS[variant],
                )
                development[window][variant] = {
                    "inner_model": inner_model, "outer_model": outer_model,
                    "inner_category_encoding": inner_maps, "outer_category_encoding": outer_maps,
                    "scoring": scoring, "evaluation": evaluation,
                }
        summary = _summary(development, m33)
        result = {
            "schema_version": SCHEMA_VERSION, "stage": "M3.8", "status": "measured",
            "run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "four rolling development windows; ten cutoff-safe sources; final week not run",
            "config": asdict(config),
            "contract": {
                "factorial_candidates": BUDGETS, "image_source_features": IMAGE_SOURCE_FEATURES,
                "ranking_features": expanded_features,
                "negative_sampling": "30 unobserved candidates per positive, source-layer x rank-tertile round robin",
                "selection": "mean MAP improves and no outer window regresses; 1e-5 simplicity tie",
                "catalog": "optimistic_all_articles", "final_week": "not_run",
            },
            "inputs": {"m33": _file_identity(m33_metrics_path), "neighbors": _file_identity(neighbor_metrics_path), "transactions": _file_identity(transactions_path)},
            "source_cache": sources, "candidate_cache": candidates, "feature_cache": features,
            "variant_cache": variants_cache, "development": development, "summary": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics = output_dir / "metrics.json"
        report = output_dir / "M3_8_REPORT.md"
        result["artifacts"] = {"metrics": str(metrics.resolve()), "report": str(report.resolve()), "artifact_dir": str(artifact_dir.resolve())}
        _write_json(metrics, result)
        report.write_text(render_report(result), encoding="utf-8", newline="\n")
        return result
    except Exception as error:
        _write_json(artifact_dir / f"failure-{time.time_ns()}.json", {
            "schema_version": "m3.8-failure-v1", "status": "failed",
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
        })
        raise
