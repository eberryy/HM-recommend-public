from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m2 import _literal, _write_json
from .m25_retrieval import _build_image_candidates, _candidate_metrics, _load_neighbors
from .m28 import _file_identity
from .m34 import OUTER_WINDOWS, _group_audit, _validate_base


SCHEMA_VERSION = "m3.5-content-season-image-cold-retrieval-v1"
SOURCE_SCHEMA_VERSION = "m3.5-cold-source-cache-v1"

CONFIG = {
    "history_weeks": 12,
    "current_short_days": 28,
    "current_background_days": 84,
    "prior_same_period_days": 28,
    "prior_background_days": 84,
    "seasonal_product_type_lift": 1.15,
    "minimum_prior_type_events_28d": 50,
    "attribute_cold_items_per_type": 100,
    "user_top_product_types": 5,
    "attribute_source_k": 100,
    "image_seed_k": 5,
    "image_neighbors_per_seed": 100,
    "raw_image_candidate_k": 300,
    "global_visual_seed_k": 50,
    "global_visual_candidate_k": 100,
    "image_source_k": 100,
    "rrf_constant": 60,
    "append_per_source": 50,
}

VARIANT_BUDGETS = {
    "base_300": 300,
    "attribute_cold_100": 100,
    "image_cold_100": 100,
    "expanded_attribute_350": 350,
    "expanded_image_350": 350,
    "vacancy_fill_combined_300": 300,
    "expanded_combined_400": 400,
}


def validate_protocol() -> None:
    if any(cutoff >= "2020-09-16" for cutoff in OUTER_WINDOWS.values()):
        raise RuntimeError("M3.5 must not read final-week interactions")
    if CONFIG["append_per_source"] != 50 or CONFIG["image_seed_k"] != 5:
        raise RuntimeError("M3.5 bounded source budget drift")
    if list(VARIANT_BUDGETS) != [
        "base_300",
        "attribute_cold_100",
        "image_cold_100",
        "expanded_attribute_350",
        "expanded_image_350",
        "vacancy_fill_combined_300",
        "expanded_combined_400",
    ]:
        raise RuntimeError("M3.5 candidate variants drifted")


def _date(value: str) -> str:
    return "DATE '" + value.replace("'", "''") + "'"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_paths(cache_dir: Path, cutoff: str) -> tuple[Path, Path, Path]:
    root = cache_dir / cutoff
    return root / "attribute-cold.parquet", root / "image-cold.parquet", root / "manifest.json"


def _global_visual_candidates(
    seeds: pd.DataFrame,
    indices: np.ndarray,
    scores: np.ndarray,
    article_ids: np.ndarray,
) -> pd.DataFrame:
    candidates: dict[int, list[float | int]] = {}
    for seed in seeds.itertuples(index=False):
        seed_row = int(seed.row_index)
        seed_rank = int(seed.seed_rank)
        seed_weight = float(np.log1p(seed.item_events)) / (1.0 + 0.05 * (seed_rank - 1))
        for neighbor_rank, (neighbor_row, cosine) in enumerate(
            zip(indices[seed_row], scores[seed_row]), start=1
        ):
            row = int(neighbor_row)
            weighted = float(cosine) * seed_weight
            prior = candidates.get(row)
            if prior is None or (weighted, -seed_rank, -neighbor_rank) > (
                float(prior[0]), -int(prior[1]), -int(prior[2])
            ):
                candidates[row] = [weighted, seed_rank, neighbor_rank]
    ordered = sorted(candidates.items(), key=lambda value: (-float(value[1][0]), value[0]))
    return pd.DataFrame(
        [
            {
                "article_id": str(article_ids[row]),
                "global_visual_raw_rank": rank,
                "global_visual_score": float(values[0]),
                "global_visual_seed_rank": int(values[1]),
                "global_visual_neighbor_rank": int(values[2]),
            }
            for rank, (row, values) in enumerate(ordered, start=1)
        ]
    )


def _cache_valid(
    attribute_path: Path,
    image_path: Path,
    manifest_path: Path,
    *,
    cutoff: str,
    inputs: dict[str, Any],
) -> dict[str, Any] | None:
    states = [attribute_path.exists(), image_path.exists(), manifest_path.exists()]
    if not any(states):
        return None
    if not all(states):
        raise FileExistsError(f"incomplete M3.5 source cache: {attribute_path.parent}")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != SOURCE_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("cutoff") != cutoff
        or manifest.get("contract") != CONFIG
    ):
        raise ValueError(f"M3.5 source manifest drift: {manifest_path}")
    for name, expected in inputs.items():
        actual = manifest["inputs"][name]
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError(f"M3.5 source input drift: {name} {cutoff}")
    for name, path in (("attribute", attribute_path), ("image", image_path)):
        if _file_identity(path) != manifest["artifacts"][name]:
            raise ValueError(f"M3.5 source artifact drift: {path}")
    return manifest


def build_cold_sources(
    *,
    raw_dir: Path,
    transactions_path: Path,
    base_candidates: dict[str, tuple[Path, Path]],
    neighbor_metrics_path: Path,
    cache_dir: Path,
    windows: dict[str, str] | None = None,
) -> dict[str, Any]:
    selected = windows or OUTER_WINDOWS
    if set(base_candidates) != set(selected.values()):
        raise ValueError("M3.5 source inputs must exactly match selected cutoffs")
    _, indices, scores, items = _load_neighbors(neighbor_metrics_path)
    article_ids = items.sort_values("row_index")["article_id"].to_numpy()
    transaction_identity = _file_identity(transactions_path)
    article_path = raw_dir / "articles.csv"
    article_identity = _file_identity(article_path)
    neighbor_identity = _file_identity(neighbor_metrics_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for window, cutoff in selected.items():
        started = time.perf_counter()
        base_path, base_manifest_path = base_candidates[cutoff]
        base_identity = _validate_base(base_path, base_manifest_path, cutoff)["artifact"]
        inputs = {
            "base": base_identity,
            "transactions": transaction_identity,
            "articles": article_identity,
            "neighbor_metrics": neighbor_identity,
        }
        attribute_path, image_path, manifest_path = _source_paths(cache_dir, cutoff)
        existing = _cache_valid(
            attribute_path, image_path, manifest_path, cutoff=cutoff, inputs=inputs
        )
        if existing is not None:
            results[cutoff] = existing
            continue
        root = attribute_path.parent
        if root.exists():
            raise FileExistsError(f"incomplete M3.5 source root: {root}")
        root.mkdir(parents=True)
        connection = duckdb.connect()
        cutoff_sql = _date(cutoff)
        try:
            connection.execute("SET threads=8")
            connection.execute("SET memory_limit='11GB'")
            connection.register("image_items_frame", items)
            connection.execute(
                f"""
                CREATE TABLE base AS SELECT * FROM read_parquet({_literal(base_path)});
                CREATE TABLE eval_users AS SELECT DISTINCT customer_id FROM base;
                CREATE TABLE image_items AS SELECT row_index,article_id FROM image_items_frame;
                CREATE TABLE article_dim AS
                SELECT article_id,try_cast(product_type_no AS INTEGER) AS product_type_no,
                       try_cast(graphical_appearance_no AS INTEGER) AS graphical_appearance_no,
                       try_cast(colour_group_code AS INTEGER) AS colour_group_code,
                       try_cast(perceived_colour_value_id AS INTEGER) AS perceived_colour_value_id
                FROM read_csv_auto({_literal(article_path)},header=true,all_varchar=true);
                CREATE TABLE warm_catalog AS
                SELECT DISTINCT article_id FROM read_parquet({_literal(transactions_path)})
                WHERE t_dat<{cutoff_sql};
                CREATE TABLE cold_catalog AS
                SELECT a.* FROM article_dim a LEFT JOIN warm_catalog w USING(article_id)
                WHERE w.article_id IS NULL;
                CREATE TABLE current_events AS
                SELECT t.t_dat,t.article_id,a.product_type_no,a.graphical_appearance_no,
                       a.colour_group_code,a.perceived_colour_value_id
                FROM read_parquet({_literal(transactions_path)}) t JOIN article_dim a USING(article_id)
                WHERE t.t_dat>={cutoff_sql}-INTERVAL 84 DAY AND t.t_dat<{cutoff_sql};
                CREATE TABLE prior_events AS
                SELECT t.t_dat,t.article_id,a.product_type_no
                FROM read_parquet({_literal(transactions_path)}) t JOIN article_dim a USING(article_id)
                WHERE t.t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 84 DAY
                  AND t.t_dat<{cutoff_sql}-INTERVAL 1 YEAR;
                """
            )
            connection.unregister("image_items_frame")
            current84, current28 = connection.execute(
                f"SELECT count(*),count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 28 DAY) FROM current_events"
            ).fetchone()
            prior84, prior28 = connection.execute(
                f"SELECT count(*),count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY) FROM prior_events"
            ).fetchone()
            if min(map(int, (current84, current28, prior84, prior28))) <= 0:
                raise RuntimeError(f"M3.5 empty temporal statistics: {cutoff}")
            connection.execute(
                f"""
                CREATE TABLE prior_type AS
                WITH s AS (
                  SELECT product_type_no,
                    count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY) AS events28,
                    count(*) AS events84,
                    count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY)::DOUBLE/{int(prior28)}.0 AS share28,
                    count(*)::DOUBLE/{int(prior84)}.0 AS share84
                  FROM prior_events GROUP BY product_type_no)
                SELECT *,share28/nullif(share84,0) AS lift FROM s;
                """
            )
            for key in (
                "product_type_no",
                "graphical_appearance_no",
                "colour_group_code",
                "perceived_colour_value_id",
            ):
                connection.execute(
                    f"""
                    CREATE TABLE current_{key} AS
                    WITH s AS (
                      SELECT {key},
                        count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 28 DAY) AS events28,
                        count(*) AS events84,
                        count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 28 DAY)::DOUBLE/{int(current28)}.0 AS share28,
                        count(*)::DOUBLE/{int(current84)}.0 AS share84
                      FROM current_events GROUP BY {key})
                    SELECT *,share28/nullif(share84,0) AS lift FROM s;
                    """
                )
            score = " + ".join(
                [
                    "greatest(ln(greatest(p.lift,1.0)),0.0)*p.events28/(p.events28+50.0)",
                    "greatest(ln(greatest(t.lift,1.0)),0.0)*t.events28/(t.events28+50.0)",
                    "0.5*greatest(ln(greatest(g.lift,1.0)),0.0)*g.events28/(g.events28+30.0)",
                    "0.5*greatest(ln(greatest(c.lift,1.0)),0.0)*c.events28/(c.events28+30.0)",
                    "0.25*greatest(ln(greatest(v.lift,1.0)),0.0)*v.events28/(v.events28+30.0)",
                ]
            )
            connection.execute(
                f"""
                CREATE TABLE cold_scored AS
                SELECT a.*,({score}) AS attribute_season_score,p.lift AS prior_type_lift,
                       p.events28 AS prior_type_events28,t.lift AS current_type_lift,
                       g.lift AS current_graphical_lift,c.lift AS current_colour_lift,
                       v.lift AS current_perceived_colour_lift
                FROM cold_catalog a
                JOIN prior_type p USING(product_type_no)
                LEFT JOIN current_product_type_no t USING(product_type_no)
                LEFT JOIN current_graphical_appearance_no g USING(graphical_appearance_no)
                LEFT JOIN current_colour_group_code c USING(colour_group_code)
                LEFT JOIN current_perceived_colour_value_id v USING(perceived_colour_value_id)
                WHERE p.lift>={CONFIG['seasonal_product_type_lift']}
                  AND p.events28>={CONFIG['minimum_prior_type_events_28d']};
                CREATE TABLE cold_type_top AS
                SELECT *,row_number() OVER(PARTITION BY product_type_no
                    ORDER BY attribute_season_score DESC,article_id) AS type_rank
                FROM cold_scored QUALIFY type_rank<={CONFIG['attribute_cold_items_per_type']};
                CREATE TABLE user_type AS
                WITH counts AS (
                  SELECT x.customer_id,a.product_type_no,count(*) AS type_events
                  FROM read_parquet({_literal(transactions_path)}) x JOIN eval_users u USING(customer_id)
                  JOIN article_dim a USING(article_id)
                  WHERE x.t_dat>={cutoff_sql}-INTERVAL {CONFIG['history_weeks']} WEEK AND x.t_dat<{cutoff_sql}
                  GROUP BY x.customer_id,a.product_type_no)
                SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY type_events DESC,product_type_no) AS type_rank
                FROM counts QUALIFY type_rank<={CONFIG['user_top_product_types']};
                CREATE TABLE user_colour AS
                SELECT x.customer_id,a.colour_group_code,count(*) AS colour_events
                FROM read_parquet({_literal(transactions_path)}) x JOIN eval_users u USING(customer_id)
                JOIN article_dim a USING(article_id)
                WHERE x.t_dat>={cutoff_sql}-INTERVAL {CONFIG['history_weeks']} WEEK AND x.t_dat<{cutoff_sql}
                GROUP BY x.customer_id,a.colour_group_code;
                CREATE TABLE attribute_personalized AS
                WITH s AS (
                  SELECT u.customer_id,i.article_id,
                         i.attribute_season_score+0.25*ln(1+u.type_events)
                           +0.1*ln(1+coalesce(c.colour_events,0)) AS score
                  FROM user_type u JOIN cold_type_top i USING(product_type_no)
                  LEFT JOIN user_colour c ON u.customer_id=c.customer_id
                    AND i.colour_group_code=c.colour_group_code)
                SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY score DESC,article_id) AS personalized_rank
                FROM s QUALIFY personalized_rank<={CONFIG['attribute_source_k']};
                CREATE TABLE attribute_global AS
                SELECT article_id,row_number() OVER(ORDER BY attribute_season_score DESC,article_id) AS global_rank
                FROM cold_scored QUALIFY global_rank<=50;
                CREATE TABLE attribute_source AS
                WITH lanes AS (
                  SELECT customer_id,article_id,personalized_rank,NULL::BIGINT AS global_rank
                  FROM attribute_personalized
                  UNION ALL
                  SELECT u.customer_id,g.article_id,NULL,g.global_rank FROM eval_users u CROSS JOIN attribute_global g
                ), merged AS (
                  SELECT customer_id,article_id,min(personalized_rank) AS personalized_rank,
                         min(global_rank) AS global_rank FROM lanes GROUP BY customer_id,article_id), scored AS (
                  SELECT *,coalesce(1.0/(60+personalized_rank),0)+coalesce(1.0/(60+global_rank),0) AS score
                  FROM merged)
                SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY score DESC,
                    coalesce(personalized_rank,1000000),coalesce(global_rank,1000000),article_id) AS attribute_rank
                FROM scored QUALIFY attribute_rank<={CONFIG['attribute_source_k']};
                CREATE TABLE seeds AS
                WITH latest AS (
                  SELECT x.customer_id,x.article_id,max(x.t_dat) AS latest_t_dat
                  FROM read_parquet({_literal(transactions_path)}) x JOIN eval_users u USING(customer_id)
                  JOIN image_items i USING(article_id)
                  WHERE x.t_dat>={cutoff_sql}-INTERVAL {CONFIG['history_weeks']} WEEK AND x.t_dat<{cutoff_sql}
                  GROUP BY x.customer_id,x.article_id), ranked AS (
                  SELECT l.*,i.row_index,row_number() OVER(PARTITION BY customer_id
                    ORDER BY latest_t_dat DESC,article_id) AS seed_rank FROM latest l JOIN image_items i USING(article_id))
                SELECT * FROM ranked WHERE seed_rank<={CONFIG['image_seed_k']};
                """
            )
            seed_frame = connection.execute(
                "SELECT customer_id,row_index,seed_rank FROM seeds ORDER BY customer_id,seed_rank"
            ).fetchdf()
            _build_image_candidates(
                connection,
                seed_frame,
                indices,
                scores,
                article_ids,
                CONFIG["raw_image_candidate_k"],
                user_chunk=1000,
            )
            global_seeds = connection.execute(
                f"""
                WITH s AS (
                  SELECT e.article_id,count(*) AS item_events
                  FROM current_events e
                  JOIN (SELECT DISTINCT product_type_no FROM cold_scored) c USING(product_type_no)
                  JOIN image_items i ON e.article_id=i.article_id
                  WHERE e.t_dat>={cutoff_sql}-INTERVAL 28 DAY
                  GROUP BY e.article_id), ranked AS (
                  SELECT s.*,i.row_index,row_number() OVER(ORDER BY item_events DESC,s.article_id) AS seed_rank
                  FROM s JOIN image_items i USING(article_id))
                SELECT row_index,item_events,seed_rank FROM ranked
                WHERE seed_rank<={CONFIG['global_visual_seed_k']} ORDER BY seed_rank
                """
            ).fetchdf()
            global_visual = _global_visual_candidates(global_seeds, indices, scores, article_ids)
            connection.register("global_visual_frame", global_visual)
            connection.execute(
                f"""
                CREATE TABLE global_visual AS SELECT * FROM global_visual_frame;
                CREATE TABLE image_personalized AS
                WITH s AS (
                  SELECT i.*,c.attribute_season_score,
                    row_number() OVER(PARTITION BY i.customer_id
                      ORDER BY c.attribute_season_score DESC,i.image_rank,i.article_id) AS attribute_rank
                  FROM image_candidates i JOIN cold_scored c USING(article_id)), scored AS (
                  SELECT *,1.0/(60+image_rank)+1.0/(60+attribute_rank) AS combined_score FROM s)
                SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY combined_score DESC,
                    image_rank,attribute_rank,article_id) AS personalized_image_rank
                FROM scored QUALIFY personalized_image_rank<={CONFIG['image_source_k']};
                CREATE TABLE global_visual_cold AS
                SELECT g.*,row_number() OVER(ORDER BY g.global_visual_raw_rank,g.article_id) AS global_visual_rank
                FROM global_visual g JOIN cold_scored c USING(article_id)
                QUALIFY global_visual_rank<={CONFIG['global_visual_candidate_k']};
                CREATE TABLE image_source AS
                WITH lanes AS (
                  SELECT customer_id,article_id,personalized_image_rank,NULL::BIGINT AS global_visual_rank
                  FROM image_personalized
                  UNION ALL
                  SELECT u.customer_id,g.article_id,NULL,g.global_visual_rank
                  FROM eval_users u CROSS JOIN global_visual_cold g), merged AS (
                  SELECT customer_id,article_id,min(personalized_image_rank) AS personalized_image_rank,
                    min(global_visual_rank) AS global_visual_rank FROM lanes GROUP BY customer_id,article_id), scored AS (
                  SELECT *,coalesce(1.0/(60+personalized_image_rank),0)+coalesce(1.0/(60+global_visual_rank),0) AS score
                  FROM merged)
                SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY score DESC,
                  coalesce(personalized_image_rank,1000000),coalesce(global_visual_rank,1000000),article_id) AS image_cold_rank
                FROM scored QUALIFY image_cold_rank<={CONFIG['image_source_k']};
                """
            )
            connection.unregister("global_visual_frame")
            connection.execute(
                "CREATE TEMP VIEW attribute_audit AS SELECT customer_id,article_id,attribute_rank AS candidate_rank FROM attribute_source"
            )
            connection.execute(
                "CREATE TEMP VIEW image_audit AS SELECT customer_id,article_id,image_cold_rank AS candidate_rank FROM image_source"
            )
            attr_audit = _group_audit(connection, "attribute_audit", 100)
            image_audit = _group_audit(connection, "image_audit", 100)
            connection.execute(
                f"COPY (SELECT * FROM attribute_source ORDER BY customer_id,attribute_rank) TO {_literal(attribute_path)} "
                "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
            )
            connection.execute(
                f"COPY (SELECT * FROM image_source ORDER BY customer_id,image_cold_rank) TO {_literal(image_path)} "
                "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
            )
            temporal = connection.execute(
                "SELECT max(t_dat) FROM current_events"
            ).fetchone()[0]
            stats = connection.execute(
                """
                SELECT (SELECT count(*) FROM cold_catalog),(SELECT count(*) FROM cold_scored),
                       (SELECT count(*) FROM eval_users),(SELECT count(DISTINCT customer_id) FROM seeds),
                       (SELECT count(*) FROM global_visual_cold)
                """
            ).fetchone()
            if str(temporal) >= cutoff:
                raise RuntimeError(f"M3.5 temporal leakage: {cutoff}")
            manifest = {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "status": "completed",
                "window": window,
                "cutoff": cutoff,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract": CONFIG,
                "inputs": inputs,
                "audit": {
                    "attribute": attr_audit,
                    "image": image_audit,
                    "cold_catalog_items": int(stats[0]),
                    "seasonal_scored_cold_items": int(stats[1]),
                    "eval_users": int(stats[2]),
                    "users_with_image_seed": int(stats[3]),
                    "global_visual_cold_items": int(stats[4]),
                    "latest_history_date": str(temporal),
                },
                "artifacts": {
                    "attribute": _file_identity(attribute_path),
                    "image": _file_identity(image_path),
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(manifest_path, manifest)
            results[cutoff] = manifest
        finally:
            connection.close()
    return results


def evaluate_window(
    *,
    window: str,
    cutoff: str,
    transactions_path: Path,
    base_path: Path,
    source_manifest: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True)
    db_path = artifact_dir / "evaluation.duckdb"
    con: duckdb.DuckDBPyConnection | None = duckdb.connect(str(db_path))
    cutoff_sql = _date(cutoff)
    try:
        con.execute("SET threads=8")
        attr_path = Path(source_manifest["artifacts"]["attribute"]["path"])
        image_path = Path(source_manifest["artifacts"]["image"]["path"])
        con.execute(
            f"""
            CREATE TABLE base AS SELECT customer_id,article_id,candidate_rank FROM read_parquet({_literal(base_path)});
            CREATE TABLE attribute_source AS SELECT * FROM read_parquet({_literal(attr_path)});
            CREATE TABLE image_source AS SELECT * FROM read_parquet({_literal(image_path)});
            CREATE TABLE eval_users AS SELECT DISTINCT customer_id FROM base;
            CREATE TABLE warm_catalog AS SELECT DISTINCT article_id FROM read_parquet({_literal(transactions_path)})
              WHERE t_dat<{cutoff_sql};
            CREATE TABLE eval_truth AS
            SELECT q.customer_id,q.article_id,CASE WHEN w.article_id IS NULL THEN 'cold' ELSE 'warm' END AS item_temperature
            FROM (SELECT DISTINCT x.customer_id,x.article_id FROM read_parquet({_literal(transactions_path)}) x
              JOIN eval_users u USING(customer_id) WHERE x.t_dat>={cutoff_sql} AND x.t_dat<{cutoff_sql}+INTERVAL 7 DAY) q
            LEFT JOIN warm_catalog w USING(article_id);
            CREATE TABLE base_300 AS SELECT * FROM base;
            CREATE TABLE attribute_cold_100 AS SELECT customer_id,article_id,attribute_rank AS candidate_rank FROM attribute_source;
            CREATE TABLE image_cold_100 AS SELECT customer_id,article_id,image_cold_rank AS candidate_rank FROM image_source;
            CREATE TABLE base_sizes AS SELECT customer_id,count(*) AS n FROM base GROUP BY customer_id;
            CREATE TABLE attr_new AS SELECT a.*,row_number() OVER(PARTITION BY a.customer_id ORDER BY a.attribute_rank,a.article_id) AS new_rank
              FROM attribute_source a WHERE NOT EXISTS(SELECT 1 FROM base b WHERE b.customer_id=a.customer_id AND b.article_id=a.article_id);
            CREATE TABLE image_new AS SELECT i.*,row_number() OVER(PARTITION BY i.customer_id ORDER BY i.image_cold_rank,i.article_id) AS new_rank
              FROM image_source i WHERE NOT EXISTS(SELECT 1 FROM base b WHERE b.customer_id=i.customer_id AND b.article_id=i.article_id);
            CREATE TABLE expanded_attribute_350 AS SELECT * FROM base UNION ALL
              SELECT a.customer_id,a.article_id,(s.n+a.new_rank)::BIGINT FROM attr_new a JOIN base_sizes s USING(customer_id) WHERE a.new_rank<=50;
            CREATE TABLE expanded_image_350 AS SELECT * FROM base UNION ALL
              SELECT i.customer_id,i.article_id,(s.n+i.new_rank)::BIGINT FROM image_new i JOIN base_sizes s USING(customer_id) WHERE i.new_rank<=50;
            CREATE TABLE attr_selected AS
              SELECT customer_id,article_id,attribute_rank AS source_rank
              FROM attr_new WHERE new_rank<=50;
            CREATE TABLE image_selected AS
            WITH filtered AS (
              SELECT i.* FROM image_source i
              WHERE NOT EXISTS(SELECT 1 FROM base b WHERE b.customer_id=i.customer_id AND b.article_id=i.article_id)
                AND NOT EXISTS(SELECT 1 FROM attr_selected a WHERE a.customer_id=i.customer_id AND a.article_id=i.article_id))
            SELECT customer_id,article_id,image_cold_rank AS source_rank,
              row_number() OVER(PARTITION BY customer_id ORDER BY image_cold_rank,article_id) AS selected_rank
            FROM filtered QUALIFY selected_rank<=50;
            CREATE TABLE combined_new AS
            WITH lanes AS (
              SELECT customer_id,article_id,1 AS route,source_rank FROM attr_selected
              UNION ALL SELECT customer_id,article_id,2,source_rank FROM image_selected)
            SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY route,source_rank,article_id) AS new_rank
            FROM lanes;
            CREATE TABLE expanded_combined_400 AS SELECT * FROM base UNION ALL
              SELECT c.customer_id,c.article_id,(s.n+c.new_rank)::BIGINT FROM combined_new c JOIN base_sizes s USING(customer_id) WHERE c.new_rank<=100;
            CREATE TABLE vacancy_fill_combined_300 AS SELECT * FROM base UNION ALL
              SELECT c.customer_id,c.article_id,(s.n+c.new_rank)::BIGINT FROM combined_new c JOIN base_sizes s USING(customer_id)
              WHERE c.new_rank<=300-s.n;
            """
        )
        audits = {name: _group_audit(con, name, budget) for name, budget in VARIANT_BUDGETS.items()}
        evaluations = {}
        for name, budget in VARIANT_BUDGETS.items():
            evaluations[name] = {
                "overall": _candidate_metrics(con, name, budget, truth_table="eval_truth"),
                "warm": _candidate_metrics(con, name, budget, "item_temperature='warm'", truth_table="eval_truth"),
                "cold": _candidate_metrics(con, name, budget, "item_temperature='cold'", truth_table="eval_truth"),
            }
        marginal = con.execute(
            """
            SELECT
              (SELECT count(*) FROM attr_new a JOIN eval_truth t USING(customer_id,article_id) WHERE t.item_temperature='cold'),
              (SELECT count(*) FROM image_new i JOIN eval_truth t USING(customer_id,article_id) WHERE t.item_temperature='cold'),
              (SELECT count(*) FROM combined_new c JOIN eval_truth t USING(customer_id,article_id) WHERE t.item_temperature='cold')
            """
        ).fetchone()
        combined_route_rows = con.execute(
            "SELECT (SELECT count(*) FROM attr_selected),"
            " (SELECT count(*) FROM image_selected)"
        ).fetchone()
        artifacts = {}
        for name in VARIANT_BUDGETS:
            path = artifact_dir / f"{name}.parquet"
            con.execute(f"COPY (SELECT * FROM {name} ORDER BY customer_id,candidate_rank,article_id) TO {_literal(path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)")
            artifacts[name] = _file_identity(path)
        con.execute("CHECKPOINT")
        con.close()
        con = None
        return {
            "window": window,
            "cutoff": cutoff,
            "audits": audits,
            "evaluations": evaluations,
            "marginal_cold_truth": {
                "attribute": int(marginal[0]),
                "image": int(marginal[1]),
                "combined": int(marginal[2]),
            },
            "combined_route_rows": {
                "attribute": int(combined_route_rows[0]),
                "image": int(combined_route_rows[1]),
            },
            "artifacts": {"evaluation_db": _file_identity(db_path), "variants": artifacts},
        }
    finally:
        if con is not None:
            con.close()


def summarize(development: dict[str, Any]) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for name, budget in VARIANT_BUDGETS.items():
        overall = {
            w: float(d["evaluations"][name]["overall"][f"candidate_recall@{budget}"])
            for w, d in development.items()
        }
        cold = {
            w: float(d["evaluations"][name]["cold"][f"candidate_recall@{budget}"])
            for w, d in development.items()
        }
        oracle = {w: float(d["evaluations"][name]["overall"]["oracle_map@12"]) for w, d in development.items()}
        variants[name] = {
            "overall_recall": overall,
            "mean_overall_recall": float(np.mean(list(overall.values()))),
            "cold_recall": cold,
            "mean_cold_recall": float(np.mean(list(cold.values()))),
            "overall_oracle_map@12": oracle,
            "mean_overall_oracle_map@12": float(np.mean(list(oracle.values()))),
        }
    base = variants["base_300"]
    for row in variants.values():
        row["overall_delta"] = {w: row["overall_recall"][w] - base["overall_recall"][w] for w in development}
        row["cold_delta"] = {w: row["cold_recall"][w] - base["cold_recall"][w] for w in development}
        row["mean_overall_delta"] = float(np.mean(list(row["overall_delta"].values())))
        row["mean_cold_delta"] = float(np.mean(list(row["cold_delta"].values())))
    combined = variants["expanded_combined_400"]
    marginal = {w: d["marginal_cold_truth"] for w, d in development.items()}
    checks = {
        "combined_cold_recall_improves_all_four": all(v > 0 for v in combined["cold_delta"].values()),
        "mean_cold_delta_at_least_0_005": combined["mean_cold_delta"] >= 0.005,
        "combined_at_least_10_marginal_cold_truth_each": all(v["combined"] >= 10 for v in marginal.values()),
        "mean_overall_delta_at_least_0_0005": combined["mean_overall_delta"] >= 0.0005,
        "attribute_unique_cold_truth_at_least_two_windows": sum(v["attribute"] > 0 for v in marginal.values()) >= 2,
        "image_unique_cold_truth_all_four_windows": all(v["image"] > 0 for v in marginal.values()),
    }
    return {
        "variants": variants,
        "marginal_cold_truth": marginal,
        "retrieval_gate": {"passed": all(checks.values()), "checks": checks},
        "final_week": "not_run",
    }


def render_report(result: dict[str, Any]) -> str:
    windows = list(OUTER_WINDOWS)
    lines = [
        "# M3.5：属性×季节×图片冷商品召回",
        "",
        f"- retrieval gate：{result['summary']['retrieval_gate']['passed']}。",
        "- final week 未运行。",
        "- `cold truth pair`：验证周的去重用户—商品正例，且商品在 cutoff 前零交易。",
        "- `marginal cold truth`：相对 base 候选池新增命中的 cold truth pair；不是跨窗口独占命中。",
        "- 下表 Recall 的统计单位是 user-item truth pair，mean delta 的参照物是 base-300。",
        "",
        "## Cold Recall",
        "",
        "| variant | " + " | ".join(windows) + " | mean | mean delta |",
        "|---|" + "---:|" * (len(windows) + 2),
    ]
    for name, row in result["summary"]["variants"].items():
        lines.append("| " + name + " | " + " | ".join(f"{row['cold_recall'][w]:.6f}" for w in windows) + f" | {row['mean_cold_recall']:.6f} | {row['mean_cold_delta']:+.6f} |")
    gate_labels = {
        "combined_cold_recall_improves_all_four": "strict combined cold Recall 四窗均提升",
        "mean_cold_delta_at_least_0_005": "四窗 mean cold Recall absolute delta >= 0.005",
        "combined_at_least_10_marginal_cold_truth_each": "每窗 strict combined 相对 base 新增命中 >= 10 cold truth pairs",
        "mean_overall_delta_at_least_0_0005": "overall mean Recall absolute delta >= 0.0005",
        "attribute_unique_cold_truth_at_least_two_windows": "attribute 至少两窗存在相对 base 新增命中",
        "image_unique_cold_truth_all_four_windows": "image 四窗均存在相对 base 新增命中",
    }
    lines.extend(
        [
            "",
            "## Gate",
            "",
            *[
                f"- {gate_labels.get(key, key)}: {value}"
                for key, value in result["summary"]["retrieval_gate"]["checks"].items()
            ],
            "",
            f"- wall time：{result['elapsed_seconds']:.2f} 秒。",
            "- 机器可读证据：`metrics.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_m35(
    *,
    raw_dir: Path,
    transactions_path: Path,
    base_candidates: dict[str, tuple[Path, Path]],
    neighbor_metrics_path: Path,
    source_cache_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    validate_protocol()
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        sources = build_cold_sources(
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            base_candidates=base_candidates,
            neighbor_metrics_path=neighbor_metrics_path,
            cache_dir=source_cache_dir,
        )
        development = {}
        for window, cutoff in OUTER_WINDOWS.items():
            print(f"M3.5 evaluating {window} ({cutoff})", flush=True)
            development[window] = evaluate_window(
                window=window,
                cutoff=cutoff,
                transactions_path=transactions_path,
                base_path=base_candidates[cutoff][0],
                source_manifest=sources[cutoff],
                artifact_dir=artifact_dir / window,
            )
            gc.collect()
        summary = summarize(development)
        result = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M3.5",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "four outer development windows; final week not run",
            "contract": {
                "config": CONFIG,
                "variants": VARIANT_BUDGETS,
                "catalog": "optimistic_all_articles",
                "cold_definition": "article has no transaction before cutoff",
                "inventory_boundary": "no inventory/exposure or launch timestamps",
            },
            "inputs": {
                "transactions": _file_identity(transactions_path),
                "articles": _file_identity(raw_dir / "articles.csv"),
                "neighbor_metrics": _file_identity(neighbor_metrics_path),
            },
            "source_cache": sources,
            "development": development,
            "summary": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M3_5_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path.resolve()),
            "report": str(report_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
        }
        _write_json(metrics_path, result)
        report_path.write_text(render_report(result), encoding="utf-8")
        return result
    except Exception as error:
        _write_json(artifact_dir / f"failure-{time.time_ns()}.json", {"schema_version": "m3.5-failure-v1", "status": "failed", "error_type": type(error).__name__, "error": str(error), "elapsed_seconds": time.perf_counter() - started})
        raise
