from __future__ import annotations

import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m2 import _literal, _write_json
from .m25_retrieval import _candidate_metrics, _load_neighbors
from .m28 import _file_identity
from .m34 import OUTER_WINDOWS, _group_audit, _validate_base
from .m35 import CONFIG as M35_CONFIG
from .m35 import _date
from .m36 import _exact_subset_neighbors, _validate_m35_metrics


SCHEMA_VERSION = "m3.7-deep-visual-soft-season-retrieval-v1"
SOURCE_VARIANTS = (
    "hard_shallow_split",
    "soft_shallow_split",
    "hard_deep_split",
    "soft_deep_split",
)
CONFIG = {
    "history_weeks": 12,
    "personalized_seed_k": 5,
    "global_prototype_k": 50,
    "shallow_neighbor_depth": 100,
    "deep_neighbor_depth": 500,
    "personalized_route_k": 50,
    "global_route_k": 50,
    "source_k": 100,
    "rrf_constant": 60,
    "visual_rrf_weight": 1.0,
    "season_rrf_weight": 0.5,
    "hard_prior_type_lift": 1.15,
    "hard_prior_type_events_28d": 50,
    "fixed_total_budget": 300,
    "expanded_total_budget": 400,
    "device_query_batch_size": 128,
}
VARIANT_BUDGETS = {
    "base_300": 300,
    "historical_control_100": 100,
    "expanded_historical_control_400": 400,
    **{f"source_{name}_100": 100 for name in SOURCE_VARIANTS},
    **{f"expanded_{name}_400": 400 for name in SOURCE_VARIANTS},
    "fixed_soft_deep_split_300": 300,
}
PRIMARY_SOURCE = "soft_deep_split"


def validate_protocol() -> None:
    if any(cutoff >= "2020-09-16" for cutoff in OUTER_WINDOWS.values()):
        raise RuntimeError("M3.7 must not read final-week interactions")
    if CONFIG["shallow_neighbor_depth"] != 100 or CONFIG["deep_neighbor_depth"] != 500:
        raise RuntimeError("M3.7 visual-depth contract drifted")
    if CONFIG["personalized_route_k"] + CONFIG["global_route_k"] != CONFIG["source_k"]:
        raise RuntimeError("M3.7 split route budget drifted")
    if CONFIG["season_rrf_weight"] != 0.5:
        raise RuntimeError("M3.7 soft-season weight drifted")
    if set(SOURCE_VARIANTS) != {
        "hard_shallow_split",
        "soft_shallow_split",
        "hard_deep_split",
        "soft_deep_split",
    }:
        raise RuntimeError("M3.7 factorial source variants drifted")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_context(
    *,
    window: str,
    cutoff: str,
    m35_db_path: Path,
    transactions_path: Path,
    articles_path: Path,
    image_items: pd.DataFrame,
    db_path: Path,
) -> dict[str, Any]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("SET threads=8")
        con.execute("SET memory_limit='11GB'")
        con.execute(f"ATTACH {_literal(m35_db_path)} AS m35 (READ_ONLY)")
        con.execute(
            """
            CREATE TABLE base AS
              SELECT customer_id,article_id,candidate_rank FROM m35.base;
            CREATE TABLE eval_truth AS SELECT * FROM m35.eval_truth;
            CREATE TABLE historical_control AS
              SELECT customer_id,article_id,image_cold_rank AS candidate_rank
              FROM m35.image_source;
            CREATE TABLE eval_users AS SELECT DISTINCT customer_id FROM base;
            DETACH m35;
            """
        )
        con.register("image_items_frame", image_items[["row_index", "article_id"]])
        con.execute(
            f"""
            CREATE TABLE image_items AS SELECT * FROM image_items_frame;
            CREATE TABLE article_dim AS
              SELECT article_id,
                     try_cast(product_type_no AS INTEGER) AS product_type_no,
                     try_cast(graphical_appearance_no AS INTEGER) AS graphical_appearance_no,
                     try_cast(colour_group_code AS INTEGER) AS colour_group_code,
                     try_cast(perceived_colour_value_id AS INTEGER) AS perceived_colour_value_id
              FROM read_csv_auto({_literal(articles_path)},header=true,all_varchar=true);
            CREATE TABLE warm_catalog AS
              SELECT DISTINCT article_id FROM read_parquet({_literal(transactions_path)})
              WHERE t_dat<{_date(cutoff)};
            CREATE TABLE cold_catalog AS
              SELECT a.*,i.row_index FROM article_dim a
              JOIN image_items i USING(article_id)
              LEFT JOIN warm_catalog w USING(article_id)
              WHERE w.article_id IS NULL;
            CREATE TABLE current_events AS
              SELECT t.t_dat,t.article_id,a.product_type_no,a.graphical_appearance_no,
                     a.colour_group_code,a.perceived_colour_value_id
              FROM read_parquet({_literal(transactions_path)}) t
              JOIN article_dim a USING(article_id)
              WHERE t.t_dat>={_date(cutoff)}-INTERVAL 84 DAY AND t.t_dat<{_date(cutoff)};
            CREATE TABLE prior_events AS
              SELECT t.t_dat,t.article_id,a.product_type_no
              FROM read_parquet({_literal(transactions_path)}) t
              JOIN article_dim a USING(article_id)
              WHERE t.t_dat>={_date(cutoff)}-INTERVAL 1 YEAR-INTERVAL 84 DAY
                AND t.t_dat<{_date(cutoff)}-INTERVAL 1 YEAR;
            """
        )
        con.unregister("image_items_frame")
        current84, current28 = con.execute(
            f"SELECT count(*),count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 28 DAY) FROM current_events"
        ).fetchone()
        prior84, prior28 = con.execute(
            f"SELECT count(*),count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 1 YEAR-INTERVAL 28 DAY) FROM prior_events"
        ).fetchone()
        if min(map(int, (current84, current28, prior84, prior28))) <= 0:
            raise RuntimeError(f"M3.7 empty temporal support: {window}")
        con.execute(
            f"""
            CREATE TABLE prior_type AS
            WITH s AS (
              SELECT product_type_no,
                count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 1 YEAR-INTERVAL 28 DAY) AS events28,
                count(*) AS events84,
                count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 1 YEAR-INTERVAL 28 DAY)::DOUBLE/{int(prior28)}.0 AS share28,
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
            con.execute(
                f"""
                CREATE TABLE current_{key} AS
                WITH s AS (
                  SELECT {key},
                    count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 28 DAY) AS events28,
                    count(*) AS events84,
                    count(*) FILTER(WHERE t_dat>={_date(cutoff)}-INTERVAL 28 DAY)::DOUBLE/{int(current28)}.0 AS share28,
                    count(*)::DOUBLE/{int(current84)}.0 AS share84
                  FROM current_events GROUP BY {key})
                SELECT *,share28/nullif(share84,0) AS lift FROM s;
                """
            )
        score_terms = [
            "coalesce(greatest(ln(greatest(p.lift,1.0)),0.0)*p.events28/(p.events28+50.0),0.0)",
            "coalesce(greatest(ln(greatest(t.lift,1.0)),0.0)*t.events28/(t.events28+50.0),0.0)",
            "0.5*coalesce(greatest(ln(greatest(g.lift,1.0)),0.0)*g.events28/(g.events28+30.0),0.0)",
            "0.5*coalesce(greatest(ln(greatest(c.lift,1.0)),0.0)*c.events28/(c.events28+30.0),0.0)",
            "0.25*coalesce(greatest(ln(greatest(v.lift,1.0)),0.0)*v.events28/(v.events28+30.0),0.0)",
        ]
        con.execute(
            f"""
            CREATE TABLE cold_features AS
            SELECT a.*,({' + '.join(score_terms)})::DOUBLE AS season_score,
                   coalesce(p.lift,0.0)::DOUBLE AS prior_type_lift,
                   coalesce(p.events28,0)::BIGINT AS prior_type_events28,
                   coalesce(p.lift>={CONFIG['hard_prior_type_lift']}
                     AND p.events28>={CONFIG['hard_prior_type_events_28d']},FALSE) AS hard_eligible
            FROM cold_catalog a
            LEFT JOIN prior_type p USING(product_type_no)
            LEFT JOIN current_product_type_no t USING(product_type_no)
            LEFT JOIN current_graphical_appearance_no g USING(graphical_appearance_no)
            LEFT JOIN current_colour_group_code c USING(colour_group_code)
            LEFT JOIN current_perceived_colour_value_id v USING(perceived_colour_value_id);
            CREATE TABLE seeds AS
            WITH latest AS (
              SELECT x.customer_id,x.article_id,max(x.t_dat) AS latest_t_dat
              FROM read_parquet({_literal(transactions_path)}) x
              JOIN eval_users u USING(customer_id)
              JOIN image_items i USING(article_id)
              WHERE x.t_dat>={_date(cutoff)}-INTERVAL {CONFIG['history_weeks']} WEEK
                AND x.t_dat<{_date(cutoff)}
              GROUP BY x.customer_id,x.article_id), ranked AS (
              SELECT l.*,i.row_index,row_number() OVER(PARTITION BY customer_id
                ORDER BY latest_t_dat DESC,article_id) AS seed_rank
              FROM latest l JOIN image_items i USING(article_id))
            SELECT * FROM ranked WHERE seed_rank<={CONFIG['personalized_seed_k']};
            CREATE TABLE global_prototypes AS
            WITH hard_types AS (
              SELECT DISTINCT product_type_no FROM cold_features WHERE hard_eligible), counts AS (
              SELECT e.article_id,count(*) AS item_events
              FROM current_events e JOIN hard_types USING(product_type_no)
              JOIN image_items i ON e.article_id=i.article_id
              WHERE e.t_dat>={_date(cutoff)}-INTERVAL 28 DAY
              GROUP BY e.article_id), ranked AS (
              SELECT c.*,i.row_index,row_number() OVER(ORDER BY item_events DESC,c.article_id) AS prototype_rank
              FROM counts c JOIN image_items i USING(article_id))
            SELECT * FROM ranked WHERE prototype_rank<={CONFIG['global_prototype_k']};
            """
        )
        latest = con.execute("SELECT max(latest_t_dat) FROM seeds").fetchone()[0]
        if latest is not None and str(latest) >= cutoff:
            raise RuntimeError(f"M3.7 temporal leakage in seeds: {window}")
        seed_frame = con.execute(
            "SELECT customer_id,row_index,seed_rank FROM seeds ORDER BY customer_id,seed_rank"
        ).fetchdf()
        global_frame = con.execute(
            "SELECT row_index,item_events,prototype_rank FROM global_prototypes ORDER BY prototype_rank"
        ).fetchdf()
        if len(global_frame) != CONFIG["global_prototype_k"]:
            raise RuntimeError(f"M3.7 expected 50 global prototypes: {window} {len(global_frame)}")
        return {
            "connection": con,
            "db_path": db_path,
            "seed_frame": seed_frame,
            "global_frame": global_frame,
            "latest_seed_date": str(latest) if latest is not None else None,
        }
    except Exception:
        con.close()
        raise


def _update_candidate(
    candidates: dict[int, list[float | int]],
    row: int,
    *,
    weighted: float,
    raw: float,
    seed_rank: int,
    neighbor_rank: int,
) -> None:
    prior = candidates.get(row)
    if prior is None:
        candidates[row] = [weighted, raw, seed_rank, neighbor_rank, 1]
        return
    prior[4] = int(prior[4]) + 1
    if (weighted, raw, -seed_rank, -neighbor_rank) > (
        float(prior[0]),
        float(prior[1]),
        -int(prior[2]),
        -int(prior[3]),
    ):
        prior[0:4] = [weighted, raw, seed_rank, neighbor_rank]


def _rank_route(
    candidates: dict[int, list[float | int]],
    cold_by_row: dict[int, tuple[str, float, bool]],
    *,
    hard_only: bool,
    route_k: int,
) -> list[dict[str, Any]]:
    visual_order = sorted(
        candidates,
        key=lambda row: (
            -float(candidates[row][0]),
            -float(candidates[row][1]),
            int(candidates[row][2]),
            int(candidates[row][3]),
            cold_by_row[row][0],
        ),
    )
    visual_rank = {row: rank for rank, row in enumerate(visual_order, start=1)}
    eligible = [row for row in visual_order if not hard_only or cold_by_row[row][2]]
    season_order = sorted(eligible, key=lambda row: (-cold_by_row[row][1], cold_by_row[row][0]))
    season_rank = {row: rank for rank, row in enumerate(season_order, start=1)}
    scored = sorted(
        eligible,
        key=lambda row: (
            -(
                CONFIG["visual_rrf_weight"] / (CONFIG["rrf_constant"] + visual_rank[row])
                + CONFIG["season_rrf_weight"] / (CONFIG["rrf_constant"] + season_rank[row])
            ),
            visual_rank[row],
            season_rank[row],
            cold_by_row[row][0],
        ),
    )[:route_k]
    return [
        {
            "article_id": cold_by_row[row][0],
            "route_rank": rank,
            "visual_rank": visual_rank[row],
            "season_rank": season_rank[row],
            "combined_score": (
                CONFIG["visual_rrf_weight"] / (CONFIG["rrf_constant"] + visual_rank[row])
                + CONFIG["season_rrf_weight"] / (CONFIG["rrf_constant"] + season_rank[row])
            ),
            "visual_score": float(candidates[row][0]),
            "season_score": float(cold_by_row[row][1]),
            "best_neighbor_rank": int(candidates[row][3]),
            "seed_support": int(candidates[row][4]),
        }
        for rank, row in enumerate(scored, start=1)
    ]


def _build_route_frames(
    *,
    context: dict[str, Any],
    query_rows: np.ndarray,
    deep_indices: np.ndarray,
    deep_scores: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    con: duckdb.DuckDBPyConnection = context["connection"]
    cold_frame = con.execute(
        "SELECT row_index,article_id,season_score,hard_eligible FROM cold_features ORDER BY row_index"
    ).fetchdf()
    cold_by_row = {
        int(row.row_index): (str(row.article_id), float(row.season_score), bool(row.hard_eligible))
        for row in cold_frame.itertuples(index=False)
    }
    query_position = {int(row): index for index, row in enumerate(query_rows)}
    personal_records: list[dict[str, Any]] = []
    grouped = context["seed_frame"].groupby("customer_id", sort=True)
    for user_index, (customer_id, seeds) in enumerate(grouped, start=1):
        shallow: dict[int, list[float | int]] = {}
        deep: dict[int, list[float | int]] = {}
        for seed in seeds.itertuples(index=False):
            seed_row = int(seed.row_index)
            seed_rank = int(seed.seed_rank)
            discount = 1.0 / (1.0 + 0.1 * (seed_rank - 1))
            position = query_position[seed_row]
            for neighbor_rank, (neighbor_row, cosine) in enumerate(
                zip(deep_indices[position], deep_scores[position]), start=1
            ):
                row = int(neighbor_row)
                if row not in cold_by_row:
                    continue
                raw = float(cosine)
                weighted = raw * discount
                _update_candidate(
                    deep,
                    row,
                    weighted=weighted,
                    raw=raw,
                    seed_rank=seed_rank,
                    neighbor_rank=neighbor_rank,
                )
                if neighbor_rank <= CONFIG["shallow_neighbor_depth"]:
                    _update_candidate(
                        shallow,
                        row,
                        weighted=weighted,
                        raw=raw,
                        seed_rank=seed_rank,
                        neighbor_rank=neighbor_rank,
                    )
        for depth_name, candidates in (("shallow", shallow), ("deep", deep)):
            for gate_name, hard_only in (("hard", True), ("soft", False)):
                variant = f"{gate_name}_{depth_name}_split"
                for record in _rank_route(
                    candidates,
                    cold_by_row,
                    hard_only=hard_only,
                    route_k=CONFIG["personalized_route_k"],
                ):
                    personal_records.append(
                        {"variant": variant, "customer_id": str(customer_id), **record}
                    )
        if user_index % 1000 == 0 or user_index == len(grouped):
            print(f"M3.7 personalized routes {user_index}/{len(grouped)} users", flush=True)

    global_candidates = {"shallow": {}, "deep": {}}
    for prototype in context["global_frame"].itertuples(index=False):
        seed_row = int(prototype.row_index)
        seed_rank = int(prototype.prototype_rank)
        weight = math.log1p(int(prototype.item_events)) / (1.0 + 0.05 * (seed_rank - 1))
        position = query_position[seed_row]
        for neighbor_rank, (neighbor_row, cosine) in enumerate(
            zip(deep_indices[position], deep_scores[position]), start=1
        ):
            row = int(neighbor_row)
            if row not in cold_by_row:
                continue
            raw = float(cosine)
            weighted = raw * weight
            _update_candidate(
                global_candidates["deep"],
                row,
                weighted=weighted,
                raw=raw,
                seed_rank=seed_rank,
                neighbor_rank=neighbor_rank,
            )
            if neighbor_rank <= CONFIG["shallow_neighbor_depth"]:
                _update_candidate(
                    global_candidates["shallow"],
                    row,
                    weighted=weighted,
                    raw=raw,
                    seed_rank=seed_rank,
                    neighbor_rank=neighbor_rank,
                )
    global_records: list[dict[str, Any]] = []
    for depth_name, candidates in global_candidates.items():
        for gate_name, hard_only in (("hard", True), ("soft", False)):
            variant = f"{gate_name}_{depth_name}_split"
            for record in _rank_route(
                candidates,
                cold_by_row,
                hard_only=hard_only,
                route_k=CONFIG["global_route_k"],
            ):
                global_records.append({"variant": variant, **record})
    personal = pd.DataFrame.from_records(personal_records)
    global_frame = pd.DataFrame.from_records(global_records)
    if personal.empty or global_frame.empty:
        raise RuntimeError("M3.7 produced an empty route frame")
    route_audit = {
        "cold_items_with_image": int(len(cold_frame)),
        "hard_eligible_cold_items": int(cold_frame["hard_eligible"].sum()),
        "users_with_personalized_seed": int(context["seed_frame"]["customer_id"].nunique()),
        "personalized_rows": int(len(personal)),
        "global_rows": int(len(global_frame)),
        "personalized_max_rows_per_user_variant": int(
            personal.groupby(["variant", "customer_id"]).size().max()
        ),
        "global_max_rows_per_variant": int(global_frame.groupby("variant").size().max()),
    }
    if route_audit["personalized_max_rows_per_user_variant"] > CONFIG["personalized_route_k"]:
        raise RuntimeError("M3.7 personalized route cap failed")
    if route_audit["global_max_rows_per_variant"] > CONFIG["global_route_k"]:
        raise RuntimeError("M3.7 global route cap failed")
    return personal, global_frame, route_audit


def _evaluate_window(
    *,
    window: str,
    cutoff: str,
    context: dict[str, Any],
    personal: pd.DataFrame,
    global_frame: pd.DataFrame,
    artifact_dir: Path,
) -> dict[str, Any]:
    con: duckdb.DuckDBPyConnection = context["connection"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    con.register("personal_frame", personal)
    con.register("global_frame", global_frame)
    con.execute("CREATE TABLE personal_routes AS SELECT * FROM personal_frame")
    con.execute("CREATE TABLE global_routes AS SELECT * FROM global_frame")
    con.unregister("personal_frame")
    con.unregister("global_frame")
    con.execute("CREATE TABLE base_300 AS SELECT * FROM base")
    con.execute("CREATE TABLE historical_control_100 AS SELECT * FROM historical_control")
    for source in SOURCE_VARIANTS:
        con.execute(
            f"""
            CREATE TABLE source_{source}_100 AS
            WITH p AS (
              SELECT customer_id,article_id,route_rank AS personalized_rank,
                     NULL::BIGINT AS global_rank,1 AS route
              FROM personal_routes WHERE variant='{source}'), g AS (
              SELECT u.customer_id,g.article_id,NULL::BIGINT AS personalized_rank,
                     g.route_rank AS global_rank,2 AS route
              FROM eval_users u CROSS JOIN global_routes g
              WHERE g.variant='{source}'
                AND NOT EXISTS (
                  SELECT 1 FROM p WHERE p.customer_id=u.customer_id AND p.article_id=g.article_id)), lanes AS (
              SELECT * FROM p UNION ALL SELECT * FROM g)
            SELECT customer_id,article_id,
                   row_number() OVER(PARTITION BY customer_id
                     ORDER BY route,coalesce(personalized_rank,global_rank),article_id) AS candidate_rank,
                   personalized_rank,global_rank
            FROM lanes;
            """
        )

    def expanded(source_table: str, output_table: str) -> None:
        con.execute(
            f"""
            CREATE TABLE {output_table} AS
            WITH sizes AS (SELECT customer_id,count(*) AS n FROM base GROUP BY customer_id), new AS (
              SELECT s.*,row_number() OVER(PARTITION BY s.customer_id
                ORDER BY s.candidate_rank,s.article_id) AS new_rank
              FROM {source_table} s
              WHERE NOT EXISTS (SELECT 1 FROM base b
                WHERE b.customer_id=s.customer_id AND b.article_id=s.article_id))
            SELECT * FROM base
            UNION ALL
            SELECT n.customer_id,n.article_id,(s.n+n.new_rank)::BIGINT AS candidate_rank
            FROM new n JOIN sizes s USING(customer_id) WHERE n.new_rank<={CONFIG['source_k']};
            """
        )

    expanded("historical_control_100", "expanded_historical_control_400")
    for source in SOURCE_VARIANTS:
        expanded(f"source_{source}_100", f"expanded_{source}_400")
    con.execute(
        f"""
        CREATE TABLE primary_new AS
        SELECT s.*,row_number() OVER(PARTITION BY s.customer_id
          ORDER BY s.candidate_rank,s.article_id) AS new_rank
        FROM source_{PRIMARY_SOURCE}_100 s
        WHERE NOT EXISTS (SELECT 1 FROM base b
          WHERE b.customer_id=s.customer_id AND b.article_id=s.article_id);
        CREATE TABLE fixed_limits AS
          SELECT u.customer_id,count(n.article_id) AS source_rows,
                 {CONFIG['fixed_total_budget']}-count(n.article_id) AS base_limit
          FROM eval_users u LEFT JOIN primary_new n USING(customer_id)
          GROUP BY u.customer_id;
        CREATE TABLE fixed_base AS
          SELECT b.* FROM base b JOIN fixed_limits l USING(customer_id)
          WHERE b.candidate_rank<=l.base_limit;
        CREATE TABLE fixed_base_counts AS
          SELECT u.customer_id,count(b.article_id) AS kept_base_rows
          FROM eval_users u LEFT JOIN fixed_base b USING(customer_id) GROUP BY u.customer_id;
        CREATE TABLE fixed_soft_deep_split_300 AS
          SELECT * FROM fixed_base
          UNION ALL
          SELECT n.customer_id,n.article_id,(c.kept_base_rows+n.new_rank)::BIGINT AS candidate_rank
          FROM primary_new n JOIN fixed_base_counts c USING(customer_id);
        """
    )
    audits = {
        name: _group_audit(con, name, budget) for name, budget in VARIANT_BUDGETS.items()
    }
    evaluations: dict[str, Any] = {}
    for name, budget in VARIANT_BUDGETS.items():
        evaluations[name] = {
            "overall": _candidate_metrics(con, name, budget, truth_table="eval_truth"),
            "warm": _candidate_metrics(
                con, name, budget, "item_temperature='warm'", truth_table="eval_truth"
            ),
            "cold": _candidate_metrics(
                con, name, budget, "item_temperature='cold'", truth_table="eval_truth"
            ),
        }
    marginal: dict[str, Any] = {}
    overlap: dict[str, Any] = {}
    for source in SOURCE_VARIANTS:
        row = con.execute(
            f"""
            WITH missed AS (
              SELECT t.customer_id,t.article_id FROM eval_truth t
              WHERE t.item_temperature='cold' AND NOT EXISTS (
                SELECT 1 FROM base b WHERE b.customer_id=t.customer_id AND b.article_id=t.article_id)), flags AS (
              SELECT m.*,
                EXISTS(SELECT 1 FROM personal_routes p WHERE p.variant='{source}'
                  AND p.customer_id=m.customer_id AND p.article_id=m.article_id) AS personal_hit,
                EXISTS(SELECT 1 FROM global_routes g WHERE g.variant='{source}'
                  AND g.article_id=m.article_id) AS global_hit,
                EXISTS(SELECT 1 FROM source_{source}_100 s
                  WHERE s.customer_id=m.customer_id AND s.article_id=m.article_id) AS source_hit
              FROM missed m)
            SELECT count(*) FILTER(WHERE source_hit),count(*) FILTER(WHERE personal_hit),
                   count(*) FILTER(WHERE global_hit),
                   count(*) FILTER(WHERE personal_hit AND NOT global_hit),
                   count(*) FILTER(WHERE global_hit AND NOT personal_hit),
                   count(*) FILTER(WHERE personal_hit AND global_hit),count(*)
            FROM flags
            """
        ).fetchone()
        marginal[source] = {
            "source_newly_hit_cold_truth_pairs_vs_base": int(row[0]),
            "personalized_newly_hit_cold_truth_pairs_vs_base": int(row[1]),
            "global_newly_hit_cold_truth_pairs_vs_base": int(row[2]),
            "personalized_exclusive_vs_global_pairs": int(row[3]),
            "global_exclusive_vs_personalized_pairs": int(row[4]),
            "hit_by_both_routes_pairs": int(row[5]),
            "base_missed_cold_truth_pairs": int(row[6]),
        }
        overlap_row = con.execute(
            f"""
            SELECT
              (SELECT count(*) FROM personal_routes p JOIN global_routes g USING(article_id)
                WHERE p.variant='{source}' AND g.variant='{source}'),
              (SELECT count(*) FROM personal_routes WHERE variant='{source}'),
              (SELECT count(*) FROM eval_users)*(SELECT count(*) FROM global_routes WHERE variant='{source}')
            """
        ).fetchone()
        overlap[source] = {
            "overlap_user_item_rows": int(overlap_row[0]),
            "personalized_user_item_rows": int(overlap_row[1]),
            "global_copied_user_item_rows": int(overlap_row[2]),
            "overlap_rate_of_personalized_rows": (
                float(overlap_row[0]) / int(overlap_row[1]) if int(overlap_row[1]) else 0.0
            ),
        }
    control_marginal = con.execute(
        """
        SELECT count(*) FROM eval_truth t
        WHERE t.item_temperature='cold'
          AND NOT EXISTS(SELECT 1 FROM base b
            WHERE b.customer_id=t.customer_id AND b.article_id=t.article_id)
          AND EXISTS(SELECT 1 FROM historical_control_100 h
            WHERE h.customer_id=t.customer_id AND h.article_id=t.article_id)
        """
    ).fetchone()[0]
    fixed = con.execute(
        """
        SELECT (SELECT count(*) FROM base)-(SELECT count(*) FROM fixed_base),
               (SELECT count(*) FROM primary_new),
               (SELECT count(*) FROM fixed_soft_deep_split_300)
        """
    ).fetchone()
    artifacts: dict[str, Any] = {}
    for name in VARIANT_BUDGETS:
        path = artifact_dir / f"{name}.parquet"
        con.execute(
            f"COPY (SELECT * FROM {name} ORDER BY customer_id,candidate_rank,article_id) "
            f"TO {_literal(path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
        )
        artifacts[name] = _file_identity(path)
    for route_name, table in (("personalized_routes", "personal_routes"), ("global_routes", "global_routes")):
        path = artifact_dir / f"{route_name}.parquet"
        con.execute(
            f"COPY (SELECT * FROM {table}) TO {_literal(path)} "
            "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
        )
        artifacts[route_name] = _file_identity(path)
    con.execute("CHECKPOINT")
    con.close()
    context["connection"] = None
    return {
        "window": window,
        "cutoff": cutoff,
        "latest_seed_date": context["latest_seed_date"],
        "audits": audits,
        "evaluations": evaluations,
        "marginal_cold_truth": {
            "historical_control_vs_base": int(control_marginal),
            **marginal,
        },
        "route_overlap": overlap,
        "fixed_budget": {
            "base_candidate_rows_replaced": int(fixed[0]),
            "source_candidate_rows_inserted": int(fixed[1]),
            "final_candidate_rows": int(fixed[2]),
        },
        "artifacts": {
            "evaluation_db": _file_identity(context["db_path"]),
            "tables": artifacts,
        },
    }


def summarize(development: dict[str, Any]) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for name, budget in VARIANT_BUDGETS.items():
        overall = {
            window: float(data["evaluations"][name]["overall"][f"candidate_recall@{budget}"])
            for window, data in development.items()
        }
        cold = {
            window: float(data["evaluations"][name]["cold"][f"candidate_recall@{budget}"])
            for window, data in development.items()
        }
        oracle = {
            window: float(data["evaluations"][name]["overall"]["oracle_map@12"])
            for window, data in development.items()
        }
        variants[name] = {
            "overall_recall": overall,
            "mean_overall_recall": float(np.mean(list(overall.values()))),
            "cold_recall": cold,
            "mean_cold_recall": float(np.mean(list(cold.values()))),
            "overall_oracle_map@12": oracle,
            "mean_overall_oracle_map@12": float(np.mean(list(oracle.values()))),
        }
    base = variants["base_300"]
    control = variants["expanded_historical_control_400"]
    primary = variants[f"expanded_{PRIMARY_SOURCE}_400"]
    for row in variants.values():
        row["overall_delta_vs_base"] = {
            window: row["overall_recall"][window] - base["overall_recall"][window]
            for window in development
        }
        row["cold_delta_vs_base"] = {
            window: row["cold_recall"][window] - base["cold_recall"][window]
            for window in development
        }
        row["mean_overall_delta_vs_base"] = float(
            np.mean(list(row["overall_delta_vs_base"].values()))
        )
        row["mean_cold_delta_vs_base"] = float(
            np.mean(list(row["cold_delta_vs_base"].values()))
        )
    structural = all(
        data["latest_seed_date"] is None or data["latest_seed_date"] < data["cutoff"]
        for data in development.values()
    ) and all(
        audit["invalid_groups"] == 0
        for data in development.values()
        for audit in data["audits"].values()
    )
    spring_marginal = development["spring_20200318"]["marginal_cold_truth"][PRIMARY_SOURCE][
        "source_newly_hit_cold_truth_pairs_vs_base"
    ]
    checks = {
        "primary_cold_recall_above_base_all_windows": all(
            primary["cold_recall"][window] > base["cold_recall"][window]
            for window in development
        ),
        "primary_cold_recall_not_below_equal_budget_control_all_windows": all(
            primary["cold_recall"][window] >= control["cold_recall"][window]
            for window in development
        ),
        "primary_mean_cold_recall_above_control": (
            primary["mean_cold_recall"] > control["mean_cold_recall"]
        ),
        "spring_at_least_10_newly_hit_cold_truth_pairs_vs_base": spring_marginal >= 10,
        "primary_mean_overall_recall_delta_vs_base_at_least_0_0005": (
            primary["mean_overall_delta_vs_base"] >= 0.0005
        ),
        "candidate_temporal_and_budget_audits_pass": structural,
    }
    return {
        "variants": variants,
        "retrieval_gate": {"passed": all(checks.values()), "checks": checks},
        "primary_source": PRIMARY_SOURCE,
        "spring_primary_newly_hit_cold_truth_pairs_vs_base": int(spring_marginal),
        "final_week": "not_run",
    }


def render_report(result: dict[str, Any]) -> str:
    windows = list(OUTER_WINDOWS)
    selected = [
        "base_300",
        "expanded_historical_control_400",
        "expanded_hard_shallow_split_400",
        "expanded_soft_shallow_split_400",
        "expanded_hard_deep_split_400",
        "expanded_soft_deep_split_400",
        "fixed_soft_deep_split_300",
    ]
    lines = [
        "# M3.7：全评测用户深层视觉召回与季节软控制",
        "",
        f"- 候选召回晋级门槛：{result['summary']['retrieval_gate']['passed']}。",
        "- final week：not run；本阶段没有训练排序器。",
        "",
        "## 术语",
        "",
        "- `base`（基础候选池）：六路倒数排名融合 Top100 加最多 200 个 Item2Vec 商品协同向量模型独有候选，每用户 100--300 个。",
        "- `hard`（季节硬门槛）：使用 M3.5 商品类型购买占比提升倍数和交易支持量门槛，不满足就删除冷商品候选。",
        "- `soft`（季节软分数）：不删除有图冷商品；季节属性只以 0.5 权重参与路内倒数排名融合排序。",
        "- `shallow/deep`（浅层/深层近邻）：每个图片查询商品分别取全目录穷举余弦相似度 Top100/Top500 视觉近邻。",
        "- `split 50+50`：个性化路径最多 50 个、全局回退路径最多 50 个，先分别截断再去重。",
        "- `marginal truth pair`（相对基础池新增命中的正例对）：该验证用户—商品正例不在该用户 base 中，但被当前方案命中。",
        "- 下表 Recall（召回率）先计算每个用户的真实商品进入候选池的比例，再对用户平均。",
        "",
        "## Cold candidate Recall",
        "",
        "| candidate variant | " + " | ".join(windows) + " | mean | mean delta vs base |",
        "|---|" + "---:|" * (len(windows) + 2),
    ]
    for name in selected:
        row = result["summary"]["variants"][name]
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(f"{row['cold_recall'][window]:.6f}" for window in windows)
            + f" | {row['mean_cold_recall']:.6f} | {row['mean_cold_delta_vs_base']:+.6f} |"
        )
    lines.extend(["", "## 主方案相对 base 新增命中的 cold truth pairs", ""])
    lines.append("| window | source pairs | personalized pairs | global pairs | personalized-only | global-only | both |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for window in windows:
        row = result["development"][window]["marginal_cold_truth"][PRIMARY_SOURCE]
        lines.append(
            f"| {window} | {row['source_newly_hit_cold_truth_pairs_vs_base']} | "
            f"{row['personalized_newly_hit_cold_truth_pairs_vs_base']} | "
            f"{row['global_newly_hit_cold_truth_pairs_vs_base']} | "
            f"{row['personalized_exclusive_vs_global_pairs']} | "
            f"{row['global_exclusive_vs_personalized_pairs']} | "
            f"{row['hit_by_both_routes_pairs']} |"
        )
    lines.extend(["", "## 晋级门槛", ""])
    for name, passed in result["summary"]["retrieval_gate"]["checks"].items():
        lines.append(f"- `{name}`：{passed}。")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- expanded 候选追加在 base 后，当前 Top12 MAP 不衡量新增图片候选的排序兑现。",
            "- `articles.csv` 没有上架、库存和曝光时间；cold 结果只适用于乐观离线目录假设。",
            "- 机器可读证据为 `metrics.json`；候选表、路径表与评测数据库在 ignored artifacts。",
            "",
        ]
    )
    return "\n".join(lines)


def run_m37(
    *,
    m35_metrics_path: Path,
    neighbor_metrics_path: Path,
    output_dir: Path,
    artifact_dir: Path,
    run_id: str,
    device: str = "cuda",
) -> dict[str, Any]:
    validate_protocol()
    output_dir = output_dir.resolve()
    artifact_dir = artifact_dir.resolve()
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(f"M3.7 output exists: {output_dir} or {artifact_dir}")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    m35_metrics_path = m35_metrics_path.resolve()
    neighbor_metrics_path = neighbor_metrics_path.resolve()
    m35 = _validate_m35_metrics(m35_metrics_path)
    if _file_identity(neighbor_metrics_path) != m35["inputs"]["neighbor_metrics"]:
        raise ValueError("M3.7 neighbor artifact differs from frozen M3.5 input")
    _, existing_indices, existing_scores, image_items = _load_neighbors(neighbor_metrics_path)
    neighbor_metrics = _read_json(neighbor_metrics_path)
    embedding_metrics_path = Path(neighbor_metrics["contract"]["source"]["metrics_path"]).resolve()
    transactions_path = Path(m35["inputs"]["transactions"]["path"]).resolve()
    articles_path = Path(m35["inputs"]["articles"]["path"]).resolve()
    contexts: dict[str, dict[str, Any]] = {}
    all_query_rows: set[int] = set()
    try:
        for window, cutoff in OUTER_WINDOWS.items():
            print(f"M3.7 preparing all-user queries {window} ({cutoff})", flush=True)
            db_identity = m35["development"][window]["artifacts"]["evaluation_db"]
            if _file_identity(Path(db_identity["path"])) != db_identity:
                raise ValueError(f"M3.7 M3.5 evaluation DB identity drift: {window}")
            base_input = Path(m35["source_cache"][cutoff]["inputs"]["base"]["path"])
            _validate_base(base_input, base_input.with_name("candidate-manifest.json"), cutoff)
            context = _prepare_context(
                window=window,
                cutoff=cutoff,
                m35_db_path=Path(db_identity["path"]),
                transactions_path=transactions_path,
                articles_path=articles_path,
                image_items=image_items,
                db_path=artifact_dir / window / "evaluation.duckdb",
            )
            contexts[window] = context
            all_query_rows.update(context["seed_frame"]["row_index"].astype(int).tolist())
            all_query_rows.update(context["global_frame"]["row_index"].astype(int).tolist())
        deep = _exact_subset_neighbors(
            query_rows=np.asarray(sorted(all_query_rows), dtype=np.int64),
            embedding_metrics_path=embedding_metrics_path,
            existing_indices=existing_indices,
            existing_scores=existing_scores,
            output_dir=artifact_dir / "all-eval-query-exact-top500",
            device=device,
            stage_label="M3.7",
            preserve_existing_top100_on_backend_drift=(device == "cpu"),
        )
        development: dict[str, Any] = {}
        route_audits: dict[str, Any] = {}
        for window, cutoff in OUTER_WINDOWS.items():
            print(f"M3.7 building and evaluating {window} ({cutoff})", flush=True)
            personal, global_frame, route_audit = _build_route_frames(
                context=contexts[window],
                query_rows=deep["query_rows"],
                deep_indices=deep["indices"],
                deep_scores=deep["scores"],
            )
            route_audits[window] = route_audit
            development[window] = _evaluate_window(
                window=window,
                cutoff=cutoff,
                context=contexts[window],
                personal=personal,
                global_frame=global_frame,
                artifact_dir=artifact_dir / window,
            )
            del personal, global_frame
            gc.collect()
        summary = summarize(development)
        result = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M3.7",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "four 10% hash development windows; final week not run; retrieval only",
            "contract": {
                "config": CONFIG,
                "source_variants": SOURCE_VARIANTS,
                "candidate_variants": VARIANT_BUDGETS,
                "primary_source": PRIMARY_SOURCE,
                "catalog": "optimistic_all_articles",
                "cold_definition": "article has no transaction before cutoff",
                "inventory_boundary": "no inventory, exposure, or launch timestamps",
            },
            "inputs": {
                "m35_metrics": _file_identity(m35_metrics_path),
                "neighbor_metrics": _file_identity(neighbor_metrics_path),
                "embedding_metrics": _file_identity(embedding_metrics_path),
                "transactions": _file_identity(transactions_path),
                "articles": _file_identity(articles_path),
            },
            "deep_neighbors": deep["evidence"],
            "route_audits": route_audits,
            "development": development,
            "summary": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M3_7_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path),
            "report": str(report_path),
            "artifact_dir": str(artifact_dir),
        }
        _write_json(metrics_path, result)
        report_path.write_text(render_report(result), encoding="utf-8")
        return result
    except Exception as error:
        _write_json(
            artifact_dir / f"failure-{time.time_ns()}.json",
            {
                "schema_version": "m3.7-failure-v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
    finally:
        for context in contexts.values():
            connection = context.get("connection")
            if connection is not None:
                connection.close()
