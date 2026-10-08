from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import duckdb

from .audit import prepare_tabular_connection


SOURCES = (
    "repurchase",
    "recent_popularity",
    "product_family",
    "user_day_covisit",
    "age_popularity",
    "attribute_content",
)

FUSION_PROFILES: dict[str, dict[str, float]] = {
    "equal": {source: 1.0 for source in SOURCES},
    "diversity": {
        "repurchase": 1.0,
        "recent_popularity": 0.5,
        "product_family": 1.0,
        "user_day_covisit": 1.5,
        "age_popularity": 0.75,
        "attribute_content": 1.25,
    },
    "collaborative": {
        "repurchase": 1.25,
        "recent_popularity": 0.5,
        "product_family": 1.25,
        "user_day_covisit": 1.5,
        "age_popularity": 0.75,
        "attribute_content": 0.75,
    },
    "popularity": {
        "repurchase": 0.75,
        "recent_popularity": 1.0,
        "product_family": 0.75,
        "user_day_covisit": 1.0,
        "age_popularity": 1.25,
        "attribute_content": 0.75,
    },
}


@dataclass(frozen=True)
class M1Config:
    cutoff: str | None = None
    sample_rate: float = 0.01
    history_weeks: int = 12
    popularity_days: int = 28
    covisit_days: int = 28
    candidate_k: int = 100
    source_k: int = 100
    covisit_neighbor_k: int = 50
    max_user_day_items: int = 20
    min_covisit_count: int = 2
    rrf_constant: int = 60
    content_seed_k: int = 20
    content_type_pool_k: int = 50
    content_garment_pool_k: int = 30
    fusion_profile: str = "equal"
    evaluation_mode: str = "diagnostic"
    catalog_protocol: str = "optimistic_all_articles"
    customer_policy: str = "age_only"
    duplicate_policy: str = "keep_all_events"

    def validate(self) -> None:
        if not 0 < self.sample_rate <= 1:
            raise ValueError("sample_rate must be in (0, 1]")
        for name in (
            "history_weeks",
            "popularity_days",
            "covisit_days",
            "candidate_k",
            "source_k",
            "covisit_neighbor_k",
            "max_user_day_items",
            "min_covisit_count",
            "rrf_constant",
            "content_seed_k",
            "content_type_pool_k",
            "content_garment_pool_k",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.catalog_protocol not in {
            "optimistic_all_articles",
            "strict",
        }:
            raise ValueError("unknown catalog_protocol")
        if self.customer_policy != "age_only":
            raise ValueError("primary M1 currently supports customer_policy=age_only")
        if self.duplicate_policy != "keep_all_events":
            raise ValueError("primary M1 keeps all transaction events")
        if self.fusion_profile not in FUSION_PROFILES:
            raise ValueError(f"unknown fusion_profile: {self.fusion_profile}")
        if self.evaluation_mode not in {"diagnostic", "final"}:
            raise ValueError("evaluation_mode must be diagnostic or final")


def _date_literal(value: str) -> str:
    return "DATE '" + value.replace("'", "''") + "'"


def _path_literal(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _scalar(connection: duckdb.DuckDBPyConnection, query: str) -> Any:
    row = connection.execute(query).fetchone()
    if row is None:
        raise RuntimeError("query returned no rows")
    return row[0]


def _resolve_cutoff(
    connection: duckdb.DuckDBPyConnection, configured: str | None
) -> str:
    if configured:
        cutoff = _scalar(connection, f"SELECT {_date_literal(configured)}")
    else:
        cutoff = _scalar(connection, "SELECT max(t_dat) - INTERVAL 6 DAY FROM transactions")
    return cutoff.strftime("%Y-%m-%d")


def _create_population(
    connection: duckdb.DuckDBPyConnection, cutoff: str, config: M1Config
) -> dict[str, int]:
    cutoff_sql = _date_literal(cutoff)
    threshold = int(config.sample_rate * 1_000_000)
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_warm_catalog AS
        SELECT DISTINCT article_id
        FROM transactions
        WHERE t_dat < {cutoff_sql}
        """
    )
    if config.catalog_protocol == "optimistic_all_articles":
        connection.execute(
            """
            CREATE OR REPLACE TEMP TABLE m1_eligible_catalog AS
            SELECT DISTINCT article_id FROM articles
            """
        )
    else:
        connection.execute(
            """
            CREATE OR REPLACE TEMP TABLE m1_eligible_catalog AS
            SELECT article_id FROM m1_warm_catalog
            """
        )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_truth_all AS
        SELECT DISTINCT customer_id, article_id
        FROM transactions
        WHERE t_dat >= {cutoff_sql}
          AND t_dat < {cutoff_sql} + INTERVAL 7 DAY
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_users AS
        SELECT DISTINCT customer_id
        FROM m1_truth_all
        WHERE hash(customer_id) % 1000000 < {threshold}
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m1_truth AS
        SELECT truth.*
        FROM m1_truth_all truth
        SEMI JOIN m1_users USING (customer_id)
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m1_truth_labeled AS
        SELECT
            truth.customer_id,
            truth.article_id,
            CASE WHEN warm.article_id IS NULL THEN 'cold' ELSE 'warm' END AS item_temperature
        FROM m1_truth truth
        LEFT JOIN m1_warm_catalog warm USING (article_id)
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW m1_truth_warm AS
        SELECT customer_id, article_id
        FROM m1_truth_labeled
        WHERE item_temperature = 'warm'
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW m1_truth_cold AS
        SELECT customer_id, article_id
        FROM m1_truth_labeled
        WHERE item_temperature = 'cold'
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_user_segments AS
        WITH counts AS (
            SELECT
                users.customer_id,
                count(history.customer_id) AS all_history_events,
                count(history.customer_id) FILTER (
                    WHERE history.t_dat >= {cutoff_sql} - INTERVAL {config.history_weeks} WEEK
                ) AS recent_history_events
            FROM m1_users users
            LEFT JOIN transactions history
              ON users.customer_id = history.customer_id
             AND history.t_dat < {cutoff_sql}
            GROUP BY users.customer_id
        )
        SELECT *,
            CASE WHEN all_history_events = 0 THEN 'cold' ELSE 'warm' END AS user_segment,
            CASE
                WHEN recent_history_events = 0 THEN 'inactive_12w'
                WHEN recent_history_events <= 5 THEN 'low_1_5'
                WHEN recent_history_events <= 20 THEN 'medium_6_20'
                ELSE 'high_21_plus'
            END AS activity_segment
        FROM counts
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_item_training_counts AS
        SELECT article_id, count(*) AS event_count
        FROM transactions
        WHERE t_dat < {cutoff_sql}
        GROUP BY article_id
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m1_item_segments AS
        WITH ranked AS (
            SELECT article_id, event_count,
                   ntile(10) OVER (ORDER BY event_count DESC, article_id) AS popularity_decile
            FROM m1_item_training_counts
        )
        SELECT article_id, event_count, popularity_decile,
            CASE
                WHEN popularity_decile = 1 THEN 'head_top10pct'
                WHEN popularity_decile <= 5 THEN 'mid_next40pct'
                ELSE 'tail_bottom50pct'
            END AS item_segment
        FROM ranked
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m1_truth_item_segments AS
        SELECT truth.*,
               coalesce(segments.item_segment, 'unseen_before_cutoff') AS item_segment
        FROM m1_truth truth
        LEFT JOIN m1_item_segments segments USING (article_id)
        """
    )
    overall_pairs = int(_scalar(connection, "SELECT count(*) FROM m1_truth"))
    warm_pairs = int(_scalar(connection, "SELECT count(*) FROM m1_truth_warm"))
    cold_pairs = int(_scalar(connection, "SELECT count(*) FROM m1_truth_cold"))
    if warm_pairs + cold_pairs != overall_pairs:
        raise RuntimeError("warm/cold truth pair conservation failed")
    return {
        "all_validation_users": int(_scalar(connection, "SELECT count(*) FROM (SELECT DISTINCT customer_id FROM m1_truth_all)")),
        "sampled_validation_users": int(_scalar(connection, "SELECT count(*) FROM m1_users")),
        "sampled_user_fingerprint": str(
            _scalar(
                connection,
                "SELECT md5(coalesce(string_agg(customer_id, '|' ORDER BY customer_id), '')) FROM m1_users",
            )
        ),
        "sampled_truth_pairs": overall_pairs,
        "warm_truth_pairs": warm_pairs,
        "cold_truth_pairs": cold_pairs,
        "warm_catalog_size": int(_scalar(connection, "SELECT count(*) FROM m1_warm_catalog")),
        "eligible_catalog_size": int(_scalar(connection, "SELECT count(*) FROM m1_eligible_catalog")),
        "eligible_catalog_leading_zero_items": int(
            _scalar(
                connection,
                "SELECT count(*) FROM m1_eligible_catalog WHERE article_id LIKE '0%'",
            )
        ),
    }


def _eligible_catalog_sql() -> str:
    return "SELECT article_id FROM m1_eligible_catalog"


def _create_repurchase(
    connection: duckdb.DuckDBPyConnection,
    cutoff_sql: str,
    config: M1Config,
) -> None:
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_repurchase AS
        WITH scored AS (
            SELECT
                t.customer_id,
                t.article_id,
                max(t.t_dat) AS last_purchase,
                count(*) AS event_count,
                pow(2.0, -date_diff('day', max(t.t_dat), {cutoff_sql}) / 14.0)
                    * (1.0 + ln(count(*))) AS source_score
            FROM transactions t
            SEMI JOIN m1_users USING (customer_id)
            WHERE t.t_dat >= {cutoff_sql} - INTERVAL {config.history_weeks} WEEK
              AND t.t_dat < {cutoff_sql}
            GROUP BY t.customer_id, t.article_id
        )
        SELECT
            customer_id,
            article_id,
            'repurchase' AS source,
            row_number() OVER (
                PARTITION BY customer_id
                ORDER BY source_score DESC, last_purchase DESC, article_id
            ) AS source_rank,
            source_score
        FROM scored
        QUALIFY source_rank <= {config.source_k}
        """
    )


def _create_recent_popularity(
    connection: duckdb.DuckDBPyConnection,
    cutoff_sql: str,
    config: M1Config,
) -> None:
    eligible = _eligible_catalog_sql()
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_item_popularity AS
        SELECT
            t.article_id,
            sum(pow(2.0, -date_diff('day', t.t_dat, {cutoff_sql}) / 7.0)) AS popularity_score
        FROM transactions t
        SEMI JOIN ({eligible}) eligible USING (article_id)
        WHERE t.t_dat >= {cutoff_sql} - INTERVAL {config.popularity_days} DAY
          AND t.t_dat < {cutoff_sql}
        GROUP BY t.article_id
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_global_top AS
        SELECT
            article_id,
            row_number() OVER (ORDER BY popularity_score DESC, article_id) AS source_rank,
            popularity_score AS source_score
        FROM m1_item_popularity
        QUALIFY source_rank <= {config.source_k}
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m1_recent_popularity AS
        SELECT
            users.customer_id,
            top.article_id,
            'recent_popularity' AS source,
            top.source_rank,
            top.source_score
        FROM m1_users users
        CROSS JOIN m1_global_top top
        """
    )


def _create_product_family(
    connection: duckdb.DuckDBPyConnection,
    cutoff_sql: str,
    config: M1Config,
) -> None:
    eligible = _eligible_catalog_sql()
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_product_family AS
        WITH user_products AS (
            SELECT
                t.customer_id,
                seed.product_code,
                max(t.t_dat) AS last_purchase,
                count(*) AS event_count,
                pow(2.0, -date_diff('day', max(t.t_dat), {cutoff_sql}) / 21.0)
                    * (1.0 + ln(count(*))) AS seed_score
            FROM transactions t
            SEMI JOIN m1_users USING (customer_id)
            JOIN articles seed USING (article_id)
            WHERE t.t_dat >= {cutoff_sql} - INTERVAL {config.history_weeks} WEEK
              AND t.t_dat < {cutoff_sql}
            GROUP BY t.customer_id, seed.product_code
        ), scored AS (
            SELECT
                up.customer_id,
                candidate.article_id,
                max(up.seed_score)
                    * (1.0 + ln(1.0 + coalesce(max(pop.popularity_score), 0.0))) AS source_score,
                max(up.last_purchase) AS last_purchase
            FROM user_products up
            JOIN articles candidate USING (product_code)
            SEMI JOIN ({eligible}) eligible ON candidate.article_id = eligible.article_id
            LEFT JOIN m1_item_popularity pop ON candidate.article_id = pop.article_id
            GROUP BY up.customer_id, candidate.article_id
        )
        SELECT
            customer_id,
            article_id,
            'product_family' AS source,
            row_number() OVER (
                PARTITION BY customer_id
                ORDER BY source_score DESC, last_purchase DESC, article_id
            ) AS source_rank,
            source_score
        FROM scored
        QUALIFY source_rank <= {config.source_k}
        """
    )


def _create_attribute_content(
    connection: duckdb.DuckDBPyConnection,
    cutoff_sql: str,
    config: M1Config,
) -> None:
    eligible = _eligible_catalog_sql()
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_content_catalog AS
        SELECT
            article.article_id,
            article.product_type_no,
            article.garment_group_no,
            article.perceived_colour_master_id,
            article.index_group_no,
            coalesce(pop.popularity_score, 0.0) AS popularity_score
        FROM articles article
        SEMI JOIN ({eligible}) eligible USING (article_id)
        LEFT JOIN m1_item_popularity pop USING (article_id)
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_content_type_pool AS
        SELECT *
        FROM m1_content_catalog
        QUALIFY row_number() OVER (
            PARTITION BY product_type_no
            ORDER BY popularity_score DESC, article_id
        ) <= {config.content_type_pool_k}
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_content_garment_pool AS
        SELECT *
        FROM m1_content_catalog
        QUALIFY row_number() OVER (
            PARTITION BY garment_group_no
            ORDER BY popularity_score DESC, article_id
        ) <= {config.content_garment_pool_k}
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_content_user_seeds AS
        WITH scored AS (
            SELECT
                t.customer_id,
                t.article_id,
                max(t.t_dat) AS last_purchase,
                pow(2.0, -date_diff('day', max(t.t_dat), {cutoff_sql}) / 21.0)
                    * (1.0 + ln(count(*))) AS seed_score
            FROM transactions t
            SEMI JOIN m1_users USING (customer_id)
            WHERE t.t_dat >= {cutoff_sql} - INTERVAL {config.history_weeks} WEEK
              AND t.t_dat < {cutoff_sql}
            GROUP BY t.customer_id, t.article_id
        )
        SELECT scored.*, article.product_type_no, article.garment_group_no,
               article.perceived_colour_master_id, article.index_group_no
        FROM scored
        JOIN articles article USING (article_id)
        QUALIFY row_number() OVER (
            PARTITION BY customer_id
            ORDER BY seed_score DESC, last_purchase DESC, article_id
        ) <= {config.content_seed_k}
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_attribute_content AS
        WITH matches AS (
            SELECT
                seed.customer_id,
                candidate.article_id,
                seed.seed_score
                    * (3.0
                       + CASE WHEN seed.garment_group_no = candidate.garment_group_no THEN 1.0 ELSE 0.0 END
                       + CASE WHEN seed.perceived_colour_master_id = candidate.perceived_colour_master_id THEN 0.5 ELSE 0.0 END
                       + CASE WHEN seed.index_group_no = candidate.index_group_no THEN 0.25 ELSE 0.0 END)
                    * (1.0 + ln(1.0 + candidate.popularity_score)) AS match_score
            FROM m1_content_user_seeds seed
            JOIN m1_content_type_pool candidate USING (product_type_no)

            UNION ALL

            SELECT
                seed.customer_id,
                candidate.article_id,
                seed.seed_score
                    * (1.5
                       + CASE WHEN seed.product_type_no = candidate.product_type_no THEN 1.0 ELSE 0.0 END
                       + CASE WHEN seed.perceived_colour_master_id = candidate.perceived_colour_master_id THEN 0.5 ELSE 0.0 END
                       + CASE WHEN seed.index_group_no = candidate.index_group_no THEN 0.25 ELSE 0.0 END)
                    * (1.0 + ln(1.0 + candidate.popularity_score)) AS match_score
            FROM m1_content_user_seeds seed
            JOIN m1_content_garment_pool candidate USING (garment_group_no)
        ), scored AS (
            SELECT customer_id, article_id, sum(match_score) AS source_score
            FROM matches
            GROUP BY customer_id, article_id
        )
        SELECT
            customer_id,
            article_id,
            'attribute_content' AS source,
            row_number() OVER (
                PARTITION BY customer_id
                ORDER BY source_score DESC, article_id
            ) AS source_rank,
            source_score
        FROM scored
        QUALIFY source_rank <= {config.source_k}
        """
    )


def _create_age_popularity(
    connection: duckdb.DuckDBPyConnection,
    cutoff_sql: str,
    config: M1Config,
) -> None:
    age_bucket = """
        CASE
            WHEN try_cast(age AS INTEGER) IS NULL THEN 'unknown'
            WHEN try_cast(age AS INTEGER) < 20 THEN '<20'
            WHEN try_cast(age AS INTEGER) < 30 THEN '20-29'
            WHEN try_cast(age AS INTEGER) < 40 THEN '30-39'
            WHEN try_cast(age AS INTEGER) < 50 THEN '40-49'
            WHEN try_cast(age AS INTEGER) < 60 THEN '50-59'
            ELSE '60+'
        END
    """
    eligible = _eligible_catalog_sql()
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_user_age AS
        SELECT users.customer_id, {age_bucket} AS age_bucket
        FROM m1_users users
        LEFT JOIN customers USING (customer_id)
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_age_top AS
        WITH scored AS (
            SELECT
                {age_bucket} AS age_bucket,
                t.article_id,
                sum(pow(2.0, -date_diff('day', t.t_dat, {cutoff_sql}) / 7.0)) AS source_score
            FROM transactions t
            JOIN customers USING (customer_id)
            SEMI JOIN ({eligible}) eligible USING (article_id)
            WHERE t.t_dat >= {cutoff_sql} - INTERVAL {config.popularity_days} DAY
              AND t.t_dat < {cutoff_sql}
            GROUP BY age_bucket, t.article_id
        )
        SELECT
            age_bucket,
            article_id,
            row_number() OVER (
                PARTITION BY age_bucket ORDER BY source_score DESC, article_id
            ) AS source_rank,
            source_score
        FROM scored
        QUALIFY source_rank <= {config.source_k}
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m1_age_popularity AS
        SELECT
            users.customer_id,
            top.article_id,
            'age_popularity' AS source,
            top.source_rank,
            top.source_score
        FROM m1_user_age users
        JOIN m1_age_top top USING (age_bucket)
        """
    )


def _create_covisit(
    connection: duckdb.DuckDBPyConnection,
    cutoff_sql: str,
    config: M1Config,
) -> dict[str, int]:
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_valid_user_days AS
        SELECT customer_id, t_dat, count(DISTINCT article_id) AS distinct_items
        FROM transactions
        WHERE t_dat >= {cutoff_sql} - INTERVAL {config.covisit_days} DAY
          AND t_dat < {cutoff_sql}
        GROUP BY customer_id, t_dat
        HAVING distinct_items BETWEEN 2 AND {config.max_user_day_items}
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_user_day_items AS
        SELECT DISTINCT t.customer_id, t.t_dat, t.article_id
        FROM transactions t
        SEMI JOIN m1_valid_user_days d
          ON t.customer_id = d.customer_id AND t.t_dat = d.t_dat
        WHERE t.t_dat >= {cutoff_sql} - INTERVAL {config.covisit_days} DAY
          AND t.t_dat < {cutoff_sql}
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m1_item_day_support AS
        SELECT article_id, count(*) AS item_days
        FROM m1_user_day_items
        GROUP BY article_id
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_covisit_pairs AS
        SELECT
            left_item.article_id AS left_article_id,
            right_item.article_id AS right_article_id,
            count(*) AS pair_days
        FROM m1_user_day_items left_item
        JOIN m1_user_day_items right_item
          ON left_item.customer_id = right_item.customer_id
         AND left_item.t_dat = right_item.t_dat
         AND left_item.article_id < right_item.article_id
        GROUP BY left_item.article_id, right_item.article_id
        HAVING pair_days >= {config.min_covisit_count}
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_covisit_neighbors AS
        WITH directed AS (
            SELECT left_article_id AS seed_article_id,
                   right_article_id AS candidate_article_id, pair_days
            FROM m1_covisit_pairs
            UNION ALL
            SELECT right_article_id, left_article_id, pair_days
            FROM m1_covisit_pairs
        ), scored AS (
            SELECT
                directed.seed_article_id,
                directed.candidate_article_id,
                directed.pair_days,
                directed.pair_days
                    / sqrt(seed.item_days * candidate.item_days) AS similarity
            FROM directed
            JOIN m1_item_day_support seed
              ON directed.seed_article_id = seed.article_id
            JOIN m1_item_day_support candidate
              ON directed.candidate_article_id = candidate.article_id
        )
        SELECT
            seed_article_id,
            candidate_article_id,
            pair_days,
            similarity,
            row_number() OVER (
                PARTITION BY seed_article_id
                ORDER BY similarity DESC, pair_days DESC, candidate_article_id
            ) AS neighbor_rank
        FROM scored
        QUALIFY neighbor_rank <= {config.covisit_neighbor_k}
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_user_day_covisit AS
        WITH user_seeds AS (
            SELECT
                t.customer_id,
                t.article_id AS seed_article_id,
                max(t.t_dat) AS last_purchase,
                count(*) AS event_count,
                pow(2.0, -date_diff('day', max(t.t_dat), {cutoff_sql}) / 14.0)
                    * (1.0 + ln(count(*))) AS seed_score
            FROM transactions t
            SEMI JOIN m1_users USING (customer_id)
            WHERE t.t_dat >= {cutoff_sql} - INTERVAL {config.history_weeks} WEEK
              AND t.t_dat < {cutoff_sql}
            GROUP BY t.customer_id, t.article_id
        ), scored AS (
            SELECT
                seeds.customer_id,
                neighbors.candidate_article_id AS article_id,
                sum(seeds.seed_score * neighbors.similarity) AS source_score,
                max(neighbors.pair_days) AS max_pair_days
            FROM user_seeds seeds
            JOIN m1_covisit_neighbors neighbors USING (seed_article_id)
            GROUP BY seeds.customer_id, neighbors.candidate_article_id
        )
        SELECT
            customer_id,
            article_id,
            'user_day_covisit' AS source,
            row_number() OVER (
                PARTITION BY customer_id
                ORDER BY source_score DESC, max_pair_days DESC, article_id
            ) AS source_rank,
            source_score
        FROM scored
        QUALIFY source_rank <= {config.source_k}
        """
    )
    all_multi_days = int(
        _scalar(
            connection,
            f"""
            SELECT count(*) FROM (
                SELECT customer_id, t_dat
                FROM transactions
                WHERE t_dat >= {cutoff_sql} - INTERVAL {config.covisit_days} DAY
                  AND t_dat < {cutoff_sql}
                GROUP BY customer_id, t_dat
                HAVING count(DISTINCT article_id) >= 2
            )
            """,
        )
    )
    retained_days = int(_scalar(connection, "SELECT count(*) FROM m1_valid_user_days"))
    return {
        "all_multi_item_user_days": all_multi_days,
        "retained_user_days": retained_days,
        "excluded_large_user_days": all_multi_days - retained_days,
        "undirected_pairs": int(_scalar(connection, "SELECT count(*) FROM m1_covisit_pairs")),
        "directed_neighbors": int(_scalar(connection, "SELECT count(*) FROM m1_covisit_neighbors")),
    }


def _source_order_sql() -> str:
    cases = " ".join(
        f"WHEN '{source}' THEN {index}" for index, source in enumerate(SOURCES, 1)
    )
    return f"CASE source {cases} END"


def _profile_weight_sql(profile: str) -> str:
    weights = FUSION_PROFILES[profile]
    cases = " ".join(
        f"WHEN '{source}' THEN {weights[source]:.6f}" for source in SOURCES
    )
    return f"CASE source {cases} END"


def _create_fused_candidates(
    connection: duckdb.DuckDBPyConnection, config: M1Config
) -> None:
    tables = {
        "repurchase": "m1_repurchase",
        "recent_popularity": "m1_recent_popularity",
        "product_family": "m1_product_family",
        "user_day_covisit": "m1_user_day_covisit",
        "age_popularity": "m1_age_popularity",
        "attribute_content": "m1_attribute_content",
    }
    union_sql = " UNION ALL ".join(f"SELECT * FROM {table}" for table in tables.values())
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_candidates_long AS
        SELECT candidates.*
        FROM ({union_sql}) candidates
        SEMI JOIN m1_eligible_catalog eligible USING (article_id)
        """
    )
    profiles = (
        tuple(FUSION_PROFILES)
        if config.evaluation_mode == "diagnostic"
        else (config.fusion_profile,)
    )
    for profile in profiles:
        weight_sql = _profile_weight_sql(profile)
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE m1_candidates_rrf_{profile} AS
            WITH fused AS (
                SELECT
                    customer_id,
                    article_id,
                    sum(({weight_sql}) / ({config.rrf_constant} + source_rank)) AS fused_score,
                    count(DISTINCT source) AS source_count,
                    string_agg(source, ',' ORDER BY source) AS sources
                FROM m1_candidates_long
                GROUP BY customer_id, article_id
            )
            SELECT
                customer_id,
                article_id,
                row_number() OVER (
                    PARTITION BY customer_id
                    ORDER BY fused_score DESC, source_count DESC, article_id
                ) AS candidate_rank,
                fused_score,
                source_count,
                sources
            FROM fused
            QUALIFY candidate_rank <= {config.candidate_k}
            """
        )
    if config.evaluation_mode == "diagnostic":
        source_order = _source_order_sql()
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE m1_candidates_round_robin AS
            WITH prioritized AS (
                SELECT
                    customer_id,
                    article_id,
                    source,
                    source_rank,
                    (source_rank - 1) * {len(SOURCES)} + ({source_order}) AS fusion_priority
                FROM m1_candidates_long
            ), deduplicated AS (
                SELECT
                    customer_id,
                    article_id,
                    min(fusion_priority) AS fusion_priority,
                    count(DISTINCT source) AS source_count,
                    string_agg(source, ',' ORDER BY source) AS sources
                FROM prioritized
                GROUP BY customer_id, article_id
            )
            SELECT
                customer_id,
                article_id,
                row_number() OVER (
                    PARTITION BY customer_id
                    ORDER BY fusion_priority, source_count DESC, article_id
                ) AS candidate_rank,
                fusion_priority,
                source_count,
                sources
            FROM deduplicated
            QUALIFY candidate_rank <= {config.candidate_k}
            """
        )
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW m1_candidates AS "
        f"SELECT * FROM m1_candidates_rrf_{config.fusion_profile}"
    )


def _metrics_for_relation(
    connection: duckdb.DuckDBPyConnection,
    relation_sql: str,
    rank_column: str,
    k: int,
    truth_table: str = "m1_truth",
) -> dict[str, float]:
    truth_pairs = int(_scalar(connection, f"SELECT count(*) FROM {truth_table}"))
    if truth_pairs == 0:
        raise ValueError(f"cannot evaluate an empty truth relation: {truth_table}")
    row = connection.execute(
        f"""
        WITH candidates AS (
            SELECT DISTINCT customer_id, article_id
            FROM ({relation_sql}) relation
            WHERE {rank_column} <= {k}
        ), truth_counts AS (
            SELECT customer_id, count(*) AS truth_count
            FROM {truth_table} GROUP BY customer_id
        ), hits AS (
            SELECT truth.customer_id, count(*) AS hit_count
            FROM {truth_table} truth
            SEMI JOIN candidates USING (customer_id, article_id)
            GROUP BY truth.customer_id
        ), per_user AS (
            SELECT
                truth_counts.customer_id,
                truth_counts.truth_count,
                coalesce(hits.hit_count, 0) AS hit_count
            FROM truth_counts
            LEFT JOIN hits USING (customer_id)
        )
        SELECT
            avg(hit_count::DOUBLE / truth_count),
            avg(CASE WHEN hit_count > 0 THEN 1.0 ELSE 0.0 END),
            avg(least(hit_count, 12)::DOUBLE / least(truth_count, 12))
        FROM per_user
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("metric query returned no rows")
    return {
        f"recall@{k}": float(row[0] or 0.0),
        f"hit_rate@{k}": float(row[1] or 0.0),
        "oracle_map@12": float(row[2] or 0.0),
    }


def _protocol_metrics(
    connection: duckdb.DuckDBPyConnection,
    relation_sql: str,
    rank_column: str,
    k: int,
    protocol: str,
) -> dict[str, dict[str, float] | None]:
    def evaluate(truth_table: str) -> dict[str, float] | None:
        truth_pairs = int(_scalar(connection, f"SELECT count(*) FROM {truth_table}"))
        if truth_pairs == 0:
            return None
        return _metrics_for_relation(
            connection,
            relation_sql,
            rank_column,
            k,
            truth_table=truth_table,
        )

    warm = evaluate("m1_truth_warm")
    if warm is None:
        raise ValueError("warm truth is empty; strict/warm evaluation is undefined")
    return {
        "overall": evaluate("m1_truth") if protocol == "optimistic_all_articles" else None,
        "warm": warm,
        "cold": evaluate("m1_truth_cold") if protocol == "optimistic_all_articles" else None,
    }


def _map_at_12(
    connection: duckdb.DuckDBPyConnection,
    relation: str = "m1_candidates",
    truth_table: str = "m1_truth",
) -> float:
    return float(
        _scalar(
            connection,
            """
            WITH truth_counts AS (
                SELECT customer_id, count(*) AS truth_count
                FROM {truth_table} GROUP BY customer_id
            ), predictions AS (
                SELECT
                    candidates.customer_id,
                    candidates.article_id,
                    candidates.candidate_rank,
                    CASE WHEN truth.article_id IS NULL THEN 0 ELSE 1 END AS is_hit
                FROM {relation} candidates
                LEFT JOIN {truth_table} truth USING (customer_id, article_id)
                WHERE candidates.candidate_rank <= 12
            ), running AS (
                SELECT *,
                    sum(is_hit) OVER (
                        PARTITION BY customer_id ORDER BY candidate_rank
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) AS cumulative_hits
                FROM predictions
            ), precision_sums AS (
                SELECT
                    customer_id,
                    sum(CASE WHEN is_hit = 1
                        THEN cumulative_hits::DOUBLE / candidate_rank ELSE 0.0 END) AS numerator
                FROM running GROUP BY customer_id
            )
            SELECT avg(coalesce(precision_sums.numerator, 0.0) / least(truth_counts.truth_count, 12))
            FROM truth_counts
            LEFT JOIN precision_sums USING (customer_id)
            """.format(relation=relation, truth_table=truth_table),
        )
        or 0.0
    )


def _source_metrics(
    connection: duckdb.DuckDBPyConnection, config: M1Config
) -> dict[str, dict[str, float]]:
    truth_table = (
        "m1_truth_warm" if config.catalog_protocol == "strict" else "m1_truth"
    )
    return {
        source: _metrics_for_relation(
            connection,
            f"SELECT * FROM m1_candidates_long WHERE source = '{source}'",
            "source_rank",
            config.source_k,
            truth_table=truth_table,
        )
        for source in SOURCES
    }


def _overlap_metrics(connection: duckdb.DuckDBPyConnection) -> dict[str, float]:
    output: dict[str, float] = {}
    for left, right in combinations(SOURCES, 2):
        value = _scalar(
            connection,
            f"""
            WITH left_counts AS (
                SELECT customer_id, count(DISTINCT article_id) AS n
                FROM m1_candidates_long WHERE source = '{left}' GROUP BY customer_id
            ), right_counts AS (
                SELECT customer_id, count(DISTINCT article_id) AS n
                FROM m1_candidates_long WHERE source = '{right}' GROUP BY customer_id
            ), intersections AS (
                SELECT l.customer_id, count(DISTINCT l.article_id) AS n
                FROM m1_candidates_long l
                JOIN m1_candidates_long r USING (customer_id, article_id)
                WHERE l.source = '{left}' AND r.source = '{right}'
                GROUP BY l.customer_id
            )
            SELECT avg(
                coalesce(intersections.n, 0)::DOUBLE /
                nullif(coalesce(left_counts.n, 0) + coalesce(right_counts.n, 0)
                       - coalesce(intersections.n, 0), 0)
            )
            FROM m1_users users
            LEFT JOIN left_counts USING (customer_id)
            LEFT JOIN right_counts USING (customer_id)
            LEFT JOIN intersections USING (customer_id)
            """,
        )
        output[f"{left}__{right}"] = float(value or 0.0)
    return output


def _ablation_metrics(
    connection: duckdb.DuckDBPyConnection, config: M1Config
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    truth_table = (
        "m1_truth_warm" if config.catalog_protocol == "strict" else "m1_truth"
    )
    weight_sql = _profile_weight_sql(config.fusion_profile)
    for excluded in SOURCES:
        relation = f"m1_without_{excluded}"
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {relation} AS
            WITH fused AS (
                SELECT
                    customer_id,
                    article_id,
                    sum(({weight_sql}) / ({config.rrf_constant} + source_rank)) AS fused_score,
                    count(DISTINCT source) AS source_count
                FROM m1_candidates_long
                WHERE source != '{excluded}'
                GROUP BY customer_id, article_id
            )
            SELECT
                customer_id,
                article_id,
                row_number() OVER (
                    PARTITION BY customer_id
                    ORDER BY fused_score DESC, source_count DESC, article_id
                ) AS candidate_rank
            FROM fused
            """
        )
        results[f"without_{excluded}"] = _metrics_for_relation(
            connection,
            f"SELECT * FROM {relation}",
            "candidate_rank",
            config.candidate_k,
            truth_table=truth_table,
        )
    return results


def _cumulative_union_metrics(
    connection: duckdb.DuckDBPyConnection,
    config: M1Config,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    included: list[str] = []
    for source in SOURCES:
        included.append(source)
        quoted = ", ".join(f"'{name}'" for name in included)
        raw_metrics = _metrics_for_relation(
            connection,
            f"""
            SELECT customer_id, article_id, 1 AS union_rank
            FROM m1_candidates_long
            WHERE source IN ({quoted})
            """,
            "union_rank",
            1,
            truth_table=(
                "m1_truth_warm"
                if config.catalog_protocol == "strict"
                else "m1_truth"
            ),
        )
        results["+".join(included)] = {
            "unbounded_recall": raw_metrics["recall@1"],
            "unbounded_hit_rate": raw_metrics["hit_rate@1"],
            "unbounded_oracle_map@12": raw_metrics["oracle_map@12"],
        }
    return results


def _segment_metrics(
    connection: duckdb.DuckDBPyConnection, config: M1Config
) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {
        "cold_warm": {},
        "activity_12w": {},
        "item_popularity": {},
    }

    def evaluate(group: str, segment: str, query: str) -> None:
        view = f"m1_truth_segment_{group}_{len(output[group])}"
        protocol_filter = (
            "SELECT segment_truth.* FROM ("
            + query
            + ") segment_truth SEMI JOIN m1_warm_catalog USING (article_id)"
            if config.catalog_protocol == "strict"
            else query
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {view} AS
            {protocol_filter}
            """
        )
        user_count = int(
            _scalar(connection, f"SELECT count(DISTINCT customer_id) FROM {view}")
        )
        pair_count = int(_scalar(connection, f"SELECT count(*) FROM {view}"))
        if user_count == 0:
            output[group][segment] = {"users": 0, "truth_pairs": 0}
            return
        metrics = _metrics_for_relation(
            connection,
            "SELECT * FROM m1_candidates",
            "candidate_rank",
            config.candidate_k,
            truth_table=view,
        )
        output[group][segment] = {
            "users": user_count,
            "truth_pairs": pair_count,
            **metrics,
        }

    for segment in ("cold", "warm"):
        evaluate(
            "cold_warm",
            segment,
            f"""
            SELECT truth.* FROM m1_truth truth
            JOIN m1_user_segments segments USING (customer_id)
            WHERE segments.user_segment = '{segment}'
            """,
        )
    for segment in ("inactive_12w", "low_1_5", "medium_6_20", "high_21_plus"):
        evaluate(
            "activity_12w",
            segment,
            f"""
            SELECT truth.* FROM m1_truth truth
            JOIN m1_user_segments segments USING (customer_id)
            WHERE segments.activity_segment = '{segment}'
            """,
        )
    for segment in (
        "head_top10pct",
        "mid_next40pct",
        "tail_bottom50pct",
        "unseen_before_cutoff",
    ):
        evaluate(
            "item_popularity",
            segment,
            f"""
            SELECT customer_id, article_id
            FROM m1_truth_item_segments
            WHERE item_segment = '{segment}'
            """,
        )
    return output


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _render_report(result: dict[str, Any]) -> str:
    config = result["config"]
    lines = [
        "# M1 Multi-source Retrieval Report",
        "",
        "## Scope",
        "",
        f"- cutoff: `{result['cutoff']}`; validation: final 7 days",
        f"- sampled validation users: {result['population']['sampled_validation_users']:,} / {result['population']['all_validation_users']:,}",
        f"- sample rate: {config['sample_rate']:.4f}; global source statistics still use full pre-cutoff history",
        f"- candidate budget: {config['candidate_k']}; per-source budget: {config['source_k']}",
        f"- primary fusion profile: `{config['fusion_profile']}`",
        f"- evaluation mode: `{config['evaluation_mode']}`",
        f"- catalog protocol: `{config['catalog_protocol']}`",
        f"- eligible catalog size: {result['eligible_catalog_size']:,}",
        f"- truth pairs overall / warm / cold: {result['truth_pairs']['overall']:,} / {result['truth_pairs']['warm']:,} / {result['truth_pairs']['cold']:,}",
        f"- catalog-unreachable truth pairs: {result['catalog_unreachable_truth_pairs']:,}",
        f"- customer policy: `{config['customer_policy']}`",
        f"- duplicate policy: `{config['duplicate_policy']}`",
        "",
        (
            "This is a full-user validation run. Interaction aggregates never use validation-week events."
            if config["sample_rate"] == 1.0
            else "This run is a sampled-user M1 experiment, not a full validation claim. Interaction aggregates never use validation-week events."
        ),
        "",
        f"## Final `{config['fusion_profile']}` RRF pool",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in result["final_metrics"].items():
        lines.append(f"| {name} | {_fmt(value)} |")
    metric_k = config["candidate_k"]
    lines.extend(
        [
            "",
            "## Protocol truth-bucket metrics",
            "",
            "| Truth bucket | Recall | HitRate | Oracle MAP@12 |",
            "|---|---:|---:|---:|",
        ]
    )
    for bucket in ("overall", "warm", "cold"):
        metrics = result["metrics"][bucket]
        if metrics is None:
            lines.append(f"| {bucket} | NA | NA | NA |")
        else:
            lines.append(
                f"| {bucket} | {_fmt(metrics[f'recall@{metric_k}'])} | "
                f"{_fmt(metrics[f'hit_rate@{metric_k}'])} | "
                f"{_fmt(metrics['oracle_map@12'])} |"
            )
    lines.extend(
        [
            "",
            "## Fixed-budget fusion comparison",
            "",
            "| Fusion | Recall | HitRate | Oracle MAP@12 | MAP@12 order |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    candidate_k = config["candidate_k"]
    for fusion, metrics in result["fusion_comparison"].items():
        lines.append(
            f"| {fusion} | {_fmt(metrics[f'recall@{candidate_k}'])} | "
            f"{_fmt(metrics[f'hit_rate@{candidate_k}'])} | "
            f"{_fmt(metrics['oracle_map@12'])} | {_fmt(metrics['map@12_order'])} |"
        )
    lines.extend(["", "## Standalone sources", "", "| Source | Recall | HitRate | Oracle MAP@12 |", "|---|---:|---:|---:|"])
    k = config["source_k"]
    for source in SOURCES:
        metrics = result["source_metrics"][source]
        lines.append(
            f"| {source} | {_fmt(metrics[f'recall@{k}'])} | "
            f"{_fmt(metrics[f'hit_rate@{k}'])} | {_fmt(metrics['oracle_map@12'])} |"
        )
    if result["leave_one_out"]:
        lines.extend(["", "## Leave-one-source-out at fixed candidate budget", "", "| Excluded | Recall | HitRate | Oracle MAP@12 |", "|---|---:|---:|---:|"])
        for name, metrics in result["leave_one_out"].items():
            lines.append(
                f"| {name.removeprefix('without_')} | {_fmt(metrics[f'recall@{candidate_k}'])} | "
                f"{_fmt(metrics[f'hit_rate@{candidate_k}'])} | {_fmt(metrics['oracle_map@12'])} |"
            )
    if result["cumulative_unbounded_union"]:
        lines.extend(
            [
                "",
                "## Cumulative source union before the fixed budget",
                "",
                "| Included sources | Unbounded recall | Unbounded hit rate |",
                "|---|---:|---:|",
            ]
        )
        for name, metrics in result["cumulative_unbounded_union"].items():
            lines.append(
                f"| {name} | {_fmt(metrics['unbounded_recall'])} | "
                f"{_fmt(metrics['unbounded_hit_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Subgroup diagnostics",
            "",
        ]
    )
    for group, segments in result["segments"].items():
        lines.extend(
            [
                "",
                f"### {group}",
                "",
                "| Segment | Users | Truth pairs | Recall | HitRate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for segment, metrics in segments.items():
            if metrics["users"] == 0:
                lines.append(f"| {segment} | 0 | 0 | NA | NA |")
            else:
                lines.append(
                    f"| {segment} | {metrics['users']:,} | {metrics['truth_pairs']:,} | "
                    f"{_fmt(metrics[f'recall@{candidate_k}'])} | "
                    f"{_fmt(metrics[f'hit_rate@{candidate_k}'])} |"
                )
    lines.extend(
        [
            "",
            f"- fixed-budget candidate rows: {result['candidate_rows']:,}",
            f"- elapsed seconds: {result['elapsed_seconds']:.2f}",
        ]
    )
    lines.extend(
        [
            "",
            "## Special treatments and reasons",
            "",
            (
                "- All articles are eligible under the optimistic final-week protocol. Only catalog membership uses full articles; behavior statistics remain strictly pre-cutoff."
                if config["catalog_protocol"] == "optimistic_all_articles"
                else "- Eligible catalog is restricted to articles observed before the cutoff for this earlier rolling window."
            ),
            "- Only age is used for customer segmentation. Snapshot-unsafe membership/news fields are excluded from primary M1.",
            "- Exact duplicate transactions are retained in event-frequency and popularity calculations.",
            f"- Co-visitation uses distinct items per user-day and excludes baskets above {config['max_user_day_items']} items to prevent duplicate multiplication and quadratic pair explosion. t_dat is not treated as a real session.",
            "- Co-visitation is cosine-normalized by item-day support to reduce pure popularity dominance.",
            f"- Attribute content uses at most {config['content_seed_k']} recent user seeds and top {config['content_type_pool_k']}/{config['content_garment_pool_k']} candidates per product type/garment group to bound active-user cost.",
            (
                f"- The primary weighted RRF profile is `{config['fusion_profile']}`; all predefined profiles are evaluated in this diagnostic run."
                if config["evaluation_mode"] == "diagnostic"
                else f"- Final mode evaluates only the cross-window-selected `{config['fusion_profile']}` profile; redundant profile, round-robin, overlap, and leave-one-out sorts are skipped."
            ),
            "",
            "## Co-visitation build",
            "",
        ]
    )
    for name, value in result["covisit_build"].items():
        lines.append(f"- {name}: {value:,}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Candidate Recall/HitRate measure retrieval coverage. Oracle MAP@12 is the ranking ceiling within the candidate pool. MAP@12 order uses an untuned heuristic order and is not an M2 ranking result.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_m1(
    raw_dir: Path,
    work_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M1Config,
) -> dict[str, Any]:
    config.validate()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    connection = prepare_tabular_connection(raw_dir, work_dir)
    try:
        cutoff = _resolve_cutoff(connection, config.cutoff)
        cutoff_sql = _date_literal(cutoff)
        population = _create_population(connection, cutoff, config)
        if population["sampled_validation_users"] == 0:
            raise ValueError("sample produced zero validation users")
        _create_repurchase(connection, cutoff_sql, config)
        _create_recent_popularity(connection, cutoff_sql, config)
        _create_product_family(connection, cutoff_sql, config)
        _create_attribute_content(connection, cutoff_sql, config)
        covisit_build = _create_covisit(connection, cutoff_sql, config)
        _create_age_popularity(connection, cutoff_sql, config)
        _create_fused_candidates(connection, config)

        protocol_metrics = _protocol_metrics(
            connection,
            "SELECT * FROM m1_candidates",
            "candidate_rank",
            config.candidate_k,
            config.catalog_protocol,
        )
        primary_truth_table = (
            "m1_truth_warm"
            if config.catalog_protocol == "strict"
            else "m1_truth"
        )
        primary_bucket = (
            "warm" if config.catalog_protocol == "strict" else "overall"
        )
        final_metrics = dict(protocol_metrics[primary_bucket] or {})
        final_metrics[f"map@12_untuned_{config.fusion_profile}_rrf"] = _map_at_12(
            connection, truth_table=primary_truth_table
        )
        profile_metrics: dict[str, dict[str, float]] = {}
        profile_protocol_metrics: dict[
            str, dict[str, dict[str, float] | None]
        ] = {}
        evaluated_profiles = (
            tuple(FUSION_PROFILES)
            if config.evaluation_mode == "diagnostic"
            else (config.fusion_profile,)
        )
        for profile in evaluated_profiles:
            table = f"m1_candidates_rrf_{profile}"
            metrics = _metrics_for_relation(
                connection,
                f"SELECT * FROM {table}",
                "candidate_rank",
                config.candidate_k,
                truth_table=primary_truth_table,
            )
            metrics["map@12_order"] = _map_at_12(
                connection, table, truth_table=primary_truth_table
            )
            profile_metrics[f"rrf_{profile}"] = metrics
            profile_protocol_metrics[f"rrf_{profile}"] = _protocol_metrics(
                connection,
                f"SELECT * FROM {table}",
                "candidate_rank",
                config.candidate_k,
                config.catalog_protocol,
            )
        if config.evaluation_mode == "diagnostic":
            round_robin_metrics = _metrics_for_relation(
                connection,
                "SELECT * FROM m1_candidates_round_robin",
                "candidate_rank",
                config.candidate_k,
                truth_table=primary_truth_table,
            )
            round_robin_metrics["map@12_order"] = _map_at_12(
                connection,
                "m1_candidates_round_robin",
                truth_table=primary_truth_table,
            )
            profile_metrics["equal_allocation_round_robin"] = round_robin_metrics
            profile_protocol_metrics[
                "equal_allocation_round_robin"
            ] = _protocol_metrics(
                connection,
                "SELECT * FROM m1_candidates_round_robin",
                "candidate_rank",
                config.candidate_k,
                config.catalog_protocol,
            )
        source_catalog_audit = {
            source: {
                "candidate_pairs": int(
                    _scalar(
                        connection,
                        f"SELECT count(*) FROM m1_candidates_long WHERE source = '{source}'",
                    )
                ),
                "cold_candidate_pairs": int(
                    _scalar(
                        connection,
                        f"""
                        SELECT count(*)
                        FROM m1_candidates_long candidates
                        ANTI JOIN m1_warm_catalog warm USING (article_id)
                        WHERE source = '{source}'
                        """,
                    )
                ),
            }
            for source in SOURCES
        }
        truth_pairs = {
            "overall": population["sampled_truth_pairs"],
            "warm": population["warm_truth_pairs"],
            "cold": population["cold_truth_pairs"],
        }
        catalog_unreachable_truth_pairs = (
            truth_pairs["cold"] if config.catalog_protocol == "strict" else 0
        )
        result: dict[str, Any] = {
            "stage": "M1",
            "status": "measured",
            "catalog_protocol": config.catalog_protocol,
            "cutoff": cutoff,
            "eligible_catalog_size": population["eligible_catalog_size"],
            "validation_users": population["sampled_validation_users"],
            "truth_pairs": truth_pairs,
            "catalog_unreachable_truth_pairs": catalog_unreachable_truth_pairs,
            "metrics": protocol_metrics,
            "config": asdict(config),
            "population": population,
            "covisit_build": covisit_build,
            "final_metrics": final_metrics,
            "fusion_comparison": profile_metrics,
            "fusion_protocol_metrics": profile_protocol_metrics,
            "source_metrics": _source_metrics(connection, config),
            "cumulative_unbounded_union": (
                _cumulative_union_metrics(connection, config)
                if config.evaluation_mode == "diagnostic"
                else {}
            ),
            "pairwise_mean_jaccard": (
                _overlap_metrics(connection)
                if config.evaluation_mode == "diagnostic"
                else {}
            ),
            "leave_one_out": (
                _ablation_metrics(connection, config)
                if config.evaluation_mode == "diagnostic"
                else {}
            ),
            "segments": _segment_metrics(connection, config),
            "leakage_audit": {
                "warm_cold_definition": "transactions.t_dat < cutoff only",
                "eligible_catalog_definition": (
                    "distinct articles.article_id (full metadata assumption)"
                    if config.catalog_protocol == "optimistic_all_articles"
                    else "distinct transactions.article_id where t_dat < cutoff"
                ),
                "validation_used_for_catalog": False,
                "behavior_statistics_filter": "t_dat < cutoff",
                "latest_behavior_date": str(
                    _scalar(
                        connection,
                        f"SELECT max(t_dat) FROM transactions WHERE t_dat < {cutoff_sql}",
                    )
                ),
                "source_catalog_audit": source_catalog_audit,
            },
            "candidate_rows": int(_scalar(connection, "SELECT count(*) FROM m1_candidates")),
        }
        result["elapsed_seconds"] = time.perf_counter() - started
        candidate_path = artifact_dir / "candidates.parquet"
        connection.execute(
            f"COPY m1_candidates TO {_path_literal(candidate_path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        result["candidate_artifact"] = str(candidate_path.resolve())
    finally:
        connection.close()

    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "M1_REPORT.md").write_text(_render_report(result), encoding="utf-8")
    return result


def evaluate_candidate_artifact(
    raw_dir: Path,
    work_dir: Path,
    output_dir: Path,
    candidate_path: Path,
    source_metrics_path: Path,
    cutoff: str,
    expected_profile: str,
    candidate_k: int = 100,
    expected_users: int | None = None,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if not source_metrics_path.is_file():
        raise FileNotFoundError(source_metrics_path)
    source_result = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    source_config = source_result.get("config", {})
    source_protocol = source_config.get(
        "catalog_protocol", source_config.get("catalog_policy")
    )
    expected_source_values = {
        "cutoff": (source_result.get("cutoff"), cutoff),
        "catalog_protocol": (source_protocol, "optimistic_all_articles"),
        "sample_rate": (source_config.get("sample_rate"), 1.0),
        "candidate_k": (source_config.get("candidate_k"), candidate_k),
        "fusion_profile": (source_config.get("fusion_profile"), expected_profile),
    }
    mismatches = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in expected_source_values.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"source metrics do not match requested artifact evaluation: {mismatches}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = M1Config(
        cutoff=cutoff,
        sample_rate=1.0,
        candidate_k=candidate_k,
        fusion_profile=expected_profile,
        evaluation_mode="final",
        catalog_protocol="optimistic_all_articles",
    )
    connection = prepare_tabular_connection(raw_dir, work_dir)
    try:
        population = _create_population(connection, cutoff, config)
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW m1_candidates AS "
            f"SELECT * FROM read_parquet({_path_literal(candidate_path)})"
        )
        schema = {
            row[0]: row[1]
            for row in connection.execute("DESCRIBE m1_candidates").fetchall()
        }
        required_columns = {
            "customer_id",
            "article_id",
            "candidate_rank",
            "fused_score",
            "source_count",
            "sources",
        }
        if not required_columns.issubset(schema):
            raise ValueError(
                f"candidate artifact schema is missing {sorted(required_columns - set(schema))}"
            )
        candidate_rows = int(_scalar(connection, "SELECT count(*) FROM m1_candidates"))
        candidate_users = int(
            _scalar(connection, "SELECT count(DISTINCT customer_id) FROM m1_candidates")
        )
        per_user = connection.execute(
            """
            SELECT min(n), max(n)
            FROM (
                SELECT customer_id, count(*) AS n
                FROM m1_candidates
                GROUP BY customer_id
            )
            """
        ).fetchone()
        min_per_user, max_per_user = (int(per_user[0]), int(per_user[1]))
        unique_pairs = int(
            _scalar(
                connection,
                "SELECT count(*) FROM (SELECT DISTINCT customer_id, article_id FROM m1_candidates)",
            )
        )
        rank_violations = int(
            _scalar(
                connection,
                f"SELECT count(*) FROM m1_candidates WHERE candidate_rank < 1 OR candidate_rank > {candidate_k}",
            )
        )
        artifact_only_users = int(
            _scalar(
                connection,
                "SELECT count(*) FROM (SELECT DISTINCT customer_id FROM m1_candidates ANTI JOIN m1_users USING (customer_id))",
            )
        )
        truth_only_users = int(
            _scalar(
                connection,
                "SELECT count(*) FROM (SELECT customer_id FROM m1_users ANTI JOIN (SELECT DISTINCT customer_id FROM m1_candidates) USING (customer_id))",
            )
        )
        checks = {
            "source_metrics_match": True,
            "candidate_users": candidate_users,
            "candidate_rows": candidate_rows,
            "min_candidates_per_user": min_per_user,
            "max_candidates_per_user": max_per_user,
            "unique_customer_article_pairs": unique_pairs,
            "rank_violations": rank_violations,
            "artifact_only_users": artifact_only_users,
            "truth_only_users": truth_only_users,
            "schema": schema,
        }
        failures: list[str] = []
        if expected_users is not None and candidate_users != expected_users:
            failures.append(f"candidate_users={candidate_users}, expected={expected_users}")
        if expected_rows is not None and candidate_rows != expected_rows:
            failures.append(f"candidate_rows={candidate_rows}, expected={expected_rows}")
        if min_per_user != candidate_k or max_per_user != candidate_k:
            failures.append(
                f"per-user candidate counts are {min_per_user}..{max_per_user}, expected {candidate_k}"
            )
        if unique_pairs != candidate_rows:
            failures.append("duplicate customer/article pairs found")
        if rank_violations:
            failures.append(f"rank_violations={rank_violations}")
        if artifact_only_users or truth_only_users:
            failures.append(
                f"candidate/truth user sets differ: artifact_only={artifact_only_users}, truth_only={truth_only_users}"
            )
        if failures:
            raise ValueError("candidate artifact validation failed: " + "; ".join(failures))

        metrics = _protocol_metrics(
            connection,
            "SELECT * FROM m1_candidates",
            "candidate_rank",
            candidate_k,
            "optimistic_all_articles",
        )
        truth_pairs = {
            "overall": population["sampled_truth_pairs"],
            "warm": population["warm_truth_pairs"],
            "cold": population["cold_truth_pairs"],
        }
        result: dict[str, Any] = {
            "stage": "M1",
            "status": "measured_reuse",
            "catalog_protocol": "optimistic_all_articles",
            "cutoff": cutoff,
            "eligible_catalog_size": population["eligible_catalog_size"],
            "validation_users": population["sampled_validation_users"],
            "truth_pairs": truth_pairs,
            "catalog_unreachable_truth_pairs": 0,
            "metrics": metrics,
            "ordering_diagnostic": {
                "map@12_untuned_rrf": _map_at_12(connection),
                "is_m2_ranking_result": False,
            },
            "population": population,
            "source_run_metrics": str(source_metrics_path.resolve()),
            "candidate_artifact": str(candidate_path.resolve()),
            "artifact_validation": checks,
            "assumption": "full articles.csv catalog is visible at prediction time; historical availability is not established",
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        connection.close()

    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# M1 Reused Candidate Artifact Evaluation",
        "",
        f"- source artifact: `{result['candidate_artifact']}`",
        f"- cutoff: `{cutoff}`",
        f"- profile: `{expected_profile}`",
        f"- validation users / candidate rows: {result['validation_users']:,} / {candidate_rows:,}",
        f"- truth pairs overall / warm / cold: {truth_pairs['overall']:,} / {truth_pairs['warm']:,} / {truth_pairs['cold']:,}",
        "- protocol: optimistic/all-articles; this assumes the complete metadata catalog was visible, not that every item was historically available.",
        "- artifact schema, source config, user set, pair uniqueness, ranks, and exactly 100 candidates per user were validated before evaluation.",
        "- ordered MAP@12 is an untuned RRF ordering diagnostic, not an M2 ranking result.",
    ]
    (output_dir / "M1_ARTIFACT_EVALUATION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result
