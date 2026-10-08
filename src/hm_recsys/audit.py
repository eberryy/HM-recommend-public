from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import duckdb

TABLE_FILES = {
    "transactions": "transactions_train.csv",
    "articles": "articles.csv",
    "customers": "customers.csv",
    "submission": "sample_submission.csv",
}


def _literal(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _one(connection: duckdb.DuckDBPyConnection, query: str) -> tuple[Any, ...]:
    row = connection.execute(query).fetchone()
    if row is None:
        raise RuntimeError("audit query returned no rows")
    return row


def _rows(connection: duckdb.DuckDBPyConnection, query: str) -> list[tuple[Any, ...]]:
    return connection.execute(query).fetchall()


def _columns(connection: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [row[0] for row in _rows(connection, f"DESCRIBE {table}")]


def _profile_columns(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    columns: Sequence[str],
    row_count: int,
) -> list[dict[str, Any]]:
    expressions: list[str] = []
    for column in columns:
        quoted = _ident(column)
        expressions.extend(
            [
                f"count(*) FILTER (WHERE {quoted} IS NULL OR "
                f"trim(cast({quoted} AS VARCHAR)) = '')",
                f"count(DISTINCT {quoted})",
            ]
        )
    values = _one(connection, f"SELECT {', '.join(expressions)} FROM {table}")
    output: list[dict[str, Any]] = []
    for index, column in enumerate(columns):
        missing = int(values[index * 2])
        output.append(
            {
                "column": column,
                "missing": missing,
                "missing_rate": missing / row_count if row_count else 0.0,
                "unique": int(values[index * 2 + 1]),
            }
        )
    return output


def _top_values(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
    limit: int = 8,
) -> list[tuple[str, int]]:
    quoted = _ident(column)
    return [
        ("<MISSING>" if value is None or str(value).strip() == "" else str(value), int(count))
        for value, count in _rows(
            connection,
            f"""
            SELECT {quoted}, count(*) AS n
            FROM {table}
            GROUP BY {quoted}
            ORDER BY n DESC, {quoted}
            LIMIT {limit}
            """,
        )
    ]


def _fmt_int(value: int | float | None) -> str:
    if value is None:
        return "NA"
    return f"{int(value):,}"


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None or math.isnan(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None or math.isnan(float(value)):
        return "NA"
    return f"{float(value) * 100:.{digits}f}%"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    divider = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def prepare_tabular_connection(
    raw_dir: Path, work_dir: Path
) -> duckdb.DuckDBPyConnection:
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = work_dir / "duckdb_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "hm_audit.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute("SET threads = 8")
    connection.execute("SET memory_limit = '12GB'")
    connection.execute(f"SET temp_directory = {_literal(temp_dir)}")

    raw_transactions = raw_dir / TABLE_FILES["transactions"]
    parquet = work_dir / "transactions.parquet"
    if not parquet.is_file() or parquet.stat().st_mtime < raw_transactions.stat().st_mtime:
        connection.execute(
            f"""
            COPY (
                SELECT
                    t_dat,
                    customer_id,
                    article_id,
                    price,
                    sales_channel_id
                FROM read_csv(
                    {_literal(raw_transactions)},
                    header = true,
                    columns = {{
                        't_dat': 'DATE',
                        'customer_id': 'VARCHAR',
                        'article_id': 'VARCHAR',
                        'price': 'DOUBLE',
                        'sales_channel_id': 'UTINYINT'
                    }}
                )
            ) TO {_literal(parquet)} (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 250000
            )
            """
        )

    connection.execute(
        f"CREATE OR REPLACE VIEW transactions AS SELECT * FROM read_parquet({_literal(parquet)})"
    )
    for table in ("articles", "customers", "submission"):
        path = raw_dir / TABLE_FILES[table]
        connection.execute(
            f"""
            CREATE OR REPLACE VIEW {table} AS
            SELECT * FROM read_csv_auto(
                {_literal(path)}, header = true, all_varchar = true
            )
            """
        )
    return connection


def collect_audit(raw_dir: Path, work_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    missing_files = [
        filename for filename in TABLE_FILES.values() if not (raw_dir / filename).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"missing required audit files: {missing_files}")

    connection = prepare_tabular_connection(raw_dir, work_dir)
    try:
        inventory: dict[str, Any] = {}
        for table, filename in TABLE_FILES.items():
            path = raw_dir / filename
            row_count = int(_one(connection, f"SELECT count(*) FROM {table}")[0])
            inventory[table] = {
                "filename": filename,
                "bytes": path.stat().st_size,
                "rows": row_count,
                "columns": _columns(connection, table),
            }

        article_columns = inventory["articles"]["columns"]
        customer_columns = inventory["customers"]["columns"]
        submission_columns = inventory["submission"]["columns"]
        profiles = {
            "articles": _profile_columns(
                connection,
                "articles",
                article_columns,
                inventory["articles"]["rows"],
            ),
            "customers": _profile_columns(
                connection,
                "customers",
                customer_columns,
                inventory["customers"]["rows"],
            ),
            "submission": _profile_columns(
                connection,
                "submission",
                submission_columns,
                inventory["submission"]["rows"],
            ),
        }

        tx_summary_row = _one(
            connection,
            """
            SELECT
                count(*), count(DISTINCT customer_id), count(DISTINCT article_id),
                min(t_dat), max(t_dat), min(price), max(price), avg(price),
                approx_quantile(price, [0.01, 0.25, 0.5, 0.75, 0.99]),
                count(*) FILTER (WHERE price IS NULL),
                count(*) FILTER (WHERE price <= 0),
                count(*) FILTER (WHERE customer_id IS NULL OR article_id IS NULL OR t_dat IS NULL)
            FROM transactions
            """,
        )
        tx_summary = {
            "rows": int(tx_summary_row[0]),
            "users": int(tx_summary_row[1]),
            "items": int(tx_summary_row[2]),
            "min_date": str(tx_summary_row[3]),
            "max_date": str(tx_summary_row[4]),
            "min_price": float(tx_summary_row[5]),
            "max_price": float(tx_summary_row[6]),
            "mean_price": float(tx_summary_row[7]),
            "price_quantiles": [float(value) for value in tx_summary_row[8]],
            "missing_price": int(tx_summary_row[9]),
            "non_positive_price": int(tx_summary_row[10]),
            "missing_core_fields": int(tx_summary_row[11]),
        }
        tx_summary["density"] = tx_summary["rows"] / (
            tx_summary["users"] * tx_summary["items"]
        )
        tx_summary["channels"] = [
            {"channel": int(channel), "rows": int(rows), "rate": int(rows) / tx_summary["rows"]}
            for channel, rows in _rows(
                connection,
                "SELECT sales_channel_id, count(*) FROM transactions GROUP BY 1 ORDER BY 1",
            )
        ]

        exact_duplicate_rows = int(
            _one(
                connection,
                """
                SELECT coalesce(sum(n - 1), 0)
                FROM (
                    SELECT count(*) AS n
                    FROM transactions
                    GROUP BY t_dat, customer_id, article_id, price, sales_channel_id
                    HAVING count(*) > 1
                )
                """,
            )[0]
        )
        event_key_duplicate_rows = int(
            _one(
                connection,
                """
                SELECT coalesce(sum(n - 1), 0)
                FROM (
                    SELECT count(*) AS n
                    FROM transactions
                    GROUP BY t_dat, customer_id, article_id
                    HAVING count(*) > 1
                )
                """,
            )[0]
        )

        user_activity = _one(
            connection,
            """
            WITH activity AS (
                SELECT customer_id, count(*) AS n, count(DISTINCT article_id) AS distinct_items
                FROM transactions GROUP BY customer_id
            )
            SELECT
                avg(n), approx_quantile(n, [0.5, 0.9, 0.95, 0.99]), max(n),
                avg(distinct_items),
                count(*) FILTER (WHERE n = 1),
                count(*)
            FROM activity
            """,
        )
        item_activity = _one(
            connection,
            """
            WITH activity AS (
                SELECT article_id, count(*) AS n, count(DISTINCT customer_id) AS distinct_users
                FROM transactions GROUP BY article_id
            )
            SELECT
                avg(n), approx_quantile(n, [0.5, 0.9, 0.95, 0.99]), max(n),
                avg(distinct_users),
                count(*) FILTER (WHERE n = 1),
                count(*)
            FROM activity
            """,
        )
        item_concentration = _one(
            connection,
            """
            WITH counts AS (
                SELECT article_id, count(*) AS n
                FROM transactions GROUP BY article_id
            ), ranked AS (
                SELECT n, row_number() OVER (ORDER BY n DESC) AS rank,
                       count(*) OVER () AS entity_count,
                       sum(n) OVER () AS total_events
                FROM counts
            )
            SELECT
                sum(n) FILTER (WHERE rank <= ceil(entity_count * 0.01)) / max(total_events),
                sum(n) FILTER (WHERE rank <= ceil(entity_count * 0.05)) / max(total_events),
                sum(n) FILTER (WHERE rank <= ceil(entity_count * 0.10)) / max(total_events)
            FROM ranked
            """,
        )
        user_concentration = _one(
            connection,
            """
            WITH counts AS (
                SELECT customer_id, count(*) AS n
                FROM transactions GROUP BY customer_id
            ), ranked AS (
                SELECT n, row_number() OVER (ORDER BY n DESC) AS rank,
                       count(*) OVER () AS entity_count,
                       sum(n) OVER () AS total_events
                FROM counts
            )
            SELECT
                sum(n) FILTER (WHERE rank <= ceil(entity_count * 0.01)) / max(total_events),
                sum(n) FILTER (WHERE rank <= ceil(entity_count * 0.05)) / max(total_events),
                sum(n) FILTER (WHERE rank <= ceil(entity_count * 0.10)) / max(total_events)
            FROM ranked
            """,
        )

        repeat_row = _one(
            connection,
            """
            WITH pairs AS (
                SELECT customer_id, article_id, count(*) AS n
                FROM transactions GROUP BY customer_id, article_id
            )
            SELECT
                count(*), count(*) FILTER (WHERE n > 1), sum(n - 1), max(n),
                count(DISTINCT customer_id) FILTER (WHERE n > 1)
            FROM pairs
            """,
        )
        basket_row = _one(
            connection,
            """
            WITH baskets AS (
                SELECT customer_id, t_dat, count(*) AS events,
                       count(DISTINCT article_id) AS distinct_items
                FROM transactions GROUP BY customer_id, t_dat
            )
            SELECT
                count(*), avg(distinct_items),
                approx_quantile(distinct_items, [0.5, 0.9, 0.95, 0.99]),
                max(distinct_items),
                count(*) FILTER (WHERE distinct_items > 1)
            FROM baskets
            """,
        )

        monthly = [
            {"month": str(month)[:7], "rows": int(rows), "users": int(users), "items": int(items)}
            for month, rows, users, items in _rows(
                connection,
                """
                SELECT date_trunc('month', t_dat) AS month, count(*) AS rows,
                       count(DISTINCT customer_id), count(DISTINCT article_id)
                FROM transactions GROUP BY month ORDER BY month
                """,
            )
        ]

        key_duplicates = {
            "articles_duplicate_article_id": int(
                _one(
                    connection,
                    "SELECT count(*) - count(DISTINCT article_id) FROM articles",
                )[0]
            ),
            "customers_duplicate_customer_id": int(
                _one(
                    connection,
                    "SELECT count(*) - count(DISTINCT customer_id) FROM customers",
                )[0]
            ),
            "submission_duplicate_customer_id": int(
                _one(
                    connection,
                    "SELECT count(*) - count(DISTINCT customer_id) FROM submission",
                )[0]
            ),
            "transactions_exact_duplicate_extra_rows": exact_duplicate_rows,
            "transactions_same_user_item_day_extra_rows": event_key_duplicate_rows,
        }

        age_row = _one(
            connection,
            """
            WITH ages AS (SELECT try_cast(age AS INTEGER) AS age FROM customers)
            SELECT
                count(*) FILTER (WHERE age IS NOT NULL), min(age), max(age),
                approx_quantile(age, [0.01, 0.25, 0.5, 0.75, 0.99]),
                count(*) FILTER (WHERE age < 10 OR age > 100)
            FROM ages
            """,
        )
        customer_detail = {
            "age_non_missing": int(age_row[0]),
            "age_min": int(age_row[1]),
            "age_max": int(age_row[2]),
            "age_quantiles": [float(value) for value in age_row[3]],
            "age_outside_10_100": int(age_row[4]),
            "club_member_status": _top_values(connection, "customers", "club_member_status"),
            "fashion_news_frequency": _top_values(
                connection, "customers", "fashion_news_frequency"
            ),
            "fn": _top_values(connection, "customers", "FN"),
            "active": _top_values(connection, "customers", "Active"),
        }

        variants = _one(
            connection,
            """
            WITH variants AS (
                SELECT product_code, count(*) AS n
                FROM articles GROUP BY product_code
            )
            SELECT count(*), avg(n), approx_quantile(n, [0.5, 0.9, 0.95, 0.99]),
                   max(n), count(*) FILTER (WHERE n > 1)
            FROM variants
            """,
        )
        article_detail = {
            "product_codes": int(variants[0]),
            "mean_variants_per_product": float(variants[1]),
            "variant_quantiles": [float(value) for value in variants[2]],
            "max_variants": int(variants[3]),
            "multi_variant_product_codes": int(variants[4]),
            "top_categories": {
                column: _top_values(connection, "articles", column)
                for column in (
                    "product_type_name",
                    "product_group_name",
                    "colour_group_name",
                    "department_name",
                    "index_name",
                    "section_name",
                    "garment_group_name",
                )
            },
        }

        submission_row = _one(
            connection,
            """
            WITH parsed AS (
                SELECT prediction, string_split(trim(prediction), ' ') AS items
                FROM submission
            )
            SELECT
                count(*), count(DISTINCT prediction),
                min(len(items)), max(len(items)),
                count(*) FILTER (WHERE len(items) <> 12),
                count(*) FILTER (WHERE len(list_distinct(items)) <> len(items))
            FROM parsed
            """,
        )
        invalid_prediction_tokens = int(
            _one(
                connection,
                """
                SELECT count(*)
                FROM submission, unnest(string_split(trim(prediction), ' ')) AS t(item)
                WHERE NOT regexp_full_match(item, '[0-9]{10}')
                """,
            )[0]
        )
        submission_detail = {
            "rows": int(submission_row[0]),
            "unique_prediction_strings": int(submission_row[1]),
            "min_tokens": int(submission_row[2]),
            "max_tokens": int(submission_row[3]),
            "rows_not_12_tokens": int(submission_row[4]),
            "rows_with_duplicate_tokens": int(submission_row[5]),
            "invalid_article_tokens": invalid_prediction_tokens,
        }

        relationship_queries = {
            "transaction_users_missing_customer": """
                SELECT count(*) FROM (
                    SELECT DISTINCT customer_id FROM transactions
                    ANTI JOIN customers USING (customer_id)
                )
            """,
            "transaction_items_missing_article": """
                SELECT count(*) FROM (
                    SELECT DISTINCT article_id FROM transactions
                    ANTI JOIN articles USING (article_id)
                )
            """,
            "customers_without_transactions": """
                SELECT count(*) FROM (
                    SELECT customer_id FROM customers
                    ANTI JOIN (SELECT DISTINCT customer_id FROM transactions) USING (customer_id)
                )
            """,
            "articles_without_transactions": """
                SELECT count(*) FROM (
                    SELECT article_id FROM articles
                    ANTI JOIN (SELECT DISTINCT article_id FROM transactions) USING (article_id)
                )
            """,
            "submission_customers_missing_customer": """
                SELECT count(*) FROM submission ANTI JOIN customers USING (customer_id)
            """,
            "customers_missing_submission": """
                SELECT count(*) FROM customers ANTI JOIN submission USING (customer_id)
            """,
        }
        relationships = {
            name: int(_one(connection, query)[0])
            for name, query in relationship_queries.items()
        }

        cutoff = str(
            _one(connection, "SELECT max(t_dat) - INTERVAL 6 DAY FROM transactions")[0]
        )[:10]
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE audit_validation_pairs AS
            SELECT DISTINCT customer_id, article_id
            FROM transactions
            WHERE t_dat >= DATE '{cutoff}'
            """
        )
        validation_summary_row = _one(
            connection,
            f"""
            SELECT count(*), count(DISTINCT customer_id), count(DISTINCT article_id)
            FROM transactions WHERE t_dat >= DATE '{cutoff}'
            """,
        )
        validation_pairs = int(
            _one(connection, "SELECT count(*) FROM audit_validation_pairs")[0]
        )
        validation_users = int(validation_summary_row[1])

        cold_users = int(
            _one(
                connection,
                f"""
                SELECT count(*) FROM (
                    SELECT DISTINCT customer_id FROM audit_validation_pairs
                    ANTI JOIN (
                        SELECT DISTINCT customer_id FROM transactions
                        WHERE t_dat < DATE '{cutoff}'
                    ) USING (customer_id)
                )
                """,
            )[0]
        )
        cold_items = int(
            _one(
                connection,
                f"""
                SELECT count(*) FROM (
                    SELECT DISTINCT article_id FROM audit_validation_pairs
                    ANTI JOIN (
                        SELECT DISTINCT article_id FROM transactions
                        WHERE t_dat < DATE '{cutoff}'
                    ) USING (article_id)
                )
                """,
            )[0]
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE audit_history_pairs_full AS
            SELECT DISTINCT customer_id, article_id
            FROM transactions WHERE t_dat < DATE '{cutoff}'
            """
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE audit_history_pairs_12w AS
            SELECT DISTINCT customer_id, article_id
            FROM transactions
            WHERE t_dat < DATE '{cutoff}'
              AND t_dat >= DATE '{cutoff}' - INTERVAL 12 WEEK
            """
        )
        repeat_full = int(
            _one(
                connection,
                """
                SELECT count(*) FROM audit_validation_pairs
                SEMI JOIN audit_history_pairs_full USING (customer_id, article_id)
                """,
            )[0]
        )
        repeat_12w = int(
            _one(
                connection,
                """
                SELECT count(*) FROM audit_validation_pairs
                SEMI JOIN audit_history_pairs_12w USING (customer_id, article_id)
                """,
            )[0]
        )
        repeat_user_hits_12w = int(
            _one(
                connection,
                """
                SELECT count(DISTINCT customer_id)
                FROM audit_validation_pairs
                SEMI JOIN audit_history_pairs_12w USING (customer_id, article_id)
                """,
            )[0]
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE audit_top_popularity_100 AS
            SELECT article_id,
                   sum(pow(2.0, -date_diff('day', t_dat, DATE '{cutoff}') / 7.0)) AS score
            FROM transactions
            WHERE t_dat < DATE '{cutoff}'
              AND t_dat >= DATE '{cutoff}' - INTERVAL 28 DAY
            GROUP BY article_id
            ORDER BY score DESC
            LIMIT 100
            """
        )
        popular_pair_hits = int(
            _one(
                connection,
                """
                SELECT count(*) FROM audit_validation_pairs
                SEMI JOIN audit_top_popularity_100 USING (article_id)
                """,
            )[0]
        )
        popular_user_hits = int(
            _one(
                connection,
                """
                SELECT count(DISTINCT customer_id)
                FROM audit_validation_pairs
                SEMI JOIN audit_top_popularity_100 USING (article_id)
                """,
            )[0]
        )
        combined_pair_hits = int(
            _one(
                connection,
                """
                SELECT count(*) FROM audit_validation_pairs v
                WHERE EXISTS (
                    SELECT 1 FROM audit_history_pairs_12w h
                    WHERE h.customer_id = v.customer_id AND h.article_id = v.article_id
                ) OR EXISTS (
                    SELECT 1 FROM audit_top_popularity_100 p
                    WHERE p.article_id = v.article_id
                )
                """,
            )[0]
        )
        combined_user_hits = int(
            _one(
                connection,
                """
                SELECT count(DISTINCT customer_id) FROM audit_validation_pairs v
                WHERE EXISTS (
                    SELECT 1 FROM audit_history_pairs_12w h
                    WHERE h.customer_id = v.customer_id AND h.article_id = v.article_id
                ) OR EXISTS (
                    SELECT 1 FROM audit_top_popularity_100 p
                    WHERE p.article_id = v.article_id
                )
                """,
            )[0]
        )

        connection.execute(
            """
            CREATE OR REPLACE TEMP VIEW audit_article_products AS
            SELECT article_id, product_code FROM articles
            """
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE audit_history_user_products_12w AS
            SELECT DISTINCT t.customer_id, a.product_code
            FROM transactions t
            JOIN audit_article_products a USING (article_id)
            WHERE t.t_dat < DATE '{cutoff}'
              AND t.t_dat >= DATE '{cutoff}' - INTERVAL 12 WEEK
            """
        )
        product_family_hits = int(
            _one(
                connection,
                """
                SELECT count(*)
                FROM audit_validation_pairs v
                JOIN audit_article_products a USING (article_id)
                SEMI JOIN audit_history_user_products_12w h
                    ON v.customer_id = h.customer_id AND a.product_code = h.product_code
                """,
            )[0]
        )
        product_family_extra = int(
            _one(
                connection,
                """
                SELECT count(*)
                FROM audit_validation_pairs v
                JOIN audit_article_products a USING (article_id)
                SEMI JOIN audit_history_user_products_12w h
                    ON v.customer_id = h.customer_id AND a.product_code = h.product_code
                ANTI JOIN audit_history_pairs_12w exact_history
                    ON v.customer_id = exact_history.customer_id
                   AND v.article_id = exact_history.article_id
                """,
            )[0]
        )

        validation = {
            "cutoff": cutoff,
            "rows": int(validation_summary_row[0]),
            "users": validation_users,
            "items": int(validation_summary_row[2]),
            "pairs": validation_pairs,
            "cold_users": cold_users,
            "cold_items": cold_items,
            "exact_repeat_pair_hits_full_history": repeat_full,
            "exact_repeat_pair_hits_12w": repeat_12w,
            "exact_repeat_user_hits_12w": repeat_user_hits_12w,
            "popular_top100_pair_hits": popular_pair_hits,
            "popular_top100_user_hits": popular_user_hits,
            "combined_pair_hits": combined_pair_hits,
            "combined_user_hits": combined_user_hits,
            "product_family_pair_hits_12w": product_family_hits,
            "product_family_extra_over_exact_12w": product_family_extra,
        }

        result = {
            "scope": {
                "raw_dir": str(raw_dir.resolve()),
                "image_audit_excluded": True,
                "elapsed_seconds": time.perf_counter() - started,
            },
            "inventory": inventory,
            "profiles": profiles,
            "key_duplicates": key_duplicates,
            "transactions": tx_summary,
            "user_activity": {
                "mean_events": float(user_activity[0]),
                "event_quantiles": [float(value) for value in user_activity[1]],
                "max_events": int(user_activity[2]),
                "mean_distinct_items": float(user_activity[3]),
                "single_event_users": int(user_activity[4]),
                "users": int(user_activity[5]),
            },
            "item_activity": {
                "mean_events": float(item_activity[0]),
                "event_quantiles": [float(value) for value in item_activity[1]],
                "max_events": int(item_activity[2]),
                "mean_distinct_users": float(item_activity[3]),
                "single_event_items": int(item_activity[4]),
                "items": int(item_activity[5]),
            },
            "activity_concentration": {
                "item_top_1pct_event_share": float(item_concentration[0]),
                "item_top_5pct_event_share": float(item_concentration[1]),
                "item_top_10pct_event_share": float(item_concentration[2]),
                "user_top_1pct_event_share": float(user_concentration[0]),
                "user_top_5pct_event_share": float(user_concentration[1]),
                "user_top_10pct_event_share": float(user_concentration[2]),
            },
            "repeat": {
                "unique_user_item_pairs": int(repeat_row[0]),
                "repeated_pairs": int(repeat_row[1]),
                "repeat_extra_events": int(repeat_row[2]),
                "max_events_per_pair": int(repeat_row[3]),
                "users_with_repeat": int(repeat_row[4]),
            },
            "baskets": {
                "user_days": int(basket_row[0]),
                "mean_distinct_items": float(basket_row[1]),
                "item_quantiles": [float(value) for value in basket_row[2]],
                "max_distinct_items": int(basket_row[3]),
                "multi_item_user_days": int(basket_row[4]),
            },
            "monthly": monthly,
            "customers": customer_detail,
            "articles": article_detail,
            "submission": submission_detail,
            "relationships": relationships,
            "validation": validation,
        }
        return result
    finally:
        connection.close()


def render_markdown(audit: dict[str, Any]) -> str:
    inventory = audit["inventory"]
    tx = audit["transactions"]
    validation = audit["validation"]
    profiles = audit["profiles"]
    repeat = audit["repeat"]
    baskets = audit["baskets"]
    articles = audit["articles"]
    customers = audit["customers"]
    relationships = audit["relationships"]
    key_duplicates = audit["key_duplicates"]
    concentration = audit["activity_concentration"]

    lines: list[str] = [
        "# H&M Personalized Fashion Recommendations 数据审计",
        "",
        "## 1. 审计范围与结论",
        "",
        "本报告对 Kaggle 提供的四张表格进行全量审计：交易、商品、客户和提交模板。"
        "本轮明确不下载、不扫描 images；图片的用途和接入条件在第 11 节说明。",
        "",
        f"- 交易数据覆盖 `{tx['min_date']}` 至 `{tx['max_date']}`，共 {_fmt_int(tx['rows'])} 条。",
        f"- 用户—商品矩阵密度仅 {_fmt_pct(tx['density'], 4)}，是典型的极稀疏隐式反馈问题。",
        f"- 最后一周验证集中，12 周精确复购只能覆盖 {_fmt_pct(validation['exact_repeat_pair_hits_12w'] / validation['pairs'])} 的真实 user-item 对。",
        f"- 全量近期 Top100 热门只能覆盖 {_fmt_pct(validation['popular_top100_pair_hits'] / validation['pairs'])}；两者合并覆盖 {_fmt_pct(validation['combined_pair_hits'] / validation['pairs'])}。",
        f"- 同 product_code 家族在 12 周内覆盖 {_fmt_pct(validation['product_family_pair_hits_12w'] / validation['pairs'])}，其中 {_fmt_int(validation['product_family_extra_over_exact_12w'])} 个验证对是精确复购没有覆盖的额外信号。",
        "- 因此 M1 不能只调 M0 权重；必须引入互补的 Item-CF/co-visitation、商品家族/内容和分群召回。",
        "",
        "## 2. 数据资产清单",
        "",
        _md_table(
            ["逻辑表", "文件", "大小", "行数", "列数"],
            [
                (
                    table,
                    info["filename"],
                    f"{info['bytes'] / 1024**2:.1f} MiB",
                    _fmt_int(info["rows"]),
                    len(info["columns"]),
                )
                for table, info in inventory.items()
            ],
        ),
        "",
        "### 表结构",
        "",
    ]
    for table, info in inventory.items():
        lines.extend([f"- `{table}`：`" + "`, `".join(info["columns"]) + "`"])

    lines.extend(
        [
            "",
            "## 3. 主键、重复与关联完整性",
            "",
            _md_table(
                ["检查项", "数量"],
                [
                    ("articles 重复 article_id", _fmt_int(key_duplicates["articles_duplicate_article_id"])),
                    ("customers 重复 customer_id", _fmt_int(key_duplicates["customers_duplicate_customer_id"])),
                    ("submission 重复 customer_id", _fmt_int(key_duplicates["submission_duplicate_customer_id"])),
                    ("transactions 完全重复的额外行", _fmt_int(key_duplicates["transactions_exact_duplicate_extra_rows"])),
                    ("同 user-item-day 的额外行", _fmt_int(key_duplicates["transactions_same_user_item_day_extra_rows"])),
                ],
            ),
            "",
            _md_table(
                ["关联检查", "未匹配数量"],
                [
                    ("交易用户在 customers 中缺失", _fmt_int(relationships["transaction_users_missing_customer"])),
                    ("交易商品在 articles 中缺失", _fmt_int(relationships["transaction_items_missing_article"])),
                    ("customers 从未发生交易", _fmt_int(relationships["customers_without_transactions"])),
                    ("articles 从未发生交易", _fmt_int(relationships["articles_without_transactions"])),
                    ("submission 用户在 customers 中缺失", _fmt_int(relationships["submission_customers_missing_customer"])),
                    ("customers 未出现在 submission", _fmt_int(relationships["customers_missing_submission"])),
                ],
            ),
            "",
            "完全重复交易不能在不了解业务语义时直接删除：它可能代表同一次购买多件同款。"
            "后续应同时保留 event count 和去重后的 user-item interaction 两种口径。",
            "",
            "## 4. 缺失值与字段基数",
            "",
        ]
    )
    for table in ("customers", "articles", "submission"):
        lines.extend(
            [
                f"### {table}",
                "",
                _md_table(
                    ["字段", "缺失", "缺失率", "唯一值"],
                    [
                        (
                            row["column"],
                            _fmt_int(row["missing"]),
                            _fmt_pct(row["missing_rate"]),
                            _fmt_int(row["unique"]),
                        )
                        for row in profiles[table]
                    ],
                ),
                "",
            ]
        )

    age_valid = customers["age_non_missing"]
    age_missing = inventory["customers"]["rows"] - age_valid
    lines.extend(
        [
            "## 5. 交易价格、渠道和时间变化",
            "",
            _md_table(
                ["价格统计", "最小", "P01", "P25", "P50", "P75", "P99", "最大", "均值"],
                [[
                    "price",
                    _fmt_float(tx["min_price"], 6),
                    *[_fmt_float(value, 6) for value in tx["price_quantiles"]],
                    _fmt_float(tx["max_price"], 6),
                    _fmt_float(tx["mean_price"], 6),
                ]],
            ),
            "",
            f"- price 缺失 {_fmt_int(tx['missing_price'])}，非正值 {_fmt_int(tx['non_positive_price'])}。",
            "- price 的业务币种/缩放方式没有在 CSV 中给出，因此只能作为相对价格、分位数和用户价格偏好使用，不能解释成实际货币金额。",
            "- sales_channel_id 只是编码；在没有官方语义证据前不能自行命名为线上或线下。",
            "",
            _md_table(
                ["sales_channel_id", "交易数", "占比"],
                [
                    (row["channel"], _fmt_int(row["rows"]), _fmt_pct(row["rate"]))
                    for row in tx["channels"]
                ],
            ),
            "",
            "月度交易量、活跃用户和活跃商品明显变化，证明随机切分会混合不同时间状态；热门、价格和商品活跃度特征必须按 cutoff 重算。",
            "",
            _md_table(
                ["月份", "交易数", "活跃用户", "活跃商品"],
                [
                    (row["month"], _fmt_int(row["rows"]), _fmt_int(row["users"]), _fmt_int(row["items"]))
                    for row in audit["monthly"]
                ],
            ),
            "",
            "## 6. 客户数据启发",
            "",
            f"- age 缺失 {_fmt_int(age_missing)}（{_fmt_pct(age_missing / inventory['customers']['rows'])}）。",
            f"- age 范围 {customers['age_min']}–{customers['age_max']}；P01/P25/P50/P75/P99 为 "
            + "/".join(_fmt_float(value, 0) for value in customers["age_quantiles"])
            + "。",
            f"- 年龄落在 10–100 之外的客户有 {_fmt_int(customers['age_outside_10_100'])} 个，应作为异常/缺失处理，而不是直接进入年龄桶。",
            "- `postal_code` 已哈希，只适合作为高基数类别或频次特征，不能解释为真实地理距离。",
            "- `FN`、`Active` 的大量空值不能擅自解释为 0；应通过类别分布和后续验证确定填充方式。",
            "",
        ]
    )
    for label, values in (
        ("club_member_status", customers["club_member_status"]),
        ("fashion_news_frequency", customers["fashion_news_frequency"]),
        ("FN", customers["fn"]),
        ("Active", customers["active"]),
    ):
        lines.extend(
            [
                f"**{label} Top values**",
                "",
                _md_table(["值", "客户数"], [(value, _fmt_int(count)) for value, count in values]),
                "",
            ]
        )

    lines.extend(
        [
            "## 7. 商品数据启发",
            "",
            f"- {_fmt_int(inventory['articles']['rows'])} 个 article 对应 {_fmt_int(articles['product_codes'])} 个 product_code。",
            f"- 平均每个 product_code 有 {_fmt_float(articles['mean_variants_per_product'], 2)} 个 article；"
            f"{_fmt_int(articles['multi_variant_product_codes'])} 个 product_code 存在多个变体。",
            "- product_code 将同款不同颜色/变体连接起来，是 M1 商品家族召回和 M2 交叉特征的直接依据。",
            "- product type、department、section、garment group、颜色等层级可用于内容召回、用户偏好画像和冷商品处理。",
            "",
            _md_table(
                ["变体数分位", "P50", "P90", "P95", "P99", "最大"],
                [[
                    "articles/product_code",
                    *[_fmt_float(value, 0) for value in articles["variant_quantiles"]],
                    _fmt_int(articles["max_variants"]),
                ]],
            ),
            "",
        ]
    )
    for column, values in articles["top_categories"].items():
        lines.extend(
            [
                f"**{column} Top values**",
                "",
                _md_table(["值", "article 数"], [(value, _fmt_int(count)) for value, count in values]),
                "",
            ]
        )

    user_activity = audit["user_activity"]
    item_activity = audit["item_activity"]
    lines.extend(
        [
            "## 8. 推荐系统结构：稀疏、长尾与复购",
            "",
            _md_table(
                ["对象", "均值", "P50", "P90", "P95", "P99", "最大", "仅 1 次"],
                [
                    (
                        "每用户交易数",
                        _fmt_float(user_activity["mean_events"], 2),
                        *[_fmt_float(value, 0) for value in user_activity["event_quantiles"]],
                        _fmt_int(user_activity["max_events"]),
                        _fmt_int(user_activity["single_event_users"]),
                    ),
                    (
                        "每商品交易数",
                        _fmt_float(item_activity["mean_events"], 2),
                        *[_fmt_float(value, 0) for value in item_activity["event_quantiles"]],
                        _fmt_int(item_activity["max_events"]),
                        _fmt_int(item_activity["single_event_items"]),
                    ),
                ],
            ),
            "",
            _md_table(
                ["集中度", "Top 1%", "Top 5%", "Top 10%"],
                [
                    (
                        "头部商品贡献的交易占比",
                        _fmt_pct(concentration["item_top_1pct_event_share"]),
                        _fmt_pct(concentration["item_top_5pct_event_share"]),
                        _fmt_pct(concentration["item_top_10pct_event_share"]),
                    ),
                    (
                        "高活跃用户贡献的交易占比",
                        _fmt_pct(concentration["user_top_1pct_event_share"]),
                        _fmt_pct(concentration["user_top_5pct_event_share"]),
                        _fmt_pct(concentration["user_top_10pct_event_share"]),
                    ),
                ],
            ),
            "",
            f"- 去重 user-item 对 {_fmt_int(repeat['unique_user_item_pairs'])}；其中 {_fmt_int(repeat['repeated_pairs'])} 对发生过重复购买（{_fmt_pct(repeat['repeated_pairs'] / repeat['unique_user_item_pairs'])}）。",
            f"- {_fmt_int(repeat['users_with_repeat'])} 个用户至少复购过一个 article（{_fmt_pct(repeat['users_with_repeat'] / tx['users'])}）。",
            f"- 重复事件比去重 interaction 多 {_fmt_int(repeat['repeat_extra_events'])} 条，因此 frequency 确实有信号，但不能把重复行一律当脏数据。",
            "- 极低矩阵密度说明不可能对全体 user-item 做笛卡尔积；两阶段召回—排序是计算和统计上的必要设计。",
            "",
            "## 9. 同日购物篮与 co-visitation 可行性",
            "",
            f"- 用户—日期组合共 {_fmt_int(baskets['user_days'])} 个，平均包含 {_fmt_float(baskets['mean_distinct_items'], 2)} 个不同 article。",
            f"- P50/P90/P95/P99 为 " + "/".join(_fmt_float(value, 0) for value in baskets["item_quantiles"]) + f"，最大 {_fmt_int(baskets['max_distinct_items'])}。",
            f"- {_fmt_int(baskets['multi_item_user_days'])} 个 user-day 含多个商品（{_fmt_pct(baskets['multi_item_user_days'] / baskets['user_days'])}）。",
            "- 这证明同日共现有可用信号，但 t_dat 不是订单/session ID；同日商品可能来自多笔订单，报告和面试中必须称为 basket-like co-visitation，不能声称是真实购物篮。",
            "- 活跃用户会产生平方级商品对，M1 必须限制用户历史长度、单日 basket 大小，并对热门偏置做 cosine/Jaccard/lift 归一化。",
            "",
            "## 10. 提交模板审计",
            "",
            _md_table(
                ["检查项", "结果"],
                [
                    ("客户行数", _fmt_int(audit["submission"]["rows"])),
                    ("唯一 prediction 字符串", _fmt_int(audit["submission"]["unique_prediction_strings"])),
                    ("每行最少/最多 token", f"{audit['submission']['min_tokens']}/{audit['submission']['max_tokens']}"),
                    ("不是 12 个 token 的行", _fmt_int(audit["submission"]["rows_not_12_tokens"])),
                    ("含重复 token 的行", _fmt_int(audit["submission"]["rows_with_duplicate_tokens"])),
                    ("非 10 位数字 article token", _fmt_int(audit["submission"]["invalid_article_tokens"])),
                ],
            ),
            "",
            "sample_submission 覆盖全部客户，但所有行使用同一个示例 prediction。它只定义提交格式，不是测试标签，也不能用于训练或评估。",
            "",
            "## 11. 最后一周验证与候选源上限",
            "",
            f"验证 cutoff 为 `{validation['cutoff']}`。",
            "",
            _md_table(
                ["验证统计", "数量", "占比"],
                [
                    ("验证交易行", _fmt_int(validation["rows"]), "—"),
                    ("验证用户", _fmt_int(validation["users"]), "—"),
                    ("验证商品", _fmt_int(validation["items"]), "—"),
                    ("验证去重 user-item 对", _fmt_int(validation["pairs"]), "100%"),
                    ("历史中未出现的验证用户", _fmt_int(validation["cold_users"]), _fmt_pct(validation["cold_users"] / validation["users"])),
                    ("历史中未出现的验证商品", _fmt_int(validation["cold_items"]), _fmt_pct(validation["cold_items"] / validation["items"])),
                    ("全历史精确复购覆盖", _fmt_int(validation["exact_repeat_pair_hits_full_history"]), _fmt_pct(validation["exact_repeat_pair_hits_full_history"] / validation["pairs"])),
                    ("12 周精确复购覆盖", _fmt_int(validation["exact_repeat_pair_hits_12w"]), _fmt_pct(validation["exact_repeat_pair_hits_12w"] / validation["pairs"])),
                    ("28 天 Top100 热门覆盖", _fmt_int(validation["popular_top100_pair_hits"]), _fmt_pct(validation["popular_top100_pair_hits"] / validation["pairs"])),
                    ("复购 ∪ Top100 热门", _fmt_int(validation["combined_pair_hits"]), _fmt_pct(validation["combined_pair_hits"] / validation["pairs"])),
                    ("12 周同 product_code 覆盖", _fmt_int(validation["product_family_pair_hits_12w"]), _fmt_pct(validation["product_family_pair_hits_12w"] / validation["pairs"])),
                    ("product_code 相对精确复购新增", _fmt_int(validation["product_family_extra_over_exact_12w"]), _fmt_pct(validation["product_family_extra_over_exact_12w"] / validation["pairs"])),
                ],
            ),
            "",
            "这些是候选源的可覆盖性诊断，不等同于最终 Recall@100：product family 可能为一个已购 product_code 产生多个候选，仍需受总候选预算约束。",
            "",
            "## 12. 审计后的阶段规划",
            "",
            _md_table(
                ["数据/信号", "启发", "使用阶段"],
                [
                    ("时间、用户、商品长尾", "固定时间切分；按活跃度和头尾商品分桶", "M0/M3"),
                    ("精确复购", "保留 recency、frequency、history window 多组召回", "M1；rank/score 进入 M2"),
                    ("同日和跨日共现", "构建 co-visitation/Item-CF，并归一化热门偏置", "M1"),
                    ("product_code 变体", "同款不同 article 的商品家族召回", "M1；交叉特征进入 M2"),
                    ("商品类别/颜色/描述", "内容相似、用户类别偏好、冷商品召回；需声明候选目录可见性假设", "M1/M2；文本 embedding 属 M4"),
                    ("age/会员/资讯偏好", "先做支持度足够的分群热门，再作为排序特征", "M1/M2"),
                    ("price/channel", "用户价格偏好、商品价格位置、渠道偏好；不可用于当前 M0 的三列输入", "M2"),
                    ("submission 用户全集", "为无近期历史用户设计全局/分群兜底并保证 12 个唯一 ID", "最终推理/提交"),
                    ("缺失和高基数字段", "显式 missing 类别、频次编码；不要随意把缺失当 0", "M1/M2"),
                ],
            ),
            "",
            "### M1 推荐实施顺序",
            "",
            "1. 先定义 eligible catalog：严格协议仅含 cutoff 前出现过的商品；冷商品协议允许全量 article 元数据但单独标注；",
            "2. 修复抽样实验：全量统计召回源，只抽样目标用户；",
            "3. 复购窗口与时间衰减消融；",
            "4. 全量 7/14/28 天热门与趋势热门；",
            "5. product_code 商品家族召回；",
            "6. basket-like 和跨日 Item-CF；",
            "7. 年龄/类别等分群热门；",
            "8. 商品属性内容召回；",
            "9. 固定 Top100 做 standalone、marginal、overlap 和 leave-one-out 消融。",
            "",
            "## 13. Images 是什么，能怎么用",
            "",
            "images 为 article 的商品图片。它们不会直接作为用户行为，而是先由视觉模型编码成固定长度 embedding，例如每件商品一个 512 维向量。",
            "",
            "典型用途：",
            "",
            "1. **视觉相似召回**：用户买过黑色连衣裙 A，用 A 的图片 embedding 在向量索引中寻找外观相似商品；",
            "2. **新品/冷商品召回**：新品没有交易共现，但图片仍能与已有商品比较；",
            "3. **排序特征**：计算用户历史图片向量均值与候选商品向量的余弦相似度；",
            "4. **多模态商品表示**：融合图片、detail_desc、类别和协同信号；",
            "5. **去重/近重复识别**：发现视觉上几乎相同但 article_id 不同的商品。",
            "",
            "images 不是把 JPG 直接塞给 LightGBM。合理流程是：",
            "",
            "```text",
            "image → 预训练视觉模型/CLIP → item embedding → ANN 近邻或相似度特征",
            "```",
            "",
            "本项目暂缓图片有三个原因：先证明表格/协同 baseline；图片计算和存储成本更高；没有强 baseline 时无法判断视觉增益。进入 M4 前应先审计图片—article 覆盖率、损坏文件、尺寸和重复图片。",
            "",
            "## 14. 限制与证据边界",
            "",
            "- 本报告审计所有 CSV，但明确没有审计图片文件。",
            "- 购买日志是隐式正反馈，没有曝光和未点击日志；未购买不能直接视为负反馈。",
            "- t_dat 只有日期，不是订单或 session ID。",
            "- sample_submission 是格式模板，不是测试标签；其中示例 prediction 不代表真实推荐。",
            "- 离线覆盖率只能决定候选上限，不能替代真实线上 CTR/CVR/A/B 测试。",
            "- 本报告中的 product family 可覆盖性是诊断上限，实际召回仍要固定候选数并做消融。",
            "- articles 没有上架/下架时间。对较早 cutoff 使用全量 article 目录可能暴露未来商品；严格评测应限制为 cutoff 前已出现商品，冷商品实验需单独声明全目录可见假设。",
            "- customers 的会员状态、资讯偏好、Active 等字段没有快照时间。跨多个较早窗口使用同一静态表可能有时间穿越风险；应优先使用稳定属性，并把动态状态结果列为敏感性实验。",
            "- 当前 M1 项目决策：最终一周验证采用 optimistic_all_articles，允许 articles 全目录作为候选目录，但所有行为统计严格限制在 cutoff 前。该口径不自动外推到更早滚动窗口。",
            "- 当前 M1 客户字段决策：主实验仅使用相对稳定的 age 分桶；FN、Active、会员状态和资讯偏好不进入主结果，只能作为单独标注的敏感性实验。",
            "",
            "## 15. 复现信息",
            "",
            f"- 审计耗时：{_fmt_float(audit['scope']['elapsed_seconds'], 2)} 秒；",
            "- 交易 CSV 先按显式 schema 缓存为 Parquet；",
            "- article_id/customer_id 均按字符串处理，保留前导零；",
            "- DuckDB 内存上限 12GB、8 threads；",
            "- 机器生成的完整审计对象保存在 `data/interim/audit/audit.json`，原始数据和中间文件不进入 Git。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(raw_dir: Path, work_dir: Path, output_path: Path) -> dict[str, Any]:
    audit = collect_audit(raw_dir, work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(audit), encoding="utf-8")
    return audit
