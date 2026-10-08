"""Cutoff-safe feasibility audit for a MIND warm-side retrieval channel.

This module performs descriptive statistics and non-neural proxy retrieval only.
It deliberately does not train MIND or alter the production candidate pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TX = ROOT / "data" / "interim" / "audit" / "transactions.parquet"
ARTICLES = ROOT / "data" / "raw" / "articles.csv"
REPORT_DIR = ROOT / "reports" / "mind_warm_side"
CONTRACT_PATH = REPORT_DIR / "MIND_WARM_SIDE_AUDIT_CONTRACT.json"
METRICS_PATH = REPORT_DIR / "MIND_WARM_SIDE_AUDIT_METRICS.json"
REPORT_PATH = REPORT_DIR / "MIND_WARM_SIDE_FEASIBILITY_AUDIT.md"

WINDOWS = {
    "winter_20200122": "2020-01-22",
    "spring_20200318": "2020-03-18",
    "early_summer_20200624": "2020-06-24",
    "late_summer_20200819": "2020-08-19",
}

CANDIDATES = {
    name: ROOT / "artifacts" / "m2_9" / "cache-v1" / cutoff / "expanded-candidates.parquet"
    for name, cutoff in WINDOWS.items()
}

SOURCE_COLUMNS = {
    "repurchase": "repurchase_present",
    "recent_popularity": "recent_popularity_present",
    "product_family": "product_family_present",
    "user_day_covisit": "user_day_covisit_present",
    "age_popularity": "age_popularity_present",
    "attribute_content": "attribute_content_present",
    "item2vec": "item2vec_present",
}

INTEREST_LEVELS = {
    "garment_group": "garment_group_name",
    "section": "section_name",
    "product_group": "product_group_name",
}


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_native(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dynamic_interest_count(history_length: int, maximum: int) -> int:
    """Project definition: floor(log2(n)), bounded to [1, maximum]."""
    if history_length <= 1:
        return 1
    return max(1, min(maximum, int(math.floor(math.log2(history_length)))))


def decide(summary: dict[str, float], rules: dict[str, Any]) -> str:
    go = rules["go"]
    conditional = rules["conditional_go"]

    def passes(rule: dict[str, float]) -> bool:
        return bool(
            summary["clear_multi_interest_truth_share_mean"]
            >= rule["clear_multi_interest_truth_share_mean_min"]
            and summary["secondary_mode_share_of_current_misses_mean"]
            >= rule["secondary_mode_share_of_current_misses_mean_min"]
            and summary["multi_profile_marginal_recall_mean"]
            >= rule["multi_profile_marginal_recall_mean_min"]
            and summary["multi_profile_positive_windows"]
            >= rule["multi_profile_marginal_recall_positive_windows_min"]
            and summary["multi_profile_unique_over_single_mean"]
            >= rule["multi_profile_unique_over_single_mean_min"]
        )

    if passes(go):
        return "GO"
    if passes(conditional):
        return "CONDITIONAL GO"
    return "NO-GO"


def _distribution(con: duckdb.DuckDBPyConnection, expression: str, table_sql: str) -> dict[str, float | int | None]:
    row = con.execute(
        f"""SELECT count(*)::BIGINT, avg({expression}),
        quantile_cont({expression},0.25),quantile_cont({expression},0.50),
        quantile_cont({expression},0.75),quantile_cont({expression},0.90),
        quantile_cont({expression},0.95),quantile_cont({expression},0.99),
        min({expression}),max({expression}) FROM {table_sql}"""
    ).fetchone()
    keys = ["count", "mean", "p25", "p50", "p75", "p90", "p95", "p99", "min", "max"]
    return _native(dict(zip(keys, row)))


def _category_tables(con: duckdb.DuckDBPyConnection) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for key, column in INTEREST_LEVELS.items():
        con.execute(
            f"""CREATE TEMP TABLE {key}_counts AS
            SELECT customer_id,coalesce({column},'__MISSING__') category,count(*)::BIGINT n
            FROM hist84_articles GROUP BY customer_id,category"""
        )
        con.execute(
            f"""CREATE TEMP TABLE {key}_stats AS WITH totals AS (
                SELECT customer_id,category,n,sum(n) OVER(PARTITION BY customer_id) total,
                    count(*) OVER(PARTITION BY customer_id) unique_categories,
                    row_number() OVER(PARTITION BY customer_id ORDER BY n DESC,category) category_rank
                FROM {key}_counts
            ), aggregate_stats AS (
                SELECT customer_id,max(total)::BIGINT total,max(unique_categories)::BIGINT unique_categories,
                    -sum((n*1.0/total)*ln(n*1.0/total)) entropy,
                    max(n*1.0/total) FILTER(WHERE category_rank=1) top1_share,
                    sum(n*1.0/total) FILTER(WHERE category_rank<=2) top2_coverage,
                    sum(n*1.0/total) FILTER(WHERE category_rank<=3) top3_coverage
                FROM totals GROUP BY customer_id
            ) SELECT *,CASE WHEN unique_categories<=1 THEN 0.0
                ELSE entropy/ln(unique_categories) END normalized_entropy
            FROM aggregate_stats"""
        )
        results[key] = {
            "users": int(con.execute(f"SELECT count(*) FROM {key}_stats").fetchone()[0]),
            "unique_categories": _distribution(con, "unique_categories", f"{key}_stats s JOIN truth_users t USING(customer_id)"),
            "normalized_entropy": _distribution(con, "normalized_entropy", f"{key}_stats s JOIN truth_users t USING(customer_id)"),
            "top1_share": _distribution(con, "top1_share", f"{key}_stats s JOIN truth_users t USING(customer_id)"),
            "top3_coverage": _distribution(con, "top3_coverage", f"{key}_stats s JOIN truth_users t USING(customer_id)"),
        }
    return results


def _proxy_metrics(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, float | int]:
    rows = int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
    users = int(con.execute(f"SELECT count(DISTINCT customer_id) FROM {table}").fetchone()[0])
    overlap = int(
        con.execute(
            f"SELECT count(*) FROM {table} p JOIN candidates c USING(customer_id,article_id)"
        ).fetchone()[0]
    )
    hits = int(con.execute(f"SELECT count(*) FROM truth_warm t JOIN {table} p USING(customer_id,article_id)").fetchone()[0])
    marginal = int(
        con.execute(
            f"""SELECT count(*) FROM truth_warm t JOIN {table} p USING(customer_id,article_id)
            LEFT JOIN candidates c USING(customer_id,article_id) WHERE c.article_id IS NULL"""
        ).fetchone()[0]
    )
    truth_pairs = int(con.execute("SELECT count(*) FROM truth_warm").fetchone()[0])
    return {
        "candidate_rows": rows,
        "users_with_candidates": users,
        "mean_candidates_per_candidate_user": rows / users if users else 0.0,
        "candidate_overlap_rows_with_current_union": overlap,
        "candidate_overlap_rate_with_current_union": overlap / rows if rows else 0.0,
        "standalone_truth_pairs": hits,
        "standalone_recall": hits / truth_pairs if truth_pairs else 0.0,
        "marginal_truth_pairs_over_current_union": marginal,
        "marginal_recall_over_current_union": marginal / truth_pairs if truth_pairs else 0.0,
    }


def audit_window(name: str, cutoff: str, candidate_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if cutoff == "2020-09-16":
        raise ValueError("final week is forbidden in the MIND feasibility audit")
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='12GB'")
    con.execute(
        f"""CREATE TEMP TABLE articles AS SELECT article_id,
        coalesce(product_group_name,'__MISSING__') product_group_name,
        coalesce(section_name,'__MISSING__') section_name,
        coalesce(garment_group_name,'__MISSING__') garment_group_name
        FROM read_csv_auto('{_sql_path(ARTICLES)}',header=true,all_varchar=true)"""
    )
    con.execute(f"CREATE TEMP TABLE candidates AS SELECT * FROM read_parquet('{_sql_path(candidate_path)}')")
    con.execute("CREATE TEMP TABLE eval_users AS SELECT DISTINCT customer_id FROM candidates")
    con.execute(
        f"""CREATE TEMP TABLE warm_catalog AS SELECT DISTINCT article_id
        FROM read_parquet('{_sql_path(TX)}') WHERE t_dat<DATE '{cutoff}'"""
    )
    con.execute(
        f"""CREATE TEMP TABLE truth_warm AS SELECT DISTINCT t.customer_id,t.article_id
        FROM read_parquet('{_sql_path(TX)}') t JOIN eval_users u USING(customer_id)
        JOIN warm_catalog w USING(article_id)
        WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY"""
    )
    con.execute("CREATE TEMP TABLE truth_users AS SELECT customer_id,count(*)::BIGINT truth_pairs FROM truth_warm GROUP BY customer_id")
    con.execute(
        f"""CREATE TEMP TABLE hist84 AS SELECT DISTINCT t.customer_id,t.t_dat,t.article_id
        FROM read_parquet('{_sql_path(TX)}') t JOIN eval_users u USING(customer_id)
        WHERE t_dat>=DATE '{cutoff}'-INTERVAL 84 DAY AND t_dat<DATE '{cutoff}'"""
    )
    con.execute(
        f"""CREATE TEMP TABLE histfull AS SELECT DISTINCT t.customer_id,t.t_dat,t.article_id
        FROM read_parquet('{_sql_path(TX)}') t JOIN eval_users u USING(customer_id)
        WHERE t_dat<DATE '{cutoff}'"""
    )
    con.execute(
        """CREATE TEMP TABLE hist84_articles AS SELECT h.*,a.product_group_name,a.section_name,a.garment_group_name
        FROM hist84 h JOIN articles a USING(article_id)"""
    )
    con.execute(
        f"""CREATE TEMP TABLE user_stats AS WITH h84 AS (
            SELECT customer_id,count(*)::BIGINT effective_interactions_84d,
                count(DISTINCT article_id)::BIGINT distinct_items_84d,
                count(DISTINCT t_dat)::BIGINT purchase_days_84d
            FROM hist84 GROUP BY customer_id
        ), hf AS (SELECT customer_id,count(*)::BIGINT effective_interactions_full FROM histfull GROUP BY customer_id)
        SELECT u.customer_id,coalesce(h.effective_interactions_84d,0)::BIGINT effective_interactions_84d,
            coalesce(h.distinct_items_84d,0)::BIGINT distinct_items_84d,
            coalesce(h.purchase_days_84d,0)::BIGINT purchase_days_84d,
            coalesce(f.effective_interactions_full,0)::BIGINT effective_interactions_full
        FROM eval_users u LEFT JOIN h84 h USING(customer_id) LEFT JOIN hf f USING(customer_id)"""
    )

    candidate_shape = con.execute(
        """SELECT count(*)::BIGINT,count(DISTINCT customer_id)::BIGINT,
        min(candidate_rank)::BIGINT,max(candidate_rank)::BIGINT,
        count(DISTINCT customer_id||'|'||article_id)::BIGINT FROM candidates"""
    ).fetchone()
    if candidate_shape[0] != candidate_shape[4]:
        raise AssertionError(f"duplicate candidate pair in {name}")

    truth_pairs = int(con.execute("SELECT count(*) FROM truth_warm").fetchone()[0])
    truth_users = int(con.execute("SELECT count(*) FROM truth_users").fetchone()[0])
    current_hits = int(con.execute("SELECT count(*) FROM truth_warm JOIN candidates USING(customer_id,article_id)").fetchone()[0])
    hit_users = int(
        con.execute(
            "SELECT count(DISTINCT t.customer_id) FROM truth_warm t JOIN candidates c USING(customer_id,article_id)"
        ).fetchone()[0]
    )

    history = {
        metric: _distribution(con, metric, "user_stats s JOIN truth_users t USING(customer_id)")
        for metric in (
            "effective_interactions_84d",
            "distinct_items_84d",
            "purchase_days_84d",
            "effective_interactions_full",
        )
    }
    thresholds: dict[str, Any] = {}
    for threshold in (5, 10, 20, 50):
        row = con.execute(
            f"""SELECT count(*) FILTER(WHERE s.effective_interactions_84d>={threshold})::BIGINT,
            sum(t.truth_pairs) FILTER(WHERE s.effective_interactions_84d>={threshold})::BIGINT
            FROM truth_users t JOIN user_stats s USING(customer_id)"""
        ).fetchone()
        thresholds[str(threshold)] = {
            "eligible_truth_users": int(row[0] or 0),
            "truth_user_share": (row[0] or 0) / truth_users if truth_users else 0.0,
            "eligible_warm_truth_pairs": int(row[1] or 0),
            "warm_truth_pair_share": (row[1] or 0) / truth_pairs if truth_pairs else 0.0,
        }

    categories = _category_tables(con)
    con.execute(
        """CREATE TEMP TABLE multi_users AS SELECT s.customer_id,
        CASE WHEN s.effective_interactions_84d>=10 AND g.unique_categories>=3
            AND g.normalized_entropy>=0.6 AND g.top1_share<=0.7 THEN 1 ELSE 0 END clear_multi_interest
        FROM user_stats s LEFT JOIN garment_group_stats g USING(customer_id)"""
    )
    multi = con.execute(
        """SELECT count(*) FILTER(WHERE m.clear_multi_interest=1)::BIGINT,
        sum(t.truth_pairs) FILTER(WHERE m.clear_multi_interest=1)::BIGINT
        FROM truth_users t JOIN multi_users m USING(customer_id)"""
    ).fetchone()

    con.execute(
        """CREATE TEMP TABLE garment_ranked AS SELECT customer_id,category,n,
        row_number() OVER(PARTITION BY customer_id ORDER BY n DESC,category) mode_rank
        FROM garment_group_counts"""
    )
    con.execute(
        f"""CREATE TEMP TABLE recent_garment_counts AS SELECT customer_id,garment_group_name category,count(*)::BIGINT n
        FROM hist84_articles WHERE t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY
        GROUP BY customer_id,category"""
    )
    con.execute(
        f"""CREATE TEMP TABLE old_garment_counts AS SELECT customer_id,garment_group_name category,count(*)::BIGINT n
        FROM hist84_articles WHERE t_dat<DATE '{cutoff}'-INTERVAL 28 DAY
        GROUP BY customer_id,category"""
    )
    con.execute(
        """CREATE TEMP TABLE recent_dominant AS SELECT customer_id,category FROM recent_garment_counts
        QUALIFY row_number() OVER(PARTITION BY customer_id ORDER BY n DESC,category)=1"""
    )
    con.execute(
        """CREATE TEMP TABLE full_dominant AS SELECT customer_id,category FROM garment_ranked WHERE mode_rank=1"""
    )
    con.execute(
        """CREATE TEMP TABLE jsd AS WITH recent_total AS (
            SELECT customer_id,sum(n) total FROM recent_garment_counts GROUP BY customer_id
        ), old_total AS (SELECT customer_id,sum(n) total FROM old_garment_counts GROUP BY customer_id),
        cats AS (SELECT customer_id,category FROM recent_garment_counts UNION SELECT customer_id,category FROM old_garment_counts),
        probs AS (SELECT c.customer_id,c.category,coalesce(r.n*1.0/rt.total,0) p,
            coalesce(o.n*1.0/ot.total,0) q FROM cats c
            JOIN recent_total rt USING(customer_id) JOIN old_total ot USING(customer_id)
            LEFT JOIN recent_garment_counts r USING(customer_id,category)
            LEFT JOIN old_garment_counts o USING(customer_id,category))
        SELECT customer_id,sum(CASE WHEN p>0 THEN 0.5*p*ln(p/((p+q)/2))/ln(2) ELSE 0 END
            +CASE WHEN q>0 THEN 0.5*q*ln(q/((p+q)/2))/ln(2) ELSE 0 END) jsd
        FROM probs GROUP BY customer_id"""
    )

    con.execute(
        """CREATE TEMP TABLE truth_features AS SELECT t.customer_id,t.article_id,a.garment_group_name,
        CASE WHEN c.article_id IS NOT NULL THEN 1 ELSE 0 END current_hit,
        coalesce(m.clear_multi_interest,0) clear_multi_interest,
        coalesce(hg.n,0)::BIGINT history_mode_count,
        coalesce(rg.n,0)::BIGINT recent_mode_count,
        coalesce(og.n,0)::BIGINT old_mode_count,
        CASE WHEN fd.category=a.garment_group_name THEN 1 ELSE 0 END full_dominant_match,
        CASE WHEN rd.category=a.garment_group_name THEN 1 ELSE 0 END recent_dominant_match,
        CASE WHEN hi.article_id IS NOT NULL THEN 1 ELSE 0 END exact_item_in_84d
        FROM truth_warm t JOIN articles a USING(article_id)
        LEFT JOIN candidates c USING(customer_id,article_id)
        LEFT JOIN multi_users m USING(customer_id)
        LEFT JOIN garment_group_counts hg ON hg.customer_id=t.customer_id AND hg.category=a.garment_group_name
        LEFT JOIN recent_garment_counts rg ON rg.customer_id=t.customer_id AND rg.category=a.garment_group_name
        LEFT JOIN old_garment_counts og ON og.customer_id=t.customer_id AND og.category=a.garment_group_name
        LEFT JOIN full_dominant fd USING(customer_id) LEFT JOIN recent_dominant rd USING(customer_id)
        LEFT JOIN (SELECT DISTINCT customer_id,article_id FROM hist84) hi USING(customer_id,article_id)"""
    )
    support = con.execute(
        """SELECT
        count(*) FILTER(WHERE history_mode_count>0)::BIGINT,
        count(*) FILTER(WHERE full_dominant_match=1)::BIGINT,
        count(*) FILTER(WHERE history_mode_count>0 AND full_dominant_match=0)::BIGINT,
        count(*) FILTER(WHERE recent_mode_count>0 AND old_mode_count>0)::BIGINT,
        count(*) FILTER(WHERE recent_mode_count>0 AND old_mode_count=0)::BIGINT,
        count(*) FILTER(WHERE recent_mode_count=0 AND old_mode_count>0)::BIGINT,
        count(*) FILTER(WHERE recent_mode_count=0 AND old_mode_count=0)::BIGINT,
        count(*) FILTER(WHERE recent_dominant_match=1)::BIGINT,
        count(*) FILTER(WHERE exact_item_in_84d=1)::BIGINT
        FROM truth_features"""
    ).fetchone()
    missed = int(con.execute("SELECT count(*) FROM truth_features WHERE current_hit=0").fetchone()[0])
    missed_secondary = int(
        con.execute(
            """SELECT count(*) FROM truth_features WHERE current_hit=0 AND clear_multi_interest=1
            AND history_mode_count>0 AND full_dominant_match=0"""
        ).fetchone()[0]
    )
    missed_old_only = int(
        con.execute(
            """SELECT count(*) FROM truth_features WHERE current_hit=0 AND clear_multi_interest=1
            AND recent_mode_count=0 AND old_mode_count>0"""
        ).fetchone()[0]
    )

    con.execute(
        """CREATE TEMP TABLE baskets AS SELECT customer_id,t_dat,count(*)::BIGINT distinct_articles
        FROM hist84 GROUP BY customer_id,t_dat"""
    )
    basket_dist = _distribution(con, "distinct_articles", "baskets")
    basket_summary = con.execute(
        """WITH multi_day_users AS (SELECT DISTINCT customer_id FROM baskets WHERE distinct_articles>=2)
        SELECT count(*)::BIGINT,count(*) FILTER(WHERE distinct_articles>=2)::BIGINT,
        (SELECT count(*) FROM user_stats WHERE effective_interactions_84d>0)::BIGINT,
        (SELECT count(*) FROM multi_day_users)::BIGINT,
        (SELECT count(*) FROM truth_users t JOIN multi_day_users m USING(customer_id))::BIGINT,
        (SELECT sum(t.truth_pairs) FROM truth_users t JOIN multi_day_users m USING(customer_id))::BIGINT
        FROM baskets"""
    ).fetchone()

    source_support_terms = [f"coalesce(c.{column},0)" for column in SOURCE_COLUMNS.values()]
    source_total = "+".join(source_support_terms)
    source_metrics: dict[str, Any] = {}
    for source, column in SOURCE_COLUMNS.items():
        population_row = con.execute(
            f"""SELECT count(*) FILTER(WHERE coalesce(c.{column},0)=1)::BIGINT,
            count(DISTINCT customer_id) FILTER(WHERE coalesce(c.{column},0)=1)::BIGINT,
            count(*) FILTER(WHERE coalesce(c.{column},0)=1 AND ({source_total})>=2)::BIGINT
            FROM candidates c"""
        ).fetchone()
        row = con.execute(
            f"""SELECT count(*) FILTER(WHERE coalesce(c.{column},0)=1)::BIGINT,
            count(*) FILTER(WHERE coalesce(c.{column},0)=1 AND ({source_total})=1)::BIGINT
            FROM truth_warm t LEFT JOIN candidates c USING(customer_id,article_id)"""
        ).fetchone()
        source_metrics[source] = {
            "retained_candidate_rows": int(population_row[0] or 0),
            "retained_candidate_users": int(population_row[1] or 0),
            "mean_retained_candidates_per_all_candidate_users": (population_row[0] or 0) / candidate_shape[1]
            if candidate_shape[1]
            else 0.0,
            "retained_candidate_rows_overlapping_another_source": int(population_row[2] or 0),
            "retained_candidate_overlap_rate": (population_row[2] or 0) / population_row[0]
            if population_row[0]
            else 0.0,
            "supported_warm_truth_pairs_in_current_union": int(row[0] or 0),
            "exclusive_supported_warm_truth_pairs_in_current_union": int(row[1] or 0),
            "recall_on_warm_truth": (row[0] or 0) / truth_pairs if truth_pairs else 0.0,
        }
    candidate_overlap = con.execute(
        f"""SELECT count(*)::BIGINT,
        count(*) FILTER(WHERE ({source_total})>=2)::BIGINT,
        count(*) FILTER(WHERE coalesce(item2vec_present,0)=1)::BIGINT,
        count(*) FILTER(WHERE coalesce(item2vec_present,0)=1 AND ({source_total})>=2)::BIGINT,
        count(*) FILTER(WHERE coalesce(item2vec_is_new,0)=1)::BIGINT FROM candidates c"""
    ).fetchone()

    con.execute(
        f"""CREATE TEMP TABLE recent_item_pop AS WITH counts AS (
            SELECT t.article_id,a.garment_group_name,count(*)::BIGINT events
            FROM read_parquet('{_sql_path(TX)}') t JOIN articles a USING(article_id)
            WHERE t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY AND t_dat<DATE '{cutoff}'
            GROUP BY t.article_id,a.garment_group_name
        ) SELECT *,row_number() OVER(PARTITION BY garment_group_name ORDER BY events DESC,article_id) item_rank
        FROM counts"""
    )
    con.execute(
        """CREATE TEMP TABLE single_proxy AS SELECT g.customer_id,p.article_id,g.mode_rank,p.item_rank
        FROM garment_ranked g JOIN recent_item_pop p ON g.category=p.garment_group_name
        WHERE g.mode_rank=1 AND p.item_rank<=60"""
    )
    con.execute(
        """CREATE TEMP TABLE recent_proxy AS SELECT g.customer_id,p.article_id,1::BIGINT mode_rank,p.item_rank
        FROM recent_dominant g JOIN recent_item_pop p ON g.category=p.garment_group_name
        WHERE p.item_rank<=60"""
    )
    con.execute(
        """CREATE TEMP TABLE multi_proxy AS SELECT g.customer_id,p.article_id,g.mode_rank,p.item_rank
        FROM garment_ranked g JOIN recent_item_pop p ON g.category=p.garment_group_name
        WHERE g.mode_rank<=3 AND g.n>=2 AND p.item_rank<=20"""
    )
    proxy = {
        "single_profile": _proxy_metrics(con, "single_proxy"),
        "recent_profile": _proxy_metrics(con, "recent_proxy"),
        "multi_profile": _proxy_metrics(con, "multi_proxy"),
    }
    multi_unique_single = int(
        con.execute(
            """SELECT count(*) FROM truth_warm t JOIN multi_proxy m USING(customer_id,article_id)
            LEFT JOIN candidates c USING(customer_id,article_id)
            LEFT JOIN single_proxy s USING(customer_id,article_id)
            WHERE c.article_id IS NULL AND s.article_id IS NULL"""
        ).fetchone()[0]
    )
    secondary_proxy = int(
        con.execute(
            """SELECT count(*) FROM truth_warm t JOIN multi_proxy m USING(customer_id,article_id)
            LEFT JOIN candidates c USING(customer_id,article_id)
            WHERE c.article_id IS NULL AND m.mode_rank>1"""
        ).fetchone()[0]
    )
    proxy["multi_profile"]["marginal_truth_pairs_unique_over_single_profile"] = multi_unique_single
    proxy["multi_profile"]["marginal_recall_unique_over_single_profile"] = multi_unique_single / truth_pairs if truth_pairs else 0.0
    proxy["multi_profile"]["marginal_truth_pairs_from_secondary_modes"] = secondary_proxy

    dynamic_k: dict[str, Any] = {}
    lengths = [int(row[0]) for row in con.execute(
        "SELECT effective_interactions_84d FROM user_stats s JOIN truth_users t USING(customer_id)"
    ).fetchall()]
    for maximum in (2, 3, 4, 5):
        values = [dynamic_interest_count(length, maximum) for length in lengths]
        dynamic_k[str(maximum)] = {
            "mean_interest_count": float(np.mean(values)) if values else 0.0,
            "distribution_users": {str(k): int(sum(v == k for v in values)) for k in range(1, maximum + 1)},
            "distribution_user_share": {
                str(k): float(sum(v == k for v in values) / len(values)) if values else 0.0
                for k in range(1, maximum + 1)
            },
        }

    temporal = {
        "recent_vs_older_garment_jsd": _distribution(con, "jsd", "jsd j JOIN truth_users t USING(customer_id)"),
        "truth_pairs_with_any_84d_garment_support": int(support[0]),
        "truth_pairs_matching_full_dominant_garment": int(support[1]),
        "truth_pairs_matching_secondary_historical_garment": int(support[2]),
        "truth_pairs_supported_in_recent_and_older": int(support[3]),
        "truth_pairs_supported_recent_only": int(support[4]),
        "truth_pairs_supported_older_only": int(support[5]),
        "truth_pairs_supported_neither": int(support[6]),
        "truth_pairs_matching_recent_dominant_garment": int(support[7]),
        "exact_repurchase_truth_pairs": int(support[8]),
    }
    share_fields = {
        "truth_pairs_with_any_84d_garment_support": "share_with_any_84d_garment_support",
        "truth_pairs_matching_full_dominant_garment": "share_matching_full_dominant_garment",
        "truth_pairs_matching_secondary_historical_garment": "share_matching_secondary_historical_garment",
        "truth_pairs_supported_in_recent_and_older": "share_supported_in_recent_and_older",
        "truth_pairs_supported_recent_only": "share_supported_recent_only",
        "truth_pairs_supported_older_only": "share_supported_older_only",
        "truth_pairs_supported_neither": "share_supported_neither",
        "truth_pairs_matching_recent_dominant_garment": "share_matching_recent_dominant_garment",
        "exact_repurchase_truth_pairs": "exact_repurchase_share",
    }
    for count_field, share_field in share_fields.items():
        temporal[share_field] = temporal[count_field] / truth_pairs if truth_pairs else 0.0

    result = {
        "window": name,
        "cutoff": cutoff,
        "population": {
            "candidate_rows": int(candidate_shape[0]),
            "candidate_users": int(candidate_shape[1]),
            "candidate_rank_min": int(candidate_shape[2]),
            "candidate_rank_max": int(candidate_shape[3]),
            "warm_truth_users": truth_users,
            "warm_truth_pairs": truth_pairs,
        },
        "current_union": {
            "warm_truth_pairs_hit": current_hits,
            "warm_candidate_recall": current_hits / truth_pairs if truth_pairs else 0.0,
            "warm_truth_users_hit": hit_users,
            "warm_hit_rate": hit_users / truth_users if truth_users else 0.0,
            "warm_truth_pairs_missed": missed,
            "candidate_rows_with_two_or_more_sources": int(candidate_overlap[1]),
            "candidate_row_multi_source_rate": candidate_overlap[1] / candidate_overlap[0] if candidate_overlap[0] else 0.0,
            "item2vec_candidate_rows": int(candidate_overlap[2]),
            "item2vec_rows_overlapping_another_retained_source": int(candidate_overlap[3]),
            "item2vec_only_candidate_rows": int(candidate_overlap[4]),
            "sources": source_metrics,
        },
        "history_distributions": history,
        "history_threshold_coverage": thresholds,
        "category_diversity": categories,
        "clear_multi_interest": {
            "truth_users": int(multi[0] or 0),
            "truth_user_share": (multi[0] or 0) / truth_users if truth_users else 0.0,
            "warm_truth_pairs": int(multi[1] or 0),
            "warm_truth_pair_share": (multi[1] or 0) / truth_pairs if truth_pairs else 0.0,
            "current_missed_secondary_mode_truth_pairs": missed_secondary,
            "secondary_mode_share_of_current_misses": missed_secondary / missed if missed else 0.0,
            "current_missed_older_only_mode_truth_pairs": missed_old_only,
            "older_only_mode_share_of_current_misses": missed_old_only / missed if missed else 0.0,
        },
        "temporal": temporal,
        "same_day_baskets": {
            "distinct_article_count_distribution": basket_dist,
            "user_day_baskets": int(basket_summary[0]),
            "multi_article_user_day_baskets": int(basket_summary[1]),
            "multi_article_user_day_share": basket_summary[1] / basket_summary[0] if basket_summary[0] else 0.0,
            "history_users": int(basket_summary[2]),
            "users_with_any_multi_article_day": int(basket_summary[3]),
            "user_share_with_any_multi_article_day": basket_summary[3] / basket_summary[2] if basket_summary[2] else 0.0,
            "warm_truth_users_with_any_multi_article_day": int(basket_summary[4]),
            "warm_truth_user_share_with_any_multi_article_day": basket_summary[4] / truth_users if truth_users else 0.0,
            "warm_truth_pairs_from_users_with_any_multi_article_day": int(basket_summary[5] or 0),
            "warm_truth_pair_share_from_users_with_any_multi_article_day": (basket_summary[5] or 0) / truth_pairs if truth_pairs else 0.0,
        },
        "dynamic_interest_count": dynamic_k,
        "proxy_retrieval": proxy,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    con.close()
    return _native(result)


def _mean(windows: dict[str, Any], accessor) -> float:
    values = [float(accessor(window)) for window in windows.values()]
    return float(np.mean(values)) if values else 0.0


def render_report(metrics: dict[str, Any]) -> str:
    windows = metrics["windows"]
    decision = metrics["executive_decision"]
    summary = metrics["decision_summary"]
    go_rule = metrics["decision_rules"]["go"]
    lines = [
        "# MIND Warm-Side Feasibility Audit",
        "",
        "## 1. Executive Decision",
        "",
        f"**{decision}（进入实验路线）**。现在不应直接把 MIND 加入正式 Warm 流水线；只允许按第10节做一次固定配置的检索级最小实验。",
        "",
        "MIND（Multi-Interest Network with Dynamic Routing，多兴趣动态路由网络，行业论文模型）用多个用户向量表示不同兴趣，目标位置是候选召回而不是最终排序。"
        "本报告中的统计对象是四个非最终周窗口的10%固定开发用户；Warm truth 指验证周中、截止日前已出现过的去重用户—商品对。",
        "",
        f"结构证据存在：清晰多兴趣用户贡献的 Warm truth 平均占 `{summary['clear_multi_interest_truth_share_mean']:.2%}`，"
        f"当前候选池遗漏中平均 `{summary['secondary_mode_share_of_current_misses_mean']:.2%}` 来自这些用户的非主导历史服装组。"
        f"但廉价多画像代理相对现有候选池的平均新增 Recall 只有 `{summary['multi_profile_marginal_recall_mean']:.6f}`，"
        f"其中真正超出单主画像的平均 Recall 为 `{summary['multi_profile_unique_over_single_mean']:.6f}`。",
        "同预算的单主画像代理总新增 Recall 反而更高；因此 GO 的依据是次级兴趣存在稳定独有命中，不是多画像代理已经战胜简单方法。"
        "MIND 实验必须把单主画像列为强制对照。",
        "",
        "项目级条件是：MIND 必须在完全相同用户与 Warm truth 分母上，证明相对现有候选并集和本报告的多画像代理都有稳定独有命中；"
        "否则不进入 LightGBM，更不能因为 standalone Recall 较高就晋级。",
        "",
        "| 预注册判据 | GO门槛 | 实测 |",
        "|---|---:|---:|",
        f"| 清晰多兴趣用户贡献的Warm truth占比 | ≥{go_rule['clear_multi_interest_truth_share_mean_min']:.0%} | {summary['clear_multi_interest_truth_share_mean']:.2%} |",
        f"| 当前miss中清晰多兴趣用户次级模式占比 | ≥{go_rule['secondary_mode_share_of_current_misses_mean_min']:.0%} | {summary['secondary_mode_share_of_current_misses_mean']:.2%} |",
        f"| 多画像代理平均新增Recall | ≥{go_rule['multi_profile_marginal_recall_mean_min']:.3f} | {summary['multi_profile_marginal_recall_mean']:.6f} |",
        f"| 多画像代理有新增命中的窗口 | 4窗中至少{go_rule['multi_profile_marginal_recall_positive_windows_min']:.0f}窗 | {summary['multi_profile_positive_windows']}/4 |",
        f"| 多画像代理超出单画像的平均Recall | ≥{go_rule['multi_profile_unique_over_single_mean_min']:.4f} | {summary['multi_profile_unique_over_single_mean']:.6f} |",
        "",
        "## 2. Current Warm-Side Baseline",
        "",
        "当前候选层由六路启发式召回组成：复购、近期热门、商品家族、用户—购买日共现、年龄分群热门、结构化属性内容；"
        "随后追加 Item2Vec-only 候选，单用户总预算100–300。Item2Vec 使用最近5个历史商品分别检索邻居后合并，因此已经部分保留多个兴趣种子，并非单用户向量。",
        "",
        "最新独立 Warm-v3 分支只读证据中的 WV3-741 沿用同一候选池，在基础 LightGBM/BPR 融合后做局部点式模型与 LambdaRank 名次融合；"
        "四窗 mean MAP@12 为0.028172。它没有新增 MIND 类召回。",
        "",
        "M3.1 的固定池失效审计显示，移除 Item2Vec 会让候选 Recall 平均下降0.020886、Oracle MAP@12下降0.021029，"
        "但最终 MAP@12 只下降0.000018且跨窗翻转。这说明当前同时存在召回缺口和更严重的排序兑现缺口。",
        "",
        "| 窗口 | Warm truth对 | 当前命中对 | 当前Recall | HitRate | 候选行 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in windows.items():
        p, c = row["population"], row["current_union"]
        lines.append(
            f"| {name} | {p['warm_truth_pairs']:,} | {c['warm_truth_pairs_hit']:,} | "
            f"{c['warm_candidate_recall']:.6f} | {c['warm_hit_rate']:.6f} | {p['candidate_rows']:,} |"
        )
    lines += [
        "",
        "下表的召回规模只统计已经进入当前100–300候选池的用户—商品行，不是每一路生成器的完整原始输出；重叠率的分母是该路保留下来的候选行。"
        "独有truth指一条已命中Warm truth在这七个保留来源标记中只由该路支持。M3.1 outage delta是冻结模型遭遇单路失效后的MAP变化，不是可相加的因果贡献。",
        "",
        "| 召回路 | 每用户平均保留候选 | 与其他路重叠率 | 保留池Warm Recall | 独有Warm Recall | M3.1 outage mean MAP delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    outage_delta = {
        "repurchase": -0.003951,
        "recent_popularity": -0.000033,
        "product_family": -0.002213,
        "user_day_covisit": -0.000581,
        "age_popularity": -0.000072,
        "attribute_content": -0.000017,
        "item2vec": -0.000018,
    }
    source_labels = {
        "repurchase": "复购",
        "recent_popularity": "近期热门",
        "product_family": "商品家族",
        "user_day_covisit": "用户—购买日共现",
        "age_popularity": "年龄分群热门",
        "attribute_content": "结构化属性内容",
        "item2vec": "Item2Vec",
    }
    for source in SOURCE_COLUMNS:
        avg_size = _mean(windows, lambda x, s=source: x["current_union"]["sources"][s]["mean_retained_candidates_per_all_candidate_users"])
        avg_overlap = _mean(windows, lambda x, s=source: x["current_union"]["sources"][s]["retained_candidate_overlap_rate"])
        avg_recall = _mean(windows, lambda x, s=source: x["current_union"]["sources"][s]["recall_on_warm_truth"])
        avg_exclusive = _mean(
            windows,
            lambda x, s=source: x["current_union"]["sources"][s]["exclusive_supported_warm_truth_pairs_in_current_union"]
            / x["population"]["warm_truth_pairs"],
        )
        lines.append(
            f"| {source_labels[source]} | {avg_size:.2f} | {avg_overlap:.2%} | {avg_recall:.6f} | {avg_exclusive:.6f} | {outage_delta[source]:+.6f} |"
        )
    lines += [
        "",
        "## 3. User History Distribution",
        "",
        "有效交互（本项目审计定义）是一个去重用户—自然日—商品三元组；它仅用于 MIND 输入统计，原始重复交易没有被删除。主历史窗口为截止日前84天。",
        "",
        "| 窗口 | Warm用户 | mean | P25 | P50 | P75 | P90 | P95 | P99 | full-history P50 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in windows.items():
        d = row["history_distributions"]["effective_interactions_84d"]
        full = row["history_distributions"]["effective_interactions_full"]
        lines.append(
            f"| {name} | {row['population']['warm_truth_users']:,} | {d['mean']:.2f} | {d['p25']:.1f} | {d['p50']:.1f} | "
            f"{d['p75']:.1f} | {d['p90']:.1f} | {d['p95']:.1f} | {d['p99']:.1f} | {full['p50']:.1f} |"
        )
    lines += [
        "",
        "阈值覆盖的用户分母是该窗全部 Warm truth 用户，truth分母是该窗全部去重 Warm 用户—商品真值对：",
        "",
        "| 窗口 | ≥5用户 | ≥5 truth | ≥10用户 | ≥10 truth | ≥20用户 | ≥20 truth | ≥50用户 | ≥50 truth |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in windows.items():
        t = row["history_threshold_coverage"]
        lines.append(
            f"| {name} | {t['5']['truth_user_share']:.2%} | {t['5']['warm_truth_pair_share']:.2%} | "
            f"{t['10']['truth_user_share']:.2%} | {t['10']['warm_truth_pair_share']:.2%} | "
            f"{t['20']['truth_user_share']:.2%} | {t['20']['warm_truth_pair_share']:.2%} | "
            f"{t['50']['truth_user_share']:.2%} | {t['50']['warm_truth_pair_share']:.2%} |"
        )
    lines += [
        "",
        "长历史用户不是少数，但所有稀疏用户必须退化为单兴趣；不能强行为每个人生成固定数量的 capsule（兴趣向量）。",
        "",
        "## 4. Evidence for / against Multi-Interest Structure",
        "",
        "类别熵（行业通用统计）衡量用户在服装组上的交互分布均匀度；归一化熵用用户实际出现的类别数取对数归一化。"
        "“清晰多兴趣”是本项目预注册标签：84天有效交互≥10、服装组≥3、归一化熵≥0.6、最大服装组占比≤0.7。它是诊断分组，不是真实潜变量标签。",
        "",
        "| 窗口 | 清晰多兴趣用户占比 | 对应Warm truth占比 | garment熵中位数 | garment Top1占比中位数 | 当前miss中的次级模式占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in windows.items():
        m = row["clear_multi_interest"]
        g = row["category_diversity"]["garment_group"]
        lines.append(
            f"| {name} | {m['truth_user_share']:.2%} | {m['warm_truth_pair_share']:.2%} | "
            f"{g['normalized_entropy']['p50']:.4f} | {g['top1_share']['p50']:.4f} | {m['secondary_mode_share_of_current_misses']:.2%} |"
        )
    lines += [
        "",
        "这些数字支持“历史中有多个类别模式”，但不能单独证明胶囊路由优于 Item2Vec 多种子或简单类别画像。关键是下面的未来行为与增量召回。",
        "",
        "三种类别层级的四窗均值如下；服装组是主判据，section/product group只做层级敏感性检查：",
        "",
        "| 类别层级 | 唯一类别数中位数均值 | 归一化熵中位数均值 | Top1占比中位数均值 |",
        "|---|---:|---:|---:|",
    ]
    level_labels = {"garment_group": "服装组", "section": "陈列区/业务区", "product_group": "商品大类"}
    for level in INTEREST_LEVELS:
        lines.append(
            f"| {level_labels[level]} | "
            f"{_mean(windows, lambda x, l=level: x['category_diversity'][l]['unique_categories']['p50']):.3f} | "
            f"{_mean(windows, lambda x, l=level: x['category_diversity'][l]['normalized_entropy']['p50']):.4f} | "
            f"{_mean(windows, lambda x, l=level: x['category_diversity'][l]['top1_share']['p50']):.4f} |"
        )
    lines += [
        "",
        "## 5. Temporal Characteristics",
        "",
        "JSD（Jensen–Shannon divergence，行业通用分布距离，0表示两段完全一致、1表示完全分离）比较最近0–28天和较早29–84天的服装组分布。"
        "同日篮子指一个用户在同一自然日购买的不同商品集合；数据没有日内顺序。",
        "",
        "| 窗口 | recent/older JSD中位数 | truth匹配近期主导组 | truth来自较早独有组 | 多商品购买日占比 | truth用户曾有多商品日 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in windows.items():
        temporal, baskets = row["temporal"], row["same_day_baskets"]
        lines.append(
            f"| {name} | {temporal['recent_vs_older_garment_jsd']['p50']:.4f} | "
            f"{temporal['share_matching_recent_dominant_garment']:.2%} | {temporal['share_supported_older_only']:.2%} | "
            f"{baskets['multi_article_user_day_share']:.2%} | {baskets['warm_truth_user_share_with_any_multi_article_day']:.2%} |"
        )
    lines += [
        "",
        "结论是弱顺序数据确实更适合集合式兴趣抽取，而不是伪造同日序列；但较明显的 recent-vs-old 漂移也说明 vanilla MIND 不能替代时间感知。"
        "现有 Warm-v3 购买日 GRU 在内层有信号、外层融合失败；非参数最大购买日匹配的用户内AUC又四窗低于 GRU。MIND 的最小实验必须同时保留时间衰减。",
        "",
        f"跨四窗平均，Warm truth 中约 `{_mean(windows, lambda x: x['temporal']['share_matching_secondary_historical_garment']):.2%}` 匹配84天历史中的非主导服装组，"
        f"而只有约 `{_mean(windows, lambda x: x['temporal']['share_matching_recent_dominant_garment']):.2%}` 匹配最近28天主导服装组。"
        "这支持保留多个历史兴趣，但较早独有模式占比和高JSD也要求显式时间衰减，不能把历史无差别聚类。",
        "",
        "## 6. Expected Incremental Value vs Existing Recall",
        "",
        "多画像代理（本项目自定义、非神经检索）：取用户84天内前三个服装组，每组至少2次有效交互，再从截止日前28天该组热门商品中各取20件；"
        "最多60件。单画像对照只用84天主导服装组取60件，近期画像对照只用最近28天主导服装组取60件。",
        "",
        "| 窗口 | 当前Recall | 单画像新增Recall | 近期画像新增Recall | 多画像新增Recall | 多画像独有于单画像 | 次级模式新增truth对 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in windows.items():
        p = row["proxy_retrieval"]
        lines.append(
            f"| {name} | {row['current_union']['warm_candidate_recall']:.6f} | "
            f"{p['single_profile']['marginal_recall_over_current_union']:.6f} | "
            f"{p['recent_profile']['marginal_recall_over_current_union']:.6f} | "
            f"{p['multi_profile']['marginal_recall_over_current_union']:.6f} | "
            f"{p['multi_profile']['marginal_recall_unique_over_single_profile']:.6f} | "
            f"{p['multi_profile']['marginal_truth_pairs_from_secondary_modes']:,} |"
        )
    lines += [
        "",
        "如果多画像只重复单画像和 Item2Vec，多兴趣结构便没有工程增量；如果它稳定找到次级模式 truth，则 MIND 才有明确可检验目标。"
        "代理使用粗类别与流行度，只是低成本下界，不代表 MIND 的可达到上限。",
        "",
        f"四窗平均，单主画像新增 Recall 为 `{_mean(windows, lambda x: x['proxy_retrieval']['single_profile']['marginal_recall_over_current_union']):.6f}`，"
        f"高于多画像的 `{summary['multi_profile_marginal_recall_mean']:.6f}`。这说明固定20/20/20配额会牺牲主兴趣覆盖；"
        "MIND 只有在学习型路由能够保留主兴趣的同时找回次级兴趣时才有价值。",
        "",
        "## 7. Leakage-Safe Training Protocol",
        "",
        "建议沿用四条滚动链，每个目标窗口都重新构造截止安全输入：",
        "",
        "```text",
        "训练样本日 d：仅使用 d 之前最多84天的用户—购买日集合",
        "        ↓",
        "标签：d 当天去重商品集合；同一天每件商品各作一个正例，不制造日内顺序",
        "        ↓",
        "MIND：动态兴趣向量 + label-aware interest selection + sampled softmax",
        "        ↓",
        "验证截止 c：只使用 c 之前历史编码用户，检索截止前已出现的 Warm catalog",
        "        ↓",
        "每兴趣TopN → 合并去重 → 与冻结现有候选池比较 Recall/Hit/独有truth",
        "```",
        "",
        "label-aware attention（标签感知兴趣选择，MIND论文机制）只能在训练损失里用当天目标商品选择对应兴趣；验证检索时不能读取未来商品。"
        "未购买商品只是 sampled unobserved item（抽样未观察商品），不是曝光后负反馈。",
        "",
        "## 8. Compute & Engineering Cost",
        "",
        "H&M 约10.5万商品，64维 float32 商品表约26 MiB；单个4090足够训练最小模型，四卡没有必要。"
        "建议最多三组：单兴趣 learned baseline、MIND Kmax=3、MIND去掉时间衰减消融。",
        "",
        "| 阶段 | 估计单窗成本 | 资源 |",
        "|---|---:|---|",
        "| 去重购买日数据与样本索引 | 10–25分钟 | CPU 8–16核，内存约8–16GB |",
        "| 单个MIND配置训练 | 30–90分钟 | 单张4090，预计4–8GB显存 |",
        "| 10%开发用户向量与候选生成 | 5–15分钟 | 单GPU批量矩阵乘；可先不用ANN |",
        "| 四窗固定回放 | 约3–8小时 | 可缓存每个cutoff的商品表和样本索引 |",
        "",
        "ANN（近似最近邻，行业通用检索）在10%开发用户 MVE 中不是必需；全量约137万用户乘多个兴趣时才值得引入 FAISS 等索引。"
        "本估算基于现有 Item2Vec/购买日GRU实测成本与模型规模外推，不是已完成的 MIND benchmark。",
        "",
        "动态兴趣数（本项目实现口径）为 `floor(log2(84天有效交互数))`，再限制到1至Kmax。以下分母是各窗全部Warm truth用户：",
        "",
        "| 窗口 | Kmax=3时K=1 | K=2 | K=3 | Kmax=4时K=4 | Kmax=5时K=5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in windows.items():
        k = row["dynamic_interest_count"]
        lines.append(
            f"| {name} | {k['3']['distribution_user_share']['1']:.2%} | {k['3']['distribution_user_share']['2']:.2%} | "
            f"{k['3']['distribution_user_share']['3']:.2%} | {k['4']['distribution_user_share']['4']:.2%} | "
            f"{k['5']['distribution_user_share']['5']:.2%} |"
        )
    lines += [
        "",
        "Kmax=3 已允许约28%–37%的用户获得3个兴趣，而约44%–54%的用户仍自动退化到1个；Kmax=5真正得到5个兴趣的用户仅约2%–4%。"
        "结合≥20交互的truth占比只有约10%–16%，更大的K更可能产生冗余或噪声，因此MVE固定Kmax=3。",
        "",
        "## 9. Alternatives Considered",
        "",
        "1. 类别多画像 + 类内热门：最便宜，本报告已直接测；若与 MIND 接近，应优先保留它。",
        "2. 对最近历史商品做 embedding 聚类，再按每个聚类中心检索：比动态路由简单，能更直接检验多个向量是否有价值。",
        "3. 现有 Item2Vec 多种子：已经是强重叠对照；MIND 必须报告超出它的独有truth。",
        "4. 购买日 GRU / SASRec：更适合兴趣演化，但 H&M 缺少日内顺序；现有 GRU 外层未稳定晋级，因此不直接扩大序列模型。",
        "",
        "## 10. Minimum Viable Experiment",
        "",
        "固定一个主实验，不做大范围搜索：",
        "",
        "- 单兴趣对照：同一 item embedding、同一训练样本、一个用户向量；",
        "- MIND：64维，Kmax=3，最多50个最近购买日商品，3轮动态路由，28天半衰期；历史不足5个有效交互时K=1；",
        "- 必要消融：MIND去掉时间衰减，区分多兴趣收益与近期加权收益；",
        "- 每个兴趣先取Top50，合并后按兴趣分数去重并限制 MIND 通道最多100件；",
        "- 先在一个历史 inner→validation 链完成 mechanics，再冻结配置跑其余四个开发窗口；",
        "- 只有检索门槛通过，才在完全相同扩展候选池上重训 LightGBM，并报告最终 MAP@12。",
        "",
        "## 11. Success / Kill Criteria",
        "",
        "检索晋级条件预注册为：",
        "",
        "- 相对当前候选并集，四窗平均新增 Warm Recall≥0.003，至少3/4窗口为正；",
        "- 相对本报告多画像代理，四窗平均独有 Recall≥0.001；",
        "- 至少20%的新增 truth 来自非主导历史服装组，且不是只集中在一个窗口；",
        "- MIND通道与现有并集候选行重叠率≤85%，并保持每用户最多100件；",
        "- 单窗训练≤90分钟、峰值显存≤10GB。",
        "",
        "任一核心增量条件失败则停止 MIND；不调 K、TopN、负采样比例或路由轮数救同一批开发窗口。即使检索通过，"
        "LightGBM 后四窗 mean MAP@12 必须提高，至少3/4不退化，最差窗口不得低于对照0.0005，才允许成为正式 Warm 通道。",
        "",
        "## 12. Recommendation",
        "",
        f"结论为 **{decision}**。",
        "",
        "决定条件不是“MIND是否经典”，而是固定 Kmax=3 的模型能否在排序之前产生跨窗独有 Warm truth。"
        "当前候选 Recall远未饱和，但 Item2Vec已经贡献大量未兑现 Oracle，故 MIND 只允许作为低成本检索假设测试；"
        "在它通过检索门槛前，不值得支付候选特征重建和 LightGBM 全链路成本。",
        "",
        "外部方法依据：MIND原论文将多个兴趣向量用于 matching/召回阶段，并用动态路由抽取兴趣、标签感知机制选择与目标相符的兴趣："
        "https://arxiv.org/abs/1904.08030 。本项目没有复现论文实验，也没有把论文结果当作本地增益证据。",
        "",
        "最终周 `2020-09-16`：`not_run`。本轮没有训练模型、没有生成最终提交、没有改变现有候选或排序器。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict[str, Any]:
    started = time.perf_counter()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract["final_week"]["status"] != "not_run":
        raise AssertionError("final week must remain not_run")
    missing = [str(path) for path in [TX, ARTICLES, *CANDIDATES.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing audit inputs: {missing}")
    windows: dict[str, Any] = {}
    for name, cutoff in WINDOWS.items():
        print({"mind_warm_audit": name, "cutoff": cutoff}, flush=True)
        windows[name] = audit_window(name, cutoff, CANDIDATES[name])
        print(
            {
                "window": name,
                "warm_recall": windows[name]["current_union"]["warm_candidate_recall"],
                "multi_truth_share": windows[name]["clear_multi_interest"]["warm_truth_pair_share"],
                "proxy_marginal_recall": windows[name]["proxy_retrieval"]["multi_profile"]["marginal_recall_over_current_union"],
            },
            flush=True,
        )
    summary = {
        "clear_multi_interest_truth_share_mean": _mean(windows, lambda x: x["clear_multi_interest"]["warm_truth_pair_share"]),
        "secondary_mode_share_of_current_misses_mean": _mean(windows, lambda x: x["clear_multi_interest"]["secondary_mode_share_of_current_misses"]),
        "multi_profile_marginal_recall_mean": _mean(windows, lambda x: x["proxy_retrieval"]["multi_profile"]["marginal_recall_over_current_union"]),
        "multi_profile_positive_windows": sum(
            row["proxy_retrieval"]["multi_profile"]["marginal_truth_pairs_over_current_union"] > 0
            for row in windows.values()
        ),
        "multi_profile_unique_over_single_mean": _mean(
            windows, lambda x: x["proxy_retrieval"]["multi_profile"]["marginal_recall_unique_over_single_profile"]
        ),
        "current_warm_candidate_recall_mean": _mean(windows, lambda x: x["current_union"]["warm_candidate_recall"]),
    }
    decision = decide(summary, contract["decision_rules"])
    metrics = {
        "schema_version": "mind-warm-side-feasibility-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_id": contract["audit_id"],
        "executive_decision": decision,
        "decision_summary": summary,
        "decision_rules": contract["decision_rules"],
        "windows": windows,
        "evidence_scope": contract["evidence_boundary"],
        "model_training": "not_run",
        "final_week": "not_run",
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json(METRICS_PATH, metrics)
    REPORT_PATH.write_text(render_report(metrics), encoding="utf-8", newline="\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    args = parser.parse_args()
    if args.command == "run":
        result = run()
        print(
            {
                "decision": result["executive_decision"],
                "runtime_seconds": result["runtime_seconds"],
                "report": str(REPORT_PATH),
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
