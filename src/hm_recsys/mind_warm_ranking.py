"""Conservative LightGBM realization test for the promoted MIND Warm retrieval source."""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from . import mind_warm_mve


ROOT = Path(__file__).resolve().parents[2]
TX = ROOT / "data" / "interim" / "audit" / "transactions.parquet"
RAW = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports" / "mind_warm_side"
CONTRACT_PATH = REPORT_DIR / "MIND_WARM_RANKING_CONTRACT.json"
METRICS_PATH = REPORT_DIR / "MIND_WARM_RANKING_METRICS.json"
REPORT_PATH = REPORT_DIR / "MIND_WARM_RANKING_FINAL.md"
PARENT_METRICS_PATH = REPORT_DIR / "MIND_WARM_MVE_METRICS.json"
PARENT_REPORT_PATH = REPORT_DIR / "MIND_WARM_MVE_FINAL.md"
ARTIFACT_ROOT = ROOT / "artifacts" / "mind_warm_side" / "MIND-WARM-RANK-001"
WARM_ROOT = Path(__file__).resolve().parents[2]

TRAIN_CUTOFF = "2019-11-27"
INNER = {
    "winter_20200122": "2019-12-25",
    "spring_20200318": "2020-02-19",
    "early_summer_20200624": "2020-05-27",
    "late_summer_20200819": "2020-07-22",
}
OUTER = dict(mind_warm_mve.OUTER)
MIND_VARIANT = "mind_learned_age"
SCHEMA = "mind-warm-ranking-v1"

MIND_FEATURES = [
    "mind_present",
    "mind_is_new",
    "mind_rank",
    "mind_rank_fraction",
    "mind_reciprocal_rank",
    "mind_score",
    "mind_score_user_z",
    "mind_best_interest",
    "mind_candidate_interest_count",
    "mind_user_interest_count",
]
NON_FEATURE_COLUMNS = {"customer_id", "article_id", "target"}
MODEL_PARAMS = {
    "objective": "lambdarank",
    "metric": ["map", "ndcg"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "seed": 20260912,
    "feature_fraction_seed": 20260912,
    "bagging_seed": 20260912,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 8,
    "verbosity": -1,
    "eval_at": [12],
    "lambdarank_truncation_level": 20,
}
ROUNDS = 250


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_native(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _guard(cutoff: str) -> None:
    if date.fromisoformat(cutoff) >= date(2020, 9, 16):
        raise ValueError("final week 2020-09-16 and later are forbidden")


def _connection(memory: str = "10GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute(f"SET memory_limit='{memory}'")
    return con


def _base_candidates(cutoff: str) -> Path:
    return ROOT / "artifacts" / "m2_9" / "cache-v1" / cutoff / "expanded-candidates.parquet"


def _mind_candidates(cutoff: str) -> Path:
    return (
        ROOT / "artifacts" / "mind_warm_side" / "MIND-WARM-MVE-001" /
        cutoff / MIND_VARIANT / "candidates.parquet"
    )


def _paths(cutoff: str, output_root: Path | None = None) -> tuple[Path, Path, Path, Path]:
    root = (ARTIFACT_ROOT if output_root is None else Path(output_root)) / cutoff
    return (
        root / "augmented-candidates.parquet",
        root / "AUGMENTED.json",
        root / "features.parquet",
        root / "FEATURES.json",
    )


def ensure_mind_candidates(cutoff: str, device: str) -> tuple[Path, dict[str, Any]]:
    _guard(cutoff)
    parent = _read_json(mind_warm_mve.CONTRACT_PATH)
    path, meta = mind_warm_mve.retrieve_variant(cutoff, MIND_VARIANT, parent, device)
    return path, meta


def build_augmented_candidates(
    cutoff: str, device: str, output_root: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Build the same candidate union, optionally under an isolated experiment root."""
    _guard(cutoff)
    output, marker, _, _ = _paths(cutoff, output_root)
    if output.exists() and marker.exists():
        saved = _read_json(marker)
        if saved.get("schema") != SCHEMA or saved.get("cutoff") != cutoff:
            raise AssertionError(f"stale augmented-candidate cache: {cutoff}")
        return output, saved
    base = _base_candidates(cutoff)
    if not base.exists():
        raise FileNotFoundError(base)
    mind, mind_meta = ensure_mind_candidates(cutoff, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with _connection("6GB") as con:
        con.execute(f"CREATE VIEW base AS SELECT * FROM read_parquet('{_sql_path(base)}')")
        con.execute(f"CREATE VIEW mind AS SELECT * FROM read_parquet('{_sql_path(mind)}')")
        schema = con.execute("DESCRIBE SELECT * FROM base").fetchall()
        columns = [row[0] for row in schema]
        types = {row[0]: row[1] for row in schema}
        required = {"customer_id", "article_id", "candidate_rank"}
        if not required.issubset(columns):
            raise ValueError(f"base candidate schema misses {sorted(required - set(columns))}")
        base_select = ",".join(f"b.{name}" for name in columns)
        new_base_columns = []
        for name in columns:
            if name == "customer_id":
                new_base_columns.append("m.customer_id")
            elif name == "article_id":
                new_base_columns.append("m.article_id")
            elif name == "candidate_rank":
                new_base_columns.append(
                    "(n.base_rows+row_number() OVER(PARTITION BY m.customer_id "
                    "ORDER BY m.mind_rank,m.article_id))::INTEGER AS candidate_rank"
                )
            else:
                new_base_columns.append(f"CAST(0 AS {types[name]}) AS {name}")
        new_base_select = ",".join(new_base_columns)
        con.execute(
            """CREATE TEMP TABLE mind_stats AS
               SELECT customer_id,count(*)::DOUBLE AS mind_count,
                      avg(mind_score)::DOUBLE AS mean_score,
                      stddev_pop(mind_score)::DOUBLE AS std_score,
                      max(interest_count)::INTEGER AS user_interest_count
               FROM mind GROUP BY customer_id"""
        )
        query = f"""
            WITH base_n AS (
                SELECT customer_id,count(*)::INTEGER AS base_rows FROM base GROUP BY customer_id
            ), base_rows AS (
                SELECT {base_select},
                       (m.article_id IS NOT NULL)::UTINYINT AS mind_present,
                       0::UTINYINT AS mind_is_new,
                       coalesce(m.mind_rank,0)::INTEGER AS mind_rank,
                       coalesce(m.mind_rank/s.mind_count,0.0)::DOUBLE AS mind_rank_fraction,
                       coalesce(1.0/(60.0+m.mind_rank),0.0)::DOUBLE AS mind_reciprocal_rank,
                       coalesce(m.mind_score,0.0)::DOUBLE AS mind_score,
                       CASE WHEN m.article_id IS NULL OR coalesce(s.std_score,0)=0 THEN 0.0
                            ELSE (m.mind_score-s.mean_score)/s.std_score END::DOUBLE AS mind_score_user_z,
                       coalesce(m.best_interest,0)::INTEGER AS mind_best_interest,
                       coalesce(m.interest_count,0)::INTEGER AS mind_candidate_interest_count,
                       coalesce(s.user_interest_count,0)::INTEGER AS mind_user_interest_count
                FROM base b
                LEFT JOIN mind m USING(customer_id,article_id)
                LEFT JOIN mind_stats s USING(customer_id)
            ), new_rows AS (
                SELECT {new_base_select},
                       1::UTINYINT AS mind_present,
                       1::UTINYINT AS mind_is_new,
                       m.mind_rank::INTEGER AS mind_rank,
                       (m.mind_rank/s.mind_count)::DOUBLE AS mind_rank_fraction,
                       (1.0/(60.0+m.mind_rank))::DOUBLE AS mind_reciprocal_rank,
                       m.mind_score::DOUBLE AS mind_score,
                       CASE WHEN coalesce(s.std_score,0)=0 THEN 0.0
                            ELSE (m.mind_score-s.mean_score)/s.std_score END::DOUBLE AS mind_score_user_z,
                       m.best_interest::INTEGER AS mind_best_interest,
                       m.interest_count::INTEGER AS mind_candidate_interest_count,
                       s.user_interest_count::INTEGER AS mind_user_interest_count
                FROM mind m
                JOIN base_n n USING(customer_id)
                JOIN mind_stats s USING(customer_id)
                ANTI JOIN base b USING(customer_id,article_id)
            )
            SELECT * FROM base_rows UNION ALL BY NAME SELECT * FROM new_rows
            ORDER BY customer_id,candidate_rank,article_id
        """
        con.execute(f"COPY ({query}) TO '{_sql_path(output)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        rows = int(con.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(output)}')").fetchone()[0])
        unique_rows = int(con.execute(
            f"SELECT count(DISTINCT (customer_id,article_id)) FROM read_parquet('{_sql_path(output)}')"
        ).fetchone()[0])
        group_stats = con.execute(
            f"""WITH g AS (SELECT customer_id,count(*) n,min(candidate_rank) lo,
                                   max(candidate_rank) hi,count(DISTINCT candidate_rank) dr
                            FROM read_parquet('{_sql_path(output)}') GROUP BY customer_id)
                SELECT count(*),min(n),max(n),count(*) FILTER(WHERE n<100 OR n>400 OR lo<>1 OR hi<>n OR dr<>n)
                FROM g"""
        ).fetchone()
        source_stats = con.execute(
            f"SELECT sum(mind_is_new),sum(mind_present) FROM read_parquet('{_sql_path(output)}')"
        ).fetchone()
    if rows != unique_rows or int(group_stats[3]):
        raise RuntimeError(f"invalid augmented candidate identity at {cutoff}")
    result = {
        "schema": SCHEMA,
        "cutoff": cutoff,
        "base": str(base.resolve()),
        "mind": str(mind.resolve()),
        "mind_model": mind_meta["model"],
        "rows": rows,
        "users": int(group_stats[0]),
        "minimum_group_rows": int(group_stats[1]),
        "maximum_group_rows": int(group_stats[2]),
        "mind_only_rows": int(source_stats[0]),
        "mind_present_rows": int(source_stats[1]),
        "identity_unique": True,
        "rank_contiguous": True,
        "elapsed_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    _write_json(marker, result)
    return output, result


def _dimension_sql(table: str, key: str, cutoff: str) -> str:
    return f"""
        CREATE TEMP TABLE {table} AS
        SELECT customer_id,{key},
               count(*) FILTER(WHERE t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY)::BIGINT AS events_28d,
               count(*)::BIGINT AS events_12w,
               date_diff('day',max(t_dat),DATE '{cutoff}')::BIGINT AS days_since
        FROM user_history GROUP BY customer_id,{key}
    """


def build_features(
    cutoff: str, device: str, output_root: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    """Reuse cutoff-safe aggregation while keeping optional new caches isolated."""
    _guard(cutoff)
    candidates, candidate_meta = build_augmented_candidates(cutoff, device, output_root)
    _, _, output, marker = _paths(cutoff, output_root)
    if output.exists() and marker.exists():
        saved = _read_json(marker)
        if saved.get("schema") != SCHEMA or saved.get("cutoff") != cutoff:
            raise AssertionError(f"stale ranking feature cache: {cutoff}")
        return output, saved
    started = time.perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with _connection("6GB" if output_root is not None else "10GB") as con:
        con.execute(f"CREATE VIEW candidates AS SELECT * FROM read_parquet('{_sql_path(candidates)}')")
        con.execute("CREATE TEMP TABLE users AS SELECT DISTINCT customer_id FROM candidates")
        con.execute("CREATE TEMP TABLE items AS SELECT DISTINCT article_id FROM candidates")
        con.execute(
            f"""CREATE TEMP TABLE articles AS
                SELECT article_id,try_cast(product_code AS INTEGER) article_product_code,
                       try_cast(product_type_no AS INTEGER) article_product_type_no,
                       try_cast(garment_group_no AS INTEGER) article_garment_group_no,
                       try_cast(department_no AS INTEGER) article_department_no,
                       try_cast(index_group_no AS INTEGER) article_index_group_no,
                       try_cast(perceived_colour_master_id AS INTEGER) article_colour_master_id
                FROM read_csv_auto('{_sql_path(RAW / 'articles.csv')}',header=true,all_varchar=true)"""
        )
        con.execute(
            f"""CREATE TEMP TABLE customers AS
                SELECT customer_id,try_cast(age AS DOUBLE) customer_age
                FROM read_csv_auto('{_sql_path(RAW / 'customers.csv')}',header=true,all_varchar=true)"""
        )
        con.execute(
            f"""CREATE TEMP TABLE global_history AS
                SELECT * FROM read_parquet('{_sql_path(TX)}')
                WHERE t_dat>=DATE '{cutoff}'-INTERVAL 84 DAY AND t_dat<DATE '{cutoff}'"""
        )
        con.execute(
            """CREATE TEMP TABLE user_history AS
               SELECT h.*,a.article_product_code,a.article_product_type_no,
                      a.article_garment_group_no,a.article_department_no,
                      a.article_index_group_no,a.article_colour_master_id
               FROM global_history h SEMI JOIN users USING(customer_id)
               JOIN articles a USING(article_id)"""
        )
        con.execute(
            f"""CREATE TEMP TABLE truth AS
                SELECT DISTINCT t.customer_id,t.article_id
                FROM read_parquet('{_sql_path(TX)}') t SEMI JOIN users USING(customer_id)
                WHERE t.t_dat>=DATE '{cutoff}' AND t.t_dat<DATE '{cutoff}'+INTERVAL 7 DAY"""
        )
        con.execute(
            f"""CREATE TEMP TABLE user_features AS
                SELECT u.customer_id,count(h.article_id)::BIGINT user_history_events_12w,
                       count(DISTINCT h.article_id)::BIGINT user_unique_items_12w,
                       avg(h.price)::DOUBLE user_avg_price_12w,
                       avg((h.sales_channel_id=2)::INTEGER)::DOUBLE user_online_share_12w,
                       date_diff('day',max(h.t_dat),DATE '{cutoff}')::BIGINT user_days_since_last_purchase
                FROM users u LEFT JOIN user_history h USING(customer_id) GROUP BY u.customer_id"""
        )
        con.execute(
            f"""CREATE TEMP TABLE item_features AS
                SELECT i.article_id,
                       count(h.article_id) FILTER(WHERE h.t_dat>=DATE '{cutoff}'-INTERVAL 7 DAY)::BIGINT item_events_7d,
                       count(h.article_id) FILTER(WHERE h.t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY)::BIGINT item_events_28d,
                       count(h.article_id)::BIGINT item_events_12w,
                       count(DISTINCT h.customer_id) FILTER(WHERE h.t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY)::BIGINT item_unique_customers_28d,
                       avg(h.price) FILTER(WHERE h.t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY)::DOUBLE item_avg_price_28d,
                       date_diff('day',max(h.t_dat),DATE '{cutoff}')::BIGINT item_days_since_last_sale
                FROM items i LEFT JOIN global_history h USING(article_id) GROUP BY i.article_id"""
        )
        for table, key in {
            "ui": "article_id",
            "upc": "article_product_code",
            "upt": "article_product_type_no",
            "udp": "article_department_no",
            "ugg": "article_garment_group_no",
            "ucl": "article_colour_master_id",
            "uig": "article_index_group_no",
        }.items():
            con.execute(_dimension_sql(table, key, cutoff))
        query = f"""
            SELECT c.*,
                   cust.customer_age,
                   (cust.customer_age IS NULL)::UTINYINT customer_age_missing,
                   coalesce(floor(cust.customer_age/5),-1)::INTEGER age_bucket,
                   coalesce(uf.user_history_events_12w,0)::BIGINT user_history_events_12w,
                   coalesce(uf.user_unique_items_12w,0)::BIGINT user_unique_items_12w,
                   uf.user_avg_price_12w,uf.user_online_share_12w,uf.user_days_since_last_purchase,
                   coalesce(it.item_events_7d,0)::BIGINT item_events_7d,
                   coalesce(it.item_events_28d,0)::BIGINT item_events_28d,
                   coalesce(it.item_events_12w,0)::BIGINT item_events_12w,
                   coalesce(it.item_unique_customers_28d,0)::BIGINT item_unique_customers_28d,
                   it.item_avg_price_28d,it.item_days_since_last_sale,
                   (coalesce(it.item_events_7d,0)+1.0)/(coalesce(it.item_events_28d,0)/4.0+1.0) item_trend_7d_vs_28d,
                   abs(it.item_avg_price_28d-uf.user_avg_price_12w) user_item_price_gap,
                   coalesce(ui.events_28d,0)::BIGINT user_item_events_28d,
                   coalesce(ui.events_12w,0)::BIGINT user_item_events_12w,
                   ui.days_since user_item_days_since_last_purchase,
                   coalesce(upc.events_28d,0)::BIGINT user_product_code_events_28d,
                   coalesce(upc.events_12w,0)::BIGINT user_product_code_events_12w,
                   upc.days_since user_product_code_days_since,
                   coalesce(upc.events_12w,0)/greatest(uf.user_history_events_12w,1.0) user_product_code_share_12w,
                   coalesce(upt.events_28d,0)::BIGINT user_product_type_events_28d,
                   coalesce(upt.events_12w,0)::BIGINT user_product_type_events_12w,
                   upt.days_since user_product_type_days_since,
                   coalesce(upt.events_12w,0)/greatest(uf.user_history_events_12w,1.0) user_product_type_share_12w,
                   coalesce(udp.events_28d,0)::BIGINT user_department_events_28d,
                   coalesce(udp.events_12w,0)::BIGINT user_department_events_12w,
                   udp.days_since user_department_days_since,
                   coalesce(udp.events_12w,0)/greatest(uf.user_history_events_12w,1.0) user_department_share_12w,
                   coalesce(ugg.events_28d,0)::BIGINT user_garment_events_28d,
                   coalesce(ugg.events_12w,0)::BIGINT user_garment_events_12w,
                   ugg.days_since user_garment_days_since,
                   coalesce(ugg.events_12w,0)/greatest(uf.user_history_events_12w,1.0) user_garment_share_12w,
                   coalesce(ucl.events_28d,0)::BIGINT user_colour_events_28d,
                   coalesce(ucl.events_12w,0)::BIGINT user_colour_events_12w,
                   ucl.days_since user_colour_days_since,
                   coalesce(ucl.events_12w,0)/greatest(uf.user_history_events_12w,1.0) user_colour_share_12w,
                   coalesce(uig.events_28d,0)::BIGINT user_index_group_events_28d,
                   coalesce(uig.events_12w,0)::BIGINT user_index_group_events_12w,
                   uig.days_since user_index_group_days_since,
                   coalesce(uig.events_12w,0)/greatest(uf.user_history_events_12w,1.0) user_index_group_share_12w,
                   a.article_product_code,a.article_product_type_no,a.article_garment_group_no,
                   a.article_department_no,a.article_index_group_no,a.article_colour_master_id,
                   (truth.article_id IS NOT NULL)::UTINYINT AS target
            FROM candidates c JOIN articles a USING(article_id)
            LEFT JOIN customers cust USING(customer_id)
            LEFT JOIN user_features uf USING(customer_id)
            LEFT JOIN item_features it USING(article_id)
            LEFT JOIN ui USING(customer_id,article_id)
            LEFT JOIN upc ON c.customer_id=upc.customer_id AND a.article_product_code=upc.article_product_code
            LEFT JOIN upt ON c.customer_id=upt.customer_id AND a.article_product_type_no=upt.article_product_type_no
            LEFT JOIN udp ON c.customer_id=udp.customer_id AND a.article_department_no=udp.article_department_no
            LEFT JOIN ugg ON c.customer_id=ugg.customer_id AND a.article_garment_group_no=ugg.article_garment_group_no
            LEFT JOIN ucl ON c.customer_id=ucl.customer_id AND a.article_colour_master_id=ucl.article_colour_master_id
            LEFT JOIN uig ON c.customer_id=uig.customer_id AND a.article_index_group_no=uig.article_index_group_no
            LEFT JOIN truth USING(customer_id,article_id)
        """
        con.execute(
            f"COPY ({query} ORDER BY customer_id,candidate_rank,article_id) TO '{_sql_path(output)}' "
            "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
        )
        stats = con.execute(
            f"""SELECT count(*),count(DISTINCT (customer_id,article_id)),count(DISTINCT customer_id),
                       sum(target),count(DISTINCT customer_id) FILTER(WHERE target=1),
                       sum(target) FILTER(WHERE mind_is_new=1)
                FROM read_parquet('{_sql_path(output)}')"""
        ).fetchone()
        latest = con.execute("SELECT max(t_dat) FROM global_history").fetchone()[0]
    if int(stats[0]) != candidate_meta["rows"] or int(stats[0]) != int(stats[1]):
        raise RuntimeError(f"feature materialization changed candidate identity: {cutoff}")
    if latest is not None and str(latest) >= cutoff:
        raise RuntimeError(f"feature leakage detected at {cutoff}")
    result = {
        "schema": SCHEMA,
        "cutoff": cutoff,
        "candidate_source": str(candidates.resolve()),
        "rows": int(stats[0]),
        "users": int(stats[2]),
        "positive_rows": int(stats[3] or 0),
        "positive_user_groups": int(stats[4] or 0),
        "mind_only_positive_rows": int(stats[5] or 0),
        "latest_behavior_date": str(latest),
        "history_strictly_before_cutoff": True,
        "label_window": f"[{cutoff}, {cutoff}+7 days)",
        "bytes": output.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    _write_json(marker, result)
    return output, result


def _feature_names(path: Path) -> tuple[list[str], list[str]]:
    with _connection("2GB") as con:
        columns = [row[0] for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(path)}')"
        ).fetchall()]
    all_features = [name for name in columns if name not in NON_FEATURE_COLUMNS]
    primary = all_features
    control = [name for name in all_features if name not in MIND_FEATURES]
    if not set(MIND_FEATURES).issubset(primary) or set(MIND_FEATURES) & set(control):
        raise AssertionError("MIND feature isolation failed")
    return control, primary


def _load_training(path: Path, features: list[str]) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    selected = ",".join(["customer_id", *features, "target"])
    with _connection("6GB") as con:
        frame = con.execute(
            f"SELECT {selected} FROM read_parquet('{_sql_path(path)}') "
            "ORDER BY customer_id,candidate_rank,article_id"
        ).fetchdf()
    sizes = frame.groupby("customer_id", sort=False).size().astype(int).tolist()
    evidence = {
        "rows": len(frame),
        "groups": len(sizes),
        "minimum_group_rows": min(sizes),
        "maximum_group_rows": max(sizes),
        "positive_rows": int(frame["target"].sum()),
        "positive_groups": int(frame.groupby("customer_id", sort=False)["target"].max().sum()),
        "mind_only_positive_rows": int(frame.loc[frame["mind_is_new"] == 1, "target"].sum()),
    }
    return frame, sizes, evidence


def _matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    values = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    return np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)


def train_models(device: str) -> tuple[dict[str, Path], dict[str, Any]]:
    import lightgbm as lgb

    model_root = ARTIFACT_ROOT / "models"
    marker = model_root / "MODELS.json"
    if marker.exists():
        saved = _read_json(marker)
        paths = {name: Path(row["path"]) for name, row in saved["models"].items()}
        if all(path.exists() for path in paths.values()):
            return paths, saved
        raise FileNotFoundError("registered ranking model is missing")
    feature_path, feature_meta = build_features(TRAIN_CUTOFF, device)
    control_features, primary_features = _feature_names(feature_path)
    frame, group_sizes, audit = _load_training(feature_path, primary_features)
    contract = _read_json(CONTRACT_PATH)
    rules = contract["input_gate"]
    checks = {
        "minimum_positive_rows": audit["positive_rows"] >= rules["minimum_positive_rows"],
        "minimum_positive_groups": audit["positive_groups"] >= rules["minimum_positive_user_groups"],
        "minimum_mind_only_positive_rows": audit["mind_only_positive_rows"] >= rules["minimum_mind_only_positive_rows"],
        "minimum_group_rows": audit["minimum_group_rows"] >= 100,
        "maximum_group_rows": audit["maximum_group_rows"] <= 400,
        "final_week_not_run": True,
    }
    if not all(checks.values()):
        result = {"schema": SCHEMA, "input": feature_meta, "audit": audit, "checks": checks, "passed": False}
        _write_json(model_root / "INPUT_GATE_FAILED.json", result)
        raise RuntimeError(f"ranking input gate failed: {checks}")
    model_root.mkdir(parents=True, exist_ok=True)
    models: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    started_all = time.perf_counter()
    for name, features in {"same_pool_control": control_features, "mind_aware": primary_features}.items():
        started = time.perf_counter()
        dataset = lgb.Dataset(
            _matrix(frame, features),
            label=frame["target"].to_numpy(np.uint8),
            group=group_sizes,
            feature_name=features,
            free_raw_data=True,
        )
        history: dict[str, Any] = {}
        model = lgb.train(
            MODEL_PARAMS,
            dataset,
            num_boost_round=ROUNDS,
            valid_sets=[dataset],
            valid_names=["train"],
            callbacks=[lgb.record_evaluation(history)],
        )
        path = model_root / f"{name}.txt"
        model.save_model(str(path))
        importance = sorted(
            [
                {"feature": feature, "gain": float(gain), "split": int(split)}
                for feature, gain, split in zip(
                    features,
                    model.feature_importance(importance_type="gain"),
                    model.feature_importance(importance_type="split"),
                    strict=True,
                )
            ],
            key=lambda row: row["gain"],
            reverse=True,
        )
        models[name] = model
        metadata[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "feature_count": len(features),
            "features": features,
            "top_feature_importance": importance[:40],
            "mind_feature_importance": [row for row in importance if row["feature"] in MIND_FEATURES],
            "train_metrics": {key: float(values[-1]) for key, values in history.get("train", {}).items()},
            "elapsed_seconds": time.perf_counter() - started,
        }
    result = {
        "schema": SCHEMA,
        "training_cutoff": TRAIN_CUTOFF,
        "input": feature_meta,
        "input_gate": {"audit": audit, "checks": checks, "passed": True},
        "parameters": MODEL_PARAMS,
        "rounds": ROUNDS,
        "models": metadata,
        "elapsed_seconds": time.perf_counter() - started_all,
        "final_week": "not_run",
    }
    _write_json(marker, result)
    del frame, models
    gc.collect()
    return {name: Path(row["path"]) for name, row in metadata.items()}, result


def _warm_rank_path(stage: str, cutoff: str) -> Path:
    role = "2020_inner" if stage == "inner" else "2020_outer"
    return WARM_ROOT / "artifacts" / "warm_v3" / "gate_data" / f"{role}_{cutoff}" / "ranks.parquet"


def _warm_swap_path(stage: str, name: str) -> Path:
    if stage == "inner":
        return (
            WARM_ROOT / "artifacts" / "warm_v3" / "WV3-740" / name /
            "pointwise_lambdarank_RRF60" / "swaps.parquet"
        )
    return WARM_ROOT / "artifacts" / "warm_v3" / "WV3-741" / name / "outer" / "swaps.parquet"


def _baseline_meta(stage: str, name: str) -> tuple[float, int]:
    if stage == "inner":
        report = _read_json(WARM_ROOT / "reports" / "warm_v3" / "WV3-740_CLEAN_MODEL_REPLAY.json")
        row = report["windows"][name]
        return float(row["policies"]["pointwise_lambdarank_RRF60"]["MAP@12"]), int(row["total_users_denominator"])
    report = _read_json(WARM_ROOT / "reports" / "warm_v3" / "WV3-741_OUTER.json")
    row = report["windows"][name]
    return float(row["MAP@12"]), int(row["total_users_denominator"])


def reconstruct_baseline(stage: str, name: str, cutoff: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    ranks_path = _warm_rank_path(stage, cutoff)
    swaps_path = _warm_swap_path(stage, name)
    with _connection("4GB") as con:
        ranks = con.execute(
            f"""SELECT customer_id,article_id,rf::INTEGER AS rf,target::UTINYINT AS target,
                       truth_count::INTEGER AS truth_count
                FROM read_parquet('{_sql_path(ranks_path)}') WHERE rf<=50
                ORDER BY customer_id,rf,article_id"""
        ).fetchdf()
        swaps = con.execute(f"SELECT * FROM read_parquet('{_sql_path(swaps_path)}')").fetchdf()
    if ranks.duplicated(["customer_id", "article_id"]).any() or ranks.duplicated(["customer_id", "rf"]).any():
        raise RuntimeError("warm baseline Top50 identity is not unique")
    challenger = swaps[["customer_id", "challenger_article_id", "victim_rank"]].rename(
        columns={"challenger_article_id": "article_id", "victim_rank": "replacement_rank"}
    )
    victim = swaps[["customer_id", "victim_article_id", "challenger_rank"]].rename(
        columns={"victim_article_id": "article_id", "challenger_rank": "replacement_rank"}
    )
    changes = pd.concat([challenger, victim], ignore_index=True)
    if changes.duplicated(["customer_id", "article_id"]).any():
        raise RuntimeError("warm champion swaps are not disjoint")
    result = ranks.merge(changes, on=["customer_id", "article_id"], how="left", validate="one_to_one")
    result["champion_rank"] = result["replacement_rank"].fillna(result["rf"]).astype(np.int32)
    if result.duplicated(["customer_id", "champion_rank"]).any():
        raise RuntimeError("reconstructed champion ranks are not unique")
    result = result.drop(columns="replacement_rank")
    baseline_map, total_users = _baseline_meta(stage, name)
    top12 = result[result["champion_rank"] <= 12].sort_values(
        ["customer_id", "champion_rank", "article_id"], kind="mergesort"
    )
    observed = _mean_ap(top12, total_users)
    if abs(observed - baseline_map) > 1e-12:
        raise AssertionError(f"{stage} {name} baseline replay drift: {observed} vs {baseline_map}")
    return result, {
        "MAP@12": baseline_map,
        "total_users": total_users,
        "replayed_MAP@12": observed,
        "rank_source": str(ranks_path),
        "swap_source": str(swaps_path),
    }


def _mean_ap(top12: pd.DataFrame, total_users: int) -> float:
    ordered = top12.sort_values(["customer_id", "champion_rank", "article_id"], kind="mergesort").copy()
    ordered["hit_index"] = ordered.groupby("customer_id", sort=False)["target"].cumsum()
    ordered["precision_hit"] = np.where(
        ordered["target"].to_numpy(np.uint8) == 1,
        ordered["hit_index"] / ordered["champion_rank"],
        0.0,
    )
    sums = ordered.groupby("customer_id", sort=False)["precision_hit"].sum()
    denominators = ordered.groupby("customer_id", sort=False)["truth_count"].first().clip(upper=12)
    return float((sums / denominators).sum() / total_users)


def select_actions(challengers: pd.DataFrame, victims: pd.DataFrame, score_column: str) -> pd.DataFrame:
    forbidden = {"target", "truth_count"}
    if forbidden & set(challengers.columns) or forbidden & set(victims.columns):
        raise ValueError("labels must not enter action selection")
    best = challengers.sort_values(
        ["customer_id", score_column, "mind_rank", "article_id"],
        ascending=[True, False, True, True], kind="mergesort",
    ).drop_duplicates("customer_id", keep="first")
    worst = victims.sort_values(
        ["customer_id", score_column, "champion_rank", "article_id"],
        ascending=[True, True, False, True], kind="mergesort",
    ).drop_duplicates("customer_id", keep="first")
    selected = best[["customer_id", "article_id", "mind_rank", score_column]].rename(
        columns={"article_id": "challenger_article_id", score_column: "challenger_score"}
    ).merge(
        worst[["customer_id", "article_id", "champion_rank", score_column]].rename(
            columns={"article_id": "victim_article_id", score_column: "victim_score"}
        ), on="customer_id", how="inner", validate="one_to_one",
    )
    selected["score_difference"] = selected["challenger_score"] - selected["victim_score"]
    return selected[selected["score_difference"] > 0].sort_values("customer_id", kind="mergesort").reset_index(drop=True)


def _apk(labels: np.ndarray, truth_count: int) -> float:
    hits = np.asarray(labels, dtype=np.uint8)[:12]
    if truth_count <= 0:
        return 0.0
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / min(truth_count, 12))


def exact_action_deltas(
    baseline: pd.DataFrame, actions: pd.DataFrame, feature_labels: pd.DataFrame
) -> pd.DataFrame:
    top12 = baseline[baseline["champion_rank"] <= 12].sort_values(
        ["customer_id", "champion_rank", "article_id"], kind="mergesort"
    )
    label_lookup = feature_labels.set_index(["customer_id", "article_id"])["target"]
    rows: list[dict[str, Any]] = []
    grouped = {customer: group for customer, group in top12.groupby("customer_id", sort=False)}
    for action in actions.itertuples(index=False):
        group = grouped[action.customer_id]
        labels = group["target"].to_numpy(np.uint8)
        truth_count = int(group["truth_count"].iloc[0])
        before = _apk(labels, truth_count)
        changed = labels.copy()
        position = int(action.champion_rank) - 1
        challenger_target = int(label_lookup.loc[(action.customer_id, action.challenger_article_id)])
        victim_target = int(label_lookup.loc[(action.customer_id, action.victim_article_id)])
        if victim_target != int(changed[position]):
            raise AssertionError("victim label does not align with reconstructed baseline")
        changed[position] = challenger_target
        after = _apk(changed, truth_count)
        rows.append({
            **action._asdict(),
            "challenger_target": challenger_target,
            "victim_target": victim_target,
            "baseline_ap": before,
            "reranked_ap": after,
            "actual_delta": after - before,
        })
    return pd.DataFrame(rows)


def evaluate_window(
    *, stage: str, name: str, cutoff: str, models: dict[str, Any], model_meta: dict[str, Any], device: str
) -> dict[str, Any]:
    feature_path, feature_meta = build_features(cutoff, device)
    baseline, baseline_meta = reconstruct_baseline(stage, name, cutoff)
    control_features = model_meta["models"]["same_pool_control"]["features"]
    primary_features = model_meta["models"]["mind_aware"]["features"]
    all_features = list(dict.fromkeys([*primary_features, "target"]))
    with _connection("6GB") as con:
        frame = con.execute(
            f"SELECT customer_id,article_id,{','.join(all_features)} "
            f"FROM read_parquet('{_sql_path(feature_path)}') ORDER BY customer_id,candidate_rank,article_id"
        ).fetchdf()
    champion = baseline[["customer_id", "article_id", "champion_rank"]]
    scored = frame.merge(champion, on=["customer_id", "article_id"], how="left", validate="one_to_one")
    results: dict[str, Any] = {}
    output_root = ARTIFACT_ROOT / stage / name
    output_root.mkdir(parents=True, exist_ok=True)
    for model_name, features in {"same_pool_control": control_features, "mind_aware": primary_features}.items():
        started = time.perf_counter()
        values = models[model_name].predict(_matrix(scored, features), num_threads=8)
        score_column = f"{model_name}_score"
        scored[score_column] = values
        challengers = scored.loc[scored["mind_is_new"] == 1, [
            "customer_id", "article_id", "mind_rank", score_column
        ]].copy()
        victims = scored.loc[scored["champion_rank"].between(8, 12), [
            "customer_id", "article_id", "champion_rank", score_column
        ]].copy()
        actions = select_actions(challengers, victims, score_column)
        deltas = exact_action_deltas(
            baseline,
            actions,
            scored[["customer_id", "article_id", "target"]],
        )
        delta = float(deltas["actual_delta"].sum() / baseline_meta["total_users"]) if len(deltas) else 0.0
        beneficial = deltas["actual_delta"] > 1e-15 if len(deltas) else pd.Series(dtype=bool)
        harmful = deltas["actual_delta"] < -1e-15 if len(deltas) else pd.Series(dtype=bool)
        neutral = ~(beneficial | harmful) if len(deltas) else pd.Series(dtype=bool)
        output = output_root / f"{model_name}-actions.parquet"
        with _connection("2GB") as con:
            con.register("actions", deltas)
            con.execute(f"COPY actions TO '{_sql_path(output)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        results[model_name] = {
            "MAP@12": baseline_meta["MAP@12"] + delta,
            "delta_vs_frozen_baseline": delta,
            "selected_users": len(deltas),
            "beneficial_users": int(beneficial.sum()) if len(deltas) else 0,
            "harmful_users": int(harmful.sum()) if len(deltas) else 0,
            "neutral_users": int(neutral.sum()) if len(deltas) else 0,
            "gross_positive_MAP": float(deltas.loc[beneficial, "actual_delta"].sum() / baseline_meta["total_users"]) if len(deltas) else 0.0,
            "gross_negative_MAP": float(deltas.loc[harmful, "actual_delta"].sum() / baseline_meta["total_users"]) if len(deltas) else 0.0,
            "actions": str(output.resolve()),
            "selection_used_labels": False,
            "maximum_actions_per_user": 1,
            "protected_ranks_1_7_changes": 0,
            "runtime_seconds": time.perf_counter() - started,
        }
    results["mind_aware"]["delta_vs_same_pool_control"] = (
        results["mind_aware"]["MAP@12"] - results["same_pool_control"]["MAP@12"]
    )
    return {
        "stage": stage,
        "name": name,
        "cutoff": cutoff,
        "baseline": baseline_meta,
        "features": feature_meta,
        "policies": results,
        "final_week": "not_run",
    }


def _gate(windows: dict[str, Any], contract_key: str) -> dict[str, Any]:
    primary = [row["policies"]["mind_aware"]["delta_vs_frozen_baseline"] for row in windows.values()]
    relative = [row["policies"]["mind_aware"]["delta_vs_same_pool_control"] for row in windows.values()]
    contract = _read_json(CONTRACT_PATH)[contract_key]
    if contract_key == "inner_gate":
        checks = {
            "positive_mean_vs_frozen_baseline": float(np.mean(primary)) > contract["primary_mean_map12_delta_vs_frozen_baseline_min_exclusive"],
            "minimum_nondegrade_windows": sum(value >= 0 for value in primary) >= contract["primary_nondegrade_windows_min"],
            "worst_window_floor": min(primary) >= contract["primary_worst_window_delta_min"],
            "positive_mean_vs_same_pool_control": float(np.mean(relative)) > contract["primary_mean_delta_vs_same_pool_control_min_exclusive"],
        }
    else:
        checks = {
            "positive_mean_vs_WV3_741": float(np.mean(primary)) > contract["mean_map12_delta_vs_WV3_741_min_exclusive"],
            "minimum_nondegrade_windows": sum(value >= 0 for value in primary) >= contract["nondegrade_windows_min"],
            "worst_window_floor": min(primary) >= contract["worst_window_delta_min"],
        }
    return {
        "mean_delta_vs_frozen_baseline": float(np.mean(primary)),
        "nondegrade_windows": sum(value >= 0 for value in primary),
        "worst_window_delta": min(primary),
        "mean_delta_vs_same_pool_control": float(np.mean(relative)),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _age_weight_rows() -> list[dict[str, Any]]:
    rows = []
    for cutoff in [TRAIN_CUTOFF, *INNER.values(), *OUTER.values()]:
        path = (
            ROOT / "artifacts" / "mind_warm_side" / "MIND-WARM-MVE-001" /
            cutoff / MIND_VARIANT / "MODEL.json"
        )
        if not path.exists():
            continue
        weights = _read_json(path)["learned_age_weights_by_interest_0_7__8_28__29_84"]
        rows.append({"cutoff": cutoff, "weights": weights})
    return rows


def render_report(result: dict[str, Any]) -> str:
    inner = result.get("inner_gate")
    outer = result.get("outer_gate")
    lines = [
        "# MIND Warm 排序兑现：保守尾部准入",
        "",
        "## 结论",
        "",
    ]
    if outer:
        lines += [
            f"外层晋级门槛：**{'通过' if outer['passed'] else '未通过'}**。主排序方案相对 WV3-741 的四窗平均 MAP@12 增量为 `{outer['mean_delta_vs_frozen_baseline']:+.9f}`。",
        ]
    elif inner:
        lines += [
            f"内层筛选门槛：**{'通过' if inner['passed'] else '未通过'}**。主排序方案相对冻结内层基线的四窗平均 MAP@12 增量为 `{inner['mean_delta_vs_frozen_baseline']:+.9f}`。",
            "内层未通过时不读取外层排序标签，也不调整阈值、树参数或候选数救援。",
        ]
    else:
        lines += ["排序训练与评测尚未完成。"]
    lines += [
        "",
        "最终周 `2020-09-16` 保持 `not_run`。",
        "",
        "## 术语与统计口径",
        "",
        "- 保守尾部准入（本项目自定义）：冻结原排序第1—7名；每名用户最多让一件不在原候选并集中的MIND商品，与原排序第8—12名中模型分数最低的一件商品交换。",
        "- MIND-only candidate（本项目自定义，报告中称MIND独有候选）：存在于本轮MIND Top100、但不存在于原六路加Item2Vec候选并集的用户—商品候选行。",
        "- same-pool control（本项目自定义，报告中称同池对照）：训练行、标签和LightGBM参数与主模型完全相同，但隐藏十个MIND来源特征；用于区分“新候选本身”与“MIND分数可用于排序”。",
        "- MIND-aware（本项目自定义，报告中称MIND感知排序）：在同池对照特征之外读取MIND名次、相似度、用户内标准分、兴趣编号和兴趣数量。",
        "- LambdaRank（行业通用学习排序）：以每名用户的一组候选为一个排序组，学习未来购买商品应位于未购买商品之前；缺失购买不是曝光后的明确负反馈。",
        "- MAP@12增量（本项目评测）：新策略MAP@12减同窗口冻结基线MAP@12；分母是该窗口全部有真值用户，不只计算执行换位的用户。",
        "- 有益、伤害、中性用户（本项目审计）：执行一次换位后，用户AP@12相对冻结基线分别上升、下降或不变；这些标签只在动作固定后用于评测。",
        "",
        "## 固定架构",
        "",
        "```text",
        "原六路+Item2Vec候选并集 + 学习年龄权重的MIND Top100",
        "  -> 按用户—商品去重，形成100—400行排序组",
        "  -> 2019-11-27单一严格历史窗口训练LambdaRank",
        "  -> 同池对照 / MIND感知排序",
        "  -> 冻结原Top1—7",
        "  -> 最多一件MIND独有候选与原Top8—12竞争",
        "  -> 先四个内层窗口，过门后才做四个外层窗口",
        "```",
        "",
    ]
    training = result.get("training")
    if training:
        audit = training["input_gate"]["audit"]
        lines += [
            "## 训练输入",
            "",
            f"训练截止日 `2019-11-27`；共 `{audit['rows']:,}` 个用户—商品候选行、`{audit['groups']:,}` 个用户排序组、`{audit['positive_rows']:,}` 个候选正例，其中MIND独有正例 `{audit['mind_only_positive_rows']:,}` 个。",
            "历史特征只读取截止日前84天，标签只读取随后7天。排序阶段没有使用固定28天半衰期特征；行为年龄已经在上游MIND表示中以0—7、8—28、29—84天三个可学习权重进入。",
            "",
            "## 学到的行为年龄权重",
            "",
            "行为年龄权重（本项目模型参数）：三项依次对应距目标日0—7天、8—28天、29—84天的历史交互；每个兴趣的三项均值归一化为1，因此大于1表示相对强调，小于1表示相对削弱。每格是一个兴趣向量的三项权重，不是样本数或概率。",
            "",
            "| 截止日 | 兴趣1：0—7 / 8—28 / 29—84天 | 兴趣2：0—7 / 8—28 / 29—84天 | 兴趣3：0—7 / 8—28 / 29—84天 |",
            "|---|---:|---:|---:|",
        ]
        for row in _age_weight_rows():
            formatted = [" / ".join(f"{value:.3f}" for value in interest) for interest in row["weights"]]
            lines.append(f"| {row['cutoff']} | {formatted[0]} | {formatted[1]} | {formatted[2]} |")
        lines += [
            "",
            "九个已运行截止日里，每个兴趣都把0—7天权重学到最高、29—84天学到最低，但旧行为权重始终大于0。这证明模型确实学出了跨截止日重复的非均匀时间结构，支持把行为年龄作为可学习变量；它不证明学习年龄版本已优于固定半衰期或无时间版本，因为召回审计中无时间对照的平均Recall仍略高。",
            "",
        ]
    for stage, title in (("inner_windows", "内层筛选"), ("outer_windows", "外层确认")):
        windows = result.get(stage)
        if not windows:
            continue
        lines += [
            f"## {title}",
            "",
            "| 窗口 | 冻结基线MAP | 同池对照增量 | MIND感知增量 | MIND相对同池对照 | MIND动作用户 | 有益用户 | 伤害用户 | 中性用户 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, row in windows.items():
            control = row["policies"]["same_pool_control"]
            primary = row["policies"]["mind_aware"]
            lines.append(
                f"| {name} | {row['baseline']['MAP@12']:.9f} | {control['delta_vs_frozen_baseline']:+.9f} | "
                f"{primary['delta_vs_frozen_baseline']:+.9f} | {primary['delta_vs_same_pool_control']:+.9f} | "
                f"{primary['selected_users']:,} | {primary['beneficial_users']:,} | {primary['harmful_users']:,} | {primary['neutral_users']:,} |"
            )
        gate = result["inner_gate" if stage == "inner_windows" else "outer_gate"]
        lines += [
            "",
            f"门槛明细：`{json.dumps(gate['checks'], ensure_ascii=False)}`；平均基线增量 `{gate['mean_delta_vs_frozen_baseline']:+.9f}`，不退化窗口 `{gate['nondegrade_windows']}/4`，最差窗口 `{gate['worst_window_delta']:+.9f}`。",
            "",
        ]
    if result.get("inner_windows"):
        primary = [row["policies"]["mind_aware"] for row in result["inner_windows"].values()]
        control = [row["policies"]["same_pool_control"] for row in result["inner_windows"].values()]
        selected = sum(row["selected_users"] for row in primary)
        beneficial = sum(row["beneficial_users"] for row in primary)
        harmful = sum(row["harmful_users"] for row in primary)
        neutral = sum(row["neutral_users"] for row in primary)
        source_delta = float(np.mean([
            row["delta_vs_same_pool_control"] for row in primary
        ]))
        lines += [
            "## 内层动作漏斗诊断",
            "",
            f"MIND感知排序共执行 `{selected:,}` 个用户级换位动作；其中 `{beneficial:,}` 个用户AP上升、`{harmful:,}` 个下降、`{neutral:,}` 个不变。中性动作占 `{neutral/max(selected,1):.2%}`，伤害动作数是有益动作数的 `{harmful/max(beneficial,1):.2f}` 倍。",
            f"MIND来源特征相对同池对照平均改善 `{source_delta:+.9f}`，并在 `{sum(row['delta_vs_same_pool_control'] >= 0 for row in primary)}/4` 个窗口不差；因此模型并非完全忽略MIND分数。失败发生在准入决策的绝对精度：零分差规则放入了大量未来未购买候选，四窗平均毛正收益 `{np.mean([row['gross_positive_MAP'] for row in primary]):+.9f}`，被平均毛负收益 `{np.mean([row['gross_negative_MAP'] for row in primary]):+.9f}` 覆盖。",
            "这组结果只支持后续审计分数差的校准与可拒绝性，不授权从本次内层标签搜索阈值；本实验按预注册停止条件保留WV3-741。",
            "",
        ]
    lines += [
        "## 决策边界",
        "",
        ("外层门槛通过，可把该保守准入方案晋级为新的Warm开发基线。" if outer and outer["passed"] else
         "未取得合格的端到端MAP证据，保留WV3-741。MIND仍是召回层的稳定正结果，但排序价值尚未兑现。"),
        "没有运行最终周、提交生成、阈值搜索、树参数搜索、候选数搜索或年龄分桶救援。",
        "",
        f"机器可读结果：`{METRICS_PATH.name}`。",
        "",
    ]
    return "\n".join(lines)


def _update_parent(result: dict[str, Any]) -> None:
    parent = _read_json(PARENT_METRICS_PATH)
    parent["ranking_status"] = result["ranking_status"]
    parent["ranking_followup"] = {
        "experiment_id": result["experiment_id"],
        "report": str(REPORT_PATH.resolve()),
        "inner_gate": result.get("inner_gate"),
        "outer_gate": result.get("outer_gate"),
        "final_week": "not_run",
    }
    _write_json(PARENT_METRICS_PATH, parent)
    PARENT_REPORT_PATH.write_text(mind_warm_mve.render_report(parent), encoding="utf-8", newline="\n")


def run(device: str = "cuda") -> dict[str, Any]:
    contract = _read_json(CONTRACT_PATH)
    if contract["data_protocol"]["final_week"] != "2020-09-16 not_run":
        raise AssertionError("final week must remain not_run")
    parent = _read_json(PARENT_METRICS_PATH)
    if not parent.get("retrieval_promotion_gate", {}).get("passed"):
        raise RuntimeError("MIND retrieval promotion gate did not pass")
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": contract["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(CONTRACT_PATH.resolve()),
        "ranking_status": "running",
        "final_week": "not_run",
    }
    model_paths, training = train_models(device)
    import lightgbm as lgb
    models = {name: lgb.Booster(model_file=str(path)) for name, path in model_paths.items()}
    result["training"] = training
    inner_windows = {}
    for name, cutoff in INNER.items():
        print({"ranking_inner": name, "cutoff": cutoff}, flush=True)
        inner_windows[name] = evaluate_window(
            stage="inner", name=name, cutoff=cutoff, models=models, model_meta=training, device=device
        )
        result["inner_windows"] = inner_windows
        _write_json(METRICS_PATH, result)
    result["inner_gate"] = _gate(inner_windows, "inner_gate")
    if not result["inner_gate"]["passed"]:
        result["ranking_status"] = "inner_failed_outer_not_run"
        result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(METRICS_PATH, result)
        REPORT_PATH.write_text(render_report(result), encoding="utf-8", newline="\n")
        _update_parent(result)
        return result
    outer_windows = {}
    for name, cutoff in OUTER.items():
        print({"ranking_outer": name, "cutoff": cutoff}, flush=True)
        outer_windows[name] = evaluate_window(
            stage="outer", name=name, cutoff=cutoff, models=models, model_meta=training, device=device
        )
        result["outer_windows"] = outer_windows
        _write_json(METRICS_PATH, result)
    result["outer_gate"] = _gate(outer_windows, "outer_gate")
    result["ranking_status"] = "outer_passed_promoted" if result["outer_gate"]["passed"] else "outer_failed"
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(METRICS_PATH, result)
    REPORT_PATH.write_text(render_report(result), encoding="utf-8", newline="\n")
    _update_parent(result)
    return result


def render_existing() -> dict[str, Any]:
    result = _read_json(METRICS_PATH)
    REPORT_PATH.write_text(render_report(result), encoding="utf-8", newline="\n")
    _update_parent(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "render"])
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    result = run(args.device) if args.command == "run" else render_existing()
    print({
        "experiment": result["experiment_id"],
        "ranking_status": result["ranking_status"],
        "inner_passed": result.get("inner_gate", {}).get("passed"),
        "outer_passed": result.get("outer_gate", {}).get("passed"),
        "report": str(REPORT_PATH),
        "final_week": result["final_week"],
    }, flush=True)


if __name__ == "__main__":
    main()
