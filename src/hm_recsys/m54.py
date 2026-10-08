from __future__ import annotations

import gc
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from .m2 import (
    M2Config,
    RETRIEVAL_FEATURES,
    _create_static_dimensions,
    build_point_in_time_dataset,
)
from .m210 import TARGET_AWARE_FEATURES, score_models as score_upstream_models
from .m212 import DECAY_FEATURES
from .m29 import ITEM2VEC_FEATURES
from .m3 import _peak_working_set_bytes
from .m33 import ROLLING_PROTOCOL, variant_feature_sets
from .m4_contract import FINAL_CUTOFF, STATIC_FIELDS, atomic_json, file_identity
from .m5_data import AFFINITY_FIELDS, SIMILARITY_FEATURES
from .m5_model import MODEL_FEATURES, _frame, _load_arrays, _load_category_maps, _write_parquet


RUN_ID = "m5-4-v1-temporal-oof-dual-channel"
K_COLD = 50
NO_DECAY_TARGET_FEATURES = [name for name in TARGET_AWARE_FEATURES if name not in DECAY_FEATURES]
WARM_EVIDENCE_FEATURES = [name for name in RETRIEVAL_FEATURES if name != "candidate_rank"] + ITEM2VEC_FEATURES
COLD_FEATURES = list(SIMILARITY_FEATURES)
SOURCE_FEATURES = [
    "candidate_rank",
    "warm_present",
    "cold_present",
    "both_present",
    "source_branch",
    "warm_rank",
    "warm_rank_pct",
    "warm_candidate_rank",
    "warm_model_score",
    "warm_model_score_available",
    "cold_rank",
    "cold_rank_pct",
    "cold_expert_score",
    "cold_expert_rank",
    "cold_expert_score_available",
    *WARM_EVIDENCE_FEATURES,
    *COLD_FEATURES,
]


LINEAGES: dict[str, dict[str, Any] | None] = {
    "2019-11-27": None,
    "2019-12-25": {"window": "winter_20200122", "kind": "inner", "train_cutoffs": ["2019-11-27"]},
    "2020-01-22": {"window": "winter_20200122", "kind": "outer", "train_cutoffs": ["2019-11-27", "2019-12-25"]},
    "2020-02-19": {"window": "spring_20200318", "kind": "inner", "train_cutoffs": ["2020-01-22"]},
    "2020-03-18": {"window": "spring_20200318", "kind": "outer", "train_cutoffs": ["2020-01-22", "2020-02-19"]},
    "2020-04-29": {"window": "spring_20200318", "kind": "forward", "train_cutoffs": ["2020-01-22", "2020-02-19"]},
    "2020-05-27": {"window": "early_summer_20200624", "kind": "inner", "train_cutoffs": ["2020-04-29"]},
    "2020-06-24": {"window": "early_summer_20200624", "kind": "outer", "train_cutoffs": ["2020-04-29", "2020-05-27"]},
    "2020-07-22": {"window": "late_summer_20200819", "kind": "inner", "train_cutoffs": ["2020-06-24"]},
    "2020-08-19": {"window": "late_summer_20200819", "kind": "outer", "train_cutoffs": ["2020-06-24", "2020-07-22"]},
}


def _literal(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _prepare_pit_connection(
    *, database_path: Path, temp_dir: Path, transactions_path: Path,
    articles_path: Path, customers_path: Path,
) -> duckdb.DuckDBPyConnection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database_path))
    connection.execute("SET threads=8")
    connection.execute("SET memory_limit='12GB'")
    connection.execute(f"SET temp_directory={_literal(temp_dir)}")
    connection.execute(
        f"CREATE OR REPLACE VIEW transactions AS SELECT * FROM read_parquet({_literal(transactions_path)})"
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW articles AS SELECT * FROM read_csv_auto({_literal(articles_path)},all_varchar=true)"
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW customers AS SELECT * FROM read_csv_auto({_literal(customers_path)},all_varchar=true)"
    )
    return connection


def _date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def validate_lineage(cutoff: str, lineage: dict[str, Any] | None) -> dict[str, Any]:
    if cutoff >= FINAL_CUTOFF:
        raise RuntimeError("M5.4 must not read the final week")
    if lineage is None:
        return {
            "available": False,
            "safe": True,
            "reason": "no earlier frozen model exists; score remains missing and availability flag is zero",
        }
    train_cutoffs = [str(value) for value in lineage["train_cutoffs"]]
    latest_label_end = max(_date(value) + timedelta(days=7) for value in train_cutoffs)
    safe = latest_label_end <= _date(cutoff) and all(value < cutoff for value in train_cutoffs)
    if not safe:
        raise RuntimeError(f"M5.4 upstream model leakage: scoring={cutoff}, training={train_cutoffs}")
    return {
        "available": True,
        "safe": True,
        "training_cutoffs": train_cutoffs,
        "latest_training_label_end": latest_label_end.isoformat(),
        "scoring_cutoff": cutoff,
        "kind": lineage["kind"],
        "window": lineage["window"],
    }


def _cold_root(cutoff: str, m5_artifact_dir: Path) -> tuple[Path, str]:
    outer_by_cutoff = {
        protocol["outer_validation"]: window for window, protocol in ROLLING_PROTOCOL.items()
    }
    if cutoff in outer_by_cutoff:
        return m5_artifact_dir / "outer-data-v1" / outer_by_cutoff[cutoff], "authoritative_outer_m4_candidate"
    return m5_artifact_dir / "train-data-v1" / cutoff, "point_in_time_training_candidate"


def _warm_score_source(
    *, cutoff: str, lineage: dict[str, Any] | None, source_root: Path,
    m5_artifact_dir: Path, output_root: Path, m3: dict[str, Any],
) -> tuple[Path | None, dict[str, Any]]:
    audit = validate_lineage(cutoff, lineage)
    if lineage is None:
        return None, audit
    window = lineage["window"]
    kind = lineage["kind"]
    warm_model_root = source_root / "artifacts" / "m3_3" / m3["run_id"] / window
    if kind == "inner":
        database = m5_artifact_dir / "warm-inner-v1" / window / "evaluation.duckdb"
        model = warm_model_root / "lightgbm-inner-anchor.txt"
        maps = warm_model_root / "inner-category-maps.json"
    elif kind == "outer":
        database = warm_model_root / "evaluation.duckdb"
        model = warm_model_root / "lightgbm-anchor.txt"
        maps = warm_model_root / "outer-category-maps.json"
    else:
        forward_root = output_root / "forward-warm" / cutoff
        database = forward_root / "evaluation.duckdb"
        predictions = forward_root / "predictions.parquet"
        model = warm_model_root / "lightgbm-anchor.txt"
        maps = warm_model_root / "outer-category-maps.json"
        if not database.is_file() or not predictions.is_file():
            forward_root.mkdir(parents=True, exist_ok=False)
            booster = lgb.Booster(model_file=str(model))
            score_upstream_models(
                dataset_path=Path(m3["feature_cache"][cutoff]["artifact"]["path"]),
                models={"anchor": (booster, variant_feature_sets()["anchor"], _load_category_maps(maps))},
                evaluation_db=database,
                prediction_path=predictions,
                config=M2Config(**m3["config"]),
            )
    for path in (database, model, maps):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit.update({
        "score_database": file_identity(database),
        "model": file_identity(model),
        "category_maps": file_identity(maps),
    })
    return database, audit


def _cold_model(
    *, lineage: dict[str, Any] | None, m5_artifact_dir: Path,
) -> tuple[lgb.Booster | None, dict[str, Any]]:
    if lineage is None:
        return None, {"available": False, "safe": True}
    suffix = "inner-cold-expert.txt" if lineage["kind"] == "inner" else "outer-cold-expert.txt"
    path = m5_artifact_dir / "models-v1" / lineage["window"] / suffix
    if not path.is_file():
        raise FileNotFoundError(path)
    return lgb.Booster(model_file=str(path)), {"model": file_identity(path)}


def _build_warm_shortlist(
    *, feature_path: Path, score_db: Path | None, output_path: Path, k_warm: int,
) -> dict[str, Any]:
    con = duckdb.connect()
    con.execute("SET threads=8")
    try:
        source = _literal(feature_path)
        if score_db is None:
            ranked = f"""
                SELECT customer_id,article_id,candidate_rank::INTEGER AS warm_rank,
                       candidate_rank::INTEGER AS warm_candidate_rank,
                       count(*) OVER(PARTITION BY customer_id)::INTEGER AS warm_group_rows,
                       NULL::DOUBLE AS warm_model_score,
                       0::UTINYINT AS warm_model_score_available
                FROM read_parquet({source})
            """
        else:
            con.execute(f"ATTACH {_literal(score_db)} AS score_db (READ_ONLY)")
            ranked = """
                SELECT customer_id,article_id,
                       row_number() OVER(
                           PARTITION BY customer_id
                           ORDER BY CASE WHEN user_history_events_12w=0 THEN candidate_rank END ASC NULLS LAST,
                                    CASE WHEN user_history_events_12w>0 THEN score_anchor END DESC NULLS LAST,
                                    candidate_rank,article_id
                       )::INTEGER AS warm_rank,
                       candidate_rank::INTEGER AS warm_candidate_rank,
                       count(*) OVER(PARTITION BY customer_id)::INTEGER AS warm_group_rows,
                       score_anchor::DOUBLE AS warm_model_score,
                       1::UTINYINT AS warm_model_score_available
                FROM score_db.predictions
            """
        warm_select = ",".join(f"f.{name}" for name in WARM_EVIDENCE_FEATURES)
        con.execute(
            f"""
            COPY (
                WITH ranked AS ({ranked})
                SELECT r.customer_id,r.article_id,
                       r.warm_rank,r.warm_rank::DOUBLE/r.warm_group_rows AS warm_rank_pct,
                       r.warm_candidate_rank,r.warm_model_score,r.warm_model_score_available,
                       {warm_select}
                FROM ranked r JOIN read_parquet({source}) f USING(customer_id,article_id)
                WHERE r.warm_rank<={k_warm}
                ORDER BY r.customer_id,r.warm_rank,r.article_id
            ) TO {_literal(output_path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)
            """
        )
        rows = con.execute(
            f"SELECT count(*),count(DISTINCT (customer_id,article_id)),count(DISTINCT customer_id),"
            f"min(warm_rank),max(warm_rank),sum(warm_model_score_available) FROM read_parquet({_literal(output_path)})"
        ).fetchone()
        if int(rows[0]) != int(rows[1]) or int(rows[3]) != 1 or int(rows[4]) > k_warm:
            raise RuntimeError("M5.4 Warm shortlist identity/rank failure")
        return {
            "rows": int(rows[0]), "users": int(rows[2]), "min_rank": int(rows[3]),
            "max_rank": int(rows[4]), "score_available_rows": int(rows[5]),
            "artifact": file_identity(output_path),
        }
    finally:
        con.close()


def _build_cold_shortlist(
    *, cold_root: Path, warm_path: Path, static_dir: Path, model: lgb.Booster | None,
    output_path: Path,
) -> dict[str, Any]:
    dataset = cold_root / "cold_candidates_top50.npz"
    users_path = cold_root / "users.csv"
    manifest_path = cold_root / "manifest.json"
    for path in (dataset, users_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    arrays = _load_arrays(dataset, users_path)
    warm_users = set(
        duckdb.connect().execute(
            f"SELECT DISTINCT customer_id FROM read_parquet({_literal(warm_path)})"
        ).fetchnumpy()["customer_id"].tolist()
    )
    user_keep = np.asarray([user in warm_users for user in arrays["_users"]], dtype=bool)
    indices = np.flatnonzero(user_keep[arrays["user_index"]])
    if not len(indices):
        raise RuntimeError("M5.4 no cold rows overlap Warm users")
    catalog_items = pd.read_csv(static_dir / "catalog_items.csv", dtype={"article_id": str})["article_id"].to_numpy()
    categories = np.load(static_dir / "catalog_categories.int32.npy", mmap_mode="r")
    frame = pd.DataFrame({
        "customer_id": arrays["_users"][arrays["user_index"][indices]],
        "article_id": catalog_items[arrays["catalog_row"][indices]],
        "cold_rank": arrays["student_rank"][indices].astype(np.int16),
    })
    frame["cold_rank_pct"] = frame["cold_rank"].astype(np.float32) / K_COLD
    for name in COLD_FEATURES:
        frame[name] = arrays[name][indices]
    if model is None:
        frame["cold_expert_score"] = np.nan
        frame["cold_expert_rank"] = np.nan
        frame["cold_expert_score_available"] = np.uint8(0)
    else:
        model_frame = _frame(arrays, indices, categories)
        score = model.predict(model_frame[MODEL_FEATURES])
        frame["cold_expert_score"] = score.astype(np.float64)
        frame["cold_expert_score_available"] = np.uint8(1)
        order = frame.sort_values(
            ["customer_id", "cold_expert_score", "cold_rank", "article_id"],
            ascending=[True, False, True, True], kind="mergesort",
        )
        order["cold_expert_rank"] = order.groupby("customer_id", sort=False).cumcount() + 1
        frame = frame.join(order["cold_expert_rank"])
        del model_frame, order
    _write_parquet(frame, output_path)
    con = duckdb.connect()
    try:
        row = con.execute(
            f"SELECT count(*),count(DISTINCT (customer_id,article_id)),count(DISTINCT customer_id),"
            f"min(cold_rank),max(cold_rank),sum(cold_expert_score_available) "
            f"FROM read_parquet({_literal(output_path)})"
        ).fetchone()
        if int(row[0]) != int(row[1]) or int(row[3]) != 1 or int(row[4]) != K_COLD:
            raise RuntimeError("M5.4 Cold shortlist identity/rank failure")
        distribution = con.execute(
            f"SELECT count(*) FILTER(WHERE cold_expert_score_available=1),"
            f"min(cold_expert_score),approx_quantile(cold_expert_score,0.5),"
            f"approx_quantile(cold_expert_score,0.95),max(cold_expert_score) "
            f"FROM read_parquet({_literal(output_path)})"
        ).fetchone()
    finally:
        con.close()
    evidence = {
        "rows": int(row[0]), "users": int(row[2]), "min_rank": int(row[3]),
        "max_rank": int(row[4]), "score_available_rows": int(row[5]),
        "score_distribution": {
            "available_rows": int(distribution[0]), "min": distribution[1],
            "median": distribution[2], "p95": distribution[3], "max": distribution[4],
        },
        "inputs": {
            "dataset": file_identity(dataset), "users": file_identity(users_path),
            "manifest": file_identity(manifest_path),
        },
        "artifact": file_identity(output_path),
    }
    del arrays, frame
    gc.collect()
    return evidence


def _build_union_candidate(
    *, warm_path: Path, cold_path: Path, output_path: Path,
) -> dict[str, Any]:
    con = duckdb.connect()
    con.execute("SET threads=8")
    try:
        warm_cols = ",".join(f"w.{name}" for name in WARM_EVIDENCE_FEATURES)
        cold_cols = ",".join(f"c.{name}" for name in COLD_FEATURES)
        con.execute(
            f"""
            COPY (
              WITH merged AS (
                SELECT coalesce(w.customer_id,c.customer_id) AS customer_id,
                       coalesce(w.article_id,c.article_id) AS article_id,
                       (w.article_id IS NOT NULL)::UTINYINT AS warm_present,
                       (c.article_id IS NOT NULL)::UTINYINT AS cold_present,
                       (w.article_id IS NOT NULL AND c.article_id IS NOT NULL)::UTINYINT AS both_present,
                       CASE WHEN w.article_id IS NOT NULL AND c.article_id IS NOT NULL THEN 'warm_and_cold'
                            WHEN w.article_id IS NOT NULL THEN 'warm_only' ELSE 'cold_only' END AS source_branch,
                       w.warm_rank,w.warm_rank_pct,w.warm_candidate_rank,
                       w.warm_model_score,w.warm_model_score_available,
                       c.cold_rank,c.cold_rank_pct,c.cold_expert_score,c.cold_expert_rank,
                       c.cold_expert_score_available,
                       {warm_cols},{cold_cols}
                FROM read_parquet({_literal(warm_path)}) w
                FULL OUTER JOIN read_parquet({_literal(cold_path)}) c USING(customer_id,article_id)
              )
              SELECT customer_id,article_id,
                     row_number() OVER(
                       PARTITION BY customer_id
                       ORDER BY CASE source_branch WHEN 'warm_and_cold' THEN 0 WHEN 'warm_only' THEN 1 ELSE 2 END,
                                warm_rank NULLS LAST,cold_rank NULLS LAST,article_id
                     )::INTEGER AS candidate_rank,
                     * EXCLUDE(customer_id,article_id)
              FROM merged ORDER BY customer_id,candidate_rank,article_id
            ) TO {_literal(output_path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)
            """
        )
        audit = con.execute(
            f"""
            WITH g AS (
              SELECT customer_id,count(*) n,count(DISTINCT article_id) u,
                     min(candidate_rank) lo,max(candidate_rank) hi,count(DISTINCT candidate_rank) r
              FROM read_parquet({_literal(output_path)}) GROUP BY customer_id
            )
            SELECT (SELECT count(*) FROM read_parquet({_literal(output_path)})),
                   (SELECT count(DISTINCT (customer_id,article_id)) FROM read_parquet({_literal(output_path)})),
                   count(*),min(n),max(n),count(*) FILTER(WHERE n<>u OR lo<>1 OR hi<>n OR r<>n),
                   (SELECT count(*) FROM read_parquet({_literal(output_path)}) WHERE source_branch='warm_only'),
                   (SELECT count(*) FROM read_parquet({_literal(output_path)}) WHERE source_branch='cold_only'),
                   (SELECT count(*) FROM read_parquet({_literal(output_path)}) WHERE source_branch='warm_and_cold')
            FROM g
            """
        ).fetchone()
        values = list(map(int, audit))
        if values[0] != values[1] or values[5]:
            raise RuntimeError("M5.4 union identity/consecutive-rank failure")
        return {
            "rows": values[0], "unique_user_item_rows": values[1], "users": values[2],
            "min_group_rows": values[3], "max_group_rows": values[4], "invalid_groups": values[5],
            "branch_rows": {"warm_only": values[6], "cold_only": values[7], "warm_and_cold": values[8]},
            "artifact": file_identity(output_path),
        }
    finally:
        con.close()


def _dimension_sql(table: str, key: str, cutoff: str) -> str:
    return f"""
      CREATE TABLE {table} AS
      SELECT customer_id,{key},
             count(*) FILTER(WHERE t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY)::BIGINT AS events_28d,
             count(*)::BIGINT AS events_12w,
             date_diff('day',max(t_dat),DATE '{cutoff}')::BIGINT AS days_since
      FROM history12 GROUP BY customer_id,{key}
    """


def _add_no_decay_features(
    *, base_path: Path, output_path: Path, transactions_path: Path, articles_path: Path,
    cutoff: str,
) -> dict[str, Any]:
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute("SET memory_limit='12GB'")
    temp = output_path.parent / "duckdb-temp"
    temp.mkdir(exist_ok=True)
    con.execute(f"SET temp_directory={_literal(temp)}")
    try:
        con.execute(f"CREATE TABLE base AS SELECT * FROM read_parquet({_literal(base_path)})")
        con.execute("CREATE TABLE users AS SELECT DISTINCT customer_id FROM base")
        con.execute(
            f"""
            CREATE TABLE article_dim AS SELECT article_id,
              try_cast(product_code AS INTEGER) AS article_product_code,
              try_cast(product_type_no AS INTEGER) AS article_product_type_no,
              try_cast(department_no AS INTEGER) AS article_department_no,
              try_cast(garment_group_no AS INTEGER) AS article_garment_group_no,
              try_cast(perceived_colour_master_id AS INTEGER) AS article_colour_master_id,
              try_cast(index_group_no AS INTEGER) AS article_index_group_no
            FROM read_csv_auto({_literal(articles_path)},all_varchar=true)
            """
        )
        con.execute(
            f"""
            CREATE TABLE history12 AS SELECT t.*,a.article_product_code,a.article_product_type_no,
              a.article_department_no,a.article_garment_group_no,a.article_colour_master_id,a.article_index_group_no
            FROM read_parquet({_literal(transactions_path)}) t JOIN users u USING(customer_id)
            JOIN article_dim a USING(article_id)
            WHERE t_dat>=DATE '{cutoff}'-INTERVAL 12 WEEK AND t_dat<DATE '{cutoff}'
            """
        )
        con.execute(
            f"""
            CREATE TABLE item_all AS SELECT article_id,count(*)::BIGINT AS events
            FROM read_parquet({_literal(transactions_path)}) WHERE t_dat<DATE '{cutoff}' GROUP BY article_id
            """
        )
        dimensions = {
            "ui": "article_id", "pc": "article_product_code", "pt": "article_product_type_no",
            "dp": "article_department_no", "gg": "article_garment_group_no",
            "cl": "article_colour_master_id", "ig": "article_index_group_no",
        }
        for table, key in dimensions.items():
            con.execute(_dimension_sql(table, key, cutoff))
        con.execute(
            f"""
            COPY (
              SELECT DATE '{cutoff}' AS target_cutoff,b.*,
                coalesce(ia.events,0)::BIGINT AS item_events_before_cutoff,
                CASE WHEN coalesce(ia.events,0)=0 THEN 0 WHEN ia.events<=5 THEN 1
                     WHEN ia.events<=20 THEN 2 ELSE 3 END::UTINYINT AS coldness_bucket,
                coalesce(ui.events_28d,0)::BIGINT AS user_item_events_28d,
                coalesce(pc.events_28d,0)::BIGINT AS user_product_code_events_28d,
                coalesce(pc.events_12w,0)::BIGINT AS user_product_code_events_12w,
                pc.days_since AS user_product_code_days_since,
                coalesce(pc.events_12w,0)/greatest(b.user_history_events_12w,1.0) AS user_product_code_share_12w,
                coalesce(pt.events_28d,0)::BIGINT AS user_product_type_events_28d,
                pt.days_since AS user_product_type_days_since,
                coalesce(dp.events_28d,0)::BIGINT AS user_department_events_28d,
                dp.days_since AS user_department_days_since,
                coalesce(gg.events_28d,0)::BIGINT AS user_garment_events_28d,
                coalesce(gg.events_12w,0)::BIGINT AS user_garment_events_12w,
                gg.days_since AS user_garment_days_since,
                coalesce(gg.events_12w,0)/greatest(b.user_history_events_12w,1.0) AS user_garment_share_12w,
                coalesce(cl.events_28d,0)::BIGINT AS user_colour_events_28d,
                coalesce(cl.events_12w,0)::BIGINT AS user_colour_events_12w,
                cl.days_since AS user_colour_days_since,
                coalesce(cl.events_12w,0)/greatest(b.user_history_events_12w,1.0) AS user_colour_share_12w,
                coalesce(ig.events_28d,0)::BIGINT AS user_index_group_events_28d,
                coalesce(ig.events_12w,0)::BIGINT AS user_index_group_events_12w,
                ig.days_since AS user_index_group_days_since,
                coalesce(ig.events_12w,0)/greatest(b.user_history_events_12w,1.0) AS user_index_group_share_12w
              FROM base b LEFT JOIN item_all ia USING(article_id)
              LEFT JOIN ui USING(customer_id,article_id)
              LEFT JOIN pc ON b.customer_id=pc.customer_id AND b.article_product_code=pc.article_product_code
              LEFT JOIN pt ON b.customer_id=pt.customer_id AND b.article_product_type_no=pt.article_product_type_no
              LEFT JOIN dp ON b.customer_id=dp.customer_id AND b.article_department_no=dp.article_department_no
              LEFT JOIN gg ON b.customer_id=gg.customer_id AND b.article_garment_group_no=gg.article_garment_group_no
              LEFT JOIN cl ON b.customer_id=cl.customer_id AND b.article_colour_master_id=cl.article_colour_master_id
              LEFT JOIN ig ON b.customer_id=ig.customer_id AND b.article_index_group_no=ig.article_index_group_no
              ORDER BY b.customer_id,b.candidate_rank,b.article_id
            ) TO {_literal(output_path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)
            """
        )
        row = con.execute(
            f"SELECT count(*),count(DISTINCT (customer_id,article_id)),sum(target),"
            f"max((SELECT max(t_dat) FROM history12)),count(*) FILTER(WHERE cold_present=1 AND item_events_before_cutoff>5) "
            f"FROM read_parquet({_literal(output_path)})"
        ).fetchone()
        latest = row[3]
        if int(row[0]) != int(row[1]) or int(row[4]) or (latest is not None and str(latest) >= cutoff):
            raise RuntimeError(f"M5.4 final feature audit failure: {cutoff}")
        return {
            "rows": int(row[0]), "unique_user_item_rows": int(row[1]), "positive_pairs": int(row[2]),
            "latest_history_date": str(latest), "cutoff_safe": latest is None or str(latest) < cutoff,
            "cold_pool_t_le_5": int(row[4]) == 0, "artifact": file_identity(output_path),
        }
    finally:
        con.close()


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# M5.4：时间外推双通道候选与特征物化",
        "",
        "## 结论",
        "",
        f"- 已按 `K_warm={result['contract']['k_warm']}` 与 `K_cold=50` 完成十个历史 cutoff 的 Warm/Cold 去重并集和时间点安全特征物化。",
        f"- stacking hard gate（上游模型防泄漏硬门禁）通过：{result['gate']['passed']}。",
        "- 2019-11-27 没有更早冻结模型，上游 learned score 保持缺失并显式标记；没有用同 cutoff 自拟合分数填充。",
        "- 最终验证周：not_run。",
        "",
        "## 术语与行单位",
        "",
        "- **时间外推/OOF 分数**：模型训练 cutoff 的未来7天标签结束日不晚于当前评分 cutoff；当前行标签没有参与上游模型训练。OOF 是行业常用 out-of-fold 缩写，本项目同时允许严格向前的 forward score。",
        "- **warm_only / cold_only / warm_and_cold**：同一 cutoff—用户—商品分别只被 Warm、只被 Cold、或同时被两路召回；每个数量的单位都是候选行。",
        "- **rank percentile**：来源内名次除以该来源候选数，用于让统一排序器比较两路相对位置；它不是概率。",
        "- **coldness bucket**：cutoff 前全历史事件数0、1--5、6--20、至少21的编码；Cold 路必须只含前两桶。",
        "",
        "## 各 cutoff 物化结果",
        "",
        "| cutoff | rows | users | warm only | cold only | both | positives | warm score rows | cold expert score rows | latest history |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cutoff, row in result["cutoffs"].items():
        branches = row["union"]["branch_rows"]
        lines.append(
            f"| {cutoff} | {row['features']['rows']} | {row['union']['users']} | {branches['warm_only']} | "
            f"{branches['cold_only']} | {branches['warm_and_cold']} | {row['features']['positive_pairs']} | "
            f"{row['warm']['score_available_rows']} | {row['cold']['score_available_rows']} | "
            f"{row['features']['latest_history_date']} |"
        )
    lines.extend([
        "",
        "## 谱系与边界",
        "",
        "- 所有上游模型、类别映射、评分数据库、Cold 候选、Student manifest 与最终 Parquet 均记录 bytes/SHA256。",
        "- Warm 召回/RRF 来源证据从冻结候选资产继承；用户历史、商品趋势、静态属性、用户—商品亲和与 no-decay target-aware 特征在新并集上重新计算。",
        "- 原始 Warm/Cold 分数没有线性相加；统一排序阶段同时保留来源名次、名次百分位和 availability flag。",
        f"- 总耗时 {result['resources']['elapsed_seconds']:.2f} 秒；峰值工作集 {result['resources']['peak_working_set_bytes']/2**30:.2f} GiB；物化文件 {result['resources']['artifact_bytes']/2**30:.2f} GiB。",
        "- optimistic all-articles 与无库存/曝光日志限制不变；本阶段只建立可训练数据，不提前声明 MAP 提升。",
        "",
    ])
    return "\n".join(lines)


def run(
    *, source_root: Path, m4_artifact_dir: Path, m5_artifact_dir: Path,
    artifact_dir: Path, m53_metrics_path: Path, report_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    m53 = json.loads(m53_metrics_path.read_text(encoding="utf-8"))
    if m53.get("status") != "measured" or m53.get("final_week") != "not_run":
        raise RuntimeError("M5.4 requires authoritative M5.3 metrics")
    k_warm = int(m53["decision"]["k_warm"])
    m3_path = source_root / "reports" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / "metrics.json"
    m3 = json.loads(m3_path.read_text(encoding="utf-8"))
    transactions = Path(m3["inputs"]["transactions"]["path"])
    articles = source_root / "data" / "raw" / "articles.csv"
    customers = source_root / "data" / "raw" / "customers.csv"
    static_dir = m4_artifact_dir / "student-v1" / "static_catalog"
    for path in (transactions, articles, customers, static_dir / "catalog_items.csv", static_dir / "catalog_categories.int32.npy"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if artifact_dir.exists() or report_dir.exists():
        raise FileExistsError(artifact_dir if artifact_dir.exists() else report_dir)
    artifact_dir.mkdir(parents=True)
    all_cutoffs = sorted(LINEAGES)
    cutoffs: dict[str, Any] = {}
    artifact_bytes = 0
    for cutoff in all_cutoffs:
        print(f"M5.4 materialize {cutoff}", flush=True)
        root = artifact_dir / cutoff
        root.mkdir()
        lineage = LINEAGES[cutoff]
        score_db, warm_lineage = _warm_score_source(
            cutoff=cutoff, lineage=lineage, source_root=source_root,
            m5_artifact_dir=m5_artifact_dir, output_root=artifact_dir, m3=m3,
        )
        cold_lineage = validate_lineage(cutoff, lineage)
        cold_model, cold_model_evidence = _cold_model(lineage=lineage, m5_artifact_dir=m5_artifact_dir)
        cold_lineage.update(cold_model_evidence)
        warm_source = Path(m3["feature_cache"][cutoff]["artifact"]["path"])
        warm_path = root / "warm_shortlist.parquet"
        warm = _build_warm_shortlist(
            feature_path=warm_source, score_db=score_db, output_path=warm_path, k_warm=k_warm,
        )
        cold_source, cold_mode = _cold_root(cutoff, m5_artifact_dir)
        cold_path = root / "cold_shortlist.parquet"
        cold = _build_cold_shortlist(
            cold_root=cold_source, warm_path=warm_path, static_dir=static_dir,
            model=cold_model, output_path=cold_path,
        )
        union_path = root / "union_candidates.parquet"
        union = _build_union_candidate(warm_path=warm_path, cold_path=cold_path, output_path=union_path)
        base_path = root / "base_features.parquet"
        pit = _prepare_pit_connection(
            database_path=root / "pit-feature-build.duckdb",
            temp_dir=root / "pit-duckdb-temp",
            transactions_path=transactions,
            articles_path=articles,
            customers_path=customers,
        )
        try:
            _create_static_dimensions(pit)
            base = build_point_in_time_dataset(
                pit, union_path, {"cutoff": cutoff, "declared_rows": union["rows"]},
                base_path, M2Config(candidate_k=k_warm + K_COLD, evaluation_role="development"),
                retrieval_features=SOURCE_FEATURES,
                candidate_group_range=(100, k_warm + K_COLD),
            )
        finally:
            pit.close()
        feature_path = root / "features.parquet"
        features = _add_no_decay_features(
            base_path=base_path, output_path=feature_path, transactions_path=transactions,
            articles_path=articles, cutoff=cutoff,
        )
        gate_checks = {
            "lineage_safe": bool(warm_lineage["safe"] and cold_lineage["safe"]),
            "candidate_unique": union["rows"] == union["unique_user_item_rows"],
            "feature_identity_conserved": features["rows"] == union["rows"],
            "cutoff_safe": bool(features["cutoff_safe"]),
            "cold_pool_t_le_5": bool(features["cold_pool_t_le_5"]),
            "final_week_not_run": cutoff < FINAL_CUTOFF,
        }
        if not all(gate_checks.values()):
            raise RuntimeError(f"M5.4 gate failed at {cutoff}: {gate_checks}")
        manifest = {
            "schema_version": "m5.4-dual-channel-cutoff-v1", "status": "completed",
            "run_id": RUN_ID, "cutoff": cutoff, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract": {
                "k_warm": k_warm, "k_cold": K_COLD,
                "branches": ["warm_only", "cold_only", "warm_and_cold"],
                "shared_features": "recomputed on union with behavior strictly before cutoff",
                "target_aware": NO_DECAY_TARGET_FEATURES,
                "raw_score_blending": "none",
            },
            "inputs": {
                "warm_source": file_identity(warm_source), "cold_source_mode": cold_mode,
                "m53_metrics": file_identity(m53_metrics_path),
                "static_catalog_items": file_identity(static_dir / "catalog_items.csv"),
                "static_catalog_categories": file_identity(static_dir / "catalog_categories.int32.npy"),
                "transactions": file_identity(transactions), "articles": file_identity(articles),
                "customers": file_identity(customers),
            },
            "lineage": {"warm_score": warm_lineage, "cold_expert_score": cold_lineage},
            "warm": warm, "cold": cold, "union": union, "base_features": base,
            "features": features, "gate_checks": gate_checks, "final_week": "not_run",
        }
        atomic_json(root / "manifest.json", manifest)
        manifest["manifest"] = file_identity(root / "manifest.json")
        cutoffs[cutoff] = manifest
        artifact_bytes += sum(path.stat().st_size for path in (warm_path, cold_path, union_path, base_path, feature_path, root / "manifest.json"))
        del cold_model
        gc.collect()
    gate_checks = {
        "all_cutoffs_materialized": set(cutoffs) == set(LINEAGES),
        "all_cutoff_gates_pass": all(all(row["gate_checks"].values()) for row in cutoffs.values()),
        "all_available_scores_strict_oof_or_forward": all(
            row["lineage"][kind]["safe"] for row in cutoffs.values()
            for kind in ("warm_score", "cold_expert_score")
        ),
        "final_week_not_run": all(cutoff < FINAL_CUTOFF for cutoff in cutoffs),
    }
    result = {
        "schema_version": "m5.4-temporal-oof-dual-channel-v1", "stage": "M5.4",
        "status": "measured", "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "k_warm": k_warm, "k_cold": K_COLD, "cold_threshold": "item_events_before_cutoff<=5",
            "warm_baseline": "M3.3 Warm-v1", "student": "M4 multimodal Student",
            "cold_expert": "optional OOF auxiliary feature only", "final_week": "not_run",
        },
        "inputs": {"m3_metrics": file_identity(m3_path), "m53_metrics": file_identity(m53_metrics_path)},
        "cutoffs": cutoffs,
        "gate": {"checks": gate_checks, "passed": all(gate_checks.values()), "next_stage": "M5.5_allowed" if all(gate_checks.values()) else "stop"},
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": _peak_working_set_bytes(), "artifact_bytes": artifact_bytes,
        },
        "final_week": "not_run",
    }
    report_dir.mkdir(parents=True)
    atomic_json(report_dir / "M5_4_metrics.json", result)
    (report_dir / "M5_4_FINAL.md").write_text(_render_report(result), encoding="utf-8")
    return result
