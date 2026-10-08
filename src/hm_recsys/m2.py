from __future__ import annotations

import hashlib
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


SCHEMA_VERSION = "m2-binary-v1"
ID_COLUMNS = ["customer_id", "article_id", "candidate_rank", "target"]
RETRIEVAL_FEATURES = ["candidate_rank", "fused_score", "source_count"] + [
    f"{source}_{suffix}"
    for source in SOURCES
    for suffix in ("present", "rank", "score", "rrf_contribution")
]
TABULAR_FEATURES = [
    "customer_age",
    "customer_age_missing",
    "age_bucket",
    "user_history_events_12w",
    "user_unique_items_12w",
    "user_avg_price_12w",
    "user_online_share_12w",
    "user_days_since_last_purchase",
    "item_events_7d",
    "item_events_28d",
    "item_events_12w",
    "item_unique_customers_28d",
    "item_avg_price_28d",
    "item_days_since_last_sale",
    "item_trend_7d_vs_28d",
    "user_item_events_12w",
    "user_item_days_since_last_purchase",
    "user_product_type_events_12w",
    "user_product_type_share_12w",
    "user_department_events_12w",
    "user_department_share_12w",
    "user_item_price_gap",
    "article_product_type_no",
    "article_garment_group_no",
    "article_department_no",
    "article_index_group_no",
    "article_colour_master_id",
]
FULL_FEATURES = RETRIEVAL_FEATURES + TABULAR_FEATURES
CATEGORICAL_FEATURES = [
    "age_bucket",
    "article_product_type_no",
    "article_garment_group_no",
    "article_department_no",
    "article_index_group_no",
    "article_colour_master_id",
]


@dataclass(frozen=True)
class M2Config:
    history_weeks: int = 12
    metric_k: int = 12
    candidate_k: int = 100
    num_boost_round: int = 200
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 100
    threads: int = 8
    prediction_chunk_vectors: int = 64
    seed: int = 20260824
    evaluation_role: str = "final"

    def validate(self) -> None:
        for name in (
            "history_weeks",
            "metric_k",
            "candidate_k",
            "num_boost_round",
            "num_leaves",
            "min_data_in_leaf",
            "threads",
            "prediction_chunk_vectors",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.evaluation_role not in {"development", "final"}:
            raise ValueError("evaluation_role must be development or final")


def _literal(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _date_literal(value: str) -> str:
    parsed = datetime.strptime(value, "%Y-%m-%d")
    return "DATE '" + parsed.strftime("%Y-%m-%d") + "'"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(connection: duckdb.DuckDBPyConnection, sql: str) -> Any:
    row = connection.execute(sql).fetchone()
    if row is None:
        raise RuntimeError("query returned no rows")
    return row[0]


def validate_candidate_artifact(
    candidate_path: Path, manifest_path: Path, candidate_k: int
) -> dict[str, Any]:
    candidate_path = candidate_path.resolve()
    manifest_path = manifest_path.resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    manifest = _read_json(manifest_path)
    schema = manifest.get("schema_version")
    if schema == "m1.5-wide-v1":
        declared = manifest.get("artifacts", {}).get("candidate_features")
        declared_bytes = manifest.get("artifacts", {}).get("candidate_features_bytes")
        declared_rows = manifest.get("wide_evidence", {}).get("rows")
        cutoff = manifest.get("cutoff")
        sample_rate = manifest.get("sample_rate")
        fingerprint = manifest.get("sampled_user_fingerprint")
        catalog = manifest.get("catalog_protocol")
        config = manifest.get("config", {})
    elif schema == "m1.6-profile-v1" and manifest.get("scope") == "hot":
        declared = manifest.get("artifacts", {}).get("candidate_features")
        declared_bytes = manifest.get("artifacts", {}).get("candidate_features_bytes")
        declared_rows = manifest.get("inputs", {}).get("reference_candidates", {}).get("rows")
        identity = manifest.get("cache_identity", {})
        cutoff = identity.get("cutoff")
        sample_rate = identity.get("sample_rate")
        fingerprint = identity.get("sampled_user_fingerprint")
        catalog = identity.get("catalog_protocol")
        config = manifest.get("config", {})
    else:
        raise ValueError(f"unsupported candidate manifest: {schema}")
    if not declared or Path(declared).resolve() != candidate_path:
        raise ValueError("candidate manifest does not point to candidate artifact")
    if declared_bytes is not None and int(declared_bytes) != candidate_path.stat().st_size:
        raise ValueError("candidate artifact bytes differ from manifest")
    if catalog != "optimistic_all_articles":
        raise ValueError("primary M2 requires optimistic_all_articles candidates")
    if config.get("fusion_profile") != "collaborative":
        raise ValueError("primary M2 requires collaborative retrieval-v1")
    if int(config.get("candidate_k", -1)) != candidate_k:
        raise ValueError("candidate_k differs from M2 contract")
    _date_literal(str(cutoff))
    return {
        "schema_version": schema,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "candidate_path": str(candidate_path),
        "candidate_bytes": candidate_path.stat().st_size,
        "candidate_sha256": _sha256(candidate_path),
        "declared_rows": int(declared_rows) if declared_rows is not None else None,
        "cutoff": str(cutoff),
        "sample_rate": float(sample_rate),
        "sampled_user_fingerprint": str(fingerprint),
        "catalog_protocol": str(catalog),
        "fusion_profile": str(config.get("fusion_profile")),
        "candidate_k": int(config.get("candidate_k")),
    }


def _validate_candidate_relation(
    connection: duckdb.DuckDBPyConnection,
    candidate_path: Path,
    identity: dict[str, Any],
    candidate_k: int,
) -> dict[str, Any]:
    path = _literal(candidate_path)
    required = set(RETRIEVAL_FEATURES + ["customer_id", "article_id"])
    schema_rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({path})"
    ).fetchall()
    schema = {row[0]: row[1] for row in schema_rows}
    missing = sorted(required - set(schema))
    if missing:
        raise ValueError(f"candidate feature columns missing: {missing}")
    stats = connection.execute(
        f"""
        SELECT count(*), count(DISTINCT (customer_id, article_id)),
               count(DISTINCT customer_id), min(candidate_rank), max(candidate_rank),
               count(*) FILTER (WHERE candidate_rank < 1 OR candidate_rank > {candidate_k})
        FROM read_parquet({path})
        """
    ).fetchone()
    if stats is None:
        raise RuntimeError("candidate stats query returned no rows")
    rows, unique_rows, users, min_rank, max_rank, rank_errors = map(int, stats)
    if rows != unique_rows:
        raise ValueError("candidate artifact is not unique by customer-item")
    if rank_errors or min_rank != 1 or max_rank != candidate_k:
        raise ValueError("candidate ranks violate the fixed TopK contract")
    if rows != users * candidate_k:
        raise ValueError("candidate groups are not exactly candidate_k rows per user")
    if identity["declared_rows"] is not None and rows != identity["declared_rows"]:
        raise ValueError("candidate row count differs from manifest")
    return {"rows": rows, "unique_rows": unique_rows, "users": users}


def _create_static_dimensions(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m2_articles AS
        SELECT article_id,
               try_cast(product_code AS INTEGER) AS article_product_code,
               try_cast(product_type_no AS INTEGER) AS article_product_type_no,
               try_cast(garment_group_no AS INTEGER) AS article_garment_group_no,
               try_cast(department_no AS INTEGER) AS article_department_no,
               try_cast(index_group_no AS INTEGER) AS article_index_group_no,
               try_cast(perceived_colour_master_id AS INTEGER) AS article_colour_master_id
        FROM articles
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m2_customers AS
        SELECT customer_id, try_cast(age AS DOUBLE) AS customer_age
        FROM customers
        """
    )


def _retrieval_select(
    prefix: str = "c", features: list[str] | None = None
) -> str:
    selected = RETRIEVAL_FEATURES if features is None else features
    return ",\n               ".join(f"{prefix}.{name}" for name in selected)


def build_point_in_time_dataset(
    connection: duckdb.DuckDBPyConnection,
    candidate_path: Path,
    candidate_identity: dict[str, Any],
    output_path: Path,
    config: M2Config,
    retrieval_features: list[str] | None = None,
    candidate_group_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    cutoff = candidate_identity["cutoff"]
    cutoff_sql = _date_literal(cutoff)
    candidates = _literal(candidate_path)
    output = _literal(output_path)
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW m2_candidates AS SELECT * FROM read_parquet({candidates})"
    )
    if candidate_group_range is None:
        relation = _validate_candidate_relation(
            connection, candidate_path, candidate_identity, config.candidate_k
        )
    else:
        minimum, maximum = candidate_group_range
        if minimum <= 0 or maximum < minimum:
            raise ValueError("invalid candidate_group_range")
        selected_features = (
            RETRIEVAL_FEATURES if retrieval_features is None else retrieval_features
        )
        required = set(selected_features + ["customer_id", "article_id"])
        schema_rows = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({candidates})"
        ).fetchall()
        missing = sorted(required - {row[0] for row in schema_rows})
        if missing:
            raise ValueError(f"candidate feature columns missing: {missing}")
        stats = connection.execute(
            f"""
            WITH groups AS (
                SELECT customer_id, count(*) AS rows,
                       min(candidate_rank) AS min_rank,
                       max(candidate_rank) AS max_rank,
                       count(DISTINCT candidate_rank) AS distinct_ranks
                FROM m2_candidates GROUP BY customer_id
            )
            SELECT (SELECT count(*) FROM m2_candidates),
                   (SELECT count(DISTINCT (customer_id, article_id)) FROM m2_candidates),
                   count(*), min(rows), max(rows),
                   count(*) FILTER (
                       WHERE rows < {minimum} OR rows > {maximum}
                          OR min_rank <> 1 OR max_rank <> rows
                          OR distinct_ranks <> rows
                   )
            FROM groups
            """
        ).fetchone()
        if stats is None:
            raise RuntimeError("candidate stats query returned no rows")
        rows, unique_rows, users, min_rows, max_rows, invalid_groups = map(int, stats)
        if rows != unique_rows:
            raise ValueError("candidate artifact is not unique by customer-item")
        if invalid_groups:
            raise ValueError(
                "candidate groups violate variable TopK contract: "
                f"invalid_groups={invalid_groups}"
            )
        relation = {
            "rows": rows,
            "unique_rows": unique_rows,
            "users": users,
            "min_group_rows": min_rows,
            "max_group_rows": max_rows,
        }
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE m2_users AS SELECT DISTINCT customer_id FROM m2_candidates"
    )
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE m2_items AS SELECT DISTINCT article_id FROM m2_candidates"
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m2_global_history AS
        SELECT t.* FROM transactions t
        WHERE t.t_dat >= {cutoff_sql} - INTERVAL {config.history_weeks} WEEK
          AND t.t_dat < {cutoff_sql}
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m2_user_history AS
        SELECT h.* FROM m2_global_history h SEMI JOIN m2_users USING (customer_id)
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m2_truth AS
        SELECT DISTINCT t.customer_id, t.article_id
        FROM transactions t SEMI JOIN m2_users USING (customer_id)
        WHERE t.t_dat >= {cutoff_sql}
          AND t.t_dat < {cutoff_sql} + INTERVAL 7 DAY
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m2_user_features AS
        SELECT u.customer_id,
               count(h.article_id) AS user_history_events_12w,
               count(DISTINCT h.article_id) AS user_unique_items_12w,
               avg(h.price) AS user_avg_price_12w,
               avg((h.sales_channel_id = 2)::INTEGER) AS user_online_share_12w,
               date_diff('day', max(h.t_dat), {cutoff_sql}) AS user_days_since_last_purchase
        FROM m2_users u LEFT JOIN m2_user_history h USING (customer_id)
        GROUP BY u.customer_id
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m2_item_features AS
        SELECT i.article_id,
               count(h.article_id) FILTER (WHERE h.t_dat >= {cutoff_sql} - INTERVAL 7 DAY) AS item_events_7d,
               count(h.article_id) FILTER (WHERE h.t_dat >= {cutoff_sql} - INTERVAL 28 DAY) AS item_events_28d,
               count(h.article_id) AS item_events_12w,
               count(DISTINCT h.customer_id) FILTER (WHERE h.t_dat >= {cutoff_sql} - INTERVAL 28 DAY) AS item_unique_customers_28d,
               avg(h.price) FILTER (WHERE h.t_dat >= {cutoff_sql} - INTERVAL 28 DAY) AS item_avg_price_28d,
               date_diff('day', max(h.t_dat), {cutoff_sql}) AS item_days_since_last_sale
        FROM m2_items i LEFT JOIN m2_global_history h USING (article_id)
        GROUP BY i.article_id
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m2_user_item_features AS
        SELECT h.customer_id, h.article_id, count(*) AS user_item_events_12w,
               date_diff('day', max(h.t_dat), {cutoff_sql}) AS user_item_days_since_last_purchase
        FROM m2_user_history h SEMI JOIN m2_candidates c USING (customer_id, article_id)
        GROUP BY h.customer_id, h.article_id
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m2_user_type_features AS
        SELECT h.customer_id, a.article_product_type_no,
               count(*) AS user_product_type_events_12w
        FROM m2_user_history h JOIN m2_articles a USING (article_id)
        GROUP BY h.customer_id, a.article_product_type_no
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE m2_user_department_features AS
        SELECT h.customer_id, a.article_department_no,
               count(*) AS user_department_events_12w
        FROM m2_user_history h JOIN m2_articles a USING (article_id)
        GROUP BY h.customer_id, a.article_department_no
        """
    )
    select_sql = f"""
        SELECT c.customer_id, c.article_id,
               {_retrieval_select(features=retrieval_features)},
               cust.customer_age,
               (cust.customer_age IS NULL)::INTEGER AS customer_age_missing,
               coalesce(floor(cust.customer_age / 5), -1)::INTEGER AS age_bucket,
               uf.user_history_events_12w, uf.user_unique_items_12w,
               uf.user_avg_price_12w, uf.user_online_share_12w,
               uf.user_days_since_last_purchase,
               item.item_events_7d, item.item_events_28d, item.item_events_12w,
               item.item_unique_customers_28d, item.item_avg_price_28d,
               item.item_days_since_last_sale,
               (item.item_events_7d + 1.0) / (item.item_events_28d / 4.0 + 1.0)
                   AS item_trend_7d_vs_28d,
               coalesce(ui.user_item_events_12w, 0) AS user_item_events_12w,
               ui.user_item_days_since_last_purchase,
               coalesce(ut.user_product_type_events_12w, 0) AS user_product_type_events_12w,
               coalesce(ut.user_product_type_events_12w, 0) / greatest(uf.user_history_events_12w, 1)
                   AS user_product_type_share_12w,
               coalesce(ud.user_department_events_12w, 0) AS user_department_events_12w,
               coalesce(ud.user_department_events_12w, 0) / greatest(uf.user_history_events_12w, 1)
                   AS user_department_share_12w,
               abs(item.item_avg_price_28d - uf.user_avg_price_12w) AS user_item_price_gap,
               a.article_product_code, a.article_product_type_no,
               a.article_garment_group_no, a.article_department_no,
               a.article_index_group_no, a.article_colour_master_id,
               (truth.article_id IS NOT NULL)::UTINYINT AS target
        FROM m2_candidates c
        JOIN m2_articles a USING (article_id)
        LEFT JOIN m2_customers cust USING (customer_id)
        LEFT JOIN m2_user_features uf USING (customer_id)
        LEFT JOIN m2_item_features item USING (article_id)
        LEFT JOIN m2_user_item_features ui USING (customer_id, article_id)
        LEFT JOIN m2_user_type_features ut
          ON c.customer_id = ut.customer_id
         AND a.article_product_type_no = ut.article_product_type_no
        LEFT JOIN m2_user_department_features ud
          ON c.customer_id = ud.customer_id
         AND a.article_department_no = ud.article_department_no
        LEFT JOIN m2_truth truth USING (customer_id, article_id)
    """
    started = time.perf_counter()
    connection.execute(
        f"COPY ({select_sql} ORDER BY customer_id, candidate_rank) TO {output} "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
    )
    elapsed = time.perf_counter() - started
    stats = connection.execute(
        f"""
        SELECT count(*), count(DISTINCT (customer_id, article_id)),
               count(DISTINCT customer_id), sum(target),
               count(DISTINCT customer_id) FILTER (WHERE target = 1),
               count(*) FILTER (WHERE target NOT IN (0, 1))
        FROM read_parquet({output})
        """
    ).fetchone()
    if stats is None:
        raise RuntimeError("M2 dataset stats query returned no rows")
    rows, unique_rows, users, positives, positive_users, target_errors = map(int, stats)
    if rows != relation["rows"] or unique_rows != rows or users != relation["users"]:
        raise RuntimeError("M2 dataset changed candidate identity")
    if target_errors:
        raise RuntimeError("M2 target contains values outside {0,1}")
    truth_pairs = int(_scalar(connection, "SELECT count(*) FROM m2_truth"))
    truth_users = int(_scalar(connection, "SELECT count(DISTINCT customer_id) FROM m2_truth"))
    latest_behavior = _scalar(connection, "SELECT max(t_dat) FROM m2_global_history")
    if latest_behavior is not None and str(latest_behavior) >= cutoff:
        raise RuntimeError("point-in-time feature leakage detected")
    return {
        "cutoff": cutoff,
        "rows": rows,
        "users": users,
        "positives": positives,
        "positive_users": positive_users,
        "truth_pairs": truth_pairs,
        "truth_users": truth_users,
        "positive_rate": positives / rows,
        "latest_behavior_date": str(latest_behavior),
        "history_filter": f"t_dat >= cutoff - {config.history_weeks} weeks AND t_dat < cutoff",
        "label_filter": "cutoff <= t_dat < cutoff + 7 days",
        "dataset_path": str(output_path.resolve()),
        "dataset_bytes": output_path.stat().st_size,
        "dataset_sha256": _sha256(output_path),
        "build_seconds": elapsed,
    }


def build_category_maps(frame: pd.DataFrame) -> dict[str, dict[int, int]]:
    mappings: dict[str, dict[int, int]] = {}
    for feature in CATEGORICAL_FEATURES:
        values = pd.to_numeric(frame[feature], errors="coerce").dropna().astype(np.int64)
        unique = sorted(int(value) for value in values.unique())
        mappings[feature] = {value: index + 1 for index, value in enumerate(unique)}
    return mappings


def _prepare_frame(
    frame: pd.DataFrame,
    features: list[str],
    category_maps: dict[str, dict[int, int]] | None = None,
) -> pd.DataFrame:
    prepared = pd.DataFrame(index=frame.index)
    category_maps = category_maps or {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        if feature in CATEGORICAL_FEATURES:
            mapping = category_maps.get(feature)
            if mapping is None:
                raise ValueError(f"missing fitted category map: {feature}")
            prepared[feature] = values.map(mapping).fillna(0).astype(np.int32)
        else:
            prepared[feature] = values.astype(np.float32)
    return prepared


def _load_training_frame(path: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        selected = ", ".join(FULL_FEATURES + ["target"])
        return connection.execute(
            f"SELECT {selected} FROM read_parquet({_literal(path)})"
        ).fetchdf()
    finally:
        connection.close()


def _train_model(
    train_frame: pd.DataFrame,
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
    x_train = _prepare_frame(train_frame, features, category_maps)
    labels = train_frame["target"].astype(np.uint8)
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    dataset = lgb.Dataset(
        x_train,
        label=labels,
        feature_name=features,
        categorical_feature=categorical,
        free_raw_data=False,
    )
    dataset.construct()
    binary_path = artifact_dir / f"train-{name}.bin"
    dataset.save_binary(str(binary_path))
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "average_precision"],
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
    final_metrics = {
        metric: float(values[-1])
        for metric, values in evaluations.get("train", {}).items()
    }
    return model, {
        "name": name,
        "features": features,
        "categorical_features": categorical,
        "category_cardinalities": {
            feature: len(category_maps[feature]) for feature in categorical
        },
        "parameters": params,
        "num_boost_round": config.num_boost_round,
        "train_metrics": final_metrics,
        "top_feature_importance": importance[:25],
        "model_path": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "binary_dataset_path": str(binary_path.resolve()),
        "binary_dataset_bytes": binary_path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _score_validation(
    dataset_path: Path,
    models: dict[str, Any],
    evaluation_db: Path,
    prediction_path: Path,
    config: M2Config,
    category_maps: dict[str, dict[int, int]],
) -> dict[str, Any]:
    if evaluation_db.exists() or prediction_path.exists():
        raise FileExistsError(evaluation_db if evaluation_db.exists() else prediction_path)
    started = time.perf_counter()
    source = duckdb.connect()
    target = duckdb.connect(str(evaluation_db))
    temp_dir = evaluation_db.parent / "duckdb-temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target.execute(f"SET threads = {config.threads}")
    target.execute("SET memory_limit = '11GB'")
    target.execute(f"SET temp_directory = {_literal(temp_dir)}")
    target.execute(
        """
        CREATE TABLE predictions(
            customer_id VARCHAR, article_id VARCHAR, candidate_rank BIGINT,
            target UTINYINT, user_history_events_12w BIGINT,
            score_retrieval DOUBLE, score_full DOUBLE
        )
        """
    )
    selected = ", ".join(
        ["customer_id", "article_id", "candidate_rank", "target", "user_history_events_12w"]
        + FULL_FEATURES
    )
    cursor = source.execute(
        f"SELECT {selected} FROM read_parquet({_literal(dataset_path)})"
    )
    rows = 0
    batches = 0
    try:
        while True:
            batch = cursor.fetch_df_chunk(config.prediction_chunk_vectors)
            if batch.empty:
                break
            output = batch[
                ["customer_id", "article_id", "candidate_rank", "target", "user_history_events_12w"]
            ].copy()
            output["score_retrieval"] = models["retrieval"].predict(
                _prepare_frame(batch, RETRIEVAL_FEATURES, category_maps)
            )
            output["score_full"] = models["full"].predict(
                _prepare_frame(batch, FULL_FEATURES, category_maps)
            )
            target.register("m2_prediction_batch", output)
            target.execute("INSERT INTO predictions SELECT * FROM m2_prediction_batch")
            target.unregister("m2_prediction_batch")
            rows += len(output)
            batches += 1
        duplicate_rows = int(
            _scalar(
                target,
                "SELECT count(*) - count(DISTINCT (customer_id, article_id)) FROM predictions",
            )
        )
        if duplicate_rows:
            raise RuntimeError("prediction rows are not unique by customer-item")
        target.execute(
            f"COPY predictions TO {_literal(prediction_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
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


def _prepare_evaluation_truth(
    connection: duckdb.DuckDBPyConnection, transactions_path: Path, cutoff: str
) -> dict[str, int]:
    cutoff_sql = _date_literal(cutoff)
    transactions = _literal(transactions_path)
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m2_eval_users AS
        SELECT DISTINCT customer_id FROM predictions;
        CREATE OR REPLACE TEMP TABLE m2_warm_catalog AS
        SELECT DISTINCT article_id FROM read_parquet({transactions})
        WHERE t_dat < {cutoff_sql};
        CREATE OR REPLACE TEMP TABLE m2_eval_truth AS
        SELECT truth.customer_id, truth.article_id,
               CASE WHEN warm.article_id IS NULL THEN 'cold' ELSE 'warm' END AS item_temperature
        FROM (
            SELECT DISTINCT t.customer_id, t.article_id
            FROM read_parquet({transactions}) t
            SEMI JOIN m2_eval_users USING (customer_id)
            WHERE t.t_dat >= {cutoff_sql}
              AND t.t_dat < {cutoff_sql} + INTERVAL 7 DAY
        ) truth
        LEFT JOIN m2_warm_catalog warm USING (article_id)
        """
    )
    mismatch = int(
        _scalar(
            connection,
            """
            SELECT count(*) FROM predictions p
            LEFT JOIN m2_eval_truth t USING (customer_id, article_id)
            WHERE p.target != (t.article_id IS NOT NULL)::UTINYINT
            """,
        )
    )
    if mismatch:
        raise RuntimeError("validation targets do not match raw temporal truth")
    return {
        "users": int(_scalar(connection, "SELECT count(*) FROM m2_eval_users")),
        "overall_pairs": int(_scalar(connection, "SELECT count(*) FROM m2_eval_truth")),
        "warm_pairs": int(
            _scalar(connection, "SELECT count(*) FROM m2_eval_truth WHERE item_temperature='warm'")
        ),
        "cold_pairs": int(
            _scalar(connection, "SELECT count(*) FROM m2_eval_truth WHERE item_temperature='cold'")
        ),
        "target_mismatch_rows": mismatch,
    }


def _evaluate_segment(
    connection: duckdb.DuckDBPyConnection,
    segment: str,
    metric_k: int,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    if segment not in {"overall", "warm", "cold"}:
        raise ValueError(f"unknown evaluation segment: {segment}")
    predicate = "TRUE" if segment == "overall" else f"item_temperature = '{segment}'"
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m2_segment_truth AS
        SELECT customer_id, article_id FROM m2_eval_truth WHERE {predicate};
        CREATE OR REPLACE TEMP TABLE m2_segment_user_metrics AS
        WITH truth_counts AS (
            SELECT customer_id, count(*) AS truth_count
            FROM m2_segment_truth GROUP BY customer_id
        ),
        top_scored AS (
            SELECT ranked.customer_id, ranked.article_id, ranked.pred_rank,
                   (truth.article_id IS NOT NULL)::INTEGER AS is_hit
            FROM m2_ranked ranked
            LEFT JOIN m2_segment_truth truth USING (customer_id, article_id)
            WHERE ranked.pred_rank <= {metric_k}
        ),
        precision_rows AS (
            SELECT *, sum(is_hit) OVER (
                PARTITION BY customer_id ORDER BY pred_rank
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_hits
            FROM top_scored
        ),
        ranking_agg AS (
            SELECT customer_id, sum(is_hit) AS topk_hits,
                   sum(CASE WHEN is_hit = 1 THEN cumulative_hits / pred_rank ELSE 0 END)
                       AS precision_sum
            FROM precision_rows GROUP BY customer_id
        ),
        candidate_agg AS (
            SELECT predictions.customer_id, count(truth.article_id) AS candidate_hits
            FROM predictions
            JOIN truth_counts USING (customer_id)
            LEFT JOIN m2_segment_truth truth USING (customer_id, article_id)
            GROUP BY predictions.customer_id
        ),
        activity AS (
            SELECT customer_id, max(user_history_events_12w) AS history_events
            FROM predictions GROUP BY customer_id
        )
        SELECT truth_counts.customer_id, truth_counts.truth_count,
               coalesce(candidate_agg.candidate_hits, 0) AS candidate_hits,
               coalesce(ranking_agg.topk_hits, 0) AS topk_hits,
               coalesce(ranking_agg.precision_sum, 0) /
                   least(truth_counts.truth_count, {metric_k}) AS ap_at_k,
               coalesce(ranking_agg.topk_hits, 0)::DOUBLE /
                   truth_counts.truth_count AS recall_at_k,
               (coalesce(ranking_agg.topk_hits, 0) > 0)::INTEGER AS hit_at_k,
               coalesce(candidate_agg.candidate_hits, 0)::DOUBLE /
                   truth_counts.truth_count AS candidate_recall,
               (coalesce(candidate_agg.candidate_hits, 0) > 0)::INTEGER AS candidate_hit,
               least(coalesce(candidate_agg.candidate_hits, 0), {metric_k})::DOUBLE /
                   least(truth_counts.truth_count, {metric_k}) AS oracle_ap_at_k,
               CASE
                   WHEN activity.history_events = 0 THEN 'inactive_12w'
                   WHEN activity.history_events <= 5 THEN 'low_1_5'
                   WHEN activity.history_events <= 20 THEN 'medium_6_20'
                   ELSE 'high_21_plus'
               END AS activity_segment
        FROM truth_counts
        LEFT JOIN candidate_agg USING (customer_id)
        LEFT JOIN ranking_agg USING (customer_id)
        LEFT JOIN activity USING (customer_id)
        """
    )
    row = connection.execute(
        """
        SELECT count(*), sum(truth_count), avg(candidate_recall), avg(candidate_hit),
               avg(oracle_ap_at_k), avg(ap_at_k), avg(recall_at_k), avg(hit_at_k)
        FROM m2_segment_user_metrics
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("segment evaluation returned no rows")
    if int(row[0]) == 0:
        return (
            {
                "users": 0,
                "truth_pairs": 0,
                "candidate_recall@100": 0.0,
                "candidate_hit_rate@100": 0.0,
                f"oracle_map@{metric_k}": 0.0,
                f"map@{metric_k}": 0.0,
                f"recall@{metric_k}": 0.0,
                f"hit_rate@{metric_k}": 0.0,
            },
            [],
        )
    metrics: dict[str, float | int] = {
        "users": int(row[0]),
        "truth_pairs": int(row[1]),
        "candidate_recall@100": float(row[2]),
        "candidate_hit_rate@100": float(row[3]),
        f"oracle_map@{metric_k}": float(row[4]),
        f"map@{metric_k}": float(row[5]),
        f"recall@{metric_k}": float(row[6]),
        f"hit_rate@{metric_k}": float(row[7]),
    }
    activity_rows = connection.execute(
        f"""
        SELECT activity_segment, count(*), sum(truth_count),
               avg(candidate_recall), avg(oracle_ap_at_k), avg(ap_at_k),
               avg(recall_at_k), avg(hit_at_k)
        FROM m2_segment_user_metrics
        GROUP BY activity_segment ORDER BY activity_segment
        """
    ).fetchall()
    activity = [
        {
            "activity_segment": str(value[0]),
            "users": int(value[1]),
            "truth_pairs": int(value[2]),
            "candidate_recall@100": float(value[3]),
            f"oracle_map@{metric_k}": float(value[4]),
            f"map@{metric_k}": float(value[5]),
            f"recall@{metric_k}": float(value[6]),
            f"hit_rate@{metric_k}": float(value[7]),
        }
        for value in activity_rows
    ]
    return metrics, activity


def _evaluate_ordering(
    connection: duckdb.DuckDBPyConnection,
    name: str,
    order_expression: str,
    metric_k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m2_ranked AS
        SELECT customer_id, article_id,
               row_number() OVER (
                   PARTITION BY customer_id
                   ORDER BY {order_expression}, candidate_rank, article_id
               ) AS pred_rank
        FROM predictions
        """
    )
    segments: dict[str, Any] = {}
    activity: list[dict[str, Any]] = []
    for segment in ("overall", "warm", "cold"):
        metrics, segment_activity = _evaluate_segment(connection, segment, metric_k)
        segments[segment] = metrics
        if segment == "overall":
            activity = segment_activity
    connection.execute("DROP TABLE m2_ranked")
    return {
        "name": name,
        "segments": segments,
        "activity_segments": activity,
        "elapsed_seconds": time.perf_counter() - started,
    }


def evaluate_predictions(
    evaluation_db: Path,
    transactions_path: Path,
    cutoff: str,
    metric_k: int,
) -> dict[str, Any]:
    connection = duckdb.connect(str(evaluation_db))
    try:
        connection.execute("SET threads = 8")
        truth = _prepare_evaluation_truth(connection, transactions_path, cutoff)
        orderings = {
            "rrf": _evaluate_ordering(
                connection, "rrf", "candidate_rank ASC", metric_k
            ),
            "lightgbm_retrieval": _evaluate_ordering(
                connection, "lightgbm_retrieval", "score_retrieval DESC", metric_k
            ),
            "lightgbm_full": _evaluate_ordering(
                connection, "lightgbm_full", "score_full DESC", metric_k
            ),
        }
        return {"truth": truth, "orderings": orderings}
    finally:
        connection.close()


def _reference_parity(
    evaluation: dict[str, Any], reference_path: Path, tolerance: float = 1e-12
) -> dict[str, Any]:
    reference = _read_json(reference_path)
    if "ordering_diagnostic" in reference:
        rrf_map = reference["ordering_diagnostic"]["map@12_untuned_rrf"]
    else:
        final_metrics = reference.get("final_metrics", {})
        map_values = [
            value
            for key, value in final_metrics.items()
            if key.startswith("map@12_untuned_")
        ]
        if len(map_values) != 1:
            raise ValueError("reference metrics must contain one untuned RRF MAP@12")
        rrf_map = map_values[0]
    expected = {
        "candidate_recall@100": float(reference["metrics"]["overall"]["recall@100"]),
        "candidate_hit_rate@100": float(reference["metrics"]["overall"]["hit_rate@100"]),
        "oracle_map@12": float(reference["metrics"]["overall"]["oracle_map@12"]),
        "map@12": float(rrf_map),
    }
    actual = evaluation["orderings"]["rrf"]["segments"]["overall"]
    differences = {key: abs(float(actual[key]) - value) for key, value in expected.items()}
    status = "passed" if max(differences.values(), default=0.0) <= tolerance else "failed"
    result = {
        "status": status,
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": _sha256(reference_path),
        "tolerance": tolerance,
        "expected": expected,
        "differences": differences,
    }
    if status != "passed":
        raise RuntimeError(f"M2 evaluation parity failed: {differences}")
    return result
