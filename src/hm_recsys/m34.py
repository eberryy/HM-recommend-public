from __future__ import annotations

import gc
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m2 import CATEGORICAL_FEATURES, _literal, _prepare_frame, _write_json
from .m25_retrieval import _candidate_metrics
from .m28 import _file_identity
from .m29 import CANDIDATE_SCHEMA_VERSION
from .m33 import (
    SEASON_SENSITIVE_MIN_PRIOR_EVENTS,
    SEASON_SENSITIVE_PRODUCT_TYPE_LIFT,
    variant_feature_sets,
)


SCHEMA_VERSION = "m3.4-season-aware-retrieval-v1"
SOURCE_SCHEMA_VERSION = "m3.4-seasonal-retrieval-source-v1"

OUTER_WINDOWS = {
    "winter_20200122": "2020-01-22",
    "spring_20200318": "2020-03-18",
    "early_summer_20200624": "2020-06-24",
    "late_summer_20200819": "2020-08-19",
}

SOURCE_CONFIG = {
    "history_weeks": 12,
    "current_days": 28,
    "prior_background_days": 84,
    "prior_same_period_days": 28,
    "seasonal_lift_threshold": SEASON_SENSITIVE_PRODUCT_TYPE_LIFT,
    "minimum_prior_28d_events": SEASON_SENSITIVE_MIN_PRIOR_EVENTS,
    "current_items_per_type": 30,
    "user_top_types": 5,
    "current_global_k": 50,
    "prior_direct_k": 50,
    "source_k": 100,
    "rrf_constant": 60,
    "lane_weights": {
        "personalized_current": 1.0,
        "current_global": 1.0,
        "prior_direct": 1.0,
    },
}

VARIANT_BUDGETS = {
    "base_300": 300,
    "seasonal_standalone_100": 100,
    "vacancy_fill_300": 300,
    "competitive_300": 300,
    "expanded_350": 350,
}

GAP_WINDOWS = (
    "spring_20200318",
    "early_summer_20200624",
    "late_summer_20200819",
)


def validate_protocol() -> None:
    if list(OUTER_WINDOWS) != [
        "winter_20200122",
        "spring_20200318",
        "early_summer_20200624",
        "late_summer_20200819",
    ]:
        raise RuntimeError("M3.4 requires exactly four ordered outer representatives")
    if any(cutoff >= "2020-09-16" for cutoff in OUTER_WINDOWS.values()):
        raise RuntimeError("M3.4 must not read the final validation week")
    if SOURCE_CONFIG["lane_weights"] != {
        "personalized_current": 1.0,
        "current_global": 1.0,
        "prior_direct": 1.0,
    }:
        raise RuntimeError("M3.4 lane weights must remain pre-registered and equal")
    if VARIANT_BUDGETS != {
        "base_300": 300,
        "seasonal_standalone_100": 100,
        "vacancy_fill_300": 300,
        "competitive_300": 300,
        "expanded_350": 350,
    }:
        raise RuntimeError("M3.4 candidate variants drifted")


def _date(value: str) -> str:
    return "DATE '" + value.replace("'", "''") + "'"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_base(candidate_path: Path, manifest_path: Path, cutoff: str) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError(f"M3.4 requires M2.9 expanded candidates: {manifest_path}")
    if manifest.get("status") != "completed" or manifest.get("cutoff") != cutoff:
        raise ValueError(f"M3.4 base candidate identity drift: {manifest_path}")
    actual = _file_identity(candidate_path)
    declared = manifest["artifact"]
    if actual["bytes"] != declared["bytes"] or actual["sha256"] != declared["sha256"]:
        raise ValueError(f"M3.4 base candidate bytes/SHA drift: {candidate_path}")
    audit = manifest["audit"]
    if int(audit["invalid_groups"]) or int(audit["min_group_rows"]) < 100:
        raise ValueError(f"M3.4 invalid base groups: {candidate_path}")
    if int(audit["max_group_rows"]) > 300:
        raise ValueError(f"M3.4 base exceeds candidate budget: {candidate_path}")
    return {"manifest": manifest, "artifact": actual}


def _source_paths(cache_dir: Path, cutoff: str) -> tuple[Path, Path, Path]:
    root = cache_dir / cutoff
    return (
        root / "seasonal-candidates.parquet",
        root / "qualifying-product-types.parquet",
        root / "source-manifest.json",
    )


def _validate_source_cache(
    candidate_path: Path,
    type_path: Path,
    manifest_path: Path,
    *,
    cutoff: str,
    base_identity: dict[str, Any],
    transaction_identity: dict[str, Any],
    article_identity: dict[str, Any],
) -> dict[str, Any] | None:
    existing = [candidate_path.exists(), type_path.exists(), manifest_path.exists()]
    if not any(existing):
        return None
    if not all(existing):
        raise FileExistsError(f"incomplete M3.4 source cache: {candidate_path.parent}")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != SOURCE_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("cutoff") != cutoff
        or manifest.get("contract") != SOURCE_CONFIG
    ):
        raise ValueError(f"M3.4 source manifest drift: {manifest_path}")
    expected_inputs = {
        "base": base_identity,
        "transactions": transaction_identity,
        "articles": article_identity,
    }
    for name, expected in expected_inputs.items():
        actual = manifest["inputs"][name]
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError(f"M3.4 source input drift: {name} {manifest_path}")
    for name, path in (("candidates", candidate_path), ("qualifying_types", type_path)):
        actual = _file_identity(path)
        declared = manifest["artifacts"][name]
        if actual["bytes"] != declared["bytes"] or actual["sha256"] != declared["sha256"]:
            raise ValueError(f"M3.4 source artifact drift: {path}")
    return manifest


def build_seasonal_source_cache(
    *,
    raw_dir: Path,
    transactions_path: Path,
    base_candidates: dict[str, tuple[Path, Path]],
    cache_dir: Path,
    windows: dict[str, str] | None = None,
) -> dict[str, Any]:
    selected_windows = windows or OUTER_WINDOWS
    if set(base_candidates) != set(selected_windows.values()):
        raise ValueError("M3.4 source inputs must exactly match selected cutoffs")
    transaction_identity = _file_identity(transactions_path)
    article_path = raw_dir / "articles.csv"
    article_identity = _file_identity(article_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for window, cutoff in selected_windows.items():
        started = time.perf_counter()
        candidate_path, base_manifest_path = base_candidates[cutoff]
        base = _validate_base(candidate_path, base_manifest_path, cutoff)["artifact"]
        source_path, type_path, manifest_path = _source_paths(cache_dir, cutoff)
        existing = _validate_source_cache(
            source_path,
            type_path,
            manifest_path,
            cutoff=cutoff,
            base_identity=base,
            transaction_identity=transaction_identity,
            article_identity=article_identity,
        )
        if existing is not None:
            results[cutoff] = existing
            continue
        root = source_path.parent
        if root.exists():
            raise FileExistsError(f"incomplete M3.4 source cache root: {root}")
        root.mkdir(parents=True)
        temp_dir = root / "duckdb-temp"
        temp_dir.mkdir()
        connection = duckdb.connect(str(root / "source-build.duckdb"))
        cutoff_sql = _date(cutoff)
        try:
            connection.execute("SET threads=8")
            connection.execute("SET memory_limit='11GB'")
            connection.execute("PRAGMA disable_progress_bar")
            connection.execute(f"SET temp_directory={_literal(temp_dir)}")
            connection.execute(
                f"""
                CREATE TABLE base AS
                SELECT * FROM read_parquet({_literal(candidate_path)});
                CREATE TABLE eval_users AS SELECT DISTINCT customer_id FROM base;
                CREATE TABLE article_dim AS
                SELECT article_id,try_cast(product_type_no AS INTEGER) AS product_type_no
                FROM read_csv_auto({_literal(article_path)},header=true,all_varchar=true);
                CREATE TABLE prior_events AS
                SELECT t.t_dat,t.article_id,a.product_type_no
                FROM read_parquet({_literal(transactions_path)}) t
                JOIN article_dim a USING(article_id)
                WHERE t.t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 84 DAY
                  AND t.t_dat<{cutoff_sql}-INTERVAL 1 YEAR;
                CREATE TABLE current_events AS
                SELECT t.t_dat,t.article_id,a.product_type_no
                FROM read_parquet({_literal(transactions_path)}) t
                JOIN article_dim a USING(article_id)
                WHERE t.t_dat>={cutoff_sql}-INTERVAL 28 DAY AND t.t_dat<{cutoff_sql};
                """
            )
            prior84, prior28 = connection.execute(
                f"""
                SELECT count(*),count(*) FILTER(
                    WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY)
                FROM prior_events
                """
            ).fetchone()
            current28 = int(connection.execute("SELECT count(*) FROM current_events").fetchone()[0])
            if min(int(prior84), int(prior28), current28) <= 0:
                raise RuntimeError(f"M3.4 empty seasonal history: {cutoff}")
            connection.execute(
                f"""
                CREATE TABLE prior_type_stats AS
                WITH stats AS (
                    SELECT product_type_no,
                           count(*)::DOUBLE/{int(prior84)}.0 AS share84,
                           count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY)
                               AS events28,
                           count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY)::DOUBLE
                               /{int(prior28)}.0 AS share28
                    FROM prior_events GROUP BY product_type_no
                )
                SELECT *,share28/nullif(share84,0) AS lift
                FROM stats;
                CREATE TABLE qualifying_types AS
                SELECT * FROM prior_type_stats
                WHERE events28>={SOURCE_CONFIG['minimum_prior_28d_events']}
                  AND lift>={SOURCE_CONFIG['seasonal_lift_threshold']};
                CREATE TABLE current_type_items AS
                WITH stats AS (
                    SELECT e.product_type_no,e.article_id,count(*) AS item_events,
                           count(*)::DOUBLE/sum(count(*)) OVER(PARTITION BY e.product_type_no)
                               AS item_share_in_type
                    FROM current_events e JOIN qualifying_types q USING(product_type_no)
                    GROUP BY e.product_type_no,e.article_id
                )
                SELECT *,row_number() OVER(
                    PARTITION BY product_type_no ORDER BY item_events DESC,article_id) AS type_item_rank
                FROM stats
                QUALIFY type_item_rank<={SOURCE_CONFIG['current_items_per_type']};
                CREATE TABLE prior_item_stats AS
                SELECT e.article_id,e.product_type_no,count(*) AS item_events
                FROM prior_events e JOIN qualifying_types q USING(product_type_no)
                WHERE e.t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY
                GROUP BY e.article_id,e.product_type_no;
                CREATE TABLE user_type_history AS
                WITH counts AS (
                    SELECT t.customer_id,a.product_type_no,count(*) AS type_events
                    FROM read_parquet({_literal(transactions_path)}) t
                    JOIN eval_users u USING(customer_id)
                    JOIN article_dim a USING(article_id)
                    JOIN qualifying_types q USING(product_type_no)
                    WHERE t.t_dat>={cutoff_sql}-INTERVAL {SOURCE_CONFIG['history_weeks']} WEEK
                      AND t.t_dat<{cutoff_sql}
                    GROUP BY t.customer_id,a.product_type_no
                ), totals AS (
                    SELECT customer_id,sum(type_events) AS qualifying_events FROM counts GROUP BY customer_id
                ), ranked AS (
                    SELECT c.*,c.type_events::DOUBLE/t.qualifying_events AS affinity,q.lift,
                           row_number() OVER(PARTITION BY c.customer_id
                               ORDER BY c.type_events::DOUBLE/t.qualifying_events*ln(q.lift) DESC,
                                        c.type_events DESC,c.product_type_no) AS type_rank
                    FROM counts c JOIN totals t USING(customer_id)
                    JOIN qualifying_types q USING(product_type_no)
                )
                SELECT * FROM ranked WHERE type_rank<={SOURCE_CONFIG['user_top_types']};
                """
            )
            connection.execute(
                f"""
                CREATE TABLE personalized_lane AS
                WITH scored AS (
                    SELECT u.customer_id,i.article_id,
                           u.affinity*ln(u.lift)*ln(1+i.item_events) AS lane_score
                    FROM user_type_history u JOIN current_type_items i USING(product_type_no)
                )
                SELECT *,row_number() OVER(PARTITION BY customer_id
                    ORDER BY lane_score DESC,article_id) AS personalized_rank
                FROM scored QUALIFY personalized_rank<={SOURCE_CONFIG['source_k']};
                CREATE TABLE current_global_items AS
                WITH scored AS (
                    SELECT i.article_id,ln(q.lift)*ln(1+i.item_events) AS lane_score
                    FROM current_type_items i JOIN qualifying_types q USING(product_type_no)
                )
                SELECT *,row_number() OVER(ORDER BY lane_score DESC,article_id) AS current_global_rank
                FROM scored QUALIFY current_global_rank<={SOURCE_CONFIG['current_global_k']};
                CREATE TABLE prior_global_items AS
                WITH scored AS (
                    SELECT i.article_id,ln(q.lift)*ln(1+i.item_events) AS lane_score
                    FROM prior_item_stats i JOIN qualifying_types q USING(product_type_no)
                )
                SELECT *,row_number() OVER(ORDER BY lane_score DESC,article_id) AS prior_direct_rank
                FROM scored QUALIFY prior_direct_rank<={SOURCE_CONFIG['prior_direct_k']};
                CREATE TABLE lane_union AS
                SELECT customer_id,article_id,personalized_rank,
                       NULL::BIGINT AS current_global_rank,NULL::BIGINT AS prior_direct_rank
                FROM personalized_lane
                UNION ALL
                SELECT u.customer_id,g.article_id,NULL,g.current_global_rank,NULL
                FROM eval_users u CROSS JOIN current_global_items g
                UNION ALL
                SELECT u.customer_id,g.article_id,NULL,NULL,g.prior_direct_rank
                FROM eval_users u CROSS JOIN prior_global_items g;
                CREATE TABLE seasonal_candidates AS
                WITH merged AS (
                    SELECT customer_id,article_id,
                           min(personalized_rank) AS personalized_rank,
                           min(current_global_rank) AS current_global_rank,
                           min(prior_direct_rank) AS prior_direct_rank
                    FROM lane_union GROUP BY customer_id,article_id
                ), scored AS (
                    SELECT *,
                           coalesce(1.0/({SOURCE_CONFIG['rrf_constant']}+personalized_rank),0.0)
                         + coalesce(1.0/({SOURCE_CONFIG['rrf_constant']}+current_global_rank),0.0)
                         + coalesce(1.0/({SOURCE_CONFIG['rrf_constant']}+prior_direct_rank),0.0)
                               AS seasonal_rrf_score
                    FROM merged
                )
                SELECT customer_id,article_id,
                       row_number() OVER(PARTITION BY customer_id ORDER BY
                           seasonal_rrf_score DESC,
                           coalesce(personalized_rank,1000000),
                           coalesce(current_global_rank,1000000),
                           coalesce(prior_direct_rank,1000000),article_id) AS seasonal_rank,
                       seasonal_rrf_score,personalized_rank,current_global_rank,prior_direct_rank,
                       (personalized_rank IS NOT NULL)::UTINYINT AS personalized_present,
                       (current_global_rank IS NOT NULL)::UTINYINT AS current_global_present,
                       (prior_direct_rank IS NOT NULL)::UTINYINT AS prior_direct_present
                FROM scored QUALIFY seasonal_rank<={SOURCE_CONFIG['source_k']};
                """
            )
            source_audit = connection.execute(
                """
                WITH groups AS (
                    SELECT customer_id,count(*) AS n,count(DISTINCT article_id) AS u,
                           min(seasonal_rank) AS lo,max(seasonal_rank) AS hi,
                           count(DISTINCT seasonal_rank) AS ranks
                    FROM seasonal_candidates GROUP BY customer_id
                )
                SELECT (SELECT count(*) FROM seasonal_candidates),
                       (SELECT count(DISTINCT (customer_id,article_id)) FROM seasonal_candidates),
                       count(*),min(n),max(n),
                       count(*) FILTER(WHERE n<>u OR lo<>1 OR hi<>n OR ranks<>n OR n>100),
                       (SELECT count(*) FROM eval_users),
                       (SELECT count(*) FROM qualifying_types),
                       (SELECT count(*) FROM current_global_items),
                       (SELECT count(*) FROM prior_global_items)
                FROM groups
                """
            ).fetchone()
            if int(source_audit[0]) != int(source_audit[1]) or int(source_audit[5]):
                raise RuntimeError(f"M3.4 source group audit failed: {cutoff} {source_audit}")
            if int(source_audit[2]) != int(source_audit[6]):
                raise RuntimeError(f"M3.4 source does not cover all eval users: {cutoff}")
            connection.execute(
                f"COPY (SELECT * FROM seasonal_candidates ORDER BY customer_id,seasonal_rank) "
                f"TO {_literal(source_path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
            )
            connection.execute(
                f"COPY (SELECT * FROM qualifying_types ORDER BY product_type_no) "
                f"TO {_literal(type_path)} (FORMAT PARQUET,COMPRESSION ZSTD)"
            )
            latest = connection.execute(
                """
                SELECT (SELECT max(t_dat) FROM current_events),
                       (SELECT max(t_dat) FROM prior_events),
                       (SELECT max(t.t_dat) FROM read_parquet(?) t JOIN eval_users u USING(customer_id)
                         WHERE t.t_dat<?)
                """,
                [str(transactions_path.resolve()), cutoff],
            ).fetchone()
            cutoff_timestamp = pd.Timestamp(cutoff)
            if pd.Timestamp(latest[0]) >= cutoff_timestamp:
                raise RuntimeError(f"M3.4 current-history leakage: {cutoff}")
            if pd.Timestamp(latest[1]) >= cutoff_timestamp - pd.DateOffset(years=1):
                raise RuntimeError(f"M3.4 prior-history leakage: {cutoff}")
            manifest = {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "status": "completed",
                "window": window,
                "cutoff": cutoff,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract": SOURCE_CONFIG,
                "inputs": {
                    "base": base,
                    "transactions": transaction_identity,
                    "articles": article_identity,
                },
                "audit": {
                    "rows": int(source_audit[0]),
                    "unique_rows": int(source_audit[1]),
                    "users": int(source_audit[2]),
                    "min_group_rows": int(source_audit[3]),
                    "max_group_rows": int(source_audit[4]),
                    "invalid_groups": int(source_audit[5]),
                    "eval_users": int(source_audit[6]),
                    "qualifying_product_types": int(source_audit[7]),
                    "current_global_items": int(source_audit[8]),
                    "prior_direct_items": int(source_audit[9]),
                    "current_rows_28d": current28,
                    "prior_rows_28d": int(prior28),
                    "prior_rows_84d": int(prior84),
                    "latest_current_history_date": str(latest[0]),
                    "latest_prior_history_date": str(latest[1]),
                    "latest_user_history_date": str(latest[2]),
                },
                "artifacts": {
                    "candidates": _file_identity(source_path),
                    "qualifying_types": _file_identity(type_path),
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(manifest_path, manifest)
            results[cutoff] = manifest
        finally:
            connection.close()
    return results


def _legacy_prepare_frame(
    frame: pd.DataFrame,
    features: list[str],
    category_maps: dict[str, dict[int, int]],
) -> pd.DataFrame:
    prepared = pd.DataFrame(index=frame.index)
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        if feature in CATEGORICAL_FEATURES:
            prepared[feature] = values.map(category_maps[feature]).fillna(0).astype(np.int32)
        else:
            prepared[feature] = values.astype(np.float32)
    return prepared


def _candidate_label_map(frame: pd.DataFrame, scores: np.ndarray, metric_k: int = 12) -> float:
    work = frame[["customer_id", "article_id", "target"]].copy()
    work["score"] = scores
    work = work.sort_values(
        ["customer_id", "score", "article_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    work["rank"] = work.groupby("customer_id", sort=False).cumcount() + 1
    work["hit_rank"] = work.groupby("customer_id", sort=False)["target"].cumsum()
    top = work[work["rank"] <= metric_k]
    precision = np.where(top["target"].to_numpy() == 1, top["hit_rank"] / top["rank"], 0.0)
    top = top.assign(precision=precision)
    numerators = top.groupby("customer_id", sort=False)["precision"].sum()
    denominators = work.groupby("customer_id", sort=False)["target"].sum().clip(upper=metric_k)
    valid = denominators > 0
    if not bool(valid.any()):
        return 0.0
    return float((numerators.reindex(denominators.index, fill_value=0)[valid] / denominators[valid]).mean())


def benchmark_prepare_frame(
    *,
    feature_path: Path,
    category_maps_path: Path,
    model_path: Path,
    sample_users: int = 500,
) -> dict[str, Any]:
    import lightgbm as lgb

    features = variant_feature_sets()["adaptive_seasonal"]
    maps_raw = _read_json(category_maps_path)
    category_maps = {
        name: {int(key): int(value) for key, value in mapping.items()}
        for name, mapping in maps_raw.items()
    }
    connection = duckdb.connect()
    try:
        selected = ",".join(["customer_id", "article_id", "candidate_rank", "target", *features])
        frame = connection.execute(
            f"""
            WITH users AS (
                SELECT DISTINCT customer_id FROM read_parquet({_literal(feature_path)})
                ORDER BY customer_id LIMIT {int(sample_users)}
            )
            SELECT {selected} FROM read_parquet({_literal(feature_path)}) f
            SEMI JOIN users USING(customer_id)
            ORDER BY customer_id,candidate_rank,article_id
            """
        ).fetchdf()
    finally:
        connection.close()
    timings: dict[str, list[float]] = {"legacy": [], "batched": []}
    prediction_timings: dict[str, list[float]] = {"legacy": [], "batched": []}
    warning_counts: dict[str, int] = {"legacy": 0, "batched": 0}
    block_counts: dict[str, int] = {}
    legacy = None
    batched = None
    for name, function in (("legacy", _legacy_prepare_frame), ("batched", _prepare_frame)):
        for _ in range(3):
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", pd.errors.PerformanceWarning)
                prepared = function(frame, features, category_maps)
            timings[name].append(time.perf_counter() - started)
            warning_counts[name] += sum(
                issubclass(item.category, pd.errors.PerformanceWarning) for item in caught
            )
        if name == "legacy":
            legacy = prepared
        else:
            batched = prepared
    assert legacy is not None and batched is not None
    value_equal = bool(
        legacy.shape == batched.shape
        and list(legacy.columns) == list(batched.columns)
        and list(legacy.index) == list(batched.index)
        and all(legacy[column].dtype == batched[column].dtype for column in features)
        and np.array_equal(legacy.to_numpy(), batched.to_numpy(), equal_nan=True)
    )
    block_counts = {
        "legacy": len(legacy._mgr.blocks),
        "batched": len(batched._mgr.blocks),
    }
    model = lgb.Booster(model_file=str(model_path))
    scores_by_name: dict[str, np.ndarray] = {}
    for name, prepared in (("legacy", legacy), ("batched", batched)):
        for _ in range(3):
            started = time.perf_counter()
            scores = model.predict(prepared)
            prediction_timings[name].append(time.perf_counter() - started)
        scores_by_name[name] = scores
    legacy_scores = scores_by_name["legacy"]
    batched_scores = scores_by_name["batched"]
    max_prediction_difference = float(np.max(np.abs(legacy_scores - batched_scores)))
    legacy_map = _candidate_label_map(frame, legacy_scores)
    batched_map = _candidate_label_map(frame, batched_scores)
    legacy_median = float(np.median(timings["legacy"]))
    batched_median = float(np.median(timings["batched"]))
    legacy_prediction_median = float(np.median(prediction_timings["legacy"]))
    batched_prediction_median = float(np.median(prediction_timings["batched"]))
    legacy_end_to_end = legacy_median + legacy_prediction_median
    batched_end_to_end = batched_median + batched_prediction_median
    passed = bool(
        value_equal
        and max_prediction_difference == 0.0
        and abs(legacy_map - batched_map) <= 1e-15
    )
    optimization_promoted = bool(
        passed
        and warning_counts["batched"] < warning_counts["legacy"]
        and legacy_end_to_end / batched_end_to_end > 1.0
    )
    return {
        "status": "passed" if passed else "failed",
        "optimization_promoted": optimization_promoted,
        "sample_users": sample_users,
        "sample_rows": len(frame),
        "features": len(features),
        "value_dtype_order_index_equal": value_equal,
        "max_prediction_absolute_difference": max_prediction_difference,
        "legacy_candidate_label_map@12": legacy_map,
        "batched_candidate_label_map@12": batched_map,
        "map_absolute_difference": abs(legacy_map - batched_map),
        "legacy_fragmentation_warnings": warning_counts["legacy"],
        "batched_fragmentation_warnings": warning_counts["batched"],
        "block_counts": block_counts,
        "legacy_seconds": timings["legacy"],
        "batched_seconds": timings["batched"],
        "legacy_median_seconds": legacy_median,
        "batched_median_seconds": batched_median,
        "construction_speedup": legacy_median / batched_median if batched_median else None,
        "legacy_prediction_seconds": prediction_timings["legacy"],
        "batched_prediction_seconds": prediction_timings["batched"],
        "legacy_prediction_median_seconds": legacy_prediction_median,
        "batched_prediction_median_seconds": batched_prediction_median,
        "prediction_speedup": (
            legacy_prediction_median / batched_prediction_median
            if batched_prediction_median
            else None
        ),
        "legacy_end_to_end_median_seconds": legacy_end_to_end,
        "batched_end_to_end_median_seconds": batched_end_to_end,
        "end_to_end_speedup": legacy_end_to_end / batched_end_to_end,
        "inputs": {
            "feature_cache": _file_identity(feature_path),
            "category_maps": _file_identity(category_maps_path),
            "model": _file_identity(model_path),
        },
    }


def _group_audit(connection: duckdb.DuckDBPyConnection, table: str, maximum: int) -> dict[str, int]:
    row = connection.execute(
        f"""
        WITH groups AS (
            SELECT customer_id,count(*) AS n,count(DISTINCT article_id) AS u,
                   min(candidate_rank) AS lo,max(candidate_rank) AS hi,
                   count(DISTINCT candidate_rank) AS ranks
            FROM {table} GROUP BY customer_id
        )
        SELECT (SELECT count(*) FROM {table}),
               (SELECT count(DISTINCT (customer_id,article_id)) FROM {table}),
               count(*),min(n),max(n),
               count(*) FILTER(WHERE n<>u OR lo<>1 OR hi<>n OR ranks<>n OR n>{maximum})
        FROM groups
        """
    ).fetchone()
    values = [int(value) for value in row]
    if values[0] != values[1] or values[5]:
        raise RuntimeError(f"M3.4 candidate group audit failed: {table} {values}")
    return {
        "rows": values[0],
        "unique_rows": values[1],
        "users": values[2],
        "min_group_rows": values[3],
        "max_group_rows": values[4],
        "invalid_groups": values[5],
    }


def _evaluate_variant(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    budget: int,
) -> dict[str, Any]:
    return {
        "budget": budget,
        "overall": _candidate_metrics(connection, table, budget, truth_table="eval_truth"),
        "warm": _candidate_metrics(
            connection, table, budget, "item_temperature='warm'", truth_table="eval_truth"
        ),
        "cold": _candidate_metrics(
            connection, table, budget, "item_temperature='cold'", truth_table="eval_truth"
        ),
        "season_sensitive": _candidate_metrics(
            connection, table, budget, truth_table="season_truth"
        ),
    }


def evaluate_window(
    *,
    window: str,
    cutoff: str,
    transactions_path: Path,
    articles_path: Path,
    base_candidate_path: Path,
    source_manifest: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True)
    database_path = artifact_dir / "evaluation.duckdb"
    connection: duckdb.DuckDBPyConnection | None = duckdb.connect(str(database_path))
    cutoff_sql = _date(cutoff)
    source_path = Path(source_manifest["artifacts"]["candidates"]["path"])
    type_path = Path(source_manifest["artifacts"]["qualifying_types"]["path"])
    try:
        connection.execute("SET threads=8")
        connection.execute("SET memory_limit='11GB'")
        connection.execute("PRAGMA disable_progress_bar")
        connection.execute(
            f"""
            CREATE TABLE base AS SELECT * FROM read_parquet({_literal(base_candidate_path)});
            CREATE TABLE source AS SELECT * FROM read_parquet({_literal(source_path)});
            CREATE TABLE eval_users AS SELECT DISTINCT customer_id FROM base;
            CREATE TABLE article_dim AS
            SELECT article_id,try_cast(product_type_no AS INTEGER) AS product_type_no
            FROM read_csv_auto({_literal(articles_path)},header=true,all_varchar=true);
            CREATE TABLE warm_catalog AS
            SELECT DISTINCT article_id FROM read_parquet({_literal(transactions_path)})
            WHERE t_dat<{cutoff_sql};
            CREATE TABLE item_popularity AS
            WITH counts AS (
                SELECT article_id,count(*) AS events
                FROM read_parquet({_literal(transactions_path)})
                WHERE t_dat>={cutoff_sql}-INTERVAL 12 WEEK AND t_dat<{cutoff_sql}
                GROUP BY article_id
            ), bucketed AS (
                SELECT *,ntile(5) OVER(ORDER BY events DESC,article_id) AS bucket FROM counts
            )
            SELECT article_id,CASE WHEN bucket=1 THEN 'head_top20pct_items'
                WHEN bucket<=3 THEN 'middle_next40pct_items'
                ELSE 'tail_bottom40pct_items' END AS popularity_segment FROM bucketed;
            CREATE TABLE eval_truth AS
            SELECT q.customer_id,q.article_id,
                   CASE WHEN w.article_id IS NULL THEN 'cold' ELSE 'warm' END AS item_temperature,
                   coalesce(p.popularity_segment,'cold_or_unseen') AS popularity_segment
            FROM (
                SELECT DISTINCT t.customer_id,t.article_id
                FROM read_parquet({_literal(transactions_path)}) t
                JOIN eval_users u USING(customer_id)
                WHERE t.t_dat>={cutoff_sql} AND t.t_dat<{cutoff_sql}+INTERVAL 7 DAY
            ) q
            LEFT JOIN warm_catalog w USING(article_id)
            LEFT JOIN item_popularity p USING(article_id);
            CREATE TABLE season_truth AS
            SELECT t.* FROM eval_truth t JOIN article_dim a USING(article_id)
            JOIN read_parquet({_literal(type_path)}) q USING(product_type_no);
            CREATE TABLE base_300 AS
            SELECT customer_id,article_id,candidate_rank FROM base;
            CREATE TABLE seasonal_standalone_100 AS
            SELECT customer_id,article_id,seasonal_rank AS candidate_rank FROM source;
            CREATE TABLE source_new AS
            SELECT s.*,row_number() OVER(PARTITION BY s.customer_id ORDER BY s.seasonal_rank,s.article_id)
                AS new_rank
            FROM source s WHERE NOT EXISTS(
                SELECT 1 FROM base b WHERE b.customer_id=s.customer_id AND b.article_id=s.article_id);
            CREATE TABLE base_sizes AS SELECT customer_id,count(*) AS n FROM base GROUP BY customer_id;
            CREATE TABLE vacancy_fill_300 AS
            SELECT customer_id,article_id,candidate_rank FROM base
            UNION ALL
            SELECT n.customer_id,n.article_id,(s.n+n.new_rank)::BIGINT AS candidate_rank
            FROM source_new n JOIN base_sizes s USING(customer_id)
            WHERE n.new_rank<=300-s.n;
            CREATE TABLE expanded_350 AS
            SELECT customer_id,article_id,candidate_rank FROM base
            UNION ALL
            SELECT n.customer_id,n.article_id,(s.n+n.new_rank)::BIGINT AS candidate_rank
            FROM source_new n JOIN base_sizes s USING(customer_id)
            WHERE n.new_rank<=50;
            CREATE TABLE competitive_pool AS
            WITH unioned AS (
                SELECT customer_id,article_id,min(item2vec_rank) AS item2vec_rank,
                       NULL::BIGINT AS seasonal_rank
                FROM base WHERE candidate_rank>100 GROUP BY customer_id,article_id
                UNION ALL
                SELECT s.customer_id,s.article_id,NULL,s.seasonal_rank
                FROM source s WHERE NOT EXISTS(
                    SELECT 1 FROM base b WHERE b.customer_id=s.customer_id
                      AND b.article_id=s.article_id AND b.candidate_rank<=100)
            )
            SELECT customer_id,article_id,min(item2vec_rank) AS item2vec_rank,
                   min(seasonal_rank) AS seasonal_rank
            FROM unioned GROUP BY customer_id,article_id;
            CREATE TABLE competitive_appendix AS
            WITH scored AS (
                SELECT *,coalesce(1.0/(60+item2vec_rank),0.0)
                         +coalesce(1.0/(60+seasonal_rank),0.0) AS score
                FROM competitive_pool
            )
            SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY score DESC,
                coalesce(item2vec_rank,1000000),coalesce(seasonal_rank,1000000),article_id) AS new_rank
            FROM scored QUALIFY new_rank<=200;
            CREATE TABLE competitive_300 AS
            SELECT customer_id,article_id,candidate_rank FROM base WHERE candidate_rank<=100
            UNION ALL
            SELECT customer_id,article_id,(100+new_rank)::BIGINT AS candidate_rank
            FROM competitive_appendix;
            """
        )
        audits = {
            name: _group_audit(connection, name, budget)
            for name, budget in VARIANT_BUDGETS.items()
        }
        evaluations = {
            name: _evaluate_variant(connection, name, budget)
            for name, budget in VARIANT_BUDGETS.items()
        }
        overlap = connection.execute(
            """
            SELECT (SELECT count(*) FROM source),
                   (SELECT count(*) FROM source s WHERE EXISTS(
                       SELECT 1 FROM base b
                       WHERE b.customer_id=s.customer_id AND b.article_id=s.article_id)),
                   (SELECT count(*) FROM source_new),
                   (SELECT count(*) FROM source s JOIN eval_truth t USING(customer_id,article_id)),
                   (SELECT count(*) FROM source_new s JOIN eval_truth t USING(customer_id,article_id)),
                   (SELECT count(*) FROM source_new s JOIN season_truth t USING(customer_id,article_id))
            """
        ).fetchone()
        lane_rows = connection.execute(
            """
            SELECT
              count(*) FILTER(WHERE personalized_present=1),
              count(*) FILTER(WHERE current_global_present=1),
              count(*) FILTER(WHERE prior_direct_present=1),
              count(*) FILTER(WHERE personalized_present=1 AND t.article_id IS NOT NULL),
              count(*) FILTER(WHERE current_global_present=1 AND t.article_id IS NOT NULL),
              count(*) FILTER(WHERE prior_direct_present=1 AND t.article_id IS NOT NULL)
            FROM source s LEFT JOIN eval_truth t USING(customer_id,article_id)
            """
        ).fetchone()
        truth_counts = connection.execute(
            """
            SELECT (SELECT count(DISTINCT customer_id) FROM eval_truth),(SELECT count(*) FROM eval_truth),
                   (SELECT count(DISTINCT customer_id) FROM season_truth),(SELECT count(*) FROM season_truth)
            """
        ).fetchone()
        prediction_paths: dict[str, Any] = {}
        for name in VARIANT_BUDGETS:
            path = artifact_dir / f"{name}.parquet"
            connection.execute(
                f"COPY (SELECT * FROM {name} ORDER BY customer_id,candidate_rank,article_id) "
                f"TO {_literal(path)} (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
            )
            prediction_paths[name] = _file_identity(path)
        connection.execute("CHECKPOINT")
        connection.close()
        connection = None
        return {
            "window": window,
            "cutoff": cutoff,
            "truth": {
                "overall_users": int(truth_counts[0]),
                "overall_pairs": int(truth_counts[1]),
                "season_sensitive_users": int(truth_counts[2]),
                "season_sensitive_pairs": int(truth_counts[3]),
            },
            "candidate_audits": audits,
            "evaluations": evaluations,
            "source_overlap": {
                "source_rows": int(overlap[0]),
                "rows_already_in_base": int(overlap[1]),
                "source_only_rows": int(overlap[2]),
                "source_truth_hits": int(overlap[3]),
                "marginal_truth_pairs": int(overlap[4]),
                "marginal_season_sensitive_truth_pairs": int(overlap[5]),
            },
            "lane_attribution": {
                "personalized_current_rows": int(lane_rows[0]),
                "current_global_rows": int(lane_rows[1]),
                "prior_direct_rows": int(lane_rows[2]),
                "personalized_current_truth_hits": int(lane_rows[3]),
                "current_global_truth_hits": int(lane_rows[4]),
                "prior_direct_truth_hits": int(lane_rows[5]),
            },
            "artifacts": {
                "evaluation_db": _file_identity(database_path),
                "candidate_variants": prediction_paths,
            },
        }
    finally:
        if connection is not None:
            connection.close()


def summarize_development(development: dict[str, Any]) -> dict[str, Any]:
    base_name = "base_300"
    variants: dict[str, Any] = {}
    for name in VARIANT_BUDGETS:
        overall = {
            window: float(row["evaluations"][name]["overall"][
                f"candidate_recall@{VARIANT_BUDGETS[name]}"
            ])
            for window, row in development.items()
        }
        seasonal = {
            window: float(row["evaluations"][name]["season_sensitive"][
                f"candidate_recall@{VARIANT_BUDGETS[name]}"
            ])
            for window, row in development.items()
        }
        oracle = {
            window: float(row["evaluations"][name]["overall"]["oracle_map@12"])
            for window, row in development.items()
        }
        variants[name] = {
            "overall_recall": overall,
            "mean_overall_recall": float(np.mean(list(overall.values()))),
            "season_sensitive_recall": seasonal,
            "mean_season_sensitive_recall": float(np.mean(list(seasonal.values()))),
            "overall_oracle_map@12": oracle,
            "mean_overall_oracle_map@12": float(np.mean(list(oracle.values()))),
        }
    base = variants[base_name]
    for name, row in variants.items():
        row["overall_recall_delta_vs_base"] = {
            window: value - base["overall_recall"][window]
            for window, value in row["overall_recall"].items()
        }
        row["season_sensitive_recall_delta_vs_base"] = {
            window: value - base["season_sensitive_recall"][window]
            for window, value in row["season_sensitive_recall"].items()
        }
        row["mean_overall_recall_delta_vs_base"] = float(
            np.mean(list(row["overall_recall_delta_vs_base"].values()))
        )
        row["mean_season_sensitive_recall_delta_vs_base"] = float(
            np.mean(list(row["season_sensitive_recall_delta_vs_base"].values()))
        )
    expanded = variants["expanded_350"]
    marginal = {
        window: int(row["source_overlap"]["marginal_season_sensitive_truth_pairs"])
        for window, row in development.items()
    }
    gate_checks = {
        "gap_windows_season_sensitive_recall_strictly_improve": all(
            expanded["season_sensitive_recall_delta_vs_base"][window] > 0
            for window in GAP_WINDOWS
        ),
        "mean_season_sensitive_recall_delta_at_least_0_002": (
            expanded["mean_season_sensitive_recall_delta_vs_base"] >= 0.002
        ),
        "gap_windows_at_least_20_marginal_season_truth_pairs": all(
            marginal[window] >= 20 for window in GAP_WINDOWS
        ),
        "mean_overall_recall_delta_at_least_0_001": (
            expanded["mean_overall_recall_delta_vs_base"] >= 0.001
        ),
    }
    return {
        "variants": variants,
        "marginal_season_sensitive_truth_pairs": marginal,
        "retrieval_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
            "next_if_passed": "build source-aware features and seasonal expert in a later stage",
            "next_if_failed": "preserve result; test season-conditioned co-visitation or content retrieval",
        },
        "final_week": "not_run",
    }


def render_report(result: dict[str, Any]) -> str:
    windows = list(OUTER_WINDOWS)
    summary = result["summary"]
    lines = [
        "# M3.4：season-aware retrieval 与固定预算压缩",
        "",
        "## 结论",
        "",
        f"- retrieval gate：{summary['retrieval_gate']['passed']}。",
        f"- `_prepare_frame` 等价门禁：{result['prepare_frame_benchmark']['status']}；"
        f"优化是否晋级：{result['prepare_frame_benchmark'].get('optimization_promoted', False)}。",
        "- final week 未运行。",
        "",
        "## Candidate Recall",
        "",
        "| variant | " + " | ".join(windows) + " | mean | mean delta vs base |",
        "|---|" + "---:|" * (len(windows) + 2),
    ]
    for name, row in summary["variants"].items():
        lines.append(
            "| " + name + " | "
            + " | ".join(f"{row['overall_recall'][window]:.6f}" for window in windows)
            + f" | {row['mean_overall_recall']:.6f} | "
            + f"{row['mean_overall_recall_delta_vs_base']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Season-sensitive Recall",
            "",
            "| variant | " + " | ".join(windows) + " | mean | mean delta vs base |",
            "|---|" + "---:|" * (len(windows) + 2),
        ]
    )
    for name, row in summary["variants"].items():
        lines.append(
            "| " + name + " | "
            + " | ".join(
                f"{row['season_sensitive_recall'][window]:.6f}" for window in windows
            )
            + f" | {row['mean_season_sensitive_recall']:.6f} | "
            + f"{row['mean_season_sensitive_recall_delta_vs_base']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            *[
                f"- {name}: {value}"
                for name, value in summary["retrieval_gate"]["checks"].items()
            ],
            "",
            f"- wall time：{result['elapsed_seconds']:.2f} 秒。",
            f"- peak working set：{result['resources']['peak_working_set_bytes']/1024**3:.2f} GiB。",
            "- 机器可读证据：`metrics.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def _peak_working_set_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    except (ImportError, AttributeError):
        return 0


def run_m34(
    *,
    raw_dir: Path,
    transactions_path: Path,
    base_candidates: dict[str, tuple[Path, Path]],
    source_cache_dir: Path,
    benchmark_feature_path: Path,
    benchmark_category_maps_path: Path,
    benchmark_model_path: Path,
    output_dir: Path,
    artifact_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    validate_protocol()
    if set(base_candidates) != set(OUTER_WINDOWS.values()):
        raise ValueError("M3.4 requires exactly the four outer base cutoffs")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        benchmark = benchmark_prepare_frame(
            feature_path=benchmark_feature_path,
            category_maps_path=benchmark_category_maps_path,
            model_path=benchmark_model_path,
        )
        if benchmark["status"] != "passed":
            raise RuntimeError("M3.4 prepare-frame equivalence gate failed")
        source_cache = build_seasonal_source_cache(
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            base_candidates=base_candidates,
            cache_dir=source_cache_dir,
        )
        development: dict[str, Any] = {}
        for window, cutoff in OUTER_WINDOWS.items():
            print(f"M3.4 evaluating {window} ({cutoff})", flush=True)
            candidate_path, _ = base_candidates[cutoff]
            development[window] = evaluate_window(
                window=window,
                cutoff=cutoff,
                transactions_path=transactions_path,
                articles_path=raw_dir / "articles.csv",
                base_candidate_path=candidate_path,
                source_manifest=source_cache[cutoff],
                artifact_dir=artifact_dir / window,
            )
            gc.collect()
        summary = summarize_development(development)
        elapsed = time.perf_counter() - started
        result = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M3.4",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "four cross-season outer development windows; final week not run",
            "contract": {
                "source": SOURCE_CONFIG,
                "variants": VARIANT_BUDGETS,
                "catalog_protocol": "optimistic_all_articles",
                "base_pool": "frozen six-source Top100 plus up to 200 Item2Vec-only",
                "selection": "pre-registered expanded-350 season-sensitive retrieval gate",
                "inventory_boundary": "current 28d transactions are a proxy, not inventory/exposure",
            },
            "inputs": {
                "transactions": _file_identity(transactions_path),
                "articles": _file_identity(raw_dir / "articles.csv"),
                "base_candidates": {
                    cutoff: {
                        "candidate": _file_identity(paths[0]),
                        "manifest": _file_identity(paths[1]),
                    }
                    for cutoff, paths in base_candidates.items()
                },
            },
            "prepare_frame_benchmark": benchmark,
            "source_cache": source_cache,
            "development": development,
            "summary": summary,
            "resources": {
                "peak_working_set_bytes": _peak_working_set_bytes(),
            },
            "elapsed_seconds": elapsed,
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M3_4_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path.resolve()),
            "report": str(report_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
            "source_cache_dir": str(source_cache_dir.resolve()),
        }
        _write_json(metrics_path, result)
        report_path.write_text(render_report(result), encoding="utf-8")
        return result
    except Exception as error:
        _write_json(
            artifact_dir / f"failure-{time.time_ns()}.json",
            {
                "schema_version": "m3.4-failure-v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
