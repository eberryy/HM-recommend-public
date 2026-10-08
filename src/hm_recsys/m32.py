from __future__ import annotations

import gc
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from .m2 import M2Config, RETRIEVAL_FEATURES, _literal, _sha256, _write_json, build_category_maps
from .m26 import _evaluate_models, _save_category_maps
from .m29 import _attach_popularity_segments, _read_json
from .m210 import _file_identity, _target_feature_paths, score_models
from .m211 import SAMPLING_SEED, load_distribution_sample, train_lambdarank
from .m212 import feature_sets, load_inner_validation, train_inner_ranker
from .m3 import ALL_CUTOFFS, ANCHOR_NAME, FALLBACK_NAME, ROLLING_PROTOCOL, _peak_working_set_bytes


SCHEMA_VERSION = "m3.2-temporal-repeat-features-v1"
CACHE_SCHEMA_VERSION = "m3.2-temporal-feature-cache-v1"

REPURCHASE_ROUTE_FEATURES = [
    "repurchase_present",
    "repurchase_rank",
    "repurchase_score",
    "repurchase_rrf_contribution",
]

REPEAT_DYNAMICS_FEATURES = [
    "user_item_purchase_days_12w",
    "user_item_purchase_days_28d",
    "user_item_median_gap_days_12w",
    "user_item_last_gap_days",
    "user_item_cadence_ratio",
    "user_product_code_purchase_days_12w",
    "exact_prior_purchase_flag",
    "exact_multi_day_repeat_flag",
    "product_code_prior_purchase_flag",
    "exact_repeat_x_item_trend",
    "exact_repeat_x_product_family",
    "exact_repeat_x_covisit",
    "exact_repeat_x_item2vec",
    "exact_repeat_x_log_user_activity",
]

TREND_ACCELERATION_FEATURES = [
    "item_events_3d",
    "item_events_previous_7d",
    "item_unique_customers_7d",
    "item_unique_customers_previous_7d",
    "item_events_week0",
    "item_events_week1",
    "item_events_week2",
    "item_events_week3",
    "item_velocity_7d_vs_previous_7d",
    "item_customer_velocity_7d_vs_previous_7d",
    "item_weekly_slope_4w",
    "item_age_days",
    "item_events_per_age_day_12w",
    "product_type_events_7d",
    "product_type_events_previous_7d",
    "product_type_velocity_7d_vs_previous_7d",
]

SEASONAL_PRIOR_FEATURES = [
    "prior_year_product_type_share_28d",
    "prior_year_garment_share_28d",
    "prior_year_department_share_28d",
    "prior_year_index_group_share_28d",
    "prior_year_colour_share_28d",
]

ALL_NEW_FEATURES = [
    *REPEAT_DYNAMICS_FEATURES,
    *TREND_ACCELERATION_FEATURES,
    *SEASONAL_PRIOR_FEATURES,
]


def variant_feature_sets() -> dict[str, list[str]]:
    anchor = list(feature_sets()[ANCHOR_NAME])
    without_repurchase = [
        feature for feature in anchor if feature not in REPURCHASE_ROUTE_FEATURES
    ]
    variants = {
        "anchor": anchor,
        "without_repurchase_route_features": without_repurchase,
        "repeat_dynamics": [*anchor, *REPEAT_DYNAMICS_FEATURES],
        "trend_acceleration": [*anchor, *TREND_ACCELERATION_FEATURES],
        "seasonal_prior": [*anchor, *SEASONAL_PRIOR_FEATURES],
        "all_temporal": [*anchor, *ALL_NEW_FEATURES],
        "repeat_substitute_without_repurchase_route_features": [
            *without_repurchase,
            *REPEAT_DYNAMICS_FEATURES,
        ],
    }
    validate_variant_feature_sets(variants)
    return variants


def validate_variant_feature_sets(variants: dict[str, list[str]]) -> None:
    if list(variants) != [
        "anchor",
        "without_repurchase_route_features",
        "repeat_dynamics",
        "trend_acceleration",
        "seasonal_prior",
        "all_temporal",
        "repeat_substitute_without_repurchase_route_features",
    ]:
        raise RuntimeError("M3.2 requires exactly seven pre-registered variants")
    anchor = set(feature_sets()[ANCHOR_NAME])
    if len(ALL_NEW_FEATURES) != len(set(ALL_NEW_FEATURES)):
        raise RuntimeError("M3.2 new feature families overlap")
    if anchor.intersection(ALL_NEW_FEATURES):
        raise RuntimeError("M3.2 new features overlap the frozen anchor")
    if set(variants["anchor"]) != anchor:
        raise RuntimeError("M3.2 anchor drift")
    expected_without = anchor - set(REPURCHASE_ROUTE_FEATURES)
    if set(variants["without_repurchase_route_features"]) != expected_without:
        raise RuntimeError("M3.2 repurchase route removal drift")
    if set(variants["all_temporal"]) != anchor | set(ALL_NEW_FEATURES):
        raise RuntimeError("M3.2 all-temporal variant is incomplete")
    if any("month" in feature or "week_of_year" in feature for feature in ALL_NEW_FEATURES):
        raise RuntimeError("M3.2 must not add group-constant plain calendar features")


def _date(value: str) -> str:
    return "DATE '" + value.replace("'", "''") + "'"


def _cache_paths(cache_dir: Path, cutoff: str) -> tuple[Path, Path]:
    root = cache_dir / cutoff
    return root / "features.parquet", root / "manifest.json"


def _validate_cache(
    feature_path: Path,
    manifest_path: Path,
    *,
    cutoff: str,
    source_identity: dict[str, Any],
    transaction_identity: dict[str, Any],
    article_identity: dict[str, Any],
) -> dict[str, Any] | None:
    if not feature_path.exists() and not manifest_path.exists():
        return None
    if not feature_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"incomplete M3.2 cache: {feature_path.parent}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError(f"M3.2 cache schema drift: {manifest_path}")
    if manifest.get("cutoff") != cutoff:
        raise ValueError(f"M3.2 cache cutoff drift: {manifest_path}")
    for name, expected in (
        ("source", source_identity),
        ("transactions", transaction_identity),
        ("articles", article_identity),
    ):
        actual = manifest["inputs"][name]
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError(f"M3.2 cache input drift: {name} {manifest_path}")
    actual_output = _file_identity(feature_path)
    declared_output = manifest["artifact"]
    if (
        actual_output["bytes"] != int(declared_output["bytes"])
        or actual_output["sha256"] != declared_output["sha256"]
    ):
        raise ValueError(f"M3.2 cache artifact drift: {feature_path}")
    return manifest


def build_temporal_feature_cache(
    *,
    raw_dir: Path,
    transactions_path: Path,
    source_cache_dir: Path,
    cache_dir: Path,
    cutoffs: tuple[str, ...] = ALL_CUTOFFS,
) -> dict[str, Any]:
    transaction_identity = _file_identity(transactions_path)
    article_path = raw_dir / "articles.csv"
    article_identity = _file_identity(article_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for cutoff in cutoffs:
        started = time.perf_counter()
        source_path, _ = _target_feature_paths(source_cache_dir, cutoff)
        source_identity = _file_identity(source_path)
        feature_path, manifest_path = _cache_paths(cache_dir, cutoff)
        existing = _validate_cache(
            feature_path,
            manifest_path,
            cutoff=cutoff,
            source_identity=source_identity,
            transaction_identity=transaction_identity,
            article_identity=article_identity,
        )
        if existing is not None:
            results[cutoff] = existing
            continue
        if feature_path.parent.exists():
            raise FileExistsError(f"incomplete M3.2 cache root: {feature_path.parent}")
        feature_path.parent.mkdir(parents=True)
        database_path = feature_path.parent / "feature-build.duckdb"
        temp_dir = feature_path.parent / "duckdb-temp"
        temp_dir.mkdir()
        connection = duckdb.connect(str(database_path))
        cutoff_sql = _date(cutoff)
        try:
            connection.execute("SET threads=8")
            connection.execute("SET memory_limit='11GB'")
            connection.execute(f"SET temp_directory={_literal(temp_dir)}")
            connection.execute(
                f"CREATE VIEW base AS SELECT * FROM read_parquet({_literal(source_path)})"
            )
            connection.execute("CREATE TABLE users AS SELECT DISTINCT customer_id FROM base")
            connection.execute(
                f"""
                CREATE TABLE article_dim AS
                SELECT article_id,
                       try_cast(product_code AS INTEGER) AS product_code,
                       try_cast(product_type_no AS INTEGER) AS product_type_no,
                       try_cast(garment_group_no AS INTEGER) AS garment_group_no,
                       try_cast(department_no AS INTEGER) AS department_no,
                       try_cast(index_group_no AS INTEGER) AS index_group_no,
                       try_cast(perceived_colour_master_id AS INTEGER) AS colour_master_id
                FROM read_csv_auto({_literal(article_path)},header=true,all_varchar=true)
                """
            )
            connection.execute(
                f"""
                CREATE TABLE history12 AS
                SELECT t.customer_id,t.article_id,t.t_dat,a.product_code
                FROM read_parquet({_literal(transactions_path)}) t
                SEMI JOIN users u USING(customer_id)
                JOIN article_dim a USING(article_id)
                WHERE t.t_dat>={cutoff_sql}-INTERVAL 12 WEEK AND t.t_dat<{cutoff_sql}
                """
            )
            connection.execute(
                """
                CREATE TABLE ui_days AS
                SELECT DISTINCT customer_id,article_id,t_dat FROM history12
                """
            )
            connection.execute(
                """
                CREATE TABLE ui_gap_rows AS
                SELECT customer_id,article_id,t_dat,
                       date_diff('day',lag(t_dat) OVER(
                           PARTITION BY customer_id,article_id ORDER BY t_dat),t_dat) AS gap_days
                FROM ui_days
                """
            )
            connection.execute(
                f"""
                CREATE TABLE repeat_ui AS
                SELECT customer_id,article_id,
                       count(*)::BIGINT AS purchase_days_12w,
                       count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 28 DAY)::BIGINT
                           AS purchase_days_28d,
                       median(gap_days)::DOUBLE AS median_gap_days_12w,
                       arg_max(gap_days,t_dat) FILTER(WHERE gap_days IS NOT NULL)::DOUBLE
                           AS last_gap_days
                FROM ui_gap_rows GROUP BY customer_id,article_id
                """
            )
            connection.execute(
                """
                CREATE TABLE repeat_product AS
                SELECT customer_id,product_code,count(DISTINCT t_dat)::BIGINT AS purchase_days_12w
                FROM history12 GROUP BY customer_id,product_code
                """
            )
            connection.execute(
                f"""
                CREATE TABLE trend28 AS
                SELECT t.article_id,
                       count(*) FILTER(WHERE t.t_dat>={cutoff_sql}-INTERVAL 3 DAY)::BIGINT AS events_3d,
                       count(*) FILTER(WHERE t.t_dat>={cutoff_sql}-INTERVAL 14 DAY
                                        AND t.t_dat<{cutoff_sql}-INTERVAL 7 DAY)::BIGINT AS events_prev7d,
                       count(DISTINCT t.customer_id) FILTER(
                           WHERE t.t_dat>={cutoff_sql}-INTERVAL 7 DAY)::BIGINT AS customers_7d,
                       count(DISTINCT t.customer_id) FILTER(
                           WHERE t.t_dat>={cutoff_sql}-INTERVAL 14 DAY
                             AND t.t_dat<{cutoff_sql}-INTERVAL 7 DAY)::BIGINT AS customers_prev7d,
                       count(*) FILTER(WHERE t.t_dat>={cutoff_sql}-INTERVAL 7 DAY)::BIGINT AS week0,
                       count(*) FILTER(WHERE t.t_dat>={cutoff_sql}-INTERVAL 14 DAY
                                        AND t.t_dat<{cutoff_sql}-INTERVAL 7 DAY)::BIGINT AS week1,
                       count(*) FILTER(WHERE t.t_dat>={cutoff_sql}-INTERVAL 21 DAY
                                        AND t.t_dat<{cutoff_sql}-INTERVAL 14 DAY)::BIGINT AS week2,
                       count(*) FILTER(WHERE t.t_dat>={cutoff_sql}-INTERVAL 28 DAY
                                        AND t.t_dat<{cutoff_sql}-INTERVAL 21 DAY)::BIGINT AS week3
                FROM read_parquet({_literal(transactions_path)}) t
                WHERE t.t_dat>={cutoff_sql}-INTERVAL 28 DAY AND t.t_dat<{cutoff_sql}
                GROUP BY t.article_id
                """
            )
            connection.execute(
                f"""
                CREATE TABLE item_first AS
                SELECT article_id,min(t_dat) AS first_sale
                FROM read_parquet({_literal(transactions_path)})
                WHERE t_dat<{cutoff_sql} GROUP BY article_id
                """
            )
            connection.execute(
                f"""
                CREATE TABLE product_type_trend AS
                SELECT a.product_type_no,
                       count(*) FILTER(WHERE t.t_dat>={cutoff_sql}-INTERVAL 7 DAY)::BIGINT AS events_7d,
                       count(*) FILTER(WHERE t.t_dat>={cutoff_sql}-INTERVAL 14 DAY
                                        AND t.t_dat<{cutoff_sql}-INTERVAL 7 DAY)::BIGINT AS events_prev7d
                FROM read_parquet({_literal(transactions_path)}) t
                JOIN article_dim a USING(article_id)
                WHERE t.t_dat>={cutoff_sql}-INTERVAL 14 DAY AND t.t_dat<{cutoff_sql}
                GROUP BY a.product_type_no
                """
            )
            connection.execute(
                f"""
                CREATE TABLE prior_year AS
                SELECT a.product_type_no,a.garment_group_no,a.department_no,
                       a.index_group_no,a.colour_master_id
                FROM read_parquet({_literal(transactions_path)}) t
                JOIN article_dim a USING(article_id)
                WHERE t.t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY
                  AND t.t_dat<{cutoff_sql}-INTERVAL 1 YEAR
                """
            )
            prior_total = int(connection.execute("SELECT count(*) FROM prior_year").fetchone()[0])
            if prior_total <= 0:
                raise RuntimeError(f"M3.2 prior-year window is empty: {cutoff}")
            for table, key in (
                ("py_product_type", "product_type_no"),
                ("py_garment", "garment_group_no"),
                ("py_department", "department_no"),
                ("py_index_group", "index_group_no"),
                ("py_colour", "colour_master_id"),
            ):
                connection.execute(
                    f"""
                    CREATE TABLE {table} AS
                    SELECT {key},count(*)::DOUBLE/{prior_total}.0 AS share_28d
                    FROM prior_year GROUP BY {key}
                    """
                )
            select_sql = f"""
                SELECT b.*,
                    coalesce(ui.purchase_days_12w,0)::BIGINT AS user_item_purchase_days_12w,
                    coalesce(ui.purchase_days_28d,0)::BIGINT AS user_item_purchase_days_28d,
                    ui.median_gap_days_12w AS user_item_median_gap_days_12w,
                    ui.last_gap_days AS user_item_last_gap_days,
                    b.user_item_days_since_last_purchase/nullif(ui.median_gap_days_12w,0)
                        AS user_item_cadence_ratio,
                    coalesce(pc.purchase_days_12w,0)::BIGINT
                        AS user_product_code_purchase_days_12w,
                    (coalesce(ui.purchase_days_12w,0)>0)::UTINYINT AS exact_prior_purchase_flag,
                    (coalesce(ui.purchase_days_12w,0)>=2)::UTINYINT AS exact_multi_day_repeat_flag,
                    (coalesce(pc.purchase_days_12w,0)>0)::UTINYINT AS product_code_prior_purchase_flag,
                    (coalesce(ui.purchase_days_12w,0)>0)::INT*coalesce(b.item_trend_7d_vs_28d,0)
                        AS exact_repeat_x_item_trend,
                    (coalesce(ui.purchase_days_12w,0)>0)::INT*coalesce(b.product_family_present,0)
                        AS exact_repeat_x_product_family,
                    (coalesce(ui.purchase_days_12w,0)>0)::INT*coalesce(b.user_day_covisit_present,0)
                        AS exact_repeat_x_covisit,
                    (coalesce(ui.purchase_days_12w,0)>0)::INT*coalesce(b.item2vec_present,0)
                        AS exact_repeat_x_item2vec,
                    (coalesce(ui.purchase_days_12w,0)>0)::INT*ln(1+coalesce(b.user_history_events_12w,0))
                        AS exact_repeat_x_log_user_activity,
                    coalesce(tr.events_3d,0)::BIGINT AS item_events_3d,
                    coalesce(tr.events_prev7d,0)::BIGINT AS item_events_previous_7d,
                    coalesce(tr.customers_7d,0)::BIGINT AS item_unique_customers_7d,
                    coalesce(tr.customers_prev7d,0)::BIGINT AS item_unique_customers_previous_7d,
                    coalesce(tr.week0,0)::BIGINT AS item_events_week0,
                    coalesce(tr.week1,0)::BIGINT AS item_events_week1,
                    coalesce(tr.week2,0)::BIGINT AS item_events_week2,
                    coalesce(tr.week3,0)::BIGINT AS item_events_week3,
                    (coalesce(tr.week0,0)+1.0)/(coalesce(tr.week1,0)+1.0)
                        AS item_velocity_7d_vs_previous_7d,
                    (coalesce(tr.customers_7d,0)+1.0)/(coalesce(tr.customers_prev7d,0)+1.0)
                        AS item_customer_velocity_7d_vs_previous_7d,
                    (3*coalesce(tr.week0,0)+coalesce(tr.week1,0)-coalesce(tr.week2,0)
                        -3*coalesce(tr.week3,0))/10.0 AS item_weekly_slope_4w,
                    date_diff('day',first.first_sale,{cutoff_sql})::BIGINT AS item_age_days,
                    coalesce(b.item_events_12w,0)/greatest(
                        least(date_diff('day',first.first_sale,{cutoff_sql}),84),1)
                        AS item_events_per_age_day_12w,
                    coalesce(pt.events_7d,0)::BIGINT AS product_type_events_7d,
                    coalesce(pt.events_prev7d,0)::BIGINT AS product_type_events_previous_7d,
                    (coalesce(pt.events_7d,0)+1.0)/(coalesce(pt.events_prev7d,0)+1.0)
                        AS product_type_velocity_7d_vs_previous_7d,
                    coalesce(pypt.share_28d,0.0) AS prior_year_product_type_share_28d,
                    coalesce(pygg.share_28d,0.0) AS prior_year_garment_share_28d,
                    coalesce(pydp.share_28d,0.0) AS prior_year_department_share_28d,
                    coalesce(pyig.share_28d,0.0) AS prior_year_index_group_share_28d,
                    coalesce(pycl.share_28d,0.0) AS prior_year_colour_share_28d
                FROM base b
                JOIN article_dim a USING(article_id)
                LEFT JOIN repeat_ui ui USING(customer_id,article_id)
                LEFT JOIN repeat_product pc
                  ON b.customer_id=pc.customer_id AND a.product_code=pc.product_code
                LEFT JOIN trend28 tr USING(article_id)
                LEFT JOIN item_first first USING(article_id)
                LEFT JOIN product_type_trend pt USING(product_type_no)
                LEFT JOIN py_product_type pypt USING(product_type_no)
                LEFT JOIN py_garment pygg USING(garment_group_no)
                LEFT JOIN py_department pydp USING(department_no)
                LEFT JOIN py_index_group pyig USING(index_group_no)
                LEFT JOIN py_colour pycl USING(colour_master_id)
            """
            connection.execute(
                f"COPY ({select_sql}) TO {_literal(feature_path)} (FORMAT PARQUET,COMPRESSION ZSTD)"
            )
            audit = connection.execute(
                f"""
                WITH output AS (SELECT * FROM read_parquet({_literal(feature_path)})),
                fwd AS (
                    SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w
                    FROM output EXCEPT ALL
                    SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w
                    FROM base
                ), rev AS (
                    SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w
                    FROM base EXCEPT ALL
                    SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w
                    FROM output
                )
                SELECT (SELECT count(*) FROM base),(SELECT count(*) FROM output),
                       (SELECT count(*) FROM fwd),(SELECT count(*) FROM rev),
                       (SELECT count(*)-count(DISTINCT (customer_id,article_id)) FROM output),
                       (SELECT max(t_dat) FROM history12)
                """
            ).fetchone()
            if tuple(int(value) for value in audit[:5]) != (
                int(audit[0]), int(audit[0]), 0, 0, 0
            ):
                raise RuntimeError(f"M3.2 cache identity audit failed: {cutoff} {audit[:5]}")
            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "completed",
                "cutoff": cutoff,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract": {
                    "repeat_history": "cutoff-12w <= t_dat < cutoff; raw events retained, distinct days only as features",
                    "trend_history": "cutoff-28d <= t_dat < cutoff; global full transactions",
                    "seasonal_history": "cutoff-1y-28d <= t_dat < cutoff-1y; global category shares",
                    "plain_calendar": "excluded because constant inside each cutoff ranking group",
                    "features": {
                        "repeat_dynamics": REPEAT_DYNAMICS_FEATURES,
                        "trend_acceleration": TREND_ACCELERATION_FEATURES,
                        "seasonal_prior": SEASONAL_PRIOR_FEATURES,
                    },
                },
                "inputs": {
                    "source": source_identity,
                    "transactions": transaction_identity,
                    "articles": article_identity,
                },
                "audit": {
                    "source_rows": int(audit[0]),
                    "output_rows": int(audit[1]),
                    "identity_except_all_forward": int(audit[2]),
                    "identity_except_all_reverse": int(audit[3]),
                    "duplicate_user_items": int(audit[4]),
                    "latest_repeat_history_date": str(audit[5]),
                    "prior_year_rows": prior_total,
                },
                "artifact": _file_identity(feature_path),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(manifest_path, manifest)
            results[cutoff] = manifest
        finally:
            connection.close()
    return results


def _classify(deltas: dict[str, float]) -> str:
    if all(value > 0 for value in deltas.values()):
        return "improves_all_three"
    if all(value < 0 for value in deltas.values()):
        return "regresses_all_three"
    return "cross_window_unstable_or_tied"


def summarize_m32(
    development: dict[str, Any], m3_metrics: dict[str, Any], metric_k: int
) -> dict[str, Any]:
    metric = f"map@{metric_k}"
    names = list(variant_feature_sets())
    orderings: dict[str, Any] = {}
    for name in names:
        ordering_name = f"{name}__inactive_rrf"
        values = {
            window: float(row["evaluation"]["orderings"][ordering_name]["segments"]["overall"][metric])
            for window, row in development.items()
        }
        orderings[name] = {
            "window_map@12": values,
            "mean_map@12": float(np.mean(list(values.values()))),
        }
    anchor = orderings["anchor"]
    for name, row in orderings.items():
        deltas = {
            window: row["window_map@12"][window] - anchor["window_map@12"][window]
            for window in development
        }
        row["window_delta_vs_anchor_map@12"] = deltas
        row["mean_delta_vs_anchor_map@12"] = float(np.mean(list(deltas.values())))
        row["classification"] = "anchor" if name == "anchor" else _classify(deltas)
        row["accepted"] = name != "anchor" and all(value >= 0 for value in deltas.values()) and row[
            "mean_delta_vs_anchor_map@12"
        ] > 0
    prior_anchor = m3_metrics["summary"]["orderings"][FALLBACK_NAME]["window_map@12"]
    reproduction = {
        window: {
            "expected": float(prior_anchor[window]),
            "actual": float(anchor["window_map@12"][window]),
            "absolute_difference": abs(
                float(prior_anchor[window]) - float(anchor["window_map@12"][window])
            ),
        }
        for window in development
    }
    candidate_parity: dict[str, Any] = {}
    for window, row in development.items():
        reference = row["evaluation"]["orderings"]["anchor__inactive_rrf"]["segments"]["overall"]
        candidate_parity[window] = {
            name: {
                "recall_difference": float(
                    row["evaluation"]["orderings"][f"{name}__inactive_rrf"]["segments"]["overall"]
                    ["candidate_recall@100"]
                    - reference["candidate_recall@100"]
                ),
                "oracle_difference": float(
                    row["evaluation"]["orderings"][f"{name}__inactive_rrf"]["segments"]["overall"]
                    [f"oracle_map@{metric_k}"]
                    - reference[f"oracle_map@{metric_k}"]
                ),
            }
            for name in names
        }
    return {
        "orderings": orderings,
        "anchor_reproduction": reproduction,
        "anchor_reproduction_passed": all(
            row["absolute_difference"] <= 1e-12 for row in reproduction.values()
        ),
        "candidate_parity": candidate_parity,
        "candidate_parity_passed": all(
            abs(value) <= 1e-12
            for row in candidate_parity.values()
            for variant in row.values()
            for value in variant.values()
        ),
        "selection": "pre_registered_variants_only_no_post_hoc_combination",
        "seasonality_boundary": "summer_windows_only_not_full_year_validation",
        "final_week": "not_run",
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    windows = list(ROLLING_PROTOCOL)
    lines = [
        "# M3.2：复购动态、趋势加速度与季节先验",
        "",
        "## 协议",
        "",
        "- frozen expanded-300 candidate pool；三条严格 inner/outer 时间链；每个 variant 独立 early stopping。",
        "- 7 个预注册 variants；不根据 outer 结果临时组合特征。",
        "- 去年同期品类 share 使用全量 cutoff-safe 交易；plain month 因组内恒定而排除。",
        "- 当前 outer 仅覆盖 6/7/8 月；seasonal 结果不能外推为全年稳定性。",
        "- final week 未运行。",
        "",
        "## MAP@12",
        "",
        "| variant | " + " | ".join(windows) + " | mean | mean delta | classification | accepted |",
        "|---|" + "---:|" * (len(windows) + 2) + "---|---|",
    ]
    for name, row in summary["orderings"].items():
        lines.append(
            "| " + name + " | "
            + " | ".join(f"{row['window_map@12'][window]:.6f}" for window in windows)
            + f" | {row['mean_map@12']:.6f} | {row['mean_delta_vs_anchor_map@12']:+.6f} | "
            + f"{row['classification']} | {row['accepted']} |"
        )
    lines.extend(
        [
            "",
            "## 守恒与资源",
            "",
            f"- M3.0 anchor exact reproduction：{summary['anchor_reproduction_passed']}。",
            f"- candidate Recall/Oracle parity：{summary['candidate_parity_passed']}。",
            f"- wall time：{result['elapsed_seconds']:.2f} 秒。",
            "- peak working set：" + (
                f"{result['resources']['peak_working_set_bytes']/1024**3:.2f} GiB。"
                if result["resources"]["peak_working_set_bytes"]
                else "未能读取。"
            ),
            f"- 主产物：{result['resources']['artifact_count']} 个，"
            f"{result['resources']['artifact_bytes']/1024**2:.2f} MiB。",
            "- final week 未运行。",
            "",
            "机器可读证据：`metrics.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def _collect_artifacts(development: dict[str, Any]) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for row in development.values():
        for model in row["inner_models"].values():
            identities.append(_file_identity(Path(model["model_path"])))
        for model in row["outer_models"].values():
            identities.append(_file_identity(Path(model["model_path"])))
        identities.append(_file_identity(Path(row["scoring"]["prediction_path"])))
        identities.append(_file_identity(Path(row["scoring"]["evaluation_db"])))
        identities.append(_file_identity(Path(row["inner_category_encoding"]["path"])))
        identities.append(_file_identity(Path(row["outer_category_encoding"]["path"])))
    return identities


def run_m32(
    *,
    raw_dir: Path,
    transactions_path: Path,
    m3_metrics_path: Path,
    source_feature_cache_dir: Path,
    temporal_cache_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    config.validate()
    variants = variant_feature_sets()
    if config.evaluation_role != "development" or config.candidate_k != 300:
        raise ValueError("M3.2 requires development-only candidate_k=300")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    m3_metrics = _read_json(m3_metrics_path.resolve())
    if m3_metrics.get("schema_version") != "m3.0-three-window-robustness-v1":
        raise ValueError("M3.2 requires M3.0 evidence")
    if m3_metrics.get("status") != "measured" or m3_metrics["contract"].get("final_week") != "not_run":
        raise ValueError("M3.2 M3.0 evidence boundary failed")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        cache = build_temporal_feature_cache(
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            source_cache_dir=source_feature_cache_dir,
            cache_dir=temporal_cache_dir,
            cutoffs=ALL_CUTOFFS,
        )
        all_features = list(dict.fromkeys(feature for values in variants.values() for feature in values))
        development: dict[str, Any] = {}
        for window, protocol in ROLLING_PROTOCOL.items():
            print(f"M3.2 {window}: inner fits", flush=True)
            window_dir = artifact_dir / window
            window_dir.mkdir()
            inner_train, inner_sizes, inner_sampling = load_distribution_sample(
                [_cache_paths(temporal_cache_dir, cutoff)[0] for cutoff in protocol["inner_train"]],
                all_features,
                seed=SAMPLING_SEED,
            )
            inner_maps = build_category_maps(inner_train)
            inner_map_evidence = _save_category_maps(window_dir / "inner-category-maps.json", inner_maps)
            inner_validation, inner_validation_sizes, truth_counts, validation_evidence = load_inner_validation(
                dataset_path=_cache_paths(temporal_cache_dir, protocol["inner_validation"])[0],
                transactions_path=transactions_path,
                cutoff=protocol["inner_validation"],
                features=all_features,
            )
            inner_models: dict[str, Any] = {}
            for name, features in variants.items():
                _, evidence = train_inner_ranker(
                    train_frame=inner_train,
                    train_group_sizes=inner_sizes,
                    train_evidence=inner_sampling,
                    validation_frame=inner_validation,
                    validation_group_sizes=inner_validation_sizes,
                    validation_truth_counts=truth_counts,
                    validation_evidence=validation_evidence,
                    features=features,
                    name=name,
                    artifact_dir=window_dir,
                    config=config,
                    category_maps=inner_maps,
                )
                inner_models[name] = evidence
            del inner_train, inner_sizes, inner_validation, inner_validation_sizes
            gc.collect()

            print(f"M3.2 {window}: outer refits and scoring", flush=True)
            outer_train, outer_sizes, outer_sampling = load_distribution_sample(
                [_cache_paths(temporal_cache_dir, cutoff)[0] for cutoff in protocol["outer_train"]],
                all_features,
                seed=SAMPLING_SEED,
            )
            outer_maps = build_category_maps(outer_train)
            outer_map_evidence = _save_category_maps(window_dir / "outer-category-maps.json", outer_maps)
            models: dict[str, tuple[Any, list[str], dict[str, dict[int, int]]]] = {}
            outer_models: dict[str, Any] = {}
            for name, features in variants.items():
                rounds = int(inner_models[name]["best_iteration"])
                model, evidence = train_lambdarank(
                    frame=outer_train,
                    group_sizes=outer_sizes,
                    group_evidence=outer_sampling,
                    features=features,
                    name=name,
                    use_ipw=False,
                    artifact_dir=window_dir,
                    config=replace(config, num_boost_round=rounds),
                    category_maps=outer_maps,
                )
                evidence["selected_rounds_from_inner"] = rounds
                evidence["inner_selection_score"] = inner_models[name]["best_score"]
                models[name] = (model, features, outer_maps)
                outer_models[name] = evidence
            del outer_train, outer_sizes
            gc.collect()
            scoring = score_models(
                dataset_path=_cache_paths(temporal_cache_dir, protocol["outer_validation"])[0],
                models=models,
                evaluation_db=window_dir / "evaluation.duckdb",
                prediction_path=window_dir / "predictions.parquet",
                config=config,
            )
            del models
            gc.collect()
            evaluation = _evaluate_models(
                evaluation_db=window_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                metric_k=config.metric_k,
                variant_names=list(variants),
            )
            evaluation = _attach_popularity_segments(
                evaluation=evaluation,
                evaluation_db=window_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                variant_names=list(variants),
                budget=config.candidate_k,
            )
            development[window] = {
                "protocol": protocol,
                "inner_models": inner_models,
                "outer_models": outer_models,
                "inner_category_encoding": inner_map_evidence,
                "outer_category_encoding": outer_map_evidence,
                "outer_sampling": outer_sampling,
                "scoring": scoring,
                "evaluation": evaluation,
            }
        summary = summarize_m32(development, m3_metrics, config.metric_k)
        if not summary["anchor_reproduction_passed"]:
            raise RuntimeError("M3.2 failed exact M3.0 anchor reproduction")
        if not summary["candidate_parity_passed"]:
            raise RuntimeError("M3.2 candidate Recall/Oracle parity failed")
        artifacts = _collect_artifacts(development)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M3.2",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "three_summer_rolling_development_windows_no_final_week",
            "config": asdict(config),
            "contract": {
                "candidate_pool": "frozen M3.0 six-source Top100 plus up to 200 Item2Vec-only",
                "catalog_protocol": "optimistic_all_articles",
                "variants": variants,
                "repurchase_route_features": REPURCHASE_ROUTE_FEATURES,
                "feature_families": {
                    "repeat_dynamics": REPEAT_DYNAMICS_FEATURES,
                    "trend_acceleration": TREND_ACCELERATION_FEATURES,
                    "seasonal_prior": SEASONAL_PRIOR_FEATURES,
                },
                "selection": "mean MAP improves and no outer window regresses; no post-hoc combinations",
                "seasonality_boundary": "outer windows cover only 2020-06/07/08",
                "rolling_protocol": ROLLING_PROTOCOL,
                "final_week": "not_run",
            },
            "inputs": {
                "m3_metrics": _file_identity(m3_metrics_path),
                "transactions": _file_identity(transactions_path),
            },
            "feature_cache": cache,
            "development": development,
            "summary": summary,
            "resources": {
                "peak_working_set_bytes": _peak_working_set_bytes(),
                "artifact_count": len(artifacts),
                "artifact_bytes": int(sum(item["bytes"] for item in artifacts)),
                "artifacts": artifacts,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "artifacts": {},
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M3_2_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path.resolve()),
            "report": str(report_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
        }
        _write_json(metrics_path, result)
        report_path.write_text(render_report(result), encoding="utf-8")
        return result
    except Exception as error:
        _write_json(
            output_dir / f"failure-{time.time_ns()}.json",
            {
                "schema_version": "m3.2-failure-v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
