from __future__ import annotations

import gc
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from .m2 import (
    M2Config,
    _evaluate_ordering,
    _literal,
    _write_json,
    build_category_maps,
)
from .m21 import CandidateWindow, inactive_fallback_order_expression
from .m26 import _evaluate_models, _save_category_maps
from .m29 import (
    M29Window,
    _attach_popularity_segments,
    _read_json,
    build_expanded_candidate_cache,
    build_feature_cache,
    build_source_cache,
)
from .m210 import (
    _file_identity,
    _target_feature_paths,
    build_target_aware_cache,
    score_models,
)
from .m211 import SAMPLING_SEED, load_distribution_sample, train_lambdarank
from .m212 import feature_sets, load_inner_validation, train_inner_ranker
from .m3 import ANCHOR_NAME, FALLBACK_NAME, _peak_working_set_bytes


SCHEMA_VERSION = "m3.3-cross-season-adaptive-seasonal-v1"
CACHE_SCHEMA_VERSION = "m3.3-seasonal-feature-cache-v1"

ALL_CUTOFFS = (
    "2019-11-27",
    "2019-12-25",
    "2020-01-22",
    "2020-02-19",
    "2020-03-18",
    "2020-04-29",
    "2020-05-27",
    "2020-06-24",
    "2020-07-22",
    "2020-08-19",
)

ROLLING_PROTOCOL = {
    "winter_20200122": {
        "season": "winter",
        "inner_train": ["2019-11-27"],
        "inner_validation": "2019-12-25",
        "outer_train": ["2019-11-27", "2019-12-25"],
        "outer_validation": "2020-01-22",
        "known_context": "Chinese New Year and promotion effects may confound weather seasonality",
    },
    "spring_20200318": {
        "season": "spring",
        "inner_train": ["2020-01-22"],
        "inner_validation": "2020-02-19",
        "outer_train": ["2020-01-22", "2020-02-19"],
        "outer_validation": "2020-03-18",
        "known_context": "COVID-era demand and supply changes may differ from prior year",
    },
    "early_summer_20200624": {
        "season": "early_summer",
        "inner_train": ["2020-04-29"],
        "inner_validation": "2020-05-27",
        "outer_train": ["2020-04-29", "2020-05-27"],
        "outer_validation": "2020-06-24",
        "known_context": "reuses a previously observed rolling development window",
    },
    "late_summer_20200819": {
        "season": "late_summer",
        "inner_train": ["2020-06-24"],
        "inner_validation": "2020-07-22",
        "outer_train": ["2020-06-24", "2020-07-22"],
        "outer_validation": "2020-08-19",
        "known_context": "reuses the M3.2 reversal window and may include season transition",
    },
}

DIMENSIONS = {
    "product_type": "product_type_no",
    "garment": "garment_group_no",
    "department": "department_no",
    "index_group": "index_group_no",
    "colour": "colour_master_id",
}

RAW_PRIOR_FEATURES = [f"prior_year_{name}_share_28d" for name in DIMENSIONS]
NORMALIZED_PRIOR_FEATURES = [
    feature
    for name in DIMENSIONS
    for feature in (
        f"prior_year_{name}_log_lift_28d_vs_84d",
        f"prior_year_{name}_log_support_28d",
    )
]
ADAPTIVE_SEASONAL_FEATURES = [
    feature
    for name in DIMENSIONS
    for feature in (
        f"current_{name}_share_28d",
        f"current_prior_{name}_log_ratio",
        f"current_prior_{name}_agreement",
        f"adaptive_prior_year_{name}_share_28d",
    )
]
ALL_SEASONAL_FEATURES = [
    *RAW_PRIOR_FEATURES,
    *NORMALIZED_PRIOR_FEATURES,
    *ADAPTIVE_SEASONAL_FEATURES,
]

SEASON_SENSITIVE_PRODUCT_TYPE_LIFT = 1.15
SEASON_SENSITIVE_MIN_PRIOR_EVENTS = 50


def validate_rolling_protocol() -> None:
    if list(ROLLING_PROTOCOL) != [
        "winter_20200122",
        "spring_20200318",
        "early_summer_20200624",
        "late_summer_20200819",
    ]:
        raise RuntimeError("M3.3 requires exactly four pre-registered seasonal representatives")
    used: set[str] = set()
    for name, row in ROLLING_PROTOCOL.items():
        sequence = [*row["inner_train"], row["inner_validation"], row["outer_validation"]]
        if sequence != sorted(sequence) or len(sequence) != len(set(sequence)):
            raise RuntimeError(f"M3.3 protocol is not strictly temporal: {name}")
        if row["outer_train"] != [*row["inner_train"], row["inner_validation"]]:
            raise RuntimeError(f"M3.3 outer refit does not extend inner training: {name}")
        used.update(sequence)
    if not used.issubset(ALL_CUTOFFS):
        raise RuntimeError("M3.3 rolling protocol references an undeclared cutoff")
    if any(cutoff >= "2020-09-16" for cutoff in ALL_CUTOFFS):
        raise RuntimeError("M3.3 must not read the final validation week")


def variant_feature_sets() -> dict[str, list[str]]:
    anchor = list(feature_sets()[ANCHOR_NAME])
    variants = {
        "anchor": anchor,
        "raw_prior": [*anchor, *RAW_PRIOR_FEATURES],
        "normalized_prior": [
            *anchor,
            *RAW_PRIOR_FEATURES,
            *NORMALIZED_PRIOR_FEATURES,
        ],
        "adaptive_seasonal": [*anchor, *ALL_SEASONAL_FEATURES],
    }
    validate_variant_feature_sets(variants)
    return variants


def validate_variant_feature_sets(variants: dict[str, list[str]]) -> None:
    if list(variants) != ["anchor", "raw_prior", "normalized_prior", "adaptive_seasonal"]:
        raise RuntimeError("M3.3 requires exactly four pre-registered variants")
    anchor = set(feature_sets()[ANCHOR_NAME])
    if len(ALL_SEASONAL_FEATURES) != len(set(ALL_SEASONAL_FEATURES)):
        raise RuntimeError("M3.3 seasonal feature families overlap")
    if anchor.intersection(ALL_SEASONAL_FEATURES):
        raise RuntimeError("M3.3 seasonal features overlap the frozen anchor")
    if set(variants["raw_prior"]) != anchor | set(RAW_PRIOR_FEATURES):
        raise RuntimeError("M3.3 raw-prior variant drift")
    if set(variants["normalized_prior"]) != anchor | set(RAW_PRIOR_FEATURES) | set(
        NORMALIZED_PRIOR_FEATURES
    ):
        raise RuntimeError("M3.3 normalized-prior variant drift")
    if set(variants["adaptive_seasonal"]) != anchor | set(ALL_SEASONAL_FEATURES):
        raise RuntimeError("M3.3 adaptive-seasonal variant drift")
    if any(feature in {"month", "week_of_year", "day_of_year"} for feature in ALL_SEASONAL_FEATURES):
        raise RuntimeError("M3.3 excludes group-constant plain calendar features")


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
        raise FileNotFoundError(f"incomplete M3.3 cache: {feature_path.parent}")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION or manifest.get("cutoff") != cutoff:
        raise ValueError(f"M3.3 cache schema/cutoff drift: {manifest_path}")
    for name, expected in (
        ("source", source_identity),
        ("transactions", transaction_identity),
        ("articles", article_identity),
    ):
        actual = manifest["inputs"][name]
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError(f"M3.3 cache input drift: {name} {manifest_path}")
    actual_output = _file_identity(feature_path)
    if actual_output["bytes"] != manifest["artifact"]["bytes"] or actual_output["sha256"] != manifest[
        "artifact"
    ]["sha256"]:
        raise ValueError(f"M3.3 cache artifact drift: {feature_path}")
    return manifest


def _js_divergence(
    connection: duckdb.DuckDBPyConnection,
    current_table: str,
    prior_table: str,
    key: str,
) -> float:
    value = connection.execute(
        f"""
        WITH joined AS (
            SELECT coalesce(c.{key},p.{key}) AS category,
                   coalesce(c.share_28d,0.0) AS current_share,
                   coalesce(p.share_28d,0.0) AS prior_share
            FROM {current_table} c FULL OUTER JOIN {prior_table} p USING({key})
        ), values AS (
            SELECT *, (current_share+prior_share)/2.0 AS midpoint FROM joined
        )
        SELECT 0.5*sum(
            CASE WHEN current_share>0 THEN current_share*ln(current_share/midpoint) ELSE 0 END
            + CASE WHEN prior_share>0 THEN prior_share*ln(prior_share/midpoint) ELSE 0 END
        ) FROM values
        """
    ).fetchone()[0]
    return float(value or 0.0)


def build_seasonal_feature_cache(
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
            raise FileExistsError(f"incomplete M3.3 cache root: {feature_path.parent}")
        feature_path.parent.mkdir(parents=True)
        database_path = feature_path.parent / "feature-build.duckdb"
        temp_dir = feature_path.parent / "duckdb-temp"
        temp_dir.mkdir()
        connection = duckdb.connect(str(database_path))
        cutoff_sql = _date(cutoff)
        try:
            connection.execute("SET threads=8")
            connection.execute("SET memory_limit='11GB'")
            connection.execute("PRAGMA disable_progress_bar")
            connection.execute(f"SET temp_directory={_literal(temp_dir)}")
            connection.execute(f"CREATE VIEW base AS SELECT * FROM read_parquet({_literal(source_path)})")
            connection.execute(
                f"""
                CREATE TABLE article_dim AS
                SELECT article_id,
                       try_cast(product_type_no AS INTEGER) AS product_type_no,
                       try_cast(garment_group_no AS INTEGER) AS garment_group_no,
                       try_cast(department_no AS INTEGER) AS department_no,
                       try_cast(index_group_no AS INTEGER) AS index_group_no,
                       try_cast(perceived_colour_master_id AS INTEGER) AS colour_master_id
                FROM read_csv_auto({_literal(article_path)},header=true,all_varchar=true)
                """
            )
            dimension_columns = ",".join(DIMENSIONS.values())
            connection.execute(
                f"""
                CREATE TABLE current_events AS
                SELECT t.t_dat,{dimension_columns}
                FROM read_parquet({_literal(transactions_path)}) t
                JOIN article_dim a USING(article_id)
                WHERE t.t_dat>={cutoff_sql}-INTERVAL 28 DAY AND t.t_dat<{cutoff_sql};
                CREATE TABLE prior_events AS
                SELECT t.t_dat,{dimension_columns}
                FROM read_parquet({_literal(transactions_path)}) t
                JOIN article_dim a USING(article_id)
                WHERE t.t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 84 DAY
                  AND t.t_dat<{cutoff_sql}-INTERVAL 1 YEAR;
                """
            )
            current_total = int(connection.execute("SELECT count(*) FROM current_events").fetchone()[0])
            prior84_total = int(connection.execute("SELECT count(*) FROM prior_events").fetchone()[0])
            prior28_total = int(
                connection.execute(
                    f"SELECT count(*) FROM prior_events WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY"
                ).fetchone()[0]
            )
            if min(current_total, prior28_total, prior84_total) <= 0:
                raise RuntimeError(f"M3.3 seasonal history window is empty: {cutoff}")

            joins: list[str] = []
            select_features: list[str] = []
            drift: dict[str, Any] = {}
            for name, key in DIMENSIONS.items():
                current_table = f"current_{name}"
                prior_table = f"prior_{name}"
                connection.execute(
                    f"""
                    CREATE TABLE {current_table} AS
                    SELECT {key},count(*)::BIGINT AS events_28d,
                           count(*)::DOUBLE/{current_total}.0 AS share_28d
                    FROM current_events GROUP BY {key};
                    CREATE TABLE {prior_table} AS
                    SELECT {key},
                           count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY)::BIGINT
                               AS events_28d,
                           count(*)::BIGINT AS events_84d,
                           count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY)::DOUBLE
                               /{prior28_total}.0 AS share_28d,
                           count(*)::DOUBLE/{prior84_total}.0 AS share_84d
                    FROM prior_events GROUP BY {key};
                    """
                )
                current_alias = f"c_{name}"
                prior_alias = f"p_{name}"
                joins.append(f"LEFT JOIN {current_table} {current_alias} ON a.{key}={current_alias}.{key}")
                joins.append(f"LEFT JOIN {prior_table} {prior_alias} ON a.{key}={prior_alias}.{key}")
                prior_share = f"coalesce({prior_alias}.share_28d,0.0)"
                prior_background = f"coalesce({prior_alias}.share_84d,0.0)"
                current_share = f"coalesce({current_alias}.share_28d,0.0)"
                log_ratio = f"ln(({current_share}+1e-12)/({prior_share}+1e-12))"
                agreement = f"exp(-abs({log_ratio}))"
                select_features.extend(
                    [
                        f"{prior_share} AS prior_year_{name}_share_28d",
                        f"ln(({prior_share}+1e-12)/({prior_background}+1e-12)) "
                        f"AS prior_year_{name}_log_lift_28d_vs_84d",
                        f"ln(1+coalesce({prior_alias}.events_28d,0)) AS prior_year_{name}_log_support_28d",
                        f"{current_share} AS current_{name}_share_28d",
                        f"{log_ratio} AS current_prior_{name}_log_ratio",
                        f"{agreement} AS current_prior_{name}_agreement",
                        f"{prior_share}*{agreement} AS adaptive_prior_year_{name}_share_28d",
                    ]
                )
                drift[name] = {
                    "current_vs_prior_js_divergence": _js_divergence(
                        connection, current_table, prior_table, key
                    ),
                    "current_categories": int(
                        connection.execute(f"SELECT count(*) FROM {current_table}").fetchone()[0]
                    ),
                    "prior_categories": int(
                        connection.execute(f"SELECT count(*) FROM {prior_table}").fetchone()[0]
                    ),
                }

            select_sql = (
                "SELECT b.*," + ",".join(select_features)
                + " FROM base b JOIN article_dim a USING(article_id) "
                + " ".join(joins)
            )
            connection.execute(
                f"COPY ({select_sql}) TO {_literal(feature_path)} (FORMAT PARQUET,COMPRESSION ZSTD)"
            )
            audit = connection.execute(
                f"""
                WITH output AS (SELECT * FROM read_parquet({_literal(feature_path)})),
                fwd AS (
                    SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w
                    FROM output EXCEPT ALL
                    SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w FROM base
                ), rev AS (
                    SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w FROM base
                    EXCEPT ALL
                    SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w FROM output
                )
                SELECT (SELECT count(*) FROM base),(SELECT count(*) FROM output),
                       (SELECT count(*) FROM fwd),(SELECT count(*) FROM rev),
                       (SELECT count(*)-count(DISTINCT (customer_id,article_id)) FROM output),
                       (SELECT max(t_dat) FROM current_events),(SELECT max(t_dat) FROM prior_events)
                """
            ).fetchone()
            if tuple(int(value) for value in audit[:5]) != (int(audit[0]), int(audit[0]), 0, 0, 0):
                raise RuntimeError(f"M3.3 cache identity audit failed: {cutoff} {audit[:5]}")
            feature_stats: dict[str, Any] = {}
            for feature in ALL_SEASONAL_FEATURES:
                values = connection.execute(
                    f"""
                    SELECT count(*) FILTER(WHERE {feature} IS NULL),
                           min({feature}),approx_quantile({feature},0.5),
                           approx_quantile({feature},0.95),max({feature})
                    FROM read_parquet({_literal(feature_path)})
                    """
                ).fetchone()
                feature_stats[feature] = {
                    "null_rows": int(values[0]),
                    "min": float(values[1]),
                    "median": float(values[2]),
                    "p95": float(values[3]),
                    "max": float(values[4]),
                }
            manifest = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "completed",
                "cutoff": cutoff,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "contract": {
                    "current_history": "cutoff-28d <= t_dat < cutoff; global full transactions",
                    "prior_history": "cutoff-1y-84d <= t_dat < cutoff-1y; last 28d is same-period prior",
                    "plain_calendar": "excluded because constant inside each cutoff ranking group",
                    "feature_families": {
                        "raw_prior": RAW_PRIOR_FEATURES,
                        "normalized_prior": NORMALIZED_PRIOR_FEATURES,
                        "adaptive_seasonal": ADAPTIVE_SEASONAL_FEATURES,
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
                    "latest_current_history_date": str(audit[5]),
                    "latest_prior_history_date": str(audit[6]),
                    "current_rows_28d": current_total,
                    "prior_rows_28d": prior28_total,
                    "prior_rows_84d": prior84_total,
                    "distribution_drift": drift,
                    "feature_stats": feature_stats,
                },
                "artifact": _file_identity(feature_path),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(manifest_path, manifest)
            results[cutoff] = manifest
        finally:
            connection.close()
    return results


def evaluate_season_sensitive_truth(
    *,
    evaluation_db: Path,
    transactions_path: Path,
    articles_path: Path,
    cutoff: str,
    variant_names: list[str],
    metric_k: int,
) -> dict[str, Any]:
    connection = duckdb.connect(str(evaluation_db))
    cutoff_sql = _date(cutoff)
    try:
        connection.execute("SET threads=8")
        connection.execute("PRAGMA disable_progress_bar")
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE m33_article_dim AS
            SELECT article_id,try_cast(product_type_no AS INTEGER) AS product_type_no
            FROM read_csv_auto({_literal(articles_path)},header=true,all_varchar=true);
            CREATE OR REPLACE TEMP TABLE m33_prior_events AS
            SELECT t.t_dat,a.product_type_no
            FROM read_parquet({_literal(transactions_path)}) t
            JOIN m33_article_dim a USING(article_id)
            WHERE t.t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 84 DAY
              AND t.t_dat<{cutoff_sql}-INTERVAL 1 YEAR;
            """
        )
        total84, total28 = connection.execute(
            f"""
            SELECT count(*),count(*) FILTER(
                WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY)
            FROM m33_prior_events
            """
        ).fetchone()
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE m33_seasonal_product_types AS
            WITH stats AS (
                SELECT product_type_no,
                       count(*)::DOUBLE/{int(total84)}.0 AS share84,
                       count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY)
                           AS events28,
                       count(*) FILTER(WHERE t_dat>={cutoff_sql}-INTERVAL 1 YEAR-INTERVAL 28 DAY)::DOUBLE
                           /{int(total28)}.0 AS share28
                FROM m33_prior_events GROUP BY product_type_no
            )
            SELECT *,share28/nullif(share84,0) AS lift
            FROM stats
            WHERE events28>={SEASON_SENSITIVE_MIN_PRIOR_EVENTS}
              AND share28/nullif(share84,0)>={SEASON_SENSITIVE_PRODUCT_TYPE_LIFT};
            CREATE OR REPLACE TEMP TABLE m33_eval_users AS SELECT DISTINCT customer_id FROM predictions;
            CREATE OR REPLACE TEMP TABLE m2_eval_truth AS
            SELECT DISTINCT t.customer_id,t.article_id,'warm'::VARCHAR AS item_temperature
            FROM read_parquet({_literal(transactions_path)}) t
            SEMI JOIN m33_eval_users USING(customer_id)
            JOIN m33_article_dim a USING(article_id)
            JOIN m33_seasonal_product_types s USING(product_type_no)
            WHERE t.t_dat>={cutoff_sql} AND t.t_dat<{cutoff_sql}+INTERVAL 7 DAY;
            """
        )
        truth_users, truth_pairs = connection.execute(
            "SELECT count(DISTINCT customer_id),count(*) FROM m2_eval_truth"
        ).fetchone()
        product_types = int(connection.execute("SELECT count(*) FROM m33_seasonal_product_types").fetchone()[0])
        orderings: dict[str, Any] = {}
        if int(truth_users) > 0:
            for name in variant_names:
                ordering = _evaluate_ordering(
                    connection,
                    f"{name}__season_sensitive",
                    inactive_fallback_order_expression(f"score_{name}"),
                    metric_k,
                )
                orderings[name] = ordering["segments"]["overall"]
        return {
            "definition": {
                "dimension": "product_type",
                "prior_year_share_lift_threshold": SEASON_SENSITIVE_PRODUCT_TYPE_LIFT,
                "minimum_prior_28d_events": SEASON_SENSITIVE_MIN_PRIOR_EVENTS,
                "truth_denominator": "all distinct validation-week truth in qualifying product types for sampled users",
            },
            "qualifying_product_types": product_types,
            "users": int(truth_users),
            "truth_pairs": int(truth_pairs),
            "orderings": orderings,
        }
    finally:
        connection.close()


def _classify(deltas: dict[str, float]) -> str:
    if all(value > 0 for value in deltas.values()):
        return "improves_all_four"
    if all(value < 0 for value in deltas.values()):
        return "regresses_all_four"
    return "cross_season_unstable_or_tied"


def summarize_m33(
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
        seasonal_values = {
            window: float(row["season_sensitive"]["orderings"].get(name, {}).get(metric, 0.0))
            for window, row in development.items()
        }
        orderings[name] = {
            "window_map@12": values,
            "mean_map@12": float(np.mean(list(values.values()))),
            "season_sensitive_window_map@12": seasonal_values,
            "season_sensitive_mean_map@12": float(np.mean(list(seasonal_values.values()))),
        }
    anchor = orderings["anchor"]
    for name, row in orderings.items():
        deltas = {
            window: row["window_map@12"][window] - anchor["window_map@12"][window]
            for window in development
        }
        seasonal_deltas = {
            window: row["season_sensitive_window_map@12"][window]
            - anchor["season_sensitive_window_map@12"][window]
            for window in development
        }
        row["window_delta_vs_anchor_map@12"] = deltas
        row["mean_delta_vs_anchor_map@12"] = float(np.mean(list(deltas.values())))
        row["season_sensitive_window_delta_vs_anchor_map@12"] = seasonal_deltas
        row["season_sensitive_mean_delta_vs_anchor_map@12"] = float(
            np.mean(list(seasonal_deltas.values()))
        )
        row["classification"] = "anchor" if name == "anchor" else _classify(deltas)
        row["accepted"] = (
            name != "anchor"
            and all(value >= 0 for value in deltas.values())
            and row["mean_delta_vs_anchor_map@12"] > 0
        )

    prior_anchor = m3_metrics["summary"]["orderings"][FALLBACK_NAME]["window_map@12"]
    shared_windows = {
        "early_summer_20200624": "roll_20200624",
        "late_summer_20200819": "roll_20200819",
    }
    reproduction = {
        window: {
            "m3_window": m3_window,
            "expected": float(prior_anchor[m3_window]),
            "actual": float(anchor["window_map@12"][window]),
            "absolute_difference": abs(
                float(prior_anchor[m3_window]) - float(anchor["window_map@12"][window])
            ),
        }
        for window, m3_window in shared_windows.items()
    }
    candidate_parity: dict[str, Any] = {}
    for window, row in development.items():
        reference = row["evaluation"]["orderings"]["anchor__inactive_rrf"]["segments"]["overall"]
        candidate_parity[window] = {
            name: {
                "recall_difference": float(
                    row["evaluation"]["orderings"][f"{name}__inactive_rrf"]["segments"]["overall"][
                        "candidate_recall@100"
                    ]
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
        "shared_anchor_reproduction": reproduction,
        "shared_anchor_reproduction_passed": all(
            row["absolute_difference"] <= 1e-12 for row in reproduction.values()
        ),
        "candidate_parity": candidate_parity,
        "candidate_parity_passed": all(
            abs(value) <= 1e-12
            for row in candidate_parity.values()
            for variant in row.values()
            for value in variant.values()
        ),
        "selection": "four pre-registered variants; mean improves and no outer season regresses",
        "fresh_outer_windows": ["winter_20200122", "spring_20200318"],
        "reused_outer_windows": ["early_summer_20200624", "late_summer_20200819"],
        "final_week": "not_run",
    }


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


def render_report(result: dict[str, Any]) -> str:
    windows = list(ROLLING_PROTOCOL)
    summary = result["summary"]
    lines = [
        "# M3.3：跨季 seasonal prior 与当年趋势一致度",
        "",
        "## 协议",
        "",
        "- winter/spring/early-summer/late-summer 四条严格 inner/outer 时间链。",
        "- raw prior、normalized prior、adaptive seasonal 三层预注册特征；不做 outer-driven 组合。",
        "- frozen expanded-300 Item2Vec pool、30:1 sampling、LambdaRank 与 inactive RRF fallback。",
        "- final week 未运行。",
        "",
        "## Overall MAP@12",
        "",
        "| variant | " + " | ".join(windows) + " | mean | mean delta | classification | accepted |",
        "|---|" + "---:|" * (len(windows) + 2) + "---|---|",
    ]
    for name, row in summary["orderings"].items():
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(f"{row['window_map@12'][window]:.6f}" for window in windows)
            + f" | {row['mean_map@12']:.6f} | {row['mean_delta_vs_anchor_map@12']:+.6f} | "
            + f"{row['classification']} | {row['accepted']} |"
        )
    lines.extend(
        [
            "",
            "## 审计",
            "",
            f"- shared summer anchor reproduction：{summary['shared_anchor_reproduction_passed']}。",
            f"- candidate Recall/Oracle parity：{summary['candidate_parity_passed']}。",
            f"- wall time：{result['elapsed_seconds']:.2f} 秒。",
            "- peak working set："
            + (
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


def run_m33(
    *,
    raw_dir: Path,
    work_dir: Path,
    transactions_path: Path,
    m28_metrics_path: Path,
    m3_metrics_path: Path,
    windows: list[CandidateWindow],
    source_cache_dir: Path,
    m29_cache_dir: Path,
    target_feature_cache_dir: Path,
    seasonal_cache_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    config.validate()
    validate_rolling_protocol()
    variants = variant_feature_sets()
    if config.evaluation_role != "development" or config.candidate_k != 300:
        raise ValueError("M3.3 requires development-only candidate_k=300")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    by_cutoff = {window.cutoff: window for window in windows}
    if tuple(sorted(by_cutoff)) != ALL_CUTOFFS:
        raise ValueError(f"M3.3 requires baseline candidate cutoffs {ALL_CUTOFFS}")
    m3_metrics = _read_json(m3_metrics_path.resolve())
    if m3_metrics.get("schema_version") != "m3.0-three-window-robustness-v1":
        raise ValueError("M3.3 requires M3.0 evidence")
    if m3_metrics.get("status") != "measured" or m3_metrics["contract"].get("final_week") != "not_run":
        raise ValueError("M3.3 M3.0 evidence boundary failed")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    stage_timers: dict[str, float] = {}
    try:
        m29_windows = [
            M29Window(window.cutoff, window.candidate_path, window.manifest_path)
            for window in windows
        ]
        stage_started = time.perf_counter()
        print("M3.3: validating/building ten cutoff-safe Item2Vec source caches", flush=True)
        sources = build_source_cache(
            windows=m29_windows,
            transactions_path=transactions_path,
            m28_metrics_path=m28_metrics_path,
            source_cache_dir=source_cache_dir,
            cutoffs=ALL_CUTOFFS,
        )
        stage_timers["item2vec_source_cache"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        candidates = build_expanded_candidate_cache(
            windows=m29_windows,
            sources=sources,
            cache_dir=m29_cache_dir,
            cutoffs=ALL_CUTOFFS,
        )
        stage_timers["expanded_candidate_cache"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        base_features = build_feature_cache(
            raw_dir=raw_dir,
            work_dir=work_dir,
            cache_dir=m29_cache_dir,
            candidates=candidates,
            config=config,
            cutoffs=ALL_CUTOFFS,
        )
        stage_timers["base_feature_cache"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        target_features = build_target_aware_cache(
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            m29_cache_dir=m29_cache_dir,
            cache_dir=target_feature_cache_dir,
            cutoffs=ALL_CUTOFFS,
        )
        stage_timers["target_feature_cache"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        seasonal_cache = build_seasonal_feature_cache(
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            source_cache_dir=target_feature_cache_dir,
            cache_dir=seasonal_cache_dir,
            cutoffs=ALL_CUTOFFS,
        )
        stage_timers["seasonal_feature_cache"] = time.perf_counter() - stage_started

        all_features = list(
            dict.fromkeys(feature for values in variants.values() for feature in values)
        )
        development: dict[str, Any] = {}
        for window, protocol in ROLLING_PROTOCOL.items():
            print(f"M3.3 {window}: inner fits", flush=True)
            window_dir = artifact_dir / window
            window_dir.mkdir()
            inner_train, inner_sizes, inner_sampling = load_distribution_sample(
                [_cache_paths(seasonal_cache_dir, cutoff)[0] for cutoff in protocol["inner_train"]],
                all_features,
                seed=SAMPLING_SEED,
            )
            inner_maps = build_category_maps(inner_train)
            inner_map_evidence = _save_category_maps(
                window_dir / "inner-category-maps.json", inner_maps
            )
            inner_validation, inner_validation_sizes, truth_counts, validation_evidence = load_inner_validation(
                dataset_path=_cache_paths(seasonal_cache_dir, protocol["inner_validation"])[0],
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

            print(f"M3.3 {window}: outer refits and scoring", flush=True)
            outer_train, outer_sizes, outer_sampling = load_distribution_sample(
                [_cache_paths(seasonal_cache_dir, cutoff)[0] for cutoff in protocol["outer_train"]],
                all_features,
                seed=SAMPLING_SEED,
            )
            outer_maps = build_category_maps(outer_train)
            outer_map_evidence = _save_category_maps(
                window_dir / "outer-category-maps.json", outer_maps
            )
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
                dataset_path=_cache_paths(seasonal_cache_dir, protocol["outer_validation"])[0],
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
            season_sensitive = evaluate_season_sensitive_truth(
                evaluation_db=window_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                articles_path=raw_dir / "articles.csv",
                cutoff=protocol["outer_validation"],
                variant_names=list(variants),
                metric_k=config.metric_k,
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
                "season_sensitive": season_sensitive,
            }
        summary = summarize_m33(development, m3_metrics, config.metric_k)
        if not summary["shared_anchor_reproduction_passed"]:
            raise RuntimeError("M3.3 failed exact M3.0 shared-window anchor reproduction")
        if not summary["candidate_parity_passed"]:
            raise RuntimeError("M3.3 candidate Recall/Oracle parity failed")
        artifacts = _collect_artifacts(development)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M3.3",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "winter_spring_early_summer_late_summer_rolling_development_no_final_week",
            "config": asdict(config),
            "contract": {
                "candidate_pool": "frozen six-source Top100 plus up to 200 Item2Vec-only",
                "catalog_protocol": "optimistic_all_articles",
                "variants": variants,
                "feature_families": {
                    "raw_prior": RAW_PRIOR_FEATURES,
                    "normalized_prior": NORMALIZED_PRIOR_FEATURES,
                    "adaptive_seasonal": ADAPTIVE_SEASONAL_FEATURES,
                },
                "rolling_protocol": ROLLING_PROTOCOL,
                "selection": "mean MAP improves and no outer season regresses; no post-hoc combinations",
                "season_sensitive_definition": {
                    "product_type_lift_threshold": SEASON_SENSITIVE_PRODUCT_TYPE_LIFT,
                    "minimum_prior_events": SEASON_SENSITIVE_MIN_PRIOR_EVENTS,
                },
                "final_week": "not_run",
            },
            "inputs": {
                "m3_metrics": _file_identity(m3_metrics_path),
                "m28_metrics": _file_identity(m28_metrics_path),
                "transactions": _file_identity(transactions_path),
                "baseline_windows": {
                    cutoff: {
                        "candidate": _file_identity(window.candidate_path),
                        "manifest": _file_identity(window.manifest_path),
                    }
                    for cutoff, window in by_cutoff.items()
                },
            },
            "prerequisite_cache": {
                "item2vec_sources": sources,
                "expanded_candidates": candidates,
                "base_features": base_features,
                "target_features": target_features,
            },
            "feature_cache": seasonal_cache,
            "development": development,
            "summary": summary,
            "resources": {
                "peak_working_set_bytes": _peak_working_set_bytes(),
                "artifact_count": len(artifacts),
                "artifact_bytes": int(sum(item["bytes"] for item in artifacts)),
                "artifacts": artifacts,
                "stage_timers_seconds": stage_timers,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "artifacts": {},
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M3_3_REPORT.md"
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
                "schema_version": "m3.3-failure-v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
