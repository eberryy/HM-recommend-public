from __future__ import annotations

"""Auditable M1.5 source caches and feature-ready candidate artifacts."""

import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import duckdb

from .audit import prepare_tabular_connection
from .m1 import (
    FUSION_PROFILES,
    SOURCES,
    M1Config,
    _create_age_popularity,
    _create_attribute_content,
    _create_covisit,
    _create_fused_candidates,
    _create_population,
    _create_product_family,
    _create_recent_popularity,
    _create_repurchase,
    _date_literal,
    _map_at_12,
    _path_literal,
    _protocol_metrics,
    _resolve_cutoff,
    _scalar,
    _segment_metrics,
    _source_metrics,
)


CACHE_SCHEMA = "m1.5-cache-v1"
FEATURE_SCHEMA = "m1.5-wide-v1"
SOURCE_TABLES = {
    "repurchase": "m1_repurchase",
    "recent_popularity": "m1_recent_popularity",
    "product_family": "m1_product_family",
    "user_day_covisit": "m1_user_day_covisit",
    "age_popularity": "m1_age_popularity",
    "attribute_content": "m1_attribute_content",
}
GLOBAL_TABLES = (
    "m1_warm_catalog",
    "m1_eligible_catalog",
    "m1_item_popularity",
    "m1_global_top",
    "m1_content_type_pool",
    "m1_content_garment_pool",
    "m1_covisit_neighbors",
    "m1_age_top",
)
SOURCE_PARAMETER_NAMES = {
    "repurchase": ("history_weeks", "source_k"),
    "recent_popularity": ("popularity_days", "source_k"),
    "product_family": ("history_weeks", "popularity_days", "source_k"),
    "user_day_covisit": (
        "history_weeks",
        "covisit_days",
        "source_k",
        "covisit_neighbor_k",
        "max_user_day_items",
        "min_covisit_count",
    ),
    "age_popularity": ("popularity_days", "source_k", "customer_policy"),
    "attribute_content": (
        "history_weeks",
        "popularity_days",
        "source_k",
        "content_seed_k",
        "content_type_pool_k",
        "content_garment_pool_k",
    ),
}


class CacheValidationError(RuntimeError):
    """Raised when an existing cache does not satisfy its manifest contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_key(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:24]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_identity(raw_dir: Path) -> dict[str, dict[str, Any]]:
    identity: dict[str, dict[str, Any]] = {}
    for name in ("transactions_train.csv", "articles.csv", "customers.csv"):
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        identity[name] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return identity


def _implementation_identity() -> dict[str, str]:
    here = Path(__file__).resolve()
    return {
        "m1.py": _sha256(here.with_name("m1.py")),
        "m15.py": _sha256(here),
    }


def validate_cache_manifest(
    manifest: dict[str, Any], expected: dict[str, Any], *, label: str = "cache"
) -> None:
    """Fail closed and report the exact top-level manifest mismatches."""
    mismatches = {
        key: {"actual": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise CacheValidationError(f"{label} manifest mismatch: {mismatches}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CacheValidationError(f"missing cache manifest: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CacheValidationError(f"invalid cache manifest: {path}: {error}") from error
    if not isinstance(value, dict):
        raise CacheValidationError(f"cache manifest is not an object: {path}")
    return value


def _relation_stats(
    connection: duckdb.DuckDBPyConnection, relation: str
) -> dict[str, Any]:
    columns = connection.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    stats: dict[str, Any] = {
        "rows": int(_scalar(connection, f"SELECT count(*) FROM {relation}")),
        "schema": [{"name": row[0], "type": row[1]} for row in columns],
    }
    names = {row[0] for row in columns}
    if {"customer_id", "article_id"}.issubset(names):
        stats["unique_user_item_rows"] = int(
            _scalar(
                connection,
                f"SELECT count(*) FROM (SELECT DISTINCT customer_id, article_id FROM {relation})",
            )
        )
    if "article_id" in names:
        stats["leading_zero_article_rows"] = int(
            _scalar(connection, f"SELECT count(*) FROM {relation} WHERE article_id LIKE '0%'")
        )
    return stats


def _global_expected(
    raw_identity: dict[str, Any], implementation: dict[str, str], cutoff: str,
    config: M1Config,
) -> dict[str, Any]:
    names = (
        "popularity_days", "covisit_days", "source_k", "covisit_neighbor_k",
        "max_user_day_items", "min_covisit_count", "content_type_pool_k",
        "content_garment_pool_k", "catalog_protocol", "customer_policy",
        "duplicate_policy",
    )
    values = asdict(config)
    return {
        "schema_version": CACHE_SCHEMA,
        "scope": "cutoff_global",
        "cutoff": cutoff,
        "catalog_protocol": config.catalog_protocol,
        "parameters": {name: values[name] for name in names},
        "raw_files": raw_identity,
        "implementation": implementation,
    }


def _target_expected(
    global_key: str, cutoff: str, config: M1Config, population: dict[str, Any]
) -> dict[str, Any]:
    values = asdict(config)
    excluded = {"cutoff", "fusion_profile", "evaluation_mode", "rrf_constant"}
    return {
        "schema_version": CACHE_SCHEMA,
        "scope": "target_sources",
        "global_key": global_key,
        "cutoff": cutoff,
        "catalog_protocol": config.catalog_protocol,
        "sample_rate": config.sample_rate,
        "sampled_user_fingerprint": population["sampled_user_fingerprint"],
        "parameters": {key: value for key, value in values.items() if key not in excluded},
    }


def _source_expected(target: dict[str, Any], source: str, config: M1Config) -> dict[str, Any]:
    values = asdict(config)
    return {
        **target,
        "scope": "target_source",
        "source": source,
        "source_parameters": {
            name: values[name] for name in SOURCE_PARAMETER_NAMES[source]
        },
    }


def _cache_dir_state(path: Path) -> str:
    if not path.exists():
        return "missing"
    if not path.is_dir():
        raise CacheValidationError(f"cache path is not a directory: {path}")
    return "nonempty" if any(path.iterdir()) else "empty"


def _write_global_cache(
    connection: duckdb.DuckDBPyConnection, root: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    state = _cache_dir_state(root)
    manifest_path = root / "manifest.json"
    if state != "missing":
        if state == "empty":
            raise CacheValidationError(f"refusing incomplete global cache: {root}")
        manifest = _read_json(manifest_path)
        validate_cache_manifest(manifest, expected, label="global cache")
        for table in GLOBAL_TABLES:
            if not (root / f"{table}.parquet").is_file():
                raise CacheValidationError(f"missing global cache table: {table}")
        return {"status": "hit", "path": str(root.resolve()), "manifest": manifest}
    root.mkdir(parents=True)
    tables: dict[str, Any] = {}
    for table in GLOBAL_TABLES:
        parquet = root / f"{table}.parquet"
        connection.execute(
            f"COPY {table} TO {_path_literal(parquet)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        tables[table] = {**_relation_stats(connection, table), "bytes": parquet.stat().st_size}
    manifest = {
        **expected,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tables": tables,
    }
    _write_json(manifest_path, manifest)
    return {"status": "built", "path": str(root.resolve()), "manifest": manifest}


def _canonicalize_source(connection: duckdb.DuckDBPyConnection, source: str) -> None:
    table = SOURCE_TABLES[source]
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m15_canonical AS
        SELECT customer_id::VARCHAR AS customer_id, article_id::VARCHAR AS article_id,
               source::VARCHAR AS source, source_rank::BIGINT AS source_rank,
               source_score::DOUBLE AS source_score
        FROM {table}
        QUALIFY row_number() OVER (
            PARTITION BY customer_id, article_id, source
            ORDER BY source_rank, source_score DESC NULLS LAST
        ) = 1
        """
    )
    connection.execute(f"CREATE OR REPLACE TEMP TABLE {table} AS SELECT * FROM m15_canonical")


def _write_target_sources(
    connection: duckdb.DuckDBPyConnection, root: Path, target: dict[str, Any],
    config: M1Config,
) -> dict[str, Any]:
    if _cache_dir_state(root) != "missing":
        raise CacheValidationError(f"refusing to overwrite target cache: {root}")
    root.mkdir(parents=True)
    manifests: dict[str, Any] = {}
    for source in SOURCES:
        table = SOURCE_TABLES[source]
        parquet = root / f"{source}.parquet"
        connection.execute(
            f"COPY {table} TO {_path_literal(parquet)} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        manifest = {
            **_source_expected(target, source, config),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": str(parquet.resolve()),
            "artifact_bytes": parquet.stat().st_size,
            "stats": _relation_stats(connection, table),
        }
        _write_json(root / f"{source}.manifest.json", manifest)
        manifests[source] = manifest
    _write_json(root / "manifest.json", {**target, "sources": list(SOURCES)})
    return manifests


def _load_target_sources(
    connection: duckdb.DuckDBPyConnection, root: Path, target: dict[str, Any],
    config: M1Config,
) -> dict[str, Any]:
    if _cache_dir_state(root) != "nonempty":
        raise CacheValidationError(f"target cache is unavailable: {root}")
    validate_cache_manifest(_read_json(root / "manifest.json"), target, label="target cache")
    manifests: dict[str, Any] = {}
    for source in SOURCES:
        parquet = root / f"{source}.parquet"
        if not parquet.is_file():
            raise CacheValidationError(f"missing source cache parquet: {parquet}")
        manifest = _read_json(root / f"{source}.manifest.json")
        validate_cache_manifest(
            manifest, _source_expected(target, source, config), label=f"{source} cache"
        )
        table = SOURCE_TABLES[source]
        connection.execute(
            f"CREATE OR REPLACE TEMP TABLE {table} AS SELECT * FROM read_parquet({_path_literal(parquet)})"
        )
        actual = _relation_stats(connection, table)
        if actual != manifest.get("stats"):
            raise CacheValidationError(
                f"{source} cache data/stat mismatch: actual={actual}, expected={manifest.get('stats')}"
            )
        manifests[source] = manifest
    return manifests


def _generate_sources(
    connection: duckdb.DuckDBPyConnection, cutoff_sql: str, config: M1Config,
    timings: dict[str, float],
) -> dict[str, int]:
    builders: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("repurchase", lambda: _create_repurchase(connection, cutoff_sql, config)),
        ("recent_popularity", lambda: _create_recent_popularity(connection, cutoff_sql, config)),
        ("product_family", lambda: _create_product_family(connection, cutoff_sql, config)),
        ("attribute_content", lambda: _create_attribute_content(connection, cutoff_sql, config)),
        ("user_day_covisit", lambda: _create_covisit(connection, cutoff_sql, config)),
        ("age_popularity", lambda: _create_age_popularity(connection, cutoff_sql, config)),
    )
    covisit_build: dict[str, int] = {}
    for source, builder in builders:
        started = time.perf_counter()
        value = builder()
        _canonicalize_source(connection, source)
        timings[source] = time.perf_counter() - started
        if source == "user_day_covisit":
            covisit_build = dict(value)
    return covisit_build


def _load_sources_with_timings(
    connection: duckdb.DuckDBPyConnection, root: Path, target: dict[str, Any],
    config: M1Config, timings: dict[str, float],
) -> dict[str, Any]:
    started = time.perf_counter()
    manifests = _load_target_sources(connection, root, target, config)
    elapsed = time.perf_counter() - started
    rows = sum(max(1, int(manifests[source]["stats"]["rows"])) for source in SOURCES)
    for source in SOURCES:
        timings[source] = elapsed * max(1, int(manifests[source]["stats"]["rows"])) / rows
    return manifests


def _create_wide_candidates(
    connection: duckdb.DuckDBPyConnection, config: M1Config
) -> None:
    fields: list[str] = []
    weights = FUSION_PROFILES[config.fusion_profile]
    for source in SOURCES:
        condition = f"long.source = '{source}'"
        fields.extend(
            [
                f"CASE WHEN count(*) FILTER (WHERE {condition}) > 0 THEN 1 ELSE 0 END AS {source}_present",
                f"max(long.source_rank) FILTER (WHERE {condition}) AS {source}_rank",
                f"max(long.source_score) FILTER (WHERE {condition}) AS {source}_score",
                (
                    f"max({weights[source]:.12f} / ({config.rrf_constant} + long.source_rank)) "
                    f"FILTER (WHERE {condition}) AS {source}_rrf_contribution"
                ),
            ]
        )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m15_wide_candidates AS
        SELECT candidates.customer_id::VARCHAR AS customer_id,
               candidates.article_id::VARCHAR AS article_id,
               candidates.candidate_rank::BIGINT AS candidate_rank,
               candidates.fused_score::DOUBLE AS fused_score,
               candidates.source_count::BIGINT AS source_count,
               candidates.sources::VARCHAR AS sources,
               {", ".join(fields)}
        FROM m1_candidates candidates
        LEFT JOIN m1_candidates_long long USING (customer_id, article_id)
        GROUP BY candidates.customer_id, candidates.article_id, candidates.candidate_rank,
                 candidates.fused_score, candidates.source_count, candidates.sources
        """
    )


def _candidate_evidence(
    connection: duckdb.DuckDBPyConnection, relation: str, candidate_k: int
) -> dict[str, Any]:
    stats = _relation_stats(connection, relation)
    row = connection.execute(
        f"""
        SELECT count(DISTINCT customer_id), min(candidate_rank), max(candidate_rank),
               count(*) - count(DISTINCT (customer_id, article_id)),
               count(*) FILTER (WHERE candidate_rank < 1 OR candidate_rank > {candidate_k})
        FROM {relation}
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("candidate evidence query returned no rows")
    return {
        **stats,
        "users": int(row[0]),
        "min_candidate_rank": int(row[1]),
        "max_candidate_rank": int(row[2]),
        "duplicate_user_item_rows": int(row[3]),
        "out_of_range_rank_rows": int(row[4]),
    }


def _wide_evidence(
    connection: duckdb.DuckDBPyConnection, config: M1Config
) -> dict[str, Any]:
    result = _candidate_evidence(connection, "m15_wide_candidates", config.candidate_k)
    null_violations: dict[str, int] = {}
    for source in SOURCES:
        null_violations[source] = int(
            _scalar(
                connection,
                f"""
                SELECT count(*) FROM m15_wide_candidates
                WHERE ({source}_present = 0 AND (
                           {source}_rank IS NOT NULL OR {source}_score IS NOT NULL
                           OR {source}_rrf_contribution IS NOT NULL))
                   OR ({source}_present = 1 AND (
                           {source}_rank IS NULL OR {source}_score IS NULL
                           OR {source}_rrf_contribution IS NULL))
                """,
            )
        )
    result["source_null_semantics_violations"] = null_violations
    result["multi_source_rows"] = int(
        _scalar(connection, "SELECT count(*) FROM m15_wide_candidates WHERE source_count > 1")
    )
    return result


def _candidate_parity(
    connection: duckdb.DuckDBPyConnection, reference: Path | None
) -> dict[str, Any]:
    if reference is None:
        return {"status": "not_requested"}
    if not reference.is_file():
        raise FileNotFoundError(reference)
    reference_sql = f"read_parquet({_path_literal(reference)})"
    missing = int(
        _scalar(
            connection,
            f"""
            SELECT count(*) FROM (
                SELECT customer_id::VARCHAR, article_id::VARCHAR, candidate_rank::BIGINT
                FROM {reference_sql}
                EXCEPT ALL
                SELECT customer_id, article_id, candidate_rank FROM m1_candidates
            )
            """,
        )
    )
    extra = int(
        _scalar(
            connection,
            f"""
            SELECT count(*) FROM (
                SELECT customer_id, article_id, candidate_rank FROM m1_candidates
                EXCEPT ALL
                SELECT customer_id::VARCHAR, article_id::VARCHAR, candidate_rank::BIGINT
                FROM {reference_sql}
            )
            """,
        )
    )
    result = {
        "status": "passed" if missing == 0 and extra == 0 else "failed",
        "reference": str(reference.resolve()),
        "missing_pair_rank_rows": missing,
        "extra_pair_rank_rows": extra,
    }
    if result["status"] != "passed":
        raise RuntimeError(f"candidate pair/rank parity failed: {result}")
    return result


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_numeric_leaves(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        leaves[prefix] = float(value)
    return leaves


def _metrics_parity(
    actual: dict[str, Any], baseline: Path | None, tolerance: float = 1e-12
) -> dict[str, Any]:
    if baseline is None:
        return {"status": "not_requested"}
    if not baseline.is_file():
        raise FileNotFoundError(baseline)
    expected_result = json.loads(baseline.read_text(encoding="utf-8"))
    expected = _numeric_leaves(expected_result.get("metrics", {}))
    observed = _numeric_leaves(actual)
    if set(expected) != set(observed):
        raise RuntimeError(
            "metric parity schema mismatch: "
            f"missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    differences = {key: abs(observed[key] - expected[key]) for key in expected}
    maximum = max(differences.values(), default=0.0)
    result = {
        "status": "passed" if maximum <= tolerance else "failed",
        "reference": str(baseline.resolve()),
        "tolerance": tolerance,
        "maximum_absolute_difference": maximum,
    }
    if result["status"] != "passed":
        worst = max(differences, key=differences.get)
        raise RuntimeError(f"metric parity failed at {worst}: {result}")
    return result


def _render_report(result: dict[str, Any]) -> str:
    cache = result["cache"]
    parity = result["parity"]
    return (
        "# M1.5 Cache and Feature Artifact Report\n\n"
        "## Boundary\n\n"
        "- Retrieval semantics: unchanged M1 retrieval-v1.\n"
        "- Modeling and image embedding: not started.\n\n"
        "## Result\n\n"
        f"- cutoff: `{result['cutoff']}`\n"
        f"- protocol: `{result['catalog_protocol']}`\n"
        f"- users / candidate rows: {result['validation_users']:,} / {result['candidate_rows']:,}\n"
        f"- global cache: `{cache['global']['status']}`\n"
        f"- target source cache: `{cache['target']['status']}`\n"
        f"- candidate parity: `{parity['candidates']['status']}`\n"
        f"- metric parity: `{parity['metrics']['status']}`\n"
        f"- elapsed seconds: {result['elapsed_seconds']:.2f}\n\n"
        "## Special treatment\n\n"
        "- Global snapshots are audited but do not bypass target generation on a target-cache miss.\n"
        "- A complete target-cache hit skips all six source generators and rebuilds fusion/features.\n"
        "- Duplicate source rows use minimum rank, then maximum score.\n"
        "- Missing source features remain null; rank zero is never used as a sentinel.\n"
    )


def _create_candidates_long(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    union_sql = " UNION ALL ".join(
        f"SELECT * FROM {SOURCE_TABLES[source]}" for source in SOURCES
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_candidates_long AS
        SELECT candidates.*
        FROM ({union_sql}) candidates
        SEMI JOIN m1_eligible_catalog eligible USING (article_id)
        """
    )


def _create_collaborative_fusion(
    connection: duckdb.DuckDBPyConnection, config: M1Config
) -> None:
    weights = FUSION_PROFILES[config.fusion_profile]
    cases = " ".join(
        f"WHEN '{source}' THEN {weights[source]:.6f}" for source in SOURCES
    )
    weight_sql = f"CASE source {cases} END"
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE m1_candidates_rrf_{config.fusion_profile} AS
        WITH fused AS (
            SELECT customer_id, article_id,
                   sum(({weight_sql}) / ({config.rrf_constant} + source_rank)) AS fused_score,
                   count(DISTINCT source) AS source_count,
                   string_agg(source, ',' ORDER BY source) AS sources
            FROM m1_candidates_long
            GROUP BY customer_id, article_id
        )
        SELECT customer_id, article_id,
               row_number() OVER (
                   PARTITION BY customer_id
                   ORDER BY fused_score DESC, source_count DESC, article_id
               ) AS candidate_rank,
               fused_score, source_count, sources
        FROM fused
        QUALIFY candidate_rank <= {config.candidate_k}
        """
    )
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW m1_candidates AS "
        f"SELECT * FROM m1_candidates_rrf_{config.fusion_profile}"
    )


def _fusion_expected(
    target_key: str, config: M1Config
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA,
        "scope": "fusion_reference",
        "target_key": target_key,
        "fusion_profile": config.fusion_profile,
        "rrf_constant": config.rrf_constant,
        "candidate_k": config.candidate_k,
    }


def _stabilize_fusion_from_reference(
    connection: duckdb.DuckDBPyConnection,
    root: Path,
    expected: dict[str, Any],
    config: M1Config,
    *,
    build: bool,
) -> dict[str, Any]:
    state = _cache_dir_state(root)
    parquet = root / "candidates.parquet"
    manifest_path = root / "manifest.json"
    if build:
        if state != "missing":
            raise CacheValidationError(f"refusing to overwrite fusion reference: {root}")
        root.mkdir(parents=True)
        connection.execute(
            f"COPY m1_candidates TO {_path_literal(parquet)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        manifest = {
            **expected,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": str(parquet.resolve()),
            "artifact_bytes": parquet.stat().st_size,
            "stats": _relation_stats(connection, "m1_candidates"),
        }
        _write_json(manifest_path, manifest)
        return {
            "status": "built",
            "path": str(root.resolve()),
            "candidate_pair_missing": 0,
            "candidate_pair_extra": 0,
            "stabilized_rank_rows": 0,
            "maximum_fused_score_difference": 0.0,
        }
    if state != "nonempty":
        raise CacheValidationError(f"required fusion reference is unavailable: {root}")
    if not parquet.is_file():
        raise CacheValidationError(f"missing fusion reference parquet: {parquet}")
    manifest = _read_json(manifest_path)
    validate_cache_manifest(manifest, expected, label="fusion reference")
    connection.execute(
        f"CREATE OR REPLACE TEMP TABLE m15_fusion_reference AS "
        f"SELECT * FROM read_parquet({_path_literal(parquet)})"
    )
    actual_stats = _relation_stats(connection, "m15_fusion_reference")
    if actual_stats != manifest.get("stats"):
        raise CacheValidationError(
            "fusion reference data/stat mismatch: "
            f"actual={actual_stats}, expected={manifest.get('stats')}"
        )
    missing = int(
        _scalar(
            connection,
            """
            SELECT count(*) FROM (
                SELECT customer_id, article_id FROM m15_fusion_reference
                EXCEPT ALL
                SELECT customer_id, article_id FROM m1_candidates
            )
            """,
        )
    )
    extra = int(
        _scalar(
            connection,
            """
            SELECT count(*) FROM (
                SELECT customer_id, article_id FROM m1_candidates
                EXCEPT ALL
                SELECT customer_id, article_id FROM m15_fusion_reference
            )
            """,
        )
    )
    if missing or extra:
        raise CacheValidationError(
            "recomputed fusion candidate set differs from reference: "
            f"missing={missing}, extra={extra}"
        )
    row = connection.execute(
        """
        SELECT count(*) FILTER (
                   WHERE current.candidate_rank != reference.candidate_rank
               ),
               max(abs(current.fused_score - reference.fused_score))
        FROM m1_candidates current
        JOIN m15_fusion_reference reference USING (customer_id, article_id)
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("fusion stabilization audit returned no row")
    stabilized_rank_rows = int(row[0])
    maximum_score_difference = float(row[1] or 0.0)
    connection.execute(
        f"CREATE OR REPLACE TEMP TABLE m1_candidates_rrf_{config.fusion_profile} AS "
        "SELECT * FROM m15_fusion_reference"
    )
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW m1_candidates AS "
        f"SELECT * FROM m1_candidates_rrf_{config.fusion_profile}"
    )
    return {
        "status": "hit",
        "path": str(root.resolve()),
        "candidate_pair_missing": missing,
        "candidate_pair_extra": extra,
        "stabilized_rank_rows": stabilized_rank_rows,
        "maximum_fused_score_difference": maximum_score_difference,
    }

def run_m15(
    raw_dir: Path,
    work_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    cache_dir: Path,
    config: M1Config,
    *,
    cache_mode: str = "auto",
    parity_reference: Path | None = None,
    baseline_metrics: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build or reuse six source caches, then emit exact M1 candidates and wide features."""
    config.validate()
    if cache_mode not in {"auto", "require"}:
        raise ValueError("cache_mode must be auto or require")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact directory: {artifact_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    raw_identity = _raw_identity(raw_dir)
    implementation = _implementation_identity()
    connection = prepare_tabular_connection(raw_dir, work_dir)
    result: dict[str, Any]
    try:
        stage_started = time.perf_counter()
        cutoff = _resolve_cutoff(connection, config.cutoff)
        cutoff_sql = _date_literal(cutoff)
        population = _create_population(connection, cutoff, config)
        timings["population_catalog_truth"] = time.perf_counter() - stage_started
        if population["sampled_validation_users"] == 0:
            raise ValueError("sample produced zero validation users")

        global_expected = _global_expected(raw_identity, implementation, cutoff, config)
        global_key = _content_key(global_expected)
        global_root = cache_dir / "global" / global_key
        target_expected = _target_expected(global_key, cutoff, config, population)
        target_key = _content_key(target_expected)
        target_root = cache_dir / "target" / target_key / "sources"
        target_state = _cache_dir_state(target_root)
        covisit_build: dict[str, int] = {}

        if target_state == "nonempty":
            source_manifests = _load_sources_with_timings(
                connection, target_root, target_expected, config, timings
            )
            target_status = "hit"
            if _cache_dir_state(global_root) != "nonempty":
                raise CacheValidationError(
                    f"target cache exists but bound global cache is unavailable: {global_root}"
                )
        elif target_state == "empty":
            raise CacheValidationError(f"refusing incomplete target cache: {target_root}")
        elif cache_mode == "require":
            raise CacheValidationError(f"required target cache is unavailable: {target_root}")
        else:
            covisit_build = _generate_sources(
                connection, cutoff_sql, config, timings
            )
            source_manifests = _write_target_sources(
                connection, target_root, target_expected, config
            )
            target_status = "built"

        global_cache = _write_global_cache(connection, global_root, global_expected)

        stage_started = time.perf_counter()
        _create_candidates_long(connection)
        timings["source_union_catalog_filter"] = time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        _create_collaborative_fusion(connection, config)
        fusion_expected = _fusion_expected(target_key, config)
        fusion_key = _content_key(fusion_expected)
        fusion_root = target_root.parent / "fusion" / fusion_key
        fusion_cache = _stabilize_fusion_from_reference(
            connection,
            fusion_root,
            fusion_expected,
            config,
            build=target_status == "built",
        )
        timings["collaborative_fusion"] = time.perf_counter() - stage_started

        protocol_metrics = _protocol_metrics(
            connection,
            "SELECT * FROM m1_candidates",
            "candidate_rank",
            config.candidate_k,
            config.catalog_protocol,
        )
        primary_truth = "m1_truth_warm" if config.catalog_protocol == "strict" else "m1_truth"
        primary_bucket = "warm" if config.catalog_protocol == "strict" else "overall"
        final_metrics = dict(protocol_metrics[primary_bucket] or {})
        final_metrics[f"map@12_untuned_{config.fusion_profile}_rrf"] = _map_at_12(
            connection, truth_table=primary_truth
        )

        stage_started = time.perf_counter()
        _create_wide_candidates(connection, config)
        timings["wide_feature_build"] = time.perf_counter() - stage_started

        candidate_path = artifact_dir / "candidates.parquet"
        wide_path = artifact_dir / "candidate_features.parquet"
        stage_started = time.perf_counter()
        connection.execute(
            f"COPY m1_candidates TO {_path_literal(candidate_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        timings["candidate_parquet_write"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        connection.execute(
            f"COPY m15_wide_candidates TO {_path_literal(wide_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        timings["wide_parquet_write"] = time.perf_counter() - stage_started

        candidate_evidence = _candidate_evidence(
            connection, "m1_candidates", config.candidate_k
        )
        wide_evidence = _wide_evidence(connection, config)
        parity = {
            "candidates": _candidate_parity(connection, parity_reference),
            "metrics": _metrics_parity(protocol_metrics, baseline_metrics),
        }
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
                        SELECT count(*) FROM m1_candidates_long candidates
                        ANTI JOIN m1_warm_catalog warm USING (article_id)
                        WHERE source = '{source}'
                        """,
                    )
                ),
            }
            for source in SOURCES
        }
        timings["total"] = time.perf_counter() - started
        result = {
            "stage": "M1.5",
            "status": "measured",
            "run_id": run_id,
            "feature_schema": FEATURE_SCHEMA,
            "catalog_protocol": config.catalog_protocol,
            "cutoff": cutoff,
            "validation_users": population["sampled_validation_users"],
            "candidate_rows": candidate_evidence["rows"],
            "config": asdict(config),
            "population": population,
            "truth_pairs": {
                "overall": population["sampled_truth_pairs"],
                "warm": population["warm_truth_pairs"],
                "cold": population["cold_truth_pairs"],
            },
            "metrics": protocol_metrics,
            "final_metrics": final_metrics,
            "source_metrics": _source_metrics(connection, config),
            "segments": _segment_metrics(connection, config),
            "covisit_build": covisit_build,
            "cache": {
                "global": {
                    "status": global_cache["status"],
                    "key": global_key,
                    "path": global_cache["path"],
                },
                "target": {
                    "status": target_status,
                    "key": target_key,
                    "path": str(target_root.resolve()),
                    "source_rows": {
                        source: source_manifests[source]["stats"]["rows"] for source in SOURCES
                    },
                },
                "fusion": {
                    **fusion_cache,
                    "key": fusion_key,
                },
                "mode": cache_mode,
                "global_cache_reused_to_skip_target_generation": False,
                "target_cache_skipped_source_generation": target_status == "hit",
            },
            "timings_seconds": timings,
            "parity": parity,
            "candidate_evidence": candidate_evidence,
            "wide_evidence": wide_evidence,
            "leakage_audit": {
                "behavior_statistics_filter": "t_dat < cutoff",
                "latest_behavior_date": str(
                    _scalar(
                        connection,
                        f"SELECT max(t_dat) FROM transactions WHERE t_dat < {cutoff_sql}",
                    )
                ),
                "validation_used_for_catalog": False,
                "source_catalog_audit": source_catalog_audit,
            },
            "artifacts": {
                "candidates": str(candidate_path.resolve()),
                "candidate_features": str(wide_path.resolve()),
                "candidate_bytes": candidate_path.stat().st_size,
                "candidate_features_bytes": wide_path.stat().st_size,
            },
            "elapsed_seconds": timings["total"],
        }
    finally:
        connection.close()

    artifact_manifest = {
        "schema_version": FEATURE_SCHEMA,
        "run_id": run_id,
        "cutoff": result["cutoff"],
        "catalog_protocol": result["catalog_protocol"],
        "sample_rate": config.sample_rate,
        "sampled_user_fingerprint": result["population"]["sampled_user_fingerprint"],
        "config": asdict(config),
        "raw_files": raw_identity,
        "implementation": implementation,
        "cache": result["cache"],
        "candidate_evidence": result["candidate_evidence"],
        "wide_evidence": result["wide_evidence"],
        "artifacts": result["artifacts"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(artifact_dir / "manifest.json", artifact_manifest)
    _write_json(output_dir / "metrics.json", result)
    (output_dir / "M1_5_REPORT.md").write_text(_render_report(result), encoding="utf-8")
    return result
