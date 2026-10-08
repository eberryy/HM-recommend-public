from __future__ import annotations

"""M1.6 operator profiling and parity-gated retrieval cost experiments."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .m1 import FUSION_PROFILES, SOURCES


PROFILE_SCHEMA = "m1.6-profile-v1"
SOURCE_SCHEMA = "m1.5-cache-v1"


def _literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"profile_file_created": False}
    value = _read_json(path)
    keys = (
        "latency",
        "cpu_time",
        "rows_returned",
        "cumulative_rows_scanned",
        "cumulative_cardinality",
        "system_peak_buffer_memory",
        "system_peak_temp_dir_size",
    )
    return {
        "profile_file_created": True,
        "profile_path": str(path.resolve()),
        **{key: value.get(key) for key in keys},
    }


def _profiled_execute(
    connection: duckdb.DuckDBPyConnection,
    label: str,
    sql: str,
    profile_dir: Path,
) -> dict[str, Any]:
    profile_path = profile_dir / f"{label}.json"
    connection.execute("PRAGMA enable_profiling='json'")
    connection.execute(f"PRAGMA profiling_output={_literal(profile_path)}")
    started = time.perf_counter()
    try:
        connection.execute(sql)
    finally:
        elapsed = time.perf_counter() - started
        connection.execute("PRAGMA disable_profiling")
    return {
        "label": label,
        "wall_seconds": elapsed,
        **_profile_summary(profile_path),
    }


def _query_scalar(connection: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = connection.execute(sql).fetchone()
    if row is None:
        raise RuntimeError("scalar query returned no row")
    return int(row[0])


def _source_manifests(source_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    root = _read_json(source_dir / "manifest.json")
    if root.get("schema_version") != SOURCE_SCHEMA or root.get("scope") != "target_sources":
        raise ValueError(f"unsupported target cache manifest: {source_dir / 'manifest.json'}")
    manifests: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        parquet = source_dir / f"{source}.parquet"
        manifest_path = source_dir / f"{source}.manifest.json"
        if not parquet.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"missing {source} cache files in {source_dir}")
        manifest = _read_json(manifest_path)
        stats = manifest.get("stats") or {}
        if manifest.get("source") != source or manifest.get("schema_version") != SOURCE_SCHEMA:
            raise ValueError(f"invalid {source} manifest identity")
        if int(stats.get("rows", -1)) != int(stats.get("unique_user_item_rows", -2)):
            raise ValueError(f"{source} is not unique by user-item; direct joins are unsafe")
        if int(manifest.get("artifact_bytes", -1)) != parquet.stat().st_size:
            raise ValueError(f"{source} artifact size differs from manifest")
        manifests[source] = manifest
    return root, manifests


def _configure(
    connection: duckdb.DuckDBPyConnection,
    threads: int,
    memory_limit: str,
    temp_dir: Path,
) -> dict[str, Any]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET threads={threads}")
    connection.execute(f"SET memory_limit={_literal(memory_limit)}")
    connection.execute(f"SET temp_directory={_literal(temp_dir)}")
    row = connection.execute(
        "SELECT current_setting('threads'), current_setting('memory_limit'), "
        "current_setting('temp_directory')"
    ).fetchone()
    if row is None:
        raise RuntimeError("DuckDB settings query returned no row")
    return {"threads": int(row[0]), "memory_limit": str(row[1]), "temp_directory": str(row[2])}


def _create_source_views(connection: duckdb.DuckDBPyConnection, source_dir: Path) -> None:
    for source in SOURCES:
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW m16_src_{source} AS "
            f"SELECT customer_id::VARCHAR customer_id, article_id::VARCHAR article_id, "
            f"\"source\"::VARCHAR AS \"source\", source_rank::BIGINT source_rank, "
            f"source_score::DOUBLE source_score "
            f"FROM read_parquet({_literal(source_dir / f'{source}.parquet')})"
        )


def _long_sql(eligible_catalog: Path) -> str:
    union = " UNION ALL ".join(f"SELECT * FROM m16_src_{source}" for source in SOURCES)
    return f"""
        CREATE OR REPLACE TEMP TABLE m16_long AS
        SELECT candidates.*
        FROM ({union}) candidates
        SEMI JOIN read_parquet({_literal(eligible_catalog)}) eligible USING (article_id)
    """


def _weight_case(profile: str) -> str:
    weights = FUSION_PROFILES[profile]
    cases = " ".join(f"WHEN '{source}' THEN {weights[source]:.6f}" for source in SOURCES)
    return f"CASE source {cases} END"


def _baseline_fusion_sql(profile: str, rrf_constant: int, candidate_k: int) -> str:
    weight = _weight_case(profile)
    return f"""
        CREATE OR REPLACE TEMP TABLE m16_baseline_candidates AS
        WITH fused AS (
            SELECT customer_id, article_id,
                   sum(({weight}) / ({rrf_constant} + source_rank)) AS fused_score,
                   count(DISTINCT source) AS source_count,
                   string_agg(source, ',' ORDER BY source) AS sources
            FROM m16_long
            GROUP BY customer_id, article_id
        )
        SELECT customer_id, article_id,
               row_number() OVER (
                   PARTITION BY customer_id
                   ORDER BY fused_score DESC, source_count DESC, article_id
               ) AS candidate_rank,
               fused_score, source_count, sources
        FROM fused
        QUALIFY candidate_rank <= {candidate_k}
    """


def _wide_aggregate_fields(profile: str, rrf_constant: int, prefix: str = "long") -> list[str]:
    weights = FUSION_PROFILES[profile]
    fields: list[str] = []
    for source in SOURCES:
        condition = f"{prefix}.source = '{source}'"
        fields.extend(
            [
                f"CASE WHEN count(*) FILTER (WHERE {condition}) > 0 THEN 1 ELSE 0 END AS {source}_present",
                f"max({prefix}.source_rank) FILTER (WHERE {condition}) AS {source}_rank",
                f"max({prefix}.source_score) FILTER (WHERE {condition}) AS {source}_score",
                f"max({weights[source]:.12f} / ({rrf_constant} + {prefix}.source_rank)) "
                f"FILTER (WHERE {condition}) AS {source}_rrf_contribution",
            ]
        )
    return fields


def _reference_groupby_wide_sql(profile: str, rrf_constant: int) -> str:
    fields = _wide_aggregate_fields(profile, rrf_constant)
    return f"""
        CREATE OR REPLACE TEMP TABLE m16_reference_groupby_wide AS
        SELECT candidates.customer_id, candidates.article_id, candidates.candidate_rank,
               candidates.fused_score, candidates.source_count, candidates.sources,
               {', '.join(fields)}
        FROM m16_reference candidates
        LEFT JOIN m16_long long USING (customer_id, article_id)
        GROUP BY candidates.customer_id, candidates.article_id, candidates.candidate_rank,
                 candidates.fused_score, candidates.source_count, candidates.sources
    """


def _pivot_sql() -> str:
    fields: list[str] = []
    for source in SOURCES:
        fields.extend(
            [
                f"max(source_rank) FILTER (WHERE source = '{source}') AS {source}_rank",
                f"max(source_score) FILTER (WHERE source = '{source}') AS {source}_score",
            ]
        )
    return f"""
        CREATE OR REPLACE TEMP TABLE m16_pivot AS
        SELECT customer_id, article_id, {', '.join(fields)}
        FROM m16_long
        GROUP BY customer_id, article_id
    """


def _pivot_fusion_sql(profile: str, rrf_constant: int, candidate_k: int) -> str:
    weights = FUSION_PROFILES[profile]
    contributions = [
        f"coalesce({weights[source]:.12f} / ({rrf_constant} + {source}_rank), 0.0)"
        for source in SOURCES
    ]
    present = [f"CASE WHEN {source}_rank IS NULL THEN 0 ELSE 1 END" for source in SOURCES]
    alpha = sorted(SOURCES)
    source_names = [
        f"CASE WHEN {source}_rank IS NOT NULL THEN '{source}' ELSE NULL END" for source in alpha
    ]
    return f"""
        CREATE OR REPLACE TEMP TABLE m16_pivot_candidates AS
        WITH scored AS (
            SELECT customer_id, article_id,
                   {' + '.join(contributions)} AS fused_score,
                   {' + '.join(present)} AS source_count,
                   concat_ws(',', {', '.join(source_names)}) AS sources
            FROM m16_pivot
        )
        SELECT customer_id, article_id,
               row_number() OVER (
                   PARTITION BY customer_id
                   ORDER BY fused_score DESC, source_count DESC, article_id
               ) AS candidate_rank,
               fused_score, source_count, sources
        FROM scored
        QUALIFY candidate_rank <= {candidate_k}
    """


def _direct_wide_sql(profile: str, rrf_constant: int) -> str:
    weights = FUSION_PROFILES[profile]
    aliases = {source: f"s{index}" for index, source in enumerate(SOURCES)}
    fields: list[str] = []
    joins: list[str] = []
    for source in SOURCES:
        alias = aliases[source]
        fields.extend(
            [
                f"CASE WHEN {alias}.article_id IS NULL THEN 0 ELSE 1 END AS {source}_present",
                f"{alias}.source_rank AS {source}_rank",
                f"{alias}.source_score AS {source}_score",
                f"CASE WHEN {alias}.source_rank IS NULL THEN NULL ELSE "
                f"{weights[source]:.12f} / ({rrf_constant} + {alias}.source_rank) END "
                f"AS {source}_rrf_contribution",
            ]
        )
        joins.append(
            f"LEFT JOIN m16_src_{source} {alias} "
            "ON candidates.customer_id = " + alias + ".customer_id "
            "AND candidates.article_id = " + alias + ".article_id"
        )
    return f"""
        CREATE OR REPLACE TEMP TABLE m16_direct_wide AS
        SELECT candidates.customer_id, candidates.article_id, candidates.candidate_rank,
               candidates.fused_score, candidates.source_count, candidates.sources,
               {', '.join(fields)}
        FROM m16_reference candidates
        {' '.join(joins)}
    """


def _candidate_parity(connection: duckdb.DuckDBPyConnection, relation: str) -> dict[str, Any]:
    missing_pairs = _query_scalar(
        connection,
        f"SELECT count(*) FROM (SELECT customer_id, article_id FROM m16_reference "
        f"EXCEPT SELECT customer_id, article_id FROM {relation})",
    )
    extra_pairs = _query_scalar(
        connection,
        f"SELECT count(*) FROM (SELECT customer_id, article_id FROM {relation} "
        "EXCEPT SELECT customer_id, article_id FROM m16_reference)",
    )
    missing_pair_rank = _query_scalar(
        connection,
        f"SELECT count(*) FROM (SELECT customer_id, article_id, candidate_rank FROM m16_reference "
        f"EXCEPT SELECT customer_id, article_id, candidate_rank FROM {relation})",
    )
    extra_pair_rank = _query_scalar(
        connection,
        f"SELECT count(*) FROM (SELECT customer_id, article_id, candidate_rank FROM {relation} "
        "EXCEPT SELECT customer_id, article_id, candidate_rank FROM m16_reference)",
    )
    row = connection.execute(
        f"SELECT max(abs(candidate.fused_score - reference.fused_score)) "
        f"FROM {relation} candidate JOIN m16_reference reference USING (customer_id, article_id)"
    ).fetchone()
    return {
        "missing_pairs": missing_pairs,
        "extra_pairs": extra_pairs,
        "missing_pair_rank_rows": missing_pair_rank,
        "extra_pair_rank_rows": extra_pair_rank,
        "maximum_fused_score_difference": None if row is None or row[0] is None else float(row[0]),
    }


def _wide_parity(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    feature_columns: list[str] = []
    for source in SOURCES:
        feature_columns.extend(
            [
                f"{source}_present",
                f"{source}_rank",
                f"{source}_score",
                f"{source}_rrf_contribution",
            ]
        )
    predicates = " OR ".join(
        f"direct.{column} IS DISTINCT FROM grouped.{column}" for column in feature_columns
    )
    mismatches = _query_scalar(
        connection,
        f"""
        SELECT count(*)
        FROM m16_direct_wide direct
        FULL OUTER JOIN m16_reference_groupby_wide grouped USING (customer_id, article_id)
        WHERE direct.customer_id IS NULL OR grouped.customer_id IS NULL OR {predicates}
        """,
    )
    direct_rows = _query_scalar(connection, "SELECT count(*) FROM m16_direct_wide")
    direct_unique = _query_scalar(
        connection,
        "SELECT count(DISTINCT (customer_id, article_id)) FROM m16_direct_wide",
    )
    return {
        "direct_rows": direct_rows,
        "direct_unique_user_item_rows": direct_unique,
        "feature_mismatch_rows": mismatches,
    }


def _external_wide_parity_sql(reference_wide: Path) -> str:
    return f"""
    CREATE OR REPLACE TEMP TABLE m16_external_wide_parity AS
    SELECT
        (SELECT count(*) FROM (
            SELECT * FROM m16_direct_wide
            EXCEPT ALL
            SELECT * FROM read_parquet({_literal(reference_wide)})
        )) AS actual_minus_reference_rows,
        (SELECT count(*) FROM (
            SELECT * FROM read_parquet({_literal(reference_wide)})
            EXCEPT ALL
            SELECT * FROM m16_direct_wide
        )) AS reference_minus_actual_rows
    """


def _external_wide_parity(connection: duckdb.DuckDBPyConnection) -> dict[str, int]:
    row = connection.execute(
        "SELECT actual_minus_reference_rows, reference_minus_actual_rows "
        "FROM m16_external_wide_parity"
    ).fetchone()
    if row is None:
        raise RuntimeError("external wide parity returned no row")
    return {
        "actual_minus_reference_rows": int(row[0]),
        "reference_minus_actual_rows": int(row[1]),
    }


def _validate_reference_manifest(
    reference_candidates: Path,
    reference_manifest: Path,
    target_key: str,
    fusion_profile: str,
    rrf_constant: int,
    candidate_k: int,
) -> dict[str, Any]:
    manifest = _read_json(reference_manifest)
    expected = {
        "schema_version": SOURCE_SCHEMA,
        "scope": "fusion_reference",
        "target_key": target_key,
        "fusion_profile": fusion_profile,
        "rrf_constant": rrf_constant,
        "candidate_k": candidate_k,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"reference manifest mismatch for {key}")
    artifact = Path(str(manifest.get("artifact", ""))).resolve()
    if artifact != reference_candidates.resolve():
        raise ValueError("reference manifest does not point to candidate artifact")
    if int(manifest.get("artifact_bytes", -1)) != reference_candidates.stat().st_size:
        raise ValueError("reference candidate size differs from manifest")
    stats = manifest.get("stats") or {}
    if int(stats.get("rows", -1)) != int(stats.get("unique_user_item_rows", -2)):
        raise ValueError("reference manifest is not unique by user-item")
    return manifest


def run_profile(
    source_dir: Path,
    reference_candidates: Path,
    output_dir: Path,
    artifact_dir: Path,
    *,
    reference_wide: Path | None = None,
    reference_manifest: Path | None = None,
    scope: str = "all",
    fusion_profile: str = "collaborative",
    rrf_constant: int = 60,
    candidate_k: int = 100,
    threads: int = 8,
    memory_limit: str = "12GB",
) -> dict[str, Any]:
    if scope not in {"all", "hot"}:
        raise ValueError("scope must be all or hot")
    if scope == "hot" and reference_manifest is None:
        raise ValueError("hot scope requires a fusion reference manifest")
    if fusion_profile not in FUSION_PROFILES:
        raise ValueError(f"unknown fusion profile: {fusion_profile}")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError("M1.6 output and artifact directories must be new")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    profile_dir = artifact_dir / "profiles"
    profile_dir.mkdir()
    temp_dir = artifact_dir / "duckdb-spill"

    root_manifest, source_manifests = _source_manifests(source_dir)
    global_key = str(root_manifest["global_key"])
    cache_root = source_dir.parents[2]
    eligible_catalog = cache_root / "global" / global_key / "m1_eligible_catalog.parquet"
    if not eligible_catalog.is_file():
        raise FileNotFoundError(eligible_catalog)
    if not reference_candidates.is_file():
        raise FileNotFoundError(reference_candidates)
    if reference_wide is not None and not reference_wide.is_file():
        raise FileNotFoundError(reference_wide)
    if reference_manifest is not None and not reference_manifest.is_file():
        raise FileNotFoundError(reference_manifest)
    validated_reference_manifest = (
        None
        if reference_manifest is None
        else _validate_reference_manifest(
            reference_candidates,
            reference_manifest,
            source_dir.parent.name,
            fusion_profile,
            rrf_constant,
            candidate_k,
        )
    )

    started = time.perf_counter()
    connection = duckdb.connect()
    stages: dict[str, dict[str, Any]] = {}
    try:
        settings = _configure(connection, threads, memory_limit, temp_dir)
        _create_source_views(connection, source_dir)
        stages["reference_load"] = _profiled_execute(
            connection,
            "reference_load",
            f"CREATE OR REPLACE TEMP TABLE m16_reference AS "
            f"SELECT * FROM read_parquet({_literal(reference_candidates)})",
            profile_dir,
        )
        reference_rows = _query_scalar(connection, "SELECT count(*) FROM m16_reference")
        reference_unique = _query_scalar(
            connection, "SELECT count(DISTINCT (customer_id, article_id)) FROM m16_reference"
        )
        if reference_rows != reference_unique:
            raise ValueError("reference candidates are not unique by user-item")

        if scope == "all":
            stages["source_union_materialize"] = _profiled_execute(
                connection, "source_union_materialize", _long_sql(eligible_catalog), profile_dir
            )
            stages["baseline_fusion"] = _profiled_execute(
                connection,
                "baseline_fusion",
                _baseline_fusion_sql(fusion_profile, rrf_constant, candidate_k),
                profile_dir,
            )
            baseline_parity = _candidate_parity(connection, "m16_baseline_candidates")
            stages["reference_groupby_wide"] = _profiled_execute(
                connection,
                "reference_groupby_wide",
                _reference_groupby_wide_sql(fusion_profile, rrf_constant),
                profile_dir,
            )
            stages["pivot_aggregate"] = _profiled_execute(
                connection, "pivot_aggregate", _pivot_sql(), profile_dir
            )
            stages["pivot_fusion"] = _profiled_execute(
                connection,
                "pivot_fusion",
                _pivot_fusion_sql(fusion_profile, rrf_constant, candidate_k),
                profile_dir,
            )
            pivot_parity = _candidate_parity(connection, "m16_pivot_candidates")
        else:
            baseline_parity = None
            pivot_parity = None

        stages["direct_reference_wide"] = _profiled_execute(
            connection,
            "direct_reference_wide",
            _direct_wide_sql(fusion_profile, rrf_constant),
            profile_dir,
        )
        if scope == "all":
            wide_parity = _wide_parity(connection)
            if wide_parity["feature_mismatch_rows"] != 0:
                raise ValueError(
                    "direct wide features differ from group-by reference"
                )
        else:
            direct_rows = _query_scalar(connection, "SELECT count(*) FROM m16_direct_wide")
            direct_unique = _query_scalar(
                connection,
                "SELECT count(DISTINCT (customer_id, article_id)) FROM m16_direct_wide",
            )
            wide_parity = {
                "direct_rows": direct_rows,
                "direct_unique_user_item_rows": direct_unique,
                "feature_mismatch_rows": None,
            }
            if direct_rows != reference_rows or direct_unique != reference_rows:
                raise ValueError(
                    "direct wide row identity differs from reference candidates"
                )

        if reference_wide is not None:
            actual_schema = connection.execute(
                "DESCRIBE SELECT * FROM m16_direct_wide"
            ).fetchall()
            reference_schema = connection.execute(
                f"DESCRIBE SELECT * FROM read_parquet({_literal(reference_wide)})"
            ).fetchall()
            if actual_schema != reference_schema:
                raise ValueError("reference wide schema differs from direct wide schema")
            stages["external_wide_parity"] = _profiled_execute(
                connection,
                "external_wide_parity",
                _external_wide_parity_sql(reference_wide),
                profile_dir,
            )
            external_wide_parity = _external_wide_parity(connection)
            if any(external_wide_parity.values()):
                raise ValueError(
                    "direct wide rows differ from external reference wide"
                )
        else:
            external_wide_parity = None

        optimized_path = artifact_dir / "candidate_features.parquet"
        stages["optimized_parquet_write"] = _profiled_execute(
            connection,
            "optimized_parquet_write",
            f"COPY m16_direct_wide TO {_literal(optimized_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)",
            profile_dir,
        )
    finally:
        connection.close()

    finished = time.perf_counter()
    source_files = {
        source: {
            "path": str((source_dir / f"{source}.parquet").resolve()),
            "bytes": (source_dir / f"{source}.parquet").stat().st_size,
            "sha256": _sha256(source_dir / f"{source}.parquet"),
            "rows": int(source_manifests[source]["stats"]["rows"]),
        }
        for source in SOURCES
    }
    result: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA,
        "stage": "M1.6",
        "status": "measured",
        "scope": scope,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "fusion_profile": fusion_profile,
            "rrf_constant": rrf_constant,
            "candidate_k": candidate_k,
            **settings,
        },
        "cache_identity": {
            "target_manifest": str((source_dir / "manifest.json").resolve()),
            "cutoff": root_manifest.get("cutoff"),
            "catalog_protocol": root_manifest.get("catalog_protocol"),
            "sample_rate": root_manifest.get("sample_rate"),
            "sampled_user_fingerprint": root_manifest.get("sampled_user_fingerprint"),
            "global_key": global_key,
        },
        "inputs": {
            "source_files": source_files,
            "reference_candidates": {
                "path": str(reference_candidates.resolve()),
                "bytes": reference_candidates.stat().st_size,
                "sha256": _sha256(reference_candidates),
                "rows": reference_rows,
            },
            "reference_manifest": None
            if reference_manifest is None
            else {
                "path": str(reference_manifest.resolve()),
                "bytes": reference_manifest.stat().st_size,
                "sha256": _sha256(reference_manifest),
                "created_at_utc": validated_reference_manifest.get("created_at_utc"),
            },
            "reference_wide": None
            if reference_wide is None
            else {
                "path": str(reference_wide.resolve()),
                "bytes": reference_wide.stat().st_size,
                "sha256": _sha256(reference_wide),
            },
            "eligible_catalog": {
                "path": str(eligible_catalog.resolve()),
                "bytes": eligible_catalog.stat().st_size,
                "sha256": _sha256(eligible_catalog),
            },
        },
        "stages": stages,
        "parity": {
            "baseline_candidates": baseline_parity,
            "deterministic_pivot_candidates": pivot_parity,
            "direct_wide_vs_groupby_wide": wide_parity,
            "direct_wide_vs_external_reference": external_wide_parity,
        },
        "artifacts": {
            "candidate_features": str(optimized_path.resolve()),
            "candidate_features_bytes": optimized_path.stat().st_size,
            "profiles": str(profile_dir.resolve()),
        },
        "total_wall_seconds": finished - started,
    }
    _write_json(output_dir / "metrics.json", result)
    _write_markdown(output_dir / "M1_6_REPORT.md", result)
    return result


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    parity = result["parity"]
    lines = [
        "# M1.6 Profiling Report",
        "",
        f"- scope: `{result['scope']}`",
        f"- cutoff: `{result['cache_identity']['cutoff']}`",
        f"- sample rate: `{result['cache_identity']['sample_rate']}`",
        f"- threads / memory: `{result['config']['threads']}` / `{result['config']['memory_limit']}`",
        f"- total wall seconds: `{result['total_wall_seconds']:.6f}`",
        "",
        "## Stage timing",
        "",
        "| Stage | Wall seconds | Peak buffer bytes | Peak temp bytes |",
        "|---|---:|---:|---:|",
    ]
    for name, stage in result["stages"].items():
        lines.append(
            f"| {name} | {stage['wall_seconds']:.6f} | "
            f"{stage.get('system_peak_buffer_memory')} | {stage.get('system_peak_temp_dir_size')} |"
        )
    lines.extend(["", "## Parity", "", "```json", json.dumps(parity, ensure_ascii=False, indent=2), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--reference-candidates", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--reference-wide", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "hot"), default="all")
    parser.add_argument("--fusion-profile", choices=tuple(FUSION_PROFILES), default="collaborative")
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="12GB")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_profile(
        args.source_dir,
        args.reference_candidates,
        args.output_dir,
        args.artifact_dir,
        reference_manifest=args.reference_manifest,
        reference_wide=args.reference_wide,
        scope=args.scope,
        fusion_profile=args.fusion_profile,
        rrf_constant=args.rrf_constant,
        candidate_k=args.candidate_k,
        threads=args.threads,
        memory_limit=args.memory_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
