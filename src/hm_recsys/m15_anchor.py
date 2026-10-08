from __future__ import annotations

"""Create a new cache root anchored to an audited historical fusion artifact."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .m1 import M1Config, SOURCES, _path_literal, _scalar
from .m15 import (
    CACHE_SCHEMA,
    _content_key,
    _fusion_expected,
    _read_json,
    _relation_stats,
    _write_json,
)


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def _pair_rank_audit(
    connection: duckdb.DuckDBPyConnection,
    historical: Path,
    recomputed: Path,
) -> dict[str, Any]:
    historical_sql = f"read_parquet({_path_literal(historical)})"
    recomputed_sql = f"read_parquet({_path_literal(recomputed)})"
    missing_pairs = int(
        _scalar(
            connection,
            f"""
            SELECT count(*) FROM (
                SELECT customer_id::VARCHAR, article_id::VARCHAR FROM {historical_sql}
                EXCEPT ALL
                SELECT customer_id::VARCHAR, article_id::VARCHAR FROM {recomputed_sql}
            )
            """,
        )
    )
    extra_pairs = int(
        _scalar(
            connection,
            f"""
            SELECT count(*) FROM (
                SELECT customer_id::VARCHAR, article_id::VARCHAR FROM {recomputed_sql}
                EXCEPT ALL
                SELECT customer_id::VARCHAR, article_id::VARCHAR FROM {historical_sql}
            )
            """,
        )
    )
    row = connection.execute(
        f"""
        SELECT count(*) FILTER (
                   WHERE historical.candidate_rank != recomputed.candidate_rank
               ),
               max(abs(historical.fused_score - recomputed.fused_score))
        FROM {historical_sql} historical
        JOIN {recomputed_sql} recomputed USING (customer_id, article_id)
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("historical fusion audit returned no row")
    return {
        "missing_pairs": missing_pairs,
        "extra_pairs": extra_pairs,
        "rank_difference_rows": int(row[0]),
        "maximum_fused_score_difference": float(row[1] or 0.0),
    }


def create_historical_fusion_anchor(
    source_cache_dir: Path,
    destination_cache_dir: Path,
    target_key: str,
    historical_candidates: Path,
    recomputed_candidates: Path,
    output_path: Path,
    config: M1Config,
) -> dict[str, Any]:
    """Copy a validated cache tree and seed its fusion reference from history."""
    if destination_cache_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite destination cache root: {destination_cache_dir}"
        )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite anchor report: {output_path}")
    _require_file(historical_candidates)
    _require_file(recomputed_candidates)

    source_target = source_cache_dir / "target" / target_key / "sources"
    target_manifest_path = source_target / "manifest.json"
    target_manifest = _read_json(target_manifest_path)
    if target_manifest.get("schema_version") != CACHE_SCHEMA:
        raise ValueError("source target cache schema mismatch")
    if target_manifest.get("cutoff") != config.cutoff:
        raise ValueError("source target cache cutoff mismatch")
    if target_manifest.get("sample_rate") != config.sample_rate:
        raise ValueError("source target cache sample-rate mismatch")
    global_key = target_manifest.get("global_key")
    if not isinstance(global_key, str) or not global_key:
        raise ValueError("source target manifest has no global_key")
    source_global = source_cache_dir / "global" / global_key
    _require_file(source_global / "manifest.json")
    for source in SOURCES:
        _require_file(source_target / f"{source}.parquet")
        _require_file(source_target / f"{source}.manifest.json")

    connection = duckdb.connect()
    try:
        audit = _pair_rank_audit(
            connection, historical_candidates, recomputed_candidates
        )
        if audit["missing_pairs"] or audit["extra_pairs"]:
            raise ValueError(
                "historical anchor rejected because candidate pair sets differ: "
                f"{audit}"
            )

        destination_global = destination_cache_dir / "global" / global_key
        destination_sources = (
            destination_cache_dir / "target" / target_key / "sources"
        )
        destination_global.parent.mkdir(parents=True)
        destination_sources.parent.mkdir(parents=True)
        shutil.copytree(source_global, destination_global)
        shutil.copytree(source_target, destination_sources)
        for source in SOURCES:
            manifest_path = destination_sources / f"{source}.manifest.json"
            manifest = _read_json(manifest_path)
            manifest["artifact"] = str(
                (destination_sources / f"{source}.parquet").resolve()
            )
            manifest["historical_anchor_copy"] = {
                "source_cache_root": str(source_cache_dir.resolve()),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json(manifest_path, manifest)

        fusion_expected = _fusion_expected(target_key, config)
        fusion_key = _content_key(fusion_expected)
        fusion_root = (
            destination_cache_dir
            / "target"
            / target_key
            / "fusion"
            / fusion_key
        )
        fusion_root.mkdir(parents=True)
        fusion_parquet = fusion_root / "candidates.parquet"
        shutil.copy2(historical_candidates, fusion_parquet)
        connection.execute(
            f"CREATE TEMP VIEW anchored_candidates AS "
            f"SELECT * FROM read_parquet({_path_literal(fusion_parquet)})"
        )
        fusion_manifest = {
            **fusion_expected,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": str(fusion_parquet.resolve()),
            "artifact_bytes": fusion_parquet.stat().st_size,
            "stats": _relation_stats(connection, "anchored_candidates"),
            "historical_anchor": {
                "reference": str(historical_candidates.resolve()),
                "recomputed_artifact": str(recomputed_candidates.resolve()),
                **audit,
            },
        }
        _write_json(fusion_root / "manifest.json", fusion_manifest)
    finally:
        connection.close()

    result = {
        "stage": "M1.5-historical-fusion-anchor",
        "status": "created",
        "source_cache_root": str(source_cache_dir.resolve()),
        "destination_cache_root": str(destination_cache_dir.resolve()),
        "target_key": target_key,
        "global_key": global_key,
        "fusion_key": fusion_key,
        "historical_candidates": str(historical_candidates.resolve()),
        "recomputed_candidates": str(recomputed_candidates.resolve()),
        "audit": audit,
        "config": {
            "cutoff": config.cutoff,
            "sample_rate": config.sample_rate,
            "fusion_profile": config.fusion_profile,
            "rrf_constant": config.rrf_constant,
            "candidate_k": config.candidate_k,
            "catalog_protocol": config.catalog_protocol,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m hm_recsys.m15_anchor")
    parser.add_argument("--source-cache-dir", required=True)
    parser.add_argument("--destination-cache-dir", required=True)
    parser.add_argument("--target-key", required=True)
    parser.add_argument("--historical-candidates", required=True)
    parser.add_argument("--recomputed-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--sample-rate", type=float, required=True)
    parser.add_argument("--fusion-profile", default="collaborative")
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument(
        "--catalog-protocol",
        choices=("optimistic_all_articles", "strict"),
        default="optimistic_all_articles",
    )
    args = parser.parse_args()
    config = M1Config(
        cutoff=args.cutoff,
        sample_rate=args.sample_rate,
        fusion_profile=args.fusion_profile,
        rrf_constant=args.rrf_constant,
        candidate_k=args.candidate_k,
        evaluation_mode="final",
        catalog_protocol=args.catalog_protocol,
    )
    result = create_historical_fusion_anchor(
        source_cache_dir=Path(args.source_cache_dir),
        destination_cache_dir=Path(args.destination_cache_dir),
        target_key=args.target_key,
        historical_candidates=Path(args.historical_candidates),
        recomputed_candidates=Path(args.recomputed_candidates),
        output_path=Path(args.output),
        config=config,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
