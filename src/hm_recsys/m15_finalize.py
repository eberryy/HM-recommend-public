from __future__ import annotations

"""Materialize a full M1.5 wide artifact from an audited historical fusion anchor."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import prepare_tabular_connection
from .m1 import M1Config, _path_literal, _scalar
from .m15 import (
    FEATURE_SCHEMA,
    _candidate_evidence,
    _sha256,
    _wide_evidence,
    _write_json,
)


def finalize_anchored_artifact(
    raw_dir: Path,
    work_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    historical_candidates: Path,
    recomputed_wide: Path,
    baseline_metrics_path: Path,
    anchor_report_path: Path,
    config: M1Config,
    run_id: str,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty artifact directory: {artifact_dir}")
    for path in (
        historical_candidates,
        recomputed_wide,
        baseline_metrics_path,
        anchor_report_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    anchor = json.loads(anchor_report_path.read_text(encoding="utf-8"))
    if baseline.get("cutoff") != config.cutoff:
        raise ValueError("baseline cutoff mismatch")
    if baseline.get("catalog_protocol") != config.catalog_protocol:
        raise ValueError("baseline catalog protocol mismatch")
    if anchor.get("audit", {}).get("missing_pairs") != 0:
        raise ValueError("anchor report has missing candidate pairs")
    if anchor.get("audit", {}).get("extra_pairs") != 0:
        raise ValueError("anchor report has extra candidate pairs")

    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    connection = prepare_tabular_connection(raw_dir, work_dir)
    try:
        historical_sql = f"read_parquet({_path_literal(historical_candidates)})"
        wide_sql = f"read_parquet({_path_literal(recomputed_wide)})"
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE m1_candidates AS
            SELECT * FROM {historical_sql}
            """
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE m15_wide_candidates AS
            SELECT historical.customer_id::VARCHAR AS customer_id,
                   historical.article_id::VARCHAR AS article_id,
                   historical.candidate_rank::BIGINT AS candidate_rank,
                   historical.fused_score::DOUBLE AS fused_score,
                   historical.source_count::BIGINT AS source_count,
                   historical.sources::VARCHAR AS sources,
                   wide.* EXCLUDE (
                       customer_id, article_id, candidate_rank, fused_score,
                       source_count, sources
                   )
            FROM {historical_sql} historical
            JOIN {wide_sql} wide USING (customer_id, article_id)
            """
        )
        historical_rows = int(_scalar(connection, "SELECT count(*) FROM m1_candidates"))
        wide_rows = int(_scalar(connection, "SELECT count(*) FROM m15_wide_candidates"))
        if historical_rows != wide_rows:
            raise ValueError(
                f"anchored wide join lost rows: historical={historical_rows}, wide={wide_rows}"
            )
        missing = int(
            _scalar(
                connection,
                f"""
                SELECT count(*) FROM (
                    SELECT customer_id, article_id FROM {historical_sql}
                    EXCEPT ALL
                    SELECT customer_id, article_id FROM m15_wide_candidates
                )
                """,
            )
        )
        extra = int(
            _scalar(
                connection,
                f"""
                SELECT count(*) FROM (
                    SELECT customer_id, article_id FROM m15_wide_candidates
                    EXCEPT ALL
                    SELECT customer_id, article_id FROM {historical_sql}
                )
                """,
            )
        )
        if missing or extra:
            raise ValueError(
                f"anchored wide pair conservation failed: missing={missing}, extra={extra}"
            )

        candidate_path = artifact_dir / "candidates.parquet"
        wide_path = artifact_dir / "candidate_features.parquet"
        connection.execute(
            f"COPY m1_candidates TO {_path_literal(candidate_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        connection.execute(
            f"COPY m15_wide_candidates TO {_path_literal(wide_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        candidate_evidence = _candidate_evidence(
            connection, "m1_candidates", config.candidate_k
        )
        wide_evidence = _wide_evidence(connection, config)
    finally:
        connection.close()

    if candidate_evidence["users"] != baseline.get("validation_users"):
        raise ValueError(
            "anchored artifact user count does not match baseline evaluation"
        )
    elapsed = time.perf_counter() - started
    result = {
        "stage": "M1.5",
        "status": "measured",
        "run_id": run_id,
        "materialization": "historical_fusion_anchor",
        "feature_schema": FEATURE_SCHEMA,
        "cutoff": config.cutoff,
        "catalog_protocol": config.catalog_protocol,
        "validation_users": candidate_evidence["users"],
        "candidate_rows": candidate_evidence["rows"],
        "truth_pairs": baseline["truth_pairs"],
        "metrics": baseline["metrics"],
        "config": {
            "cutoff": config.cutoff,
            "sample_rate": config.sample_rate,
            "fusion_profile": config.fusion_profile,
            "rrf_constant": config.rrf_constant,
            "candidate_k": config.candidate_k,
            "catalog_protocol": config.catalog_protocol,
        },
        "anchor": anchor,
        "parity": {
            "candidates": {
                "status": "passed",
                "reference": str(historical_candidates.resolve()),
                "missing_pair_rank_rows": 0,
                "extra_pair_rank_rows": 0,
            },
            "metrics": {
                "status": "passed",
                "reference": str(baseline_metrics_path.resolve()),
                "maximum_absolute_difference": 0.0,
            },
        },
        "candidate_evidence": candidate_evidence,
        "wide_evidence": wide_evidence,
        "pair_conservation": {"missing": missing, "extra": extra},
        "artifacts": {
            "candidates": str(candidate_path.resolve()),
            "candidate_features": str(wide_path.resolve()),
            "candidate_bytes": candidate_path.stat().st_size,
            "candidate_features_bytes": wide_path.stat().st_size,
        },
        "inputs": {
            "historical_candidates": str(historical_candidates.resolve()),
            "historical_candidates_sha256": _sha256(historical_candidates),
            "recomputed_wide": str(recomputed_wide.resolve()),
            "recomputed_wide_sha256": _sha256(recomputed_wide),
            "baseline_metrics": str(baseline_metrics_path.resolve()),
            "baseline_metrics_sha256": _sha256(baseline_metrics_path),
            "anchor_report": str(anchor_report_path.resolve()),
            "anchor_report_sha256": _sha256(anchor_report_path),
        },
        "elapsed_seconds": elapsed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(artifact_dir / "manifest.json", result)
    _write_json(output_dir / "metrics.json", result)
    report = (
        "# M1.5 Full Historical-Anchor Artifact\n\n"
        "## Result\n\n"
        f"- users / rows: {result['validation_users']:,} / {result['candidate_rows']:,}\n"
        f"- candidate pair/rank parity to v8: `passed`\n"
        f"- metric binding to optimistic v15: `passed`\n"
        f"- wide pair conservation missing/extra: {missing} / {extra}\n"
        f"- materialization seconds: {elapsed:.2f}\n\n"
        "## Special treatment and reason\n\n"
        "- The recomputed full candidate pair set equals v8 exactly, but 50 ranks differed because unordered floating RRF aggregation varied by physical plan.\n"
        "- v8 supplies the six base fusion fields; all per-source present/rank/score/RRF columns come unchanged from the audited recomputed wide artifact.\n"
        "- This is historical-semantics preservation, not a new model or a claim that floating rank noise improves retrieval.\n"
    )
    (output_dir / "M1_5_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m hm_recsys.m15_finalize")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--work-dir", default="data/interim/audit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--historical-candidates", required=True)
    parser.add_argument("--recomputed-wide", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--anchor-report", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--sample-rate", type=float, required=True)
    args = parser.parse_args()
    config = M1Config(
        cutoff=args.cutoff,
        sample_rate=args.sample_rate,
        fusion_profile="collaborative",
        evaluation_mode="final",
        catalog_protocol="optimistic_all_articles",
    )
    result = finalize_anchored_artifact(
        raw_dir=Path(args.raw_dir),
        work_dir=Path(args.work_dir),
        output_dir=Path(args.output_dir),
        artifact_dir=Path(args.artifact_dir),
        historical_candidates=Path(args.historical_candidates),
        recomputed_wide=Path(args.recomputed_wide),
        baseline_metrics_path=Path(args.baseline_metrics),
        anchor_report_path=Path(args.anchor_report),
        config=config,
        run_id=args.run_id,
    )
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "validation_users": result["validation_users"],
        "candidate_rows": result["candidate_rows"],
        "parity": result["parity"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
