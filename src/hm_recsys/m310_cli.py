from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m2 import M2Config
from .m310 import run_m310


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M3.10 source-specific image admission")
    parser.add_argument("--variant-cache-dir", required=True)
    parser.add_argument("--soft-cache-dir", required=True)
    parser.add_argument("--neighbor-metrics", required=True)
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--m33-metrics", required=True)
    parser.add_argument("--direct-cache-dir", default="artifacts/m3_10/direct-visual-cache-v1")
    parser.add_argument("--enriched-cache-dir", default="artifacts/m3_10/enriched-cache-v1")
    parser.add_argument("--output-dir", default="reports/m3_10/m3-10-v1-source-specific-image-admission")
    parser.add_argument("--artifact-dir", default="artifacts/m3_10/m3-10-v1-source-specific-image-admission")
    parser.add_argument("--run-id", default="m3-10-v1-source-specific-image-admission")
    args = parser.parse_args(argv)
    config = M2Config(
        history_weeks=12,
        metric_k=12,
        candidate_k=500,
        num_boost_round=200,
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=100,
        threads=8,
        prediction_chunk_vectors=64,
        seed=20260824,
        evaluation_role="development",
    )
    try:
        result = run_m310(
            variant_cache_dir=Path(args.variant_cache_dir),
            soft_cache_dir=Path(args.soft_cache_dir),
            neighbor_metrics_path=Path(args.neighbor_metrics),
            transactions_path=Path(args.transactions),
            m33_metrics_path=Path(args.m33_metrics),
            direct_cache_dir=Path(args.direct_cache_dir),
            enriched_cache_dir=Path(args.enriched_cache_dir),
            output_dir=Path(args.output_dir),
            artifact_dir=Path(args.artifact_dir),
            config=config,
            run_id=args.run_id,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as error:
        print(f"M3.10 run failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "accepted": result["summary"]["accepted"],
        "mechanism_gate": result["summary"]["mechanism_gate"],
        "mean_map@12": result["summary"]["mean_map@12"],
        "mean_delta_vs_base": result["summary"]["mean_delta_vs_base"],
        "elapsed_seconds": result["elapsed_seconds"],
        "report": result["artifacts"]["report"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
