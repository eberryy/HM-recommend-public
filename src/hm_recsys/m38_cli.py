from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m2 import M2Config
from .m38 import run_m38


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M3.8 image-source factorial ranking")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--work-dir", default="data/interim/audit")
    parser.add_argument("--transactions", default="data/interim/audit/transactions.parquet")
    parser.add_argument("--neighbor-metrics", default="reports/m2_5/m2-5-v2-exact-neighbors-full/metrics.json")
    parser.add_argument("--m33-metrics", default="reports/m3_3/m3-3-v1-cross-season-adaptive-seasonal/metrics.json")
    parser.add_argument("--m29-cache-dir", default="artifacts/m2_9/cache-v1")
    parser.add_argument("--historical-cache-dir", default="artifacts/m3_5/cold-source-cache-v1")
    parser.add_argument("--soft-cache-dir", default="artifacts/m3_8/soft-source-cache-v1")
    parser.add_argument("--candidate-cache-dir", default="artifacts/m3_8/candidate-cache-v1")
    parser.add_argument("--feature-cache-dir", default="artifacts/m3_8/feature-cache-v1")
    parser.add_argument("--variant-cache-dir", default="artifacts/m3_8/variant-cache-v1")
    parser.add_argument("--run-id", default="m3-8-v1-image-source-factorial-ranking")
    args = parser.parse_args(argv)
    config = M2Config(
        history_weeks=12, metric_k=12, candidate_k=500, num_boost_round=200,
        learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, threads=8,
        prediction_chunk_vectors=64, seed=20260824, evaluation_role="development",
    )
    try:
        result = run_m38(
            raw_dir=Path(args.raw_dir), work_dir=Path(args.work_dir),
            transactions_path=Path(args.transactions), neighbor_metrics_path=Path(args.neighbor_metrics),
            m29_cache_dir=Path(args.m29_cache_dir), historical_cache_dir=Path(args.historical_cache_dir),
            soft_cache_dir=Path(args.soft_cache_dir), candidate_cache_dir=Path(args.candidate_cache_dir),
            feature_cache_dir=Path(args.feature_cache_dir), variant_cache_dir=Path(args.variant_cache_dir),
            m33_metrics_path=Path(args.m33_metrics), output_dir=Path("reports/m3_8") / args.run_id,
            artifact_dir=Path("artifacts/m3_8") / args.run_id, config=config, run_id=args.run_id,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as error:
        print(f"M3.8 run failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"], "run_id": result["run_id"],
        "accepted": result["summary"]["accepted_variants"],
        "selected": result["summary"]["selected_variant"],
        "elapsed_seconds": result["elapsed_seconds"], "report": result["artifacts"]["report"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
