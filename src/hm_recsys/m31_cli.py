from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m2 import M2Config
from .m31 import run_m31


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M3.1 bounded source and feature ablations")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--transactions", default="data/interim/audit/transactions.parquet")
    parser.add_argument("--m3-metrics", default="reports/m3/m3-0-v1-three-window-robustness/metrics.json")
    parser.add_argument("--m29-cache-dir", default="artifacts/m2_9/cache-v1")
    parser.add_argument("--feature-cache-dir", default="artifacts/m2_10/feature-cache-v1")
    parser.add_argument("--run-id", default="m3-1-v1-bounded-source-feature-ablation")
    args = parser.parse_args(argv)
    config = M2Config(
        history_weeks=12,
        metric_k=12,
        candidate_k=300,
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
        result = run_m31(
            raw_dir=Path(args.raw_dir),
            transactions_path=Path(args.transactions),
            m3_metrics_path=Path(args.m3_metrics),
            m29_cache_dir=Path(args.m29_cache_dir),
            feature_cache_dir=Path(args.feature_cache_dir),
            output_dir=Path("reports/m3_1") / args.run_id,
            artifact_dir=Path("artifacts/m3_1") / args.run_id,
            config=config,
            run_id=args.run_id,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as error:
        print(f"M3.1 run failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "elapsed_seconds": result["elapsed_seconds"],
        "report": result["artifacts"]["report"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
