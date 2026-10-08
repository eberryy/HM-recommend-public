from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m2 import M2Config
from .m212 import run_m212


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run M2.12 inner temporal selection and no-decay ablation"
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument(
        "--transactions", default="data/interim/audit/transactions.parquet"
    )
    parser.add_argument(
        "--m21-metrics",
        default="reports/m2_1/m2-1-v1-dev-rolling-10pct/metrics.json",
    )
    parser.add_argument(
        "--m210-metrics",
        default="reports/m2_10/m2-10-v1-target-aware-ranking-protocols/metrics.json",
    )
    parser.add_argument(
        "--m211-metrics",
        default="reports/m2_11/m2-11-v1-distribution-sampling-ablation/metrics.json",
    )
    parser.add_argument("--m29-cache-dir", default="artifacts/m2_9/cache-v1")
    parser.add_argument(
        "--feature-cache-dir", default="artifacts/m2_10/feature-cache-v1"
    )
    parser.add_argument(
        "--m210-artifact-dir",
        default="artifacts/m2_10/m2-10-v1-target-aware-ranking-protocols",
    )
    parser.add_argument("--run-id", default="m2-12-v1-inner-temporal-selection")
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
        result = run_m212(
            raw_dir=Path(args.raw_dir),
            transactions_path=Path(args.transactions),
            m21_metrics_path=Path(args.m21_metrics),
            m210_metrics_path=Path(args.m210_metrics),
            m211_metrics_path=Path(args.m211_metrics),
            m29_cache_dir=Path(args.m29_cache_dir),
            feature_cache_dir=Path(args.feature_cache_dir),
            m210_artifact_dir=Path(args.m210_artifact_dir),
            output_dir=Path("reports/m2_12") / args.run_id,
            artifact_dir=Path("artifacts/m2_12") / args.run_id,
            config=config,
            run_id=args.run_id,
        )
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        FloatingPointError,
    ) as error:
        print(f"M2.12 run failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "gate": result["development_summary"]["inner_selected_policy"],
                "elapsed_seconds": result["elapsed_seconds"],
                "report": result["artifacts"]["report"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
