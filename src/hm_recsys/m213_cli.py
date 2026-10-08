from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m2 import M2Config
from .m213 import run_m213


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run M2.13A image expanded-pool ranking revisit"
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
        "--m26-metrics",
        default="reports/m2_6/m2-6-v4-cross-seed/metrics.json",
    )
    parser.add_argument(
        "--m212-metrics",
        default="reports/m2_12/m2-12-v1-inner-temporal-selection/metrics.json",
    )
    parser.add_argument(
        "--source-cache-dir", default="artifacts/m2_6/cache-v4-cross-seed"
    )
    parser.add_argument(
        "--feature-cache-dir", default="artifacts/m2_13/image-feature-cache-v1"
    )
    parser.add_argument(
        "--m26-artifact-dir", default="artifacts/m2_6/m2-6-v4-cross-seed"
    )
    parser.add_argument("--run-id", default="m2-13a-v1-image-ranking-revisit")
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
        result = run_m213(
            raw_dir=Path(args.raw_dir),
            transactions_path=Path(args.transactions),
            m21_metrics_path=Path(args.m21_metrics),
            m26_metrics_path=Path(args.m26_metrics),
            m212_metrics_path=Path(args.m212_metrics),
            source_cache_dir=Path(args.source_cache_dir),
            feature_cache_dir=Path(args.feature_cache_dir),
            m26_artifact_dir=Path(args.m26_artifact_dir),
            output_dir=Path("reports/m2_13") / args.run_id,
            artifact_dir=Path("artifacts/m2_13") / args.run_id,
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
        print(f"M2.13A run failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "gates": {
                    "training": result["development_summary"][
                        "new_training_protocol_gate_vs_m2_6"
                    ],
                    "target_aware": result["development_summary"][
                        "target_aware_gate_vs_image_control"
                    ],
                    "deployment": result["development_summary"][
                        "primary_deployment_gate"
                    ],
                },
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
