from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m2 import M2Config
from .m22 import run_m22


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hm_recsys.m22_cli",
        description=(
            "Run development-only pooled LambdaRank feature-group ablations"
        ),
    )
    parser.add_argument("--m21-artifact-dir", required=True)
    parser.add_argument("--m21-metrics", required=True)
    parser.add_argument(
        "--transactions-path",
        default="data/interim/audit/transactions.parquet",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--history-weeks", type=int, default=12)
    parser.add_argument("--metric-k", type=int, default=12)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prediction-chunk-vectors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--min-mean-gain", type=float, default=0.0002)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = M2Config(
        history_weeks=args.history_weeks,
        metric_k=args.metric_k,
        candidate_k=args.candidate_k,
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        threads=args.threads,
        prediction_chunk_vectors=args.prediction_chunk_vectors,
        seed=args.seed,
        evaluation_role="development",
    )
    try:
        result = run_m22(
            m21_artifact_dir=Path(args.m21_artifact_dir),
            m21_metrics_path=Path(args.m21_metrics),
            transactions_path=Path(args.transactions_path),
            output_dir=Path(args.output_dir),
            artifact_dir=Path(args.artifact_dir),
            config=config,
            run_id=args.run_id,
            min_mean_gain=args.min_mean_gain,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"M2.2 run failed: {error}", file=sys.stderr)
        return 2
    analysis = result["analysis"]
    selected = analysis["selected_development_candidate"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "scope": result["scope"],
                "selected_development_candidate": selected,
                "selected_metrics": result["ordering_summary"][selected],
                "group_classification": {
                    group: row["classification"]
                    for group, row in analysis["groups"].items()
                },
                "elapsed_seconds": result["elapsed_seconds"],
                "final_confirmation_run": analysis["final_confirmation_run"],
                "report": result["artifacts"]["report"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
