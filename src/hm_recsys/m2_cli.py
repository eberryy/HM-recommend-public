from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m2 import M2Config
from .m2_run import run_m2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hm_recsys.m2_cli",
        description="Build point-in-time features and run M2 LightGBM baselines",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--work-dir", default="data/interim/audit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-candidates", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-candidates", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--m1-reference-metrics", required=True)
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
    parser.add_argument(
        "--evaluation-role", choices=("development", "final"), default="final"
    )
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
        evaluation_role=args.evaluation_role,
    )
    try:
        result = run_m2(
            raw_dir=Path(args.raw_dir),
            work_dir=Path(args.work_dir),
            output_dir=Path(args.output_dir),
            artifact_dir=Path(args.artifact_dir),
            train_candidates=Path(args.train_candidates),
            train_manifest=Path(args.train_manifest),
            validation_candidates=Path(args.validation_candidates),
            validation_manifest=Path(args.validation_manifest),
            m1_reference_metrics=Path(args.m1_reference_metrics),
            config=config,
            run_id=args.run_id,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"M2 run failed: {error}", file=sys.stderr)
        return 2
    overall = result["evaluation"]["orderings"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "training_scope": result["training_scope"],
                "validation_scope": result["validation_scope"],
                "rrf_map@12": overall["rrf"]["segments"]["overall"]["map@12"],
                "retrieval_lgb_map@12": overall["lightgbm_retrieval"]["segments"]["overall"]["map@12"],
                "full_lgb_map@12": overall["lightgbm_full"]["segments"]["overall"]["map@12"],
                "m1_parity": result["m1_parity"]["status"],
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
