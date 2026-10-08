from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m2 import M2Config
from .m27 import run_m27


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hm_recsys.m27_cli",
        description="Run development-only M2.7 two-stage image ranking diagnostics",
    )
    parser.add_argument("--transactions", default="data/interim/audit/transactions.parquet")
    parser.add_argument("--m26-metrics", required=True)
    parser.add_argument("--m21-metrics", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage1-top-k", type=int, default=100)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-data-in-leaf", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = M2Config(
        history_weeks=12,
        metric_k=12,
        candidate_k=300,
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        threads=args.threads,
        seed=args.seed,
        evaluation_role="development",
    )
    try:
        result = run_m27(
            transactions_path=Path(args.transactions),
            m26_metrics_path=Path(args.m26_metrics),
            m21_metrics_path=Path(args.m21_metrics),
            cache_dir=Path(args.cache_dir),
            output_dir=Path(args.output_dir),
            artifact_dir=Path(args.artifact_dir),
            config=config,
            run_id=args.run_id,
            stage1_top_k=args.stage1_top_k,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"M2.7 run failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "gate": result["development_summary"]["selection_gate"],
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
