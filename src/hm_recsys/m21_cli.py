from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m2 import M2Config
from .m21 import CandidateWindow, run_m21


WINDOW_KEYS = ("20200527", "20200624", "20200722", "20200819")
WINDOW_CUTOFFS = {
    "20200527": "2020-05-27",
    "20200624": "2020-06-24",
    "20200722": "2020-07-22",
    "20200819": "2020-08-19",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m hm_recsys.m21_cli",
        description=(
            "Run development-only single/pooled binary/LambdaRank M2.1 comparisons"
        ),
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--work-dir", default="data/interim/audit")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--run-id", required=True)
    for key in WINDOW_KEYS:
        parser.add_argument(f"--candidates-{key}", required=True)
        parser.add_argument(f"--manifest-{key}", required=True)
    parser.add_argument("--reference-20200722", required=True)
    parser.add_argument("--reference-20200819", required=True)
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
    return parser


def _windows(args: argparse.Namespace) -> list[CandidateWindow]:
    references = {
        "20200722": Path(args.reference_20200722),
        "20200819": Path(args.reference_20200819),
    }
    return [
        CandidateWindow(
            cutoff=WINDOW_CUTOFFS[key],
            candidate_path=Path(getattr(args, f"candidates_{key}")),
            manifest_path=Path(getattr(args, f"manifest_{key}")),
            reference_metrics_path=references.get(key),
        )
        for key in WINDOW_KEYS
    ]


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
        result = run_m21(
            raw_dir=Path(args.raw_dir),
            work_dir=Path(args.work_dir),
            output_dir=Path(args.output_dir),
            artifact_dir=Path(args.artifact_dir),
            windows=_windows(args),
            config=config,
            run_id=args.run_id,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"M2.1 run failed: {error}", file=sys.stderr)
        return 2
    summary = result["development_summary"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "scope": result["scope"],
                "selected_development_candidate": summary[
                    "selected_development_candidate"
                ],
                "best_pure_variant": summary["best_pure_variant"],
                "selected_metrics": summary["orderings"][
                    summary["selected_development_candidate"]
                ],
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
