from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m2 import M2Config
from .m21 import CandidateWindow
from .m3 import ALL_CUTOFFS, run_m3


DEFAULT_WINDOWS = {
    "2020-04-29": "m3-build-20200429-10pct-optimistic",
    "2020-05-27": "m1-5-v13-build-20200527-10pct-optimistic",
    "2020-06-24": "m1-5-v14-build-20200624-10pct-optimistic",
    "2020-07-22": "m1-5-v12-build-20200722-10pct-optimistic",
    "2020-08-19": "m1-5-v2-build-20200819-10pct",
}


def _key(cutoff: str) -> str:
    return cutoff.replace("-", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M3.0 three-window robustness evaluation")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--work-dir", default="data/interim/audit")
    parser.add_argument("--transactions", default="data/interim/audit/transactions.parquet")
    parser.add_argument("--m21-metrics", default="reports/m2_1/m2-1-v1-dev-rolling-10pct/metrics.json")
    parser.add_argument("--m28-metrics", default="reports/m2_8/m2-8-v1-item2vec-dev/metrics.json")
    parser.add_argument("--m212-metrics", default="reports/m2_12/m2-12-v1-inner-temporal-selection/metrics.json")
    parser.add_argument("--source-cache-dir", default="artifacts/m2_9/item2vec-source-v1")
    parser.add_argument("--m29-cache-dir", default="artifacts/m2_9/cache-v1")
    parser.add_argument("--feature-cache-dir", default="artifacts/m2_10/feature-cache-v1")
    parser.add_argument("--run-id", default="m3-0-v1-three-window-robustness")
    for cutoff in ALL_CUTOFFS:
        root = f"artifacts/m1_5/{DEFAULT_WINDOWS[cutoff]}"
        parser.add_argument(f"--candidates-{_key(cutoff)}", default=f"{root}/candidate_features.parquet")
        parser.add_argument(f"--manifest-{_key(cutoff)}", default=f"{root}/manifest.json")
    args = parser.parse_args(argv)
    windows = [
        CandidateWindow(
            cutoff=cutoff,
            candidate_path=Path(getattr(args, f"candidates_{_key(cutoff)}")),
            manifest_path=Path(getattr(args, f"manifest_{_key(cutoff)}")),
        )
        for cutoff in ALL_CUTOFFS
    ]
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
        result = run_m3(
            raw_dir=Path(args.raw_dir),
            work_dir=Path(args.work_dir),
            transactions_path=Path(args.transactions),
            m21_metrics_path=Path(args.m21_metrics),
            m28_metrics_path=Path(args.m28_metrics),
            m212_metrics_path=Path(args.m212_metrics),
            windows=windows,
            source_cache_dir=Path(args.source_cache_dir),
            m29_cache_dir=Path(args.m29_cache_dir),
            feature_cache_dir=Path(args.feature_cache_dir),
            output_dir=Path("reports/m3") / args.run_id,
            artifact_dir=Path("artifacts/m3") / args.run_id,
            config=config,
            run_id=args.run_id,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as error:
        print(f"M3.0 run failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "gate": result["summary"]["stability_gate"],
        "elapsed_seconds": result["elapsed_seconds"],
        "report": result["artifacts"]["report"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
