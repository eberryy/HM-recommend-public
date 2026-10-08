from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m34 import OUTER_WINDOWS, run_m34


WINDOW_RUNS = {
    "2020-01-22": "m3-3-build-20200122-10pct-optimistic",
    "2020-03-18": "m3-3-build-20200318-10pct-optimistic",
    "2020-06-24": "m1-5-v14-build-20200624-10pct-optimistic",
    "2020-08-19": "m1-5-v2-build-20200819-10pct",
}


def _key(cutoff: str) -> str:
    return cutoff.replace("-", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M3.4 season-aware retrieval")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--transactions", default="data/interim/audit/transactions.parquet")
    parser.add_argument(
        "--source-cache-dir", default="artifacts/m3_4/seasonal-retrieval-cache-v1"
    )
    parser.add_argument(
        "--benchmark-features",
        default="artifacts/m3_3/seasonal-feature-cache-v1/2020-06-24/features.parquet",
    )
    parser.add_argument(
        "--benchmark-category-maps",
        default=(
            "artifacts/m3_3/m3-3-v1-cross-season-adaptive-seasonal/"
            "early_summer_20200624/outer-category-maps.json"
        ),
    )
    parser.add_argument(
        "--benchmark-model",
        default=(
            "artifacts/m3_3/m3-3-v1-cross-season-adaptive-seasonal/"
            "early_summer_20200624/lightgbm-adaptive_seasonal.txt"
        ),
    )
    parser.add_argument("--run-id", default="m3-4-v1-season-aware-retrieval")
    for cutoff in OUTER_WINDOWS.values():
        root = f"artifacts/m2_9/cache-v1/{cutoff}"
        parser.add_argument(
            f"--candidates-{_key(cutoff)}", default=f"{root}/expanded-candidates.parquet"
        )
        parser.add_argument(
            f"--manifest-{_key(cutoff)}", default=f"{root}/candidate-manifest.json"
        )
    args = parser.parse_args(argv)
    base_candidates = {
        cutoff: (
            Path(getattr(args, f"candidates_{_key(cutoff)}")),
            Path(getattr(args, f"manifest_{_key(cutoff)}")),
        )
        for cutoff in OUTER_WINDOWS.values()
    }
    try:
        result = run_m34(
            raw_dir=Path(args.raw_dir),
            transactions_path=Path(args.transactions),
            base_candidates=base_candidates,
            source_cache_dir=Path(args.source_cache_dir),
            benchmark_feature_path=Path(args.benchmark_features),
            benchmark_category_maps_path=Path(args.benchmark_category_maps),
            benchmark_model_path=Path(args.benchmark_model),
            output_dir=Path("reports/m3_4") / args.run_id,
            artifact_dir=Path("artifacts/m3_4") / args.run_id,
            run_id=args.run_id,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"M3.4 run failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "prepare_frame_gate": result["prepare_frame_benchmark"]["status"],
                "retrieval_gate": result["summary"]["retrieval_gate"]["passed"],
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
