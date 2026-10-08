from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m1 import M1Config
from .m15 import run_m15
from .m2 import M2Config
from .m21 import CandidateWindow
from .m33 import ALL_CUTOFFS, run_m33


WINDOW_RUNS = {
    "2019-11-27": "m3-3-build-20191127-10pct-optimistic",
    "2019-12-25": "m3-3-build-20191225-10pct-optimistic",
    "2020-01-22": "m3-3-build-20200122-10pct-optimistic",
    "2020-02-19": "m3-3-build-20200219-10pct-optimistic",
    "2020-03-18": "m3-3-build-20200318-10pct-optimistic",
    "2020-04-29": "m3-build-20200429-10pct-optimistic",
    "2020-05-27": "m1-5-v13-build-20200527-10pct-optimistic",
    "2020-06-24": "m1-5-v14-build-20200624-10pct-optimistic",
    "2020-07-22": "m1-5-v12-build-20200722-10pct-optimistic",
    "2020-08-19": "m1-5-v2-build-20200819-10pct",
}

NEW_CUTOFFS = tuple(list(WINDOW_RUNS)[:5])


def _key(cutoff: str) -> str:
    return cutoff.replace("-", "")


def _ensure_m15_prerequisites(raw_dir: Path, work_dir: Path) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for cutoff in NEW_CUTOFFS:
        run_name = WINDOW_RUNS[cutoff]
        output_dir = Path("reports/m3_3") / f"prereq-m1-5-{_key(cutoff)}-10pct"
        artifact_dir = Path("artifacts/m1_5") / run_name
        candidate_path = artifact_dir / "candidate_features.parquet"
        manifest_path = artifact_dir / "manifest.json"
        if candidate_path.is_file() and manifest_path.is_file():
            evidence.append({"cutoff": cutoff, "status": "reused", "artifact_dir": str(artifact_dir)})
            continue
        if output_dir.exists() or artifact_dir.exists():
            raise FileExistsError(f"incomplete M3.3 M1.5 prerequisite must be preserved: {cutoff}")
        print(f"M3.3 prerequisite {cutoff}: building M1.5 Top100", flush=True)
        result = run_m15(
            raw_dir=raw_dir,
            work_dir=work_dir,
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            cache_dir=Path("artifacts/m1_5/cache"),
            config=M1Config(
                cutoff=cutoff,
                sample_rate=0.1,
                history_weeks=12,
                popularity_days=28,
                covisit_days=28,
                candidate_k=100,
                source_k=100,
                fusion_profile="collaborative",
                catalog_protocol="optimistic_all_articles",
            ),
            cache_mode="auto",
            run_id=f"m3-3-prereq-m1-5-{_key(cutoff)}-10pct",
        )
        evidence.append(
            {
                "cutoff": cutoff,
                "status": "built",
                "elapsed_seconds": result["elapsed_seconds"],
                "artifact_dir": str(artifact_dir),
            }
        )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M3.3 cross-season seasonal ranking experiment")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--work-dir", default="data/interim/audit")
    parser.add_argument("--transactions", default="data/interim/audit/transactions.parquet")
    parser.add_argument("--m28-metrics", default="reports/m2_8/m2-8-v1-item2vec-dev/metrics.json")
    parser.add_argument("--m3-metrics", default="reports/m3/m3-0-v1-three-window-robustness/metrics.json")
    parser.add_argument("--source-cache-dir", default="artifacts/m2_9/item2vec-source-v1")
    parser.add_argument("--m29-cache-dir", default="artifacts/m2_9/cache-v1")
    parser.add_argument("--target-feature-cache-dir", default="artifacts/m2_10/feature-cache-v1")
    parser.add_argument("--seasonal-cache-dir", default="artifacts/m3_3/seasonal-feature-cache-v1")
    parser.add_argument("--run-id", default="m3-3-v1-cross-season-adaptive-seasonal")
    parser.add_argument("--skip-m15-prerequisites", action="store_true")
    for cutoff in ALL_CUTOFFS:
        root = f"artifacts/m1_5/{WINDOW_RUNS[cutoff]}"
        parser.add_argument(f"--candidates-{_key(cutoff)}", default=f"{root}/candidate_features.parquet")
        parser.add_argument(f"--manifest-{_key(cutoff)}", default=f"{root}/manifest.json")
    args = parser.parse_args(argv)
    raw_dir = Path(args.raw_dir)
    work_dir = Path(args.work_dir)
    try:
        prereq = [] if args.skip_m15_prerequisites else _ensure_m15_prerequisites(raw_dir, work_dir)
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
        result = run_m33(
            raw_dir=raw_dir,
            work_dir=work_dir,
            transactions_path=Path(args.transactions),
            m28_metrics_path=Path(args.m28_metrics),
            m3_metrics_path=Path(args.m3_metrics),
            windows=windows,
            source_cache_dir=Path(args.source_cache_dir),
            m29_cache_dir=Path(args.m29_cache_dir),
            target_feature_cache_dir=Path(args.target_feature_cache_dir),
            seasonal_cache_dir=Path(args.seasonal_cache_dir),
            output_dir=Path("reports/m3_3") / args.run_id,
            artifact_dir=Path("artifacts/m3_3") / args.run_id,
            config=config,
            run_id=args.run_id,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as error:
        print(f"M3.3 run failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "m15_prerequisites": prereq,
                "accepted": [
                    name for name, row in result["summary"]["orderings"].items() if row["accepted"]
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
