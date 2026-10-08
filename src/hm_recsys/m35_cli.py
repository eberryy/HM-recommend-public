from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m34 import OUTER_WINDOWS
from .m35 import run_m35


def _key(cutoff: str) -> str:
    return cutoff.replace("-", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M3.5 content-season-image cold retrieval")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--transactions", default="data/interim/audit/transactions.parquet")
    parser.add_argument("--neighbor-metrics", default="reports/m2_5/m2-5-v2-exact-neighbors-full/metrics.json")
    parser.add_argument("--source-cache-dir", default="artifacts/m3_5/cold-source-cache-v1")
    parser.add_argument("--run-id", default="m3-5-v1-content-season-image-cold")
    for cutoff in OUTER_WINDOWS.values():
        root = f"artifacts/m2_9/cache-v1/{cutoff}"
        parser.add_argument(f"--candidates-{_key(cutoff)}", default=f"{root}/expanded-candidates.parquet")
        parser.add_argument(f"--manifest-{_key(cutoff)}", default=f"{root}/candidate-manifest.json")
    args = parser.parse_args(argv)
    bases = {cutoff: (Path(getattr(args, f"candidates_{_key(cutoff)}")), Path(getattr(args, f"manifest_{_key(cutoff)}"))) for cutoff in OUTER_WINDOWS.values()}
    try:
        result = run_m35(
            raw_dir=Path(args.raw_dir),
            transactions_path=Path(args.transactions),
            base_candidates=bases,
            neighbor_metrics_path=Path(args.neighbor_metrics),
            source_cache_dir=Path(args.source_cache_dir),
            output_dir=Path("reports/m3_5") / args.run_id,
            artifact_dir=Path("artifacts/m3_5") / args.run_id,
            run_id=args.run_id,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"M3.5 run failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "run_id": result["run_id"], "retrieval_gate": result["summary"]["retrieval_gate"]["passed"], "elapsed_seconds": result["elapsed_seconds"], "report": result["artifacts"]["report"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
