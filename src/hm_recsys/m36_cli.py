from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m36 import run_m36


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M3.6 cold-truth retrieval funnel")
    parser.add_argument(
        "--m35-metrics",
        default="reports/m3_5/m3-5-v2-content-season-image-cold/metrics.json",
    )
    parser.add_argument(
        "--neighbor-metrics",
        default="reports/m2_5/m2-5-v2-exact-neighbors-full/metrics.json",
    )
    parser.add_argument("--run-id", default="m3-6-v1-cold-truth-funnel")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    try:
        result = run_m36(
            m35_metrics_path=Path(args.m35_metrics),
            neighbor_metrics_path=Path(args.neighbor_metrics),
            output_dir=Path("reports/m3_6") / args.run_id,
            artifact_dir=Path("artifacts/m3_6") / args.run_id,
            run_id=args.run_id,
            device=args.device,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"M3.6 run failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
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
