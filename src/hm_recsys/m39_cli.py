from __future__ import annotations

import argparse
import json
from pathlib import Path

from .m39 import run_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the read-only M3.9 image-source ranking audit")
    parser.add_argument(
        "--m38-metrics",
        type=Path,
        default=Path("reports/m3_8/m3-8-v3-image-source-factorial-ranking/metrics.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/m3_9/m3-9-v1-image-positive-ranking-audit"),
    )
    args = parser.parse_args()
    result = run_audit(m38_metrics_path=args.m38_metrics, output_dir=args.output_dir)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"], "output": result["artifacts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
