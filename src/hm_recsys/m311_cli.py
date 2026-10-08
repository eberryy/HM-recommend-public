from __future__ import annotations

import argparse
import json
from pathlib import Path

from .m311 import run_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M3.11A image precision and local replacement oracle audit")
    parser.add_argument("--m310-metrics", type=Path, default=Path("reports/m3_10/m3-10-v3-source-specific-image-admission-rank-fix/metrics.json"))
    parser.add_argument("--m33-metrics", type=Path, default=Path(__file__).resolve().parents[2] / 'reports/m3_3/m3-3-v1-cross-season-adaptive-seasonal/metrics.json')
    parser.add_argument("--transactions", type=Path, default=Path(__file__).resolve().parents[2] / 'data/interim/audit/transactions.parquet')
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m3_11/m3-11a-v1-image-precision-oracle-audit"))
    parser.add_argument("--run-id", default="m3-11a-v1-image-precision-oracle-audit")
    args = parser.parse_args()
    result = run_audit(
        m310_metrics_path=args.m310_metrics,
        m33_metrics_path=args.m33_metrics,
        transactions_path=args.transactions,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "recommendation": result["summary"]["recommendation"],
        "elapsed_seconds": result["elapsed_seconds"],
        "report": result["artifacts"]["report"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
