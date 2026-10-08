from __future__ import annotations

import argparse
import json
from pathlib import Path

from .m311b import run_m311b


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M3.11B constrained image replacement ranking")
    parser.add_argument("--m310-metrics", type=Path, default=Path("reports/m3_10/m3-10-v3-source-specific-image-admission-rank-fix/metrics.json"))
    parser.add_argument("--m311a-metrics", type=Path, default=Path("reports/m3_11/m3-11a-v1-image-precision-oracle-audit/metrics.json"))
    parser.add_argument("--m33-metrics", type=Path, default=Path(__file__).resolve().parents[2] / 'reports/m3_3/m3-3-v1-cross-season-adaptive-seasonal/metrics.json')
    parser.add_argument("--transactions", type=Path, default=Path(__file__).resolve().parents[2] / 'data/interim/audit/transactions.parquet')
    parser.add_argument("--output-dir", type=Path, default=Path("reports/m3_11/m3-11b-v5-constrained-local-replacement"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/m3_11/m3-11b-v5-constrained-local-replacement"))
    parser.add_argument("--run-id", default="m3-11b-v5-constrained-local-replacement")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    result = run_m311b(
        m310_metrics_path=args.m310_metrics,
        m311a_metrics_path=args.m311a_metrics,
        m33_metrics_path=args.m33_metrics,
        transactions_path=args.transactions,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        run_id=args.run_id,
        threads=args.threads,
        seed=args.seed,
    )
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "accepted_as_new_baseline": result["summary"]["accepted_as_new_baseline"],
        "mean_delta_map@12": result["summary"]["mean_delta_map@12"],
        "elapsed_seconds": result["elapsed_seconds"],
        "report": result["artifacts"]["report"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
