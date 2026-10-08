from __future__ import annotations

import argparse
import json
from pathlib import Path

from .m28 import run_m28


def main() -> int:
    parser=argparse.ArgumentParser(description="Run M2.8 cutoff-safe Item2Vec retrieval")
    parser.add_argument("--transactions",default="data/interim/audit/transactions.parquet")
    parser.add_argument("--reference-metrics",default="reports/m2_1/m2-1-v1-dev-rolling-10pct/metrics.json")
    parser.add_argument("--image-metrics",default="reports/m2_5/m2-5-v4-image-retrieval-dev-headtail/metrics.json")
    parser.add_argument("--dev-a-candidates",default="artifacts/m1_5/m1-5-v12-build-20200722-10pct-optimistic/candidate_features.parquet")
    parser.add_argument("--dev-a-manifest",default="artifacts/m1_5/m1-5-v12-build-20200722-10pct-optimistic/manifest.json")
    parser.add_argument("--dev-b-candidates",default="artifacts/m1_5/m1-5-v2-build-20200819-10pct/candidate_features.parquet")
    parser.add_argument("--dev-b-manifest",default="artifacts/m1_5/m1-5-v2-build-20200819-10pct/manifest.json")
    parser.add_argument("--run-id",default="m2-8-v1-item2vec-dev")
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=parser.parse_args()
    metrics=run_m28(
        transactions_path=Path(args.transactions),reference_metrics_path=Path(args.reference_metrics),
        image_metrics_path=Path(args.image_metrics),
        windows=[("2020-07-22",Path(args.dev_a_candidates),Path(args.dev_a_manifest)),
                 ("2020-08-19",Path(args.dev_b_candidates),Path(args.dev_b_manifest))],
        output_dir=Path("artifacts/m2_8")/args.run_id,
        report_dir=Path("reports/m2_8")/args.run_id,device=args.device)
    print(json.dumps({"run_id":metrics["run_id"],"gate":metrics["development_gate"],
        "elapsed_seconds":metrics["elapsed_seconds"]},indent=2,ensure_ascii=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
