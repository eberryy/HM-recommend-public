from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .m2 import M2Config
from .m29 import M29Window, REQUIRED_CUTOFFS, run_m29


KEYS=("20200527","20200624","20200722","20200819")
CUTOFFS=dict(zip(KEYS,REQUIRED_CUTOFFS,strict=True))
DEFAULT_DIRS={
 "20200527":"m1-5-v13-build-20200527-10pct-optimistic",
 "20200624":"m1-5-v14-build-20200624-10pct-optimistic",
 "20200722":"m1-5-v12-build-20200722-10pct-optimistic",
 "20200819":"m1-5-v2-build-20200819-10pct",
}


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="Run M2.9 Item2Vec expanded-pool LambdaRank")
    parser.add_argument("--raw-dir",default="data/raw");parser.add_argument("--work-dir",default="data/interim/audit")
    parser.add_argument("--transactions",default="data/interim/audit/transactions.parquet")
    parser.add_argument("--m21-metrics",default="reports/m2_1/m2-1-v1-dev-rolling-10pct/metrics.json")
    parser.add_argument("--m28-metrics",default="reports/m2_8/m2-8-v1-item2vec-dev/metrics.json")
    parser.add_argument("--source-cache-dir",default="artifacts/m2_9/item2vec-source-v1")
    parser.add_argument("--cache-dir",default="artifacts/m2_9/cache-v1")
    parser.add_argument("--run-id",default="m2-9-v1-item2vec-rank")
    for key in KEYS:
        root=f"artifacts/m1_5/{DEFAULT_DIRS[key]}"
        parser.add_argument(f"--candidates-{key}",default=f"{root}/candidate_features.parquet")
        parser.add_argument(f"--manifest-{key}",default=f"{root}/manifest.json")
    args=parser.parse_args(argv)
    windows=[M29Window(cutoff=CUTOFFS[key],candidate_path=Path(getattr(args,f"candidates_{key}")),
        manifest_path=Path(getattr(args,f"manifest_{key}"))) for key in KEYS]
    config=M2Config(history_weeks=12,metric_k=12,candidate_k=300,num_boost_round=200,
        learning_rate=0.05,num_leaves=31,min_data_in_leaf=100,threads=8,
        prediction_chunk_vectors=64,seed=20260824,evaluation_role="development")
    try:
        result=run_m29(raw_dir=Path(args.raw_dir),work_dir=Path(args.work_dir),
            transactions_path=Path(args.transactions),m21_metrics_path=Path(args.m21_metrics),
            m28_metrics_path=Path(args.m28_metrics),windows=windows,
            source_cache_dir=Path(args.source_cache_dir),cache_dir=Path(args.cache_dir),
            output_dir=Path("reports/m2_9")/args.run_id,
            artifact_dir=Path("artifacts/m2_9")/args.run_id,config=config,run_id=args.run_id)
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as error:
        print(f"M2.9 run failed: {error}",file=sys.stderr);return 2
    print(json.dumps({"status":result["status"],"run_id":result["run_id"],
        "gate":result["development_summary"]["selection_gate"],
        "elapsed_seconds":result["elapsed_seconds"],"report":result["artifacts"]["report"]},
        indent=2,ensure_ascii=False));return 0


if __name__=="__main__": raise SystemExit(main())
