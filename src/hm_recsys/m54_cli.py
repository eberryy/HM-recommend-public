from __future__ import annotations

import argparse
from pathlib import Path

from .m54 import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M5.4 temporal OOF dual-channel materialization")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--m4-artifact-dir", type=Path, default=Path("artifacts/m4/m4-v1-supervised-cold-representation"))
    parser.add_argument("--m5-artifact-dir", type=Path, default=Path("artifacts/m5/m5-v1-cold-expert-admission"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/m5_4/m5-4-v1-temporal-oof-dual-channel"))
    parser.add_argument("--m53-metrics", type=Path, default=Path("reports/m5_3/M5_3_metrics.json"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/m5_4"))
    args = parser.parse_args(argv)
    run(
        source_root=args.source_root.resolve(), m4_artifact_dir=args.m4_artifact_dir.resolve(),
        m5_artifact_dir=args.m5_artifact_dir.resolve(), artifact_dir=args.artifact_dir.resolve(),
        m53_metrics_path=args.m53_metrics.resolve(), report_dir=args.report_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
