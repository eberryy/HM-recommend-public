from __future__ import annotations

import argparse
from pathlib import Path

from .m56 import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M5.6 Cold supervision suppression diagnostic")
    parser.add_argument("--m54-metrics", type=Path, default=Path("reports/m5_4/M5_4_metrics.json"))
    parser.add_argument(
        "--m54-artifact-dir", type=Path,
        default=Path("artifacts/m5_4/m5-4-v1-temporal-oof-dual-channel"),
    )
    parser.add_argument("--m55-metrics", type=Path, default=Path("reports/m5_5/M5_5_metrics.json"))
    parser.add_argument(
        "--m55-verification", type=Path, default=Path("reports/m5_5/M5_5_VERIFICATION.json"),
    )
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=Path("artifacts/m5_6/m5-6-v1-cold-supervision-suppression"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports/m5_6"))
    args = parser.parse_args(argv)
    run(
        m54_metrics_path=args.m54_metrics.resolve(),
        m54_artifact_dir=args.m54_artifact_dir.resolve(),
        m55_metrics_path=args.m55_metrics.resolve(),
        m55_verification_path=args.m55_verification.resolve(),
        artifact_dir=args.artifact_dir.resolve(),
        report_dir=args.report_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
