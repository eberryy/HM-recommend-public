from __future__ import annotations

import argparse
from pathlib import Path

from .m55 import render_existing, run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M5.5 source-aware dual-channel Fusion")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--m54-artifact-dir", type=Path, default=Path("artifacts/m5_4/m5-4-v1-temporal-oof-dual-channel"))
    parser.add_argument("--m54-metrics", type=Path, default=Path("reports/m5_4/M5_4_metrics.json"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/m5_5/m5-5-v1-source-aware-dual-channel-fusion"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/m5_5"))
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--verification", type=Path, default=Path("reports/m5_5/M5_5_VERIFICATION.json"))
    args = parser.parse_args(argv)
    if args.render_only:
        verification = args.verification.resolve() if args.verification.exists() else None
        render_existing(
            metrics_path=(args.report_dir / "M5_5_metrics.json").resolve(),
            report_path=(args.report_dir / "M5_5_FINAL.md").resolve(),
            verification_path=verification,
        )
        return 0
    run(
        source_root=args.source_root.resolve(), m54_artifact_dir=args.m54_artifact_dir.resolve(),
        m54_metrics_path=args.m54_metrics.resolve(), artifact_dir=args.artifact_dir.resolve(),
        report_dir=args.report_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
