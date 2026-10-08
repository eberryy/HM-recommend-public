from __future__ import annotations

import argparse
from pathlib import Path

from .m5_data import run_m50
from .m5_model import run_m51_m52


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bounded M5 cold-expert experiment")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report-dir", type=Path, default=Path("reports/m5"))
    parser.add_argument(
        "--m4-artifact-dir", type=Path,
        default=Path("artifacts/m4/m4-v1-supervised-cold-representation"),
    )
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=Path("artifacts/m5/m5-v1-cold-expert-admission"),
    )
    parser.add_argument("--stage", choices=("m50", "m51-m52"), default="m50")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_dir = args.report_dir.resolve()
    if args.stage == "m50":
        metrics = report_dir / "M5_0_metrics.json"
        if metrics.exists():
            raise FileExistsError(f"refusing to overwrite formal M5.0 evidence: {metrics}")
        run_m50(
            source_root=args.source_root.resolve(),
            m4_artifact_dir=args.m4_artifact_dir.resolve(),
            artifact_dir=args.artifact_dir.resolve(),
            report_dir=report_dir,
            device_name=args.device,
        )
    else:
        metrics = report_dir / "M5_2_metrics.json"
        if metrics.exists():
            raise FileExistsError(f"refusing to overwrite formal M5.1/M5.2 evidence: {metrics}")
        run_m51_m52(
            source_root=args.source_root.resolve(),
            m4_artifact_dir=args.m4_artifact_dir.resolve(),
            artifact_dir=args.artifact_dir.resolve(),
            report_dir=report_dir,
            device_name=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
