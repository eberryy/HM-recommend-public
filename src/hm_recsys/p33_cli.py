from __future__ import annotations

import argparse
from pathlib import Path

from .p33 import RUN_ID, run_p33
from .p33_report import write_p33
from .p33_verify import verify_p33


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P3.3 multi-view graph-teacher distillation")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase3"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/phase3/phase3-p3.3-v1-multiview-graph-student"),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--reuse-completed", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    report_dir = (repo_root / args.report_dir).resolve() if not args.report_dir.is_absolute() else args.report_dir.resolve()
    artifact_dir = (repo_root / args.artifact_dir).resolve() if not args.artifact_dir.is_absolute() else args.artifact_dir.resolve()
    if not args.verify_only:
        result = run_p33(
            source_root=args.source_root.resolve(),
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            report_dir=report_dir,
            device_name=args.device,
            reuse_completed=args.reuse_completed,
        )
        write_p33(report_dir, result)
    verification = verify_p33(report_dir=report_dir, artifact_dir=artifact_dir)
    print(
        f"P3.3 {RUN_ID}: verification={verification['status']} "
        f"promotion={__import__('json').loads((report_dir / 'P3_3_metrics.json').read_text(encoding='utf-8'))['comparison']['promotion_passed']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
