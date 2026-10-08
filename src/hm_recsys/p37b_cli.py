from __future__ import annotations

import argparse
import json
from pathlib import Path

from .p37b import RUN_ID, run_with_failure_record
from .p37b_report import write_preregistered_contract
from .p37b_verify import verify_p37b


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed P3.7B time-aware cold reranker")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase3"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/phase3/phase3-p3.7b-v1-time-aware-hybrid"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    report_dir = (
        args.report_dir.resolve()
        if args.report_dir.is_absolute()
        else (repo_root / args.report_dir).resolve()
    )
    artifact_dir = (
        args.artifact_dir.resolve()
        if args.artifact_dir.is_absolute()
        else (repo_root / args.artifact_dir).resolve()
    )
    if not args.verify_only:
        write_preregistered_contract(
            report_dir, repo_root, args.source_root.resolve(), artifact_dir
        )
        run_with_failure_record(
            source_root=args.source_root.resolve(),
            repo_root=repo_root,
            artifact_dir=artifact_dir,
            report_dir=report_dir,
            device_name=args.device,
        )
    verification = verify_p37b(report_dir=report_dir)
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "verification": verification["status"],
                "decision": verification["recomputed_decision"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if verification["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
