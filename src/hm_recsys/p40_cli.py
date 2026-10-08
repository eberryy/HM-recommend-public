from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .m4_contract import atomic_json
from .p40 import run
from .p40_contract import RUN_ID, write_contract
from .p40_report import write_manifest, write_report
from .p40_verify import verify


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run preregistered P4.0 Warm/Cold Fusion v2")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/phase4/phase4-p4.0-v1-b0-warm-fusion"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase4"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preregister-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    artifact = args.artifact_dir if args.artifact_dir.is_absolute() else repo / args.artifact_dir
    report = args.report_dir if args.report_dir.is_absolute() else repo / args.report_dir
    write_contract(repo, args.source_root.resolve(), artifact.resolve(), report.resolve())
    if args.preregister_only:
        print(json.dumps({"run_id": RUN_ID, "status": "preregistered"}, indent=2))
        return 0
    if not args.verify_only:
        try:
            result = run(repo_root=repo, source_root=args.source_root.resolve(),
                         artifact_dir=artifact.resolve(), report_dir=report.resolve(),
                         device_name=args.device)
            write_report(report.resolve(), result)
            write_manifest(repo, artifact.resolve(), report.resolve())
        except Exception as exc:
            atomic_json(report / f"P4_0_FAILURE_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json", {
                "stage": "P4.0", "run_id": RUN_ID, "status": "failed_closed",
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "final_week": "not_run",
            })
            raise
    verification = verify(report.resolve())
    atomic_json(report / "P4_0_VERIFICATION.json", verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2), flush=True)
    return 0 if verification["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
