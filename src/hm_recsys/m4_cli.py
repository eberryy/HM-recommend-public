from __future__ import annotations

import argparse
import json
from pathlib import Path

from .m4_contract import run_m40
from .m4_report import write_m40, write_m41, write_m42, write_m43
from .m4_retrieval import run_m43
from .m4_student import run_m42
from .m4_teacher import build_teacher_relations


def _existing(path: Path) -> dict | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "measured" or value.get("final_week") != "not_run":
        raise ValueError(f"existing M4 evidence is not reusable: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bounded M4 supervised cold-item representation experiment")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--report-dir", type=Path, default=Path("reports/m4"))
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=Path("artifacts/m4/m4-v1-supervised-cold-representation"),
    )
    parser.add_argument("--stage", choices=("m40", "m41", "m42", "m43", "all"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--reuse-completed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_dir = args.report_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    selected = ["m40", "m41", "m42", "m43"] if args.stage == "all" else [args.stage]
    report_dir.mkdir(parents=True, exist_ok=True)
    if "m40" in selected:
        path = report_dir / "M4_0_metrics.json"
        result = _existing(path) if args.reuse_completed else None
        if result is None:
            if path.exists():
                raise FileExistsError(f"refusing to overwrite formal M4.0 evidence: {path}")
            result = run_m40(source_root=args.source_root.resolve(), output_dir=report_dir)
        write_m40(report_dir, result)
    if "m41" in selected:
        path = report_dir / "M4_1_metrics.json"
        result = _existing(path) if args.reuse_completed else None
        if result is None:
            if path.exists():
                raise FileExistsError(f"refusing to overwrite formal M4.1 evidence: {path}")
            result = build_teacher_relations(
                source_root=args.source_root.resolve(),
                artifact_dir=artifact_dir / "teacher-v1",
                output_dir=report_dir,
            )
        write_m41(report_dir, result)
    if "m42" in selected:
        path = report_dir / "M4_2_metrics.json"
        result = _existing(path) if args.reuse_completed else None
        if result is None:
            if path.exists():
                raise FileExistsError(f"refusing to overwrite formal M4.2 evidence: {path}")
            result = run_m42(
                source_root=args.source_root.resolve(),
                teacher_dir=artifact_dir / "teacher-v1",
                artifact_dir=artifact_dir / "student-v1",
                output_dir=report_dir,
                device_name=args.device,
            )
        write_m42(report_dir, result)
    if "m43" in selected:
        path = report_dir / "M4_3_metrics.json"
        result = _existing(path) if args.reuse_completed else None
        if result is None:
            if path.exists():
                raise FileExistsError(f"refusing to overwrite formal M4.3 evidence: {path}")
            result = run_m43(
                source_root=args.source_root.resolve(),
                student_dir=artifact_dir / "student-v1",
                static_catalog_dir=artifact_dir / "student-v1" / "static_catalog",
                artifact_dir=artifact_dir / "retrieval-v1",
                output_dir=report_dir,
                device_name=args.device,
            )
        write_m43(report_dir, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
