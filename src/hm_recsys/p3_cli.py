from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .m4_contract import atomic_json, file_identity
from .p3 import RUN_ID, run_p30, run_p31, run_p32
from .p3_report import write_p30, write_p31, write_p32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3 Version A candidate-aware cold reranking")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase3"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/phase3/phase3-v1-candidate-aware-history"))
    parser.add_argument("--stage", choices=("p30", "p31", "p32", "all"), default="all")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--reuse-completed", action="store_true")
    return parser


def _contract(report_dir: Path) -> None:
    contract = {
        "schema_version": "phase3-version-a-round1-contract-v1",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": ["P3.0 freeze/reuse gate", "P3.1 Student-space Top200", "P3.2 candidate-aware reranker"],
        "frozen": {"cold_event_threshold": 5, "history_n": 20, "coarse_k": 200, "final_week": "not_run"},
        "primary_model": {
            "attention_interaction": "[z_c,z_h,z_c*z_h,abs(z_c-z_h),log1p(days_since_purchase)]",
            "candidate_conditioned_user": "sum(softmax(attention_logits) * z_h)",
            "scorer_interaction": "[u(c),z_c,u(c)*z_c,abs(u(c)-z_c)]",
            "hidden_dim": 128,
            "loss": "same-user pairwise logistic/BPR",
            "maximum_negatives_per_positive": 50,
        },
        "prohibited": [
            "two-tower", "user tower", "DeepWalk teacher", "LightGCN", "PCA",
            "FashionCLIP retraining", "M4 Student retraining", "Warm+Cold final fusion",
            "M5.5/M5.6 rescue search", "K_admit", "local replacement", "final week",
        ],
        "code": {
            "runner": file_identity(Path(__file__).with_name("p3.py")),
            "model": file_identity(Path(__file__).with_name("p3_model.py")),
            "report": file_identity(Path(__file__).with_name("p3_report.py")),
        },
        "final_week": "not_run",
    }
    atomic_json(report_dir / "experiment_contract.json", contract)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = args.source_root.resolve()
    repo_root = args.repo_root.resolve()
    report_dir = args.report_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    m4_artifact_dir = repo_root / "artifacts" / "m4" / "m4-v1-supervised-cold-representation"
    m5_artifact_dir = repo_root / "artifacts" / "m5" / "m5-v1-cold-expert-admission"
    report_dir.mkdir(parents=True, exist_ok=True)
    _contract(report_dir)
    selected = ["p30", "p31", "p32"] if args.stage == "all" else [args.stage]
    if "p30" in selected:
        result = run_p30(
            source_root=source_root,
            repo_root=repo_root,
            m4_artifact_dir=m4_artifact_dir,
            m5_artifact_dir=m5_artifact_dir,
            artifact_dir=artifact_dir / "freeze-v1",
            report_dir=report_dir,
        )
        write_p30(report_dir, result)
    if "p31" in selected:
        if not (report_dir / "P3_0_metrics.json").is_file():
            raise FileNotFoundError("P3.1 requires completed P3.0 evidence")
        result = run_p31(
            source_root=source_root,
            repo_root=repo_root,
            m4_artifact_dir=m4_artifact_dir,
            m5_artifact_dir=m5_artifact_dir,
            artifact_dir=artifact_dir,
            report_dir=report_dir,
            device_name=args.device,
            reuse_completed=args.reuse_completed,
        )
        write_p31(report_dir, result)
    if "p32" in selected:
        if not (report_dir / "P3_1_metrics.json").is_file():
            raise FileNotFoundError("P3.2 requires completed P3.1 evidence")
        result = run_p32(
            source_root=source_root,
            repo_root=repo_root,
            m4_artifact_dir=m4_artifact_dir,
            m5_artifact_dir=m5_artifact_dir,
            artifact_dir=artifact_dir,
            report_dir=report_dir,
            device_name=args.device,
        )
        write_p32(report_dir, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
