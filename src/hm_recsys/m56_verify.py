from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb

from .m33 import ROLLING_PROTOCOL
from .m4_contract import FINAL_CUTOFF, atomic_json, file_identity
from .m55 import DUAL_FEATURES
from .m56 import (
    FIXED_ROUNDS,
    RUN_ID as M56_RUN_ID,
    SOURCE_FEATURES,
    VARIANTS,
    _deployment_gate,
    _mechanism_summary,
)


RUN_ID = "m5-6-v1-independent-verification"


def _identity_matches(identity: dict[str, Any]) -> dict[str, Any]:
    path = Path(identity["path"])
    actual = file_identity(path) if path.is_file() else None
    return {
        "path": str(path),
        "expected_bytes": int(identity["bytes"]),
        "actual_bytes": actual["bytes"] if actual else None,
        "expected_sha256": identity["sha256"],
        "actual_sha256": actual["sha256"] if actual else None,
        "passed": actual == identity,
    }


def _collect_identities(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}

    def add(identity: dict[str, Any]) -> None:
        expected[str(Path(identity["path"]).resolve())] = identity

    for identity in metrics["inputs"].values():
        add(identity)
    for identity in metrics["audits"].values():
        add(identity)
    for row in metrics["windows"].values():
        add(row["candidate_feature_identity"])
        add(row["predictions"])
        add(row["training"]["category_maps_A"])
        add(row["training"]["category_maps_BC"])
        for key in ("A", "B_inner", "B_outer", "C"):
            add(row["training"][key]["model"])
        for identity in row["control_inputs"].values():
            add(identity)
    return [_identity_matches(identity) for identity in expected.values()]


def verify(*, metrics_path: Path, report_path: Path, output_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    identities = _collect_identities(metrics)
    model_checks: dict[str, Any] = {}
    for window, row in metrics["windows"].items():
        checks: dict[str, bool] = {}
        for name in ("A", "B_outer", "C"):
            model = lgb.Booster(model_file=row["training"][name]["model"]["path"])
            checks[f"{name}_feature_schema"] = model.feature_name() == DUAL_FEATURES
        checks["A_exactly_40_trees"] = (
            row["training"]["A"]["best_iteration"] == FIXED_ROUNDS
            and row["training"]["A"]["actual_num_trees"] == FIXED_ROUNDS
        )
        checks["C_exactly_40_trees"] = (
            row["training"]["C"]["best_iteration"] == FIXED_ROUNDS
            and row["training"]["C"]["actual_num_trees"] == FIXED_ROUNDS
        )
        checks["B_rounds_from_inner_only"] = (
            row["training"]["B_outer"]["best_iteration"]
            == row["training"]["B_inner"]["best_iteration"]
        )
        checks["A_sampling_matches_M5.5"] = row["sampling_audit"]["A_matches_M5.5_control"] is True
        checks["B_C_sample_identity"] = row["sampling_audit"]["B_C_sample_identity"] is True
        checks["negative_cap"] = (
            row["sampling_audit"]["conditional_outer"]["audit"]["max_total_negative_per_positive"] <= 30
        )
        checks["no_cold_positive_cap"] = (
            row["sampling_audit"]["conditional_outer"]["audit"]["max_cold_negatives_in_no_cold_positive_group"] <= 3
        )
        model_checks[window] = checks

    recomputed_mechanism = _mechanism_summary(metrics["windows"])
    recomputed_deployment = _deployment_gate(metrics["windows"], recomputed_mechanism)
    final_safe = (
        metrics["final_week"] == "not_run"
        and FINAL_CUTOFF not in metrics["windows"]
        and all(
            protocol["outer_validation"] < FINAL_CUTOFF
            and metrics["windows"][window]["final_week"] == "not_run"
            for window, protocol in ROLLING_PROTOCOL.items()
        )
    )
    checks = {
        "authoritative_run_id": metrics.get("run_id") == M56_RUN_ID,
        "status_measured": metrics.get("status") == "measured",
        "exact_four_development_windows": set(metrics["windows"]) == set(ROLLING_PROTOCOL),
        "exact_three_new_variants": set(VARIANTS) == {
            "A_current_sampling_fixed40",
            "B_conditional_sampling_temporal_early_stopping",
            "C_conditional_sampling_fixed40",
        },
        "frozen_23_source_features": len(SOURCE_FEATURES) == 23,
        "all_models_and_artifacts_rehashed": all(row["passed"] for row in identities),
        "all_window_training_contracts": all(
            all(value for value in row.values()) for row in model_checks.values()
        ),
        "mechanism_gate_recomputed_identically": recomputed_mechanism == metrics["summary"]["mechanism"],
        "deployment_gate_recomputed_identically": recomputed_deployment == metrics["summary"]["deployment"],
        "human_report_present": report_path.is_file(),
        "final_week_not_run": final_safe,
    }
    report_identity = file_identity(report_path) if report_path.is_file() else None
    result = {
        "schema_version": "m5.6-independent-verification-v1",
        "stage": "M5.6 verification",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "meaning": "evidence integrity passing does not mean the mechanism or deployment gate passed",
        "checks": checks,
        "window_training_contracts": model_checks,
        "recomputed_mechanism": recomputed_mechanism,
        "recomputed_deployment": recomputed_deployment,
        "rehash": {
            "files": len(identities),
            "bytes": sum(int(row["actual_bytes"] or 0) for row in identities),
            "identities": identities,
            "metrics": file_identity(metrics_path),
            "report": report_identity,
        },
        "final_week": "not_run",
    }
    atomic_json(output_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently verify M5.6 evidence")
    parser.add_argument("--metrics", type=Path, default=Path("reports/m5_6/metrics.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/m5_6/M5_6_FINAL.md"))
    parser.add_argument("--output", type=Path, default=Path("reports/m5_6/M5_6_VERIFICATION.json"))
    args = parser.parse_args(argv)
    result = verify(
        metrics_path=args.metrics.resolve(), report_path=args.report.resolve(),
        output_path=args.output.resolve(),
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
