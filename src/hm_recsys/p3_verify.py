from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .m4_contract import atomic_json, file_identity
from .p3 import FINAL_CUTOFF, K_VALUES, ROLLING_PROTOCOL, _json


def _close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _ranks_from_scores(candidates: dict[str, np.ndarray], scores: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(scores), dtype=np.uint16)
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and candidates["user_index"][end] == candidates["user_index"][cursor]:
            end += 1
        order = sorted(
            range(cursor, end),
            key=lambda index: (
                -float(scores[index]),
                int(candidates["rank"][index]),
                int(candidates["catalog_row"][index]),
            ),
        )
        ranks[np.asarray(order, dtype=np.int64)] = np.arange(1, len(order) + 1, dtype=np.uint16)
        cursor = end
    return ranks


def verify(*, repo_root: Path, report_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    p30 = _json(report_dir / "P3_0_metrics.json")
    p31 = _json(report_dir / "P3_1_metrics.json")
    p32 = _json(report_dir / "P3_2_metrics.json")
    contract = _json(report_dir / "experiment_contract.json")
    checks: dict[str, bool] = {
        "all_status_measured": all(value.get("status") == "measured" for value in (p30, p31, p32)),
        "all_final_week_not_run": all(value.get("final_week") == "not_run" for value in (p30, p31, p32, contract)),
        "all_outer_cutoffs_before_final": all(
            protocol["outer_validation"] < FINAL_CUTOFF for protocol in ROLLING_PROTOCOL.values()
        ),
        "p30_gate_passed": bool(p30.get("gate_passed")),
    }
    windows: dict[str, Any] = {}
    for window, row in p32["windows"].items():
        candidate_path = artifact_dir / "coarse-v1" / "outer" / window / "candidates_top200.npz"
        history_path = artifact_dir / "coarse-v1" / "outer" / window / "histories.npz"
        manifest_path = artifact_dir / "coarse-v1" / "outer" / window / "manifest.json"
        model_manifest_path = artifact_dir / "models-v1" / window / "manifest.json"
        manifest = _json(manifest_path)
        model_manifest = _json(model_manifest_path)
        holders = np.load(candidate_path)
        candidates = {name: np.asarray(holders[name]) for name in holders.files}
        histories_npz = np.load(history_path)
        histories = {name: np.asarray(histories_npz[name]) for name in histories_npz.files}
        scores_path = Path(row["scoring"]["scores"]["path"])
        attention_path = Path(row["scoring"]["attention"]["path"])
        scores = np.load(scores_path, mmap_mode="r")
        attention = np.load(attention_path, mmap_mode="r")
        rerank = _ranks_from_scores(candidates, np.asarray(scores))
        identities = candidates["user_index"].astype(np.int64) * 105_542 + candidates["catalog_row"]
        active_users = np.unique(candidates["user_index"])
        per_user_budget = np.bincount(candidates["user_index"], minlength=len(histories["catalog_row"]))
        ranks_continuous = True
        cursor = 0
        while cursor < len(rerank):
            end = cursor + 1
            while end < len(rerank) and candidates["user_index"][end] == candidates["user_index"][cursor]:
                end += 1
            if not np.array_equal(np.sort(rerank[cursor:end]), np.arange(1, end - cursor + 1)):
                ranks_continuous = False
                break
            cursor = end
        masks = histories["mask"][candidates["user_index"]].astype(bool)
        attention_float = np.asarray(attention, dtype=np.float32)
        valid_sums = np.sum(np.where(masks, attention_float, 0.0), axis=1)
        padded_max = float(np.max(np.abs(attention_float[~masks]))) if np.any(~masks) else 0.0
        metric_checks: dict[str, bool] = {}
        for variant, ranks in (("coarse_student_order", candidates["rank"]), ("candidate_aware_order", rerank)):
            reported = row["metrics"][variant]
            for k in K_VALUES:
                selected = ranks <= k
                rows_at_k = int(np.count_nonzero(selected))
                positives_at_k = int(np.count_nonzero(selected & (candidates["target"] == 1)))
                density = positives_at_k / max(rows_at_k, 1)
                metric_checks[f"{variant}_density_{k}"] = (
                    rows_at_k == reported["at_k"][str(k)]["candidate_rows"]
                    and positives_at_k == reported["at_k"][str(k)]["positive_rows"]
                    and _close(density, reported["at_k"][str(k)]["positive_density"])
                )
        artifact_checks = {
            "candidate_sha": file_identity(candidate_path)["sha256"] == manifest["artifacts"]["candidates"]["sha256"],
            "history_sha": file_identity(history_path)["sha256"] == manifest["artifacts"]["histories"]["sha256"],
            "score_sha": file_identity(scores_path)["sha256"] == row["scoring"]["scores"]["sha256"],
            "attention_sha": file_identity(attention_path)["sha256"] == row["scoring"]["attention"]["sha256"],
            "outer_model_sha": file_identity(Path(row["outer_training"]["model"]["path"]))["sha256"] == row["outer_training"]["model"]["sha256"],
            "model_manifest_sha": file_identity(model_manifest_path)["sha256"] == row["model_manifest"]["sha256"],
        }
        window_checks = {
            "candidate_rows_match_scores": len(candidates["user_index"]) == len(scores) == len(attention),
            "candidate_identity_unique": len(np.unique(identities)) == len(identities),
            "active_user_budget_200": bool(np.all(per_user_budget[active_users] == 200)),
            "rerank_continuous": ranks_continuous,
            "attention_valid_sum_one": bool(np.allclose(valid_sums, 1.0, atol=2e-3)),
            "attention_padding_zero": padded_max == 0.0,
            "top200_positive_parity": (
                row["metrics"]["coarse_student_order"]["at_k"]["200"]["positive_rows"]
                == row["metrics"]["candidate_aware_order"]["at_k"]["200"]["positive_rows"]
            ),
            "outer_cutoff_safe": row["cutoff"] < FINAL_CUTOFF,
            **artifact_checks,
            **metric_checks,
        }
        windows[window] = {
            "checks": window_checks,
            "passed": all(window_checks.values()),
            "attention_valid_sum_min": float(valid_sums.min()),
            "attention_valid_sum_max": float(valid_sums.max()),
            "attention_padding_abs_max": padded_max,
            "candidate_rows": len(scores),
        }
    checks["all_window_artifact_and_metric_checks"] = all(row["passed"] for row in windows.values())
    checks["reported_gate_recomputes_false"] = p32["mechanism_gate"]["gate_passed"] is False
    artifact_manifest = {
        "schema_version": "phase3-version-a-artifact-index-v1",
        "run_id": p32["run_id"],
        "p3_0_freeze_manifest": file_identity(artifact_dir / "freeze-v1" / "manifest.json"),
        "p3_1_assets": {
            key: row["artifacts"] for key, row in p31["all_assets"].items()
        },
        "p3_2_scoring_and_model_manifests": {
            window: {"scoring": row["scoring"], "model_manifest": row["model_manifest"]}
            for window, row in p32["windows"].items()
        },
        "final_week": "not_run",
    }
    model_manifest = {
        "schema_version": "phase3-version-a-model-index-v1",
        "run_id": p32["run_id"],
        "architecture": {
            "embedding_dim": 128,
            "hidden_dim": 128,
            "candidate_aware_history_items": 20,
            "loss": "same-user pairwise logistic/BPR",
        },
        "windows": {
            window: {
                "selected_inner_epoch": row["inner_training"]["selected_epoch"],
                "inner_model": row["inner_training"]["model"],
                "outer_model": row["outer_training"]["model"],
                "manifest": row["model_manifest"],
            }
            for window, row in p32["windows"].items()
        },
        "mechanism_gate": p32["mechanism_gate"],
        "final_week": "not_run",
    }
    atomic_json(report_dir / "artifact_manifest.json", artifact_manifest)
    atomic_json(report_dir / "model_manifest.json", model_manifest)
    result = {
        "schema_version": "phase3-version-a-round1-verification-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "windows": windows,
        "inputs": {
            "p30": file_identity(report_dir / "P3_0_metrics.json"),
            "p31": file_identity(report_dir / "P3_1_metrics.json"),
            "p32": file_identity(report_dir / "P3_2_metrics.json"),
            "contract": file_identity(report_dir / "experiment_contract.json"),
            "artifact_manifest": file_identity(report_dir / "artifact_manifest.json"),
            "model_manifest": file_identity(report_dir / "model_manifest.json"),
        },
        "final_week": "not_run",
    }
    atomic_json(report_dir / "P3_VERIFICATION.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 3 Version A evidence")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase3"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/phase3/phase3-v1-candidate-aware-history"))
    args = parser.parse_args(argv)
    result = verify(
        repo_root=args.repo_root.resolve(),
        report_dir=args.report_dir.resolve(),
        artifact_dir=args.artifact_dir.resolve(),
    )
    print(json.dumps({"status": result["status"], "checks": result["checks"]}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
