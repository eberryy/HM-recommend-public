from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .m4_contract import atomic_json, file_identity
from .p33 import RUN_ID, VIEWS


def verify_p33(*, report_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    metrics_path = report_dir / "P3_3_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "status_measured": metrics.get("status") == "measured",
        "run_id": metrics.get("run_id") == RUN_ID,
        "final_week_not_run": metrics.get("final_week") == "not_run",
        "three_equal_weight_views": metrics.get("contract", {}).get("student", {}).get("teacher_view_weights")
        == {view: 1.0 / 3.0 for view in VIEWS},
        "teacher_embedding_not_in_inference": metrics.get("contract", {}).get("student", {}).get(
            "teacher_embeddings_at_inference"
        ) is False,
    }
    artifact_checks: dict[str, bool] = {}
    for cutoff, teacher in metrics["teachers"]["cutoffs"].items():
        neighbor_views = {}
        neighbor_lookups = {}
        for view in VIEWS:
            neighbor_path = artifact_dir / "teachers-v1" / cutoff / f"neighbors_{view}.npz"
            neighbor_value = np.load(neighbor_path)
            neighbor_views[view] = {
                "anchors": np.asarray(neighbor_value["anchors"]),
                "neighbors": np.asarray(neighbor_value["neighbors"]),
            }
            neighbor_lookups[view] = {
                int(anchor): row
                for row, anchor in enumerate(neighbor_views[view]["anchors"].tolist())
            }
        for view in VIEWS:
            path = artifact_dir / "teachers-v1" / cutoff / f"relations_{view}.npz"
            identity = file_identity(path)
            declared = teacher["relations"][view]
            key = f"teacher:{cutoff}:{view}"
            artifact_checks[key] = identity["sha256"] == declared["sha256"] and identity["bytes"] == declared["bytes"]
            value = np.load(path)
            artifact_checks[key + ":shape"] = all(
                len(value[name]) == len(value["anchor"])
                for name in ("positive", "negative", "teacher_cosine", "teacher_rank")
            )
            artifact_checks[key + ":identity"] = bool(
                np.all(value["anchor"] != value["positive"])
                and np.all(value["anchor"] != value["negative"])
                and np.all(value["positive"] != value["negative"])
            )
            exclusion_ok = True
            anchors = np.asarray(value["anchor"])
            negatives = np.asarray(value["negative"])
            for anchor in np.unique(anchors).tolist():
                excluded_parts = []
                for other_view in VIEWS:
                    row = neighbor_lookups[other_view].get(int(anchor))
                    if row is not None:
                        values = neighbor_views[other_view]["neighbors"][row]
                        excluded_parts.append(values[values >= 0])
                excluded = np.unique(np.concatenate(excluded_parts)) if excluded_parts else np.empty(0, dtype=np.int32)
                if np.isin(negatives[anchors == anchor], excluded).any():
                    exclusion_ok = False
                    break
            artifact_checks[key + ":negative_exclusion"] = exclusion_ok
    for role, assets in metrics["students"]["assets"].items():
        for label, manifest in assets.items():
            if role == "training":
                path = artifact_dir / "students-v1" / "training" / label / "catalog_embeddings.float16.npy"
            else:
                path = artifact_dir / "students-v1" / "outer" / label / "catalog_embeddings.float16.npy"
            value = np.load(path, mmap_mode="r")
            key = f"student:{role}:{label}"
            identity = file_identity(path)
            artifact_checks[key] = (
                tuple(value.shape) == (105_542, 128)
                and str(value.dtype) == "float16"
                and identity["sha256"] == manifest["encoding"]["artifact"]["sha256"]
                and manifest["student_inference_audit"]["content_only"] is True
            )
    recomputed = {}
    for metric in ("strict_recall200", "sparse_recall200", "density20", "mrr", "top200_to_top20"):
        old = float(np.mean([
            metrics["comparison"]["windows"][window][metric]["single_teacher"]
            for window in metrics["comparison"]["windows"]
        ]))
        new = float(np.mean([
            metrics["comparison"]["windows"][window][metric]["multiview_teacher"]
            for window in metrics["comparison"]["windows"]
        ]))
        declared = metrics["comparison"]["means"][metric]
        recomputed[metric] = {
            "single_teacher": old,
            "multiview_teacher": new,
            "matches": abs(old - declared["single_teacher"]) <= 1e-15
            and abs(new - declared["multiview_teacher"]) <= 1e-15
            and abs((new - old) - declared["delta"]) <= 1e-15,
        }
    checks["all_artifacts"] = all(artifact_checks.values())
    checks["all_mean_metrics_recomputed"] = all(value["matches"] for value in recomputed.values())
    result = {
        "schema_version": "phase3-p3.3-verification-v1",
        "stage": "P3.3 independent verification",
        "status": "passed" if all(checks.values()) else "failed",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "artifact_checks": artifact_checks,
        "recomputed_means": recomputed,
        "inputs": {"metrics": file_identity(metrics_path)},
        "final_week": "not_run",
    }
    atomic_json(report_dir / "P3_3_VERIFICATION.json", result)
    if result["status"] != "passed":
        raise RuntimeError(f"P3.3 verification failed: {checks}")
    return result
