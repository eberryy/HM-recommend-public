from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .m25_retrieval import _sha256


def iter_identities(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        if set(("path", "bytes", "sha256")).issubset(value):
            yield value
        for child in value.values():
            yield from iter_identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_identities(child)


def verify_identity(identity: dict[str, Any]) -> None:
    path = Path(identity["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(identity["bytes"]):
        raise ValueError(f"artifact byte-size drift: {path}")
    if _sha256(path) != identity["sha256"]:
        raise ValueError(f"artifact SHA256 drift: {path}")


def verify_m4(report_dir: Path) -> dict[str, Any]:
    metric_paths = [report_dir / f"M4_{stage}_metrics.json" for stage in range(4)]
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    if any(value.get("status") != "measured" for value in metrics):
        raise ValueError("all M4.0-M4.3 stages must be measured")
    if any(value.get("final_week") != "not_run" for value in metrics):
        raise ValueError("M4 final-week boundary drift")
    unique_identities: dict[tuple[str, int, str], dict[str, Any]] = {}
    for value in metrics:
        for identity in iter_identities(value):
            key = (str(Path(identity["path"]).resolve()), int(identity["bytes"]), identity["sha256"])
            unique_identities[key] = identity
    for identity in unique_identities.values():
        verify_identity(identity)

    student = metrics[2]
    embedding_count = 0
    for window in student["windows"].values():
        for variant in window["variants"].values():
            path = Path(variant["encoding"]["artifact"]["path"])
            array = np.load(path, mmap_mode="r")
            if array.shape != (105_542, 128) or str(array.dtype) != "float16":
                raise ValueError(f"student embedding schema drift: {path}")
            sample = np.asarray(array[::101], dtype=np.float32)
            norms = np.linalg.norm(sample, axis=1)
            if not np.isfinite(sample).all() or np.min(norms) < 0.995 or np.max(norms) > 1.005:
                raise ValueError(f"student embedding finite/norm drift: {path}")
            embedding_count += 1

    retrieval = metrics[3]
    candidate_artifact_count = 0
    for window in retrieval["windows"].values():
        for variant in window["variants"].values():
            array = np.load(variant["artifact"]["path"])
            users = np.asarray(array["user_index"], dtype=np.int64)
            items = np.asarray(array["catalog_row"], dtype=np.int64)
            ranks = np.asarray(array["rank"], dtype=np.int64)
            scores = np.asarray(array["best_recency_weighted_cosine"], dtype=np.float32)
            expected_rows = int(variant["resources"]["active_users"]) * 100
            if len(users) != expected_rows or len(items) != expected_rows or len(ranks) != expected_rows:
                raise ValueError("retrieval artifact row-count drift")
            if ranks.min() != 1 or ranks.max() != 100 or not np.isfinite(scores).all():
                raise ValueError("retrieval rank/finite audit failed")
            keys = (users << 17) | items
            if len(np.unique(keys)) != len(keys):
                raise ValueError("retrieval contains duplicate user-item candidates")
            candidate_artifact_count += 1
    if not retrieval["summary"]["gate_passed"] or not all(retrieval["summary"]["gates"].values()):
        raise ValueError("M4.3 recorded gate is not fully passed")
    return {
        "status": "passed",
        "metric_files": len(metric_paths),
        "unique_recorded_artifact_identities": len(unique_identities),
        "student_embedding_artifacts": embedding_count,
        "retrieval_candidate_artifacts": candidate_artifact_count,
        "m4_3_gate_passed": True,
        "final_week": "not_run",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=Path("reports/m4"))
    args = parser.parse_args()
    print(json.dumps(verify_m4(args.report_dir.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
