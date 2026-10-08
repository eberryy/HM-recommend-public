from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np

from .m33 import ROLLING_PROTOCOL
from .m4_contract import FINAL_CUTOFF, WARM_MAP, atomic_json
from .m53 import choose_k
from .m54 import LINEAGES, RUN_ID as M54_RUN_ID
from .m55 import (
    DUAL_FEATURES,
    MAP_TOLERANCE,
    RUN_ID as M55_RUN_ID,
    WARM_FEATURES,
    WINDOW_TOLERANCE,
)


RUN_ID = "m5-5-v1-independent-verification"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_identity(identity: dict[str, Any]) -> dict[str, Any]:
    path = Path(identity["path"])
    exists = path.is_file()
    actual_bytes = path.stat().st_size if exists else None
    actual_sha256 = _sha256(path) if exists else None
    return {
        "path": str(path),
        "expected_bytes": int(identity["bytes"]),
        "actual_bytes": actual_bytes,
        "expected_sha256": identity["sha256"],
        "actual_sha256": actual_sha256,
        "passed": bool(
            exists
            and actual_bytes == int(identity["bytes"])
            and actual_sha256 == identity["sha256"]
        ),
    }


def _recompute_m55_gates(m55: dict[str, Any], m54_passed: bool) -> dict[str, bool]:
    dual_deltas = {
        window: row["evaluations"]["dual_fusion"]["segments"]["overall"]["delta_vs_warm_map@12"]
        for window, row in m55["windows"].items()
    }
    warm_deltas = {
        window: row["evaluations"]["dual_fusion"]["segments"]["warm_21_plus"]["delta_vs_warm_map@12"]
        for window, row in m55["windows"].items()
    }
    inserted = sum(
        row["evaluations"]["dual_fusion"]["inserted_cold_sparse_positive_pairs"]
        for row in m55["windows"].values()
    )
    removed = sum(
        row["evaluations"]["dual_fusion"]["removed_warm_positive_pairs"]
        for row in m55["windows"].values()
    )
    mean_delta = float(np.mean(list(dual_deltas.values())))
    return {
        "mean_overall_map_strictly_above_warm": mean_delta > MAP_TOLERANCE,
        "at_least_3_of_4_outer_windows_non_degrading": (
            sum(value >= -MAP_TOLERANCE for value in dual_deltas.values()) >= 3
        ),
        "worst_window_within_0.0002": min(dual_deltas.values()) >= -WINDOW_TOLERANCE,
        "cold_only_truth_enters_top12_in_at_least_2_windows": sum(
            row["evaluations"]["dual_fusion"]["cold_only_truth_conversion"]["top12"] > 0
            for row in m55["windows"].values()
        ) >= 2,
        "warm_map_not_systematically_broken": (
            float(np.mean(list(warm_deltas.values()))) >= -WINDOW_TOLERANCE
            and sum(value < -WINDOW_TOLERANCE for value in warm_deltas.values()) <= 1
        ),
        "pair_loss_controlled_or_map_clearly_covers": (
            inserted >= removed or mean_delta >= WINDOW_TOLERANCE
        ),
        "oof_cutoff_candidate_feature_identity_pass": bool(m54_passed) and all(
            row["final_week"] == "not_run" for row in m55["windows"].values()
        ),
    }


def _score_distribution(path: Path, score: str, available: str) -> dict[str, Any]:
    literal = "'" + path.resolve().as_posix().replace("'", "''") + "'"
    connection = duckdb.connect()
    try:
        row = connection.execute(
            f"""
            SELECT count(*) FILTER (WHERE {available}=1)::BIGINT,
                   min({score}) FILTER (WHERE {available}=1),
                   approx_quantile({score},0.5) FILTER (WHERE {available}=1),
                   approx_quantile({score},0.95) FILTER (WHERE {available}=1),
                   max({score}) FILTER (WHERE {available}=1)
            FROM read_parquet({literal})
            """
        ).fetchone()
    finally:
        connection.close()
    return {
        "available_rows": int(row[0]),
        "min": float(row[1]),
        "median": float(row[2]),
        "p95": float(row[3]),
        "max": float(row[4]),
    }


def _branch_supervision(paths: list[Path]) -> dict[str, Any]:
    literals = ",".join(
        "'" + path.resolve().as_posix().replace("'", "''") + "'" for path in paths
    )
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT source_branch,count(*)::BIGINT AS candidate_rows,
                   sum(target)::BIGINT AS positive_pairs
            FROM read_parquet([{literals}])
            GROUP BY source_branch ORDER BY source_branch
            """
        ).fetchall()
    finally:
        connection.close()
    total_positives = sum(int(row[2]) for row in rows)
    return {
        str(branch): {
            "candidate_rows": int(candidate_rows),
            "positive_pairs": int(positive_pairs),
            "positive_density": int(positive_pairs) / max(int(candidate_rows), 1),
            "share_of_all_positive_pairs": int(positive_pairs) / max(total_positives, 1),
        }
        for branch, candidate_rows, positive_pairs in rows
    }


def verify(
    *, m53_path: Path, m54_path: Path, m55_path: Path, output_path: Path,
) -> dict[str, Any]:
    m53 = json.loads(m53_path.read_text(encoding="utf-8"))
    m54 = json.loads(m54_path.read_text(encoding="utf-8"))
    m55 = json.loads(m55_path.read_text(encoding="utf-8"))

    identities: list[dict[str, Any]] = []
    identities.append(_verify_identity(m53["inputs"]["transactions"]))
    identities.extend(
        _verify_identity(value) for value in m53["inputs"]["warm_evaluation_dbs"].values()
    )
    for cutoff in m54["cutoffs"].values():
        identities.append(_verify_identity(cutoff["features"]["artifact"]))
        identities.append(_verify_identity(cutoff["manifest"]))
    for window in m55["windows"].values():
        identities.append(_verify_identity(window["predictions"]))
        for variant in window["training"].values():
            identities.append(_verify_identity(variant["inner_model"]["model"]))
            identities.append(_verify_identity(variant["outer_model"]["model"]))
            identities.append(_verify_identity(variant["outer_category_maps"]))

    selected_k, selected_reason = choose_k({
        window: row["k_metrics"] for window, row in m53["windows"].items()
    })
    m53_decision_matches = (
        selected_k == int(m53["decision"]["k_warm"])
        and selected_reason == m53["decision"]["reason"]
        and all(
            float(row["k_metrics"][str(selected_k)]["warm_positive_retention@k"]) >= 0.95
            for row in m53["windows"].values()
        )
    )

    lineage_checks: dict[str, bool] = {}
    for cutoff, row in m54["cutoffs"].items():
        safe = cutoff < FINAL_CUTOFF and all(bool(value) for value in row["gate_checks"].values())
        for score_name, lineage in row["lineage"].items():
            safe = safe and bool(lineage["safe"])
            if lineage["available"]:
                safe = safe and lineage["scoring_cutoff"] == cutoff
                safe = safe and lineage["latest_training_label_end"] < cutoff
                safe = safe and all(value < cutoff for value in lineage["training_cutoffs"])
            else:
                safe = safe and cutoff == min(LINEAGES)
        lineage_checks[cutoff] = safe

    baseline_anchor_matches = all(
        abs(
            float(row["evaluations"]["frozen_warm_v1"]["segments"]["overall"]["map@12"])
            - float(WARM_MAP[window])
        ) <= 1e-12
        for window, row in m55["windows"].items()
    )
    recomputed_gates = _recompute_m55_gates(m55, bool(m54["gate"]["passed"]))
    stored_gates = {name: bool(value) for name, value in m55["summary"]["gates"].items()}

    source_features = sorted(set(DUAL_FEATURES) - set(WARM_FEATURES))
    full_source_gain: dict[str, Any] = {}
    model_feature_schema_matches = True
    for window, row in m55["windows"].items():
        model_path = Path(row["training"]["dual_fusion"]["outer_model"]["model"]["path"])
        model = lgb.Booster(model_file=str(model_path))
        model_names = model.feature_name()
        model_feature_schema_matches = model_feature_schema_matches and model_names == DUAL_FEATURES
        gains = dict(zip(model_names, model.feature_importance("gain")))
        splits = dict(zip(model_names, model.feature_importance("split")))
        full_source_gain[window] = [
            {
                "feature": feature,
                "gain": float(gains.get(feature, 0.0)),
                "split": int(splits.get(feature, 0)),
            }
            for feature in source_features
        ]

    upstream: dict[str, Any] = {}
    branch_supervision: dict[str, Any] = {}
    for window, protocol in ROLLING_PROTOCOL.items():
        cutoff = protocol["outer_validation"]
        feature_path = Path(m54["cutoffs"][cutoff]["features"]["artifact"]["path"])
        upstream[window] = {
            "warm_model_score": _score_distribution(
                feature_path, "warm_model_score", "warm_model_score_available"
            ),
            "cold_expert_score": _score_distribution(
                feature_path, "cold_expert_score", "cold_expert_score_available"
            ),
        }
        branch_supervision[window] = _branch_supervision([
            Path(m54["cutoffs"][train_cutoff]["features"]["artifact"]["path"])
            for train_cutoff in protocol["outer_train"]
        ])

    final_week_safe = (
        m53["final_week"] == "not_run"
        and m54["final_week"] == "not_run"
        and m55["final_week"] == "not_run"
        and FINAL_CUTOFF not in m54["cutoffs"]
        and all(
            row["protocol"]["outer_validation"] < FINAL_CUTOFF
            and row["final_week"] == "not_run"
            for row in m55["windows"].values()
        )
    )
    checks = {
        "authoritative_run_ids_match": (
            m54["run_id"] == M54_RUN_ID and m55["run_id"] == M55_RUN_ID
        ),
        "m53_selection_recomputed": m53_decision_matches,
        "m54_all_ten_cutoffs_present": set(m54["cutoffs"]) == set(LINEAGES),
        "m54_lineage_recomputed_safe": all(lineage_checks.values()),
        "all_rehashed_artifacts_match": all(row["passed"] for row in identities),
        "m55_warm_anchor_matches": baseline_anchor_matches,
        "m55_gates_recomputed_identically": recomputed_gates == stored_gates,
        "m55_gate_summary_matches": bool(m55["summary"]["gate_passed"]) == all(recomputed_gates.values()),
        "m55_model_feature_schema_matches": model_feature_schema_matches,
        "final_week_not_run": final_week_safe,
    }
    result = {
        "schema_version": "m5.5-independent-verification-v1",
        "stage": "M5.5 verification",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "meaning": "evidence integrity passed does not mean the M5.5 model promotion gate passed",
        "checks": checks,
        "m53_recomputed_decision": {"k_warm": selected_k, "reason": selected_reason},
        "m54_lineage_checks": lineage_checks,
        "m55_recomputed_gates": recomputed_gates,
        "source_aware_feature_gain_full": full_source_gain,
        "outer_training_branch_supervision": branch_supervision,
        "upstream_score_distributions": upstream,
        "rehash": {
            "files": len(identities),
            "bytes": sum(int(row["actual_bytes"] or 0) for row in identities),
            "identities": identities,
        },
        "final_week": "not_run",
    }
    atomic_json(output_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify M5.3-M5.5 evidence and temporal gates")
    parser.add_argument("--m53", type=Path, default=Path("reports/m5_3/M5_3_metrics.json"))
    parser.add_argument("--m54", type=Path, default=Path("reports/m5_4/M5_4_metrics.json"))
    parser.add_argument("--m55", type=Path, default=Path("reports/m5_5/M5_5_metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/m5_5/M5_5_VERIFICATION.json"))
    args = parser.parse_args(argv)
    result = verify(
        m53_path=args.m53.resolve(), m54_path=args.m54.resolve(),
        m55_path=args.m55.resolve(), output_path=args.output.resolve(),
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
