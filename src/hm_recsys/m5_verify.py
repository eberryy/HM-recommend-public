from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .m4_contract import WARM_MAP, file_identity
from .m5_data import PRIMARY_K
from .m5_model import K_ADMIT, MAP_TOLERANCE, MAX_SINGLE_WINDOW_REGRESSION


def _identity_matches(record: dict[str, Any]) -> bool:
    actual = file_identity(Path(record["path"]))
    return actual["bytes"] == record["bytes"] and actual["sha256"] == record["sha256"]


def verify(report_dir: Path) -> dict[str, Any]:
    m50 = json.loads((report_dir / "M5_0_metrics.json").read_text(encoding="utf-8"))
    m52 = json.loads((report_dir / "M5_2_metrics.json").read_text(encoding="utf-8"))
    checks: dict[str, bool] = {
        "m50_measured": m50.get("status") == "measured" and m50.get("final_week") == "not_run",
        "m52_measured": m52.get("status") == "measured" and m52.get("final_week") == "not_run",
        "m50_gate_passed": m50.get("gate_passed") is True,
        "m52_gate_failed": m52.get("summary", {}).get("gate_passed") is False,
        "m6_not_authorized": m52.get("summary", {}).get("next_stage") == "stop_cold_admission_keep_M4_retrieval_evidence",
    }
    identities = 0
    candidate_files = 0
    for cutoff, row in m50["cutoffs"].items():
        checks[f"cutoff_safe_{cutoff}"] = row["seed_audit"]["cutoff_safe"] and row["final_week"] == "not_run"
        for name, identity in row["artifacts"].items():
            if name == "manifest":
                continue
            checks[f"identity_{cutoff}_{name}"] = _identity_matches(identity)
            identities += 1
        dataset = Path(row["artifacts"]["dataset"]["path"])
        with np.load(dataset) as arrays:
            users = np.asarray(arrays["user_index"])
            catalog = np.asarray(arrays["catalog_row"])
            ranks = np.asarray(arrays["student_rank"])
            keys = users.astype(np.int64) * 200_000 + catalog.astype(np.int64)
            checks[f"candidate_unique_{cutoff}"] = len(np.unique(keys)) == len(keys)
            checks[f"top50_budget_{cutoff}"] = (
                len(users) == int(row["retrieval"]["users_with_seed"]) * PRIMARY_K
                and int(ranks.min()) == 1 and int(ranks.max()) == PRIMARY_K
            )
            checks[f"finite_{cutoff}"] = all(
                np.isfinite(arrays[name]).all() for name in arrays.files
                if np.issubdtype(arrays[name].dtype, np.floating)
            )
        candidate_files += 1
    for window, row in m52["windows"].items():
        inner = row["inner"]["admission_results"]
        best = max(value["segments"]["overall"]["map@12"] for value in inner.values())
        expected_k = min(
            int(key) for key, value in inner.items()
            if best - value["segments"]["overall"]["map@12"] <= MAP_TOLERANCE
        )
        checks[f"inner_selection_{window}"] = row["selected_k_from_inner"] == expected_k
        base = row["outer"]["all_k_diagnostic"]["0"]["segments"]["overall"]["map@12"]
        checks[f"warm_anchor_{window}"] = abs(base - WARM_MAP[window]) <= MAP_TOLERANCE
        checks[f"outer_score_identity_{window}"] = _identity_matches(row["outer"]["scores"])
        identities += 1
    deltas = m52["summary"]["window_deltas_vs_warm_map@12"]
    selected = [row["outer"]["selected_result"] for row in m52["windows"].values()]
    warm_base = float(np.mean([
        row["outer"]["all_k_diagnostic"]["0"]["segments"]["warm_21_plus"]["map@12"]
        for row in m52["windows"].values()
    ]))
    warm_selected = float(np.mean([row["segments"]["warm_21_plus"]["map@12"] for row in selected]))
    inserted = sum(row["inserted_positive_pairs"] for row in selected)
    removed = sum(row["removed_positive_pairs"] for row in selected)
    recomputed = {
        "mean_map_strictly_above_warm": float(np.mean(list(deltas.values()))) > MAP_TOLERANCE,
        "at_least_3_of_4_windows_non_degrading": sum(value >= -MAP_TOLERANCE for value in deltas.values()) >= 3,
        "worst_window_within_tolerance": min(deltas.values()) >= -MAX_SINGLE_WINDOW_REGRESSION,
        "strict_cold_truth_enters_top12": sum(row["inserted_strict_cold_positive_pairs"] for row in selected) > 0,
        "cold_and_sparse_map_positive": (
            float(np.mean([row["segments"]["strict_cold"]["map@12"] for row in selected])) > 0
            and float(np.mean([row["segments"]["sparse_1_5"]["map@12"] for row in selected])) > 0
        ),
        "warm_preserved_or_pair_gain_dominates": warm_selected >= warm_base - MAP_TOLERANCE or inserted > removed,
        "inserted_positive_pairs_exceed_removed": inserted > removed,
        "cutoff_and_anchor_audits_pass": all(row["outer"]["seed_audit"]["cutoff_safe"] for row in m52["windows"].values()),
    }
    checks["gate_recomputation"] = recomputed == m52["summary"]["gates"] and not all(recomputed.values())
    checks["diagnostics_complete"] = set(m52.get("failure_diagnostics", {})) == set(m52["windows"])
    checks["reports_exist"] = all((report_dir / name).is_file() for name in (
        "M5_0_FINAL.md", "M5_1_FINAL.md", "M5_2_FINAL.md", "M5_FINAL.md"
    ))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("M5 verification failed: " + ", ".join(failed))
    return {
        "status": "passed",
        "checks": len(checks),
        "artifact_identities_rehashed": identities,
        "candidate_files_checked": candidate_files,
        "selected_k": {window: row["selected_k_from_inner"] for window, row in m52["windows"].items()},
        "m5_gate_passed": False,
        "final_week": "not_run",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently verify M5 artifacts and gates")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/m5"))
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.report_dir.resolve()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
