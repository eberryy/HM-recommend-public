"""Independent replay for FINAL-E24 subgroup Recall and model identities."""
from pathlib import Path
import json
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np

from .final_candidate_e2 import WINDOWS
from .final_oracle_audit import dump
from .final_temporal_item_e16 import FEATURES, PREPARED, RUN as F70_RUN, TEMPORAL_NAMES, final_matrix


REPORT = Path("reports/final")
SOURCE = REPORT / "F70_EXPLAINABILITY_AUDIT.json"
OUTPUT = REPORT / "F70_EXPLAINABILITY_AUDIT_VERIFICATION.json"


def independent_ranks(frame, values: np.ndarray) -> np.ndarray:
    ranks = np.zeros(len(frame), dtype=np.int16)
    users = frame.user_index.to_numpy(np.int64)
    for user in np.unique(users):
        rows = np.flatnonzero(users == user)
        # row index is the deterministic secondary key, independent of main helper.
        order = np.lexsort((rows, -values[rows]))
        ranks[rows[order]] = np.arange(1, len(rows) + 1, dtype=np.int16)
    return ranks


def run(repo: Path) -> dict:
    repo = Path(repo)
    source = json.loads((repo / SOURCE).read_text(encoding="utf-8"))
    checks = []
    max_shap_error = 0.0
    for window, cutoff in WINDOWS.items():
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        cold = data["cold"]
        model = lgb.Booster(model_file=str(repo / F70_RUN / f"{cutoff}-F70.txt"))
        matrix, _ = final_matrix(repo, cutoff, data)
        score = model.predict(matrix, num_threads=4)
        ranks = {
            "B0": independent_ranks(cold, -cold.b0_rank.to_numpy(float)),
            "F70": independent_ranks(cold, score),
        }
        count = cold.interaction_count_before_cutoff.to_numpy(np.int64)
        strict = cold.strict_cold_flag.to_numpy(np.int8) == 1
        sparse = cold.sparse1_5_flag.to_numpy(np.int8) == 1
        target = cold.target.to_numpy(np.int8) == 1
        checks.append({
            "name": f"{window}:strict_flag_matches_count_zero",
            "passed": bool(np.array_equal(strict, count == 0)),
        })
        checks.append({
            "name": f"{window}:sparse_flag_matches_count_one_to_five",
            "passed": bool(np.array_equal(sparse, (count >= 1) & (count <= 5))),
        })
        for subgroup, mask in (("strict_cold", strict), ("sparse_1_5", sparse)):
            denominator = int((target & mask).sum())
            expected = source["windows"][window]["subgroups"][subgroup]
            checks.append({
                "name": f"{window}:{subgroup}:positive_denominator",
                "passed": denominator == expected["B0"]["positive_pair_denominator"],
                "actual": denominator,
            })
            for method in ("B0", "F70"):
                for k in (1, 5):
                    hits = int((target & mask & (ranks[method] <= k)).sum())
                    checks.append({
                        "name": f"{window}:{subgroup}:{method}:hits_at_{k}",
                        "passed": hits == expected[method]["hits"][str(k)],
                        "actual": hits,
                    })
        explain_rows = np.flatnonzero((ranks["F70"] == 1) | target)[:128]
        contribution = model.predict(matrix[explain_rows], pred_contrib=True, num_threads=4)
        raw = model.predict(matrix[explain_rows], raw_score=True, num_threads=4)
        max_shap_error = max(max_shap_error, float(np.max(np.abs(contribution.sum(axis=1) - raw))))
        checks.append({
            "name": f"{window}:model_feature_order",
            "passed": model.feature_name() == FEATURES,
        })
    checks.extend([
        {
            "name": "no_strict_sparse_positive_overlap",
            "passed": source["verification"]["strict_sparse_overlap_positive_pairs"] == 0,
        },
        {
            "name": "all_264_positive_pairs_partitioned",
            "passed": (
                source["micro_aggregate"]["strict_cold"]["B0"]["positive_pair_denominator"]
                + source["micro_aggregate"]["sparse_1_5"]["B0"]["positive_pair_denominator"]
                == source["verification"]["all_positive_pairs_total"] == 264
            ),
        },
        {
            "name": "models_fit_zero",
            "passed": source["verification"]["models_fit"] == 0,
        },
        {
            "name": "final_week_not_run",
            "passed": source["verification"]["final_week"] == "not_run",
        },
        {
            "name": "tree_shap_subset_reconstruction",
            "passed": max_shap_error < 1e-10,
            "max_abs_error": max_shap_error,
        },
    ])
    result = {
        "status": "pass" if all(check["passed"] for check in checks) else "fail",
        "stage": "FINAL-E24 independent replay",
        "checks": checks,
        "passed": int(sum(check["passed"] for check in checks)),
        "total": int(len(checks)),
        "models_fit": 0,
        "recommendations_or_admissions_changed": False,
        "final_week": "not_run",
    }
    dump(repo / OUTPUT, result)
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(run(Path.cwd()), ensure_ascii=False, indent=2), flush=True)
    except Exception:
        dump(REPORT / f"F70_EXPLAINABILITY_AUDIT_VERIFICATION_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
