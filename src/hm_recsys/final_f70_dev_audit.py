"""Read-only four-window candidate audit for the frozen E16 F70 models."""
from pathlib import Path
import json
import time
import traceback

import joblib
import lightgbm as lgb

from .final_candidate_e2 import WINDOWS
from .final_oracle_audit import dump
from .final_relation_pilot import identify
from .final_temporal_item_e16 import PREPARED, final_matrix


MODEL = Path("artifacts/final/temporal-item-e16-v1")
REPORT = Path("reports/final")


def run(repo: Path):
    repo = Path(repo)
    windows = {}
    for window, cutoff in WINDOWS.items():
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        model = lgb.Booster(model_file=str(repo / MODEL / f"{cutoff}-F70.txt"))
        matrix, mapping = final_matrix(repo, cutoff, data)
        score = model.predict(matrix, num_threads=4)
        windows[window] = {
            "cutoff": cutoff,
            "B0": identify(data, -data["cold"].b0_rank.to_numpy(float)),
            "F70": identify(data, score),
            "mapping": mapping,
        }
    output = {
        "status": "completed", "stage": "FINAL-E16 F70 四开发窗候选只读复核",
        "definitions": {
            "F70": "E16 的70维 Cold 候选分类器：57维用户—商品/多视角关系加13维截止日前商品新鲜度与短期趋势。",
            "Recall_at_k": "每窗 Cold50 内全部未来正例用户—商品对为分母，按每用户候选排序后进入前k名的正例对比例。",
            "Precision_at_1": "每窗有 Cold 候选的用户为分母，用户内第一名实际成为未来正例的比例。",
        },
        "windows": windows, "models_fit": 0,
        "labels_used_only_for_candidate_audit": True,
        "recommendations_or_admissions_changed": False,
        "final_week": "not_run",
    }
    dump(repo / REPORT / "F70_DEVELOPMENT_CANDIDATE_AUDIT.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"F70_DEVELOPMENT_CANDIDATE_AUDIT_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
