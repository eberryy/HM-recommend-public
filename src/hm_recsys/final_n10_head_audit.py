"""Read-only audit of N10 top-candidate gains and cross-user confidence."""
from pathlib import Path
import json
import time
import traceback

import joblib
import numpy as np
import pandas as pd

from .final_binary_admission_e5 import one_per_user
from .final_candidate_e2 import VALID
from .final_oracle_audit import dump


PREPARED = Path("artifacts/final/integration-10pct-v1/prepared")
E8 = Path("artifacts/final/full-cold-e8-v2")
E9 = Path("artifacts/final/multiview-cold-e9-v3")
E13 = Path("artifacts/final/negative-scale-e13-v1")
REPORT = Path("reports/final")


def rank_in_user(data, score):
    rank = np.zeros(len(score), dtype=np.int16)
    for rows in data["cold"].groupby("user_index", sort=False).indices.values():
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(-score[rows], kind="stable")
        rank[rows[order]] = np.arange(1, len(rows) + 1)
    return rank


def top(data, score):
    chosen = one_per_user(data, score)
    margin = []
    for rows in data["cold"].groupby("user_index", sort=False).indices.values():
        rows = np.asarray(rows, dtype=np.int64)
        values = np.sort(score[rows])
        margin.append(float(values[-1] - values[-2]) if len(values) > 1 else 0.)
    return chosen, np.asarray(margin)


def head(values, labels, customer):
    output = {}
    for quota in (.001, .002, .005, .01, .02, .05, .10):
        count = max(1, int(np.ceil(quota * len(values))))
        selected = np.lexsort((customer, -values))[:count]
        output[f"q={quota:g}"] = {
            "users": int(count), "positive_users": int(labels[selected].sum()),
            "precision": float(labels[selected].mean()),
        }
    return output


def run(repo: Path):
    repo = Path(repo)
    windows, frames = {}, []
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        cold = data["cold"]
        scores = {
            "B0": -cold.b0_rank.to_numpy(float),
            "F35": np.load(repo / E8 / f"{cutoff}-F35-scores.npy"),
            "F57": np.load(repo / E9 / f"{cutoff}-F57-scores.npy"),
            "N10": np.load(repo / E13 / f"{cutoff}-F57N10-scores.npy"),
        }
        chosen, margins = {}, {}
        for name, score in scores.items():
            chosen[name], margins[name] = top(data, score)
        n10 = chosen["N10"]
        b0 = chosen["B0"]
        n10_target = cold.target.to_numpy(np.int8)[n10]
        b0_target = cold.target.to_numpy(np.int8)[b0]
        same = cold.article_id.to_numpy(str)[n10] == cold.article_id.to_numpy(str)[b0]
        agreement = np.zeros(len(n10), dtype=np.int8)
        n10_articles = cold.article_id.to_numpy(str)[n10]
        for name in ("B0", "F35", "F57"):
            agreement += (n10_articles == cold.article_id.to_numpy(str)[chosen[name]]).astype(np.int8)
        customer = cold.iloc[n10].customer_id.to_numpy(str)
        n10_score = scores["N10"][n10]
        windows[cutoff] = {
            "candidate_users": int(len(n10)),
            "B0_top1_positive": int(b0_target.sum()),
            "N10_top1_positive": int(n10_target.sum()),
            "same_candidate_users": int(same.sum()),
            "different_candidate_users": int((~same).sum()),
            "N10_adds_truth_when_different": int(((n10_target == 1) & (b0_target == 0)).sum()),
            "N10_loses_truth_when_different": int(((n10_target == 0) & (b0_target == 1)).sum()),
            "both_truth_when_different": int(((n10_target == 1) & (b0_target == 1)).sum()),
            "score_head": head(n10_score, n10_target, customer),
            "margin_head": head(margins["N10"], n10_target, customer),
            "agreement": {
                str(level): {
                    "users": int((agreement == level).sum()),
                    "positive_users": int(n10_target[agreement == level].sum()),
                    "precision": float(n10_target[agreement == level].mean()) if (agreement == level).any() else None,
                } for level in range(4)
            },
        }
        frames.append(pd.DataFrame({
            "cutoff": cutoff, "customer_id": customer,
            "target": n10_target, "score": n10_score,
            "margin": margins["N10"], "agreement": agreement,
            "same_b0": same,
        }))
    pooled = pd.concat(frames, ignore_index=True)
    output = {
        "status": "completed", "stage": "FINAL-E13 N10 高分头只读诊断",
        "definitions": {
            "N10": "使用全部正例和稳定散列10%负例训练的57维 Cold 候选分类器。",
            "top1_positive": "每个有 Cold50 的用户只保留该排序器第一名后，未来7天实际购买该商品的用户数；分母是候选用户数。",
            "agreement": "N10 第一名同时被 B0、F35、F57 选为第一名的模型个数，范围0到3。",
            "head": "把候选用户按 N10 分数或用户内第一、二名分差排序后取指定比例；positive_users 是该部分实际命中的用户数。",
        },
        "windows": windows,
        "pooled": {
            "rows": int(len(pooled)), "positive_users": int(pooled.target.sum()),
            "score_head": head(pooled.score.to_numpy(), pooled.target.to_numpy(), pooled.customer_id.to_numpy()),
            "margin_head": head(pooled.margin.to_numpy(), pooled.target.to_numpy(), pooled.customer_id.to_numpy()),
            "agreement": {
                str(level): {
                    "users": int(pooled.agreement.eq(level).sum()),
                    "positive_users": int(pooled.loc[pooled.agreement.eq(level), "target"].sum()),
                    "precision": float(pooled.loc[pooled.agreement.eq(level), "target"].mean()) if pooled.agreement.eq(level).any() else None,
                } for level in range(4)
            },
        },
        "labels_used_only_for_audit": True,
        "models_fit": 0, "final_week": "not_run",
    }
    dump(repo / REPORT / "N10_HEAD_AUDIT.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"N10_HEAD_AUDIT_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
