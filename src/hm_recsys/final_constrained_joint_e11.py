"""FINAL-E11: constrain real-only action utility to full-supervision Top5 candidates."""
from pathlib import Path
import gc
import json
import math
import shutil
import subprocess
import time
import traceback

import joblib
import numpy as np
import pandas as pd

from .final_b0_admission_e4 import segment_metrics
from .final_candidate_e2 import VALID, WINDOWS
from .final_full_cold_e8 import fit as fit35, final_matrix as matrix35
from .final_multiview_cold_e9 import fit as fit57, final_matrix as matrix57
from .final_oracle_audit import dump
from .final_real_action_e10 import fit as fit_action
from .final_pseudo_action_e6 import PREPARED, score_real
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN = Path("artifacts/final/constrained-joint-e11-v1")
REPORT = Path("reports/final")
E8 = Path("artifacts/final/full-cold-e8-v2")
E9 = Path("artifacts/final/multiview-cold-e9-v3")
E10 = Path("artifacts/final/real-action-e10-v1")
SELECTORS = ("F35_top5", "F57_top5", "union_top5")
QUOTAS = (.001, .005, .01, .02)


def candidate_rank(data: dict, score: np.ndarray) -> np.ndarray:
    rank = np.zeros(len(score), dtype=np.int16)
    for rows in data["cold"].groupby("user_index", sort=False).indices.values():
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(-score[rows], kind="stable")
        rank[rows[order]] = np.arange(1, len(rows) + 1)
    return rank


def selected_edges(data: dict, action_score: np.ndarray, f35: np.ndarray,
                   f57: np.ndarray, selector: str) -> np.ndarray:
    r35 = candidate_rank(data, f35)
    r57 = candidate_rank(data, f57)
    if selector == "F35_top5":
        allowed = r35 <= 5
    elif selector == "F57_top5":
        allowed = r57 <= 5
    elif selector == "union_top5":
        allowed = (r35 <= 5) | (r57 <= 5)
    else:
        raise KeyError(selector)
    edges = []
    for rows in data["cold"].groupby("user_index", sort=False).indices.values():
        rows = np.asarray(rows, dtype=np.int64)
        rows = rows[allowed[rows]]
        local = int(np.argmax(action_score[rows].ravel()))
        edges.append(int(rows[local // 12] * 12 + local % 12))
    return np.asarray(edges, dtype=np.int64)


def evaluate(data: dict, action_score: np.ndarray, f35: np.ndarray,
             f57: np.ndarray, selector: str, quota: float):
    edges = selected_edges(data, action_score, f35, f57, selector)
    values = action_score.ravel()[edges]
    candidates = edges // 12
    customer = data["cold"].iloc[candidates].customer_id.to_numpy(str)
    count = max(1, int(math.ceil(quota * len(edges))))
    selected = np.lexsort((customer, -values))[:count]
    lists = data["warm_lists"].copy()
    events = []
    for local in selected:
        edge = int(edges[local])
        candidate, slot = divmod(edge, 12)
        row = data["cold"].iloc[candidate]
        user = int(row.user_index)
        victim = lists[user, slot]
        lists[user, slot] = row.article_id
        delta = exact_ap(lists[user], data["truthsets"][data["users"][user]]) - data["baseline_ap"][user]
        events.append({
            "customer_id": data["users"][user], "candidate": row.article_id,
            "victim": victim, "slot": slot + 1, "score": float(values[local]),
            "delta": float(delta), "inserted": int(row.target),
            "removed": int(victim in data["truthsets"][data["users"][user]]),
        })
    events = pd.DataFrame(events)
    return {
        "delta_map": float(events.delta.sum() / len(data["users"])),
        "admissions": int(len(events)),
        "beneficial": int(events.delta.gt(0).sum()),
        "harmful": int(events.delta.lt(0).sum()),
        "neutral": int(events.delta.eq(0).sum()),
        "inserted": int(events.inserted.sum()),
        "removed": int(events.removed.sum()),
    }, lists, events


def historical_scores(repo: Path, cutoff: str):
    return (
        np.load(repo / E8 / f"{cutoff}-F35-scores.npy"),
        np.load(repo / E9 / f"{cutoff}-F57-scores.npy"),
        np.load(repo / E10 / f"{cutoff}-real_risk_separated-scores.npy"),
    )


def development_scores(repo: Path, cutoff: str, data: dict):
    folder = repo / RUN / "models"
    folder.mkdir(exist_ok=True)
    model35, info35 = fit35(repo, cutoff, folder)
    x35 = matrix35(repo, cutoff, data)
    score35 = model35.booster_.predict(x35, num_threads=4)
    model57, info57 = fit57(repo, cutoff, folder)
    x57, map57 = matrix57(repo, cutoff, data)
    score57 = model57.booster_.predict(x57, num_threads=4)
    action, info_action = fit_action(repo, cutoff, "real_risk_separated", folder)
    action_score = score_real(repo, cutoff, data, action)
    del model35, x35, model57, x57, action
    gc.collect()
    return score35, score57, action_score, {
        "F35": info35, "F57": info57, "action": info_action,
        "F57_heldout_feature_audit": map57,
    }


def register(repo: Path):
    contract = {
        "status": "preregistered_before_E11_evaluation",
        "stage": "FINAL-E11 候选排序与替换风险解耦后再合并",
        "observed_bottleneck": "E8/E9 的真实全量监督改善部分历史窗 Cold Top5；E10 真实动作模型在四窗找到2个 Cold 正例但误删3个 Warm 正例。",
        "hypothesis": "让真实全量候选模型先把 Cold50 缩为 Top5，再由真实跨时间动作模型只负责 Top5 内商品和 Warm 位置选择，可降低极稀疏动作模型同时搜索 600 个动作的难度。",
        "selectors": {
            "F35_top5": "单教师 35 特征分类器在每个用户 Cold50 内的前5名。",
            "F57_top5": "加入多视角教师的 57 特征分类器在每个用户 Cold50 内的前5名。",
            "union_top5": "上述两个前5名集合的并集，每用户至多10个 Cold 候选。",
        },
        "action_model": "冻结 E10 的 real_risk_separated 结构和训练规则，在允许候选乘以 Warm Top12 的动作中选预测期望变化最高者。",
        "admission": "再按该最高动作分数在候选用户间排序，执行头部0.1%、0.5%、1%、2%。",
        "historical_gate": "三窗 mean delta_MAP>0、至少两窗不退化、插入正例不少于2且插入数不少于误删数。",
        "development_gate": "历史通过后冻结唯一 selector 与 quota，运行既有四窗扩量门槛。",
        "if_fail": "保留 WV3-741；不扩量。",
        "cost_limit": "历史阶段只复用分数；若通过，四窗重新拟合已有结构并重算必要的多视角关系，预计20分钟内。",
        "historical_validation": VALID, "development_windows": WINDOWS,
        "final_week": "not_run", "no_commit_push": True,
        "free_disk_gib": shutil.disk_usage(repo).free / 2**30,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(repo / REPORT / "CONSTRAINED_JOINT_E11_CONTRACT.json", contract)
    return contract


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E11")
    (repo / RUN).mkdir(parents=True)
    contract = register(repo)
    start = time.perf_counter()
    historical, aggregate = {}, {}
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        f35, f57, action = historical_scores(repo, cutoff)
        historical[cutoff] = {}
        for selector in SELECTORS:
            historical[cutoff][selector] = {}
            for quota in QUOTAS:
                key = f"q={quota:g}"
                historical[cutoff][selector][key], _, _ = evaluate(
                    data, action, f35, f57, selector, quota,
                )
        del data, f35, f57, action
        gc.collect()
    for selector in SELECTORS:
        for quota in QUOTAS:
            key = f"{selector}|q={quota:g}"
            rows = [historical[t][selector][f"q={quota:g}"] for t in VALID]
            aggregate[key] = {
                "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                "admissions": int(sum(r["admissions"] for r in rows)),
                "inserted": int(sum(r["inserted"] for r in rows)),
                "removed": int(sum(r["removed"] for r in rows)),
                "beneficial": int(sum(r["beneficial"] for r in rows)),
                "harmful": int(sum(r["harmful"] for r in rows)),
                "selector": selector, "quota": quota,
            }
    eligible = [(key, value) for key, value in aggregate.items()
                if value["mean_delta"] > 0 and value["nonnegative"] >= 2 and
                value["inserted"] >= 2 and value["inserted"] >= value["removed"]]
    eligible.sort(key=lambda pair: (-pair[1]["mean_delta"], pair[1]["removed"],
                                    pair[1]["admissions"], SELECTORS.index(pair[1]["selector"])))
    selected = eligible[0][0] if eligible else None
    development = {}
    gates = {"not_run": "historical gate failed"}
    passed = False
    if selected:
        spec = aggregate[selected]
        for window, cutoff in WINDOWS.items():
            data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
            f35, f57, action, fit_info = development_scores(repo, cutoff, data)
            result, lists, events = evaluate(
                data, action, f35, f57, spec["selector"], spec["quota"],
            )
            result["segments"] = segment_metrics(data, lists)
            result["fit"] = fit_info
            development[window] = result
            save_frame(events, repo / RUN / f"{window}-events.parquet")
            del data, f35, f57, action, lists, events
            gc.collect()
        delta = np.array([r["delta_map"] for r in development.values()])
        warm = np.mean([r["segments"]["warm_21_plus"] for r in development.values()])
        cold = np.mean([r["segments"]["all_cold_sparse"] for r in development.values()])
        inserted = sum(r["inserted"] for r in development.values())
        removed = sum(r["removed"] for r in development.values())
        gates = {
            "mean_positive": bool(delta.mean() > 0),
            "three_nondegrade": bool((delta >= -1e-5).sum() >= 3),
            "worst": bool(delta.min() >= -5e-5),
            "warm": bool(warm >= -2e-5),
            "cold_sparse": bool(cold >= 0),
            "inserted_positive": bool(inserted >= 1),
            "efficiency": bool(inserted >= removed),
        }
        passed = all(gates.values())
    output = {
        "status": "completed", "stage": contract["stage"],
        "historical": historical, "historical_aggregate": aggregate,
        "selected": selected, "historical_gate_pass": selected is not None,
        "development": development, "development_gates": gates,
        "fullscale_allowed": passed, "fullscale_status": "not_run",
        "fallback": "challenger" if passed else "WV3-741",
        "final_week": "not_run", "seconds": time.perf_counter() - start,
    }
    dump(repo / REPORT / "CONSTRAINED_JOINT_E11.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"CONSTRAINED_JOINT_E11_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
