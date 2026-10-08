"""FINAL-E12: fixed full-supervision candidate, learned victim, fused admission."""
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
from .final_constrained_joint_e11 import E8, E9, E10
from .final_full_cold_e8 import fit as fit35, final_matrix as matrix35
from .final_multiview_cold_e9 import fit as fit57, final_matrix as matrix57
from .final_oracle_audit import dump
from .final_real_action_e10 import fit as fit_action
from .final_pseudo_action_e6 import PREPARED, score_real
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN = Path("artifacts/final/fixed-candidate-e12-v1")
REPORT = Path("reports/final")
SELECTORS = ("F35_top1", "F57_top1")
ADMISSIONS = ("action", "candidate", "rrf_score", "rrf_margin")
QUOTAS = (.001, .005, .01, .02)


def rank_descending(values: np.ndarray, customer: np.ndarray) -> np.ndarray:
    order = np.lexsort((customer, -values))
    rank = np.empty(len(values), dtype=np.int32)
    rank[order] = np.arange(1, len(values) + 1)
    return rank


def candidate_choice(data: dict, score: np.ndarray):
    chosen, margin = [], []
    for rows in data["cold"].groupby("user_index", sort=False).indices.values():
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(-score[rows], kind="stable")
        chosen.append(int(rows[order[0]]))
        margin.append(float(score[rows[order[0]]] - score[rows[order[1]]]) if len(rows) > 1 else 0.)
    return np.asarray(chosen, dtype=np.int64), np.asarray(margin, dtype=float)


def decisions(data: dict, action_score: np.ndarray, candidate_score: np.ndarray,
              admission: str):
    chosen, margin = candidate_choice(data, candidate_score)
    slot = np.argmax(action_score[chosen], axis=1)
    action = action_score[chosen, slot]
    candidate = candidate_score[chosen]
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    if admission == "action":
        value = action
    elif admission == "candidate":
        value = candidate
    elif admission == "rrf_score":
        value = 1 / (60 + rank_descending(action, customer)) + 1 / (60 + rank_descending(candidate, customer))
    elif admission == "rrf_margin":
        value = 1 / (60 + rank_descending(action, customer)) + 1 / (60 + rank_descending(margin, customer))
    else:
        raise KeyError(admission)
    return chosen, slot, value


def evaluate(data: dict, action_score: np.ndarray, candidate_score: np.ndarray,
             admission: str, quota: float):
    chosen, slots, values = decisions(data, action_score, candidate_score, admission)
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    count = max(1, int(math.ceil(quota * len(chosen))))
    selected = np.lexsort((customer, -values))[:count]
    lists = data["warm_lists"].copy()
    events = []
    for local in selected:
        candidate = int(chosen[local])
        slot = int(slots[local])
        row = data["cold"].iloc[candidate]
        user = int(row.user_index)
        victim = lists[user, slot]
        lists[user, slot] = row.article_id
        delta = exact_ap(lists[user], data["truthsets"][data["users"][user]]) - data["baseline_ap"][user]
        events.append({
            "customer_id": data["users"][user], "candidate": row.article_id,
            "victim": victim, "slot": slot + 1, "admission_value": float(values[local]),
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
        "status": "preregistered_before_E12_evaluation",
        "stage": "FINAL-E12 固定 Cold 商品与 Warm 被替换位置的分工",
        "observed_bottleneck": "E11 的 Top5 联合搜索在历史窗只有1个净正例；动作模型仍可能在候选集合内选错商品。",
        "hypothesis": "全量真实 Cold 分类器固定每个用户最可能购买的一个 Cold 商品，真实动作模型只选 Warm 被替换位置；跨用户准入再分别检验动作安全与候选置信证据。",
        "candidate_selectors": {
            "F35_top1": "35 特征真实 Cold 分类器的用户内第一名。",
            "F57_top1": "57 特征多视角真实 Cold 分类器的用户内第一名。",
        },
        "victim": "E10 风险分离模型在固定 Cold 商品对应的12个位置中选择预测期望变化最高者。",
        "admission_rules": {
            "action": "只按所选替换动作的预测期望变化排序。",
            "candidate": "只按所选 Cold 商品的跨用户购买分数排序。",
            "rrf_score": "动作分数名次与 Cold 商品分数名次等权倒数名次融合。",
            "rrf_margin": "动作分数名次与 Cold 用户内第一、二名分差名次等权倒数名次融合。",
        },
        "quotas": "候选用户头部0.1%、0.5%、1%、2%。",
        "historical_gate": "三窗 mean delta_MAP>0、至少两窗不退化、插入正例不少于2且插入数不少于误删数。",
        "development_gate": "历史通过后冻结唯一规则，运行既有四窗扩量门槛。",
        "if_fail": "保留 WV3-741；不扩量。",
        "cost_limit": "历史只复用已有分数；若通过，四窗预计20分钟内。",
        "historical_validation": VALID, "development_windows": WINDOWS,
        "final_week": "not_run", "no_commit_push": True,
        "free_disk_gib": shutil.disk_usage(repo).free / 2**30,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(repo / REPORT / "FIXED_CANDIDATE_E12_CONTRACT.json", contract)
    return contract


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E12")
    (repo / RUN).mkdir(parents=True)
    contract = register(repo)
    start = time.perf_counter()
    historical, aggregate = {}, {}
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        f35, f57, action = historical_scores(repo, cutoff)
        historical[cutoff] = {}
        for selector, candidate_score in (("F35_top1", f35), ("F57_top1", f57)):
            historical[cutoff][selector] = {}
            for admission in ADMISSIONS:
                historical[cutoff][selector][admission] = {}
                for quota in QUOTAS:
                    historical[cutoff][selector][admission][f"q={quota:g}"], _, _ = evaluate(
                        data, action, candidate_score, admission, quota,
                    )
        del data, f35, f57, action
        gc.collect()
    for selector in SELECTORS:
        for admission in ADMISSIONS:
            for quota in QUOTAS:
                key = f"{selector}|{admission}|q={quota:g}"
                rows = [historical[t][selector][admission][f"q={quota:g}"] for t in VALID]
                aggregate[key] = {
                    "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                    "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                    "admissions": int(sum(r["admissions"] for r in rows)),
                    "inserted": int(sum(r["inserted"] for r in rows)),
                    "removed": int(sum(r["removed"] for r in rows)),
                    "beneficial": int(sum(r["beneficial"] for r in rows)),
                    "harmful": int(sum(r["harmful"] for r in rows)),
                    "selector": selector, "admission": admission, "quota": quota,
                }
    eligible = [(key, value) for key, value in aggregate.items()
                if value["mean_delta"] > 0 and value["nonnegative"] >= 2 and
                value["inserted"] >= 2 and value["inserted"] >= value["removed"]]
    eligible.sort(key=lambda pair: (-pair[1]["mean_delta"], pair[1]["removed"],
                                    pair[1]["admissions"], SELECTORS.index(pair[1]["selector"]),
                                    ADMISSIONS.index(pair[1]["admission"])))
    selected = eligible[0][0] if eligible else None
    development = {}
    gates = {"not_run": "historical gate failed"}
    passed = False
    if selected:
        spec = aggregate[selected]
        for window, cutoff in WINDOWS.items():
            data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
            f35, f57, action, fit_info = development_scores(repo, cutoff, data)
            candidate_score = f35 if spec["selector"] == "F35_top1" else f57
            result, lists, events = evaluate(
                data, action, candidate_score, spec["admission"], spec["quota"],
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
    dump(repo / REPORT / "FIXED_CANDIDATE_E12.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"FIXED_CANDIDATE_E12_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
