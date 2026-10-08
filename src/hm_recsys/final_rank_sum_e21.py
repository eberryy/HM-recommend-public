"""FINAL-E21: frozen F70 + 7-day trend + qW safety rank-sum admission."""
from __future__ import annotations

from pathlib import Path
import json
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .final_b0_admission_e4 import segment_metrics
from .final_candidate_e2 import WINDOWS
from .final_fixed_candidate_e12 import candidate_choice, rank_descending
from .final_oracle_audit import dump
from .final_temporal_item_e16 import PREPARED, RUN as E16, TEMPORAL_NAMES, final_matrix
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN = Path("artifacts/final/rank-sum-e21-v1")
REPORT = Path("reports/final")
QUOTA = .02


def register(repo: Path):
    dump(repo / REPORT / "RANK_SUM_E21_CONTRACT.json", {
        "status": "registered_before_E21_development_labels",
        "stage": "FINAL-E21 冻结三信号名次和准入",
        "observed_bottleneck": "F70 用户内候选排序在四开发窗均优于 B0，但跨用户固定分位和学习型元准入未稳定兑现；E20 只用历史窗发现三信号组合满足历史门槛。",
        "frozen_rule": {
            "candidate": "每名用户固定选择 F70 分数最高的 Cold50 商品。",
            "victim": "固定替换 WV3-741 第12位。",
            "signals": "候选 F70 分数、候选商品截止日前7日交易次数、原第12位的 qW 安全度（1减qW用户内百分位）。",
            "fusion": "三个信号各自在当窗候选用户间降序排名；取三个负名次之和，数值越大越优。该方法是等权名次和/Borda式融合，不是倒数名次融合。",
            "admission": "按融合分执行候选用户头部2%；并列按 customer_id 确定性打破。",
        },
        "selection_provenance": "规则和2%比例由 FINAL-E20 的三个历史前向窗唯一选出。E20 文件名中的 rrf_f70_events7_qw 是遗留误称；其实际实现也是本合同所述等权名次和。",
        "development_gate": "四窗整体 mean delta_MAP>0、至少3窗delta>=-1e-5、最差>=-5e-5、暖组mean>=-2e-5、冷稀疏组mean>=0、插入正例至少1且插入数不少于误删数。",
        "if_fail": "保留 WV3-741；F70 仅作为候选层 challenger，停止在当前四窗继续搜索准入公式。",
        "development_windows": WINDOWS, "final_week": "not_run",
        "models_fit": 0, "no_commit_push": True,
    })


def evaluate(data: dict, chosen: np.ndarray, value: np.ndarray):
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    count = max(1, int(np.ceil(QUOTA * len(chosen))))
    selected = np.lexsort((customer, -value))[:count]
    lists = data["warm_lists"].copy()
    events = []
    for local in selected:
        candidate = int(chosen[local])
        row = data["cold"].iloc[candidate]
        user = int(row.user_index)
        victim = lists[user, 11]
        lists[user, 11] = row.article_id
        delta = exact_ap(lists[user], data["truthsets"][data["users"][user]]) - data["baseline_ap"][user]
        events.append({
            "customer_id": data["users"][user], "candidate": row.article_id,
            "victim": victim, "slot": 12, "admission_value": float(value[local]),
            "delta": float(delta), "inserted": int(row.target),
            "removed": int(victim in data["truthsets"][data["users"][user]]),
        })
    events = pd.DataFrame(events)
    return {
        "delta_map": float(events.delta.sum() / len(data["users"])),
        "admissions": int(len(events)), "beneficial": int(events.delta.gt(0).sum()),
        "harmful": int(events.delta.lt(0).sum()), "neutral": int(events.delta.eq(0).sum()),
        "inserted": int(events.inserted.sum()), "removed": int(events.removed.sum()),
    }, lists, events


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E21")
    (repo / RUN).mkdir(parents=True)
    register(repo)
    start = time.perf_counter()
    development = {}
    for window, cutoff in WINDOWS.items():
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        model = lgb.Booster(model_file=str(repo / E16 / f"{cutoff}-F70.txt"))
        x, mapping = final_matrix(repo, cutoff, data)
        score = model.predict(x, num_threads=4)
        chosen, _ = candidate_choice(data, score)
        user_index = data["cold"].iloc[chosen].user_index.to_numpy(np.int64)
        rank12 = data["warm"].iloc[11::12].reset_index(drop=True).iloc[user_index]
        event7_index = -len(TEMPORAL_NAMES) + TEMPORAL_NAMES.index("item_events_7d")
        event7 = x[chosen, event7_index]
        qw_safety = 1 - rank12.qW_within_user_percentile.to_numpy(float)
        customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
        value = -(
            rank_descending(score[chosen], customer)
            + rank_descending(event7, customer)
            + rank_descending(qw_safety, customer)
        ).astype(float)
        result, lists, events = evaluate(data, chosen, value)
        result["segments"] = segment_metrics(data, lists)
        result["audit"] = {
            "candidate_users": int(len(chosen)), "mapping": mapping,
            "model_reused": True, "models_fit": 0,
        }
        development[window] = result
        save_frame(events, repo / RUN / f"{window}-events.parquet")
        print("E21", window, result, flush=True)
    delta = np.asarray([r["delta_map"] for r in development.values()])
    warm = float(np.mean([r["segments"]["warm_21_plus"] for r in development.values()]))
    cold = float(np.mean([r["segments"]["all_cold_sparse"] for r in development.values()]))
    inserted = int(sum(r["inserted"] for r in development.values()))
    removed = int(sum(r["removed"] for r in development.values()))
    gates = {
        "mean_positive": bool(delta.mean() > 0),
        "three_nondegrade": bool((delta >= -1e-5).sum() >= 3),
        "worst": bool(delta.min() >= -5e-5),
        "warm": bool(warm >= -2e-5), "cold_sparse": bool(cold >= 0),
        "inserted_positive": bool(inserted >= 1), "efficiency": bool(inserted >= removed),
    }
    passed = all(gates.values())
    output = {
        "status": "completed", "stage": "FINAL-E21 冻结三信号名次和准入",
        "rule": "rank_sum_f70_events7_qw|q=0.02|rank12",
        "development": development,
        "aggregate": {
            "mean_delta": float(delta.mean()), "worst_delta": float(delta.min()),
            "warm_mean_delta": warm, "cold_sparse_mean_delta": cold,
            "inserted": inserted, "removed": removed,
            "admissions": int(sum(r["admissions"] for r in development.values())),
        },
        "development_gates": gates, "fullscale_allowed": passed,
        "fullscale_status": "not_run", "fallback": "challenger" if passed else "WV3-741",
        "models_fit": 0, "final_week": "not_run", "seconds": time.perf_counter() - start,
    }
    dump(repo / REPORT / "RANK_SUM_E21.json", output)
    print(json.dumps({
        "aggregate": output["aggregate"], "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"RANK_SUM_E21_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
