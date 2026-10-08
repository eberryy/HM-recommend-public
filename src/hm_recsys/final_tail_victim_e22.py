"""FINAL-E22: bounded historical selection of a safe victim in ranks 8--12."""
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
from .final_candidate_e2 import VALID, WINDOWS
from .final_fixed_candidate_e12 import candidate_choice, rank_descending
from .final_oracle_audit import dump
from .final_temporal_item_e16 import PREPARED, RUN as E16, TEMPORAL_NAMES, final_matrix
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN = Path("artifacts/final/tail-victim-e22-v1")
REPORT = Path("reports/final")
VICTIMS = ("rank12", "warm_safest", "qw_safest", "joint_safest")
ADMISSIONS = ("f70", "f70_events7", "f70_events7_victim_qw")
QUOTAS = (.002, .005, .01, .02)


def register(repo: Path):
    dump(repo / REPORT / "TAIL_VICTIM_E22_CONTRACT.json", {
        "status": "registered_before_E22_historical_labels",
        "stage": "FINAL-E22 第8至12位安全牺牲项",
        "observed_bottleneck": "E21 四开发窗已插入2个 Cold 正例，但固定删除第12位误删7个 Warm 正例；候选层有稳定提升，主要剩余损失来自被替换商品。",
        "candidate": "每名用户固定为 F70 用户内第一名，不改候选模型。",
        "victim_policies": {
            "rank12": "固定删除WV3-741第12位。",
            "warm_safest": "在原第8至12位中删除 Warm 用户内百分位最低者，即既有排序器置信度最低者。",
            "qw_safest": "在原第8至12位中删除 qW 用户内百分位最低者，即历史保留风险模型认为最安全者。",
            "joint_safest": "在第8至12位内，将 Warm 安全名次和 qW 安全名次等权相加，删除合计最安全者。",
        },
        "admission_policies": {
            "f70": "按 F70 分数跨用户排序。",
            "f70_events7": "F70分数与候选商品近7日交易次数的跨用户等权名次和。",
            "f70_events7_victim_qw": "前两项再加所选牺牲项qW安全度的跨用户等权名次和。",
        },
        "search_scope": "4种牺牲项×3种准入证据×4个固定比例，共48项；只用三个历史前向窗选唯一策略。",
        "historical_gate": "mean delta_MAP>0、至少两窗不退化、累计插入不少于2且不少于误删；按mean、误删、动作数及固定枚举顺序选择。",
        "development_gate": "唯一历史优胜策略才进入四开发窗；使用FINAL-WRAPUP扩量门槛。",
        "if_fail": "保留WV3-741；F70只保留为候选层challenger并停止本轮Cold搜索。",
        "historical_validation": VALID, "development_windows": WINDOWS,
        "final_week": "not_run", "models_fit": 0, "no_commit_push": True,
    })


def score_for(repo: Path, cutoff: str, data: dict):
    saved = repo / E16 / f"{cutoff}-F70-scores.npy"
    if saved.exists():
        return np.load(saved), True
    model = lgb.Booster(model_file=str(repo / E16 / f"{cutoff}-F70.txt"))
    x, _ = final_matrix(repo, cutoff, data)
    return model.predict(x, num_threads=4), True


def pick_victims(data: dict, chosen: np.ndarray, policy: str):
    user_index = data["cold"].iloc[chosen].user_index.to_numpy(np.int64)
    warm = data["warm"]
    slots = np.empty(len(chosen), dtype=np.int8)
    safety = np.empty(len(chosen), dtype=float)
    for out, user in enumerate(user_index):
        tail = warm.iloc[user * 12 + 7:user * 12 + 12]
        warm_safe = 1 - tail.warm_user_percentile.to_numpy(float)
        qw_safe = 1 - tail.qW_within_user_percentile.to_numpy(float)
        if policy == "rank12":
            local = 4
        elif policy == "warm_safest":
            local = int(np.nanargmax(np.nan_to_num(warm_safe, nan=-np.inf)))
        elif policy == "qw_safest":
            local = int(np.nanargmax(np.nan_to_num(qw_safe, nan=-np.inf)))
        elif policy == "joint_safest":
            # Five-item within-user rank-sum; deterministic first maximum breaks ties.
            ids = tail.article_id.to_numpy(str)
            combined = -rank_descending(warm_safe, ids) - rank_descending(qw_safe, ids)
            local = int(np.argmax(combined))
        else:
            raise KeyError(policy)
        slots[out] = local + 7
        safety[out] = qw_safe[local] if np.isfinite(qw_safe[local]) else 0.
    return slots, safety


def admission_values(data: dict, chosen: np.ndarray, score: np.ndarray,
                     event7: np.ndarray, victim_qw: np.ndarray, policy: str):
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    if policy == "f70":
        return score[chosen]
    value = -rank_descending(score[chosen], customer) - rank_descending(event7, customer)
    if policy == "f70_events7_victim_qw":
        value = value - rank_descending(victim_qw, customer)
    return value.astype(float)


def evaluate(data: dict, chosen: np.ndarray, slots: np.ndarray,
             values: np.ndarray, quota: float):
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    count = max(1, int(np.ceil(quota * len(chosen))))
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
        "admissions": int(len(events)), "beneficial": int(events.delta.gt(0).sum()),
        "harmful": int(events.delta.lt(0).sum()), "neutral": int(events.delta.eq(0).sum()),
        "inserted": int(events.inserted.sum()), "removed": int(events.removed.sum()),
    }, lists, events


def window_material(repo: Path, cutoff: str):
    data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
    score, reused = score_for(repo, cutoff, data)
    x, mapping = final_matrix(repo, cutoff, data)
    chosen, _ = candidate_choice(data, score)
    event7_index = -len(TEMPORAL_NAMES) + TEMPORAL_NAMES.index("item_events_7d")
    return data, score, chosen, x[chosen, event7_index], {"mapping": mapping, "model_reused": reused}


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E22")
    (repo / RUN).mkdir(parents=True)
    register(repo)
    start = time.perf_counter()
    historical = {}
    for cutoff in VALID:
        data, score, chosen, event7, audit = window_material(repo, cutoff)
        historical[cutoff] = {"audit": audit, "rules": {}}
        for victim in VICTIMS:
            slots, safety = pick_victims(data, chosen, victim)
            historical[cutoff]["rules"][victim] = {}
            for admission in ADMISSIONS:
                values = admission_values(data, chosen, score, event7, safety, admission)
                historical[cutoff]["rules"][victim][admission] = {
                    f"q={q:g}": evaluate(data, chosen, slots, values, q)[0] for q in QUOTAS
                }
    aggregate = {}
    for victim in VICTIMS:
        for admission in ADMISSIONS:
            for quota in QUOTAS:
                key = f"{victim}|{admission}|q={quota:g}"
                rows = [historical[t]["rules"][victim][admission][f"q={quota:g}"] for t in VALID]
                aggregate[key] = {
                    "victim": victim, "admission": admission, "quota": quota,
                    "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                    "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                    "inserted": int(sum(r["inserted"] for r in rows)),
                    "removed": int(sum(r["removed"] for r in rows)),
                    "admissions": int(sum(r["admissions"] for r in rows)),
                }
    eligible = [(key, value) for key, value in aggregate.items()
                if value["mean_delta"] > 0 and value["nonnegative"] >= 2
                and value["inserted"] >= 2 and value["inserted"] >= value["removed"]]
    eligible.sort(key=lambda pair: (
        -pair[1]["mean_delta"], pair[1]["removed"], pair[1]["admissions"],
        VICTIMS.index(pair[1]["victim"]), ADMISSIONS.index(pair[1]["admission"]),
    ))
    selected = eligible[0][0] if eligible else None
    development = {}
    gates = {"not_run": "historical gate failed"}
    passed = False
    if selected:
        spec = aggregate[selected]
        for window, cutoff in WINDOWS.items():
            data, score, chosen, event7, audit = window_material(repo, cutoff)
            slots, safety = pick_victims(data, chosen, spec["victim"])
            values = admission_values(data, chosen, score, event7, safety, spec["admission"])
            result, lists, events = evaluate(data, chosen, slots, values, spec["quota"])
            result["segments"] = segment_metrics(data, lists)
            result["audit"] = audit
            development[window] = result
            save_frame(events, repo / RUN / f"{window}-events.parquet")
            print("E22", window, result, flush=True)
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
        "status": "completed", "stage": "FINAL-E22 第8至12位安全牺牲项",
        "historical": historical, "historical_aggregate": aggregate,
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "development_gates": gates,
        "fullscale_allowed": passed, "fullscale_status": "not_run",
        "fallback": "challenger" if passed else "WV3-741",
        "models_fit": 0, "final_week": "not_run", "seconds": time.perf_counter() - start,
    }
    dump(repo / REPORT / "TAIL_VICTIM_E22.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"TAIL_VICTIM_E22_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
