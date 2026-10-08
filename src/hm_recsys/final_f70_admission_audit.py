"""FINAL-E20: historical-only admission-signal audit for the F70 top candidate."""
from __future__ import annotations

from pathlib import Path
import json
import time
import traceback

import joblib
import numpy as np

from .final_candidate_e2 import VALID
from .final_fixed_candidate_e12 import candidate_choice, rank_descending
from .final_meta_admission_e19 import PREPARED
from .final_oracle_audit import dump
from .final_temporal_item_e16 import RUN as E16, TEMPORAL_NAMES, final_matrix
from .p43a_policy import exact_ap


REPORT = Path("reports/final")
QUOTAS = (.001, .002, .005, .01, .02, .05)


def percentile_score(value: np.ndarray, customer: np.ndarray) -> np.ndarray:
    return -rank_descending(value, customer).astype(float)


def evaluate(data: dict, chosen: np.ndarray, value: np.ndarray, quota: float):
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    count = max(1, int(np.ceil(quota * len(chosen))))
    selected = np.lexsort((customer, -value))[:count]
    delta = inserted = removed = beneficial = harmful = 0
    for local in selected:
        row = data["cold"].iloc[int(chosen[local])]
        user = int(row.user_index)
        before = data["warm_lists"][user]
        after = before.copy()
        victim = after[11]
        after[11] = row.article_id
        change = exact_ap(after, data["truthsets"][data["users"][user]]) - data["baseline_ap"][user]
        delta += change
        inserted += int(row.target)
        removed += int(victim in data["truthsets"][data["users"][user]])
        beneficial += int(change > 0)
        harmful += int(change < 0)
    return {
        "delta_map": float(delta / len(data["users"])), "admissions": int(count),
        "inserted": int(inserted), "removed": int(removed),
        "beneficial": int(beneficial), "harmful": int(harmful),
    }


def register(repo: Path):
    dump(repo / REPORT / "F70_ADMISSION_AUDIT_CONTRACT.json", {
        "status": "registered_before_historical_labels",
        "stage": "FINAL-E20 F70 历史准入信号审计",
        "purpose": "E16/E19 已表明用户内候选改善但跨用户元模型未命中正例；只在三个历史前向窗比较已有单变量与少量固定融合，定位是否存在可重复的准入信号。",
        "unit": "每名具有 Cold50 候选的用户一行；商品固定为 F70 用户内第一名，第12位固定为被替换商品。",
        "signals": "F70 分数与用户内分差、13项截止日前商品时效特征、候选商品跨用户支持数、B0名次、两项第12位安全度，以及预先固定的时效/安全倒数名次融合。",
        "denominator": "每个日期全部具有 Cold50 候选的用户；执行比例分别为0.1%、0.2%、0.5%、1%、2%、5%。",
        "selection_rule": "审计本身不晋级；若出现三窗至少两窗不退化、累计插入不少于2且不低于误删的规则，才可登记一个冻结开发窗对照。",
        "historical_validation": VALID, "development_windows": "not_read",
        "final_week": "not_run", "no_training": True, "no_commit_push": True,
    })


def run(repo: Path):
    repo = Path(repo)
    register(repo)
    start = time.perf_counter()
    windows = {}
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        score = np.load(repo / E16 / f"{cutoff}-F70-scores.npy")
        x, mapping = final_matrix(repo, cutoff, data)
        chosen, margin2 = candidate_choice(data, score)
        customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
        item = data["cold"].iloc[chosen].article_id.to_numpy(str)
        rank12 = data["warm"].iloc[11::12].reset_index(drop=True).iloc[
            data["cold"].iloc[chosen].user_index.to_numpy(np.int64)
        ]
        counts = dict(zip(*np.unique(item, return_counts=True)))
        values = {"f70_score": score[chosen], "f70_margin_rank2": margin2}
        for index, name in enumerate(TEMPORAL_NAMES):
            value = x[chosen, -len(TEMPORAL_NAMES) + index].astype(float)
            if name in ("days_since_last_sale", "days_since_first_sale"):
                value = -value
            values[name] = value
        values["top1_item_user_support"] = np.asarray([counts[v] for v in item], dtype=float)
        values["b0_rank_quality"] = -data["cold"].iloc[chosen].b0_rank.to_numpy(float)
        values["rank12_warm_safety"] = 1 - rank12.warm_user_percentile.to_numpy(float)
        values["rank12_qw_safety"] = 1 - rank12.qW_within_user_percentile.to_numpy(float)
        ranks = {name: percentile_score(value, customer) for name, value in values.items()}
        values["rrf_f70_events7"] = ranks["f70_score"] + ranks["item_events_7d"]
        values["rrf_margin_events7"] = ranks["f70_margin_rank2"] + ranks["item_events_7d"]
        values["rrf_f70_velocity7"] = ranks["f70_score"] + ranks["item_velocity_7d_vs_28d"]
        values["rrf_f70_events7_warm"] = (
            ranks["f70_score"] + ranks["item_events_7d"] + ranks["rank12_warm_safety"]
        )
        values["rrf_f70_events7_qw"] = (
            ranks["f70_score"] + ranks["item_events_7d"] + ranks["rank12_qw_safety"]
        )
        result = {}
        for name, value in values.items():
            result[name] = {f"q={q:g}": evaluate(data, chosen, value, q) for q in QUOTAS}
        windows[cutoff] = {"mapping": mapping, "rules": result}
    aggregate = {}
    for name in windows[VALID[0]]["rules"]:
        for quota in QUOTAS:
            key = f"{name}|q={quota:g}"
            rows = [windows[t]["rules"][name][f"q={quota:g}"] for t in VALID]
            aggregate[key] = {
                "signal": name, "quota": quota,
                "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                "inserted": int(sum(r["inserted"] for r in rows)),
                "removed": int(sum(r["removed"] for r in rows)),
                "admissions": int(sum(r["admissions"] for r in rows)),
            }
    eligible = [
        (key, value) for key, value in aggregate.items()
        if value["mean_delta"] > 0 and value["nonnegative"] >= 2
        and value["inserted"] >= 2 and value["inserted"] >= value["removed"]
    ]
    eligible.sort(key=lambda pair: (-pair[1]["mean_delta"], pair[1]["removed"], pair[1]["admissions"], pair[0]))
    output = {
        "status": "completed", "stage": "FINAL-E20 F70 历史准入信号审计",
        "windows": windows, "aggregate": aggregate,
        "eligible_for_one_frozen_development_check": eligible[:10],
        "development_windows": "not_read", "models_fit": 0,
        "final_week": "not_run", "seconds": time.perf_counter() - start,
    }
    dump(repo / REPORT / "F70_ADMISSION_AUDIT.json", output)
    print(json.dumps({"eligible": eligible[:10], "seconds": output["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"F70_ADMISSION_AUDIT_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
