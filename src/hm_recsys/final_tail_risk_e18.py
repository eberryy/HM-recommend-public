"""FINAL-E18: stricter rank-12 Warm-risk gates after E17 localization."""
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
from .final_binary_admission_e5 import one_per_user
from .final_candidate_e2 import VALID, WINDOWS
from .final_oracle_audit import dump
from .final_temporal_item_e16 import PREPARED, fit as fit_f70, final_matrix
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN = Path("artifacts/final/tail-risk-e18-v1")
REPORT = Path("reports/final")
E16 = Path("artifacts/final/temporal-item-e16-v1")
RULES = ("safe_quarter_warm", "safe_quarter_qw", "safe_half_both")
QUOTAS = (.01, .02, .05)


def choices(data: dict, score: np.ndarray, rule: str):
    chosen = one_per_user(data, score)
    user_index = data["cold"].iloc[chosen].user_index.to_numpy(np.int64)
    rank12 = data["warm"].iloc[11::12].reset_index(drop=True).iloc[user_index]
    warm_vulnerability = 1 - rank12.warm_user_percentile.to_numpy(float)
    qw_vulnerability = 1 - rank12.qW_within_user_percentile.to_numpy(float)
    warm_vulnerability[~np.isfinite(warm_vulnerability)] = 0.
    qw_vulnerability[~np.isfinite(qw_vulnerability)] = 0.
    if rule == "safe_quarter_warm":
        keep = warm_vulnerability >= np.quantile(warm_vulnerability, .75)
    elif rule == "safe_quarter_qw":
        keep = qw_vulnerability >= np.quantile(qw_vulnerability, .75)
    elif rule == "safe_half_both":
        keep = ((warm_vulnerability >= np.median(warm_vulnerability)) &
                (qw_vulnerability >= np.median(qw_vulnerability)))
    else:
        raise KeyError(rule)
    return chosen[keep], score[chosen[keep]], {
        "candidate_users": int(len(chosen)), "eligible_users": int(keep.sum()),
        "mean_rank12_warm_vulnerability": float(warm_vulnerability[keep].mean()),
        "mean_rank12_qw_vulnerability": float(qw_vulnerability[keep].mean()),
    }


def evaluate(data: dict, score: np.ndarray, rule: str, quota: float):
    chosen, values, audit = choices(data, score, rule)
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    count = max(1, int(math.ceil(quota * len(chosen))))
    selected = np.lexsort((customer, -values))[:count]
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
            "victim": victim, "slot": 12, "admission_value": float(values[local]),
            "delta": float(delta), "inserted": int(row.target),
            "removed": int(victim in data["truthsets"][data["users"][user]]),
        })
    events = pd.DataFrame(events)
    return {
        **audit, "delta_map": float(events.delta.sum() / len(data["users"])),
        "admissions": int(len(events)), "beneficial": int(events.delta.gt(0).sum()),
        "harmful": int(events.delta.lt(0).sum()), "neutral": int(events.delta.eq(0).sum()),
        "inserted": int(events.inserted.sum()), "removed": int(events.removed.sum()),
    }, lists, events


def register(repo: Path):
    contract = {
        "status": "preregistered_before_E18_evaluation",
        "stage": "FINAL-E18 更严格的第12位风险门",
        "observed_bottleneck": "E17 的半数 Warm 安全门把 E16 四窗误删从9降到4，同时保留3个 Cold 正例；仍差一次误删才满足效率。",
        "hypothesis": "把第12位安全约束收紧到最安全四分之一，或同时满足 Warm 与 qW 两个半数安全门，可进一步减少误删且保留至少一个 Cold 正例。",
        "rules": {
            "safe_quarter_warm": "只保留第12位 Warm 脆弱度位于当窗最高四分之一的候选用户。",
            "safe_quarter_qw": "只保留第12位 qW 脆弱度位于当窗最高四分之一的候选用户。",
            "safe_half_both": "第12位同时满足 Warm 脆弱度和 qW 脆弱度不低于各自中位数。",
        },
        "candidate": "冻结 E16 F70 用户内第一名；按 F70 分数排序。",
        "victim": "固定原 WV3 第12位。",
        "quotas": "相对风险门后合格用户执行头部1%、2%、5%。",
        "historical_gate": "三窗 mean delta_MAP>0、至少两窗不退化、插入正例不少于2且插入数不少于误删数。",
        "development_gate": "历史通过后冻结唯一规则，运行既有四窗扩量门槛。",
        "if_fail": "保留 WV3-741；不扩量并停止策略搜索。",
        "cost_limit": "历史复用分数；通过后四窗预计15分钟内。",
        "historical_validation": VALID, "development_windows": WINDOWS,
        "final_week": "not_run", "no_commit_push": True,
        "free_disk_gib": shutil.disk_usage(repo).free / 2**30,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(repo / REPORT / "TAIL_RISK_E18_CONTRACT.json", contract)
    return contract


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E18")
    (repo / RUN).mkdir(parents=True)
    contract = register(repo)
    start = time.perf_counter()
    historical, aggregate = {}, {}
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        score = np.load(repo / E16 / f"{cutoff}-F70-scores.npy")
        historical[cutoff] = {}
        for rule in RULES:
            historical[cutoff][rule] = {}
            for quota in QUOTAS:
                historical[cutoff][rule][f"q={quota:g}"], _, _ = evaluate(data, score, rule, quota)
        del data, score
        gc.collect()
    for rule in RULES:
        for quota in QUOTAS:
            key = f"{rule}|q={quota:g}"
            rows = [historical[t][rule][f"q={quota:g}"] for t in VALID]
            aggregate[key] = {
                "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                "admissions": int(sum(r["admissions"] for r in rows)),
                "inserted": int(sum(r["inserted"] for r in rows)),
                "removed": int(sum(r["removed"] for r in rows)),
                "beneficial": int(sum(r["beneficial"] for r in rows)),
                "harmful": int(sum(r["harmful"] for r in rows)),
                "rule": rule, "quota": quota,
            }
    eligible = [(key, value) for key, value in aggregate.items()
                if value["mean_delta"] > 0 and value["nonnegative"] >= 2 and
                value["inserted"] >= 2 and value["inserted"] >= value["removed"]]
    eligible.sort(key=lambda pair: (-pair[1]["mean_delta"], pair[1]["removed"],
                                    pair[1]["admissions"], RULES.index(pair[1]["rule"])))
    selected = eligible[0][0] if eligible else None
    development = {}
    gates = {"not_run": "historical gate failed"}
    passed = False
    if selected:
        spec = aggregate[selected]
        for window, cutoff in WINDOWS.items():
            data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
            model, fit_info = fit_f70(repo, cutoff, repo / RUN)
            x, mapping = final_matrix(repo, cutoff, data)
            score = model.booster_.predict(x, num_threads=4)
            result, lists, events = evaluate(data, score, spec["rule"], spec["quota"])
            result["segments"] = segment_metrics(data, lists)
            result["fit"] = {"candidate": fit_info, "mapping": mapping}
            development[window] = result
            save_frame(events, repo / RUN / f"{window}-events.parquet")
            del data, model, x, score, lists, events
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
    dump(repo / REPORT / "TAIL_RISK_E18.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"TAIL_RISK_E18_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
