"""FINAL-E17: F70 Cold candidate with explicit rank-12 Warm risk admission."""
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


RUN = Path("artifacts/final/tail-risk-e17-v1")
REPORT = Path("reports/final")
E16 = Path("artifacts/final/temporal-item-e16-v1")
RULES = ("candidate", "rrf_warm", "rrf_qw", "safe_half_warm", "safe_half_qw")
QUOTAS = (.005, .01, .02)


def rank_descending(values: np.ndarray, customer: np.ndarray):
    order = np.lexsort((customer, -values))
    rank = np.empty(len(values), dtype=np.int32)
    rank[order] = np.arange(1, len(values) + 1)
    return rank


def decisions(data: dict, score: np.ndarray, rule: str):
    chosen = one_per_user(data, score)
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    candidate_value = score[chosen]
    warm = data["warm"]
    rank12 = warm.iloc[11::12].reset_index(drop=True)
    # Candidate-bearing users are a subset of the full user roster.  The chosen
    # row carries the original roster index, which selects the aligned rank-12 row.
    user_index = data["cold"].iloc[chosen].user_index.to_numpy(np.int64)
    rank12 = rank12.iloc[user_index]
    warm_vulnerability = 1 - rank12.warm_user_percentile.to_numpy(float)
    qw_vulnerability = 1 - rank12.qW_within_user_percentile.to_numpy(float)
    warm_vulnerability[~np.isfinite(warm_vulnerability)] = 0.
    qw_vulnerability[~np.isfinite(qw_vulnerability)] = 0.
    keep = np.ones(len(chosen), dtype=bool)
    if rule == "candidate":
        value = candidate_value
    elif rule == "rrf_warm":
        value = (1 / (60 + rank_descending(candidate_value, customer)) +
                 1 / (60 + rank_descending(warm_vulnerability, customer)))
    elif rule == "rrf_qw":
        value = (1 / (60 + rank_descending(candidate_value, customer)) +
                 1 / (60 + rank_descending(qw_vulnerability, customer)))
    elif rule == "safe_half_warm":
        keep = warm_vulnerability >= np.nanmedian(warm_vulnerability)
        value = candidate_value
    elif rule == "safe_half_qw":
        keep = qw_vulnerability >= np.nanmedian(qw_vulnerability)
        value = candidate_value
    else:
        raise KeyError(rule)
    return chosen[keep], value[keep], {
        "candidate_users": int(len(chosen)), "eligible_users": int(keep.sum()),
        "mean_rank12_warm_vulnerability": float(warm_vulnerability[keep].mean()),
        "mean_rank12_qw_vulnerability": float(qw_vulnerability[keep].mean()),
    }


def evaluate(data: dict, score: np.ndarray, rule: str, quota: float):
    chosen, values, audit = decisions(data, score, rule)
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
    result = {
        **audit, "delta_map": float(events.delta.sum() / len(data["users"])),
        "admissions": int(len(events)), "beneficial": int(events.delta.gt(0).sum()),
        "harmful": int(events.delta.lt(0).sum()), "neutral": int(events.delta.eq(0).sum()),
        "inserted": int(events.inserted.sum()), "removed": int(events.removed.sum()),
    }
    return result, lists, events


def register(repo: Path):
    contract = {
        "status": "preregistered_before_E17_evaluation",
        "stage": "FINAL-E17 第12位 Warm 风险约束",
        "observed_bottleneck": "E16 四窗已插入5个 Cold 正例，但固定第12位误删9个 Warm 正例，整体 mean delta_MAP 仍略负。",
        "hypothesis": "保留 F70 的 Cold 商品与固定第12位，只在第12位 Warm 模型置信较低的用户准入，可减少误删而不破坏 Cold 候选增益。",
        "warm_vulnerability": "1-warm_user_percentile；数值越高表示原 Warm 排序器认为第12位越弱。行业常用的脆弱度变换。",
        "qw_vulnerability": "1-qW_within_user_percentile；数值越高表示 P4.2 的 Warm 保留概率模型在用户内认为第12位越弱。本项目自定义风险量。",
        "rules": {
            "candidate": "E16 对照：只按 F70 Cold 分数。",
            "rrf_warm": "F70 分数跨用户名次与 Warm 脆弱度名次等权倒数名次融合。",
            "rrf_qw": "F70 分数跨用户名次与 qW 脆弱度名次等权倒数名次融合。",
            "safe_half_warm": "只保留 Warm 脆弱度不低于当窗中位数的候选用户，再按 F70 分数。",
            "safe_half_qw": "只保留 qW 脆弱度不低于当窗中位数的候选用户，再按 F70 分数。",
        },
        "victim": "固定替换原 WV3 第12位。",
        "quotas": "相对各规则合格用户执行头部0.5%、1%、2%。",
        "historical_gate": "三窗 mean delta_MAP>0、至少两窗不退化、插入正例不少于2且插入数不少于误删数。",
        "development_gate": "历史通过后冻结唯一规则，运行既有四窗扩量门槛。",
        "if_fail": "保留 WV3-741；不扩量。",
        "cost_limit": "历史仅复用 E16 分数；若通过，四窗只重算 F70，预计15分钟内。",
        "historical_validation": VALID, "development_windows": WINDOWS,
        "final_week": "not_run", "no_commit_push": True,
        "free_disk_gib": shutil.disk_usage(repo).free / 2**30,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(repo / REPORT / "TAIL_RISK_E17_CONTRACT.json", contract)
    return contract


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E17")
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
    dump(repo / REPORT / "TAIL_RISK_E17.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"TAIL_RISK_E17_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
