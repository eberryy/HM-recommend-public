"""FINAL-E15: N10 candidate admission with fixed rank-12 replacement."""
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
from .final_full_cold_e8 import fit as fit35, final_matrix as matrix35
from .final_multiview_cold_e9 import fit as fit57, final_matrix as matrix57
from .final_negative_scale_e13 import fit as fit_n10
from .final_oracle_audit import dump
from .final_real_action_e10 import fit as fit_action
from .final_pseudo_action_e6 import PREPARED, score_real
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN = Path("artifacts/final/safe-tail-e15-v1")
REPORT = Path("reports/final")
E8 = Path("artifacts/final/full-cold-e8-v2")
E9 = Path("artifacts/final/multiview-cold-e9-v3")
E10 = Path("artifacts/final/real-action-e10-v1")
E13 = Path("artifacts/final/negative-scale-e13-v1")
RULES = ("all_candidate", "all_action", "agreement2_action", "agreement3_action")
QUOTAS = (.01, .02, .05, .10)


def choices(data: dict, f35: np.ndarray, f57: np.ndarray, n10: np.ndarray,
            action: np.ndarray, rule: str):
    cold = data["cold"]
    top_n10 = one_per_user(data, n10)
    article = cold.article_id.to_numpy(str)
    agreement = np.zeros(len(top_n10), dtype=np.int8)
    for score in (-cold.b0_rank.to_numpy(float), f35, f57):
        agreement += (article[top_n10] == article[one_per_user(data, score)]).astype(np.int8)
    if rule.startswith("agreement2"):
        keep = agreement >= 2
    elif rule.startswith("agreement3"):
        keep = agreement >= 3
    else:
        keep = np.ones(len(top_n10), dtype=bool)
    chosen = top_n10[keep]
    if rule.endswith("candidate"):
        value = n10[chosen]
    else:
        value = action[chosen].max(axis=1)
    return chosen, value, agreement


def evaluate(data: dict, f35: np.ndarray, f57: np.ndarray, n10: np.ndarray,
             action: np.ndarray, rule: str, quota: float):
    chosen, values, agreement = choices(data, f35, f57, n10, action, rule)
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    count = max(1, int(math.ceil(quota * len(chosen))))
    selected = np.lexsort((customer, -values))[:count]
    lists = data["warm_lists"].copy()
    events = []
    for local in selected:
        candidate = int(chosen[local])
        row = data["cold"].iloc[candidate]
        user = int(row.user_index)
        slot = 11
        victim = lists[user, slot]
        lists[user, slot] = row.article_id
        delta = exact_ap(lists[user], data["truthsets"][data["users"][user]]) - data["baseline_ap"][user]
        events.append({
            "customer_id": data["users"][user], "candidate": row.article_id,
            "victim": victim, "slot": 12, "admission_value": float(values[local]),
            "delta": float(delta), "inserted": int(row.target),
            "removed": int(victim in data["truthsets"][data["users"][user]]),
        })
    events = pd.DataFrame(events)
    return {
        "eligible_users": int(len(chosen)), "delta_map": float(events.delta.sum() / len(data["users"])),
        "admissions": int(len(events)), "beneficial": int(events.delta.gt(0).sum()),
        "harmful": int(events.delta.lt(0).sum()), "neutral": int(events.delta.eq(0).sum()),
        "inserted": int(events.inserted.sum()), "removed": int(events.removed.sum()),
        "agreement_population": {"at_least_2": int((agreement >= 2).sum()), "exactly_3": int((agreement == 3).sum())},
    }, lists, events


def historical_scores(repo: Path, cutoff: str):
    return (
        np.load(repo / E8 / f"{cutoff}-F35-scores.npy"),
        np.load(repo / E9 / f"{cutoff}-F57-scores.npy"),
        np.load(repo / E13 / f"{cutoff}-F57N10-scores.npy"),
        np.load(repo / E10 / f"{cutoff}-real_risk_separated-scores.npy"),
    )


def development_scores(repo: Path, cutoff: str, data: dict):
    folder = repo / RUN / "models"
    folder.mkdir(exist_ok=True)
    model35, info35 = fit35(repo, cutoff, folder)
    x35 = matrix35(repo, cutoff, data)
    score35 = model35.booster_.predict(x35, num_threads=4)
    model57, info57 = fit57(repo, cutoff, folder)
    x57, mapping = matrix57(repo, cutoff, data)
    score57 = model57.booster_.predict(x57, num_threads=4)
    model_n10, info_n10 = fit_n10(repo, cutoff, folder)
    score_n10 = model_n10.booster_.predict(x57, num_threads=4)
    action_model, info_action = fit_action(repo, cutoff, "real_risk_separated", folder)
    action_score = score_real(repo, cutoff, data, action_model)
    del model35, x35, model57, model_n10, x57, action_model
    gc.collect()
    return score35, score57, score_n10, action_score, {
        "F35": info35, "F57": info57, "N10": info_n10,
        "action": info_action, "heldout_feature_audit": mapping,
    }


def register(repo: Path):
    contract = {
        "status": "preregistered_before_E15_evaluation",
        "stage": "FINAL-E15 固定第12位的保守 Cold 注入",
        "observed_bottleneck": "E10/E12 的主要净损失来自误删 Warm 正例；E13 已把三窗 Cold Top1 命中提高到13个，但学得位置选择仍不稳定。",
        "hypothesis": "固定只替换原 Warm 第12位，可把误删概率与损失幅度限制在尾部；N10 负责选 Cold 商品，真实动作模型只作为跨用户安全置信，不再决定位置。",
        "candidate": "N10 用户内第一名。",
        "victim": "固定原 WV3 Warm Top12 的第12位；不做末位后移，不改变其他11位。",
        "rules": {
            "all_candidate": "全部候选用户中按 N10 商品分数排序。",
            "all_action": "全部候选用户中按该 Cold 商品12个动作的最高安全分排序，但实际仍替换第12位。",
            "agreement2_action": "N10 第一名与 B0/F35/F57 至少两个第一名一致，再按动作安全分排序。",
            "agreement3_action": "N10 第一名与 B0/F35/F57 全部一致，再按动作安全分排序。",
        },
        "quotas": "相对各规则的合格用户取头部1%、2%、5%、10%。",
        "historical_gate": "三窗 mean delta_MAP>0、至少两窗不退化、插入正例不少于2且插入数不少于误删数。",
        "development_gate": "历史通过后冻结唯一规则，运行既有四窗扩量门槛。",
        "if_fail": "保留 WV3-741；不扩量。",
        "cost_limit": "历史只复用已有分数；若通过，四窗预计30分钟内。",
        "historical_validation": VALID, "development_windows": WINDOWS,
        "final_week": "not_run", "no_commit_push": True,
        "free_disk_gib": shutil.disk_usage(repo).free / 2**30,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(repo / REPORT / "SAFE_TAIL_E15_CONTRACT.json", contract)
    return contract


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E15")
    (repo / RUN).mkdir(parents=True)
    contract = register(repo)
    start = time.perf_counter()
    historical, aggregate = {}, {}
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        f35, f57, n10, action = historical_scores(repo, cutoff)
        historical[cutoff] = {}
        for rule in RULES:
            historical[cutoff][rule] = {}
            for quota in QUOTAS:
                historical[cutoff][rule][f"q={quota:g}"], _, _ = evaluate(
                    data, f35, f57, n10, action, rule, quota,
                )
        del data, f35, f57, n10, action
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
            f35, f57, n10, action, fit_info = development_scores(repo, cutoff, data)
            result, lists, events = evaluate(
                data, f35, f57, n10, action, spec["rule"], spec["quota"],
            )
            result["segments"] = segment_metrics(data, lists)
            result["fit"] = fit_info
            development[window] = result
            save_frame(events, repo / RUN / f"{window}-events.parquet")
            del data, f35, f57, n10, action, lists, events
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
    dump(repo / REPORT / "SAFE_TAIL_E15.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"SAFE_TAIL_E15_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
