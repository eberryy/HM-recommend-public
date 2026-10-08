"""FINAL-E14: consensus-gated N10 candidate with real-risk victim selection."""
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


RUN = Path("artifacts/final/consensus-e14-v1")
REPORT = Path("reports/final")
E8 = Path("artifacts/final/full-cold-e8-v2")
E9 = Path("artifacts/final/multiview-cold-e9-v3")
E10 = Path("artifacts/final/real-action-e10-v1")
E13 = Path("artifacts/final/negative-scale-e13-v1")
AGREEMENTS = (2, 3)
ADMISSIONS = ("action", "candidate", "rrf")
QUOTAS = (.01, .05, .10, .20)


def rank_descending(values: np.ndarray, customer: np.ndarray):
    order = np.lexsort((customer, -values))
    rank = np.empty(len(values), dtype=np.int32)
    rank[order] = np.arange(1, len(values) + 1)
    return rank


def top(data: dict, score: np.ndarray):
    return one_per_user(data, score)


def decisions(data: dict, f35: np.ndarray, f57: np.ndarray, n10: np.ndarray,
              action: np.ndarray, minimum_agreement: int, admission: str):
    cold = data["cold"]
    top_b0 = top(data, -cold.b0_rank.to_numpy(float))
    top_f35 = top(data, f35)
    top_f57 = top(data, f57)
    top_n10 = top(data, n10)
    article = cold.article_id.to_numpy(str)
    selected_article = article[top_n10]
    agreement = (
        (selected_article == article[top_b0]).astype(np.int8) +
        (selected_article == article[top_f35]).astype(np.int8) +
        (selected_article == article[top_f57]).astype(np.int8)
    )
    keep = agreement >= minimum_agreement
    chosen = top_n10[keep]
    slots = np.argmax(action[chosen], axis=1)
    action_value = action[chosen, slots]
    candidate_value = n10[chosen]
    customer = cold.iloc[chosen].customer_id.to_numpy(str)
    if admission == "action":
        value = action_value
    elif admission == "candidate":
        value = candidate_value
    elif admission == "rrf":
        value = (1 / (60 + rank_descending(action_value, customer)) +
                 1 / (60 + rank_descending(candidate_value, customer)))
    else:
        raise KeyError(admission)
    return chosen, slots, value, agreement


def evaluate(data: dict, f35: np.ndarray, f57: np.ndarray, n10: np.ndarray,
             action: np.ndarray, minimum_agreement: int, admission: str, quota: float):
    chosen, slots, values, agreement = decisions(
        data, f35, f57, n10, action, minimum_agreement, admission,
    )
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
        "eligible_users": int(len(chosen)), "delta_map": float(events.delta.sum() / len(data["users"])),
        "admissions": int(len(events)), "beneficial": int(events.delta.gt(0).sum()),
        "harmful": int(events.delta.lt(0).sum()), "neutral": int(events.delta.eq(0).sum()),
        "inserted": int(events.inserted.sum()), "removed": int(events.removed.sum()),
        "agreement_population": {
            "at_least_2": int((agreement >= 2).sum()), "exactly_3": int((agreement == 3).sum()),
        },
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
        "status": "preregistered_before_E14_evaluation",
        "stage": "FINAL-E14 多排序器共识准入",
        "observed_bottleneck": "N10 三个历史窗用户内第一名共13个正例，但按绝对分数取头部时冬窗前10%仍无正例；绝对分数不能跨用户稳定排序。只读审计显示 N10 与 B0/F35/F57 三者全部一致时为5/2178，正例密度约为总体的3.2倍。",
        "hypothesis": "先用跨模型用户内第一名共识定义较可靠候选人群，再在该人群内用候选分数、动作风险或二者等权名次融合排序，可绕开跨用户原始概率失准。",
        "agreement": "N10 第一名同时等于 B0、F35、F57 第一名的模型个数；检验至少2个与全部3个一致。该计数不使用未来标签。",
        "candidate": "固定为 N10 用户内第一名。",
        "victim": "E10 真实风险分离动作模型在该 Cold 商品的12个 Warm 位置中选预测风险最低且期望变化最高的位置。",
        "admission": {
            "action": "在共识人群内按动作预测期望变化排序。",
            "candidate": "在共识人群内按 N10 候选分数排序。",
            "rrf": "在共识人群内等权融合动作分数名次与 N10 分数名次。",
        },
        "quotas": "相对共识人群执行头部1%、5%、10%、20%，而不是相对全部用户。",
        "historical_gate": "三窗 mean delta_MAP>0、至少两窗不退化、插入正例不少于2且插入数不少于误删数。",
        "development_gate": "历史通过后冻结唯一规则，运行既有四窗扩量门槛。",
        "if_fail": "保留 WV3-741；不扩量。",
        "cost_limit": "历史只复用已有分数；若通过，四窗预计30分钟内。",
        "historical_validation": VALID, "development_windows": WINDOWS,
        "final_week": "not_run", "no_commit_push": True,
        "free_disk_gib": shutil.disk_usage(repo).free / 2**30,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(repo / REPORT / "CONSENSUS_E14_CONTRACT.json", contract)
    return contract


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E14")
    (repo / RUN).mkdir(parents=True)
    contract = register(repo)
    start = time.perf_counter()
    historical, aggregate = {}, {}
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        f35, f57, n10, action = historical_scores(repo, cutoff)
        historical[cutoff] = {}
        for agreement in AGREEMENTS:
            historical[cutoff][str(agreement)] = {}
            for admission in ADMISSIONS:
                historical[cutoff][str(agreement)][admission] = {}
                for quota in QUOTAS:
                    historical[cutoff][str(agreement)][admission][f"q={quota:g}"], _, _ = evaluate(
                        data, f35, f57, n10, action, agreement, admission, quota,
                    )
        del data, f35, f57, n10, action
        gc.collect()
    for agreement in AGREEMENTS:
        for admission in ADMISSIONS:
            for quota in QUOTAS:
                key = f"agreement={agreement}|{admission}|q={quota:g}"
                rows = [historical[t][str(agreement)][admission][f"q={quota:g}"] for t in VALID]
                aggregate[key] = {
                    "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                    "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                    "admissions": int(sum(r["admissions"] for r in rows)),
                    "inserted": int(sum(r["inserted"] for r in rows)),
                    "removed": int(sum(r["removed"] for r in rows)),
                    "beneficial": int(sum(r["beneficial"] for r in rows)),
                    "harmful": int(sum(r["harmful"] for r in rows)),
                    "agreement": agreement, "admission": admission, "quota": quota,
                }
    eligible = [(key, value) for key, value in aggregate.items()
                if value["mean_delta"] > 0 and value["nonnegative"] >= 2 and
                value["inserted"] >= 2 and value["inserted"] >= value["removed"]]
    eligible.sort(key=lambda pair: (-pair[1]["mean_delta"], pair[1]["removed"],
                                    pair[1]["admissions"], -pair[1]["agreement"],
                                    ADMISSIONS.index(pair[1]["admission"])))
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
                data, f35, f57, n10, action, spec["agreement"],
                spec["admission"], spec["quota"],
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
    dump(repo / REPORT / "CONSENSUS_E14.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"CONSENSUS_E14_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
