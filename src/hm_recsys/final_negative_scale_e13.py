"""FINAL-E13: denser real-Cold negatives for high-precision candidate admission."""
from pathlib import Path
import gc
import json
import math
import shutil
import subprocess
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .final_b0_admission_e4 import segment_metrics
from .final_binary_admission_e5 import one_per_user
from .final_candidate_e2 import VALID, WINDOWS
from .final_full_cold_e8 import PARAMS, PREPARED, action_scores, earlier_full, rrf
from .final_multiview_cold_e9 import (
    FEATURES, candidate_path, feature_arrays, feature_dir, final_matrix,
)
from .final_oracle_audit import dump
from .final_real_action_e10 import fit as fit_action
from .final_pseudo_action_e6 import score_real
from .final_relation_pilot import identify
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN = Path("artifacts/final/negative-scale-e13-v1")
REPORT = Path("reports/final")
E10 = Path("artifacts/final/real-action-e10-v1")
MODES = ("N10_best", "N10_rrf025_best")
VICTIMS = ("old_action", "real_risk")
QUOTAS = (.001, .005, .01, .02)
ALPHAS = (.05, .10, .25, .50, 1.0)


def sampled_full10(repo: Path, cutoff: str):
    with np.load(candidate_path(repo, cutoff), mmap_mode="r") as candidates:
        user = candidates["user_index"]
        item = candidates["catalog_row"]
        target = candidates["target"].astype(np.int8)
        seed = np.uint64(int(cutoff.replace("-", "")))
        mix = ((user.astype(np.uint64) * np.uint64(11400714819323198485)) ^
               (item.astype(np.uint64) * np.uint64(14029467366897019727)) ^ seed)
        keep = (target > 0) | ((mix % np.uint64(10)) == 0)
        users = user[keep]
        y = target[keep]
        full_rows = len(target)
    m4, p33_rel, p33_global, user_state, candidate_state = feature_arrays(feature_dir(repo, cutoff))
    x = np.concatenate([
        m4[keep], p33_rel[keep], p33_global[keep], user_state[users], candidate_state[keep]
    ], axis=1).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    assert x.shape == (len(y), len(FEATURES))
    return x, y, {
        "cutoff": cutoff, "full_rows": int(full_rows), "retained_rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "negative_sampling": "按用户与商品稳定散列保留10%负例；正例全部保留",
    }


def fit(repo: Path, cutoff: str, folder: Path):
    xs, ys, sources = [], [], []
    for date in earlier_full(cutoff):
        x, y, audit = sampled_full10(repo, date)
        xs.append(x)
        ys.append(y)
        sources.append(audit)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    model = lgb.LGBMClassifier(**PARAMS)
    with threadpool_limits(limits=4):
        model.fit(x, y, feature_name=FEATURES)
    model.booster_.save_model(str(folder / f"{cutoff}-F57N10.txt"))
    info = {
        "rows": int(len(y)), "positives": int(y.sum()),
        "sampled_positive_rate": float(y.mean()),
        "training": earlier_full(cutoff), "sources": sources,
    }
    del x, y, xs, ys
    gc.collect()
    return model, info


def choose(data: dict, score: np.ndarray, mode: str):
    if mode == "N10_best":
        order = score
    elif mode == "N10_rrf025_best":
        order = rrf(data, score, .25)
    else:
        raise KeyError(mode)
    return one_per_user(data, order)


def evaluate(data: dict, candidate_score: np.ndarray, victim_score: np.ndarray,
             mode: str, quota: float):
    chosen = choose(data, candidate_score, mode)
    slots = np.argmax(victim_score[chosen], axis=1)
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    count = max(1, int(math.ceil(quota * len(chosen))))
    selected = np.lexsort((customer, -candidate_score[chosen]))[:count]
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
            "victim": victim, "slot": slot + 1,
            "candidate_score": float(candidate_score[candidate]),
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


def victim_scores(repo: Path, cutoff: str, data: dict, historical: bool, victim: str, folder: Path):
    if victim == "old_action":
        return action_scores(repo, cutoff, data, historical), None
    if historical:
        return np.load(repo / E10 / f"{cutoff}-real_risk_separated-scores.npy"), None
    model, info = fit_action(repo, cutoff, "real_risk_separated", folder)
    scores = score_real(repo, cutoff, data, model)
    del model
    gc.collect()
    return scores, info


def register(repo: Path):
    contract = {
        "status": "preregistered_before_E13_fit",
        "stage": "FINAL-E13 真实 Cold 负例扩密",
        "observed_bottleneck": "E8/E9 对完整候选池只抽1%负例，样本正例率被大幅抬高；Top5 局部改善但跨用户高分头部不可靠。",
        "hypothesis": "把负例保留率从1%提高到10%，让模型看到十倍更多真实易混淆候选，可提高 Cold 商品 Top1 与跨用户准入精度。",
        "features": "沿用 E9 的57维多视角真实 Cold 特征，特征定义与时间边界不变。",
        "training_population": "P3.7B 每个严格较早日期的全部正例加稳定散列10%负例；不按未来标签筛选用户。",
        "parameters": PARAMS,
        "candidate_modes": {
            "N10_best": "10%负例模型的用户内第一名。",
            "N10_rrf025_best": "10%负例模型名次与 B0 名次按0.25权重倒数名次融合后的第一名。",
        },
        "victim_modes": {
            "old_action": "原 action 模型选择 Warm Top12 被替换位置。",
            "real_risk": "E10 真实风险分离动作模型选择被替换位置。",
        },
        "admission": "所有组合均按所选 Cold 商品的 N10 分数跨用户排序，执行头部0.1%、0.5%、1%、2%。",
        "historical_gate": "三窗 mean delta_MAP>0、至少两窗不退化、插入正例不少于2且插入数不少于误删数。",
        "development_gate": "历史通过后冻结唯一组合，运行既有四窗扩量门槛。",
        "if_fail": "保留 WV3-741；不扩量。",
        "cost_limit": "本机内存预计低于2 GiB、CPU 预计30分钟内；不重训 embedding。",
        "historical_validation": VALID, "development_windows": WINDOWS,
        "final_week": "not_run", "no_commit_push": True,
        "free_disk_gib": shutil.disk_usage(repo).free / 2**30,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(repo / REPORT / "NEGATIVE_SCALE_E13_CONTRACT.json", contract)
    return contract


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E13")
    (repo / RUN).mkdir(parents=True)
    contract = register(repo)
    start = time.perf_counter()
    historical, aggregate = {}, {}
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        model, fit_info = fit(repo, cutoff, repo / RUN)
        x, mapping = final_matrix(repo, cutoff, data)
        score = model.booster_.predict(x, num_threads=4)
        np.save(repo / RUN / f"{cutoff}-F57N10-scores.npy", score)
        historical[cutoff] = {
            "fit": fit_info, "mapping": mapping,
            "B0": identify(data, -data["cold"].b0_rank.to_numpy(float)),
            "N10": identify(data, score), "rules": {},
        }
        for alpha in ALPHAS:
            historical[cutoff][f"N10-rrf-{alpha:g}"] = identify(data, rrf(data, score, alpha))
        for victim in VICTIMS:
            victim_score, _ = victim_scores(repo, cutoff, data, True, victim, repo / RUN)
            for mode in MODES:
                for quota in QUOTAS:
                    key = f"{victim}|{mode}|q={quota:g}"
                    historical[cutoff]["rules"][key], _, _ = evaluate(
                        data, score, victim_score, mode, quota,
                    )
            del victim_score
        print("E13", cutoff, historical[cutoff]["N10"], flush=True)
        del data, model, x, score
        gc.collect()
    for victim in VICTIMS:
        for mode in MODES:
            for quota in QUOTAS:
                key = f"{victim}|{mode}|q={quota:g}"
                rows = [historical[t]["rules"][key] for t in VALID]
                aggregate[key] = {
                    "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                    "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                    "admissions": int(sum(r["admissions"] for r in rows)),
                    "inserted": int(sum(r["inserted"] for r in rows)),
                    "removed": int(sum(r["removed"] for r in rows)),
                    "beneficial": int(sum(r["beneficial"] for r in rows)),
                    "harmful": int(sum(r["harmful"] for r in rows)),
                    "victim": victim, "mode": mode, "quota": quota,
                }
    eligible = [(key, value) for key, value in aggregate.items()
                if value["mean_delta"] > 0 and value["nonnegative"] >= 2 and
                value["inserted"] >= 2 and value["inserted"] >= value["removed"]]
    eligible.sort(key=lambda pair: (-pair[1]["mean_delta"], pair[1]["removed"],
                                    pair[1]["admissions"], VICTIMS.index(pair[1]["victim"]),
                                    MODES.index(pair[1]["mode"])))
    selected = eligible[0][0] if eligible else None
    development = {}
    gates = {"not_run": "historical gate failed"}
    passed = False
    if selected:
        spec = aggregate[selected]
        for window, cutoff in WINDOWS.items():
            data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
            model, fit_info = fit(repo, cutoff, repo / RUN)
            x, mapping = final_matrix(repo, cutoff, data)
            score = model.booster_.predict(x, num_threads=4)
            victim_score, victim_info = victim_scores(
                repo, cutoff, data, False, spec["victim"], repo / RUN,
            )
            result, lists, events = evaluate(
                data, score, victim_score, spec["mode"], spec["quota"],
            )
            result["segments"] = segment_metrics(data, lists)
            result["fit"] = {"candidate": fit_info, "victim": victim_info, "mapping": mapping}
            development[window] = result
            save_frame(events, repo / RUN / f"{window}-events.parquet")
            del data, model, x, score, victim_score, lists, events
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
    dump(repo / REPORT / "NEGATIVE_SCALE_E13.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"NEGATIVE_SCALE_E13_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
