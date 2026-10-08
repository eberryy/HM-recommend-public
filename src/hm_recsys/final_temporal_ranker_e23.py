"""FINAL-E23: pairwise F70 feature ranker for Cold Top5-to-Top1 conversion."""
from __future__ import annotations

from pathlib import Path
import gc
import json
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
from .final_fixed_candidate_e12 import rank_descending
from .final_full_cold_e8 import earlier_full
from .final_multiview_cold_e9 import candidate_path, feature_arrays, feature_dir
from .final_oracle_audit import dump
from .final_relation_pilot import identify
from .final_temporal_item_e16 import (
    FEATURES, PREPARED, RUN as E16, TEMPORAL_NAMES, final_matrix, item_features,
)
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN = Path("artifacts/final/temporal-ranker-e23-v1")
REPORT = Path("reports/final")
MODES = ("r70_score", "r70_events7_qw")
QUOTAS = (.005, .01, .02)
PARAMS = {
    "objective": "lambdarank", "metric": "None", "learning_rate": .04,
    "n_estimators": 350, "num_leaves": 31, "max_depth": 7,
    "min_child_samples": 50, "subsample": 1., "colsample_bytree": 1.,
    "reg_lambda": 2., "reg_alpha": 0., "random_state": 20260914,
    "n_jobs": 4, "verbosity": -1, "deterministic": True,
    "force_col_wise": True, "label_gain": [0, 1],
}


def register(repo: Path):
    dump(repo / REPORT / "TEMPORAL_RANKER_E23_CONTRACT.json", {
        "status": "registered_before_E23_labels_or_fit",
        "stage": "FINAL-E23 F70特征组内成对排序",
        "observed_bottleneck": "F70四开发窗Cold候选Recall@5合计106/264，Recall@1仅41/264；65个正例已进入前5但没有成为用户内第一名。",
        "hypothesis": "同样70维特征改用LambdaRank的用户内成对目标，可改善Top5到Top1转化；它不承诺解决跨用户准入。",
        "unit": "训练组是一名用户在一个历史日期的保留候选；全部正例加稳定10%负例。无正例组保留但不产生有效正负排序对。",
        "model": "固定参数LightGBM LambdaRank；无参数搜索，与F70分类器使用完全相同的70维截止日前特征。",
        "candidate_gate": "三个历史前向窗R70的Top1正例合计严格高于F70，且至少两窗不减少；Top5合计不得低于F70。",
        "admission": {
            "r70_score": "按R70原始分数跨用户排序。",
            "r70_events7_qw": "R70分数、商品近7日交易次数、原第12位qW安全度的跨用户等权名次和。",
            "victim": "固定WV3-741第12位。",
            "quotas": "候选用户头部0.5%、1%、2%。",
        },
        "historical_map_gate": "candidate gate通过后，mean delta_MAP>0、至少两窗不退化、累计插入不少于2且不少于误删；唯一选择mean最高项。",
        "development_gate": "历史两道门均通过才冻结唯一规则跑四开发窗FINAL-WRAPUP门槛。",
        "if_fail": "保留WV3-741；F70分类器保留为候选层结果，并停止本轮Cold探索。",
        "historical_validation": VALID, "development_windows": WINDOWS,
        "final_week": "not_run", "no_commit_push": True,
    })


def sampled_grouped(repo: Path, cutoff: str):
    with np.load(candidate_path(repo, cutoff), mmap_mode="r") as candidates:
        user = candidates["user_index"]
        item = candidates["catalog_row"]
        target = candidates["target"].astype(np.int8)
        seed = np.uint64(int(cutoff.replace("-", "")))
        mix = ((user.astype(np.uint64) * np.uint64(11400714819323198485)) ^
               (item.astype(np.uint64) * np.uint64(14029467366897019727)) ^ seed)
        keep = (target > 0) | ((mix % np.uint64(10)) == 0)
        users = user[keep]
        items = item[keep]
        y = target[keep]
        full_rows = len(target)
    m4, p33_rel, p33_global, user_state, candidate_state = feature_arrays(feature_dir(repo, cutoff))
    temporal = item_features(repo, cutoff)
    x = np.concatenate([
        m4[keep], p33_rel[keep], p33_global[keep], user_state[users],
        candidate_state[keep], temporal[items],
    ], axis=1).astype(np.float32)
    order = np.argsort(users, kind="stable")
    x = x[order]
    y = y[order]
    users = users[order]
    _, group = np.unique(users, return_counts=True)
    x[~np.isfinite(x)] = np.nan
    return x, y, group.astype(np.int32), {
        "cutoff": cutoff, "full_rows": int(full_rows), "retained_rows": int(len(y)),
        "positive_rows": int(y.sum()), "groups": int(len(group)),
        "groups_with_positive": int(np.unique(users[y > 0]).size),
        "negative_sampling": "全部正例加稳定散列10%负例",
    }


def fit(repo: Path, cutoff: str, folder: Path):
    xs, ys, groups, sources = [], [], [], []
    for date in earlier_full(cutoff):
        x, y, group, audit = sampled_grouped(repo, date)
        xs.append(x); ys.append(y); groups.extend(group.tolist()); sources.append(audit)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    model = lgb.LGBMRanker(**PARAMS)
    with threadpool_limits(limits=4):
        model.fit(x, y, group=groups, feature_name=FEATURES)
    model.booster_.save_model(str(folder / f"{cutoff}-R70.txt"))
    info = {
        "rows": int(len(y)), "positives": int(y.sum()), "groups": int(len(groups)),
        "training": earlier_full(cutoff), "sources": sources,
    }
    del x, y, xs, ys
    gc.collect()
    return model, info


def choose_and_value(data: dict, score: np.ndarray, x: np.ndarray, mode: str):
    chosen = one_per_user(data, score)
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    if mode == "r70_score":
        return chosen, score[chosen]
    event7_index = -len(TEMPORAL_NAMES) + TEMPORAL_NAMES.index("item_events_7d")
    event7 = x[chosen, event7_index]
    user_index = data["cold"].iloc[chosen].user_index.to_numpy(np.int64)
    rank12 = data["warm"].iloc[11::12].reset_index(drop=True).iloc[user_index]
    qw_safety = 1 - rank12.qW_within_user_percentile.to_numpy(float)
    value = -(
        rank_descending(score[chosen], customer)
        + rank_descending(event7, customer)
        + rank_descending(qw_safety, customer)
    ).astype(float)
    return chosen, value


def evaluate(data: dict, chosen: np.ndarray, value: np.ndarray, quota: float):
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    count = max(1, int(np.ceil(quota * len(chosen))))
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
        raise FileExistsError("Do not overwrite E23")
    (repo / RUN).mkdir(parents=True)
    register(repo)
    start = time.perf_counter()
    historical = {}
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        model, fit_info = fit(repo, cutoff, repo / RUN)
        x, mapping = final_matrix(repo, cutoff, data)
        score = model.booster_.predict(x, num_threads=4)
        f70 = np.load(repo / E16 / f"{cutoff}-F70-scores.npy")
        historical[cutoff] = {
            "fit": fit_info, "mapping": mapping,
            "F70": identify(data, f70), "R70": identify(data, score), "rules": {},
        }
        for mode in MODES:
            chosen, value = choose_and_value(data, score, x, mode)
            historical[cutoff]["rules"][mode] = {
                f"q={q:g}": evaluate(data, chosen, value, q)[0] for q in QUOTAS
            }
        print("E23 history", cutoff, historical[cutoff]["F70"], historical[cutoff]["R70"], flush=True)
    f70_top1 = sum(historical[t]["F70"]["topk_positive_pairs"][1] for t in VALID)
    r70_top1 = sum(historical[t]["R70"]["topk_positive_pairs"][1] for t in VALID)
    f70_top5 = sum(historical[t]["F70"]["topk_positive_pairs"][5] for t in VALID)
    r70_top5 = sum(historical[t]["R70"]["topk_positive_pairs"][5] for t in VALID)
    nondecrease = sum(
        historical[t]["R70"]["topk_positive_pairs"][1]
        >= historical[t]["F70"]["topk_positive_pairs"][1] for t in VALID
    )
    candidate_gate = {
        "top1_strictly_better": bool(r70_top1 > f70_top1),
        "two_window_nondecrease": bool(nondecrease >= 2),
        "top5_not_lower": bool(r70_top5 >= f70_top5),
        "F70_top1": int(f70_top1), "R70_top1": int(r70_top1),
        "F70_top5": int(f70_top5), "R70_top5": int(r70_top5),
        "top1_nondecrease_windows": int(nondecrease),
    }
    aggregate = {}
    for mode in MODES:
        for quota in QUOTAS:
            key = f"{mode}|q={quota:g}"
            rows = [historical[t]["rules"][mode][f"q={quota:g}"] for t in VALID]
            aggregate[key] = {
                "mode": mode, "quota": quota,
                "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                "inserted": int(sum(r["inserted"] for r in rows)),
                "removed": int(sum(r["removed"] for r in rows)),
                "admissions": int(sum(r["admissions"] for r in rows)),
            }
    eligible = [(key, value) for key, value in aggregate.items()
                if all(candidate_gate[k] for k in ("top1_strictly_better", "two_window_nondecrease", "top5_not_lower"))
                and value["mean_delta"] > 0 and value["nonnegative"] >= 2
                and value["inserted"] >= 2 and value["inserted"] >= value["removed"]]
    eligible.sort(key=lambda pair: (-pair[1]["mean_delta"], pair[1]["removed"], pair[1]["admissions"], MODES.index(pair[1]["mode"])))
    selected = eligible[0][0] if eligible else None
    development = {}
    gates = {"not_run": "historical candidate or MAP gate failed"}
    passed = False
    if selected:
        spec = aggregate[selected]
        for window, cutoff in WINDOWS.items():
            data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
            model, fit_info = fit(repo, cutoff, repo / RUN)
            x, mapping = final_matrix(repo, cutoff, data)
            score = model.booster_.predict(x, num_threads=4)
            f70_model = lgb.Booster(model_file=str(repo / E16 / f"{cutoff}-F70.txt"))
            f70 = f70_model.predict(x, num_threads=4)
            chosen, value = choose_and_value(data, score, x, spec["mode"])
            result, lists, events = evaluate(data, chosen, value, spec["quota"])
            result["segments"] = segment_metrics(data, lists)
            result["fit"] = fit_info
            result["candidate"] = {"F70": identify(data, f70), "R70": identify(data, score)}
            development[window] = result
            save_frame(events, repo / RUN / f"{window}-events.parquet")
            print("E23 dev", window, result, flush=True)
        delta = np.asarray([r["delta_map"] for r in development.values()])
        warm = float(np.mean([r["segments"]["warm_21_plus"] for r in development.values()]))
        cold = float(np.mean([r["segments"]["all_cold_sparse"] for r in development.values()]))
        inserted = int(sum(r["inserted"] for r in development.values()))
        removed = int(sum(r["removed"] for r in development.values()))
        gates = {
            "mean_positive": bool(delta.mean() > 0),
            "three_nondegrade": bool((delta >= -1e-5).sum() >= 3),
            "worst": bool(delta.min() >= -5e-5), "warm": bool(warm >= -2e-5),
            "cold_sparse": bool(cold >= 0), "inserted_positive": bool(inserted >= 1),
            "efficiency": bool(inserted >= removed),
        }
        passed = all(gates.values())
    output = {
        "status": "completed", "stage": "FINAL-E23 F70特征组内成对排序",
        "historical": historical, "candidate_gate": candidate_gate,
        "historical_aggregate": aggregate, "selected": selected,
        "development": development, "development_gates": gates,
        "fullscale_allowed": passed, "fullscale_status": "not_run",
        "fallback": "challenger" if passed else "WV3-741",
        "final_week": "not_run", "seconds": time.perf_counter() - start,
    }
    dump(repo / REPORT / "TEMPORAL_RANKER_E23.json", output)
    print(json.dumps({
        "candidate_gate": candidate_gate, "selected": selected,
        "selected_historical": aggregate.get(selected), "development": development,
        "gates": gates, "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"TEMPORAL_RANKER_E23_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
