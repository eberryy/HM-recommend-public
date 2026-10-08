"""FINAL-E19: F70 benefit/harm meta-admission for a fixed rank-12 action."""
from __future__ import annotations

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
from .final_candidate_e2 import DATES, VALID, WINDOWS
from .final_fixed_candidate_e12 import candidate_choice, rank_descending
from .final_full_cold_e8 import earlier_full
from .final_oracle_audit import dump
from .final_temporal_item_e16 import (
    FEATURES as F70_FEATURES,
    PREPARED,
    RUN as E16,
    fit as fit_f70,
    final_matrix,
)
from .p42f_contract import earlier
from .p42f_data import save_frame
from .p43a_policy import exact_ap


RUN = Path("artifacts/final/meta-admission-e19-v1")
REPORT = Path("reports/final")
SCORE_DATES = DATES
QUOTAS = (.001, .002, .005, .01, .02)
MODES = ("benefit", "expected", "safe_benefit", "rank_expected")

STATE_FEATURES = [
    "user_past_purchase_count", "novel_purchase_count_0_5",
    "novel_purchase_share_0_5", "low_pop_purchase_count_0_20",
    "low_pop_purchase_share_0_20", "novel_purchase_count_0_5_recent84d",
    "novel_purchase_share_0_5_recent84d", "days_since_last_0_5_purchase",
    "median_item_interaction_count_at_purchase",
    "median_item_popularity_percentile_at_purchase", "user_purchase_events_12w",
    "user_unique_items_12w", "user_unique_items_all", "active_purchase_days_12w",
    "active_purchase_days_all", "days_since_last_purchase",
    "historical_action_windows_available",
]
WARM_FEATURES = [
    "warm_model_score", "warm_model_score_available", "fused_score", "source_count",
    "repurchase_rrf_contribution", "recent_popularity_rrf_contribution",
    "product_family_rrf_contribution", "user_day_covisit_rrf_contribution",
    "age_popularity_rrf_contribution", "attribute_content_rrf_contribution",
    "repurchase_score", "item2vec_rank", "item2vec_cosine",
    "item2vec_seed_support", "item2vec_vocab_count", "item_events_7d",
    "item_events_28d", "item_events_12w", "item_unique_customers_28d",
    "item_days_since_last_sale", "item_trend_7d_vs_28d",
    "user_item_events_12w", "user_item_events_28d",
    "user_item_days_since_last_purchase", "warm_user_percentile",
    "warm_user_zscore", "qW_within_user_rank_pct", "qW_within_user_percentile",
    "interaction_count_before_cutoff",
]
SUMMARY_FEATURES = [
    "f70_score", "f70_margin_rank2", "f70_margin_rank5", "f70_user_mean",
    "f70_user_std", "f70_top1_zscore",
]
BENEFIT_FEATURES = F70_FEATURES + SUMMARY_FEATURES + [f"state_{x}" for x in STATE_FEATURES]
HARM_FEATURES = [f"warm_{x}" for x in WARM_FEATURES] + [f"state_{x}" for x in STATE_FEATURES]
PARAMS = {
    "objective": "binary", "learning_rate": .04, "n_estimators": 350,
    "num_leaves": 15, "max_depth": 5, "min_child_samples": 50,
    "subsample": 1., "colsample_bytree": 1., "reg_lambda": 2.,
    "reg_alpha": 0., "random_state": 20260914, "n_jobs": 4,
    "verbosity": -1, "deterministic": True, "force_col_wise": True,
}


class ConstantProbability:
    """Fallback for an earlier training slice containing only one class."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict_proba(self, x):
        p = np.full(len(x), self.value, dtype=float)
        return np.column_stack([1 - p, p])


def score_path(repo: Path, cutoff: str) -> Path:
    existing = repo / E16 / f"{cutoff}-F70-scores.npy"
    return existing if existing.exists() else repo / RUN / "scores" / f"{cutoff}-F70-scores.npy"


def model_path(repo: Path, cutoff: str) -> Path:
    existing = repo / E16 / f"{cutoff}-F70.txt"
    return existing if existing.exists() else repo / RUN / "scores" / f"{cutoff}-F70.txt"


def ensure_f70(repo: Path, cutoff: str, data: dict) -> tuple[np.ndarray, dict]:
    scores = score_path(repo, cutoff)
    model_file = model_path(repo, cutoff)
    scores.parent.mkdir(parents=True, exist_ok=True)
    fit_info = {"reused_model": False, "training": earlier_full(cutoff)}
    if model_file.exists():
        model = lgb.Booster(model_file=str(model_file))
        fit_info["reused_model"] = True
    else:
        if not earlier_full(cutoff):
            raise RuntimeError(f"No strictly earlier F70 training date for {cutoff}")
        model, trained = fit_f70(repo, cutoff, scores.parent)
        fit_info.update(trained)
    if scores.exists():
        value = np.load(scores)
        fit_info["reused_scores"] = True
    else:
        x, mapping = final_matrix(repo, cutoff, data)
        booster = model if isinstance(model, lgb.Booster) else model.booster_
        value = booster.predict(x, num_threads=4)
        np.save(scores, value)
        fit_info["reused_scores"] = False
        fit_info["mapping"] = mapping
        del x
    del model
    gc.collect()
    return value, fit_info


def action_frame(repo: Path, cutoff: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
    score, score_info = ensure_f70(repo, cutoff, data)
    cold_x, mapping = final_matrix(repo, cutoff, data)
    chosen, margin2 = candidate_choice(data, score)
    user_index = data["cold"].iloc[chosen].user_index.to_numpy(np.int64)
    rank12 = data["warm"].iloc[11::12].reset_index(drop=True).iloc[user_index]
    state = data["state"].set_index("customer_id").reindex(data["users"])
    state_x = state[STATE_FEATURES].to_numpy(np.float32)[user_index]

    summary = np.empty((len(chosen), len(SUMMARY_FEATURES)), dtype=np.float32)
    for out_row, rows in enumerate(data["cold"].groupby("user_index", sort=False).indices.values()):
        rows = np.asarray(rows, dtype=np.int64)
        values = np.sort(score[rows])[::-1]
        top = values[0]
        mean = values.mean()
        std = values.std()
        summary[out_row] = (
            top,
            margin2[out_row],
            top - values[min(4, len(values) - 1)],
            mean,
            std,
            (top - mean) / std if std > 1e-12 else 0.,
        )
    benefit_x = np.concatenate([cold_x[chosen], summary, state_x], axis=1).astype(np.float32)
    harm_x = np.concatenate([
        rank12[WARM_FEATURES].to_numpy(np.float32), state_x,
    ], axis=1).astype(np.float32)
    benefit_x[~np.isfinite(benefit_x)] = np.nan
    harm_x[~np.isfinite(harm_x)] = np.nan
    inserted = data["cold"].iloc[chosen].target.to_numpy(np.int8)
    removed = rank12.target.to_numpy(np.int8)
    meta = {
        "cutoff": cutoff, "users": int(len(data["users"])),
        "candidate_users": int(len(chosen)), "inserted_positive": int(inserted.sum()),
        "rank12_positive": int(removed.sum()), "chosen_rows": chosen,
        "mapping": mapping, "f70": score_info,
    }
    return benefit_x, harm_x, np.column_stack([inserted, removed]), meta, data


def fit_binary(x: np.ndarray, y: np.ndarray, feature_names: list[str], path: Path):
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        return ConstantProbability(float(y.mean())), {
            "rows": int(len(y)), "positives": positives, "constant": True,
        }
    params = dict(PARAMS)
    params["scale_pos_weight"] = float((len(y) - positives) / positives)
    model = lgb.LGBMClassifier(**params)
    with threadpool_limits(limits=4):
        model.fit(x, y, feature_name=feature_names)
    model.booster_.save_model(str(path))
    return model, {
        "rows": int(len(y)), "positives": positives, "constant": False,
        "scale_pos_weight": params["scale_pos_weight"],
    }


def train_meta(repo: Path, cutoff: str, folder: Path):
    benefit, harm, labels, dates = [], [], [], []
    source_audits = []
    for date in earlier(cutoff):
        bx, hx, y, meta, data = action_frame(repo, date)
        benefit.append(bx)
        harm.append(hx)
        labels.append(y)
        dates.extend([date] * len(y))
        source_audits.append({k: v for k, v in meta.items() if k not in ("chosen_rows",)})
        del bx, hx, y, data
        gc.collect()
    if not benefit:
        raise RuntimeError(f"No strictly earlier action date for {cutoff}")
    bx = np.concatenate(benefit)
    hx = np.concatenate(harm)
    y = np.concatenate(labels)
    benefit_model, benefit_info = fit_binary(
        bx, y[:, 0], BENEFIT_FEATURES, folder / f"{cutoff}-benefit.txt",
    )
    harm_model, harm_info = fit_binary(
        hx, y[:, 1], HARM_FEATURES, folder / f"{cutoff}-harm.txt",
    )
    info = {
        "training_dates": earlier(cutoff), "source_audits": source_audits,
        "benefit": benefit_info, "harm": harm_info,
        "strictly_earlier": True,
    }
    del bx, hx, y, benefit, harm, labels
    gc.collect()
    return benefit_model, harm_model, info


def probability(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def admission_value(mode: str, pb: np.ndarray, ph: np.ndarray, customer: np.ndarray):
    if mode == "benefit":
        return pb
    if mode == "expected":
        return pb - ph
    if mode == "safe_benefit":
        return pb * (1 - ph)
    if mode == "rank_expected":
        benefit_rank = rank_descending(pb, customer)
        safety_rank = rank_descending(-ph, customer)
        return 1 / (60 + benefit_rank) + 1 / (60 + safety_rank)
    raise KeyError(mode)


def evaluate(data: dict, chosen: np.ndarray, value: np.ndarray, quota: float):
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    count = max(1, int(math.ceil(quota * len(chosen))))
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


def score_cutoff(repo: Path, cutoff: str, folder: Path):
    benefit_model, harm_model, fit_info = train_meta(repo, cutoff, folder)
    bx, hx, labels, meta, data = action_frame(repo, cutoff)
    chosen = meta.pop("chosen_rows")
    pb = probability(benefit_model, bx)
    ph = probability(harm_model, hx)
    del benefit_model, harm_model, bx, hx
    gc.collect()
    return data, chosen, pb, ph, labels, {"fit": fit_info, "scoring": meta}


def register(repo: Path):
    contract = {
        "status": "preregistered_before_E19_labels_or_fit",
        "stage": "FINAL-E19 F70 双风险元准入",
        "observed_bottleneck": "E16 的 F70 在历史窗显著提高 Cold 候选 Top1/Top5，四开发窗只读复核也全部提高；但固定分位准入在四窗插入5个正例同时误删9个 Warm 正例。E17/E18 的手工 Warm 风险门降低误删但也过滤正例。",
        "hypothesis": "固定 F70 用户内第一名与第12位替换后，分别用严格更早日期的完整候选用户学习冷商品未来购买概率和第12位未来命中概率，可比固定分位或手工中位数门更好地跨用户排序净收益。",
        "unit": "一行是一名具有 Cold50 候选的用户；候选固定为 F70 用户内第一名，被替换商品固定为 WV3-741 第12位。",
        "targets": {
            "benefit": "该 F70 第一名是否在随后一周被购买；分母为该日期全部有 Cold50 候选的用户。",
            "harm": "原 WV3-741 第12位是否在随后一周被购买；分母同上。",
        },
        "features": {
            "benefit": "F70 的70维用户—候选关系/多视角/商品时效特征，加6维用户内分数形态和17维截止日前用户状态。",
            "harm": "29维第12位 Warm 排序/召回/时效特征，加17维截止日前用户状态。",
            "temporal_boundary": "所有特征只使用对应日期截止日前数据；训练标签日期严格早于评分日期至少一周。",
        },
        "models": "两个固定参数 LightGBM 二分类器；按训练总体正负比设置类别权重，只用于当窗跨用户排序。",
        "admission_modes": {
            "benefit": "只按冷商品购买概率。",
            "expected": "冷商品购买概率减去第12位命中概率。",
            "safe_benefit": "冷商品购买概率乘以第12位不命中概率。",
            "rank_expected": "冷收益名次与第12位安全名次等权倒数名次融合。",
        },
        "quotas": "按该日期候选用户数执行头部0.1%、0.2%、0.5%、1%、2%。",
        "historical_gate": "三个历史前向窗 mean delta_MAP>0、至少两窗不退化、插入正例不少于2且插入数不少于误删数；按 mean、误删、动作数、固定模式顺序唯一选择。",
        "development_gate": "历史通过后冻结唯一模式与比例；四开发窗沿用 FINAL-WRAPUP 扩量门槛。",
        "if_fail": "保留 WV3-741；F70 只保留为候选层 challenger，不再搜索当前开发窗门槛。",
        "cost_limit": "本机 CPU，预计30分钟内；只补两个缺失的日期级 F70 模型，其余复用。",
        "historical_validation": VALID, "development_windows": WINDOWS,
        "final_week": "not_run", "no_commit_push": True,
        "free_disk_gib": shutil.disk_usage(repo).free / 2**30,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(repo / REPORT / "META_ADMISSION_E19_CONTRACT.json", contract)
    return contract


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E19")
    (repo / RUN / "models").mkdir(parents=True)
    contract = register(repo)
    start = time.perf_counter()
    historical, aggregate = {}, {}
    for cutoff in VALID:
        data, chosen, pb, ph, labels, audit = score_cutoff(repo, cutoff, repo / RUN / "models")
        customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
        historical[cutoff] = {"audit": audit, "rules": {}}
        for mode in MODES:
            value = admission_value(mode, pb, ph, customer)
            historical[cutoff]["rules"][mode] = {}
            for quota in QUOTAS:
                result, _, _ = evaluate(data, chosen, value, quota)
                historical[cutoff]["rules"][mode][f"q={quota:g}"] = result
        print("E19 history", cutoff, audit["fit"], flush=True)
        del data, chosen, pb, ph, labels
        gc.collect()
    for mode in MODES:
        for quota in QUOTAS:
            key = f"{mode}|q={quota:g}"
            rows = [historical[t]["rules"][mode][f"q={quota:g}"] for t in VALID]
            aggregate[key] = {
                "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                "admissions": int(sum(r["admissions"] for r in rows)),
                "inserted": int(sum(r["inserted"] for r in rows)),
                "removed": int(sum(r["removed"] for r in rows)),
                "beneficial": int(sum(r["beneficial"] for r in rows)),
                "harmful": int(sum(r["harmful"] for r in rows)),
                "mode": mode, "quota": quota,
            }
    eligible = [(key, value) for key, value in aggregate.items()
                if value["mean_delta"] > 0 and value["nonnegative"] >= 2 and
                value["inserted"] >= 2 and value["inserted"] >= value["removed"]]
    eligible.sort(key=lambda pair: (
        -pair[1]["mean_delta"], pair[1]["removed"], pair[1]["admissions"],
        MODES.index(pair[1]["mode"]),
    ))
    selected = eligible[0][0] if eligible else None
    development = {}
    gates = {"not_run": "historical gate failed"}
    passed = False
    if selected:
        spec = aggregate[selected]
        for window, cutoff in WINDOWS.items():
            data, chosen, pb, ph, labels, audit = score_cutoff(repo, cutoff, repo / RUN / "models")
            customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
            value = admission_value(spec["mode"], pb, ph, customer)
            result, lists, events = evaluate(data, chosen, value, spec["quota"])
            result["segments"] = segment_metrics(data, lists)
            result["audit"] = audit
            development[window] = result
            save_frame(events, repo / RUN / f"{window}-events.parquet")
            del data, chosen, pb, ph, labels, lists, events
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
            "warm": bool(warm >= -2e-5), "cold_sparse": bool(cold >= 0),
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
    dump(repo / REPORT / "META_ADMISSION_E19.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"META_ADMISSION_E19_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
