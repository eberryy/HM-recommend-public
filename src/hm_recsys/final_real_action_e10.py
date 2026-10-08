"""FINAL-E10: real-only cross-time candidate-victim action utility."""
from pathlib import Path
import gc
import json
import shutil
import subprocess
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np
from threadpoolctl import threadpool_limits

from .final_b0_admission_e4 import segment_metrics
from .final_candidate_e2 import VALID, WINDOWS
from .final_oracle_audit import dump
from .final_pseudo_action_e6 import (
    DATES, RUN as E6, FEATURES, PREPARED, evaluate, score_real,
)
from .p42f_contract import earlier
from .p42f_data import save_frame


RUN = Path("artifacts/final/real-action-e10-v1")
REPORT = Path("reports/final")
ARMS = ("real_regression", "real_risk_separated")
MODES = ("b0_fixed", "model_full")
QUOTAS = (.001, .005, .01, .02)
BASE_PARAMS = {
    "learning_rate": .05, "n_estimators": 300, "num_leaves": 15,
    "max_depth": 6, "min_child_samples": 100, "subsample": 1.,
    "colsample_bytree": 1., "reg_lambda": 1., "reg_alpha": 0.,
    "random_state": 20260914, "n_jobs": 4, "verbosity": -1,
    "deterministic": True, "force_col_wise": True,
}


class RiskModel:
    def __init__(self, benefit, harm, positive_mean, negative_mean):
        self.benefit = benefit
        self.harm = harm
        self.positive_mean = positive_mean
        self.negative_mean = negative_mean

    def predict(self, x, num_threads=4):
        p_benefit = self.benefit.predict(x, num_threads=num_threads)
        p_harm = self.harm.predict(x, num_threads=num_threads)
        return p_benefit * self.positive_mean + p_harm * self.negative_mean


def training_arrays(repo: Path, cutoff: str):
    xs, ys, weights = [], [], []
    for date in earlier(cutoff):
        folder = repo / E6 / "prepared" / date
        xs.append(np.load(folder / "real-X.npy", mmap_mode="r"))
        ys.append(np.load(folder / "real-y.npy", mmap_mode="r"))
        # E6 multiplied every real row by five.  A constant domain multiplier
        # does not affect a real-only fit, so divide it out for clearer audit.
        weights.append(np.load(folder / "real-w.npy", mmap_mode="r") / 5.)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)


def fit(repo: Path, cutoff: str, arm: str, folder: Path):
    x, y, weight = training_arrays(repo, cutoff)
    if arm == "real_regression":
        model = lgb.LGBMRegressor(objective="regression_l2", **BASE_PARAMS)
        with threadpool_limits(limits=4):
            model.fit(x, y, sample_weight=weight, feature_name=FEATURES)
        model.booster_.save_model(str(folder / f"{cutoff}-{arm}.txt"))
    elif arm == "real_risk_separated":
        benefit = lgb.LGBMClassifier(objective="binary", **BASE_PARAMS)
        harm = lgb.LGBMClassifier(objective="binary", **BASE_PARAMS)
        with threadpool_limits(limits=4):
            benefit.fit(x, (y > 0).astype(np.int8), sample_weight=weight, feature_name=FEATURES)
            harm.fit(x, (y < 0).astype(np.int8), sample_weight=weight, feature_name=FEATURES)
        positive_mean = float(y[y > 0].mean())
        negative_mean = float(y[y < 0].mean())
        benefit.booster_.save_model(str(folder / f"{cutoff}-{arm}-benefit.txt"))
        harm.booster_.save_model(str(folder / f"{cutoff}-{arm}-harm.txt"))
        model = RiskModel(benefit.booster_, harm.booster_, positive_mean, negative_mean)
    else:
        raise KeyError(arm)
    info = {
        "rows": int(len(y)), "beneficial": int((y > 0).sum()),
        "harmful": int((y < 0).sum()), "neutral_retained": int((y == 0).sum()),
        "weighted_rows": float(weight.sum()), "train_dates": earlier(cutoff),
        "positive_mean_delta": float(y[y > 0].mean()),
        "negative_mean_delta": float(y[y < 0].mean()),
    }
    del x, y, weight
    gc.collect()
    return model, info


def register(repo: Path):
    contract = {
        "status": "preregistered_before_E10_fit",
        "stage": "FINAL-E10 真实 Cold 跨时间联合决策",
        "observed_bottleneck": "E8/E9 的 Cold 候选内部 Top5 有局部增益，但候选置信度与旧 action 模型无法共同找到安全替换。",
        "hypothesis": "E6 的伪冷样本仍占主要正例监督并可能产生域偏移；只用严格早于当前日期的真实 Cold 候选—Warm 位置标签，联合学习候选、替换位置与准入效用。",
        "population": "每个历史 10% 队列中的真实 Cold50 候选乘以原 WV3 Warm Top12 位置；不使用未来条件筛选用户。",
        "target": "单次原位替换后的精确 AP@12 变化；正值表示净获益，负值表示误删损失，零表示无变化。",
        "features": {
            "definition": "每条候选—位置动作含 24 维用户与 Cold 候选关系、Warm 候选特征、17 维用户状态、Cold 来源名次百分位和 Warm 位置百分位。",
            "unit": "一行对应一个用户、一个 Cold 候选、一个 Warm Top12 被替换位置。",
        },
        "arms": {
            "real_regression": "直接回归精确 AP@12 变化。",
            "real_risk_separated": "分别估计获益概率与损失概率，再乘训练期平均正负幅度得到期望变化。",
        },
        "training": "严格使用当前评分日期至少提前一周结束的历史日期；全部非零动作加稳定 2% 零动作，零动作以 50 倍权重还原总体。",
        "candidate_modes": {
            "b0_fixed": "Cold 商品固定为 B0 在该用户可用候选中的第一名，模型只决定替换位置和是否准入。",
            "model_full": "模型在 Cold50 × Warm12 的完整动作空间同时选择商品与替换位置。",
        },
        "quotas": "按预测效用从高到低，只执行候选用户数的 0.1%、0.5%、1%、2%。",
        "historical_gate": "三窗 mean delta_MAP>0、至少两窗不退化、插入正例不少于2且插入数不少于误删数。",
        "development_gate": "若历史门槛通过，冻结唯一规则后运行既有四窗扩量门槛。",
        "if_fail": "保留 WV3-741；不扩量。",
        "cost_limit": "复用 E6 已物化真实动作特征，本机 CPU 预计 5 分钟内。",
        "parameters": BASE_PARAMS,
        "historical_validation": VALID,
        "development_windows": WINDOWS,
        "final_week": "not_run", "no_commit_push": True,
        "free_disk_gib": shutil.disk_usage(repo).free / 2**30,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    dump(repo / REPORT / "REAL_ACTION_E10_CONTRACT.json", contract)
    return contract


def run(repo: Path):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E10")
    (repo / RUN).mkdir(parents=True)
    contract = register(repo)
    start = time.perf_counter()
    historical, aggregate = {}, {}
    for cutoff in VALID:
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        historical[cutoff] = {}
        for arm in ARMS:
            model, fit_info = fit(repo, cutoff, arm, repo / RUN)
            scores = score_real(repo, cutoff, data, model)
            np.save(repo / RUN / f"{cutoff}-{arm}-scores.npy", scores)
            historical[cutoff][arm] = {"fit": fit_info, "rules": {}}
            for mode in MODES:
                for quota in QUOTAS:
                    key = f"{mode}|q={quota:g}"
                    result, _, _ = evaluate(data, scores, mode, quota)
                    historical[cutoff][arm]["rules"][key] = result
            print("E10", cutoff, arm, fit_info, flush=True)
            del model, scores
            gc.collect()
        del data
        gc.collect()
    for arm in ARMS:
        for mode in MODES:
            for quota in QUOTAS:
                key = f"{arm}|{mode}|q={quota:g}"
                rows = [historical[t][arm]["rules"][f"{mode}|q={quota:g}"] for t in VALID]
                aggregate[key] = {
                    "mean_delta": float(np.mean([r["delta_map"] for r in rows])),
                    "nonnegative": int(sum(r["delta_map"] >= 0 for r in rows)),
                    "admissions": int(sum(r["admissions"] for r in rows)),
                    "inserted": int(sum(r["inserted"] for r in rows)),
                    "removed": int(sum(r["removed"] for r in rows)),
                    "beneficial": int(sum(r["beneficial"] for r in rows)),
                    "harmful": int(sum(r["harmful"] for r in rows)),
                    "arm": arm, "mode": mode, "quota": quota,
                }
    eligible = [(key, value) for key, value in aggregate.items()
                if value["mean_delta"] > 0 and value["nonnegative"] >= 2 and
                value["inserted"] >= 2 and value["inserted"] >= value["removed"]]
    eligible.sort(key=lambda pair: (-pair[1]["mean_delta"], pair[1]["admissions"],
                                    pair[1]["mode"] != "b0_fixed",
                                    pair[1]["arm"] != "real_regression"))
    selected = eligible[0][0] if eligible else None
    development = {}
    gates = {"not_run": "historical gate failed"}
    passed = False
    if selected:
        spec = aggregate[selected]
        for window, cutoff in WINDOWS.items():
            data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
            model, fit_info = fit(repo, cutoff, spec["arm"], repo / RUN)
            scores = score_real(repo, cutoff, data, model)
            result, lists, events = evaluate(data, scores, spec["mode"], spec["quota"])
            result["segments"] = segment_metrics(data, lists)
            result["fit"] = fit_info
            development[window] = result
            save_frame(events, repo / RUN / f"{window}-events.parquet")
            del data, model, scores, lists, events
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
    dump(repo / REPORT / "REAL_ACTION_E10.json", output)
    print(json.dumps({
        "selected": selected, "selected_historical": aggregate.get(selected),
        "development": development, "gates": gates,
        "fullscale_allowed": passed, "seconds": output["seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"REAL_ACTION_E10_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(), "final_week": "not_run",
        })
        raise
