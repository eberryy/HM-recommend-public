"""WV3-641: Top100 candidate propensity with protected one-swap admission."""
from __future__ import annotations

import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import evidence_id, now, read, write
from .warm_v2_engine import save_parquet
from .warm_v3_expert import screening_gate
from .warm_v3_pointwise_propensity import FEATURES, PARAMS, ROUNDS, candidate_frame, choose_propensity_decisions, fit_model
from .warm_v3_residual_admission import HISTORICAL, INNER, OUTER, pair_frame, source


TRIAL = "WV3-641"
CONTRACT = common.REPORT / "WV3-641_TOP100_PROPENSITY_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-641_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-641_SCREEN.md"
OUTER_REPORT = common.REPORT / "WV3-641_OUTER.json"
MODEL_ROOT = common.ART / TRIAL


def historical_training_data() -> tuple[pd.DataFrame, dict]:
    frames = []
    sources = []
    for name, folder in HISTORICAL:
        _, meta = source(folder)
        frame, details = candidate_frame(meta["cutoff"], with_labels=True, rank_min=1, rank_max=100)
        frame["source_window"] = name
        frames.append(frame)
        sources.append({**details, "window": name})
    training = pd.concat(frames, ignore_index=True)
    audit = {
        "role": "strictly_historical_top100_candidate_purchase_supervision",
        "sources": sources,
        "rows": len(training),
        "positive_rows": int(training.target.sum()),
        "positive_rate": float(training.target.mean()),
        "unique_user_item_window_rows": int(training.groupby(["source_window", "customer_id", "article_id"]).ngroups),
        "features": FEATURES,
        "training_rank_range": [1, 100],
        "inference_victim_rank_range": [8, 12],
        "inference_challenger_rank_range": [13, 100],
        "class_weighting": "none",
        "latest_label_end_exclusive": "2019-11-27",
        "earliest_inner_cutoff": "2019-12-25",
        "temporal_gap_days": 28,
        "final_week": "not_run",
    }
    audit["identity_unique"] = audit["unique_user_item_window_rows"] == audit["rows"]
    audit["passed"] = bool(audit["identity_unique"] and audit["positive_rows"] >= 5000 and audit["positive_rate"] < 0.10)
    return training, audit


def evaluate(folder: str, name: str, model: lgb.Booster, stage: str):
    path, meta = source(folder)
    started = time.perf_counter()
    candidates, details = candidate_frame(meta["cutoff"], with_labels=True, rank_min=8, rank_max=100)
    probability = model.predict(candidates[FEATURES].to_numpy(np.float32), num_threads=4)
    probability_frame = candidates[["customer_id", "article_id"]].copy()
    probability_frame["purchase_probability"] = probability
    pairs = pair_frame(path, discordant_only=False, challenger_high=100)
    chosen = choose_propensity_decisions(pairs, probability_frame)
    beneficial = chosen.actual_delta > 1e-15
    harmful = chosen.actual_delta < -1e-15
    neutral = ~(beneficial | harmful)
    delta = float(chosen.actual_delta.sum() / meta["total_users"])
    baseline = float(meta["baseline_map_population_component"])
    result = {
        "window": name,
        "cutoff": meta["cutoff"],
        "role": stage,
        "total_users_denominator": meta["total_users"],
        "candidate_rows": len(candidates),
        "pair_rows": len(pairs),
        "selected_users": len(chosen),
        "beneficial_selected_users": int(beneficial.sum()),
        "harmful_selected_users": int(harmful.sum()),
        "neutral_selected_users": int(neutral.sum()),
        "selected_neutral_share": float(neutral.mean()) if len(chosen) else 0.0,
        "gross_positive_MAP": float(chosen.loc[beneficial, "actual_delta"].sum() / meta["total_users"]),
        "gross_negative_MAP": float(chosen.loc[harmful, "actual_delta"].sum() / meta["total_users"]),
        "baseline_MAP@12": baseline,
        "MAP@12": baseline + delta,
        "population_delta": delta,
        "probability_quantiles": np.quantile(probability, [0, 0.1, 0.5, 0.9, 1]).tolist(),
        "protected_ranks1_7_changes": 0,
        "maximum_swaps_per_user": 1,
        "candidate_input": details,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    return result, chosen


def experiment_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-631",
        "parent_diagnostic": "WV3-640",
        "architecture_family": "top100_candidate_purchase_propensity_one_swap",
        "hypothesis": "The stable candidate-level propensity architecture can convert the material, cross-window marginal truth headroom at ranks51-100 without destabilizing the protected Top12 head.",
        "features": FEATURES,
        "training": "unweighted binary LightGBM on active rank1-100 candidates from four 2019 sources",
        "model": {"params": PARAMS, "rounds": ROUNDS},
        "decision_score": "(P(challenger purchased)-P(victim purchased))*exact position unit_gain",
        "action": "one maximum positive expected-gain swap; ranks1-7 immutable",
        "candidate_pool": "same frozen full pool; model support ranks1-100, inference challengers13-100 and victims8-12",
        "input_gate": "unique user-item-window rows, at least 5000 positives, positive rate below 10 percent",
        "inner_gate": {
            "standard_vs_WV2_601": "mean>=+0.0001, >=3 positive, worst>=-0.0005",
            "incremental_vs_WV3_631": "mean>0, >=3 nondegrade, worst>=-0.0002",
        },
        "expected_minutes": 20,
        "outer_policy": "one frozen exposure only if input and both inner gates pass; no support-size, class-weight, threshold, feature or tree rescue",
        "fallback": "retain WV3-631 Top50 stable candidate",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(20)
    assert read(common.REPORT / "WV3-640_TOP100_ORACLE_AUDIT.json")["top100_propensity_authorized"]
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "top100_candidate_purchase_propensity_one_swap",
            experiment_contract()["hypothesis"],
            candidate_protocol=experiment_contract()["candidate_pool"],
            training_protocol=experiment_contract()["training"],
            features=FEATURES,
            params={"model": experiment_contract()["model"], "decision_score": experiment_contract()["decision_score"], "inner_gate": experiment_contract()["inner_gate"], "final_week": "not_run"},
            expected_minutes=20,
        )


def incremental_gate(windows: dict) -> dict:
    prior = read(common.REPORT / "WV3-631_SCREEN.json")["windows"]
    delta = {name: row["MAP@12"] - prior[name]["MAP@12"] for name, row in windows.items()}
    values = list(delta.values())
    return {
        "per_window_delta_vs_WV3_631": delta,
        "mean_delta_vs_WV3_631": float(np.mean(values)),
        "nondegrade_windows_vs_WV3_631": sum(value >= 0 for value in values),
        "worst_delta_vs_WV3_631": min(values),
        "passed": bool(np.mean(values) > 0 and sum(value >= 0 for value in values) >= 3 and min(values) >= -0.0002),
    }


def render_screen(result: dict) -> str:
    standard = result["standard_screening_vs_WV2_601"]
    incremental = result["incremental_screening_vs_WV3_631"]
    lines = [
        "# WV3-641 Top100 候选商品级购买倾向准入",
        "",
        "## 结论",
        "",
        f"内层总门槛：**{'pass' if result['passed'] else 'fail'}**。相对 WV2-601 平均增加 {standard['mean_population_delta']:+.9f}；相对 WV3-631 平均增加 {incremental['mean_delta_vs_WV3_631']:+.9f}。",
        "",
        "## 术语与实验对象",
        "",
        "- Top100 候选支持（本项目自定义）：模型训练读取基线第1–100名，推理只允许第13–100名挑战第8–12名；第1–7名不发生变化。",
        "- 候选商品级购买倾向（行业常见 pointwise propensity 思路）：一个用户—商品—窗口一行，预测下一周是否购买；模型不做类别重加权。",
        "- 边际概率差换位（本项目自定义）：挑战商品概率减被替换商品概率，再乘对应位置 AP@12 权重；每位用户最多执行一次正期望换位。",
        "- 中性替换（本项目自定义）：两件商品在监督周同为买或同为未买，直接交换的实际 AP@12 变化为0；中性占比分母是实际执行替换的用户数。",
        "- population delta（本项目沿用指标）：以窗口全部真值用户为分母，实验 MAP@12 减 WV2-601 MAP@12。",
        "",
        f"训练行数 {result['training_audit']['rows']:,}，正例 {result['training_audit']['positive_rows']:,}，正例率 {result['training_audit']['positive_rate']:.3%}。候选预算扩展由 WV3-640 四窗边际 Oracle 审计预先授权。",
        "",
        "| 内层窗口 | 选择用户数 | 有益 / 有害 / 中性 | 中性占比 | 相对 WV2-601 | 相对 WV3-631 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(f"| {name} | {row['selected_users']:,} | {row['beneficial_selected_users']:,} / {row['harmful_selected_users']:,} / {row['neutral_selected_users']:,} | {row['selected_neutral_share']:.2%} | {row['population_delta']:+.9f} | {incremental['per_window_delta_vs_WV3_631'][name]:+.9f} |")
    lines += [
        "",
        "只有相对 WV2-601 和 WV3-631 的两个内层门槛同时通过，才允许冻结模型并读取一次外层。失败后不调候选末位、类别权重、阈值、特征或树参数。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(20)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    started = time.perf_counter()
    training, audit = historical_training_data()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    write(MODEL_ROOT / "TRAINING_AUDIT.json", audit)
    if not audit["passed"]:
        common.update(TRIAL, decision="reject_input_audit", inner_evidence=audit, runtime=time.perf_counter() - started, artifact_paths=[str(MODEL_ROOT / "TRAINING_AUDIT.json")])
        raise RuntimeError("WV3-641 Top100 input gate failed")
    model = fit_model(training)
    del training
    gc.collect()
    model_path = MODEL_ROOT / "MODEL.txt"
    model.save_model(str(model_path))
    write(MODEL_ROOT / "MODEL.json", {"experiment_id": TRIAL, "model": evidence_id(model_path, reason="explicit_registry_evidence"), "features": FEATURES, "params": PARAMS, "rounds": ROUNDS, "training_audit": str(MODEL_ROOT / "TRAINING_AUDIT.json"), "outer_labels_used": False, "final_week": "not_run"})
    windows = {}
    for name, folder in INNER:
        row, chosen = evaluate(folder, name, model, "top100_historical_propensity_inner_screen")
        windows[name] = row
        root = MODEL_ROOT / name / "inner"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"top100_propensity_inner": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = screening_gate(row["population_delta"] for row in windows.values())
    incremental = incremental_gate(windows)
    passed = standard["passed"] and incremental["passed"]
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "top100_candidate_purchase_propensity_one_swap", "training_audit": audit, "windows": windows, "standard_screening_vs_WV2_601": standard, "incremental_screening_vs_WV3_631": incremental, "passed": passed, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render_screen(result), encoding="utf-8")
    common.update(TRIAL, decision="inner_pass" if passed else "reject_inner", inner_evidence={"input": audit, "standard": standard, "incremental": incremental, "passed": passed}, runtime=result["runtime_seconds"], artifact_paths=[str(SCREEN_REPORT), str(SCREEN_MARKDOWN), str(MODEL_ROOT / "MODEL.json")])
    common.log(
        f"{TRIAL} inner screen",
        "WV3-640 found ranks51-100 add 0.002267 mean one-swap oracle headroom and 272-360 newly beneficial users per inner window.",
        experiment_contract()["hypothesis"],
        f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",
        f"input pass={audit['passed']}; positives={audit['positive_rows']}; standard mean={standard['mean_population_delta']:+.9f}; incremental mean={incremental['mean_delta_vs_WV3_631']:+.9f}; pass={passed}.",
        "Freeze and expose once." if passed else "Reject Top100 propensity; retain WV3-631 without support-size tuning.",
        "Run one outer confirmation if both gates pass; otherwise diagnose a different bounded action family.",
        alternatives="Top100 was selected by a fixed marginal-oracle cost gate, not by trying several candidate budgets.",
        experiment="Fit the same pointwise architecture on historical Top100 and retain one protected tail swap.",
        reflection="The experiment isolates candidate support while holding model class and decision form fixed.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "passed": passed, "seconds": result["runtime_seconds"]}, flush=True)
    return result


def confirm() -> dict:
    common.setup()
    common.budget(10)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "inner_pass" and entry["outer_exposures"] == 1
    assert read(SCREEN_REPORT)["passed"]
    if OUTER_REPORT.exists():
        return read(OUTER_REPORT)
    model = lgb.Booster(model_file=str(MODEL_ROOT / "MODEL.txt"))
    started = time.perf_counter()
    windows = {}
    for name, folder in OUTER:
        row, chosen = evaluate(folder, name, model, "frozen_top100_propensity_outer")
        windows[name] = row
        root = MODEL_ROOT / name / "outer"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"top100_propensity_outer": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    prior = read(common.REPORT / "WV3-631_OUTER.json")
    delta = {name: windows[name]["MAP@12"] - prior["per_window_MAP"][name] for name in windows}
    values = list(delta.values())
    incremental = {"per_window_delta_vs_WV3_631": delta, "mean_delta_vs_WV3_631": float(np.mean(values)), "nondegrade_windows_vs_WV3_631": sum(value >= 0 for value in values), "worst_delta_vs_WV3_631": min(values)}
    better = standard["stable"] and incremental["mean_delta_vs_WV3_631"] > 0
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "top100_candidate_purchase_propensity_one_swap", "windows": windows, **standard, "incremental_vs_WV3_631": incremental, "better_than_current_champion": better, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(OUTER_REPORT, result)
    common.update(TRIAL, decision="promote_candidate" if better else "reject_outer", outer_MAP_by_window=standard["per_window_MAP"], mean_MAP=standard["mean_MAP"], delta_vs_WV2_601=standard["delta_vs_WV2_601"], nondegrade_windows=standard["nondegrade_windows"], worst_delta=standard["worst_delta"], runtime=result["runtime_seconds"], artifact_paths=entry["artifact_paths"] + [str(OUTER_REPORT)])
    common.log(
        f"{TRIAL} outer closure",
        "The Top100 propensity model passed both inner gates before being frozen.",
        experiment_contract()["hypothesis"],
        str(OUTER_REPORT),
        f"mean MAP={standard['mean_MAP']:.9f}; delta vs WV2-601={standard['delta_vs_WV2_601']:+.9f}; incremental mean vs WV3-631={incremental['mean_delta_vs_WV3_631']:+.9f}; stable={standard['stable']}; better={better}.",
        "Promote as current stable candidate." if better else "Reject and retain WV3-631.",
        "Continue through another bounded, evidence-backed mechanism only if target0.03 remains unmet and time permits.",
        alternatives="No outer-driven candidate budget, class weight, threshold, feature or tree rescue is allowed.",
        experiment="One outer exposure of the exact frozen Top100 historical propensity model.",
        reflection="This verifies whether marginal candidate headroom converts out of sample.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "better": better}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "screen", "confirm"])
    globals()[parser.parse_args().command]()
