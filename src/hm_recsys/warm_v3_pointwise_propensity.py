"""WV3-630: candidate-level purchase propensity for one-swap Warm admission."""
from __future__ import annotations

import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import evidence_id, now, read, write
from .warm_v2_engine import load_parquet, save_parquet
from .warm_v3_expert import screening_gate
from .warm_v3_residual_admission import HISTORICAL, INNER, OUTER, pair_frame, source


TRIAL = "WV3-630"
CONTRACT = common.REPORT / "WV3-630_POINTWISE_PROPENSITY_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-630_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-630_SCREEN.md"
OUTER_REPORT = common.REPORT / "WV3-630_OUTER.json"
MODEL_ROOT = common.ART / TRIAL

CACHED_FEATURES = [
    "r0_reciprocal",
    "r1_reciprocal",
    "r0_fraction",
    "r1_fraction",
    "rank_gap",
    "r0_top12",
    "r1_top12",
    "r0_top50",
    "r1_top50",
    "latent_within_user_z",
    "bpr_missing",
    "log_history",
    "log_unique",
    "log_recency",
    "repurchase_present",
    "item2vec_is_new",
    "user_item_events_12w",
    "user_item_days_since_last_purchase",
    "user_product_type_share_12w",
    "user_department_share_12w",
    "item_days_since_last_sale",
]
RANK_FEATURES = ["rf_reciprocal", "rf_fraction", "rf_victim_band"]
FEATURES = CACHED_FEATURES + RANK_FEATURES
PARAMS = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 500,
    "lambda_l2": 10.0,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "seed": 20260910,
    "feature_fraction_seed": 20260910,
    "bagging_seed": 20260910,
    "verbosity": -1,
}
ROUNDS = 100


def candidate_frame(
    cutoff: str,
    with_labels: bool = True,
    rank_min: int = 8,
    rank_max: int = 50,
) -> tuple[pd.DataFrame, dict]:
    root = common.ART / "candidate_gate_data" / cutoff
    meta = read(root / "DATA.json")
    assert meta["cutoff"] == cutoff
    assert meta["features"] == CACHED_FEATURES
    assert not meta["future_labels_are_inputs"] and meta["final_week"] == "not_run"
    keys = load_parquet(meta["keys"])
    arrays = np.load(root / "arrays.npz")
    assert len(keys) == meta["rows"] == len(arrays["x"])
    assert np.array_equal(keys.target.to_numpy(np.uint8), arrays["target"])
    active = (keys.user_history_events_12w > 0).to_numpy()
    assert rank_min in (1, 8)
    assert rank_max in (50, 100)
    mask = active & keys.rf.between(rank_min, rank_max).to_numpy()
    columns = ["customer_id", "article_id", "rf"] + (["target", "truth_count"] if with_labels else [])
    frame = keys.loc[mask, columns].reset_index(drop=True)
    x = arrays["x"][mask].astype(np.float32, copy=False)
    for index, name in enumerate(CACHED_FEATURES):
        frame[name] = x[:, index]
    rf = frame.rf.to_numpy(np.float32)
    frame["rf_reciprocal"] = 1.0 / (60.0 + rf)
    frame["rf_fraction"] = rf / 50.0
    frame["rf_victim_band"] = ((rf >= 8) & (rf <= 12)).astype(np.float32)
    assert not frame.duplicated(["customer_id", "article_id"]).any()
    assert np.isfinite(frame[FEATURES].to_numpy(np.float64)).all()
    details = {
        "cutoff": cutoff,
        "source": str(root / "DATA.json"),
        "rows": len(frame),
        "positive_rows": int(frame.target.sum()) if with_labels else None,
        "features": FEATURES,
        "source_identity": meta["source_identity"],
        "active_only": True,
        "baseline_rank_range": [rank_min, rank_max],
        "future_labels_are_inputs": False,
        "final_week": "not_run",
    }
    return frame, details


def historical_training_data() -> tuple[pd.DataFrame, dict]:
    frames = []
    sources = []
    for name, folder in HISTORICAL:
        _, meta = source(folder)
        frame, details = candidate_frame(meta["cutoff"])
        frame["source_window"] = name
        frames.append(frame)
        sources.append({**details, "window": name})
    training = pd.concat(frames, ignore_index=True)
    audit = {
        "role": "strictly_historical_candidate_level_purchase_supervision",
        "sources": sources,
        "rows": len(training),
        "positive_rows": int(training.target.sum()),
        "positive_rate": float(training.target.mean()),
        "unique_user_item_window_rows": int(training.groupby(["source_window", "customer_id", "article_id"]).ngroups),
        "features": FEATURES,
        "class_weighting": "none; preserve marginal purchase-probability ordering",
        "latest_label_end_exclusive": "2019-11-27",
        "earliest_inner_cutoff": "2019-12-25",
        "temporal_gap_days": 28,
        "final_week": "not_run",
    }
    audit["identity_unique"] = audit["unique_user_item_window_rows"] == audit["rows"]
    audit["passed"] = bool(audit["identity_unique"] and audit["positive_rows"] >= 5000 and audit["positive_rate"] < 0.10)
    return training, audit


def fit_model(training: pd.DataFrame) -> lgb.Booster:
    labels = training.target.to_numpy(np.uint8)
    assert 0 < labels.sum() < len(labels)
    dataset = lgb.Dataset(
        training[FEATURES].to_numpy(np.float32),
        label=labels,
        feature_name=FEATURES,
        free_raw_data=True,
    )
    return lgb.train(PARAMS, dataset, num_boost_round=ROUNDS)


def choose_propensity_decisions(pairs: pd.DataFrame, probabilities: pd.DataFrame) -> pd.DataFrame:
    challenger = probabilities.rename(
        columns={"article_id": "challenger_article_id", "purchase_probability": "challenger_probability"}
    )
    victim = probabilities.rename(
        columns={"article_id": "victim_article_id", "purchase_probability": "victim_probability"}
    )
    columns = [
        "customer_id",
        "challenger_article_id",
        "victim_article_id",
        "challenger_target",
        "victim_target",
        "challenger_rank",
        "victim_rank",
        "unit_gain",
    ]
    chosen = pairs[columns].merge(
        challenger,
        on=["customer_id", "challenger_article_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        victim,
        on=["customer_id", "victim_article_id"],
        how="left",
        validate="many_to_one",
    )
    assert chosen[["challenger_probability", "victim_probability"]].notna().all().all()
    chosen["probability_difference"] = chosen.challenger_probability - chosen.victim_probability
    chosen["expected_delta"] = chosen.probability_difference * chosen.unit_gain
    chosen = chosen[chosen.expected_delta > 0]
    chosen = chosen.sort_values(
        [
            "customer_id",
            "expected_delta",
            "probability_difference",
            "victim_rank",
            "challenger_rank",
            "challenger_article_id",
            "victim_article_id",
        ],
        ascending=[True, False, False, False, True, True, True],
        kind="mergesort",
    ).drop_duplicates("customer_id", keep="first")
    chosen["actual_delta"] = (
        chosen.challenger_target.astype(np.int8) - chosen.victim_target.astype(np.int8)
    ) * chosen.unit_gain
    return chosen.sort_values("customer_id", kind="mergesort").reset_index(drop=True)


def evaluate(folder: str, name: str, model: lgb.Booster, stage: str):
    path, meta = source(folder)
    started = time.perf_counter()
    candidates, details = candidate_frame(meta["cutoff"], with_labels=True)
    probability = model.predict(candidates[FEATURES].to_numpy(np.float32), num_threads=4)
    probability_frame = candidates[["customer_id", "article_id"]].copy()
    probability_frame["purchase_probability"] = probability
    pairs = pair_frame(path, discordant_only=False)
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
        "parent_diagnostic": "WV3-621",
        "architecture_family": "candidate_level_pointwise_purchase_propensity",
        "hypothesis": "Representing each candidate once and subtracting marginal purchase probabilities will avoid neutral-pair inflation and improve one-swap decisions beyond WV3-610.",
        "features": FEATURES,
        "training": "unweighted binary LightGBM on active rank8-50 candidates from four 2019 historical sources; one row per user-item-window",
        "why_unweighted": "class reweighting would distort the marginal probability scale used in challenger-minus-victim expected value",
        "model": {"params": PARAMS, "rounds": ROUNDS},
        "decision_score": "(P(challenger purchased)-P(victim purchased))*exact position unit_gain",
        "action": "one maximum positive expected-gain swap; ranks1-7 immutable",
        "candidate_pool": "unchanged WV2-601 pool; ranks8-50 only",
        "input_gate": "unique user-item-window rows, at least 5000 positives, positive rate below 10 percent",
        "inner_gate": {
            "standard_vs_WV2_601": "mean>=+0.0001, >=3 positive, worst>=-0.0005",
            "incremental_vs_WV3_610": "mean>0, >=3 nondegrade, worst>=-0.0002",
        },
        "expected_minutes": 15,
        "outer_policy": "one frozen exposure only if input and both inner gates pass; no class-weight, threshold, feature or tree rescue",
        "fallback": "retain WV3-610 stable candidate and close pointwise label-factorization family",
        "comparison_to_WV3_501": "WV3-501 broadly reordered Top50 with a neural list-context residual; WV3-630 protects ranks1-7 and makes at most one exact expected-value swap",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(15)
    assert read(common.REPORT / "WV3-621_PAIR_FACTORIZATION_AUDIT.json")["candidate_level_propensity_experiment_authorized"]
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "candidate_level_pointwise_purchase_propensity",
            experiment_contract()["hypothesis"],
            candidate_protocol=experiment_contract()["candidate_pool"],
            training_protocol=experiment_contract()["training"],
            features=FEATURES,
            params={"model": experiment_contract()["model"], "decision_score": experiment_contract()["decision_score"], "inner_gate": experiment_contract()["inner_gate"], "final_week": "not_run"},
            expected_minutes=15,
        )


def incremental_gate(windows: dict) -> dict:
    prior = read(common.REPORT / "WV3-610_SCREEN.json")["windows"]
    delta = {name: row["MAP@12"] - prior[name]["MAP@12"] for name, row in windows.items()}
    values = list(delta.values())
    return {
        "per_window_delta_vs_WV3_610": delta,
        "mean_delta_vs_WV3_610": float(np.mean(values)),
        "nondegrade_windows_vs_WV3_610": sum(value >= 0 for value in values),
        "worst_delta_vs_WV3_610": min(values),
        "passed": bool(np.mean(values) > 0 and sum(value >= 0 for value in values) >= 3 and min(values) >= -0.0002),
    }


def render_screen(result: dict) -> str:
    lines = [
        "# WV3-630 候选商品级购买倾向局部准入",
        "",
        "## 结论",
        "",
        f"内层总门槛：**{'pass' if result['passed'] else 'fail'}**。相对 WV2-601 的平均 population delta 为 {result['standard_screening_vs_WV2_601']['mean_population_delta']:+.9f}；相对 WV3-610 为 {result['incremental_screening_vs_WV3_610']['mean_delta_vs_WV3_610']:+.9f}。",
        "",
        "## 术语与模型",
        "",
        "- pointwise purchase propensity（行业常见，本报告称候选商品级购买倾向）：对一个用户—商品在下一周是否购买做二分类概率估计；每个历史用户—商品—窗口只出现一行。概率是排序代理，不宣称完成严格校准。",
        "- 边际概率差换位（本项目自定义）：挑战商品的预测购买概率减被替换商品的预测购买概率，再乘该位置直接换位对应的 AP@12 权重；只执行期望值大于0的最佳一次换位。",
        "- 局部准入（本项目自定义）：第1–7名固定，挑战商品来自第13–50名，被替换商品来自第8–12名，每位活跃用户最多换一次；候选池不变。",
        "- 中性替换（本项目自定义）：监督周两件商品同为买或同为未买，直接交换的真实 AP@12 变化为0；中性占比分母为实际执行替换的用户数。",
        "- population delta（本项目沿用指标）：以窗口全部真值用户为分母，实验 MAP@12 减 WV2-601 MAP@12。",
        "",
        f"训练共有 {result['training_audit']['rows']:,} 个唯一用户—商品—窗口样本，正例 {result['training_audit']['positive_rows']:,}，正例率 {result['training_audit']['positive_rate']:.3%}；未做类别重加权，以免改变用于作差的边际概率尺度。",
        "",
        "## 四窗内层结果",
        "",
        "| 窗口 | 选择用户数 | 有益 / 有害 / 中性 | 中性占比 | 相对 WV2-601 | 相对 WV3-610 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    incremental = result["incremental_screening_vs_WV3_610"]["per_window_delta_vs_WV3_610"]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['selected_users']:,} | {row['beneficial_selected_users']:,} / {row['harmful_selected_users']:,} / {row['neutral_selected_users']:,} | {row['selected_neutral_share']:.2%} | {row['population_delta']:+.9f} | {incremental[name]:+.9f} |"
        )
    lines += [
        "",
        "只有标准门槛和相对 WV3-610 的增量门槛同时通过，才允许冻结模型并读取一次外层。失败后不调整类别权重、阈值、特征或树参数。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(15)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    started = time.perf_counter()
    training, audit = historical_training_data()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    write(MODEL_ROOT / "TRAINING_AUDIT.json", audit)
    if not audit["passed"]:
        common.update(TRIAL, decision="reject_input_audit", inner_evidence=audit, runtime=time.perf_counter() - started, artifact_paths=[str(MODEL_ROOT / "TRAINING_AUDIT.json")])
        raise RuntimeError("WV3-630 candidate-level input gate failed")
    model = fit_model(training)
    del training
    gc.collect()
    model_path = MODEL_ROOT / "MODEL.txt"
    model.save_model(str(model_path))
    write(MODEL_ROOT / "MODEL.json", {"experiment_id": TRIAL, "model": evidence_id(model_path, reason="explicit_registry_evidence"), "features": FEATURES, "params": PARAMS, "rounds": ROUNDS, "training_audit": str(MODEL_ROOT / "TRAINING_AUDIT.json"), "outer_labels_used": False, "final_week": "not_run"})
    windows = {}
    for name, folder in INNER:
        row, chosen = evaluate(folder, name, model, "historical_candidate_propensity_inner_screen")
        windows[name] = row
        root = MODEL_ROOT / name / "inner"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"pointwise_inner": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = screening_gate(row["population_delta"] for row in windows.values())
    incremental = incremental_gate(windows)
    passed = standard["passed"] and incremental["passed"]
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "candidate_level_pointwise_purchase_propensity", "training_audit": audit, "windows": windows, "standard_screening_vs_WV2_601": standard, "incremental_screening_vs_WV3_610": incremental, "passed": passed, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render_screen(result), encoding="utf-8")
    common.update(TRIAL, decision="inner_pass" if passed else "reject_inner", inner_evidence={"input": audit, "standard": standard, "incremental": incremental, "passed": passed}, runtime=result["runtime_seconds"], artifact_paths=[str(SCREEN_REPORT), str(SCREEN_MARKDOWN), str(MODEL_ROOT / "MODEL.json")])
    common.log(
        f"{TRIAL} inner screen",
        "WV3-621 found only 1.258 percent actionable historical pairs and about 96 percent neutral selected swaps.",
        experiment_contract()["hypothesis"],
        f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",
        f"input pass={audit['passed']}; positive rate={audit['positive_rate']:.3%}; standard mean={standard['mean_population_delta']:+.9f}; incremental mean={incremental['mean_delta_vs_WV3_610']:+.9f}; pass={passed}.",
        "Freeze and expose once." if passed else "Reject pointwise propensity at inner; do not rescue its weighting or threshold.",
        "Run one outer confirmation if both gates pass; otherwise retain WV3-610 and diagnose a different architecture family.",
        alternatives="Unlike WV3-501, this model cannot reorder the full Top50 and changes at most one protected-tail relation.",
        experiment="Fit one unweighted candidate-level binary tree model on four historical sources and compute exact probability-difference swap value.",
        reflection="This changes label factorization while holding the candidate set and action space fixed.",
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
        row, chosen = evaluate(folder, name, model, "frozen_candidate_propensity_outer")
        windows[name] = row
        root = MODEL_ROOT / name / "outer"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"pointwise_outer": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    prior = read(common.REPORT / "WV3-610_OUTER.json")
    delta = {name: windows[name]["MAP@12"] - prior["per_window_MAP"][name] for name in windows}
    values = list(delta.values())
    incremental = {"per_window_delta_vs_WV3_610": delta, "mean_delta_vs_WV3_610": float(np.mean(values)), "nondegrade_windows_vs_WV3_610": sum(value >= 0 for value in values), "worst_delta_vs_WV3_610": min(values)}
    better = standard["stable"] and incremental["mean_delta_vs_WV3_610"] > 0
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "candidate_level_pointwise_purchase_propensity", "windows": windows, **standard, "incremental_vs_WV3_610": incremental, "better_than_highest_mean_candidate": better, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(OUTER_REPORT, result)
    common.update(TRIAL, decision="promote_candidate" if better else "reject_outer", outer_MAP_by_window=standard["per_window_MAP"], mean_MAP=standard["mean_MAP"], delta_vs_WV2_601=standard["delta_vs_WV2_601"], nondegrade_windows=standard["nondegrade_windows"], worst_delta=standard["worst_delta"], runtime=result["runtime_seconds"], artifact_paths=entry["artifact_paths"] + [str(OUTER_REPORT)])
    common.log(
        f"{TRIAL} outer closure",
        "The pointwise propensity architecture passed both inner gates before freezing.",
        experiment_contract()["hypothesis"],
        str(OUTER_REPORT),
        f"mean MAP={standard['mean_MAP']:.9f}; delta vs WV2-601={standard['delta_vs_WV2_601']:+.9f}; incremental mean vs WV3-610={incremental['mean_delta_vs_WV3_610']:+.9f}; stable={standard['stable']}; better={better}.",
        "Promote as highest-mean stable candidate." if better else "Reject outer and retain WV3-610.",
        "Continue only through a preregistered different architecture family if target 0.03 remains unmet and time permits.",
        alternatives="No outer-driven class weight, threshold, feature or tree rescue is allowed.",
        experiment="One outer exposure of the exact frozen historical candidate-level model.",
        reflection="The outer check tests temporal transfer of the changed supervision factorization.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "better": better}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "screen", "confirm"])
    globals()[parser.parse_args().command]()
