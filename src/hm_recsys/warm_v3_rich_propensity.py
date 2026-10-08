"""WV3-661: rich-feature Top50 candidate propensity with protected one-swap admission."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import evidence_id, now, read, write
from .warm_v2_engine import connection, literal, save_parquet
from .warm_v3_expert import screening_gate
from .warm_v3_pointwise_propensity import FEATURES as BASE_FEATURES
from .warm_v3_pointwise_propensity import PARAMS, ROUNDS, candidate_frame, choose_propensity_decisions
from .warm_v3_residual_admission import HISTORICAL, INNER, OUTER, pair_frame, source
from .warm_v3_rich_feature_audit import EXTRA_FEATURES, FEATURE_GROUPS


TRIAL = "WV3-661"
CONTRACT = common.REPORT / "WV3-661_RICH_PROPENSITY_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-661_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-661_SCREEN.md"
OUTER_REPORT = common.REPORT / "WV3-661_OUTER.json"
MODEL_ROOT = common.ART / TRIAL
FEATURES = BASE_FEATURES + EXTRA_FEATURES


def rich_candidate_frame(cutoff: str, rank_min: int = 1) -> tuple[pd.DataFrame, dict]:
    frame, basic = candidate_frame(cutoff, with_labels=True, rank_min=rank_min, rank_max=50)
    candidate_meta = read(common.ART / "candidate_gate_data" / cutoff / "DATA.json")
    base_path = Path(candidate_meta["source_identity"]["base_path"])
    keys_path = Path(candidate_meta["keys"])
    feature_sql = ",".join(f"f.{name}" for name in EXTRA_FEATURES)
    with connection() as con:
        extra = con.execute(
            f"""SELECT k.customer_id,k.article_id,{feature_sql}
            FROM read_parquet({literal(keys_path)}) k
            JOIN read_parquet({literal(base_path)}) f USING(customer_id,article_id)
            WHERE k.user_history_events_12w>0 AND k.rf BETWEEN {rank_min} AND 50"""
        ).fetchdf()
    assert len(extra) == len(frame)
    assert not extra.duplicated(["customer_id", "article_id"]).any()
    result = frame.merge(extra, on=["customer_id", "article_id"], how="left", validate="one_to_one")
    assert len(result) == len(frame)
    values = result[EXTRA_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
    result[EXTRA_FEATURES] = values
    assert np.isfinite(result[FEATURES].to_numpy(np.float64)).all()
    details = {
        **basic,
        "rich_base_path": str(base_path),
        "extra_feature_count": len(EXTRA_FEATURES),
        "extra_features": EXTRA_FEATURES,
        "explicit_missing_value_policy": "numeric coercion then NaN=0, +Inf=1e6, -Inf=-1e6",
        "target_excluded_from_features": "target" not in FEATURES,
    }
    return result, details


def historical_training_data() -> tuple[pd.DataFrame, dict]:
    frames = []
    sources = []
    for name, folder in HISTORICAL:
        _, meta = source(folder)
        frame, details = rich_candidate_frame(meta["cutoff"], rank_min=1)
        frame["source_window"] = name
        frames.append(frame)
        sources.append({**details, "window": name})
    training = pd.concat(frames, ignore_index=True)
    matrix = training[EXTRA_FEATURES].to_numpy(np.float32)
    varying = (matrix.max(axis=0) - matrix.min(axis=0)) > 0
    audit = {
        "role": "strictly_historical_rich_candidate_purchase_supervision",
        "sources": sources,
        "rows": len(training),
        "positive_rows": int(training.target.sum()),
        "positive_rate": float(training.target.mean()),
        "unique_user_item_window_rows": int(training.groupby(["source_window", "customer_id", "article_id"]).ngroups),
        "base_feature_count": len(BASE_FEATURES),
        "extra_feature_count": len(EXTRA_FEATURES),
        "total_feature_count": len(FEATURES),
        "varying_extra_features": [name for name, flag in zip(EXTRA_FEATURES, varying) if flag],
        "constant_extra_features": [name for name, flag in zip(EXTRA_FEATURES, varying) if not flag],
        "feature_groups": FEATURE_GROUPS,
        "target_excluded": "target" not in FEATURES,
        "class_weighting": "none",
        "latest_label_end_exclusive": "2019-11-27",
        "earliest_inner_cutoff": "2019-12-25",
        "temporal_gap_days": 28,
        "final_week": "not_run",
    }
    audit["identity_unique"] = audit["unique_user_item_window_rows"] == audit["rows"]
    audit["passed"] = bool(audit["identity_unique"] and audit["positive_rows"] >= 5000 and len(audit["varying_extra_features"]) >= 60 and audit["target_excluded"])
    return training, audit


def fit_model(training: pd.DataFrame) -> lgb.Booster:
    labels = training.target.to_numpy(np.uint8)
    dataset = lgb.Dataset(training[FEATURES].to_numpy(np.float32), label=labels, feature_name=FEATURES, free_raw_data=True)
    return lgb.train(PARAMS, dataset, num_boost_round=ROUNDS)


def evaluate(folder: str, name: str, model: lgb.Booster, stage: str):
    path, meta = source(folder)
    started = time.perf_counter()
    candidates, details = rich_candidate_frame(meta["cutoff"], rank_min=8)
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
    return {
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
    }, chosen


def experiment_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-631",
        "parent_diagnostic": "WV3-660",
        "architecture_family": "rich_feature_candidate_purchase_propensity",
        "hypothesis": "Direct retrieval-source, item-trend and multi-level user-attribute affinity signals will improve candidate purchase-propensity ordering beyond the narrow 24-feature WV3-631 model.",
        "base_features": BASE_FEATURES,
        "added_feature_groups": FEATURE_GROUPS,
        "feature_policy": "use all 75 preregistered extra fields; no outcome-driven column selection or ablation",
        "missing_value_policy": "numeric coercion then NaN=0, +Inf=1e6, -Inf=-1e6",
        "training": "unweighted binary LightGBM on active Top50 candidates from four 2019 historical sources",
        "model": {"params": PARAMS, "rounds": ROUNDS},
        "decision_score": "(P(challenger purchased)-P(victim purchased))*exact position unit_gain",
        "candidate_pool": "WV3-631 Top50; ranks1-7 protected, challengers13-50, victims8-12, maximum one swap",
        "input_gate": "identity unique, at least5000 positives, at least60/75 extra fields vary, target excluded",
        "inner_gate": {
            "standard_vs_WV2_601": "mean>=+0.0001, >=3 positive, worst>=-0.0005",
            "incremental_vs_WV3_631": "mean>0, >=3 nondegrade, worst>=-0.0002",
        },
        "expected_minutes": 30,
        "outer_policy": "one frozen exposure only if input and both inner gates pass; no feature removal, decay edit, class weight, threshold or tree rescue",
        "fallback": "retain WV3-650/WV3-631 and close frozen-cache rich-feature model if it fails",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(30)
    assert read(common.REPORT / "WV3-660_RICH_FEATURE_AUDIT.json")["rich_pointwise_experiment_authorized"]
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(TRIAL, "rich_feature_candidate_purchase_propensity", experiment_contract()["hypothesis"], candidate_protocol=experiment_contract()["candidate_pool"], training_protocol=experiment_contract()["training"], features=FEATURES, params={"model": experiment_contract()["model"], "feature_policy": experiment_contract()["feature_policy"], "inner_gate": experiment_contract()["inner_gate"], "final_week": "not_run"}, expected_minutes=30)


def incremental_gate(windows: dict) -> dict:
    prior = read(common.REPORT / "WV3-631_SCREEN.json")["windows"]
    delta = {name: row["MAP@12"] - prior[name]["MAP@12"] for name, row in windows.items()}
    values = list(delta.values())
    return {"per_window_delta_vs_WV3_631": delta, "mean_delta_vs_WV3_631": float(np.mean(values)), "nondegrade_windows_vs_WV3_631": sum(value >= 0 for value in values), "worst_delta_vs_WV3_631": min(values), "passed": bool(np.mean(values) > 0 and sum(value >= 0 for value in values) >= 3 and min(values) >= -0.0002)}


def render_screen(result: dict) -> str:
    standard = result["standard_screening_vs_WV2_601"]
    incremental = result["incremental_screening_vs_WV3_631"]
    audit = result["training_audit"]
    lines = [
        "# WV3-661 丰富候选特征购买倾向准入",
        "",
        "## 结论",
        "",
        f"内层总门槛：**{'pass' if result['passed'] else 'fail'}**。相对 WV2-601 平均增加 {standard['mean_population_delta']:+.9f}；相对24维 WV3-631 平均增加 {incremental['mean_delta_vs_WV3_631']:+.9f}。",
        "",
        "## 术语与特殊处理",
        "",
        "- 丰富候选特征（本项目自定义）：24个原输入加75个冻结缓存字段，共99维；新增字段覆盖召回来源证据、用户/商品状态、商品趋势和五个属性层级上的用户偏好。完整分组定义见 WV3-660 报告。",
        "- 候选商品级购买倾向（行业常见 pointwise propensity 思路）：每个历史用户—商品—窗口一行，预测下一周购买；不做类别重加权。",
        "- 属性交叉偏好（推荐系统常见）：候选所属商品编码、品类、部门、服装组、颜色或索引组在用户近期历史中的事件数、占比、间隔及衰减计数。",
        "- 缺失值策略（本项目自定义）：数值转换失败和 NaN 置0，正负无穷分别截为正负1e6；策略在训练前固定，所有窗口一致。",
        "- population delta（本项目沿用指标）：以窗口全部真值用户为分母，实验 MAP@12 减 WV2-601 MAP@12。",
        "",
        f"训练 {audit['rows']:,} 行、正例 {audit['positive_rows']:,}；75个新增特征中 {len(audit['varying_extra_features'])} 个在历史训练中有变化，常量列 {len(audit['constant_extra_features'])} 个。",
        "",
        "| 内层窗口 | 选择用户数 | 有益 / 有害 / 中性 | 中性占比 | 相对 WV2-601 | 相对 WV3-631 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(f"| {name} | {row['selected_users']:,} | {row['beneficial_selected_users']:,} / {row['harmful_selected_users']:,} / {row['neutral_selected_users']:,} | {row['selected_neutral_share']:.2%} | {row['population_delta']:+.9f} | {incremental['per_window_delta_vs_WV3_631'][name]:+.9f} |")
    lines += [
        "",
        "本实验一次性使用预注册字段，不按结果删掉衰减、趋势或来源列。只有两个内层门槛同时通过才读取一次外层；失败后不做字段消融或参数救援。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(30)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    started = time.perf_counter()
    training, audit = historical_training_data()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    write(MODEL_ROOT / "TRAINING_AUDIT.json", audit)
    if not audit["passed"]:
        common.update(TRIAL, decision="reject_input_audit", inner_evidence=audit, runtime=time.perf_counter() - started, artifact_paths=[str(MODEL_ROOT / "TRAINING_AUDIT.json")])
        raise RuntimeError("WV3-661 rich feature input gate failed")
    model = fit_model(training)
    del training
    gc.collect()
    model_path = MODEL_ROOT / "MODEL.txt"
    model.save_model(str(model_path))
    write(MODEL_ROOT / "MODEL.json", {"experiment_id": TRIAL, "model": evidence_id(model_path, reason="explicit_registry_evidence"), "features": FEATURES, "params": PARAMS, "rounds": ROUNDS, "training_audit": str(MODEL_ROOT / "TRAINING_AUDIT.json"), "outer_labels_used": False, "final_week": "not_run"})
    windows = {}
    for name, folder in INNER:
        row, chosen = evaluate(folder, name, model, "rich_propensity_inner_screen")
        windows[name] = row
        root = MODEL_ROOT / name / "inner"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"rich_propensity_inner": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = screening_gate(row["population_delta"] for row in windows.values())
    incremental = incremental_gate(windows)
    passed = standard["passed"] and incremental["passed"]
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "rich_feature_candidate_purchase_propensity", "training_audit": audit, "windows": windows, "standard_screening_vs_WV2_601": standard, "incremental_screening_vs_WV3_631": incremental, "passed": passed, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render_screen(result), encoding="utf-8")
    common.update(TRIAL, decision="inner_pass" if passed else "reject_inner", inner_evidence={"input": audit, "standard": standard, "incremental": incremental, "passed": passed}, runtime=result["runtime_seconds"], artifact_paths=[str(SCREEN_REPORT), str(SCREEN_MARKDOWN), str(MODEL_ROOT / "MODEL.json")])
    common.log(
        f"{TRIAL} inner screen",
        "WV3-660 found 75 frozen retrieval, trend and affinity fields missing from the stable 24-feature propensity model, with complete schemas and no target input.",
        experiment_contract()["hypothesis"],
        f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",
        f"input pass={audit['passed']}; varying extras={len(audit['varying_extra_features'])}/75; standard mean={standard['mean_population_delta']:+.9f}; incremental mean={incremental['mean_delta_vs_WV3_631']:+.9f}; pass={passed}.",
        "Freeze and expose once." if passed else "Reject the all-feature model; no outcome-driven column removal.",
        "Run one outer confirmation if both gates pass; otherwise retain WV3-650/WV3-631.",
        alternatives="A bounded semantic group was fixed before outcomes; feature-by-feature search was explicitly excluded.",
        experiment="Fit the same historical Top50 pointwise model with all 75 additional frozen-cache fields.",
        reflection="The comparison isolates representation breadth rather than tree or action changes.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "passed": passed, "seconds": result["runtime_seconds"]}, flush=True)
    return result


def confirm() -> dict:
    common.setup()
    common.budget(20)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "inner_pass" and entry["outer_exposures"] == 1
    assert read(SCREEN_REPORT)["passed"]
    if OUTER_REPORT.exists():
        return read(OUTER_REPORT)
    model = lgb.Booster(model_file=str(MODEL_ROOT / "MODEL.txt"))
    started = time.perf_counter()
    windows = {}
    for name, folder in OUTER:
        row, chosen = evaluate(folder, name, model, "frozen_rich_propensity_outer")
        windows[name] = row
        root = MODEL_ROOT / name / "outer"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"rich_propensity_outer": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    prior = read(common.REPORT / "WV3-650_OUTER.json")
    delta = {name: windows[name]["MAP@12"] - prior["per_window_MAP"][name] for name in windows}
    values = list(delta.values())
    incremental = {"per_window_delta_vs_WV3_650": delta, "mean_delta_vs_WV3_650": float(np.mean(values)), "nondegrade_windows_vs_WV3_650": sum(value >= 0 for value in values), "worst_delta_vs_WV3_650": min(values)}
    better = bool(standard["stable"] and np.mean(values) > 0 and sum(value >= 0 for value in values) >= 3 and min(values) >= -0.0002)
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "rich_feature_candidate_purchase_propensity", "windows": windows, **standard, "incremental_vs_WV3_650": incremental, "better_than_current_champion": better, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(OUTER_REPORT, result)
    common.update(TRIAL, decision="promote_candidate" if better else "reject_outer", outer_MAP_by_window=standard["per_window_MAP"], mean_MAP=standard["mean_MAP"], delta_vs_WV2_601=standard["delta_vs_WV2_601"], nondegrade_windows=standard["nondegrade_windows"], worst_delta=standard["worst_delta"], runtime=result["runtime_seconds"], artifact_paths=entry["artifact_paths"] + [str(OUTER_REPORT)])
    common.log(
        f"{TRIAL} outer closure",
        "The all-feature propensity model passed its schema, leakage and two inner gates before freezing.",
        experiment_contract()["hypothesis"],
        str(OUTER_REPORT),
        f"mean MAP={standard['mean_MAP']:.9f}; delta vs WV2-601={standard['delta_vs_WV2_601']:+.9f}; incremental mean vs WV3-650={incremental['mean_delta_vs_WV3_650']:+.9f}; nondegrade={incremental['nondegrade_windows_vs_WV3_650']}/4; better={better}.",
        "Promote the rich-feature model." if better else "Reject and retain WV3-650/WV3-631.",
        "Continue only through a different bounded mechanism if time permits.",
        alternatives="No outer-driven feature removal, decay edit, class weight, threshold or tree rescue is allowed.",
        experiment="One outer exposure of the exact frozen 99-feature model.",
        reflection="The check tests whether broader candidate representation transfers across all four outer windows.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "better": better}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "screen", "confirm"])
    globals()[parser.parse_args().command]()
