"""WV3-670: causal refresh of the rich-feature Top50 propensity architecture."""
from __future__ import annotations

import gc
import time
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import evidence_id, now, read, write
from .warm_v2_engine import save_parquet
from .warm_v3_expert import screening_gate
from .warm_v3_residual_admission import HISTORICAL, INNER, OUTER, source
from .warm_v3_rich_propensity import FEATURES, PARAMS, ROUNDS, evaluate, fit_model, rich_candidate_frame


TRIAL = "WV3-670"
CONTRACT = common.REPORT / "WV3-670_CAUSAL_RICH_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-670_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-670_SCREEN.md"
OUTER_REPORT = common.REPORT / "WV3-670_OUTER.json"
MODEL_ROOT = common.ART / TRIAL
_FRAME_CACHE: dict[str, tuple[pd.DataFrame, dict]] = {}


def cutoff(folder: str) -> str:
    return source(folder)[1]["cutoff"]


def label_end(value: str) -> date:
    return date.fromisoformat(value) + timedelta(days=7)


def cached_frame(value: str) -> tuple[pd.DataFrame, dict]:
    if value not in _FRAME_CACHE:
        _FRAME_CACHE[value] = rich_candidate_frame(value, rank_min=1)
    return _FRAME_CACHE[value]


def training_data(cutoffs: list[str], evaluation_cutoff: str, role: str) -> tuple[pd.DataFrame, dict]:
    assert len(cutoffs) == len(set(cutoffs))
    assert all(label_end(value) < date.fromisoformat(evaluation_cutoff) for value in cutoffs)
    frames = []
    sources = []
    for value in cutoffs:
        frame, details = cached_frame(value)
        part = frame.copy()
        part["source_cutoff"] = value
        frames.append(part)
        sources.append(details)
    training = pd.concat(frames, ignore_index=True)
    audit = {
        "role": role,
        "evaluation_cutoff": evaluation_cutoff,
        "training_cutoffs": cutoffs,
        "training_label_end_exclusive": [(value, label_end(value).isoformat()) for value in cutoffs],
        "all_labels_strictly_before_evaluation_cutoff": True,
        "rows": len(training),
        "positive_rows": int(training.target.sum()),
        "positive_rate": float(training.target.mean()),
        "unique_user_item_window_rows": int(sum(len(frame) for frame in frames)),
        "feature_count": len(FEATURES),
        "target_excluded": "target" not in FEATURES,
        "class_weighting": "none",
        "final_week": "not_run",
    }
    audit["identity_unique"] = audit["unique_user_item_window_rows"] == audit["rows"]
    audit["passed"] = bool(audit["identity_unique"] and audit["positive_rows"] >= 5000 and audit["positive_rate"] < 0.10 and audit["target_excluded"])
    return training, audit


def fit_for_cutoffs(cutoffs: list[str], evaluation_cutoff: str, role: str, root):
    training, audit = training_data(cutoffs, evaluation_cutoff, role)
    root.mkdir(parents=True, exist_ok=True)
    write(root / "TRAINING_AUDIT.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"{TRIAL} input gate failed for {evaluation_cutoff}")
    model = fit_model(training)
    del training
    gc.collect()
    model_path = root / "MODEL.txt"
    model.save_model(str(model_path))
    write(root / "MODEL.json", {"experiment_id": TRIAL, "model": evidence_id(model_path, reason="explicit_registry_evidence"), "features": FEATURES, "params": PARAMS, "rounds": ROUNDS, "training_audit": str(root / "TRAINING_AUDIT.json"), "outer_labels_used": False, "final_week": "not_run"})
    return model, audit


def experiment_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parents": ["WV3-650", "WV3-661"],
        "architecture_family": "causal_refresh_rich_feature_candidate_propensity",
        "hypothesis": "The independently stable rich representation and causal supervision refresh contain complementary signal and will improve the protected Top50 propensity policy when combined without changing its action space.",
        "inner_protocol": "forward-chain four 2020 inner windows; train on four 2019 sources plus only earlier inner labels",
        "outer_protocol": "pretrain and freeze all four models before scoring; outer i uses four 2019 sources plus inner labels through i, all strictly earlier",
        "features": "fixed 99 features from WV3-661; no feature selection",
        "model": {"params": PARAMS, "rounds": ROUNDS, "class_weighting": "none"},
        "candidate_pool": "WV3-631 Top50; ranks1-7 protected, challengers13-50, victims8-12, maximum one swap",
        "inner_gate": {
            "standard_vs_WV2_601": "mean>=+0.0001, >=3 positive, worst>=-0.0005",
            "incremental_vs_WV3_661": "mean>0, >=3 nondegrade, worst>=-0.0002",
        },
        "expected_minutes": 90,
        "outer_policy": "one frozen algorithm exposure only if both inner gates pass; no feature, decay, time-window, class-weight, threshold or tree rescue",
        "fallback": "retain simpler WV3-661 or WV3-650 according to final stable-selection policy",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(90)
    assert read(common.REPORT / "WV3-650_OUTER.json")["better_than_current_champion"]
    assert read(common.REPORT / "WV3-661_OUTER.json")["better_than_current_champion"]
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(TRIAL, "causal_refresh_rich_feature_candidate_propensity", experiment_contract()["hypothesis"], candidate_protocol=experiment_contract()["candidate_pool"], training_protocol=experiment_contract()["inner_protocol"], features=FEATURES, params={"model": experiment_contract()["model"], "inner_gate": experiment_contract()["inner_gate"], "outer_protocol": experiment_contract()["outer_protocol"], "final_week": "not_run"}, expected_minutes=90)


def incremental_gate(windows: dict) -> dict:
    prior = read(common.REPORT / "WV3-661_SCREEN.json")["windows"]
    delta = {name: row["MAP@12"] - prior[name]["MAP@12"] for name, row in windows.items()}
    values = list(delta.values())
    return {"per_window_delta_vs_WV3_661": delta, "mean_delta_vs_WV3_661": float(np.mean(values)), "nondegrade_windows_vs_WV3_661": sum(value >= 0 for value in values), "worst_delta_vs_WV3_661": min(values), "passed": bool(np.mean(values) > 0 and sum(value >= 0 for value in values) >= 3 and min(values) >= -0.0002)}


def render_screen(result: dict) -> str:
    standard = result["standard_screening_vs_WV2_601"]
    incremental = result["incremental_screening_vs_WV3_661"]
    lines = [
        "# WV3-670 丰富特征 + 因果扩窗购买倾向",
        "",
        "## 结论",
        "",
        f"内层总门槛：**{'pass' if result['passed'] else 'fail'}**。相对 WV2-601 平均增加 {standard['mean_population_delta']:+.9f}；相对静态丰富特征 WV3-661 平均增加 {incremental['mean_delta_vs_WV3_661']:+.9f}。",
        "",
        "## 术语与控制变量",
        "",
        "- 因果扩窗更新（行业常见前向验证思想）：每个评测截止日只加入监督周已经结束的更早标签切片；训练集合随时间增加，不做近期加权或旧样本删除。",
        "- 丰富候选特征（本项目自定义）：沿用 WV3-661 预注册的99维输入，覆盖召回来源、热度趋势和多层用户属性偏好；不在本实验中重新选列。",
        "- 机制交叉（本项目自定义）：把已分别通过外层稳定门槛的“近期监督更新”和“丰富候选表示”组合，模型参数、Top50候选和一换一动作保持不变。",
        "- population delta（本项目沿用指标）：以窗口全部真值用户为分母，实验 MAP@12 减 WV2-601 MAP@12。",
        "",
        "| 内层窗口 | 训练切片数 | 相对 WV2-601 | 相对 WV3-661 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(f"| {name} | {len(result['training_audits'][name]['training_cutoffs'])} | {row['population_delta']:+.9f} | {incremental['per_window_delta_vs_WV3_661'][name]:+.9f} |")
    lines += [
        "",
        "第一个内层窗口与 WV3-661 是严格同训练支持的确定性对照；组合必须在后续窗口产生前向增益。失败后不搜索回看长度、衰减率或字段子集。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(90)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    started = time.perf_counter()
    historical_cutoffs = [cutoff(folder) for _, folder in HISTORICAL]
    inner_cutoffs = [cutoff(folder) for _, folder in INNER]
    windows = {}
    audits = {}
    for index, (name, folder) in enumerate(INNER):
        root = MODEL_ROOT / name / "inner"
        model, audit = fit_for_cutoffs(historical_cutoffs + inner_cutoffs[:index], inner_cutoffs[index], "causal_rich_inner_training", root)
        row, chosen = evaluate(folder, name, model, "causal_rich_inner_screen")
        windows[name] = row
        audits[name] = audit
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"causal_rich_inner": name, "delta": row["population_delta"], "training_cutoffs": len(audit["training_cutoffs"])}, flush=True)
        del model
        gc.collect()
    standard = screening_gate(row["population_delta"] for row in windows.values())
    incremental = incremental_gate(windows)
    passed = standard["passed"] and incremental["passed"]
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "causal_refresh_rich_feature_candidate_propensity", "training_audits": audits, "windows": windows, "standard_screening_vs_WV2_601": standard, "incremental_screening_vs_WV3_661": incremental, "passed": passed, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render_screen(result), encoding="utf-8")
    common.update(TRIAL, decision="inner_pass" if passed else "reject_inner", inner_evidence={"standard": standard, "incremental": incremental, "passed": passed}, runtime=result["runtime_seconds"], artifact_paths=[str(SCREEN_REPORT), str(SCREEN_MARKDOWN)] + [str(MODEL_ROOT / name / "inner" / "MODEL.json") for name, _ in INNER])
    common.log(
        f"{TRIAL} inner screen",
        "WV3-650 causal refresh and WV3-661 rich representation independently passed their outer stability gates, but their signal overlap was unknown.",
        experiment_contract()["hypothesis"],
        f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",
        f"standard mean={standard['mean_population_delta']:+.9f}; incremental mean={incremental['mean_delta_vs_WV3_661']:+.9f}; nondegrade={incremental['nondegrade_windows_vs_WV3_661']}/4; pass={passed}.",
        "Freeze the combined causal algorithm and expose once." if passed else "Reject the combination; do not tune recency or feature subsets.",
        "Pretrain all four outer models before any outer scoring if passed; otherwise retain the simpler stable candidate.",
        alternatives="The trial crosses two validated axes without changing tree capacity, candidate support or action count.",
        experiment="Forward-chain the fixed 99-feature propensity model on Top50.",
        reflection="The experiment tests complementarity rather than assuming two individually positive mechanisms add.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "passed": passed, "seconds": result["runtime_seconds"]}, flush=True)
    return result


def confirm() -> dict:
    common.setup()
    common.budget(90)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "inner_pass" and entry["outer_exposures"] == 1
    assert read(SCREEN_REPORT)["passed"]
    if OUTER_REPORT.exists():
        return read(OUTER_REPORT)
    started = time.perf_counter()
    historical_cutoffs = [cutoff(folder) for _, folder in HISTORICAL]
    inner_cutoffs = [cutoff(folder) for _, folder in INNER]
    outer_cutoffs = [cutoff(folder) for _, folder in OUTER]
    frozen_models = {}
    audits = {}
    for index, (name, _) in enumerate(OUTER):
        root = MODEL_ROOT / name / "outer_frozen"
        model, audit = fit_for_cutoffs(historical_cutoffs + inner_cutoffs[: index + 1], outer_cutoffs[index], "causal_rich_outer_training_before_scoring", root)
        frozen_models[name] = model
        audits[name] = audit
        print({"causal_rich_frozen": name, "training_cutoffs": len(audit["training_cutoffs"])}, flush=True)
    windows = {}
    for name, folder in OUTER:
        row, chosen = evaluate(folder, name, frozen_models[name], "frozen_causal_rich_outer")
        windows[name] = row
        root = MODEL_ROOT / name / "outer"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"causal_rich_outer": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    prior = read(common.REPORT / "WV3-661_OUTER.json")
    delta = {name: windows[name]["MAP@12"] - prior["per_window_MAP"][name] for name in windows}
    values = list(delta.values())
    incremental = {"per_window_delta_vs_WV3_661": delta, "mean_delta_vs_WV3_661": float(np.mean(values)), "nondegrade_windows_vs_WV3_661": sum(value >= 0 for value in values), "worst_delta_vs_WV3_661": min(values)}
    better = bool(standard["stable"] and np.mean(values) > 0 and sum(value >= 0 for value in values) >= 3 and min(values) >= -0.0002)
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "causal_refresh_rich_feature_candidate_propensity", "training_audits": audits, "windows": windows, **standard, "incremental_vs_WV3_661": incremental, "better_than_current_champion": better, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(OUTER_REPORT, result)
    common.update(TRIAL, decision="promote_candidate" if better else "reject_outer", outer_MAP_by_window=standard["per_window_MAP"], mean_MAP=standard["mean_MAP"], delta_vs_WV2_601=standard["delta_vs_WV2_601"], nondegrade_windows=standard["nondegrade_windows"], worst_delta=standard["worst_delta"], runtime=result["runtime_seconds"], artifact_paths=entry["artifact_paths"] + [str(OUTER_REPORT)])
    common.log(
        f"{TRIAL} outer closure",
        "The fixed combination of two independently stable mechanisms passed the forward inner screen and all outer models were frozen before scoring.",
        experiment_contract()["hypothesis"],
        str(OUTER_REPORT),
        f"mean MAP={standard['mean_MAP']:.9f}; delta vs WV2-601={standard['delta_vs_WV2_601']:+.9f}; incremental mean vs WV3-661={incremental['mean_delta_vs_WV3_661']:+.9f}; nondegrade={incremental['nondegrade_windows_vs_WV3_661']}/4; better={better}.",
        "Promote the combined causal-rich algorithm." if better else "Reject the combination and retain WV3-661.",
        "Continue only through a different bounded mechanism if time remains.",
        alternatives="No outer-driven recency, feature, weight, threshold or tree rescue is allowed.",
        experiment="One outer exposure of four 99-feature causal models frozen before scoring.",
        reflection="This verifies whether the two stable axes are additive under temporal holdout.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "better": better}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "screen", "confirm"])
    globals()[parser.parse_args().command]()
