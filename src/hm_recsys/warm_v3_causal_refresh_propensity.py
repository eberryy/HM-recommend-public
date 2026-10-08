"""WV3-650: causal expanding-window refresh of the stable Top50 propensity model."""
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
from .warm_v3_fulltop_propensity import incremental_gate as _unused_incremental_gate
from .warm_v3_pointwise_propensity import FEATURES, PARAMS, ROUNDS, candidate_frame, evaluate, fit_model
from .warm_v3_residual_admission import HISTORICAL, INNER, OUTER, source


TRIAL = "WV3-650"
CONTRACT = common.REPORT / "WV3-650_CAUSAL_REFRESH_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-650_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-650_SCREEN.md"
OUTER_REPORT = common.REPORT / "WV3-650_OUTER.json"
MODEL_ROOT = common.ART / TRIAL


def cutoff(folder: str) -> str:
    return source(folder)[1]["cutoff"]


def label_end(cutoff_value: str) -> date:
    return date.fromisoformat(cutoff_value) + timedelta(days=7)


def training_data(cutoffs: list[str], evaluation_cutoff: str, role: str) -> tuple[pd.DataFrame, dict]:
    assert len(cutoffs) == len(set(cutoffs))
    assert all(label_end(value) < date.fromisoformat(evaluation_cutoff) for value in cutoffs)
    frames = []
    sources = []
    for value in cutoffs:
        frame, details = candidate_frame(value, with_labels=True, rank_min=1, rank_max=50)
        frame["source_cutoff"] = value
        frames.append(frame)
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
        "unique_user_item_window_rows": int(training.groupby(["source_cutoff", "customer_id", "article_id"]).ngroups),
        "features": FEATURES,
        "class_weighting": "none",
        "sources": sources,
        "final_week": "not_run",
    }
    audit["identity_unique"] = audit["unique_user_item_window_rows"] == audit["rows"]
    audit["passed"] = bool(audit["identity_unique"] and audit["positive_rows"] >= 5000 and audit["positive_rate"] < 0.10)
    return training, audit


def fit_for_cutoffs(cutoffs: list[str], evaluation_cutoff: str, role: str, root) -> tuple[lgb.Booster, dict]:
    training, audit = training_data(cutoffs, evaluation_cutoff, role)
    root.mkdir(parents=True, exist_ok=True)
    write(root / "TRAINING_AUDIT.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"{TRIAL} causal input gate failed for {evaluation_cutoff}")
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
        "parent": "WV3-631",
        "architecture_family": "causal_expanding_window_candidate_propensity_refresh",
        "hypothesis": "Refreshing the stable Top50 candidate-level propensity model with every strictly completed inner label week will improve later windows by reducing behavioral staleness.",
        "inner_protocol": "for inner window i, train on four 2019 sources plus only earlier 2020 inner label weeks; the first inner model equals WV3-631 training support",
        "outer_protocol": "before any outer scoring, freeze four models; outer window i uses four 2019 sources plus inner label weeks through i, each ending 21 days before its paired outer cutoff",
        "features": FEATURES,
        "model": {"params": PARAMS, "rounds": ROUNDS, "class_weighting": "none"},
        "candidate_pool": "WV3-631 Top50; ranks1-7 protected, challengers13-50, victims8-12, maximum one swap",
        "inner_gate": {
            "standard_vs_WV2_601": "mean>=+0.0001, >=3 positive, worst>=-0.0005",
            "incremental_vs_WV3_631": "mean>0, >=3 nondegrade, worst>=-0.0002",
        },
        "expected_minutes": 60,
        "outer_policy": "one frozen algorithm exposure only if both inner gates pass; no decay weighting, recency cutoff, threshold, feature or tree rescue",
        "fallback": "retain static WV3-631; close simple expanding-window refresh if it fails",
        "distinction_from_prior_temporal_trials": "WV3-351 ensembled old rank checkpoints and WV3-371 routed between finished rankers; WV3-650 refreshes the newly stable candidate-purchase estimator itself",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(60)
    assert read(common.REPORT / "WV3-631_OUTER.json")["better_than_highest_mean_candidate"]
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "causal_expanding_window_candidate_propensity_refresh",
            experiment_contract()["hypothesis"],
            candidate_protocol=experiment_contract()["candidate_pool"],
            training_protocol=experiment_contract()["inner_protocol"],
            features=FEATURES,
            params={"model": experiment_contract()["model"], "inner_gate": experiment_contract()["inner_gate"], "outer_protocol": experiment_contract()["outer_protocol"], "final_week": "not_run"},
            expected_minutes=60,
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
        "# WV3-650 因果扩窗购买倾向更新",
        "",
        "## 结论",
        "",
        f"内层总门槛：**{'pass' if result['passed'] else 'fail'}**。相对 WV2-601 平均增加 {standard['mean_population_delta']:+.9f}；相对静态 WV3-631 平均增加 {incremental['mean_delta_vs_WV3_631']:+.9f}。",
        "",
        "## 术语与时间协议",
        "",
        "- causal expanding-window refresh（行业常见时间验证思想，本报告称因果扩窗更新）：评测一个截止日时，只把标签周已经严格结束的更早切片加入训练；训练集随时间扩张，不删除旧样本、不对近期样本调权。",
        "- 静态模型（本项目对照）：WV3-631 始终只用四个 2019 历史切片训练，同一个模型评测四个 2020 内层窗口。",
        "- 内层前向筛选（本项目自定义）：第一个内层窗口只用2019训练；后续内层窗口依次加入更早内层标签。当前窗口自己的标签只用于评分，不进入该窗口模型。",
        "- 外层冻结算法（本项目自定义）：若内层通过，在读取任何外层评分前先冻结四个模型；每个外层窗口允许加入与其配对、且标签结束日早于外层截止日21天的内层切片。",
        "- 候选商品级购买倾向（行业常见 pointwise propensity 思路）：每个用户—商品—窗口一行，预测下一周购买；以挑战商品概率减被替换商品概率决定一次尾部换位。",
        "- population delta（本项目沿用指标）：以窗口全部真值用户为分母，实验 MAP@12 减 WV2-601 MAP@12。",
        "",
        "| 内层窗口 | 训练切片数 | 最晚训练标签结束日 | 相对 WV2-601 | 相对静态 WV3-631 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        audit = result["training_audits"][name]
        lines.append(f"| {name} | {len(audit['training_cutoffs'])} | {audit['training_label_end_exclusive'][-1][1]} | {row['population_delta']:+.9f} | {incremental['per_window_delta_vs_WV3_631'][name]:+.9f} |")
    lines += [
        "",
        "模型参数、候选池和动作空间与 WV3-631 相同；本实验只改变每个时间点可用的历史监督。失败后不搜索衰减率、回看长度或更新频率。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(60)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    started = time.perf_counter()
    historical_cutoffs = [cutoff(folder) for _, folder in HISTORICAL]
    inner_cutoffs = [cutoff(folder) for _, folder in INNER]
    windows = {}
    audits = {}
    for index, (name, folder) in enumerate(INNER):
        evaluation_cutoff = inner_cutoffs[index]
        training_cutoffs = historical_cutoffs + inner_cutoffs[:index]
        root = MODEL_ROOT / name / "inner"
        model, audit = fit_for_cutoffs(training_cutoffs, evaluation_cutoff, "causal_inner_training", root)
        row, chosen = evaluate(folder, name, model, "causal_refresh_inner_screen")
        windows[name] = row
        audits[name] = audit
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"causal_refresh_inner": name, "delta": row["population_delta"], "training_cutoffs": len(training_cutoffs)}, flush=True)
        del model
        gc.collect()
    standard = screening_gate(row["population_delta"] for row in windows.values())
    incremental = incremental_gate(windows)
    passed = standard["passed"] and incremental["passed"]
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "causal_expanding_window_candidate_propensity_refresh", "training_audits": audits, "windows": windows, "standard_screening_vs_WV2_601": standard, "incremental_screening_vs_WV3_631": incremental, "passed": passed, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render_screen(result), encoding="utf-8")
    common.update(TRIAL, decision="inner_pass" if passed else "reject_inner", inner_evidence={"standard": standard, "incremental": incremental, "passed": passed}, runtime=result["runtime_seconds"], artifact_paths=[str(SCREEN_REPORT), str(SCREEN_MARKDOWN)] + [str(MODEL_ROOT / name / "inner" / "MODEL.json") for name, _ in INNER])
    common.log(
        f"{TRIAL} inner screen",
        "The stable WV3-631 model used only four 2019 quarterly snapshots for all 2020 windows, leaving supervision freshness as an untested bottleneck.",
        experiment_contract()["hypothesis"],
        f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",
        f"standard mean={standard['mean_population_delta']:+.9f}; incremental mean={incremental['mean_delta_vs_WV3_631']:+.9f}; nondegrade={incremental['nondegrade_windows_vs_WV3_631']}/4; pass={passed}.",
        "Freeze the causal update algorithm and expose once." if passed else "Reject simple expanding-window refresh without decay or horizon rescue.",
        "Pretrain all four causal outer models before any outer scoring if passed; otherwise retain WV3-631.",
        alternatives="Prior temporal trials combined or routed finished rankers; this experiment updates the successful candidate-propensity estimator itself.",
        experiment="Forward-chain 2019 plus strictly earlier inner labels with fixed Top50 propensity parameters.",
        reflection="The first window is an exact static control; gains must emerge prospectively in later windows.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "passed": passed, "seconds": result["runtime_seconds"]}, flush=True)
    return result


def confirm() -> dict:
    common.setup()
    common.budget(60)
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
    # Freeze every model before any outer candidate labels are scored.
    for index, (name, _) in enumerate(OUTER):
        training_cutoffs = historical_cutoffs + inner_cutoffs[: index + 1]
        root = MODEL_ROOT / name / "outer_frozen"
        model, audit = fit_for_cutoffs(training_cutoffs, outer_cutoffs[index], "causal_outer_training_before_scoring", root)
        frozen_models[name] = model
        audits[name] = audit
        print({"causal_refresh_frozen": name, "training_cutoffs": len(training_cutoffs)}, flush=True)
    windows = {}
    for name, folder in OUTER:
        row, chosen = evaluate(folder, name, frozen_models[name], "frozen_causal_refresh_outer")
        windows[name] = row
        root = MODEL_ROOT / name / "outer"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"causal_refresh_outer": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    prior = read(common.REPORT / "WV3-631_OUTER.json")
    delta = {name: windows[name]["MAP@12"] - prior["per_window_MAP"][name] for name in windows}
    values = list(delta.values())
    incremental = {"per_window_delta_vs_WV3_631": delta, "mean_delta_vs_WV3_631": float(np.mean(values)), "nondegrade_windows_vs_WV3_631": sum(value >= 0 for value in values), "worst_delta_vs_WV3_631": min(values)}
    better = bool(standard["stable"] and np.mean(values) > 0 and sum(value >= 0 for value in values) >= 3 and min(values) >= -0.0002)
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "causal_expanding_window_candidate_propensity_refresh", "training_audits": audits, "windows": windows, **standard, "incremental_vs_WV3_631": incremental, "better_than_current_champion": better, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(OUTER_REPORT, result)
    common.update(TRIAL, decision="promote_candidate" if better else "reject_outer", outer_MAP_by_window=standard["per_window_MAP"], mean_MAP=standard["mean_MAP"], delta_vs_WV2_601=standard["delta_vs_WV2_601"], nondegrade_windows=standard["nondegrade_windows"], worst_delta=standard["worst_delta"], runtime=result["runtime_seconds"], artifact_paths=entry["artifact_paths"] + [str(OUTER_REPORT)])
    common.log(
        f"{TRIAL} outer closure",
        "The causal expanding-window algorithm passed its forward inner screen before all outer models were pretrained and frozen.",
        experiment_contract()["hypothesis"],
        str(OUTER_REPORT),
        f"mean MAP={standard['mean_MAP']:.9f}; delta vs WV2-601={standard['delta_vs_WV2_601']:+.9f}; incremental mean vs WV3-631={incremental['mean_delta_vs_WV3_631']:+.9f}; nondegrade={incremental['nondegrade_windows_vs_WV3_631']}/4; better={better}.",
        "Promote the causal refresh algorithm." if better else "Reject causal refresh and retain static WV3-631.",
        "Continue only through a different bounded mechanism if the target remains unmet and time permits.",
        alternatives="No outer-driven decay, lookback, update frequency, threshold, feature or tree rescue is allowed.",
        experiment="One outer exposure of four causal models frozen before any outer scoring.",
        reflection="This checks whether supervision freshness transfers under the exact operational time boundary.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "better": better}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "screen", "confirm"])
    globals()[parser.parse_args().command]()
