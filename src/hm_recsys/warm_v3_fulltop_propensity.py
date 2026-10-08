"""WV3-631: full-Top50 historical propensity training, tail-only one-swap inference."""
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
from .warm_v3_pointwise_propensity import FEATURES, PARAMS, ROUNDS, candidate_frame, evaluate, fit_model
from .warm_v3_residual_admission import HISTORICAL, INNER, OUTER, source


TRIAL = "WV3-631"
CONTRACT = common.REPORT / "WV3-631_FULLTOP_PROPENSITY_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-631_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-631_SCREEN.md"
OUTER_REPORT = common.REPORT / "WV3-631_OUTER.json"
MODEL_ROOT = common.ART / TRIAL


def historical_training_data() -> tuple[pd.DataFrame, dict]:
    frames = []
    sources = []
    for name, folder in HISTORICAL:
        _, meta = source(folder)
        frame, details = candidate_frame(meta["cutoff"], with_labels=True, rank_min=1)
        frame["source_window"] = name
        frames.append(frame)
        sources.append({**details, "window": name})
    training = pd.concat(frames, ignore_index=True)
    audit = {
        "role": "strictly_historical_full_top50_candidate_purchase_supervision",
        "sources": sources,
        "rows": len(training),
        "positive_rows": int(training.target.sum()),
        "positive_rate": float(training.target.mean()),
        "unique_user_item_window_rows": int(training.groupby(["source_window", "customer_id", "article_id"]).ngroups),
        "features": FEATURES,
        "training_rank_range": [1, 50],
        "inference_rank_range": [8, 50],
        "class_weighting": "none",
        "latest_label_end_exclusive": "2019-11-27",
        "earliest_inner_cutoff": "2019-12-25",
        "temporal_gap_days": 28,
        "final_week": "not_run",
    }
    audit["identity_unique"] = audit["unique_user_item_window_rows"] == audit["rows"]
    audit["passed"] = bool(audit["identity_unique"] and audit["positive_rows"] >= 5000 and audit["positive_rate"] < 0.10)
    return training, audit


def experiment_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent_input_stop": "WV3-630",
        "architecture_family": "full_top50_candidate_purchase_propensity_tail_inference",
        "hypothesis": "Top1-7 historical candidates add enough positive support for a marginal purchase model, while rank features allow the same model to transfer to the unchanged rank8-50 tail action space.",
        "data_change": "training support expands from baseline ranks8-50 to ranks1-50; inference and decisions remain ranks8-50",
        "features": FEATURES,
        "training": "unweighted binary LightGBM on active rank1-50 candidates from four 2019 sources; one row per user-item-window",
        "model": {"params": PARAMS, "rounds": ROUNDS},
        "decision_score": "(P(challenger purchased)-P(victim purchased))*exact position unit_gain",
        "action": "one maximum positive expected-gain swap; ranks1-7 immutable at inference",
        "candidate_pool": "unchanged WV2-601 pool; inference compares ranks13-50 with ranks8-12",
        "input_gate": "unique user-item-window rows, at least 5000 positives, positive rate below 10 percent",
        "inner_gate": {
            "standard_vs_WV2_601": "mean>=+0.0001, >=3 positive, worst>=-0.0005",
            "incremental_vs_WV3_610": "mean>0, >=3 nondegrade, worst>=-0.0002",
        },
        "expected_minutes": 15,
        "outer_policy": "one frozen exposure only if input and both inner gates pass; no class-weight, threshold, feature or tree rescue",
        "fallback": "retain WV3-610 and close pointwise propensity family",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(15)
    registry = read(common.REGISTRY)
    stopped = next(row for row in registry["trials"] if row["experiment_id"] == "WV3-630")
    assert stopped["decision"] == "reject_input_audit" and stopped["outer_exposures"] == 0
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "full_top50_candidate_purchase_propensity_tail_inference",
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
    standard = result["standard_screening_vs_WV2_601"]
    incremental = result["incremental_screening_vs_WV3_610"]
    lines = [
        "# WV3-631 全 Top50 历史监督、尾部购买倾向准入",
        "",
        "## 结论",
        "",
        f"内层总门槛：**{'pass' if result['passed'] else 'fail'}**。相对 WV2-601 平均增加 {standard['mean_population_delta']:+.9f}；相对 WV3-610 平均增加 {incremental['mean_delta_vs_WV3_610']:+.9f}。",
        "",
        "## 术语与特殊处理",
        "",
        "- 全 Top50 历史监督（本项目自定义）：训练标签取基线第1–50名中的所有活跃用户—商品；目的是补充 WV3-630 在第8–50名中不足5,000条的正例。它只扩大历史训练支持，不扩大推理候选池。",
        "- 尾部推理（本项目自定义）：实际决策仍只比较第13–50名挑战商品与第8–12名被替换商品，第1–7名完全冻结，每位用户最多换一次。",
        "- 候选商品级购买倾向（行业常见 pointwise propensity 思路）：一个用户—商品—窗口一行，预测下一周是否购买；没有类别重加权，避免扭曲用于商品间作差的概率尺度。",
        "- 边际概率差换位（本项目自定义）：挑战商品概率减被替换商品概率，再乘换位位置对应的 AP@12 权重；只选期望值大于0的一次最佳换位。",
        "- population delta（本项目沿用指标）：用窗口全部真值用户作分母，实验 MAP@12 减 WV2-601 MAP@12。",
        "",
        f"训练行数 {result['training_audit']['rows']:,}，正例 {result['training_audit']['positive_rows']:,}，正例率 {result['training_audit']['positive_rate']:.3%}。最新历史标签结束日 2019-11-27，距最早内层截止日 28 天。",
        "",
        "## 四窗内层结果",
        "",
        "| 窗口 | 选择用户数 | 有益 / 有害 / 中性 | 中性占比 | 相对 WV2-601 | 相对 WV3-610 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(f"| {name} | {row['selected_users']:,} | {row['beneficial_selected_users']:,} / {row['harmful_selected_users']:,} / {row['neutral_selected_users']:,} | {row['selected_neutral_share']:.2%} | {row['population_delta']:+.9f} | {incremental['per_window_delta_vs_WV3_610'][name]:+.9f} |")
    lines += [
        "",
        "只有两个内层门槛同时通过才允许冻结并读取一次外层；失败后关闭点式购买倾向方案族，不调整类别权重、阈值、特征或树参数。最终周 `2020-09-16` 保持 `not_run`。",
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
        raise RuntimeError("WV3-631 full-Top50 input gate failed")
    model = fit_model(training)
    del training
    gc.collect()
    model_path = MODEL_ROOT / "MODEL.txt"
    model.save_model(str(model_path))
    write(MODEL_ROOT / "MODEL.json", {"experiment_id": TRIAL, "model": evidence_id(model_path, reason="explicit_registry_evidence"), "features": FEATURES, "params": PARAMS, "rounds": ROUNDS, "training_audit": str(MODEL_ROOT / "TRAINING_AUDIT.json"), "outer_labels_used": False, "final_week": "not_run"})
    windows = {}
    for name, folder in INNER:
        row, chosen = evaluate(folder, name, model, "full_top50_historical_propensity_inner_screen")
        windows[name] = row
        root = MODEL_ROOT / name / "inner"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"fulltop_propensity_inner": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = screening_gate(row["population_delta"] for row in windows.values())
    incremental = incremental_gate(windows)
    passed = standard["passed"] and incremental["passed"]
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "full_top50_candidate_purchase_propensity_tail_inference", "training_audit": audit, "windows": windows, "standard_screening_vs_WV2_601": standard, "incremental_screening_vs_WV3_610": incremental, "passed": passed, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render_screen(result), encoding="utf-8")
    common.update(TRIAL, decision="inner_pass" if passed else "reject_inner", inner_evidence={"input": audit, "standard": standard, "incremental": incremental, "passed": passed}, runtime=result["runtime_seconds"], artifact_paths=[str(SCREEN_REPORT), str(SCREEN_MARKDOWN), str(MODEL_ROOT / "MODEL.json")])
    common.log(
        f"{TRIAL} inner screen",
        "WV3-630 stopped before training because rank8-50 history contained 4490 positives, below its preregistered 5000-positive input gate.",
        experiment_contract()["hypothesis"],
        f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",
        f"input pass={audit['passed']}; positives={audit['positive_rows']}; standard mean={standard['mean_population_delta']:+.9f}; incremental mean={incremental['mean_delta_vs_WV3_610']:+.9f}; pass={passed}.",
        "Freeze and expose once." if passed else "Reject and close the pointwise propensity family without parameter rescue.",
        "Run one outer confirmation if both gates pass; otherwise retain WV3-610 and move to a different architecture family.",
        alternatives="This adds historical Top1-7 supervision but leaves inference Top1-7 immutable; it does not lower WV3-630's input gate.",
        experiment="Fit one unweighted candidate-level binary model on historical ranks1-50, then make one tail-only expected-value swap.",
        reflection="The train-support expansion is fixed before model outcomes and directly addresses the measured positive-count deficit.",
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
        row, chosen = evaluate(folder, name, model, "frozen_full_top50_propensity_outer")
        windows[name] = row
        root = MODEL_ROOT / name / "outer"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"fulltop_propensity_outer": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    prior = read(common.REPORT / "WV3-610_OUTER.json")
    delta = {name: windows[name]["MAP@12"] - prior["per_window_MAP"][name] for name in windows}
    values = list(delta.values())
    incremental = {"per_window_delta_vs_WV3_610": delta, "mean_delta_vs_WV3_610": float(np.mean(values)), "nondegrade_windows_vs_WV3_610": sum(value >= 0 for value in values), "worst_delta_vs_WV3_610": min(values)}
    better = standard["stable"] and incremental["mean_delta_vs_WV3_610"] > 0
    result = {"created_at": now(), "experiment_id": TRIAL, "architecture": "full_top50_candidate_purchase_propensity_tail_inference", "windows": windows, **standard, "incremental_vs_WV3_610": incremental, "better_than_highest_mean_candidate": better, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(OUTER_REPORT, result)
    common.update(TRIAL, decision="promote_candidate" if better else "reject_outer", outer_MAP_by_window=standard["per_window_MAP"], mean_MAP=standard["mean_MAP"], delta_vs_WV2_601=standard["delta_vs_WV2_601"], nondegrade_windows=standard["nondegrade_windows"], worst_delta=standard["worst_delta"], runtime=result["runtime_seconds"], artifact_paths=entry["artifact_paths"] + [str(OUTER_REPORT)])
    common.log(
        f"{TRIAL} outer closure",
        "The full-Top50 historical propensity architecture passed its input and both inner gates before freezing.",
        experiment_contract()["hypothesis"],
        str(OUTER_REPORT),
        f"mean MAP={standard['mean_MAP']:.9f}; delta vs WV2-601={standard['delta_vs_WV2_601']:+.9f}; incremental mean vs WV3-610={incremental['mean_delta_vs_WV3_610']:+.9f}; stable={standard['stable']}; better={better}.",
        "Promote as highest-mean stable candidate." if better else "Reject and retain WV3-610.",
        "Continue only through a different preregistered architecture family if time remains.",
        alternatives="No outer-driven class weight, threshold, feature or tree rescue is allowed.",
        experiment="One outer exposure of the exact frozen full-Top50 historical model with tail-only inference.",
        reflection="The check measures transfer of expanded historical support without changing deployment actions.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "better": better}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "screen", "confirm"])
    globals()[parser.parse_args().command]()
