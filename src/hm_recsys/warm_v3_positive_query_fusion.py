"""WV3-800/801: positive-bearing-query pointwise view and fixed three-model fusion."""
from __future__ import annotations

import argparse
import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import evidence_id, now, read, write
from .warm_v2_engine import save_parquet
from .warm_v3_clean_model_replay import LAMBDA_MODEL, POINT_MODEL, summarize_policy
from .warm_v3_expert import screening_gate
from .warm_v3_label_free_repair import choose_label_free_swaps, unlabeled_proposals
from .warm_v3_residual_admission import INNER, OUTER, pair_frame, source
from .warm_v3_rich_propensity import FEATURES, PARAMS, ROUNDS, historical_training_data, rich_candidate_frame


AUDIT_TRIAL = "WV3-800"
TRIAL = "WV3-801"
AUDIT_CONTRACT = common.REPORT / "WV3-800_POSITIVE_QUERY_AUDIT_CONTRACT.json"
AUDIT_REPORT = common.REPORT / "WV3-800_POSITIVE_QUERY_AUDIT.json"
AUDIT_MARKDOWN = common.REPORT / "WV3-800_POSITIVE_QUERY_AUDIT.md"
CONTRACT = common.REPORT / "WV3-801_POSITIVE_QUERY_FUSION_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-801_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-801_SCREEN.md"
OUTER_REPORT = common.REPORT / "WV3-801_OUTER.json"
FINAL_MARKDOWN = common.REPORT / "WV3-801_FINAL.md"
ART_ROOT = common.ART / TRIAL
RRF_CONSTANT = 60


def audit_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": AUDIT_TRIAL,
        "role": "strict_history_positive_bearing_query_supervision_audit",
        "hypothesis": (
            "The rich pointwise model is diluted by historical user-window groups with no Top50 positives; "
            "positive-bearing groups may support a complementary within-user propensity view."
        ),
        "group_definition": "one strict 2019 source window and one active Warm customer, with exactly its frozen Top50 candidates",
        "eligible_group": "group target sum is at least one; retain all candidates from that group, not only positive rows",
        "gates": {"eligible_groups_min": 5000, "positive_rows_min": 7000, "negative_rows_min": 250000, "all_groups_have_positive_and_negative": True, "target_excluded_from_features": True},
        "training": "none in audit",
        "outer": "not_run",
        "expected_minutes": 10,
        "success_action": "register one fixed positive-bearing-query model and equal three-view RRF60",
        "failure_action": "do not train WV3-801; retain WV3-741",
        "final_week": "2020-09-16 not_run",
    }


def register_audit() -> dict:
    common.setup()
    common.budget(10)
    if not AUDIT_CONTRACT.exists():
        write(AUDIT_CONTRACT, audit_contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == AUDIT_TRIAL for row in state["trials"]):
        common.register(AUDIT_TRIAL, "positive_bearing_query_supervision_audit", audit_contract()["hypothesis"], candidate_protocol="unchanged historical Top50", training_protocol="none", features=FEATURES, params={"group": audit_contract()["group_definition"], "gates": audit_contract()["gates"], "final_week": "not_run"}, expected_minutes=10)
    return read(AUDIT_CONTRACT)


def select_positive_groups(training: pd.DataFrame) -> pd.DataFrame:
    group_sum = training.groupby(["source_window", "customer_id"], sort=False).target.transform("sum")
    return training.loc[group_sum > 0].copy()


def audit() -> dict:
    common.setup()
    common.budget(10)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == AUDIT_TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-800 must start preregistered and unexposed")
    started = time.perf_counter()
    training, source_audit = historical_training_data()
    eligible = select_positive_groups(training)
    groups = eligible.groupby(["source_window", "customer_id"], sort=False).target.agg(["sum", "count"])
    positive_rows = int(eligible.target.sum())
    negative_rows = len(eligible) - positive_rows
    all_mixed = bool((groups["sum"] > 0).all() and (groups["sum"] < groups["count"]).all())
    gates = audit_contract()["gates"]
    passed = bool(len(groups) >= gates["eligible_groups_min"] and positive_rows >= gates["positive_rows_min"] and negative_rows >= gates["negative_rows_min"] and all_mixed and "target" not in FEATURES)
    result = {
        "created_at": now(),
        "experiment_id": AUDIT_TRIAL,
        "all_training_rows": len(training),
        "all_user_window_groups": int(training.groupby(["source_window", "customer_id"], sort=False).ngroups),
        "eligible_rows": len(eligible),
        "eligible_groups": len(groups),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "all_eligible_groups_have_positive_and_negative": all_mixed,
        "features": FEATURES,
        "target_excluded": "target" not in FEATURES,
        "source_audit": source_audit,
        "training_authorized": passed,
        "runtime_seconds": time.perf_counter() - started,
        "outer_exposure": 0,
        "final_week": "not_run",
    }
    write(AUDIT_REPORT, result)
    AUDIT_MARKDOWN.write_text("\n".join([
        "# WV3-800：含正例用户组监督审计",
        "",
        "## 结论",
        "",
        f"WV3-801 训练授权：**{'通过' if passed else '未通过'}**。本轮没有训练模型，也没有读取2020内层或外层。",
        "",
        "## 术语",
        "",
        "- 用户—窗口组（学习排序常用分组）：同一个历史截止窗口内，同一活跃 Warm 用户的冻结 Top50 候选；统计单位是一名用户在一个窗口的一组50件商品。",
        "- 含正例组（本项目自定义）：该组至少一件候选在后续7天被用户购买；保留组内全部正负候选，避免把问题偷换成只训练正例。",
        "- 全负组（分类训练常用描述）：Top50 中没有后续购买商品的用户—窗口组；它能提供总体负反馈，但不能提供该用户的候选内部正负区分。",
        "- 补充模型视角（本项目自定义）：新模型与原点式模型同为99维 LightGBM，但训练总体不同；只作为第三个排序信号，不替代原模型。",
        "",
        f"原训练 `{len(training):,}` 行、`{result['all_user_window_groups']:,}` 组；含正例监督 `{len(eligible):,}` 行、`{len(groups):,}` 组，其中正例 `{positive_rows:,}`、负例 `{negative_rows:,}`。",
        "",
        "监督仅来自四个严格2019窗口；`target` 只用于筛选历史训练组和模型标签，不在输入特征中。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]), encoding="utf-8")
    common.update(AUDIT_TRIAL, decision="diagnostic_supports_positive_query_view" if passed else "diagnostic_rejects_positive_query_view", inner_evidence={"eligible_groups": len(groups), "positive_rows": positive_rows, "negative_rows": negative_rows, "authorized": passed}, runtime=result["runtime_seconds"], artifact_paths=[str(AUDIT_CONTRACT), str(AUDIT_REPORT), str(AUDIT_MARKDOWN)], validity="valid_strict_historical_supervision_audit")
    return result


def model_contract() -> dict:
    if not read(AUDIT_REPORT)["training_authorized"]:
        raise AssertionError("WV3-800 did not authorize training")
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-741",
        "architecture_family": "positive_bearing_query_pointwise_plus_fixed_three_view_rank_fusion",
        "hypothesis": (
            "A rich pointwise model trained only on historically informative user groups will add complementary personalized ordering to the frozen all-group pointwise and LambdaRank views."
        ),
        "training": "strict 2019 Top50 groups with at least one positive; all 50 rows retained; unweighted binary LightGBM",
        "model": {"params": PARAMS, "rounds": ROUNDS},
        "fusion": "equal sum of 1/(60+within-user rank) for frozen WV3-661, frozen WV3-680, and the new positive-bearing-query model",
        "action": "same label-free maximum two disjoint swaps as WV3-741; challengers13-50, victims8-12, ranks1-7 protected",
        "candidate_pool": "unchanged WV3-741 Top50",
        "inner_gate": "mean incremental vs WV3-741 >0, >=3/4 nondegrade, worst>=-0.0002; standard WV2 screen also passes",
        "outer_policy": "one frozen exposure only if inner passes; no class weighting, threshold, tree, RRF weight or action rescue",
        "expected_minutes": 25,
        "fallback": "retain WV3-741 and close positive-bearing-query family",
        "final_week": "2020-09-16 not_run",
    }


def register_model() -> dict:
    common.setup()
    common.budget(25)
    if not CONTRACT.exists():
        write(CONTRACT, model_contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(TRIAL, model_contract()["architecture_family"], model_contract()["hypothesis"], candidate_protocol=model_contract()["candidate_pool"], training_protocol=model_contract()["training"], features=FEATURES, params={"model": model_contract()["model"], "fusion": model_contract()["fusion"], "action": model_contract()["action"], "inner_gate": model_contract()["inner_gate"], "final_week": "not_run"}, expected_minutes=25)
    return read(CONTRACT)


def fit_model(training: pd.DataFrame) -> lgb.Booster:
    labels = training.target.to_numpy(np.uint8)
    dataset = lgb.Dataset(training[FEATURES].to_numpy(np.float32), label=labels, feature_name=FEATURES, free_raw_data=True)
    return lgb.train(PARAMS, dataset, num_boost_round=ROUNDS)


def rank_view(frame: pd.DataFrame, values: np.ndarray, column: str) -> pd.DataFrame:
    result = frame[["customer_id", "article_id"]].copy()
    result[column] = np.asarray(values, dtype=np.float64)
    result = result.sort_values(["customer_id", column, "article_id"], ascending=[True, False, True], kind="mergesort")
    result[f"{column}_rank"] = result.groupby("customer_id", sort=False).cumcount() + 1
    return result[["customer_id", "article_id", f"{column}_rank"]]


def three_view_scores(candidates: pd.DataFrame, point_values: np.ndarray, lambda_values: np.ndarray, query_values: np.ndarray) -> pd.DataFrame:
    merged = rank_view(candidates, point_values, "point").merge(rank_view(candidates, lambda_values, "lambda"), on=["customer_id", "article_id"], validate="one_to_one").merge(rank_view(candidates, query_values, "query"), on=["customer_id", "article_id"], validate="one_to_one")
    merged["ranking_score"] = 1.0 / (RRF_CONSTANT + merged.point_rank) + 1.0 / (RRF_CONSTANT + merged.lambda_rank) + 1.0 / (RRF_CONSTANT + merged.query_rank)
    return merged[["customer_id", "article_id", "ranking_score"]]


def evaluate(name: str, folder: str, role: str, query_model: lgb.Booster, point: lgb.Booster, lambdarank: lgb.Booster) -> dict:
    path, meta = source(folder)
    candidates, details = rich_candidate_frame(meta["cutoff"], rank_min=1)
    matrix = candidates[FEATURES].to_numpy(np.float32)
    scores = three_view_scores(candidates, point.predict(matrix, num_threads=4), lambdarank.predict(matrix, num_threads=4), query_model.predict(matrix, num_threads=4))
    pairs = pair_frame(path, discordant_only=False)
    swaps = choose_label_free_swaps(unlabeled_proposals(pairs), scores)
    summary, users = summarize_policy(candidates, swaps, meta)
    if role == "inner":
        control = read(common.REPORT / "WV3-740_CLEAN_MODEL_REPLAY.json")["windows"][name]["policies"]["pointwise_lambdarank_RRF60"]
    else:
        control = read(common.REPORT / "WV3-741_OUTER.json")["windows"][name]
    summary.update({"window": name, "cutoff": meta["cutoff"], "total_users_denominator": int(meta["total_users"]), "same_pool_control_MAP@12": control["MAP@12"], "incremental_delta_vs_WV3_741": summary["MAP@12"] - control["MAP@12"], "decision_uses_target_or_unit_gain": False, "rich_candidate_input": details, "final_week": "not_run"})
    root = ART_ROOT / name / role
    root.mkdir(parents=True, exist_ok=True)
    save_parquet(swaps, root / "swaps.parquet")
    save_parquet(users, root / "users.parquet")
    write(root / "REVIEW.json", summary)
    return summary


def gate(windows: dict) -> tuple[dict, dict, bool]:
    increments = {name: row["incremental_delta_vs_WV3_741"] for name, row in windows.items()}
    values = list(increments.values())
    incremental = {"per_window_delta_vs_WV3_741": increments, "mean_delta_vs_WV3_741": float(np.mean(values)), "nondegrade_windows_vs_WV3_741": sum(value >= 0 for value in values), "worst_delta_vs_WV3_741": min(values)}
    incremental["passed"] = bool(incremental["mean_delta_vs_WV3_741"] > 0 and incremental["nondegrade_windows_vs_WV3_741"] >= 3 and incremental["worst_delta_vs_WV3_741"] >= -0.0002)
    standard = screening_gate(row["MAP@12"] - row["baseline_MAP@12"] for row in windows.values())
    return incremental, standard, bool(incremental["passed"] and standard["passed"])


def render(result: dict, outer: bool) -> str:
    phase = "外层确认" if outer else "内层筛选"
    lines = [
        f"# WV3-801：含正例组点式模型的三路名次融合（{phase}）",
        "",
        "## 结论",
        "",
        f"门槛：**{'通过' if result['passed'] else '未通过'}**；相对 WV3-741 平均 `{result['incremental_vs_WV3_741']['mean_delta_vs_WV3_741']:+.9f}`。",
        "",
        "## 术语",
        "",
        "- 含正例组点式模型（本项目自定义）：仅在历史 Top50 中至少有一个下一周购买正例的用户—窗口组上训练，但保留该组全部正负候选；输入仍为冻结99维特征。",
        "- 三路名次融合（推荐系统常见）：分别把原点式模型、LambdaRank 和含正例组点式模型的分数转为用户内名次，再等权相加 `1/(60+名次)`；60为冻结常数。",
        "- 无标签尾部换位（本项目自定义）：只按三路融合分数选择最多两组互不冲突换位，挑战商品来自第13–50名，被替换商品来自第8–12名，第1–7名不变。",
        "- 增量（本项目评测口径）：本轮 MAP@12 减同窗口 WV3-741 MAP@12，分母为窗口全部有真值用户。",
        "",
        f"| {phase}窗口 | 选择用户 | 有益/有害/中性 | 相对 WV3-741 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(f"| {name} | {row['selected_users']:,} | {row['beneficial_selected_users']:,}/{row['harmful_selected_users']:,}/{row['neutral_selected_users']:,} | {row['incremental_delta_vs_WV3_741']:+.9f} |")
    lines += ["", "失败后不调整样本、类别权重、模型参数、融合权重或换位数。最终周 `2020-09-16` 保持 `not_run`。", ""]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(25)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-801 must start preregistered and unexposed")
    started = time.perf_counter()
    training, _ = historical_training_data()
    eligible = select_positive_groups(training)
    model = fit_model(eligible)
    ART_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = ART_ROOT / "MODEL.txt"
    model.save_model(str(model_path))
    write(ART_ROOT / "MODEL.json", {"experiment_id": TRIAL, "model": evidence_id(model_path, reason="explicit_registry_evidence"), "features": FEATURES, "params": PARAMS, "rounds": ROUNDS, "training_rows": len(eligible), "training_positive_rows": int(eligible.target.sum()), "training_groups": int(eligible.groupby(["source_window", "customer_id"], sort=False).ngroups), "outer_labels_used": False, "final_week": "not_run"})
    del training, eligible
    gc.collect()
    point = lgb.Booster(model_file=str(POINT_MODEL))
    lambdarank = lgb.Booster(model_file=str(LAMBDA_MODEL))
    windows = {}
    for name, folder in INNER:
        windows[name] = evaluate(name, folder, "inner", model, point, lambdarank)
        print({"positive_query_fusion_inner": name, "incremental": windows[name]["incremental_delta_vs_WV3_741"]}, flush=True)
    incremental, standard, passed = gate(windows)
    result = {"created_at": now(), "experiment_id": TRIAL, "windows": windows, "incremental_vs_WV3_741": incremental, "standard_vs_WV2_601": standard, "passed": passed, "new_training": True, "candidate_pool_changed": False, "outer_exposure": 0, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render(result, outer=False), encoding="utf-8")
    common.update(TRIAL, decision="inner_pass" if passed else "reject_inner", inner_evidence={"incremental": incremental, "standard": standard, "passed": passed}, runtime=result["runtime_seconds"], artifact_paths=[str(CONTRACT), str(SCREEN_REPORT), str(SCREEN_MARKDOWN), str(ART_ROOT / "MODEL.json")], validity="valid_label_free_inner_screen")
    return result


def confirm() -> dict:
    common.setup()
    common.budget(15)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "inner_pass" or entry["outer_exposures"] != 1:
        raise AssertionError("WV3-801 requires inner pass and exactly one recorded outer exposure")
    model = lgb.Booster(model_file=str(ART_ROOT / "MODEL.txt"))
    point = lgb.Booster(model_file=str(POINT_MODEL))
    lambdarank = lgb.Booster(model_file=str(LAMBDA_MODEL))
    started = time.perf_counter()
    windows = {}
    for name, folder in OUTER:
        windows[name] = evaluate(name, folder, "outer", model, point, lambdarank)
        print({"positive_query_fusion_outer": name, "incremental": windows[name]["incremental_delta_vs_WV3_741"]}, flush=True)
    incremental, _, incremental_pass = gate(windows)
    standard = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    passed = bool(standard["stable"] and incremental_pass)
    result = {"created_at": now(), "experiment_id": TRIAL, "windows": windows, **standard, "incremental_vs_WV3_741": incremental, "passed": passed, "new_training": True, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(OUTER_REPORT, result)
    FINAL_MARKDOWN.write_text(render(result, outer=True), encoding="utf-8")
    common.update(TRIAL, decision="valid_development_champion" if passed else "reject_outer", outer_MAP_by_window=standard["per_window_MAP"], mean_MAP=standard["mean_MAP"], delta_vs_WV2_601=standard["delta_vs_WV2_601"], nondegrade_windows=standard["nondegrade_windows"], worst_delta=standard["worst_delta"], runtime=result["runtime_seconds"], artifact_paths=entry["artifact_paths"] + [str(OUTER_REPORT), str(FINAL_MARKDOWN)], validity="valid_label_free_outer_confirmation")
    if passed:
        state = read(common.REGISTRY)
        state["current_champion"] = TRIAL
        state["current_valid_champion"] = TRIAL
        state["best_stable"] = {"experiment_id": TRIAL, "mean_MAP": standard["mean_MAP"], "delta_vs_WV2_601": standard["delta_vs_WV2_601"], "nondegrade_windows": standard["nondegrade_windows"], "worst_delta": standard["worst_delta"]}
        state["status"] = "running_below_target"
        write(common.REGISTRY, state)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register-audit", "audit", "register-model", "screen", "confirm"])
    command = parser.parse_args().command
    {"register-audit": register_audit, "audit": audit, "register-model": register_model, "screen": screen, "confirm": confirm}[command]()
