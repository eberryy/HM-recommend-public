"""WV3-790/791: conservative harm-class veto over frozen WV3-741 actions."""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import load_parquet, save_parquet
from .warm_v3_clean_model_replay import summarize_policy
from .warm_v3_direct_relation import FEATURES, MODEL_ROOT as RELATION_ROOT, augment_rich_deltas
from .warm_v3_expert import screening_gate
from .warm_v3_residual_admission import INNER, OUTER, pair_frame, source
from .warm_v3_rich_propensity import rich_candidate_frame


AUDIT_TRIAL = "WV3-790"
TRIAL = "WV3-791"
AUDIT_CONTRACT = common.REPORT / "WV3-790_HARM_VETO_AUDIT_CONTRACT.json"
AUDIT_REPORT = common.REPORT / "WV3-790_HARM_VETO_AUDIT.json"
AUDIT_MARKDOWN = common.REPORT / "WV3-790_HARM_VETO_AUDIT.md"
CONTRACT = common.REPORT / "WV3-791_HARM_VETO_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-791_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-791_SCREEN.md"
OUTER_REPORT = common.REPORT / "WV3-791_OUTER.json"
FINAL_MARKDOWN = common.REPORT / "WV3-791_FINAL.md"
ART_ROOT = common.ART / TRIAL
PAIR_KEYS = ["customer_id", "challenger_article_id", "victim_article_id", "challenger_rank", "victim_rank"]


def audit_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": AUDIT_TRIAL,
        "role": "mechanical_reuse_audit_for_relation_model_as_harm_only_veto",
        "hypothesis": (
            "WV3-751 may be useful as a conservative harm detector even though letting it replace WV3-741 action ranking over-vetoed valid actions."
        ),
        "mechanism": (
            "preserve every WV3-741 selected swap and its order; remove only swaps whose frozen WV3-751 model argmax class is harm; "
            "never add or reorder a swap"
        ),
        "gates": {
            "frozen_model_exists": True,
            "all_WV3_741_actions_join_exactly_once": True,
            "veto_only_subset_invariant": True,
            "decision_fields_exclude_labels": True,
        },
        "training": "none",
        "inner_labels": "not read in audit",
        "outer": "not_run",
        "expected_minutes": 5,
        "success_action": "register exactly one hard harm-argmax veto screen",
        "failure_action": "do not run WV3-791; retain WV3-741",
        "final_week": "2020-09-16 not_run",
    }


def register_audit() -> dict:
    common.setup()
    common.budget(5)
    if not AUDIT_CONTRACT.exists():
        write(AUDIT_CONTRACT, audit_contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == AUDIT_TRIAL for row in state["trials"]):
        common.register(
            AUDIT_TRIAL,
            "frozen_relation_model_harm_only_veto_audit",
            audit_contract()["hypothesis"],
            candidate_protocol="unchanged WV3-741 Top50 and exact selected action set",
            training_protocol="none; reuse frozen WV3-751 model",
            features=FEATURES,
            params={"mechanism": audit_contract()["mechanism"], "gates": audit_contract()["gates"], "final_week": "not_run"},
            expected_minutes=5,
        )
    return read(AUDIT_CONTRACT)


def source_swaps(name: str, role: str) -> pd.DataFrame:
    if role == "inner":
        path = common.ART / "WV3-740" / name / "pointwise_lambdarank_RRF60" / "swaps.parquet"
    elif role == "outer":
        path = common.ART / "WV3-741" / name / "outer" / "swaps.parquet"
    else:
        raise ValueError(role)
    swaps = load_parquet(path)
    if swaps.duplicated(PAIR_KEYS).any():
        raise AssertionError("WV3-741 action identities are not unique")
    return swaps


def exact_action_pairs(name: str, folder: str, role: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path, meta = source(folder)
    actions = source_swaps(name, role)
    pairs = pair_frame(path, discordant_only=False)
    selected = actions[PAIR_KEYS + ["swap_order"]].merge(pairs, on=PAIR_KEYS, how="left", validate="one_to_one", indicator=True)
    if len(selected) != len(actions) or not selected._merge.eq("both").all():
        raise AssertionError("Not every frozen WV3-741 action joined to exactly one pair row")
    return actions, selected.drop(columns="_merge"), meta


def audit() -> dict:
    common.setup()
    common.budget(5)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == AUDIT_TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-790 must start preregistered and unexposed")
    model_path = RELATION_ROOT / "MODEL.txt"
    checks = []
    for name, folder in INNER:
        actions, selected, _ = exact_action_pairs(name, folder, "inner")
        checks.append({
            "window": name,
            "WV3_741_action_rows": len(actions),
            "joined_action_rows": len(selected),
            "identity_exact": len(actions) == len(selected),
            "maximum_swap_order": int(actions.swap_order.max()),
        })
    passed = bool(
        model_path.exists()
        and all(row["identity_exact"] and row["maximum_swap_order"] <= 2 for row in checks)
        and "relation_label" not in FEATURES
        and "challenger_target" not in FEATURES
        and "victim_target" not in FEATURES
    )
    result = {
        "created_at": now(),
        "experiment_id": AUDIT_TRIAL,
        "checks": checks,
        "frozen_model_path": str(model_path),
        "model_exists": model_path.exists(),
        "features": FEATURES,
        "training_authorized": passed,
        "new_training": False,
        "inner_labels_used_for_decision": False,
        "outer_exposure": 0,
        "final_week": "not_run",
    }
    write(AUDIT_REPORT, result)
    AUDIT_MARKDOWN.write_text(
        "\n".join([
            "# WV3-790：冻结关系模型的有害动作否决审计",
            "",
            "## 结论",
            "",
            f"WV3-791 内层测试授权：**{'通过' if passed else '未通过'}**。本轮没有训练模型、没有读取内层结果标签，也没有读取外层。",
            "",
            "## 术语",
            "",
            "- 有害动作否决器（本项目自定义）：保持 WV3-741 已选换位及次序不变，只删除冻结三分类模型预测类别为‘有害’的换位；它不能新增、改选或重排商品。",
            "- argmax 类别（机器学习通用）：三类预测概率中数值最大者对应的类别；本轮只有‘有害’为最大类时才否决，不搜索概率阈值。",
            "- 子集不变量（软件审计概念）：WV3-791 的动作必须是 WV3-741 动作的严格子集，用户、挑战商品、被替换商品和原顺序均不可改变。",
            "- 精确连接（数据审计概念）：每条 WV3-741 动作必须且只能对应一条同窗口候选商品对特征记录。",
            "",
            "| 内层窗口 | WV3-741 动作行 | 精确连接行 | 最大换位序号 |",
            "|---|---:|---:|---:|",
            *[f"| {row['window']} | {row['WV3_741_action_rows']:,} | {row['joined_action_rows']:,} | {row['maximum_swap_order']} |" for row in checks],
            "",
            "模型输入不包含验证标签；正式内层阶段只能执行这个固定硬否决规则。最终周 `2020-09-16` 保持 `not_run`。",
            "",
        ]),
        encoding="utf-8",
    )
    common.update(
        AUDIT_TRIAL,
        decision="diagnostic_supports_harm_veto" if passed else "diagnostic_rejects_harm_veto",
        inner_evidence={"mechanics_passed": passed, "checks": checks},
        artifact_paths=[str(AUDIT_CONTRACT), str(AUDIT_REPORT), str(AUDIT_MARKDOWN)],
        validity="valid_label_free_mechanical_audit",
    )
    return result


def model_contract() -> dict:
    if not read(AUDIT_REPORT)["training_authorized"]:
        raise AssertionError("WV3-790 did not authorize the veto screen")
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-741",
        "architecture_family": "frozen_direct_relation_harm_only_veto_over_WV3_741",
        "hypothesis": (
            "Using the rejected relation model only to remove actions it classifies as harm can reduce WV3-741's residual harmful swaps without surrendering action selection to that model."
        ),
        "model": "exact frozen WV3-751 three-class LightGBM; no training",
        "decision": "start from exact WV3-741 swaps; veto if argmax[P(harm),P(neutral),P(benefit)] is harm; retain original swap order",
        "candidate_pool": "unchanged Top50; exact WV3-741 action universe",
        "inner_gate": "mean incremental vs WV3-741 >0, >=3/4 nondegrade, worst>=-0.0001; standard WV2 screen also passes",
        "outer_policy": "one frozen exposure only if inner passes; no probability threshold or class-weight rescue",
        "expected_minutes": 10,
        "fallback": "retain WV3-741 and close relation-model reuse",
        "final_week": "2020-09-16 not_run",
    }


def register_model() -> dict:
    common.setup()
    common.budget(10)
    if not CONTRACT.exists():
        write(CONTRACT, model_contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(
            TRIAL,
            model_contract()["architecture_family"],
            model_contract()["hypothesis"],
            candidate_protocol=model_contract()["candidate_pool"],
            training_protocol=model_contract()["model"],
            features=FEATURES,
            params={"decision": model_contract()["decision"], "inner_gate": model_contract()["inner_gate"], "final_week": "not_run"},
            expected_minutes=10,
        )
    return read(CONTRACT)


def evaluate(name: str, folder: str, role: str, model: lgb.Booster) -> dict:
    actions, selected_pairs, meta = exact_action_pairs(name, folder, role)
    candidates, details = rich_candidate_frame(meta["cutoff"], rank_min=1)
    rich_pairs = augment_rich_deltas(selected_pairs, candidates)
    probabilities = model.predict(rich_pairs[FEATURES].to_numpy(np.float32), num_threads=4)
    prediction = probabilities.argmax(axis=1)
    decisions = actions.copy()
    decisions["predicted_relation_class"] = prediction
    decisions["harm_probability"] = probabilities[:, 0]
    decisions["neutral_probability"] = probabilities[:, 1]
    decisions["benefit_probability"] = probabilities[:, 2]
    retained = decisions.loc[decisions.predicted_relation_class != 0].copy()
    if not retained[PAIR_KEYS].merge(actions[PAIR_KEYS], on=PAIR_KEYS, how="left", indicator=True)._merge.eq("both").all():
        raise AssertionError("Veto output is not a subset of WV3-741 actions")
    summary, users = summarize_policy(candidates, retained, meta)
    if role == "inner":
        control = read(common.REPORT / "WV3-740_CLEAN_MODEL_REPLAY.json")["windows"][name]["policies"]["pointwise_lambdarank_RRF60"]
    else:
        control = read(common.REPORT / "WV3-741_OUTER.json")["windows"][name]
    summary.update({
        "window": name,
        "cutoff": meta["cutoff"],
        "total_users_denominator": int(meta["total_users"]),
        "same_pool_control_MAP@12": control["MAP@12"],
        "incremental_delta_vs_WV3_741": summary["MAP@12"] - control["MAP@12"],
        "WV3_741_action_rows": len(actions),
        "vetoed_action_rows": int((prediction == 0).sum()),
        "retained_action_rows": len(retained),
        "subset_invariant": True,
        "decision_uses_target_or_unit_gain": False,
        "rich_candidate_input": details,
        "final_week": "not_run",
    })
    root = ART_ROOT / name / role
    root.mkdir(parents=True, exist_ok=True)
    save_parquet(decisions, root / "scored_actions.parquet")
    save_parquet(retained, root / "retained_swaps.parquet")
    save_parquet(users, root / "users.parquet")
    write(root / "REVIEW.json", summary)
    return summary


def gate(windows: dict) -> tuple[dict, dict, bool]:
    increments = {name: row["incremental_delta_vs_WV3_741"] for name, row in windows.items()}
    incremental = {
        "per_window_delta_vs_WV3_741": increments,
        "mean_delta_vs_WV3_741": float(np.mean(list(increments.values()))),
        "nondegrade_windows_vs_WV3_741": sum(value >= 0 for value in increments.values()),
        "worst_delta_vs_WV3_741": min(increments.values()),
    }
    incremental["passed"] = bool(incremental["mean_delta_vs_WV3_741"] > 0 and incremental["nondegrade_windows_vs_WV3_741"] >= 3 and incremental["worst_delta_vs_WV3_741"] >= -0.0001)
    standard = screening_gate(row["MAP@12"] - row["baseline_MAP@12"] for row in windows.values())
    return incremental, standard, bool(incremental["passed"] and standard["passed"])


def render(result: dict, outer: bool) -> str:
    phase = "外层确认" if outer else "内层筛选"
    lines = [
        f"# WV3-791：三分类关系模型的保守有害动作否决（{phase}）",
        "",
        "## 结论",
        "",
        f"门槛：**{'通过' if result['passed'] else '未通过'}**；相对 WV3-741 平均 `{result['incremental_vs_WV3_741']['mean_delta_vs_WV3_741']:+.9f}`。",
        "",
        "## 术语",
        "",
        "- 保守有害动作否决（本项目自定义）：WV3-741 仍负责选挑战商品、被替换商品和换位顺序；冻结三分类模型只在‘有害’为最高概率类别时删除该动作。",
        "- 三分类关系模型（本项目自定义）：WV3-751 在严格历史数据上学习有害、中性、有益三类；本轮不重训，也不搜索概率阈值。",
        "- 动作子集不变量（本项目评测约束）：本轮只能删除 WV3-741 动作，不能新增、替换或改变剩余动作顺序；关闭否决应精确回到 WV3-741。",
        "- 增量（本项目评测口径）：本轮 MAP@12 减同窗口 WV3-741 MAP@12；分母保持窗口全部有真值用户。",
        "",
        f"| {phase}窗口 | 原动作行 | 否决行 | 保留行 | 有益/有害/中性用户 | 相对 WV3-741 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['WV3_741_action_rows']:,} | {row['vetoed_action_rows']:,} | {row['retained_action_rows']:,} | "
            f"{row['beneficial_selected_users']:,}/{row['harmful_selected_users']:,}/{row['neutral_selected_users']:,} | "
            f"{row['incremental_delta_vs_WV3_741']:+.9f} |"
        )
    lines += [
        "",
        "失败后不调整类别权重、否决阈值或模型参数。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(10)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-791 must start preregistered and unexposed")
    model = lgb.Booster(model_file=str(RELATION_ROOT / "MODEL.txt"))
    started = time.perf_counter()
    windows = {}
    for name, folder in INNER:
        windows[name] = evaluate(name, folder, "inner", model)
        print({"harm_veto_inner": name, "incremental": windows[name]["incremental_delta_vs_WV3_741"], "vetoed": windows[name]["vetoed_action_rows"]}, flush=True)
    incremental, standard, passed = gate(windows)
    result = {"created_at": now(), "experiment_id": TRIAL, "windows": windows, "incremental_vs_WV3_741": incremental, "standard_vs_WV2_601": standard, "passed": passed, "new_training": False, "candidate_pool_changed": False, "outer_exposure": 0, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render(result, outer=False), encoding="utf-8")
    common.update(TRIAL, decision="inner_pass" if passed else "reject_inner", inner_evidence={"incremental": incremental, "standard": standard, "passed": passed}, runtime=result["runtime_seconds"], artifact_paths=[str(CONTRACT), str(SCREEN_REPORT), str(SCREEN_MARKDOWN)], validity="valid_label_free_inner_screen")
    return result


def confirm() -> dict:
    common.setup()
    common.budget(10)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "inner_pass" or entry["outer_exposures"] != 1:
        raise AssertionError("WV3-791 requires inner pass and exactly one recorded outer exposure")
    model = lgb.Booster(model_file=str(RELATION_ROOT / "MODEL.txt"))
    started = time.perf_counter()
    windows = {}
    for name, folder in OUTER:
        windows[name] = evaluate(name, folder, "outer", model)
        print({"harm_veto_outer": name, "incremental": windows[name]["incremental_delta_vs_WV3_741"], "vetoed": windows[name]["vetoed_action_rows"]}, flush=True)
    incremental, _, incremental_pass = gate(windows)
    standard = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    passed = bool(standard["stable"] and incremental_pass)
    result = {"created_at": now(), "experiment_id": TRIAL, "windows": windows, **standard, "incremental_vs_WV3_741": incremental, "passed": passed, "new_training": False, "candidate_pool_changed": False, "runtime_seconds": time.perf_counter() - started, "final_week": "not_run"}
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
