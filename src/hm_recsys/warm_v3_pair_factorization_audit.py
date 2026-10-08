"""WV3-621: read-only audit of neutral-pair inflation and residual decisions."""
from __future__ import annotations

import numpy as np

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import load_parquet


TRIAL = "WV3-621"
CONTRACT = common.REPORT / "WV3-621_PAIR_FACTORIZATION_AUDIT_CONTRACT.json"
OUTPUT = common.REPORT / "WV3-621_PAIR_FACTORIZATION_AUDIT.json"
MARKDOWN = common.REPORT / "WV3-621_PAIR_FACTORIZATION_AUDIT.md"
MODELS = ["WV3-601", "WV3-610", "WV3-620"]
WINDOWS = [
    "winter_20200122",
    "spring_20200318",
    "early_summer_20200624",
    "late_summer_20200819",
]


def experiment_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "role": "read_only_pair_factorization_diagnostic",
        "hypothesis": "Residual admission is limited by neutral pair inflation: pairwise heads repeatedly score combinations in which neither item is bought, while candidate-level purchase supervision would represent each item once and make the exact expected swap value a probability difference.",
        "inputs": [
            "completed WV3-601/WV3-610/WV3-620 inner decision files",
            "completed WV3-620 historical input audit",
            "completed WV3-600 one-swap oracle audit",
        ],
        "statistics": [
            "historical actionable-pair rate",
            "selected neutral share and beneficial-user capture",
            "realized one-swap oracle ratio",
            "decision identity overlap across models",
            "actual-delta change when WV3-620 and WV3-610 disagree",
        ],
        "decision_gate": {
            "historical_actionable_pair_rate_max": 0.02,
            "WV3_610_mean_selected_neutral_share_min": 0.90,
            "WV3_610_mean_beneficial_capture_max": 0.10,
            "WV3_620_incremental_gate_must_fail": True,
        },
        "authorized_followup_if_pass": "one preregistered candidate-level purchase-propensity model; convert challenger-minus-victim probabilities into exact one-swap expected AP gain",
        "training": "none",
        "new_outer_exposure": 0,
        "fallback": "If the gate fails, retain WV3-610 and stop changing label factorization.",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(5)
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "pair_factorization_failure_diagnostic",
            experiment_contract()["hypothesis"],
            candidate_protocol="unchanged; completed inner decision evidence only",
            training_protocol="read-only; no training and no new outer labels",
            params={"decision_gate": experiment_contract()["decision_gate"], "final_week": "not_run"},
            expected_minutes=5,
        )


def _decision_frame(model: str, window: str):
    return load_parquet(common.ART / model / window / "inner" / "decisions.parquet")


def run() -> dict:
    common.setup()
    common.budget(5)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    historical = read(common.REPORT / "WV3-620_TARGET_AWARE_INPUT_AUDIT.json")
    oracle = read(common.REPORT / "WV3-600_RESIDUAL_ADMISSION_AUDIT.json")
    oracle_inner = {row["window"]: row for row in oracle["windows"] if row["role"] == "inner_screen"}
    screens = {model: read(common.REPORT / f"{model}_SCREEN.json") for model in MODELS}
    per_model: dict[str, dict] = {}
    for model in MODELS:
        rows = {}
        for window in WINDOWS:
            result = screens[model]["windows"][window]
            reference = oracle_inner[window]
            selected_users = result["selected_users"]
            neutral_share = result.get(
                "selected_neutral_share",
                result["neutral_selected_users"] / selected_users if selected_users else 0.0,
            )
            rows[window] = {
                "selected_users": selected_users,
                "selected_neutral_share": neutral_share,
                "beneficial_selected_users": result["beneficial_selected_users"],
                "harmful_selected_users": result["harmful_selected_users"],
                "beneficial_user_capture": result["beneficial_selected_users"] / reference["beneficial_users"],
                "realized_oracle_ratio": result["population_delta"] / reference["one_swap_oracle_delta"],
                "population_delta": result["population_delta"],
            }
        per_model[model] = {
            "windows": rows,
            "mean_selected_neutral_share": float(np.mean([x["selected_neutral_share"] for x in rows.values()])),
            "mean_beneficial_user_capture": float(np.mean([x["beneficial_user_capture"] for x in rows.values()])),
            "mean_realized_oracle_ratio": float(np.mean([x["realized_oracle_ratio"] for x in rows.values()])),
        }
    disagreement = {}
    for window in WINDOWS:
        left = _decision_frame("WV3-610", window).set_index("customer_id")
        right = _decision_frame("WV3-620", window).set_index("customer_id")
        ids = left.index.union(right.index)
        left_delta = left.actual_delta.reindex(ids).fillna(0.0)
        right_delta = right.actual_delta.reindex(ids).fillna(0.0)
        same_pair = (
            left.challenger_article_id.reindex(ids).fillna("").eq(right.challenger_article_id.reindex(ids).fillna(""))
            & left.victim_article_id.reindex(ids).fillna("").eq(right.victim_article_id.reindex(ids).fillna(""))
        )
        changed = ~same_pair
        diff = right_delta - left_delta
        disagreement[window] = {
            "decision_union_users": len(ids),
            "same_pair_users": int(same_pair.sum()),
            "different_or_one_sided_users": int(changed.sum()),
            "different_or_one_sided_share": float(changed.mean()),
            "WV3_620_better_changed_users": int(((diff > 1e-15) & changed).sum()),
            "WV3_620_worse_changed_users": int(((diff < -1e-15) & changed).sum()),
            "neutral_change_users": int(((diff.abs() <= 1e-15) & changed).sum()),
            "actual_delta_sum_change": float(diff.sum()),
        }
    actionable_rate = historical["actionable_rows"] / historical["actionability_rows"]
    gate = experiment_contract()["decision_gate"]
    passed = bool(
        actionable_rate <= gate["historical_actionable_pair_rate_max"]
        and per_model["WV3-610"]["mean_selected_neutral_share"] >= gate["WV3_610_mean_selected_neutral_share_min"]
        and per_model["WV3-610"]["mean_beneficial_user_capture"] <= gate["WV3_610_mean_beneficial_capture_max"]
        and not screens["WV3-620"]["incremental_screening_vs_WV3_610"]["passed"]
    )
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "historical_pair_rows": historical["actionability_rows"],
        "historical_actionable_pair_rows": historical["actionable_rows"],
        "historical_actionable_pair_rate": actionable_rate,
        "models": per_model,
        "WV3_610_vs_WV3_620_decision_disagreement": disagreement,
        "candidate_level_propensity_experiment_authorized": passed,
        "new_training": False,
        "new_outer_exposure": 0,
        "final_week": "not_run",
    }
    write(OUTPUT, result)
    lines = [
        "# WV3-621 商品对监督分解审计",
        "",
        "## 结论",
        "",
        f"候选商品级购买概率实验授权门槛：**{'pass' if passed else 'fail'}**。四个历史源共有 {historical['actionability_rows']:,} 个挑战—被替换商品对，其中恰好一件被购买的可行动商品对 {historical['actionable_rows']:,} 个，占 {actionable_rate:.3%}。",
        "",
        "## 术语",
        "",
        "- 商品对膨胀（本项目自定义诊断）：同一候选商品会与第8–12名中的多件商品重复组合；训练和推理行数按用户—挑战商品—被替换商品三元组计算，而非唯一用户—商品对。",
        "- 可行动商品对（本项目自定义）：同一用户的两件候选在监督周恰好一件被购买；只有这种商品对的直接换位会产生非零 AP@12 变化。可行动率分母是四个历史源的全部商品对。",
        "- 有益用户捕获率（本项目自定义）：模型选中有益交换的用户数除以 WV3-600 一换一 Oracle 中至少存在有益交换的用户数；按窗口分别计算后取算术平均。",
        "- Oracle 兑现率（本项目自定义）：模型实际 population delta 除以同一动作空间的一换一 Oracle population delta。Oracle 使用标签选最佳交换，只代表上限，不是可部署模型。",
        "- 候选商品级购买概率（行业常见 pointwise propensity 思路，本报告称点式购买倾向）：每个用户—商品只预测一次未来购买概率，再用“挑战商品概率减被替换商品概率”乘位置 AP 权重，得到一次换位的期望收益。",
        "",
        "## 三代局部准入对比",
        "",
        "| 方案 | 平均入选中性占比 | 平均有益用户捕获率 | 平均 Oracle 兑现率 |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        row = per_model[model]
        lines.append(f"| {model} | {row['mean_selected_neutral_share']:.2%} | {row['mean_beneficial_user_capture']:.2%} | {row['mean_realized_oracle_ratio']:.2%} |")
    lines += [
        "",
        "WV3-620 已经加入候选商品自身特征，但没有通过相对 WV3-610 的内层增量门槛；若本审计通过，说明问题不只是模型没看见商品特征，而是当前把同一商品重复展开成大量中性商品对的监督分解效率低。后续只授权一个点式购买倾向模型，动作空间、候选池和外层门槛保持不变。",
        "",
        "本审计只读取已完成内层决策及审计摘要，没有训练模型，没有新增外层读取。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    MARKDOWN.write_text("\n".join(lines), encoding="utf-8")
    common.update(
        TRIAL,
        decision="diagnostic_supports_pointwise_propensity" if passed else "diagnostic_rejects_pointwise_propensity",
        inner_evidence={
            "historical_actionable_pair_rate": actionable_rate,
            "WV3_610_mean_selected_neutral_share": per_model["WV3-610"]["mean_selected_neutral_share"],
            "WV3_610_mean_beneficial_user_capture": per_model["WV3-610"]["mean_beneficial_user_capture"],
            "authorized": passed,
        },
        runtime=0.0,
        artifact_paths=[str(OUTPUT), str(MARKDOWN)],
    )
    common.log(
        f"{TRIAL} pair factorization audit",
        "WV3-620 changed decisions with candidate-specific features but failed its incremental inner gate, while all residual variants still selected mostly neutral pairs.",
        experiment_contract()["hypothesis"],
        str(OUTPUT),
        f"historical actionable-pair rate={actionable_rate:.3%}; WV3-610 neutral share={per_model['WV3-610']['mean_selected_neutral_share']:.2%}; beneficial capture={per_model['WV3-610']['mean_beneficial_user_capture']:.2%}; gate={passed}.",
        "Authorize one candidate-level purchase-propensity experiment." if passed else "Do not change the supervision factorization.",
        "Estimate each candidate once, then use the exact probability difference times position gain inside the same protected one-swap action space.",
        alternatives="This is not a threshold rescue of WV3-610/620 and not the broad Top50 reorder used by WV3-501.",
        experiment="Read-only comparison of completed inner decisions and historical sample density.",
        reflection="The audit separates feature availability from label factorization before paying for another model.",
    )
    print({"trial": TRIAL, "actionable_pair_rate": actionable_rate, "authorized": passed}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "run"])
    globals()[parser.parse_args().command]()
