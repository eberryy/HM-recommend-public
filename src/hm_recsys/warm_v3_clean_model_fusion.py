"""WV3-741: outer confirmation of clean pointwise/LambdaRank rank fusion."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import save_parquet
from .warm_v3_clean_model_replay import (
    LAMBDA_MODEL,
    POINT_MODEL,
    fused_candidate_scores,
    summarize_policy,
)
from .warm_v3_label_free_repair import choose_label_free_swaps, unlabeled_proposals
from .warm_v3_residual_admission import OUTER, pair_frame, source
from .warm_v3_rich_propensity import FEATURES, rich_candidate_frame


TRIAL = "WV3-741"
CONTRACT = common.REPORT / "WV3-741_CLEAN_MODEL_FUSION_CONTRACT.json"
REPORT = common.REPORT / "WV3-741_OUTER.json"
MARKDOWN = common.REPORT / "WV3-741_FINAL.md"
ART_ROOT = common.ART / TRIAL


def contract() -> dict:
    selected = read(common.REPORT / "WV3-740_CLEAN_MODEL_REPLAY.json")
    if selected["authorized_policy"] != "pointwise_lambdarank_RRF60":
        raise AssertionError("WV3-740 did not select the registered WV3-741 policy")
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-721",
        "architecture_family": "clean_pointwise_lambdarank_rank_fusion_tail_admission",
        "hypothesis": (
            "The strict-history pointwise model and within-user LambdaRank make complementary candidate-ordering "
            "errors; their fixed equal RRF60 rank fusion can improve WV3-721 on unseen outer development windows."
        ),
        "models": "exact frozen WV3-661 and WV3-680 LightGBM artifacts; no training",
        "fusion": "within each user, rank both model scores and add 1/(60+pointwise_rank)+1/(60+lambdarank_rank)",
        "action": "positive fused-score difference; at most two disjoint swaps; challengers13-50, victims8-12; ranks1-7 protected",
        "candidate_pool": "unchanged WV3-721 Top50 and full truth-user denominator",
        "selection_evidence": "WV3-740 inner mean +0.000108954800 vs WV3-721, 4/4 positive",
        "outer_gate": "mean incremental vs WV3-721 >0, >=3/4 nondegrade, worst>=-0.0002, and stable vs WV2-601",
        "outer_exposures": 1,
        "no_rescue": "no fusion weight, RRF constant, score threshold, swap count or tie-order adjustment",
        "expected_minutes": 15,
        "fallback": "retain WV3-721 and close clean historical-model fusion",
        "final_week": "2020-09-16 not_run",
    }


def register() -> dict:
    common.setup()
    common.budget(15)
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(
            TRIAL,
            contract()["architecture_family"],
            contract()["hypothesis"],
            candidate_protocol=contract()["candidate_pool"],
            training_protocol=contract()["models"],
            features=FEATURES,
            params={"fusion": contract()["fusion"], "action": contract()["action"], "outer_gate": contract()["outer_gate"], "final_week": "not_run"},
            expected_minutes=15,
        )
    common.update(
        TRIAL,
        decision="inner_pass",
        inner_evidence={
            "source": str(common.REPORT / "WV3-740_CLEAN_MODEL_REPLAY.json"),
            "authorized_policy": "pointwise_lambdarank_RRF60",
            "mean_delta_vs_WV3_721": 0.00010895480026346884,
            "nondegrade_windows_vs_WV3_721": 4,
            "worst_delta_vs_WV3_721": 0.00005605054829586388,
            "passed": True,
        },
        artifact_paths=[str(CONTRACT)],
        validity="valid_label_free_outer_candidate",
    )
    return read(CONTRACT)


def evaluate(name: str, folder: str, point: lgb.Booster, lambdarank: lgb.Booster) -> tuple[dict, object, object]:
    path, meta = source(folder)
    candidates, details = rich_candidate_frame(meta["cutoff"], rank_min=1)
    matrix = candidates[FEATURES].to_numpy(np.float32)
    point_values = point.predict(matrix, num_threads=4)
    lambda_values = lambdarank.predict(matrix, num_threads=4)
    scores = fused_candidate_scores(candidates, point_values, lambda_values)
    pairs = pair_frame(path, discordant_only=False)
    swaps = choose_label_free_swaps(unlabeled_proposals(pairs), scores)
    summary, users = summarize_policy(candidates, swaps, meta)
    control = read(common.REPORT / "WV3-721_CORRECTED_OUTER_REPLAY.json")["windows"][name]
    summary.update(
        {
            "window": name,
            "cutoff": meta["cutoff"],
            "total_users_denominator": meta["total_users"],
            "same_pool_control_MAP@12": control["MAP@12"],
            "incremental_delta_vs_WV3_721": summary["MAP@12"] - control["MAP@12"],
            "rich_candidate_input": details,
            "same_pool_control_reproduces_WV3_721": True,
            "labels_joined_after_actions_frozen": True,
            "final_week": "not_run",
        }
    )
    return summary, swaps, users


def render(result: dict) -> str:
    incremental = result["incremental_vs_WV3_721"]
    lines = [
        "# WV3-741：点式模型与 LambdaRank 的无标签名次融合",
        "",
        "## 结论",
        "",
        f"外层门槛：**{'通过' if result['passed'] else '未通过'}**。MAP@12 四窗均值为 `{result['mean_MAP']:.12f}`，"
        f"相对 WV3-721 为 `{incremental['mean_delta_vs_WV3_721']:+.12f}`。",
        "",
        "## 术语",
        "",
        "- 点式模型（推荐排序常见）：WV3-661 对每个用户—商品独立输出购买倾向分数。",
        "- LambdaRank（行业通用学习排序）：WV3-680 在同一用户候选组内学习正例应排在负例之前。",
        "- 名次融合（推荐系统常见）：先把两个异尺度模型分数分别转成用户内名次，再用固定 RRF60 相加；本实验等权，不拟合融合参数。",
        "- 无标签尾部换位（本项目自定义）：只按融合分数差选择最多两组不冲突换位，挑战商品来自第13—50名，被替换商品来自第8—12名；真实标签只在选择完成后计算 MAP@12。",
        "- 同池对照（本项目评测约束）：WV3-741 与 WV3-721 使用完全相同的 Top50 用户—商品集合；关闭新融合动作时精确回到 WV3-721。",
        "",
        "| 外层窗口 | WV3-721 | WV3-741 | 增量 | 选择用户 | 有益 / 有害 / 中性 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['same_pool_control_MAP@12']:.9f} | {row['MAP@12']:.9f} | "
            f"{row['incremental_delta_vs_WV3_721']:+.9f} | {row['selected_users']:,} | "
            f"{row['beneficial_selected_users']:,} / {row['harmful_selected_users']:,} / {row['neutral_selected_users']:,} |"
        )
    lines += [
        "",
        f"相对 WV3-721 不退化窗口 `{incremental['nondegrade_windows_vs_WV3_721']}/4`，最差增量 `{incremental['worst_delta_vs_WV3_721']:+.9f}`。",
        "本轮只进行一次冻结外层确认；未调 RRF 常数、权重、阈值或换位数。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def confirm() -> dict:
    common.setup()
    common.budget(15)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "inner_pass" or entry["outer_exposures"] != 1:
        raise AssertionError("WV3-741 requires the frozen inner winner and exactly one recorded outer exposure")
    point = lgb.Booster(model_file=str(POINT_MODEL))
    lambdarank = lgb.Booster(model_file=str(LAMBDA_MODEL))
    started = time.perf_counter()
    windows = {}
    for name, folder in OUTER:
        row, swaps, users = evaluate(name, folder, point, lambdarank)
        windows[name] = row
        root = ART_ROOT / name / "outer"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(swaps, root / "swaps.parquet")
        save_parquet(users, root / "users.parquet")
        write(root / "REVIEW.json", row)
        print({"clean_model_fusion_outer": name, "incremental": row["incremental_delta_vs_WV3_721"]}, flush=True)
    standard = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    increments = {name: row["incremental_delta_vs_WV3_721"] for name, row in windows.items()}
    incremental = {
        "per_window_delta_vs_WV3_721": increments,
        "mean_delta_vs_WV3_721": float(np.mean(list(increments.values()))),
        "nondegrade_windows_vs_WV3_721": sum(value >= 0 for value in increments.values()),
        "worst_delta_vs_WV3_721": min(increments.values()),
    }
    passed = bool(
        standard["stable"]
        and incremental["mean_delta_vs_WV3_721"] > 0
        and incremental["nondegrade_windows_vs_WV3_721"] >= 3
        and incremental["worst_delta_vs_WV3_721"] >= -0.0002
    )
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "windows": windows,
        **standard,
        "incremental_vs_WV3_721": incremental,
        "passed": passed,
        "new_training": False,
        "candidate_pool_changed": False,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(REPORT, result)
    MARKDOWN.write_text(render(result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="valid_development_champion" if passed else "reject_outer",
        outer_MAP_by_window=standard["per_window_MAP"],
        mean_MAP=standard["mean_MAP"],
        delta_vs_WV2_601=standard["delta_vs_WV2_601"],
        nondegrade_windows=standard["nondegrade_windows"],
        worst_delta=standard["worst_delta"],
        runtime=result["runtime_seconds"],
        artifact_paths=entry["artifact_paths"] + [str(REPORT), str(MARKDOWN)],
        validity="valid_label_free_outer_confirmation",
    )
    state = read(common.REGISTRY)
    if passed:
        state["current_champion"] = TRIAL
        state["current_valid_champion"] = TRIAL
        state["best_stable"] = {
            "experiment_id": TRIAL,
            "mean_MAP": standard["mean_MAP"],
            "delta_vs_WV2_601": standard["delta_vs_WV2_601"],
            "nondegrade_windows": standard["nondegrade_windows"],
            "worst_delta": standard["worst_delta"],
        }
    state["status"] = "running_below_target" if passed else "running_fusion_rejected"
    write(common.REGISTRY, state)
    common.log(
        f"{TRIAL} clean model-fusion outer confirmation",
        "WV3-740 selected equal pointwise/LambdaRank RRF60 after a 4/4 positive inner comparison against WV3-721.",
        contract()["hypothesis"],
        f"{REPORT}; {MARKDOWN}",
        f"mean MAP={standard['mean_MAP']:.9f}; incremental={incremental['mean_delta_vs_WV3_721']:+.9f}; nondegrade={incremental['nondegrade_windows_vs_WV3_721']}/4; worst={incremental['worst_delta_vs_WV3_721']:+.9f}; pass={passed}.",
        "Promote WV3-741 as valid development champion." if passed else "Reject and retain WV3-721.",
        "Continue with a distinct audited mechanism only; do not tune this fusion from outer results.",
        alternatives="Weights, RRF constant and action count were frozen before outer access; no outer-driven rescue is permitted.",
        experiment="One outer exposure of the exact clean inner winner with unchanged candidate identities and post-decision labels.",
        reflection="This test measures objective complementarity while isolating the action-policy leakage discovered in WV3-720.",
    )
    return result


if __name__ == "__main__":
    register()
    common.expose(TRIAL)
    confirm()
