"""WV3-700: audit whether WV3-691 needs a learned challenger-victim gate."""
from __future__ import annotations

import itertools
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v3_residual_admission import INNER, pair_frame, source
from .warm_v3_residual_admission_audit import apk_from_targets
from .warm_v3_rich_propensity import FEATURES, rich_candidate_frame


TRIAL = "WV3-700"
TOP_PAIRS = 5
CONTRACT = common.REPORT / "WV3-700_PAIR_GATE_AUDIT_CONTRACT.json"
REPORT = common.REPORT / "WV3-700_PAIR_GATE_AUDIT.json"
MARKDOWN = common.REPORT / "WV3-700_PAIR_GATE_AUDIT.md"
MODEL_PATH = common.ART / "WV3-661" / "MODEL.txt"


def top_pair_proposals(
    pairs: pd.DataFrame,
    probabilities: pd.DataFrame,
    top_pairs: int = TOP_PAIRS,
) -> pd.DataFrame:
    """Return each user's highest positive pointwise-implied swap proposals."""
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
    result = pairs[columns].merge(
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
    assert result[["challenger_probability", "victim_probability"]].notna().all().all()
    result["probability_difference"] = result.challenger_probability - result.victim_probability
    result["pointwise_expected_gain"] = result.probability_difference * result.unit_gain
    result = result[result.pointwise_expected_gain > 0].sort_values(
        [
            "customer_id",
            "pointwise_expected_gain",
            "probability_difference",
            "victim_rank",
            "challenger_rank",
            "challenger_article_id",
            "victim_article_id",
        ],
        ascending=[True, False, False, False, True, True, True],
        kind="mergesort",
    )
    result["proposal_rank"] = result.groupby("customer_id", sort=False).cumcount() + 1
    return result[result.proposal_rank <= top_pairs].reset_index(drop=True)


def best_exact_gain(base: pd.DataFrame, proposals: pd.DataFrame) -> tuple[float, int]:
    """Exact best AP@12 gain from no action, one swap, or two disjoint swaps."""
    ordered = base.sort_values("rf", kind="mergesort")
    values = ordered.target.to_numpy(np.uint8)
    truth_count = int(ordered.truth_count.iloc[0])
    before = apk_from_targets(values, truth_count)
    best = before
    best_count = 0
    rows = list(proposals.itertuples(index=False))
    choices = [(row,) for row in rows]
    choices.extend(
        pair
        for pair in itertools.combinations(rows, 2)
        if pair[0].challenger_article_id != pair[1].challenger_article_id
        and pair[0].victim_article_id != pair[1].victim_article_id
    )
    for choice in choices:
        changed = values.copy()
        for row in choice:
            changed[int(row.victim_rank) - 1], changed[int(row.challenger_rank) - 1] = (
                changed[int(row.challenger_rank) - 1],
                changed[int(row.victim_rank) - 1],
            )
        after = apk_from_targets(changed, truth_count)
        if after > best + 1e-15:
            best = after
            best_count = len(choice)
    return best - before, best_count


def audit_window(folder: str, name: str, model: lgb.Booster) -> dict:
    path, meta = source(folder)
    candidates, details = rich_candidate_frame(meta["cutoff"], rank_min=1)
    score = model.predict(candidates[FEATURES].to_numpy(np.float32), num_threads=4)
    probabilities = candidates[["customer_id", "article_id"]].copy()
    probabilities["purchase_probability"] = score
    proposals = top_pair_proposals(pair_frame(path, discordant_only=False), probabilities)
    isolated = (
        proposals.challenger_target.astype(np.int8) - proposals.victim_target.astype(np.int8)
    ) * proposals.unit_gain
    proposals["isolated_actual_gain"] = isolated

    groups = {customer: group for customer, group in proposals.groupby("customer_id", sort=False)}
    oracle_gain = 0.0
    oracle_users = 0
    oracle_second = 0
    for customer, base in candidates.groupby("customer_id", sort=False):
        group = groups.get(customer)
        if group is None:
            continue
        gain, count = best_exact_gain(base, group)
        oracle_gain += gain
        oracle_users += int(gain > 1e-15)
        oracle_second += int(count == 2)

    current = read(common.REPORT / "WV3-691_SCREEN.json")["windows"][name]
    restricted_delta = float(oracle_gain / meta["total_users"])
    result = {
        "window": name,
        "cutoff": meta["cutoff"],
        "total_users_denominator": meta["total_users"],
        "proposal_rows": len(proposals),
        "proposal_users": int(proposals.customer_id.nunique()),
        "beneficial_proposal_rows": int((isolated > 1e-15).sum()),
        "harmful_proposal_rows": int((isolated < -1e-15).sum()),
        "neutral_proposal_rows": int((np.abs(isolated) <= 1e-15).sum()),
        "beneficial_proposal_users": int(proposals.loc[isolated > 1e-15, "customer_id"].nunique()),
        "restricted_two_swap_oracle_users": oracle_users,
        "restricted_two_swap_oracle_second_swap_users": oracle_second,
        "restricted_two_swap_oracle_population_delta": restricted_delta,
        "wv3_691_population_delta": current["population_delta"],
        "remaining_population_headroom_vs_wv3_691": restricted_delta - current["population_delta"],
        "top_pair_limit_per_user": TOP_PAIRS,
        "candidate_input": details,
        "new_training": False,
        "outer_labels_used": False,
        "final_week": "not_run",
    }
    return result


def contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-691",
        "architecture_family": "replacement_relation_error_audit",
        "hypothesis": "WV3-691 leaves enough AP@12 headroom inside its five strongest pointwise swap proposals, and those proposals contain enough beneficial and non-beneficial labels to justify a separately learned replacement gate.",
        "model": "none; exact read-only audit with frozen WV3-661 scores",
        "candidate_pool": "unchanged Top50; ranks1-7 protected, challengers13-50, victims8-12",
        "proposal_policy": "fixed top5 positive pointwise-implied challenger-victim pairs per active user; no TopN search",
        "authorization_gate": {
            "beneficial_proposal_rows_total_min": 500,
            "beneficial_proposal_users_each_window_min": 100,
            "mean_remaining_population_headroom_min": 0.0005,
            "remaining_population_headroom_each_window_min": 0.0002,
            "both_harmful_and_neutral_examples_each_window": True,
        },
        "expected_minutes": 10,
        "fallback": "If the gate fails, do not train a pair gate; pivot to candidate coverage or user-interest representation.",
        "final_week": "2020-09-16 not_run",
    }


def render(result: dict) -> str:
    lines = [
        "# WV3-700 挑战商品—被替换商品关系审计",
        "",
        "## 结论",
        "",
        f"第二阶段替换准入模型：**{'获准' if result['pair_gate_authorized'] else '未获准'}**。",
        "",
        "## 术语",
        "",
        "- 替换提案（本项目自定义）：同一用户中，以冻结 WV3-661 购买倾向分数计算后为正的挑战商品—被替换商品组合；挑战商品来自基线第13–50名，被替换商品来自第8–12名。统计单位为用户—商品对—商品对。",
        "- Top5 提案（本项目自定义）：每位用户按购买倾向差乘位置收益排序后的前5个正收益替换提案；5是实验前固定的计算预算，不是按结果搜索得到。",
        "- 受限两次换位 Oracle（本项目自定义上界）：只在上述 Top5 提案中，使用真实标签穷举不操作、一次或两次互不冲突换位所得的最佳 AP@12；它是诊断上界，不是可部署模型。",
        "- remaining headroom（本报告称剩余提升空间）：受限 Oracle 的全体真值用户平均 MAP 增益减 WV3-691 已实现增益；分母是该窗口全部真值用户。",
        "- 有益／有害／中性提案：孤立执行该换位时 AP@12 分别上升、下降或不变；中性通常表示两件商品都未购买或都购买。",
        "",
        "| 内层窗口 | 提案行 | 有益 / 有害 / 中性 | 有益用户 | 受限 Oracle 增益 | WV3-691 增益 | 剩余空间 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['proposal_rows']:,} | {row['beneficial_proposal_rows']:,} / "
            f"{row['harmful_proposal_rows']:,} / {row['neutral_proposal_rows']:,} | "
            f"{row['beneficial_proposal_users']:,} | {row['restricted_two_swap_oracle_population_delta']:+.9f} | "
            f"{row['wv3_691_population_delta']:+.9f} | {row['remaining_population_headroom_vs_wv3_691']:+.9f} |"
        )
    lines += [
        "",
        "本审计未训练新模型、未读取外层窗口、未改变候选池。只有全部预注册门槛同时满足，后续才允许使用严格历史的窗口外预测构造第二阶段监督。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    common.setup()
    common.budget(10)
    assert MODEL_PATH.is_file()
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "replacement_relation_error_audit",
            contract()["hypothesis"],
            candidate_protocol=contract()["candidate_pool"],
            training_protocol="none; frozen WV3-661 model and 2020 inner labels for diagnosis only",
            params={"top_pair_limit_per_user": TOP_PAIRS, "authorization_gate": contract()["authorization_gate"], "final_week": "not_run"},
            expected_minutes=10,
        )
    started = time.perf_counter()
    model = lgb.Booster(model_file=str(MODEL_PATH))
    windows = {name: audit_window(folder, name, model) for name, folder in INNER}
    headroom = [row["remaining_population_headroom_vs_wv3_691"] for row in windows.values()]
    gate = contract()["authorization_gate"]
    authorized = bool(
        sum(row["beneficial_proposal_rows"] for row in windows.values()) >= gate["beneficial_proposal_rows_total_min"]
        and all(row["beneficial_proposal_users"] >= gate["beneficial_proposal_users_each_window_min"] for row in windows.values())
        and float(np.mean(headroom)) >= gate["mean_remaining_population_headroom_min"]
        and min(headroom) >= gate["remaining_population_headroom_each_window_min"]
        and all(row["harmful_proposal_rows"] > 0 and row["neutral_proposal_rows"] > 0 for row in windows.values())
    )
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "windows": windows,
        "aggregate": {
            "beneficial_proposal_rows_total": sum(row["beneficial_proposal_rows"] for row in windows.values()),
            "mean_remaining_population_headroom_vs_wv3_691": float(np.mean(headroom)),
            "minimum_remaining_population_headroom_vs_wv3_691": min(headroom),
        },
        "authorization_gate": gate,
        "pair_gate_authorized": authorized,
        "runtime_seconds": time.perf_counter() - started,
        "new_training": False,
        "new_outer_exposure": 0,
        "final_week": "not_run",
    }
    write(REPORT, result)
    MARKDOWN.write_text(render(result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="diagnostic_authorizes_stacked_pair_gate" if authorized else "diagnostic_rejects_pair_gate",
        inner_evidence=result,
        runtime=result["runtime_seconds"],
        artifact_paths=[str(CONTRACT), str(REPORT), str(MARKDOWN)],
    )
    common.log(
        f"{TRIAL} replacement-relation audit",
        "WV3-691 uses one pointwise score difference to choose the challenger, victim and whether to act; selected users remain mostly neutral.",
        contract()["hypothesis"],
        f"{REPORT}; {MARKDOWN}",
        f"beneficial rows={result['aggregate']['beneficial_proposal_rows_total']:,}; mean remaining headroom={result['aggregate']['mean_remaining_population_headroom_vs_wv3_691']:+.9f}; minimum={result['aggregate']['minimum_remaining_population_headroom_vs_wv3_691']:+.9f}; authorized={authorized}.",
        "Authorize a stacked pair gate with historical out-of-window supervision." if authorized else "Reject pair-gate training and pivot away from replacement relation modeling.",
        "If authorized, preregister one fixed stacked regression model before fitting; otherwise audit candidate coverage or multi-interest evidence.",
        alternatives="Top100, rich causal refresh and rich LambdaRank already have negative or low-increment evidence; no threshold or TopN search is performed.",
        experiment="Read-only exact audit of the five strongest positive pointwise replacement proposals per user.",
        reflection="The audit separates remaining action-selection headroom from a need for additional retrieval candidates.",
    )
    print({"trial": TRIAL, "authorized": authorized, "aggregate": result["aggregate"]}, flush=True)
    return result


if __name__ == "__main__":
    run()
