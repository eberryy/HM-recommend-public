"""WV3-720: quantify target-dependent unit_gain in Warm-v3 action selection."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v3_residual_admission import INNER, OUTER, pair_frame, source
from .warm_v3_rich_propensity import FEATURES, rich_candidate_frame
from .warm_v3_two_swap_rich_propensity import choose_two_swaps, exact_user_deltas


TRIAL = "WV3-720"
CONTRACT = common.REPORT / "WV3-720_UNIT_GAIN_LEAKAGE_AUDIT_CONTRACT.json"
REPORT = common.REPORT / "WV3-720_UNIT_GAIN_LEAKAGE_AUDIT.json"
MARKDOWN = common.REPORT / "WV3-720_UNIT_GAIN_LEAKAGE_AUDIT.md"
MODEL_PATH = common.ART / "WV3-661" / "MODEL.txt"


def choose_two_swaps_label_free(pairs: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Mechanical repair diagnostic: order only by model score differences."""
    challenger = scores.rename(columns={"article_id": "challenger_article_id", "ranking_score": "challenger_score"})
    victim = scores.rename(columns={"article_id": "victim_article_id", "ranking_score": "victim_score"})
    columns = [
        "customer_id",
        "challenger_article_id",
        "victim_article_id",
        "challenger_rank",
        "victim_rank",
        "unit_gain",
    ]
    ranked = pairs[columns].merge(
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
    assert ranked[["challenger_score", "victim_score"]].notna().all().all()
    ranked["score_difference"] = ranked.challenger_score - ranked.victim_score
    ranked = ranked[ranked.score_difference > 0].sort_values(
        [
            "customer_id",
            "score_difference",
            "victim_rank",
            "challenger_rank",
            "challenger_article_id",
            "victim_article_id",
        ],
        ascending=[True, False, False, True, True, True],
        kind="mergesort",
    )
    selected = []
    for _, group in ranked.groupby("customer_id", sort=False):
        used_challengers = set()
        used_victims = set()
        order = 0
        for row in group.itertuples(index=False):
            if row.challenger_article_id in used_challengers or row.victim_article_id in used_victims:
                continue
            order += 1
            record = row._asdict()
            record["swap_order"] = order
            selected.append(record)
            used_challengers.add(row.challenger_article_id)
            used_victims.add(row.victim_article_id)
            if order == 2:
                break
    return pd.DataFrame(selected, columns=list(ranked.columns) + ["swap_order"])


def swap_keys(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(
        zip(
            frame.customer_id.astype(str),
            frame.challenger_article_id.astype(str),
            frame.victim_article_id.astype(str),
        )
    )


def audit_window(folder: str, name: str, model: lgb.Booster, stage: str) -> dict:
    path, meta = source(folder)
    candidates, details = rich_candidate_frame(meta["cutoff"], rank_min=1)
    values = model.predict(candidates[FEATURES].to_numpy(np.float32), num_threads=4)
    scores = candidates[["customer_id", "article_id"]].copy()
    scores["ranking_score"] = values
    pairs = pair_frame(path, discordant_only=False)

    leaked_swaps = choose_two_swaps(pairs, scores)
    safe_swaps = choose_two_swaps_label_free(pairs, scores)
    leaked_users = exact_user_deltas(candidates, leaked_swaps)
    safe_users = exact_user_deltas(candidates, safe_swaps)
    leaked_delta = float(leaked_users.actual_delta.sum() / meta["total_users"])
    safe_delta = float(safe_users.actual_delta.sum() / meta["total_users"])
    expected_source = common.REPORT / ("WV3-691_SCREEN.json" if stage == "inner" else "WV3-691_OUTER.json")
    expected = read(expected_source)["windows"][name]["population_delta"]
    if abs(leaked_delta - expected) > 1e-12:
        raise AssertionError(f"WV3-691 reproduction failed for {name}: {leaked_delta} vs {expected}")

    leaked_keys = swap_keys(leaked_swaps)
    safe_keys = swap_keys(safe_swaps)
    changed_users = set(leaked_swaps.customer_id) ^ set(safe_swaps.customer_id)
    both_users = set(leaked_swaps.customer_id) & set(safe_swaps.customer_id)
    leaked_by_user = {u: set(zip(g.challenger_article_id, g.victim_article_id)) for u, g in leaked_swaps.groupby("customer_id")}
    safe_by_user = {u: set(zip(g.challenger_article_id, g.victim_article_id)) for u, g in safe_swaps.groupby("customer_id")}
    changed_users.update(u for u in both_users if leaked_by_user[u] != safe_by_user[u])

    return {
        "window": name,
        "cutoff": meta["cutoff"],
        "role": stage,
        "total_users_denominator": meta["total_users"],
        "leaked_policy_population_delta": leaked_delta,
        "label_free_mechanical_policy_population_delta": safe_delta,
        "leaked_minus_label_free_delta": leaked_delta - safe_delta,
        "leaked_selected_pairs": len(leaked_swaps),
        "label_free_selected_pairs": len(safe_swaps),
        "identical_selected_pairs": len(leaked_keys & safe_keys),
        "changed_decision_users": len(changed_users),
        "leaked_policy_reproduced": True,
        "selection_uses_target_derived_unit_gain": True,
        "target_derived_fields": ["positives_before", "later_positive_inverse", "truth_count"],
        "candidate_input": details,
        "new_training": False,
        "final_week": "not_run",
    }


def contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-691",
        "architecture_family": "evaluation_contract_label_leakage_audit",
        "hypothesis": "The unit_gain used to order Warm-v3 swap actions depends on future labels and materially changes selected challenger-victim pairs, invalidating promotion evidence even if the trained propensity model itself remains cutoff-safe.",
        "audit_only": True,
        "mechanical_comparison": "Remove unit_gain from action ordering; retain frozen WV3-661 scores, positive score-difference condition, two-swap cap and deterministic ties. Labels remain only in post-decision MAP calculation.",
        "scope": "all four already-used inner and outer windows; no new model, candidate, parameter, or final-week access",
        "decision_rule": "Any target-derived field influencing selected pairs is a severe contract failure; impact magnitude is descriptive, not an acceptance threshold.",
        "expected_minutes": 10,
        "fallback": "Revert current valid champion to WV2-601 and require a separately preregistered label-free action repair before further architecture research.",
        "final_week": "2020-09-16 not_run",
    }


def render(result: dict) -> str:
    lines = [
        "# WV3-720 换位决策标签泄漏审计",
        "",
        "## 结论",
        "",
        "**确认存在严重评测合同错误。WV3-691 及共享该决策代码的实验不能作为有效晋级证据；当前有效 Warm champion 回退为 WV2-601。**",
        "",
        "## 泄漏机制",
        "",
        "- `unit_gain`（本项目原自定义名，原报告称精确位置收益权重）：它并非只由候选名次决定，而是使用验证周真实标签计算被替换位置之前的正例数、之后正例的倒数名次和该用户真实购买商品总数。",
        "- 决策阶段标签泄漏（行业通用风险）：模型预测完成后，代码以“模型分数差 × unit_gain”排序并选择换位，因此未来一周真实购买结果改变了系统实际选择的挑战商品和被替换商品。真实标签本应只能用于选择完成后的 MAP@12 评分。",
        "- label-free mechanical policy（本报告称无标签机械对照）：保持同一冻结模型、候选池、正分数差条件和最多两次换位，只按模型分数差排序；真实标签只用于事后计分。它用于量化影响，不构成已预注册的新 baseline。",
        "- changed decision users（本报告称决策改变用户数）：泄漏策略与无标签机械对照选择的挑战商品—被替换商品集合不同的用户数；统计对象是当前排名模型覆盖的活跃用户。",
        "",
        "| 阶段 / 窗口 | 泄漏策略增益 | 无标签机械对照增益 | 泄漏额外增益 | 决策改变用户 |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in ("inner", "outer"):
        for name, row in result[stage].items():
            lines.append(
                f"| {stage} / {name} | {row['leaked_policy_population_delta']:+.9f} | "
                f"{row['label_free_mechanical_policy_population_delta']:+.9f} | "
                f"{row['leaked_minus_label_free_delta']:+.9f} | {row['changed_decision_users']:,} |"
            )
    lines += [
        "",
        "影响大小不改变定性判断：只要未来标签参与动作选择，结果即不能作为可部署策略证据。训练于严格历史标签的 WV3-661 LightGBM 模型文件本身可以保留，但所有使用 target-derived unit_gain 选择换位的 MAP 与 promotion 状态必须作废。",
        "",
        "本审计只读取已经暴露过的 inner/outer 证据；没有产生新的模型或外层架构暴露。最终周 `2020-09-16` 保持 `not_run`。后续必须先讨论并注册无标签动作策略，再恢复实验。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    common.setup()
    common.budget(10)
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            contract()["architecture_family"],
            contract()["hypothesis"],
            candidate_protocol="unchanged WV3-691 Top50",
            training_protocol="none; frozen WV3-661 model",
            params={"audit_only": True, "decision_rule": contract()["decision_rule"], "final_week": "not_run"},
            expected_minutes=10,
        )
    started = time.perf_counter()
    model = lgb.Booster(model_file=str(MODEL_PATH))
    inner = {name: audit_window(folder, name, model, "inner") for name, folder in INNER}
    outer = {name: audit_window(folder, name, model, "outer") for name, folder in OUTER}
    all_rows = [*inner.values(), *outer.values()]
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "severity": "severe_evaluation_contract_failure",
        "confirmed": True,
        "root_cause": "target-derived unit_gain participated in challenger-victim action ordering",
        "inner": inner,
        "outer": outer,
        "aggregate": {
            "changed_decision_users_total_across_windows": sum(row["changed_decision_users"] for row in all_rows),
            "mean_leaked_minus_label_free_delta": float(np.mean([row["leaked_minus_label_free_delta"] for row in all_rows])),
            "all_leaked_runs_exactly_reproduced": all(row["leaked_policy_reproduced"] for row in all_rows),
        },
        "affected_experiment_ids": [
            "WV3-601", "WV3-602", "WV3-610", "WV3-620", "WV3-621", "WV3-630", "WV3-631",
            "WV3-641", "WV3-650", "WV3-661", "WV3-670", "WV3-680", "WV3-691", "WV3-700", "WV3-710",
        ],
        "trained_propensity_model_reusable_after_action_repair": True,
        "current_valid_champion": "WV2-601",
        "new_training": False,
        "new_outer_architecture_exposure": 0,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(REPORT, result)
    MARKDOWN.write_text(render(result), encoding="utf-8")
    state = read(common.REGISTRY)
    for entry in state["trials"]:
        if entry["experiment_id"] in result["affected_experiment_ids"]:
            entry["validity"] = "invalid_target_dependent_action_selection"
    entry = next(row for row in state["trials"] if row["experiment_id"] == TRIAL)
    entry.update(
        decision="confirmed_severe_contract_failure",
        inner_evidence={"confirmed": True, "severity": result["severity"], "aggregate": result["aggregate"]},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(CONTRACT), str(REPORT), str(MARKDOWN)],
        validity="valid_diagnostic",
    )
    state["current_champion"] = "WV2-601"
    state["status"] = "blocked_severe_evaluation_contract_failure"
    state["validity_addendum"] = {
        "created_at": result["created_at"],
        "audit": str(REPORT),
        "invalidated_champion": "WV3-691",
        "current_valid_champion": "WV2-601",
        "reason": result["root_cause"],
        "final_week": "not_run",
    }
    write(common.REGISTRY, state)
    common.log(
        f"{TRIAL} severe evaluation-contract audit",
        "The exact unit_gain used in Warm-v3 swap ordering contains future validation labels (positives_before, later_positive_inverse and truth_count).",
        contract()["hypothesis"],
        f"{REPORT}; {MARKDOWN}",
        f"confirmed=True; changed decision users across eight already-used windows={result['aggregate']['changed_decision_users_total_across_windows']:,}; mean leaked-minus-label-free delta={result['aggregate']['mean_leaked_minus_label_free_delta']:+.9f}.",
        "Invalidate affected promotion evidence and revert current valid champion to WV2-601.",
        "Stop architecture search for human review; only after approval, preregister a label-free action rule and rebuild affected comparisons.",
        alternatives="Continuing new retrieval or ranking experiments would compound an invalid baseline; merely reporting the magnitude cannot legalize target-dependent decisions.",
        experiment="Reproduce WV3-691 exactly, then remove only target-derived unit_gain from action ordering while leaving labels exclusively in post-decision scoring.",
        reflection="The issue is an evaluation-contract failure, not an ordinary negative model result; the trained historical propensity representation may still be reusable after a clean action repair.",
    )
    print({"trial": TRIAL, "severity": result["severity"], "aggregate": result["aggregate"]}, flush=True)
    return result


if __name__ == "__main__":
    run()
