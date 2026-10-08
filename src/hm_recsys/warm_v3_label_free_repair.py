"""WV3-721: label-free repair of the frozen rich-propensity tail swaps."""
from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import save_parquet
from .warm_v3_expert import screening_gate
from .warm_v3_residual_admission import INNER, OUTER, pair_frame, source
from .warm_v3_rich_propensity import FEATURES, rich_candidate_frame
from .warm_v3_two_swap_rich_propensity import exact_user_deltas


TRIAL = "WV3-721"
CONTRACT = common.REPORT / "WV3-721_LABEL_FREE_REPAIR_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-721_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-721_SCREEN.md"
OUTER_REPORT = common.REPORT / "WV3-721_CORRECTED_OUTER_REPLAY.json"
OUTER_MARKDOWN = common.REPORT / "WV3-721_FINAL.md"
MODEL_PATH = common.ART / "WV3-661" / "MODEL.txt"
ART_ROOT = common.ART / TRIAL

PROPOSAL_COLUMNS = [
    "customer_id",
    "challenger_article_id",
    "victim_article_id",
    "challenger_rank",
    "victim_rank",
]
SCORE_COLUMNS = ["customer_id", "article_id", "ranking_score"]
FORBIDDEN_DECISION_COLUMNS = {
    "target",
    "challenger_target",
    "victim_target",
    "truth_count",
    "unit_gain",
    "positives_before",
    "later_positive_inverse",
    "actual_delta",
}


def unlabeled_proposals(pair_rows: pd.DataFrame) -> pd.DataFrame:
    """Project the evaluator's pair table onto deployable decision fields only."""
    missing = set(PROPOSAL_COLUMNS) - set(pair_rows.columns)
    if missing:
        raise ValueError(f"Missing proposal columns: {sorted(missing)}")
    return pair_rows[PROPOSAL_COLUMNS].copy()


def choose_label_free_swaps(proposals: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Select at most two disjoint swaps using only identities, ranks and model scores."""
    if list(proposals.columns) != PROPOSAL_COLUMNS:
        raise ValueError("Decision proposals must contain exactly the registered label-free columns")
    if list(scores.columns) != SCORE_COLUMNS:
        raise ValueError("Decision scores must contain exactly customer_id, article_id and ranking_score")
    if FORBIDDEN_DECISION_COLUMNS & (set(proposals.columns) | set(scores.columns)):
        raise AssertionError("Future-label-derived fields reached action selection")

    challenger = scores.rename(
        columns={"article_id": "challenger_article_id", "ranking_score": "challenger_score"}
    )
    victim = scores.rename(
        columns={"article_id": "victim_article_id", "ranking_score": "victim_score"}
    )
    ranked = proposals.merge(
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
    if not ranked[["challenger_score", "victim_score"]].notna().all().all():
        raise AssertionError("A proposal item is missing its frozen model score")
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
    selected: list[dict] = []
    for _, group in ranked.groupby("customer_id", sort=False):
        used_challengers: set[str] = set()
        used_victims: set[str] = set()
        for row in group.itertuples(index=False):
            if row.challenger_article_id in used_challengers or row.victim_article_id in used_victims:
                continue
            record = row._asdict()
            record["swap_order"] = len(used_challengers) + 1
            selected.append(record)
            used_challengers.add(row.challenger_article_id)
            used_victims.add(row.victim_article_id)
            if len(used_challengers) == 2:
                break
    result = pd.DataFrame(selected, columns=list(ranked.columns) + ["swap_order"])
    if FORBIDDEN_DECISION_COLUMNS & set(result.columns):
        raise AssertionError("Selected actions contain a future-label-derived field")
    return result


def decision_keys(frame: pd.DataFrame) -> list[tuple[str, str, str, int]]:
    return list(
        zip(
            frame.customer_id.astype(str),
            frame.challenger_article_id.astype(str),
            frame.victim_article_id.astype(str),
            frame.swap_order.astype(int),
        )
    )


def decision_invariance_audit(pair_rows: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """Verify that mutating evaluator-only label fields cannot alter selected actions."""
    original = choose_label_free_swaps(unlabeled_proposals(pair_rows), scores)
    mutated = pair_rows.copy()
    for column in FORBIDDEN_DECISION_COLUMNS & set(mutated.columns):
        if pd.api.types.is_numeric_dtype(mutated[column]):
            mutated[column] = np.arange(len(mutated), dtype=np.float64)[::-1] + 12345.0
        else:
            mutated[column] = "mutated_future_label"
    changed = choose_label_free_swaps(unlabeled_proposals(mutated), scores)
    identical = decision_keys(original) == decision_keys(changed)
    if not identical:
        raise AssertionError("Action choices changed after evaluator-only labels were mutated")
    return {
        "passed": True,
        "decision_input_columns": PROPOSAL_COLUMNS + SCORE_COLUMNS,
        "forbidden_columns_absent": not bool(
            FORBIDDEN_DECISION_COLUMNS & set(original.columns)
        ),
        "choices_identical_after_label_mutation": identical,
        "selected_action_rows": len(original),
    }


def contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent_valid_champion": "WV2-601",
        "reused_model_artifact": "WV3-661/MODEL.txt",
        "architecture_family": "label_free_rich_propensity_maximum_two_tail_swaps",
        "hypothesis": (
            "The historically trained WV3-661 model remains useful after repairing the action boundary: "
            "a policy using only item identities, baseline ranks and frozen model scores can improve all-window "
            "development MAP@12 without target-derived decision inputs."
        ),
        "candidate_pool": "unchanged original Top50; challengers ranks13-50, victims ranks8-12",
        "model": "exact frozen WV3-661 LightGBM model; no fit or parameter change",
        "decision_rule": (
            "keep proposals with challenger_score-victim_score>0; sort by score difference, then victim rank "
            "descending, challenger rank ascending and item identifiers; greedily take at most two disjoint swaps"
        ),
        "decision_input_columns": PROPOSAL_COLUMNS + SCORE_COLUMNS,
        "forbidden_decision_columns": sorted(FORBIDDEN_DECISION_COLUMNS),
        "evaluation_boundary": "join targets and truth_count only after the full action set is frozen",
        "inner_gate": "mean delta vs WV2-601 >=0.0001; >=3/4 positive windows; worst >=-0.0005",
        "outer_gate": "mean delta vs WV2-601 >0; >=3/4 nondegrade windows; worst >=-0.0005",
        "evidence_classification": (
            "post_audit_corrected_development_replay; WV3-720 already displayed these 2020 outcomes, "
            "so this is not a blind or independent confirmation"
        ),
        "success_action": "set WV3-721 as valid development champion and continue bounded architecture search",
        "failure_action": "retain WV2-601 and abandon the repaired two-swap policy",
        "no_rescue": "no threshold, swap-count, tie-order or window-specific adjustment after replay",
        "expected_minutes": 15,
        "final_week": "2020-09-16 not_run",
    }


def register() -> dict:
    common.setup()
    common.budget(15)
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(
            TRIAL,
            contract()["architecture_family"],
            contract()["hypothesis"],
            candidate_protocol=contract()["candidate_pool"],
            training_protocol=contract()["model"],
            features=FEATURES,
            params={
                "decision_rule": contract()["decision_rule"],
                "decision_input_columns": contract()["decision_input_columns"],
                "forbidden_decision_columns": contract()["forbidden_decision_columns"],
                "inner_gate": contract()["inner_gate"],
                "outer_gate": contract()["outer_gate"],
                "evidence_classification": contract()["evidence_classification"],
                "final_week": "not_run",
            },
            expected_minutes=15,
        )
    state = read(common.REGISTRY)
    state["status"] = "running_human_approved_label_free_repair"
    state["repair_authorization"] = {
        "approved": True,
        "scope": "WV3-721 label-free action repair followed by bounded continued research",
        "deadline_utc": "2026-09-10T18:19:04.0890781+00:00",
        "final_week": "not_run",
    }
    write(common.REGISTRY, state)
    return read(CONTRACT)


def expected_safe_delta(stage: str, name: str) -> float:
    audit = read(common.REPORT / "WV3-720_UNIT_GAIN_LEAKAGE_AUDIT.json")
    return float(audit[stage][name]["label_free_mechanical_policy_population_delta"])


def evaluate(folder: str, name: str, model: lgb.Booster, stage: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    path, meta = source(folder)
    started = time.perf_counter()
    candidates, details = rich_candidate_frame(meta["cutoff"], rank_min=1)
    values = model.predict(candidates[FEATURES].to_numpy(np.float32), num_threads=4)
    scores = candidates[["customer_id", "article_id"]].copy()
    scores["ranking_score"] = values
    pairs = pair_frame(path, discordant_only=False)

    invariance = decision_invariance_audit(pairs, scores)
    swaps = choose_label_free_swaps(unlabeled_proposals(pairs), scores)
    users = exact_user_deltas(candidates, swaps)
    delta = float(users.actual_delta.sum() / meta["total_users"])
    expected = expected_safe_delta(stage, name)
    if abs(delta - expected) > 1e-12:
        raise AssertionError(f"Corrected replay drift for {name}: {delta} vs {expected}")
    beneficial = users.actual_delta > 1e-15
    harmful = users.actual_delta < -1e-15
    neutral = ~(beneficial | harmful)
    baseline = float(meta["baseline_map_population_component"])
    result = {
        "window": name,
        "cutoff": meta["cutoff"],
        "role": stage,
        "evidence_classification": "post_audit_corrected_development_replay",
        "total_users_denominator": meta["total_users"],
        "selected_users": len(users),
        "selected_swap_rows": len(swaps),
        "users_with_two_swaps": int((users.swap_count == 2).sum()),
        "beneficial_selected_users": int(beneficial.sum()),
        "harmful_selected_users": int(harmful.sum()),
        "neutral_selected_users": int(neutral.sum()),
        "baseline_MAP@12": baseline,
        "MAP@12": baseline + delta,
        "population_delta": delta,
        "decision_invariance": invariance,
        "matches_WV3_720_label_free_replay": True,
        "protected_ranks1_7_changes": 0,
        "maximum_swaps_per_user": 2,
        "evaluation_labels_joined_after_decision": True,
        "candidate_input": details,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    return result, swaps, users


def render(stage_name: str, result: dict) -> str:
    summary = result["screening"] if stage_name == "inner" else result["summary_vs_WV2_601"]
    lines = [
        f"# WV3-721 无标签动作修复：{'内层重放' if stage_name == 'inner' else '四窗更正结论'}",
        "",
        "## 结论",
        "",
        (
            f"门槛：**{'pass' if result['passed'] else 'fail'}**。四窗平均 MAP@12 为 "
            f"`{result['mean_MAP']:.12f}`，相对有效基线 WV2-601 为 "
            f"`{result['mean_delta_vs_WV2_601']:+.12f}`。"
        ),
        "",
        "这是 **WV3-720 审计后的更正开发集重放**：该策略结果已经在泄漏审计中被观察过，"
        "因此不能称为盲测或独立确认。它只修复决策合同，不重训模型、不改变候选池。",
        "",
        "## 术语",
        "",
        "- 无标签动作策略（本项目自定义）：选择挑战商品和被替换商品时，只读取用户/商品标识、基线名次和冻结模型分数；验证周是否购买及其派生量不进入选择。",
        "- 决策不变性（软件与评测审计概念）：任意改写评测专用标签字段后，所选用户—挑战商品—被替换商品及换位次序必须完全不变。",
        "- 更正开发集重放（本项目自定义）：在错误被发现后，用预先固定的合法代码重新计算已经看过的开发窗口；可纠正无效结果，但不提供新的盲测证据。",
        "- population delta（本项目沿用指标）：实验 MAP@12 减 WV2-601 MAP@12，分母为该窗口全部有真值用户，包括候选未覆盖用户。",
        "- 最多两次尾部换位（本项目自定义）：只允许基线第13—50名商品挑战第8—12名，每位用户最多选择两组商品互不重复的换位，第1—7名不变。",
        "",
        "## 四窗结果",
        "",
        "| 窗口 | WV2-601 MAP@12 | WV3-721 MAP@12 | 差值 | 选择用户 | 有益 / 有害 / 中性 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['baseline_MAP@12']:.9f} | {row['MAP@12']:.9f} | "
            f"{row['population_delta']:+.9f} | {row['selected_users']:,} | "
            f"{row['beneficial_selected_users']:,} / {row['harmful_selected_users']:,} / {row['neutral_selected_users']:,} |"
        )
    lines += [
        "",
        "所有窗口均通过标签字段改写不影响动作的检查，并与 WV3-720 中无标签机械重放的数值精确一致。",
        f"门槛明细：`{summary}`。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(15)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-721 inner replay must start from the preregistered, unexposed state")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    started = time.perf_counter()
    windows = {}
    for name, folder in INNER:
        row, swaps, users = evaluate(folder, name, model, "inner")
        windows[name] = row
        root = ART_ROOT / name / "inner"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(swaps, root / "swaps.parquet")
        save_parquet(users, root / "users.parquet")
        write(root / "REVIEW.json", row)
        print({"label_free_inner": name, "delta": row["population_delta"]}, flush=True)
    gate = screening_gate(row["population_delta"] for row in windows.values())
    mean_map = float(np.mean([row["MAP@12"] for row in windows.values()]))
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "evidence_classification": "post_audit_corrected_development_replay",
        "windows": windows,
        "screening": gate,
        "passed": gate["passed"],
        "mean_MAP": mean_map,
        "mean_delta_vs_WV2_601": gate["mean_population_delta"],
        "new_training": False,
        "candidate_pool_changed": False,
        "all_decision_invariance_checks_passed": all(
            row["decision_invariance"]["passed"] for row in windows.values()
        ),
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render("inner", result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="inner_pass" if result["passed"] else "reject_inner",
        inner_evidence={"screening": gate, "decision_invariance": True, "passed": result["passed"]},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(CONTRACT), str(SCREEN_REPORT), str(SCREEN_MARKDOWN)],
        validity="valid_label_free_corrected_replay",
    )
    common.log(
        f"{TRIAL} inner corrected replay",
        "WV3-720 invalidated target-dependent swap ordering but showed a stable label-free mechanical comparison.",
        contract()["hypothesis"],
        f"{CONTRACT}; {SCREEN_REPORT}; {SCREEN_MARKDOWN}",
        f"mean delta={gate['mean_population_delta']:+.9f}; positive={gate['positive_windows']}/4; worst={gate['worst_delta']:+.9f}; pass={result['passed']}.",
        "Record one corrected outer replay." if result["passed"] else "Reject repair and retain WV2-601.",
        "If the frozen gate passes, expose the already-seen outer policy once and preserve the non-blind classification.",
        alternatives="Do not tune score thresholds, tie order or swap count from the already-observed WV3-720 values.",
        experiment="Freeze a pure decision interface, mutate evaluator-only labels as an invariance check, then score exact MAP only after actions are fixed.",
        reflection="A valid software boundary, not the small numerical difference from the leaked run, is the acceptance condition for the repair.",
    )
    print({"trial": TRIAL, "stage": "inner", "gate": gate}, flush=True)
    return result


def confirm() -> dict:
    common.setup()
    common.budget(15)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "inner_pass" or entry["outer_exposures"] != 1:
        raise AssertionError("WV3-721 corrected outer replay requires one recorded exposure after inner pass")
    if not read(SCREEN_REPORT)["passed"]:
        raise AssertionError("WV3-721 inner gate did not pass")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    started = time.perf_counter()
    windows = {}
    for name, folder in OUTER:
        row, swaps, users = evaluate(folder, name, model, "outer")
        windows[name] = row
        root = ART_ROOT / name / "outer"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(swaps, root / "swaps.parquet")
        save_parquet(users, root / "users.parquet")
        write(root / "REVIEW.json", row)
        print({"label_free_outer": name, "delta": row["population_delta"]}, flush=True)
    summary = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    passed = bool(summary["stable"])
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "evidence_classification": "post_audit_corrected_development_replay_not_blind",
        "independent_confirmation": False,
        "windows": windows,
        "summary_vs_WV2_601": summary,
        "passed": passed,
        "mean_MAP": summary["mean_MAP"],
        "mean_delta_vs_WV2_601": summary["delta_vs_WV2_601"],
        "new_training": False,
        "candidate_pool_changed": False,
        "all_decision_invariance_checks_passed": all(
            row["decision_invariance"]["passed"] for row in windows.values()
        ),
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(OUTER_REPORT, result)
    OUTER_MARKDOWN.write_text(render("outer", result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="valid_development_champion" if passed else "reject_corrected_replay",
        outer_MAP_by_window=summary["per_window_MAP"],
        mean_MAP=summary["mean_MAP"],
        delta_vs_WV2_601=summary["delta_vs_WV2_601"],
        nondegrade_windows=summary["nondegrade_windows"],
        worst_delta=summary["worst_delta"],
        runtime=result["runtime_seconds"],
        artifact_paths=entry["artifact_paths"] + [str(OUTER_REPORT), str(OUTER_MARKDOWN)],
        validity="valid_development_replay_not_independent_confirmation",
    )
    state = read(common.REGISTRY)
    if passed:
        state["current_champion"] = TRIAL
        state["current_valid_champion"] = TRIAL
        state["best_stable"] = {
            "experiment_id": TRIAL,
            "mean_MAP": summary["mean_MAP"],
            "delta_vs_WV2_601": summary["delta_vs_WV2_601"],
            "nondegrade_windows": summary["nondegrade_windows"],
            "worst_delta": summary["worst_delta"],
            "evidence_classification": result["evidence_classification"],
        }
        state["status"] = "running_valid_repair_below_target"
    else:
        state["status"] = "running_repair_rejected"
    write(common.REGISTRY, state)
    common.log(
        f"{TRIAL} corrected outer replay",
        "The preregistered pure action boundary passed the inner gate and all label-mutation invariance checks.",
        contract()["hypothesis"],
        f"{OUTER_REPORT}; {OUTER_MARKDOWN}",
        f"mean MAP={summary['mean_MAP']:.9f}; delta={summary['delta_vs_WV2_601']:+.9f}; nondegrade={summary['nondegrade_windows']}/4; worst={summary['worst_delta']:+.9f}; pass={passed}.",
        "Use WV3-721 as the valid development champion while retaining the non-blind evidence label." if passed else "Retain WV2-601.",
        "Continue only with a new mechanism whose decision inputs are audited before scoring.",
        alternatives="No result-driven rescue of this repaired policy; subsequent trials must compare against the same clean WV3-721 action boundary.",
        experiment="One registered corrected replay of the already-observed outer windows; the exposure count is recorded even though this is not fresh confirmation.",
        reflection="The repaired policy can restore a valid development baseline, but a future untouched period would still be required for independent confirmation.",
    )
    print({"trial": TRIAL, "stage": "outer", "summary": summary, "passed": passed}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "screen", "confirm"])
    globals()[parser.parse_args().command]()
