"""WV3-740: compare reusable historical models under label-free action policies."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import now, read, write
from .warm_v2_engine import save_parquet
from .warm_v3_expert import screening_gate
from .warm_v3_label_free_repair import (
    choose_label_free_swaps,
    unlabeled_proposals,
)
from .warm_v3_residual_admission import INNER, pair_frame, source
from .warm_v3_rich_propensity import FEATURES as RICH_FEATURES, rich_candidate_frame
from .warm_v3_target_aware_admission import (
    FEATURES as PAIR_FEATURES,
    augment_pairs,
    target_table,
)
from .warm_v3_two_swap_rich_propensity import exact_user_deltas


TRIAL = "WV3-740"
CONTRACT = common.REPORT / "WV3-740_CLEAN_MODEL_REPLAY_CONTRACT.json"
REPORT = common.REPORT / "WV3-740_CLEAN_MODEL_REPLAY.json"
MARKDOWN = common.REPORT / "WV3-740_CLEAN_MODEL_REPLAY.md"
ART_ROOT = common.ART / TRIAL
POINT_MODEL = common.ART / "WV3-661" / "MODEL.txt"
LAMBDA_MODEL = common.ART / "WV3-680" / "MODEL.txt"
PAIR_PREFERENCE_MODEL = common.ART / "WV3-620" / "PREFERENCE_MODEL.txt"
PAIR_ACTION_MODEL = common.ART / "WV3-620" / "ACTIONABILITY_MODEL.txt"
POLICIES = [
    "rich_pointwise_control",
    "rich_lambdarank",
    "pointwise_lambdarank_RRF60",
    "target_aware_two_head",
]


def rank_score(frame: pd.DataFrame, score_column: str, output_column: str) -> pd.DataFrame:
    ordered = frame.sort_values(
        ["customer_id", score_column, "rf", "article_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).copy()
    ordered[output_column] = ordered.groupby("customer_id", sort=False).cumcount() + 1
    return ordered[["customer_id", "article_id", output_column]]


def fused_candidate_scores(candidates: pd.DataFrame, point: np.ndarray, lambdarank: np.ndarray) -> pd.DataFrame:
    scores = candidates[["customer_id", "article_id", "rf"]].copy()
    scores["point_score"] = np.asarray(point, dtype=np.float64)
    scores["lambda_score"] = np.asarray(lambdarank, dtype=np.float64)
    point_rank = rank_score(scores, "point_score", "point_rank")
    lambda_rank = rank_score(scores, "lambda_score", "lambda_rank")
    result = scores[["customer_id", "article_id"]].merge(
        point_rank, on=["customer_id", "article_id"], validate="one_to_one"
    ).merge(lambda_rank, on=["customer_id", "article_id"], validate="one_to_one")
    result["ranking_score"] = 1.0 / (60.0 + result.point_rank) + 1.0 / (60.0 + result.lambda_rank)
    return result[["customer_id", "article_id", "ranking_score"]]


def choose_pair_actions(proposals: pd.DataFrame, policy_score: np.ndarray) -> pd.DataFrame:
    required = [
        "customer_id",
        "challenger_article_id",
        "victim_article_id",
        "challenger_rank",
        "victim_rank",
    ]
    if list(proposals.columns) != required:
        raise ValueError("Pair-policy proposals must contain exactly the label-free action fields")
    ranked = proposals.copy()
    ranked["policy_score"] = np.asarray(policy_score, dtype=np.float64)
    ranked = ranked[ranked.policy_score > 0].sort_values(
        [
            "customer_id",
            "policy_score",
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
    return pd.DataFrame(selected, columns=list(ranked.columns) + ["swap_order"])


def summarize_policy(candidates: pd.DataFrame, swaps: pd.DataFrame, meta: dict) -> tuple[dict, pd.DataFrame]:
    users = exact_user_deltas(candidates, swaps)
    beneficial = users.actual_delta > 1e-15
    harmful = users.actual_delta < -1e-15
    neutral = ~(beneficial | harmful)
    delta = float(users.actual_delta.sum() / meta["total_users"])
    baseline = float(meta["baseline_map_population_component"])
    return {
        "selected_users": len(users),
        "selected_swap_rows": len(swaps),
        "users_with_two_swaps": int((users.swap_count == 2).sum()),
        "beneficial_selected_users": int(beneficial.sum()),
        "harmful_selected_users": int(harmful.sum()),
        "neutral_selected_users": int(neutral.sum()),
        "gross_positive_MAP": float(users.loc[beneficial, "actual_delta"].sum() / meta["total_users"]),
        "gross_negative_MAP": float(users.loc[harmful, "actual_delta"].sum() / meta["total_users"]),
        "baseline_MAP@12": baseline,
        "MAP@12": baseline + delta,
        "population_delta": delta,
        "decision_uses_target_or_unit_gain": False,
        "maximum_swaps_per_user": 2,
    }, users


def evaluate_window(name: str, folder: str, models: dict[str, lgb.Booster]) -> dict:
    path, meta = source(folder)
    candidates, details = rich_candidate_frame(meta["cutoff"], rank_min=1)
    matrix = candidates[RICH_FEATURES].to_numpy(np.float32)
    point = models["point"].predict(matrix, num_threads=4)
    lambdarank = models["lambda"].predict(matrix, num_threads=4)
    score_frames = {
        "rich_pointwise_control": candidates[["customer_id", "article_id"]].assign(ranking_score=point),
        "rich_lambdarank": candidates[["customer_id", "article_id"]].assign(ranking_score=lambdarank),
        "pointwise_lambdarank_RRF60": fused_candidate_scores(candidates, point, lambdarank),
    }
    pairs = pair_frame(path, discordant_only=False)
    proposals = unlabeled_proposals(pairs)
    policies = {}
    for policy, scores in score_frames.items():
        swaps = choose_label_free_swaps(proposals, scores)
        summary, users = summarize_policy(candidates, swaps, meta)
        policies[policy] = summary
        root = ART_ROOT / name / policy
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(swaps, root / "swaps.parquet")
        save_parquet(users, root / "users.parquet")

    targets, target_details = target_table(meta["cutoff"])
    augmented = augment_pairs(pairs, targets)
    pair_matrix = augmented[PAIR_FEATURES].to_numpy(np.float32)
    preference = models["preference"].predict(pair_matrix, num_threads=4)
    actionability = models["action"].predict(pair_matrix, num_threads=4)
    pair_score = actionability * (2.0 * preference - 1.0)
    pair_swaps = choose_pair_actions(proposals, pair_score)
    pair_summary, pair_users = summarize_policy(candidates, pair_swaps, meta)
    policies["target_aware_two_head"] = pair_summary
    root = ART_ROOT / name / "target_aware_two_head"
    root.mkdir(parents=True, exist_ok=True)
    save_parquet(pair_swaps, root / "swaps.parquet")
    save_parquet(pair_users, root / "users.parquet")

    expected = read(common.REPORT / "WV3-721_SCREEN.json")["windows"][name]["population_delta"]
    reproduced = policies["rich_pointwise_control"]["population_delta"]
    if abs(reproduced - expected) > 1e-12:
        raise AssertionError(f"WV3-721 control drift in {name}: {reproduced} vs {expected}")
    return {
        "window": name,
        "cutoff": meta["cutoff"],
        "total_users_denominator": meta["total_users"],
        "policies": policies,
        "rich_candidate_input": details,
        "target_pair_input": target_details,
        "WV3_721_control_exactly_reproduced": True,
        "labels_used_only_by_post_decision_evaluation": True,
        "final_week": "not_run",
    }


def contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "role": "inner_only_label_free_policy_comparison_of_reusable_historical_models",
        "hypothesis": (
            "The earlier strict-history LambdaRank or target-aware two-head representations may contain valid "
            "replacement signal that was obscured by target-derived unit_gain in their old action selectors."
        ),
        "frozen_policies": {
            "rich_pointwise_control": "WV3-661 candidate score difference; exact WV3-721 maximum-two-swap control",
            "rich_lambdarank": "WV3-680 candidate score difference; maximum two disjoint positive-score swaps",
            "pointwise_lambdarank_RRF60": "equal RRF60 of within-user WV3-661 and WV3-680 ranks; maximum two disjoint positive-score swaps",
            "target_aware_two_head": "WV3-620 P(actionable)*(2*P(challenger wins)-1), without unit_gain; maximum two disjoint positive-score swaps",
        },
        "model_provenance": "all model fits use only four 2019 historical label windows and exclude target from feature columns",
        "candidate_pool": "unchanged Top50; challengers13-50, victims8-12, ranks1-7 protected",
        "selection": "choose the passing non-control policy with highest four-window inner mean; deterministic policy-name tie break",
        "authorization_gate_vs_WV3_721": "mean>0, >=3/4 nondegrade, worst>=-0.0002; standard WV2 screen must also pass",
        "training": "none",
        "outer": "not run in WV3-740; at most one separately preregistered winner may be exposed",
        "expected_minutes": 20,
        "fallback": "retain WV3-721 if no alternative passes; no score threshold, fusion weight or swap-count rescue",
        "final_week": "2020-09-16 not_run",
    }


def register() -> dict:
    common.setup()
    common.budget(20)
    required = [POINT_MODEL, LAMBDA_MODEL, PAIR_PREFERENCE_MODEL, PAIR_ACTION_MODEL]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("A frozen strict-history model artifact is missing")
    if not CONTRACT.exists():
        write(CONTRACT, contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(
            TRIAL,
            "clean_replay_of_reusable_historical_model_policies",
            contract()["hypothesis"],
            candidate_protocol=contract()["candidate_pool"],
            training_protocol="none; reuse strict-history WV3-661, WV3-680 and WV3-620 model artifacts",
            features={"candidate": RICH_FEATURES, "pair": PAIR_FEATURES},
            params={"frozen_policies": contract()["frozen_policies"], "gate": contract()["authorization_gate_vs_WV3_721"], "final_week": "not_run"},
            expected_minutes=20,
        )
    return read(CONTRACT)


def render(result: dict) -> str:
    lines = [
        "# WV3-740：历史模型的无标签动作重放",
        "",
        "## 结论",
        "",
        f"后续外层候选：`{result['authorized_policy'] or '无'}`；本轮只读内层窗口，未新增训练、未读外层。",
        "",
        "## 术语",
        "",
        "- 点式购买倾向（推荐排序常见建模方式）：对每个用户—商品独立估计购买排序分数；这里复用 WV3-661 的99维丰富特征模型。",
        "- LambdaRank（行业通用学习排序方法）：以同一用户候选组内的相对顺序训练；这里复用 WV3-680，只把模型分数差用于换位，不乘验证标签派生权重。",
        "- 双头替换模型（本项目自定义）：WV3-620 分别估计一对商品是否存在明确胜负、以及挑战商品是否胜出；本轮决策分数为两者乘积，但删除所有 `unit_gain`。",
        "- RRF60（行业常用倒数名次融合，本项目固定常数）：先将两个模型分数转成用户内名次，再相加 `1/(60+名次)`；权重相等，60 沿用冻结基线，不搜索。",
        "- 无标签动作重放（本项目自定义）：模型训练可使用更早历史标签，但当前窗口的动作选择只能读取特征、模型分数、商品标识和名次；真实标签只在动作冻结后评分。",
        "",
        "| 策略 | 四窗平均相对 WV2 | 相对 WV3-721 | 不退化窗口 | 最差相对 WV3-721 | 授权门槛 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for policy, row in result["policy_summary"].items():
        lines.append(
            f"| {policy} | {row['mean_delta_vs_WV2_601']:+.9f} | {row['mean_delta_vs_WV3_721']:+.9f} | "
            f"{row['nondegrade_windows_vs_WV3_721']}/4 | {row['worst_delta_vs_WV3_721']:+.9f} | "
            f"{'pass' if row['passed'] else 'fail'} |"
        )
    lines += [
        "",
        "控制策略必须精确复现 WV3-721。若无替代策略同时通过相对 WV3-721 和标准 WV2 门槛，"
        "则关闭模型复用路线；不依据本表调整权重、阈值或换位数。最终周 `2020-09-16` 保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    common.setup()
    common.budget(20)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-740 must start preregistered and unexposed")
    started = time.perf_counter()
    models = {
        "point": lgb.Booster(model_file=str(POINT_MODEL)),
        "lambda": lgb.Booster(model_file=str(LAMBDA_MODEL)),
        "preference": lgb.Booster(model_file=str(PAIR_PREFERENCE_MODEL)),
        "action": lgb.Booster(model_file=str(PAIR_ACTION_MODEL)),
    }
    windows = {}
    for name, folder in INNER:
        windows[name] = evaluate_window(name, folder, models)
        print(
            {"clean_model_replay": name, "deltas": {policy: windows[name]["policies"][policy]["population_delta"] for policy in POLICIES}},
            flush=True,
        )
    control = {name: row["policies"]["rich_pointwise_control"]["population_delta"] for name, row in windows.items()}
    policy_summary = {}
    for policy in POLICIES:
        values = {name: row["policies"][policy]["population_delta"] for name, row in windows.items()}
        incremental = {name: values[name] - control[name] for name in values}
        standard = screening_gate(values.values())
        mean_incremental = float(np.mean(list(incremental.values())))
        nondegrade = sum(value >= 0 for value in incremental.values())
        worst = min(incremental.values())
        passed = policy != "rich_pointwise_control" and standard["passed"] and mean_incremental > 0 and nondegrade >= 3 and worst >= -0.0002
        policy_summary[policy] = {
            "per_window_delta_vs_WV2_601": values,
            "per_window_delta_vs_WV3_721": incremental,
            "mean_delta_vs_WV2_601": float(np.mean(list(values.values()))),
            "mean_delta_vs_WV3_721": mean_incremental,
            "nondegrade_windows_vs_WV3_721": nondegrade,
            "worst_delta_vs_WV3_721": worst,
            "standard_screening": standard,
            "passed": bool(passed),
        }
    passing = sorted(
        (policy for policy in POLICIES if policy_summary[policy]["passed"]),
        key=lambda policy: (-policy_summary[policy]["mean_delta_vs_WV2_601"], policy),
    )
    authorized = passing[0] if passing else None
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "windows": windows,
        "policy_summary": policy_summary,
        "authorized_policy": authorized,
        "outer_candidate_authorized": authorized is not None,
        "selection_rule": contract()["selection"],
        "new_training": False,
        "outer_exposure": 0,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(REPORT, result)
    MARKDOWN.write_text(render(result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="diagnostic_authorizes_" + authorized if authorized else "diagnostic_rejects_reusable_models",
        inner_evidence={"policy_summary": policy_summary, "authorized_policy": authorized},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(CONTRACT), str(REPORT), str(MARKDOWN)],
        validity="valid_label_free_inner_model_selection",
    )
    mean_text = ", ".join(
        f"{policy}: {policy_summary[policy]['mean_delta_vs_WV3_721']:+.9f}"
        for policy in POLICIES
    )
    common.log(
        f"{TRIAL} clean reusable-model replay",
        "WV3-720 invalidated action selectors, not all strict-history model fits; their clean action behavior had not been compared.",
        contract()["hypothesis"],
        f"{REPORT}; {MARKDOWN}",
        f"authorized_policy={authorized}; mean incremental by policy={{ {mean_text} }}.",
        "Preregister one outer candidate." if authorized else "Retain WV3-721 and close reusable-model replay.",
        "Only the deterministic winning policy may proceed; no fusion or action parameter rescue.",
        alternatives="Training another model before determining whether existing strict-history representations survive a clean decision boundary would add avoidable cost.",
        experiment="Replay four frozen model policies with identical Top50 proposals, exact label-free selection and post-decision AP recomputation.",
        reflection="This separates learned representation value from the invalid target-derived position weighting used by the old selectors.",
    )
    return result


if __name__ == "__main__":
    register()
    run()
