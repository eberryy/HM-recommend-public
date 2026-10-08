"""WV3-750/751: direct three-class challenger-victim relation modeling."""
from __future__ import annotations

import argparse
import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import evidence_id, now, read, write
from .warm_v2_engine import load_parquet, save_parquet
from .warm_v3_clean_model_replay import LAMBDA_MODEL, POINT_MODEL, fused_candidate_scores, summarize_policy
from .warm_v3_expert import screening_gate
from .warm_v3_residual_admission import FEATURES as BASE_PAIR_FEATURES
from .warm_v3_residual_admission import HISTORICAL, INNER, pair_frame, source
from .warm_v3_rich_propensity import FEATURES as RICH_FEATURES, rich_candidate_frame


AUDIT_TRIAL = "WV3-750"
TRIAL = "WV3-751"
AUDIT_CONTRACT = common.REPORT / "WV3-750_DIRECT_RELATION_AUDIT_CONTRACT.json"
AUDIT_REPORT = common.REPORT / "WV3-750_DIRECT_RELATION_AUDIT.json"
AUDIT_MARKDOWN = common.REPORT / "WV3-750_DIRECT_RELATION_AUDIT.md"
CONTRACT = common.REPORT / "WV3-751_DIRECT_RELATION_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-751_SCREEN.json"
SCREEN_MARKDOWN = common.REPORT / "WV3-751_SCREEN.md"
AUDIT_ART = common.ART / AUDIT_TRIAL
MODEL_ROOT = common.ART / TRIAL

DELTA_FEATURES = [f"rich_delta_{name}" for name in RICH_FEATURES]
FEATURES = BASE_PAIR_FEATURES + DELTA_FEATURES
PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "None",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 100,
    "lambda_l2": 10.0,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "seed": 20260910,
    "feature_fraction_seed": 20260910,
    "bagging_seed": 20260910,
    "verbosity": -1,
}
ROUNDS = 120
NEUTRAL_ROWS_PER_USER = 8


def relation_label(pairs: pd.DataFrame) -> np.ndarray:
    benefit = (pairs.challenger_target == 1) & (pairs.victim_target == 0)
    harm = (pairs.challenger_target == 0) & (pairs.victim_target == 1)
    return np.where(benefit, 2, np.where(harm, 0, 1)).astype(np.uint8)


def deterministic_training_sample(pairs: pd.DataFrame) -> pd.DataFrame:
    labels = relation_label(pairs)
    actionable = labels != 1
    neutral = pairs.loc[~actionable].sort_values(
        [
            "customer_id",
            "baseline_gap",
            "challenger_rank",
            "victim_rank",
            "challenger_article_id",
            "victim_article_id",
        ],
        kind="mergesort",
    ).groupby("customer_id", sort=False).head(NEUTRAL_ROWS_PER_USER)
    selected = pd.concat([pairs.loc[actionable], neutral], ignore_index=True)
    selected["relation_label"] = relation_label(selected)
    return selected.sort_values(
        ["customer_id", "challenger_rank", "victim_rank", "challenger_article_id", "victim_article_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def augment_rich_deltas(pairs: pd.DataFrame, rich: pd.DataFrame) -> pd.DataFrame:
    candidate = rich[["customer_id", "article_id", *RICH_FEATURES]].copy()
    challenger = candidate.rename(
        columns={"article_id": "challenger_article_id", **{name: f"rich_challenger_{name}" for name in RICH_FEATURES}}
    )
    victim = candidate.rename(
        columns={"article_id": "victim_article_id", **{name: f"rich_victim_{name}" for name in RICH_FEATURES}}
    )
    result = pairs.merge(
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
    if len(result) != len(pairs):
        raise AssertionError("Rich pair augmentation changed row population")
    delta_values = {}
    for source_name, output_name in zip(RICH_FEATURES, DELTA_FEATURES):
        delta_values[output_name] = (
            result[f"rich_challenger_{source_name}"].to_numpy(np.float32)
            - result[f"rich_victim_{source_name}"].to_numpy(np.float32)
        )
    result = pd.concat([result, pd.DataFrame(delta_values, index=result.index)], axis=1)
    result = result[list(dict.fromkeys([*pairs.columns, *DELTA_FEATURES]))].copy()
    if result[FEATURES].isna().any().any() or not np.isfinite(result[FEATURES].to_numpy(np.float64)).all():
        raise AssertionError("Direct relation model features are incomplete or non-finite")
    return result


def audit_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": AUDIT_TRIAL,
        "role": "strictly_historical_direct_relation_supervision_and_feature_audit",
        "hypothesis": (
            "A direct benefit/neutral/harm target can avoid the compounded errors of the rejected actionability-times-direction "
            "factorization, provided all three classes and rich challenger-victim differences have enough historical support."
        ),
        "label": {"0": "challenger not bought and victim bought (harm)", "1": "equal outcomes (neutral)", "2": "challenger bought and victim not bought (benefit)"},
        "sampling": "retain every historical benefit/harm pair; deterministically keep at most 8 closest-rank neutral pairs per user-window",
        "feature_policy": "23 existing cutoff-safe pair features plus challenger-minus-victim difference for all 99 rich candidate features",
        "gates": {
            "sample_rows_min": 100000,
            "benefit_rows_min": 10000,
            "harm_rows_min": 20000,
            "benefit_and_harm_each_window_min": 2000,
            "varying_rich_delta_features_min": 80,
            "identity_and_finite": True,
        },
        "training": "none in audit",
        "expected_minutes": 20,
        "failure_action": "do not train WV3-751; retain WV3-741",
        "final_week": "2020-09-16 not_run",
    }


def register_audit() -> dict:
    common.setup()
    common.budget(20)
    if not AUDIT_CONTRACT.exists():
        write(AUDIT_CONTRACT, audit_contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == AUDIT_TRIAL for row in state["trials"]):
        common.register(
            AUDIT_TRIAL,
            "direct_three_class_relation_supervision_audit",
            audit_contract()["hypothesis"],
            candidate_protocol="unchanged Top50 pair universe; audit strictly historical 2019 supervision only",
            training_protocol="none",
            features=FEATURES,
            params={"sampling": audit_contract()["sampling"], "gates": audit_contract()["gates"], "final_week": "not_run"},
            expected_minutes=20,
        )
    return read(AUDIT_CONTRACT)


def render_audit(result: dict) -> str:
    lines = [
        "# WV3-750：直接替换关系监督审计",
        "",
        "## 结论",
        "",
        f"WV3-751 训练授权：**{'通过' if result['training_authorized'] else '未通过'}**。本轮没有训练模型，也没有读取2020内层或外层标签。",
        "",
        "## 术语",
        "",
        "- 直接三分类关系（本项目自定义）：对挑战商品—被替换商品这一对样本直接预测“有害 / 中性 / 有益”，而不是分别训练可行动性和胜负方向后相乘。",
        "- 有益样本：历史下一周购买挑战商品但未购买被替换商品；有害样本相反；中性样本是两者都买或都不买。统计单位为用户—挑战商品—被替换商品—历史窗口。",
        "- 丰富差分特征（本项目自定义）：对 WV3-661 的99个用户—商品特征逐列计算“挑战商品值减被替换商品值”；再与既有23个名次/分数关系特征合并，共122维。",
        "- 有界中性抽样（本项目自定义）：保留全部有益和有害样本，每个用户—窗口最多保留8个名次最接近的中性商品对，避免四百多万中性对淹没有方向的监督。",
        "",
        "| 历史窗口 | 抽样行 | 有害 / 中性 / 有益 |",
        "|---|---:|---:|",
    ]
    for row in result["sources"]:
        lines.append(
            f"| {row['window']} | {row['sample_rows']:,} | {row['class_rows']['harm']:,} / "
            f"{row['class_rows']['neutral']:,} / {row['class_rows']['benefit']:,} |"
        )
    lines += [
        "",
        f"总样本 `{result['rows']:,}`；有害 / 中性 / 有益为 `{result['class_rows']['harm']:,}` / "
        f"`{result['class_rows']['neutral']:,}` / `{result['class_rows']['benefit']:,}`；有变化的丰富差分列 "
        f"`{result['varying_rich_delta_features']}/{len(DELTA_FEATURES)}`。",
        "训练数据的最新标签结束时间早于最早2020内层截止日28天；`relation_label` 不在模型特征中。最终周保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def audit() -> dict:
    common.setup()
    common.budget(20)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == AUDIT_TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-750 must start preregistered and unexposed")
    started = time.perf_counter()
    frames = []
    sources = []
    for name, folder in HISTORICAL:
        path, meta = source(folder)
        pairs = deterministic_training_sample(pair_frame(path, discordant_only=False))
        rich, _ = rich_candidate_frame(meta["cutoff"], rank_min=1)
        sampled = augment_rich_deltas(pairs, rich)
        sampled["source_window"] = name
        counts = sampled.relation_label.value_counts().to_dict()
        sources.append(
            {
                "window": name,
                "cutoff": meta["cutoff"],
                "sample_rows": len(sampled),
                "class_rows": {"harm": int(counts.get(0, 0)), "neutral": int(counts.get(1, 0)), "benefit": int(counts.get(2, 0))},
                "latest_label_end_exclusive": meta.get("label_end_exclusive"),
            }
        )
        frames.append(sampled[["source_window", "customer_id", "challenger_article_id", "victim_article_id", "relation_label", *FEATURES]])
        print({"direct_relation_audit": name, "rows": len(sampled), "classes": counts}, flush=True)
        del pairs, rich, sampled
        gc.collect()
    training = pd.concat(frames, ignore_index=True)
    matrix = training[DELTA_FEATURES].to_numpy(np.float32)
    varying = int(((matrix.max(axis=0) - matrix.min(axis=0)) > 0).sum())
    counts = training.relation_label.value_counts().to_dict()
    gates = audit_contract()["gates"]
    passed = bool(
        len(training) >= gates["sample_rows_min"]
        and counts.get(2, 0) >= gates["benefit_rows_min"]
        and counts.get(0, 0) >= gates["harm_rows_min"]
        and all(min(row["class_rows"]["benefit"], row["class_rows"]["harm"]) >= gates["benefit_and_harm_each_window_min"] for row in sources)
        and varying >= gates["varying_rich_delta_features_min"]
        and not training.duplicated(["source_window", "customer_id", "challenger_article_id", "victim_article_id"]).any()
        and np.isfinite(training[FEATURES].to_numpy(np.float64)).all()
        and "relation_label" not in FEATURES
    )
    AUDIT_ART.mkdir(parents=True, exist_ok=True)
    training_path = AUDIT_ART / "training.parquet"
    save_parquet(training, training_path)
    result = {
        "created_at": now(),
        "experiment_id": AUDIT_TRIAL,
        "sources": sources,
        "rows": len(training),
        "class_rows": {"harm": int(counts.get(0, 0)), "neutral": int(counts.get(1, 0)), "benefit": int(counts.get(2, 0))},
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "varying_rich_delta_features": varying,
        "identity_unique": not training.duplicated(["source_window", "customer_id", "challenger_article_id", "victim_article_id"]).any(),
        "finite": bool(np.isfinite(training[FEATURES].to_numpy(np.float64)).all()),
        "target_excluded": "relation_label" not in FEATURES,
        "training_path": str(training_path),
        "latest_label_end_exclusive": "2019-11-27",
        "earliest_inner_cutoff": "2019-12-25",
        "training_authorized": passed,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(AUDIT_REPORT, result)
    AUDIT_MARKDOWN.write_text(render_audit(result), encoding="utf-8")
    common.update(
        AUDIT_TRIAL,
        decision="diagnostic_supports_direct_relation" if passed else "diagnostic_rejects_direct_relation",
        inner_evidence={"rows": len(training), "class_rows": result["class_rows"], "varying": varying, "authorized": passed},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(AUDIT_CONTRACT), str(AUDIT_REPORT), str(AUDIT_MARKDOWN)],
        validity="valid_historical_supervision_audit",
    )
    common.log(
        f"{AUDIT_TRIAL} direct relation supervision audit",
        "WV3-741 improved objective complementarity, while the old factorized actionability-times-direction model failed under a clean selector and most selected swaps remain neutral.",
        audit_contract()["hypothesis"],
        f"{AUDIT_REPORT}; {AUDIT_MARKDOWN}",
        f"rows={len(training)}; classes={result['class_rows']}; varying rich deltas={varying}; authorized={passed}.",
        "Preregister one direct three-class model." if passed else "Retain WV3-741 and do not train this model.",
        "Use fixed class balancing, fixed trees and a clean decision score only if all supervision gates pass.",
        alternatives="A threshold sweep on WV3-741 would tune the same outer-exposed family; direct relation supervision changes the modeled target instead.",
        experiment="Audit all historical actionable pairs plus a deterministic bounded neutral sample, then construct only cutoff-safe rich feature differences.",
        reflection="The audit tests whether a direct replacement target is statistically supported before paying for another model fit.",
    )
    return result


def model_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-741",
        "architecture_family": "direct_three_class_relation_gate_on_clean_rank_fusion",
        "hypothesis": (
            "A single model predicting benefit, neutral and harm from rich challenger-victim differences can gate "
            "the WV3-741 proposal universe more accurately than the rejected two-head factorization."
        ),
        "training": "WV3-750 strict 2019 sample only; inverse-frequency balanced three-class LightGBM",
        "features": "23 cutoff-safe pair features plus 99 rich challenger-minus-victim features",
        "params": PARAMS,
        "rounds": ROUNDS,
        "decision": (
            "first require positive WV3-741 fused score difference; rank pairs by P(benefit)-P(harm); "
            "keep only positive direct score and greedily take at most two disjoint swaps"
        ),
        "candidate_pool": "unchanged Top50; ranks1-7 protected, challengers13-50, victims8-12",
        "inner_gate": "mean incremental vs WV3-741 >0, >=3/4 nondegrade, worst>=-0.0002; standard WV2 screen also passes",
        "outer_policy": "one frozen exposure only if inner passes; no class weight, score threshold, tree or action-count rescue",
        "expected_minutes": 25,
        "fallback": "retain WV3-741 and close direct relation family",
        "final_week": "2020-09-16 not_run",
    }


def register_model() -> dict:
    common.setup()
    common.budget(25)
    if not read(AUDIT_REPORT)["training_authorized"]:
        raise AssertionError("WV3-750 did not authorize direct relation training")
    if not CONTRACT.exists():
        write(CONTRACT, model_contract())
    state = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in state["trials"]):
        common.register(
            TRIAL,
            model_contract()["architecture_family"],
            model_contract()["hypothesis"],
            candidate_protocol=model_contract()["candidate_pool"],
            training_protocol=model_contract()["training"],
            features=FEATURES,
            params={"model": PARAMS, "rounds": ROUNDS, "decision": model_contract()["decision"], "inner_gate": model_contract()["inner_gate"], "final_week": "not_run"},
            expected_minutes=25,
        )
    return read(CONTRACT)


def fit_model(training: pd.DataFrame) -> lgb.Booster:
    labels = training.relation_label.to_numpy(np.uint8)
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    weights = len(labels) / (3.0 * counts[labels])
    dataset = lgb.Dataset(
        training[FEATURES].to_numpy(np.float32),
        label=labels,
        weight=weights,
        feature_name=FEATURES,
        free_raw_data=True,
    )
    return lgb.train(PARAMS, dataset, num_boost_round=ROUNDS)


def choose_direct_actions(proposals: pd.DataFrame, fusion_scores: np.ndarray, relation_scores: np.ndarray) -> pd.DataFrame:
    columns = ["customer_id", "challenger_article_id", "victim_article_id", "challenger_rank", "victim_rank"]
    if list(proposals.columns) != columns:
        raise ValueError("Direct relation decisions require exactly the label-free pair fields")
    ranked = proposals.copy()
    ranked["fusion_score_difference"] = np.asarray(fusion_scores, dtype=np.float64)
    ranked["direct_relation_score"] = np.asarray(relation_scores, dtype=np.float64)
    ranked = ranked[(ranked.fusion_score_difference > 0) & (ranked.direct_relation_score > 0)].sort_values(
        ["customer_id", "direct_relation_score", "fusion_score_difference", "victim_rank", "challenger_rank", "challenger_article_id", "victim_article_id"],
        ascending=[True, False, False, False, True, True, True],
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


def evaluate_window(name: str, folder: str, model: lgb.Booster, point: lgb.Booster, lambdarank: lgb.Booster) -> dict:
    path, meta = source(folder)
    candidates, details = rich_candidate_frame(meta["cutoff"], rank_min=1)
    matrix = candidates[RICH_FEATURES].to_numpy(np.float32)
    fusion = fused_candidate_scores(
        candidates,
        point.predict(matrix, num_threads=4),
        lambdarank.predict(matrix, num_threads=4),
    )
    pairs = pair_frame(path, discordant_only=False)
    proposals = pairs[["customer_id", "challenger_article_id", "victim_article_id", "challenger_rank", "victim_rank"]].copy()
    rich_pairs = augment_rich_deltas(pairs, candidates)
    probabilities = model.predict(rich_pairs[FEATURES].to_numpy(np.float32), num_threads=4)
    relation_score = probabilities[:, 2] - probabilities[:, 0]
    challenger = fusion.rename(columns={"article_id": "challenger_article_id", "ranking_score": "challenger_fusion"})
    victim = fusion.rename(columns={"article_id": "victim_article_id", "ranking_score": "victim_fusion"})
    fusion_pairs = proposals.merge(challenger, on=["customer_id", "challenger_article_id"], validate="many_to_one").merge(victim, on=["customer_id", "victim_article_id"], validate="many_to_one")
    fusion_difference = fusion_pairs.challenger_fusion.to_numpy() - fusion_pairs.victim_fusion.to_numpy()
    swaps = choose_direct_actions(proposals, fusion_difference, relation_score)
    summary, users = summarize_policy(candidates, swaps, meta)
    control = read(common.REPORT / "WV3-740_CLEAN_MODEL_REPLAY.json")["windows"][name]["policies"]["pointwise_lambdarank_RRF60"]
    summary.update(
        {
            "window": name,
            "cutoff": meta["cutoff"],
            "total_users_denominator": meta["total_users"],
            "same_pool_control_MAP@12": control["MAP@12"],
            "incremental_delta_vs_WV3_741_inner_policy": summary["MAP@12"] - control["MAP@12"],
            "rich_candidate_input": details,
            "decision_uses_target_or_unit_gain": False,
            "labels_joined_after_actions_frozen": True,
            "final_week": "not_run",
        }
    )
    root = MODEL_ROOT / name / "inner"
    root.mkdir(parents=True, exist_ok=True)
    save_parquet(swaps, root / "swaps.parquet")
    save_parquet(users, root / "users.parquet")
    write(root / "REVIEW.json", summary)
    return summary


def render_screen(result: dict) -> str:
    gate = result["incremental_vs_WV3_741"]
    lines = [
        "# WV3-751：直接三分类替换关系模型",
        "",
        "## 结论",
        "",
        f"内层门槛：**{'通过' if result['passed'] else '未通过'}**；相对 WV3-741 内层策略平均 `{gate['mean_delta_vs_WV3_741']:+.9f}`。",
        "",
        "## 固定方法与术语",
        "",
        "- 三分类关系模型（本项目自定义）：LightGBM 同时输出有害、中性、有益三类概率；决策分数固定为 `P(有益)-P(有害)`，不是未来标签或 AP 权重。",
        "- 融合提案门（本项目自定义）：只允许 WV3-741 的点式/LambdaRank RRF60 分数认为挑战商品优于被替换商品的商品对进入关系模型选择范围。",
        "- 类别反频率权重（机器学习常见）：每个历史类别总权重相同，避免中性样本数量更大；权重只由历史训练标签频数计算，不进入当前窗口动作。",
        "- 精确换位评测：动作完全冻结后才读取当前窗口标签，逐用户重建 Top12 并计算 AP@12；分母保持全部真值用户。",
        "",
        "| 内层窗口 | 选择用户 | 有益 / 有害 / 中性 | 相对 WV3-741 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['selected_users']:,} | {row['beneficial_selected_users']:,} / "
            f"{row['harmful_selected_users']:,} / {row['neutral_selected_users']:,} | "
            f"{row['incremental_delta_vs_WV3_741_inner_policy']:+.9f} |"
        )
    lines += [
        "",
        "失败后不调整类别权重、决策阈值、树参数或换位数；只有内层相对门槛和标准门槛同时通过才允许一次外层。最终周保持 `not_run`。",
        "",
    ]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(25)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    if entry["decision"] != "preregistered" or entry["outer_exposures"] != 0:
        raise AssertionError("WV3-751 must start preregistered and unexposed")
    started = time.perf_counter()
    training = load_parquet(read(AUDIT_REPORT)["training_path"])
    model = fit_model(training)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_ROOT / "MODEL.txt"
    model.save_model(str(model_path))
    write(
        MODEL_ROOT / "MODEL.json",
        {
            "experiment_id": TRIAL,
            "model": evidence_id(model_path, reason="explicit_registry_evidence"),
            "features": FEATURES,
            "params": PARAMS,
            "rounds": ROUNDS,
            "training_audit": str(AUDIT_REPORT),
            "outer_labels_used": False,
            "final_week": "not_run",
        },
    )
    del training
    gc.collect()
    point = lgb.Booster(model_file=str(POINT_MODEL))
    lambdarank = lgb.Booster(model_file=str(LAMBDA_MODEL))
    windows = {}
    for name, folder in INNER:
        windows[name] = evaluate_window(name, folder, model, point, lambdarank)
        print({"direct_relation_inner": name, "incremental": windows[name]["incremental_delta_vs_WV3_741_inner_policy"]}, flush=True)
        gc.collect()
    increments = {name: row["incremental_delta_vs_WV3_741_inner_policy"] for name, row in windows.items()}
    incremental = {
        "per_window_delta_vs_WV3_741": increments,
        "mean_delta_vs_WV3_741": float(np.mean(list(increments.values()))),
        "nondegrade_windows_vs_WV3_741": sum(value >= 0 for value in increments.values()),
        "worst_delta_vs_WV3_741": min(increments.values()),
    }
    incremental["passed"] = bool(
        incremental["mean_delta_vs_WV3_741"] > 0
        and incremental["nondegrade_windows_vs_WV3_741"] >= 3
        and incremental["worst_delta_vs_WV3_741"] >= -0.0002
    )
    standard = screening_gate(
        row["MAP@12"] - row["baseline_MAP@12"] for row in windows.values()
    )
    passed = bool(incremental["passed"] and standard["passed"])
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "windows": windows,
        "incremental_vs_WV3_741": incremental,
        "standard_vs_WV2_601": standard,
        "passed": passed,
        "training_rows": read(AUDIT_REPORT)["rows"],
        "new_training": True,
        "candidate_pool_changed": False,
        "outer_exposure": 0,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(SCREEN_REPORT, result)
    SCREEN_MARKDOWN.write_text(render_screen(result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="inner_pass" if passed else "reject_inner",
        inner_evidence={"incremental": incremental, "standard": standard, "passed": passed},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(CONTRACT), str(SCREEN_REPORT), str(SCREEN_MARKDOWN), str(MODEL_ROOT / "MODEL.json")],
        validity="valid_label_free_inner_screen",
    )
    common.log(
        f"{TRIAL} direct relation inner screen",
        "WV3-750 found sufficient strict-history support for a direct benefit/neutral/harm target after the factorized two-head model failed clean replay.",
        model_contract()["hypothesis"],
        f"{SCREEN_REPORT}; {SCREEN_MARKDOWN}",
        f"mean incremental={incremental['mean_delta_vs_WV3_741']:+.9f}; nondegrade={incremental['nondegrade_windows_vs_WV3_741']}/4; worst={incremental['worst_delta_vs_WV3_741']:+.9f}; pass={passed}.",
        "Authorize one frozen outer confirmation." if passed else "Reject and retain WV3-741.",
        "Expose once only if both gates pass; otherwise close direct relation modeling.",
        alternatives="No threshold or class-weight rescue; those would use the same inner outcomes as a tuning surface.",
        experiment="Fit one fixed three-class model on the audited historical sample, gate the frozen WV3-741 proposal universe, and recompute exact AP after decisions.",
        reflection="This directly tests whether relation supervision can reduce harmful swaps without relying on future-label position weights.",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register-audit", "audit", "register-model", "screen"])
    command = parser.parse_args().command
    {"register-audit": register_audit, "audit": audit, "register-model": register_model, "screen": screen}[command]()
