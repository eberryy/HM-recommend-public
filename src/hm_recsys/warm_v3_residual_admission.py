"""WV3-601: historical multi-window residual admission for the frozen Warm Top12."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import evidence_id, now, read, write
from .warm_v2_engine import connection, literal, save_parquet
from .warm_v3_expert import screening_gate


TRIAL = "WV3-601"
CONTRACT = common.REPORT / "WV3-601_RESIDUAL_ADMISSION_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-601_SCREEN.json"
OUTER_REPORT = common.REPORT / "WV3-601_OUTER.json"
MODEL_ROOT = common.ART / TRIAL

HISTORICAL = (
    ("history_20190220", "2019_outer_2019-02-20"),
    ("history_20190522", "2019_outer_2019-05-22"),
    ("history_20190821", "2019_outer_2019-08-21"),
    ("history_20191120", "2019_outer_2019-11-20"),
)
INNER = (
    ("winter_20200122", "2020_inner_2019-12-25"),
    ("spring_20200318", "2020_inner_2020-02-19"),
    ("early_summer_20200624", "2020_inner_2020-05-27"),
    ("late_summer_20200819", "2020_inner_2020-07-22"),
)
OUTER = (
    ("winter_20200122", "2020_outer_2020-01-22"),
    ("spring_20200318", "2020_outer_2020-03-18"),
    ("early_summer_20200624", "2020_outer_2020-06-24"),
    ("late_summer_20200819", "2020_outer_2020-08-19"),
)

FEATURES = [
    "log_history_events",
    "log_unique_items",
    "recency_days",
    "victim_rank",
    "challenger_rank",
    "baseline_gap",
    "candidate_rank_advantage",
    "e0_rank_advantage",
    "e1_rank_advantage",
    "base_score_delta",
    "bpr_score_delta",
    "latent_score_delta",
    "rrf_score_delta",
    "challenger_candidate_rank",
    "victim_candidate_rank",
    "challenger_e0_rank",
    "victim_e0_rank",
    "challenger_e1_rank",
    "victim_e1_rank",
    "challenger_bpr_missing",
    "victim_bpr_missing",
    "expert_preference_count",
    "rank_disagreement_delta",
]

PARAMS = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
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
ROUNDS = 80
THRESHOLD = 0.5


def source(folder: str) -> tuple[Path, dict]:
    root = common.ART / "gate_data" / folder
    meta = read(root / "DATA.json")
    assert meta["final_week"] == "not_run"
    path = Path(meta["ranks_path"])
    assert path.is_file()
    return path, meta


def pair_frame(path: Path, discordant_only: bool, challenger_high: int = 50) -> pd.DataFrame:
    assert challenger_high in (50, 100)
    discordant = "AND c.target<>v.target" if discordant_only else ""
    with connection() as con:
        frame = con.execute(
            f"""WITH top50 AS (
                SELECT customer_id,article_id,ap_rf baseline_rank,target,truth_count,
                    user_history_events_12w,user_unique_items_12w,user_days_since_last_purchase,
                    candidate_rank,r0,r1,score_base,score_bpr,latent,missing,rrf_score,
                    coalesce(sum(target) OVER(PARTITION BY customer_id ORDER BY ap_rf
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) positives_before,
                    coalesce(sum(CASE WHEN ap_rf<=12 AND target=1 THEN 1.0/ap_rf ELSE 0 END)
                        OVER(PARTITION BY customer_id ORDER BY ap_rf
                        ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING),0) later_positive_inverse
                FROM read_parquet({literal(path)}) WHERE ap_rf<={challenger_high}
            ), pairs AS (
                SELECT c.customer_id,c.article_id challenger_article_id,v.article_id victim_article_id,
                    c.target challenger_target,v.target victim_target,
                    c.baseline_rank challenger_rank,v.baseline_rank victim_rank,
                    ((v.positives_before+1.0)/v.baseline_rank+v.later_positive_inverse)
                        /greatest(least(v.truth_count,12),1) unit_gain,
                    ln(1+c.user_history_events_12w) log_history_events,
                    ln(1+c.user_unique_items_12w) log_unique_items,
                    c.user_days_since_last_purchase recency_days,
                    c.baseline_rank-v.baseline_rank baseline_gap,
                    v.candidate_rank-c.candidate_rank candidate_rank_advantage,
                    v.r0-c.r0 e0_rank_advantage,
                    v.r1-c.r1 e1_rank_advantage,
                    c.score_base-v.score_base base_score_delta,
                    c.score_bpr-v.score_bpr bpr_score_delta,
                    c.latent-v.latent latent_score_delta,
                    c.rrf_score-v.rrf_score rrf_score_delta,
                    c.candidate_rank challenger_candidate_rank,v.candidate_rank victim_candidate_rank,
                    c.r0 challenger_e0_rank,v.r0 victim_e0_rank,
                    c.r1 challenger_e1_rank,v.r1 victim_e1_rank,
                    c.missing challenger_bpr_missing,v.missing victim_bpr_missing,
                    CAST(c.candidate_rank<v.candidate_rank AS INTEGER)
                      +CAST(c.r0<v.r0 AS INTEGER)+CAST(c.r1<v.r1 AS INTEGER) expert_preference_count,
                    abs(c.r0-c.r1)-abs(v.r0-v.r1) rank_disagreement_delta
                FROM top50 c JOIN top50 v USING(customer_id)
                WHERE c.user_history_events_12w>0
                  AND c.baseline_rank BETWEEN 13 AND {challenger_high}
                  AND v.baseline_rank BETWEEN 8 AND 12 {discordant}
            ) SELECT * FROM pairs ORDER BY customer_id,challenger_rank,victim_rank,
                challenger_article_id,victim_article_id"""
        ).fetchdf()
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if discordant_only:
        assert (frame.challenger_target != frame.victim_target).all()
    assert (frame.unit_gain > 0).all()
    return frame


def fit_model(training: pd.DataFrame) -> lgb.Booster:
    labels = training.challenger_target.to_numpy(np.uint8)
    assert 0 < labels.sum() < len(labels)
    dataset = lgb.Dataset(
        training[FEATURES].to_numpy(np.float32),
        label=labels,
        weight=training.unit_gain.to_numpy(np.float64),
        feature_name=FEATURES,
        free_raw_data=True,
    )
    return lgb.train(PARAMS, dataset, num_boost_round=ROUNDS)


def choose_decisions(pairs: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    chosen = pairs[
        [
            "customer_id",
            "challenger_article_id",
            "victim_article_id",
            "challenger_target",
            "victim_target",
            "challenger_rank",
            "victim_rank",
            "unit_gain",
        ]
    ].copy()
    chosen["admission_probability"] = np.asarray(scores, dtype=np.float64)
    chosen["predicted_delta"] = (2.0 * chosen.admission_probability - 1.0) * chosen.unit_gain
    chosen = chosen[chosen.admission_probability > THRESHOLD]
    chosen = chosen.sort_values(
        [
            "customer_id",
            "predicted_delta",
            "admission_probability",
            "victim_rank",
            "challenger_rank",
            "challenger_article_id",
            "victim_article_id",
        ],
        ascending=[True, False, False, False, True, True, True],
        kind="mergesort",
    )
    chosen = chosen.drop_duplicates("customer_id", keep="first")
    chosen["actual_delta"] = (
        chosen.challenger_target.astype(np.int8) - chosen.victim_target.astype(np.int8)
    ) * chosen.unit_gain
    return chosen.sort_values("customer_id", kind="mergesort").reset_index(drop=True)


def training_data() -> tuple[pd.DataFrame, dict]:
    frames = []
    sources = []
    for name, folder in HISTORICAL:
        path, meta = source(folder)
        check = meta.get("independent_source_check", {})
        assert check and all(value == 0 for value in check.values())
        frame = pair_frame(path, discordant_only=True)
        frame["source_window"] = name
        frames.append(frame)
        sources.append(
            {
                "window": name,
                "cutoff": meta["cutoff"],
                "rows": len(frame),
                "positive_rows": int(frame.challenger_target.sum()),
                "negative_rows": int((frame.challenger_target == 0).sum()),
                "weight_sum": float(frame.unit_gain.sum()),
                "path": str(path),
                "source_identity_checks": check,
            }
        )
    training = pd.concat(frames, ignore_index=True)
    assert np.isfinite(training[FEATURES].to_numpy(np.float64)).all()
    audit = {
        "role": "strictly_historical_pair_supervision",
        "sources": sources,
        "rows": len(training),
        "users_by_window_sum": int(training.groupby(["source_window", "customer_id"]).ngroups),
        "positive_rows": int(training.challenger_target.sum()),
        "negative_rows": int((training.challenger_target == 0).sum()),
        "weighted_positive_share": float(
            training.loc[training.challenger_target == 1, "unit_gain"].sum() / training.unit_gain.sum()
        ),
        "features": FEATURES,
        "latest_label_end_exclusive": "2019-11-27",
        "earliest_inner_cutoff": "2019-12-25",
        "temporal_gap_days": 28,
        "final_week": "not_run",
    }
    return training, audit


def evaluate(folder: str, name: str, model: lgb.Booster, stage: str) -> tuple[dict, pd.DataFrame]:
    path, meta = source(folder)
    started = time.perf_counter()
    pairs = pair_frame(path, discordant_only=False)
    scores = model.predict(pairs[FEATURES].to_numpy(np.float32), num_threads=4)
    chosen = choose_decisions(pairs, scores)
    beneficial = chosen.actual_delta > 1e-15
    harmful = chosen.actual_delta < -1e-15
    neutral = ~(beneficial | harmful)
    delta = float(chosen.actual_delta.sum() / meta["total_users"])
    baseline = float(meta["baseline_map_population_component"])
    result = {
        "window": name,
        "cutoff": meta["cutoff"],
        "role": stage,
        "source": str(path),
        "total_users_denominator": meta["total_users"],
        "included_ranked_users": meta["included_users"],
        "pair_rows": len(pairs),
        "selected_users": len(chosen),
        "selected_share_of_full_denominator": len(chosen) / meta["total_users"],
        "beneficial_selected_users": int(beneficial.sum()),
        "harmful_selected_users": int(harmful.sum()),
        "neutral_selected_users": int(neutral.sum()),
        "gross_positive_MAP": float(chosen.loc[beneficial, "actual_delta"].sum() / meta["total_users"]),
        "gross_negative_MAP": float(chosen.loc[harmful, "actual_delta"].sum() / meta["total_users"]),
        "baseline_MAP@12": baseline,
        "MAP@12": baseline + delta,
        "population_delta": delta,
        "protected_ranks1_7_changes": 0,
        "maximum_swaps_per_user": 1,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    return result, chosen


def experiment_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV2-601",
        "architecture_family": "historical_pairwise_residual_admission",
        "hypothesis": "A model trained only on four strictly earlier 2019 windows can predict whether one ranks13-50 challenger should replace one ranks8-12 baseline item while preserving ranks1-7.",
        "candidate_pool": "unchanged WV2-601 pool; inference considers existing baseline ranks8-50 only",
        "training_rows": "only discordant challenger-victim pairs from four 2019 outer rank caches; label is one when challenger was purchased and victim was not",
        "training_weight": "absolute exact AP@12 change of that direct swap",
        "model": "fixed LightGBM binary classifier",
        "features": FEATURES,
        "params": PARAMS,
        "rounds": ROUNDS,
        "threshold": THRESHOLD,
        "action": "at most one swap per user; ranks1-7 immutable; no-op unless the best pair probability is greater than0.5",
        "tie_order": "predicted delta descending, probability descending, victim rank descending, challenger rank ascending, article ids ascending",
        "inner_gate": {
            "mean_population_delta_min": 0.0001,
            "positive_windows_min": 3,
            "worst_delta_min": -0.0005,
        },
        "outer_policy": "one frozen 2019-trained model may be exposed once only if the inner gate passes; no inner or outer threshold tuning",
        "expected_minutes": 15,
        "failure_modes": [
            "historical pair preference is not portable to 2020",
            "neutral pairs receive overconfident positive scores and cause excessive swaps",
            "tail protection is insufficient even when ranks1-7 are frozen",
        ],
        "fallback": "exact WV2-601",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(15)
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "historical_pairwise_residual_admission",
            experiment_contract()["hypothesis"],
            candidate_protocol=experiment_contract()["candidate_pool"],
            training_protocol="four 2019 historical windows only; fixed binary model and exact AP swap weights; four 2020 inner screens before any outer",
            features=FEATURES,
            params={
                "model": PARAMS,
                "rounds": ROUNDS,
                "threshold": THRESHOLD,
                "protected_ranks": [1, 7],
                "victim_ranks": [8, 12],
                "challenger_ranks": [13, 50],
                "maximum_swaps_per_user": 1,
                "inner_gate": experiment_contract()["inner_gate"],
                "final_week": "not_run",
            },
            expected_minutes=15,
        )


def render_report(result: dict) -> str:
    lines = [
        "# WV3-601 历史多窗口局部残差准入筛选",
        "",
        "## 结论",
        "",
        f"内层门槛：**{'pass' if result['screening']['passed'] else 'fail'}**。模型仅用四个 2019 历史监督周训练，四个 2020 内层周只用于筛选。",
        "",
        "## 术语与模型",
        "",
        "- 局部残差准入（本项目自定义）：冻结 WV2-601 第1–7名，只允许第13–50名候选与第8–12名商品发生至多一次直接换位。",
        "- challenger / victim（本报告分别称挑战商品/被替换商品）：挑战商品来自基线第13–50名；被替换商品来自第8–12名，统计单位均为用户—商品对。",
        "- discordant pair（行业常用成对监督概念，本报告称异标签商品对）：同一用户的一对商品中恰好一件在监督周被购买；标签为挑战商品是否是被购买的一件。",
        "- residual admission probability（本项目自定义准入概率）：二分类树对挑战商品优于被替换商品的估计；固定大于0.5才允许换位，不把它解释为商品购买概率。",
        "- LightGBM binary classifier（行业通用）：梯度提升树二分类器；本实验固定80轮，不做阈值、轮数或树参数搜索。",
        "- population delta（本项目沿用指标）：全部真值用户分母下，准入后 MAP@12 减 WV2-601 MAP@12。",
        "",
        "训练只保留异标签对，样本权重等于该直接交换造成的 AP@12 绝对变化。两件都买或都不买的中性商品对不提供训练标签，但推理时仍会被模型评估；因此报告必须单列中性、真正有益和真正有害的换位。",
        "## 训练证据",
        "",
        f"训练行数 {result['training']['rows']:,}，正类 {result['training']['positive_rows']:,}，负类 {result['training']['negative_rows']:,}；加权正类占比 {result['training']['weighted_positive_share']:.4%}。最新训练标签在 2019-11-27 前结束，距最早 2020 内层截止日 28 天。",
        "",
        "## 四窗内层结果",
        "",
        "| 窗口 | 选择用户 | 有益/有害/中性 | 基线 MAP@12 | 准入 MAP@12 | population delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["windows"].items():
        lines.append(
            f"| {name} | {row['selected_users']:,} | {row['beneficial_selected_users']:,}/"
            f"{row['harmful_selected_users']:,}/{row['neutral_selected_users']:,} | "
            f"{row['baseline_MAP@12']:.9f} | {row['MAP@12']:.9f} | {row['population_delta']:+.9f} |"
        )
    gate = result["screening"]
    lines += [
        "",
        f"均值增益 {gate['mean_population_delta']:+.9f}，正增窗口 {gate['positive_windows']}/4，最差窗口 {gate['worst_delta']:+.9f}。",
        "",
        "只有通过固定内层门槛才允许让这个完全冻结的 2019 模型读取一次四窗外层；失败则关闭该具体模型，不调阈值或树参数。第1–7名变更数恒为0，候选池未改变，最终周 2020-09-16 保持 not_run。",
        "",
    ]
    return "\n".join(lines)


def screen() -> dict:
    common.setup()
    common.budget(15)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    started = time.perf_counter()
    training, audit = training_data()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    training_audit_path = MODEL_ROOT / "TRAINING_AUDIT.json"
    write(training_audit_path, audit)
    model = fit_model(training)
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
            "threshold": THRESHOLD,
            "training_audit": str(training_audit_path),
            "outer_labels_used": False,
            "final_week": "not_run",
        },
    )
    del training
    gc.collect()

    windows = {}
    for name, folder in INNER:
        row, chosen = evaluate(folder, name, model, "historical_to_2020_inner_screen")
        windows[name] = row
        root = MODEL_ROOT / name / "inner"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"residual_inner": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    gate = screening_gate(row["population_delta"] for row in windows.values())
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "architecture": "historical_pairwise_residual_admission",
        "training": audit,
        "windows": windows,
        "screening": gate,
        "features": FEATURES,
        "params": PARAMS,
        "rounds": ROUNDS,
        "threshold": THRESHOLD,
        "candidate_pool_changed": False,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(SCREEN_REPORT, result)
    (common.REPORT / "WV3-601_SCREEN.md").write_text(render_report(result), encoding="utf-8")
    common.update(
        TRIAL,
        decision="inner_pass" if gate["passed"] else "reject_inner",
        inner_evidence=gate,
        runtime=result["runtime_seconds"],
        artifact_paths=[str(SCREEN_REPORT), str(common.REPORT / "WV3-601_SCREEN.md"), str(MODEL_ROOT / "MODEL.json")],
    )
    common.log(
        f"{TRIAL} inner screen",
        "WV3-600 established enough tail-only oracle headroom; learnability must be tested without current-window labels.",
        experiment_contract()["hypothesis"],
        f"{training_audit_path}; {SCREEN_REPORT}",
        f"mean delta={gate['mean_population_delta']:+.9f}; positive windows={gate['positive_windows']}/4; worst={gate['worst_delta']:+.9f}; gate={'pass' if gate['passed'] else 'fail'}.",
        "Freeze and expose once on outer." if gate["passed"] else "Reject WV3-601 and do not tune its threshold, rounds or tree parameters.",
        "Run exactly one four-window outer confirmation if passed; otherwise decompose selection errors before choosing a genuinely different mechanism.",
        alternatives="A same-window cross-fitted ranker already failed. The historical-only model is preferred because it tests cross-time portability directly.",
        experiment="One fixed 80-round binary LightGBM fit on four 2019 discordant-pair sources; probability threshold0.5; one swap; Top1-7 immutable.",
        reflection="The screen distinguishes action-space headroom from historical feature portability; it does not use 2020 inner labels for fitting.",
    )
    print({"trial": TRIAL, "screening": gate, "seconds": result["runtime_seconds"]}, flush=True)
    return result


def confirm() -> dict:
    common.setup()
    common.budget(10)
    registry = read(common.REGISTRY)
    entry = next(row for row in registry["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "inner_pass" and entry["outer_exposures"] == 1
    assert read(SCREEN_REPORT)["screening"]["passed"]
    if OUTER_REPORT.exists():
        return read(OUTER_REPORT)
    model = lgb.Booster(model_file=str(MODEL_ROOT / "MODEL.txt"))
    started = time.perf_counter()
    windows = {}
    for name, folder in OUTER:
        row, chosen = evaluate(folder, name, model, "frozen_historical_model_outer")
        windows[name] = row
        root = MODEL_ROOT / name / "outer"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"residual_outer": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    summary = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "architecture": "historical_pairwise_residual_admission",
        "windows": windows,
        **summary,
        "model_source": str(MODEL_ROOT / "MODEL.json"),
        "threshold": THRESHOLD,
        "candidate_pool_changed": False,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(OUTER_REPORT, result)
    common.update(
        TRIAL,
        decision="promote_candidate" if summary["stable"] else "reject_outer",
        outer_MAP_by_window=summary["per_window_MAP"],
        mean_MAP=summary["mean_MAP"],
        delta_vs_WV2_601=summary["delta_vs_WV2_601"],
        nondegrade_windows=summary["nondegrade_windows"],
        worst_delta=summary["worst_delta"],
        runtime=result["runtime_seconds"],
        artifact_paths=entry["artifact_paths"] + [str(OUTER_REPORT)],
    )
    common.log(
        f"{TRIAL} outer closure",
        "The historical-only residual model passed the fixed four-window inner screen and was frozen before outer exposure.",
        experiment_contract()["hypothesis"],
        str(OUTER_REPORT),
        f"mean MAP={summary['mean_MAP']:.9f}; delta={summary['delta_vs_WV2_601']:+.9f}; nondegrade={summary['nondegrade_windows']}/4; worst={summary['worst_delta']:+.9f}; stable={summary['stable']}.",
        "Retain as a promotion candidate." if summary["stable"] else "Reject this concrete variant; no threshold or model rescue.",
        "If stable but below0.03, preserve it and continue only with a new evidence-backed family; if rejected, run failure decomposition before pivoting.",
        alternatives="No 2020 labels, outer values or candidate-pool changes were used to fit or tune the frozen model.",
        experiment="One outer exposure of the exact 2019-trained model, fixed0.5 threshold and one-swap policy.",
        reflection="Outer evidence determines temporal portability; mean gain alone cannot override the worst-window condition.",
    )
    print({"trial": TRIAL, "outer": summary}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "screen", "confirm"])
    globals()[parser.parse_args().command]()
