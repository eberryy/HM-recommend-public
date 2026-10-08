from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from .m2 import CATEGORICAL_FEATURES as BASE_CATEGORICAL_FEATURES
from .m2 import _prepare_frame, build_category_maps
from .m3 import _peak_working_set_bytes
from .m33 import ROLLING_PROTOCOL, variant_feature_sets
from .m4_contract import FINAL_CUTOFF, WARM_MAP, atomic_json, file_identity
from .m54 import COLD_FEATURES, RUN_ID as M54_RUN_ID
from .m5_model import _write_parquet


RUN_ID = "m5-5-v1-source-aware-dual-channel-fusion"
SEED = 20260903
NEGATIVES_PER_POSITIVE = 30
MAX_BOOST_ROUNDS = 200
EARLY_STOPPING_ROUNDS = 20
MAP_TOLERANCE = 1e-12
WINDOW_TOLERANCE = 0.0002
BRANCH_MAP = {"warm_only": 0, "cold_only": 1, "warm_and_cold": 2}

BASE_ANCHOR = [name for name in variant_feature_sets()["anchor"] if name != "candidate_rank"]
WARM_FEATURES = list(dict.fromkeys(BASE_ANCHOR + [
    "warm_rank", "warm_rank_pct", "warm_candidate_rank", "warm_model_score",
    "warm_model_score_available", "item_events_before_cutoff", "coldness_bucket",
]))
DUAL_FEATURES = list(dict.fromkeys(WARM_FEATURES + [
    "warm_present", "cold_present", "both_present", "source_branch",
    "cold_rank", "cold_rank_pct", *COLD_FEATURES,
    "cold_expert_score", "cold_expert_rank", "cold_expert_score_available",
]))
DUAL_WITHOUT_EXPERT_FEATURES = [
    name for name in DUAL_FEATURES
    if name not in {"cold_expert_score", "cold_expert_rank", "cold_expert_score_available"}
]
CATEGORICAL_FEATURES = [
    *[name for name in BASE_CATEGORICAL_FEATURES if name in DUAL_FEATURES],
    "source_branch", "coldness_bucket",
]
VARIANTS = {
    "warm_shortlist_control": {"features": WARM_FEATURES, "warm_only": True},
    "dual_fusion": {"features": DUAL_FEATURES, "warm_only": False},
    "dual_without_cold_expert": {"features": DUAL_WITHOUT_EXPERT_FEATURES, "warm_only": False},
}


def _literal(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _feature_path(artifact_dir: Path, cutoff: str) -> Path:
    return artifact_dir / cutoff / "features.parquet"


def _prepare(
    frame: pd.DataFrame, features: list[str], category_maps: dict[str, dict[int, int]],
) -> pd.DataFrame:
    source = frame.copy()
    if "source_branch" in source:
        source["source_branch"] = source["source_branch"].map(BRANCH_MAP)
    prepared = _prepare_frame(source, features, category_maps)
    for name in ("source_branch", "coldness_bucket"):
        if name in prepared:
            prepared[name] = pd.Categorical(
                pd.to_numeric(prepared[name], errors="coerce"), categories=range(4)
            )
    return prepared


def _sample_relation(path: Path, *, warm_only: bool) -> str:
    where = "WHERE warm_present=1" if warm_only else ""
    return f"""
      WITH source AS (
        SELECT *,
          CASE WHEN least(coalesce(warm_rank_pct,2.0),coalesce(cold_rank_pct,2.0))<=1.0/3 THEN 'hard'
               WHEN least(coalesce(warm_rank_pct,2.0),coalesce(cold_rank_pct,2.0))<=2.0/3 THEN 'medium'
               ELSE 'easy' END AS sample_rank_bucket
        FROM read_parquet({_literal(path)}) {where}
      ), positive_groups AS (
        SELECT target_cutoff,customer_id,sum(target)::BIGINT AS positives
        FROM source GROUP BY target_cutoff,customer_id HAVING positives>0
      ), positives AS (
        SELECT s.*,1::BIGINT AS stratum_rows,1::BIGINT AS selected_stratum_rows
        FROM source s JOIN positive_groups g USING(target_cutoff,customer_id) WHERE target=1
      ), negative_source AS (
        SELECT s.*,g.positives,
          count(*) OVER(PARTITION BY s.target_cutoff,s.customer_id,s.source_branch,s.sample_rank_bucket)::BIGINT AS stratum_rows,
          row_number() OVER(
            PARTITION BY s.target_cutoff,s.customer_id,s.source_branch,s.sample_rank_bucket
            ORDER BY hash(s.target_cutoff,s.customer_id,s.article_id,{SEED}),s.candidate_rank,s.article_id
          )::BIGINT AS stratum_sample_rank
        FROM source s JOIN positive_groups g USING(target_cutoff,customer_id) WHERE target=0
      ), negative_ordered AS (
        SELECT *,row_number() OVER(
          PARTITION BY target_cutoff,customer_id
          ORDER BY stratum_sample_rank,source_branch,sample_rank_bucket,
                   hash(target_cutoff,customer_id,article_id,{SEED}),candidate_rank,article_id
        )::BIGINT AS negative_sample_rank
        FROM negative_source
      ), selected_pre AS (
        SELECT * FROM negative_ordered WHERE negative_sample_rank<={NEGATIVES_PER_POSITIVE}*positives
      ), selected_negatives AS (
        SELECT * EXCLUDE(positives,stratum_sample_rank,negative_sample_rank),
          count(*) OVER(PARTITION BY target_cutoff,customer_id,source_branch,sample_rank_bucket)::BIGINT AS selected_stratum_rows
        FROM selected_pre
      )
      SELECT * FROM positives UNION ALL BY NAME SELECT * FROM selected_negatives
    """


def load_training_sample(
    *, paths: list[Path], features: list[str], warm_only: bool,
) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    sizes: list[int] = []
    evidence: dict[str, Any] = {"cutoffs": {}, "sampling_cells": {}}
    metadata = ["target_cutoff", "customer_id", "article_id", "candidate_rank", "target"]
    for path in paths:
        con = duckdb.connect()
        try:
            relation = _sample_relation(path, warm_only=warm_only)
            selected = list(dict.fromkeys(metadata + features))
            frame = con.execute(
                f"SELECT {','.join(selected)} FROM ({relation}) "
                "ORDER BY target_cutoff,customer_id,candidate_rank,article_id"
            ).fetchdf()
            group_rows = con.execute(
                f"SELECT count(*)::BIGINT,sum(target)::BIGINT FROM ({relation}) "
                "GROUP BY target_cutoff,customer_id ORDER BY target_cutoff,customer_id"
            ).fetchall()
            source_where = "WHERE warm_present=1" if warm_only else ""
            source_stats = con.execute(
                f"SELECT count(*)::BIGINT,sum(target)::BIGINT,count(DISTINCT customer_id)::BIGINT "
                f"FROM read_parquet({_literal(path)}) {source_where}"
            ).fetchone()
            cells = con.execute(
                f"""
                WITH source AS (
                  SELECT *,CASE WHEN least(coalesce(warm_rank_pct,2.0),coalesce(cold_rank_pct,2.0))<=1.0/3 THEN 'hard'
                    WHEN least(coalesce(warm_rank_pct,2.0),coalesce(cold_rank_pct,2.0))<=2.0/3 THEN 'medium' ELSE 'easy' END AS rb
                  FROM read_parquet({_literal(path)}) {source_where}
                ), selected AS ({relation})
                SELECT s.source_branch,s.rb,count(*)::BIGINT,
                  (SELECT count(*) FROM selected x WHERE x.target=0 AND x.source_branch=s.source_branch AND x.sample_rank_bucket=s.rb)::BIGINT
                FROM source s WHERE s.target=0 GROUP BY s.source_branch,s.rb ORDER BY s.source_branch,s.rb
                """
            ).fetchall()
        finally:
            con.close()
        if any(int(row[1]) <= 0 for row in group_rows):
            raise RuntimeError("M5.5 sampling retained a zero-positive training group")
        frames.append(frame)
        sizes.extend(int(row[0]) for row in group_rows)
        cutoff = str(frame["target_cutoff"].iloc[0])[:10]
        positives = int(frame["target"].sum())
        negatives = len(frame) - positives
        evidence["cutoffs"][cutoff] = {
            "source_rows": int(source_stats[0]), "source_positive_pairs": int(source_stats[1]),
            "source_users": int(source_stats[2]), "sampled_rows": len(frame),
            "positive_pairs": positives, "sampled_unobserved_pairs": negatives,
            "unobserved_per_positive": negatives / max(positives, 1),
            "positive_groups": len(group_rows),
        }
        for branch, bucket, source_rows, selected_rows in cells:
            key = f"{branch}:{bucket}"
            cell = evidence["sampling_cells"].setdefault(key, {"source_rows": 0, "selected_rows": 0})
            cell["source_rows"] += int(source_rows)
            cell["selected_rows"] += int(selected_rows)
    result = pd.concat(frames, ignore_index=True)
    if sum(sizes) != len(result):
        raise RuntimeError("M5.5 group sizes do not conserve sampled rows")
    positives = int(result["target"].sum())
    negatives = len(result) - positives
    if negatives > NEGATIVES_PER_POSITIVE * positives:
        raise RuntimeError("M5.5 negative cap exceeded")
    evidence.update({
        "rows": len(result), "positive_pairs": positives,
        "sampled_unobserved_pairs": negatives,
        "unobserved_per_positive": negatives / max(positives, 1),
        "groups": len(sizes), "min_group_rows": min(sizes), "max_group_rows": max(sizes),
        "sample_rule": "all positives plus fixed-hash branch x rank-bucket negatives, cap 30 per positive",
    })
    return result, sizes, evidence


def _load_validation(
    *, path: Path, features: list[str], warm_only: bool,
) -> tuple[pd.DataFrame, list[int]]:
    where = "WHERE warm_present=1" if warm_only else ""
    selected = list(dict.fromkeys([
        "target_cutoff", "customer_id", "article_id", "candidate_rank", "target",
        "user_history_events_12w", "warm_rank", "source_branch", "item_events_before_cutoff",
        *features,
    ]))
    con = duckdb.connect()
    try:
        frame = con.execute(
            f"SELECT {','.join(selected)} FROM read_parquet({_literal(path)}) {where} "
            "ORDER BY customer_id,candidate_rank,article_id"
        ).fetchdf()
    finally:
        con.close()
    sizes = frame.groupby("customer_id", sort=False, observed=True).size().astype(int).tolist()
    if sum(sizes) != len(frame):
        raise RuntimeError("M5.5 validation group sizes do not conserve rows")
    return frame, sizes


def _truth_and_counts(
    *, transactions_path: Path, cutoff: str, users: list[str],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    con = duckdb.connect()
    user_frame = pd.DataFrame({"customer_id": users})
    con.register("eval_users", user_frame)
    try:
        truth_rows = con.execute(
            f"SELECT DISTINCT t.customer_id,t.article_id FROM read_parquet({_literal(transactions_path)}) t "
            f"JOIN eval_users u USING(customer_id) WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY"
        ).fetchall()
        counts = con.execute(
            f"SELECT article_id,count(*)::BIGINT FROM read_parquet({_literal(transactions_path)}) "
            f"WHERE t_dat<DATE '{cutoff}' GROUP BY article_id"
        ).fetchall()
    finally:
        con.unregister("eval_users")
        con.close()
    truth = {user: set() for user in users}
    for user, item in truth_rows:
        truth[str(user)].add(str(item))
    if any(not values for values in truth.values()):
        raise RuntimeError("M5.5 validation includes a user with no next-week truth")
    return truth, {str(item): int(value) for item, value in counts}


def _exact_map_from_arrays(
    predictions: np.ndarray, labels: np.ndarray, sizes: list[int], truth_counts: list[int],
    inactive: np.ndarray, warm_rank: np.ndarray, candidate_rank: np.ndarray,
) -> float:
    total = 0.0
    offset = 0
    for size, truth_count in zip(sizes, truth_counts):
        end = offset + size
        if inactive[offset]:
            order = np.lexsort((candidate_rank[offset:end], np.nan_to_num(warm_rank[offset:end], nan=1e9)))
        else:
            order = np.lexsort((candidate_rank[offset:end], -predictions[offset:end]))
        rel = labels[offset:end][order[:12]].astype(np.float64)
        if rel.any():
            total += float(np.sum(np.cumsum(rel) / np.arange(1, len(rel) + 1) * rel)) / min(truth_count, 12)
        offset = end
    return total / max(len(sizes), 1)


def _truth_counts_for_frame(frame: pd.DataFrame, truth: dict[str, set[str]]) -> list[int]:
    users = frame.groupby("customer_id", sort=False, observed=True).size().index.tolist()
    return [len(truth[str(user)]) for user in users]


def train_ranker(
    *, train: pd.DataFrame, train_sizes: list[int], validation: pd.DataFrame | None,
    validation_sizes: list[int] | None, validation_truth: dict[str, set[str]] | None,
    features: list[str], category_maps: dict[str, dict[int, int]], output_path: Path,
    rounds: int | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    started = time.perf_counter()
    train_x = _prepare(train, features, category_maps)
    categorical = [name for name in CATEGORICAL_FEATURES if name in features]
    train_set = lgb.Dataset(
        train_x, label=train["target"].astype(np.uint8), group=train_sizes,
        feature_name=features, categorical_feature=categorical, free_raw_data=True,
    )
    params = {
        "objective": "lambdarank", "metric": "None", "learning_rate": 0.05,
        "num_leaves": 31, "min_data_in_leaf": 100, "feature_fraction": 1.0,
        "bagging_fraction": 1.0, "bagging_freq": 0, "seed": SEED,
        "feature_fraction_seed": SEED, "bagging_seed": SEED, "deterministic": True,
        "force_col_wise": True, "num_threads": 8, "verbosity": -1,
        "lambdarank_truncation_level": 20,
    }
    valid_sets = None
    callbacks: list[Any] = [lgb.log_evaluation(period=0)]
    feval = None
    if validation is not None:
        assert validation_sizes is not None and validation_truth is not None
        valid_x = _prepare(validation, features, category_maps)
        valid_set = lgb.Dataset(
            valid_x, label=validation["target"].astype(np.uint8), group=validation_sizes,
            feature_name=features, categorical_feature=categorical, reference=train_set,
        )
        valid_sets = [valid_set]
        truth_counts = _truth_counts_for_frame(validation, validation_truth)
        labels = validation["target"].to_numpy(dtype=np.uint8)
        inactive = validation["user_history_events_12w"].to_numpy(dtype=np.float64) <= 0
        warm_rank = pd.to_numeric(validation["warm_rank"], errors="coerce").to_numpy(dtype=np.float64)
        candidate_rank = validation["candidate_rank"].to_numpy(dtype=np.int64)

        def exact_map(predictions: np.ndarray, _dataset: lgb.Dataset) -> tuple[str, float, bool]:
            return (
                "exact_complete_map@12",
                _exact_map_from_arrays(
                    predictions, labels, validation_sizes, truth_counts,
                    inactive, warm_rank, candidate_rank,
                ),
                True,
            )

        feval = exact_map
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False))
    num_rounds = int(rounds or MAX_BOOST_ROUNDS)
    model = lgb.train(
        params, train_set, num_boost_round=num_rounds, valid_sets=valid_sets,
        valid_names=["inner_validation"] if valid_sets else None, feval=feval, callbacks=callbacks,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    importance = sorted(
        [
            {"feature": name, "gain": float(gain), "split": int(split)}
            for name, gain, split in zip(
                features, model.feature_importance("gain"), model.feature_importance("split")
            )
        ], key=lambda row: (-row["gain"], row["feature"]),
    )
    evidence = {
        "best_iteration": int(model.best_iteration or num_rounds),
        "train_rows": len(train), "train_positive_pairs": int(train["target"].sum()),
        "validation_rows": len(validation) if validation is not None else 0,
        "validation_positive_pairs": int(validation["target"].sum()) if validation is not None else 0,
        "top_feature_importance": importance[:40], "model": file_identity(output_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return model, evidence


def _baseline_top12(evaluation_db: Path) -> pd.DataFrame:
    con = duckdb.connect(str(evaluation_db), read_only=True)
    try:
        return con.execute(
            """
            SELECT customer_id,article_id,warm_rank FROM (
              SELECT customer_id,article_id,row_number() OVER(
                PARTITION BY customer_id
                ORDER BY CASE WHEN user_history_events_12w=0 THEN candidate_rank END ASC NULLS LAST,
                         CASE WHEN user_history_events_12w>0 THEN score_anchor END DESC NULLS LAST,
                         candidate_rank,article_id
              )::INTEGER AS warm_rank FROM predictions
            ) WHERE warm_rank<=12 ORDER BY customer_id,warm_rank
            """
        ).fetchdf()
    finally:
        con.close()


def _rank_top12(frame: pd.DataFrame, score: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    ranked = frame[[
        "customer_id", "article_id", "candidate_rank", "target", "user_history_events_12w",
        "warm_rank", "source_branch", "item_events_before_cutoff",
    ]].copy()
    ranked["score"] = score
    inactive = ranked["user_history_events_12w"].to_numpy(dtype=np.float64) <= 0
    ranked["active_score"] = np.where(inactive, -np.inf, score)
    ranked["fallback_rank"] = np.where(
        inactive, pd.to_numeric(ranked["warm_rank"], errors="coerce").fillna(1e9), 0,
    )
    ranked.sort_values(
        ["customer_id", "active_score", "fallback_rank", "candidate_rank", "article_id"],
        ascending=[True, False, True, True, True], kind="mergesort", inplace=True,
    )
    ranked["final_rank"] = ranked.groupby("customer_id", sort=False).cumcount() + 1
    top12 = ranked.loc[ranked["final_rank"] <= 12].copy()
    ranks = ranked.sort_index()["final_rank"].to_numpy(dtype=np.int32)
    return top12, ranks


def _ap(ranked: list[str], truth: set[str]) -> float:
    if not truth:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(ranked[:12], start=1):
        if item in truth:
            hits += 1
            score += hits / rank
    return score / min(len(truth), 12)


def _segments(truth: dict[str, set[str]], counts: dict[str, int]) -> dict[str, dict[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = {}
    for user, items in truth.items():
        result[user] = {
            "overall": set(items),
            "strict_cold": {item for item in items if counts.get(item, 0) == 0},
            "sparse_1_5": {item for item in items if 1 <= counts.get(item, 0) <= 5},
            "warm_21_plus": {item for item in items if counts.get(item, 0) >= 21},
        }
    return result


def _evaluate(
    *, frame: pd.DataFrame, top12: pd.DataFrame, ranks: np.ndarray,
    baseline_top12: pd.DataFrame, truth: dict[str, set[str]], counts: dict[str, int],
) -> dict[str, Any]:
    predictions = {
        str(user): group.sort_values("final_rank")["article_id"].astype(str).tolist()
        for user, group in top12.groupby("customer_id", sort=False)
    }
    baseline = {
        str(user): group.sort_values("warm_rank")["article_id"].astype(str).tolist()
        for user, group in baseline_top12.groupby("customer_id", sort=False)
    }
    segments = _segments(truth, counts)
    metrics: dict[str, Any] = {}
    for name in ("overall", "warm_21_plus", "strict_cold", "sparse_1_5"):
        users = [user for user in truth if segments[user][name]]
        map_value = float(np.mean([_ap(predictions.get(user, []), segments[user][name]) for user in users])) if users else 0.0
        base_map = float(np.mean([_ap(baseline.get(user, []), segments[user][name]) for user in users])) if users else 0.0
        metrics[name] = {
            "map@12": map_value, "warm_baseline_map@12": base_map,
            "delta_vs_warm_map@12": map_value - base_map,
            "truth_users": len(users), "truth_pairs": sum(len(segments[user][name]) for user in users),
        }
    candidate_sets = {
        str(user): set(group["article_id"].astype(str))
        for user, group in frame.groupby("customer_id", sort=False)
    }
    recalls = []
    oracles = []
    for user, items in truth.items():
        hits = len(items & candidate_sets.get(user, set()))
        recalls.append(hits / len(items))
        oracles.append(min(hits, 12) / min(len(items), 12))
    baseline_sets = {user: set(items) for user, items in baseline.items()}
    predicted_sets = {user: set(items) for user, items in predictions.items()}
    inserted = removed = inserted_cold_sparse = removed_warm = 0
    for user, items in truth.items():
        inserted_items = (predicted_sets.get(user, set()) - baseline_sets.get(user, set())) & items
        removed_items = (baseline_sets.get(user, set()) - predicted_sets.get(user, set())) & items
        inserted += len(inserted_items)
        removed += len(removed_items)
        inserted_cold_sparse += sum(counts.get(item, 0) <= 5 for item in inserted_items)
        removed_warm += sum(counts.get(item, 0) >= 21 for item in removed_items)
    positive_cold_only = (frame["source_branch"] == "cold_only") & (frame["target"] == 1)
    conversions = {
        "union": int(positive_cold_only.sum()),
        "top50": int((positive_cold_only & (ranks <= 50)).sum()),
        "top20": int((positive_cold_only & (ranks <= 20)).sum()),
        "top12": int((positive_cold_only & (ranks <= 12)).sum()),
    }
    composition = top12["source_branch"].value_counts().to_dict()
    return {
        "segments": metrics,
        "candidate_recall": float(np.mean(recalls)), "oracle_map@12": float(np.mean(oracles)),
        "cold_only_truth_conversion": conversions,
        "final_top12_branch_composition": {name: int(composition.get(name, 0)) for name in BRANCH_MAP},
        "inserted_positive_pairs": inserted, "removed_positive_pairs": removed,
        "inserted_cold_sparse_positive_pairs": inserted_cold_sparse,
        "removed_warm_positive_pairs": removed_warm,
    }


def _load_report_inputs(result: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Load measured M5.3/M5.4 evidence used only to make the M5.5 report self-contained."""
    try:
        m54_path = Path(result["inputs"]["m54_metrics"]["path"])
        m54 = json.loads(m54_path.read_text(encoding="utf-8"))
        m53_path = Path(m54["inputs"]["m53_metrics"]["path"])
        m53 = json.loads(m53_path.read_text(encoding="utf-8"))
        return m53, m54
    except (KeyError, OSError, json.JSONDecodeError):
        return None, None


def _render_report(
    result: dict[str, Any], *, verification: dict[str, Any] | None = None,
) -> str:
    m53, _ = _load_report_inputs(result)
    dual_deltas = result["summary"]["dual_window_delta_vs_warm_map@12"]
    non_degrading = sum(float(value) >= -MAP_TOLERANCE for value in dual_deltas.values())
    cold_top12_windows = sum(
        row["evaluations"]["dual_fusion"]["cold_only_truth_conversion"]["top12"] > 0
        for row in result["windows"].values()
    )
    lines = [
        "# M5.5：来源感知双通道统一排序",
        "",
        "## 结论",
        "",
        f"- M5.5 gate（晋级门禁）通过：{result['summary']['gate_passed']}；下一阶段：`{result['summary']['next_stage']}`。",
        "- Primary（主方案）是 Warm150 与 Student Cold50 去重并集上的 LightGBM LambdaRank；没有固定冷候选插入数量，也没有线性混合两路分数。",
        f"- 四窗 mean MAP@12 相对冻结 Warm-v1 为 {result['summary']['dual_mean_delta_vs_warm_map@12']:+.6f}；不退化窗口 {non_degrading}/4，最差单窗 {min(dual_deltas.values()):+.6f}，cold-only 正例进入 Top12 的窗口 {cold_top12_windows}/4。",
        "- final week：not_run。",
        "",
        "## 术语与统计单位",
        "",
        "- **Warm-shortlist-only Fusion control**：只保留 Warm150 候选、用相同时间链与30:1分层负采样重新训练的控制组；用于区分“截短/重训效应”和 Cold 分支增量。",
        "- **source-aware Fusion**：来源感知统一排序；模型同时看到 Warm/Cold 是否出现、各自名次、相似度和时间点安全上游分数，直接在并集内学习相对次序。",
        "- **ranking group**：行业通用排序组；本报告一组是一个 target cutoff—用户，组内候选互相比较。零正例组不训练，但完整保留在验证 MAP 分母。",
        "- **cold-only truth conversion**：只由 Cold 路召回的正例用户—商品对，在完整并集及最终统一名次 Top50/Top20/Top12 的数量；每个阶段分母都是 union 列的 cold-only 正例数。",
        "- **inserted/removed positive pair**：相对冻结 Warm-v1 Top12，新进入或被挤出的真实购买用户—商品对；不是所有插入/移除候选行。",
        "- **feature gain**：LightGBM 行业通用的分裂增益累计值；0 表示该特征未被树用于有效分裂，但它不是单独特征的因果贡献。",
        "- **sampling cell**：本项目自定义的负样本分层单元，由候选来源分支与其名次难度桶组成；表中 source 是可抽取的未观察候选行数，selected 是实际进入训练的未观察候选行数。",
        "- **upstream OOF score distribution**：上游模型的时间外推/折外分数分布；统计单位是当前 outer cutoff 并集候选行，分母仅为 availability=1 的行。",
        "",
        "## 四窗 MAP@12",
        "",
        "| window | frozen Warm-v1 | Warm-only control | Dual Fusion | Dual w/o Cold Expert | Dual delta | warm bucket delta | cold-only truth Top12 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for window, row in result["windows"].items():
        baseline = row["evaluations"]["frozen_warm_v1"]["segments"]["overall"]["map@12"]
        control = row["evaluations"]["warm_shortlist_control"]["segments"]["overall"]["map@12"]
        dual = row["evaluations"]["dual_fusion"]
        no_expert = row["evaluations"]["dual_without_cold_expert"]["segments"]["overall"]["map@12"]
        lines.append(
            f"| {window} | {baseline:.6f} | {control:.6f} | {dual['segments']['overall']['map@12']:.6f} | "
            f"{no_expert:.6f} | {dual['segments']['overall']['delta_vs_warm_map@12']:+.6f} | "
            f"{dual['segments']['warm_21_plus']['delta_vs_warm_map@12']:+.6f} | "
            f"{dual['cold_only_truth_conversion']['top12']} |"
        )
    lines.extend([
        "",
        "## 分人群 MAP@12",
        "",
        "warm_21_plus 表示候选商品在 cutoff 前至少有 21 次购买事件；strict_cold 表示 0 次；sparse_1_5 表示 1--5 次。每格分母是该窗口该人群的完整 truth users，未召回用户仍计 0。",
        "",
        "| window | variant | overall | warm_21_plus | strict_cold | sparse_1_5 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    variant_labels = {
        "frozen_warm_v1": "Frozen Warm-v1",
        "warm_shortlist_control": "Warm-only control",
        "dual_fusion": "Dual Fusion",
        "dual_without_cold_expert": "Dual w/o Cold Expert",
    }
    for window, row in result["windows"].items():
        for variant, label in variant_labels.items():
            segments = row["evaluations"][variant]["segments"]
            lines.append(
                f"| {window} | {label} | {segments['overall']['map@12']:.6f} | "
                f"{segments['warm_21_plus']['map@12']:.6f} | "
                f"{segments['strict_cold']['map@12']:.6f} | "
                f"{segments['sparse_1_5']['map@12']:.6f} |"
            )
    if m53 is not None:
        selected_k = str(m53["decision"]["k_warm"])
        lines.extend([
            "",
            "## Warm shortlist 正例保留",
            "",
            f"这里复述 M5.3 冻结结果 K_warm={selected_k}。positive pair retention 的分母是冻结 Warm-v1 Top300 中命中的真实购买用户—商品对；truth-user retention 的分母是 Top300 至少命中一个真实购买的用户。M5.5 未再用 outer MAP 选择 K。",
            "",
            "| window | Top300 positive pairs | Top150 retained pairs | pair retention | truth-user retention |",
            "|---|---:|---:|---:|---:|",
        ])
        for window, row in m53["windows"].items():
            metric = row["k_metrics"][selected_k]
            lines.append(
                f"| {window} | {metric['warm_top300_positive_pairs']} | "
                f"{metric['retained_positive_pairs']} | "
                f"{metric['warm_positive_retention@k']:.6f} | "
                f"{metric['truth_user_retention@k']:.6f} |"
            )
    lines.extend([
        "",
        "## Primary 漏斗与替换损益",
        "",
        "| window | candidate Recall | Oracle MAP@12 | cold-only union | Top50 | Top20 | Top12 | inserted cold/sparse positives | removed Warm positives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["windows"].items():
        dual = row["evaluations"]["dual_fusion"]
        conv = dual["cold_only_truth_conversion"]
        lines.append(
            f"| {window} | {dual['candidate_recall']:.6f} | {dual['oracle_map@12']:.6f} | "
            f"{conv['union']} | {conv['top50']} | {conv['top20']} | {conv['top12']} | "
            f"{dual['inserted_cold_sparse_positive_pairs']} | {dual['removed_warm_positive_pairs']} |"
        )
    lines.extend([
        "",
        "## 最终 Top12 来源构成与全量正例替换",
        "",
        "来源构成的单位是所有验证用户的推荐槽位数；warm_and_cold 表示同一用户—商品同时被两路召回。inserted/removed all positives 是相对冻结 Warm-v1 Top12 新增/损失的全部真实购买对。",
        "",
        "| window | warm_only slots | cold_only slots | warm_and_cold slots | inserted all positives | removed all positives |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["windows"].items():
        dual = row["evaluations"]["dual_fusion"]
        composition = dual["final_top12_branch_composition"]
        lines.append(
            f"| {window} | {composition['warm_only']} | {composition['cold_only']} | "
            f"{composition['warm_and_cold']} | {dual['inserted_positive_pairs']} | "
            f"{dual['removed_positive_pairs']} |"
        )

    feature_totals: dict[str, dict[str, float | int]] = {}
    feature_rows = (
        verification["source_aware_feature_gain_full"].values()
        if verification is not None and "source_aware_feature_gain_full" in verification
        else (row["source_aware_feature_gain"] for row in result["windows"].values())
    )
    for rows in feature_rows:
        for feature in rows:
            current = feature_totals.setdefault(feature["feature"], {"gain": 0.0, "nonzero_windows": 0})
            current["gain"] = float(current["gain"]) + float(feature["gain"])
            current["nonzero_windows"] = int(current["nonzero_windows"]) + int(float(feature["gain"]) > 0)
    lines.extend([
        "",
        "## 来源感知特征增益",
        "",
        "下表汇总四个 outer 模型；仅统计 Dual 相比 Warm-only 新增的来源/Cold 特征。",
        "",
        "| feature | total gain | windows with non-zero gain |",
        "|---|---:|---:|",
    ])
    for feature, values in sorted(feature_totals.items()):
        lines.append(f"| {feature} | {float(values['gain']):.6f} | {int(values['nonzero_windows'])}/4 |")
    if feature_totals and all(float(values["gain"]) == 0.0 for values in feature_totals.values()):
        lines.extend([
            "",
            "所有新增来源/Cold 特征的 gain 均为 0；同时 Dual 与去掉 Cold Expert 的消融在四窗逐窗完全同分。这说明本轮训练出的树没有实际使用 Cold 分支证据，不能把失败归因于 Cold Expert 某一个分数。",
        ])

    sampling_totals: dict[str, dict[str, int]] = {}
    for row in result["windows"].values():
        cells = row["training"]["dual_fusion"]["outer_sampling"]["sampling_cells"]
        for cell, values in cells.items():
            total = sampling_totals.setdefault(cell, {"source_rows": 0, "selected_rows": 0})
            total["source_rows"] += int(values["source_rows"])
            total["selected_rows"] += int(values["selected_rows"])
    lines.extend([
        "",
        "## 负采样分层覆盖",
        "",
        "四个 outer 训练集汇总；positive 不计入下表，所有 positive 均另行全保留。hard/medium/easy 分别对应候选在可用来源中的最佳名次百分位前 1/3、中 1/3、后 1/3；两路同时命中时取两者更靠前的百分位。",
        "",
        "| branch x rank bucket | source unobserved rows | selected unobserved rows | selected/source |",
        "|---|---:|---:|---:|",
    ])
    for cell, values in sorted(sampling_totals.items()):
        ratio = values["selected_rows"] / max(values["source_rows"], 1)
        lines.append(
            f"| {cell} | {values['source_rows']} | {values['selected_rows']} | {ratio:.6f} |"
        )

    if verification is not None and "outer_training_branch_supervision" in verification:
        lines.extend([
            "",
            "## Outer 训练监督的来源分布",
            "",
            "每窗汇总该 outer 模型使用的两个更早训练 cutoff；positive pair 是下一周真实购买的截止日—用户—商品对。positive share 的分母是同一窗口三种来源分支的全部训练正例。",
            "",
            "| window | branch | candidate rows | positive pairs | positive density | positive share |",
            "|---|---|---:|---:|---:|---:|",
        ])
        for window, branches in verification["outer_training_branch_supervision"].items():
            for branch, values in branches.items():
                lines.append(
                    f"| {window} | {branch} | {values['candidate_rows']} | "
                    f"{values['positive_pairs']} | {values['positive_density']:.8f} | "
                    f"{values['share_of_all_positive_pairs']:.6f} |"
                )
        lines.extend([
            "",
            "## 时间内验证选择的树轮数",
            "",
            "inner rounds 是每窗只用更早 inner validation 选择的 boosting 轮数，随后原样用于 outer 训练；它不是根据 outer MAP 调出的参数。",
            "",
            "| window | inner-selected rounds | sampled outer rows | sampled outer positives |",
            "|---|---:|---:|---:|",
        ])
        for window, row in result["windows"].items():
            training = row["training"]["dual_fusion"]
            lines.append(
                f"| {window} | {training['inner_model']['best_iteration']} | "
                f"{training['outer_sampling']['rows']} | {training['outer_sampling']['positive_pairs']} |"
            )

    if verification is not None and "upstream_score_distributions" in verification:
        lines.extend([
            "",
            "## 上游 OOF/forward 分数分布",
            "",
            "数值来自独立 verifier 对四个 outer feature parquet 的只读聚合。warm_model_score 是 Warm-v1 的时间外推分数；cold_expert_score 是可选 Cold Expert 的时间外推分数。",
            "",
            "| window | score | available rows | min | median | p95 | max |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for window, scores in verification["upstream_score_distributions"].items():
            for score_name, values in scores.items():
                lines.append(
                    f"| {window} | {score_name} | {values['available_rows']} | "
                    f"{values['min']:.8g} | {values['median']:.8g} | "
                    f"{values['p95']:.8g} | {values['max']:.8g} |"
                )
    lines.extend([
        "",
        "## 门禁、采样与边界",
        "",
    ])
    for name, value in result["summary"]["gates"].items():
        lines.append(f"- `{name}`：{value}")
    lines.extend([
        "",
        "- 负采样保留全部正例；未观察候选按来源分支×名次难度桶固定哈希抽取，每个正例最多30个。target=0 仍只代表下一周未观察购买，不代表曝光后拒绝。",
        "- inactive 用户没有 Cold seed，继续使用冻结 Warm 顺序；这是 Warm-v1 延续策略，已从模型排序用户中明确分开。",
        "- 所有上游 score 来自 M5.4 通过的 temporal OOF/forward 谱系；2019-11-27 缺少更早模型时保持缺失，不使用自拟合分数。",
        f"- 独立证据校验：{verification['status'] if verification is not None else 'not_run'}；该状态只表示文件身份、时间谱系与门禁重算是否一致，不会把失败的模型门禁改成通过。",
        f"- 总耗时 {result['resources']['elapsed_seconds']:.2f} 秒；峰值工作集 {result['resources']['peak_working_set_bytes']/2**30:.2f} GiB；模型与预测文件 {result['resources']['artifact_bytes']/2**20:.2f} MiB。",
        "- optimistic all-articles、无库存与无曝光日志限制不变；若门禁失败，M6 与 final week 继续不运行。",
        "",
        "## Failure decomposition",
        "",
    ])
    for item in result["summary"]["failure_decomposition"]:
        lines.append(f"- {item}")
    if verification is not None and "outer_training_branch_supervision" in verification:
        branches = verification["outer_training_branch_supervision"]
        cold_shares = [row["cold_only"]["share_of_all_positive_pairs"] for row in branches.values()]
        density_ratios = [
            row["warm_only"]["positive_density"] / max(row["cold_only"]["positive_density"], 1e-15)
            for row in branches.values()
        ]
        rounds = [
            row["training"]["dual_fusion"]["inner_model"]["best_iteration"]
            for row in result["windows"].values()
        ]
        lines.extend([
            f"- 机制审计：cold-only 正例只占各窗 outer 训练正例的 {min(cold_shares):.2%}--{max(cold_shares):.2%}，warm-only 正例密度是 cold-only 的 {min(density_ratios):.1f}--{max(density_ratios):.1f} 倍；inner 选择的树轮数为 {'/'.join(str(value) for value in rounds)}。",
            "- 上述证据支持“总体 LambdaRank 目标下的来源监督不平衡”是失败的重要候选原因，但不能单独区分损失权重、组内采样、截断深度或特征表达哪一个是充分原因；在新阶段获批前应先做能区分这些因素的最小诊断。",
        ])
    lines.append("")
    return "\n".join(lines)


def render_existing(
    *, metrics_path: Path, report_path: Path, verification_path: Path | None = None,
) -> None:
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    verification = None
    if verification_path is not None:
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
    report_path.write_text(_render_report(result, verification=verification), encoding="utf-8")


def run(
    *, source_root: Path, m54_artifact_dir: Path, m54_metrics_path: Path,
    artifact_dir: Path, report_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    m54 = json.loads(m54_metrics_path.read_text(encoding="utf-8"))
    if m54.get("run_id") != M54_RUN_ID or not m54.get("gate", {}).get("passed"):
        raise RuntimeError("M5.5 requires authoritative passing M5.4")
    m3_path = source_root / "reports" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / "metrics.json"
    m3 = json.loads(m3_path.read_text(encoding="utf-8"))
    transactions = Path(m3["inputs"]["transactions"]["path"])
    if artifact_dir.exists() or report_dir.exists():
        raise FileExistsError(artifact_dir if artifact_dir.exists() else report_dir)
    artifact_dir.mkdir(parents=True)
    windows: dict[str, Any] = {}
    artifact_bytes = 0
    for window, protocol in ROLLING_PROTOCOL.items():
        print(f"M5.5 train/evaluate {window}", flush=True)
        root = artifact_dir / window
        root.mkdir()
        inner_train_paths = [_feature_path(m54_artifact_dir, value) for value in protocol["inner_train"]]
        inner_valid_path = _feature_path(m54_artifact_dir, protocol["inner_validation"])
        outer_train_paths = [_feature_path(m54_artifact_dir, value) for value in protocol["outer_train"]]
        outer_valid_path = _feature_path(m54_artifact_dir, protocol["outer_validation"])
        cutoff = protocol["outer_validation"]
        if cutoff >= FINAL_CUTOFF:
            raise RuntimeError("M5.5 must not read final week")
        validation_users = duckdb.connect().execute(
            f"SELECT DISTINCT customer_id FROM read_parquet({_literal(inner_valid_path)}) ORDER BY customer_id"
        ).fetchnumpy()["customer_id"].tolist()
        inner_truth, _ = _truth_and_counts(
            transactions_path=transactions, cutoff=protocol["inner_validation"],
            users=[str(value) for value in validation_users],
        )
        models: dict[str, Any] = {}
        training: dict[str, Any] = {}
        for name, spec in VARIANTS.items():
            features = spec["features"]
            inner_train, inner_sizes, inner_sample = load_training_sample(
                paths=inner_train_paths, features=features, warm_only=spec["warm_only"],
            )
            inner_valid, inner_valid_sizes = _load_validation(
                path=inner_valid_path, features=features, warm_only=spec["warm_only"],
            )
            inner_maps = build_category_maps(inner_train)
            inner_model, inner_evidence = train_ranker(
                train=inner_train, train_sizes=inner_sizes,
                validation=inner_valid, validation_sizes=inner_valid_sizes,
                validation_truth=inner_truth, features=features, category_maps=inner_maps,
                output_path=root / f"inner-{name}.txt",
            )
            rounds = inner_evidence["best_iteration"]
            del inner_model, inner_train, inner_valid
            gc.collect()
            outer_train, outer_sizes, outer_sample = load_training_sample(
                paths=outer_train_paths, features=features, warm_only=spec["warm_only"],
            )
            outer_maps = build_category_maps(outer_train)
            atomic_json(root / f"outer-{name}-category-maps.json", outer_maps)
            outer_model, outer_evidence = train_ranker(
                train=outer_train, train_sizes=outer_sizes,
                validation=None, validation_sizes=None, validation_truth=None,
                features=features, category_maps=outer_maps,
                output_path=root / f"outer-{name}.txt", rounds=rounds,
            )
            models[name] = (outer_model, features, outer_maps, spec["warm_only"])
            training[name] = {
                "inner_sampling": inner_sample, "inner_model": inner_evidence,
                "outer_sampling": outer_sample, "outer_model": outer_evidence,
                "outer_category_maps": file_identity(root / f"outer-{name}-category-maps.json"),
            }
            del outer_train
            gc.collect()
        union_frame, _ = _load_validation(path=outer_valid_path, features=DUAL_FEATURES, warm_only=False)
        users = union_frame.groupby("customer_id", sort=False, observed=True).size().index.astype(str).tolist()
        truth, counts = _truth_and_counts(transactions_path=transactions, cutoff=cutoff, users=users)
        baseline_db = Path(m3["development"][window]["scoring"]["evaluation_db"])
        baseline_top12 = _baseline_top12(baseline_db)
        evaluations: dict[str, Any] = {}
        baseline_ranks = np.full(len(union_frame), 999, dtype=np.int32)
        baseline_eval = _evaluate(
            frame=union_frame, top12=baseline_top12.rename(columns={"warm_rank": "final_rank"}).assign(
                candidate_rank=0, target=0, user_history_events_12w=0, warm_rank=lambda x: x["final_rank"],
                source_branch="warm_only", item_events_before_cutoff=0,
            ), ranks=baseline_ranks, baseline_top12=baseline_top12, truth=truth, counts=counts,
        )
        if abs(baseline_eval["segments"]["overall"]["map@12"] - WARM_MAP[window]) > 1e-12:
            raise RuntimeError(f"M5.5 Warm-v1 anchor mismatch: {window}")
        evaluations["frozen_warm_v1"] = baseline_eval
        prediction_output = union_frame[[
            "target_cutoff", "customer_id", "article_id", "candidate_rank", "target",
            "user_history_events_12w", "warm_rank", "source_branch", "item_events_before_cutoff",
        ]].copy()
        for name, (model, features, maps, warm_only) in models.items():
            frame = union_frame.loc[union_frame["warm_present"] == 1].copy() if warm_only else union_frame
            scores = model.predict(_prepare(frame, features, maps))
            top12, ranks = _rank_top12(frame, scores)
            evaluations[name] = _evaluate(
                frame=frame, top12=top12, ranks=ranks, baseline_top12=baseline_top12,
                truth=truth, counts=counts,
            )
            if warm_only:
                prediction_output[f"score_{name}"] = np.nan
                prediction_output.loc[frame.index, f"score_{name}"] = scores
            else:
                prediction_output[f"score_{name}"] = scores
        prediction_path = root / "outer-predictions.parquet"
        _write_parquet(prediction_output, prediction_path)
        branch_importance = [
            row for row in training["dual_fusion"]["outer_model"]["top_feature_importance"]
            if row["feature"] in set(DUAL_FEATURES) - set(WARM_FEATURES)
        ]
        windows[window] = {
            "protocol": protocol, "training": training, "evaluations": evaluations,
            "source_aware_feature_gain": branch_importance,
            "inputs": {"outer_features": file_identity(outer_valid_path), "warm_evaluation_db": file_identity(baseline_db)},
            "predictions": file_identity(prediction_path), "final_week": "not_run",
        }
        artifact_bytes += prediction_path.stat().st_size
        artifact_bytes += sum(
            path.stat().st_size for path in root.iterdir()
            if path.is_file() and path != prediction_path
        )
        del union_frame, prediction_output, models
        gc.collect()
    dual_deltas = {
        window: row["evaluations"]["dual_fusion"]["segments"]["overall"]["delta_vs_warm_map@12"]
        for window, row in windows.items()
    }
    warm_deltas = {
        window: row["evaluations"]["dual_fusion"]["segments"]["warm_21_plus"]["delta_vs_warm_map@12"]
        for window, row in windows.items()
    }
    inserted = sum(row["evaluations"]["dual_fusion"]["inserted_cold_sparse_positive_pairs"] for row in windows.values())
    removed = sum(row["evaluations"]["dual_fusion"]["removed_warm_positive_pairs"] for row in windows.values())
    mean_delta = float(np.mean(list(dual_deltas.values())))
    gates = {
        "mean_overall_map_strictly_above_warm": mean_delta > MAP_TOLERANCE,
        "at_least_3_of_4_outer_windows_non_degrading": sum(value >= -MAP_TOLERANCE for value in dual_deltas.values()) >= 3,
        "worst_window_within_0.0002": min(dual_deltas.values()) >= -WINDOW_TOLERANCE,
        "cold_only_truth_enters_top12_in_at_least_2_windows": sum(
            row["evaluations"]["dual_fusion"]["cold_only_truth_conversion"]["top12"] > 0
            for row in windows.values()
        ) >= 2,
        "warm_map_not_systematically_broken": (
            float(np.mean(list(warm_deltas.values()))) >= -WINDOW_TOLERANCE
            and sum(value < -WINDOW_TOLERANCE for value in warm_deltas.values()) <= 1
        ),
        "pair_loss_controlled_or_map_clearly_covers": inserted >= removed or mean_delta >= WINDOW_TOLERANCE,
        "oof_cutoff_candidate_feature_identity_pass": bool(m54["gate"]["passed"]) and all(
            row["final_week"] == "not_run" for row in windows.values()
        ),
    }
    failure = []
    if not gates["mean_overall_map_strictly_above_warm"]:
        failure.append("Dual Fusion 的四窗 mean MAP@12 未超过冻结 Warm-v1；新增候选没有形成可部署的总体净收益。")
    if not gates["cold_only_truth_enters_top12_in_at_least_2_windows"]:
        failure.append("cold-only 正例在少于两个窗口进入 Top12，瓶颈仍位于并集候选到最终位置的排序兑现。")
    if inserted < removed:
        failure.append(f"四窗插入 cold/sparse 正例 {inserted} 对，移除 Warm 正例 {removed} 对；局部替换损失仍未被统一排序完全消除。")
    if not failure:
        failure.append("预注册门禁全部通过；failure decomposition 不适用。")
    result = {
        "schema_version": "m5.5-source-aware-dual-channel-fusion-v1", "stage": "M5.5",
        "status": "measured", "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "candidate_union": "Warm Top150 union Student Cold Top50, deduplicated",
            "group_key": ["target_cutoff", "customer_id"],
            "negative_sampling": "branch x rank bucket fixed hash; all positives; <=30 unobserved per positive",
            "primary_model": "LightGBM LambdaRank", "inner_selection": "boosting rounds only",
            "variants": list(VARIANTS), "inactive_fallback": "exact frozen Warm order",
            "final_week": "not_run",
        },
        "inputs": {"m54_metrics": file_identity(m54_metrics_path), "m3_metrics": file_identity(m3_path)},
        "windows": windows,
        "summary": {
            "dual_window_delta_vs_warm_map@12": dual_deltas,
            "dual_mean_delta_vs_warm_map@12": mean_delta,
            "warm_window_delta_vs_warm_map@12": warm_deltas,
            "inserted_cold_sparse_positive_pairs": inserted,
            "removed_warm_positive_pairs": removed,
            "gates": gates, "gate_passed": all(gates.values()),
            "next_stage": "M5.6_or_M6_allowed" if all(gates.values()) else "stop_at_M5.5_failure_decomposition",
            "failure_decomposition": failure,
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": _peak_working_set_bytes(), "artifact_bytes": artifact_bytes,
        },
        "final_week": "not_run",
    }
    report_dir.mkdir(parents=True)
    atomic_json(report_dir / "M5_5_metrics.json", result)
    (report_dir / "M5_5_FINAL.md").write_text(_render_report(result), encoding="utf-8")
    return result
