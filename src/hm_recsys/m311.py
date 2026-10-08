from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m2 import _literal, build_category_maps
from .m310 import _enriched_path, _score_image_rows, load_image_sample
from .m33 import ROLLING_PROTOCOL


SCHEMA_VERSION = "m3.11a-image-precision-oracle-audit-v1"
AUTHORITATIVE_M310_RUN = "m3-10-v3-source-specific-image-admission-rank-fix"
FINAL_WEEK_CUTOFF = "2020-09-16"
DEPTHS = (1, 3, 5, 10, 20, 50, 100)
ORACLE_DEPTHS = (1, 3, 5, 10)
MAX_REPLACEMENTS = (1, 2)
PROTECTED_BASE_RANK = 7


ORDERINGS: dict[str, dict[str, Any]] = {
    "all_soft": {"filter": None, "column": "soft_rank", "ascending": True},
    "personalized": {"filter": "soft_personalized_present", "column": "soft_personalized_rank", "ascending": True},
    "global": {"filter": "soft_global_present", "column": "soft_global_rank", "ascending": True},
    "image_expert": {"filter": None, "column": "image_expert_score", "ascending": False},
    "direct_decay": {"filter": None, "column": "direct_visual_decay_max", "ascending": False},
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def validate_protocol(m310: dict[str, Any], m33: dict[str, Any]) -> None:
    if m310.get("run_id") != AUTHORITATIVE_M310_RUN or m310.get("evidence_status") != "authoritative":
        raise ValueError("M3.11A requires the authoritative M3.10 v3 run")
    if m310.get("contract", {}).get("final_week") != "not_run":
        raise ValueError("M3.11A refuses an M3.10 input that read the final week")
    if m33.get("contract", {}).get("final_week") not in {None, "not_run"}:
        raise ValueError("M3.11A refuses an M3.3 input that read the final week")
    cutoffs = [row["outer_validation"] for row in ROLLING_PROTOCOL.values()]
    if len(cutoffs) != 4 or any(cutoff >= FINAL_WEEK_CUTOFF for cutoff in cutoffs):
        raise RuntimeError("M3.11A requires four pre-final-week outer windows")


def _load_image_frame(*, enriched_path: Path, transactions_path: Path, cutoff: str) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        return connection.execute(
            f"""
            WITH warm AS (
              SELECT DISTINCT article_id FROM read_parquet({_literal(transactions_path)})
              WHERE t_dat<DATE '{cutoff}'
            )
            SELECT c.customer_id,c.article_id,c.target,c.user_history_events_12w,
              c.soft_rank,c.soft_personalized_present,c.soft_personalized_rank,
              c.soft_global_present,c.soft_global_rank,c.direct_visual_raw_max,
              c.direct_visual_decay_max,(w.article_id IS NULL)::UTINYINT AS item_is_cold
            FROM read_parquet({_literal(enriched_path)}) c
            LEFT JOIN warm w USING(article_id)
            WHERE c.base_present=0 AND c.soft_present=1 AND c.user_history_events_12w>0
            ORDER BY c.customer_id,c.soft_rank,c.article_id
            """
        ).fetchdf()
    finally:
        connection.close()


def _rank(frame: pd.DataFrame, ordering: str) -> pd.DataFrame:
    spec = ORDERINGS[ordering]
    ranked = frame
    filter_column = spec["filter"]
    if filter_column:
        ranked = ranked.loc[ranked[filter_column].astype(bool)]
    ranked = ranked.sort_values(
        ["evaluation_group", spec["column"], "article_id"],
        ascending=[True, bool(spec["ascending"]), True],
        kind="mergesort",
    ).copy()
    ranked["ordering_rank"] = ranked.groupby("evaluation_group", sort=False).cumcount() + 1
    ranked["ordering_rows"] = ranked.groupby("evaluation_group", sort=False)["article_id"].transform("size")
    return ranked


def _ordering_metrics(frame: pd.DataFrame, ordering: str) -> dict[str, Any]:
    ranked = _rank(frame, ordering)
    positive = ranked.loc[ranked["target"].astype(bool)]
    positive_pairs = int(len(positive))
    positive_users = int(positive["evaluation_group"].nunique())
    depths: dict[str, Any] = {}
    for depth in DEPTHS:
        top = ranked.loc[ranked["ordering_rank"] <= depth]
        top_positive = top.loc[top["target"].astype(bool)]
        rows = int(len(top))
        positives = int(len(top_positive))
        hit_users = int(top_positive["evaluation_group"].nunique())
        depths[str(depth)] = {
            "candidate_rows": rows,
            "positive_pairs": positives,
            "positive_density": positives / rows if rows else None,
            "positive_recall": positives / positive_pairs if positive_pairs else None,
            "hit_users": hit_users,
            "positive_user_hit_rate": hit_users / positive_users if positive_users else None,
        }
    first_positive = positive.groupby("evaluation_group", sort=False)["ordering_rank"].min()
    mrr = float((1.0 / first_positive).mean()) if not first_positive.empty else None
    top10_positive = positive.loc[positive["ordering_rank"] <= 10].copy()
    if top10_positive.empty:
        ndcg10 = 0.0 if positive_users else None
    else:
        top10_positive["discount"] = 1.0 / np.log2(top10_positive["ordering_rank"].astype(float) + 1.0)
        dcg = top10_positive.groupby("evaluation_group", sort=False)["discount"].sum()
        positives_per_user = positive.groupby("evaluation_group", sort=False).size()
        idcg = positives_per_user.map(
            lambda count: sum(1.0 / math.log2(rank + 1.0) for rank in range(1, min(int(count), 10) + 1))
        )
        ndcg10 = float((dcg.reindex(idcg.index, fill_value=0.0) / idcg).mean())
    percentile = positive["ordering_rank"] / positive["ordering_rows"] if positive_pairs else pd.Series(dtype=float)
    bands: dict[str, Any] = {}
    for name, lo, hi in (("rank_001_010", 1, 10), ("rank_011_050", 11, 50), ("rank_051_100", 51, 100)):
        band = ranked.loc[ranked["ordering_rank"].between(lo, hi)]
        rows = int(len(band))
        positives = int(band["target"].sum())
        bands[name] = {
            "candidate_rows": rows,
            "positive_pairs": positives,
            "positive_density": positives / rows if rows else None,
        }
    return {
        "ordering": ordering,
        "candidate_rows": int(len(ranked)),
        "users": int(ranked["evaluation_group"].nunique()),
        "positive_pairs": positive_pairs,
        "positive_users": positive_users,
        "depths": depths,
        "mrr": mrr,
        "ndcg@10": ndcg10,
        "median_positive_percentile_rank": float(percentile.median()) if not percentile.empty else None,
        "rank_bands": bands,
    }


def _load_base_frame(*, prediction_path: Path, transactions_path: Path, cutoff: str) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        frame = connection.execute(
            f"""
            WITH warm AS (
              SELECT DISTINCT article_id FROM read_parquet({_literal(transactions_path)}) WHERE t_dat<DATE '{cutoff}'
            ), future AS (
              SELECT DISTINCT customer_id,article_id FROM read_parquet({_literal(transactions_path)})
              WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY
            ), truth AS (
              SELECT f.customer_id,count(*)::BIGINT AS truth_count,
                count(*) FILTER(WHERE w.article_id IS NULL)::BIGINT AS cold_truth_count
              FROM future f LEFT JOIN warm w USING(article_id) GROUP BY f.customer_id
            )
            SELECT p.customer_id,p.article_id,p.candidate_rank,p.target,p.user_history_events_12w,p.score_anchor,
              (p.target=1 AND w.article_id IS NULL)::UTINYINT AS cold_target,t.truth_count,t.cold_truth_count
            FROM read_parquet({_literal(prediction_path)}) p
            JOIN truth t USING(customer_id) LEFT JOIN warm w USING(article_id)
            """
        ).fetchdf()
    finally:
        connection.close()
    frame["ordering_score"] = np.where(
        frame["user_history_events_12w"].to_numpy() > 0,
        frame["score_anchor"].to_numpy(),
        -frame["candidate_rank"].to_numpy(dtype=float),
    )
    frame = frame.sort_values(
        ["customer_id", "ordering_score", "candidate_rank", "article_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).copy()
    frame["base_final_rank"] = frame.groupby("customer_id", sort=False).cumcount() + 1
    return frame.loc[frame["base_final_rank"] <= 12].copy()


def _average_precision(labels: np.ndarray, truth_count: int) -> float:
    denominator = min(int(truth_count), 12)
    if denominator <= 0:
        return 0.0
    cumulative = np.cumsum(labels.astype(np.int64))
    ranks = np.arange(1, len(labels) + 1)
    return float(np.sum((cumulative / ranks) * labels) / denominator)


def _base_maps(base_top12: pd.DataFrame) -> dict[str, float]:
    overall: list[float] = []
    cold: list[float] = []
    for _customer, group in base_top12.groupby("customer_id", sort=False):
        group = group.sort_values("base_final_rank")
        overall.append(_average_precision(group["target"].to_numpy(), int(group["truth_count"].iloc[0])))
        cold_count = int(group["cold_truth_count"].iloc[0])
        if cold_count > 0:
            cold.append(_average_precision(group["cold_target"].to_numpy(), cold_count))
    return {"map@12": float(np.mean(overall)), "cold_map@12": float(np.mean(cold)) if cold else 0.0, "users": len(overall), "cold_users": len(cold)}


def _oracle_replace(
    *,
    base_top12: pd.DataFrame,
    image_frame: pd.DataFrame,
    ordering: str,
    image_depth: int,
    max_replacements: int,
) -> dict[str, Any]:
    image_ranked = _rank(image_frame, ordering)
    positive_images = image_ranked.loc[
        (image_ranked["ordering_rank"] <= image_depth) & image_ranked["target"].astype(bool)
    ]
    positive_counts = positive_images.groupby("customer_id", sort=False).size().to_dict()
    cold_positive_counts = positive_images.loc[positive_images["item_is_cold"].astype(bool)].groupby("customer_id", sort=False).size().to_dict()
    overall_values: list[float] = []
    cold_values: list[float] = []
    admitted = 0
    admitted_cold = 0
    benefited = 0
    available_mutable_negative_slots = 0
    for customer_id, group in base_top12.groupby("customer_id", sort=False):
        group = group.sort_values("base_final_rank")
        labels = group["target"].to_numpy(dtype=np.int8).copy()
        cold_labels = group["cold_target"].to_numpy(dtype=np.int8).copy()
        mutable_negative = [index for index in range(PROTECTED_BASE_RANK, min(12, len(labels))) if labels[index] == 0]
        available_mutable_negative_slots += len(mutable_negative)
        count = min(int(max_replacements), int(positive_counts.get(customer_id, 0)), len(mutable_negative))
        cold_count = min(count, int(cold_positive_counts.get(customer_id, 0)))
        if count:
            benefited += 1
            admitted += count
            admitted_cold += cold_count
            for offset, index in enumerate(mutable_negative[:count]):
                labels[index] = 1
                if offset < cold_count:
                    cold_labels[index] = 1
        overall_values.append(_average_precision(labels, int(group["truth_count"].iloc[0])))
        truth_cold = int(group["cold_truth_count"].iloc[0])
        if truth_cold > 0:
            cold_values.append(_average_precision(cold_labels, truth_cold))
    return {
        "ordering": ordering,
        "image_depth": int(image_depth),
        "max_replacements": int(max_replacements),
        "oracle_map@12": float(np.mean(overall_values)),
        "oracle_cold_map@12": float(np.mean(cold_values)) if cold_values else 0.0,
        "admitted_image_positive_pairs": int(admitted),
        "admitted_cold_image_positive_pairs": int(admitted_cold),
        "benefited_users": int(benefited),
        "displaced_base_positive_pairs": 0,
        "available_mutable_negative_slots": int(available_mutable_negative_slots),
    }


def _pooled_metrics(frames: list[pd.DataFrame]) -> dict[str, Any]:
    pooled = pd.concat(frames, ignore_index=True)
    return {name: _ordering_metrics(pooled, name) for name in ORDERINGS}


def _summary(development: dict[str, Any], pooled: dict[str, Any]) -> dict[str, Any]:
    expert_deltas: dict[str, Any] = {}
    for window, row in development.items():
        source = row["orderings"]["all_soft"]
        expert = row["orderings"]["image_expert"]
        expert_deltas[window] = {
            "recall@5": float(expert["depths"]["5"]["positive_recall"] - source["depths"]["5"]["positive_recall"]),
            "ndcg@10": float(expert["ndcg@10"] - source["ndcg@10"]),
            "mrr": float(expert["mrr"] - source["mrr"]),
        }
    pooled_recall_delta = float(
        pooled["image_expert"]["depths"]["5"]["positive_recall"]
        - pooled["all_soft"]["depths"]["5"]["positive_recall"]
    )
    positive_recall_windows = sum(row["recall@5"] > 0 for row in expert_deltas.values())
    all_metric_direction_windows = sum(
        row["recall@5"] > 0 and row["ndcg@10"] > 0 and row["mrr"] > 0
        for row in expert_deltas.values()
    )
    expert_gate = bool(positive_recall_windows >= 3 and all_metric_direction_windows >= 3 and pooled_recall_delta >= 0.05)

    oracle_configs: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for window, row in development.items():
        base_map = float(row["base"]["map@12"])
        for record in row["oracle"]:
            enriched = dict(record)
            enriched["window"] = window
            enriched["delta_map@12"] = float(record["oracle_map@12"] - base_map)
            oracle_configs.setdefault((record["ordering"], int(record["image_depth"]), int(record["max_replacements"])), []).append(enriched)
    ranked_configs = []
    for key, records in oracle_configs.items():
        ranked_configs.append({
            "ordering": key[0],
            "image_depth": key[1],
            "max_replacements": key[2],
            "positive_windows": sum(row["delta_map@12"] > 1e-12 for row in records),
            "mean_delta_map@12": float(np.mean([row["delta_map@12"] for row in records])),
            "admitted_image_positive_pairs": int(sum(row["admitted_image_positive_pairs"] for row in records)),
            "window_results": records,
        })
    ranked_configs.sort(key=lambda row: (row["positive_windows"], row["mean_delta_map@12"], row["admitted_image_positive_pairs"]), reverse=True)
    best_oracle = ranked_configs[0]
    oracle_gate = bool(best_oracle["positive_windows"] >= 3 and best_oracle["admitted_image_positive_pairs"] >= 10)

    soft_top = pooled["all_soft"]["rank_bands"]["rank_001_010"]
    soft_tail = pooled["all_soft"]["rank_bands"]["rank_051_100"]
    density_lift = (
        float(soft_top["positive_density"] / soft_tail["positive_density"])
        if soft_top["positive_density"] is not None and soft_tail["positive_density"] not in {None, 0}
        else None
    )
    precision_gradient_gate = bool(density_lift is not None and density_lift >= 2.0)
    all_soft_rows = int(pooled["all_soft"]["candidate_rows"])
    all_soft_positives = int(pooled["all_soft"]["positive_pairs"])
    personalized_rows = int(pooled["personalized"]["candidate_rows"])
    personalized_positives = int(pooled["personalized"]["positive_pairs"])
    if oracle_gate and expert_gate:
        recommendation = "constrained_third_ranker_then_scale_if_uncertain"
    elif oracle_gate and precision_gradient_gate:
        recommendation = "high_precision_subpool_then_constrained_third_ranker"
    else:
        recommendation = "retrieval_precision_or_representation_first"
    return {
        "expert_vs_source_window_deltas": expert_deltas,
        "pooled_expert_recall@5_delta": pooled_recall_delta,
        "expert_positive_recall_windows": positive_recall_windows,
        "expert_all_metric_direction_windows": all_metric_direction_windows,
        "expert_internal_ranking_gate": expert_gate,
        "soft_top10_to_rank51_100_density_lift": density_lift,
        "precision_gradient_gate": precision_gradient_gate,
        "personalized_candidate_row_fraction": personalized_rows / all_soft_rows if all_soft_rows else None,
        "personalized_positive_pair_retention": personalized_positives / all_soft_positives if all_soft_positives else None,
        "pooled_top5_positive_pairs": {
            name: int(metric["depths"]["5"]["positive_pairs"])
            for name, metric in pooled.items()
        },
        "best_local_replacement_oracle": best_oracle,
        "local_replacement_oracle_gate": oracle_gate,
        "recommendation": recommendation,
    }


def _render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    recommendation = {
        "constrained_third_ranker_then_scale_if_uncertain": "先训练受约束的第三排序器；若方向稳定但样本不足，再做30%扩样。",
        "high_precision_subpool_then_constrained_third_ranker": "先从现有图片源截取高精度子池，再训练受约束的第三排序器；暂不扩大样本。",
        "retrieval_precision_or_representation_first": "先改图片召回精度或表示，暂不训练第三排序器和扩样。",
    }[summary["recommendation"]]
    lines = [
        "# M3.11A：图片召回精度、内部排序与局部替换上限审计", "",
        "## 结论", "",
        f"- 运行状态：{result['status']}；数据为10%用户开发样本；最终验证周：{result['contract']['final_week']}。",
        f"- 图片专家内部排序门槛：{summary['expert_internal_ranking_gate']}；pooled Recall@5 相对原始 soft 顺序变化 {summary['pooled_expert_recall@5_delta']:+.6f}。",
        f"- 原始 soft Top10 相对第51--100名的正例密度倍数：{summary['soft_top10_to_rank51_100_density_lift'] if summary['soft_top10_to_rank51_100_density_lift'] is not None else 'NA'}。",
        f"- 个性化路径使用 all-soft 的 {summary['personalized_candidate_row_fraction']:.2%} 候选行，保留 {summary['personalized_positive_pair_retention']:.2%} 图片正例候选观察。",
        f"- 四窗合并 Top5 图片正例数：原始 soft={summary['pooled_top5_positive_pairs']['all_soft']}、图片专家={summary['pooled_top5_positive_pairs']['image_expert']}、直接视觉时间衰减={summary['pooled_top5_positive_pairs']['direct_decay']}。",
        f"- 局部替换 Oracle 门槛：{summary['local_replacement_oracle_gate']}。",
        f"- 下一步建议：{recommendation}", "",
        "## 术语、单位与边界", "",
        "- `outer development window`（外层开发窗口，行业常见的时间验证叫法）：以某个 cutoff（时间截点）之前的数据构造候选和特征，并以其后7天购买作为标签；本报告共四窗，均不含最终验证周。",
        "- `base`（基础排序，本项目自定义参照）：冻结的 M3.3 base-300 最终商品顺序；本次审计只研究基础排序未包含的图片候选。",
        "- `active user`（可使用图片专家的用户，本项目自定义分组）：在当前窗口拥有可评分 soft-shallow 图片候选的评测用户；本报告的图片池统计只以这组用户为对象。",
        "- `soft-shallow`（软约束浅层图片召回，本项目自定义召回路）：M3.7 冻结的图片候选生成方式；用视觉近邻和属性季节分数软融合，不用季节硬门槛，并合并个性化与全局两条路径。",
        "- `candidate observation/positive pair`（候选观察/图片正例观察）：一个 `(cutoff, user, item)` 用户—商品候选行；若商品出现在该用户 cutoff 后7天购买中，该行记为正例。相同用户—商品若出现在不同窗口，按不同观察计数。",
        "- `ordering`（图片池内部顺序）：固定某条图片候选路径后，用来源名次、图片专家分数或直接视觉分数得到的候选先后次序；它不是最终推荐列表。",
        "- `positive density`（图片候选正例密度，行业通用统计）：某一候选集合中命中未来7天购买的用户—商品候选行数，除以该集合全部用户—商品候选行数；未购买仅是未观测，不是可靠负例。",
        "- `positive recall`（图片正例召回率）：排序前K中图片正例用户—商品对数，除以同一路径完整图片池内的图片正例用户—商品对数。",
        "- `positive-user HitRate`（正例用户命中率）：图片池中至少存在一个正例的用户里，前K至少命中一个正例的用户比例。",
        "- `NDCG@10`（归一化折损累计增益，行业通用排序指标）：在每个正例用户的图片池内衡量正例是否集中在前10，再对正例用户平均。",
        "- `MRR`（首个正例倒数排名，行业通用排序指标）：每个正例用户第一个图片正例名次的倒数，再对正例用户平均。",
        "- `all-soft/personalized/global`（本项目图片候选路径）：分别表示全部 soft-shallow 图片候选、个性化路径候选和全局原型路径候选；统计对象均排除 base 已有商品。",
        "- `direct-decay`（直接视觉时间衰减顺序，本项目自定义名）：按用户近期图片 seed 与候选的最大时间衰减余弦相似度降序。",
        "- `image expert`（图片专家）：M3.10 v3 只在 soft-shallow 图片候选内部训练的 LambdaRank。",
        "- `pooled`（跨窗口合并统计）：把每个 cutoff、用户、商品视为独立候选观察后合并四个 outer 开发窗口，不是最终周结果。",
        "- `local replacement Oracle`（局部替换理论上限，本项目诊断）：保护基础排序1--7名，允许开发标签从图片前N选正例替换基础8--12名中的非正例；可放弃替换，每用户最多1或2项。它使用未来标签，只能表示上限，不能部署。", "",
        "## 四窗合并的图片池效率", "",
        "候选占比和正例保留率都以 all-soft 为参照；Top5正例密度分母是该排序在所有 active 用户产生的Top5候选行。", "",
        "| ordering | pool candidate rows | pool positive pairs | pool density | Top5 positive pairs | Top5 density |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ordering in ORDERINGS:
        metric = result["pooled_orderings"][ordering]
        top5 = metric["depths"]["5"]
        pool_density = metric["positive_pairs"] / metric["candidate_rows"] if metric["candidate_rows"] else 0.0
        lines.append(f"| {ordering} | {metric['candidate_rows']} | {metric['positive_pairs']} | {pool_density:.8f} | {top5['positive_pairs']} | {top5['positive_density']:.8f} |")
    lines.extend([
        "",
        "## 图片召回 Top10 精度与覆盖", "",
        "候选行与正例的单位都是 `(cutoff, user, item)` 候选观察；Recall 分母是该窗口、该路径完整图片池的正例候选数。", "",
        "| window | ordering | positive pairs in pool | Top10 rows | Top10 positives | Top10 density | Top10 positive recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["development"].items():
        for ordering in ORDERINGS:
            metric = row["orderings"][ordering]
            top = metric["depths"]["10"]
            lines.append(f"| {window} | {ordering} | {metric['positive_pairs']} | {top['candidate_rows']} | {top['positive_pairs']} | {top['positive_density']:.8f} | {top['positive_recall']:.6f} |")
    lines.extend([
        "", "## 图片专家与原始 soft 顺序", "",
        "下表固定同一 all-soft 候选集合；正值才表示图片专家改善内部排序。", "",
        "| window | image positive pairs | source Recall@5 | expert Recall@5 | delta | NDCG@10 delta | MRR delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for window, row in result["development"].items():
        source = row["orderings"]["all_soft"]
        expert = row["orderings"]["image_expert"]
        delta = summary["expert_vs_source_window_deltas"][window]
        lines.append(f"| {window} | {source['positive_pairs']} | {source['depths']['5']['positive_recall']:.6f} | {expert['depths']['5']['positive_recall']:.6f} | {delta['recall@5']:+.6f} | {delta['ndcg@10']:+.6f} | {delta['mrr']:+.6f} |")
    best = summary["best_local_replacement_oracle"]
    lines.extend([
        "", "## 最佳局部替换 Oracle", "",
        f"最佳配置：ordering={best['ordering']}，图片池深度={best['image_depth']}，每用户最多替换={best['max_replacements']}。",
        "`delta MAP` 的分母是该窗口 M3.3 候选用户；admitted positives 单位是图片正例用户—商品对。", "",
        "| window | base MAP@12 | oracle MAP@12 | delta MAP | admitted positives | benefited users | displaced base positives |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in best["window_results"]:
        base_map = result["development"][row["window"]]["base"]["map@12"]
        lines.append(f"| {row['window']} | {base_map:.6f} | {row['oracle_map@12']:.6f} | {row['delta_map@12']:+.6f} | {row['admitted_image_positive_pairs']} | {row['benefited_users']} | {row['displaced_base_positive_pairs']} |")
    lines.extend([
        "", "## 决策解释", "",
        "- measured（已测量）：上述密度、内部排序和 Oracle 数字均来自四个 outer 开发窗口；最终周未读取。",
        "- inference（基于证据的推断）：下一步只按预注册三项门槛选择召回改造、受约束第三排序器或30%扩样，不把 Oracle 当作可实现成绩。",
        "- unresolved（未解决）：没有曝光日志，因此正例密度衡量购买标签稀疏性，不等同于真实点击率或推荐曝光后的转化率。", "",
        f"机器可读证据：`{result['artifacts']['metrics']}`。", "",
    ])
    return "\n".join(lines)


def run_audit(*, m310_metrics_path: Path, m33_metrics_path: Path, transactions_path: Path, output_dir: Path, run_id: str) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    m310 = _read_json(m310_metrics_path)
    m33 = _read_json(m33_metrics_path)
    validate_protocol(m310, m33)
    features = list(m310["features"])
    enriched_root = Path(next(iter(m310["enriched_cache"].values()))["path"]).parents[1]
    frames: list[pd.DataFrame] = []
    development: dict[str, Any] = {}
    try:
        import lightgbm as lgb

        for window, protocol in ROLLING_PROTOCOL.items():
            window_started = time.perf_counter()
            cutoff = protocol["outer_validation"]
            print(f"M3.11A {window}: reconstructing image expert scores", flush=True)
            outer_train, _sizes, _sampling = load_image_sample(
                [_enriched_path(enriched_root, train_cutoff) for train_cutoff in protocol["outer_train"]],
                features,
                seed=int(m310["config"]["seed"]),
            )
            category_maps = build_category_maps(outer_train)
            model_path = Path(m310["development"][window]["outer"]["model"]["model_path"])
            if _identity(model_path)["sha256"] != m310["development"][window]["outer"]["model"]["model_sha256"]:
                raise ValueError(f"M3.11A image model hash drift: {window}")
            model = lgb.Booster(model_file=str(model_path))
            enriched_path = _enriched_path(enriched_root, cutoff)
            scores = _score_image_rows(model, enriched_path, features, category_maps)
            image = _load_image_frame(enriched_path=enriched_path, transactions_path=transactions_path, cutoff=cutoff)
            image = image.merge(scores[["customer_id", "article_id", "image_expert_score"]], on=["customer_id", "article_id"], how="left", validate="one_to_one")
            if image["image_expert_score"].isna().any() or image.duplicated(["customer_id", "article_id"]).any():
                raise RuntimeError(f"M3.11A image score coverage failed: {window}")
            image["window"] = window
            image["evaluation_group"] = window + "|" + image["customer_id"].astype(str)
            orderings = {name: _ordering_metrics(image, name) for name in ORDERINGS}

            prediction_path = Path(m33["development"][window]["scoring"]["prediction_path"])
            prediction_identity = _identity(prediction_path)
            if prediction_identity["sha256"] != m33["development"][window]["scoring"]["prediction_sha256"]:
                raise ValueError(f"M3.11A base prediction hash drift: {window}")
            base = _load_base_frame(prediction_path=prediction_path, transactions_path=transactions_path, cutoff=cutoff)
            base_metrics = _base_maps(base)
            expected = float(m33["development"][window]["evaluation"]["orderings"]["anchor__inactive_rrf"]["segments"]["overall"]["map@12"])
            if abs(float(base_metrics["map@12"]) - expected) > 1e-12:
                raise RuntimeError(f"M3.11A base reproduction failed: {window} {base_metrics['map@12']} != {expected}")
            oracle = [
                _oracle_replace(
                    base_top12=base,
                    image_frame=image,
                    ordering=ordering,
                    image_depth=depth,
                    max_replacements=max_replacements,
                )
                for ordering in ORDERINGS
                for depth in ORACLE_DEPTHS
                for max_replacements in MAX_REPLACEMENTS
            ]
            development[window] = {
                "cutoff": cutoff,
                "image_candidate_rows": int(len(image)),
                "image_candidate_users": int(image["customer_id"].nunique()),
                "image_positive_pairs": int(image["target"].sum()),
                "image_cold_positive_pairs": int(image.loc[image["item_is_cold"].astype(bool), "target"].sum()),
                "orderings": orderings,
                "base": base_metrics,
                "oracle": oracle,
                "inputs": {"enriched": _identity(enriched_path), "image_model": _identity(model_path), "base_predictions": prediction_identity},
                "elapsed_seconds": time.perf_counter() - window_started,
            }
            frames.append(image)
            del outer_train, scores, image, base, model
        pooled = _pooled_metrics(frames)
        summary = _summary(development, pooled)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M3.11A",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "four M3.10 outer development windows; 10% sampled users; final week not run",
            "contract": {
                "image_population": "active users; base_present=0 AND soft_present=1",
                "depths": list(DEPTHS),
                "oracle_depths": list(ORACLE_DEPTHS),
                "protected_base_ranks": f"1-{PROTECTED_BASE_RANK}",
                "mutable_base_ranks": f"{PROTECTED_BASE_RANK + 1}-12",
                "max_replacements": list(MAX_REPLACEMENTS),
                "catalog": "optimistic_all_articles",
                "final_week": "not_run",
                "oracle_uses_development_truth": True,
            },
            "inputs": {"m310_metrics": _identity(m310_metrics_path), "m33_metrics": _identity(m33_metrics_path), "transactions": _identity(transactions_path)},
            "development": development,
            "pooled_orderings": pooled,
            "summary": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M3_11A_FINAL.md"
        result["artifacts"] = {"metrics": str(metrics_path.resolve()), "report": str(report_path.resolve())}
        _write_json(metrics_path, result)
        report_path.write_text(_render_report(result), encoding="utf-8", newline="\n")
        return result
    except Exception as error:
        _write_json(output_dir / f"failure-{time.time_ns()}.json", {
            "schema_version": "m3.11a-failure-v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
        })
        raise
