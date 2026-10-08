from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from .m33 import ROLLING_PROTOCOL


SCHEMA_VERSION = "m3.9-image-positive-ranking-audit-v1"
VARIANTS = (
    "base_historical_400",
    "base_soft_shallow_400",
    "base_historical_soft_500",
)
DETAILED_VARIANT = "base_historical_soft_500"
FINAL_WEEK_CUTOFF = "2020-09-16"


def validate_protocol() -> None:
    cutoffs = {
        cutoff
        for row in ROLLING_PROTOCOL.values()
        for key in ("inner_train", "outer_train")
        for cutoff in row[key]
    }
    cutoffs.update(row["inner_validation"] for row in ROLLING_PROTOCOL.values())
    cutoffs.update(row["outer_validation"] for row in ROLLING_PROTOCOL.values())
    if FINAL_WEEK_CUTOFF in cutoffs or any(cutoff >= FINAL_WEEK_CUTOFF for cutoff in cutoffs):
        raise RuntimeError("M3.9 must not read the final week")
    if len(ROLLING_PROTOCOL) != 4:
        raise RuntimeError("M3.9 requires the frozen four-window protocol")


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rows(cursor: duckdb.DuckDBPyConnection, query: str, parameters: list[Any] | None = None) -> list[dict[str, Any]]:
    result = cursor.execute(query, parameters or [])
    columns = [item[0] for item in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def _rate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        denominator = int(row["candidate_observations"])
        positives = int(row["positive_observations"])
        row["positive_density"] = positives / denominator if denominator else None
    return rows


def _rank_bucket_sql(column: str) -> str:
    return (
        f"CASE WHEN {column} BETWEEN 1 AND 10 THEN 'rank_001_010' "
        f"WHEN {column} BETWEEN 11 AND 50 THEN 'rank_011_050' "
        f"WHEN {column} BETWEEN 51 AND 100 THEN 'rank_051_100' ELSE 'rank_other' END"
    )


def _variant_supervision(path: Path) -> list[dict[str, Any]]:
    connection = duckdb.connect()
    try:
        return _rate_rows(
            _rows(
                connection,
                """
                SELECT m38_source_layer AS source_layer,
                       count(*)::BIGINT AS candidate_observations,
                       sum(target)::BIGINT AS positive_observations,
                       count(DISTINCT customer_id)::BIGINT AS users
                FROM read_parquet(?)
                GROUP BY m38_source_layer
                ORDER BY m38_source_layer
                """,
                [str(path)],
            )
        )
    finally:
        connection.close()


def audit_supervision(metrics: dict[str, Any]) -> dict[str, Any]:
    per_cutoff: dict[str, Any] = {}
    for cutoff, variants in metrics["variant_cache"].items():
        per_cutoff[cutoff] = {}
        for variant in VARIANTS:
            per_cutoff[cutoff][variant] = _variant_supervision(Path(variants[variant]["path"]))

    def aggregate(cutoffs: list[str], variant: str) -> list[dict[str, Any]]:
        combined: dict[str, dict[str, int]] = {}
        for cutoff in cutoffs:
            for row in per_cutoff[cutoff][variant]:
                entry = combined.setdefault(
                    str(row["source_layer"]),
                    {"candidate_observations": 0, "positive_observations": 0},
                )
                entry["candidate_observations"] += int(row["candidate_observations"])
                entry["positive_observations"] += int(row["positive_observations"])
        result = [
            {"source_layer": layer, **values}
            for layer, values in sorted(combined.items())
        ]
        return _rate_rows(result)

    all_cutoffs = sorted(per_cutoff)
    all_ten = {variant: aggregate(all_cutoffs, variant) for variant in VARIANTS}
    per_model: dict[str, Any] = {}
    for window, protocol in ROLLING_PROTOCOL.items():
        per_model[window] = {}
        for variant in VARIANTS:
            per_model[window][variant] = {
                "inner_train_cutoffs": list(protocol["inner_train"]),
                "inner_train": aggregate(list(protocol["inner_train"]), variant),
                "outer_train_cutoffs": list(protocol["outer_train"]),
                "outer_train": aggregate(list(protocol["outer_train"]), variant),
            }
    return {
        "unit": "one (cutoff, customer_id, article_id) candidate-label row before negative sampling",
        "per_cutoff": per_cutoff,
        "all_ten_cutoffs": all_ten,
        "per_model_training_scope": per_model,
    }


def _prepare_members(connection: duckdb.DuckDBPyConnection, path: Path, transactions: Path, cutoff: str) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE warm_items AS
        SELECT DISTINCT article_id FROM read_parquet(?) WHERE t_dat < CAST(? AS DATE);
        """,
        [str(transactions), cutoff],
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE image_members AS
        WITH source AS (SELECT * FROM read_parquet(?)), members AS (
          SELECT customer_id,article_id,target,user_history_events_12w,
                 'historical' AS image_source,historical_rank AS source_rank,
                 CASE WHEN historical_personalized_rank IS NOT NULL AND historical_global_rank IS NOT NULL THEN 'personalized_and_global'
                      WHEN historical_personalized_rank IS NOT NULL THEN 'personalized_only'
                      WHEN historical_global_rank IS NOT NULL THEN 'global_only' ELSE 'route_missing' END AS route,
                 historical_score AS primary_score,NULL::DOUBLE AS visual_score,NULL::DOUBLE AS season_score,
                 NULL::DOUBLE AS seed_support
          FROM source WHERE base_present=0 AND historical_present=1
          UNION ALL
          SELECT customer_id,article_id,target,user_history_events_12w,
                 'soft_shallow' AS image_source,soft_rank AS source_rank,
                 CASE WHEN soft_personalized_present=1 AND soft_global_present=1 THEN 'personalized_and_global'
                      WHEN soft_personalized_present=1 THEN 'personalized_only'
                      WHEN soft_global_present=1 THEN 'global_only' ELSE 'route_missing' END AS route,
                 soft_combined_score AS primary_score,soft_visual_score AS visual_score,
                 soft_season_score AS season_score,soft_seed_support::DOUBLE AS seed_support
          FROM source WHERE base_present=0 AND soft_present=1
        )
        SELECT m.*,
               CASE WHEN m.user_history_events_12w>0 THEN 'active' ELSE 'inactive' END AS user_activity,
               CASE WHEN w.article_id IS NULL THEN 'cold' ELSE 'warm' END AS item_temperature
        FROM members m LEFT JOIN warm_items w USING(article_id)
        """,
        [str(path)],
    )


def audit_source_precision(metrics: dict[str, Any], transactions: Path) -> dict[str, Any]:
    connection = duckdb.connect()
    connection.execute("SET threads=8")
    per_cutoff: dict[str, Any] = {}
    try:
        for cutoff, variants in metrics["variant_cache"].items():
            _prepare_members(connection, Path(variants[DETAILED_VARIANT]["path"]), transactions, cutoff)
            rank_rows = _rate_rows(
                _rows(
                    connection,
                    f"""
                    SELECT image_source,{_rank_bucket_sql('source_rank')} AS source_rank_bucket,
                           count(*)::BIGINT AS candidate_observations,
                           sum(target)::BIGINT AS positive_observations,
                           count(DISTINCT customer_id)::BIGINT AS users
                    FROM image_members GROUP BY image_source,source_rank_bucket
                    ORDER BY image_source,source_rank_bucket
                    """,
                )
            )
            route_rows = _rate_rows(
                _rows(
                    connection,
                    """
                    SELECT image_source,route,count(*)::BIGINT AS candidate_observations,
                           sum(target)::BIGINT AS positive_observations,
                           count(DISTINCT customer_id)::BIGINT AS users
                    FROM image_members GROUP BY image_source,route ORDER BY image_source,route
                    """,
                )
            )
            segment_rows = _rate_rows(
                _rows(
                    connection,
                    """
                    SELECT image_source,user_activity,item_temperature,
                           count(*)::BIGINT AS candidate_observations,
                           sum(target)::BIGINT AS positive_observations,
                           count(DISTINCT customer_id)::BIGINT AS users
                    FROM image_members GROUP BY image_source,user_activity,item_temperature
                    ORDER BY image_source,user_activity,item_temperature
                    """,
                )
            )
            score_summary = _rows(
                connection,
                """
                WITH long AS (
                  SELECT image_source,target,'primary_score' AS feature,primary_score AS value FROM image_members
                  UNION ALL SELECT image_source,target,'visual_score',visual_score FROM image_members
                  UNION ALL SELECT image_source,target,'season_score',season_score FROM image_members
                  UNION ALL SELECT image_source,target,'seed_support',seed_support FROM image_members
                  UNION ALL SELECT image_source,target,'source_rank',source_rank::DOUBLE FROM image_members)
                SELECT image_source,target,feature,count(*)::BIGINT AS candidate_observations,
                       count(value)::BIGINT AS non_missing_observations,
                       avg(value) AS mean_value,median(value) AS median_value,
                       quantile_cont(value,0.25) AS p25_value,quantile_cont(value,0.75) AS p75_value
                FROM long GROUP BY image_source,target,feature ORDER BY image_source,feature,target
                """,
            )
            score_deciles = _rate_rows(
                _rows(
                    connection,
                    """
                    WITH scored AS (
                      SELECT *,ntile(10) OVER(PARTITION BY image_source ORDER BY primary_score DESC NULLS LAST,
                        customer_id,article_id) AS score_decile
                      FROM image_members), grouped AS (
                      SELECT image_source,score_decile,count(*)::BIGINT AS candidate_observations,
                             sum(target)::BIGINT AS positive_observations
                      FROM scored GROUP BY image_source,score_decile)
                    SELECT * FROM grouped ORDER BY image_source,score_decile
                    """,
                )
            )
            per_cutoff[cutoff] = {
                "source_rank_buckets": rank_rows,
                "routes": route_rows,
                "activity_temperature": segment_rows,
                "score_summary": score_summary,
                "primary_score_deciles": score_deciles,
            }
    finally:
        connection.close()

    def combine(section: str, dimensions: tuple[str, ...]) -> list[dict[str, Any]]:
        aggregate: dict[tuple[Any, ...], dict[str, Any]] = {}
        for payload in per_cutoff.values():
            for row in payload[section]:
                key = tuple(row[name] for name in dimensions)
                entry = aggregate.setdefault(
                    key,
                    {**dict(zip(dimensions, key, strict=True)), "candidate_observations": 0, "positive_observations": 0},
                )
                entry["candidate_observations"] += int(row["candidate_observations"])
                entry["positive_observations"] += int(row["positive_observations"])
        return _rate_rows(list(aggregate.values()))

    return {
        "unit_note": "source-membership candidate observations; a historical-soft overlap candidate appears once under each source",
        "per_cutoff": per_cutoff,
        "all_ten_rank_buckets": combine("source_rank_buckets", ("image_source", "source_rank_bucket")),
        "all_ten_routes": combine("routes", ("image_source", "route")),
        "all_ten_activity_temperature": combine(
            "activity_temperature", ("image_source", "user_activity", "item_temperature")
        ),
        "all_ten_primary_score_deciles": combine(
            "primary_score_deciles", ("image_source", "score_decile")
        ),
    }


def audit_ranking_funnel(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    thresholds = (12, 20, 50, 100, 200)
    for window, variants in metrics["development"].items():
        result[window] = {}
        outer_cutoff = ROLLING_PROTOCOL[window]["outer_validation"]
        for variant in VARIANTS:
            dataset = Path(metrics["variant_cache"][outer_cutoff][variant]["path"])
            evaluation_db = Path(variants[variant]["scoring"]["evaluation_db"])
            connection = duckdb.connect(str(evaluation_db), read_only=True)
            try:
                select_thresholds = ",".join(
                    f"count(*) FILTER(WHERE model_rank<={value})::BIGINT AS model_top{value}_positive_observations"
                    for value in thresholds
                )
                rows = _rows(
                    connection,
                    f"""
                    WITH image_truth AS (
                      SELECT d.customer_id,d.article_id,d.candidate_rank AS initial_rank,
                             d.m38_source_layer AS source_layer,
                             CASE WHEN d.user_history_events_12w>0 THEN 'active' ELSE 'inactive' END AS user_activity,
                             t.item_temperature,r.candidate_rank AS model_rank
                      FROM read_parquet(?) d JOIN m38_ranked r USING(customer_id,article_id)
                      JOIN m38_truth t USING(customer_id,article_id)
                      WHERE d.base_present=0 AND d.target=1)
                    SELECT source_layer,user_activity,item_temperature,
                           count(*)::BIGINT AS image_truth_candidate_observations,
                           count(DISTINCT customer_id)::BIGINT AS users,
                           {select_thresholds},
                           count(*) FILTER(WHERE model_rank<initial_rank)::BIGINT AS improved_rank_observations,
                           min(model_rank)::BIGINT AS best_model_rank,
                           median(model_rank) AS median_model_rank,
                           quantile_cont(model_rank,0.25) AS p25_model_rank,
                           quantile_cont(model_rank,0.75) AS p75_model_rank,
                           max(model_rank)::BIGINT AS worst_model_rank,
                           median(initial_rank) AS median_initial_rank
                    FROM image_truth GROUP BY source_layer,user_activity,item_temperature
                    ORDER BY source_layer,user_activity,item_temperature
                    """,
                    [str(dataset)],
                )
                totals = _rows(
                    connection,
                    f"""
                    WITH image_truth AS (
                      SELECT d.customer_id,d.article_id,d.candidate_rank AS initial_rank,
                             CASE WHEN d.user_history_events_12w>0 THEN 'active' ELSE 'inactive' END AS user_activity,
                             r.candidate_rank AS model_rank
                      FROM read_parquet(?) d JOIN m38_ranked r USING(customer_id,article_id)
                      JOIN m38_truth t USING(customer_id,article_id)
                      WHERE d.base_present=0 AND d.target=1)
                    SELECT user_activity,count(*)::BIGINT AS image_truth_candidate_observations,
                           count(DISTINCT customer_id)::BIGINT AS users,{select_thresholds},
                           count(*) FILTER(WHERE model_rank<initial_rank)::BIGINT AS improved_rank_observations,
                           min(model_rank)::BIGINT AS best_model_rank,median(model_rank) AS median_model_rank,
                           quantile_cont(model_rank,0.25) AS p25_model_rank,
                           quantile_cont(model_rank,0.75) AS p75_model_rank,max(model_rank)::BIGINT AS worst_model_rank,
                           median(initial_rank) AS median_initial_rank
                    FROM image_truth GROUP BY user_activity ORDER BY user_activity
                    """,
                    [str(dataset)],
                )
                result[window][variant] = {"cutoff": outer_cutoff, "segments": rows, "activity_totals": totals}
            finally:
                connection.close()
    return result


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    rank_rows = result["source_precision"]["all_ten_rank_buckets"]
    rank_signal: dict[str, Any] = {}
    for source in ("historical", "soft_shallow"):
        rows = {row["source_rank_bucket"]: row for row in rank_rows if row["image_source"] == source}
        top = rows.get("rank_001_010", {})
        tail = rows.get("rank_051_100", {})
        top_rate = top.get("positive_density")
        tail_rate = tail.get("positive_density")
        rank_signal[source] = {
            "top10_positive_observations": top.get("positive_observations", 0),
            "top10_positive_density": top_rate,
            "rank51_100_positive_observations": tail.get("positive_observations", 0),
            "rank51_100_positive_density": tail_rate,
            "top10_to_tail_density_ratio": (
                top_rate / tail_rate if top_rate is not None and tail_rate not in (None, 0) else None
            ),
        }
    funnel_totals = {"image_truth": 0, "top12": 0, "top20": 0, "top50": 0, "top100": 0, "top200": 0}
    active_totals = dict(funnel_totals)
    for variants in result["ranking_funnel"].values():
        for payload in variants.values():
            for row in payload["activity_totals"]:
                target = active_totals if row["user_activity"] == "active" else None
                funnel_totals["image_truth"] += int(row["image_truth_candidate_observations"])
                for value in (12, 20, 50, 100, 200):
                    funnel_totals[f"top{value}"] += int(row[f"model_top{value}_positive_observations"])
                if target is not None:
                    target["image_truth"] += int(row["image_truth_candidate_observations"])
                    for value in (12, 20, 50, 100, 200):
                        target[f"top{value}"] += int(row[f"model_top{value}_positive_observations"])
    outer_support: dict[str, Any] = {}
    for window, variants in result["supervision"]["per_model_training_scope"].items():
        outer_support[window] = {}
        for variant, payload in variants.items():
            image_rows = [row for row in payload["outer_train"] if row["source_layer"] not in {"baseline_top100", "item2vec_only"}]
            outer_support[window][variant] = {
                "image_candidate_observations": sum(int(row["candidate_observations"]) for row in image_rows),
                "image_positive_observations": sum(int(row["positive_observations"]) for row in image_rows),
                "source_layers": image_rows,
            }
    return {
        "rank_signal": rank_signal,
        "ranking_funnel_all_window_variant_observations": funnel_totals,
        "ranking_funnel_active_only": active_totals,
        "outer_model_image_training_support": outer_support,
        "final_week": "not_run",
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# M3.9：图片来源正例密度与排序兑现漏斗审计",
        "",
        "## 术语与边界",
        "",
        "- `候选观察`：一条 `(cutoff, 用户, 商品)` 候选—标签行；跨 cutoff 重复出现会重复计数。",
        "- `图片独有候选`：相对 base-300 新增、由 historical 或 soft-shallow 图片源召回的候选；LightGBM 没有直接读取图片像素或完整 embedding。",
        "- `正例密度`：某集合内未来7天被购买的候选观察数除以该集合全部候选观察数。",
        "- `source membership`（来源成员关系，本项目统计单位）：一个候选若同时属于两条图片源，会在每条来源下各计一次；不用于计算去重候选总量。",
        "- 本阶段只读 M3.8 已有资产，不训练模型、不改候选、不运行最终周。",
        "",
        "## 源内名次与正例密度",
        "",
        "| 图片来源 | 源内名次 | 候选观察数 | 正例观察数 | 正例密度 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in sorted(result["source_precision"]["all_ten_rank_buckets"], key=lambda item: (item["image_source"], item["source_rank_bucket"])):
        lines.append(
            f"| {row['image_source']} | {row['source_rank_bucket']} | {row['candidate_observations']:,} | "
            f"{row['positive_observations']:,} | {row['positive_density']:.8f} |"
        )
    lines.extend(["", "## 外层模型实际图片正例监督", ""])
    lines.append("| window | variant | outer train cutoffs | 图片候选观察数 | 图片正例观察数 |")
    lines.append("|---|---|---|---:|---:|")
    for window, variants in summary["outer_model_image_training_support"].items():
        protocol = ROLLING_PROTOCOL[window]
        for variant, row in variants.items():
            lines.append(
                f"| {window} | {variant} | {','.join(protocol['outer_train'])} | "
                f"{row['image_candidate_observations']:,} | {row['image_positive_observations']:,} |"
            )
    lines.extend(["", "## 排序兑现总漏斗", ""])
    lines.append("本表把四窗口×三方案作为12次模型评测观察相加；同一正例在不同方案命中会重复计数。")
    lines.append("")
    lines.append("| 范围 | 图片 truth 候选观察 | Top12 | Top20 | Top50 | Top100 | Top200 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, row in (("全部", summary["ranking_funnel_all_window_variant_observations"]), ("active 用户", summary["ranking_funnel_active_only"])):
        lines.append(f"| {label} | {row['image_truth']} | {row['top12']} | {row['top20']} | {row['top50']} | {row['top100']} | {row['top200']} |")
    lines.extend(["", "## 初步信号摘要", ""])
    for source, row in summary["rank_signal"].items():
        ratio = row["top10_to_tail_density_ratio"]
        ratio_text = "不可计算" if ratio is None else f"{ratio:.3f}"
        lines.append(
            f"- {source}：Top10 正例 {row['top10_positive_observations']}，51--100名正例 "
            f"{row['rank51_100_positive_observations']}，Top10/尾部密度比 {ratio_text}。"
        )
    lines.extend([
        "",
        "完整的逐 cutoff、路径、active/inactive、warm/cold、分数分布和逐模型名次分位数见 `metrics.json`。",
        "本自动报告只陈述测量值；修召回、修排序或扩训练规模的最终解释写入人工复核后的 `M3_9_FINAL.md`。",
        "",
    ])
    return "\n".join(lines)


def run_audit(*, m38_metrics_path: Path, output_dir: Path) -> dict[str, Any]:
    validate_protocol()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    metrics = _read_json(m38_metrics_path)
    if metrics.get("schema_version") != "m3.8-image-source-factorial-ranking-v1":
        raise ValueError("M3.9 requires authoritative M3.8 metrics")
    if metrics.get("summary", {}).get("final_week") != "not_run":
        raise ValueError("M3.9 input must not contain final-week evaluation")
    transactions = Path(metrics["inputs"]["transactions"]["path"])
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M3.9",
        "status": "measured",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only audit of ten cutoff candidate-label assets and twelve outer ranking evaluations",
        "contract": {
            "candidate_observation_unit": "(cutoff, customer_id, article_id)",
            "detailed_variant": DETAILED_VARIANT,
            "model_training": "not_run",
            "candidate_rebuild": "not_run",
            "final_week": "not_run",
        },
        "inputs": {"m38_metrics": _identity(m38_metrics_path), "transactions": _identity(transactions)},
    }
    result["supervision"] = audit_supervision(metrics)
    result["source_precision"] = audit_source_precision(metrics, transactions)
    result["ranking_funnel"] = audit_ranking_funnel(metrics)
    result["summary"] = summarize(result)
    result["elapsed_seconds"] = time.perf_counter() - started
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "M3_9_AUDIT.md"
    result["artifacts"] = {"metrics": str(metrics_path.resolve()), "report": str(report_path.resolve())}
    _write_json(metrics_path, result)
    report_path.write_text(render_report(result), encoding="utf-8", newline="\n")
    return result
