from __future__ import annotations

import gc
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m2 import (
    M2Config,
    RETRIEVAL_FEATURES,
    _evaluate_ordering,
    _literal,
    _prepare_evaluation_truth,
    _prepare_frame,
    _sha256,
    _write_json,
    build_category_maps,
)
from .m21 import inactive_fallback_order_expression
from .m22 import FEATURE_GROUPS as LEGACY_FEATURE_GROUPS
from .m26 import _evaluate_models, _save_category_maps
from .m29 import ITEM2VEC_FEATURES, _attach_popularity_segments, _read_json
from .m210 import _file_identity, _target_feature_paths, build_target_aware_cache, score_models
from .m211 import SAMPLING_SEED, _load_category_maps, load_distribution_sample, train_lambdarank
from .m212 import COUNT_FEATURES, RECENCY_FEATURES, SHARE_FEATURES, feature_sets, load_inner_validation, train_inner_ranker
from .m3 import ALL_CUTOFFS, ANCHOR_NAME, FALLBACK_NAME, ROLLING_PROTOCOL, _peak_working_set_bytes


SCHEMA_VERSION = "m3.1-bounded-source-feature-ablation-v1"
M1_SOURCES = (
    "repurchase",
    "recent_popularity",
    "product_family",
    "user_day_covisit",
    "age_popularity",
    "attribute_content",
)
SOURCE_NAMES = (*M1_SOURCES, "item2vec")


def feature_groups() -> dict[str, list[str]]:
    groups = {
        "retrieval_all": list(RETRIEVAL_FEATURES),
        **{name: list(values) for name, values in LEGACY_FEATURE_GROUPS.items()},
        "item2vec": list(ITEM2VEC_FEATURES),
        "target_count": list(COUNT_FEATURES),
        "target_recency": list(RECENCY_FEATURES),
        "target_share": list(SHARE_FEATURES),
    }
    validate_feature_groups(groups)
    return groups


def validate_feature_groups(groups: dict[str, list[str]]) -> None:
    anchor = feature_sets()[ANCHOR_NAME]
    flattened = [feature for values in groups.values() for feature in values]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("M3.1 feature groups overlap")
    if set(flattened) != set(anchor) or len(flattened) != len(anchor):
        raise RuntimeError("M3.1 feature groups must partition the 84-feature anchor")
    if len(groups) != 10:
        raise RuntimeError("M3.1 requires exactly ten bounded feature groups")


def ablation_feature_sets() -> dict[str, list[str]]:
    anchor = feature_sets()[ANCHOR_NAME]
    return {
        f"without_{name}": [feature for feature in anchor if feature not in removed]
        for name, removed in feature_groups().items()
    }


def apply_source_outage(frame: pd.DataFrame, source: str) -> tuple[pd.DataFrame, np.ndarray]:
    if source not in SOURCE_NAMES:
        raise ValueError(f"unknown source: {source}")
    modified = frame.copy()
    if source == "item2vec":
        keep = modified["item2vec_is_new"].fillna(0).to_numpy() == 0
        for feature in ITEM2VEC_FEATURES:
            if feature in {"item2vec_present", "item2vec_is_new", "item2vec_vocab_count"}:
                modified[feature] = 0
            else:
                modified[feature] = np.nan
        return modified, keep

    present = modified[f"{source}_present"].fillna(0).to_numpy(dtype=np.int64)
    source_count = modified["source_count"].fillna(0).to_numpy(dtype=np.int64)
    item2vec_present = modified["item2vec_present"].fillna(0).to_numpy(dtype=np.int64)
    keep = ~((present > 0) & (source_count <= 1) & (item2vec_present == 0))
    contribution = modified[f"{source}_rrf_contribution"].fillna(0.0)
    modified["fused_score"] = modified["fused_score"].fillna(0.0) - contribution
    modified["source_count"] = np.maximum(source_count - present, 0)
    modified[f"{source}_present"] = 0
    for suffix in ("rank", "score", "rrf_contribution"):
        modified[f"{source}_{suffix}"] = np.nan
    return modified, keep


def _label_expanded(ordering: dict[str, Any]) -> None:
    for segment in ordering["segments"].values():
        segment["candidate_recall@expanded_pool"] = segment["candidate_recall@100"]
        segment["candidate_hit_rate@expanded_pool"] = segment["candidate_hit_rate@100"]
    for segment in ordering["activity_segments"]:
        segment["candidate_recall@expanded_pool"] = segment["candidate_recall@100"]


def _source_diagnostics(dataset_path: Path, source: str) -> dict[str, int]:
    connection = duckdb.connect()
    try:
        relation = f"read_parquet({_literal(dataset_path)})"
        if source == "item2vec":
            row = connection.execute(
                f"""
                SELECT count(*) FILTER(WHERE item2vec_present=1),
                       count(*) FILTER(WHERE item2vec_is_new=1),
                       count(*) FILTER(WHERE item2vec_is_new=1 AND target=1),
                       count(*) FILTER(WHERE item2vec_present=1 AND item2vec_is_new=0)
                FROM {relation}
                """
            ).fetchone()
        else:
            present = f"{source}_present"
            row = connection.execute(
                f"""
                SELECT count(*) FILTER(WHERE {present}=1),
                       count(*) FILTER(
                           WHERE {present}=1 AND source_count=1 AND item2vec_present=0),
                       count(*) FILTER(
                           WHERE {present}=1 AND source_count=1 AND item2vec_present=0
                             AND target=1),
                       count(*) FILTER(
                           WHERE {present}=1 AND (source_count>1 OR item2vec_present=1))
                FROM {relation}
                """
            ).fetchone()
    finally:
        connection.close()
    return {
        "source_supported_rows": int(row[0]),
        "unique_supported_rows_removed": int(row[1]),
        "positive_rows_removed": int(row[2]),
        "overlap_supported_rows_retained": int(row[3]),
    }


def score_source_outages(
    *,
    dataset_path: Path,
    transactions_path: Path,
    cutoff: str,
    model_path: Path,
    category_maps_path: Path,
    output_db: Path,
    config: M2Config,
) -> dict[str, Any]:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required; install dependency.txt") from error
    if output_db.exists():
        raise FileExistsError(output_db)
    started = time.perf_counter()
    model = lgb.Booster(model_file=str(model_path))
    maps = _load_category_maps(category_maps_path)
    features = feature_sets()[ANCHOR_NAME]
    identity = ["customer_id", "article_id", "candidate_rank", "target", "user_history_events_12w"]
    selected = list(dict.fromkeys([*identity, *features]))
    source = duckdb.connect()
    target = duckdb.connect(str(output_db))
    temp_dir = output_db.parent / "duckdb-temp-source-outage"
    temp_dir.mkdir()
    target.execute(f"SET threads={config.threads}")
    target.execute("SET memory_limit='11GB'")
    target.execute(f"SET temp_directory={_literal(temp_dir)}")
    cursor = source.execute(
        f"SELECT {','.join(selected)} FROM read_parquet({_literal(dataset_path)})"
    )
    rows = 0
    batches = 0
    initialized = False
    try:
        while True:
            batch = cursor.fetch_df_chunk(config.prediction_chunk_vectors)
            if batch.empty:
                break
            output = batch[identity].copy()
            for route in SOURCE_NAMES:
                modified, keep = apply_source_outage(batch, route)
                output[f"keep_{route}"] = keep.astype(np.uint8)
                output[f"score_drop_{route}"] = model.predict(
                    _prepare_frame(modified, features, maps)
                )
                del modified
            target.register("m31_source_batch", output)
            if not initialized:
                target.execute(
                    "CREATE TABLE source_predictions AS "
                    "SELECT * FROM m31_source_batch WHERE FALSE"
                )
                initialized = True
            target.execute("INSERT INTO source_predictions SELECT * FROM m31_source_batch")
            target.unregister("m31_source_batch")
            rows += len(output)
            batches += 1
        if not initialized:
            raise RuntimeError("M3.1 source outage dataset produced no rows")
        duplicates = int(
            target.execute(
                "SELECT count(*)-count(DISTINCT (customer_id,article_id)) FROM source_predictions"
            ).fetchone()[0]
        )
        if duplicates:
            raise RuntimeError("M3.1 source outage rows are not unique")
        evaluations: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        for route in SOURCE_NAMES:
            score = f"score_drop_{route}"
            target.execute(
                f"""
                CREATE OR REPLACE TEMP VIEW predictions AS
                SELECT customer_id,article_id,candidate_rank,target,
                       user_history_events_12w,{score}
                FROM source_predictions
                WHERE keep_{route}=1
                """
            )
            truth = _prepare_evaluation_truth(target, transactions_path, cutoff)
            ordering = _evaluate_ordering(
                target,
                f"drop_{route}__inactive_rrf",
                inactive_fallback_order_expression(score),
                config.metric_k,
            )
            _label_expanded(ordering)
            evaluations[route] = {"truth": truth, "ordering": ordering}
            diagnostics[route] = {
                **_source_diagnostics(dataset_path, route),
                "retained_rows": int(
                    target.execute(f"SELECT count(*) FROM source_predictions WHERE keep_{route}=1").fetchone()[0]
                ),
                "retained_users": int(
                    target.execute(f"SELECT count(DISTINCT customer_id) FROM source_predictions WHERE keep_{route}=1").fetchone()[0]
                ),
                "minimum_group_rows": int(
                    target.execute(
                        f"SELECT min(n) FROM (SELECT customer_id,count(*) n FROM source_predictions WHERE keep_{route}=1 GROUP BY customer_id)"
                    ).fetchone()[0]
                ),
                "maximum_group_rows": int(
                    target.execute(
                        f"SELECT max(n) FROM (SELECT customer_id,count(*) n FROM source_predictions WHERE keep_{route}=1 GROUP BY customer_id)"
                    ).fetchone()[0]
                ),
            }
    finally:
        source.close()
        target.close()
    return {
        "rows": rows,
        "batches": batches,
        "model": _file_identity(model_path),
        "category_maps": _file_identity(category_maps_path),
        "dataset": _file_identity(dataset_path),
        "evaluation_db": _file_identity(output_db),
        "diagnostics": diagnostics,
        "evaluations": evaluations,
        "elapsed_seconds": time.perf_counter() - started,
        "contract": {
            "mode": "fixed_pool_route_outage_no_refill_no_refusion_no_retrain",
            "unique_pair_removal": "remove only rows unsupported by any remaining M1 route or Item2Vec",
            "overlap_rows": "retain pair, clear failed-route features, keep frozen candidate_rank",
            "score": "frozen M3.0 anchor rescored after route-feature removal",
        },
    }


def _classify(deltas: dict[str, float]) -> str:
    if all(value > 0 for value in deltas.values()):
        return "removal_improves_all_three"
    if all(value < 0 for value in deltas.values()):
        return "removal_hurts_all_three"
    return "cross_window_unstable_or_tied"


def summarize_m31(
    feature_development: dict[str, Any],
    source_development: dict[str, Any],
    m3_metrics: dict[str, Any],
    metric_k: int,
) -> dict[str, Any]:
    metric = f"map@{metric_k}"
    anchor_values = m3_metrics["summary"]["orderings"][FALLBACK_NAME]["window_map@12"]
    anchor_ceiling = m3_metrics["summary"]["candidate_ceiling"]
    feature_summary: dict[str, Any] = {}
    for group in feature_groups():
        variant = f"without_{group}__inactive_rrf"
        values = {
            window: float(row["evaluation"]["orderings"][variant]["segments"]["overall"][metric])
            for window, row in feature_development.items()
        }
        deltas = {window: values[window] - float(anchor_values[window]) for window in values}
        feature_summary[group] = {
            "removed_features": feature_groups()[group],
            "window_map@12": values,
            "window_delta_vs_anchor_map@12": deltas,
            "mean_map@12": float(np.mean(list(values.values()))),
            "mean_delta_vs_anchor_map@12": float(np.mean(list(deltas.values()))),
            "classification": _classify(deltas),
        }

    source_summary: dict[str, Any] = {}
    for route in SOURCE_NAMES:
        rows = {}
        map_delta = {}
        recall_delta = {}
        oracle_delta = {}
        for window, result in source_development.items():
            ordering = result["evaluations"][route]["ordering"]["segments"]["overall"]
            rows[window] = {
                "map@12": float(ordering[metric]),
                "candidate_recall@expanded_pool": float(ordering["candidate_recall@expanded_pool"]),
                "candidate_hit_rate@expanded_pool": float(ordering["candidate_hit_rate@expanded_pool"]),
                "oracle_map@12": float(ordering[f"oracle_map@{metric_k}"]),
                **result["diagnostics"][route],
            }
            map_delta[window] = rows[window]["map@12"] - float(anchor_values[window])
            recall_delta[window] = rows[window]["candidate_recall@expanded_pool"] - float(
                anchor_ceiling[window]["candidate_recall@expanded_pool"]
            )
            oracle_delta[window] = rows[window]["oracle_map@12"] - float(
                anchor_ceiling[window]["oracle_map@12"]
            )
        source_summary[route] = {
            "windows": rows,
            "window_delta_vs_anchor_map@12": map_delta,
            "window_delta_candidate_recall": recall_delta,
            "window_delta_oracle_map@12": oracle_delta,
            "mean_delta_vs_anchor_map@12": float(np.mean(list(map_delta.values()))),
            "mean_delta_candidate_recall": float(np.mean(list(recall_delta.values()))),
            "mean_delta_oracle_map@12": float(np.mean(list(oracle_delta.values()))),
            "classification": _classify(map_delta),
        }
    return {
        "anchor": {
            "window_map@12": anchor_values,
            "mean_map@12": float(np.mean(list(anchor_values.values()))),
        },
        "feature_ablation": feature_summary,
        "source_outage": source_summary,
        "selection": "diagnostic_only_no_post_hoc_combination",
        "final_week": "not_run",
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    windows = list(ROLLING_PROTOCOL)
    lines = [
        "# M3.1：三窗口有界 source / feature 消融",
        "",
        "## 协议边界",
        "",
        "- feature：10 个互斥组完整覆盖 anchor 84 维；逐次仅删一组、独立 inner early stopping，不组合删除。",
        "- source：7 路固定池 outage sensitivity；删除仅由失效路支持的 pair、清空该路特征、冻结模型重打分。",
        "- source outage 不补位、不重新 RRF、不重训，因此是服务失效敏感性，不是新架构的完整 refusion ablation。",
        "- 10% hash development users；optimistic all-articles；final week 未运行。",
        "",
        "## Feature leave-one-group-out",
        "",
        "| removed group | " + " | ".join(windows) + " | mean delta | classification |",
        "|---|" + "---:|" * (len(windows) + 1) + "---|",
    ]
    for group, row in summary["feature_ablation"].items():
        lines.append(
            "| " + group + " | "
            + " | ".join(f"{row['window_delta_vs_anchor_map@12'][window]:+.6f}" for window in windows)
            + f" | {row['mean_delta_vs_anchor_map@12']:+.6f} | {row['classification']} |"
        )
    lines.extend([
        "",
        "## Source outage sensitivity",
        "",
        "| failed route | mean Recall delta | mean Oracle delta | mean MAP delta | classification |",
        "|---|---:|---:|---:|---|",
    ])
    for route, row in summary["source_outage"].items():
        lines.append(
            f"| {route} | {row['mean_delta_candidate_recall']:+.6f} | "
            f"{row['mean_delta_oracle_map@12']:+.6f} | {row['mean_delta_vs_anchor_map@12']:+.6f} | "
            f"{row['classification']} |"
        )
    lines.extend([
        "",
        "## 资源与证据边界",
        "",
        f"- wall time：{result['elapsed_seconds']:.2f} 秒。",
        "- peak working set：" + (
            f"{result['resources']['peak_working_set_bytes']/1024**3:.2f} GiB。"
            if result["resources"]["peak_working_set_bytes"] else "未能读取。"
        ),
        f"- 主产物：{result['resources']['artifact_count']} 个，"
        f"{result['resources']['artifact_bytes']/1024**2:.2f} MiB，均绑定 path/bytes/SHA256。",
        "- 任何单项改善都只作诊断，不根据本轮三窗继续组合删除或修改路由。",
        "- final week 未运行。",
        "",
        "机器可读证据：`metrics.json`。",
        "",
    ])
    return "\n".join(lines)


def _collect_artifacts(development: dict[str, Any], source_development: dict[str, Any]) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for row in development.values():
        for model in row["inner_models"].values():
            identities.append(_file_identity(Path(model["model_path"])))
        for model in row["outer_models"].values():
            identities.append(_file_identity(Path(model["model_path"])))
        identities.append(_file_identity(Path(row["scoring"]["prediction_path"])))
        identities.append(_file_identity(Path(row["scoring"]["evaluation_db"])))
        identities.append(_file_identity(Path(row["inner_category_encoding"]["path"])))
        identities.append(_file_identity(Path(row["outer_category_encoding"]["path"])))
    for row in source_development.values():
        identities.append(row["evaluation_db"])
    return identities


def run_m31(
    *,
    raw_dir: Path,
    transactions_path: Path,
    m3_metrics_path: Path,
    m29_cache_dir: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    config.validate()
    groups = feature_groups()
    variants = ablation_feature_sets()
    if config.evaluation_role != "development" or config.candidate_k != 300:
        raise ValueError("M3.1 requires development-only candidate_k=300")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    m3_metrics = _read_json(m3_metrics_path.resolve())
    if m3_metrics.get("schema_version") != "m3.0-three-window-robustness-v1" or m3_metrics.get("status") != "measured":
        raise ValueError("M3.1 requires measured M3.0 evidence")
    if m3_metrics["contract"].get("final_week") != "not_run":
        raise ValueError("M3.0 final-week boundary is not intact")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        cache = build_target_aware_cache(
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            m29_cache_dir=m29_cache_dir,
            cache_dir=feature_cache_dir,
            cutoffs=ALL_CUTOFFS,
        )
        anchor_features = feature_sets()[ANCHOR_NAME]
        feature_development: dict[str, Any] = {}
        source_development: dict[str, Any] = {}
        for window, protocol in ROLLING_PROTOCOL.items():
            print(f"M3.1 {window}: feature inner fits", flush=True)
            window_dir = artifact_dir / window
            window_dir.mkdir()
            inner_train, inner_sizes, inner_sampling = load_distribution_sample(
                [_target_feature_paths(feature_cache_dir, cutoff)[0] for cutoff in protocol["inner_train"]],
                anchor_features,
                seed=SAMPLING_SEED,
            )
            inner_maps = build_category_maps(inner_train)
            inner_map_evidence = _save_category_maps(window_dir / "inner-category-maps.json", inner_maps)
            inner_validation, inner_validation_sizes, truth_counts, validation_evidence = load_inner_validation(
                dataset_path=_target_feature_paths(feature_cache_dir, protocol["inner_validation"])[0],
                transactions_path=transactions_path,
                cutoff=protocol["inner_validation"],
                features=anchor_features,
            )
            inner_models: dict[str, Any] = {}
            for name, features in variants.items():
                _, evidence = train_inner_ranker(
                    train_frame=inner_train,
                    train_group_sizes=inner_sizes,
                    train_evidence=inner_sampling,
                    validation_frame=inner_validation,
                    validation_group_sizes=inner_validation_sizes,
                    validation_truth_counts=truth_counts,
                    validation_evidence=validation_evidence,
                    features=features,
                    name=name,
                    artifact_dir=window_dir,
                    config=config,
                    category_maps=inner_maps,
                )
                inner_models[name] = evidence
            del inner_train, inner_sizes, inner_validation, inner_validation_sizes
            gc.collect()

            print(f"M3.1 {window}: feature outer refits and scoring", flush=True)
            outer_train, outer_sizes, outer_sampling = load_distribution_sample(
                [_target_feature_paths(feature_cache_dir, cutoff)[0] for cutoff in protocol["outer_train"]],
                anchor_features,
                seed=SAMPLING_SEED,
            )
            outer_maps = build_category_maps(outer_train)
            outer_map_evidence = _save_category_maps(window_dir / "outer-category-maps.json", outer_maps)
            models: dict[str, tuple[Any, list[str], dict[str, dict[int, int]]]] = {}
            outer_models: dict[str, Any] = {}
            for name, features in variants.items():
                rounds = int(inner_models[name]["best_iteration"])
                model, evidence = train_lambdarank(
                    frame=outer_train,
                    group_sizes=outer_sizes,
                    group_evidence=outer_sampling,
                    features=features,
                    name=name,
                    use_ipw=False,
                    artifact_dir=window_dir,
                    config=replace(config, num_boost_round=rounds),
                    category_maps=outer_maps,
                )
                evidence["selected_rounds_from_inner"] = rounds
                evidence["inner_selection_score"] = inner_models[name]["best_score"]
                models[name] = (model, features, outer_maps)
                outer_models[name] = evidence
            del outer_train, outer_sizes
            gc.collect()
            scoring = score_models(
                dataset_path=_target_feature_paths(feature_cache_dir, protocol["outer_validation"])[0],
                models=models,
                evaluation_db=window_dir / "feature-evaluation.duckdb",
                prediction_path=window_dir / "feature-predictions.parquet",
                config=config,
            )
            del models
            gc.collect()
            evaluation = _evaluate_models(
                evaluation_db=window_dir / "feature-evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                metric_k=config.metric_k,
                variant_names=list(variants),
            )
            evaluation = _attach_popularity_segments(
                evaluation=evaluation,
                evaluation_db=window_dir / "feature-evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                variant_names=list(variants),
                budget=config.candidate_k,
            )
            feature_development[window] = {
                "protocol": protocol,
                "inner_models": inner_models,
                "outer_models": outer_models,
                "inner_category_encoding": inner_map_evidence,
                "outer_category_encoding": outer_map_evidence,
                "outer_sampling": outer_sampling,
                "scoring": scoring,
                "evaluation": evaluation,
            }

            print(f"M3.1 {window}: seven frozen-model source outages", flush=True)
            m3_window = m3_metrics["development"][window]
            source_dir = window_dir / "source-outage"
            source_dir.mkdir()
            source_development[window] = score_source_outages(
                dataset_path=_target_feature_paths(feature_cache_dir, protocol["outer_validation"])[0],
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                model_path=Path(m3_window["outer_model"]["model_path"]),
                category_maps_path=Path(m3_window["outer_category_encoding"]["path"]),
                output_db=source_dir / "source-outage-evaluation.duckdb",
                config=config,
            )

        summary = summarize_m31(feature_development, source_development, m3_metrics, config.metric_k)
        artifacts = _collect_artifacts(feature_development, source_development)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M3.1",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "three_rolling_development_windows_no_final_week",
            "config": asdict(config),
            "contract": {
                "candidate_pool": "frozen M3.0 six-source Top100 plus up to 200 Item2Vec-only",
                "catalog_protocol": "optimistic_all_articles",
                "feature_groups": groups,
                "feature_ablation": "leave exactly one of ten disjoint groups out; independent inner early stopping",
                "source_routes": SOURCE_NAMES,
                "source_outage": "fixed pool, unique-pair removal, no refill, no refusion, frozen anchor rescore, no retrain",
                "selection": "diagnostic only; no combined deletion or routing promotion",
                "rolling_protocol": ROLLING_PROTOCOL,
                "final_week": "not_run",
            },
            "inputs": {
                "m3_metrics": _file_identity(m3_metrics_path),
                "transactions": _file_identity(transactions_path),
            },
            "feature_cache": cache,
            "feature_development": feature_development,
            "source_development": source_development,
            "summary": summary,
            "resources": {
                "peak_working_set_bytes": _peak_working_set_bytes(),
                "artifact_count": len(artifacts),
                "artifact_bytes": int(sum(item["bytes"] for item in artifacts)),
                "artifacts": artifacts,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "artifacts": {},
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M3_1_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path.resolve()),
            "report": str(report_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
        }
        _write_json(metrics_path, result)
        report_path.write_text(render_report(result), encoding="utf-8")
        return result
    except Exception as error:
        _write_json(
            output_dir / f"failure-{time.time_ns()}.json",
            {
                "schema_version": "m3.1-failure-v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
