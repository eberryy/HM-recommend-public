from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .audit import prepare_tabular_connection
from .m2 import (
    CATEGORICAL_FEATURES,
    FULL_FEATURES,
    RETRIEVAL_FEATURES,
    M2Config,
    _create_static_dimensions,
    _evaluate_ordering,
    _literal,
    _prepare_evaluation_truth,
    _prepare_frame,
    _reference_parity,
    _sha256,
    _write_json,
    build_category_maps,
    build_point_in_time_dataset,
    validate_candidate_artifact,
)


SCHEMA_VERSION = "m2.1-development-v1"
FEATURE_SETS = {"retrieval": RETRIEVAL_FEATURES, "full": FULL_FEATURES}
TRAIN_SCOPES = ("single", "pooled")
OBJECTIVES = ("binary", "lambdarank")


@dataclass(frozen=True)
class CandidateWindow:
    cutoff: str
    candidate_path: Path
    manifest_path: Path
    reference_metrics_path: Path | None = None


def _load_window_frame(path: Path, cutoff: str) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        selected = ", ".join(
            ["customer_id", "article_id", "candidate_rank"]
            + FULL_FEATURES
            + ["target"]
        )
        frame = connection.execute(
            f"SELECT {selected} FROM read_parquet({_literal(path)})"
        ).fetchdf()
    finally:
        connection.close()
    frame["target_cutoff"] = cutoff
    return frame


def prepare_ranking_frame(
    frame: pd.DataFrame, candidate_k: int
) -> tuple[pd.DataFrame, list[int], dict[str, int | float]]:
    required = {"target_cutoff", "customer_id", "article_id", "candidate_rank", "target"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ranking frame missing columns: {missing}")
    ordered = frame.sort_values(
        ["target_cutoff", "customer_id", "candidate_rank", "article_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    grouped = ordered.groupby(
        ["target_cutoff", "customer_id"], sort=False, observed=True
    )["target"]
    sizes = grouped.size()
    if sizes.empty or not bool((sizes == candidate_k).all()):
        invalid = sizes[sizes != candidate_k]
        raise ValueError(
            f"ranking groups must contain exactly {candidate_k} rows; "
            f"invalid_groups={len(invalid)}"
        )
    positives = grouped.sum()
    positive_groups = int((positives > 0).sum())
    groups = int(len(sizes))
    evidence: dict[str, int | float] = {
        "groups": groups,
        "positive_groups": positive_groups,
        "zero_positive_groups": groups - positive_groups,
        "positive_group_rate": positive_groups / groups,
        "rows": int(len(ordered)),
    }
    return ordered, sizes.astype(int).tolist(), evidence


def _train_variant(
    train_frame: pd.DataFrame,
    features: list[str],
    objective: str,
    name: str,
    artifact_dir: Path,
    config: M2Config,
    category_maps: dict[str, dict[int, int]],
) -> tuple[Any, dict[str, Any]]:
    if objective not in OBJECTIVES:
        raise ValueError(f"unsupported objective: {objective}")
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required; install dependency.txt") from error
    started = time.perf_counter()
    ordered, group_sizes, group_evidence = prepare_ranking_frame(
        train_frame, config.candidate_k
    )
    x_train = _prepare_frame(ordered, features, category_maps)
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    dataset_kwargs: dict[str, Any] = {}
    if objective == "lambdarank":
        dataset_kwargs["group"] = group_sizes
    dataset = lgb.Dataset(
        x_train,
        label=ordered["target"].astype(np.uint8),
        feature_name=features,
        categorical_feature=categorical,
        free_raw_data=False,
        **dataset_kwargs,
    )
    params: dict[str, Any] = {
        "objective": objective,
        "metric": (
            ["binary_logloss", "average_precision"]
            if objective == "binary"
            else ["map", "ndcg"]
        ),
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_data_in_leaf": config.min_data_in_leaf,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "seed": config.seed,
        "feature_fraction_seed": config.seed,
        "bagging_seed": config.seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": config.threads,
        "verbosity": -1,
    }
    if objective == "lambdarank":
        params.update({"eval_at": [config.metric_k], "lambdarank_truncation_level": 20})
    evaluations: dict[str, Any] = {}
    model = lgb.train(
        params,
        dataset,
        num_boost_round=config.num_boost_round,
        valid_sets=[dataset],
        valid_names=["train"],
        callbacks=[lgb.record_evaluation(evaluations)],
    )
    model_path = artifact_dir / f"lightgbm-{name}.txt"
    model.save_model(str(model_path))
    importance = sorted(
        (
            {"feature": feature, "gain": float(gain), "split": int(split)}
            for feature, gain, split in zip(
                features,
                model.feature_importance(importance_type="gain"),
                model.feature_importance(importance_type="split"),
                strict=True,
            )
        ),
        key=lambda row: row["gain"],
        reverse=True,
    )
    train_metrics = {
        metric: float(values[-1])
        for metric, values in evaluations.get("train", {}).items()
    }
    return model, {
        "name": name,
        "objective": objective,
        "features": features,
        "categorical_features": categorical,
        "category_cardinalities": {
            feature: len(category_maps[feature]) for feature in categorical
        },
        "parameters": params,
        "num_boost_round": config.num_boost_round,
        "group_evidence": group_evidence,
        "train_metrics": train_metrics,
        "top_feature_importance": importance[:25],
        "model_path": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "elapsed_seconds": time.perf_counter() - started,
        "binary_dataset_cache": "not_materialized_avoid_redundant_2x2x2_storage",
    }


def _score_variants(
    dataset_path: Path,
    variants: dict[str, tuple[Any, list[str], dict[str, dict[int, int]]]],
    evaluation_db: Path,
    prediction_path: Path,
    config: M2Config,
) -> dict[str, Any]:
    if evaluation_db.exists() or prediction_path.exists():
        raise FileExistsError(evaluation_db if evaluation_db.exists() else prediction_path)
    started = time.perf_counter()
    source = duckdb.connect()
    target = duckdb.connect(str(evaluation_db))
    temp_dir = evaluation_db.parent / "duckdb-temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    target.execute(f"SET threads = {config.threads}")
    target.execute("SET memory_limit = '11GB'")
    target.execute(f"SET temp_directory = {_literal(temp_dir)}")
    selected = ", ".join(
        ["customer_id", "article_id", "candidate_rank", "target", "user_history_events_12w"]
        + FULL_FEATURES
    )
    cursor = source.execute(
        f"SELECT {selected} FROM read_parquet({_literal(dataset_path)})"
    )
    rows = 0
    batches = 0
    initialized = False
    try:
        while True:
            batch = cursor.fetch_df_chunk(config.prediction_chunk_vectors)
            if batch.empty:
                break
            output = batch[
                [
                    "customer_id",
                    "article_id",
                    "candidate_rank",
                    "target",
                    "user_history_events_12w",
                ]
            ].copy()
            for name, (model, features, category_maps) in variants.items():
                output[f"score_{name}"] = model.predict(
                    _prepare_frame(batch, features, category_maps)
                )
            target.register("m21_prediction_batch", output)
            if not initialized:
                target.execute(
                    "CREATE TABLE predictions AS "
                    "SELECT * FROM m21_prediction_batch WHERE FALSE"
                )
                initialized = True
            target.execute("INSERT INTO predictions SELECT * FROM m21_prediction_batch")
            target.unregister("m21_prediction_batch")
            rows += len(output)
            batches += 1
        if not initialized:
            raise RuntimeError("validation dataset produced no prediction rows")
        duplicates = target.execute(
            "SELECT count(*) - count(DISTINCT (customer_id, article_id)) "
            "FROM predictions"
        ).fetchone()[0]
        if int(duplicates):
            raise RuntimeError("prediction rows are not unique by customer-item")
        target.execute(
            f"COPY predictions TO {_literal(prediction_path)} "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )
    finally:
        source.close()
        target.close()
    return {
        "rows": rows,
        "batches": batches,
        "score_columns": [f"score_{name}" for name in variants],
        "prediction_path": str(prediction_path.resolve()),
        "prediction_bytes": prediction_path.stat().st_size,
        "prediction_sha256": _sha256(prediction_path),
        "evaluation_db": str(evaluation_db.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }


def inactive_fallback_order_expression(score_column: str) -> str:
    if not score_column.replace("_", "").isalnum():
        raise ValueError("score column must be an identifier")
    return (
        "CASE WHEN user_history_events_12w = 0 "
        f"THEN -candidate_rank ELSE {score_column} END DESC"
    )


def _evaluate_variants(
    evaluation_db: Path,
    transactions_path: Path,
    cutoff: str,
    metric_k: int,
    variant_names: list[str],
) -> dict[str, Any]:
    connection = duckdb.connect(str(evaluation_db))
    try:
        connection.execute("SET threads = 8")
        truth = _prepare_evaluation_truth(connection, transactions_path, cutoff)
        orderings = {
            "rrf": _evaluate_ordering(
                connection, "rrf", "candidate_rank ASC", metric_k
            )
        }
        for name in variant_names:
            score = f"score_{name}"
            orderings[name] = _evaluate_ordering(
                connection, name, f"{score} DESC", metric_k
            )
            fallback_name = f"{name}__inactive_rrf"
            orderings[fallback_name] = _evaluate_ordering(
                connection,
                fallback_name,
                inactive_fallback_order_expression(score),
                metric_k,
            )
        return {"truth": truth, "orderings": orderings}
    finally:
        connection.close()


def _activity_map(ordering: dict[str, Any], metric_key: str) -> dict[str, float]:
    return {
        row["activity_segment"]: float(row[metric_key])
        for row in ordering["activity_segments"]
    }


def summarize_development(
    development: dict[str, Any], variant_names: list[str], metric_k: int
) -> dict[str, Any]:
    metric_key = f"map@{metric_k}"
    ordering_names = ["rrf"] + [
        name
        for variant in variant_names
        for name in (variant, f"{variant}__inactive_rrf")
    ]
    summaries: dict[str, Any] = {}
    for name in ordering_names:
        window_maps = {
            window_name: float(
                result["evaluation"]["orderings"][name]["segments"]["overall"][
                    metric_key
                ]
            )
            for window_name, result in development.items()
        }
        rrf_maps = {
            window_name: float(
                result["evaluation"]["orderings"]["rrf"]["segments"]["overall"][
                    metric_key
                ]
            )
            for window_name, result in development.items()
        }
        values = list(window_maps.values())
        summaries[name] = {
            "window_map@12": window_maps,
            "mean_map@12": float(np.mean(values)),
            "min_map@12": float(np.min(values)),
            "mean_minus_rrf_map@12": float(
                np.mean(
                    [
                        window_maps[window] - rrf_maps[window]
                        for window in window_maps
                    ]
                )
            ),
            "all_windows_above_rrf": all(
                window_maps[window] > rrf_maps[window] for window in window_maps
            ),
        }
    fallback_gates: dict[str, Any] = {}
    for variant in variant_names:
        fallback = f"{variant}__inactive_rrf"
        inactive_deltas: dict[str, float] = {}
        for window_name, result in development.items():
            pure_ordering = result["evaluation"]["orderings"][variant]
            fallback_ordering = result["evaluation"]["orderings"][fallback]
            pure_activity = _activity_map(pure_ordering, metric_key)
            fallback_activity = _activity_map(fallback_ordering, metric_key)
            inactive_deltas[window_name] = (
                fallback_activity.get("inactive_12w", 0.0)
                - pure_activity.get("inactive_12w", 0.0)
            )
        mean_delta = (
            summaries[fallback]["mean_map@12"] - summaries[variant]["mean_map@12"]
        )
        accepted = mean_delta >= 0 and all(
            delta > 0 for delta in inactive_deltas.values()
        )
        fallback_gates[variant] = {
            "accepted": accepted,
            "rule": (
                "mean overall MAP non-decreasing and inactive_12w MAP improves "
                "in every development window"
            ),
            "mean_overall_delta_map@12": mean_delta,
            "inactive_window_delta_map@12": inactive_deltas,
        }
    eligible = list(variant_names)
    eligible.extend(
        f"{variant}__inactive_rrf"
        for variant in variant_names
        if fallback_gates[variant]["accepted"]
    )
    selected = max(
        eligible,
        key=lambda name: (
            summaries[name]["mean_map@12"],
            summaries[name]["min_map@12"],
            name,
        ),
    )
    best_pure = max(
        variant_names,
        key=lambda name: (
            summaries[name]["mean_map@12"],
            summaries[name]["min_map@12"],
            name,
        ),
    )
    return {
        "orderings": summaries,
        "fallback_gates": fallback_gates,
        "best_pure_variant": best_pure,
        "selected_development_candidate": selected,
        "selection_rule": "highest mean MAP@12, then highest worst-window MAP@12",
        "final_confirmation_run": "not_run",
    }


def _render_report(result: dict[str, Any]) -> str:
    summary = result["development_summary"]
    window_names = list(result["development"])
    selected = summary["selected_development_candidate"]
    lines = [
        "# M2.1 pooled binary / LambdaRank development",
        "",
        "## 结论",
        "",
        f"- development-only 选择候选：{selected}；本阶段未再次运行 2020-09-16 最终周。",
        f"- best pure variant：{summary['best_pure_variant']}。",
        "- 固定候选池、Top100、特征定义、树参数和 200 rounds；只改变训练周数、objective 与 feature set。",
        "- inactive fallback 只对 12 周无历史用户恢复 RRF，不搜索额外阈值。",
        "",
        "## Overall MAP@12",
        "",
        "| ordering | "
        + " | ".join(window_names)
        + " | mean | worst | mean - RRF | all > RRF |",
        "|---|" + "---:|" * (len(window_names) + 4),
    ]
    ordered_names = ["rrf"] + sorted(
        name for name in summary["orderings"] if name != "rrf"
    )
    for name in ordered_names:
        row = summary["orderings"][name]
        values = [row["window_map@12"][window] for window in window_names]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{value:.6f}" for value in values)
            + f" | {row['mean_map@12']:.6f} | {row['min_map@12']:.6f} | "
            f"{row['mean_minus_rrf_map@12']:+.6f} | "
            f"{'yes' if row['all_windows_above_rrf'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Inactive fallback gate",
            "",
            "| base variant | accepted | mean overall delta | inactive deltas by window |",
            "|---|---|---:|---|",
        ]
    )
    for name, gate in sorted(summary["fallback_gates"].items()):
        deltas = ", ".join(
            f"{window}={delta:+.6f}"
            for window, delta in gate["inactive_window_delta_map@12"].items()
        )
        lines.append(
            f"| {name} | {'yes' if gate['accepted'] else 'no'} | "
            f"{gate['mean_overall_delta_map@12']:+.6f} | {deltas} |"
        )
    lines.extend(["", "## Window contract", ""])
    for name, window in result["development"].items():
        lines.append(
            f"- {name}：single={window['single_train_cutoff']}；"
            f"pooled={' + '.join(window['pooled_train_cutoffs'])}；"
            f"validation={window['validation_cutoff']}；"
            f"parity={window['m1_parity']['status']}。"
        )
    lines.extend(
        [
            "",
            "## 重要证据边界",
            "",
            "- 这是 ranker 的早期 development comparison，不是新的 final score。",
            "- retrieval-v1 collaborative profile 曾由 2020-07-22 与 2020-08-19 选择；"
            "因此这些窗口用于冻结候选池后的 ranker robustness / selection，"
            "不是对整条 retrieval + ranking pipeline 的 nested untouched validation。",
            "- optimistic_all_articles 在越早窗口对真实可售目录的假设越强；"
            "行为特征仍严格只读取各 cutoff 之前 12 周。",
            "- LambdaRank group 是 (target_cutoff, customer_id)；"
            "无候选内正例 group 已单独记录，不能视为有效 pairwise ranking signal。",
            "- 没有曝光日志，target=0 仍是未观察样本，不是真实负反馈。",
            "- 图片/文本 embedding 未启动，M3 未启动。",
            "",
            "## 复现产物",
            "",
            f"- metrics：{result['artifacts']['metrics']}",
            f"- private manifest：{result['artifacts']['manifest']}",
            f"- models/features/predictions：{result['artifacts']['artifact_dir']}（Git ignored）",
            "",
        ]
    )
    return "\n".join(lines)


def _save_category_maps(
    path: Path, maps: dict[str, dict[int, int]]
) -> dict[str, Any]:
    _write_json(
        path,
        {
            feature: {str(raw): encoded for raw, encoded in mapping.items()}
            for feature, mapping in maps.items()
        },
    )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "cardinalities": {
            feature: len(mapping) for feature, mapping in maps.items()
        },
    }


def run_m21(
    raw_dir: Path,
    work_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    windows: list[CandidateWindow],
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    config.validate()
    if config.evaluation_role != "development":
        raise ValueError("M2.1 is development-only")
    for directory in (output_dir, artifact_dir):
        if directory.exists():
            raise FileExistsError(directory)
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    by_cutoff = {window.cutoff: window for window in windows}
    required_cutoffs = ("2020-05-27", "2020-06-24", "2020-07-22", "2020-08-19")
    if tuple(sorted(by_cutoff)) != required_cutoffs:
        raise ValueError(f"M2.1 requires candidate windows {required_cutoffs}")
    identities: dict[str, dict[str, Any]] = {}
    for cutoff in required_cutoffs:
        window = by_cutoff[cutoff]
        identity = validate_candidate_artifact(
            window.candidate_path, window.manifest_path, config.candidate_k
        )
        if identity["cutoff"] != cutoff:
            raise ValueError(f"candidate cutoff mismatch for {cutoff}")
        if identity["sample_rate"] != 0.1:
            raise ValueError("M2.1 development windows must use hash 10% users")
        identities[cutoff] = identity
    datasets: dict[str, dict[str, Any]] = {}
    dataset_paths: dict[str, Path] = {}
    connection = prepare_tabular_connection(raw_dir, work_dir)
    try:
        _create_static_dimensions(connection)
        for cutoff in required_cutoffs:
            path = artifact_dir / f"features-{cutoff}.parquet"
            dataset_paths[cutoff] = path
            datasets[cutoff] = build_point_in_time_dataset(
                connection,
                by_cutoff[cutoff].candidate_path,
                identities[cutoff],
                path,
                config,
            )
    finally:
        connection.close()
    protocol = {
        "dev_a": {
            "single": ["2020-06-24"],
            "pooled": ["2020-05-27", "2020-06-24"],
            "validation": "2020-07-22",
        },
        "dev_b": {
            "single": ["2020-07-22"],
            "pooled": ["2020-06-24", "2020-07-22"],
            "validation": "2020-08-19",
        },
    }
    frame_cache = {
        cutoff: _load_window_frame(dataset_paths[cutoff], cutoff)
        for cutoff in required_cutoffs
    }
    development: dict[str, Any] = {}
    variant_names = [
        f"{scope}_{objective}_{feature_set}"
        for scope in TRAIN_SCOPES
        for objective in OBJECTIVES
        for feature_set in FEATURE_SETS
    ]
    for dev_name, dev_protocol in protocol.items():
        dev_dir = artifact_dir / dev_name
        dev_dir.mkdir()
        variants: dict[
            str, tuple[Any, list[str], dict[str, dict[int, int]]]
        ] = {}
        model_evidence: dict[str, Any] = {}
        encoding_evidence: dict[str, Any] = {}
        for scope in TRAIN_SCOPES:
            cutoffs = dev_protocol[scope]
            train_frame = pd.concat(
                [frame_cache[cutoff] for cutoff in cutoffs], ignore_index=True
            )
            category_maps = build_category_maps(train_frame)
            encoding_evidence[scope] = _save_category_maps(
                dev_dir / f"category-maps-{scope}.json", category_maps
            )
            for objective in OBJECTIVES:
                for feature_set, features in FEATURE_SETS.items():
                    name = f"{scope}_{objective}_{feature_set}"
                    model, evidence = _train_variant(
                        train_frame,
                        features,
                        objective,
                        name,
                        dev_dir,
                        config,
                        category_maps,
                    )
                    variants[name] = (model, features, category_maps)
                    model_evidence[name] = evidence
            del train_frame
        valid_cutoff = dev_protocol["validation"]
        scoring = _score_variants(
            dataset_paths[valid_cutoff],
            variants,
            dev_dir / "evaluation.duckdb",
            dev_dir / "validation-predictions.parquet",
            config,
        )
        del variants
        evaluation = _evaluate_variants(
            dev_dir / "evaluation.duckdb",
            work_dir / "transactions.parquet",
            valid_cutoff,
            config.metric_k,
            variant_names,
        )
        reference = by_cutoff[valid_cutoff].reference_metrics_path
        if reference is None:
            raise ValueError(f"missing M1 reference metrics for {valid_cutoff}")
        parity = _reference_parity(evaluation, reference)
        development[dev_name] = {
            "single_train_cutoff": dev_protocol["single"][0],
            "pooled_train_cutoffs": dev_protocol["pooled"],
            "validation_cutoff": valid_cutoff,
            "models": model_evidence,
            "category_encoding": encoding_evidence,
            "scoring": scoring,
            "evaluation": evaluation,
            "m1_parity": parity,
        }
    del frame_cache
    summary = summarize_development(development, variant_names, config.metric_k)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "candidate_inputs": identities,
        "datasets": datasets,
        "protocol": protocol,
        "group_key": ["target_cutoff", "customer_id"],
        "target_cutoff_feature": False,
        "negative_sampling": "none_keep_all_candidates",
        "fallback_rule": "user_history_events_12w = 0 uses RRF; otherwise model",
        "retrieval_selection_boundary": (
            "fixed collaborative retrieval-v1 was selected on 2020-07-22 and "
            "2020-08-19; M2.1 is ranker development, not nested pipeline validation"
        ),
    }
    manifest_path = artifact_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M2.1",
        "status": "measured",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "development_only_no_final_week",
        "config": asdict(config),
        "contract": {
            "label": "1 iff distinct user-item appears in target week; else 0",
            "group_key": ["target_cutoff", "customer_id"],
            "catalog_protocol": "optimistic_all_articles",
            "customer_snapshot_policy": "age_only",
            "negative_sampling": "none_keep_all_candidates",
            "lambda_truncation_level": 20,
        },
        "candidate_inputs": identities,
        "datasets": datasets,
        "development": development,
        "development_summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "M2_1_REPORT.md"
    result["artifacts"] = {
        "metrics": str(metrics_path.resolve()),
        "report": str(report_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
    }
    _write_json(metrics_path, result)
    report_path.write_text(_render_report(result), encoding="utf-8")
    return result
