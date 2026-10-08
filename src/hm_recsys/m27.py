"""M2.7 two-stage image pruning and hard-negative diagnostics."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .m2 import (
    CATEGORICAL_FEATURES,
    FULL_FEATURES,
    M2Config,
    TABULAR_FEATURES,
    _literal,
    _prepare_frame,
    _sha256,
    _write_json,
    build_category_maps,
)
from .m26 import (
    IMAGE_FEATURES,
    PROTOCOL,
    REQUIRED_CUTOFFS,
    _evaluate_models,
    _save_category_maps,
    _score_models,
    _train_lambdarank,
)


SCHEMA_VERSION = "m2.7-two-stage-development-v2"
ALL_FEATURES = list(dict.fromkeys(FULL_FEATURES + IMAGE_FEATURES))
STAGE1_FEATURES = list(dict.fromkeys(TABULAR_FEATURES + IMAGE_FEATURES))
VARIANTS = ("two_stage_200", "hardneg_100_full_eval")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_m26_reuse(
    metrics_path: Path, cache_dir: Path
) -> tuple[dict[str, Any], dict[str, Path]]:
    metrics_path = metrics_path.resolve()
    metrics = _read_json(metrics_path)
    if (
        metrics.get("schema_version") != "m2.6-image-ranker-development-v1"
        or metrics.get("status") != "measured"
        or metrics.get("development_summary", {}).get("final_week") != "not_run"
    ):
        raise ValueError("M2.7 requires measured development-only M2.6 metrics")
    if metrics.get("run_id") != "m2-6-v4-cross-seed":
        raise ValueError("M2.7 requires authoritative M2.6 cross-seed run")
    if Path(metrics["artifacts"]["cache_dir"]).resolve() != cache_dir.resolve():
        raise ValueError("M2.6 cache directory differs from metrics")
    paths: dict[str, Path] = {}
    for cutoff in REQUIRED_CUTOFFS:
        candidate = metrics["candidate_cache"][cutoff]
        feature = metrics["feature_cache"][cutoff]
        if candidate.get("schema_version") != "m2.6-expanded-cache-v2":
            raise ValueError(f"M2.6 candidate cache schema mismatch: {cutoff}")
        path = Path(feature["dataset_path"]).resolve()
        if (
            not path.is_file()
            or path.stat().st_size != int(feature["dataset_bytes"])
            or _sha256(path) != feature["dataset_sha256"]
        ):
            raise ValueError(f"M2.6 feature identity mismatch: {cutoff}")
        if feature["candidate_sha256"] != candidate["artifact"]["sha256"]:
            raise ValueError(f"M2.6 feature/candidate binding mismatch: {cutoff}")
        paths[cutoff] = path
    return metrics, paths


def _load_rows(
    feature_paths: dict[str, Path], cutoffs: list[str], predicate: str
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    selected = ",".join(
        ["customer_id", "article_id"] + ALL_FEATURES + ["target"]
    )
    for cutoff in cutoffs:
        connection = duckdb.connect()
        try:
            frame = connection.execute(
                f"SELECT {selected} FROM read_parquet({_literal(feature_paths[cutoff])}) "
                f"WHERE {predicate}"
            ).fetchdf()
        finally:
            connection.close()
        frame["target_cutoff"] = cutoff
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _train_stage1_binary(
    *,
    frame: pd.DataFrame,
    artifact_dir: Path,
    config: M2Config,
    category_maps: dict[str, dict[int, int]],
) -> tuple[Any, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required; install dependency.txt") from error
    if frame.empty or not bool((frame["candidate_rank"] > 100).all()):
        raise ValueError("stage-1 training requires image-only candidates")
    positives = int(frame["target"].sum())
    if positives == 0:
        raise ValueError("stage-1 training has no positive image-only rows")
    started = time.perf_counter()
    x_train = _prepare_frame(frame, STAGE1_FEATURES, category_maps)
    categorical = [
        feature for feature in CATEGORICAL_FEATURES if feature in STAGE1_FEATURES
    ]
    dataset = lgb.Dataset(
        x_train,
        label=frame["target"].astype(np.uint8),
        feature_name=STAGE1_FEATURES,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "average_precision"],
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
    evaluations: dict[str, Any] = {}
    model = lgb.train(
        params,
        dataset,
        num_boost_round=config.num_boost_round,
        valid_sets=[dataset],
        valid_names=["train"],
        callbacks=[lgb.record_evaluation(evaluations)],
    )
    model_path = artifact_dir / "lightgbm-stage1-image-binary.txt"
    model.save_model(str(model_path))
    importance = sorted(
        (
            {"feature": feature, "gain": float(gain), "split": int(split)}
            for feature, gain, split in zip(
                STAGE1_FEATURES,
                model.feature_importance(importance_type="gain"),
                model.feature_importance(importance_type="split"),
                strict=True,
            )
        ),
        key=lambda row: row["gain"],
        reverse=True,
    )
    evidence = {
        "objective": "binary_stage1_within_user_top100_only",
        "features": STAGE1_FEATURES,
        "parameters": params,
        "rows": len(frame),
        "positives": positives,
        "positive_rate": positives / len(frame),
        "unobserved_rows": len(frame) - positives,
        "class_weight": "none",
        "negative_sampling": "none_stage1_uses_all_image_only_train_rows",
        "train_metrics": {
            metric: float(values[-1])
            for metric, values in evaluations.get("train", {}).items()
        },
        "top_feature_importance": importance[:30],
        "model_path": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    del dataset, x_train
    gc.collect()
    return model, evidence


JOINT_FEATURES = ALL_FEATURES + ["stage1_score"]


def _score_and_rank_stage1(
    *,
    frame: pd.DataFrame,
    model: Any,
    category_maps: dict[str, dict[int, int]],
    output_path: Path,
    top_k: int,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    started = time.perf_counter()
    frame["stage1_score"] = model.predict(
        _prepare_frame(frame, STAGE1_FEATURES, category_maps)
    )
    frame.sort_values(
        [
            "target_cutoff",
            "customer_id",
            "stage1_score",
            "image_rank",
            "candidate_rank",
            "article_id",
        ],
        ascending=[True, True, False, True, True, True],
        kind="mergesort",
        inplace=True,
    )
    frame["stage1_rank"] = (
        frame.groupby(
            ["target_cutoff", "customer_id"], sort=False, observed=True
        ).cumcount()
        + 1
    )
    selected = frame["stage1_rank"] <= top_k
    positives = frame["target"].astype(bool)
    selected_positives = int((selected & positives).sum())
    total_positives = int(positives.sum())
    minimal = frame[
        [
            "target_cutoff",
            "customer_id",
            "article_id",
            "candidate_rank",
            "stage1_rank",
            "stage1_score",
            "target",
        ]
    ].copy()
    connection = duckdb.connect()
    try:
        connection.register("stage1_scores_frame", minimal)
        connection.execute(
            f"COPY stage1_scores_frame TO {_literal(output_path)} "
            "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
        )
    finally:
        connection.close()
    by_cutoff: dict[str, Any] = {}
    for cutoff, group in frame.groupby("target_cutoff", sort=True):
        keep = group["stage1_rank"] <= top_k
        truth = group["target"].astype(bool)
        positive_count = int(truth.sum())
        kept = int((keep & truth).sum())
        by_cutoff[str(cutoff)] = {
            "rows": len(group),
            "selected_rows": int(keep.sum()),
            "positives": positive_count,
            "selected_positives": kept,
            "positive_pair_recall": kept / positive_count if positive_count else None,
        }
    return {
        "top_k": top_k,
        "rows": len(frame),
        "selected_rows": int(selected.sum()),
        "positives": total_positives,
        "selected_positives": selected_positives,
        "positive_pair_recall": (
            selected_positives / total_positives if total_positives else None
        ),
        "by_cutoff": by_cutoff,
        "score_path": str(output_path.resolve()),
        "score_bytes": output_path.stat().st_size,
        "score_sha256": _sha256(output_path),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _load_joint_rows(
    *,
    feature_paths: dict[str, Path],
    cutoffs: list[str],
    stage_scores_path: Path,
    mode: str,
    top_k: int,
) -> pd.DataFrame:
    if mode not in {"strict", "rescue", "full"}:
        raise ValueError(f"unknown joint row mode: {mode}")
    frames: list[pd.DataFrame] = []
    base_select = ",".join(f"f.{feature}" for feature in ALL_FEATURES)
    for cutoff in cutoffs:
        if mode == "strict":
            predicate = f"f.candidate_rank<=100 OR s.stage1_rank<={top_k}"
        elif mode == "rescue":
            predicate = (
                f"f.candidate_rank<=100 OR s.stage1_rank<={top_k} "
                "OR (f.candidate_rank>100 AND f.target=1)"
            )
        else:
            predicate = "TRUE"
        connection = duckdb.connect()
        try:
            frame = connection.execute(
                f"""
                SELECT f.customer_id,f.article_id,{base_select},f.target,
                       s.stage1_score
                FROM read_parquet({_literal(feature_paths[cutoff])}) f
                LEFT JOIN read_parquet({_literal(stage_scores_path)}) s
                  ON s.target_cutoff='{cutoff}'
                 AND f.customer_id=s.customer_id
                 AND f.article_id=s.article_id
                WHERE {predicate}
                """
            ).fetchdf()
        finally:
            connection.close()
        frame["target_cutoff"] = cutoff
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if mode != "full" and len(result) == 0:
        raise RuntimeError("joint candidate selection returned no rows")
    return result


def _combine_stage_score_artifacts(
    paths: list[Path], output_path: Path
) -> dict[str, Any]:
    if len(paths) != 2 or output_path.exists():
        raise ValueError("OOF stage-score combine requires two new fold artifacts")
    source_list = ",".join(_literal(path) for path in paths)
    connection = duckdb.connect()
    try:
        connection.execute(
            f"COPY (SELECT * FROM read_parquet([{source_list}]) "
            "ORDER BY target_cutoff,customer_id,stage1_rank,article_id) "
            f"TO {_literal(output_path)} "
            "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
        )
        stats = connection.execute(
            f"SELECT count(*),count(DISTINCT target_cutoff),"
            f"count(DISTINCT (target_cutoff,customer_id,article_id)) "
            f"FROM read_parquet({_literal(output_path)})"
        ).fetchone()
    finally:
        connection.close()
    if stats is None or int(stats[0]) != int(stats[2]) or int(stats[1]) != 2:
        raise RuntimeError("combined OOF stage-score identity failed")
    return {
        "path": str(output_path.resolve()),
        "rows": int(stats[0]),
        "cutoffs": int(stats[1]),
        "bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
    }

def _prepare_variable_groups(
    frame: pd.DataFrame, minimum: int, maximum: int
) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    ordered = frame.sort_values(
        ["target_cutoff", "customer_id", "candidate_rank", "article_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    grouped = ordered.groupby(
        ["target_cutoff", "customer_id"], sort=False, observed=True
    )["target"]
    sizes = grouped.size()
    invalid = sizes[(sizes < minimum) | (sizes > maximum)]
    if sizes.empty or len(invalid):
        raise ValueError(
            f"joint groups must contain {minimum}..{maximum} rows; "
            f"invalid_groups={len(invalid)}"
        )
    positives = grouped.sum()
    positive_groups = int((positives > 0).sum())
    evidence = {
        "groups": len(sizes),
        "rows": len(ordered),
        "min_group_rows": int(sizes.min()),
        "max_group_rows": int(sizes.max()),
        "positive_groups": positive_groups,
        "zero_positive_groups": len(sizes) - positive_groups,
        "positive_group_rate": positive_groups / len(sizes),
    }
    return ordered, sizes.astype(int).tolist(), evidence


def _score_joint_frame(
    *,
    frame: pd.DataFrame,
    model: Any,
    category_maps: dict[str, dict[int, int]],
    variant: str,
    evaluation_db: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    if evaluation_db.exists() or prediction_path.exists():
        raise FileExistsError(evaluation_db if evaluation_db.exists() else prediction_path)
    started = time.perf_counter()
    output = frame[
        [
            "customer_id",
            "article_id",
            "candidate_rank",
            "target",
            "user_history_events_12w",
        ]
    ].copy()
    output[f"score_{variant}"] = model.predict(
        _prepare_frame(frame, JOINT_FEATURES, category_maps)
    )
    connection = duckdb.connect(str(evaluation_db))
    try:
        connection.register("m27_predictions", output)
        connection.execute("CREATE TABLE predictions AS SELECT * FROM m27_predictions")
        duplicates = connection.execute(
            "SELECT count(*)-count(DISTINCT (customer_id,article_id)) FROM predictions"
        ).fetchone()[0]
        if int(duplicates):
            raise RuntimeError("M2.7 predictions are not unique by customer-item")
        connection.execute(
            f"COPY predictions TO {_literal(prediction_path)} "
            "(FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)"
        )
    finally:
        connection.close()
    return {
        "rows": len(output),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_bytes": prediction_path.stat().st_size,
        "prediction_sha256": _sha256(prediction_path),
        "evaluation_db": str(evaluation_db.resolve()),
        "elapsed_seconds": time.perf_counter() - started,
    }

def _summary(
    development: dict[str, Any],
    m26_metrics: dict[str, Any],
    m21_metrics: dict[str, Any],
    metric_k: int,
) -> dict[str, Any]:
    metric = f"map@{metric_k}"
    orderings: dict[str, Any] = {}
    for variant in VARIANTS:
        name = f"{variant}__inactive_rrf"
        values = {
            dev: float(
                result["variants"][variant]["evaluation"]["orderings"][name][
                    "segments"
                ]["overall"][metric]
            )
            for dev, result in development.items()
        }
        orderings[name] = {
            "window_map@12": values,
            "mean_map@12": float(np.mean(list(values.values()))),
            "min_map@12": float(np.min(list(values.values()))),
        }
    frozen = m26_metrics["development_summary"]["frozen_m2_2"]
    m26_name = m26_metrics["development_summary"]["selection_gate"][
        "selected_variant"
    ]
    m26 = m26_metrics["development_summary"]["orderings"][m26_name]
    selected = max(
        orderings,
        key=lambda name: (
            orderings[name]["mean_map@12"],
            orderings[name]["min_map@12"],
            name,
        ),
    )
    selected_values = orderings[selected]["window_map@12"]
    deltas = {
        dev: selected_values[dev] - float(frozen["window_map@12"][dev])
        for dev in PROTOCOL
    }
    gate = {
        "accepted": all(delta >= 0 for delta in deltas.values())
        and orderings[selected]["mean_map@12"] > float(frozen["mean_map@12"]),
        "rule": "mean MAP@12 improves over frozen M2.2 and neither dev window regresses",
        "selected_variant": selected,
        "window_delta_vs_frozen_map@12": deltas,
        "mean_delta_vs_frozen_map@12": (
            orderings[selected]["mean_map@12"] - float(frozen["mean_map@12"])
        ),
    }
    frozen_name = m21_metrics["development_summary"][
        "selected_development_candidate"
    ]
    activity: dict[str, Any] = {}
    for dev in PROTOCOL:
        variant = selected.removesuffix("__inactive_rrf")
        new_rows = development[dev]["variants"][variant]["evaluation"][
            "orderings"
        ][selected]["activity_segments"]
        old_rows = m21_metrics["development"][dev]["evaluation"]["orderings"][
            frozen_name
        ]["activity_segments"]
        new_map = {row["activity_segment"]: row for row in new_rows}
        old_map = {row["activity_segment"]: row for row in old_rows}
        activity[dev] = {
            segment: {
                "map@12": float(new_map[segment][metric]),
                "delta_vs_frozen_map@12": (
                    float(new_map[segment][metric])
                    - float(old_map[segment][metric])
                ),
            }
            for segment in sorted(new_map)
        }
    return {
        "orderings": orderings,
        "frozen_m2_2": frozen,
        "m2_6_best": {"name": m26_name, **m26},
        "selection_gate": gate,
        "selected_activity": activity,
        "final_week": "not_run",
    }


def _render_report(result: dict[str, Any]) -> str:
    summary = result["development_summary"]
    gate = summary["selection_gate"]
    lines = [
        "# M2.7 two-stage image pruning and hard-negative diagnostic",
        "",
        "## 结论",
        "",
        f"- development gate：{'通过' if gate['accepted'] else '未通过'}；"
        f"最佳候选 `{gate['selected_variant']}`。",
        f"- 相对冻结 M2.2 mean MAP@12："
        f"{gate['mean_delta_vs_frozen_map@12']:+.6f}。",
        "- stage-1 是无 class weight、无负采样的 image-only binary scorer；"
        "训练 joint 的 stage-1 分数来自 customer-hash OOF；验证分数来自 pooled-train scorer；"
        "只用于每用户 Top100 pruning，不能解释为 CTR。",
        "- two-stage strict train/validation 都不补回漏召回正例；hard-negative 只在训练"
        "补回正例，验证仍使用完整 expanded-300。",
        "- 最终周未运行；本报告不是 Kaggle leaderboard 分数。",
        "",
        "## MAP@12",
        "",
        "| ordering | dev-A | dev-B | mean | worst |",
        "|---|---:|---:|---:|---:|",
    ]
    frozen = summary["frozen_m2_2"]
    lines.append(
        f"| frozen M2.2 | {frozen['window_map@12']['dev_a']:.6f} | "
        f"{frozen['window_map@12']['dev_b']:.6f} | "
        f"{frozen['mean_map@12']:.6f} | "
        f"{min(frozen['window_map@12'].values()):.6f} |"
    )
    m26 = summary["m2_6_best"]
    lines.append(
        f"| M2.6 best | {m26['window_map@12']['dev_a']:.6f} | "
        f"{m26['window_map@12']['dev_b']:.6f} | {m26['mean_map@12']:.6f} | "
        f"{m26['min_map@12']:.6f} |"
    )
    for name, row in summary["orderings"].items():
        lines.append(
            f"| {name} | {row['window_map@12']['dev_a']:.6f} | "
            f"{row['window_map@12']['dev_b']:.6f} | {row['mean_map@12']:.6f} | "
            f"{row['min_map@12']:.6f} |"
        )
    lines.extend(["", "## Stage-1 positive retention", ""])
    for dev, evidence in result["development"].items():
        validation = evidence["stage1_validation"]
        lines.append(
            f"- {dev} ({evidence['validation_cutoff']})：image-only positives "
            f"{validation['selected_positives']}/{validation['positives']}，"
            f"Top100 retention={validation['positive_pair_recall']:.2%}。"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 所有 stage-1/joint 模型只使用各验证 cutoff 之前的训练目标周；"
            "validation truth 不参与 pruning scorer、训练采样或 category encoding。",
            "- two-stage 的 candidate Recall/Oracle 是实际 selected pool 的指标；"
            "hard-negative 的 candidate Recall/Oracle 来自完整 expanded-300。",
            "- hard-negative 正例补回只发生在训练集；这是 label-aware training sampling，"
            "不是可用于线上候选生成的规则。",
            "- 未观察候选仍不是真负例；optimistic_all_articles 与非 nested retrieval "
            "selection 边界继续存在。",
            "",
            "## 产物",
            "",
            f"- metrics：{result['artifacts']['metrics']}",
            f"- private artifacts：{result['artifacts']['artifact_dir']}（Git ignored）",
            "",
        ]
    )
    return "\n".join(lines)


def run_m27(
    *,
    transactions_path: Path,
    m26_metrics_path: Path,
    m21_metrics_path: Path,
    cache_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
    stage1_top_k: int = 100,
) -> dict[str, Any]:
    if config.evaluation_role != "development":
        raise ValueError("M2.7 is development-only")
    if stage1_top_k != 100:
        raise ValueError("M2.7 freezes stage1_top_k=100")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        m26_metrics, feature_paths = validate_m26_reuse(
            m26_metrics_path, cache_dir
        )
        m21_metrics = _read_json(m21_metrics_path.resolve())
        if (
            m21_metrics.get("development_summary", {}).get(
                "selected_development_candidate"
            )
            != "pooled_lambdarank_full__inactive_rrf"
        ):
            raise ValueError("unexpected frozen M2.2 model form")
        development: dict[str, Any] = {}
        for dev, protocol in PROTOCOL.items():
            dev_dir = artifact_dir / dev
            dev_dir.mkdir()
            oof_folds: dict[str, Any] = {}
            oof_paths: list[Path] = []
            for holdout_fold in (0, 1):
                fold_name = f"fold_{holdout_fold}"
                fold_dir = dev_dir / f"stage1-oof-{fold_name}"
                fold_dir.mkdir()
                train_predicate = (
                    "candidate_rank>100 AND "
                    f"hash(customer_id)%2<>{holdout_fold}"
                )
                holdout_predicate = (
                    "candidate_rank>100 AND "
                    f"hash(customer_id)%2={holdout_fold}"
                )
                fold_train = _load_rows(
                    feature_paths, protocol["train"], train_predicate
                )
                fold_maps = build_category_maps(fold_train)
                fold_map_evidence = _save_category_maps(
                    fold_dir / "category-maps.json", fold_maps
                )
                fold_model, fold_model_evidence = _train_stage1_binary(
                    frame=fold_train,
                    artifact_dir=fold_dir,
                    config=config,
                    category_maps=fold_maps,
                )
                del fold_train
                gc.collect()
                fold_holdout = _load_rows(
                    feature_paths, protocol["train"], holdout_predicate
                )
                fold_score_path = fold_dir / "holdout-scores.parquet"
                fold_scoring = _score_and_rank_stage1(
                    frame=fold_holdout,
                    model=fold_model,
                    category_maps=fold_maps,
                    output_path=fold_score_path,
                    top_k=stage1_top_k,
                )
                del fold_holdout, fold_model
                gc.collect()
                oof_paths.append(fold_score_path)
                oof_folds[fold_name] = {
                    "train_cutoffs": protocol["train"],
                    "train_user_hash_fold": 1-holdout_fold,
                    "holdout_user_hash_fold": holdout_fold,
                    "model": fold_model_evidence,
                    "category_encoding": fold_map_evidence,
                    "holdout_scoring": fold_scoring,
                }
            train_scores_path = dev_dir / "stage1-train-oof-scores.parquet"
            oof_artifact = _combine_stage_score_artifacts(
                oof_paths, train_scores_path
            )
            fold_rows = [fold["holdout_scoring"] for fold in oof_folds.values()]
            oof_positives = sum(row["positives"] for row in fold_rows)
            oof_selected_positives = sum(
                row["selected_positives"] for row in fold_rows
            )
            by_cutoff: dict[str, Any] = {}
            for cutoff in protocol["train"]:
                cutoff_rows = [
                    row["by_cutoff"][cutoff] for row in fold_rows
                ]
                cutoff_positives = sum(row["positives"] for row in cutoff_rows)
                cutoff_selected_positives = sum(
                    row["selected_positives"] for row in cutoff_rows
                )
                by_cutoff[cutoff] = {
                    "rows": sum(row["rows"] for row in cutoff_rows),
                    "selected_rows": sum(
                        row["selected_rows"] for row in cutoff_rows
                    ),
                    "positives": cutoff_positives,
                    "selected_positives": cutoff_selected_positives,
                    "positive_pair_recall": (
                        cutoff_selected_positives / cutoff_positives
                        if cutoff_positives else None
                    ),
                }
            stage_train = {
                "protocol": "two_fold_customer_hash_oof",
                "fold_rule": "duckdb hash(customer_id) % 2; same user same fold",
                "top_k": stage1_top_k,
                "rows": sum(row["rows"] for row in fold_rows),
                "selected_rows": sum(row["selected_rows"] for row in fold_rows),
                "positives": oof_positives,
                "selected_positives": oof_selected_positives,
                "positive_pair_recall": (
                    oof_selected_positives / oof_positives
                ),
                "by_cutoff": by_cutoff,
                "artifact": oof_artifact,
            }
            image_train = _load_rows(
                feature_paths, protocol["train"], "candidate_rank>100"
            )
            stage_maps = build_category_maps(image_train)
            stage_map_evidence = _save_category_maps(
                dev_dir / "category-maps-stage1-final.json", stage_maps
            )
            stage_model, stage_model_evidence = _train_stage1_binary(
                frame=image_train,
                artifact_dir=dev_dir,
                config=config,
                category_maps=stage_maps,
            )
            del image_train
            gc.collect()
            validation_cutoff = protocol["validation"]
            image_valid = _load_rows(
                feature_paths, [validation_cutoff], "candidate_rank>100"
            )
            valid_scores_path = dev_dir / "stage1-validation-scores.parquet"
            stage_validation = _score_and_rank_stage1(
                frame=image_valid,
                model=stage_model,
                category_maps=stage_maps,
                output_path=valid_scores_path,
                top_k=stage1_top_k,
            )
            del image_valid, stage_model
            gc.collect()
            variants: dict[str, Any] = {}
            variant_modes = {
                "two_stage_200": ("strict", "strict"),
                "hardneg_100_full_eval": ("rescue", "full"),
            }
            for variant, (train_mode, valid_mode) in variant_modes.items():
                variant_dir = dev_dir / variant
                variant_dir.mkdir()
                train_frame = _load_joint_rows(
                    feature_paths=feature_paths,
                    cutoffs=protocol["train"],
                    stage_scores_path=train_scores_path,
                    mode=train_mode,
                    top_k=stage1_top_k,
                )
                train_frame, group_sizes, group_evidence = _prepare_variable_groups(
                    train_frame, 100, 300
                )
                maps = build_category_maps(train_frame)
                map_evidence = _save_category_maps(
                    variant_dir / "category-maps.json", maps
                )
                model, model_evidence = _train_lambdarank(
                    frame=train_frame,
                    group_sizes=group_sizes,
                    group_evidence=group_evidence,
                    features=JOINT_FEATURES,
                    name=variant,
                    artifact_dir=variant_dir,
                    config=config,
                    category_maps=maps,
                )
                del train_frame
                gc.collect()
                valid_frame = _load_joint_rows(
                    feature_paths=feature_paths,
                    cutoffs=[validation_cutoff],
                    stage_scores_path=valid_scores_path,
                    mode=valid_mode,
                    top_k=stage1_top_k,
                )
                valid_frame, _, valid_group_evidence = _prepare_variable_groups(
                    valid_frame, 100, 300
                )
                scoring = _score_joint_frame(
                    frame=valid_frame,
                    model=model,
                    category_maps=maps,
                    variant=variant,
                    evaluation_db=variant_dir / "evaluation.duckdb",
                    prediction_path=variant_dir / "validation-predictions.parquet",
                )
                del valid_frame, model
                gc.collect()
                evaluation = _evaluate_models(
                    evaluation_db=variant_dir / "evaluation.duckdb",
                    transactions_path=transactions_path,
                    cutoff=validation_cutoff,
                    metric_k=config.metric_k,
                    variant_names=[variant],
                )
                if variant == "two_stage_200":
                    for ordering in evaluation["orderings"].values():
                        for segment in ordering["segments"].values():
                            segment["candidate_recall@selected_pool"] = segment[
                                "candidate_recall@expanded_pool"
                            ]
                variants[variant] = {
                    "train_mode": train_mode,
                    "validation_mode": valid_mode,
                    "model": model_evidence,
                    "category_encoding": map_evidence,
                    "validation_groups": valid_group_evidence,
                    "scoring": scoring,
                    "evaluation": evaluation,
                }
            development[dev] = {
                "train_cutoffs": protocol["train"],
                "validation_cutoff": validation_cutoff,
                "stage1_oof_folds": oof_folds,
                "stage1_oof_artifact": oof_artifact,
                "stage1_model": stage_model_evidence,
                "stage1_category_encoding": stage_map_evidence,
                "stage1_train": stage_train,
                "stage1_validation": stage_validation,
                "variants": variants,
            }
        summary = _summary(
            development, m26_metrics, m21_metrics, config.metric_k
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M2.7",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "development_only_no_final_week",
            "config": {**asdict(config), "stage1_top_k": stage1_top_k},
            "contract": {
                "stage1_train_scores": "two-fold customer-hash OOF binary scores",
                "stage1_validation_scores": "pooled-train binary scores",
                "stage1": "binary image-only all rows, no class weight",
                "stage1_features": STAGE1_FEATURES,
                "joint_features": JOINT_FEATURES,
                "two_stage_200": "baseline100 + strict stage1 image top100",
                "hardneg_100_full_eval_train": (
                    "baseline100 + stage1 image top100 + all image positives"
                ),
                "hardneg_100_full_eval_validation": "full expanded300",
                "final_week": "not_run",
            },
            "inputs": {
                "m26_metrics": str(m26_metrics_path.resolve()),
                "m26_metrics_sha256": _sha256(m26_metrics_path.resolve()),
                "m21_metrics": str(m21_metrics_path.resolve()),
                "m21_metrics_sha256": _sha256(m21_metrics_path.resolve()),
                "feature_paths": {
                    cutoff: {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for cutoff, path in feature_paths.items()
                },
            },
            "development": development,
            "development_summary": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M2_7_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path.resolve()),
            "report": str(report_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
        }
        _write_json(metrics_path, result)
        report_path.write_text(_render_report(result), encoding="utf-8")
        return result
    except Exception as error:
        _write_json(
            output_dir / f"failure-{time.time_ns()}.json",
            {
                "schema_version": "m2.7-failure-v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
