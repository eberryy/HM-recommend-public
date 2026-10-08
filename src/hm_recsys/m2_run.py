from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import prepare_tabular_connection
from .m2 import (
    CATEGORICAL_FEATURES,
    FULL_FEATURES,
    RETRIEVAL_FEATURES,
    SCHEMA_VERSION,
    M2Config,
    _create_static_dimensions,
    _load_training_frame,
    _reference_parity,
    _sha256,
    _score_validation,
    _train_model,
    _write_json,
    build_category_maps,
    build_point_in_time_dataset,
    evaluate_predictions,
    validate_candidate_artifact,
)


def _render_report(result: dict[str, Any]) -> str:
    config = result["config"]
    metric_k = config["metric_k"]
    map_key = f"map@{metric_k}"
    recall_key = f"recall@{metric_k}"
    hit_key = f"hit_rate@{metric_k}"
    train = result["datasets"]["train"]
    valid = result["datasets"]["validation"]
    train_rate = result["candidate_inputs"]["train"]["sample_rate"]
    valid_rate = result["candidate_inputs"]["validation"]["sample_rate"]
    is_final = config["evaluation_role"] == "final"
    evaluation_label = "最终验证周" if is_final else "开发验证周"
    evaluation_scope = "full" if valid_rate == 1.0 else f"hash {valid_rate:.0%}"
    orderings = result["evaluation"]["orderings"]
    primary_map = orderings["lightgbm_full"]["segments"]["overall"][map_key]
    rrf_map = orderings["rrf"]["segments"]["overall"][map_key]
    retrieval_map = orderings["lightgbm_retrieval"]["segments"]["overall"][map_key]
    lines = [
        "# M2 LightGBM binary baseline",
        "",
        "## 结论",
        "",
        f"- primary model：`lightgbm_full`；最终周 full MAP@{metric_k} = `{primary_map:.6f}`。",
        f"- 原 retrieval-v1 RRF MAP@{metric_k} = `{rrf_map:.6f}`；"
        f"绝对变化 `{result['ranking_gain']['full_minus_rrf_map@12']:+.6f}`。",
        f"- 仅召回特征 LightGBM MAP@{metric_k} = `{retrieval_map:.6f}`；"
        f"full 相对 retrieval-only 变化 `{result['ranking_gain']['full_minus_retrieval_map@12']:+.6f}`。",
        "- 排序没有改变候选池；M1 Recall@100、HitRate@100、Oracle MAP@12 parity 已通过。",
        "",
        "## 实验合同",
        "",
        f"- 训练目标周：`{train['cutoff']}`，固定 hash 10% 用户，"
        f"{train['rows']:,} 行、{train['positives']:,} 个候选内正例。",
        f"- 最终验证周：`{valid['cutoff']}`，full {valid['users']:,} 用户、"
        f"{valid['rows']:,} 行；最终周未用于 fitting、early stopping 或参数选择。",
        "- label：目标周 distinct user-item 出现为 1，否则为 0；0 仅表示未观察到购买。",
        "- negative sampling：none；保留全部召回但未购买候选。",
        f"- point-in-time history：cutoff 前 {config['history_weeks']} 周；customer snapshot 主实验只使用 age。",
        "- catalog：optimistic_all_articles；商品 metadata 假设预测时可见，真实 listing 时间未知。",
        "",
        "## Overall 指标",
        "",
        "| ordering | MAP@12 | Recall@12 | HitRate@12 | Candidate Recall@100 | Oracle MAP@12 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("rrf", "lightgbm_retrieval", "lightgbm_full"):
        metric = orderings[name]["segments"]["overall"]
        lines.append(
            f"| {name} | {metric[map_key]:.6f} | {metric[recall_key]:.6f} | "
            f"{metric[hit_key]:.6f} | {metric['candidate_recall@100']:.6f} | "
            f"{metric[f'oracle_map@{metric_k}']:.6f} |"
        )
    lines.extend(["", "## Warm / cold", "", "| segment | RRF MAP@12 | Full LGB MAP@12 | Candidate Recall@100 |", "|---|---:|---:|---:|"])
    for segment in ("warm", "cold"):
        rrf = orderings["rrf"]["segments"][segment]
        full = orderings["lightgbm_full"]["segments"][segment]
        lines.append(
            f"| {segment} | {rrf[map_key]:.6f} | {full[map_key]:.6f} | "
            f"{full['candidate_recall@100']:.6f} |"
        )
    lines.extend(["", "## Primary model top feature gain", "", "| feature | gain | splits |", "|---|---:|---:|"])
    for row in result["models"]["full"]["top_feature_importance"][:15]:
        lines.append(f"| {row['feature']} | {row['gain']:.2f} | {row['split']} |")
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- 这是 sampled-training/full-validation 的第一版 binary baseline，不是全量多周训练结论。",
            "- 只有一次较早训练目标周；跨窗口稳定性、LambdaRank 和正式特征消融仍属于后续 M2/M3。",
            "- 没有曝光日志，因此不能把 target=0 解释为用户看过但不感兴趣。",
            "- full 模型是预注册 primary；最终周结果不会反向用于本轮调参。",
            "- 图片、文本 embedding 未启动。",
            "",
            "## 复现产物",
            "",
            f"- metrics：`{result['artifacts']['metrics']}`",
            f"- private manifest：`{result['artifacts']['manifest']}`",
            f"- models / binary datasets / predictions：`{result['artifacts']['artifact_dir']}`（Git ignored）",
            "",
        ]
    )
    return "\n".join(lines)


def run_m2(
    raw_dir: Path,
    work_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    train_candidates: Path,
    train_manifest: Path,
    validation_candidates: Path,
    validation_manifest: Path,
    m1_reference_metrics: Path,
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    config.validate()
    for directory in (output_dir, artifact_dir):
        if directory.exists():
            raise FileExistsError(directory)
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    train_identity = validate_candidate_artifact(train_candidates, train_manifest, config.candidate_k)
    valid_identity = validate_candidate_artifact(validation_candidates, validation_manifest, config.candidate_k)
    if train_identity["cutoff"] >= valid_identity["cutoff"]:
        raise ValueError("training target cutoff must precede validation cutoff")
    if config.evaluation_role == "final" and valid_identity["sample_rate"] != 1.0:
        raise ValueError("primary M2 final validation must use full users")
    connection = prepare_tabular_connection(raw_dir, work_dir)
    train_dataset_path = artifact_dir / "train_features.parquet"
    valid_dataset_path = artifact_dir / "validation_features.parquet"
    try:
        _create_static_dimensions(connection)
        train_dataset = build_point_in_time_dataset(
            connection, train_candidates, train_identity, train_dataset_path, config
        )
        valid_dataset = build_point_in_time_dataset(
            connection, validation_candidates, valid_identity, valid_dataset_path, config
        )
    finally:
        connection.close()
    contract = {
        "objective": "binary",
        "group_key": "customer_id",
        "label": "1 iff distinct user-item appears in target week; else 0",
        "zero_label_semantics": "retrieved candidate with no observed target-week purchase; not proven negative",
        "negative_sampling": "none_keep_all_candidates",
        "customer_snapshot_policy": "age_only",
        "catalog_protocol": "optimistic_all_articles",
        "evaluation_role": config.evaluation_role,
    }
    dataset_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "contract": contract,
        "candidate_inputs": {"train": train_identity, "validation": valid_identity},
        "datasets": {"train": train_dataset, "validation": valid_dataset},
        "features": {
            "retrieval": RETRIEVAL_FEATURES,
            "tabular_and_cross": [feature for feature in FULL_FEATURES if feature not in RETRIEVAL_FEATURES],
            "categorical": CATEGORICAL_FEATURES,
        },
        "raw_files": {
            name: {
                "path": str((raw_dir / name).resolve()),
                "bytes": (raw_dir / name).stat().st_size,
                "mtime_ns": (raw_dir / name).stat().st_mtime_ns,
            }
            for name in ("transactions_train.csv", "articles.csv", "customers.csv")
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    _write_json(manifest_path, dataset_manifest)
    train_frame = _load_training_frame(train_dataset_path)
    category_maps = build_category_maps(train_frame)
    category_map_path = artifact_dir / "category_maps.json"
    _write_json(
        category_map_path,
        {
            feature: {str(raw): encoded for raw, encoded in mapping.items()}
            for feature, mapping in category_maps.items()
        },
    )
    dataset_manifest["category_encoding"] = {
        "policy": "fit_dense_codes_on_training_only_unknown_or_missing_to_zero",
        "cardinalities": {feature: len(mapping) for feature, mapping in category_maps.items()},
        "path": str(category_map_path.resolve()),
        "bytes": category_map_path.stat().st_size,
        "sha256": _sha256(category_map_path),
    }
    _write_json(manifest_path, dataset_manifest)
    models: dict[str, Any] = {}
    model_evidence: dict[str, Any] = {}
    for name, features in (("retrieval", RETRIEVAL_FEATURES), ("full", FULL_FEATURES)):
        models[name], model_evidence[name] = _train_model(
            train_frame, features, name, artifact_dir, config, category_maps
        )
    del train_frame
    prediction_path = artifact_dir / "validation_predictions.parquet"
    evaluation_db = artifact_dir / "evaluation.duckdb"
    scoring = _score_validation(
        valid_dataset_path, models, evaluation_db, prediction_path, config, category_maps
    )
    del models
    evaluation = evaluate_predictions(
        evaluation_db, work_dir / "transactions.parquet", valid_identity["cutoff"], config.metric_k
    )
    parity = _reference_parity(evaluation, m1_reference_metrics)
    base = evaluation["orderings"]["rrf"]["segments"]["overall"]["map@12"]
    retrieval = evaluation["orderings"]["lightgbm_retrieval"]["segments"]["overall"]["map@12"]
    full = evaluation["orderings"]["lightgbm_full"]["segments"]["overall"]["map@12"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M2",
        "status": "measured",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_scope": f"deterministic_hash_{train_identity['sample_rate']:.6g}_users_single_earlier_week",
        "validation_scope": (
            "full_final_week" if config.evaluation_role == "final"
            else f"deterministic_hash_{valid_identity['sample_rate']:.6g}_users_development_week"
        ),
        "primary_model": "lightgbm_full",
        "config": asdict(config),
        "contract": contract,
        "candidate_inputs": dataset_manifest["candidate_inputs"],
        "datasets": dataset_manifest["datasets"],
        "models": model_evidence,
        "scoring": scoring,
        "evaluation": evaluation,
        "m1_parity": parity,
        "ranking_gain": {
            "full_minus_rrf_map@12": full - base,
            "retrieval_minus_rrf_map@12": retrieval - base,
            "full_minus_retrieval_map@12": full - retrieval,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "M2_REPORT.md"
    result["artifacts"] = {
        "metrics": str(metrics_path.resolve()),
        "report": str(report_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
    }
    _write_json(metrics_path, result)
    if config.evaluation_role == "development":
        from .m2_dev_report import render_development_report

        report_text = render_development_report(result)
    else:
        report_text = _render_report(result)
    report_path.write_text(report_text, encoding="utf-8")
    return result
