from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .m2 import (
    FULL_FEATURES,
    RETRIEVAL_FEATURES,
    SCHEMA_VERSION as M2_SCHEMA_VERSION,
    TABULAR_FEATURES,
    M2Config,
    _read_json,
    _reference_parity,
    _sha256,
    _write_json,
)
from .m21 import (
    SCHEMA_VERSION as M21_SCHEMA_VERSION,
    _evaluate_variants,
    _load_window_frame,
    _score_variants,
    _train_variant,
)


SCHEMA_VERSION = "m2.2-feature-ablation-v1"
FEATURE_GROUPS: dict[str, list[str]] = {
    "customer_static": [
        "customer_age",
        "customer_age_missing",
        "age_bucket",
    ],
    "user_history": [
        "user_history_events_12w",
        "user_unique_items_12w",
        "user_avg_price_12w",
        "user_online_share_12w",
        "user_days_since_last_purchase",
    ],
    "item_trend": [
        "item_events_7d",
        "item_events_28d",
        "item_events_12w",
        "item_unique_customers_28d",
        "item_avg_price_28d",
        "item_days_since_last_sale",
        "item_trend_7d_vs_28d",
    ],
    "user_item_affinity": [
        "user_item_events_12w",
        "user_item_days_since_last_purchase",
        "user_product_type_events_12w",
        "user_product_type_share_12w",
        "user_department_events_12w",
        "user_department_share_12w",
        "user_item_price_gap",
    ],
    "article_attributes": [
        "article_product_type_no",
        "article_garment_group_no",
        "article_department_no",
        "article_index_group_no",
        "article_colour_master_id",
    ],
}
BASELINE_ALIASES = {
    "retrieval": "pooled_lambdarank_retrieval",
    "full": "pooled_lambdarank_full",
}
DEVELOPMENT_PROTOCOL = {
    "dev_a": {
        "train": ["2020-05-27", "2020-06-24"],
        "validation": "2020-07-22",
    },
    "dev_b": {
        "train": ["2020-06-24", "2020-07-22"],
        "validation": "2020-08-19",
    },
}


def validate_feature_groups() -> None:
    flattened = [
        feature for features in FEATURE_GROUPS.values() for feature in features
    ]
    duplicates = sorted(
        {feature for feature in flattened if flattened.count(feature) > 1}
    )
    if duplicates:
        raise ValueError(f"feature groups overlap: {duplicates}")
    missing = sorted(set(TABULAR_FEATURES) - set(flattened))
    extra = sorted(set(flattened) - set(TABULAR_FEATURES))
    if missing or extra:
        raise ValueError(f"feature group partition mismatch: missing={missing}, extra={extra}")


def build_ablation_features() -> dict[str, list[str]]:
    validate_feature_groups()
    variants: dict[str, list[str]] = {}
    for group, features in FEATURE_GROUPS.items():
        variants[f"add_{group}"] = RETRIEVAL_FEATURES + features
        excluded = set(features)
        variants[f"drop_{group}"] = [
            feature for feature in FULL_FEATURES if feature not in excluded
        ]
    return variants


def _require_bound_file(
    evidence: dict[str, Any],
    path_key: str,
    bytes_key: str,
    sha_key: str,
    root: Path,
) -> Path:
    path = Path(evidence[path_key]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"reused artifact escapes M2.1 root: {path}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(evidence[bytes_key]):
        raise ValueError(f"reused artifact bytes mismatch: {path}")
    if _sha256(path) != evidence[sha_key]:
        raise ValueError(f"reused artifact SHA256 mismatch: {path}")
    return path


def _load_category_maps(path: Path) -> dict[str, dict[int, int]]:
    raw = _read_json(path)
    return {
        feature: {int(value): int(code) for value, code in mapping.items()}
        for feature, mapping in raw.items()
    }


def validate_m21_reuse(
    m21_artifact_dir: Path, m21_metrics_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = m21_artifact_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    metrics = _read_json(m21_metrics_path.resolve())
    if manifest.get("schema_version") != M21_SCHEMA_VERSION:
        raise ValueError("M2.1 manifest schema mismatch")
    if metrics.get("schema_version") != M21_SCHEMA_VERSION:
        raise ValueError("M2.1 metrics schema mismatch")
    if metrics.get("status") != "measured":
        raise ValueError("M2.1 source is not measured")
    if metrics.get("scope") != "development_only_no_final_week":
        raise ValueError("M2.1 source is not development-only")
    if metrics.get("development_summary", {}).get("final_confirmation_run") != "not_run":
        raise ValueError("M2.1 source final-week boundary changed")
    if manifest.get("run_id") != metrics.get("run_id"):
        raise ValueError("M2.1 run identity mismatch")
    declared_manifest = Path(metrics["artifacts"]["manifest"]).resolve()
    if declared_manifest != manifest_path:
        raise ValueError("M2.1 metrics point to a different manifest")
    required_cutoffs = {
        cutoff
        for protocol in DEVELOPMENT_PROTOCOL.values()
        for cutoff in protocol["train"] + [protocol["validation"]]
    }
    if set(manifest.get("datasets", {})) != required_cutoffs:
        raise ValueError("M2.1 feature cutoff set changed")
    dataset_paths: dict[str, str] = {}
    for cutoff in sorted(required_cutoffs):
        manifest_evidence = manifest["datasets"][cutoff]
        metrics_evidence = metrics["datasets"][cutoff]
        for key in ("dataset_path", "dataset_bytes", "dataset_sha256", "cutoff"):
            if manifest_evidence[key] != metrics_evidence[key]:
                raise ValueError(f"M2.1 dataset evidence mismatch: {cutoff} {key}")
        if metrics_evidence["cutoff"] != cutoff:
            raise ValueError(f"M2.1 dataset cutoff mismatch: {cutoff}")
        path = _require_bound_file(
            metrics_evidence,
            "dataset_path",
            "dataset_bytes",
            "dataset_sha256",
            root,
        )
        dataset_paths[cutoff] = str(path)
    category_maps: dict[str, Any] = {}
    baseline_models: dict[str, Any] = {}
    references: dict[str, str] = {}
    for dev_name, protocol in DEVELOPMENT_PROTOCOL.items():
        dev = metrics["development"][dev_name]
        if dev["pooled_train_cutoffs"] != protocol["train"]:
            raise ValueError(f"M2.1 pooled protocol mismatch: {dev_name}")
        if dev["validation_cutoff"] != protocol["validation"]:
            raise ValueError(f"M2.1 validation protocol mismatch: {dev_name}")
        if dev["m1_parity"]["status"] != "passed":
            raise ValueError(f"M2.1 parity did not pass: {dev_name}")
        encoding = dev["category_encoding"]["pooled"]
        category_path = _require_bound_file(
            encoding, "path", "bytes", "sha256", root
        )
        category_maps[dev_name] = {
            "path": str(category_path),
            "bytes": category_path.stat().st_size,
            "sha256": _sha256(category_path),
        }
        baseline_models[dev_name] = {}
        for alias, source_name in BASELINE_ALIASES.items():
            model = dev["models"][source_name]
            model_path = _require_bound_file(
                model, "model_path", "model_bytes", "model_sha256", root
            )
            baseline_models[dev_name][alias] = {
                "source_name": source_name,
                "path": str(model_path),
                "bytes": model_path.stat().st_size,
                "sha256": _sha256(model_path),
            }
        reference = Path(dev["m1_parity"]["reference_path"]).resolve()
        if not reference.is_file():
            raise FileNotFoundError(reference)
        if _sha256(reference) != dev["m1_parity"]["reference_sha256"]:
            raise ValueError(f"M1 reference SHA256 mismatch: {dev_name}")
        references[dev_name] = str(reference)
    evidence = {
        "source_metrics": str(m21_metrics_path.resolve()),
        "source_metrics_bytes": m21_metrics_path.stat().st_size,
        "source_metrics_sha256": _sha256(m21_metrics_path),
        "source_manifest": str(manifest_path),
        "source_manifest_bytes": manifest_path.stat().st_size,
        "source_manifest_sha256": _sha256(manifest_path),
        "dataset_paths": dataset_paths,
        "category_maps": category_maps,
        "baseline_models": baseline_models,
        "m1_references": references,
    }
    return metrics, evidence


def _normalize_baselines(source_orderings: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for alias, source in BASELINE_ALIASES.items():
        normalized[alias] = copy.deepcopy(source_orderings[source])
        normalized[f"{alias}__inactive_rrf"] = copy.deepcopy(
            source_orderings[f"{source}__inactive_rrf"]
        )
    return normalized


def _overall_map(
    development: dict[str, Any], window: str, ordering: str, metric_k: int
) -> float:
    return float(
        development[window]["evaluation"]["orderings"][ordering]["segments"][
            "overall"
        ][f"map@{metric_k}"]
    )


def analyze_groups(
    development: dict[str, Any], metric_k: int, min_mean_gain: float
) -> dict[str, Any]:
    windows = list(DEVELOPMENT_PROTOCOL)
    groups: dict[str, Any] = {}
    for group in FEATURE_GROUPS:
        add_name = f"add_{group}"
        drop_name = f"drop_{group}"
        add_delta = {
            window: _overall_map(development, window, add_name, metric_k)
            - _overall_map(development, window, "retrieval", metric_k)
            for window in windows
        }
        drop_delta = {
            window: _overall_map(development, window, drop_name, metric_k)
            - _overall_map(development, window, "full", metric_k)
            for window in windows
        }
        if all(delta > 0 for delta in add_delta.values()) and all(
            delta < 0 for delta in drop_delta.values()
        ):
            classification = "stable_helpful"
        elif all(delta < 0 for delta in add_delta.values()) and all(
            delta > 0 for delta in drop_delta.values()
        ):
            classification = "stable_harmful"
        else:
            classification = "interaction_or_unstable"
        groups[group] = {
            "add_vs_retrieval_map@12": add_delta,
            "drop_vs_full_map@12": drop_delta,
            "mean_add_vs_retrieval_map@12": float(np.mean(list(add_delta.values()))),
            "mean_drop_vs_full_map@12": float(np.mean(list(drop_delta.values()))),
            "classification": classification,
        }
    baseline = "full__inactive_rrf"
    candidate_rows: dict[str, Any] = {}
    for group in FEATURE_GROUPS:
        name = f"drop_{group}__inactive_rrf"
        deltas = {
            window: _overall_map(development, window, name, metric_k)
            - _overall_map(development, window, baseline, metric_k)
            for window in windows
        }
        mean_delta = float(np.mean(list(deltas.values())))
        candidate_rows[name] = {
            "window_delta_map@12": deltas,
            "mean_delta_map@12": mean_delta,
            "passes": all(delta > 0 for delta in deltas.values())
            and mean_delta >= min_mean_gain,
        }
    passed = [name for name, row in candidate_rows.items() if row["passes"]]
    selected = baseline
    if passed:
        selected = max(
            passed,
            key=lambda name: (
                np.mean(
                    [
                        _overall_map(development, window, name, metric_k)
                        for window in windows
                    ]
                ),
                min(
                    _overall_map(development, window, name, metric_k)
                    for window in windows
                ),
                name,
            ),
        )
    return {
        "groups": groups,
        "selection_candidates": candidate_rows,
        "baseline": baseline,
        "selected_development_candidate": selected,
        "minimum_mean_gain": min_mean_gain,
        "selection_rule": (
            "drop-group fallback must beat full fallback in both windows and "
            "mean gain must meet minimum"
        ),
        "final_confirmation_run": "not_run",
    }


def _ordering_summary(
    development: dict[str, Any], ordering_names: list[str], metric_k: int
) -> dict[str, Any]:
    windows = list(DEVELOPMENT_PROTOCOL)
    summary: dict[str, Any] = {}
    for name in ordering_names:
        values = {
            window: _overall_map(development, window, name, metric_k)
            for window in windows
        }
        summary[name] = {
            "window_map@12": values,
            "mean_map@12": float(np.mean(list(values.values()))),
            "min_map@12": float(np.min(list(values.values()))),
        }
    return summary


def run_m22(
    m21_artifact_dir: Path,
    m21_metrics_path: Path,
    transactions_path: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
    min_mean_gain: float = 0.0002,
) -> dict[str, Any]:
    config.validate()
    if config.evaluation_role != "development":
        raise ValueError("M2.2 is development-only")
    if min_mean_gain < 0:
        raise ValueError("min_mean_gain must be non-negative")
    for directory in (output_dir, artifact_dir):
        if directory.exists():
            raise FileExistsError(directory)
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    source_metrics, reuse = validate_m21_reuse(
        m21_artifact_dir, m21_metrics_path
    )
    feature_variants = build_ablation_features()
    train_cutoffs = sorted(
        {
            cutoff
            for protocol in DEVELOPMENT_PROTOCOL.values()
            for cutoff in protocol["train"]
        }
    )
    frame_cache = {
        cutoff: _load_window_frame(Path(reuse["dataset_paths"][cutoff]), cutoff)
        for cutoff in train_cutoffs
    }
    development: dict[str, Any] = {}
    for dev_name, protocol in DEVELOPMENT_PROTOCOL.items():
        dev_dir = artifact_dir / dev_name
        dev_dir.mkdir()
        category_evidence = reuse["category_maps"][dev_name]
        category_maps = _load_category_maps(Path(category_evidence["path"]))
        train_frame = pd.concat(
            [frame_cache[cutoff] for cutoff in protocol["train"]],
            ignore_index=True,
        )
        variants: dict[
            str, tuple[Any, list[str], dict[str, dict[int, int]]]
        ] = {}
        model_evidence: dict[str, Any] = {}
        for name, features in feature_variants.items():
            model, evidence = _train_variant(
                train_frame,
                features,
                "lambdarank",
                name,
                dev_dir,
                config,
                category_maps,
            )
            variants[name] = (model, features, category_maps)
            model_evidence[name] = evidence
        del train_frame
        validation_cutoff = protocol["validation"]
        scoring = _score_variants(
            Path(reuse["dataset_paths"][validation_cutoff]),
            variants,
            dev_dir / "evaluation.duckdb",
            dev_dir / "validation-predictions.parquet",
            config,
        )
        del variants
        evaluation = _evaluate_variants(
            dev_dir / "evaluation.duckdb",
            transactions_path,
            validation_cutoff,
            config.metric_k,
            list(feature_variants),
        )
        reference = Path(reuse["m1_references"][dev_name])
        parity = _reference_parity(evaluation, reference)
        source_orderings = source_metrics["development"][dev_name]["evaluation"][
            "orderings"
        ]
        source_rrf = float(
            source_orderings["rrf"]["segments"]["overall"][
                f"map@{config.metric_k}"
            ]
        )
        current_rrf = float(
            evaluation["orderings"]["rrf"]["segments"]["overall"][
                f"map@{config.metric_k}"
            ]
        )
        reuse_rrf_difference = abs(source_rrf - current_rrf)
        if reuse_rrf_difference > 1e-12:
            raise RuntimeError(
                f"M2.1 reused RRF parity failed: {dev_name} {reuse_rrf_difference}"
            )
        evaluation["orderings"].update(_normalize_baselines(source_orderings))
        development[dev_name] = {
            "train_cutoffs": protocol["train"],
            "validation_cutoff": validation_cutoff,
            "models": model_evidence,
            "scoring": scoring,
            "evaluation": evaluation,
            "m1_parity": parity,
            "reuse_parity": {
                "status": "passed",
                "rrf_map@12_absolute_difference": reuse_rrf_difference,
                "baseline_training": "not_repeated_reused_m2.1_measured_evidence",
            },
        }
    del frame_cache
    analysis = analyze_groups(development, config.metric_k, min_mean_gain)
    ordering_names = ["retrieval", "full"]
    ordering_names.extend(feature_variants)
    ordering_names.extend(
        f"{name}__inactive_rrf"
        for name in ["retrieval", "full"] + list(feature_variants)
    )
    summary = _ordering_summary(development, ordering_names, config.metric_k)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "minimum_mean_gain": min_mean_gain,
        "source_reuse": reuse,
        "development_protocol": DEVELOPMENT_PROTOCOL,
        "feature_groups": FEATURE_GROUPS,
        "feature_variants": feature_variants,
        "baseline_training": "not_repeated_reused_m2.1_measured_evidence",
        "final_confirmation_run": "not_run",
    }
    manifest_path = artifact_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "M2.2",
        "status": "measured",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "development_only_no_final_week",
        "config": asdict(config),
        "minimum_mean_gain": min_mean_gain,
        "source_reuse": reuse,
        "development_protocol": DEVELOPMENT_PROTOCOL,
        "feature_groups": FEATURE_GROUPS,
        "feature_variants": feature_variants,
        "development": development,
        "analysis": analysis,
        "ordering_summary": summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "M2_2_REPORT.md"
    result["artifacts"] = {
        "metrics": str(metrics_path.resolve()),
        "report": str(report_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
    }
    _write_json(metrics_path, result)
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    analysis = result["analysis"]
    summary = result["ordering_summary"]
    selected = analysis["selected_development_candidate"]
    baseline = analysis["baseline"]
    lines = [
        "# M2.2 feature-group ablation",
        "",
        "## 结论",
        "",
        f"- development candidate：{selected}。",
        f"- selection baseline：{baseline}；minimum mean gain = "
        f"{analysis['minimum_mean_gain']:.6f}。",
        "- 最终周未运行；full/retrieval baseline 未重训，复用前完成 manifest/SHA256 校验。",
        "",
        "## Feature-group diagnosis（pure pooled LambdaRank）",
        "",
        "| group | add dev-A | add dev-B | add mean | drop dev-A | drop dev-B | drop mean | classification |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for group, row in analysis["groups"].items():
        add = row["add_vs_retrieval_map@12"]
        drop = row["drop_vs_full_map@12"]
        lines.append(
            f"| {group} | {add['dev_a']:+.6f} | {add['dev_b']:+.6f} | "
            f"{row['mean_add_vs_retrieval_map@12']:+.6f} | "
            f"{drop['dev_a']:+.6f} | {drop['dev_b']:+.6f} | "
            f"{row['mean_drop_vs_full_map@12']:+.6f} | "
            f"{row['classification']} |"
        )
    lines.extend(
        [
            "",
            "add = retrieval + group minus retrieval；drop = full - group minus full。",
            "",
            "## Freeze gate（with exact inactive RRF fallback）",
            "",
            "| candidate | dev-A delta | dev-B delta | mean delta | passes |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, row in analysis["selection_candidates"].items():
        delta = row["window_delta_map@12"]
        lines.append(
            f"| {name} | {delta['dev_a']:+.6f} | {delta['dev_b']:+.6f} | "
            f"{row['mean_delta_map@12']:+.6f} | "
            f"{'yes' if row['passes'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Selected ordering",
            "",
            "| ordering | dev-A MAP@12 | dev-B MAP@12 | mean | worst |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    rendered_orderings: set[str] = set()
    for name in ("retrieval__inactive_rrf", baseline, selected):
        if name not in summary or name in rendered_orderings:
            continue
        rendered_orderings.add(name)
        row = summary[name]
        values = row["window_map@12"]
        lines.append(
            f"| {name} | {values['dev_a']:.6f} | {values['dev_b']:.6f} | "
            f"{row['mean_map@12']:.6f} | {row['min_map@12']:.6f} |"
        )
    lines.extend(["", "## Runtime and supervision", ""])
    for dev_name, dev in result["development"].items():
        model_seconds = sum(
            float(model["elapsed_seconds"]) for model in dev["models"].values()
        )
        example_model = next(iter(dev["models"].values()))
        groups = example_model["group_evidence"]
        lines.append(
            f"- {dev_name}：10 new models train total {model_seconds:.2f}s；"
            f"scoring {dev['scoring']['elapsed_seconds']:.2f}s；"
            f"groups {groups['groups']:,}，positive {groups['positive_groups']:,}，"
            f"zero-positive {groups['zero_positive_groups']:,}。"
        )
    lines.extend(
        [
            "",
            "## Evidence boundaries",
            "",
            "- add/drop 使用相同 pooled two-week labels、LambdaRank、200 rounds、"
            "Top100 candidates 和训练期 category maps。",
            "- group diagnosis 使用 pure model，freeze gate 使用带 exact inactive fallback 的 ordering。",
            "- 只允许单组 drop 替换 full；没有组合多个 drop，也没有逐特征贪心搜索。",
            "- M2.1 feature parquets、category maps、baseline model files、metrics 和 manifest"
            " 均做 path/bytes/SHA256 fail-closed 校验。",
            "- 两个验证窗口的 M1 parity 与复用 RRF parity 必须通过。",
            "- retrieval-v1 profile 已使用这些早期窗口，因此这是 ranker development，"
            "不是整条 pipeline 的 untouched evaluation。",
            "- optimistic catalog 假设仍然成立；缺少曝光日志，target=0 仍不是证明负反馈。",
            "- cold candidate coverage 不会因 ranking feature ablation 改善。",
            "- final confirmation、M3 和图片/文本 embedding 均未运行。",
            "",
            "## Artifacts",
            "",
            f"- metrics：{result['artifacts']['metrics']}",
            f"- private manifest：{result['artifacts']['manifest']}",
            f"- models/predictions：{result['artifacts']['artifact_dir']}（Git ignored）",
            "",
        ]
    )
    return "\n".join(lines)
