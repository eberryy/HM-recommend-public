from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import duckdb
import numpy as np
import pandas as pd

from .m2 import (
    CATEGORICAL_FEATURES,
    M2Config,
    _literal,
    _prepare_frame,
    _sha256,
    _write_json,
    build_category_maps,
)
from .m26 import PROTOCOL, _evaluate_models, _frozen_reference, _save_category_maps
from .m29 import _attach_popularity_segments, _read_json
from .m210 import (
    ALL_FEATURES,
    M29_FEATURES,
    TARGET_AWARE_FEATURES,
    _file_identity,
    _target_feature_paths,
    build_target_aware_cache,
    score_models,
)
from .m211 import (
    FEATURE_FAMILIES as M211_FEATURE_FAMILIES,
    SAMPLING_SEED,
    _load_category_maps,
    load_distribution_sample,
    train_lambdarank,
)


SCHEMA_VERSION = "m2.12-inner-temporal-selection-v1"
EARLY_STOPPING_ROUNDS = 20

INNER_PROTOCOL = {
    "dev_a": {
        "inner_train": ["2020-05-27"],
        "inner_validation": "2020-06-24",
        "outer_train": ["2020-05-27", "2020-06-24"],
        "outer_validation": "2020-07-22",
    },
    "dev_b": {
        "inner_train": ["2020-06-24"],
        "inner_validation": "2020-07-22",
        "outer_train": ["2020-06-24", "2020-07-22"],
        "outer_validation": "2020-08-19",
    },
}

COUNT_FEATURES = [
    "user_item_events_28d",
    "user_product_code_events_28d",
    "user_product_code_events_12w",
    "user_product_type_events_28d",
    "user_department_events_28d",
    "user_garment_events_28d",
    "user_garment_events_12w",
    "user_colour_events_28d",
    "user_colour_events_12w",
    "user_index_group_events_28d",
    "user_index_group_events_12w",
]
RECENCY_FEATURES = [
    *M211_FEATURE_FAMILIES["product_code_recency"],
    *M211_FEATURE_FAMILIES["taxonomy_recency"],
]
SHARE_FEATURES = list(M211_FEATURE_FAMILIES["share"])
DECAY_FEATURES = list(M211_FEATURE_FAMILIES["decay"])
REMAINING_FAMILIES = {
    "count": COUNT_FEATURES,
    "recency": RECENCY_FEATURES,
    "share": SHARE_FEATURES,
}


def feature_sets() -> dict[str, list[str]]:
    no_decay = [feature for feature in ALL_FEATURES if feature not in DECAY_FEATURES]
    result = {"no_decay_anchor": no_decay}
    for family, removed in REMAINING_FAMILIES.items():
        result[f"no_decay_without_{family}"] = [
            feature for feature in no_decay if feature not in removed
        ]
    result["m29_only"] = list(M29_FEATURES)
    validate_feature_sets(result)
    return result


def validate_feature_sets(sets: dict[str, list[str]]) -> None:
    flattened = [feature for values in REMAINING_FAMILIES.values() for feature in values]
    expected = set(TARGET_AWARE_FEATURES) - set(DECAY_FEATURES)
    if len(flattened) != len(set(flattened)) or set(flattened) != expected:
        raise RuntimeError("M2.12 count/recency/share must partition no-decay features")
    if len(expected) != 21:
        raise RuntimeError("M2.12 expected exactly 21 target-aware no-decay features")
    if set(sets["no_decay_anchor"]) != set(M29_FEATURES) | expected:
        raise RuntimeError("M2.12 no-decay anchor differs from frozen feature contract")
    if sets["m29_only"] != list(M29_FEATURES):
        raise RuntimeError("M2.12 M2.9-only control differs from M2.9 features")


def validate_inner_protocol() -> None:
    for dev, row in INNER_PROTOCOL.items():
        outer = PROTOCOL[dev]
        if row["outer_train"] != outer["train"]:
            raise RuntimeError(f"{dev} outer train differs from frozen protocol")
        if row["outer_validation"] != outer["validation"]:
            raise RuntimeError(f"{dev} outer validation differs from frozen protocol")
        sequence = [
            *row["inner_train"],
            row["inner_validation"],
            row["outer_validation"],
        ]
        if sequence != sorted(sequence) or len(sequence) != len(set(sequence)):
            raise RuntimeError(f"{dev} inner/outer cutoffs are not strictly temporal")
        if row["outer_train"] != [*row["inner_train"], row["inner_validation"]]:
            raise RuntimeError(f"{dev} refit cutoffs do not extend inner training")


def exact_group_map_at_k(
    predictions: np.ndarray,
    labels: np.ndarray,
    group_sizes: list[int],
    truth_counts: np.ndarray,
    *,
    k: int,
) -> float:
    predictions = np.asarray(predictions, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.uint8)
    truth_counts = np.asarray(truth_counts, dtype=np.int64)
    if predictions.ndim != 1 or labels.shape != predictions.shape:
        raise ValueError("predictions and labels must be aligned one-dimensional arrays")
    if sum(group_sizes) != len(predictions) or len(group_sizes) != len(truth_counts):
        raise ValueError("group sizes or truth counts do not align with rows")
    if not group_sizes or min(group_sizes) <= 0 or (truth_counts <= 0).any():
        raise ValueError("validation groups and truth counts must be positive")
    max_size = max(group_sizes)
    groups = len(group_sizes)
    score_matrix = np.full((groups, max_size), -np.inf, dtype=np.float64)
    label_matrix = np.zeros((groups, max_size), dtype=np.uint8)
    row_groups = np.repeat(np.arange(groups, dtype=np.int64), group_sizes)
    row_positions = np.concatenate(
        [np.arange(size, dtype=np.int64) for size in group_sizes]
    )
    score_matrix[row_groups, row_positions] = predictions
    label_matrix[row_groups, row_positions] = labels
    # Rows arrive in candidate_rank/article_id order. Stable sort therefore gives the
    # same deterministic tie-break used by the outer evaluator.
    order = np.argsort(-score_matrix, axis=1, kind="stable")[:, :k]
    top_labels = np.take_along_axis(label_matrix, order, axis=1)
    cumulative = np.cumsum(top_labels, axis=1, dtype=np.float64)
    precision = cumulative / np.arange(1, top_labels.shape[1] + 1)
    numerator = np.sum(precision * top_labels, axis=1)
    denominator = np.minimum(truth_counts, k)
    return float(np.mean(numerator / denominator))


def make_exact_map_evaluator(
    labels: np.ndarray,
    group_sizes: list[int],
    truth_counts: np.ndarray,
    *,
    k: int,
) -> Callable[[np.ndarray, Any], tuple[str, float, bool]]:
    def evaluate(predictions: np.ndarray, _: Any) -> tuple[str, float, bool]:
        return (
            f"exact_map_at_{k}",
            exact_group_map_at_k(
                predictions, labels, group_sizes, truth_counts, k=k
            ),
            True,
        )

    return evaluate


def load_inner_validation(
    *,
    dataset_path: Path,
    transactions_path: Path,
    cutoff: str,
    features: list[str],
) -> tuple[pd.DataFrame, list[int], np.ndarray, dict[str, Any]]:
    columns = list(
        dict.fromkeys(
            ["customer_id", "article_id", *features, "target", "user_history_events_12w"]
        )
    )
    source = f"read_parquet({_literal(dataset_path)})"
    transactions = f"read_parquet({_literal(transactions_path)})"
    cutoff_sql = "DATE '" + cutoff.replace("'", "''") + "'"
    connection = duckdb.connect()
    started = time.perf_counter()
    try:
        query = f"""
        WITH eligible AS (
            SELECT customer_id
            FROM {source}
            GROUP BY customer_id
            HAVING sum(target)>0 AND max(user_history_events_12w)>0
        ), truth AS (
            SELECT customer_id,count(DISTINCT article_id)::BIGINT AS truth_count
            FROM {transactions}
            WHERE t_dat>={cutoff_sql} AND t_dat<{cutoff_sql}+INTERVAL 7 DAY
            GROUP BY customer_id
        )
        SELECT {','.join('s.' + column for column in columns)},t.truth_count
        FROM {source} s
        JOIN eligible e USING(customer_id)
        JOIN truth t USING(customer_id)
        ORDER BY s.customer_id,s.candidate_rank,s.article_id
        """
        frame = connection.execute(query).fetchdf()
        source_stats = connection.execute(
            f"""
            SELECT count(*)::BIGINT,count(DISTINCT customer_id)::BIGINT,
                   sum(target)::BIGINT,
                   count(DISTINCT customer_id) FILTER(WHERE target=1)::BIGINT
            FROM {source}
            """
        ).fetchone()
    finally:
        connection.close()
    if frame.empty:
        raise RuntimeError("inner validation has no covered active groups")
    group_frame = frame.groupby("customer_id", sort=False).agg(
        rows=("target", "size"),
        candidate_hits=("target", "sum"),
        history_events=("user_history_events_12w", "max"),
        truth_count=("truth_count", "first"),
    )
    sizes = group_frame["rows"].astype(int).tolist()
    truth_counts = group_frame["truth_count"].to_numpy(dtype=np.int64)
    if min(sizes) < 100 or max(sizes) > 300:
        raise RuntimeError("inner validation groups violate expanded-300 contract")
    if (group_frame["candidate_hits"] <= 0).any():
        raise RuntimeError("inner validation retained candidate-miss groups")
    if (group_frame["history_events"] <= 0).any():
        raise RuntimeError("inner validation retained inactive groups")
    if (group_frame["candidate_hits"] > group_frame["truth_count"]).any():
        raise RuntimeError("candidate positives exceed distinct temporal truth")
    if int(sum(sizes)) != len(frame):
        raise RuntimeError("inner validation row/group mismatch")
    evidence = {
        "role": "active_candidate_covered_round_selection_only",
        "cutoff": cutoff,
        "source_rows": int(source_stats[0]),
        "source_groups": int(source_stats[1]),
        "source_candidate_hits": int(source_stats[2]),
        "source_covered_groups": int(source_stats[3]),
        "selected_rows": len(frame),
        "selected_groups": len(sizes),
        "selected_candidate_hits": int(group_frame["candidate_hits"].sum()),
        "selected_truth_pairs": int(group_frame["truth_count"].sum()),
        "min_group_rows": min(sizes),
        "max_group_rows": max(sizes),
        "metric_denominator": "complete distinct validation-week truth per customer",
        "excluded_terms": (
            "inactive fixed-RRF contribution and candidate-miss zero contribution; "
            "both are constant with respect to model rounds"
        ),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return frame, sizes, truth_counts, evidence


def train_inner_ranker(
    *,
    train_frame: pd.DataFrame,
    train_group_sizes: list[int],
    train_evidence: dict[str, Any],
    validation_frame: pd.DataFrame,
    validation_group_sizes: list[int],
    validation_truth_counts: np.ndarray,
    validation_evidence: dict[str, Any],
    features: list[str],
    name: str,
    artifact_dir: Path,
    config: M2Config,
    category_maps: dict[str, dict[int, int]],
) -> tuple[Any, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required; install dependency.txt") from error
    started = time.perf_counter()
    train_prepared = _prepare_frame(train_frame, features, category_maps)
    validation_prepared = _prepare_frame(validation_frame, features, category_maps)
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    train_dataset = lgb.Dataset(
        train_prepared,
        label=train_frame["target"].astype(np.uint8),
        group=train_group_sizes,
        feature_name=features,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    validation_labels = validation_frame["target"].to_numpy(dtype=np.uint8)
    validation_dataset = lgb.Dataset(
        validation_prepared,
        label=validation_labels,
        group=validation_group_sizes,
        feature_name=features,
        categorical_feature=categorical,
        reference=train_dataset,
        free_raw_data=True,
    )
    params = {
        "objective": "lambdarank",
        "metric": "None",
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
        "lambdarank_truncation_level": 20,
    }
    evaluations: dict[str, Any] = {}
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=config.num_boost_round,
        valid_sets=[validation_dataset],
        valid_names=["inner_validation"],
        feval=make_exact_map_evaluator(
            validation_labels,
            validation_group_sizes,
            validation_truth_counts,
            k=config.metric_k,
        ),
        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING_ROUNDS,
                first_metric_only=True,
                verbose=False,
            ),
            lgb.record_evaluation(evaluations),
        ],
    )
    metric_name = f"exact_map_at_{config.metric_k}"
    curve = evaluations["inner_validation"][metric_name]
    best_iteration = int(model.best_iteration or len(curve))
    if not 1 <= best_iteration <= config.num_boost_round:
        raise RuntimeError("inner best iteration is outside configured round budget")
    best_score = float(curve[best_iteration - 1])
    model_path = artifact_dir / f"lightgbm-inner-{name}.txt"
    model.save_model(str(model_path), num_iteration=best_iteration)
    evidence = {
        "name": name,
        "features": features,
        "parameters": params,
        "max_boost_round": config.num_boost_round,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": best_iteration,
        "best_score": best_score,
        "stopped_before_max": best_iteration < config.num_boost_round,
        "metric": metric_name,
        "metric_curve": [float(value) for value in curve],
        "train_sampling": train_evidence,
        "validation": validation_evidence,
        "model_path": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    del train_dataset, validation_dataset, train_prepared, validation_prepared
    gc.collect()
    return model, evidence


def select_inner_variant(inner_models: dict[str, dict[str, Any]]) -> str:
    if not inner_models:
        raise ValueError("inner model evidence is empty")
    return max(
        inner_models,
        key=lambda name: (
            float(inner_models[name]["best_score"]),
            -int(inner_models[name]["best_iteration"]),
            name,
        ),
    )


def _ordering_values(
    development: dict[str, Any], name: str, metric: str
) -> dict[str, float]:
    fallback = f"{name}__inactive_rrf"
    return {
        dev: float(
            row["evaluation"]["orderings"][fallback]["segments"]["overall"][metric]
        )
        for dev, row in development.items()
    }


def summarize(
    development: dict[str, Any],
    frozen: dict[str, Any],
    m211_metrics: dict[str, Any],
    metric_k: int,
) -> dict[str, Any]:
    metric = f"map@{metric_k}"
    names = list(feature_sets())
    frozen_values = {
        dev: float(frozen[dev]["segments"]["overall"][metric]) for dev in PROTOCOL
    }
    frozen_mean = float(np.mean(list(frozen_values.values())))
    orderings = {}
    for name in names:
        values = _ordering_values(development, name, metric)
        orderings[f"{name}__inactive_rrf"] = {
            "window_map@12": values,
            "mean_map@12": float(np.mean(list(values.values()))),
            "min_map@12": float(np.min(list(values.values()))),
        }
    selected_names = {
        dev: row["inner_selection"]["selected_variant"]
        for dev, row in development.items()
    }
    selected_values = {
        dev: orderings[f"{selected_names[dev]}__inactive_rrf"]["window_map@12"][dev]
        for dev in PROTOCOL
    }
    selected_mean = float(np.mean(list(selected_values.values())))
    selected_delta = {
        dev: selected_values[dev] - frozen_values[dev] for dev in PROTOCOL
    }
    anchor = orderings["no_decay_anchor__inactive_rrf"]
    anchor_delta = {
        dev: anchor["window_map@12"][dev] - frozen_values[dev] for dev in PROTOCOL
    }
    family_diagnostics = {}
    for family in REMAINING_FAMILIES:
        name = f"no_decay_without_{family}__inactive_rrf"
        values = orderings[name]["window_map@12"]
        delta = {
            dev: values[dev] - anchor["window_map@12"][dev] for dev in PROTOCOL
        }
        if all(value > 0 for value in delta.values()):
            classification = "removed_family_improves_both_outer_windows"
        elif all(value < 0 for value in delta.values()):
            classification = "removed_family_hurts_both_outer_windows"
        else:
            classification = "cross_window_unstable_or_tied"
        family_diagnostics[family] = {
            "removed_features": REMAINING_FAMILIES[family],
            "window_delta_vs_no_decay_anchor_map@12": delta,
            "mean_delta_vs_no_decay_anchor_map@12": float(np.mean(list(delta.values()))),
            "classification": classification,
            "outer_only_diagnostic_not_selection_rule": True,
        }
    previous = m211_metrics["development_summary"]["orderings"][
        "ablation_without_decay__inactive_rrf"
    ]
    candidate_ceiling = {}
    for dev, row in development.items():
        overall = row["evaluation"]["orderings"][
            "no_decay_anchor__inactive_rrf"
        ]["segments"]["overall"]
        candidate_ceiling[dev] = {
            "candidate_recall@expanded_pool": float(
                overall["candidate_recall@expanded_pool"]
            ),
            "oracle_map@12": float(overall[f"oracle_map@{metric_k}"]),
        }
    best_outer = max(
        orderings,
        key=lambda name: (
            orderings[name]["mean_map@12"],
            orderings[name]["min_map@12"],
            name,
        ),
    )
    return {
        "orderings": orderings,
        "frozen_m2_2": {
            "window_map@12": frozen_values,
            "mean_map@12": frozen_mean,
        },
        "m2_11_remove_decay_200_rounds": previous,
        "inner_selected_policy": {
            "selected_variant_by_window": selected_names,
            "same_variant_both_windows": len(set(selected_names.values())) == 1,
            "window_map@12": selected_values,
            "mean_map@12": selected_mean,
            "window_delta_vs_frozen_map@12": selected_delta,
            "mean_delta_vs_frozen_map@12": selected_mean - frozen_mean,
            "accepted": (
                all(value >= 0 for value in selected_delta.values())
                and selected_mean > frozen_mean
            ),
            "rule": (
                "feature set and boosting round selected only on the earlier inner "
                "window; outer mean improves over frozen M2.2 and neither outer "
                "window regresses"
            ),
        },
        "no_decay_anchor_gate": {
            "window_delta_vs_frozen_map@12": anchor_delta,
            "mean_delta_vs_frozen_map@12": anchor["mean_map@12"] - frozen_mean,
            "accepted": (
                all(value >= 0 for value in anchor_delta.values())
                and anchor["mean_map@12"] > frozen_mean
            ),
        },
        "conditional_family_diagnostics": family_diagnostics,
        "best_outer_diagnostic_not_selected": best_outer,
        "candidate_ceiling": candidate_ceiling,
        "final_week": "not_run",
        "evidence_boundary": (
            "outer dev windows were observed in earlier stages; this is rolling "
            "development evidence, not a pristine blind test"
        ),
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["development_summary"]
    policy = summary["inner_selected_policy"]
    frozen = summary["frozen_m2_2"]
    lines = [
        "# M2.12 inner temporal validation and no-decay conditional ablation",
        "",
        "## 结论",
        "",
        f"- inner-only selection policy gate：{'通过' if policy['accepted'] else '未通过'}；"
        f"mean delta vs frozen M2.2={policy['mean_delta_vs_frozen_map@12']:+.6f}。",
        f"- 两窗口是否选中同一特征集：{policy['same_variant_both_windows']}；"
        f"选择={policy['selected_variant_by_window']}。",
        "- outer windows 在旧阶段已经被观察；本结果不是 pristine blind test，final week 未运行。",
        "",
        "## MAP@12（完整 expanded pool + exact inactive fallback）",
        "",
        "| ordering | dev-A | dev-B | mean | worst |",
        "|---|---:|---:|---:|---:|",
        f"| frozen M2.2 | {frozen['window_map@12']['dev_a']:.6f} | "
        f"{frozen['window_map@12']['dev_b']:.6f} | {frozen['mean_map@12']:.6f} | "
        f"{min(frozen['window_map@12'].values()):.6f} |",
    ]
    for name, row in summary["orderings"].items():
        lines.append(
            f"| {name} | {row['window_map@12']['dev_a']:.6f} | "
            f"{row['window_map@12']['dev_b']:.6f} | {row['mean_map@12']:.6f} | "
            f"{row['min_map@12']:.6f} |"
        )
    lines.extend(["", "## Inner selection", ""])
    lines.append(
        "- Inner MAP 只在 active + candidate-covered groups 上用于选择，数值尺度不能与 "
        "outer overall MAP 直接比较。"
    )
    for dev, row in result["development"].items():
        selection = row["inner_selection"]
        lines.append(
            f"- {dev}：{selection['inner_train_cutoffs']} -> "
            f"{selection['inner_validation_cutoff']}；selected="
            f"{selection['selected_variant']}，rounds={selection['selected_rounds']}，"
            f"inner MAP@12={selection['selected_score']:.6f}。"
        )
        for name, model in row["inner_models"].items():
            lines.append(
                f"  - {name}: best round {model['best_iteration']}, "
                f"MAP@12 {model['best_score']:.6f}, "
                f"stopped={model['stopped_before_max']}"
            )
    lines.extend(["", "## Conditional family diagnostics", ""])
    for family, row in summary["conditional_family_diagnostics"].items():
        delta = row["window_delta_vs_no_decay_anchor_map@12"]
        lines.append(
            f"- remove {family}: dev-A {delta['dev_a']:+.6f}, "
            f"dev-B {delta['dev_b']:+.6f}, mean "
            f"{row['mean_delta_vs_no_decay_anchor_map@12']:+.6f}; "
            f"{row['classification']}。"
        )
    lines.extend(
        [
            "",
            "## 审计边界",
            "",
            "- inner train、inner validation、outer refit、outer validation 严格按时间递增。",
            "- inner MAP 分母为完整 validation-week distinct truth count；未把召回缺失 truth 删除。",
            "- inactive fixed fallback 与 candidate-miss zero groups 对轮数恒定，只从 stopping metric 排除。",
            "- outer validation 始终完整 score 100--300 候选；Recall/Oracle 不应随 ranker 改变。",
            "- outer variant 表仅作诊断，不用于同轮继续组合删特征；final week 未运行。",
            "",
            "## 产物",
            "",
            f"- metrics：{result['artifacts']['metrics']}",
            f"- private artifacts：{result['artifacts']['artifact_dir']}（Git ignored）",
            f"- feature cache：{result['artifacts']['feature_cache_dir']}（Git ignored）",
            "",
        ]
    )
    return "\n".join(lines)


def run_m212(
    *,
    raw_dir: Path,
    transactions_path: Path,
    m21_metrics_path: Path,
    m210_metrics_path: Path,
    m211_metrics_path: Path,
    m29_cache_dir: Path,
    feature_cache_dir: Path,
    m210_artifact_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    config.validate()
    validate_inner_protocol()
    sets = feature_sets()
    if config.evaluation_role != "development" or config.candidate_k != 300:
        raise ValueError("M2.12 requires development-only candidate_k=300")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    m210_metrics = _read_json(m210_metrics_path.resolve())
    m211_metrics = _read_json(m211_metrics_path.resolve())
    if m210_metrics.get("schema_version") != "m2.10-target-aware-ranking-protocol-v1":
        raise ValueError("M2.12 requires measured M2.10 cache evidence")
    if m211_metrics.get("schema_version") != "m2.11-distribution-sampling-ablation-v1":
        raise ValueError("M2.12 requires measured M2.11 evidence")
    if m210_metrics.get("status") != "measured" or m211_metrics.get("status") != "measured":
        raise ValueError("M2.10/M2.11 evidence must be measured")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        print("M2.12: validating reusable target-aware feature cache", flush=True)
        feature_cache = build_target_aware_cache(
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            m29_cache_dir=m29_cache_dir,
            cache_dir=feature_cache_dir,
        )
        m21 = _read_json(m21_metrics_path.resolve())
        frozen_name, frozen = _frozen_reference(m21)
        development: dict[str, Any] = {}
        for dev, protocol in INNER_PROTOCOL.items():
            print(f"M2.12 {dev}: loading inner temporal data", flush=True)
            dev_dir = artifact_dir / dev
            dev_dir.mkdir()
            inner_train_paths = [
                _target_feature_paths(feature_cache_dir, cutoff)[0]
                for cutoff in protocol["inner_train"]
            ]
            inner_train, inner_sizes, inner_sampling = load_distribution_sample(
                inner_train_paths, sets["no_decay_anchor"], seed=SAMPLING_SEED
            )
            inner_maps = build_category_maps(inner_train)
            inner_map_evidence = _save_category_maps(
                dev_dir / "inner-category-maps.json", inner_maps
            )
            inner_validation, inner_validation_sizes, truth_counts, validation_evidence = (
                load_inner_validation(
                    dataset_path=_target_feature_paths(
                        feature_cache_dir, protocol["inner_validation"]
                    )[0],
                    transactions_path=transactions_path,
                    cutoff=protocol["inner_validation"],
                    features=sets["no_decay_anchor"],
                )
            )
            inner_models: dict[str, dict[str, Any]] = {}
            for name, features in sets.items():
                print(f"M2.12 {dev}: inner fit {name}", flush=True)
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
                    artifact_dir=dev_dir,
                    config=config,
                    category_maps=inner_maps,
                )
                inner_models[name] = evidence
            selected = select_inner_variant(inner_models)
            selected_rounds = int(inner_models[selected]["best_iteration"])
            del inner_train, inner_sizes, inner_validation, inner_validation_sizes
            gc.collect()

            print(f"M2.12 {dev}: refitting outer models", flush=True)
            outer_train_paths = [
                _target_feature_paths(feature_cache_dir, cutoff)[0]
                for cutoff in protocol["outer_train"]
            ]
            outer_frame, outer_sizes, outer_sampling = load_distribution_sample(
                outer_train_paths, sets["no_decay_anchor"], seed=SAMPLING_SEED
            )
            source_maps_path = m210_artifact_dir / dev / "category-maps.json"
            outer_maps = _load_category_maps(source_maps_path)
            outer_map_evidence = _save_category_maps(
                dev_dir / "outer-category-maps.json", outer_maps
            )
            outer_map_evidence["reused_from"] = _file_identity(source_maps_path)
            models: dict[str, tuple[Any, list[str], dict[str, dict[int, int]]]] = {}
            outer_models: dict[str, Any] = {}
            for name, features in sets.items():
                rounds = int(inner_models[name]["best_iteration"])
                print(f"M2.12 {dev}: outer fit {name} rounds={rounds}", flush=True)
                model, evidence = train_lambdarank(
                    frame=outer_frame,
                    group_sizes=outer_sizes,
                    group_evidence=outer_sampling,
                    features=features,
                    name=name,
                    use_ipw=False,
                    artifact_dir=dev_dir,
                    config=replace(config, num_boost_round=rounds),
                    category_maps=outer_maps,
                )
                evidence["selected_rounds_from_inner"] = rounds
                evidence["inner_selection_score"] = inner_models[name]["best_score"]
                models[name] = (model, features, outer_maps)
                outer_models[name] = evidence
            del outer_frame, outer_sizes
            gc.collect()
            print(f"M2.12 {dev}: scoring complete outer expanded pool", flush=True)
            scoring = score_models(
                dataset_path=_target_feature_paths(
                    feature_cache_dir, protocol["outer_validation"]
                )[0],
                models=models,
                evaluation_db=dev_dir / "evaluation.duckdb",
                prediction_path=dev_dir / "outer-validation-predictions.parquet",
                config=config,
            )
            variant_names = list(models)
            del models
            gc.collect()
            evaluation = _evaluate_models(
                evaluation_db=dev_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                metric_k=config.metric_k,
                variant_names=variant_names,
            )
            evaluation = _attach_popularity_segments(
                evaluation=evaluation,
                evaluation_db=dev_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=protocol["outer_validation"],
                variant_names=variant_names,
                budget=config.candidate_k,
            )
            development[dev] = {
                "inner_selection": {
                    "inner_train_cutoffs": protocol["inner_train"],
                    "inner_validation_cutoff": protocol["inner_validation"],
                    "selected_variant": selected,
                    "selected_rounds": selected_rounds,
                    "selected_score": inner_models[selected]["best_score"],
                    "tie_break": "higher exact MAP, then fewer rounds, then name",
                },
                "inner_models": inner_models,
                "inner_category_encoding": inner_map_evidence,
                "outer_train_cutoffs": protocol["outer_train"],
                "outer_validation_cutoff": protocol["outer_validation"],
                "outer_sampling": outer_sampling,
                "outer_models": outer_models,
                "outer_category_encoding": outer_map_evidence,
                "scoring": scoring,
                "evaluation": evaluation,
            }
        summary = summarize(development, frozen, m211_metrics, config.metric_k)
        result = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M2.12",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "rolling_development_only_no_final_week",
            "config": asdict(config),
            "contract": {
                "candidate_pool": "exact M2.9 six-source Top100 plus up to 200 Item2Vec-only",
                "sampling": "M2.11 unweighted source-rank-bucket hash 30x",
                "inner_protocol": INNER_PROTOCOL,
                "feature_families": REMAINING_FAMILIES,
                "removed_decay_features": DECAY_FEATURES,
                "feature_sets": sets,
                "max_boost_round": config.num_boost_round,
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                "inner_metric": "exact MAP@12 on active candidate-covered groups with full truth denominator",
                "outer_validation": "complete variable 100..300 pool with exact inactive RRF fallback",
                "frozen_reference": frozen_name,
                "final_week": "not_run",
            },
            "inputs": {
                "m21_metrics": _file_identity(m21_metrics_path),
                "m210_metrics": _file_identity(m210_metrics_path),
                "m211_metrics": _file_identity(m211_metrics_path),
                "transactions": _file_identity(transactions_path),
            },
            "feature_cache": feature_cache,
            "development": development,
            "development_summary": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M2_12_REPORT.md"
        result["artifacts"] = {
            "metrics": str(metrics_path.resolve()),
            "report": str(report_path.resolve()),
            "artifact_dir": str(artifact_dir.resolve()),
            "feature_cache_dir": str(feature_cache_dir.resolve()),
        }
        _write_json(metrics_path, result)
        report_path.write_text(render_report(result), encoding="utf-8")
        return result
    except Exception as error:
        _write_json(
            output_dir / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "stage": "M2.12",
                "status": "failed",
                "run_id": run_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
