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
    M2Config,
    _literal,
    _prepare_frame,
    _sha256,
    _write_json,
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


SCHEMA_VERSION = "m2.11-distribution-sampling-ablation-v1"
NEGATIVE_RATIO = 30
SAMPLING_SEED = 20260831
PRIMARY_MODEL = "distribution_30x_unweighted_full"
IPW_MODEL = "distribution_30x_ipw_full"

FEATURE_FAMILIES = {
    "product_code_recency": ["user_product_code_days_since"],
    "taxonomy_recency": [
        "user_product_type_days_since",
        "user_department_days_since",
        "user_garment_days_since",
        "user_colour_days_since",
        "user_index_group_days_since",
    ],
    "share": [
        "user_product_code_share_12w",
        "user_garment_share_12w",
        "user_colour_share_12w",
        "user_index_group_share_12w",
    ],
    "decay": [
        "user_item_decay_28d_halflife_12w",
        "user_product_code_decay_28d_halflife_12w",
        "user_product_type_decay_28d_halflife_12w",
        "user_department_decay_28d_halflife_12w",
        "user_garment_decay_28d_halflife_12w",
        "user_colour_decay_28d_halflife_12w",
        "user_index_group_decay_28d_halflife_12w",
    ],
}


def feature_sets() -> dict[str, list[str]]:
    result = {
        PRIMARY_MODEL: list(ALL_FEATURES),
        IPW_MODEL: list(ALL_FEATURES),
        "ablation_all_target_aware": list(M29_FEATURES),
    }
    for family, removed in FEATURE_FAMILIES.items():
        result[f"ablation_without_{family}"] = [
            feature for feature in ALL_FEATURES if feature not in removed
        ]
    return result


def _distribution_relation_sql(path: Path, *, seed: int) -> str:
    source = f"read_parquet({_literal(path)})"
    return f"""
    WITH positive_groups AS (
        SELECT customer_id,sum(target)::BIGINT AS positives
        FROM {source}
        GROUP BY customer_id
        HAVING positives>0
    ), positives AS (
        SELECT f.*,
               'positive'::VARCHAR AS _m211_source_layer,
               0::INTEGER AS _m211_rank_bucket,
               1::BIGINT AS _m211_stratum_rows,
               1::BIGINT AS _m211_selected_stratum_rows,
               1.0::DOUBLE AS _m211_inclusion_probability,
               1.0::DOUBLE AS _m211_inverse_probability_weight
        FROM {source} f
        JOIN positive_groups g USING(customer_id)
        WHERE f.target=1
    ), negative_source AS (
        SELECT f.*,g.positives,
               CASE WHEN f.candidate_rank<=100
                    THEN 'baseline_top100' ELSE 'item2vec_only' END
                    AS _m211_source_layer
        FROM {source} f
        JOIN positive_groups g USING(customer_id)
        WHERE f.target=0
    ), negative_bucketed AS (
        SELECT *,ntile(3) OVER(
                   PARTITION BY customer_id,_m211_source_layer
                   ORDER BY candidate_rank,article_id
               )::INTEGER AS _m211_rank_bucket
        FROM negative_source
    ), negative_strata AS (
        SELECT *,
               count(*) OVER(
                   PARTITION BY customer_id,_m211_source_layer,
                                _m211_rank_bucket
               )::BIGINT AS _m211_stratum_rows,
               row_number() OVER(
                   PARTITION BY customer_id,_m211_source_layer,
                                _m211_rank_bucket
                   ORDER BY hash(customer_id,article_id,{seed}),
                            candidate_rank,article_id
               ) AS _m211_stratum_sample_rank
        FROM negative_bucketed
    ), negative_ordered AS (
        SELECT *,row_number() OVER(
                   PARTITION BY customer_id
                   ORDER BY _m211_stratum_sample_rank,
                            _m211_source_layer,_m211_rank_bucket,
                            hash(customer_id,article_id,{seed}),
                            candidate_rank,article_id
               ) AS _m211_negative_sample_rank
        FROM negative_strata
    ), selected_pre AS (
        SELECT *
        FROM negative_ordered
        WHERE _m211_negative_sample_rank<={NEGATIVE_RATIO}*positives
    ), selected_counted AS (
        SELECT *,count(*) OVER(
                   PARTITION BY customer_id,_m211_source_layer,
                                _m211_rank_bucket
               )::BIGINT AS _m211_selected_stratum_rows
        FROM selected_pre
    ), sampled_negatives AS (
        SELECT * EXCLUDE(
                   positives,_m211_stratum_sample_rank,
                   _m211_negative_sample_rank
               ),
               _m211_selected_stratum_rows::DOUBLE/
                   _m211_stratum_rows AS _m211_inclusion_probability,
               _m211_stratum_rows::DOUBLE/
                   _m211_selected_stratum_rows
                   AS _m211_inverse_probability_weight
        FROM selected_counted
    )
    SELECT * FROM positives
    UNION ALL BY NAME
    SELECT * FROM sampled_negatives
    """


def load_distribution_sample(
    dataset_paths: list[Path],
    features: list[str],
    *,
    seed: int = SAMPLING_SEED,
) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    sizes: list[int] = []
    source_rows = 0
    strata: dict[tuple[str, int], dict[str, float]] = {}
    for path in dataset_paths:
        connection = duckdb.connect()
        try:
            relation = _distribution_relation_sql(path, seed=seed)
            helper = [
                "_m211_source_layer",
                "_m211_rank_bucket",
                "_m211_stratum_rows",
                "_m211_selected_stratum_rows",
                "_m211_inclusion_probability",
                "_m211_inverse_probability_weight",
            ]
            selected = ",".join(features + ["target"] + helper)
            frame = connection.execute(
                f"SELECT {selected} FROM ({relation}) "
                "ORDER BY customer_id,candidate_rank,article_id"
            ).fetchdf()
            group_rows = connection.execute(
                f"SELECT count(*),sum(target)::BIGINT FROM ({relation}) "
                "GROUP BY customer_id ORDER BY customer_id"
            ).fetchall()
            source_rows += int(
                connection.execute(
                    f"SELECT count(*) FROM read_parquet({_literal(path)})"
                ).fetchone()[0]
            )
            audit_rows = connection.execute(
                f"""
                SELECT _m211_source_layer,_m211_rank_bucket,
                       sum(_m211_selected_stratum_rows::DOUBLE/
                           _m211_inclusion_probability/
                           _m211_selected_stratum_rows)::BIGINT AS source_rows,
                       count(*)::BIGINT AS selected_rows,
                       min(_m211_inclusion_probability),
                       max(_m211_inclusion_probability),
                       min(candidate_rank),max(candidate_rank)
                FROM ({relation})
                WHERE target=0
                GROUP BY _m211_source_layer,_m211_rank_bucket
                ORDER BY _m211_source_layer,_m211_rank_bucket
                """
            ).fetchall()
        finally:
            connection.close()
        frames.append(frame)
        sizes.extend(int(row[0]) for row in group_rows)
        if any(int(row[1]) <= 0 for row in group_rows):
            raise RuntimeError("distribution sampling retained zero-positive group")
        for row in audit_rows:
            key = (str(row[0]), int(row[1]))
            entry = strata.setdefault(
                key,
                {
                    "source_rows": 0.0,
                    "selected_rows": 0.0,
                    "min_probability": 1.0,
                    "max_probability": 0.0,
                    "min_candidate_rank": float("inf"),
                    "max_candidate_rank": 0.0,
                },
            )
            entry["source_rows"] += float(row[2])
            entry["selected_rows"] += float(row[3])
            entry["min_probability"] = min(entry["min_probability"], float(row[4]))
            entry["max_probability"] = max(entry["max_probability"], float(row[5]))
            entry["min_candidate_rank"] = min(entry["min_candidate_rank"], float(row[6]))
            entry["max_candidate_rank"] = max(entry["max_candidate_rank"], float(row[7]))
    if not frames:
        raise ValueError("training paths are empty")
    result = pd.concat(frames, ignore_index=True)
    if sum(sizes) != len(result):
        raise RuntimeError("sampled rows differ from group sizes")
    if not sizes or min(sizes) < 2 or max(sizes) > 300:
        raise ValueError("sampled ranking group size outside 2..300")
    positives = int(result["target"].sum())
    negatives = int((result["target"] == 0).sum())
    if negatives / max(positives, 1) > NEGATIVE_RATIO:
        raise RuntimeError("distribution sampling exceeded frozen ratio")
    probabilities = result.loc[
        result["target"] == 0, "_m211_inclusion_probability"
    ].to_numpy(dtype=np.float64)
    raw_weights = result["_m211_inverse_probability_weight"].to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(probabilities).all()
        or not np.isfinite(raw_weights).all()
        or (probabilities <= 0).any()
        or (probabilities > 1).any()
        or (raw_weights < 1).any()
    ):
        raise RuntimeError("invalid sampling probability or inverse weight")
    normalized = raw_weights / raw_weights.mean()
    result["_m211_normalized_ipw"] = normalized
    stratum_evidence = []
    for (source_layer, rank_bucket), row in sorted(strata.items()):
        stratum_evidence.append(
            {
                "source_layer": source_layer,
                "rank_bucket": rank_bucket,
                "source_rows": int(row["source_rows"]),
                "selected_rows": int(row["selected_rows"]),
                "retention": row["selected_rows"] / row["source_rows"],
                "min_probability": row["min_probability"],
                "max_probability": row["max_probability"],
                "min_candidate_rank": int(row["min_candidate_rank"]),
                "max_candidate_rank": int(row["max_candidate_rank"]),
            }
        )
    evidence = {
        "sample_mode": "source_rank_bucket_hash_30x",
        "seed": seed,
        "source_rows": source_rows,
        "sampled_rows": len(result),
        "row_retention": len(result) / source_rows,
        "groups": len(sizes),
        "positive_groups": len(sizes),
        "zero_positive_groups": 0,
        "min_group_rows": min(sizes),
        "max_group_rows": max(sizes),
        "mean_group_rows": float(np.mean(sizes)),
        "positives": positives,
        "sampled_unobserved_items": negatives,
        "unobserved_per_positive": negatives / max(positives, 1),
        "negative_probability_min": float(probabilities.min()),
        "negative_probability_max": float(probabilities.max()),
        "raw_ipw_mean": float(raw_weights.mean()),
        "raw_ipw_max": float(raw_weights.max()),
        "normalized_ipw_mean": float(normalized.mean()),
        "normalized_ipw_max": float(normalized.max()),
        "strata": stratum_evidence,
    }
    return result, sizes, evidence


def _load_category_maps(path: Path) -> dict[str, dict[int, int]]:
    raw = _read_json(path)
    return {
        feature: {int(value): int(encoded) for value, encoded in mapping.items()}
        for feature, mapping in raw.items()
    }


def train_lambdarank(
    *,
    frame: pd.DataFrame,
    group_sizes: list[int],
    group_evidence: dict[str, Any],
    features: list[str],
    name: str,
    use_ipw: bool,
    artifact_dir: Path,
    config: M2Config,
    category_maps: dict[str, dict[int, int]],
) -> tuple[Any, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is required; install dependency.txt") from error
    started = time.perf_counter()
    prepared = _prepare_frame(frame, features, category_maps)
    weights = (
        frame["_m211_normalized_ipw"].to_numpy(dtype=np.float64)
        if use_ipw
        else None
    )
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    dataset = lgb.Dataset(
        prepared,
        label=frame["target"].astype(np.uint8),
        weight=weights,
        group=group_sizes,
        feature_name=features,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    params = {
        "objective": "lambdarank",
        "metric": ["map", "ndcg"],
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
        "eval_at": [config.metric_k],
        "lambdarank_truncation_level": 20,
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
    evidence = {
        "name": name,
        "features": features,
        "removed_target_aware_features": [
            feature for feature in TARGET_AWARE_FEATURES if feature not in features
        ],
        "weighting": (
            "normalized_row_level_inverse_inclusion_probability"
            if use_ipw
            else "none"
        ),
        "weighting_caveat": (
            "LightGBM row weights; sensitivity approximation, not an exact "
            "pairwise Horvitz-Thompson estimator"
            if use_ipw
            else None
        ),
        "parameters": params,
        "num_boost_round": config.num_boost_round,
        "group_evidence": group_evidence,
        "train_metrics": {
            metric: float(values[-1])
            for metric, values in evaluations.get("train", {}).items()
        },
        "top_feature_importance": importance[:40],
        "target_aware_feature_importance": [
            row for row in importance if row["feature"] in TARGET_AWARE_FEATURES
        ],
        "model_path": str(model_path.resolve()),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    del dataset, prepared
    gc.collect()
    return model, evidence


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
    m29_metrics: dict[str, Any],
    m210_metrics: dict[str, Any],
    metric_k: int,
) -> dict[str, Any]:
    metric = f"map@{metric_k}"
    names = sorted(set.intersection(*[set(row["models"]) for row in development.values()]))
    orderings = {}
    for name in names:
        values = _ordering_values(development, name, metric)
        orderings[f"{name}__inactive_rrf"] = {
            "window_map@12": values,
            "mean_map@12": float(np.mean(list(values.values()))),
            "min_map@12": float(np.min(list(values.values()))),
        }
    frozen_values = {
        dev: float(frozen[dev]["segments"]["overall"][metric]) for dev in PROTOCOL
    }
    frozen_mean = float(np.mean(list(frozen_values.values())))

    def m29_values(name: str) -> dict[str, float]:
        return {
            dev: float(
                m29_metrics["development"][dev]["evaluation"]["orderings"][name]
                ["segments"]["overall"][metric]
            )
            for dev in PROTOCOL
        }

    primary_name = f"{PRIMARY_MODEL}__inactive_rrf"
    primary = orderings[primary_name]
    primary_delta = {
        dev: primary["window_map@12"][dev] - frozen_values[dev] for dev in PROTOCOL
    }
    best_name = max(
        orderings,
        key=lambda name: (
            orderings[name]["mean_map@12"],
            orderings[name]["min_map@12"],
            name,
        ),
    )
    ablations = {}
    for name in [
        "ablation_all_target_aware",
        *[f"ablation_without_{family}" for family in FEATURE_FAMILIES],
    ]:
        values = orderings[f"{name}__inactive_rrf"]["window_map@12"]
        delta = {
            dev: values[dev] - primary["window_map@12"][dev] for dev in PROTOCOL
        }
        if all(value > 0 for value in delta.values()):
            classification = "removed_family_stably_harmful"
        elif all(value < 0 for value in delta.values()):
            classification = "removed_family_stably_useful"
        else:
            classification = "cross_window_unstable_or_tied"
        ablations[name] = {
            "removed_features": (
                TARGET_AWARE_FEATURES
                if name == "ablation_all_target_aware"
                else FEATURE_FAMILIES[name.removeprefix("ablation_without_")]
            ),
            "window_delta_vs_full_sampled_map@12": delta,
            "mean_delta_vs_full_sampled_map@12": float(np.mean(list(delta.values()))),
            "classification": classification,
        }
    candidate_ceiling = {}
    for dev, row in development.items():
        overall = row["evaluation"]["orderings"][primary_name]["segments"]["overall"]
        candidate_ceiling[dev] = {
            "candidate_recall@expanded_pool": float(
                overall["candidate_recall@expanded_pool"]
            ),
            "oracle_map@12": float(overall[f"oracle_map@{metric_k}"]),
        }
    return {
        "orderings": orderings,
        "frozen_m2_2": {
            "window_map@12": frozen_values,
            "mean_map@12": frozen_mean,
        },
        "m2_9_best_same_pool_control": {
            "ordering": "expanded_full_no_item2vec__inactive_rrf",
            "window_map@12": m29_values("expanded_full_no_item2vec__inactive_rrf"),
        },
        "m2_10_full_pool_control": m210_metrics["development_summary"]["orderings"]
        ["a_all_groups_full_lambdarank__inactive_rrf"],
        "m2_10_rank_prefix_30x_control": m210_metrics["development_summary"]
        ["orderings"]["b_stratified_30x_lambdarank__inactive_rrf"],
        "primary_gate": {
            "accepted": (
                all(value >= 0 for value in primary_delta.values())
                and primary["mean_map@12"] > frozen_mean
            ),
            "rule": (
                "pre-registered unweighted distribution-aware model improves "
                "mean MAP@12 over frozen M2.2 and neither dev window regresses"
            ),
            "selected_variant": primary_name,
            "window_delta_vs_frozen_map@12": primary_delta,
            "mean_delta_vs_frozen_map@12": primary["mean_map@12"] - frozen_mean,
        },
        "best_exploratory_variant": best_name,
        "ablations": ablations,
        "candidate_ceiling": candidate_ceiling,
        "final_week": "not_run",
    }


def render_report(result: dict[str, Any]) -> str:
    summary = result["development_summary"]
    gate = summary["primary_gate"]
    frozen = summary["frozen_m2_2"]
    lines = [
        "# M2.11 distribution-aware negative sampling and bounded ablation",
        "",
        "## 结论",
        "",
        f"- primary gate：{'通过' if gate['accepted'] else '未通过'}；预注册主模型为 "
        f"{gate['selected_variant']}。",
        f"- 相对冻结 M2.2 mean MAP@12："
        f"{gate['mean_delta_vs_frozen_map@12']:+.6f}；final week 未运行。",
        "- IPW 仅为 LightGBM row-weight sensitivity，不声明为 pairwise 无偏校正。",
        "- 特征族只做 leave-one-family-out 诊断；本轮不根据同一 dev 结果拼接新模型。",
        "",
        "## MAP@12（exact inactive fallback）",
        "",
        "| ordering | dev-A | dev-B | mean | worst |",
        "|---|---:|---:|---:|---:|",
        f"| frozen M2.2 | {frozen['window_map@12']['dev_a']:.6f} | "
        f"{frozen['window_map@12']['dev_b']:.6f} | "
        f"{frozen['mean_map@12']:.6f} | "
        f"{min(frozen['window_map@12'].values()):.6f} |",
    ]
    for name, row in summary["orderings"].items():
        lines.append(
            f"| {name} | {row['window_map@12']['dev_a']:.6f} | "
            f"{row['window_map@12']['dev_b']:.6f} | "
            f"{row['mean_map@12']:.6f} | {row['min_map@12']:.6f} |"
        )
    lines.extend(["", "## 有界特征消融", ""])
    for name, row in summary["ablations"].items():
        delta = row["window_delta_vs_full_sampled_map@12"]
        lines.append(
            f"- {name}：dev-A {delta['dev_a']:+.6f}，"
            f"dev-B {delta['dev_b']:+.6f}，mean "
            f"{row['mean_delta_vs_full_sampled_map@12']:+.6f}；"
            f"{row['classification']}。"
        )
    lines.extend(["", "## 采样审计", ""])
    for dev, row in result["development"].items():
        sample = row["sampling"]
        lines.append(
            f"### {dev}: rows={sample['sampled_rows']:,} "
            f"({sample['row_retention']:.2%})，groups={sample['groups']:,}，"
            f"unobserved/positive={sample['unobserved_per_positive']:.2f}，"
            f"raw IPW max={sample['raw_ipw_max']:.2f}。"
        )
        lines.append("")
        for stratum in sample["strata"]:
            lines.append(
                f"- {stratum['source_layer']} bucket "
                f"{stratum['rank_bucket']}："
                f"{stratum['selected_rows']:,}/{stratum['source_rows']:,}，"
                f"candidate rank {stratum['min_candidate_rank']}.."
                f"{stratum['max_candidate_rank']}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 审计边界",
            "",
            "- sampled rows 只用于训练；validation 始终完整 score 100--300 候选。",
            "- sampled unobserved items 不是曝光后确认的真实负例。",
            "- 候选池、cutoff-safe feature cache、200 rounds 和 inactive fallback 冻结。",
            "- 未加入 early stopping、未扩大候选预算、未运行 final week。",
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


def run_m211(
    *,
    raw_dir: Path,
    transactions_path: Path,
    m21_metrics_path: Path,
    m29_metrics_path: Path,
    m210_metrics_path: Path,
    m29_cache_dir: Path,
    feature_cache_dir: Path,
    m210_artifact_dir: Path,
    output_dir: Path,
    artifact_dir: Path,
    config: M2Config,
    run_id: str,
) -> dict[str, Any]:
    config.validate()
    if config.evaluation_role != "development" or config.candidate_k != 300:
        raise ValueError("M2.11 requires development-only candidate_k=300")
    if output_dir.exists() or artifact_dir.exists():
        raise FileExistsError(output_dir if output_dir.exists() else artifact_dir)
    m29_metrics = _read_json(m29_metrics_path.resolve())
    m210_metrics = _read_json(m210_metrics_path.resolve())
    if m210_metrics.get("schema_version") != "m2.10-target-aware-ranking-protocol-v1":
        raise ValueError("M2.11 requires measured M2.10 evidence")
    if m210_metrics.get("status") != "measured":
        raise ValueError("M2.10 evidence is not measured")
    output_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    started = time.perf_counter()
    try:
        print("M2.11: validating reusable target-aware feature cache", flush=True)
        feature_cache = build_target_aware_cache(
            raw_dir=raw_dir,
            transactions_path=transactions_path,
            m29_cache_dir=m29_cache_dir,
            cache_dir=feature_cache_dir,
        )
        m21 = _read_json(m21_metrics_path.resolve())
        frozen_name, frozen = _frozen_reference(m21)
        sets = feature_sets()
        development: dict[str, Any] = {}
        for dev, protocol in PROTOCOL.items():
            print(f"M2.11 {dev}: loading distribution-aware sample", flush=True)
            dev_dir = artifact_dir / dev
            dev_dir.mkdir()
            train_paths = [
                _target_feature_paths(feature_cache_dir, cutoff)[0]
                for cutoff in protocol["train"]
            ]
            frame, sizes, sampling = load_distribution_sample(
                train_paths, ALL_FEATURES, seed=SAMPLING_SEED
            )
            source_maps_path = m210_artifact_dir / dev / "category-maps.json"
            category_maps = _load_category_maps(source_maps_path)
            category_evidence = _save_category_maps(
                dev_dir / "category-maps.json", category_maps
            )
            category_evidence["reused_from"] = _file_identity(source_maps_path)
            models: dict[str, tuple[Any, list[str], dict[str, dict[int, int]]]] = {}
            model_evidence: dict[str, Any] = {}
            for name, features in sets.items():
                use_ipw = name == IPW_MODEL
                print(
                    f"M2.11 {dev}: training {name} "
                    f"features={len(features)} ipw={use_ipw}",
                    flush=True,
                )
                model, evidence = train_lambdarank(
                    frame=frame,
                    group_sizes=sizes,
                    group_evidence=sampling,
                    features=features,
                    name=name,
                    use_ipw=use_ipw,
                    artifact_dir=dev_dir,
                    config=config,
                    category_maps=category_maps,
                )
                models[name] = (model, features, category_maps)
                model_evidence[name] = evidence
            del frame, sizes
            gc.collect()
            validation_cutoff = protocol["validation"]
            print(f"M2.11 {dev}: scoring complete expanded pool", flush=True)
            scoring = score_models(
                dataset_path=_target_feature_paths(
                    feature_cache_dir, validation_cutoff
                )[0],
                models=models,
                evaluation_db=dev_dir / "evaluation.duckdb",
                prediction_path=dev_dir / "validation-predictions.parquet",
                config=config,
            )
            variant_names = list(models)
            del models
            gc.collect()
            evaluation = _evaluate_models(
                evaluation_db=dev_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=validation_cutoff,
                metric_k=config.metric_k,
                variant_names=variant_names,
            )
            evaluation = _attach_popularity_segments(
                evaluation=evaluation,
                evaluation_db=dev_dir / "evaluation.duckdb",
                transactions_path=transactions_path,
                cutoff=validation_cutoff,
                variant_names=variant_names,
                budget=config.candidate_k,
            )
            development[dev] = {
                "train_cutoffs": protocol["train"],
                "validation_cutoff": validation_cutoff,
                "sampling": sampling,
                "models": model_evidence,
                "category_encoding": category_evidence,
                "scoring": scoring,
                "evaluation": evaluation,
            }
        summary = summarize(
            development, frozen, m29_metrics, m210_metrics, config.metric_k
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "stage": "M2.11",
            "status": "measured",
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "development_only_no_final_week",
            "config": asdict(config),
            "contract": {
                "candidate_pool": "exact M2.9 six-source Top100 plus up to 200 Item2Vec-only",
                "sampling": {
                    "negative_ratio": NEGATIVE_RATIO,
                    "seed": SAMPLING_SEED,
                    "source_layers": ["baseline_top100", "item2vec_only"],
                    "rank_buckets": "ntile(3) within customer and source layer",
                    "within_bucket": "stable customer-item hash",
                    "allocation": "round-robin across source x rank bucket cells",
                },
                "primary_model": PRIMARY_MODEL,
                "ipw_sensitivity": IPW_MODEL,
                "feature_families": FEATURE_FAMILIES,
                "feature_sets": sets,
                "validation": "complete variable 100..300 candidate pool",
                "boosting_rounds": 200,
                "early_stopping": "deferred_to_avoid_confounding",
                "frozen_reference": frozen_name,
            },
            "inputs": {
                "m21_metrics": _file_identity(m21_metrics_path),
                "m29_metrics": _file_identity(m29_metrics_path),
                "m210_metrics": _file_identity(m210_metrics_path),
                "transactions": _file_identity(transactions_path),
            },
            "feature_cache": feature_cache,
            "development": development,
            "development_summary": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        metrics_path = output_dir / "metrics.json"
        report_path = output_dir / "M2_11_REPORT.md"
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
                "stage": "M2.11",
                "status": "failed",
                "run_id": run_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
