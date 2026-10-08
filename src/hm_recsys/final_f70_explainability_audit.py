"""Read-only F70 importance/TreeSHAP and strict-vs-sparse Recall audit."""
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np

from .final_candidate_e2 import WINDOWS
from .final_oracle_audit import dump
from .final_temporal_item_e16 import (
    FEATURES,
    PREPARED,
    RUN as F70_RUN,
    TEMPORAL_NAMES,
    final_matrix,
)


REPORT = Path("reports/final")
CONTRACT = REPORT / "F70_EXPLAINABILITY_AUDIT_CONTRACT.json"
OUTPUT = REPORT / "F70_EXPLAINABILITY_AUDIT.json"
KS = (1, 5)
SUBGROUPS = {
    "strict_cold": "strict_cold_flag",
    "sparse_1_5": "sparse1_5_flag",
}
TEMPORAL_GROUPS = {
    "short_window_events": [
        "item_events_1d", "item_events_3d", "item_events_7d", "item_events_14d",
    ],
    "medium_window_demand": [
        "item_events_28d", "item_events_84d", "item_customers_7d",
        "item_customers_28d",
    ],
    "catalog_lifecycle": ["days_since_last_sale", "days_since_first_sale"],
    "demand_velocity": [
        "item_velocity_3d_vs_28d", "item_velocity_7d_vs_28d",
        "item_velocity_14d_vs_84d",
    ],
}
SHAP_POPULATIONS = (
    "all_f70_top1",
    "all_positive_pairs",
    "strict_cold_positive_pairs",
    "sparse_1_5_positive_pairs",
    "f70_new_top1_positive_pairs",
    "f70_new_top5_positive_pairs",
    "f70_lost_top1_positive_pairs",
    "f70_lost_top5_positive_pairs",
)


def contract(repo: Path) -> dict:
    return {
        "status": "registered_before_explainability_and_subgroup_label_read",
        "stage": "FINAL-E24 F70可解释性与Cold子组Recall只读审计",
        "purpose": "回答F70为何有效，以及对零历史交互商品和1至5次历史交互商品中的哪一类更有效。",
        "models": "复用E16四个现成F70 LightGBM模型；models_fit=0。",
        "feature_explanation": {
            "gain_importance": "每个模型按特征累计分裂增益归一化到和为1，再对四窗等权平均。行业通用模型重要性，存在连续特征/相关特征偏置。",
            "tree_shap": "LightGBM原生pred_contrib精确分解树模型原始分数；统计绝对贡献和有符号贡献。行业通用解释方法，不是因果效应。",
            "tree_shap_rows": "四窗全部F70用户内Top1候选，加Cold50中的全部未来正例用户—商品对；两者并集逐窗计算，无随机抽样。",
            "no_grouped_ablation": "不做置零伪消融；置零会产生训练分布外输入。严谨分组消融需要重训，与本轮只读约束冲突。",
        },
        "subgroup_recall": {
            "ranking": "B0与F70均先在每名用户完整Cold50上排序，不在子组内重新排序。",
            "strict_cold": "分母为strict_cold_flag=1且随后一周购买的用户—商品正例对；该标记表示截止日前交互次数为0。",
            "sparse_1_5": "分母为sparse1_5_flag=1且随后一周购买的用户—商品正例对；该标记表示截止日前交互次数为1至5。",
            "numerator": "对应子组正例对中，完整Cold50排序名次不大于K的数量。",
            "aggregation": "四窗合计使用正例对微平均；即先加分子、分母，再计算Recall。",
        },
        "windows": WINDOWS,
        "latest_behavior_exclusive": "each window cutoff",
        "labels_used_only_for_read_only_recall_and_positive_SHAP_cohorting": True,
        "recommendations_or_admissions_changed": False,
        "final_week": "not_run",
        "no_commit_push": True,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    }


def rank_positions(frame, score: np.ndarray) -> np.ndarray:
    """Return one-based full-Cold50 ranks, with stable original-row tie breaks."""
    ranks = np.empty(len(frame), dtype=np.int16)
    for rows in frame.groupby("user_index", sort=False).indices.values():
        rows = np.asarray(rows, dtype=np.int64)
        order = np.argsort(-score[rows], kind="stable")
        ranks[rows[order]] = np.arange(1, len(rows) + 1, dtype=np.int16)
    return ranks


def subgroup_recall(frame, ranks: np.ndarray, flag_column: str) -> dict:
    target = frame.target.to_numpy(np.int8) == 1
    group = frame[flag_column].to_numpy(np.int8) == 1
    mask = target & group
    denominator = int(mask.sum())
    hits = {str(k): int(np.sum(mask & (ranks <= k))) for k in KS}
    return {
        "positive_pair_denominator": denominator,
        "hits": hits,
        "recall": {str(k): hits[str(k)] / denominator if denominator else None for k in KS},
    }


def add_shap(accumulator: dict, name: str, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    accumulator[name]["count"] += int(len(values))
    accumulator[name]["abs_sum"] += np.abs(values).sum(axis=0)
    accumulator[name]["signed_sum"] += values.sum(axis=0)


def summarize_shap(accumulator: dict) -> dict:
    feature_index = {name: i for i, name in enumerate(FEATURES)}
    result = {}
    for population in SHAP_POPULATIONS:
        entry = accumulator[population]
        count = entry["count"]
        if count == 0:
            result[population] = {"candidate_rows": 0, "available": False}
            continue
        mean_abs = entry["abs_sum"] / count
        mean_signed = entry["signed_sum"] / count
        total_abs = float(mean_abs.sum())
        feature_rows = [
            {
                "feature": feature,
                "mean_abs_shap_raw_margin": float(mean_abs[index]),
                "mean_signed_shap_raw_margin": float(mean_signed[index]),
                "abs_share": float(mean_abs[index] / total_abs) if total_abs else 0.0,
            }
            for index, feature in enumerate(FEATURES)
        ]
        feature_rows.sort(key=lambda row: (-row["mean_abs_shap_raw_margin"], row["feature"]))
        groups = {}
        for group, features in TEMPORAL_GROUPS.items():
            indices = [feature_index[feature] for feature in features]
            groups[group] = {
                "features": features,
                "mean_abs_shap_raw_margin": float(mean_abs[indices].sum()),
                "mean_signed_shap_raw_margin": float(mean_signed[indices].sum()),
                "abs_share_of_all_70_features": float(mean_abs[indices].sum() / total_abs) if total_abs else 0.0,
            }
        temporal_indices = [feature_index[feature] for feature in TEMPORAL_NAMES]
        result[population] = {
            "candidate_rows": count,
            "available": True,
            "all_70_mean_abs_shap_raw_margin": total_abs,
            "temporal_13_mean_abs_shap_raw_margin": float(mean_abs[temporal_indices].sum()),
            "temporal_13_mean_signed_shap_raw_margin": float(mean_signed[temporal_indices].sum()),
            "temporal_13_abs_share_of_all_70_features": float(mean_abs[temporal_indices].sum() / total_abs) if total_abs else 0.0,
            "temporal_groups": groups,
            "top_features": feature_rows[:20],
            "temporal_features": [row for row in feature_rows if row["feature"] in TEMPORAL_NAMES],
        }
    return result


def summarize_importance(per_window: dict) -> dict:
    rows = []
    for index, feature in enumerate(FEATURES):
        gain = [per_window[window]["gain_normalized"][index] for window in WINDOWS]
        split = [per_window[window]["split_normalized"][index] for window in WINDOWS]
        rows.append({
            "feature": feature,
            "mean_normalized_gain": float(np.mean(gain)),
            "min_normalized_gain": float(np.min(gain)),
            "max_normalized_gain": float(np.max(gain)),
            "mean_normalized_split_count": float(np.mean(split)),
            "gain_rank_by_window": {
                window: int(per_window[window]["gain_rank"][index]) for window in WINDOWS
            },
        })
    rows.sort(key=lambda row: (-row["mean_normalized_gain"], row["feature"]))
    temporal = [row for row in rows if row["feature"] in TEMPORAL_NAMES]
    temporal_share = {
        window: float(sum(
            per_window[window]["gain_normalized"][FEATURES.index(feature)]
            for feature in TEMPORAL_NAMES
        ))
        for window in WINDOWS
    }
    groups = {}
    for group, features in TEMPORAL_GROUPS.items():
        groups[group] = {
            "features": features,
            "mean_normalized_gain": float(sum(
                next(row["mean_normalized_gain"] for row in rows if row["feature"] == feature)
                for feature in features
            )),
        }
    return {
        "definition": "每窗分裂gain先归一化到70特征之和为1，四窗等权平均。",
        "temporal_13_gain_share_by_window": temporal_share,
        "temporal_13_mean_gain_share": float(np.mean(list(temporal_share.values()))),
        "temporal_groups": groups,
        "top_features": rows[:20],
        "temporal_features": temporal,
    }


def micro_aggregate(windows: dict) -> dict:
    result = {}
    for subgroup in SUBGROUPS:
        methods = {}
        denominator = sum(
            windows[window]["subgroups"][subgroup]["B0"]["positive_pair_denominator"]
            for window in WINDOWS
        )
        for method in ("B0", "F70"):
            hits = {
                str(k): sum(
                    windows[window]["subgroups"][subgroup][method]["hits"][str(k)]
                    for window in WINDOWS
                )
                for k in KS
            }
            methods[method] = {
                "positive_pair_denominator": int(denominator),
                "hits": hits,
                "recall": {
                    str(k): hits[str(k)] / denominator if denominator else None for k in KS
                },
            }
        methods["delta_recall"] = {
            str(k): methods["F70"]["recall"][str(k)] - methods["B0"]["recall"][str(k)]
            if denominator else None
            for k in KS
        }
        result[subgroup] = methods
    return result


def run(repo: Path) -> dict:
    repo = Path(repo)
    if OUTPUT.exists() or CONTRACT.exists():
        raise FileExistsError("F70 explainability audit evidence is immutable; use a new stage name")
    dump(repo / CONTRACT, contract(repo))
    started = time.perf_counter()
    windows = {}
    importance_by_window = {}
    shap_acc = {
        name: {
            "count": 0,
            "abs_sum": np.zeros(len(FEATURES), dtype=np.float64),
            "signed_sum": np.zeros(len(FEATURES), dtype=np.float64),
        }
        for name in SHAP_POPULATIONS
    }
    shap_reconstruction_max_error = 0.0
    for window, cutoff in WINDOWS.items():
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        cold = data["cold"]
        model = lgb.Booster(model_file=str(repo / F70_RUN / f"{cutoff}-F70.txt"))
        if model.feature_name() != FEATURES:
            raise RuntimeError(f"F70 feature order mismatch at {window}")
        matrix, mapping = final_matrix(repo, cutoff, data)
        score = model.predict(matrix, num_threads=4)
        b0_score = -cold.b0_rank.to_numpy(float)
        b0_rank = rank_positions(cold, b0_score)
        f70_rank = rank_positions(cold, score)
        target = cold.target.to_numpy(np.int8) == 1
        strict = cold.strict_cold_flag.to_numpy(np.int8) == 1
        sparse = cold.sparse1_5_flag.to_numpy(np.int8) == 1

        subgroups = {}
        for subgroup, flag in SUBGROUPS.items():
            subgroups[subgroup] = {
                "definition_flag": flag,
                "candidate_row_count": int((cold[flag].to_numpy(np.int8) == 1).sum()),
                "B0": subgroup_recall(cold, b0_rank, flag),
                "F70": subgroup_recall(cold, f70_rank, flag),
            }
        overlap = strict & sparse
        unclassified = ~(strict | sparse)
        windows[window] = {
            "cutoff": cutoff,
            "candidate_rows": int(len(cold)),
            "positive_pairs": int(target.sum()),
            "strict_sparse_overlap_candidate_rows": int(overlap.sum()),
            "strict_sparse_overlap_positive_pairs": int((overlap & target).sum()),
            "neither_subgroup_candidate_rows": int(unclassified.sum()),
            "neither_subgroup_positive_pairs": int((unclassified & target).sum()),
            "subgroups": subgroups,
            "mapping": mapping,
        }

        gain = model.feature_importance(importance_type="gain").astype(float)
        split = model.feature_importance(importance_type="split").astype(float)
        gain_normalized = gain / gain.sum() if gain.sum() else gain
        split_normalized = split / split.sum() if split.sum() else split
        gain_order = np.argsort(-gain_normalized, kind="stable")
        gain_rank = np.empty(len(gain_order), dtype=np.int16)
        gain_rank[gain_order] = np.arange(1, len(gain_order) + 1, dtype=np.int16)
        importance_by_window[window] = {
            "gain_normalized": gain_normalized,
            "split_normalized": split_normalized,
            "gain_rank": gain_rank,
        }

        population_masks = {
            "all_f70_top1": f70_rank == 1,
            "all_positive_pairs": target,
            "strict_cold_positive_pairs": target & strict,
            "sparse_1_5_positive_pairs": target & sparse,
            "f70_new_top1_positive_pairs": target & (f70_rank <= 1) & (b0_rank > 1),
            "f70_new_top5_positive_pairs": target & (f70_rank <= 5) & (b0_rank > 5),
            "f70_lost_top1_positive_pairs": target & (f70_rank > 1) & (b0_rank <= 1),
            "f70_lost_top5_positive_pairs": target & (f70_rank > 5) & (b0_rank <= 5),
        }
        union = np.zeros(len(cold), dtype=bool)
        for mask in population_masks.values():
            union |= mask
        union_indices = np.flatnonzero(union)
        contribution = model.predict(matrix[union_indices], pred_contrib=True, num_threads=4)
        raw_score = model.predict(matrix[union_indices], raw_score=True, num_threads=4)
        shap_values = contribution[:, :-1]
        reconstructed = contribution.sum(axis=1)
        shap_reconstruction_max_error = max(
            shap_reconstruction_max_error,
            float(np.max(np.abs(reconstructed - raw_score))),
        )
        union_position = np.full(len(cold), -1, dtype=np.int64)
        union_position[union_indices] = np.arange(len(union_indices), dtype=np.int64)
        for population, mask in population_masks.items():
            positions = union_position[np.flatnonzero(mask)]
            add_shap(shap_acc, population, shap_values[positions])
        print(json.dumps({
            "window": window,
            "positive_pairs": int(target.sum()),
            "strict_positive_pairs": int((target & strict).sum()),
            "sparse_positive_pairs": int((target & sparse).sum()),
            "tree_shap_rows": int(len(union_indices)),
        }, ensure_ascii=False), flush=True)

    importance = summarize_importance(importance_by_window)
    shap = summarize_shap(shap_acc)
    aggregate = micro_aggregate(windows)
    output = {
        "status": "completed",
        "stage": "FINAL-E24 F70可解释性与Cold子组Recall只读审计",
        "definitions": {
            "gain_importance": "树分裂带来的训练目标增益；四窗内先归一化再等权平均，不代表因果。",
            "tree_shap": "单个候选的模型原始分数相对基准值由各特征贡献构成；绝对值衡量模型依赖，有符号值表示推高或压低原始分数。",
            "f70_new_top_k_positive_pairs": "真实正例用户—商品对中，F70完整Cold50名次不大于K且B0名次大于K的正例对。",
            "micro_recall": "四窗正例命中数之和除以四窗正例对分母之和。",
        },
        "windows": windows,
        "micro_aggregate": aggregate,
        "feature_importance": importance,
        "tree_shap": shap,
        "verification": {
            "models_fit": 0,
            "model_count": len(WINDOWS),
            "feature_count": len(FEATURES),
            "temporal_feature_count": len(TEMPORAL_NAMES),
            "tree_shap_raw_margin_reconstruction_max_abs_error": shap_reconstruction_max_error,
            "strict_sparse_overlap_positive_pairs": int(sum(
                windows[window]["strict_sparse_overlap_positive_pairs"] for window in WINDOWS
            )),
            "all_positive_pairs_total": int(sum(
                windows[window]["positive_pairs"] for window in WINDOWS
            )),
            "recommendations_or_admissions_changed": False,
            "final_week": "not_run",
        },
        "seconds": time.perf_counter() - started,
    }
    dump(repo / OUTPUT, output)
    return output


if __name__ == "__main__":
    try:
        result = run(Path.cwd())
        print(json.dumps({
            "status": result["status"],
            "micro_aggregate": result["micro_aggregate"],
            "temporal_gain_share": result["feature_importance"]["temporal_13_mean_gain_share"],
            "top_temporal_features_by_gain": result["feature_importance"]["temporal_features"][:5],
            "positive_temporal_shap": {
                key: result["tree_shap"][key] for key in (
                    "all_positive_pairs", "strict_cold_positive_pairs",
                    "sparse_1_5_positive_pairs", "f70_new_top1_positive_pairs",
                    "f70_new_top5_positive_pairs",
                )
            },
            "verification": result["verification"],
            "seconds": result["seconds"],
        }, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        dump(REPORT / f"F70_EXPLAINABILITY_AUDIT_FAILURE_{time.time_ns()}.json", {
            "error": traceback.format_exc(),
            "final_week": "not_run",
        })
        raise
