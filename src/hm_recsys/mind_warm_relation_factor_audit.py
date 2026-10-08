"""Post-decision factor oracle for MIND-002; inference only, never a policy run.

A fixes the model's favorite challenger, B fixes its favorite victim, and C
relaxes both. All three may refuse replacement and use exact full-list AP.
"""
from __future__ import annotations

import ctypes
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from .mind_warm_relation_audit import (
    ART, REPORT, KEYS, FORBIDDEN_FEATURES, TOL, ap_matrix, audit_pairs,
    constrained_oracle, frozen_baseline, parquet_literal, read, require, write,
)


def available_memory_gib() -> float:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *[(field, ctypes.c_ulonglong) for field in
                      ("total", "available", "page_total", "page_free", "virtual_total", "virtual_free", "extended")]]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    require(bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))), "unable to check free physical memory")
    available = status.available / 2**30
    require(available >= 4, f"diagnostic stopped: only {available:.2f} GiB available RAM")
    return available


def best_without_rejection(scores: pd.DataFrame) -> pd.DataFrame:
    require(set(scores.columns) == set(KEYS + ["replacement_score"]), "argmax input must contain only identities and model utility")
    require(np.isfinite(scores.replacement_score.to_numpy(float)).all(), "nonfinite model utility")
    require(scores.champion_rank.between(8, 12).all(), "unregistered victim position")
    return scores.sort_values(
        ["customer_id", "replacement_score", "challenger_article_id", "victim_article_id"],
        ascending=[True, False, True, True], kind="mergesort",
    ).drop_duplicates("customer_id").reset_index(drop=True)


def favorite_pairs(db: duckdb.DuckDBPyConnection, cutoff: str, training: list[str], features: list[str]) -> tuple[pd.DataFrame, dict]:
    available = available_memory_gib()
    model_path = ART / "models" / ("through_" + training[-1]) / "dense_primary.txt"
    model_meta = read(model_path.with_suffix(".json"))
    require(model_meta["training"] == training and model_meta["features"] == features, "model provenance differs from registered prediction inputs")
    require(not FORBIDDEN_FEATURES.intersection(features), "label column in model input")
    model = lgb.Booster(model_file=str(model_path))
    require(model.feature_name() == features, "actual saved model feature list differs from allowlist")
    started = time.perf_counter()
    pieces, rows = [], 0
    # The label columns physically present in the parquet are not projected.
    columns = list(dict.fromkeys(KEYS + features))
    cursor = db.execute(f"SELECT {','.join(columns)} FROM read_parquet(?)", [str(ART / cutoff / "pairs.parquet")])
    while True:
        batch = cursor.fetch_df_chunk(32)
        if batch.empty:
            break
        matrix = batch[features].to_numpy(np.float32)
        matrix[~np.isfinite(matrix)] = np.nan
        probabilities = model.predict(matrix, num_threads=2)
        require(probabilities.shape == (len(batch), 3), "three-class prediction required")
        require(np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all(), "invalid probabilities")
        require(np.allclose(probabilities.sum(axis=1), 1., atol=1e-6), "probability sum differs from one")
        selected = batch[KEYS].copy()
        selected["replacement_score"] = (probabilities[:, 2] - probabilities[:, 0]) / selected.champion_rank.to_numpy()
        pieces.append(best_without_rejection(selected))
        rows += len(batch)
    require(bool(pieces), "empty registered pair input")
    favorite = best_without_rejection(pd.concat(pieces, ignore_index=True))
    decisions = db.execute("SELECT * FROM read_parquet(?)", [str(ART / "outer" / cutoff / "dense_primary-decisions.parquet")]).fetchdf()
    accepted = favorite[favorite.replacement_score > 0].sort_values(KEYS).reset_index(drop=True)
    original = decisions.sort_values(KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(accepted[KEYS], original[KEYS], check_dtype=False)
    require(np.allclose(accepted.replacement_score, original.replacement_score, atol=TOL, rtol=0), "re-prediction changes the formal decision score")
    return favorite, {"predicted_pair_rows": rows, "favorite_pair_users": len(favorite),
                      "formal_positive_utility_actions_exactly_reproduced": True,
                      "inference_seconds": time.perf_counter() - started, "available_memory_gib_before_inference": available,
                      "model": str(model_path), "prediction_label_columns": [], "large_score_cache_written": False}


def factor_statistics(labels: np.ndarray, truth: np.ndarray, has_positive: np.ndarray,
                      favorite_positions: np.ndarray, favorite_targets: np.ndarray,
                      has_favorite: np.ndarray, accepted: np.ndarray) -> dict:
    """Pure full-population exact-AP calculation, applied only after argmax."""
    baseline_ap = ap_matrix(labels, truth)
    rows = np.flatnonzero(has_favorite)
    positions = favorite_positions[rows]
    chosen_labels = labels.copy()
    chosen_labels[rows, positions] = favorite_targets[rows]
    pair_delta = ap_matrix(chosen_labels, truth) - baseline_ap
    a = constrained_oracle(labels, truth, has_favorite & (favorite_targets == 1))
    b_labels = labels.copy()
    b_rows = np.flatnonzero(has_favorite & has_positive)
    b_labels[b_rows, favorite_positions[b_rows]] = 1
    b = np.maximum(ap_matrix(b_labels, truth) - baseline_ap, 0.)
    c = constrained_oracle(labels, truth, has_positive)
    require(np.all(a <= c + TOL) and np.all(b <= c + TOL), "partial oracle exceeds full oracle")
    require(np.all(np.maximum(pair_delta, 0) <= a + TOL) and np.all(np.maximum(pair_delta, 0) <= b + TOL), "fixed-pair gain exceeds a relaxed oracle")
    rejected = has_favorite & ~accepted
    n = len(labels)
    positive_users = int(has_positive.sum())
    captured = int((has_favorite & (favorite_targets == 1)).sum())
    def outcome(values: np.ndarray) -> dict:
        return {"users_with_positive_gain": int((values > TOL).sum()), "maximum_MAP_delta": float(values.mean()),
                "oracle_MAP@12": float((baseline_ap + values).mean())}
    return {
        "total_evaluation_users": n,
        "users_with_any_mind_only_candidate": int(has_favorite.sum()),
        "users_with_any_positive_mind_only_candidate": positive_users,
        "model_favorite_challenger_positive_users": captured,
        "model_favorite_challenger_capture_rate": captured / positive_users if positive_users else None,
        "positive_candidate_users_missed_by_favorite_challenger": positive_users - captured,
        "A_fixed_favorite_challenger_oracle_victim": outcome(a),
        "B_fixed_favorite_victim_oracle_challenger": outcome(b),
        "C_full_constrained_oracle": outcome(c),
        "A_share_of_C_gain": float(a.sum() / c.sum()) if c.sum() else None,
        "B_share_of_C_gain": float(b.sum() / c.sum()) if c.sum() else None,
        "C_minus_A_MAP_headroom": float((c - a).mean()),
        "C_minus_B_MAP_headroom": float((c - b).mean()),
        "original_rejection_gate": {
            "accepted_users": int(accepted.sum()), "rejected_favorite_pair_users": int(rejected.sum()),
            "rejected_favorite_pair_would_be_beneficial_users": int((rejected & (pair_delta > TOL)).sum()),
            "rejected_favorite_pair_positive_MAP_potential": float(pair_delta[rejected & (pair_delta > TOL)].sum() / n),
            "rejected_favorite_challenger_has_oracle_victim_gain_users": int((rejected & (a > TOL)).sum()),
            "rejected_favorite_challenger_oracle_victim_MAP_potential": float(a[rejected].sum() / n),
            "rejected_users_with_any_positive_mind_only_candidate": int((rejected & has_positive).sum()),
            "rejected_users_full_oracle_MAP_potential": float(c[rejected].sum() / n),
        },
        "unfiltered_favorite_pair_outcomes_afterhoc": {
            "beneficial_users": int((has_favorite & (pair_delta > TOL)).sum()),
            "harmful_users": int((has_favorite & (pair_delta < -TOL)).sum()),
            "neutral_users": int((has_favorite & (np.abs(pair_delta) <= TOL)).sum()),
            "signed_MAP_delta_without_rejection_diagnostic_only": float(pair_delta.mean()),
        },
        "formal_decision_MAP_delta_reproduced": float(pair_delta[accepted].sum() / n),
    }


def run() -> dict:
    started = time.perf_counter()
    output = REPORT / "MIND_WARM_RANK002_FACTOR_AUDIT.json"
    metrics = read(REPORT / "MIND_WARM_RANK002_METRICS.json")
    verification = read(REPORT / "MIND_WARM_RANK002_VERIFICATION.json")
    earlier_diagnostic = read(REPORT / "MIND_WARM_RANK002_DIAGNOSTIC.json")
    schedule = read(REPORT / "MIND_WARM_RANK002_CONTRACT_V2.json")["temporal_protocol"]["chronological_development"]
    features = read(ART / "FEATURE_ALLOWLIST.json")["dense_primary"]
    require(verification["passed"] and metrics["status"] in {"development_failed_baseline_retained", "development_passed_not_promoted"}, "completed and independently verified formal run required")
    report = {"schema": "mind-warm-rank002-factor-audit-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "read_only_experiment_artifacts": True, "new_model_training": False,
              "policy_or_parameter_selection": False, "formal_actions_changed": False,
              "used_for_decisions": False, "independent_confirmation": "not_run", "final_week": "not_run",
              "definitions_zh": {
                  "model_favorite_pair": "每用户在已登记全部MIND独有候选与原第8至12名五个商品的完整组合中，取模型预测效用最高的一对，即使效用不大于零也保留用于事后诊断。效用=(P有益-P伤害)/原商品排名，沿用正式模型与并列规则。",
                  "A": "固定模型首选商品对中的挑战商品，事后允许真实标签选择最佳第8至12名原商品，且允许拒绝；每用户至多替换一次。",
                  "B": "固定模型首选商品对中的原商品及其位置，事后允许真实标签从全部MIND独有候选中选择挑战商品，且允许拒绝。",
                  "C": "事后同时允许真实标签选择挑战商品和第8至12名原商品，且允许拒绝，是完整单换位可达上限。",
                  "maximum_MAP_delta": "相应事后上限的逐用户真实AP@12增量之和除以该窗全部原评测用户数；没有MIND候选的用户增量为零。",
                  "challenger_capture_rate": "模型首选挑战商品实际为正例的用户数除以全部至少有一个正例MIND独有候选的用户数；分母按用户，不按正例商品对。",
                  "rejected_pair_potential": "模型首选的具体商品对若执行本来会提高用户AP，但原效用大于零的门槛拒绝了它；这与换一个原商品以后才存在收益的A上限分别记录。",
                  "factor_headroom": "C-A度量固定当前挑战商品以后损失的Oracle上限，C-B度量固定当前原商品以后损失的Oracle上限；二者含交互且不可相加为独立因果贡献。",
              },
              "limitations": ["所有上限使用动作冻结后已有的购买标签，不是可部署策略表现。", "拒绝门槛遗漏正收益也可能同时避免负收益，不能仅根据遗漏数取消门槛。", "A、B、C是受限反事实诊断，不是互不重叠的失败归因。", "开发窗已有暴露，不能称盲测或独立确认。"],
              "windows": {}}
    try:
        with duckdb.connect() as db:
            db.execute("SET threads=2")
            db.execute("SET memory_limit='2GiB'")
            for cutoff, training in schedule.items():
                favorite, inference = favorite_pairs(db, cutoff, training, features)
                # Targets are joined only after all favorite actions are fixed.
                meta = read(ART / cutoff / "PAIRS.json")
                pair_audit = audit_pairs(db, cutoff, meta)
                top, arrays = frozen_baseline(db, meta["baseline"])
                require(inference["predicted_pair_rows"] == pair_audit["pair_rows"], "inference did not cover the entire registered candidate product")
                db.register("favorite_actions", favorite)
                choice = db.execute("SELECT a.*,g.y1 AS challenger_target FROM favorite_actions a JOIN challenger_groups g ON a.customer_id=g.customer_id AND a.challenger_article_id=g.challenger_article_id ORDER BY a.customer_id").fetchdf()
                require(len(choice) == len(favorite), "favorite challenger lacks a unique source label")
                positive_users = set(db.execute("SELECT DISTINCT customer_id FROM challenger_groups WHERE y1=1").fetchnumpy()["customer_id"])
                users = pd.Index(arrays["users"])
                row = users.get_indexer(choice.customer_id)
                require((row >= 0).all(), "favorite user outside full denominator")
                has_favorite, accepted = np.zeros(len(users), bool), np.zeros(len(users), bool)
                positions, targets = np.zeros(len(users), dtype=int), np.zeros(len(users), dtype=np.int8)
                has_favorite[row] = True
                accepted[row] = choice.replacement_score.to_numpy() > 0
                positions[row] = choice.champion_rank.to_numpy(int) - 1
                targets[row] = choice.challenger_target.to_numpy(np.int8)
                items = top.article_id.to_numpy().reshape(-1, 12)
                require(np.array_equal(items[row, positions[row]], choice.victim_article_id.to_numpy()), "favorite victim is not the baseline item at that position")
                has_positive = np.array([user in positive_users for user in users], dtype=bool)
                statistics = factor_statistics(arrays["labels"], arrays["truth"], has_positive, positions, targets, has_favorite, accepted)
                expected = metrics["chronological_development"][cutoff]["dense_primary"]["delta_vs_baseline"]
                require(abs(statistics["formal_decision_MAP_delta_reproduced"] - expected) <= TOL, "formal delta failed to reproduce")
                old_oracle = earlier_diagnostic["windows"][cutoff]["constrained_oracle"]["maximum_MAP_delta"]
                require(abs(statistics["C_full_constrained_oracle"]["maximum_MAP_delta"] - old_oracle) <= TOL, "C does not reproduce earlier independent oracle")
                report["windows"][cutoff] = {"inference": inference, **statistics}
                print(json.dumps({"cutoff": cutoff, "A": statistics["A_fixed_favorite_challenger_oracle_victim"]["maximum_MAP_delta"], "B": statistics["B_fixed_favorite_victim_oracle_challenger"]["maximum_MAP_delta"], "C": old_oracle, "capture": statistics["model_favorite_challenger_capture_rate"]}), flush=True)
        report["status"] = "completed_not_used_for_decisions"
    except Exception as error:
        report["status"] = "stopped"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write(output, report)
    return report


if __name__ == "__main__":
    run()
