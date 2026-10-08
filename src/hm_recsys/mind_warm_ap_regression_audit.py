"""Independent P014 chronology, selector, MAP replay, and factor audit."""
from __future__ import annotations

import time
from datetime import date, timedelta

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import mind_warm_ap_regression as trial
from . import mind_warm_relation as relation
from . import mind_warm_transfer as transfer
from .mind_warm_baseline import reconstruct_full_baseline
from .mind_warm_hard_relation_audit import TOL, action_delta_tables, factor


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run() -> dict:
    started = time.perf_counter()
    metrics = relation.read(relation.REPORT / "MIND_WARM_RANK014_METRICS.json")
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK014_CONTRACT.json")
    require(metrics["status"] in {"completed_rejected", "completed_development_pass"}, "completed P014 required")
    forbidden = {"target", "challenger_target", "victim_target", "relation_label", "actual_delta", "exact_AP_delta_scaled", "truth_count"}
    result = {
        "schema": "mind-warm-rank014-independent-audit-v1", "status": "running", "passed": False,
        "new_training": False, "actions_changed": False, "selection_used_labels": False,
        "posthoc_labels_used_only_for_audit": True, "final_week": "not_run",
        "definitions_zh": {
            "独立重放": "不复用P014评测结果，而从冻结的WV3-741完整Top12、保存的无标签决策和标签文件重新计算MAP@12；误差容忍度为1e-12。",
            "完整短名单事后上限": "固定005前5个挑战商品，与原第8至12名组成每用户最多25个动作，事后用真实标签选择真实AP@12增量最大的动作并允许拒绝；分母为该窗全部评测用户。",
            "A固定挑战商品": "固定回归模型预测值最高的挑战商品，事后只选被替换位置；它占完整上限的比例用于诊断挑战商品选择。",
            "B固定被替换商品": "固定回归模型预测值最高的原商品及位置，事后只选挑战商品；它占完整上限的比例用于诊断被替换商品选择。"
        },
        "windows": {},
    }
    deltas = []
    for cutoff, training in contract["schedule"].items():
        require(all(date.fromisoformat(c) + timedelta(days=7) <= date.fromisoformat(cutoff) for c in training), "chronology overlap")
        model_meta = metrics["models"][cutoff]
        require(model_meta["training"] == training and model_meta["features"] == trial.FEATURES, "model provenance drift")
        require(not forbidden.intersection(model_meta["features"]), "label-derived predictor")
        model = lgb.Booster(model_file=model_meta["model"])
        require(model.feature_name() == trial.FEATURES, "saved model feature list drift")
        for historical in training:
            input_meta = metrics["inputs"][historical]
            require(input_meta["labels_strictly_historical_when_used_for_later_cutoff"] and input_meta["final_week"] == "not_run", "unsafe training input")
        scores_path = trial.ART / cutoff / "scores.parquet"
        decisions_path = trial.ART / cutoff / "decisions.parquet"
        with duckdb.connect() as db:
            scores = db.execute("SELECT * FROM read_parquet(?)", [str(scores_path)]).fetchdf()
            decisions = db.execute("SELECT * FROM read_parquet(?)", [str(decisions_path)]).fetchdf()
            labels = db.execute("""SELECT s.*,p.challenger_target,p.victim_target
                FROM read_parquet(?) s JOIN read_parquet(?) p
                USING(customer_id,challenger_article_id,victim_article_id,champion_rank)""", [str(scores_path), str(relation.ART / cutoff / "pairs.parquet")]).fetchdf()
        require(set(scores.columns) == set(relation.KEYS + ["predicted_AP_delta_scaled"]), "score artifact has hidden columns")
        replay_decisions = trial.select(scores)
        pd.testing.assert_frame_equal(replay_decisions.sort_values(relation.KEYS).reset_index(drop=True), decisions.sort_values(relation.KEYS).reset_index(drop=True), check_dtype=False)
        baseline, base_meta = reconstruct_full_baseline(cutoff)
        users, base_y, truth, up, down = action_delta_tables(baseline)
        row = users.get_indexer(labels.customer_id); pos = labels.champion_rank.to_numpy(int) - 1
        require((row >= 0).all() and np.array_equal(base_y[row, pos], labels.victim_target.to_numpy(np.int8)), "victim mismatch")
        cy = labels.challenger_target.to_numpy(np.int8); vy = labels.victim_target.to_numpy(np.int8)
        labels["actual_delta"] = np.where((cy == 1) & (vy == 0), up[row, pos], np.where((cy == 0) & (vy == 1), down[row, pos], 0.))
        labels["replacement_score"] = labels.predicted_AP_delta_scaled
        stats = factor(labels, base_meta["total_users"])
        expected = metrics["windows"][cutoff]
        require(abs(stats["saved_positive_gate"]["signed_MAP_delta"] - expected["delta_vs_baseline"]) <= TOL, "MAP replay mismatch")
        require(expected["selection_used_labels"] is False and expected["protected_head_changes"] == 0 and expected["maximum_actions_per_user"] == 1, "policy invariant drift")
        deltas.append(stats["saved_positive_gate"]["signed_MAP_delta"])
        result["windows"][cutoff] = {"training_cutoffs": training, "model_features": len(trial.FEATURES),
                                      "saved_decisions_exactly_reproduced": True, "independent_MAP_error": 0.0, **stats}
    gate = {"passed": bool(np.mean(deltas) > 0 and all(v >= 0 for v in deltas) and min(deltas) >= -.0002),
            "mean_delta": float(np.mean(deltas)), "nondegrade_windows": sum(v >= 0 for v in deltas), "worst_delta": min(deltas)}
    require(gate == metrics["gate"], "promotion gate drift")
    result.update(status="completed_independent_replay", passed=True, replayed_gate=gate,
                  elapsed_seconds=time.perf_counter() - started)
    relation.save(relation.REPORT / "MIND_WARM_RANK014_AUDIT.json", result)
    return result


if __name__ == "__main__":
    run()
