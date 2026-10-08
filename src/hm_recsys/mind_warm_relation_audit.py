"""Independent post-run audit of MIND-WARM-RANK-002; never fits a model.

Only completed registered artifacts are read. The oracle uses already-scored
labels after action freezing and is diagnostic, never an action-selection input.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts/mind_warm_side/MIND-WARM-RANK-002"
REPORT = ROOT / "reports/mind_warm_side"
KEYS = ["customer_id", "challenger_article_id", "victim_article_id", "champion_rank"]
PROBS = ["p_harm", "p_neutral", "p_benefit"]
DECISION_COLUMNS = KEYS + PROBS + ["replacement_score"]
EVALUATION_COLUMNS = DECISION_COLUMNS + [
    "challenger_target", "victim_target", "baseline_ap", "reranked_ap", "actual_delta"
]
VARIANTS = ("context_control", "dense_primary")
FORBIDDEN_FEATURES = {
    "target", "challenger_target", "victim_target", "truth_count", "relation_label",
    "actual_delta", "baseline_ap", "reranked_ap", "unit_gain",
}
TOL = 1e-12


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_cutoff(cutoff: str) -> None:
    require(date.fromisoformat(cutoff) < date(2020, 9, 16), "final week is prohibited")


def parquet_literal(path: str | Path) -> str:
    # DuckDB cannot parameterize CREATE VIEW; quote only the filepath literal.
    return "'" + str(path).replace("'", "''") + "'"


def ap_matrix(labels: np.ndarray, truth_counts: np.ndarray) -> np.ndarray:
    """Compute each user's complete Top12 AP without experiment code reuse."""
    labels = np.asarray(labels, dtype=np.int8)
    truth_counts = np.asarray(truth_counts, dtype=np.int64)
    require(labels.ndim == 2 and labels.shape[1] == 12, "AP requires exactly 12 positions")
    require(len(truth_counts) == len(labels) and bool(np.all(truth_counts > 0)), "invalid truth denominator")
    require(bool(np.isin(labels, [0, 1]).all()), "nonbinary purchase label")
    precision = np.cumsum(labels, axis=1) / np.arange(1, 13, dtype=float)
    return (precision * labels).sum(axis=1) / np.minimum(truth_counts, 12)


def constrained_oracle(labels: np.ndarray, truth_counts: np.ndarray, has_positive: np.ndarray) -> np.ndarray:
    """After-the-fact maximum gain of one new positive at positions 8..12."""
    original = ap_matrix(labels, truth_counts)
    best = np.zeros(len(labels), dtype=float)
    for position in range(7, 12):
        updated = labels.copy()
        eligible = np.asarray(has_positive, dtype=bool) & (labels[:, position] == 0)
        updated[eligible, position] = 1
        best = np.maximum(best, ap_matrix(updated, truth_counts) - original)
    return best


def frozen_baseline(db: duckdb.DuckDBPyConnection, metadata: dict) -> tuple[pd.DataFrame, dict]:
    """Reconstruct saved WV3-741 swaps with SQL, independently of its helper."""
    rank_path, swap_path = Path(metadata["rank_source"]), Path(metadata["swap_source"])
    source_meta = read(rank_path.with_name("DATA.json"))
    require(source_meta["role"] == "outer", "covered-active inner ranks are prohibited")
    require(source_meta["included_users"] == source_meta["total_users"] == metadata["total_users"], "partial baseline population")
    db.execute(f"CREATE OR REPLACE TEMP VIEW raw_rank AS SELECT customer_id,article_id,rf,ap_rf,candidate_rank,user_history_events_12w,target,truth_count FROM read_parquet({parquet_literal(rank_path)}) WHERE ap_rf<=50")
    swaps = db.execute("SELECT customer_id,challenger_article_id,victim_article_id,challenger_rank,victim_rank FROM read_parquet(?)", [str(swap_path)]).fetchdf()
    require(bool(swaps.victim_rank.between(8, 12).all() and swaps.challenger_rank.between(13, 50).all()), "invalid original frozen swap positions")
    changes = pd.concat([
        swaps[["customer_id", "challenger_article_id", "challenger_rank", "victim_rank"]].rename(columns={"challenger_article_id": "article_id", "challenger_rank": "original_rank", "victim_rank": "new_rank"}),
        swaps[["customer_id", "victim_article_id", "victim_rank", "challenger_rank"]].rename(columns={"victim_article_id": "article_id", "victim_rank": "original_rank", "challenger_rank": "new_rank"}),
    ], ignore_index=True)
    require(not changes.duplicated(["customer_id", "article_id"]).any(), "original frozen swaps overlap")
    db.register("swap_changes", changes)
    invalid = db.execute("SELECT count(*) FROM swap_changes s LEFT JOIN raw_rank r USING(customer_id,article_id) WHERE r.article_id IS NULL OR s.original_rank<>r.ap_rf OR r.user_history_events_12w=0").fetchone()[0]
    require(invalid == 0, "frozen swaps differ from their original items/ranks")
    fallback_errors = db.execute("SELECT count(*) FROM raw_rank WHERE ap_rf<>CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE rf END").fetchone()[0]
    require(fallback_errors == 0, "inactive fallback rank mismatch")
    db.execute("CREATE OR REPLACE TEMP VIEW baseline50 AS SELECT r.*,coalesce(s.new_rank,r.ap_rf)::INTEGER AS champion_rank FROM raw_rank r LEFT JOIN swap_changes s USING(customer_id,article_id)")
    bad_groups = db.execute("SELECT count(*) FROM (SELECT customer_id,count(*) n,count(DISTINCT article_id) items,count(DISTINCT champion_rank) ranks FROM baseline50 GROUP BY 1) WHERE n<>50 OR items<>50 OR ranks<>50").fetchone()[0]
    require(bad_groups == 0, "baseline Top50 identity conservation failed")
    top = db.execute("SELECT customer_id,article_id,champion_rank,target,truth_count,user_history_events_12w FROM baseline50 WHERE champion_rank<=12 ORDER BY customer_id,champion_rank").fetchdf()
    require(len(top) == 12 * metadata["total_users"], "full Top12 user denominator mismatch")
    users = top.customer_id.to_numpy()[::12]
    require(np.array_equal(top.customer_id.to_numpy(), np.repeat(users, 12)), "noncontiguous full user lists")
    require(np.array_equal(top.champion_rank.to_numpy(), np.tile(np.arange(1, 13), len(users))), "noncontiguous full Top12 ranks")
    labels = top.target.to_numpy(np.int8).reshape(-1, 12)
    truth = top.truth_count.to_numpy(np.int64).reshape(-1, 12)
    require(bool(np.all(truth == truth[:, :1])), "truth denominator changes within a user")
    aps = ap_matrix(labels, truth[:, 0])
    require(abs(float(aps.mean()) - metadata["MAP@12"]) <= TOL, "independent frozen MAP replay failed")
    return top, {"users": users, "labels": labels, "truth": truth[:, 0], "aps": aps}


def audit_pairs(db: duckdb.DuckDBPyConnection, cutoff: str, meta: dict) -> dict:
    safe_cutoff(cutoff)
    require(meta["cutoff"] == cutoff and meta["stage"] == "outer" and meta["final_week"] == "not_run", "unsafe pair metadata")
    pair_path = ART / cutoff / "pairs.parquet"
    db.execute(f"CREATE OR REPLACE TEMP VIEW pairs AS SELECT customer_id,challenger_article_id,victim_article_id,champion_rank,challenger_target,victim_target,relation_label FROM read_parquet({parquet_literal(pair_path)})")
    db.execute(f"CREATE OR REPLACE TEMP VIEW original_source AS SELECT customer_id,article_id,target,mind_is_new FROM read_parquet({parquet_literal(meta['source'])})")
    db.execute("CREATE OR REPLACE TEMP VIEW challenger_groups AS SELECT customer_id,challenger_article_id,count(*) n,count(DISTINCT victim_article_id) victims,count(DISTINCT champion_rank) ranks,min(champion_rank) lo,max(champion_rank) hi,min(challenger_target) y0,max(challenger_target) y1 FROM pairs GROUP BY 1,2")
    row_count, unique_count, bad_labels = db.execute("SELECT count(*),count(DISTINCT(customer_id,challenger_article_id,victim_article_id)),count(*) FILTER(WHERE relation_label<>challenger_target::INTEGER-victim_target::INTEGER+1 OR challenger_target NOT IN (0,1) OR victim_target NOT IN (0,1)) FROM pairs").fetchone()
    bad_products = db.execute("SELECT count(*) FROM challenger_groups WHERE n<>5 OR victims<>5 OR ranks<>5 OR lo<>8 OR hi<>12 OR y0<>y1").fetchone()[0]
    identity_error = db.execute("SELECT count(*) FROM (SELECT customer_id,article_id,target FROM original_source WHERE mind_is_new=1) s FULL JOIN challenger_groups g ON s.customer_id=g.customer_id AND s.article_id=g.challenger_article_id WHERE s.article_id IS NULL OR g.challenger_article_id IS NULL OR s.target<>g.y1").fetchone()[0]
    require(row_count == unique_count == meta["pair_rows"], "pair rows are missing or duplicated")
    require(bad_labels == bad_products == identity_error == 0, "pair labels, full product, or source coverage invalid")
    counts = {str(k): int(v) for k, v in db.execute("SELECT relation_label,count(*) FROM pairs GROUP BY 1").fetchall()}
    require(counts == meta["class_counts"], "pair-class metadata disagrees with stored rows")
    challengers, positives = db.execute("SELECT count(*),sum(y1) FROM challenger_groups").fetchone()
    require(challengers == meta["challenger_rows"] and positives == meta["challenger_positive_rows"], "candidate count metadata mismatch")
    return {"cutoff": cutoff, "pair_rows": row_count, "class_counts": counts,
            "challenger_rows": challengers, "challenger_positive_rows": positives,
            "all_mind_only_positive_density": positives / challengers if challengers else 0.,
            "complete_challenger_times_five": True, "source_key_and_label_errors": identity_error,
            "duplicate_pairs": row_count - unique_count, "invalid_label_rows": bad_labels}


def audit_variant(db: duckdb.DuckDBPyConnection, cutoff: str, variant: str, expected: dict,
                  top: pd.DataFrame, arrays: dict, pair_audit: dict, training: list[str], allowlist: dict) -> tuple[dict, dict, pd.DataFrame]:
    folder = ART / "outer" / cutoff
    actions = db.execute("SELECT * FROM read_parquet(?)", [str(folder / f"{variant}-decisions.parquet")]).fetchdf()
    evaluated = db.execute("SELECT * FROM read_parquet(?)", [str(folder / f"{variant}-evaluated.parquet")]).fetchdf()
    result = read(folder / f"{variant}-RESULT.json")
    require(set(actions.columns) == set(DECISION_COLUMNS), "decision artifact schema contains extra or missing fields")
    require(set(evaluated.columns) == set(EVALUATION_COLUMNS), "evaluated action schema drift")
    require(not actions.customer_id.duplicated().any(), "more than one action per user")
    require(bool(actions.champion_rank.between(8, 12).all()), "action alters protected ranks")
    pd.testing.assert_frame_equal(actions.sort_values(KEYS).reset_index(drop=True), evaluated[actions.columns].sort_values(KEYS).reset_index(drop=True), check_dtype=False, check_exact=True)
    p = actions[PROBS].to_numpy(float)
    require(bool(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()), "invalid saved probabilities")
    require(bool(np.allclose(p.sum(1), 1., atol=1e-6)), "saved probabilities do not sum to one")
    score = (actions.p_benefit - actions.p_harm) / actions.champion_rank
    require(bool((score > 0).all() and np.allclose(score, actions.replacement_score, atol=TOL, rtol=0)), "saved decisions violate positive-utility rule")
    require(training == expected["training_cutoffs"] == result["training_cutoffs"], "declared training cutoffs drift")
    for source in training:
        safe_cutoff(source)
        require(date.fromisoformat(source) + timedelta(days=7) <= date.fromisoformat(cutoff), "training labels reach evaluation cutoff")
    model_meta = read(ART / "models" / ("through_" + training[-1]) / f"{variant}.json")
    require(model_meta["training"] == training and model_meta["features"] == allowlist[variant], "model artifact does not match cutoff or allowlist")
    source_metas = [read(ART / source / "PAIRS.json") for source in training]
    require(model_meta["pair_rows"] == sum(m["pair_rows"] for m in source_metas), "model training row count differs from complete prior pair sets")
    require(model_meta["class_counts"] == {str(k): sum(m["class_counts"].get(str(k), 0) for m in source_metas) for k in range(3)}, "model class counts differ from complete prior pair sets")
    require(not FORBIDDEN_FEATURES.intersection(model_meta["features"]), "label-derived model feature")
    require(len(model_meta["features"]) == len(set(model_meta["features"])), "duplicate model feature")
    db.register("saved_actions", actions)
    invalid_pairs = db.execute("SELECT count(*) FROM saved_actions a ANTI JOIN pairs p USING(customer_id,challenger_article_id,victim_article_id,champion_rank)").fetchone()[0]
    require(invalid_pairs == 0, "saved decision not in registered full pair space")
    db.execute(f"CREATE OR REPLACE TEMP VIEW saved_labels AS SELECT * FROM read_parquet({parquet_literal(ART / cutoff / 'labels.parquet')})")
    checked = db.execute("SELECT a.customer_id,a.challenger_article_id,a.victim_article_id,a.champion_rank,c.target c_y,v.target v_y FROM saved_actions a LEFT JOIN saved_labels c ON a.customer_id=c.customer_id AND a.challenger_article_id=c.article_id LEFT JOIN saved_labels v ON a.customer_id=v.customer_id AND a.victim_article_id=v.article_id ORDER BY a.customer_id").fetchdf()
    require(len(checked) == len(actions) and not checked[["c_y", "v_y"]].isna().any().any(), "action labels missing or duplicated")
    users = pd.Index(arrays["users"])
    rows = users.get_indexer(checked.customer_id)
    positions = checked.champion_rank.to_numpy(int) - 1
    require(bool((rows >= 0).all()), "action user outside full cohort")
    items = top.article_id.to_numpy().reshape(-1, 12).copy()
    changed_labels = arrays["labels"].copy()
    require(np.array_equal(items[rows, positions], checked.victim_article_id.to_numpy()), "saved victim identity does not match baseline position")
    require(np.array_equal(changed_labels[rows, positions], checked.v_y.to_numpy()), "saved victim label differs from baseline")
    items[rows, positions] = checked.challenger_article_id.to_numpy()
    changed_labels[rows, positions] = checked.c_y.to_numpy(np.int8)
    require(np.array_equal(items[:, :7], top.article_id.to_numpy().reshape(-1, 12)[:, :7]), "head changed during replay")
    require(all(len(set(row)) == 12 for row in items), "replayed Top12 contains duplicate items")
    after = ap_matrix(changed_labels, arrays["truth"])
    delta = after - arrays["aps"]
    full_map = float(after.mean())
    replayed = checked.assign(baseline_ap=arrays["aps"][rows], reranked_ap=after[rows], actual_delta=delta[rows])
    ev = evaluated.sort_values("customer_id").reset_index(drop=True)
    for column in ("baseline_ap", "reranked_ap", "actual_delta"):
        require(bool(np.allclose(replayed[column], ev[column], atol=TOL, rtol=0)), f"independent per-user {column} differs")
    require(np.array_equal(replayed.c_y.to_numpy(), ev.challenger_target.to_numpy()) and np.array_equal(replayed.v_y.to_numpy(), ev.victim_target.to_numpy()), "evaluated labels differ from independently joined labels")
    for reported in (result, expected):
        require(abs(full_map - reported["MAP@12"]) <= TOL, "independent complete-cohort MAP mismatch")
        require(abs(float(delta.mean()) - reported["delta_vs_baseline"]) <= TOL, "independent complete-cohort MAP delta mismatch")
        require(len(actions) == reported["selected_users"], "selected count mismatch")
    beneficial, harmful = int((delta > TOL).sum()), int((delta < -TOL).sum())
    neutral = len(actions) - beneficial - harmful
    require((beneficial, harmful, neutral) == (expected["beneficial_users"], expected["harmful_users"], expected["neutral_users"]), "action outcome counts mismatch")
    positive = int(checked.c_y.sum())
    victim_positive = int(checked.v_y.sum())
    verification = {"passed": True, "decision_columns": list(actions.columns), "decision_artifact_has_no_label_fields": True,
                    "decision_evaluated_key_and_value_identity": True, "maximum_actions_per_user": int(actions.groupby("customer_id").size().max()) if len(actions) else 0,
                    "protected_head_changes": 0, "independent_MAP@12": full_map,
                    "independent_delta_vs_baseline": float(delta.mean()), "MAP_error": abs(full_map - expected["MAP@12"]),
                    "training_cutoffs": training, "latest_training_label_end_exclusive": str(max(date.fromisoformat(x) + timedelta(days=7) for x in training)),
                    "training_labels_strictly_earlier": True, "model_allowlist_excludes_labels": True}
    diagnostic = {"selected_actions": len(actions), "selected_positive_challengers": positive,
                  "selected_challenger_positive_density": positive / len(actions) if len(actions) else 0.,
                  "positive_challenger_retention": positive / pair_audit["challenger_positive_rows"] if pair_audit["challenger_positive_rows"] else 0.,
                  "sacrificed_positive_items": victim_positive, "sacrificed_positive_density": victim_positive / len(actions) if len(actions) else 0.,
                  "beneficial_actions": beneficial, "harmful_actions": harmful, "neutral_actions": neutral,
                  "benefit_rate_per_selected_action": beneficial / len(actions) if len(actions) else 0.,
                  "harm_rate_per_selected_action": harmful / len(actions) if len(actions) else 0.,
                  "neutral_rate_per_selected_action": neutral / len(actions) if len(actions) else 0.,
                  "gross_positive_MAP": float(delta[delta > 0].sum() / len(delta)),
                  "gross_negative_MAP": float(delta[delta < 0].sum() / len(delta)),
                  "MAP@12": full_map, "delta_vs_baseline": float(delta.mean())}
    return verification, diagnostic, evaluated


def action_overlap(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    a = set(map(tuple, left[KEYS].itertuples(index=False, name=None)))
    b = set(map(tuple, right[KEYS].itertuples(index=False, name=None)))
    u, v = set(left.customer_id), set(right.customer_id)
    return {"exact_action_intersection": len(a & b), "exact_action_union": len(a | b),
            "exact_action_jaccard": len(a & b) / len(a | b) if a | b else 0.,
            "context_only_actions": len(a - b), "dense_only_actions": len(b - a),
            "selected_user_intersection": len(u & v), "selected_user_union": len(u | v),
            "both_select_same_user_but_different_action": len(u & v) - len(a & b)}


def run() -> tuple[dict, dict]:
    started = time.perf_counter()
    metrics = read(REPORT / "MIND_WARM_RANK002_METRICS.json")
    config = read(REPORT / "MIND_WARM_RANK002_CONTRACT_V2.json")
    allowlist = read(ART / "FEATURE_ALLOWLIST.json")
    require(metrics["status"] in {"development_failed_baseline_retained", "development_passed_not_promoted"}, "run is incomplete: audit must wait for final result")
    require(metrics["final_week"] == metrics["independent_confirmation"] == "not_run", "evaluation boundary violated")
    schedule = config["temporal_protocol"]["chronological_development"]
    require(set(schedule) == set(metrics["chronological_development"]) and len(schedule) == 3, "three completed chronological tests required")
    verification = {"schema": "mind-warm-rank002-independent-verification-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_status": metrics["status"], "new_training": False, "final_week": "not_run", "independent_confirmation": "not_run",
                    "pairs": {}, "windows": {}, "passed": False}
    diagnostic = {"schema": "mind-warm-rank002-afterhoc-diagnostic-v1", "created_at": verification["created_at"],
                  "source_status": metrics["status"], "new_training": False, "threshold_selection": False,
                  "oracle_used_for_decisions": False, "final_week": "not_run", "windows": {},
                  "definitions_zh": {
                      "all_mind_only_positive_density": "MIND召回且原候选并集未召回的用户—商品对中，下周有购买记录的行数除以全部这些候选行数。",
                      "selected_challenger_positive_density": "实际决定加入的挑战商品中正例行数除以执行的用户级换位动作数；每用户最多一次。",
                      "positive_challenger_retention": "实际选中的正例挑战商品数除以该窗全部MIND独有候选中的正例数。",
                      "benefit_or_harm_rate": "用户AP@12提高或下降的换位动作数除以该方案执行动作数。",
                      "action_overlap": "精确动作以用户、挑战商品、原商品、原位置四元组计数；用户交集另计，不能混用。",
                      "constrained_oracle": "事后使用已评测标签，对每用户允许从本次MIND独有候选中加入至多一件商品并替换原第8至12名之一时，能够达到的最大MAP@12增量；分母为该窗全部原评测用户。",
                  },
                  "limitations": ["开发窗口已有历史暴露，不能称独立确认。", "Oracle仅用于动作冻结后的上界诊断，未选择阈值或重新选择实际动作。", "存盘schema和特征白名单检查证明标签字段未进入这些持久化接口，不能单凭文件schema证明模型分数的全部因果来源。", "未重复模型推断，无法独立检验每用户保存动作是否为全部模型分数的全局最大值。", "无购买记录为未观测购买，缺少曝光日志，不是明确负反馈。"]}
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GiB'")
        all_cutoffs = sorted(set(schedule) | {x for sources in schedule.values() for x in sources})
        for cutoff in all_cutoffs:
            meta = read(ART / cutoff / "PAIRS.json")
            verification["pairs"][cutoff] = audit_pairs(db, cutoff, meta)
            top, arrays = frozen_baseline(db, meta["baseline"])
            # Confirm every distinct victim in the pair table is the named item
            # at that actual frozen baseline position, not merely any five IDs.
            bad_victims = db.execute("SELECT count(*) FROM (SELECT DISTINCT customer_id,victim_article_id,champion_rank,victim_target FROM pairs) p LEFT JOIN baseline50 b ON p.customer_id=b.customer_id AND p.victim_article_id=b.article_id WHERE b.article_id IS NULL OR b.champion_rank<>p.champion_rank OR b.target<>p.victim_target").fetchone()[0]
            require(bad_victims == 0, "pair victim identity/position/label mismatch")
            if cutoff not in schedule:
                continue
            window_v, window_d, evaluated = {}, {}, {}
            for variant in VARIANTS:
                window_v[variant], window_d[variant], evaluated[variant] = audit_variant(
                    db, cutoff, variant, metrics["chronological_development"][cutoff][variant],
                    top, arrays, verification["pairs"][cutoff], schedule[cutoff], allowlist)
            positive_users = set(db.execute("SELECT DISTINCT customer_id FROM challenger_groups WHERE y1=1").fetchnumpy()["customer_id"])
            has_positive = np.array([user in positive_users for user in arrays["users"]], dtype=bool)
            upper = constrained_oracle(arrays["labels"], arrays["truth"], has_positive)
            window_d.update({"total_users": len(arrays["users"]), "all_mind_candidates": verification["pairs"][cutoff],
                             "actions_overlap": action_overlap(evaluated["context_control"], evaluated["dense_primary"]),
                             "constrained_oracle": {"afterhoc_only": True, "maximum_replacements_per_user": 1,
                                 "allowed_positions": [8, 9, 10, 11, 12], "users_with_any_positive_mind_candidate": int(has_positive.sum()),
                                 "users_with_positive_oracle_gain": int((upper > TOL).sum()),
                                 "maximum_MAP_delta": float(upper.mean()),
                                 "baseline_MAP@12": float(arrays["aps"].mean()),
                                 "oracle_MAP@12": float((arrays["aps"] + upper).mean())}})
            verification["windows"][cutoff], diagnostic["windows"][cutoff] = window_v, window_d
    verification["passed"] = True
    verification["elapsed_seconds"] = diagnostic["elapsed_seconds"] = time.perf_counter() - started
    write(REPORT / "MIND_WARM_RANK002_VERIFICATION.json", verification)
    write(REPORT / "MIND_WARM_RANK002_DIAGNOSTIC.json", diagnostic)
    print(json.dumps({"passed": True, "windows": list(schedule), "elapsed_seconds": verification["elapsed_seconds"]}))
    return verification, diagnostic


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
