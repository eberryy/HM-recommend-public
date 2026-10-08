"""Registered 006: common purchase-score comparison, no fit."""
import time

import lightgbm as lgb
import numpy as np

from . import mind_warm_transfer as transfer
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation


SCORE_COLUMNS = relation.KEYS + ["candidate_purchase_score", "victim_purchase_score"]


def select(scores):
    if set(scores.columns) != set(SCORE_COLUMNS):
        raise ValueError("shared selector accepts only identity and two predictions")
    if not scores.champion_rank.between(8, 12).all():
        raise ValueError("protected head")
    p = scores[["candidate_purchase_score", "victim_purchase_score"]].to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("invalid purchase scores")
    result = scores.copy()
    result["replacement_score"] = (result.candidate_purchase_score - result.victim_purchase_score) / result.champion_rank
    return result.loc[result.replacement_score > 0].sort_values(["customer_id", "replacement_score", "challenger_article_id", "victim_article_id"], ascending=[True, False, True, True], kind="mergesort").drop_duplicates("customer_id").reset_index(drop=True)


def comparison(cutoff, training):
    folder = transfer.ART / "MIND-WARM-RANK-005" / cutoff
    fields = [*relation.KEYS, *relation.USER, *["v_" + f for f in relation.ITEM + candidate.DENSE_FEATURES]]
    with candidate.connection() as db:
        frame = db.execute(f"SELECT {','.join('p.' + f for f in fields)},r.score candidate_purchase_score FROM read_parquet(?) p JOIN read_parquet(?) r ON p.customer_id=r.customer_id AND p.challenger_article_id=r.article_id WHERE r.rank=1 AND p.v_user_item_events_12w=0", [str(relation.ART / cutoff / "pairs.parquet"), str(folder / "order.parquet")]).fetchdf()
    model_path = transfer.ART / "MIND-WARM-RANK-005" / "models" / ("through_" + training[-1]) / "model.txt"
    model = lgb.Booster(model_file=str(model_path))
    input_columns = [f if f in relation.USER else "v_" + f for f in candidate.FEATURES]
    matrix = frame[input_columns].to_numpy(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    scores = frame[relation.KEYS + ["candidate_purchase_score"]].copy()
    scores["victim_purchase_score"] = model.predict(matrix, num_threads=2)
    return scores


def run():
    start = time.perf_counter()
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK006_CONTRACT.json")
    path = relation.REPORT / "MIND_WARM_RANK006_METRICS.json"
    if path.exists() and relation.read(path).get("status", "").startswith("completed"):
        return relation.read(path)
    result = {"status": "running", "windows": {}, "training": "none; reuse005", "final_week": "not_run", "independent_confirmation": "not_run"}
    try:
        for cutoff, training in contract["schedule"].items():
            transfer.budget()
            relation.check_temporal(training, cutoff)
            scores = comparison(cutoff, training)
            folder = transfer.ART / "MIND-WARM-RANK-006" / cutoff
            relation.parquet(scores, folder / "comparisons.parquet")
            actions = select(scores)
            relation.parquet(actions, folder / "decisions.parquet")
            evaluation = transfer.evaluate_actions(folder, cutoff, actions)
            evaluation["eligible_comparisons"] = len(scores)
            evaluation["eligible_users"] = scores.customer_id.nunique()
            result["windows"][cutoff] = evaluation
            relation.save(path, result)
            relation.progress(f"006 {cutoff}: deltaMAP={evaluation['delta_vs_baseline']:+.8f},actions={len(actions)}")
        deltas = [v["delta_vs_baseline"] for v in result["windows"].values()]
        result["gate"] = {"passed": bool(np.mean(deltas) > 0 and all(d >= 0 for d in deltas) and min(deltas) >= -.0002), "mean_delta": float(np.mean(deltas)), "nondegrade_windows": sum(d >= 0 for d in deltas), "worst_delta": min(deltas)}
        result["status"] = "completed_development_pass" if result["gate"]["passed"] else "completed_rejected"
    except Exception as exc:
        result.update(status="engineering_stopped", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.perf_counter() - start
        relation.save(path, result)
    return result


if __name__ == "__main__":
    run()
