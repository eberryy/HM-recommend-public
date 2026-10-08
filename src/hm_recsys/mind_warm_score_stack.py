"""008: low-dimensional chronological OOF expected-AP stacking."""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from threadpoolctl import threadpool_limits

from . import mind_warm_transfer as transfer
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation
from . import mind_warm_relation_audit as independent

FEATURES = ["old_benefit_per_rank", "old_harm_per_rank", "new_purchase_per_rank", "base_purchase_per_rank", "purchase_logit_difference", "base_original_score", "base_bpr_score", "inverse_rank"]
ART = transfer.ART / "MIND-WARM-RANK-008"


def inputs(cutoff):
    path = ART / cutoff / "features.parquet"
    if path.exists():
        with candidate.connection() as db:
            return db.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf()
    transfer.budget()
    with candidate.connection() as db:
        fields = list(dict.fromkeys(relation.KEYS + relation.FEATURES["dense_primary"]))
        frame = db.execute(f"SELECT {','.join('p.' + f for f in fields)},s.candidate_purchase_score,s.victim_purchase_score FROM read_parquet(?) p JOIN read_parquet(?) s USING(customer_id,challenger_article_id,victim_article_id,champion_rank)", [str(relation.ART / cutoff / "pairs.parquet"), str(transfer.ART / "MIND-WARM-RANK-006" / cutoff / "comparisons.parquet")]).fetchdf()
    schedule = relation.read(relation.REPORT / "MIND_WARM_RANK005_CONTRACT.json")["schedule"]
    relation.check_temporal(schedule[cutoff], cutoff)
    model = lgb.Booster(model_file=str(relation.model_path(schedule[cutoff], "dense_primary")))
    matrix = frame[relation.FEATURES["dense_primary"]].to_numpy(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    probabilities = model.predict(matrix, num_threads=2)
    rank = frame.champion_rank.to_numpy(float)
    result = frame[relation.KEYS].copy()
    result["old_benefit_per_rank"] = probabilities[:, 2] / rank
    result["old_harm_per_rank"] = probabilities[:, 0] / rank
    result["new_purchase_per_rank"] = frame.candidate_purchase_score / rank
    result["base_purchase_per_rank"] = frame.victim_purchase_score / rank
    result["purchase_logit_difference"] = logit(np.clip(frame.candidate_purchase_score, 1e-8, 1 - 1e-8)) - logit(np.clip(frame.victim_purchase_score, 1e-8, 1 - 1e-8))
    result["base_original_score"] = frame.v_score_base
    result["base_bpr_score"] = frame.v_score_bpr
    result["inverse_rank"] = 1 / rank
    candidate.require(np.isfinite(result[FEATURES]).all().all(), "nonfinite stack input; no silent filling")
    relation.parquet(result, path)
    return result


def targets(cutoff, frame):
    with candidate.connection() as db:
        meta = relation.read(relation.ART / cutoff / "PAIRS.json")["baseline"]
        _, arrays = independent.frozen_baseline(db, meta)
        labels = db.execute("SELECT customer_id,article_id,target FROM read_parquet(?)", [str(candidate.ART / cutoff / "candidates.parquet")]).fetchdf()
    joined = frame.merge(labels.rename(columns={"article_id": "challenger_article_id"}), on=["customer_id", "challenger_article_id"], validate="many_to_one")
    index = pd.Index(arrays["users"]).get_indexer(joined.customer_id)
    candidate.require((index >= 0).all() and len(joined) == len(frame), "target identity changed")
    before = arrays["labels"][index]
    after = before.copy()
    after[np.arange(len(after)), joined.champion_rank.to_numpy(int) - 1] = joined.target.to_numpy(np.uint8)
    truth = arrays["truth"][index]
    result = joined[relation.KEYS].copy()
    result["actual_delta"] = independent.ap_matrix(after, truth) - independent.ap_matrix(before, truth)
    return frame.merge(result, on=relation.KEYS, validate="one_to_one")


def fit(frame):
    x, y = frame[FEATURES].to_numpy(float), frame.actual_delta.to_numpy(float)
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    standardized = (x - mean) / scale
    intercept = float(y.mean())
    with threadpool_limits(limits=2):
        coefficients = np.linalg.solve(standardized.T @ standardized + np.eye(len(FEATURES)), standardized.T @ (y - intercept))
    return {"features": FEATURES, "mean": mean.tolist(), "scale": scale.tolist(), "coefficients": coefficients.tolist(), "intercept": intercept, "alpha": 1., "rows": len(frame), "positive_rows": int((y > 0).sum()), "negative_rows": int((y < 0).sum()), "zero_rows": int((y == 0).sum())}


def predict(model, frame):
    candidate.require(model["features"] == FEATURES, "stack allowlist drift")
    return ((frame[FEATURES].to_numpy(float) - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["coefficients"]) + model["intercept"]


def select(scores):
    if set(scores.columns) != set(relation.KEYS + ["replacement_score"]):
        raise ValueError("stack selector accepts only identities and prediction")
    if not scores.champion_rank.between(8, 12).all() or not np.isfinite(scores.replacement_score).all():
        raise ValueError("invalid stack scores or protected rank")
    return scores.loc[scores.replacement_score > 0].sort_values(["customer_id", "replacement_score", "challenger_article_id", "victim_article_id"], ascending=[True, False, True, True]).drop_duplicates("customer_id").reset_index(drop=True)


def run():
    start = time.perf_counter()
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK008_CONTRACT.json")
    output = relation.REPORT / "MIND_WARM_RANK008_METRICS.json"
    result = {"status": "running", "models": {}, "windows": {}, "final_week": "not_run", "independent_confirmation": "not_run"}
    try:
        for cutoff, history in contract["schedule"].items():
            transfer.budget()
            frame = inputs(cutoff)
            scores = frame[relation.KEYS].copy()
            if not history:
                scores["replacement_score"] = 0.
                meta = {"status": "no_prior_OOF_noop", "history": []}
            else:
                relation.check_temporal(history, cutoff)
                training = pd.concat([targets(t, inputs(t)) for t in history], ignore_index=True)
                relation.parquet(training, ART / cutoff / "training.parquet")
                model = fit(training)
                meta = {"status": "fitted", "history": history, "model": model}
                scores["replacement_score"] = predict(model, frame)
            result["models"][cutoff] = meta
            relation.save(ART / cutoff / "MODEL.json", meta)
            relation.parquet(scores, ART / cutoff / "scores.parquet")
            actions = select(scores)
            relation.parquet(actions, ART / cutoff / "decisions.parquet")
            result["windows"][cutoff] = transfer.evaluate_actions(ART / cutoff, cutoff, actions)
            relation.save(output, result)
            relation.progress(f"008 {cutoff}: deltaMAP={result['windows'][cutoff]['delta_vs_baseline']:+.8f},actions={len(actions)}")
        deltas = [v["delta_vs_baseline"] for v in result["windows"].values()]
        result["gate"] = {"passed": bool(np.mean(deltas) > 0 and all(d >= 0 for d in deltas) and min(deltas) >= -.0002), "mean_delta": float(np.mean(deltas)), "nondegrade_windows": sum(d >= 0 for d in deltas), "worst_delta": min(deltas)}
        result["status"] = "completed_development_pass" if result["gate"]["passed"] else "completed_rejected"
    except Exception as exc:
        result.update(status="resource_or_engineering_stopped", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.perf_counter() - start
        relation.save(output, result)
    return result


if __name__ == "__main__":
    run()
