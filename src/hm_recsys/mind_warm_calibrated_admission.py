"""007: temporal out-of-fit calibration for the actual admission risk sets."""
import time
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from . import mind_warm_transfer as transfer
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation
from . import mind_warm_shared_admission as shared


def fit_mapping(scores, labels):
    x = logit(np.clip(np.asarray(scores, dtype=float), 1e-8, 1 - 1e-8))
    y = np.asarray(labels, dtype=float)
    candidate.require(np.isin(y, [0, 1]).all() and 0 < y.sum() < len(y), "both classes required")
    def loss(theta):
        a, b = theta
        z = a * x + b
        residual = expit(z) - y
        return np.logaddexp(0, z).sum() - y @ z + .5 * a * a, np.array([residual @ x + a, residual.sum()])
    with threadpool_limits(limits=2):
        fitted = minimize(loss, [1., 0.], jac=True, bounds=[(0, None), (None, None)], method="L-BFGS-B", options={"maxiter": 200, "ftol": 1e-10})
    candidate.require(fitted.success, "calibration optimizer did not converge")
    return {"slope": float(fitted.x[0]), "intercept": float(fitted.x[1]), "rows": len(y), "positives": int(y.sum()), "iterations": int(fitted.nit), "success": True}


def predict(mapping, scores):
    return expit(mapping["slope"] * logit(np.clip(np.asarray(scores, float), 1e-8, 1 - 1e-8)) + mapping["intercept"])


def calibration_rows(cutoff):
    with candidate.connection() as db:
        p = str(relation.ART / cutoff / "pairs.parquet")
        s = str(transfer.ART / "MIND-WARM-RANK-006" / cutoff / "comparisons.parquet")
        rows = db.execute("SELECT s.*,p.challenger_target,p.victim_target FROM read_parquet(?) s JOIN read_parquet(?) p USING(customer_id,challenger_article_id,victim_article_id,champion_rank)", [s, p]).fetchdf()
    outputs = {}
    for pool, item, score, label in [("mind", "challenger_article_id", "candidate_purchase_score", "challenger_target"), ("base", "victim_article_id", "victim_purchase_score", "victim_target")]:
        block = rows[["customer_id", item, score, label]].drop_duplicates(["customer_id", item])
        block = block.rename(columns={item: "article_id", score: "score", label: "target"})
        block["cutoff"] = cutoff
        outputs[pool] = block
    return outputs


def run():
    start = time.perf_counter()
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK007_CONTRACT.json")
    output = relation.REPORT / "MIND_WARM_RANK007_METRICS.json"
    result = {"status": "running", "windows": {}, "models": {}, "final_week": "not_run", "independent_confirmation": "not_run"}
    try:
        for cutoff, history in contract["schedule"].items():
            transfer.budget()
            folder = transfer.ART / "MIND-WARM-RANK-007" / cutoff
            with candidate.connection() as db:
                scores = db.execute("SELECT * FROM read_parquet(?)", [str(transfer.ART / "MIND-WARM-RANK-006" / cutoff / "comparisons.parquet")]).fetchdf()
            if not history:
                actions = shared.select(scores.iloc[:0])
                result["models"][cutoff] = {"status": "no_prior_OOF_noop", "history": []}
            else:
                relation.check_temporal(history, cutoff)
                inputs = [calibration_rows(t) for t in history]
                models = {}
                for pool in ("mind", "base"):
                    block = pd.concat([x[pool] for x in inputs], ignore_index=True)
                    relation.parquet(block, folder / (pool + "-calibration-input.parquet"))
                    models[pool] = fit_mapping(block.score, block.target)
                result["models"][cutoff] = {"status": "fitted", "history": history, "mappings": models}
                relation.save(folder / "CALIBRATION.json", result["models"][cutoff])
                scores["candidate_purchase_score"] = predict(models["mind"], scores.candidate_purchase_score)
                scores["victim_purchase_score"] = predict(models["base"], scores.victim_purchase_score)
                actions = shared.select(scores)
            relation.parquet(scores, folder / "comparisons.parquet")
            relation.parquet(actions, folder / "decisions.parquet")
            result["windows"][cutoff] = transfer.evaluate_actions(folder, cutoff, actions)
            relation.save(output, result)
            relation.progress(f"007 {cutoff}: deltaMAP={result['windows'][cutoff]['delta_vs_baseline']:+.8f},actions={len(actions)}")
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


def wait_then_run():
    deadline = datetime.fromisoformat(relation.read(transfer.SESSION)["deadline_utc"])
    relation.progress("007 waiting for preregistered4GiB free RAM; absolute hour deadline unchanged")
    while datetime.now(timezone.utc) < deadline:
        try:
            transfer.budget()
        except RuntimeError as exc:
            if "resource gate failed" not in str(exc):
                raise
            remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
            time.sleep(max(0, min(45, remaining)))
            continue
        # Reserve a minute for this bounded calibration and complete replay.
        if (deadline - datetime.now(timezone.utc)).total_seconds() < 60:
            break
        relation.progress("007 resource gate recovered; executing unchanged registered calibration")
        return run()
    marker = {"status": "deadline_stopped_before_fit", "reason": "4GiB RAM gate did not recover with at least60s remaining", "final_week": "not_run", "windows": {}, "models": {}, "new_training": False}
    relation.save(relation.REPORT / "MIND_WARM_RANK007_METRICS.json", marker)
    relation.progress("007 stopped at absolute hour deadline without new fitting")
    return marker


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-resources", action="store_true")
    args = parser.parse_args()
    wait_then_run() if args.wait_resources else run()
