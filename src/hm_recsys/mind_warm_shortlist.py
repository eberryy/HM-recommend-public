"""010: fixed pre-existing Top5 endpoint instead of premature Top1."""
import time
import lightgbm as lgb
import numpy as np
from . import mind_warm_transfer as transfer
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation


def run():
    start = time.perf_counter()
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK010_CONTRACT.json")
    path = relation.REPORT / "MIND_WARM_RANK010_METRICS.json"
    result = {"status": "running", "windows": {}, "new_training": False, "final_week": "not_run", "independent_confirmation": "not_run"}
    try:
        for cutoff, training in contract["schedule"].items():
            transfer.budget()
            relation.check_temporal(training, cutoff)
            fields = list(dict.fromkeys(relation.KEYS + relation.FEATURES["dense_primary"]))
            with candidate.connection() as db:
                frame = db.execute(f"SELECT {','.join('p.' + f for f in fields)} FROM read_parquet(?) p JOIN read_parquet(?) r ON p.customer_id=r.customer_id AND p.challenger_article_id=r.article_id WHERE r.rank<=5", [str(relation.ART / cutoff / "pairs.parquet"), str(transfer.ART / "MIND-WARM-RANK-005" / cutoff / "order.parquet")]).fetchdf()
            model = lgb.Booster(model_file=str(relation.model_path(training, "dense_primary")))
            x = frame[relation.FEATURES["dense_primary"]].to_numpy(np.float32)
            x[~np.isfinite(x)] = np.nan
            scores = frame[relation.KEYS].copy()
            scores[relation.PROBS] = model.predict(x, num_threads=2)
            folder = transfer.ART / "MIND-WARM-RANK-010" / cutoff
            relation.parquet(scores, folder / "scores.parquet")
            actions = relation.select_actions(scores)
            relation.parquet(actions, folder / "decisions.parquet")
            result["windows"][cutoff] = transfer.evaluate_actions(folder, cutoff, actions)
            relation.save(path, result)
            relation.progress(f"010 {cutoff}: deltaMAP={result['windows'][cutoff]['delta_vs_baseline']:+.8f},actions={len(actions)}")
        values = [v["delta_vs_baseline"] for v in result["windows"].values()]
        result["gate"] = {"passed": bool(np.mean(values) > 0 and all(v >= 0 for v in values) and min(values) >= -.0002), "mean_delta": float(np.mean(values)), "nondegrade_windows": sum(v >= 0 for v in values), "worst_delta": min(values)}
        result["status"] = "completed_development_pass" if result["gate"]["passed"] else "completed_rejected"
    except Exception as exc:
        result.update(status="resource_or_engineering_stopped", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.perf_counter() - start
        relation.save(path, result)
    return result


if __name__ == "__main__":
    run()
