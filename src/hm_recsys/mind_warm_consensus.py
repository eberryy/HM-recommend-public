"""009 fixed exact-action unanimity; no new model or score threshold."""
import time
import numpy as np
from . import mind_warm_transfer as transfer
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation


def run():
    start = time.perf_counter()
    config = relation.read(relation.REPORT / "MIND_WARM_RANK009_CONTRACT.json")
    result = {"status": "running", "windows": {}, "new_training": False, "final_week": "not_run", "independent_confirmation": "not_run"}
    output = relation.REPORT / "MIND_WARM_RANK009_METRICS.json"
    try:
        for cutoff, history in config["schedule"].items():
            transfer.budget()
            relation.check_temporal(history, cutoff)
            with candidate.connection() as db:
                actions = db.execute("SELECT a.* FROM read_parquet(?) a SEMI JOIN read_parquet(?) b USING(customer_id,challenger_article_id,victim_article_id,champion_rank)", [str(transfer.ART / "MIND-WARM-RANK-005" / cutoff / "decisions.parquet"), str(transfer.ART / "MIND-WARM-RANK-006" / cutoff / "decisions.parquet")]).fetchdf()
            folder = transfer.ART / "MIND-WARM-RANK-009" / cutoff
            relation.parquet(actions, folder / "decisions.parquet")
            result["windows"][cutoff] = transfer.evaluate_actions(folder, cutoff, actions)
            relation.save(output, result)
            relation.progress(f"009 {cutoff}: deltaMAP={result['windows'][cutoff]['delta_vs_baseline']:+.8f},actions={len(actions)}")
        values = [v["delta_vs_baseline"] for v in result["windows"].values()]
        result["gate"] = {"passed": bool(np.mean(values) > 0 and all(v >= 0 for v in values) and min(values) >= -.0002), "mean_delta": float(np.mean(values)), "nondegrade_windows": sum(v >= 0 for v in values), "worst_delta": min(values)}
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
