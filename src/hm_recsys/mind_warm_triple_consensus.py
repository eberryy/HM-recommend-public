"""P016 exact action intersection of frozen P009 and P014."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import duckdb
import numpy as np

from . import mind_warm_ap_regression as direct
from . import mind_warm_relation as relation
from . import mind_warm_transfer as transfer


TRIAL = "MIND-WARM-RANK-016"
ART = transfer.ART / TRIAL


def run():
    started = time.perf_counter()
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK016_CONTRACT.json")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(contract["cost_and_stop"]["absolute_session_deadline_utc"]):
        raise TimeoutError("3h deadline reached before016")
    output = relation.REPORT / "MIND_WARM_RANK016_METRICS.json"
    result = {"status": "running", "windows": {}, "source_action_counts_without_labels": {},
              "model_training": False, "selection_used_labels": False, "final_week": "not_run", "independent_confirmation": "not_run"}
    try:
        for cutoff in contract["schedule"]:
            folder = ART / cutoff; folder.mkdir(parents=True, exist_ok=True)
            p009 = transfer.ART / "MIND-WARM-RANK-009" / cutoff / "decisions.parquet"
            p014 = direct.ART / cutoff / "decisions.parquet"
            with duckdb.connect() as db:
                db.execute("SET memory_limit='256MiB'"); db.execute("SET threads=2")
                actions = db.execute("SELECT a.customer_id,a.challenger_article_id,a.victim_article_id,a.champion_rank FROM read_parquet(?) a JOIN read_parquet(?) b USING(customer_id,challenger_article_id,victim_article_id,champion_rank) ORDER BY customer_id", [str(p009), str(p014)]).fetchdf()
                counts = {"P009_actions": db.execute("SELECT count(*) FROM read_parquet(?)", [str(p009)]).fetchone()[0],
                          "P014_actions": db.execute("SELECT count(*) FROM read_parquet(?)", [str(p014)]).fetchone()[0],
                          "exact_intersection_actions": len(actions)}
            relation.parquet(actions, folder / "decisions.parquet")
            result["source_action_counts_without_labels"][cutoff] = counts
            result["windows"][cutoff] = transfer.evaluate_actions(folder, cutoff, actions)
            relation.progress(f"016 {cutoff}:deltaMAP={result['windows'][cutoff]['delta_vs_baseline']:+.8f},actions={len(actions)}")
            relation.save(output, result)
        values = [v["delta_vs_baseline"] for v in result["windows"].values()]
        result["gate"] = {"passed": bool(np.mean(values) > 0 and all(v >= 0 for v in values) and min(values) >= -.0002),
                          "mean_delta": float(np.mean(values)), "nondegrade_windows": sum(v >= 0 for v in values), "worst_delta": min(values)}
        result["status"] = "completed_development_pass" if result["gate"]["passed"] else "completed_rejected"
    except Exception as exc:
        result.update(status="engineering_stopped", error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        result["seconds"] = time.perf_counter() - started; relation.save(output, result)
    return result


if __name__ == "__main__": run()
