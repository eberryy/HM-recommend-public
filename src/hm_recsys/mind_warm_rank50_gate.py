"""P018 fixed architecture-native raw MIND Top50 gate on P017 actions."""
from __future__ import annotations

import time
from datetime import datetime, timezone

import duckdb
import numpy as np

from . import mind_warm_interest_gate as prior
from . import mind_warm_relation as relation
from . import mind_warm_transfer as transfer


TRIAL = "MIND-WARM-RANK-018"
ART = transfer.ART / TRIAL


def run():
    started = time.perf_counter(); contract = relation.read(relation.REPORT / "MIND_WARM_RANK018_CONTRACT.json")
    if datetime.now(timezone.utc) >= datetime.fromisoformat(contract["cost_and_stop"]["absolute_session_deadline_utc"]):
        raise TimeoutError("3h deadline reached before018")
    output = relation.REPORT / "MIND_WARM_RANK018_METRICS.json"
    result = {"status": "running", "windows": {}, "counts_before_label_read": {}, "model_training": False,
              "selection_used_labels": False, "final_week": "not_run", "independent_confirmation": "not_run"}
    try:
        for cutoff in contract["schedule"]:
            folder = ART / cutoff; folder.mkdir(parents=True, exist_ok=True)
            source = transfer.source(cutoff); actions_path = prior.ART / cutoff / "decisions.parquet"
            with duckdb.connect() as db:
                db.execute("SET memory_limit='256MiB'"); db.execute("SET threads=2")
                actions = db.execute("""SELECT a.* FROM read_parquet(?) a JOIN read_parquet(?) s
                    ON a.customer_id=s.customer_id AND a.challenger_article_id=s.article_id
                    WHERE s.mind_is_new=1 AND s.mind_rank<=50 ORDER BY a.customer_id""", [str(actions_path), str(source)]).fetchdf()
                original = db.execute("SELECT count(*) FROM read_parquet(?)", [str(actions_path)]).fetchone()[0]
            relation.parquet(actions, folder / "decisions.parquet")
            result["counts_before_label_read"][cutoff] = {"P017_actions": original, "raw_MIND_Top50_actions": len(actions)}
            result["windows"][cutoff] = transfer.evaluate_actions(folder, cutoff, actions)
            relation.progress(f"018 {cutoff}:deltaMAP={result['windows'][cutoff]['delta_vs_baseline']:+.8f},actions={len(actions)}")
            relation.save(output, result)
        values = [v["delta_vs_baseline"] for v in result["windows"].values()]
        result["gate"] = {"passed": bool(np.mean(values) > 0 and all(v >= 0 for v in values) and min(values) >= -.0002),
                          "mean_delta": float(np.mean(values)), "nondegrade_windows": sum(v >= 0 for v in values), "worst_delta": min(values)}
        result["status"] = "completed_development_pass_pending_independent_replay" if result["gate"]["passed"] else "completed_rejected"
    except Exception as exc:
        result.update(status="engineering_stopped", error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        result["seconds"] = time.perf_counter() - started; relation.save(output, result)
    return result


if __name__ == "__main__": run()
