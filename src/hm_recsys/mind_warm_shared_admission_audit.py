"""Independent small-memory SQL and complete-AP replay for006/007."""
import argparse
import time
from datetime import date, timedelta

import duckdb
import numpy as np
from scipy.special import expit, logit

from . import mind_warm_relation_audit as independent


ROOT, REPORT = independent.ROOT, independent.REPORT
ART = ROOT / "artifacts/mind_warm_side/MIND-WARM-TRANSFER"


def verify(number):
    started = time.perf_counter()
    metrics = independent.read(REPORT / f"MIND_WARM_RANK{number:03d}_METRICS.json")
    independent.require(metrics["status"].startswith("completed"), "completed run required")
    checks, records = {}, {}
    with duckdb.connect() as db:
        db.execute("SET memory_limit='512MiB'")
        db.execute("SET threads=2")
        for cutoff, expected in metrics["windows"].items():
            independent.safe_cutoff(cutoff)
            folder = ART / f"MIND-WARM-RANK-{number:03d}" / cutoff
            db.execute("CREATE OR REPLACE TEMP TABLE comparisons AS SELECT * FROM read_parquet(?)", [str(folder / "comparisons.parquet")])
            db.execute("CREATE OR REPLACE TEMP TABLE decisions AS SELECT * FROM read_parquet(?)", [str(folder / "decisions.parquet")])
            columns = {r[0] for r in db.execute("DESCRIBE decisions").fetchall()}
            checks[cutoff + "_label_free_schema"] = columns == set(independent.KEYS + ["candidate_purchase_score", "victim_purchase_score", "replacement_score"])
            noop = number == 7 and not metrics["models"][cutoff]["history"]
            predicate = "FALSE" if noop else "candidate_purchase_score>victim_purchase_score"
            db.execute(f"CREATE OR REPLACE TEMP TABLE replay AS SELECT *, (candidate_purchase_score-victim_purchase_score)/champion_rank AS replacement_score FROM comparisons WHERE {predicate} QUALIFY row_number() OVER(PARTITION BY customer_id ORDER BY replacement_score DESC,challenger_article_id,victim_article_id)=1")
            error = db.execute("SELECT count(*) FROM decisions d FULL JOIN replay r USING(customer_id,challenger_article_id,victim_article_id,champion_rank) WHERE d.customer_id IS NULL OR r.customer_id IS NULL OR abs(d.replacement_score-r.replacement_score)>1e-12 OR abs(d.candidate_purchase_score-r.candidate_purchase_score)>1e-12 OR abs(d.victim_purchase_score-r.victim_purchase_score)>1e-12").fetchone()[0]
            checks[cutoff + "_independent_selection"] = error == 0
            bad = db.execute("SELECT count(*) FROM decisions WHERE champion_rank NOT BETWEEN 8 AND 12 OR replacement_score<=0").fetchone()[0]
            counts = db.execute("SELECT count(*),count(DISTINCT customer_id) FROM decisions").fetchone()
            checks[cutoff + "_protected_head_one_action"] = bad == 0 and counts[0] == counts[1]
            pairs = str(independent.ART / cutoff / "pairs.parquet")
            bad = db.execute("SELECT count(*) FROM comparisons s LEFT JOIN read_parquet(?) p USING(customer_id,challenger_article_id,victim_article_id,champion_rank) WHERE p.customer_id IS NULL OR p.v_user_item_events_12w<>0", [pairs]).fetchone()[0]
            checks[cutoff + "_fixed_nonrepurchase_risk_set"] = bad == 0
            order = str(ART / "MIND-WARM-RANK-005" / cutoff / "order.parquet")
            bad = db.execute("SELECT count(*) FROM comparisons s LEFT JOIN read_parquet(?) r ON s.customer_id=r.customer_id AND s.challenger_article_id=r.article_id WHERE r.rank IS DISTINCT FROM 1", [order]).fetchone()[0]
            checks[cutoff + "_005_top1_identity"] = bad == 0
            # Exact set identity against all eligible frozen pairs, not just saved actions.
            expected_n = db.execute("SELECT count(*) FROM read_parquet(?) p JOIN read_parquet(?) r ON p.customer_id=r.customer_id AND p.challenger_article_id=r.article_id WHERE r.rank=1 AND p.v_user_item_events_12w=0", [pairs, order]).fetchone()[0]
            counts = db.execute("SELECT count(*),count(DISTINCT(customer_id,challenger_article_id,victim_article_id,champion_rank)) FROM comparisons").fetchone()
            checks[cutoff + "_full_risk_set"] = expected_n == counts[0] == counts[1]
            if number == 7 and not noop:
                meta = metrics["models"][cutoff]
                checks[cutoff + "_calibration_chronology"] = all(date.fromisoformat(t) + timedelta(days=7) <= date.fromisoformat(cutoff) for t in meta["history"])
                frame = db.execute("SELECT s.candidate_purchase_score AS c_new,s.victim_purchase_score AS v_new,r.candidate_purchase_score AS c_old,r.victim_purchase_score AS v_old FROM comparisons s JOIN read_parquet(?) r USING(customer_id,challenger_article_id,victim_article_id,champion_rank)", [str(ART / "MIND-WARM-RANK-006" / cutoff / "comparisons.parquet")]).fetchdf()
                for pool, prefix in [("mind", "c"), ("base", "v")]:
                    m = meta["mappings"][pool]
                    raw = frame[prefix + "_old"].to_numpy()
                    pred = expit(m["slope"] * logit(np.clip(raw, 1e-8, 1 - 1e-8)) + m["intercept"])
                    checks[cutoff + "_" + pool + "_mapping"] = m["slope"] >= 0 and np.allclose(pred, frame[prefix + "_new"], atol=1e-12, rtol=0)
            base_meta = independent.read(independent.ART / cutoff / "PAIRS.json")["baseline"]
            top, arrays = independent.frozen_baseline(db, base_meta)
            db.register("top", top)
            bad = db.execute("SELECT count(*) FROM decisions d LEFT JOIN top t ON d.customer_id=t.customer_id AND d.victim_article_id=t.article_id WHERE t.champion_rank IS DISTINCT FROM d.champion_rank").fetchone()[0]
            checks[cutoff + "_exact_baseline_victims"] = bad == 0
            rows = db.execute("SELECT d.*,p.challenger_target FROM decisions d JOIN read_parquet(?) p USING(customer_id,challenger_article_id,victim_article_id,champion_rank)", [pairs]).fetchdf()
            lookup = {u: i for i, u in enumerate(arrays["users"])}
            changed = arrays["labels"].copy()
            for row in rows.itertuples():
                changed[lookup[row.customer_id], row.champion_rank - 1] = row.challenger_target
            delta = float((independent.ap_matrix(changed, arrays["truth"]) - arrays["aps"]).mean())
            checks[cutoff + "_independent_full_MAP"] = abs(delta - expected["delta_vs_baseline"]) < 1e-12
            records[cutoff] = {"delta": delta, "actions": len(rows), "users": len(arrays["users"])}
    deltas = [v["delta"] for v in records.values()]
    checks["gate_replay"] = bool(np.mean(deltas) > 0 and all(d >= 0 for d in deltas) and min(deltas) >= -.0002) == metrics["gate"]["passed"]
    checks["final_week_not_run"] = metrics["final_week"] == "not_run"
    result = {"passed": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()}, "windows": records, "seconds": time.perf_counter() - started, "new_training": False, "final_week": "not_run"}
    independent.write(REPORT / f"MIND_WARM_RANK{number:03d}_VERIFICATION.json", result)
    independent.require(result["passed"], str([k for k, v in checks.items() if not v]))
    print({"passed": result["passed"], "checks": len(checks), "seconds": result["seconds"]}, flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("number", type=int)
    verify(p.parse_args().number)
