"""Independent SQL checks of MIND-003 candidate orders and optional admission."""
import json
import time
from datetime import date, timedelta

import duckdb
import lightgbm as lgb
import numpy as np

from . import mind_warm_candidate_ranker as run
from . import mind_warm_relation as prior
from . import mind_warm_relation_audit as independent


def verify():
    started = time.perf_counter()
    m = prior.read(run.METRICS)
    c = prior.read(run.CONTRACT)
    if not m["status"].startswith("completed"):
        raise RuntimeError("completed003 run required")
    checks = {}
    records = {}
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='2GiB'")
        for cutoff, training in c["schedule"].items():
            folder = run.ART / cutoff
            path = folder / "candidates.parquet"
            db.execute("CREATE OR REPLACE TEMP TABLE truth AS SELECT * FROM read_parquet(?)", [str(path)])
            db.execute("CREATE OR REPLACE TEMP TABLE original AS SELECT customer_id,challenger_article_id AS article_id,challenger_target AS target FROM read_parquet(?) WHERE champion_rank=8", [str(prior.ART / cutoff / "pairs.parquet")])
            mismatch = db.execute("SELECT count(*) FROM truth t FULL JOIN original o USING(customer_id,article_id) WHERE t.article_id IS NULL OR o.article_id IS NULL OR t.target<>o.target").fetchone()[0]
            checks[cutoff + "_input_identity"] = mismatch == 0
            rows, n_positive, positive_users = db.execute("SELECT count(*),sum(target),count(DISTINCT customer_id) FILTER(WHERE target=1) FROM truth").fetchone()
            records[cutoff] = {}
            for ordering in run.ORDERINGS:
                opath = folder / f"{ordering}-order.parquet"
                columns = [x[0] for x in db.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(opath)]).fetchall()]
                checks[f"{cutoff}_{ordering}_unlabeled_schema"] = set(columns) == {"customer_id", "article_id", "score", "rank"}
                db.execute("CREATE OR REPLACE TEMP TABLE ranked AS SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY score DESC,article_id) AS expected_rank FROM read_parquet(?)", [str(opath)])
                errors = db.execute("SELECT count(*) FROM ranked r FULL JOIN truth t USING(customer_id,article_id) WHERE r.article_id IS NULL OR t.article_id IS NULL OR r.rank<>r.expected_rank OR NOT isfinite(r.score)").fetchone()[0]
                unique, saved_rows = db.execute("SELECT count(DISTINCT(customer_id,article_id)),count(*) FROM ranked").fetchone()
                checks[f"{cutoff}_{ordering}_identity_and_order"] = errors == 0 and unique == rows == saved_rows
                if ordering == "raw_mind":
                    errors = db.execute("SELECT count(*) FROM ranked r JOIN read_parquet(?) s USING(customer_id,article_id) WHERE r.score<>-s.mind_rank", [str(prior.old._mind_candidates(cutoff))]).fetchone()[0]
                    checks[cutoff + "_original_mind_score_replay"] = errors == 0
                numbers = {"positive_candidate_users": positive_users, "positive_candidate_rows": n_positive, "candidate_rows": rows}
                total = m["inputs"][cutoff]["total_users"]
                for k in (1, 5):
                    hit_users, hit_pairs = db.execute("SELECT count(DISTINCT customer_id),count(*) FROM ranked JOIN truth USING(customer_id,article_id) WHERE target=1 AND rank<=?", [k]).fetchone()
                    numbers.update({f"hit_users_at_{k}": hit_users, f"conditional_hit_at_{k}": hit_users / positive_users,
                                    f"all_user_hit_at_{k}": hit_users / total, f"positive_pair_recall_at_{k}": hit_pairs / n_positive})
                reciprocal = db.execute("SELECT sum(1.0/r) FROM (SELECT customer_id,min(rank) r FROM ranked JOIN truth USING(customer_id,article_id) WHERE target=1 GROUP BY customer_id)").fetchone()[0]
                numbers["conditional_MRR"] = reciprocal / positive_users
                numbers["all_user_MRR"] = reciprocal / total
                comparisons = [abs(value - m["internal"][cutoff][ordering][key]) for key, value in numbers.items()]
                checks[f"{cutoff}_{ordering}_independent_metrics"] = max(comparisons) <= 1e-12
                records[cutoff][ordering] = numbers
            meta = m["models"][cutoff]
            checks[cutoff + "_training_time"] = meta["training"] == training and all(date.fromisoformat(t) + timedelta(days=7) <= date.fromisoformat(cutoff) for t in training)
            model = lgb.Booster(model_file=meta["model"])
            checks[cutoff + "_model_features"] = model.feature_name() == run.FEATURES and not (set(model.feature_name()) & {"target", "truth_count", "victim_target", "relation_label"})
            checks[cutoff + "_fixed_rounds"] = model.current_iteration() == c["model"]["rounds"] and meta["config"] == c["model"]
            # Re-predict the small complete candidate matrix, independently of ranking.
            frame = db.execute(f"SELECT {','.join(run.KEYS + run.FEATURES)},r.score AS saved_score FROM truth t JOIN read_parquet(?) r USING(customer_id,article_id)", [str(folder / "candidate_ranker-order.parquet")]).fetchdf()
            matrix = frame[run.FEATURES].to_numpy(np.float32)
            matrix[~np.isfinite(matrix)] = np.nan
            predicted = model.predict(matrix, num_threads=2)
            checks[cutoff + "_model_score_replay"] = np.allclose(predicted, frame.saved_score, atol=1e-12, rtol=0)
            if m["admission_status"] == "completed":
                original_meta = prior.read(prior.ART / cutoff / "PAIRS.json")
                _, arrays = independent.frozen_baseline(db, original_meta["baseline"])
                decisions = db.execute("SELECT * FROM read_parquet(?)", [str(folder / "admission-decisions.parquet")]).fetchdf()
                checks[cutoff + "_admission_schema"] = set(decisions.columns) == set(prior.KEYS + prior.PROBS + ["replacement_score"])
                checks[cutoff + "_protected_head"] = decisions.champion_rank.between(8, 12).all() and not decisions.customer_id.duplicated().any()
                db.register("decisions", decisions)
                errors = db.execute("SELECT count(*) FROM decisions d LEFT JOIN read_parquet(?) r ON d.customer_id=r.customer_id AND d.challenger_article_id=r.article_id WHERE r.rank IS DISTINCT FROM 1", [str(folder / "candidate_ranker-order.parquet")]).fetchone()[0]
                checks[cutoff + "_winner_frozen_before_admission"] = errors == 0
                evaluated = db.execute("SELECT d.*,t.target AS actual_y FROM decisions d JOIN truth t ON d.customer_id=t.customer_id AND d.challenger_article_id=t.article_id ORDER BY d.customer_id").fetchdf()
                idx = {u: i for i, u in enumerate(arrays["users"])}
                labels = arrays["labels"].copy()
                before = independent.ap_matrix(labels, arrays["truth"])
                for a in evaluated.itertuples():
                    labels[idx[a.customer_id], a.champion_rank - 1] = a.actual_y
                delta = float((independent.ap_matrix(labels, arrays["truth"]) - before).mean())
                checks[cutoff + "_admission_independent_delta"] = abs(delta - m["admission"][cutoff]["delta_vs_baseline"]) <= 1e-12
    gate = run.internal_gate(records)
    checks["gate_exactly_reproduced"] = gate == m["internal_gate"]
    checks["conditional_execution"] = bool(gate["passed"]) == (m["admission_status"] == "completed")
    checks["final_week_not_run"] = m["final_week"] == "not_run"
    report = {"schema": "mind-warm-rank003-independent-verification-v1", "passed": bool(all(checks.values())),
              "checks": {k: bool(v) for k, v in checks.items()}, "windows": records, "gate": gate,
              "training": "none in verification", "final_week": "not_run", "elapsed_seconds": time.perf_counter() - started}
    prior.save(run.REPORT / "MIND_WARM_RANK003_VERIFICATION.json", report)
    run.require(report["passed"], "003 independent verification failed: " + str([k for k, v in checks.items() if not v]))
    print(json.dumps({"passed": report["passed"], "checks": len(checks), "seconds": report["elapsed_seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    verify()
