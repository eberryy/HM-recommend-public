"""Independent replay of registered shared-supervision trials; no fitting."""
import argparse
import time
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import mind_warm_transfer as run
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as prior
from . import mind_warm_relation_audit as independent


def verify(number):
    started = time.perf_counter()
    tag = f"MIND_WARM_RANK{number:03d}"
    trial = f"MIND-WARM-RANK-{number:03d}"
    metrics = prior.read(prior.REPORT / (tag + "_METRICS.json"))
    contract = prior.read(prior.REPORT / (tag + "_CONTRACT.json"))
    candidate.require(metrics["status"].startswith("completed"), "completed run required")
    checks, windows = {}, {}
    with candidate.connection() as db:
        for cutoff, training in contract["schedule"].items():
            folder = run.ART / trial / cutoff
            db.execute("CREATE OR REPLACE TEMP TABLE truth AS SELECT * FROM read_parquet(?)", [str(candidate.ART / cutoff / "candidates.parquet")])
            db.execute("CREATE OR REPLACE TEMP TABLE ranked AS SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY score DESC,article_id) expected_rank FROM read_parquet(?)", [str(folder / "order.parquet")])
            bad = db.execute("SELECT count(*) FROM ranked r FULL JOIN truth t USING(customer_id,article_id) WHERE r.article_id IS NULL OR t.article_id IS NULL OR r.rank<>r.expected_rank OR NOT isfinite(r.score)").fetchone()[0]
            checks[cutoff + "_candidate_identity_order"] = bad == 0
            counts = db.execute("SELECT count(*),count(DISTINCT(customer_id,article_id)) FROM ranked").fetchone()
            checks[cutoff + "_unique_candidates"] = counts[0] == counts[1]
            posusers = db.execute("SELECT count(DISTINCT customer_id) FROM truth WHERE target=1").fetchone()[0]
            hit = db.execute("SELECT count(DISTINCT customer_id) FROM ranked JOIN truth USING(customer_id,article_id) WHERE target=1 AND rank=1").fetchone()[0]
            checks[cutoff + "_Hit1"] = abs(hit / posusers - metrics["windows"][cutoff]["internal"]["conditional_hit_at_1"]) < 1e-12
            meta = metrics["models"][cutoff]
            checks[cutoff + "_time_boundary"] = meta["training"] == training and all(date.fromisoformat(t) + timedelta(days=7) <= date.fromisoformat(cutoff) for t in training)
            model = lgb.Booster(model_file=meta["model"])
            checks[cutoff + "_model_allowlist"] = model.feature_name() == candidate.FEATURES and not set(model.feature_name()).intersection(independent.FORBIDDEN_FEATURES)
            checks[cutoff + "_fixed_rounds"] = model.current_iteration() == 150
            frame = db.execute(f"SELECT {','.join('t.' + f for f in candidate.FEATURES)},r.score saved_score FROM truth t JOIN ranked r USING(customer_id,article_id)").fetchdf()
            x = frame[candidate.FEATURES].to_numpy(np.float32)
            x[~np.isfinite(x)] = np.nan
            checks[cutoff + "_model_replay"] = np.allclose(model.predict(x, num_threads=2), frame.saved_score, atol=1e-12, rtol=0)
            decisions = db.execute("SELECT * FROM read_parquet(?)", [str(folder / "decisions.parquet")]).fetchdf()
            checks[cutoff + "_decision_schema"] = set(decisions.columns) == set(prior.KEYS + prior.PROBS + ["replacement_score"])
            checks[cutoff + "_head_protection"] = decisions.champion_rank.between(8, 12).all() and not decisions.customer_id.duplicated().any()
            db.register("decisions", decisions)
            checks[cutoff + "_top1_only"] = db.execute("SELECT count(*) FROM decisions d LEFT JOIN ranked r ON d.customer_id=r.customer_id AND d.challenger_article_id=r.article_id WHERE r.rank IS DISTINCT FROM 1").fetchone()[0] == 0
            # Independently reconstruct the five-victim comparison and rejection.
            fields = list(dict.fromkeys(prior.KEYS + prior.FEATURES["dense_primary"]))
            pairs = db.execute(f"SELECT {','.join('p.' + f for f in fields)} FROM read_parquet(?) p JOIN ranked r ON p.customer_id=r.customer_id AND p.challenger_article_id=r.article_id WHERE r.rank=1", [str(prior.ART / cutoff / "pairs.parquet")]).fetchdf()
            relation_model = lgb.Booster(model_file=str(prior.model_path(training, "dense_primary")))
            px = pairs[prior.FEATURES["dense_primary"]].to_numpy(np.float32)
            px[~np.isfinite(px)] = np.nan
            probabilities = relation_model.predict(px, num_threads=2)
            replay = pairs[prior.KEYS].copy()
            replay[prior.PROBS] = probabilities
            replay["replacement_score"] = (probabilities[:, 2] - probabilities[:, 0]) / replay.champion_rank
            replay = replay.loc[replay.replacement_score > 0].sort_values(["customer_id", "replacement_score", "challenger_article_id", "victim_article_id"], ascending=[True, False, True, True]).drop_duplicates("customer_id")
            pd.testing.assert_frame_equal(replay.sort_values(prior.KEYS).reset_index(drop=True), decisions.sort_values(prior.KEYS).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=0)
            checks[cutoff + "_admission_replay"] = True
            base_meta = prior.read(prior.ART / cutoff / "PAIRS.json")["baseline"]
            top, arrays = independent.frozen_baseline(db, base_meta)
            db.register("baseline", top)
            badvictims = db.execute("SELECT count(*) FROM decisions d LEFT JOIN baseline b ON d.customer_id=b.customer_id AND d.victim_article_id=b.article_id WHERE b.champion_rank IS DISTINCT FROM d.champion_rank").fetchone()[0]
            checks[cutoff + "_correct_victims"] = badvictims == 0
            ev = db.execute("SELECT d.*,t.target actual_y FROM decisions d JOIN truth t ON d.customer_id=t.customer_id AND d.challenger_article_id=t.article_id").fetchdf()
            indexes = {u: i for i, u in enumerate(arrays["users"])}
            labels = arrays["labels"].copy()
            for action in ev.itertuples():
                labels[indexes[action.customer_id], action.champion_rank - 1] = action.actual_y
            delta = float((independent.ap_matrix(labels, arrays["truth"]) - arrays["aps"]).mean())
            checks[cutoff + "_independent_full_MAP"] = abs(delta - metrics["windows"][cutoff]["admission"]["delta_vs_baseline"]) < 1e-12
            windows[cutoff] = {"delta": delta, "selected_users": len(decisions), "positive_first_choices": hit, "positive_candidate_users": posusers}
        for cutoff, meta in metrics["inputs"].items():
            path = run.ART / "shared" / cutoff / "features.parquet"
            original = run.source(cutoff)
            errors = db.execute("SELECT count(*) FROM read_parquet(?) s FULL JOIN (SELECT customer_id,article_id,target,CASE WHEN mind_is_new=1 THEN 'mind_new' ELSE 'base_novel' END pool FROM read_parquet(?) WHERE mind_is_new=1 OR user_item_events_12w=0) o USING(customer_id,article_id) WHERE s.article_id IS NULL OR o.article_id IS NULL OR s.target<>o.target OR s.pool<>o.pool", [str(path), str(original)]).fetchone()[0]
            checks[cutoff + "_training_rows_labels_full_conservation"] = errors == 0
    values = [v["delta"] for v in windows.values()]
    passed = np.mean(values) > 0 and all(v >= 0 for v in values) and min(values) >= -.0002
    checks["gate_replayed"] = bool(passed) == metrics["gate"]["passed"]
    checks["final_week_not_run"] = metrics["final_week"] == "not_run"
    result = {"passed": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()}, "windows": windows, "seconds": time.perf_counter() - started, "training": "none", "final_week": "not_run"}
    prior.save(prior.REPORT / (tag + "_VERIFICATION.json"), result)
    candidate.require(result["passed"], "independent checks failed: " + str([k for k, v in checks.items() if not v]))
    print({"passed": result["passed"], "checks": len(checks), "windows": windows, "seconds": result["seconds"]}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("number", type=int)
    verify(parser.parse_args().number)
