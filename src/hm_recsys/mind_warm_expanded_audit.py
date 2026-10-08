"""Independent replay of011 expanded ranking and Top5 relation admission."""
import time
from datetime import date, timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import mind_warm_expanded as run
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation
from . import mind_warm_relation_audit as independent


def verify(number=11):
    started = time.perf_counter()
    metrics = relation.read(relation.REPORT / f"MIND_WARM_RANK{number:03d}_METRICS.json")
    contract = relation.read(relation.REPORT / f"MIND_WARM_RANK{number:03d}_CONTRACT.json")
    candidate.require(metrics["status"].startswith("completed"), f"completed{number:03d} required")
    artifact_root = run.transfer.ART / f"MIND-WARM-RANK-{number:03d}"
    checks, windows = {}, {}
    with candidate.connection() as db:
        db.execute("SET memory_limit='2GiB'")
        for cutoff, training in contract["schedule"].items():
            folder = artifact_root / cutoff
            order = folder / "order.parquet"
            cand = candidate.ART / cutoff / "candidates.parquet"
            db.execute("CREATE OR REPLACE TEMP TABLE truth AS SELECT * FROM read_parquet(?)", [str(cand)])
            db.execute("CREATE OR REPLACE TEMP TABLE ranked AS SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY score DESC,article_id) expected_rank FROM read_parquet(?)", [str(order)])
            bad = db.execute("SELECT count(*) FROM ranked r FULL JOIN truth t USING(customer_id,article_id) WHERE r.article_id IS NULL OR t.article_id IS NULL OR r.rank<>r.expected_rank OR NOT isfinite(r.score)").fetchone()[0]
            checks[cutoff + "_candidate_identity_order"] = bad == 0
            model_meta = metrics["models"][cutoff]
            model = lgb.Booster(model_file=model_meta["model"])
            checks[cutoff + "_feature_allowlist"] = model.feature_name() == run.FEATURES and not set(model.feature_name()).intersection(independent.FORBIDDEN_FEATURES)
            checks[cutoff + "_fixed_model"] = model.current_iteration() == 150 and model_meta["training"] == training
            checks[cutoff + "_chronology"] = all(date.fromisoformat(t) + timedelta(days=7) <= date.fromisoformat(cutoff) for t in training)
            source = run.transfer.source(cutoff)
            frame = db.execute(f"SELECT c.*,{','.join('s.' + f for f in run.EXTRA)},r.score saved FROM truth c JOIN read_parquet(?) s USING(customer_id,article_id) JOIN ranked r USING(customer_id,article_id)", [str(source)]).fetchdf()
            x = frame[run.FEATURES].to_numpy(np.float32)
            x[~np.isfinite(x)] = np.nan
            checks[cutoff + "_model_score_replay"] = np.allclose(model.predict(x, num_threads=2), frame.saved, atol=1e-12, rtol=0)
            posusers = db.execute("SELECT count(DISTINCT customer_id) FROM truth WHERE target=1").fetchone()[0]
            hit1, hit5 = db.execute("SELECT count(DISTINCT customer_id) FILTER(WHERE target=1 AND rank=1),count(DISTINCT customer_id) FILTER(WHERE target=1 AND rank<=5) FROM ranked JOIN truth USING(customer_id,article_id)").fetchone()
            expected_internal = metrics["windows"][cutoff]["internal"]
            checks[cutoff + "_candidate_metrics"] = hit1 == expected_internal["hit_users_at_1"] and hit5 == expected_internal["hit_users_at_5"] and abs(hit1 / posusers - expected_internal["conditional_hit_at_1"]) < 1e-12
            scores = db.execute("SELECT * FROM read_parquet(?)", [str(folder / "admission-scores.parquet")]).fetchdf()
            actions = db.execute("SELECT * FROM read_parquet(?)", [str(folder / "decisions.parquet")]).fetchdf()
            checks[cutoff + "_label_free_schemas"] = set(scores.columns) == set(relation.KEYS + relation.PROBS) and set(actions.columns) == set(relation.KEYS + relation.PROBS + ["replacement_score"])
            db.register("saved_scores", scores)
            bad = db.execute("SELECT count(*) FROM saved_scores s LEFT JOIN ranked r ON s.customer_id=r.customer_id AND s.challenger_article_id=r.article_id WHERE r.rank IS NULL OR r.rank>5").fetchone()[0]
            expected_rows = db.execute("SELECT count(*)*5 FROM ranked WHERE rank<=5").fetchone()[0]
            checks[cutoff + "_exact_Top5_times5"] = bad == 0 and len(scores) == expected_rows and not scores.duplicated(relation.KEYS).any()
            fields = relation.FEATURES["dense_primary"]
            pairs = db.execute(f"SELECT s.*,{','.join('p.' + f for f in fields if f not in relation.KEYS)} FROM saved_scores s JOIN read_parquet(?) p USING(customer_id,challenger_article_id,victim_article_id,champion_rank)", [str(relation.ART / cutoff / "pairs.parquet")]).fetchdf()
            px = pairs[fields].to_numpy(np.float32)
            px[~np.isfinite(px)] = np.nan
            old_model = lgb.Booster(model_file=str(relation.model_path(training, "dense_primary")))
            checks[cutoff + "_frozen_relation_replay"] = np.allclose(old_model.predict(px, num_threads=2), pairs[relation.PROBS], atol=1e-12, rtol=0)
            replay = relation.select_actions(scores)
            pd.testing.assert_frame_equal(replay.sort_values(relation.KEYS).reset_index(drop=True), actions.sort_values(relation.KEYS).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=0)
            checks[cutoff + "_action_replay"] = True
            base_meta = relation.read(relation.ART / cutoff / "PAIRS.json")["baseline"]
            top, arrays = independent.frozen_baseline(db, base_meta)
            db.register("baseline", top)
            bad = db.execute("SELECT count(*) FROM read_parquet(?) a LEFT JOIN baseline b ON a.customer_id=b.customer_id AND a.victim_article_id=b.article_id WHERE b.champion_rank IS DISTINCT FROM a.champion_rank", [str(folder / "decisions.parquet")]).fetchone()[0]
            checks[cutoff + "_original_victim_head_protection"] = bad == 0 and actions.champion_rank.between(8, 12).all() and not actions.customer_id.duplicated().any()
            labels = db.execute("SELECT customer_id,article_id,target FROM truth").fetchdf()
            evaluated = actions.merge(labels.rename(columns={"article_id": "challenger_article_id", "target": "challenger_target"}), on=["customer_id", "challenger_article_id"], validate="one_to_one")
            idx = pd.Index(arrays["users"]).get_indexer(evaluated.customer_id)
            altered = arrays["labels"].copy()
            altered[idx, evaluated.champion_rank.to_numpy(int) - 1] = evaluated.challenger_target.to_numpy(np.uint8)
            delta = float((independent.ap_matrix(altered, arrays["truth"]) - arrays["aps"]).mean())
            expected_delta = metrics["windows"][cutoff]["admission"]["delta_vs_baseline"]
            checks[cutoff + "_independent_full_MAP"] = abs(delta - expected_delta) < 1e-12
            windows[cutoff] = {"hit1": hit1, "hit5": hit5, "actions": len(actions), "delta": delta}
        for cutoff, meta in metrics.get("inputs", {}).items():
            path = run.ART / "input" / cutoff / "features.parquet"
            shared = run.transfer.ART / "shared" / cutoff / "features.parquet"
            errors = db.execute("SELECT count(*) FROM read_parquet(?) e FULL JOIN read_parquet(?) s USING(customer_id,article_id) WHERE e.article_id IS NULL OR s.article_id IS NULL OR e.pool<>s.pool OR e.target<>s.target", [str(path), str(shared)]).fetchone()[0]
            checks[cutoff + "_training_population_label_conservation"] = errors == 0 and meta["rows"]["row_count"] == meta["rows"]["unique_rows"]
    values = [v["delta"] for v in windows.values()]
    gate = np.mean(values) > 0 and all(v >= 0 for v in values) and min(values) >= -.0002
    checks["gate_replay"] = bool(gate) == metrics["gate"]["passed"]
    checks["final_week_not_run"] = metrics["final_week"] == "not_run"
    result = {"passed": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()}, "windows": windows,
              "seconds": time.perf_counter() - started, "new_training": False, "final_week": "not_run"}
    relation.save(relation.REPORT / f"MIND_WARM_RANK{number:03d}_VERIFICATION.json", result)
    candidate.require(result["passed"], str([k for k, v in checks.items() if not v]))
    print({"passed": result["passed"], "checks": len(checks), "windows": windows, "seconds": result["seconds"]}, flush=True)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("number", nargs="?", type=int, default=11)
    verify(parser.parse_args().number)
