"""Independent replay of008 ridge and009 exact-action consensus."""
import argparse
import time
from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from . import mind_warm_relation_audit as old

ART = old.ROOT / "artifacts/mind_warm_side/MIND-WARM-TRANSFER"


def verify(number):
    started = time.perf_counter()
    m = old.read(old.REPORT / f"MIND_WARM_RANK{number:03d}_METRICS.json")
    old.require(m["status"].startswith("completed"), "completed run required")
    checks, records = {}, {}
    with duckdb.connect() as db:
        db.execute("SET memory_limit='512MiB'")
        db.execute("SET threads=2")
        for cutoff, expected in m["windows"].items():
            old.safe_cutoff(cutoff)
            folder = ART / f"MIND-WARM-RANK-{number:03d}" / cutoff
            actions = db.execute("SELECT * FROM read_parquet(?)", [str(folder / "decisions.parquet")]).fetchdf()
            db.register("actions", actions)
            checks[cutoff + "_one_action_protected_head"] = not actions.customer_id.duplicated().any() and actions.champion_rank.between(8, 12).all()
            if number == 10:
                scores = db.execute("SELECT * FROM read_parquet(?)", [str(folder / "scores.parquet")]).fetchdf()
                db.register("scores", scores)
                checks[cutoff + "_label_free_schema"] = set(scores.columns) == set(old.KEYS + old.PROBS)
                replay = db.execute("SELECT *, (p_benefit-p_harm)/champion_rank AS replacement_score FROM scores WHERE p_benefit>p_harm QUALIFY row_number() OVER(PARTITION BY customer_id ORDER BY replacement_score DESC,challenger_article_id,victim_article_id)=1").fetchdf()
                pd.testing.assert_frame_equal(replay.sort_values(old.KEYS).reset_index(drop=True), actions.sort_values(old.KEYS).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=0)
                checks[cutoff + "_independent_selection"] = True
                bad = db.execute("SELECT count(*) FROM scores s LEFT JOIN read_parquet(?) r ON s.customer_id=r.customer_id AND s.challenger_article_id=r.article_id WHERE r.rank IS NULL OR r.rank>5", [str(ART / "MIND-WARM-RANK-005" / cutoff / "order.parquet")]).fetchone()[0]
                expected_count = db.execute("SELECT count(*)*5 FROM read_parquet(?) WHERE rank<=5", [str(ART / "MIND-WARM-RANK-005" / cutoff / "order.parquet")]).fetchone()[0]
                checks[cutoff + "_exact_Top5_times5"] = bad == 0 and len(scores) == expected_count and not scores.duplicated(old.KEYS).any()
                import lightgbm as lgb
                from . import mind_warm_relation as relation
                contract = old.read(old.REPORT / "MIND_WARM_RANK010_CONTRACT.json")
                training = contract["schedule"][cutoff]
                checks[cutoff + "_chronology"] = all(date.fromisoformat(t) + timedelta(days=7) <= date.fromisoformat(cutoff) for t in training)
                fields = relation.FEATURES["dense_primary"]
                frame = db.execute(f"SELECT s.*, {','.join('p.' + f for f in fields if f not in old.KEYS)} FROM scores s JOIN read_parquet(?) p USING(customer_id,challenger_article_id,victim_article_id,champion_rank)", [str(old.ART / cutoff / "pairs.parquet")]).fetchdf()
                matrix = frame[fields].to_numpy(np.float32)
                matrix[~np.isfinite(matrix)] = np.nan
                model = lgb.Booster(model_file=str(relation.model_path(training, "dense_primary")))
                checks[cutoff + "_frozen_relation_score_replay"] = np.allclose(model.predict(matrix, num_threads=2), frame[old.PROBS], atol=1e-12, rtol=0)
            elif number == 9:
                replay = db.execute("SELECT a.* FROM read_parquet(?) a JOIN read_parquet(?) b USING(customer_id,challenger_article_id,victim_article_id,champion_rank)", [str(ART / "MIND-WARM-RANK-005" / cutoff / "decisions.parquet"), str(ART / "MIND-WARM-RANK-006" / cutoff / "decisions.parquet")]).fetchdf()
                pd.testing.assert_frame_equal(replay.sort_values(old.KEYS).reset_index(drop=True), actions.sort_values(old.KEYS).reset_index(drop=True), check_dtype=False, check_exact=True)
                checks[cutoff + "_exact_frozen_action_intersection"] = True
                checks[cutoff + "_label_free_schema"] = set(actions.columns) == set(old.KEYS + old.PROBS + ["replacement_score"])
            else:
                meta = m["models"][cutoff]
                x = db.execute("SELECT * FROM read_parquet(?)", [str(folder / "features.parquet")]).fetchdf()
                scores = db.execute("SELECT * FROM read_parquet(?)", [str(folder / "scores.parquet")]).fetchdf()
                checks[cutoff + "_label_free_schema"] = set(actions.columns) == set(old.KEYS + ["replacement_score"]) == set(scores.columns)
                db.register("scores", scores)
                replay = db.execute("SELECT * FROM scores WHERE replacement_score>0 QUALIFY row_number() OVER(PARTITION BY customer_id ORDER BY replacement_score DESC,challenger_article_id,victim_article_id)=1").fetchdf()
                pd.testing.assert_frame_equal(replay.sort_values(old.KEYS).reset_index(drop=True), actions.sort_values(old.KEYS).reset_index(drop=True), check_dtype=False, check_exact=True)
                checks[cutoff + "_independent_best_positive_selection"] = True
                if not meta["history"]:
                    checks[cutoff + "_warmup_noop"] = actions.empty and (scores.replacement_score == 0).all()
                else:
                    model = meta["model"]
                    features = model["features"]
                    checks[cutoff + "_chronology"] = all(date.fromisoformat(t) + timedelta(days=7) <= date.fromisoformat(cutoff) for t in meta["history"])
                    checks[cutoff + "_eight_features_no_labels"] = len(features) == 8 and not set(features).intersection(old.FORBIDDEN_FEATURES)
                    train = db.execute("SELECT * FROM read_parquet(?)", [str(folder / "training.parquet")]).fetchdf()
                    xtrain = train[features].to_numpy(float)
                    mean, std = xtrain.mean(0), xtrain.std(0)
                    std[std == 0] = 1
                    checks[cutoff + "_training_only_scaling"] = np.allclose(mean, model["mean"], atol=1e-12, rtol=0) and np.allclose(std, model["scale"], atol=1e-12, rtol=0)
                    centered = (xtrain - mean) / std
                    y = train.actual_delta.to_numpy(float)
                    # Augmented least squares is independent of normal-equation fit.
                    coefficients = np.linalg.lstsq(np.vstack([centered, np.eye(8)]), np.r_[y - y.mean(), np.zeros(8)], rcond=None)[0]
                    checks[cutoff + "_independent_ridge_solution"] = np.allclose(coefficients, model["coefficients"], atol=1e-10, rtol=0) and abs(y.mean() - model["intercept"]) < 1e-12
                    predictions = ((x[features].to_numpy(float) - mean) / std) @ coefficients + y.mean()
                    replay_scores = x[old.KEYS].copy()
                    replay_scores["prediction"] = predictions
                    joined = scores.merge(replay_scores, on=old.KEYS, validate="one_to_one")
                    checks[cutoff + "_model_prediction_replay"] = len(joined) == len(x) == len(scores) and np.allclose(joined.replacement_score, joined.prediction, atol=1e-10, rtol=0)
                    # Reconstruct the full historical supervision, with no model call.
                    originals = []
                    for history in meta["history"]:
                        prior_frame = db.execute("SELECT * FROM read_parquet(?)", [str(ART / "MIND-WARM-RANK-008" / history / "features.parquet")]).fetchdf()
                        bmeta = old.read(old.ART / history / "PAIRS.json")["baseline"]
                        _, arrays = old.frozen_baseline(db, bmeta)
                        db.register("prior_frame", prior_frame)
                        labels = db.execute("SELECT f.*,p.challenger_target FROM prior_frame f JOIN read_parquet(?) p USING(customer_id,challenger_article_id,victim_article_id,champion_rank)", [str(old.ART / history / "pairs.parquet")]).fetchdf()
                        idx = pd.Index(arrays["users"]).get_indexer(labels.customer_id)
                        before = arrays["labels"][idx]
                        after = before.copy()
                        after[np.arange(len(after)), labels.champion_rank.to_numpy(int) - 1] = labels.challenger_target
                        labels["actual_delta"] = old.ap_matrix(after, arrays["truth"][idx]) - old.ap_matrix(before, arrays["truth"][idx])
                        originals.append(labels[train.columns])
                    reconstructed = pd.concat(originals, ignore_index=True)
                    pd.testing.assert_frame_equal(train.sort_values(old.KEYS + ["actual_delta"]).reset_index(drop=True), reconstructed.sort_values(old.KEYS + ["actual_delta"]).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=0)
                    checks[cutoff + "_exact_historical_target_conservation"] = True
            base_meta = old.read(old.ART / cutoff / "PAIRS.json")["baseline"]
            top, arrays = old.frozen_baseline(db, base_meta)
            db.register("baseline", top)
            bad = db.execute("SELECT count(*) FROM actions a LEFT JOIN baseline b ON a.customer_id=b.customer_id AND a.victim_article_id=b.article_id WHERE b.champion_rank IS DISTINCT FROM a.champion_rank").fetchone()[0]
            checks[cutoff + "_exact_original_victims"] = bad == 0
            labeled = db.execute("SELECT a.*,p.challenger_target FROM actions a JOIN read_parquet(?) p USING(customer_id,challenger_article_id,victim_article_id,champion_rank)", [str(old.ART / cutoff / "pairs.parquet")]).fetchdf()
            idx = pd.Index(arrays["users"]).get_indexer(labeled.customer_id)
            after = arrays["labels"].copy()
            if len(labeled):
                after[idx, labeled.champion_rank.to_numpy(int) - 1] = labeled.challenger_target
            delta = float((old.ap_matrix(after, arrays["truth"]) - arrays["aps"]).mean())
            checks[cutoff + "_independent_full_MAP"] = abs(delta - expected["delta_vs_baseline"]) < 1e-12
            records[cutoff] = {"delta": delta, "actions": len(actions)}
    values = [v["delta"] for v in records.values()]
    checks["gate_replay"] = bool(np.mean(values) > 0 and all(v >= 0 for v in values) and min(values) >= -.0002) == m["gate"]["passed"]
    checks["final_week_not_run"] = m["final_week"] == "not_run"
    out = {"passed": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()}, "windows": records, "seconds": time.perf_counter() - started, "training": "no recommendation training; algebraic coefficient verification", "final_week": "not_run"}
    old.write(old.REPORT / f"MIND_WARM_RANK{number:03d}_VERIFICATION.json", out)
    old.require(out["passed"], str([k for k, v in checks.items() if not v]))
    print({"passed": out["passed"], "checks": len(checks), "seconds": out["seconds"]}, flush=True)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("number", type=int)
    verify(p.parse_args().number)
