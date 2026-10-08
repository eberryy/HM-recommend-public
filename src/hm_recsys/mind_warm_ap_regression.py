"""P014 direct expected AP-delta regression over the frozen 005 Top5 action space."""
from __future__ import annotations

import gc
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_hard_relation as hard
from . import mind_warm_relation as relation
from . import mind_warm_transfer as transfer
from .mind_warm_baseline import reconstruct_full_baseline
from .mind_warm_hard_relation_audit import action_delta_tables


TRIAL = "MIND-WARM-RANK-014"
ART = transfer.ART / TRIAL
FEATURES = hard.FEATURES
SCALE = 10000.0


def labeled_input(cutoff: str):
    folder = ART / "input" / cutoff
    path, marker = folder / "pairs.parquet", folder / "INPUT.json"
    if path.exists() and marker.exists():
        meta = relation.read(marker)
        candidate.require(meta["features"] == FEATURES and meta["cutoff"] == cutoff, "stale014 input")
        return path, meta
    started = time.perf_counter()
    transfer.budget()
    folder.mkdir(parents=True, exist_ok=True)
    source, source_meta = hard.input_table(cutoff)
    baseline, baseline_meta = reconstruct_full_baseline(cutoff)
    users, labels, truth, up, down = action_delta_tables(baseline)
    with candidate.connection() as db:
        frame = db.execute(f"SELECT {','.join(relation.KEYS + FEATURES)} FROM read_parquet(?) ORDER BY customer_id,challenger_article_id,champion_rank", [str(source)]).fetchdf()
        label_frame = db.execute("SELECT customer_id,challenger_article_id,victim_article_id,champion_rank,challenger_target,victim_target FROM read_parquet(?)", [str(relation.ART / cutoff / "pairs.parquet")]).fetchdf()
    frame = frame.merge(label_frame, on=relation.KEYS, how="left", validate="one_to_one")
    candidate.require(not frame[["challenger_target", "victim_target"]].isna().any().any(), "014 training labels missing")
    row = users.get_indexer(frame.customer_id)
    pos = frame.champion_rank.to_numpy(int) - 1
    cy = frame.challenger_target.to_numpy(np.int8); vy = frame.victim_target.to_numpy(np.int8)
    candidate.require((row >= 0).all() and np.array_equal(labels[row, pos], vy), "014 victim/baseline mismatch")
    frame["exact_AP_delta_scaled"] = np.where((cy == 1) & (vy == 0), up[row, pos], np.where((cy == 0) & (vy == 1), down[row, pos], 0.)) * SCALE
    relation.parquet(frame[relation.KEYS + FEATURES + ["exact_AP_delta_scaled"]], path)
    values = frame.exact_AP_delta_scaled.to_numpy(float) / SCALE
    meta = {"cutoff": cutoff, "source": str(source), "source_shortlist": source_meta["shortlist"], "features": FEATURES,
            "rows": len(frame), "users": int(frame.customer_id.nunique()), "positive_delta_rows": int((values > 1e-15).sum()),
            "negative_delta_rows": int((values < -1e-15).sum()), "zero_delta_rows": int((np.abs(values) <= 1e-15).sum()),
            "maximum_training_oracle_MAP_delta": float(pd.Series(values).groupby(frame.customer_id).max().clip(lower=0).sum() / baseline_meta["total_users"]),
            "labels_strictly_historical_when_used_for_later_cutoff": True, "final_week": "not_run", "seconds": time.perf_counter() - started}
    relation.save(marker, meta)
    relation.progress(f"014 input {cutoff}:{len(frame):,} actions,+{meta['positive_delta_rows']}/-{meta['negative_delta_rows']}")
    return path, meta


def fit(training: list[str]):
    folder = ART / "models" / ("through_" + training[-1])
    path, marker = folder / "model.txt", folder / "MODEL.json"
    if path.exists() and marker.exists():
        meta = relation.read(marker)
        candidate.require(meta["training"] == training and meta["features"] == FEATURES, "stale014 model")
        return lgb.Booster(model_file=str(path)), meta
    started = time.perf_counter()
    hardware = transfer.budget()
    candidate.require(hardware["available_ram_gib"] >= 6, "014 requires at least 6 GiB free RAM")
    folder.mkdir(parents=True, exist_ok=True)
    with candidate.connection() as db:
        union = " UNION ALL ".join(f"SELECT * FROM read_parquet('{relation.old._sql_path(ART / 'input' / c / 'pairs.parquet')}')" for c in training)
        frame = db.execute(f"SELECT exact_AP_delta_scaled,{','.join(FEATURES)} FROM ({union})").fetchdf()
    candidate.require(len(frame) <= 1000000, "014 row cap exceeded")
    y = frame.exact_AP_delta_scaled.to_numpy(np.float32)
    x = frame[FEATURES].to_numpy(np.float32); x[~np.isfinite(x)] = np.nan
    summary = {"rows": len(frame), "positive": int((y > 0).sum()), "negative": int((y < 0).sum()), "zero": int((y == 0).sum()),
               "mean_scaled_target": float(y.mean()), "target_scale": SCALE}
    del frame; gc.collect()
    params = {"objective": "regression", "metric": "l2", "learning_rate": .03, "num_leaves": 15, "max_depth": 4,
              "min_data_in_leaf": 200, "lambda_l2": 10., "seed": 20260912, "num_threads": 2,
              "deterministic": True, "force_col_wise": True, "verbosity": -1}
    dataset = lgb.Dataset(x, label=y, feature_name=FEATURES, free_raw_data=True)
    deadline = datetime.fromisoformat(relation.read(transfer.SESSION)["deadline_utc"])
    def stop(_):
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("3h deadline reached inside014 fit")
    relation.progress(f"014 regression fit through {training[-1]}:{len(x):,} action rows")
    model = lgb.train(params, dataset, num_boost_round=150, callbacks=[stop])
    model.save_model(str(path))
    meta = {"training": training, "features": FEATURES, "params": params, "rounds": 150, "input": summary,
            "resources": hardware, "model": str(path), "seconds": time.perf_counter() - started,
            "importance": sorted([{"feature": f, "gain": float(g)} for f, g in zip(FEATURES, model.feature_importance("gain"))], key=lambda z: -z["gain"])}
    relation.save(marker, meta)
    del x, y, dataset; gc.collect()
    return model, meta


def select(scores: pd.DataFrame) -> pd.DataFrame:
    candidate.require(set(scores.columns) == set(relation.KEYS + ["predicted_AP_delta_scaled"]), "014 selector schema drift")
    candidate.require(np.isfinite(scores.predicted_AP_delta_scaled.to_numpy(float)).all(), "014 nonfinite predictions")
    return scores[scores.predicted_AP_delta_scaled > 0].sort_values(
        ["customer_id", "predicted_AP_delta_scaled", "challenger_article_id", "victim_article_id"],
        ascending=[True, False, True, True], kind="mergesort",
    ).drop_duplicates("customer_id").reset_index(drop=True)


def evaluate(cutoff: str, model: lgb.Booster):
    folder = ART / cutoff; folder.mkdir(parents=True, exist_ok=True)
    source, _ = hard.input_table(cutoff)
    with candidate.connection() as db:
        frame = db.execute(f"SELECT {','.join(relation.KEYS + FEATURES)} FROM read_parquet(?)", [str(source)]).fetchdf()
    x = frame[FEATURES].to_numpy(np.float32); x[~np.isfinite(x)] = np.nan
    scores = frame[relation.KEYS].copy(); scores["predicted_AP_delta_scaled"] = model.predict(x, num_threads=2)
    relation.parquet(scores, folder / "scores.parquet")
    actions = select(scores); relation.parquet(actions, folder / "decisions.parquet")
    result = transfer.evaluate_actions(folder, cutoff, actions)
    result.update(prediction_target_scale=SCALE, predicted_positive_actions=len(scores[scores.predicted_AP_delta_scaled > 0]))
    return result


def run():
    started = time.perf_counter()
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK014_CONTRACT.json")
    output = relation.REPORT / "MIND_WARM_RANK014_METRICS.json"
    result = {"status": "running", "inputs": {}, "models": {}, "windows": {}, "final_week": "not_run", "independent_confirmation": "not_run"}
    try:
        for cutoff, training in contract["schedule"].items():
            relation.check_temporal(training, cutoff)
            for t in training:
                _, result["inputs"][t] = labeled_input(t)
            model, result["models"][cutoff] = fit(training)
            result["windows"][cutoff] = evaluate(cutoff, model)
            relation.save(output, result)
            relation.progress(f"014 {cutoff}:deltaMAP={result['windows'][cutoff]['delta_vs_baseline']:+.8f},actions={result['windows'][cutoff]['selected_users']}")
            del model; gc.collect()
        values = [v["delta_vs_baseline"] for v in result["windows"].values()]
        result["gate"] = {"passed": bool(np.mean(values) > 0 and all(v >= 0 for v in values) and min(values) >= -.0002),
                          "mean_delta": float(np.mean(values)), "nondegrade_windows": sum(v >= 0 for v in values), "worst_delta": min(values)}
        result["status"] = "completed_development_pass" if result["gate"]["passed"] else "completed_rejected"
    except Exception as exc:
        result.update(status="resource_or_engineering_stopped", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["seconds"] = time.perf_counter() - started
        relation.save(output, result)
    return result


if __name__ == "__main__":
    run()
