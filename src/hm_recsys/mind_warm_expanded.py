"""011 expanded cutoff-safe hard-negative features on the fixed MIND pool."""
from __future__ import annotations
import gc
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np

from . import mind_warm_transfer as transfer
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation

TRIAL = "MIND-WARM-RANK-011"
ART = transfer.ART / TRIAL
EXTRA = [
    "customer_age", "customer_age_missing", "age_bucket", "user_item_events_28d",
    "user_product_code_events_28d", "user_product_code_share_12w",
    "user_product_type_events_28d", "user_product_type_events_12w", "user_product_type_days_since",
    "user_department_events_28d", "user_department_events_12w", "user_department_days_since",
    "user_garment_events_28d", "user_garment_events_12w", "user_garment_days_since",
    "user_colour_events_28d", "user_colour_events_12w", "user_colour_days_since",
    "user_index_group_events_28d", "user_index_group_events_12w", "user_index_group_days_since", "user_index_group_share_12w",
    "mind_rank", "mind_rank_fraction", "mind_reciprocal_rank", "mind_score", "mind_score_user_z",
    "mind_best_interest", "mind_candidate_interest_count", "mind_user_interest_count",
]
FEATURES = candidate.FEATURES + EXTRA


def resources(min_ram=6):
    status = transfer.budget()
    candidate.require(status["available_ram_gib"] >= min_ram, f"011 requires{min_ram}GiB available RAM; observed{status['available_ram_gib']:.2f}")
    return status


def input_table(cutoff):
    folder = ART / "input" / cutoff
    path, marker = folder / "features.parquet", folder / "INPUT.json"
    if path.exists() and marker.exists():
        meta = relation.read(marker)
        candidate.require(meta["features"] == FEATURES and meta["cutoff"] == cutoff, "stale011 input")
        return path, meta
    start = time.perf_counter()
    resources()
    shared, shared_meta = transfer.shared_input(cutoff)
    source = transfer.source(cutoff)
    folder.mkdir(parents=True, exist_ok=True)
    with candidate.connection() as db:
        db.execute("SET memory_limit='4GiB'")
        db.execute(f"COPY (SELECT s.*,{','.join('o.' + f for f in EXTRA)} FROM read_parquet('{relation.old._sql_path(shared)}') s JOIN read_parquet('{relation.old._sql_path(source)}') o USING(customer_id,article_id) ORDER BY s.customer_id,s.pool,s.article_id) TO '{relation.old._sql_path(path)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        counts = db.execute("SELECT count(*) row_count,count(DISTINCT(customer_id,article_id)) unique_rows,sum(target) positives,count(*) FILTER(WHERE pool='mind_new') mind_rows,sum(target) FILTER(WHERE pool='mind_new') mind_positives FROM read_parquet(?)", [str(path)]).fetchdf().to_dict("records")[0]
        missing = {f: db.execute(f"SELECT count(*) FILTER(WHERE {f} IS NOT NULL) mind_nonmissing,count(*) FILTER(WHERE pool='base_novel' AND {f} IS NOT NULL) base_nonmissing FROM read_parquet(?) WHERE pool='mind_new' OR pool='base_novel'", [str(path)]).fetchdf().to_dict("records")[0] for f in EXTRA[-8:]}
    candidate.require(counts["row_count"] == counts["unique_rows"] == shared_meta["rows"], "011 row/key conservation failed")
    meta = {"cutoff": cutoff, "features": FEATURES, "rows": counts, "MIND_source_feature_nonmissing": missing,
            "source": str(source), "shared": str(shared), "seconds": time.perf_counter() - start, "final_week": "not_run"}
    relation.save(marker, meta)
    relation.progress(f"011 input {cutoff}: {counts['row_count']:,} rows,{meta['seconds']:.1f}s")
    return path, meta


def fit(training):
    folder = ART / "models" / ("through_" + training[-1])
    model_path, marker = folder / "model.txt", folder / "MODEL.json"
    if model_path.exists() and marker.exists():
        meta = relation.read(marker)
        candidate.require(meta["training"] == training and meta["features"] == FEATURES, "stale011 model")
        return lgb.Booster(model_file=str(model_path)), meta
    start = time.perf_counter()
    hardware = resources()
    folder.mkdir(parents=True, exist_ok=True)
    params = relation.read(candidate.CONTRACT)["model"].copy()
    params.pop("selection")
    rounds = params.pop("rounds")
    params.update(objective="binary", metric="binary_logloss")
    params.pop("eval_at")
    params.pop("lambdarank_truncation_level")
    with candidate.connection() as db:
        db.execute("SET memory_limit='5GiB'")
        union = " UNION ALL ".join(f"SELECT '{c}' sample_cutoff,* FROM read_parquet('{relation.old._sql_path(ART / 'input' / c / 'features.parquet')}')" for c in training)
        db.execute("CREATE VIEW train AS " + union)
        n = db.execute("SELECT count(*) FROM train").fetchone()[0]
        candidate.require(n <= 6500000, "011 registered row cap exceeded")
        frame = db.execute(f"SELECT sample_cutoff,customer_id,pool,target,{','.join(FEATURES)} FROM train ORDER BY sample_cutoff,customer_id,pool,article_id").fetchdf()
    counts = frame.groupby("pool").size().to_dict()
    factor = counts["mind_new"] / counts["base_novel"]
    weights = np.where(frame.pool == "mind_new", 1., factor).astype(np.float32)
    labels = frame.target.to_numpy(np.int8)
    matrix = frame[FEATURES].to_numpy(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    evidence = {"rows": len(frame), "positives": int(labels.sum()), "source_rows": counts, "base_weight": factor}
    del frame
    gc.collect()
    dataset = lgb.Dataset(matrix, label=labels, weight=weights, feature_name=FEATURES, free_raw_data=True)
    deadline = datetime.fromisoformat(relation.read(transfer.SESSION)["deadline_utc"])
    def stop(_):
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("3h deadline reached inside011 fitting")
    relation.progress(f"011 fit through {training[-1]}: {len(matrix):,} rows,{len(FEATURES)} features")
    model = lgb.train(params, dataset, num_boost_round=rounds, callbacks=[stop])
    model.save_model(str(model_path))
    meta = {"training": training, "features": FEATURES, "params": params, "rounds": rounds, "input": evidence,
            "resources": hardware, "seconds": time.perf_counter() - start, "model": str(model_path),
            "importance": sorted([{"feature": f, "gain": float(g)} for f, g in zip(FEATURES, model.feature_importance("gain"))], key=lambda z: -z["gain"])}
    relation.save(marker, meta)
    del matrix, labels, weights, dataset
    gc.collect()
    return model, meta


def score_and_admit(cutoff, training, model, artifact_root=ART):
    folder = artifact_root / cutoff
    folder.mkdir(parents=True, exist_ok=True)
    source = transfer.source(cutoff)
    with candidate.connection() as db:
        frame = db.execute(f"SELECT c.*,{','.join('s.' + f for f in EXTRA)} FROM read_parquet(?) c JOIN read_parquet(?) s USING(customer_id,article_id) ORDER BY c.customer_id,c.article_id", [str(candidate.ART / cutoff / "candidates.parquet"), str(source)]).fetchdf()
    x = frame[FEATURES].to_numpy(np.float32)
    x[~np.isfinite(x)] = np.nan
    scores = frame[candidate.KEYS].copy()
    scores["score"] = model.predict(x, num_threads=2)
    order = candidate.order_candidates(scores)
    relation.parquet(order, folder / "order.parquet")
    internal = candidate.candidate_metrics(order, frame[candidate.KEYS + ["target"]], relation.read(candidate.ART / cutoff / "INPUT.json")["total_users"])
    with candidate.connection() as db:
        fields = list(dict.fromkeys(relation.KEYS + relation.FEATURES["dense_primary"]))
        pairs = db.execute(f"SELECT {','.join('p.' + f for f in fields)} FROM read_parquet(?) p JOIN read_parquet(?) r ON p.customer_id=r.customer_id AND p.challenger_article_id=r.article_id WHERE r.rank<=5", [str(relation.ART / cutoff / "pairs.parquet"), str(folder / "order.parquet")]).fetchdf()
    relation_model = lgb.Booster(model_file=str(relation.model_path(training, "dense_primary")))
    px = pairs[relation.FEATURES["dense_primary"]].to_numpy(np.float32)
    px[~np.isfinite(px)] = np.nan
    relation_scores = pairs[relation.KEYS].copy()
    relation_scores[relation.PROBS] = relation_model.predict(px, num_threads=2)
    relation.parquet(relation_scores, folder / "admission-scores.parquet")
    actions = relation.select_actions(relation_scores)
    relation.parquet(actions, folder / "decisions.parquet")
    admission = transfer.evaluate_actions(folder, cutoff, actions)
    return internal, admission


def run():
    started = time.perf_counter()
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK011_CONTRACT.json")
    output = relation.REPORT / "MIND_WARM_RANK011_METRICS.json"
    result = {"status": "running", "inputs": {}, "models": {}, "windows": {}, "final_week": "not_run", "independent_confirmation": "not_run"}
    try:
        for cutoff, training in contract["schedule"].items():
            relation.check_temporal(training, cutoff)
            for t in training:
                _, result["inputs"][t] = input_table(t)
            model, result["models"][cutoff] = fit(training)
            internal, admission = score_and_admit(cutoff, training, model)
            result["windows"][cutoff] = {"internal": internal, "admission": admission}
            relation.save(output, result)
            relation.progress(f"011 {cutoff}: Hit1={internal['conditional_hit_at_1']:.5f},Hit5={internal['conditional_hit_at_5']:.5f},deltaMAP={admission['delta_vs_baseline']:+.8f}")
            del model
            gc.collect()
        values = [w["admission"]["delta_vs_baseline"] for w in result["windows"].values()]
        result["gate"] = {"passed": bool(np.mean(values) > 0 and all(v >= 0 for v in values) and min(values) >= -.0002), "mean_delta": float(np.mean(values)), "nondegrade_windows": sum(v >= 0 for v in values), "worst_delta": min(values)}
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
