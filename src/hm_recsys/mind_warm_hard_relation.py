"""013 OOF-shortlist-aligned challenger/victim relation model."""
from __future__ import annotations
import gc
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np

from . import mind_warm_transfer as transfer
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation

TRIAL = "MIND-WARM-RANK-013"
ART = transfer.ART / TRIAL
MIND = ["mind_rank", "mind_rank_fraction", "mind_reciprocal_rank", "mind_score", "mind_score_user_z", "mind_best_interest", "mind_candidate_interest_count", "mind_user_interest_count"]
DETAIL = [
    "user_item_events_28d", "user_product_code_events_28d", "user_product_code_share_12w",
    "user_product_type_events_28d", "user_product_type_events_12w", "user_product_type_days_since",
    "user_department_events_28d", "user_department_events_12w", "user_department_days_since",
    "user_garment_events_28d", "user_garment_events_12w", "user_garment_days_since",
    "user_colour_events_28d", "user_colour_events_12w", "user_colour_days_since",
    "user_index_group_events_28d", "user_index_group_events_12w", "user_index_group_days_since", "user_index_group_share_12w",
]
EXTRA = ["customer_age", "customer_age_missing", "age_bucket", *["c_" + f for f in MIND], *[prefix + f for f in DETAIL for prefix in ("c_", "v_", "diff_")]]
FEATURES = relation.FEATURES["dense_primary"] + EXTRA


def input_table(cutoff):
    folder = ART / "input" / cutoff
    path, marker = folder / "pairs.parquet", folder / "INPUT.json"
    if path.exists() and marker.exists():
        meta = relation.read(marker)
        candidate.require(meta["features"] == FEATURES and meta["cutoff"] == cutoff, "stale013 input")
        return path, meta
    start = time.perf_counter()
    transfer.budget()
    folder.mkdir(parents=True, exist_ok=True)
    pair_path, source = relation.ART / cutoff / "pairs.parquet", transfer.source(cutoff)
    if cutoff == "2020-01-22":
        short = f"SELECT customer_id,article_id FROM read_parquet('{relation.old._sql_path(source)}') WHERE mind_is_new=1 AND mind_rank<=5"
        shortlist = "raw_MIND_Top5_anchor"
    else:
        order = transfer.ART / "MIND-WARM-RANK-005" / cutoff / "order.parquet"
        short = f"SELECT customer_id,article_id FROM read_parquet('{relation.old._sql_path(order)}') WHERE rank<=5"
        shortlist = "chronological_OOF_005_Top5"
    detail = ",".join(["c." + f + " c_" + f + ",v." + f + " v_" + f + ",(c." + f + "-v." + f + ") diff_" + f for f in DETAIL])
    select = ",".join("p." + f for f in list(dict.fromkeys(relation.KEYS + ["relation_label"] + relation.FEATURES["dense_primary"])))
    with candidate.connection() as db:
        db.execute("SET memory_limit='3GiB'")
        query = f"SELECT {select},c.customer_age,c.customer_age_missing,c.age_bucket,{','.join('c.' + f + ' c_' + f for f in MIND)},{detail} FROM read_parquet('{relation.old._sql_path(pair_path)}') p JOIN ({short}) q ON p.customer_id=q.customer_id AND p.challenger_article_id=q.article_id JOIN read_parquet('{relation.old._sql_path(source)}') c ON p.customer_id=c.customer_id AND p.challenger_article_id=c.article_id JOIN read_parquet('{relation.old._sql_path(source)}') v ON p.customer_id=v.customer_id AND p.victim_article_id=v.article_id"
        db.execute(f"COPY ({query} ORDER BY p.customer_id,p.challenger_article_id,p.champion_rank) TO '{relation.old._sql_path(path)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        counts = db.execute("SELECT count(*) row_count,count(DISTINCT(customer_id,challenger_article_id,victim_article_id)) unique_rows,count(DISTINCT(customer_id,challenger_article_id)) challengers,count(DISTINCT customer_id) users FROM read_parquet(?)", [str(path)]).fetchdf().to_dict("records")[0]
        classes = {str(k): v for k, v in db.execute("SELECT relation_label,count(*) FROM read_parquet(?) GROUP BY 1 ORDER BY 1", [str(path)]).fetchall()}
    candidate.require(counts["row_count"] == counts["unique_rows"] == 5 * counts["challengers"], "013 full shortlist times five failed")
    meta = {"cutoff": cutoff, "features": FEATURES, "rows": counts, "classes": classes, "shortlist": shortlist,
            "history_strictly_before_cutoff": True, "seconds": time.perf_counter() - start, "final_week": "not_run"}
    relation.save(marker, meta)
    relation.progress(f"013 input {cutoff}:{counts['row_count']:,} pairs,classes={classes}")
    return path, meta


def fit(training):
    folder = ART / "models" / ("through_" + training[-1])
    path, marker = folder / "model.txt", folder / "MODEL.json"
    if path.exists() and marker.exists():
        meta = relation.read(marker)
        candidate.require(meta["training"] == training and meta["features"] == FEATURES, "stale013 model")
        return lgb.Booster(model_file=str(path)), meta
    start = time.perf_counter()
    hardware = transfer.budget()
    folder.mkdir(parents=True, exist_ok=True)
    with candidate.connection() as db:
        union = " UNION ALL ".join(f"SELECT * FROM read_parquet('{relation.old._sql_path(ART / 'input' / c / 'pairs.parquet')}')" for c in training)
        frame = db.execute(f"SELECT relation_label,{','.join(FEATURES)} FROM ({union})").fetchdf()
    candidate.require(len(frame) <= 1000000 and set(frame.relation_label.unique()) == {0, 1, 2}, "013 row/class gate")
    labels = frame.relation_label.to_numpy(np.int8)
    matrix = frame[FEATURES].to_numpy(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    classes = {str(k): int((labels == k).sum()) for k in range(3)}
    del frame
    gc.collect()
    params = {"objective": "multiclass", "num_class": 3, "learning_rate": .03, "num_leaves": 15, "max_depth": 4, "min_data_in_leaf": 200, "lambda_l2": 10., "seed": 20260912, "num_threads": 2, "deterministic": True, "force_col_wise": True, "verbosity": -1}
    dataset = lgb.Dataset(matrix, label=labels, feature_name=FEATURES, free_raw_data=True)
    deadline = datetime.fromisoformat(relation.read(transfer.SESSION)["deadline_utc"])
    def stop(_):
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("3h deadline reached inside013 fit")
    relation.progress(f"013 fit through {training[-1]}:{len(matrix):,} pairs")
    model = lgb.train(params, dataset, num_boost_round=150, callbacks=[stop])
    model.save_model(str(path))
    meta = {"training": training, "features": FEATURES, "params": params, "rounds": 150, "classes": classes, "resources": hardware,
            "model": str(path), "seconds": time.perf_counter() - start,
            "importance": sorted([{"feature": f, "gain": float(g)} for f, g in zip(FEATURES, model.feature_importance("gain"))], key=lambda z: -z["gain"])}
    relation.save(marker, meta)
    del matrix, labels, dataset
    gc.collect()
    return model, meta


def evaluate(cutoff, model):
    folder = ART / cutoff
    folder.mkdir(parents=True, exist_ok=True)
    pair_path, source = relation.ART / cutoff / "pairs.parquet", transfer.source(cutoff)
    order = transfer.ART / "MIND-WARM-RANK-005" / cutoff / "order.parquet"
    detail = ",".join(["c." + f + " c_" + f + ",v." + f + " v_" + f + ",(c." + f + "-v." + f + ") diff_" + f for f in DETAIL])
    select = ",".join("p." + f for f in list(dict.fromkeys(relation.KEYS + relation.FEATURES["dense_primary"])))
    with candidate.connection() as db:
        query = f"SELECT {select},c.customer_age,c.customer_age_missing,c.age_bucket,{','.join('c.' + f + ' c_' + f for f in MIND)},{detail} FROM read_parquet(?) p JOIN read_parquet(?) r ON p.customer_id=r.customer_id AND p.challenger_article_id=r.article_id JOIN read_parquet(?) c ON p.customer_id=c.customer_id AND p.challenger_article_id=c.article_id JOIN read_parquet(?) v ON p.customer_id=v.customer_id AND p.victim_article_id=v.article_id WHERE r.rank<=5"
        frame = db.execute(query, [str(pair_path), str(order), str(source), str(source)]).fetchdf()
    x = frame[FEATURES].to_numpy(np.float32)
    x[~np.isfinite(x)] = np.nan
    scores = frame[relation.KEYS].copy()
    scores[relation.PROBS] = model.predict(x, num_threads=2)
    relation.parquet(scores, folder / "scores.parquet")
    actions = relation.select_actions(scores)
    relation.parquet(actions, folder / "decisions.parquet")
    return transfer.evaluate_actions(folder, cutoff, actions)


def run():
    started = time.perf_counter()
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK013_CONTRACT.json")
    output = relation.REPORT / "MIND_WARM_RANK013_METRICS.json"
    result = {"status": "running", "inputs": {}, "models": {}, "windows": {}, "final_week": "not_run", "independent_confirmation": "not_run"}
    try:
        for cutoff, training in contract["schedule"].items():
            relation.check_temporal(training, cutoff)
            for t in training:
                _, result["inputs"][t] = input_table(t)
            model, result["models"][cutoff] = fit(training)
            result["windows"][cutoff] = evaluate(cutoff, model)
            relation.save(output, result)
            relation.progress(f"013 {cutoff}:deltaMAP={result['windows'][cutoff]['delta_vs_baseline']:+.8f},actions={result['windows'][cutoff]['selected_users']}")
            del model
            gc.collect()
        values = [v["delta_vs_baseline"] for v in result["windows"].values()]
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
