"""P015 chronological OOF hard-negative MIND candidate LambdaRank."""
from __future__ import annotations

import gc
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np

from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_expanded as expanded
from . import mind_warm_relation as relation
from . import mind_warm_transfer as transfer


TRIAL = "MIND-WARM-RANK-015"
ART = transfer.ART / TRIAL
FEATURES = expanded.FEATURES


def resources():
    status = transfer.budget()
    candidate.require(status["available_ram_gib"] >= 6, f"015 requires6GiB free RAM; observed{status['available_ram_gib']:.2f}")
    return status


def hard_input(cutoff: str):
    folder = ART / "input" / cutoff
    path, marker = folder / "features.parquet", folder / "INPUT.json"
    if path.exists() and marker.exists():
        meta = relation.read(marker)
        candidate.require(meta["features"] == FEATURES and meta["cutoff"] == cutoff, "stale015 input")
        return path, meta
    started = time.perf_counter(); resources(); folder.mkdir(parents=True, exist_ok=True)
    base = candidate.ART / cutoff / "candidates.parquet"
    source = transfer.source(cutoff)
    if cutoff == "2020-01-22":
        rank_join = ""
        rank_value = "s.mind_rank"
        rank_source = "raw_MIND_rank_anchor"
    else:
        order = ART / cutoff / "order.parquet"
        candidate.require(order.exists(), f"missing chronological OOF015 order for {cutoff}")
        rank_join = f"JOIN read_parquet('{relation.old._sql_path(order)}') o USING(customer_id,article_id)"
        rank_value = "o.rank"
        rank_source = "chronological_OOF_015_rank"
    extras = ",".join("s." + f for f in expanded.EXTRA)
    with candidate.connection() as db:
        db.execute("SET memory_limit='4GiB'")
        query = f"""WITH all_rows AS (
            SELECT c.*,{extras},{rank_value} mining_rank
            FROM read_parquet('{relation.old._sql_path(base)}') c
            JOIN read_parquet('{relation.old._sql_path(source)}') s USING(customer_id,article_id)
            {rank_join}
        ), kept AS (
            SELECT * FROM all_rows WHERE target=1 OR mining_rank<=20
        ), eligible AS (
            SELECT customer_id FROM kept GROUP BY customer_id HAVING sum(target)>0 AND sum(target)<count(*)
        ) SELECT k.customer_id,k.article_id,k.target,{','.join('k.' + f for f in FEATURES)},k.mining_rank
          FROM kept k JOIN eligible USING(customer_id)
          ORDER BY k.customer_id,k.mining_rank,k.article_id"""
        db.execute(f"COPY ({query}) TO '{relation.old._sql_path(path)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        counts = db.execute("WITH g AS(SELECT customer_id,count(*) n,sum(target) p FROM read_parquet(?) GROUP BY 1) SELECT count(*) group_count,sum(n) candidate_rows,sum(p) positives,min(n) min_group,max(n) max_group,count(*) FILTER(WHERE p=0 OR p=n) invalid_groups FROM g", [str(path)]).fetchdf().to_dict("records")[0]
        unique = db.execute("SELECT count(*)=count(DISTINCT(customer_id,article_id)) FROM read_parquet(?)", [str(path)]).fetchone()[0]
    candidate.require(unique and counts["invalid_groups"] == 0 and counts["positives"] >= 100, "015 hard input integrity/supervision gate")
    meta = {"cutoff": cutoff, "features": FEATURES, "rank_source": rank_source, "fixed_negative_depth": 20,
            "groups": counts["group_count"], "candidate_rows": counts["candidate_rows"], "positive_rows": counts["positives"],
            "minimum_group_rows": counts["min_group"], "maximum_group_rows": counts["max_group"],
            "all_positive_rows_retained": True, "only_historical_labels_used": True, "final_week": "not_run", "seconds": time.perf_counter() - started}
    relation.save(marker, meta); relation.progress(f"015 hard input {cutoff}:{meta['candidate_rows']:,} rows,{meta['groups']} mixed groups")
    return path, meta


def fit(training: list[str]):
    folder = ART / "models" / ("through_" + training[-1])
    path, marker = folder / "model.txt", folder / "MODEL.json"
    if path.exists() and marker.exists():
        meta = relation.read(marker); candidate.require(meta["training"] == training and meta["features"] == FEATURES, "stale015 model")
        return lgb.Booster(model_file=str(path)), meta
    started = time.perf_counter(); hardware = resources(); folder.mkdir(parents=True, exist_ok=True)
    with candidate.connection() as db:
        union = " UNION ALL ".join(f"SELECT '{c}' sample_cutoff,* FROM read_parquet('{relation.old._sql_path(ART / 'input' / c / 'features.parquet')}')" for c in training)
        frame = db.execute(f"SELECT sample_cutoff,customer_id,target,{','.join(FEATURES)} FROM ({union}) ORDER BY sample_cutoff,customer_id").fetchdf()
    groups = frame.groupby(["sample_cutoff", "customer_id"], sort=False).size().astype(int).tolist()
    y = frame.target.to_numpy(np.int8); x = frame[FEATURES].to_numpy(np.float32); x[~np.isfinite(x)] = np.nan
    candidate.require(sum(groups) == len(frame) and len(frame) <= 250000 and int(y.sum()) >= 100, "015 fit gate")
    summary = {"rows": len(frame), "groups": len(groups), "positive_rows": int(y.sum())}
    del frame; gc.collect()
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [1, 5], "lambdarank_truncation_level": 5,
              "learning_rate": .03, "num_leaves": 15, "max_depth": 4, "min_data_in_leaf": 100, "lambda_l2": 10.,
              "seed": 20260912, "num_threads": 2, "deterministic": True, "force_col_wise": True, "verbosity": -1}
    dataset = lgb.Dataset(x, label=y, group=groups, feature_name=FEATURES, free_raw_data=True)
    deadline = datetime.fromisoformat(relation.read(transfer.SESSION)["deadline_utc"])
    def stop(_):
        if datetime.now(timezone.utc) >= deadline: raise TimeoutError("3h deadline reached inside015 fit")
    relation.progress(f"015 LambdaRank fit through {training[-1]}:{len(x):,} hard rows")
    model = lgb.train(params, dataset, num_boost_round=150, callbacks=[stop]); model.save_model(str(path))
    meta = {"training": training, "features": FEATURES, "params": params, "rounds": 150, "input": summary, "resources": hardware,
            "model": str(path), "seconds": time.perf_counter() - started,
            "importance": sorted([{"feature": f, "gain": float(g)} for f, g in zip(FEATURES, model.feature_importance("gain"))], key=lambda z: -z["gain"])}
    relation.save(marker, meta); del x, y, dataset; gc.collect(); return model, meta


def evaluate_order(cutoff: str, model: lgb.Booster):
    folder = ART / cutoff; folder.mkdir(parents=True, exist_ok=True)
    base = candidate.ART / cutoff / "candidates.parquet"; source = transfer.source(cutoff)
    with candidate.connection() as db:
        frame = db.execute(f"SELECT c.*,{','.join('s.' + f for f in expanded.EXTRA)} FROM read_parquet(?) c JOIN read_parquet(?) s USING(customer_id,article_id) ORDER BY c.customer_id,c.article_id", [str(base), str(source)]).fetchdf()
    x = frame[FEATURES].to_numpy(np.float32); x[~np.isfinite(x)] = np.nan
    scores = frame[candidate.KEYS].copy(); scores["score"] = model.predict(x, num_threads=2)
    order = candidate.order_candidates(scores); relation.parquet(order, folder / "order.parquet")
    metrics = candidate.candidate_metrics(order, frame[candidate.KEYS + ["target"]], relation.read(candidate.ART / cutoff / "INPUT.json")["total_users"])
    relation.save(folder / "INTERNAL.json", metrics); return metrics


def run():
    started = time.perf_counter(); contract = relation.read(relation.REPORT / "MIND_WARM_RANK015_CONTRACT.json")
    output = relation.REPORT / "MIND_WARM_RANK015_METRICS.json"
    result = {"status": "running", "inputs": {}, "models": {}, "windows": {}, "final_week": "not_run", "admission": "not_run"}
    try:
        for cutoff, training in contract["schedule"].items():
            relation.check_temporal(training, cutoff)
            for t in training:
                _, result["inputs"][t] = hard_input(t)
            model, result["models"][cutoff] = fit(training)
            result["windows"][cutoff] = evaluate_order(cutoff, model)
            relation.save(output, result)
            relation.progress(f"015 {cutoff}:Hit1={result['windows'][cutoff]['conditional_hit_at_1']:.5f},Hit5={result['windows'][cutoff]['conditional_hit_at_5']:.5f}")
            del model; gc.collect()
        base = relation.read(relation.REPORT / "MIND_WARM_RANK005_METRICS.json")["windows"]
        d1 = {c: result["windows"][c]["conditional_hit_at_1"] - base[c]["internal"]["conditional_hit_at_1"] for c in result["windows"]}
        d5 = {c: result["windows"][c]["conditional_hit_at_5"] - base[c]["internal"]["conditional_hit_at_5"] for c in result["windows"]}
        passed = np.mean(list(d1.values())) > 0 and sum(v >= 0 for v in d1.values()) >= 2 and np.mean(list(d5.values())) >= 0 and sum(v >= 0 for v in d5.values()) >= 2
        result["diagnostic_gate"] = {"passed": bool(passed), "Hit1_delta_vs_005": d1, "Hit1_mean_delta": float(np.mean(list(d1.values()))),
                                     "Hit5_delta_vs_005": d5, "Hit5_mean_delta": float(np.mean(list(d5.values())))}
        result["status"] = "completed_diagnostic_pass" if passed else "completed_diagnostic_rejected"
    except Exception as exc:
        result.update(status="resource_or_engineering_stopped", error=f"{type(exc).__name__}: {exc}"); raise
    finally:
        result["seconds"] = time.perf_counter() - started; relation.save(output, result)
    return result


if __name__ == "__main__": run()
