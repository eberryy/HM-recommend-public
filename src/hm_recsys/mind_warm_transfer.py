"""One-hour MIND supervision transfer audit and explicitly registered trials."""
from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation
from .mind_warm_dense import build_dense_features
from .mind_warm_baseline import reconstruct_full_baseline
from . import mind_warm_relation_audit as independent

ROOT, REPORT = relation.ROOT, relation.REPORT
ART = ROOT / "artifacts/mind_warm_side/MIND-WARM-TRANSFER"
TRAIN_CUTOFFS = ["2020-01-22", "2020-03-18", "2020-06-24"]
SESSION = REPORT / ("MIND_WARM_3H_CONTRACT.json" if (REPORT / "MIND_WARM_3H_CONTRACT.json").exists() else "MIND_WARM_HOUR_CONTRACT.json")


def budget():
    config = relation.read(SESSION)
    remaining = (datetime.fromisoformat(config["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        raise TimeoutError("one-hour research deadline reached; no new work allowed")
    result = relation.resources()
    result["remaining_seconds"] = remaining
    return result


def source(cutoff):
    relation.old._guard(cutoff)
    return relation.ART / "aggregate_adapter" / cutoff / "features.parquet"


def audit():
    start = time.perf_counter()
    budget()
    result = {"schema": "mind-supervision-transfer-audit-v1", "cutoffs": {}, "new_training": False,
              "only_historical_training_windows": TRAIN_CUTOFFS, "final_week": "not_run"}
    fields = ["item_events_7d", "item_trend_7d_vs_28d", "item_days_since_last_sale", "user_history_events_12w", "user_product_type_share_12w", "user_item_price_gap"]
    with candidate.connection() as db:
        for cutoff in TRAIN_CUTOFFS:
            budget()
            db.execute(f"CREATE OR REPLACE TEMP VIEW f AS SELECT *,CASE WHEN mind_is_new=1 THEN 'mind_new' WHEN user_item_events_12w=0 THEN 'base_novel' ELSE 'base_repeat' END pool FROM read_parquet('{relation.old._sql_path(source(cutoff))}')")
            counts = db.execute("SELECT pool,count(*) candidate_rows,sum(target) positives,count(DISTINCT customer_id) users,count(DISTINCT customer_id) FILTER(WHERE target=1) positive_users FROM f GROUP BY pool ORDER BY pool").fetchdf().to_dict("records")
            overlap = db.execute("SELECT count(*) positive_mind_pairs,count(*) FILTER(WHERE EXISTS(SELECT 1 FROM f b WHERE b.pool='base_novel' AND b.target=1 AND b.article_product_type_no=m.article_product_type_no)) type_supported,count(*) FILTER(WHERE EXISTS(SELECT 1 FROM f b WHERE b.pool='base_novel' AND b.target=1 AND b.article_id=m.article_id)) item_supported,sum((m.user_item_events_12w=0)::INTEGER) novel_to_user FROM f m WHERE m.pool='mind_new' AND m.target=1").fetchone()
            distributions = {}
            for field in fields:
                values = db.execute(f"SELECT pool,count({field}) n,quantile_cont({field},[0.05,0.5,0.95]) q FROM f WHERE target=1 GROUP BY pool ORDER BY pool").fetchall()
                bounds = db.execute(f"SELECT quantile_cont({field},.05),quantile_cont({field},.95) FROM f WHERE pool='base_novel' AND target=1").fetchone()
                supported = db.execute(f"SELECT count(*) FILTER(WHERE {field} BETWEEN ? AND ?),count({field}) FROM f WHERE pool='mind_new' AND target=1", list(bounds)).fetchone()
                distributions[field] = {"positive_distributions": [{"pool": p, "nonmissing": n, "q05_q50_q95": list(q)} for p, n, q in values], "mind_positive_inside_base_novel_5_95_count": supported[0], "mind_positive_nonmissing": supported[1]}
            result["cutoffs"][cutoff] = {"source": str(source(cutoff)), "pool_counts": counts,
                "mind_positive_pairs": overlap[0], "mind_positive_type_supported_by_base_novel": overlap[1],
                "mind_positive_item_supported_by_other_base_novel_user": overlap[2], "mind_positive_novel_to_user": overlap[3], "distributions": distributions}
    result["elapsed_seconds"] = time.perf_counter() - start
    relation.save(REPORT / "MIND_WARM_TRANSFER_AUDIT.json", result)
    return result


def shared_input(cutoff):
    folder = ART / "shared" / cutoff
    folder.mkdir(parents=True, exist_ok=True)
    path, marker = folder / "features.parquet", folder / "INPUT.json"
    if path.exists() and marker.exists():
        return path, relation.read(marker)
    start = time.perf_counter()
    budget()
    columns = ["customer_id", "article_id", "target", "mind_is_new", *relation.USER, *relation.ITEM]
    with candidate.connection() as db:
        frame = db.execute(f"SELECT {','.join(columns)} FROM read_parquet(?) WHERE mind_is_new=1 OR user_item_events_12w=0 ORDER BY customer_id,article_id", [str(source(cutoff))]).fetchdf()
    frame["pool"] = np.where(frame.mind_is_new == 1, "mind_new", "base_novel")
    candidate.require(not frame.duplicated(candidate.KEYS).any(), "duplicate shared-input keys")
    # Call the frozen encoder with only identities, no targets or source flags.
    import torch
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        dense, dense_meta = build_dense_features(cutoff, frame[candidate.KEYS].copy(), "cpu")
    finally:
        torch.set_num_threads(old_threads)
    for f in candidate.DENSE_FEATURES:
        frame[f] = dense[f].to_numpy()
    del dense
    gc.collect()
    # Preserve the exact frozen MIND-side input. Re-batching dot products over
    # the larger base pool changes a few float32 ULPs, not their semantics.
    with candidate.connection() as db:
        frozen_mind = db.execute("SELECT * FROM read_parquet(?)", [str(candidate.ART / cutoff / "candidates.parquet")]).fetchdf()
    frozen_mind["pool"] = "mind_new"
    output_columns = candidate.KEYS + ["pool", "target"] + candidate.FEATURES
    frame = pd.concat([frame.loc[frame.pool == "base_novel", output_columns], frozen_mind[output_columns]], ignore_index=True)
    candidate.require(not frame.duplicated(candidate.KEYS).any(), "shared frozen-side duplicate keys")
    relation.parquet(frame[candidate.KEYS + ["pool", "target"] + candidate.FEATURES], path)
    with candidate.connection() as db:
        groups = db.execute("SELECT pool,count(*) group_count,sum(n) candidate_rows,sum(p) positive_rows,count(*) FILTER(WHERE p>0 AND p<n) mixed_groups FROM (SELECT customer_id,pool,count(*) n,sum(target) p FROM read_parquet(?) GROUP BY customer_id,pool) GROUP BY pool", [str(path)]).fetchdf().to_dict("records")
        cpath = candidate.ART / cutoff / "candidates.parquet"
        fields = " AND ".join(f"s.{f}::FLOAT IS NOT DISTINCT FROM c.{f}::FLOAT" for f in candidate.FEATURES)
        mismatch = db.execute(f"SELECT count(*) FROM read_parquet(?) s FULL JOIN read_parquet(?) c ON s.customer_id=c.customer_id AND s.article_id=c.article_id WHERE (s.pool='mind_new' OR c.article_id IS NOT NULL) AND (s.article_id IS NULL OR c.article_id IS NULL OR s.target<>c.target OR NOT({fields}))", [str(path), str(cpath)]).fetchone()[0]
    candidate.require(mismatch == 0, "shared features changed MIND side relative to003")
    meta = {"schema": "mind-shared-novel-supervision-v1", "cutoff": cutoff, "rows": len(frame), "groups": groups,
            "dense": dense_meta, "MIND_features_exactly_reproduced": True, "MIND_feature_provenance": "exact frozen 003 input; only base side uses rebatched dense calculation", "seconds": time.perf_counter() - start, "final_week": "not_run"}
    relation.save(marker, meta)
    relation.progress(f"shared features {cutoff}: {len(frame):,} rows, {meta['seconds']:.1f}s")
    return path, meta


def fit_shared(trial, training, objective):
    folder = ART / trial / "models" / ("through_" + training[-1])
    folder.mkdir(parents=True, exist_ok=True)
    path, marker = folder / "model.txt", folder / "MODEL.json"
    if path.exists() and marker.exists():
        return lgb.Booster(model_file=str(path)), relation.read(marker)
    start = time.perf_counter()
    hardware = budget()
    params = relation.read(candidate.CONTRACT)["model"].copy()
    params.pop("selection")
    rounds = params.pop("rounds")
    if objective == "binary":
        params.update(objective="binary", metric="binary_logloss")
        params.pop("eval_at")
        params.pop("lambdarank_truncation_level")
    with candidate.connection() as db:
        q = " UNION ALL ".join(f"SELECT '{cutoff}' sample_cutoff,* FROM read_parquet('{relation.old._sql_path(ART / 'shared' / cutoff / 'features.parquet')}')" for cutoff in training)
        db.execute("CREATE VIEW training AS " + q)
        n = db.execute("SELECT count(*) FROM training").fetchone()[0]
        candidate.require(n <= 6500000, "registered training-row cap exceeded")
        frame = db.execute(f"SELECT sample_cutoff,customer_id,pool,target,{','.join(candidate.FEATURES)} FROM training ORDER BY sample_cutoff,customer_id,pool,article_id").fetchdf()
    g = frame.groupby(["sample_cutoff", "customer_id", "pool"], sort=False).target.agg(["sum", "size"])
    active = g[(g["sum"] > 0) & (g["sum"] < g["size"])].groupby(level="pool").size().to_dict()
    if objective == "lambdarank":
        factor = active["mind_new"] / active["base_novel"]
        reason = "balance mixed-label group counts across source tasks"
    else:
        counts = frame.groupby("pool").size().to_dict()
        factor = counts["mind_new"] / counts["base_novel"]
        reason = "balance source row mass without class reweighting; scores not globally calibrated probabilities"
    weights = np.where(frame.pool == "mind_new", 1., factor).astype(np.float32)
    labels = frame.target.to_numpy(np.int8)
    matrix = frame[candidate.FEATURES].to_numpy(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    groups = g["size"].astype(int).tolist()
    summary = {"candidate_rows": len(frame), "positive_rows": int(labels.sum()), "groups": len(groups),
               "mixed_groups_by_source": active, "base_weight": factor, "mind_weight": 1., "weight_reason": reason}
    del frame, g
    gc.collect()
    dataset = lgb.Dataset(matrix, label=labels, weight=weights, group=groups if objective == "lambdarank" else None, feature_name=candidate.FEATURES, free_raw_data=True)
    def deadline(_):
        if datetime.now(timezone.utc) >= datetime.fromisoformat(relation.read(SESSION)["deadline_utc"]):
            raise TimeoutError("one-hour deadline reached inside fitting")
    relation.progress(f"{trial} {objective} fit through {training[-1]}: {n:,} rows; base weight={factor:.4f}")
    model = lgb.train(params, dataset, num_boost_round=rounds, callbacks=[deadline])
    model.save_model(str(path))
    meta = {"trial": trial, "objective": objective, "training": training, "features": candidate.FEATURES, "params": params,
            "rounds": rounds, "model": str(path), "input": summary, "resources": hardware, "seconds": time.perf_counter() - start,
            "importance": sorted([{"feature": f, "gain": float(v)} for f, v in zip(candidate.FEATURES, model.feature_importance("gain"))], key=lambda x: -x["gain"])}
    relation.save(marker, meta)
    del matrix, labels, weights, dataset
    gc.collect()
    return model, meta


def evaluate_order(trial, cutoff, model):
    folder = ART / trial / cutoff
    folder.mkdir(parents=True, exist_ok=True)
    with candidate.connection() as db:
        frame = db.execute(f"SELECT {','.join(candidate.KEYS + candidate.FEATURES)} FROM read_parquet(?) ORDER BY customer_id,article_id", [str(candidate.ART / cutoff / "candidates.parquet")]).fetchdf()
    x = frame[candidate.FEATURES].to_numpy(np.float32)
    x[~np.isfinite(x)] = np.nan
    scores = frame[candidate.KEYS].copy()
    scores["score"] = model.predict(x, num_threads=2)
    order = candidate.order_candidates(scores)
    relation.parquet(order, folder / "order.parquet")
    with candidate.connection() as db:
        labels = db.execute("SELECT customer_id,article_id,target FROM read_parquet(?)", [str(candidate.ART / cutoff / "candidates.parquet")]).fetchdf()
    total = relation.read(candidate.ART / cutoff / "INPUT.json")["total_users"]
    return candidate.candidate_metrics(order, labels, total)


def admit(trial, cutoff, training):
    folder = ART / trial / cutoff
    with candidate.connection() as db:
        winners = db.execute("SELECT customer_id,article_id AS challenger_article_id FROM read_parquet(?) WHERE rank=1", [str(folder / "order.parquet")]).fetchdf()
        db.register("winners", winners)
        features = relation.FEATURES["dense_primary"]
        fields = list(dict.fromkeys(relation.KEYS + features))
        frame = db.execute(f"SELECT {','.join('p.' + x for x in fields)} FROM read_parquet(?) p SEMI JOIN winners USING(customer_id,challenger_article_id)", [str(relation.ART / cutoff / "pairs.parquet")]).fetchdf()
    candidate.require(len(frame) == len(winners) * 5, "winner x5 identity changed")
    model = lgb.Booster(model_file=str(relation.model_path(training, "dense_primary")))
    x = frame[features].to_numpy(np.float32)
    x[~np.isfinite(x)] = np.nan
    scores = frame[relation.KEYS].copy()
    scores[relation.PROBS] = model.predict(x, num_threads=2)
    actions = relation.select_actions(scores)
    relation.parquet(actions, folder / "decisions.parquet")
    return evaluate_actions(folder, cutoff, actions)


def evaluate_actions(folder, cutoff, actions):
    baseline, meta = reconstruct_full_baseline(cutoff)
    with candidate.connection() as db:
        labels = db.execute("SELECT * FROM read_parquet(?)", [str(relation.ART / cutoff / "labels.parquet")]).fetchdf()
    evaluated = relation.old.exact_action_deltas(baseline, actions, labels)
    if evaluated.empty:
        evaluated = actions.assign(actual_delta=pd.Series(dtype=float), challenger_target=pd.Series(dtype=int), victim_target=pd.Series(dtype=int))
    relation.parquet(evaluated, folder / "evaluated.parquet")
    top = baseline[baseline.champion_rank <= 12].sort_values(["customer_id", "champion_rank"])
    users = pd.Index(top.customer_id.drop_duplicates())
    y = top.target.to_numpy(np.uint8).reshape(-1, 12)
    truth = top.groupby("customer_id", sort=False).truth_count.first().to_numpy(int)
    changed = y.copy()
    if len(evaluated):
        indexes = users.get_indexer(evaluated.customer_id)
        candidate.require((indexes >= 0).all(), "action user outside baseline")
        changed[indexes, evaluated.champion_rank.to_numpy(int) - 1] = evaluated.challenger_target.to_numpy(np.uint8)
    delta = float(evaluated.actual_delta.sum() / meta["total_users"])
    independent_delta = float((independent.ap_matrix(changed, truth) - independent.ap_matrix(y, truth)).mean())
    candidate.require(abs(delta - independent_delta) < 1e-12, "independent MAP mismatch")
    return {"MAP@12": meta["MAP@12"] + delta, "baseline_MAP@12": meta["MAP@12"], "delta_vs_baseline": delta,
            "total_users": meta["total_users"], "selected_users": len(evaluated), "beneficial_users": int((evaluated.actual_delta > 1e-15).sum()),
            "harmful_users": int((evaluated.actual_delta < -1e-15).sum()), "positive_challengers": int(evaluated.challenger_target.sum()),
            "lost_positive_victims": int(evaluated.victim_target.sum()), "independent_MAP_error": abs(delta - independent_delta),
            "selection_used_labels": False, "protected_head_changes": 0, "maximum_actions_per_user": 1}


def run_trial(number):
    trial = f"MIND-WARM-RANK-{number:03d}"
    contract = REPORT / f"MIND_WARM_RANK{number:03d}_CONTRACT.json"
    config = relation.read(contract)
    output = REPORT / f"MIND_WARM_RANK{number:03d}_METRICS.json"
    if output.exists() and relation.read(output).get("status", "").startswith("completed"):
        return relation.read(output)
    objective = "lambdarank" if number == 4 else "binary"
    result = {"trial": trial, "status": "running", "windows": {}, "models": {}, "inputs": {}, "final_week": "not_run", "independent_confirmation": "not_run"}
    start = time.perf_counter()
    try:
        for cutoff, training in config["schedule"].items():
            budget()
            relation.check_temporal(training, cutoff)
            for t in training:
                _, result["inputs"][t] = shared_input(t)
            model, result["models"][cutoff] = fit_shared(trial, training, objective)
            internal = evaluate_order(trial, cutoff, model)
            actions = admit(trial, cutoff, training)
            result["windows"][cutoff] = {"internal": internal, "admission": actions}
            relation.save(output, result)
            relation.progress(f"{trial} {cutoff}: Hit1={internal['conditional_hit_at_1']:.5f}; deltaMAP={actions['delta_vs_baseline']:+.8f}; actions={actions['selected_users']}")
            del model
            gc.collect()
        values = [v["admission"]["delta_vs_baseline"] for v in result["windows"].values()]
        result["gate"] = {"passed": bool(np.mean(values) > 0 and all(v >= 0 for v in values) and min(values) >= -.0002), "mean_delta": float(np.mean(values)), "nondegrade_windows": sum(v >= 0 for v in values), "worst_delta": min(values)}
        result["status"] = "completed_development_pass" if result["gate"]["passed"] else "completed_rejected"
    except Exception as exc:
        result["status"] = "deadline_stopped" if isinstance(exc, TimeoutError) else "engineering_stopped"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["seconds"] = time.perf_counter() - start
        relation.save(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "004", "005"])
    args = parser.parse_args()
    if args.command == "audit":
        output = audit()
        print({k: {"pools": v["pool_counts"], "mind_pos": v["mind_positive_pairs"], "type_supported": v["mind_positive_type_supported_by_base_novel"], "item_supported": v["mind_positive_item_supported_by_other_base_novel_user"], "novel": v["mind_positive_novel_to_user"]} for k, v in output["cutoffs"].items()}, flush=True)
    else:
        run_trial(int(args.command))
