"""MIND-WARM-RANK-003: direct within-source ordering, then gated admission."""
from __future__ import annotations

import gc
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import mind_warm_relation as relation
from . import mind_warm_relation_audit as independent
from .mind_warm_dense import DENSE_FEATURES
from .mind_warm_baseline import reconstruct_full_baseline

ROOT = relation.ROOT
REPORT = relation.REPORT
ART = ROOT / "artifacts/mind_warm_side/MIND-WARM-RANK-003"
CONTRACT = REPORT / "MIND_WARM_RANK003_CONTRACT.json"
METRICS = REPORT / "MIND_WARM_RANK003_METRICS.json"
SCHEMA = "mind-warm-candidate-ranker-v1"
FEATURES = [*relation.USER, *relation.ITEM, *DENSE_FEATURES]
SOURCE_FEATURES = [*relation.USER, *[f"c_{f}" for f in [*relation.ITEM, *DENSE_FEATURES]]]
KEYS = ["customer_id", "article_id"]
ORDERINGS = ["raw_mind", "rank002_favorite", "candidate_ranker"]


def connection():
    db = relation.con()
    db.execute("SET threads=2")
    return db


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def budget(start):
    stats = relation.resources()
    require(time.perf_counter() - start < 45 * 60, "registered45-minute budget reached; stop without downsizing")
    return stats


def order_candidates(scores: pd.DataFrame) -> pd.DataFrame:
    require(set(scores.columns) == set(KEYS + ["score"]), "candidate order takes only identity and score, never labels")
    require(not scores[KEYS].isna().any().any() and not scores.duplicated(KEYS).any(), "candidate keys must be unique/non-null")
    require(np.isfinite(scores.score.to_numpy(float)).all(), "ordering scores must be finite")
    result = scores.sort_values(["customer_id", "score", "article_id"], ascending=[True, False, True], kind="mergesort").copy()
    result["rank"] = result.groupby("customer_id", sort=False).cumcount() + 1
    return result.reset_index(drop=True)


def candidate_metrics(order: pd.DataFrame, labels: pd.DataFrame, total_users: int) -> dict:
    require(set(order.columns) == set(KEYS + ["score", "rank"]), "invalid frozen ordering schema")
    require(not labels.duplicated(KEYS).any() and labels.target.isin([0, 1]).all(), "invalid candidate labels")
    joined = order.merge(labels, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    require((joined._merge == "both").all(), "candidate ordering changed the source identity")
    require(joined.groupby("customer_id")["rank"].apply(lambda x: sorted(x) == list(range(1, len(x) + 1))).all(), "noncontiguous candidate ranks")
    positives = joined.groupby("customer_id").target.sum()
    positive_users = int((positives > 0).sum())
    n_positive = int(positives.sum())
    require(total_users >= len(positives) and positive_users > 0, "invalid evaluation population")
    out = {"total_users": total_users, "candidate_users": len(positives), "positive_candidate_users": positive_users,
           "candidate_rows": len(joined), "positive_candidate_rows": n_positive}
    for k in (1, 5):
        counts = joined.loc[(joined["rank"] <= k) & (joined.target == 1)].groupby("customer_id").size()
        hits = len(counts)
        out[f"hit_users_at_{k}"] = hits
        out[f"conditional_hit_at_{k}"] = hits / positive_users
        out[f"all_user_hit_at_{k}"] = hits / total_users
        out[f"positive_pair_recall_at_{k}"] = int(counts.sum()) / n_positive
        out[f"macro_pool_recall_at_{k}"] = float((counts.reindex(positives.index, fill_value=0)[positives > 0] / positives[positives > 0]).mean())
    first = joined[joined.target == 1].groupby("customer_id")["rank"].min()
    out["conditional_MRR"] = float((1. / first).sum() / positive_users)
    out["all_user_MRR"] = float((1. / first).sum() / total_users)
    out["candidate_identity_conserved"] = True
    return out


def input_table(cutoff: str):
    relation.old._guard(cutoff)
    folder = ART / cutoff
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "candidates.parquet"
    marker = folder / "INPUT.json"
    if path.exists() and marker.exists():
        meta = relation.read(marker)
        require(meta["schema"] == SCHEMA and meta["features"] == FEATURES, "stale003 input cache")
        return path, meta
    source = relation.ART / cutoff / "pairs.parquet"
    meta002 = relation.read(source.with_name("PAIRS.json"))
    require(meta002["schema"] == relation.SCHEMA and meta002["stage"] == "outer", "complete chronological002 source required")
    columns = [f"{source_name} AS {name}" for source_name, name in zip(SOURCE_FEATURES, FEATURES)]
    with connection() as db:
        db.execute(f"CREATE VIEW source AS SELECT * FROM read_parquet('{relation.old._sql_path(source)}')")
        # Check that five copies really contain identical challenger evidence.
        vector = ",".join(["challenger_target", *SOURCE_FEATURES])
        errors = db.execute(f"SELECT count(*) FROM (SELECT customer_id,challenger_article_id FROM source GROUP BY 1,2 HAVING count(*)<>5 OR count(DISTINCT champion_rank)<>5 OR min(champion_rank)<>8 OR max(champion_rank)<>12 OR count(DISTINCT({vector}))<>1)").fetchone()[0]
        require(errors == 0, "source pair copies are incomplete or challenger fields disagree")
        db.execute(f"COPY (SELECT customer_id,challenger_article_id AS article_id,challenger_target::UTINYINT AS target,{','.join(columns)} FROM source WHERE champion_rank=8 ORDER BY customer_id,article_id) TO '{relation.old._sql_path(path)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        counts = db.execute("SELECT count(*),count(DISTINCT(customer_id,article_id)),sum(target),count(DISTINCT customer_id) FROM read_parquet(?)", [str(path)]).fetchone()
        groups = db.execute("WITH g AS (SELECT customer_id,count(*) n,sum(target) p FROM read_parquet(?) GROUP BY 1) SELECT count(*),count(*) FILTER(WHERE p=0),count(*) FILTER(WHERE p>0),count(*) FILTER(WHERE p>0 AND p<n),sum(n) FILTER(WHERE p>0 AND p<n),min(n),max(n) FROM g", [str(path)]).fetchone()
    require(counts[0] == counts[1] == meta002["challenger_rows"] and counts[2] == meta002["challenger_positive_rows"], "candidate identity/positive conservation failed")
    meta = {"schema": SCHEMA, "cutoff": cutoff, "features": FEATURES, "source": str(source), "candidate_rows": counts[0],
            "positive_rows": counts[2], "users": counts[3], "zero_positive_groups": groups[1], "positive_groups": groups[2],
            "mixed_label_groups": groups[3], "rows_in_mixed_label_groups": groups[4], "min_candidates": groups[5], "max_candidates": groups[6],
            "five_copies_identical": True, "candidate_identity_conserved": True, "total_users": meta002["baseline"]["total_users"], "final_week": "not_run"}
    relation.save(marker, meta)
    return path, meta


def fit(training: list[str], config: dict, started: float):
    folder = ART / "models" / ("through_" + training[-1])
    folder.mkdir(parents=True, exist_ok=True)
    model_path, marker = folder / "model.txt", folder / "MODEL.json"
    if model_path.exists() and marker.exists():
        meta = relation.read(marker)
        require(meta["schema"] == SCHEMA and meta["training"] == training and meta["config"] == config and meta["features"] == FEATURES, "stale model evidence")
        return lgb.Booster(model_file=str(model_path)), meta
    start = time.perf_counter()
    hardware = budget(started)
    with connection() as db:
        queries = [f"SELECT '{cutoff}' sample_cutoff,* FROM read_parquet('{relation.old._sql_path(ART / cutoff / 'candidates.parquet')}')" for cutoff in training]
        db.execute("CREATE VIEW training AS " + " UNION ALL ".join(queries))
        count = db.execute("SELECT count(*) FROM training").fetchone()[0]
        require(count <= 1500000, "candidate training resource cap exceeded")
        frame = db.execute(f"SELECT sample_cutoff,customer_id,{','.join(FEATURES)},target FROM training ORDER BY sample_cutoff,customer_id,article_id").fetchdf()
    sizes = frame.groupby(["sample_cutoff", "customer_id"], sort=False).size().astype(int).tolist()
    grouped = frame.groupby(["sample_cutoff", "customer_id"], sort=False).target.agg(["sum", "size"])
    mixed = (grouped["sum"] > 0) & (grouped["sum"] < grouped["size"])
    require(sum(sizes) == len(frame) and int(frame.target.sum()) >= 100 and int(mixed.sum()) >= 50, "insufficient usable within-user supervision")
    evidence = {"candidate_rows": len(frame), "groups": len(sizes), "positive_rows": int(frame.target.sum()),
                "zero_positive_groups": int((grouped["sum"] == 0).sum()), "mixed_label_groups": int(mixed.sum()),
                "rows_in_mixed_label_groups": int(grouped.loc[mixed, "size"].sum()), "group_key": "cutoff+customer_id"}
    labels = frame.target.to_numpy(np.int8)
    matrix = frame[FEATURES].to_numpy(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    del frame, grouped
    gc.collect()
    params = {k: v for k, v in config.items() if k not in {"rounds", "selection"}}
    dataset = lgb.Dataset(matrix, label=labels, group=sizes, feature_name=FEATURES, free_raw_data=True)
    def deadline(_):
        require(time.perf_counter() - started < 45 * 60, "fit stopped at registered deadline")
    relation.progress(f"003 fit through {training[-1]}: {len(matrix):,} candidates, {len(sizes)} groups, {evidence['mixed_label_groups']} mixed-label groups")
    model = lgb.train(params, dataset, num_boost_round=config["rounds"], callbacks=[deadline])
    model.save_model(str(model_path))
    meta = {"schema": SCHEMA, "training": training, "config": config, "features": FEATURES, "input": evidence,
            "resources": hardware, "model": str(model_path), "elapsed_seconds": time.perf_counter() - start,
            "importance": sorted([{"feature": f, "gain": float(g)} for f, g in zip(FEATURES, model.feature_importance("gain"))], key=lambda x: -x["gain"])}
    relation.save(marker, meta)
    del dataset, matrix, labels
    gc.collect()
    return model, meta


def frozen_relation_scores(cutoff, training):
    output = ART / cutoff / "rank002-candidate-scores.parquet"
    require(not output.exists(), "frozen score cache already exists without completed window; preserve and inspect before resuming")
    path = relation.model_path(training, "dense_primary")
    meta = relation.read(path.with_suffix(".json"))
    require(meta["training"] == training and meta["features"] == relation.FEATURES["dense_primary"], "wrong frozen relation prefix")
    model = lgb.Booster(model_file=str(path))
    require(model.feature_name() == meta["features"], "actual frozen model feature mismatch")
    parts = []
    with connection() as db:
        selected = list(dict.fromkeys(relation.KEYS + meta["features"]))
        cursor = db.execute(f"SELECT {','.join(selected)} FROM read_parquet(?)", [str(relation.ART / cutoff / "pairs.parquet")])
        while True:
            batch = cursor.fetch_df_chunk(32)
            if batch.empty:
                break
            matrix = batch[meta["features"]].to_numpy(np.float32)
            matrix[~np.isfinite(matrix)] = np.nan
            probs = model.predict(matrix, num_threads=2)
            require(np.isfinite(probs).all() and np.allclose(probs.sum(axis=1), 1.), "invalid relation probabilities")
            values = batch[["customer_id", "challenger_article_id"]].rename(columns={"challenger_article_id": "article_id"})
            values["score"] = (probs[:, 2] - probs[:, 0]) / batch.champion_rank.to_numpy()
            parts.append(values.groupby(KEYS, sort=False, as_index=False).score.max())
    scores = pd.concat(parts, ignore_index=True).groupby(KEYS, sort=False, as_index=False).score.max()
    relation.parquet(scores, output)
    return scores


def evaluate_internal(cutoff, training, model, input_meta):
    folder = ART / cutoff
    result_path = folder / "INTERNAL.json"
    if result_path.exists():
        return relation.read(result_path)
    relation.check_temporal(training, cutoff)
    with connection() as db:
        frame = db.execute(f"SELECT {','.join(KEYS + FEATURES)} FROM read_parquet(?) ORDER BY customer_id,article_id", [str(folder / "candidates.parquet")]).fetchdf()
        raw = db.execute("SELECT m.customer_id,m.article_id,-m.mind_rank::DOUBLE AS score FROM read_parquet(?) m SEMI JOIN read_parquet(?) c USING(customer_id,article_id)", [str(relation.old._mind_candidates(cutoff)), str(folder / "candidates.parquet")]).fetchdf()
    matrix = frame[FEATURES].to_numpy(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    scores = frame[KEYS].copy()
    scores["score"] = model.predict(matrix, num_threads=2)
    old_scores = frozen_relation_scores(cutoff, training)
    # Freeze all three label-free orderings before reading evaluation labels.
    orders = {k: order_candidates(v) for k, v in zip(ORDERINGS, [raw, old_scores, scores])}
    for key, order in orders.items():
        relation.parquet(order, folder / f"{key}-order.parquet")
    with connection() as db:
        labels = db.execute("SELECT customer_id,article_id,target FROM read_parquet(?)", [str(folder / "candidates.parquet")]).fetchdf()
    result = {k: candidate_metrics(v, labels, input_meta["total_users"]) for k, v in orders.items()}
    factor = relation.read(REPORT / "MIND_WARM_RANK002_FACTOR_AUDIT.json")["windows"][cutoff]
    require(result["rank002_favorite"]["hit_users_at_1"] == factor["model_favorite_challenger_positive_users"], "frozen002 first-choice replay differs")
    relation.save(result_path, result)
    relation.progress(f"003 {cutoff} conditionalHit@1 " + str({k: round(v['conditional_hit_at_1'], 6) for k, v in result.items()}))
    return result


def internal_gate(windows):
    comparisons = {}
    for reference in ORDERINGS[:2]:
        changes = {cutoff: value["candidate_ranker"]["conditional_hit_at_1"] - value[reference]["conditional_hit_at_1"] for cutoff, value in windows.items()}
        comparisons[reference] = {"per_window_delta": changes, "mean_delta": float(np.mean(list(changes.values()))),
                                  "nondegrade_windows": sum(v >= 0 for v in changes.values())}
    passed = len(windows) == 3 and all(x["mean_delta"] > 0 and x["nondegrade_windows"] == 3 for x in comparisons.values())
    return {"passed": passed, "comparisons": comparisons}


def admission(cutoff, training):
    folder = ART / cutoff
    with connection() as db:
        winners = db.execute("SELECT customer_id,article_id AS challenger_article_id FROM read_parquet(?) WHERE rank=1", [str(folder / "candidate_ranker-order.parquet")]).fetchdf()
        db.register("winners", winners)
        features = relation.FEATURES["dense_primary"]
        fields = list(dict.fromkeys(relation.KEYS + features))
        frame = db.execute(f"SELECT {','.join('p.' + x for x in fields)} FROM read_parquet(?) p SEMI JOIN winners USING(customer_id,challenger_article_id)", [str(relation.ART / cutoff / "pairs.parquet")]).fetchdf()
    require(len(frame) == 5 * len(winners), "each frozen winner needs exactly five original victims")
    model = lgb.Booster(model_file=str(relation.model_path(training, "dense_primary")))
    matrix = frame[features].to_numpy(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    scores = frame[relation.KEYS].copy()
    scores[relation.PROBS] = model.predict(matrix, num_threads=2)
    actions = relation.select_actions(scores)
    relation.parquet(actions, folder / "admission-decisions.parquet")
    baseline, meta = reconstruct_full_baseline(cutoff)
    with connection() as db:
        labels = db.execute("SELECT * FROM read_parquet(?)", [str(relation.ART / cutoff / "labels.parquet")]).fetchdf()
    evaluated = relation.old.exact_action_deltas(baseline, actions, labels)
    if evaluated.empty:
        evaluated = actions.assign(actual_delta=pd.Series(dtype=float), challenger_target=pd.Series(dtype=int), victim_target=pd.Series(dtype=int))
    relation.parquet(evaluated, folder / "admission-evaluated.parquet")
    delta = float(evaluated.actual_delta.sum() / meta["total_users"])
    # Independent array calculation verifies the complete Top12, not additive approximations.
    top = baseline[baseline.champion_rank <= 12].sort_values(["customer_id", "champion_rank"])
    users = pd.Index(top.customer_id.drop_duplicates())
    y = top.target.to_numpy(np.uint8).reshape(-1, 12)
    truth = top.groupby("customer_id", sort=False).truth_count.first().to_numpy(int)
    before = independent.ap_matrix(y, truth)
    altered = y.copy()
    rows = users.get_indexer(evaluated.customer_id)
    require((rows >= 0).all(), "action user outside cohort")
    altered[rows, evaluated.champion_rank.to_numpy(int) - 1] = evaluated.challenger_target.to_numpy(np.uint8)
    after = independent.ap_matrix(altered, truth)
    require(abs(float((after - before).mean()) - delta) < 1e-12, "independent admission MAP mismatch")
    out = {"MAP@12": meta["MAP@12"] + delta, "baseline_MAP@12": meta["MAP@12"], "delta_vs_baseline": delta,
           "delta_vs_rank002": delta - relation.read(REPORT / "MIND_WARM_RANK002_METRICS.json")["chronological_development"][cutoff]["dense_primary"]["delta_vs_baseline"],
           "selected_users": len(evaluated), "beneficial_users": int((evaluated.actual_delta > 1e-15).sum()),
           "harmful_users": int((evaluated.actual_delta < -1e-15).sum()), "positive_challengers": int(evaluated.challenger_target.sum()),
           "lost_positive_victims": int(evaluated.victim_target.sum()), "maximum_actions_per_user": 1,
           "protected_head_changes": 0, "independent_MAP_error": abs(float((after - before).mean()) - delta)}
    relation.save(folder / "ADMISSION.json", out)
    return out


def run():
    if METRICS.exists() and relation.read(METRICS).get("status", "").startswith("completed"):
        relation.progress("003 already complete; no re-evaluation")
        return relation.read(METRICS)
    started = time.perf_counter()
    config = relation.read(CONTRACT)
    require(relation.read(REPORT / "MIND_WARM_RANK002_VERIFICATION.json")["passed"], "verified002 inputs required")
    ART.mkdir(parents=True, exist_ok=True)
    relation.save(ART / "FEATURE_ALLOWLIST.json", {"schema": SCHEMA, "features": FEATURES})
    result = {"schema": SCHEMA, "experiment_id": "MIND-WARM-RANK-003", "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "inputs": {}, "models": {}, "internal": {}, "admission": {}, "admission_status": "not_run",
              "final_week": "not_run", "independent_confirmation": "not_run", "promotion": "not_promoted"}
    try:
        for cutoff, training in config["schedule"].items():
            budget(started)
            relation.check_temporal(training, cutoff)
            for source in [*training, cutoff]:
                _, result["inputs"][source] = input_table(source)
            model, metadata = fit(training, config["model"], started)
            result["models"][cutoff] = metadata
            result["internal"][cutoff] = evaluate_internal(cutoff, training, model, result["inputs"][cutoff])
            relation.save(METRICS, result)
            del model
            gc.collect()
        result["internal_gate"] = internal_gate(result["internal"])
        if result["internal_gate"]["passed"]:
            for cutoff, training in config["schedule"].items():
                budget(started)
                result["admission"][cutoff] = admission(cutoff, training)
                relation.save(METRICS, result)
            deltas = [v["delta_vs_baseline"] for v in result["admission"].values()]
            result["admission_status"] = "completed"
            result["admission_gate"] = {"mean_delta": float(np.mean(deltas)), "nondegrade_windows": sum(x >= 0 for x in deltas),
                                        "worst_delta": min(deltas), "passed": bool(np.mean(deltas) > 0 and all(x >= 0 for x in deltas) and min(deltas) >= -.0002)}
            result["status"] = "completed_development_pass_not_promoted" if result["admission_gate"]["passed"] else "completed_admission_failed_baseline_retained"
        else:
            result["status"] = "completed_internal_failed_admission_not_run"
    except Exception as error:
        result["status"] = "engineering_stopped"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        relation.save(METRICS, result)
    relation.progress(result["status"])
    return result


if __name__ == "__main__":
    run()
