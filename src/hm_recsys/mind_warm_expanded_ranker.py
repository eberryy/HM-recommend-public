"""012 within-MIND LambdaRank using the fixed011 expanded representation."""
import gc
import time
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np

from . import mind_warm_expanded as expanded
from . import mind_warm_transfer as transfer
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation

TRIAL = "MIND-WARM-RANK-012"
ART = transfer.ART / TRIAL


def fit(training):
    folder = ART / "models" / ("through_" + training[-1])
    path, marker = folder / "model.txt", folder / "MODEL.json"
    if path.exists() and marker.exists():
        meta = relation.read(marker)
        candidate.require(meta["training"] == training and meta["features"] == expanded.FEATURES, "stale012 model")
        return lgb.Booster(model_file=str(path)), meta
    start = time.perf_counter()
    hardware = transfer.budget()
    folder.mkdir(parents=True, exist_ok=True)
    config = relation.read(candidate.CONTRACT)["model"].copy()
    rounds = config.pop("rounds")
    config.pop("selection")
    with candidate.connection() as db:
        query = " UNION ALL ".join(f"SELECT '{cutoff}' sample_cutoff,* FROM read_parquet('{relation.old._sql_path(expanded.ART / 'input' / cutoff / 'features.parquet')}') WHERE pool='mind_new'" for cutoff in training)
        db.execute("CREATE VIEW train AS " + query)
        frame = db.execute(f"SELECT sample_cutoff,customer_id,target,{','.join(expanded.FEATURES)} FROM train ORDER BY sample_cutoff,customer_id,article_id").fetchdf()
    sizes = frame.groupby(["sample_cutoff", "customer_id"], sort=False).size().astype(int).tolist()
    groups = frame.groupby(["sample_cutoff", "customer_id"], sort=False).target.agg(["sum", "size"])
    mixed = (groups["sum"] > 0) & (groups["sum"] < groups["size"])
    candidate.require(len(frame) <= 1500000 and int(mixed.sum()) >= 50, "012 supervision/resource gate")
    labels = frame.target.to_numpy(np.int8)
    matrix = frame[expanded.FEATURES].to_numpy(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    evidence = {"rows": len(frame), "positives": int(labels.sum()), "groups": len(sizes), "mixed_groups": int(mixed.sum()), "mixed_rows": int(groups.loc[mixed, "size"].sum()), "zero_positive_groups": int((groups["sum"] == 0).sum())}
    del frame, groups
    gc.collect()
    dataset = lgb.Dataset(matrix, label=labels, group=sizes, feature_name=expanded.FEATURES, free_raw_data=True)
    deadline = datetime.fromisoformat(relation.read(transfer.SESSION)["deadline_utc"])
    def stop(_):
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("3h deadline reached inside012 fit")
    relation.progress(f"012 fit through {training[-1]}:{len(matrix):,} rows,{evidence['mixed_groups']} mixed groups")
    model = lgb.train(config, dataset, num_boost_round=rounds, callbacks=[stop])
    model.save_model(str(path))
    meta = {"training": training, "features": expanded.FEATURES, "params": config, "rounds": rounds, "input": evidence, "resources": hardware, "model": str(path), "seconds": time.perf_counter() - start,
            "importance": sorted([{"feature": f, "gain": float(g)} for f, g in zip(expanded.FEATURES, model.feature_importance("gain"))], key=lambda z: -z["gain"])}
    relation.save(marker, meta)
    del matrix, labels, dataset
    gc.collect()
    return model, meta


def run():
    started = time.perf_counter()
    contract = relation.read(relation.REPORT / "MIND_WARM_RANK012_CONTRACT.json")
    output = relation.REPORT / "MIND_WARM_RANK012_METRICS.json"
    result = {"status": "running", "models": {}, "windows": {}, "final_week": "not_run", "independent_confirmation": "not_run"}
    try:
        for cutoff, training in contract["schedule"].items():
            relation.check_temporal(training, cutoff)
            for t in training:
                expanded.input_table(t)
            model, result["models"][cutoff] = fit(training)
            internal, admission = expanded.score_and_admit(cutoff, training, model, ART)
            result["windows"][cutoff] = {"internal": internal, "admission": admission}
            relation.save(output, result)
            relation.progress(f"012 {cutoff}:Hit1={internal['conditional_hit_at_1']:.5f},Hit5={internal['conditional_hit_at_5']:.5f},deltaMAP={admission['delta_vs_baseline']:+.8f}")
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
