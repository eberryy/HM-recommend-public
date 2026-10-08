"""Historical out-of-fit score reliability audit; no recommendation fitting."""
import lightgbm as lgb
import numpy as np

from . import mind_warm_transfer as transfer
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation


def oof_input(cutoff):
    """OOF here means strictly earlier dated fitting, not random folds."""
    candidate.require(cutoff in ("2020-03-18", "2020-06-24"), "only earlier available calibration periods")
    path = transfer.ART / "calibration" / cutoff / "oof.parquet"
    if path.exists():
        return path
    transfer.budget()
    metrics = relation.read(relation.REPORT / "MIND_WARM_RANK005_METRICS.json")
    meta = metrics["models"][cutoff]
    relation.check_temporal(meta["training"], cutoff)
    with candidate.connection() as db:
        frame = db.execute("SELECT * FROM read_parquet(?)", [str(transfer.ART / "shared" / cutoff / "features.parquet")]).fetchdf()
    model = lgb.Booster(model_file=meta["model"])
    x = frame[candidate.FEATURES].to_numpy(np.float32)
    x[~np.isfinite(x)] = np.nan
    result = frame[candidate.KEYS + ["pool", "target"]].copy()
    result["score"] = model.predict(x, num_threads=2)
    relation.parquet(result, path)
    relation.save(path.with_name("OOF.json"), {"cutoff": cutoff, "model_training": meta["training"], "rows": len(result), "score_before_labels": "features exclude labels; historical calibration labels attached after prediction", "final_week": "not_run"})
    return path


def audit():
    result = {"windows": {}, "final_week": "not_run", "bins": [0, .001, .002, .005, .01, .02, .05, 1], "purpose": "descriptive reliability only; no threshold selection"}
    with candidate.connection() as db:
        for cutoff in ("2020-03-18", "2020-06-24"):
            path = oof_input(cutoff)
            db.execute("CREATE OR REPLACE TEMP TABLE oof AS SELECT * FROM read_parquet(?)", [str(path)])
            summaries = db.execute("SELECT pool,count(*) row_count,sum(target) positives,avg(target) actual_rate,avg(score) predicted_rate FROM oof GROUP BY pool").fetchdf().to_dict("records")
            bins = []
            for lo, hi in zip(result["bins"][:-1], result["bins"][1:]):
                for row in db.execute("SELECT pool,count(*) row_count,sum(target) positives,avg(score) predicted_rate,avg(target) actual_rate FROM oof WHERE score>=? AND score<? GROUP BY pool", [lo, hi]).fetchdf().to_dict("records"):
                    bins.append(dict(lower=lo, upper=hi, **row))
            result["windows"][cutoff] = {"summary": summaries, "bins": bins}
    relation.save(relation.REPORT / "MIND_WARM_SCORE_RELIABILITY_AUDIT.json", result)
    print({c: r["summary"] for c, r in result["windows"].items()}, flush=True)
    return result


if __name__ == "__main__":
    audit()
