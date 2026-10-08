"""Post-freeze upper-bound and paired-user uncertainty audit, no selection."""
import time
import duckdb
import numpy as np
from . import mind_warm_relation_audit as old

ART = old.ROOT / "artifacts/mind_warm_side/MIND-WARM-TRANSFER"


def bootstrap_deltas(delta, total_users, rng, repetitions=20000):
    values, counts = np.unique(np.r_[np.asarray(delta, float), np.zeros(total_users - len(delta))], return_counts=True)
    # Multinomial counts exactly reproduce resampling all paired user deltas,
    # including unchanged users, without allocating repeated full user matrices.
    draw = rng.multinomial(total_users, counts / total_users, size=repetitions)
    samples = (draw @ values) / total_users
    return samples


def run():
    started = time.perf_counter()
    rng = np.random.default_rng(20260912)
    result = {"windows": {}, "uncertainty": {}, "posthoc_only": True, "final_week": "not_run",
              "uncertainty_scope": "paired-user bootstrap conditional on fixed development windows, not temporal generalization, not selection-adjusted after adaptive experiments",
              "bootstrap_replicates": 20000, "seed": 20260912}
    draws = {5: [], 9: []}
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.execute("SET memory_limit='512MiB'")
        for cutoff in ("2020-03-18", "2020-06-24", "2020-08-19"):
            meta = old.read(old.ART / cutoff / "PAIRS.json")["baseline"]
            _, arrays = old.frozen_baseline(db, meta)
            candidates = old.ROOT / "artifacts/mind_warm_side/MIND-WARM-RANK-003" / cutoff / "candidates.parquet"
            positive_users = db.execute("SELECT DISTINCT customer_id FROM read_parquet(?) WHERE target=1", [str(candidates)]).fetchnumpy()["customer_id"]
            first_positive = db.execute("SELECT r.customer_id FROM read_parquet(?) r JOIN read_parquet(?) t USING(customer_id,article_id) WHERE r.rank=1 AND t.target=1", [str(ART / "MIND-WARM-RANK-005" / cutoff / "order.parquet"), str(candidates)]).fetchnumpy()["customer_id"]
            full = old.constrained_oracle(arrays["labels"], arrays["truth"], np.isin(arrays["users"], positive_users))
            top1 = old.constrained_oracle(arrays["labels"], arrays["truth"], np.isin(arrays["users"], first_positive))
            result["windows"][cutoff] = {"users": len(arrays["users"]), "MIND_positive_candidate_users": len(positive_users), "005_positive_first_choices": len(first_positive), "full_MIND_best_single_tail_replacement_oracle_delta": float(full.mean()), "005_fixed_first_choice_oracle_delta": float(top1.mean()), "retained_oracle_fraction": float(top1.sum() / full.sum())}
            for number in (5, 9):
                metrics = old.read(old.REPORT / f"MIND_WARM_RANK{number:03d}_METRICS.json")
                window = metrics["windows"][cutoff]
                if number == 5:
                    window = window["admission"]
                delta = db.execute("SELECT actual_delta FROM read_parquet(?)", [str(ART / f"MIND-WARM-RANK-{number:03d}" / cutoff / "evaluated.parquet")]).fetchnumpy()["actual_delta"]
                old.require(abs(float(delta.sum() / len(arrays["users"])) - window["delta_vs_baseline"]) < 1e-12, "paired baseline mismatch")
                samples = bootstrap_deltas(delta, len(arrays["users"]), rng)
                draws[number].append(samples)
                result["uncertainty"].setdefault(str(number), {})[cutoff] = {"mean_delta": window["delta_vs_baseline"], "paired_user_percentile_CI95": np.quantile(samples, [.025, .975]).tolist(), "nonzero_user_deltas": int(np.count_nonzero(delta)), "selected_users": len(delta), "all_users": len(arrays["users"])}
    for number, values in draws.items():
        samples = np.mean(values, axis=0)
        result["uncertainty"][str(number)]["three_window_equal_mean_CI95"] = np.quantile(samples, [.025, .975]).tolist()
    result["seconds"] = time.perf_counter() - started
    old.write(old.REPORT / "MIND_WARM_HOUR_DIAGNOSTICS.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    run()
