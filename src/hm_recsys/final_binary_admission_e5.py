"""FINAL-E5A: sampled binary supervision for globally comparable Cold admission."""
from pathlib import Path
import gc
import json
import shutil
import subprocess
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .final_b0_admission_e4 import evaluate_rule, stable_percentile
from .final_candidate_e2 import VALID, rrf
from .final_oracle_audit import dump
from .final_pseudocold_e3 import FEATURES, pseudo_matrix, cold_matrix
from .final_relation_pilot import identify
from .p42f_contract import earlier
from .p42f_data import frame
from .p43a_run import guard


ROOT = Path("artifacts/final/binary-admission-e5a-v1")
REPORT = Path("reports/final")
E3 = Path("artifacts/final/pseudocold-e3-v1")
E2 = Path("artifacts/final/candidate-e2-v1")
PREPARED = Path("artifacts/final/integration-10pct-v1/prepared")
HISTORY_ACTION = Path("artifacts/final/relation-pilot-v1")
B0_FIELDS = ["b0_score", "b0_rank_pct", "b0_user_zscore", "normalized_margin_to_rank2",
             "normalized_margin_to_rank5", "normalized_margin_to_user_median"]
ARMS = ("PB41", "PC42", "PC48")
ALPHAS = (0.05, 0.10, 0.25, 0.50)
QUOTAS = (0.001, 0.005, 0.01, 0.02)
PARAMS = dict(objective="binary", learning_rate=0.05, n_estimators=300, num_leaves=15,
              max_depth=6, min_child_samples=100, subsample=1.0, colsample_bytree=1.0,
              reg_lambda=1.0, reg_alpha=0.0, random_state=20260913, n_jobs=4,
              verbosity=-1, deterministic=True, force_col_wise=True)


def sampled_mask(keys, target, modulus):
    """Keep every positive and a stable hashed fraction of observed negatives."""
    hashed = pd.util.hash_pandas_object(keys[["customer_id", "article_id"]], index=False).to_numpy(np.uint64)
    return (target > 0) | (hashed % np.uint64(modulus) == 0)


def pseudo_training(repo, cutoff, arm):
    x, y, _ = pseudo_matrix(repo, repo / E3, cutoff)
    keys = frame(repo / E3 / cutoff / "pseudo.parquet")
    keep = sampled_mask(keys, y, 20)
    x = x[keep]; y = y[keep]
    domain = np.zeros((len(x), 1), np.float32)
    if arm == "PB41":
        return x, y, np.ones(len(y), np.float32)
    x = np.concatenate([x, domain], axis=1)
    if arm == "PC48":
        x = np.concatenate([x, np.full((len(x), len(B0_FIELDS)), np.nan, np.float32)], axis=1)
    return x, y, np.ones(len(y), np.float32)


def cold_training(repo, cutoff, arm):
    data, x, y, _ = cold_matrix(repo, repo / E2, cutoff)
    keys = data["cold"][["customer_id", "article_id"]]
    keep = sampled_mask(keys, y, 100)
    x = x[keep]; y = y[keep]
    x = np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1)
    if arm == "PC48":
        extra = data["cold"][B0_FIELDS].to_numpy(np.float32)[keep]
        x = np.concatenate([x, extra], axis=1)
    return x, y, np.full(len(y), 5.0, np.float32)


def validation_matrix(repo, cutoff, arm):
    data, x, y, _ = cold_matrix(repo, repo / E2, cutoff)
    if arm != "PB41":
        x = np.concatenate([x, np.ones((len(x), 1), np.float32)], axis=1)
    if arm == "PC48":
        x = np.concatenate([x, data["cold"][B0_FIELDS].to_numpy(np.float32)], axis=1)
    return data, x, y


def one_per_user(data, order_score):
    """Return one candidate row per candidate-bearing user, higher score first."""
    cold = data["cold"]
    work = pd.DataFrame({"index": np.arange(len(cold)), "user_index": cold.user_index.to_numpy(),
                         "score": np.asarray(order_score), "article_id": cold.article_id.to_numpy(str)})
    selected = work.sort_values(["user_index", "score", "article_id"], ascending=[True, False, True],
                                kind="stable").groupby("user_index", sort=False).head(1)
    if selected.user_index.duplicated().any():
        raise ValueError("candidate selector returned duplicate user")
    return selected["index"].to_numpy(int)


def action_scores(repo, cutoff, data):
    from .p42f_core import batches
    model = lgb.Booster(model_file=str(repo / HISTORY_ACTION / cutoff / "base70.txt"))
    scores = np.empty((len(data["cold"]), 12), float)
    for lo, cc, x, _ in batches(data, labels=False):
        scores[lo:lo + len(cc)] = model.predict(x, num_threads=4).reshape(-1, 12)
    return scores


def register(repo):
    contract = dict(status="preregistered_before_E5A_fit", stage="FINAL-E5A binary cross-user admission",
        motivation="E3 LambdaRank produced within-user scores and failed candidate Recall@1; E4-v2/v3 found no stable high-confidence region for B0-best using hand-built signals. Binary loss now targets cross-user comparability.",
        validation=VALID, training={t: earlier(t) for t in VALID},
        pseudo_population="WV3 WarmTop50, globally>5 pre-cutoff events, user-unseen; exact E3 materialization reused",
        real_population="actual time-safe Cold50 from earlier cutoffs; validation-date Cold labels never fit",
        features={"PB41": FEATURES, "PC42": FEATURES + ["real_cold_domain"],
                  "PC48": FEATURES + ["real_cold_domain"] + B0_FIELDS},
        sampling="all positives; stable pandas 64-bit hash of user-item keeps1/20 pseudo negatives and1/100 real-Cold negatives",
        weighting="pseudo retained rows weight1; real-Cold retained rows weight5; sampling creates ranking scores, not calibrated purchase probabilities",
        arms={"PB41":"pseudo only", "PC42":"pseudo plus real Cold and domain flag",
              "PC48":"PC42 plus six B0 fields, missing for pseudo rows"}, params=PARAMS,
        candidate_evaluation="raw classifier and RRF60 with B0 at alpha0.05/0.10/0.25/0.50; report Recall@1/@5, Precision@1 and conditional MRR",
        admission_evaluation="for each classifier, compare B0-best, classifier-best, and RRF(alpha0.25)-best candidate; classifier probability is the cross-user admission score; quotas0.1%,0.5%,1%,2%; action model chooses any or tail8-12 position",
        admission_gate="mean exact delta_MAP>0, >=2/3 dates nonnegative, aggregate inserted>=2 and inserted>=removed",
        next="four development replay only if admission gate passes; otherwise stop binary/content route",
        final_week="not_run", four_window_map="not_run_in_E5A", fullscale="not_run_in_E5A",
        no_commit_push=True, budget=dict(max_seconds=7200, threads=4, min_disk_gib=15),
        resources=dict(free_disk_gib=shutil.disk_usage(repo).free / 2**30),
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    dump(repo / REPORT / "BINARY_ADMISSION_E5A_CONTRACT.json", contract)
    return contract


def run(repo):
    repo = Path(repo)
    if (repo / ROOT).exists():
        raise FileExistsError("Do not overwrite E5A evidence")
    (repo / ROOT).mkdir(parents=True)
    contract = register(repo)
    start = time.perf_counter(); deadline = time.time() + 7200
    candidate = {}; admission = {}; saved = {}
    for cutoff in VALID:
        guard(repo, deadline)
        data = joblib.load(repo / PREPARED / cutoff / "data.joblib")
        candidate[cutoff] = {"B0": identify(data, -data["cold"].b0_rank.to_numpy(float))}
        admission[cutoff] = {}
        actions = action_scores(repo, cutoff, data)
        saved[cutoff] = {}
        for arm in ARMS:
            xs = []; ys = []; ws = []
            for train_cutoff in earlier(cutoff):
                x, y, w = pseudo_training(repo, train_cutoff, arm); xs.append(x); ys.append(y); ws.append(w)
                if arm != "PB41":
                    x, y, w = cold_training(repo, train_cutoff, arm); xs.append(x); ys.append(y); ws.append(w)
            x = np.concatenate(xs); y = np.concatenate(ys); weight = np.concatenate(ws)
            names = contract["features"][arm]
            assert x.shape[1] == len(names) and len(x) == len(y) == len(weight)
            model = lgb.LGBMClassifier(**PARAMS)
            with threadpool_limits(limits=4):
                model.fit(x, y, sample_weight=weight, feature_name=names)
            model.booster_.save_model(str(repo / ROOT / f"{cutoff}-{arm}.txt"))
            fit_stats = dict(rows=len(y), positives=int(y.sum()), weighted_rows=float(weight.sum()),
                             train_dates=earlier(cutoff))
            del x, y, weight, xs, ys, ws; gc.collect()
            _, vx, _ = validation_matrix(repo, cutoff, arm)
            score = model.predict_proba(vx, num_threads=4)[:, 1]
            np.save(repo / ROOT / f"{cutoff}-{arm}-scores.npy", score)
            candidate[cutoff][arm] = identify(data, score)
            candidate[cutoff][arm]["fit"] = fit_stats
            for alpha in ALPHAS:
                candidate[cutoff][f"{arm}-rrf-{alpha:g}"] = identify(data, rrf(data, score, alpha))
            modes = {"b0_best": -data["cold"].b0_rank.to_numpy(float),
                     "classifier_best": score,
                     "rrf025_best": rrf(data, score, 0.25)}
            for mode, order_score in modes.items():
                chosen = one_per_user(data, order_score)
                signal = stable_percentile(score[chosen])
                for quota in QUOTAS:
                    for slot_policy in ("model_any", "model_tail"):
                        key = f"{arm}|{mode}|q={quota:g}|{slot_policy}"
                        result, _, _ = evaluate_rule(data, chosen, actions, signal, quota, slot_policy)
                        admission[cutoff][key] = result
            saved[cutoff][arm] = fit_stats
            print("E5A", cutoff, arm, candidate[cutoff][arm], flush=True)
            del model, vx, score; gc.collect()
        del data, actions; gc.collect()
    candidate_aggregate = {}
    for key in candidate[VALID[0]]:
        candidate_aggregate[key] = dict(
            mean_recall1=float(np.mean([candidate[t][key]["recall"]["1"] for t in VALID])),
            mean_recall5=float(np.mean([candidate[t][key]["recall"]["5"] for t in VALID])),
            mean_precision1=float(np.mean([candidate[t][key]["precision"]["1"] for t in VALID])),
            mean_mrr=float(np.mean([candidate[t][key]["conditional_mrr"] for t in VALID])))
    admission_aggregate = {}
    for key in admission[VALID[0]]:
        rows = [admission[t][key] for t in VALID]
        admission_aggregate[key] = dict(mean_delta=float(np.mean([x["delta_map"] for x in rows])),
            nonnegative=int(sum(x["delta_map"] >= 0 for x in rows)), admissions=sum(x["admissions"] for x in rows),
            beneficial=sum(x["beneficial"] for x in rows), harmful=sum(x["harmful"] for x in rows),
            inserted=sum(x["inserted"] for x in rows), removed=sum(x["removed"] for x in rows))
    eligible = [(k, v) for k, v in admission_aggregate.items()
                if v["mean_delta"] > 0 and v["nonnegative"] >= 2 and v["inserted"] >= 2 and v["inserted"] >= v["removed"]]
    eligible.sort(key=lambda x: (-x[1]["mean_delta"], x[1]["admissions"], x[0]))
    selected = eligible[0][0] if eligible else None
    output = dict(status="completed", stage=contract["stage"], candidate_windows=candidate,
                  candidate_aggregate=candidate_aggregate, admission_windows=admission,
                  admission_aggregate=admission_aggregate, selected_admission=selected,
                  admission_gate_pass=selected is not None,
                  next="E5B_frozen_four_window_replay" if selected else "stop_binary_content_route",
                  final_week="not_run", four_window_map="not_run", fullscale="not_run",
                  seconds=time.perf_counter() - start)
    dump(repo / REPORT / "BINARY_ADMISSION_E5A.json", output)
    best_candidates = sorted(candidate_aggregate.items(), key=lambda x:(-x[1]["mean_recall1"], -x[1]["mean_recall5"]))[:5]
    print(json.dumps(dict(best_candidates=best_candidates, selected_admission=selected,
                          selected_metrics=admission_aggregate.get(selected), seconds=output["seconds"]), indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"BINARY_ADMISSION_E5A_FAILURE_{time.time_ns()}.json",
             dict(error=traceback.format_exc(), final_week="not_run"))
        raise
