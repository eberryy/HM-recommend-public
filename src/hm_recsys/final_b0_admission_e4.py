"""FINAL-E4 v3: broaden a failed ultra-conservative B0 admission audit.

This is a bounded historical selection followed by a frozen four-window replay.
It never reads or evaluates the sealed 2020-09-16 week.
"""
from pathlib import Path
import json
import math
import shutil
import subprocess
import time
import traceback

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .final_candidate_e2 import WINDOWS, VALID
from .final_oracle_audit import dump
from .p42f_core import batches
from .p43a_policy import contexts, exact_ap
from .p43a_run import guard


HISTORY_ROOT = Path("artifacts/final/integration-10pct-v1/prepared")
ACTION_ROOT = Path("artifacts/final/integration-10pct-v1/models")
RELATION_ROOT = Path("artifacts/final/candidate-e2-v1")
HISTORICAL_MODEL_ROOT = Path("artifacts/final/relation-pilot-v1")
REPORT = Path("reports/final")
RUN = Path("artifacts/final/b0-admission-e4-v3")

QUOTAS = (0.01, 0.02, 0.05, 0.10)
SLOT_POLICIES = ("model_any", "model_tail")
SIGNALS = (
    "action_max", "b0_score", "margin2", "margin5",
    "relation_recent_max", "relation_all_max", "relation_same_type",
    "relation_same_garment", "b0_strength", "action_b0",
    "action_relation", "all_evidence",
)


def stable_percentile(values):
    """Ascending stable fractional rank in (0,1], one value per user."""
    values = np.asarray(values, float)
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values), float)
    out[order] = np.arange(1, len(values) + 1) / max(len(values), 1)
    return out


def b0_indices(data):
    """One best available B0 candidate after frozen Warm-overlap removal."""
    cold = data["cold"]
    chosen = cold.sort_values(["user_index", "b0_rank", "article_id"], kind="stable").groupby(
        "user_index", sort=False
    ).head(1)
    expected = cold.groupby("user_index", sort=False).b0_rank.min().to_numpy()
    if chosen.user_index.duplicated().any() or not np.array_equal(chosen.b0_rank.to_numpy(), expected):
        raise ValueError("B0 top1 identity failure")
    return chosen.index.to_numpy(int)


def action_scores(repo, cutoff, data, historical):
    """Frozen action-score matrix; historical models are strict earlier-date fits."""
    if historical:
        model = lgb.Booster(model_file=str(repo / HISTORICAL_MODEL_ROOT / cutoff / "base70.txt"))
        scores = np.empty((len(data["cold"]), 12), float)
        for lo, cc, x, _ in batches(data, labels=False):
            scores[lo : lo + len(cc)] = model.predict(x, num_threads=4).reshape(-1, 12)
        return scores
    window = next(name for name, date in WINDOWS.items() if date == cutoff)
    return np.load(repo / ACTION_ROOT / window / "scores.npy")


def make_signals(data, chosen, scores, relation):
    c = data["cold"].iloc[chosen]
    r = relation[chosen].reshape(-1, 4, 6)
    raw = {
        "action_max": scores[chosen].max(axis=1),
        "b0_score": c.b0_score.to_numpy(float),
        "margin2": c.normalized_margin_to_rank2.to_numpy(float),
        "margin5": c.normalized_margin_to_rank5.to_numpy(float),
        "relation_recent_max": r[:, 0, 2],
        "relation_all_max": r[:, :, 2].max(axis=1),
        "relation_same_type": r[:, :, 4].sum(axis=1),
        "relation_same_garment": r[:, :, 5].sum(axis=1),
    }
    pct = {name: stable_percentile(np.nan_to_num(value, nan=-np.inf)) for name, value in raw.items()}
    pct["b0_strength"] = np.mean([pct["b0_score"], pct["margin2"], pct["margin5"]], axis=0)
    pct["action_b0"] = np.mean([pct["action_max"], pct["b0_strength"]], axis=0)
    pct["action_relation"] = np.mean([
        pct["action_max"], pct["relation_recent_max"], pct["relation_all_max"],
        pct["relation_same_type"], pct["relation_same_garment"],
    ], axis=0)
    pct["all_evidence"] = np.mean([
        pct["action_max"], pct["b0_strength"], pct["relation_recent_max"],
        pct["relation_all_max"], pct["relation_same_type"], pct["relation_same_garment"],
    ], axis=0)
    return pct


def best_slot_delta(data, candidate_index, allowed_slots):
    """Truth-only oracle for diagnostic headroom; not a deployable decision."""
    ui = int(data["cold"].iloc[candidate_index].user_index)
    item = data["cold"].iloc[candidate_index].article_id
    best = (0.0, None)
    for slot in allowed_slots:
        items = data["warm_lists"][ui].copy()
        items[slot] = item
        delta = exact_ap(items, data["truthsets"][data["users"][ui]]) - data["baseline_ap"][ui]
        if delta > best[0] + 1e-15:
            best = (float(delta), int(slot))
    return best


def evaluate_rule(data, chosen, scores, signal, quota, slot_policy):
    nusers = len(data["users"])
    count = max(1, int(math.ceil(quota * len(chosen))))
    customer = data["cold"].iloc[chosen].customer_id.to_numpy(str)
    order = np.lexsort((customer, -signal))[:count]
    lists = data["warm_lists"].copy()
    events = []
    for local in order:
        ci = int(chosen[local])
        row = data["cold"].iloc[ci]
        ui = int(row.user_index)
        allowed = np.arange(12) if slot_policy == "model_any" else np.arange(7, 12)
        slot = int(allowed[np.argmax(scores[ci, allowed])])
        old = lists[ui, slot]
        lists[ui, slot] = row.article_id
        delta = exact_ap(lists[ui], data["truthsets"][data["users"][ui]]) - data["baseline_ap"][ui]
        events.append(dict(user_index=ui, customer_id=data["users"][ui], candidate=row.article_id,
                           slot=slot + 1, delta=float(delta), inserted=int(row.target),
                           removed=int(old in data["truthsets"][data["users"][ui]]), signal=float(signal[local])))
    ap = np.array([exact_ap(items, data["truthsets"][u]) for u, items in zip(data["users"], lists)])
    e = pd.DataFrame(events)
    delta = float(ap.mean() - data["baseline_map"])
    assert abs(delta - e.delta.sum() / nusers) < 1e-12
    return dict(delta_map=delta, admissions=len(e), beneficial=int(e.delta.gt(0).sum()),
                harmful=int(e.delta.lt(0).sum()), neutral=int(e.delta.eq(0).sum()),
                inserted=int(e.inserted.sum()), removed=int(e.removed.sum())), lists, e


def window_audit(repo, cutoff, historical):
    data = joblib.load(repo / HISTORY_ROOT / cutoff / "data.joblib")
    if cutoff >= "2020-09-16":
        raise ValueError("sealed final week prohibited")
    scores = action_scores(repo, cutoff, data, historical)
    if scores.shape != (len(data["cold"]), 12) or not np.isfinite(scores).all():
        raise ValueError("invalid complete action score matrix")
    relation = np.load(repo / RELATION_ROOT / cutoff / "m4-relation.npy", mmap_mode="r")
    chosen = b0_indices(data)
    signals = make_signals(data, chosen, scores, relation)
    positives = int(data["cold"].iloc[chosen].target.sum())
    any_oracle = [best_slot_delta(data, ci, range(12))[0] for ci in chosen]
    tail_oracle = [best_slot_delta(data, ci, range(7, 12))[0] for ci in chosen]
    summary = dict(cutoff=cutoff, users=len(data["users"]), candidate_users=len(chosen),
                   b0_top1_positive=positives,
                   b0_top1_oracle_any_delta=float(sum(any_oracle) / len(data["users"])),
                   b0_top1_oracle_tail_delta=float(sum(tail_oracle) / len(data["users"])),
                   positive_signal_percentiles={name: dict(
                       values=[float(x) for x in value[data["cold"].iloc[chosen].target.to_numpy(bool)]],
                       maximum=float(value[data["cold"].iloc[chosen].target.to_numpy(bool)].max()),
                       median=float(np.median(value[data["cold"].iloc[chosen].target.to_numpy(bool)])))
                       for name, value in signals.items()})
    return data, chosen, scores, signals, summary


def segment_metrics(data, lists):
    truths, valid, base = contexts(data)
    values = np.array([[exact_ap(items, truth) for truth in row] for items, row in zip(lists, truths)])
    denom = valid.sum(axis=0)
    delta = np.divide((values - base).sum(axis=0), denom, out=np.zeros(5), where=denom > 0)
    return {name: float(delta[i]) for i, name in enumerate(
        ["overall", "warm_21_plus", "strict_cold", "sparse1_5", "all_cold_sparse"]
    )}


def register(repo):
    contract = dict(
        status="preregistered_before_e4_computation",
        stage="FINAL-E4 v3 B0-best-available wider admission",
        hypothesis="B0 already ranks real Cold positives better than learned challengers; freeze its top item and only learn whether/where to admit it.",
        historical_validation=VALID,
        development_windows=WINDOWS,
        candidate="exactly one lowest-original-B0-rank item among candidates remaining after frozen Warm overlap removal; original rank can exceed1 and is not renumbered; no candidate reranking",
        signals={name: "within-window stable percentile; larger means stronger evidence" for name in SIGNALS},
        composites={"b0_strength":"mean percentile of B0 score and normalized margins to rank2/rank5",
                    "action_b0":"mean action-max and b0-strength percentile",
                    "action_relation":"mean action and four direct M4 relation percentiles",
                    "all_evidence":"mean action, B0 strength and four relation percentiles"},
        relation="time-safe M4 Student cosine/same-attribute evidence against latest20 distinct user-history items",
        reason_for_wider_quotas="E4-v2 tested0.05%-0.5%; every selected action was neutral and none inserted a positive, while B0-best oracle remained positive. v3 expands to1%-10% without changing candidate or evidence.",
        quotas=QUOTAS, slot_policies={"model_any":"action model chooses positions1-12",
                                     "model_tail":"action model chooses positions8-12"},
        historical_selection="highest mean exact delta_MAP; ties prefer fewer admissions, tail position, simpler named signal",
        historical_gate="mean delta>0; at least2/3 dates nonnegative; aggregate inserted>=2 and inserted>=removed",
        development_gate="frozen existing expansion gate: mean>0, >=3/4 delta>=-1e-5, worst>=-5e-5, warm mean>=-2e-5, cold/sparse mean>=0, inserted>=1, inserted>=removed",
        oracle="B0-top1 truth-best position, any and tail, is diagnostic only and cannot select a rule",
        fallback="WV3-741 unchanged; if historical gate fails, do not evaluate four development windows; if development gate fails, no full-scale run",
        final_week="not_run", fullscale="only if development gate passes", no_commit_push=True,
        budget=dict(max_seconds=3600, threads=4, min_disk_gib=15),
        resources=dict(free_disk_gib=shutil.disk_usage(repo).free / 2**30),
        git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    dump(repo / REPORT / "B0_ADMISSION_E4_V3_CONTRACT.json", contract)
    return contract


def run(repo):
    repo = Path(repo)
    if (repo / RUN).exists():
        raise FileExistsError("Do not overwrite E4 evidence")
    (repo / RUN).mkdir(parents=True)
    contract = register(repo)
    started = time.perf_counter()
    guard(repo)
    historical = {}
    cache = {}
    for cutoff in VALID:
        data, chosen, scores, signals, audit = window_audit(repo, cutoff, True)
        cache[cutoff] = (data, chosen, scores, signals)
        historical[cutoff] = {"audit": audit, "rules": {}}
    keys = [(s, q, p) for s in SIGNALS for q in QUOTAS for p in SLOT_POLICIES]
    aggregate = {}
    for signal, quota, slot_policy in keys:
        name = f"{signal}|q={quota:g}|{slot_policy}"
        rows = []
        for cutoff in VALID:
            data, chosen, scores, signals = cache[cutoff]
            result, _, _ = evaluate_rule(data, chosen, scores, signals[signal], quota, slot_policy)
            historical[cutoff]["rules"][name] = result
            rows.append(result)
        aggregate[name] = dict(mean_delta=float(np.mean([x["delta_map"] for x in rows])),
                               nonnegative=int(sum(x["delta_map"] >= 0 for x in rows)),
                               admissions=sum(x["admissions"] for x in rows),
                               beneficial=sum(x["beneficial"] for x in rows),
                               harmful=sum(x["harmful"] for x in rows),
                               inserted=sum(x["inserted"] for x in rows),
                               removed=sum(x["removed"] for x in rows),
                               signal=signal, quota=quota, slot_policy=slot_policy)
    def eligible(x):
        return x[1]["mean_delta"] > 0 and x[1]["nonnegative"] >= 2 and x[1]["inserted"] >= 2 and x[1]["inserted"] >= x[1]["removed"]
    valid = [x for x in aggregate.items() if eligible(x)]
    valid.sort(key=lambda x: (-x[1]["mean_delta"], x[1]["admissions"],
                              x[1]["slot_policy"] != "model_tail", SIGNALS.index(x[1]["signal"])))
    selected = valid[0][0] if valid else None
    historical_gate = selected is not None
    for cutoff in list(cache):
        del cache[cutoff]
    development = {}
    development_gate = False
    if historical_gate:
        spec = aggregate[selected]
        for window, cutoff in WINDOWS.items():
            data, chosen, scores, signals, audit = window_audit(repo, cutoff, False)
            result, lists, events = evaluate_rule(data, chosen, scores, signals[spec["signal"]], spec["quota"], spec["slot_policy"])
            result["segments"] = segment_metrics(data, lists)
            result["audit"] = audit
            development[window] = result
            events.to_parquet(repo / RUN / f"{window}-events.parquet", index=False)
        delta = np.array([x["delta_map"] for x in development.values()])
        warm = np.mean([x["segments"]["warm_21_plus"] for x in development.values()])
        cold = np.mean([x["segments"]["all_cold_sparse"] for x in development.values()])
        inserted = sum(x["inserted"] for x in development.values())
        removed = sum(x["removed"] for x in development.values())
        gates = dict(mean_positive=bool(delta.mean() > 0), three_nondegrade=bool((delta >= -1e-5).sum() >= 3),
                     worst=bool(delta.min() >= -5e-5), warm=bool(warm >= -2e-5), cold_sparse=bool(cold >= 0),
                     inserted_positive=bool(inserted >= 1), efficiency=bool(inserted >= removed))
        development_gate = all(gates.values())
    else:
        gates = {"not_run": "historical gate failed"}
    output = dict(status="completed", stage=contract["stage"], historical=historical, aggregate=aggregate,
                  selected=selected, historical_gate_pass=historical_gate, development=development,
                  development_gates=gates, development_gate_pass=development_gate,
                  fullscale_allowed=development_gate, fullscale_status="not_run",
                  final_week="not_run", fallback="WV3-741", seconds=time.perf_counter() - started)
    dump(repo / REPORT / "B0_ADMISSION_E4_V3.json", output)
    print(json.dumps({"selected": selected,
                      "selected_historical": aggregate.get(selected),
                      "historical_gate_pass": historical_gate,
                      "development": development,
                      "development_gates": gates,
                      "fullscale_allowed": development_gate,
                      "seconds": output["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    try:
        run(Path.cwd())
    except Exception:
        dump(REPORT / f"B0_ADMISSION_E4_V3_FAILURE_{time.time_ns()}.json",
             dict(error=traceback.format_exc(), final_week="not_run"))
        raise
