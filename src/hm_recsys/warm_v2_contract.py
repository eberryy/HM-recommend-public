"""Warm-only branch contract and append-preserving experiment registry."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

BRANCH = "warm-v2-autonomous-lab"
BASE_COMMIT = "cb79790e1a2d3dc17989b04192b60c7f15382b5f"
FINAL = "2020-09-16"
REPORT = Path("reports/warm_v2")
ARTIFACT = Path("artifacts/warm_v2")
M33 = Path("reports/m3_3/m3-3-v1-cross-season-adaptive-seasonal/metrics.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".part")
    tmp.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    tmp.replace(path)


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def guard(cutoff):
    c = date.fromisoformat(cutoff)
    if c >= date.fromisoformat(FINAL) or c+timedelta(days=7) > date.fromisoformat(FINAL):
        raise ValueError("Warm-v2 forbids final-week labels or overlap")
    return c.isoformat()


def assert_branch():
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("Warm-v2 refuses writes outside its isolated branch")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], check=True)


def evidence_id(path, *, reason):
    """Only call for a trusted prior comparison or the explicit evidence contract."""
    if reason not in {"compare_frozen_m33", "explicit_registry_evidence"}:
        raise ValueError("hashing needs a concrete integrity/evidence purpose")
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            h.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest(), "hash_purpose": reason}


def preregister():
    assert_branch()
    path = REPORT/"WARM_V2_EXPERIMENT_CONTRACT.json"
    if path.exists():
        return read(path)
    m = read(M33)
    c = {"stage": "WV2.0 baseline then WV2.1 feature exploration", "created_at_utc": now(),
        "status": "preregistered_before_new_training", "branch": BRANCH, "base_commit": BASE_COMMIT,
        "historical_metrics": str(M33.resolve()), "frozen_feature_columns": m["contract"]["variants"]["anchor"],
        "rolling_protocol": m["contract"]["rolling_protocol"], "config": m["config"],
        "candidate_contract": "exact M3.3 cached rows; six-source weighted RRF Top100 + up to200 Item2Vec-only; 10% deterministic users, full-history global statistics; optimistic all-articles",
        "truth_contract": "distinct purchased items in [cutoff,cutoff+7d); complete truth denominator min(ntruth,12); ALL original W0 users",
        "protected": ["M4", "P3.1", "P3.2", "P3.3", "P3.4", "P3.5", "P3.6", "P3.7A", "P3.7B", "P4.0", "P4.1A", "docs/ROADMAP_PHASE3.zh-CN.md", "docs/ROADMAP_PHASE4.zh-CN.md"],
        "training": {"sampling": "exact M2.11 source-layer/rank-tertile/hash 30:1", "sampling_seed": 20260831,
            "objective": "lambdarank", "weighting": "none", "inner_early_stopping_patience": 20,
            "rounds": "each variant independently selected using historical inner only, max200; then refit two training cutoffs",
            "params": "exact historical per-window inner/outer parameters, no sweep", "fallback": "history_events_12w==0 uses original candidate_rank; others score DESC, candidate_rank,article_id"},
        "baseline_parity": {"candidate_and_feature_identity": "read exact historical frozen parquet and same ordered84 model columns; verify trusted prior SHA once",
            "inner_rounds": "exact", "sample_rows_groups_positives": "exact historical recorded counts + fixed SQL sampler",
            "model_predictions": "max absolute difference <=1e-12 against saved anchor on all outer candidates",
            "top12_identity_order": "exact", "MAP": "absolute error <=1e-12 (summation order only)"},
        "screening": {"labels": "only four historical inner validations: 2019-12-25,2020-02-19,2020-05-27,2020-07-22",
            "metric": "inner covered-active AP delta times covered-active users/all W0 users; fixed inactive contribution and uncovered zeros cancel",
            "positive_windows_min": 3, "mean_population_delta_min": .0001, "worst_population_delta_min": -.0005,
            "family_outer_confirmation_limit": 2, "outer_trial_limit_first_feature_phase": 3,
            "rule": "screen distinct bundles before outer; champion chosen with gates, never tune leaves/depth/seed to outer; no family changes driven by small outer flips"},
        "initial_hypotheses": [
            {"family": "item_demand", "hypothesis": "7/28/84d counts miss very recent demand and item lifecycle; 1/3/14d sales and historical launch/last-sale support may rank warm demand better"},
            {"family": "user_rhythm", "hypothesis": "12w user summary lacks purchase-day rhythm and current activity; test 7/28d activity, intervals and price context"},
            {"family": "repeat_affinity", "hypothesis": "existing hierarchy counts lack 7d recency and frequency-recency interactions; test repeated article/product hierarchy and current preference match"},
            {"family": "source_agreement", "hypothesis": "several weak independent retrieval ranks may provide support distinct from simple source_count; test rank strength/dispersion and overlap"}],
        "champion_gate": {"mean_delta_vs_current_champion": ">0", "nondegrade_vs_v1_min": 3, "worst_delta_vs_v1_min": -.0003},
        "milestone_gate": {"mean_delta_vs_v1_min": .0005, "nondegrade_min": 3, "worst_delta_min": -.0003,
            "large_gain_manual_review": "mean_delta>=.001 even if stability gate fails; do not auto-promote unstable result",
            "on_pass": "stop broad exploration; verify, ablate, manifest, milestone report and local commit; request human review, no merge"},
        "resources": {"baseline_minutes_estimate": [5,15], "first_family_screen_minutes_estimate": [30,90],
            "cpu_threads": 8, "gpu": False, "target_peak_memory_gib": 8, "new_disk_budget_gib": 5,
            "space_stop_floor_gib": 5, "compute_review_hours": 3,
            "stop_not_abandon": "at cost limit retain checkpoints; diagnose efficiency and narrow work, not unexplained failure termination"},
        "hash_policy": "once compare old mutable D-drive assets against trusted M3.3 recorded values; new SHA only explicit registry/milestone package, no repeated completion hashing",
        "final_week": "not_run", "automatic_push_or_merge": False}
    for row in c["rolling_protocol"].values():
        for cutoff in [*row["outer_train"], row["outer_validation"]]:
            guard(cutoff)
    write(path, c)
    write(REPORT/"WARM_EXPERIMENT_REGISTRY.json", {"schema": "warm-v2-registry-v1", "branch": BRANCH,
        "frozen_baseline": "WV2-000", "current_champion": "WV2-000", "outer_confirmation_counts": {}, "trials": [], "final_week": "not_run"})
    return c


def register(spec):
    assert_branch()
    registry = read(REPORT/"WARM_EXPERIMENT_REGISTRY.json")
    prior = next((t for t in registry["trials"] if t["experiment_id"] == spec["experiment_id"]), None)
    if prior:
        for key in ('candidate_contract','feature_columns','feature_bundle','parent_experiment'):
            if prior[key]!=spec[key]:
                raise ValueError(f'Cannot repurpose registered experiment {spec["experiment_id"]}: {key}')
        if prior.get('parameter_overrides',{})!=spec.get('parameter_overrides',{}):
            raise ValueError('Cannot change parameters of a registered trial')
        return prior
    spec = {"timestamp": now(), "git_commit": git("rev-parse", "HEAD"),
        "implementation_dirty_at_registration": bool(git("status", "--porcelain", "--", "src/hm_recsys/warm_v2_*.py", "tests/test_warm_v2.py")),
        "status": "preregistered", **spec}
    registry["trials"].append(spec)
    write(REPORT/"WARM_EXPERIMENT_REGISTRY.json", registry)
    return spec


def record(trial_id, updates):
    registry = read(REPORT/"WARM_EXPERIMENT_REGISTRY.json")
    trial = next(t for t in registry["trials"] if t["experiment_id"] == trial_id)
    trial.update(updates)
    write(REPORT/"WARM_EXPERIMENT_REGISTRY.json", registry)
