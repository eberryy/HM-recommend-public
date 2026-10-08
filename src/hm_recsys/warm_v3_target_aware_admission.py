"""WV3-620: target-aware two-head residual admission on a protected Warm slate."""
from __future__ import annotations

import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import evidence_id, now, read, write
from .warm_v2_engine import load_parquet, save_parquet
from .warm_v3_expert import screening_gate
from .warm_v3_residual_admission import FEATURES as BASE_FEATURES
from .warm_v3_residual_admission import HISTORICAL, INNER, OUTER, pair_frame, source
from .warm_v3_two_head_admission import choose_expected_decisions


TRIAL = "WV3-620"
CONTRACT = common.REPORT / "WV3-620_TARGET_AWARE_ADMISSION_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-620_SCREEN.json"
OUTER_REPORT = common.REPORT / "WV3-620_OUTER.json"
INPUT_AUDIT = common.REPORT / "WV3-620_TARGET_AWARE_INPUT_AUDIT.json"
MODEL_ROOT = common.ART / TRIAL

# These seven columns are the non-rank, candidate-specific tail of the existing
# 21-column candidate-gate cache. The other 14 are user state or rank transforms
# already represented in BASE_FEATURES; excluding them is fixed before outcomes.
TARGET_FEATURES = [
    "repurchase_present",
    "item2vec_is_new",
    "user_item_events_12w",
    "user_item_days_since_last_purchase",
    "user_product_type_share_12w",
    "user_department_share_12w",
    "item_days_since_last_sale",
]
TARGET_PAIR_FEATURES = (
    [f"challenger_{name}" for name in TARGET_FEATURES]
    + [f"victim_{name}" for name in TARGET_FEATURES]
    + [f"delta_{name}" for name in TARGET_FEATURES]
)
FEATURES = BASE_FEATURES + TARGET_PAIR_FEATURES

PREFERENCE_PARAMS = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 100,
    "lambda_l2": 10.0,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "seed": 20260910,
    "feature_fraction_seed": 20260910,
    "bagging_seed": 20260910,
    "verbosity": -1,
}
ACTION_PARAMS = {**PREFERENCE_PARAMS, "min_data_in_leaf": 500}
PREFERENCE_ROUNDS = 80
ACTION_ROUNDS = 60


def target_table(cutoff: str) -> tuple[pd.DataFrame, dict]:
    root = common.ART / "candidate_gate_data" / cutoff
    meta = read(root / "DATA.json")
    assert meta["cutoff"] == cutoff
    assert not meta["future_labels_are_inputs"] and meta["final_week"] == "not_run"
    assert all(name in meta["features"] for name in TARGET_FEATURES)
    keys = load_parquet(meta["keys"])
    arrays = np.load(root / "arrays.npz")
    assert len(keys) == meta["rows"] == len(arrays["x"])
    assert np.array_equal(keys.target.to_numpy(np.uint8), arrays["target"])
    assert np.array_equal((keys.user_history_events_12w > 0).to_numpy(np.uint8), arrays["active"])
    mask = (keys.rf <= 50).to_numpy()
    table = keys.loc[mask, ["customer_id", "article_id"]].reset_index(drop=True)
    for name in TARGET_FEATURES:
        index = meta["features"].index(name)
        table[name] = arrays["x"][mask, index].astype(np.float32, copy=False)
    assert not table.duplicated(["customer_id", "article_id"]).any()
    assert np.isfinite(table[TARGET_FEATURES].to_numpy(np.float64)).all()
    details = {
        "cutoff": cutoff,
        "source": str(root / "DATA.json"),
        "rows": len(table),
        "features": TARGET_FEATURES,
        "unique_values": {name: int(table[name].nunique(dropna=False)) for name in TARGET_FEATURES},
        "source_identity": meta["source_identity"],
        "future_labels_are_inputs": False,
        "final_week": "not_run",
    }
    return table, details


def augment_pairs(pairs: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    challenger = targets.rename(
        columns={"article_id": "challenger_article_id", **{name: f"challenger_{name}" for name in TARGET_FEATURES}}
    )
    victim = targets.rename(
        columns={"article_id": "victim_article_id", **{name: f"victim_{name}" for name in TARGET_FEATURES}}
    )
    original = len(pairs)
    result = pairs.merge(
        challenger,
        on=["customer_id", "challenger_article_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        victim,
        on=["customer_id", "victim_article_id"],
        how="left",
        validate="many_to_one",
    )
    assert len(result) == original
    for name in TARGET_FEATURES:
        result[f"delta_{name}"] = result[f"challenger_{name}"] - result[f"victim_{name}"]
    assert np.isfinite(result[FEATURES].to_numpy(np.float64)).all()
    return result


def historical_training_data():
    action_x, action_y, action_weight = [], [], []
    preference_x, preference_y, preference_weight = [], [], []
    sources = []
    for name, folder in HISTORICAL:
        path, meta = source(folder)
        candidate, details = target_table(meta["cutoff"])
        pairs = augment_pairs(pair_frame(path, discordant_only=False), candidate)
        x = pairs[FEATURES].to_numpy(np.float32)
        actionable = (pairs.challenger_target != pairs.victim_target).to_numpy(np.uint8)
        weight = pairs.unit_gain.to_numpy(np.float32)
        action_x.append(x)
        action_y.append(actionable)
        action_weight.append(weight)
        preference_x.append(x[actionable == 1])
        preference_y.append(pairs.loc[actionable == 1, "challenger_target"].to_numpy(np.uint8))
        preference_weight.append(weight[actionable == 1])
        sources.append(
            {
                **details,
                "window": name,
                "pair_rows": len(pairs),
                "actionable_rows": int(actionable.sum()),
                "target_pair_feature_nonzero_share": {
                    feature: float((pairs[feature] != 0).mean()) for feature in TARGET_PAIR_FEATURES
                },
            }
        )
        del candidate, pairs, x, actionable, weight
        gc.collect()
    ax = np.concatenate(action_x)
    ay = np.concatenate(action_y)
    aw = np.concatenate(action_weight)
    px = np.concatenate(preference_x)
    py = np.concatenate(preference_y)
    pw = np.concatenate(preference_weight)
    audit = {
        "role": "target_aware_historical_pair_input_audit",
        "sources": sources,
        "base_pair_features": BASE_FEATURES,
        "added_candidate_specific_features": TARGET_FEATURES,
        "added_pair_columns": TARGET_PAIR_FEATURES,
        "total_model_features": len(FEATURES),
        "actionability_rows": len(ay),
        "actionable_rows": int(ay.sum()),
        "preference_rows": len(py),
        "preference_positive_rows": int(py.sum()),
        "all_added_features_vary_in_at_least_one_source": all(
            any(source_row["unique_values"][name] > 1 for source_row in sources) for name in TARGET_FEATURES
        ),
        "all_pair_deltas_nonzero_somewhere": all(
            any(source_row["target_pair_feature_nonzero_share"][f"delta_{name}"] > 0 for source_row in sources)
            for name in TARGET_FEATURES
        ),
        "candidate_identity_complete": True,
        "latest_label_end_exclusive": "2019-11-27",
        "earliest_inner_cutoff": "2019-12-25",
        "final_week": "not_run",
    }
    audit["passed"] = (
        audit["all_added_features_vary_in_at_least_one_source"]
        and audit["all_pair_deltas_nonzero_somewhere"]
        and audit["candidate_identity_complete"]
        and audit["actionable_rows"] > 10000
    )
    return ax, ay, aw, px, py, pw, audit


def fit_binary(x, y, weight, params, rounds):
    assert 0 < y.sum() < len(y)
    dataset = lgb.Dataset(x, label=y, weight=weight, feature_name=FEATURES, free_raw_data=True)
    return lgb.train(params, dataset, num_boost_round=rounds)


def evaluate(folder, name, preference, action, stage):
    path, meta = source(folder)
    started = time.perf_counter()
    candidate, details = target_table(meta["cutoff"])
    pairs = augment_pairs(pair_frame(path, discordant_only=False), candidate)
    x = pairs[FEATURES].to_numpy(np.float32)
    pref = preference.predict(x, num_threads=4)
    act = action.predict(x, num_threads=4)
    chosen = choose_expected_decisions(pairs, pref, act)
    beneficial = chosen.actual_delta > 1e-15
    harmful = chosen.actual_delta < -1e-15
    neutral = ~(beneficial | harmful)
    delta = float(chosen.actual_delta.sum() / meta["total_users"])
    baseline = float(meta["baseline_map_population_component"])
    result = {
        "window": name,
        "cutoff": meta["cutoff"],
        "role": stage,
        "total_users_denominator": meta["total_users"],
        "pair_rows": len(pairs),
        "selected_users": len(chosen),
        "beneficial_selected_users": int(beneficial.sum()),
        "harmful_selected_users": int(harmful.sum()),
        "neutral_selected_users": int(neutral.sum()),
        "selected_neutral_share": float(neutral.mean()) if len(chosen) else 0.0,
        "gross_positive_MAP": float(chosen.loc[beneficial, "actual_delta"].sum() / meta["total_users"]),
        "gross_negative_MAP": float(chosen.loc[harmful, "actual_delta"].sum() / meta["total_users"]),
        "baseline_MAP@12": baseline,
        "MAP@12": baseline + delta,
        "population_delta": delta,
        "protected_ranks1_7_changes": 0,
        "maximum_swaps_per_user": 1,
        "target_input": details,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    return result, chosen


def experiment_contract():
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-610",
        "architecture_family": "target_aware_two_head_residual_admission",
        "hypothesis": "Candidate-specific repeat, Item2Vec-source, user-item recency/affinity, category-share and item-freshness features can improve pair actionability and direction beyond rank and score geometry.",
        "added_features": TARGET_FEATURES,
        "feature_policy": "add challenger absolute, victim absolute and challenger-minus-victim value for all seven columns; no feature selection after outcomes",
        "excluded_cached_columns": "14 user/rank transforms already represented in the 23 base pair features are not duplicated",
        "models": {
            "preference": {"params": PREFERENCE_PARAMS, "rounds": PREFERENCE_ROUNDS},
            "actionability": {"params": ACTION_PARAMS, "rounds": ACTION_ROUNDS},
        },
        "training": "both heads use only four 2019 historical sources; preference uses discordant pairs, actionability uses all pairs",
        "decision_score": "P(actionable)*(2*P(challenger wins|actionable)-1)*exact position unit_gain",
        "action": "one maximum positive expected-gain swap; ranks1-7 immutable",
        "candidate_pool": "unchanged WV2-601 pool; ranks8-50 only",
        "input_gate": "all seven features vary, all seven pair deltas are sometimes nonzero, identity complete, actionable rows>10000",
        "inner_gate": {
            "standard_vs_WV2_601": "mean>=+0.0001, >=3 positive, worst>=-0.0005",
            "incremental_vs_WV3_610": "mean>0, >=3 nondegrade, worst>=-0.0002",
        },
        "expected_minutes": 20,
        "outer_policy": "one frozen exposure only if input and both inner gates pass; no feature, tree or score rescue",
        "fallback": "WV3-601/WV3-610 stable candidates",
        "final_week": "2020-09-16 not_run",
    }


def register():
    common.setup()
    common.budget(20)
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "target_aware_two_head_residual_admission",
            experiment_contract()["hypothesis"],
            candidate_protocol=experiment_contract()["candidate_pool"],
            training_protocol=experiment_contract()["training"],
            features=FEATURES,
            params={
                "preference": experiment_contract()["models"]["preference"],
                "actionability": experiment_contract()["models"]["actionability"],
                "decision_score": experiment_contract()["decision_score"],
                "inner_gate": experiment_contract()["inner_gate"],
                "final_week": "not_run",
            },
            expected_minutes=20,
        )


def incremental_gate(windows):
    prior = read(common.REPORT / "WV3-610_SCREEN.json")["windows"]
    delta = {name: row["MAP@12"] - prior[name]["MAP@12"] for name, row in windows.items()}
    values = list(delta.values())
    return {
        "per_window_delta_vs_WV3_610": delta,
        "mean_delta_vs_WV3_610": float(np.mean(values)),
        "nondegrade_windows_vs_WV3_610": sum(value >= 0 for value in values),
        "worst_delta_vs_WV3_610": min(values),
        "passed": bool(np.mean(values) > 0 and sum(value >= 0 for value in values) >= 3 and min(values) >= -0.0002),
    }


def screen():
    common.setup()
    common.budget(20)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    started = time.perf_counter()
    ax, ay, aw, px, py, pw, audit = historical_training_data()
    write(INPUT_AUDIT, audit)
    if not audit["passed"]:
        common.update(TRIAL, decision="reject_input_audit", inner_evidence=audit, runtime=time.perf_counter() - started, artifact_paths=[str(INPUT_AUDIT)])
        raise RuntimeError("WV3-620 target-aware input gate failed")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    preference = fit_binary(px, py, pw, PREFERENCE_PARAMS, PREFERENCE_ROUNDS)
    del px, py, pw
    gc.collect()
    action = fit_binary(ax, ay, aw, ACTION_PARAMS, ACTION_ROUNDS)
    del ax, ay, aw
    gc.collect()
    preference_path = MODEL_ROOT / "PREFERENCE_MODEL.txt"
    action_path = MODEL_ROOT / "ACTIONABILITY_MODEL.txt"
    preference.save_model(str(preference_path))
    action.save_model(str(action_path))
    write(
        MODEL_ROOT / "MODEL.json",
        {
            "experiment_id": TRIAL,
            "preference_model": evidence_id(preference_path, reason="explicit_registry_evidence"),
            "actionability_model": evidence_id(action_path, reason="explicit_registry_evidence"),
            "features": FEATURES,
            "target_features": TARGET_FEATURES,
            "preference_params": PREFERENCE_PARAMS,
            "actionability_params": ACTION_PARAMS,
            "preference_rounds": PREFERENCE_ROUNDS,
            "actionability_rounds": ACTION_ROUNDS,
            "input_audit": str(INPUT_AUDIT),
            "outer_labels_used": False,
            "final_week": "not_run",
        },
    )
    windows = {}
    for name, folder in INNER:
        row, chosen = evaluate(folder, name, preference, action, "historical_target_aware_inner_screen")
        windows[name] = row
        root = MODEL_ROOT / name / "inner"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"target_aware_inner": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = screening_gate(row["population_delta"] for row in windows.values())
    incremental = incremental_gate(windows)
    passed = standard["passed"] and incremental["passed"]
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "architecture": "target_aware_two_head_residual_admission",
        "input_audit": audit,
        "windows": windows,
        "standard_screening_vs_WV2_601": standard,
        "incremental_screening_vs_WV3_610": incremental,
        "passed": passed,
        "features": FEATURES,
        "target_features": TARGET_FEATURES,
        "candidate_pool_changed": False,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(SCREEN_REPORT, result)
    common.update(
        TRIAL,
        decision="inner_pass" if passed else "reject_inner",
        inner_evidence={"input": audit, "standard": standard, "incremental": incremental, "passed": passed},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(INPUT_AUDIT), str(SCREEN_REPORT), str(MODEL_ROOT / "MODEL.json")],
    )
    common.log(
        f"{TRIAL} inner screen",
        "The rank-only two-head model was stable but barely improved the one-head policy; candidate-specific cross features were absent from both heads.",
        experiment_contract()["hypothesis"],
        f"{INPUT_AUDIT}; {SCREEN_REPORT}",
        f"input pass={audit['passed']}; standard mean={standard['mean_population_delta']:+.9f}; incremental mean={incremental['mean_delta_vs_WV3_610']:+.9f}; overall pass={passed}.",
        "Freeze and expose once." if passed else "Reject target-aware two-head admission at inner; no feature or parameter rescue.",
        "Run one outer confirmation if both gates pass; otherwise decompose whether the remaining gap is candidate identification or user-level purchase incidence.",
        alternatives="WV3-501 used the same seven signals inside a broad Top50 neural reorder and moved about seven items per user; the present one-swap pair factorization is a distinct mechanism.",
        experiment="Train fixed historical preference and actionability heads with 23 base pair features plus challenger/victim/difference versions of seven candidate-specific cached features.",
        reflection="The feature policy is fixed by semantic nonredundancy before outcomes; no column ablation or selection follows the screen.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "passed": passed, "seconds": result["runtime_seconds"]}, flush=True)
    return result


def confirm():
    common.setup()
    common.budget(10)
    registry = read(common.REGISTRY)
    entry = next(row for row in registry["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "inner_pass" and entry["outer_exposures"] == 1
    assert read(SCREEN_REPORT)["passed"]
    if OUTER_REPORT.exists():
        return read(OUTER_REPORT)
    preference = lgb.Booster(model_file=str(MODEL_ROOT / "PREFERENCE_MODEL.txt"))
    action = lgb.Booster(model_file=str(MODEL_ROOT / "ACTIONABILITY_MODEL.txt"))
    started = time.perf_counter()
    windows = {}
    for name, folder in OUTER:
        row, chosen = evaluate(folder, name, preference, action, "frozen_target_aware_outer")
        windows[name] = row
        root = MODEL_ROOT / name / "outer"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"target_aware_outer": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    prior = read(common.REPORT / "WV3-610_OUTER.json")
    delta = {name: windows[name]["MAP@12"] - prior["per_window_MAP"][name] for name in windows}
    values = list(delta.values())
    incremental = {
        "per_window_delta_vs_WV3_610": delta,
        "mean_delta_vs_WV3_610": float(np.mean(values)),
        "nondegrade_windows_vs_WV3_610": sum(value >= 0 for value in values),
        "worst_delta_vs_WV3_610": min(values),
    }
    better = standard["stable"] and incremental["mean_delta_vs_WV3_610"] > 0
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "architecture": "target_aware_two_head_residual_admission",
        "windows": windows,
        **standard,
        "incremental_vs_WV3_610": incremental,
        "better_than_highest_mean_candidate": better,
        "candidate_pool_changed": False,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(OUTER_REPORT, result)
    common.update(
        TRIAL,
        decision="promote_candidate" if better else "reject_outer",
        outer_MAP_by_window=standard["per_window_MAP"],
        mean_MAP=standard["mean_MAP"],
        delta_vs_WV2_601=standard["delta_vs_WV2_601"],
        nondegrade_windows=standard["nondegrade_windows"],
        worst_delta=standard["worst_delta"],
        runtime=result["runtime_seconds"],
        artifact_paths=entry["artifact_paths"] + [str(OUTER_REPORT)],
    )
    common.log(
        f"{TRIAL} outer closure",
        "The target-aware architecture passed its input audit and both inner gates before being frozen.",
        experiment_contract()["hypothesis"],
        str(OUTER_REPORT),
        f"mean MAP={standard['mean_MAP']:.9f}; delta vs WV2-601={standard['delta_vs_WV2_601']:+.9f}; incremental mean vs WV3-610={incremental['mean_delta_vs_WV3_610']:+.9f}; stable={standard['stable']}; better={better}.",
        "Promote as highest-mean stable candidate." if better else "Reject this target-aware variant and keep the better stable candidate.",
        "Continue only through a new bottleneck audit if target0.03 remains unmet and time permits.",
        alternatives="No outer-driven feature removal, threshold change or refit is allowed.",
        experiment="One outer exposure of the exact two frozen target-aware historical heads.",
        reflection="The test measures whether semantically richer pair inputs transfer, not whether their historical fit improves.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "better": better}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "screen", "confirm"])
    globals()[parser.parse_args().command]()
