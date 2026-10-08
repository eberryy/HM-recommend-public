"""WV3-610: two-head expected-gain admission with historical actionability."""
from __future__ import annotations

import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import warm_v3_common as common
from .warm_v2_contract import evidence_id, now, read, write
from .warm_v2_engine import save_parquet
from .warm_v3_expert import screening_gate
from .warm_v3_residual_admission import FEATURES, HISTORICAL, INNER, OUTER, pair_frame, source


TRIAL = "WV3-610"
CONTRACT = common.REPORT / "WV3-610_TWO_HEAD_ADMISSION_CONTRACT.json"
SCREEN_REPORT = common.REPORT / "WV3-610_SCREEN.json"
OUTER_REPORT = common.REPORT / "WV3-610_OUTER.json"
MODEL_ROOT = common.ART / TRIAL
PREFERENCE_MODEL = common.ART / "WV3-601" / "MODEL.txt"

ACTION_PARAMS = {
    "objective": "binary",
    "metric": "None",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_data_in_leaf": 500,
    "lambda_l2": 10.0,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "seed": 20260910,
    "feature_fraction_seed": 20260910,
    "bagging_seed": 20260910,
    "verbosity": -1,
}
ACTION_ROUNDS = 60


def choose_expected_decisions(
    pairs: pd.DataFrame, preference_probability: np.ndarray, actionability_probability: np.ndarray
) -> pd.DataFrame:
    chosen = pairs[
        [
            "customer_id",
            "challenger_article_id",
            "victim_article_id",
            "challenger_target",
            "victim_target",
            "challenger_rank",
            "victim_rank",
            "unit_gain",
        ]
    ].copy()
    chosen["preference_probability"] = np.asarray(preference_probability, dtype=np.float64)
    chosen["actionability_probability"] = np.asarray(actionability_probability, dtype=np.float64)
    chosen["expected_delta"] = (
        chosen.actionability_probability
        * (2.0 * chosen.preference_probability - 1.0)
        * chosen.unit_gain
    )
    chosen = chosen[chosen.expected_delta > 0]
    chosen = chosen.sort_values(
        [
            "customer_id",
            "expected_delta",
            "actionability_probability",
            "preference_probability",
            "victim_rank",
            "challenger_rank",
            "challenger_article_id",
            "victim_article_id",
        ],
        ascending=[True, False, False, False, False, True, True, True],
        kind="mergesort",
    ).drop_duplicates("customer_id", keep="first")
    chosen["actual_delta"] = (
        chosen.challenger_target.astype(np.int8) - chosen.victim_target.astype(np.int8)
    ) * chosen.unit_gain
    return chosen.sort_values("customer_id", kind="mergesort").reset_index(drop=True)


def actionability_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    matrices = []
    labels = []
    weights = []
    sources = []
    for name, folder in HISTORICAL:
        path, meta = source(folder)
        check = meta.get("independent_source_check", {})
        assert check and all(value == 0 for value in check.values())
        pairs = pair_frame(path, discordant_only=False)
        y = (pairs.challenger_target != pairs.victim_target).to_numpy(np.uint8)
        matrices.append(pairs[FEATURES].to_numpy(np.float32))
        labels.append(y)
        weights.append(pairs.unit_gain.to_numpy(np.float32))
        sources.append(
            {
                "window": name,
                "cutoff": meta["cutoff"],
                "rows": len(pairs),
                "actionable_rows": int(y.sum()),
                "actionable_share": float(y.mean()),
                "path": str(path),
            }
        )
        del pairs, y
        gc.collect()
    x = np.concatenate(matrices)
    y = np.concatenate(labels)
    weight = np.concatenate(weights)
    del matrices, labels, weights
    assert np.isfinite(x).all() and np.isfinite(weight).all()
    audit = {
        "role": "strictly_historical_all_pair_actionability_supervision",
        "sources": sources,
        "rows": len(y),
        "actionable_rows": int(y.sum()),
        "neutral_rows": int((y == 0).sum()),
        "actionable_share": float(y.mean()),
        "weighted_actionable_share": float(weight[y == 1].sum() / weight.sum()),
        "features": FEATURES,
        "latest_label_end_exclusive": "2019-11-27",
        "earliest_inner_cutoff": "2019-12-25",
        "final_week": "not_run",
    }
    return x, y, weight, audit


def fit_actionability(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> lgb.Booster:
    assert 0 < y.sum() < len(y)
    dataset = lgb.Dataset(x, label=y, weight=weight, feature_name=FEATURES, free_raw_data=True)
    return lgb.train(ACTION_PARAMS, dataset, num_boost_round=ACTION_ROUNDS)


def evaluate(folder: str, name: str, preference: lgb.Booster, action: lgb.Booster, stage: str):
    path, meta = source(folder)
    started = time.perf_counter()
    pairs = pair_frame(path, discordant_only=False)
    x = pairs[FEATURES].to_numpy(np.float32)
    preference_probability = preference.predict(x, num_threads=4)
    actionability_probability = action.predict(x, num_threads=4)
    chosen = choose_expected_decisions(pairs, preference_probability, actionability_probability)
    beneficial = chosen.actual_delta > 1e-15
    harmful = chosen.actual_delta < -1e-15
    neutral = ~(beneficial | harmful)
    delta = float(chosen.actual_delta.sum() / meta["total_users"])
    baseline = float(meta["baseline_map_population_component"])
    result = {
        "window": name,
        "cutoff": meta["cutoff"],
        "role": stage,
        "source": str(path),
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
        "activity_probability_quantiles": np.quantile(actionability_probability, [0, 0.5, 0.9, 0.99, 1]).tolist(),
        "protected_ranks1_7_changes": 0,
        "maximum_swaps_per_user": 1,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    return result, chosen


def experiment_contract() -> dict:
    return {
        "created_at": now(),
        "experiment_id": TRIAL,
        "parent": "WV3-601",
        "architecture_family": "two_head_expected_gain_admission",
        "hypothesis": "A separate historical actionability head can prioritize non-neutral pairs, while the frozen WV3-601 preference head predicts direction; their product estimates signed AP swap value.",
        "preference_head": "exact frozen WV3-601 2019-only classifier",
        "actionability_head": "fixed LightGBM binary classifier trained on all 2019 challenger-victim pairs; label is whether exactly one item was purchased",
        "actionability_features": FEATURES,
        "actionability_params": ACTION_PARAMS,
        "actionability_rounds": ACTION_ROUNDS,
        "decision_score": "P(actionable) * (2*P(challenger wins given actionable)-1) * exact position unit_gain",
        "decision": "at most one maximum positive expected-delta swap; Top1-7 immutable; no additional threshold",
        "candidate_pool": "unchanged WV2-601 ranks1-50 subset of the existing 100-300 pool",
        "inner_gate": {
            "standard_vs_WV2_601": "mean>=+0.0001, at least3 positive, worst>=-0.0005",
            "incremental_vs_WV3_601": "mean>0, at least3 nondegrade, worst>=-0.0002",
        },
        "outer_policy": "one exposure only if both inner gates pass; no action-head or decision-score rescue",
        "expected_minutes": 15,
        "fallback": "WV3-601 stable candidate, otherwise WV2-601",
        "final_week": "2020-09-16 not_run",
    }


def register() -> None:
    common.setup()
    common.budget(15)
    if not CONTRACT.exists():
        write(CONTRACT, experiment_contract())
    registry = read(common.REGISTRY)
    if not any(row["experiment_id"] == TRIAL for row in registry["trials"]):
        common.register(
            TRIAL,
            "two_head_expected_gain_admission",
            experiment_contract()["hypothesis"],
            candidate_protocol=experiment_contract()["candidate_pool"],
            training_protocol="frozen WV3-601 preference head plus one all-pair actionability head trained only on four 2019 sources",
            features=FEATURES,
            params={
                "actionability": ACTION_PARAMS,
                "rounds": ACTION_ROUNDS,
                "decision_score": experiment_contract()["decision_score"],
                "inner_gate": experiment_contract()["inner_gate"],
                "final_week": "not_run",
            },
            expected_minutes=15,
        )


def incremental_gate(windows: dict) -> dict:
    prior = read(common.REPORT / "WV3-601_SCREEN.json")["windows"]
    delta = {name: row["MAP@12"] - prior[name]["MAP@12"] for name, row in windows.items()}
    values = list(delta.values())
    return {
        "per_window_delta_vs_WV3_601": delta,
        "mean_delta_vs_WV3_601": float(np.mean(values)),
        "nondegrade_windows_vs_WV3_601": sum(value >= 0 for value in values),
        "worst_delta_vs_WV3_601": min(values),
        "passed": bool(np.mean(values) > 0 and sum(value >= 0 for value in values) >= 3 and min(values) >= -0.0002),
    }


def screen() -> dict:
    common.setup()
    common.budget(15)
    entry = next(row for row in read(common.REGISTRY)["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "preregistered" and entry["outer_exposures"] == 0
    assert PREFERENCE_MODEL.is_file()
    started = time.perf_counter()
    x, y, weight, audit = actionability_training_data()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    training_audit_path = MODEL_ROOT / "TRAINING_AUDIT.json"
    write(training_audit_path, audit)
    action = fit_actionability(x, y, weight)
    del x, y, weight
    gc.collect()
    model_path = MODEL_ROOT / "ACTIONABILITY_MODEL.txt"
    action.save_model(str(model_path))
    write(
        MODEL_ROOT / "MODEL.json",
        {
            "experiment_id": TRIAL,
            "preference_model": str(PREFERENCE_MODEL),
            "actionability_model": evidence_id(model_path, reason="explicit_registry_evidence"),
            "features": FEATURES,
            "params": ACTION_PARAMS,
            "rounds": ACTION_ROUNDS,
            "training_audit": str(training_audit_path),
            "outer_labels_used": False,
            "final_week": "not_run",
        },
    )
    preference = lgb.Booster(model_file=str(PREFERENCE_MODEL))
    windows = {}
    for name, folder in INNER:
        row, chosen = evaluate(folder, name, preference, action, "historical_two_head_inner_screen")
        windows[name] = row
        root = MODEL_ROOT / name / "inner"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"two_head_inner": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = screening_gate(row["population_delta"] for row in windows.values())
    incremental = incremental_gate(windows)
    passed = standard["passed"] and incremental["passed"]
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "architecture": "two_head_expected_gain_admission",
        "training": audit,
        "windows": windows,
        "standard_screening_vs_WV2_601": standard,
        "incremental_screening_vs_WV3_601": incremental,
        "passed": passed,
        "features": FEATURES,
        "actionability_params": ACTION_PARAMS,
        "actionability_rounds": ACTION_ROUNDS,
        "candidate_pool_changed": False,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(SCREEN_REPORT, result)
    common.update(
        TRIAL,
        decision="inner_pass" if passed else "reject_inner",
        inner_evidence={"standard": standard, "incremental": incremental, "passed": passed},
        runtime=result["runtime_seconds"],
        artifact_paths=[str(SCREEN_REPORT), str(MODEL_ROOT / "MODEL.json")],
    )
    common.log(
        f"{TRIAL} inner screen",
        "WV3-602 showed 96.56% neutral selections and only 4.56% oracle-beneficial user capture for the one-head model.",
        experiment_contract()["hypothesis"],
        f"{training_audit_path}; {SCREEN_REPORT}",
        f"standard mean={standard['mean_population_delta']:+.9f}, standard pass={standard['passed']}; incremental mean={incremental['mean_delta_vs_WV3_601']:+.9f}, incremental pass={incremental['passed']}; overall pass={passed}.",
        "Freeze both heads and expose once." if passed else "Reject the two-head architecture at inner; no score or round rescue.",
        "Run one outer confirmation only if both gates pass; otherwise audit whether candidate-specific target features, not pair actionability, are missing.",
        alternatives="The new architecture changes the factorized target rather than tuning WV3-601 probability threshold or swap count.",
        experiment="Train one fixed60-round actionability classifier on all historical pairs and multiply it with the frozen preference margin and exact AP position weight.",
        reflection="The incremental gate prevents a second outer exposure merely for reproducing the existing stable candidate.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "passed": passed, "seconds": result["runtime_seconds"]}, flush=True)
    return result


def confirm() -> dict:
    common.setup()
    common.budget(10)
    registry = read(common.REGISTRY)
    entry = next(row for row in registry["trials"] if row["experiment_id"] == TRIAL)
    assert entry["decision"] == "inner_pass" and entry["outer_exposures"] == 1
    assert read(SCREEN_REPORT)["passed"]
    if OUTER_REPORT.exists():
        return read(OUTER_REPORT)
    preference = lgb.Booster(model_file=str(PREFERENCE_MODEL))
    action = lgb.Booster(model_file=str(MODEL_ROOT / "ACTIONABILITY_MODEL.txt"))
    started = time.perf_counter()
    windows = {}
    for name, folder in OUTER:
        row, chosen = evaluate(folder, name, preference, action, "frozen_two_head_outer")
        windows[name] = row
        root = MODEL_ROOT / name / "outer"
        root.mkdir(parents=True, exist_ok=True)
        save_parquet(chosen, root / "decisions.parquet")
        write(root / "REVIEW.json", row)
        print({"two_head_outer": name, "delta": row["population_delta"], "selected": row["selected_users"]}, flush=True)
    standard = common.summary({name: row["MAP@12"] for name, row in windows.items()}, 2020)
    prior = read(common.REPORT / "WV3-601_OUTER.json")
    delta = {name: windows[name]["MAP@12"] - prior["per_window_MAP"][name] for name in windows}
    values = list(delta.values())
    incremental = {
        "per_window_delta_vs_WV3_601": delta,
        "mean_delta_vs_WV3_601": float(np.mean(values)),
        "nondegrade_windows_vs_WV3_601": sum(value >= 0 for value in values),
        "worst_delta_vs_WV3_601": min(values),
    }
    better_champion = standard["stable"] and incremental["mean_delta_vs_WV3_601"] > 0
    result = {
        "created_at": now(),
        "experiment_id": TRIAL,
        "architecture": "two_head_expected_gain_admission",
        "windows": windows,
        **standard,
        "incremental_vs_WV3_601": incremental,
        "better_than_current_champion": better_champion,
        "model_source": str(MODEL_ROOT / "MODEL.json"),
        "candidate_pool_changed": False,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    write(OUTER_REPORT, result)
    common.update(
        TRIAL,
        decision="promote_candidate" if better_champion else "reject_outer",
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
        "The two-head expected-gain model passed both fixed inner gates and was frozen before one outer exposure.",
        experiment_contract()["hypothesis"],
        str(OUTER_REPORT),
        f"mean MAP={standard['mean_MAP']:.9f}; delta vs WV2-601={standard['delta_vs_WV2_601']:+.9f}; mean delta vs WV3-601={incremental['mean_delta_vs_WV3_601']:+.9f}; stable={standard['stable']}; better champion={better_champion}.",
        "Promote as the current candidate." if better_champion else "Reject this concrete two-head variant; keep WV3-601.",
        "Continue only with a new diagnostic-supported family if target0.03 remains unmet and time permits.",
        alternatives="No outer-aware adjustment to activity rounds, factorization or decision score is allowed.",
        experiment="One outer exposure of the frozen historical preference and actionability heads.",
        reflection="The outer comparison is against both the original baseline and the current stable candidate, so a redundant stable result cannot replace it.",
    )
    print({"trial": TRIAL, "standard": standard, "incremental": incremental, "better": better_champion}, flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["register", "screen", "confirm"])
    globals()[parser.parse_args().command]()
