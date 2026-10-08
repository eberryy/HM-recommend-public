"""MIND-WARM-RANK-002: chronological, rejectable challenger/victim relations.

Predictions use an explicit feature allowlist; labels enter only historical
fitting or evaluation after actions are saved. No retrieval/model tuning here.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import shutil
import time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import mind_warm_ranking as old
from .mind_warm_dense import DENSE_FEATURES, build_dense_features

ROOT = old.ROOT
ART = ROOT / "artifacts/mind_warm_side/MIND-WARM-RANK-002"
REPORT = ROOT / "reports/mind_warm_side"
CONTRACT = REPORT / "MIND_WARM_RANK002_CONTRACT_V2.json"
METRICS = REPORT / "MIND_WARM_RANK002_METRICS.json"
SCHEMA = "mind-warm-relation-v2-full-chronological"
USER = ["user_history_events_12w", "user_unique_items_12w", "user_days_since_last_purchase",
        "user_avg_price_12w", "user_online_share_12w"]
ITEM = ["item_events_7d", "item_events_28d", "item_events_12w", "item_unique_customers_28d",
        "item_avg_price_28d", "item_days_since_last_sale", "item_trend_7d_vs_28d",
        "user_item_price_gap", "user_item_events_12w", "user_item_days_since_last_purchase",
        "user_product_code_events_12w", "user_product_code_days_since",
        "user_product_type_share_12w", "user_department_share_12w",
        "user_garment_share_12w", "user_colour_share_12w"]
CATEGORIES = ["article_product_code", "article_product_type_no", "article_department_no",
              "article_garment_group_no", "article_colour_master_id"]
VICTIM = ["score_base", "score_bpr", "latent", "missing", "r0", "r1", "rrf_score", "rf"]
KEYS = ["customer_id", "challenger_article_id", "victim_article_id", "champion_rank"]
PROBS = ["p_harm", "p_neutral", "p_benefit"]
CONTEXT = [*USER, *[f"{prefix}_{f}" for f in ITEM for prefix in ("c", "v", "diff")],
           *[f"same_{f}" for f in CATEGORIES], *[f"v_{f}" for f in VICTIM], "champion_rank"]
FEATURES = {"context_control": CONTEXT,
            "dense_primary": [*CONTEXT, *[f"{p}_{f}" for f in DENSE_FEATURES for p in ("c", "v", "diff")]]}


def con() -> duckdb.DuckDBPyConnection:
    result = duckdb.connect()
    result.execute("SET threads=4")
    result.execute("SET memory_limit='6GiB'")
    return result


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value: dict) -> None:
    old._write_json(path, value)


def progress(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def resources(start: float | None = None) -> dict:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *[(x, ctypes.c_ulonglong) for x in ("total", "available", "page_total", "page_free", "virtual_total", "virtual_free", "extended")]]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("cannot verify available physical memory")
    available = status.available / 2**30
    disk = shutil.disk_usage(ROOT).free / 2**30
    if available < 4 or disk < 10:
        raise RuntimeError(f"resource gate failed: RAM={available:.2f} GiB; disk={disk:.2f} GiB")
    if start is not None and time.perf_counter() - start > 90 * 60:
        raise RuntimeError("registered 90-minute runtime cap reached; no downsizing/rescue")
    return {"available_ram_gib": available, "free_disk_gib": disk}


def check_temporal(training: list[str], evaluation: str) -> None:
    old._guard(evaluation)
    if not training or len(set(training)) != len(training):
        raise ValueError("unique historical training windows required")
    for cutoff in training:
        old._guard(cutoff)
        if date.fromisoformat(cutoff) + timedelta(days=7) > date.fromisoformat(evaluation):
            raise ValueError("training label interval overlaps evaluation cutoff")


def relation_labels(challenger: np.ndarray, victim: np.ndarray) -> np.ndarray:
    if not np.isin(challenger, [0, 1]).all() or not np.isin(victim, [0, 1]).all():
        raise ValueError("only binary implicit purchase labels accepted")
    return (np.asarray(challenger, dtype=np.int8) - np.asarray(victim, dtype=np.int8) + 1).astype(np.int8)


def select_actions(scores: pd.DataFrame) -> pd.DataFrame:
    """One label-free best positive-utility action per user; protected head."""
    if set(scores.columns) != set(KEYS + PROBS):
        raise ValueError("selector accepts only keys and three probabilities; labels forbidden")
    if scores[KEYS].isna().any().any() or not scores.champion_rank.between(8, 12).all():
        raise ValueError("invalid key or unprotected victim rank")
    probs = scores[PROBS].to_numpy(float)
    if not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(1), 1, atol=1e-6):
        raise ValueError("invalid class probabilities")
    result = scores.copy()
    result["replacement_score"] = (result.p_benefit - result.p_harm) / result.champion_rank
    result = result[result.replacement_score > 0]
    return result.sort_values(
        ["customer_id", "replacement_score", "challenger_article_id", "victim_article_id"],
        ascending=[True, False, True, True], kind="mergesort",
    ).drop_duplicates("customer_id").reset_index(drop=True)


def name_for(stage: str, cutoff: str) -> str:
    return next(k for k, v in (old.INNER if stage == "inner" else old.OUTER).items() if v == cutoff)


def parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with con() as db:
        # Bound the common MIND writer too, not only the newer fit/read paths.
        db.execute("SET threads=2")
        db.register("out_frame", frame)
        db.execute(f"COPY out_frame TO '{old._sql_path(path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def source_features(cutoff: str) -> Path:
    old_path = old.ARTIFACT_ROOT / cutoff / "features.parquet"
    if old_path.exists():
        marker = read(old_path.with_name("FEATURES.json"))
        if marker["cutoff"] != cutoff or not marker["history_strictly_before_cutoff"]:
            raise ValueError("unsafe reused feature metadata")
        return old_path
    # This adapter materializes cutoff-safe aggregate columns only, not a model.
    if not old._mind_candidates(cutoff).exists() or not old._mind_candidates(cutoff).with_name("RETRIEVAL.json").exists():
        raise FileNotFoundError("frozen MIND candidates absent; implicit training forbidden")
    path, _ = old.build_features(cutoff, "cpu", output_root=ART / "aggregate_adapter")
    return path


def build_pairs(stage: str, cutoff: str) -> tuple[Path, dict]:
    old._guard(cutoff)
    folder = ART / cutoff
    output, marker = folder / "pairs.parquet", folder / "PAIRS.json"
    if output.exists() and marker.exists():
        meta = read(marker)
        if meta["schema"] != SCHEMA or meta["features"] != FEATURES or meta["stage"] != stage:
            raise ValueError("incompatible pair cache")
        return output, meta
    started = time.perf_counter()
    resources()
    source = source_features(cutoff)
    from .mind_warm_baseline import reconstruct_full_baseline
    baseline, replay = reconstruct_full_baseline(cutoff)
    tail = baseline.loc[baseline.champion_rank.between(8, 12), ["customer_id", "article_id", "champion_rank", "target"]]
    if not (tail.groupby("customer_id").size() == 5).all():
        raise AssertionError("every baseline user must have five tail victims")
    columns = ["customer_id", "article_id", "target", "mind_is_new", *USER, *ITEM, *CATEGORIES]
    with con() as db:
        db.register("tail", tail)
        frame = db.execute(f"SELECT {','.join('f.' + x for x in columns)} FROM read_parquet(?) f "
                           "WHERE f.mind_is_new=1 OR EXISTS(SELECT 1 FROM tail t WHERE f.customer_id=t.customer_id AND f.article_id=t.article_id)", [str(source)]).fetchdf()
        confidence = db.execute(f"SELECT r.customer_id,r.article_id,{','.join('r.' + x for x in VICTIM)} "
                                "FROM read_parquet(?) r SEMI JOIN tail USING(customer_id,article_id)", [str(old._warm_rank_path(stage, cutoff))]).fetchdf()
    if len(frame) > 800000 or frame.duplicated(["customer_id", "article_id"]).any():
        raise RuntimeError("candidate identity/resource gate failed")
    if not frame.loc[frame.mind_is_new == 1, "customer_id"].isin(tail.customer_id).all():
        raise AssertionError("challenger user without baseline tail; never silently filter users")
    labels = frame[["customer_id", "article_id", "target"]].copy()
    parquet(labels, folder / "labels.parquet")
    dense, dense_meta = build_dense_features(cutoff, frame[["customer_id", "article_id"]].copy(), "cpu")
    frame = frame.merge(dense, on=["customer_id", "article_id"], validate="one_to_one")
    challengers = frame[frame.mind_is_new == 1].copy()
    victims = tail.rename(columns={"target": "replay_target"}).merge(frame, on=["customer_id", "article_id"], validate="one_to_one")
    victims = victims.merge(confidence, on=["customer_id", "article_id"], validate="one_to_one")
    if len(victims) != len(tail) or (victims.target != victims.replay_target).any() or victims.mind_is_new.any():
        raise AssertionError("baseline tail not conserved or label mismatch")
    if not challengers.customer_id.isin(victims.customer_id).all():
        raise AssertionError("challenger user without baseline tail")
    expected = 5 * len(challengers)
    select = ["c.customer_id", "c.article_id AS challenger_article_id", "v.article_id AS victim_article_id",
              "v.champion_rank::INTEGER AS champion_rank", "c.target::UTINYINT AS challenger_target", "v.target::UTINYINT AS victim_target",
              "(c.target::INTEGER-v.target::INTEGER+1)::TINYINT AS relation_label"]
    select += [f"c.{f}::FLOAT AS {f}" for f in USER]
    for f in [*ITEM, *DENSE_FEATURES]:
        select += [f"c.{f}::FLOAT AS c_{f}", f"v.{f}::FLOAT AS v_{f}", f"(c.{f}-v.{f})::FLOAT AS diff_{f}"]
    select += [f"(c.{f}=v.{f})::FLOAT AS same_{f}" for f in CATEGORIES]
    select += [f"v.{f}::FLOAT AS v_{f}" for f in VICTIM]
    with con() as db:
        db.register("c", challengers)
        db.register("v", victims)
        db.execute(f"COPY (SELECT {','.join(select)} FROM c JOIN v USING(customer_id)) TO '{old._sql_path(output)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
        counts = db.execute("SELECT relation_label,count(*) FROM read_parquet(?) GROUP BY 1 ORDER BY 1", [str(output)]).fetchall()
        identity = db.execute("SELECT count(*),count(DISTINCT(customer_id,challenger_article_id,victim_article_id)),count(DISTINCT(customer_id,challenger_article_id)) FILTER(WHERE challenger_target=1) FROM read_parquet(?)", [str(output)]).fetchone()
    if identity[0] != expected or identity[1] != expected or identity[2] != int(challengers.target.sum()):
        raise AssertionError("complete pair product/positive coverage failed")
    meta = {"schema": SCHEMA, "stage": stage, "cutoff": cutoff, "features": FEATURES,
            "source": str(source), "candidate_rows": len(frame), "challenger_rows": len(challengers),
            "challenger_positive_rows": int(challengers.target.sum()), "victim_rows": len(victims),
            "pair_rows": expected, "class_counts": {str(k): int(v) for k, v in counts},
            "complete_pair_product": True, "positive_challenger_coverage": 1.0,
            "baseline": replay, "dense": dense_meta, "elapsed_seconds": time.perf_counter() - started,
            "final_week": "not_run"}
    save(marker, meta)
    progress(f"pairs {cutoff}: {expected:,} rows; classes={meta['class_counts']}")
    return output, meta


def model_path(training: list[str], variant: str) -> Path:
    return ART / "models" / ("through_" + training[-1]) / (variant + ".txt")


def fit(training: list[str], variant: str, config: dict) -> tuple[lgb.Booster, dict]:
    path = model_path(training, variant)
    marker = path.with_suffix(".json")
    features = FEATURES[variant]
    if path.exists() and marker.exists():
        meta = read(marker)
        if meta["training"] != training or meta["features"] != features or meta.get("schema") != SCHEMA or meta.get("registered_model") != config:
            raise ValueError("stale model cache")
        return lgb.Booster(model_file=str(path)), meta
    started = time.perf_counter()
    hardware = resources()
    paths = [str(ART / x / "pairs.parquet") for x in training]
    with con() as db:
        n = db.execute("SELECT count(*) FROM read_parquet(?)", [paths]).fetchone()[0]
        counts = dict(db.execute("SELECT relation_label,count(*) FROM read_parquet(?) GROUP BY 1", [paths]).fetchall())
        if n > 7000000 or counts.get(0, 0) < 100 or counts.get(2, 0) < 100:
            raise RuntimeError("training resource/class-count gate failed")
        frame = db.execute(f"SELECT {','.join(features)},relation_label FROM read_parquet(?)", [paths]).fetchdf()
    labels = frame.pop("relation_label").to_numpy(np.int8)
    matrix = frame.to_numpy(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    del frame
    gc.collect()
    params = {k: config[k] for k in ("objective", "num_class", "learning_rate", "num_leaves", "max_depth", "min_data_in_leaf", "lambda_l2", "seed")}
    params.update(num_threads=4, deterministic=True, force_col_wise=True, verbosity=-1)
    dataset = lgb.Dataset(matrix, label=labels, feature_name=features, free_raw_data=True)
    progress(f"fit {variant} through {training[-1]}: {n:,} pairs; {len(features)} features")
    model = lgb.train(params, dataset, num_boost_round=config["rounds"])
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    importance = sorted([{"feature": f, "gain": float(g)} for f, g in zip(features, model.feature_importance("gain"))], key=lambda r: -r["gain"])
    meta = {"schema": SCHEMA, "registered_model": config, "path": str(path), "training": training, "features": features, "pair_rows": n,
            "class_counts": {str(k): int(v) for k, v in counts.items()}, "params": params, "rounds": config["rounds"],
            "importance": importance, "resources": hardware, "elapsed_seconds": time.perf_counter() - started}
    save(marker, meta)
    del dataset, matrix, labels
    gc.collect()
    return model, meta


def evaluate(stage: str, cutoff: str, training: list[str], variant: str, model: lgb.Booster, pair_meta: dict) -> dict:
    check_temporal(training, cutoff)
    folder = ART / stage / cutoff
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    best_chunks = []
    calibration = {str(i): {"rows": 0, "sum_p": 0., "positive_rows": 0, "brier_sum": 0.} for i in range(3)}
    with con() as db:
        # No labels in prediction/selection path, including unused extra columns.
        selected = list(dict.fromkeys(KEYS + FEATURES[variant]))
        cursor = db.execute(f"SELECT {','.join(selected)} FROM read_parquet(?)", [str(ART / cutoff / "pairs.parquet")])
        while True:
            batch = cursor.fetch_df_chunk(32)
            if not len(batch):
                break
            matrix = batch[FEATURES[variant]].to_numpy(np.float32)
            matrix[~np.isfinite(matrix)] = np.nan
            probs = model.predict(matrix, num_threads=4)
            scores = batch[KEYS].copy()
            scores[PROBS] = probs
            best_chunks.append(select_actions(scores)[KEYS + PROBS])
    scores = pd.concat(best_chunks, ignore_index=True) if best_chunks else pd.DataFrame(columns=KEYS + PROBS)
    actions = select_actions(scores)
    # Freeze the action artifact before opening held-out labels for evaluation.
    action_file = folder / f"{variant}-decisions.parquet"
    parquet(actions, action_file)
    from .mind_warm_baseline import reconstruct_full_baseline
    baseline, replay = reconstruct_full_baseline(cutoff)
    with con() as db:
        labels = db.execute("SELECT * FROM read_parquet(?)", [str(ART / cutoff / "labels.parquet")]).fetchdf()
    deltas = old.exact_action_deltas(baseline, actions, labels)
    if not len(deltas):
        deltas = actions.assign(challenger_target=pd.Series(dtype=int), victim_target=pd.Series(dtype=int),
                                baseline_ap=pd.Series(dtype=float), reranked_ap=pd.Series(dtype=float), actual_delta=pd.Series(dtype=float))
    parquet(deltas, folder / f"{variant}-evaluated.parquet")
    # Independent full Top12 replay verifies the additive action delta result.
    top = baseline[baseline.champion_rank <= 12].copy()
    replacements = deltas[["customer_id", "champion_rank", "challenger_article_id", "challenger_target"]]
    full = top.merge(replacements, on=["customer_id", "champion_rank"], how="left", validate="one_to_one")
    has = full.challenger_article_id.notna()
    full.loc[has, "article_id"] = full.loc[has, "challenger_article_id"]
    full.loc[has, "target"] = full.loc[has, "challenger_target"].astype(np.uint8)
    full_map = old._mean_ap(full, replay["total_users"])
    delta = float(deltas.actual_delta.sum() / replay["total_users"])
    if abs(full_map - replay["MAP@12"] - delta) > 1e-12 or full.duplicated(["customer_id", "article_id"]).any():
        raise AssertionError("independent full-list replay/uniqueness check failed")
    # Post-freeze class calibration: scalar pooled diagnostics, not tuning.
    with con() as db:
        cursor = db.execute(f"SELECT {','.join(FEATURES[variant])},relation_label FROM read_parquet(?)", [str(ART / cutoff / "pairs.parquet")])
        while True:
            batch = cursor.fetch_df_chunk(32)
            if not len(batch):
                break
            y = batch.pop("relation_label").to_numpy(np.int8)
            matrix = batch.to_numpy(np.float32)
            matrix[~np.isfinite(matrix)] = np.nan
            probs = model.predict(matrix, num_threads=4)
            for i in range(3):
                c = calibration[str(i)]
                c["rows"] += len(y)
                c["sum_p"] += float(probs[:, i].sum())
                c["positive_rows"] += int((y == i).sum())
                c["brier_sum"] += float(((probs[:, i] - (y == i)) ** 2).sum())
    for c in calibration.values():
        c.update(mean_predicted_probability=c["sum_p"] / c["rows"], observed_class_rate=c["positive_rows"] / c["rows"], brier_score=c["brier_sum"] / c["rows"])
    good = deltas.actual_delta > 1e-15
    bad = deltas.actual_delta < -1e-15
    result = {"variant": variant, "training_cutoffs": training, "MAP@12": full_map,
              "baseline_MAP@12": replay["MAP@12"], "total_users": replay["total_users"],
              "delta_vs_baseline": delta, "selected_users": len(deltas),
              "beneficial_users": int(good.sum()), "harmful_users": int(bad.sum()), "neutral_users": int((~(good | bad)).sum()),
              "gross_positive_MAP": float(deltas.loc[good, "actual_delta"].sum() / replay["total_users"]),
              "gross_negative_MAP": float(deltas.loc[bad, "actual_delta"].sum() / replay["total_users"]),
              "selected_positive_challengers": int(deltas.challenger_target.sum()),
              "positive_challenger_retention": float(deltas.challenger_target.sum() / max(pair_meta["challenger_positive_rows"], 1)),
              "selected_victim_positive_rows": int(deltas.victim_target.sum()), "class_calibration": calibration,
              "actions": str(action_file), "protected_ranks_1_7_changes": 0, "maximum_actions_per_user": 1,
              "selection_used_labels": False, "independent_MAP_replay_error": abs(full_map - replay["MAP@12"] - delta),
              "elapsed_seconds": time.perf_counter() - started}
    save(folder / f"{variant}-RESULT.json", result)
    progress(f"{stage} {cutoff} {variant}: delta={delta:+.8f}; actions={len(deltas)}, benefit={good.sum()}, harm={bad.sum()}")
    return result


def gate(windows: dict, stage: str) -> dict:
    rows = list(windows.values())
    values = [x["dense_primary"]["delta_vs_baseline"] for x in rows]
    mean = float(np.mean(values))
    checks = {"complete_windows": len(rows) == (3 if stage == "inner" else 4),
              "mean_positive": mean > 0,
              "nondegrade_windows": sum(x >= 0 for x in values) >= 3,
              "worst_delta": min(values) >= (-0.0002 if stage == "inner" else -0.0005)}
    context_delta = float(np.mean([x["dense_primary"]["MAP@12"] - x["context_control"]["MAP@12"] for x in rows]))
    if stage == "inner":
        checks["dense_beats_context"] = context_delta > 0
        checks["active_admission_windows"] = sum(x["dense_primary"]["selected_users"] > 0 for x in rows) >= 2
    return {"passed": all(checks.values()), "checks": checks, "mean_delta_vs_baseline": mean,
            "mean_delta_vs_context": context_delta, "worst_delta": min(values), "nondegrade_windows": sum(x >= 0 for x in values)}


def run() -> dict:
    started = time.perf_counter()
    if METRICS.exists():
        prior = read(METRICS)
        if prior.get("schema") == SCHEMA and prior.get("status") in {
            "development_passed_not_promoted", "development_failed_baseline_retained"
        }:
            progress("completed registered run already exists; returning evidence without new evaluation")
            return prior
    config = read(CONTRACT)
    original = read(REPORT / "MIND_WARM_RANK002_CONTRACT.json")
    ART.mkdir(parents=True, exist_ok=True)
    feature_contract = ART / "FEATURE_ALLOWLIST.json"
    if feature_contract.exists() and read(feature_contract) != FEATURES:
        raise ValueError("feature allowlist drift")
    save(feature_contract, FEATURES)
    result = {"schema": SCHEMA, "experiment_id": "MIND-WARM-RANK-002", "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "contract_revision": 2, "final_week": "not_run", "independent_confirmation": "not_run", "chronological_development": {}, "models": {}, "pairs": {}}
    save(METRICS, result)
    try:
        for stage, schedule in (("outer", config["temporal_protocol"]["chronological_development"]),):
            for cutoff, training in schedule.items():
                resources(started)
                check_temporal(training, cutoff)
                for source in training:
                    _, meta = build_pairs("outer", source)
                    result["pairs"][source] = meta
                _, pair_meta = build_pairs(stage, cutoff)
                result["pairs"][cutoff] = pair_meta
                results = {}
                for variant in FEATURES:
                    resources(started)
                    model, model_meta = fit(training, variant, original["model"])
                    result["models"][f"{training[-1]}_{variant}"] = model_meta
                    results[variant] = evaluate(stage, cutoff, training, variant, model, pair_meta)
                    del model
                    gc.collect()
                result["chronological_development"][cutoff] = results
                save(METRICS, result)
            result["development_gate"] = gate(result["chronological_development"], "inner")
            save(METRICS, result)
        result["status"] = ("development_passed_not_promoted" if result["development_gate"]["passed"] else "development_failed_baseline_retained")
    except Exception as error:
        result["status"] = "engineering_stopped"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        save(METRICS, result)
    progress(result["status"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-cutoff", choices=list(old.OUTER.values()))
    args = parser.parse_args()
    if args.build_cutoff:
        build_pairs("outer", args.build_cutoff)
    else:
        run()
