from __future__ import annotations

import gc
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from .m2 import M2Config
from .m210 import score_models
from .m211 import _load_category_maps
from .m3 import _peak_working_set_bytes
from .m33 import ROLLING_PROTOCOL, variant_feature_sets
from .m4_contract import STATIC_FIELDS, WARM_MAP, atomic_json, file_identity, stable_u64
from .m4_retrieval import _history_counts_and_seeds
from .m5_data import (
    AFFINITY_FIELDS,
    M5_RUN_ID,
    PRIMARY_K,
    SIMILARITY_FEATURES,
    _append_context,
    _truth_keys,
    _user_context,
)


K_ADMIT = (0, 1, 3)
MAP_TOLERANCE = 1e-12
MAX_SINGLE_WINDOW_REGRESSION = 0.0002
NEGATIVE_PER_POSITIVE = 100
RANK_BUCKETS = 5
MODEL_FEATURES = [
    *SIMILARITY_FEATURES,
    "item_events_before_cutoff",
    "item_log1p_events_before_cutoff",
    "item_is_strict_cold",
    "user_history_events_12w",
    "user_log1p_history_events_12w",
    "user_history_distinct_items_12w",
    "user_days_since_last_event",
    "user_age",
    *[f"user_{field}_share_12w" for field in AFFINITY_FIELDS],
    *[f"item_{field}" for field in STATIC_FIELDS],
]
CATEGORICAL_FEATURES = ["item_is_strict_cold", *[f"item_{field}" for field in STATIC_FIELDS]]


def _stable_row_hash(
    *, user_index: np.ndarray, catalog_row: np.ndarray, rank: np.ndarray, salt: int,
) -> np.ndarray:
    with np.errstate(over="ignore"):
        value = (
            user_index.astype(np.uint64) * np.uint64(0x9E3779B185EBCA87)
            + catalog_row.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
            + rank.astype(np.uint64) * np.uint64(0x165667B19E3779F9)
            + np.uint64(salt)
        )
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return value


def _load_users(path: Path) -> list[str]:
    return pd.read_csv(path, dtype={"customer_id": str})["customer_id"].tolist()


def _sample_training_indices(
    arrays: dict[str, np.ndarray], *, cutoff: str, scale: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    users = arrays["_users"]
    threshold = {"10pct": 100_000, "30pct": 300_000, "full": 1_000_000}[scale]
    hashes = np.asarray([stable_u64(user) % 1_000_000 for user in users], dtype=np.uint64)
    scale_mask = hashes[arrays["user_index"]] < threshold
    positive = np.flatnonzero(scale_mask & (arrays["target"] == 1))
    negative = np.flatnonzero(scale_mask & (arrays["target"] == 0))
    if not len(positive):
        raise RuntimeError(f"M5 training cutoff has zero positives: {cutoff}")
    ranks = arrays["student_rank"]
    selected_negatives: list[np.ndarray] = []
    target_total = min(len(negative), NEGATIVE_PER_POSITIVE * len(positive))
    base_target = target_total // RANK_BUCKETS
    remainder = target_total % RANK_BUCKETS
    for bucket in range(RANK_BUCKETS):
        lower = bucket * (PRIMARY_K // RANK_BUCKETS) + 1
        upper = (bucket + 1) * (PRIMARY_K // RANK_BUCKETS)
        candidates = negative[(ranks[negative] >= lower) & (ranks[negative] <= upper)]
        take = min(len(candidates), base_target + int(bucket < remainder))
        if take:
            hashes = _stable_row_hash(
                user_index=arrays["user_index"][candidates],
                catalog_row=arrays["catalog_row"][candidates],
                rank=arrays["student_rank"][candidates],
                salt=stable_u64(f"{cutoff}:{bucket}"),
            )
            chosen = candidates[np.argpartition(hashes, take - 1)[:take]]
            selected_negatives.append(np.sort(chosen))
    selected_negative = np.concatenate(selected_negatives) if selected_negatives else np.empty(0, dtype=np.int64)
    selected = np.sort(np.concatenate([positive, selected_negative]))
    evidence = {
        "scale": scale,
        "all_candidate_rows_at_scale": int(np.count_nonzero(scale_mask)),
        "positive_rows_retained": int(len(positive)),
        "negative_rows_retained": int(len(selected_negative)),
        "negative_per_positive": float(len(selected_negative) / len(positive)),
        "rank_bucket_negative_rows": [int(len(value)) for value in selected_negatives],
        "unobserved_label_warning": "target=0 means no purchase observed in the next week, not a proven exposed rejection",
    }
    return selected, evidence


def _load_arrays(dataset_path: Path, users_path: Path) -> dict[str, np.ndarray]:
    with np.load(dataset_path) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    arrays["_users"] = np.asarray(_load_users(users_path), dtype=object)
    return arrays


def _frame(
    arrays: dict[str, np.ndarray], indices: np.ndarray, categories: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=np.arange(len(indices)))
    for name in SIMILARITY_FEATURES:
        frame[name] = arrays[name][indices]
    frame["item_events_before_cutoff"] = arrays["item_events_before_cutoff"][indices]
    frame["item_log1p_events_before_cutoff"] = np.log1p(
        arrays["item_events_before_cutoff"][indices].astype(np.float32)
    )
    frame["item_is_strict_cold"] = arrays["item_is_strict_cold"][indices]
    frame["user_history_events_12w"] = arrays["user_history_events_12w"][indices]
    frame["user_log1p_history_events_12w"] = np.log1p(
        arrays["user_history_events_12w"][indices].astype(np.float32)
    )
    for name in ("user_history_distinct_items_12w", "user_days_since_last_event", "user_age"):
        frame[name] = arrays[name][indices]
    for field in AFFINITY_FIELDS:
        frame[f"user_{field}_share_12w"] = arrays[f"user_{field}_share_12w"][indices]
    candidate_rows = arrays["catalog_row"][indices]
    for column, field in enumerate(STATIC_FIELDS):
        frame[f"item_{field}"] = categories[candidate_rows, column]
    frame["item_is_strict_cold"] = pd.Categorical(
        frame["item_is_strict_cold"], categories=[0, 1]
    )
    for column, field in enumerate(STATIC_FIELDS):
        name = f"item_{field}"
        frame[name] = pd.Categorical(
            frame[name], categories=range(int(categories[:, column].max()) + 1)
        )
    frame["target"] = arrays["target"][indices].astype(np.uint8)
    frame["user_index"] = arrays["user_index"][indices].astype(np.int32)
    frame["catalog_row"] = candidate_rows.astype(np.int32)
    return frame


def load_training_frame(
    *, data_dir: Path, cutoffs: list[str], categories: np.ndarray, scale: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    evidence: dict[str, Any] = {}
    user_offset = 0
    for cutoff in cutoffs:
        root = data_dir / cutoff
        arrays = _load_arrays(root / "cold_candidates_top50.npz", root / "users.csv")
        selected, sample = _sample_training_indices(arrays, cutoff=cutoff, scale=scale)
        frame = _frame(arrays, selected, categories)
        frame["user_index"] += user_offset
        user_offset += len(arrays["_users"])
        frame["cutoff"] = cutoff
        frames.append(frame)
        evidence[cutoff] = sample
        del arrays
        gc.collect()
    result = pd.concat(frames, ignore_index=True)
    return result, {
        "cutoffs": evidence,
        "rows": len(result),
        "positive_rows": int(result["target"].sum()),
        "positive_groups": int(result.loc[result["target"] == 1, "user_index"].nunique()),
    }


def train_binary(
    *, train: pd.DataFrame, validation: pd.DataFrame | None, output_path: Path,
    rounds: int | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    started = time.perf_counter()
    train_set = lgb.Dataset(
        train[MODEL_FEATURES], label=train["target"],
        categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False,
    )
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "verbosity": -1,
        "seed": 20260903,
        "feature_fraction_seed": 20260903,
        "bagging_seed": 20260903,
        "num_threads": 8,
    }
    callbacks: list[Any] = [lgb.log_evaluation(period=0)]
    valid_sets = None
    valid_names = None
    num_round = int(rounds or 300)
    if validation is not None:
        valid_set = lgb.Dataset(
            validation[MODEL_FEATURES], label=validation["target"],
            categorical_feature=CATEGORICAL_FEATURES, reference=train_set,
        )
        valid_sets = [valid_set]
        valid_names = ["inner_validation"]
        callbacks.append(lgb.early_stopping(30, verbose=False))
    model = lgb.train(
        params, train_set, num_boost_round=num_round, valid_sets=valid_sets,
        valid_names=valid_names, callbacks=callbacks,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    importance = sorted(
        (
            {"feature": name, "gain": float(gain), "split": int(split)}
            for name, gain, split in zip(
                MODEL_FEATURES,
                model.feature_importance(importance_type="gain"),
                model.feature_importance(importance_type="split"),
            )
        ), key=lambda row: (-row["gain"], row["feature"]),
    )
    return model, {
        "best_iteration": int(model.best_iteration or num_round),
        "train_rows": len(train),
        "train_positive_rows": int(train["target"].sum()),
        "validation_rows": len(validation) if validation is not None else 0,
        "validation_positive_rows": int(validation["target"].sum()) if validation is not None else 0,
        "top_feature_importance": importance[:25],
        "elapsed_seconds": time.perf_counter() - started,
        "model": file_identity(output_path),
    }


def prepare_inner_warm_dbs(
    *, source_root: Path, artifact_dir: Path,
) -> dict[str, Any]:
    metrics_path = source_root / "reports" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = M2Config(**metrics["config"])
    features = variant_feature_sets()["anchor"]
    output: dict[str, Any] = {}
    for window, protocol in ROLLING_PROTOCOL.items():
        cutoff = protocol["inner_validation"]
        root = artifact_dir / window
        root.mkdir(parents=True, exist_ok=True)
        db_path = root / "evaluation.duckdb"
        prediction_path = root / "predictions.parquet"
        if db_path.exists() and prediction_path.exists():
            output[window] = {
                "mode": "reused",
                "cutoff": cutoff,
                "evaluation_db": file_identity(db_path),
                "predictions": file_identity(prediction_path),
            }
            continue
        model_path = source_root / "artifacts" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / window / "lightgbm-inner-anchor.txt"
        maps_path = model_path.parent / "inner-category-maps.json"
        model = lgb.Booster(model_file=str(model_path))
        evidence = score_models(
            dataset_path=Path(metrics["feature_cache"][cutoff]["artifact"]["path"]),
            models={"anchor": (model, features, _load_category_maps(maps_path))},
            evaluation_db=db_path, prediction_path=prediction_path, config=config,
        )
        output[window] = {
            "mode": "generated_from_frozen_m3_3_inner_anchor",
            "cutoff": cutoff,
            "scoring": evidence,
            "model": file_identity(model_path),
            "category_maps": file_identity(maps_path),
        }
    return output


def _raw_features_for_existing(
    *, arrays: dict[str, np.ndarray], users: list[str], seeds: dict[str, list[tuple[int, int, float]]],
    fashionclip_root: Path, image_rows: np.ndarray, device: torch.device,
) -> None:
    source = np.load(fashionclip_root / "embeddings.float16.npy", mmap_mode="r")
    unique_users = np.unique(arrays["user_index"])
    for name in (
        "raw_best_recency_weighted_cosine", "raw_best_cosine", "raw_top3_mean_cosine",
        "raw_top3_mean_recency_weighted_cosine", "raw_supporting_seed_count",
    ):
        dtype = np.uint8 if name.endswith("count") else np.float32
        arrays[name] = np.zeros(len(arrays["user_index"]), dtype=dtype)
    for start in range(0, len(unique_users), 64):
        batch_users = unique_users[start:start + 64]
        left = np.searchsorted(arrays["user_index"], batch_users, side="left")
        right = np.searchsorted(arrays["user_index"], batch_users, side="right")
        row_groups = [np.arange(begin, end, dtype=np.int64) for begin, end in zip(left, right)]
        if any(len(group) != PRIMARY_K for group in row_groups):
            raise RuntimeError("M5 outer candidate rows are not exact Top50 per active user")
        candidate_rows = np.stack([arrays["catalog_row"][group] for group in row_groups])
        candidate_image_rows = image_rows[candidate_rows]
        candidate_np = np.zeros((*candidate_rows.shape, 512), dtype=np.float32)
        present = candidate_image_rows >= 0
        candidate_np[present] = np.asarray(source[candidate_image_rows[present]], dtype=np.float32)
        candidate_np /= np.maximum(np.linalg.norm(candidate_np, axis=2, keepdims=True), 1e-12)
        seed_np = np.zeros((len(batch_users), 5, 512), dtype=np.float32)
        weights = np.zeros((len(batch_users), 5), dtype=np.float32)
        valid = np.zeros((len(batch_users), 5), dtype=bool)
        for local, user_index in enumerate(batch_users):
            user = users[int(user_index)]
            for slot, (catalog_row, _rank, weight) in enumerate(seeds.get(user, [])[:5]):
                image_row = int(image_rows[catalog_row])
                if image_row >= 0:
                    seed_np[local, slot] = np.asarray(source[image_row], dtype=np.float32)
                    norm = np.linalg.norm(seed_np[local, slot])
                    seed_np[local, slot] /= max(float(norm), 1e-12)
                weights[local, slot] = weight
                valid[local, slot] = True
        seed_t = torch.from_numpy(seed_np).to(device)
        candidate_t = torch.from_numpy(candidate_np).to(device)
        sim = torch.einsum("bsd,bkd->bsk", seed_t, candidate_t)
        valid_t = torch.from_numpy(valid).to(device)
        sim = sim.masked_fill(~valid_t[:, :, None], -torch.inf)
        weight_t = torch.from_numpy(weights).to(device)
        weighted = torch.where(valid_t[:, :, None], sim * weight_t[:, :, None], torch.full_like(sim, -torch.inf))
        count = torch.clamp(valid_t.sum(dim=1), max=3).float()[:, None]
        top_sim = torch.topk(sim, k=3, dim=1).values
        top_weighted = torch.topk(weighted, k=3, dim=1).values
        values = {
            "raw_best_cosine": sim.max(dim=1).values.cpu().numpy(),
            "raw_best_recency_weighted_cosine": weighted.max(dim=1).values.cpu().numpy(),
            "raw_top3_mean_cosine": torch.where(torch.isfinite(top_sim), top_sim, 0).sum(dim=1).div(count).cpu().numpy(),
            "raw_top3_mean_recency_weighted_cosine": torch.where(
                torch.isfinite(top_weighted), top_weighted, 0
            ).sum(dim=1).div(count).cpu().numpy(),
            "raw_supporting_seed_count": (sim > 0).sum(dim=1).cpu().numpy().astype(np.uint8),
        }
        for local, group in enumerate(row_groups):
            for name, value in values.items():
                arrays[name][group] = value[local]
    arrays["student_raw_weighted_gap"] = (
        arrays["student_best_recency_weighted_cosine"] - arrays["raw_best_recency_weighted_cosine"]
    ).astype(np.float32)


def build_outer_validation_dataset(
    *, source_root: Path, m4_artifact_dir: Path, artifact_dir: Path,
    window: str, device: torch.device,
) -> dict[str, Any]:
    root = artifact_dir / window
    root.mkdir(parents=True, exist_ok=False)
    cutoff = ROLLING_PROTOCOL[window]["outer_validation"]
    warm_db = source_root / "artifacts" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / window / "evaluation.duckdb"
    con = duckdb.connect(str(warm_db), read_only=True)
    try:
        users = [str(row[0]) for row in con.execute("SELECT customer_id FROM m29_eval_users ORDER BY customer_id").fetchall()]
    finally:
        con.close()
    static_dir = m4_artifact_dir / "student-v1" / "static_catalog"
    catalog = pd.read_csv(static_dir / "catalog_items.csv", dtype={"article_id": str})
    catalog_items = catalog["article_id"].tolist()
    article_to_row = {item: row for row, item in enumerate(catalog_items)}
    categories = np.load(static_dir / "catalog_categories.int32.npy", mmap_mode="r")
    image_rows = np.load(static_dir / "catalog_image_rows.int32.npy", mmap_mode="r")
    source_path = m4_artifact_dir / "retrieval-v1" / window / "multimodal" / "retrieval_top100.npz"
    with np.load(source_path) as source:
        mask = source["rank"] <= PRIMARY_K
        arrays = {
            "user_index": np.asarray(source["user_index"][mask], dtype=np.int32),
            "catalog_row": np.asarray(source["catalog_row"][mask], dtype=np.int32),
            "student_rank": np.asarray(source["rank"][mask], dtype=np.uint8),
            "student_best_recency_weighted_cosine": np.asarray(source["best_recency_weighted_cosine"][mask], dtype=np.float32),
            "student_best_cosine": np.asarray(source["best_cosine"][mask], dtype=np.float32),
            "student_top3_mean_cosine": np.asarray(source["top3_mean_cosine"][mask], dtype=np.float32),
            "student_top3_mean_recency_weighted_cosine": np.asarray(source["top3_mean_recency_weighted_cosine"][mask], dtype=np.float32),
            "student_supporting_seed_count": np.asarray(source["supporting_seed_count"][mask], dtype=np.uint8),
            "student_best_seed_rank": np.asarray(source["best_seed_rank"][mask], dtype=np.uint8),
            "student_best_seed_recency_weight": np.asarray(source["best_seed_recency_weight"][mask], dtype=np.float32),
        }
    transactions = source_root / "data" / "interim" / "audit" / "transactions.parquet"
    counts, seeds, seed_audit = _history_counts_and_seeds(
        transactions_path=transactions, cutoff=cutoff, users=users, article_to_row=article_to_row
    )
    active_users = [users[int(index)] for index in np.unique(arrays["user_index"])]
    _raw_features_for_existing(
        arrays=arrays, users=users, seeds=seeds,
        fashionclip_root=source_root / "artifacts" / "m2_4" / "fashionclip-full-v1",
        image_rows=image_rows, device=device,
    )
    context = _user_context(
        transactions_path=transactions, customers_path=source_root / "data" / "raw" / "customers.csv",
        cutoff=cutoff, users=users, catalog_items=catalog_items, categories=categories,
    )
    truth_keys = _truth_keys(
        transactions_path=transactions, cutoff=cutoff, users=users, article_to_row=article_to_row
    )
    _append_context(
        arrays=arrays, users=users, active_users=active_users, counts=counts, context=context,
        categories=categories, truth_keys=truth_keys, catalog_size=len(catalog_items),
    )
    dataset_path = root / "cold_candidates_top50.npz"
    np.savez_compressed(dataset_path, **arrays)
    users_path = root / "users.csv"
    pd.DataFrame({"customer_id": users}).to_csv(users_path, index=False)
    result = {
        "window": window,
        "cutoff": cutoff,
        "rows": len(arrays["user_index"]),
        "users": len(users),
        "active_users": len(active_users),
        "positive_rows": int(arrays["target"].sum()),
        "strict_cold_positive_rows": int(np.count_nonzero((arrays["target"] == 1) & (arrays["item_is_strict_cold"] == 1))),
        "seed_audit": seed_audit,
        "inputs": {"m4_candidates": file_identity(source_path), "warm_db": file_identity(warm_db)},
        "artifacts": {"dataset": file_identity(dataset_path), "users": file_identity(users_path)},
        "final_week": "not_run",
    }
    atomic_json(root / "manifest.json", result)
    return result


def _warm_state(
    *, evaluation_db: Path, transactions_path: Path, cutoff: str,
) -> tuple[list[str], dict[str, list[str]], dict[str, set[str]], dict[str, set[str]]]:
    con = duckdb.connect(str(evaluation_db), read_only=True)
    try:
        users = [str(row[0]) for row in con.execute("SELECT DISTINCT customer_id FROM predictions ORDER BY customer_id").fetchall()]
        ranked = con.execute(
            """
            SELECT customer_id,article_id,row_number() OVER(
                PARTITION BY customer_id ORDER BY
                CASE WHEN user_history_events_12w=0 THEN -candidate_rank ELSE score_anchor END DESC,
                candidate_rank,article_id
            ) AS final_rank FROM predictions
            """
        ).fetch_df()
    finally:
        con.close()
    warm_top12 = {
        user: group.sort_values("final_rank")["article_id"].astype(str).tolist()[:12]
        for user, group in ranked.groupby("customer_id")
    }
    warm_pool = {
        user: set(group["article_id"].astype(str)) for user, group in ranked.groupby("customer_id")
    }
    user_frame = pd.DataFrame({"customer_id": users})
    con = duckdb.connect()
    con.register("m5_eval_users", user_frame)
    tx = transactions_path.resolve().as_posix().replace("'", "''")
    try:
        truth_frame = con.execute(
            f"SELECT DISTINCT t.customer_id,t.article_id FROM read_parquet('{tx}') t "
            "JOIN m5_eval_users u USING(customer_id) "
            f"WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY"
        ).fetch_df()
    finally:
        con.unregister("m5_eval_users")
        con.close()
    truth = {user: set(group["article_id"].astype(str)) for user, group in truth_frame.groupby("customer_id")}
    return users, warm_top12, warm_pool, truth


def _average_precision(ranked: list[str], truth: set[str]) -> float:
    if not truth:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(ranked[:12], start=1):
        if item in truth:
            hits += 1
            score += hits / rank
    return score / min(len(truth), 12)


def _evaluate_admission(
    *, users: list[str], warm_top12: dict[str, list[str]], warm_pool: dict[str, set[str]],
    truth: dict[str, set[str]], cold_scores: pd.DataFrame, catalog_items: list[str],
    item_counts: np.ndarray, k: int,
) -> dict[str, Any]:
    scored_by_user: dict[str, list[str]] = {}
    for user, group in cold_scores.groupby("customer_id", sort=False):
        blocked = warm_pool.get(str(user), set())
        ranked = group.sort_values(["score", "student_rank", "article_id"], ascending=[False, True, True])
        scored_by_user[str(user)] = [item for item in ranked["article_id"].astype(str) if item not in blocked]
    count_map = {item: int(item_counts[row]) for row, item in enumerate(catalog_items)}
    metrics = {name: [] for name in ("overall", "strict_cold", "sparse_1_5", "warm_21_plus")}
    inserted = removed = inserted_positive = removed_positive = inserted_strict = inserted_sparse = 0
    for user in users:
        base = warm_top12.get(user, [])[:12]
        chosen = scored_by_user.get(user, [])[:k]
        actual = min(len(chosen), k, len(base))
        final = base if actual == 0 else base[:-actual] + chosen[:actual]
        user_truth = truth.get(user, set())
        removed_items = base[-actual:] if actual else []
        inserted += actual
        removed += actual
        inserted_positive += sum(item in user_truth for item in chosen[:actual])
        removed_positive += sum(item in user_truth for item in removed_items)
        inserted_strict += sum(item in user_truth and count_map.get(item, 0) == 0 for item in chosen[:actual])
        inserted_sparse += sum(item in user_truth and 1 <= count_map.get(item, 0) <= 5 for item in chosen[:actual])
        predicates = {
            "overall": user_truth,
            "strict_cold": {item for item in user_truth if count_map.get(item, 0) == 0},
            "sparse_1_5": {item for item in user_truth if 1 <= count_map.get(item, 0) <= 5},
            "warm_21_plus": {item for item in user_truth if count_map.get(item, 0) >= 21},
        }
        for name, relevant in predicates.items():
            if relevant:
                metrics[name].append(_average_precision(final, relevant))
    return {
        "k_admit": k,
        "segments": {
            name: {"map@12": float(np.mean(values)) if values else 0.0, "truth_users": len(values)}
            for name, values in metrics.items()
        },
        "inserted_candidate_pairs": inserted,
        "removed_candidate_pairs": removed,
        "inserted_positive_pairs": inserted_positive,
        "removed_positive_pairs": removed_positive,
        "inserted_strict_cold_positive_pairs": inserted_strict,
        "inserted_sparse_1_5_positive_pairs": inserted_sparse,
    }


def _score_frame(
    *, model: lgb.Booster, arrays: dict[str, np.ndarray], categories: np.ndarray,
    allowed_users: set[str] | None = None,
) -> pd.DataFrame:
    indices = np.arange(len(arrays["user_index"]), dtype=np.int64)
    if allowed_users is not None:
        users = arrays["_users"]
        keep_user = np.asarray([str(user) in allowed_users for user in users], dtype=bool)
        indices = indices[keep_user[arrays["user_index"]]]
    frame = _frame(arrays, indices, categories)
    scores = model.predict(frame[MODEL_FEATURES], num_iteration=model.best_iteration)
    users = arrays["_users"]
    catalog_rows = arrays["catalog_row"][indices]
    return pd.DataFrame({
        "customer_id": users[arrays["user_index"][indices]],
        "catalog_row": catalog_rows,
        "student_rank": arrays["student_rank"][indices],
        "score": scores,
    })


def _add_article_ids(frame: pd.DataFrame, catalog_items: list[str]) -> pd.DataFrame:
    output = frame.copy()
    items = np.asarray(catalog_items, dtype=object)
    output["article_id"] = items[output["catalog_row"].to_numpy(dtype=np.int64)]
    return output


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    con = duckdb.connect()
    con.register("m5_output", frame)
    escaped = temporary.resolve().as_posix().replace("'", "''")
    try:
        con.execute(
            f"COPY m5_output TO '{escaped}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)"
        )
    finally:
        con.unregister("m5_output")
        con.close()
    temporary.replace(path)


def run_m51_m52(
    *, source_root: Path, m4_artifact_dir: Path, artifact_dir: Path,
    report_dir: Path, device_name: str = "cuda",
) -> dict[str, Any]:
    started = time.perf_counter()
    m50 = json.loads((report_dir / "M5_0_metrics.json").read_text(encoding="utf-8"))
    if not m50.get("gate_passed") or m50["decision"]["selected_model_type"] != "binary":
        raise RuntimeError("M5.0 did not authorize the pre-registered binary expert")
    scale = m50["decision"]["selected_training_scale"]
    device = torch.device(device_name)
    static_dir = m4_artifact_dir / "student-v1" / "static_catalog"
    categories = np.load(static_dir / "catalog_categories.int32.npy", mmap_mode="r")
    catalog_items = pd.read_csv(static_dir / "catalog_items.csv", dtype={"article_id": str})["article_id"].tolist()
    transactions = source_root / "data" / "interim" / "audit" / "transactions.parquet"
    train_data_dir = artifact_dir / "train-data-v1"
    inner_warm = prepare_inner_warm_dbs(source_root=source_root, artifact_dir=artifact_dir / "warm-inner-v1")
    outer_data: dict[str, Any] = {}
    for window in ROLLING_PROTOCOL:
        print(f"M5.1 outer feature dataset {window}", flush=True)
        manifest_path = artifact_dir / "outer-data-v1" / window / "manifest.json"
        if manifest_path.is_file():
            outer_data[window] = json.loads(manifest_path.read_text(encoding="utf-8"))
            outer_data[window]["reuse_mode"] = "reused_after_schema-only_rerun"
        else:
            outer_data[window] = build_outer_validation_dataset(
                source_root=source_root, m4_artifact_dir=m4_artifact_dir,
                artifact_dir=artifact_dir / "outer-data-v1", window=window, device=device,
            )
    windows: dict[str, Any] = {}
    for window, protocol in ROLLING_PROTOCOL.items():
        print(f"M5.1 train/select/evaluate {window}", flush=True)
        window_dir = artifact_dir / "models-v1" / window
        inner_train, inner_train_evidence = load_training_frame(
            data_dir=train_data_dir, cutoffs=protocol["inner_train"], categories=categories, scale=scale,
        )
        inner_validation_arrays = _load_arrays(
            train_data_dir / protocol["inner_validation"] / "cold_candidates_top50.npz",
            train_data_dir / protocol["inner_validation"] / "users.csv",
        )
        inner_validation_frame = _frame(
            inner_validation_arrays,
            np.arange(len(inner_validation_arrays["user_index"]), dtype=np.int64), categories,
        )
        inner_model, inner_model_evidence = train_binary(
            train=inner_train, validation=inner_validation_frame,
            output_path=window_dir / "inner-cold-expert.txt",
        )
        inner_db = artifact_dir / "warm-inner-v1" / window / "evaluation.duckdb"
        inner_users, inner_top12, inner_pool, inner_truth = _warm_state(
            evaluation_db=inner_db, transactions_path=transactions, cutoff=protocol["inner_validation"],
        )
        inner_scores = _add_article_ids(
            _score_frame(
                model=inner_model, arrays=inner_validation_arrays, categories=categories,
                allowed_users=set(inner_users),
            ), catalog_items,
        )
        article_to_row = {item: row for row, item in enumerate(catalog_items)}
        inner_counts, _, _ = _history_counts_and_seeds(
            transactions_path=transactions, cutoff=protocol["inner_validation"],
            users=inner_users, article_to_row=article_to_row,
        )
        inner_results = {
            str(k): _evaluate_admission(
                users=inner_users, warm_top12=inner_top12, warm_pool=inner_pool,
                truth=inner_truth, cold_scores=inner_scores, catalog_items=catalog_items,
                item_counts=inner_counts, k=k,
            ) for k in K_ADMIT
        }
        best_map = max(value["segments"]["overall"]["map@12"] for value in inner_results.values())
        selected_k = min(
            int(key) for key, value in inner_results.items()
            if best_map - value["segments"]["overall"]["map@12"] <= MAP_TOLERANCE
        )
        rounds = inner_model_evidence["best_iteration"]
        del inner_model, inner_train, inner_validation_frame
        gc.collect()
        outer_train, outer_train_evidence = load_training_frame(
            data_dir=train_data_dir, cutoffs=protocol["outer_train"], categories=categories, scale=scale,
        )
        outer_model, outer_model_evidence = train_binary(
            train=outer_train, validation=None,
            output_path=window_dir / "outer-cold-expert.txt", rounds=rounds,
        )
        outer_root = artifact_dir / "outer-data-v1" / window
        outer_arrays = _load_arrays(
            outer_root / "cold_candidates_top50.npz", outer_root / "users.csv",
        )
        outer_scores = _add_article_ids(
            _score_frame(model=outer_model, arrays=outer_arrays, categories=categories), catalog_items,
        )
        warm_db = source_root / "artifacts" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / window / "evaluation.duckdb"
        outer_users, outer_top12, outer_pool, outer_truth = _warm_state(
            evaluation_db=warm_db, transactions_path=transactions, cutoff=protocol["outer_validation"],
        )
        outer_counts, _, outer_seed_audit = _history_counts_and_seeds(
            transactions_path=transactions, cutoff=protocol["outer_validation"],
            users=outer_users, article_to_row=article_to_row,
        )
        outer_all_k = {
            str(k): _evaluate_admission(
                users=outer_users, warm_top12=outer_top12, warm_pool=outer_pool,
                truth=outer_truth, cold_scores=outer_scores, catalog_items=catalog_items,
                item_counts=outer_counts, k=k,
            ) for k in K_ADMIT
        }
        selected = outer_all_k[str(selected_k)]
        if abs(outer_all_k["0"]["segments"]["overall"]["map@12"] - WARM_MAP[window]) > 1e-12:
            raise RuntimeError(f"M5 Warm-v1 anchor reproduction failed: {window}")
        score_path = window_dir / "outer-cold-scores.parquet"
        _write_parquet(outer_scores, score_path)
        windows[window] = {
            "protocol": protocol,
            "selected_k_from_inner": selected_k,
            "inner": {
                "training_data": inner_train_evidence,
                "model": inner_model_evidence,
                "admission_results": inner_results,
                "warm_evidence": inner_warm[window],
            },
            "outer": {
                "training_data": outer_train_evidence,
                "model": outer_model_evidence,
                "all_k_diagnostic": outer_all_k,
                "selected_result": selected,
                "delta_vs_warm_map@12": selected["segments"]["overall"]["map@12"] - WARM_MAP[window],
                "seed_audit": outer_seed_audit,
                "scores": file_identity(score_path),
            },
        }
        del inner_validation_arrays, outer_arrays, outer_model, outer_train, outer_scores
        gc.collect()
    deltas = {window: row["outer"]["delta_vs_warm_map@12"] for window, row in windows.items()}
    selected_results = [row["outer"]["selected_result"] for row in windows.values()]
    warm_base_mean = float(np.mean([row["outer"]["all_k_diagnostic"]["0"]["segments"]["warm_21_plus"]["map@12"] for row in windows.values()]))
    warm_selected_mean = float(np.mean([row["segments"]["warm_21_plus"]["map@12"] for row in selected_results]))
    inserted_positive = sum(row["inserted_positive_pairs"] for row in selected_results)
    removed_positive = sum(row["removed_positive_pairs"] for row in selected_results)
    gates = {
        "mean_map_strictly_above_warm": float(np.mean(list(deltas.values()))) > MAP_TOLERANCE,
        "at_least_3_of_4_windows_non_degrading": sum(value >= -MAP_TOLERANCE for value in deltas.values()) >= 3,
        "worst_window_within_tolerance": min(deltas.values()) >= -MAX_SINGLE_WINDOW_REGRESSION,
        "strict_cold_truth_enters_top12": sum(row["inserted_strict_cold_positive_pairs"] for row in selected_results) > 0,
        "cold_and_sparse_map_positive": (
            float(np.mean([row["segments"]["strict_cold"]["map@12"] for row in selected_results])) > 0
            and float(np.mean([row["segments"]["sparse_1_5"]["map@12"] for row in selected_results])) > 0
        ),
        "warm_preserved_or_pair_gain_dominates": (
            warm_selected_mean >= warm_base_mean - MAP_TOLERANCE or inserted_positive > removed_positive
        ),
        "inserted_positive_pairs_exceed_removed": inserted_positive > removed_positive,
        "cutoff_and_anchor_audits_pass": all(
            row["outer"]["seed_audit"]["cutoff_safe"] for row in windows.values()
        ),
    }
    result = {
        "schema_version": "m5.1-m5.2-cold-expert-admission-v1",
        "stage": "M5.1-M5.2",
        "status": "measured",
        "run_id": M5_RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "warm_baseline": "M3.3 base-300 anchor__inactive_rrf, exact frozen order",
            "cold_candidates": "M4 multimodal Student Top50, deduplicated against the full Warm-v1 candidate pool at admission",
            "model": "single pooled binary LightGBM selected by M5.0; no LambdaRank/MLP comparison",
            "negative_sampling": "all positives plus at most 100 rank-stratified unobserved candidates per positive; selection key is stable hash(cutoff,user_index,catalog_row,student_rank)",
            "k_admit": list(K_ADMIT),
            "placement": "K=1 replaces rank12; K=3 replaces ranks10-12; inner MAP tie chooses smaller K",
            "map_tolerance": MAP_TOLERANCE,
            "max_single_window_regression": MAX_SINGLE_WINDOW_REGRESSION,
        },
        "m50_decision": m50["decision"],
        "outer_validation_data": outer_data,
        "windows": windows,
        "summary": {
            "window_deltas_vs_warm_map@12": deltas,
            "mean_delta_vs_warm_map@12": float(np.mean(list(deltas.values()))),
            "selected_k_by_window": {window: row["selected_k_from_inner"] for window, row in windows.items()},
            "inserted_positive_pairs": inserted_positive,
            "removed_positive_pairs": removed_positive,
            "inserted_strict_cold_positive_pairs": sum(row["inserted_strict_cold_positive_pairs"] for row in selected_results),
            "inserted_sparse_1_5_positive_pairs": sum(row["inserted_sparse_1_5_positive_pairs"] for row in selected_results),
            "warm_21_plus_base_mean_map@12": warm_base_mean,
            "warm_21_plus_selected_mean_map@12": warm_selected_mean,
            "gates": gates,
            "gate_passed": all(gates.values()),
            "next_stage": "M6_allowed" if all(gates.values()) else "stop_cold_admission_keep_M4_retrieval_evidence",
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": _peak_working_set_bytes(),
        },
        "final_week": "not_run",
    }
    atomic_json(report_dir / "M5_2_metrics.json", result)
    return result
