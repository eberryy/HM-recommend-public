from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch

from .m3 import _peak_working_set_bytes
from .m33 import ROLLING_PROTOCOL
from .m4_contract import STATIC_FIELDS, atomic_json, file_identity, stable_u64
from .m4_retrieval import (
    HISTORY_WEEKS,
    RECENCY_HALF_LIFE_DAYS,
    SEED_K,
    _aligned_raw_embeddings,
    _history_counts_and_seeds,
)
from .m4_student import CatalogAccessor, _load_relations, encode_catalog, train_encoder


M5_RUN_ID = "m5-v1-cold-expert-admission"
PRIMARY_K = 50
AFFINITY_FIELDS = (
    "product_type_no",
    "garment_group_no",
    "department_no",
    "index_group_no",
    "perceived_colour_master_id",
)
SIMILARITY_FEATURES = (
    "student_rank",
    "student_best_recency_weighted_cosine",
    "student_best_cosine",
    "student_top3_mean_cosine",
    "student_top3_mean_recency_weighted_cosine",
    "student_supporting_seed_count",
    "student_best_seed_rank",
    "student_best_seed_recency_weight",
    "raw_best_recency_weighted_cosine",
    "raw_best_cosine",
    "raw_top3_mean_cosine",
    "raw_top3_mean_recency_weighted_cosine",
    "raw_supporting_seed_count",
    "student_raw_weighted_gap",
)


def _active_label_users(transactions_path: Path, cutoff: str) -> list[str]:
    con = duckdb.connect()
    tx = transactions_path.resolve().as_posix().replace("'", "''")
    try:
        rows = con.execute(
            f"SELECT DISTINCT customer_id FROM read_parquet('{tx}') "
            f"WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY "
            "ORDER BY customer_id"
        ).fetchall()
    finally:
        con.close()
    return [str(row[0]) for row in rows]


def _truth_keys(
    *, transactions_path: Path, cutoff: str, users: list[str], article_to_row: dict[str, int],
) -> np.ndarray:
    con = duckdb.connect()
    con.register("m5_users", pd.DataFrame({"customer_id": users, "user_index": np.arange(len(users), dtype=np.int32)}))
    tx = transactions_path.resolve().as_posix().replace("'", "''")
    try:
        frame = con.execute(
            f"SELECT DISTINCT u.user_index,t.article_id FROM read_parquet('{tx}') t "
            "JOIN m5_users u USING(customer_id) "
            f"WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY"
        ).fetch_df()
    finally:
        con.unregister("m5_users")
        con.close()
    rows = frame["article_id"].astype(str).map(article_to_row)
    valid = rows.notna().to_numpy()
    return (
        frame.loc[valid, "user_index"].to_numpy(dtype=np.int64) * len(article_to_row)
        + rows.loc[valid].to_numpy(dtype=np.int64)
    )


def _user_context(
    *, transactions_path: Path, customers_path: Path, cutoff: str, users: list[str],
    catalog_items: list[str], categories: np.ndarray,
) -> dict[str, np.ndarray]:
    user_frame = pd.DataFrame({"customer_id": users, "user_index": np.arange(len(users), dtype=np.int32)})
    article_frame = pd.DataFrame({"article_id": catalog_items})
    for column, field in enumerate(STATIC_FIELDS):
        article_frame[field] = categories[:, column]
    con = duckdb.connect()
    con.register("m5_users", user_frame)
    con.register("m5_articles", article_frame)
    tx = transactions_path.resolve().as_posix().replace("'", "''")
    try:
        history = con.execute(
            f"""
            SELECT u.user_index,count(*) AS history_events,
                   count(DISTINCT t.article_id) AS history_distinct_items,
                   date_diff('day',max(t_dat),DATE '{cutoff}') AS days_since_last_event
            FROM read_parquet('{tx}') t JOIN m5_users u USING(customer_id)
            WHERE t_dat>=DATE '{cutoff}'-INTERVAL {HISTORY_WEEKS} WEEK AND t_dat<DATE '{cutoff}'
            GROUP BY u.user_index ORDER BY u.user_index
            """
        ).fetch_df()
        affinity_frames: dict[str, pd.DataFrame] = {}
        for field in AFFINITY_FIELDS:
            affinity_frames[field] = con.execute(
                f"""
                SELECT u.user_index,a.{field} AS category_value,count(*) AS matched_events
                FROM read_parquet('{tx}') t JOIN m5_users u USING(customer_id)
                JOIN m5_articles a USING(article_id)
                WHERE t_dat>=DATE '{cutoff}'-INTERVAL {HISTORY_WEEKS} WEEK AND t_dat<DATE '{cutoff}'
                GROUP BY u.user_index,a.{field}
                """
            ).fetch_df()
    finally:
        con.unregister("m5_articles")
        con.unregister("m5_users")
        con.close()
    n = len(users)
    events = np.zeros(n, dtype=np.int32)
    distinct = np.zeros(n, dtype=np.int32)
    recency = np.full(n, 10_000, dtype=np.int16)
    indices = history["user_index"].to_numpy(dtype=np.int64)
    events[indices] = history["history_events"].to_numpy(dtype=np.int32)
    distinct[indices] = history["history_distinct_items"].to_numpy(dtype=np.int32)
    recency[indices] = history["days_since_last_event"].to_numpy(dtype=np.int16)
    customers = pd.read_csv(customers_path, usecols=["customer_id", "age"])
    age_map = dict(zip(customers["customer_id"].astype(str), customers["age"]))
    age = np.asarray([age_map.get(user, np.nan) for user in users], dtype=np.float32)
    age = np.nan_to_num(age, nan=-1.0)
    result: dict[str, np.ndarray] = {
        "user_history_events_12w": events,
        "user_history_distinct_items_12w": distinct,
        "user_days_since_last_event": recency,
        "user_age": age,
    }
    for field, frame in affinity_frames.items():
        cardinality = int(categories[:, STATIC_FIELDS.index(field)].max()) + 1
        keys = frame["user_index"].to_numpy(dtype=np.int64) * cardinality + frame["category_value"].to_numpy(dtype=np.int64)
        order = np.argsort(keys)
        result[f"_{field}_keys"] = keys[order]
        result[f"_{field}_shares"] = (
            frame["matched_events"].to_numpy(dtype=np.float32)[order]
            / np.maximum(events[frame["user_index"].to_numpy(dtype=np.int64)[order]], 1)
        )
        result[f"_{field}_cardinality"] = np.asarray([cardinality], dtype=np.int64)
    return result


def _lookup_affinity(
    *, user_indices: np.ndarray, candidate_rows: np.ndarray, categories: np.ndarray,
    context: dict[str, np.ndarray], field: str,
) -> np.ndarray:
    column = STATIC_FIELDS.index(field)
    cardinality = int(context[f"_{field}_cardinality"][0])
    query = user_indices.astype(np.int64) * cardinality + categories[candidate_rows, column].astype(np.int64)
    keys = context[f"_{field}_keys"]
    shares = context[f"_{field}_shares"]
    positions = np.searchsorted(keys, query)
    valid = positions < len(keys)
    output = np.zeros(len(query), dtype=np.float32)
    matched = valid.copy()
    matched[valid] = keys[positions[valid]] == query[valid]
    output[matched] = shares[positions[matched]]
    return output


def _allocate(rows: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "user_index": np.empty(rows, dtype=np.int32),
        "catalog_row": np.empty(rows, dtype=np.int32),
        "student_rank": np.empty(rows, dtype=np.uint8),
    }
    for name in SIMILARITY_FEATURES[1:]:
        dtype = np.uint8 if name in {"student_supporting_seed_count", "student_best_seed_rank", "raw_supporting_seed_count"} else np.float32
        arrays[name] = np.empty(rows, dtype=dtype)
    return arrays


def retrieve_feature_rows(
    *, embedding_path: Path, fashionclip_root: Path, image_rows: np.ndarray,
    candidate_rows: np.ndarray, users: list[str], seeds: dict[str, list[tuple[int, int, float]]],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, Any]]:
    started = time.perf_counter()
    active_users = [user for user in users if seeds.get(user)]
    user_to_original = {user: index for index, user in enumerate(users)}
    student = np.load(embedding_path, mmap_mode="r")
    student_candidates = torch.from_numpy(np.asarray(student[candidate_rows], dtype=np.float32)).to(device)
    raw_candidates = torch.from_numpy(_aligned_raw_embeddings(
        fashionclip_root=fashionclip_root, image_rows=image_rows, selected_rows=candidate_rows
    )).to(device)
    arrays = _allocate(len(active_users) * PRIMARY_K)
    cursor = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_start in range(0, len(active_users), 32):
            batch_users = active_users[batch_start:batch_start + 32]
            b = len(batch_users)
            seed_rows = np.full((b, SEED_K), -1, dtype=np.int64)
            seed_weights = np.zeros((b, SEED_K), dtype=np.float32)
            seed_ranks = np.zeros((b, SEED_K), dtype=np.uint8)
            for local, user in enumerate(batch_users):
                for slot, (catalog_row, seed_rank, weight) in enumerate(seeds[user][:SEED_K]):
                    seed_rows[local, slot] = catalog_row
                    seed_weights[local, slot] = weight
                    seed_ranks[local, slot] = seed_rank
            flat = seed_rows.reshape(-1)
            valid = flat >= 0
            student_seed_np = np.zeros((len(flat), 128), dtype=np.float32)
            student_seed_np[valid] = np.asarray(student[flat[valid]], dtype=np.float32)
            raw_seed_np = np.zeros((len(flat), 512), dtype=np.float32)
            if np.any(valid):
                raw_seed_np[valid] = _aligned_raw_embeddings(
                    fashionclip_root=fashionclip_root, image_rows=image_rows, selected_rows=flat[valid]
                )
            student_seeds = torch.from_numpy(student_seed_np.reshape(b, SEED_K, 128)).to(device)
            raw_seeds = torch.from_numpy(raw_seed_np.reshape(b, SEED_K, 512)).to(device)
            valid_t = torch.from_numpy(seed_rows >= 0).to(device)
            weights_t = torch.from_numpy(seed_weights).to(device)
            student_sim = torch.einsum("bsd,cd->bsc", student_seeds, student_candidates)
            student_sim = student_sim.masked_fill(~valid_t[:, :, None], -torch.inf)
            student_weighted = torch.where(
                valid_t[:, :, None], student_sim * weights_t[:, :, None], torch.full_like(student_sim, -torch.inf)
            )
            values, local_rows = torch.topk(student_weighted.max(dim=1).values, k=PRIMARY_K, dim=1, sorted=True)
            selected_catalog = candidate_rows[local_rows.cpu().numpy()]
            selected_raw = raw_candidates[local_rows]
            raw_sim = torch.einsum("bsd,bkd->bsk", raw_seeds, selected_raw)
            raw_sim = raw_sim.masked_fill(~valid_t[:, :, None], -torch.inf)
            raw_weighted = torch.where(
                valid_t[:, :, None], raw_sim * weights_t[:, :, None], torch.full_like(raw_sim, -torch.inf)
            )
            student_selected = torch.gather(student_sim, 2, local_rows[:, None, :].expand(-1, SEED_K, -1))
            student_selected_weighted = torch.gather(
                student_weighted, 2, local_rows[:, None, :].expand(-1, SEED_K, -1)
            )
            length = b * PRIMARY_K
            sl = slice(cursor, cursor + length)
            arrays["user_index"][sl] = np.repeat(
                np.asarray([user_to_original[user] for user in batch_users], dtype=np.int32), PRIMARY_K
            )
            arrays["catalog_row"][sl] = selected_catalog.reshape(-1)
            arrays["student_rank"][sl] = np.tile(np.arange(1, PRIMARY_K + 1, dtype=np.uint8), b)
            for prefix, sim, weighted in (
                ("student", student_selected, student_selected_weighted),
                ("raw", raw_sim, raw_weighted),
            ):
                valid_count = torch.from_numpy(np.sum(seed_rows >= 0, axis=1)).to(device)
                top_count = min(3, SEED_K)
                top3_sim = torch.topk(sim, k=top_count, dim=1).values
                top3_weighted = torch.topk(weighted, k=top_count, dim=1).values
                finite_sim = torch.where(torch.isfinite(top3_sim), top3_sim, torch.zeros_like(top3_sim))
                finite_weighted = torch.where(
                    torch.isfinite(top3_weighted), top3_weighted, torch.zeros_like(top3_weighted)
                )
                divisor = torch.clamp(valid_count, max=top_count).float()[:, None]
                arrays[f"{prefix}_best_cosine"][sl] = sim.max(dim=1).values.cpu().numpy().reshape(-1)
                arrays[f"{prefix}_best_recency_weighted_cosine"][sl] = weighted.max(dim=1).values.cpu().numpy().reshape(-1)
                arrays[f"{prefix}_top3_mean_cosine"][sl] = (finite_sim.sum(dim=1) / divisor).cpu().numpy().reshape(-1)
                arrays[f"{prefix}_top3_mean_recency_weighted_cosine"][sl] = (
                    finite_weighted.sum(dim=1) / divisor
                ).cpu().numpy().reshape(-1)
                arrays[f"{prefix}_supporting_seed_count"][sl] = (
                    (sim > 0).sum(dim=1).cpu().numpy().astype(np.uint8).reshape(-1)
                )
            best_seed = student_selected_weighted.argmax(dim=1).cpu().numpy()
            arrays["student_best_seed_rank"][sl] = np.take_along_axis(
                seed_ranks, best_seed, axis=1
            ).reshape(-1)
            arrays["student_best_seed_recency_weight"][sl] = np.take_along_axis(
                seed_weights, best_seed, axis=1
            ).reshape(-1)
            arrays["student_raw_weighted_gap"][sl] = (
                arrays["student_best_recency_weighted_cosine"][sl]
                - arrays["raw_best_recency_weighted_cosine"][sl]
            )
            cursor += length
            if (batch_start // 32 + 1) % 100 == 0:
                print(f"M5 retrieval {min(batch_start + 32, len(active_users))}/{len(active_users)}", flush=True)
    if cursor != len(arrays["user_index"]):
        raise RuntimeError("M5 candidate array conservation failed")
    del student_candidates, raw_candidates
    if device.type == "cuda":
        peak_vram = int(torch.cuda.max_memory_allocated())
        torch.cuda.empty_cache()
    else:
        peak_vram = 0
    return arrays, active_users, {
        "label_week_users": len(users),
        "users_with_seed": len(active_users),
        "candidate_rows": cursor,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_bytes": peak_vram,
    }


def _append_context(
    *, arrays: dict[str, np.ndarray], users: list[str], active_users: list[str],
    counts: np.ndarray, context: dict[str, np.ndarray], categories: np.ndarray,
    truth_keys: np.ndarray, catalog_size: int,
) -> None:
    original_index = {user: index for index, user in enumerate(users)}
    active_original = np.asarray([original_index[user] for user in active_users], dtype=np.int32)
    remap = np.full(len(users), -1, dtype=np.int32)
    remap[active_original] = np.arange(len(active_users), dtype=np.int32)
    arrays["active_user_index"] = remap[arrays["user_index"]]
    keys = arrays["user_index"].astype(np.int64) * catalog_size + arrays["catalog_row"].astype(np.int64)
    arrays["target"] = np.isin(keys, truth_keys, assume_unique=False).astype(np.uint8)
    arrays["item_events_before_cutoff"] = np.clip(counts[arrays["catalog_row"]], 0, 32767).astype(np.int16)
    arrays["item_is_strict_cold"] = (arrays["item_events_before_cutoff"] == 0).astype(np.uint8)
    for name in ("user_history_events_12w", "user_history_distinct_items_12w", "user_days_since_last_event", "user_age"):
        arrays[name] = context[name][arrays["user_index"]]
    for field in AFFINITY_FIELDS:
        arrays[f"user_{field}_share_12w"] = _lookup_affinity(
            user_indices=arrays["user_index"], candidate_rows=arrays["catalog_row"],
            categories=categories, context=context, field=field,
        )


def _scale_stats(arrays: dict[str, np.ndarray], users: list[str]) -> dict[str, Any]:
    hashes = np.asarray([stable_u64(user) % 1_000_000 for user in users], dtype=np.uint64)
    output: dict[str, Any] = {}
    for label, threshold in (("10pct", 100_000), ("30pct", 300_000), ("full", 1_000_000)):
        row_mask = hashes[arrays["user_index"]] < threshold
        positives = row_mask & (arrays["target"] == 1)
        positive_user_indices, per_group = np.unique(arrays["user_index"][positives], return_counts=True)
        output[label] = {
            "eligible_users": int(np.count_nonzero(hashes < threshold)),
            "candidate_rows": int(np.count_nonzero(row_mask)),
            "positive_pairs": int(np.count_nonzero(positives)),
            "strict_cold_positive_pairs": int(np.count_nonzero(positives & (arrays["item_is_strict_cold"] == 1))),
            "sparse_1_5_positive_pairs": int(np.count_nonzero(positives & (arrays["item_is_strict_cold"] == 0))),
            "positive_groups": int(len(positive_user_indices)),
            "positive_density": float(np.count_nonzero(positives) / max(np.count_nonzero(row_mask), 1)),
            "positive_per_positive_group_median": float(np.median(per_group)) if len(per_group) else 0.0,
        }
    return output


def build_cutoff_dataset(
    *, source_root: Path, teacher_dir: Path, static_catalog_dir: Path, artifact_dir: Path,
    cutoff: str, lambda_pairwise: float, device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    cutoff_dir = artifact_dir / cutoff
    cutoff_dir.mkdir(parents=True, exist_ok=False)
    transactions = source_root / "data" / "interim" / "audit" / "transactions.parquet"
    customers = source_root / "data" / "raw" / "customers.csv"
    fashionclip = source_root / "artifacts" / "m2_4" / "fashionclip-full-v1"
    catalog_frame = pd.read_csv(static_catalog_dir / "catalog_items.csv", dtype={"article_id": str})
    catalog_items = catalog_frame["article_id"].tolist()
    article_to_row = {item: row for row, item in enumerate(catalog_items)}
    categories = np.load(static_catalog_dir / "catalog_categories.int32.npy", mmap_mode="r")
    image_rows = np.load(static_catalog_dir / "catalog_image_rows.int32.npy", mmap_mode="r")
    accessor = CatalogAccessor(fashionclip_root=fashionclip, catalog_dir=static_catalog_dir, device=device)
    relation_path = teacher_dir / cutoff / "teacher_relations.npz"
    relation = _load_relations([relation_path])
    model, training = train_encoder(
        variant="multimodal", relation=relation, catalog_items=catalog_items, accessor=accessor,
        cardinalities=[int(value) for value in (np.max(categories, axis=0) + 1)],
        lambda_pairwise=lambda_pairwise, output_dir=cutoff_dir / "student", max_epochs=3,
    )
    embedding_path = cutoff_dir / "student" / "catalog_embeddings.float16.npy"
    encoding = encode_catalog(
        model=model, accessor=accessor, rows=len(catalog_items), output_path=embedding_path
    )
    del model, relation
    users = _active_label_users(transactions, cutoff)
    counts, seeds, seed_audit = _history_counts_and_seeds(
        transactions_path=transactions, cutoff=cutoff, users=users, article_to_row=article_to_row
    )
    if not seed_audit["cutoff_safe"]:
        raise RuntimeError(f"M5 history cutoff audit failed: {cutoff}")
    candidate_rows = np.flatnonzero(counts <= 5).astype(np.int32)
    arrays, active_users, retrieval = retrieve_feature_rows(
        embedding_path=embedding_path, fashionclip_root=fashionclip, image_rows=image_rows,
        candidate_rows=candidate_rows, users=users, seeds=seeds, device=device,
    )
    context = _user_context(
        transactions_path=transactions, customers_path=customers, cutoff=cutoff, users=users,
        catalog_items=catalog_items, categories=categories,
    )
    truth_keys = _truth_keys(
        transactions_path=transactions, cutoff=cutoff, users=users, article_to_row=article_to_row
    )
    _append_context(
        arrays=arrays, users=users, active_users=active_users, counts=counts, context=context,
        categories=categories, truth_keys=truth_keys, catalog_size=len(catalog_items),
    )
    if len(arrays["user_index"]) != len(active_users) * PRIMARY_K:
        raise RuntimeError("M5 Top50 budget invariant failed")
    if len(np.unique(arrays["user_index"].astype(np.int64) * len(catalog_items) + arrays["catalog_row"])) != len(arrays["user_index"]):
        raise RuntimeError("M5 duplicate user-candidate rows")
    dataset_path = cutoff_dir / "cold_candidates_top50.npz"
    np.savez_compressed(dataset_path, **arrays)
    users_path = cutoff_dir / "users.csv"
    pd.DataFrame({"customer_id": users}).to_csv(users_path, index=False)
    result = {
        "cutoff": cutoff,
        "status": "completed",
        "contract": {
            "student_teacher_relation": "same-cutoff relation built only from transactions before cutoff",
            "user_denominator": "distinct next-week purchasing users; retrieval requires at least one 12-week seed",
            "candidate_universe": "item_events_before_cutoff <= 5",
            "candidate_budget": PRIMARY_K,
            "label_window": f"[{cutoff}, {cutoff}+7d)",
        },
        "training": training,
        "encoding": encoding,
        "seed_audit": seed_audit,
        "retrieval": retrieval,
        "candidate_universe": {
            "strict_cold_items": int(np.count_nonzero(counts[candidate_rows] == 0)),
            "sparse_1_5_items": int(np.count_nonzero((counts[candidate_rows] >= 1) & (counts[candidate_rows] <= 5))),
        },
        "scales": _scale_stats(arrays, users),
        "artifacts": {
            "dataset": file_identity(dataset_path),
            "users": file_identity(users_path),
            "student_embedding": file_identity(embedding_path),
            "teacher_relation": file_identity(relation_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_working_set_bytes": _peak_working_set_bytes(),
        "final_week": "not_run",
    }
    atomic_json(cutoff_dir / "manifest.json", result)
    result["artifacts"]["manifest"] = file_identity(cutoff_dir / "manifest.json")
    return result


def run_m50(
    *, source_root: Path, m4_artifact_dir: Path, artifact_dir: Path, report_dir: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for M5.0 but unavailable")
    device = torch.device(device_name)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    m4_metrics = json.loads((report_dir.parent / "m4" / "M4_2_metrics.json").read_text(encoding="utf-8"))
    lambda_pairwise = float(m4_metrics["mechanics"]["selected_lambda"])
    cutoffs = sorted({cutoff for p in ROLLING_PROTOCOL.values() for cutoff in p["outer_train"]})
    rows: dict[str, Any] = {}
    for cutoff in cutoffs:
        print(f"M5.0 cutoff {cutoff}", flush=True)
        rows[cutoff] = build_cutoff_dataset(
            source_root=source_root,
            teacher_dir=m4_artifact_dir / "teacher-v1",
            static_catalog_dir=m4_artifact_dir / "student-v1" / "static_catalog",
            artifact_dir=artifact_dir / "train-data-v1",
            cutoff=cutoff, lambda_pairwise=lambda_pairwise, device=device,
        )
    aggregate: dict[str, Any] = {}
    for window, protocol in ROLLING_PROTOCOL.items():
        aggregate[window] = {}
        for scale in ("10pct", "30pct", "full"):
            selected = [rows[cutoff]["scales"][scale] for cutoff in protocol["outer_train"]]
            aggregate[window][scale] = {
                key: sum(int(value[key]) for value in selected)
                for key in (
                    "candidate_rows", "positive_pairs", "strict_cold_positive_pairs",
                    "sparse_1_5_positive_pairs", "positive_groups",
                )
            }
            aggregate[window][scale]["positive_density"] = (
                aggregate[window][scale]["positive_pairs"]
                / max(aggregate[window][scale]["candidate_rows"], 1)
            )
            medians = [value["positive_per_positive_group_median"] for value in selected if value["positive_groups"]]
            aggregate[window][scale]["positive_per_positive_group_median"] = float(np.median(medians)) if medians else 0.0
    scale = "10pct"
    if any(value["10pct"]["positive_pairs"] < 500 for value in aggregate.values()):
        scale = "30pct"
    if scale == "30pct" and any(value["30pct"]["positive_pairs"] < 1000 for value in aggregate.values()):
        scale = "full"
    model_type = "binary" if any(
        value["full"]["positive_groups"] < 1000
        or value["full"]["positive_per_positive_group_median"] <= 1
        for value in aggregate.values()
    ) else "lambdarank"
    gates = {
        "all_cutoffs_completed": all(value["status"] == "completed" for value in rows.values()),
        "all_cutoff_audits_pass": all(value["seed_audit"]["cutoff_safe"] for value in rows.values()),
        "full_positive_groups_at_least_100_each_outer_train": all(
            value["full"]["positive_groups"] >= 100 for value in aggregate.values()
        ),
        "all_outer_trains_have_positive_pairs": all(value["full"]["positive_pairs"] > 0 for value in aggregate.values()),
    }
    result = {
        "schema_version": "m5.0-scale-gate-v1",
        "stage": "M5.0",
        "status": "measured",
        "run_id": M5_RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "scales": ["10pct", "30pct", "full"],
            "scale_rule": "10% unless any outer-train positives<500; then 30%; full if any 30% positives<1000",
            "model_type_rule": "binary if any full outer-train positive groups<1000 or median positives per positive group<=1; otherwise LambdaRank",
            "point_in_time_student": True,
        },
        "cutoffs": rows,
        "outer_train_aggregate": aggregate,
        "decision": {"selected_training_scale": scale, "selected_model_type": model_type},
        "gates": gates,
        "gate_passed": all(gates.values()),
        "resources": {"peak_working_set_bytes": _peak_working_set_bytes()},
        "final_week": "not_run",
    }
    atomic_json(report_dir / "M5_0_metrics.json", result)
    return result
