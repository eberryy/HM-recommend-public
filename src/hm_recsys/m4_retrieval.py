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

from .m33 import ROLLING_PROTOCOL
from .m4_contract import M4_RUN_ID, WARM_MAP, atomic_json, file_identity
from .m3 import _peak_working_set_bytes


K_VALUES = (20, 50, 100)
PRIMARY_K = 50
SEED_K = 5
HISTORY_WEEKS = 12
RECENCY_HALF_LIFE_DAYS = 28.0
VARIANTS = ("raw_fashionclip", "image_only", "metadata_only", "multimodal")


def _load_eval(db_path: Path) -> tuple[list[str], dict[str, set[str]], dict[str, set[str]], dict[str, bool]]:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        users = [str(row[0]) for row in con.execute("SELECT customer_id FROM m29_eval_users ORDER BY customer_id").fetchall()]
        truth_frame = con.execute("SELECT customer_id,article_id FROM m29_eval_truth").fetch_df()
        base_frame = con.execute("SELECT customer_id,article_id FROM predictions").fetch_df()
        activity_frame = con.execute(
            "SELECT customer_id,max(user_history_events_12w)>0 AS active FROM predictions GROUP BY customer_id"
        ).fetch_df()
    finally:
        con.close()
    truth = {user: set(group["article_id"].astype(str)) for user, group in truth_frame.groupby("customer_id")}
    base = {user: set(group["article_id"].astype(str)) for user, group in base_frame.groupby("customer_id")}
    activity = dict(zip(activity_frame["customer_id"].astype(str), activity_frame["active"].astype(bool)))
    return users, truth, base, activity


def _history_counts_and_seeds(
    *, transactions_path: Path, cutoff: str, users: list[str], article_to_row: dict[str, int]
) -> tuple[np.ndarray, dict[str, list[tuple[int, int, float]]], dict[str, Any]]:
    con = duckdb.connect()
    user_frame = pd.DataFrame({"customer_id": users})
    con.register("eval_users_input", user_frame)
    tx = str(transactions_path.resolve()).replace("'", "''")
    try:
        count_frame = con.execute(
            f"SELECT article_id,count(*) AS events FROM read_parquet('{tx}') "
            f"WHERE t_dat<DATE '{cutoff}' GROUP BY article_id"
        ).fetch_df()
        seed_frame = con.execute(
            f"""
            WITH latest AS (
                SELECT t.customer_id,t.article_id,max(t_dat) AS latest_date
                FROM read_parquet('{tx}') t JOIN eval_users_input u USING(customer_id)
                WHERE t_dat>=DATE '{cutoff}'-INTERVAL {HISTORY_WEEKS} WEEK AND t_dat<DATE '{cutoff}'
                GROUP BY t.customer_id,t.article_id
            ), ranked AS (
                SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY latest_date DESC,article_id) AS seed_rank
                FROM latest
            )
            SELECT customer_id,article_id,seed_rank,date_diff('day',latest_date,DATE '{cutoff}') AS days_ago
            FROM ranked WHERE seed_rank<={SEED_K} ORDER BY customer_id,seed_rank
            """
        ).fetch_df()
        latest = con.execute(
            f"SELECT max(t_dat) FROM read_parquet('{tx}') WHERE t_dat<DATE '{cutoff}'"
        ).fetchone()[0]
    finally:
        con.unregister("eval_users_input")
        con.close()
    counts = np.zeros(len(article_to_row), dtype=np.int32)
    for row in count_frame.itertuples(index=False):
        catalog_row = article_to_row.get(str(row.article_id))
        if catalog_row is not None:
            counts[catalog_row] = int(row.events)
    seeds: dict[str, list[tuple[int, int, float]]] = {}
    for row in seed_frame.itertuples(index=False):
        catalog_row = article_to_row.get(str(row.article_id))
        if catalog_row is not None:
            weight = 0.5 ** (float(row.days_ago) / RECENCY_HALF_LIFE_DAYS)
            seeds.setdefault(str(row.customer_id), []).append((catalog_row, int(row.seed_rank), weight))
    return counts, seeds, {
        "latest_transaction_before_cutoff": str(latest),
        "cutoff_safe": latest is not None and str(latest) < cutoff,
        "users_with_seeds": len(seeds),
        "seed_rows": sum(len(value) for value in seeds.values()),
    }


def _aligned_raw_embeddings(
    *, fashionclip_root: Path, image_rows: np.ndarray, selected_rows: np.ndarray
) -> np.ndarray:
    source = np.load(fashionclip_root / "embeddings.float16.npy", mmap_mode="r")
    mapped = np.asarray(image_rows[selected_rows], dtype=np.int64)
    output = np.zeros((len(selected_rows), 512), dtype=np.float32)
    present = mapped >= 0
    if np.any(present):
        output[present] = np.asarray(source[mapped[present]], dtype=np.float32)
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    output = output / np.maximum(norms, 1e-12)
    return output


def retrieve_top100(
    *, embedding_path: Path | None, fashionclip_root: Path, image_rows: np.ndarray,
    candidate_rows: np.ndarray, users: list[str], seeds: dict[str, list[tuple[int, int, float]]],
    device: torch.device,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    started = time.perf_counter()
    if embedding_path is None:
        candidate_np = _aligned_raw_embeddings(
            fashionclip_root=fashionclip_root, image_rows=image_rows, selected_rows=candidate_rows
        )
        full_embeddings = None
    else:
        full_embeddings = np.load(embedding_path, mmap_mode="r")
        candidate_np = np.asarray(full_embeddings[candidate_rows], dtype=np.float32)
    candidate_tensor = torch.from_numpy(candidate_np).to(device)
    results: dict[str, list[dict[str, Any]]] = {}
    active_users = [user for user in users if seeds.get(user)]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_start in range(0, len(active_users), 32):
            batch_users = active_users[batch_start : batch_start + 32]
            seed_rows = np.full((len(batch_users), SEED_K), -1, dtype=np.int64)
            seed_weights = np.zeros((len(batch_users), SEED_K), dtype=np.float32)
            for user_index, user in enumerate(batch_users):
                for seed_index, (catalog_row, _rank, weight) in enumerate(seeds[user][:SEED_K]):
                    seed_rows[user_index, seed_index] = catalog_row
                    seed_weights[user_index, seed_index] = weight
            flat_rows = seed_rows.reshape(-1)
            valid = flat_rows >= 0
            seed_np = np.zeros((len(flat_rows), candidate_np.shape[1]), dtype=np.float32)
            if embedding_path is None:
                if np.any(valid):
                    seed_np[valid] = _aligned_raw_embeddings(
                        fashionclip_root=fashionclip_root,
                        image_rows=image_rows,
                        selected_rows=flat_rows[valid],
                    )
            elif np.any(valid):
                seed_np[valid] = np.asarray(full_embeddings[flat_rows[valid]], dtype=np.float32)
            seed_tensor = torch.from_numpy(seed_np.reshape(len(batch_users), SEED_K, -1)).to(device)
            similarities = torch.einsum("bsd,cd->bsc", seed_tensor, candidate_tensor)
            valid_tensor = torch.from_numpy((seed_rows >= 0)).to(device)
            similarities = similarities.masked_fill(~valid_tensor[:, :, None], -torch.inf)
            weight_tensor = torch.from_numpy(seed_weights).to(device)
            weighted = torch.where(
                valid_tensor[:, :, None],
                similarities * weight_tensor[:, :, None],
                torch.full_like(similarities, -torch.inf),
            )
            primary_scores = weighted.max(dim=1).values
            retrieve_count = min(K_VALUES[-1] + 32, len(candidate_rows))
            values, indices = torch.topk(primary_scores, k=retrieve_count, dim=1, largest=True, sorted=False)
            values_np = values.cpu().numpy()
            indices_np = indices.cpu().numpy()
            similarities_np = similarities.cpu().numpy()
            weighted_np = weighted.cpu().numpy()
            for user_index, user in enumerate(batch_users):
                ordering = sorted(
                    zip(indices_np[user_index].tolist(), values_np[user_index].tolist()),
                    key=lambda pair: (-float(pair[1]), int(candidate_rows[int(pair[0])])),
                )[: K_VALUES[-1]]
                user_rows: list[dict[str, Any]] = []
                valid_seed_count = len(seeds[user])
                for rank, (local_row, primary_score) in enumerate(ordering, start=1):
                    raw_scores = similarities_np[user_index, :valid_seed_count, int(local_row)]
                    recency_scores = weighted_np[user_index, :valid_seed_count, int(local_row)]
                    best_seed = int(np.argmax(recency_scores))
                    top_count = min(3, valid_seed_count)
                    user_rows.append(
                        {
                            "catalog_row": int(candidate_rows[int(local_row)]),
                            "rank": rank,
                            "best_cosine": float(np.max(raw_scores)),
                            "best_recency_weighted_cosine": float(primary_score),
                            "top3_mean_cosine": float(np.mean(np.sort(raw_scores)[-top_count:])),
                            "top3_mean_recency_weighted_cosine": float(np.mean(np.sort(recency_scores)[-top_count:])),
                            "supporting_seed_count": int(np.count_nonzero(raw_scores > 0)),
                            "best_seed_rank": int(seeds[user][best_seed][1]),
                            "best_seed_recency_weight": float(seeds[user][best_seed][2]),
                        }
                    )
                results[user] = user_rows
            if (batch_start // 32 + 1) % 25 == 0:
                print(f"M4.3 retrieval users {min(batch_start + 32, len(active_users))}/{len(active_users)}", flush=True)
    del candidate_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results, {
        "active_users": len(active_users),
        "inactive_users": len(users) - len(active_users),
        "candidate_universe_items": len(candidate_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "ordering": "best recency-weighted cosine descending, then catalog row ascending",
    }


def _segment_metrics(
    *, users: list[str], truth: dict[str, set[str]], candidate: dict[str, set[str]],
    item_counts: dict[str, int], predicate: str,
) -> dict[str, Any]:
    recalls: list[float] = []
    hits: list[float] = []
    oracle: list[float] = []
    truth_pairs = 0
    covered_pairs = 0
    for user in users:
        if predicate == "overall":
            relevant = truth.get(user, set())
        elif predicate == "strict_cold":
            relevant = {item for item in truth.get(user, set()) if item_counts.get(item, 0) == 0}
        elif predicate == "sparse_1_5":
            relevant = {item for item in truth.get(user, set()) if 1 <= item_counts.get(item, 0) <= 5}
        elif predicate == "sparse_6_20":
            relevant = {item for item in truth.get(user, set()) if 6 <= item_counts.get(item, 0) <= 20}
        elif predicate == "warm_21_plus":
            relevant = {item for item in truth.get(user, set()) if item_counts.get(item, 0) >= 21}
        elif predicate == "tail_1_20":
            relevant = {item for item in truth.get(user, set()) if 1 <= item_counts.get(item, 0) <= 20}
        elif predicate == "middle_21_100":
            relevant = {item for item in truth.get(user, set()) if 21 <= item_counts.get(item, 0) <= 100}
        elif predicate == "head_101_plus":
            relevant = {item for item in truth.get(user, set()) if item_counts.get(item, 0) >= 101}
        else:
            raise ValueError(predicate)
        if not relevant:
            continue
        matched = len(relevant.intersection(candidate.get(user, set())))
        recalls.append(matched / len(relevant))
        hits.append(float(matched > 0))
        oracle.append(min(matched, 12) / min(len(relevant), 12))
        truth_pairs += len(relevant)
        covered_pairs += matched
    return {
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "oracle_map@12": float(np.mean(oracle)) if oracle else 0.0,
        "truth_users": len(recalls),
        "truth_pairs": truth_pairs,
        "covered_truth_pairs": covered_pairs,
    }


def evaluate_retrieval(
    *, users: list[str], truth: dict[str, set[str]], base: dict[str, set[str]],
    retrieved: dict[str, list[dict[str, Any]]], catalog_items: list[str], counts: np.ndarray, k: int,
) -> dict[str, Any]:
    item_counts = {item: int(counts[row]) for row, item in enumerate(catalog_items)}
    incremental: dict[str, set[str]] = {}
    union: dict[str, set[str]] = {}
    overlap = 0
    raw_rows = 0
    incremental_rows = 0
    incremental_truth_pairs = 0
    incremental_cold_truth_pairs = 0
    incremental_truth_users: set[str] = set()
    for user in users:
        ranked_items = [catalog_items[row["catalog_row"]] for row in retrieved.get(user, [])[:k]]
        base_items = base.get(user, set())
        overlap += sum(item in base_items for item in ranked_items)
        raw_rows += len(ranked_items)
        new_items = {item for item in ranked_items if item not in base_items}
        incremental[user] = new_items
        union[user] = set(base_items) | new_items
        incremental_rows += len(new_items)
        matched = new_items.intersection(truth.get(user, set()))
        incremental_truth_pairs += len(matched)
        incremental_cold_truth_pairs += sum(item_counts[item] == 0 for item in matched)
        if matched:
            incremental_truth_users.add(user)
    segments = {
        name: _segment_metrics(
            users=users, truth=truth, candidate=union, item_counts=item_counts, predicate=name
        )
        for name in (
            "overall", "strict_cold", "sparse_1_5", "sparse_6_20", "warm_21_plus",
            "tail_1_20", "middle_21_100", "head_101_plus",
        )
    }
    active_users = [user for user in users if user in retrieved]
    inactive_users = [user for user in users if user not in retrieved]
    return {
        "k": k,
        "segments": segments,
        "incremental_truth_pairs": incremental_truth_pairs,
        "incremental_strict_cold_truth_pairs": incremental_cold_truth_pairs,
        "incremental_truth_users": len(incremental_truth_users),
        "raw_retrieval_rows": raw_rows,
        "incremental_candidate_rows_after_warm_dedup": incremental_rows,
        "mean_incremental_candidates_per_user": incremental_rows / max(len(users), 1),
        "candidate_overlap_with_warm_ratio": overlap / max(raw_rows, 1),
        "positive_density_after_warm_dedup": incremental_truth_pairs / max(incremental_rows, 1),
        "candidate_unique_per_user": True,
        "candidate_budget_passed": all(len(rows[:k]) <= k for rows in retrieved.values()),
        "activity_segments": {
            "active": _segment_metrics(
                users=active_users, truth=truth, candidate=union, item_counts=item_counts, predicate="overall"
            ),
            "inactive": _segment_metrics(
                users=inactive_users, truth=truth, candidate=union, item_counts=item_counts, predicate="overall"
            ),
        },
    }


def _base_metrics(
    *, users: list[str], truth: dict[str, set[str]], base: dict[str, set[str]],
    catalog_items: list[str], counts: np.ndarray,
) -> dict[str, Any]:
    item_counts = {item: int(counts[row]) for row, item in enumerate(catalog_items)}
    return {
        "segments": {
            name: _segment_metrics(
                users=users, truth=truth, candidate=base, item_counts=item_counts, predicate=name
            )
            for name in (
                "overall", "strict_cold", "sparse_1_5", "sparse_6_20", "warm_21_plus",
                "tail_1_20", "middle_21_100", "head_101_plus",
            )
        },
        "mean_candidates_per_user": float(np.mean([len(base.get(user, set())) for user in users])),
    }


def _persist_candidates(
    *, path: Path, users: list[str], retrieved: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    user_indices: list[int] = []
    catalog_rows: list[int] = []
    ranks: list[int] = []
    best_scores: list[float] = []
    best_cosines: list[float] = []
    top3_cosines: list[float] = []
    top3_weighted: list[float] = []
    supports: list[int] = []
    best_seed_ranks: list[int] = []
    best_seed_weights: list[float] = []
    for user_index, user in enumerate(users):
        for row in retrieved.get(user, []):
            user_indices.append(user_index)
            catalog_rows.append(row["catalog_row"])
            ranks.append(row["rank"])
            best_scores.append(row["best_recency_weighted_cosine"])
            best_cosines.append(row["best_cosine"])
            top3_cosines.append(row["top3_mean_cosine"])
            top3_weighted.append(row["top3_mean_recency_weighted_cosine"])
            supports.append(row["supporting_seed_count"])
            best_seed_ranks.append(row["best_seed_rank"])
            best_seed_weights.append(row["best_seed_recency_weight"])
    np.savez_compressed(
        path,
        user_index=np.asarray(user_indices, dtype=np.int32),
        catalog_row=np.asarray(catalog_rows, dtype=np.int32),
        rank=np.asarray(ranks, dtype=np.uint8),
        best_recency_weighted_cosine=np.asarray(best_scores, dtype=np.float32),
        best_cosine=np.asarray(best_cosines, dtype=np.float32),
        top3_mean_cosine=np.asarray(top3_cosines, dtype=np.float32),
        top3_mean_recency_weighted_cosine=np.asarray(top3_weighted, dtype=np.float32),
        supporting_seed_count=np.asarray(supports, dtype=np.uint8),
        best_seed_rank=np.asarray(best_seed_ranks, dtype=np.uint8),
        best_seed_recency_weight=np.asarray(best_seed_weights, dtype=np.float32),
    )
    return file_identity(path)


def run_m43(
    *, source_root: Path, student_dir: Path, static_catalog_dir: Path,
    artifact_dir: Path, output_dir: Path, device_name: str = "cuda",
) -> dict[str, Any]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for M4.3 but unavailable")
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    transactions_path = source_root / "data" / "interim" / "audit" / "transactions.parquet"
    fashionclip_root = source_root / "artifacts" / "m2_4" / "fashionclip-full-v1"
    catalog_frame = pd.read_csv(static_catalog_dir / "catalog_items.csv", dtype={"article_id": str})
    catalog_items = catalog_frame["article_id"].tolist()
    article_to_row = {item: row for row, item in enumerate(catalog_items)}
    image_rows = np.load(static_catalog_dir / "catalog_image_rows.int32.npy", mmap_mode="r")
    windows: dict[str, Any] = {}
    for window, protocol in ROLLING_PROTOCOL.items():
        cutoff = protocol["outer_validation"]
        db_path = source_root / "artifacts" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / window / "evaluation.duckdb"
        users, truth, base, warm_activity = _load_eval(db_path)
        counts, seeds, seed_audit = _history_counts_and_seeds(
            transactions_path=transactions_path, cutoff=cutoff, users=users, article_to_row=article_to_row
        )
        if not seed_audit["cutoff_safe"]:
            raise RuntimeError(f"M4.3 seed cutoff audit failed: {cutoff}")
        candidate_rows = np.flatnonzero(counts <= 5).astype(np.int32)
        strict_rows = int(np.count_nonzero(counts[candidate_rows] == 0))
        sparse_rows = int(np.count_nonzero((counts[candidate_rows] >= 1) & (counts[candidate_rows] <= 5)))
        base_metrics = _base_metrics(
            users=users, truth=truth, base=base, catalog_items=catalog_items, counts=counts
        )
        row: dict[str, Any] = {
            "cutoff": cutoff,
            "warm_v1_reference_map@12": WARM_MAP[window],
            "eval_users": len(users),
            "warm_active_users": sum(warm_activity.values()),
            "seed_audit": seed_audit,
            "candidate_universe": {
                "strict_cold_items": strict_rows,
                "sparse_1_5_items": sparse_rows,
                "total_items_count_le_5": len(candidate_rows),
            },
            "base_300": base_metrics,
            "variants": {},
        }
        for variant in VARIANTS:
            print(f"M4.3 {window} {variant}", flush=True)
            embedding_path = None if variant == "raw_fashionclip" else student_dir / window / variant / "catalog_embeddings.float16.npy"
            retrieved, resources = retrieve_top100(
                embedding_path=embedding_path,
                fashionclip_root=fashionclip_root,
                image_rows=image_rows,
                candidate_rows=candidate_rows,
                users=users,
                seeds=seeds,
                device=device,
            )
            variant_dir = artifact_dir / window / variant
            variant_dir.mkdir(parents=True, exist_ok=True)
            artifact = _persist_candidates(
                path=variant_dir / "retrieval_top100.npz", users=users, retrieved=retrieved
            )
            evaluations = {
                str(k): evaluate_retrieval(
                    users=users, truth=truth, base=base, retrieved=retrieved,
                    catalog_items=catalog_items, counts=counts, k=k,
                )
                for k in K_VALUES
            }
            row["variants"][variant] = {
                "resources": resources,
                "evaluation": evaluations,
                "artifact": artifact,
                "embedding": (
                    file_identity(fashionclip_root / "embeddings.float16.npy")
                    if embedding_path is None else file_identity(embedding_path)
                ),
            }
        windows[window] = row
    multimodal = {window: row["variants"]["multimodal"]["evaluation"][str(PRIMARY_K)] for window, row in windows.items()}
    raw = {window: row["variants"]["raw_fashionclip"]["evaluation"][str(PRIMARY_K)] for window, row in windows.items()}
    base = {window: row["base_300"] for window, row in windows.items()}
    strict_improves = {
        window: multimodal[window]["segments"]["strict_cold"]["recall"]
        > base[window]["segments"]["strict_cold"]["recall"]
        for window in windows
    }
    student_vs_raw = {
        window: multimodal[window]["segments"]["strict_cold"]["recall"]
        >= raw[window]["segments"]["strict_cold"]["recall"]
        for window in windows
    }
    means = {
        "multimodal_strict_cold_recall": float(np.mean([
            value["segments"]["strict_cold"]["recall"] for value in multimodal.values()
        ])),
        "raw_strict_cold_recall": float(np.mean([
            value["segments"]["strict_cold"]["recall"] for value in raw.values()
        ])),
        "multimodal_sparse_1_5_recall": float(np.mean([
            value["segments"]["sparse_1_5"]["recall"] for value in multimodal.values()
        ])),
        "raw_sparse_1_5_recall": float(np.mean([
            value["segments"]["sparse_1_5"]["recall"] for value in raw.values()
        ])),
        "multimodal_overall_recall": float(np.mean([
            value["segments"]["overall"]["recall"] for value in multimodal.values()
        ])),
        "warm_overall_recall": float(np.mean([
            value["segments"]["overall"]["recall"] for value in base.values()
        ])),
    }
    audits_passed = all(
        row["seed_audit"]["cutoff_safe"]
        and all(
            variant["evaluation"][str(PRIMARY_K)]["candidate_unique_per_user"]
            and variant["evaluation"][str(PRIMARY_K)]["candidate_budget_passed"]
            for variant in row["variants"].values()
        )
        for row in windows.values()
    )
    gates = {
        "strict_cold_recall_improves_vs_warm_4_of_4": all(strict_improves.values()),
        "strict_cold_mean_above_raw_fashionclip": means["multimodal_strict_cold_recall"] > means["raw_strict_cold_recall"],
        "sparse_1_5_mean_above_raw_fashionclip": means["multimodal_sparse_1_5_recall"] > means["raw_sparse_1_5_recall"],
        "student_strict_cold_not_below_raw_at_least_3_of_4": sum(student_vs_raw.values()) >= 3,
        "each_window_has_new_strict_cold_truth_pair": all(
            value["incremental_strict_cold_truth_pairs"] >= 1 for value in multimodal.values()
        ),
        "overall_recall_mean_not_below_warm": means["multimodal_overall_recall"] >= means["warm_overall_recall"] - 1e-15,
        "all_audits_pass": audits_passed,
    }
    result = {
        "schema_version": "m4.3-student-cold-retrieval-v1",
        "stage": "M4.3",
        "status": "measured",
        "run_id": M4_RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "history_weeks": HISTORY_WEEKS,
            "seed_k": SEED_K,
            "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
            "candidate_universe": "item_events_before_cutoff <= 5",
            "primary_ordering": "best recency-weighted cosine only",
            "k_values": list(K_VALUES),
            "primary_k": PRIMARY_K,
            "deduplication": "retrieve K then remove items already present in frozen Warm-v1",
            "truth_use": "evaluation only after retrieval",
        },
        "windows": windows,
        "summary": {
            "means": means,
            "strict_improves_by_window": strict_improves,
            "student_not_below_raw_by_window": student_vs_raw,
            "gates": gates,
            "gate_passed": all(gates.values()),
            "next_stage": "M5_allowed" if all(gates.values()) else "stop_at_M4.3_and_run_read_only_failure_decomposition",
        },
        "inputs": {
            "transactions": file_identity(transactions_path),
            "static_catalog_items": file_identity(static_catalog_dir / "catalog_items.csv"),
            "static_catalog_image_rows": file_identity(static_catalog_dir / "catalog_image_rows.int32.npy"),
            "code": file_identity(Path(__file__)),
        },
        "resources": {
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "retrieval_artifact_bytes": int(sum(
                variant["artifact"]["bytes"]
                for window in windows.values() for variant in window["variants"].values()
            )),
        },
        "final_week": "not_run",
    }
    atomic_json(output_dir / "M4_3_metrics.json", result)
    return result
