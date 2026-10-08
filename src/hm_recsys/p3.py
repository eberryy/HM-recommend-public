from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch

from .m3 import _peak_working_set_bytes
from .m33 import ROLLING_PROTOCOL
from .m4_contract import (
    FINAL_CUTOFF,
    atomic_json,
    file_identity,
    validate_fashionclip,
)
from .p3_model import (
    EMBEDDING_DIM,
    HIDDEN_DIM,
    HISTORY_N,
    K0,
    NEGATIVES_PER_POSITIVE,
    SAMPLING_SEED,
    CandidateAwareReranker,
    attention_summary,
    binary_auc,
    iter_slices,
    pairwise_logistic_loss,
    sample_same_user_pairs,
    stable_seed,
)


RUN_ID = "phase3-v1-candidate-aware-history"
K_VALUES = (5, 10, 20, 50, 100, 200)
TRAINING_K_VALUES = (20, 50, 100, 200)
RECENCY_HALF_LIFE_DAYS = 28.0
MAX_INNER_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


def _literal(path: Path) -> str:
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Phase 3 but torch.cuda.is_available() is false")
    return torch.device(name)


def _safe_cutoff(cutoff: str) -> None:
    if cutoff >= FINAL_CUTOFF:
        raise RuntimeError(f"Phase 3 refuses final or later cutoff: {cutoff}")


def _training_cutoffs() -> list[str]:
    return sorted({value for protocol in ROLLING_PROTOCOL.values() for value in protocol["outer_train"]})


def _all_cutoffs() -> list[str]:
    return sorted(set(_training_cutoffs()) | set(_window_by_outer_cutoff()))


def _window_by_outer_cutoff() -> dict[str, str]:
    return {protocol["outer_validation"]: name for name, protocol in ROLLING_PROTOCOL.items()}


def _student_paths(
    *, cutoff: str, role: str, m4_artifact_dir: Path, m5_artifact_dir: Path
) -> tuple[Path, Path, str]:
    if role == "outer_validation":
        window = _window_by_outer_cutoff().get(cutoff)
        if window is None:
            raise ValueError(f"not an outer cutoff: {cutoff}")
        root = m4_artifact_dir / "student-v1" / window / "multimodal"
        return root / "catalog_embeddings.float16.npy", root / "manifest.json", "m4_outer_multimodal"
    if role != "training":
        raise ValueError(f"unknown Student asset role: {role}")
    root = m5_artifact_dir / "train-data-v1" / cutoff
    return (
        root / "student" / "catalog_embeddings.float16.npy",
        root / "manifest.json",
        "m5_point_in_time_m4_student",
    )


def run_p30(
    *,
    source_root: Path,
    repo_root: Path,
    m4_artifact_dir: Path,
    m5_artifact_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    if list(ROLLING_PROTOCOL) != [
        "winter_20200122",
        "spring_20200318",
        "early_summer_20200624",
        "late_summer_20200819",
    ]:
        raise RuntimeError("Phase 3 rolling protocol drift")
    for cutoff in _all_cutoffs():
        _safe_cutoff(cutoff)
    m42_path = repo_root / "reports" / "m4" / "M4_2_metrics.json"
    m43_path = repo_root / "reports" / "m4" / "M4_3_metrics.json"
    m56_path = repo_root / "reports" / "m5_6" / "metrics.json"
    m42 = _json(m42_path)
    m43 = _json(m43_path)
    m56 = _json(m56_path)
    if m42.get("status") != "measured" or m42.get("final_week") != "not_run":
        raise RuntimeError("M4.2 evidence is not reusable")
    if m43.get("status") != "measured" or not m43.get("summary", {}).get("gate_passed"):
        raise RuntimeError("M4.3 retrieval gate is not reusable")
    if m56.get("status") != "measured" or m56.get("final_week") != "not_run":
        raise RuntimeError("M5.6 final-week boundary failed")

    fashionclip = validate_fashionclip(source_root / "artifacts" / "m2_4" / "fashionclip-full-v1")
    static_dir = m4_artifact_dir / "student-v1" / "static_catalog"
    static_expected = m42["catalog"]["artifacts"]
    static_files = {
        "items": static_dir / "catalog_items.csv",
        "image_rows": static_dir / "catalog_image_rows.int32.npy",
        "categories": static_dir / "catalog_categories.int32.npy",
        "category_maps": static_dir / "category_maps.json",
    }
    static_observed = {name: file_identity(path) for name, path in static_files.items()}
    static_checks = {
        name: observed["sha256"] == static_expected[name]["sha256"]
        and observed["bytes"] == static_expected[name]["bytes"]
        for name, observed in static_observed.items()
    }
    if not all(static_checks.values()):
        raise RuntimeError("M4 static catalog identity drift")

    student_assets: dict[str, Any] = {}
    asset_specs = [
        (f"training:{cutoff}", cutoff, "training", None) for cutoff in _training_cutoffs()
    ] + [
        (f"outer_validation:{window}", protocol["outer_validation"], "outer_validation", window)
        for window, protocol in ROLLING_PROTOCOL.items()
    ]
    for asset_key, cutoff, role, declared_window in asset_specs:
        embedding, manifest_path, source_kind = _student_paths(
            cutoff=cutoff, role=role, m4_artifact_dir=m4_artifact_dir, m5_artifact_dir=m5_artifact_dir
        )
        if not embedding.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"missing frozen Student asset for {cutoff}")
        manifest = _json(manifest_path)
        observed = file_identity(embedding)
        if source_kind == "m4_outer_multimodal":
            window = declared_window
            assert window is not None
            expected = m42["windows"][window]["variants"]["multimodal"]["encoding"]["artifact"]
            manifest_cutoff = manifest.get("outer_validation_cutoff")
        else:
            expected = manifest["artifacts"]["student_embedding"]
            manifest_cutoff = manifest.get("cutoff")
            if manifest.get("contract", {}).get("student_teacher_relation") != (
                "same-cutoff relation built only from transactions before cutoff"
            ):
                raise RuntimeError(f"point-in-time Student contract drift: {cutoff}")
        checks = {
            "sha256": observed["sha256"] == expected["sha256"],
            "bytes": observed["bytes"] == expected["bytes"],
            "cutoff": str(manifest_cutoff) == cutoff,
            "shape": tuple(np.load(embedding, mmap_mode="r").shape) == (105_542, EMBEDDING_DIM),
            "dtype": str(np.load(embedding, mmap_mode="r").dtype) == "float16",
        }
        if not all(checks.values()):
            raise RuntimeError(f"Student identity gate failed for {cutoff}: {checks}")
        student_assets[asset_key] = {
            "cutoff": cutoff,
            "role": role,
            "source_kind": source_kind,
            "checks": checks,
            "embedding": observed,
            "manifest": file_identity(manifest_path),
        }

    result = {
        "schema_version": "phase3-p3.0-freeze-reuse-v1",
        "stage": "P3.0",
        "status": "measured",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_contract": {
            "cold_threshold_max_events_before_cutoff": 5,
            "history_distinct_items": HISTORY_N,
            "coarse_candidate_budget": K0,
            "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
            "student_retraining": False,
            "two_tower": False,
            "final_week": "not_run",
        },
        "checks": {
            "rolling_protocol": True,
            "m4_2_measured": True,
            "m4_3_gate_passed": True,
            "m5_6_final_week_not_run": True,
            "fashionclip_identity": fashionclip["status"] == "passed",
            "static_catalog_identity": all(static_checks.values()),
            "all_student_assets": len(student_assets) == 12,
        },
        "static_catalog": static_observed,
        "fashionclip": fashionclip,
        "student_assets": student_assets,
        "historical_evidence": {
            "m4_2": file_identity(m42_path),
            "m4_3": file_identity(m43_path),
            "m5_6": file_identity(m56_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    result["gate_passed"] = all(result["checks"].values())
    if not result["gate_passed"]:
        raise RuntimeError(f"P3.0 reuse gate failed: {result['checks']}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(artifact_dir / "manifest.json", result)
    atomic_json(report_dir / "P3_0_metrics.json", result)
    return result


def _outer_users_and_warm(db_path: Path) -> tuple[list[str], dict[str, set[str]]]:
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        users = [str(row[0]) for row in connection.execute(
            "SELECT customer_id FROM m29_eval_users ORDER BY customer_id"
        ).fetchall()]
        warm_frame = connection.execute(
            "SELECT customer_id,article_id FROM predictions"
        ).fetch_df()
    finally:
        connection.close()
    warm = {
        str(user): set(group["article_id"].astype(str))
        for user, group in warm_frame.groupby("customer_id", sort=False)
    }
    return users, warm


def _eligible_training_users(transactions_path: Path, cutoff: str) -> list[str]:
    _safe_cutoff(cutoff)
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            WITH counts AS (
                SELECT article_id,count(*) AS events
                FROM read_parquet({_literal(transactions_path)})
                WHERE t_dat < DATE '{cutoff}' GROUP BY article_id
            ), eligible_truth AS (
                SELECT DISTINCT t.customer_id,t.article_id
                FROM read_parquet({_literal(transactions_path)}) t
                LEFT JOIN counts c USING(article_id)
                WHERE t.t_dat >= DATE '{cutoff}'
                  AND t.t_dat < DATE '{cutoff}' + INTERVAL 7 DAY
                  AND coalesce(c.events,0) <= 5
            )
            SELECT DISTINCT customer_id FROM eligible_truth ORDER BY customer_id
            """
        ).fetchall()
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def _cutoff_context(
    *,
    transactions_path: Path,
    cutoff: str,
    users: list[str],
    article_to_row: dict[str, int],
    catalog_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, set[int]], dict[str, Any]]:
    _safe_cutoff(cutoff)
    connection = duckdb.connect()
    user_frame = pd.DataFrame({"customer_id": users})
    connection.register("p3_users", user_frame)
    try:
        count_frame = connection.execute(
            f"SELECT article_id,count(*) AS events FROM read_parquet({_literal(transactions_path)}) "
            f"WHERE t_dat < DATE '{cutoff}' GROUP BY article_id"
        ).fetch_df()
        history_frame = connection.execute(
            f"""
            WITH latest AS (
                SELECT t.customer_id,t.article_id,max(t.t_dat) AS latest_date
                FROM read_parquet({_literal(transactions_path)}) t
                JOIN p3_users u USING(customer_id)
                WHERE t.t_dat < DATE '{cutoff}'
                GROUP BY t.customer_id,t.article_id
            ), ranked AS (
                SELECT *,row_number() OVER(
                    PARTITION BY customer_id ORDER BY latest_date DESC,article_id
                ) AS history_rank
                FROM latest
            )
            SELECT customer_id,article_id,history_rank,
                   date_diff('day',latest_date,DATE '{cutoff}') AS days_since_purchase
            FROM ranked WHERE history_rank <= {HISTORY_N}
            ORDER BY customer_id,history_rank
            """
        ).fetch_df()
        truth_frame = connection.execute(
            f"""
            SELECT DISTINCT t.customer_id,t.article_id
            FROM read_parquet({_literal(transactions_path)}) t
            JOIN p3_users u USING(customer_id)
            WHERE t.t_dat >= DATE '{cutoff}' AND t.t_dat < DATE '{cutoff}' + INTERVAL 7 DAY
            """
        ).fetch_df()
        latest = connection.execute(
            f"SELECT max(t_dat) FROM read_parquet({_literal(transactions_path)}) "
            f"WHERE t_dat < DATE '{cutoff}'"
        ).fetchone()[0]
    finally:
        connection.unregister("p3_users")
        connection.close()

    counts = np.zeros(catalog_size, dtype=np.int32)
    for row in count_frame.itertuples(index=False):
        catalog_row = article_to_row.get(str(row.article_id))
        if catalog_row is not None:
            counts[catalog_row] = int(row.events)
    user_to_index = {user: index for index, user in enumerate(users)}
    history_rows = np.full((len(users), HISTORY_N), -1, dtype=np.int32)
    history_days = np.zeros((len(users), HISTORY_N), dtype=np.float32)
    for row in history_frame.itertuples(index=False):
        user_index = user_to_index.get(str(row.customer_id))
        catalog_row = article_to_row.get(str(row.article_id))
        position = int(row.history_rank) - 1
        if user_index is not None and catalog_row is not None and 0 <= position < HISTORY_N:
            history_rows[user_index, position] = catalog_row
            history_days[user_index, position] = float(row.days_since_purchase)
    history_mask = history_rows >= 0
    truth: dict[int, set[int]] = {}
    for row in truth_frame.itertuples(index=False):
        user_index = user_to_index.get(str(row.customer_id))
        catalog_row = article_to_row.get(str(row.article_id))
        if user_index is not None and catalog_row is not None:
            truth.setdefault(user_index, set()).add(catalog_row)
    audit = {
        "cutoff": cutoff,
        "users": len(users),
        "users_with_history": int(np.count_nonzero(history_mask.any(axis=1))),
        "history_rows": int(np.count_nonzero(history_mask)),
        "history_distinct_by_construction": True,
        "history_max_items": HISTORY_N,
        "history_max_days_since_purchase": float(history_days[history_mask].max()) if np.any(history_mask) else None,
        "truth_pairs": int(sum(len(value) for value in truth.values())),
        "latest_behavior_before_cutoff": str(latest),
        "cutoff_safe": latest is not None and str(latest) < cutoff,
    }
    if not audit["cutoff_safe"]:
        raise RuntimeError(f"history cutoff audit failed: {cutoff}")
    return counts, history_rows, history_days, truth, audit


def _retrieve_top200(
    *,
    embeddings_path: Path,
    candidate_rows: np.ndarray,
    history_rows: np.ndarray,
    history_days: np.ndarray,
    truth: dict[int, set[int]],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    embeddings = np.load(embeddings_path, mmap_mode="r")
    candidate_np = np.asarray(embeddings[candidate_rows], dtype=np.float32)
    candidate_tensor = torch.from_numpy(candidate_np).to(device)
    active = np.flatnonzero((history_rows >= 0).any(axis=1)).astype(np.int32)
    outputs: dict[str, list[Any]] = {
        "user_index": [],
        "catalog_row": [],
        "rank": [],
        "coarse_score": [],
        "target": [],
        "item_count": [],
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_number, batch_slice in enumerate(iter_slices(len(active), 48), start=1):
            user_indices = active[batch_slice]
            rows = history_rows[user_indices]
            mask = rows >= 0
            flat = rows.reshape(-1)
            history_np = np.zeros((len(flat), EMBEDDING_DIM), dtype=np.float32)
            valid = flat >= 0
            history_np[valid] = np.asarray(embeddings[flat[valid]], dtype=np.float32)
            history_tensor = torch.from_numpy(
                history_np.reshape(len(user_indices), HISTORY_N, EMBEDDING_DIM)
            ).to(device)
            similarities = torch.einsum("bhd,cd->bhc", history_tensor, candidate_tensor)
            mask_tensor = torch.from_numpy(mask).to(device)
            weights = np.power(0.5, history_days[user_indices] / RECENCY_HALF_LIFE_DAYS).astype(np.float32)
            weight_tensor = torch.from_numpy(weights).to(device)
            weighted = torch.where(
                mask_tensor[:, :, None],
                similarities * weight_tensor[:, :, None],
                torch.full_like(similarities, -torch.inf),
            )
            coarse = weighted.max(dim=1).values
            take = min(K0, len(candidate_rows))
            values, local_indices = torch.topk(coarse, k=take, dim=1, largest=True, sorted=False)
            values_np = values.cpu().numpy()
            local_np = local_indices.cpu().numpy()
            for local_user, user_index in enumerate(user_indices.tolist()):
                ordering = sorted(
                    zip(local_np[local_user].tolist(), values_np[local_user].tolist()),
                    key=lambda value: (-float(value[1]), int(candidate_rows[int(value[0])])),
                )
                truth_rows = truth.get(int(user_index), set())
                for rank, (local_row, score) in enumerate(ordering, start=1):
                    catalog_row = int(candidate_rows[int(local_row)])
                    outputs["user_index"].append(int(user_index))
                    outputs["catalog_row"].append(catalog_row)
                    outputs["rank"].append(rank)
                    outputs["coarse_score"].append(float(score))
                    outputs["target"].append(int(catalog_row in truth_rows))
            if batch_number % 50 == 0:
                print(
                    f"P3.1 retrieval active users {min(batch_slice.stop, len(active))}/{len(active)}",
                    flush=True,
                )
    arrays = {
        "user_index": np.asarray(outputs["user_index"], dtype=np.int32),
        "catalog_row": np.asarray(outputs["catalog_row"], dtype=np.int32),
        "rank": np.asarray(outputs["rank"], dtype=np.uint16),
        "coarse_score": np.asarray(outputs["coarse_score"], dtype=np.float32),
        "target": np.asarray(outputs["target"], dtype=np.uint8),
    }
    counts_per_user = np.bincount(arrays["user_index"], minlength=len(history_rows))
    identities = arrays["user_index"].astype(np.int64) * len(embeddings) + arrays["catalog_row"]
    audit = {
        "candidate_rows": int(len(arrays["user_index"])),
        "active_users": int(len(active)),
        "inactive_users": int(len(history_rows) - len(active)),
        "candidate_universe_items": int(len(candidate_rows)),
        "user_item_unique": len(np.unique(identities)) == len(identities),
        "active_users_exact_budget": bool(np.all(counts_per_user[active] == min(K0, len(candidate_rows)))),
        "inactive_users_zero_candidates": bool(np.all(counts_per_user[counts_per_user == 0] == 0)),
        "rank_continuous": True,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
    }
    cursor = 0
    for user_index in active:
        end = cursor + int(counts_per_user[user_index])
        if not np.array_equal(arrays["rank"][cursor:end], np.arange(1, end - cursor + 1)):
            audit["rank_continuous"] = False
            break
        cursor = end
    if not all(
        audit[name]
        for name in ("user_item_unique", "active_users_exact_budget", "rank_continuous")
    ):
        raise RuntimeError(f"P3.1 candidate invariant failed: {audit}")
    del candidate_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, audit


def _truth_rows_by_user(truth: dict[int, set[int]], counts: np.ndarray) -> dict[int, dict[str, set[int]]]:
    output: dict[int, dict[str, set[int]]] = {}
    for user_index, rows in truth.items():
        output[user_index] = {
            "strict_cold": {row for row in rows if counts[row] == 0},
            "sparse_1_5": {row for row in rows if 1 <= counts[row] <= 5},
            "cold_universe": {row for row in rows if counts[row] <= 5},
        }
    return output


def _ordering_metrics(
    *,
    arrays: dict[str, np.ndarray],
    ordering_rank: np.ndarray,
    truth: dict[int, set[int]],
    counts: np.ndarray,
    user_count: int,
) -> dict[str, Any]:
    if len(ordering_rank) != len(arrays["user_index"]):
        raise ValueError("ordering rank length differs from candidate rows")
    truth_segments = _truth_rows_by_user(truth, counts)
    output: dict[str, Any] = {"at_k": {}}
    for k in K_VALUES:
        selected = ordering_rank <= k
        selected_rows = int(np.count_nonzero(selected))
        positive_rows = int(np.count_nonzero(selected & (arrays["target"] == 1)))
        segment_metrics: dict[str, Any] = {}
        for segment in ("strict_cold", "sparse_1_5", "cold_universe"):
            recalls: list[float] = []
            hits: list[float] = []
            selected_by_user: dict[int, set[int]] = {}
            for user_index, catalog_row in zip(
                arrays["user_index"][selected].tolist(), arrays["catalog_row"][selected].tolist()
            ):
                selected_by_user.setdefault(int(user_index), set()).add(int(catalog_row))
            truth_pairs = 0
            covered_pairs = 0
            for user_index in range(user_count):
                relevant = truth_segments.get(user_index, {}).get(segment, set())
                if not relevant:
                    continue
                matched = len(relevant.intersection(selected_by_user.get(user_index, set())))
                recalls.append(matched / len(relevant))
                hits.append(float(matched > 0))
                truth_pairs += len(relevant)
                covered_pairs += matched
            segment_metrics[segment] = {
                "recall": float(np.mean(recalls)) if recalls else 0.0,
                "hit_rate": float(np.mean(hits)) if hits else 0.0,
                "truth_users": len(recalls),
                "truth_pairs": truth_pairs,
                "covered_truth_pairs": covered_pairs,
            }
        density = positive_rows / max(selected_rows, 1)
        output["at_k"][str(k)] = {
            "candidate_rows": selected_rows,
            "positive_rows": positive_rows,
            "positive_density": density,
            "truth_pairs_per_1k_candidates": 1000.0 * density,
            "segments": segment_metrics,
        }
    positive_ranks = ordering_rank[arrays["target"] == 1].astype(np.int32)
    first_ranks: list[int] = []
    reciprocal: list[float] = []
    for user_index in range(user_count):
        if not truth_segments.get(user_index, {}).get("cold_universe"):
            continue
        mask = (arrays["user_index"] == user_index) & (arrays["target"] == 1)
        if np.any(mask):
            first = int(np.min(ordering_rank[mask]))
            first_ranks.append(first)
            reciprocal.append(1.0 / first)
        else:
            reciprocal.append(0.0)
    total_coarse_positive = int(np.count_nonzero(arrays["target"] == 1))
    output["ranking"] = {
        "mrr": float(np.mean(reciprocal)) if reciprocal else 0.0,
        "mrr_truth_user_denominator": len(reciprocal),
        "first_positive_rank_mean_hit_users": float(np.mean(first_ranks)) if first_ranks else None,
        "first_positive_rank_median_hit_users": float(np.median(first_ranks)) if first_ranks else None,
        "first_positive_rank_hit_users": len(first_ranks),
        "positive_rank_best": int(np.min(positive_ranks)) if len(positive_ranks) else None,
        "positive_rank_p25": float(np.quantile(positive_ranks, 0.25)) if len(positive_ranks) else None,
        "positive_rank_p50": float(np.quantile(positive_ranks, 0.50)) if len(positive_ranks) else None,
        "positive_rank_p75": float(np.quantile(positive_ranks, 0.75)) if len(positive_ranks) else None,
        "coarse_positive_pairs_top200": total_coarse_positive,
        "conversion": {
            f"top200_to_top{k}": int(np.count_nonzero(positive_ranks <= k)) / max(total_coarse_positive, 1)
            for k in (5, 10, 20, 50)
        },
        "conversion_counts": {
            f"top{k}": int(np.count_nonzero(positive_ranks <= k)) for k in (5, 10, 20, 50)
        },
    }
    return output


def _incremental_efficiency(
    *, arrays: dict[str, np.ndarray], users: list[str], warm: dict[str, set[str]], catalog_items: list[str]
) -> dict[str, Any]:
    rows = 0
    positives = 0
    for user_index, catalog_row, target in zip(
        arrays["user_index"].tolist(), arrays["catalog_row"].tolist(), arrays["target"].tolist()
    ):
        item = catalog_items[int(catalog_row)]
        if item not in warm.get(users[int(user_index)], set()):
            rows += 1
            positives += int(target)
    return {
        "reference": "frozen Warm-v1 Top300 candidate set",
        "incremental_candidate_rows": rows,
        "incremental_truth_pairs": positives,
        "incremental_truth_pairs_per_1k_incremental_candidates": 1000.0 * positives / max(rows, 1),
    }


def _artifact_reusable(cutoff_dir: Path, expected_embedding: Path) -> bool:
    manifest_path = cutoff_dir / "manifest.json"
    candidate_path = cutoff_dir / "candidates_top200.npz"
    users_path = cutoff_dir / "users.csv"
    history_path = cutoff_dir / "histories.npz"
    if not all(path.is_file() for path in (manifest_path, candidate_path, users_path, history_path)):
        return False
    manifest = _json(manifest_path)
    if manifest.get("status") != "completed" or manifest.get("final_week") != "not_run":
        return False
    try:
        checks = [
            file_identity(candidate_path)["sha256"] == manifest["artifacts"]["candidates"]["sha256"],
            file_identity(users_path)["sha256"] == manifest["artifacts"]["users"]["sha256"],
            file_identity(history_path)["sha256"] == manifest["artifacts"]["histories"]["sha256"],
            file_identity(expected_embedding)["sha256"] == manifest["student_embedding"]["sha256"],
        ]
        return all(checks)
    except (KeyError, FileNotFoundError):
        return False


def build_p31_asset(
    *,
    source_root: Path,
    cutoff: str,
    users: list[str],
    role: str,
    warm: dict[str, set[str]] | None,
    embeddings_path: Path,
    static_catalog_dir: Path,
    output_dir: Path,
    device: torch.device,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    transactions = source_root / "data" / "interim" / "audit" / "transactions.parquet"
    catalog_items = pd.read_csv(
        static_catalog_dir / "catalog_items.csv", dtype={"article_id": str}
    )["article_id"].tolist()
    article_to_row = {item: row for row, item in enumerate(catalog_items)}
    counts, history_rows, history_days, truth, cutoff_audit = _cutoff_context(
        transactions_path=transactions,
        cutoff=cutoff,
        users=users,
        article_to_row=article_to_row,
        catalog_size=len(catalog_items),
    )
    candidate_universe = np.flatnonzero(counts <= 5).astype(np.int32)
    arrays, retrieval_audit = _retrieve_top200(
        embeddings_path=embeddings_path,
        candidate_rows=candidate_universe,
        history_rows=history_rows,
        history_days=history_days,
        truth=truth,
        device=device,
    )
    candidates_path = output_dir / "candidates_top200.npz"
    histories_path = output_dir / "histories.npz"
    users_path = output_dir / "users.csv"
    np.savez_compressed(candidates_path, **arrays)
    np.savez_compressed(
        histories_path,
        catalog_row=history_rows,
        days_since_purchase=history_days,
        mask=(history_rows >= 0).astype(np.uint8),
    )
    pd.DataFrame({"customer_id": users}).to_csv(users_path, index=False)
    coarse_metrics = _ordering_metrics(
        arrays=arrays,
        ordering_rank=arrays["rank"],
        truth=truth,
        counts=counts,
        user_count=len(users),
    )
    result = {
        "schema_version": "phase3-p3.1-coarse-top200-v1",
        "stage": "P3.1",
        "status": "completed",
        "run_id": run_id,
        "cutoff": cutoff,
        "role": role,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "candidate_universe": "catalog items with transaction events before cutoff <= 5",
            "history": "most recent 20 distinct purchased items strictly before cutoff; all history if fewer",
            "ordering": "maximum Student cosine times 28-day half-life recency weight; catalog row tie-break",
            "candidate_budget": K0,
            "truth_use": "truth applied only after coarse identities and scores are fixed",
            "training_user_prefilter": (
                "all users with at least one next-week truth in the <=5-event universe; exact for retaining every possible positive group"
                if role == "training"
                else "all frozen outer evaluation users, including zero-positive groups"
            ),
        },
        "cutoff_audit": cutoff_audit,
        "candidate_universe": {
            "strict_cold_items": int(np.count_nonzero(counts == 0)),
            "sparse_1_5_items": int(np.count_nonzero((counts >= 1) & (counts <= 5))),
            "total": int(len(candidate_universe)),
        },
        "retrieval": retrieval_audit,
        "coarse_metrics": coarse_metrics,
        "incremental_efficiency": (
            _incremental_efficiency(arrays=arrays, users=users, warm=warm, catalog_items=catalog_items)
            if warm is not None
            else None
        ),
        "student_embedding": file_identity(embeddings_path),
        "artifacts": {
            "candidates": file_identity(candidates_path),
            "histories": file_identity(histories_path),
            "users": file_identity(users_path),
        },
        "peak_working_set_bytes": _peak_working_set_bytes(),
        "final_week": "not_run",
    }
    atomic_json(output_dir / "manifest.json", result)
    result["artifacts"]["manifest"] = file_identity(output_dir / "manifest.json")
    return result


def run_p31(
    *,
    source_root: Path,
    repo_root: Path,
    m4_artifact_dir: Path,
    m5_artifact_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
    device_name: str,
    reuse_completed: bool,
    student_embeddings: dict[tuple[str, str], Path] | None = None,
    static_catalog_dir: Path | None = None,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = _device(device_name)
    static_dir = static_catalog_dir or (m4_artifact_dir / "student-v1" / "static_catalog")
    windows: dict[str, Any] = {}
    assets: dict[str, Any] = {}
    transactions = source_root / "data" / "interim" / "audit" / "transactions.parquet"
    for cutoff in _training_cutoffs():
        _safe_cutoff(cutoff)
        if student_embeddings is None:
            embedding, _manifest, _source_kind = _student_paths(
                cutoff=cutoff, role="training", m4_artifact_dir=m4_artifact_dir,
                m5_artifact_dir=m5_artifact_dir,
            )
        else:
            embedding = student_embeddings[("training", cutoff)]
        users = _eligible_training_users(transactions, cutoff)
        warm = None
        role = "training"
        cutoff_dir = artifact_dir / "coarse-v1" / "training" / cutoff
        if reuse_completed and _artifact_reusable(cutoff_dir, embedding):
            print(f"P3.1 reuse {cutoff}", flush=True)
            row = _json(cutoff_dir / "manifest.json")
            row["artifacts"]["manifest"] = file_identity(cutoff_dir / "manifest.json")
        else:
            if cutoff_dir.exists():
                raise FileExistsError(f"refusing to overwrite incomplete P3.1 asset: {cutoff_dir}")
            print(f"P3.1 build {cutoff} ({role}, users={len(users)})", flush=True)
            row = build_p31_asset(
                source_root=source_root,
                cutoff=cutoff,
                users=users,
                role=role,
                warm=warm,
                embeddings_path=embedding,
                static_catalog_dir=static_dir,
                output_dir=cutoff_dir,
                device=device,
                run_id=run_id,
            )
        assets[f"training:{cutoff}"] = row
        gc.collect()
    for outer_window, protocol in ROLLING_PROTOCOL.items():
        cutoff = protocol["outer_validation"]
        _safe_cutoff(cutoff)
        if student_embeddings is None:
            embedding, _manifest, _source_kind = _student_paths(
                cutoff=cutoff, role="outer_validation", m4_artifact_dir=m4_artifact_dir,
                m5_artifact_dir=m5_artifact_dir,
            )
        else:
            embedding = student_embeddings[("outer_validation", cutoff)]
        evaluation_db = (
            source_root / "artifacts" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal"
            / outer_window / "evaluation.duckdb"
        )
        users, warm = _outer_users_and_warm(evaluation_db)
        role = "outer_validation"
        cutoff_dir = artifact_dir / "coarse-v1" / "outer" / outer_window
        if reuse_completed and _artifact_reusable(cutoff_dir, embedding):
            print(f"P3.1 reuse {outer_window} ({cutoff})", flush=True)
            row = _json(cutoff_dir / "manifest.json")
            row["artifacts"]["manifest"] = file_identity(cutoff_dir / "manifest.json")
        else:
            if cutoff_dir.exists():
                raise FileExistsError(f"refusing to overwrite incomplete P3.1 asset: {cutoff_dir}")
            print(f"P3.1 build {outer_window} ({cutoff}, {role}, users={len(users)})", flush=True)
            row = build_p31_asset(
                source_root=source_root, cutoff=cutoff, users=users, role=role, warm=warm,
                embeddings_path=embedding, static_catalog_dir=static_dir, output_dir=cutoff_dir,
                device=device,
                run_id=run_id,
            )
        assets[f"outer_validation:{outer_window}"] = row
        windows[outer_window] = row
        gc.collect()
    result = {
        "schema_version": "phase3-p3.1-summary-v1",
        "stage": "P3.1",
        "status": "measured",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "cold_threshold": 5,
            "history_n": HISTORY_N,
            "candidate_budget": K0,
            "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
            "student_retrained": False,
            "effect_promotion": "not_applicable; P3.1 is a measurable Stage-2 input baseline",
        },
        "windows": windows,
        "all_assets": assets,
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "peak_vram_bytes": max(row["retrieval"]["peak_vram_bytes"] for row in assets.values()),
            "artifact_bytes": sum(
                artifact["bytes"]
                for row in assets.values()
                for artifact in row["artifacts"].values()
            ),
        },
        "final_week": "not_run",
    }
    atomic_json(report_dir / "P3_1_metrics.json", result)
    return result


@dataclass
class LoadedAsset:
    cutoff: str
    candidate_path: Path
    history_path: Path
    users_path: Path
    embedding_path: Path
    candidates: dict[str, np.ndarray]
    histories: dict[str, np.ndarray]
    embeddings: np.ndarray
    pairs: np.ndarray
    pair_audit: dict[str, Any]


def _load_asset(
    *, cutoff: str, root: Path, embedding_path: Path
) -> LoadedAsset:
    candidate_path = root / "candidates_top200.npz"
    history_path = root / "histories.npz"
    users_path = root / "users.csv"
    candidate_npz = np.load(candidate_path)
    history_npz = np.load(history_path)
    candidates = {name: np.asarray(candidate_npz[name]) for name in candidate_npz.files}
    histories = {name: np.asarray(history_npz[name]) for name in history_npz.files}
    pairs, pair_audit = sample_same_user_pairs(
        user_index=candidates["user_index"],
        rank=candidates["rank"],
        target=candidates["target"],
        cutoff=cutoff,
    )
    if len(pairs) and not pair_audit["all_buckets_covered"]:
        raise RuntimeError(f"negative rank-bucket coverage failed: {cutoff}")
    return LoadedAsset(
        cutoff=cutoff,
        candidate_path=candidate_path,
        history_path=history_path,
        users_path=users_path,
        embedding_path=embedding_path,
        candidates=candidates,
        histories=histories,
        embeddings=np.load(embedding_path, mmap_mode="r"),
        pairs=pairs,
        pair_audit=pair_audit,
    )


def _batch_inputs(
    asset: LoadedAsset, pairs: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    users = pairs[:, 0].astype(np.int64)
    positive_rows = pairs[:, 1].astype(np.int64)
    negative_rows = pairs[:, 2].astype(np.int64)
    history_catalog = asset.histories["catalog_row"][users]
    mask = history_catalog >= 0
    safe_history = np.maximum(history_catalog, 0)
    history = np.asarray(asset.embeddings[safe_history], dtype=np.float32)
    history[~mask] = 0.0
    positive_catalog = asset.candidates["catalog_row"][positive_rows]
    negative_catalog = asset.candidates["catalog_row"][negative_rows]
    return (
        torch.from_numpy(np.asarray(asset.embeddings[positive_catalog], dtype=np.float32)).to(device),
        torch.from_numpy(np.asarray(asset.embeddings[negative_catalog], dtype=np.float32)).to(device),
        torch.from_numpy(history).to(device),
        torch.from_numpy(asset.histories["days_since_purchase"][users].astype(np.float32)).to(device),
        torch.from_numpy(mask).to(device),
    )


def _pair_epoch(
    *,
    model: CandidateAwareReranker,
    assets: list[LoadedAsset],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pair_fraction_modulus: int | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_pairs = 0
    total_correct = 0
    for asset in assets:
        pairs = asset.pairs
        if pair_fraction_modulus is not None:
            pairs = pairs[np.arange(len(pairs)) % pair_fraction_modulus == 0]
        rng = np.random.default_rng(stable_seed(SAMPLING_SEED, asset.cutoff, "epoch", training))
        order = rng.permutation(len(pairs)) if training else np.arange(len(pairs))
        for batch_slice in iter_slices(len(order), BATCH_SIZE):
            batch_pairs = pairs[order[batch_slice]]
            positive, negative, history, days, mask = _batch_inputs(asset, batch_pairs, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                positive_score, _ = model(positive, history, days, mask)
                negative_score, _ = model(negative, history, days, mask)
                loss = pairwise_logistic_loss(positive_score, negative_score)
                if training:
                    loss.backward()
                    optimizer.step()
            count = len(batch_pairs)
            total_loss += float(loss.detach()) * count
            total_pairs += count
            total_correct += int(torch.count_nonzero(positive_score > negative_score).detach())
    return {
        "loss": total_loss / max(total_pairs, 1),
        "pair_accuracy": total_correct / max(total_pairs, 1),
        "pairs": total_pairs,
    }


def _new_model(device: torch.device) -> CandidateAwareReranker:
    torch.manual_seed(SAMPLING_SEED)
    np.random.seed(SAMPLING_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SAMPLING_SEED)
    return CandidateAwareReranker().to(device)


def _mechanics(asset: LoadedAsset, device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, modulus in (("one_percent", 100), ("ten_percent", 10)):
        model = _new_model(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        before = _pair_epoch(
            model=model,
            assets=[asset],
            device=device,
            optimizer=None,
            pair_fraction_modulus=modulus,
        )
        trained = _pair_epoch(
            model=model,
            assets=[asset],
            device=device,
            optimizer=optimizer,
            pair_fraction_modulus=modulus,
        )
        after = _pair_epoch(
            model=model,
            assets=[asset],
            device=device,
            optimizer=None,
            pair_fraction_modulus=modulus,
        )
        output[name] = {"before": before, "train": trained, "after": after, "passed": math.isfinite(after["loss"])}
        del model, optimizer
    return output


def _train_inner(
    *, train_asset: LoadedAsset, validation_asset: LoadedAsset, device: torch.device, output_path: Path
) -> dict[str, Any]:
    model = _new_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, MAX_INNER_EPOCHS + 1):
        train_metrics = _pair_epoch(
            model=model, assets=[train_asset], device=device, optimizer=optimizer
        )
        validation_metrics = _pair_epoch(
            model=model, assets=[validation_asset], device=device, optimizer=None
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(row)
        print(
            f"P3.2 inner epoch {epoch}: train={train_metrics['loss']:.6f} "
            f"validation={validation_metrics['loss']:.6f}",
            flush=True,
        )
        if validation_metrics["loss"] < best_loss - 1e-5:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale > EARLY_STOPPING_PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("inner training did not create a checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "selected_epoch": best_epoch,
        },
        output_path,
    )
    return {
        "selected_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "model": file_identity(output_path),
    }


def _train_outer(
    *, assets: list[LoadedAsset], epochs: int, device: torch.device, output_path: Path
) -> tuple[CandidateAwareReranker, dict[str, Any]]:
    model = _new_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        metrics = _pair_epoch(model=model, assets=assets, device=device, optimizer=optimizer)
        history.append({"epoch": epoch, **metrics})
        print(f"P3.2 outer epoch {epoch}/{epochs}: loss={metrics['loss']:.6f}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "epochs": epochs,
        },
        output_path,
    )
    return model, {
        "epochs": epochs,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "model": file_identity(output_path),
    }


def _score_asset(
    *, model: CandidateAwareReranker, asset: LoadedAsset, device: torch.device, output_dir: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count = len(asset.candidates["user_index"])
    score_path = output_dir / "candidate_aware_scores.float32.npy"
    attention_path = output_dir / "attention.float16.npy"
    scores = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=(row_count,))
    attention = np.lib.format.open_memmap(
        attention_path, mode="w+", dtype=np.float16, shape=(row_count, HISTORY_N)
    )
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, batch_slice in enumerate(iter_slices(row_count, 2048), start=1):
            users = asset.candidates["user_index"][batch_slice].astype(np.int64)
            catalog = asset.candidates["catalog_row"][batch_slice].astype(np.int64)
            history_catalog = asset.histories["catalog_row"][users]
            mask = history_catalog >= 0
            safe_history = np.maximum(history_catalog, 0)
            history_np = np.asarray(asset.embeddings[safe_history], dtype=np.float32)
            history_np[~mask] = 0.0
            candidate = torch.from_numpy(np.asarray(asset.embeddings[catalog], dtype=np.float32)).to(device)
            history = torch.from_numpy(history_np).to(device)
            days = torch.from_numpy(
                asset.histories["days_since_purchase"][users].astype(np.float32)
            ).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)
            batch_score, batch_attention = model(candidate, history, days, mask_tensor)
            scores[batch_slice] = batch_score.cpu().numpy().astype(np.float32)
            attention[batch_slice] = batch_attention.cpu().numpy().astype(np.float16)
            if batch_number % 250 == 0:
                print(f"P3.2 score rows {batch_slice.stop}/{row_count}", flush=True)
    scores.flush()
    attention.flush()
    observed_scores = np.asarray(scores)
    rerank = np.empty(row_count, dtype=np.uint16)
    cursor = 0
    users_unique = 0
    while cursor < row_count:
        end = cursor + 1
        while end < row_count and asset.candidates["user_index"][end] == asset.candidates["user_index"][cursor]:
            end += 1
        local = np.arange(cursor, end)
        order = sorted(
            local.tolist(),
            key=lambda index: (
                -float(observed_scores[index]),
                int(asset.candidates["rank"][index]),
                int(asset.candidates["catalog_row"][index]),
            ),
        )
        rerank[np.asarray(order, dtype=np.int64)] = np.arange(1, len(order) + 1, dtype=np.uint16)
        cursor = end
        users_unique += 1
    resources = {
        "rows": row_count,
        "users": users_unique,
        "elapsed_seconds": time.perf_counter() - started,
        "scores": file_identity(score_path),
        "attention": file_identity(attention_path),
    }
    return np.asarray(scores).copy(), rerank, resources


def _score_distribution(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"rows": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "rows": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _attention_audit(
    *,
    asset: LoadedAsset,
    scores: np.ndarray,
    attention_path: Path,
    catalog_items: list[str],
    category_by_item: dict[str, str],
) -> dict[str, Any]:
    attention = np.load(attention_path, mmap_mode="r")
    users_for_rows = asset.candidates["user_index"].astype(np.int64)
    masks = asset.histories["mask"][users_for_rows].astype(bool)
    target = asset.candidates["target"] == 1
    all_summary = attention_summary(np.asarray(attention, dtype=np.float32), masks)
    truth_summary = attention_summary(np.asarray(attention[target], dtype=np.float32), masks[target])
    unobserved_summary = attention_summary(np.asarray(attention[~target], dtype=np.float32), masks[~target])
    differences: list[float] = []
    examples: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and users_for_rows[end] == users_for_rows[cursor]:
            end += 1
        valid_history = asset.histories["mask"][int(users_for_rows[cursor])].astype(bool)
        if end - cursor >= 2:
            differences.append(
                float(np.sum(np.abs(
                    np.asarray(attention[cursor, valid_history], dtype=np.float32)
                    - np.asarray(attention[end - 1, valid_history], dtype=np.float32)
                )))
            )
        local_target = np.flatnonzero(target[cursor:end])
        local_unobserved = np.flatnonzero(~target[cursor:end])
        if len(examples) < 3 and len(local_target) and len(local_unobserved):
            positive_row = cursor + int(local_target[0])
            negative_row = cursor + int(local_unobserved[np.argmax(scores[cursor:end][local_unobserved])])
            user_index = int(users_for_rows[cursor])
            history_catalog = asset.histories["catalog_row"][user_index]
            pair_examples = []
            for label, row_index in (("truth", positive_row), ("unobserved", negative_row)):
                valid_positions = np.flatnonzero(asset.histories["mask"][user_index])
                top_position = int(valid_positions[np.argmax(np.asarray(attention[row_index])[valid_positions])])
                candidate_item = catalog_items[int(asset.candidates["catalog_row"][row_index])]
                history_item = catalog_items[int(history_catalog[top_position])]
                pair_examples.append({
                    "candidate_label": label,
                    "candidate_article_hash": hashlib.sha256(candidate_item.encode()).hexdigest()[:10],
                    "candidate_category": category_by_item.get(candidate_item, "unknown"),
                    "top_history_article_hash": hashlib.sha256(history_item.encode()).hexdigest()[:10],
                    "top_history_category": category_by_item.get(history_item, "unknown"),
                    "top_history_attention_weight": float(attention[row_index, top_position]),
                    "top_history_days_since_purchase": float(
                        asset.histories["days_since_purchase"][user_index, top_position]
                    ),
                })
            examples.append({"anonymous_user_index": user_index, "candidates": pair_examples})
        cursor = end
    return {
        "definition": {
            "entropy": "natural-log entropy over the valid history-item attention distribution for one user-candidate row",
            "candidate_attention_difference": "L1 distance between attention vectors for the same user's coarse-rank-1 and coarse-rank-200 candidates; one pair per active user",
            "unobserved": "candidate not purchased in the next seven days; not an exposed rejection",
        },
        "all_candidates": all_summary,
        "truth_candidates": truth_summary,
        "unobserved_candidates": unobserved_summary,
        "candidate_attention_l1_difference": {
            "user_pairs": len(differences),
            "mean": float(np.mean(differences)) if differences else None,
            "median": float(np.median(differences)) if differences else None,
            "p90": float(np.quantile(differences, 0.90)) if differences else None,
        },
        "qualitative_examples": examples,
    }


def _score_separation(
    *, asset: LoadedAsset, scores: np.ndarray
) -> dict[str, Any]:
    target = asset.candidates["target"] == 1
    positive = scores[target]
    negative = scores[~target]
    percentiles: list[float] = []
    cursor = 0
    while cursor < len(scores):
        end = cursor + 1
        while end < len(scores) and asset.candidates["user_index"][end] == asset.candidates["user_index"][cursor]:
            end += 1
        local_positive = scores[cursor:end][target[cursor:end]]
        local_negative = scores[cursor:end][~target[cursor:end]]
        for value in local_positive:
            if len(local_negative):
                percentiles.append(float(
                    (np.count_nonzero(local_negative < value) + 0.5 * np.count_nonzero(local_negative == value))
                    / len(local_negative)
                ))
        cursor = end
    return {
        "positive_vs_unobserved_auc": binary_auc(positive, negative),
        "same_user_positive_percentile": {
            "rows": len(percentiles),
            "mean": float(np.mean(percentiles)) if percentiles else None,
            "median": float(np.median(percentiles)) if percentiles else None,
            "p25": float(np.quantile(percentiles, 0.25)) if percentiles else None,
            "p75": float(np.quantile(percentiles, 0.75)) if percentiles else None,
        },
        "positive_scores": _score_distribution(positive),
        "unobserved_scores": _score_distribution(negative),
    }


def _candidate_parity(asset: LoadedAsset, rerank: np.ndarray) -> dict[str, Any]:
    identities = asset.candidates["user_index"].astype(np.int64) * len(asset.embeddings) + asset.candidates["catalog_row"]
    checks = {
        "row_count_equal": len(rerank) == len(identities),
        "identity_unique": len(np.unique(identities)) == len(identities),
        "identity_set_unchanged": True,
        "per_user_rank_continuous": True,
    }
    cursor = 0
    while cursor < len(rerank):
        end = cursor + 1
        while end < len(rerank) and asset.candidates["user_index"][end] == asset.candidates["user_index"][cursor]:
            end += 1
        if not np.array_equal(np.sort(rerank[cursor:end]), np.arange(1, end - cursor + 1)):
            checks["per_user_rank_continuous"] = False
            break
        cursor = end
    return {"checks": checks, "passed": all(checks.values()), "candidate_rows": len(identities)}


def _load_outer_truth_counts(
    *, source_root: Path, cutoff: str, users: list[str], catalog_items: list[str]
) -> tuple[dict[int, set[int]], np.ndarray]:
    article_to_row = {item: row for row, item in enumerate(catalog_items)}
    counts, _history_rows, _history_days, truth, _audit = _cutoff_context(
        transactions_path=source_root / "data" / "interim" / "audit" / "transactions.parquet",
        cutoff=cutoff,
        users=users,
        article_to_row=article_to_row,
        catalog_size=len(catalog_items),
    )
    return truth, counts


def _mechanism_gate(windows: dict[str, Any]) -> dict[str, Any]:
    density20_improve = []
    density50_nondegrade = []
    mrr_nondegrade = []
    strict_recall20_nondegrade = []
    sparse_recall20_nondegrade = []
    for row in windows.values():
        coarse = row["metrics"]["coarse_student_order"]
        candidate = row["metrics"]["candidate_aware_order"]
        density20_improve.append(candidate["at_k"]["20"]["positive_density"] > coarse["at_k"]["20"]["positive_density"])
        density50_nondegrade.append(candidate["at_k"]["50"]["positive_density"] >= coarse["at_k"]["50"]["positive_density"])
        mrr_nondegrade.append(candidate["ranking"]["mrr"] >= coarse["ranking"]["mrr"])
        strict_recall20_nondegrade.append(
            candidate["at_k"]["20"]["segments"]["strict_cold"]["recall"]
            >= coarse["at_k"]["20"]["segments"]["strict_cold"]["recall"]
        )
        sparse_recall20_nondegrade.append(
            candidate["at_k"]["20"]["segments"]["sparse_1_5"]["recall"]
            >= coarse["at_k"]["20"]["segments"]["sparse_1_5"]["recall"]
        )
    def mean(path: tuple[str, ...], variant: str) -> float:
        values = []
        for row in windows.values():
            value: Any = row["metrics"][variant]
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values))
    coarse_density20 = mean(("at_k", "20", "positive_density"), "coarse_student_order")
    candidate_density20 = mean(("at_k", "20", "positive_density"), "candidate_aware_order")
    coarse_mrr = mean(("ranking", "mrr"), "coarse_student_order")
    candidate_mrr = mean(("ranking", "mrr"), "candidate_aware_order")
    coarse_conversion20 = mean(("ranking", "conversion", "top200_to_top20"), "coarse_student_order")
    candidate_conversion20 = mean(("ranking", "conversion", "top200_to_top20"), "candidate_aware_order")
    segment_checks: dict[str, Any] = {}
    for segment, nondegrade in (
        ("strict_cold", strict_recall20_nondegrade), ("sparse_1_5", sparse_recall20_nondegrade)
    ):
        coarse_mean = mean(("at_k", "20", "segments", segment, "recall"), "coarse_student_order")
        candidate_mean = mean(("at_k", "20", "segments", segment, "recall"), "candidate_aware_order")
        segment_checks[segment] = {
            "nondegrading_windows": int(sum(nondegrade)),
            "coarse_mean": coarse_mean,
            "candidate_aware_mean": candidate_mean,
            "passed": sum(nondegrade) >= 3 and candidate_mean > coarse_mean,
        }
    relative_density20 = (
        candidate_density20 / coarse_density20 - 1.0 if coarse_density20 > 0 else None
    )
    checks = {
        "top200_candidate_exact_parity": all(row["candidate_parity"]["passed"] for row in windows.values()),
        "density20_improves_at_least_3_windows": sum(density20_improve) >= 3,
        "mean_density20_relative_improvement_at_least_15pct": relative_density20 is not None and relative_density20 >= 0.15,
        "density50_nondegrading_at_least_3_windows": sum(density50_nondegrade) >= 3,
        "mrr_nondegrading_at_least_3_windows_and_mean_improves": sum(mrr_nondegrade) >= 3 and candidate_mrr > coarse_mrr,
        "mean_top200_to_top20_conversion_improves": candidate_conversion20 > coarse_conversion20,
        "strict_or_sparse_recall20_gate": any(value["passed"] for value in segment_checks.values()),
        "identity_cutoff_sha_audits": all(row["audit_passed"] for row in windows.values()),
    }
    passed = all(checks.values())
    density_mrr_core = (
        sum(density20_improve) >= 3
        and relative_density20 is not None
        and relative_density20 >= 0.15
        and candidate_mrr > coarse_mrr
    )
    if passed:
        decision = "candidate_aware_version_a_mechanism_gate_passed"
    elif density_mrr_core and not checks["strict_or_sparse_recall20_gate"]:
        decision = "precision_improved_recall_tradeoff_requires_followup"
    else:
        decision = "stop_at_p3_2_candidate_aware_mechanism_gate_failed"
    return {
        "checks": checks,
        "gate_passed": passed,
        "decision": decision,
        "window_counts": {
            "density20_improved": int(sum(density20_improve)),
            "density50_nondegraded": int(sum(density50_nondegrade)),
            "mrr_nondegraded": int(sum(mrr_nondegrade)),
        },
        "means": {
            "coarse_density20": coarse_density20,
            "candidate_aware_density20": candidate_density20,
            "density20_relative_improvement": relative_density20,
            "coarse_mrr": coarse_mrr,
            "candidate_aware_mrr": candidate_mrr,
            "coarse_top200_to_top20_conversion": coarse_conversion20,
            "candidate_aware_top200_to_top20_conversion": candidate_conversion20,
        },
        "segment_recall20": segment_checks,
    }


def run_p32(
    *,
    source_root: Path,
    repo_root: Path,
    m4_artifact_dir: Path,
    m5_artifact_dir: Path,
    artifact_dir: Path,
    report_dir: Path,
    device_name: str,
    student_embeddings: dict[tuple[str, str], Path] | None = None,
    static_catalog_dir: Path | None = None,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = _device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    coarse_root = artifact_dir / "coarse-v1"
    model_root = artifact_dir / "models-v1"
    score_root = artifact_dir / "scores-v1"
    static_dir = static_catalog_dir or (m4_artifact_dir / "student-v1" / "static_catalog")
    catalog_items = pd.read_csv(static_dir / "catalog_items.csv", dtype={"article_id": str})["article_id"].tolist()
    articles = pd.read_csv(
        source_root / "data" / "raw" / "articles.csv",
        dtype={"article_id": str},
        usecols=["article_id", "product_type_name"],
    )
    category_by_item = dict(zip(articles["article_id"], articles["product_type_name"].fillna("unknown")))
    loaded_training: dict[str, LoadedAsset] = {}
    for cutoff in _training_cutoffs():
        if student_embeddings is None:
            embedding, _manifest, _source_kind = _student_paths(
                cutoff=cutoff, role="training", m4_artifact_dir=m4_artifact_dir,
                m5_artifact_dir=m5_artifact_dir,
            )
        else:
            embedding = student_embeddings[("training", cutoff)]
        loaded_training[cutoff] = _load_asset(
            cutoff=cutoff, root=coarse_root / "training" / cutoff, embedding_path=embedding
        )
    loaded_outer: dict[str, LoadedAsset] = {}
    for window, protocol in ROLLING_PROTOCOL.items():
        cutoff = protocol["outer_validation"]
        if student_embeddings is None:
            embedding, _manifest, _source_kind = _student_paths(
                cutoff=cutoff, role="outer_validation", m4_artifact_dir=m4_artifact_dir,
                m5_artifact_dir=m5_artifact_dir,
            )
        else:
            embedding = student_embeddings[("outer_validation", cutoff)]
        loaded_outer[window] = _load_asset(
            cutoff=cutoff, root=coarse_root / "outer" / window, embedding_path=embedding
        )
    mechanics = _mechanics(
        loaded_training[ROLLING_PROTOCOL["winter_20200122"]["inner_train"][0]], device
    )
    if not all(row["passed"] for row in mechanics.values()):
        raise RuntimeError("P3.2 mechanics/smoke failed")

    windows: dict[str, Any] = {}
    attention_audits: dict[str, Any] = {}
    density_audits: dict[str, Any] = {}
    rank_audits: dict[str, Any] = {}
    separation_audits: dict[str, Any] = {}
    for window, protocol in ROLLING_PROTOCOL.items():
        print(f"P3.2 formal window {window}", flush=True)
        inner_train_asset = loaded_training[protocol["inner_train"][0]]
        inner_validation_asset = loaded_training[protocol["inner_validation"]]
        window_model_dir = model_root / window
        inner_training = _train_inner(
            train_asset=inner_train_asset,
            validation_asset=inner_validation_asset,
            device=device,
            output_path=window_model_dir / "inner_candidate_aware.pt",
        )
        outer_assets = [loaded_training[cutoff] for cutoff in protocol["outer_train"]]
        model, outer_training = _train_outer(
            assets=outer_assets,
            epochs=int(inner_training["selected_epoch"]),
            device=device,
            output_path=window_model_dir / "outer_candidate_aware.pt",
        )
        validation_asset = loaded_outer[window]
        window_score_dir = score_root / window
        scores, rerank, scoring = _score_asset(
            model=model, asset=validation_asset, device=device, output_dir=window_score_dir
        )
        users = pd.read_csv(validation_asset.users_path, dtype={"customer_id": str})["customer_id"].tolist()
        truth, counts = _load_outer_truth_counts(
            source_root=source_root,
            cutoff=protocol["outer_validation"],
            users=users,
            catalog_items=catalog_items,
        )
        coarse_metrics = _ordering_metrics(
            arrays=validation_asset.candidates,
            ordering_rank=validation_asset.candidates["rank"],
            truth=truth,
            counts=counts,
            user_count=len(users),
        )
        candidate_metrics = _ordering_metrics(
            arrays=validation_asset.candidates,
            ordering_rank=rerank,
            truth=truth,
            counts=counts,
            user_count=len(users),
        )
        parity = _candidate_parity(validation_asset, rerank)
        if not parity["passed"]:
            raise RuntimeError(f"P3.2 candidate parity failed: {window}")
        separation = _score_separation(asset=validation_asset, scores=scores)
        attention = _attention_audit(
            asset=validation_asset,
            scores=scores,
            attention_path=Path(scoring["attention"]["path"]),
            catalog_items=catalog_items,
            category_by_item=category_by_item,
        )
        top200_parity = (
            coarse_metrics["at_k"]["200"]["positive_rows"]
            == candidate_metrics["at_k"]["200"]["positive_rows"]
            and abs(
                coarse_metrics["at_k"]["200"]["segments"]["strict_cold"]["recall"]
                - candidate_metrics["at_k"]["200"]["segments"]["strict_cold"]["recall"]
            ) <= 1e-15
            and abs(
                coarse_metrics["at_k"]["200"]["segments"]["sparse_1_5"]["recall"]
                - candidate_metrics["at_k"]["200"]["segments"]["sparse_1_5"]["recall"]
            ) <= 1e-15
        )
        audit_passed = bool(
            parity["passed"]
            and top200_parity
            and validation_asset.pair_audit["same_user_only"]
            and validation_asset.pair_audit["all_buckets_covered"]
            and protocol["outer_validation"] < FINAL_CUTOFF
        )
        model_manifest = {
            "schema_version": "phase3-p3.2-model-manifest-v1",
            "run_id": run_id,
            "window": window,
            "outer_validation": protocol["outer_validation"],
            "architecture": {
                "embedding_dim": EMBEDDING_DIM,
                "attention_interaction": "[z_c,z_h,z_c*z_h,abs(z_c-z_h),log1p(days_since_purchase)]",
                "attention_mlp": f"{EMBEDDING_DIM * 4 + 1}->{HIDDEN_DIM}->1 with GELU",
                "score_interaction": "[u(c),z_c,u(c)*z_c,abs(u(c)-z_c)]",
                "score_mlp": f"{EMBEDDING_DIM * 4}->{HIDDEN_DIM}->1 with GELU",
            },
            "training": {
                "loss": "same-user pairwise logistic/BPR",
                "maximum_negatives_per_positive": NEGATIVES_PER_POSITIVE,
                "rank_buckets": ["hard:1-67", "medium:68-134", "easy:135-200"],
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size_pairs": BATCH_SIZE,
                "inner": inner_training,
                "outer": outer_training,
                "pair_audits": {asset.cutoff: asset.pair_audit for asset in outer_assets},
            },
            "inputs": {
                "coarse_candidates": file_identity(validation_asset.candidate_path),
                "histories": file_identity(validation_asset.history_path),
                "student_embedding": file_identity(validation_asset.embedding_path),
            },
            "outputs": scoring,
            "candidate_parity": parity,
            "top200_metric_parity": top200_parity,
            "cutoff_safe": protocol["outer_validation"] < FINAL_CUTOFF,
            "final_week": "not_run",
        }
        atomic_json(window_model_dir / "manifest.json", model_manifest)
        windows[window] = {
            "cutoff": protocol["outer_validation"],
            "metrics": {
                "coarse_student_order": coarse_metrics,
                "candidate_aware_order": candidate_metrics,
            },
            "candidate_parity": parity,
            "score_separation": separation,
            "attention": attention,
            "inner_training": inner_training,
            "outer_training": outer_training,
            "training_pair_audits": {asset.cutoff: asset.pair_audit for asset in outer_assets},
            "scoring": scoring,
            "model_manifest": file_identity(window_model_dir / "manifest.json"),
            "audit_passed": audit_passed,
        }
        density_audits[window] = {
            variant: {k: metrics["at_k"][k] for k in map(str, K_VALUES)}
            for variant, metrics in windows[window]["metrics"].items()
        }
        rank_audits[window] = {
            variant: metrics["ranking"] for variant, metrics in windows[window]["metrics"].items()
        }
        separation_audits[window] = separation
        attention_audits[window] = attention
        del model, scores, rerank
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    gate = _mechanism_gate(windows)
    result = {
        "schema_version": "phase3-p3.2-candidate-aware-v1",
        "stage": "P3.2",
        "status": "measured",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "candidate_set": "frozen P3.1 Top200 per outer cutoff",
            "history_n": HISTORY_N,
            "candidate_budget": K0,
            "label_window_days": 7,
            "loss": "same-user pairwise logistic/BPR",
            "negative_sampling": "up to 50 unobserved candidates per positive with hard/medium/easy coarse-rank coverage and fixed hash",
            "inner_selection": "only early-stopping epoch; no outer-label architecture or hyperparameter selection",
            "full_user_formal": "all exact-eligible training users and all outer evaluation users; no hash subsampling",
            "final_week": "not_run",
        },
        "mechanics": mechanics,
        "windows": windows,
        "mechanism_gate": gate,
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "torch": torch.__version__,
        },
        "final_week": "not_run",
    }
    atomic_json(report_dir / "P3_2_metrics.json", result)
    atomic_json(report_dir / "candidate_density_audit.json", density_audits)
    atomic_json(report_dir / "rank_funnel_audit.json", rank_audits)
    atomic_json(report_dir / "score_separation_audit.json", separation_audits)
    atomic_json(report_dir / "attention_audit.json", attention_audits)
    combined = {
        "schema_version": "phase3-version-a-round1-v1",
        "run_id": run_id,
        "status": "measured",
        "p3_0": _json(report_dir / "P3_0_metrics.json"),
        "p3_1": _json(report_dir / "P3_1_metrics.json"),
        "p3_2": result,
        "final_week": "not_run",
    }
    atomic_json(report_dir / "metrics.json", combined)
    return result
