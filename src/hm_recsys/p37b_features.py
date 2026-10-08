from __future__ import annotations

import csv
import gc
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .m4_contract import FINAL_CUTOFF, atomic_json, file_identity


K0 = 200
HISTORY_N = 20
P33_HALF_LIFE_DAYS = 28.0

AGE_BUCKET_NAMES = ("0_7", "8_28", "29_84", "over_84")
AGE_BUCKET_BOUNDS_DAYS: tuple[tuple[float, float | None], ...] = (
    (0.0, 7.0),
    (7.0, 28.0),
    (28.0, 84.0),
    (84.0, None),
)
M4_RELATION_COLUMNS = (
    "bucket_present",
    "history_count",
    "max_cosine",
    "top3_mean_cosine",
    "same_product_type_history_count",
    "same_garment_group_history_count",
)
P33_RELATION_COLUMNS = (
    "p33_max_cosine",
    "p33_top3_mean_cosine",
    "p33_minus_m4_max_cosine",
    "p33_minus_m4_top3_mean",
)
P33_GLOBAL_COLUMNS = (
    "p33_fixed_decay_score",
    "p33_minus_m4_fixed_decay_score",
)
USER_STATE_COLUMNS = (
    "history_count_0_7",
    "history_count_8_28",
    "history_count_29_84",
    "history_count_over_84",
    "days_since_last_purchase",
    "recent_0_28_purchase_share",
    "recent_vs_older_profile_cosine",
    "recent_vs_older_profile_available",
    "recent_28d_product_type_entropy",
    "recent_28d_distinct_product_types",
)
CANDIDATE_STATE_COLUMNS = (
    "m4_coarse_score",
    "m4_coarse_rank",
    "interaction_count_before_cutoff",
    "strict_cold_flag",
    "sparse1_5_flag",
)
OUTPUT_FILENAMES = {
    "m4_relation": "m4_relation.float32.npy",
    "p33_relation": "p33_relation.float32.npy",
    "p33_global": "p33_global.float32.npy",
    "user_state": "user_state.float32.npy",
    "candidate_state": "candidate_state.float32.npy",
}


@dataclass(frozen=True)
class FrozenFeaturePaths:
    """Filesystem inputs whose identities and cutoff lineage must be audited."""

    candidates: Path
    histories: Path
    users: Path
    m4_embeddings: Path
    p33_embeddings: Path
    p31_manifest: Path
    p33_manifest: Path
    m4_manifest: Path | None = None


def assign_age_buckets(days_since_purchase: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Assign every valid history cell to exactly one of the four frozen age buckets.

    The returned int8 array has values 0..3 for valid history cells and -1 for
    padded cells. Boundaries are (inclusive notation): 0--7, 8--28, 29--84,
    and greater than 84 calendar days.
    """

    days = np.asarray(days_since_purchase)
    valid = np.asarray(mask, dtype=bool)
    if days.shape != valid.shape:
        raise ValueError("days_since_purchase and mask must have identical shapes")
    if np.any(~np.isfinite(days[valid])) or np.any(days[valid] < 0):
        raise ValueError("valid history ages must be finite and non-negative")
    if np.any(days[valid] != np.floor(days[valid])):
        raise ValueError("valid history ages must be integer calendar-day differences")
    result = np.full(days.shape, -1, dtype=np.int8)
    result[valid & (days <= 7.0)] = 0
    result[valid & (days > 7.0) & (days <= 28.0)] = 1
    result[valid & (days > 28.0) & (days <= 84.0)] = 2
    result[valid & (days > 84.0)] = 3
    if np.any(result[valid] < 0) or np.any(result[~valid] != -1):
        raise RuntimeError("age-bucket assignment is not exhaustive and exclusive")
    return result


def _safe_cutoff(cutoff: str) -> None:
    if cutoff >= FINAL_CUTOFF:
        raise RuntimeError(f"P3.7B feature builder refuses final or later cutoff: {cutoff}")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _row in reader)


def _declared_identity_matches(path: Path, declared: Mapping[str, Any], label: str) -> dict[str, Any]:
    observed = file_identity(path)
    checks = {
        "bytes": int(observed["bytes"]) == int(declared.get("bytes", -1)),
        "sha256": observed["sha256"] == declared.get("sha256"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{label} identity drift: {checks}")
    return {"observed": observed, "declared": dict(declared), "checks": checks}


def _embedding_identity_from_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    encoding = manifest.get("encoding")
    if isinstance(encoding, Mapping) and isinstance(encoding.get("artifact"), Mapping):
        return encoding["artifact"]
    student = manifest.get("student_embedding")
    return student if isinstance(student, Mapping) else None


def audit_cutoff_lineage(*, cutoff: str, paths: FrozenFeaturePaths) -> dict[str, Any]:
    """Fail closed on file identity drift, future assets, or teacher inference leakage."""

    _safe_cutoff(cutoff)
    p31 = _read_json(paths.p31_manifest)
    p33 = _read_json(paths.p33_manifest)
    if p31.get("status") != "completed" or p31.get("cutoff") != cutoff:
        raise RuntimeError("P3.1 manifest is not a completed exact-cutoff asset")
    cutoff_audit = p31.get("cutoff_audit", {})
    latest = str(cutoff_audit.get("latest_behavior_before_cutoff", ""))
    if cutoff_audit.get("cutoff_safe") is not True or not latest or latest >= cutoff:
        raise RuntimeError("P3.1 history manifest does not prove strict pre-cutoff behavior")
    if p31.get("final_week") != "not_run":
        raise RuntimeError("P3.1 manifest does not preserve the final-week boundary")
    p31_artifacts = p31.get("artifacts", {})
    required_p31 = {
        "candidates": paths.candidates,
        "histories": paths.histories,
        "users": paths.users,
    }
    p31_identity = {
        name: _declared_identity_matches(path, p31_artifacts.get(name, {}), f"P3.1 {name}")
        for name, path in required_p31.items()
    }
    p31_m4 = p31.get("student_embedding", {})
    p31_identity["m4_embeddings"] = _declared_identity_matches(
        paths.m4_embeddings, p31_m4, "P3.1 M4 Student embedding"
    )

    if p33.get("status") != "completed" or p33.get("final_week") != "not_run":
        raise RuntimeError("P3.3 manifest is not completed or violates the final-week boundary")
    training_cutoffs = [str(value) for value in p33.get("training_cutoffs", [])]
    if not training_cutoffs or any(value > cutoff for value in training_cutoffs):
        raise RuntimeError("a future P3.3 Student would be reused backwards")
    label = str(p33.get("label", ""))
    if label.startswith("training:") and label.partition(":")[2] != cutoff:
        raise RuntimeError("P3.3 training Student is not aligned to the feature cutoff")
    inference = p33.get("student_inference_audit", {})
    if (
        inference.get("content_only") is not True
        or inference.get("teacher_embedding_used_at_inference") is not False
    ):
        raise RuntimeError("P3.3 auxiliary embedding is not proven content-only at inference")
    p33_declared = _embedding_identity_from_manifest(p33)
    if p33_declared is None:
        raise RuntimeError("P3.3 manifest does not bind its catalog embedding")
    p33_identity = _declared_identity_matches(
        paths.p33_embeddings, p33_declared, "P3.3 auxiliary Student embedding"
    )

    m4_manifest_audit: dict[str, Any] | None = None
    if paths.m4_manifest is not None:
        m4 = _read_json(paths.m4_manifest)
        m4_training = [str(value) for value in m4.get("training_cutoffs", [])]
        if m4.get("status") != "completed" or any(value > cutoff for value in m4_training):
            raise RuntimeError("M4 manifest is incomplete or contains a future training cutoff")
        if m4.get("final_week") != "not_run":
            raise RuntimeError("M4 manifest does not preserve the final-week boundary")
        declared = _embedding_identity_from_manifest(m4)
        if declared is not None:
            identity = _declared_identity_matches(
                paths.m4_embeddings, declared, "M4 Student embedding manifest"
            )
        else:
            identity = {"observed": file_identity(paths.m4_embeddings), "declared": None, "checks": {}}
        outer_cutoff = m4.get("outer_validation_cutoff")
        if outer_cutoff is not None and str(outer_cutoff) != cutoff:
            raise RuntimeError("M4 outer Student is not aligned to the requested cutoff")
        m4_manifest_audit = {
            "manifest": file_identity(paths.m4_manifest),
            "training_cutoffs": m4_training,
            "embedding_identity": identity,
        }

    return {
        "cutoff": cutoff,
        "cutoff_strictly_before_final_week": True,
        "p31_manifest": file_identity(paths.p31_manifest),
        "p31_latest_behavior_before_cutoff": latest,
        "p31_asset_identity": p31_identity,
        "p33_manifest": file_identity(paths.p33_manifest),
        "p33_training_cutoffs": training_cutoffs,
        "p33_content_only_inference": True,
        "p33_embedding_identity": p33_identity,
        "m4_manifest_audit": m4_manifest_audit,
        "no_future_p33_student_reused_backwards": True,
        "passed": True,
    }


def _embedding_audit(embeddings: np.ndarray, catalog_size: int, label: str) -> dict[str, Any]:
    if catalog_size <= 0:
        raise ValueError("catalog must contain at least one item")
    if embeddings.ndim != 2 or len(embeddings) != catalog_size or embeddings.shape[1] <= 0:
        raise ValueError(f"{label} embedding shape does not match the catalog")
    norm_min = math.inf
    norm_max = -math.inf
    norm_sum = 0.0
    norm_rows = 0
    for start in range(0, catalog_size, 8192):
        block = np.asarray(embeddings[start : start + 8192], dtype=np.float32)
        if not np.all(np.isfinite(block)):
            raise ValueError(f"{label} embeddings contain non-finite values")
        norms = np.linalg.norm(block, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
        norm_sum += float(norms.sum(dtype=np.float64))
        norm_rows += len(norms)
    maximum_unit_norm_error = max(abs(norm_min - 1.0), abs(norm_max - 1.0))
    if maximum_unit_norm_error > 5e-3:
        raise ValueError(
            f"{label} embeddings are not unit-normalized; dot product would not be cosine"
        )
    return {
        "rows": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "finite": True,
        "norm_min": norm_min,
        "norm_mean": norm_sum / max(norm_rows, 1),
        "norm_max": norm_max,
        "maximum_unit_norm_error": maximum_unit_norm_error,
        "dot_product_is_cosine": True,
    }


def validate_frozen_inputs(
    *,
    candidates: Mapping[str, np.ndarray],
    histories: Mapping[str, np.ndarray],
    m4_embeddings: np.ndarray,
    p33_embeddings: np.ndarray,
    catalog_product_type: np.ndarray,
    catalog_garment_group: np.ndarray,
    interaction_counts: np.ndarray,
    expected_users: int | None = None,
) -> dict[str, Any]:
    """Validate frozen-candidate, recent-distinct-history, and catalog invariants."""

    required_candidates = {"user_index", "catalog_row", "rank", "coarse_score", "target"}
    required_histories = {"catalog_row", "days_since_purchase", "mask"}
    if not required_candidates.issubset(candidates):
        raise ValueError(f"candidate fields missing: {sorted(required_candidates - set(candidates))}")
    if not required_histories.issubset(histories):
        raise ValueError(f"history fields missing: {sorted(required_histories - set(histories))}")
    candidate_rows = len(candidates["user_index"])
    if candidate_rows == 0 or candidate_rows % K0:
        raise ValueError("candidate rows must be a positive multiple of frozen K0=200")
    if any(len(np.asarray(candidates[name])) != candidate_rows for name in required_candidates):
        raise ValueError("candidate columns have different row counts")
    history_rows = np.asarray(histories["catalog_row"])
    history_days = np.asarray(histories["days_since_purchase"])
    history_mask = np.asarray(histories["mask"], dtype=bool)
    if history_rows.ndim != 2 or history_rows.shape[1] != HISTORY_N:
        raise ValueError("history catalog rows must have frozen shape [users,20]")
    if history_days.shape != history_rows.shape or history_mask.shape != history_rows.shape:
        raise ValueError("history arrays have inconsistent shapes")
    user_count = len(history_rows)
    if expected_users is not None and user_count != expected_users:
        raise ValueError("users.csv row count differs from history user count")
    if not np.array_equal(history_mask, history_rows >= 0):
        raise ValueError("history mask must equal catalog_row >= 0")
    if np.any(history_rows[~history_mask] != -1):
        raise ValueError("padded history catalog rows must use the frozen -1 sentinel")

    catalog_size = len(interaction_counts)
    product_type = np.asarray(catalog_product_type)
    garment_group = np.asarray(catalog_garment_group)
    counts = np.asarray(interaction_counts)
    if product_type.shape != (catalog_size,) or garment_group.shape != (catalog_size,):
        raise ValueError("catalog attribute vectors must align with interaction counts")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("interaction counts must be finite and non-negative")
    if np.any(counts != np.floor(counts)):
        raise ValueError("interaction counts must be integer event counts")
    embedding_audit = {
        "m4": _embedding_audit(m4_embeddings, catalog_size, "M4"),
        "p33": _embedding_audit(p33_embeddings, catalog_size, "P3.3"),
    }

    users = np.asarray(candidates["user_index"], dtype=np.int64).reshape(-1, K0)
    items = np.asarray(candidates["catalog_row"], dtype=np.int64).reshape(-1, K0)
    ranks = np.asarray(candidates["rank"], dtype=np.int64).reshape(-1, K0)
    if not np.all(users == users[:, :1]):
        raise ValueError("candidate rows are not contiguous K0 groups by user")
    active_users = users[:, 0]
    if (
        np.any(active_users < 0)
        or np.any(active_users >= user_count)
        or np.any(np.diff(active_users) <= 0)
    ):
        raise ValueError("candidate user groups must be unique, ordered, and inside users.csv")
    if np.any(items < 0) or np.any(items >= catalog_size):
        raise ValueError("candidate catalog row is outside the embedding catalog")
    if not np.all(ranks == np.arange(1, K0 + 1, dtype=np.int64)[None, :]):
        raise ValueError("candidate rank must preserve exact raw 1..200 order")
    if not np.all(np.isfinite(np.asarray(candidates["coarse_score"], dtype=np.float32))):
        raise ValueError("frozen M4 coarse scores contain non-finite values")
    if not np.isin(np.asarray(candidates["target"]), (0, 1)).all():
        raise ValueError("candidate target must be binary next-week truth")
    candidate_counts = counts[items]
    if np.any(candidate_counts > 5):
        raise ValueError("candidate pool contains an item outside the frozen <=5-event universe")
    identity_multiplier = np.int64(catalog_size)
    identities = users.reshape(-1) * identity_multiplier + items.reshape(-1)
    if len(np.unique(identities)) != candidate_rows:
        raise ValueError("candidate user-item identities are not unique")

    bucket_totals = np.zeros(4, dtype=np.int64)
    valid_total = 0
    users_with_history = 0
    for start in range(0, user_count, 65536):
        rows = history_rows[start : start + 65536]
        days = history_days[start : start + 65536]
        mask = history_mask[start : start + 65536]
        if np.any(rows[mask] >= catalog_size):
            raise ValueError("history catalog row is outside the embedding catalog")
        if np.any(~np.isfinite(days[mask])) or np.any(days[mask] < 0):
            raise ValueError("valid history age is non-finite or negative")
        sorted_rows = np.sort(np.where(mask, rows, -1), axis=1)
        if np.any((sorted_rows[:, 1:] == sorted_rows[:, :-1]) & (sorted_rows[:, 1:] >= 0)):
            raise ValueError("history is not distinct by article within user")
        assigned = assign_age_buckets(days, mask)
        bucket_totals += np.asarray(
            [np.count_nonzero(assigned == bucket) for bucket in range(4)], dtype=np.int64
        )
        valid_total += int(np.count_nonzero(mask))
        users_with_history += int(np.count_nonzero(mask.any(axis=1)))
    if int(bucket_totals.sum()) != valid_total:
        raise RuntimeError("valid history rows are not conserved across age buckets")
    expected_active_users = np.flatnonzero(history_mask.any(axis=1)).astype(np.int64)
    if not np.array_equal(active_users, expected_active_users):
        raise ValueError("candidate-bearing users differ from frozen history-active users")

    return {
        "candidate_rows": int(candidate_rows),
        "candidate_user_groups": int(len(active_users)),
        "user_rows": int(user_count),
        "users_with_history": users_with_history,
        "history_rows": valid_total,
        "history_distinct_by_article_within_user": True,
        "history_n": HISTORY_N,
        "candidate_budget": K0,
        "candidate_rank_raw_1_200": True,
        "candidate_user_item_unique": True,
        "candidate_users_exactly_match_history_active_users": True,
        "candidate_items_all_interaction_count_le_5": True,
        "age_bucket_history_rows": {
            name: int(bucket_totals[index]) for index, name in enumerate(AGE_BUCKET_NAMES)
        },
        "age_bucket_mutually_exclusive": True,
        "every_valid_history_assigned_exactly_once": True,
        "embedding_audit": embedding_audit,
    }


def _normalized_profile(embeddings: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = valid.astype(np.float32)
    summed = np.einsum("bhd,bh->bd", embeddings, weights, optimize=True)
    counts = weights.sum(axis=1)
    mean = summed / np.maximum(counts[:, None], 1.0)
    norms = np.linalg.norm(mean, axis=1)
    available = (counts > 0) & np.isfinite(norms) & (norms > 1e-12)
    output = np.zeros_like(mean, dtype=np.float32)
    output[available] = mean[available] / norms[available, None]
    return output, available


def compute_user_state_batch(
    *,
    history_rows: np.ndarray,
    history_days: np.ndarray,
    history_mask: np.ndarray,
    m4_embeddings: np.ndarray,
    catalog_product_type: np.ndarray,
) -> np.ndarray:
    """Compute the fixed ten-dimensional, candidate-independent raw user state."""

    rows = np.asarray(history_rows, dtype=np.int64)
    days = np.asarray(history_days, dtype=np.float32)
    mask = np.asarray(history_mask, dtype=bool)
    if rows.ndim != 2 or rows.shape[1] != HISTORY_N:
        raise ValueError("history_rows must be [users,20]")
    if rows.shape != days.shape or rows.shape != mask.shape:
        raise ValueError("history batch arrays have different shapes")
    buckets = assign_age_buckets(days, mask)
    batch_size = len(rows)
    output = np.zeros((batch_size, len(USER_STATE_COLUMNS)), dtype=np.float32)
    for bucket in range(4):
        output[:, bucket] = np.count_nonzero(buckets == bucket, axis=1).astype(np.float32)
    valid_count = np.count_nonzero(mask, axis=1)
    minimum_age = np.min(np.where(mask, days, np.inf), axis=1)
    # Candidate-bearing users always have history. Keep the unused full-user
    # rows finite because the fixed ten-field contract has no separate
    # availability bit for days_since_last_purchase.
    output[:, 4] = np.where(np.isfinite(minimum_age), minimum_age, 0.0)
    recent = mask & (days <= 28.0)
    older = mask & (days > 28.0) & (days <= 84.0)
    output[:, 5] = np.divide(
        np.count_nonzero(recent, axis=1),
        valid_count,
        out=np.zeros(batch_size, dtype=np.float32),
        where=valid_count > 0,
    )

    safe_rows = np.maximum(rows, 0)
    history_embeddings = np.asarray(m4_embeddings[safe_rows], dtype=np.float32)
    history_embeddings[~mask] = 0.0
    recent_profile, recent_available = _normalized_profile(history_embeddings, recent)
    older_profile, older_available = _normalized_profile(history_embeddings, older)
    profile_available = recent_available & older_available
    profile_cosine = np.sum(recent_profile * older_profile, axis=1, dtype=np.float32)
    output[:, 6] = np.where(profile_available, profile_cosine, np.nan)
    output[:, 7] = profile_available.astype(np.float32)

    product_type = np.asarray(catalog_product_type)
    history_types = product_type[safe_rows]
    valid_types = recent & (history_types >= 0)
    same_type = history_types[:, :, None] == history_types[:, None, :]
    same_type &= valid_types[:, :, None] & valid_types[:, None, :]
    type_frequency = np.count_nonzero(same_type, axis=2)
    typed_count = np.count_nonzero(valid_types, axis=1)
    probability = np.divide(
        type_frequency,
        typed_count[:, None],
        out=np.ones_like(type_frequency, dtype=np.float64),
        where=valid_types & (typed_count[:, None] > 0),
    )
    entropy_terms = np.where(valid_types, -np.log(probability) / np.maximum(typed_count[:, None], 1), 0.0)
    output[:, 8] = np.sum(entropy_terms, axis=1).astype(np.float32)
    reciprocal_frequency = np.divide(
        1.0,
        type_frequency,
        out=np.zeros_like(type_frequency, dtype=np.float64),
        where=valid_types & (type_frequency > 0),
    )
    output[:, 9] = np.rint(np.sum(reciprocal_frequency, axis=1)).astype(np.float32)
    return output


def compute_candidate_state(
    *, candidates: Mapping[str, np.ndarray], interaction_counts: np.ndarray
) -> np.ndarray:
    """Compute the fixed five raw candidate-state fields without rank normalization."""

    scores = np.asarray(candidates["coarse_score"], dtype=np.float32)
    ranks = np.asarray(candidates["rank"], dtype=np.float32)
    items = np.asarray(candidates["catalog_row"], dtype=np.int64)
    if not (len(scores) == len(ranks) == len(items)):
        raise ValueError("candidate state columns have different lengths")
    counts = np.asarray(interaction_counts)[items]
    if np.any(counts < 0) or np.any(counts > 5):
        raise ValueError("candidate interaction count is outside the frozen 0..5 universe")
    return np.column_stack(
        [
            scores,
            ranks,
            counts.astype(np.float32),
            (counts == 0).astype(np.float32),
            ((counts >= 1) & (counts <= 5)).astype(np.float32),
        ]
    ).astype(np.float32, copy=False)


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _cosine_tensor(
    *,
    candidate_rows: np.ndarray,
    history_rows: np.ndarray,
    history_mask: np.ndarray,
    embeddings: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    safe_history = np.maximum(history_rows, 0)
    candidate_np = np.asarray(embeddings[candidate_rows], dtype=np.float32)
    history_np = np.asarray(embeddings[safe_history], dtype=np.float32)
    history_np[~history_mask] = 0.0
    if not np.all(np.isfinite(candidate_np)) or not np.all(np.isfinite(history_np)):
        raise ValueError("selected embedding rows contain non-finite values")
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else None
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            candidate_tensor = torch.from_numpy(candidate_np).to(device)
            history_tensor = torch.from_numpy(history_np).to(device)
            similarity = torch.einsum("bkd,bhd->bkh", candidate_tensor, history_tensor)
            result = similarity.cpu().numpy()
            del candidate_tensor, history_tensor, similarity
    finally:
        if previous_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    return result


def _bucket_cosine_summaries(
    similarity: np.ndarray, bucket_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch_size, candidate_count, history_count = similarity.shape
    maxima = np.full((batch_size, candidate_count, 4), np.nan, dtype=np.float32)
    top3_mean = np.full_like(maxima, np.nan)
    counts = np.empty((batch_size, 4), dtype=np.int32)
    take = min(3, history_count)
    for bucket in range(4):
        valid = bucket_index == bucket
        count = np.count_nonzero(valid, axis=1)
        counts[:, bucket] = count
        masked = np.where(valid[:, None, :], similarity, -np.inf)
        maximum = np.max(masked, axis=2)
        maxima[:, :, bucket] = np.where(count[:, None] > 0, maximum, np.nan)
        top = np.partition(masked, history_count - take, axis=2)[:, :, -take:]
        finite = np.isfinite(top)
        divisor = np.minimum(count, take)[:, None]
        mean = np.divide(
            np.where(finite, top, 0.0).sum(axis=2, dtype=np.float32),
            divisor,
            out=np.full((batch_size, candidate_count), np.nan, dtype=np.float32),
            where=divisor > 0,
        )
        top3_mean[:, :, bucket] = mean
    return maxima, top3_mean, counts


def _same_attribute_counts(
    *,
    candidate_rows: np.ndarray,
    history_rows: np.ndarray,
    bucket_index: np.ndarray,
    catalog_attribute: np.ndarray,
) -> np.ndarray:
    candidate_value = np.asarray(catalog_attribute)[candidate_rows]
    safe_history = np.maximum(history_rows, 0)
    history_value = np.asarray(catalog_attribute)[safe_history]
    output = np.empty((*candidate_rows.shape, 4), dtype=np.float32)
    for bucket in range(4):
        valid = bucket_index == bucket
        same = (
            (candidate_value[:, :, None] >= 0)
            & (history_value[:, None, :] >= 0)
            & (candidate_value[:, :, None] == history_value[:, None, :])
            & valid[:, None, :]
        )
        output[:, :, bucket] = np.count_nonzero(same, axis=2).astype(np.float32)
    return output


def compute_relation_feature_batch(
    *,
    candidate_rows: np.ndarray,
    m4_coarse_scores: np.ndarray,
    history_rows: np.ndarray,
    history_days: np.ndarray,
    history_mask: np.ndarray,
    m4_embeddings: np.ndarray,
    p33_embeddings: np.ndarray,
    catalog_product_type: np.ndarray,
    catalog_garment_group: np.ndarray,
    device: str | torch.device = "cpu",
    authoritative_p33_fixed_decay_scores: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute one user-group batch of raw M4 and P3.3 relation features.

    Candidate rows and scores are [batch_users, candidates_per_user]; history
    inputs are [batch_users, 20]. Missing-bucket cosine values remain NaN in
    this raw asset. The downstream train-only preprocessor is responsible for
    mapping those missing values to zero in standardized space.
    """

    candidate_rows = np.asarray(candidate_rows, dtype=np.int64)
    m4_scores = np.asarray(m4_coarse_scores, dtype=np.float32)
    history_rows = np.asarray(history_rows, dtype=np.int64)
    history_days = np.asarray(history_days, dtype=np.float32)
    history_mask = np.asarray(history_mask, dtype=bool)
    if candidate_rows.ndim != 2 or candidate_rows.shape != m4_scores.shape:
        raise ValueError("candidate rows and M4 scores must be matching 2D group matrices")
    if history_rows.shape != history_days.shape or history_rows.shape != history_mask.shape:
        raise ValueError("history batch arrays have different shapes")
    if len(candidate_rows) != len(history_rows) or history_rows.shape[1] != HISTORY_N:
        raise ValueError("candidate and frozen 20-item history batches do not align")
    resolved_device = _resolve_device(device)
    bucket_index = assign_age_buckets(history_days, history_mask)
    m4_similarity = _cosine_tensor(
        candidate_rows=candidate_rows,
        history_rows=history_rows,
        history_mask=history_mask,
        embeddings=m4_embeddings,
        device=resolved_device,
    )
    p33_similarity = _cosine_tensor(
        candidate_rows=candidate_rows,
        history_rows=history_rows,
        history_mask=history_mask,
        embeddings=p33_embeddings,
        device=resolved_device,
    )
    m4_max, m4_top3, bucket_counts = _bucket_cosine_summaries(m4_similarity, bucket_index)
    p33_max, p33_top3, p33_bucket_counts = _bucket_cosine_summaries(p33_similarity, bucket_index)
    if not np.array_equal(bucket_counts, p33_bucket_counts):
        raise RuntimeError("M4 and P3.3 relation summaries used different history buckets")
    product_counts = _same_attribute_counts(
        candidate_rows=candidate_rows,
        history_rows=history_rows,
        bucket_index=bucket_index,
        catalog_attribute=catalog_product_type,
    )
    garment_counts = _same_attribute_counts(
        candidate_rows=candidate_rows,
        history_rows=history_rows,
        bucket_index=bucket_index,
        catalog_attribute=catalog_garment_group,
    )
    batch_size, candidate_count = candidate_rows.shape
    m4_relation = np.empty((batch_size, candidate_count, 4, 6), dtype=np.float32)
    m4_relation[:, :, :, 0] = (bucket_counts > 0)[:, None, :]
    m4_relation[:, :, :, 1] = bucket_counts[:, None, :]
    m4_relation[:, :, :, 2] = m4_max
    m4_relation[:, :, :, 3] = m4_top3
    m4_relation[:, :, :, 4] = product_counts
    m4_relation[:, :, :, 5] = garment_counts
    p33_relation = np.stack(
        [p33_max, p33_top3, p33_max - m4_max, p33_top3 - m4_top3], axis=3
    ).astype(np.float32, copy=False)

    weights = np.power(0.5, history_days / P33_HALF_LIFE_DAYS).astype(np.float32)
    weighted = np.where(
        history_mask[:, None, :],
        p33_similarity * weights[:, None, :],
        -np.inf,
    )
    direct_p33_fixed = np.max(weighted, axis=2).astype(np.float32)
    direct_p33_fixed[~history_mask.any(axis=1), :] = np.nan
    if authoritative_p33_fixed_decay_scores is None:
        p33_fixed = direct_p33_fixed
    else:
        p33_fixed = np.asarray(authoritative_p33_fixed_decay_scores, dtype=np.float32)
        if p33_fixed.shape != candidate_rows.shape or not np.all(np.isfinite(p33_fixed)):
            raise ValueError("authoritative P3.3 fixed-decay scores are invalid")
    p33_global = np.stack([p33_fixed, p33_fixed - m4_scores], axis=2).astype(
        np.float32, copy=False
    )
    present = m4_relation[:, :, :, 0].astype(bool)
    if not np.array_equal(np.isnan(m4_relation[:, :, :, 2]), ~present):
        raise RuntimeError("M4 missing cosine does not match bucket_present")
    if not np.array_equal(np.isnan(p33_relation[:, :, :, 0]), ~present):
        raise RuntimeError("P3.3 missing cosine does not match bucket_present")
    return {
        "m4_relation": m4_relation,
        "p33_relation": p33_relation,
        "p33_global": p33_global,
        "direct_p33_fixed_decay": direct_p33_fixed,
        "bucket_counts": bucket_counts,
    }


def _staged_memmap(final_path: Path, *, shape: tuple[int, ...]) -> tuple[Path, np.memmap]:
    temporary = final_path.with_suffix(final_path.suffix + ".part")
    if final_path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite feature artifact: {final_path}")
    return temporary, np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=shape
    )


def _publish_memmap(temporary: Path, memmap: np.memmap, final_path: Path) -> None:
    memmap.flush()
    del memmap
    gc.collect()
    temporary.replace(final_path)


def build_raw_feature_assets(
    *,
    cutoff: str,
    candidates: Mapping[str, np.ndarray],
    histories: Mapping[str, np.ndarray],
    m4_embeddings: np.ndarray,
    p33_embeddings: np.ndarray,
    catalog_product_type: np.ndarray,
    catalog_garment_group: np.ndarray,
    interaction_counts: np.ndarray,
    output_dir: Path,
    source_artifacts: Mapping[str, Path] | None = None,
    lineage_audit: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    batch_users: int = 32,
    authoritative_p33_fixed_decay_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Materialize bounded-memory raw P3.7B features for one exact cutoff.

    This function deliberately does not normalize features. That operation must
    be fitted on the current outer chain's training cutoff data by the P3.7B
    training pipeline. A completed manifest is written only after all arrays
    have been closed, published, hashed, and checked.
    """

    started = time.perf_counter()
    _safe_cutoff(cutoff)
    if batch_users <= 0:
        raise ValueError("batch_users must be positive")
    input_audit = validate_frozen_inputs(
        candidates=candidates,
        histories=histories,
        m4_embeddings=m4_embeddings,
        p33_embeddings=p33_embeddings,
        catalog_product_type=catalog_product_type,
        catalog_garment_group=catalog_garment_group,
        interaction_counts=interaction_counts,
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite completed feature asset: {manifest_path}")
    candidate_rows_count = input_audit["candidate_rows"]
    user_count = input_audit["user_rows"]
    candidate_group_count = input_audit["candidate_user_groups"]
    final_paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    shapes = {
        "m4_relation": (candidate_rows_count, 4, len(M4_RELATION_COLUMNS)),
        "p33_relation": (candidate_rows_count, 4, len(P33_RELATION_COLUMNS)),
        "p33_global": (candidate_rows_count, len(P33_GLOBAL_COLUMNS)),
        "user_state": (user_count, len(USER_STATE_COLUMNS)),
        "candidate_state": (candidate_rows_count, len(CANDIDATE_STATE_COLUMNS)),
    }
    staged: dict[str, Path] = {}
    arrays: dict[str, np.memmap] = {}
    for name, shape in shapes.items():
        staged[name], arrays[name] = _staged_memmap(final_paths[name], shape=shape)

    history_rows = np.asarray(histories["catalog_row"], dtype=np.int32)
    history_days = np.asarray(histories["days_since_purchase"], dtype=np.float32)
    history_mask = np.asarray(histories["mask"], dtype=bool)
    for start in range(0, user_count, max(1024, batch_users * 64)):
        end = min(start + max(1024, batch_users * 64), user_count)
        arrays["user_state"][start:end] = compute_user_state_batch(
            history_rows=history_rows[start:end],
            history_days=history_days[start:end],
            history_mask=history_mask[start:end],
            m4_embeddings=m4_embeddings,
            catalog_product_type=catalog_product_type,
        )

    candidate_fields = {
        name: np.asarray(candidates[name]) for name in ("coarse_score", "rank", "catalog_row")
    }
    candidate_chunk = max(K0, batch_users * K0 * 8)
    for start in range(0, candidate_rows_count, candidate_chunk):
        end = min(start + candidate_chunk, candidate_rows_count)
        arrays["candidate_state"][start:end] = compute_candidate_state(
            candidates={name: values[start:end] for name, values in candidate_fields.items()},
            interaction_counts=interaction_counts,
        )

    grouped_users = np.asarray(candidates["user_index"], dtype=np.int32).reshape(-1, K0)[:, 0]
    grouped_items = np.asarray(candidates["catalog_row"], dtype=np.int32).reshape(-1, K0)
    grouped_m4_scores = np.asarray(candidates["coarse_score"], dtype=np.float32).reshape(-1, K0)
    authoritative = (
        None
        if authoritative_p33_fixed_decay_scores is None
        else np.asarray(authoritative_p33_fixed_decay_scores, dtype=np.float32).reshape(-1, K0)
    )
    direct_vs_authoritative_max_abs_error = 0.0
    bucket_candidate_rows = np.zeros(4, dtype=np.int64)
    for group_start in range(0, candidate_group_count, batch_users):
        group_end = min(group_start + batch_users, candidate_group_count)
        users = grouped_users[group_start:group_end]
        row_start = group_start * K0
        row_end = group_end * K0
        result = compute_relation_feature_batch(
            candidate_rows=grouped_items[group_start:group_end],
            m4_coarse_scores=grouped_m4_scores[group_start:group_end],
            history_rows=history_rows[users],
            history_days=history_days[users],
            history_mask=history_mask[users],
            m4_embeddings=m4_embeddings,
            p33_embeddings=p33_embeddings,
            catalog_product_type=catalog_product_type,
            catalog_garment_group=catalog_garment_group,
            device=device,
            authoritative_p33_fixed_decay_scores=(
                None if authoritative is None else authoritative[group_start:group_end]
            ),
        )
        arrays["m4_relation"][row_start:row_end] = result["m4_relation"].reshape(
            -1, 4, len(M4_RELATION_COLUMNS)
        )
        arrays["p33_relation"][row_start:row_end] = result["p33_relation"].reshape(
            -1, 4, len(P33_RELATION_COLUMNS)
        )
        arrays["p33_global"][row_start:row_end] = result["p33_global"].reshape(
            -1, len(P33_GLOBAL_COLUMNS)
        )
        bucket_candidate_rows += (
            (result["bucket_counts"] > 0).sum(axis=0).astype(np.int64) * K0
        )
        if authoritative is not None:
            difference = np.abs(
                result["direct_p33_fixed_decay"].astype(np.float64)
                - authoritative[group_start:group_end].astype(np.float64)
            )
            direct_vs_authoritative_max_abs_error = max(
                direct_vs_authoritative_max_abs_error, float(np.max(difference))
            )

    for name in OUTPUT_FILENAMES:
        _publish_memmap(staged[name], arrays.pop(name), final_paths[name])

    output_identities = {name: file_identity(path) for name, path in final_paths.items()}
    source_identities = (
        {name: file_identity(Path(path)) for name, path in source_artifacts.items()}
        if source_artifacts is not None
        else {}
    )
    manifest = {
        "schema_version": "phase3-p3.7b-raw-features-v1",
        "stage": "P3.7B",
        "status": "completed",
        "cutoff": cutoff,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "candidate_checkpoint": "exact frozen P3.1 M4-space Top200",
            "history": "cutoff-before most-recent 20 distinct purchased items",
            "age_buckets_days": {
                "0_7": "0 <= age <= 7",
                "8_28": "7 < age <= 28",
                "29_84": "28 < age <= 84",
                "over_84": "age > 84",
            },
            "age_buckets_are_soft_expert_inputs_not_a_hard_router": True,
            "missing_bucket_cosine_raw_value": "NaN",
            "missing_bucket_downstream_rule": (
                "fit statistics on current-chain training finite values only, then map missing to "
                "normalized zero while retaining bucket_present=0"
            ),
            "missing_catalog_attributes_compare_equal": False,
            "top3_mean": "mean of the best min(3, valid history count) cosine values inside the bucket",
            "p33_fixed_decay_formula": "max_h cosine(z_G(candidate),z_G(history_h))*2^(-age_h/28)",
            "candidate_rank_storage": "raw frozen integer rank 1..200; downstream fixed transform to [0,1]",
            "normalization_performed_here": False,
            "final_week": "not_run",
        },
        "columns": {
            "m4_relation": list(M4_RELATION_COLUMNS),
            "p33_relation": list(P33_RELATION_COLUMNS),
            "p33_global": list(P33_GLOBAL_COLUMNS),
            "user_state": list(USER_STATE_COLUMNS),
            "candidate_state": list(CANDIDATE_STATE_COLUMNS),
        },
        "shapes": {name: list(shape) for name, shape in shapes.items()},
        "dtype": "float32",
        "input_audit": input_audit,
        "lineage_audit": dict(lineage_audit) if lineage_audit is not None else None,
        "source_artifacts": source_identities,
        "build_audit": {
            "candidate_rows_with_bucket_present": {
                name: int(bucket_candidate_rows[index])
                for index, name in enumerate(AGE_BUCKET_NAMES)
            },
            "candidate_rows_with_bucket_missing": {
                name: int(candidate_rows_count - bucket_candidate_rows[index])
                for index, name in enumerate(AGE_BUCKET_NAMES)
            },
            "authoritative_p33_fixed_decay_scores_used": authoritative is not None,
            "p33_output_has_exact_authoritative_values": authoritative is not None,
            "direct_formula_vs_authoritative_max_abs_error": (
                direct_vs_authoritative_max_abs_error if authoritative is not None else None
            ),
            "bounded_memory_user_batch": max(1024, batch_users * 64),
            "bounded_memory_relation_batch_users": batch_users,
            "device": str(_resolve_device(device)),
        },
        "artifacts": output_identities,
        "elapsed_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    atomic_json(manifest_path, manifest)
    return manifest


def build_raw_feature_assets_from_paths(
    *,
    cutoff: str,
    paths: FrozenFeaturePaths,
    catalog_product_type: np.ndarray,
    catalog_garment_group: np.ndarray,
    interaction_counts: np.ndarray,
    output_dir: Path,
    device: str | torch.device = "cpu",
    batch_users: int = 32,
    authoritative_p33_fixed_decay_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Audited path-based entry point intended for formal P3.7B runs."""

    lineage = audit_cutoff_lineage(cutoff=cutoff, paths=paths)
    candidates = _load_npz(paths.candidates)
    histories = _load_npz(paths.histories)
    expected_users = _csv_data_rows(paths.users)
    if expected_users != len(histories["catalog_row"]):
        raise RuntimeError("users.csv and histories.npz have different user denominators")
    m4_embeddings = np.load(paths.m4_embeddings, mmap_mode="r")
    p33_embeddings = np.load(paths.p33_embeddings, mmap_mode="r")
    source_artifacts = {
        "candidates": paths.candidates,
        "histories": paths.histories,
        "users": paths.users,
        "m4_embeddings": paths.m4_embeddings,
        "p33_embeddings": paths.p33_embeddings,
        "p31_manifest": paths.p31_manifest,
        "p33_manifest": paths.p33_manifest,
    }
    if paths.m4_manifest is not None:
        source_artifacts["m4_manifest"] = paths.m4_manifest
    return build_raw_feature_assets(
        cutoff=cutoff,
        candidates=candidates,
        histories=histories,
        m4_embeddings=m4_embeddings,
        p33_embeddings=p33_embeddings,
        catalog_product_type=catalog_product_type,
        catalog_garment_group=catalog_garment_group,
        interaction_counts=interaction_counts,
        output_dir=output_dir,
        source_artifacts=source_artifacts,
        lineage_audit=lineage,
        device=device,
        batch_users=batch_users,
        authoritative_p33_fixed_decay_scores=authoritative_p33_fixed_decay_scores,
    )
