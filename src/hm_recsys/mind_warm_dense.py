"""Read-only, comparable MIND/history features for any frozen candidate pair.

The unit of every output row is one user--candidate item pair. Cosines use the
same frozen Item2Vec coordinate system for existing and newly retrieved items;
they do not encode which retrieval route admitted an item. No target/label is
accepted, and no training or candidate-generation entry point is called.
"""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from hm_recsys import mind_warm_mve as mve


AGE_BINS = (("0_7", 1, 7), ("8_28", 8, 28), ("29_84", 29, 84))
DENSE_FEATURES = [
    "dense_mind_max_cosine",
    "dense_mind_mean_cosine",
    "dense_mind_top2_gap",
    *[f"dense_history_{stat}_cosine_{name}" for name, _, _ in AGE_BINS for stat in ("max", "mean")],
    *[f"dense_history_support_{name}" for name, _, _ in AGE_BINS],
    "dense_i2v_vocab_count",
    "dense_item_missing",
    "dense_user_history_missing",
]


def _validate_keys(keys: pd.DataFrame) -> None:
    """Reject labels and ambiguous identities before reading model assets."""
    if not isinstance(keys, pd.DataFrame):
        raise TypeError("keys must be a pandas DataFrame")
    if len(keys.columns) != 2 or set(keys.columns) != {"customer_id", "article_id"}:
        raise ValueError("keys must contain exactly customer_id and article_id; targets/labels are forbidden")
    if keys.isna().any().any():
        raise ValueError("candidate keys cannot contain nulls")
    if keys.duplicated(["customer_id", "article_id"]).any():
        raise ValueError("candidate user-item keys must be unique")
    for column in ("customer_id", "article_id"):
        if not keys[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must be strings; item leading zeroes must be preserved")


def _user_features(
    candidate_vectors: torch.Tensor,
    candidate_present: torch.Tensor,
    interests: torch.Tensor,
    active_interests: torch.Tensor,
    history_vectors: torch.Tensor,
    history_ages: torch.Tensor,
    vocab_counts: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Pure bounded matrix calculation for one user's candidate block.

    History rows are distinct user-day-item events (not distinct items), with
    strictly positive age in days. A history-bin mean gives each retained event
    equal weight; its support is the number of usable events in that bin.
    Interest means use only active, nonzero interest vectors. Top-two gap is
    undefined (NaN) when fewer than two interests are active. Missing item or
    history vectors yield NaN cosines, never a misleading neutral zero score.
    """
    if candidate_vectors.ndim != 2 or interests.ndim != 2 or history_vectors.ndim != 2:
        raise ValueError("vectors must be two-dimensional")
    count, dimension = candidate_vectors.shape
    if interests.shape[1] != dimension or history_vectors.shape[1] != dimension:
        raise ValueError("all vectors must share their frozen item-space dimension")
    if candidate_present.shape != (count,) or vocab_counts.shape != (count,):
        raise ValueError("candidate masks/counts must have one value per candidate")
    if active_interests.shape != (len(interests),) or history_ages.shape != (len(history_vectors),):
        raise ValueError("interest and history masks do not match their vectors")
    if bool(((history_ages <= 0) | (history_ages > 84)).any()):
        raise ValueError("history ages must be cutoff-safe positive days in [1,84]")
    if not bool(torch.isfinite(candidate_vectors).all()):
        raise ValueError("candidate vectors must be finite; use candidate_present to mark missing items")
    if not bool(torch.isfinite(interests).all() & torch.isfinite(history_vectors).all()):
        raise ValueError("interest/history vectors must be finite")

    device = candidate_vectors.device
    result = {
        name: torch.full((count,), float("nan"), dtype=torch.float32, device=device)
        for name in DENSE_FEATURES
    }
    usable_items = candidate_present.bool() & (candidate_vectors.norm(dim=1) > 0)
    usable_history = history_vectors.norm(dim=1) > 0
    history_vectors = history_vectors[usable_history]
    history_ages = history_ages[usable_history]
    no_history = len(history_vectors) == 0
    result["dense_item_missing"] = (~usable_items).to(torch.float32)
    result["dense_user_history_missing"].fill_(float(no_history))
    result["dense_i2v_vocab_count"] = torch.where(
        candidate_present.bool(), vocab_counts.to(torch.float32), float("nan")
    )
    candidate_unit = F.normalize(candidate_vectors, dim=1)

    usable_interests = active_interests.bool() & (interests.norm(dim=1) > 0)
    if not no_history and bool(usable_interests.any()) and bool(usable_items.any()):
        scores = candidate_unit[usable_items] @ F.normalize(interests[usable_interests], dim=1).T
        result["dense_mind_max_cosine"][usable_items] = scores.max(dim=1).values
        result["dense_mind_mean_cosine"][usable_items] = scores.mean(dim=1)
        if scores.shape[1] >= 2:
            top = scores.topk(2, dim=1).values
            result["dense_mind_top2_gap"][usable_items] = top[:, 0] - top[:, 1]

    for name, lower, upper in AGE_BINS:
        mask = (history_ages >= lower) & (history_ages <= upper)
        support = int(mask.sum())
        result[f"dense_history_support_{name}"].fill_(float(support))
        if support and bool(usable_items.any()):
            scores = candidate_unit[usable_items] @ F.normalize(history_vectors[mask], dim=1).T
            result[f"dense_history_max_cosine_{name}"][usable_items] = scores.max(dim=1).values
            result[f"dense_history_mean_cosine_{name}"][usable_items] = scores.mean(dim=1)
    return {name: values.detach().cpu().numpy() for name, values in result.items()}


def build_dense_features(
    cutoff: str, keys: pd.DataFrame, device: str = "cpu"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score existing/new pairs identically from a completed frozen MIND model.

    This function reads only model metadata/state, frozen item vectors, the
    existing evaluation-user identities and transactions strictly before the
    cutoff. It never fits a model, reads future labels or writes an artifact.
    CPU is the default; an explicit CUDA device is required for GPU execution.
    Work is bounded to 64 users for encoding and 256 candidates for scoring.
    """
    mve._guard(cutoff)
    _validate_keys(keys)
    selected_device = torch.device(device)
    if selected_device.type not in ("cpu", "cuda"):
        raise ValueError("only cpu or an explicitly requested cuda device is supported")
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA request cannot run because CUDA is unavailable")

    started = time.perf_counter()
    variant = "mind_learned_age"
    folder = mve.ARTIFACT_ROOT / cutoff / variant
    metadata_path = folder / "MODEL.json"
    checkpoint_path = folder / "model.pt"
    if not metadata_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"completed frozen MIND model required: {folder}; training is forbidden here")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    params = metadata["params"]
    if (
        metadata.get("cutoff") != cutoff
        or metadata.get("variant") != variant
        or metadata.get("schema") != mve.MODEL_SCHEMA
        or not metadata.get("all_registered_epochs_completed")
        or metadata.get("outer_labels_used") is not False
        or metadata.get("final_week") != "not_run"
        or params.get("embedding_trainable") is not False
        or params.get("maximum_interests") != 3
        or params.get("time_mode") != "learned_buckets"
    ):
        raise ValueError("frozen MIND model metadata does not satisfy cutoff/variant/completion invariants")
    items, vectors, source = mve.load_item_table(cutoff)
    if metadata.get("item_source", {}).get("vocabulary_items") != len(items):
        raise ValueError("frozen model and Item2Vec vocabulary sizes disagree")
    if items["article_id"].duplicated().any():
        raise ValueError("frozen item vocabulary contains duplicate identities")
    if "token_count" not in items:
        raise ValueError("frozen item table is missing token_count")
    users, history, ages, lengths, history_evidence = mve.prepare_query_histories(cutoff, items)
    if users["customer_id"].duplicated().any():
        raise ValueError("query users must be unique")
    user_rows = pd.Index(users["customer_id"]).get_indexer(keys["customer_id"])
    if np.any(user_rows < 0):
        raise ValueError("all candidate users must belong to the frozen cutoff query population")
    item_rows = pd.Index(items["article_id"]).get_indexer(keys["article_id"])
    token_counts = pd.to_numeric(items["token_count"], errors="raise").to_numpy(np.float32)
    if not np.isfinite(token_counts).all() or np.any(token_counts < 0):
        raise ValueError("vocabulary token counts must be finite and non-negative")

    output = {name: np.full(len(keys), np.nan, np.float32) for name in DENSE_FEATURES}
    original_threads = torch.get_num_threads()
    if selected_device.type == "cpu":
        torch.set_num_threads(min(4, original_threads))
    try:
        # Direct state-only loading is deliberate: mve._load_model may train.
        # Constructor noise is overwritten by the checkpoint; preserve the
        # caller's random stream so feature extraction cannot steer later fits.
        with torch.random.fork_rng(devices=[]):
            model = mve.MindEncoder(3, "learned_buckets", dimension=int(params["history_dimension"]))
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        model.to(selected_device).eval()
        frozen_vectors = torch.from_numpy(vectors).to(selected_device)
        sorted_rows = np.argsort(user_rows, kind="stable")
        sorted_users = user_rows[sorted_rows]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_users)) + 1, len(keys)]
        if not len(keys):
            boundaries = np.array([0], dtype=np.int64)
        group_count = len(boundaries) - 1
        with torch.inference_mode():
            for begin in range(0, group_count, 64):
                group_ids = np.arange(begin, min(begin + 64, group_count))
                encoded_rows = sorted_users[boundaries[group_ids]]
                interests, active = model(
                    frozen_vectors,
                    torch.as_tensor(history[encoded_rows], device=selected_device),
                    torch.as_tensor(ages[encoded_rows], device=selected_device),
                    torch.as_tensor(lengths[encoded_rows], device=selected_device),
                    int(params["routing_iterations"]),
                )
                for local, group in enumerate(group_ids):
                    query = int(encoded_rows[local])
                    length = int(lengths[query])
                    history_ids = history[query, :length]
                    if np.any(history_ids < 0) or np.any(history_ids >= len(items)):
                        raise ValueError("valid query history refers to an invalid item row")
                    hist_vectors = frozen_vectors[torch.as_tensor(history_ids, dtype=torch.long, device=selected_device)]
                    hist_ages = torch.as_tensor(ages[query, :length], device=selected_device)
                    for start in range(int(boundaries[group]), int(boundaries[group + 1]), 256):
                        rows = sorted_rows[start : min(start + 256, int(boundaries[group + 1]))]
                        ids = item_rows[rows]
                        present = ids >= 0
                        safe_ids = np.maximum(ids, 0)
                        values = _user_features(
                            frozen_vectors[torch.as_tensor(safe_ids, dtype=torch.long, device=selected_device)],
                            torch.as_tensor(present, device=selected_device),
                            interests[local], active[local], hist_vectors, hist_ages,
                            torch.as_tensor(token_counts[safe_ids], device=selected_device),
                        )
                        for name, block in values.items():
                            output[name][rows] = block
    finally:
        if selected_device.type == "cpu":
            torch.set_num_threads(original_threads)

    result = keys.copy()
    for name, values in output.items():
        result[name] = values
    if not result[list(keys.columns)].equals(keys) or len(result) != len(keys):
        raise AssertionError("dense feature calculation changed candidate identities or row order")
    evidence = {
        "schema": "mind-warm-dense-features-v1",
        "cutoff": cutoff,
        "variant": variant,
        "model_metadata": str(metadata_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "item_source": source,
        "query_history": history_evidence,
        "candidate_rows": len(keys),
        "candidate_users": int(keys["customer_id"].nunique()),
        "item_missing_rows": int(result["dense_item_missing"].sum()),
        "user_history_missing_rows": int(result["dense_user_history_missing"].sum()),
        "features": DENSE_FEATURES,
        "age_bins_days": {name: [lower, upper] for name, lower, upper in AGE_BINS},
        "history_support_unit": "distinct retained user-day-item events in the age bin, maximum 50 across bins",
        "history_cosine_mean_denominator": "usable retained history events in the stated age bin",
        "interest_cosine_mean_denominator": "active nonzero interests for that user",
        "top2_gap_if_single_interest": "NaN",
        "unknown_item_or_history_cosine": "NaN",
        "same_features_for_every_candidate_source": True,
        "exact_key_and_row_order_conservation": True,
        "training_called": False,
        "labels_read": False,
        "device": str(selected_device),
        "cpu_threads": min(4, original_threads) if selected_device.type == "cpu" else None,
        "encoding_user_batch": 64,
        "candidate_block": 256,
        "seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    return result, evidence
