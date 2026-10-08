from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .m3 import _peak_working_set_bytes
from .m33 import ROLLING_PROTOCOL
from .m4_contract import FINAL_CUTOFF, atomic_json, file_identity
from .p3 import _cutoff_context, _ordering_metrics


RUN_ID = "phase3-p3.7a-v1-representation-retrieval-2x2-diagnostic"
STAGE = "P3.7A"
K0 = 200
HISTORY_N = 20
HALF_LIFE_DAYS = 28.0
K_VALUES = (5, 10, 20, 50, 100, 200)
WINDOWS = (
    "winter_20200122",
    "spring_20200318",
    "early_summer_20200624",
    "late_summer_20200819",
)
SEGMENTS = ("strict_cold", "sparse_1_5", "cold_universe")
VARIANTS = {
    "A00": ("R_M", "S_M"),
    "A01": ("R_M", "S_G"),
    "A10": ("R_G", "S_M"),
    "A11": ("R_G", "S_G"),
}
TRAINING_ALLOWED = False
SCORE_RECONSTRUCTION_ATOL = 1e-5


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_cutoff(cutoff: str) -> None:
    if cutoff >= FINAL_CUTOFF:
        raise RuntimeError(f"P3.7A refuses final or later cutoff: {cutoff}")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _require_identity(path: Path, declared: dict[str, Any], label: str) -> dict[str, Any]:
    observed = file_identity(path)
    checks = {
        "bytes": int(observed["bytes"]) == int(declared["bytes"]),
        "sha256": observed["sha256"] == declared["sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"{label} identity drift: {checks}")
    return {"observed": observed, "declared": declared, "checks": checks}


def _rank_by_score(users: np.ndarray, items: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if not (len(users) == len(items) == len(scores)):
        raise ValueError("rank inputs have different lengths")
    ranks = np.empty(len(users), dtype=np.int32)
    cursor = 0
    while cursor < len(users):
        end = cursor + 1
        while end < len(users) and users[end] == users[cursor]:
            end += 1
        order = np.lexsort((items[cursor:end], -scores[cursor:end]))
        ranks[cursor + order] = np.arange(1, end - cursor + 1, dtype=np.int32)
        cursor = end
    return ranks


def _candidate_matrices(candidates: dict[str, np.ndarray], user_count: int) -> tuple[np.ndarray, np.ndarray]:
    required = {"user_index", "catalog_row", "rank", "coarse_score", "target"}
    if not required.issubset(candidates):
        raise RuntimeError(f"candidate asset missing fields: {sorted(required - set(candidates))}")
    rows = len(candidates["user_index"])
    if rows == 0 or rows % K0:
        raise RuntimeError("candidate rows are not positive multiples of 200")
    if any(len(value) != rows for value in candidates.values()):
        raise RuntimeError("candidate columns have inconsistent lengths")
    group_users = candidates["user_index"].reshape(-1, K0)
    group_items = candidates["catalog_row"].reshape(-1, K0)
    group_ranks = candidates["rank"].reshape(-1, K0)
    if not np.all(group_users == group_users[:, :1]):
        raise RuntimeError("candidate rows are not contiguous by user")
    active_users = group_users[:, 0].astype(np.int32)
    if np.any(active_users < 0) or np.any(active_users >= user_count):
        raise RuntimeError("candidate user index is outside users.csv")
    if len(np.unique(active_users)) != len(active_users) or np.any(np.diff(active_users) <= 0):
        raise RuntimeError("candidate user groups are not unique and strictly ordered")
    expected = np.arange(1, K0 + 1, dtype=group_ranks.dtype)
    if not np.all(group_ranks == expected[None, :]):
        raise RuntimeError("authoritative candidate rank is not continuous 1..200")
    identities = candidates["user_index"].astype(np.int64) * 105_542 + candidates["catalog_row"]
    if len(np.unique(identities)) != rows:
        raise RuntimeError("candidate user-item identity is not unique")
    return active_users, group_items.astype(np.int32, copy=False)


def _score_candidates(
    candidates: dict[str, np.ndarray],
    histories: dict[str, np.ndarray],
    embeddings: np.ndarray,
    *,
    device: torch.device,
    batch_users: int = 64,
) -> np.ndarray:
    active_users, item_matrix = _candidate_matrices(candidates, len(histories["catalog_row"]))
    history_rows = histories["catalog_row"].astype(np.int32, copy=False)
    history_days = histories["days_since_purchase"].astype(np.float32, copy=False)
    history_mask = histories["mask"].astype(bool, copy=False)
    scores = np.empty(item_matrix.shape, dtype=np.float32)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        for start in range(0, len(active_users), batch_users):
            end = min(start + batch_users, len(active_users))
            users = active_users[start:end]
            candidate_rows = item_matrix[start:end]
            h_rows = history_rows[users]
            h_mask = history_mask[users]
            safe_h_rows = np.maximum(h_rows, 0)
            candidate_np = np.asarray(embeddings[candidate_rows], dtype=np.float32)
            history_np = np.asarray(embeddings[safe_h_rows], dtype=np.float32)
            candidate_tensor = torch.from_numpy(candidate_np).to(device)
            history_tensor = torch.from_numpy(history_np).to(device)
            similarity = torch.einsum("bkd,bhd->bkh", candidate_tensor, history_tensor)
            weights = np.power(0.5, history_days[users] / HALF_LIFE_DAYS).astype(np.float32)
            weighted = similarity * torch.from_numpy(weights).to(device)[:, None, :]
            mask_tensor = torch.from_numpy(h_mask).to(device)[:, None, :]
            weighted = torch.where(mask_tensor, weighted, torch.full_like(weighted, -torch.inf))
            scores[start:end] = weighted.max(dim=2).values.cpu().numpy()
            del candidate_tensor, history_tensor, similarity, weighted, mask_tensor
    return scores.reshape(-1)


def _score_universe_for_frozen_sets(
    candidate_sets: dict[str, dict[str, np.ndarray]],
    histories: dict[str, np.ndarray],
    embeddings: np.ndarray,
    candidate_universe: np.ndarray,
    *,
    authoritative_set: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reproduce P3.1's full-universe kernel and extract scores for frozen sets."""
    active_users: np.ndarray | None = None
    item_matrices: dict[str, np.ndarray] = {}
    for name, candidates in candidate_sets.items():
        users, items = _candidate_matrices(candidates, len(histories["catalog_row"]))
        if active_users is None:
            active_users = users
        elif not np.array_equal(active_users, users):
            raise RuntimeError("frozen candidate sets have different active users")
        item_matrices[name] = items
    assert active_users is not None
    if authoritative_set not in candidate_sets:
        raise KeyError(authoritative_set)
    catalog_to_local = np.full(len(embeddings), -1, dtype=np.int32)
    catalog_to_local[candidate_universe] = np.arange(len(candidate_universe), dtype=np.int32)
    local_matrices = {name: catalog_to_local[items] for name, items in item_matrices.items()}
    if any(np.any(local < 0) for local in local_matrices.values()):
        raise RuntimeError("frozen candidate lies outside reconstructed cold/sparse universe")
    outputs = {name: np.empty(items.shape, dtype=np.float32) for name, items in item_matrices.items()}
    reconstructed_items = np.empty(item_matrices[authoritative_set].shape, dtype=np.int32)
    reconstructed_scores = np.empty(item_matrices[authoritative_set].shape, dtype=np.float32)
    history_rows = histories["catalog_row"].astype(np.int32, copy=False)
    history_days = histories["days_since_purchase"].astype(np.float32, copy=False)
    history_mask = histories["mask"].astype(bool, copy=False)
    candidate_np = np.asarray(embeddings[candidate_universe], dtype=np.float32)
    candidate_tensor = torch.from_numpy(candidate_np).to(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        for start in range(0, len(active_users), 48):
            end = min(start + 48, len(active_users))
            users = active_users[start:end]
            rows = history_rows[users]
            mask = rows >= 0
            flat = rows.reshape(-1)
            history_np = np.zeros((len(flat), embeddings.shape[1]), dtype=np.float32)
            valid = flat >= 0
            history_np[valid] = np.asarray(embeddings[flat[valid]], dtype=np.float32)
            history_tensor = torch.from_numpy(
                history_np.reshape(len(users), HISTORY_N, embeddings.shape[1])
            ).to(device)
            similarities = torch.einsum("bhd,cd->bhc", history_tensor, candidate_tensor)
            weights = np.power(0.5, history_days[users] / HALF_LIFE_DAYS).astype(np.float32)
            weighted = torch.where(
                torch.from_numpy(mask).to(device)[:, :, None],
                similarities * torch.from_numpy(weights).to(device)[:, :, None],
                torch.full_like(similarities, -torch.inf),
            )
            coarse = weighted.max(dim=1).values
            for name, local in local_matrices.items():
                gather = torch.from_numpy(local[start:end].astype(np.int64, copy=False)).to(device)
                outputs[name][start:end] = torch.gather(coarse, 1, gather).cpu().numpy()
            values, local_indices = torch.topk(coarse, k=K0, dim=1, largest=True, sorted=False)
            values_np = values.cpu().numpy()
            local_np = local_indices.cpu().numpy()
            for local_user in range(len(users)):
                ordering = sorted(
                    zip(local_np[local_user].tolist(), values_np[local_user].tolist()),
                    key=lambda value: (-float(value[1]), int(candidate_universe[int(value[0])])),
                )
                reconstructed_items[start + local_user] = np.asarray(
                    [int(candidate_universe[int(local_row)]) for local_row, _score in ordering],
                    dtype=np.int32,
                )
                reconstructed_scores[start + local_user] = np.asarray(
                    [float(score) for _local_row, score in ordering], dtype=np.float32
                )
            del history_tensor, similarities, weighted, coarse, values, local_indices
    authoritative = candidate_sets[authoritative_set]
    authoritative_items = item_matrices[authoritative_set]
    authoritative_scores = authoritative["coarse_score"].reshape(-1, K0).astype(np.float32)
    audit = {
        "candidate_universe_items": int(len(candidate_universe)),
        "active_users": int(len(active_users)),
        "full_universe_kernel": "torch.einsum bhd,cd->bhc with user batch 48, matching P3.1",
        "candidate_identity_exact_parity": bool(np.array_equal(reconstructed_items, authoritative_items)),
        "rank_continuous_1_200": True,
        "score_max_abs_error": float(
            np.max(np.abs(reconstructed_scores.astype(np.float64) - authoritative_scores.astype(np.float64)))
        ),
        "score_atol": SCORE_RECONSTRUCTION_ATOL,
    }
    audit["score_within_atol"] = audit["score_max_abs_error"] <= SCORE_RECONSTRUCTION_ATOL
    audit["passed"] = audit["candidate_identity_exact_parity"] and audit["score_within_atol"]
    if not audit["passed"]:
        raise RuntimeError(f"full-universe authoritative reconstruction failed: {audit}")
    del candidate_tensor
    return {name: value.reshape(-1) for name, value in outputs.items()}, audit


def _validate_candidate_asset(
    candidates: dict[str, np.ndarray],
    histories: dict[str, np.ndarray],
    truth: dict[int, set[int]],
    counts: np.ndarray,
) -> dict[str, Any]:
    active_users, item_matrix = _candidate_matrices(candidates, len(histories["catalog_row"]))
    history_active = np.flatnonzero(histories["mask"].astype(bool).any(axis=1)).astype(np.int32)
    if not np.array_equal(active_users, history_active):
        raise RuntimeError("candidate active users differ from frozen history-active users")
    if np.any(counts[candidates["catalog_row"]] > 5):
        raise RuntimeError("candidate outside frozen cold/sparse universe")
    expected_target = np.zeros(len(candidates["target"]), dtype=np.uint8)
    for group_index, user_index in enumerate(active_users.tolist()):
        relevant = truth.get(int(user_index), set())
        if relevant:
            start = group_index * K0
            expected_target[start : start + K0] = np.isin(
                item_matrix[group_index], np.fromiter(relevant, dtype=np.int32)
            ).astype(np.uint8)
    if not np.array_equal(expected_target, candidates["target"].astype(np.uint8)):
        raise RuntimeError("candidate target labels differ from frozen next-seven-day truth")
    return {
        "candidate_rows": int(len(candidates["user_index"])),
        "active_users": int(len(active_users)),
        "inactive_users": int(len(histories["catalog_row"]) - len(active_users)),
        "user_item_unique": True,
        "exact_budget_200": True,
        "rank_continuous_1_200": True,
        "active_users_match_histories": True,
        "all_candidates_in_events_le_5_universe": True,
        "target_labels_match_truth": True,
    }


def _formula_audit(
    candidates: dict[str, np.ndarray], reconstructed_scores: np.ndarray
) -> dict[str, Any]:
    reconstructed_rank = _rank_by_score(
        candidates["user_index"], candidates["catalog_row"], reconstructed_scores
    )
    max_abs_error = float(
        np.max(np.abs(reconstructed_scores.astype(np.float64) - candidates["coarse_score"].astype(np.float64)))
    )
    result = {
        "score_rows": int(len(reconstructed_scores)),
        "score_max_abs_error": max_abs_error,
        "score_atol": SCORE_RECONSTRUCTION_ATOL,
        "score_within_atol": max_abs_error <= SCORE_RECONSTRUCTION_ATOL,
        "rank_exact_parity": bool(np.array_equal(reconstructed_rank, candidates["rank"])),
    }
    result["passed"] = result["score_within_atol"] and result["rank_exact_parity"]
    if not result["passed"]:
        raise RuntimeError(f"authoritative score formula reconstruction failed: {result}")
    return result


def _segment_name(event_count: int) -> str | None:
    if event_count == 0:
        return "strict_cold"
    if 1 <= event_count <= 5:
        return "sparse_1_5"
    return None


def _candidate_overlap(
    m4_candidates: dict[str, np.ndarray],
    p33_candidates: dict[str, np.ndarray],
    counts: np.ndarray,
) -> dict[str, Any]:
    m_users, m_items = _candidate_matrices(m4_candidates, int(max(m4_candidates["user_index"])) + 1)
    g_users, g_items = _candidate_matrices(p33_candidates, int(max(p33_candidates["user_index"])) + 1)
    if not np.array_equal(m_users, g_users):
        raise RuntimeError("M4 and P3.3 active-user identities differ")
    intersections: list[int] = []
    changed = {
        "strict_cold": {"m4_only_candidate_rows": 0, "p33_only_candidate_rows": 0},
        "sparse_1_5": {"m4_only_candidate_rows": 0, "p33_only_candidate_rows": 0},
    }
    for left, right in zip(m_items, g_items):
        common = np.intersect1d(left, right, assume_unique=True)
        intersections.append(int(len(common)))
        left_only = np.setdiff1d(left, right, assume_unique=True)
        right_only = np.setdiff1d(right, left, assume_unique=True)
        for segment, mask in (
            ("strict_cold", counts[left_only] == 0),
            ("sparse_1_5", (counts[left_only] >= 1) & (counts[left_only] <= 5)),
        ):
            changed[segment]["m4_only_candidate_rows"] += int(np.count_nonzero(mask))
        for segment, mask in (
            ("strict_cold", counts[right_only] == 0),
            ("sparse_1_5", (counts[right_only] >= 1) & (counts[right_only] <= 5)),
        ):
            changed[segment]["p33_only_candidate_rows"] += int(np.count_nonzero(mask))
    intersection_array = np.asarray(intersections, dtype=np.int32)
    jaccard = intersection_array / (2 * K0 - intersection_array)
    intersection_rows = int(intersection_array.sum())
    m4_only = int(len(m4_candidates["user_index"]) - intersection_rows)
    p33_only = int(len(p33_candidates["user_index"]) - intersection_rows)
    union_rows = intersection_rows + m4_only + p33_only
    changed["cold_universe"] = {
        "m4_only_candidate_rows": sum(row["m4_only_candidate_rows"] for row in changed.values()),
        "p33_only_candidate_rows": sum(row["p33_only_candidate_rows"] for row in changed.values()),
    }
    return {
        "active_users": int(len(m_users)),
        "per_user_jaccard": {
            "mean": float(np.mean(jaccard)),
            "median": float(np.median(jaccard)),
            "minimum": float(np.min(jaccard)),
            "maximum": float(np.max(jaccard)),
        },
        "per_user_intersection_size": {
            "mean": float(np.mean(intersection_array)),
            "median": float(np.median(intersection_array)),
            "minimum": int(np.min(intersection_array)),
            "maximum": int(np.max(intersection_array)),
        },
        "candidate_rows": {
            "m4_total": int(len(m4_candidates["user_index"])),
            "p33_total": int(len(p33_candidates["user_index"])),
            "intersection": intersection_rows,
            "m4_only": m4_only,
            "p33_only": p33_only,
            "union": union_rows,
        },
        "changed_candidate_rows_by_segment": changed,
        "conservation_passed": bool(m4_only == p33_only and union_rows == intersection_rows + m4_only + p33_only),
    }


def _truth_flow(
    m4_candidates: dict[str, np.ndarray],
    p33_candidates: dict[str, np.ndarray],
    truth: dict[int, set[int]],
    counts: np.ndarray,
    user_count: int,
    overlap: dict[str, Any],
) -> dict[str, Any]:
    m_users, m_items = _candidate_matrices(m4_candidates, user_count)
    g_users, g_items = _candidate_matrices(p33_candidates, user_count)
    m_lookup = {int(user): set(items.tolist()) for user, items in zip(m_users, m_items)}
    g_lookup = {int(user): set(items.tolist()) for user, items in zip(g_users, g_items)}
    categories = ("in_both", "only_in_R_M", "only_in_R_G", "in_neither")
    result = {segment: {category: 0 for category in categories} for segment in SEGMENTS}
    for user_index in range(user_count):
        left = m_lookup.get(user_index, set())
        right = g_lookup.get(user_index, set())
        for item in truth.get(user_index, set()):
            segment = _segment_name(int(counts[item]))
            if segment is None:
                continue
            in_left = item in left
            in_right = item in right
            category = (
                "in_both" if in_left and in_right else
                "only_in_R_M" if in_left else
                "only_in_R_G" if in_right else
                "in_neither"
            )
            result[segment][category] += 1
            result["cold_universe"][category] += 1
    for segment in SEGMENTS:
        row = result[segment]
        row["truth_pairs"] = int(sum(row[category] for category in categories))
        row["gained_truth_pairs"] = int(row["only_in_R_G"])
        row["lost_truth_pairs"] = int(row["only_in_R_M"])
        row["net_truth_pairs"] = int(row["only_in_R_G"] - row["only_in_R_M"])
        changed = overlap["changed_candidate_rows_by_segment"][segment]
        row["gained_truth_pairs_per_1k_p33_only_candidate_rows"] = (
            1000.0 * row["gained_truth_pairs"] / max(changed["p33_only_candidate_rows"], 1)
        )
        row["lost_truth_pairs_per_1k_m4_only_candidate_rows"] = (
            1000.0 * row["lost_truth_pairs"] / max(changed["m4_only_candidate_rows"], 1)
        )
        row["mutually_exclusive_and_exhaustive"] = True
    return result


def _union_oracle(
    m4_candidates: dict[str, np.ndarray],
    p33_candidates: dict[str, np.ndarray],
    truth: dict[int, set[int]],
    counts: np.ndarray,
    user_count: int,
    overlap: dict[str, Any],
    truth_flow: dict[str, Any],
) -> dict[str, Any]:
    m_users, m_items = _candidate_matrices(m4_candidates, user_count)
    g_users, g_items = _candidate_matrices(p33_candidates, user_count)
    union_lookup = {
        int(user): set(left.tolist()).union(right.tolist())
        for user, left, right in zip(m_users, m_items, g_items)
    }
    segments: dict[str, Any] = {}
    for segment in SEGMENTS:
        recalls: list[float] = []
        hits: list[float] = []
        for user_index in range(user_count):
            relevant = {
                item for item in truth.get(user_index, set())
                if (
                    segment == "cold_universe" and counts[item] <= 5
                    or segment == "strict_cold" and counts[item] == 0
                    or segment == "sparse_1_5" and 1 <= counts[item] <= 5
                )
            }
            if not relevant:
                continue
            matched = len(relevant.intersection(union_lookup.get(user_index, set())))
            recalls.append(matched / len(relevant))
            hits.append(float(matched > 0))
        flow = truth_flow[segment]
        changed = overlap["changed_candidate_rows_by_segment"][segment]
        segments[segment] = {
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "hit_rate": float(np.mean(hits)) if hits else 0.0,
            "truth_users": int(len(recalls)),
            "truth_pairs": int(flow["truth_pairs"]),
            "covered_truth_pairs": int(flow["in_both"] + flow["only_in_R_M"] + flow["only_in_R_G"]),
            "incremental_truth_pairs_over_R_M": int(flow["only_in_R_G"]),
            "incremental_candidate_rows_over_R_M": int(changed["p33_only_candidate_rows"]),
            "truth_pairs_per_1k_union_incremental_candidates": (
                1000.0 * flow["only_in_R_G"] / max(changed["p33_only_candidate_rows"], 1)
            ),
        }
    rows = overlap["candidate_rows"]
    return {
        "purpose": "reachability complementarity diagnostic only; not a K0=400 deployment proposal",
        "candidate_rows": {
            "union": int(rows["union"]),
            "incremental_over_R_M": int(rows["p33_only"]),
            "per_active_user_mean": float(rows["union"] / max(overlap["active_users"], 1)),
        },
        "segments": segments,
        "identity_conservation_passed": bool(rows["union"] == rows["m4_total"] + rows["p33_only"]),
    }


def _movement(
    candidates: dict[str, np.ndarray],
    base_rank: np.ndarray,
    new_rank: np.ndarray,
    counts: np.ndarray,
) -> dict[str, Any]:
    target = candidates["target"].astype(bool)
    item_counts = counts[candidates["catalog_row"]]
    masks = {
        "strict_cold": target & (item_counts == 0),
        "sparse_1_5": target & (item_counts >= 1) & (item_counts <= 5),
        "cold_universe": target & (item_counts <= 5),
    }
    output: dict[str, Any] = {}
    for segment, mask in masks.items():
        delta = base_rank[mask].astype(np.int32) - new_rank[mask].astype(np.int32)
        output[segment] = {
            "truth_pairs_in_fixed_top200": int(len(delta)),
            "truth_moved_up": int(np.count_nonzero(delta > 0)),
            "truth_same_rank": int(np.count_nonzero(delta == 0)),
            "truth_moved_down": int(np.count_nonzero(delta < 0)),
            "median_rank_delta": float(np.median(delta)) if len(delta) else None,
            "mean_rank_delta": float(np.mean(delta)) if len(delta) else None,
            "top20_truth_before": int(np.count_nonzero(mask & (base_rank <= 20))),
            "top20_truth_after": int(np.count_nonzero(mask & (new_rank <= 20))),
            "top20_net_truth": int(np.count_nonzero(mask & (new_rank <= 20)) - np.count_nonzero(mask & (base_rank <= 20))),
        }
    return output


def _metric(metrics: dict[str, Any], variant: str, name: str) -> float:
    row = metrics[variant]
    if name == "density20":
        return float(row["at_k"]["20"]["positive_density"])
    if name == "mrr":
        return float(row["ranking"]["mrr"])
    if name == "top200_to_top20":
        return float(row["ranking"]["conversion"]["top200_to_top20"])
    raise KeyError(name)


def scoring_verdict(
    windows: dict[str, dict[str, Any]], base_variant: str, new_variant: str, identity_exact: bool
) -> dict[str, Any]:
    metric_names = ("density20", "mrr", "top200_to_top20")
    means: dict[str, Any] = {}
    nondegrading_windows = 0
    per_window: dict[str, Any] = {}
    for window in WINDOWS:
        counts = 0
        deltas: dict[str, float] = {}
        for name in metric_names:
            base = _metric(windows[window], base_variant, name)
            new = _metric(windows[window], new_variant, name)
            deltas[name] = new - base
            counts += int(new >= base)
        per_window[window] = {"deltas": deltas, "nondegrading_metric_count": counts}
        nondegrading_windows += int(counts >= 2)
    all_improve = True
    none_improve = True
    for name in metric_names:
        base_mean = float(np.mean([_metric(windows[w], base_variant, name) for w in WINDOWS]))
        new_mean = float(np.mean([_metric(windows[w], new_variant, name) for w in WINDOWS]))
        means[name] = {"base": base_mean, "new": new_mean, "delta": new_mean - base_mean}
        all_improve &= new_mean > base_mean
        none_improve &= new_mean <= base_mean
    if identity_exact and all_improve and nondegrading_windows >= 3:
        verdict = "supported"
    elif none_improve:
        verdict = "rejected"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "base_variant": base_variant,
        "new_variant": new_variant,
        "candidate_identity_exact": identity_exact,
        "means": means,
        "windows_with_at_least_2_of_3_metrics_nondegrading": nondegrading_windows,
        "per_window": per_window,
    }


def candidate_set_verdict(
    windows: dict[str, dict[str, Any]], truth_flows: dict[str, Any]
) -> dict[str, Any]:
    segment_evidence: dict[str, Any] = {}
    mean_deltas: dict[str, float] = {}
    for segment in ("strict_cold", "sparse_1_5"):
        base_values = [
            float(windows[w]["A00"]["at_k"]["200"]["segments"][segment]["recall"])
            for w in WINDOWS
        ]
        new_values = [
            float(windows[w]["A10"]["at_k"]["200"]["segments"][segment]["recall"])
            for w in WINDOWS
        ]
        deltas = [new - base for base, new in zip(base_values, new_values)]
        mean_delta = float(np.mean(new_values) - np.mean(base_values))
        mean_deltas[segment] = mean_delta
        segment_evidence[segment] = {
            "m4_mean_recall200": float(np.mean(base_values)),
            "p33_mean_recall200": float(np.mean(new_values)),
            "mean_delta": mean_delta,
            "improving_windows": int(sum(delta > 0 for delta in deltas)),
            "non_improving_windows": int(sum(delta <= 0 for delta in deltas)),
            "per_window_delta": {window: float(delta) for window, delta in zip(WINDOWS, deltas)},
        }
    gained = int(sum(truth_flows[w]["cold_universe"]["gained_truth_pairs"] for w in WINDOWS))
    lost = int(sum(truth_flows[w]["cold_universe"]["lost_truth_pairs"] for w in WINDOWS))
    significant_threshold = max(5, math.ceil(0.10 * (gained + lost)))
    significant_lost = lost > gained and lost - gained >= significant_threshold
    combined_nonnegative = int(
        sum(truth_flows[w]["cold_universe"]["net_truth_pairs"] >= 0 for w in WINDOWS)
    )
    opposite_signs = mean_deltas["strict_cold"] * mean_deltas["sparse_1_5"] < 0
    beneficial = (
        all(value >= 0 for value in mean_deltas.values())
        and any(value > 0 for value in mean_deltas.values())
        and gained > lost
        and combined_nonnegative >= 3
    )
    harmful_by_segment = any(
        evidence["mean_delta"] < 0 and evidence["non_improving_windows"] >= 3
        for evidence in segment_evidence.values()
    )
    if beneficial:
        verdict = "beneficial"
    elif opposite_signs:
        verdict = "mixed"
    elif harmful_by_segment or significant_lost:
        verdict = "harmful"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "segments": segment_evidence,
        "pooled_truth_flow": {
            "gained_truth_pairs": gained,
            "lost_truth_pairs": lost,
            "net_truth_pairs": gained - lost,
            "lost_significance_absolute_and_relative_threshold": significant_threshold,
            "pooled_lost_significantly_greater": significant_lost,
            "combined_nonnegative_windows": combined_nonnegative,
        },
        "opposite_segment_mean_directions": opposite_signs,
    }


def _diagnosis(candidate_effect: str, scoring_effect: str) -> str:
    if scoring_effect == "supported" and candidate_effect in {"harmful", "mixed"}:
        return "candidate_membership_instability_is_primary"
    if scoring_effect == "rejected" and candidate_effect == "harmful":
        return "both_candidate_membership_and_scoring_geometry"
    if scoring_effect == "rejected" and candidate_effect == "beneficial":
        return "scoring_geometry_is_primary"
    if scoring_effect == "supported" and candidate_effect == "beneficial":
        return "neither_is_supported_as_primary_failure_in_this_coarse_only_diagnostic"
    return "evidence_remains_insufficient"


def _f(value: Any, digits: int = 6) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
    ]


def _render_report(master: dict[str, Any]) -> str:
    verdicts = master["verdicts"]
    lines = [
        "# P3.7A：表示空间与粗召回候选集合的 2×2 诊断",
        "",
        "## 结论",
        "",
        f"- P3.3 候选集合效应：`{verdicts['p33_candidate_set_effect']}`。",
        f"- P3.3 在固定 M4 候选集合上的打分效应：`{verdicts['p33_scoring_on_m4_pool']}`。",
        f"- P3.3 在自身候选集合上的打分效应：`{verdicts['p33_scoring_on_p33_pool']}`。",
        f"- P3.7B 辅助打分视角：`{verdicts['p33_auxiliary_view_for_p37b']}`。",
        f"- 主要失败位置：`{verdicts['primary_failure_location']}`。",
        f"- 粗召回 checkpoint（粗召回正式冻结版本）：`{verdicts['coarse_retrieval_checkpoint']}`。",
        "- 本阶段没有训练模型、没有运行 P3.2/P3.7B，也没有读取最终周；最终周 `2020-09-16` 保持 `not_run`。",
        "",
        "## 术语与统计口径",
        "",
        "- **2×2 诊断（本项目自定义实验）**：把候选集合由谁生成、候选由哪个表示空间打分两项因素正交组合。A00 是 M4 集合+M4 打分，A01 是 M4 集合+P3.3 打分，A10 是 P3.3 集合+M4 打分，A11 是 P3.3 集合+P3.3 打分。每名有历史用户的候选预算均为200。",
        "- **candidate membership（候选成员身份，行业常用概念）**：某个用户的 Top200 中具体包含哪些商品，只讨论集合是否覆盖未来购买，不讨论集合内名次。",
        "- **scoring geometry（表示空间打分几何，本项目表述）**：用同一公式在 M4 或 P3.3 的128维归一化商品向量中计算候选与用户历史的相似度；本报告不训练新的打分器。",
        "- **strict-cold / sparse1-5（本项目冷度分群）**：商品在截止日前购买事件分别为0次、1至5次；二者并集称 cold universe（冷/稀疏目录）。",
        "- **Recall@K（行业通用召回率）**：对每位具有相应分群未来真实购买的用户，计算前K覆盖比例后平均；未命中用户计0。",
        "- **HitRate@K（行业通用命中率）**：相应分群真实用户中，前K至少覆盖一件真实商品的用户比例。",
        "- **MRR（Mean Reciprocal Rank，行业通用平均倒数名次）**：每位 cold-universe 真实用户的首个命中名次取倒数后平均；Top200 未命中计0。",
        "- **positive density@K（前K正例密度，本项目核心指标）**：所有活动用户前K候选行中，未来7天被购买的用户—商品行数除以全部候选行数。",
        "- **Top200→TopK conversion（正例集中率，本项目自定义指标）**：已经存在于该方案 Top200 的正例中，最终位于前K的比例；分母随候选集合而定，因此只在固定集合的打分对比中解释。",
        "- **truth flow（正例流转，本项目自定义审计）**：以一个截止日—用户—商品未来正例对为单位，互斥分为两集合都覆盖、仅M4覆盖、仅P3.3覆盖、两者都未覆盖。",
        "- **Jaccard（行业通用集合重合率）**：同一用户两套 Top200 交集大小除以并集大小；分母是该用户的两个候选集合并集。",
        "- **union oracle（并集可达上限，本项目只读诊断）**：R_M 与 R_G 的集合并集，只回答互补覆盖，不是把线上预算改成约400的方案。",
        "- **rank delta（名次变化，本项目自定义）**：旧名次减新名次，正数表示正例上移。",
        "- **changed candidate row（变动候选行，本项目自定义）**：同一用户只出现在一套候选集合中的一条用户—商品记录；效率指标以这种记录为分母。",
        "",
        "## 冻结输入与守恒",
        "",
        f"- 运行标识：`{master['run_id']}`；T_cold=5、history_N=20、K0=200、时间半衰期28天。",
        "- M4 与 P3.3 的用户文件、历史商品身份、历史年龄和有效掩码逐窗完全相同；所有候选均为用户—商品唯一、每活动用户200条、名次1至200连续。",
        "- A00/A01 与 A10/A11 的候选身份、Recall@200 均精确守恒；权威自产打分的重算名次与冻结资产精确一致。",
        "",
        "## 候选集合重合",
        "",
    ]
    rows = []
    for window in WINDOWS:
        row = master["candidate_overlap"]["windows"][window]
        rows.append([
            window,
            _f(row["per_user_jaccard"]["mean"]),
            _f(row["per_user_jaccard"]["median"]),
            _f(row["per_user_intersection_size"]["mean"], 2),
            _f(row["per_user_intersection_size"]["median"], 2),
            row["candidate_rows"]["m4_only"],
            row["candidate_rows"]["p33_only"],
        ])
    lines.extend(_table(["窗口", "Jaccard均值", "中位数", "交集均值", "交集中位数", "仅M4候选行", "仅P3.3候选行"], rows))
    lines.extend(["", "变动候选行的 cold/sparse 构成：", ""])
    rows = []
    for window in WINDOWS:
        changed = master["candidate_overlap"]["windows"][window]["changed_candidate_rows_by_segment"]
        rows.append([window, changed["strict_cold"]["m4_only_candidate_rows"], changed["strict_cold"]["p33_only_candidate_rows"], changed["sparse_1_5"]["m4_only_candidate_rows"], changed["sparse_1_5"]["p33_only_candidate_rows"]])
    lines.extend(_table(["窗口", "strict仅M4", "strict仅P3.3", "sparse仅M4", "sparse仅P3.3"], rows))
    lines.extend(["", "## 正例流转", ""])
    rows = []
    for window in WINDOWS:
        for segment in SEGMENTS:
            row = master["truth_flow"]["windows"][window][segment]
            rows.append([window, segment, row["in_both"], row["only_in_R_M"], row["only_in_R_G"], row["in_neither"], row["net_truth_pairs"], _f(row["gained_truth_pairs_per_1k_p33_only_candidate_rows"], 4), _f(row["lost_truth_pairs_per_1k_m4_only_candidate_rows"], 4)])
    lines.extend(_table(["窗口", "分群", "两者命中", "仅M4", "仅P3.3", "都未命中", "净正例", "每千P3.3独有候选新增", "每千M4独有候选损失"], rows))
    lines.extend(["", "## 四方案核心指标", ""])
    rows = []
    for window in WINDOWS:
        for variant in VARIANTS:
            metric = master["four_variant_metrics"]["windows"][window][variant]
            rows.append([
                window, variant,
                _f(metric["at_k"]["200"]["segments"]["strict_cold"]["recall"]),
                _f(metric["at_k"]["200"]["segments"]["sparse_1_5"]["recall"]),
                _f(metric["at_k"]["20"]["positive_density"], 8),
                _f(metric["ranking"]["mrr"]),
                _f(metric["ranking"]["conversion"]["top200_to_top20"]),
            ])
    lines.extend(_table(["窗口", "方案", "strict R@200", "sparse R@200", "density@20", "MRR", "200→20"], rows))
    lines.extend(["", "每个方案的 Recall、HitRate、密度均完整保存 K=5/10/20/50/100/200；首命中均值/中位数、正例名次四分位和200→50/20/10/5见机器 JSON。", "", "## 固定候选集合的正例名次变化", ""])
    rows = []
    for comparison in ("A00_vs_A01", "A10_vs_A11"):
        for window in WINDOWS:
            for segment in SEGMENTS:
                row = master["rank_movement"]["comparisons"][comparison][window][segment]
                rows.append([comparison, window, segment, row["truth_moved_up"], row["truth_same_rank"], row["truth_moved_down"], _f(row["median_rank_delta"], 2), _f(row["mean_rank_delta"], 2), row["top20_net_truth"]])
    lines.extend(_table(["固定池对比", "窗口", "分群", "上移", "不变", "下移", "中位Δrank", "均值Δrank", "Top20净正例"], rows))
    lines.extend(["", "## 并集可达上限", ""])
    rows = []
    for window in WINDOWS:
        union = master["union_oracle"]["windows"][window]
        rows.append([window, union["candidate_rows"]["union"], _f(union["candidate_rows"]["per_active_user_mean"], 2), _f(union["segments"]["strict_cold"]["recall"]), _f(union["segments"]["sparse_1_5"]["recall"]), union["segments"]["cold_universe"]["incremental_truth_pairs_over_R_M"], _f(union["segments"]["cold_universe"]["truth_pairs_per_1k_union_incremental_candidates"], 4)])
    lines.extend(_table(["窗口", "并集候选行", "每活动用户均值", "strict Recall", "sparse Recall", "相对M4新增正例", "每千增量候选正例"], rows))
    lines.extend([
        "",
        "该并集只用于诊断两个表示空间是否覆盖不同正例；没有生成部署候选，也没有改变 K0。",
        "",
        "## 三类效应与预注册判定",
        "",
        f"1. **candidate_set_effect（候选集合效应）**：`{verdicts['p33_candidate_set_effect']}`。证据只看 A10/A00 的 Recall@200 与正例流转。",
        f"2. **scoring_geometry_effect_on_M4_pool（固定M4池的表示打分效应）**：`{verdicts['p33_scoring_on_m4_pool']}`。证据只看 A01/A00。",
        f"3. **scoring_geometry_effect_on_P33_pool（固定P3.3池的表示打分效应）**：`{verdicts['p33_scoring_on_p33_pool']}`。证据只看 A11/A10。",
        "",
        "“显著多损失”在计算前固定为：pooled lost > gained，且差额至少为5对、同时至少占 gained+lost 变动正例的10%。候选 strict 与 sparse 四窗均值方向相反时优先判 `mixed`，不把一个分群的均值掩盖另一个分群。",
        "",
        "## 本轮要求的三个回答",
        "",
        f"1. P3.3 主要坏在哪里：`{verdicts['primary_failure_location']}`。",
        f"2. P3.3 Student 是否允许作为 P3.7B 辅助打分视角：`{verdicts['p33_auxiliary_view_for_p37b']}`。",
        f"3. M4 Student 是否继续独占粗召回 checkpoint：`{verdicts['coarse_retrieval_checkpoint']}`。",
        "",
        "## 工程证据与边界",
        "",
        f"- 运行耗时 `{master['resources']['elapsed_seconds'] / 60.0:.2f}` 分钟；峰值进程工作集 `{master['resources']['peak_working_set_bytes'] / (1024 ** 3):.2f}` GiB；设备 `{master['resources']['device']}`。",
        "- 四窗 Student、候选、用户和历史文件均按报告声明的字节数与 SHA256 复核；原始交易只用于按冻结规则重建历史、冷度计数和未来7天标签。",
        "- 这是一项 coarse-only（只到粗召回内部排序）的只读诊断；不能把结论外推为 P3.2 或最终 Top12 已改善。",
        "- H&M 数据缺少曝光、库存和上架时间；未购买候选仍不能解释为用户明确负反馈。",
        "- 本轮到此停止，没有启动 P3.7B。",
    ])
    return "\n".join(lines) + "\n"


def run(repo_root: Path, device_name: str = "auto") -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = repo_root.resolve()
    report_dir = repo_root / "reports" / "phase3"
    contract_path = report_dir / "P3_7A_EXPERIMENT_CONTRACT.json"
    contract = _json(contract_path)
    if contract.get("status") != "preregistered_before_formal_computation":
        raise RuntimeError("P3.7A contract was not preregistered")
    if contract.get("run_id") != RUN_ID or contract["execution_boundary"].get("training_allowed"):
        raise RuntimeError("P3.7A contract execution boundary drift")
    if tuple(ROLLING_PROTOCOL) != WINDOWS:
        raise RuntimeError("rolling protocol drift")
    for window in WINDOWS:
        cutoff = str(ROLLING_PROTOCOL[window]["outer_validation"])
        _safe_cutoff(cutoff)
        if cutoff != contract["frozen_windows"][window]:
            raise RuntimeError(f"contract cutoff drift: {window}")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not visible")
    device = torch.device(device_name)

    p31_path = report_dir / "P3_1_metrics.json"
    p33_path = report_dir / "P3_3_metrics.json"
    p31 = _json(p31_path)
    p33 = _json(p33_path)
    if p31.get("status") != "measured" or p31.get("final_week") != "not_run":
        raise RuntimeError("P3.1 authority is not reusable")
    if p33.get("status") != "measured" or p33.get("final_week") != "not_run":
        raise RuntimeError("P3.3 authority is not reusable")
    p33_p31 = p33["pipeline"]["p3_1"]
    if p33_p31.get("status") != "measured" or p33_p31.get("final_week") != "not_run":
        raise RuntimeError("P3.3 coarse authority is not reusable")

    static_catalog = repo_root / "artifacts" / "m4" / "m4-v1-supervised-cold-representation" / "student-v1" / "static_catalog" / "catalog_items.csv"
    local_transactions = repo_root / "data" / "interim" / "audit" / "transactions.parquet"
    p36_path = report_dir / "P3_6_SIMILARITY_COMPLEMENTARITY_AUDIT.json"
    p36 = _json(p36_path)
    declared_transactions = p36["input_identities"]["transactions"]
    transactions = local_transactions if local_transactions.is_file() else Path(declared_transactions["path"])
    transaction_identity_audit = _require_identity(
        transactions, declared_transactions, "P3.7A frozen transaction parquet"
    )
    catalog_items = pd.read_csv(static_catalog, dtype={"article_id": str})["article_id"].tolist()
    article_to_row = {item: row for row, item in enumerate(catalog_items)}
    if len(catalog_items) != 105_542:
        raise RuntimeError("static catalog row count drift")

    four_metrics: dict[str, Any] = {}
    overlaps: dict[str, Any] = {}
    truth_flows: dict[str, Any] = {}
    unions: dict[str, Any] = {}
    rank_m4: dict[str, Any] = {}
    rank_p33: dict[str, Any] = {}
    asset_audits: dict[str, Any] = {}
    parity: dict[str, Any] = {}

    for window in WINDOWS:
        cutoff = str(ROLLING_PROTOCOL[window]["outer_validation"])
        print(f"P3.7A {window}: verify frozen assets", flush=True)
        m_row = p31["windows"][window]
        g_row = p33_p31["windows"][window]
        m_paths = {
            "embedding": Path(m_row["student_embedding"]["path"]),
            "candidates": Path(m_row["artifacts"]["candidates"]["path"]),
            "histories": Path(m_row["artifacts"]["histories"]["path"]),
            "users": Path(m_row["artifacts"]["users"]["path"]),
        }
        g_paths = {
            "embedding": Path(g_row["student_embedding"]["path"]),
            "candidates": Path(g_row["artifacts"]["candidates"]["path"]),
            "histories": Path(g_row["artifacts"]["histories"]["path"]),
            "users": Path(g_row["artifacts"]["users"]["path"]),
        }
        m_declared = {
            "embedding": m_row["student_embedding"], **m_row["artifacts"]
        }
        g_declared = {
            "embedding": g_row["student_embedding"], **g_row["artifacts"]
        }
        identity = {
            "m4": {name: _require_identity(path, m_declared[name], f"{window} M4 {name}") for name, path in m_paths.items()},
            "p33": {name: _require_identity(path, g_declared[name], f"{window} P3.3 {name}") for name, path in g_paths.items()},
        }
        m_manifest = _json(m_paths["candidates"].parent / "manifest.json")
        g_manifest = _json(g_paths["candidates"].parent / "manifest.json")
        g_student_manifest = _json(g_paths["embedding"].parent / "manifest.json")
        if m_manifest["student_embedding"]["sha256"] != m_declared["embedding"]["sha256"]:
            raise RuntimeError(f"{window} M4 candidate manifest Student drift")
        if g_manifest["student_embedding"]["sha256"] != g_declared["embedding"]["sha256"]:
            raise RuntimeError(f"{window} P3.3 candidate manifest Student drift")
        if g_student_manifest["encoding"]["artifact"]["sha256"] != g_declared["embedding"]["sha256"]:
            raise RuntimeError(f"{window} P3.3 Student manifest embedding drift")
        if identity["m4"]["histories"]["observed"]["sha256"] != identity["p33"]["histories"]["observed"]["sha256"]:
            raise RuntimeError(f"{window} M4/P3.3 history file identity differs")
        if identity["m4"]["users"]["observed"]["sha256"] != identity["p33"]["users"]["observed"]["sha256"]:
            raise RuntimeError(f"{window} M4/P3.3 users file identity differs")

        m_candidates = _load_npz(m_paths["candidates"])
        g_candidates = _load_npz(g_paths["candidates"])
        m_histories = _load_npz(m_paths["histories"])
        g_histories = _load_npz(g_paths["histories"])
        if set(m_histories) != set(g_histories) or any(
            not np.array_equal(m_histories[name], g_histories[name]) for name in m_histories
        ):
            raise RuntimeError(f"{window} M4/P3.3 history arrays differ")
        m_users = pd.read_csv(m_paths["users"], dtype={"customer_id": str})["customer_id"].tolist()
        g_users = pd.read_csv(g_paths["users"], dtype={"customer_id": str})["customer_id"].tolist()
        if m_users != g_users:
            raise RuntimeError(f"{window} M4/P3.3 user identity differs")
        if m_histories["catalog_row"].shape != (len(m_users), HISTORY_N):
            raise RuntimeError(f"{window} frozen history shape drift")
        counts, rebuilt_rows, rebuilt_days, truth, cutoff_audit = _cutoff_context(
            transactions_path=transactions,
            cutoff=cutoff,
            users=m_users,
            article_to_row=article_to_row,
            catalog_size=len(catalog_items),
        )
        rebuilt_mask = (rebuilt_rows >= 0).astype(np.uint8)
        history_rebuild = {
            "catalog_row_exact": bool(np.array_equal(rebuilt_rows, m_histories["catalog_row"])),
            "days_since_purchase_exact": bool(np.array_equal(rebuilt_days, m_histories["days_since_purchase"])),
            "mask_exact": bool(np.array_equal(rebuilt_mask, m_histories["mask"])),
            "history_n": int(m_histories["catalog_row"].shape[1]),
            "latest_history_before_cutoff": bool(cutoff_audit["cutoff_safe"]),
        }
        if not all(value for key, value in history_rebuild.items() if key != "history_n") or history_rebuild["history_n"] != HISTORY_N:
            raise RuntimeError(f"{window} frozen history reconstruction failed: {history_rebuild}")
        candidate_audit = {
            "m4": _validate_candidate_asset(m_candidates, m_histories, truth, counts),
            "p33": _validate_candidate_asset(g_candidates, g_histories, truth, counts),
        }
        m_embedding = np.load(m_paths["embedding"], mmap_mode="r")
        g_embedding = np.load(g_paths["embedding"], mmap_mode="r")
        if m_embedding.shape != (len(catalog_items), 128) or g_embedding.shape != (len(catalog_items), 128):
            raise RuntimeError(f"{window} Student embedding shape drift")
        if str(m_embedding.dtype) != "float16" or str(g_embedding.dtype) != "float16":
            raise RuntimeError(f"{window} Student embedding dtype drift")

        print(f"P3.7A {window}: score four frozen combinations", flush=True)
        candidate_universe = np.flatnonzero(counts <= 5).astype(np.int32)
        m_space_scores, m_space_reconstruction = _score_universe_for_frozen_sets(
            {"R_M": m_candidates, "R_G": g_candidates},
            m_histories,
            m_embedding,
            candidate_universe,
            authoritative_set="R_M",
            device=device,
        )
        g_space_scores, g_space_reconstruction = _score_universe_for_frozen_sets(
            {"R_M": m_candidates, "R_G": g_candidates},
            g_histories,
            g_embedding,
            candidate_universe,
            authoritative_set="R_G",
            device=device,
        )
        score_mm = m_space_scores["R_M"]
        score_gm = g_space_scores["R_M"]
        score_mg = m_space_scores["R_G"]
        score_gg = g_space_scores["R_G"]
        formula = {
            "A00_authoritative_M4_pool_M4_score": _formula_audit(m_candidates, score_mm),
            "A11_authoritative_P33_pool_P33_score": _formula_audit(g_candidates, score_gg),
            "M4_full_universe_candidate_reconstruction": m_space_reconstruction,
            "P33_full_universe_candidate_reconstruction": g_space_reconstruction,
        }
        ranks = {
            "A00": _rank_by_score(m_candidates["user_index"], m_candidates["catalog_row"], score_mm),
            "A01": _rank_by_score(m_candidates["user_index"], m_candidates["catalog_row"], score_gm),
            "A10": _rank_by_score(g_candidates["user_index"], g_candidates["catalog_row"], score_mg),
            "A11": _rank_by_score(g_candidates["user_index"], g_candidates["catalog_row"], score_gg),
        }
        metrics = {
            "A00": _ordering_metrics(arrays=m_candidates, ordering_rank=ranks["A00"], truth=truth, counts=counts, user_count=len(m_users)),
            "A01": _ordering_metrics(arrays=m_candidates, ordering_rank=ranks["A01"], truth=truth, counts=counts, user_count=len(m_users)),
            "A10": _ordering_metrics(arrays=g_candidates, ordering_rank=ranks["A10"], truth=truth, counts=counts, user_count=len(m_users)),
            "A11": _ordering_metrics(arrays=g_candidates, ordering_rank=ranks["A11"], truth=truth, counts=counts, user_count=len(m_users)),
        }
        recall_parity = {
            "A00_A01": all(
                metrics["A00"]["at_k"]["200"]["segments"][segment]["recall"]
                == metrics["A01"]["at_k"]["200"]["segments"][segment]["recall"]
                for segment in SEGMENTS
            ),
            "A10_A11": all(
                metrics["A10"]["at_k"]["200"]["segments"][segment]["recall"]
                == metrics["A11"]["at_k"]["200"]["segments"][segment]["recall"]
                for segment in SEGMENTS
            ),
        }
        candidate_identity_parity = {
            "A00_A01": True,
            "A10_A11": True,
        }
        if not all(recall_parity.values()):
            raise RuntimeError(f"{window} same-pool Recall@200 conservation failed")
        overlap = _candidate_overlap(m_candidates, g_candidates, counts)
        if not overlap["conservation_passed"]:
            raise RuntimeError(f"{window} candidate overlap conservation failed")
        flow = _truth_flow(m_candidates, g_candidates, truth, counts, len(m_users), overlap)
        union = _union_oracle(m_candidates, g_candidates, truth, counts, len(m_users), overlap, flow)
        if not union["identity_conservation_passed"]:
            raise RuntimeError(f"{window} union identity conservation failed")

        four_metrics[window] = metrics
        overlaps[window] = overlap
        truth_flows[window] = flow
        unions[window] = union
        rank_m4[window] = _movement(m_candidates, ranks["A00"], ranks["A01"], counts)
        rank_p33[window] = _movement(g_candidates, ranks["A10"], ranks["A11"], counts)
        parity[window] = {
            "candidate_identity": candidate_identity_parity,
            "recall200": recall_parity,
            "score_formula": formula,
        }
        asset_audits[window] = {
            "cutoff": cutoff,
            "identity": identity,
            "manifest_checks": {
                "m4_candidate_manifest_student_sha": True,
                "p33_candidate_manifest_student_sha": True,
                "p33_student_manifest_embedding_sha": True,
            },
            "history_cross_space_file_sha_exact": True,
            "users_cross_space_file_sha_exact": True,
            "history_arrays_cross_space_exact": True,
            "users_cross_space_exact": True,
            "history_reconstruction": history_rebuild,
            "cutoff_audit": cutoff_audit,
            "candidate_invariants": candidate_audit,
        }
        del m_candidates, g_candidates, m_histories, g_histories, m_embedding, g_embedding
        del score_mm, score_gm, score_mg, score_gg, ranks, truth, counts
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    m4_score = scoring_verdict(four_metrics, "A00", "A01", identity_exact=True)
    p33_score = scoring_verdict(four_metrics, "A10", "A11", identity_exact=True)
    candidate_effect = candidate_set_verdict(four_metrics, truth_flows)
    auxiliary = "allowed" if m4_score["verdict"] == "supported" else "not_allowed"
    checkpoint = (
        "M4 Student remains the exclusive coarse retrieval checkpoint"
        if candidate_effect["verdict"] != "beneficial"
        else "manual promotion review required; P3.7A does not auto-promote P3.3"
    )
    verdicts = {
        "p33_candidate_set_effect": candidate_effect["verdict"],
        "p33_scoring_on_m4_pool": m4_score["verdict"],
        "p33_scoring_on_p33_pool": p33_score["verdict"],
        "p33_auxiliary_view_for_p37b": auxiliary,
        "primary_failure_location": _diagnosis(candidate_effect["verdict"], m4_score["verdict"]),
        "coarse_retrieval_checkpoint": checkpoint,
    }
    all_parity = all(
        all(row["candidate_identity"].values())
        and all(row["recall200"].values())
        and all(item["passed"] for item in row["score_formula"].values())
        for row in parity.values()
    )
    all_asset_checks = all(
        row["history_cross_space_file_sha_exact"]
        and row["users_cross_space_file_sha_exact"]
        and row["history_arrays_cross_space_exact"]
        and row["users_cross_space_exact"]
        and row["cutoff_audit"]["cutoff_safe"]
        for row in asset_audits.values()
    )
    verification = {
        "contract_preregistered_before_formal_computation": True,
        "training_calls": 0,
        "new_model_calls": 0,
        "p3_2_calls": 0,
        "p3_7b_calls": 0,
        "final_week": "not_run",
        "asset_and_cutoff_checks_passed": all_asset_checks,
        "all_same_pool_and_formula_conservation_passed": all_parity,
        "all_overlap_conservation_passed": all(row["conservation_passed"] for row in overlaps.values()),
        "all_truth_flow_conservation_passed": all(
            row[segment]["mutually_exclusive_and_exhaustive"]
            for row in truth_flows.values() for segment in SEGMENTS
        ),
        "all_union_conservation_passed": all(row["identity_conservation_passed"] for row in unions.values()),
    }
    verification["passed"] = all(
        bool(value) for key, value in verification.items()
        if key not in {"training_calls", "new_model_calls", "p3_2_calls", "p3_7b_calls", "final_week"}
    ) and all(verification[key] == 0 for key in ("training_calls", "new_model_calls", "p3_2_calls", "p3_7b_calls"))
    if not verification["passed"]:
        raise RuntimeError(f"P3.7A verification failed: {verification}")

    candidate_overlap_output = {
        "schema_version": "phase3-p3.7a-candidate-overlap-v1", "stage": STAGE,
        "status": "measured", "run_id": RUN_ID, "windows": overlaps, "final_week": "not_run",
    }
    truth_flow_output = {
        "schema_version": "phase3-p3.7a-truth-flow-v1", "stage": STAGE,
        "status": "measured", "run_id": RUN_ID, "windows": truth_flows,
        "candidate_set_verdict": candidate_effect, "final_week": "not_run",
    }
    metrics_output = {
        "schema_version": "phase3-p3.7a-four-variant-metrics-v1", "stage": STAGE,
        "status": "measured", "run_id": RUN_ID, "variant_definitions": VARIANTS,
        "windows": four_metrics, "same_pool_parity": parity, "final_week": "not_run",
    }
    movement_output = {
        "schema_version": "phase3-p3.7a-rank-movement-v1", "stage": STAGE,
        "status": "measured", "run_id": RUN_ID,
        "comparisons": {"A00_vs_A01": rank_m4, "A10_vs_A11": rank_p33},
        "scoring_verdicts": {"m4_pool": m4_score, "p33_pool": p33_score},
        "final_week": "not_run",
    }
    union_output = {
        "schema_version": "phase3-p3.7a-union-oracle-v1", "stage": STAGE,
        "status": "measured", "run_id": RUN_ID, "deployment_proposal": False,
        "windows": unions, "final_week": "not_run",
    }
    outputs = {
        "p3_7a_candidate_overlap.json": candidate_overlap_output,
        "p3_7a_truth_flow.json": truth_flow_output,
        "p3_7a_four_variant_metrics.json": metrics_output,
        "p3_7a_rank_movement.json": movement_output,
        "p3_7a_union_oracle.json": union_output,
    }
    for name, value in outputs.items():
        atomic_json(report_dir / name, value)

    master = {
        "schema_version": "phase3-p3.7a-2x2-diagnostic-v1",
        "stage": STAGE,
        "status": "measured",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": file_identity(contract_path),
        "authoritative_inputs": {
            "p3_1_metrics": file_identity(p31_path),
            "p3_3_metrics": file_identity(p33_path),
            "p3_6_source_identity_evidence": file_identity(p36_path),
            "static_catalog": file_identity(static_catalog),
            "transactions": transaction_identity_audit,
        },
        "asset_audits": asset_audits,
        "candidate_overlap": candidate_overlap_output,
        "truth_flow": truth_flow_output,
        "four_variant_metrics": metrics_output,
        "rank_movement": movement_output,
        "union_oracle": union_output,
        "verdict_evidence": {
            "candidate_set": candidate_effect,
            "scoring_on_m4_pool": m4_score,
            "scoring_on_p33_pool": p33_score,
        },
        "verdicts": verdicts,
        "verification": verification,
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "training_gpu_hours": 0,
        },
        "final_week": "not_run",
    }
    main_json = report_dir / "P3_7A_2X2_DIAGNOSTIC.json"
    main_md = report_dir / "P3_7A_2X2_DIAGNOSTIC.md"
    atomic_json(main_json, master)
    main_md.write_text(_render_report(master), encoding="utf-8")
    output_names = [
        "P3_7A_EXPERIMENT_CONTRACT.json",
        "P3_7A_2X2_DIAGNOSTIC.md",
        "P3_7A_2X2_DIAGNOSTIC.json",
        *outputs.keys(),
    ]
    manifest = {
        "schema_version": "phase3-p3.7a-output-manifest-v1",
        "stage": STAGE,
        "status": "measured",
        "run_id": RUN_ID,
        "outputs": {name: file_identity(report_dir / name) for name in output_names},
        "verification": verification,
        "verdicts": verdicts,
        "final_week": "not_run",
    }
    atomic_json(report_dir / "P3_7A_OUTPUT_MANIFEST.json", manifest)
    return master


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the read-only P3.7A 2x2 diagnostic")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    try:
        result = run(args.repo_root, args.device)
    except Exception as exc:
        report_dir = args.repo_root.resolve() / "reports" / "phase3"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        atomic_json(report_dir / f"P3_7A_FAILURE_{timestamp}.json", {
            "schema_version": "phase3-p3.7a-failure-v1",
            "stage": STAGE,
            "status": "failed_closed",
            "run_id": RUN_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "training_calls": 0,
            "final_week": "not_run",
        })
        raise
    print(json.dumps({"status": result["status"], "verdicts": result["verdicts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
