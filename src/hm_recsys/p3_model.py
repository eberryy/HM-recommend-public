from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


EMBEDDING_DIM = 128
HISTORY_N = 20
K0 = 200
NEGATIVES_PER_POSITIVE = 50
SAMPLING_SEED = 20260905
HIDDEN_DIM = 128


def stable_seed(*values: object) -> int:
    payload = "|".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


class CandidateAwareReranker(nn.Module):
    """Candidate-conditioned attention over a padded user purchase history."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate: torch.Tensor,
        history: torch.Tensor,
        days_since_purchase: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate.ndim != 2 or history.ndim != 3:
            raise ValueError("candidate must be [B,D] and history must be [B,N,D]")
        if not bool(history_mask.any(dim=1).all()):
            raise ValueError("each scored candidate must have at least one valid history item")
        expanded = candidate[:, None, :].expand_as(history)
        interaction = torch.cat(
            [
                expanded,
                history,
                expanded * history,
                torch.abs(expanded - history),
                torch.log1p(torch.clamp(days_since_purchase, min=0.0))[:, :, None],
            ],
            dim=-1,
        )
        logits = self.attention(interaction).squeeze(-1)
        logits = logits.masked_fill(~history_mask, -torch.inf)
        attention = torch.softmax(logits, dim=1)
        user = torch.sum(attention[:, :, None] * history, dim=1)
        score_input = torch.cat(
            [user, candidate, user * candidate, torch.abs(user - candidate)], dim=-1
        )
        score = self.scorer(score_input).squeeze(-1)
        return score, attention


def pairwise_logistic_loss(positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:
    return F.softplus(-(positive_score - negative_score)).mean()


def rank_bucket(rank: int, budget: int = K0) -> str:
    if rank < 1 or rank > budget:
        raise ValueError(f"rank must be in [1,{budget}]")
    first = math.ceil(budget / 3)
    second = math.ceil(2 * budget / 3)
    if rank <= first:
        return "hard"
    if rank <= second:
        return "medium"
    return "easy"


def sample_same_user_pairs(
    *,
    user_index: np.ndarray,
    rank: np.ndarray,
    target: np.ndarray,
    cutoff: str,
    max_negatives: int = NEGATIVES_PER_POSITIVE,
    budget: int = K0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return [user_index, positive_row, negative_row, bucket_id] pair rows."""
    if not (len(user_index) == len(rank) == len(target)):
        raise ValueError("candidate arrays have inconsistent lengths")
    bucket_names = ("hard", "medium", "easy")
    bucket_id = {name: index for index, name in enumerate(bucket_names)}
    quotas = {
        "hard": max_negatives // 3 + int(max_negatives % 3 > 0),
        "medium": max_negatives // 3 + int(max_negatives % 3 > 1),
        "easy": max_negatives // 3,
    }
    output: list[tuple[int, int, int, int]] = []
    positive_rows = 0
    positive_groups = 0
    cursor = 0
    while cursor < len(user_index):
        end = cursor + 1
        while end < len(user_index) and user_index[end] == user_index[cursor]:
            end += 1
        group_rows = np.arange(cursor, end, dtype=np.int64)
        positives = group_rows[target[cursor:end] == 1]
        if len(positives):
            positive_groups += 1
            positive_rows += len(positives)
            negatives_by_bucket = {
                name: group_rows[
                    (target[cursor:end] == 0)
                    & np.asarray([rank_bucket(int(value), budget) == name for value in rank[cursor:end]])
                ]
                for name in bucket_names
            }
            for positive_row in positives:
                selected: list[tuple[int, str]] = []
                for name in bucket_names:
                    choices = negatives_by_bucket[name]
                    if not len(choices):
                        continue
                    rng = np.random.default_rng(
                        stable_seed(SAMPLING_SEED, cutoff, int(user_index[cursor]), int(positive_row), name)
                    )
                    take = min(quotas[name], len(choices))
                    picked = rng.choice(choices, size=take, replace=False)
                    selected.extend((int(value), name) for value in picked)
                if len(selected) < max_negatives:
                    used = {row for row, _ in selected}
                    remaining = [
                        int(row)
                        for row in group_rows[target[cursor:end] == 0]
                        if int(row) not in used
                    ]
                    rng = np.random.default_rng(
                        stable_seed(SAMPLING_SEED, cutoff, int(user_index[cursor]), int(positive_row), "fill")
                    )
                    if remaining:
                        extra = rng.choice(
                            remaining,
                            size=min(max_negatives - len(selected), len(remaining)),
                            replace=False,
                        )
                        selected.extend((int(value), rank_bucket(int(rank[int(value)]), budget)) for value in extra)
                for negative_row, name in selected[:max_negatives]:
                    output.append(
                        (int(user_index[cursor]), int(positive_row), negative_row, bucket_id[name])
                    )
        cursor = end
    pairs = np.asarray(output, dtype=np.int64).reshape(-1, 4)
    bucket_counts = {
        name: int(np.count_nonzero(pairs[:, 3] == index)) if len(pairs) else 0
        for name, index in bucket_id.items()
    }
    audit = {
        "positive_candidate_rows": int(positive_rows),
        "positive_groups": int(positive_groups),
        "pair_rows": int(len(pairs)),
        "maximum_negatives_per_positive": max_negatives,
        "same_user_only": bool(
            all(user_index[pos] == user_index[neg] == user for user, pos, neg, _ in pairs)
        ),
        "bucket_counts": bucket_counts,
        "all_buckets_covered": all(value > 0 for value in bucket_counts.values()) if len(pairs) else False,
        "fixed_hash_seed": SAMPLING_SEED,
    }
    return pairs, audit


def binary_auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Tie-aware Mann-Whitney AUC without an optional scipy dependency."""
    if not len(positive) or not len(negative):
        return None
    values = np.concatenate([positive, negative]).astype(np.float64, copy=False)
    labels = np.concatenate(
        [np.ones(len(positive), dtype=np.uint8), np.zeros(len(negative), dtype=np.uint8)]
    )
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - len(positive) * (len(positive) + 1) / 2.0) / (
        len(positive) * len(negative)
    )


def attention_summary(attention: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    if attention.shape != mask.shape:
        raise ValueError("attention and mask shapes differ")
    if not len(attention):
        return {"rows": 0, "entropy_mean": 0.0, "top1_weight_mean": 0.0, "top3_weight_sum_mean": 0.0}
    safe = np.where(mask, attention, 0.0)
    entropy = -np.sum(np.where(safe > 0, safe * np.log(np.maximum(safe, 1e-12)), 0.0), axis=1)
    ordered = np.sort(safe, axis=1)
    return {
        "rows": int(len(attention)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_median": float(np.median(entropy)),
        "entropy_p90": float(np.quantile(entropy, 0.90)),
        "top1_weight_mean": float(np.mean(ordered[:, -1])),
        "top1_weight_median": float(np.median(ordered[:, -1])),
        "top3_weight_sum_mean": float(np.mean(np.sum(ordered[:, -3:], axis=1))),
        "top3_weight_sum_median": float(np.median(np.sum(ordered[:, -3:], axis=1))),
    }


@dataclass(frozen=True)
class CutoffAsset:
    cutoff: str
    candidate_path: str
    users_path: str
    history_path: str
    embedding_path: str


def iter_slices(length: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))
