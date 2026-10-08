from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineOutput:
    candidates: dict[str, list[int]]
    predictions: dict[str, list[int]]
    candidate_rows: pd.DataFrame


def build_ground_truth(validation: pd.DataFrame) -> dict[str, list[int]]:
    return (
        validation.groupby("customer_id", observed=True)["article_id"]
        .agg(lambda values: list(dict.fromkeys(int(v) for v in values)))
        .to_dict()
    )


def _rank_global_popularity(
    history: pd.DataFrame, cutoff: pd.Timestamp, window_days: int
) -> list[int]:
    window_start = cutoff - pd.Timedelta(days=window_days)
    recent = history.loc[history["t_dat"] >= window_start].copy()
    if recent.empty:
        recent = history.copy()
    age_days = (cutoff - recent["t_dat"]).dt.days.clip(lower=0)
    recent["weight"] = np.exp2(-age_days / 7.0)
    scores = recent.groupby("article_id", observed=True)["weight"].sum()
    return [int(item) for item in scores.sort_values(ascending=False).index]


def generate_baseline(
    history: pd.DataFrame,
    target_users: list[str],
    cutoff: pd.Timestamp,
    *,
    candidate_k: int = 100,
    metric_k: int = 12,
    popular_window_days: int = 28,
) -> BaselineOutput:
    """Repurchase + recent-popularity retrieval with reciprocal-rank fusion."""
    if candidate_k < metric_k:
        raise ValueError("candidate_k must be >= metric_k")

    popular = _rank_global_popularity(history, cutoff, popular_window_days)
    user_history = history.loc[history["customer_id"].isin(target_users)].copy()
    personal = (
        user_history.groupby(["customer_id", "article_id"], observed=True)
        .agg(last_purchase=("t_dat", "max"), purchase_count=("article_id", "size"))
        .reset_index()
        .sort_values(
            ["customer_id", "last_purchase", "purchase_count", "article_id"],
            ascending=[True, False, False, True],
        )
    )
    personal_lists = (
        personal.groupby("customer_id", observed=True)["article_id"]
        .agg(lambda values: [int(v) for v in values])
        .to_dict()
    )

    candidate_map: dict[str, list[int]] = {}
    prediction_map: dict[str, list[int]] = {}
    rows: list[dict[str, object]] = []
    popular_limit = min(candidate_k, len(popular))
    popular_ranks = {item: rank for rank, item in enumerate(popular[:popular_limit], 1)}

    for user in target_users:
        personal_items = personal_lists.get(user, [])[:candidate_k]
        personal_ranks = {item: rank for rank, item in enumerate(personal_items, 1)}
        union = list(dict.fromkeys(personal_items + popular[:candidate_k]))

        scored: list[tuple[float, int]] = []
        for item in union:
            personal_rank = personal_ranks.get(item)
            popular_rank = popular_ranks.get(item)
            score = 0.0
            if personal_rank is not None:
                score += 2.0 / (10 + personal_rank)
            if popular_rank is not None:
                score += 1.0 / (10 + popular_rank)
            scored.append((score, item))
            rows.append(
                {
                    "customer_id": user,
                    "article_id": item,
                    "personal_rank": personal_rank,
                    "popularity_rank": popular_rank,
                    "baseline_score": score,
                }
            )

        ranked = [item for _, item in sorted(scored, key=lambda x: (-x[0], x[1]))]
        candidate_map[user] = ranked[:candidate_k]
        prediction_map[user] = ranked[:metric_k]

    candidate_rows = pd.DataFrame.from_records(rows)
    return BaselineOutput(candidate_map, prediction_map, candidate_rows)
