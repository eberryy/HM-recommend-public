from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence


def _unique(values: Iterable[Hashable]) -> list[Hashable]:
    return list(dict.fromkeys(values))


def apk(
    actual: Sequence[Hashable], predicted: Sequence[Hashable], k: int = 12
) -> float:
    """Average precision at k using the H&M/Kaggle denominator convention."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(actual)
    if not relevant:
        return 0.0

    score = 0.0
    hits = 0
    seen: set[Hashable] = set()
    for rank, item in enumerate(predicted[:k], start=1):
        if item in relevant and item not in seen:
            hits += 1
            score += hits / rank
        seen.add(item)
    return score / min(len(relevant), k)


def mapk(
    actual: Mapping[Hashable, Sequence[Hashable]],
    predicted: Mapping[Hashable, Sequence[Hashable]],
    k: int = 12,
) -> float:
    """Mean AP@k over users with at least one ground-truth item."""
    users = [user for user, items in actual.items() if items]
    if not users:
        return 0.0
    return sum(apk(actual[user], predicted.get(user, []), k) for user in users) / len(users)


def recall_at_k(
    actual: Mapping[Hashable, Sequence[Hashable]],
    candidates: Mapping[Hashable, Sequence[Hashable]],
    k: int,
) -> float:
    """Macro-average candidate recall at k over active users."""
    recalls: list[float] = []
    for user, items in actual.items():
        relevant = set(items)
        if not relevant:
            continue
        retrieved = set(_unique(candidates.get(user, []))[:k])
        recalls.append(len(relevant & retrieved) / len(relevant))
    return sum(recalls) / len(recalls) if recalls else 0.0


def hit_rate_at_k(
    actual: Mapping[Hashable, Sequence[Hashable]],
    candidates: Mapping[Hashable, Sequence[Hashable]],
    k: int,
) -> float:
    """Fraction of active users for whom at least one relevant item is retrieved."""
    hits: list[bool] = []
    for user, items in actual.items():
        relevant = set(items)
        if not relevant:
            continue
        retrieved = set(_unique(candidates.get(user, []))[:k])
        hits.append(bool(relevant & retrieved))
    return sum(hits) / len(hits) if hits else 0.0


def oracle_mapk(
    actual: Mapping[Hashable, Sequence[Hashable]],
    candidates: Mapping[Hashable, Sequence[Hashable]],
    candidate_k: int,
    metric_k: int = 12,
) -> float:
    """Best possible MAP@metric_k if the retrieved pool were ranked perfectly."""
    oracle_predictions: dict[Hashable, list[Hashable]] = {}
    for user, items in actual.items():
        relevant = set(items)
        pool = _unique(candidates.get(user, []))[:candidate_k]
        positives = [item for item in pool if item in relevant]
        negatives = [item for item in pool if item not in relevant]
        oracle_predictions[user] = positives + negatives
    return mapk(actual, oracle_predictions, metric_k)
