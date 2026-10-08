"""Pure, label-free P4.2 probability comparison and exact admission matching.

Array axes are always Cold candidate first, original Warm slot second.  Callers
must put Cold candidates in (B0 rank, article_id) order and Warm candidates in
their frozen W0 slot order.  This module never reads truth or changes W0 order.

The primary utility is separable, ``logit(qC) - logit(qW)``; consequently several
assignments can share the same maximum total weight.  We do not claim that the
optimum is unique or perturb the weights to break those ties.  The preregistered
tie policy is the deterministic SciPy 1.17.1 linear_sum_assignment result on
canonical Warm rows and canonical Cold columns followed by unmatched dummies.
That policy must not be selected again after observing MAP.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment


EPSILON = 1e-6
TAUS = (0.0, 0.6931471805599453, 1.3862943611198906)
MATCHING_SOLVER = "scipy.optimize.linear_sum_assignment"
MATCHING_SOLVER_VERSION = "1.17.1"


def clipped_logit(q: ArrayLike) -> np.ndarray:
    """Log odds after the fixed [1e-6, 1-1e-6] propensity clipping.

    These are next-week truth propensities conditional on their candidate pool,
    not online exposure-conditioned purchase probabilities.  Invalid model
    outputs fail closed instead of being converted into an admission score.
    """
    values = np.asarray(q, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("propensities must be finite")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("propensities must be in [0, 1]")
    clipped = np.clip(values, EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def pair_utilities(cold_q: ArrayLike, warm_q: ArrayLike) -> np.ndarray:
    """Return the unmodified Cold x Warm log-odds-difference matrix."""
    cold = np.asarray(cold_q, dtype=np.float64)
    warm = np.asarray(warm_q, dtype=np.float64)
    if cold.ndim != 1 or warm.ndim != 1:
        raise ValueError("cold_q and warm_q must be one-dimensional")
    result = clipped_logit(cold)[:, None] - clipped_logit(warm)[None, :]
    _validated_utility(result, TAUS[0])
    return result


def _validated_utility(utility: ArrayLike, tau: float) -> np.ndarray:
    values = np.asarray(utility, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("utility must have shape (Cold candidates, Warm slots)")
    if values.shape[0] > 50 or values.shape[1] > 12:
        raise ValueError("P4.2 permits at most Cold50 x WarmTop12")
    if not np.all(np.isfinite(values)):
        raise ValueError("utility must be finite")
    if not np.isfinite(tau) or float(tau) not in TAUS:
        raise ValueError("tau must be one of the three preregistered P4.2 values")
    return values


def exact_matching(utility: ArrayLike, tau: float) -> list[tuple[int, int]]:
    """Maximum total positive (U-tau) weight, with unmatched nodes permitted.

    Return ``(cold_index, warm_index)`` pairs sorted by original Warm slot.  An
    edge exists iff U > tau, so threshold equality is rejected.  Warm rows have
    one zero-cost dummy column each; there is no hard cap on admissions beyond
    the one-to-one graph constraint.  Positive edge costs are negated and all
    prohibited edges have infinite cost.  The unchanged float64 objective is
    solved exactly by the assignment algorithm, not by the diagnostic greedy
    rule below.  Floating-point solver arithmetic is still subject to normal
    machine precision; no epsilon perturbation or score rounding is applied.
    """
    values = _validated_utility(utility, tau)
    n_cold, n_warm = values.shape
    edges = values > tau
    if n_cold == 0 or n_warm == 0 or not edges.any():
        return []
    costs = np.full((n_warm, n_cold + n_warm), np.inf, dtype=np.float64)
    costs[:, :n_cold] = np.where(edges.T, -(values.T - tau), np.inf)
    costs[:, n_cold:] = 0.0
    warm_rows, columns = linear_sum_assignment(costs)
    matches = [(int(cold), int(warm)) for warm, cold in zip(warm_rows, columns)
               if cold < n_cold]
    _validate_matches(values, tau, matches)
    return matches


def _validate_matches(
    values: np.ndarray, tau: float, matches: Sequence[tuple[int, int]],
) -> None:
    if len({c for c, _ in matches}) != len(matches):
        raise ValueError("a Cold node may be matched at most once")
    if len({w for _, w in matches}) != len(matches):
        raise ValueError("a Warm slot may be matched at most once")
    for cold, warm in matches:
        if not (0 <= cold < values.shape[0] and 0 <= warm < values.shape[1]):
            raise ValueError("matching index is outside the graph")
        if not values[cold, warm] > tau:
            raise ValueError("matching contains a prohibited or equality edge")


def _diagnostic_greedy(values: np.ndarray, tau: float) -> list[tuple[int, int]]:
    """Label-free comparison only; never used to construct a primary list."""
    cold_indices, warm_indices = np.nonzero(values > tau)
    ordered = sorted(zip(cold_indices.tolist(), warm_indices.tolist()),
                     key=lambda pair: (-values[pair], pair[1], pair[0]))
    used_cold: set[int] = set()
    used_warm: set[int] = set()
    matches = []
    for cold, warm in ordered:
        if cold not in used_cold and warm not in used_warm:
            matches.append((cold, warm))
            used_cold.add(cold)
            used_warm.add(warm)
    return sorted(matches, key=lambda pair: pair[1])


def matching_diagnostics(
    utility: ArrayLike, tau: float,
    matches: Sequence[tuple[int, int]] | None = None,
) -> dict:
    """One-user graph diagnostics; no labels and no alternative MAP evaluation.

    Conflict counts are *nodes* incident to more than one surviving edge, not
    pairs.  Efficiency is matched edge count / surviving edge count (zero if
    the latter is zero).  Greedy difference compares edge sets; tied optima can
    therefore differ without a difference in objective value.
    """
    values = _validated_utility(utility, tau)
    selected = exact_matching(values, tau) if matches is None else list(matches)
    _validate_matches(values, tau, selected)
    edges = values > tau
    surviving = int(edges.sum())
    greedy = _diagnostic_greedy(values, tau)
    total_weight = float(sum(values[c, w] - tau for c, w in selected))
    greedy_weight = float(sum(values[c, w] - tau for c, w in greedy))
    return {
        "cold_nodes": int(values.shape[0]),
        "warm_nodes": int(values.shape[1]),
        "pair_rows": int(values.size),
        "edges_above_tau": surviving,
        "matched_edges": len(selected),
        "matching_efficiency": len(selected) / surviving if surviving else 0.0,
        "matched_utility_total": float(sum(values[c, w] for c, w in selected)),
        "matched_weight_total": total_weight,
        "same_warm_conflict_count": int(np.count_nonzero(edges.sum(axis=0) > 1)),
        "same_cold_conflict_count": int(np.count_nonzero(edges.sum(axis=1) > 1)),
        "greedy_would_differ": set(selected) != set(greedy),
        "greedy_weight_total_diagnostic_only": greedy_weight,
        "exact_minus_greedy_weight_diagnostic_only": total_weight - greedy_weight,
    }


def apply_admissions(
    warm_articles: Sequence[str], cold_articles: Sequence[str],
    matches: Sequence[tuple[int, int]],
) -> list[str]:
    """Replace matched W0 slots in place, preserving all other Warm positions.

    The caller must already have excluded *all* Cold articles present in W0.
    Reject overlap even if that overlapping Cold article is not matched.  No
    extra candidate, fallback, truncation, or Warm reordering is introduced.
    """
    warm, cold = list(warm_articles), list(cold_articles)
    if len(warm) != 12 or len(set(warm)) != 12:
        raise ValueError("W0 must contain exactly 12 unique articles")
    if len(cold) > 50 or len(set(cold)) != len(cold):
        raise ValueError("Cold candidates must be unique and at most 50")
    if set(warm) & set(cold):
        raise ValueError("Cold admission candidates must exclude W0 overlap")
    if len({c for c, _ in matches}) != len(matches):
        raise ValueError("a Cold candidate may be inserted at most once")
    if len({w for _, w in matches}) != len(matches):
        raise ValueError("a Warm slot may be replaced at most once")
    result = warm.copy()
    for cold_index, warm_index in matches:
        if not (0 <= cold_index < len(cold) and 0 <= warm_index < 12):
            raise ValueError("matching index is outside the candidate lists")
        result[warm_index] = cold[cold_index]
    if len(result) != 12 or len(set(result)) != 12:
        raise AssertionError("admission violated the unique Top12 invariant")
    return result
