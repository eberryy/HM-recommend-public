"""Pure, label-separated descriptive statistics for P4.1A (no estimator)."""
from __future__ import annotations

import numpy as np
import pandas as pd

LCM = 27720
WEIGHTS = LCM // np.arange(1, 13, dtype=np.int64)


def ap_units(relevance):
    r = np.asarray(relevance, dtype=np.int64)
    return np.sum(r * np.cumsum(r, axis=-1) * WEIGHTS, axis=-1)


def replacement_units(relevance, cold_positive, slot):
    """Exact AP-numerator change; slot is one-based. No floating sign test."""
    r = np.asarray(relevance, dtype=np.int64)
    j = slot - 1
    weight = (r[..., :j].sum(axis=-1) + 1) * WEIGHTS[j]
    weight = weight + (r[..., j+1:] * WEIGHTS[j+1:]).sum(axis=-1)
    return (np.asarray(cold_positive, dtype=np.int64) - r[..., j]) * weight


def distribution(values):
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    out = {"rows": len(x), "finite_rows": len(finite)}
    keys = ("mean", "std", "min", "p10", "p25", "median", "p75", "p90", "max", "iqr")
    if not len(finite):
        return {**out, **dict.fromkeys(keys)}
    q = np.quantile(finite, [0, .1, .25, .5, .75, .9, 1])
    return {**out, "mean": float(finite.mean()), "std": float(finite.std()),
            **{key: float(v) for key, v in zip(keys[2:-1], q)}, "iqr": float(q[4]-q[2])}


def auc_metrics(score, label):
    x, y = np.asarray(score, float), np.asarray(label, bool)
    valid = np.isfinite(x)
    x, y = x[valid], y[valid]
    n, p = len(x), int(y.sum())
    out = {"eligible_rows": n, "beneficial_rows": p, "base_rate": p/n if n else None,
           "roc_auc": None, "pr_auc": None}
    if not n:
        return out
    _, inv, total = np.unique(x, return_inverse=True, return_counts=True)
    pos = np.bincount(inv, weights=y, minlength=len(total))
    neg = total - pos
    if 0 < p < n:
        out["roc_auc"] = float(np.sum(pos * (np.cumsum(neg) - .5*neg))/(p*(n-p)))
    if p:
        out["pr_auc"] = float(np.sum(pos[::-1]/p * np.cumsum(pos[::-1])/np.cumsum(total[::-1])))
    return out


def unlabeled_bins(score, bins):
    """No label parameter by construction. Equal scores always stay together."""
    x = np.asarray(score, float)
    valid = np.isfinite(x)
    boundaries = np.quantile(x[valid], np.arange(1, bins)/bins) if valid.any() else np.full(bins-1, np.nan)
    assignments = np.full(len(x), -1, dtype=np.int16)
    assignments[valid] = np.searchsorted(boundaries, x[valid], side="left")
    return boundaries, assignments


def reliability(score, label, bins):
    boundaries, ids = unlabeled_bins(score, bins)
    y = np.asarray(label, bool)
    valid = ids >= 0
    n, p = int(valid.sum()), int(y[valid].sum())
    base = p/n if n else None
    rows = []
    for i in range(bins):
        mask = ids == i
        count, positive = int(mask.sum()), int(y[mask].sum())
        rate = positive/count if count else None
        rows.append({"bin": i+1, "rows": count, "beneficial_rows": positive,
                     "beneficial_rate": rate, "lift": rate/base if rate is not None and base else None})
    return {"boundaries": [float(v) if np.isfinite(v) else None for v in boundaries],
            "boundary_method": "label-free signed confidence quantile; ties side=left",
            "eligible_rows": n, "base_rate": base, "bins": rows}


def normalize_cold50(cold):
    """Score population is complete Cold50, never Top10/opportunity/truth rows."""
    df = cold.copy()
    score = df.b0_score.astype(float).where(df.b0_score_available.astype(bool))
    score = score.where(np.isfinite(score))
    grouped = score.groupby(df.customer_id, sort=False)
    n = grouped.transform("count")
    sd = grouped.transform(lambda x: x.std(ddof=0))
    mean, median = grouped.transform("mean"), grouped.transform("median")
    df["b0_user_percentile"] = ((grouped.rank(method="average")-1)/(n-1)).where(n >= 2)
    df["b0_user_zscore"] = ((score-mean)/sd).where(sd > 1e-12)
    for rank in (2, 5):
        ref = pd.Series(score[df.b0_rank == rank].values, index=df.loc[df.b0_rank == rank, "customer_id"])
        df[f"margin_to_rank{rank}"] = score - df.customer_id.map(ref)
    df["margin_to_user_median"] = score - median
    for key in ("b0_user_percentile", "b0_user_zscore", "margin_to_rank2", "margin_to_rank5", "margin_to_user_median"):
        df[key + "_available"] = np.isfinite(df[key]).astype(np.uint8)
    df["b0_score"] = score
    df["b0_delta_vs_m4"] = df.b0_delta_vs_m4.where(df.b0_delta_available.astype(bool))
    return df
