"""Read-only P4.2D statistics; no estimator or inference code."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def summary(x):
    x = np.asarray(x, dtype=float).ravel()
    v = x[np.isfinite(x)]
    keys = ['min', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'p99', 'max']
    return {'rows': len(x), 'finite_rows': len(v), 'mean': float(v.mean()) if len(v) else None,
            **(dict(zip(keys, map(float, np.quantile(v, [0,.1,.25,.5,.75,.9,.95,.99,1])))) if len(v) else
            dict.fromkeys(keys))}


def discrimination(score, label):
    x, y = np.asarray(score, float).ravel(), np.asarray(label, bool).ravel()
    valid = np.isfinite(x)
    x, y = x[valid], y[valid]
    n, p = len(y), int(y.sum())
    out = dict(rows=n, positives=p, base_rate=p/n if n else None, roc_auc=None, pr_auc=None)
    if not n or not p or p == n:
        return out
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1])+1]
    total = np.diff(np.r_[starts, n])
    pos = np.add.reduceat(y.astype(np.int64), starts)
    neg = total-pos
    out['roc_auc'] = float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(p*(n-p)))
    out['pr_auc'] = float(np.sum(pos[::-1]/p*np.cumsum(pos[::-1])/np.cumsum(total[::-1])))
    return out


def correlations(x, y):
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(rows=len(x), pearson=None, spearman=None)
    return dict(rows=len(x), pearson=float(np.corrcoef(x,y)[0,1]),
                spearman=float(np.corrcoef(rankdata(x),rankdata(y))[0,1]))


def rate(p, n):
    return int(p)/int(n) if n else None


def rank_metrics(ranks):
    r = np.asarray(ranks, float)
    return {'positive_candidates': len(r), 'recall': {str(k): rate((r<=k).sum(),len(r)) for k in (1,5,10,20,50)},
            'mean_reciprocal_positive_rank': float(np.mean(1/r)) if len(r) else None}


def single_ap_delta(relevance, cold_label, truth_count):
    """Closed-form all12-slot AP changes; independent of matching and q."""
    r = np.asarray(relevance, np.int64)
    inv = 1/np.arange(1,13,dtype=float)
    prefix = np.cumsum(r, axis=1)-r
    weighted = r*inv
    suffix = np.cumsum(weighted[:,::-1],axis=1)[:,::-1]-weighted
    return (np.asarray(cold_label)[:,None]-r)*((prefix+1)*inv+suffix)/np.minimum(truth_count,12)[:,None]
