"""Signed exactly-K additive matching followed by exact final-list AP selection.

This constrained procedure is not the global combinatorial maximum-AP oracle.
Negative and zero edges remain available when exactly K requires them.
"""
from pathlib import Path
import time
import gc

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .p41a_contract import read_json, write_json
from .p42f_contract import RUN_ID as F_RUN, WINDOWS
from .p42f_data import save_frame
from .p42d_stats import single_ap_delta
from .p43a_policy import check_deadline, exact_ap, contexts, SEGMENTS


def exactly_k_matching(weight, k):
    """Canonical assignment with exactly K real edges; None means infeasible.

    Real Cold rows and real Warm columns precede dummies. The forbidden
    dummy-to-dummy block forces exactly K real-to-real assignments. No score
    offset, sign gate, epsilon perturbation or posthoc trimming is applied.
    """
    w = np.asarray(weight, float)
    if w.ndim != 2 or not np.isfinite(w).all() or k < 0:
        raise ValueError('invalid signed exact-K graph')
    nc, nw = w.shape
    if k > min(nc, nw):
        return None
    if k == 0:
        return []
    size = nc + nw - k
    cost = np.zeros((size, size))
    cost[:nc, :nw] = -w
    cost[nc:, nw:] = np.inf
    rows, cols = linear_sum_assignment(cost)
    matches = sorted([(int(r), int(c)) for r, c in zip(rows, cols) if r < nc and c < nw], key=lambda x: (x[1], x[0]))
    assert len(matches) == k and len({i for i, _ in matches}) == k and len({j for _, j in matches}) == k
    return matches


def oracle_user(warm, cold, truth, weight):
    if len(warm) != 12 or len(set(warm)) != 12 or len(set(cold)) != len(cold) or set(warm).intersection(cold):
        raise ValueError('invalid frozen action items')
    if np.shape(weight) != (len(cold), 12):
        raise ValueError('oracle score matrix shape mismatch')
    aps = np.full(13, np.nan)
    by_k = []
    best_k, best_ap, best_items = 0, exact_ap(warm, truth), list(warm)
    for k in range(13):
        matches = exactly_k_matching(weight, k)
        by_k.append(matches)
        if matches is None:
            continue
        items = list(warm)
        for ci, j in matches:
            items[j] = cold[ci]
        aps[k] = exact_ap(items, truth)
        if aps[k] > best_ap:
            best_k, best_ap, best_items = k, float(aps[k]), items
    return dict(k_star=best_k, best_ap=best_ap, best_items=best_items, ap_by_k=aps, matching_by_k=by_k)


def restricted_oracle_user(warm, cold, cold_ranks, truth, weight):
    """P4.1A Top10 x slots10-12 x at-most1; ties rank,slot,article."""
    baseline = exact_ap(warm, truth)
    candidates = [(float(weight[i, j]), int(cold_ranks[i]), j, str(cold[i]), i)
                  for i in range(len(cold)) if cold_ranks[i] <= 10 for j in (9, 10, 11) if weight[i, j] > 0]
    if not candidates:
        return baseline, []
    _, _, j, _, i = min(candidates, key=lambda x: (-x[0], x[1], x[2], x[3]))
    items = list(warm)
    items[j] = cold[i]
    return exact_ap(items, truth), [(i, j)]


def _date_oracle(repo, root, cutoff, deadline_epoch):
    if cutoff >= '2020-09-16':
        raise ValueError('sealed final week prohibited')
    folder = Path(root) / 'oracle' / cutoff
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'RESULT.json').exists():
        return read_json(folder / 'RESULT.json')
    check_deadline(deadline_epoch)
    data = joblib.load(Path(repo) / 'artifacts/phase4' / F_RUN / 'prepared' / cutoff / 'data.joblib')
    n = len(data['users'])
    truths, valid, base = contexts(data)
    resumed = (folder / 'PROGRESS.json').exists()
    progress = read_json(folder / 'PROGRESS.json') if resumed else dict(next_user=0, seconds=0.)
    mode = 'r+' if resumed else 'w+'
    arrays = {}
    definitions = {'ap_by_k': ((n, 13), 'float64'), 'matching_by_k': ((n, 13, 12, 2), 'int32'),
                   'k_star': ((n,), 'uint8'), 'best_segments': ((n, 5), 'float64'),
                   'restricted_ap': ((n,), 'float64'), 'restricted_match': ((n, 2), 'int32')}
    for name, (shape, dtype) in definitions.items():
        arrays[name] = np.lib.format.open_memmap(folder / (name + '.npy'), mode=mode, dtype=dtype, shape=shape)
    if not resumed:
        arrays['ap_by_k'][:] = np.nan
        arrays['matching_by_k'][:] = -1
        arrays['restricted_match'][:] = -1
    groups = data['cold'].groupby('user_index', sort=False).indices
    start = time.perf_counter()
    next_user = progress['next_user']
    def checkpoint():
        for a in arrays.values():
            a.flush()
        write_json(folder / 'PROGRESS.json', dict(status='partial', next_user=next_user,
                   seconds=progress['seconds'] + time.perf_counter() - start))
    try:
        for ui in range(next_user, n):
            check_deadline(deadline_epoch)
            indices = np.asarray(groups.get(ui, []), dtype=np.int64)
            cold = data['cold'].iloc[indices]
            y = single_ap_delta(np.repeat(data['relevance'][ui:ui + 1], len(cold), axis=0),
                                cold.target.to_numpy(), np.repeat(data['truth_count'][ui], len(cold)))
            warm, articles = list(data['warm_lists'][ui]), cold.article_id.tolist()
            result = oracle_user(warm, articles, truths[ui][0], y)
            arrays['ap_by_k'][ui] = result['ap_by_k']
            arrays['k_star'][ui] = result['k_star']
            arrays['best_segments'][ui] = [exact_ap(result['best_items'], t) for t in truths[ui]]
            for k, matching in enumerate(result['matching_by_k']):
                if matching:
                    arrays['matching_by_k'][ui, k, :len(matching)] = [(indices[i], j) for i, j in matching]
            rap, selected = restricted_oracle_user(warm, articles, cold.b0_rank.to_numpy(), truths[ui][0], y)
            arrays['restricted_ap'][ui] = rap
            if selected:
                i, j = selected[0]
                arrays['restricted_match'][ui] = [indices[i], j]
            next_user = ui + 1
            if next_user % 256 == 0:
                checkpoint()
    finally:
        checkpoint()
    best = arrays['best_segments']
    means = np.divide(best.sum(axis=0), valid.sum(axis=0), out=np.zeros(5), where=valid.sum(axis=0) != 0)
    baselines = np.divide(base.sum(axis=0), valid.sum(axis=0), out=np.zeros(5), where=valid.sum(axis=0) != 0)
    kstar = np.asarray(arrays['k_star'])
    result = dict(status='completed', cutoff=cutoff, users=n, overall_map=float(means[0]),
                  delta_map=float(means[0] - baselines[0]), w0_map=float(baselines[0]),
                  definition='for each exactlyK signed max-sum edge matching, evaluate exact final AP; then choose smallest maximizing K; not global combinatorial AP oracle',
                  k_star_distribution={str(k): int((kstar == k).sum()) for k in range(13)},
                  segments={name: dict(map=float(means[j]) if valid[:, j].any() else None,
                                       delta=float(means[j] - baselines[j]) if valid[:, j].any() else None,
                                       truth_users=int(valid[:, j].sum())) for j, name in enumerate(SEGMENTS) if j},
                  by_k={str(k): dict(feasible_users=int(np.isfinite(arrays['ap_by_k'][:, k]).sum()),
                                     map_among_feasible=float(np.nanmean(arrays['ap_by_k'][:, k])) if np.isfinite(arrays['ap_by_k'][:, k]).any() else None)
                        for k in range(13)},
                  restricted=dict(map=float(arrays['restricted_ap'].mean()),
                                  delta=float((arrays['restricted_ap'] - base[:, 0]).mean()),
                                  admitted_users=int((arrays['restricted_match'][:, 0] >= 0).sum())),
                  seconds=progress['seconds'] + time.perf_counter() - start,
                  artifacts={name: str((folder / (name + '.npy')).resolve()) for name in arrays}, final_week='not_run')
    window = next((w for w, date in WINDOWS.items() if date == cutoff), None)
    if window:
        old = read_json(Path(repo) / 'reports/phase4/P4_1A_metrics.json')['windows'][window]
        assert old['users'] == n
        np.testing.assert_allclose(result['restricted']['map'], old['map@12'], rtol=0, atol=1e-12)
        np.testing.assert_allclose(result['restricted']['delta'], old['delta_MAP_vs_W0'], rtol=0, atol=1e-12)
        assert result['restricted']['admitted_users'] == old['positive_opportunity_users']
        result['P4_1A_replay'] = dict(status='pass', reference=old['map@12'], current=result['restricted']['map'])
    save_frame(pd.DataFrame(dict(customer_id=data['users'], k_star=kstar,
               oracle_ap=np.asarray(best[:, 0]), baseline_ap=base[:, 0],
               oracle_gain=np.asarray(best[:, 0]) - base[:, 0])), folder / 'user-oracle.parquet')
    write_json(folder / 'RESULT.json', result)
    del data, arrays
    gc.collect()
    return result


def run_oracle(repo, root, dates, deadline_epoch=None):
    dates = list(dict.fromkeys(dates))
    if any(date >= '2020-09-16' for date in dates):
        raise ValueError('sealed final week prohibited')
    result = dict(stage='P4.3A', status='running', dates={}, final_week='not_run')
    destination = Path(repo) / 'reports/phase4/p4_3a_oracle_headroom.json'
    try:
        for cutoff in dates:
            check_deadline(deadline_epoch)
            result['dates'][cutoff] = _date_oracle(repo, root, cutoff, deadline_epoch)
            write_json(destination, result)
        development = [result['dates'][date] for date in WINDOWS.values() if date in result['dates']]
        result['development_mean_map'] = float(np.mean([v['overall_map'] for v in development])) if development else None
        result['development_mean_delta'] = float(np.mean([v['delta_map'] for v in development])) if development else None
        result['status'] = 'completed'
    except TimeoutError:
        result['status'] = 'paused_deadline'
        raise
    finally:
        write_json(destination, result)
    return result
