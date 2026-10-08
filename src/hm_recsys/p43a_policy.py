"""P4.3A exact bounded policy enumeration with per-user decision-mask reuse.

No score-sign gate is used. Full-user stable edge ranks define strictly positive
1/rank matching weights. Different policies can share an identical final list;
its AP is computed from that actual list once, never from summed edge deltas.
Global gates use empirical average-tie percentiles over complete window edges,
not interpolated score quantiles. Constant scores all have percentile0.5.
"""
from pathlib import Path
from itertools import product
import time

import numpy as np

from .p41a_contract import read_json, write_json


SEGMENTS = ('overall', 'warm_21_plus', 'strict_cold', 'sparse1_5', 'all_cold_sparse')
CAPS = (1, 2, 3, 12)


def check_deadline(deadline_epoch):
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise TimeoutError('P4.3A authorized wall-clock deadline reached; preserve progress')


def policies():
    grid = product((1, 2, 3, 5, 10), (90, 95, 98, 99, 99.5, 99.9),
                   (5, 10, 20, 50), (12, 10, 7, 1), CAPS)
    return [dict(id=f'p43a-policy-{i:04d}', top_edge=e, global_percentile=g,
                 candidate_topk=c, slot_floor=s, max_admissions=k)
            for i, (e, g, c, s, k) in enumerate(grid)]


def exact_ap(items, truth):
    if len(items) != 12 or len(set(items)) != 12:
        raise ValueError('recommendation must contain exactly12 unique articles')
    if not truth:
        return 0.
    hit = np.fromiter((item in truth for item in items), dtype=np.int64)
    return float(np.sum(hit * np.cumsum(hit) / np.arange(1, 13)) / min(len(truth), 12))


def contexts(data):
    """Same segment truth sets and truth-user denominators as frozen Phase4."""
    users = data['users']
    truth = data['truth']
    count = truth.interaction_count_before_cutoff
    dictionaries = [data['truthsets']]
    for mask in (count >= 21, count == 0, (count >= 1) & (count <= 5), count <= 5):
        dictionaries.append({u: set(g.article_id) for u, g in truth.loc[mask].groupby('customer_id', sort=False)})
    truths = [[dictionary.get(u, set()) for dictionary in dictionaries] for u in users]
    valid = np.array([[True] + [bool(t) for t in row[1:]] for row in truths], bool)
    base = np.array([[exact_ap(items, t) for t in row] for items, row in zip(data['warm_lists'], truths)])
    return truths, valid, base


def global_midrank_thresholds(scores, levels):
    """Smallest accepted observed value for each average-tie percentile gate.

    L=count(score<s), R=count(score<=s), percentile=(L+R-1)/(2*(N-1)).
    A singleton has percentile0.5. None means no score passes (or no edges).
    Returning an observed-value boundary makes >= exactly equivalent on this
    scoring population, without allocating a percentile array per policy.
    """
    values = np.asarray(scores).ravel()
    if not np.isfinite(values).all():
        raise ValueError('nonfinite score in global empirical percentile')
    if any(not 0 <= level <= 100 for level in levels):
        raise ValueError('percentile gate must be between0 and100')
    if len(values) == 0:
        return {level: None for level in levels}
    unique, count = np.unique(values, return_counts=True)
    right = np.cumsum(count, dtype=np.int64)
    left = right - count
    percentiles = (left + right - 1) / (2. * (len(values) - 1)) if len(values) > 1 else np.array([.5])
    result = {}
    for level in levels:
        i = int(np.searchsorted(percentiles, level / 100, side='left'))
        result[level] = float(unique[i]) if i < len(unique) else None
    return result


def matching_table(cold_indices, warm_slots, weights, caps=CAPS):
    """All <=10-edge subgraphs and exact maximum-weight matching of size <=K.

    Every feasible subset is initialized explicitly; subset dynamic programming
    propagates its best score into all containing eligibility masks. Equal sums
    choose the smaller unsigned selected-edge bitmask, fixed before outcomes.
    """
    cold, slot, weight = map(np.asarray, (cold_indices, warm_slots, weights))
    n = len(cold)
    if n > 10 or slot.shape != cold.shape or weight.shape != cold.shape:
        raise ValueError('matching table requires at most10 aligned edges')
    if not np.isfinite(weight).all():
        raise ValueError('nonfinite matching weight')
    if any(k < 0 for k in caps):
        raise ValueError('negative admission cap')
    size = 1 << n
    masks = np.arange(size, dtype=np.int32)
    conflicts = [sum(1 << j for j in range(n) if j != i and
                     (cold[i] == cold[j] or slot[i] == slot[j])) for i in range(n)]
    valid = np.ones(size, bool)
    cardinality = np.zeros(size, np.int8)
    score = np.zeros(size)
    for mask in range(1, size):
        bit = mask & -mask
        i, rest = bit.bit_length() - 1, mask ^ bit
        valid[mask] = valid[rest] and not (rest & conflicts[i])
        cardinality[mask] = cardinality[rest] + 1
        score[mask] = score[rest] + weight[i]
    best = np.where(valid[None, :] & (cardinality[None, :] <= np.asarray(caps)[:, None]), score, -np.inf)
    choice = np.broadcast_to(masks, (len(caps), size)).copy()
    for i in range(n):
        ids = masks[(masks & (1 << i)) != 0]
        other = ids ^ (1 << i)
        improve = ((best[:, other] > best[:, ids]) |
                   ((best[:, other] == best[:, ids]) & (choice[:, other] < choice[:, ids])))
        best[:, ids] = np.where(improve, best[:, other], best[:, ids])
        choice[:, ids] = np.where(improve, choice[:, other], choice[:, ids])
    return choice.astype(np.uint16), best


def selected_lists(warm, cold_articles, top_local_edges, choice):
    """Rebuild one exact Top12 from a compact Top10 edge-selection bitmask."""
    result = list(warm)
    used_cold, used_slot = set(), set()
    for i, edge in enumerate(top_local_edges):
        if not (int(choice) & (1 << i)):
            continue
        ci, slot = divmod(int(edge), 12)
        if ci in used_cold or slot in used_slot:
            raise ValueError('selected mask is not one-to-one')
        used_cold.add(ci)
        used_slot.add(slot)
        result[slot] = cold_articles[ci]
    if len(set(result)) != 12:
        raise ValueError('selected list lost unique12 invariant or contains W0 overlap')
    return result, sorted(used_cold), sorted(used_slot)


def _checkpoint(path, next_user, elapsed, sums, counts, decisions, top_edges):
    decisions.flush()
    top_edges.flush()
    # Single-file atomic numerical state; JSON is descriptive only.
    temporary = path / 'accumulators.npz.part'
    with temporary.open('wb') as stream:
        np.savez(stream, next_user=np.array(next_user), elapsed=np.array(elapsed), sums=sums, counts=counts)
    temporary.replace(path / 'accumulators.npz')
    write_json(path / 'PROGRESS.json', dict(status='partial', completed_users=next_user, seconds=elapsed))


def evaluate_grid(data, scores, policy_list, output_dir, deadline_epoch=None):
    """Return every completed policy's exact metrics, with resumable shared work."""
    folder = Path(output_dir)
    if data['cutoff'] >= '2020-09-16':
        raise ValueError('sealed final week is prohibited')
    scores = np.asarray(scores)
    if scores.shape != (len(data['cold']), 12) or not np.isfinite(scores).all():
        raise ValueError('invalid complete action score matrix')
    if not policy_list or len({p['id'] for p in policy_list}) != len(policy_list):
        raise ValueError('policies must be nonempty with unique IDs')
    for p in policy_list:
        assert p['top_edge'] in (1, 2, 3, 5, 10) and p['max_admissions'] in CAPS
        assert p['candidate_topk'] in (5, 10, 20, 50) and p['slot_floor'] in (1, 7, 10, 12)
        assert p['global_percentile'] in (90, 95, 98, 99, 99.5, 99.9)
    start = time.perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(cutoff=data['cutoff'], users=len(data['users']), cold_rows=len(data['cold']), policies=policy_list,
                    top_edge_rule='full-user stable score rank then AND candidate/slot/global restrictions',
                    global_gate='complete-window average-tie empirical percentile (L+R-1)/(2*(N-1)) >= level/100; singleton0.5; constant0.5',
                    weights='positive 1/full-user stable edge rank; no raw score>0 gate',
                    tie_rule='smaller unsigned selected Top10 edge bitmask', final_week='not_run')
    if (folder / 'POLICIES.json').exists():
        assert read_json(folder / 'POLICIES.json') == metadata
    else:
        write_json(folder / 'POLICIES.json', metadata)
    if (folder / 'EVALUATION.json').exists():
        return read_json(folder / 'EVALUATION.json')['rows']
    check_deadline(deadline_epoch)
    n, npolicy = len(data['users']), len(policy_list)
    truths, valid, base = contexts(data)
    denominators = valid.sum(axis=0)
    base_mean = np.divide(base.sum(axis=0), denominators, out=np.zeros(5), where=denominators != 0)
    threshold_levels = sorted({p['global_percentile'] for p in policy_list})
    thresholds = global_midrank_thresholds(scores, threshold_levels)
    cap_indices = np.array([CAPS.index(p['max_admissions']) for p in policy_list])
    top_limits = np.array([p['top_edge'] for p in policy_list])
    global_cuts = np.array([thresholds[p['global_percentile']] if thresholds[p['global_percentile']] is not None
                            else np.inf for p in policy_list])
    cold_limits = np.array([p['candidate_topk'] for p in policy_list])
    slot_limits = np.array([p['slot_floor'] for p in policy_list])
    decision_path, edge_path = folder / 'policy-user-selected-mask.npy', folder / 'user-top10-global-edge.npy'
    resumed = (folder / 'accumulators.npz').exists()
    mode = 'r+' if resumed else 'w+'
    decisions = np.lib.format.open_memmap(decision_path, mode=mode, dtype=np.uint16, shape=(npolicy, n))
    top_edges = np.lib.format.open_memmap(edge_path, mode=mode, dtype=np.int64, shape=(n, 10))
    if resumed:
        with np.load(folder / 'accumulators.npz') as old:
            next_user, previous = int(old['next_user']), float(old['elapsed'])
            sums, counts = old['sums'], old['counts']
    else:
        decisions[:] = 0
        top_edges[:] = -1
        next_user, previous = 0, 0.
        # First5 are exact final-list AP sums; last5 are per-user AP deltas.
        # Summing deltas directly makes a truly unchanged policy exactly zero.
        sums, counts = np.zeros((npolicy, 10)), np.zeros((npolicy, 4), np.int64)
    groups = data['cold'].groupby('user_index', sort=False).indices
    try:
        for ui in range(next_user, n):
            check_deadline(deadline_epoch)
            indices = np.asarray(groups.get(ui, []), dtype=np.int64)
            if len(indices) == 0:
                sums[:, :5] += base[ui]
                next_user = ui + 1
                continue
            cc = data['cold'].iloc[indices]
            articles = cc.article_id.tolist()
            assert len(set(articles)) == len(articles) <= 50
            assert not set(articles).intersection(data['warm_lists'][ui])
            flat = scores[indices].ravel()
            edges = np.argsort(-flat, kind='stable')[:10]
            ci, slot = edges // 12, edges % 12
            length = len(edges)
            top_edges[ui, :length] = indices[ci] * 12 + slot
            flags = ((np.arange(length)[None, :] < top_limits[:, None]) &
                     (flat[edges][None, :] >= global_cuts[:, None]) &
                     (cc.b0_rank.to_numpy()[ci][None, :] <= cold_limits[:, None]) &
                     (slot[None, :] + 1 >= slot_limits[:, None]))
            allowed = (flags * (1 << np.arange(length))).sum(axis=1)
            table, _ = matching_table(ci, slot, 1. / np.arange(1, length + 1))
            chosen = table[cap_indices, allowed]
            decisions[:, ui] = chosen
            unique, inverse = np.unique(chosen, return_inverse=True)
            metrics = np.empty((len(unique), 5))
            additions = np.empty((len(unique), 4), np.int64)
            for j, mask in enumerate(unique):
                items, cold_selected, warm_selected = selected_lists(data['warm_lists'][ui], articles, edges, mask)
                metrics[j] = [exact_ap(items, truth) for truth in truths[ui]]
                ins = sum(articles[c] in truths[ui][0] for c in cold_selected)
                rem = sum(data['warm_lists'][ui, slot] in truths[ui][0] for slot in warm_selected)
                additions[j] = [ins, rem, bool(cold_selected), len(cold_selected)]
            sums[:, :5] += metrics[inverse]
            sums[:, 5:] += metrics[inverse] - base[ui]
            counts += additions[inverse]
            next_user = ui + 1
            if next_user % 256 == 0:
                _checkpoint(folder, next_user, previous + time.perf_counter() - start, sums, counts, decisions, top_edges)
    finally:
        elapsed = previous + time.perf_counter() - start
        _checkpoint(folder, next_user, elapsed, sums, counts, decisions, top_edges)
    deltas = np.divide(sums[:, 5:], denominators, out=np.zeros((npolicy, 5)), where=denominators != 0)
    means = base_mean + deltas
    rows = []
    for i, p in enumerate(policy_list):
        segments = {name: dict(map=float(means[i, j]) if denominators[j] else None,
                               delta=float(deltas[i, j]) if denominators[j] else None,
                               map12=float(means[i, j]) if denominators[j] else None,
                               delta_vs_w0=float(deltas[i, j]) if denominators[j] else None,
                               truth_users=int(denominators[j])) for j, name in enumerate(SEGMENTS) if j}
        rows.append(dict(policy_id=p['id'], policy=p, overall_map=float(means[i, 0]),
                         delta_map=float(deltas[i, 0]), segments=segments,
                         inserted_positives=int(counts[i, 0]), removed_positives=int(counts[i, 1]),
                         admitted_users=int(counts[i, 2]), coverage=float(counts[i, 2] / n),
                         replacements=int(counts[i, 3]), users=n,
                         global_score_threshold=float(global_cuts[i]) if np.isfinite(global_cuts[i]) else None,
                         global_score_threshold_status='finite' if np.isfinite(global_cuts[i]) else
                             'no_score_reaches_percentile' if scores.size else 'empty_action_space',
                         seconds=elapsed / npolicy, shared_grid_seconds=elapsed,
                         timing='shared total grid wall time equally allocated, not individual-policy standalone timing',
                         saved_decisions=str(decision_path.resolve()), saved_top_edges=str(edge_path.resolve()),
                         decision_policy_index=i, status='completed', exact_final_list_AP=True))
    write_json(folder / 'EVALUATION.json', dict(status='completed', cutoff=data['cutoff'], rows=rows,
                                           seconds=elapsed, completed_users=n, final_week='not_run'))
    return rows


def evaluate_residual(data, scores, output_dir, deadline_epoch=None):
    """E only: challenger TopK by allowed-slot max, then positive-score matching.

    The 16 policies are separate from the 1920 universal policies. Original
    Warm positions remain fixed. Probability0 is an unmatched tie, not a new
    learned threshold; all strictly positive allowed edges remain available.
    """
    from .p42_matching import exact_matching
    scores = np.asarray(scores)
    if data['cutoff'] >= '2020-09-16':
        raise ValueError('sealed final week prohibited')
    if scores.shape != (len(data['cold']), 12) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('E requires the complete binary-probability action score matrix')
    configs = [dict(id=f'p43a-E-k{k}-floor{floor}', candidate_topk=k, slot_floor=floor)
               for k, floor in product((5, 10, 20, 50), (12, 10, 7, 1))]
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(arm='E', cutoff=data['cutoff'], policies=configs, users=len(data['users']),
                    cold_rows=len(data['cold']), candidate_order='descending allowed-slot max score, then B0 rank, article',
                    matching='positive probability edges, one-to-one, no hard admission cap; score0 rejected as empty tie',
                    final_week='not_run')
    if (folder / 'POLICIES.json').exists():
        assert read_json(folder / 'POLICIES.json') == metadata
    else:
        write_json(folder / 'POLICIES.json', metadata)
    if (folder / 'EVALUATION.json').exists():
        return read_json(folder / 'EVALUATION.json')['rows']
    check_deadline(deadline_epoch)
    start = time.perf_counter()
    truths, valid, base = contexts(data)
    denominator = valid.sum(axis=0)
    baseline = np.divide(base.sum(axis=0), denominator, out=np.zeros(5), where=denominator != 0)
    n, p = len(data['users']), len(configs)
    resumed = (folder / 'E_accumulators.npz').exists()
    decisions = np.lib.format.open_memmap(folder / 'E-policy-user-matched-pairs.npy', mode='r+' if resumed else 'w+',
                                         dtype=np.int32, shape=(p, n, 12, 2))
    if resumed:
        with np.load(folder / 'E_accumulators.npz') as old:
            sums, counts = old['sums'], old['counts']
            next_user, previous = int(old['next_user']), float(old['elapsed'])
    else:
        decisions[:] = -1
        sums, counts = np.zeros((p, 10)), np.zeros((p, 4), np.int64)
        next_user, previous = 0, 0.
    def checkpoint():
        decisions.flush()
        temp = folder / 'E_accumulators.npz.part'
        with temp.open('wb') as stream:
            np.savez(stream, sums=sums, counts=counts, next_user=np.array(next_user),
                     elapsed=np.array(previous + time.perf_counter() - start))
        temp.replace(folder / 'E_accumulators.npz')
        write_json(folder / 'PROGRESS.json', dict(status='partial', completed_users=next_user,
                                                 seconds=previous + time.perf_counter() - start))
    groups = data['cold'].groupby('user_index', sort=False).indices
    try:
        for ui in range(next_user, n):
            check_deadline(deadline_epoch)
            indices = np.asarray(groups.get(ui, []), np.int64)
            if len(indices) == 0:
                sums[:, :5] += base[ui]
                next_user = ui + 1
                continue
            cc = data['cold'].iloc[indices]
            articles = cc.article_id.to_numpy()
            assert not set(articles).intersection(data['warm_lists'][ui])
            local = scores[indices]
            orders = {}
            for floor in (12, 10, 7, 1):
                best = local[:, floor - 1:].max(axis=1)
                orders[floor] = np.lexsort((articles.astype(str), cc.b0_rank.to_numpy(), -best))
            per_metrics, per_counts = np.empty((p, 5)), np.empty((p, 4), np.int64)
            cache = {}
            for pi, config in enumerate(configs):
                selected = orders[config['slot_floor']][:config['candidate_topk']]
                allowed_scores = local[selected].copy()
                allowed_scores[:, :config['slot_floor'] - 1] = 0.
                matches = exact_matching(allowed_scores, 0.)
                actual = tuple((int(selected[ci]), slot) for ci, slot in matches)
                decisions[pi, ui] = -1
                if actual:
                    decisions[pi, ui, :len(actual)] = [(indices[ci], slot) for ci, slot in actual]
                if actual not in cache:
                    items = list(data['warm_lists'][ui])
                    ins = rem = 0
                    for ci, slot in actual:
                        ins += int(articles[ci] in truths[ui][0])
                        rem += int(items[slot] in truths[ui][0])
                        items[slot] = articles[ci]
                    cache[actual] = ([exact_ap(items, truth) for truth in truths[ui]],
                                     [ins, rem, bool(actual), len(actual)])
                per_metrics[pi], per_counts[pi] = cache[actual]
            sums[:, :5] += per_metrics
            sums[:, 5:] += per_metrics - base[ui]
            counts += per_counts
            next_user = ui + 1
            if next_user % 256 == 0:
                checkpoint()
    finally:
        checkpoint()
    elapsed = previous + time.perf_counter() - start
    delta = np.divide(sums[:, 5:], denominator, out=np.zeros((p, 5)), where=denominator != 0)
    means = baseline + delta
    rows = []
    for i, config in enumerate(configs):
        segments = {name: dict(map=float(means[i, j]) if denominator[j] else None,
                               delta=float(delta[i, j]) if denominator[j] else None,
                               map12=float(means[i, j]) if denominator[j] else None,
                               delta_vs_w0=float(delta[i, j]) if denominator[j] else None,
                               truth_users=int(denominator[j])) for j, name in enumerate(SEGMENTS) if j}
        rows.append(dict(policy_id=config['id'], policy=config, overall_map=float(means[i, 0]),
                         delta_map=float(delta[i, 0]), segments=segments,
                         inserted_positives=int(counts[i, 0]), removed_positives=int(counts[i, 1]),
                         admitted_users=int(counts[i, 2]), coverage=float(counts[i, 2] / n),
                         replacements=int(counts[i, 3]), users=n, seconds=elapsed / p,
                         shared_grid_seconds=elapsed, timing='shared E grid wall time equally allocated',
                         saved_decisions=str((folder / 'E-policy-user-matched-pairs.npy').resolve()),
                         decision_policy_index=i, status='completed', exact_final_list_AP=True))
    write_json(folder / 'EVALUATION.json', dict(status='completed', cutoff=data['cutoff'], rows=rows,
                                               seconds=elapsed, final_week='not_run'))
    return rows
