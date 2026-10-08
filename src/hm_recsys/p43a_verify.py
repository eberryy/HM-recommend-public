"""Read-only, independently reconstructed P4.3A completed-snapshot audit.

This module never imports the tournament/policy/oracle execution modules and
never trains a model. Its AP definition comes from the pre-existing metrics.apk.
The default bounded scope is the earliest A1 model with four completed windows;
explicit model_ids can widen a later audit without treating unseen work as pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta, date
import gc
import hashlib
import itertools
from pathlib import Path
import subprocess
import time
import traceback

import joblib
import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from .metrics import apk
from .p41a_contract import read_json, write_json
from .p42f_core import batches
from .p42e_resources import rss

WINDOWS = {'winter_20200122': '2020-01-22', 'spring_20200318': '2020-03-18',
           'early_summer_20200624': '2020-06-24', 'late_summer_20200819': '2020-08-19'}
SEGMENTS = ['overall', 'warm_21_plus', 'strict_cold', 'sparse1_5', 'all_cold_sparse']
FIXED_POLICY_IDS = ['p43a-policy-0000', 'p43a-policy-0319',
                    'p43a-policy-1635', 'p43a-policy-1919']
CORE_NAMES = {'p43a_run.py', 'p43a_data.py', 'p43a_policy.py',
              'p43a_contract.py', 'p43a_oracle.py'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


class AuditGuard:
    def __init__(self, deadline_epoch=None):
        self.deadline_epoch = deadline_epoch
        self.peak_rss_gib = 0.

    def __call__(self):
        if self.deadline_epoch is not None and time.time() >= self.deadline_epoch:
            raise TimeoutError('Audit deadline reached; partial evidence is not completion')
        current = rss() / 2**30
        self.peak_rss_gib = max(self.peak_rss_gib, current)
        if current >= 2.:
            raise MemoryError('Independent audit RSS reached the 2 GiB limit')


def close(actual, expected, message='', atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=message)


def guard_cutoff(cutoff):
    value = date.fromisoformat(cutoff)
    if value + timedelta(days=7) > date(2020, 9, 16):
        raise ValueError('Audit refuses final-week truth or overlap')
    return cutoff


def empirical_percentiles(scores):
    """Independent average-tie empirical ranks over ALL complete-window edges."""
    values = np.asarray(scores).ravel()
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite score')
    if not len(values):
        return np.empty(0, dtype=np.float64)
    if len(values) == 1:
        return np.array([.5])
    return (rankdata(values, method='average') - 1.) / (len(values) - 1.)


def fixed_policies():
    """Frozen policy order, recreated without importing production policies."""
    return [dict(id=f'p43a-policy-{i:04d}', top_edge=a, global_percentile=b,
                 candidate_topk=c, slot_floor=d, max_admissions=e)
            for i, (a, b, c, d, e) in enumerate(itertools.product(
                (1, 2, 3, 5, 10), (90, 95, 98, 99, 99.5, 99.9),
                (5, 10, 20, 50), (12, 10, 7, 1), (1, 2, 3, 12)))]


def bits(mask):
    return [i for i in range(10) if int(mask) & (1 << i)]


def brute_matching(edge_ids, allowed, cap):
    """Independent exhaustive feasible subsets; objective is harmonic edge rank."""
    edge_ids = np.asarray(edge_ids, dtype=np.int64)
    positions = [i for i in range(len(edge_ids)) if allowed[i]]
    best, best_mask = 0., 0
    for size in range(1, min(cap, len(positions)) + 1):
        for selected in itertools.combinations(positions, size):
            ee = edge_ids[list(selected)]
            if len(set(ee // 12)) != size or len(set(ee % 12)) != size:
                continue
            value = sum(1. / (i + 1) for i in selected)
            mask = sum(1 << i for i in selected)
            if value > best or (value == best and mask < best_mask):
                best, best_mask = value, mask
    return best, best_mask


def reconstruct(warm, articles, global_edges):
    result = list(warm)
    chosen = np.asarray(global_edges, dtype=np.int64)
    if len(chosen):
        if len(set(chosen // 12)) != len(chosen) or len(set(chosen % 12)) != len(chosen):
            raise AssertionError('Cold/slot collision in saved matching')
        for edge in chosen:
            result[int(edge % 12)] = articles[int(edge // 12)]
    if len(result) != 12 or len(set(result)) != 12:
        raise AssertionError('Final list is not exactly 12 unique articles')
    return result


def independent_context(data):
    """Truth denominators rebuilt from the frozen observation table, not receipts."""
    users, frame = data['users'], data['truth']
    counts = frame.interaction_count_before_cutoff
    dictionaries = []
    for mask in (np.ones(len(frame), bool), counts >= 21, counts == 0,
                 (counts >= 1) & (counts <= 5), counts <= 5):
        dictionaries.append({u: set(g.article_id) for u, g in
                             frame.loc[mask].groupby('customer_id', sort=False)})
    truths = [[m.get(u, set()) for m in dictionaries] for u in users]
    for u, tt in zip(users, truths):
        assert tt[0] == data['truthsets'].get(u, set())
    valid = np.asarray([[True] + [bool(t) for t in tt[1:]] for tt in truths])
    base = np.asarray([[apk(t, w, 12) for t in tt]
                       for w, tt in zip(data['warm_lists'], truths)], dtype=np.float64)
    return truths, valid, base


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def provenance(repo, root, contract, selected, guard):
    """Hashes answer explicit trusted-cache/registered-source integrity questions."""
    checked = {}
    def check(record, alternatives=()):
        original = Path(record['path'])
        for path in [original, *alternatives]:
            guard()
            if not path.exists() or path.stat().st_size != record['bytes']:
                continue
            key = str(path.resolve())
            if key not in checked:
                checked[key] = _digest(path)
            if checked[key] == record['sha256']:
                return dict(registered=str(original), verified_copy=key,
                            archived=path != original, sha256=record['sha256'])
        raise AssertionError(f'No trusted current/archived copy: {original}')

    start = read_json(root / 'EXECUTION_START.json')
    contract_check = check(start['contract'])
    assert contract['development_windows'] == WINDOWS
    assert contract['final_week'] == start['final_week'] == 'not_run'
    assert contract['sealed_confirmatory_holdout'] == '2020-09-16'
    assert contract['branch'] == 'main'
    branch = subprocess.run(['git', 'branch', '--show-current'], cwd=repo,
                            check=True, text=True, capture_output=True).stdout.strip()
    assert branch == 'main'
    for cutoff in WINDOWS.values():
        guard_cutoff(cutoff)
    authorities = [check(v) for v in contract['authority']]
    trusted = [check(v) for v in contract['trusted_F_prepared']]
    shared = [check(v) for v in start.get('shared_source', [])]
    receipts, source_checks = {}, []
    for config_id, window, entry in selected:
        fit = entry['fit']
        assert fit['status'] == 'completed' and fit['window'] == window
        assert fit['cutoff'] == WINDOWS[window]
        assert fit['config']['id'] == config_id
        for cutoff in fit['training_cutoffs']:
            guard_cutoff(cutoff)
            assert date.fromisoformat(cutoff) + timedelta(days=7) < date.fromisoformat(WINDOWS[window])
        path = fit['source_receipt']
        if path in receipts:
            continue
        source = read_json(Path(path))
        check(source['contract'])
        names = {Path(r['path']).name for r in source['source']}
        assert CORE_NAMES <= names
        rows = [check(v, (root / 'source-archive/initial-tournament' / Path(v['path']).name,))
                for v in source['source']]
        source_checks.append(dict(receipt=path, source=rows))
        receipts[path] = True
    return dict(status='pass', contract=contract_check, authority_files=len(authorities),
                trusted_F_prepared_files=len(trusted), shared_source_files=len(shared),
                fitted_source_receipts=source_checks, branch=branch,
                purpose='Explicit contract identities + cached source provenance; not routine completion hashing',
                final_week='not_run')


def population(data, cutoff):
    assert data['cutoff'] == cutoff
    users, cold = data['users'], data['cold']
    assert list(data['state'].customer_id) == list(users)
    assert len(set(users)) == len(users)
    assert len(data['warm_lists']) == len(users)
    assert not cold.duplicated(['customer_id', 'article_id']).any()
    assert np.all(np.diff(cold.user_index.to_numpy(int)) >= 0)
    groups = cold.groupby('user_index', sort=False).indices
    for ui, u in enumerate(users):
        w = data['warm_lists'][ui]
        assert len(w) == len(set(w)) == 12
        ii = groups.get(ui, [])
        if len(ii):
            cc = cold.iloc[ii]
            assert set(cc.customer_id) == {u}
            assert not set(cc.article_id) & set(w)
            assert cc.b0_rank.between(1, 50).all()
    warm = data['warm']
    if 'article_id' in warm.columns:
        assert warm.article_id.tolist() == [a for w in data['warm_lists'] for a in w]
    return dict(status='pass', users=len(users), cold_candidate_pairs=len(cold),
                complete_action_edges=len(cold)*12, action_users=len(groups),
                no_cold_users=len(users)-len(groups), source='contract-hash-verified F data.joblib',
                W0='Exact frozen F warm list with 12 unique items; P4.1A W0 independently matched')


def verify_oracle(repo, root, window, data, guard):
    cutoff = WINDOWS[window]
    folder = root / 'oracle' / cutoff
    result = read_json(folder / 'RESULT.json')
    assert result['status'] == 'completed' and result['final_week'] == 'not_run'
    assert result['cutoff'] == cutoff
    arrays = {key: np.load(value, mmap_mode='r') for key, value in result['artifacts'].items()}
    truths, valid, base = independent_context(data)
    n = len(data['users'])
    assert arrays['ap_by_k'].shape == (n, 13)
    assert arrays['matching_by_k'].shape == (n, 13, 12, 2)
    groups = data['cold'].groupby('user_index', sort=False).indices
    articles = data['cold'].article_id.to_numpy()
    ranks = data['cold'].b0_rank.to_numpy()
    best = np.zeros((n, 5))
    restricted = np.zeros(n)
    feasible_total = 0
    for ui in range(n):
        if ui % 128 == 0:
            guard()
        warm, truth = data['warm_lists'][ui], truths[ui][0]
        allowed = set(groups.get(ui, []))
        saved = arrays['ap_by_k'][ui]
        independent = np.full(13, np.nan)
        for k in range(13):
            pairs = arrays['matching_by_k'][ui, k]
            pairs = pairs[pairs[:, 0] >= 0]
            feasible = k <= min(len(allowed), 12)
            assert bool(np.isfinite(saved[k])) == feasible
            if not feasible:
                assert not len(pairs)
                continue
            assert len(pairs) == k
            assert set(pairs[:, 0]) <= allowed and np.all((pairs[:, 1] >= 0) & (pairs[:, 1] < 12))
            prediction = reconstruct(warm, articles, pairs[:, 0]*12+pairs[:, 1])
            independent[k] = apk(truth, prediction, 12)
            close(independent[k], saved[k], f'{window} user {ui} oracle K{k}', atol=1e-15)
            feasible_total += 1
        # Exact stored arithmetic determines ties; independent AP agrees to 1e-15.
        kstar = int(np.nanargmax(saved))
        assert kstar == int(arrays['k_star'][ui])
        close(independent[kstar], np.nanmax(independent), atol=1e-15)
        pairs = arrays['matching_by_k'][ui, kstar]
        pairs = pairs[pairs[:, 0] >= 0]
        prediction = reconstruct(warm, articles, pairs[:, 0]*12+pairs[:, 1])
        best[ui] = [apk(t, prediction, 12) for t in truths[ui]]
        close(best[ui], arrays['best_segments'][ui], atol=1e-15)
        pair = arrays['restricted_match'][ui]
        if pair[0] >= 0:
            assert pair[0] in allowed and ranks[pair[0]] <= 10 and pair[1] in (9, 10, 11)
            prediction = reconstruct(warm, articles, [pair[0]*12+pair[1]])
        else:
            prediction = list(warm)
        restricted[ui] = apk(truth, prediction, 12)
        # Independent direct list evaluation of every restricted eligible edge.
        expected = base[ui, 0]
        for ci in sorted(allowed):
            if ranks[ci] <= 10 and articles[ci] in truth:
                for slot in (9, 10, 11):
                    expected = max(expected, apk(truth, reconstruct(warm, articles, [ci*12+slot]), 12))
        close(restricted[ui], expected, atol=1e-15)
        close(restricted[ui], arrays['restricted_ap'][ui], atol=1e-15)
    den = valid.sum(axis=0)
    mean = np.divide(best.sum(axis=0), den, out=np.zeros(5), where=den != 0)
    baseline = np.divide(base.sum(axis=0), den, out=np.zeros(5), where=den != 0)
    close(mean[0], result['overall_map'])
    close(mean[0]-baseline[0], result['delta_map'])
    close(baseline[0], result['w0_map'])
    for j, name in enumerate(SEGMENTS[1:], 1):
        assert result['segments'][name]['truth_users'] == int(den[j])
        close(mean[j], result['segments'][name]['map'])
        close(mean[j]-baseline[j], result['segments'][name]['delta'])
    old = read_json(repo/'reports/phase4/P4_1A_metrics.json')['windows'][window]
    assert old['users'] == n
    close(restricted.mean(), old['map@12'])
    close((restricted-base[:, 0]).mean(), old['delta_MAP_vs_W0'])
    assert int((arrays['restricted_match'][:, 0] >= 0).sum()) == old['positive_opportunity_users']
    close(restricted.mean(), result['restricted']['map'])
    for k in range(13):
        assert int((arrays['k_star'] == k).sum()) == result['k_star_distribution'][str(k)]
        values = arrays['ap_by_k'][:, k]
        assert int(np.isfinite(values).sum()) == result['by_k'][str(k)]['feasible_users']
        if np.isfinite(values).any():
            close(float(np.nanmean(values)), result['by_k'][str(k)]['map_among_feasible'])
    return dict(status='pass', users=n, feasible_saved_K_lists=feasible_total,
                all_saved_lists_AP_recomputed=True, smallest_K_tie=True,
                all_lists_12_unique=True, P4_1A_replay='pass', overall_map=float(mean[0]),
                restricted_map=float(restricted.mean()),
                limit='Verifies all saved exactly-K proposals and best K, not a proof of global combinatorial AP optimality')


def verify_evaluation(root, config_id, window, entry, data, guard):
    from lightgbm import Booster
    receipt = read_json(Path(entry['receipt']))
    assert receipt['status'] == 'completed' and receipt['final_week'] == 'not_run'
    rows = receipt['rows']
    assert len(rows) == entry['evaluated_policies'] and rows
    frozen = {p['id']: p for p in fixed_policies()}
    for row in rows:
        assert row['status'] == 'completed' and row['policy'] == frozen[row['policy_id']]
    folder = Path(entry['receipt']).parent.parent
    prediction = read_json(folder/'PREDICTION.json')
    assert prediction['status'] == 'completed' and prediction['final_week'] == 'not_run'
    assert prediction['scoring_cutoff'] == WINDOWS[window]
    scores = np.load(prediction['path'], mmap_mode='r')
    assert scores.shape == (len(data['cold']), 12)
    percentiles = empirical_percentiles(scores).reshape(scores.shape)
    guard()
    decisions = np.load(rows[0]['saved_decisions'], mmap_mode='r')
    top_edges = np.load(rows[0]['saved_top_edges'], mmap_mode='r')
    n, m = len(data['users']), len(rows)
    assert decisions.shape == (m, n) and top_edges.shape == (n, 10)
    row_indices = np.array([r['decision_policy_index'] for r in rows], dtype=int)
    assert sorted(row_indices.tolist()) == list(range(m))
    truths, valid, base = independent_context(data)
    counts = np.zeros((m, 4), dtype=np.int64)
    sums = np.zeros((m, 5), dtype=np.float64)
    deltas = np.zeros((m, 5), dtype=np.float64)
    groups = data['cold'].groupby('user_index', sort=False).indices
    articles, ranks = data['cold'].article_id.to_numpy(), data['cold'].b0_rank.to_numpy()
    topk = np.array([r['policy']['candidate_topk'] for r in rows])
    floor = np.array([r['policy']['slot_floor'] for r in rows])
    levels = np.array([r['policy']['global_percentile']/100 for r in rows])
    topn = np.array([r['policy']['top_edge'] for r in rows])
    caps = np.array([r['policy']['max_admissions'] for r in rows])
    boundary_cache = {}
    for level in sorted(set(levels.tolist())):
        eligible = percentiles >= level
        boundary_cache[level] = float(np.min(scores[eligible])) if eligible.any() else None
    for row in rows:
        expected = boundary_cache[row['policy']['global_percentile']/100]
        assert expected == row['global_score_threshold']
    del eligible
    fixed_action_users = set(sorted(groups)[:8])
    fixed_pi = [i for i, r in enumerate(rows) if r['policy_id'] in FIXED_POLICY_IDS]
    graph_checks, unique_lists, unchanged = 0, 0, 0
    for ui in range(n):
        if ui % 128 == 0:
            guard()
        ii = np.asarray(groups.get(ui, []), dtype=np.int64)
        chosen = decisions[row_indices, ui]
        if not len(ii):
            assert not np.any(chosen) and np.all(top_edges[ui] == -1)
            sums += base[ui]
            unchanged += 1
            unique_lists += 1
            continue
        flat = scores[ii].ravel()
        # Python's tuple ordering provides a separate deterministic stable tie path.
        local = np.lexsort((np.arange(len(flat)), -flat))[:10]
        ee = ii[local//12]*12+local%12
        assert np.array_equal(top_edges[ui, :len(ee)], ee)
        assert np.all(top_edges[ui, len(ee):] == -1)
        raw_percentile = percentiles.ravel()[ee]
        allowed = ((np.arange(len(ee))[None, :] < topn[:, None]) &
                   (raw_percentile[None, :] >= levels[:, None]) &
                   (ranks[ee//12][None, :] <= topk[:, None]) &
                   (ee[None, :] % 12 + 1 >= floor[:, None]))
        chosen_flags = (chosen[:, None].astype(np.int32) & (1 << np.arange(len(ee)))) != 0
        assert not np.any(chosen_flags & ~allowed)
        assert np.all(chosen_flags.sum(axis=1) <= caps)
        unique, inverse = np.unique(chosen, return_inverse=True)
        local_ap, local_counts = [], []
        for mask in unique:
            pp = bits(mask)
            assert not pp or max(pp) < len(ee)
            pred = reconstruct(data['warm_lists'][ui], articles, ee[pp])
            assert set(ee[pp]//12) <= set(ii)
            local_ap.append([apk(t, pred, 12) for t in truths[ui]])
            inserted = sum(articles[e//12] in truths[ui][0] for e in ee[pp])
            removed = sum(data['warm_lists'][ui][e%12] in truths[ui][0] for e in ee[pp])
            local_counts.append([inserted, removed, int(bool(pp)), len(pp)])
        ap_values = np.asarray(local_ap)[inverse]
        sums += ap_values
        deltas += ap_values-base[ui]
        counts += np.asarray(local_counts, dtype=np.int64)[inverse]
        unique_lists += len(unique)
        if ui in fixed_action_users:
            for pi in fixed_pi:
                objective, _ = brute_matching(ee, allowed[pi], int(caps[pi]))
                observed = sum(1./(p+1) for p in bits(chosen[pi]))
                close(observed, objective, 'Independent rank-weight matching objective', atol=1e-12)
                graph_checks += 1
    denominator = valid.sum(axis=0)
    means = np.divide(sums, denominator, out=np.zeros_like(sums), where=denominator != 0)
    delta_means = np.divide(deltas, denominator, out=np.zeros_like(deltas), where=denominator != 0)
    for i, row in enumerate(rows):
        assert row['users'] == n
        close(means[i, 0], row['overall_map'], f'{window} {row["policy_id"]} MAP')
        close(delta_means[i, 0], row['delta_map'])
        for j, name in enumerate(SEGMENTS[1:], 1):
            sg = row['segments'][name]
            assert int(denominator[j]) == sg['truth_users']
            close(means[i, j], sg['map12'])
            close(delta_means[i, j], sg['delta_vs_w0'])
        for j, key in enumerate(('inserted_positives', 'removed_positives', 'admitted_users', 'replacements')):
            assert counts[i, j] == row[key], (row['policy_id'], key)
        close(counts[i, 2]/n, row['coverage'])
    booster = Booster(model_file=str(root/'models'/config_id/window/'model.txt'))
    _, cc, x, _ = next(batches(data, size=32, labels=False))
    if entry['fit']['config']['arm'] == 'D':
        meta_audit = read_json(root/'prepared'/WINDOWS[window]/'meta/AUDIT.json')
        assert meta_audit['cutoff'] == WINDOWS[window]
        for line in meta_audit['lineage'].values():
            assert not line['availability'] or line['training_label_end'] < WINDOWS[window]
        meta = np.load(meta_audit['paths']['X'], mmap_mode='r')
        x = np.concatenate((x, meta[:len(x)]), axis=1)
        assert booster.feature_name() == entry['fit']['features']
    assert booster.num_feature() == x.shape[1]
    replay = booster.predict(x, num_threads=1).reshape(-1, 12)
    close(replay, scores[:len(cc)], 'Saved model sample prediction', atol=0)
    return dict(status='pass', model_id=config_id, window=window, users=n,
                policy_rows=m, user_policy_observations=m*n, unique_saved_final_lists=unique_lists,
                AP_recomputed_with='pre-existing hm_recsys.metrics.apk',
                AP_sum_and_delta_tolerance=1e-12, denominators=dict(zip(SEGMENTS, map(int, denominator))),
                all_policy_rows_all_users_replayed=True, all_saved_decisions_feasible=True,
                all_scores_global_average_tie_gate_verified=True,
                no_cold_users_unchanged=unchanged, first_action_users=sorted(map(int, fixed_action_users)),
                fixed_policy_ids=[rows[i]['policy_id'] for i in fixed_pi],
                independent_exhaustive_matching_graphs=graph_checks,
                model_prediction_rows=len(replay)*12, exact_prediction_replay=True,
                AP_sums_minmax={name:[float(sums[:,j].min()),float(sums[:,j].max())]
                               for j,name in enumerate(SEGMENTS)})


def completed_snapshot(state, model_ids=None):
    candidates = []
    for config_id, record in state['configs'].items():
        if model_ids is not None and config_id not in model_ids:
            continue
        if record['config']['arm'] not in ('A', 'B', 'D'):
            continue
        rows = [(config_id, w, record['windows'][w]) for w in WINDOWS
                if w in record.get('windows', {}) and
                record['windows'][w].get('fit', {}).get('status') == 'completed' and
                record['windows'][w].get('receipt')]
        if model_ids is None:
            if not config_id.startswith('A1-') or len(rows) != 4:
                continue
        if rows:
            candidates.append((min(r[2]['fit']['finished_at'] for r in rows), config_id, rows))
    candidates.sort()
    if model_ids is None:
        candidates = candidates[:1]
    return [row for _, _, rows in candidates for row in rows]


def run_verify(repo, root, deadline_epoch=None, model_ids=None):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    guard = AuditGuard(deadline_epoch)
    attempt_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + str(time.time_ns())
    attempt = root/'verification-attempts'/attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    state = read_json(root/'TOURNAMENT_STATE.json')
    selected = completed_snapshot(state, model_ids=model_ids)
    result = dict(stage='P4.3A', status='running', attempt_id=attempt_id, at=stamp(),
                  audit_source=str(Path(__file__).resolve()), fit_count=0, policy_changes=0,
                  metric_overwrites=0, final_week='not_run', overall_stage_verified=False,
                  snapshot_state_status=state['status'], scope_model_windows=[{'model_id':c,'window':w} for c,w,_ in selected],
                  provenance=None, population={}, oracle={}, evaluation=[],
                  not_run=['Other model-window receipts outside this completed snapshot',
                           'All-six-arm tournament completion and top2/full-scale gates',
                           'C/D/E/F family-specific score reconstruction',
                           'Training refit; final-week data; full-scale experiments'],
                  complete_stage_pass=False)
    write_json(attempt/'START.json', result)
    try:
        if not selected:
            result['status'] = 'not_run_no_completed_scope'
            return result
        contract = read_json(repo/'reports/phase4/P4_3A_EXPERIMENT_CONTRACT.json')
        assert state['final_week'] == 'not_run'
        result['provenance'] = provenance(repo, root, contract, selected, guard)
        froot = repo/'artifacts/phase4/p4-2f-v1-relative-utility-hash10-user-oof'
        with threadpool_limits(limits=1):
            for window, cutoff in WINDOWS.items():
                guard()
                current = [r for r in selected if r[1] == window]
                if not current:
                    continue
                data = joblib.load(froot/'prepared'/cutoff/'data.joblib')
                result['population'][window] = population(data, cutoff)
                result['oracle'][window] = verify_oracle(repo, root, window, data, guard)
                for config_id, _, entry in current:
                    result['evaluation'].append(verify_evaluation(root, config_id, window, entry, data, guard))
                    print(f'P4.3A independent audit passed {config_id} {window}', flush=True)
                    write_json(attempt/'PROGRESS.json', result)
                del data
                gc.collect()
        result['status'] = 'pass_completed_snapshot'
        result['scope_policy_rows'] = sum(r['policy_rows'] for r in result['evaluation'])
        result['scope_user_policy_observations'] = sum(r['user_policy_observations'] for r in result['evaluation'])
        result['scope_note'] = 'Only enumerated completed snapshot passed; this is NOT a full tournament success or promotion'
        return result
    except (TimeoutError, MemoryError) as exc:
        result.update(status='partial_pass' if result['evaluation'] else 'paused_before_completed_evaluation',
                      pause_reason=str(exc), incomplete_requested_scope=True)
        return result
    except Exception:
        result.update(status='fail', error=traceback.format_exc())
        raise
    finally:
        result.update(finished_at=stamp(), seconds=time.perf_counter()-start, peak_rss_gib=guard.peak_rss_gib)
        write_json(attempt/'RESULT.json', result)
        # Every failed/partial attempt remains immutable; this is only the latest snapshot pointer.
        write_json(repo/'reports/phase4/P4_3A_VERIFICATION.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path('.'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--model', action='append')
    args = parser.parse_args()
    root = args.root or args.repo/'artifacts/phase4/p4-3a-v1-map-cashout-hash10-tournament'
    outcome = run_verify(args.repo, root, args.deadline, args.model)
    print(outcome['status'], flush=True)
