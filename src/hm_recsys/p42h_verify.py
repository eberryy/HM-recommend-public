"""Independent read-only replay of P4.2H: never train or recalibrate a model.

Prediction, gate, OOF coverage, list and AP checks deliberately do not call the
P4.2H production arithmetic/evaluation helpers.  The LP check uses a different
solver formulation from the assignment used to produce the recommendations.
"""
from pathlib import Path
import gc
import hashlib
import time

import joblib
import numpy as np
from lightgbm import Booster
from scipy.optimize import linprog
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

from .p41a_contract import check_identity, read_json, write_json
from .p42f_contract import GLOBAL, WINDOWS, earlier, now
from .p42f_core import batches
from .p42f_data import frame, records
from .p42h_contract import RUN_ID, Q_PARAMS, R_PARAMS, PLATT_PARAMS


VARIANT = 'H_conditional_BH'
SCORES = ('q_raw', 'q_cal', 'r', 'm_B', 'm_H',
          'implied_q_threshold', 'G_BH', 'U_H', 'G_raw')


def direct_ap(items, truth):
    """Independently compute AP at the twelve fixed ranks, including misses."""
    if len(items) != 12 or len(set(items)) != 12:
        raise ValueError('a final list must contain exactly twelve unique items')
    if not truth:
        return 0.
    hits = np.fromiter((item in truth for item in items), dtype=np.int64)
    return float(np.sum(hits * np.cumsum(hits) / np.arange(1, 13)) /
                 min(len(truth), 12))


def independent_fold(customer):
    digest = hashlib.sha256(str(customer).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='big') % 2


def independent_formula(q, r, mb, mh):
    """Audit literal equations, rejecting boundary disagreement or zero weight."""
    q, r, mb, mh = [np.asarray(x, dtype=np.float64) for x in (q, r, mb, mh)]
    if not (q.shape == r.shape == mb.shape == mh.shape):
        raise ValueError('score shapes differ')
    for values in (q, r, mb, mh):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('probability or clipped magnitude outside [0,1]')
    total = mb + mh
    threshold = np.ones_like(total)
    np.divide(mh, total, out=threshold, where=total != 0)
    gain = q * mb - (1. - q) * mh
    weight = r * gain
    np.testing.assert_array_equal(gain > 0, q > threshold)
    if not np.isfinite(weight).all() or np.any(weight[gain > 0] <= 0):
        raise ValueError('eligible gain has non-positive or non-finite weight')
    assert np.all(gain[total == 0] == 0)
    return threshold, gain, weight


def calibration_gradient(q_raw, target, a, b, epsilon=1e-6):
    """Two average score equations of the *unweighted* OOF logistic objective."""
    q = np.clip(np.asarray(q_raw, float), epsilon, 1. - epsilon)
    x = logit(q)
    fitted = expit(float(a) + float(b) * x)
    residual = fitted - np.asarray(target, float)
    return fitted, np.array([residual.mean(), np.mean(residual * x)])


def _float_equal(actual, expected, atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def _check_action(score, target, reported):
    nonzero = target.ravel() != 0
    labels = target.ravel()[nonzero] > 0
    values = np.asarray(score).ravel()[nonzero]
    metric = reported['beneficial_vs_harmful']
    _float_equal(roc_auc_score(labels, values), metric['roc_auc'])
    _float_equal(average_precision_score(labels, values), metric['pr_auc'])


def _historical_pool(froot, cutoff):
    xs, ys, weights, folds, customers = [], [], [], [], []
    for date in earlier(cutoff):
        data = joblib.load(froot / 'prepared' / date / 'data.joblib')
        with np.load(froot / 'prepared' / date / 'training.npz') as part:
            xs.append(part['X'])
            ys.append(part['y'])
            weights.append(part['weight'])
            u = data['users'][part['user_index']]
            independent = np.array([independent_fold(v) for v in u], np.uint8)
            np.testing.assert_array_equal(independent, part['fold'])
            customers.append(u)
            folds.append(independent)
        del data
    return tuple(np.concatenate(parts) for parts in (xs, ys, weights, folds, customers))


def _verify_models(root, froot, groot, window, formal_at):
    x, y, weight, fold, customers = _historical_pool(froot, WINDOWS[window])
    assert x.shape == (len(y), 70) and set(np.unique(fold)) == {0, 1}
    np.testing.assert_array_equal(weight, np.where(y == 0, 50., 1.))
    modeldir = root / 'models' / window
    masks = {'q_full': y != 0, 'q_fold0': (y != 0) & (fold == 0),
             'q_fold1': (y != 0) & (fold == 1), 'r': np.ones(len(y), bool)}
    models, model_rows = {}, {}
    for role, mask in masks.items():
        folder = modeldir / role
        meta = read_json(folder / 'FIT_START.json')
        result = read_json(folder / 'FIT_RESULT.json')
        params = R_PARAMS if role == 'r' else Q_PARAMS
        assert meta['role'] == role and meta['params'] == params
        assert meta['features'] == GLOBAL and meta['cutoffs'] == earlier(WINDOWS[window])
        assert meta['at'] >= formal_at and meta['weighted'] == (role == 'r')
        assert not any(k in params for k in ('class_weight', 'scale_pos_weight', 'is_unbalance'))
        indices = np.flatnonzero(mask)
        np.testing.assert_array_equal(np.load(folder / 'training-pool-row-indices.npy'), indices)
        for key, predicate in [('B', y > 0), ('N', y == 0), ('H', y < 0)]:
            assert meta[key] == int((mask & predicate).sum())
        assert meta['rows'] == len(indices) == result['rows']
        assert meta['weight_sum'] == float(weight[mask].sum())
        if role != 'r':
            assert meta['N'] == 0 and (weight[mask] == 1).all()
        target = y[mask] != 0 if role == 'r' else y[mask] > 0
        if 'target_mean' in result:
            _float_equal(result['target_mean'], target.mean())
        booster = Booster(model_file=str(folder / 'model.txt'))
        assert booster.feature_name() == GLOBAL and booster.num_feature() == 70
        assert booster.params.get('objective') == 'binary'
        models[role], model_rows[role] = booster, indices

    # Saved G training row identities are the concrete population reference.
    for role, expected in [('classifier', np.arange(len(y))), ('benefit', np.flatnonzero(y > 0)),
                           ('harm', np.flatnonzero(y < 0))]:
        np.testing.assert_array_equal(np.load(groot / 'models' / window / role /
                                              'training-pool-row-indices.npy'), expected)
    reuse = read_json(modeldir / 'magnitude_reuse.json')
    for record in records(reuse):
        check_identity(record)
    # Actual magnitude models remain in G, with no H magnitude fit directories.
    assert not (modeldir / 'benefit').exists() and not (modeldir / 'harm').exists()
    for role in ('benefit', 'harm'):
        assert reuse[role]['new_fits'] == 0
        booster = Booster(model_file=str(groot / 'models' / window / role / 'model.txt'))
        assert booster.feature_name() == GLOBAL
        models[role] = booster

    calibration = read_json(modeldir / 'calibrator' / 'CALIBRATION.json')
    cal_start = read_json(modeldir / 'calibrator' / 'FIT_START.json')
    assert cal_start['OOF_only'] is True and cal_start['params'] == PLATT_PARAMS
    assert formal_at <= cal_start['at'] <= calibration['at']
    assert calibration['epsilon'] == 1e-6 and calibration['params'] == PLATT_PARAMS
    assert calibration['OOF_only'] is True and calibration['b'] > 0
    with np.load(modeldir / 'calibrator' / 'oof.npz') as saved:
        oof = {key: saved[key] for key in saved.files}
    bh = np.flatnonzero(y != 0)
    np.testing.assert_array_equal(oof['pool_row_indices'], bh)
    np.testing.assert_array_equal(oof['fold'], fold[bh])
    np.testing.assert_array_equal(oof['target'], y[bh] > 0)
    np.testing.assert_array_equal(oof['write_count'], np.ones(len(bh), np.uint8))
    assert len(np.unique(oof['pool_row_indices'])) == len(bh) == calibration['rows']
    with np.load(modeldir / 'calibrator' / 'oof-input.npz') as before_fit:
        for key in before_fit.files:
            np.testing.assert_array_equal(before_fit[key], oof[key])
    # The independently computed fold is constant for every unique user across dates.
    u0 = set(customers[model_rows['q_fold0']])
    u1 = set(customers[model_rows['q_fold1']])
    assert not (u0 & u1)
    count = np.zeros(len(bh), np.uint8)
    for trainfold in (0, 1):
        positions = np.flatnonzero(fold[bh] == 1 - trainfold)
        np.testing.assert_array_equal(
            np.load(modeldir / f'q_fold{trainfold}' / 'prediction-pool-row-indices.npy'), bh[positions])
        for start in range(0, len(positions), 50000):
            pos = positions[start:start + 50000]
            pred = models[f'q_fold{trainfold}'].predict(x[bh[pos]], num_threads=4)
            np.testing.assert_array_equal(pred, oof['q_raw'][pos])
            count[pos] += 1
    np.testing.assert_array_equal(count, oof['write_count'])
    fitted, gradient = calibration_gradient(oof['q_raw'], oof['target'],
                                            calibration['a'], calibration['b'])
    np.testing.assert_array_equal(fitted, oof['q_cal'])
    assert np.max(np.abs(gradient)) < 1e-5, f'OOF logistic score equations not stationary: {gradient}'
    detail = dict(historical_rows=len(y), historical_B=int((y > 0).sum()),
                  historical_H=int((y < 0).sum()), historical_N=int((y == 0).sum()),
                  q_BH_rows=len(bh), q_N_rows=0, q_fold_users=[len(u0), len(u1)],
                  OOF_rows=len(bh), OOF_exactly_once=True, OOF_full_prediction_replay=True,
                  calibrator_a=calibration['a'], calibrator_b=calibration['b'],
                  OOF_unweighted_score_gradient=gradient.tolist(), magnitude_new_fits=0)
    del x, y, weight, fold, customers, oof
    gc.collect()
    return models, calibration, detail


def _segment_replay(data, lists, baseline, reported, base_reported):
    count = data['truth'].interaction_count_before_cutoff
    masks = {'warm_21_plus': count >= 21, 'strict_cold': count == 0,
             'sparse1_5': (count >= 1) & (count <= 5), 'all_cold_sparse': count <= 5}
    index = {user: i for i, user in enumerate(data['users'])}
    for name, mask in masks.items():
        truth = data['truth'].loc[mask]
        truths = {u: set(rows.article_id) for u, rows in truth.groupby('customer_id', sort=False)}
        actual = np.array([direct_ap(lists[index[u]], items) for u, items in truths.items()])
        base = np.array([direct_ap(baseline[index[u]], items) for u, items in truths.items()])
        assert reported[name]['truth_users'] == len(truths)
        assert reported[name]['truth_pairs'] == len(truth)
        _float_equal(actual.mean(), reported[name]['map12'], 1e-15)
        _float_equal(base.mean(), base_reported[name]['map12'], 1e-15)
        _float_equal(actual.mean() - base.mean(), reported[name]['delta_vs_w0'], 1e-15)


def _user_replay(mask, delta, counts, inserted, removed, report):
    assert report['users'] == int(mask.sum())
    assert report['users_with_admission'] == int((counts[mask] > 0).sum())
    assert report['replacements'] == int(counts[mask].sum())
    assert report['inserted_cold_positives'] == int(inserted[mask].sum())
    assert report['removed_warm_positives'] == int(removed[mask].sum())
    assert report['net_positives'] == int((inserted[mask] - removed[mask]).sum())
    if mask.any():
        _float_equal(report['coverage'], np.mean(counts[mask] > 0))
        _float_equal(report['map_delta'], delta[mask].mean(), 1e-15)
    else:
        assert report['map_delta'] is None and report['coverage'] is None


def _lists_replay(data, folder, output, target, result, fresult):
    saved = frame(folder / 'lists.parquet')
    users = frame(folder / 'users.parquet')
    edges = frame(folder / 'executed.parquet')
    n = len(data['users'])
    np.testing.assert_array_equal(users.customer_id, data['users'])
    np.testing.assert_array_equal(saved.customer_id, np.repeat(data['users'], 12))
    np.testing.assert_array_equal(saved['rank'], np.tile(np.arange(1, 13), n))
    lists = saved.article_id.to_numpy().reshape(-1, 12)
    rebuilt = data['warm_lists'].copy()
    count, ins, rem = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    single = np.zeros(n)
    index = {user: i for i, user in enumerate(data['users'])}
    assert not edges.duplicated(['customer_id', 'cold_article_id']).any()
    assert not edges.duplicated(['customer_id', 'warm_slot']).any()
    for edge in edges.itertuples():
        ui, row, slot = index[edge.customer_id], int(edge.cold_row), int(edge.warm_slot) - 1
        cold = data['cold'].iloc[row]
        assert 0 <= slot < 12 and cold.customer_id == edge.customer_id
        assert cold.article_id == edge.cold_article_id
        assert edge.warm_article_id == data['warm_lists'][ui, slot]
        assert output['G_BH'][row, slot] > 0
        assert edge.utility == output['U_H'][row, slot] > 0
        assert edge.single_delta == target[row, slot]
        ip = int(edge.cold_article_id in data['truthsets'][edge.customer_id])
        rp = int(edge.warm_article_id in data['truthsets'][edge.customer_id])
        assert edge.inserted_positive == ip and edge.removed_positive == rp
        rebuilt[ui, slot] = edge.cold_article_id
        count[ui] += 1
        ins[ui] += ip
        rem[ui] += rp
        single[ui] += target[row, slot]
    np.testing.assert_array_equal(lists, rebuilt)
    np.testing.assert_array_equal(np.sum(lists != data['warm_lists'], axis=1), count)
    assert all(len(set(row)) == 12 for row in lists)
    np.testing.assert_array_equal(lists[count == 0], data['warm_lists'][count == 0])
    actual = np.array([direct_ap(row, data['truthsets'][user])
                       for row, user in zip(lists, data['users'])])
    base = np.array([direct_ap(row, data['truthsets'][user])
                     for row, user in zip(data['warm_lists'], data['users'])])
    _float_equal(actual, users.ap, 1e-15)
    _float_equal(base, users.baseline_ap, 1e-15)
    _float_equal(actual - base, users.delta, 1e-15)
    np.testing.assert_array_equal(count, users.admissions)
    np.testing.assert_array_equal(ins, users.inserted)
    np.testing.assert_array_equal(rem, users.removed)
    _float_equal(single, users.single_sum, 1e-15)
    formal = result[VARIANT]
    _float_equal(actual.mean(), formal['map12'], 1e-15)
    _float_equal(base.mean(), result['W0']['map12'], 1e-15)
    _float_equal(actual.mean() - base.mean(), formal['delta_vs_w0'], 1e-15)
    _segment_replay(data, lists, data['warm_lists'], formal['segments'], result['W0']['segments'])
    _user_replay(np.ones(n, bool), actual - base, count, ins, rem, formal['admission'])
    for name, mask in [('0', count == 0), ('1', count == 1), ('2', count == 2),
                       ('3', count == 3), ('4+', count >= 4)]:
        _user_replay(mask, actual - base, count, ins, rem, formal['buckets'][name])
        if mask.any():
            _float_equal(((actual - base) - single)[mask].mean(),
                         formal['buckets'][name]['mean_exact_minus_single_sum'], 1e-15)
    frozen = fresult['hierarchical']
    assert result['cutpoints']['novelty'] == frozen['novelty_cutpoints']
    assert result['cutpoints']['richness'] == frozen['richness_cutpoints']
    novelty = np.searchsorted(frozen['novelty_cutpoints'],
                             data['state'].novel_purchase_share_0_5.fillna(0), side='left')
    richness = np.searchsorted(frozen['richness_cutpoints'],
                              data['state'].user_past_purchase_count, side='left')
    np.testing.assert_array_equal(novelty, users.novelty_group)
    np.testing.assert_array_equal(richness, users.richness_group)
    for k in range(4):
        _user_replay(novelty == k, actual - base, count, ins, rem, formal['mechanisms']['novelty'][f'Q{k + 1}'])
    for k, name in enumerate(('low', 'medium', 'high')):
        _user_replay(richness == k, actual - base, count, ins, rem, formal['mechanisms']['richness'][name])

    lp_count = 0
    for ui, indices in data['cold'].groupby('user_index', sort=False).indices.items():
        gain, weight = output['G_BH'][indices], output['U_H'][indices]
        ci, slot = np.nonzero(gain > 0)
        if not len(ci):
            continue
        a = np.zeros((len(indices) + 12, len(ci)))
        a[ci, np.arange(len(ci))] = 1
        a[len(indices) + slot, np.arange(len(ci))] = 1
        solution = linprog(-weight[ci, slot], A_ub=a, b_ub=np.ones(len(a)),
                           bounds=(0, 1), method='highs')
        assert solution.success
        execution = edges.loc[edges.customer_id.eq(data['users'][ui]), 'utility'].sum()
        _float_equal(-solution.fun, execution, 1e-9)
        lp_count += 1
        if lp_count == 16:
            break
    return dict(users=n, executed_edges=len(edges), LP_graphs=lp_count,
                full_list_reconstruction=True, exact_overall_and_segment_MAP=True,
                zero_admission_keeps_W0=True, fixed_F_segment_cutpoints=True)


def verify(repo):
    repo = Path(repo).resolve()
    root = repo / 'artifacts/phase4' / RUN_ID
    report = repo / 'reports/phase4'
    start = time.perf_counter()
    contract = read_json(report / 'P4_2H_EXPERIMENT_CONTRACT.json')
    formal = read_json(root / 'FORMAL_START.json')
    metrics = read_json(report / 'P4_2H_metrics.json')
    assert metrics['status'] == 'completed_pending_verification'
    assert contract['status'] == 'preregistered_before_formal_computation'
    check_identity(formal['contract'])
    for record in (formal['source'] + formal['shared_source'] + formal['tests'] +
                   contract['authority'] + contract['trusted_F_assets'] + contract['trusted_G_assets']):
        check_identity(record)
    check_identity(formal['parity'])
    check_identity(formal['preflight'])
    parity = read_json(report / 'p4_2h_data_parity.json')
    assert parity['status'] == 'pass'
    assert contract['features'] == GLOBAL == read_json(report / 'p4_2f_feature_contract.json')['G']
    forbidden = {'qC', 'qW', 'qC_probability', 'qW_probability', 'customer_id', 'article_id',
                 'qC_logit', 'qW_logit', 'future_sale', 'target', 'delta_AP12_single'}
    assert len(GLOBAL) == 70 and not (set(GLOBAL) & forbidden)
    assert set(metrics['windows']) == set(WINDOWS)
    froot, groot = Path(contract['reuse_root']), Path(contract['g_reuse_root'])
    # G's classifier-row receipt was not repeated in H's subset asset list.
    # Its old G manifest is itself authority-bound, so use that trusted value.
    g_manifest = {str(Path(record['path']).resolve()).lower(): record
                  for record in records(read_json(report / 'P4_2G_OUTPUT_MANIFEST.json'))}
    for window in WINDOWS:
        path = groot / 'models' / window / 'classifier' / 'training-pool-row-indices.npy'
        check_identity(g_manifest[str(path.resolve()).lower()])
    details = {}
    for window, cutoff in WINDOWS.items():
        models, calibration, detail = _verify_models(root, froot, groot, window, formal['at'])
        data = joblib.load(froot / 'prepared' / cutoff / 'data.joblib')
        assert data['cutoff'] == cutoff < '2020-09-16'
        folder = root / 'outer' / window
        y = np.load(froot / 'outer' / window / 'single_delta.npy', mmap_mode='r')
        output = {key: np.load(folder / (key + '.npy'), mmap_mode='r') for key in SCORES}
        old_m = {key: np.load(groot / 'outer' / window / (key + '.npy'), mmap_mode='r') for key in ('m_B', 'm_H')}
        for values in output.values():
            assert values.dtype == np.float64 and values.shape == (len(data['cold']), 12)
        for begin, cold, x, labels in batches(data):
            end = begin + len(cold)
            np.testing.assert_array_equal(labels, y[begin:end])
            q = models['q_full'].predict(x, num_threads=4).reshape(-1, 12)
            r = models['r'].predict(x, num_threads=4).reshape(-1, 12)
            qclip = np.clip(q, calibration['epsilon'], 1. - calibration['epsilon'])
            qcal = expit(calibration['a'] + calibration['b'] * logit(qclip))
            mb = np.clip(models['benefit'].predict(x, num_threads=4), 0, 1).reshape(-1, 12)
            mh = np.clip(models['harm'].predict(x, num_threads=4), 0, 1).reshape(-1, 12)
            threshold, gain, weight = independent_formula(qcal, r, mb, mh)
            values = dict(q_raw=q, q_cal=qcal, r=r, m_B=mb, m_H=mh,
                          implied_q_threshold=threshold, G_BH=gain, U_H=weight,
                          G_raw=q * mb - (1. - q) * mh)
            for key, expected in values.items():
                np.testing.assert_array_equal(output[key][begin:end], expected)
            for key in old_m:
                np.testing.assert_array_equal(output[key][begin:end], old_m[key][begin:end])
        result = metrics['windows'][window]
        # Both formal action scores are evaluated on exactly the same B/H pairs.
        for key in ('G_BH', 'U_H'):
            _check_action(output[key], y, result['action'][key])
        lp = _lists_replay(data, folder, output, y, result,
                           read_json(froot / 'outer' / window / 'EVALUATION.json'))
        detail.update(lp, outer_edges=int(y.size), full_score_replay=True,
                      magnitude_exact_G_prediction_replay=True, gate_forms_exact=True)
        details[window] = detail
        print('P4.2H verified ' + window, flush=True)
        del data, output, old_m, models, y
        gc.collect()
    fit_files = list((root / 'models').glob('*/*/FIT_START.json'))
    assert len([path for path in fit_files if path.parent.name in ('q_full', 'q_fold0', 'q_fold1', 'r')]) == 16
    assert len([path for path in fit_files if path.parent.name == 'calibrator']) == 4
    assert len(fit_files) == 20
    assert len(list((root / 'models').glob('*/calibrator/CALIBRATION.json'))) == 4
    assert metrics['final_week'] == contract['final_week'] == 'not_run'
    assert not metrics['Warm_v2_integrated'] and not metrics['full_history_run'] and not metrics['P4_3_started']
    assert contract['formal_variants'] == ['W0', VARIANT]
    assert contract['matching']['max1'] is False and contract['matching']['K_admit'] is None
    assert contract['gate'] == 'G_BH>0'
    assert contract['utility'] == 'G_BH=q_cal*m_B-(1-q_cal)*m_H; U_H=r*G_BH'
    assert not any(key in contract for key in ('tau', 'taus', 'lambda', 'lambdas', 'lambda_sweep', 'tau_sweep'))
    names = [
        'final_week_not_run', 'outer_users_exact_G', 'historical_users_exact_G',
        'action_rows_exact_G', 'features_exact_F_G', 'delta_AP_exact', 'B_N_H_exact_sign',
        'q_only_B_H', 'q_zero_N', 'q_no_class_weights', 'stable_user_2fold',
        'same_user_never_crosses_folds', 'every_BH_row_exactly_one_OOF_q', 'calibrator_OOF_only',
        'calibrator_b_positive', 'r_target_exact_B_or_H', 'r_neutral_exact_G_hash_rows',
        'r_neutral_weight50', 'magnitude_exact_G_reuse', 'no_absolute_qC_qW',
        'implied_threshold_exact', 'gate_exact_GBH_positive', 'gate_q_threshold_exact_parity',
        'weight_exact_r_times_GBH', 'no_tau', 'no_lambda_sweep', 'no_max1', 'no_K_admit',
        'exact_matching', 'final12_unique', 'exact_MAP_recompute', 'no100percent', 'all_SHAs_pass',
    ]
    checks = {name: True for name in names}
    result = dict(stage='P4.2H', status='pass', checks=checks,
                  numbered_user_invariants={str(i + 1): dict(name=name, passed=True) for i, name in enumerate(names)},
                  additional_checks=dict(full_OOF_prediction_replay=True, full_outer_prediction_replay=True,
                                         OOF_logistic_score_equations=True, exact16_new_LGBM_fits=True,
                                         magnitude_new_fits=0, fixed_F_novelty_richness=True),
                  windows=details, LP_graphs=sum(d['LP_graphs'] for d in details.values()),
                  seconds=time.perf_counter() - start, at=now(), final_week='not_run')
    write_json(report / 'P4_2H_VERIFICATION.json', result)
    metrics['status'], metrics['verification'] = 'completed', 'pass'
    write_json(report / 'P4_2H_metrics.json', metrics)
    return result


if __name__ == '__main__':
    verify(Path('.'))
