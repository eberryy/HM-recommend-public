"""Frozen P4.3A C: historical-oracle count/challenger/removal imitation.

Feature construction never reads oracle or future-label fields. Three unit-
weighted heads are followed by the fixed26 rank blends and exactly predicted K
matching, including zero-score ties. Nothing executes merely on import.
"""
from pathlib import Path
import gc
import time
import traceback
import warnings

import joblib
import numpy as np

from .p41a_contract import read_json, write_json, identity
from .p42f_contract import COLD, WARM, USER, PAIR, GLOBAL, WINDOWS, earlier, now
from .p42f_core import batches
from .p43a_data import META_COLUMNS, ensure_meta
from .p43a_oracle import _date_oracle, exactly_k_matching
from .p43a_policy import contexts, exact_ap, SEGMENTS, check_deadline


FAMILY = 'C-oracle'


def validate_lineage(audit, cutoff):
    if cutoff >= '2020-09-16' or audit['cutoff'] != cutoff:
        raise ValueError('C refuses sealed or mismatched feature cutoff')
    if audit['columns'] != list(META_COLUMNS):
        raise ValueError('C meta schema differs from frozen14 columns')
    for source, line in audit['lineage'].items():
        if line['availability'] and not line['training_label_end'] < cutoff:
            raise ValueError(f'C unsafe historical meta lineage: {source}')


def _finite(values, operation):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(getattr(np, operation)(values)) if len(values) else np.nan


def _topmean(values, k):
    values = np.asarray(values, float)
    values = np.where(np.isfinite(values), values, 0.)
    return float(np.sort(values)[-k:].mean()) if len(values) else 0.


def build_features(data, meta, schema, deadline_epoch=None):
    """Fixed37/51/46 PIT columns in exact contract order; no target access."""
    n, nc = len(data['users']), len(data['cold'])
    meta = np.asarray(meta)
    if meta.shape != (nc * 12, len(META_COLUMNS)) or not np.isfinite(meta).all():
        raise ValueError('C meta matrix shape/nonfinite mismatch')
    for j in range(0, len(META_COLUMNS), 2):
        if not np.isin(meta[:, j + 1], [0, 1]).all() or np.any((meta[:, j + 1] == 0) & (meta[:, j] != 0)):
            raise ValueError('C missing meta must have percentile0 and availability0')
    cc, state, warm = data['cold'], data['state'], data['warm']
    ui = cc.user_index.to_numpy(int)
    np.testing.assert_array_equal(state.customer_id, data['users'])
    np.testing.assert_array_equal(warm.customer_id.to_numpy(), np.repeat(data['users'], 12))
    user = state[USER].to_numpy(float)
    max_pair = np.empty((nc, len(PAIR)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for start, block, x, _ in batches(data, labels=False):
            check_deadline(deadline_epoch)
            values = x[:, [GLOBAL.index(name) for name in PAIR]].reshape(-1, 12, len(PAIR))
            max_pair[start:start + len(block)] = np.nanmax(values, axis=1)
    meta3 = meta.reshape(nc, 12, len(META_COLUMNS))
    cold_names = COLD + USER + ['max_pair_' + name for name in PAIR] + ['max_meta_' + name for name in META_COLUMNS]
    cold_x = np.concatenate((cc[COLD].to_numpy(float), user[ui], max_pair, meta3.max(axis=1)), axis=1)
    warm_names = WARM + USER
    warm_x = np.concatenate((warm[WARM].to_numpy(float), np.repeat(user, 12, axis=0)), axis=1)
    count_names = USER + [f'{source}_top{k}_mean' for source in ('B0', 'qC', 'old_U', 'H_q_raw') for k in (1, 3, 5)]
    count_names += ['qW_vulnerability_top1', 'qW_vulnerability_top2', 'warm_source_count_mean',
                    'warm_source_count_min', 'warm_source_count_max', 'warm_zscore_min', 'warm_zscore_max', 'warm_zscore_std']
    count_x = np.zeros((n, len(count_names)))
    count_x[:, :len(USER)] = user
    groups = cc.groupby('user_index', sort=False).indices
    for index in range(n):
        if index % 256 == 0:
            check_deadline(deadline_epoch)
        rows = np.asarray(groups.get(index, []), int)
        cold = cc.iloc[rows]
        sources = [cold.b0_user_percentile.to_numpy(float),
                   np.where(cold.qC_relative_available.to_numpy() == 1, cold.qC_within_user_percentile.to_numpy(float), 0.)]
        for name in ('old_U', 'H_q_raw'):
            col = META_COLUMNS.index(name + '_percentile')
            values = np.where(meta3[rows, :, col + 1] == 1, meta3[rows, :, col], 0.)
            sources.append(values.max(axis=1) if len(rows) else np.empty(0))
        aggregate = [_topmean(values, k) for values in sources for k in (1, 3, 5)]
        ww = warm.iloc[index*12:(index+1)*12]
        q = ww.qW_within_user_percentile.to_numpy(float)
        available = (ww.qW_relative_available.to_numpy() == 1) & np.isfinite(q)
        vulnerability = np.sort(np.where(available, 1. - q, 0.))[::-1]
        aggregate += [float(vulnerability[0]), float(vulnerability[1])]
        aggregate += [_finite(ww.source_count.to_numpy(), op) for op in ('mean', 'min', 'max')]
        aggregate += [_finite(ww.warm_user_zscore.to_numpy(), op) for op in ('min', 'max', 'std')]
        count_x[index, len(USER):] = aggregate
    names = dict(count=count_names, cold=cold_names, warm=warm_names)
    for role in names:
        if names[role] != schema[role]:
            raise ValueError('C exact feature order mismatch for ' + role)
    result = dict(count=count_x, cold=cold_x, warm=warm_x)
    for role, values in result.items():
        values[~np.isfinite(values)] = np.nan
        result[role] = values.astype(np.float32)
    result['action_users'] = np.array(sorted(groups), np.int64)
    return result


def oracle_targets(data, k_star, matching_by_k):
    """Count clips K at3; challenger/removal targets retain the full oracle K*."""
    n, nc = len(data['users']), len(data['cold'])
    if np.shape(k_star) != (n,) or np.shape(matching_by_k) != (n, 13, 12, 2):
        raise ValueError('C oracle target artifact shape mismatch')
    yc = np.zeros(nc, np.int32)
    yw = np.zeros(n * 12, np.int32)
    for ui, k in enumerate(k_star):
        k = int(k)
        if not 0 <= k <= 12:
            raise ValueError('oracle K outside0..12')
        pairs = np.asarray(matching_by_k[ui, k])
        pairs = pairs[pairs[:, 0] >= 0]
        if len(pairs) != k or len(set(pairs[:, 0])) != k or len(set(pairs[:, 1])) != k:
            raise ValueError('oracle selected rows violate exactlyK identity')
        for cr, slot in pairs:
            if not (0 <= cr < nc and 0 <= slot < 12) or int(data['cold'].iloc[cr].user_index) != ui:
                raise ValueError('oracle row belongs to another user')
            yc[cr] = 1
            yw[ui * 12 + slot] = 1
    return dict(count=np.minimum(np.asarray(k_star), 3).astype(np.int32), cold=yc, warm=yw)


def count_decision(probabilities, cold_counts):
    p = np.asarray(probabilities, float)
    sizes = np.asarray(cold_counts, int)
    if p.shape != (len(sizes), 4) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('C count requires four class probabilities in0,1,2,3 order')
    np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-10, rtol=0)
    return np.minimum(np.argmax(p, axis=1), sizes).astype(np.int32)


def native_parameters(params):
    """Native API preserves fixed4 count classes even if a class is absent."""
    result = dict(params)
    rounds = int(result.pop('n_estimators'))
    for old, new in [('n_jobs', 'num_threads'), ('random_state', 'seed'), ('reg_alpha', 'lambda_l1'),
                     ('reg_lambda', 'lambda_l2'), ('subsample', 'bagging_fraction'), ('colsample_bytree', 'feature_fraction')]:
        if old in result:
            result[new] = result.pop(old)
    if result['objective'] == 'multiclass' and result.get('num_class') != 4:
        raise ValueError('C-count class space must stay fixed at4')
    return result, rounds


def _prepare(repo, root, cutoff, contract, deadline_epoch, labels):
    from .p43a_run import guard
    guard(repo, deadline_epoch)
    if cutoff >= '2020-09-16':
        raise ValueError('C refuses sealed cutoff before feature I/O')
    folder = root / 'C-prepared' / cutoff
    receipt = folder / ('TRAINING.json' if labels else 'FEATURES.json')
    if receipt.exists():
        saved = read_json(receipt)
        assert saved['status'] == 'completed' and saved['cutoff'] == cutoff
        assert saved['features'] == {role: contract['features']['C_exact'][role] for role in ('count', 'cold', 'warm')}
        return saved
    data = joblib.load(Path(contract['reuse_root']) / 'prepared' / cutoff / 'data.joblib')
    audit = ensure_meta(repo, root, cutoff, deadline_epoch)
    validate_lineage(audit, cutoff)
    meta = np.load(audit['paths']['X'], mmap_mode='r')
    if not (folder / 'FEATURES.json').exists():
        features = build_features(data, meta, contract['features']['C_exact'], deadline_epoch)
        folder.mkdir(parents=True, exist_ok=True)
        for role in ('count', 'cold', 'warm'):
            np.save(folder / (role + '-X.npy'), features[role])
        np.save(folder / 'action-users.npy', features['action_users'])
        f = dict(status='completed', cutoff=cutoff, features={role: contract['features']['C_exact'][role] for role in ('count', 'cold', 'warm')},
                 paths={role: str((folder / (role + '-X.npy')).resolve()) for role in ('count', 'cold', 'warm')},
                 action_users=str((folder / 'action-users.npy').resolve()), lineage=audit['lineage'], meta=audit,
                 rows={role: len(features[role]) for role in ('count', 'cold', 'warm')}, final_week='not_run')
        write_json(folder / 'FEATURES.json', f)
        del features
    feature_receipt = read_json(folder / 'FEATURES.json')
    if not labels:
        return feature_receipt
    # Historical oracle labels are joined only after independent PIT features.
    _date_oracle(repo, root, cutoff, deadline_epoch)
    oracle = root / 'oracle' / cutoff
    target = oracle_targets(data, np.load(oracle / 'k_star.npy', mmap_mode='r'),
                            np.load(oracle / 'matching_by_k.npy', mmap_mode='r'))
    action = np.load(feature_receipt['action_users'])
    indices = dict(count=action, cold=np.arange(len(data['cold'])),
                   warm=(action[:, None] * 12 + np.arange(12)).ravel())
    paths = {}
    for role in target:
        np.save(folder / (role + '-y.npy'), target[role])
        np.save(folder / (role + '-training-indices.npy'), indices[role])
        paths[role] = dict(X=feature_receipt['paths'][role], y=str((folder / (role + '-y.npy')).resolve()),
                           indices=str((folder / (role + '-training-indices.npy')).resolve()), rows=len(indices[role]))
    result = dict(status='completed', cutoff=cutoff, paths=paths, features=feature_receipt['features'],
                  feature_receipt=str(folder / 'FEATURES.json'), oracle_receipt=str(oracle / 'RESULT.json'),
                  weight=1., population='historical action users; all Cold and12Warm of those users', final_week='not_run')
    write_json(receipt, result)
    del data, meta, target
    gc.collect()
    return result


def _fit(repo, root, window, config, prepared, contract, deadline_epoch, source_receipt):
    import lightgbm as lgb
    from threadpoolctl import threadpool_limits
    from .p43a_run import guard
    role = config['role']
    folder = root / 'models' / config['id'] / window
    receipt = folder / 'FIT_RESULT.json'
    if receipt.exists():
        saved = read_json(receipt)
        assert saved['status'] == 'completed' and saved['params'] == config['params']
        assert saved['training_cutoffs'] == earlier(WINDOWS[window])
        return lgb.Booster(model_file=str(folder / 'model.txt')), saved
    if (folder / 'FIT_START.json').exists():
        raise RuntimeError('Prior incomplete C fit preserved; no silent refit')
    check_deadline(deadline_epoch)
    folder.mkdir(parents=True, exist_ok=True)
    rows = sum(source['paths'][role]['rows'] for source in prepared)
    fields = contract['features']['C_exact'][role]
    x = np.lib.format.open_memmap(folder / 'training-X.npy', mode='w+', dtype=np.float32, shape=(rows, len(fields)))
    y = np.lib.format.open_memmap(folder / 'training-y.npy', mode='w+', dtype=np.int32, shape=(rows,))
    offset, lineage = 0, []
    for source in prepared:
        guard(repo, deadline_epoch)
        paths = source['paths'][role]
        xx, yy, idx = np.load(paths['X'], mmap_mode='r'), np.load(paths['y'], mmap_mode='r'), np.load(paths['indices'])
        for start in range(0, len(idx), 50000):
            guard(repo, deadline_epoch)
            selected = idx[start:start + 50000]
            x[offset:offset + len(selected)], y[offset:offset + len(selected)] = xx[selected], yy[selected]
            offset += len(selected)
        lineage.append(dict(cutoff=source['cutoff'], rows=len(idx), source=source))
    assert offset == rows and [source['cutoff'] for source in prepared] == earlier(WINDOWS[window])
    x.flush(); y.flush()
    params, rounds = native_parameters(config['params'])
    start = time.perf_counter()
    record = dict(status='running', role=role, config=config, params=config['params'], native_params=params,
                  training_cutoffs=earlier(WINDOWS[window]), cutoff=WINDOWS[window], window=window,
                  rows=rows, features=fields, weight=1., class_counts={str(v): int((y == v).sum()) for v in range(4 if role == 'count' else 2)},
                  source_receipt=source_receipt, implementation=identity(Path(__file__)), sources=lineage, at=now())
    write_json(folder / 'FIT_START.json', record)
    def callback(env):
        try:
            guard(repo, deadline_epoch)
        except (TimeoutError, MemoryError):
            env.model.save_model(str(folder / 'partial-model.txt'))
            write_json(folder / 'FIT_PAUSED.json', dict(completed_iterations=env.iteration + 1,
                       planned_iterations=rounds, valid_for_tournament=False, at=now()))
            raise
    callback.order, callback.before_iteration = 50, False
    try:
        guard(repo, deadline_epoch)
        dataset = lgb.Dataset(x, label=y, feature_name=fields, free_raw_data=True)
        with threadpool_limits(limits=4):
            model = lgb.train(params, dataset, num_boost_round=rounds, callbacks=[callback])
        assert model.feature_name() == fields
        if role == 'count':
            assert model.num_model_per_iteration() == 4
        model.save_model(str(folder / 'model.txt'))
        record.update(status='completed', seconds=time.perf_counter() - start, iterations=model.current_iteration(), at_end=now())
        write_json(receipt, record)
        return model, record
    except Exception:
        write_json(folder / 'FIT_EXCEPTION.json', dict(at=now(), error=traceback.format_exc(), valid_for_tournament=False))
        raise
    finally:
        del x, y
        gc.collect()


def selected_matching(k, cold_scores, warm_scores, cold_ranks, cold_articles, meta, blend):
    """Pure prediction-only selected-side matching; zero blend scores keep K."""
    nc = len(cold_scores)
    if int(k) < 0 or not np.isfinite(cold_scores).all() or not np.isfinite(warm_scores).all():
        raise ValueError('invalid predicted C count or challenger scores')
    k = min(int(k), nc, 3)
    if k == 0:
        return []
    if np.shape(warm_scores) != (12,) or np.shape(meta) != (nc, 12, len(META_COLUMNS)):
        raise ValueError('C inference candidate shape mismatch')
    cold = np.lexsort((np.asarray(cold_articles).astype(str), np.asarray(cold_ranks), -np.asarray(cold_scores)))[:k]
    slots = np.argsort(-np.asarray(warm_scores), kind='stable')[:k]
    value = np.zeros((k, k))
    for name, coefficient in blend.items():
        field = {'old_U': 'old_U_percentile', 'G_UR': 'G_U_R_percentile', 'H_q_cal': 'H_q_cal_percentile'}[name]
        value += coefficient * meta[cold][:, slots, META_COLUMNS.index(field)]
    matching = exactly_k_matching(value, k)
    assert matching is not None and len(matching) == k
    return sorted([(int(cold[i]), int(slots[j])) for i, j in matching], key=lambda pair: pair[1])


def _evaluate(data, meta, predictions, blends, folder, deadline_epoch):
    if data.get('cutoff', '2020-09-16') >= '2020-09-16':
        raise ValueError('C refuses sealed cutoff before evaluation')
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / 'EVALUATION.json'
    if receipt.exists():
        return read_json(receipt)['rows']
    start = time.perf_counter()
    truths, valid, base = contexts(data)
    n, nb = len(data['users']), len(blends)
    groups = data['cold'].groupby('user_index', sort=False).indices
    sizes = np.zeros(n, int)
    for ui, idx in groups.items(): sizes[ui] = len(idx)
    khat = count_decision(predictions['count'], sizes)
    np.save(folder / 'Khat.npy', khat)
    resumed = (folder / 'ACCUMULATORS.npz').exists()
    decisions = np.lib.format.open_memmap(folder / 'C-policy-user-matched-pairs.npy', mode='r+' if resumed else 'w+',
                                         dtype=np.int32, shape=(nb, n, 3, 2))
    if resumed:
        with np.load(folder / 'ACCUMULATORS.npz') as old:
            sums, counts, next_user, previous = old['sums'], old['counts'], int(old['next_user']), float(old['seconds'])
    else:
        decisions[:] = -1
        sums, counts, next_user, previous = np.zeros((nb, 5)), np.zeros((nb, 4), np.int64), 0, 0.
    def checkpoint():
        decisions.flush()
        temp = folder / 'ACCUMULATORS.npz.part'
        with temp.open('wb') as stream:
            np.savez(stream, sums=sums, counts=counts, next_user=np.array(next_user), seconds=np.array(previous + time.perf_counter() - start))
        temp.replace(folder / 'ACCUMULATORS.npz')
    meta3 = np.asarray(meta).reshape(len(data['cold']), 12, len(META_COLUMNS))
    try:
        for ui in range(next_user, n):
            check_deadline(deadline_epoch)
            indices = np.asarray(groups.get(ui, []), int)
            if not khat[ui]:
                next_user = ui + 1
                continue
            cc = data['cold'].iloc[indices]
            user_metrics, user_counts, cache = np.zeros((nb, 5)), np.zeros((nb, 4), np.int64), {}
            for pi, blend in enumerate(blends):
                chosen = selected_matching(khat[ui], predictions['cold'][indices], predictions['warm'].reshape(n, 12)[ui],
                    cc.b0_rank.to_numpy(), cc.article_id.to_numpy(), meta3[indices], blend['weights'])
                decisions[pi, ui] = -1
                decisions[pi, ui, :len(chosen)] = [(indices[i], j) for i, j in chosen]
                key = tuple(chosen)
                if key not in cache:
                    items = list(data['warm_lists'][ui]); ins = rem = 0
                    for ci, slot in chosen:
                        article = cc.iloc[ci].article_id
                        ins += int(article in truths[ui][0]); rem += int(items[slot] in truths[ui][0]); items[slot] = article
                    cache[key] = (np.array([exact_ap(items, truth) for truth in truths[ui]]) - base[ui],
                                  [ins, rem, int(bool(chosen)), len(chosen)])
                user_metrics[pi], user_counts[pi] = cache[key]
            sums += user_metrics; counts += user_counts; next_user = ui + 1
            if next_user % 256 == 0: checkpoint()
    finally:
        checkpoint()
    denominator = valid.sum(axis=0)
    baseline = np.divide(base.sum(axis=0), denominator, out=np.zeros(5), where=denominator != 0)
    delta = np.divide(sums, denominator, out=np.zeros_like(sums), where=denominator != 0)
    seconds = previous + time.perf_counter() - start
    rows = []
    for pi, blend in enumerate(blends):
        segments = {name: dict(map=float(baseline[j] + delta[pi, j]) if denominator[j] else None,
                               delta=float(delta[pi, j]) if denominator[j] else None,
                               map12=float(baseline[j] + delta[pi, j]) if denominator[j] else None,
                               delta_vs_w0=float(delta[pi, j]) if denominator[j] else None,
                               truth_users=int(denominator[j])) for j, name in enumerate(SEGMENTS) if j}
        rows.append(dict(policy_id=blend['id'], policy=blend, overall_map=float(baseline[0] + delta[pi, 0]),
                         delta_map=float(delta[pi, 0]), segments=segments, inserted_positives=int(counts[pi, 0]),
                         removed_positives=int(counts[pi, 1]), admitted_users=int(counts[pi, 2]), coverage=float(counts[pi, 2]/n),
                         replacements=int(counts[pi, 3]), users=n, seconds=seconds/nb, shared_grid_seconds=seconds,
                         saved_decisions=str((folder / 'C-policy-user-matched-pairs.npy').resolve()), decision_policy_index=pi,
                         status='completed', exact_final_list_AP=True))
    write_json(receipt, dict(status='completed', rows=rows, seconds=seconds, final_week='not_run'))
    return rows


def run_c(repo, root, deadline_epoch, source_receipt,max_new_windows=0):
    from .p43a_run import guard
    from .p43a_session import pause_on_gate
    repo, root = Path(repo).resolve(), Path(root).resolve()
    contract = read_json(repo / 'reports/phase4/P4_3A_EXPERIMENT_CONTRACT.json')
    configs = [c for c in contract['model_configs'] if c['arm'] == 'C']
    assert [c['role'] for c in configs] == ['count', 'cold', 'warm']
    path = root / 'TOURNAMENT_STATE.json'
    state = read_json(path)
    record = state['configs'].setdefault(FAMILY, dict(config=dict(id=FAMILY, arm='C', models=configs), windows={}, policies={}, status='not_run'))
    new_windows=0
    try:
        for window, cutoff in WINDOWS.items():
            if window in record['windows']: continue
            guard(repo, deadline_epoch)
            active = [blend for blend in contract['C']['blends']
                      if record['policies'].get(blend['id'], {}).get('status') != 'pruned_bad_config']
            if not active:
                record['status'] = 'all_policies_pruned'
                break
            prepared = [_prepare(repo, root, t, contract, deadline_epoch, True) for t in earlier(cutoff)]
            models, fit_results = {}, {}
            for config in configs:
                models[config['role']], fit_results[config['role']] = _fit(repo, root, window, config, prepared,
                                                                          contract, deadline_epoch, source_receipt)
            feature = _prepare(repo, root, cutoff, contract, deadline_epoch, False)
            data = joblib.load(Path(contract['reuse_root']) / 'prepared' / cutoff / 'data.joblib')
            audit = ensure_meta(repo, root, cutoff, deadline_epoch)
            validate_lineage(audit, cutoff)
            folder = root / 'evaluation' / FAMILY / window
            folder.mkdir(parents=True, exist_ok=True)
            predictions = {}
            for role in ('count', 'cold', 'warm'):
                guard(repo, deadline_epoch)
                x = np.load(feature['paths'][role], mmap_mode='r')
                shape = (len(x), 4) if role == 'count' else (len(x),)
                output = np.lib.format.open_memmap(folder / (role + '-score.npy'), mode='w+', dtype=np.float64, shape=shape)
                for start in range(0, len(x), 50000):
                    guard(repo, deadline_epoch)
                    predicted = models[role].predict(x[start:start+50000], num_threads=4)
                    if not np.isfinite(predicted).all() or np.any((predicted < 0) | (predicted > 1)):
                        raise ValueError('invalid C predicted probabilities')
                    output[start:start+50000] = predicted
                output.flush(); predictions[role] = output
            rows = _evaluate(data, np.load(audit['paths']['X'], mmap_mode='r'), predictions,
                             active, folder / 'policy', deadline_epoch)
            record['windows'][window] = dict(fit=fit_results, evaluated_policies=len(rows), receipt=str(folder / 'policy/EVALUATION.json'))
            if len(record['windows']) == 2 and contract['pruning']['enabled']:
                first = next(iter(record['windows'].values()))['receipt']
                first_rows = {row['policy_id']: row for row in read_json(Path(first))['rows']}
                for row in rows:
                    deltas = [first_rows[row['policy_id']]['delta_map'], row['delta_map']]
                    if all(value < 0 for value in deltas) and float(np.mean(deltas)) < -.0015:
                        record['policies'][row['policy_id']] = dict(status='pruned_bad_config')
            record['status'] = 'completed' if len(record['windows']) == 4 else 'partial'
            write_json(path, state)
            del data, models, predictions
            gc.collect()
            if pause_on_gate(root, state, record):
                return state
            new_windows+=1
            if max_new_windows and new_windows>=max_new_windows:
                state['status']='yielded_window_boundary'
                return state
        return state
    except TimeoutError:
        record['status'] = 'paused_time_budget'; state['status'] = 'paused_time_budget'
        return state
    except MemoryError:
        record['status'] = 'paused_resources'; state['status'] = 'paused_resources'
        return state
    except Exception:
        record['status'] = 'engineering_failure'; record['error'] = traceback.format_exc()
        raise
    finally:
        write_json(path, state)
