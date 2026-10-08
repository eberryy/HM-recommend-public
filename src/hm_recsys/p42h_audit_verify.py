"""Post-run independent descriptive-audit replay; no fit or score mutation.

This was added after formal computation. It reads fixed score arrays, labels,
and reports; it is not part of the recommendation-producing implementation.
Only P4_2H_AUDIT_VERIFICATION.json is written, never formal scores or metrics.
"""
from pathlib import Path
import gc
import time
import traceback

import joblib
import numpy as np

from .p41a_contract import read_json, write_json
from .p42f_contract import WINDOWS, now
from .p42f_data import frame


RUN_ID = 'p4-2h-v1-conditional-bh-hash10-user-oof'
VARIANT = 'H_conditional_BH'


def equal(actual, expected, tolerance=1e-12):
    if actual is None or expected is None:
        assert actual is expected
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def gate_counts(score, target, reported):
    live = score > 0
    n = int(live.sum())
    b, h = int((live & (target > 0)).sum()), int((live & (target < 0)).sum())
    z = n - b - h
    for key, count in [('eligible_edges', n), ('eligible_B', b), ('eligible_N', z), ('eligible_H', h)]:
        assert reported[key] == count
    for key, count in [('B_share', b), ('N_share', z), ('H_share', h)]:
        equal(count / n if n else None, reported[key])
    equal(b / h if h else None, reported['B_H_ratio'])
    assert reported['B_H_ratio_status'] == ('finite' if h else 'infinite' if b else 'undefined')
    return dict(edges=n, B=b, N=z, H=h)


def survival(cold, score, target, reported):
    assert not cold.duplicated(['customer_id', 'article_id']).any()
    positive = cold.target.to_numpy() == 1
    live = positive & np.any((score > 0) & (target > 0), axis=1)
    strict = cold.strict_cold_flag.to_numpy() == 1
    sparse = cold.sparse1_5_flag.to_numpy() == 1
    counts = dict(main_positive_candidates=int(positive.sum()),
                  surviving_positive_candidates=int(live.sum()),
                  strict_main=int((positive & strict).sum()),
                  sparse_main=int((positive & sparse).sum()),
                  strict_surviving=int((live & strict).sum()),
                  sparse_surviving=int((live & sparse).sum()))
    for key, count in counts.items():
        assert reported[key] == count
    equal(live.sum() / positive.sum() if positive.any() else None, reported['survival_rate'])
    return counts


def probability_audit(probability, binary_target, reported):
    q = np.asarray(probability).ravel()
    y = np.asarray(binary_target, int).ravel()
    assert len(q) == len(y) and reported['rows'] == len(q)
    assert reported['positives'] == int(y.sum())
    equal(y.mean(), reported['observed_rate'])
    equal(q.mean(), reported['predicted_mean'])
    equal(np.mean((q - y) ** 2), reported['Brier'])
    clipped = np.clip(q, np.finfo(float).eps, 1 - np.finfo(float).eps)
    equal(np.mean(-np.where(y == 1, np.log(clipped), np.log1p(-clipped))), reported['logloss'])
    # Fixed ten equal-width bins; q==1 belongs to the final bin.
    buckets = np.minimum(np.floor(q * 10).astype(int), 9)
    ece = 0.
    for k, saved in enumerate(reported['reliability']):
        mask = buckets == k
        rows = int(mask.sum())
        assert saved['rows'] == rows and saved['positives'] == int(y[mask].sum())
        assert saved['low'] == k / 10 and saved['high'] == (k + 1) / 10
        observed, predicted = (float(y[mask].mean()), float(q[mask].mean())) if rows else (None, None)
        equal(observed, saved['observed_rate'])
        equal(predicted, saved['predicted_mean'])
        if rows:
            ece += rows / len(q) * abs(observed - predicted)
    equal(ece, reported['ECE'])


def quantile_audit(raw, calibrated, target, reported):
    cuts = np.quantile(raw, [.5, .8, .9, .95, .99], method='linear')
    np.testing.assert_array_equal(cuts, reported['raw_q_cutpoints'])
    groups = np.searchsorted(cuts, raw, side='left')
    levels = [0., .5, .8, .9, .95, .99, 1.]
    for k, saved in enumerate(reported['quantile_bins']):
        mask = groups == k
        rows, b = int(mask.sum()), int((target[mask] > 0).sum())
        assert saved['rows'] == rows and saved['B'] == b and saved['H'] == rows - b
        assert saved['quantile_low'] == levels[k] and saved['quantile_high'] == levels[k + 1]
        equal(b / rows if rows else None, saved['observed_B_rate'])
        equal(raw[mask].mean() if rows else None, saved['mean_q_raw'])
        equal(calibrated[mask].mean() if rows else None, saved['mean_q_cal'])
        equal(raw[mask].min() if rows else None, saved['raw_value_min'])
        equal(raw[mask].max() if rows else None, saved['raw_value_max'])
    assert sum(v['rows'] for v in reported['quantile_bins']) == len(raw)


def summary_audit(values, reported, quantiles=True):
    x = np.asarray(values).ravel()
    finite = x[np.isfinite(x)]
    assert reported['rows'] == len(x) and reported['finite_rows'] == len(finite)
    for key, value in [('mean', finite.mean()), ('min', finite.min()), ('max', finite.max())]:
        equal(float(value), reported[key])
    if quantiles:
        levels = [('p10', .1), ('p25', .25), ('median', .5), ('p75', .75),
                  ('p90', .9), ('p95', .95), ('p99', .99)]
        computed = np.quantile(finite, [q for _, q in levels])
        for (key, _), value in zip(levels, computed):
            equal(float(value), reported[key])


def distribution_audit(outputs, target, reported):
    flat = target.ravel()
    # All/N summaries replay census, mean and extrema; B/H replay every quantile.
    # This covers the key no-B-survival claim without costly repeated N sorting.
    for label, mask in [('all', np.ones(len(flat), bool)), ('B', flat > 0),
                        ('H', flat < 0), ('N', flat == 0)]:
        saved = reported[label]
        assert saved['edges'] == int(mask.sum())
        for name in ('q_raw', 'q_cal', 'r', 'm_B', 'm_H', 'implied_q_threshold'):
            summary_audit(outputs[name].ravel()[mask], saved[name], quantiles=label in ('B', 'H'))
        q = outputs['q_cal'].ravel()[mask]
        threshold = outputs['implied_q_threshold'].ravel()[mask]
        gap = q - threshold
        summary_audit(gap, saved['q_minus_threshold'], quantiles=label in ('B', 'H'))
        assert saved['q_above_threshold'] == int((gap > 0).sum())
        assert saved['q_equal_threshold'] == int((gap == 0).sum())
        assert saved['q_below_threshold'] == int((gap < 0).sum())
        mb, mh = outputs['m_B'].ravel()[mask], outputs['m_H'].ravel()[mask]
        ratio = saved['benefit_to_harm_ratio']
        assert ratio['infinite_edges'] == int(((mh == 0) & (mb > 0)).sum())
        assert ratio['undefined_edges'] == int(((mh == 0) & (mb == 0)).sum())
        if (mh > 0).any():
            summary_audit(mb[mh > 0] / mh[mh > 0], ratio['finite_positive_denominator'],
                          quantiles=label in ('B', 'H'))
    b = flat > 0
    return dict(beneficial_edges=int(b.sum()),
                beneficial_qcal_max=float(outputs['q_cal'].ravel()[b].max()),
                beneficial_threshold_min=float(outputs['implied_q_threshold'].ravel()[b].min()),
                beneficial_max_q_minus_threshold=float((outputs['q_cal'].ravel()[b] -
                                                        outputs['implied_q_threshold'].ravel()[b]).max()),
                B_H_all_quantiles_checked=True, all_N_census_mean_extrema_checked=True)


def priority_audit(cold, gain, utility, saved_rows, reported):
    groups = cold.groupby('user_index', sort=False).indices
    expected = []
    for ui, indices in groups.items():
        local_g, local_u = gain[indices].ravel(), utility[indices].ravel()
        ids = np.flatnonzero(local_g > 0)
        if not len(ids):
            continue
        # Python's stable sorting provides a separate implementation of tie order.
        by_g = sorted(ids.tolist(), key=lambda i: -local_g[i])
        by_u = sorted(ids.tolist(), key=lambda i: -local_u[i])
        row = dict(user_index=int(ui), eligible_edges=len(ids))
        for k in (1, 5):
            denominator = min(k, len(ids))
            common = len(set(by_g[:denominator]).intersection(by_u[:denominator]))
            row.update({f'top{k}_intersection': common, f'top{k}_denominator': denominator,
                        f'top{k}_overlap': common / denominator})
        expected.append(row)
    assert len(saved_rows) == len(expected) == reported['users_with_eligible_edges']
    assert reported['action_users'] == len(groups)
    assert reported['action_users_without_eligible_edges'] == len(groups) - len(expected)
    for saved, recomputed in zip(saved_rows.to_dict('records'), expected):
        for key, value in recomputed.items():
            equal(value, saved[key])
    for k in (1, 5):
        saved = reported['top_overlap'][str(k)]
        intersection = sum(row[f'top{k}_intersection'] for row in expected)
        denominator = sum(row[f'top{k}_denominator'] for row in expected)
        assert saved['users'] == len(expected)
        assert saved['pooled_intersection'] == intersection and saved['pooled_denominator'] == denominator
        equal(np.mean([row[f'top{k}_overlap'] for row in expected]) if expected else None,
              saved['mean_user_overlap'])
        equal(intersection / denominator if denominator else None, saved['pooled_overlap'])
    assert reported['alternative_matching'] is False and reported['alternative_MAP'] is False
    return dict(eligible_users=len(expected), all_user_top1_top5_rows_checked=True,
                top1=reported['top_overlap']['1'], top5=reported['top_overlap']['5'])


def literal_verdict(metrics, old_g, old_f):
    windows = metrics['windows']
    h = [windows[w][VARIANT] for w in WINDOWS]
    g = [old_g[w]['R'] for w in WINDOWS]
    f = [old_f[w] for w in WINDOWS]
    auc = np.array([row['action']['beneficial_vs_harmful']['roc_auc'] for row in h])
    pr = np.array([row['action']['beneficial_vs_harmful']['pr_auc'] for row in h])
    gauc = np.array([row['action']['beneficial_vs_harmful']['roc_auc'] for row in g])
    gpr = np.array([row['action']['beneficial_vs_harmful']['pr_auc'] for row in g])
    action_pass = auc.mean() >= gauc.mean() and (auc >= gauc).sum() >= 3 and pr.mean() >= gpr.mean()
    action = 'supported' if action_pass else 'mixed' if auc.mean() > gauc.mean() or pr.mean() > gpr.mean() else 'rejected'
    surviving = sum(windows[w]['survival']['H']['surviving_positive_candidates'] for w in WINDOWS)
    strict = sum(windows[w]['survival']['H']['strict_surviving'] for w in WINDOWS)
    improved = sum(windows[w]['survival']['H']['surviving_positive_candidates'] >
                   old_g[w]['survival']['R']['surviving_positive_candidates'] for w in WINDOWS)
    survival_pass = surviving >= 8 and improved >= 3 and strict >= 1
    survival_result = 'supported' if survival_pass else 'mixed' if surviving >= 2 else 'rejected'
    inserted = sum(row['admission']['inserted_cold_positives'] for row in h)
    removed = sum(row['admission']['removed_warm_positives'] for row in h)
    ratio = inserted / removed if removed else float('inf') if inserted else 0.
    precision_pass = ratio >= .25 and inserted >= 3 and removed <= 20
    precision = 'supported' if precision_pass else 'mixed' if ratio >= .15 and ratio > 1/15 else 'rejected'
    delta = np.array([row['delta_vs_w0'] for row in h])
    warm = np.array([row['segments']['warm_21_plus']['delta_vs_w0'] for row in h])
    fd = np.array([row['delta_vs_w0'] for row in f])
    fw = np.array([row['segments']['warm_21_plus']['delta_vs_w0'] for row in f])
    safety_pass = delta.mean() >= -.0005 and delta.min() >= -.001 and warm.mean() >= -.0005
    safety = 'supported' if safety_pass else 'mixed' if delta.mean() > fd.mean() and delta.min() > fd.min() and warm.mean() > fw.mean() else 'rejected'
    scaleup = action != 'rejected' and survival_result != 'rejected' and precision != 'rejected' and safety == 'supported'
    expected = dict(conditional_bh_action_signal=action, conditional_bh_survival=survival_result,
                    conditional_bh_policy_precision=precision, conditional_bh_map_safety=safety,
                    full_history_scaleup_allowed=bool(scaleup))
    for key, value in expected.items():
        assert metrics['verdicts'][key] == value
    observed = metrics['verdicts']
    for key, value in dict(mean_auc=auc.mean(), mean_PR=pr.mean(), G_R_mean_auc=gauc.mean(),
                           G_R_mean_PR=gpr.mean(), mean_delta=delta.mean(), worst_delta=delta.min(),
                           mean_warm_delta=warm.mean()).items():
        equal(value, observed[key])
    for key, value in dict(auc_nondegrade=int((auc >= gauc).sum()), surviving=surviving,
                           strict_surviving=strict, survival_improved_windows=improved,
                           inserted=inserted, removed=removed).items():
        assert observed[key] == value
    equal(ratio if removed else None, observed['ratio'])
    assert observed['ratio_status'] == ('finite' if removed else 'infinite' if inserted else 'undefined')
    return expected


def run(repo):
    repo = Path(repo).resolve()
    report = repo / 'reports/phase4'
    destination = report / 'P4_2H_AUDIT_VERIFICATION.json'
    if destination.exists():
        raise FileExistsError('post-run audit receipt already exists; no silent replacement')
    root = repo / 'artifacts/phase4' / RUN_ID
    contract = read_json(report / 'P4_2H_EXPERIMENT_CONTRACT.json')
    froot = Path(contract['reuse_root'])
    metrics = read_json(report / 'P4_2H_metrics.json')
    assert metrics['status'] in ('completed_pending_verification', 'completed')
    start = time.perf_counter()
    result = dict(stage='P4.2H_post_run_audit', status='running', windows={},
                  model_fits=0, score_mutations=0, alternative_policy_MAP_runs=0,
                  provenance='post-formal independent descriptive-audit checks, not formal runtime',
                  final_week='not_run')
    try:
        for window, cutoff in WINDOWS.items():
            data = joblib.load(froot / 'prepared' / cutoff / 'data.joblib')
            folder = root / 'outer' / window
            target = np.load(froot / 'outer' / window / 'single_delta.npy', mmap_mode='r')
            scores = {name: np.load(folder / (name + '.npy'), mmap_mode='r')
                      for name in ('G_BH', 'G_raw', 'U_H', 'q_raw', 'q_cal', 'r', 'm_B', 'm_H', 'implied_q_threshold')}
            saved = metrics['windows'][window]
            gate = gate_counts(scores['G_BH'], target, saved['gate']['H'])
            raw_gate = gate_counts(scores['G_raw'], target, saved['diagnostics']['D1_raw_q']['gate'])
            survival_h = survival(data['cold'], scores['G_BH'], target, saved['survival']['H'])
            survival_raw = survival(data['cold'], scores['G_raw'], target, saved['survival']['raw'])
            assert saved['survival']['raw'] == saved['diagnostics']['D1_raw_q']['survival']
            nonzero = target.ravel() != 0
            raw = scores['q_raw'].ravel()[nonzero]
            calibrated = scores['q_cal'].ravel()[nonzero]
            y = target.ravel()[nonzero]
            for name, q in [('q_raw', raw), ('q_cal', calibrated)]:
                probability_audit(q, y > 0, saved['q_calibration'][name])
                assert saved['q_calibration'][name]['diagnostic_calibration']['audit_only_not_applied']
            quantile_audit(raw, calibrated, y, saved['q_calibration'])
            probability_audit(scores['r'].ravel(), nonzero, saved['nonzero_head'])
            priority = priority_audit(data['cold'], scores['G_BH'], scores['U_H'],
                                      frame(folder / 'priority-diagnostic.parquet'),
                                      saved['diagnostics']['D3_no_r_priority'])
            distributions = distribution_audit(scores, target, saved['conditional_distributions'])
            result['windows'][window] = dict(
                gate=gate, raw_gate=raw_gate, unique_positive_survival=survival_h,
                raw_unique_positive_survival=survival_raw, BH_probability_metrics_recomputed=True,
                quantile_bins_recomputed=True, nonzero_probability_metrics_recomputed=True,
                priority=priority, distributions=distributions)
            print('P4.2H independent audit verified ' + window, flush=True)
            del data, scores, target, raw, calibrated, y, nonzero
            gc.collect()
        result['literal_verdicts'] = literal_verdict(metrics,
            read_json(report / 'P4_2G_metrics.json')['windows'],
            read_json(report / 'p4_2f_global_utility.json')['data'])
        assert metrics['final_week'] == 'not_run' and metrics['full_history_run'] is False
        result['status'] = 'pass'
    except Exception:
        result['status'] = 'fail'
        result['error'] = traceback.format_exc()
        raise
    finally:
        result.update(seconds=time.perf_counter() - start, at=now())
        write_json(destination, result)
    return result


if __name__ == '__main__':
    run(Path('.'))
