"""One formal conditional-risk policy and fixed, non-policy diagnostics.

All edge arrays use (original Cold row, original Warm slot), with float64
scores. Diagnostic raw-q and no-r scores never generate recommendation lists.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .p41a_contract import read_json, write_json
from .p42d_stats import correlations, discrimination, summary
from .p42f_core import batches
from .p42f_data import save_frame
from .p42f_evaluate import user_summary
from .p42_evaluate import _segment_truth, _segment_metrics
from .p42_matching import exact_matching, apply_admissions
from .metrics import apk
from .p42g_core import action, tail, survival
from .p42h_core import utilities, calibrated, binary_calibration


VARIANT = 'H_conditional_BH'
SCORES = ('q_raw', 'q_cal', 'r', 'm_B', 'm_H',
          'implied_q_threshold', 'G_BH', 'U_H', 'G_raw')


def conditional_bins(raw, calibrated_q, target):
    """Six raw-q value-quantile bins, using the same B/H rows for raw/cal."""
    raw = np.asarray(raw, float).ravel()
    calibrated_q = np.asarray(calibrated_q, float).ravel()
    target = np.asarray(target).ravel()
    assert len(raw) == len(calibrated_q) == len(target)
    assert np.all(target != 0)
    levels = [0., .5, .8, .9, .95, .99, 1.]
    cutpoints = np.quantile(raw, levels[1:-1])
    groups = np.searchsorted(cutpoints, raw, side='left')
    rows = []
    for k, (low, high) in enumerate(zip(levels[:-1], levels[1:])):
        mask = groups == k
        n = int(mask.sum())
        b = int((target[mask] > 0).sum())
        rows.append(dict(quantile_low=low, quantile_high=high, rows=n, B=b,
                         H=n-b, observed_B_rate=b/n if n else None,
                         mean_q_raw=float(raw[mask].mean()) if n else None,
                         mean_q_cal=float(calibrated_q[mask].mean()) if n else None,
                         raw_value_min=float(raw[mask].min()) if n else None,
                         raw_value_max=float(raw[mask].max()) if n else None))
    assert sum(row['rows'] for row in rows) == len(raw)
    return dict(raw_q_cutpoints=cutpoints.tolist(), quantile_bins=rows,
                bin_rule='raw-q value quantiles among outer B/H edges; equal values enter lower bin; raw/cal share rows',
                denominator='complete outer B/H action edges; empty bins retained')


def priority_diagnostic(cold, gain, utility, guard):
    """Compare ordering only; no alternative matching or recommendation."""
    eligible = np.asarray(gain) > 0
    np.testing.assert_array_equal(eligible, np.asarray(utility) > 0)
    top = {1: [], 5: []}
    intersections = {1: 0, 5: 0}
    denominators = {1: 0, 5: 0}
    records = []
    groups = cold.groupby('user_index', sort=False).indices
    for serial, (ui, indices) in enumerate(groups.items()):
        if serial % 256 == 0:
            guard()
        indices = np.asarray(indices)
        local_gain = np.asarray(gain[indices]).ravel()
        local_utility = np.asarray(utility[indices]).ravel()
        edge_ids = np.flatnonzero(local_gain > 0)
        if not len(edge_ids):
            continue
        # Stable descending sorts retain canonical Cold-row then Warm-slot ties.
        by_gain = edge_ids[np.argsort(-local_gain[edge_ids], kind='stable')]
        by_utility = edge_ids[np.argsort(-local_utility[edge_ids], kind='stable')]
        row = dict(user_index=int(ui), eligible_edges=len(edge_ids))
        for k in (1, 5):
            denominator = min(k, len(edge_ids))
            overlap = len(set(by_gain[:denominator]) & set(by_utility[:denominator]))
            top[k].append(overlap / denominator)
            intersections[k] += overlap
            denominators[k] += denominator
            row.update({f'top{k}_intersection': overlap,
                        f'top{k}_denominator': denominator,
                        f'top{k}_overlap': overlap / denominator})
        records.append(row)
    flat_gain = np.asarray(gain).ravel()
    flat_utility = np.asarray(utility).ravel()
    eligible_flat = eligible.ravel()
    result = dict(
        all_edges=correlations(flat_gain, flat_utility),
        eligible_edges=correlations(flat_gain[eligible_flat], flat_utility[eligible_flat]),
        action_users=len(groups), users_with_eligible_edges=len(records),
        action_users_without_eligible_edges=len(groups)-len(records),
        top_overlap={str(k): dict(
            users=len(top[k]),
            mean_user_overlap=float(np.mean(top[k])) if top[k] else None,
            pooled_intersection=intersections[k], pooled_denominator=denominators[k],
            pooled_overlap=intersections[k]/denominators[k] if denominators[k] else None)
            for k in (1, 5)},
        denominator='same-user eligible action edges; per-user intersection/min(K,eligible count); zero-eligible users excluded',
        tie_rule='original Cold row then original Warm slot, stable descending score',
        alternative_matching=False, alternative_MAP=False)
    return result, records


def conditional_distributions(outputs, target, guard):
    """Label-conditioned diagnostics only, never inputs to prediction/gating."""
    flat_target = np.asarray(target).ravel()
    result = {}
    for label, mask in [('all', np.ones(len(flat_target), bool)),
                        ('B', flat_target > 0), ('H', flat_target < 0),
                        ('N', flat_target == 0)]:
        guard()
        dist = {}
        for key in ('q_raw', 'q_cal', 'r', 'm_B', 'm_H', 'implied_q_threshold'):
            dist[key] = summary(np.asarray(outputs[key]).ravel()[mask])
        gap = (np.asarray(outputs['q_cal']).ravel()[mask] -
               np.asarray(outputs['implied_q_threshold']).ravel()[mask])
        dist['q_minus_threshold'] = summary(gap)
        dist['q_above_threshold'] = int((gap > 0).sum())
        dist['q_equal_threshold'] = int((gap == 0).sum())
        dist['q_below_threshold'] = int((gap < 0).sum())
        dist['edges'] = int(mask.sum())
        mb = np.asarray(outputs['m_B']).ravel()[mask]
        mh = np.asarray(outputs['m_H']).ravel()[mask]
        # A ratio is diagnostic, not part of gate; represent zero denominators.
        ratio = mb[mh > 0] / mh[mh > 0]
        dist['benefit_to_harm_ratio'] = dict(
            finite_positive_denominator=summary(ratio),
            infinite_edges=int(((mh == 0) & (mb > 0)).sum()),
            undefined_edges=int(((mh == 0) & (mb == 0)).sum()))
        result[label] = dist
    return result


def evaluate(repo, data, models, folder, ffolder, gfolder, guard):
    repo, folder, ffolder, gfolder = map(Path, (repo, folder, ffolder, gfolder))
    folder.mkdir(parents=True, exist_ok=False)
    cold = data['cold']
    n, nc = len(data['users']), len(cold)
    outputs = {key: np.lib.format.open_memmap(folder/(key+'.npy'), mode='w+',
               dtype='float64', shape=(nc, 12)) for key in SCORES}
    target = np.load(ffolder/'single_delta.npy', mmap_mode='r')
    old_magnitude = {key: np.load(gfolder/(key+'.npy'), mmap_mode='r')
                     for key in ('m_B_raw', 'm_H_raw', 'm_B', 'm_H')}
    for start, cc, x, y in batches(data):
        guard()
        end = start + len(cc)
        qraw = models['q'].predict(x, num_threads=4)
        qcal = calibrated(qraw, models['calibrator'])
        r = models['r'].predict(x, num_threads=4)
        mbraw = models['benefit'].predict(x, num_threads=4).reshape(-1, 12)
        mhraw = models['harm'].predict(x, num_threads=4).reshape(-1, 12)
        np.testing.assert_array_equal(y, target[start:end])
        np.testing.assert_array_equal(mbraw, old_magnitude['m_B_raw'][start:end])
        np.testing.assert_array_equal(mhraw, old_magnitude['m_H_raw'][start:end])
        mb = np.clip(mbraw, 0, 1)
        mh = np.clip(mhraw, 0, 1)
        np.testing.assert_array_equal(mb, old_magnitude['m_B'][start:end])
        np.testing.assert_array_equal(mh, old_magnitude['m_H'][start:end])
        threshold, gbh, uh = utilities(qcal.reshape(-1, 12), r.reshape(-1, 12), mb, mh)
        graw = qraw.reshape(-1, 12)*mb-(1-qraw.reshape(-1, 12))*mh
        for key, value in [('q_raw', qraw), ('q_cal', qcal), ('r', r),
                           ('m_B', mb), ('m_H', mh),
                           ('implied_q_threshold', threshold), ('G_BH', gbh),
                           ('U_H', uh), ('G_raw', graw)]:
            assert np.isfinite(value).all()
            outputs[key][start:end] = np.asarray(value).reshape(-1, 12)
    for array in outputs.values():
        array.flush()

    guard()
    gbh, uh, graw = (outputs[key] for key in ('G_BH', 'U_H', 'G_raw'))
    target_flat = target.ravel()
    nonzero = target_flat != 0
    raw_bh = outputs['q_raw'].ravel()[nonzero]
    cal_bh = outputs['q_cal'].ravel()[nonzero]
    y_bh = target_flat[nonzero]
    calibration = dict(
        q_raw=binary_calibration(raw_bh, y_bh > 0, diagnostic_fit=True),
        q_cal=binary_calibration(cal_bh, y_bh > 0, diagnostic_fit=True),
        q_raw_discrimination=discrimination(raw_bh, y_bh > 0),
        q_cal_discrimination=discrimination(cal_bh, y_bh > 0),
        outer_diagnostic_only=True,
        fitted_outer_diagnostics_consumed_by_policy=False,
        **conditional_bins(raw_bh, cal_bh, y_bh))
    guard()
    nonzero_audit = binary_calibration(outputs['r'].ravel(), nonzero, diagnostic_fit=False)
    nonzero_audit.update(target='B or H iff exact single_delta != 0',
                         denominator='complete outer B/N/H action edges',
                         additional_probability_calibration=False)
    guard()
    primary_actions = dict(G_BH=action(gbh, target), U_H=action(uh, target))
    diagnostic_raw = dict(
        action=dict(beneficial_vs_harmful=discrimination(graw.ravel()[nonzero], y_bh > 0)),
        gate=tail(graw, target), survival=survival(cold, graw, target),
        formal_matching=False, formal_MAP=False)
    diagnostic_q = dict(beneficial_vs_harmful=discrimination(cal_bh, y_bh > 0),
                        formal_matching=False, formal_MAP=False)
    priority, priority_rows = priority_diagnostic(cold, gbh, uh, guard)
    distributions = conditional_distributions(outputs, target, guard)

    fscore = np.load(ffolder/'G_expected_deltaAP-utility.npy', mmap_mode='r')
    gscore = np.load(gfolder/'U_R.npy', mmap_mode='r')
    fresult = read_json(ffolder/'EVALUATION.json')
    gresult = read_json(gfolder/'EVALUATION.json')
    old = read_json(repo/'reports/phase4/P4_2D_metrics.json')['windows'][folder.name]['pair']
    assert old['comparison']['secondary']['rows'] == target.size
    assert old['comparison']['primary']['positives'] == int((target > 0).sum())
    magnitude = {}
    for name, mask, key in [('benefit', target > 0, 'm_B'), ('harm', target < 0, 'm_H')]:
        pred = outputs[key][mask]
        truth = np.abs(target[mask])
        raw = old_magnitude[key+'_raw']
        magnitude[name] = dict(
            edges=int(mask.sum()), observed_mean=float(truth.mean()),
            predicted_mean=float(pred.mean()),
            conditional_MAE=float(np.abs(pred-truth).mean()),
            conditional_RMSE=float(np.sqrt(np.square(pred-truth).mean())),
            clipped_low_edges=int((raw < 0).sum()),
            clipped_high_edges=int((raw > 1).sum()),
            exact_reuse_P4_2G=True, exact_prediction_parity=True)

    # Exactly one recommendation-producing policy: G_BH gate, U_H weights.
    lists = data['warm_lists'].copy()
    counts, ins, rem = (np.zeros(n, int) for _ in range(3))
    sums = np.zeros(n)
    strict, sparse, coldonly = (np.zeros(n, int) for _ in range(3))
    rows = []
    for serial, (ui, indices) in enumerate(cold.groupby('user_index', sort=False).indices.items()):
        if serial % 256 == 0:
            guard()
        indices = np.asarray(indices)
        cc = cold.iloc[indices]
        np.testing.assert_array_equal(gbh[indices] > 0, uh[indices] > 0)
        matches = exact_matching(np.where(gbh[indices] > 0, uh[indices], 0.), 0.)
        lists[ui] = apply_admissions(list(lists[ui]), cc.article_id.tolist(), matches)
        counts[ui] = len(matches)
        for ci, j in matches:
            cr = int(indices[ci])
            item = cc.iloc[ci]
            ip = int(item.target)
            rp = int(data['relevance'][ui, j])
            value = float(target[cr, j])
            ins[ui] += ip
            rem[ui] += rp
            sums[ui] += value
            strict[ui] += ip*int(item.strict_cold_flag)
            sparse[ui] += ip*int(item.sparse1_5_flag)
            coldonly[ui] += ip*int(item.cold_only)
            rows.append(dict(
                customer_id=data['users'][ui], cold_row=cr,
                cold_article_id=item.article_id, warm_article_id=data['warm_lists'][ui, j],
                warm_slot=j+1, utility=float(uh[cr, j]),
                conditional_gain=float(gbh[cr, j]), single_delta=value,
                inserted_positive=ip, removed_positive=rp,
                strict_cold=int(item.strict_cold_flag), sparse1_5=int(item.sparse1_5_flag)))
    ap = np.array([apk(list(data['truthsets'][u]), list(pred))
                   for u, pred in zip(data['users'], lists)])
    delta = ap-data['baseline_ap']
    admission = user_summary(np.ones(n, bool), delta, counts, ins, rem)
    admission.update(mean_per_admitted=float(counts[counts > 0].mean()) if (counts > 0).any() else 0.,
                     mean_per_all_users=float(counts.mean()), max_replacements=int(counts.max()),
                     strict_positive_inserted=int(strict.sum()), sparse_positive_inserted=int(sparse.sum()),
                     cold_only_positive_inserted=int(coldonly.sum()))
    buckets = {}
    for name, mask in [('0', counts == 0), ('1', counts == 1), ('2', counts == 2),
                       ('3', counts == 3), ('4+', counts >= 4)]:
        buckets[name] = user_summary(mask, delta, counts, ins, rem)
        buckets[name]['mean_exact_minus_single_sum'] = float((delta-sums)[mask].mean()) if mask.any() else None
    cuts = fresult['hierarchical']
    nq = np.searchsorted(cuts['novelty_cutpoints'], data['state'].novel_purchase_share_0_5.fillna(0), side='left')
    rq = np.searchsorted(cuts['richness_cutpoints'], data['state'].user_past_purchase_count, side='left')
    mechanisms = dict(
        novelty={f'Q{k+1}': user_summary(nq == k, delta, counts, ins, rem) for k in range(4)},
        richness={name: user_summary(rq == k, delta, counts, ins, rem)
                  for k, name in enumerate(['low', 'medium', 'high'])})
    segments = _segment_truth(data)
    base = _segment_metrics(data, data['warm_lists'], segments)
    hresult = dict(map12=float(ap.mean()), delta_vs_w0=float(ap.mean()-data['baseline_map']),
                   segments=_segment_metrics(data, lists, segments, base),
                   action=primary_actions['U_H'], admission=admission,
                   buckets=buckets, mechanisms=mechanisms)
    result = dict(
        W0=dict(map12=data['baseline_map'], delta_vs_w0=0., segments=base),
        H_conditional_BH=hresult,
        action=primary_actions, q_calibration=calibration, nonzero_head=nonzero_audit,
        magnitude=magnitude, conditional_distributions=distributions,
        diagnostics=dict(D1_raw_q=diagnostic_raw, D2_classification_only=diagnostic_q,
                         D3_no_r_priority=priority),
        references=dict(G_R=gresult['R']['action'],
                        F_G=fresult['variants']['G_expected_deltaAP']['action'],
                        old_U=dict(beneficial_vs_harmful=old['comparison']['primary'],
                                   beneficial_vs_all_nonbeneficial=old['comparison']['secondary'],
                                   correlation=old['alignment']['overall'])),
        gate=dict(H=tail(gbh, target), G_R=tail(gscore, target), F_G=tail(fscore, target)),
        survival=dict(H=survival(cold, gbh, target), raw=survival(cold, graw, target),
                      G_R=survival(cold, gscore, target), F_G=survival(cold, fscore, target),
                      old_U=old['survival']['A_tau0']),
        cutpoints=dict(novelty=cuts['novelty_cutpoints'], richness=cuts['richness_cutpoints']),
        output_score_keys=list(SCORES), formal_variants=['W0', VARIANT],
        matching_calls_variant_count=1, final_week='not_run')
    save_frame(pd.DataFrame(rows, columns=[
        'customer_id', 'cold_row', 'cold_article_id', 'warm_article_id', 'warm_slot',
        'utility', 'conditional_gain', 'single_delta', 'inserted_positive',
        'removed_positive', 'strict_cold', 'sparse1_5']), folder/'executed.parquet')
    save_frame(pd.DataFrame(dict(customer_id=data['users'], ap=ap,
        baseline_ap=data['baseline_ap'], delta=delta, admissions=counts,
        inserted=ins, removed=rem, single_sum=sums, novelty_group=nq,
        richness_group=rq)), folder/'users.parquet')
    save_frame(pd.DataFrame(dict(customer_id=np.repeat(data['users'], 12),
        rank=np.tile(np.arange(1, 13), n), article_id=lists.ravel())), folder/'lists.parquet')
    save_frame(pd.DataFrame(priority_rows, columns=[
        'user_index', 'eligible_edges', 'top1_intersection', 'top1_denominator', 'top1_overlap',
        'top5_intersection', 'top5_denominator', 'top5_overlap']), folder/'priority-diagnostic.parquet')
    write_json(folder/'EVALUATION.json', result)
    return result
