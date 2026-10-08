"""Frozen601 Top50 -> list-context bounded residual scorer with exact AP swap weights.

Only historical2019 OUTER expert predictions provide meta-supervision. The four
2020 INNER weeks screen a single fixed variant. This module never registers or
opens an outer trial, and importing it never loads labels or trains a model.
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from . import warm_v3_common as c
from . import warm_v3_candidate_gate as cg
from .warm_v3_gate import safe_training
from .warm_v2_engine import connection, literal, load_parquet, save_parquet
from .warm_v2_rank_fusion import ap_table, final_rank_sql
from .m3 import _peak_working_set_bytes


TRIAL = 'WV3-501'
PARAMS = {'top_n': 50, 'hidden': 32, 'epochs': 8, 'batch_users': 64,
          'learning_rate': .001, 'weight_decay': .0001, 'temperature': 10.,
          'residual_bound': 2., 'residual_penalty': .001, 'seed': 20260909,
          'threads': 4, 'device': 'cpu', 'k': 12,
          'loss': 'full_truth_delta_AP_weighted_all_positive_negative_pairs_divided_by50_user_mean',
          'stage1': 'WV2-601_full_pool_top50',
          'context': 'shared21to32_relu_then_mean_max_context_96to32_relu_to1'}


class Top50Residual(nn.Module):
    """Permutation-equivariant item scores; no labels or ranks hidden in context."""
    def __init__(self, dim=21):
        super().__init__()
        self.item = nn.Sequential(nn.Linear(dim, 32), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x, stage1_rank):
        if x.ndim != 3 or x.shape[1] != 50 or stage1_rank.shape != x.shape[:2]:
            raise ValueError('one batch consists of complete 50-item user lists')
        h = self.item(x)
        mean = h.mean(dim=1, keepdim=True).expand_as(h)
        maximum = h.max(dim=1, keepdim=True).values.expand_as(h)
        raw = self.output(torch.cat((h, mean, maximum), dim=-1)).squeeze(-1)
        residual = 2*torch.tanh(raw)
        return -torch.log(stage1_rank.to(x.dtype))+residual, residual


@torch.no_grad()
def delta_ap_weights(labels, scores, truth_count, k=12):
    """Return BxNxN |delta AP@k| for positive-i/negative-j swaps only.

Ranks are the CURRENT scores with stable input-list order resolving ties. Full
truth_count, not retained positives, is the AP denominator. Each matrix is one
user, so cross-user or cross-cutoff pairs cannot be formed. Weights are detached.
"""
    if labels.ndim != 2 or scores.shape != labels.shape:
        raise ValueError('labels and scores must be same BxN shape')
    if truth_count.shape != labels.shape[:1]:
        raise ValueError('one full-truth denominator is required per user')
    if not torch.isfinite(scores).all() or not torch.isfinite(truth_count).all():
        raise ValueError('non-finite scores or truth counts')
    if not ((labels == 0) | (labels == 1)).all() or not (truth_count > 0).all():
        raise ValueError('binary labels and positive complete-truth counts required')
    if not (labels.sum(dim=1) <= truth_count).all():
        raise ValueError('retained positives cannot exceed complete truth')
    dtype = scores.dtype
    y = labels.to(dtype)
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    sorted_y = y.gather(1, order)
    positions = torch.arange(1, y.shape[1]+1, dtype=dtype, device=y.device)[None, :]
    p = sorted_y.cumsum(dim=1)
    weighted_prefix = (sorted_y/positions*(positions <= k)).cumsum(dim=1)
    inverse = torch.empty_like(order)
    inverse.scatter_(1, order, torch.arange(y.shape[1], device=y.device)[None, :].expand_as(order))
    r = inverse.to(dtype)+1
    p = p.gather(1, inverse)
    wp = weighted_prefix.gather(1, inverse)
    ri, rj = r[:, :, None], r[:, None, :]
    before = ri < rj
    a, b = torch.minimum(ri, rj), torch.maximum(ri, rj)
    pa = torch.where(before, p[:, :, None], p[:, None, :])
    pb = torch.where(before, p[:, None, :], p[:, :, None])
    ya = torch.where(before, y[:, :, None], y[:, None, :])
    yb = torch.where(before, y[:, None, :], y[:, :, None])
    wa = torch.where(before, wp[:, :, None], wp[:, None, :])
    wb = torch.where(before, wp[:, None, :], wp[:, :, None])
    middle = wb-yb/b*(b <= k)-wa
    change = ((pa+1-ya)/a-pb/b*(b <= k)+middle)*(a <= k)
    normalizer = truth_count.to(dtype).clamp(max=k)[:, None, None]
    positive_negative = (y[:, :, None] == 1) & (y[:, None, :] == 0)
    return change.abs()/normalizer*positive_negative


def ap_pair_loss(scores, residual, labels, truth_count):
    weights = delta_ap_weights(labels, scores.detach(), truth_count, k=12)
    comparison = F.softplus(-10*(scores[:, :, None]-scores[:, None, :]))
    per_user = (weights*comparison).sum(dim=(1, 2))/50
    regularizer = .001*residual.square().mean(dim=1)
    return (per_user+regularizer).mean()


def top50_from_arrays(data, keys, info):
    """Select active users'601 Top50 without reading target as a selection feature."""
    if len(keys) != len(data['x']) or len(info['features']) != 21:
        raise ValueError('candidate arrays/key identity or feature count mismatch')
    if 'target' in info['features'] or 'truth_count' in info['features']:
        raise ValueError('supervision cannot be a model input')
    if keys.duplicated(['customer_id', 'article_id']).any():
        raise ValueError('duplicate candidate identity')
    for col, arr in [('target', data['target']), ('truth_count', data['truth_count'])]:
        if not np.array_equal(keys[col].to_numpy(), arr):
            raise ValueError('array/key ordering changed')
    active = keys.user_history_events_12w.to_numpy() > 0
    if not np.array_equal(active, data['active'].astype(bool)):
        raise ValueError('active population changed')
    ids = keys.customer_id.to_numpy()
    expected = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1])+1, len(ids)]
    if not np.array_equal(expected, data['offset']):
        raise ValueError('user boundary or array ordering changed')
    selected = keys.loc[active & (keys.rf.to_numpy() <= 50)].copy()
    selected['array_row'] = np.flatnonzero(active & (keys.rf.to_numpy() <= 50))
    selected = selected.sort_values(['customer_id', 'rf', 'candidate_rank', 'article_id'], kind='stable')
    sizes = selected.groupby('customer_id', sort=False).size()
    if not len(sizes) or not (sizes == 50).all():
        raise ValueError('every active user requires exactly50 frozen Stage1 candidates')
    index = selected.array_row.to_numpy().reshape(-1, 50)
    ranks = selected.rf.to_numpy().reshape(-1, 50)
    if not np.array_equal(ranks, np.broadcast_to(np.arange(1, 51), ranks.shape)):
        raise ValueError('Stage1 ranks are not the exact1..50 permutation')
    truth = data['truth_count'][index]
    if not (truth == truth[:, :1]).all() or not (truth[:, 0] > 0).all():
        raise ValueError('full truth count must be one positive denominator per group')
    return {'x': data['x'][index], 'target': data['target'][index],
            'truth_count': truth[:, 0], 'stage1_rank': ranks.astype(np.float32),
            'array_row': index, 'customer_id': selected.customer_id.to_numpy().reshape(-1, 50)[:, 0],
            'article_id': selected.article_id.to_numpy().reshape(-1, 50)}


def top50_data(meta):
    c.guard(meta['cutoff'])
    data, info = cg.arrays(meta)
    keys = load_parquet(info['keys'])
    top = top50_from_arrays(data, keys, info)
    del data
    return top, info, keys


def validate_training_metadata(metadata, prediction_cutoff):
    c.guard(prediction_cutoff)
    if not metadata or any(m.get('year') != 2019 or m.get('role') != 'outer' for m in metadata):
        raise ValueError('only2019 historical OUTER metadata may supervise this scorer')
    if len(safe_training(metadata, prediction_cutoff)) != len(metadata):
        raise ValueError('training label week overlaps or follows prediction cutoff')
    if len({m['cutoff'] for m in metadata}) != len(metadata):
        raise ValueError('duplicate training cutoff')


def train_tensors(data, check_budget=True):
    """Fixed optimization; this helper also permits a label-free synthetic cost pilot."""
    torch.set_num_threads(4)
    torch.manual_seed(PARAMS['seed'])
    rng = np.random.default_rng(PARAMS['seed'])
    x = data['x']
    if x.shape[1:] != (50, 21) or not np.isfinite(x).all():
        raise ValueError('finite full Top50x21 input required')
    mean = x.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(x.std(axis=(0, 1), dtype=np.float64).astype(np.float32), .01)
    tx = torch.from_numpy((x-mean)/std)
    ranks = torch.from_numpy(data['stage1_rank'])
    labels = torch.from_numpy(data['target'])
    truth = torch.from_numpy(data['truth_count'])
    model = Top50Residual(21)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    curve = []
    start = time.perf_counter()
    for epoch in range(8):
        loss_sum = 0.
        order = rng.permutation(len(tx))
        for begin in range(0, len(order), 64):
            ix = order[begin:begin+64]
            scores, residual = model(tx[ix], ranks[ix])
            loss = ap_pair_loss(scores, residual, labels[ix], truth[ix])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite AP weighted loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(ix)
        curve.append(loss_sum/len(tx))
        if check_budget:
            c.budget(1)
    return model, mean, std, {'params': PARAMS, 'users': len(tx), 'rows': int(np.prod(x.shape[:2])),
        'zero_positive_groups': int((data['target'].sum(axis=1) == 0).sum()),
        'zero_positive_group_share': float((data['target'].sum(axis=1) == 0).mean()),
        'complete_truth_denominator': True, 'loss_curve': curve,
        'fit_seconds': time.perf_counter()-start,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'feature_standardization_population': 'historical2019outer active-user Top50 only',
        'pair_population': 'all positive-negative pairs within each50-item user list; zero-positive groups regularization-only'}


def fit(metadata, prediction_cutoff):
    validate_training_metadata(metadata, prediction_cutoff)
    parts = []
    sources = []
    for meta in metadata:
        top, info, keys = top50_data(meta)
        parts.append({k: top[k] for k in ['x', 'target', 'truth_count', 'stage1_rank']})
        sources.append({'cutoff': meta['cutoff'], 'source_identity': info['source_identity'],
                        'active_users': len(top['x']), 'all_users': meta['total_users']})
        del top, keys
        gc.collect()
    combined = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    del parts
    model, mean, std, out = train_tensors(combined)
    out.update({'features': info['features'], 'training_sources': sources,
                'meta_cutoffs': [m['cutoff'] for m in metadata],
                'prediction_cutoff': prediction_cutoff, 'no_future_meta': True,
                'model_inputs_exclude_labels': True, 'final_week': 'not_run'})
    return model, mean, std, out


def residual_scores(stage1_rank, residual):
    """Double-precision inference preserves exact neutral Stage1 order."""
    residual = np.asarray(residual, np.float64)
    if not np.isfinite(residual).all() or not (np.abs(residual) <= 2+1e-7).all():
        raise ValueError('nonfinite or out-of-bound residual')
    return -np.log(np.asarray(stage1_rank, np.float64))+residual


def score(meta, model, mean, std, root):
    top, info, frame = top50_data(meta)
    model.eval()
    residual = []
    with torch.no_grad():
        for lo in range(0, len(top['x']), 256):
            x = torch.from_numpy(((top['x'][lo:lo+256]-mean)/std).astype(np.float32))
            rank = torch.from_numpy(top['stage1_rank'][lo:lo+256])
            _, r = model(x, rank)
            residual.append(r.numpy())
    residual = np.concatenate(residual)
    active = frame.user_history_events_12w.to_numpy() > 0
    frame['stage1_rank'] = np.where(active, frame.rf, frame.candidate_rank)
    frame['score'] = -1e6-frame.stage1_rank.to_numpy(np.float64)
    frame['residual'] = np.nan
    ix = top['array_row'].reshape(-1)
    frame.loc[ix, 'score'] = residual_scores(top['stage1_rank'], residual).reshape(-1)
    frame.loc[ix, 'residual'] = residual.reshape(-1)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with connection() as con:
        con.register('frame', frame)
        con.execute(f'CREATE TEMP TABLE ranked AS SELECT *,{final_rank_sql()} final_rank FROM frame')
        ap = ap_table(con, 'ranked', 'final_rank')
        base = ap_table(con, 'ranked', 'stage1_rank')
        assert con.execute('SELECT count(*) FROM ranked WHERE final_rank<=12 AND stage1_rank>50').fetchone()[0] == 0
        assert con.execute('SELECT count(*) FROM ranked WHERE user_history_events_12w=0 AND final_rank<>candidate_rank').fetchone()[0] == 0
        con.execute(f'COPY ranked TO {literal(root/"ranks.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
    baseline = float(base.ap.sum()/meta['total_users'])
    value = float(ap.ap.sum()/meta['total_users'])
    assert abs(baseline-meta['baseline_map_population_component']) < 1e-12
    return {'cutoff': meta['cutoff'], 'map_component': value, 'baseline_component': baseline,
        'population_delta': value-baseline, 'total_users': meta['total_users'],
        'active_users_reranked': len(top['x']), 'rows': len(frame),
        'retained_top50_rows': len(ix), 'candidate_pool_changed': False,
        'stage2_admission_cap': 50, 'rank51plus_promoted': 0,
        'inactive_fallback_exact': True, 'residual_quantiles': np.quantile(residual, [0, .1, .5, .9, 1]).tolist(),
        'rank_path': str(root/'ranks.parquet'), 'source_identity': info['source_identity']}


def screen():
    c.setup()
    c.budget(15)
    entries = [v for v in c.read(c.REGISTRY)['trials'] if v['experiment_id'] == TRIAL]
    if len(entries) != 1 or entries[0]['outer_exposures'] != 0 or entries[0]['params'] != PARAMS:
        raise ValueError('exact variant must be preregistered before screening')
    destination = c.REPORT/f'{TRIAL}_SCREEN.json'
    if destination.exists():
        return c.read(destination)
    oracle = c.read(c.REPORT/'WV3_100_GATE_ORACLE_AUDIT.json')
    headroom = c.read(c.REPORT/'MULTISTAGE_HEADROOM_AUDIT.json')
    if headroom['fixed_K'] != 50 or headroom['prior_totals']['top50_positive_pairs'] <= 1000:
        raise ValueError('preregistered Top50 headroom audit prerequisite failed')
    first = min(m['cutoff'] for m in oracle['inner'])
    training = safe_training(oracle['historical_meta'], first)
    assert headroom['safe_prior_meta_by_inner_target'][first] == [m['cutoff'] for m in training]
    started = time.perf_counter()
    model, mean, std, metadata = fit(training, first)
    root = c.ART/TRIAL
    root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), root/'model.pt')
    np.savez(root/'scale.npz', mean=mean, std=std)
    c.write(root/'TRAINING.json', metadata)
    windows = {m['window']: score(m, model, mean, std, root/m['window']/'inner') for m in oracle['inner']}
    deltas = [v['population_delta'] for v in windows.values()]
    passed = np.mean(deltas) >= .0001 and sum(d > 0 for d in deltas) >= 3 and min(deltas) >= -.0005
    result = {'experiment_id': TRIAL, 'params': PARAMS, 'windows': windows, 'training': metadata,
        'mean_population_delta': float(np.mean(deltas)), 'positive_windows': sum(d > 0 for d in deltas),
        'worst_delta': min(deltas), 'passed': bool(passed),
        'runtime_seconds': time.perf_counter()-started, 'final_week': 'not_run'}
    c.write(destination, result)
    print({k: v for k, v in result.items() if k not in ['windows', 'training', 'params']}, flush=True)
    return result


def synthetic_pilot():
    """No real customers/labels and no formal model saved; measures fixed8-epoch cost."""
    rng = np.random.default_rng(20260909)
    groups = 512
    y = (rng.random((groups, 50)) < .04).astype(np.uint8)
    y[::4] = 0
    data = {'x': rng.normal(size=(groups, 50, 21)).astype(np.float32), 'target': y,
            'truth_count': np.maximum(y.sum(axis=1)+2, 1).astype(np.int32),
            'stage1_rank': np.tile(np.arange(1, 51, dtype=np.float32), (groups, 1))}
    model, mean, std, report = train_tensors(data, check_budget=False)
    report.update({'scope': 'synthetic512userlists only; no real labels, no formal fit or MAP',
        'feature_standardization_population': 'synthetic512user Top50 only; no historical data read',
        'peak_process_working_set_bytes': _peak_working_set_bytes(),
        'projected_30000_user_8epoch_seconds': report['fit_seconds']*30000/groups,
        'features': 21, 'gpu_used': False, 'final_week': 'not_run'})
    c.write(c.REPORT/'MULTISTAGE_RESOURCE_AUDIT.json', report)
    print(report, flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['synthetic-pilot', 'screen'])
    args = parser.parse_args()
    synthetic_pilot() if args.command == 'synthetic-pilot' else screen()
