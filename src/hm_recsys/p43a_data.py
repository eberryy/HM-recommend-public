"""Bounded-memory P4.3A data adapters; no recommendation model is fitted here.

Original F70 columns retain their full-Cold50/Top12 normalization. A1/D reuse
the exact F retained rows, but training weights are one rather than IPW 50.
The hard pool is shared by A2/B2 and E; E excludes its extra easy neutrals.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import gc
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .p41a_contract import read_json, write_json, check_identity
from .p42f_contract import RUN_ID as F_RUN, DATES, WINDOWS, GLOBAL, earlier, now
from .p42f_data import records, frame, paths_for
from .p42f_core import batches

G_RUN = 'p4-2g-v1-risk-separated-hash10'
H_RUN = 'p4-2h-v1-conditional-bh-hash10-user-oof'
R3_RUN = 'p4-2r3-v1-lbfgs-solver-boundary-repair'
META_SCORES = ['old_U','F_G','G_U_R','H_q_raw','H_q_cal','H_G_BH','H_r']
META_COLUMNS = [column for name in META_SCORES
                for column in (name+'_percentile', name+'_available')]
_MANIFESTS = {}
_CHECKED = set()


def _guard(deadline_epoch=None):
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise TimeoutError('P4.3A cumulative time budget reached; preserve partial files')


def _date_guard(t, historical=False):
    allowed = DATES if historical else sorted(set(DATES + list(WINDOWS.values())))
    if t not in allowed:
        raise ValueError('cutoff is outside the preregistered historical/development dates')


def source_for(t):
    """Latest existing model whose complete label pool ends strictly before t."""
    _date_guard(t)
    choices = []
    for window, source_cutoff in WINDOWS.items():
        cutoffs = earlier(source_cutoff)
        if not cutoffs:
            continue
        end = max((date.fromisoformat(s) + timedelta(days=7)).isoformat() for s in cutoffs)
        if source_cutoff <= t and end < t:
            choices.append(dict(window=window, scoring_cutoff=t,
                score_source_cutoff=source_cutoff, training_label_end=end,
                training_cutoffs=cutoffs, availability=1))
    return choices[-1] if choices else dict(window=None, scoring_cutoff=t,
        score_source_cutoff=None, training_label_end=None,
        training_cutoffs=[], availability=0)


def trusted_source(repo, path, manifest_name):
    """Compare reused assets with their prior recorded identity, once per stat."""
    repo, path = Path(repo).resolve(), Path(path).resolve()
    key = (str(repo).lower(), manifest_name)
    if key not in _MANIFESTS:
        manifest = read_json(repo / 'reports/phase4' / manifest_name)
        _MANIFESTS[key] = {str(Path(r['path']).resolve()).lower(): r for r in records(manifest)}
    record = _MANIFESTS[key].get(str(path).lower())
    if record is None:
        raise ValueError(f'No prior trusted identity for {path}')
    st = path.stat()
    checked = (str(path).lower(), st.st_size, st.st_mtime_ns, record['sha256'])
    if checked not in _CHECKED:
        check_identity(record)
        _CHECKED.add(checked)
    return record


def load_f_data(repo, t, deadline_epoch=None):
    _date_guard(t); _guard(deadline_epoch)
    repo = Path(repo).resolve()
    path = repo / 'artifacts/phase4' / F_RUN / 'prepared' / t / 'data.joblib'
    identity = trusted_source(repo, path, 'P4_2F_OUTPUT_MANIFEST.json')
    data = joblib.load(path)
    if data['cutoff'] != t:
        raise ValueError('F data cutoff differs from requested cutoff')
    np.testing.assert_array_equal(data['state'].customer_id.to_numpy(), data['users'])
    ui = data['cold'].user_index.to_numpy(np.int64)
    if len(ui) and (np.diff(ui) < 0).any():
        raise ValueError('Cold rows must follow canonical contiguous user order')
    return data, identity


def _counts(y):
    return dict(B=int(np.count_nonzero(y > 0)), N=int(np.count_nonzero(y == 0)),
                H=int(np.count_nonzero(y < 0)))


def _group_arrays(user_index):
    ui = np.asarray(user_index, dtype=np.int64)
    if len(ui) and (np.diff(ui) < 0).any():
        raise ValueError('Rank groups are not contiguous in canonical user order')
    starts = np.r_[0, np.flatnonzero(np.diff(ui)) + 1] if len(ui) else np.array([], np.int64)
    group = np.diff(np.r_[starts, len(ui)]).astype(np.int32) if len(ui) else np.array([], np.int32)
    return group, ui[starts].astype(np.int32)


def recover_edge_indices(data, x, ui):
    """Unique F-row identity from user, original B0 rank and rank-times-slot.

    Both integer fields are <=600 and exactly represented by F float32.
    This avoids rescanning all discarded neutral rows just to reuse A1.
    """
    ui = np.asarray(ui, np.int64)
    rank_raw = np.asarray(x[:, GLOBAL.index('b0_rank')], np.float64)
    product = np.asarray(x[:, GLOBAL.index('cold_rank_times_warm_slot')], np.float64)
    ranks = rank_raw.astype(np.int64)
    if not np.array_equal(rank_raw, ranks) or ((ranks < 1) | (ranks > 50)).any():
        raise ValueError('Invalid original Cold50 B0 rank')
    slot_raw = product / rank_raw
    slots = slot_raw.astype(np.int64)
    if not np.array_equal(slot_raw, slots) or ((slots < 1) | (slots > 12)).any():
        raise ValueError('Pair feature no longer uniquely identifies original Warm slot')
    lookup = np.full((len(data['users']), 51), -1, dtype=np.int64)
    cu = data['cold'].user_index.to_numpy(np.int64)
    cr = data['cold'].b0_rank.to_numpy(np.int64)
    if len(set(zip(cu.tolist(), cr.tolist()))) != len(cu):
        raise ValueError('Duplicate user/original-rank Cold identity')
    lookup[cu, cr] = np.arange(len(cu))
    ci = lookup[ui, ranks]
    if (ci < 0).any():
        raise ValueError('Retained row not found in frozen post-overlap Cold pool')
    edge = ci * 12 + slots - 1
    if len(edge) > 1 and (np.diff(edge) <= 0).any():
        raise ValueError('Retained edges must have unique increasing canonical identities')
    return edge


def _save_array(path, values, dtype, deadline_epoch=None, chunk=65536):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Preserve existing partial artifact: {path}')
    output = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=values.shape)
    for start in range(0, len(values), chunk):
        _guard(deadline_epoch)
        output[start:start+chunk] = values[start:start+chunk]
    output.flush()
    del output
    return str(path.resolve())


def _write_groups(folder, ui, deadline_epoch=None):
    group, group_users = _group_arrays(ui)
    return dict(group=_save_array(folder/'group.npy', group, np.int32, deadline_epoch),
        group_user_index=_save_array(folder/'group_user_index.npy', group_users, np.int32, deadline_epoch),
        groups=len(group))


def _merge_date_audit(folder, entry):
    path = folder/'AUDIT.json'
    audit = read_json(path) if path.exists() else dict(cutoff=entry['cutoff'], variants={}, final_week='not_run')
    audit['variants'][entry['sampling']] = entry
    write_json(path, audit)
    return dict(entry, audit_path=str((folder/entry['sampling']/'AUDIT.json').resolve()),
                date_audit_path=str(path.resolve()))


def _prepare_a1(repo, root, t, deadline_epoch=None):
    folder = root/'prepared'/t/'A1'
    if (folder/'AUDIT.json').exists():
        return _merge_date_audit(folder.parent, read_json(folder/'AUDIT.json'))
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data, source_data = load_f_data(repo, t, deadline_epoch)
    npz = repo/'artifacts/phase4'/F_RUN/'prepared'/t/'training.npz'
    source_rows = trusted_source(repo, npz, 'P4_2F_OUTPUT_MANIFEST.json')
    with np.load(npz, allow_pickle=False) as saved:
        x, y, ui, weight = (saved[k] for k in ('X','y','user_index','weight'))
    if x.dtype != np.float32 or x.shape != (len(y), len(GLOBAL)) or not np.isfinite(y).all():
        raise ValueError('Frozen F retained matrix schema changed')
    np.testing.assert_array_equal(weight, np.where(y == 0, 50., 1.))
    edge = recover_edge_indices(data, x, ui)
    paths = {name:_save_array(folder/f'{name}.npy', value, dtype, deadline_epoch)
             for name,value,dtype in [('X',x,np.float32),('y',y,np.float64),
                 ('user_index',ui,np.int32),('edge_index',edge,np.int64)]}
    paths.update(_write_groups(folder, ui, deadline_epoch))
    # Exact content parity is cheap on the retained pool, unlike full neutral reconstruction.
    for name, expected in [('X',x),('y',y),('user_index',ui)]:
        np.testing.assert_array_equal(np.load(paths[name], mmap_mode='r'), expected)
    prior = read_json(repo/'reports/phase4/p4_2g_data_parity.json')['cutoffs'][t]
    count = _counts(y)
    if len(y) != prior['retained_rows'] or count['B'] != prior['B'] or count['H'] != prior['H']:
        raise ValueError('A1 no longer equals the complete F retained B/H population')
    entry = dict(status='completed', cutoff=t, sampling='A1', paths=paths,
        rows=len(y), groups=paths['groups'], counts=count,
        full_counts={k:prior[k] for k in ('B','N','H')}, full_edges=prior['edges'],
        features=list(GLOBAL), columns=len(GLOBAL), training_weight=1,
        source_identity=dict(data=source_data, retained_rows=source_rows),
        source_row_parity=True, original_IPW_removed=True,
        sampling_rule='exact frozen F retained rows: ALL nonzero + deterministic 2% neutral',
        edge_identity='user_index + original b0_rank + exact cold_rank_times_warm_slot; canonical flat Cold-row*12+slot0',
        group_definition='one contiguous historical user-window; group.npy counts retained action edges, zero-row users excluded',
        seconds=time.perf_counter()-started, completed_at=now(), final_week='not_run')
    write_json(folder/'AUDIT.json', entry)
    del data,x,y,ui,weight,edge; gc.collect()
    return _merge_date_audit(folder.parent, entry)


def edge_percentiles(values, cold_user_index, deadline_epoch=None):
    """Average-tie percentile across all post-overlap edges of each user.

    High evidence has high percentile. Equal scores all receive 0.5 when the
    group has more than one edge, not arbitrarily different ordinal ranks.
    """
    values = np.asarray(values)
    ui = np.asarray(cold_user_index, np.int64)
    if values.ndim != 2 or values.shape != (len(ui),12) or not np.isfinite(values).all():
        raise ValueError('Expected finite canonical Cold-row by twelve-slot scores')
    group, _ = _group_arrays(ui)
    output = np.empty(values.shape, np.float32)
    start = 0
    for size in group:
        _guard(deadline_epoch)
        stop = start + int(size)
        block = values[start:stop].ravel()
        output[start:stop] = ((rankdata(block,method='average')-1)/(len(block)-1)).reshape(-1,12)
        start = stop
    return output


def _chunk_predict(function, fitted, features, deadline_epoch=None):
    out = np.empty(len(features), np.float64)
    # Explicitly exclude target even though saved preprocessors select names.
    features = features.drop(columns='target', errors='ignore')
    for start in range(0,len(features),16384):
        _guard(deadline_epoch)
        out[start:start+16384] = function(fitted,features.iloc[start:start+16384])
    if not np.isfinite(out).all():
        raise ValueError('Saved propensity model produced nonfinite output')
    return out


def attach_frozen_cold_rank(features, selection, preprocessing):
    """Saved normalized qC features: attach identity/rank only, never renormalize."""
    keys=['customer_id','article_id']
    required=set(preprocessing['numeric']+preprocessing['binary'])
    if not required.issubset(features.columns):
        raise ValueError('Saved qC normalized model features are incomplete')
    if features.duplicated(keys).any() or selection.duplicated(keys).any():
        raise ValueError('Cold feature/selection identity is not one-to-one')
    ranks=selection[keys+['b0_rank']]
    if (len(features)!=len(ranks) or
            set(map(tuple,features[keys].to_numpy()))!=set(map(tuple,ranks[keys].to_numpy()))):
        raise ValueError('Saved qC population differs from frozen full Cold50 selection')
    result=features.merge(ranks,on=keys,how='left',validate='one_to_one',sort=False)
    for _,group in result.groupby('customer_id',sort=False):
        if not np.array_equal(np.sort(group.b0_rank.to_numpy()),np.arange(1,51)):
            raise ValueError('Frozen Cold ranks are not exactly 1..50 per user')
    pd.testing.assert_frame_equal(result[features.columns],features.reset_index(drop=True))
    return result


def ensure_old_u(repo, root, t, deadline_epoch=None, data=None):
    """Replay original R3 qC-original/qW-R2, never substitute E-full qC.

    The input propensity features are calculated on complete Cold50/Top12;
    original in-place-replacement logit difference is ranked only after W0
    overlap exclusion. The same source matrix applies to all dated scorers.
    """
    from .p42_data import normalize_cold_confidence
    from .p42_propensity import predict_propensity
    from .p42r_propensity import predict_repaired_propensity
    from .p42_matching import clipped_logit
    _date_guard(t); _guard(deadline_epoch)
    repo,root = Path(repo).resolve(),Path(root).resolve()
    folder = root/'prepared'/t
    receipt = folder/'OLD_U_AUDIT.json'
    if receipt.exists():
        return read_json(receipt)
    folder.mkdir(parents=True,exist_ok=True)
    if data is None:
        data,_ = load_f_data(repo,t,deadline_epoch)
    if data['cutoff'] != t:
        raise ValueError('Wrong dated F data supplied to old-U adapter')
    line = source_for(t)
    inputs = []
    if not line['availability']:
        raw = np.zeros((len(data['cold']),12),np.float64)
        percent = np.zeros(raw.shape,np.float32)
    else:
        w = line['window']
        r3 = read_json(repo/'reports/phase4/P4_2R3_EXPERIMENT_CONTRACT.json')
        qc = r3['qC_reuse']['windows'][w]
        if qc['latest_training_label_end'] != line['training_label_end']:
            raise ValueError('R3 qC temporal lineage differs from source matrix')
        fitted_c = {}
        for key in ('model','preprocessing'):
            p = Path(qc[key]['path'])
            inputs.append(trusted_source(repo,p,'P4_2R3_OUTPUT_MANIFEST.json'))
            fitted_c[key] = read_json(p)
        fitted_w = {}
        for key in ('model','preprocessing'):
            p = repo/'artifacts/phase4'/R3_RUN/'models'/w/'qW-R2'/(key+'.json')
            inputs.append(trusted_source(repo,p,'P4_2R3_OUTPUT_MANIFEST.json'))
            fitted_w[key] = read_json(p)
        actual_line = fitted_w['model']['lineage']
        if actual_line['latest_training_label_end'] != line['training_label_end']:
            raise ValueError('R3 qW temporal lineage differs from source matrix')
        fc = read_json(repo/'reports/phase4/P4_2F_EXPERIMENT_CONTRACT.json')
        sources = paths_for(repo,fc,t)
        for key,manifest in [('cold','P4_2R_OUTPUT_MANIFEST.json' if t in DATES else 'P4_2_OUTPUT_MANIFEST.json'),
                             ('warm','P4_2_OUTPUT_MANIFEST.json')]:
            inputs.append(trusted_source(repo,sources[key],manifest))
        cold = frame(sources['cold'])
        if 'b0_score' in cold.columns:
            cold = normalize_cold_confidence(cold)
        else:
            # Original development qC-features contains the already-normalized
            # model inputs but omits raw scores/ranks. Do not invent scores or
            # re-normalize a post-overlap subset. Attach only frozen full50 rank.
            inputs.append(trusted_source(repo,sources['selection'],'P4_2_OUTPUT_MANIFEST.json'))
            cold = attach_frozen_cold_rank(cold,frame(sources['selection']),fitted_c['preprocessing'])
        cold = cold.sort_values(['customer_id','b0_rank','article_id'],ignore_index=True)
        warm = frame(sources['warm']).sort_values(['customer_id','warm_rank'],ignore_index=True)
        np.testing.assert_array_equal(warm[['customer_id','article_id','warm_rank']].to_numpy(),
                                      data['warm'][['customer_id','article_id','warm_rank']].to_numpy())
        if not cold.groupby('customer_id',sort=False).size().eq(50).all():
            raise ValueError('old-U prediction inputs must include complete original Cold50')
        qc_values = _chunk_predict(predict_propensity,fitted_c,cold,deadline_epoch)
        qw_values = _chunk_predict(predict_repaired_propensity,fitted_w,warm,deadline_epoch).reshape(-1,12)
        user_lookup = pd.Index(data['users'])
        full_ui = user_lookup.get_indexer(cold.customer_id)
        if (full_ui < 0).any():
            raise ValueError('Cold50 user absent from frozen W0 roster')
        full_raw = clipped_logit(qc_values)[:,None] - clipped_logit(qw_values)[full_ui]
        source_keys = pd.MultiIndex.from_frame(cold[['customer_id','article_id']])
        main_keys = pd.MultiIndex.from_frame(data['cold'][['customer_id','article_id']])
        take = source_keys.get_indexer(main_keys)
        if (take < 0).any() or len(set(take.tolist())) != len(take):
            raise ValueError('post-overlap Cold identity not found in original full50')
        raw = full_raw[take]
        percent = edge_percentiles(raw,data['cold'].user_index.to_numpy(),deadline_epoch)
        # Current-dev cached R3 scores have identical action identity. For other
        # historical dates these are new label-free predictions, not old rows.
        if t == line['score_source_cutoff']:
            prior = repo/'artifacts/phase4'/R3_RUN/'outer'/w/'pair-utility.float64.npy'
            inputs.append(trusted_source(repo,prior,'P4_2R3_OUTPUT_MANIFEST.json'))
            saved = np.load(prior,mmap_mode='r',allow_pickle=False)
            np.testing.assert_allclose(raw,saved,rtol=0,atol=1e-12)
            # Keep the exact historical floating-point source on existing dev
            # cutoffs; chunked matrix multiplication can differ by last bits.
            raw = np.asarray(saved)
            percent = edge_percentiles(raw,data['cold'].user_index.to_numpy(),deadline_epoch)
        del full_raw,cold,warm,qc_values,qw_values
    paths = dict(raw=_save_array(folder/'old_U.npy',raw,np.float64,deadline_epoch),
                 percentile=_save_array(folder/'old_U_percentile.npy',percent,np.float32,deadline_epoch))
    result = dict(status='completed',cutoff=t,lineage=line,paths=paths,
        source_identity=inputs,rows=len(raw),edges=int(raw.size),
        definition='clipped_logit(original R qC-original)-clipped_logit(R3 qW-R2); not E-full qC',
        normalization='full50/Top12 prediction first; overlap excluded; average-rank percentile among surviving same-user edges',
        missing='value0 availability0; never same-cutoff user-OOF backfill',final_week='not_run')
    write_json(receipt,result)
    return result


def neutral_hash_keep(t,user,item,slot,threshold=50):
    """Same F SHA256 integer rule; 50/10000 is the extra easy-neutral 0.5%."""
    key = json.dumps([t,str(user),str(item),int(slot)],ensure_ascii=False,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'big') % 10000 < threshold


def hard_conditions(data, cc, old_percentile, old_available):
    """Five preregistered OR terms; return per-edge masks before label join.

    qW within-user percentile increases with probability of keeping the Warm
    item. Vulnerability is its complement, not the original W0 score rank.
    """
    n = len(cc); idx = cc.user_index.to_numpy(np.int64)
    shape = (n,12)
    qc = (cc.qC_relative_available.to_numpy() == 1) & (cc.qC_within_user_percentile.to_numpy() >= .9)
    qw = data['warm'].qW_within_user_percentile.to_numpy().reshape(-1,12)[idx]
    qa = data['warm'].qW_relative_available.to_numpy().reshape(-1,12)[idx]
    return dict(
        b0_top10=np.broadcast_to(cc.b0_rank.to_numpy()[:,None] <= 10,shape),
        qC_top_decile=np.broadcast_to(qc[:,None],shape),
        warm_slots_9_12=np.broadcast_to(np.arange(1,13)[None,:] >= 9,shape),
        qW_vulnerable_top_quintile=(qa == 1) & np.isfinite(qw) & ((1-qw) >= .8),
        old_U_top_decile=np.full(shape,False) if not old_available else np.asarray(old_percentile) >= .9)


def _e_view(audit):
    """E retains nonzero and hard-neutral rows of A2, without copying X."""
    return dict(audit, requested_sampling='E', storage='A2', rows=audit['E_rows'],
        counts=audit['E_counts'], row_selector='(y != 0) | hard_neutral',
        selected_groups='recompute group counts after applying row_selector',
        hard_neutral=audit['paths']['hard_neutral'])


def _prepare_hard(repo,root,t,deadline_epoch=None,requested='A2'):
    from .p42d_stats import single_ap_delta
    folder = root/'prepared'/t/'A2'
    if (folder/'AUDIT.json').exists():
        out = _merge_date_audit(folder.parent,read_json(folder/'AUDIT.json'))
        return _e_view(out) if requested == 'E' else out
    started = time.perf_counter()
    data,source_data = load_f_data(repo,t,deadline_epoch)
    old = ensure_old_u(repo,root,t,deadline_epoch,data=data)
    old_percent = np.load(old['paths']['percentile'],mmap_mode='r')
    nc = len(data['cold'])
    # Two booleans per complete action edge are bounded (<14 MB per date).
    keep = np.zeros((nc,12),bool); hard_n = np.zeros_like(keep)
    counts = dict(B=0,N=0,H=0)
    condition_counts = {k:0 for k in ('b0_top10','qC_top_decile','warm_slots_9_12',
                                     'qW_vulnerable_top_quintile','old_U_top_decile')}
    for start in range(0,nc,4096):
        _guard(deadline_epoch)
        cc = data['cold'].iloc[start:start+4096]; n = len(cc)
        idx = cc.user_index.to_numpy(np.int64)
        y = single_ap_delta(data['relevance'][idx],cc.target.to_numpy(),data['truth_count'][idx])
        neutral = y == 0
        conditions = hard_conditions(data,cc,old_percent[start:start+n],old['lineage']['availability'])
        hard = np.logical_or.reduce(list(conditions.values()))
        for name,mask in conditions.items():
            condition_counts[name] += int(np.count_nonzero(mask & neutral))
        hh = hard & neutral
        selected = ~neutral | hh
        users,items = cc.customer_id.to_numpy(),cc.article_id.to_numpy()
        for i,j in zip(*np.nonzero(neutral & ~hard)):
            selected[i,j] = neutral_hash_keep(t,users[i],items[i],j+1)
        keep[start:start+n] = selected; hard_n[start:start+n] = hh
        part = _counts(y)
        for name in counts: counts[name] += part[name]
    prior = read_json(repo/'reports/phase4/p4_2g_data_parity.json')['cutoffs'][t]
    if counts != {k:prior[k] for k in counts}:
        raise ValueError('Complete hard-neutral reconstruction differs from frozen F action census')
    nrows = int(keep.sum()); nhard = int(hard_n.sum())
    ecounts = dict(B=counts['B'],N=nhard,H=counts['H'])
    expected = dict(B=counts['B'],N=nrows-counts['B']-counts['H'],H=counts['H'])
    folder.mkdir(parents=True,exist_ok=False)
    maps = {}; paths = {}
    for name,dtype,shape in [('X',np.float32,(nrows,len(GLOBAL))),('y',np.float64,(nrows,)),
            ('user_index',np.int32,(nrows,)),('edge_index',np.int64,(nrows,)),('hard_neutral',np.bool_,(nrows,))]:
        p = folder/(name+'.npy')
        maps[name] = np.lib.format.open_memmap(p,mode='w+',dtype=dtype,shape=shape)
        paths[name] = str(p.resolve())
    cursor=0; actual=dict(B=0,N=0,H=0)
    for start,cc,x,y in batches(data):
        _guard(deadline_epoch)
        selected = keep[start:start+len(cc)]; n=int(selected.sum()); end=cursor+n
        ui = np.broadcast_to(cc.user_index.to_numpy()[:,None],y.shape)
        edge = np.arange(start*12,(start+len(cc))*12).reshape(-1,12)
        maps['X'][cursor:end] = x[selected.ravel()]
        maps['y'][cursor:end] = y[selected]
        maps['user_index'][cursor:end] = ui[selected]
        maps['edge_index'][cursor:end] = edge[selected]
        maps['hard_neutral'][cursor:end] = hard_n[start:start+len(cc)][selected]
        for name,value in _counts(y[selected]).items(): actual[name] += value
        cursor=end
    if cursor != nrows or actual != expected:
        raise ValueError('Hard-pool count differs between counting and materialization passes')
    for mapped in maps.values(): mapped.flush()
    paths.update(_write_groups(folder,maps['user_index'],deadline_epoch))
    if int(np.count_nonzero((maps['y'] != 0) | maps['hard_neutral'])) != sum(ecounts.values()):
        raise ValueError('E hard-only view does not preserve every nonzero edge')
    entry=dict(status='completed',cutoff=t,sampling='A2',storage='A2',paths=paths,
        rows=nrows,groups=paths['groups'],counts=actual,full_counts=counts,full_edges=sum(counts.values()),
        E_rows=sum(ecounts.values()),E_counts=ecounts,E_row_selector='(y != 0) | hard_neutral',
        features=list(GLOBAL),columns=len(GLOBAL),training_weight=1,
        source_identity=dict(data=source_data),source_row_parity=True,
        old_u_lineage=old['lineage'],old_u_audit=str((folder.parent/'OLD_U_AUDIT.json').resolve()),
        hard_neutral_rows=nhard,hard_neutral_condition_counts=condition_counts,
        remaining_neutral_population=counts['N']-nhard,
        remaining_neutral_retained=actual['N']-nhard,
        sampling_rule='ALL B/H + ALL five-condition hard N + SHA integer<50 of remaining N',
        unchanged_F70=True,no_IPW=True,seconds=time.perf_counter()-started,
        completed_at=now(),final_week='not_run')
    write_json(folder/'AUDIT.json',entry)
    del maps,data,keep,hard_n,old_percent;gc.collect()
    out=_merge_date_audit(folder.parent,entry)
    return _e_view(out) if requested == 'E' else out


def ensure_meta(repo,root,t,deadline_epoch=None):
    """Materialize fourteen frozen rank/availability fields on ALL action edges.

    B0/M4/qC/qW ranks already belong to F70 and are not duplicated here.
    No historical OOF score, current-label diagnostic calibrator or new fit is
    permitted. Callers subset this matrix with retained ``edge_index.npy``.
    """
    from lightgbm import Booster
    from .p42h_core import calibrated
    _date_guard(t); _guard(deadline_epoch)
    repo,root=Path(repo).resolve(),Path(root).resolve()
    folder=root/'prepared'/t/'meta'
    if (folder/'AUDIT.json').exists():
        return read_json(folder/'AUDIT.json')
    data,source_data=load_f_data(repo,t,deadline_epoch)
    old=ensure_old_u(repo,root,t,deadline_epoch,data=data)
    line=source_for(t);shape=(len(data['cold']),12);assets=[];raw={}
    if line['availability']:
        w=line['window']
        if t==line['score_source_cutoff']:
            cache_specs={
                'F_G':(F_RUN,'G_expected_deltaAP-utility.npy','F'),
                'G_U_R':(G_RUN,'U_R.npy','G'),
                'H_q_raw':(H_RUN,'q_raw.npy','H'),
                'H_q_cal':(H_RUN,'q_cal.npy','H'),
                'H_G_BH':(H_RUN,'G_BH.npy','H'),
                'H_r':(H_RUN,'r.npy','H')}
            for name,(run,filename,stage) in cache_specs.items():
                _guard(deadline_epoch)
                p=repo/'artifacts/phase4'/run/'outer'/w/filename
                assets.append(trusted_source(repo,p,f'P4_2{stage}_OUTPUT_MANIFEST.json'))
                raw[name]=np.load(p,mmap_mode='r',allow_pickle=False)
                if raw[name].shape != shape:
                    raise ValueError('Cached meta-score action order/shape differs from frozen F data')
        else:
            specs={
                'F_G':(F_RUN,'G','F'),
                'G_classifier':(G_RUN,'classifier','G'),
                'm_B':(G_RUN,'benefit','G'),
                'm_H':(G_RUN,'harm','G'),
                'H_q':(H_RUN,'q_full','H'),
                'H_r':(H_RUN,'r','H')}
            models={}
            for name,(run,role,stage) in specs.items():
                p=repo/'artifacts/phase4'/run/'models'/w/role/'model.txt'
                assets.append(trusted_source(repo,p,f'P4_2{stage}_OUTPUT_MANIFEST.json'))
                models[name]=Booster(model_file=str(p))
                if models[name].feature_name() != list(GLOBAL):
                    raise ValueError('A frozen meta-teacher no longer has exactly the F70 feature order')
            cp=repo/'artifacts/phase4'/H_RUN/'models'/w/'calibrator'/'CALIBRATION.json'
            assets.append(trusted_source(repo,cp,'P4_2H_OUTPUT_MANIFEST.json'))
            cal=read_json(cp)
            if not cal['OOF_only'] or cal['b'] <= 0:
                raise ValueError('H formal historical OOF calibrator is invalid')
            raw={name:np.empty(shape,np.float64) for name in META_SCORES if name != 'old_U'}
            for start,cc,x,_ in batches(data,labels=False):
                _guard(deadline_epoch)
                n=len(cc);end=start+n
                predictions={name:model.predict(x,num_threads=4) for name,model in models.items()}
                mb=np.clip(predictions['m_B'],0,1);mh=np.clip(predictions['m_H'],0,1)
                p=predictions['G_classifier']
                if p.shape != (len(x),3):
                    raise ValueError('G classifier B/N/H output class shape changed')
                q=predictions['H_q'];qcal=calibrated(q,cal)
                block=dict(F_G=predictions['F_G'],G_U_R=p[:,0]*mb-p[:,2]*mh,
                    H_q_raw=q,H_q_cal=qcal,H_G_BH=qcal*mb-(1-qcal)*mh,H_r=predictions['H_r'])
                for name,value in block.items():
                    if not np.isfinite(value).all():
                        raise ValueError('Nonfinite strictly earlier-model meta prediction')
                    raw[name][start:end]=value.reshape(-1,12)
            del models
    folder.mkdir(parents=True,exist_ok=False)
    path=folder/'X.npy'
    matrix=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(shape[0]*12,len(META_COLUMNS)))
    matrix[:]=0
    for number,name in enumerate(META_SCORES):
        _guard(deadline_epoch)
        available=old['lineage']['availability'] if name=='old_U' else line['availability']
        if available:
            percent=(np.load(old['paths']['percentile'],mmap_mode='r') if name=='old_U'
                     else edge_percentiles(raw[name],data['cold'].user_index.to_numpy(),deadline_epoch))
            matrix[:,2*number]=percent.ravel();matrix[:,2*number+1]=1
    matrix.flush();del matrix
    audit=dict(status='completed',cutoff=t,paths=dict(X=str(path.resolve())),
        columns=list(META_COLUMNS),rows=shape[0]*12,shape=[shape[0]*12,len(META_COLUMNS)],
        lineage={name:(old['lineage'] if name=='old_U' else line) for name in META_SCORES},
        source_identity=dict(data=source_data,score_assets=assets),
        new_model_fits=0,prediction_cache_reused=t==line['score_source_cutoff'],
        normalization='post-overlap all same-user Cold-by-W0 edges; ascending average-tie (rank-1)/(n-1)',
        missing='value0 and available0; current-date user OOF forbidden',
        edge_order='frozen F Cold row then original Warm slot0..11',final_week='not_run')
    write_json(folder/'AUDIT.json',audit)
    return audit


def prepare_date(repo, root, t, deadline_epoch=None, sampling='A1'):
    """Prepare one independently resumable historical variant, never fit a model.

    Return ``{status,cutoff,sampling,paths:{X,y,user_index,edge_index,group,
    group_user_index},rows,groups,counts,full_counts,training_weight,audit_path}``.
    Request D/B1 as A1 and B2 as A2. E shares A2 and supplies a row selector.
    """
    _date_guard(t, historical=True); _guard(deadline_epoch)
    repo, root = Path(repo).resolve(), Path(root).resolve()
    if sampling in ('A1','B1','D'):
        return _prepare_a1(repo,root,t,deadline_epoch)
    if sampling in ('A2','B2','E'):
        return _prepare_hard(repo,root,t,deadline_epoch,requested=sampling)
    raise ValueError('sampling must be A1/B1/D or A2/B2/E')
