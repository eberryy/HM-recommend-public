"""P4.3A frozen heuristic arm F: label-free blending, then exact-list audit.

No model fit, hyperparameter generation or evaluation is run on import. This
module only implements the predeclared 128 configurations passed by the caller.
"""
from __future__ import annotations

from pathlib import Path
import gc
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

from .p41a_contract import read_json,write_json
from .p42f_contract import WINDOWS,now
from .p42_matching import apply_admissions
from .p43a_data import load_f_data,ensure_meta,META_COLUMNS
from .p43a_contract import F_FEATURES,GLOBAL_PERCENTILES
from .p43a_policy import check_deadline,global_midrank_thresholds,contexts,exact_ap,SEGMENTS


def at_most_k_matching(weights,eligible,cap):
    """Exact positive-weight matching with a hard cap and explicit forbidden edges.

    Add K zero-weight virtual Cold nodes, and require exactly K augmented
    matches. Virtual nodes represent keeping the Warm recommendation unchanged.
    Every original matching with <=K edges has such a completion, so one LAP
    solve equals taking the best solution over original exact cardinalities.
    Canonical real Cold, virtual Cold and filler rows precede real Warm and
    dummy columns. No epsilon or label-dependent tie-breaking is introduced.
    """
    w=np.asarray(weights,np.float64);allowed=np.asarray(eligible,bool)
    if w.ndim!=2 or allowed.shape!=w.shape or not np.isfinite(w).all():
        raise ValueError('finite aligned weight and eligibility matrices required')
    if not isinstance(cap,(int,np.integer)) or cap<0:
        raise ValueError('admission cap must be a nonnegative integer')
    nc,nw=w.shape
    if nc>50 or nw>12:
        raise ValueError('frozen Cold50 by WarmTop12 action-space limit exceeded')
    if (w[allowed]<=0).any():
        raise ValueError('eligible blend edges must already have strictly positive weights')
    k=min(int(cap),nc,nw)
    if not k or not allowed.any():
        return []
    # nc+k augmented Cold rows, nw real Warm columns. The standard exactly-K
    # augmentation has (nc+k)+nw-k == nc+nw rows and columns.
    n_aug_cold=nc+k;size=nc+nw
    costs=np.zeros((size,size),np.float64)
    costs[:nc,:nw]=np.where(allowed,-w,np.inf)
    costs[n_aug_cold:,nw:]=np.inf
    rows,cols=linear_sum_assignment(costs)
    result=sorted([(int(r),int(c)) for r,c in zip(rows,cols) if r<nc and c<nw],
                  key=lambda pair:(pair[1],pair[0]))
    if len(result)>k or len({a for a,b in result})!=len(result) or len({b for a,b in result})!=len(result):
        raise AssertionError('cardinality or uniqueness violation')
    if any(not allowed[a,b] for a,b in result):
        raise AssertionError('matching used a prohibited edge')
    return result


def cross_user_percentile(values):
    """One row per frozen W0 user; finite users define ranks, missing maps to0."""
    x=np.asarray(values,np.float64)
    if x.ndim!=1:
        raise ValueError('one-dimensional unique-user values required')
    good=np.isfinite(x);out=np.zeros(len(x),np.float32);n=int(good.sum())
    if n:
        out[good]=(rankdata(x[good],method='average')-1)/(n-1) if n>1 else .5
    return out,dict(total_users=len(x),available_users=n,missing_users=int((~good).sum()),
        rank_denominator=n-1 if n>1 else None,singleton_value=.5,
        missing_rule='no finite historical value => transformed value0; not a fabricated raw0 observation')


def _edge_percentile_masked(values,availability,cold_ui,deadline_epoch=None):
    values=np.asarray(values,np.float64);available=np.asarray(availability,bool)
    if values.shape!=available.shape or values.ndim!=2 or values.shape[1]!=12:
        raise ValueError('aligned action score and availability matrices required')
    ui=np.asarray(cold_ui,np.int64);out=np.zeros(values.shape,np.float32)
    starts=np.r_[0,np.flatnonzero(np.diff(ui))+1,len(ui)] if len(ui) else np.array([0])
    for begin,end in zip(starts[:-1],starts[1:]):
        check_deadline(deadline_epoch)
        valid=available[begin:end].ravel();block=values[begin:end].ravel()
        if valid.any() and not valid.all():
            raise ValueError('partial within-user source availability needs an explicit contract, not silent normalization')
        if not valid.any():
            continue
        if not np.isfinite(block).all():
            raise ValueError('an available source contains nonfinite scores')
        out[begin:end]=((rankdata(block,method='average')-1)/(len(block)-1)).reshape(-1,12)
    return out


def blend_feature_matrix(data,meta,output_path=None,deadline_epoch=None):
    """Build ten registered features without accessing truth, targets or labels."""
    cold,warm,state=data['cold'],data['warm'],data['state']
    ui=cold.user_index.to_numpy(np.int64);n=len(cold);shape=(n,12)
    meta=np.asarray(meta)
    if meta.shape!=(n*12,len(META_COLUMNS)):
        raise ValueError('F requires the complete canonical fourteen-column meta matrix')
    if not np.array_equal(state.customer_id.to_numpy(),data['users']):
        raise ValueError('user state is not aligned with the full frozen W0 roster')
    dims=(n*12,len(F_FEATURES))
    matrix=(np.lib.format.open_memmap(output_path,mode='w+',dtype=np.float32,shape=dims)
            if output_path is not None else np.empty(dims,np.float32))
    scalars={
        'B0':(np.broadcast_to(cold.b0_user_percentile.to_numpy()[:,None],shape),
              np.broadcast_to(np.isfinite(cold.b0_user_percentile.to_numpy())[:,None],shape)),
        'qC':(np.broadcast_to(cold.qC_within_user_percentile.to_numpy()[:,None],shape),
              np.broadcast_to((cold.qC_relative_available.to_numpy()==1)[:,None],shape)),
        'warm_vulnerability':(1-warm.qW_within_user_percentile.to_numpy().reshape(-1,12)[ui],
                              warm.qW_relative_available.to_numpy().reshape(-1,12)[ui]==1)}
    source_mapping={'old_U':'old_U','F_G':'F_G','G_UR':'G_U_R','H_q_raw':'H_q_raw','H_q_cal':'H_q_cal'}
    for name,source in source_mapping.items():
        scalars[name]=(meta[:,META_COLUMNS.index(source+'_percentile')].reshape(shape),
                       meta[:,META_COLUMNS.index(source+'_available')].reshape(shape)==1)
    audit={}
    for name,(value,availability) in scalars.items():
        transformed=_edge_percentile_masked(value,availability,ui,deadline_epoch)
        matrix[:,F_FEATURES.index(name)]=transformed.ravel()
        audit[name]=dict(available_action_edges=int(np.count_nonzero(availability)),
                        action_edges=n*12,normalization='same-user post-overlap action-edge average-tie percentile')
    for name,column in [('novelty','novel_purchase_share_0_5'),('richness','user_past_purchase_count')]:
        transformed,detail=cross_user_percentile(state[column].to_numpy())
        matrix[:,F_FEATURES.index(name)]=np.repeat(transformed[ui],12)
        audit[name]=detail
    if not np.isfinite(matrix).all() or ((matrix<0)|(matrix>1)).any():
        raise ValueError('blending feature is outside finite [0,1]')
    if hasattr(matrix,'flush'):
        matrix.flush()
    return matrix,audit


def ensure_f_features(repo,root,t,deadline_epoch=None):
    root=Path(root).resolve();folder=root/'prepared'/t/'F_features'
    if (folder/'AUDIT.json').exists():
        return read_json(folder/'AUDIT.json')
    check_deadline(deadline_epoch)
    data,source=load_f_data(repo,t,deadline_epoch)
    meta=ensure_meta(repo,root,t,deadline_epoch)
    folder.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();path=folder/'X.npy'
    values,audit=blend_feature_matrix(data,np.load(meta['paths']['X'],mmap_mode='r'),path,deadline_epoch)
    result=dict(status='completed',cutoff=t,path=str(path.resolve()),rows=len(values),columns=list(F_FEATURES),
        source_identity=source,meta_receipt=str((root/'prepared'/t/'meta'/'AUDIT.json').resolve()),
        normalization_audit=audit,labels_used=False,model_fits=0,seconds=time.perf_counter()-start,
        user_population='all frozen W0 users, one row per user before novelty/richness broadcast',final_week='not_run')
    write_json(folder/'AUDIT.json',result)
    del values,data;gc.collect()
    return result


def _validate_f_config(config):
    if config.get('arm')!='F' or set(config['weights'])!=set(F_FEATURES):
        raise ValueError('registered F configuration and exact ten source names required')
    weights=np.array([config['weights'][k] for k in F_FEATURES],np.float64)
    if not np.isfinite(weights).all() or (weights<=0).any() or not np.isclose(weights.sum(),1.,rtol=0,atol=1e-12):
        raise ValueError('invalid frozen Dirichlet weights')
    if config['candidate_topK'] not in (5,10,20,50) or config['replaceable_slot_floor'] not in (1,7,10,12):
        raise ValueError('candidate/slot policy outside F contract')
    if config['max_admissions'] not in (1,2,3,12) or config['score_percentile_gate'] not in GLOBAL_PERCENTILES:
        raise ValueError('admission/percentile policy outside F contract')
    return weights


def evaluate_f(data,features,config,output_dir,deadline_epoch=None):
    """One frozen F config. Select and persist ALL matches before reading truth."""
    if data['cutoff'] not in WINDOWS.values():
        raise ValueError('F formal evaluation is limited to the four development cutoffs')
    check_deadline(deadline_epoch)
    weights=_validate_f_config(config)
    if np.shape(features)!=(len(data['cold'])*12,len(F_FEATURES)):
        raise ValueError('F feature matrix does not match frozen action population')
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    metadata=dict(config=config,cutoff=data['cutoff'],feature_columns=list(F_FEATURES),
        users=len(data['users']),cold_rows=len(data['cold']),final_week='not_run',
        matching='exact <=K positive blend via K zero-weight virtual Cold nodes; canonical SciPy assignment',
        gate='complete-window average-tie empirical percentile then original B0 budget and Warm slot floor')
    if (folder/'POLICIES.json').exists():
        if read_json(folder/'POLICIES.json')!=metadata:
            raise ValueError('F resume configuration or population differs from prior receipt')
    else:
        write_json(folder/'POLICIES.json',metadata)
    if (folder/'EVALUATION.json').exists():
        return read_json(folder/'EVALUATION.json')['rows']
    start=time.perf_counter();n=len(data['users'])
    scores=np.empty((len(data['cold']),12),np.float64)
    for lo in range(0,len(features),65536):
        check_deadline(deadline_epoch)
        scores.ravel()[lo:lo+65536]=np.asarray(features[lo:lo+65536],np.float64)@weights
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite frozen weighted blend')
    # The immutable F contract stores fractions (.90 ... .999), while the
    # shared empirical-percentile helper accepts percentages (90 ... 99.9).
    # Convert only at this API boundary; retain the original config unchanged.
    helper_percentile=100.*float(config['score_percentile_gate'])
    threshold=global_midrank_thresholds(scores,[helper_percentile])[helper_percentile]
    resumed=(folder/'SELECTION_PROGRESS.json').exists()
    progress=read_json(folder/'SELECTION_PROGRESS.json') if resumed else dict(next_user=0,seconds=0.)
    chosen=np.lib.format.open_memmap(folder/'selected-pairs.npy',mode='r+' if resumed else 'w+',
        dtype=np.int32,shape=(n,12,2))
    if not resumed:chosen[:]=-1
    groups=data['cold'].groupby('user_index',sort=False).indices
    next_user=int(progress['next_user'])
    def checkpoint_selection():
        chosen.flush()
        write_json(folder/'SELECTION_PROGRESS.json',dict(next_user=next_user,
            seconds=progress['seconds']+time.perf_counter()-start,labels_read=False,
            status='completed' if next_user==n else 'partial'))
    try:
        for ui in range(next_user,n):
            check_deadline(deadline_epoch)
            ids=np.asarray(groups.get(ui,[]),np.int64)
            if len(ids) and threshold is not None:
                local=scores[ids]
                allowed=(local>=threshold)&(data['cold'].b0_rank.to_numpy()[ids,None]<=config['candidate_topK'])
                allowed&=np.arange(1,13)[None,:]>=config['replaceable_slot_floor']
                matches=at_most_k_matching(local,allowed,config['max_admissions'])
                chosen[ui]=-1
                if matches:
                    chosen[ui,:len(matches)]=[(ids[ci],slot) for ci,slot in matches]
            next_user=ui+1
            if next_user%256==0:checkpoint_selection()
    finally:
        checkpoint_selection()
    # Only now inspect targets: every user decision is already immutable on disk.
    selection_seconds=progress['seconds']+time.perf_counter()-start
    write_json(folder/'SELECTION_COMPLETED_BEFORE_TRUTH.json',dict(at=now(),users=n,
        global_score_threshold=threshold,selected_pairs=str((folder/'selected-pairs.npy').resolve()),labels_read=False))
    truths,valid,base=contexts(data)
    denominators=valid.sum(axis=0)
    baseline=np.divide(base.sum(axis=0),denominators,out=np.zeros(5),where=denominators!=0)
    sums=np.zeros(5);counts=np.zeros(4,np.int64)
    aps=np.lib.format.open_memmap(folder/'user-exact-ap.npy',mode='w+',dtype=np.float64,shape=(n,5))
    for ui in range(n):
        check_deadline(deadline_epoch)
        items=list(data['warm_lists'][ui]);selected=chosen[ui];selected=selected[selected[:,0]>=0]
        cold_ids=selected[:,0];slots=selected[:,1]
        if len(set(cold_ids.tolist()))!=len(selected) or len(set(slots.tolist()))!=len(selected):
            raise ValueError('persisted F selections violate one-to-one matching')
        articles=data['cold'].article_id.iloc[cold_ids].tolist()
        if any(item in items for item in articles):
            raise ValueError('Cold selection overlaps frozen Warm Top12')
        ins=sum(item in truths[ui][0] for item in articles)
        rem=sum(items[slot] in truths[ui][0] for slot in slots)
        for item,slot in zip(articles,slots):items[slot]=item
        ap=np.array([exact_ap(items,truth) for truth in truths[ui]])
        aps[ui]=ap;sums+=ap-base[ui]
        counts+=np.array([ins,rem,bool(len(selected)),len(selected)],np.int64)
    aps.flush()
    delta=np.divide(sums,denominators,out=np.zeros(5),where=denominators!=0);means=baseline+delta
    segments={name:dict(map=float(means[j]) if denominators[j] else None,
        delta=float(delta[j]) if denominators[j] else None,map12=float(means[j]) if denominators[j] else None,
        delta_vs_w0=float(delta[j]) if denominators[j] else None,truth_users=int(denominators[j]))
        for j,name in enumerate(SEGMENTS) if j}
    elapsed=progress['seconds']+time.perf_counter()-start
    row=dict(policy_id=config['id'],policy=config,overall_map=float(means[0]),delta_map=float(delta[0]),
        segments=segments,inserted_positives=int(counts[0]),removed_positives=int(counts[1]),
        admitted_users=int(counts[2]),coverage=float(counts[2]/n) if n else 0.,
        replacements=int(counts[3]),users=n,global_score_threshold=threshold,
        global_score_threshold_status='finite' if threshold is not None else 'no_score_reaches_percentile',
        seconds=elapsed,selection_seconds=selection_seconds,status='completed',exact_final_list_AP=True,
        saved_decisions=str((folder/'selected-pairs.npy').resolve()),saved_user_ap=str((folder/'user-exact-ap.npy').resolve()),
        all_matches_fixed_before_truth_access=True,trainable=False)
    write_json(folder/'EVALUATION.json',dict(status='completed',cutoff=data['cutoff'],rows=[row],
        seconds=elapsed,completed_users=n,final_week='not_run'))
    return [row]


def evaluate_f_config(repo,root,window,config,deadline_epoch=None):
    """Runner entry point; caller owns immutable config selection and ledger."""
    if window not in WINDOWS:
        raise ValueError('not a preregistered development window')
    t=WINDOWS[window]
    saved=ensure_f_features(repo,root,t,deadline_epoch)
    data,_=load_f_data(repo,t,deadline_epoch)
    features=np.load(saved['path'],mmap_mode='r')
    return evaluate_f(data,features,config,Path(root)/'evaluation'/config['id']/window/'policy',deadline_epoch)
