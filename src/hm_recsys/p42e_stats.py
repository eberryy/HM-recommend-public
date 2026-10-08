"""Frozen P4.2E probability/ranking and read-only threshold diagnostics."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .p42_propensity import clip_propensity, propensity_calibration
from .p42d_stats import discrimination, correlations, single_ap_delta
from .p42e_contract import FRACTIONS, BINS


def monotonic_parity(raw,cal):
    raw,cal=np.asarray(raw),np.asarray(cal)
    if not np.isfinite(raw).all() or not np.isfinite(cal).all(): raise ValueError('nonfinite probability')
    order=np.argsort(raw,kind='stable')
    np.testing.assert_array_equal(order,np.argsort(cal,kind='stable'),err_msg='calibration ordering changed')
    np.testing.assert_array_equal(raw[order][1:]==raw[order][:-1],cal[order][1:]==cal[order][:-1],
                                  err_msg='calibration changed tie partition')
    return {'rows':len(raw),'order_exact':True,'tie_partition_exact':True,'spearman':1.0,
            'proof':'identical stable total order and identical tie partitions imply identical average ranks'}


def calibrator(z,y):
    """Single unweighted 1D MLE; inputs are user-disjoint OOF logits only."""
    z,y=np.asarray(z),np.asarray(y)
    if not np.isfinite(z).all() or not 0<y.sum()<len(y): raise ValueError('invalid OOF labels/logits')
    def objective(theta):
        # Bound temporary memory; all rows contribute, no stochastic optimizer.
        loss=0.; grad=np.zeros(2)
        for start in range(0,len(y),250000):
            x=z[start:start+250000]; t=y[start:start+250000]
            v=theta[0]+theta[1]*x; r=expit(v)-t
            loss+=np.sum(np.logaddexp(0,v)-t*v)
            grad+=np.array([r.sum(),np.dot(r,x)])
        return loss/len(y),grad/len(y)
    r=minimize(objective,np.array([0.,1.]),jac=True,method='L-BFGS-B',
               options=dict(maxiter=200,ftol=1e-12,gtol=1e-8))
    out=dict(a=float(r.x[0]),b=float(r.x[1]),success=bool(r.success),iterations=int(r.nit),message=str(r.message),
             rows=len(y),positives=int(y.sum()),objective=float(r.fun),input='OOF unclipped raw logistic linear score',
             class_weight=None,sampling=False,regularization=None)
    if not r.success or not np.isfinite(r.x).all(): raise ValueError('OOF calibration solver failure: '+str(out))
    if r.x[1]<=0: raise ValueError('calibration_direction_failure: '+str(out))
    return out


def probability(y,q,diagnostic=True):
    if diagnostic:
        out=propensity_calibration(y,q)
    else:
        raw=np.asarray(q); q=clip_propensity(q); y=np.asarray(y)
        order=np.argsort(q,kind='stable'); bins=[]
        for idx in np.array_split(order,10):
            bins.append(dict(rows=len(idx),mean_probability=float(q[idx].mean()),observed_positive_rate=float(y[idx].mean())))
        out=dict(rows=len(y),positives=int(y.sum()),observed_positive_rate=float(y.mean()),
                 mean_predicted_probability=float(q.mean()),unclipped_mean_predicted_probability=float(raw.mean()),
                 predicted_to_observed_rate_ratio=float(q.mean()/y.mean()),brier_score=float(np.mean((q-y)**2)),
                 ece=float(sum(x['rows']*abs(x['mean_probability']-x['observed_positive_rate']) for x in bins)/len(y)),
                 calibration_bins=bins)
    p=clip_propensity(q); t=np.asarray(y)
    out['logloss']=float(-np.mean(t*np.log(p)+(1-t)*np.log1p(-p)))
    return out


def ranks(frame,q):
    f=frame[['customer_id','b0_rank','article_id']].copy(); f['q']=q
    r=f.sort_values(['customer_id','q','b0_rank','article_id'],ascending=[True,False,True,True]).groupby('customer_id',sort=False).cumcount()+1
    return r.reindex(frame.index).to_numpy()


def ranking(frame,q):
    r=ranks(frame,q); y=frame.target.to_numpy().astype(bool); positive=r[y]
    p=frame.loc[y,['customer_id']].copy(); p['rank']=positive
    first=p.groupby('customer_id')['rank'].min()
    return dict(**discrimination(q,y),positive_mean_rank=float(positive.mean()),positive_median_rank=float(np.median(positive)),
        positive_mean_reciprocal_rank=float(np.mean(1/positive)),conditional_user_mrr=float((1/first).mean()),
        positive_users=len(first),recall={str(k):float(np.mean(positive<=k)) for k in (1,5,10,20,50)}),r


def tails(frame,q,r):
    q=np.asarray(q); y=frame.target.to_numpy(); order=np.argsort(q,kind='stable'); n=len(q); base=y.mean()
    def describe(idx):
        p=int(y[idx].sum()); mean=float(q[idx].mean()) if len(idx) else None; observed=p/len(idx) if len(idx) else None
        return dict(rows=len(idx),positives=p,precision=observed,lift=observed/base if len(idx) else None,
            mean_predicted=mean,predicted_over_observed=mean/observed if observed else None,
            ratio_status='finite' if observed else 'infinite',
            strict_rows=int(frame.strict_cold_flag.iloc[idx].sum()),sparse_rows=int(frame.sparse1_5_flag.iloc[idx].sum()),
            mean_B0_rank=float(frame.b0_rank.iloc[idx].mean()) if len(idx) else None)
    return dict(global_top={str(f):describe(order[-math.ceil(n*f):]) for f in FRACTIONS},
        within_user={str(k):describe(np.flatnonzero(r<=k)) for k in (1,2,5,10)},
        bins=[dict(low=lo,high=hi,**describe(order[math.floor(n*lo):math.floor(n*hi)])) for lo,hi in zip(BINS[:-1],BINS[1:])])


def readiness(cold,warm,eligible,truth_count,q):
    idx=eligible.cold_row_index.to_numpy(int); ui=eligible.user_index.to_numpy(int)
    cq=clip_propensity(np.asarray(q)[idx]); wq=clip_propensity(warm.propensity.to_numpy()).reshape(-1,12)[ui]
    u=np.log(cq/(1-cq))[:,None]-np.log(wq/(1-wq))
    cy=eligible.target.to_numpy(np.int8); wy=warm.target.to_numpy(np.int8).reshape(-1,12)[ui]
    labels=cy[:,None]-wy; use=labels!=0
    ap=single_ap_delta(wy,cy,truth_count[ui])
    survival={}
    for name,mask in {'all':cy==1,'strict':(cy==1)&eligible.strict_cold_flag.eq(1).to_numpy(),
                      'sparse':(cy==1)&eligible.sparse1_5_flag.eq(1).to_numpy()}.items():
        survival[name]=dict(positives=int(mask.sum()),**{k:int(np.sum(mask&(u.max(axis=1)>v)))
                             for k,v in [('tau0',0.),('ln2',math.log(2)),('ln4',math.log(4))]})
    return dict(pair_discrimination=discrimination(u[use],labels[use]>0),
                ap_correlations=correlations(u,ap),survival=survival,pair_rows=int(u.size)),u


def verdicts(ws):
    a=list(ws.values())
    def avg(role,key,section='ranking'): return float(np.mean([w[section][role][key] for w in a]))
    roc=avg('Q_full_raw','roc_auc')>=avg('Q_old','roc_auc'); ap=avg('Q_full_raw','pr_auc')>=avg('Q_old','pr_auc')
    ng=sum(w['ranking']['Q_full_raw']['roc_auc']>=w['ranking']['Q_old']['roc_auc'] for w in a)
    rank='supported' if roc and ap and ng>=3 else 'rejected' if not roc and not ap and ng<=1 else 'mixed'
    def pooled(role,f):
        xs=[w['tails'][role]['global_top'][str(f)] for w in a]
        return sum(x['positives'] for x in xs)/sum(x['rows'] for x in xs)
    ok=[pooled('Q_full_raw',f)>=pooled('Q_old',f) for f in [.01,.005,.001]]
    ngtail=[sum(w['tails']['Q_full_raw']['global_top'][str(f)]['precision']>=w['tails']['Q_old']['global_top'][str(f)]['precision'] for w in a)>=3 for f in [.01,.005,.001]]
    tail='supported' if all(ok) and sum(ngtail)>=2 else 'rejected' if not any(ok) else 'mixed'
    ratio=sum(abs(math.log(w['probability']['Q_full_cal']['predicted_to_observed_rate_ratio']))<abs(math.log(w['probability']['Q_full_raw']['predicted_to_observed_rate_ratio'])) for w in a)
    ll=sum(w['probability']['Q_full_cal']['logloss']<=w['probability']['Q_full_raw']['logloss'] for w in a)
    br=sum(w['probability']['Q_full_cal']['brier_score']<=w['probability']['Q_full_raw']['brier_score'] for w in a)
    slope=sum(w['probability']['Q_full_cal']['calibration_slope'] is not None and w['probability']['Q_full_raw']['calibration_slope'] is not None and
        abs(w['probability']['Q_full_cal']['calibration_slope']-1)<abs(w['probability']['Q_full_raw']['calibration_slope']-1) for w in a)
    cal='supported' if min(ratio,ll,br,slope)>=3 else 'rejected' if avg('Q_full_cal','logloss','probability')>avg('Q_full_raw','logloss','probability') and avg('Q_full_cal','brier_score','probability')>avg('Q_full_raw','brier_score','probability') else 'mixed'
    new=sum(w['readiness']['Q_full_cal']['survival']['all']['tau0'] for w in a)
    old=sum(w['readiness']['Q_old']['survival']['all']['tau0'] for w in a)
    auc=lambda r: np.mean([w['readiness'][r]['pair_discrimination']['roc_auc'] for w in a])
    ready='rejected'
    if new>old and auc('Q_full_cal')>=auc('Q_old')-.01:
        ready='weak'
        if sum(w['readiness']['Q_full_cal']['survival']['all']['tau0']>w['readiness']['Q_old']['survival']['all']['tau0'] for w in a)>=3 and sum(w['readiness']['Q_full_cal']['survival']['strict']['tau0'] for w in a)>0: ready='supported'
    return dict(full_history_ranking_signal=rank,full_history_tail_signal=tail,oof_calibration_improves_probability_scale=cal,
        cross_source_threshold_readiness=ready,admission_rerun_allowed=rank!='rejected' and cal=='supported' and ready in ('weak','supported'),
        calibration_gate_counts=dict(ratio_improve=ratio,logloss_nonworse=ll,brier_nonworse=br,slope_improve=slope))
