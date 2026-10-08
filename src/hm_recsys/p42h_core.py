"""Conditional utilities, unweighted monotonic Platt, fixed descriptive verdicts."""
import warnings
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .p42h_contract import PLATT_PARAMS

def probabilities(q):
    q=np.asarray(q,dtype=np.float64)
    if not np.isfinite(q).all() or not ((q>=0)&(q<=1)).all(): raise ValueError('invalid probability')
    return q

def calibrated(q,cal):
    q=probabilities(q)
    if not np.isfinite([cal['a'],cal['b']]).all() or cal['b']<=0:
        raise ValueError('calibration_direction_failure')
    return expit(cal['a']+cal['b']*logit(np.clip(q,cal['epsilon'],1-cal['epsilon'])))

def utilities(q,r,mb,mh):
    q=probabilities(q);r=probabilities(r);mb=probabilities(mb);mh=probabilities(mh)
    if not (q.shape==r.shape==mb.shape==mh.shape): raise ValueError('shape mismatch')
    denom=mb+mh;threshold=np.divide(mh,denom,out=np.ones_like(mh),where=denom!=0)
    g=q*mb-(1-q)*mh;u=r*g;gate=g>0
    if not np.array_equal(gate,q>threshold): raise ValueError('gate_formula_parity_failure')
    if np.any(gate&((r<=0)|(u<=0))): raise ValueError('eligible_weight_nonpositive')
    if not np.isfinite(g).all() or not np.isfinite(u).all(): raise ValueError('nonfinite utility')
    return threshold,g,u

def fit_platt(q,y,require_positive=True):
    q=probabilities(q).ravel();y=np.asarray(y,dtype=np.int32).ravel()
    if len(q)!=len(y) or set(y.tolist())!={0,1}: raise ValueError('Platt needs both binary classes')
    x=logit(np.clip(q,1e-6,1-1e-6)).reshape(-1,1)
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=4):
        warnings.simplefilter('always',ConvergenceWarning)
        model=LogisticRegression(**PLATT_PARAMS).fit(x,y)
    failed=any(issubclass(w.category,ConvergenceWarning) for w in caught)
    a=float(model.intercept_[0]);b=float(model.coef_[0,0])
    result=dict(a=a,b=b,epsilon=1e-6,params=PLATT_PARAMS,rows=len(y),positives=int(y.sum()),
        observed_B_rate=float(y.mean()),n_iter=int(model.n_iter_[0]),converged=not failed,
        warnings=[str(w.message) for w in caught])
    if require_positive and (failed or b<=0 or not np.isfinite([a,b]).all()):
        raise ValueError('calibration_direction_failure' if b<=0 else 'calibration_convergence_failure')
    return result

def binary_calibration(q,y,diagnostic_fit=True):
    q=probabilities(q).ravel();y=np.asarray(y,dtype=np.int32).ravel()
    assert len(q)==len(y) and np.isin(y,[0,1]).all()
    n=len(y);b=np.minimum((q*10).astype(int),9);ece=0.;bins=[]
    for i in range(10):
        mask=b==i;rows=int(mask.sum());observed=float(y[mask].mean()) if rows else None;pred=float(q[mask].mean()) if rows else None
        if rows:ece+=rows/n*abs(observed-pred)
        bins.append(dict(low=i/10,high=(i+1)/10,rows=rows,positives=int(y[mask].sum()),observed_rate=observed,predicted_mean=pred))
    p=np.clip(q,np.finfo(float).eps,1-np.finfo(float).eps)
    result=dict(rows=n,positives=int(y.sum()),observed_rate=float(y.mean()) if n else None,
        predicted_mean=float(q.mean()) if n else None,Brier=float(np.square(q-y).mean()) if n else None,
        logloss=float(-(y*np.log(p)+(1-y)*np.log1p(-p)).mean()) if n else None,ECE=float(ece),reliability=bins)
    if diagnostic_fit:
        result['diagnostic_calibration']=fit_platt(q,y,require_positive=False)
        result['diagnostic_calibration']['audit_only_not_applied']=True
    return result

def verdict(windows,g_windows,f_reference):
    hh=[v['H_conditional_BH'] for v in windows.values()];gg=[g_windows[w]['R'] for w in windows];ff=[f_reference[w] for w in windows]
    def mean(rows,k): return float(np.mean([r['action']['beneficial_vs_harmful'][k] for r in rows]))
    auc=mean(hh,'roc_auc');pr=mean(hh,'pr_auc');ga=mean(gg,'roc_auc');gp=mean(gg,'pr_auc')
    non=sum(h['action']['beneficial_vs_harmful']['roc_auc']>=g['action']['beneficial_vs_harmful']['roc_auc'] for h,g in zip(hh,gg))
    a='supported' if auc>=ga and non>=3 and pr>=gp else 'mixed' if auc>ga or pr>gp else 'rejected'
    survive=sum(v['survival']['H']['surviving_positive_candidates'] for v in windows.values())
    strict=sum(v['survival']['H']['strict_surviving'] for v in windows.values())
    snon=sum(v['survival']['H']['surviving_positive_candidates']>g_windows[w]['survival']['R']['surviving_positive_candidates'] for w,v in windows.items())
    s='supported' if survive>=8 and snon>=3 and strict>=1 else 'mixed' if survive>=2 else 'rejected'
    ins=sum(r['admission']['inserted_cold_positives'] for r in hh);rem=sum(r['admission']['removed_warm_positives'] for r in hh)
    ratio=ins/rem if rem else float('inf') if ins else 0.
    p='supported' if ratio>=.25 and ins>=3 and rem<=20 else 'mixed' if ratio>=.15 and ratio>1/15 else 'rejected'
    d=np.array([r['delta_vs_w0'] for r in hh]);wd=np.array([r['segments']['warm_21_plus']['delta_vs_w0'] for r in hh])
    fd=np.array([r['delta_vs_w0'] for r in ff]);fw=np.array([r['segments']['warm_21_plus']['delta_vs_w0'] for r in ff])
    safety='supported' if d.mean()>=-.0005 and d.min()>=-.001 and wd.mean()>=-.0005 else 'mixed' if d.mean()>fd.mean() and d.min()>fd.min() and wd.mean()>fw.mean() else 'rejected'
    return dict(conditional_bh_action_signal=a,conditional_bh_survival=s,conditional_bh_policy_precision=p,conditional_bh_map_safety=safety,
        full_history_scaleup_allowed=a!='rejected' and s in ['supported','mixed'] and p in ['supported','mixed'] and safety=='supported',
        mean_auc=auc,G_R_mean_auc=ga,mean_PR=pr,G_R_mean_PR=gp,auc_nondegrade=non,
        surviving=survive,strict_surviving=strict,survival_improved_windows=snon,inserted=ins,removed=rem,
        ratio=ratio if rem else None,ratio_status='finite' if rem else 'infinite' if ins else 'undefined',
        mean_map=float(np.mean([h['map12'] for h in hh])),mean_delta=float(d.mean()),worst_delta=float(d.min()),mean_warm_delta=float(wd.mean()),selected='W0')
