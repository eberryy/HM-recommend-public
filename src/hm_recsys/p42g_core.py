"""Pure risk separation and preregistered descriptive pilot verdicts."""
import numpy as np
from .p42d_stats import discrimination, correlations

def classes(y):
    y=np.asarray(y)
    if not np.isfinite(y).all(): raise ValueError('nonfinite label')
    return np.where(y>0,0,np.where(y<0,2,1)).astype(np.int32)

def utilities(p, mb_raw, mh_raw):
    p=np.asarray(p,float); mb_raw=np.asarray(mb_raw); mh_raw=np.asarray(mh_raw)
    if p.ndim!=2 or p.shape[1]!=3: raise ValueError('class order must be B,N,H')
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all(): raise ValueError('invalid probability')
    np.testing.assert_allclose(p.sum(axis=1),1.,rtol=0,atol=1e-12)
    if not np.isfinite(mb_raw).all() or not np.isfinite(mh_raw).all(): raise ValueError('invalid magnitude')
    mb=np.clip(mb_raw,0,1); mh=np.clip(mh_raw,0,1)
    return p[:,0]*mb-p[:,2]*mh,p[:,0]-p[:,2],mb,mh

def action(u,y):
    u=np.asarray(u).ravel(); y=np.asarray(y).ravel(); nz=y!=0
    return dict(beneficial_vs_harmful=discrimination(u[nz],y[nz]>0),
                beneficial_vs_all_nonbeneficial=discrimination(u,y>0),correlation=correlations(u,y))

def tail(u,y):
    e=np.asarray(u)>0; n=int(e.sum()); b=int(((y>0)&e).sum()); h=int(((y<0)&e).sum()); z=n-b-h
    return dict(eligible_edges=n,eligible_B=b,eligible_N=z,eligible_H=h,
                B_share=b/n if n else None,N_share=z/n if n else None,H_share=h/n if n else None,
                B_H_ratio=b/h if h else None,B_H_ratio_status='finite' if h else 'infinite' if b else 'undefined')

def survival(cold,u,y):
    positive=cold.target.to_numpy()==1; live=((u>0)&(y>0)).any(axis=1)&positive
    n=int(positive.sum())
    return dict(main_positive_candidates=n,surviving_positive_candidates=int(live.sum()),survival_rate=float(live.sum()/n) if n else None,
        strict_main=int((positive&cold.strict_cold_flag.eq(1).to_numpy()).sum()),
        sparse_main=int((positive&cold.sparse1_5_flag.eq(1).to_numpy()).sum()),
        strict_surviving=int((live&cold.strict_cold_flag.eq(1).to_numpy()).sum()),
        sparse_surviving=int((live&cold.sparse1_5_flag.eq(1).to_numpy()).sum()))

def calibration(p,y):
    y=classes(y); n=len(y); bins={}
    for k,name in [(0,'p_B'),(2,'p_H')]:
        b=np.minimum((p[:,k]*10).astype(int),9); rows=[]
        for i in range(10):
            m=b==i; count=int(m.sum())
            rows.append(dict(low=i/10,high=(i+1)/10,rows=count,positives=int((y[m]==k).sum()),
                observed_rate=float((y[m]==k).mean()) if count else None,predicted_mean=float(p[m,k].mean()) if count else None))
        bins[name]=rows
    return dict(edges=n,classes={name:dict(observed_rate=float((y==k).mean()),predicted_mean=float(p[:,k].mean()),
        observed_count=int((y==k).sum())) for k,name in enumerate(['B','N','H'])},
        multiclass_logloss=float(-np.log(np.maximum(p[np.arange(n),y],np.finfo(float).eps)).mean()),
        multiclass_Brier=float((np.square(p).sum(axis=1)-2*p[np.arange(n),y]+1).mean()),reliability=bins)

def verdict(windows,reference):
    rr=[x['R'] for x in windows.values()]; gg=[reference[w] for w in windows]
    def mean(rows,key):return float(np.mean([r['action']['beneficial_vs_harmful'][key] for r in rows]))
    auc=mean(rr,'roc_auc'); pr=mean(rr,'pr_auc'); ga=mean(gg,'roc_auc'); gp=mean(gg,'pr_auc')
    non=sum(r['action']['beneficial_vs_harmful']['roc_auc']>=g['action']['beneficial_vs_harmful']['roc_auc'] for r,g in zip(rr,gg))
    a='supported' if auc>ga and non>=3 and pr>=gp else 'mixed' if auc>ga or pr>gp else 'rejected'
    ins=sum(r['admission']['inserted_cold_positives'] for r in rr); rem=sum(r['admission']['removed_warm_positives'] for r in rr)
    ratio=ins/rem if rem else float('inf') if ins else 0.
    p='supported' if ratio>=.25 and ratio>=3*12/257 and rem<257 else 'mixed' if ratio>12/257 and rem<257 else 'rejected'
    d=np.array([r['delta_vs_w0'] for r in rr]); wd=np.array([r['segments']['warm_21_plus']['delta_vs_w0'] for r in rr])
    gd=np.array([r['delta_vs_w0'] for r in gg]); gw=np.array([r['segments']['warm_21_plus']['delta_vs_w0'] for r in gg])
    s='supported' if d.mean()>=-.0005 and d.min()>=-.001 and wd.mean()>=-.0005 else 'mixed' if d.mean()>gd.mean() and d.min()>gd.min() and wd.mean()>gw.mean() else 'rejected'
    return dict(risk_separation_action_signal=a,risk_separation_policy_precision=p,risk_separation_map_safety=s,
        full_history_scaleup_allowed=a!='rejected' and p=='supported' and s in ['supported','mixed'],
        inserted=ins,removed=rem,ratio=ratio if rem else None,ratio_status='finite' if rem else 'infinite' if ins else 'undefined',
        ratio_improvement=ratio/(12/257) if rem else None,mean_auc=auc,G_mean_auc=ga,auc_nondegrade=non,mean_PR=pr,G_mean_PR=gp,
        mean_map=float(np.mean([r['map12'] for r in rr])),mean_delta=float(d.mean()),worst_delta=float(d.min()),mean_warm_delta=float(wd.mean()),selected='W0')
