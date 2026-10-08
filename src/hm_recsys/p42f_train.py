"""Frozen LightGBM G / user-cross-fitted H training, no outer tuning."""
from pathlib import Path
import time
import gc
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .p41a_contract import read_json,write_json,identity
from .p42f_contract import DATES,WINDOWS,GLOBAL,BASE,USER,PARAMS,earlier,now
from .p42f_core import batches,sample_edges,fold,prior_residual
from .p42f_data import save_frame,frame
from .p42f_features import load

def prepare(repo,c,root,guard):
    audits={}
    for t in sorted(set(DATES+list(WINDOWS.values()))):
        guard(); started=time.perf_counter(); data=load(repo,c,t,root)
        folder=root/'prepared'/t; folder.mkdir(parents=True,exist_ok=False)
        joblib.dump(data,folder/'data.joblib',compress=3)
        if t not in DATES: continue
        xs=[]; ys=[]; weights=[]; us=[]; total=np.zeros(3,np.int64); retained=np.zeros(3,np.int64)
        g=np.zeros(len(data['users'])); active=np.zeros(len(g),bool)
        for _,cc,x,y in batches(data):
            guard(); idx=cc.user_index.to_numpy(int); active[idx]=True
            np.maximum.at(g,idx,np.maximum(0,y.max(axis=1)))
            keep,weight=sample_edges(t,cc.customer_id.to_numpy(),cc.article_id.to_numpy(),y)
            yy=y[keep]; xs.append(x[keep.ravel()]); ys.append(yy); weights.append(weight)
            us.append(np.broadcast_to(idx[:,None],y.shape)[keep])
            total+=np.array([(y>0).sum(),(y<0).sum(),(y==0).sum()])
            retained+=np.array([(yy>0).sum(),(yy<0).sum(),(yy==0).sum()])
        x=np.concatenate(xs); y=np.concatenate(ys); weight=np.concatenate(weights); uidx=np.concatenate(us)
        folds=np.array([fold(u) for u in data['users']],np.uint8)
        np.savez(folder/'training.npz',X=x,y=y,weight=weight,user_index=uidx,fold=folds[uidx])
        meta=data['state'].copy(); meta['g']=g; meta['active']=active; meta['fold']=folds; meta['cutoff']=t
        save_frame(meta,folder/'user-windows.parquet')
        assert np.array_equal(total[:2],retained[:2]) and np.array_equal(weight,np.where(y==0,50.,1.))
        audits[t]=dict(full_edges=int(total.sum()),beneficial=int(total[0]),harmful=int(total[1]),neutral=int(total[2]),
                       retained_beneficial=int(retained[0]),retained_harmful=int(retained[1]),retained_neutral=int(retained[2]),
                       neutral_rate_observed=float(retained[2]/total[2]),probability_fixed=.02,neutral_weight=50,
                       historical_action_users=int(active.sum()),training_rows=len(y),seconds=time.perf_counter()-started)
        write_json(root/'ACTION_PREPARATION.json',audits)
        print(f'P4.2F prepared {t}: {len(y)} sampled edges, {total.tolist()} full beneficial/harmful/neutral',flush=True)
        del data,x,y,xs,ys,weights,us; gc.collect()
    return audits

def fit_model(x,y,weight,names,folder,guard):
    guard(); folder.mkdir(parents=True,exist_ok=False); start=time.perf_counter()
    write_json(folder/'FIT_START.json',dict(at=now(),params=PARAMS,rows=len(y),features=names))
    model=LGBMRegressor(**PARAMS)
    with threadpool_limits(limits=4): model.fit(x,y,sample_weight=weight,feature_name=names)
    pred=model.predict(x[:min(1000,len(x))])
    if not np.isfinite(pred).all(): raise ValueError('nonfinite fitted prediction')
    model.booster_.save_model(str(folder/'model.txt'))
    audit=dict(rows=len(y),features=len(names),seconds=time.perf_counter()-start,trees=model.booster_.num_trees(),params=PARAMS)
    write_json(folder/'FIT_RESULT.json',audit)
    return model,audit

def train_outer(repo,c,root,w,guard):
    t=WINDOWS[w]; dates=earlier(t); started=time.perf_counter(); folder=root/'models'/w
    if folder.exists(): raise FileExistsError('outer fit cannot be silently repeated')
    parts=[np.load(root/'prepared'/s/'training.npz') for s in dates]
    x=np.concatenate([p['X'] for p in parts]); y=np.concatenate([p['y'] for p in parts]); weight=np.concatenate([p['weight'] for p in parts]); folds=np.concatenate([p['fold'] for p in parts])
    for p in parts:p.close()
    del parts; gc.collect()
    G,ga=fit_model(x,y,weight,GLOBAL,folder/'G',guard)
    S,sa=fit_model(np.ascontiguousarray(x[:,:len(BASE)]),y,weight,BASE,folder/'H_pair',guard)
    oof=[]; fa=[]
    for k in [0,1]:
        mask=folds!=k
        model,a=fit_model(np.ascontiguousarray(x[mask,:len(BASE)]),y[mask],weight[mask],BASE,folder/f'H_oof_predict_fold{k}',guard)
        a.update(training_fold=1-k,prediction_fold=k); oof.append(model); fa.append(a)
    del x,y,weight,folds; gc.collect()
    metadata=[]
    for s in dates:
        guard(); data=joblib.load(root/'prepared'/s/'data.joblib'); meta=frame(root/'prepared'/s/'user-windows.parquet')
        maximum=np.zeros(len(meta))
        for _,cc,x,_ in batches(data,labels=False):
            idx=cc.user_index.to_numpy(int); f=meta.fold.to_numpy()[idx]
            for k in [0,1]:
                keep=f==k
                if not keep.any(): continue
                mask=np.repeat(keep,12); v=oof[k].predict(x[mask,:len(BASE)]).reshape(-1,12)
                if not np.isfinite(v).all(): raise ValueError('nonfinite OOF score')
                np.maximum.at(maximum,idx[keep],np.maximum(0,v.max(axis=1)))
        meta['s_oof']=maximum; meta['d']=maximum-meta.g
        metadata.append(meta.loc[meta.active].copy())
        del data; gc.collect()
    meta=pd.concat(metadata,ignore_index=True); a=meta[USER].to_numpy(float)
    med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in a.T])
    a=np.where(np.isfinite(a),a,med)
    with threadpool_limits(limits=4):
        scaler=StandardScaler().fit(a); ridge=Ridge(alpha=1.,solver='svd').fit(scaler.transform(a),meta.d.to_numpy())
    meta['tau_fixed']=ridge.predict(scaler.transform(a)); meta['residual']=meta.d-meta.tau_fixed
    # Historical prior diagnostics never include the current row or later dates.
    meta['prior_n']=0; meta['prior_b']=0.
    for s in dates:
        mask=meta.cutoff.eq(s); n,b=prior_residual(meta,meta.loc[mask,'customer_id'].to_numpy(),s)
        meta.loc[mask,'prior_n']=n; meta.loc[mask,'prior_b']=b
    save_frame(meta,folder/'threshold-training.parquet')
    threshold=dict(median=med.tolist(),mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                   coefficient=ridge.coef_.tolist(),intercept=float(ridge.intercept_),alpha=1.,shrink_kappa=2.,columns=USER)
    write_json(folder/'threshold.json',threshold)
    write_json(folder/'TRAINING.json',dict(cutoffs=dates,G=ga,H_pair=sa,H_oof=fa,threshold_rows=len(meta),seconds=time.perf_counter()-started,
               user_folds_disjoint=True,historical_prior_strict=True,outer_labels_used=False))
    return G,S,threshold,meta

def thresholds(threshold,history,state,t):
    a=state[USER].to_numpy(float); a=np.where(np.isfinite(a),a,np.array(threshold['median']))
    fixed=((a-threshold['mean'])/threshold['scale'])@np.array(threshold['coefficient'])+threshold['intercept']
    n,b=prior_residual(history,state.customer_id.to_numpy(),t)
    return fixed,n,b
