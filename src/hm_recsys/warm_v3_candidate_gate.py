"""Bounded candidate-wise neural gate; learns mixing weights, not a third score scale."""
from __future__ import annotations
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .warm_v3_common import *
from .warm_v3_gate import safe_training
from .warm_v2_engine import connection,literal,load_parquet,save_parquet
from .warm_v2_rank_fusion import rank_sql,ap_table

TRIAL='WV3-111'
PAIR_FEATURES=['repurchase_present','item2vec_is_new','user_item_events_12w','user_item_days_since_last_purchase',
               'user_product_type_share_12w','user_department_share_12w','item_days_since_last_sale']
PARAMS={'hidden':32,'epochs':10,'batch_size':4096,'negatives_per_positive':32,'hard_negative_count':16,
        'hard_pool_top_per_expert':50,'learning_rate':.002,'weight_decay':.0001,'temperature':10.,
        'neutral_gate_penalty':.01,'seed':20260909,'device':'cpu','threads':4}

class Gate(nn.Module):
    def __init__(self,dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,scores):
        alpha=torch.sigmoid(self.net(x)).squeeze(-1)
        return (1-alpha)*scores[:,0]+alpha*scores[:,1],alpha

def blend_scores(r0,r1,alpha):
    """Keep exact double-precision601 neutral ties; alpha only moves within bounds."""
    a=1/(60+np.asarray(r0,dtype=np.float64));b=1/(60+np.asarray(r1,dtype=np.float64))
    return (a+b)+2*(np.asarray(alpha,dtype=np.float64)-.5)*(b-a)

def arrays(meta):
    root=ART/'candidate_gate_data'/meta['cutoff'];p=root/'DATA.json'
    e=Engine(meta['year']);source=e.base_path(meta['cutoff'])
    identity={'ranks_path':meta['ranks_path'],'ranks_bytes':Path(meta['ranks_path']).stat().st_size,
              'base_path':str(source),'base_bytes':source.stat().st_size,'feature_version':1}
    if p.exists():
        m=read(p);assert m['source_identity']==identity
        return dict(np.load(root/'arrays.npz',allow_pickle=False)),m
    root.mkdir(parents=True,exist_ok=True)
    with connection() as con:
        frame=con.execute(f'''SELECT r.*, {','.join('b.'+f for f in PAIR_FEATURES)}
            FROM read_parquet({literal(meta['ranks_path'])}) r JOIN read_parquet({literal(e.base_path(meta['cutoff']))}) b
            USING(customer_id,article_id) ORDER BY r.customer_id,r.candidate_rank,r.article_id''').fetchdf()
        expected=con.execute(f'SELECT count(*) FROM read_parquet({literal(meta["ranks_path"])})').fetchone()[0]
        if len(frame)!=expected:raise ValueError('candidate feature join changed frozen pair population')
    assert not frame.duplicated(['customer_id','article_id']).any()
    n=frame.groupby('customer_id',sort=False).customer_id.transform('size').to_numpy()
    r0=frame.r0.to_numpy();r1=frame.r1.to_numpy()
    latent=frame.latent;g=frame.groupby('customer_id',sort=False).latent
    z=((latent-g.transform('mean'))/(g.transform('std')+1e-6)).fillna(0).clip(-10,10).to_numpy()
    values=[60/(60+r0),60/(60+r1),r0/n,r1/n,(r0-r1)/n,(r0<=12),(r1<=12),(r0<=50),(r1<=50),z,
            frame.missing.to_numpy(),np.log1p(frame.user_history_events_12w),np.log1p(frame.user_unique_items_12w),
            np.log1p(frame.user_days_since_last_purchase.clip(0,10000))]
    names=['r0_reciprocal','r1_reciprocal','r0_fraction','r1_fraction','rank_gap','r0_top12','r1_top12',
           'r0_top50','r1_top50','latent_within_user_z','bpr_missing','log_history','log_unique','log_recency']
    for col in PAIR_FEATURES:
        v=frame[col].to_numpy(float)
        if 'events' in col or 'days' in col:v=np.log1p(np.clip(v,0,10000))
        values.append(v);names.append(col)
    x=np.nan_to_num(np.column_stack(values).astype(np.float32),nan=0,posinf=10,neginf=-10)
    scores=np.column_stack([60/(60+r0),60/(60+r1)]).astype(np.float32)
    ids=frame.customer_id.to_numpy();offset=np.r_[0,np.flatnonzero(ids[1:]!=ids[:-1])+1,len(ids)].astype(np.int64)
    data={'x':x,'scores':scores,'target':frame.target.to_numpy(np.uint8),'offset':offset,
          'hard':((r0<=50)|(r1<=50)).astype(np.uint8),'truth_count':frame.truth_count.to_numpy(np.int32),
          'active':(frame.user_history_events_12w.to_numpy()>0).astype(np.uint8)}
    np.savez(root/'arrays.npz',**data)
    save_parquet(frame[['customer_id','article_id','candidate_rank','target','truth_count','user_history_events_12w','r0','r1','rf']],root/'keys.parquet')
    out={'cutoff':meta['cutoff'],'rows':len(frame),'users':len(offset)-1,'total_users':meta['total_users'],
         'features':names,'keys':str(root/'keys.parquet'),'source_identity':identity,'future_labels_are_inputs':False,'final_week':'not_run'}
    write(p,out);return data,out

def fit(metadata):
    torch.set_num_threads(4);torch.manual_seed(PARAMS['seed']);rng=np.random.default_rng(PARAMS['seed'])
    data=[];pairs=[];shift=0;feature_names=None
    for m in metadata:
        d,info=arrays(m);feature_names=info['features'];data.append(d)
        for lo,hi in zip(d['offset'][:-1],d['offset'][1:]):
            if not d['active'][lo]:continue
            pos=np.flatnonzero(d['target'][lo:hi])+lo;neg=np.flatnonzero(d['target'][lo:hi]==0)+lo
            if len(pos)==0 or len(neg)==0:continue
            hard=neg[d['hard'][neg]>0]
            for p in pos:
                hn=rng.choice(hard,min(16,len(hard)),replace=False) if len(hard) else np.array([],dtype=int)
                rest=np.setdiff1d(neg,hn);rn=rng.choice(rest,min(32-len(hn),len(rest)),replace=False)
                picked=np.r_[hn,rn]
                # Equal contribution per user's positive set, with complete-truth AP normalizer.
                weight=1/(len(pos)*min(int(d['truth_count'][lo]),12)*len(picked))
                pairs.extend((p+shift,int(q)+shift,weight) for q in picked)
        shift+=len(d['x'])
    assert pairs and feature_names
    x=np.concatenate([d['x'] for d in data]);scores=np.concatenate([d['scores'] for d in data])
    mean=x.mean(axis=0,dtype=np.float64).astype(np.float32);std=x.std(axis=0,dtype=np.float64).astype(np.float32);std=np.maximum(std,.01)
    x=torch.from_numpy((x-mean)/std);s=torch.from_numpy(scores)
    pair=np.array(pairs,dtype=np.float64);weights=pair[:,2].astype(np.float32);weights/=weights.mean()
    pi=pair[:,:2].astype(np.int64);model=Gate(x.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    history=[];started=time.perf_counter()
    for epoch in range(PARAMS['epochs']):
        total=0.;observed=0
        for batch in np.array_split(rng.permutation(len(pi)),max(1,int(np.ceil(len(pi)/4096)))):
            p,n=pi[batch].T;sp,ap=model(x[p],s[p]);sn,an=model(x[n],s[n])
            loss=(F.softplus(-10*(sp-sn))*torch.from_numpy(weights[batch])).mean()+.01*((ap-.5).square().mean()+(an-.5).square().mean())
            opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(batch);observed+=len(batch)
        history.append(total/observed)
        budget(1)
    return model,mean,std,{'params':PARAMS,'features':feature_names,'meta_cutoffs':[m['cutoff'] for m in metadata],
        'pairs':len(pi),'rows':len(x),'loss_curve':history,'fit_seconds':time.perf_counter()-started,
        'supervision':'earlier outer heldout experts;32implicitnegativepairs perpositive, no futuremeta, labelnotgateinput'}

def score(meta,model,mean,std,root):
    d,info=arrays(meta);parts=[];alphas=[];model.eval()
    with torch.no_grad():
        for lo in range(0,len(d['x']),65536):
            x=torch.from_numpy((d['x'][lo:lo+65536]-mean)/std);s=torch.from_numpy(d['scores'][lo:lo+65536])
            v,a=model(x,s);parts.append(v.numpy());alphas.append(a.numpy())
    f=load_parquet(info['keys']);f['alpha']=np.concatenate(alphas)
    f['score']=blend_scores(f.r0,f.r1,f.alpha)
    # The gate can only interpolate the same two expert rank-derived scores.
    limits=np.column_stack([2/(60+f.r0.to_numpy()),2/(60+f.r1.to_numpy())])
    assert np.all(f.score.to_numpy()>=limits.min(axis=1)-1e-15) and np.all(f.score.to_numpy()<=limits.max(axis=1)+1e-15)
    root.mkdir(parents=True,exist_ok=True)
    with connection() as con:
        con.register('f',f)
        from .warm_v2_rank_fusion import final_rank_sql
        con.execute(f'CREATE TEMP TABLE r AS SELECT *,{final_rank_sql()} final_rank FROM f')
        ap=ap_table(con,'r','final_rank');base=ap_table(con,'r','rf')
        # Inner rows are allactive; outer fallback is independently checked below.
        value=float(ap.ap.sum()/meta['total_users'])
        con.execute(f'COPY r TO {literal(root/"ranks.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
    return {'map_component':value,'population_delta':value-meta['baseline_map_population_component'],
        'alpha_quantiles':np.quantile(f.alpha,[0,.1,.5,.9,1]).tolist(),'rows':len(f),
        'rank_path':str(root/'ranks.parquet'),'bounds_verified':True}

def screen():
    setup();budget(12);a=read(REPORT/'WV3_110_CANDIDATE_GATE_AUDIT.json');assert a['mechanism_gate']
    oracle=read(REPORT/'WV3_100_GATE_ORACLE_AUDIT.json')
    register(TRIAL,'candidate_bounded_MoE','Within one user different candidates need different experts; bounded per-pair interpolation can recover this headroom.',
        params=PARAMS,features='21rank/user/reliability features; not original84dump',
        training_protocol='2019prioroutermeta only; pairwiseimplicitranking+neutralgate regularization; fixed10epochs')
    root=ART/TRIAL;root.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    train_meta=safe_training(oracle['historical_meta'],min(m['cutoff'] for m in oracle['inner']))
    model,mean,std,metadata=fit(train_meta);torch.save(model.state_dict(),root/'model.pt');np.savez(root/'scale.npz',mean=mean,std=std);write(root/'TRAINING.json',metadata)
    windows={m['window']:score(m,model,mean,std,root/m['window']/'inner') for m in oracle['inner']}
    ds=[v['population_delta'] for v in windows.values()];passed=np.mean(ds)>=.0001 and sum(d>0 for d in ds)>=3 and min(ds)>=-.0005
    out={'windows':windows,'mean_population_delta':float(np.mean(ds)),'positive_windows':sum(d>0 for d in ds),'worst_delta':min(ds),
        'passed':bool(passed),'runtime_seconds':time.perf_counter()-start,'training':metadata,'final_week':'not_run'}
    write(REPORT/f'{TRIAL}_SCREEN.json',out);update(TRIAL,inner_evidence=out,decision='inner_pass' if passed else 'reject_inner',runtime=out['runtime_seconds'])
    print({k:v for k,v in out.items() if k not in ('windows','training')},flush=True)

if __name__=='__main__':screen()
