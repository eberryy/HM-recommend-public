"""INNER-only diagnostic of compressed query vs candidate-conditioned basket memory."""
from __future__ import annotations
import gc
import time
import numpy as np
import pandas as pd
from . import warm_v3_common as c
from . import warm_v3_sequence as seq
from .warm_v2_engine import load_parquet, save_parquet

TRIAL='WV3-310'
PARAMS={'max_baskets':8,'attention_temperature':.1,'half_life_days':28,
        'heterogeneous_basket_mean_cosine_below':.7,'minimum_multi_basket_user_share':.25,
        'mean_within_user_auc_improvement_min':.01,'positive_windows_min':3}


def memory_scores(vectors, baskets, lengths, age):
    """Per-candidate attention has no learned parameters or future inputs."""
    sim=np.asarray(vectors,dtype=np.float32) @ np.asarray(baskets,dtype=np.float32).T
    sim=sim[:,:int(lengths)]
    maximum=sim.max(axis=1)
    logits=sim/.1-np.log(2)*np.asarray(age[:int(lengths)])[None,:]/28
    logits-=logits.max(axis=1,keepdims=True)
    weights=np.exp(logits);weights/=weights.sum(axis=1,keepdims=True)
    return maximum,(weights*sim).sum(axis=1)


def user_auc(labels,scores):
    labels=np.asarray(labels);scores=np.asarray(scores)
    ok=np.isfinite(scores);pos=scores[ok&(labels==1)];neg=scores[ok&(labels==0)]
    if len(pos)==0 or len(neg)==0:return np.nan
    d=pos[:,None]-neg[None,:]
    return float(((d>0)+.5*(d==0)).mean())


def run():
    c.setup();c.budget(10)
    registry=c.read(c.REGISTRY)
    if not any(t['experiment_id']==TRIAL for t in registry['trials']):
        c.register(TRIAL,'sequence_memory_diagnostic',
            'A single compressed recent-intent query may mix distinct daily interests; candidate-conditioned basket memory might preserve useful local matches.',
            params=PARAMS,training_protocol='No model fit; original four INNER labels only; not an outer rescue of302',
            candidate_protocol='unchanged original candidate pairs; proxies on exactly joint-available candidates',
            expected_minutes=5)
    destination=c.REPORT/'SEQUENCE_MEMORY_AUDIT.json'
    if destination.exists():return c.read(destination)
    windows={};start=time.perf_counter()
    for w,p in c.contracts()['rolling_protocol'].items():
        cutoff=p['inner_validation'];root=c.ART/'sequence_memory_audit'/cutoff;root.mkdir(parents=True,exist_ok=True)
        meta=c.read(c.ART/'gate_data'/f'2020_inner_{cutoff}'/'DATA.json')
        f=load_parquet(meta['ranks_path'])
        original_rows=len(f)
        sf=load_parquet(seq.ART/cutoff/'features.parquet')
        f=f.merge(sf,on=['customer_id','article_id'],validate='one_to_one')
        assert len(f)==original_rows and f.customer_id.nunique()==meta['included_users']
        items,vectors,source=seq.source(cutoff);vocab=pd.Index(items.article_id)
        users=load_parquet(seq.ART/cutoff/'users.parquet').set_index('customer_id')
        with np.load(seq.ART/cutoff/'prepared.npz',allow_pickle=False) as z:
            data={k:z[k] for k in ['basket_vectors','starts','dates','gaps','counts']}
        day=np.datetime64(cutoff,'D').astype(np.int32)
        rows=[];scores=[]
        for uid, group in f.groupby('customer_id',sort=True):
            if uid not in users.index:continue
            end=int(users.loc[uid,'basket_index']);begin=max(int(data['starts'][end]),end-7)
            baskets=data['basket_vectors'][begin:end+1];ages=day-data['dates'][begin:end+1]
            assert (ages>0).all()
            ix=vocab.get_indexer(group.article_id);ok=ix>=0
            if not ok.any():continue
            kept=group.loc[ok].copy();v=vectors[ix[ok]]
            maximum,attention=memory_scores(v,baskets,len(baskets),ages)
            kept['memory_max']=maximum;kept['memory_attention']=attention
            assert np.isfinite(kept.wv3_sequence_score).all()
            labels=kept.target.to_numpy()
            aucs={name:user_auc(labels,kept[col].to_numpy()) for name,col in
                [('gru','wv3_sequence_score'),('last','wv3_last_basket_cosine_control'),('maximum','memory_max'),('attention','memory_attention')]}
            gram=baskets@baskets.T
            cos=float(gram[np.triu_indices(len(baskets),1)].mean()) if len(baskets)>1 else None
            rows.append({'customer_id':uid,'baskets':len(baskets),'basket_mean_pair_cosine':cos,
                'joint_available_pairs':len(kept),'joint_available_positives':int(labels.sum()),**aucs})
            scores.append(kept[['customer_id','article_id','target','memory_max','memory_attention']])
        audit=pd.DataFrame(rows);valid=audit[audit.gru.notna()].copy()
        assert np.isfinite(valid[['gru','last','maximum','attention']]).all().all()
        multi=audit.baskets>1
        value={'cutoff':cutoff,'original_covered_active_users':int(f.customer_id.nunique()),
            'users_with_memory':len(audit),'users_with_joint_positive_and_negative':len(valid),
            'joint_available_pairs':int(audit.joint_available_pairs.sum()),
            'joint_available_positives':int(audit.joint_available_positives.sum()),
            'multi_basket_user_share':float(multi.mean()),
            'heterogeneous_among_multi_basket_share':float((audit.loc[multi,'basket_mean_pair_cosine']<.7).mean()),
            'mean_basket_pair_cosine_among_multi':float(audit.loc[multi,'basket_mean_pair_cosine'].mean()),
            'mean_user_auc':{k:float(valid[k].mean()) for k in ['gru','last','maximum','attention']},
            'delta_maximum_vs_gru':float((valid.maximum-valid.gru).mean()),
            'delta_attention_vs_gru':float((valid.attention-valid.gru).mean()),
            'source':source,'candidate_pool_changed':False,'model_fits':0,'final_week':'not_run'}
        save_parquet(audit,root/'user_audit.parquet');save_parquet(pd.concat(scores,ignore_index=True),root/'memory_scores.parquet')
        c.write(root/'AUDIT.json',value);windows[w]=value
        del f,sf,data,rows,scores,audit,valid;gc.collect()
    ds=[v['delta_maximum_vs_gru'] for v in windows.values()]
    passed=np.mean(ds)>=.01 and sum(d>0 for d in ds)>=3 and min(v['multi_basket_user_share'] for v in windows.values())>=.25
    result={'created_at':c.now(),'experiment_id':TRIAL,'params':PARAMS,'windows':windows,
        'mean_auc_delta_maximum_vs_gru':float(np.mean(ds)),'positive_windows':sum(d>0 for d in ds),
        'mechanism_gate':bool(passed),'runtime_seconds':time.perf_counter()-start,
        'scope':'INNER diagnostic; no trained memory ranker, no fused MAP, no claim of promotion or of causal GRU failure',
        'definitions_zh':{'user_auc':'行业排序代理：同一用户内、四种表示共同可用的正负候选对中，正例分数更高的比例，并列计0.5；先逐用户计算，再对同时有正负例的用户等权平均。不是完整人口MAP。',
        'memory_max':'项目候选条件记忆：候选与最近最多8个购买日集合向量余弦的最大值；集合向量仅使用截止前历史，不含待预测标签。',
        'memory_attention':'项目非参数注意力对照：以余弦/0.1减去28天半衰期惩罚的softmax权重，对各购买日余弦加权。没有训练新网络。',
        'heterogeneous':'项目预先固定的描述阈值：至少两个购买日时，日集合向量两两平均余弦低于0.7；分母为至少两个购买日的用户，不等于已证明存在独立品类兴趣。'},
        'final_week':'not_run'}
    c.write(destination,result);c.update(TRIAL,decision='memory_mechanism_pass' if passed else 'reject_memory_mechanism',
        inner_evidence={'mean_auc_delta':result['mean_auc_delta_maximum_vs_gru'],'positive_windows':result['positive_windows'],'gate':bool(passed)},
        runtime=result['runtime_seconds'],artifact_paths=[str(destination)])
    print({k:result[k] for k in ['mean_auc_delta_maximum_vs_gru','positive_windows','mechanism_gate','runtime_seconds']},flush=True)
    return result


if __name__=='__main__':run()
