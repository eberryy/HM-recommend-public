"""Cutoff-safe BPR factorization as two additional frozen-pool ranking columns."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from .warm_v2_contract import ARTIFACT,REPORT,assert_branch,evidence_id,guard,now,read,write
from .warm_v2_engine import Engine,connection,literal,load_parquet,save_parquet

BPR_SPEC={'bpr_match':{
    'bpr_user_item_score':'项目特征：行业BPR矩阵分解的用户因子与候选商品因子内积，含实现中的商品偏置；越大表示相对偏好更强，不是概率。仅用截止前全历史去重购买对训练。',
    'bpr_unavailable':'项目特征：用户或商品在截止前训练词表中不存在为1，此时匹配分数为NaN；否则为0。'}}
BPR_SPEC['bpr_bias']={'bpr_item_bias':'项目消融特征：同一已训练BPR商品因子的最后一列偏置，与用户无关；用于区分完整匹配增益与商品偏置增益。',
    'bpr_unavailable':BPR_SPEC['bpr_match']['bpr_unavailable']}
PARAMS={'factors':100,'learning_rate':.01,'regularization':.01,'iterations':100,
    'num_threads':8,'verify_negative_samples':True,'random_state':20260908}
ROOT=ARTIFACT/'bpr-match-v1'


def contract():
    path=REPORT/'WV2_501_CONTRACT.json'
    if path.exists():
        value=read(path)
        assert value['params']==PARAMS and value['features']['bpr_match']==BPR_SPEC['bpr_match'], 'registered BPR semantics changed; create a new version'
        return value
    assert read(REPORT/'WV2_500_BPR_AUDIT.json')['opportunity_gate_passed']
    out={'registered_at':now(),'stage':'WV2.6 learned user-item feature','trial':'WV2-501',
        'hypothesis':'Directly learned user-item ID matching contains collaborative information missing from existing item-item and hierarchy summaries.',
        'features':BPR_SPEC,'params':PARAMS,'library':'implicit0.7.3 CPU Windows binary; no framework migration',
        'history':'all original transaction dates strictlybefore each feature cutoff; all users, not only10% target users',
        'events':'BPR binary implicit relation: repeated purchases map to one nonzero user-item entry; raw data is unchanged. Unpurchased is an implicit sampled negative, not a proven dislike.',
        'boundaries':'unchanged original training weeks,10% target users,30:1 LGB sample,84 baseline features plus2, all candidate IDs/budget, inner/outer dates and complete MAP denominator. No recent-target orXE_NDCG combination.',
        'screen_gate':{'mean_min':.0001,'positive_windows_min':3,'worst_min':-.0005},
        'outer_confirmations_max':1,'milestone_gate':{'mean_min':.0005,'nondegrade_min':3,'worst_min':-.0003,'large_gain_manual_review':.001},
        'cost':'First old training cutoff pilot before8-cutoff historical batch. CPU8; model fitting projected<=300sec after5epochs, total cutoff<=900sec; disk floor5GiB. Expected20-60min historical batch subject to pilot; noGPU/sharedserver.',
        'stopping':'If pilot exceeds cost, preserve incomplete evidence and diagnose; no reduced epochs picked from validation. If inner fails, no outer confirmation. If outer fails, keep Warm-v1 and analyze a different identifiable bottleneck, not seed/epoch sweep.',
        'reproducibility':'Fixed random_state and thread count, no seed selection. Parallel SGD updates can be nondeterministic; save exact factors and generated scores. Do not promise bitwise retraining identity.',
        'sources':['https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129','https://arxiv.org/abs/1205.2618','https://benfred.github.io/implicit/api/models/cpu/bpr.html'],
        'final_week':'not_run','automatic_merge_or_push':False}
    write(path,out);return out


def relation(con,transactions,cutoff):
    guard(cutoff)
    con.execute(f"CREATE VIEW history AS SELECT customer_id,article_id,t_dat FROM read_parquet({literal(transactions)}) WHERE t_dat<DATE '{cutoff}'")
    con.execute('CREATE TEMP TABLE users AS SELECT customer_id,(row_number() OVER(ORDER BY customer_id)-1)::INTEGER user_index FROM (SELECT DISTINCT customer_id FROM history)')
    con.execute('CREATE TEMP TABLE items AS SELECT article_id,(row_number() OVER(ORDER BY article_id)-1)::INTEGER item_index FROM (SELECT DISTINCT article_id FROM history)')
    pairs=con.execute('SELECT DISTINCT u.user_index,i.item_index FROM history h JOIN users u USING(customer_id) JOIN items i USING(article_id) ORDER BY u.user_index,i.item_index').fetchdf()
    users=con.execute('SELECT * FROM users ORDER BY user_index').fetchdf()
    items=con.execute('SELECT * FROM items ORDER BY item_index').fetchdf()
    latest=str(con.execute('SELECT max(t_dat) FROM history').fetchone()[0]);assert latest<cutoff
    matrix=csr_matrix((np.ones(len(pairs),np.float32),(pairs.user_index.to_numpy(),pairs.item_index.to_numpy())),shape=(len(users),len(items)))
    assert matrix.nnz==len(pairs) and np.all(matrix.data==1)
    return matrix,users,items,latest


def fit(cutoff,engine):
    assert_branch();guard(cutoff);contract()
    root=ROOT/cutoff;meta=root/'model.json'
    if meta.exists():
        value=read(meta)
        assert value['cutoff']==cutoff and value['params']==PARAMS
        assert value['transactions']['sha256']==engine.history['inputs']['transactions']['sha256']
        for artifact in value['artifacts'].values():
            assert Path(artifact['path']).stat().st_size==artifact['bytes']
        return value
    root.mkdir(parents=True,exist_ok=True)
    if (root/'factors.npz').exists():raise RuntimeError('preserve incomplete BPR model')
    assert shutil.disk_usage('.').free>5*1024**3
    start=time.perf_counter();print(f'BPR {cutoff}: full pre-cutoff binary relation',flush=True)
    with connection() as con:
        spill=root/'spill';spill.mkdir(exist_ok=True);con.execute(f'SET temp_directory={literal(spill.resolve())}')
        matrix,users,items,latest=relation(con,engine.transactions,cutoff)
    prep=time.perf_counter()-start
    from implicit.cpu.bpr import BayesianPersonalizedRanking
    import implicit
    model=BayesianPersonalizedRanking(**PARAMS);epochs=[]
    review=REPORT/'WV2_501_RESOURCE_REVIEW.json'
    fit_limit=read(review)['max_projected_fit_seconds'] if review.exists() else 300
    completion=REPORT/'WV2_501_COMPLETION_COST_REVIEW.json'
    if completion.exists() and read(completion)['cutoff']==cutoff:
        fit_limit=read(completion)['max_projected_fit_seconds']
    def callback(epoch,elapsed,correct,skipped):
        epochs.append({'epoch':int(epoch),'seconds':float(elapsed),
            'correct_sampled_pairs':int(correct),'skipped_known_positive_pairs':int(skipped),
            'sampled_pair_accuracy':float(correct/max(1,matrix.nnz-skipped))})
        if time.perf_counter()-start>900:
            write(root/f'COST_STOP_ACTUAL_{time.time_ns()}.json',{'epochs':epochs,'elapsed':time.perf_counter()-start,'final_week':'not_run'})
            raise RuntimeError('BPR actual cutoff cost exceeds900seconds')
        if (epoch+1)%10==0: print(f'BPR {cutoff}: epoch{epoch+1}, fit_seconds={sum(v["seconds"] for v in epochs):.1f}',flush=True)
        if epoch==4 and sum(v['seconds'] for v in epochs)/5*100>fit_limit:
            write(root/f'COST_STOP_{time.time_ns()}.json',{'epochs':epochs,'prepare_seconds':prep,'projected_fit_seconds':sum(v['seconds'] for v in epochs)/5*100,'final_week':'not_run'})
            raise RuntimeError('BPR projected fitting cost exceeds pilot contract; no automatic epoch reduction')
    t=time.perf_counter()
    # OpenMP8 runs BPR; BLAS1 avoids nested oversubscription, not an objective change.
    with threadpool_limits(limits=1,user_api='blas'):
        model.fit(matrix,show_progress=False,callback=callback)
    fit_seconds=time.perf_counter()-t
    assert np.isfinite(model.user_factors).all() and np.isfinite(model.item_factors).all()
    np.savez(root/'factors.npz',user_factors=model.user_factors,item_factors=model.item_factors)
    save_parquet(users,root/'users.parquet');save_parquet(items,root/'items.parquet')
    out={'cutoff':cutoff,'latest_history_date':latest,'pit_safe':True,'params':PARAMS,'library':implicit.__version__,
        'binary_pairs':matrix.nnz,'users':matrix.shape[0],'items':matrix.shape[1],
        'prepare_seconds':prep,'fit_seconds':fit_seconds,'runtime_seconds':time.perf_counter()-start,
        'epochs':epochs,'factor_shapes':[list(model.user_factors.shape),list(model.item_factors.shape)],
        'artifacts':{k:evidence_id(root/k,reason='explicit_registry_evidence') for k in ('factors.npz','users.parquet','items.parquet')},
        'transactions':engine.history['inputs']['transactions'],'final_week':'not_run'}
    write(meta,out);print(f'BPR {cutoff}: done {out["runtime_seconds"]:.1f}s',flush=True)
    del model,matrix,users,items;gc.collect()
    return out


def match_scores(frame,users,items,user_factors,item_factors):
    ui=users.get_indexer(frame.customer_id);ii=items.get_indexer(frame.article_id)
    good=(ui>=0)&(ii>=0)
    score=np.full(len(frame),np.nan,np.float32)
    score[good]=np.einsum('ij,ij->i',user_factors[ui[good]],item_factors[ii[good]])
    return score,(~good).astype(np.float32)


def build_features(cutoff,engine):
    guard(cutoff);root=ROOT/cutoff;meta=root/'features.json';path=root/'features.parquet'
    if meta.exists():
        value=read(meta)
        assert value['cutoff']==cutoff and value['source_sha256']==engine.history['feature_cache'][cutoff]['artifact']['sha256']
        assert path.stat().st_size==value['artifact']['bytes']
        return path,value
    start=time.perf_counter();model=fit(cutoff,engine)
    if model['runtime_seconds']>900:raise RuntimeError('BPR cutoff resource gate')
    with np.load(root/'factors.npz',allow_pickle=False) as f:
        uf=f['user_factors'];itf=f['item_factors']
    users=pd.Index(load_parquet(root/'users.parquet').customer_id)
    items=pd.Index(load_parquet(root/'items.parquet').article_id)
    chunks=[]
    with connection() as con:
        cursor=con.execute(f'SELECT customer_id,article_id FROM read_parquet({literal(engine.base_path(cutoff))})')
        while True:
            frame=cursor.fetch_df_chunk(32)
            if frame.empty:break
            score,missing=match_scores(frame,users,items,uf,itf)
            frame['wv2_bpr_user_item_score']=score;frame['wv2_bpr_unavailable']=missing;chunks.append(frame)
    output=pd.concat(chunks,ignore_index=True);assert not output.duplicated(['customer_id','article_id']).any()
    assert np.isfinite(output.loc[output.wv2_bpr_unavailable==0,'wv2_bpr_user_item_score']).all()
    save_parquet(output,path)
    result={'cutoff':cutoff,'family':'bpr_match','keys':['customer_id','article_id'],'rows':len(output),
        'features':['wv2_bpr_user_item_score','wv2_bpr_unavailable'],'latest_history_date':model['latest_history_date'],
        'pit_safe':True,'future_labels_used':False,'source':str(engine.base_path(cutoff)),
        'source_sha256':engine.history['feature_cache'][cutoff]['artifact']['sha256'],
        'unavailable_pairs':int(output.wv2_bpr_unavailable.sum()),'runtime_seconds':time.perf_counter()-start,
        'model_metadata':str(root/'model.json'),'artifact':evidence_id(path,reason='explicit_registry_evidence')}
    write(meta,result);return path,result


def build_bias(cutoff,engine):
    guard(cutoff)
    assert read(REPORT/'WV2-501_SCREEN.json')['screening']['passed'], 'conditional diagnostic only'
    root=ROOT/cutoff;meta=root/'bias_features.json';path=root/'bias_features.parquet'
    if meta.exists():return path,read(meta)
    start=time.perf_counter()
    full=read(root/'features.json');frame=load_parquet(root/'features.parquet')
    items=pd.Index(load_parquet(root/'items.parquet').article_id)
    with np.load(root/'factors.npz',allow_pickle=False) as f:itf=f['item_factors']
    index=items.get_indexer(frame.article_id);good=frame.wv2_bpr_unavailable.to_numpy()==0
    assert (index[good]>=0).all()
    scores=np.full(len(frame),np.nan,np.float32);scores[good]=itf[index[good],-1]
    frame=frame.drop(columns=['wv2_bpr_user_item_score']);frame['wv2_bpr_item_bias']=scores
    save_parquet(frame,path)
    result={**full,'family':'bpr_bias','features':['wv2_bpr_item_bias','wv2_bpr_unavailable'],
        'runtime_seconds':time.perf_counter()-start,'artifact':evidence_id(path,reason='explicit_registry_evidence'),
        'mechanism':'Same saved BPR factors, only item bias. No BPR retraining.'}
    write(meta,result);return path,result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['register','build']);p.add_argument('--cutoff');a=p.parse_args()
    if a.command=='register':contract()
    else:build_features(a.cutoff,Engine(read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json')))
