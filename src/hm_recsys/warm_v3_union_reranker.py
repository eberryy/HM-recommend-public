"""WV3-381 recent-window cross-fitted LambdaRank over a frozen Top12 union."""
from __future__ import annotations
import gc
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from . import warm_v3_common as common
from .warm_v2_contract import evidence_id,read,write
from .warm_v2_engine import connection,literal,save_parquet
from .warm_v2_rank_fusion import ap_table
from .warm_v3_expert import screening_gate
from .warm_v3_user_router import fold,sources

TRIAL='WV3-381'
FEATURES=['candidate_rank','baseline_rank','rj','fusion_rank','r0','score_base','score_bpr','score_joint',
    'baseline_top12','joint_top12','rank_difference']
PARAMS={'objective':'lambdarank','metric':'None','learning_rate':.05,'num_leaves':15,
    'min_data_in_leaf':20,'deterministic':True,'force_col_wise':True,'num_threads':4,
    'seed':20260909,'feature_fraction_seed':20260909,'bagging_seed':20260909,
    'lambdarank_truncation_level':12,'verbosity':-1}
ROUNDS=30


def union_frame(path):
    with connection() as con:
        frame=con.execute(f'''SELECT customer_id,article_id,candidate_rank,target,truth_count,
            user_history_events_12w,baseline_rank,rj,fusion_rank,r0,score_base,score_bpr,score_joint,
            CAST(baseline_rank<=12 AS INTEGER) baseline_top12,CAST(fusion_rank<=12 AS INTEGER) joint_top12,
            abs(baseline_rank-fusion_rank) rank_difference
            FROM read_parquet({literal(path)}) WHERE baseline_rank<=12 OR fusion_rank<=12
            ORDER BY customer_id,baseline_rank,candidate_rank,article_id''').fetchdf()
    assert not frame.duplicated(['customer_id','article_id']).any()
    sizes=frame.groupby('customer_id',sort=False).size()
    assert sizes.between(12,24).all()
    return frame


def fit(frame):
    frame=frame.sort_values(['customer_id','baseline_rank','candidate_rank','article_id'],kind='mergesort')
    groups=frame.groupby('customer_id',sort=False).size().to_numpy()
    ds=lgb.Dataset(frame[FEATURES].to_numpy(np.float32),label=frame.target.to_numpy(np.uint8),group=groups,
        feature_name=FEATURES,free_raw_data=True)
    return lgb.train(PARAMS,ds,num_boost_round=ROUNDS)


def rank_and_ap(frame,scores,total_users):
    ranked=frame.copy();ranked['rerank_score']=scores
    active=ranked.user_history_events_12w>0
    ranked=ranked.sort_values(['customer_id','rerank_score','baseline_rank','candidate_rank','article_id'],
        ascending=[True,False,True,True,True],kind='mergesort')
    ranked['model_rank']=ranked.groupby('customer_id',sort=False).cumcount()+1
    ranked['final_rank']=np.where(active.loc[ranked.index],ranked.model_rank,ranked.baseline_rank)
    ranked=ranked.sort_values(['customer_id','final_rank'],kind='mergesort')
    with connection() as con:
        con.register('ranked',ranked)
        aps=None
        for label,rank in [('baseline','baseline_rank'),('union','final_rank')]:
            one=ap_table(con,'ranked',rank).rename(columns={'ap':'ap_'+label})
            aps=one if aps is None else aps.merge(one,on='customer_id',validate='one_to_one')
    assert len(aps)<=total_users
    return ranked,aps


def screen_window(window,total_users):
    root=common.ART/TRIAL/window/'inner';report=root/'REVIEW.json'
    if report.is_file():return read(report)
    root.mkdir(parents=True,exist_ok=True);started=time.perf_counter();rank_path,_=sources(window,'inner')
    frame=union_frame(rank_path);folds=np.array([fold(v) for v in frame.customer_id]);scores=np.zeros(len(frame));fold_meta=[]
    for held in (0,1):
        train_users={u for u in frame.loc[folds!=held,'customer_id'].unique()}
        train=frame[frame.customer_id.isin(train_users)].copy();model=fit(train)
        mask=folds==held;scores[mask]=model.predict(frame.loc[mask,FEATURES],num_threads=4)
        fold_meta.append({'heldout_fold':held,'training_users':len(train_users),'training_rows':len(train)})
        del model,train;gc.collect()
    ranked,aps=rank_and_ap(frame,scores,total_users);delta=aps.ap_union-aps.ap_baseline
    baseline=float(aps.ap_baseline.sum()/total_users);model=fit(frame);model_path=root/'MODEL.txt';model.save_model(str(model_path))
    write(root/'MODEL.json',{'window':window,'params':PARAMS,'rounds':ROUNDS,'features':FEATURES,
        'training_users':frame.customer_id.nunique(),'training_rows':len(frame),'model':evidence_id(model_path,reason='explicit_registry_evidence'),
        'outer_labels_used':False,'final_week':'not_run'})
    save_parquet(ranked,root/'ranks.parquet');save_parquet(aps,root/'user_ap.parquet')
    result={'window':window,'role':'two_fold_user_OOF_inner','included_users':len(aps),'total_users':total_users,
        'union_rows':len(frame),'mean_union_size':float(len(frame)/len(aps)),'positive_rows':int(frame.target.sum()),
        'positive_density':float(frame.target.mean()),'baseline_MAP':baseline,
        'MAP':float(aps.ap_union.sum()/total_users),'population_delta':float(delta.sum()/total_users),
        'positive_user_count':int((delta>1e-15).sum()),'harmed_user_count':int((delta< -1e-15).sum()),
        'folds':fold_meta,'model':str(root/'MODEL.json'),'candidate_pool_changed':False,
        'runtime_seconds':time.perf_counter()-started,'final_week':'not_run'}
    write(report,result);print({'union_inner':window,'delta':result['population_delta'],'density':result['positive_density']},flush=True);return result


def screen():
    common.setup();common.budget(8);entry=next(t for t in read(common.REGISTRY)['trials'] if t['experiment_id']==TRIAL)
    assert entry['decision']=='preregistered' and entry['outer_exposures']==0
    engine=common.Engine(2020);started=time.perf_counter();rows={}
    for window,p in engine.contract['rolling_protocol'].items():
        meta=read(common.ART/'gate_data'/f"2020_inner_{p['inner_validation']}"/'DATA.json')
        rows[window]=screen_window(window,meta['total_users'])
        assert abs(rows[window]['baseline_MAP']-meta['baseline_map_population_component'])<1e-12
    gate=screening_gate(v['population_delta'] for v in rows.values())
    result={'experiment_id':TRIAL,'architecture':'recent_inner_crossfit_top12_union_reranker','windows':rows,
        'screening':gate,'params':PARAMS,'rounds':ROUNDS,'features':FEATURES,'candidate_pool_changed':False,
        'runtime_seconds':time.perf_counter()-started,'no_registry_mutation':True,'final_week':'not_run'}
    write(common.REPORT/(TRIAL+'_SCREEN.json'),result);print(gate,flush=True);return result


def confirm():
    common.setup();common.budget(10);entry=next(t for t in read(common.REGISTRY)['trials'] if t['experiment_id']==TRIAL)
    assert entry['outer_exposures']==1 and read(common.REPORT/(TRIAL+'_SCREEN.json'))['screening']['passed']
    destination=common.REPORT/(TRIAL+'_OUTER.json')
    if destination.is_file():return read(destination)
    base=read(common.ROOT/'reports/warm_v2/WV2-601_OUTER.json')['per_window_MAP'];rows={};started=time.perf_counter()
    for window in common.contracts()['rolling_protocol']:
        rank_path,_=sources(window,'outer');frame=union_frame(rank_path);model_meta=read(common.ART/TRIAL/window/'inner'/'MODEL.json')
        model=lgb.Booster(model_file=model_meta['model']['path']);scores=model.predict(frame[FEATURES],num_threads=4)
        ranked,aps=rank_and_ap(frame,scores,len(frame.customer_id.unique()));delta=aps.ap_union-aps.ap_baseline
        assert abs(float(aps.ap_baseline.mean())-base[window])<1e-12
        root=common.ART/TRIAL/window/'outer';root.mkdir(parents=True,exist_ok=True);top=ranked[ranked.final_rank<=12].copy()
        assert len(top)==12*len(aps);top_path=root/'top12.parquet';save_parquet(top,top_path);save_parquet(aps,root/'user_ap.parquet')
        rows[window]={'window':window,'role':'heldout_outer','total_users':len(aps),'union_rows':len(frame),
            'positive_density':float(frame.target.mean()),'baseline_MAP':base[window],'MAP':float(aps.ap_union.mean()),
            'population_delta':float(delta.mean()),'positive_user_count':int((delta>1e-15).sum()),
            'harmed_user_count':int((delta< -1e-15).sum()),'model_source':str(common.ART/TRIAL/window/'inner'/'MODEL.json'),
            'top12':evidence_id(top_path,reason='explicit_registry_evidence'),'candidate_pool_changed':False,'final_week':'not_run'}
        write(root/'REVIEW.json',rows[window]);print({'union_outer':window,'delta':rows[window]['population_delta']},flush=True)
    result={'experiment_id':TRIAL,'architecture':'recent_inner_crossfit_top12_union_reranker','windows':rows,
        **common.summary({w:r['MAP'] for w,r in rows.items()},2020),'params':PARAMS,'rounds':ROUNDS,'features':FEATURES,
        'candidate_pool_changed':False,'runtime_seconds':time.perf_counter()-started,'no_registry_mutation':True,'final_week':'not_run'}
    write(destination,result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['screen','confirm'],nargs='?',default='screen')
    globals()[parser.parse_args().command]()
