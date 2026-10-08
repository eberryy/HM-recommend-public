"""WV3-351 causal current/previous checkpoint rank ensemble; INNER screen."""
from __future__ import annotations
from datetime import date, timedelta
import gc
import time

import lightgbm as lgb
import numpy as np

from . import warm_v2_engine as we
from . import warm_v3_common as common
from .metrics import apk
from .warm_v2_contract import evidence_id, read, write
from .warm_v2_rank_fusion import ap_table, rank_sql
from .warm_v3_expert import screening_gate

TRIAL='WV3-351'


def predecessor_map(protocol):
    windows=list(protocol)
    return {window:(windows[i-1] if i else None) for i,window in enumerate(windows)}


def assert_causal(previous_root, target_cutoff):
    meta=read(previous_root/'outer_metrics.json')
    latest=max(date.fromisoformat(v) for v in meta['train_cutoffs'])+timedelta(days=7)
    assert latest<=date.fromisoformat(target_cutoff), (latest,target_cutoff)
    return {'previous_training_cutoffs':meta['train_cutoffs'],
        'latest_previous_training_label_end_exclusive':latest.isoformat(),'target_cutoff':target_cutoff}


def window(engine, name, protocol, previous):
    cutoff=protocol['inner_validation']; baseline=read(common.ART/'gate_data'/f'2020_inner_{cutoff}'/'DATA.json')
    total=baseline['total_users']; root=common.ART/TRIAL/name/'inner'; output=root/'REVIEW.json'
    if output.is_file():return read(output)
    root.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    if previous is None:
        result={'window':name,'cutoff':cutoff,'previous_window':None,'no_op':True,
            'baseline_MAP':baseline['baseline_map_population_component'],
            'MAP':baseline['baseline_map_population_component'],'population_delta':0.,
            'candidate_pool_changed':False,'runtime_seconds':time.perf_counter()-started,'final_week':'not_run'}
        write(output,result);return result
    base_path,_=engine.cached_data(cutoff,'inner');frame=we.load_parquet(base_path)
    old=we.load_parquet(baseline['ranks_path'])
    frame=old[['customer_id','article_id']].merge(frame,on=['customer_id','article_id'],how='left',validate='one_to_one')
    assert len(frame)==len(old) and not frame.isna().all(axis=1).any()
    for col in ('customer_id','article_id','candidate_rank','target','truth_count','user_history_events_12w'):
        assert np.array_equal(frame[col].to_numpy(),old[col].to_numpy())
    scores=frame[['customer_id','article_id','candidate_rank','target','truth_count','user_history_events_12w']].copy()
    causal=[]
    for suffix,label,families in [('000','e0',[]),('501','e1',['bpr_match'])]:
        model_root=common.OLD/f'WV2-{suffix}'/previous
        causal.append(assert_causal(model_root,cutoff))
        model=lgb.Booster(model_file=str(model_root/'outer_model.txt'))
        maps={k:{int(a):int(b) for a,b in v.items()} for k,v in read(model_root/'outer_category_maps.json').items()}
        ff=common.attach_features(frame,cutoff,families,engine) if families else frame
        scores['lag_'+label]=model.predict(we.prepare(ff,model.feature_name(),maps),num_threads=8)
        del model,ff
    scores['current_rank']=old.ap_rf.to_numpy()
    with we.connection() as con:
        con.register('scores',scores)
        con.execute(f'''CREATE TEMP TABLE lag_models AS SELECT *,{rank_sql('lag_e0')} lag_r0,
            {rank_sql('lag_e1')} lag_r1 FROM scores''')
        con.execute('''CREATE TEMP TABLE lag_score AS SELECT *,
            1.0/(60+lag_r0)+1.0/(60+lag_r1) previous_score FROM lag_models''')
        con.execute(f'''CREATE TEMP TABLE previous_ranked AS SELECT *,{rank_sql('previous_score')} previous_rank
            FROM lag_score''')
        con.execute('''CREATE TEMP TABLE fused_score AS SELECT *,
            1.0/(60+current_rank)+1.0/(60+previous_rank) temporal_score FROM previous_ranked''')
        con.execute(f'''CREATE TEMP TABLE raw_ranked AS SELECT *,{rank_sql('temporal_score')} temporal_rank
            FROM fused_score''')
        con.execute('''CREATE TEMP TABLE ranked AS SELECT *,CASE WHEN user_history_events_12w=0
            THEN candidate_rank ELSE temporal_rank END final_rank FROM raw_ranked''')
        aps=None
        for label,rank in [('baseline','current_rank'),('temporal','final_rank')]:
            one=ap_table(con,'ranked',rank).rename(columns={'ap':'ap_'+label})
            aps=one if aps is None else aps.merge(one,on='customer_id',validate='one_to_one')
        assert len(aps)==baseline['included_users']
        baseline_map=float(aps.ap_baseline.sum()/total)
        assert abs(baseline_map-baseline['baseline_map_population_component'])<1e-12
        delta=aps.ap_temporal-aps.ap_baseline
        rank_path=root/'ranks.parquet'
        con.execute(f'COPY ranked TO {we.literal(rank_path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
    we.save_parquet(aps,root/'user_ap.parquet')
    result={'window':name,'cutoff':cutoff,'previous_window':previous,'no_op':False,
        'baseline_MAP':baseline_map,'MAP':float(aps.ap_temporal.sum()/total),
        'population_delta':float(delta.sum()/total),'positive_user_count':int((delta>1e-15).sum()),
        'harmed_user_count':int((delta< -1e-15).sum()),'causal_guards':causal,
        'candidate_rows':len(scores),'included_users':len(aps),'total_users':total,
        'candidate_pool_changed':False,'rank_path':str(rank_path),
        'runtime_seconds':time.perf_counter()-started,'final_week':'not_run'}
    write(output,result);print({'temporal_inner':name,'delta':result['population_delta']},flush=True)
    del frame,old,scores,aps;gc.collect();return result


def screen():
    common.setup();common.budget(8)
    entry=next(t for t in read(common.REGISTRY)['trials'] if t['experiment_id']==TRIAL)
    assert entry['decision']=='preregistered' and entry['outer_exposures']==0
    engine=common.Engine(2020);pred=predecessor_map(engine.contract['rolling_protocol']);started=time.perf_counter()
    rows={name:window(engine,name,p,pred[name]) for name,p in engine.contract['rolling_protocol'].items()}
    gate=screening_gate(v['population_delta'] for v in rows.values())
    result={'experiment_id':TRIAL,'architecture':'causal_previous_checkpoint_rank_ensemble',
        'windows':rows,'screening':gate,'new_training':False,'candidate_pool_changed':False,
        'runtime_seconds':time.perf_counter()-started,'no_registry_mutation':True,'final_week':'not_run'}
    write(common.REPORT/(TRIAL+'_SCREEN.json'),result);print(gate,flush=True);return result


def outer_window(engine,name,protocol,previous):
    cutoff=protocol['outer_validation'];root=common.ART/TRIAL/name/'outer';output=root/'REVIEW.json'
    if output.is_file():return read(output)
    root.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    baseline_review=read(common.ART/'WV3-331'/name/'MULTIVIEW_OUTER_REVIEW.json')
    baseline_map=baseline_review['baseline_full_MAP'];total=baseline_review['full_users']
    if previous is None:
        result={'window':name,'cutoff':cutoff,'previous_window':None,'no_op':True,
            'baseline_MAP':baseline_map,'MAP':baseline_map,'population_delta':0.,
            'candidate_pool_changed':False,'runtime_seconds':time.perf_counter()-started,'final_week':'not_run'}
        write(output,result);return result
    source=engine.base_path(cutoff);base_ranks=common.ART/'WV3-331'/name/'outer_multiview_ranks.parquet'
    score_db=root/'lag_scores.duckdb'
    if not score_db.exists():
        models=[];causal=[]
        for suffix in ('000','501'):
            model_root=common.OLD/f'WV2-{suffix}'/previous;causal.append(assert_causal(model_root,cutoff))
            models.append((lgb.Booster(model_file=str(model_root/'outer_model.txt')),
                {k:{int(a):int(b) for a,b in v.items()} for k,v in read(model_root/'outer_category_maps.json').items()}))
        src=we.connection();dest=we.duckdb.connect(str(score_db));rows=0;batches=0
        try:
            cols=list(dict.fromkeys(['customer_id','article_id',*engine.features,'target','user_history_events_12w']))
            cursor=src.execute(f'SELECT {",".join(cols)} FROM read_parquet({we.literal(source)})')
            while True:
                frame=cursor.fetch_df_chunk(vectors_per_chunk=48)
                if frame.empty:break
                ff=common.attach_features(frame,cutoff,['bpr_match'],engine)
                scored=frame[['customer_id','article_id','candidate_rank','target','user_history_events_12w']].copy()
                scored['lag_e0']=models[0][0].predict(we.prepare(frame,models[0][0].feature_name(),models[0][1]),num_threads=8)
                scored['lag_e1']=models[1][0].predict(we.prepare(ff,models[1][0].feature_name(),models[1][1]),num_threads=8)
                dest.register('batch',scored)
                dest.execute('CREATE TABLE predictions AS SELECT * FROM batch' if batches==0 else 'INSERT INTO predictions SELECT * FROM batch')
                rows+=len(frame);batches+=1
        finally:
            src.close();dest.close()
        assert rows==baseline_review['candidate_rows']
        write(root/'LAG_SCORE_META.json',{'cutoff':cutoff,'previous_window':previous,'causal_guards':causal,
            'candidate_rows':rows,'batches':batches,'candidate_pool_changed':False,'final_week':'not_run'})
        del models;gc.collect()
    lag_meta=read(root/'LAG_SCORE_META.json')
    output_db=root/'evaluation.duckdb';resume_output=output_db.exists()
    with we.connection() as con:
        con.execute(f'CREATE TEMP TABLE base AS SELECT customer_id,article_id,candidate_rank,target,truth_count,user_history_events_12w,baseline_rank current_rank FROM read_parquet({we.literal(base_ranks)})')
        con.execute(f'ATTACH {we.literal(score_db)} AS lag (READ_ONLY)')
        errors=con.execute('''SELECT count(*) FROM base b FULL JOIN lag.predictions l USING(customer_id,article_id)
            WHERE b.customer_id IS NULL OR l.customer_id IS NULL OR b.target<>l.target
            OR b.candidate_rank<>l.candidate_rank OR b.user_history_events_12w<>l.user_history_events_12w''').fetchone()[0]
        assert errors==0
        con.execute('''CREATE TEMP TABLE scores AS SELECT b.*,l.lag_e0,l.lag_e1
            FROM base b JOIN lag.predictions l USING(customer_id,article_id)''')
        con.execute(f'''CREATE TEMP TABLE lag_models AS SELECT *,{rank_sql('lag_e0')} lag_r0,
            {rank_sql('lag_e1')} lag_r1 FROM scores''')
        con.execute('''CREATE TEMP TABLE lag_score AS SELECT *,
            1.0/(60+lag_r0)+1.0/(60+lag_r1) previous_score FROM lag_models''')
        con.execute(f'''CREATE TEMP TABLE previous_ranked AS SELECT *,{rank_sql('previous_score')} previous_rank FROM lag_score''')
        con.execute('''CREATE TEMP TABLE fused_score AS SELECT *,
            1.0/(60+current_rank)+1.0/(60+previous_rank) temporal_score FROM previous_ranked''')
        con.execute(f'''CREATE TEMP TABLE raw_ranked AS SELECT *,{rank_sql('temporal_score')} temporal_rank FROM fused_score''')
        con.execute('''CREATE TEMP TABLE ranked AS SELECT *,CASE WHEN user_history_events_12w=0
            THEN candidate_rank ELSE temporal_rank END final_rank FROM raw_ranked''')
        rows,users=con.execute('SELECT count(*),count(DISTINCT customer_id) FROM ranked').fetchone();assert users==total
        inactive_errors=con.execute('SELECT count(*) FROM ranked WHERE user_history_events_12w=0 AND final_rank<>candidate_rank').fetchone()[0]
        assert inactive_errors==0
        aps=None
        for label,rank in [('baseline','current_rank'),('temporal','final_rank')]:
            one=ap_table(con,'ranked',rank).rename(columns={'ap':'ap_'+label})
            aps=one if aps is None else aps.merge(one,on='customer_id',validate='one_to_one')
        assert len(aps)==total;checked_base=float(aps.ap_baseline.mean());assert abs(checked_base-baseline_map)<1e-12
        delta=aps.ap_temporal-aps.ap_baseline
        con.execute(f'ATTACH {we.literal(output_db)} AS output'+(' (READ_ONLY)' if resume_output else ''))
        if resume_output:
            resumed=con.execute('''SELECT count(*),max(abs(o.score-r.temporal_score)) FROM output.predictions o
                JOIN ranked r USING(customer_id,article_id)''').fetchone()
            assert resumed[0]==rows and resumed[1]<=1e-12
            assert con.execute('SELECT count(*) FROM output.top12').fetchone()[0]==12*users
        else:
            con.execute('CREATE TABLE output.predictions AS SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w,temporal_score score FROM ranked')
            con.execute('CREATE TABLE output.top12 AS SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w,temporal_score score,final_rank FROM ranked WHERE final_rank<=12')
        top=con.execute('SELECT customer_id,article_id,final_rank FROM ranked WHERE final_rank<=12 ORDER BY customer_id,final_rank').fetchdf()
        con.execute(f'''CREATE TEMP TABLE truth AS SELECT DISTINCT customer_id,article_id FROM read_parquet({we.literal(engine.transactions)})
            WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY AND hash(customer_id)%1000000<100000''')
        truth=con.execute('SELECT * FROM truth ORDER BY customer_id,article_id').fetchdf()
        truthsets={u:set(g.article_id) for u,g in truth.groupby('customer_id',sort=False)}
        pred={u:g.article_id.tolist() for u,g in top.groupby('customer_id',sort=False)}
        assert set(pred)==set(truthsets) and all(len(v)==len(set(v))==12 for v in pred.values())
        independent=float(np.mean([apk(list(t),pred[u]) for u,t in truthsets.items()]))
        new_map=float(aps.ap_temporal.mean());assert abs(independent-new_map)<1e-12
        top_path=root/'top12.parquet';con.execute(f'COPY output.top12 TO {we.literal(top_path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
    we.save_parquet(aps,root/'user_ap.parquet')
    result={'window':name,'cutoff':cutoff,'previous_window':previous,'no_op':False,
        'baseline_MAP':checked_base,'MAP':new_map,'population_delta':float(delta.mean()),
        'positive_user_count':int((delta>1e-15).sum()),'harmed_user_count':int((delta< -1e-15).sum()),
        'candidate_rows':rows,'total_users':total,'truth_pairs':len(truth),'candidate_identity_errors':errors,
        'inactive_rank_errors':inactive_errors,'independent_MAP_error':abs(independent-new_map),
        'lag_score_metadata':lag_meta,'top12':evidence_id(top_path,reason='explicit_registry_evidence'),
        'candidate_pool_changed':False,'runtime_seconds':time.perf_counter()-started,'final_week':'not_run'}
    write(output,result);print({'temporal_outer':name,'delta':result['population_delta']},flush=True);return result


def confirm():
    common.setup();common.budget(10)
    entry=next(t for t in read(common.REGISTRY)['trials'] if t['experiment_id']==TRIAL)
    assert entry['outer_exposures']==1 and read(common.REPORT/(TRIAL+'_SCREEN.json'))['screening']['passed']
    destination=common.REPORT/(TRIAL+'_OUTER.json')
    if destination.is_file():return read(destination)
    engine=common.Engine(2020);pred=predecessor_map(engine.contract['rolling_protocol']);started=time.perf_counter()
    rows={name:outer_window(engine,name,p,pred[name]) for name,p in engine.contract['rolling_protocol'].items()}
    result={'experiment_id':TRIAL,'architecture':'causal_previous_checkpoint_rank_ensemble',
        'windows':rows,**common.summary({w:r['MAP'] for w,r in rows.items()},2020),
        'new_training':False,'candidate_pool_changed':False,'runtime_seconds':time.perf_counter()-started,
        'no_registry_mutation':True,'final_week':'not_run'}
    write(destination,result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['screen','confirm'],nargs='?',default='screen')
    globals()[parser.parse_args().command]()
