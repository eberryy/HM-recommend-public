"""WV3-361 fixed fusion of WV2-601 model rank and frozen retrieval candidate rank."""
from __future__ import annotations
import time
import numpy as np
from . import warm_v3_common as common
from .metrics import apk
from .warm_v2_contract import evidence_id,read,write
from .warm_v2_engine import connection,literal,save_parquet
from .warm_v2_rank_fusion import ap_table,rank_sql
from .warm_v3_expert import screening_gate

TRIAL='WV3-361'


def create_ranks(con,source='base'):
    con.execute(f'''CREATE TEMP TABLE prior_score AS SELECT *,
        1.0/(60+baseline_rank)+1.0/(60+candidate_rank) fused_score FROM {source}''')
    con.execute(f'''CREATE TEMP TABLE prior_raw_rank AS SELECT *,{rank_sql('fused_score')} raw_rank FROM prior_score''')
    con.execute('''CREATE TEMP TABLE prior_ranked AS SELECT *,CASE WHEN user_history_events_12w=0
        THEN candidate_rank ELSE raw_rank END final_rank FROM prior_raw_rank''')
    return 'prior_ranked'


def evaluate(window,stage,path,baseline_column,total_users,expected):
    root=common.ART/TRIAL/window/stage;report=root/'REVIEW.json'
    if report.is_file():return read(report)
    root.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    with connection() as con:
        con.execute(f'''CREATE TEMP TABLE base AS SELECT customer_id,article_id,candidate_rank,target,truth_count,
            user_history_events_12w,{baseline_column} baseline_rank FROM read_parquet({literal(path)})''')
        table=create_ranks(con);rows,users=con.execute(f'SELECT count(*),count(DISTINCT customer_id) FROM {table}').fetchone()
        assert 0<users<=total_users
        permutation_errors=con.execute(f'''SELECT count(*) FROM (SELECT customer_id,count(*) n,
            count(DISTINCT final_rank) nr,min(final_rank) lo,max(final_rank) hi FROM {table} GROUP BY customer_id)
            WHERE n<>nr OR lo<>1 OR hi<>n''').fetchone()[0]
        inactive_errors=con.execute(f'''SELECT count(*) FROM {table} WHERE user_history_events_12w=0
            AND final_rank<>candidate_rank''').fetchone()[0]
        assert permutation_errors==inactive_errors==0
        aps=None
        for label,rank in [('baseline','baseline_rank'),('prior','final_rank')]:
            one=ap_table(con,table,rank).rename(columns={'ap':'ap_'+label})
            aps=one if aps is None else aps.merge(one,on='customer_id',validate='one_to_one')
        baseline=float(aps.ap_baseline.sum()/total_users);assert abs(baseline-expected)<1e-12
        score=float(aps.ap_prior.sum()/total_users);delta=aps.ap_prior-aps.ap_baseline
        rank_path=root/'ranks.parquet';con.execute(f'COPY {table} TO {literal(rank_path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
    save_parquet(aps,root/'user_ap.parquet')
    result={'window':window,'stage':stage,'candidate_rows':rows,'included_users':users,'total_users':total_users,
        'baseline_MAP':baseline,'MAP':score,'population_delta':float(delta.sum()/total_users),
        'positive_user_count':int((delta>1e-15).sum()),'harmed_user_count':int((delta< -1e-15).sum()),
        'candidate_pool_changed':False,'permutation_errors':permutation_errors,'inactive_rank_errors':inactive_errors,
        'rank_path':str(rank_path),'runtime_seconds':time.perf_counter()-started,'final_week':'not_run'}
    write(report,result);print({'prior_fusion':stage,'window':window,'delta':result['population_delta']},flush=True);return result


def screen():
    common.setup();common.budget(5);entry=next(t for t in read(common.REGISTRY)['trials'] if t['experiment_id']==TRIAL)
    assert entry['decision']=='preregistered' and entry['outer_exposures']==0
    engine=common.Engine(2020);started=time.perf_counter();rows={}
    for window,p in engine.contract['rolling_protocol'].items():
        cutoff=p['inner_validation'];meta=read(common.ART/'gate_data'/f'2020_inner_{cutoff}'/'DATA.json')
        rows[window]=evaluate(window,'inner',meta['ranks_path'],'ap_rf',meta['total_users'],meta['baseline_map_population_component'])
    gate=screening_gate(v['population_delta'] for v in rows.values())
    result={'experiment_id':TRIAL,'architecture':'retrieval_prior_regularized_rank_fusion','windows':rows,
        'screening':gate,'new_training':False,'candidate_pool_changed':False,'runtime_seconds':time.perf_counter()-started,
        'no_registry_mutation':True,'final_week':'not_run'}
    write(common.REPORT/(TRIAL+'_SCREEN.json'),result);print(gate,flush=True);return result


def confirm():
    common.setup();common.budget(10);entry=next(t for t in read(common.REGISTRY)['trials'] if t['experiment_id']==TRIAL)
    assert entry['outer_exposures']==1 and read(common.REPORT/(TRIAL+'_SCREEN.json'))['screening']['passed']
    destination=common.REPORT/(TRIAL+'_OUTER.json')
    if destination.is_file():return read(destination)
    started=time.perf_counter();rows={};baseline=read(common.ROOT/'reports/warm_v2/WV2-601_OUTER.json')['per_window_MAP']
    for window in common.contracts()['rolling_protocol']:
        review=read(common.ART/'WV3-331'/window/'MULTIVIEW_OUTER_REVIEW.json')
        path=common.ART/'WV3-331'/window/'outer_multiview_ranks.parquet'
        rows[window]=evaluate(window,'outer',path,'baseline_rank',review['full_users'],baseline[window])
    maps={w:r['MAP'] for w,r in rows.items()}
    result={'experiment_id':TRIAL,'architecture':'retrieval_prior_regularized_rank_fusion',
        'windows':rows,**common.summary(maps,2020),'new_training':False,'candidate_pool_changed':False,
        'runtime_seconds':time.perf_counter()-started,'no_registry_mutation':True,'final_week':'not_run'}
    write(destination,result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['screen','confirm'],nargs='?',default='screen')
    globals()[parser.parse_args().command]()
