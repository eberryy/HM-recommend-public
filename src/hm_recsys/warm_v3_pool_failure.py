"""Read-only 2x2 INNER decomposition of frozen/refit models and old/expanded pools."""
from __future__ import annotations

import gc
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from . import warm_v3_common as common
from . import warm_v2_engine as we
from .warm_v2_contract import read, write, now
from .warm_v2_rank_fusion import rank_sql, ap_table
from .warm_v3_pool import ART, INNER_CUTOFFS, connection, PoolEngine
from .warm_v3_pool_experiment import APPROVED, screen_gate

REPORT = common.REPORT / 'POOL_FAILURE_AUDIT.json'


def rerank(con, source, name):
    """Each expert is reranked inside the requested pool, then fixed RRF60."""
    con.execute(f'''CREATE TEMP TABLE {name}_x AS SELECT *,
        {rank_sql('score_e0')} r0,{rank_sql('score_e1')} r1 FROM ({source})''')
    con.execute(f'''CREATE TEMP TABLE {name}_y AS SELECT *,1.0/(60+r0)+1.0/(60+r1) fusion_score FROM {name}_x''')
    con.execute(f'''CREATE TEMP TABLE {name} AS SELECT *,
        CASE WHEN user_history_events_12w=0 THEN candidate_rank
        ELSE {rank_sql('fusion_score')} END rf FROM {name}_y''')


def contributions(con, table, output):
    con.execute(f'''CREATE TEMP TABLE {output} AS WITH h AS (
        SELECT customer_id,article_id,target,truth_count,rf,
        sum(target) OVER(PARTITION BY customer_id ORDER BY rf ROWS UNBOUNDED PRECEDING) hits
        FROM {table}) SELECT customer_id,article_id,rf,
        hits*1.0/rf/least(truth_count,12) contribution FROM h WHERE target=1 AND rf<=12''')


def decompose(con, before, after, users):
    rows = con.execute(f'''WITH p AS (
        SELECT coalesce(a.customer_id,b.customer_id) customer_id,
        coalesce(a.article_id,b.article_id) article_id,
        CASE WHEN a.article_id IS NULL THEN 'new_top12_truth'
             WHEN b.article_id IS NULL THEN 'lost_top12_truth' ELSE 'shared_top12_truth' END component,
        coalesce(b.contribution,0)-coalesce(a.contribution,0) delta
        FROM {before} a FULL JOIN {after} b USING(customer_id,article_id))
        SELECT component,count(*) pairs,count(DISTINCT customer_id) users,sum(delta)/{users} population_delta
        FROM p GROUP BY component ORDER BY component''').fetchdf()
    return rows.to_dict(orient='records')


def one_window(engine, w, screen):
    cutoff = screen['cutoff']
    assert cutoff in INNER_CUTOFFS and screen['final_week'] == 'not_run'
    outdir = ART / 'failure_2x2' / w
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / 'AUDIT.json'
    if meta_path.exists():
        return read(meta_path)
    started = time.perf_counter()
    fp, stat = engine.cached_data(cutoff, 'inner')
    frame = we.load_parquet(fp)
    n = stat['source_users']
    assert n == screen['full_users']
    scored = frame[['customer_id','article_id','candidate_rank','target','truth_count','user_history_events_12w']].copy()
    model_paths = []
    for trial, label, families in (('WV2-000','e0',[]),('WV2-501','e1',['bpr_match'])):
        root = common.OLD / trial / w
        meta = read(root / 'inner_metrics.json')
        assert meta['cutoff'] == cutoff
        path = root / 'inner_model.txt'
        assert path.stat().st_size == meta['model']['bytes']
        model = lgb.Booster(model_file=str(path))
        maps = {k:{int(a):int(b) for a,b in v.items()} for k,v in read(root / 'inner_category_maps.json').items()}
        source = common.attach_features(frame,cutoff,families,engine) if families else frame
        scored['score_'+label] = model.predict(we.prepare(source,model.feature_name(),maps),num_threads=4)
        model_paths.append(str(path))
    with connection() as con:
        con.register('scored',scored)
        con.execute(f'''CREATE TEMP TABLE old_pairs AS SELECT customer_id,article_id FROM read_parquet({we.literal(engine.original.base_path(cutoff))})''')
        old = screen['old_baseline']
        con.execute(f'''CREATE TEMP TABLE A AS SELECT customer_id,article_id,candidate_rank,target,truth_count,user_history_events_12w,
            score_base score_e0,score_bpr score_e1,r0,r1,rf FROM read_parquet({we.literal(old['ranks_path'])})''')
        con.execute(f'''CREATE TEMP TABLE D AS SELECT * FROM read_parquet({we.literal(screen['artifacts']['ranks'])})''')
        rerank(con,'SELECT * FROM scored','B')
        rerank(con,'SELECT s.* EXCLUDE(r0,r1,rrf_score,rf) FROM D s JOIN old_pairs USING(customer_id,article_id)','C')
        # Frozen models must reproduce their previous old-pool scores and final ranks.
        rerank(con,'SELECT s.* FROM scored s JOIN old_pairs USING(customer_id,article_id)','replay')
        errors = con.execute('''SELECT count(*) FILTER(WHERE abs(a.score_e0-r.score_e0)>1e-12 OR abs(a.score_e1-r.score_e1)>1e-12
            OR a.rf<>r.rf OR a.r0<>r.r0 OR a.r1<>r.r1 OR a.target<>r.target OR a.truth_count<>r.truth_count)
            FROM A a FULL JOIN replay r USING(customer_id,article_id) WHERE a.customer_id IS NOT NULL''').fetchone()[0]
        assert errors == 0
        assert con.execute('SELECT count(*) FROM A a LEFT JOIN replay r USING(customer_id,article_id) WHERE r.customer_id IS NULL').fetchone()[0] == 0
        con.execute('''CREATE TEMP TABLE user_ids AS SELECT DISTINCT customer_id FROM D''')
        aps = con.execute('SELECT * FROM user_ids ORDER BY customer_id').fetchdf()
        components = {}
        for name in ('A','B','C','D'):
            one = ap_table(con,name,'rf').rename(columns={'ap':'ap_'+name})
            aps = aps.merge(one,on='customer_id',how='left',validate='one_to_one').fillna({'ap_'+name:0})
            contributions(con,name,'hits_'+name)
            components[name] = float(aps['ap_'+name].sum()/n)
        assert abs(components['A']-old['baseline_map_population_component']) < 1e-12
        assert abs(components['D']-screen['systems']['rrf']['population_map_component']) < 1e-12
        con.register('ap_users',aps)
        contrasts = {}
        for before,after in (('A','B'),('A','C'),('C','D'),('A','D')):
            delta = components[after]-components[before]
            pieces = decompose(con,'hits_'+before,'hits_'+after,n)
            assert abs(sum(p['population_delta'] for p in pieces)-delta) < 1e-12
            contrasts[after+'-'+before] = {'population_delta':delta,'truth_top12_decomposition':pieces}
        con.execute(f'''CREATE TEMP TABLE fresh AS SELECT customer_id,article_id FROM read_parquet({we.literal(engine.base_path(cutoff))}) WHERE bpr_is_new=1''')
        new_hits = {}
        for label in ('B','D'):
            x = con.execute(f'''SELECT count(*),count(DISTINCT h.customer_id),coalesce(sum(contribution),0)/{n}
                FROM hits_{label} h JOIN fresh f USING(customer_id,article_id)''').fetchone()
            new_hits[label] = {'top12_new_source_positive_pairs':x[0],'users':x[1],'population_ap_contribution':x[2]}
        population_parts = con.execute(f'''SELECT a.customer_id IS NULL newly_covered,count(*) users,
            sum(p.ap_B-p.ap_A)/{n} B_minus_A,sum(p.ap_C-p.ap_A)/{n} C_minus_A,
            sum(p.ap_D-p.ap_C)/{n} D_minus_C,sum(p.ap_D-p.ap_A)/{n} D_minus_A
            FROM ap_users p LEFT JOIN (SELECT DISTINCT customer_id FROM A) a USING(customer_id)
            GROUP BY newly_covered ORDER BY newly_covered''').fetchdf().to_dict(orient='records')
        con.execute(f'COPY ap_users TO {we.literal(outdir / "user_ap.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        con.execute(f'''COPY (SELECT customer_id,article_id,candidate_rank,target,truth_count,rf,score_e0,score_e1
            FROM B) TO {we.literal(outdir / 'B_ranks.parquet')} (FORMAT PARQUET,COMPRESSION ZSTD)''')
        con.execute(f'COPY hits_B TO {we.literal(outdir / "B_positive_contributions.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
    result = {'window':w,'cutoff':cutoff,'role':'inner_only_readonly_no_fit','full_users':n,
        'covered_expanded_active_users':len(aps),'models':model_paths,'old_model_score_rank_replay_errors':errors,
        'population_MAP_components':components,'contrasts':contrasts,'new_BPR_truth_hits':new_hits,
        'user_population_decomposition':population_parts,'runtime_seconds':time.perf_counter()-started,
        'artifacts':{'user_ap':str(outdir/'user_ap.parquet'),'B_ranks':str(outdir/'B_ranks.parquet')},
        'outer':'not_run','final_week':'not_run'}
    write(meta_path,result)
    print({'2x2':w,'contrasts':{k:v['population_delta'] for k,v in contrasts.items()},'seconds':result['runtime_seconds']},flush=True)
    del frame,source,scored,aps
    gc.collect()
    return result


def run():
    common.setup()
    common.budget(5)
    started = time.perf_counter()
    screen = read(ART/'SCREEN.json')
    assert screen['screening']['passed'] is False and screen['outer_authorized'] is False
    engine = PoolEngine(allowed_cutoffs=APPROVED,device='cpu')
    rows = {w:one_window(engine,w,s) for w,s in screen['windows'].items()}
    ds = [r['contrasts']['B-A']['population_delta'] for r in rows.values()]
    result = {'created_at':now(),'parent':'WV3-401','diagnostic_only':True,'new_fits':0,
        'design':{'A':'original601models/originalpool','B':'original601models/expandedpool',
                  'C':'401refitmodels/originalpool','D':'401refitmodels/expandedpool'},
        'denominator':'full original10percent truthusers; unchanged inactive fallback cancels; newlycoveredusers oldpool AP zero',
        'windows':rows,'frozen_model_expansion_inner_gate_only':screen_gate(ds),
        'runtime_seconds':time.perf_counter()-started,'outer':'not_run','final_week':'not_run','automatic_promotion':False}
    write(REPORT,result)
    return result


if __name__=='__main__':
    run()
