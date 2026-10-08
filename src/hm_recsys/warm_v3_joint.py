"""88-feature joint BPR+sequence expert and fixed E0/joint two-way RRF; INNER only."""
from __future__ import annotations
import gc
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from . import warm_v3_common as common
from . import warm_v2_engine as we
from .warm_v2_contract import read,write,now
from .warm_v2_rank_fusion import rank_sql,ap_table
from .warm_v3_expert import screening_gate
from .warm_v3_retrieval_audit import CUTOFFS as INNER_CUTOFFS

STANDALONE='WV3-321'
FUSION='WV3-322'
FAMILIES=['bpr_match','sequence']
EXTRA=['wv2_bpr_user_item_score','wv2_bpr_unavailable','wv3_sequence_score','wv3_sequence_unavailable']
CANONICAL_PARAMS={'architecture':'joint88_BPR_sequence_LambdaRank',
    'families':FAMILIES,'extra_columns':EXTRA,'feature_count':88,
    'source_selection':'sequence highest four-inner conditional correction of rawBPR-missed AP weight; noouterselection',
    'new_representation_training':False,'candidate_pool':'original frozen100-300',
    'tree_parameters':'exact original84anchor parameters; no overrides',
    'sampling':'original30:1 two-strata distribution-aware, allpositives, noIPW',
    'early_stopping':'originalinner MAP12,max200,patience20',
    'ordering_321':'joint88 score','ordering_322':'1/(60+E0rank)+1/(60+joint88rank)',
    'fusion_threeway':False,'inactive':'exact originalcandidate_rank fallback',
    'full_truth_denominator':True,'inner_gate':{'mean_min':.0001,'positive_min':3,'worst_min':-.0005},
    'only_root_selects_and_exposes_outer':True,'final_week':'not_run'}


class JointEngine(common.Engine):
    def signal_path(self,family,cutoff):
        # Cached-only path: never invokes sequence.fit, even if a cache is missing.
        if family=='bpr_match':return common.bpr_path(cutoff)
        if family!='sequence':return None
        root=common.ART/'sequence'/cutoff
        path=root/'features.parquet'
        if not path.is_file() or not (root/'FEATURES.json').is_file() or not (root/'MODEL.json').is_file():
            raise RuntimeError('Joint expert requires an existing frozen sequence cache; no representation fitting allowed')
        meta=read(root/'FEATURES.json');model=read(root/'MODEL.json')
        assert meta['cutoff']==model['cutoff']==cutoff
        assert meta['candidate_source']==str(self.base_path(cutoff).resolve())
        assert meta['candidate_identity_unchanged']
        assert meta['features']==['wv3_sequence_score','wv3_sequence_unavailable']
        assert model['preparation']['history_strictly_before_cutoff']
        assert model['preparation']['latest_history_date']<cutoff
        return path,meta


def ranks(con,source):
    con.execute(f'''CREATE TEMP TABLE jx AS SELECT *,{rank_sql('score_base')} r0,
        {rank_sql('score_joint')} rj FROM {source}''')
    con.execute('CREATE TEMP TABLE jy AS SELECT *,1.0/(60+r0)+1.0/(60+rj) fusion_score FROM jx')
    con.execute(f'''CREATE TEMP TABLE jz AS SELECT *,{rank_sql('fusion_score')} r2 FROM jy''')
    con.execute('''CREATE TEMP TABLE joint_ranked AS SELECT *,
        CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE rj END standalone_rank,
        CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE r2 END fusion_rank FROM jz''')
    return 'joint_ranked'


def summarize(aps,total_users):
    assert 0<len(aps)<=total_users
    systems={}
    for label in ('standalone','fusion'):
        d=aps['ap_'+label]-aps.ap_baseline
        systems[label]={'population_delta':float(d.sum()/total_users),
            'population_map_component':float(aps['ap_'+label].sum()/total_users),
            'covered_active_MAP':float(aps['ap_'+label].mean()),
            'improved_users':int((d>1e-15).sum()),'harmed_users':int((d< -1e-15).sum()),
            'gross_positive_MAP':float(d[d>0].sum()/total_users),'gross_negative_MAP':float(d[d<0].sum()/total_users)}
    return systems


def inner_window(engine,w,p):
    started=time.perf_counter()
    cutoff=p['inner_validation']
    assert cutoff in INNER_CUTOFFS,'Only four original INNER validation cutoffs are authorized here'
    root=common.ART/STANDALONE/w
    ready=root/'JOINT_INNER_REVIEW.json'
    if ready.exists():return read(ready)
    # All producer sources must already exist. Same pool/sample mechanics inherited unchanged.
    for c in p['inner_train']+[cutoff]:
        for family in FAMILIES:engine.signal_path(family,c)
    metadata=root/'inner_metrics.json'
    we.ARTIFACT=common.ART
    fit=read(metadata) if metadata.exists() else engine.train_inner(w,STANDALONE,FAMILIES,EXTRA)
    assert fit['cutoff']==cutoff and fit['features']==engine.features+EXTRA
    assert fit['feature_count']==len(engine.features)+len(EXTRA)
    assert fit['families']==FAMILIES
    assert Path(fit['model']['path']).resolve()==(root/'inner_model.txt').resolve()
    fp,stat=engine.cached_data(cutoff,'inner')
    frame=we.load_parquet(fp)
    model=lgb.Booster(model_file=str(root/'inner_model.txt'))
    maps={k:{int(a):int(b) for a,b in v.items()} for k,v in read(root/'inner_category_maps.json').items()}
    ff=common.attach_features(frame,cutoff,FAMILIES,engine)
    scores=frame[['customer_id','article_id','candidate_rank','target','truth_count','user_history_events_12w']].copy()
    scores['score_joint']=model.predict(we.prepare(ff,model.feature_name(),maps),num_threads=4)
    assert np.isfinite(scores.score_joint).all()
    baseline=read(common.ART/'gate_data'/f'2020_inner_{cutoff}'/'DATA.json')
    assert baseline['cutoff']==cutoff and baseline['total_users']==stat['source_users']
    with we.connection() as con:
        con.register('newscore',scores)
        con.execute(f'CREATE TEMP TABLE old AS SELECT * FROM read_parquet({we.literal(baseline["ranks_path"])})')
        assert con.execute('''SELECT count(*) FROM old o FULL JOIN newscore n USING(customer_id,article_id)
            WHERE o.customer_id IS NULL OR n.customer_id IS NULL OR o.target<>n.target OR o.candidate_rank<>n.candidate_rank
            OR o.truth_count<>n.truth_count OR o.user_history_events_12w<>n.user_history_events_12w''').fetchone()[0]==0
        con.execute(f'''CREATE TEMP TABLE truth AS SELECT DISTINCT customer_id,article_id FROM read_parquet({we.literal(common.TX)})
            WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY AND hash(customer_id)%1000000<100000''')
        con.execute('CREATE TEMP TABLE truth_users AS SELECT customer_id,count(*) truth_count FROM truth GROUP BY customer_id')
        con.execute(f'CREATE VIEW fullpool AS SELECT * FROM read_parquet({we.literal(engine.base_path(cutoff))})')
        errors=con.execute('''SELECT count(*) FROM (SELECT DISTINCT customer_id FROM fullpool) f FULL JOIN truth_users t USING(customer_id)
            WHERE f.customer_id IS NULL OR t.customer_id IS NULL''').fetchone()[0]
        assert errors==0 and con.execute('SELECT count(*) FROM truth_users').fetchone()[0]==stat['source_users']
        assert con.execute('''SELECT count(*) FROM fullpool p LEFT JOIN truth t USING(customer_id,article_id)
            WHERE p.target<>CAST(t.article_id IS NOT NULL AS INTEGER)''').fetchone()[0]==0
        assert con.execute('''SELECT count(*) FROM newscore n JOIN truth_users t USING(customer_id) WHERE n.truth_count<>t.truth_count''').fetchone()[0]==0
        con.execute('''CREATE TEMP TABLE merged AS SELECT n.*,o.score_base,o.score_bpr,o.ap_rf baseline_rank
            FROM newscore n JOIN old o USING(customer_id,article_id)''')
        table=ranks(con,'merged')
        aps=None
        for name,rank in [('baseline','baseline_rank'),('standalone','standalone_rank'),('fusion','fusion_rank')]:
            one=ap_table(con,table,rank).rename(columns={'ap':'ap_'+name})
            aps=one if aps is None else aps.merge(one,on='customer_id',validate='one_to_one')
        assert abs(aps.ap_baseline.sum()/stat['source_users']-baseline['baseline_map_population_component'])<1e-12
        assert abs(aps.ap_standalone.mean()-fit['inner_covered_active_map'])<1e-12
        assert con.execute('SELECT count(*) FROM joint_ranked j JOIN old o USING(customer_id,article_id) WHERE j.r0<>o.r0').fetchone()[0]==0
        con.execute(f'COPY {table} TO {we.literal(root/"inner_joint_ranks.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
    we.save_parquet(aps,root/'inner_user_ap.parquet')
    systems=summarize(aps,stat['source_users'])
    result={'window':w,'cutoff':cutoff,'role':'original_inner_only','fit':fit,'systems':systems,
        'total_users':stat['source_users'],'covered_active_users':len(aps),'source_identity_errors':0,
        'raw_full_population_and_truth_checked':True,'original601_metric_replayed':True,
        'ranks_path':str(root/'inner_joint_ranks.parquet'),'runtime_seconds':time.perf_counter()-started,'final_week':'not_run'}
    write(ready,result)
    print({'joint88_inner':w,'systems':{k:v['population_delta'] for k,v in systems.items()},'seconds':result['runtime_seconds']},flush=True)
    del frame,ff,scores,aps,model
    gc.collect()
    return result


def screen():
    common.setup();common.budget(5)
    registered={r['experiment_id'] for r in read(common.REGISTRY)['trials']}
    if not {STANDALONE,FUSION}<=registered:raise RuntimeError('Root must preregister both joint INNER orderings before training')
    started=time.perf_counter()
    engine=JointEngine(2020)
    assert len(engine.features)==84
    rows={w:inner_window(engine,w,p) for w,p in engine.contract['rolling_protocol'].items()}
    result={}
    for label,trial in (('standalone',STANDALONE),('fusion',FUSION)):
        reduced={w:{**r['systems'][label],'fit':r['fit'],'ranks_path':r['ranks_path'],
                    'raw_full_population_and_truth_checked':r['raw_full_population_and_truth_checked'],
                    'original601_metric_replayed':r['original601_metric_replayed'],
                    'total_users':r['total_users'],'covered_active_users':r['covered_active_users']} for w,r in rows.items()}
        value={'experiment_id':trial,'architecture':CANONICAL_PARAMS['architecture'],'ordering':label,'canonical_params':CANONICAL_PARAMS,
            'producer_model_trial':STANDALONE,'producer_features':engine.features+EXTRA,'windows':reduced,
            'screening':screening_gate(r['population_delta'] for r in reduced.values()),'candidate_pool_changed':False,
            'runtime_seconds':time.perf_counter()-started,'no_registry_mutation':True,'final_week':'not_run',
            'no_outer_authorization':True}
        write(common.REPORT/(trial+'_SCREEN.json'),value)
        result[trial]=value
    print({t:r['screening'] for t,r in result.items()},flush=True)
    return result


if __name__=='__main__':screen()
