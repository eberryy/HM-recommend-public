"""Audited, fixed equal-weight ranker fusion; no parameter search or new fitting."""
from __future__ import annotations

import argparse
import time
import lightgbm as lgb
import numpy as np

from .warm_v2_contract import ARTIFACT,REPORT,assert_branch,guard,now,read,write,register,record,evidence_id
from .warm_v2_engine import Engine,connection,literal,load_parquet,prepare
from .warm_v2_features import attach_features
from .warm_v2_lab import log,spec

TRIAL='WV2-601'
ROOT=ARTIFACT/'ranker-fusion-v1'


def contract():
    p=REPORT/'WV2_601_CONTRACT.json'
    if p.exists():return read(p)
    bpr=read(REPORT/'WV2-501_OUTER.json')
    assert bpr['decision']=='reject' and bpr['non_degrade_windows']==3 and bpr['delta_vs_Warm_v1']>0
    c={'registered_at':now(),'stage':'WV2.7 fixed two-ranker fusion','trial':TRIAL,
        'evidence':'501 improves outer mean and3windows but fails worst-window gate; upstream feature training is frozen, no BPR parameter/source changes.',
        'hypothesis':'Different Top12 errors of baseline and learned-match rankers may admit a scale-independent conservative combination.',
        'audit_gate':'In at least3of4 original inner windows, BOTH rankers have>=25 positive candidate pairs in their ownTop12 that the other ranker places below12.',
        'fusion':'equal weights:1/(60+baseline_model_rank)+1/(60+BPR_model_rank); each rank over unchanged full user candidate pool; ties use original candidate_rank,article_id.60 is existing projectRRF default, not selected from scores.',
        'not_retrieval':'These are two learned rankers, not source recall lanes. Candidate pair set and all source features remain unchanged.',
        'scope':'same inner and outer dates, original10% users, complete truth denominator,Top12; inactive users still original candidate_rank fallback.',
        'inner_gate':{'mean_min':.0001,'positive_windows_min':3,'worst_min':-.0005},
        'outer_gate':{'mean_min':.0005,'nondegrade_min':3,'worst_min':-.0003,'large_gain_review':.001},
        'budget':'one fixed combination; zero new model fits,5-10min CPU8; at most one new outer comparison only after complementarity audit and inner gate.',
        'exposure':'Independent ranking-strategy stage under renewed autonomous research authorization. Would be fifth new outer comparison in this branch, not a reclassification or extra parameter trial forBPR. Old feature/BPR failures and their stage budgets stay closed. Reused development windows remain subject to adaptive-selection bias.',
        'failure':'No weight/constant/season-router sweep; keepWV2-000 if rejected. Investigate a different bottleneck, not small outer-driven fusion adjustments.',
        'final_week':'not_run','automatic_merge_or_push':False}
    write(p,c);return c


def rank_sql(score):
    return f'row_number() OVER(PARTITION BY customer_id ORDER BY {score} DESC,candidate_rank,article_id)'


def final_rank_sql():
    return '''row_number() OVER(PARTITION BY customer_id ORDER BY
        CASE WHEN user_history_events_12w=0 THEN candidate_rank END ASC NULLS LAST,
        CASE WHEN user_history_events_12w>0 THEN score END DESC NULLS LAST,candidate_rank,article_id)::INTEGER'''


def ap_table(con,table,rank):
    return con.execute(f'''WITH a AS (SELECT customer_id,target,{rank} r,truth_count,
        sum(target) OVER(PARTITION BY customer_id ORDER BY {rank} ROWS UNBOUNDED PRECEDING) hits FROM {table})
        SELECT customer_id,sum(CASE WHEN r<=12 THEN target*hits*1.0/r ELSE 0 END)/least(max(truth_count),12) ap
        FROM a GROUP BY customer_id ORDER BY customer_id''').fetchdf()


def inner_ranks(w,engine):
    p=engine.contract['rolling_protocol'][w];cutoff=p['inner_validation']
    path=ROOT/w/'inner_ranks.parquet';meta=path.with_suffix('.json')
    if meta.exists():return path,read(meta)
    fp,stats=engine.cached_data(cutoff,'inner');f=load_parquet(fp)
    out=f[['customer_id','article_id','candidate_rank','target','truth_count']].copy()
    for trial,label,families in [('WV2-000','base',[]),('WV2-501','bpr',['bpr_match'])]:
        model=lgb.Booster(model_file=str(ARTIFACT/trial/w/'inner_model.txt'))
        maps=read(ARTIFACT/trial/w/'inner_category_maps.json');maps={k:{int(a):int(b) for a,b in v.items()} for k,v in maps.items()}
        ff=attach_features(f,cutoff,families,engine) if families else f
        out['score_'+label]=model.predict(prepare(ff,model.feature_name(),maps),num_threads=8)
    with connection() as con:
        con.register('f',out)
        con.execute(f'CREATE TEMP TABLE ranked AS SELECT *,{rank_sql("score_base")} base_rank,{rank_sql("score_bpr")} bpr_rank FROM f')
        path.parent.mkdir(parents=True,exist_ok=True)
        con.execute(f'COPY ranked TO {literal(path)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        old=float(ap_table(con,'ranked','base_rank').ap.mean());new=float(ap_table(con,'ranked','bpr_rank').ap.mean())
        reference=read(REPORT/'WARM_V1_REPRODUCTION.json')['windows'][w]['inner']['inner_covered_active_map']
        bref=read(REPORT/'WV2-501_SCREEN.json')['windows'][w]['inner_covered_active_map']
        assert abs(old-reference)<1e-12 and abs(new-bref)<1e-12
        unique=con.execute('''SELECT count(*) FILTER(WHERE target=1 AND base_rank<=12 AND bpr_rank>12),
            count(*) FILTER(WHERE target=1 AND bpr_rank<=12 AND base_rank>12),
            count(*) FILTER(WHERE target=1 AND bpr_rank<=12 AND base_rank<=12) FROM ranked''').fetchone()
    result={'cutoff':cutoff,'base_only_correct_top12_pairs':unique[0],'bpr_only_correct_top12_pairs':unique[1],
        'shared_correct_top12_pairs':unique[2],'population_weight':stats['groups']/stats['source_users'],
        'base_inner_map':old,'bpr_inner_map':new,'exact_saved_inner_model_metric_parity':True,
        'artifact':evidence_id(path,reason='explicit_registry_evidence')}
    write(meta,result);return path,result


def screen():
    assert_branch();contract();c=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json');engine=Engine(c)
    output=REPORT/f'{TRIAL}_SCREEN.json'
    if output.exists():return read(output)
    entry=spec(engine,TRIAL,[],[],'fixed_ranker_fusion_screen','冻结两个排序器，审计互补错误后试一次等权RRF，不搜索权重')
    entry.update(research_stage='WV2.7',fusion_contract=str(REPORT/'WV2_601_CONTRACT.json'),parent_experiment='WV2-000 + WV2-501')
    register(entry);start=time.perf_counter();audits={};paths={}
    for w in c['rolling_protocol']:paths[w],audits[w]=inner_ranks(w,engine)
    gate=sum(a['base_only_correct_top12_pairs']>=25 and a['bpr_only_correct_top12_pairs']>=25 for a in audits.values())>=3
    write(REPORT/'WV2_600_COMPLEMENTARITY_AUDIT.json',{'registered_contract':str(REPORT/'WV2_601_CONTRACT.json'),
        'windows':audits,'gate_passed':gate,'denominator':'positive user-item candidate pairs in original covered-active historical validation groups; exclusive means one MODEL Top12 only, not one recall source','final_week':'not_run'})
    if not gate:
        record(TRIAL,{'status':'audit_rejected','decision':'reject','final_week':'not_run'});log();return
    results={}
    for w,path in paths.items():
        with connection() as con:
            con.execute(f'CREATE TEMP TABLE x AS SELECT *,1.0/(60+base_rank)+1.0/(60+bpr_rank) fusion_score FROM read_parquet({literal(path)})')
            con.execute(f'CREATE TEMP TABLE combined AS SELECT *,{rank_sql("fusion_score")} fusion_rank FROM x')
            value=float(ap_table(con,'combined','fusion_rank').ap.mean())
        a=audits[w];delta=(value-a['base_inner_map'])*a['population_weight']
        results[w]={'inner_map':value,'population_delta_vs_v1':delta,
            'population_delta_vs_bpr':(value-a['bpr_inner_map'])*a['population_weight'],'population_weight':a['population_weight']}
        print(w,results[w],flush=True)
    ds=[r['population_delta_vs_v1'] for r in results.values()]
    summary={'mean_population_delta':float(np.mean(ds)),'positive_windows':sum(d>0 for d in ds),'worst_population_delta':min(ds),
        'passed':bool(np.mean(ds)>=.0001 and sum(d>0 for d in ds)>=3 and min(ds)>=-.0005)}
    result={'experiment_id':TRIAL,'families':[],'windows':results,'screening':summary,'runtime_seconds':time.perf_counter()-start,'final_week':'not_run'}
    write(output,result);record(TRIAL,{'status':'screened','screening':summary,'decision':'retain_for_ablation' if summary['passed'] else 'reject','result_path':str(output)});log();return result


def confirm():
    assert_branch();assert read(REPORT/f'{TRIAL}_SCREEN.json')['screening']['passed'];contract()
    output=REPORT/f'{TRIAL}_OUTER.json'
    if output.exists():return read(output)
    c=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json');engine=Engine(c)
    registry=read(REPORT/'WARM_EXPERIMENT_REGISTRY.json');entry=next(t for t in registry['trials'] if t['experiment_id']==TRIAL)
    assert registry['current_champion']=='WV2-000'
    if not entry['outer_confirmations']:
        assert registry['outer_confirmation_counts'].get('fixed_ranker_rrf',0)==0
        entry.update(outer_confirmations=1,status='outer_preregistered',outer_registered_at=now())
        registry['outer_confirmation_counts']['fixed_ranker_rrf']=1;write(REPORT/'WARM_EXPERIMENT_REGISTRY.json',registry)
    b=read(REPORT/'WARM_V1_REPRODUCTION.json');results={};start=time.perf_counter()
    for w,p in c['rolling_protocol'].items():
        root=ARTIFACT/TRIAL/w;root.mkdir(parents=True,exist_ok=True)
        with connection() as con:
            con.execute(f'ATTACH {literal(ARTIFACT/"WV2-000"/w/"evaluation.duckdb")} AS base (READ_ONLY)')
            con.execute(f'ATTACH {literal(ARTIFACT/"WV2-501"/w/"evaluation.duckdb")} AS learned (READ_ONLY)')
            # Fail closed on source pair/label/rank mismatch before fusion.
            check=con.execute('''SELECT count(*) FROM base.predictions b FULL JOIN learned.predictions n USING(customer_id,article_id)
                WHERE b.customer_id IS NULL OR n.customer_id IS NULL OR b.target<>n.target OR b.candidate_rank<>n.candidate_rank OR b.user_history_events_12w<>n.user_history_events_12w''').fetchone()[0]
            assert check==0
            con.execute(f'''CREATE TEMP TABLE merged AS SELECT b.* EXCLUDE(score),b.score score_base,n.score score_bpr
                FROM base.predictions b JOIN learned.predictions n USING(customer_id,article_id)''')
            con.execute(f'CREATE TEMP TABLE ranks AS SELECT *,{rank_sql("score_base")} base_rank,{rank_sql("score_bpr")} bpr_rank FROM merged')
            con.execute('CREATE TEMP TABLE scored AS SELECT *,1.0/(60+base_rank)+1.0/(60+bpr_rank) score FROM ranks')
            # Materialize exactly the schema expected by the independent reviewer.
            con.execute(f'ATTACH {literal(root/"evaluation.duckdb")} AS outdb')
            con.execute('CREATE TABLE outdb.predictions AS SELECT customer_id,article_id,candidate_rank,target,user_history_events_12w,score FROM scored')
            con.execute(f'CREATE TABLE outdb.top12 AS SELECT * FROM (SELECT *,{final_rank_sql()} final_rank FROM outdb.predictions) WHERE final_rank<=12')
            cutoff=guard(p['outer_validation'])
            con.execute(f"CREATE TEMP TABLE tc AS SELECT customer_id,count(DISTINCT article_id) truth_count FROM read_parquet({literal(engine.transactions)}) WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY GROUP BY customer_id")
            con.execute('CREATE TEMP TABLE final_ranks AS SELECT t.*,tc.truth_count FROM outdb.top12 t JOIN tc USING(customer_id)')
            value=float(ap_table(con,'final_ranks','final_rank').ap.mean())
            rows,users=con.execute('SELECT count(*),count(DISTINCT customer_id) FROM outdb.predictions').fetchone()
            assert users==b['windows'][w]['outer']['evaluation']['users']
            assert con.execute('SELECT count(*) FROM outdb.top12').fetchone()[0]==12*users
            assert con.execute('SELECT count(*) FROM (SELECT customer_id FROM final_ranks GROUP BY customer_id)').fetchone()[0]==users
            # Identity/label equality above proves these candidate-only metrics unchanged.
            reference=b['windows'][w]['outer']['evaluation']
            candidate_metrics={k:reference[k] for k in ('candidate_recall','candidate_oracle_map@12','truth_pairs')}
            top=root/'top12.parquet';con.execute(f'COPY (SELECT * FROM outdb.top12 ORDER BY customer_id,final_rank) TO {literal(top)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        results[w]={'evaluation':{'map@12':value,'candidate_rows':rows,'users':users,**candidate_metrics,'top12':evidence_id(top,reason='explicit_registry_evidence')},
            'feature_importance':[],'producer_models':['WV2-000','WV2-501'],'fits':0}
    vals={w:r['evaluation']['map@12'] for w,r in results.items()};ds={w:v-b['per_window_MAP'][w] for w,v in vals.items()}
    mean=float(np.mean(list(vals.values())));delta=mean-b['mean_MAP'];stable=sum(d>=0 for d in ds.values())>=3 and min(ds.values())>=-.0003
    decision='milestone' if (stable and delta>=.0005) or delta>=.001 else 'new_champion' if stable and delta>0 else 'reject'
    result={'experiment_id':TRIAL,'families':[],'windows':results,'per_window_MAP':vals,'per_window_delta':ds,'mean_MAP':mean,
        'delta_vs_Warm_v1':delta,'delta_vs_current_champion':delta,'non_degrade_windows':sum(d>=0 for d in ds.values()),'worst_window_delta':min(ds.values()),
        'stable':stable,'decision':decision,'runtime_seconds':time.perf_counter()-start,'candidate_pool_changed':False,'training_protocol_changed':False,
        'training_protocol_changed_scope':'No new fits in this fusion trial; producer501 added upstream BPR training and two ranker columns',
        'ranking_policy_changed':'equal-weightRRF of two frozen rankers, fixed60; inactive fallback unchanged','final_week':'not_run'}
    write(output,result);record(TRIAL,{'status':'completed','result_path':str(output),**{k:v for k,v in result.items() if k not in ('windows','families')}})
    if stable and delta>0:
        registry=read(REPORT/'WARM_EXPERIMENT_REGISTRY.json');registry['current_champion']=TRIAL;write(REPORT/'WARM_EXPERIMENT_REGISTRY.json',registry)
    log();print({k:result[k] for k in ('mean_MAP','per_window_delta','decision')},flush=True);return result


def audit():
    """Independent pandas rank replay of ALL frozen scores, not only final MAP."""
    assert_branch();c=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json');results={}
    for w in c['rolling_protocol']:
        with connection() as con:
            for alias,trial in [('b','WV2-000'),('n','WV2-501'),('f',TRIAL)]:
                con.execute(f'ATTACH {literal(ARTIFACT/trial/w/"evaluation.duckdb")} AS {alias} (READ_ONLY)')
            frame=con.execute('''SELECT b.customer_id,b.article_id,b.candidate_rank,b.score old_score,n.score learned_score,f.score observed_score
                FROM b.predictions b JOIN n.predictions n USING(customer_id,article_id)
                JOIN f.predictions f USING(customer_id,article_id) ORDER BY b.customer_id,b.candidate_rank,b.article_id''').fetchdf()
            # method=first follows the independently materialized stable tie order.
            rb=frame.groupby('customer_id',sort=False).old_score.rank(method='first',ascending=False)
            rn=frame.groupby('customer_id',sort=False).learned_score.rank(method='first',ascending=False)
            expected=1/(60+rb.to_numpy())+1/(60+rn.to_numpy())
            error=float(np.max(np.abs(expected-frame.observed_score.to_numpy())))
            assert error<1e-15
            invalid=con.execute('''SELECT count(*) FROM (SELECT customer_id,count(*) n,count(DISTINCT article_id) unique_items,
                min(final_rank) lo,max(final_rank) hi FROM f.top12 GROUP BY customer_id) WHERE n<>12 OR unique_items<>12 OR lo<>1 OR hi<>12''').fetchone()[0]
            assert invalid==0
            results[w]={'candidate_rows':len(frame),'max_fusion_score_difference':error,'invalid_top12_users':invalid,
                'method':'pandas per-user stable rank on both saved producer scores, full candidate population; no labels used'}
            del frame,rb,rn,expected
    output={'created_at':now(),'status':'passed','windows':results,'no_training':True,'final_week':'not_run'}
    write(REPORT/'WV2-601_FUSION_REPLAY.json',output);print('full fusion replay passed',flush=True);return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['screen','confirm','audit']);a=p.parse_args()
    {'screen':screen,'confirm':confirm,'audit':audit}[a.command]()
