"""User-level expert selection: historical oracle, safe meta-data, two inner-only screens."""
from __future__ import annotations
import argparse
from datetime import date,timedelta
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from .metrics import apk
from .warm_v3_common import *
from .warm_v2_engine import connection,literal,load_parquet,save_parquet,prepare
from .warm_v2_rank_fusion import rank_sql,ap_table

FEATURES=['history_events','unique_items','recency','bpr_available','bpr_mean','bpr_std','bpr_margin',
          'overlap12','overlap20','disagreement','rank_corr','e0_std','e1_std','e0_margin','e1_margin','candidate_count']
EXPERTS=['e0','e1','rrf']

def verify_outer_sources(w,year,cutoff):
    root=FRESH if year==2019 else OLD;prefix='FRESH' if year==2019 else 'WV2'
    with connection() as con:
        for alias,t in [('b','000'),('n','501')]:
            con.execute(f'ATTACH {literal(root/f"{prefix}-{t}"/w/"evaluation.duckdb")} AS {alias} (READ_ONLY)')
        mismatch=con.execute('''SELECT count(*) FROM b.predictions b FULL JOIN n.predictions n USING(customer_id,article_id)
            WHERE b.customer_id IS NULL OR n.customer_id IS NULL OR b.target<>n.target
            OR b.candidate_rank<>n.candidate_rank OR b.user_history_events_12w<>n.user_history_events_12w''').fetchone()[0]
        assert mismatch==0
        con.execute(f'''CREATE TEMP TABLE truth AS SELECT DISTINCT customer_id,article_id FROM read_parquet({literal(TX)})
            WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY AND hash(customer_id)%1000000<100000''')
        population=con.execute('''SELECT count(*) FROM (SELECT DISTINCT customer_id FROM b.predictions) b
            FULL JOIN (SELECT DISTINCT customer_id FROM truth) t USING(customer_id) WHERE b.customer_id IS NULL OR t.customer_id IS NULL''').fetchone()[0]
        labels=con.execute('''SELECT count(*) FROM b.predictions b LEFT JOIN truth t USING(customer_id,article_id)
            WHERE b.target<>CASE WHEN t.article_id IS NULL THEN 0 ELSE 1 END''').fetchone()[0]
        duplicates=con.execute('SELECT count(*)-count(DISTINCT (customer_id,article_id)) FROM b.predictions').fetchone()[0]
        assert population==labels==duplicates==0
    return {'source_identity_errors':0,'raw_hash_population_errors':0,'raw_label_errors':0,'duplicate_pairs':0}

def dataset(w,year,role):
    e=Engine(year);p=e.contract['rolling_protocol'][w];cutoff=p['inner_validation' if role=='inner' else 'outer_validation']
    guard(cutoff);root=ART/'gate_data'/f'{year}_{role}_{cutoff}';meta=root/'DATA.json'
    if meta.exists():
        m=read(meta)
        if role=='outer' and 'independent_source_check' not in m:
            m['independent_source_check']=verify_outer_sources(w,year,cutoff);write(meta,m)
        return m
    budget(2);start=time.perf_counter();root.mkdir(parents=True,exist_ok=True)
    bp,bm=bpr_path(cutoff)
    cols=['user_history_events_12w','user_unique_items_12w','user_days_since_last_purchase']
    if role=='inner':
        fp,stats=e.cached_data(cutoff,'inner');f=load_parquet(fp)
        out=f[['customer_id','article_id','candidate_rank','target','truth_count',*cols]].copy()
        for trial,label,families in [('WV2-000','base',[]),('WV2-501','bpr',['bpr_match'])]:
            model=lgb.Booster(model_file=str(OLD/trial/w/'inner_model.txt'))
            maps=read(OLD/trial/w/'inner_category_maps.json');maps={k:{int(a):int(b) for a,b in v.items()} for k,v in maps.items()}
            ff=attach_features(f,cutoff,families,e) if families else f
            out['score_'+label]=model.predict(prepare(ff,model.feature_name(),maps),num_threads=8)
        total_users=stats['source_users']
    else:
        old=FRESH if year==2019 else OLD;prefix='FRESH' if year==2019 else 'WV2'
        with connection() as con:
            con.execute(f'ATTACH {literal(old/f"{prefix}-000"/w/"evaluation.duckdb")} AS b (READ_ONLY)')
            con.execute(f'ATTACH {literal(old/f"{prefix}-501"/w/"evaluation.duckdb")} AS n (READ_ONLY)')
            con.execute(f'CREATE VIEW features AS SELECT customer_id,article_id,user_unique_items_12w,user_days_since_last_purchase FROM read_parquet({literal(e.base_path(cutoff))})')
            out=con.execute(f'''WITH t AS (SELECT customer_id,count(DISTINCT article_id) truth_count FROM read_parquet({literal(TX)})
                WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY GROUP BY customer_id)
                SELECT b.* EXCLUDE(score),b.score score_base,n.score score_bpr,f.user_unique_items_12w,f.user_days_since_last_purchase,t.truth_count
                FROM b.predictions b JOIN n.predictions n USING(customer_id,article_id)
                JOIN features f USING(customer_id,article_id) JOIN t USING(customer_id)''').fetchdf()
            assert len(out)==con.execute('SELECT count(*) FROM b.predictions').fetchone()[0]
        total_users=out.customer_id.nunique()
    with connection() as con:
        con.register('out',out)
        con.execute(f'''CREATE TEMP TABLE x AS SELECT o.*,s.wv2_bpr_user_item_score latent,s.wv2_bpr_unavailable missing,
            {rank_sql('score_base')} r0,{rank_sql('score_bpr')} r1
            FROM out o JOIN read_parquet({literal(bp)}) s USING(customer_id,article_id)''')
        assert con.execute('SELECT count(*) FROM x').fetchone()[0]==len(out)
        con.execute('CREATE TEMP TABLE y AS SELECT *,1.0/(60+r0)+1.0/(60+r1) rrf_score FROM x')
        con.execute(f'''CREATE TEMP TABLE z AS SELECT *,{rank_sql('rrf_score')} rf,
            {rank_sql('coalesce(latent,-1e30)')} rl FROM y''')
        con.execute('''CREATE TEMP TABLE a AS SELECT *,
            CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE r0 END ap_r0,
            CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE r1 END ap_r1,
            CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE rf END ap_rf FROM z''')
        state=con.execute('''SELECT customer_id,max(user_history_events_12w) history_events,max(user_unique_items_12w) unique_items,
            max(user_days_since_last_purchase) recency,avg(1-missing) bpr_available,avg(latent) bpr_mean,stddev_pop(latent) bpr_std,
            max(CASE WHEN rl=1 THEN latent END)-max(CASE WHEN rl=5 THEN latent END) bpr_margin,
            count(*) FILTER(WHERE r0<=12 AND r1<=12)/12.0 overlap12,count(*) FILTER(WHERE r0<=20 AND r1<=20)/20.0 overlap20,
            avg(abs(r0-r1))/count(*) disagreement,corr(r0,r1) rank_corr,stddev_pop(score_base) e0_std,stddev_pop(score_bpr) e1_std,
            max(CASE WHEN r0=1 THEN score_base END)-max(CASE WHEN r0=5 THEN score_base END) e0_margin,
            max(CASE WHEN r1=1 THEN score_bpr END)-max(CASE WHEN r1=5 THEN score_bpr END) e1_margin,count(*) candidate_count
            FROM a GROUP BY customer_id ORDER BY customer_id''').fetchdf()
        for label,r in zip(EXPERTS,['ap_r0','ap_r1','ap_rf']):
            ap=ap_table(con,'a',r).rename(columns={'ap':'ap_'+label});state=state.merge(ap,on='customer_id',validate='one_to_one')
        con.execute(f'COPY a TO {literal(root/"ranks.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
    state['cutoff']=cutoff;save_parquet(state,root/'users.parquet')
    aps=state[['ap_'+x for x in EXPERTS]].to_numpy();best=aps.max(axis=1)
    tied=np.isclose(aps,best[:,None],rtol=0,atol=1e-15).sum(axis=1)>1
    headroom=float((best-aps[:,2]).sum()/total_users)
    r={'window':w,'year':year,'role':role,'cutoff':cutoff,'total_users':int(total_users),'included_users':len(state),
       'population_weight':len(state)/total_users,'oracle_headroom_population_MAP':headroom,
       'exclusive_winners':{x:int(((aps[:,i]==best)&~tied).sum()) for i,x in enumerate(EXPERTS)},'ties':int(tied.sum()),
       'baseline_map_population_component':float(aps[:,2].sum()/total_users),
       'users_path':str(root/'users.parquet'),'ranks_path':str(root/'ranks.parquet'),'features':FEATURES,
       'runtime_seconds':time.perf_counter()-start,'final_week':'not_run',
       'population':'covered-active only for inner; full original cohort for outer; all-user denominator retained for deltas',
       'expert_selection_note':'inner checkpoint was chosen on that same week; inner screening is not untouched expert OOF; gate fitting uses prior OUTER expert scores only'}
    if role=='outer':r['independent_source_check']=verify_outer_sources(w,year,cutoff)
    write(meta,r);print({'gate_data':cutoff,'users':len(state),'oracle_headroom':headroom},flush=True);return r

def train(train_meta,method):
    if not train_meta:return None,{'fallback':'no_prior_meta_week'}
    frames=[load_parquet(m['users_path']) for m in train_meta];f=pd.concat(frames,ignore_index=True)
    x=f[FEATURES].replace([np.inf,-np.inf],np.nan)
    reward=f[['ap_'+k for k in EXPERTS]].to_numpy();span=reward.max(axis=1)-reward.min(axis=1)
    informative=span>1e-12
    if informative.sum()<100:return None,{'fallback':'less_than100_informative_meta_users'}
    if method=='logistic':
        # AP-regret weights focus choice on users whose recommendations matter;
        # ties prefer the already-promoted fixed fusion, not an arbitrary label.
        y=reward.argmax(axis=1);y[reward[:,2]>=reward.max(axis=1)-1e-15]=2
        if len(np.unique(y[informative]))<2:return None,{'fallback':'single_class_meta_supervision'}
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1,max_iter=500,solver='lbfgs',random_state=20260909))
        weights=span[informative];weights=weights/weights.mean()
        model.fit(x[informative],y[informative],logisticregression__sample_weight=weights)
    else:
        # Two shallow utility regressors predict expected AP change versus RRF.
        model=[]
        for i in (0,1):
            m=lgb.LGBMRegressor(n_estimators=60,num_leaves=7,max_depth=3,min_child_samples=250,learning_rate=.03,
                reg_lambda=10,n_jobs=4,verbosity=-1,random_state=20260909,deterministic=True,force_col_wise=True)
            m.fit(x,reward[:,i]-reward[:,2]);model.append(m)
    return model,{'train_cutoffs':[m['cutoff'] for m in train_meta],'users':len(f),'informative_users':int(informative.sum()),
        'target':'historical full-truth AP expert utility; candidate labels are never gate input','method':method}

def predict(model,method,f):
    if model is None:return np.full(len(f),2,np.int8)
    x=f[FEATURES].replace([np.inf,-np.inf],np.nan)
    if method=='logistic':choice=model.predict(x).astype(np.int8)
    else:
        utility=np.column_stack([model[0].predict(x),model[1].predict(x)])
        choice=utility.argmax(axis=1).astype(np.int8)
        choice[utility.max(axis=1)<=0]=2
    choice[f.history_events.to_numpy()==0]=2
    return choice

def safe_training(history,cutoff):
    return [m for m in history if date.fromisoformat(m['cutoff'])+timedelta(days=7)<=date.fromisoformat(cutoff)]

def score_gate(meta,model,method,outroot):
    f=load_parquet(meta['users_path']);choice=predict(model,method,f)
    reward=f[['ap_'+k for k in EXPERTS]].to_numpy();ap=reward[np.arange(len(f)),choice];delta=ap-reward[:,2]
    f['choice']=choice;f['ap_gate']=ap;outroot.mkdir(parents=True,exist_ok=True);save_parquet(f,outroot/'user_choices.parquet')
    return {'population_delta':float(delta.sum()/meta['total_users']),'map_component':float(ap.sum()/meta['total_users']),
        'choice_counts':{k:int((choice==i).sum()) for i,k in enumerate(EXPERTS)},'users_improved':int((delta>1e-15).sum()),
        'users_harmed':int((delta< -1e-15).sum()),'gross_positive':float(delta[delta>0].sum()/meta['total_users']),
        'gross_negative':float(delta[delta<0].sum()/meta['total_users']), 'user_choices':str(outroot/'user_choices.parquet')}

def audit():
    setup();budget(10);start=time.perf_counter()
    historical=[dataset(w,2019,'outer') for w in contracts(2019)['rolling_protocol']]
    inner=[dataset(w,2020,'inner') for w in contracts(2020)['rolling_protocol']]
    segments={}
    for m in inner:
        f=load_parquet(m['users_path']);r=f[['ap_'+x for x in EXPERTS]].to_numpy();f['headroom']=r.max(axis=1)-r[:,2]
        result={}
        for col,bins in [('history_events',[-1,0,5,20,np.inf]),('unique_items',[-1,5,20,np.inf]),
                         ('bpr_available',[-.1,.5,.95,1.1]),('overlap12',[-.1,.33,.66,1.1]),
                         ('disagreement',[-.1,.1,.25,1.1]),('bpr_std',[-.1,.1,.5,1,10,np.inf])]:
            groups=f.groupby(pd.cut(f[col],bins),observed=True)
            result[col]=[{'bin':str(k),'users':len(g),'mean_headroom':float(g.headroom.mean()),
                'all_users_contribution':float(g.headroom.sum()/m['total_users'])} for k,g in groups]
        segments[m['window']]=result
    head=[m['oracle_headroom_population_MAP'] for m in inner]
    out={'historical_meta':historical,'inner':inner,'segments':segments,'oracle_mean_headroom':float(np.mean(head)),
         'oracle_gate':np.mean(head)>=.001 and sum(h>=.0005 for h in head)>=3,
         'policy':'gate targets from prior FRESH outer expert scores; first2019 replay falls backRRF if no earlier metadata; no2019-specific params',
         'runtime_seconds':time.perf_counter()-start,'final_week':'not_run'}
    out['oracle_gate']=bool(out['oracle_gate']);write(REPORT/'WV3_100_GATE_ORACLE_AUDIT.json',out);return out

def screen():
    setup();a=read(REPORT/'WV3_100_GATE_ORACLE_AUDIT.json');assert a['oracle_gate'];results={}
    for method,trial in [('logistic','WV3-101'),('utility_tree','WV3-102')]:
        if (REPORT/f'{trial}_SCREEN.json').exists():continue
        register(trial,'user_expert_gate','User/expert confidence can predict which frozen ordering has higher AP without changing candidate pool.',
            params={'method':method,'C':1,'utility_trees':60,'leaves':7,'depth':3,'min_child':250,'l2':10,'seed':20260909},
            features=FEATURES,training_protocol='prior2019 outer meta labels only; no same-window checkpoint selection for gate training',
            screen_gate={'mean_delta_min':.0001,'positive_min':3,'worst_min':-.0005},
            candidate_protocol='choose one entire E0/E1/RRF ordering per user; exact original inactive fallback')
        t=time.perf_counter();windows={}
        for m in a['inner']:
            train_meta=safe_training(a['historical_meta'],m['cutoff']);model,training=train(train_meta,method)
            r=score_gate(m,model,method,ART/trial/m['window']/'inner');windows[m['window']]={**r,'training':training}
        ds=[x['population_delta'] for x in windows.values()];passed=np.mean(ds)>=.0001 and sum(d>0 for d in ds)>=3 and min(ds)>=-.0005
        out={'windows':windows,'mean_population_delta':float(np.mean(ds)),'positive_windows':sum(d>0 for d in ds),
            'worst_delta':min(ds),'passed':bool(passed),'runtime_seconds':time.perf_counter()-t,'final_week':'not_run'}
        write(REPORT/f'{trial}_SCREEN.json',out);update(trial,inner_evidence=out,decision='inner_pass' if passed else 'reject_inner',runtime=out['runtime_seconds'])
        print({trial:{k:v for k,v in out.items() if k!='windows'}},flush=True);results[trial]=out
    return results

def confirm(trial='WV3-102',year=2020):
    setup();budget(8);a=read(REPORT/'WV3_100_GATE_ORACLE_AUDIT.json')
    method={'WV3-101':'logistic','WV3-102':'utility_tree'}[trial]
    output=REPORT/f'{trial}_{"OUTER" if year==2020 else "ROBUSTNESS"}.json'
    assert not output.exists(),'concrete variant already evaluated; no tiny-parameter rescue'
    if year==2020:
        assert read(REPORT/f'{trial}_SCREEN.json')['passed'];expose(trial)
    start=time.perf_counter();windows={};maps={}
    for w,p in contracts(year)['rolling_protocol'].items():
        m=dataset(w,year,'outer');training=safe_training(a['historical_meta'],m['cutoff'])
        assert all(date.fromisoformat(t['cutoff'])+timedelta(days=7)<=date.fromisoformat(m['cutoff']) for t in training)
        model,metadata=train(training,method);root=ART/trial/f'{year}_{w}'
        r=score_gate(m,model,method,root);joblib.dump(model,root/'gate.joblib')
        write(root/'training.json',metadata)
        choices=load_parquet(root/'user_choices.parquet')[['customer_id','choice']]
        db=root/'evaluation.duckdb';assert not db.exists()
        with connection() as con:
            con.register('choices',choices)
            con.execute(f'ATTACH {literal(db)} AS outdb')
            con.execute(f'''CREATE TABLE outdb.predictions AS SELECT r.customer_id,r.article_id,r.candidate_rank,r.target,r.user_history_events_12w,
                CASE c.choice WHEN 0 THEN 1.0/(60+r.r0) WHEN 1 THEN 1.0/(60+r.r1) ELSE r.rrf_score END score,c.choice
                FROM read_parquet({literal(m['ranks_path'])}) r JOIN choices c USING(customer_id)''')
            from .warm_v2_rank_fusion import final_rank_sql
            con.execute(f'CREATE TABLE outdb.top12 AS SELECT * FROM (SELECT *,{final_rank_sql()} final_rank FROM outdb.predictions) WHERE final_rank<=12')
            top=con.execute('SELECT customer_id,article_id,final_rank FROM outdb.top12 ORDER BY customer_id,final_rank').fetchdf()
            truth=con.execute(f'''SELECT DISTINCT customer_id,article_id FROM read_parquet({literal(TX)})
                WHERE t_dat>=DATE '{m['cutoff']}' AND t_dat<DATE '{m['cutoff']}'+INTERVAL 7 DAY AND hash(customer_id)%1000000<100000''').fetchdf()
            con.execute(f'COPY (SELECT * FROM outdb.top12 ORDER BY customer_id,final_rank) TO {literal(root/"top12.parquet")} (FORMAT PARQUET,COMPRESSION ZSTD)')
        t={u:g.article_id.tolist() for u,g in truth.groupby('customer_id')}
        pred={u:g.article_id.tolist() for u,g in top.groupby('customer_id')}
        assert set(pred)==set(t) and all(len(v)==len(set(v))==12 for v in pred.values())
        value=float(np.mean([apk(t[u],pred[u]) for u in t]));assert abs(value-r['map_component'])<1e-12
        # Validate ordered selection against parent lists, including inactive fallback.
        old=FRESH if year==2019 else OLD;prefix='FRESH' if year==2019 else 'WV2'
        cm=dict(zip(choices.customer_id,choices.choice));bad=0
        for i,suffix in enumerate(('000','501','601')):
            with connection() as con:
                par=con.execute(f'SELECT customer_id,article_id FROM read_parquet({literal(old/f"{prefix}-{suffix}"/w/"top12.parquet")}) ORDER BY customer_id,final_rank').fetchdf()
            for u,g in par.groupby('customer_id'):
                if cm[u]==i:bad+=g.article_id.tolist()!=pred[u]
        assert bad==0
        maps[w]=value;windows[w]={**r,'map@12':value,'training':metadata,'exact_parent_top12_mismatches':bad,
            'candidate_pool_changed':False,'full_raw_population_exact':True,'cutoff':m['cutoff'],
            'top12':evidence_id(root/'top12.parquet',reason='explicit_registry_evidence')}
    out={**summary(maps,year),'trial':trial,'year':year,'windows':windows,'runtime_seconds':time.perf_counter()-start,
        'candidate_pool_changed':False,'score_semantics':'user gate chooses one fixed complete ordering; selected expert reciprocal rank, not probability',
        'replay_description':'rolling prior2019 outer supervision; not fresh; earliest2019week falls back tobaseline without prior meta-data'}
    write(output,out)
    if year==2020:update(trial,outer_MAP_by_window=maps,mean_MAP=out['mean_MAP'],delta_vs_WV2_601=out['delta_vs_WV2_601'],
        nondegrade_windows=out['nondegrade_windows'],worst_delta=out['worst_delta'],decision='stable_candidate' if out['stable'] else 'reject_outer',
        runtime=out['runtime_seconds'],artifact_paths=[str(output)])
    else:update(trial,cross_year_replay=out)
    print({k:v for k,v in out.items() if k!='windows'},flush=True);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit','screen','confirm','replay']);p.add_argument('--trial',default='WV3-102');a=p.parse_args()
    if a.command in ('audit','screen'):{'audit':audit,'screen':screen}[a.command]()
    else:confirm(a.trial,2019 if a.command=='replay' else 2020)
