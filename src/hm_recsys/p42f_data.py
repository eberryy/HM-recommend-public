"""P4.2F label-free relative evidence and purchase-time novelty state."""
from pathlib import Path
import hashlib
import json
import time
import shutil
import numpy as np
import pandas as pd
import duckdb
from .p41a_contract import read_json, write_json, identity, check_identity
from .p42_contract import guard_cutoff, WINDOWS
from .p42e_contract import RUN_ID as E_RUN
from .p42f_contract import RUN_ID, DATES, USER, earlier, now

OLD = 'p4-2-v1-two-sided-risk-controlled-admission'
REPAIR = 'p4-2r-v1-population-numerical-repair'

def records(x):
    if isinstance(x,dict):
        if 'path' in x and 'sha256' in x and 'bytes' in x: yield x
        for v in x.values(): yield from records(v)
    elif isinstance(x,list):
        for v in x: yield from records(v)

def paths_for(repo,c,t):
    repair=repo/'artifacts/phase4'/REPAIR/'prepared'/t
    if t not in DATES:
        return dict(cold=repo/'artifacts/phase4'/OLD/'prepared'/t/'qC-features.parquet',
                    warm=repo/'artifacts/phase4'/OLD/'prepared'/t/'qW-features.parquet',
                    selection=Path(c['inputs'][t]['cold50']['path']),warm150=Path(c['inputs'][t]['warm150']['path']))
    return dict(cold=repair/'qC-features.parquet',
                warm=repo/'artifacts/phase4'/OLD/'prepared'/t/'qW-features.parquet',
                selection=repair/'cold50-selection-before-labels.parquet',
                roster=repair/'shared-population-roster.parquet',
                warm150=Path(c['inputs'][t]['warm150']['path']))

def verify_inputs(repo,c):
    """Compare actually used assets with earlier independently bound manifests."""
    known={}
    for name in ['P4_2_OUTPUT_MANIFEST.json','P4_2R_OUTPUT_MANIFEST.json','P4_2R3_OUTPUT_MANIFEST.json','P4_2E_OUTPUT_MANIFEST.json','P4_2E_EXPERIMENT_CONTRACT.json']:
        for r in records(read_json(repo/'reports/phase4'/name)):
            known[str(Path(r['path']).resolve()).lower()]=r
    wanted=[Path(c['transactions']['path']),Path(c['catalog']['path'])]
    for t in sorted(set(DATES+list(WINDOWS.values()))): wanted+=list(paths_for(repo,c,t).values())
    output=[]
    for p in dict.fromkeys(wanted):
        r=known.get(str(p.resolve()).lower())
        if r is None: raise ValueError('no trusted identity for '+str(p))
        check_identity(r); output.append(r)
    return output

def frame(p):
    with duckdb.connect() as db:
        return db.execute('SELECT * FROM read_parquet(?)',[str(p)]).fetchdf()

def save_frame(f,p):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with duckdb.connect() as db:
        db.register('output_frame',f)
        db.execute('COPY output_frame TO ? (FORMAT PARQUET)',[str(p)])

def purchase_state(repo,c,root):
    """Materialize PIT item-day counters before any action target is computed."""
    started=time.perf_counter(); tx=c['transactions']['path']; catalog=c['catalog']['path']
    dates=sorted(set(DATES+list(WINDOWS.values())))
    rosters=[]
    for t in dates:
        users=frame(paths_for(repo,c,t)['warm'])[['customer_id']].drop_duplicates()
        users['cutoff']=t; rosters.append(users)
    rosters=pd.concat(rosters,ignore_index=True)
    people=rosters[['customer_id']].drop_duplicates()
    db=duckdb.connect(str(root/'novelty.duckdb'))
    db.execute('SET threads=4'); db.execute("SET memory_limit='2GB'")
    db.register('people',people)
    db.execute("CREATE TABLE events AS SELECT t.customer_id,t.article_id,t.t_dat FROM read_parquet(?) t JOIN people USING(customer_id) WHERE t_dat<DATE '2020-08-19'",[tx])
    db.execute("CREATE TABLE daily AS SELECT article_id,t_dat,count(*) n FROM read_parquet(?) WHERE t_dat<DATE '2020-08-19' GROUP BY article_id,t_dat",[tx])
    db.execute('CREATE TABLE cumulative AS SELECT article_id,t_dat,sum(n) OVER(PARTITION BY article_id ORDER BY t_dat ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)::BIGINT cnt FROM daily')
    # Strict > in ASOF excludes the WHOLE purchase day, including duplicates.
    db.execute('CREATE TABLE event_coldness AS SELECT e.*,coalesce(c.cnt,0)::BIGINT before_count FROM events e ASOF LEFT JOIN cumulative c ON e.article_id=c.article_id AND e.t_dat>c.t_dat')
    db.execute('CREATE TABLE catalog AS SELECT article_id FROM read_csv_auto(?, all_varchar=true)',[catalog])
    ncat=db.execute('SELECT count(*) FROM catalog').fetchone()[0]
    # Distributions have one row per distinct count, not a 70M-row persisted grid.
    days=db.execute('SELECT DISTINCT t_dat FROM events ORDER BY t_dat').fetchall()
    db.execute('CREATE TABLE count_cdf(t_dat DATE,before_count BIGINT,percentile DOUBLE)')
    for i,(day,) in enumerate(days):
        db.execute('''INSERT INTO count_cdf
            WITH snapshot AS (SELECT a.article_id,coalesce(c.cnt,0)::BIGINT n
              FROM (SELECT article_id,?::DATE d FROM catalog) a ASOF LEFT JOIN cumulative c
              ON a.article_id=c.article_id AND a.d>c.t_dat),
            histogram AS (SELECT n,count(*) k FROM snapshot GROUP BY n)
            SELECT ?::DATE,n, (sum(k) OVER(ORDER BY n)-k+(k-1)/2.0)/? FROM histogram''',[day,day,ncat-1])
        if i%100==0: print(f'purchase-time PIT percentiles {i}/{len(days)} dates',flush=True)
        if time.perf_counter()-started>c['budget']['pilot_seconds']:
            raise TimeoutError('novelty fixed preflight exceeded 10 minute pilot; no fit started')
        if shutil.disk_usage(repo).free<15*2**30: raise RuntimeError('disk guard')
    db.execute('CREATE TABLE enriched AS SELECT e.*,p.percentile FROM event_coldness e LEFT JOIN count_cdf p USING(t_dat,before_count)')
    if db.execute('SELECT count(*) FROM enriched WHERE percentile IS NULL').fetchone()[0]:
        raise ValueError('purchase-time percentile has unmapped catalog events')
    summaries={}
    for t in dates:
        wanted=rosters.loc[rosters.cutoff.eq(t),['customer_id']]; db.register('wanted',wanted)
        result=db.execute('''SELECT customer_id,
          count(*)::BIGINT user_past_purchase_count,
          count(*) FILTER(WHERE before_count<=5)::BIGINT novel_purchase_count_0_5,
          avg((before_count<=5)::INT) novel_purchase_share_0_5,
          count(*) FILTER(WHERE before_count<=20)::BIGINT low_pop_purchase_count_0_20,
          avg((before_count<=20)::INT) low_pop_purchase_share_0_20,
          count(*) FILTER(WHERE before_count<=5 AND t_dat>=?::DATE-INTERVAL 84 DAY)::BIGINT novel_purchase_count_0_5_recent84d,
          avg((before_count<=5)::INT) FILTER(WHERE t_dat>=?::DATE-INTERVAL 84 DAY) novel_purchase_share_0_5_recent84d,
          date_diff('day',max(t_dat) FILTER(WHERE before_count<=5),?::DATE) days_since_last_0_5_purchase,
          median(before_count) median_item_interaction_count_at_purchase,
          median(percentile) median_item_popularity_percentile_at_purchase,
          count(*) FILTER(WHERE t_dat>=?::DATE-INTERVAL 84 DAY)::BIGINT user_purchase_events_12w,
          count(DISTINCT article_id) FILTER(WHERE t_dat>=?::DATE-INTERVAL 84 DAY)::BIGINT user_unique_items_12w,
          count(DISTINCT article_id)::BIGINT user_unique_items_all,
          count(DISTINCT t_dat) FILTER(WHERE t_dat>=?::DATE-INTERVAL 84 DAY)::BIGINT active_purchase_days_12w,
          count(DISTINCT t_dat)::BIGINT active_purchase_days_all,
          date_diff('day',max(t_dat),?::DATE) days_since_last_purchase
          FROM enriched JOIN wanted USING(customer_id) WHERE t_dat<?::DATE GROUP BY customer_id''',[t]*8).fetchdf()
        result=wanted.merge(result,on='customer_id',how='left',validate='one_to_one')
        for k in USER:
            if k in result and ('count' in k and not k.startswith('median') or k.startswith(('user_unique','user_purchase_events','active_purchase'))): result[k]=result[k].fillna(0)
        prior=[]
        for s in earlier(t):
            r=frame(paths_for(repo,c,s)['roster']); prior.extend(r.loc[r.has_mapped_history,'customer_id'].tolist())
        support=pd.Series(prior,dtype=str).value_counts()
        result['historical_action_windows_available']=result.customer_id.map(support).fillna(0).astype(int)
        assert not result.duplicated('customer_id').any()
        save_frame(result,root/f'user-state-{t}.parquet')
        summaries[t]=dict(users=len(result),with_history=int(result.user_past_purchase_count.gt(0).sum()))
    events=db.execute('SELECT count(*) FROM events').fetchone()[0]
    db.close()
    receipt=dict(stage='P4.2F',status='completed',seconds=time.perf_counter()-started,events=events,
        global_counts='all users; strict earlier date',distinct_purchase_dates=len(days),catalog_items=ncat,cutoffs=summaries,
        model_fits=0,action_labels_materialized=False,final_week='not_run')
    write_json(root/'NOVELTY_PREPARATION.json',receipt)
    return receipt
