"""FINAL-E1: bounded attribute-history evidence ablation, no policy search."""
from pathlib import Path
import gc
import json
import time
import traceback
import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .final_oracle_audit import dump
from .p42f_contract import GLOBAL, earlier
from .p42f_core import batches, sample_edges
from .p43a_run import relevance, guard

VALID = ['2020-02-19', '2020-04-29', '2020-07-22']
ATTRS = ['product_code', 'product_type_no', 'colour_group_code']
REL = [a+'_'+f for a in ATTRS for f in ['share28', 'share84', 'days_since84']]
EXTRA = [side+'_'+f for side in ['cold_rel','warm_rel','diff_rel'] for f in REL]


def history_features(keys, history, catalog, cutoff):
    """Input history may include future rows; SQL firewall removes them."""
    with duckdb.connect(config={'threads':4,'memory_limit':'2GB'}) as db:
        db.register('keys',keys)
        db.register('history_input',history)
        db.register('catalog',catalog)
        db.execute('CREATE TEMP TABLE h AS SELECT h.customer_id,h.article_id,CAST(h.t_dat AS DATE) AS purchase_day FROM history_input h '
                   'WHERE CAST(h.t_dat AS DATE)<?::DATE AND CAST(h.t_dat AS DATE)>=?::DATE-INTERVAL 84 DAY', [cutoff,cutoff])
        db.execute('CREATE TEMP TABLE totals AS SELECT customer_id,count(*) n84,'
                   'count(*) FILTER(WHERE purchase_day>=?::DATE-INTERVAL 28 DAY) n28 FROM h GROUP BY customer_id',[cutoff])
        base=db.execute('SELECT k.*,c.product_code,c.product_type_no,c.colour_group_code FROM keys k '
                        'LEFT JOIN catalog c USING(article_id) ORDER BY k.row_id').fetchdf()
        assert not base[ATTRS].isna().any().any()
        db.register('base',base)
        result=[]
        for a in ATTRS:
            db.execute(f'CREATE OR REPLACE TEMP TABLE agg AS SELECT customer_id,{a},count(*) cnt84,'
                       'count(*) FILTER(WHERE purchase_day>=?::DATE-INTERVAL 28 DAY) cnt28,max(purchase_day) last_day '
                       f'FROM h JOIN catalog USING(article_id) GROUP BY customer_id,{a}',[cutoff])
            f=db.execute(f'SELECT coalesce(agg.cnt28/nullif(t.n28,0),0) share28,'
                         'coalesce(agg.cnt84/nullif(t.n84,0),0) share84,'
                         'date_diff(\'day\',agg.last_day,?::DATE) days_since84 '
                         f'FROM base b LEFT JOIN agg USING(customer_id,{a}) LEFT JOIN totals t USING(customer_id) ORDER BY b.row_id',[cutoff]).fetchnumpy()
            cols=[]
            for name in ['share28','share84','days_since84']:
                arr=f[name]
                if np.ma.isMaskedArray(arr): arr=arr.astype(float).filled(np.nan)
                cols.append(np.asarray(arr,dtype=np.float32))
            result.append(np.column_stack(cols))
        return np.concatenate(result,axis=1)


def append_features(x,cold_features,warm_features,user_index):
    n=len(cold_features)
    cc=np.repeat(cold_features,12,axis=0)
    ww=warm_features[np.asarray(user_index,int)].reshape(n*12,len(REL))
    return np.concatenate([x,cc,ww,cc-ww],axis=1).astype(np.float32)


def identify(data, scores):
    c=data['cold'];groups=c.groupby('user_index',sort=False).indices
    positive_total=int(c.target.sum())
    counts={k:0 for k in [1,5]};slots={k:0 for k in [1,5]}
    rr=[]; hits=0
    for ui,idx in groups.items():
        order=np.argsort(-scores[idx],kind='stable')
        target=c.target.to_numpy()[idx][order]
        for k in counts:
            counts[k]+=int(target[:k].sum());slots[k]+=min(k,len(target))
        if target.any():
            first=int(np.flatnonzero(target)[0])+1
            rr.append(1/first)
            hits+=int(first==1)
    return dict(users=len(data['users']),candidate_users=len(groups),candidate_pairs=len(c),positive_pairs=positive_total,
        opportunity_users=len(rr),positive_hits_at1=hits,
        recall={str(k):counts[k]/positive_total if positive_total else 0. for k in counts},
        precision={str(k):counts[k]/slots[k] if slots[k] else 0. for k in counts},
        conditional_mrr=float(np.mean(rr)) if rr else 0.,topk_positive_pairs=counts)


def run(repo):
    repo=Path(repo);root=repo/'artifacts/final/relation-pilot-v1';report=repo/'reports/final'
    source=repo/'artifacts/final/integration-10pct-v1'
    if root.exists(): raise FileExistsError('Do not overwrite previous experiment')
    guard(repo)
    original=json.loads((report/'FINAL_INTEGRATION_CONTRACT.json').read_text(encoding='utf-8'))
    dates=sorted(set(VALID+sum([earlier(t) for t in VALID],[])))
    root.mkdir(parents=True)
    c=dict(stage='FINAL-E1: historical attribute-relation pilot',status='preregistered',validation_dates=VALID,
        training_dates={t:earlier(t) for t in VALID},prepared_dates=dates,attributes=ATTRS,features=GLOBAL,
        added_features=EXTRA,params=original['model']['params'],arms=['base70','plus27'],
        feature_semantics='For each side and attribute: share of all user purchase EVENTS in last28/84 days matching candidate; days since latest match within84d; missing recency NaN, no history shares0. Append cold,warm,cold-warm.',
        timing='history [t-84d,t), duplicate events preserved, static optimistic article metadata; no current-week features',
        sampling='exact saved original A1 all nonzero + deterministic2%neutral; all weights1; verify base X,y identity during extension',
        model='same300-tree LambdaRank, graded AP target, groups user-date; no early stopping or hyperparameter selection',
        metric='all full Cold50 candidates, rank by max predicted score over12 slots; B0 rank reference; positive pair recall@1/5; precision denominator all selected topK rows incl zero-positive users',
        gate='plus27 mean Recall@1 and Recall@5 strictly greater than BOTH base70 and B0; for EACH recall and EACH comparator at least2 of3 dates nonworse; mean Precision@1 greater than both; otherwise STOP',
        gate_scope='mechanism screen, not statistical significance, not final promotion; no fusion unless candidate gate passes',
        conditional_next='if gate passes report readiness; four-window integration requires follow-up execution with original frozen gates, no policy search',
        no_image_change=True,no_candidate_change=True,final_week='not_run',outer_four_windows='not_run_this_pilot',
        budget=dict(max_seconds=7200,pilot_first_date_max_seconds=600,threads=4,min_disk_gib=15,min_ram_gib=1.5),
        fallback='WV3 retained; if gate fails no follow-on fit/search; preserve evidence',
        sources=dict(transactions=str(repo/'data/interim/audit/transactions.parquet'),articles=str(repo/'data/raw/articles.csv')))
    dump(report/'RELATION_PILOT_CONTRACT.json',c)
    started=time.perf_counter();deadline=time.time()+7200
    cat=pd.read_csv(c['sources']['articles'],dtype={'article_id':str},usecols=['article_id',*ATTRS])
    assert cat.article_id.is_unique
    prep={}
    for t in dates:
        guard(repo,deadline);begin=time.perf_counter()
        data=joblib.load(source/'prepared'/t/'data.joblib')
        keys=pd.concat([data['cold'][['customer_id','article_id']],data['warm'][['customer_id','article_id']]],ignore_index=True)
        keys['row_id']=np.arange(len(keys))
        wanted=pd.DataFrame({'customer_id':data['users']})
        with duckdb.connect(config={'threads':4,'memory_limit':'2GB'}) as db:
            db.register('wanted',wanted)
            h=db.execute('SELECT customer_id,article_id,t_dat FROM read_parquet(?) JOIN wanted USING(customer_id) '
                         'WHERE t_dat<?::DATE AND t_dat>=?::DATE-INTERVAL 84 DAY',[c['sources']['transactions'],t,t]).fetchdf()
        features=history_features(keys,h,cat,t)
        nc=len(data['cold']);cf=features[:nc];wf=features[nc:].reshape(-1,12,len(REL))
        folder=root/t;folder.mkdir()
        np.save(folder/'cold.npy',cf);np.save(folder/'warm.npy',wf)
        xold=np.load(source/'prepared'/t/'X.npy',mmap_mode='r')
        yold=np.load(source/'prepared'/t/'y.npy',mmap_mode='r')
        xplus=np.lib.format.open_memmap(folder/'Xplus.npy',mode='w+',dtype=np.float32,shape=(len(xold),len(GLOBAL)+len(EXTRA)))
        cursor=0
        for lo,cc,x,y in batches(data):
            guard(repo,deadline)
            keep,_=sample_edges(t,cc.customer_id.to_numpy(),cc.article_id.to_numpy(),y)
            expected=x[keep.ravel()];n=len(expected)
            np.testing.assert_allclose(xold[cursor:cursor+n],expected,atol=0,rtol=0,equal_nan=True)
            np.testing.assert_array_equal(yold[cursor:cursor+n],y[keep])
            xx=append_features(x,cf[lo:lo+len(cc)],wf,cc.user_index)
            xplus[cursor:cursor+n]=xx[keep.ravel()];cursor+=n
        assert cursor==len(xold)
        xplus.flush()
        prep[t]=dict(rows=cursor,history_events=len(h),history_max_date=str(h.t_dat.max()),
            seconds=time.perf_counter()-begin,base_X_y_exact=True,
            missing_by_feature={name:int(np.isnan(features[:,j]).sum()) for j,name in enumerate(REL)})
        dump(folder/'PREPARED.json',prep[t]);print('PREPARED',t,prep[t]['seconds'],flush=True)
        del data,h,features,xplus,xold,yold;gc.collect()
        if t==dates[0] and prep[t]['seconds']>600: raise TimeoutError('First-date cost exceeded preregistered bound')
    results={}
    for t in VALID:
        train=earlier(t)
        data=joblib.load(source/'prepared'/t/'data.joblib')
        results[t]={'B0':identify(data,-data['cold'].b0_rank.to_numpy())}
        for arm in ['base70','plus27']:
            guard(repo,deadline);begin=time.perf_counter()
            xp=[(source/'prepared'/d/'X.npy' if arm=='base70' else root/d/'Xplus.npy') for d in train]
            x=np.concatenate([np.load(p) for p in xp])
            y=np.concatenate([np.load(source/'prepared'/d/'y.npy') for d in train])
            groups=np.concatenate([np.load(source/'prepared'/d/'groups.npy') for d in train])
            assert groups.sum()==len(x)==len(y)
            features=GLOBAL if arm=='base70' else GLOBAL+EXTRA
            model=lgb.LGBMRanker(**c['params'])
            with threadpool_limits(limits=4): model.fit(x,relevance(y),group=groups,feature_name=features)
            model.booster_.save_model(str(root/t/(arm+'.txt')))
            del x,y,groups;gc.collect()
            cf=np.load(root/t/'cold.npy');wf=np.load(root/t/'warm.npy')
            s=np.empty(len(data['cold']),np.float64)
            for lo,cc,x,_ in batches(data,labels=False):
                guard(repo,deadline)
                if arm=='plus27':x=append_features(x,cf[lo:lo+len(cc)],wf,cc.user_index)
                s[lo:lo+len(cc)]=model.predict(x,num_threads=4).reshape(-1,12).max(axis=1)
            assert np.isfinite(s).all()
            np.save(root/t/(arm+'-candidate-scores.npy'),s)
            results[t][arm]=identify(data,s)
            results[t][arm]['seconds']=time.perf_counter()-begin
            results[t][arm]['train_dates']=train
            print('RESULT',t,arm,json.dumps(results[t][arm]),flush=True)
            del model,s;gc.collect()
        dump(root/t/'RESULT.json',results[t]);del data;gc.collect()
    gates={}
    for k in ['1','5']:
        for ref in ['base70','B0']:
            delta=np.array([results[t]['plus27']['recall'][k]-results[t][ref]['recall'][k] for t in VALID])
            gates[f'recall{k}_vs_{ref}']=bool(delta.mean()>0 and (delta>=0).sum()>=2)
    for ref in ['base70','B0']:
        gates['precision1_vs_'+ref]=bool(np.mean([results[t]['plus27']['precision']['1']-results[t][ref]['precision']['1'] for t in VALID])>0)
    result=dict(status='completed',stage=c['stage'],windows=results,preparation=prep,gates=gates,
        candidate_gate_pass=all(gates.values()),fusion='not_run',final_week='not_run',baseline='WV3-741 unchanged',
        seconds=time.perf_counter()-started)
    dump(report/'RELATION_PILOT.json',result)
    print(json.dumps(dict(gates=gates,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    try:run(Path.cwd())
    except Exception:
        dump(Path('reports/final')/f'RELATION_PILOT_FAILURE_{time.time_ns()}.json',dict(error=traceback.format_exc(),final_week='not_run'))
        raise
