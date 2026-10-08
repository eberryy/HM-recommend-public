"""FINAL-E3 pseudo-cold supervision from globally warm, user-unseen Warm50 items."""
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
from .final_candidate_e2 import DATES,VALID,M4COL,ALPHAS,PARAMS,embedding_path,histories,relation_batch,rrf
from .final_relation_pilot import identify
from .p42f_contract import USER,earlier
from .p42f_data import save_frame
from .p43a_run import guard

FEATURES=M4COL+USER

def candidates(repo,t,data):
    users=data['users'];stage=repo/'artifacts/final/history-rebuild-v1'/t/'stage.duckdb'
    with duckdb.connect(str(stage),read_only=True) as db:
        warm=db.execute('SELECT customer_id,article_id,ap_rf FROM top50 ORDER BY customer_id,ap_rf').fetchdf()
    assert len(warm)==len(users)*50 and np.array_equal(warm.customer_id.drop_duplicates(),users)
    with duckdb.connect(config={'threads':4,'memory_limit':'2GB'}) as db:
        db.register('warm',warm)
        f=db.execute('''WITH counts AS (SELECT article_id,count(*) n FROM read_parquet(?) WHERE t_dat<?::DATE GROUP BY article_id),
          seen AS (SELECT DISTINCT x.customer_id,x.article_id FROM read_parquet(?) x JOIN warm w USING(customer_id,article_id) WHERE x.t_dat<?::DATE)
          SELECT w.customer_id,w.article_id,w.ap_rf,coalesce(c.n,0) global_events
          FROM warm w LEFT JOIN counts c USING(article_id) LEFT JOIN seen s USING(customer_id,article_id)
          WHERE s.article_id IS NULL AND coalesce(c.n,0)>5 ORDER BY w.customer_id,w.ap_rf''',
          [str(repo/'data/interim/audit/transactions.parquet'),t,str(repo/'data/interim/audit/transactions.parquet'),t]).fetchdf()
    ui={u:i for i,u in enumerate(users)};f['user_index']=f.customer_id.map(ui).astype(np.int32)
    f['target']=[int(i in data['truthsets'][u]) for u,i in zip(f.customer_id,f.article_id)]
    assert not f.duplicated(['customer_id','article_id']).any()
    return f

def materialize(repo,root,t,catalog,article_to_row,ptype,garment,device):
    begin=time.perf_counter();data=joblib.load(repo/'artifacts/final/integration-10pct-v1/prepared'/t/'data.joblib')
    f=candidates(repo,t,data);folder=root/t;folder.mkdir(parents=True,exist_ok=True)
    rows,days,mask,audit=histories(repo,data['users'],t,article_to_row)
    embeddings=np.load(embedding_path(repo,t),mmap_mode='r')
    out=np.lib.format.open_memmap(folder/'pseudo-relation.npy',mode='w+',dtype=np.float32,shape=(len(f),24))
    groups=f.groupby('user_index',sort=False).indices;active=np.array(list(groups),int)
    for start in range(0,len(active),128):
        guard(repo);uis=active[start:start+128];indices=[np.asarray(groups[i],int) for i in uis]
        k=max(map(len,indices));items=np.zeros((len(uis),k),np.int32)
        for j,idx in enumerate(indices):items[j,:len(idx)]=[article_to_row[a] for a in f.iloc[idx].article_id]
        z=relation_batch(items,rows[uis],days[uis],mask[uis],embeddings,ptype,garment,device)
        for j,idx in enumerate(indices):out[idx]=z[j,:len(idx)].reshape(len(idx),24)
    out.flush();del out,embeddings;gc.collect()
    save_frame(f,folder/'pseudo.parquet')
    audit.update(candidate_pairs=len(f),candidate_users=f.customer_id.nunique(),positive_pairs=int(f.target.sum()),
        users_with_positive=int(f.groupby('customer_id').target.max().sum()),globally_warm_gt5=True,user_unseen=True,
        relation_shape=[len(f),24],seconds=time.perf_counter()-begin)
    dump(folder/'PSEUDO.json',audit);print('PSEUDO',t,audit,flush=True)

def pseudo_matrix(repo,root,t):
    from .p42f_data import frame
    data=joblib.load(repo/'artifacts/final/integration-10pct-v1/prepared'/t/'data.joblib')
    f=frame(root/t/'pseudo.parquet');rel=np.load(root/t/'pseudo-relation.npy')
    x=np.concatenate([rel,data['state'][USER].to_numpy(float)[f.user_index.to_numpy(int)]],axis=1).astype(np.float32)
    x[~np.isfinite(x)]=np.nan;group=f.groupby('user_index',sort=False).size().to_numpy(np.int32)
    assert group.sum()==len(x)==len(f)
    return x,f.target.to_numpy(np.int32),group

def cold_matrix(repo,e2,t):
    data=joblib.load(repo/'artifacts/final/integration-10pct-v1/prepared'/t/'data.joblib')
    rel=np.load(e2/t/'m4-relation.npy');idx=data['cold'].user_index.to_numpy(int)
    x=np.concatenate([rel,data['state'][USER].to_numpy(float)[idx]],axis=1).astype(np.float32);x[~np.isfinite(x)]=np.nan
    group=data['cold'].groupby('user_index',sort=False).size().to_numpy(np.int32)
    return data,x,data['cold'].target.to_numpy(np.int32),group

def run(repo):
    repo=Path(repo);root=repo/'artifacts/final/pseudocold-e3-v1';e2=repo/'artifacts/final/candidate-e2-v1';report=repo/'reports/final'
    if root.exists():raise FileExistsError('Do not overwrite E3')
    guard(repo);root.mkdir(parents=True)
    device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
    contract=dict(stage='FINAL-E3 pseudo-cold content supervision',status='preregistered',dates=DATES,
      validation=VALID,training={t:earlier(t) for t in VALID},
      pseudo_definition='WV3 reconstructed Warm Top50; globally >5 pre-cutoff events; article never purchased by this user before cutoff; all groups retained; label next7day purchase.',
      rationale='Increase user-content supervision without treating 12 replacement slots as independent candidate positives.',
      features=FEATURES,feature_scope='raw M4 relation4x6 plus17 user state; no Warm score/rank, item popularity/count, item ID, current-week aggregate or image retraining.',
      arms={'P41':'pseudo only','PC41':'pseudo plus real Cold50 as separate user-date query groups'},params=PARAMS,
      blends={'formula':'1/(60+B0_rank)+alpha/(60+model_rank)','alpha':ALPHAS},
      selection='same lexicographic mean Recall@1, Recall@5, Precision@1, conditional MRR, lower alpha; B0 included.',
      gate='winner strictly exceeds B0 mean Recall@1, Recall@5, Precision@1 and is nonworse >=2/3 dates for each recall.',
      leakage='each validation model uses only earlier cutoff labels; pseudo candidates and relations cutoff-safe; real Cold validation labels never fit.',
      limitations='Warm candidate-selection bias and globally warm-to-cold distribution shift; simulated cold is not real cold.',
      if_fail='stop pseudo-cold route; next bounded option is B0-preserving admission audit, not more content hyperparameters.',
      if_pass='freeze selector then build historical conservative admission; four-window MAP only after admission selection.',
      final_week='not_run',four_window_map='not_run',budget_seconds=7200,no_commit_push=True)
    dump(report/'PSEUDOCOLD_E3_CONTRACT.json',contract);started=time.perf_counter();deadline=time.time()+7200
    catalog=pd.read_csv(repo/'artifacts/m4/m4-v1-supervised-cold-representation/student-v1/static_catalog/catalog_items.csv',dtype={'article_id':str})
    articles=pd.read_csv(repo/'data/raw/articles.csv',dtype={'article_id':str},usecols=['article_id','product_type_no','garment_group_no']).set_index('article_id')
    aligned=articles.reindex(catalog.article_id);assert not aligned.isna().any().any()
    ptype=aligned.product_type_no.to_numpy(np.int32);garment=aligned.garment_group_no.to_numpy(np.int32);mapping={a:i for i,a in enumerate(catalog.article_id)}
    for t in DATES:guard(repo,deadline);materialize(repo,root,t,catalog,mapping,ptype,garment,device)
    results={}
    for t in VALID:
        data=joblib.load(repo/'artifacts/final/integration-10pct-v1/prepared'/t/'data.joblib')
        results[t]={'B0':identify(data,-data['cold'].b0_rank.to_numpy(float))}
        for arm in ['P41','PC41']:
            xs=[];ys=[];groups=[];begin=time.perf_counter()
            for d in earlier(t):
                x,y,g=pseudo_matrix(repo,root,d);xs.append(x);ys.append(y);groups.append(g)
                if arm=='PC41':
                    _,x,y,g=cold_matrix(repo,e2,d);xs.append(x);ys.append(y);groups.append(g)
            x=np.concatenate(xs);y=np.concatenate(ys);g=np.concatenate(groups);assert g.sum()==len(x)==len(y)
            model=lgb.LGBMRanker(**PARAMS)
            with threadpool_limits(limits=4):model.fit(x,y,group=g,feature_name=FEATURES)
            model.booster_.save_model(str(root/t/(arm+'.txt')))
            del x,y,g,xs,ys,groups;gc.collect()
            _,vx,_,_=cold_matrix(repo,e2,t);score=model.predict(vx,num_threads=4);np.save(root/t/(arm+'-scores.npy'),score)
            results[t][arm]=identify(data,score);results[t][arm]['seconds']=time.perf_counter()-begin
            for alpha in ALPHAS:results[t][f'{arm}-rrf-{alpha:g}']=identify(data,rrf(data,score,alpha))
            print('E3',t,arm,results[t][arm],flush=True);del model,vx,score;gc.collect()
        dump(root/t/'RESULT.json',results[t]);del data;gc.collect()
    keys=list(results[VALID[0]]);agg={}
    for key in keys:
        agg[key]=dict(mean_recall1=float(np.mean([results[t][key]['recall']['1'] for t in VALID])),
          mean_recall5=float(np.mean([results[t][key]['recall']['5'] for t in VALID])),
          mean_precision1=float(np.mean([results[t][key]['precision']['1'] for t in VALID])),
          mean_mrr=float(np.mean([results[t][key]['conditional_mrr'] for t in VALID])),
          nonworse_recall1=int(sum(results[t][key]['recall']['1']>=results[t]['B0']['recall']['1'] for t in VALID)),
          nonworse_recall5=int(sum(results[t][key]['recall']['5']>=results[t]['B0']['recall']['5'] for t in VALID)))
    def order(k):
        a=agg[k];alpha=0 if k=='B0' else float(k.rsplit('-',1)[1]) if '-rrf-' in k else 99
        return (-a['mean_recall1'],-a['mean_recall5'],-a['mean_precision1'],-a['mean_mrr'],alpha,k)
    selected=sorted(keys,key=order)[0];s=agg[selected];b=agg['B0']
    gates=dict(recall1=s['mean_recall1']>b['mean_recall1'],recall5=s['mean_recall5']>b['mean_recall5'],
      precision1=s['mean_precision1']>b['mean_precision1'],nonworse_recall1=s['nonworse_recall1']>=2,nonworse_recall5=s['nonworse_recall5']>=2)
    out=dict(status='completed',stage=contract['stage'],selected=selected,aggregate=agg,windows=results,gates=gates,
      candidate_gate_pass=all(gates.values()),next='admission' if all(gates.values()) else 'stop_content_route',
      seconds=time.perf_counter()-started,final_week='not_run',four_window_map='not_run',fullscale='not_run')
    dump(report/'PSEUDOCOLD_E3.json',out);print(json.dumps(dict(selected=selected,selected_metrics=s,B0=b,gates=gates,seconds=out['seconds']),indent=2),flush=True)

if __name__=='__main__':
  try:run(Path.cwd())
  except Exception:
    dump(Path('reports/final')/f'PSEUDOCOLD_E3_FAILURE_{time.time_ns()}.json',dict(error=traceback.format_exc(),final_week='not_run'))
    raise
