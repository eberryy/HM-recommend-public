"""FINAL-E2 candidate-first ranker using direct M4 history relation evidence."""
from pathlib import Path
import gc
import json
import shutil
import time
import traceback
import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from .final_oracle_audit import dump, WINDOWS
from .final_relation_pilot import VALID, identify
from .p42f_contract import COLD, USER, earlier
from .p43a_run import guard
from .p37b_features import assign_age_buckets, _cosine_tensor, _bucket_cosine_summaries, _same_attribute_counts

DATES=['2019-12-25','2020-01-22','2020-02-19','2020-03-18','2020-04-29','2020-05-27','2020-06-24','2020-07-22']
M4COL=[f'm4_{bucket}_{field}' for bucket in ['0_7','8_28','29_84','over_84'] for field in
       ['bucket_present','history_count','max_cosine','top3_mean_cosine','same_product_type_count','same_garment_group_count']]
BASE=COLD+USER
PARAMS=dict(objective='lambdarank',metric='None',learning_rate=.05,n_estimators=300,num_leaves=15,max_depth=6,
    min_child_samples=50,subsample=1.,colsample_bytree=1.,reg_lambda=1.,reg_alpha=0.,random_state=20260913,
    n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True,label_gain=[0,1])
ALPHAS=[.25,.5,1.,2.,4.]


def embedding_path(repo,t):
    inv={d:w for w,d in WINDOWS.items()}
    if t in inv:
        return repo/'artifacts/m4/m4-v1-supervised-cold-representation/student-v1'/inv[t]/'multimodal/catalog_embeddings.float16.npy'
    return repo/'artifacts/m5/m5-v1-cold-expert-admission/train-data-v1'/t/'student/catalog_embeddings.float16.npy'


def histories(repo,users,t,article_to_row):
    wanted=pd.DataFrame({'customer_id':users})
    with duckdb.connect(config={'threads':4,'memory_limit':'2GB'}) as db:
        db.register('wanted',wanted)
        h=db.execute('''WITH latest AS (
          SELECT x.customer_id,x.article_id,max(x.t_dat) latest_date
          FROM read_parquet(?) x JOIN wanted USING(customer_id)
          WHERE x.t_dat<?::DATE GROUP BY x.customer_id,x.article_id), ranked AS (
          SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY latest_date DESC,article_id) r FROM latest)
          SELECT customer_id,article_id,r,date_diff('day',latest_date,?::DATE) age FROM ranked WHERE r<=20 ORDER BY customer_id,r''',
          [str(repo/'data/interim/audit/transactions.parquet'),t,t]).fetchdf()
    ui={u:i for i,u in enumerate(users)};rows=np.full((len(users),20),-1,np.int32);days=np.zeros((len(users),20),np.float32)
    for x in h.itertuples(index=False):
        i=ui.get(x.customer_id);j=article_to_row.get(x.article_id)
        if i is not None and j is not None: rows[i,x.r-1]=j;days[i,x.r-1]=x.age
    mask=rows>=0
    assert np.all(days[mask]>=0)
    return rows,days,mask,dict(users=len(users),users_with_history=int(mask.any(1).sum()),history_rows=int(mask.sum()),
        max_age=float(days[mask].max()) if mask.any() else None,latest_input_day=str((pd.Timestamp(t)-pd.Timedelta(days=int(days[mask].min()))).date()) if mask.any() else None)


def relation_batch(candidate_rows,history_rows,history_days,history_mask,embeddings,ptype,garment,device):
    device=torch.device(device)
    bucket=assign_age_buckets(history_days,history_mask)
    sim=_cosine_tensor(candidate_rows=candidate_rows,history_rows=history_rows,history_mask=history_mask,
                       embeddings=embeddings,device=device)
    maximum,top3,count=_bucket_cosine_summaries(sim,bucket)
    product=_same_attribute_counts(candidate_rows=candidate_rows,history_rows=history_rows,bucket_index=bucket,catalog_attribute=ptype)
    garment_count=_same_attribute_counts(candidate_rows=candidate_rows,history_rows=history_rows,bucket_index=bucket,catalog_attribute=garment)
    out=np.empty((*candidate_rows.shape,4,6),np.float32)
    out[:,:,:,0]=(count>0)[:,None,:];out[:,:,:,1]=count[:,None,:]
    out[:,:,:,2]=maximum;out[:,:,:,3]=top3;out[:,:,:,4]=product;out[:,:,:,5]=garment_count
    return out


def build_relation(repo,root,t,catalog,article_to_row,ptype,garment,device):
    folder=root/t;folder.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    data=joblib.load(repo/'artifacts/final/integration-10pct-v1/prepared'/t/'data.joblib')
    users=data['users'];cold=data['cold'];rows,days,mask,audit=histories(repo,users,t,article_to_row)
    embeddings=np.load(embedding_path(repo,t),mmap_mode='r')
    assert embeddings.shape[0]==len(catalog) and embeddings.shape[1]==128
    result=np.lib.format.open_memmap(folder/'m4-relation.npy',mode='w+',dtype=np.float32,shape=(len(cold),24))
    groups=cold.groupby('user_index',sort=False).indices;active=np.array(list(groups),dtype=int)
    for start in range(0,len(active),128):
        guard(repo);uis=active[start:start+128];indices=[np.asarray(groups[i],int) for i in uis]
        k=max(map(len,indices));items=np.zeros((len(uis),k),np.int32);valid=np.zeros((len(uis),k),bool)
        for j,idx in enumerate(indices):
            mapped=np.array([article_to_row[a] for a in cold.iloc[idx].article_id],np.int32)
            items[j,:len(idx)]=mapped;valid[j,:len(idx)]=True
        z=relation_batch(items,rows[uis],days[uis],mask[uis],embeddings,ptype,garment,device)
        for j,idx in enumerate(indices): result[idx]=z[j,:len(idx)].reshape(len(idx),24)
    result.flush();del result,embeddings;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    audit.update(candidates=len(cold),embedding=str(embedding_path(repo,t)),seconds=time.perf_counter()-begin,
        feature_shape=[len(cold),24],label_columns_used=False)
    dump(folder/'RELATION.json',audit);print('RELATION',t,audit['seconds'],flush=True)


def matrix(repo,root,t,arm):
    data=joblib.load(repo/'artifacts/final/integration-10pct-v1/prepared'/t/'data.joblib')
    cold=data['cold'];idx=cold.user_index.to_numpy(int)
    x=np.concatenate([cold[COLD].to_numpy(float),data['state'][USER].to_numpy(float)[idx]],axis=1).astype(np.float32)
    if arm=='C58':x=np.concatenate([x,np.load(root/t/'m4-relation.npy')],axis=1)
    x[~np.isfinite(x)]=np.nan
    group=cold.groupby('user_index',sort=False).size().to_numpy(np.int32)
    assert group.sum()==len(x) and x.shape[1]==len(BASE)+(24 if arm=='C58' else 0)
    return data,x,cold.target.to_numpy(np.int32),group


def ranks_for(data,scores):
    out=np.zeros(len(scores),np.int32);groups=data['cold'].groupby('user_index',sort=False).indices
    for idx in groups.values():
        idx=np.asarray(idx);order=np.argsort(-scores[idx],kind='stable');out[idx[order]]=np.arange(1,len(idx)+1)
    return out


def rrf(data,model_scores,alpha):
    rank=ranks_for(data,model_scores);return 1/(60+data['cold'].b0_rank.to_numpy(float))+alpha/(60+rank)


def run(repo):
    repo=Path(repo);root=repo/'artifacts/final/candidate-e2-v1';report=repo/'reports/final'
    if root.exists():raise FileExistsError('Do not overwrite earlier E2 evidence')
    guard(repo);root.mkdir(parents=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    contract=dict(stage='FINAL-E2 candidate-first direct relation',status='preregistered_before_new_run',
      hypothesis='B0 relation signal was compressed into rank/margins before final decision; candidate-level supervision plus direct relation can restore identification.',
      historical_validation=VALID,training_dates={t:earlier(t) for t in VALID},prepared_dates=DATES,
      arms={'B0':'frozen original order','C34':'candidate LambdaRank with17 Cold+17 user fields','C58':'C34 plus raw4x6 M4 relation fields'},
      m4_relation='For each candidate vs latest20 distinct user purchases, four age buckets0-7,8-28,29-84,>84; bucket present/count, max/top3 cosine, same product type/garment counts.',
      embeddings={t:str(embedding_path(repo,t)) for t in DATES},history='latest20 distinct articles strictly before cutoff; duplicate events only determine latest date, not repeated seeds',
      params=PARAMS,target='binary candidate purchased in next7days; one row per user-item-date, not12 action copies',
      groups='all candidate-bearing user-date groups, including no-positive groups; no future-conditioned population filter',
      feature_missing='NaN native LightGBM; bucket present retained; no outer-fit normalization',
      blends={'formula':'1/(60+B0_rank)+alpha/(60+model_rank)','alpha':ALPHAS,'models':['C34','C58']},
      selection='highest mean Recall@1; tie mean Recall@5, mean Precision@1, mean conditional MRR, lower alpha, simpler model; B0 participates.',
      gate='selected must exceed B0 mean Recall@1 and@5 and Precision@1; Recall@1 and@5 each nonworse at least2/3 dates. Pass permits four development scoring, not MAP promotion.',
      if_pass='freeze candidate selector; next use original edge model only for position, and select a conservative historical admission rule before four-window MAP; no policy grid explosion.',
      if_fail='stop candidate route; do not run four development windows; WV3 fallback.',no_image_retraining=True,
      final_week='not_run',budget=dict(session_user_hours=5,stage_seconds=7200,threads=4,min_disk_gib=15,min_ram_gib=1.5),
      resources=dict(free_disk_gib=shutil.disk_usage(repo).free/2**30,device=str(device)),no_commit_push=True)
    dump(report/'CANDIDATE_E2_CONTRACT.json',contract)
    started=time.perf_counter();deadline=time.time()+7200
    catalog=pd.read_csv(repo/'artifacts/m4/m4-v1-supervised-cold-representation/student-v1/static_catalog/catalog_items.csv',dtype={'article_id':str})
    articles=pd.read_csv(repo/'data/raw/articles.csv',dtype={'article_id':str},usecols=['article_id','product_type_no','garment_group_no']).set_index('article_id')
    aligned=articles.reindex(catalog.article_id);assert not aligned.isna().any().any()
    ptype=aligned.product_type_no.to_numpy(np.int32);garment=aligned.garment_group_no.to_numpy(np.int32)
    article_to_row={a:i for i,a in enumerate(catalog.article_id)}
    for t in DATES:
        guard(repo,deadline);build_relation(repo,root,t,catalog,article_to_row,ptype,garment,device)
    results={}
    for t in VALID:
        data=joblib.load(repo/'artifacts/final/integration-10pct-v1/prepared'/t/'data.joblib')
        results[t]={'B0':identify(data,-data['cold'].b0_rank.to_numpy(float))}
        for arm in ['C34','C58']:
            train=earlier(t);begin=time.perf_counter();xs=[];ys=[];gs=[]
            for d in train:
                _,x,y,g=matrix(repo,root,d,arm);xs.append(x);ys.append(y);gs.append(g)
            x=np.concatenate(xs);y=np.concatenate(ys);groups=np.concatenate(gs)
            assert groups.sum()==len(x)==len(y)
            model=lgb.LGBMRanker(**PARAMS)
            names=BASE+(M4COL if arm=='C58' else [])
            with threadpool_limits(limits=4):model.fit(x,y,group=groups,feature_name=names)
            model.booster_.save_model(str(root/t/(arm+'.txt')))
            del x,y,groups,xs,ys,gs;gc.collect()
            _,vx,_,_=matrix(repo,root,t,arm);score=model.predict(vx,num_threads=4)
            np.save(root/t/(arm+'-scores.npy'),score)
            results[t][arm]=identify(data,score);results[t][arm]['seconds']=time.perf_counter()-begin
            for alpha in ALPHAS:
                key=f'{arm}-rrf-{alpha:g}';results[t][key]=identify(data,rrf(data,score,alpha))
            print('E2',t,arm,results[t][arm],flush=True)
            del vx,score,model;gc.collect()
        dump(root/t/'RESULT.json',results[t]);del data;gc.collect()
    keys=list(results[VALID[0]])
    aggregate={}
    for key in keys:
        aggregate[key]=dict(mean_recall1=float(np.mean([results[t][key]['recall']['1'] for t in VALID])),
            mean_recall5=float(np.mean([results[t][key]['recall']['5'] for t in VALID])),
            mean_precision1=float(np.mean([results[t][key]['precision']['1'] for t in VALID])),
            mean_mrr=float(np.mean([results[t][key]['conditional_mrr'] for t in VALID])),
            nonworse_recall1=int(sum(results[t][key]['recall']['1']>=results[t]['B0']['recall']['1'] for t in VALID)),
            nonworse_recall5=int(sum(results[t][key]['recall']['5']>=results[t]['B0']['recall']['5'] for t in VALID)))
    def order(key):
        a=aggregate[key];alpha=0 if key=='B0' else (float(key.rsplit('-',1)[1]) if '-rrf-' in key else 99)
        complexity=0 if key=='B0' else 1 if key.startswith('C34') else 2
        return (-a['mean_recall1'],-a['mean_recall5'],-a['mean_precision1'],-a['mean_mrr'],alpha,complexity,key)
    selected=sorted(keys,key=order)[0];b=aggregate['B0'];s=aggregate[selected]
    gates=dict(recall1=s['mean_recall1']>b['mean_recall1'],recall5=s['mean_recall5']>b['mean_recall5'],
        precision1=s['mean_precision1']>b['mean_precision1'],nonworse_recall1=s['nonworse_recall1']>=2,
        nonworse_recall5=s['nonworse_recall5']>=2)
    out=dict(status='completed',stage=contract['stage'],selected=selected,aggregate=aggregate,windows=results,gates=gates,
        candidate_gate_pass=all(gates.values()),next='development_admission_only_if_pass' if all(gates.values()) else 'stop_WV3',
        seconds=time.perf_counter()-started,final_week='not_run',four_window_map='not_run',fullscale='not_run')
    dump(report/'CANDIDATE_E2.json',out);print(json.dumps(dict(selected=selected,selected_metrics=s,B0=b,gates=gates,seconds=out['seconds']),indent=2),flush=True)

if __name__=='__main__':
  try:run(Path.cwd())
  except Exception:
    dump(Path('reports/final')/f'CANDIDATE_E2_FAILURE_{time.time_ns()}.json',dict(error=traceback.format_exc(),final_week='not_run'))
    raise
