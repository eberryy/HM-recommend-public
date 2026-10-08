"""Independent feature spot checks and full candidate metric replay; no fits."""
from pathlib import Path
import json
import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from .final_oracle_audit import dump
from .final_relation_pilot import ATTRS,REL,EXTRA,VALID,append_features
from .p42f_contract import GLOBAL
from .p42f_core import batches


def run(repo):
    repo=Path(repo);root=repo/'artifacts/final/relation-pilot-v1';source=repo/'artifacts/final/integration-10pct-v1'
    report=repo/'reports/final';result=json.loads((report/'RELATION_PILOT.json').read_text(encoding='utf-8'))
    contract=json.loads((report/'RELATION_PILOT_CONTRACT.json').read_text(encoding='utf-8'))
    cat=pd.read_csv(repo/'data/raw/articles.csv',dtype={'article_id':str},usecols=['article_id',*ATTRS]).set_index('article_id')
    feature_error=score_error=0.;checked=0
    records={}
    for t in contract['prepared_dates']:
        d=joblib.load(source/'prepared'/t/'data.joblib')
        cf=np.load(root/t/'cold.npy');wf=np.load(root/t/'warm.npy')
        keys=pd.concat([d['cold'][['customer_id','article_id']],d['warm'][['customer_id','article_id']]],ignore_index=True)
        full=np.concatenate([cf,wf.reshape(-1,len(REL))])
        ix=np.unique(np.linspace(0,len(keys)-1,32,dtype=int));sample=keys.iloc[ix]
        wanted=sample[['customer_id']].drop_duplicates()
        cutoff=pd.Timestamp(t)
        with duckdb.connect(config={'threads':4,'memory_limit':'2GB'}) as db:
            db.register('wanted',wanted)
            h=db.execute('SELECT customer_id,article_id,t_dat FROM read_parquet(?) JOIN wanted USING(customer_id) '
                         'WHERE t_dat<?::DATE AND t_dat>=?::DATE-INTERVAL 84 DAY',
                         [str(repo/'data/interim/audit/transactions.parquet'),t,t]).fetchdf()
        h=h.join(cat,on='article_id');h['date']=pd.to_datetime(h.t_dat)
        for row_idx in ix:
            row=keys.iloc[row_idx];hist=h[h.customer_id==row.customer_id]
            out=[]
            for a in ATTRS:
                same=hist[hist[a]==cat.loc[row.article_id,a]]
                recent=hist[hist.date>=cutoff-pd.Timedelta(days=28)]
                recent_same=same[same.date>=cutoff-pd.Timedelta(days=28)]
                out.extend([len(recent_same)/len(recent) if len(recent) else 0,
                    len(same)/len(hist) if len(hist) else 0,
                    (cutoff-same.date.max()).days if len(same) else np.nan])
            np.testing.assert_allclose(full[row_idx],out,atol=1e-6,rtol=0,equal_nan=True)
            feature_error=max(feature_error,float(np.nanmax(np.abs(full[row_idx]-out))))
            checked+=1
        if t not in VALID:continue
        table=d['cold'][['customer_id','article_id','b0_rank','target']].copy();table['original_row']=np.arange(len(table))
        records[t]={}
        for arm in ['B0','base70','plus27']:
            s=-table.b0_rank.to_numpy() if arm=='B0' else np.load(root/t/(arm+'-candidate-scores.npy'))
            table['score']=s
            ordered=table.sort_values(['customer_id','score','original_row'],ascending=[True,False,True])
            ordered['rank']=ordered.groupby('customer_id').cumcount()+1
            y=ordered[ordered.target==1]
            mrr=float((1/y.groupby('customer_id')['rank'].min()).mean())
            expected=result['windows'][t][arm]
            assert abs(mrr-expected['conditional_mrr'])<1e-12
            for k in [1,5]:
                top=ordered[ordered['rank']<=k];hits=int(top.target.sum())
                assert hits==expected['topk_positive_pairs'][str(k)]
                assert abs(hits/len(top)-expected['precision'][str(k)])<1e-12
            if arm!='B0':
                model=lgb.Booster(model_file=str(root/t/(arm+'.txt')))
                assert model.feature_name()==(GLOBAL if arm=='base70' else GLOBAL+EXTRA)
                lo,cc,x,_=next(batches(d,size=64,labels=False))
                if arm=='plus27':x=append_features(x,cf[:len(cc)],wf,cc.user_index)
                predicted=model.predict(x,num_threads=4).reshape(-1,12).max(axis=1)
                np.testing.assert_allclose(predicted,s[:len(cc)],atol=1e-12,rtol=0)
                score_error=max(score_error,float(np.max(np.abs(predicted-s[:len(cc)]))))
                gain=model.feature_importance(importance_type='gain')
                records[t][arm]=dict(added_feature_gain_share=float(gain[len(GLOBAL):].sum()/gain.sum()),
                    used_added_features=int((model.feature_importance()[len(GLOBAL):]>0).sum()))
    out=dict(status='pass',feature_sample_rows=checked,feature_max_error=feature_error,
        full_candidate_metrics_replayed=True,model_score_sample_max_error=score_error,feature_usage=records,
        limits='sampled real feature/persisted model verification; full metric replay; not independent refit',final_week='not_run')
    dump(report/'RELATION_PILOT_VERIFICATION.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':run(Path.cwd())
