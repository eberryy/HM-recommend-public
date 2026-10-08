"""Independent single-policy replay and old-AP verification; no new selection."""
import json
import gc
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from scipy.stats import rankdata
from .final_integration import ROOT, RUN, REPORT, CONTRACT, read, write, WINDOWS, GLOBAL, earlier
from .metrics import apk
from .p42f_core import batches
from .p42f_data import frame


def verify():
    contract=read(CONTRACT);result={}
    for w,t in WINDOWS.items():
        folder=RUN/'models'/w; row=read(folder/'RESULT.json')
        fit=read(folder/'FIT_START.json')
        assert fit['params']==contract['model']['params'] and fit['features']==GLOBAL and fit['dates']==earlier(t)
        assert fit['row_weight']==1
        data=joblib.load(RUN/'prepared'/t/'data.joblib')
        model=lgb.Booster(model_file=str(folder/'model.txt'));assert model.feature_name()==GLOBAL
        scores=np.load(folder/'scores.npy');flat=scores.ravel()
        _,cc,x,_=next(batches(data,size=128,labels=False))
        np.testing.assert_allclose(model.predict(x,num_threads=4).reshape(-1,12),scores[:len(cc)],rtol=0,atol=0)
        percent=(rankdata(flat,method='average')-1)/(len(flat)-1) if len(flat)>1 else np.full(len(flat),.5)
        percent=percent.reshape(scores.shape)
        selected=[];lists=data['warm_lists'].copy()
        for ui,indices in data['cold'].groupby('user_index',sort=False).indices.items():
            # With frozen top_edge=1 and max_admissions=1, matching reduces to
            # retaining that first maximum only if it meets all remaining gates.
            ci,slot=divmod(int(np.argmax(scores[indices].ravel())),12)
            ix=int(indices[ci]);candidate=data['cold'].iloc[ix]
            if percent[ix,slot] < .999 or candidate.b0_rank>5: continue
            selected.append((data['users'][ui],candidate.article_id,lists[ui,slot],slot+1))
            lists[ui,slot]=candidate.article_id
        actual=frame(folder/'executed.parquet')
        keys=['customer_id','cold_article_id','removed_article_id','slot']
        assert set(selected)==set(map(tuple,actual[keys].to_numpy()))
        output=frame(folder/'final-lists.parquet').sort_values(['customer_id','rank'])
        np.testing.assert_array_equal(output.article_id.to_numpy().reshape(-1,12),lists)
        assert all(len(set(items))==12 for items in lists)
        truths=data['truth'];n=truths.interaction_count_before_cutoff
        masks={'overall':np.ones(len(truths),bool),'warm_21_plus':n>=21,'strict_cold':n==0,
               'sparse1_5':(n>=1)&(n<=5),'all_cold_sparse':n<=5}
        metrics={}
        for name,mask in masks.items():
            ts={u:set(g.article_id) for u,g in truths.loc[mask].groupby('customer_id',sort=False)}
            pairs=[(i,u) for i,u in enumerate(data['users']) if u in ts]
            before=np.array([apk(list(ts[u]),list(data['warm_lists'][i])) for i,u in pairs])
            after=np.array([apk(list(ts[u]),list(lists[i])) for i,u in pairs])
            expected=row['overall_map'] if name=='overall' else row['segments'][name]['map']
            expected_delta=row['delta_map'] if name=='overall' else row['segments'][name]['delta']
            assert abs(after.mean()-expected)<1e-12 and abs((after-before).mean()-expected_delta)<1e-12
            metrics[name]=dict(users=len(pairs),baseline=float(before.mean()),integrated=float(after.mean()),delta=float((after-before).mean()))
        no_candidates=set(data['users'])-set(data['cold'].customer_id)
        assert not no_candidates.intersection(actual.customer_id)
        result[w]=dict(status='pass',segments=metrics,actions=len(selected),no_candidate_users=len(no_candidates),
            decision_replay='independent scipy average rank percentiles + first argmax, exact frozen top1/max1',
            serialized_model_scores_exact=True,configuration_exact=True)
        del data,scores,flat,percent,lists;gc.collect()
    output=dict(status='pass',windows=result,final_week='not_run',fullscale='not_run',new_models=0,new_policies=0,
        scope='independent decisions and all segment old-AP replay, serialized prediction subset, train parameters/time cuts and unique12')
    write(REPORT/'FINAL_INTEGRATION_10PCT_VERIFICATION.json',output);print(json.dumps(output))


if __name__=='__main__': verify()
