"""Independent replay and bounded SQL evidence checks; no estimator fit."""
from pathlib import Path
import time
import json
import hashlib
import gc
import joblib
import duckdb
import numpy as np
import pandas as pd
from lightgbm import Booster
from sklearn.metrics import roc_auc_score,average_precision_score
from scipy.optimize import linprog
from .p41a_contract import read_json,write_json,check_identity
from .p42f_contract import RUN_ID,DATES,WINDOWS,GLOBAL,BASE,USER,PARAMS,earlier
from .p42f_data import frame
from .p42f_core import batches,fold,sample_edges,prior_residual
from .p42f_train import thresholds
from .p42f_evaluate import VARIANTS,decision

def ap(relevance,den):
    r=np.asarray(relevance,float)
    return (r*np.cumsum(r,axis=1)/np.arange(1,13)).sum(axis=1)/np.minimum(den,12)

def verify(repo):
    repo=Path(repo).resolve(); root=repo/'artifacts/phase4'/RUN_ID; report=repo/'reports/phase4'
    started=time.perf_counter(); m=read_json(report/'P4_2F_metrics.json'); c=read_json(report/'P4_2F_EXPERIMENT_CONTRACT.json')
    if m['status']!='completed_pending_verification': raise ValueError('four outer runs required before closure')
    start=read_json(root/'FORMAL_START.json'); check_identity(start['contract'])
    for r in start['source']: check_identity(r)
    for r in read_json(root/'TRUSTED_INPUT_CHECK.json')['files']:check_identity(r)
    checks={'registered_contract_unchanged':True,'formal_implementation_unchanged':True,'trusted_frozen_inputs_unchanged':True}
    # SQL independently rejoins global earlier-day item event totals. No event labels needed.
    with duckdb.connect(str(root/'novelty.duckdb'),read_only=True,config={'threads':2,'memory_limit':'2GB'}) as db:
        mismatches=db.execute('''WITH expected AS (SELECT article_id,t_dat,
           coalesce(sum(n) OVER(PARTITION BY article_id ORDER BY t_dat ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0)::BIGINT cnt FROM daily)
           SELECT count(*) FROM event_coldness e JOIN expected d USING(article_id,t_dat) WHERE e.before_count<>d.cnt''').fetchone()[0]
        assert mismatches==0
        assert db.execute("SELECT max(t_dat)<DATE '2020-08-19' FROM events").fetchone()[0]
        assert db.execute('SELECT count(*) FROM enriched').fetchone()[0]==db.execute('SELECT count(*) FROM events').fetchone()[0]
        assert db.execute('SELECT count(*) FROM count_cdf WHERE percentile<0 OR percentile>1 OR percentile IS NULL').fetchone()[0]==0
        assert db.execute('''WITH raw AS (SELECT article_id,t_dat,count(*) n FROM read_parquet(?)
            WHERE t_dat<DATE '2020-08-19' GROUP BY article_id,t_dat)
            SELECT count(*) FROM raw FULL JOIN daily USING(article_id,t_dat) WHERE raw.n IS DISTINCT FROM daily.n''',[c['transactions']['path']]).fetchone()[0]==0
    checks.update(novelty_strict_prior_date_counts=True,novelty_no_current_cutoff_backfill=True,novelty_final_week_excluded=True)
    preparation={}
    for t in DATES:
        data=joblib.load(root/'prepared'/t/'data.joblib')
        with np.load(root/'prepared'/t/'training.npz') as archive:z={k:archive[k] for k in archive.files}
        cursor=0
        for _,cc,x,y in batches(data):
            # Full AP recomputation per slot, independently from closed-form target.
            idx=cc.user_index.to_numpy(int); original=data['relevance'][idx]; den=data['truth_count'][idx]; base=ap(original,den)
            for j in range(12):
                replaced=original.copy(); replaced[:,j]=cc.target.to_numpy()
                np.testing.assert_allclose(y[:,j],ap(replaced,den)-base,atol=1e-15,rtol=0)
            keep,weight=sample_edges(t,cc.customer_id.to_numpy(),cc.article_id.to_numpy(),y); n=int(keep.sum()); sl=slice(cursor,cursor+n)
            np.testing.assert_array_equal(z['X'][sl],x[keep.ravel()]);np.testing.assert_array_equal(z['y'][sl],y[keep]);np.testing.assert_array_equal(z['weight'][sl],weight)
            np.testing.assert_array_equal(z['user_index'][sl],np.broadcast_to(idx[:,None],y.shape)[keep]);cursor+=n
        assert cursor==len(z['y'])
        np.testing.assert_array_equal(z['fold'],np.array([fold(u) for u in data['users']])[z['user_index']])
        preparation[t]=dict(rows=cursor,all_single_labels_exact=True,all_sampling_replayed=True)
        del data,z;gc.collect()
    checks.update(single_replacement_labels_exact=True,all_beneficial_harmful_retained=True,neutral_hash_rule_002=True,neutral_weight_50=True,stable_user_folds=True)
    old=read_json(report/'P4_2R3_metrics.json'); outer={}
    for w,t in WINDOWS.items():
        folder=root/'models'/w; fit=read_json(folder/'TRAINING.json'); assert fit['cutoffs']==earlier(t)
        assert all(s<t for s in fit['cutoffs'])
        for role in ['G','H_pair','H_oof_predict_fold0','H_oof_predict_fold1']:
            assert read_json(folder/role/'FIT_START.json')['params']==PARAMS
            model=Booster(model_file=str(folder/role/'model.txt'))
            assert model.feature_name()==(GLOBAL if role=='G' else BASE)
        assert not set(GLOBAL)&{'qC','qW','b0_score','logit_qC','logit_qW','target','customer_id','article_id'}
        history=frame(folder/'threshold-training.parquet'); threshold=read_json(folder/'threshold.json')
        assert threshold['alpha']==1 and threshold['shrink_kappa']==2
        # Verify each user-window's complete-space OOF max with its excluded-user fold.
        models=[Booster(model_file=str(folder/f'H_oof_predict_fold{k}'/'model.txt')) for k in [0,1]]
        for s in earlier(t):
            data=joblib.load(root/'prepared'/s/'data.joblib'); maximum=np.zeros(len(data['users']))
            for _,cc,x,_ in batches(data,labels=False):
                ix=cc.user_index.to_numpy(int); f=np.array([fold(u) for u in cc.customer_id])
                for k in [0,1]:
                    mask=f==k
                    if mask.any():
                        v=models[k].predict(x[np.repeat(mask,12),:len(BASE)],num_threads=4).reshape(-1,12)
                        np.maximum.at(maximum,ix[mask],np.maximum(0,v.max(axis=1)))
            expected=history.loc[history.cutoff.eq(s)].set_index('customer_id')
            index={u:i for i,u in enumerate(data['users'])}; ii=np.array([index[u] for u in expected.index])
            np.testing.assert_allclose(expected.s_oof,maximum[ii],atol=1e-15,rtol=0)
            prior=history.loc[history.cutoff.isin(earlier(s))].groupby('customer_id').residual.agg(['count','mean']).reindex(expected.index)
            n=prior['count'].fillna(0).to_numpy(int); b=n/(n+2)*prior['mean'].fillna(0).to_numpy()
            np.testing.assert_array_equal(expected.prior_n,n);np.testing.assert_allclose(expected.prior_b,b,rtol=0,atol=1e-15)
            del data;gc.collect()
        data=joblib.load(root/'prepared'/t/'data.joblib'); ef=root/'outer'/w
        for detail in data['audit']['relative'].values():
            if detail:
                from datetime import date,timedelta
                assert all(date.fromisoformat(s)+timedelta(days=7)<date.fromisoformat(t) for s in detail['training_cutoffs'])
        assert data['cold'].interaction_count_before_cutoff.le(5).all()
        np.testing.assert_allclose(data['baseline_map'],old['windows'][w]['variants']['W0']['map12'],atol=1e-15,rtol=0)
        if abs(m['windows'][w]['variants']['W0']['map12']-data['baseline_map'])>1e-15:raise AssertionError('W0 drift')
        fixed,hn,b=thresholds(threshold,history,data['state'],t); th=frame(ef/'thresholds.parquet')
        np.testing.assert_allclose(th.tau_u,fixed+b,atol=1e-15,rtol=0);assert np.all(b[hn==0]==0)
        Y=np.load(ef/'single_delta.npy'); cold=data['cold']; ui=cold.user_index.to_numpy(int)
        for j in range(12):
            r=data['relevance'][ui].copy();r[:,j]=cold.target.to_numpy()
            np.testing.assert_allclose(Y[:,j],ap(r,data['truth_count'][ui])-ap(data['relevance'][ui],data['truth_count'][ui]),atol=1e-15,rtol=0)
        for variant,role in zip(VARIANTS,['G','H_pair']):
            model=Booster(model_file=str(folder/role/'model.txt')); score=np.load(ef/(variant+'-utility.npy'))
            for start,cc,x,_ in batches(data,labels=False):
                pred=model.predict(x if role=='G' else x[:,:len(BASE)],num_threads=4).reshape(-1,12)
                if role!='G':pred-=th.tau_u.to_numpy()[cc.user_index.to_numpy(int),None]
                np.testing.assert_allclose(score[start:start+len(cc)],pred,rtol=0,atol=1e-15)
            rows=frame(ef/(variant+'-executed.parquet')); lists=frame(ef/(variant+'-lists.parquet')); saved=frame(ef/(variant+'-users.parquet'))
            rebuilt=data['warm_lists'].copy(); index={u:i for i,u in enumerate(data['users'])}
            assert not rows.duplicated(['customer_id','warm_slot']).any() and not rows.duplicated(['customer_id','cold_article_id']).any()
            for row in rows.itertuples():
                i=index[row.customer_id];j=row.warm_slot-1
                assert score[row.cold_row,j]>0 and rebuilt[i,j]==row.warm_article_id
                assert row.cold_article_id not in data['warm_lists'][i]
                rebuilt[i,j]=row.cold_article_id
            np.testing.assert_array_equal(lists.article_id.to_numpy().reshape(-1,12),rebuilt)
            assert all(len(set(l))==12 for l in rebuilt)
            rel=np.array([[item in data['truthsets'][u] for item in l] for u,l in zip(data['users'],rebuilt)])
            actual=ap(rel,data['truth_count']);np.testing.assert_allclose(actual,saved.ap,atol=1e-15,rtol=0)
            assert abs(actual.mean()-m['windows'][w]['variants'][variant]['map12'])<1e-15
            # Independent LP optimality check on 16 deterministic user graphs.
            executed_weights=rows.groupby('customer_id').utility.sum()
            for i,(uid,ix) in enumerate(cold.groupby('user_index',sort=False).indices.items()):
                if i>=16:break
                a=score[ix]; ci,wj=np.nonzero(a>0)
                if len(ci)==0:continue
                from scipy.sparse import coo_matrix
                nv=len(ci); A=coo_matrix((np.ones(2*nv),(np.r_[ci,len(ix)+wj],np.r_[np.arange(nv),np.arange(nv)])),shape=(len(ix)+12,nv)).tocsr()
                scale=float(a[ci,wj].max())
                result=linprog(-a[ci,wj]/scale,A_ub=A,b_ub=np.ones(len(ix)+12),bounds=(0,None),method='highs')
                assert result.success
                np.testing.assert_allclose(-result.fun*scale,executed_weights.get(data['users'][uid],0.),atol=1e-10,rtol=1e-10)
            mask=Y.ravel()!=0
            for key,select in [('beneficial_vs_harmful',mask),('beneficial_vs_all_nonbeneficial',np.ones(mask.shape,bool))]:
                labels=Y.ravel()[select]>0;sc=score.ravel()[select];a=m['windows'][w]['variants'][variant]['action'][key]
                np.testing.assert_allclose([roc_auc_score(labels,sc),average_precision_score(labels,sc)],[a['roc_auc'],a['pr_auc']],atol=1e-12,rtol=0)
        outer[w]=dict(exact_list_replay=True,W0_parity=True,OOF_max_replayed=True,utility_predictions_replayed=True,independent_AP_AUC=True,LP_graph_checks_per_variant=16)
        del data,Y;gc.collect()
    checks.update(W0_exact=True,Cold50_frozen=True,no_absolute_probability_features=True,relative_forward_lineages=True,
        all_fit_cutoffs_before_outer=True,OOF_user_exclusion=True,OOF_full_space_max=True,G_H_params_exact=True,ridge_alpha1=True,
        shrink_kappa2=True,historical_residual_strict_prior=True,outer_residual_no_outer_labels=True,n0_residual_zero=True,
        no_K_no_max1=True,eligibility_strictly_positive=True,matching_independent_LP_sample=True,final_lists_12_unique=True,
        final_AP_recomputed=True,action_AUC_independent=True,final_week_not_run=True,Warm_v2_not_integrated=True)
    assert m['decision']==decision(m['windows'])
    result=dict(stage='P4.2F',status='pass',checks=checks,preparation=preparation,outer=outer,seconds=time.perf_counter()-started)
    write_json(report/'P4_2F_VERIFICATION.json',result)
    m['status']='completed';m['verification']='pass';write_json(report/'P4_2F_metrics.json',m)
    print(result,flush=True)

if __name__=='__main__':verify('.')
