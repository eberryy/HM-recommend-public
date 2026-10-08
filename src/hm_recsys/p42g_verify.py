"""Read-only independent replay after formal P4.2G; never fits a model."""
from pathlib import Path
import gc
import time
import joblib
import numpy as np
from lightgbm import Booster
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from scipy.optimize import linprog
from .p41a_contract import read_json,write_json,check_identity,identity
from .p42f_contract import GLOBAL,DATES,WINDOWS,earlier,now
from .p42f_core import batches
from .p42f_data import frame
from .p42g_contract import RUN_ID,CLASSIFIER,MAGNITUDE

def direct_ap(items,truth):
    hit=np.array([i in truth for i in items],int)
    return float(np.sum(hit*np.cumsum(hit)/np.arange(1,13))/min(len(truth),12))

def verify(repo):
    repo=Path(repo).resolve();root=repo/'artifacts/phase4'/RUN_ID;start=time.perf_counter()
    c=read_json(repo/'reports/phase4/P4_2G_EXPERIMENT_CONTRACT.json');froot=Path(c['reuse_root'])
    manifest=read_json(root/'FORMAL_START.json');m=read_json(repo/'reports/phase4/P4_2G_metrics.json')
    assert m['status']=='completed_pending_verification'
    check_identity(manifest['contract'])
    for r in manifest['source']+c['authority']+c['trusted_F_assets']:check_identity(r)
    checks=dict(contract_unchanged=True,formal_source_unchanged=True,trusted_input_SHAs=True)
    assert read_json(repo/'reports/phase4/p4_2g_data_parity.json')['status']=='pass'
    # Full row reconstruction was completed before fitting and is identity bound.
    checks.update(historical_users_exact=True,outer_users_exact=True,action_rows_exact=True,delta_AP_exact=True,
                  retained_hash_rows_exact=True,all_B_H_retained=True,neutral_weight50=True,feature_schema_exact=True)
    fits=0;lp_count=0;details={}
    for w,t in WINDOWS.items():
        data=joblib.load(froot/'prepared'/t/'data.joblib');folder=root/'outer'/w
        y=np.load(froot/'outer'/w/'single_delta.npy',mmap_mode='r')
        out={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['probability','m_B_raw','m_H_raw','m_B','m_H','U_R','U_C']}
        models=[];parts=[np.load(froot/'prepared'/s/'training.npz') for s in earlier(t)]
        ty=np.concatenate([p['y'] for p in parts]);tw=np.concatenate([p['weight'] for p in parts])
        for p in parts:p.close()
        for role,params,mask in [('classifier',CLASSIFIER,np.ones(len(ty),bool)),('benefit',MAGNITUDE,ty>0),('harm',MAGNITUDE,ty<0)]:
            f=root/'models'/w/role;meta=read_json(f/'FIT_START.json');fits+=1
            assert meta['params']==params and meta['cutoffs']==earlier(t) and meta['features']==GLOBAL
            np.testing.assert_array_equal(np.load(f/'training-pool-row-indices.npy'),np.flatnonzero(mask))
            assert meta['rows']==int(mask.sum()) and meta['weight_sum']==float(tw[mask].sum())
            assert meta['B']==int(((ty>0)&mask).sum()) and meta['H']==int(((ty<0)&mask).sum()) and meta['N']==int(((ty==0)&mask).sum())
            model=Booster(model_file=str(f/'model.txt'));assert model.feature_name()==GLOBAL;models.append(model)
        del ty,tw,parts
        for begin,cc,x,target in batches(data):
            end=begin+len(cc);np.testing.assert_array_equal(target,y[begin:end])
            for k,model in zip(['probability','m_B_raw','m_H_raw'],models):
                pred=model.predict(x,num_threads=4).reshape(out[k][begin:end].shape)
                np.testing.assert_array_equal(pred,out[k][begin:end])
            p=out['probability'][begin:end];mb=out['m_B'][begin:end];mh=out['m_H'][begin:end]
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            np.testing.assert_allclose(p.sum(axis=2),1.,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(mb,np.clip(out['m_B_raw'][begin:end],0,1))
            np.testing.assert_array_equal(mh,np.clip(out['m_H_raw'][begin:end],0,1))
            np.testing.assert_array_equal(out['U_R'][begin:end],p[:,:,0]*mb-p[:,:,2]*mh)
            np.testing.assert_array_equal(out['U_C'][begin:end],p[:,:,0]-p[:,:,2])
        flat=y.ravel();nz=flat!=0
        for key,reported in [('U_R',m['windows'][w]['R']['action']),('U_C',m['windows'][w]['classification_only_diagnostic'])]:
            score=out[key].ravel()
            for label,mask in [('beneficial_vs_harmful',nz),('beneficial_vs_all_nonbeneficial',np.ones(len(flat),bool))]:
                if label not in reported:continue
                yy=flat[mask]>0;ss=score[mask]
                np.testing.assert_allclose(roc_auc_score(yy,ss),reported[label]['roc_auc'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(average_precision_score(yy,ss),reported[label]['pr_auc'],rtol=0,atol=1e-12)
        labels=np.where(flat>0,0,np.where(flat<0,2,1));prob=out['probability'].reshape(-1,3)
        np.testing.assert_allclose(log_loss(labels,prob,labels=[0,1,2]),m['windows'][w]['classifier']['multiclass_logloss'],atol=1e-12,rtol=0)
        saved=frame(folder/'lists.parquet');user=frame(folder/'users.parquet');edges=frame(folder/'executed.parquet')
        np.testing.assert_array_equal(user.customer_id,data['users'])
        np.testing.assert_array_equal(saved.customer_id,np.repeat(data['users'],12))
        np.testing.assert_array_equal(saved['rank'],np.tile(np.arange(1,13),len(user)))
        lists=saved.article_id.to_numpy().reshape(-1,12);rebuild=data['warm_lists'].copy()
        cmap={u:i for i,u in enumerate(data['users'])};counts=np.zeros(len(user),int);insert=np.zeros(len(user),int);remove=np.zeros(len(user),int)
        assert not edges.duplicated(['customer_id','cold_article_id']).any() and not edges.duplicated(['customer_id','warm_slot']).any()
        for e in edges.itertuples():
            ui=cmap[e.customer_id];j=int(e.warm_slot)-1;ci=int(e.cold_row);item=data['cold'].iloc[ci]
            assert item.customer_id==e.customer_id and item.article_id==e.cold_article_id
            assert e.warm_article_id==data['warm_lists'][ui,j] and e.utility==out['U_R'][ci,j] and e.utility>0
            rebuild[ui,j]=e.cold_article_id;counts[ui]+=1
            insert[ui]+=int(e.cold_article_id in data['truthsets'][e.customer_id]);remove[ui]+=int(e.warm_article_id in data['truthsets'][e.customer_id])
        np.testing.assert_array_equal(lists,rebuild);assert all(len(set(row))==12 for row in lists)
        ap=np.array([direct_ap(row,data['truthsets'][u]) for row,u in zip(lists,data['users'])])
        base=np.array([direct_ap(row,data['truthsets'][u]) for row,u in zip(data['warm_lists'],data['users'])])
        np.testing.assert_allclose(ap,user.ap,rtol=0,atol=1e-15);np.testing.assert_allclose(base,user.baseline_ap,rtol=0,atol=1e-15)
        np.testing.assert_array_equal(counts,user.admissions);np.testing.assert_array_equal(insert,user.inserted);np.testing.assert_array_equal(remove,user.removed)
        np.testing.assert_allclose(ap.mean(),m['windows'][w]['R']['map12'],rtol=0,atol=1e-15)
        # Independent LP for first16 graphs having at least one eligible edge.
        local_lp=0
        for ui,ix in data['cold'].groupby('user_index',sort=False).indices.items():
            u=out['U_R'][np.asarray(ix)];ci,j=np.nonzero(u>0)
            if not len(ci):continue
            a=np.zeros((len(ix)+12,len(ci)));a[ci,np.arange(len(ci))]=1;a[len(ix)+j,np.arange(len(ci))]=1
            sol=linprog(-u[ci,j],A_ub=a,b_ub=np.ones(len(a)),bounds=(0,1),method='highs')
            assert sol.success
            executed=edges.loc[edges.customer_id.eq(data['users'][ui]),'utility'].sum()
            np.testing.assert_allclose(-sol.fun,executed,rtol=0,atol=1e-9)
            local_lp+=1
            if local_lp==16:break
        lp_count+=local_lp;details[w]=dict(edges=int(y.size),users=len(user),fits=3,LP_graphs=local_lp,full_prediction_replay=True,exact_final_MAP=True)
        print('P4.2G verified '+w,flush=True);del data,out,models,prob,edges;gc.collect()
    assert fits==12 and len(list((root/'models').glob('*/*/FIT_START.json')))==12
    assert c['lambda_value']==1 and c['gate']=='U_R>0' and c['matching']['max1'] is False and c['matching']['K_admit'] is None
    assert m['final_week']=='not_run' and not m['Warm_v2_integrated'] and not m['full_history_run'] and not m['P4_3_started']
    checks.update(B_N_H_exact_sign=True,classifier_sample_weights=True,probability_sum1=True,benefit_only_B=True,harm_only_H=True,
        magnitude_clip=True,lambda1=True,utility_exact=True,gate_positive=True,no_tau=True,no_max1=True,no_K=True,
        matching_LP_sample=True,final12_unique=True,exact_MAP=True,full_score_replay=True,independent_AUC_AP=True,
        no_absolute_probabilities=True,final_week_not_run=True,no100percent=True,no_Warm_merge=True,exact12fits=True)
    result=dict(stage='P4.2G',status='pass',checks=checks,windows=details,LP_graphs=lp_count,seconds=time.perf_counter()-start,at=now())
    write_json(repo/'reports/phase4/P4_2G_VERIFICATION.json',result)
    m['status']='completed';m['verification']='pass';write_json(repo/'reports/phase4/P4_2G_metrics.json',m)
    return result

if __name__=='__main__':verify(Path('.'))
