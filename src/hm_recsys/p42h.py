"""P4.2H CLI: exact F/G reuse, four user-OOF conditional heads per outer."""
import argparse
import gc
import shutil
import time
import traceback
from pathlib import Path
import joblib
import numpy as np
from lightgbm import Booster, LGBMClassifier
from threadpoolctl import threadpool_limits
from .p41a_contract import read_json,write_json,identity,check_identity
from .p42_contract import branch_guard
from .p42f_contract import DATES,WINDOWS,GLOBAL,earlier,now
from .p42f_core import batches,sample_edges,fold
from .p42g_core import classes
from .p42h_contract import RUN_ID,Q_PARAMS,R_PARAMS,PLATT_PARAMS,register
from .p42h_core import fit_platt,calibrated,binary_calibration,verdict
from .p42e_resources import rss,memory

def parity(repo,c,root):
    froot=Path(c['reuse_root']);result={}
    # Credible integrity question: older cached rows/models must match their trusted manifests.
    for record in c['trusted_F_assets']+c['trusted_G_assets']+c['authority']: check_identity(record)
    assert c['features']==GLOBAL==read_json(repo/'reports/phase4/p4_2f_feature_contract.json')['G']
    ga=read_json(repo/'reports/phase4/p4_2g_data_parity.json')['cutoffs']
    for t in sorted(set(DATES+list(WINDOWS.values()))):
        data=joblib.load(froot/'prepared'/t/'data.joblib');total=np.zeros(3,np.int64);cursor=0
        if t in DATES:
            with np.load(froot/'prepared'/t/'training.npz') as z: part={k:z[k] for k in z.files}
        else:part=None
        w=next((w for w,d in WINDOWS.items() if d==t),None)
        yy=np.load(froot/'outer'/w/'single_delta.npy',mmap_mode='r') if w else None
        assert data['cutoff']==t and t<'2020-09-16'
        np.testing.assert_array_equal(data['state'].customer_id,data['users'])
        for start,cc,x,y in batches(data):
            total+=np.bincount(classes(y).ravel(),minlength=3)
            if yy is not None:np.testing.assert_array_equal(y,yy[start:start+len(cc)])
            if part is not None:
                keep,weight=sample_edges(t,cc.customer_id.to_numpy(),cc.article_id.to_numpy(),y)
                n=int(keep.sum());end=cursor+n;ui=np.broadcast_to(cc.user_index.to_numpy()[:,None],y.shape)[keep]
                np.testing.assert_array_equal(part['X'][cursor:end],x[keep.ravel()])
                for k,value in [('y',y[keep]),('weight',weight),('user_index',ui),('fold',np.array([fold(u) for u in data['users'][ui]]))]:
                    np.testing.assert_array_equal(part[k][cursor:end],value)
                cursor=end
        if part is not None: assert cursor==len(part['y'])==ga[t]['retained_rows']
        np.testing.assert_array_equal(total,[ga[t][k] for k in ['B','N','H']])
        result[t]=dict(users=len(data['users']),action_users=int(data['cold'].user_index.nunique()),cold_rows=len(data['cold']),
            edges=int(total.sum()),B=int(total[0]),N=int(total[1]),H=int(total[2]),retained_rows=cursor if part is not None else None,
            exact_features_labels_weights_users_and_retention=True,outer_saved_labels_exact=yy is not None)
        for k in ['users','action_users','cold_rows','edges']:assert result[t][k]==ga[t][k]
        write_json(root/'PARITY_PROGRESS.json',result);print('P4.2H parity '+t,flush=True)
        del data,part,yy;gc.collect()
    out=dict(stage='P4.2H',status='pass',cutoffs=result,feature_columns=GLOBAL,feature_count=70,
        trusted_F_files=len(c['trusted_F_assets']),trusted_G_files=len(c['trusted_G_assets']),
        mode='complete retained-feature/label/hash/user-fold reconstruction against immutable F plus exact G census',final_week='not_run')
    write_json(repo/'reports/phase4/p4_2h_data_parity.json',out);return out

def preflight(repo):
    repo=Path(repo).resolve();branch_guard(repo);root=repo/'artifacts/phase4'/RUN_ID
    root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    c=read_json(repo/'reports/phase4/P4_2H_EXPERIMENT_CONTRACT.json')
    try:
        free=memory().available/2**30
        if free<c['budget']['min_free_ram_gib']:raise MemoryError(f'free RAM {free:.2f}GiB below registered6GiB')
        if shutil.disk_usage(repo).free<15*2**30:raise RuntimeError('disk below15GiB')
        write_json(root/'PREFLIGHT_START.json',dict(at=now(),free_ram_gib=free,contract=identity(repo/'reports/phase4/P4_2H_EXPERIMENT_CONTRACT.json')))
        parity(repo,c,root)
        write_json(root/'PREFLIGHT.json',dict(status='pass',seconds=time.perf_counter()-start,peak_gib=rss()/2**30))
    except Exception:
        write_json(root/'PREFLIGHT_FAILURE.json',dict(at=now(),traceback=traceback.format_exc(),fits=0,final_week='not_run'));raise

def train(root,froot,groot,w,guard):
    cutoffs=earlier(WINDOWS[w]);parts=[]
    for t in cutoffs:
        with np.load(froot/'prepared'/t/'training.npz') as p:parts.append({k:p[k] for k in ['X','y','weight','fold']})
    x=np.concatenate([p['X'] for p in parts]);y=np.concatenate([p['y'] for p in parts]);weight=np.concatenate([p['weight'] for p in parts])
    folds=np.concatenate([p['fold'] for p in parts]);del parts;gc.collect()
    nz=y!=0;assert (weight[nz]==1).all() and (weight[~nz]==50).all()
    bhrows=np.flatnonzero(nz);bhy=(y[nz]>0).astype(np.int32);bhfold=folds[nz]
    oof=np.full(len(bhrows),np.nan);count=np.zeros(len(bhrows),np.uint8);audits={};models={}
    for role,params,mask in [('q_fold0',Q_PARAMS,nz&(folds==0)),('q_fold1',Q_PARAMS,nz&(folds==1)),
                              ('q_full',Q_PARAMS,nz),('r',R_PARAMS,np.ones(len(y),bool))]:
        guard();folder=root/'models'/w/role;folder.mkdir(parents=True,exist_ok=False)
        target=(y!=0).astype(np.int32) if role=='r' else (y>0).astype(np.int32)
        assert set(target[mask].tolist())=={0,1};rows=np.flatnonzero(mask)
        np.save(folder/'training-pool-row-indices.npy',rows)
        weights=weight[mask] if role=='r' else np.ones(len(rows))
        meta=dict(at=now(),params=params,cutoffs=cutoffs,rows=len(rows),B=int((y[mask]>0).sum()),N=int((y[mask]==0).sum()),
            H=int((y[mask]<0).sum()),weight_sum=float(weights.sum()),features=GLOBAL,role=role,weighted=role=='r')
        write_json(folder/'FIT_START.json',meta);start=time.perf_counter()
        model=LGBMClassifier(**params)
        with threadpool_limits(limits=4):
            if role=='r':model.fit(x[mask],target[mask],sample_weight=weight[mask],feature_name=GLOBAL)
            else:model.fit(x[mask],target[mask],feature_name=GLOBAL)
        np.testing.assert_array_equal(model.classes_,[0,1]);booster=model.booster_
        assert booster.feature_name()==GLOBAL;booster.save_model(str(folder/'model.txt'))
        audits[role]=dict(meta,seconds=time.perf_counter()-start,num_trees=booster.num_trees(),target_mean=float(target[mask].mean()))
        write_json(folder/'FIT_RESULT.json',audits[role])
        if role.startswith('q_fold'):
            trained_fold=int(role[-1]);test=bhfold!=trained_fold
            np.save(folder/'prediction-pool-row-indices.npy',bhrows[test])
            oof[test]=booster.predict(x[bhrows[test]],num_threads=4);count[test]+=1
        else:models['q' if role=='q_full' else 'r']=booster
        print(f'P4.2H {w} {role} fitted {len(rows)} rows',flush=True)
        del model,booster;gc.collect()
    assert (count==1).all() and np.isfinite(oof).all()
    folder=root/'models'/w/'calibrator';folder.mkdir(parents=True,exist_ok=False)
    # Persist OOF evidence before the calibration fit, including failures.
    np.savez(folder/'oof-input.npz',pool_row_indices=bhrows,fold=bhfold,write_count=count,q_raw=oof,target=bhy)
    write_json(folder/'FIT_START.json',dict(at=now(),OOF_only=True,rows=len(bhrows),params=PLATT_PARAMS))
    start=time.perf_counter();cal=fit_platt(oof,bhy);oq=calibrated(oof,cal)
    cal.update(OOF_only=True,at=now(),seconds=time.perf_counter()-start,
        raw=binary_calibration(oof,bhy,False),calibrated=binary_calibration(oq,bhy,False))
    np.savez(folder/'oof.npz',pool_row_indices=bhrows,fold=bhfold,write_count=count,q_raw=oof,target=bhy,q_cal=oq)
    write_json(folder/'CALIBRATION.json',cal);models['calibrator']=cal;audits['calibrator']=cal
    reuse={}
    for role in ['benefit','harm']:
        source=groot/'models'/w/role;models[role]=Booster(model_file=str(source/'model.txt'))
        assert models[role].feature_name()==GLOBAL
        reuse[role]=dict(model=identity(source/'model.txt'),training_rows=identity(source/'training-pool-row-indices.npy'),
            fit_start=identity(source/'FIT_START.json'),fit_result=identity(source/'FIT_RESULT.json'),new_fits=0)
    write_json(root/'models'/w/'magnitude_reuse.json',reuse);audits['magnitude_reuse']=reuse
    del x,y,weight,folds;gc.collect();return models,audits

def run(repo):
    from .p42h_evaluate import evaluate
    repo=Path(repo).resolve();branch_guard(repo);root=repo/'artifacts/phase4'/RUN_ID
    c=read_json(repo/'reports/phase4/P4_2H_EXPERIMENT_CONTRACT.json');froot=Path(c['reuse_root']);groot=Path(c['g_reuse_root'])
    assert read_json(root/'PREFLIGHT.json')['status']=='pass'
    if (root/'FORMAL_START.json').exists():raise FileExistsError('formal attempt already exists; no silent retry')
    check_identity(read_json(root/'PREFLIGHT_START.json')['contract'])
    write_json(root/'FORMAL_START.json',dict(at=now(),contract=identity(repo/'reports/phase4/P4_2H_EXPERIMENT_CONTRACT.json'),
        parity=identity(repo/'reports/phase4/p4_2h_data_parity.json'),preflight=identity(root/'PREFLIGHT.json'),
        source=[identity(p) for p in sorted((repo/'src/hm_recsys').glob('p42h*.py'))],
        shared_source=[identity(repo/'src/hm_recsys'/n) for n in ['p42_matching.py','p42_evaluate.py','p42d_stats.py','p42g_core.py','p42e_resources.py','metrics.py']],
        tests=[identity(p) for p in sorted((repo/'tests').glob('test_p42h*.py'))]))
    start=time.perf_counter();state=dict(stage='P4.2H',run_id=RUN_ID,status='running',windows={},training={},
        final_week='not_run',Warm_v2_integrated=False,full_history_run=False,P4_3_started=False)
    def guard():
        if time.perf_counter()-start>c['budget']['formal_max_seconds']:raise TimeoutError('formal budget reached')
        if rss()>c['budget']['process_peak_gib']*2**30:raise MemoryError('peak process RAM>6GiB')
        if shutil.disk_usage(repo).free<15*2**30:raise RuntimeError('free disk<15GiB')
    try:
        for w,t in WINDOWS.items():
            guard();models,fit=train(root,froot,groot,w,guard);state['training'][w]=fit
            data=joblib.load(froot/'prepared'/t/'data.joblib')
            state['windows'][w]=evaluate(repo,data,models,root/'outer'/w,froot/'outer'/w,groot/'outer'/w,guard)
            write_json(repo/'reports/phase4/P4_2H_metrics.json',state);print('P4.2H completed '+w,flush=True)
            del models,data;gc.collect()
        state['verdicts']=verdict(state['windows'],read_json(repo/'reports/phase4/P4_2G_metrics.json')['windows'],
            read_json(repo/'reports/phase4/p4_2f_global_utility.json')['data'])
        guard();state['status']='completed_pending_verification'
    except Exception:
        state['status']='engineering_failure';state['error']=traceback.format_exc()
        write_json(root/'FAILURE_FORMAL.json',dict(at=now(),error=state['error'],selected='W0'));raise
    finally:
        state.update(seconds=time.perf_counter()-start,peak_gib=rss()/2**30,finished_at=now())
        write_json(repo/'reports/phase4/P4_2H_metrics.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','preflight','run']);p.add_argument('--repo',default='.')
    a=p.parse_args();{'register':register,'preflight':preflight,'run':run}[a.command](a.repo)
