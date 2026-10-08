"""P4.2G CLI: immutable registration, full parity, then twelve bounded fits."""
import argparse
import gc
import time
import traceback
import shutil
from pathlib import Path
import joblib
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from threadpoolctl import threadpool_limits
from .p41a_contract import read_json,write_json,identity,check_identity
from .p42_contract import branch_guard
from .p42f_contract import DATES,WINDOWS,GLOBAL,earlier,now
from .p42f_core import batches,sample_edges,fold
from .p42g_contract import RUN_ID,CLASSIFIER,MAGNITUDE,register
from .p42g_core import classes,verdict
from .p42e_resources import rss,memory

def parity(repo,c,root):
    """Replay every historical retained row and every outer label without fits."""
    froot=Path(c['reuse_root']); result={}
    for record in c['trusted_F_assets']: check_identity(record)
    assert c['features']==GLOBAL==read_json(repo/'reports/phase4/p4_2f_feature_contract.json')['G']
    oldaudit=read_json(repo/'reports/phase4/p4_2f_action_training_audit.json')['data']['cutoffs']
    for t in sorted(set(DATES+list(WINDOWS.values()))):
        data=joblib.load(froot/'prepared'/t/'data.joblib'); total=np.zeros(3,np.int64); cursor=0
        part=np.load(froot/'prepared'/t/'training.npz') if t in DATES else None
        outer_name=next((w for w,d in WINDOWS.items() if d==t),None)
        yy=np.load(froot/'outer'/outer_name/'single_delta.npy',mmap_mode='r') if outer_name else None
        assert data['cutoff']==t and t<'2020-09-16'
        assert np.array_equal(data['state'].customer_id,data['users'])
        for start,cc,x,y in batches(data):
            total+=np.bincount(classes(y).ravel(),minlength=3)
            if yy is not None: np.testing.assert_array_equal(y,yy[start:start+len(cc)])
            if part is not None:
                keep,weight=sample_edges(t,cc.customer_id.to_numpy(),cc.article_id.to_numpy(),y)
                n=int(keep.sum()); end=cursor+n; idx=cc.user_index.to_numpy()
                np.testing.assert_array_equal(part['X'][cursor:end],x[keep.ravel()])
                np.testing.assert_array_equal(part['y'][cursor:end],y[keep])
                np.testing.assert_array_equal(part['weight'][cursor:end],weight)
                ui=np.broadcast_to(idx[:,None],y.shape)[keep]
                np.testing.assert_array_equal(part['user_index'][cursor:end],ui)
                np.testing.assert_array_equal(part['fold'][cursor:end],[fold(u) for u in data['users'][ui]])
                cursor=end
        if part is not None:
            assert cursor==len(part['y'])==oldaudit[t]['training_rows']
            np.testing.assert_array_equal(total,[oldaudit[t]['beneficial'],oldaudit[t]['neutral'],oldaudit[t]['harmful']])
            part.close()
        result[t]=dict(users=len(data['users']),action_users=int(data['cold'].user_index.nunique()),cold_rows=len(data['cold']),
            edges=int(total.sum()),B=int(total[0]),N=int(total[1]),H=int(total[2]),retained_rows=cursor if t in DATES else None,
            exact_features_labels_weights_users_and_retention=True,outer_saved_labels_exact=yy is not None)
        write_json(root/'PARITY_PROGRESS.json',result); print('P4.2G parity '+t,flush=True)
        del data,part,yy;gc.collect()
    output=dict(stage='P4.2G',status='pass',cutoffs=result,feature_columns=GLOBAL,feature_count=len(GLOBAL),
                trusted_F_files=len(c['trusted_F_assets']),mode='direct immutable reuse plus complete retained-row reconstruction',final_week='not_run')
    write_json(repo/'reports/phase4/p4_2g_data_parity.json',output)
    return output

def preflight(repo):
    repo=Path(repo).resolve();branch_guard(repo);root=repo/'artifacts/phase4'/RUN_ID
    root.mkdir(parents=True,exist_ok=False)
    c=read_json(repo/'reports/phase4/P4_2G_EXPERIMENT_CONTRACT.json');start=time.perf_counter()
    try:
        free=memory().available/2**30
        if free<c['budget']['min_free_ram_gib']: raise MemoryError(f'free RAM {free:.2f} GiB < registered 6 GiB')
        if shutil.disk_usage(repo).free<15*2**30: raise RuntimeError('disk below15GiB')
        write_json(root/'PREFLIGHT_START.json',dict(at=now(),free_ram_gib=free,contract=identity(repo/'reports/phase4/P4_2G_EXPERIMENT_CONTRACT.json')))
        parity(repo,c,root)
        write_json(root/'PREFLIGHT.json',dict(status='pass',seconds=time.perf_counter()-start,peak_gib=rss()/2**30))
    except Exception:
        write_json(root/'PREFLIGHT_FAILURE.json',dict(at=now(),traceback=traceback.format_exc(),fits=0,final_week='not_run'))
        raise

def train(root,froot,w,guard):
    parts=[np.load(froot/'prepared'/t/'training.npz') for t in earlier(WINDOWS[w])]
    x=np.concatenate([p['X'] for p in parts]);y=np.concatenate([p['y'] for p in parts]);weight=np.concatenate([p['weight'] for p in parts])
    for p in parts:p.close()
    del parts;gc.collect();lab=classes(y);assert set(lab)=={0,1,2}
    models=[];audits={}
    for role,params,mask,target in [('classifier',CLASSIFIER,np.ones(len(y),bool),lab),('benefit',MAGNITUDE,y>0,y),('harm',MAGNITUDE,y<0,np.abs(y))]:
        guard();folder=root/'models'/w/role;folder.mkdir(parents=True,exist_ok=False)
        weights=weight[mask];assert role=='classifier' or (weights==1).all()
        rows=np.flatnonzero(mask);np.save(folder/'training-pool-row-indices.npy',rows)
        start=time.perf_counter()
        write_json(folder/'FIT_START.json',dict(at=now(),params=params,cutoffs=earlier(WINDOWS[w]),rows=len(rows),
                   B=int((lab[mask]==0).sum()),N=int((lab[mask]==1).sum()),H=int((lab[mask]==2).sum()),weight_sum=float(weights.sum()),features=GLOBAL))
        model=(LGBMClassifier if role=='classifier' else LGBMRegressor)(**params)
        with threadpool_limits(limits=4): model.fit(x[mask],target[mask],sample_weight=weights,feature_name=GLOBAL)
        if role=='classifier':np.testing.assert_array_equal(model.classes_,[0,1,2])
        booster=model.booster_; assert booster.feature_name()==GLOBAL
        booster.save_model(str(folder/'model.txt'))
        audits[role]=dict(rows=len(rows),seconds=time.perf_counter()-start,params=params,
            target_mean=float(target[mask].mean()),weight_sum=float(weights.sum()),num_trees=booster.num_trees(),
            class_counts={k:int((lab[mask]==i).sum()) for i,k in enumerate(['B','N','H'])})
        write_json(folder/'FIT_RESULT.json',audits[role]);models.append(booster)
        print(f'P4.2G {w} {role} fitted {len(rows)} rows',flush=True)
    del x,y,weight;gc.collect()
    return models,audits

def run(repo):
    from .p42g_evaluate import evaluate
    repo=Path(repo).resolve();branch_guard(repo);root=repo/'artifacts/phase4'/RUN_ID
    c=read_json(repo/'reports/phase4/P4_2G_EXPERIMENT_CONTRACT.json');froot=Path(c['reuse_root'])
    assert read_json(root/'PREFLIGHT.json')['status']=='pass'
    if (root/'FORMAL_START.json').exists():raise FileExistsError('formal attempt already exists; no silent repeat')
    for r in c['trusted_F_assets']:check_identity(r)
    for r in c['authority']:check_identity(r)
    write_json(root/'FORMAL_START.json',dict(at=now(),contract=identity(repo/'reports/phase4/P4_2G_EXPERIMENT_CONTRACT.json'),
        source=[identity(p) for p in sorted((repo/'src/hm_recsys').glob('p42g*.py'))]))
    start=time.perf_counter();state=dict(stage='P4.2G',status='running',windows={},training={},final_week='not_run',Warm_v2_integrated=False,full_history_run=False,P4_3_started=False)
    def guard():
        if time.perf_counter()-start>c['budget']['formal_max_seconds']:raise TimeoutError('formal budget reached')
        if rss()>6*2**30:raise MemoryError('peak process RAM>6GiB')
        if shutil.disk_usage(repo).free<15*2**30:raise RuntimeError('free disk<15GiB')
    try:
        for w,t in WINDOWS.items():
            guard();models,fit=train(root,froot,w,guard)
            data=joblib.load(froot/'prepared'/t/'data.joblib')
            state['training'][w]=fit;state['windows'][w]=evaluate(repo,data,models,root/'outer'/w,froot/'outer'/w,guard)
            write_json(repo/'reports/phase4/P4_2G_metrics.json',state)
            print(f'P4.2G completed {w}',flush=True);del models,data;gc.collect()
        state['verdicts']=verdict(state['windows'],read_json(repo/'reports/phase4/p4_2f_global_utility.json')['data'])
        state['status']='completed_pending_verification'
    except Exception:
        state['status']='engineering_failure';state['error']=traceback.format_exc()
        write_json(root/'FAILURE_FORMAL.json',dict(at=now(),error=state['error'],selected='W0'));raise
    finally:
        state.update(seconds=time.perf_counter()-start,peak_gib=rss()/2**30,finished_at=now())
        write_json(repo/'reports/phase4/P4_2G_metrics.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','preflight','run']);p.add_argument('--repo',default='.')
    a=p.parse_args();{'register':register,'preflight':preflight,'run':run}[a.command](a.repo)
