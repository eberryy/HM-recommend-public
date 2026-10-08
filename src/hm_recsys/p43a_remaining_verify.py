"""Read-only C/E/F saved-list and all-model temporal-contract audit.

Rebuilds AP using pre-existing metrics.apk, not production policy evaluation.
This verifies all saved C/E/F decisions but is not an independent retraining or
a proof of each saved assignment's optimality. Never generates recommendations.
"""
from pathlib import Path
import argparse
import gc
import time
import traceback
import csv
import joblib
import numpy as np
from .p41a_contract import read_json, write_json
from .p43a_verify import WINDOWS, SEGMENTS, independent_context, reconstruct, close, population, stamp
from .metrics import apk


def verify_pairs(warm, articles, owners, ui, pairs):
    pairs=np.asarray(pairs)
    active=pairs[pairs[:,0]>=0]
    if len(active):
        assert np.all((active[:,0]>=0)&(active[:,0]<len(articles)))
        assert np.all((active[:,1]>=0)&(active[:,1]<12))
        assert np.all(owners[active[:,0]]==ui)
    return reconstruct(warm,articles,active[:,0]*12+active[:,1]),active


def model_contracts(root, contract):
    configs={c['id']:c for c in contract['model_configs']}
    checked=[]
    for path in sorted((root/'models').glob('*/*/FIT_RESULT.json')):
        fit=read_json(path); config=configs[path.parent.parent.name]
        assert fit['status']=='completed' and fit['config']==config and fit['params']==config['params']
        w=path.parent.name; cutoff=WINDOWS[w]
        from datetime import date,timedelta
        assert fit['cutoff']==cutoff
        assert all((date.fromisoformat(t)+timedelta(days=7)).isoformat()<cutoff for t in fit['training_cutoffs'])
        assert (path.parent/'model.txt').exists()
        checked.append(dict(model=config['id'],window=w,training_cutoffs=fit['training_cutoffs']))
    for p in sorted((root/'prepared').glob('*/meta/AUDIT.json')):
        a=read_json(p); t=a['cutoff']; assert t<'2020-09-16'
        assert a['new_model_fits']==0
        for line in a['lineage'].values():
            assert not line['availability'] or line['training_label_end']<t
    return checked


def ledger_gate(repo):
    """Independent scalar arithmetic on every ledger row; no new selection search."""
    seen=set(); qualified=set(); counts={}; maximum=-np.inf
    with (repo/'reports/phase4/p4_3a_tournament_ledger.csv').open(encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            key=(row['model_config_id'],row['policy_id']); assert key not in seen;seen.add(key)
            status=row['status'];counts[status]=counts.get(status,0)+1
            assert status in ('completed','pruned_bad_config')
            if status=='pruned_bad_config':
                d=[float(row[w+'_delta']) for w in list(WINDOWS)[:2]]
                assert all(x<0 for x in d) and sum(d)/2<-.0015
                assert not row['mean_delta'] and not row['mean_map']
                continue
            d=[float(row[w+'_delta']) for w in WINDOWS]
            maps=[float(row[w+'_map']) for w in WINDOWS]
            close(sum(d)/4,float(row['mean_delta']),atol=1e-15)
            close(sum(maps)/4,float(row['mean_map']),atol=1e-15)
            assert min(d)==float(row['worst_delta']) and sum(x>=0 for x in d)==int(row['nondegrade_windows'])
            maximum=max(maximum,sum(maps)/4)
            if row['arm']!='F' and sum(d)/4>.0001 and sum(x>=0 for x in d)>=2 and min(d)>=-.001:
                qualified.add(row['arm'])
    assert len(seen)==107690
    choice=read_json(repo/'reports/phase4/p4_3a_top2_selection.json')
    assert choice['tournament_complete'] and set(choice['qualifying_trainable_families'])==qualified
    assert choice['full_scale_allowed']==bool(qualified)
    assert choice['top2_full_scale_allowed']==(len(qualified)>=2)
    assert choice['top1_full_scale_allowed']==(len(qualified)==1)
    top=read_json(repo/'reports/phase4/p4_3a_top20.json')['rows'][0]
    assert maximum-top['mean_map']<=1e-8+1e-15
    assert len({x['arm'] for x in choice['provisional_top2']})==len(choice['provisional_top2'])
    return dict(status='pass',rows=len(seen),counts=counts,qualified_trainable_families=sorted(qualified),
        complete_window_means_pruning_and_scaleup_gates=True,maximum_mean_map=maximum,
        limitation='Checks ledger arithmetic and declared gate sets; does not duplicate every historical model-policy score computation')


def audit_window(root, contract, state, window, data, guard):
    population(data,WINDOWS[window]); truths,valid,base=independent_context(data)
    n=len(data['users']); articles=data['cold'].article_id.to_numpy()
    owners=data['cold'].user_index.to_numpy(int)
    entries=[]; loaded={}; c_khat=None
    for model, record in state['configs'].items():
        arm=record['config']['arm']
        if arm not in ('C','E','F') or window not in record.get('windows',{}): continue
        saved=record['windows'][window]
        receipt=read_json(Path(saved['receipt'])); assert receipt['status']=='completed'
        assert receipt['final_week']=='not_run'
        for row in receipt['rows']:
            assert row['status']=='completed'
            if arm=='F':
                original=next(c for c in contract['F']['configs'] if c['id']==model)
                assert row['policy']==original and row['all_matches_fixed_before_truth_access']
            elif arm=='C':
                assert row['policy'] in contract['C']['blends']
            else:
                p=row['policy']; canonical=f"E-K{p['candidate_topk']}-S{p['slot_floor']}"
                registered=next(p for p in contract['E']['policies'] if p['id']==canonical)
                assert registered['candidate_topK']==p['candidate_topk']
                assert registered['replaceable_slot_floor']==p['slot_floor']
            path=row['saved_decisions']
            if path not in loaded: loaded[path]=np.load(path,mmap_mode='r')
            arr=loaded[path]
            choices=arr if arm=='F' else arr[row['decision_policy_index']]
            assert choices.shape[0]==n and choices.shape[-1]==2
            entries.append((arm,model,row,choices))
        if arm=='C':
            folder=Path(saved['receipt']).parent
            c_khat=np.load(folder/'Khat.npy')
            probabilities=np.load(folder.parent/'count-score.npy',mmap_mode='r')
            sizes=np.bincount(owners,minlength=n)
            np.testing.assert_array_equal(c_khat,np.minimum(np.argmax(probabilities,axis=1),sizes))
            close(probabilities.sum(axis=1),np.ones(n),atol=1e-10)
    sums=np.zeros((len(entries),5)); counts=np.zeros((len(entries),4),np.int64)
    unique_lists=0
    for ui in range(n):
        if ui%128==0: guard()
        cache={}
        for j,(arm,model,row,choices) in enumerate(entries):
            pp=choices[ui]; active=pp[pp[:,0]>=0]
            if arm=='C': assert len(active)==c_khat[ui]
            if arm=='F':
                assert len(active)<=row['policy']['max_admissions']
                if len(active):
                    assert np.all(data['cold'].b0_rank.to_numpy()[active[:,0]]<=row['policy']['candidate_topK'])
                    assert np.all(active[:,1]+1>=row['policy']['replaceable_slot_floor'])
            if arm=='E' and len(active): assert np.all(active[:,1]+1>=row['policy']['slot_floor'])
            key=tuple(map(tuple,active))
            if key not in cache:
                pred,active=verify_pairs(data['warm_lists'][ui],articles,owners,ui,pp)
                values=np.array([apk(truth,pred,12) for truth in truths[ui]])
                ins=sum(articles[ci] in truths[ui][0] for ci,slot in active)
                rem=sum(data['warm_lists'][ui][slot] in truths[ui][0] for ci,slot in active)
                cache[key]=(values-base[ui],np.array([ins,rem,bool(len(active)),len(active)]))
            delta,cc=cache[key]; sums[j]+=delta; counts[j]+=cc
        unique_lists+=len(cache)
    den=valid.sum(axis=0); baseline=base.sum(axis=0)/den
    delta=np.divide(sums,den,out=np.zeros_like(sums),where=den!=0)
    for j,(_,model,row,_) in enumerate(entries):
        close(baseline[0]+delta[j,0],row['overall_map'])
        close(delta[j,0],row['delta_map'])
        for col,name in enumerate(SEGMENTS[1:],1):
            assert row['segments'][name]['truth_users']==den[col]
            if den[col]:
                close(delta[j,col],row['segments'][name]['delta_vs_w0'])
                close(baseline[col]+delta[j,col],row['segments'][name]['map12'])
        for col,name in enumerate(('inserted_positives','removed_positives','admitted_users','replacements')):
            assert counts[j,col]==row[name],(window,model,row['policy_id'],name)
        close(counts[j,2]/n,row['coverage'])
    result=dict(window=window,status='pass',policy_rows=len(entries),user_policy_observations=n*len(entries),
        unique_lists_rebuilt=unique_lists,all_saved_lists_12_unique=True,all_owners_and_slot_constraints=True,
        all_C_predicted_K_replayed=c_khat is not None,AP_implementation='pre-existing metrics.apk',
        scope=[dict(arm=a,model=m,policy=r['policy_id']) for a,m,r,_ in entries])
    del loaded,entries;gc.collect()
    return result


def run(repo,deadline):
    repo=Path(repo).resolve();root=repo/'artifacts/phase4/p4-3a-v1-map-cashout-hash10-tournament'
    state=read_json(root/'TOURNAMENT_STATE.json')
    if state['status']!='tournament_completed': raise ValueError('Wait for settled complete tournament')
    contract=read_json(repo/'reports/phase4/P4_3A_EXPERIMENT_CONTRACT.json')
    dest=root/'verification-attempts'/('remaining-'+str(time.time_ns()));dest.mkdir(parents=True)
    start=time.perf_counter();out=dict(stage='P4.3A',status='running',at=stamp(),windows=[],
        new_fits=0,new_policy_evaluations=0,metric_overwrites=0,final_week='not_run',
        scope_limit='All C/E/F saved-list AP and feasibility; all fitted model contract/temporal receipts. Does not independently retrain models or prove every matching optimal. A/B/D use separate snapshot audit.')
    def guard():
        if time.time()>=deadline: raise TimeoutError('Read-only audit budget reached')
    try:
        assert state['final_week']=='not_run' and not state['full_history_run']
        out['ledger_gate']=ledger_gate(repo)
        out['model_contract_checks']=model_contracts(root,contract)
        for w,t in WINDOWS.items():
            guard();data=joblib.load(Path(contract['reuse_root'])/'prepared'/t/'data.joblib')
            out['windows'].append(audit_window(root,contract,state,w,data,guard))
            write_json(dest/'PROGRESS.json',out)
            print('Remaining independent list audit passed '+w,flush=True)
            del data;gc.collect()
        out['status']='pass_declared_scope'
    except Exception:
        out['status']='fail';out['error']=traceback.format_exc();raise
    finally:
        out['seconds']=time.perf_counter()-start;out['finished_at']=stamp()
        write_json(dest/'RESULT.json',out)
        write_json(repo/'reports/phase4/P4_3A_REMAINING_VERIFICATION.json',out)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--deadline',type=float,required=True);args=p.parse_args()
    print(run('.',args.deadline)['status'])
