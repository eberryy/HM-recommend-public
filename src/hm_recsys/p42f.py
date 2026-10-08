"""P4.2F staged CLI. Never silently restart completed/failed computation."""
import argparse
import traceback
import time
import shutil
import gc
import joblib
from pathlib import Path
from .p41a_contract import read_json, write_json, identity
from .p42_contract import branch_guard
from .p42f_contract import RUN_ID, register, now

def run(repo):
    from .p42f_train import prepare,train_outer
    from .p42f_evaluate import evaluate,decision
    from .p42f_contract import WINDOWS
    from .p42e_resources import rss
    repo=Path(repo).resolve(); branch_guard(repo); root=repo/'artifacts/phase4'/RUN_ID
    c=read_json(repo/'reports/phase4/P4_2F_EXPERIMENT_CONTRACT.json')
    assert read_json(root/'NOVELTY_PREPARATION.json')['status']=='completed'
    if (root/'FORMAL_START.json').exists(): raise FileExistsError('formal attempt exists')
    write_json(root/'FORMAL_START.json',dict(at=now(),contract=identity(repo/'reports/phase4/P4_2F_EXPERIMENT_CONTRACT.json'),
        source=[identity(p) for p in sorted((repo/'src/hm_recsys').glob('p42f*.py'))]))
    started=time.perf_counter(); state=dict(stage='P4.2F',run_id=RUN_ID,status='running',windows={},final_week='not_run',Warm_v2_integrated=False,P4_3_started=False)
    report=repo/'reports/phase4/P4_2F_metrics.json'
    def guard():
        if time.perf_counter()-started>c['budget']['formal_seconds']: raise TimeoutError('formal compute budget exhausted')
        if shutil.disk_usage(repo).free<15*2**30: raise RuntimeError('free disk <15GiB')
        if rss()>6*2**30: raise MemoryError('process RAM budget exceeded')
    try:
        state['training_audit']=prepare(repo,c,root,guard); write_json(report,state)
        for w,t in WINDOWS.items():
            guard(); print('P4.2F fit '+w,flush=True)
            G,S,threshold,history=train_outer(repo,c,root,w,guard)
            data=joblib.load(root/'prepared'/t/'data.joblib')
            state['windows'][w]=evaluate(data,G,S,threshold,history,root/'outer'/w,guard)
            write_json(report,state); print({w:{v:r['map12'] for v,r in state['windows'][w]['variants'].items()}},flush=True)
            del G,S,threshold,history,data;gc.collect()
        state['decision']=decision(state['windows']); state['status']='completed_pending_verification'
    except Exception:
        state['status']='engineering_failure';state['failure']=traceback.format_exc();state['selected']='W0'
        write_json(root/'FAILURE_FORMAL.json',dict(at=now(),traceback=state['failure']))
        raise
    finally:
        state['seconds']=time.perf_counter()-started;state['at']=now();write_json(report,state)

def preflight(repo, recover_input_mapping=False):
    from .p42f_data import verify_inputs, purchase_state
    repo=Path(repo).resolve(); branch_guard(repo)
    c=read_json(repo/'reports/phase4/P4_2F_EXPERIMENT_CONTRACT.json')
    root=repo/'artifacts/phase4'/RUN_ID
    if root.exists():
        if not recover_input_mapping or not (root/'FAILURE_PREFLIGHT.json').exists() or (root/'novelty.duckdb').exists():
            raise FileExistsError('only pre-data input mapping recovery allowed')
        failures=sorted(root.glob('FAILURE*.json'))
        failure=read_json(failures[-1])
        assert 'no trusted identity' in failure['traceback'] or 'Unable to find a usable engine' in failure['traceback']
        attempt=len(failures)
        recovery=root/f'INTERFACE_RECOVERY_{attempt}.json'
        if recovery.exists(): raise FileExistsError('recovery already attempted')
        write_json(recovery,dict(at=now(),reason='pre-data asset/parquet interface only; use DuckDB already installed; no parameter or population change'))
    else: root.mkdir(parents=True)
    startfile=root/(f'RECOVERY_START_{attempt}.json' if recover_input_mapping else 'EXECUTION_START.json')
    write_json(startfile,dict(at=now(),contract=identity(repo/'reports/phase4/P4_2F_EXPERIMENT_CONTRACT.json'),
        source=[identity(p) for p in sorted((repo/'src/hm_recsys').glob('p42f*.py'))]))
    try:
        verified=verify_inputs(repo,c)
        write_json(root/'TRUSTED_INPUT_CHECK.json',dict(status='pass',files=verified))
        print(f'trusted frozen input comparisons passed: {len(verified)}',flush=True)
        result=purchase_state(repo,c,root)
        print(result,flush=True)
    except Exception:
        write_json(root/(f'FAILURE_RECOVERY_{attempt}.json' if recover_input_mapping else 'FAILURE_PREFLIGHT.json'),dict(at=now(),traceback=traceback.format_exc(),action_labels_materialized=False,
            model_fits=0,matching_calls=0,final_week='not_run',selected='W0'))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('command',choices=['register','preflight','run']); p.add_argument('--repo',default='.'); p.add_argument('--recover-input-mapping',action='store_true')
    a=p.parse_args()
    if a.command=='register': register(a.repo)
    elif a.command=='preflight': preflight(a.repo,a.recover_input_mapping)
    else: run(a.repo)
