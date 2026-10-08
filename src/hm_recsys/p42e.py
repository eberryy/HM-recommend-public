"""P4.2E staged entry point. Never calls matching or recommendation scoring."""
from __future__ import annotations

import shutil
import sys
import time
import traceback
from pathlib import Path

from .p42e_resources import rss, memory

from .p41a_contract import read_json, write_json, check_identity, identity
from .p41a_data import identities
from .p42_contract import branch_guard
from .p42e_contract import RUN_ID, preregister, now


def context(repo):
    repo=Path(repo).resolve(); branch_guard(repo)
    c=read_json(repo/'reports/phase4/P4_2E_EXPERIMENT_CONTRACT.json')
    assert c['status']=='preregistered_before_formal_computation' and c['run_id']==RUN_ID
    return c,repo/'artifacts/phase4'/RUN_ID


def generate_all(repo,resume_schema=False,resume_optional=False):
    from .p42e_data import generate
    c,root=context(repo)
    # This is a prerequisite, not a claim that 10 GiB guarantees the largest fit.
    if memory().available<10*2**30:
        raise RuntimeError('preflight insufficient memory: require >=10GiB available before generation; no sample reduction')
    if root.exists() and not (resume_schema or resume_optional): raise FileExistsError('no overwrite: '+str(root))
    if resume_optional:
        failure=read_json(root/'FAILURE_schema_repair.json')
        if 'generation-users.parquet' not in failure.get('failure','') or (root/'OPTIONAL_REPAIR_START.json').exists():
            raise ValueError('resume only once for absent optional restoration assets')
        check_identity(read_json(root/'EXECUTION_START.json')['contract'])
        write_json(root/'OPTIONAL_REPAIR_START.json',dict(at=now(),original_failure=identity(root/'FAILURE_schema_repair.json'),
            change='only read restoration files when original source does not cover all H users; reuse completed cutoff'))
    elif resume_schema:
        failure=read_json(root/'FAILURE.json')
        if failure.get('failure')!="'b0_rank'" or (root/'SCHEMA_REPAIR_START.json').exists():
            raise ValueError('resume only once for missing-reference-column engineering bug')
        check_identity(read_json(root/'EXECUTION_START.json')['contract'])
        write_json(root/'SCHEMA_REPAIR_START.json',dict(at=now(),original_failure=identity(root/'FAILURE.json'),
            change='use old unlabelled Cold50 selection rather than model feature table; preserve first generated chunk',
            earlier_target_column_loaded_but_not_used=True,
            boundary_note='attempt01 loaded old historical target column into memory; not used to generate/select; new path projects no labels'))
    else:
        root.mkdir(parents=True)
        write_json(root/'EXECUTION_START.json',dict(at=now(),contract=identity(repo/'reports/phase4/P4_2E_EXPERIMENT_CONTRACT.json')))
    started=time.perf_counter()
    out=dict(stage='P4.2E',status='running_generation',population={},windows={},final_week='not_run',
             qC_fits=0,qW_fits=0,matching_calls=0,new_recommendations=0,new_MAP=0,Warm_v2_integrated=False)
    peak=0
    def guard():
        nonlocal peak
        peak=max(peak,rss())
        if shutil.disk_usage(repo).free<15*2**30: raise RuntimeError('disk guard <15GiB')
        if time.perf_counter()-started>7200: raise RuntimeError('2h generation budget exceeded')
    try:
        # Trusted older manifest comparisons address cross-task reuse integrity.
        trusted={x['path']:x for x in identities(read_json(repo/'reports/phase4/P4_2R3_OUTPUT_MANIFEST.json'))}
        trusted.update({x['path']:x for x in identities(read_json(repo/'reports/phase4/P4_2R_OUTPUT_MANIFEST.json'))})
        for x in trusted.values(): check_identity(x)
        for x in c['authority'].values(): check_identity(x)
        write_json(root/'INPUT_VERIFICATION.json',dict(trusted_records=list(trusted.values()),comparisons=len(trusted),passed=True))
        for t in c['historical_cutoffs']:
            complete=root/'prepared'/t/'COMPLETE.json'
            out['population'][t]=read_json(complete) if resume_optional and complete.exists() else generate(repo,c,t,root,guard)
            write_json(root/'PROGRESS.json',out)
        out['status']='generation_complete_training_not_run'
    except Exception as exc:
        out.update(status='engineering_failure',failure=str(exc),traceback=traceback.format_exc())
        write_json(root/('FAILURE_optional_repair.json' if resume_optional else 'FAILURE_schema_repair.json' if resume_schema else 'FAILURE.json'),dict(at=now(),**out))
        raise
    finally:
        out['generation_seconds']=time.perf_counter()-started; out['sampled_peak_rss_bytes']=peak
        write_json(root/'PROGRESS.json',out)
        write_json(repo/'reports/phase4/p4_2e_full_history_population.json',out['population'])
        write_json(repo/'reports/phase4/P4_2E_metrics.json',out)


if __name__=='__main__':
    repo=Path.cwd()
    if sys.argv[1]=='preregister':
        c=preregister(repo); print(c['budget'])
    elif sys.argv[1]=='generate': generate_all(repo)
    elif sys.argv[1]=='resume-schema-repair': generate_all(repo,True)
    elif sys.argv[1]=='resume-optional-repair': generate_all(repo,resume_optional=True)
    elif sys.argv[1]=='train':
        from .p42e_train import train_all
        train_all(repo)
    else: raise ValueError('choose preregister/generate/train')
