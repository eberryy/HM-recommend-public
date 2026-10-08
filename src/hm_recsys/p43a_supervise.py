"""Sequential one-window process isolation; never resets the session budget."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import subprocess
import sys
import time


def read(path):
    with Path(path).open(encoding='utf8') as stream:return json.load(stream)


def write(path,value):
    temporary=Path(str(path)+'.supervisor-part')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    temporary.replace(path)


def now():return datetime.now(timezone.utc).isoformat()


def finished_windows(state):
    return sum(len(record.get('windows',{})) for record in state['configs'].values())


def supervise(repo,session_path):
    repo,session_path=Path(repo).resolve(),Path(session_path).resolve()
    session=read(session_path)
    deadline=session['budget']['deadline_epoch']
    root=repo/'artifacts/phase4/p4-3a-v1-map-cashout-hash10-tournament'
    state_path=root/'TOURNAMENT_STATE.json'
    output=root/f"SUPERVISOR_{session['session_id']}_{time.time_ns()}.json"
    journal=dict(started_at=now(),session_path=str(session_path),deadline_epoch=deadline,
        workers=[],parallel_workers=1,max_new_windows_per_worker=1,
        model_or_policy_changes=False,final_week='not_run')
    try:
        while True:
            state=read(state_path)
            if state['status']!='yielded_window_boundary':
                journal['status']=state['status'];break
            if time.time()>=deadline:
                state.update(status='paused_time_budget',paused_at=now(),reason='Session deadline between isolated workers')
                write(state_path,state);journal['status']=state['status'];break
            before=finished_windows(state)
            command=[sys.executable,'-m','hm_recsys.p43a_run','run','--repo',str(repo),
                     '--session',str(session_path),'--max-new-windows','1']
            record=dict(started_at=now(),before_completed_windows=before,command=command)
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            process=subprocess.Popen(command,cwd=repo,creationflags=flags)
            record['pid']=process.pid;journal['workers'].append(record);write(output,journal)
            record['exit_code']=process.wait();record['finished_at']=now()
            state=read(state_path)
            record['after_completed_windows']=finished_windows(state)
            record['worker_status']=state['status']
            write(output,journal)
            if record['exit_code']!=0:
                journal['status']='engineering_failure';break
            if state['status']=='yielded_window_boundary' and finished_windows(state)<=before:
                journal['status']='engineering_failure_no_progress';break
            if state['status']!='yielded_window_boundary':
                journal['status']=state['status'];break
    finally:
        journal['finished_at']=now();write(output,journal)
    print('P4.3A supervisor stopped: '+journal['status'],flush=True)
    return journal


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',default='.')
    parser.add_argument('--session',required=True)
    args=parser.parse_args()
    supervise(args.repo,args.session)
