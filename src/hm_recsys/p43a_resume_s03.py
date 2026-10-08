"""Register the user-requested remaining tournament; never enlarge the search."""
from pathlib import Path
from datetime import datetime, timezone
import shutil
from .p41a_contract import read_json, write_json, check_identity
from .p43a_run import guard

START=1789223744  # 2026-09-12T14:35:44 UTC, includes preparation
SESSION='s03-20260912T143544'


def register(repo):
    repo=Path(repo).resolve(); root=repo/'artifacts/phase4/p4-3a-v1-map-cashout-hash10-tournament'
    target=repo/'reports/phase4/P4_3A_SESSION_03.json'
    if target.exists(): raise FileExistsError('Do not overwrite session authorization')
    guard(repo,START+21600)
    state=read_json(root/'TOURNAMENT_STATE.json')
    if state['status']!='paused_time_budget': raise ValueError('Expected stopped session02')
    expected=read_json(root/'EXECUTION_START.json')['contract']; check_identity(expected)
    source=root/'models/D-L63-D-1-M200/spring_20200318'
    paused=read_json(source/'FIT_PAUSED.json')
    fit=read_json(source/'FIT_START.json')
    contract=read_json(Path(expected['path']))
    config=next(c for c in contract['model_configs'] if c['id']=='D-L63-D-1-M200')
    if (paused['completed_iterations']!=406 or paused['planned_iterations']!=500
            or (source/'FIT_RESULT.json').exists() or fit['config']!=config):
        raise ValueError('Partial fit differs from inspected frozen 406/500 target')
    archive=root/'session-history'/SESSION
    archive.mkdir(parents=True,exist_ok=False)
    shutil.copy2(root/'TOURNAMENT_STATE.json',archive/'TOURNAMENT_STATE.before.json')
    (archive/'reports').mkdir()
    for p in (repo/'reports/phase4').iterdir():
        if p.is_file() and p.name.lower().startswith('p4_3a'):
            shutil.copy2(p,archive/'reports'/p.name)
    destination=archive/'partial-fit-D-L63-D-1-M200-spring'
    if not source.resolve().is_relative_to(root.resolve()) or not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError('Recovery path escaped this experiment')
    value=dict(stage='P4.3A',session_id=SESSION,status='approved_before_resume',
        registered_at=datetime.now(timezone.utc).isoformat(),user_authorization='继续完成剩下的吧',
        original_contract=expected,search_space_changed=False,execution_priority_changed=False,
        budget=dict(start_epoch=START,total_seconds=21600,deadline_epoch=START+21600,
            start_utc=datetime.fromtimestamp(START,timezone.utc).isoformat(),
            deadline_utc=datetime.fromtimestamp(START+21600,timezone.utc).isoformat(),includes_preparation=True,
            note='Agent safety cap retained from prior session; estimate 2-4 hours, not a user-requested new six-hour duration'),
        pause_on_candidate_numeric_gate=False,
        stop_semantics='Complete remaining frozen tournament; no early winner selection. Stop on completion, safety budget or resource/engineering blocker; no automatic full-scale execution.',
        recovery=dict(source=str(source),destination=str(destination),completed_iterations=406,
            planned_iterations=500,method='preserve_partial_restart_only_incomplete_window_from_zero',
            unchanged_data_parameters_seed=True),
        previous_reports_archive=str(archive/'reports'),fullscale_execution_authorized=False,
        final_week='not_run',Warm_v2_integrated=False)
    write_json(target,value)
    # Exact resolved experiment-local target, preserve all original partial files.
    if destination.exists(): raise FileExistsError(destination)
    shutil.move(str(source),str(destination))
    if not (destination/'FIT_PAUSED.json').exists() or source.exists():
        raise RuntimeError('Partial-fit archival did not complete')
    return value


if __name__=='__main__':
    print(register(Path('.')))
