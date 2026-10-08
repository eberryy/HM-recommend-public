"""Append-only execution budget amendments; the search contract stays fixed."""
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from .p41a_contract import read_json, write_json, identity, check_identity
from .p42f_contract import WINDOWS, now

SESSION_ID = 's02-20260912T081629'
START = 1789200989
SECONDS = 6 * 3600


def register_s02(repo, root):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    target = repo/'reports/phase4/P4_3A_SESSION_02.json'
    if target.exists():
        raise FileExistsError('Do not overwrite an execution authorization')
    expected = read_json(root/'EXECUTION_START.json')['contract']
    check_identity(expected)
    previous = read_json(root/'TOURNAMENT_STATE.json')
    if previous['status'] != 'paused_time_budget':
        raise ValueError('Expected the verified first-session budget pause')
    archive = root/'session-history'/SESSION_ID
    if not (archive/'TOURNAMENT_STATE.before.json').exists():
        raise ValueError('Prior session snapshot is required before continuation')
    source = root/'models/A2-L15-D6-M50/late_summer_20200819'
    paused = read_json(source/'FIT_PAUSED.json')
    started = read_json(source/'FIT_START.json')
    if paused['completed_iterations'] != 50 or paused['planned_iterations'] != 300 or (source/'FIT_RESULT.json').exists():
        raise ValueError('Recovery target differs from the inspected 50/300 partial fit')
    contract = read_json(Path(expected['path']))
    config = next(c for c in contract['model_configs'] if c['id'] == 'A2-L15-D6-M50')
    if started['config'] != config:
        raise ValueError('Partial fit is not the frozen config')
    value = dict(stage='P4.3A', session_id=SESSION_ID, status='approved_before_resume', registered_at=now(),
        user_authorization='按原计划继续，这次运行6个小时暂停汇报，或者达到扩量门槛',
        original_contract=expected, search_space_changed=False, execution_priority_changed=False,
        budget=dict(start_epoch=START, total_seconds=SECONDS, deadline_epoch=START+SECONDS,
            start_utc=datetime.fromtimestamp(START,timezone.utc).isoformat(),
            deadline_utc=datetime.fromtimestamp(START+SECONDS,timezone.utc).isoformat(),
            includes_preparation=True),
        pause_on_candidate_numeric_gate=True,
        candidate_gate=dict(mean_delta_strictly_above=.0001, minimum_nondegrade_windows=2,
            worst_delta_at_least=-.001, complete_development_windows=4, trainable_only=True),
        stop_semantics='Pause on first completed trainable model with a numerically qualifying four-window policy, or session deadline. This operational stop is NOT complete-tournament Top2 selection or permission to execute full scale.',
        recovery=dict(method='archive_partial_fit_then_restart_only_incomplete_window_from_zero',
            source=str(source), destination=str(archive/'partial-fit-A2-L15-D6-M50-late'),
            expected_completed_iterations=50, planned_iterations=300, estimated_seconds=360,
            completed_fits_reused=47, unchanged_data_parameters_seed=True),
        previous_state=identity(archive/'TOURNAMENT_STATE.before.json'),
        previous_reports_archive=str(archive/'reports'),
        fullscale_execution_authorized=False, final_week='not_run', Warm_v2_integrated=False)
    write_json(target,value)
    return value


def load_session(path, root):
    value = read_json(Path(path))
    if value['status'] != 'approved_before_resume' or value['final_week'] != 'not_run':
        raise ValueError('Invalid execution authorization')
    if value['search_space_changed'] or value['execution_priority_changed'] or value['fullscale_execution_authorized']:
        raise ValueError('This continuation cannot mutate search or execute full scale')
    budget=value['budget']
    if budget['deadline_epoch']-budget['start_epoch'] != budget['total_seconds']:
        raise ValueError('Budget cannot reset at process start')
    expected=read_json(Path(root)/'EXECUTION_START.json')['contract']
    if value['original_contract'] != expected:
        raise ValueError('Execution amendment belongs to another search contract')
    check_identity(expected)
    return dict(value, authorization_path=str(Path(path).resolve()))


def numeric_gate(rows_by_window):
    """Four complete windows only; no extrapolated or pruned mean."""
    if set(rows_by_window) != set(WINDOWS):
        return None
    maps={w:{r['policy_id']:r for r in rows if r.get('status')=='completed'}
          for w,rows in rows_by_window.items()}
    common=set.intersection(*(set(x) for x in maps.values()))
    qualified=[]
    for pid in sorted(common):
        ds=np.array([maps[w][pid]['delta_map'] for w in WINDOWS],float)
        if np.isfinite(ds).all() and ds.mean()>.0001 and (ds>=0).sum()>=2 and ds.min()>=-.001:
            qualified.append(dict(policy_id=pid, mean_delta=float(ds.mean()),
                window_deltas={w:float(d) for w,d in zip(WINDOWS,ds)},
                nondegrade_windows=int((ds>=0).sum()), worst_delta=float(ds.min())))
    if not qualified:
        return None
    qualified.sort(key=lambda x:(-x['mean_delta'],x['policy_id']))
    return dict(qualified_policy_count=len(qualified), example=qualified[0],
                not_final_tournament_winner=True)


def pause_on_gate(root,state,record):
    session=state.get('active_session',{})
    if not session.get('pause_on_candidate_numeric_gate') or record.get('config',{}).get('arm')=='F':
        return False
    if set(record.get('windows',{}))!=set(WINDOWS):
        return False
    if record.get('numeric_gate_checked_session') == session.get('session_id'):
        return False
    observed={w:read_json(Path(v['receipt']))['rows'] for w,v in record['windows'].items()}
    result=numeric_gate(observed)
    record['numeric_gate_checked_session']=session['session_id']
    if result is None:
        return False
    state.update(status='paused_candidate_numeric_gate', paused_at=now(),
        reason='User requested pause on candidate numerical scale-up gate; full tournament may be incomplete',
        candidate_numeric_gate=dict(result,model_id=record['config']['id'],arm=record['config']['arm']))
    write_json(Path(root)/'CANDIDATE_GATE_PAUSE.json',dict(stage='P4.3A',at=now(),
        session_id=session['session_id'],result=state['candidate_numeric_gate'],
        fullscale_started=False, final_week='not_run'))
    write_json(Path(root)/'TOURNAMENT_STATE.json',state)
    return True


def execution_budget(contract,state):
    return state.get('active_session',{}).get('budget',contract['budget'])
