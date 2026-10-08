"""No-training dispatch tests; all receipts are synthetic temporary files."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from hm_recsys import p43a_run as run
from hm_recsys.p43a_contract import f_configs,model_configs
from hm_recsys.p42f_contract import WINDOWS
from hm_recsys.p41a_contract import read_json,write_json


def _write_f_receipt(root,config,window,delta):
    folder=Path(root)/'evaluation'/config['id']/window/'policy'
    folder.mkdir(parents=True,exist_ok=True)
    row=dict(policy_id=config['id'],delta_map=delta,overall_map=.02+delta,status='completed')
    write_json(folder/'EVALUATION.json',dict(status='completed',rows=[row]))
    return [row]


def _initial_state(root):
    state=run._ledger_state(Path(root),[],[])
    write_json(Path(root)/'TOURNAMENT_STATE.json',state)
    return state


def _completed_c(repo,root,deadline,source):
    state=read_json(Path(root)/'TOURNAMENT_STATE.json')
    state['configs']['C-oracle']=dict(config=dict(id='C-oracle',arm='C'),
        windows={w:dict(synthetic=True) for w in WINDOWS},policies={},status='completed')
    write_json(Path(root)/'TOURNAMENT_STATE.json',state)
    return state


def test_model_priority_is_A_B_E_D_C_and_f_is_separate_128_frozen_configs():
    ordered=run.model_order(model_configs())
    compact=[]
    for c in ordered:
        if not compact or c['arm']!=compact[-1]:compact.append(c['arm'])
    assert compact==['A','B','E','D','C']
    assert sum(c['arm']=='C' for c in ordered)==3
    assert len(f_configs())==128 and all(not c['trainable'] for c in f_configs())


def test_tournament_dispatches_c_jointly_once_after_edge_families_before_f():
    configs=[dict(id=arm+'-synthetic',arm=arm) for arm in ('D','B','E','A')]
    configs.extend(dict(id='C-'+role,arm='C',role=role) for role in ('count','cold','warm'))
    calls=[];policy=[dict(id='synthetic-policy')]
    def fitted(repo,root,window,config,deadline,source):
        assert config['arm']!='C'
        calls.append(config['arm'])
        return object(),dict(status='completed')
    def predicted(repo,root,froot,window,config,booster,deadline):
        return dict(config=config,window=window),np.zeros((0,12)),Path(root)/'evaluation'/config['id']/window
    def evaluated(data,score,policies_or_output,output=None,deadline_epoch=None):
        folder=Path(output if output is not None else policies_or_output)
        folder.mkdir(parents=True,exist_ok=True)
        rows=[dict(policy_id='synthetic-policy',delta_map=0.)]
        write_json(folder/'EVALUATION.json',dict(rows=rows,status='completed'))
        return rows
    def joint_c(*args):
        calls.append('C-joint');return _completed_c(*args)
    def f_after(repo,root,contract,deadline):
        calls.append('F');return read_json(Path(root)/'TOURNAMENT_STATE.json')
    with TemporaryDirectory() as tmp,ExitStack() as stack:
        root=Path(tmp)
        for name,value in [('guard',lambda *a,**k:None),('rss',lambda:0),('fit',fitted),('predict',predicted),('run_f',f_after)]:
            stack.enter_context(patch.object(run,name,side_effect=value))
        stack.enter_context(patch('hm_recsys.p43a_contract.model_configs',return_value=configs))
        stack.enter_context(patch('hm_recsys.p43a_policy.policies',return_value=policy))
        stack.enter_context(patch('hm_recsys.p43a_policy.evaluate_grid',side_effect=evaluated))
        stack.enter_context(patch('hm_recsys.p43a_policy.evaluate_residual',side_effect=evaluated))
        stack.enter_context(patch('hm_recsys.p43a_c.run_c',side_effect=joint_c))
        state=run.tournament(root,root,{},None,'synthetic-source')
        assert calls==['A']*4+['B']*4+['E']*4+['D']*4+['C-joint','F']
        assert state['status']=='tournament_completed'
        assert state['configs']['C-oracle']['status']=='completed'
        assert all(state['configs']['C-'+role]['status']=='not_run' for role in ('count','cold','warm'))


def _f_pruning_case(deltas,enabled=True):
    config=f_configs()[0];calls=[]
    with TemporaryDirectory() as tmp:
        root=Path(tmp);_initial_state(root)
        def evaluated(repo,out,window,cfg,deadline):
            calls.append(window)
            return _write_f_receipt(root,cfg,window,deltas[len(calls)-1])
        with patch.object(run,'guard'),patch('hm_recsys.p43a_special.evaluate_f_config',side_effect=evaluated):
            state=run.run_f(root,root,dict(F=dict(configs=[config]),pruning=dict(enabled=enabled)),None)
        return state['configs'][config['id']],calls


def test_f_pruning_uses_strict_first_two_window_condition():
    bad,calls=_f_pruning_case([-.002,-.002])
    assert len(calls)==2 and bad['status']=='all_policies_pruned'
    assert bad['policies'][bad['config']['id']]['status']=='pruned_bad_config'
    boundary,calls=_f_pruning_case([-.0015,-.0015,0.,0.])
    assert len(calls)==4 and boundary['status']=='completed'
    one_positive,calls=_f_pruning_case([.0001,-.004,0.,0.])
    assert len(calls)==4 and one_positive['status']=='completed'
    disabled,calls=_f_pruning_case([-.002,-.002,0.,0.],enabled=False)
    assert len(calls)==4 and disabled['status']=='completed'


def test_f_pruned_config_is_not_restarted_or_given_a_four_window_mean():
    config=f_configs()[0]
    with TemporaryDirectory() as tmp:
        root=Path(tmp);state=_initial_state(root)
        state['configs'][config['id']]=dict(config=config,windows={w:dict(synthetic=True) for w in list(WINDOWS)[:2]},
            policies={config['id']:dict(status='pruned_bad_config')},status='all_policies_pruned')
        write_json(root/'TOURNAMENT_STATE.json',state)
        with patch.object(run,'guard'),patch('hm_recsys.p43a_special.evaluate_f_config') as evaluate:
            result=run.run_f(root,root,dict(F=dict(configs=[config]),pruning=dict(enabled=True)),None)
        evaluate.assert_not_called()
        assert len(result['configs'][config['id']]['windows'])==2
        assert 'mean' not in result['configs'][config['id']]


def test_c_pause_does_not_enter_f_and_persists_joint_family_record():
    for reason in ('paused_resources','paused_time_budget'):
        with TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp)
            def paused_c(repo,out,deadline,source):
                state=_completed_c(repo,out,deadline,source)
                state['status']=reason;state['configs']['C-oracle']['status']=reason
                write_json(Path(out)/'TOURNAMENT_STATE.json',state)
                return state
            stack.enter_context(patch.object(run,'guard'))
            stack.enter_context(patch.object(run,'rss',return_value=0))
            stack.enter_context(patch('hm_recsys.p43a_contract.model_configs',return_value=[]))
            stack.enter_context(patch('hm_recsys.p43a_policy.policies',return_value=[]))
            stack.enter_context(patch('hm_recsys.p43a_c.run_c',side_effect=paused_c))
            f=stack.enter_context(patch.object(run,'run_f'))
            state=run.tournament(root,root,{},None,'synthetic-source')
            f.assert_not_called()
            saved=read_json(root/'TOURNAMENT_STATE.json')
            assert state['status']==saved['status']==reason
            assert saved['configs']['C-oracle']['status']==reason


def test_parent_pause_does_not_overwrite_f_completed_window_with_stale_state():
    for exception,status in [(run.BudgetPause,'paused_time_budget'),(MemoryError,'paused_resources')]:
        with TemporaryDirectory() as tmp,ExitStack() as stack:
            root=Path(tmp);config=f_configs()[0];first_done=[False];calls=[]
            def bounded_guard(*args,**kwargs):
                if first_done[0]:raise exception('synthetic stop after first persisted F window')
            def evaluated(repo,out,window,cfg,deadline):
                calls.append(window)
                rows=_write_f_receipt(root,cfg,window,.001)
                first_done[0]=True
                return rows
            stack.enter_context(patch.object(run,'guard',side_effect=bounded_guard))
            stack.enter_context(patch.object(run,'rss',return_value=0))
            stack.enter_context(patch('hm_recsys.p43a_contract.model_configs',return_value=[]))
            stack.enter_context(patch('hm_recsys.p43a_policy.policies',return_value=[]))
            stack.enter_context(patch('hm_recsys.p43a_c.run_c',side_effect=_completed_c))
            stack.enter_context(patch('hm_recsys.p43a_special.evaluate_f_config',side_effect=evaluated))
            contract=dict(F=dict(configs=[config]),pruning=dict(enabled=True))
            state=run.tournament(root,root,contract,None,'synthetic-source')
            saved=read_json(root/'TOURNAMENT_STATE.json')
            assert state['status']==saved['status']==status
            assert calls==[next(iter(WINDOWS))]
            assert list(saved['configs'][config['id']]['windows'])==calls
            assert saved['configs']['C-oracle']['status']=='completed'
            receipt=Path(saved['configs'][config['id']]['windows'][calls[0]]['receipt'])
            assert read_json(receipt)['rows'][0]['delta_map']==.001


def test_deadline_before_first_edge_fit_stops_later_families():
    configs=[dict(id='A-not-started',arm='A')]
    with TemporaryDirectory() as tmp,ExitStack() as stack:
        root=Path(tmp)
        stack.enter_context(patch('hm_recsys.p43a_contract.model_configs',return_value=configs))
        stack.enter_context(patch('hm_recsys.p43a_policy.policies',return_value=[dict(id='p')]))
        stack.enter_context(patch.object(run,'guard',side_effect=run.BudgetPause('deadline')))
        stack.enter_context(patch.object(run,'rss',return_value=0))
        fit=stack.enter_context(patch.object(run,'fit'))
        c=stack.enter_context(patch('hm_recsys.p43a_c.run_c'))
        f=stack.enter_context(patch.object(run,'run_f'))
        state=run.tournament(root,root,{},None,'synthetic-source')
        fit.assert_not_called();c.assert_not_called();f.assert_not_called()
        assert state['status']=='paused_time_budget'
        assert state['configs']['A-not-started']['status']=='not_run'


def load_tests(loader,tests,pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(function)
        for name,function in sorted(globals().items()) if name.startswith('test_'))


if __name__=='__main__':
    unittest.main()
