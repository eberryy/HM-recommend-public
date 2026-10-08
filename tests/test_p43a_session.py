import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from hm_recsys.p43a_session import numeric_gate, pause_on_gate, execution_budget
from hm_recsys.p42f_contract import WINDOWS
from hm_recsys.p41a_contract import write_json, read_json


def evidence(deltas):
    return {w:[dict(policy_id='p',status='completed',delta_map=d)] for w,d in zip(WINDOWS,deltas)}


class SessionTests(unittest.TestCase):
    def test_mean_gate_is_strict(self):
        self.assertIsNone(numeric_gate(evidence([.0001]*4)))
        self.assertIsNotNone(numeric_gate(evidence([.000101]*4)))

    def test_worst_is_inclusive_and_nondegrade_is_required(self):
        self.assertIsNotNone(numeric_gate(evidence([-.001,0,.001,.001])))
        self.assertIsNone(numeric_gate(evidence([-.00101,0,.001,.001])))
        self.assertIsNone(numeric_gate(evidence([-.00001,-.00001,-.00001,.001])))

    def test_incomplete_or_pruned_not_eligible(self):
        data=evidence([.01]*3)
        self.assertIsNone(numeric_gate(data))
        data=evidence([.01]*4)
        data[next(iter(WINDOWS))][0]['status']='pruned_bad_config'
        self.assertIsNone(numeric_gate(data))

    def test_execution_budget_is_additive_authorization_not_contract_edit(self):
        c=dict(budget=dict(total_seconds=7200,deadline_epoch=10))
        state=dict(active_session=dict(budget=dict(total_seconds=21600,deadline_epoch=50)))
        self.assertEqual(execution_budget(c,state)['deadline_epoch'],50)
        self.assertEqual(c['budget']['deadline_epoch'],10)
        self.assertEqual(execution_budget(c,{}),c['budget'])

    def test_gate_pause_saves_completed_evidence_but_no_fullscale(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            record=dict(config=dict(id='A-new',arm='A'),windows={})
            for w,rows in evidence([.0002]*4).items():
                p=root/(w+'.json');write_json(p,dict(rows=rows))
                record['windows'][w]=dict(receipt=str(p))
            state=dict(active_session=dict(session_id='s02',pause_on_candidate_numeric_gate=True),configs={'A-new':record})
            self.assertTrue(pause_on_gate(root,state,record))
            self.assertEqual(state['status'],'paused_candidate_numeric_gate')
            self.assertEqual(len(state['configs']['A-new']['windows']),4)
            self.assertFalse(read_json(root/'CANDIDATE_GATE_PAUSE.json')['fullscale_started'])
            record['config']['arm']='F'
            self.assertFalse(pause_on_gate(root,state,record))


if __name__=='__main__':unittest.main()
