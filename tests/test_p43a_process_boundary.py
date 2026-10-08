import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from hm_recsys.p43a_resources import snapshot
from hm_recsys.p43a_run import tournament
from hm_recsys.p41a_contract import read_json


class ProcessBoundaryTests(unittest.TestCase):
    def test_current_counter_not_confused_with_peak(self):
        r=snapshot()
        self.assertGreater(r['pid'],0)
        self.assertGreaterEqual(r['peak_working_gib'],r['current_working_gib'])
        self.assertGreaterEqual(r['private_commit_gib'],0)

    def test_one_window_yields_without_fitting_next_window(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            config=dict(id='A-probe',arm='A')
            with patch('hm_recsys.p43a_contract.model_configs',return_value=[config]), \
                 patch('hm_recsys.p43a_policy.policies',return_value=[dict(id='p')]), \
                 patch('hm_recsys.p43a_run.guard'), \
                 patch('hm_recsys.p43a_run.fit',return_value=(object(),dict(seconds=1))) as fitting, \
                 patch('hm_recsys.p43a_run.predict',return_value=({},None,root)), \
                 patch('hm_recsys.p43a_policy.evaluate_grid',return_value=[dict(policy_id='p',delta_map=0)]):
                result=tournament(root,root,{},9999999999,'synthetic',max_new_windows=1)
            self.assertEqual(fitting.call_count,1)
            self.assertEqual(result['status'],'yielded_window_boundary')
            self.assertEqual(len(result['configs']['A-probe']['windows']),1)
            self.assertEqual(read_json(root/'TOURNAMENT_STATE.json')['status'],'yielded_window_boundary')


if __name__=='__main__':unittest.main()
