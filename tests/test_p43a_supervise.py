import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from hm_recsys.p43a_supervise import supervise,write,read


class SupervisorTests(unittest.TestCase):
    def fixture(self,folder,status='yielded_window_boundary',deadline=9999999999):
        repo=Path(folder);root=repo/'artifacts/phase4/p4-3a-v1-map-cashout-hash10-tournament';root.mkdir(parents=True)
        session=repo/'session.json';write(session,dict(session_id='test',budget=dict(deadline_epoch=deadline)))
        state=dict(status=status,configs={'A':dict(windows={})},final_week='not_run')
        write(root/'TOURNAMENT_STATE.json',state)
        return repo,root,session

    def test_deadline_between_workers_does_not_launch(self):
        with TemporaryDirectory() as temp:
            repo,root,session=self.fixture(temp,deadline=0)
            with patch('hm_recsys.p43a_supervise.subprocess.Popen') as proc:
                result=supervise(repo,session)
            self.assertFalse(proc.called)
            self.assertEqual(result['status'],'paused_time_budget')

    def test_resource_pause_has_no_automatic_retry(self):
        with TemporaryDirectory() as temp:
            repo,root,session=self.fixture(temp,status='paused_resources')
            with patch('hm_recsys.p43a_supervise.subprocess.Popen') as proc:
                result=supervise(repo,session)
            self.assertFalse(proc.called)
            self.assertEqual(result['status'],'paused_resources')

    def test_candidate_gate_stops_sequential_controller(self):
        with TemporaryDirectory() as temp:
            repo,root,session=self.fixture(temp)
            def finish():
                value=read(root/'TOURNAMENT_STATE.json')
                value.update(status='paused_candidate_numeric_gate')
                value['configs']['A']['windows']['winter']={}
                write(root/'TOURNAMENT_STATE.json',value)
                return 0
            with patch('hm_recsys.p43a_supervise.subprocess.Popen') as proc:
                proc.return_value.pid=7;proc.return_value.wait.side_effect=finish
                result=supervise(repo,session)
            self.assertEqual(proc.call_count,1)
            self.assertEqual(result['status'],'paused_candidate_numeric_gate')
            command=proc.call_args.args[0]
            self.assertEqual(command[-2:],['--max-new-windows','1'])


if __name__=='__main__':unittest.main()
