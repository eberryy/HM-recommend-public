"""Only temporary fixture closeouts; no real experiment/report rendering."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from hm_recsys.p41a_contract import identity, read_json, write_json
from hm_recsys.p43a_close import close, receipt_paths, structural_metadata


class CloseTests(unittest.TestCase):
    def fixture(self, base, status='paused_time_budget'):
        repo = Path(base)/'repo'
        root = repo/'artifacts/phase4/run'
        root.mkdir(parents=True)
        report = repo/'reports/phase4'
        report.mkdir(parents=True)
        contract = report/'P4_3A_EXPERIMENT_CONTRACT.json'
        write_json(contract, dict(final_week='not_run', sealed_confirmatory_holdout='2020-09-16',
            budget=dict(deadline_epoch=1234, deadline_utc='1970-01-01T00:20:34Z')))
        write_json(root/'EXECUTION_START.json', dict(contract=identity(contract)))
        write_json(root/'TOURNAMENT_STATE.json', dict(status=status, final_week='not_run',
            paused_at='2026-09-12T07:50:58Z'))
        write_json(report/'P4_3A_VERIFICATION.json', dict(status='pass_completed_snapshot',
            attempt_id='fixture', scope_policy_rows=1920, complete_stage_pass=False))
        return repo, root

    def test_running_rejected_before_report(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, root = self.fixture(temp, 'running')
            with patch('hm_recsys.p43a_report.write_report') as render:
                with self.assertRaises(RuntimeError):
                    close(repo, root)
                render.assert_not_called()
            self.assertFalse((root/'CLOSURE.json').exists())

    def test_paused_close_preserves_evidence_and_second_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, root = self.fixture(temp)
            verification = repo/'reports/phase4/P4_3A_VERIFICATION.json'
            original = verification.read_bytes()
            np.save(root/'scores.npy', np.ones((3, 12)))
            with patch('hm_recsys.p43a_report.write_report', return_value={'status':'partial'}):
                result = close(repo, root)
                self.assertEqual(result['status'], 'paused_time_budget')
                self.assertFalse(result['complete_stage_pass'])
                self.assertEqual(verification.read_bytes(), original)
                self.assertEqual(result['artifact_metadata'][0]['shape'], [3,12])
                self.assertIsNone(result['artifact_metadata'][0]['sha256'])
                again = close(repo, root)
            self.assertEqual(len(again['previous_closures_archived']), 2)
            for record in again['previous_closures_archived']:
                self.assertTrue(Path(record['archived']).is_file())
            self.assertEqual(read_json(root/'CLOSURE.json')['immutable_deadline_epoch'], 1234)

    def test_contract_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, root = self.fixture(temp)
            write_json(repo/'reports/phase4/P4_3A_EXPERIMENT_CONTRACT.json', {'changed':True})
            with self.assertRaises(ValueError):
                close(repo, root)

    def test_report_must_not_overwrite_independent_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, root = self.fixture(temp)
            def bad_report(*_):
                write_json(repo/'reports/phase4/P4_3A_VERIFICATION.json', {'status':'pass'})
                return {'status':'partial'}
            with patch('hm_recsys.p43a_report.write_report', side_effect=bad_report):
                with self.assertRaises(RuntimeError):
                    close(repo, root)

    def test_structural_npz_and_excluded_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            np.savez_compressed(root/'data.npz', X=np.zeros((5, 7), np.float32))
            metadata = structural_metadata(root/'data.npz')
            self.assertEqual(metadata['members'][0]['shape'], [5,7])
            self.assertEqual(metadata['members'][0]['dtype'], 'float32')
            write_json(root/'models/a/FIT_RESULT.json', {'status':'completed'})
            write_json(root/'closure-history/old/CLOSURE.json', {})
            write_json(root/'verification-attempts/old/RESULT.json', {})
            self.assertEqual([p.name for p in receipt_paths(root)], ['FIT_RESULT.json'])


if __name__ == '__main__':
    unittest.main()
