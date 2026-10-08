"""Runner mechanics tests; synthetic data only, never fit a model."""
import tempfile
import time
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from hm_recsys.p42f_contract import GLOBAL, WINDOWS, earlier
from hm_recsys.p43a_run import BudgetPause, guard, relevance, pool, model_order, predict
from hm_recsys.p43a_data import META_COLUMNS
from hm_recsys.p43a_contract import model_configs


class RunnerTests(unittest.TestCase):
    def test_exact_relevance_boundaries(self):
        y = np.array([-.5, -.1, -1e-12, 0., 1e-12, .099999, .1, .8])
        np.testing.assert_array_equal(relevance(y), [0, 0, 0, 1, 3, 3, 4, 4])

    def test_deadline_checked_before_filesystem(self):
        with self.assertRaises(BudgetPause):
            guard(Path('no-such-path'), time.time()-1)

    def test_frozen_priority(self):
        configs = [dict(arm=a, id=a) for a in 'DCBEA']
        self.assertEqual([c['arm'] for c in model_order(configs)], ['A','B','E','D','C'])

    def test_pool_groups_do_not_merge_user_across_dates(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            root = repo/'artifacts'
            window = 'spring_20200318'
            def prepare(_repo, _root, t, **kwargs):
                folder = repo/'fake'/t
                folder.mkdir(parents=True, exist_ok=True)
                values = dict(X=np.zeros((2,len(GLOBAL)), np.float32), y=np.array([0., .25]),
                              user_index=np.array([0,0]), group=np.array([2]))
                for k,v in values.items():
                    np.save(folder/(k+'.npy'), v)
                return dict(rows=2, paths={k:str(folder/(k+'.npy')) for k in values})
            with patch('hm_recsys.p43a_data.prepare_date', side_effect=prepare), patch('hm_recsys.p43a_run.guard'):
                result = pool(repo,root,window,'A1',None)
            dates = earlier(WINDOWS[window])
            self.assertEqual(result['rows'],2*len(dates))
            np.testing.assert_array_equal(np.load(result['group']),[2]*len(dates))
            self.assertEqual(result['counts'],dict(B=len(dates),N=len(dates),H=0))

    def test_D_sparse_edge_identity_joins_all_meta_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp); root = repo/'artifacts'; window = 'spring_20200318'
            dates = earlier(WINDOWS[window]); expected = []
            def prepare(_repo, _root, t, **kwargs):
                self.assertEqual(kwargs['sampling'], 'A1')
                folder = repo/'fake'/t; folder.mkdir(parents=True, exist_ok=True)
                values = dict(X=np.full((3, len(GLOBAL)), dates.index(t), np.float32),
                              y=np.array([.25, 0., -.1]), user_index=np.array([0, 0, 1]),
                              edge_index=np.array([1, 5, 15], np.int64))
                for name, value in values.items(): np.save(folder/(name+'.npy'), value)
                return dict(rows=3, paths={name:str(folder/(name+'.npy')) for name in values})
            def meta(_repo, _root, t, **kwargs):
                values = np.zeros((24, len(META_COLUMNS)), np.float32)
                for j in range(7):
                    available = (j + dates.index(t)) % 2
                    values[:, 2*j+1] = available
                    values[:, 2*j] = available * (np.arange(24) + j) / 50.
                folder = repo/'fake'/t; np.save(folder/'meta.npy', values)
                expected.append(values[[1, 5, 15]])
                return dict(paths=dict(X=str(folder/'meta.npy')))
            with patch('hm_recsys.p43a_data.prepare_date', side_effect=prepare), \
                 patch('hm_recsys.p43a_data.ensure_meta', side_effect=meta), \
                 patch('hm_recsys.p43a_run.guard'):
                result = pool(repo, root, window, 'D', None)
            self.assertEqual(result['features'], GLOBAL + META_COLUMNS)
            values = np.load(result['X'])
            self.assertEqual(values.shape, (3*len(dates), 84))
            np.testing.assert_array_equal(values[:, len(GLOBAL):], np.concatenate(expected))
            np.testing.assert_array_equal(values[:, 0], np.repeat(np.arange(len(dates)), 3))
            np.testing.assert_array_equal(np.load(result['group']), [2, 1]*len(dates))
            self.assertEqual(result['counts'], dict(B=len(dates), N=len(dates), H=len(dates)))

    def test_D_prediction_batch_join_matches_complete_edge_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp); root = repo/'artifacts'; window = 'winter_20200122'
            cold = pd.DataFrame(dict(user_index=[0, 0, 1]))
            data = dict(cutoff=WINDOWS[window], cold=cold)
            values = np.zeros((36, len(META_COLUMNS)), np.float32)
            values[:, ::2] = np.arange(36)[:, None] / 50.
            values[:, 1::2] = 1.
            np.save(repo/'meta.npy', values)
            blocks = [np.full((24, len(GLOBAL)), 11., np.float32),
                      np.full((12, len(GLOBAL)), 22., np.float32)]
            def batches(_data, **kwargs):
                self.assertFalse(kwargs['labels'])
                yield 0, cold.iloc[:2], blocks[0], None
                yield 2, cold.iloc[2:], blocks[1], None
            expected = [np.concatenate((blocks[0], values[:24]), axis=1),
                        np.concatenate((blocks[1], values[24:]), axis=1)]
            class Booster:
                calls = 0
                def feature_name(self): return GLOBAL + META_COLUMNS
                def predict(self, x, **kwargs):
                    np.testing.assert_array_equal(x, expected[self.calls])
                    self.calls += 1
                    return x[:, len(GLOBAL)]
            booster = Booster()
            with patch('hm_recsys.p43a_run.joblib.load', return_value=data), \
                 patch('hm_recsys.p43a_run.batches', side_effect=batches), \
                 patch('hm_recsys.p43a_data.ensure_meta', return_value=dict(paths=dict(X=str(repo/'meta.npy')))), \
                 patch('hm_recsys.p43a_run.guard'):
                _, score, _ = predict(repo, root, repo, window, dict(id='fake-D', arm='D'), booster, None)
                np.testing.assert_array_equal(score, values[:, 0].astype(np.float64).reshape(3, 12))
                del score
            self.assertEqual(booster.calls, 2)

    @unittest.skipUnless(__import__("os").environ.get("HM_RESEARCH_ASSETS") == "1", "requires withheld historical contracts/checkpoints and original research branch")
    def test_D_eight_configs_equal_frozen_contract(self):
        repo = Path(__file__).resolve().parents[1]
        contract = json.loads((repo/'reports/phase4/P4_3A_EXPERIMENT_CONTRACT.json').read_text(encoding='utf8'))
        configs = [c for c in model_configs() if c['arm'] == 'D']
        self.assertEqual(configs, [c for c in contract['model_configs'] if c['arm'] == 'D'])
        self.assertEqual(len(configs), 8)
        self.assertEqual(contract['features']['D_exact84'], GLOBAL + META_COLUMNS)
        structures = {(c['params']['max_depth'], c['params']['num_leaves']) for c in configs}
        self.assertEqual(structures, {(4, 15), (6, 15), (6, 31), (-1, 63)})
        for config in configs:
            self.assertEqual(config['params']['n_estimators'], 500)
            self.assertEqual(config['params']['learning_rate'], .03)
            self.assertEqual(config['params']['objective'], 'lambdarank')
            self.assertEqual(config['row_weight'], 1.)
            self.assertEqual(config['sampling'], 'A1')


if __name__ == '__main__':
    unittest.main()
