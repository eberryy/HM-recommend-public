"""No-fit tests for the complete, bounded P4.3A search specification."""
import unittest
from collections import Counter
from itertools import product

import numpy as np

from hm_recsys.p43a_contract import (
    CANDIDATE_TOPK, COMMON_PARAMS, F_FEATURES, GLOBAL_PERCENTILES, LABEL_GAIN,
    MAX_ADMISSIONS, META_FEATURES, MODEL_CONFIGS, SLOT_FLOORS, TOP_EDGES,
    _deadline, c_blends, f_configs, model_configs,
)


class P43AContractTests(unittest.TestCase):
    def test_exact_model_counts_and_ids(self):
        configs = model_configs()
        self.assertEqual(Counter(c['arm'] for c in configs),
                         {'A': 16, 'B': 32, 'C': 3, 'D': 8, 'E': 1})
        self.assertEqual(len({c['id'] for c in configs}), 60)
        for c in configs:
            self.assertTrue({'id', 'arm', 'role', 'sampling', 'params', 'pos_weight'} <= c.keys())
            self.assertTrue(c['trainable'])
            self.assertEqual(c['params']['n_jobs'], 4)
            self.assertEqual(c['params']['random_state'], 20260912)
            self.assertTrue(c['params']['deterministic'])

    def test_A_full_grid_and_no_old_neutral_IPW(self):
        configs = [c for c in model_configs() if c['arm'] == 'A']
        observed = {(c['sampling'], c['params']['num_leaves'], c['params']['max_depth'],
                     c['params']['min_child_samples']) for c in configs}
        self.assertEqual(observed, set(product(('A1', 'A2'), (15, 31), (4, 6), (50, 200))))
        for c in configs:
            self.assertEqual(c['params']['label_gain'], [0, 1, 4, 10, 20])
            self.assertEqual(c['params']['n_estimators'], 300)
            self.assertEqual(c['group'], 'user-window')
            self.assertEqual(c['row_weight'], 1.)

    def test_B_full_grid_weights_and_shared_sampling(self):
        configs = [c for c in model_configs() if c['arm'] == 'B']
        observed = {(c['sampling'], c['pos_weight'], c['params']['num_leaves'],
                     c['params']['max_depth']) for c in configs}
        self.assertEqual(observed, set(product(('A1', 'A2'), (20., 50., 100., 200.), (15, 31), (4, 6))))
        for c in configs:
            self.assertEqual(c['negative_weight'], 1.)
            self.assertEqual(c['params']['objective'], 'binary')
            self.assertEqual(c['params']['min_child_samples'], 100)

    def test_D_explicit_structures_not_erroneous_prose_count(self):
        configs = [c for c in model_configs() if c['arm'] == 'D']
        observed = {(c['params']['max_depth'], c['params']['num_leaves'],
                     c['params']['min_child_samples']) for c in configs}
        expected = {(d, l, m) for d, l in ((4, 15), (6, 15), (6, 31), (-1, 63)) for m in (50, 200)}
        self.assertEqual(observed, expected)
        for c in configs:
            self.assertEqual(c['sampling'], 'A1')
            self.assertEqual(c['row_weight'], 1.)
            self.assertEqual(c['params']['learning_rate'], .03)
            self.assertEqual(c['params']['n_estimators'], 500)
        self.assertEqual(len(META_FEATURES), 14)
        self.assertEqual(len(set(META_FEATURES)), 14)

    def test_C_and_E_single_frozen_structures(self):
        configs = [c for c in model_configs() if c['arm'] in ('C', 'E')]
        self.assertEqual(len(configs), 4)
        for c in configs:
            for key in ('num_leaves', 'max_depth', 'min_child_samples', 'learning_rate',
                        'n_estimators', 'reg_lambda', 'reg_alpha'):
                self.assertEqual(c['params'][key], COMMON_PARAMS[key])
            self.assertEqual(c['pos_weight'], 1.)
            self.assertEqual(c['negative_weight'], 1.)
        c_count = next(c for c in configs if c['id'] == 'C-count')
        self.assertEqual(c_count['params']['objective'], 'multiclass')
        self.assertEqual(c_count['params']['num_class'], 4)
        self.assertEqual(next(c for c in configs if c['arm'] == 'E')['sampling'], 'E')

    def test_policy_grid_has_no_hidden_extra_dimension(self):
        configs = list(product(TOP_EDGES, GLOBAL_PERCENTILES, CANDIDATE_TOPK, SLOT_FLOORS, MAX_ADMISSIONS))
        self.assertEqual(len(configs), 1920)
        self.assertEqual(len(CANDIDATE_TOPK)*len(SLOT_FLOORS), 16)

    def test_C_blends_exact_26_including_proportional_duplicates(self):
        blends = c_blends()
        self.assertEqual(len(blends), 26)
        self.assertEqual(len({c['id'] for c in blends}), 26)
        observed = {tuple(c['weights'].values()) for c in blends}
        expected = set(product((0., .5, 1.), repeat=3)) - {(0., 0., 0.)}
        self.assertEqual(observed, expected)
        self.assertIn((.5, .5, .5), observed)
        self.assertIn((1., 1., 1.), observed)

    def test_F_exact128_repeatable_weights_and_knobs(self):
        first, second = f_configs(), f_configs()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 128)
        self.assertEqual(len({c['id'] for c in first}), 128)
        for c in first:
            self.assertFalse(c['trainable'])
            self.assertEqual(list(c['weights']), F_FEATURES)
            self.assertTrue(all(w > 0 for w in c['weights'].values()))
            self.assertAlmostEqual(sum(c['weights'].values()), 1., places=14)
            self.assertIn(c['candidate_topK'], CANDIDATE_TOPK)
            self.assertIn(c['replaceable_slot_floor'], SLOT_FLOORS)
            self.assertIn(c['max_admissions'], MAX_ADMISSIONS)
            self.assertIn(c['score_percentile_gate'], GLOBAL_PERCENTILES)

    def test_exported_configs_are_independent_copies(self):
        configs = model_configs()
        configs[0]['params']['num_leaves'] = -100
        configs[0]['params']['label_gain'][0] = -100
        self.assertEqual(MODEL_CONFIGS[0]['params']['num_leaves'], 15)
        self.assertEqual(MODEL_CONFIGS[0]['params']['label_gain'], LABEL_GAIN)
        blends = f_configs()
        blends[0]['weights']['B0'] = -100
        self.assertGreater(f_configs()[0]['weights']['B0'], 0)

    def test_two_hour_deadline_not_a_new_process_timer(self):
        budget = _deadline(1000., 8200.)
        self.assertEqual(budget['deadline_epoch'], 8200.)
        self.assertEqual(budget['total_seconds'], 7200)
        self.assertIn('implementation', budget['includes'])
        for start, stop in [(0, 7199), (0, 7201), (np.nan, 7200), (0, np.inf)]:
            with self.assertRaises(ValueError):
                _deadline(start, stop)


if __name__ == '__main__':
    unittest.main()
