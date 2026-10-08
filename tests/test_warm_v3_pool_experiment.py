"""INNER expanded-pool screen protocol tests without model fits or data access."""
import unittest
import pandas as pd
from hm_recsys import warm_v3_pool_experiment as experiment


class PoolExperimentTests(unittest.TestCase):
    def test_only_eight_inner_related_dates(self):
        self.assertEqual(len(experiment.APPROVED), 8)
        self.assertNotIn('2020-03-18', experiment.APPROVED)
        self.assertNotIn('2020-08-19', experiment.APPROVED)
        self.assertNotIn('2020-09-16', experiment.APPROVED)
        self.assertIn('2020-01-22', experiment.TRAIN_CUTOFFS)

    def test_newly_covered_users_get_zero_old_ap_full_denominator(self):
        new = pd.DataFrame({'customer_id': ['a', 'b'], 'ap_e0': [.2, .1], 'ap_e1': [.3, .2], 'ap_rrf': [.3, .2]})
        old = pd.DataFrame({'customer_id': ['a'], 'ap_rrf': [.1]})
        merged, values = experiment.compare_ap(new, old, 4)
        self.assertEqual(merged.newly_covered.tolist(), [False, True])
        self.assertAlmostEqual(values['rrf']['population_delta'], .1)
        self.assertEqual(values['rrf']['improved_users'], 2)

    def test_old_covered_population_cannot_disappear(self):
        new = pd.DataFrame({'customer_id': ['a'], 'ap_e0': [.1], 'ap_e1': [.1], 'ap_rrf': [.1]})
        old = pd.DataFrame({'customer_id': ['b'], 'ap_rrf': [.1]})
        with self.assertRaises(ValueError):
            experiment.compare_ap(new, old, 3)

    def test_fixed_gate_requires_mean_three_wins_and_worst_bound(self):
        self.assertTrue(experiment.screen_gate([.001, .0002, .0001, -.0003])['passed'])
        self.assertFalse(experiment.screen_gate([.00001] * 4)['passed'])
        self.assertFalse(experiment.screen_gate([.002, .001, .001, -.0006])['passed'])
        self.assertFalse(experiment.screen_gate([.002, .001, -.0001, -.0001])['passed'])

    def test_gate_cannot_choose_favorable_subset_of_windows(self):
        with self.assertRaises(ValueError):
            experiment.screen_gate([.001, .002])


if __name__ == '__main__':
    unittest.main()
