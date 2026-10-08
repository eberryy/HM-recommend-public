import numpy as np
import unittest
from hm_recsys.p43a_verify import (empirical_percentiles, brute_matching,
    fixed_policies, reconstruct, guard_cutoff, completed_snapshot)


class VerifyTests(unittest.TestCase):
    def test_average_ties_and_singleton(self):
        np.testing.assert_equal(empirical_percentiles([1, 1, 3, 5]), [1/6, 1/6, 2/3, 1])
        np.testing.assert_equal(empirical_percentiles([9]), [.5])
        np.testing.assert_equal(empirical_percentiles([9, 9]), [.5, .5])
        with self.assertRaises(ValueError):
            empirical_percentiles([np.nan])


    def test_matching_constraints_independent_enumeration(self):
        edges = np.array([0, 1, 12])
        self.assertEqual(brute_matching(edges, [True]*3, 2), (1., 1))
        value, mask = brute_matching(edges, [False, True, True], 2)
        self.assertAlmostEqual(value, 5/6)
        self.assertEqual(mask, 6)
        self.assertEqual(brute_matching(edges, [False, False, False], 12), (0., 0))


    def test_reconstruct_fixed_position_and_collision(self):
        warm = list(range(12))
        self.assertEqual(reconstruct(warm, [99], [11]), list(range(11))+[99])
        with self.assertRaises(AssertionError):
            reconstruct(warm, [99], [0, 1])
        with self.assertRaises(AssertionError):
            reconstruct(warm, [1], [0])


    def test_no_final_and_frozen_grid(self):
        self.assertEqual(guard_cutoff('2020-08-19'), '2020-08-19')
        with self.assertRaises(ValueError):
            guard_cutoff('2020-09-16')
        self.assertEqual(len(fixed_policies()), 1920)
        self.assertEqual(fixed_policies()[0]['slot_floor'], 12)
        self.assertEqual(fixed_policies()[-1]['max_admissions'], 12)


    def test_no_partial_model_default_scope(self):
        state = dict(configs={'A1-foo':dict(config={'arm':'A'}, windows={})})
        self.assertEqual(completed_snapshot(state), [])


if __name__ == '__main__':
    unittest.main()
