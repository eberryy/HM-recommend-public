"""Signed exact-K matching checked against independent enumeration and LP."""
from itertools import combinations, permutations
import unittest

import numpy as np
from scipy.optimize import linprog

from hm_recsys.p43a_oracle import exactly_k_matching, oracle_user, restricted_oracle_user


def brute_value(weights, k):
    best = -np.inf
    nc, nw = weights.shape
    for cold in combinations(range(nc), k):
        for warm in permutations(range(nw), k):
            best = max(best, sum(weights[i, j] for i, j in zip(cold, warm)))
    return best


class OracleTests(unittest.TestCase):
    def test_signed_exactK_matches_exhaustive(self):
        rng = np.random.default_rng(383)
        for _ in range(10):
            weight = rng.integers(-5, 6, size=(4, 3)).astype(float)
            for k in range(4):
                matches = exactly_k_matching(weight, k)
                self.assertEqual(len(matches), k)
                self.assertEqual(sum(weight[i, j] for i, j in matches), brute_value(weight, k))

    def test_scaled_independent_LP_for_small_coefficients(self):
        rng = np.random.default_rng(42)
        weight = rng.normal(size=(6, 5)) * 1e-8
        nc, nw = weight.shape
        a = np.zeros((nc + nw, weight.size))
        for i in range(nc):
            for j in range(nw):
                a[i, i*nw+j] = 1
                a[nc+j, i*nw+j] = 1
        scale = np.abs(weight).max()
        for k in range(6):
            solved = linprog(-weight.ravel()/scale, A_ub=a, b_ub=np.ones(nc+nw),
                             A_eq=np.ones((1, weight.size)), b_eq=[k], bounds=(0, 1), method='highs')
            self.assertTrue(solved.success)
            matches = exactly_k_matching(weight, k)
            self.assertAlmostEqual(sum(weight[i, j] for i, j in matches), -solved.fun*scale, places=18)

    def test_negative_edges_cannot_be_rejected_when_K_required(self):
        weight = np.array([[-1., -2.], [-3., -4.]])
        self.assertEqual(len(exactly_k_matching(weight, 2)), 2)
        self.assertEqual(sum(weight[i, j] for i, j in exactly_k_matching(weight, 2)), -5.)
        self.assertIsNone(exactly_k_matching(weight, 3))
        self.assertEqual(exactly_k_matching(np.empty((0, 12)), 0), [])
        self.assertIsNone(exactly_k_matching(np.empty((0, 12)), 1))

    def test_deterministic_zero_ties_and_smallest_K(self):
        weight = np.zeros((3, 12))
        self.assertEqual(exactly_k_matching(weight, 2), exactly_k_matching(weight, 2))
        warm, cold = [f'w{i}' for i in range(12)], ['c0', 'c1', 'c2']
        result = oracle_user(warm, cold, {'not-in-pool'}, weight)
        self.assertEqual(result['k_star'], 0)
        self.assertEqual(result['best_items'], warm)
        self.assertTrue(np.isnan(result['ap_by_k'][4:]).all())

    def test_exact_final_list_AP_and_restricted_slots(self):
        warm, cold = [f'w{i}' for i in range(12)], ['c0', 'c1']
        weight = np.tile(1/np.arange(1, 13), (2, 1))
        result = oracle_user(warm, cold, set(cold), weight)
        self.assertEqual(result['k_star'], 2)
        self.assertEqual(result['best_ap'], 1.)
        self.assertEqual(len(set(result['best_items'])), 12)
        value, matching = restricted_oracle_user(warm, cold, [1, 2], set(cold), weight)
        self.assertEqual(matching, [(0, 9)])
        self.assertAlmostEqual(value, 1/20)

    def test_no_candidates_only_K0_feasible(self):
        warm = [f'w{i}' for i in range(12)]
        result = oracle_user(warm, [], {'w0'}, np.empty((0, 12)))
        self.assertEqual(result['k_star'], 0)
        self.assertEqual(result['best_ap'], 1.)
        self.assertTrue(np.isnan(result['ap_by_k'][1:]).all())


if __name__ == '__main__':
    unittest.main()
