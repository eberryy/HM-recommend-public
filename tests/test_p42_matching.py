import itertools
import unittest

import numpy as np

from hm_recsys.p42_matching import (
    EPSILON, TAUS, apply_admissions, clipped_logit, exact_matching,
    matching_diagnostics, pair_utilities,
)


def exhaustive_best_weight(values, tau):
    """Independent enumeration of partial one-to-one assignments for tiny graphs."""
    best = 0.0
    n_cold, n_warm = values.shape
    for choices in itertools.product(range(-1, n_cold), repeat=n_warm):
        used = [cold for cold in choices if cold >= 0]
        if len(used) != len(set(used)):
            continue
        if any(cold >= 0 and values[cold, warm] <= tau
               for warm, cold in enumerate(choices)):
            continue
        best = max(best, sum(values[cold, warm] - tau
                             for warm, cold in enumerate(choices) if cold >= 0))
    return best


class P42MatchingTest(unittest.TestCase):
    def test_probability_clip_and_utility_exact(self):
        self.assertEqual(EPSILON, 1e-6)
        self.assertEqual(TAUS, (0.0, 0.6931471805599453, 1.3862943611198906))
        q = np.array([0., 1e-10, .01, .5, .8, 1 - 1e-10, 1.])
        expected_q = np.clip(q, 1e-6, 1 - 1e-6)
        expected_logit = np.log(expected_q / (1 - expected_q))
        np.testing.assert_array_equal(clipped_logit(q), expected_logit)
        np.testing.assert_array_equal(pair_utilities(q[:3], q[3:]),
                                      expected_logit[:3, None] - expected_logit[None, 3:])
        self.assertEqual(clipped_logit(0.5), 0.0)

    def test_invalid_inputs_fail_closed(self):
        for q in ([np.nan], [np.inf], [-.1], [1.1]):
            with self.assertRaises(ValueError):
                clipped_logit(q)
        for values, tau in (([[np.nan]], 0), ([[np.inf]], 0), ([1], 0),
                            (np.zeros((51, 12)), 0), (np.zeros((1, 13)), 0),
                            ([[1]], .1), ([[1]], np.nan)):
            with self.assertRaises(ValueError):
                exact_matching(values, tau)
        with self.assertRaises(ValueError):
            pair_utilities([[.1]], [.2])

    def test_threshold_equality_is_rejected(self):
        for tau in TAUS:
            values = np.array([[tau, np.nextafter(tau, np.inf)]])
            self.assertEqual(exact_matching(values, tau), [(0, 1)])
            self.assertEqual(exact_matching([[tau]], tau), [])

    def test_empty_and_no_positive_edges(self):
        for values in (np.empty((0, 12)), np.empty((5, 0)), np.empty((0, 0)),
                       np.zeros((50, 12)), -np.ones((2, 12))):
            self.assertEqual(exact_matching(values, 0), [])
            diagnostic = matching_diagnostics(values, 0)
            self.assertEqual(diagnostic["matched_edges"], 0)
            self.assertEqual(diagnostic["matching_efficiency"], 0)
            self.assertFalse(diagnostic["greedy_would_differ"])

    def test_exact_solver_beats_greedy_counterexample(self):
        values = np.array([[10., 9.], [9., 0.]])
        self.assertEqual(exact_matching(values, 0), [(1, 0), (0, 1)])
        audit = matching_diagnostics(values, 0)
        self.assertEqual(audit["matched_weight_total"], 18.)
        self.assertEqual(audit["greedy_weight_total_diagnostic_only"], 10.)
        self.assertTrue(audit["greedy_would_differ"])
        self.assertEqual(audit["same_warm_conflict_count"], 1)
        self.assertEqual(audit["same_cold_conflict_count"], 1)
        self.assertEqual(audit["matching_efficiency"], 2 / 3)

    def test_exhaustive_small_graph_optimum_all_thresholds(self):
        rng = np.random.default_rng(20260909)
        for n_cold, n_warm in ((1, 1), (2, 3), (3, 2), (3, 4), (4, 3)):
            for _ in range(20):
                values = rng.integers(-2, 5, size=(n_cold, n_warm)).astype(float)
                for tau in TAUS:
                    matches = exact_matching(values, tau)
                    observed = sum(values[c, w] - tau for c, w in matches)
                    self.assertAlmostEqual(observed, exhaustive_best_weight(values, tau),
                                           places=12)
                    self.assertEqual(len({c for c, _ in matches}), len(matches))
                    self.assertEqual(len({w for _, w in matches}), len(matches))
                    self.assertEqual(matches, sorted(matches, key=lambda pair: pair[1]))
                    self.assertTrue(all(values[c, w] > tau for c, w in matches))

    def test_ties_are_deterministic_without_numeric_jitter(self):
        values = np.ones((50, 12))
        expected = [(slot, slot) for slot in range(12)]
        for _ in range(10):
            self.assertEqual(exact_matching(values.copy(), 0), expected)
        np.testing.assert_array_equal(values, np.ones((50, 12)))
        # Separable utilities have equal-sum assignment permutations; a repeat
        # must be stable, but we deliberately do not claim a unique optimum.
        separated = np.array([3., 2., 1.])[:, None] - np.array([.2, .1])[None, :]
        expected = exact_matching(separated, 0)
        self.assertEqual(exact_matching(separated.copy(), 0), expected)
        self.assertAlmostEqual(sum(separated[c, w] for c, w in expected),
                               exhaustive_best_weight(separated, 0), places=12)

    def test_no_max_one_cap_and_600_pair_budget(self):
        values = np.ones((50, 12))
        matches = exact_matching(values, 0)
        self.assertEqual(len(matches), 12)
        audit = matching_diagnostics(values, 0, matches)
        self.assertEqual(audit["pair_rows"], 600)
        self.assertEqual(audit["edges_above_tau"], 600)
        self.assertEqual(audit["matched_edges"], 12)
        self.assertEqual(audit["same_warm_conflict_count"], 12)
        self.assertEqual(audit["same_cold_conflict_count"], 50)
        self.assertEqual(audit["matching_efficiency"], .02)

    def test_diagnostic_rejects_invalid_matching(self):
        values = np.array([[2., 0.], [1., 1.]])
        for matches in ([(0, 0), (0, 1)], [(0, 0), (1, 0)], [(0, 1)], [(2, 0)]):
            with self.assertRaises(ValueError):
                matching_diagnostics(values, 0, matches)

    def test_final_list_exact_slots_unique_and_no_edge_parity(self):
        warm = [f"w{slot}" for slot in range(12)]
        cold = [f"c{rank}" for rank in range(50)]
        self.assertEqual(apply_admissions(warm, cold, []), warm)
        changed = apply_admissions(warm, cold, [(4, 2), (0, 9)])
        self.assertEqual(changed[2], "c4")
        self.assertEqual(changed[9], "c0")
        self.assertEqual(len(set(changed)), 12)
        self.assertEqual([changed[i] for i in range(12) if i not in (2, 9)],
                         [warm[i] for i in range(12) if i not in (2, 9)])
        twelve = apply_admissions(warm, cold, [(i, i) for i in range(12)])
        self.assertEqual(twelve, cold[:12])
        for cold_bad, matches in ((["w0"], []), (["c0", "c0"], []),
                                  (cold, [(0, 0), (0, 1)]),
                                  (cold, [(0, 0), (1, 0)]),
                                  (cold, [(50, 0)]), (cold, [(0, 12)])):
            with self.assertRaises(ValueError):
                apply_admissions(warm, cold_bad, matches)


if __name__ == "__main__":
    unittest.main()
