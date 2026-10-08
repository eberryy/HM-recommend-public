"""Independent exhaustive graph and exact-list tests, no model fits."""
from itertools import combinations
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from hm_recsys.p43a_policy import (
    policies, matching_table, exact_ap, selected_lists, evaluate_grid, contexts, evaluate_residual,
    global_midrank_thresholds,
)
from hm_recsys.metrics import apk


def toy_data(no_cold=False):
    warm = np.array([[f'w{i}' for i in range(12)], [f'v{i}' for i in range(12)]], object)
    cold = pd.DataFrame(dict(user_index=[] if no_cold else [0, 0],
                             customer_id=[] if no_cold else ['u0', 'u0'],
                             article_id=[] if no_cold else ['c0', 'c1'],
                             b0_rank=[] if no_cold else [1, 2], target=[] if no_cold else [1, 1]))
    truth = pd.DataFrame(dict(customer_id=['u0', 'u0', 'u0', 'u1'],
                              article_id=['c0', 'c1', 'w4', 'v0'],
                              interaction_count_before_cutoff=[0, 2, 30, 30]))
    truths = {'u0': {'c0', 'c1', 'w4'}, 'u1': {'v0'}}
    base = np.array([exact_ap(items, truths[user]) for items, user in zip(warm, ['u0', 'u1'])])
    return dict(cutoff='2020-01-22', users=np.array(['u0', 'u1']), cold=cold,
                warm_lists=warm, truth=truth, truthsets=truths, baseline_ap=base,
                baseline_map=float(base.mean()))


class PolicyTests(unittest.TestCase):
    def test_global_midrank_constant_and_singleton(self):
        for values in (np.full(100, -3.), np.array([-1.])):
            thresholds = global_midrank_thresholds(values, [50, 90, 95, 99.9])
            self.assertEqual(thresholds[50], float(values[0]))
            self.assertTrue(all(thresholds[q] is None for q in (90, 95, 99.9)))
        self.assertEqual(global_midrank_thresholds([], [90]), {90: None})

    def test_global_midrank_large_tied_blocks(self):
        values = np.r_[np.zeros(90), np.ones(10)]
        thresholds = global_midrank_thresholds(values, [90, 95, 98, 99])
        self.assertEqual(thresholds, {90: 1., 95: 1., 98: None, 99: None})
        # Literal row-wise formula independently checks every selected edge.
        for q, threshold in thresholds.items():
            expected = np.array([((values < s).sum() + (values <= s).sum() - 1) /
                                  (2 * (len(values) - 1)) >= q / 100 for s in values])
            actual = values >= threshold if threshold is not None else np.zeros(len(values), bool)
            np.testing.assert_array_equal(actual, expected)

    def test_global_midrank_continuous_and_small_sample(self):
        self.assertEqual(global_midrank_thresholds(np.arange(101.), [90, 95, 99.9]),
                         {90: 90., 95: 95., 99.9: 100.})
        self.assertEqual(global_midrank_thresholds([-3., -2., -1.], [90, 95, 99.9]),
                         {90: -1., 95: -1., 99.9: -1.})

    def test_constant_grid_scores_do_not_all_pass_quantile_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = evaluate_grid(toy_data(), np.full((2, 12), .5), policies()[:4], Path(directory))
            for row in rows:
                self.assertEqual(row['replacements'], 0)
                self.assertEqual(row['delta_map'], 0.)
                self.assertIsNone(row['global_score_threshold'])
                self.assertEqual(row['global_score_threshold_status'], 'no_score_reaches_percentile')

    def test_exact_AP_legacy_parity(self):
        data = toy_data()
        truths, valid, base = contexts(data)
        for i, items in enumerate(data['warm_lists']):
            for j, truth in enumerate(truths[i]):
                self.assertAlmostEqual(base[i, j], apk(list(truth), list(items)), places=15)

    def test_grid_exact_count(self):
        grid = policies()
        self.assertEqual(len(grid), 1920)
        self.assertEqual(len({v['id'] for v in grid}), 1920)

    def test_DP_matches_exhaustive_every_mask_and_cap(self):
        rng = np.random.default_rng(2834)
        for _ in range(8):
            cold, slots = rng.integers(0, 4, size=(2, 7))
            weights = rng.integers(-2, 6, size=7).astype(float)
            caps = (0, 1, 2, 3, 12)
            table, value = matching_table(cold, slots, weights, caps)
            for allowed in range(128):
                available = [i for i in range(7) if allowed & (1 << i)]
                for row, cap in enumerate(caps):
                    best, winner = 0., 0
                    for n in range(1, min(cap, len(available)) + 1):
                        for selected in combinations(available, n):
                            if len({cold[i] for i in selected}) != n or len({slots[i] for i in selected}) != n:
                                continue
                            objective = float(sum(weights[i] for i in selected))
                            mask = sum(1 << i for i in selected)
                            if objective > best or (objective == best and mask < winner):
                                best, winner = objective, mask
                    self.assertEqual(float(value[row, allowed]), best)
                    self.assertEqual(int(table[row, allowed]), winner)

    def test_tie_and_empty_graph(self):
        table, _ = matching_table([0, 1], [0, 0], [1., 1.], (1, 12))
        np.testing.assert_array_equal(table[:, 3], [1, 1])
        empty, score = matching_table([], [], [], (1, 2, 12))
        np.testing.assert_array_equal(empty, np.zeros((3, 1), np.uint16))
        np.testing.assert_array_equal(score, np.zeros((3, 1)))

    def test_two_replacements_exact_AP_not_single_delta_sum(self):
        warm = [f'w{i}' for i in range(12)]
        truth = {'c0', 'c1'}
        one, _, _ = selected_lists(warm, ['c0', 'c1'], [0, 13], 1)
        two, _, _ = selected_lists(warm, ['c0', 'c1'], [0, 13], 2)
        both, _, _ = selected_lists(warm, ['c0', 'c1'], [0, 13], 3)
        self.assertEqual(exact_ap(both, truth), 1.)
        self.assertEqual(exact_ap(one, truth) + exact_ap(two, truth), .75)
        with self.assertRaises(ValueError):
            selected_lists(warm, ['c0', 'c1'], [0, 12], 3)

    def test_negative_raw_scores_pass_explicit_percentile_gate(self):
        data = toy_data()
        score = -np.arange(1, 25, dtype=float).reshape(2, 12)
        policy = dict(id='negative', top_edge=10, global_percentile=90,
                      candidate_topk=50, slot_floor=1, max_admissions=12)
        with tempfile.TemporaryDirectory() as directory:
            rows = evaluate_grid(data, score, [policy], Path(directory))
            self.assertGreater(rows[0]['replacements'], 0)
            saved = np.load(rows[0]['saved_decisions'])
            top = np.load(rows[0]['saved_top_edges'])
            self.assertGreater(int(saved[0, 0]), 0)
            self.assertEqual(int(saved[0, 1]), 0)
            items, _, _ = selected_lists(data['warm_lists'][0], ['c0', 'c1'], top[0][top[0] >= 0], saved[0, 0])
            expected = (exact_ap(items, data['truthsets']['u0']) + data['baseline_ap'][1]) / 2
            self.assertAlmostEqual(rows[0]['overall_map'], expected)
            self.assertEqual(rows, evaluate_grid(data, score, [policy], Path(directory)))

    def test_no_candidates_keeps_all_users(self):
        data = toy_data(no_cold=True)
        with tempfile.TemporaryDirectory() as directory:
            rows = evaluate_grid(data, np.empty((0, 12)), policies()[:2], Path(directory))
            for row in rows:
                self.assertEqual(row['users'], 2)
                self.assertEqual(row['replacements'], 0)
                self.assertEqual(row['delta_map'], 0.)

    def test_final_week_prohibited(self):
        data = toy_data()
        data['cutoff'] = '2020-09-16'
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            evaluate_grid(data, np.zeros((2, 12)), policies()[:1], Path(directory))

    def test_residual16_policies_preserve_slots_and_evidence(self):
        data = toy_data()
        score = np.full((2, 12), .1)
        score[0, 11], score[1, 10] = .8, .9
        with tempfile.TemporaryDirectory() as directory:
            rows = evaluate_residual(data, score, Path(directory))
            self.assertEqual(len(rows), 16)
            pairs = np.load(rows[0]['saved_decisions'])
            for pi, row in enumerate(rows):
                result = list(data['warm_lists'][0])
                selected = pairs[pi, 0]
                selected = selected[selected[:, 0] >= 0]
                for cr, slot in selected:
                    self.assertGreaterEqual(slot + 1, row['policy']['slot_floor'])
                    result[slot] = data['cold'].iloc[cr].article_id
                expected = (exact_ap(result, data['truthsets']['u0']) + data['baseline_ap'][1]) / 2
                self.assertAlmostEqual(row['overall_map'], expected)
                self.assertEqual(row['replacements'], len(selected))
                self.assertTrue(np.all(pairs[pi, 1] == -1))

    def test_residual_zero_scores_tie_to_no_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = evaluate_residual(toy_data(), np.zeros((2, 12)), Path(directory))
            self.assertTrue(all(row['replacements'] == 0 and row['delta_map'] == 0 for row in rows))


if __name__ == '__main__':
    unittest.main()
