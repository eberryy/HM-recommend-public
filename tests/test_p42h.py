"""Synthetic P4.2H arithmetic and audit tests; no recommendation model fitting."""
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from hm_recsys.p42f_core import fold
from hm_recsys.p42g_core import classes
from hm_recsys.p42h_core import calibrated, utilities, binary_calibration, verdict
from hm_recsys.p42h_evaluate import conditional_bins, priority_diagnostic
from hm_recsys.p42h_verify import (
    calibration_gradient, direct_ap, independent_fold, independent_formula,
)
from hm_recsys.p42_matching import apply_admissions, exact_matching


class ConditionalRiskTests(unittest.TestCase):
    def test_exact_sign_no_tolerance(self):
        np.testing.assert_array_equal(classes([1e-300, 0., -1e-300, -0.]), [0, 1, 2, 1])

    def test_literal_formula_and_nonzero_weight(self):
        q = np.array([.75, .25, .4])
        r = np.array([.1, .9, .2])
        mb = np.array([.2, .2, .8])
        mh = np.array([.1, .1, .2])
        actual = utilities(q, r, mb, mh)
        expected = independent_formula(q, r, mb, mh)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(actual[1], [.125, -.025, .2], rtol=0, atol=1e-15)
        np.testing.assert_array_equal(actual[1] > 0, actual[2] > 0)

    def test_zero_magnitude_rejects(self):
        threshold, gain, weight = utilities(np.array([0., .5, 1.]), np.ones(3), np.zeros(3), np.zeros(3))
        np.testing.assert_array_equal(threshold, [1., 1., 1.])
        np.testing.assert_array_equal(gain, [0., 0., 0.])
        np.testing.assert_array_equal(weight, [0., 0., 0.])

    def test_equality_is_rejected_without_extra_half_threshold(self):
        threshold, gain, weight = utilities(np.array([.25, .3, .5]), np.ones(3),
                                             np.array([.75, .75, .5]), np.array([.25, .25, .5]))
        np.testing.assert_array_equal(gain > 0, [False, True, False])
        self.assertGreater(weight[1], 0.)
        self.assertLess(.3, .5)
        np.testing.assert_array_equal(threshold, [.25, .25, .5])

    def test_zero_r_does_not_silently_drop_eligible_edge(self):
        with self.assertRaises(ValueError):
            utilities(np.array([.9]), np.array([0.]), np.array([1.]), np.array([.1]))
        with self.assertRaises(ValueError):
            independent_formula(np.array([.9]), np.array([0.]), np.array([1.]), np.array([.1]))
        # Zero r is valid when the conditional gate rejects the edge already.
        _, gain, weight = utilities(np.array([.1]), np.array([0.]), np.array([.1]), np.array([1.]))
        self.assertLess(gain[0], 0.)
        self.assertEqual(weight[0], 0.)

    def test_invalid_scores_fail_closed(self):
        for invalid in (np.nan, np.inf, -.01, 1.01):
            with self.assertRaises(ValueError):
                utilities(np.array([invalid]), np.ones(1), np.ones(1), np.ones(1))
        with self.assertRaises(ValueError):
            utilities(np.array([.2, .3]), np.ones(1), np.ones(1), np.ones(1))

    def test_monotonic_clipped_platt(self):
        q = np.array([0., 1e-9, .2, .5, .8, 1. - 1e-9, 1.])
        cal = dict(a=-.2, b=.7, epsilon=1e-6)
        expected = expit(-.2 + .7 * logit(np.clip(q, 1e-6, 1-1e-6)))
        got = calibrated(q, cal)
        np.testing.assert_array_equal(got, expected)
        self.assertTrue(np.all(np.diff(got) >= 0))
        self.assertEqual(got[0], got[1])
        self.assertEqual(got[-1], got[-2])
        for slope in (0., -1., np.nan):
            with self.assertRaises(ValueError):
                calibrated(q, dict(a=0., b=slope, epsilon=1e-6))

    def test_unweighted_calibration_score_equations(self):
        # A constant calibrated 0.5 and balanced targets is a stationary point;
        # this checks equations only and deliberately does not call any fitter.
        fitted, gradient = calibration_gradient(np.full(4, .5), [1, 0, 0, 1], 0., 1.)
        np.testing.assert_array_equal(fitted, [.5] * 4)
        np.testing.assert_array_equal(gradient, [0., 0.])

    def test_calibration_metrics_without_fit(self):
        metrics = binary_calibration([.1, .9], [0, 1], diagnostic_fit=False)
        self.assertNotIn('diagnostic_calibration', metrics)
        self.assertAlmostEqual(metrics['Brier'], .01)
        self.assertAlmostEqual(metrics['ECE'], .1)
        self.assertAlmostEqual(metrics['logloss'], -np.log(.9))

    def test_quantile_bins_share_rows_and_retain_ties(self):
        raw = np.full(20, .2)
        cal = np.full(20, .3)
        labels = np.array([1, -1] * 10)
        result = conditional_bins(raw, cal, labels)
        self.assertEqual(len(result['quantile_bins']), 6)
        self.assertEqual(result['quantile_bins'][0]['rows'], 20)
        self.assertEqual(sum(b['rows'] for b in result['quantile_bins'][1:]), 0)
        self.assertEqual(sum(b['B'] for b in result['quantile_bins']), 10)

    def test_r_changes_priority_but_not_gate(self):
        cold = pd.DataFrame({'user_index': [0, 0, 1]})
        gain = np.zeros((3, 12))
        gain[0, 0], gain[1, 1] = .2, .1
        utility = gain.copy()
        utility[0, 0] *= .1
        result, rows = priority_diagnostic(cold, gain, utility, lambda: None)
        self.assertFalse(result['alternative_matching'])
        self.assertFalse(result['alternative_MAP'])
        self.assertEqual(result['users_with_eligible_edges'], 1)
        self.assertEqual(result['top_overlap']['1']['mean_user_overlap'], 0.)
        self.assertEqual(result['top_overlap']['5']['mean_user_overlap'], 1.)
        self.assertEqual(rows[0]['top5_denominator'], 2)

    def test_no_matching_quota_and_unchanged_positions(self):
        gain = np.eye(12) * .2
        matches = exact_matching(gain, 0.)
        self.assertEqual(len(matches), 12)
        warm = [f'w{i}' for i in range(12)]
        cold = ['c0', 'c1']
        result = apply_admissions(warm, cold, [(0, 1), (1, 10)])
        self.assertEqual(result[1], 'c0')
        self.assertEqual(result[10], 'c1')
        for i in set(range(12)) - {1, 10}:
            self.assertEqual(result[i], warm[i])
        self.assertEqual(exact_matching(np.zeros((2, 12)), 0.), [])

    def test_direct_AP_truth_denominator(self):
        items = [str(i) for i in range(12)]
        self.assertAlmostEqual(direct_ap(items, {'0', '2'}), (1 + 2/3) / 2)
        self.assertEqual(direct_ap(items, {'no-hit'}), 0.)
        with self.assertRaises(ValueError):
            direct_ap(['repeat'] * 12, {'repeat'})

    def test_stable_user_fold_does_not_depend_on_window(self):
        for user in ('abc', 'long-hash-user', '用户', '0001'):
            self.assertEqual(independent_fold(user), fold(user))
            self.assertIn(independent_fold(user), (0, 1))

    @staticmethod
    def verdict_fixture():
        action = {'beneficial_vs_harmful': {'roc_auc': .7, 'pr_auc': .1}}
        windows, old, f_reference = {}, {}, {}
        for i in range(4):
            key = str(i)
            policy = dict(action=deepcopy(action), map12=.025, delta_vs_w0=0.,
                          segments={'warm_21_plus': {'delta_vs_w0': 0.}},
                          admission={'inserted_cold_positives': 1, 'removed_warm_positives': 4})
            windows[key] = {'H_conditional_BH': policy,
                            'survival': {'H': {'surviving_positive_candidates': 2,
                                               'strict_surviving': 1 if i == 0 else 0}}}
            old[key] = {'R': deepcopy(policy), 'survival': {'R': {
                'surviving_positive_candidates': 1 if i == 3 else 0}}}
            f_reference[key] = dict(delta_vs_w0=-.005,
                                    segments={'warm_21_plus': {'delta_vs_w0': -.005}})
        return windows, old, f_reference

    def test_verdict_action_equal_reference_and_survival_eight(self):
        windows, old, reference = self.verdict_fixture()
        result = verdict(windows, old, reference)
        self.assertEqual(result['conditional_bh_action_signal'], 'supported')
        self.assertEqual(result['conditional_bh_survival'], 'supported')
        self.assertEqual(result['conditional_bh_policy_precision'], 'supported')
        self.assertEqual(result['conditional_bh_map_safety'], 'supported')
        self.assertTrue(result['full_history_scaleup_allowed'])
        self.assertEqual(result['selected'], 'W0')

    def test_verdict_survival_mixed_starts_at_two(self):
        windows, old, reference = self.verdict_fixture()
        for row in windows.values():
            row['survival']['H'].update(surviving_positive_candidates=0, strict_surviving=0)
        windows['0']['survival']['H']['surviving_positive_candidates'] = 2
        self.assertEqual(verdict(windows, old, reference)['conditional_bh_survival'], 'mixed')
        windows['0']['survival']['H']['surviving_positive_candidates'] = 1
        result = verdict(windows, old, reference)
        self.assertEqual(result['conditional_bh_survival'], 'rejected')
        self.assertFalse(result['full_history_scaleup_allowed'])

    def test_verdict_precision_zero_denominator(self):
        windows, old, reference = self.verdict_fixture()
        for row in windows.values():
            row['H_conditional_BH']['admission']['removed_warm_positives'] = 0
        result = verdict(windows, old, reference)
        self.assertEqual(result['conditional_bh_policy_precision'], 'supported')
        self.assertEqual(result['ratio_status'], 'infinite')
        self.assertIsNone(result['ratio'])
        for row in windows.values():
            row['H_conditional_BH']['admission']['inserted_cold_positives'] = 0
        result = verdict(windows, old, reference)
        self.assertEqual(result['conditional_bh_policy_precision'], 'rejected')
        self.assertEqual(result['ratio_status'], 'undefined')
        self.assertFalse(result['full_history_scaleup_allowed'])

    def test_verdict_safety_equality_is_supported(self):
        windows, old, reference = self.verdict_fixture()
        for row in windows.values():
            row['H_conditional_BH']['delta_vs_w0'] = -.0005
            row['H_conditional_BH']['segments']['warm_21_plus']['delta_vs_w0'] = -.0005
        self.assertEqual(verdict(windows, old, reference)['conditional_bh_map_safety'], 'supported')
        windows['0']['H_conditional_BH']['delta_vs_w0'] = -.001
        windows['1']['H_conditional_BH']['delta_vs_w0'] = 0.
        self.assertEqual(verdict(windows, old, reference)['conditional_bh_map_safety'], 'supported')
        windows['0']['H_conditional_BH']['delta_vs_w0'] = -.00100001
        result = verdict(windows, old, reference)
        self.assertEqual(result['conditional_bh_map_safety'], 'mixed')
        self.assertFalse(result['full_history_scaleup_allowed'])


if __name__ == '__main__':
    unittest.main()
