import copy
import json
import unittest

import numpy as np
import pandas as pd

from hm_recsys.p42_propensity import (
    EPSILON, MODEL_PARAMS, _design_matrix, clip_propensity,
    fit_propensity, predict_propensity, propensity_calibration,
)


class P42PropensityTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "x": [0, 1, np.nan, 3, 4, 5, 6, 7],
            "empty": [np.nan] * 8,
            "flag": [0, 1, np.nan, 0, 1, 0, 1, 0],
            "target": [0, 0, 0, 1, 0, 1, 1, 1],
        })
        self.spec = {"numeric": ["x", "empty"], "binary": ["flag"]}

    def test_train_only_median_scaler_and_binary_availability(self):
        matrix, prep = _design_matrix(self.frame, None, self.spec)
        self.assertEqual(prep["median"], {"x": 4.0, "empty": 0.0})
        self.assertEqual(prep["all_missing_numeric"], ["empty"])
        self.assertEqual(prep["scaler"]["n_samples_seen"], 8)
        np.testing.assert_allclose(matrix[:, 0].mean(), 0, atol=1e-15)
        np.testing.assert_allclose(matrix[:, 0].var(), 1, atol=1e-15)
        np.testing.assert_array_equal(matrix[:, 2], [0, 1, 0, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(matrix[:, 3], [1, 1, 0, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(matrix[:, 4], np.zeros(8))
        frozen = copy.deepcopy(prep)
        outer = self.frame.copy()
        outer["x"] = [np.nan, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
        transformed, _ = _design_matrix(outer, prep, None)
        self.assertEqual(prep, frozen)
        self.assertAlmostEqual(transformed[0, 0],
                               (4.0-prep["scaler"]["mean"][0])/prep["scaler"]["scale"][0])
        self.assertGreater(transformed[1, 0], 1000)

    def test_full_pool_fixed_fit_serialized_prediction_and_no_label_dependency(self):
        fitted = fit_propensity(self.frame, self.spec)
        self.assertEqual(fitted["audit"]["fitted_rows"], len(self.frame))
        self.assertEqual(fitted["audit"]["positives"], 4)
        self.assertFalse(fitted["audit"]["negative_sampling"])
        self.assertFalse(fitted["audit"]["oversampling"])
        self.assertIsNone(fitted["audit"]["class_weight"])
        self.assertEqual(fitted["model"]["params"], MODEL_PARAMS)
        self.assertEqual(fitted["model"]["params"]["solver"], "lbfgs")
        self.assertEqual(fitted["model"]["params"]["C"], 1.0)
        self.assertEqual(fitted["audit"]["status"], "converged")
        before = copy.deepcopy(fitted)
        q = predict_propensity(fitted, self.frame)
        replay = json.loads(json.dumps(fitted, allow_nan=False))
        np.testing.assert_array_equal(q, predict_propensity(replay, self.frame.drop(columns="target")))
        changed = self.frame.copy()
        changed["target"] = 1-changed.target
        np.testing.assert_array_equal(q, predict_propensity(fitted, changed))
        self.assertEqual(fitted, before)
        self.assertGreater(q[-1], q[0])

    def test_separate_scalers_and_model_data_insufficiency(self):
        first = fit_propensity(self.frame, self.spec)
        alternate = self.frame.copy()
        alternate["x"] = alternate.x + 100
        second = fit_propensity(alternate, self.spec)
        self.assertNotEqual(first["preprocessing"]["median"], second["preprocessing"]["median"])
        self.assertAlmostEqual(second["preprocessing"]["scaler"]["mean"][0] -
                               first["preprocessing"]["scaler"]["mean"][0], 100)
        with self.assertRaisesRegex(ValueError, "insufficiency"):
            fit_propensity(self.frame.assign(target=0), self.spec)
        with self.assertRaisesRegex(ValueError, "target"):
            fit_propensity(self.frame, {"numeric": ["target"], "binary": []})

    def test_invalid_binary_feature_and_name_collision_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside 0/1"):
            _design_matrix(self.frame.assign(flag=2), None, self.spec)
        with self.assertRaisesRegex(ValueError, "unique"):
            _design_matrix(self.frame, None, {"numeric": ["x"], "binary": ["x_available"]})

    def test_clipping_exact_epsilon_and_invalid_probabilities(self):
        self.assertEqual(EPSILON, 1e-6)
        np.testing.assert_array_equal(clip_propensity([0, EPSILON, .5, 1-EPSILON, 1]),
                                      [EPSILON, EPSILON, .5, 1-EPSILON, 1-EPSILON])
        for invalid in ([np.nan], [np.inf], [-.1], [1.1]):
            with self.assertRaises(ValueError):
                clip_propensity(invalid)

    def test_calibration_metrics_bins_and_tie_membership_are_label_free(self):
        y = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        q = np.array([.1]*4+[.2]*4+[.5]*5)
        audit = propensity_calibration(y, q)
        self.assertEqual(audit["rows"], len(y))
        self.assertEqual(audit["positives"], y.sum())
        self.assertAlmostEqual(audit["brier_score"], np.mean((q-y)**2))
        self.assertEqual([b["rows"] for b in audit["calibration_bins"]], [2, 2, 2, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(sum(b["positives"] for b in audit["calibration_bins"]), y.sum())
        second = propensity_calibration(1-y, q)
        for a, b in zip(audit["calibration_bins"], second["calibration_bins"]):
            for key in ("rows", "min_probability", "max_probability", "mean_probability"):
                self.assertEqual(a[key], b[key])
        self.assertTrue(audit["calibration_line"]["diagnostic_only"])
        json.dumps(audit, allow_nan=False)

    def test_calibration_single_class_constant_empty_and_clipped_values(self):
        one = propensity_calibration([0, 0], [.1, .2])
        self.assertIsNone(one["roc_auc"])
        self.assertIsNone(one["pr_auc"])
        self.assertIsNone(one["calibration_slope"])
        tied = propensity_calibration([0, 1], [.2, .2])
        self.assertEqual(tied["roc_auc"], .5)
        self.assertEqual(tied["pr_auc"], .5)
        self.assertIsNone(tied["calibration_slope"])
        self.assertIn("constant_prediction", tied["warnings"])
        empty = propensity_calibration([], [])
        self.assertEqual(empty["rows"], 0)
        self.assertEqual(len(empty["calibration_bins"]), 10)
        self.assertIsNone(empty["ece"])
        clipped = propensity_calibration([0, 1], [0, 1])
        self.assertEqual(clipped["clipped_low_rows"], 1)
        self.assertEqual(clipped["clipped_high_rows"], 1)
        self.assertAlmostEqual(clipped["brier_score"], 1e-12, places=20)
        self.assertIsNone(clipped["calibration_slope"])
        self.assertEqual(clipped["calibration_line"]["status"], "undefined_separation")

    def test_calibration_line_recovers_perfect_groupwise_calibration(self):
        levels = [.1, .2, .5, .8, .9]
        probability = np.repeat(levels, 1000)
        labels = np.concatenate([np.concatenate([np.ones(int(p*1000)), np.zeros(1000-int(p*1000))])
                                 for p in levels])
        audit = propensity_calibration(labels, probability)
        self.assertTrue(audit["calibration_line"]["success"])
        self.assertAlmostEqual(audit["calibration_intercept"], 0, places=7)
        self.assertAlmostEqual(audit["calibration_slope"], 1, places=7)
        self.assertAlmostEqual(audit["predicted_to_observed_rate_ratio"], 1, places=15)


if __name__ == "__main__":
    unittest.main()
