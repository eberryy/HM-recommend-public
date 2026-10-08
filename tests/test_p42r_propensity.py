import copy
import json
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd
from scipy.special import expit

from hm_recsys.p42_propensity import MODEL_PARAMS
from hm_recsys.p42r_propensity import (
    R2_COUNT_FEATURES, fit_repaired_propensity, predict_repaired_propensity,
    repaired_design_matrix,
)


class P42RPropensityTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "z_signal": np.arange(8, dtype=float),
            "a_signal": np.arange(8, dtype=float),
            "near_signal": [0, 1, 2, 3, 4, 5, 6, 7.01],
            "observed": [np.nan, 1, np.nan, 2, 3, np.nan, 4, 5],
            "empty": [np.nan] * 8, "constant": [5.] * 8,
            "observed_present": [0, 1, 0, 1, 1, 0, 1, 1],
            "target": [0, 1, 0, 0, 1, 0, 1, 1],
        })
        self.spec = {"numeric": ["z_signal", "a_signal", "near_signal", "observed", "empty", "constant"],
                     "binary": ["observed_present"]}

    def test_r1_exact_cleanup_priorities_and_original_order(self):
        matrix, prep, cleanup = repaired_design_matrix(self.frame, self.spec)
        self.assertEqual(cleanup["dims_before"], 13)
        self.assertEqual(cleanup["constant_dropped_count"], 7)
        self.assertEqual(cleanup["exact_duplicate_dropped_count"], 2)
        self.assertEqual(cleanup["dims_after"], 4)
        self.assertEqual(cleanup["kept_columns"], ["a_signal", "near_signal", "observed", "observed_present"])
        mapping = {row["feature"]: row["survivor"] for row in cleanup["exact_duplicate_dropped"]}
        self.assertEqual(mapping["z_signal"], "a_signal")
        self.assertEqual(mapping["observed_available"], "observed_present")
        self.assertEqual(cleanup["high_correlation_columns_dropped"], 0)
        self.assertGreater(np.corrcoef(self.frame.a_signal, self.frame.near_signal)[0, 1], .995)
        np.testing.assert_allclose(matrix[:, :3].mean(axis=0), 0, atol=1e-15)
        np.testing.assert_allclose(matrix[:, :3].var(axis=0), 1, atol=1e-15)
        np.testing.assert_array_equal(matrix[:, 3], self.frame.observed_present)
        self.assertEqual(prep["median"]["observed"], 3)
        self.assertEqual(prep["median"]["empty"], 0)

    def test_cleanup_is_label_free_and_not_refit_on_outer(self):
        matrix, prep, cleanup = repaired_design_matrix(self.frame, self.spec)
        alternate = self.frame.assign(target=1-self.frame.target)
        other, other_prep, other_cleanup = repaired_design_matrix(alternate, self.spec)
        np.testing.assert_array_equal(matrix, other)
        self.assertEqual(prep, other_prep)
        self.assertEqual(cleanup, other_cleanup)
        frozen = copy.deepcopy(prep)
        outer = self.frame.copy()
        outer["constant"] = np.arange(8)
        outer["z_signal"] = 99
        outer["a_signal"] = 1234
        outer["observed"] = np.nan
        replay, replay_prep, replay_cleanup = repaired_design_matrix(outer.drop(columns="target"), preprocessing=prep)
        self.assertEqual(replay_prep, frozen)
        self.assertEqual(replay_cleanup, cleanup)
        self.assertEqual(prep, frozen)
        self.assertEqual(replay.shape[1], 4)
        self.assertGreater(replay[0, 0], 100)
        j = prep["scaled_numeric"].index("observed")
        np.testing.assert_allclose(replay[:, 2], (3-prep["scaler"]["mean"][j])/prep["scaler"]["scale"][j])

    def test_log1p_list_exact_and_pipeline_order(self):
        self.assertEqual(len(R2_COUNT_FEATURES), 19)
        self.assertEqual(len(set(R2_COUNT_FEATURES)), 19)
        expected = {
            "source_count", "item2vec_seed_support", "item2vec_vocab_count",
            "user_history_events_12w", "user_unique_items_12w", "item_events_7d", "item_events_28d",
            "item_events_12w", "item_unique_customers_28d", "user_item_events_12w", "user_item_events_28d",
            "user_product_code_events_28d", "user_product_code_events_12w",
            "user_product_type_events_28d", "user_product_type_events_12w",
            "user_department_events_28d", "user_department_events_12w",
            "user_garment_events_28d", "user_garment_events_12w",
        }
        self.assertEqual(set(R2_COUNT_FEATURES), expected)
        frame = pd.DataFrame({"source_count": [0., 1, np.nan, 9, 99, 999],
                              "rank": [1., 2, 3, 4, 5, 6], "target": [0, 1, 0, 0, 1, 0]})
        spec = {"numeric": ["source_count", "rank"], "binary": []}
        x, prep, _ = repaired_design_matrix(frame, spec, repair="R2")
        imputed = np.array([0, 1, 9, 9, 99, 999])
        transformed = np.log1p(imputed)
        self.assertEqual(prep["median"]["source_count"], 9)
        self.assertEqual(prep["log1p_applied_features"], ["source_count"])
        self.assertAlmostEqual(prep["scaler"]["mean"][0], transformed.mean())
        np.testing.assert_allclose(x[:, 0], (transformed-transformed.mean())/transformed.std())
        np.testing.assert_allclose(x[:, 1], (frame["rank"]-frame["rank"].mean())/frame["rank"].std(ddof=0))
        frozen = copy.deepcopy(prep)
        replay, _, _ = repaired_design_matrix(frame, preprocessing=json.loads(json.dumps(prep)))
        np.testing.assert_allclose(x, replay, atol=1e-15)
        self.assertEqual(prep, frozen)

    def test_r2_negative_observed_counts_fail_even_if_constant_dropped(self):
        frame = self.frame.assign(source_count=-1.)
        spec = {"numeric": self.spec["numeric"]+["source_count"], "binary": self.spec["binary"]}
        with self.assertRaisesRegex(ValueError, "negative finite count.*source_count"):
            repaired_design_matrix(frame, spec, repair="R2")
        # R1 is not authorized to log counts and therefore has no log-domain gate.
        _, _, cleanup = repaired_design_matrix(frame, spec, repair="R1")
        self.assertIn("source_count", [row["feature"] for row in cleanup["constant_dropped"]])
        _, prep, _ = repaired_design_matrix(frame.assign(source_count=0.), spec, repair="R2")
        with self.assertRaisesRegex(ValueError, "negative finite count"):
            repaired_design_matrix(frame, preprocessing=prep)

    def test_fit_replay_full_pool_fixed_parameters_and_finite_diagnostics(self):
        fitted = fit_repaired_propensity(self.frame, self.spec, repair="R1")
        self.assertEqual(fitted["model"]["params"], MODEL_PARAMS)
        self.assertEqual(fitted["audit"]["status"], "converged")
        self.assertTrue(fitted["audit"]["converged"])
        self.assertEqual(fitted["audit"]["fitted_rows"], len(self.frame))
        self.assertEqual(fitted["audit"]["positives"], 4)
        self.assertFalse(fitted["audit"]["negative_sampling"])
        self.assertIsNone(fitted["audit"]["sample_weight"])
        before = copy.deepcopy(fitted)
        q = predict_repaired_propensity(fitted, self.frame)
        replay = json.loads(json.dumps(fitted, allow_nan=False))
        np.testing.assert_array_equal(q, predict_repaired_propensity(replay, self.frame.drop(columns="target")))
        np.testing.assert_array_equal(q, predict_repaired_propensity(replay, self.frame.assign(target=1-self.frame.target)))
        self.assertEqual(fitted, before)
        diag = fitted["diagnostics"]
        self.assertEqual(diag["outer_rows_read"], 0)
        self.assertEqual(diag["optimizer_calls"], 0)
        self.assertGreater(diag["local_regularized_hessian_condition_number"], 0)
        x, _, _ = repaired_design_matrix(self.frame, preprocessing=fitted["preprocessing"])
        beta = np.asarray(fitted["model"]["coefficient"])
        logits = x @ beta + fitted["model"]["intercept"]
        residual = expit(logits)-self.frame.target.to_numpy()
        gradient = np.r_[x.T@residual/len(x)+beta/len(x), residual.mean()]
        self.assertAlmostEqual(diag["gradient_infinity_norm"], np.abs(gradient).max(), places=15)
        weights = q*(1-q)
        augmented = np.column_stack([x, np.ones(len(x))])
        hessian = augmented.T @ (augmented*weights[:, None])/len(x)
        hessian[np.arange(len(beta)), np.arange(len(beta))] += 1/len(x)
        self.assertAlmostEqual(diag["local_regularized_hessian_condition_number"], np.linalg.cond(hessian), places=10)

    def test_no_numeric_survivors_still_preserves_variable_availability(self):
        frame = pd.DataFrame({"x": [0, np.nan, 0, np.nan], "target": [0, 1, 1, 0]})
        fitted = fit_repaired_propensity(frame, {"numeric": ["x"], "binary": []})
        self.assertEqual(fitted["preprocessing"]["columns"], ["x_available"])
        self.assertEqual(fitted["preprocessing"]["scaled_numeric"], [])
        np.testing.assert_allclose(predict_repaired_propensity(fitted, frame), .5)

    def test_log_transform_only_applies_surviving_names_after_logical_cleanup(self):
        # Lexical survivor is not a count: do not secretly log it because its
        # discarded exact duplicate happened to be on the fixed count list.
        frame = pd.DataFrame({"a_numeric": [0., 1, 3, 8], "source_count": [0., 1, 3, 8]})
        spec = {"numeric": ["source_count", "a_numeric"], "binary": []}
        x, prep, cleanup = repaired_design_matrix(frame, spec, repair="R2")
        self.assertEqual(cleanup["kept_columns"], ["a_numeric"])
        self.assertEqual(prep["log1p_applied_features"], [])
        self.assertEqual(prep["log1p_count_features_dropped_by_cleanup"], ["source_count"])
        np.testing.assert_allclose(x[:, 0], (frame.a_numeric-frame.a_numeric.mean())/frame.a_numeric.std(ddof=0))

    def test_errors_do_not_relax_or_retry(self):
        with self.assertRaisesRegex(ValueError, "repair must"):
            repaired_design_matrix(self.frame, self.spec, repair="R3")
        with self.assertRaisesRegex(ValueError, "outside 0/1"):
            repaired_design_matrix(self.frame.assign(observed_present=2), self.spec)
        with self.assertRaisesRegex(ValueError, "needs both classes"):
            fit_repaired_propensity(self.frame.assign(target=0), self.spec)
        with self.assertRaisesRegex(ValueError, "target"):
            fit_repaired_propensity(self.frame, {"numeric": ["target"], "binary": []})
        with self.assertRaisesRegex(ValueError, "no varying features"):
            repaired_design_matrix(pd.DataFrame({"x": [1, 1]}), {"numeric": ["x"], "binary": []})

    def test_convergence_warning_is_retained_without_retry_or_r2(self):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression

        original_fit = LogisticRegression.fit
        calls = []

        def one_synthetic_fit(estimator, x, y):
            calls.append((estimator.get_params(), x.shape))
            result = original_fit(estimator, x, y)
            warnings.warn("synthetic fixed-budget nonconvergence", ConvergenceWarning)
            return result

        with mock.patch.object(LogisticRegression, "fit", one_synthetic_fit):
            fitted = fit_repaired_propensity(self.frame, self.spec, repair="R1")
        self.assertEqual(len(calls), 1)
        self.assertEqual(fitted["audit"]["status"], "non_converged")
        self.assertFalse(fitted["audit"]["converged"])
        self.assertEqual(fitted["preprocessing"]["repair"], "R1")
        self.assertEqual(fitted["preprocessing"]["log1p_applied_features"], [])
        self.assertEqual(fitted["audit"]["convergence_warnings"], ["synthetic fixed-budget nonconvergence"])
        for name in ("solver", "C", "tol", "max_iter", "class_weight"):
            self.assertEqual(calls[0][0][name], MODEL_PARAMS[name])


if __name__ == "__main__":
    unittest.main()
