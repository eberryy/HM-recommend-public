import copy
import json
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult

from hm_recsys.p42_propensity import MODEL_PARAMS
from hm_recsys.p42r_propensity import fit_repaired_propensity, repaired_design_matrix
from hm_recsys.p42r3_propensity import (
    QW_MODEL_PARAMS, capture_solver_result, fit_boundary_propensity, parameter_delta,
)


class P42R3PropensityTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "source_count": [0., 1, np.nan, 2, 5, 20, 7, 10, 3, 8],
            "other": [4., 1, 3, 0, 2, 8, 3, 4, 5, 2],
            "flag": [0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
            "target": [0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
        })
        self.spec = {"numeric": ["source_count", "other"], "binary": ["flag"]}

    def test_only_local_qw_budget_changes(self):
        before = copy.deepcopy(MODEL_PARAMS)
        fitted = fit_boundary_propensity(self.frame, self.spec)
        self.assertEqual(MODEL_PARAMS, before)
        self.assertEqual(MODEL_PARAMS["max_iter"], 1000)
        self.assertEqual(QW_MODEL_PARAMS, {**MODEL_PARAMS, "max_iter": 1200})
        self.assertEqual(fitted["model"]["params"], QW_MODEL_PARAMS)
        self.assertEqual(fitted["audit"]["repair"], "R2")
        self.assertEqual(fitted["audit"]["training_rows"], len(self.frame))
        self.assertEqual(fitted["audit"]["convergence_warning_count"], 0)
        self.assertEqual(fitted["audit"]["status"], "converged")
        json.dumps(fitted, allow_nan=False)

    def test_actual_solver_result_and_uninstrumented_same_budget_parity(self):
        from sklearn.linear_model import LogisticRegression
        from threadpoolctl import threadpool_limits
        from hm_recsys.p42_propensity import THREAD_LIMIT

        fitted = fit_boundary_propensity(self.frame, self.spec)
        observed = fitted["audit"]["solver_result"]
        self.assertTrue(observed["success"])
        self.assertEqual(observed["status"], 0)
        self.assertIn("CONVERGENCE", observed["message"])
        self.assertEqual(observed["nit"], fitted["model"]["n_iter"][0])
        self.assertLess(observed["nit"], 1200)
        self.assertGreaterEqual(observed["nfev"], observed["nit"])
        self.assertFalse(observed["observer_changes_solver_options"])
        self.assertFalse(observed["observer_changes_result"])
        self.assertTrue(observed["original_check_called_unchanged"])
        self.assertTrue(observed["returned_parameters_match_estimator"])
        with threadpool_limits(limits=THREAD_LIMIT):
            x, _, _ = repaired_design_matrix(self.frame, self.spec, repair="R2")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                plain = LogisticRegression(**QW_MODEL_PARAMS).fit(x, self.frame.target)
        np.testing.assert_array_equal(plain.coef_[0], fitted["model"]["coefficient"])
        np.testing.assert_array_equal(plain.intercept_, [fitted["model"]["intercept"]])
        np.testing.assert_array_equal(plain.n_iter_, fitted["model"]["n_iter"])

    def test_observer_returns_original_check_result_and_never_mutates_result(self):
        import sklearn.linear_model._logistic as logistic

        original = logistic._check_optimize_result
        result = OptimizeResult(success=True, status=0, message="fixture success", nit=9,
            nfev=10, njev=10, fun=.4, jac=np.array([1e-8, -1e-9]), x=np.array([.2, -.4]))
        frozen = copy.deepcopy(result)
        with capture_solver_result() as records:
            returned = logistic._check_optimize_result("lbfgs", result, max_iter=1200)
        self.assertIs(logistic._check_optimize_result, original)
        self.assertEqual(returned, 9)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["message"], "fixture success")
        self.assertEqual(records[0]["jac_infinity_norm"], 1e-8)
        for key, value in frozen.items():
            np.testing.assert_array_equal(result[key], value)

    def test_observer_restores_original_after_exception(self):
        import sklearn.linear_model._logistic as logistic

        original = logistic._check_optimize_result
        with self.assertRaisesRegex(RuntimeError, "fixture abort"):
            with capture_solver_result():
                raise RuntimeError("fixture abort")
        self.assertIs(logistic._check_optimize_result, original)

    def test_actual_non_success_not_overridden_by_small_final_gradient(self):
        import sklearn.linear_model._logistic as logistic

        original_minimize = logistic.optimize.minimize

        def fixture_boundary(*args, **kwargs):
            result = original_minimize(*args, **kwargs)
            self.assertLess(np.abs(result.jac).max(), MODEL_PARAMS["tol"])
            result.success = False
            result.status = 1
            result.message = "fixture iteration boundary"
            result.nit = 1200
            return result

        with mock.patch.object(logistic.optimize, "minimize", fixture_boundary):
            fitted = fit_boundary_propensity(self.frame, self.spec)
        self.assertEqual(fitted["audit"]["status"], "non_converged")
        self.assertFalse(fitted["audit"]["solver_result"]["success"])
        self.assertEqual(fitted["audit"]["solver_result"]["message"], "fixture iteration boundary")
        self.assertEqual(fitted["audit"]["convergence_warning_count"], 1)
        self.assertLess(fitted["diagnostics"]["gradient_infinity_norm"], MODEL_PARAMS["tol"])

    def test_reaching_1200_fails_even_with_success_fixture(self):
        import sklearn.linear_model._logistic as logistic

        original_minimize = logistic.optimize.minimize

        def fixture_cap(*args, **kwargs):
            result = original_minimize(*args, **kwargs)
            result.nit = 1200
            return result

        with mock.patch.object(logistic.optimize, "minimize", fixture_cap):
            fitted = fit_boundary_propensity(self.frame, self.spec)
        self.assertTrue(fitted["audit"]["solver_result"]["success"])
        self.assertEqual(fitted["audit"]["convergence_warning_count"], 0)
        self.assertFalse(fitted["audit"]["converged"])
        self.assertEqual(fitted["audit"]["status"], "non_converged")

    def test_1000_vs1200_preprocessing_and_early_solution_parity(self):
        old = fit_repaired_propensity(self.frame, self.spec, repair="R2")
        new = fit_boundary_propensity(self.frame, self.spec)
        old["preprocessing"]["lineage"] = {"stage": "old"}
        new["preprocessing"]["lineage"] = {"stage": "new"}
        delta = parameter_delta(self.frame, old, new)
        self.assertTrue(delta["preprocessing_exact_except_lineage"])
        self.assertTrue(delta["exact_coefficient_intercept_parity"])
        self.assertTrue(delta["exact_training_prediction_parity"])
        self.assertEqual(delta["coefficient_delta_l2"], 0)
        self.assertEqual(delta["intercept_delta"], 0)
        self.assertEqual(delta["training_log_loss_delta"], 0)
        self.assertEqual(delta["training_prediction_abs_delta"]["max"], 0)
        self.assertEqual(delta["additional_iterations_beyond_old_cap"], 0)
        self.assertEqual(delta["new_optimizer_calls"], 0)

    def test_parameter_delta_rejects_preprocessing_budget_and_row_drift(self):
        old = fit_repaired_propensity(self.frame, self.spec, repair="R2")
        new = fit_boundary_propensity(self.frame, self.spec)
        broken = copy.deepcopy(new)
        broken["preprocessing"]["median"]["source_count"] += 1
        with self.assertRaisesRegex(ValueError, "preprocessing drift"):
            parameter_delta(self.frame, old, broken)
        broken = copy.deepcopy(new)
        broken["model"]["params"]["C"] = 2
        with self.assertRaisesRegex(ValueError, "exact old1000/new1200"):
            parameter_delta(self.frame, old, broken)
        with self.assertRaisesRegex(ValueError, "complete original training rows"):
            parameter_delta(self.frame.iloc[:3], old, new)

    def test_fixed_feature_validation_rejects_negative_count_and_label_feature(self):
        with self.assertRaisesRegex(ValueError, "negative finite count"):
            fit_boundary_propensity(self.frame.assign(source_count=-1), self.spec)
        with self.assertRaisesRegex(ValueError, "target"):
            fit_boundary_propensity(self.frame, {"numeric": ["target"], "binary": []})


if __name__ == "__main__":
    unittest.main()
