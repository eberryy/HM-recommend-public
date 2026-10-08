import copy
import unittest

import numpy as np
import pandas as pd

from hm_recsys.p42_verify import _assignment, _metrics_from_lists, _replay, _verify_preprocessing


class P42VerifyTest(unittest.TestCase):
    def test_assignment_rejects_equality_allows_multiple_and_is_global_optimum(self):
        matrix = np.full((2, 12), -10.0)
        matrix[0, :2] = [10, 9]
        matrix[1, :2] = [8, 0]
        self.assertEqual(_assignment(matrix, 0), [(1, 0), (0, 1)])
        matrix[:] = 0
        self.assertEqual(_assignment(matrix, 0), [])
        matrix = np.full((12, 12), -1.0)
        np.fill_diagonal(matrix, 1.0)
        self.assertEqual(len(_assignment(matrix, 0)), 12)

    def test_independent_preprocessing_and_coefficient_replay_no_target(self):
        frame = pd.DataFrame({"x": [1., 3., np.nan, 5.], "flag": [0., 1., np.nan, 0.], "target": [0, 1, 0, 1]})
        spec = {"numeric": ["x"], "binary": ["flag"]}
        prep = {**spec, "columns": ["x", "flag", "x_available"], "training_rows": 4,
            "median": {"x": 3.0}, "missing_training_rows": {"x": 1}, "all_missing_numeric": [],
            "scaler": {"mean": [3.0], "var": [2.0], "scale": [np.sqrt(2)], "n_samples_seen": 4}}
        audit = {"training_rows": 4, "fitted_rows": 4, "positives": 2, "base_rate": .5,
            "negative_sampling": False, "oversampling": False, "class_weight": None, "sample_weight": None,
            "status": "converged", "convergence_warnings": []}
        _verify_preprocessing([frame], prep, audit, spec)
        stopped = {**audit, "status": "non_converged", "convergence_warnings": ["iteration limit"]}
        with self.assertRaises(AssertionError):
            _verify_preprocessing([frame], prep, stopped, spec)
        # Failure-closure verification can validate data/preprocessing while
        # retaining the failed convergence gate; no estimator is instantiated.
        _verify_preprocessing([frame], prep, stopped, spec, require_convergence=False)
        corrupted = copy.deepcopy(prep)
        corrupted["median"]["x"] = 10
        with self.assertRaises(AssertionError):
            _verify_preprocessing([frame], corrupted, audit, spec)
        model = {"classes": [0, 1], "coefficient": [1., 2., -.5], "intercept": .1}
        np.testing.assert_array_equal(_replay(frame, model, prep), _replay(frame.drop(columns="target"), model, prep))

    def test_segment_complete_truth_user_denominator_and_original_positions(self):
        users = np.array(["a", "b"])
        lists = np.array([[str(i) for i in range(12)], [str(i) for i in range(12)]])
        truth = pd.DataFrame({"customer_id": ["a", "a", "b"], "article_id": ["0", "1", "11"],
                              "interaction_count_before_cutoff": [21, 0, 0]})
        result, _, _ = _metrics_from_lists(users, lists, truth, lists)
        self.assertEqual(result["segments"]["strict_cold"]["truth_users"], 2)
        self.assertEqual(result["segments"]["strict_cold"]["truth_pairs"], 2)
        self.assertAlmostEqual(result["segments"]["strict_cold"]["map12"], (.5+1/12)/2)
        self.assertEqual(result["segments"]["warm_21_plus"]["truth_users"], 1)
        self.assertEqual(result["segments"]["sparse1_5"]["map12"], None)


if __name__ == "__main__":
    unittest.main()
