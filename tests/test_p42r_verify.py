import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hm_recsys.p42r_propensity import fit_repaired_propensity, predict_repaired_propensity
from hm_recsys.p42r_verify import (
    INVARIANTS, independent_qw_matrix, independent_qw_training, validate_attempt_sequence, verify,
)


class P42RVerifyTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "source_count": [0., 1, 2, 4, 9, np.nan, 2, 30],
            "z_source_duplicate": [0., 1, 2, 4, 9, np.nan, 2, 30],
            "other": [1., 0, 1, 0, 1, 0, 1, 0],
            "empty": [np.nan]*8,
            "flag": [1, 1, 1, 1, 1, 0, 1, 1],
            "target": [0, 1, 0, 0, 1, 0, 0, 1],
        })
        self.spec = {"numeric": ["source_count", "z_source_duplicate", "other", "empty"], "binary": ["flag"]}

    def _fit(self, repair="R1"):
        fitted = fit_repaired_propensity(self.frame, self.spec, repair=repair)
        audit = {**fitted["audit"], "cleanup": fitted["cleanup"], "diagnostics": fitted["diagnostics"]}
        return fitted, audit

    def test_independent_r1_and_r2_full_train_replay(self):
        for repair in ("R1", "R2"):
            fitted, audit = self._fit(repair)
            frames = [self.frame.iloc[:3], self.frame.iloc[3:]]
            receipt = independent_qw_training(frames, fitted["preprocessing"], audit, fitted["model"], self.spec)
            self.assertTrue(receipt["pass"])
            self.assertEqual(receipt["rows"], 8)
            self.assertEqual(receipt["positives"], 3)
            self.assertEqual(receipt["negative_finite_count_check"], "pass" if repair == "R2" else "not_run")
            x = independent_qw_matrix(self.frame.drop(columns="target"), fitted["preprocessing"])
            from scipy.special import expit
            q = expit(x@np.asarray(fitted["model"]["coefficient"])+fitted["model"]["intercept"])
            np.testing.assert_array_equal(q, predict_repaired_propensity(fitted, self.frame))

    def test_median_or_scaler_tampering_rejected(self):
        fitted, audit = self._fit()
        for kind in ("median", "scaler"):
            prep = copy.deepcopy(fitted["preprocessing"])
            if kind == "median":
                prep["median"]["source_count"] += 1
            else:
                prep["scaler"]["mean"][0] += 1
            with self.assertRaises(AssertionError):
                independent_qw_training([self.frame], prep, audit, fitted["model"], self.spec)

    def test_cleanup_priority_and_false_highcorr_drop_rejected(self):
        fitted, audit = self._fit()
        prep = copy.deepcopy(fitted["preprocessing"])
        row = prep["cleanup"]["exact_duplicate_dropped"][0]
        row["survivor"] = "other"
        bad_audit = {**audit, "cleanup": prep["cleanup"]}
        with self.assertRaises(AssertionError):
            independent_qw_training([self.frame], prep, bad_audit, fitted["model"], self.spec)
        prep = copy.deepcopy(fitted["preprocessing"])
        prep["cleanup"]["high_correlation_columns_dropped"] = 1
        with self.assertRaises(AssertionError):
            independent_qw_training([self.frame], prep, {**audit, "cleanup": prep["cleanup"]}, fitted["model"], self.spec)

    def test_wrong_r2_list_and_negative_evaluation_count_rejected(self):
        fitted, audit = self._fit("R2")
        prep = copy.deepcopy(fitted["preprocessing"])
        prep["r2_count_features"].append("other")
        with self.assertRaises(AssertionError):
            independent_qw_training([self.frame], prep, audit, fitted["model"], self.spec)
        with self.assertRaises(AssertionError):
            independent_qw_matrix(self.frame.assign(source_count=-1), fitted["preprocessing"])

    def test_gradient_tampering_rejected(self):
        fitted, audit = self._fit()
        audit["diagnostics"] = {**audit["diagnostics"], "gradient_infinity_norm": 1.0}
        with self.assertRaises(AssertionError):
            independent_qw_training([self.frame], fitted["preprocessing"], audit, fitted["model"], self.spec)

    @staticmethod
    def _attempt(side="qW", repair="R1", status="non_converged", window="winter_20200122"):
        return {"window": window, "side": side, "repair": repair, "status": status,
                "convergence_warnings": [] if status == "converged" else ["fixed budget exhausted"]}

    def test_legitimate_r1_success_and_r1_r2_failure_are_verifiable(self):
        q_c = self._attempt("qC", "original", "converged")
        r1 = self._attempt(status="converged")
        self.assertTrue(validate_attempt_sequence([q_c, r1], "cold_gain_not_recovered", "R1")["pass"])
        r1 = self._attempt()
        r2 = self._attempt(repair="R2")
        self.assertTrue(validate_attempt_sequence([q_c, r1, r2], "qW_R1_and_R2_convergence_failure", None)["pass"])

    def test_unauthorized_r2_and_duplicate_retry_rejected(self):
        r1 = self._attempt(status="converged")
        r2 = self._attempt(repair="R2", status="converged")
        for attempts in ([r2], [r1, r2], [r1, r1]):
            with self.assertRaises(AssertionError):
                validate_attempt_sequence(attempts, "engineering_failure", "R2")
        with self.assertRaises(AssertionError):
            validate_attempt_sequence([self._attempt("qC", "original", "converged"), r2,
                                       self._attempt(status="non_converged")], "engineering_failure", "R2")
        with self.assertRaises(AssertionError):
            validate_attempt_sequence([r1, self._attempt(repair="R2", window="spring_20200318")], "engineering_failure", "R1")

    def test_absent_evidence_is_failure_not_false_pass(self):
        self.assertEqual(len(INVARIANTS), 26)
        self.assertEqual(len(set(INVARIANTS)), 26)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = verify(root)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["recommendation_model_fits"], 0)
            self.assertEqual(result["summary"]["pass"], 0)
            output = root/"reports/phase4/P4_2R_VERIFICATION.json"
            self.assertTrue(output.exists())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "fail")


if __name__ == "__main__":
    unittest.main()
