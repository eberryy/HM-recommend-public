"""No-fit synthetic tests of the independent R3 evidence verifier."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from hm_recsys.p42_contract import WINDOWS
from hm_recsys.p42_propensity import MODEL_PARAMS
from hm_recsys.p42r_propensity import repaired_design_matrix, _numerical_diagnostics
from hm_recsys.p42r3_propensity import parameter_delta
from hm_recsys.p42r3_verify import (
    INVARIANTS, independent_parameter_delta, independent_qw_training,
    same_preprocessing, verify_solver_sequence, verify,
)
from hm_recsys.p42_verify import _close


class P42R3VerifyTest(unittest.TestCase):
    def attempts(self):
        return [{"window":w,"side":"qW","repair":"R2","status":"converged",
                 "convergence_warnings":[],"solver_result":{"success":True,"status":0,"nit":23,"message":"converged"}}
                for w in WINDOWS]

    def bundle(self):
        index=np.arange(40)
        frame=pd.DataFrame({"user_item_events_28d":index%7,"age":index%19,
                            "flag":index%2,"target":(index%3==0).astype(int)})
        spec={"numeric":["user_item_events_28d","age"],"binary":["flag"]}
        x,prep,cleanup=repaired_design_matrix(frame,spec,repair="R2")
        model={"params":{**MODEL_PARAMS,"max_iter":1200},"classes":[0,1],
               "coefficient":[.2,-.3,.4],"intercept":-.4,"n_iter":[23]}
        self.assertEqual(x.shape[1],3)
        diag=_numerical_diagnostics(x,frame.target.to_numpy(),model,prep)
        audit={"training_rows":len(frame),"fitted_rows":len(frame),"positives":int(frame.target.sum()),
               "base_rate":float(frame.target.mean()),"cleanup":cleanup,"diagnostics":diag,
               "negative_sampling":False,"oversampling":False,"class_weight":None,"sample_weight":None}
        return frame,spec,model,prep,audit

    def test_exact_25_unique_required_invariants(self):
        self.assertEqual(len(INVARIANTS),25)
        self.assertEqual(len(set(INVARIANTS)),25)

    def test_solver_failure_not_overridden_by_small_gradient_and_no_later_work(self):
        rows=self.attempts()
        rows[2].update(status="non_converged",convergence_warnings=["iteration limit"])
        rows[2]["solver_result"].update(success=False,status=1,nit=1200)
        rows[2]["diagnostics"]={"gradient_infinity_norm":1e-20}
        receipt=verify_solver_sequence(rows[:3],{}, {},"qW_solver_boundary_repair_failure")
        self.assertFalse(receipt["all_four_converged"])
        for calibration,admission in (({"window":{}},{}),({}, {"window":{}})):
            with self.assertRaises(AssertionError):
                verify_solver_sequence(rows[:3],calibration,admission,"qW_solver_boundary_repair_failure")
        with self.assertRaises(AssertionError):
            verify_solver_sequence(rows,{}, {},"qW_solver_boundary_repair_failure")

    def test_returned_success_at_cap_is_still_failure_and_complete_success_passes(self):
        rows=self.attempts()
        self.assertTrue(verify_solver_sequence(rows,{"window":{}},{},"engineering_failure")["all_four_converged"])
        rows[-1]["solver_result"]["nit"]=1200
        rows[-1]["status"]="non_converged"
        self.assertFalse(verify_solver_sequence(rows,{}, {},"qW_solver_boundary_repair_failure")["all_four_converged"])

    def test_independent_1200_training_geometry_no_estimator_fit(self):
        frame,spec,model,prep,audit=self.bundle()
        with patch("sklearn.linear_model.LogisticRegression.fit",side_effect=AssertionError("must not fit")):
            receipt=independent_qw_training([frame],prep,audit,model,spec=spec)
        self.assertTrue(receipt["pass"])
        self.assertTrue(receipt["no_estimator_fit"])
        self.assertEqual(receipt["rows"],40)
        bad=copy.deepcopy(model); bad["params"]["max_iter"]=1000
        with self.assertRaises(AssertionError): independent_qw_training([frame],prep,audit,bad,spec=spec)

    def test_independent_full_row_parameter_delta_matches_production_diagnostic(self):
        frame,spec,new,prep,audit=self.bundle()
        old=copy.deepcopy(new); old["params"]=dict(MODEL_PARAMS); old["n_iter"]=[1000]
        new["n_iter"]=[1001]; old["coefficient"][0]-=.001; old["intercept"]+=.002
        x,_,_=repaired_design_matrix(frame,preprocessing=prep)
        old_diag=_numerical_diagnostics(x,frame.target.to_numpy(),old,prep)
        new_diag=_numerical_diagnostics(x,frame.target.to_numpy(),new,prep)
        expected=independent_parameter_delta([frame.iloc[:17],frame.iloc[17:]],old,new,prep,prep)
        actual=parameter_delta(frame,{"model":old,"preprocessing":prep,"diagnostics":old_diag},
                               {"model":new,"preprocessing":prep,"diagnostics":new_diag})
        _close(actual,expected)
        self.assertEqual(expected["additional_iterations_beyond_old_cap"],1)
        self.assertGreater(expected["training_prediction_abs_delta"]["max"],0)

    def test_exact_parity_and_preprocessing_drift_detection(self):
        frame,spec,new,prep,audit=self.bundle()
        old=copy.deepcopy(new); old["params"]=dict(MODEL_PARAMS)
        previous=copy.deepcopy(prep); previous["lineage"]={"old":True}
        self.assertTrue(same_preprocessing(previous,prep))
        delta=independent_parameter_delta([frame],old,new,previous,prep)
        self.assertTrue(delta["exact_training_prediction_parity"])
        self.assertTrue(delta["exact_coefficient_intercept_parity"])
        altered=copy.deepcopy(prep); altered["median"]["age"]+=1
        with self.assertRaises(AssertionError): independent_parameter_delta([frame],old,new,altered,prep)

    def test_wrong_branch_rejected_before_any_receipt_write(self):
        with patch("hm_recsys.p42r3_verify.subprocess.check_output",return_value="warm-v2-autonomous-lab"), \
             patch("hm_recsys.p42r3_verify.write_json") as write:
            with self.assertRaisesRegex(ValueError,"requires main"): verify(".")
            write.assert_not_called()


if __name__=="__main__": unittest.main()
