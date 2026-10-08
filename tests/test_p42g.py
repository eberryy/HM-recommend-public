"""Bounded synthetic arithmetic checks; no model fitting."""
import unittest
import numpy as np
from hm_recsys.p42g_core import classes,utilities,tail,calibration
from hm_recsys.p42_matching import exact_matching

class RiskTests(unittest.TestCase):
    def test_exact_sign(self):
        np.testing.assert_array_equal(classes([1e-300,0,-1e-300,-0.]),[0,1,2,1])

    def test_formula_clip(self):
        ur,uc,mb,mh=utilities(np.array([[.2,.7,.1],[.1,.8,.1]]),[-1,2],[.5,-2])
        np.testing.assert_array_equal(mb,[0,1]);np.testing.assert_array_equal(mh,[.5,0])
        np.testing.assert_allclose(ur,[-.05,.1]);np.testing.assert_allclose(uc,[.1,0])

    def test_bad_probability(self):
        with self.assertRaises(AssertionError):utilities([[.2,.1,.1]],[1],[1])
        with self.assertRaises(ValueError):utilities([[.2,np.nan,.1]],[1],[1])

    def test_zero_gate(self):
        t=tail(np.array([-1,0,1]),np.array([1,-1,0]))
        self.assertEqual(t['eligible_edges'],1);self.assertEqual(t['eligible_N'],1)

    def test_no_cap(self):
        u=np.eye(12)*.2;self.assertEqual(len(exact_matching(u,0)),12)
        self.assertEqual(exact_matching(np.zeros((2,12)),0),[])

    def test_probability_metrics(self):
        p=np.array([[1,0,0],[0,1,0],[0,0,1]],float)
        a=calibration(p,np.array([1,0,-1]));self.assertEqual(a['multiclass_logloss'],0)
        self.assertEqual(a['multiclass_Brier'],0)
        self.assertEqual(sum(b['rows'] for b in a['reliability']['p_B']),3)

if __name__=='__main__':unittest.main()
