"""Bounded P4.2E pure checks; no model fitting or candidate generation."""
import unittest

import numpy as np
import pandas as pd

from hm_recsys.p42e_data import fold_for_user, parity
from hm_recsys.p42e_resources import memory, rss
from hm_recsys.p42e_fit import preprocess_inplace
from hm_recsys.p42_propensity import _design_matrix
from hm_recsys.p42e_stats import monotonic_parity
from scipy.special import expit


class PreflightTest(unittest.TestCase):
    def test_fold_stability(self):
        users=['a','b','customer-001','customer-002']
        self.assertEqual([fold_for_user(u) for u in users],[fold_for_user(u) for u in users])
        self.assertTrue(set(map(fold_for_user,users)) <= {0,1})

    def test_parity_permutation(self):
        f=pd.DataFrame({'customer_id':['a','a'],'rank':[1,2],'article_id':['x','y'],'score':[.1,.2]})
        self.assertTrue(parity(f.iloc[::-1],f,['customer_id','rank','article_id'],['score'])['exact'])

    def test_rank_swap_rejected(self):
        f=pd.DataFrame({'customer_id':['a','a'],'rank':[1,2],'article_id':['x','y']})
        changed=f.copy(); changed['rank']=[2,1]
        with self.assertRaises(AssertionError): parity(changed,f,['customer_id','rank','article_id'])

    def test_score_tolerance(self):
        f=pd.DataFrame({'key':[1,2],'score':[.1,.2]}); g=f.copy(); g['score']+=2e-5
        with self.assertRaises(AssertionError): parity(g,f,['key'],['score'])

    def test_resource_counters(self):
        m=memory()
        self.assertGreater(m.total,0); self.assertGreaterEqual(m.available,0)
        self.assertLessEqual(m.available,m.total); self.assertGreater(rss(),0)

    def test_column_preprocessing_equivalent(self):
        rng=np.random.default_rng(123)
        f=pd.DataFrame({'a':rng.normal(size=1000),'b':rng.normal(size=1000),'c':np.ones(1000),'flag':rng.integers(0,2,1000)})
        f.loc[::7,'a']=np.nan; f.loc[::17,'b']=np.inf
        spec={'numeric':['a','b','c'],'binary':['flag']}
        expected,ep=_design_matrix(f,None,spec)
        actual=np.empty_like(expected); actual[:,:4]=f[['a','b','c','flag']].to_numpy()
        ap=preprocess_inplace(actual,spec)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        for key in ['mean','scale','var']: np.testing.assert_allclose(ap['scaler'][key],ep['scaler'][key],rtol=0,atol=1e-12)
        self.assertEqual(ap['median'],ep['median'])

    def test_calibration_order(self):
        raw=expit(np.array([-20.,-15.,-15.,-8.,0.]))
        cal=expit(-1+.5*np.array([-20.,-15.,-15.,-8.,0.]))
        self.assertEqual(monotonic_parity(raw,cal)['spearman'],1.)

    def test_clipping_tie_collapse_rejected(self):
        with self.assertRaises(AssertionError): monotonic_parity(np.array([1e-8,1e-7,.1]),np.array([1e-6,1e-6,.1]))


if __name__=='__main__': unittest.main()
