import unittest
import numpy as np
import pandas as pd
from hm_recsys.p42f_core import fold,neutral_keep,sample_edges,prior_residual
from hm_recsys.p42d_stats import single_ap_delta
from hm_recsys.p42_matching import exact_matching
from hm_recsys.p42f_contract import GLOBAL,BASE,USER,PERSONAL,PARAMS

class ContractTests(unittest.TestCase):
    def test_features(self):
        self.assertEqual(len(GLOBAL),len(set(GLOBAL)))
        self.assertFalse(set(BASE)&set(USER+PERSONAL))
        self.assertFalse(set(GLOBAL)&{'qC','qW','b0_score','logit_qC','customer_id','article_id','target'})
        self.assertEqual(PARAMS['n_estimators'],250)
    def test_hash(self):
        self.assertEqual(fold('u'),fold('u'))
        self.assertEqual(neutral_keep('2019-12-25','u','a',1),neutral_keep('2019-12-25','u','a',1))
    def test_sample(self):
        y=np.array([[0.,.1,-.1]*4]); keep,w=sample_edges('2019-12-25',['u'],['a'],y)
        self.assertTrue(keep[y!=0].all()); self.assertTrue((w[y[keep]!=0]==1).all()); self.assertTrue((w[y[keep]==0]==50).all())
    def test_delta_exhaustive(self):
        r=((np.arange(4096)[:,None]>>np.arange(12))&1).astype(float)
        den=np.maximum(r.sum(axis=1)+1,1); ap=lambda v:(v*np.cumsum(v,axis=1)/np.arange(1,13)).sum(axis=1)/np.minimum(den,12)
        baseline=ap(r)
        for label in [0,1]:
            got=single_ap_delta(r,np.full(4096,label),den)
            for j in range(12):
                v=r.copy(); v[:,j]=label
                np.testing.assert_allclose(got[:,j],ap(v)-baseline,atol=1e-15)
    def test_prior(self):
        h=pd.DataFrame({'customer_id':['u','u','u'],'cutoff':['2019-12-25','2020-01-22','2020-02-19'],'residual':[.3,9.,100.]})
        n,b=prior_residual(h,['u','new'],'2020-01-22')
        np.testing.assert_array_equal(n,[1,0]); np.testing.assert_allclose(b,[.1,0])
    def test_no_cap(self):
        self.assertEqual(len(exact_matching(np.eye(12),0)),12)
        self.assertEqual(exact_matching(np.zeros((50,12)),0),[])
    def test_matching_not_greedy(self):
        self.assertEqual(set(exact_matching(np.array([[5.,4.],[4.,-1.]]),0)),{(0,1),(1,0)})

if __name__=='__main__': unittest.main()
