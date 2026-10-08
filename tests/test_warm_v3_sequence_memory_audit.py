import unittest
import numpy as np
from hm_recsys.warm_v3_sequence_memory_audit import memory_scores,user_auc

class MemoryAuditTest(unittest.TestCase):
    def test_padded_basket_ignored(self):
        v=np.eye(2,dtype=np.float32);b=np.eye(2,dtype=np.float32)
        a,m=memory_scores(v,b,1,np.array([1,1]))
        np.testing.assert_array_equal(a,[1,0]);np.testing.assert_array_equal(m,[1,0])
    def test_permutation_invariant_baskets_with_ages(self):
        v=np.array([[1.,0.]]);b=np.eye(2);ages=np.array([1,80])
        a,x=memory_scores(v,b,2,ages);aa,xx=memory_scores(v,b[::-1],2,ages[::-1])
        np.testing.assert_allclose(a,aa);np.testing.assert_allclose(x,xx)
    def test_auc_ties_and_missing(self):
        self.assertEqual(user_auc([1,0],[.5,.5]),.5)
        self.assertEqual(user_auc([1,0],[1,0]),1.)
        self.assertTrue(np.isnan(user_auc([1,0],[np.nan,0])))

if __name__=='__main__':unittest.main()
