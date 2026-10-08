import unittest
import numpy as np
import pandas as pd
from hm_recsys.final_candidate_e2 import histories,relation_batch,ranks_for,rrf,M4COL

class E2Test(unittest.TestCase):
    def test_relation_shape_and_missing(self):
        emb=np.eye(5,dtype=np.float32);cand=np.array([[0,1]]);hist=np.array([[0,2]+[-1]*18])
        days=np.array([[2,10]+[0]*18],np.float32);mask=hist>=0;attr=np.array([1,1,2,2,3])
        z=relation_batch(cand,hist,days,mask,emb,attr,attr,'cpu')
        self.assertEqual(z.shape,(1,2,4,6));self.assertEqual(z.reshape(2,24).shape[1],len(M4COL))
        self.assertEqual(z[0,0,0,2],1);self.assertEqual(z[0,1,0,4],1)
        self.assertTrue(np.isnan(z[0,0,2,2]))

    def test_rrf_is_group_local(self):
        cold=pd.DataFrame({'user_index':[0,0,1,1],'b0_rank':[1,2,1,2]})
        data={'cold':cold};score=np.array([0,1,3,2.])
        self.assertEqual(ranks_for(data,score).tolist(),[2,1,1,2])
        out=rrf(data,score,1)
        self.assertAlmostEqual(out[0],out[1]);self.assertGreater(out[2],out[3])
