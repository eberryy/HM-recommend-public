import unittest
import numpy as np
import pandas as pd
from hm_recsys.final_relation_pilot import history_features,append_features,REL

class RelationTest(unittest.TestCase):
    def test_history_firewall_and_duplicates(self):
        keys=pd.DataFrame({'customer_id':['u','v'],'article_id':['a','b'],'row_id':[0,1]})
        cat=pd.DataFrame({'article_id':['a','b'],'product_code':[1,2],'product_type_no':[1,1],'colour_group_code':[1,2]})
        h=pd.DataFrame({'customer_id':['u']*4,'article_id':['a','a','b','b'],
            't_dat':['2020-01-30','2020-01-30','2020-01-10','2020-02-01']})
        x=history_features(keys,h,cat,'2020-02-01')
        np.testing.assert_allclose(x[0,:3],[2/3,2/3,2])
        np.testing.assert_allclose(x[0,3:6],[1,1,2])
        self.assertTrue(np.isnan(x[1,2]));self.assertEqual(x[1,0],0)
        future=pd.concat([h,h.iloc[[-1]]],ignore_index=True)
        np.testing.assert_allclose(x,history_features(keys,future,cat,'2020-02-01'),equal_nan=True)

    def test_two_sided_alignment(self):
        x=np.zeros((24,70),np.float32);cold=np.ones((2,len(REL)),np.float32)
        warm=np.full((2,12,len(REL)),3,np.float32)
        warm[0]=2
        z=append_features(x,cold,warm,[1,0])
        self.assertEqual(z.shape,(24,97));np.testing.assert_array_equal(z[:12,-9:],-2)
        np.testing.assert_array_equal(z[12:,-9:],-1)
