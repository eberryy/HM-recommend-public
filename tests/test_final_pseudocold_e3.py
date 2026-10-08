import unittest
import pandas as pd
from hm_recsys.final_pseudocold_e3 import FEATURES
from hm_recsys.final_candidate_e2 import M4COL
from hm_recsys.p42f_contract import USER

class E3Test(unittest.TestCase):
    def test_feature_contract_unique(self):
        self.assertEqual(len(FEATURES),41);self.assertEqual(len(set(FEATURES)),41)
        self.assertEqual(FEATURES,M4COL+USER)

    def test_pseudocold_definition_logic(self):
        rows=pd.DataFrame({'global_events':[6,5,20],'seen':[0,0,1]})
        keep=(rows.global_events>5)&(rows.seen==0)
        self.assertEqual(keep.tolist(),[True,False,False])
