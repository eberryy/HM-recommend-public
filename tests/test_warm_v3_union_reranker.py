import unittest
import numpy as np
import pandas as pd
from hm_recsys.warm_v3_union_reranker import FEATURES,fit,rank_and_ap


class UnionRerankerTests(unittest.TestCase):
    def test_fixed_feature_contract(self):
        self.assertEqual(len(FEATURES),11);self.assertIn('baseline_rank',FEATURES);self.assertIn('fusion_rank',FEATURES)

    def test_inactive_uses_baseline_rank(self):
        rows=[]
        for i in range(1,13):
            row={f:float(i) for f in FEATURES};row.update(customer_id='u',article_id=i,target=int(i==1),truth_count=1,
                user_history_events_12w=0,baseline_rank=i,candidate_rank=i)
            rows.append(row)
        frame=pd.DataFrame(rows);ranked,aps=rank_and_ap(frame,np.arange(12),1)
        self.assertTrue((ranked.final_rank==ranked.baseline_rank).all());self.assertEqual(len(aps),1)


if __name__=='__main__':unittest.main()
