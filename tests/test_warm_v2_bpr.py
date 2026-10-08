import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from hm_recsys.warm_v2_bpr import relation,match_scores
from hm_recsys.warm_v2_engine import connection,save_parquet


class BPRTests(unittest.TestCase):
    def test_bias_ablation_never_outer_candidate(self):
        from hm_recsys.warm_v2_lab import confirm
        with patch('hm_recsys.warm_v2_lab.assert_branch'):
            with self.assertRaisesRegex(ValueError,'inner-only'):
                confirm('WV2-502')

    def test_installed_callback_and_bias_shape(self):
        from scipy.sparse import csr_matrix
        from implicit.cpu.bpr import BayesianPersonalizedRanking
        matrix=csr_matrix(np.eye(3,dtype=np.float32))
        rows=[]
        model=BayesianPersonalizedRanking(factors=4,iterations=1,num_threads=1,random_state=1)
        model.fit(matrix,show_progress=False,callback=lambda epoch,seconds,correct,skipped: rows.append((epoch,seconds,correct,skipped)))
        self.assertEqual(len(rows),1)
        self.assertEqual(model.user_factors.shape,(3,5))
        self.assertTrue(np.isfinite(model.item_factors).all())

    def test_binary_pairs_ignore_future_not_raw_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'tx.parquet'
            f=pd.DataFrame({'customer_id':['a','a','b','future'],
                'article_id':['01','01','02','03'],
                't_dat':pd.to_datetime(['2020-01-01','2020-01-01','2020-01-02','2020-01-08'])})
            save_parquet(f,p)
            with connection() as con:
                matrix,users,items,latest=relation(con,p,'2020-01-08')
            self.assertEqual(matrix.nnz,2)
            self.assertEqual(users.customer_id.tolist(),['a','b'])
            self.assertEqual(items.article_id.tolist(),['01','02'])
            self.assertLess(latest,'2020-01-08')
            self.assertEqual(len(f),4)

    def test_match_score_and_missing_identity(self):
        f=pd.DataFrame({'customer_id':['b','a','new'],'article_id':['02','missing','01']})
        scores,missing=match_scores(f,pd.Index(['a','b']),pd.Index(['01','02']),
            np.array([[1,2],[3,4]],np.float32),np.array([[5,6],[7,8]],np.float32))
        self.assertEqual(scores[0],53)
        self.assertTrue(np.isnan(scores[1:]).all())
        np.testing.assert_array_equal(missing,[0,1,1])


if __name__=='__main__':unittest.main()
