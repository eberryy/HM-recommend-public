import unittest
import duckdb
import pandas as pd
from hm_recsys.warm_v3_prior_fusion import create_ranks


class PriorFusionTests(unittest.TestCase):
    def test_complete_rank_permutation_and_inactive_fallback(self):
        frame=pd.DataFrame([{'customer_id':u,'article_id':i,'candidate_rank':i,'target':0,'truth_count':1,
            'user_history_events_12w':active,'baseline_rank':6-i if active else 6-i}
            for u,active in [('active',2),('inactive',0)] for i in range(1,6)])
        with duckdb.connect() as con:
            con.register('base',frame);table=create_ranks(con)
            rows=con.execute(f'SELECT customer_id,candidate_rank,final_rank FROM {table}').fetchdf()
        for _,g in rows.groupby('customer_id'):self.assertEqual(sorted(g.final_rank),[1,2,3,4,5])
        inactive=rows[rows.customer_id=='inactive']
        self.assertTrue((inactive.candidate_rank==inactive.final_rank).all())


if __name__=='__main__':unittest.main()
