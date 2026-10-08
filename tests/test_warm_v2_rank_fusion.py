import unittest
import pandas as pd
from hm_recsys.warm_v2_engine import connection
from hm_recsys.warm_v2_rank_fusion import ap_table,rank_sql,final_rank_sql


class RankFusionTests(unittest.TestCase):
    def test_ties_and_inactive_fallback(self):
        frame=pd.DataFrame({'customer_id':['a']*3+['b']*3,
            'article_id':['03','02','01']*2,'candidate_rank':[3,2,1]*2,
            'score':[9.,9.,0.]*2,'user_history_events_12w':[1]*3+[0]*3})
        with connection() as con:
            con.register('f',frame)
            ranks=con.execute(f'SELECT *,{rank_sql("score")} model_rank,{final_rank_sql()} final_rank FROM f ORDER BY customer_id,final_rank').fetchdf()
        self.assertEqual(ranks[ranks.customer_id=='a'].article_id.tolist(),['02','03','01'])
        self.assertEqual(ranks[ranks.customer_id=='b'].article_id.tolist(),['01','02','03'])

    def test_complete_truth_denominator_not_retrieved_positives(self):
        frame=pd.DataFrame({'customer_id':['a']*13+['b']*13,
            'r':list(range(1,14))*2,'target':[0,1]+[0]*10+[1]+[0]*13,
            'truth_count':[4]*13+[2]*13})
        with connection() as con:
            con.register('f',frame)
            scores=ap_table(con,'f','r')
        self.assertEqual(scores.ap.tolist(),[.125,0.])
        self.assertEqual(scores.ap.mean(),.0625)

    def test_equal_rrf_symmetric_and_not_raw_score_scale(self):
        frame=pd.DataFrame({'customer_id':['a']*3,'article_id':['01','02','03'],
            'candidate_rank':[1,2,3],'base_rank':[1,3,2],'bpr_rank':[3,1,2]})
        with connection() as con:
            con.register('f',frame)
            con.execute('CREATE TEMP TABLE x AS SELECT *,1.0/(60+base_rank)+1.0/(60+bpr_rank) score FROM f')
            scores=con.execute(f'SELECT *,{rank_sql("score")} r FROM x ORDER BY r').fetchdf()
        self.assertEqual(scores.score[0],scores.score[1])
        self.assertEqual(scores.article_id.tolist(),['01','02','03'])


if __name__=='__main__':unittest.main()
