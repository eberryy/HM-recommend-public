"""Small deterministic checks for the no-training 2x2 audit."""
import unittest
import duckdb
from hm_recsys.warm_v3_pool_failure import rerank, contributions, decompose


class FailureAuditTests(unittest.TestCase):
    def test_inactive_always_original_rank(self):
        with duckdb.connect() as con:
            con.execute('''CREATE TABLE s AS SELECT * FROM (VALUES
                ('u','a',1,0,1,0,0.0,0.0),('u','b',2,1,1,0,9.0,9.0))
                t(customer_id,article_id,candidate_rank,target,truth_count,user_history_events_12w,score_e0,score_e1)''')
            rerank(con,'SELECT * FROM s','r')
            self.assertEqual(con.execute('SELECT article_id FROM r ORDER BY rf').fetchall(),[('a',),('b',)])

    def test_truth_contributions_reconcile_new_lost_shared(self):
        with duckdb.connect() as con:
            con.execute("CREATE TABLE a AS SELECT * FROM (VALUES ('u','x',1,1,2),('u','y',1,13,2)) t(customer_id,article_id,target,rf,truth_count)")
            con.execute("CREATE TABLE b AS SELECT * FROM (VALUES ('u','x',1,2,2),('u','y',1,1,2)) t(customer_id,article_id,target,rf,truth_count)")
            contributions(con,'a','ha')
            contributions(con,'b','hb')
            rows = decompose(con,'ha','hb',10)
            self.assertAlmostEqual(sum(r['population_delta'] for r in rows),.05)
            self.assertEqual({r['component'] for r in rows},{'new_top12_truth','shared_top12_truth'})
            back = decompose(con,'hb','ha',10)
            self.assertAlmostEqual(sum(r['population_delta'] for r in back),-.05)
            self.assertIn('lost_top12_truth',{r['component'] for r in back})

    def test_filter_recomputes_entire_expert_rank(self):
        with duckdb.connect() as con:
            con.execute('''CREATE TABLE s AS SELECT * FROM (VALUES
                ('u','a',1,0,1,1,4.0,1.0),('u','b',2,1,1,1,1.0,4.0),('u','new',3,0,1,1,5.0,5.0))
                t(customer_id,article_id,candidate_rank,target,truth_count,user_history_events_12w,score_e0,score_e1)''')
            rerank(con,'SELECT * FROM s','expanded')
            rerank(con,"SELECT * FROM s WHERE article_id<>'new'",'old')
            self.assertEqual(con.execute("SELECT r0,r1 FROM old WHERE article_id='a'").fetchone(),(1,2))
            self.assertEqual(con.execute("SELECT r0,r1 FROM expanded WHERE article_id='a'").fetchone(),(2,3))


if __name__=='__main__':
    unittest.main()
