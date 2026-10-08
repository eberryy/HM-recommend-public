import unittest
from unittest.mock import patch
import duckdb
import pandas as pd
from hm_recsys import warm_v3_joint as joint


class JointExpertTests(unittest.TestCase):
    def test_joint_features_and_closed_two_way_definition(self):
        self.assertEqual(joint.FAMILIES,['bpr_match','sequence'])
        self.assertEqual(len(joint.EXTRA),4)
        self.assertEqual(joint.CANONICAL_PARAMS['feature_count'],88)
        self.assertFalse(joint.CANONICAL_PARAMS['fusion_threeway'])

    def test_two_way_excludes_old_bpr_vote_and_retains_inactive_fallback(self):
        with duckdb.connect() as con:
            con.execute('''CREATE TABLE s AS SELECT * FROM (VALUES
                ('u','a',1,1,1,1,9.0,1.0,999.0),('u','b',2,0,1,1,1.0,9.0,-999.0),
                ('v','a',1,1,1,0,1.0,1.0,1.0),('v','b',2,0,1,0,9.0,9.0,9.0))
                t(customer_id,article_id,candidate_rank,target,truth_count,user_history_events_12w,score_base,score_joint,score_bpr)''')
            table=joint.ranks(con,'s')
            rows=con.execute(f'SELECT customer_id,article_id,fusion_rank FROM {table} ORDER BY customer_id,fusion_rank').fetchall()
            self.assertEqual(rows,[('u','a',1),('u','b',2),('v','a',1),('v','b',2)])
            scores=con.execute('SELECT fusion_score FROM joint_ranked WHERE customer_id=\'u\'').fetchall()
            self.assertAlmostEqual(scores[0][0],1/61+1/62)
            self.assertEqual(scores[0],scores[1])

    def test_full_denominator_not_only_covered_users(self):
        frame=pd.DataFrame({'ap_baseline':[.1,.2],'ap_standalone':[.3,.1],'ap_fusion':[.2,.3]})
        result=joint.summarize(frame,10)
        self.assertAlmostEqual(result['standalone']['population_delta'],.01)
        self.assertAlmostEqual(result['fusion']['population_delta'],.02)

    def test_sequence_missing_cache_never_calls_builder(self):
        engine=object.__new__(joint.JointEngine)
        with patch.object(joint.Path,'is_file',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'no representation fitting'):
                engine.signal_path('sequence','2019-12-25')


if __name__=='__main__':unittest.main()
