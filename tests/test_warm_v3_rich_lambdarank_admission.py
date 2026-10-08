import unittest
import pandas as pd
from hm_recsys.warm_v3_rich_lambdarank_admission import choose_rank_decisions


class RichLambdaRankAdmissionTests(unittest.TestCase):
    def test_uses_within_user_score_difference_and_noop(self):
        pairs=pd.DataFrame({"customer_id":["a","b"],"challenger_article_id":["c1","c2"],"victim_article_id":["v1","v2"],"challenger_target":[1,0],"victim_target":[0,0],"challenger_rank":[13,13],"victim_rank":[12,12],"unit_gain":[.1,.1]})
        scores=pd.DataFrame({"customer_id":["a","a","b","b"],"article_id":["c1","v1","c2","v2"],"ranking_score":[.4,.2,.1,.2]})
        chosen=choose_rank_decisions(pairs,scores)
        self.assertEqual(chosen.customer_id.tolist(),["a"])
        self.assertAlmostEqual(chosen.expected_ordering_gain.iloc[0],.02)


if __name__ == "__main__": unittest.main()
