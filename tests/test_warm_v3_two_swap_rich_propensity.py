import unittest
import pandas as pd
from hm_recsys.warm_v3_two_swap_rich_propensity import choose_two_swaps


class TwoSwapTests(unittest.TestCase):
    def test_selects_two_disjoint_pairs(self):
        pairs=pd.DataFrame({"customer_id":["u","u","u"],"challenger_article_id":["c1","c1","c2"],"victim_article_id":["v1","v2","v2"],"challenger_rank":[13,13,14],"victim_rank":[12,11,11],"unit_gain":[.1,.1,.1]})
        scores=pd.DataFrame({"customer_id":["u"]*4,"article_id":["c1","c2","v1","v2"],"ranking_score":[.9,.8,.1,.2]})
        chosen=choose_two_swaps(pairs,scores)
        self.assertEqual(len(chosen),2);self.assertEqual(chosen.challenger_article_id.nunique(),2);self.assertEqual(chosen.victim_article_id.nunique(),2)


if __name__=="__main__":unittest.main()
