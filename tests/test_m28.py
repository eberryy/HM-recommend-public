from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from hm_recsys.m28 import _aggregate_item2vec_candidates, _stable_hash


class M28Tests(unittest.TestCase):
    def test_stable_hash_is_repeatable_and_sensitive(self) -> None:
        self.assertEqual(_stable_hash("0123456789"),_stable_hash("0123456789"))
        self.assertNotEqual(_stable_hash("0123456789"),_stable_hash("0123456788"))
        self.assertGreaterEqual(_stable_hash("x"),0)
        self.assertLess(_stable_hash("x"),2**32)

    def test_candidate_aggregation_is_unique_ranked_and_support_aware(self) -> None:
        seeds=pd.DataFrame({"customer_id":["u","u"],"row_index":[0,1],"seed_rank":[1,2]})
        neighbor_indices=np.array([[2,3],[2,4]],dtype=np.int32)
        neighbor_scores=np.array([[0.8,0.7],[0.9,0.6]],dtype=np.float32)
        article_ids=np.array(["a0","a1","a2","a3","a4"])
        actual=_aggregate_item2vec_candidates(seeds=seeds,neighbor_indices=neighbor_indices,
            neighbor_scores=neighbor_scores,article_ids=article_ids,max_candidates=3)
        self.assertEqual(actual["article_id"].tolist(),["a2","a3","a4"])
        self.assertEqual(actual["item2vec_rank"].tolist(),[1,2,3])
        self.assertEqual(actual["seed_support"].tolist(),[2,1,1])
        self.assertEqual(len(actual[["customer_id","article_id"]].drop_duplicates()),len(actual))
        self.assertAlmostEqual(float(actual.loc[0,"item2vec_score"]),0.9/1.1,places=6)

    def test_candidate_tie_break_uses_article_id(self) -> None:
        seeds=pd.DataFrame({"customer_id":["u"],"row_index":[0],"seed_rank":[1]})
        indices=np.array([[2,1]],dtype=np.int32); scores=np.array([[0.5,0.5]],dtype=np.float32)
        article_ids=np.array(["seed","b","a"])
        actual=_aggregate_item2vec_candidates(seeds=seeds,neighbor_indices=indices,
            neighbor_scores=scores,article_ids=article_ids,max_candidates=2)
        self.assertEqual(actual["article_id"].tolist(),["a","b"])


if __name__=="__main__":
    unittest.main()
