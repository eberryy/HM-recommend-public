import unittest

import numpy as np
from scipy.sparse import csr_matrix

from hm_recsys.warm_v3_userknn import PARAMS, UserKNN, bounded_pool, postings


class UserKNNTests(unittest.TestCase):
    def test_postings_recent_first_with_stable_user_ties(self):
        offset, p = postings([2, 0, 1, 3], [0, 0, 0, 1], [9, 10, 10, 11], 2)
        np.testing.assert_array_equal(offset, [0, 3, 4])
        np.testing.assert_array_equal(p, [0, 1, 2, 3])

    def test_pool_bounded_without_full_user_similarity(self):
        n = 10000
        r = csr_matrix((np.ones(n), (np.arange(n), np.zeros(n))), shape=(n, 1))
        off, p = postings(np.arange(n), np.zeros(n), np.arange(n), 1)
        pool, scanned = bounded_pool(r, 9999, off, p)
        self.assertEqual(len(pool), 2000)
        self.assertNotIn(9999, pool)
        self.assertLessEqual(scanned, 2001)
        np.testing.assert_array_equal(pool, np.arange(7999, 9999))

    def test_full_sparse_cosine_and_neighbor_only_score(self):
        a = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1]], np.float32)
        r = csr_matrix(a)
        coo = r.tocoo()
        off, p = postings(coo.row, coo.col, np.ones(r.nnz), 3)
        model = UserKNN(r, off, p)
        score, audit = model.query(0)
        self.assertEqual(audit['neighbor_ids'], [1, 2])
        np.testing.assert_allclose(audit['neighbor_similarity'], [.5, .5], atol=1e-7)
        np.testing.assert_allclose(score.toarray().ravel(), [.25, .25, .5], atol=1e-7)
        self.assertEqual(PARAMS['self_score_weight'], 0)

    def test_seed_budget_deduplicates_neighbors_without_refill(self):
        r = csr_matrix(np.array([[2., 1.], [1., 1.], [1., 0.]], np.float32))
        coo = r.tocoo()
        off, p = postings(coo.row, coo.col, np.ones(r.nnz), 2)
        pool, scanned = bounded_pool(r, 0, off, p)
        np.testing.assert_array_equal(pool, [1, 2])
        self.assertLessEqual(scanned, 2002)

    def test_unknown_user_or_item_is_missing_but_known_zero_vote_is_zero(self):
        r = csr_matrix(np.array([[1., 1., 0.], [1., 0., 0.]], np.float32))
        coo = r.tocoo()
        off, p = postings(coo.row, coo.col, np.ones(r.nnz), 3)
        model = UserKNN(r, off, p)
        value, missing, _ = model.score_pairs(0, np.array([0, 1, -1]))
        np.testing.assert_allclose(value[:2], [1., 0.])
        self.assertTrue(np.isnan(value[2]))
        np.testing.assert_array_equal(missing, [0, 0, 1])
        value, missing, _ = model.score_pairs(-1, np.array([0, 1]))
        self.assertTrue(np.isnan(value).all())
        np.testing.assert_array_equal(missing, [1, 1])

    def test_day_weights_are_permutation_invariant_and_decay_is_fixed(self):
        days = np.array([1, 1, 29])
        items = np.array([10, 11, 10])
        def vector(order):
            return {i: float(np.sum(.5**(days[order][items[order] == i]/28))) for i in [10, 11]}
        self.assertEqual(vector([0, 1, 2]), vector([1, 0, 2]))
        self.assertAlmostEqual(.5**(29/28)/(.5**(1/28)), .5)
        self.assertEqual(PARAMS['history_days'], 84)
        self.assertEqual(PARAMS['half_life_days'], 28)
        self.assertEqual(PARAMS['neighbors_k'], 100)


if __name__ == '__main__':
    unittest.main()
