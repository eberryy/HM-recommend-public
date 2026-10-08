"""Small deterministic retrieval tests without H&M asset access."""
import unittest
import duckdb
import numpy as np
from hm_recsys.warm_v3_retrieval_audit import allowed, topk_exact, pool_metrics, CUTOFFS


class BPRRetrievalTests(unittest.TestCase):
    def test_only_registered_inner_dates(self):
        for cutoff in CUTOFFS:
            allowed(cutoff)
        for cutoff in ('2020-01-22', '2019-02-20', '2020-09-16'):
            with self.assertRaises(ValueError):
                allowed(cutoff)

    def test_ties_are_article_order_not_input_order(self):
        q = np.array([[1, 1]], np.float32)
        f = np.array([[1, 2], [3, 0], [0, 1]], np.float32)
        ids, scores = topk_exact(q, f, np.array(['b', 'a', 'c']), k=2)
        self.assertEqual(ids.tolist(), [['a', 'b']])
        np.testing.assert_array_equal(scores, [[3, 3]])

    def test_bias_column_is_part_of_inner_product(self):
        q = np.array([[2, 1]], np.float32)
        f = np.array([[3, -10], [0, 1]], np.float32)
        ids, scores = topk_exact(q, f, np.array(['a', 'b']), k=2)
        self.assertEqual(ids.tolist(), [['b', 'a']])
        np.testing.assert_array_equal(scores, [[1, -4]])

    def test_duplicate_catalog_rejected(self):
        with self.assertRaises(ValueError):
            topk_exact(np.ones((1, 2), np.float32), np.ones((2, 2), np.float32), ['a', 'a'], k=2)

    def test_batching_does_not_change_exact_order(self):
        q = np.array([[1, 2], [2, 1], [0, 1]], np.float32)
        f = np.array([[0, 1], [1, 0], [1, 1]], np.float32)
        first = topk_exact(q, f, np.array(['c', 'a', 'b']), k=3, batch=1)
        second = topk_exact(q, f, np.array(['c', 'a', 'b']), k=3, batch=3)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_empty_bpr_users_keep_full_truth_denominator(self):
        with duckdb.connect() as con:
            con.execute("CREATE TABLE truth AS SELECT * FROM (VALUES ('u1','a'),('u1','b'),('u2','c')) t(customer_id,article_id)")
            con.execute('CREATE TABLE truth_users AS SELECT customer_id,count(*) truth_count FROM truth GROUP BY customer_id')
            result = pool_metrics(con, "SELECT 'u1' customer_id,'a' article_id")
        self.assertEqual(result['users_denominator'], 2)
        self.assertAlmostEqual(result['macro_recall'], 0.25)
        self.assertAlmostEqual(result['oracle_map12'], 0.25)


if __name__ == '__main__':
    unittest.main()
