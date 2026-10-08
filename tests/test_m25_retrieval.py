import unittest

import duckdb
import numpy as np
import pandas as pd

from hm_recsys.m25_retrieval import (
    _build_image_candidates,
    _candidate_metrics,
    _evaluate_table,
)


class M25RetrievalTests(unittest.TestCase):
    def test_candidate_metrics_average_over_truth_users(self):
        con = duckdb.connect()
        con.execute("CREATE TABLE eval_truth(customer_id VARCHAR,article_id VARCHAR,item_temperature VARCHAR,popularity_segment VARCHAR)")
        con.execute("INSERT INTO eval_truth VALUES ('u1','a','warm','head_top20pct_items'),('u1','b','cold','cold_or_unseen'),('u2','c','warm','tail_bottom40pct_items')")
        con.execute("CREATE TABLE candidates(customer_id VARCHAR,article_id VARCHAR,candidate_rank INTEGER)")
        con.execute("INSERT INTO candidates VALUES ('u1','a',1),('u1','x',2),('u2','x',1)")
        metrics = _candidate_metrics(con, "candidates", 2)
        self.assertAlmostEqual(metrics["candidate_recall@2"], 0.25)
        self.assertAlmostEqual(metrics["candidate_hit_rate@2"], 0.5)
        self.assertAlmostEqual(metrics["oracle_map@12"], 0.25)
        self.assertAlmostEqual(metrics["map@12"], 0.25)
        con.close()

    def test_image_candidate_aggregation_is_ranked_and_unique(self):
        con = duckdb.connect()
        seeds = pd.DataFrame(
            {"customer_id": ["u1", "u1"], "row_index": [0, 1], "seed_rank": [1, 2]}
        )
        indices = np.array([[2, 3], [2, 0]], dtype=np.int32)
        scores = np.array([[0.9, 0.8], [1.0, 0.7]], dtype=np.float16)
        stats = _build_image_candidates(
            con, seeds, indices, scores, np.array(["a", "b", "c", "d"]), 3
        )
        rows = con.execute(
            "SELECT article_id,image_rank,seed_support FROM image_candidates ORDER BY image_rank"
        ).fetchall()
        self.assertEqual(stats["rows"], 3)
        self.assertEqual([row[0] for row in rows], ["c", "d", "a"])
        self.assertEqual(rows[0][2], 2)
        con.close()

    def test_activity_evaluation_uses_view_without_mutating_truth(self):
        con = duckdb.connect()
        con.execute("CREATE TABLE eval_truth(customer_id VARCHAR,article_id VARCHAR,item_temperature VARCHAR,popularity_segment VARCHAR)")
        con.execute("INSERT INTO eval_truth VALUES ('u1','a','warm','head_top20pct_items'),('u2','b','cold','cold_or_unseen')")
        con.execute("CREATE TABLE user_activity(customer_id VARCHAR,events INTEGER,activity_segment VARCHAR)")
        con.execute("INSERT INTO user_activity VALUES ('u1',0,'inactive_12w'),('u2',2,'low_1_5')")
        con.execute("CREATE TABLE candidates(customer_id VARCHAR,article_id VARCHAR,candidate_rank INTEGER)")
        con.execute("INSERT INTO candidates VALUES ('u1','a',1),('u2','x',1)")
        result = _evaluate_table(con, "candidates", 1)
        self.assertEqual(result["segments"]["overall"]["truth_pairs"], 2)
        self.assertAlmostEqual(
            result["activity_segments"]["inactive_12w"]["candidate_recall@1"], 1.0
        )
        self.assertAlmostEqual(
            result["activity_segments"]["low_1_5"]["candidate_recall@1"], 0.0
        )
        self.assertEqual(con.execute("SELECT count(*) FROM eval_truth").fetchone()[0], 2)
        con.close()


if __name__ == "__main__":
    unittest.main()
