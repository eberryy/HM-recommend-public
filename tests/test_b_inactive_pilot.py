"""Cheap, synthetic tests; no project data or labels accessed."""
import importlib.util
from pathlib import Path
import unittest

import duckdb
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('b_pilot', Path(__file__).resolve().parents[1] / 'scripts/run_b_inactive_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class RankingTests(unittest.TestCase):
    def test_rrf_matches_sql_with_ties_and_nontrivial_index(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame({
            'customer_id': np.repeat(['a', 'b', 'c'], 100),
            'article_id': [f'{i:010d}' for i in range(300)],
            'candidate_rank': np.tile(np.arange(1, 101), 3),
            'score_base': rng.integers(0, 4, 300),
            'score_bpr': rng.integers(0, 4, 300),
        }).sample(frac=1, random_state=7)
        actual = pilot.rank_scores(frame)
        with duckdb.connect() as db:
            db.register('scores', frame)
            expected = db.execute('''WITH a AS (SELECT *,
              row_number() OVER(PARTITION BY customer_id ORDER BY score_base DESC,candidate_rank,article_id) r0,
              row_number() OVER(PARTITION BY customer_id ORDER BY score_bpr DESC,candidate_rank,article_id) r1 FROM scores)
              SELECT customer_id,article_id,row_number() OVER(PARTITION BY customer_id
              ORDER BY 1.0/(60+r0)+1.0/(60+r1) DESC,candidate_rank,article_id) output_rank FROM a''').fetchdf()
        merged = actual.merge(expected, on=['customer_id', 'article_id'], validate='one_to_one')
        self.assertTrue(merged.output_rank_x.eq(merged.output_rank_y).all())

    def test_identical_models_preserve_their_order_not_candidate_order(self):
        frame = pd.DataFrame({'customer_id': ['a'] * 3, 'article_id': ['1', '2', '3'],
                              'candidate_rank': [1, 2, 3], 'score_base': [0, 2, 1], 'score_bpr': [0, 2, 1]})
        self.assertEqual(pilot.rank_scores(frame).article_id.tolist(), ['2', '3', '1'])

    def test_all_ties_use_candidate_rank(self):
        frame = pd.DataFrame({'customer_id': ['a'] * 3, 'article_id': ['1', '2', '3'],
                              'candidate_rank': [3, 2, 1], 'score_base': [0] * 3, 'score_bpr': [0] * 3})
        self.assertEqual(pilot.rank_scores(frame).article_id.tolist(), ['3', '2', '1'])


if __name__ == '__main__':
    unittest.main()
