"""Synthetic fixed-pool mechanics; no real data or training."""
import unittest
from unittest.mock import patch
from pathlib import Path
import duckdb
import numpy as np
from hm_recsys import warm_v3_pool as pool
from hm_recsys import warm_v2_engine as we
from hm_recsys.m211 import _distribution_relation_sql, SAMPLING_SEED


class PoolTests(unittest.TestCase):
    def test_existing_pool_contract_never_resets_model_output_root(self):
        sentinel = pool.ART / 'models'
        with patch.object(pool.Path, 'cwd', return_value=pool.common.ROOT), \
             patch.object(pool.common, 'git', return_value='warm-v3-architecture-lab'), \
             patch.object(pool.Path, 'exists', return_value=True), \
             patch.object(pool, 'read', return_value={'retrieval_k': 100, 'maximum_pool': 400}), \
             patch.object(pool.common, 'setup', side_effect=AssertionError('must not reset trainer output')) as setup, \
             patch.object(we, 'ARTIFACT', sentinel):
            self.assertEqual(pool.contract()['maximum_pool'], 400)
            setup.assert_not_called()
            self.assertEqual(we.ARTIFACT, sentinel)

    def test_default_only_original_inner_cutoffs(self):
        for cutoff in pool.INNER_CUTOFFS:
            pool.allowed(cutoff)
        for cutoff in ('2019-11-27', '2020-01-22', '2020-09-16'):
            with self.assertRaises(ValueError):
                pool.allowed(cutoff)

    def test_explicit_training_date_permission_still_forbids_final(self):
        pool.allowed('2019-11-27', pool.ALL_CUTOFFS)
        with self.assertRaises(ValueError):
            pool.allowed('2020-09-16', ('2020-09-16',))

    def test_two_way_sampler_only_renames_semantic_label(self):
        path = Path('synthetic.parquet')
        expected = _distribution_relation_sql(path, seed=SAMPLING_SEED)
        self.assertEqual(pool.sampled_relation(path).replace("'expanded_tail'", "'item2vec_only'"), expected)

    def test_append_excludes_old_pairs_preserves_contiguous_rank(self):
        with duckdb.connect() as con:
            con.execute("CREATE TABLE original AS SELECT * FROM (VALUES ('u','a',1),('u','b',2),('v','c',1)) t(customer_id,article_id,candidate_rank)")
            con.execute("CREATE TABLE retrieved AS SELECT * FROM (VALUES ('u','b',1),('u','e',3),('u','d',3),('v','c',1),('v','z',2)) t(customer_id,article_id,retrieval_rank)")
            output = con.execute(pool.append_relation() + ' ORDER BY customer_id,candidate_rank').fetchall()
        self.assertEqual([(r[0], r[1], r[-1]) for r in output], [('u', 'd', 3), ('u', 'e', 4), ('v', 'z', 2)])

    def test_pool_engine_overrides_old_cache_and_signal_dispatch(self):
        self.assertIn('cached_data', pool.PoolEngine.__dict__)
        self.assertIn('signal_path', pool.PoolEngine.__dict__)
        self.assertIn('base_path', pool.PoolEngine.__dict__)

    def test_partial_selection_matches_full_stable_sort_including_boundary_ties(self):
        scores = np.array([[1, 3, 3, 3, 0, 2], [0, 0, 0, 0, 0, 0], [5, 4, 3, 2, 1, 0]], np.float32)
        expected = np.argsort(-scores, axis=1, kind='stable')[:, :3]
        np.testing.assert_array_equal(pool.partial_topk(scores, 3), expected)

    def test_cpu_partial_dot_retrieval_matches_full_sort(self):
        query = np.array([[1, 0, 1], [0, 1, 1]], np.float32)
        items = np.array([[1, 2, 1], [2, 1, 1], [1, 1, 1], [0, 1, 0]], np.float32)
        ids = np.array(['d', 'b', 'a', 'c'])
        full = pool.topk_exact(query, items, ids, k=3, device='cpu')
        part = pool.cpu_topk_exact(query, items, ids, k=3, batch=1)
        np.testing.assert_array_equal(full[0], part[0])
        np.testing.assert_array_equal(full[1], part[1])


if __name__ == '__main__':
    unittest.main()
