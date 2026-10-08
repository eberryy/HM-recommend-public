import unittest

import duckdb
import numpy as np
import pandas as pd

from hm_recsys.metrics import apk
from hm_recsys.warm_v3_expert import diagnose, rank_bundle, report_path, screening_gate, FEATURES


class ExpertTests(unittest.TestCase):
    def frame(self):
        rows = []
        for user, events in [('active', 3), ('inactive', 0)]:
            for index in range(1, 21):
                rows.append({'customer_id': user, 'article_id': index, 'candidate_rank': index,
                    'target': int(index in (2, 18)), 'truth_count': 3,
                    'user_history_events_12w': events, 'score_base': float(21-index),
                    'score_bpr': float(index), 'score_expert': float(index % 5)})
        return pd.DataFrame(rows)

    def test_fusion_formula_and_inactive_fallback(self):
        with duckdb.connect() as con:
            con.register('frame', self.frame())
            table = rank_bundle(con, 'frame')
            f = con.execute(f'SELECT * FROM {table}').fetchdf()
        np.testing.assert_allclose(f.fusion_score, 1/(60+f.r0)+1/(60+f.r1)+1/(60+f.r2), rtol=0, atol=0)
        inactive = f[f.customer_id == 'inactive']
        np.testing.assert_array_equal(inactive.ap_r3, inactive.candidate_rank)
        np.testing.assert_array_equal(inactive.ap_r2, inactive.candidate_rank)

    def test_full_truth_and_population_denominator(self):
        with duckdb.connect() as con:
            con.register('frame', self.frame())
            table = rank_bundle(con, 'frame')
            result, aps = diagnose(con, table, total_users=4)
            ranked = con.execute(f'SELECT * FROM {table} ORDER BY customer_id,ap_r3').fetchdf()
        independent = [apk([2, 18, 999], g.head(12).article_id.tolist())
                       for _, g in ranked.groupby('customer_id', sort=False)]
        self.assertAlmostEqual(result['systems']['fusion']['covered_map'], np.mean(independent))
        self.assertAlmostEqual(result['systems']['fusion']['population_map_component'], sum(independent)/4)
        self.assertAlmostEqual(result['systems']['fusion']['population_delta'],
                               float((aps.ap_fusion-aps.ap_rrf).sum()/4))

    def test_inner_gate_exact_threshold_and_worst_window(self):
        self.assertTrue(screening_gate([.0002, .0002, .0002, -.0001])['passed'])
        self.assertFalse(screening_gate([.002, .002, .002, -.000501])['passed'])
        self.assertFalse(screening_gate([.002, .002, 0, 0])['passed'])
        with self.assertRaises(ValueError):
            screening_gate([.001])

    def test_84_plus_two_not_bpr_plus_graph(self):
        self.assertEqual(len(FEATURES['graph']), 2)
        self.assertEqual(len(FEATURES['sequence']), 2)
        self.assertFalse(any('bpr' in f for v in FEATURES.values() for f in v))
        self.assertNotEqual(report_path('X', 'SCREEN', 2019), report_path('X', 'SCREEN', 2020))


if __name__ == '__main__':
    unittest.main()
