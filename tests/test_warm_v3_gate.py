"""Cheap, synthetic guardrails for temporal expert gating; no project data reads."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

from hm_recsys import warm_v3_gate as gate
from hm_recsys import warm_v3_common as common


def users(n=3, activity=None):
    frame = pd.DataFrame({name: np.ones(n) for name in gate.FEATURES})
    frame['history_events'] = np.full(n, 5) if activity is None else activity
    frame['customer_id'] = [f'u{i}' for i in range(n)]
    frame['cutoff'] = '2019-02-20'
    for expert in gate.EXPERTS:
        frame['ap_' + expert] = 0.0
    return frame


class ConstantPredictor:
    def __init__(self, value):
        self.value = value
        self.seen_columns = None

    def predict(self, frame):
        self.seen_columns = list(frame.columns)
        return np.full(len(frame), self.value)


class GateTemporalTests(unittest.TestCase):
    def test_safe_training_accepts_exact_label_end_boundary(self):
        history = [{'cutoff': '2019-02-20', 'id': 'ended'}]
        self.assertEqual(gate.safe_training(history, '2019-02-27'), history)
        self.assertEqual(gate.safe_training(history, '2019-02-26'), [])

    def test_safe_training_excludes_same_week_and_future(self):
        history = [
            {'cutoff': '2019-02-20', 'id': 'past'},
            {'cutoff': '2019-05-22', 'id': 'current'},
            {'cutoff': '2019-08-21', 'id': 'future'},
        ]
        original = [dict(row) for row in history]
        self.assertEqual(gate.safe_training(history, '2019-05-22'), history[:1])
        self.assertEqual(history, original, 'Temporal filtering must not mutate the source registry')

    def test_safe_training_first_replay_has_no_meta(self):
        history = [
            {'cutoff': '2019-02-20'}, {'cutoff': '2019-05-22'},
            {'cutoff': '2019-08-21'}, {'cutoff': '2019-11-20'},
        ]
        self.assertEqual(gate.safe_training(history, '2019-02-20'), [])

    def test_safe_training_december_screen_can_use_all_2019_meta(self):
        history = [{'cutoff': d} for d in ('2019-02-20', '2019-05-22', '2019-08-21', '2019-11-20')]
        self.assertEqual(gate.safe_training(history, '2019-12-25'), history)


class GateFallbackTests(unittest.TestCase):
    def test_empty_meta_falls_back_without_reading_any_asset(self):
        for method in ('logistic', 'utility_tree'):
            with self.subTest(method=method), patch.object(gate, 'load_parquet') as load:
                model, evidence = gate.train([], method)
                self.assertIsNone(model)
                self.assertTrue(evidence.get('fallback'))
                load.assert_not_called()

    def test_too_few_informative_users_fall_back(self):
        frame = users(99)
        frame['ap_e0'] = 0.5
        meta = [{'cutoff': '2019-02-20', 'users_path': 'synthetic'}]
        with patch.object(gate, 'load_parquet', return_value=frame):
            model, evidence = gate.train(meta, 'logistic')
        self.assertIsNone(model)
        self.assertTrue(evidence.get('fallback'))

    def test_many_uninformative_ties_do_not_count_as_supervision(self):
        frame = users(200)
        meta = [{'cutoff': '2019-02-20', 'users_path': 'synthetic'}]
        with patch.object(gate, 'load_parquet', return_value=frame):
            model, evidence = gate.train(meta, 'utility_tree')
        self.assertIsNone(model)
        self.assertTrue(evidence.get('fallback'))

    def test_single_winner_class_is_handled_without_fit_exception(self):
        frame = users(110)
        frame['ap_e0'] = 0.5
        meta = [{'cutoff': '2019-02-20', 'users_path': 'synthetic'}]
        with patch.object(gate, 'load_parquet', return_value=frame):
            model, evidence = gate.train(meta, 'logistic')
        choice = gate.predict(model, 'logistic', frame)
        self.assertEqual(len(choice), len(frame))
        self.assertTrue(np.isin(choice, [0, 1, 2]).all())
        if model is None:
            self.assertTrue(evidence.get('fallback'))

    def test_missing_model_selects_baseline_for_every_user(self):
        frame = users(3, activity=[0, 3, 20])
        np.testing.assert_array_equal(gate.predict(None, 'logistic', frame), [2, 2, 2])

    def test_inactive_is_forced_to_baseline_for_both_methods(self):
        frame = users(3, activity=[0, 3, 20])
        logistic = ConstantPredictor(0)
        np.testing.assert_array_equal(gate.predict(logistic, 'logistic', frame), [2, 0, 0])
        regressors = [ConstantPredictor(-1), ConstantPredictor(1)]
        np.testing.assert_array_equal(gate.predict(regressors, 'utility_tree', frame), [2, 1, 1])

    def test_zero_or_negative_utility_keeps_baseline(self):
        frame = users()
        for a, b in ((0, 0), (-1, 0), (0, -1), (-1, -2)):
            with self.subTest(e0=a, e1=b):
                regressors = [ConstantPredictor(a), ConstantPredictor(b)]
                np.testing.assert_array_equal(gate.predict(regressors, 'utility_tree', frame), [2, 2, 2])

    def test_labels_and_truth_counts_never_reach_gate_input(self):
        frame = users()
        frame['target'] = [0, 1, 1]
        frame['truth_count'] = [2, 100, 1]
        frame['ap_e0'] = [0.1, 1.0, 0.0]
        predictor = ConstantPredictor(0)
        gate.predict(predictor, 'logistic', frame)
        self.assertEqual(predictor.seen_columns, gate.FEATURES)
        self.assertTrue({'target', 'truth_count', 'ap_e0', 'ap_e1', 'ap_rrf', 'cutoff'}.isdisjoint(predictor.seen_columns))


class GateRankScaleTests(unittest.TestCase):
    def test_positive_affine_score_changes_preserve_full_pool_rank_fusion(self):
        # Scores deliberately include ties; frozen candidate_rank/article_id resolve them.
        n = 30
        frame = pd.DataFrame({
            'customer_id': ['u'] * n,
            'article_id': np.arange(100, 100 + n),
            'candidate_rank': np.arange(1, n + 1),
            's0': np.floor(np.arange(n) / 3),
            's1': np.floor(np.arange(n)[::-1] / 4),
        })
        with duckdb.connect() as con:
            con.register('f', frame)
            scored = con.execute(f'''
                SELECT article_id, candidate_rank,
                {gate.rank_sql('s0')} r0,
                {gate.rank_sql('s1')} r1,
                {gate.rank_sql('17*s0+91')} r0_scaled,
                {gate.rank_sql('0.25*s1-103')} r1_scaled
                FROM f ORDER BY candidate_rank,article_id
            ''').fetchdf()
        np.testing.assert_array_equal(scored.r0, scored.r0_scaled)
        np.testing.assert_array_equal(scored.r1, scored.r1_scaled)
        baseline = 1 / (60 + scored.r0) + 1 / (60 + scored.r1)
        scaled = 1 / (60 + scored.r0_scaled) + 1 / (60 + scored.r1_scaled)
        np.testing.assert_array_equal(baseline, scaled)
        self.assertEqual(int(scored.r0.max()), n, 'Ranks must cover all candidates, not only Top12')


class GateEvaluationTests(unittest.TestCase):
    def test_subset_delta_uses_full_user_denominator(self):
        frame = users(2)
        frame['ap_rrf'] = [0.2, 0.1]
        frame['ap_e0'] = [0.4, 0.05]
        meta = {'users_path': 'synthetic', 'total_users': 4}
        with patch.object(gate, 'load_parquet', return_value=frame), \
                patch.object(gate, 'save_parquet') as save, \
                patch.object(Path, 'mkdir'):
            result = gate.score_gate(meta, ConstantPredictor(0), 'logistic', Path('synthetic-out'))
        self.assertAlmostEqual(result['population_delta'], 0.0375)
        self.assertAlmostEqual(result['gross_positive'], 0.05)
        self.assertAlmostEqual(result['gross_negative'], -0.0125)
        self.assertEqual(result['users_improved'], 1)
        self.assertEqual(result['users_harmed'], 1)
        save.assert_called_once()

    def test_2019_robustness_uses_its_own_two_window_tolerance(self):
        reference = {'windows': {k: {'systems': {'FRESH-601': {'map@12': 0.02}}}
                                  for k in ('a', 'b', 'c', 'd')}}
        maps = {'a': 0.0201, 'b': 0.0201, 'c': 0.0197, 'd': 0.0193}
        with patch.object(common, 'read', return_value=reference):
            result = common.summary(maps, year=2019)
        self.assertTrue(result['robustness_pass'])
        self.assertFalse(result['stable'], 'A permissible replay is not a new 2020 champion')
        self.assertFalse(result['target_hit'])

    def test_2019_replay_rejects_excess_mean_loss(self):
        reference = {'windows': {k: {'systems': {'FRESH-601': {'map@12': 0.02}}}
                                  for k in ('a', 'b', 'c', 'd')}}
        maps = {'a': 0.0201, 'b': 0.0201, 'c': 0.019, 'd': 0.019}
        with patch.object(common, 'read', return_value=reference):
            result = common.summary(maps, year=2019)
        self.assertFalse(result['robustness_pass'])


if __name__ == '__main__':
    unittest.main()
