"""Frozen C schema, target isolation and selected-side tests without model fits."""
from pathlib import Path
import json
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from hm_recsys.p42f_contract import COLD, WARM, USER
from hm_recsys.p43a_data import META_COLUMNS
from hm_recsys.p43a_c import (
    build_features, validate_lineage, oracle_targets, count_decision,
    native_parameters, selected_matching, _evaluate,
)
from hm_recsys.p43a_policy import exact_ap


def schema():
    repo = Path(__file__).resolve().parents[1]
    return json.loads((repo / 'reports/phase4/P4_3A_EXPERIMENT_CONTRACT.json').read_text(encoding='utf8'))


def feature_fixture():
    users = np.array(['u0', 'u1'])
    state = pd.DataFrame({name: [10., 20.] for name in USER})
    state['customer_id'] = users
    warm = pd.DataFrame({name: np.zeros(24) for name in WARM})
    warm['customer_id'] = np.repeat(users, 12)
    warm['warm_rank'] = np.tile(np.arange(1, 13), 2)
    warm['warm_rank_pct'] = (warm.warm_rank - 1) / 11
    warm['qW_relative_available'] = 1.
    warm['qW_within_user_percentile'] = np.tile(np.linspace(0, 1, 12), 2)
    warm['source_count'] = np.tile(np.arange(12), 2)
    warm['warm_user_zscore'] = np.nan
    cold = pd.DataFrame({name: np.zeros(3) for name in COLD})
    cold['customer_id'], cold['user_index'], cold['article_id'] = ['u0']*3, [0]*3, ['c0', 'c1', 'c2']
    cold['b0_rank'], cold['b0_rank_pct'] = [1., 2., 3.], [0., .1, .2]
    cold['b0_user_percentile'] = [.9, .8, .7]
    cold['qC_relative_available'], cold['qC_within_user_percentile'] = [1., 0., 1.], [.1, np.nan, .8]
    meta = np.zeros((36, len(META_COLUMNS)), np.float32)
    for ci, value in enumerate((.5, .2, .8)):
        meta[ci*12:(ci+1)*12, 0] = value
        meta[ci*12:(ci+1)*12, 1] = 1.
    # Deliberately no target, truth, relevance or oracle fields in this fixture.
    return dict(users=users, state=state, warm=warm, cold=cold), meta


@unittest.skipUnless(__import__("os").environ.get("HM_RESEARCH_ASSETS") == "1", "requires withheld historical contracts/checkpoints and original research branch")
class CTests(unittest.TestCase):
    def test_exact_schema_without_any_label_fields(self):
        data, meta = feature_fixture()
        names = schema()['features']['C_exact']
        features = build_features(data, meta, names)
        for role, rows in [('count', 2), ('cold', 3), ('warm', 24)]:
            self.assertEqual(features[role].shape, (rows, len(names[role])))
            self.assertEqual(features[role].dtype, np.float32)
        np.testing.assert_array_equal(features['action_users'], [0])
        for key, value in [('B0_top3_mean', .8), ('qC_top3_mean', .3), ('old_U_top3_mean', .5),
                            ('old_U_top1_mean', .8), ('H_q_raw_top1_mean', 0.),
                            ('qW_vulnerability_top1', 1.), ('warm_source_count_mean', 5.5)]:
            self.assertAlmostEqual(float(features['count'][0, names['count'].index(key)]), value, places=6)
        self.assertTrue(np.isnan(features['count'][0, names['count'].index('warm_zscore_std')]))
        self.assertEqual(features['count'][1, names['count'].index('B0_top1_mean')], 0.)

    def test_candidate_meta_max_precedes_topK_mean(self):
        data, meta = feature_fixture()
        j = META_COLUMNS.index('H_q_raw_percentile')
        meta[:, j + 1] = 1.
        meta[:12, j] = .1
        meta[11, j] = .9
        meta[12:24, j] = .4
        meta[24:, j] = .5
        names = schema()['features']['C_exact']
        features = build_features(data, meta, names)
        self.assertAlmostEqual(float(features['count'][0, names['count'].index('H_q_raw_top3_mean')]), .6, places=6)
        self.assertAlmostEqual(float(features['cold'][0, names['cold'].index('max_meta_H_q_raw_percentile')]), .9, places=6)

    def test_missing_meta_and_wrong_schema_fail(self):
        data, meta = feature_fixture()
        meta[0, 4] = .5  # unavailable score cannot carry information
        with self.assertRaises(ValueError):
            build_features(data, meta, schema()['features']['C_exact'])
        data, meta = feature_fixture()
        names = schema()['features']['C_exact']
        names['cold'] = list(reversed(names['cold']))
        with self.assertRaises(ValueError):
            build_features(data, meta, names)

    def test_count_fixed_classes_and_noCold(self):
        probabilities = np.array([[.1, .2, .3, .4], [.5, .5, 0., 0.], [0., 0., 0., 1.], [.1, .1, .1, .7]])
        np.testing.assert_array_equal(count_decision(probabilities, [5, 5, 0, 2]), [3, 0, 0, 2])
        params = next(c['params'] for c in schema()['model_configs'] if c['id'] == 'C-count')
        native, rounds = native_parameters(params)
        self.assertEqual(native['num_class'], 4)
        self.assertEqual(native['objective'], 'multiclass')
        self.assertEqual(native['num_threads'], 4)
        self.assertEqual(rounds, 300)
        self.assertNotIn('n_estimators', native)

    def test_oracle_Kclip_does_not_clip_selected_item_targets(self):
        data = dict(users=['u0'], cold=pd.DataFrame({'user_index': [0]*4}))
        matches = np.full((1, 13, 12, 2), -1, np.int32)
        matches[0, 4, :4] = [[0, 0], [1, 1], [2, 2], [3, 3]]
        target = oracle_targets(data, np.array([4]), matches)
        np.testing.assert_array_equal(target['count'], [3])
        self.assertEqual(target['cold'].sum(), 4)
        self.assertEqual(target['warm'].sum(), 4)
        matches[0, 4, 3] = [3, 2]
        with self.assertRaises(ValueError): oracle_targets(data, np.array([4]), matches)

    def test_zero_meta_still_matches_predicted_K(self):
        meta = np.zeros((3, 12, len(META_COLUMNS)))
        selected = selected_matching(2, [.2, .8, .6], np.arange(12, dtype=float),
                                     [1, 2, 3], ['c0', 'c1', 'c2'], meta,
                                     {'old_U': 0., 'G_UR': 0., 'H_q_cal': .5})
        self.assertEqual(len(selected), 2)
        self.assertEqual({i for i, _ in selected}, {1, 2})
        self.assertEqual({j for _, j in selected}, {10, 11})
        self.assertEqual(selected_matching(0, [], np.ones(12), [], [], np.empty((0, 12, len(META_COLUMNS))),
                                            {'old_U': 1., 'G_UR': 0., 'H_q_cal': 0.}), [])

    def test_blend_controls_pairing_not_challenger_membership(self):
        meta = np.zeros((2, 12, len(META_COLUMNS)))
        j = META_COLUMNS.index('old_U_percentile')
        meta[0, 11, j], meta[1, 10, j] = 1., 1.
        selected = selected_matching(2, [.5, .5], np.arange(12, dtype=float), [2, 1], ['c0', 'c1'], meta,
                                     {'old_U': 1., 'G_UR': 0., 'H_q_cal': 0.})
        self.assertEqual(selected, [(1, 10), (0, 11)])

    def test_strict_meta_lineage_and_sealed_week(self):
        audit = dict(cutoff='2020-01-22', columns=list(META_COLUMNS),
                     lineage={'old_U': dict(availability=1, training_label_end='2020-01-01')})
        validate_lineage(audit, '2020-01-22')
        audit['lineage']['old_U']['training_label_end'] = '2020-01-22'
        with self.assertRaises(ValueError): validate_lineage(audit, '2020-01-22')
        audit['lineage']['old_U']['availability'] = 0
        validate_lineage(audit, '2020-01-22')
        with self.assertRaises(ValueError): validate_lineage(audit, '2020-09-16')

    def test_final_list_AP_and_noaction_exact_zero(self):
        data, meta = feature_fixture()
        data['cutoff'] = '2020-01-22'
        data['warm_lists'] = [[f'w{u}-{i}' for i in range(12)] for u in range(2)]
        data['truthsets'] = {'u0': {'c0', 'w0-10'}, 'u1': {'w1-0'}}
        data['truth'] = pd.DataFrame(dict(customer_id=['u0', 'u0', 'u1'],
            article_id=['c0', 'w0-10', 'w1-0'], interaction_count_before_cutoff=[0, 30, 30]))
        predictions = dict(count=np.array([[0., 1., 0., 0.], [1., 0., 0., 0.]]),
                           cold=np.array([.9, .2, .1]), warm=np.tile(np.arange(12, dtype=float), 2))
        blends = [dict(id='one', weights={'old_U': 1., 'G_UR': 0., 'H_q_cal': 0.})]
        with TemporaryDirectory() as folder:
            row = _evaluate(data, meta, predictions, blends, Path(folder), None)[0]
            expected = list(data['warm_lists'][0]); expected[11] = 'c0'
            expected_map = (exact_ap(expected, data['truthsets']['u0']) + 1.) / 2
            self.assertAlmostEqual(row['overall_map'], expected_map, places=15)
            self.assertEqual(row['inserted_positives'], 1)
            self.assertEqual(row['removed_positives'], 0)
            self.assertEqual(row['replacements'], 1)
            self.assertIsNone(row['segments']['sparse1_5']['map'])
            chosen = np.load(Path(folder) / 'C-policy-user-matched-pairs.npy')
            np.testing.assert_array_equal(chosen[0, 0, 0], [0, 11])
            self.assertTrue((chosen[0, 1] == -1).all())
        predictions['count'] = np.array([[1., 0., 0., 0.], [0., 0., 0., 1.]])
        with TemporaryDirectory() as folder:
            row = _evaluate(data, meta, predictions, blends, Path(folder), None)[0]
            self.assertEqual(row['delta_map'], 0.)
            self.assertEqual(row['replacements'], 0)

    def test_feature_deadline_is_checked_without_model_fit(self):
        data, meta = feature_fixture()
        with self.assertRaises(TimeoutError):
            build_features(data, meta, schema()['features']['C_exact'], deadline_epoch=0)


if __name__ == '__main__':
    unittest.main()
