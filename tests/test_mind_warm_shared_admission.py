import numpy as np
import pandas as pd
import unittest

from hm_recsys.mind_warm_shared_admission import select, SCORE_COLUMNS
from hm_recsys.mind_warm_calibrated_admission import fit_mapping, predict


def sample():
    return pd.DataFrame([
        ["a", "new", "v8", 8, .03, .01],
        ["a", "new", "v12", 12, .03, .01],
        ["b", "new", "v8", 8, .01, .02],
        ["c", "new", "v8", 8, .02, .02],
    ], columns=SCORE_COLUMNS)


class SharedAdmissionTests(unittest.TestCase):
    def test_one_positive_net_action_and_noop(self):
        out = select(sample())
        self.assertEqual(out.customer_id.tolist(), ["a"])
        self.assertEqual(out.champion_rank.tolist(), [8])
        self.assertAlmostEqual(out.replacement_score.iloc[0], .02 / 8)

    def test_reject_labels_and_protected_head(self):
        with self.assertRaises(ValueError):
            select(sample().assign(target=1))
        bad = sample()
        bad.loc[0, "champion_rank"] = 7
        with self.assertRaises(ValueError):
            select(bad)

    def test_invalid_predictions(self):
        for value in [np.nan, np.inf, -.1, 1.1]:
            with self.subTest(value=value):
                bad = sample()
                bad.loc[0, "candidate_purchase_score"] = value
                with self.assertRaises(ValueError):
                    select(bad)

    def test_stable_identity_ties_and_empty(self):
        frame = sample().iloc[[0, 0]].copy()
        frame.iloc[0, frame.columns.get_loc("victim_article_id")] = "z"
        frame.iloc[1, frame.columns.get_loc("victim_article_id")] = "a"
        self.assertEqual(select(frame).victim_article_id.tolist(), ["a"])
        self.assertTrue(select(sample().iloc[:0]).empty)

    def test_calibration_monotone_and_no_missing_class_fit(self):
        scores = np.linspace(.001, .1, 100)
        labels = np.zeros(100)
        labels[80:] = 1
        mapping = fit_mapping(scores, labels)
        self.assertTrue(mapping["success"])
        self.assertGreaterEqual(mapping["slope"], 0)
        predictions = predict(mapping, scores)
        self.assertTrue(np.all(np.diff(predictions) >= 0))
        self.assertEqual(mapping["rows"], 100)
        self.assertEqual(mapping["positives"], 20)
        with self.assertRaises(AssertionError):
            fit_mapping(scores, np.zeros(100))
