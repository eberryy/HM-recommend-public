"""P4.2 preregistration checks: no model fits or experiment metrics."""

import copy
import inspect
import json
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from hm_recsys import p42_contract, p42_data, p42_matching, p42_propensity


@unittest.skipUnless(__import__("os").environ.get("HM_RESEARCH_ASSETS") == "1", "requires withheld historical contracts/checkpoints and original research branch")
class P42ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "reports/phase4/P4_2_EXPERIMENT_CONTRACT.json"
        with path.open(encoding="utf-8") as stream:
            cls.frozen = json.load(stream)

    def test_final_and_overlapping_truth_intervals_fail_closed(self):
        for guard in (p42_contract.guard_cutoff, p42_data.guard_cutoff):
            for cutoff in ("2020-09-16", "2020-09-17", "2021-01-01",
                           "2020-09-10", "2020-09-15"):
                with self.subTest(guard=guard.__module__, cutoff=cutoff):
                    with self.assertRaises(ValueError):
                        guard(cutoff)
            self.assertEqual(guard("2020-09-09"), "2020-09-09")
            self.assertEqual(guard("2020-08-19"), "2020-08-19")
        self.assertEqual(self.frozen["reject_cutoff_on_or_after"], "2020-09-16")
        self.assertEqual(self.frozen["final_week"], "not_run")

    def test_historical_label_end_strictly_before_not_equal(self):
        first = "2019-11-27"
        self.assertNotIn(first, p42_contract.history_cutoffs("2019-12-04"))
        self.assertIn(first, p42_contract.history_cutoffs("2019-12-05"))
        available = {cutoff: True for cutoff in p42_contract.CUTOFFS}
        available[first] = False
        self.assertNotIn(first, p42_contract.history_cutoffs("2019-12-05", available))
        for outer in self.frozen["windows"].values():
            for cutoff in p42_contract.history_cutoffs(outer):
                self.assertLess(date.fromisoformat(cutoff) + timedelta(days=7),
                                date.fromisoformat(outer))

    def test_exact_ten_cutoffs_expanding_pools_and_qc_first_exclusion(self):
        expected = ["2019-11-27", "2019-12-25", "2020-01-22", "2020-02-19",
                    "2020-03-18", "2020-04-29", "2020-05-27", "2020-06-24",
                    "2020-07-22", "2020-08-19"]
        self.assertEqual(p42_contract.CUTOFFS, expected)
        self.assertEqual(self.frozen["existing_cutoff_sequence"], expected)
        self.assertEqual(self.frozen["windows"], p42_contract.WINDOWS)
        available = {cutoff: item["b0_lineage"]["available"]
                     for cutoff, item in self.frozen["inputs"].items()}
        self.assertFalse(available[expected[0]])
        previous = {"qC": set(), "qW": set()}
        for window, outer in self.frozen["windows"].items():
            pools = self.frozen["historical_pools"][window]
            self.assertEqual(pools["qC"], p42_contract.history_cutoffs(outer, available))
            self.assertEqual(pools["qW"], p42_contract.history_cutoffs(outer))
            self.assertNotIn(expected[0], pools["qC"])
            self.assertIn(expected[0], pools["qW"])
            for side in ("qC", "qW"):
                self.assertTrue(previous[side] <= set(pools[side]))
                previous[side] = set(pools[side])
                self.assertNotIn(outer, pools[side])
        self.assertEqual(self.frozen["historical_pools"]["winter_20200122"]["qC"],
                         ["2019-12-25"])

    def test_exact_feature_lists_and_forbidden_inputs(self):
        self.assertEqual(self.frozen["feature_spec"], p42_contract.FEATURE_SPEC)
        self.assertEqual(self.frozen["feature_contract"]["feature_spec"],
                         p42_contract.FEATURE_SPEC)
        cold = self.frozen["feature_spec"]["qC"]
        warm = self.frozen["feature_spec"]["qW"]
        self.assertEqual(cold["numeric"], [
            "b0_rank_pct", "b0_user_percentile", "b0_user_zscore",
            "normalized_margin_to_rank2", "normalized_margin_to_rank5",
            "normalized_margin_to_user_median", "m4_coarse_rank_pct",
            "interaction_count_before_cutoff", "history_count_0_7", "history_count_8_28",
            "history_count_29_84", "history_count_over_84", "days_since_last_purchase",
            "recent_0_28_purchase_share"])
        self.assertEqual(cold["binary"], ["strict_cold_flag", "sparse1_5_flag"])
        cold_inputs, warm_inputs = cold["numeric"] + cold["binary"], warm["numeric"] + warm["binary"]
        for forbidden in ("b0_score", "b0_delta_vs_m4", "article_id", "target",
                          "customer_id", "label"):
            self.assertNotIn(forbidden, cold_inputs)
        raw_ids = {"article_id", "customer_id", "product_code", "product_type_no",
                   "department_no", "colour_group_code", "article_product_code",
                   "article_product_type_no", "article_department_no",
                   "article_colour_master_id", "article_garment_group_no",
                   "article_index_group_no"}
        self.assertFalse(raw_ids & set(warm_inputs))
        self.assertFalse({"target", "label", "cold_present", "source_branch",
                          "interaction_count_before_cutoff", "strict_cold_flag",
                          "sparse1_5_flag"} & set(warm_inputs))
        self.assertFalse(any(name.startswith(("b0_", "m4_", "cold_")) for name in warm_inputs))
        self.assertIn("warm_model_score_available", warm["binary"])
        self.assertIn("warm_user_percentile", warm["numeric"])
        self.assertIn("warm_user_zscore", warm["numeric"])
        for side, specification in self.frozen["feature_spec"].items():
            numeric, binary, actual_order = p42_propensity._feature_names(specification)
            self.assertEqual(actual_order, numeric + binary + [x + "_available" for x in numeric])
            self.assertEqual(len(set(actual_order)), len(actual_order))
            self.assertEqual(self.frozen["feature_contract"]["missing_indicators"][side],
                             [x + "_available" for x in numeric])
        self.assertEqual(tuple(p42_contract.STATE), p42_data.STATE_COLUMNS)

    def test_fixed_model_no_sampling_and_prediction_has_no_labels(self):
        parameters = self.frozen["model"]
        for name, value in p42_propensity.MODEL_PARAMS.items():
            self.assertEqual(parameters[name], value)
        self.assertEqual(parameters["solver"], "lbfgs")
        self.assertEqual(parameters["penalty"], "l2")
        self.assertEqual(parameters["C"], 1.0)
        self.assertEqual(parameters["max_iter"], 1000)
        self.assertEqual(parameters["random_state"], 20260909)
        self.assertIsNone(parameters["class_weight"])
        self.assertIs(parameters["downsample"], False)
        self.assertIs(parameters["oversample"], False)
        self.assertIs(parameters["keep_zero_positive_users"], True)
        self.assertEqual(parameters["thread_limit"], p42_propensity.THREAD_LIMIT)
        self.assertEqual(list(inspect.signature(p42_propensity.predict_propensity).parameters),
                         ["fitted", "frame"])
        self.assertEqual(list(inspect.signature(p42_propensity.fit_propensity).parameters),
                         ["frame", "feature_spec", "target_column"])

    def test_fixed_taus_epsilon_matching_and_preserved_boundaries(self):
        self.assertEqual(self.frozen["taus"], p42_contract.TAUS)
        self.assertEqual(tuple(self.frozen["taus"].values()), p42_matching.TAUS)
        self.assertEqual(self.frozen["epsilon"], 1e-6)
        self.assertEqual(self.frozen["epsilon"], p42_matching.EPSILON)
        self.assertEqual(self.frozen["epsilon"], p42_propensity.EPSILON)
        self.assertEqual(self.frozen["software_versions"]["scipy"],
                         p42_matching.MATCHING_SOLVER_VERSION)
        self.assertEqual(self.frozen["pairing"]["max_pairs_per_user"], 600)
        self.assertIsNone(self.frozen["pairing"]["max_admission_hard_cap"])
        self.assertFalse(self.frozen["pairing"]["position_multiplier"])
        for name in ("p4_1b_allowed", "P4_1B_started", "Warm_v2_integrated", "P4_3_started"):
            self.assertIs(self.frozen["historical_boundaries"][name], False)

    def test_branch_guard_rejects_warm_and_other_nonmain_before_writes(self):
        with patch.object(p42_contract.subprocess, "check_output", return_value="main\n"):
            self.assertEqual(p42_contract.branch_guard("unused"), "main")
        for branch in ("warm-v2-autonomous-lab", "codex/other", ""):
            with patch.object(p42_contract.subprocess, "check_output", return_value=branch):
                with self.assertRaises(ValueError):
                    p42_contract.branch_guard("unused")

    def test_frozen_upstream_lineage_allows_boundary_but_rejects_future(self):
        # Existing upstream training may end exactly at its scoring cutoff.
        # This differs intentionally from the new heads' stricter pool rule.
        cutoff = "2020-01-01"
        lineage = {
            "b0_lineage": {"available": True, "lineage_safe": True,
                           "model_training_cutoffs": ["2019-12-25"],
                           "model_label_end": cutoff, "model": {"path": "B0_outer.pt"}},
            "warm_lineage": {"available": True, "safe": True,
                             "training_cutoffs": ["2019-12-25"],
                             "latest_training_label_end": cutoff},
        }
        p42_data.validate_lineages(lineage, cutoff)
        for side, end, safe in (("b0_lineage", "model_label_end", "lineage_safe"),
                                ("warm_lineage", "latest_training_label_end", "safe")):
            future = copy.deepcopy(lineage)
            future[side][end] = "2020-01-02"
            with self.assertRaises(ValueError):
                p42_data.validate_lineages(future, cutoff)
            unsafe = copy.deepcopy(lineage)
            unsafe[side][safe] = False
            with self.assertRaises(ValueError):
                p42_data.validate_lineages(unsafe, cutoff)
        wrong_model = copy.deepcopy(lineage)
        wrong_model["b0_lineage"]["model"]["path"] = "B1_outer.pt"
        with self.assertRaises(ValueError):
            p42_data.validate_lineages(wrong_model, cutoff)


if __name__ == "__main__":
    unittest.main()
