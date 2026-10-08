from __future__ import annotations

import unittest

import numpy as np
import torch

from hm_recsys.m4_contract import (
    FINAL_CUTOFF,
    FORBIDDEN_STUDENT_FIELDS,
    STATIC_FIELDS,
    validate_protocol,
)
from hm_recsys.m4_retrieval import K_VALUES, PRIMARY_K, evaluate_retrieval
from hm_recsys.m4_student import StaticItemEncoder, VARIANTS
from hm_recsys.m4_teacher import TEACHER_CONTRACT, required_teacher_cutoffs
from hm_recsys.m4_verify import iter_identities


class M4ProtocolTests(unittest.TestCase):
    def test_protocol_never_reaches_final_week(self) -> None:
        validate_protocol()
        self.assertTrue(all(cutoff < FINAL_CUTOFF for cutoff in required_teacher_cutoffs()))
        self.assertEqual(len(required_teacher_cutoffs()), 8)

    def test_student_static_fields_exclude_behavior_and_identifiers(self) -> None:
        self.assertNotIn("article_id", STATIC_FIELDS)
        self.assertNotIn("product_code", STATIC_FIELDS)
        self.assertIn("product_code", FORBIDDEN_STUDENT_FIELDS)
        self.assertNotIn("item2vec_vector", STATIC_FIELDS)

    def test_teacher_contract_is_frozen(self) -> None:
        self.assertEqual(TEACHER_CONTRACT["positive_top_k"], 20)
        self.assertEqual(TEACHER_CONTRACT["hard_negative_exclusion_top_k"], 100)
        self.assertEqual(TEACHER_CONTRACT["workers"], 1)

    def test_encoder_variants_produce_normalized_128d_vectors(self) -> None:
        images = torch.randn(4, 512)
        categories = torch.tensor([[1, 1], [2, 0], [1, 2], [0, 0]], dtype=torch.long)
        missing = torch.tensor([0.0, 0.0, 1.0, 1.0])
        for variant in VARIANTS:
            model = StaticItemEncoder(variant=variant, cardinalities=[3, 3])
            output = model(images, categories, missing)
            self.assertEqual(tuple(output.shape), (4, 128))
            np.testing.assert_allclose(output.detach().norm(dim=1).numpy(), 1.0, atol=1e-5)

    def test_retrieval_metrics_deduplicate_warm_and_respect_budget(self) -> None:
        users = ["u1", "u2"]
        truth = {"u1": {"a", "b"}, "u2": {"c"}}
        base = {"u1": {"a"}, "u2": set()}
        catalog = ["a", "b", "c", "d"]
        counts = np.array([20, 0, 3, 0], dtype=np.int32)
        retrieved = {
            "u1": [{"catalog_row": 0}, {"catalog_row": 1}],
            "u2": [{"catalog_row": 2}, {"catalog_row": 3}],
        }
        result = evaluate_retrieval(
            users=users,
            truth=truth,
            base=base,
            retrieved=retrieved,
            catalog_items=catalog,
            counts=counts,
            k=2,
        )
        self.assertEqual(result["incremental_candidate_rows_after_warm_dedup"], 3)
        self.assertEqual(result["incremental_truth_pairs"], 2)
        self.assertEqual(result["incremental_strict_cold_truth_pairs"], 1)
        self.assertAlmostEqual(result["segments"]["overall"]["recall"], 1.0)
        self.assertTrue(result["candidate_budget_passed"])

    def test_candidate_budgets_are_bounded_and_primary_is_fixed(self) -> None:
        self.assertEqual(K_VALUES, (20, 50, 100))
        self.assertEqual(PRIMARY_K, 50)

    def test_identity_walker_finds_only_complete_identity_records(self) -> None:
        value = {
            "good": {"path": "x", "bytes": 1, "sha256": "a", "note": "ok"},
            "partial": {"path": "y", "bytes": 2},
            "list": [{"path": "z", "bytes": 3, "sha256": "b"}],
        }
        self.assertEqual([row["path"] for row in iter_identities(value)], ["x", "z"])


if __name__ == "__main__":
    unittest.main()
