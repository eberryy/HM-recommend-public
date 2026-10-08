from __future__ import annotations

import inspect
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from hm_recsys.p36_audit import (
    LABEL_TO_CODE,
    TeacherAssets,
    _age_codes,
    _compute_pair_features,
    _safe_cutoff,
    _topk_mean,
    _validate_candidate_identity,
    empirical_mid_percentile,
    relation_labels,
    relation_matrix,
    select_warm_proxies,
)


class P36AuditTests(unittest.TestCase):
    def test_final_week_is_rejected(self) -> None:
        _safe_cutoff("2020-08-19")
        with self.assertRaises(RuntimeError):
            _safe_cutoff("2020-09-16")

    def test_empirical_percentile_is_bounded_and_preserves_missing(self) -> None:
        result = empirical_mid_percentile(np.asarray([3.0, 1.0, 3.0, np.nan], dtype=np.float32))
        self.assertTrue(np.isnan(result[-1]))
        self.assertTrue(np.all((result[:-1] >= 0) & (result[:-1] <= 1)))
        self.assertEqual(result[0], result[2])

    def test_direct_support_must_be_zero_for_multihop_proxy(self) -> None:
        labels = relation_labels(
            np.asarray([0.1, 0.1]),
            np.asarray([False, False]),
            np.asarray([0.1, 0.1]),
            np.asarray([0.9, 0.9]),
            np.asarray([0.9, 0.9]),
            np.asarray([0, 2]),
            content_q50=0.2,
            content_q75=0.8,
        )
        self.assertEqual(labels[0], LABEL_TO_CODE["multihop_complement_like"])
        self.assertNotEqual(labels[1], LABEL_TO_CODE["multihop_complement_like"])

    def test_same_type_never_enters_primary_complement_proxies(self) -> None:
        labels = relation_labels(
            np.asarray([0.1]),
            np.asarray([True]),
            np.asarray([0.95]),
            np.asarray([0.95]),
            np.asarray([0.95]),
            np.asarray([0]),
            content_q50=0.2,
            content_q75=0.8,
        )
        self.assertNotIn(
            int(labels[0]),
            [LABEL_TO_CODE["direct_complement_like"], LABEL_TO_CODE["multihop_complement_like"]],
        )

    def test_secondary_labels_are_one_code_per_observation(self) -> None:
        labels = relation_labels(
            np.asarray([0.9, 0.1, 0.1, 0.4, np.nan]),
            np.asarray([True, False, False, True, False]),
            np.asarray([0.2, 0.9, 0.1, 0.8, np.nan]),
            np.asarray([0.2, 0.2, 0.9, 0.2, np.nan]),
            np.asarray([0.2, 0.9, 0.9, 0.8, np.nan]),
            np.asarray([0, 1, 0, 0, 0]),
            content_q50=0.2,
            content_q75=0.8,
        )
        self.assertEqual(labels.shape, (5,))
        self.assertTrue(np.all(labels < 6))

    def test_primary_matrix_observations_conserve(self) -> None:
        valid = np.asarray([[True, True], [True, False]])
        result = relation_matrix(
            content_codes=np.asarray([[0, 1], [255, 255]], dtype=np.uint8),
            behavior_codes=np.asarray([[0, 3], [2, 255]], dtype=np.uint8),
            same_type=np.asarray([[True, False], [False, False]]),
            valid_history=valid,
            target=np.asarray([True, False]),
            history_age=np.asarray([[1.0, 9.0], [40.0, 0.0]]),
            raw_content=np.asarray([[0.1, 0.5], [np.nan, np.nan]]),
            behavior_any=np.asarray([[0.1, 0.9], [0.7, np.nan]]),
        )
        self.assertTrue(result["observation_conservation_passed"])
        self.assertEqual(result["eligible_observations"] + result["ineligible_observations"], 3)
        self.assertEqual(result["overall_candidate_truth_rate"], 0.5)
        first = result["cells"]["content_Q1__behavior_Q1__same_type"]
        self.assertEqual(first["truth_rate"], 1.0)
        self.assertEqual(first["truth_rate_lift_vs_all_valid_history_candidates"], 2.0)

    def test_strict_cold_item_maps_only_into_supplied_warm_pool(self) -> None:
        vectors = np.eye(6, dtype=np.float32)
        vectors[0] = np.asarray([0.9, 0.1, 0, 0, 0, 0], dtype=np.float32)
        vectors[0] /= np.linalg.norm(vectors[0])
        types = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
        garments = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
        warm_pool = np.asarray([1, 2, 3, 4, 5], dtype=np.int32)
        proxies, _scores, _fallback = select_warm_proxies(
            np.asarray([0]), warm_pool, vectors, types, garments, k=5, device_name="cpu"
        )
        self.assertEqual(proxies.shape, (1, 5))
        self.assertTrue(set(proxies[0].tolist()).issubset(set(warm_pool.tolist())))
        self.assertNotIn(0, proxies[0].tolist())

    def test_proxy_selector_interface_excludes_truth_and_teacher_inputs(self) -> None:
        names = set(inspect.signature(select_warm_proxies).parameters)
        self.assertFalse(names & {"target", "truth", "item2vec", "deepwalk", "direct_covisit"})

    def test_future_item_cannot_enter_when_absent_from_frozen_warm_pool(self) -> None:
        vectors = np.eye(4, dtype=np.float32)
        types = np.zeros(4, dtype=np.int32)
        garments = np.zeros(4, dtype=np.int32)
        proxies, _scores, _fallback = select_warm_proxies(
            np.asarray([0]), np.asarray([1, 2]), vectors, types, garments, k=2, device_name="cpu"
        )
        self.assertNotIn(3, proxies[0].tolist())

    def test_raw_content_and_student_reductions_remain_distinct(self) -> None:
        raw = np.asarray([[0.9, 0.7, 0.1]], dtype=np.float32)
        student = np.asarray([[0.4, 0.3, 0.2]], dtype=np.float32)
        self.assertNotEqual(float(_topk_mean(raw)[0]), float(_topk_mean(student)[0]))

    def test_pair_feature_history_mask_broadcasts_across_top200(self) -> None:
        size = 205
        vectors = np.zeros((size, 2), dtype=np.float32)
        vectors[:, 0] = 1.0
        candidates = {
            "user_index": np.zeros(200, dtype=np.int32),
            "catalog_row": np.arange(200, dtype=np.int32),
            "rank": np.arange(1, 201, dtype=np.int32),
        }
        histories = {
            "catalog_row": np.asarray([[0] + [-1] * 19], dtype=np.int32),
            "days_since_purchase": np.asarray([[1.0] + [0.0] * 19], dtype=np.float32),
            "mask": np.asarray([[1] + [0] * 19], dtype=np.uint8),
        }
        proxies = np.tile(np.arange(200, 205, dtype=np.int32), (size, 1))
        empty = sparse.csr_matrix((size, size), dtype=np.float32)
        teacher = TeacherAssets(
            cutoff="2020-01-01",
            i2v_vectors=vectors,
            i2v_valid=np.ones(size, dtype=bool),
            deep_vectors=vectors,
            deep_valid=np.ones(size, dtype=bool),
            direct_score=empty,
            direct_recent=empty,
            direct_older=empty,
            direct_last_age=empty,
            pair_support=pd.DataFrame(),
            category_pair={},
            identities={},
            cutoff_audit={},
        )
        pair, audit = _compute_pair_features(
            candidates=candidates,
            histories=histories,
            proxies_by_catalog=proxies,
            fashion=vectors,
            student=vectors,
            product_type=np.zeros(size, dtype=np.int32),
            garment_group=np.zeros(size, dtype=np.int32),
            department=np.zeros(size, dtype=np.int32),
            teacher=teacher,
            device_name="cpu",
            user_batch=1,
        )
        self.assertEqual(audit["rows"], 200)
        self.assertTrue(np.all(np.isfinite(pair["student_content"][:, 0])))
        self.assertTrue(np.all(np.isnan(pair["student_content"][:, 1:])))

    def test_candidate_identity_requires_exact_top200(self) -> None:
        candidates = {
            "user_index": np.repeat(np.arange(2, dtype=np.int32), 200),
            "catalog_row": np.tile(np.arange(200, dtype=np.int32), 2),
            "rank": np.tile(np.arange(1, 201, dtype=np.int32), 2),
        }
        self.assertTrue(_validate_candidate_identity(candidates)["exact_top200_groups"])
        candidates["rank"][0] = 2
        with self.assertRaises(RuntimeError):
            _validate_candidate_identity(candidates)

    def test_p32_and_p33_identity_can_share_same_frozen_candidate_object(self) -> None:
        candidates = {
            "user_index": np.zeros(200, dtype=np.int32),
            "catalog_row": np.arange(200, dtype=np.int32),
            "rank": np.arange(1, 201, dtype=np.int32),
        }
        first = _validate_candidate_identity(candidates)["identity_sha256"]
        second = _validate_candidate_identity(candidates)["identity_sha256"]
        self.assertEqual(first, second)

    def test_history_age_buckets_are_exhaustive(self) -> None:
        codes = _age_codes(np.asarray([1.0, 7.0, 8.0, 28.0, 29.0, 84.0, 85.0, 500.0]))
        self.assertEqual(codes.tolist(), [0, 0, 1, 1, 2, 2, 3, 3])


if __name__ == "__main__":
    unittest.main()
