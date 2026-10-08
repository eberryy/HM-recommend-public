from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from hm_recsys.p3_model import sample_same_user_pairs
from hm_recsys.p37b import _new_model
from hm_recsys.p37b_model import (
    AGE_BUCKET_COUNT,
    CANDIDATE_STATE_DIM,
    CANDIDATE_STATE_SPECS,
    FeaturePreprocessor,
    M4_RELATION_DIM,
    M4_RELATION_SPECS,
    P33_GLOBAL_DIM,
    P33_RELATION_DIM,
    TrainOnlyPreprocessor,
    FeatureSpec,
    TimeAwareHybrid,
    USER_STATE_DIM,
    assign_age_buckets,
    feature_specs_for_variant,
    preprocessing_contracts_compatible,
    safe_cutoff,
    sample_training_pairs,
    validate_cutoff_safety,
)


def _model_inputs(batch: int = 5) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(37)
    return (
        torch.randn(batch),
        torch.randn(batch, AGE_BUCKET_COUNT, M4_RELATION_DIM),
        torch.randn(batch, USER_STATE_DIM),
        torch.randn(batch, CANDIDATE_STATE_DIM),
        torch.randn(batch, AGE_BUCKET_COUNT, P33_RELATION_DIM),
        torch.randn(batch, P33_GLOBAL_DIM),
    )


class P37BModelTests(unittest.TestCase):
    def test_paired_initialization_keeps_all_common_b0_b1_parameters_identical(self) -> None:
        b0 = _new_model("B0", seed=314159, device=torch.device("cpu"))
        b1 = _new_model("B1", seed=314159, device=torch.device("cpu"))
        audit0 = b0._p37b_initialization_audit
        audit1 = b1._p37b_initialization_audit
        self.assertEqual(audit0["scheme"], "paired-layerwise-v1")
        self.assertEqual(
            audit0["common_relation_input_sha256"],
            audit1["common_relation_input_sha256"],
        )
        self.assertEqual(
            audit0["shared_downstream_sha256"],
            audit1["shared_downstream_sha256"],
        )
        self.assertIsNone(audit0["auxiliary_relation_input_sha256"])
        self.assertIsNotNone(audit1["auxiliary_relation_input_sha256"])
        b0_first = b0.relation_encoder[0]
        b1_first = b1.relation_encoder[0]
        self.assertTrue(torch.equal(b0_first.weight[:, :6], b1_first.weight[:, :6]))
        self.assertTrue(torch.equal(b0_first.weight[:, 6:], b1_first.weight[:, 12:]))
        self.assertTrue(torch.equal(b0_first.bias, b1_first.bias))

    def test_age_buckets_are_mutually_exclusive_and_exhaustive(self) -> None:
        days = np.asarray([0, 7, 8, 28, 29, 84, 85, 365, np.nan])
        valid = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
        buckets = assign_age_buckets(days, valid)
        self.assertEqual(buckets.tolist(), [0, 0, 1, 1, 2, 2, 3, 3, -1])
        self.assertEqual(int(np.count_nonzero(buckets >= 0)), int(valid.sum()))
        for bucket in range(AGE_BUCKET_COUNT):
            self.assertTrue(np.all((buckets == bucket) <= valid))

    def test_invalid_history_age_is_rejected(self) -> None:
        self.assertEqual(
            assign_age_buckets(np.asarray([7.5]), np.asarray([True])).tolist(), [1]
        )
        with self.assertRaises(ValueError):
            assign_age_buckets(np.asarray([-1.0]), np.asarray([True]))

    def test_preprocessor_uses_finite_training_values_and_fixed_rules(self) -> None:
        specs = (
            FeatureSpec("available", "binary"),
            FeatureSpec("similarity", "continuous", availability="available"),
            FeatureSpec("events", "count"),
            FeatureSpec("rank", "rank"),
            FeatureSpec("flag", "binary"),
        )
        train = np.asarray(
            [
                [1, 1.0, 0, 1, 0],
                [1, 3.0, 3, 200, 1],
                [0, np.nan, 8, 100, 0],
            ],
            dtype=np.float64,
        )
        preprocessor = TrainOnlyPreprocessor(specs).fit(
            train, training_cutoff="2020-06-24"
        )
        transformed = preprocessor.transform(train)
        self.assertAlmostEqual(preprocessor.statistics["similarity"]["mean"], 2.0)
        self.assertAlmostEqual(preprocessor.statistics["similarity"]["std"], 1.0)
        self.assertEqual(transformed[2, 1], 0.0)
        expected_count_mean = float(np.mean(np.log1p([0.0, 3.0, 8.0])))
        self.assertAlmostEqual(preprocessor.statistics["events"]["mean"], expected_count_mean)
        self.assertEqual(transformed[0, 3], 0.0)
        self.assertEqual(transformed[1, 3], 1.0)
        self.assertAlmostEqual(transformed[2, 3], 99.0 / 199.0)
        self.assertTrue(np.array_equal(transformed[:, 0], train[:, 0]))
        self.assertTrue(np.array_equal(transformed[:, 4], train[:, 4]))

    def test_transform_data_cannot_change_train_only_statistics(self) -> None:
        values = np.zeros((4, len(CANDIDATE_STATE_SPECS)), dtype=np.float64)
        values[:, 0] = [0.1, 0.2, 0.3, 0.4]
        values[:, 1] = [1, 2, 3, 4]
        values[:, 2] = [0, 1, 2, 3]
        values[:, 3] = [1, 0, 0, 0]
        values[:, 4] = [0, 1, 1, 1]
        preprocessor = TrainOnlyPreprocessor(CANDIDATE_STATE_SPECS).fit(
            values, training_cutoff="2020-06-24"
        )
        before = copy.deepcopy(preprocessor.to_manifest())
        validation = values.copy()
        validation[:, 0] = 1_000_000.0
        validation[:, 2] = 1_000_000.0
        preprocessor.transform(validation)
        self.assertEqual(before, preprocessor.to_manifest())

    def test_missing_without_explicit_availability_is_rejected(self) -> None:
        preprocessor = TrainOnlyPreprocessor((FeatureSpec("x", "continuous"),))
        with self.assertRaises(ValueError):
            preprocessor.fit(
                np.asarray([[1.0], [np.nan]]), training_cutoff="2020-06-24"
            )

    def test_preprocessor_manifest_round_trip(self) -> None:
        rows = np.asarray(
            [
                [1, 2, 0.3, 0.2, 1, 0],
                [1, 4, 0.7, 0.5, 2, 1],
            ],
            dtype=np.float64,
        )
        original = TrainOnlyPreprocessor(M4_RELATION_SPECS).fit(
            rows, training_cutoff="2020-07-22"
        )
        restored = TrainOnlyPreprocessor.from_manifest(original.to_manifest())
        self.assertTrue(np.array_equal(original.transform(rows), restored.transform(rows)))

    def test_streaming_fit_matches_concatenated_population_statistics(self) -> None:
        specs = (FeatureSpec("value", "continuous"), FeatureSpec("count", "count"))
        first = np.asarray([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
        second = np.asarray([[4.0, 8.0], [8.0, 15.0]], dtype=np.float32)
        streamed = TrainOnlyPreprocessor(specs).fit_many(
            (first, second),
            training_cutoff="2020-06-24",
            chunk_rows=1,
            quantile_sample_size=100,
        )
        concatenated = TrainOnlyPreprocessor(specs).fit(
            np.concatenate([first, second]), training_cutoff="2020-06-24"
        )
        for name in ("value", "count"):
            self.assertAlmostEqual(
                streamed.statistics[name]["mean"],
                concatenated.statistics[name]["mean"],
            )
            self.assertAlmostEqual(
                streamed.statistics[name]["std"],
                concatenated.statistics[name]["std"],
            )
            self.assertEqual(
                streamed.statistics[name]["quantiles"],
                concatenated.statistics[name]["quantiles"],
            )

    def test_composite_fit_many_is_bounded_and_common_statistics_match(self) -> None:
        def source(offset: float, cutoff: str) -> dict[str, np.ndarray | str]:
            rows = 3
            m4 = np.zeros((rows, 4, 6), dtype=np.float32)
            m4[..., 0] = 1
            m4[..., 1] = np.arange(rows)[:, None]
            m4[..., 2:] = offset + np.arange(rows)[:, None, None]
            user = np.zeros((rows, 10), dtype=np.float32)
            user[:, :4] = np.arange(rows)[:, None]
            user[:, 4] = 2 + offset
            user[:, 5] = 0.5
            user[:, 6] = 0.25
            user[:, 7] = 1
            user[:, 8] = 0.4
            user[:, 9] = 2
            candidate = np.zeros((rows, 5), dtype=np.float32)
            candidate[:, 0] = offset + np.arange(rows)
            candidate[:, 1] = [1, 100, 200]
            candidate[:, 2] = [0, 2, 5]
            candidate[:, 3] = [1, 0, 0]
            candidate[:, 4] = [0, 1, 1]
            p33_relation = np.ones((rows, 4, 4), dtype=np.float32) * offset
            p33_global = np.ones((rows, 2), dtype=np.float32) * offset
            return {
                "m4_relation": m4,
                "user_state": user,
                "candidate_state": candidate,
                "p33_relation": p33_relation,
                "p33_global": p33_global,
                "cutoff": cutoff,
            }

        sources = [source(0.0, "2020-04-29"), source(3.0, "2020-05-27")]
        b0_sources = [
            {key: value for key, value in row.items() if not key.startswith("p33")}
            for row in sources
        ]
        b0 = FeaturePreprocessor("B0").fit_many(
            b0_sources, training_cutoff="2020-06-24", chunk_rows=1
        )
        b1 = FeaturePreprocessor("B1").fit_many(
            sources, training_cutoff="2020-06-24", chunk_rows=1
        )
        b0_manifest = b0.to_manifest()
        b1_manifest = b1.to_manifest()
        for group in ("user_state", "candidate_state"):
            self.assertEqual(
                b0_manifest["groups"][group]["statistics"],
                b1_manifest["groups"][group]["statistics"],
            )
        for spec in M4_RELATION_SPECS:
            self.assertEqual(
                b0_manifest["groups"]["relation"]["statistics"][spec.name],
                b1_manifest["groups"]["relation"]["statistics"][spec.name],
            )
        transformed = b1.transform(
            m4_relation=sources[0]["m4_relation"],
            user_state=sources[0]["user_state"],
            candidate_state=sources[0]["candidate_state"],
            p33_relation=sources[0]["p33_relation"],
            p33_global=sources[0]["p33_global"],
        )
        self.assertEqual(set(transformed), {
            "m4_relation", "user_state", "candidate_state", "p33_relation", "p33_global"
        })
        self.assertTrue(all(value.dtype == np.float32 for value in transformed.values()))

    def test_composite_fit_accepts_unique_users_but_transform_requires_alignment(self) -> None:
        m4 = np.zeros((4, 4, 6), dtype=np.float32)
        m4[..., 0] = 1
        candidates = np.asarray(
            [[0.1, 1, 0, 1, 0], [0.2, 2, 1, 0, 1], [0.3, 3, 2, 0, 1], [0.4, 4, 3, 0, 1]],
            dtype=np.float32,
        )
        unique_users = np.zeros((2, 10), dtype=np.float32)
        unique_users[:, 5] = 0.5
        preprocessor = FeaturePreprocessor("B0").fit(
            m4_relation=m4,
            user_state=unique_users,
            candidate_state=candidates,
            training_cutoff="2020-06-24",
        )
        with self.assertRaises(ValueError):
            preprocessor.transform(
                m4_relation=m4,
                user_state=unique_users,
                candidate_state=candidates,
            )

    def test_composite_fit_rejects_future_source_cutoff(self) -> None:
        m4 = np.zeros((1, 4, 6), dtype=np.float32)
        m4[..., 0] = 1
        users = np.zeros((1, 10), dtype=np.float32)
        candidates = np.asarray([[0.1, 1, 0, 1, 0]], dtype=np.float32)
        with self.assertRaises(RuntimeError):
            FeaturePreprocessor("B0").fit(
                m4_relation=m4,
                user_state=users,
                candidate_state=candidates,
                source_cutoff="2020-08-19",
                training_cutoff="2020-06-24",
            )

    def test_b0_b1_preprocessing_contract_diff_is_auxiliary_only(self) -> None:
        self.assertTrue(preprocessing_contracts_compatible())
        b0 = feature_specs_for_variant("B0")
        b1 = feature_specs_for_variant("B1")
        self.assertEqual(b0["user_state"], b1["user_state"])
        self.assertEqual(b0["candidate_state"], b1["candidate_state"])
        self.assertFalse(any("p33" in spec.name for group in b0.values() for spec in group))
        self.assertEqual(len(b1["relation"]) - len(b0["relation"]), 4)
        self.assertEqual(len(b1["p33_global"]), 2)

    def test_model_dimensions_are_fixed_and_equal_outside_auxiliary_encoder(self) -> None:
        b0 = TimeAwareHybrid("B0")
        b1 = TimeAwareHybrid("B1")
        self.assertEqual(b0.relation_input_dim, 10)
        self.assertEqual(b1.relation_input_dim, 16)
        self.assertEqual(b0.gate_input_dim, b1.gate_input_dim)
        self.assertEqual(b0.gate_input_dim, 35)
        self.assertEqual(b0.delta_input_dim, b1.delta_input_dim)
        self.assertEqual(b0.delta_input_dim, 16)

    def test_b1_global_auxiliary_is_broadcast_only_inside_relation_encoder(self) -> None:
        coarse, m4, user, candidate, p33_relation, p33_global = _model_inputs(batch=2)
        model = TimeAwareHybrid("B1")
        observed: list[torch.Tensor] = []

        def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            observed.append(inputs[0].detach().clone())

        handle = model.relation_encoder.register_forward_pre_hook(capture)
        try:
            model(
                coarse,
                m4,
                user,
                candidate,
                p33_relation=p33_relation,
                p33_global=p33_global,
            )
        finally:
            handle.remove()
        relation_input = observed[0]
        self.assertEqual(relation_input.shape, (2, 4, 16))
        self.assertTrue(torch.equal(relation_input[..., :6], m4))
        self.assertTrue(torch.equal(relation_input[..., 6:10], p33_relation))
        self.assertTrue(
            torch.equal(relation_input[..., 10:12], p33_global[:, None, :].expand(-1, 4, -1))
        )
        expected_age = torch.eye(4).unsqueeze(0).expand(2, -1, -1)
        self.assertTrue(torch.equal(relation_input[..., 12:16], expected_age))

    def test_zero_initialized_residual_starts_at_m4_score(self) -> None:
        coarse, m4, user, candidate, _, _ = _model_inputs()
        model = TimeAwareHybrid("B0")
        output = model(coarse, m4, user, candidate)
        final_layer = model.delta_scorer[-1]
        self.assertTrue(torch.count_nonzero(final_layer.weight) == 0)
        self.assertTrue(torch.count_nonzero(final_layer.bias) == 0)
        self.assertTrue(torch.equal(output.delta, torch.zeros_like(output.delta)))
        self.assertTrue(torch.equal(output.score, coarse))

    def test_alpha_over_four_age_experts_and_null_sums_to_one(self) -> None:
        coarse, m4, user, candidate, p33_relation, p33_global = _model_inputs()
        for model, kwargs in (
            (TimeAwareHybrid("B0"), {}),
            (
                TimeAwareHybrid("B1"),
                {"p33_relation": p33_relation, "p33_global": p33_global},
            ),
        ):
            output = model(coarse, m4, user, candidate, **kwargs)
            self.assertEqual(output.alpha.shape, (len(coarse), 5))
            self.assertTrue(
                torch.allclose(output.alpha.sum(dim=1), torch.ones(len(coarse)), atol=1e-6)
            )

    def test_null_expert_uses_state_but_zero_relation_and_age_inputs(self) -> None:
        coarse, m4, user, candidate, _, _ = _model_inputs(batch=2)
        model = TimeAwareHybrid("B0")
        observed: list[torch.Tensor] = []

        def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            observed.append(inputs[0].detach().clone())

        handle = model.gate.register_forward_pre_hook(capture)
        try:
            model(coarse, m4, user, candidate)
        finally:
            handle.remove()
        self.assertEqual(len(observed), 2)
        null_input = observed[1]
        self.assertEqual(null_input.shape, (2, 35))
        self.assertTrue(torch.equal(null_input[:, :16], torch.zeros(2, 16)))
        self.assertTrue(torch.equal(null_input[:, 16:26], user))
        self.assertTrue(torch.equal(null_input[:, 26:31], candidate))
        self.assertTrue(torch.equal(null_input[:, 31:35], torch.zeros(2, 4)))

    def test_b0_rejects_and_b1_requires_p33_fields(self) -> None:
        coarse, m4, user, candidate, p33_relation, p33_global = _model_inputs()
        with self.assertRaises(ValueError):
            TimeAwareHybrid("B0")(
                coarse,
                m4,
                user,
                candidate,
                p33_relation=p33_relation,
                p33_global=p33_global,
            )
        with self.assertRaises(ValueError):
            TimeAwareHybrid("B1")(coarse, m4, user, candidate)

    def test_p32_sampler_is_reused_with_exact_parity(self) -> None:
        self.assertIs(sample_training_pairs, sample_same_user_pairs)
        users = np.zeros(200, dtype=np.int32)
        ranks = np.arange(1, 201, dtype=np.uint16)
        targets = np.zeros(200, dtype=np.uint8)
        targets[[4, 90]] = 1
        expected_pairs, expected_audit = sample_same_user_pairs(
            user_index=users, rank=ranks, target=targets, cutoff="2020-06-24"
        )
        observed_pairs, observed_audit = sample_training_pairs(
            user_index=users, rank=ranks, target=targets, cutoff="2020-06-24"
        )
        self.assertTrue(np.array_equal(expected_pairs, observed_pairs))
        self.assertEqual(expected_audit, observed_audit)

    def test_cutoff_binding_rejects_future_p33_student_and_future_behavior(self) -> None:
        audit = validate_cutoff_safety(
            example_cutoff="2020-06-24",
            user_state_cutoff="2020-06-24",
            m4_embedding_cutoff="2020-06-24",
            p33_embedding_cutoff="2020-06-24",
            behavior_max_dates={
                "user_state": "2020-06-23",
                "m4_embedding": "2020-06-23",
                "p33_embedding": "2020-06-23",
            },
            require_p33=True,
        )
        self.assertTrue(audit["asset_cutoff_exact_parity"])
        with self.assertRaises(RuntimeError):
            validate_cutoff_safety(
                example_cutoff="2020-06-24",
                user_state_cutoff="2020-06-24",
                m4_embedding_cutoff="2020-06-24",
                p33_embedding_cutoff="2020-08-19",
                require_p33=True,
            )
        with self.assertRaises(RuntimeError):
            validate_cutoff_safety(
                example_cutoff="2020-06-24",
                user_state_cutoff="2020-06-24",
                m4_embedding_cutoff="2020-06-24",
                behavior_max_dates={"user_state": "2020-06-24"},
            )

    def test_final_week_is_rejected_everywhere(self) -> None:
        safe_cutoff("2020-08-19")
        with self.assertRaises(RuntimeError):
            safe_cutoff("2020-09-16")
        with self.assertRaises(RuntimeError):
            TrainOnlyPreprocessor(CANDIDATE_STATE_SPECS).fit(
                np.asarray([[0.1, 1, 0, 1, 0]], dtype=np.float64),
                training_cutoff="2020-09-16",
            )


if __name__ == "__main__":
    unittest.main()
