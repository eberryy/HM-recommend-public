from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Literal, Mapping, NamedTuple, Sequence

import numpy as np
import torch
from torch import nn

from .p3_model import sample_same_user_pairs


FINAL_CUTOFF = "2020-09-16"
K0 = 200
AGE_BUCKET_NAMES = ("age_0_7", "age_8_28", "age_29_84", "age_over_84")
AGE_BUCKET_COUNT = len(AGE_BUCKET_NAMES)

M4_RELATION_DIM = 6
P33_RELATION_DIM = 4
P33_GLOBAL_DIM = 2
USER_STATE_DIM = 10
CANDIDATE_STATE_DIM = 5
RELATION_HIDDEN_DIM = 16
RESIDUAL_HIDDEN_DIM = 32

Variant = Literal["B0", "B1"]
FeatureKind = Literal["continuous", "count", "rank", "binary"]


def _as_date(value: str, *, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO date, got {value!r}") from exc


def safe_cutoff(cutoff: str) -> date:
    """Reject the final Kaggle week and any later date."""
    parsed = _as_date(cutoff, label="cutoff")
    if parsed >= _as_date(FINAL_CUTOFF, label="final cutoff"):
        raise RuntimeError(f"P3.7B refuses final or later cutoff: {cutoff}")
    return parsed


def validate_cutoff_safety(
    *,
    example_cutoff: str,
    user_state_cutoff: str,
    m4_embedding_cutoff: str,
    p33_embedding_cutoff: str | None = None,
    behavior_max_dates: Mapping[str, str] | None = None,
    require_p33: bool = False,
) -> dict[str, Any]:
    """Fail closed unless every feature asset is bound to the example cutoff.

    ``behavior_max_dates`` is optional because some manifests prove temporal
    safety through hashes and training-cutoff lineage instead of a maximum
    event date.  When supplied, each date must be strictly before the example
    cutoff.  A P3.3 asset from a later (or merely different) cutoff is never
    accepted for an earlier example.
    """

    parsed_example = safe_cutoff(example_cutoff)
    asset_cutoffs: dict[str, str] = {
        "user_state": user_state_cutoff,
        "m4_embedding": m4_embedding_cutoff,
    }
    if p33_embedding_cutoff is not None:
        asset_cutoffs["p33_embedding"] = p33_embedding_cutoff
    if require_p33 and p33_embedding_cutoff is None:
        raise RuntimeError("B1 requires a cutoff-aligned P3.3 Student")
    if not require_p33 and p33_embedding_cutoff is not None:
        raise RuntimeError("B0 must not bind a P3.3 Student")

    drift = {
        name: cutoff
        for name, cutoff in asset_cutoffs.items()
        if _as_date(cutoff, label=f"{name} cutoff") != parsed_example
    }
    if drift:
        raise RuntimeError(
            f"P3.7B cutoff binding mismatch for {example_cutoff}: {drift}"
        )

    checked_max_dates: dict[str, str] = {}
    for name, maximum in (behavior_max_dates or {}).items():
        parsed_maximum = _as_date(maximum, label=f"{name} behavior maximum")
        if parsed_maximum >= parsed_example:
            raise RuntimeError(
                f"{name} behavior maximum {maximum} is not strictly before "
                f"example cutoff {example_cutoff}"
            )
        checked_max_dates[name] = maximum
    return {
        "example_cutoff": example_cutoff,
        "asset_cutoffs": asset_cutoffs,
        "asset_cutoff_exact_parity": True,
        "behavior_max_dates": checked_max_dates,
        "behavior_strictly_before_cutoff": True,
        "final_week": "not_run",
    }


def assign_age_buckets(
    days_since_purchase: np.ndarray | Sequence[float],
    valid_mask: np.ndarray | Sequence[bool] | None = None,
) -> np.ndarray:
    """Assign valid calendar-day ages to exactly one fixed P3.7B bucket.

    The returned integer array uses 0, 1, 2, and 3 for 0--7, over-7--28,
    over-28--84, and over-84 days respectively. Invalid padded history
    positions are -1. The open lower boundaries keep the assignment exhaustive
    even if an upstream asset stores a non-integer day value.
    """

    days = np.asarray(days_since_purchase, dtype=np.float64)
    if valid_mask is None:
        valid = np.ones(days.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != days.shape:
            raise ValueError("valid_mask and days_since_purchase shapes differ")
    valid_days = days[valid]
    if not np.isfinite(valid_days).all():
        raise ValueError("valid history ages must be finite")
    if np.any(valid_days < 0):
        raise ValueError("valid history ages must be non-negative")
    buckets = np.full(days.shape, -1, dtype=np.int8)
    buckets[valid & (days <= 7)] = 0
    buckets[valid & (days > 7) & (days <= 28)] = 1
    buckets[valid & (days > 28) & (days <= 84)] = 2
    buckets[valid & (days > 84)] = 3
    if int(np.count_nonzero(buckets >= 0)) != int(np.count_nonzero(valid)):
        raise RuntimeError("not every valid history age was assigned exactly once")
    return buckets


@dataclass(frozen=True)
class FeatureSpec:
    """One raw feature and its fixed normalization rule."""

    name: str
    kind: FeatureKind
    availability: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("feature name must not be empty")
        if self.kind not in {"continuous", "count", "rank", "binary"}:
            raise ValueError(f"unsupported feature kind: {self.kind}")
        if self.availability is not None and self.kind not in {"continuous", "count"}:
            raise ValueError("availability is only valid for continuous/count features")


M4_RELATION_SPECS = (
    FeatureSpec("bucket_present", "binary"),
    FeatureSpec("history_count", "count"),
    FeatureSpec("max_cosine", "continuous", availability="bucket_present"),
    FeatureSpec("top3_mean_cosine", "continuous", availability="bucket_present"),
    FeatureSpec("same_product_type_history_count", "count"),
    FeatureSpec("same_garment_group_history_count", "count"),
)
P33_RELATION_SPECS = (
    FeatureSpec("p33_max_cosine", "continuous", availability="bucket_present"),
    FeatureSpec("p33_top3_mean_cosine", "continuous", availability="bucket_present"),
    FeatureSpec("p33_minus_m4_max_cosine", "continuous", availability="bucket_present"),
    FeatureSpec("p33_minus_m4_top3_mean", "continuous", availability="bucket_present"),
)
P33_GLOBAL_SPECS = (
    FeatureSpec("p33_fixed_decay_score", "continuous"),
    FeatureSpec("p33_minus_m4_fixed_decay_score", "continuous"),
)
USER_STATE_SPECS = (
    FeatureSpec("history_count_0_7", "count"),
    FeatureSpec("history_count_8_28", "count"),
    FeatureSpec("history_count_29_84", "count"),
    FeatureSpec("history_count_over_84", "count"),
    FeatureSpec("days_since_last_purchase", "continuous"),
    FeatureSpec("recent_0_28_purchase_share", "continuous"),
    FeatureSpec(
        "recent_vs_older_profile_cosine",
        "continuous",
        availability="recent_vs_older_profile_available",
    ),
    FeatureSpec("recent_vs_older_profile_available", "binary"),
    FeatureSpec("recent_28d_product_type_entropy", "continuous"),
    FeatureSpec("recent_28d_distinct_product_types", "count"),
)
CANDIDATE_STATE_SPECS = (
    FeatureSpec("m4_coarse_score", "continuous"),
    FeatureSpec("m4_coarse_rank", "rank"),
    FeatureSpec("interaction_count_before_cutoff", "count"),
    FeatureSpec("strict_cold_flag", "binary"),
    FeatureSpec("sparse1_5_flag", "binary"),
)


def feature_specs_for_variant(variant: Variant) -> dict[str, tuple[FeatureSpec, ...]]:
    """Return the frozen B0/B1 raw-feature normalization contract."""

    if variant == "B0":
        relation = M4_RELATION_SPECS
        p33_global: tuple[FeatureSpec, ...] = ()
    elif variant == "B1":
        relation = M4_RELATION_SPECS + P33_RELATION_SPECS
        p33_global = P33_GLOBAL_SPECS
    else:
        raise ValueError(f"unsupported P3.7B variant: {variant}")
    return {
        "relation": relation,
        "user_state": USER_STATE_SPECS,
        "candidate_state": CANDIDATE_STATE_SPECS,
        "p33_global": p33_global,
    }


def preprocessing_contracts_compatible() -> bool:
    """Confirm B1 differs from B0 only by the declared P3.3 inputs."""

    b0 = feature_specs_for_variant("B0")
    b1 = feature_specs_for_variant("B1")
    return (
        b0["user_state"] == b1["user_state"]
        and b0["candidate_state"] == b1["candidate_state"]
        and b1["relation"][: len(b0["relation"])] == b0["relation"]
        and b1["relation"][len(b0["relation"]) :] == P33_RELATION_SPECS
        and b0["p33_global"] == ()
        and b1["p33_global"] == P33_GLOBAL_SPECS
    )


class TrainOnlyPreprocessor:
    """Fit fixed feature transforms on one outer-chain training cutoff only."""

    def __init__(
        self,
        specs: Sequence[FeatureSpec],
        *,
        rank_budget: int = K0,
        epsilon: float = 1e-12,
    ) -> None:
        self.specs = tuple(specs)
        self.rank_budget = int(rank_budget)
        self.epsilon = float(epsilon)
        if not self.specs:
            raise ValueError("preprocessor requires at least one feature")
        names = [spec.name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        if self.rank_budget <= 1:
            raise ValueError("rank_budget must be greater than one")
        by_name = {spec.name: spec for spec in self.specs}
        for spec in self.specs:
            if spec.availability is not None:
                availability = by_name.get(spec.availability)
                if availability is None:
                    raise ValueError(
                        f"{spec.name} availability feature {spec.availability!r} is absent"
                    )
                if availability.kind != "binary":
                    raise ValueError(
                        f"{spec.name} availability feature must be binary"
                    )
        self.training_cutoff: str | None = None
        self.training_rows: int | None = None
        self.statistics: dict[str, dict[str, Any]] = {}

    def _matrix(self, values: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(values)
        if matrix.ndim < 2 or matrix.shape[-1] != len(self.specs):
            raise ValueError(
                f"expected [...,{len(self.specs)}] feature matrix, got {matrix.shape}"
            )
        if not np.issubdtype(matrix.dtype, np.number):
            raise ValueError("feature matrix must be numeric")
        return matrix

    def _validate(self, matrix: np.ndarray) -> None:
        flattened = matrix.reshape(-1, matrix.shape[-1])
        by_name = {spec.name: index for index, spec in enumerate(self.specs)}
        for index, spec in enumerate(self.specs):
            column = flattened[:, index]
            finite = np.isfinite(column)
            if spec.kind in {"binary", "rank"} and not finite.all():
                raise ValueError(f"{spec.name} must not contain missing values")
            if spec.kind == "binary" and not np.isin(column, (0.0, 1.0)).all():
                raise ValueError(f"{spec.name} must contain only 0/1")
            if spec.kind == "rank":
                if np.any(column < 1) or np.any(column > self.rank_budget):
                    raise ValueError(
                        f"{spec.name} must be in [1,{self.rank_budget}]"
                    )
                if np.any(column != np.floor(column)):
                    raise ValueError(f"{spec.name} must contain integer ranks")
            if spec.kind == "count" and np.any(column[finite] < 0):
                raise ValueError(f"{spec.name} must contain non-negative counts")
            if np.any(~finite):
                if spec.availability is None:
                    raise ValueError(
                        f"{spec.name} contains missing values without an availability flag"
                    )
                available = flattened[:, by_name[spec.availability]]
                if np.any(available[~finite] != 0):
                    raise ValueError(
                        f"{spec.name} is missing where {spec.availability} is not zero"
                    )

    @staticmethod
    def _pre_stat_values(column: np.ndarray, kind: FeatureKind) -> np.ndarray:
        finite = np.isfinite(column)
        selected = column[finite]
        if kind == "count":
            selected = np.log1p(selected)
        return selected

    def fit(
        self,
        values: np.ndarray | Sequence[Sequence[float]],
        *,
        training_cutoff: str,
    ) -> TrainOnlyPreprocessor:
        return self.fit_many((values,), training_cutoff=training_cutoff)

    def fit_many(
        self,
        value_batches: Iterable[np.ndarray | Sequence[Sequence[float]]],
        *,
        training_cutoff: str,
        chunk_rows: int = 262_144,
        quantile_sample_size: int = 200_000,
    ) -> TrainOnlyPreprocessor:
        """Fit exact population mean/std and bounded deterministic quantiles.

        Arrays are scanned in bounded chunks and are never concatenated. Mean
        and population variance are exact up to float64 arithmetic. Quantiles
        are exact when the finite row count does not exceed
        ``quantile_sample_size``; otherwise a deterministic bottom-hash sample
        is used and explicitly identified in the manifest.
        """

        safe_cutoff(training_cutoff)
        if chunk_rows < 1 or quantile_sample_size < 1:
            raise ValueError("chunk_rows and quantile_sample_size must be positive")
        accumulators = {
            spec.name: {
                "count": 0,
                "mean": 0.0,
                "m2": 0.0,
                "sample_values": np.empty(0, dtype=np.float64),
                "sample_priorities": np.empty(0, dtype=np.uint64),
            }
            for spec in self.specs
            if spec.kind in {"continuous", "count"}
        }
        total_rows = 0
        batch_count = 0
        for raw_matrix in value_batches:
            matrix = self._matrix(raw_matrix)
            flattened = matrix.reshape(-1, matrix.shape[-1])
            batch_count += 1
            for start in range(0, len(flattened), chunk_rows):
                chunk = flattened[start : start + chunk_rows]
                self._validate(chunk)
                row_ordinals = np.arange(
                    total_rows, total_rows + len(chunk), dtype=np.uint64
                )
                priorities = _splitmix64(row_ordinals)
                for index, spec in enumerate(self.specs):
                    if spec.kind not in {"continuous", "count"}:
                        continue
                    raw_column = np.asarray(chunk[:, index], dtype=np.float64)
                    finite = np.isfinite(raw_column)
                    selected = raw_column[finite]
                    if spec.kind == "count":
                        selected = np.log1p(selected)
                    if not len(selected):
                        continue
                    state = accumulators[spec.name]
                    batch_n = int(len(selected))
                    batch_mean = float(np.mean(selected, dtype=np.float64))
                    batch_m2 = float(
                        np.sum((selected - batch_mean) ** 2, dtype=np.float64)
                    )
                    previous_n = int(state["count"])
                    if previous_n == 0:
                        state["mean"] = batch_mean
                        state["m2"] = batch_m2
                    else:
                        combined_n = previous_n + batch_n
                        delta = batch_mean - float(state["mean"])
                        state["mean"] = float(state["mean"]) + delta * batch_n / combined_n
                        state["m2"] = (
                            float(state["m2"])
                            + batch_m2
                            + delta * delta * previous_n * batch_n / combined_n
                        )
                    state["count"] = previous_n + batch_n

                    combined_values = np.concatenate(
                        [state["sample_values"], selected]
                    )
                    combined_priorities = np.concatenate(
                        [state["sample_priorities"], priorities[finite]]
                    )
                    if len(combined_values) > quantile_sample_size:
                        keep = np.argpartition(
                            combined_priorities, quantile_sample_size - 1
                        )[:quantile_sample_size]
                        combined_values = combined_values[keep]
                        combined_priorities = combined_priorities[keep]
                    state["sample_values"] = combined_values
                    state["sample_priorities"] = combined_priorities
                total_rows += len(chunk)
        if batch_count == 0 or total_rows == 0:
            raise ValueError("fit_many requires at least one non-empty feature matrix")

        statistics: dict[str, dict[str, Any]] = {}
        for index, spec in enumerate(self.specs):
            if spec.kind in {"continuous", "count"}:
                state = accumulators[spec.name]
                finite_rows = int(state["count"])
                if finite_rows == 0:
                    raise ValueError(f"{spec.name} has no finite training values")
                mean = float(state["mean"])
                observed_std = float(np.sqrt(float(state["m2"]) / finite_rows))
                std = observed_std if observed_std > self.epsilon else 1.0
                sampled = np.asarray(state["sample_values"], dtype=np.float64)
                quantiles = np.quantile(sampled, [0.0, 0.25, 0.5, 0.75, 1.0])
                statistics[spec.name] = {
                    "mean": mean,
                    "std": std,
                    "observed_std": observed_std,
                    "constant_column": observed_std <= self.epsilon,
                    "finite_rows": finite_rows,
                    "missing_rows": int(total_rows - finite_rows),
                    "quantiles": {
                        "q0": float(quantiles[0]),
                        "q25": float(quantiles[1]),
                        "q50": float(quantiles[2]),
                        "q75": float(quantiles[3]),
                        "q100": float(quantiles[4]),
                    },
                    "quantile_method": (
                        "exact_all_finite_values"
                        if finite_rows <= quantile_sample_size
                        else "deterministic_bottom_hash_sample"
                    ),
                    "quantile_sample_rows": int(len(sampled)),
                }
            elif spec.kind == "rank":
                statistics[spec.name] = {
                    "fixed_min_rank": 1,
                    "fixed_max_rank": self.rank_budget,
                    "formula": "(rank - 1) / (rank_budget - 1)",
                }
            else:
                statistics[spec.name] = {
                    "allowed_values": [0, 1],
                    "standardized": False,
                }
        self.training_cutoff = training_cutoff
        self.training_rows = total_rows
        self.statistics = statistics
        return self

    @property
    def fitted(self) -> bool:
        return self.training_cutoff is not None and bool(self.statistics)

    def transform(
        self, values: np.ndarray | Sequence[Sequence[float]]
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("preprocessor must be fit on training data first")
        matrix = self._matrix(values)
        self._validate(matrix)
        output = np.zeros(matrix.shape, dtype=np.float32)
        for index, spec in enumerate(self.specs):
            column = matrix[..., index]
            finite = np.isfinite(column)
            if spec.kind in {"continuous", "count"}:
                transformed = column[finite]
                if spec.kind == "count":
                    transformed = np.log1p(transformed)
                stats = self.statistics[spec.name]
                output_column = output[..., index]
                output_column[finite] = (
                    (transformed - float(stats["mean"])) / float(stats["std"])
                ).astype(np.float32)
                # Missing values deliberately remain zero in normalized space.
            elif spec.kind == "rank":
                output[..., index] = (
                    (column - 1.0) / float(self.rank_budget - 1)
                ).astype(np.float32)
            else:
                output[..., index] = column.astype(np.float32)
        return output

    def to_manifest(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("cannot serialize an unfitted preprocessor")
        return {
            "schema_version": "phase3-p3.7b-train-only-preprocessor-v1",
            "training_cutoff": self.training_cutoff,
            "training_rows": self.training_rows,
            "rank_budget": self.rank_budget,
            "epsilon": self.epsilon,
            "features": [
                {
                    "name": spec.name,
                    "kind": spec.kind,
                    "availability": spec.availability,
                }
                for spec in self.specs
            ],
            "statistics": copy.deepcopy(self.statistics),
            "fit_scope": "current outer-chain training cutoff only",
            "missing_rule": "normalized zero plus explicit availability flag",
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> TrainOnlyPreprocessor:
        if manifest.get("schema_version") != "phase3-p3.7b-train-only-preprocessor-v1":
            raise ValueError("unsupported preprocessing manifest")
        specs = tuple(
            FeatureSpec(
                str(row["name"]),
                str(row["kind"]),  # type: ignore[arg-type]
                None if row.get("availability") is None else str(row["availability"]),
            )
            for row in manifest["features"]
        )
        result = cls(
            specs,
            rank_budget=int(manifest["rank_budget"]),
            epsilon=float(manifest["epsilon"]),
        )
        result.training_cutoff = str(manifest["training_cutoff"])
        safe_cutoff(result.training_cutoff)
        result.training_rows = int(manifest["training_rows"])
        result.statistics = copy.deepcopy(dict(manifest["statistics"]))
        expected_names = {spec.name for spec in specs}
        if set(result.statistics) != expected_names:
            raise ValueError("preprocessing manifest statistics do not match feature names")
        return result


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Deterministic vectorized 64-bit mixing for bounded quantile sampling."""

    with np.errstate(over="ignore"):
        mixed = values.astype(np.uint64, copy=True) + np.uint64(0x9E3779B97F4A7C15)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return mixed ^ (mixed >> np.uint64(31))


class FeaturePreprocessor:
    """Composite train-only preprocessing for one fixed B0 or B1 contract."""

    def __init__(self, variant: Variant, *, rank_budget: int = K0) -> None:
        self.variant = variant
        specs = feature_specs_for_variant(variant)
        self.relation = TrainOnlyPreprocessor(specs["relation"], rank_budget=rank_budget)
        self.user_state = TrainOnlyPreprocessor(specs["user_state"], rank_budget=rank_budget)
        self.candidate_state = TrainOnlyPreprocessor(
            specs["candidate_state"], rank_budget=rank_budget
        )
        self.p33_global = (
            TrainOnlyPreprocessor(specs["p33_global"], rank_budget=rank_budget)
            if specs["p33_global"]
            else None
        )
        self.training_cutoff: str | None = None
        self.source_cutoffs: list[str] = []

    def _validate_source(
        self, source: Mapping[str, Any], *, require_candidate_aligned_user_state: bool
    ) -> int:
        required = {"m4_relation", "user_state", "candidate_state"}
        if self.variant == "B1":
            required.update({"p33_relation", "p33_global"})
        missing = required - set(source)
        if missing:
            raise ValueError(f"{self.variant} raw feature source is missing {sorted(missing)}")
        if self.variant == "B0" and ({"p33_relation", "p33_global"} & set(source)):
            raise ValueError("B0 raw feature source must contain zero P3.3 fields")
        m4 = np.asarray(source["m4_relation"])
        users = np.asarray(source["user_state"])
        candidates = np.asarray(source["candidate_state"])
        if m4.ndim != 3 or m4.shape[1:] != (AGE_BUCKET_COUNT, M4_RELATION_DIM):
            raise ValueError(f"m4_relation must be [N,4,6], got {m4.shape}")
        rows = len(m4)
        if users.ndim != 2 or users.shape[1] != USER_STATE_DIM:
            raise ValueError(f"user_state must be [U,10] or [N,10], got {users.shape}")
        if require_candidate_aligned_user_state and len(users) != rows:
            raise ValueError(
                f"transform user_state must be candidate-aligned [N,10], got {users.shape} "
                f"for N={rows}"
            )
        if candidates.shape != (rows, CANDIDATE_STATE_DIM):
            raise ValueError(f"candidate_state must be [N,5], got {candidates.shape}")
        if self.variant == "B1":
            relation = np.asarray(source["p33_relation"])
            global_features = np.asarray(source["p33_global"])
            if relation.shape != (rows, AGE_BUCKET_COUNT, P33_RELATION_DIM):
                raise ValueError(f"p33_relation must be [N,4,4], got {relation.shape}")
            if global_features.shape != (rows, P33_GLOBAL_DIM):
                raise ValueError(f"p33_global must be [N,2], got {global_features.shape}")
        return rows

    @staticmethod
    def _source(
        *,
        m4_relation: np.ndarray,
        user_state: np.ndarray,
        candidate_state: np.ndarray,
        p33_relation: np.ndarray | None,
        p33_global: np.ndarray | None,
        source_cutoff: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "m4_relation": m4_relation,
            "user_state": user_state,
            "candidate_state": candidate_state,
        }
        if p33_relation is not None:
            result["p33_relation"] = p33_relation
        if p33_global is not None:
            result["p33_global"] = p33_global
        if source_cutoff is not None:
            result["cutoff"] = source_cutoff
        return result

    def fit(
        self,
        *,
        m4_relation: np.ndarray,
        user_state: np.ndarray,
        candidate_state: np.ndarray,
        training_cutoff: str,
        p33_relation: np.ndarray | None = None,
        p33_global: np.ndarray | None = None,
        source_cutoff: str | None = None,
    ) -> FeaturePreprocessor:
        return self.fit_many(
            (
                self._source(
                    m4_relation=m4_relation,
                    user_state=user_state,
                    candidate_state=candidate_state,
                    p33_relation=p33_relation,
                    p33_global=p33_global,
                    source_cutoff=source_cutoff,
                ),
            ),
            training_cutoff=training_cutoff,
        )

    @classmethod
    def fit_from_arrays(
        cls,
        variant: Variant,
        *,
        m4_relation: np.ndarray,
        user_state: np.ndarray,
        candidate_state: np.ndarray,
        training_cutoff: str,
        p33_relation: np.ndarray | None = None,
        p33_global: np.ndarray | None = None,
        source_cutoff: str | None = None,
    ) -> FeaturePreprocessor:
        return cls(variant).fit(
            m4_relation=m4_relation,
            user_state=user_state,
            candidate_state=candidate_state,
            training_cutoff=training_cutoff,
            p33_relation=p33_relation,
            p33_global=p33_global,
            source_cutoff=source_cutoff,
        )

    def fit_many(
        self,
        sources: Iterable[Mapping[str, Any]],
        *,
        training_cutoff: str,
        chunk_rows: int = 262_144,
        quantile_sample_size: int = 200_000,
    ) -> FeaturePreprocessor:
        parsed_training_cutoff = safe_cutoff(training_cutoff)
        source_list = list(sources)
        if not source_list:
            raise ValueError("fit_many requires at least one raw feature source")
        for source in source_list:
            self._validate_source(source, require_candidate_aligned_user_state=False)
        self.source_cutoffs = [
            str(source["cutoff"]) for source in source_list if source.get("cutoff") is not None
        ]
        for cutoff in self.source_cutoffs:
            parsed_source = safe_cutoff(cutoff)
            if parsed_source > parsed_training_cutoff:
                raise RuntimeError(
                    f"preprocessing source cutoff {cutoff} is after training cutoff "
                    f"{training_cutoff}"
                )

        def relation_batches() -> Iterable[np.ndarray]:
            for source in source_list:
                m4 = np.asarray(source["m4_relation"])
                p33 = np.asarray(source["p33_relation"]) if self.variant == "B1" else None
                for start in range(0, len(m4), chunk_rows):
                    end = min(start + chunk_rows, len(m4))
                    if p33 is None:
                        yield m4[start:end]
                    else:
                        yield np.concatenate([m4[start:end], p33[start:end]], axis=-1)

        self.relation.fit_many(
            relation_batches(),
            training_cutoff=training_cutoff,
            chunk_rows=chunk_rows * AGE_BUCKET_COUNT,
            quantile_sample_size=quantile_sample_size,
        )
        self.user_state.fit_many(
            (np.asarray(source["user_state"]) for source in source_list),
            training_cutoff=training_cutoff,
            chunk_rows=chunk_rows,
            quantile_sample_size=quantile_sample_size,
        )
        self.candidate_state.fit_many(
            (np.asarray(source["candidate_state"]) for source in source_list),
            training_cutoff=training_cutoff,
            chunk_rows=chunk_rows,
            quantile_sample_size=quantile_sample_size,
        )
        if self.p33_global is not None:
            self.p33_global.fit_many(
                (np.asarray(source["p33_global"]) for source in source_list),
                training_cutoff=training_cutoff,
                chunk_rows=chunk_rows,
                quantile_sample_size=quantile_sample_size,
            )
        self.training_cutoff = training_cutoff
        return self

    def transform(
        self,
        *,
        m4_relation: np.ndarray,
        user_state: np.ndarray,
        candidate_state: np.ndarray,
        p33_relation: np.ndarray | None = None,
        p33_global: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        source = self._source(
            m4_relation=m4_relation,
            user_state=user_state,
            candidate_state=candidate_state,
            p33_relation=p33_relation,
            p33_global=p33_global,
        )
        self._validate_source(source, require_candidate_aligned_user_state=True)
        if self.variant == "B1":
            assert p33_relation is not None and p33_global is not None
            combined_relation = np.concatenate([m4_relation, p33_relation], axis=-1)
        else:
            combined_relation = m4_relation
        normalized_relation = self.relation.transform(combined_relation)
        output = {
            "m4_relation": normalized_relation[..., :M4_RELATION_DIM],
            "user_state": self.user_state.transform(user_state),
            "candidate_state": self.candidate_state.transform(candidate_state),
        }
        if self.variant == "B1":
            assert self.p33_global is not None and p33_global is not None
            output["p33_relation"] = normalized_relation[..., M4_RELATION_DIM:]
            output["p33_global"] = self.p33_global.transform(p33_global)
        if not all(value.dtype == np.float32 for value in output.values()):
            raise RuntimeError("normalized P3.7B features must be float32")
        return output

    def to_manifest(self) -> dict[str, Any]:
        if self.training_cutoff is None:
            raise RuntimeError("cannot serialize an unfitted composite preprocessor")
        return {
            "schema_version": "phase3-p3.7b-feature-preprocessor-v1",
            "variant": self.variant,
            "training_cutoff": self.training_cutoff,
            "source_cutoffs": list(self.source_cutoffs),
            "groups": {
                "relation": self.relation.to_manifest(),
                "user_state": self.user_state.to_manifest(),
                "candidate_state": self.candidate_state.to_manifest(),
                "p33_global": (
                    self.p33_global.to_manifest() if self.p33_global is not None else None
                ),
            },
            "common_field_rule": (
                "B0/B1 M4 relation, user-state, and candidate-state columns use "
                "identical per-column training statistics"
            ),
        }

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> FeaturePreprocessor:
        if manifest.get("schema_version") != "phase3-p3.7b-feature-preprocessor-v1":
            raise ValueError("unsupported composite preprocessing manifest")
        variant = str(manifest["variant"])
        if variant not in {"B0", "B1"}:
            raise ValueError(f"invalid preprocessing variant: {variant}")
        result = cls(variant)  # type: ignore[arg-type]
        groups = manifest["groups"]
        result.relation = TrainOnlyPreprocessor.from_manifest(groups["relation"])
        result.user_state = TrainOnlyPreprocessor.from_manifest(groups["user_state"])
        result.candidate_state = TrainOnlyPreprocessor.from_manifest(groups["candidate_state"])
        result.p33_global = (
            None
            if groups.get("p33_global") is None
            else TrainOnlyPreprocessor.from_manifest(groups["p33_global"])
        )
        result.training_cutoff = str(manifest["training_cutoff"])
        safe_cutoff(result.training_cutoff)
        result.source_cutoffs = [str(value) for value in manifest.get("source_cutoffs", [])]
        return result


class HybridOutput(NamedTuple):
    score: torch.Tensor
    alpha: torch.Tensor
    delta: torch.Tensor
    s_time: torch.Tensor

    @property
    def time_evidence(self) -> torch.Tensor:
        """Readable alias for the contract name ``s_time``."""

        return self.s_time


class TimeAwareHybrid(nn.Module):
    """Fixed P3.7B four-timescale residual reranker.

    B0 consumes six M4 relation features per age bucket. B1 additionally
    consumes four P3.3 relation features per bucket and two candidate-level
    P3.3 scores broadcast to all four buckets. Candidate state remains five
    dimensional in both variants, which keeps the shared gate and residual
    scorer at exactly the same capacity.
    """

    def __init__(self, variant: Variant) -> None:
        super().__init__()
        if variant not in {"B0", "B1"}:
            raise ValueError(f"unsupported P3.7B variant: {variant}")
        self.variant: Variant = variant
        relation_input_dim = M4_RELATION_DIM + AGE_BUCKET_COUNT
        if variant == "B1":
            relation_input_dim += P33_RELATION_DIM + P33_GLOBAL_DIM
        self.relation_input_dim = relation_input_dim
        self.gate_input_dim = (
            RELATION_HIDDEN_DIM
            + USER_STATE_DIM
            + CANDIDATE_STATE_DIM
            + AGE_BUCKET_COUNT
        )
        self.delta_input_dim = 1 + USER_STATE_DIM + CANDIDATE_STATE_DIM

        self.relation_encoder = nn.Sequential(
            nn.Linear(self.relation_input_dim, RELATION_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(RELATION_HIDDEN_DIM, RELATION_HIDDEN_DIM),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.gate_input_dim, RELATION_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(RELATION_HIDDEN_DIM, 1),
        )
        self.evidence = nn.Linear(RELATION_HIDDEN_DIM, 1)
        self.delta_scorer = nn.Sequential(
            nn.Linear(self.delta_input_dim, RESIDUAL_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(RESIDUAL_HIDDEN_DIM, 1),
        )
        final_layer = self.delta_scorer[-1]
        assert isinstance(final_layer, nn.Linear)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        self.register_buffer(
            "age_bucket_one_hot",
            torch.eye(AGE_BUCKET_COUNT, dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _require_shape(value: torch.Tensor, expected: tuple[int | None, ...], name: str) -> None:
        if value.ndim != len(expected):
            raise ValueError(f"{name} must have {len(expected)} dimensions, got {value.shape}")
        for observed, wanted in zip(value.shape, expected):
            if wanted is not None and observed != wanted:
                raise ValueError(f"{name} expected shape {expected}, got {tuple(value.shape)}")

    def forward(
        self,
        m4_coarse_score: torch.Tensor,
        m4_relation: torch.Tensor,
        user_state: torch.Tensor,
        candidate_state: torch.Tensor,
        *,
        p33_relation: torch.Tensor | None = None,
        p33_global: torch.Tensor | None = None,
    ) -> HybridOutput:
        batch = int(m4_relation.shape[0]) if m4_relation.ndim else 0
        self._require_shape(
            m4_relation,
            (None, AGE_BUCKET_COUNT, M4_RELATION_DIM),
            "m4_relation",
        )
        self._require_shape(user_state, (batch, USER_STATE_DIM), "user_state")
        self._require_shape(candidate_state, (batch, CANDIDATE_STATE_DIM), "candidate_state")
        if m4_coarse_score.ndim == 2 and m4_coarse_score.shape == (batch, 1):
            coarse = m4_coarse_score[:, 0]
        else:
            self._require_shape(m4_coarse_score, (batch,), "m4_coarse_score")
            coarse = m4_coarse_score

        if self.variant == "B0":
            if p33_relation is not None or p33_global is not None:
                raise ValueError("B0 must not receive any P3.3 auxiliary fields")
        else:
            if p33_relation is None or p33_global is None:
                raise ValueError("B1 requires both P3.3 relation and global auxiliary fields")
            self._require_shape(
                p33_relation,
                (batch, AGE_BUCKET_COUNT, P33_RELATION_DIM),
                "p33_relation",
            )
            self._require_shape(p33_global, (batch, P33_GLOBAL_DIM), "p33_global")

        age = self.age_bucket_one_hot.to(dtype=m4_relation.dtype).unsqueeze(0).expand(
            batch, -1, -1
        )
        relation_parts = [m4_relation]
        if self.variant == "B1":
            assert p33_relation is not None and p33_global is not None
            relation_parts.extend(
                [
                    p33_relation,
                    p33_global[:, None, :].expand(-1, AGE_BUCKET_COUNT, -1),
                ]
            )
        relation_parts.append(age)
        encoded = self.relation_encoder(torch.cat(relation_parts, dim=-1))

        expanded_user = user_state[:, None, :].expand(-1, AGE_BUCKET_COUNT, -1)
        expanded_candidate = candidate_state[:, None, :].expand(-1, AGE_BUCKET_COUNT, -1)
        age_gate_input = torch.cat(
            [encoded, expanded_user, expanded_candidate, age], dim=-1
        )
        age_logits = self.gate(age_gate_input).squeeze(-1)

        # The null/no-adjustment expert reuses the same gate. Its encoded
        # relation and age one-hot are zero, while user/candidate state remains,
        # making the null logit state-dependent without adding parameters.
        null_input = torch.cat(
            [
                torch.zeros(
                    (batch, RELATION_HIDDEN_DIM),
                    dtype=encoded.dtype,
                    device=encoded.device,
                ),
                user_state,
                candidate_state,
                torch.zeros(
                    (batch, AGE_BUCKET_COUNT),
                    dtype=encoded.dtype,
                    device=encoded.device,
                ),
            ],
            dim=-1,
        )
        null_logit = self.gate(null_input).squeeze(-1)
        alpha = torch.softmax(torch.cat([age_logits, null_logit[:, None]], dim=1), dim=1)

        per_age_evidence = self.evidence(encoded).squeeze(-1)
        time_evidence = torch.sum(alpha[:, :AGE_BUCKET_COUNT] * per_age_evidence, dim=1)
        delta_input = torch.cat([time_evidence[:, None], user_state, candidate_state], dim=1)
        delta = self.delta_scorer(delta_input).squeeze(-1)
        return HybridOutput(coarse + delta, alpha, delta, time_evidence)


# This is deliberately a direct alias, not a forked implementation: P3.7B's
# sampling contract requires exact reuse of the audited P3.2 sampler.
sample_training_pairs = sample_same_user_pairs
