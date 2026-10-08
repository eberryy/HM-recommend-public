from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from .m3 import _peak_working_set_bytes
from .m33 import ROLLING_PROTOCOL
from .m4_contract import FINAL_CUTOFF, atomic_json, file_identity
from .p3 import _cutoff_context
from .p3_model import (
    K0,
    NEGATIVES_PER_POSITIVE,
    SAMPLING_SEED,
    binary_auc,
    iter_slices,
    pairwise_logistic_loss,
    sample_same_user_pairs,
    stable_seed,
)
from .p37a_audit import _rank_by_score, _score_universe_for_frozen_sets
from .p37b_features import OUTPUT_FILENAMES, build_raw_feature_assets
from .p37b_report import write_outputs
from .p37b_model import (
    CANDIDATE_STATE_SPECS,
    M4_RELATION_SPECS,
    P33_GLOBAL_SPECS,
    P33_RELATION_SPECS,
    USER_STATE_SPECS,
    FeaturePreprocessor,
    FeatureSpec,
    TimeAwareHybrid,
    TrainOnlyPreprocessor,
)


RUN_ID = "phase3-p3.7b-v1-time-aware-relation-conditioned-hybrid"
STAGE = "P3.7B"
WINDOWS = tuple(ROLLING_PROTOCOL)
VARIANTS = ("C0", "C1", "B0", "B1")
K_VALUES = (5, 10, 20, 50, 100, 200)
AGE_BUCKET_NAMES = ("0_7", "8_28", "29_84", "over_84")
ALPHA_NAMES = (*AGE_BUCKET_NAMES, "null")
MAX_EPOCHS = 30
PATIENCE = 4
BATCH_SIZE = 512
SCORE_BATCH_SIZE = 16_384
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
FINAL_DECISIONS = {
    "promote_M4_only_time_aware_hybrid",
    "promote_time_aware_hybrid_with_p33_aux",
    "combined_success_requires_p33_aux",
    "hybrid_not_better_than_static_p33_scoring",
    "stop_p3_7b_hybrid_after_gate_failure",
    "engineering_failure",
}


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_cutoff(cutoff: str) -> None:
    if cutoff >= FINAL_CUTOFF:
        raise RuntimeError(f"P3.7B refuses final or later cutoff: {cutoff}")


def _array_sha(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


@dataclass
class FrozenAsset:
    role_key: str
    role: str
    cutoff: str
    window: str | None
    root: Path
    candidate_path: Path
    history_path: Path
    users_path: Path
    m4_embedding_path: Path
    p33_embedding_path: Path
    p33_manifest_path: Path
    candidates: dict[str, np.ndarray]
    histories: dict[str, np.ndarray]
    users: list[str]
    counts: np.ndarray
    truth: dict[int, set[int]]
    pairs: np.ndarray
    pair_audit: dict[str, Any]
    identity_audit: dict[str, Any]
    feature_dir: Path | None = None


@dataclass(frozen=True)
class MaterializedFeatureAsset:
    frozen: FrozenAsset
    root: Path
    m4_relation_path: Path
    p33_relation_path: Path
    p33_global_path: Path
    user_state_path: Path
    candidate_state_path: Path
    manifest_path: Path

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "m4_relation": np.load(self.m4_relation_path, mmap_mode="r"),
            "p33_relation": np.load(self.p33_relation_path, mmap_mode="r"),
            "p33_global": np.load(self.p33_global_path, mmap_mode="r"),
            "user_state": np.load(self.user_state_path, mmap_mode="r"),
            "candidate_state": np.load(self.candidate_state_path, mmap_mode="r"),
        }


@dataclass
class PreprocessingBundle:
    variant: str
    relation: TrainOnlyPreprocessor
    user_state: TrainOnlyPreprocessor
    candidate_state: TrainOnlyPreprocessor
    p33_global: TrainOnlyPreprocessor | None
    training_cutoffs: list[str]

    def manifest(self) -> dict[str, Any]:
        components: dict[str, Any] = {
            "relation": self.relation.to_manifest(),
            "user_state": self.user_state.to_manifest(),
            "candidate_state": self.candidate_state.to_manifest(),
        }
        if self.p33_global is not None:
            components["p33_global"] = self.p33_global.to_manifest()
        return {
            "schema_version": "phase3-p3.7b-preprocessing-bundle-v1",
            "variant": self.variant,
            "training_cutoffs": self.training_cutoffs,
            "fit_scope": "all unexpanded candidate rows, and unique user-state rows, from only the current outer chain training cutoffs",
            "components": components,
            "final_week": "not_run",
        }


def _part_column(parts: Sequence[np.ndarray], widths: Sequence[int], index: int) -> np.ndarray:
    offset = 0
    for part, width in zip(parts, widths):
        if index < offset + width:
            return np.asarray(part[..., index - offset], dtype=np.float64).reshape(-1)
        offset += width
    raise IndexError(index)


def _fit_preprocessor_many(
    *,
    specs: Sequence[FeatureSpec],
    sources: Sequence[Sequence[np.ndarray]],
    training_cutoffs: Sequence[str],
) -> TrainOnlyPreprocessor:
    if not sources or not training_cutoffs:
        raise ValueError("train-only preprocessing requires at least one source")
    widths = [int(part.shape[-1]) for part in sources[0]]
    if sum(widths) != len(specs):
        raise ValueError("preprocessing part widths do not match feature specs")
    for source in sources:
        if [int(part.shape[-1]) for part in source] != widths:
            raise ValueError("preprocessing source part width drift")
        prefix = source[0].shape[:-1]
        if any(part.shape[:-1] != prefix for part in source):
            raise ValueError("preprocessing source prefix shape drift")
    preprocessor = TrainOnlyPreprocessor(specs)
    by_name = {spec.name: index for index, spec in enumerate(specs)}
    statistics: dict[str, dict[str, Any]] = {}
    for index, spec in enumerate(specs):
        raw_columns = [_part_column(source, widths, index) for source in sources]
        for source, column in zip(sources, raw_columns):
            finite = np.isfinite(column)
            if spec.kind in {"binary", "rank"} and not finite.all():
                raise ValueError(f"{spec.name} must be finite")
            if spec.kind == "binary" and not np.isin(column, (0.0, 1.0)).all():
                raise ValueError(f"{spec.name} must contain only 0/1")
            if spec.kind == "rank" and (
                np.any(column < 1) or np.any(column > K0) or np.any(column != np.floor(column))
            ):
                raise ValueError(f"{spec.name} must contain integer ranks 1..{K0}")
            if spec.kind == "count" and np.any(column[finite] < 0):
                raise ValueError(f"{spec.name} must be non-negative")
            if np.any(~finite):
                if spec.availability is None:
                    raise ValueError(f"{spec.name} missing without availability flag")
                availability = _part_column(source, widths, by_name[spec.availability])
                if np.any(availability[~finite] != 0):
                    raise ValueError(f"{spec.name} missing while {spec.availability}=1")
        if spec.kind in {"continuous", "count"}:
            selected_chunks = []
            missing_rows = 0
            total_rows = 0
            for column in raw_columns:
                finite = np.isfinite(column)
                selected = column[finite]
                if spec.kind == "count":
                    selected = np.log1p(selected)
                selected_chunks.append(selected)
                missing_rows += int(np.count_nonzero(~finite))
                total_rows += len(column)
            selected_all = np.concatenate(selected_chunks)
            if not len(selected_all):
                raise ValueError(f"{spec.name} has no finite train-only values")
            mean = float(np.mean(selected_all, dtype=np.float64))
            observed_std = float(np.std(selected_all, dtype=np.float64))
            std = observed_std if observed_std > 1e-12 else 1.0
            quantiles = np.quantile(selected_all, [0.01, 0.25, 0.50, 0.75, 0.99])
            statistics[spec.name] = {
                "mean": mean,
                "std": std,
                "observed_std": observed_std,
                "constant_column": observed_std <= 1e-12,
                "finite_rows": int(len(selected_all)),
                "missing_rows": missing_rows,
                "quantiles": {
                    "q01": float(quantiles[0]),
                    "q25": float(quantiles[1]),
                    "q50": float(quantiles[2]),
                    "q75": float(quantiles[3]),
                    "q99": float(quantiles[4]),
                },
            }
            del selected_all, selected_chunks
        elif spec.kind == "rank":
            statistics[spec.name] = {
                "fixed_min_rank": 1,
                "fixed_max_rank": K0,
                "formula": "(rank - 1) / (rank_budget - 1)",
            }
        else:
            statistics[spec.name] = {"allowed_values": [0, 1], "standardized": False}
    latest_cutoff = max(training_cutoffs)
    _safe_cutoff(latest_cutoff)
    preprocessor.training_cutoff = latest_cutoff
    preprocessor.training_rows = int(
        sum(np.prod(source[0].shape[:-1], dtype=np.int64) for source in sources)
    )
    preprocessor.statistics = statistics
    return preprocessor


def _fit_preprocessing_bundle(
    assets: Sequence[MaterializedFeatureAsset], variant: str
) -> FeaturePreprocessor:
    if variant not in {"B0", "B1"}:
        raise ValueError(variant)
    loaded = [asset.arrays() for asset in assets]
    sources: list[dict[str, Any]] = []
    for asset, row in zip(assets, loaded):
        active_users = np.unique(
            asset.frozen.candidates["user_index"].astype(np.int64, copy=False)
        )
        source: dict[str, Any] = {
            "m4_relation": row["m4_relation"],
            "user_state": np.asarray(row["user_state"][active_users], dtype=np.float32),
            "candidate_state": row["candidate_state"],
            "cutoff": asset.frozen.cutoff,
        }
        if variant == "B1":
            source["p33_relation"] = row["p33_relation"]
            source["p33_global"] = row["p33_global"]
        sources.append(source)
    bundle = FeaturePreprocessor(variant).fit_many(  # type: ignore[arg-type]
        sources,
        training_cutoff=max(asset.frozen.cutoff for asset in assets),
    )
    if variant == "B1":
        b0_reference = _fit_preprocessing_bundle(assets, "B0")
        common_checks = {
            "m4_relation": all(
                bundle.relation.statistics[spec.name] == b0_reference.relation.statistics[spec.name]
                for spec in M4_RELATION_SPECS
            ),
            "user_state": bundle.user_state.statistics == b0_reference.user_state.statistics,
            "candidate_state": bundle.candidate_state.statistics == b0_reference.candidate_state.statistics,
        }
        if not all(common_checks.values()):
            raise RuntimeError(f"B0/B1 common preprocessing drift: {common_checks}")
    return bundle


def _resolve_p33_embedding(repo_root: Path, *, role: str, cutoff: str, window: str | None) -> tuple[Path, Path]:
    root = (
        repo_root
        / "artifacts"
        / "phase3"
        / "phase3-p3.3-v1-multiview-graph-student"
        / "students-v1"
    )
    leaf = root / "training" / cutoff if role == "training" else root / "outer" / str(window)
    embedding = leaf / "catalog_embeddings.float16.npy"
    manifest = leaf / "manifest.json"
    if not embedding.is_file() or not manifest.is_file():
        raise RuntimeError(
            f"cutoff-safe P3.3 Student is missing for role={role}, cutoff={cutoff}, window={window}"
        )
    return embedding, manifest


def _load_frozen_asset(
    *,
    repo_root: Path,
    p31: dict[str, Any],
    transactions_path: Path,
    catalog_items: list[str],
    role: str,
    cutoff: str,
    window: str | None,
) -> FrozenAsset:
    _safe_cutoff(cutoff)
    p31_key = f"training:{cutoff}" if role == "training" else f"outer_validation:{window}"
    declared = p31["all_assets"][p31_key]
    candidate_path = Path(declared["artifacts"]["candidates"]["path"])
    history_path = Path(declared["artifacts"]["histories"]["path"])
    users_path = Path(declared["artifacts"]["users"]["path"])
    m4_embedding_path = Path(declared["student_embedding"]["path"])
    paths = {
        "candidates": candidate_path,
        "histories": history_path,
        "users": users_path,
        "m4_embedding": m4_embedding_path,
    }
    identity: dict[str, Any] = {}
    declared_map = {
        "candidates": declared["artifacts"]["candidates"],
        "histories": declared["artifacts"]["histories"],
        "users": declared["artifacts"]["users"],
        "m4_embedding": declared["student_embedding"],
    }
    for name, path in paths.items():
        observed = file_identity(path)
        expected = declared_map[name]
        passed = observed["sha256"] == expected["sha256"] and observed["bytes"] == expected["bytes"]
        identity[name] = {"observed": observed, "declared": expected, "passed": passed}
        if not passed:
            raise RuntimeError(f"P3.1 frozen asset identity drift: {p31_key}/{name}")

    p33_embedding_path, p33_manifest_path = _resolve_p33_embedding(
        repo_root, role=role, cutoff=cutoff, window=window
    )
    p33_manifest = _json(p33_manifest_path)
    p33_identity = file_identity(p33_embedding_path)
    manifest_embedding = p33_manifest.get("encoding", {}).get("artifact", {})
    if p33_identity["sha256"] != manifest_embedding.get("sha256"):
        raise RuntimeError(f"P3.3 Student manifest identity drift: {p31_key}")
    if p33_manifest.get("final_week") != "not_run":
        raise RuntimeError(f"P3.3 Student final-week boundary drift: {p31_key}")
    if role == "training":
        declared_cutoffs = p33_manifest.get("training_cutoffs", [])
        if declared_cutoffs != [cutoff]:
            raise RuntimeError(f"P3.3 point-in-time Student reused backward: {p31_key}")
    else:
        expected_cutoffs = list(ROLLING_PROTOCOL[str(window)]["outer_train"])
        if p33_manifest.get("training_cutoffs") != expected_cutoffs:
            raise RuntimeError(f"P3.3 outer Student training chain drift: {p31_key}")
    identity["p33_embedding"] = {
        "observed": p33_identity,
        "manifest": file_identity(p33_manifest_path),
        "declared_training_cutoffs": p33_manifest.get("training_cutoffs"),
        "content_only_inference": bool(
            p33_manifest.get("student_inference_audit", {}).get("content_only")
        )
        and not bool(
            p33_manifest.get("student_inference_audit", {}).get(
                "teacher_embedding_used_at_inference"
            )
        ),
        "passed": True,
    }
    if not identity["p33_embedding"]["content_only_inference"]:
        raise RuntimeError(f"P3.3 inference is not content-only: {p31_key}")

    candidates = _load_npz(candidate_path)
    histories = _load_npz(history_path)
    users = pd.read_csv(users_path, dtype={"customer_id": str})["customer_id"].tolist()
    required = {"user_index", "catalog_row", "rank", "coarse_score", "target"}
    if not required.issubset(candidates):
        raise RuntimeError(f"candidate schema drift: {p31_key}")
    if histories["catalog_row"].shape != (len(users), 20):
        raise RuntimeError(f"history shape drift: {p31_key}")
    if len(candidates["user_index"]) % K0:
        raise RuntimeError(f"candidate row count is not a multiple of {K0}: {p31_key}")
    article_to_row = {item: index for index, item in enumerate(catalog_items)}
    counts, rebuilt_rows, rebuilt_days, truth, cutoff_audit = _cutoff_context(
        transactions_path=transactions_path,
        cutoff=cutoff,
        users=users,
        article_to_row=article_to_row,
        catalog_size=len(catalog_items),
    )
    history_checks = {
        "catalog_row_exact": bool(np.array_equal(rebuilt_rows, histories["catalog_row"])),
        "days_exact": bool(np.array_equal(rebuilt_days, histories["days_since_purchase"])),
        "mask_exact": bool(np.array_equal((rebuilt_rows >= 0).astype(np.uint8), histories["mask"])),
        "latest_behavior_before_cutoff": bool(cutoff_audit["cutoff_safe"]),
    }
    if not all(history_checks.values()):
        raise RuntimeError(f"cutoff-safe history reconstruction failed: {p31_key}: {history_checks}")
    pairs, pair_audit = sample_same_user_pairs(
        user_index=candidates["user_index"],
        rank=candidates["rank"],
        target=candidates["target"],
        cutoff=cutoff,
    )
    if len(pairs) and not (
        pair_audit["same_user_only"] and pair_audit["all_buckets_covered"]
    ):
        raise RuntimeError(f"P3.2 sampler parity invariant failed: {p31_key}")
    pair_audit = {
        **pair_audit,
        "pair_array_sha256": _array_sha(pairs),
        "exact_reuse_function": "hm_recsys.p3_model.sample_same_user_pairs",
    }
    identity["history_reconstruction"] = history_checks
    identity["cutoff"] = cutoff_audit
    identity["candidate_identity_sha256"] = _array_sha(
        candidates["user_index"].astype(np.int32, copy=False),
        candidates["catalog_row"].astype(np.int32, copy=False),
    )
    return FrozenAsset(
        role_key=p31_key,
        role=role,
        cutoff=cutoff,
        window=window,
        root=candidate_path.parent,
        candidate_path=candidate_path,
        history_path=history_path,
        users_path=users_path,
        m4_embedding_path=m4_embedding_path,
        p33_embedding_path=p33_embedding_path,
        p33_manifest_path=p33_manifest_path,
        candidates=candidates,
        histories=histories,
        users=users,
        counts=counts,
        truth=truth,
        pairs=pairs,
        pair_audit=pair_audit,
        identity_audit=identity,
    )


def _rank_hybrid(candidates: dict[str, np.ndarray], scores: np.ndarray) -> np.ndarray:
    rows = len(scores)
    if rows != len(candidates["rank"]) or rows % K0:
        raise ValueError("hybrid score/candidate shape drift")
    score_matrix = np.asarray(scores, dtype=np.float32).reshape(-1, K0)
    # Candidate rows are frozen M4 rank order. Stable sorting therefore implements
    # score desc, M4 rank asc; catalog row is only a final unreachable tie-break.
    order = np.argsort(-score_matrix, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.uint16)
    row_ids = np.arange(len(order))[:, None]
    ranks[row_ids, order] = np.arange(1, K0 + 1, dtype=np.uint16)[None, :]
    return ranks.reshape(-1)


def _truth_segment_arrays(
    truth: dict[int, set[int]], counts: np.ndarray, user_count: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for segment in ("strict_cold", "sparse_1_5", "cold_universe"):
        denom = np.zeros(user_count, dtype=np.int32)
        for user, items in truth.items():
            if segment == "strict_cold":
                denom[user] = sum(counts[item] == 0 for item in items)
            elif segment == "sparse_1_5":
                denom[user] = sum(1 <= counts[item] <= 5 for item in items)
            else:
                denom[user] = sum(counts[item] <= 5 for item in items)
        result[segment] = (denom, denom > 0)
    return result


def _ordering_metrics_fast(
    *,
    candidates: dict[str, np.ndarray],
    ordering_rank: np.ndarray,
    truth: dict[int, set[int]],
    counts: np.ndarray,
    user_count: int,
) -> dict[str, Any]:
    if len(ordering_rank) != len(candidates["user_index"]):
        raise ValueError("ordering rank length differs from candidate rows")
    users = candidates["user_index"].astype(np.int64, copy=False)
    items = candidates["catalog_row"].astype(np.int64, copy=False)
    target = candidates["target"].astype(bool, copy=False)
    segment_truth = _truth_segment_arrays(truth, counts, user_count)
    output: dict[str, Any] = {"at_k": {}}
    for k in K_VALUES:
        selected = ordering_rank <= k
        selected_rows = int(np.count_nonzero(selected))
        positive = selected & target
        positive_rows = int(np.count_nonzero(positive))
        segment_metrics: dict[str, Any] = {}
        for segment, (denom, valid_users) in segment_truth.items():
            if segment == "strict_cold":
                segment_rows = positive & (counts[items] == 0)
            elif segment == "sparse_1_5":
                segment_rows = positive & (counts[items] >= 1) & (counts[items] <= 5)
            else:
                segment_rows = positive & (counts[items] <= 5)
            matched = np.bincount(users[segment_rows], minlength=user_count)
            recalls = matched[valid_users] / denom[valid_users]
            segment_metrics[segment] = {
                "recall": float(np.mean(recalls)) if len(recalls) else 0.0,
                "hit_rate": float(np.mean(matched[valid_users] > 0)) if np.any(valid_users) else 0.0,
                "truth_users": int(np.count_nonzero(valid_users)),
                "truth_pairs": int(denom.sum()),
                "covered_truth_pairs": int(matched.sum()),
            }
        density = positive_rows / max(selected_rows, 1)
        output["at_k"][str(k)] = {
            "candidate_rows": selected_rows,
            "positive_rows": positive_rows,
            "positive_density": density,
            "truth_pairs_per_1k_candidates": 1000.0 * density,
            "segments": segment_metrics,
        }
    positive_ranks = ordering_rank[target].astype(np.int32)
    cold_denom, cold_users = segment_truth["cold_universe"]
    first = np.full(user_count, K0 + 1, dtype=np.int32)
    np.minimum.at(first, users[target], ordering_rank[target].astype(np.int32))
    reciprocal = np.where(first[cold_users] <= K0, 1.0 / first[cold_users], 0.0)
    hit_first = first[cold_users][first[cold_users] <= K0]
    total_positive = int(np.count_nonzero(target))
    output["ranking"] = {
        "mrr": float(np.mean(reciprocal)) if len(reciprocal) else 0.0,
        "mrr_truth_user_denominator": int(np.count_nonzero(cold_users)),
        "first_positive_rank_mean_hit_users": float(np.mean(hit_first)) if len(hit_first) else None,
        "first_positive_rank_median_hit_users": float(np.median(hit_first)) if len(hit_first) else None,
        "first_positive_rank_hit_users": int(len(hit_first)),
        "positive_rank_best": int(np.min(positive_ranks)) if len(positive_ranks) else None,
        "positive_rank_p25": float(np.quantile(positive_ranks, 0.25)) if len(positive_ranks) else None,
        "positive_rank_p50": float(np.quantile(positive_ranks, 0.50)) if len(positive_ranks) else None,
        "positive_rank_p75": float(np.quantile(positive_ranks, 0.75)) if len(positive_ranks) else None,
        "coarse_positive_pairs_top200": total_positive,
        "conversion": {
            f"top200_to_top{k}": int(np.count_nonzero(positive_ranks <= k)) / max(total_positive, 1)
            for k in (5, 10, 20, 50)
        },
        "conversion_counts": {
            f"top{k}": int(np.count_nonzero(positive_ranks <= k)) for k in (5, 10, 20, 50)
        },
    }
    return output


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"rows": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "rows": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _score_diagnostics(candidates: dict[str, np.ndarray], scores: np.ndarray) -> dict[str, Any]:
    target = candidates["target"].astype(bool, copy=False)
    positive = np.asarray(scores[target], dtype=np.float64)
    negative = np.asarray(scores[~target], dtype=np.float64)
    users = candidates["user_index"]
    percentiles: list[float] = []
    matrix_scores = np.asarray(scores).reshape(-1, K0)
    matrix_target = target.reshape(-1, K0)
    for local_scores, local_target in zip(matrix_scores, matrix_target):
        negatives = local_scores[~local_target]
        for value in local_scores[local_target]:
            if len(negatives):
                percentiles.append(
                    float((np.count_nonzero(negatives < value) + 0.5 * np.count_nonzero(negatives == value)) / len(negatives))
                )
    del users
    percentile_array = np.asarray(percentiles, dtype=np.float64)
    return {
        "positive_vs_unobserved_auc": binary_auc(positive, negative),
        "same_user_truth_percentile": _distribution(percentile_array),
        "truth_score_distribution": _distribution(positive),
        "unobserved_score_distribution": _distribution(negative),
        "unobserved_definition": "候选商品未在该截止日后的7天标签窗被购买；数据没有曝光日志，不能解释为用户明确负反馈。",
    }


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    if key == "density20":
        return float(metrics["at_k"]["20"]["positive_density"])
    if key == "mrr":
        return float(metrics["ranking"]["mrr"])
    if key == "conversion20":
        return float(metrics["ranking"]["conversion"]["top200_to_top20"])
    if key == "strict_recall20":
        return float(metrics["at_k"]["20"]["segments"]["strict_cold"]["recall"])
    if key == "sparse_recall20":
        return float(metrics["at_k"]["20"]["segments"]["sparse_1_5"]["recall"])
    raise KeyError(key)


def _formal_gate(windows: dict[str, Any], variant: str, baseline: str = "C0") -> dict[str, Any]:
    values: dict[str, dict[str, list[float]]] = {}
    for key in ("density20", "mrr", "conversion20", "strict_recall20", "sparse_recall20"):
        values[key] = {
            variant: [_metric_value(windows[w]["variants"][variant], key) for w in WINDOWS],
            baseline: [_metric_value(windows[w]["variants"][baseline], key) for w in WINDOWS],
        }
    checks = {
        "mean_density20_strictly_improves": np.mean(values["density20"][variant]) > np.mean(values["density20"][baseline]),
        "density20_nondegrade_windows_at_least_3": sum(a >= b for a, b in zip(values["density20"][variant], values["density20"][baseline])) >= 3,
        "mean_mrr_strictly_improves": np.mean(values["mrr"][variant]) > np.mean(values["mrr"][baseline]),
        "mrr_nondegrade_windows_at_least_3": sum(a >= b for a, b in zip(values["mrr"][variant], values["mrr"][baseline])) >= 3,
        "mean_conversion20_strictly_improves": np.mean(values["conversion20"][variant]) > np.mean(values["conversion20"][baseline]),
    }
    segment_checks: dict[str, Any] = {}
    for segment_key in ("strict_recall20", "sparse_recall20"):
        candidate_values = values[segment_key][variant]
        baseline_values = values[segment_key][baseline]
        segment_checks[segment_key] = {
            "mean_strictly_improves": bool(np.mean(candidate_values) > np.mean(baseline_values)),
            "nondegrade_windows": int(sum(a >= b for a, b in zip(candidate_values, baseline_values))),
            "passed": bool(
                np.mean(candidate_values) > np.mean(baseline_values)
                and sum(a >= b for a, b in zip(candidate_values, baseline_values)) >= 3
            ),
        }
    checks["strict_or_sparse_recall20_passes"] = any(row["passed"] for row in segment_checks.values())
    checks["candidate_identity_exact"] = all(
        windows[w]["candidate_identity"]["all_variants_exact"] for w in WINDOWS
    )
    passed = all(bool(value) for value in checks.values())
    return {
        "variant": variant,
        "baseline": baseline,
        "status": "supported" if passed else "rejected",
        "passed": passed,
        "checks": checks,
        "segment_checks": segment_checks,
        "machine_values": values,
    }


def _complexity_gate(windows: dict[str, Any]) -> dict[str, Any]:
    b1 = {key: [_metric_value(windows[w]["variants"]["B1"], key) for w in WINDOWS] for key in ("density20", "mrr", "conversion20")}
    c1 = {key: [_metric_value(windows[w]["variants"]["C1"], key) for w in WINDOWS] for key in ("density20", "mrr", "conversion20")}
    nondegrade = [
        (b1["density20"][index] >= c1["density20"][index])
        or (b1["mrr"][index] >= c1["mrr"][index])
        for index in range(len(WINDOWS))
    ]
    checks = {
        "mean_density20_strictly_improves": np.mean(b1["density20"]) > np.mean(c1["density20"]),
        "mean_mrr_strictly_improves": np.mean(b1["mrr"]) > np.mean(c1["mrr"]),
        "mean_conversion20_nondegrades": np.mean(b1["conversion20"]) >= np.mean(c1["conversion20"]),
        "density_or_mrr_nondegrade_windows_at_least_3": sum(nondegrade) >= 3,
    }
    passed = all(bool(value) for value in checks.values())
    return {
        "variant": "B1",
        "baseline": "C1",
        "status": "supported" if passed else "rejected",
        "passed": passed,
        "checks": checks,
        "window_nondegrade": dict(zip(WINDOWS, nondegrade)),
        "machine_values": {"B1": b1, "C1": c1},
    }


def _b1_vs_b0_selection(windows: dict[str, Any]) -> dict[str, Any]:
    b1 = {key: [_metric_value(windows[w]["variants"]["B1"], key) for w in WINDOWS] for key in ("density20", "mrr", "conversion20")}
    b0 = {key: [_metric_value(windows[w]["variants"]["B0"], key) for w in WINDOWS] for key in ("density20", "mrr", "conversion20")}
    nondegrade = [
        b1["density20"][index] >= b0["density20"][index]
        or b1["mrr"][index] >= b0["mrr"][index]
        for index in range(len(WINDOWS))
    ]
    checks = {
        "mean_density20_strictly_improves": np.mean(b1["density20"]) > np.mean(b0["density20"]),
        "mean_mrr_strictly_improves": np.mean(b1["mrr"]) > np.mean(b0["mrr"]),
        "mean_conversion20_nondegrades": np.mean(b1["conversion20"]) >= np.mean(b0["conversion20"]),
        "density_or_mrr_nondegrade_windows_at_least_3": sum(nondegrade) >= 3,
    }
    return {"passed": all(bool(v) for v in checks.values()), "checks": checks, "window_nondegrade": dict(zip(WINDOWS, nondegrade))}


def _final_decision(b0: dict[str, Any], b1: dict[str, Any], complexity: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    if b0["passed"] and not b1["passed"]:
        status, selected = "promote_M4_only_time_aware_hybrid", "B0"
    elif not b0["passed"] and b1["passed"]:
        if complexity["passed"]:
            status, selected = "combined_success_requires_p33_aux", "B1"
        else:
            status, selected = "hybrid_not_better_than_static_p33_scoring", "C1"
    elif b0["passed"] and b1["passed"]:
        if complexity["passed"] and selection["passed"]:
            status, selected = "promote_time_aware_hybrid_with_p33_aux", "B1"
        else:
            status, selected = "promote_M4_only_time_aware_hybrid", "B0"
    else:
        # The learned-hybrid experiment stops, but P3.7A already established
        # exact fixed-pool C1 gains in all four windows. Retain that simpler
        # static scorer when its parity is reconfirmed in this run.
        status, selected = "stop_p3_7b_hybrid_after_gate_failure", "C1"
    if status not in FINAL_DECISIONS:
        raise AssertionError(status)
    return {
        "status": status,
        "B0_gate": b0["status"],
        "B1_gate": b1["status"],
        "B1_vs_C1_complexity_gate": complexity["status"],
        "selected_variant": selected,
    }


def _alpha_summary(alpha: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(alpha[mask], dtype=np.float32)
    if not len(selected):
        return {
            "candidate_rows": 0,
            "mean": {name: None for name in ALPHA_NAMES},
            "median": {name: None for name in ALPHA_NAMES},
        }
    return {
        "candidate_rows": int(len(selected)),
        "mean": {name: float(np.mean(selected[:, index])) for index, name in enumerate(ALPHA_NAMES)},
        "median": {name: float(np.median(selected[:, index])) for index, name in enumerate(ALPHA_NAMES)},
    }


def _rank_gain_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"truth_pairs": 0, "mean": None, "median": None, "improved_share": None}
    return {
        "truth_pairs": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "improved_share": float(np.mean(values > 0)),
    }


def _time_gate_mechanism_audit(
    *,
    frozen: FrozenAsset,
    raw_user_state: np.ndarray,
    alpha: np.ndarray,
    hybrid_rank: np.ndarray,
    profile_quartiles: Sequence[float],
) -> dict[str, Any]:
    candidates = frozen.candidates
    target = candidates["target"].astype(bool, copy=False)
    item_counts = frozen.counts[candidates["catalog_row"].astype(np.int64)]
    masks = {
        "all_candidates": np.ones(len(target), dtype=bool),
        "truth_candidates": target,
        "unobserved_candidates": ~target,
        "strict_cold_candidates": item_counts == 0,
        "sparse_1_5_candidates": (item_counts >= 1) & (item_counts <= 5),
    }
    alpha_statistics = {name: _alpha_summary(alpha, mask) for name, mask in masks.items()}
    users = candidates["user_index"].astype(np.int64, copy=False)
    truth_users = users[target]
    gain = candidates["rank"][target].astype(np.float32) - hybrid_rank[target].astype(np.float32)
    recent_active_by_user = (raw_user_state[:, 0] + raw_user_state[:, 1]) > 0
    availability_by_user = raw_user_state[:, 7] > 0.5
    profile_by_user = raw_user_state[:, 6]
    rank_gain = {
        "definition": "仅以冻结Top200内的未来正例用户—商品对为分母；C0原始名次减Hybrid名次，正数表示正例上移。",
        "recent_active": _rank_gain_summary(gain[recent_active_by_user[truth_users]]),
        "inactive_recent": _rank_gain_summary(gain[~recent_active_by_user[truth_users]]),
    }
    quartiles = np.asarray(profile_quartiles, dtype=np.float64)
    if quartiles.shape != (3,):
        raise ValueError("profile quartiles must contain train-only q25/q50/q75")
    labels = ("q1_low", "q2", "q3", "q4_high")
    available_truth = availability_by_user[truth_users]
    truth_profile = profile_by_user[truth_users]
    bucket = np.digitize(truth_profile, quartiles, right=True)
    rank_gain["profile_unavailable"] = _rank_gain_summary(gain[~available_truth])
    for index, label in enumerate(labels):
        rank_gain[label] = _rank_gain_summary(gain[available_truth & (bucket == index)])
    return {
        "alpha_order": list(ALPHA_NAMES),
        "alpha_definition": "softmax门控在四个时间范围专家与不调整专家上的权重；每条候选行五项之和为1。",
        "alpha_sum_max_abs_error": float(np.max(np.abs(np.asarray(alpha, dtype=np.float64).sum(axis=1) - 1.0))),
        "statistics": alpha_statistics,
        "rank_gain_by_user_state": rank_gain,
        "profile_quartile_boundaries_from_outer_training": {
            "q25": float(quartiles[0]),
            "q50": float(quartiles[1]),
            "q75": float(quartiles[2]),
        },
        "promotion_metric": False,
    }


def _profile_quartiles(feature_assets: Sequence[Any]) -> list[float]:
    values: list[np.ndarray] = []
    for asset in feature_assets:
        user_state = np.load(asset.user_state_path, mmap_mode="r")
        active_users = np.unique(
            asset.frozen.candidates["user_index"].astype(np.int64, copy=False)
        )
        active_state = np.asarray(user_state[active_users], dtype=np.float32)
        available = active_state[:, 7] > 0.5
        if np.any(available):
            values.append(np.asarray(active_state[available, 6], dtype=np.float64))
    if not values:
        return [0.0, 0.0, 0.0]
    merged = np.concatenate(values)
    return [float(value) for value in np.quantile(merged, [0.25, 0.50, 0.75])]


def _exact_p33_static_score(
    *,
    frozen: FrozenAsset,
    p33_metrics: dict[str, Any],
    p37a: dict[str, Any] | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    p33_row = p33_metrics["pipeline"]["p3_1"]["all_assets"][frozen.role_key]
    p33_candidate_path = Path(p33_row["artifacts"]["candidates"]["path"])
    observed = file_identity(p33_candidate_path)
    declared = p33_row["artifacts"]["candidates"]
    if observed["sha256"] != declared["sha256"] or observed["bytes"] != declared["bytes"]:
        raise RuntimeError(f"P3.3 reconstruction-only candidate asset drift: {frozen.role_key}")
    p33_candidates = _load_npz(p33_candidate_path)
    embeddings = np.load(frozen.p33_embedding_path, mmap_mode="r")
    candidate_universe = np.flatnonzero(frozen.counts <= 5).astype(np.int32)
    score_sets, reconstruction = _score_universe_for_frozen_sets(
        {"R_M": frozen.candidates, "R_G": p33_candidates},
        frozen.histories,
        embeddings,
        candidate_universe,
        authoritative_set="R_G",
        device=device,
    )
    scores = score_sets["R_M"]
    ranks = _rank_by_score(
        frozen.candidates["user_index"], frozen.candidates["catalog_row"], scores
    ).astype(np.uint16)
    metrics = _ordering_metrics_fast(
        candidates=frozen.candidates,
        ordering_rank=ranks,
        truth=frozen.truth,
        counts=frozen.counts,
        user_count=len(frozen.users),
    )
    metric_exact: bool | None = None
    if frozen.role == "outer_validation":
        if p37a is None or frozen.window is None:
            raise RuntimeError("outer C1 reconstruction requires P3.7A authority")
        expected = p37a["four_variant_metrics"]["windows"][frozen.window]["A01"]
        metric_exact = metrics == expected
        if not metric_exact:
            raise RuntimeError(f"P3.7A C1 metric reproduction failed: {frozen.window}")
    audit = {
        "definition": "为复现P3.7A A01的float32并列语义，临时读取P3.3候选仅作完整冷/稀疏目录数值核重建；正式候选身份仍100%来自P3.1 M4 Top200。",
        "reconstruction_only_p33_candidate_asset": observed,
        "formal_candidate_asset": file_identity(frozen.candidate_path),
        "full_universe_reconstruction": reconstruction,
        "p3_7a_A01_metrics_exact": metric_exact,
        "formal_candidate_source_is_M4_only": True,
    }
    return scores, ranks, metrics, audit


def _feature_batch(
    *,
    feature: MaterializedFeatureAsset,
    rows: np.ndarray,
    preprocessing: FeaturePreprocessor,
    device: torch.device,
    mask_p33: bool = False,
) -> dict[str, torch.Tensor | None]:
    arrays = feature.arrays()
    rows = np.asarray(rows, dtype=np.int64)
    users = feature.frozen.candidates["user_index"][rows].astype(np.int64, copy=False)
    m4_raw = np.asarray(arrays["m4_relation"][rows], dtype=np.float32)
    if preprocessing.variant == "B1":
        p33_raw = np.asarray(arrays["p33_relation"][rows], dtype=np.float32)
        relation_raw = np.concatenate([m4_raw, p33_raw], axis=-1)
    else:
        p33_raw = None
        relation_raw = m4_raw
    relation = preprocessing.relation.transform(relation_raw)
    m4_relation = relation[..., : len(M4_RELATION_SPECS)]
    p33_relation = relation[..., len(M4_RELATION_SPECS) :] if preprocessing.variant == "B1" else None
    user_state = preprocessing.user_state.transform(
        np.asarray(arrays["user_state"][users], dtype=np.float32)
    )
    candidate_state = preprocessing.candidate_state.transform(
        np.asarray(arrays["candidate_state"][rows], dtype=np.float32)
    )
    p33_global = None
    if preprocessing.variant == "B1":
        assert preprocessing.p33_global is not None
        p33_global = preprocessing.p33_global.transform(
            np.asarray(arrays["p33_global"][rows], dtype=np.float32)
        )
        if mask_p33:
            p33_relation = np.zeros_like(p33_relation, dtype=np.float32)
            p33_global = np.zeros_like(p33_global, dtype=np.float32)
    return {
        "m4_coarse_score": torch.from_numpy(
            feature.frozen.candidates["coarse_score"][rows].astype(np.float32, copy=False)
        ).to(device),
        "m4_relation": torch.from_numpy(np.ascontiguousarray(m4_relation)).to(device),
        "user_state": torch.from_numpy(np.ascontiguousarray(user_state)).to(device),
        "candidate_state": torch.from_numpy(np.ascontiguousarray(candidate_state)).to(device),
        "p33_relation": None
        if p33_relation is None
        else torch.from_numpy(np.ascontiguousarray(p33_relation)).to(device),
        "p33_global": None
        if p33_global is None
        else torch.from_numpy(np.ascontiguousarray(p33_global)).to(device),
    }


def _forward(model: TimeAwareHybrid, batch: dict[str, torch.Tensor | None]) -> Any:
    return model(
        batch["m4_coarse_score"],  # type: ignore[arg-type]
        batch["m4_relation"],  # type: ignore[arg-type]
        batch["user_state"],  # type: ignore[arg-type]
        batch["candidate_state"],  # type: ignore[arg-type]
        p33_relation=batch["p33_relation"],  # type: ignore[arg-type]
        p33_global=batch["p33_global"],  # type: ignore[arg-type]
    )


def _new_model(variant: str, *, seed: int, device: torch.device) -> TimeAwareHybrid:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = TimeAwareHybrid(variant)  # type: ignore[arg-type]
    _initialize_paired_variant_weights(model, seed=seed)
    return model.to(device)


def _seeded_generator(seed: int, label: str) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(seed, "P3.7B-paired-init-v1", label) % (2**63 - 1))
    return generator


def _reset_linear_with_named_seed(layer: torch.nn.Linear, *, seed: int, label: str) -> None:
    generator = _seeded_generator(seed, label)
    torch.nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
    if layer.bias is not None:
        bound = 1.0 / math.sqrt(float(layer.weight.shape[1]))
        torch.nn.init.uniform_(layer.bias, -bound, bound, generator=generator)


def _tensor_group_sha256(values: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _paired_initialization_audit(model: TimeAwareHybrid, *, seed: int) -> dict[str, Any]:
    first = model.relation_encoder[0]
    if not isinstance(first, torch.nn.Linear):
        raise TypeError("P3.7B relation encoder first layer must be Linear")
    m4_dim = len(M4_RELATION_SPECS)
    age_dim = len(AGE_BUCKET_NAMES)
    if model.variant == "B0":
        common_weight = first.weight[:, : m4_dim + age_dim]
        auxiliary: list[tuple[str, torch.Tensor]] = []
    else:
        auxiliary_dim = len(P33_RELATION_SPECS) + len(P33_GLOBAL_SPECS)
        common_weight = torch.cat(
            [first.weight[:, :m4_dim], first.weight[:, m4_dim + auxiliary_dim :]],
            dim=1,
        )
        auxiliary = [("relation_encoder.0.auxiliary_weight", first.weight[:, m4_dim : m4_dim + auxiliary_dim])]
    shared_downstream = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name not in {"relation_encoder.0.weight", "relation_encoder.0.bias"}
    ]
    return {
        "scheme": "paired-layerwise-v1",
        "seed": int(seed),
        "common_relation_input_sha256": _tensor_group_sha256(
            [
                ("relation_encoder.0.common_weight", common_weight),
                ("relation_encoder.0.bias", first.bias),
            ]
        ),
        "shared_downstream_sha256": _tensor_group_sha256(shared_downstream),
        "auxiliary_relation_input_sha256": (
            _tensor_group_sha256(auxiliary) if auxiliary else None
        ),
    }


def _initialize_paired_variant_weights(model: TimeAwareHybrid, *, seed: int) -> None:
    """Make B0/B1 common initial parameters identical despite B1's wider input."""
    first = model.relation_encoder[0]
    second = model.relation_encoder[2]
    gate_first = model.gate[0]
    gate_last = model.gate[2]
    evidence = model.evidence
    delta_first = model.delta_scorer[0]
    delta_last = model.delta_scorer[2]
    linear_layers = (first, second, gate_first, gate_last, evidence, delta_first, delta_last)
    if not all(isinstance(layer, torch.nn.Linear) for layer in linear_layers):
        raise TypeError("P3.7B paired initialization requires the frozen Linear architecture")

    m4_dim = len(M4_RELATION_SPECS)
    age_dim = len(AGE_BUCKET_NAMES)
    common_dim = m4_dim + age_dim
    common_generator = _seeded_generator(seed, "relation_encoder.0.common")
    common_weight = torch.empty((first.out_features, common_dim), dtype=first.weight.dtype)
    torch.nn.init.kaiming_uniform_(common_weight, a=math.sqrt(5), generator=common_generator)
    common_bias = torch.empty((first.out_features,), dtype=first.bias.dtype)
    torch.nn.init.uniform_(
        common_bias,
        -1.0 / math.sqrt(float(common_dim)),
        1.0 / math.sqrt(float(common_dim)),
        generator=common_generator,
    )
    with torch.no_grad():
        if model.variant == "B0":
            if first.in_features != common_dim:
                raise RuntimeError("B0 relation input dimension drift")
            first.weight.copy_(common_weight)
        else:
            auxiliary_dim = len(P33_RELATION_SPECS) + len(P33_GLOBAL_SPECS)
            if first.in_features != common_dim + auxiliary_dim:
                raise RuntimeError("B1 relation input dimension drift")
            first.weight[:, :m4_dim].copy_(common_weight[:, :m4_dim])
            auxiliary_weight = torch.empty(
                (first.out_features, auxiliary_dim), dtype=first.weight.dtype
            )
            torch.nn.init.kaiming_uniform_(
                auxiliary_weight,
                a=math.sqrt(5),
                generator=_seeded_generator(seed, "relation_encoder.0.p33_auxiliary"),
            )
            first.weight[:, m4_dim : m4_dim + auxiliary_dim].copy_(auxiliary_weight)
            first.weight[:, m4_dim + auxiliary_dim :].copy_(common_weight[:, m4_dim:])
        first.bias.copy_(common_bias)

    for label, layer in (
        ("relation_encoder.2", second),
        ("gate.0", gate_first),
        ("gate.2", gate_last),
        ("evidence", evidence),
        ("delta_scorer.0", delta_first),
    ):
        _reset_linear_with_named_seed(layer, seed=seed, label=label)
    torch.nn.init.zeros_(delta_last.weight)
    torch.nn.init.zeros_(delta_last.bias)
    model._p37b_initialization_audit = _paired_initialization_audit(model, seed=seed)


def _train_epoch(
    *,
    model: TimeAwareHybrid,
    features: Sequence[MaterializedFeatureAsset],
    preprocessing: FeaturePreprocessor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    loss_sum = 0.0
    correct = 0
    pair_rows = 0
    for feature in features:
        pairs = feature.frozen.pairs
        rng = np.random.default_rng(
            stable_seed(SAMPLING_SEED, feature.frozen.cutoff, "P3.7B", epoch)
        )
        order = rng.permutation(len(pairs))
        for batch_slice in iter_slices(len(order), BATCH_SIZE):
            batch_pairs = pairs[order[batch_slice]]
            positive_rows = batch_pairs[:, 1].astype(np.int64, copy=False)
            negative_rows = batch_pairs[:, 2].astype(np.int64, copy=False)
            joined_rows = np.concatenate([positive_rows, negative_rows])
            batch = _feature_batch(
                feature=feature,
                rows=joined_rows,
                preprocessing=preprocessing,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            count = len(batch_pairs)
            positive_score = output.score[:count]
            negative_score = output.score[count:]
            loss = pairwise_logistic_loss(positive_score, negative_score)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * count
            correct += int(torch.count_nonzero(positive_score > negative_score).detach())
            pair_rows += count
    return {
        "epoch": epoch,
        "bpr_loss": loss_sum / max(pair_rows, 1),
        "pair_accuracy": correct / max(pair_rows, 1),
        "pair_rows": pair_rows,
    }


def _score_model(
    *,
    model: TimeAwareHybrid,
    feature: MaterializedFeatureAsset,
    preprocessing: FeaturePreprocessor,
    device: torch.device,
    output_dir: Path | None = None,
    mask_p33: bool = False,
    keep_alpha: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    started = time.perf_counter()
    rows = len(feature.frozen.candidates["user_index"])
    if output_dir is None:
        scores: np.ndarray = np.empty(rows, dtype=np.float32)
        alpha: np.ndarray | None = np.empty((rows, 5), dtype=np.float16) if keep_alpha else None
        score_path = alpha_path = rank_path = None
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "p33_masked" if mask_p33 else "full"
        score_path = output_dir / f"{preprocessing.variant}_{suffix}_scores.float32.npy"
        rank_path = output_dir / f"{preprocessing.variant}_{suffix}_rank.uint16.npy"
        alpha_path = output_dir / f"{preprocessing.variant}_{suffix}_alpha.float16.npy"
        scores = np.lib.format.open_memmap(score_path, mode="w+", dtype=np.float32, shape=(rows,))
        alpha = (
            np.lib.format.open_memmap(alpha_path, mode="w+", dtype=np.float16, shape=(rows, 5))
            if keep_alpha
            else None
        )
    model.eval()
    with torch.inference_mode():
        for batch_slice in iter_slices(rows, SCORE_BATCH_SIZE):
            row_indices = np.arange(batch_slice.start, batch_slice.stop, dtype=np.int64)
            batch = _feature_batch(
                feature=feature,
                rows=row_indices,
                preprocessing=preprocessing,
                device=device,
                mask_p33=mask_p33,
            )
            output = _forward(model, batch)
            scores[batch_slice] = output.score.cpu().numpy().astype(np.float32)
            if alpha is not None:
                alpha[batch_slice] = output.alpha.cpu().numpy().astype(np.float16)
    if isinstance(scores, np.memmap):
        scores.flush()
    if isinstance(alpha, np.memmap):
        alpha.flush()
    observed_scores = np.asarray(scores).copy() if output_dir is None else np.asarray(scores)
    ranks = _rank_hybrid(feature.frozen.candidates, observed_scores)
    artifacts: dict[str, Any] = {}
    if output_dir is not None:
        assert rank_path is not None and score_path is not None
        np.save(rank_path, ranks, allow_pickle=False)
        artifacts["scores"] = file_identity(score_path)
        artifacts["rank"] = file_identity(rank_path)
        if alpha_path is not None and keep_alpha:
            artifacts["alpha"] = file_identity(alpha_path)
    resource = {
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "mask_p33_auxiliary": mask_p33,
        "artifacts": artifacts,
    }
    return observed_scores, ranks, alpha, resource


def _initial_residual_audit(
    *,
    model: TimeAwareHybrid,
    feature: MaterializedFeatureAsset,
    preprocessing: FeaturePreprocessor,
    device: torch.device,
) -> dict[str, Any]:
    rows = np.arange(min(2048, len(feature.frozen.candidates["rank"])), dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        output = _forward(
            model,
            _feature_batch(
                feature=feature, rows=rows, preprocessing=preprocessing, device=device
            ),
        )
    coarse = feature.frozen.candidates["coarse_score"][rows].astype(np.float32)
    return {
        "rows": int(len(rows)),
        "delta_max_abs": float(torch.max(torch.abs(output.delta)).cpu()),
        "score_vs_raw_m4_max_abs": float(np.max(np.abs(output.score.cpu().numpy() - coarse))),
        "last_layer_weight_zero": bool(
            torch.count_nonzero(model.delta_scorer[-1].weight.detach()).item() == 0
        ),
        "last_layer_bias_zero": bool(
            torch.count_nonzero(model.delta_scorer[-1].bias.detach()).item() == 0
        ),
    }


def _train_inner(
    *,
    window: str,
    variant: str,
    train_feature: MaterializedFeatureAsset,
    validation_feature: MaterializedFeatureAsset,
    output_dir: Path,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    preprocessing = _fit_preprocessing_bundle([train_feature], variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_path = output_dir / f"{variant}_inner_preprocessing.json"
    preprocessing_manifest = preprocessing.to_manifest()
    atomic_json(preprocessing_path, preprocessing_manifest)
    seed = stable_seed(SAMPLING_SEED, window, "inner") % (2**31 - 1)
    model = _new_model(variant, seed=seed, device=device)
    initialization = dict(model._p37b_initialization_audit)
    initial = _initial_residual_audit(
        model=model,
        feature=train_feature,
        preprocessing=preprocessing,
        device=device,
    )
    if initial["delta_max_abs"] != 0.0 or initial["score_vs_raw_m4_max_abs"] != 0.0:
        raise RuntimeError(f"residual initialization drift: {window}/{variant}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    trace: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf)
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_row = _train_epoch(
            model=model,
            features=[train_feature],
            preprocessing=preprocessing,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
        )
        scores, ranks, _alpha, resources = _score_model(
            model=model,
            feature=validation_feature,
            preprocessing=preprocessing,
            device=device,
            keep_alpha=False,
        )
        metrics = _ordering_metrics_fast(
            candidates=validation_feature.frozen.candidates,
            ordering_rank=ranks,
            truth=validation_feature.frozen.truth,
            counts=validation_feature.frozen.counts,
            user_count=len(validation_feature.frozen.users),
        )
        density20 = float(metrics["at_k"]["20"]["positive_density"])
        mrr = float(metrics["ranking"]["mrr"])
        conversion20 = float(metrics["ranking"]["conversion"]["top200_to_top20"])
        key = (density20, mrr)
        trace.append(
            {
                **train_row,
                "inner_density20": density20,
                "inner_mrr": mrr,
                "inner_top200_to_top20": conversion20,
                "scoring_seconds": resources["elapsed_seconds"],
            }
        )
        print(
            f"P3.7B {window} {variant} inner epoch={epoch} loss={train_row['bpr_loss']:.6f} "
            f"density20={density20:.8f} mrr={mrr:.8f}",
            flush=True,
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        del scores, ranks, metrics
        if stale >= PATIENCE:
            break
    if best_state is None or best_epoch < 1:
        raise RuntimeError(f"inner early stopping did not select an epoch: {window}/{variant}")
    model_path = output_dir / f"{variant}_inner_selected.pt"
    torch.save(
        {
            "state_dict": best_state,
            "variant": variant,
            "selected_epoch": best_epoch,
            "seed": seed,
        },
        model_path,
    )
    audit = {
        "variant": variant,
        "training_cutoff": train_feature.frozen.cutoff,
        "inner_validation_cutoff": validation_feature.frozen.cutoff,
        "inner_denominator": "P3.1训练资产保留的未来一周存在cold/sparse truth的用户；不是outer full-user口径。",
        "selection": "lexicographic (inner density@20, inner MRR)",
        "max_epochs": MAX_EPOCHS,
        "patience_consecutive_nonimprovements": PATIENCE,
        "selected_epoch": best_epoch,
        "selected_key": {"density20": best_key[0], "mrr": best_key[1]},
        "trace": trace,
        "initialization": initialization,
        "initial_residual": initial,
        "model": file_identity(model_path),
        "preprocessing": file_identity(preprocessing_path),
        "preprocessing_manifest": preprocessing_manifest,
        "pair_audit": train_feature.frozen.pair_audit,
    }
    del model, optimizer, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return best_epoch, audit


def _train_outer(
    *,
    window: str,
    variant: str,
    training_features: Sequence[MaterializedFeatureAsset],
    epochs: int,
    output_dir: Path,
    device: torch.device,
) -> tuple[TimeAwareHybrid, FeaturePreprocessor, dict[str, Any]]:
    preprocessing = _fit_preprocessing_bundle(training_features, variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_path = output_dir / f"{variant}_outer_preprocessing.json"
    manifest = preprocessing.to_manifest()
    atomic_json(preprocessing_path, manifest)
    seed = stable_seed(SAMPLING_SEED, window, "outer") % (2**31 - 1)
    model = _new_model(variant, seed=seed, device=device)
    initialization = dict(model._p37b_initialization_audit)
    initial = _initial_residual_audit(
        model=model,
        feature=training_features[0],
        preprocessing=preprocessing,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    trace = []
    for epoch in range(1, epochs + 1):
        row = _train_epoch(
            model=model,
            features=training_features,
            preprocessing=preprocessing,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
        )
        trace.append(row)
        print(
            f"P3.7B {window} {variant} outer-refit epoch={epoch}/{epochs} "
            f"loss={row['bpr_loss']:.6f}",
            flush=True,
        )
    model_path = output_dir / f"{variant}_outer.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "variant": variant,
            "epochs": epochs,
            "seed": seed,
        },
        model_path,
    )
    audit = {
        "variant": variant,
        "training_cutoffs": [asset.frozen.cutoff for asset in training_features],
        "epochs_from_inner_early_stopping": epochs,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "pair_batch_size": BATCH_SIZE,
        },
        "loss": "same-user pairwise logistic/BPR",
        "trace": trace,
        "initialization": initialization,
        "initial_residual": initial,
        "model": file_identity(model_path),
        "preprocessing": file_identity(preprocessing_path),
        "preprocessing_manifest": manifest,
        "pair_audits": {
            asset.frozen.cutoff: asset.frozen.pair_audit for asset in training_features
        },
    }
    return model, preprocessing, audit


def _persist_exact_static_score(
    *,
    artifact_root: Path,
    frozen: FrozenAsset,
    scores: np.ndarray,
    ranks: np.ndarray,
) -> dict[str, Any]:
    label = frozen.cutoff if frozen.role == "training" else str(frozen.window)
    root = artifact_root / "static-control-v1" / frozen.role / label
    root.mkdir(parents=True, exist_ok=True)
    score_path = root / "p33_fixed_decay_on_m4_top200.float32.npy"
    rank_path = root / "p33_static_rank.uint16.npy"
    for path, values in ((score_path, scores.astype(np.float32)), (rank_path, ranks.astype(np.uint16))):
        if path.exists():
            existing = np.load(path, mmap_mode="r")
            if not np.array_equal(existing, values):
                raise RuntimeError(f"refusing drifted static-control artifact: {path}")
        else:
            np.save(path, values, allow_pickle=False)
    return {
        "scores": file_identity(score_path),
        "rank": file_identity(rank_path),
        "cutoff_identity": frozen.cutoff,
    }


def _feature_asset_from_root(frozen: FrozenAsset, root: Path) -> MaterializedFeatureAsset:
    paths = {name: root / filename for name, filename in OUTPUT_FILENAMES.items()}
    manifest_path = root / "manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("cutoff") != frozen.cutoff or manifest.get("final_week") != "not_run":
        raise RuntimeError(f"raw feature cutoff drift: {root}")
    for name, path in paths.items():
        observed = file_identity(path)
        expected = manifest["artifacts"][name]
        if observed["sha256"] != expected["sha256"] or observed["bytes"] != expected["bytes"]:
            raise RuntimeError(f"raw feature identity drift: {root}/{name}")
    return MaterializedFeatureAsset(
        frozen=frozen,
        root=root,
        m4_relation_path=paths["m4_relation"],
        p33_relation_path=paths["p33_relation"],
        p33_global_path=paths["p33_global"],
        user_state_path=paths["user_state"],
        candidate_state_path=paths["candidate_state"],
        manifest_path=manifest_path,
    )


def _materialize_feature_asset(
    *,
    frozen: FrozenAsset,
    artifact_root: Path,
    catalog_product_type: np.ndarray,
    catalog_garment_group: np.ndarray,
    exact_p33_scores: np.ndarray,
    device: torch.device,
) -> MaterializedFeatureAsset:
    label = frozen.cutoff if frozen.role == "training" else str(frozen.window)
    root = artifact_root / "features-v1" / frozen.role / label
    if (root / "manifest.json").is_file():
        return _feature_asset_from_root(frozen, root)
    print(f"P3.7B {frozen.role_key}: materialize fixed raw features", flush=True)
    build_raw_feature_assets(
        cutoff=frozen.cutoff,
        candidates=frozen.candidates,
        histories=frozen.histories,
        m4_embeddings=np.load(frozen.m4_embedding_path, mmap_mode="r"),
        p33_embeddings=np.load(frozen.p33_embedding_path, mmap_mode="r"),
        catalog_product_type=catalog_product_type,
        catalog_garment_group=catalog_garment_group,
        interaction_counts=frozen.counts,
        output_dir=root,
        source_artifacts={
            "candidates": frozen.candidate_path,
            "histories": frozen.history_path,
            "users": frozen.users_path,
            "m4_embeddings": frozen.m4_embedding_path,
            "p33_embeddings": frozen.p33_embedding_path,
            "p31_manifest": frozen.root / "manifest.json",
            "p33_manifest": frozen.p33_manifest_path,
        },
        lineage_audit=frozen.identity_audit,
        device=device,
        batch_users=32,
        authoritative_p33_fixed_decay_scores=exact_p33_scores,
    )
    return _feature_asset_from_root(frozen, root)


def _metric_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "density20": _metric_value(candidate, "density20") - _metric_value(baseline, "density20"),
        "mrr": _metric_value(candidate, "mrr") - _metric_value(baseline, "mrr"),
        "top200_to_top20": _metric_value(candidate, "conversion20") - _metric_value(baseline, "conversion20"),
    }


def _candidate_identity_row(
    frozen: FrozenAsset,
    variant_metrics: dict[str, dict[str, Any]],
    variant_ranks: dict[str, np.ndarray],
) -> dict[str, Any]:
    identity = frozen.identity_audit["candidate_identity_sha256"]
    rows = len(frozen.candidates["user_index"])
    checks: dict[str, Any] = {}
    for variant in VARIANTS:
        ranks = variant_ranks[variant]
        per_group = ranks.reshape(-1, K0)
        checks[variant] = {
            "candidate_identity_sha256": identity,
            "candidate_rows": rows,
            "per_user_rank_permutation_1_200": bool(
                np.all(np.sort(per_group, axis=1) == np.arange(1, K0 + 1)[None, :])
            ),
            "top200_truth_pairs": int(variant_metrics[variant]["at_k"]["200"]["positive_rows"]),
            "strict_recall200": float(
                variant_metrics[variant]["at_k"]["200"]["segments"]["strict_cold"]["recall"]
            ),
            "sparse_recall200": float(
                variant_metrics[variant]["at_k"]["200"]["segments"]["sparse_1_5"]["recall"]
            ),
        }
    baseline = checks["C0"]
    exact = all(
        row["candidate_identity_sha256"] == baseline["candidate_identity_sha256"]
        and row["candidate_rows"] == baseline["candidate_rows"]
        and row["top200_truth_pairs"] == baseline["top200_truth_pairs"]
        and row["strict_recall200"] == baseline["strict_recall200"]
        and row["sparse_recall200"] == baseline["sparse_recall200"]
        and row["per_user_rank_permutation_1_200"]
        for row in checks.values()
    )
    return {
        "definition": "四个角色共享同一份冻结P3.1 M4 Top200用户—商品行；摘要哈希的统计对象是按原行序连接的user_index与catalog_row。",
        "variants": checks,
        "all_variants_exact": exact,
        "candidate_identity_exact": exact,
        "candidate_rows_exact": exact,
        "recall200_exact": exact,
        "top200_truth_pairs_exact": exact,
        "m4_cutoff_safe": True,
        "p33_cutoff_safe": True,
    }


def _append_large_artifacts(
    output: list[dict[str, Any]], identities: Iterable[dict[str, Any]], cutoff: str
) -> None:
    seen = {str(row["path"]) for row in output}
    for identity in identities:
        if str(identity["path"]) in seen:
            continue
        output.append({**identity, "cutoff_identity": cutoff})
        seen.add(str(identity["path"]))


def run_p37b(
    *,
    source_root: Path,
    repo_root: Path,
    artifact_dir: Path,
    report_dir: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    artifact_dir = artifact_dir.resolve()
    report_dir = report_dir.resolve()
    contract_path = report_dir / "P3_7B_EXPERIMENT_CONTRACT.json"
    feature_contract_path = report_dir / "p3_7b_feature_contract.json"
    contract = _json(contract_path)
    if contract.get("status") != "preregistered_before_formal_computation":
        raise RuntimeError("P3.7B experiment contract is not preregistered")
    if contract.get("run_id") != RUN_ID or contract.get("final_week") != "not_run":
        raise RuntimeError("P3.7B experiment contract drift")
    if tuple(ROLLING_PROTOCOL) != WINDOWS:
        raise RuntimeError("rolling window protocol drift")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not visible")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False

    p31_path = report_dir / "P3_1_metrics.json"
    p33_path = report_dir / "P3_3_metrics.json"
    p37a_path = report_dir / "P3_7A_2X2_DIAGNOSTIC.json"
    p36_path = report_dir / "P3_6_SIMILARITY_COMPLEMENTARITY_AUDIT.json"
    p31, p33, p37a, p36 = map(_json, (p31_path, p33_path, p37a_path, p36_path))
    if any(row.get("final_week") != "not_run" for row in (p31, p33, p37a, p36)):
        raise RuntimeError("authoritative input crossed the final-week boundary")
    if p37a.get("verdicts", {}).get("p33_auxiliary_view_for_p37b") != "allowed":
        raise RuntimeError("P3.7A did not allow P3.3 as an auxiliary view")
    transactions_path = source_root / "data" / "interim" / "audit" / "transactions.parquet"
    transaction_identity = file_identity(transactions_path)
    declared_transactions = p36["input_identities"]["transactions"]
    if (
        transaction_identity["sha256"] != declared_transactions["sha256"]
        or transaction_identity["bytes"] != declared_transactions["bytes"]
    ):
        raise RuntimeError("transaction parquet identity drift")
    static_dir = (
        repo_root
        / "artifacts"
        / "m4"
        / "m4-v1-supervised-cold-representation"
        / "student-v1"
        / "static_catalog"
    )
    catalog_frame = pd.read_csv(
        static_dir / "catalog_items.csv", dtype={"article_id": str}
    ).sort_values("catalog_row")
    catalog_items = catalog_frame["article_id"].astype(str).tolist()
    attributes = pd.read_csv(
        source_root / "data" / "raw" / "articles.csv",
        dtype={"article_id": str},
        usecols=["article_id", "product_type_no", "garment_group_no"],
    )
    aligned = catalog_frame[["article_id", "catalog_row"]].merge(
        attributes, on="article_id", how="left", validate="one_to_one"
    ).sort_values("catalog_row")
    product_type = aligned["product_type_no"].fillna(-1).to_numpy(dtype=np.int32)
    garment_group = aligned["garment_group_no"].fillna(-1).to_numpy(dtype=np.int32)

    frozen_training: dict[str, FrozenAsset] = {}
    for cutoff in sorted(
        {value for protocol in ROLLING_PROTOCOL.values() for value in protocol["outer_train"]}
    ):
        print(f"P3.7B load and reconstruct training cutoff {cutoff}", flush=True)
        frozen_training[cutoff] = _load_frozen_asset(
            repo_root=repo_root,
            p31=p31,
            transactions_path=transactions_path,
            catalog_items=catalog_items,
            role="training",
            cutoff=cutoff,
            window=None,
        )
    frozen_outer: dict[str, FrozenAsset] = {}
    for window, protocol in ROLLING_PROTOCOL.items():
        print(f"P3.7B load and reconstruct outer window {window}", flush=True)
        frozen_outer[window] = _load_frozen_asset(
            repo_root=repo_root,
            p31=p31,
            transactions_path=transactions_path,
            catalog_items=catalog_items,
            role="outer_validation",
            cutoff=str(protocol["outer_validation"]),
            window=window,
        )

    static_controls: dict[str, dict[str, Any]] = {}
    exact_scores: dict[str, np.ndarray] = {}
    exact_ranks: dict[str, np.ndarray] = {}
    exact_metrics: dict[str, dict[str, Any]] = {}
    all_frozen = [*frozen_training.values(), *frozen_outer.values()]
    for frozen in all_frozen:
        label = frozen.cutoff if frozen.role == "training" else str(frozen.window)
        control_root = artifact_dir / "static-control-v1" / frozen.role / label
        cached_score_path = control_root / "p33_fixed_decay_on_m4_top200.float32.npy"
        cached_rank_path = control_root / "p33_static_rank.uint16.npy"
        if cached_score_path.is_file() and cached_rank_path.is_file():
            print(f"P3.7B {frozen.role_key}: reuse SHA-bound exact P3.3 static score", flush=True)
            scores = np.load(cached_score_path, mmap_mode="r")
            ranks = np.load(cached_rank_path, mmap_mode="r")
            if len(scores) != len(frozen.candidates["rank"]) or len(ranks) != len(scores):
                raise RuntimeError(f"static-control cached shape drift: {frozen.role_key}")
            metrics = _ordering_metrics_fast(
                candidates=frozen.candidates,
                ordering_rank=np.asarray(ranks),
                truth=frozen.truth,
                counts=frozen.counts,
                user_count=len(frozen.users),
            )
            p37a_exact = None
            if frozen.role == "outer_validation":
                assert frozen.window is not None
                p37a_exact = metrics == p37a["four_variant_metrics"]["windows"][frozen.window]["A01"]
                if not p37a_exact:
                    raise RuntimeError(f"cached C1 no longer reproduces P3.7A: {frozen.window}")
            audit = {
                "definition": "复用首次正式运行已经由完整冷/稀疏目录float32数值核生成的静态P3.3分数；当前重跑重新核对文件身份、行数、名次与外层P3.7A指标。",
                "reused_after_report_serialization_failure": True,
                "formal_candidate_source_is_M4_only": True,
                "p3_7a_A01_metrics_exact": p37a_exact,
            }
            control_artifacts = {
                "scores": file_identity(cached_score_path),
                "rank": file_identity(cached_rank_path),
                "cutoff_identity": frozen.cutoff,
            }
        else:
            print(f"P3.7B {frozen.role_key}: reconstruct exact P3.3 static score", flush=True)
            scores, ranks, metrics, audit = _exact_p33_static_score(
                frozen=frozen,
                p33_metrics=p33,
                p37a=p37a if frozen.role == "outer_validation" else None,
                device=device,
            )
            control_artifacts = _persist_exact_static_score(
                artifact_root=artifact_dir, frozen=frozen, scores=scores, ranks=ranks
            )
        exact_scores[frozen.role_key] = np.load(
            control_artifacts["scores"]["path"], mmap_mode="r"
        )
        exact_ranks[frozen.role_key] = np.load(
            control_artifacts["rank"]["path"], mmap_mode="r"
        )
        exact_metrics[frozen.role_key] = metrics
        static_controls[frozen.role_key] = {"audit": audit, "artifacts": control_artifacts}

    feature_training: dict[str, MaterializedFeatureAsset] = {}
    feature_outer: dict[str, MaterializedFeatureAsset] = {}
    for cutoff, frozen in frozen_training.items():
        feature_training[cutoff] = _materialize_feature_asset(
            frozen=frozen,
            artifact_root=artifact_dir,
            catalog_product_type=product_type,
            catalog_garment_group=garment_group,
            exact_p33_scores=exact_scores[frozen.role_key],
            device=device,
        )
    for window, frozen in frozen_outer.items():
        feature_outer[window] = _materialize_feature_asset(
            frozen=frozen,
            artifact_root=artifact_dir,
            catalog_product_type=product_type,
            catalog_garment_group=garment_group,
            exact_p33_scores=exact_scores[frozen.role_key],
            device=device,
        )

    windows: dict[str, Any] = {}
    training_windows: dict[str, Any] = {}
    identity_windows: dict[str, Any] = {}
    time_gate_windows: dict[str, Any] = {}
    auxiliary_windows: dict[str, Any] = {}
    large_artifacts: list[dict[str, Any]] = []
    for feature in [*feature_training.values(), *feature_outer.values()]:
        manifest = _json(feature.manifest_path)
        _append_large_artifacts(
            large_artifacts,
            (file_identity(feature.manifest_path),),
            feature.frozen.cutoff,
        )
        _append_large_artifacts(
            large_artifacts,
            manifest["artifacts"].values(),
            feature.frozen.cutoff,
        )
        controls = static_controls[feature.frozen.role_key]["artifacts"]
        _append_large_artifacts(
            large_artifacts,
            (controls["scores"], controls["rank"]),
            feature.frozen.cutoff,
        )

    for window, protocol in ROLLING_PROTOCOL.items():
        print(f"P3.7B formal rolling chain {window}", flush=True)
        inner_train = feature_training[str(protocol["inner_train"][0])]
        inner_valid = feature_training[str(protocol["inner_validation"])]
        outer_train = [feature_training[str(cutoff)] for cutoff in protocol["outer_train"]]
        outer = feature_outer[window]
        model_dir = artifact_dir / "models-v1" / window
        score_dir = artifact_dir / "scores-v1" / window
        variant_metrics: dict[str, dict[str, Any]] = {}
        variant_ranks: dict[str, np.ndarray] = {
            "C0": outer.frozen.candidates["rank"].astype(np.uint16, copy=False),
            "C1": np.asarray(exact_ranks[outer.frozen.role_key], dtype=np.uint16),
        }
        c0_metrics = _ordering_metrics_fast(
            candidates=outer.frozen.candidates,
            ordering_rank=variant_ranks["C0"],
            truth=outer.frozen.truth,
            counts=outer.frozen.counts,
            user_count=len(outer.frozen.users),
        )
        expected_c0 = p37a["four_variant_metrics"]["windows"][window]["A00"]
        if c0_metrics != expected_c0:
            raise RuntimeError(f"C0 did not exactly reproduce P3.7A A00: {window}")
        variant_metrics["C0"] = c0_metrics
        variant_metrics["C1"] = exact_metrics[outer.frozen.role_key]
        window_training: dict[str, Any] = {}
        window_gate: dict[str, Any] = {}
        variant_scores: dict[str, np.ndarray] = {}
        variant_alphas: dict[str, np.ndarray] = {}
        outer_preprocessing: dict[str, FeaturePreprocessor] = {}
        for variant in ("B0", "B1"):
            selected_epoch, inner_audit = _train_inner(
                window=window,
                variant=variant,
                train_feature=inner_train,
                validation_feature=inner_valid,
                output_dir=model_dir,
                device=device,
            )
            model, preprocessing, outer_audit = _train_outer(
                window=window,
                variant=variant,
                training_features=outer_train,
                epochs=selected_epoch,
                output_dir=model_dir,
                device=device,
            )
            scores, ranks, alpha, scoring = _score_model(
                model=model,
                feature=outer,
                preprocessing=preprocessing,
                device=device,
                output_dir=score_dir,
                keep_alpha=True,
            )
            assert alpha is not None
            metrics = _ordering_metrics_fast(
                candidates=outer.frozen.candidates,
                ordering_rank=ranks,
                truth=outer.frozen.truth,
                counts=outer.frozen.counts,
                user_count=len(outer.frozen.users),
            )
            variant_metrics[variant] = metrics
            variant_ranks[variant] = ranks
            variant_scores[variant] = scores
            variant_alphas[variant] = np.asarray(alpha)
            outer_preprocessing[variant] = preprocessing
            raw_user_state = np.load(outer.user_state_path, mmap_mode="r")
            window_gate[variant] = _time_gate_mechanism_audit(
                frozen=outer.frozen,
                raw_user_state=raw_user_state,
                alpha=np.asarray(alpha),
                hybrid_rank=ranks,
                profile_quartiles=_profile_quartiles(outer_train),
            )
            window_training[variant] = {
                "inner": inner_audit,
                "outer": outer_audit,
                "score_diagnostics": _score_diagnostics(outer.frozen.candidates, scores),
                "scoring": scoring,
            }
            for identity in scoring["artifacts"].values():
                _append_large_artifacts(large_artifacts, (identity,), outer.frozen.cutoff)
            _append_large_artifacts(
                large_artifacts,
                (outer_audit["model"],),
                outer.frozen.cutoff,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        b0_manifest = outer_preprocessing["B0"].to_manifest()
        b1_manifest = outer_preprocessing["B1"].to_manifest()
        preprocessing_parity = {
            "user_state": b0_manifest["groups"]["user_state"] == b1_manifest["groups"]["user_state"],
            "candidate_state": b0_manifest["groups"]["candidate_state"] == b1_manifest["groups"]["candidate_state"],
            "m4_relation": all(
                b0_manifest["groups"]["relation"]["statistics"][spec.name]
                == b1_manifest["groups"]["relation"]["statistics"][spec.name]
                for spec in M4_RELATION_SPECS
            ),
            "B0_has_zero_p33_fields": b0_manifest["groups"]["p33_global"] is None
            and len(b0_manifest["groups"]["relation"]["features"]) == len(M4_RELATION_SPECS),
            "B1_only_adds_declared_auxiliary_fields": len(
                b1_manifest["groups"]["relation"]["features"]
            )
            == len(M4_RELATION_SPECS) + len(P33_RELATION_SPECS),
        }
        if not all(preprocessing_parity.values()):
            raise RuntimeError(f"B0/B1 preprocessing attribution drift: {window}")

        masked_scores, masked_rank, _masked_alpha, masked_resource = _score_model(
            model=_reload_outer_model(
                variant="B1", model_path=Path(window_training["B1"]["outer"]["model"]["path"]), device=device
            ),
            feature=outer,
            preprocessing=outer_preprocessing["B1"],
            device=device,
            mask_p33=True,
            keep_alpha=False,
        )
        masked_metrics = _ordering_metrics_fast(
            candidates=outer.frozen.candidates,
            ordering_rank=masked_rank,
            truth=outer.frozen.truth,
            counts=outer.frozen.counts,
            user_count=len(outer.frozen.users),
        )
        auxiliary_windows[window] = {
            "definition": "将B1全部4个分桶P3.3字段与2个全局P3.3字段在标准化空间置0（训练均值），不重训模型；比较分母仍为同一冻结M4 Top200。",
            "B1_full": {
                "density20": _metric_value(variant_metrics["B1"], "density20"),
                "mrr": _metric_value(variant_metrics["B1"], "mrr"),
                "top200_to_top20": _metric_value(variant_metrics["B1"], "conversion20"),
            },
            "B1_p33_auxiliary_masked": {
                "density20": _metric_value(masked_metrics, "density20"),
                "mrr": _metric_value(masked_metrics, "mrr"),
                "top200_to_top20": _metric_value(masked_metrics, "conversion20"),
            },
            "full_minus_masked": _metric_delta(variant_metrics["B1"], masked_metrics),
            "scoring": masked_resource,
            "promotion_metric": False,
        }
        identity = _candidate_identity_row(outer.frozen, variant_metrics, variant_ranks)
        if not identity["all_variants_exact"]:
            raise RuntimeError(f"candidate/Recall200 conservation failed: {window}")
        windows[window] = {
            "cutoff": str(protocol["outer_validation"]),
            "variants": variant_metrics,
            "candidate_identity": identity,
            "score_diagnostics": {
                variant: window_training[variant]["score_diagnostics"] for variant in ("B0", "B1")
            },
            "p3_7a_parity": {
                "C0_A00_metrics_exact": variant_metrics["C0"]
                == p37a["four_variant_metrics"]["windows"][window]["A00"],
                "C1_A01_metrics_exact": variant_metrics["C1"]
                == p37a["four_variant_metrics"]["windows"][window]["A01"],
                "static_score_audit": static_controls[outer.frozen.role_key]["audit"],
            },
            "preprocessing_parity": preprocessing_parity,
        }
        training_windows[window] = window_training
        identity_windows[window] = identity
        time_gate_windows[window] = window_gate
        del variant_scores, variant_alphas, masked_scores, masked_rank, masked_metrics
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    b0_gate = _formal_gate(windows, "B0")
    b1_gate = _formal_gate(windows, "B1")
    complexity = _complexity_gate(windows)
    selection = _b1_vs_b0_selection(windows)
    decision = _final_decision(b0_gate, b1_gate, complexity, selection)
    fusion_allowed = decision["selected_variant"] in {"B0", "B1"} and (
        b0_gate["passed"] if decision["selected_variant"] == "B0" else b1_gate["passed"]
    )
    decision["warm_cold_fusion_allowed"] = bool(fusion_allowed)
    gates = {
        "B0": b0_gate,
        "B1": b1_gate,
        "B1_vs_C1": complexity,
        "B1_vs_B0_selection_if_both_supported": selection,
    }
    candidate_identity_audit = {
        "schema_version": "phase3-p3.7b-candidate-identity-audit-v1",
        "status": "measured",
        "windows": identity_windows,
        "all_windows_exact": all(row["all_variants_exact"] for row in identity_windows.values()),
        "final_week": "not_run",
    }
    training_audit = {
        "schema_version": "phase3-p3.7b-training-audit-v1",
        "status": "measured",
        "sampler": {
            "exact_function_reuse": "hm_recsys.p3_model.sample_same_user_pairs",
            "all_positives_retained": True,
            "same_user_only": True,
            "rank_buckets": ["1-67", "68-134", "135-200"],
            "maximum_negatives_per_positive": NEGATIVES_PER_POSITIVE,
            "fixed_hash_seed": SAMPLING_SEED,
        },
        "windows": training_windows,
        "static_control_assets": static_controls,
        "final_week": "not_run",
    }
    time_gate_audit = {
        "schema_version": "phase3-p3.7b-time-gate-audit-v1",
        "status": "measured",
        "windows": time_gate_windows,
        "promotion_metric": False,
        "final_week": "not_run",
    }
    p33_auxiliary_audit = {
        "schema_version": "phase3-p3.7b-p33-auxiliary-audit-v1",
        "status": "measured",
        "windows": auxiliary_windows,
        "promotion_metric": False,
        "final_week": "not_run",
    }
    rank_funnel = {
        "schema_version": "phase3-p3.7b-rank-funnel-v1",
        "status": "measured",
        "definition": "每窗、每角色在同一冻结M4 Top200上的完整K值密度、分群召回/命中与Top200到TopK正例转化。",
        "windows": {window: row["variants"] for window, row in windows.items()},
        "final_week": "not_run",
    }
    master = {
        "schema_version": "phase3-p3.7b-metrics-v1",
        "stage": STAGE,
        "status": "measured",
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": file_identity(contract_path),
        "feature_contract": file_identity(feature_contract_path),
        "authoritative_inputs": {
            "P3_1_metrics": file_identity(p31_path),
            "P3_3_metrics": file_identity(p33_path),
            "P3_7A_2x2": file_identity(p37a_path),
            "P3_6_relation_audit": file_identity(p36_path),
            "transactions": transaction_identity,
            "articles": file_identity(source_root / "data" / "raw" / "articles.csv"),
            "static_catalog": file_identity(static_dir / "catalog_items.csv"),
        },
        "contract_summary": {
            "coarse_retriever": "M4 single-teacher content-only Student only",
            "candidate_set": "exact frozen P3.1 M4-space Top200",
            "p33_role": "auxiliary scoring geometry only; never candidate generation",
            "age_buckets_days": ["0-7", "8-28", "29-84", ">84"],
            "history": "cutoff-before latest 20 distinct purchased items",
            "final_week": "not_run",
        },
        "windows": windows,
        "gates": gates,
        "decision": decision,
        "training_audit": training_audit,
        "candidate_identity_audit": candidate_identity_audit,
        "time_gate_audit": time_gate_audit,
        "p33_auxiliary_audit": p33_auxiliary_audit,
        "rank_funnel": rank_funnel,
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
            "p33_replay_performed": False,
            "feature_artifact_bytes": int(
                sum(row["bytes"] for row in large_artifacts if "features-v1" in str(row["path"]))
            ),
        },
        "large_artifacts": large_artifacts,
        "final_week": "not_run",
    }
    master = _json_native(master)
    write_outputs(
        report_dir,
        master,
        training_audit=training_audit,
        candidate_identity_audit=candidate_identity_audit,
        time_gate_audit=time_gate_audit,
        p33_auxiliary_audit=p33_auxiliary_audit,
        rank_funnel=rank_funnel,
        large_artifacts=large_artifacts,
    )
    return master


def _reload_outer_model(*, variant: str, model_path: Path, device: torch.device) -> TimeAwareHybrid:
    payload = torch.load(model_path, map_location=device, weights_only=True)
    model = _new_model(variant, seed=int(payload["seed"]), device=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def run_with_failure_record(**kwargs: Any) -> dict[str, Any]:
    try:
        return run_p37b(**kwargs)
    except Exception as exc:
        report_dir = Path(kwargs["report_dir"]).resolve()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        atomic_json(
            report_dir / f"P3_7B_FAILURE_{timestamp}.json",
            {
                "schema_version": "phase3-p3.7b-failure-v1",
                "stage": STAGE,
                "status": "failed_closed",
                "run_id": RUN_ID,
                "decision": "engineering_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "final_week": "not_run",
            },
        )
        raise
