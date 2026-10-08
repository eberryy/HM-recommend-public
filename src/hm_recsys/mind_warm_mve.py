"""Bounded, cutoff-safe MIND-style warm retrieval experiment.

The item table is the frozen cutoff-specific Item2Vec representation already
used by the baseline.  This experiment learns only the user-side projection,
dynamic-routing representation and, for the primary variant, separate behavior
age weights for each interest capsule.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
TX = ROOT / "data" / "interim" / "audit" / "transactions.parquet"
ARTICLES = ROOT / "data" / "raw" / "articles.csv"
REPORT_DIR = ROOT / "reports" / "mind_warm_side"
CONTRACT_PATH = REPORT_DIR / "MIND_WARM_MVE_CONTRACT.json"
METRICS_PATH = REPORT_DIR / "MIND_WARM_MVE_METRICS.json"
REPORT_PATH = REPORT_DIR / "MIND_WARM_MVE_FINAL.md"
ARTIFACT_ROOT = ROOT / "artifacts" / "mind_warm_side" / "MIND-WARM-MVE-001"

MECHANICS = {"mechanics_20191225": "2019-12-25"}
OUTER = {
    "winter_20200122": "2020-01-22",
    "spring_20200318": "2020-03-18",
    "early_summer_20200624": "2020-06-24",
    "late_summer_20200819": "2020-08-19",
}
ALL_WINDOWS = {**MECHANICS, **OUTER}
VARIANTS = {
    "single_learned_age": {"maximum_interests": 1, "time_mode": "learned_buckets"},
    "mind_learned_age": {"maximum_interests": 3, "time_mode": "learned_buckets"},
    "mind_no_time": {"maximum_interests": 3, "time_mode": "none"},
    "mind_fixed_28d": {"maximum_interests": 3, "time_mode": "fixed_28d"},
}
MODEL_SCHEMA = "mind-warm-mve-model-v1"
PREP_SCHEMA = "mind-warm-mve-prepared-v1"


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_native(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _guard(cutoff: str) -> date:
    value = date.fromisoformat(cutoff)
    if value >= date(2020, 9, 16):
        raise ValueError("final week 2020-09-16 and later are forbidden")
    return value


def _connection(memory: str = "8GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute(f"SET memory_limit='{memory}'")
    return con


def candidate_path(cutoff: str) -> Path:
    return ROOT / "artifacts" / "m2_9" / "cache-v1" / cutoff / "expanded-candidates.parquet"


def _source_manifest(cutoff: str) -> Path:
    return ROOT / "artifacts" / "m2_9" / "item2vec-source-v1" / cutoff / "source-manifest.json"


def load_item_table(cutoff: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    _guard(cutoff)
    manifest_path = _source_manifest(cutoff)
    manifest = _read_json(manifest_path)
    if manifest.get("cutoff") != cutoff:
        raise AssertionError("Item2Vec manifest cutoff mismatch")
    paths: dict[str, Path] = {}
    for key in ("items", "normalized_vectors"):
        entry = manifest["artifacts"][key]
        path = Path(entry["path"])
        if not path.exists():
            path = manifest_path.parent / path.name
        if not path.exists() or path.stat().st_size != int(entry["bytes"]):
            raise FileNotFoundError(f"invalid Item2Vec {key} artifact: {path}")
        paths[key] = path
    items = pd.read_csv(paths["items"], dtype={"article_id": str})
    vectors = np.load(paths["normalized_vectors"], allow_pickle=False).astype(np.float32, copy=False)
    if not np.array_equal(items["row_index"].to_numpy(), np.arange(len(items))):
        raise AssertionError("Item2Vec row index is not dense and ordered")
    if vectors.shape != (len(items), 64) or not np.isfinite(vectors).all():
        raise AssertionError("unexpected Item2Vec matrix")
    return items, vectors, {
        "manifest": str(manifest_path.resolve()),
        "items": str(paths["items"].resolve()),
        "vectors": str(paths["normalized_vectors"].resolve()),
        "vocabulary_items": len(items),
    }


def dynamic_interest_count(history_length: torch.Tensor, maximum: int) -> torch.Tensor:
    """Return 1 for sparse histories, otherwise floor(log2(n)) bounded to 1..K."""
    lengths = history_length.to(torch.float32).clamp_min(1)
    values = torch.floor(torch.log2(lengths)).to(torch.long).clamp(1, maximum)
    return torch.where(history_length < 5, torch.ones_like(values), values)


def age_bucket(age_days: torch.Tensor) -> torch.Tensor:
    """Map positive age in days to 0-7, 8-28, or 29-84-day bucket indices."""
    return torch.where(age_days <= 7, 0, torch.where(age_days <= 28, 1, 2)).to(torch.long)


class MindEncoder(nn.Module):
    """Small MIND-style dynamic-routing encoder over frozen item embeddings."""

    def __init__(self, maximum_interests: int, time_mode: str, dimension: int = 64) -> None:
        super().__init__()
        if maximum_interests not in (1, 3):
            raise ValueError("this MVE registers only K=1 or K=3")
        if time_mode not in ("learned_buckets", "none", "fixed_28d"):
            raise ValueError(time_mode)
        self.maximum_interests = maximum_interests
        self.time_mode = time_mode
        self.dimension = dimension
        eye = torch.eye(dimension).repeat(maximum_interests, 1, 1)
        noise = torch.randn_like(eye) * 0.01
        self.projection = nn.Parameter(eye + noise)
        self.routing_seed = nn.Parameter(torch.randn(maximum_interests, dimension) * 0.02)
        self.age_logits = (
            nn.Parameter(torch.zeros(maximum_interests, 3))
            if time_mode == "learned_buckets"
            else None
        )

    def age_weights(self, ages: torch.Tensor) -> torch.Tensor:
        """Return B x K x N non-negative history weights."""
        if self.time_mode == "learned_buckets":
            gates = torch.softmax(self.age_logits, dim=1) * 3.0
            buckets = age_bucket(ages)
            return gates[:, buckets].permute(1, 0, 2)
        if self.time_mode == "fixed_28d":
            return torch.exp(-math.log(2.0) * ages.to(torch.float32) / 28.0).unsqueeze(1)
        return torch.ones((*ages.shape[:1], 1, ages.shape[1]), dtype=torch.float32, device=ages.device)

    def forward(
        self,
        item_vectors: torch.Tensor,
        history_ids: torch.Tensor,
        ages: torch.Tensor,
        lengths: torch.Tensor,
        routing_iterations: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_ids.ndim != 2 or ages.shape != history_ids.shape:
            raise ValueError("history_ids and ages must be B x N")
        positions = torch.arange(history_ids.shape[1], device=history_ids.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        safe_ids = history_ids.clamp_min(0)
        history = item_vectors[safe_ids]
        projected = torch.einsum("bnd,ked->bkne", history, self.projection)
        weights = self.age_weights(ages).expand(-1, self.maximum_interests, -1)
        weights = weights * valid.unsqueeze(1)
        interest_count = dynamic_interest_count(lengths, self.maximum_interests)
        active = (
            torch.arange(self.maximum_interests, device=history_ids.device).unsqueeze(0)
            < interest_count.unsqueeze(1)
        )
        logits = torch.einsum("bknd,kd->bkn", projected, self.routing_seed)
        logits = logits.masked_fill(~active.unsqueeze(2), -1e4)
        interests = torch.zeros(
            history_ids.shape[0], self.maximum_interests, self.dimension,
            device=history_ids.device, dtype=history.dtype,
        )
        for iteration in range(routing_iterations):
            assignment = torch.softmax(logits, dim=1) * valid.unsqueeze(1)
            combined = assignment * weights
            denominator = combined.sum(dim=2, keepdim=True).clamp_min(1e-6)
            interests = (combined.unsqueeze(3) * projected).sum(dim=2) / denominator
            interests = F.normalize(interests, dim=2)
            interests = interests * active.unsqueeze(2)
            if iteration + 1 < routing_iterations:
                logits = logits + torch.einsum("bknd,bkd->bkn", projected, interests)
                logits = logits.masked_fill(~active.unsqueeze(2), -1e4)
        return interests, active

    def learned_age_weights(self) -> list[list[float]] | None:
        if self.age_logits is None:
            return None
        return (torch.softmax(self.age_logits.detach().cpu(), dim=1) * 3.0).tolist()


@dataclass
class ReservoirRow:
    history: np.ndarray
    ages: np.ndarray
    positive: int
    target_group: int


def _iter_user_ranges(customers: np.ndarray) -> Iterable[tuple[int, int]]:
    if not len(customers):
        return
    starts = np.r_[0, np.flatnonzero(customers[1:] != customers[:-1]) + 1]
    ends = np.r_[starts[1:], len(customers)]
    yield from zip(starts.tolist(), ends.tolist())


def prepare_training(cutoff: str, contract: dict[str, Any]) -> dict[str, Any]:
    """Scan all eligible examples, then retain a seeded reservoir of at most 250k."""
    _guard(cutoff)
    out = ARTIFACT_ROOT / cutoff
    marker = out / "PREPARED.json"
    arrays_path = out / "prepared.npz"
    params = {
        "schema": PREP_SCHEMA,
        "history_days": 84,
        "target_days": 84,
        "maximum_history": 50,
        "maximum_training_examples": int(contract["training_data"]["maximum_training_examples"]),
        "seed": int(contract["shared_model"]["random_seed"]),
    }
    if marker.exists() and arrays_path.exists():
        saved = _read_json(marker)
        if saved.get("params") != params:
            raise AssertionError("stale MIND prepared cache has a different contract")
        return saved

    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    items, _, source = load_item_table(cutoff)
    with _connection() as con:
        con.register("vocabulary", items[["article_id", "row_index"]])
        frame = con.execute(
            f"""SELECT x.customer_id,
                date_diff('day',DATE '1970-01-01',x.t_dat)::INTEGER AS day,
                v.row_index::INTEGER AS item
            FROM (
                SELECT DISTINCT customer_id,t_dat,article_id
                FROM read_parquet('{_sql_path(TX)}')
                WHERE t_dat>=DATE '{cutoff}'-INTERVAL 168 DAY AND t_dat<DATE '{cutoff}'
            ) x JOIN vocabulary v USING(article_id)
            ORDER BY x.customer_id,day,item"""
        ).fetchdf()
    if frame.empty:
        raise RuntimeError("no cutoff-safe training interactions")
    cutoff_day = int(np.datetime64(cutoff, "D").astype(np.int32))
    target_start = cutoff_day - params["target_days"]
    customers = frame["customer_id"].to_numpy()
    days = frame["day"].to_numpy(np.int32)
    item_ids = frame["item"].to_numpy(np.int32)
    rng = np.random.default_rng(params["seed"])
    reservoir: list[ReservoirRow] = []
    target_groups: list[np.ndarray] = []
    eligible_examples = 0
    eligible_target_days = 0
    maximum = params["maximum_training_examples"]
    maximum_history = params["maximum_history"]
    for lo, hi in _iter_user_ranges(customers):
        user_days = days[lo:hi]
        boundaries = np.r_[0, np.flatnonzero(user_days[1:] != user_days[:-1]) + 1, hi - lo]
        for day_position in range(1, len(boundaries) - 1):
            target_lo = lo + int(boundaries[day_position])
            target_hi = lo + int(boundaries[day_position + 1])
            target_day = int(days[target_lo])
            if target_day < target_start:
                continue
            prior_lo = lo + int(np.searchsorted(user_days, target_day - 84, side="left"))
            prior_hi = target_lo
            if prior_hi <= prior_lo:
                continue
            hist_items = item_ids[prior_lo:prior_hi][-maximum_history:]
            hist_days = days[prior_lo:prior_hi][-maximum_history:]
            hist_ages = target_day - hist_days
            if not len(hist_items) or np.any(hist_ages <= 0) or np.any(hist_ages > 84):
                raise AssertionError("training history is not strictly earlier and within 84 days")
            positives = np.unique(item_ids[target_lo:target_hi]).astype(np.int32)
            target_group = len(target_groups)
            target_groups.append(positives)
            eligible_target_days += 1
            for positive in positives:
                row = ReservoirRow(hist_items.copy(), hist_ages.astype(np.int16), int(positive), target_group)
                eligible_examples += 1
                if len(reservoir) < maximum:
                    reservoir.append(row)
                else:
                    replacement = int(rng.integers(eligible_examples))
                    if replacement < maximum:
                        reservoir[replacement] = row
    if not reservoir:
        raise RuntimeError("no training examples survived")
    history = np.full((len(reservoir), maximum_history), -1, np.int32)
    ages = np.zeros((len(reservoir), maximum_history), np.int16)
    lengths = np.empty(len(reservoir), np.int16)
    positives = np.empty(len(reservoir), np.int32)
    groups = np.empty(len(reservoir), np.int32)
    for index, row in enumerate(reservoir):
        length = len(row.history)
        history[index, :length] = row.history
        ages[index, :length] = row.ages
        lengths[index] = length
        positives[index] = row.positive
        groups[index] = row.target_group
    group_offsets = np.r_[0, np.cumsum([len(x) for x in target_groups])].astype(np.int64)
    group_items = np.concatenate(target_groups).astype(np.int32, copy=False)
    np.savez_compressed(
        arrays_path,
        history=history,
        ages=ages,
        lengths=lengths,
        positives=positives,
        groups=groups,
        group_offsets=group_offsets,
        group_items=group_items,
    )
    latest_day = int(days.max())
    if latest_day >= cutoff_day:
        raise AssertionError("cutoff day leaked into training")
    result = {
        "schema": PREP_SCHEMA,
        "cutoff": cutoff,
        "params": params,
        "source": source,
        "full_scanned_distinct_user_day_item_rows": len(frame),
        "full_scanned_users": int(pd.Series(customers).nunique()),
        "eligible_target_days": eligible_target_days,
        "eligible_positive_examples_before_sampling": eligible_examples,
        "retained_training_examples": len(reservoir),
        "sampling_rate": len(reservoir) / eligible_examples,
        "history_length": {
            "mean": float(lengths.mean()),
            "p50": float(np.quantile(lengths, 0.5)),
            "p95": float(np.quantile(lengths, 0.95)),
            "max": int(lengths.max()),
        },
        "latest_training_interaction_date": str(
            np.datetime64("1970-01-01", "D") + np.timedelta64(latest_day, "D")
        ),
        "history_strictly_before_target_day": True,
        "same_day_target_items_excluded_from_negative_sampling": True,
        "same_day_order": "none; target items and same-day history ties use sorted item ids only for deterministic storage",
        "arrays": str(arrays_path.resolve()),
        "arrays_bytes": arrays_path.stat().st_size,
        "preparation_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }
    _write_json(marker, result)
    return result


def _training_batch(
    arrays: dict[str, np.ndarray], indices: np.ndarray, negative_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    exclusions = np.zeros((len(indices), len(negative_ids)), dtype=bool)
    for row, sample_index in enumerate(indices):
        group = int(arrays["groups"][sample_index])
        lo = int(arrays["group_offsets"][group])
        hi = int(arrays["group_offsets"][group + 1])
        exclusions[row] = np.isin(negative_ids, arrays["group_items"][lo:hi])
    return (
        arrays["history"][indices], arrays["ages"][indices], arrays["lengths"][indices],
        arrays["positives"][indices], negative_ids, exclusions,
    )


def _diversity_penalty(interests: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    if interests.shape[1] == 1:
        return interests.new_zeros(())
    similarities = torch.einsum("bkd,bjd->bkj", interests, interests)
    pair_mask = active.unsqueeze(2) & active.unsqueeze(1)
    pair_mask = pair_mask & ~torch.eye(interests.shape[1], dtype=torch.bool, device=interests.device).unsqueeze(0)
    if not pair_mask.any():
        return interests.new_zeros(())
    return similarities[pair_mask].square().mean()


def train_variant(
    cutoff: str, variant: str, contract: dict[str, Any], device: str = "cuda"
) -> dict[str, Any]:
    _guard(cutoff)
    if variant not in VARIANTS:
        raise KeyError(variant)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no silent CPU training fallback")
    prep = prepare_training(cutoff, contract)
    out = ARTIFACT_ROOT / cutoff / variant
    marker = out / "MODEL.json"
    checkpoint = out / "model.pt"
    shared = contract["shared_model"]
    params = {"schema": MODEL_SCHEMA, "variant": variant, **VARIANTS[variant], **shared}
    if marker.exists() and checkpoint.exists():
        saved = _read_json(marker)
        if saved.get("params") != params:
            raise AssertionError("stale MIND model cache has a different contract")
        return saved
    out.mkdir(parents=True, exist_ok=True)
    _, vectors, source = load_item_table(cutoff)
    arrays = dict(np.load(prep["arrays"], allow_pickle=False))
    item_vectors = torch.from_numpy(vectors).to(device)
    seed = int(shared["random_seed"])
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    torch.set_num_threads(4)
    model = MindEncoder(
        maximum_interests=int(VARIANTS[variant]["maximum_interests"]),
        time_mode=str(VARIANTS[variant]["time_mode"]),
    ).to(device)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(shared["learning_rate"]), weight_decay=float(shared["weight_decay"])
    )
    batch_size = int(shared["batch_size"])
    negative_count = int(shared["uniform_sampled_unobserved_items"])
    epochs = int(shared["epochs"])
    temperature = float(shared["temperature"])
    diversity_weight = float(shared["diversity_penalty"])
    routing_iterations = int(shared["routing_iterations"])
    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    sample_count = len(arrays["positives"])
    for epoch in range(epochs):
        order = np_rng.permutation(sample_count)
        total_loss = total_ce = total_diversity = 0.0
        correct = observations = 0
        interest_usage = np.zeros(int(VARIANTS[variant]["maximum_interests"]), np.int64)
        for begin in range(0, sample_count, batch_size):
            indices = order[begin : begin + batch_size]
            negative_ids = np_rng.integers(len(vectors), size=negative_count, dtype=np.int64)
            values = _training_batch(arrays, indices, negative_ids)
            history_ids = torch.as_tensor(values[0], device=device)
            ages = torch.as_tensor(values[1], device=device)
            lengths = torch.as_tensor(values[2], device=device)
            positives = torch.as_tensor(values[3], device=device)
            negatives = torch.as_tensor(values[4], device=device)
            exclusions = torch.as_tensor(values[5], device=device)
            interests, active = model(item_vectors, history_ids, ages, lengths, routing_iterations)
            positive_vectors = item_vectors[positives]
            positive_by_interest = torch.einsum("bkd,bd->bk", interests, positive_vectors)
            positive_by_interest = positive_by_interest.masked_fill(~active, -1e4)
            selected_interest = positive_by_interest.argmax(dim=1)
            selected = interests[torch.arange(len(indices), device=device), selected_interest]
            positive_score = (selected * positive_vectors).sum(dim=1, keepdim=True)
            negative_score = selected @ item_vectors[negatives].T
            negative_score = negative_score.masked_fill(exclusions, -1e4)
            logits = torch.cat((positive_score, negative_score), dim=1) / temperature
            cross_entropy = F.cross_entropy(logits, torch.zeros(len(indices), dtype=torch.long, device=device))
            diversity = _diversity_penalty(interests, active)
            loss = cross_entropy + diversity_weight * diversity
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite MIND loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_n = len(indices)
            total_loss += float(loss.detach()) * batch_n
            total_ce += float(cross_entropy.detach()) * batch_n
            total_diversity += float(diversity.detach()) * batch_n
            correct += int(logits.detach().argmax(dim=1).eq(0).sum())
            observations += batch_n
            interest_usage += np.bincount(
                selected_interest.detach().cpu().numpy(), minlength=len(interest_usage)
            )
            if time.perf_counter() - started > 90 * 60:
                raise RuntimeError("registered 90-minute per-variant training cap exceeded")
        row = {
            "epoch": epoch + 1,
            "sampled_cross_entropy": total_ce / observations,
            "objective_with_diversity": total_loss / observations,
            "diversity_penalty_unscaled": total_diversity / observations,
            "sampled_top1_accuracy": correct / observations,
            "observations": observations,
            "label_selected_interest_counts": interest_usage.tolist(),
            "learned_age_weights_by_interest_0_7__8_28__29_84": model.learned_age_weights(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print({"mind_train": cutoff, "variant": variant, **row}, flush=True)
    parameter_squared_change = sum(
        float((value.detach().cpu() - initial[name]).square().sum())
        for name, value in model.state_dict().items()
    )
    if not parameter_squared_change > 0:
        raise AssertionError("model parameters did not change")
    torch.save(model.state_dict(), checkpoint)
    peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    result = {
        "schema": MODEL_SCHEMA,
        "cutoff": cutoff,
        "variant": variant,
        "params": params,
        "preparation": str((ARTIFACT_ROOT / cutoff / "PREPARED.json").resolve()),
        "item_source": source,
        "training_examples": sample_count,
        "epochs": history,
        "trainable_parameters": sum(value.numel() for value in model.parameters()),
        "parameter_squared_change": parameter_squared_change,
        "learned_age_weights_by_interest_0_7__8_28__29_84": model.learned_age_weights(),
        "training_seconds": time.perf_counter() - started,
        "device": device,
        "torch_version": torch.__version__,
        "peak_cuda_allocated_bytes": peak,
        "all_registered_epochs_completed": True,
        "outer_labels_used": False,
        "final_week": "not_run",
    }
    _write_json(marker, result)
    del model, optimizer, item_vectors, arrays
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def _load_model(cutoff: str, variant: str, contract: dict[str, Any], device: str) -> tuple[MindEncoder, dict[str, Any]]:
    meta = train_variant(cutoff, variant, contract, device)
    model = MindEncoder(
        int(VARIANTS[variant]["maximum_interests"]), str(VARIANTS[variant]["time_mode"])
    ).to(device)
    state = torch.load(
        ARTIFACT_ROOT / cutoff / variant / "model.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    return model, meta


def prepare_query_histories(cutoff: str, items: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Create cutoff-time histories for exactly the frozen evaluation users."""
    candidates = candidate_path(cutoff)
    with _connection() as con:
        con.register("vocabulary", items[["article_id", "row_index"]])
        con.execute(f"CREATE TEMP VIEW base AS SELECT * FROM read_parquet('{_sql_path(candidates)}')")
        frame = con.execute(
            f"""SELECT x.customer_id,
                date_diff('day',DATE '1970-01-01',x.t_dat)::INTEGER AS day_index,
                v.row_index::INTEGER item
            FROM (
                SELECT DISTINCT t.customer_id,t.t_dat,t.article_id
                FROM read_parquet('{_sql_path(TX)}') t
                SEMI JOIN (SELECT DISTINCT customer_id FROM base) u USING(customer_id)
                WHERE t.t_dat>=DATE '{cutoff}'-INTERVAL 84 DAY AND t.t_dat<DATE '{cutoff}'
            ) x JOIN vocabulary v USING(article_id)
            ORDER BY x.customer_id,day_index,item"""
        ).fetchdf()
        frame = frame.rename(columns={"day_index": "day"})
        users = con.execute("SELECT DISTINCT customer_id FROM base ORDER BY customer_id").fetchdf()
    user_index = pd.Index(users["customer_id"])
    maximum_history = 50
    history = np.full((len(users), maximum_history), -1, np.int32)
    ages = np.zeros((len(users), maximum_history), np.int16)
    lengths = np.zeros(len(users), np.int16)
    cutoff_day = int(np.datetime64(cutoff, "D").astype(np.int32))
    if len(frame):
        customers = frame["customer_id"].to_numpy()
        frame_items = frame["item"].to_numpy(np.int32)
        frame_days = frame["day"].to_numpy(np.int32)
        for lo, hi in _iter_user_ranges(customers):
            user_row = int(user_index.get_loc(customers[lo]))
            selected_items = frame_items[lo:hi][-maximum_history:]
            selected_days = frame_days[lo:hi][-maximum_history:]
            selected_ages = cutoff_day - selected_days
            if np.any(selected_ages <= 0) or np.any(selected_ages > 84):
                raise AssertionError("query history outside cutoff-safe 84-day window")
            length = len(selected_items)
            history[user_row, :length] = selected_items
            ages[user_row, :length] = selected_ages.astype(np.int16)
            lengths[user_row] = length
    evidence = {
        "evaluation_users": len(users),
        "users_with_vocabulary_history": int((lengths > 0).sum()),
        "mean_history_length_all_evaluation_users": float(lengths.mean()),
        "maximum_history": int(lengths.max()),
        "history_strictly_before_cutoff": True,
    }
    return users, history, ages, lengths, evidence


def _save_candidates(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connection("4GB") as con:
        con.register("frame", frame)
        con.execute(
            f"COPY frame TO '{_sql_path(path)}' (FORMAT PARQUET,COMPRESSION ZSTD)"
        )


def retrieve_variant(cutoff: str, variant: str, contract: dict[str, Any], device: str = "cuda") -> tuple[Path, dict[str, Any]]:
    _guard(cutoff)
    out = ARTIFACT_ROOT / cutoff / variant
    candidate_output = out / "candidates.parquet"
    marker = out / "RETRIEVAL.json"
    if marker.exists() and candidate_output.exists():
        return candidate_output, _read_json(marker)
    model, model_meta = _load_model(cutoff, variant, contract, device)
    items, vectors, source = load_item_table(cutoff)
    users, history, ages, lengths, history_meta = prepare_query_histories(cutoff, items)
    available = np.flatnonzero(lengths > 0)
    item_vectors = torch.from_numpy(vectors).to(device)
    per_interest = int(contract["retrieval"]["per_interest_top_n"])
    channel_cap = int(contract["retrieval"]["per_user_channel_cap"])
    routing_iterations = int(contract["shared_model"]["routing_iterations"])
    started = time.perf_counter()
    rows: list[tuple[str, str, int, float, int, int]] = []
    article_ids = items["article_id"].to_numpy()
    interest_counts = np.zeros(model.maximum_interests + 1, np.int64)
    with torch.no_grad():
        for begin in range(0, len(available), 128):
            user_rows = available[begin : begin + 128]
            h = torch.as_tensor(history[user_rows], device=device)
            a = torch.as_tensor(ages[user_rows], device=device)
            n = torch.as_tensor(lengths[user_rows], device=device)
            interests, active = model(item_vectors, h, a, n, routing_iterations)
            counts = active.sum(dim=1).cpu().numpy().astype(np.int64)
            for value in counts:
                interest_counts[value] += 1
            per_user: list[dict[int, tuple[float, int]]] = [dict() for _ in user_rows]
            for interest_index in range(model.maximum_interests):
                active_rows = np.flatnonzero(counts > interest_index)
                if not len(active_rows):
                    continue
                scores = interests[active_rows, interest_index] @ item_vectors.T
                values, indices = torch.topk(scores, k=min(per_interest, len(vectors)), dim=1)
                values_np = values.cpu().numpy()
                indices_np = indices.cpu().numpy()
                for local_active, local_user in enumerate(active_rows):
                    target = per_user[int(local_user)]
                    for item_index, score in zip(indices_np[local_active], values_np[local_active]):
                        previous = target.get(int(item_index))
                        evidence = (float(score), interest_index + 1)
                        if previous is None or evidence[0] > previous[0]:
                            target[int(item_index)] = evidence
                del scores, values, indices
            for local_user, evidence in enumerate(per_user):
                ranked = sorted(
                    evidence.items(), key=lambda pair: (-pair[1][0], str(article_ids[pair[0]]))
                )[:channel_cap]
                customer_id = str(users.iloc[int(user_rows[local_user])]["customer_id"])
                for rank, (item_index, (score, interest_index)) in enumerate(ranked, start=1):
                    rows.append(
                        (customer_id, str(article_ids[item_index]), rank, score,
                         interest_index, int(counts[local_user]))
                    )
    frame = pd.DataFrame(
        rows,
        columns=["customer_id", "article_id", "mind_rank", "mind_score", "best_interest", "interest_count"],
    )
    if frame.duplicated(["customer_id", "article_id"]).any():
        raise AssertionError("duplicate MIND candidate pair")
    group_sizes = frame.groupby("customer_id").size() if len(frame) else pd.Series(dtype=int)
    if len(group_sizes) and int(group_sizes.max()) > channel_cap:
        raise AssertionError("MIND channel cap exceeded")
    _save_candidates(frame, candidate_output)
    result = {
        "cutoff": cutoff,
        "variant": variant,
        "model": str((out / "MODEL.json").resolve()),
        "item_source": source,
        "query_history": history_meta,
        "candidate_rows": len(frame),
        "candidate_users": int(frame["customer_id"].nunique()) if len(frame) else 0,
        "mean_candidates_per_candidate_user": float(group_sizes.mean()) if len(group_sizes) else 0.0,
        "maximum_candidates_per_user": int(group_sizes.max()) if len(group_sizes) else 0,
        "users_by_dynamic_interest_count": {
            str(index): int(value) for index, value in enumerate(interest_counts) if index > 0
        },
        "candidate_identity_unique": not frame.duplicated(["customer_id", "article_id"]).any(),
        "retrieval_seconds": time.perf_counter() - started,
        "model_training_seconds": model_meta["training_seconds"],
        "peak_cuda_allocated_bytes_training": model_meta["peak_cuda_allocated_bytes"],
        "final_week": "not_run",
    }
    _write_json(marker, result)
    del model, item_vectors, frame
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return candidate_output, result


def _oracle_map(con: duckdb.DuckDBPyConnection, table: str) -> float:
    return float(
        con.execute(
            f"""WITH totals AS (
                SELECT customer_id,count(*)::DOUBLE truth_n FROM truth_warm GROUP BY customer_id
            ), hits AS (
                SELECT t.customer_id,count(*)::DOUBLE hit_n
                FROM truth_warm t JOIN {table} c USING(customer_id,article_id) GROUP BY t.customer_id
            )
            SELECT avg(least(coalesce(h.hit_n,0),12)/least(t.truth_n,12))
            FROM totals t LEFT JOIN hits h USING(customer_id)"""
        ).fetchone()[0]
    )


def evaluate_variant(cutoff: str, variant: str, generated: Path) -> dict[str, Any]:
    _guard(cutoff)
    base = candidate_path(cutoff)
    with _connection() as con:
        con.execute(f"CREATE TEMP TABLE base AS SELECT customer_id,article_id FROM read_parquet('{_sql_path(base)}')")
        con.execute(f"CREATE TEMP TABLE mind AS SELECT * FROM read_parquet('{_sql_path(generated)}')")
        con.execute(
            f"""CREATE TEMP TABLE articles AS SELECT article_id,
            coalesce(garment_group_name,'__MISSING__') garment_group_name
            FROM read_csv_auto('{_sql_path(ARTICLES)}',header=true,all_varchar=true)"""
        )
        con.execute("CREATE TEMP TABLE eval_users AS SELECT DISTINCT customer_id FROM base")
        con.execute(
            f"""CREATE TEMP TABLE warm_catalog AS SELECT DISTINCT article_id
            FROM read_parquet('{_sql_path(TX)}') WHERE t_dat<DATE '{cutoff}'"""
        )
        con.execute(
            f"""CREATE TEMP TABLE truth_warm AS SELECT DISTINCT t.customer_id,t.article_id
            FROM read_parquet('{_sql_path(TX)}') t JOIN eval_users u USING(customer_id)
            JOIN warm_catalog w USING(article_id)
            WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY"""
        )
        con.execute(
            f"""CREATE TEMP TABLE hist84 AS SELECT DISTINCT t.customer_id,t.t_dat,t.article_id,a.garment_group_name
            FROM read_parquet('{_sql_path(TX)}') t JOIN eval_users u USING(customer_id)
            JOIN articles a USING(article_id)
            WHERE t_dat>=DATE '{cutoff}'-INTERVAL 84 DAY AND t_dat<DATE '{cutoff}'"""
        )
        con.execute(
            """CREATE TEMP TABLE group_counts AS SELECT customer_id,garment_group_name,count(*) n
            FROM hist84 GROUP BY customer_id,garment_group_name"""
        )
        con.execute(
            """CREATE TEMP TABLE dominant_group AS SELECT customer_id,garment_group_name
            FROM group_counts QUALIFY row_number() OVER(PARTITION BY customer_id ORDER BY n DESC,garment_group_name)=1"""
        )
        con.execute(
            """CREATE TEMP TABLE top_groups AS SELECT customer_id,garment_group_name
            FROM group_counts WHERE n>=2 QUALIFY row_number() OVER(
            PARTITION BY customer_id ORDER BY n DESC,garment_group_name)<=3"""
        )
        con.execute(
            f"""CREATE TEMP TABLE recent_group_pop AS SELECT garment_group_name,article_id
            FROM (SELECT a.garment_group_name,t.article_id,count(*) n
                FROM read_parquet('{_sql_path(TX)}') t JOIN articles a USING(article_id)
                WHERE t_dat>=DATE '{cutoff}'-INTERVAL 28 DAY AND t_dat<DATE '{cutoff}'
                GROUP BY a.garment_group_name,t.article_id)
            QUALIFY row_number() OVER(PARTITION BY garment_group_name ORDER BY n DESC,article_id)<=20"""
        )
        con.execute(
            """CREATE TEMP TABLE multi_profile_proxy AS SELECT DISTINCT g.customer_id,p.article_id
            FROM top_groups g JOIN recent_group_pop p USING(garment_group_name)"""
        )
        con.execute(
            """CREATE TEMP TABLE augmented AS SELECT customer_id,article_id FROM base
            UNION SELECT customer_id,article_id FROM mind"""
        )
        truth_pairs, truth_users = con.execute(
            "SELECT count(*),count(DISTINCT customer_id) FROM truth_warm"
        ).fetchone()
        base_hits, base_hit_users = con.execute(
            """SELECT count(*),count(DISTINCT t.customer_id)
            FROM truth_warm t JOIN base b USING(customer_id,article_id)"""
        ).fetchone()
        standalone_hits, standalone_hit_users = con.execute(
            """SELECT count(*),count(DISTINCT t.customer_id)
            FROM truth_warm t JOIN mind m USING(customer_id,article_id)"""
        ).fetchone()
        marginal_hits = int(
            con.execute(
                """SELECT count(*) FROM truth_warm t JOIN mind m USING(customer_id,article_id)
                LEFT JOIN base b USING(customer_id,article_id) WHERE b.article_id IS NULL"""
            ).fetchone()[0]
        )
        unique_beyond_proxy = int(
            con.execute(
                """SELECT count(*) FROM truth_warm t JOIN mind m USING(customer_id,article_id)
                LEFT JOIN base b USING(customer_id,article_id)
                LEFT JOIN multi_profile_proxy p USING(customer_id,article_id)
                WHERE b.article_id IS NULL AND p.article_id IS NULL"""
            ).fetchone()[0]
        )
        non_dominant = int(
            con.execute(
                """SELECT count(*) FROM truth_warm t JOIN mind m USING(customer_id,article_id)
                JOIN articles a USING(article_id)
                JOIN group_counts g ON g.customer_id=t.customer_id AND g.garment_group_name=a.garment_group_name
                LEFT JOIN dominant_group d USING(customer_id)
                LEFT JOIN base b USING(customer_id,article_id)
                WHERE b.article_id IS NULL AND a.garment_group_name<>d.garment_group_name"""
            ).fetchone()[0]
        )
        candidate_rows, candidate_users, overlap_rows = con.execute(
            """SELECT count(*),count(DISTINCT m.customer_id),
            count(*) FILTER(WHERE b.article_id IS NOT NULL)
            FROM mind m LEFT JOIN base b USING(customer_id,article_id)"""
        ).fetchone()
        augmented_hits, augmented_hit_users = con.execute(
            """SELECT count(*),count(DISTINCT t.customer_id)
            FROM truth_warm t JOIN augmented a USING(customer_id,article_id)"""
        ).fetchone()
        return {
            "cutoff": cutoff,
            "variant": variant,
            "denominators": {"warm_truth_pairs": int(truth_pairs), "warm_truth_users": int(truth_users)},
            "current_union": {
                "truth_pairs_retrieved": int(base_hits),
                "candidate_recall": base_hits / truth_pairs if truth_pairs else 0.0,
                "truth_users_hit": int(base_hit_users),
                "hit_rate": base_hit_users / truth_users if truth_users else 0.0,
                "oracle_map12": _oracle_map(con, "base"),
            },
            "standalone_mind": {
                "candidate_rows": int(candidate_rows),
                "candidate_users": int(candidate_users),
                "truth_pairs_retrieved": int(standalone_hits),
                "candidate_recall": standalone_hits / truth_pairs if truth_pairs else 0.0,
                "truth_users_hit": int(standalone_hit_users),
                "hit_rate": standalone_hit_users / truth_users if truth_users else 0.0,
            },
            "increment_over_current_union": {
                "marginal_truth_pairs": marginal_hits,
                "marginal_recall": marginal_hits / truth_pairs if truth_pairs else 0.0,
                "truth_pairs_unique_beyond_current_union_and_multi_profile_proxy": unique_beyond_proxy,
                "unique_recall_beyond_current_union_and_multi_profile_proxy": unique_beyond_proxy / truth_pairs if truth_pairs else 0.0,
                "non_dominant_historical_garment_group_truth_pairs": non_dominant,
                "non_dominant_share_of_marginal_truth": non_dominant / marginal_hits if marginal_hits else 0.0,
            },
            "candidate_overlap": {
                "rows_also_in_current_union": int(overlap_rows),
                "row_overlap_rate": overlap_rows / candidate_rows if candidate_rows else 0.0,
            },
            "augmented_union": {
                "truth_pairs_retrieved": int(augmented_hits),
                "candidate_recall": augmented_hits / truth_pairs if truth_pairs else 0.0,
                "truth_users_hit": int(augmented_hit_users),
                "hit_rate": augmented_hit_users / truth_users if truth_users else 0.0,
                "oracle_map12": _oracle_map(con, "augmented"),
                "oracle_map12_delta": _oracle_map(con, "augmented") - _oracle_map(con, "base"),
            },
        }


def run_window(name: str, cutoff: str, contract: dict[str, Any], device: str) -> dict[str, Any]:
    if not candidate_path(cutoff).exists():
        raise FileNotFoundError(candidate_path(cutoff))
    started = time.perf_counter()
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        print({"mind_window": name, "cutoff": cutoff, "variant": variant}, flush=True)
        generated, retrieval = retrieve_variant(cutoff, variant, contract, device)
        evaluation = evaluate_variant(cutoff, variant, generated)
        variants[variant] = {"retrieval": retrieval, "evaluation": evaluation}
        print(
            {
                "mind_eval": name,
                "variant": variant,
                "marginal_recall": evaluation["increment_over_current_union"]["marginal_recall"],
                "unique_recall": evaluation["increment_over_current_union"]["unique_recall_beyond_current_union_and_multi_profile_proxy"],
                "overlap": evaluation["candidate_overlap"]["row_overlap_rate"],
            },
            flush=True,
        )
    return {
        "name": name,
        "cutoff": cutoff,
        "variants": variants,
        "runtime_seconds": time.perf_counter() - started,
        "final_week": "not_run",
    }


def mechanics_pass(window: dict[str, Any], contract: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, bool] = {}
    for variant, row in window["variants"].items():
        retrieval = row["retrieval"]
        model = _read_json(Path(retrieval["model"]))
        checks[f"{variant}_epochs_complete"] = bool(model["all_registered_epochs_completed"])
        checks[f"{variant}_parameter_change"] = bool(model["parameter_squared_change"] > 0)
        checks[f"{variant}_candidate_unique"] = bool(retrieval["candidate_identity_unique"])
        checks[f"{variant}_candidate_cap"] = bool(retrieval["maximum_candidates_per_user"] <= 100)
        checks[f"{variant}_finite_metrics"] = all(
            math.isfinite(float(value))
            for value in (
                row["evaluation"]["standalone_mind"]["candidate_recall"],
                row["evaluation"]["increment_over_current_union"]["marginal_recall"],
                row["evaluation"]["candidate_overlap"]["row_overlap_rate"],
            )
        )
    checks["final_week_not_run"] = contract["development_protocol"]["final_week"]["status"] == "not_run"
    return all(checks.values()), checks


def promotion_gate(windows: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    primary = contract["retrieval_promotion_gate"]["primary_variant"]
    rows = [window["variants"][primary] for window in windows.values()]
    marginal = [row["evaluation"]["increment_over_current_union"]["marginal_recall"] for row in rows]
    unique = [row["evaluation"]["increment_over_current_union"]["unique_recall_beyond_current_union_and_multi_profile_proxy"] for row in rows]
    non_dominant = [row["evaluation"]["increment_over_current_union"]["non_dominant_share_of_marginal_truth"] for row in rows]
    overlap = [row["evaluation"]["candidate_overlap"]["row_overlap_rate"] for row in rows]
    training_minutes = [row["retrieval"]["model_training_seconds"] / 60 for row in rows]
    peak_gib = [row["retrieval"]["peak_cuda_allocated_bytes_training"] / 2**30 for row in rows]
    rules = contract["retrieval_promotion_gate"]
    checks = {
        "mean_marginal_recall": float(np.mean(marginal)) >= rules["mean_marginal_warm_recall_vs_current_union_min"],
        "positive_windows": sum(value > 0 for value in marginal) >= rules["positive_windows_min"],
        "mean_unique_recall": float(np.mean(unique)) >= rules["mean_marginal_warm_recall_unique_beyond_current_union_and_multi_profile_proxy_min"],
        "mean_non_dominant_share": float(np.mean(non_dominant)) >= rules["new_truth_from_non_dominant_historical_garment_group_share_min"],
        "each_window_has_non_dominant_truth": all(value > 0 for value in non_dominant),
        "maximum_candidate_overlap": max(overlap) <= rules["candidate_row_overlap_with_current_union_max"],
        "maximum_training_minutes": max(training_minutes) <= rules["per_window_training_minutes_max"],
        "maximum_peak_vram_gib": max(peak_gib) <= rules["peak_vram_gib_max"],
    }
    return {
        "primary_variant": primary,
        "observed": {
            "mean_marginal_recall": float(np.mean(marginal)),
            "positive_windows": sum(value > 0 for value in marginal),
            "mean_unique_recall_beyond_current_union_and_multi_profile_proxy": float(np.mean(unique)),
            "mean_non_dominant_share_of_marginal_truth": float(np.mean(non_dominant)),
            "maximum_candidate_row_overlap": max(overlap),
            "maximum_training_minutes": max(training_minutes),
            "maximum_peak_vram_gib": max(peak_gib),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "failure_action": "stop before LightGBM ranking rebuild and retain WV3-741",
    }


def _mean_variant(windows: dict[str, Any], variant: str, path: tuple[str, ...]) -> float:
    values = []
    for window in windows.values():
        value: Any = window["variants"][variant]
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def render_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# MIND Warm MVE：学习型行为年龄权重与多兴趣召回",
        "",
        "## 结论",
        "",
    ]
    if metrics.get("outer_status") == "completed":
        gate = metrics["retrieval_promotion_gate"]
        lines += [
            f"检索晋级门槛：**{'通过' if gate['passed'] else '未通过'}**。",
            f"主模型四窗平均边际 Warm Recall 为 `{gate['observed']['mean_marginal_recall']:.6f}`；"
            f"超出当前并集和简单多画像代理的平均独有 Recall 为 "
            f"`{gate['observed']['mean_unique_recall_beyond_current_union_and_multi_profile_proxy']:.6f}`。",
            "",
            "本阶段只在检索层决定是否值得重建排序训练集。"
            + ("检索门槛通过，排序仍未运行。" if gate["passed"] else "检索门槛未通过，因此 LightGBM 排序没有运行。"),
        ]
    else:
        lines += ["目前只完成历史机制链；四个2020开发窗口尚未运行。"]
    lines += [
        "",
        "最终周 `2020-09-16` 保持 `not_run`。",
        "",
        "## 术语与统计口径",
        "",
        "- MVE（Minimum Viable Experiment，最小可行实验）：用固定规模训练样本和10%开发用户判断机制是否值得继续；不是全量生产模型。",
        "- MIND-style（受MIND启发的多兴趣编码）：把一名用户的历史商品分配到最多三个兴趣向量；本项目只复现动态路由思想，不宣称逐项复现论文。",
        "- 行为年龄权重（本项目模型参数）：历史交互距目标日或截止日的天数落入0–7、8–28、29–84天后，每个兴趣分别学习三个相对权重；数值按每个兴趣均值为1归一化。",
        "- 边际 Warm Recall（本项目评测）：MIND命中且当前候选并集未命中的不同Warm用户—商品真值对数，除以该窗全部不同Warm用户—商品真值对数。",
        "- 独有 Recall（本项目评测）：同时未被当前候选并集和简单多画像代理命中、但被MIND命中的Warm真值对数，除以该窗全部Warm真值对数。",
        "- 候选行重叠率（本项目评测）：MIND用户—商品候选行中也存在于当前候选并集的行数，除以MIND全部候选行数。",
        "- Oracle MAP@12（不可部署诊断上界）：假设预知验证标签并把候选池内真值排在最前得到的MAP@12；不代表排序器实际可实现。",
        "- 未观察商品（隐式反馈常见口径）：抽样商品没有购买记录，但数据没有曝光日志，不能解释为用户明确拒绝。",
        "",
        "## 固定实验设计",
        "",
        "```text",
        "截止日前完整交易总体",
        "  -> 每个目标购买日只读取更早84天、最多50个用户日商品",
        "  -> 冻结Item2Vec商品向量",
        "  -> 单兴趣 / 学习年龄多兴趣 / 无时间多兴趣 / 固定28天多兴趣",
        "  -> 每个兴趣Top50，合并去重后每用户最多100件",
        "  -> 与当前候选并集及简单多画像代理比较",
        "```",
        "",
        "训练流完整扫描所有符合条件的用户日正例，再用固定随机种子保留至多250,000个正例；"
        "同一购买日每个不同商品都作为正例，历史严格来自更早日期，不制造日内顺序。商品向量冻结，"
        "因此本轮学习的是用户侧表示与行为年龄，不是重训整张商品表。",
        "",
        "## 历史机制链",
        "",
    ]
    mechanics = metrics.get("mechanics")
    if mechanics:
        lines += [
            f"机制门槛：**{'通过' if metrics['mechanics_gate']['passed'] else '未通过'}**；"
            f"窗口截止日 `{mechanics['cutoff']}`，总耗时 `{mechanics['runtime_seconds']:.1f}` 秒。",
            "",
            "| 方案 | 新增Recall | 独有Recall | 候选重叠率 | 训练秒数 | 峰值显存GiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for variant, row in mechanics["variants"].items():
            evaluation = row["evaluation"]
            retrieval = row["retrieval"]
            lines.append(
                f"| {variant} | {evaluation['increment_over_current_union']['marginal_recall']:.6f} | "
                f"{evaluation['increment_over_current_union']['unique_recall_beyond_current_union_and_multi_profile_proxy']:.6f} | "
                f"{evaluation['candidate_overlap']['row_overlap_rate']:.4f} | "
                f"{retrieval['model_training_seconds']:.1f} | "
                f"{retrieval['peak_cuda_allocated_bytes_training']/2**30:.3f} |"
            )
    if metrics.get("outer_status") == "completed":
        lines += [
            "",
            "## 四窗固定回放",
            "",
            "| 窗口 | 方案 | 新增Warm真值对 | 边际Recall | 独有Recall | 非主导历史组占新增真值 | 候选重叠率 | Oracle增量 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, window in metrics["outer_windows"].items():
            for variant, row in window["variants"].items():
                evaluation = row["evaluation"]
                increment = evaluation["increment_over_current_union"]
                lines.append(
                    f"| {name} | {variant} | {increment['marginal_truth_pairs']} | "
                    f"{increment['marginal_recall']:.6f} | "
                    f"{increment['unique_recall_beyond_current_union_and_multi_profile_proxy']:.6f} | "
                    f"{increment['non_dominant_share_of_marginal_truth']:.2%} | "
                    f"{evaluation['candidate_overlap']['row_overlap_rate']:.4f} | "
                    f"{evaluation['augmented_union']['oracle_map12_delta']:+.6f} |"
                )
        lines += ["", "四窗方案均值：", "", "| 方案 | 边际Recall | 独有Recall | 候选重叠率 |", "|---|---:|---:|---:|"]
        for variant in VARIANTS:
            lines.append(
                f"| {variant} | {_mean_variant(metrics['outer_windows'], variant, ('evaluation','increment_over_current_union','marginal_recall')):.6f} | "
                f"{_mean_variant(metrics['outer_windows'], variant, ('evaluation','increment_over_current_union','unique_recall_beyond_current_union_and_multi_profile_proxy')):.6f} | "
                f"{_mean_variant(metrics['outer_windows'], variant, ('evaluation','candidate_overlap','row_overlap_rate')):.4f} |"
            )
        lines += [
            "",
            "## 决策",
            "",
            f"主模型 `mind_learned_age` 的门槛明细：`{json.dumps(metrics['retrieval_promotion_gate']['checks'], ensure_ascii=False)}`。",
            "",
            ("检索门槛通过；下一阶段才允许在完全相同扩展候选池上重训排序器。"
             if metrics["retrieval_promotion_gate"]["passed"] else
             "检索门槛未通过；停止MIND路线，不调整年龄边界、兴趣数、候选数或负采样比例救援，保留WV3-741。"),
        ]
    ranking = metrics.get("ranking_followup")
    if ranking:
        inner = ranking.get("inner_gate")
        outer = ranking.get("outer_gate")
        lines += ["", "## 排序兑现状态", ""]
        if outer:
            lines += [
                f"保守尾部准入的外层门槛：**{'通过' if outer['passed'] else '未通过'}**；"
                f"相对 WV3-741 的四窗平均 MAP@12 增量为 `{outer['mean_delta_vs_frozen_baseline']:+.9f}`。"
            ]
        elif inner:
            lines += [
                f"保守尾部准入的内层门槛：**{'通过' if inner['passed'] else '未通过'}**；"
                f"四窗平均 MAP@12 增量为 `{inner['mean_delta_vs_frozen_baseline']:+.9f}`。",
                "内层未通过时外层排序标签保持未读取。",
            ]
        lines += [
            "保守尾部准入（本项目自定义）：冻结原排序第1—7名，每名用户最多让一件MIND独有候选与原第8—12名中的一件商品交换。",
            f"完整排序报告：`{Path(ranking['report']).name}`。",
        ]
    lines += [
        "",
        "## 边界",
        "",
        "固定Item2Vec商品向量由同截止日前12周数据训练；本实验没有为每个历史训练目标重训逐日商品表。"
        "这不读取验证周标签，但意味着本轮是截止日部署式训练，不是每条历史样本都具有独立冻结词表的严格逐日回放。",
        "候选检索使用截止日前已出现且进入Item2Vec词表的商品；H&M没有曝光、库存和精确上架时间。",
        "最终周未读取、未评分，也没有生成提交。",
        "",
        f"机器可读结果：`{METRICS_PATH.name}`。",
        "",
    ]
    return "\n".join(lines)


def run(phase: str, device: str = "cuda") -> dict[str, Any]:
    contract = _read_json(CONTRACT_PATH)
    if contract["development_protocol"]["final_week"]["status"] != "not_run":
        raise AssertionError("final week must remain not_run")
    metrics = _read_json(METRICS_PATH) if METRICS_PATH.exists() else {
        "schema": "mind-warm-mve-results-v1",
        "experiment_id": contract["experiment_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": str(CONTRACT_PATH.resolve()),
        "outer_status": "not_run",
        "ranking_status": "not_run",
        "final_week": "not_run",
    }
    if phase in ("mechanics", "all") and "mechanics" not in metrics:
        name, cutoff = next(iter(MECHANICS.items()))
        mechanics = run_window(name, cutoff, contract, device)
        passed, checks = mechanics_pass(mechanics, contract)
        metrics["mechanics"] = mechanics
        metrics["mechanics_gate"] = {"passed": passed, "checks": checks}
        _write_json(METRICS_PATH, metrics)
        REPORT_PATH.write_text(render_report(metrics), encoding="utf-8", newline="\n")
        if not passed:
            return metrics
    if phase in ("outer", "all"):
        if not metrics.get("mechanics_gate", {}).get("passed"):
            raise RuntimeError("mechanics gate has not passed")
        outer_windows = metrics.setdefault("outer_windows", {})
        for name, cutoff in OUTER.items():
            if name not in outer_windows:
                outer_windows[name] = run_window(name, cutoff, contract, device)
                _write_json(METRICS_PATH, metrics)
        metrics["outer_status"] = "completed"
        metrics["retrieval_promotion_gate"] = promotion_gate(outer_windows, contract)
        metrics["ranking_status"] = (
            "eligible_not_run" if metrics["retrieval_promotion_gate"]["passed"] else "not_run_retrieval_gate_failed"
        )
        metrics["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(METRICS_PATH, metrics)
        REPORT_PATH.write_text(render_report(metrics), encoding="utf-8", newline="\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--phase", choices=["mechanics", "outer", "all"], default="all")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    result = run(args.phase, args.device)
    print(
        {
            "experiment": result["experiment_id"],
            "mechanics_pass": result.get("mechanics_gate", {}).get("passed"),
            "outer_status": result.get("outer_status"),
            "retrieval_gate": result.get("retrieval_promotion_gate", {}).get("passed"),
            "ranking_status": result.get("ranking_status"),
            "report": str(REPORT_PATH),
            "final_week": result.get("final_week"),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
