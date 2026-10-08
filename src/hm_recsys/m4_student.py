from __future__ import annotations

import copy
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .m33 import ROLLING_PROTOCOL
from .m4_contract import M4_RUN_ID, STATIC_FIELDS, atomic_json, file_identity, stable_u64
from .m3 import _peak_working_set_bytes


VARIANTS = ("image_only", "metadata_only", "multimodal")
LAMBDA_GRID = (0.25, 0.5)
TRAINING = {
    "batch_size": 512,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "temperature": 0.10,
    "pairwise_margin": 0.10,
    "max_epochs": 3,
    "early_stopping_patience": 1,
    "validation_anchor_fraction": 0.10,
    "seed": 20260903,
}


class StaticItemEncoder(nn.Module):
    def __init__(self, *, variant: str, cardinalities: list[int]) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.image_branch = nn.Sequential(nn.Linear(512, 256), nn.GELU())
        dimensions = [max(8, min(32, int(round(math.sqrt(value + 1))))) for value in cardinalities]
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, dimension, padding_idx=0) for cardinality, dimension in zip(cardinalities, dimensions)]
        )
        metadata_width = sum(dimensions) + len(cardinalities)
        self.metadata_branch = nn.Sequential(nn.Linear(metadata_width, 128), nn.GELU())
        if variant == "multimodal":
            self.fusion = nn.Sequential(nn.Linear(256 + 128 + 1, 256), nn.GELU(), nn.Linear(256, 128))
        elif variant == "image_only":
            self.fusion = nn.Linear(256 + 1, 128)
        else:
            self.fusion = nn.Linear(128, 128)

    def forward(
        self, images: torch.Tensor, categories: torch.Tensor, image_missing: torch.Tensor
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        missing = (categories == 0).to(images.dtype)
        if self.variant in {"image_only", "multimodal"}:
            image_hidden = self.image_branch(images)
        if self.variant in {"metadata_only", "multimodal"}:
            embedded = [layer(categories[:, index]) for index, layer in enumerate(self.category_embeddings)]
            metadata_hidden = self.metadata_branch(torch.cat([*embedded, missing], dim=1))
        if self.variant == "image_only":
            parts = [image_hidden, image_missing[:, None]]
        elif self.variant == "metadata_only":
            parts = [metadata_hidden]
        else:
            parts = [image_hidden, metadata_hidden, image_missing[:, None]]
        return F.normalize(self.fusion(torch.cat(parts, dim=1)), dim=1)


def build_static_catalog(
    *, articles_path: Path, fashionclip_root: Path, artifact_dir: Path
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    articles = pd.read_csv(articles_path, dtype=str)
    image_items = pd.read_csv(fashionclip_root / "items.csv", dtype={"article_id": str}).sort_values("row_index")
    image_lookup = dict(zip(image_items["article_id"], image_items["row_index"].astype(int)))
    image_rows = np.asarray([image_lookup.get(item, -1) for item in articles["article_id"]], dtype=np.int32)
    category_maps: dict[str, dict[str, int]] = {}
    categories = np.zeros((len(articles), len(STATIC_FIELDS)), dtype=np.int32)
    for column_index, field in enumerate(STATIC_FIELDS):
        values = articles[field].fillna("__MISSING__").astype(str)
        vocabulary = sorted(value for value in values.unique().tolist() if value != "__MISSING__")
        mapping = {value: index + 1 for index, value in enumerate(vocabulary)}
        category_maps[field] = mapping
        categories[:, column_index] = values.map(mapping).fillna(0).to_numpy(dtype=np.int32)
    np.save(artifact_dir / "catalog_image_rows.int32.npy", image_rows, allow_pickle=False)
    np.save(artifact_dir / "catalog_categories.int32.npy", categories, allow_pickle=False)
    item_path = artifact_dir / "catalog_items.csv"
    articles[["article_id"]].assign(catalog_row=np.arange(len(articles), dtype=np.int32)).to_csv(
        item_path, index=False, lineterminator="\n"
    )
    atomic_json(artifact_dir / "category_maps.json", category_maps)
    return {
        "rows": len(articles),
        "image_covered": int(np.count_nonzero(image_rows >= 0)),
        "image_missing": int(np.count_nonzero(image_rows < 0)),
        "fields": list(STATIC_FIELDS),
        "cardinalities": [len(category_maps[field]) + 1 for field in STATIC_FIELDS],
        "artifacts": {
            name: file_identity(artifact_dir / filename)
            for name, filename in {
                "image_rows": "catalog_image_rows.int32.npy",
                "categories": "catalog_categories.int32.npy",
                "items": "catalog_items.csv",
                "category_maps": "category_maps.json",
            }.items()
        },
    }


class CatalogAccessor:
    def __init__(self, *, fashionclip_root: Path, catalog_dir: Path, device: torch.device) -> None:
        self.images = np.load(fashionclip_root / "embeddings.float16.npy", mmap_mode="r")
        self.image_rows = np.load(catalog_dir / "catalog_image_rows.int32.npy", mmap_mode="r")
        self.categories_np = np.load(catalog_dir / "catalog_categories.int32.npy", mmap_mode="r")
        self.categories = torch.as_tensor(
            np.array(self.categories_np, dtype=np.int32, copy=True), dtype=torch.long, device=device
        )
        self.device = device

    def batch(self, indices: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(indices, torch.Tensor):
            rows = indices.detach().cpu().numpy().astype(np.int64, copy=False)
            index_tensor = indices.to(self.device, dtype=torch.long)
        else:
            rows = np.asarray(indices, dtype=np.int64)
            index_tensor = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        image_rows = np.asarray(self.image_rows[rows], dtype=np.int64)
        missing = image_rows < 0
        images = np.zeros((len(rows), 512), dtype=np.float32)
        if np.any(~missing):
            images[~missing] = np.asarray(self.images[image_rows[~missing]], dtype=np.float32)
        return (
            torch.from_numpy(images).to(self.device),
            self.categories[index_tensor],
            torch.from_numpy(missing.astype(np.float32)).to(self.device),
        )


def _load_relations(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in ("anchor", "positive", "negative", "teacher_cosine", "teacher_rank")
    }
    for path in paths:
        relation = np.load(path)
        for key in arrays:
            arrays[key].append(np.asarray(relation[key]))
    return {key: np.concatenate(values) for key, values in arrays.items()}


def _split_mask(anchor: np.ndarray, catalog_items: list[str]) -> np.ndarray:
    codes = np.asarray([stable_u64(item + ":m4-validation") % 10 for item in catalog_items], dtype=np.uint8)
    return codes[anchor] == 0


def _positive_weight(cosine: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    weight = cosine.clamp(min=0.0) / torch.log2(rank.to(cosine.dtype) + 1.0)
    return weight / weight.mean().clamp_min(1e-6)


def _relation_loss(
    *, model: StaticItemEncoder, accessor: CatalogAccessor, relation: dict[str, np.ndarray],
    row_indices: np.ndarray, lambda_pairwise: float, amp_enabled: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    a = relation["anchor"][row_indices]
    p = relation["positive"][row_indices]
    n = relation["negative"][row_indices]
    ai, ac, am = accessor.batch(a)
    pi, pc, pm = accessor.batch(p)
    ni, nc, nm = accessor.batch(n)
    cosine = torch.as_tensor(relation["teacher_cosine"][row_indices], device=accessor.device)
    rank = torch.as_tensor(relation["teacher_rank"][row_indices], device=accessor.device)
    with torch.amp.autocast(device_type=accessor.device.type, dtype=torch.float16, enabled=amp_enabled):
        ae = model(ai, ac, am)
        pe = model(pi, pc, pm)
        ne = model(ni, nc, nm)
        weight = _positive_weight(cosine, rank)
        logits = (ae @ pe.T) / TRAINING["temperature"]
        contrastive_rows = F.cross_entropy(
            logits, torch.arange(len(row_indices), device=accessor.device), reduction="none"
        )
        contrastive = (contrastive_rows * weight).mean()
        positive_similarity = (ae * pe).sum(dim=1)
        negative_similarity = (ae * ne).sum(dim=1)
        pairwise = (
            F.softplus(TRAINING["pairwise_margin"] - positive_similarity + negative_similarity) * weight
        ).mean()
        loss = contrastive + lambda_pairwise * pairwise
    return loss, {
        "contrastive": float(contrastive.detach()),
        "pairwise": float(pairwise.detach()),
        "pair_accuracy": float((positive_similarity > negative_similarity).float().mean().detach()),
    }


def _evaluate_pairs(
    *, model: StaticItemEncoder, accessor: CatalogAccessor, relation: dict[str, np.ndarray],
    indices: np.ndarray, lambda_pairwise: float, batch_size: int = 1024,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "contrastive": 0.0, "pairwise": 0.0, "pair_accuracy": 0.0}
    rows = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            loss, parts = _relation_loss(
                model=model, accessor=accessor, relation=relation, row_indices=selected,
                lambda_pairwise=lambda_pairwise, amp_enabled=False,
            )
            count = len(selected)
            totals["loss"] += float(loss) * count
            for key, value in parts.items():
                totals[key] += value * count
            rows += count
    return {key: value / max(rows, 1) for key, value in totals.items()} | {"rows": rows}


def train_encoder(
    *, variant: str, relation: dict[str, np.ndarray], catalog_items: list[str],
    accessor: CatalogAccessor, cardinalities: list[int], lambda_pairwise: float,
    output_dir: Path, max_epochs: int, subset_modulus: int | None = None,
) -> tuple[StaticItemEncoder, dict[str, Any]]:
    torch.manual_seed(TRAINING["seed"])
    np.random.seed(TRAINING["seed"])
    if accessor.device.type == "cuda":
        torch.cuda.manual_seed_all(TRAINING["seed"])
        torch.cuda.reset_peak_memory_stats()
    model = StaticItemEncoder(variant=variant, cardinalities=cardinalities).to(accessor.device)
    validation_mask = _split_mask(relation["anchor"], catalog_items)
    all_indices = np.arange(len(relation["anchor"]), dtype=np.int64)
    train_indices = all_indices[~validation_mask]
    validation_indices = all_indices[validation_mask]
    if subset_modulus is not None:
        train_indices = train_indices[(train_indices * 2_654_435_761) % subset_modulus == 0]
        validation_indices = validation_indices[(validation_indices * 2_654_435_761) % subset_modulus == 0]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=TRAINING["learning_rate"], weight_decay=TRAINING["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=accessor.device.type == "cuda")
    rng = np.random.default_rng(TRAINING["seed"])
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        train_loss = 0.0
        train_rows = 0
        for start in range(0, len(shuffled), TRAINING["batch_size"]):
            selected = shuffled[start : start + TRAINING["batch_size"]]
            if len(selected) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _relation_loss(
                model=model, accessor=accessor, relation=relation, row_indices=selected,
                lambda_pairwise=lambda_pairwise, amp_enabled=accessor.device.type == "cuda",
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach()) * len(selected)
            train_rows += len(selected)
        if len(validation_indices) > 100_000:
            valid_subset = validation_indices[
                np.linspace(0, len(validation_indices) - 1, 100_000, dtype=np.int64)
            ]
        else:
            valid_subset = validation_indices
        validation = _evaluate_pairs(
            model=model, accessor=accessor, relation=relation, indices=valid_subset,
            lambda_pairwise=lambda_pairwise,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_rows, 1),
            "train_rows": train_rows,
            "validation": validation,
        }
        history.append(row)
        print(
            f"M4.2 {variant} epoch {epoch}: train_loss={row['train_loss']:.5f} "
            f"valid_loss={validation['loss']:.5f} pair_acc={validation['pair_accuracy']:.4f}",
            flush=True,
        )
        if validation["loss"] < best_loss - 1e-5:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale > TRAINING["early_stopping_patience"]:
                break
    if best_state is None:
        raise RuntimeError("student training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "student_model.pt"
    torch.save(
        {"state_dict": model.state_dict(), "variant": variant, "cardinalities": cardinalities}, model_path
    )
    evidence = {
        "variant": variant,
        "lambda_pairwise": lambda_pairwise,
        "train_relation_rows": int(len(train_indices)),
        "validation_relation_rows": int(len(validation_indices)),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "rows_per_second": len(train_indices) * len(history) / max(time.perf_counter() - started, 1e-9),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if accessor.device.type == "cuda" else 0,
        "model": file_identity(model_path),
    }
    return model, evidence


def encode_catalog(
    *, model: StaticItemEncoder, accessor: CatalogAccessor, rows: int, output_path: Path,
    batch_size: int = 2048,
) -> dict[str, Any]:
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=(rows, 128))
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, rows, batch_size):
            indices = np.arange(start, min(start + batch_size, rows), dtype=np.int64)
            images, categories, missing = accessor.batch(indices)
            with torch.amp.autocast(
                device_type=accessor.device.type, dtype=torch.float16, enabled=accessor.device.type == "cuda"
            ):
                encoded = model(images, categories, missing)
            output[start : start + len(indices)] = encoded.float().cpu().numpy().astype(np.float16)
    output.flush()
    del output
    check = np.load(output_path, mmap_mode="r")
    norms = np.linalg.norm(np.asarray(check[: min(rows, 10_000)], dtype=np.float32), axis=1)
    if not np.isfinite(norms).all() or float(np.min(norms)) < 0.995 or float(np.max(norms)) > 1.005:
        raise RuntimeError("student catalog embedding normalization gate failed")
    return {
        "rows": rows,
        "dimensions": 128,
        "dtype": "float16",
        "sample_norm_min": float(norms.min()),
        "sample_norm_mean": float(norms.mean()),
        "sample_norm_max": float(norms.max()),
        "elapsed_seconds": time.perf_counter() - started,
        "artifact": file_identity(output_path),
    }


def representation_diagnostics(
    *, embeddings_path: Path, relation_paths: list[Path], catalog_items: list[str],
    device: torch.device, max_anchors_per_cutoff: int = 500,
) -> dict[str, Any]:
    embeddings = np.load(embeddings_path, mmap_mode="r")
    per_cutoff: dict[str, Any] = {}
    pooled_pairs = 0
    pooled_correct = 0
    bucket_rows: dict[str, list[int]] = {"head": [0, 0], "middle": [0, 0], "tail": [0, 0]}
    for path in relation_paths:
        rel = np.load(path)
        anchors = np.asarray(rel["anchor"])
        validation = _split_mask(anchors, catalog_items)
        candidate_rows = np.flatnonzero(validation)
        selected = candidate_rows[: min(100_000, len(candidate_rows))]
        a = np.asarray(embeddings[np.asarray(rel["anchor"])[selected]], dtype=np.float32)
        p = np.asarray(embeddings[np.asarray(rel["positive"])[selected]], dtype=np.float32)
        n = np.asarray(embeddings[np.asarray(rel["negative"])[selected]], dtype=np.float32)
        correct = (np.sum(a * p, axis=1) > np.sum(a * n, axis=1))
        token_count = np.asarray(rel["catalog_token_count"])[np.asarray(rel["anchor"])[selected]]
        positive_counts = token_count[token_count > 0]
        q1, q2 = np.quantile(positive_counts, [1 / 3, 2 / 3]) if len(positive_counts) else (0, 0)
        for name, mask in {
            "tail": token_count <= q1,
            "middle": (token_count > q1) & (token_count <= q2),
            "head": token_count > q2,
        }.items():
            bucket_rows[name][0] += int(correct[mask].sum())
            bucket_rows[name][1] += int(mask.sum())
        pooled_pairs += len(selected)
        pooled_correct += int(correct.sum())

        all_unique, first_indices = np.unique(anchors, return_index=True)
        first_by_anchor = {int(anchor): int(index) for anchor, index in zip(all_unique, first_indices)}
        unique_anchors = np.unique(anchors[validation])[:max_anchors_per_cutoff]
        teacher_vocab = np.unique(anchors)
        vocab_embeddings = torch.from_numpy(np.asarray(embeddings[teacher_vocab], dtype=np.float32)).to(device)
        recalls20: list[float] = []
        recalls50: list[float] = []
        ordering: list[float] = []
        for start in range(0, len(unique_anchors), 64):
            batch_anchors = unique_anchors[start : start + 64]
            query = torch.from_numpy(np.asarray(embeddings[batch_anchors], dtype=np.float32)).to(device)
            scores = query @ vocab_embeddings.T
            top50 = torch.topk(scores, k=min(51, len(teacher_vocab)), dim=1).indices.cpu().numpy()
            for row_index, anchor in enumerate(batch_anchors):
                first = first_by_anchor[int(anchor)]
                positions = np.arange(first, first + 20, dtype=np.int64)
                positive_ids = np.asarray(rel["positive"])[positions]
                teacher_ranks = np.asarray(rel["teacher_rank"])[positions].astype(np.float32)
                truth = set(positive_ids.tolist())
                ranked = [int(teacher_vocab[index]) for index in top50[row_index] if int(teacher_vocab[index]) != int(anchor)][:50]
                recalls20.append(len(truth.intersection(ranked[:20])) / max(len(truth), 1))
                recalls50.append(len(truth.intersection(ranked[:50])) / max(len(truth), 1))
                positive_embeddings = np.asarray(embeddings[positive_ids], dtype=np.float32)
                student_scores = positive_embeddings @ np.asarray(embeddings[int(anchor)], dtype=np.float32)
                correlation = np.corrcoef(-teacher_ranks, student_scores)[0, 1] if len(student_scores) > 1 else np.nan
                if np.isfinite(correlation):
                    ordering.append(float(correlation))
        per_cutoff[path.parent.name] = {
            "diagnostic_anchor_count": len(unique_anchors),
            "teacher_neighbor_recall@20": float(np.mean(recalls20)) if recalls20 else 0.0,
            "teacher_neighbor_recall@50": float(np.mean(recalls50)) if recalls50 else 0.0,
            "teacher_rank_spearman_mean": float(np.mean(ordering)) if ordering else None,
            "pair_accuracy": float(correct.mean()) if len(correct) else 0.0,
            "pair_denominator": len(selected),
        }
    return {
        "per_cutoff": per_cutoff,
        "pooled_pair_accuracy": pooled_correct / max(pooled_pairs, 1),
        "pooled_pair_denominator": pooled_pairs,
        "teacher_popularity_buckets": {
            name: {"pair_accuracy": correct / max(total, 1), "pair_denominator": total}
            for name, (correct, total) in bucket_rows.items()
        },
        "note": "head/middle/tail use within-cutoff teacher token-count tertiles; diagnostic anchors are deterministic held-out anchors capped per cutoff",
    }


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for M4.2 but torch.cuda.is_available() is false")
    return torch.device(requested)


def run_m42(
    *, source_root: Path, teacher_dir: Path, artifact_dir: Path, output_dir: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = _device(device_name)
    fashionclip_root = source_root / "artifacts" / "m2_4" / "fashionclip-full-v1"
    catalog_dir = artifact_dir / "static_catalog"
    catalog = build_static_catalog(
        articles_path=source_root / "data" / "raw" / "articles.csv",
        fashionclip_root=fashionclip_root,
        artifact_dir=catalog_dir,
    )
    catalog_items = pd.read_csv(catalog_dir / "catalog_items.csv", dtype={"article_id": str})["article_id"].tolist()
    accessor = CatalogAccessor(fashionclip_root=fashionclip_root, catalog_dir=catalog_dir, device=device)
    first_cutoff = sorted(path.name for path in teacher_dir.iterdir() if path.is_dir())[0]
    mechanics_relation = _load_relations([teacher_dir / first_cutoff / "teacher_relations.npz"])
    mechanics_dir = artifact_dir / "mechanics"
    _, one_percent = train_encoder(
        variant="multimodal", relation=mechanics_relation, catalog_items=catalog_items,
        accessor=accessor, cardinalities=catalog["cardinalities"], lambda_pairwise=LAMBDA_GRID[0],
        output_dir=mechanics_dir / "one_percent", max_epochs=1, subset_modulus=100,
    )
    lambda_trials: dict[str, Any] = {}
    selected_lambda = None
    selected_score = float("inf")
    for value in LAMBDA_GRID:
        _, evidence = train_encoder(
            variant="multimodal", relation=mechanics_relation, catalog_items=catalog_items,
            accessor=accessor, cardinalities=catalog["cardinalities"], lambda_pairwise=value,
            output_dir=mechanics_dir / f"lambda_{value}", max_epochs=1, subset_modulus=10,
        )
        lambda_trials[str(value)] = evidence
        if evidence["best_validation_loss"] < selected_score:
            selected_score = evidence["best_validation_loss"]
            selected_lambda = value
    assert selected_lambda is not None
    windows: dict[str, Any] = {}
    for window, protocol in ROLLING_PROTOCOL.items():
        relation_paths = [teacher_dir / cutoff / "teacher_relations.npz" for cutoff in protocol["outer_train"]]
        relation = _load_relations(relation_paths)
        windows[window] = {"training_cutoffs": protocol["outer_train"], "variants": {}}
        for variant in VARIANTS:
            variant_dir = artifact_dir / window / variant
            model, training = train_encoder(
                variant=variant, relation=relation, catalog_items=catalog_items,
                accessor=accessor, cardinalities=catalog["cardinalities"],
                lambda_pairwise=float(selected_lambda), output_dir=variant_dir,
                max_epochs=TRAINING["max_epochs"],
            )
            embedding_path = variant_dir / "catalog_embeddings.float16.npy"
            encoding = encode_catalog(
                model=model, accessor=accessor, rows=catalog["rows"], output_path=embedding_path
            )
            diagnostics = representation_diagnostics(
                embeddings_path=embedding_path, relation_paths=relation_paths,
                catalog_items=catalog_items, device=device,
            )
            manifest = {
                "schema_version": "m4.2-student-v1",
                "stage": "M4.2",
                "status": "completed",
                "run_id": M4_RUN_ID,
                "window": window,
                "outer_validation_cutoff": protocol["outer_validation"],
                "variant": variant,
                "training_cutoffs": protocol["outer_train"],
                "architecture": {
                    "image_branch": "512->256",
                    "categorical_embedding_dimensions": [
                        max(8, min(32, int(round(math.sqrt(value + 1))))) for value in catalog["cardinalities"]
                    ],
                    "metadata_branch": "categorical embeddings plus missing flags ->128",
                    "fusion": "image-only:257->128; metadata-only:128->128; multimodal:385->256->128",
                    "output": "128-dimensional L2-normalized item vector",
                },
                "loss": {
                    "weighted_contrastive": "in-batch positive cross-entropy weighted by teacher cosine/log2(rank+1)",
                    "pairwise": "softplus(margin - positive similarity + hard-negative similarity)",
                    "lambda_pairwise": selected_lambda,
                },
                "training": training,
                "encoding": encoding,
                "diagnostics": diagnostics,
                "inputs": {
                    "teacher_relations": [file_identity(path) for path in relation_paths],
                    "static_catalog": catalog,
                    "code": file_identity(Path(__file__)),
                },
                "device": {
                    "type": device.type,
                    "name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
                    "torch": torch.__version__,
                    "amp": device.type == "cuda",
                },
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "final_week": "not_run",
            }
            manifest_path = variant_dir / "manifest.json"
            atomic_json(manifest_path, manifest)
            windows[window]["variants"][variant] = {**manifest, "manifest": file_identity(manifest_path)}
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    result = {
        "schema_version": "m4.2-student-summary-v1",
        "stage": "M4.2",
        "status": "measured",
        "run_id": M4_RUN_ID,
        "training_contract": TRAINING,
        "mechanics": {
            "one_percent": one_percent,
            "ten_percent_lambda_trials": lambda_trials,
            "selected_lambda": selected_lambda,
            "selection_metric": "held-out teacher-relation validation loss; no outer purchase label",
        },
        "catalog": catalog,
        "windows": windows,
        "resources": {
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "artifact_bytes": int(sum(
                value["encoding"]["artifact"]["bytes"] + value["training"]["model"]["bytes"]
                for window in windows.values() for value in window["variants"].values()
            )),
        },
        "final_week": "not_run",
    }
    atomic_json(output_dir / "M4_2_metrics.json", result)
    return result
