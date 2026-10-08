from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    for attempt in range(1, 6):
        try:
            temporary.write_text(body, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * attempt)


def _stable_topk(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    order = np.lexsort((candidate_indices, -candidate_scores))
    ordered_indices = candidate_indices[order]
    ordered_scores = candidate_scores[order]
    ambiguous = bool(
        len(ordered_scores) > top_k
        and ordered_scores[top_k - 1] == ordered_scores[-1]
    )
    return ordered_indices[:top_k], ordered_scores[:top_k], ambiguous


def _load_embedding_contract(metrics_path: Path) -> dict[str, Any]:
    metrics_path = metrics_path.expanduser().resolve()
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("schema_version") != "m2.4-fashionclip-full-v1":
        raise ValueError("M2.5 requires m2.4-fashionclip-full-v1")
    if metrics.get("status") != "completed":
        raise ValueError("full embedding status is not completed")
    embeddings = metrics.get("embeddings", {})
    if int(embeddings.get("failed_rows", -1)) != 0:
        raise ValueError("full embedding contains failed rows")
    if not embeddings.get("all_valid_finite"):
        raise ValueError("full embedding finite gate did not pass")
    embedding_path = Path(str(embeddings["artifact"])).resolve()
    items_path = Path(str(embeddings["items_artifact"])).resolve()
    for path, size, sha in (
        (
            embedding_path,
            int(embeddings["artifact_bytes"]),
            str(embeddings["artifact_sha256"]),
        ),
        (
            items_path,
            items_path.stat().st_size if items_path.exists() else -1,
            str(embeddings["items_sha256"]),
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != size:
            raise ValueError(f"artifact byte mismatch: {path}")
        if _sha256(path) != sha:
            raise ValueError(f"artifact SHA256 mismatch: {path}")
    matrix = np.load(embedding_path, mmap_mode="r")
    rows = int(embeddings["rows"])
    dimensions = int(embeddings["dimensions"])
    if matrix.shape != (rows, dimensions) or matrix.dtype != np.float16:
        raise ValueError("embedding shape or dtype differs from metrics")
    items = pd.read_csv(items_path, dtype={"article_id": str})
    if len(items) != rows:
        raise ValueError("items row count differs from embeddings")
    if items["row_index"].tolist() != list(range(rows)):
        raise ValueError("items row_index is not contiguous and ordered")
    if items["article_id"].duplicated().any():
        raise ValueError("items article_id is not unique")
    return {
        "metrics_path": str(metrics_path),
        "metrics_sha256": _sha256(metrics_path),
        "embedding_path": str(embedding_path),
        "embedding_sha256": embeddings["artifact_sha256"],
        "items_path": str(items_path),
        "items_sha256": embeddings["items_sha256"],
        "source_contract_sha256": metrics["contract_sha256"],
        "rows": rows,
        "dimensions": dimensions,
    }


def _render_neighbor_report(metrics: dict[str, Any]) -> str:
    runtime = metrics["runtime"]
    audit = metrics["audit"]
    return "\n".join(
        [
            "# M2.5 exact image-neighbor artifact",
            "",
            f"- status: {metrics['status']}",
            f"- query rows / corpus rows: {metrics['query_rows']} / {metrics['corpus_rows']}",
            f"- top-k: {metrics['top_k']}",
            f"- self neighbors: {audit['self_neighbors']}",
            f"- duplicate neighbors within row: {audit['duplicate_neighbors']}",
            f"- ambiguous buffered boundaries: {audit['ambiguous_boundaries']}",
            f"- score min / mean / max: {audit['score_min']:.6f} / {audit['score_mean']:.6f} / {audit['score_max']:.6f}",
            f"- search seconds: {runtime['search_seconds']:.3f}",
            f"- queries per second: {runtime['queries_per_second']:.3f}",
            f"- peak CUDA MiB: {runtime['cuda_peak_allocated_bytes'] / 1024 ** 2:.1f}",
            "",
            "## Evidence boundary",
            "",
            "- This is an exact static item-to-item cosine-neighbor artifact.",
            "- It uses no transaction rows and makes no candidate Recall or MAP claim.",
            "- User seeds and temporal evaluation must be rebuilt per cutoff in M2.5.",
            "",
        ]
    )


def build_exact_image_neighbors(
    *,
    embedding_metrics_path: Path,
    output_dir: Path,
    report_dir: Path,
    top_k: int = 100,
    tie_buffer: int = 32,
    query_batch_size: int = 512,
    query_limit: int | None = None,
    device: str = "cuda",
    resume: bool = False,
) -> dict[str, Any]:
    if top_k < 1 or tie_buffer < 1 or query_batch_size < 1:
        raise ValueError("top_k, tie_buffer, and query_batch_size must be positive")
    source = _load_embedding_contract(embedding_metrics_path)
    corpus_rows = int(source["rows"])
    query_rows = corpus_rows if query_limit is None else min(query_limit, corpus_rows)
    if query_rows < 1 or top_k + tie_buffer >= corpus_rows:
        raise ValueError("invalid query limit or neighbor budget")

    output_dir = output_dir.expanduser().resolve()
    report_dir = report_dir.expanduser().resolve()
    if output_dir.exists() and not resume:
        raise FileExistsError(f"neighbor output exists: {output_dir}")
    if report_dir.exists() and not resume:
        raise FileExistsError(f"neighbor report exists: {report_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    invocation_started = time.perf_counter()

    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
        contract: dict[str, Any] = {
            "schema_version": "m2.5-exact-image-neighbors-contract-v1",
            "source": source,
            "query_rule": "first_rows_in_frozen_lexicographic_item_order",
            "query_rows": query_rows,
            "corpus_rows": corpus_rows,
            "top_k": top_k,
            "tie_buffer": tie_buffer,
            "query_batch_size": query_batch_size,
            "compute_dtype": "float32",
            "tie_order": "cosine_desc_then_neighbor_row_index_asc",
            "device": device,
            "device_name": device_name,
            "torch": torch.__version__,
        }
        canonical = json.dumps(
            contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        contract_sha = hashlib.sha256(canonical).hexdigest()
        contract_path = output_dir / "contract.json"
        progress_path = output_dir / "progress.json"
        indices_path = output_dir / "neighbor_indices.int32.npy"
        scores_path = output_dir / "neighbor_scores.float16.npy"
        ambiguous_path = output_dir / "ambiguous_rows.int64.npy"

        if contract_path.exists():
            if not resume:
                raise FileExistsError(contract_path)
            if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
                raise ValueError("neighbor resume contract mismatch")
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("contract_sha256") != contract_sha:
                raise ValueError("neighbor progress contract mismatch")
            indices = np.load(indices_path, mmap_mode="r+")
            scores_out = np.load(scores_path, mmap_mode="r+")
            next_row = int(progress["next_row"])
        else:
            if resume:
                raise FileNotFoundError("resume requested but contract is missing")
            if any(output_dir.iterdir()):
                raise FileExistsError("new neighbor output directory is not empty")
            _atomic_json(contract_path, contract)
            indices = np.lib.format.open_memmap(
                indices_path, mode="w+", dtype=np.int32, shape=(query_rows, top_k)
            )
            scores_out = np.lib.format.open_memmap(
                scores_path, mode="w+", dtype=np.float16, shape=(query_rows, top_k)
            )
            indices[:] = -1
            scores_out[:] = np.nan
            indices.flush()
            scores_out.flush()
            next_row = 0
            progress = {
                "schema_version": "m2.5-exact-image-neighbors-progress-v1",
                "status": "running",
                "contract_sha256": contract_sha,
                "next_row": 0,
                "query_rows": query_rows,
                "search_seconds_accumulated": 0.0,
            }
            _atomic_json(progress_path, progress)

        if indices.shape != (query_rows, top_k) or scores_out.shape != (query_rows, top_k):
            raise ValueError("neighbor resume array shape mismatch")
        if not 0 <= next_row <= query_rows:
            raise ValueError("neighbor resume next_row out of range")
        if next_row and (np.any(indices[:next_row] < 0) or not np.isfinite(scores_out[:next_row]).all()):
            raise ValueError("neighbor processed prefix is invalid")
        suffix_written = np.any(indices[next_row:] != -1, axis=1) | np.any(
            np.isfinite(scores_out[next_row:]), axis=1
        )
        recovered_uncommitted_rows = int(np.count_nonzero(suffix_written))
        if recovered_uncommitted_rows:
            indices[next_row:] = -1
            scores_out[next_row:] = np.nan
            indices.flush()
            scores_out.flush()
            progress["recovered_uncommitted_rows"] = int(
                progress.get("recovered_uncommitted_rows", 0)
            ) + recovered_uncommitted_rows
            _atomic_json(progress_path, progress)

        matrix = np.load(Path(source["embedding_path"]), mmap_mode="r")
        corpus = torch.from_numpy(np.asarray(matrix, dtype=np.float32)).to(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        search_this_run = 0.0
        ambiguous_rows: list[int] = []
        retrieve_k = top_k + tie_buffer
        for batch_number, batch_start in enumerate(
            range(next_row, query_rows, query_batch_size), start=1
        ):
            batch_end = min(batch_start + query_batch_size, query_rows)
            started = time.perf_counter()
            query = corpus[batch_start:batch_end]
            similarities = query @ corpus.T
            local = torch.arange(batch_end - batch_start, device=device)
            global_rows = torch.arange(batch_start, batch_end, device=device)
            similarities[local, global_rows] = -torch.inf
            values, neighbors = torch.topk(
                similarities, k=retrieve_k, dim=1, largest=True, sorted=False
            )
            candidate_scores = values.cpu().numpy()
            candidate_indices = neighbors.cpu().numpy()
            del similarities, values, neighbors
            for local_row in range(batch_end - batch_start):
                stable_indices, stable_scores, ambiguous = _stable_topk(
                    candidate_indices[local_row], candidate_scores[local_row], top_k
                )
                row_index = batch_start + local_row
                indices[row_index] = stable_indices.astype(np.int32)
                scores_out[row_index] = stable_scores.astype(np.float16)
                if ambiguous:
                    ambiguous_rows.append(row_index)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            search_this_run += elapsed
            indices.flush()
            scores_out.flush()
            progress.update(
                {
                    "status": "running" if batch_end < query_rows else "searched",
                    "next_row": batch_end,
                    "search_seconds_accumulated": float(
                        progress.get("search_seconds_accumulated", 0.0)
                    )
                    + elapsed,
                }
            )
            _atomic_json(progress_path, progress)
            if batch_number == 1 or batch_number % 10 == 0 or batch_end == query_rows:
                print(
                    f"neighbor progress {batch_end}/{query_rows} "
                    f"({batch_end / query_rows:.1%}), ambiguous={len(ambiguous_rows)}",
                    flush=True,
                )

        np.save(ambiguous_path, np.asarray(ambiguous_rows, dtype=np.int64), allow_pickle=False)
        self_neighbors = int(
            sum(np.count_nonzero(indices[row] == row) for row in range(query_rows))
        )
        duplicate_neighbors = int(
            sum(top_k - len(np.unique(indices[row])) for row in range(query_rows))
        )
        finite = bool(np.isfinite(scores_out).all())
        ordered = bool(np.all(scores_out[:, :-1] >= scores_out[:, 1:]))
        if self_neighbors or duplicate_neighbors or not finite or not ordered or ambiguous_rows:
            raise RuntimeError(
                "neighbor audit failed: "
                f"self={self_neighbors}, duplicate={duplicate_neighbors}, "
                f"finite={finite}, ordered={ordered}, ambiguous={len(ambiguous_rows)}"
            )
        total_search = float(progress["search_seconds_accumulated"])
        metrics: dict[str, Any] = {
            "schema_version": "m2.5-exact-image-neighbors-v1",
            "status": "completed",
            "boundary": "static exact image neighbors; no user retrieval or MAP claim",
            "contract": contract,
            "contract_sha256": contract_sha,
            "query_rows": query_rows,
            "corpus_rows": corpus_rows,
            "top_k": top_k,
            "artifacts": {
                "indices": str(indices_path),
                "indices_bytes": indices_path.stat().st_size,
                "indices_sha256": _sha256(indices_path),
                "scores": str(scores_path),
                "scores_bytes": scores_path.stat().st_size,
                "scores_sha256": _sha256(scores_path),
                "ambiguous_rows": str(ambiguous_path),
                "ambiguous_rows_sha256": _sha256(ambiguous_path),
            },
            "audit": {
                "all_finite": finite,
                "scores_nonincreasing": ordered,
                "self_neighbors": self_neighbors,
                "duplicate_neighbors": duplicate_neighbors,
                "ambiguous_boundaries": len(ambiguous_rows),
                "index_min": int(indices.min()),
                "index_max": int(indices.max()),
                "score_min": float(scores_out.min()),
                "score_mean": float(scores_out.mean()),
                "score_max": float(scores_out.max()),
            },
            "runtime": {
                "search_seconds": total_search,
                "queries_per_second": query_rows / total_search,
                "cuda_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
                ),
                "invocation_seconds": time.perf_counter() - invocation_started,
                "resumed": resume,
                "recovered_uncommitted_rows": int(
                    progress.get("recovered_uncommitted_rows", 0)
                ),
            },
        }
        _atomic_json(output_dir / "metrics.json", metrics)
        progress.update({"status": "completed", "next_row": query_rows})
        _atomic_json(progress_path, progress)
        report_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(report_dir / "metrics.json", metrics)
        report = _render_neighbor_report(metrics)
        (output_dir / "M2_5_EXACT_NEIGHBORS.md").write_text(report, encoding="utf-8")
        (report_dir / "M2_5_EXACT_NEIGHBORS.md").write_text(report, encoding="utf-8")
        return metrics
    except Exception as error:
        _atomic_json(
            output_dir / f"failure-{time.time_ns()}.json",
            {
                "schema_version": "m2.5-exact-image-neighbors-failure-v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - invocation_started,
            },
        )
        raise
