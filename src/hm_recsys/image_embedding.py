from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FASHION_CLIP_MODEL_ID = "patrickjohncyh/fashion-clip"
FASHION_CLIP_REVISION = "83cb9b65be402bbdb4d0e1b84bd53555028bfed8"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_image_name(path: Path, images_dir: Path) -> str:
    relative = path.relative_to(images_dir).as_posix()
    return "images/" + relative


def select_image_paths(images_dir: Path, limit: int) -> list[Path]:
    images_dir = images_dir.expanduser().resolve()
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    paths = [
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    ]
    if not paths:
        raise ValueError(f"no JPEG images found under {images_dir}")
    return sorted(
        paths,
        key=lambda path: hashlib.sha256(
            _relative_image_name(path, images_dir).encode("utf-8")
        ).digest(),
    )[: min(limit, len(paths))]


def _load_audit_contract(
    audit_metrics_path: Path,
    actual_image_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    audit_metrics_path = audit_metrics_path.expanduser().resolve()
    if not audit_metrics_path.is_file():
        raise FileNotFoundError(f"audit metrics not found: {audit_metrics_path}")
    audit = json.loads(audit_metrics_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != "m2.3-image-audit-v2":
        raise ValueError("embedding smoke requires m2.3-image-audit-v2")
    if audit.get("status") != "completed":
        raise ValueError("image audit status is not completed")
    mapping = audit.get("mapping", {})
    expected = int(mapping.get("valid_image_files", -1))
    if expected != actual_image_count:
        raise ValueError(
            f"image count mismatch: audit={expected}, directory={actual_image_count}"
        )
    if not mapping.get("manifest_sha256"):
        raise ValueError("image audit manifest SHA256 is missing")
    contract = {
        "path": str(audit_metrics_path),
        "bytes": audit_metrics_path.stat().st_size,
        "sha256": _file_sha256(audit_metrics_path),
        "schema_version": audit["schema_version"],
        "source_manifest_sha256": mapping["manifest_sha256"],
        "valid_image_files": expected,
        "article_coverage_rate": audit["article_coverage"]["coverage_rate"],
    }
    return audit, contract


def _render_report(metrics: dict[str, object]) -> str:
    embeddings = metrics["embeddings"]
    runtime = metrics["runtime"]
    return chr(10).join(
        [
            "# M2.4 FashionCLIP embedding smoke",
            "",
            "## 状态",
            "",
            f"- status: {metrics['status']}",
            f"- model: {metrics['model']['id']}",
            f"- revision: {metrics['model']['revision']}",
            f"- device: {metrics['environment']['device_name']}",
            "",
            "## 输出",
            "",
            f"- images: {embeddings['rows']}",
            f"- dimensions: {embeddings['dimensions']}",
            f"- storage dtype: {embeddings['storage_dtype']}",
            f"- finite: {embeddings['all_finite']}",
            f"- norm min/mean/max: {embeddings['norm_min']:.6f} / "
            f"{embeddings['norm_mean']:.6f} / {embeddings['norm_max']:.6f}",
            "",
            "## 成本",
            "",
            f"- model load seconds: {runtime['model_load_seconds']:.3f}",
            f"- inference seconds: {runtime['inference_seconds']:.3f}",
            f"- images per second: {runtime['images_per_second']:.3f}",
            f"- CUDA peak allocated MiB: "
            f"{runtime['cuda_peak_allocated_bytes'] / 1024 ** 2:.1f}",
            "",
            "## 边界",
            "",
            "- 这是 bounded embedding mechanics smoke，不是 image recall 或 MAP 结果。",
            "- 图片选择由 SHA256(relative filename) 确定，与目录遍历顺序无关。",
            "- embedding 已 L2 normalize；全量编码仍需单独 run ID 与全量失败清单。",
            "",
        ]
    )


def run_embedding_smoke(
    *,
    images_dir: Path,
    audit_metrics_path: Path,
    output_dir: Path,
    cache_dir: Path,
    limit: int = 128,
    batch_size: int = 16,
    device: str = "cuda",
    model_id: str = FASHION_CLIP_MODEL_ID,
    revision: str = FASHION_CLIP_REVISION,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    images_dir = images_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite embedding run: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()

    try:
        all_image_count = sum(
            1
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        )
        _, audit_contract = _load_audit_contract(
            audit_metrics_path, all_image_count
        )
        selected = select_image_paths(images_dir, limit)

        import torch
        import transformers
        from transformers import CLIPModel, CLIPProcessor

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        cache_dir.mkdir(parents=True, exist_ok=True)
        load_started = time.perf_counter()
        processor = CLIPProcessor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=False,
            use_fast=False,
        )
        model = CLIPModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=False,
            weights_only=True,
        )
        model.eval()
        model.to(device)
        model_load_seconds = time.perf_counter() - load_started

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()
        batches: list[np.ndarray] = []
        records: list[dict[str, object]] = []
        for batch_start in range(0, len(selected), batch_size):
            batch_paths = selected[batch_start : batch_start + batch_size]
            images: list[Image.Image] = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16
                    ):
                        features = model.get_image_features(
                            pixel_values=pixel_values
                        )
                else:
                    features = model.get_image_features(
                        pixel_values=pixel_values
                    )
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = torch.nn.functional.normalize(
                features.float(), p=2, dim=1
            )
            batches.append(features.cpu().numpy())
            for path in batch_paths:
                name = _relative_image_name(path, images_dir)
                records.append(
                    {
                        "row_index": len(records),
                        "article_id": path.stem,
                        "image_name": name,
                        "image_bytes": path.stat().st_size,
                    }
                )
            del images, inputs, pixel_values, features
        if device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started

        matrix = np.concatenate(batches, axis=0)
        norms = np.linalg.norm(matrix, axis=1)
        storage = matrix.astype(np.float16)
        embeddings_path = output_dir / "embeddings.float16.npy"
        items_path = output_dir / "items.csv"
        np.save(embeddings_path, storage, allow_pickle=False)
        pd.DataFrame.from_records(records).to_csv(items_path, index=False)

        metrics: dict[str, object] = {
            "schema_version": "m2.4-fashionclip-smoke-v1",
            "status": "completed",
            "boundary": (
                "bounded mechanics smoke; no retrieval or ranking claim"
            ),
            "audit_contract": audit_contract,
            "selection": {
                "rule": "smallest_sha256(relative_image_name)",
                "requested_limit": limit,
                "selected_images": len(selected),
                "first_image_names": [
                    _relative_image_name(path, images_dir)
                    for path in selected[:10]
                ],
            },
            "model": {
                "id": model_id,
                "revision": revision,
                "trust_remote_code": False,
                "weights_only": True,
                "processor_use_fast": False,
                "processor_class": type(processor).__name__,
                "model_class": type(model).__name__,
            },
            "environment": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "device": device,
                "device_name": (
                    torch.cuda.get_device_name(0)
                    if device == "cuda"
                    else "cpu"
                ),
                "cuda_runtime": torch.version.cuda,
                "batch_size": batch_size,
            },
            "embeddings": {
                "rows": int(storage.shape[0]),
                "dimensions": int(storage.shape[1]),
                "storage_dtype": str(storage.dtype),
                "all_finite": bool(np.isfinite(storage).all()),
                "norm_min": float(norms.min()),
                "norm_mean": float(norms.mean()),
                "norm_max": float(norms.max()),
                "artifact": str(embeddings_path),
                "artifact_bytes": embeddings_path.stat().st_size,
                "artifact_sha256": _file_sha256(embeddings_path),
                "items_artifact": str(items_path),
                "items_bytes": items_path.stat().st_size,
                "items_sha256": _file_sha256(items_path),
            },
            "runtime": {
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
                "images_per_second": (
                    len(selected) / inference_seconds
                    if inference_seconds
                    else 0.0
                ),
                "cuda_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if device == "cuda"
                    else 0
                ),
                "total_seconds": time.perf_counter() - started,
            },
        }
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_dir / "M2_4_EMBEDDING_SMOKE.md").write_text(
            _render_report(metrics), encoding="utf-8"
        )
        return metrics
    except Exception as error:
        failure = {
            "schema_version": "m2.4-fashionclip-smoke-v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise

def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def select_all_image_paths(images_dir: Path) -> list[Path]:
    images_dir = images_dir.expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    paths = sorted(
        (
            path
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        ),
        key=lambda path: _relative_image_name(path, images_dir),
    )
    if not paths:
        raise ValueError(f"no JPEG images found under {images_dir}")
    return paths


def _full_item_records(
    paths: list[Path], images_dir: Path
) -> list[dict[str, object]]:
    return [
        {
            "row_index": index,
            "article_id": path.stem,
            "image_name": _relative_image_name(path, images_dir),
            "image_bytes": path.stat().st_size,
        }
        for index, path in enumerate(paths)
    ]


def _item_manifest_sha256(records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            (
                f"{record['row_index']}\t{record['article_id']}\t"
                f"{record['image_name']}\t{record['image_bytes']}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _encode_image_paths(
    *, paths: list[Path], processor: object, model: object, device: str
) -> np.ndarray:
    import torch

    images: list[Image.Image] = []
    try:
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    features = model.get_image_features(pixel_values=pixel_values)
            else:
                features = model.get_image_features(pixel_values=pixel_values)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
        return features.cpu().numpy().astype(np.float16)
    finally:
        for image in images:
            image.close()


def _render_full_embedding_report(metrics: dict[str, object]) -> str:
    embeddings = metrics["embeddings"]
    runtime = metrics["runtime"]
    return chr(10).join(
        [
            "# M2.4 full FashionCLIP item embeddings",
            "",
            "## 状态",
            "",
            f"- status: {metrics['status']}",
            f"- rows: {embeddings['rows']}",
            f"- valid rows: {embeddings['valid_rows']}",
            f"- failed rows: {embeddings['failed_rows']}",
            f"- dimensions: {embeddings['dimensions']}",
            f"- finite valid embeddings: {embeddings['all_valid_finite']}",
            "",
            "## 冻结契约",
            "",
            f"- audit SHA256: {metrics['contract']['audit_metrics_sha256']}",
            f"- item manifest SHA256: {metrics['contract']['item_manifest_sha256']}",
            f"- model revision: {metrics['contract']['revision']}",
            f"- processor use_fast: {metrics['contract']['processor_use_fast']}",
            f"- contract SHA256: {metrics['contract_sha256']}",
            "",
            "## 成本",
            "",
            f"- model load seconds: {runtime['model_load_seconds']:.3f}",
            f"- inference seconds: {runtime['inference_seconds']:.3f}",
            f"- images per second: {runtime['images_per_second']:.3f}",
            f"- peak CUDA MiB: {runtime['cuda_peak_allocated_bytes'] / 1024 ** 2:.1f}",
            f"- wall seconds this invocation: {runtime['invocation_seconds']:.3f}",
            "",
            "## 证据边界",
            "",
            "- 这是静态 item embedding 资产，不包含用户行为或验证周交互。",
            "- 本结果只证明全量图片编码完成；尚未证明 image retrieval 或 MAP@12 增益。",
            "- 断点续跑仅在审计、item manifest、模型、processor、环境、batch 与形状契约完全相同时开放。",
            "",
        ]
    )


def run_full_embeddings(
    *,
    images_dir: Path,
    audit_metrics_path: Path,
    output_dir: Path,
    report_dir: Path,
    cache_dir: Path,
    batch_size: int = 64,
    device: str = "cuda",
    resume: bool = False,
    model_id: str = FASHION_CLIP_MODEL_ID,
    revision: str = FASHION_CLIP_REVISION,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    images_dir = images_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    report_dir = report_dir.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()
    invocation_started = time.perf_counter()

    if output_dir.exists() and not resume:
        raise FileExistsError(
            f"embedding run exists; pass resume only for the same contract: {output_dir}"
        )
    if report_dir.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite report run: {report_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        paths = select_all_image_paths(images_dir)
        _, audit_contract = _load_audit_contract(audit_metrics_path, len(paths))
        records = _full_item_records(paths, images_dir)
        item_manifest_sha256 = _item_manifest_sha256(records)
        items_csv_text = pd.DataFrame.from_records(records).to_csv(index=False)
        items_sha256 = hashlib.sha256(items_csv_text.encode("utf-8" )).hexdigest()

        import torch
        import transformers
        from transformers import CLIPModel, CLIPProcessor

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        cache_dir.mkdir(parents=True, exist_ok=True)
        load_started = time.perf_counter()
        processor = CLIPProcessor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=False,
            use_fast=False,
        )
        model = CLIPModel.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            trust_remote_code=False,
            weights_only=True,
        )
        model.eval()
        model.to(device)
        model_load_seconds = time.perf_counter() - load_started
        dimensions = int(model.config.projection_dim)
        device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"

        contract: dict[str, object] = {
            "schema_version": "m2.4-fashionclip-full-contract-v1",
            "audit_metrics_sha256": audit_contract["sha256"],
            "source_manifest_sha256": audit_contract["source_manifest_sha256"],
            "image_count": len(paths),
            "item_order": "lexicographic_relative_image_name",
            "item_manifest_sha256": item_manifest_sha256,
            "items_sha256": items_sha256,
            "model_id": model_id,
            "revision": revision,
            "trust_remote_code": False,
            "weights_only": True,
            "processor_use_fast": False,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
            "device_name": device_name,
            "autocast_dtype": "float16" if device == "cuda" else None,
            "batch_size": batch_size,
            "dimensions": dimensions,
            "storage_dtype": "float16",
        }
        contract_bytes = json.dumps(
            contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
        contract_path = output_dir / "contract.json"
        items_path = output_dir / "items.csv"
        embeddings_path = output_dir / "embeddings.float16.npy"
        row_status_path = output_dir / "row_status.uint8.npy"
        progress_path = output_dir / "progress.json"
        failures_path = output_dir / "item_failures.jsonl"

        if contract_path.exists():
            existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing_contract != contract:
                raise ValueError("resume contract mismatch; refusing to mix embedding runs")
            if not resume:
                raise FileExistsError(f"embedding contract already exists: {contract_path}")
            if not items_path.is_file() or _file_sha256(items_path) != contract["items_sha256"]:
                raise ValueError("resume items.csv hash mismatch")
            if not progress_path.is_file():
                raise ValueError("resume progress.json is missing")
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("contract_sha256") != contract_sha256:
                raise ValueError("resume progress contract SHA256 mismatch")
            embeddings = np.load(embeddings_path, mmap_mode="r+")
            row_status = np.load(row_status_path, mmap_mode="r+")
            if embeddings.shape != (len(paths), dimensions):
                raise ValueError("resume embedding shape mismatch")
            if row_status.shape != (len(paths),):
                raise ValueError("resume row status shape mismatch")
            next_row = int(progress["next_row"])
            if not 0 <= next_row <= len(paths):
                raise ValueError("resume next_row is out of range")
            if next_row and np.any(row_status[:next_row] == 0):
                raise ValueError("resume processed prefix contains unprocessed rows")
            if np.any(row_status[next_row:] != 0):
                raise ValueError("resume suffix contains unexpected processed rows")
        else:
            if resume:
                raise FileNotFoundError("resume requested but contract.json is missing")
            if any(output_dir.iterdir()):
                raise FileExistsError("new embedding output directory is not empty")
            items_temporary = items_path.with_suffix(".csv.part")
            items_temporary.write_bytes(items_csv_text.encode("utf-8"))
            items_temporary.replace(items_path)
            _atomic_write_json(contract_path, contract)
            embeddings = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(paths), dimensions),
            )
            row_status = np.lib.format.open_memmap(
                row_status_path,
                mode="w+",
                dtype=np.uint8,
                shape=(len(paths),),
            )
            row_status[:] = 0
            row_status.flush()
            next_row = 0
            progress = {
                "schema_version": "m2.4-fashionclip-full-progress-v1",
                "status": "running",
                "contract_sha256": contract_sha256,
                "next_row": 0,
                "total_rows": len(paths),
                "valid_rows": 0,
                "failed_rows": 0,
                "inference_seconds_accumulated": 0.0,
            }
            _atomic_write_json(progress_path, progress)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        inference_this_invocation = 0.0
        for batch_number, batch_start in enumerate(
            range(next_row, len(paths), batch_size), start=1
        ):
            batch_end = min(batch_start + batch_size, len(paths))
            batch_paths = paths[batch_start:batch_end]
            batch_started = time.perf_counter()
            try:
                batch_matrix = _encode_image_paths(
                    paths=batch_paths,
                    processor=processor,
                    model=model,
                    device=device,
                )
                if batch_matrix.shape != (len(batch_paths), dimensions):
                    raise ValueError(f"unexpected embedding shape: {batch_matrix.shape}")
                if not np.isfinite(batch_matrix).all():
                    raise ValueError("batch contains non-finite embeddings")
                embeddings[batch_start:batch_end] = batch_matrix
                row_status[batch_start:batch_end] = 1
            except Exception as batch_error:
                if device == "cuda":
                    torch.cuda.empty_cache()
                for offset, path in enumerate(batch_paths):
                    row_index = batch_start + offset
                    try:
                        single = _encode_image_paths(
                            paths=[path],
                            processor=processor,
                            model=model,
                            device=device,
                        )
                        if single.shape != (1, dimensions) or not np.isfinite(single).all():
                            raise ValueError("single-item fallback produced invalid embedding")
                        embeddings[row_index] = single[0]
                        row_status[row_index] = 1
                    except Exception as item_error:
                        embeddings[row_index] = 0
                        row_status[row_index] = 2
                        with failures_path.open("a", encoding="utf-8") as stream:
                            stream.write(
                                json.dumps(
                                    {
                                        "row_index": row_index,
                                        "article_id": path.stem,
                                        "image_name": _relative_image_name(path, images_dir),
                                        "batch_error_type": type(batch_error).__name__,
                                        "batch_error": str(batch_error),
                                        "item_error_type": type(item_error).__name__,
                                        "item_error": str(item_error),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
            if device == "cuda":
                torch.cuda.synchronize()
            inference_this_invocation += time.perf_counter() - batch_started
            embeddings.flush()
            row_status.flush()
            processed = batch_end
            valid_rows = int(np.count_nonzero(row_status[:processed] == 1))
            failed_rows = int(np.count_nonzero(row_status[:processed] == 2))
            progress.update(
                {
                    "status": "running" if processed < len(paths) else "encoded",
                    "next_row": processed,
                    "valid_rows": valid_rows,
                    "failed_rows": failed_rows,
                    "inference_seconds_accumulated": float(
                        progress.get("inference_seconds_accumulated", 0.0)
                    )
                    + (time.perf_counter() - batch_started),
                }
            )
            _atomic_write_json(progress_path, progress)
            if batch_number == 1 or batch_number % 10 == 0 or processed == len(paths):
                print(
                    f"embedding progress {processed}/{len(paths)} "
                    f"({processed / len(paths):.1%}), valid={valid_rows}, failed={failed_rows}",
                    flush=True,
                )

        valid_mask = row_status == 1
        valid_matrix = np.asarray(embeddings[valid_mask], dtype=np.float32)
        all_valid_finite = bool(np.isfinite(valid_matrix).all())
        norms = np.linalg.norm(valid_matrix, axis=1) if len(valid_matrix) else np.array([])
        total_inference = float(progress.get("inference_seconds_accumulated", 0.0))
        valid_rows = int(np.count_nonzero(valid_mask))
        failed_rows = int(np.count_nonzero(row_status == 2))
        status = "completed" if failed_rows == 0 else "completed_with_item_failures"
        metrics: dict[str, object] = {
            "schema_version": "m2.4-fashionclip-full-v1",
            "status": status,
            "boundary": "full static item embedding asset; no retrieval or ranking claim",
            "contract": contract,
            "contract_sha256": contract_sha256,
            "embeddings": {
                "rows": len(paths),
                "valid_rows": valid_rows,
                "failed_rows": failed_rows,
                "dimensions": dimensions,
                "storage_dtype": "float16",
                "all_valid_finite": all_valid_finite,
                "norm_min": float(norms.min()) if len(norms) else None,
                "norm_mean": float(norms.mean()) if len(norms) else None,
                "norm_max": float(norms.max()) if len(norms) else None,
                "artifact": str(embeddings_path),
                "artifact_bytes": embeddings_path.stat().st_size,
                "artifact_sha256": _file_sha256(embeddings_path),
                "row_status_artifact": str(row_status_path),
                "row_status_sha256": _file_sha256(row_status_path),
                "items_artifact": str(items_path),
                "items_sha256": _file_sha256(items_path),
                "item_failures_artifact": str(failures_path) if failures_path.exists() else None,
            },
            "runtime": {
                "model_load_seconds": model_load_seconds,
                "inference_seconds": total_inference,
                "images_per_second": valid_rows / total_inference if total_inference else 0.0,
                "cuda_peak_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
                ),
                "invocation_seconds": time.perf_counter() - invocation_started,
                "resumed": resume,
            },
        }
        _atomic_write_json(output_dir / "metrics.json", metrics)
        progress.update({"status": status, "next_row": len(paths)})
        _atomic_write_json(progress_path, progress)
        report_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(report_dir / "metrics.json", metrics)
        report_text = _render_full_embedding_report(metrics)
        (output_dir / "M2_4_FULL_EMBEDDINGS.md").write_text(report_text, encoding="utf-8")
        (report_dir / "M2_4_FULL_EMBEDDINGS.md").write_text(report_text, encoding="utf-8")
        return metrics
    except Exception as error:
        failure = {
            "schema_version": "m2.4-fashionclip-full-failure-v1",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "resume_requested": resume,
            "elapsed_seconds": time.perf_counter() - invocation_started,
        }
        failure_path = output_dir / f"failure-{time.time_ns()}.json"
        _atomic_write_json(failure_path, failure)
        raise
