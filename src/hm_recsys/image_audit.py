from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable
from zipfile import ZipFile

import duckdb
import pandas as pd
from PIL import Image

from .kaggle_data import COMPETITION

_IMAGE_PATTERN = re.compile(
    r"^images/(?P<prefix>[0-9]{3})/(?P<article_id>[0-9]{10})[.](?:jpg|jpeg)$",
    re.IGNORECASE,
)
_DEFAULT_CUTOFFS = ("2020-07-22", "2020-08-19", "2020-09-16")


@dataclass(frozen=True)
class ImageEntry:
    name: str
    article_id: str
    size_bytes: int
    crc32: int | None


@dataclass(frozen=True)
class ImageSource:
    kind: str
    path: Path


def _normalise_article_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit() or len(text) > 10:
        raise ValueError(f"invalid article_id: {value!r}")
    return text.zfill(10)


def _resolve_source(
    raw_dir: Path, source: str, archive_path: Path | None
) -> ImageSource:
    images_dir = raw_dir / "images"
    archive = archive_path or raw_dir / f"{COMPETITION}.zip"
    if source == "directory":
        if not images_dir.is_dir():
            raise FileNotFoundError(f"image directory not found: {images_dir}")
        return ImageSource("directory", images_dir.resolve())
    if source == "archive":
        if not archive.is_file():
            raise FileNotFoundError(f"competition archive not found: {archive}")
        return ImageSource("archive", archive.resolve())
    if source != "auto":
        raise ValueError(f"unknown image source: {source}")
    if images_dir.is_dir():
        return ImageSource("directory", images_dir.resolve())
    if archive.is_file():
        return ImageSource("archive", archive.resolve())
    raise FileNotFoundError(
        f"no image source found under {raw_dir}; expected images/ or {archive.name}"
    )


def _parse_image_name(name: str) -> tuple[str | None, str | None]:
    normalised = PurePosixPath(name.replace(chr(92), "/")).as_posix()
    match = _IMAGE_PATTERN.fullmatch(normalised)
    if match is None:
        return None, "path_pattern"
    article_id = match.group("article_id")
    if match.group("prefix") != article_id[:3]:
        return None, "prefix_mismatch"
    return article_id, None


def _collect_archive_entries(
    archive: Path,
) -> tuple[list[ImageEntry], Counter[str], list[str]]:
    entries: list[ImageEntry] = []
    invalid = Counter()
    invalid_examples: list[str] = []
    with ZipFile(archive) as zipped:
        for info in zipped.infolist():
            if info.is_dir() or not info.filename.replace(chr(92), "/").startswith("images/"):
                continue
            article_id, reason = _parse_image_name(info.filename)
            if reason is not None:
                invalid[reason] += 1
                if len(invalid_examples) < 20:
                    invalid_examples.append(info.filename)
                continue
            entries.append(
                ImageEntry(
                    name=info.filename.replace(chr(92), "/"),
                    article_id=article_id,
                    size_bytes=int(info.file_size),
                    crc32=int(info.CRC),
                )
            )
    return entries, invalid, invalid_examples


def _collect_directory_entries(
    images_dir: Path,
) -> tuple[list[ImageEntry], Counter[str], list[str]]:
    entries: list[ImageEntry] = []
    invalid = Counter()
    invalid_examples: list[str] = []
    raw_dir = images_dir.parent
    for path in images_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.relative_to(raw_dir).as_posix()
        article_id, reason = _parse_image_name(name)
        if reason is not None:
            invalid[reason] += 1
            if len(invalid_examples) < 20:
                invalid_examples.append(name)
            continue
        entries.append(
            ImageEntry(
                name=name,
                article_id=article_id,
                size_bytes=path.stat().st_size,
                crc32=None,
            )
        )
    return entries, invalid, invalid_examples


def _manifest_sha256(entries: Iterable[ImageEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda row: row.name):
        digest.update(
            (
                f"{entry.name}|{entry.article_id}|{entry.size_bytes}|"
                f"{entry.crc32 if entry.crc32 is not None else ''}" + chr(10)
            ).encode("utf-8")
        )
    return digest.hexdigest()


def _sample_entries(entries: list[ImageEntry], limit: int) -> list[ImageEntry]:
    if limit < 0:
        raise ValueError("verify_sample must be non-negative")
    if limit == 0 or not entries:
        return []
    return sorted(
        entries,
        key=lambda entry: hashlib.sha256(entry.name.encode("utf-8")).digest(),
    )[: min(limit, len(entries))]


def _difference_hash(image: Image.Image) -> str:
    grayscale = image.convert("L").resize((9, 8))
    if hasattr(grayscale, "get_flattened_data"):
        pixels = list(grayscale.get_flattened_data())
    else:
        pixels = list(grayscale.getdata())
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return f"{bits:016x}"


def _decode_image(stream: BinaryIO) -> tuple[str, tuple[int, int], str]:
    payload = stream.read()
    with Image.open(BytesIO(payload)) as image:
        image.load()
        return image.mode, image.size, _difference_hash(image)


def _verify_sample(
    source: ImageSource, entries: list[ImageEntry], limit: int
) -> dict[str, object]:
    sampled = _sample_entries(entries, limit)
    corrupt: list[dict[str, str]] = []
    modes: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    dhashes: defaultdict[str, list[str]] = defaultdict(list)

    if source.kind == "archive":
        with ZipFile(source.path) as zipped:
            for entry in sampled:
                try:
                    with zipped.open(entry.name) as stream:
                        mode, size, image_hash = _decode_image(stream)
                except Exception as error:
                    corrupt.append(
                        {
                            "name": entry.name,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                modes[mode] += 1
                sizes[f"{size[0]}x{size[1]}"] += 1
                dhashes[image_hash].append(entry.name)
    else:
        raw_dir = source.path.parent
        for entry in sampled:
            try:
                with (raw_dir / PurePosixPath(entry.name)).open("rb") as stream:
                    mode, size, image_hash = _decode_image(stream)
            except Exception as error:
                corrupt.append(
                    {
                        "name": entry.name,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            modes[mode] += 1
            sizes[f"{size[0]}x{size[1]}"] += 1
            dhashes[image_hash].append(entry.name)

    hash_collision_groups = [
        names for names in dhashes.values() if len(names) > 1
    ]
    return {
        "selection": "smallest_sha256(filename)",
        "requested": limit,
        "sampled": len(sampled),
        "decoded": len(sampled) - len(corrupt),
        "corrupt_count": len(corrupt),
        "corrupt_examples": corrupt[:20],
        "mode_counts": dict(sorted(modes.items())),
        "top_size_counts": dict(sizes.most_common(20)),
        "sample_dhash_collision_groups": len(hash_collision_groups),
        "sample_dhash_collision_files": sum(
            len(group) for group in hash_collision_groups
        ),
        "sample_dhash_collision_examples": hash_collision_groups[:10],
        "boundary": (
            "Decode and dHash-collision results cover only the deterministic "
            "sample. A dHash collision is a review candidate, not proof of "
            "near-duplicate content; manifest and mapping cover the full source."
        ),
    }


def _stream_sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _verify_exact_duplicates(
    source: ImageSource,
    candidate_groups: list[list[str]],
) -> dict[str, object]:
    candidate_names = sorted(
        {name for group in candidate_groups for name in group}
    )
    digests: defaultdict[str, list[str]] = defaultdict(list)
    if source.kind == "archive":
        with ZipFile(source.path) as zipped:
            for name in candidate_names:
                with zipped.open(name) as stream:
                    digests[_stream_sha256(stream)].append(name)
        status = "full_crc_size_candidates_verified"
    else:
        raw_dir = source.path.parent
        for name in candidate_names:
            with (raw_dir / PurePosixPath(name)).open("rb") as stream:
                digests[_stream_sha256(stream)].append(name)
        status = (
            "no_candidates_from_directory"
            if not candidate_names
            else "provided_candidates_verified"
        )
    exact_groups = [names for names in digests.values() if len(names) > 1]
    return {
        "status": status,
        "candidate_groups": len(candidate_groups),
        "candidate_files": len(candidate_names),
        "hashed_files": len(candidate_names),
        "exact_sha256_groups": len(exact_groups),
        "exact_sha256_files": sum(len(names) for names in exact_groups),
        "exact_sha256_examples": exact_groups[:10],
    }

def _article_coverage(articles_path: Path, image_ids: set[str]) -> dict[str, object]:
    articles = pd.read_csv(
        articles_path,
        usecols=["article_id"],
        dtype={"article_id": "string"},
    )
    article_ids = {
        _normalise_article_id(value)
        for value in articles["article_id"].dropna().tolist()
    }
    covered = article_ids & image_ids
    image_only = image_ids - article_ids
    return {
        "articles_rows": int(len(articles)),
        "unique_article_ids": len(article_ids),
        "covered_article_ids": len(covered),
        "coverage_rate": len(covered) / len(article_ids) if article_ids else 0.0,
        "missing_image_article_ids": len(article_ids - image_ids),
        "image_ids_not_in_articles": len(image_only),
        "missing_examples": sorted(article_ids - image_ids)[:20],
        "image_only_examples": sorted(image_only)[:20],
    }


def _transaction_window_coverage(
    transactions_path: Path,
    image_ids: set[str],
    cutoffs: tuple[str, ...],
) -> list[dict[str, object]]:
    parsed = [date.fromisoformat(value) for value in cutoffs]
    if len(set(parsed)) != len(parsed):
        raise ValueError("cutoffs must be unique")
    image_frame = pd.DataFrame({"article_id": sorted(image_ids)})
    windows = pd.DataFrame(
        {
            "cutoff": parsed,
            "end_date": [value + timedelta(days=7) for value in parsed],
        }
    )
    connection = duckdb.connect()
    try:
        connection.register("image_items", image_frame)
        connection.register("audit_windows", windows)
        escaped = str(transactions_path.resolve()).replace("'", "''")
        query = f"""
        WITH tx AS (
            SELECT
                CAST(t_dat AS DATE) AS t_dat,
                customer_id,
                lpad(article_id, 10, '0') AS article_id
            FROM read_csv_auto('{escaped}', header = true, all_varchar = true)
        ),
        truth AS (
            SELECT DISTINCT
                w.cutoff,
                t.customer_id,
                t.article_id
            FROM tx t
            JOIN audit_windows w
              ON t.t_dat >= w.cutoff
             AND t.t_dat < w.end_date
        ),
        history_items AS (
            SELECT DISTINCT
                w.cutoff,
                t.article_id
            FROM tx t
            JOIN audit_windows w
              ON t.t_dat < w.cutoff
        ),
        labelled AS (
            SELECT
                t.cutoff,
                t.customer_id,
                t.article_id,
                i.article_id IS NOT NULL AS has_image,
                h.article_id IS NOT NULL AS is_warm
            FROM truth t
            LEFT JOIN image_items i USING(article_id)
            LEFT JOIN history_items h
              ON t.cutoff = h.cutoff
             AND t.article_id = h.article_id
        )
        SELECT
            cutoff,
            count(*) AS truth_pairs,
            count(DISTINCT customer_id) AS truth_users,
            count(DISTINCT article_id) AS truth_items,
            count(*) FILTER (WHERE has_image) AS image_truth_pairs,
            count(DISTINCT article_id) FILTER (WHERE has_image) AS image_truth_items,
            count(*) FILTER (WHERE is_warm) AS warm_truth_pairs,
            count(*) FILTER (WHERE NOT is_warm) AS cold_truth_pairs,
            count(*) FILTER (WHERE is_warm AND has_image) AS image_warm_truth_pairs,
            count(*) FILTER (WHERE NOT is_warm AND has_image) AS image_cold_truth_pairs
        FROM labelled
        GROUP BY cutoff
        ORDER BY cutoff
        """
        frame = connection.execute(query).fetchdf()
    finally:
        connection.close()

    rows: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        truth_pairs = int(row["truth_pairs"])
        warm_pairs = int(row["warm_truth_pairs"])
        cold_pairs = int(row["cold_truth_pairs"])
        payload = {
            key: str(value)[:10] if key == "cutoff" else int(value)
            for key, value in row.items()
        }
        payload["image_truth_pair_coverage"] = (
            int(row["image_truth_pairs"]) / truth_pairs if truth_pairs else 0.0
        )
        payload["image_warm_pair_coverage"] = (
            int(row["image_warm_truth_pairs"]) / warm_pairs if warm_pairs else 0.0
        )
        payload["image_cold_pair_coverage"] = (
            int(row["image_cold_truth_pairs"]) / cold_pairs if cold_pairs else 0.0
        )
        rows.append(payload)
    return rows


def _render_report(metrics: dict[str, object]) -> str:
    source = metrics["source"]
    mapping = metrics["mapping"]
    article = metrics["article_coverage"]
    sample = metrics["decode_sample"]
    lines = [
        "# M2.3 H&M 图片数据门禁报告",
        "",
        "## 结论边界",
        "",
        "- 当前 catalog protocol：optimistic_all_articles。",
        "- 完整 manifest 与 article mapping 已审计；图片解码和感知重复只覆盖确定性样本。",
        "- 图片相似不等于购买意图；本报告不包含 embedding、召回或 MAP 提升结论。",
        "",
        "## 数据源",
        "",
        f"- 类型：{source['kind']}",
        f"- 路径：{source['path']}",
        f"- 字节数：{source['size_bytes']}",
        f"- 完整 image files：{mapping['valid_image_files']}",
        f"- 唯一 image article IDs：{mapping['unique_image_article_ids']}",
        f"- manifest SHA256：{mapping['manifest_sha256']}",
        "",
        "## 映射与覆盖",
        "",
        f"- articles.csv 唯一商品：{article['unique_article_ids']}",
        f"- 有图片商品：{article['covered_article_ids']}",
        f"- article coverage：{article['coverage_rate']:.6f}",
        f"- articles 中缺图：{article['missing_image_article_ids']}",
        f"- 图片 ID 不在 articles.csv：{article['image_ids_not_in_articles']}",
        f"- 重复 article ID groups：{mapping['duplicate_article_id_groups']}",
        f"- 零字节图片：{mapping['zero_byte_files']}",
        f"- SHA256 精确重复 groups：{mapping['exact_duplicate_verification']['exact_sha256_groups']}",
        "",
        "## 确定性解码样本",
        "",
        f"- 请求/实际：{sample['requested']} / {sample['sampled']}",
        f"- 成功解码：{sample['decoded']}",
        f"- 损坏：{sample['corrupt_count']}",
        f"- 样本 dHash collision groups：{sample['sample_dhash_collision_groups']}",
        "",
        "## 验证周 truth 图片覆盖",
        "",
        "| cutoff | truth pairs | image pairs | coverage | warm coverage | cold coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["transaction_windows"]:
        lines.append(
            "| {cutoff} | {truth_pairs} | {image_truth_pairs} | "
            "{image_truth_pair_coverage:.6f} | {image_warm_pair_coverage:.6f} | "
            "{image_cold_pair_coverage:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 下一门槛",
            "",
            "- 若 mapping/truth coverage 足够且样本解码无系统性失败，才安装并冻结视觉模型。",
            "- embedding 必须绑定模型 revision、预处理、article 顺序和输入 manifest。",
            "- image recall 先测 standalone、union、fixed-100、expanded-200/300 与 Oracle。",
            "",
        ]
    )
    return chr(10).join(lines)


def run_image_audit(
    *,
    raw_dir: Path,
    output_dir: Path,
    source: str = "auto",
    archive_path: Path | None = None,
    verify_sample: int = 1000,
    cutoffs: tuple[str, ...] = _DEFAULT_CUTOFFS,
) -> dict[str, object]:
    started = time.perf_counter()
    raw_dir = raw_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "M2_3_IMAGE_AUDIT.md"
    if metrics_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {output_dir}")

    articles_path = raw_dir / "articles.csv"
    transactions_path = raw_dir / "transactions_train.csv"
    for required in (articles_path, transactions_path):
        if not required.is_file():
            raise FileNotFoundError(f"required tabular file not found: {required}")

    resolved_source = _resolve_source(raw_dir, source, archive_path)
    if resolved_source.kind == "archive":
        entries, invalid, invalid_examples = _collect_archive_entries(
            resolved_source.path
        )
    else:
        entries, invalid, invalid_examples = _collect_directory_entries(
            resolved_source.path
        )
    if not entries:
        raise ValueError(f"no valid H&M images found in {resolved_source.path}")

    article_to_files: defaultdict[str, list[str]] = defaultdict(list)
    for entry in entries:
        article_to_files[entry.article_id].append(entry.name)
    image_ids = set(article_to_files)
    duplicate_article_groups = {
        article_id: names
        for article_id, names in article_to_files.items()
        if len(names) > 1
    }
    crc_groups: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    for entry in entries:
        if entry.crc32 is not None:
            crc_groups[(entry.size_bytes, entry.crc32)].append(entry.name)
    potential_exact_groups = [
        names for names in crc_groups.values() if len(names) > 1
    ]
    exact_duplicates = _verify_exact_duplicates(
        resolved_source, potential_exact_groups
    )

    metrics: dict[str, object] = {
        "schema_version": "m2.3-image-audit-v2",
        "status": "completed",
        "protocol": {
            "catalog": "optimistic_all_articles",
            "transaction_aggregates": "cutoff_safe",
            "image_availability_boundary": (
                "articles.csv has no effective image availability timestamp; "
                "all mapped images are treated as catalog-static under the "
                "optimistic protocol."
            ),
        },
        "source": {
            **asdict(resolved_source),
            "path": str(resolved_source.path),
            "size_bytes": resolved_source.path.stat().st_size
            if resolved_source.path.is_file()
            else sum(entry.size_bytes for entry in entries),
        },
        "mapping": {
            "valid_image_files": len(entries),
            "unique_image_article_ids": len(image_ids),
            "manifest_sha256": _manifest_sha256(entries),
            "invalid_path_counts": dict(sorted(invalid.items())),
            "invalid_path_examples": invalid_examples,
            "zero_byte_files": sum(entry.size_bytes == 0 for entry in entries),
            "duplicate_article_id_groups": len(duplicate_article_groups),
            "duplicate_article_id_files": sum(
                len(names) for names in duplicate_article_groups.values()
            ),
            "duplicate_article_id_examples": list(duplicate_article_groups.items())[:10],
            "potential_exact_crc_size_groups": len(potential_exact_groups),
            "potential_exact_crc_size_files": sum(
                len(names) for names in potential_exact_groups
            ),
            "potential_exact_boundary": (
                "CRC32 plus size only selects candidates; exact duplicates "
                "are confirmed below with SHA256."
            ),
            "exact_duplicate_verification": exact_duplicates,
        },
        "article_coverage": _article_coverage(articles_path, image_ids),
        "decode_sample": _verify_sample(
            resolved_source, entries, verify_sample
        ),
        "transaction_windows": _transaction_window_coverage(
            transactions_path, image_ids, cutoffs
        ),
    }
    metrics["elapsed_seconds"] = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    report_path.write_text(_render_report(metrics), encoding="utf-8")
    return metrics
