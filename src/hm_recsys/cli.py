from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .audit import run_audit
from .baseline import build_ground_truth, generate_baseline
from .data import load_temporal_split
from .kaggle_data import (
    COMPETITION,
    download_competition_data,
    extract_image_archive,
    get_data_status,
)
from .image_audit import run_image_audit
from .image_embedding import (
    FASHION_CLIP_MODEL_ID,
    FASHION_CLIP_REVISION,
    run_embedding_smoke,
    run_full_embeddings,
)
from .m25 import build_exact_image_neighbors
from .m25_retrieval import run_m25_retrieval
from .metrics import hit_rate_at_k, mapk, oracle_mapk, recall_at_k
from .m1 import FUSION_PROFILES, M1Config, evaluate_candidate_artifact, run_m1
from .m15 import run_m15


def _validate(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    transactions_path = Path(args.transactions).expanduser().resolve()
    split = load_temporal_split(
        transactions_path,
        cutoff=args.cutoff,
        history_weeks=args.history_weeks,
        sample_rate=args.sample_rate,
        chunksize=args.chunksize,
    )
    truth = build_ground_truth(split.validation)
    users = sorted(truth)
    output = generate_baseline(
        split.history,
        users,
        split.cutoff,
        candidate_k=args.candidate_k,
        metric_k=args.k,
        popular_window_days=args.popular_window_days,
    )

    metrics = {
        "transactions_file": transactions_path.name,
        "transactions_size_bytes": transactions_path.stat().st_size,
        "cutoff": split.cutoff.strftime("%Y-%m-%d"),
        "data_end": split.data_end.strftime("%Y-%m-%d"),
        "sample_rate": args.sample_rate,
        "history_weeks": args.history_weeks,
        "chunksize": args.chunksize,
        "candidate_k": args.candidate_k,
        "metric_k": args.k,
        "popular_window_days": args.popular_window_days,
        "history_rows": len(split.history),
        "validation_rows": len(split.validation),
        "active_validation_users": len(users),
        f"candidate_recall@{args.candidate_k}": recall_at_k(
            truth, output.candidates, args.candidate_k
        ),
        f"candidate_hit_rate@{args.candidate_k}": hit_rate_at_k(
            truth, output.candidates, args.candidate_k
        ),
        f"oracle_map@{args.k}": oracle_mapk(
            truth, output.candidates, args.candidate_k, args.k
        ),
        f"map@{args.k}": mapk(truth, output.predictions, args.k),
    }
    metrics["elapsed_seconds"] = time.perf_counter() - started_at
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        output.candidate_rows.to_csv(
            output_dir / "validation_candidates.csv.gz", index=False, compression="gzip"
        )
    return 0


def _data_status(args: argparse.Namespace) -> int:
    status = get_data_status(args.raw_dir, args.env_file)
    payload = {
        "authenticated": status.authenticated,
        "auth_source": status.auth_source,
        "raw_dir": str(status.raw_dir),
        "transactions_path": str(status.transactions_path),
        "mvp_data_ready": status.ready,
        "tabular_data_ready": status.tabular_ready,
        "tabular_files": status.tabular_files,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status.ready else 1


def _data_download(args: argparse.Namespace) -> int:
    try:
        status = download_competition_data(
            raw_dir=args.raw_dir,
            env_file=args.env_file,
            force=args.force,
            extract=not args.no_extract,
            all_files=args.all_files,
            tabular=args.tabular,
            download_attempts=args.download_attempts,
        )
    except (RuntimeError, FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
        print(f"data download failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "raw_dir": str(status.raw_dir),
                "transactions_path": str(status.transactions_path),
                "mvp_data_ready": status.ready,
                "tabular_data_ready": status.tabular_ready,
                "tabular_files": status.tabular_files,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _data_extract_images(args: argparse.Namespace) -> int:
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    archive = (
        Path(args.archive).expanduser().resolve()
        if args.archive
        else raw_dir / f"{COMPETITION}.zip"
    )
    try:
        result = extract_image_archive(archive, raw_dir)
    except (ValueError, FileNotFoundError, OSError) as error:
        print(f"image extraction failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "completed",
                "archive": str(archive),
                "raw_dir": str(raw_dir),
                **result,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0

def _audit_data(args: argparse.Namespace) -> int:
    audit = run_audit(
        raw_dir=Path(args.raw_dir),
        work_dir=Path(args.work_dir),
        output_path=Path(args.output),
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "elapsed_seconds": audit["scope"]["elapsed_seconds"],
                "image_audit_excluded": audit["scope"]["image_audit_excluded"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _audit_images(args: argparse.Namespace) -> int:
    kwargs: dict[str, object] = {}
    if args.cutoffs:
        kwargs["cutoffs"] = tuple(args.cutoffs)
    try:
        audit = run_image_audit(
            raw_dir=Path(args.raw_dir),
            output_dir=Path(args.output_dir),
            source=args.source,
            archive_path=Path(args.archive) if args.archive else None,
            verify_sample=args.verify_sample,
            **kwargs,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"image audit failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": audit["status"],
                "source": audit["source"],
                "mapping": audit["mapping"],
                "article_coverage": audit["article_coverage"],
                "decode_sample": audit["decode_sample"],
                "transaction_windows": audit["transaction_windows"],
                "elapsed_seconds": audit["elapsed_seconds"],
                "report": str(
                    (Path(args.output_dir) / "M2_3_IMAGE_AUDIT.md").resolve()
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0

def _image_embedding_smoke(args: argparse.Namespace) -> int:
    try:
        result = run_embedding_smoke(
            images_dir=Path(args.images_dir),
            audit_metrics_path=Path(args.audit_metrics),
            output_dir=Path(args.output_dir),
            cache_dir=Path(args.cache_dir),
            limit=args.limit,
            batch_size=args.batch_size,
            device=args.device,
            model_id=args.model_id,
            revision=args.revision,
        )
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        OSError,
    ) as error:
        print(f"embedding smoke failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "boundary": result["boundary"],
                "audit_contract": result["audit_contract"],
                "model": result["model"],
                "environment": result["environment"],
                "embeddings": result["embeddings"],
                "runtime": result["runtime"],
                "report": str(
                    (Path(args.output_dir) / "M2_4_EMBEDDING_SMOKE.md").resolve()
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0

def _image_embedding_full(args: argparse.Namespace) -> int:
    try:
        result = run_full_embeddings(
            images_dir=Path(args.images_dir),
            audit_metrics_path=Path(args.audit_metrics),
            output_dir=Path(args.output_dir),
            report_dir=Path(args.report_dir),
            cache_dir=Path(args.cache_dir),
            batch_size=args.batch_size,
            device=args.device,
            resume=args.resume,
            model_id=args.model_id,
            revision=args.revision,
        )
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        OSError,
    ) as error:
        print(f"full embedding failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "boundary": result["boundary"],
                "contract_sha256": result["contract_sha256"],
                "embeddings": result["embeddings"],
                "runtime": result["runtime"],
                "report": str(
                    (Path(args.report_dir) / "M2_4_FULL_EMBEDDINGS.md").resolve()
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0

def _image_neighbors_exact(args: argparse.Namespace) -> int:
    try:
        result = build_exact_image_neighbors(
            embedding_metrics_path=Path(args.embedding_metrics),
            output_dir=Path(args.output_dir),
            report_dir=Path(args.report_dir),
            top_k=args.top_k,
            tie_buffer=args.tie_buffer,
            query_batch_size=args.query_batch_size,
            query_limit=args.query_limit,
            device=args.device,
            resume=args.resume,
        )
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        OSError,
    ) as error:
        print(f"exact image neighbors failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "boundary": result["boundary"],
                "contract_sha256": result["contract_sha256"],
                "query_rows": result["query_rows"],
                "corpus_rows": result["corpus_rows"],
                "top_k": result["top_k"],
                "audit": result["audit"],
                "runtime": result["runtime"],
                "report": str(
                    (Path(args.report_dir) / "M2_5_EXACT_NEIGHBORS.md").resolve()
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0

def _image_retrieval_dev(args: argparse.Namespace) -> int:
    try:
        result = run_m25_retrieval(
            neighbor_metrics_path=Path(args.neighbor_metrics),
            transactions_path=Path(args.transactions),
            reference_metrics_path=Path(args.reference_metrics),
            windows=[
                (
                    "2020-07-22",
                    Path(args.candidate_20200722),
                    Path(args.manifest_20200722),
                ),
                (
                    "2020-08-19",
                    Path(args.candidate_20200819),
                    Path(args.manifest_20200819),
                ),
            ],
            output_dir=Path(args.output_dir),
            report_dir=Path(args.report_dir),
            history_weeks=args.history_weeks,
            seed_k=args.seed_k,
            image_candidate_k=args.image_candidate_k,
            rrf_constant=args.rrf_constant,
            image_weight=args.image_weight,
        )
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        OSError,
    ) as error:
        print(f"image retrieval development failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "boundary": result["boundary"],
                "development_gate": result["development_gate"],
                "elapsed_seconds": result["elapsed_seconds"],
                "metrics": str((Path(args.report_dir) / "metrics.json").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0

def _m1_validate(args: argparse.Namespace) -> int:
    config = M1Config(
        cutoff=args.cutoff,
        sample_rate=args.sample_rate,
        history_weeks=args.history_weeks,
        popularity_days=args.popularity_days,
        covisit_days=args.covisit_days,
        candidate_k=args.candidate_k,
        source_k=args.source_k,
        covisit_neighbor_k=args.covisit_neighbor_k,
        max_user_day_items=args.max_user_day_items,
        min_covisit_count=args.min_covisit_count,
        rrf_constant=args.rrf_constant,
        content_seed_k=args.content_seed_k,
        content_type_pool_k=args.content_type_pool_k,
        content_garment_pool_k=args.content_garment_pool_k,
        fusion_profile=args.fusion_profile,
        evaluation_mode=args.evaluation_mode,
        catalog_protocol=args.catalog_protocol,
    )
    try:
        result = run_m1(
            raw_dir=Path(args.raw_dir),
            work_dir=Path(args.work_dir),
            output_dir=Path(args.output_dir),
            artifact_dir=Path(args.artifact_dir),
            config=config,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"M1 validation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "cutoff": result["cutoff"],
                "sampled_validation_users": result["population"]["sampled_validation_users"],
                "final_metrics": result["final_metrics"],
                "elapsed_seconds": result["elapsed_seconds"],
                "report": str((Path(args.output_dir) / "M1_REPORT.md").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _m1_evaluate_artifact(args: argparse.Namespace) -> int:
    try:
        result = evaluate_candidate_artifact(
            raw_dir=Path(args.raw_dir),
            work_dir=Path(args.work_dir),
            output_dir=Path(args.output_dir),
            candidate_path=Path(args.candidate_path),
            source_metrics_path=Path(args.source_metrics),
            cutoff=args.cutoff,
            expected_profile=args.fusion_profile,
            candidate_k=args.candidate_k,
            expected_users=args.expected_users,
            expected_rows=args.expected_rows,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"M1 artifact evaluation failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "cutoff": result["cutoff"],
                "catalog_protocol": result["catalog_protocol"],
                "truth_pairs": result["truth_pairs"],
                "metrics": result["metrics"],
                "report": str(
                    (Path(args.output_dir) / "M1_ARTIFACT_EVALUATION.md").resolve()
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _m15_run(args: argparse.Namespace) -> int:
    config = M1Config(
        cutoff=args.cutoff,
        sample_rate=args.sample_rate,
        history_weeks=args.history_weeks,
        popularity_days=args.popularity_days,
        covisit_days=args.covisit_days,
        candidate_k=args.candidate_k,
        source_k=args.source_k,
        covisit_neighbor_k=args.covisit_neighbor_k,
        max_user_day_items=args.max_user_day_items,
        min_covisit_count=args.min_covisit_count,
        rrf_constant=args.rrf_constant,
        content_seed_k=args.content_seed_k,
        content_type_pool_k=args.content_type_pool_k,
        content_garment_pool_k=args.content_garment_pool_k,
        fusion_profile=args.fusion_profile,
        evaluation_mode="final",
        catalog_protocol=args.catalog_protocol,
    )
    try:
        result = run_m15(
            raw_dir=Path(args.raw_dir),
            work_dir=Path(args.work_dir),
            output_dir=Path(args.output_dir),
            artifact_dir=Path(args.artifact_dir),
            cache_dir=Path(args.cache_dir),
            config=config,
            cache_mode=args.cache_mode,
            parity_reference=Path(args.parity_reference) if args.parity_reference else None,
            baseline_metrics=Path(args.baseline_metrics) if args.baseline_metrics else None,
            run_id=args.run_id,
        )
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"M1.5 run failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_id": result["run_id"],
                "cutoff": result["cutoff"],
                "catalog_protocol": result["catalog_protocol"],
                "validation_users": result["validation_users"],
                "candidate_rows": result["candidate_rows"],
                "cache": result["cache"],
                "parity": result["parity"],
                "elapsed_seconds": result["elapsed_seconds"],
                "report": str((Path(args.output_dir) / "M1_5_REPORT.md").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hm-recsys")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate", help="Run a leakage-safe last-week offline validation"
    )
    validate.add_argument("--transactions", required=True)
    validate.add_argument("--cutoff", help="Validation start date, YYYY-MM-DD")
    validate.add_argument("--history-weeks", type=int, default=12)
    validate.add_argument("--sample-rate", type=float, default=0.01)
    validate.add_argument("--chunksize", type=int, default=2_000_000)
    validate.add_argument("--candidate-k", type=int, default=100)
    validate.add_argument("--k", type=int, default=12)
    validate.add_argument("--popular-window-days", type=int, default=28)
    validate.add_argument("--output-dir")
    validate.set_defaults(func=_validate)

    data = subparsers.add_parser("data", help="Inspect or download Kaggle data")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)

    data_status = data_subparsers.add_parser(
        "status", help="Check credentials and MVP data without exposing secrets"
    )
    data_status.add_argument("--raw-dir", default="data/raw")
    data_status.add_argument("--env-file", default=".env")
    data_status.set_defaults(func=_data_status)

    data_download = data_subparsers.add_parser(
        "download", help="Download and extract the H&M competition archive"
    )
    data_download.add_argument("--raw-dir", default="data/raw")
    data_download.add_argument("--env-file", default=".env")
    data_download.add_argument("--force", action="store_true")
    data_download.add_argument("--no-extract", action="store_true")
    data_download.add_argument("--download-attempts", type=int, default=5)
    data_download.add_argument(
        "--all-files",
        action="store_true",
        help="Download the complete competition bundle, including images",
    )
    data_download.add_argument(
        "--tabular",
        action="store_true",
        help="Download all CSV tables but skip the image archive",
    )
    data_download.set_defaults(func=_data_download)

    data_extract_images = data_subparsers.add_parser(
        "extract-images",
        help="Safely extract only images/ from the complete competition archive",
    )
    data_extract_images.add_argument("--raw-dir", default="data/raw")
    data_extract_images.add_argument("--archive")
    data_extract_images.set_defaults(func=_data_extract_images)
    audit = subparsers.add_parser("audit", help="Run reproducible data audits")
    audit_subparsers = audit.add_subparsers(dest="audit_command", required=True)
    audit_data = audit_subparsers.add_parser(
        "data", help="Audit all H&M CSV tables while excluding images"
    )
    audit_data.add_argument("--raw-dir", default="data/raw")
    audit_data.add_argument("--work-dir", default="data/interim/audit")
    audit_data.add_argument(
        "--output", default="reports/m0_5/HM_DATA_AUDIT.md"
    )
    audit_data.set_defaults(func=_audit_data)

    audit_images = audit_subparsers.add_parser(
        "images", help="Audit H&M image mapping, coverage, and a decode sample"
    )
    audit_images.add_argument("--raw-dir", default="data/raw")
    audit_images.add_argument("--output-dir", required=True)
    audit_images.add_argument(
        "--source", choices=("auto", "archive", "directory"), default="auto"
    )
    audit_images.add_argument("--archive")
    audit_images.add_argument("--verify-sample", type=int, default=1000)
    audit_images.add_argument(
        "--cutoff",
        dest="cutoffs",
        action="append",
        help="Seven-day truth window start; repeat for multiple windows",
    )
    audit_images.set_defaults(func=_audit_images)
    image = subparsers.add_parser(
        "image", help="Run image embedding and retrieval experiments"
    )
    image_subparsers = image.add_subparsers(
        dest="image_command", required=True
    )
    embedding_smoke = image_subparsers.add_parser(
        "embedding-smoke",
        help="Run a bounded FashionCLIP image-embedding mechanics smoke",
    )
    embedding_smoke.add_argument("--images-dir", default="data/raw/images")
    embedding_smoke.add_argument("--audit-metrics", required=True)
    embedding_smoke.add_argument("--output-dir", required=True)
    embedding_smoke.add_argument(
        "--cache-dir", default="artifacts/m2_4/huggingface"
    )
    embedding_smoke.add_argument("--limit", type=int, default=128)
    embedding_smoke.add_argument("--batch-size", type=int, default=16)
    embedding_smoke.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda"
    )
    embedding_smoke.add_argument(
        "--model-id", default=FASHION_CLIP_MODEL_ID
    )
    embedding_smoke.add_argument(
        "--revision", default=FASHION_CLIP_REVISION
    )
    embedding_smoke.set_defaults(func=_image_embedding_smoke)
    embedding_full = image_subparsers.add_parser(
        "embedding-full",
        help="Build or resume the full FashionCLIP item-embedding artifact",
    )
    embedding_full.add_argument("--images-dir", default="data/raw/images")
    embedding_full.add_argument("--audit-metrics", required=True)
    embedding_full.add_argument("--output-dir", required=True)
    embedding_full.add_argument("--report-dir", required=True)
    embedding_full.add_argument(
        "--cache-dir", default="artifacts/m2_4/huggingface"
    )
    embedding_full.add_argument("--batch-size", type=int, default=64)
    embedding_full.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda"
    )
    embedding_full.add_argument("--resume", action="store_true")
    embedding_full.add_argument(
        "--model-id", default=FASHION_CLIP_MODEL_ID
    )
    embedding_full.add_argument(
        "--revision", default=FASHION_CLIP_REVISION
    )
    embedding_full.set_defaults(func=_image_embedding_full)
    neighbors_exact = image_subparsers.add_parser(
        "neighbors-exact",
        help="Build or resume exact cosine image neighbors",
    )
    neighbors_exact.add_argument("--embedding-metrics", required=True)
    neighbors_exact.add_argument("--output-dir", required=True)
    neighbors_exact.add_argument("--report-dir", required=True)
    neighbors_exact.add_argument("--top-k", type=int, default=100)
    neighbors_exact.add_argument("--tie-buffer", type=int, default=32)
    neighbors_exact.add_argument("--query-batch-size", type=int, default=512)
    neighbors_exact.add_argument("--query-limit", type=int)
    neighbors_exact.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda"
    )
    neighbors_exact.add_argument("--resume", action="store_true")
    neighbors_exact.set_defaults(func=_image_neighbors_exact)
    retrieval_dev = image_subparsers.add_parser(
        "retrieval-dev", help="Evaluate image retrieval on frozen dev-A/dev-B"
    )
    retrieval_dev.add_argument("--neighbor-metrics", required=True)
    retrieval_dev.add_argument("--transactions", required=True)
    retrieval_dev.add_argument("--reference-metrics", required=True)
    retrieval_dev.add_argument("--candidate-20200722", required=True)
    retrieval_dev.add_argument("--manifest-20200722", required=True)
    retrieval_dev.add_argument("--candidate-20200819", required=True)
    retrieval_dev.add_argument("--manifest-20200819", required=True)
    retrieval_dev.add_argument("--output-dir", required=True)
    retrieval_dev.add_argument("--report-dir", required=True)
    retrieval_dev.add_argument("--history-weeks", type=int, default=12)
    retrieval_dev.add_argument("--seed-k", type=int, default=5)
    retrieval_dev.add_argument("--image-candidate-k", type=int, default=300)
    retrieval_dev.add_argument("--rrf-constant", type=int, default=60)
    retrieval_dev.add_argument("--image-weight", type=float, default=1.0)
    retrieval_dev.set_defaults(func=_image_retrieval_dev)
    m1 = subparsers.add_parser("m1", help="Run M1 multi-source retrieval")
    m1_subparsers = m1.add_subparsers(dest="m1_command", required=True)
    m1_validate = m1_subparsers.add_parser(
        "validate", help="Build and evaluate leakage-controlled M1 candidates"
    )
    m1_validate.add_argument("--raw-dir", default="data/raw")
    m1_validate.add_argument("--work-dir", default="data/interim/audit")
    m1_validate.add_argument("--output-dir", required=True)
    m1_validate.add_argument("--artifact-dir", required=True)
    m1_validate.add_argument("--cutoff")
    m1_validate.add_argument("--sample-rate", type=float, default=0.01)
    m1_validate.add_argument("--history-weeks", type=int, default=12)
    m1_validate.add_argument("--popularity-days", type=int, default=28)
    m1_validate.add_argument("--covisit-days", type=int, default=28)
    m1_validate.add_argument("--candidate-k", type=int, default=100)
    m1_validate.add_argument("--source-k", type=int, default=100)
    m1_validate.add_argument("--covisit-neighbor-k", type=int, default=50)
    m1_validate.add_argument("--max-user-day-items", type=int, default=20)
    m1_validate.add_argument("--min-covisit-count", type=int, default=2)
    m1_validate.add_argument("--rrf-constant", type=int, default=60)
    m1_validate.add_argument("--content-seed-k", type=int, default=20)
    m1_validate.add_argument("--content-type-pool-k", type=int, default=50)
    m1_validate.add_argument("--content-garment-pool-k", type=int, default=30)
    m1_validate.add_argument(
        "--fusion-profile", choices=tuple(FUSION_PROFILES), default="equal"
    )
    m1_validate.add_argument(
        "--evaluation-mode", choices=("diagnostic", "final"), default="diagnostic"
    )
    m1_validate.add_argument(
        "--catalog-protocol",
        choices=("optimistic_all_articles", "strict"),
        default="optimistic_all_articles",
    )
    m1_validate.set_defaults(func=_m1_validate)

    m1_artifact = m1_subparsers.add_parser(
        "evaluate-artifact",
        help="Validate and re-evaluate an existing optimistic candidate artifact",
    )
    m1_artifact.add_argument("--raw-dir", default="data/raw")
    m1_artifact.add_argument("--work-dir", default="data/interim/audit")
    m1_artifact.add_argument("--output-dir", required=True)
    m1_artifact.add_argument("--candidate-path", required=True)
    m1_artifact.add_argument("--source-metrics", required=True)
    m1_artifact.add_argument("--cutoff", required=True)
    m1_artifact.add_argument(
        "--fusion-profile", choices=tuple(FUSION_PROFILES), required=True
    )
    m1_artifact.add_argument("--candidate-k", type=int, default=100)
    m1_artifact.add_argument("--expected-users", type=int)
    m1_artifact.add_argument("--expected-rows", type=int)
    m1_artifact.set_defaults(func=_m1_evaluate_artifact)
    m15 = subparsers.add_parser(
        "m1-5", help="Build auditable M1.5 source caches and feature artifacts"
    )
    m15_subparsers = m15.add_subparsers(dest="m15_command", required=True)
    m15_run = m15_subparsers.add_parser(
        "run", help="Build or require caches, then fuse and emit wide features"
    )
    m15_run.add_argument("--raw-dir", default="data/raw")
    m15_run.add_argument("--work-dir", default="data/interim/audit")
    m15_run.add_argument("--output-dir", required=True)
    m15_run.add_argument("--artifact-dir", required=True)
    m15_run.add_argument("--cache-dir", default="artifacts/m1_5/cache")
    m15_run.add_argument("--run-id")
    m15_run.add_argument("--cutoff")
    m15_run.add_argument("--sample-rate", type=float, default=0.01)
    m15_run.add_argument("--history-weeks", type=int, default=12)
    m15_run.add_argument("--popularity-days", type=int, default=28)
    m15_run.add_argument("--covisit-days", type=int, default=28)
    m15_run.add_argument("--candidate-k", type=int, default=100)
    m15_run.add_argument("--source-k", type=int, default=100)
    m15_run.add_argument("--covisit-neighbor-k", type=int, default=50)
    m15_run.add_argument("--max-user-day-items", type=int, default=20)
    m15_run.add_argument("--min-covisit-count", type=int, default=2)
    m15_run.add_argument("--rrf-constant", type=int, default=60)
    m15_run.add_argument("--content-seed-k", type=int, default=20)
    m15_run.add_argument("--content-type-pool-k", type=int, default=50)
    m15_run.add_argument("--content-garment-pool-k", type=int, default=30)
    m15_run.add_argument(
        "--fusion-profile", choices=tuple(FUSION_PROFILES), default="collaborative"
    )
    m15_run.add_argument(
        "--catalog-protocol",
        choices=("optimistic_all_articles", "strict"),
        default="optimistic_all_articles",
    )
    m15_run.add_argument("--cache-mode", choices=("auto", "require"), default="auto")
    m15_run.add_argument("--parity-reference")
    m15_run.add_argument("--baseline-metrics")
    m15_run.set_defaults(func=_m15_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
