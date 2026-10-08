from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from .m25_retrieval import _sha256
from .m33 import ROLLING_PROTOCOL


FINAL_CUTOFF = "2020-09-16"
M4_RUN_ID = "m4-v1-supervised-cold-representation"
EXPECTED_FASHIONCLIP = {
    "rows": 105_100,
    "shape": (105_100, 512),
    "dtype": "float16",
    "valid_rows": 105_100,
    "failed_rows": 0,
    "revision": "83cb9b65be402bbdb4d0e1b84bd53555028bfed8",
    "processor_use_fast": False,
    "embedding_sha256": "d297e93ffbc8b2f8d594d5577c68ed1a37ebac9d7c14d7ebe80bd2f8e1762339",
    "contract_payload_sha256": "73e3d99a4b49b2e763866c0a2d446159e722e3206ed6b871641f7ecb86e2c090",
}
WARM_MAP = {
    "winter_20200122": 0.026324180392231614,
    "spring_20200318": 0.03340846836517207,
    "early_summer_20200624": 0.021139407105326802,
    "late_summer_20200819": 0.02696251799221642,
}
STATIC_FIELDS = (
    "product_type_no",
    "product_group_name",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_value_id",
    "perceived_colour_master_id",
    "department_no",
    "index_code",
    "index_group_no",
    "section_no",
    "garment_group_no",
)
FORBIDDEN_STUDENT_FIELDS = (
    "article_id_embedding",
    "product_code",
    "item2vec_vector",
    "deepwalk_vector",
    "co_vis_vector",
    "popularity",
    "transaction_count",
    "first_sale",
    "validation_behavior",
)


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def validate_protocol() -> None:
    if list(ROLLING_PROTOCOL) != [
        "winter_20200122",
        "spring_20200318",
        "early_summer_20200624",
        "late_summer_20200819",
    ]:
        raise RuntimeError("M4 requires the frozen four-window M3.3 protocol")
    all_dates = {
        date
        for protocol in ROLLING_PROTOCOL.values()
        for date in [
            *protocol["inner_train"],
            protocol["inner_validation"],
            *protocol["outer_train"],
            protocol["outer_validation"],
        ]
    }
    if any(date >= FINAL_CUTOFF for date in all_dates):
        raise RuntimeError("M4 must not read the final week")


def validate_fashionclip(root: Path) -> dict[str, Any]:
    paths = {
        "metrics": root / "metrics.json",
        "contract_file": root / "contract.json",
        "embeddings": root / "embeddings.float16.npy",
        "items": root / "items.csv",
        "row_status": root / "row_status.uint8.npy",
    }
    missing = [str(path.resolve()) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing FashionCLIP assets: " + "; ".join(missing))
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    contract = metrics.get("contract", {})
    embeddings = metrics.get("embeddings", {})
    array = np.load(paths["embeddings"], mmap_mode="r")
    status = np.load(paths["row_status"], mmap_mode="r")
    checks = {
        "metrics_completed": metrics.get("status") == "completed",
        "rows": int(embeddings.get("rows", -1)) == EXPECTED_FASHIONCLIP["rows"],
        "shape": tuple(array.shape) == EXPECTED_FASHIONCLIP["shape"],
        "dtype": str(array.dtype) == EXPECTED_FASHIONCLIP["dtype"],
        "valid_rows": int(embeddings.get("valid_rows", -1)) == EXPECTED_FASHIONCLIP["valid_rows"],
        "failed_rows": int(embeddings.get("failed_rows", -1)) == EXPECTED_FASHIONCLIP["failed_rows"],
        "row_status": status.shape == (EXPECTED_FASHIONCLIP["rows"],) and bool(np.all(status == 1)),
        "revision": contract.get("revision") == EXPECTED_FASHIONCLIP["revision"],
        "processor_use_fast": contract.get("processor_use_fast") is EXPECTED_FASHIONCLIP["processor_use_fast"],
        "embedding_sha256": _sha256(paths["embeddings"]) == EXPECTED_FASHIONCLIP["embedding_sha256"],
        "contract_payload_sha256": metrics.get("contract_sha256")
        == EXPECTED_FASHIONCLIP["contract_payload_sha256"],
        "items_sha256": _sha256(paths["items"]) == embeddings.get("items_sha256"),
        "row_status_sha256": _sha256(paths["row_status"]) == embeddings.get("row_status_sha256"),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError("FashionCLIP identity gate failed: " + ", ".join(failed))
    return {
        "status": "passed",
        "checks": checks,
        "expected": EXPECTED_FASHIONCLIP,
        "observed": {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "contract_payload_sha256": metrics["contract_sha256"],
            "contract_file_sha256": _sha256(paths["contract_file"]),
        },
        "artifacts": {name: file_identity(path) for name, path in paths.items()},
        "note": "contract_payload_sha256 is the authoritative canonical payload hash; contract_file_sha256 is only the serialized JSON file hash",
    }


def validate_warm_baseline(metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("run_id") != "m3-3-v1-cross-season-adaptive-seasonal":
        raise ValueError("M4 requires the exact M3.3 run")
    if metrics.get("status") != "measured" or metrics.get("contract", {}).get("final_week") != "not_run":
        raise ValueError("M3.3 measured/final-week boundary failed")
    observed = {
        name: float(row["evaluation"]["orderings"]["anchor__inactive_rrf"]["segments"]["overall"]["map@12"])
        for name, row in metrics["development"].items()
    }
    checks = {name: abs(observed[name] - expected) <= 1e-15 for name, expected in WARM_MAP.items()}
    if set(observed) != set(WARM_MAP) or not all(checks.values()):
        raise ValueError("M3.3 Warm-v1 four-window reference drift")
    evaluation_dbs = {
        name: file_identity(Path(row["scoring"]["evaluation_db"]))
        for name, row in metrics["development"].items()
    }
    declared = {
        Path(item["path"]).resolve(): item
        for item in metrics["resources"]["artifacts"]
        if item["path"].endswith("evaluation.duckdb")
    }
    for identity in evaluation_dbs.values():
        expected = declared.get(Path(identity["path"]).resolve())
        if expected is None or identity["sha256"] != expected["sha256"] or identity["bytes"] != expected["bytes"]:
            raise ValueError("Warm-v1 evaluation artifact identity drift")
    return {
        "status": "passed",
        "run_id": metrics["run_id"],
        "metrics": file_identity(metrics_path),
        "map@12": observed,
        "checks": checks,
        "evaluation_dbs": evaluation_dbs,
        "contract": {
            "candidate_pool": "six-source weighted RRF Top100 plus up to 200 Item2Vec-only candidates",
            "features": "84-dimensional no-decay target-aware schema",
            "sampling": "distribution-aware unweighted 30:1 negative sampling",
            "ranker": "LightGBM LambdaRank with inner-temporal early stopping",
            "fallback": "inactive users retain exact RRF/candidate_rank ordering",
        },
    }


def audit_coldness(
    *, transactions_path: Path, articles_path: Path, warm: dict[str, Any]
) -> dict[str, Any]:
    con = duckdb.connect()
    tx = str(transactions_path.resolve()).replace("'", "''")
    articles = str(articles_path.resolve()).replace("'", "''")
    results: dict[str, Any] = {}
    try:
        for window, protocol in ROLLING_PROTOCOL.items():
            cutoff = protocol["outer_validation"]
            db_path = warm["evaluation_dbs"][window]["path"].replace("'", "''")
            alias = "warm_db"
            con.execute(f"ATTACH '{db_path}' AS {alias} (READ_ONLY)")
            rows = con.execute(
                f"""
                WITH counts AS (
                    SELECT article_id, count(*) AS events
                    FROM read_parquet('{tx}') WHERE t_dat < DATE '{cutoff}' GROUP BY article_id
                ), catalog AS (
                    SELECT a.article_id, coalesce(c.events,0) AS events
                    FROM read_csv_auto('{articles}', all_varchar=true) a LEFT JOIN counts c USING(article_id)
                ), bucketed AS (
                    SELECT article_id, CASE WHEN events=0 THEN 'strict_cold'
                        WHEN events<=5 THEN 'sparse_1_5' WHEN events<=20 THEN 'sparse_6_20'
                        ELSE 'warm_21_plus' END AS bucket FROM catalog
                ), truth AS (
                    SELECT DISTINCT customer_id,article_id FROM {alias}.m29_eval_truth
                ), truth_bucket AS (
                    SELECT t.customer_id,t.article_id,b.bucket FROM truth t JOIN bucketed b USING(article_id)
                )
                SELECT b.bucket, count(*) AS catalog_items,
                       (SELECT count(*) FROM truth_bucket tb WHERE tb.bucket=b.bucket) AS truth_pairs,
                       (SELECT count(DISTINCT customer_id) FROM truth_bucket tb WHERE tb.bucket=b.bucket) AS truth_users
                FROM bucketed b GROUP BY b.bucket ORDER BY b.bucket
                """
            ).fetchall()
            latest = con.execute(
                f"SELECT max(t_dat) FROM read_parquet('{tx}') WHERE t_dat < DATE '{cutoff}'"
            ).fetchone()[0]
            results[window] = {
                "cutoff": cutoff,
                "latest_transaction_before_cutoff": str(latest),
                "buckets": {
                    str(bucket): {
                        "catalog_items": int(catalog_items),
                        "truth_pairs": int(truth_pairs),
                        "truth_users": int(truth_users),
                    }
                    for bucket, catalog_items, truth_pairs, truth_users in rows
                },
            }
            con.execute(f"DETACH {alias}")
    finally:
        con.close()
    return results


def run_m40(*, source_root: Path, output_dir: Path) -> dict[str, Any]:
    validate_protocol()
    output_dir.mkdir(parents=True, exist_ok=True)
    fashionclip_root = source_root / "artifacts" / "m2_4" / "fashionclip-full-v1"
    warm_metrics = source_root / "reports" / "m3_3" / "m3-3-v1-cross-season-adaptive-seasonal" / "metrics.json"
    transactions = source_root / "data" / "interim" / "audit" / "transactions.parquet"
    articles = source_root / "data" / "raw" / "articles.csv"
    fashionclip = validate_fashionclip(fashionclip_root)
    warm = validate_warm_baseline(warm_metrics)
    result = {
        "schema_version": "m4.0-contract-v1",
        "stage": "M4.0",
        "status": "measured",
        "run_id": M4_RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fashionclip": fashionclip,
        "warm_v1": warm,
        "coldness": audit_coldness(transactions_path=transactions, articles_path=articles, warm=warm),
        "student_input_contract": {
            "static_fields": list(STATIC_FIELDS),
            "missing_flags": "one reserved missing category per field plus image-missing indicator",
            "forbidden_fields": list(FORBIDDEN_STUDENT_FIELDS),
            "all_inputs_cutoff_independent": True,
        },
        "inputs": {
            "transactions": file_identity(transactions),
            "articles": file_identity(articles),
        },
        "final_week": "not_run",
    }
    atomic_json(output_dir / "M4_0_metrics.json", result)
    return result
