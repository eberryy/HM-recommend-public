"""One-shot 2020-09-16 confirmation of frozen WV3-741 plus P018.

The module deliberately separates ``prepare`` from ``evaluate``.  ``prepare``
may read only behavior strictly before the cutoff and must freeze every action.
Only ``evaluate`` may open the seven-day label interval.  This is a policy-blind
temporal confirmation, not a pristine dataset-blind experiment: the fixed
68,984-user cohort was originally defined from final-week truth users.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import shutil
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from . import m2, m28, m29, mind_warm_dense as dense, mind_warm_mve as mve
from . import mind_warm_candidate_ranker as candidate
from . import mind_warm_relation as relation
from . import mind_warm_shared_admission as shared


ROOT = Path(__file__).resolve().parents[2]
CUTOFF = "2020-09-16"
END = "2020-09-23"
TOTAL_USERS = 68_984
ART = ROOT / "artifacts/mind_warm_side/MIND-WARM-FINAL-001"
REPORT = ROOT / "reports/mind_warm_side"
CONTRACT = REPORT / "MIND_WARM_FINAL001_CONTRACT.json"
PREPARED = REPORT / "MIND_WARM_FINAL001_PREPARED.json"
METRICS = REPORT / "MIND_WARM_FINAL001_METRICS.json"
FINAL_REPORT = REPORT / "MIND_WARM_FINAL001_FINAL.md"
TX = ROOT / "data/interim/audit/transactions.parquet"
RAW = ROOT / "data/raw"
BASE100 = ROOT / "artifacts/m1_5/m1-5-v11-full-v8-anchored/candidates.parquet"
BASE100_MANIFEST = ROOT / "artifacts/m1_5/m1-5-v11-full-v8-anchored/manifest.json"
BASE_WIDE = ROOT / "artifacts/m1_6/m1-6-v8-hot-full-manifest-bound/candidate_features.parquet"

SOURCE_ROOT = ART / "item2vec-source"
EXPANDED = ART / "expanded-candidates.parquet"
BASE_FEATURES0 = ART / "base-features-without-affinity.parquet"
BASE_FEATURES = ART / "base-features.parquet"
BPR_ROOT = ART / "bpr"
RANK_DB = ART / "rank-build.duckdb"
RANKS = ART / "ranks-top50.parquet"
WV3_SCORES = ART / "wv3-scores.parquet"
WV3_SWAPS = ART / "wv3-741-swaps.parquet"
WV3_BASELINE = ART / "wv3-741-top50.parquet"
MIND_ROOT = ART / "mind"
MIND_ONLY = ART / "mind-only-three-interest.parquet"
MIND_FEATURES = ART / "mind-only-features.parquet"
MIND_DENSE = ART / "mind-only-dense.parquet"
MIND_ORDER = ART / "p005-order.parquet"
TAIL_KEYS = ART / "wv3-tail-keys.parquet"
TAIL_DENSE = ART / "wv3-tail-dense.parquet"
PAIR_FEATURES = ART / "winner-victim-features.parquet"
P005_ACTIONS = ART / "p005-actions.parquet"
P006_ACTIONS = ART / "p006-actions.parquet"
P018_ACTIONS = ART / "p018-frozen-actions.parquet"
ACTION_MANIFEST = ART / "ACTION_MANIFEST.json"

WARM_ROOT = Path(__file__).resolve().parents[2]
WV2_BASE = ROOT / "artifacts/warm_v2/WV2-000/late_summer_20200819"
WV2_BPR = ROOT / "artifacts/warm_v2/WV2-501/late_summer_20200819"
WV3_POINT = WARM_ROOT / "artifacts/warm_v3/WV3-661/MODEL.txt"
WV3_LAMBDA = WARM_ROOT / "artifacts/warm_v3/WV3-680/MODEL.txt"
P005_MODEL = ROOT / "artifacts/mind_warm_side/MIND-WARM-TRANSFER/MIND-WARM-RANK-005/models/through_2020-06-24/model.txt"
P005_META = P005_MODEL.with_name("MODEL.json")
RELATION_MODEL = ROOT / "artifacts/mind_warm_side/MIND-WARM-RANK-002/models/through_2020-06-24/dense_primary.txt"
RELATION_META = RELATION_MODEL.with_suffix(".json")

BPR_PARAMS = {
    "factors": 100,
    "learning_rate": 0.01,
    "regularization": 0.01,
    "iterations": 100,
    "num_threads": 8,
    "verify_negative_samples": True,
    "random_state": 20260908,
}

PAIR_TRANSFORM = [
    "repurchase_present", "item2vec_is_new", "user_item_events_12w",
    "user_item_days_since_last_purchase", "user_product_type_share_12w",
    "user_department_share_12w", "item_days_since_last_sale",
]

# Exact WV3-661/680 feature contract: 24 rank/state fields followed by the
# 75 rich source, trend and affinity fields registered in WV3-660.
BASE_RICH_FEATURES = [
    "r0_reciprocal", "r1_reciprocal", "r0_fraction", "r1_fraction",
    "rank_gap", "r0_top12", "r1_top12", "r0_top50", "r1_top50",
    "latent_within_user_z", "bpr_missing", "log_history", "log_unique",
    "log_recency", *PAIR_TRANSFORM, "rf_reciprocal", "rf_fraction",
    "rf_victim_band",
]
EXTRA_FEATURES = [
    "fused_score", "source_count", "repurchase_rank", "repurchase_score",
    "repurchase_rrf_contribution", "recent_popularity_present",
    "recent_popularity_rank", "recent_popularity_score",
    "recent_popularity_rrf_contribution", "product_family_present",
    "product_family_rank", "product_family_score",
    "product_family_rrf_contribution", "user_day_covisit_present",
    "user_day_covisit_rank", "user_day_covisit_score",
    "user_day_covisit_rrf_contribution", "age_popularity_present",
    "age_popularity_rank", "age_popularity_score",
    "age_popularity_rrf_contribution", "attribute_content_present",
    "attribute_content_rank", "attribute_content_score",
    "attribute_content_rrf_contribution", "item2vec_present", "item2vec_rank",
    "item2vec_score", "item2vec_cosine", "item2vec_best_seed_rank",
    "item2vec_best_neighbor_rank", "item2vec_seed_support",
    "item2vec_vocab_count", "customer_age", "customer_age_missing",
    "age_bucket", "user_avg_price_12w", "user_online_share_12w",
    "item_events_7d", "item_events_28d", "item_events_12w",
    "item_unique_customers_28d", "item_avg_price_28d",
    "item_trend_7d_vs_28d", "user_item_price_gap", "user_item_events_28d",
    "user_item_decay_28d_halflife_12w", "user_product_code_events_28d",
    "user_product_code_events_12w", "user_product_code_days_since",
    "user_product_code_share_12w", "user_product_code_decay_28d_halflife_12w",
    "user_product_type_events_28d", "user_product_type_events_12w",
    "user_product_type_days_since", "user_product_type_decay_28d_halflife_12w",
    "user_department_events_28d", "user_department_events_12w",
    "user_department_days_since", "user_department_decay_28d_halflife_12w",
    "user_garment_events_28d", "user_garment_events_12w",
    "user_garment_days_since", "user_garment_share_12w",
    "user_garment_decay_28d_halflife_12w", "user_colour_events_28d",
    "user_colour_events_12w", "user_colour_days_since", "user_colour_share_12w",
    "user_colour_decay_28d_halflife_12w", "user_index_group_events_28d",
    "user_index_group_events_12w", "user_index_group_days_since",
    "user_index_group_share_12w", "user_index_group_decay_28d_halflife_12w",
]
WV3_FEATURES = BASE_RICH_FEATURES + EXTRA_FEATURES


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sql(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def progress(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def resource_gate() -> dict[str, float]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), *[
            (x, ctypes.c_ulonglong) for x in (
                "total", "available", "page_total", "page_free",
                "virtual_total", "virtual_free", "extended",
            )
        ]]
    status = MemoryStatus(); status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("cannot read physical-memory status")
    ram = status.available / 2**30
    disk = shutil.disk_usage(ROOT).free / 2**30
    if ram < 4 or disk < 10:
        raise RuntimeError(f"resource floor failed: RAM={ram:.2f} GiB, disk={disk:.2f} GiB")
    return {"available_ram_gib": ram, "free_disk_gib": disk}


def connection(memory: str = "6GB", database: Path | None = None) -> duckdb.DuckDBPyConnection:
    if database is not None:
        database.parent.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(str(database) if database is not None else ":memory:")
    db.execute("SET threads=4")
    db.execute(f"SET memory_limit='{memory}'")
    spill = ART / "duckdb-spill"; spill.mkdir(parents=True, exist_ok=True)
    db.execute(f"SET temp_directory='{sql(spill)}'")
    return db


def metadata(path: Path) -> dict[str, Any]:
    """Continuity metadata, deliberately not a standalone authenticity claim."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": "not_computed_direct_continuity"}


def parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connection("2GB") as db:
        db.register("out_frame", frame)
        db.execute(f"COPY out_frame TO '{sql(path)}' (FORMAT PARQUET,COMPRESSION ZSTD)")


def check_inputs() -> dict[str, Any]:
    if not CONTRACT.is_file():
        raise FileNotFoundError("final confirmation contract must exist before preparation")
    contract = read(CONTRACT)
    if not contract.get("registered_before_item_level_final_label_read"):
        raise AssertionError("label firewall contract is not registered")
    required = [TX, RAW / "articles.csv", RAW / "customers.csv", BASE100,
                BASE100_MANIFEST, BASE_WIDE, WV2_BASE / "outer_model.txt",
                WV2_BPR / "outer_model.txt", WV3_POINT, WV3_LAMBDA,
                P005_MODEL, P005_META, RELATION_MODEL, RELATION_META]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen inputs: {missing}")
    manifest = read(BASE100_MANIFEST)
    if (manifest.get("cutoff") != CUTOFF or manifest.get("validation_users") != TOTAL_USERS
            or manifest.get("candidate_rows") != TOTAL_USERS * 100):
        raise AssertionError("final base cohort/candidate manifest drift")
    with connection("2GB") as db:
        mismatch = db.execute(
            f"SELECT (SELECT count(*) FROM read_parquet('{sql(BASE100)}') x ANTI JOIN read_parquet('{sql(BASE_WIDE)}') y USING(customer_id,article_id)),"
            f"(SELECT count(*) FROM read_parquet('{sql(BASE_WIDE)}') y ANTI JOIN read_parquet('{sql(BASE100)}') x USING(customer_id,article_id))"
        ).fetchone()
        wide_columns = {row[0] for row in db.execute(f"DESCRIBE SELECT * FROM read_parquet('{sql(BASE_WIDE)}')").fetchall()}
    if mismatch != (0, 0) or not set(m2.RETRIEVAL_FEATURES).issubset(wide_columns):
        raise AssertionError("M1.6 wide table is not the exact feature-enriched M1.5 candidate set")
    return {"resources": resource_gate(), "required_inputs": len(required),
            "base_rows": manifest["candidate_rows"], "base_users": manifest["validation_users"],
            "wide_identity_mismatch_each_direction": list(mismatch)}


def build_item2vec_source() -> dict[str, Any]:
    marker = SOURCE_ROOT / "source-manifest.json"
    if marker.is_file():
        saved = read(marker)
        if saved.get("cutoff") != CUTOFF or saved.get("contract") != m29.ITEM2VEC_CONFIG:
            raise ValueError("stale final Item2Vec source")
        return saved
    if SOURCE_ROOT.exists():
        raise FileExistsError("preserve incomplete Item2Vec source and inspect before retry")
    base_manifest = read(BASE100_MANIFEST)
    baseline = {
        "cutoff": CUTOFF, "sample_rate": 1.0,
        "candidate_path": str(BASE100.resolve()),
        "candidate_bytes": BASE100.stat().st_size,
        "candidate_sha256": base_manifest["inputs"]["historical_candidates_sha256"],
        "declared_rows": TOTAL_USERS * 100,
    }
    transaction_identity = metadata(TX)
    original_m29 = m29._file_identity
    original_m28 = m28._file_identity
    m29._file_identity = metadata
    m28._file_identity = metadata
    try:
        result = m29._build_new_source(
            cutoff=CUTOFF,
            window=m29.M29Window(CUTOFF, BASE100, BASE100_MANIFEST),
            baseline_identity=baseline,
            transactions_path=TX,
            transaction_identity=transaction_identity,
            source_dir=SOURCE_ROOT,
        )
    finally:
        m29._file_identity = original_m29
        m28._file_identity = original_m28
    progress(f"final Item2Vec source complete: {result['audit']['rows']:,} rows")
    return result


def build_expanded_candidates(source: dict[str, Any]) -> dict[str, Any]:
    marker = EXPANDED.with_suffix(".json")
    if EXPANDED.is_file() and marker.is_file():
        saved = read(marker)
        if saved.get("cutoff") != CUTOFF:
            raise ValueError("stale expanded candidate cache")
        return saved
    if EXPANDED.exists() or marker.exists():
        raise FileExistsError("preserve incomplete expanded candidate cache")
    started = time.perf_counter(); resource_gate()
    source_candidate = Path(source["artifacts"]["candidates"]["path"])
    items_path = Path(source["artifacts"]["items"]["path"])
    with connection("7GB") as db:
        # BASE_WIDE is the feature-enriched materialization of the exact
        # BASE100 identity set (the preflight verifies bidirectional equality).
        db.execute(f"CREATE VIEW baseline AS SELECT * FROM read_parquet('{sql(BASE_WIDE)}')")
        db.execute(f"CREATE VIEW item2vec_source AS SELECT * FROM read_parquet('{sql(source_candidate)}')")
        db.execute(f"CREATE TEMP TABLE item2vec_vocab AS SELECT article_id,token_count FROM read_csv_auto('{sql(items_path)}',header=true,all_varchar=false)")
        db.execute("""CREATE TEMP TABLE item2vec_new AS
            SELECT i.*,row_number() OVER(PARTITION BY customer_id ORDER BY item2vec_rank,article_id) new_rank
            FROM item2vec_source i ANTI JOIN baseline b USING(customer_id,article_id)
            QUALIFY new_rank<=200""")
        base_projection = m29._base_retrieval_projection("b")
        empty_projection = m29._empty_retrieval_projection()
        query = f"""
            SELECT b.customer_id,b.article_id,{base_projection},
                (i.article_id IS NOT NULL)::INTEGER item2vec_present,0::INTEGER item2vec_is_new,
                i.item2vec_rank,i.item2vec_score,i.item2vec_cosine,
                i.best_seed_rank item2vec_best_seed_rank,
                i.best_neighbor_rank item2vec_best_neighbor_rank,
                i.seed_support item2vec_seed_support,
                coalesce(v.token_count,0)::BIGINT item2vec_vocab_count
            FROM baseline b LEFT JOIN item2vec_source i USING(customer_id,article_id)
            LEFT JOIN item2vec_vocab v USING(article_id)
            UNION ALL
            SELECT n.customer_id,n.article_id,(100+n.new_rank)::INTEGER candidate_rank,
                {empty_projection},1::INTEGER item2vec_present,1::INTEGER item2vec_is_new,
                n.item2vec_rank,n.item2vec_score,n.item2vec_cosine,
                n.best_seed_rank,n.best_neighbor_rank,n.seed_support,
                coalesce(v.token_count,0)::BIGINT item2vec_vocab_count
            FROM item2vec_new n LEFT JOIN item2vec_vocab v USING(article_id)
        """
        db.execute(f"COPY ({query} ORDER BY customer_id,candidate_rank,article_id) TO '{sql(EXPANDED)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)")
        stats = db.execute(f"""WITH g AS (SELECT customer_id,count(*) n,count(DISTINCT article_id) u,
            min(candidate_rank) lo,max(candidate_rank) hi,count(DISTINCT candidate_rank) dr
            FROM read_parquet('{sql(EXPANDED)}') GROUP BY 1)
            SELECT (SELECT count(*) FROM read_parquet('{sql(EXPANDED)}')),
                count(*),min(n),max(n),count(*) FILTER(WHERE n<>u OR lo<>1 OR hi<>n OR dr<>n OR n<100 OR n>300),
                (SELECT count(*) FROM read_parquet('{sql(EXPANDED)}') WHERE item2vec_is_new=1) FROM g""").fetchone()
    result = {"cutoff": CUTOFF, "rows": int(stats[0]), "users": int(stats[1]),
              "min_group_rows": int(stats[2]), "max_group_rows": int(stats[3]),
              "invalid_groups": int(stats[4]), "item2vec_only_rows": int(stats[5]),
              "history_strictly_before_cutoff": True, "labels_read": False,
              "seconds": time.perf_counter() - started}
    if result["users"] != TOTAL_USERS or result["invalid_groups"]:
        raise AssertionError("expanded candidates violate the frozen variable-group contract")
    write(marker, result)
    progress(f"expanded candidates complete: {result['rows']:,} rows")
    return result


def build_base_features(expanded: dict[str, Any]) -> dict[str, Any]:
    marker = BASE_FEATURES.with_suffix(".json")
    if BASE_FEATURES.is_file() and marker.is_file():
        saved = read(marker)
        if saved.get("cutoff") != CUTOFF:
            raise ValueError("stale base feature cache")
        return saved
    if BASE_FEATURES.exists() or marker.exists():
        raise FileExistsError("preserve incomplete base feature cache")
    started = time.perf_counter(); resource_gate()
    work = ART / "base-feature-work"
    db = m2.prepare_tabular_connection(RAW, work)
    original_sha = m2._sha256
    m2._sha256 = lambda _: "not_computed_direct_continuity"
    try:
        # Firewall: m2_truth becomes empty because this view contains no row at
        # or after the held-out cutoff.  The target column is a zero placeholder.
        db.execute(f"CREATE OR REPLACE VIEW transactions AS SELECT * FROM read_parquet('{sql(TX)}') WHERE t_dat<DATE '{CUTOFF}'")
        m2._create_static_dimensions(db)
        evidence = m2.build_point_in_time_dataset(
            db, EXPANDED, {"cutoff": CUTOFF, "declared_rows": expanded["rows"]},
            BASE_FEATURES0, m2.M2Config(candidate_k=300),
            retrieval_features=m2.RETRIEVAL_FEATURES + m29.ITEM2VEC_FEATURES,
            candidate_group_range=(100, 300),
        )
    finally:
        m2._sha256 = original_sha
        db.close()
    if evidence["positives"] or evidence["truth_pairs"]:
        raise AssertionError("final labels crossed the base-feature firewall")
    append_affinity_features(BASE_FEATURES0, BASE_FEATURES)
    with connection("3GB") as db:
        stats = db.execute(f"SELECT count(*),count(DISTINCT(customer_id,article_id)),count(DISTINCT customer_id),sum(target) FROM read_parquet('{sql(BASE_FEATURES)}')").fetchone()
        latest = db.execute(f"SELECT max(t_dat) FROM read_parquet('{sql(TX)}') WHERE t_dat<DATE '{CUTOFF}'").fetchone()[0]
    result = {"cutoff": CUTOFF, "rows": int(stats[0]), "unique_rows": int(stats[1]),
              "users": int(stats[2]), "placeholder_target_sum": int(stats[3]),
              "latest_history_date": str(latest), "history_strictly_before_cutoff": str(latest) < CUTOFF,
              "labels_read": False, "feature_count": len(m2.FULL_FEATURES + m29.ITEM2VEC_FEATURES + m210_features()),
              "seconds": time.perf_counter() - started}
    if not (result["rows"] == result["unique_rows"] == expanded["rows"] and result["users"] == TOTAL_USERS
            and result["placeholder_target_sum"] == 0 and result["history_strictly_before_cutoff"]):
        raise AssertionError("base feature conservation/firewall failed")
    write(marker, result)
    progress(f"base features complete: {result['rows']:,} rows")
    return result


def m210_features() -> list[str]:
    from .m210 import TARGET_AWARE_FEATURES
    return TARGET_AWARE_FEATURES


def append_affinity_features(source: Path, output: Path) -> None:
    """Add the exact M2.10 target-aware fields without opening future labels."""
    from .m210 import TARGET_AWARE_FEATURES
    with connection("7GB") as db:
        db.execute(f"CREATE VIEW base AS SELECT * FROM read_parquet('{sql(source)}')")
        db.execute("CREATE TEMP TABLE users AS SELECT DISTINCT customer_id FROM base")
        db.execute(f"""CREATE TEMP TABLE articles AS SELECT article_id,
            try_cast(product_code AS INTEGER) article_product_code,
            try_cast(product_type_no AS INTEGER) article_product_type_no,
            try_cast(garment_group_no AS INTEGER) article_garment_group_no,
            try_cast(department_no AS INTEGER) article_department_no,
            try_cast(index_group_no AS INTEGER) article_index_group_no,
            try_cast(perceived_colour_master_id AS INTEGER) article_colour_master_id
            FROM read_csv_auto('{sql(RAW / 'articles.csv')}',header=true,all_varchar=true)""")
        db.execute(f"""CREATE TEMP TABLE hist AS SELECT t.customer_id,t.article_id,t.t_dat,
            a.article_product_code,a.article_product_type_no,a.article_garment_group_no,
            a.article_department_no,a.article_index_group_no,a.article_colour_master_id
            FROM read_parquet('{sql(TX)}') t SEMI JOIN users USING(customer_id)
            JOIN articles a USING(article_id)
            WHERE t.t_dat>=DATE '{CUTOFF}'-INTERVAL 12 WEEK AND t.t_dat<DATE '{CUTOFF}'""")
        dimensions = {
            "ui": "article_id", "pc": "article_product_code", "pt": "article_product_type_no",
            "dp": "article_department_no", "gg": "article_garment_group_no",
            "cl": "article_colour_master_id", "ig": "article_index_group_no",
        }
        for table, key in dimensions.items():
            db.execute(f"""CREATE TEMP TABLE {table} AS SELECT customer_id,{key},
                count(*) FILTER(WHERE t_dat>=DATE '{CUTOFF}'-INTERVAL 28 DAY)::BIGINT events_28d,
                count(*)::BIGINT events_12w,date_diff('day',max(t_dat),DATE '{CUTOFF}')::BIGINT days_since,
                sum(exp(-ln(2.0)*date_diff('day',t_dat,DATE '{CUTOFF}')/28.0))::DOUBLE decay
                FROM hist GROUP BY customer_id,{key}""")
        select = f"""SELECT b.*,
            coalesce(ui.events_28d,0)::BIGINT user_item_events_28d,
            coalesce(ui.decay,0.0) user_item_decay_28d_halflife_12w,
            coalesce(pc.events_28d,0)::BIGINT user_product_code_events_28d,
            coalesce(pc.events_12w,0)::BIGINT user_product_code_events_12w,
            pc.days_since user_product_code_days_since,
            coalesce(pc.events_12w,0)/greatest(b.user_history_events_12w,1.0) user_product_code_share_12w,
            coalesce(pc.decay,0.0) user_product_code_decay_28d_halflife_12w,
            coalesce(pt.events_28d,0)::BIGINT user_product_type_events_28d,
            pt.days_since user_product_type_days_since,coalesce(pt.decay,0.0) user_product_type_decay_28d_halflife_12w,
            coalesce(dp.events_28d,0)::BIGINT user_department_events_28d,
            dp.days_since user_department_days_since,coalesce(dp.decay,0.0) user_department_decay_28d_halflife_12w,
            coalesce(gg.events_28d,0)::BIGINT user_garment_events_28d,
            coalesce(gg.events_12w,0)::BIGINT user_garment_events_12w,
            gg.days_since user_garment_days_since,
            coalesce(gg.events_12w,0)/greatest(b.user_history_events_12w,1.0) user_garment_share_12w,
            coalesce(gg.decay,0.0) user_garment_decay_28d_halflife_12w,
            coalesce(cl.events_28d,0)::BIGINT user_colour_events_28d,
            coalesce(cl.events_12w,0)::BIGINT user_colour_events_12w,
            cl.days_since user_colour_days_since,
            coalesce(cl.events_12w,0)/greatest(b.user_history_events_12w,1.0) user_colour_share_12w,
            coalesce(cl.decay,0.0) user_colour_decay_28d_halflife_12w,
            coalesce(ig.events_28d,0)::BIGINT user_index_group_events_28d,
            coalesce(ig.events_12w,0)::BIGINT user_index_group_events_12w,
            ig.days_since user_index_group_days_since,
            coalesce(ig.events_12w,0)/greatest(b.user_history_events_12w,1.0) user_index_group_share_12w,
            coalesce(ig.decay,0.0) user_index_group_decay_28d_halflife_12w
            FROM base b
            LEFT JOIN ui USING(customer_id,article_id)
            LEFT JOIN pc ON b.customer_id=pc.customer_id AND b.article_product_code=pc.article_product_code
            LEFT JOIN pt ON b.customer_id=pt.customer_id AND b.article_product_type_no=pt.article_product_type_no
            LEFT JOIN dp ON b.customer_id=dp.customer_id AND b.article_department_no=dp.article_department_no
            LEFT JOIN gg ON b.customer_id=gg.customer_id AND b.article_garment_group_no=gg.article_garment_group_no
            LEFT JOIN cl ON b.customer_id=cl.customer_id AND b.article_colour_master_id=cl.article_colour_master_id
            LEFT JOIN ig ON b.customer_id=ig.customer_id AND b.article_index_group_no=ig.article_index_group_no"""
        db.execute(f"COPY ({select} ORDER BY customer_id,candidate_rank,article_id) TO '{sql(output)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)")
        schema = {row[0] for row in db.execute(f"DESCRIBE SELECT * FROM read_parquet('{sql(output)}')").fetchall()}
    missing = set(TARGET_AWARE_FEATURES) - schema
    if missing:
        raise AssertionError(f"affinity features missing: {sorted(missing)}")


def build_bpr() -> dict[str, Any]:
    marker = BPR_ROOT / "MODEL.json"
    factors_path = BPR_ROOT / "factors.npz"
    if marker.is_file() and factors_path.is_file():
        saved = read(marker)
        if saved.get("params") != BPR_PARAMS or saved.get("cutoff") != CUTOFF:
            raise ValueError("stale final BPR model")
        return saved
    if BPR_ROOT.exists():
        raise FileExistsError("preserve incomplete final BPR model")
    BPR_ROOT.mkdir(parents=True)
    started = time.perf_counter(); resource_gate()
    with connection("7GB") as db:
        db.execute(f"CREATE VIEW history AS SELECT customer_id,article_id,t_dat FROM read_parquet('{sql(TX)}') WHERE t_dat<DATE '{CUTOFF}'")
        db.execute("CREATE TEMP TABLE users AS SELECT customer_id,(row_number() OVER(ORDER BY customer_id)-1)::INTEGER user_index FROM (SELECT DISTINCT customer_id FROM history)")
        db.execute("CREATE TEMP TABLE items AS SELECT article_id,(row_number() OVER(ORDER BY article_id)-1)::INTEGER item_index FROM (SELECT DISTINCT article_id FROM history)")
        pairs = db.execute("SELECT DISTINCT u.user_index,i.item_index FROM history h JOIN users u USING(customer_id) JOIN items i USING(article_id) ORDER BY 1,2").fetchdf()
        users = db.execute("SELECT * FROM users ORDER BY user_index").fetchdf()
        items = db.execute("SELECT * FROM items ORDER BY item_index").fetchdf()
        latest = str(db.execute("SELECT max(t_dat) FROM history").fetchone()[0])
    matrix = csr_matrix((np.ones(len(pairs), np.float32),
                         (pairs.user_index.to_numpy(), pairs.item_index.to_numpy())),
                        shape=(len(users), len(items)))
    del pairs; gc.collect()
    from implicit.cpu.bpr import BayesianPersonalizedRanking
    model = BayesianPersonalizedRanking(**BPR_PARAMS)
    epochs: list[dict[str, Any]] = []
    def callback(epoch: int, elapsed: float, correct: int, skipped: int) -> None:
        epochs.append({"epoch": int(epoch), "seconds": float(elapsed),
                       "correct": int(correct), "skipped": int(skipped)})
        if (epoch + 1) % 10 == 0:
            progress(f"BPR epoch {epoch + 1}/100")
    with threadpool_limits(limits=1, user_api="blas"):
        model.fit(matrix, show_progress=False, callback=callback)
    np.savez(factors_path, user_factors=model.user_factors, item_factors=model.item_factors)
    parquet(users, BPR_ROOT / "users.parquet")
    parquet(items, BPR_ROOT / "items.parquet")
    result = {"cutoff": CUTOFF, "params": BPR_PARAMS, "latest_history_date": latest,
              "binary_distinct_pairs": int(matrix.nnz), "users": len(users), "items": len(items),
              "factor_shapes": [list(model.user_factors.shape), list(model.item_factors.shape)],
              "epochs": epochs, "history_strictly_before_cutoff": latest < CUTOFF,
              "labels_read": False, "seconds": time.perf_counter() - started}
    if not result["history_strictly_before_cutoff"]:
        raise AssertionError("BPR temporal boundary failed")
    write(marker, result)
    del model, matrix, users, items; gc.collect()
    progress(f"BPR complete: {result['binary_distinct_pairs']:,} distinct pairs")
    return result


def category_maps(path: Path) -> dict[str, dict[int, int]]:
    raw = read(path)
    return {name: {int(k): int(v) for k, v in values.items()} for name, values in raw.items()}


def build_ranks() -> dict[str, Any]:
    marker = RANKS.with_suffix(".json")
    if RANKS.is_file() and marker.is_file():
        saved = read(marker)
        if saved.get("cutoff") != CUTOFF:
            raise ValueError("stale rank cache")
        return saved
    if RANKS.exists() or marker.exists():
        raise FileExistsError("preserve incomplete rank cache")
    started = time.perf_counter(); resource_gate(); build_bpr()
    base_model = lgb.Booster(model_file=str(WV2_BASE / "outer_model.txt"))
    bpr_model = lgb.Booster(model_file=str(WV2_BPR / "outer_model.txt"))
    base_names, bpr_names = base_model.feature_name(), bpr_model.feature_name()
    if len(base_names) != 84 or len(bpr_names) != 86 or bpr_names[-2:] != ["wv2_bpr_user_item_score", "wv2_bpr_unavailable"]:
        raise AssertionError("frozen WV2 model feature contract drift")
    maps0 = category_maps(WV2_BASE / "outer_category_maps.json")
    maps1 = category_maps(WV2_BPR / "outer_category_maps.json")
    with np.load(BPR_ROOT / "factors.npz", allow_pickle=False) as factors:
        uf = factors["user_factors"]; itf = factors["item_factors"]
    with connection("2GB") as db:
        users = db.execute(f"SELECT * FROM read_parquet('{sql(BPR_ROOT / 'users.parquet')}') ORDER BY user_index").fetchdf()
        items = db.execute(f"SELECT * FROM read_parquet('{sql(BPR_ROOT / 'items.parquet')}') ORDER BY item_index").fetchdf()
    user_index = pd.Index(users.customer_id); item_index = pd.Index(items.article_id)
    del users, items
    if RANK_DB.exists():
        raise FileExistsError("preserve incomplete rank database")
    target = connection("7GB", RANK_DB)
    source = duckdb.connect()
    rows = batches = 0
    try:
        select = list(dict.fromkeys(["customer_id", "article_id", "candidate_rank", "user_history_events_12w", *base_names]))
        cursor = source.execute(f"SELECT {','.join(select)} FROM read_parquet('{sql(BASE_FEATURES)}')")
        initialized = False
        while True:
            frame = cursor.fetch_df_chunk(32)
            if frame.empty:
                break
            ui = user_index.get_indexer(frame.customer_id)
            ii = item_index.get_indexer(frame.article_id)
            good = (ui >= 0) & (ii >= 0)
            latent = np.full(len(frame), np.nan, np.float32)
            latent[good] = np.einsum("ij,ij->i", uf[ui[good]], itf[ii[good]])
            frame["wv2_bpr_user_item_score"] = latent
            frame["wv2_bpr_unavailable"] = (~good).astype(np.float32)
            out = frame[["customer_id", "article_id", "candidate_rank", "user_history_events_12w"]].copy()
            out["score_base"] = base_model.predict(m2._prepare_frame(frame, base_names, maps0), num_threads=4)
            out["score_bpr"] = bpr_model.predict(m2._prepare_frame(frame, bpr_names, maps1), num_threads=4)
            out["latent"] = latent
            out["missing"] = (~good).astype(np.float32)
            target.register("batch", out)
            if not initialized:
                target.execute("CREATE TABLE predictions AS SELECT * FROM batch WHERE FALSE")
                initialized = True
            target.execute("INSERT INTO predictions SELECT * FROM batch")
            target.unregister("batch")
            rows += len(out); batches += 1
            if batches % 25 == 0:
                progress(f"WV2 scoring: {rows:,} rows")
            del frame, out, latent; gc.collect()
        target.execute("""CREATE TABLE ranked AS SELECT *,
            row_number() OVER(PARTITION BY customer_id ORDER BY score_base DESC,candidate_rank,article_id)::INTEGER r0,
            row_number() OVER(PARTITION BY customer_id ORDER BY score_bpr DESC,candidate_rank,article_id)::INTEGER r1,
            count(*) OVER(PARTITION BY customer_id)::INTEGER candidate_count,
            avg(latent) OVER(PARTITION BY customer_id)::FLOAT latent_mean,
            stddev_samp(latent) OVER(PARTITION BY customer_id)::FLOAT latent_std
            FROM predictions""")
        target.execute("""CREATE TABLE fused AS SELECT *,1.0/(60+r0)+1.0/(60+r1) rrf_score FROM ranked""")
        target.execute("""CREATE TABLE final_ranked AS SELECT *,
            row_number() OVER(PARTITION BY customer_id ORDER BY rrf_score DESC,candidate_rank,article_id)::INTEGER rf
            FROM fused""")
        target.execute(f"""COPY (SELECT * FROM (SELECT *,
            CASE WHEN user_history_events_12w=0 THEN candidate_rank ELSE rf END::INTEGER ap_rf
            FROM final_ranked) x WHERE ap_rf<=50 ORDER BY customer_id,ap_rf,article_id)
            TO '{sql(RANKS)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)""")
        stats = target.execute(f"""WITH g AS (SELECT customer_id,count(*) n,count(DISTINCT article_id) u,
            min(ap_rf) lo,max(ap_rf) hi,count(DISTINCT ap_rf) dr FROM read_parquet('{sql(RANKS)}') GROUP BY 1)
            SELECT count(*),sum(n),count(*) FILTER(WHERE n<>50 OR n<>u OR lo<>1 OR hi<>50 OR dr<>50),
            count(*) FILTER(WHERE EXISTS(SELECT 1 FROM read_parquet('{sql(RANKS)}') x WHERE x.customer_id=g.customer_id AND x.user_history_events_12w=0)) FROM g""").fetchone()
    finally:
        source.close(); target.close()
    result = {"cutoff": CUTOFF, "scored_rows": rows, "batches": batches,
              "top50_rows": int(stats[1]), "users": int(stats[0]), "invalid_groups": int(stats[2]),
              "inactive_users": int(stats[3]), "base_feature_count": len(base_names),
              "bpr_feature_count": len(bpr_names), "labels_read": False,
              "seconds": time.perf_counter() - started}
    if result["users"] != TOTAL_USERS or result["invalid_groups"] or result["top50_rows"] != TOTAL_USERS * 50:
        raise AssertionError("WV2-601 final rank conservation failed")
    write(marker, result)
    # This database contains four redundant 20M-row staging tables.  The
    # parquet Top50 is the completed reusable output; deleting only this
    # regenerable staging file protects the tight local disk budget.
    RANK_DB.unlink(missing_ok=True)
    progress(f"WV2-601 Top50 complete: {result['top50_rows']:,} rows")
    return result


def wv3_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Reconstruct the exact 99-column WV3-661/680 inference matrix."""
    r0 = frame.r0.to_numpy(np.float64)
    r1 = frame.r1.to_numpy(np.float64)
    n = frame.candidate_count.to_numpy(np.float64)
    latent = frame.latent.to_numpy(np.float64)
    mean = frame.latent_mean.to_numpy(np.float64)
    std = frame.latent_std.to_numpy(np.float64)
    z = (latent - mean) / (std + 1e-6)
    values: list[np.ndarray] = [
        60.0 / (60.0 + r0), 60.0 / (60.0 + r1), r0 / n, r1 / n,
        (r0 - r1) / n, r0 <= 12, r1 <= 12, r0 <= 50, r1 <= 50,
        z, frame.missing.to_numpy(float),
        np.log1p(frame.user_history_events_12w.to_numpy(float)),
        np.log1p(frame.user_unique_items_12w.to_numpy(float)),
        np.log1p(np.clip(frame.user_days_since_last_purchase.to_numpy(float), 0, 10_000)),
    ]
    for name in PAIR_TRANSFORM:
        value = frame[name].to_numpy(float)
        if "events" in name or "days" in name:
            value = np.log1p(np.clip(value, 0, 10_000))
        values.append(value)
    rf = frame.rf.to_numpy(np.float64)
    values += [1.0 / (60.0 + rf), rf / 50.0, ((rf >= 8) & (rf <= 12)).astype(float)]
    basic = np.nan_to_num(np.column_stack(values).astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
    extra = frame[EXTRA_FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    extra = np.nan_to_num(extra, nan=0.0, posinf=1e6, neginf=-1e6)
    result = np.column_stack((basic, extra)).astype(np.float32, copy=False)
    if result.shape[1] != len(WV3_FEATURES) or not np.isfinite(result).all():
        raise AssertionError("WV3 inference matrix violates the frozen 99-feature contract")
    return result


def score_wv3() -> dict[str, Any]:
    marker = WV3_SCORES.with_suffix(".json")
    if WV3_SCORES.is_file() and marker.is_file():
        saved = read(marker)
        if saved.get("cutoff") != CUTOFF or saved.get("features") != WV3_FEATURES:
            raise ValueError("stale WV3 score cache")
        return saved
    if WV3_SCORES.exists() or marker.exists():
        raise FileExistsError("preserve incomplete WV3 score cache")
    started = time.perf_counter(); resource_gate()
    point = lgb.Booster(model_file=str(WV3_POINT))
    lambdarank = lgb.Booster(model_file=str(WV3_LAMBDA))
    if point.feature_name() != WV3_FEATURES or lambdarank.feature_name() != WV3_FEATURES:
        raise AssertionError("frozen WV3 model feature names differ from the registered 99 columns")
    stage = ART / "wv3-score-build.duckdb"
    if stage.exists():
        raise FileExistsError("preserve incomplete WV3 score database")
    target = connection("5GB", stage)
    source = duckdb.connect()
    rows = batches = 0
    try:
        columns = list(dict.fromkeys([
            "customer_id", "article_id", "rf", "r0", "r1", "candidate_count",
            "latent", "latent_mean", "latent_std", "missing",
            "user_history_events_12w", "user_unique_items_12w",
            "user_days_since_last_purchase", *PAIR_TRANSFORM, *EXTRA_FEATURES,
        ]))
        select = ",".join(f"r.{name}" if name in {
            "customer_id", "article_id", "rf", "r0", "r1", "candidate_count",
            "latent", "latent_mean", "latent_std", "missing", "user_history_events_12w"
        } else f"f.{name}" for name in columns)
        cursor = source.execute(
            f"SELECT {select} FROM read_parquet('{sql(RANKS)}') r "
            f"JOIN read_parquet('{sql(BASE_FEATURES)}') f USING(customer_id,article_id) "
            "WHERE r.user_history_events_12w>0 ORDER BY r.customer_id,r.rf,r.article_id"
        )
        initialized = False
        while True:
            frame = cursor.fetch_df_chunk(16)
            if frame.empty:
                break
            matrix = wv3_matrix(frame)
            out = frame[["customer_id", "article_id", "rf"]].copy()
            out["point_score"] = point.predict(matrix, num_threads=4)
            out["lambda_score"] = lambdarank.predict(matrix, num_threads=4)
            target.register("batch", out)
            if not initialized:
                target.execute("CREATE TABLE raw_scores AS SELECT * FROM batch WHERE FALSE")
                initialized = True
            target.execute("INSERT INTO raw_scores SELECT * FROM batch")
            target.unregister("batch")
            rows += len(out); batches += 1
            if batches % 20 == 0:
                progress(f"WV3 frozen-model scoring: {rows:,} rows")
            del frame, matrix, out; gc.collect()
        target.execute("""CREATE TABLE score_ranks AS SELECT *,
            row_number() OVER(PARTITION BY customer_id ORDER BY point_score DESC,rf,article_id)::INTEGER point_rank,
            row_number() OVER(PARTITION BY customer_id ORDER BY lambda_score DESC,rf,article_id)::INTEGER lambda_rank
            FROM raw_scores""")
        target.execute(f"""COPY (SELECT customer_id,article_id,rf,point_rank,lambda_rank,
            1.0/(60.0+point_rank)+1.0/(60.0+lambda_rank) ranking_score
            FROM score_ranks ORDER BY customer_id,rf,article_id)
            TO '{sql(WV3_SCORES)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)""")
        stats = target.execute(f"SELECT count(*),count(DISTINCT(customer_id,article_id)),count(DISTINCT customer_id) FROM read_parquet('{sql(WV3_SCORES)}')").fetchone()
    finally:
        source.close(); target.close()
    stage.unlink(missing_ok=True)
    result = {"cutoff": CUTOFF, "rows": int(stats[0]), "unique_rows": int(stats[1]),
              "active_users": int(stats[2]), "batches": batches, "features": WV3_FEATURES,
              "labels_read": False, "seconds": time.perf_counter() - started}
    if result["rows"] != result["unique_rows"] or result["rows"] != rows:
        raise AssertionError("WV3 scoring changed active Top50 identities")
    write(marker, result)
    progress(f"WV3 scores complete: {rows:,} active candidate rows")
    return result


def choose_wv3_swaps(scores: pd.DataFrame) -> pd.DataFrame:
    """Exact WV3-741 greedy maximum-two disjoint positive-difference policy."""
    required = {"customer_id", "article_id", "rf", "ranking_score"}
    if set(scores.columns) != required or scores.duplicated(["customer_id", "article_id"]).any():
        raise ValueError("WV3 selector accepts only unique identity, rank and frozen score")
    selected: list[dict[str, Any]] = []
    for customer_id, group in scores.groupby("customer_id", sort=False):
        victims = group[group.rf.between(8, 12)]
        challengers = group[group.rf.between(13, 50)]
        options: list[tuple[Any, ...]] = []
        for c in challengers.itertuples(index=False):
            for v in victims.itertuples(index=False):
                difference = float(c.ranking_score - v.ranking_score)
                if difference > 0:
                    options.append((-difference, -int(v.rf), int(c.rf), str(c.article_id), str(v.article_id), c, v))
        options.sort(key=lambda row: row[:5])
        used_c: set[str] = set(); used_v: set[str] = set()
        for option in options:
            c, v = option[5], option[6]
            if c.article_id in used_c or v.article_id in used_v:
                continue
            selected.append({
                "customer_id": str(customer_id), "challenger_article_id": str(c.article_id),
                "victim_article_id": str(v.article_id), "challenger_rank": int(c.rf),
                "victim_rank": int(v.rf), "challenger_score": float(c.ranking_score),
                "victim_score": float(v.ranking_score),
                "score_difference": float(c.ranking_score - v.ranking_score),
                "swap_order": len(used_c) + 1,
            })
            used_c.add(str(c.article_id)); used_v.add(str(v.article_id))
            if len(used_c) == 2:
                break
    return pd.DataFrame(selected, columns=[
        "customer_id", "challenger_article_id", "victim_article_id", "challenger_rank",
        "victim_rank", "challenger_score", "victim_score", "score_difference", "swap_order",
    ])


def build_wv3_baseline() -> dict[str, Any]:
    marker = WV3_BASELINE.with_suffix(".json")
    if all(path.is_file() for path in (WV3_BASELINE, WV3_SWAPS, marker)):
        saved = read(marker)
        if saved.get("cutoff") != CUTOFF:
            raise ValueError("stale WV3-741 cache")
        return saved
    if any(path.exists() for path in (WV3_BASELINE, WV3_SWAPS, marker)):
        raise FileExistsError("preserve incomplete WV3-741 cache")
    started = time.perf_counter(); score_wv3()
    with connection("3GB") as db:
        scores = db.execute(f"SELECT customer_id,article_id,rf,ranking_score FROM read_parquet('{sql(WV3_SCORES)}') ORDER BY customer_id,rf,article_id").fetchdf()
    swaps = choose_wv3_swaps(scores)
    parquet(swaps, WV3_SWAPS)
    changes = pd.concat([
        swaps[["customer_id", "challenger_article_id", "victim_rank"]].rename(columns={"challenger_article_id": "article_id", "victim_rank": "replacement_rank"}),
        swaps[["customer_id", "victim_article_id", "challenger_rank"]].rename(columns={"victim_article_id": "article_id", "challenger_rank": "replacement_rank"}),
    ], ignore_index=True)
    with connection("5GB") as db:
        db.register("changes", changes)
        db.execute(f"""COPY (SELECT r.*,coalesce(c.replacement_rank,r.ap_rf)::INTEGER champion_rank
            FROM read_parquet('{sql(RANKS)}') r LEFT JOIN changes c USING(customer_id,article_id)
            ORDER BY customer_id,champion_rank,article_id)
            TO '{sql(WV3_BASELINE)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)""")
        stats = db.execute(f"""WITH g AS (SELECT customer_id,count(*) n,count(DISTINCT article_id) u,
            count(DISTINCT champion_rank) dr,min(champion_rank) lo,max(champion_rank) hi
            FROM read_parquet('{sql(WV3_BASELINE)}') GROUP BY 1)
            SELECT count(*),sum(n),count(*) FILTER(WHERE n<>50 OR n<>u OR dr<>50 OR lo<>1 OR hi<>50),
            (SELECT count(*) FROM read_parquet('{sql(WV3_BASELINE)}') WHERE ap_rf<=7 AND champion_rank<>ap_rf)
            FROM g""").fetchone()
    result = {"cutoff": CUTOFF, "users": int(stats[0]), "rows": int(stats[1]),
              "invalid_groups": int(stats[2]), "protected_head_changes": int(stats[3]),
              "swap_rows": len(swaps), "selected_users": int(swaps.customer_id.nunique()),
              "users_with_two_swaps": int((swaps.groupby("customer_id").size() == 2).sum()),
              "labels_read": False, "seconds": time.perf_counter() - started}
    if result["users"] != TOTAL_USERS or result["rows"] != TOTAL_USERS * 50 or result["invalid_groups"] or result["protected_head_changes"]:
        raise AssertionError("WV3-741 baseline invariants failed")
    write(marker, result)
    progress(f"WV3-741 actions frozen: {len(swaps):,} swaps for {result['selected_users']:,} users")
    return result


def configure_final_mve() -> dict[str, Any]:
    """Point the frozen MIND implementation at isolated final-cutoff assets."""
    def guard(value: str) -> date:
        if value != CUTOFF:
            raise ValueError("final confirmation adapter accepts only the frozen cutoff")
        return date.fromisoformat(value)
    mve._guard = guard
    mve.candidate_path = lambda value: EXPANDED if value == CUTOFF else Path("__forbidden__")
    mve._source_manifest = lambda value: SOURCE_ROOT / "source-manifest.json" if value == CUTOFF else Path("__forbidden__")
    mve.ARTIFACT_ROOT = MIND_ROOT
    return {"adapter": "isolated final cutoff", "candidate_path": str(EXPANDED),
            "source_manifest": str(SOURCE_ROOT / "source-manifest.json"),
            "history_filter": f"t_dat < {CUTOFF}"}


def build_mind_retrieval() -> dict[str, Any]:
    marker = MIND_ROOT / CUTOFF / "mind_learned_age" / "RETRIEVAL.json"
    configure_final_mve()
    contract = read(REPORT / "MIND_WARM_MVE_CONTRACT.json")
    generated, evidence = mve.retrieve_variant(CUTOFF, "mind_learned_age", contract, "cuda")
    if generated != MIND_ROOT / CUTOFF / "mind_learned_age" / "candidates.parquet":
        raise AssertionError("MIND final adapter wrote outside its isolated artifact root")
    if not marker.is_file() or evidence.get("candidate_identity_unique") is not True:
        raise AssertionError("MIND retrieval did not complete its frozen contract")
    progress(f"MIND retrieval complete: {evidence['candidate_rows']:,} rows")
    return evidence


def build_mind_only() -> dict[str, Any]:
    marker = MIND_ONLY.with_suffix(".json")
    if MIND_ONLY.is_file() and marker.is_file():
        return read(marker)
    if MIND_ONLY.exists() or marker.exists():
        raise FileExistsError("preserve incomplete MIND-only cache")
    retrieval = build_mind_retrieval()
    source = MIND_ROOT / CUTOFF / "mind_learned_age" / "candidates.parquet"
    with connection("5GB") as db:
        db.execute(f"""COPY (SELECT m.* FROM read_parquet('{sql(source)}') m
            ANTI JOIN read_parquet('{sql(EXPANDED)}') b USING(customer_id,article_id)
            WHERE m.interest_count=3 ORDER BY customer_id,mind_rank,article_id)
            TO '{sql(MIND_ONLY)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)""")
        stats = db.execute(f"SELECT count(*),count(DISTINCT(customer_id,article_id)),count(DISTINCT customer_id),min(mind_rank),max(mind_rank) FROM read_parquet('{sql(MIND_ONLY)}')").fetchone()
    result = {"cutoff": CUTOFF, "rows": int(stats[0]), "unique_rows": int(stats[1]),
              "users": int(stats[2]), "min_raw_mind_rank": int(stats[3]),
              "max_raw_mind_rank": int(stats[4]), "interest_count_gate": 3,
              "labels_read": False, "retrieval": retrieval}
    if result["rows"] != result["unique_rows"]:
        raise AssertionError("MIND-only identities are not unique")
    write(marker, result)
    progress(f"P017 early gate retained {result['rows']:,} MIND-only rows")
    return result


def build_aggregate_features(keys_path: Path, output: Path) -> dict[str, Any]:
    """Cutoff-safe 84-day user/item/category aggregates for arbitrary keys."""
    started = time.perf_counter()
    with connection("7GB") as db:
        db.execute(f"CREATE VIEW candidates AS SELECT * FROM read_parquet('{sql(keys_path)}')")
        db.execute("CREATE TEMP TABLE users AS SELECT DISTINCT customer_id FROM candidates")
        db.execute("CREATE TEMP TABLE items AS SELECT DISTINCT article_id FROM candidates")
        db.execute(f"""CREATE TEMP TABLE articles AS SELECT article_id,
            try_cast(product_code AS INTEGER) article_product_code,
            try_cast(product_type_no AS INTEGER) article_product_type_no,
            try_cast(garment_group_no AS INTEGER) article_garment_group_no,
            try_cast(department_no AS INTEGER) article_department_no,
            try_cast(index_group_no AS INTEGER) article_index_group_no,
            try_cast(perceived_colour_master_id AS INTEGER) article_colour_master_id
            FROM read_csv_auto('{sql(RAW / 'articles.csv')}',header=true,all_varchar=true)""")
        db.execute(f"""CREATE TEMP TABLE customers AS SELECT customer_id,try_cast(age AS DOUBLE) customer_age
            FROM read_csv_auto('{sql(RAW / 'customers.csv')}',header=true,all_varchar=true)""")
        db.execute(f"""CREATE TEMP TABLE global_history AS SELECT * FROM read_parquet('{sql(TX)}')
            WHERE t_dat>=DATE '{CUTOFF}'-INTERVAL 84 DAY AND t_dat<DATE '{CUTOFF}'""")
        db.execute("""CREATE TEMP TABLE user_history AS SELECT h.*,a.article_product_code,a.article_product_type_no,
            a.article_garment_group_no,a.article_department_no,a.article_index_group_no,a.article_colour_master_id
            FROM global_history h SEMI JOIN users USING(customer_id) JOIN articles a USING(article_id)""")
        db.execute(f"""CREATE TEMP TABLE user_features AS SELECT u.customer_id,count(h.article_id)::BIGINT user_history_events_12w,
            count(DISTINCT h.article_id)::BIGINT user_unique_items_12w,avg(h.price)::DOUBLE user_avg_price_12w,
            avg((h.sales_channel_id=2)::INTEGER)::DOUBLE user_online_share_12w,
            date_diff('day',max(h.t_dat),DATE '{CUTOFF}')::BIGINT user_days_since_last_purchase
            FROM users u LEFT JOIN user_history h USING(customer_id) GROUP BY 1""")
        db.execute(f"""CREATE TEMP TABLE item_features AS SELECT i.article_id,
            count(h.article_id) FILTER(WHERE h.t_dat>=DATE '{CUTOFF}'-INTERVAL 7 DAY)::BIGINT item_events_7d,
            count(h.article_id) FILTER(WHERE h.t_dat>=DATE '{CUTOFF}'-INTERVAL 28 DAY)::BIGINT item_events_28d,
            count(h.article_id)::BIGINT item_events_12w,
            count(DISTINCT h.customer_id) FILTER(WHERE h.t_dat>=DATE '{CUTOFF}'-INTERVAL 28 DAY)::BIGINT item_unique_customers_28d,
            avg(h.price) FILTER(WHERE h.t_dat>=DATE '{CUTOFF}'-INTERVAL 28 DAY)::DOUBLE item_avg_price_28d,
            date_diff('day',max(h.t_dat),DATE '{CUTOFF}')::BIGINT item_days_since_last_sale
            FROM items i LEFT JOIN global_history h USING(article_id) GROUP BY 1""")
        dims = {"ui":"article_id", "upc":"article_product_code", "upt":"article_product_type_no",
                "udp":"article_department_no", "ugg":"article_garment_group_no", "ucl":"article_colour_master_id"}
        for table, key in dims.items():
            db.execute(f"""CREATE TEMP TABLE {table} AS SELECT customer_id,{key},
                count(*)::BIGINT events_12w,date_diff('day',max(t_dat),DATE '{CUTOFF}')::BIGINT days_since
                FROM user_history GROUP BY customer_id,{key}""")
        query = f"""SELECT c.*,cust.customer_age,(cust.customer_age IS NULL)::UTINYINT customer_age_missing,
            coalesce(floor(cust.customer_age/5),-1)::INTEGER age_bucket,
            coalesce(uf.user_history_events_12w,0)::BIGINT user_history_events_12w,
            coalesce(uf.user_unique_items_12w,0)::BIGINT user_unique_items_12w,
            uf.user_avg_price_12w,uf.user_online_share_12w,uf.user_days_since_last_purchase,
            coalesce(it.item_events_7d,0)::BIGINT item_events_7d,coalesce(it.item_events_28d,0)::BIGINT item_events_28d,
            coalesce(it.item_events_12w,0)::BIGINT item_events_12w,
            coalesce(it.item_unique_customers_28d,0)::BIGINT item_unique_customers_28d,
            it.item_avg_price_28d,it.item_days_since_last_sale,
            (coalesce(it.item_events_7d,0)+1.0)/(coalesce(it.item_events_28d,0)/4.0+1.0) item_trend_7d_vs_28d,
            abs(it.item_avg_price_28d-uf.user_avg_price_12w) user_item_price_gap,
            coalesce(ui.events_12w,0)::BIGINT user_item_events_12w,ui.days_since user_item_days_since_last_purchase,
            coalesce(upc.events_12w,0)::BIGINT user_product_code_events_12w,upc.days_since user_product_code_days_since,
            coalesce(upc.events_12w,0)/greatest(uf.user_history_events_12w,1.0) user_product_code_share_12w,
            coalesce(upt.events_12w,0)/greatest(uf.user_history_events_12w,1.0) user_product_type_share_12w,
            coalesce(udp.events_12w,0)/greatest(uf.user_history_events_12w,1.0) user_department_share_12w,
            coalesce(ugg.events_12w,0)/greatest(uf.user_history_events_12w,1.0) user_garment_share_12w,
            coalesce(ucl.events_12w,0)/greatest(uf.user_history_events_12w,1.0) user_colour_share_12w,
            a.article_product_code,a.article_product_type_no,a.article_garment_group_no,a.article_department_no,
            a.article_index_group_no,a.article_colour_master_id
            FROM candidates c JOIN articles a USING(article_id) LEFT JOIN customers cust USING(customer_id)
            LEFT JOIN user_features uf USING(customer_id) LEFT JOIN item_features it USING(article_id)
            LEFT JOIN ui USING(customer_id,article_id)
            LEFT JOIN upc ON c.customer_id=upc.customer_id AND a.article_product_code=upc.article_product_code
            LEFT JOIN upt ON c.customer_id=upt.customer_id AND a.article_product_type_no=upt.article_product_type_no
            LEFT JOIN udp ON c.customer_id=udp.customer_id AND a.article_department_no=udp.article_department_no
            LEFT JOIN ugg ON c.customer_id=ugg.customer_id AND a.article_garment_group_no=ugg.article_garment_group_no
            LEFT JOIN ucl ON c.customer_id=ucl.customer_id AND a.article_colour_master_id=ucl.article_colour_master_id"""
        db.execute(f"COPY ({query} ORDER BY customer_id,article_id) TO '{sql(output)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)")
        stats = db.execute(f"SELECT count(*),count(DISTINCT(customer_id,article_id)),count(DISTINCT customer_id) FROM read_parquet('{sql(output)}')").fetchone()
        latest = db.execute("SELECT max(t_dat) FROM global_history").fetchone()[0]
    result = {"rows": int(stats[0]), "unique_rows": int(stats[1]), "users": int(stats[2]),
              "latest_history_date": str(latest), "history_strictly_before_cutoff": str(latest) < CUTOFF,
              "labels_read": False, "seconds": time.perf_counter() - started}
    if result["rows"] != result["unique_rows"] or not result["history_strictly_before_cutoff"]:
        raise AssertionError("aggregate-feature identity or temporal boundary failed")
    return result


def build_mind_features() -> dict[str, Any]:
    marker = MIND_FEATURES.with_suffix(".json")
    if MIND_FEATURES.is_file() and marker.is_file() and MIND_DENSE.is_file():
        return read(marker)
    if any(path.exists() for path in (MIND_FEATURES, MIND_DENSE, marker)):
        raise FileExistsError("preserve incomplete MIND feature cache")
    build_mind_only(); resource_gate()
    aggregate = build_aggregate_features(MIND_ONLY, MIND_FEATURES)
    configure_final_mve()
    with connection("3GB") as db:
        keys = db.execute(f"SELECT customer_id,article_id FROM read_parquet('{sql(MIND_ONLY)}') ORDER BY customer_id,article_id").fetchdf()
    values, dense_evidence = dense.build_dense_features(CUTOFF, keys, "cuda")
    parquet(values, MIND_DENSE)
    result = {"cutoff": CUTOFF, "aggregate": aggregate, "dense": dense_evidence,
              "rows": len(keys), "labels_read": False}
    write(marker, result)
    progress(f"MIND aggregate+dense features complete: {len(keys):,} rows")
    return result


def score_mind_winners() -> dict[str, Any]:
    marker = MIND_ORDER.with_suffix(".json")
    if MIND_ORDER.is_file() and marker.is_file():
        return read(marker)
    if MIND_ORDER.exists() or marker.exists():
        raise FileExistsError("preserve incomplete P005 ordering cache")
    started = time.perf_counter(); build_mind_features()
    model_meta = read(P005_META)
    model = lgb.Booster(model_file=str(P005_MODEL))
    if model.feature_name() != candidate.FEATURES or model_meta.get("features") != candidate.FEATURES:
        raise AssertionError("P005 candidate model feature contract drift")
    stage = ART / "p005-order-build.duckdb"
    if stage.exists():
        raise FileExistsError("preserve incomplete P005 order database")
    target = connection("4GB", stage); source = duckdb.connect(); rows = batches = 0
    try:
        cursor = source.execute(f"""SELECT f.customer_id,f.article_id,f.mind_rank,f.interest_count,
            {','.join('f.' + x for x in relation.USER + relation.ITEM)},
            {','.join('d.' + x for x in dense.DENSE_FEATURES)}
            FROM read_parquet('{sql(MIND_FEATURES)}') f JOIN read_parquet('{sql(MIND_DENSE)}') d USING(customer_id,article_id)
            ORDER BY f.customer_id,f.article_id""")
        initialized = False
        while True:
            frame = cursor.fetch_df_chunk(24)
            if frame.empty:
                break
            matrix = frame[candidate.FEATURES].to_numpy(np.float32); matrix[~np.isfinite(matrix)] = np.nan
            out = frame[["customer_id", "article_id", "mind_rank", "interest_count"]].copy()
            out["score"] = model.predict(matrix, num_threads=4)
            target.register("batch", out)
            if not initialized:
                target.execute("CREATE TABLE candidate_scores AS SELECT * FROM batch WHERE FALSE"); initialized = True
            target.execute("INSERT INTO candidate_scores SELECT * FROM batch"); target.unregister("batch")
            rows += len(out); batches += 1
            if batches % 20 == 0:
                progress(f"P005 candidate scoring: {rows:,} rows")
            del frame, matrix, out; gc.collect()
        target.execute(f"""COPY (SELECT * FROM (SELECT *,row_number() OVER(
            PARTITION BY customer_id ORDER BY score DESC,article_id)::INTEGER rank FROM candidate_scores) x
            WHERE rank=1 ORDER BY customer_id)
            TO '{sql(MIND_ORDER)}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
        stats = target.execute(f"SELECT count(*),count(DISTINCT customer_id),count(*) FILTER(WHERE mind_rank<=50),count(*) FILTER(WHERE interest_count<>3) FROM read_parquet('{sql(MIND_ORDER)}')").fetchone()
    finally:
        source.close(); target.close()
    stage.unlink(missing_ok=True)
    result = {"cutoff": CUTOFF, "scored_rows": rows, "candidate_users": int(stats[0]),
              "unique_users": int(stats[1]), "winner_raw_mind_rank_le_50": int(stats[2]),
              "wrong_interest_count": int(stats[3]), "features": candidate.FEATURES,
              "labels_read": False, "seconds": time.perf_counter() - started}
    if result["candidate_users"] != result["unique_users"] or result["wrong_interest_count"]:
        raise AssertionError("P005 winner conservation failed")
    write(marker, result)
    progress(f"P005 winner ordering complete: {result['candidate_users']:,} users")
    return result


def build_pair_features() -> dict[str, Any]:
    marker = PAIR_FEATURES.with_suffix(".json")
    required_outputs = (PAIR_FEATURES, TAIL_KEYS, TAIL_DENSE, marker)
    if all(path.is_file() for path in required_outputs):
        return read(marker)
    if any(path.exists() for path in required_outputs):
        raise FileExistsError("preserve incomplete P005/P006 pair-feature cache")
    started = time.perf_counter(); score_mind_winners(); build_wv3_baseline()
    # P018 is a post-P017 raw-rank gate.  Applying both frozen gates before
    # expensive tail feature generation is action-equivalent and saves work.
    with connection("5GB") as db:
        db.execute(f"""COPY (SELECT b.customer_id,b.article_id FROM read_parquet('{sql(WV3_BASELINE)}') b
            SEMI JOIN (SELECT customer_id FROM read_parquet('{sql(MIND_ORDER)}') WHERE mind_rank<=50) w USING(customer_id)
            WHERE b.champion_rank BETWEEN 8 AND 12 ORDER BY b.customer_id,b.champion_rank)
            TO '{sql(TAIL_KEYS)}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
        tail_stats = db.execute(f"SELECT count(*),count(DISTINCT customer_id) FROM read_parquet('{sql(TAIL_KEYS)}')").fetchone()
    if int(tail_stats[0]) != 5 * int(tail_stats[1]):
        raise AssertionError("every gated winner must retain five WV3-741 tail victims")
    configure_final_mve()
    with connection("2GB") as db:
        tail_keys = db.execute(f"SELECT customer_id,article_id FROM read_parquet('{sql(TAIL_KEYS)}') ORDER BY customer_id,article_id").fetchdf()
    tail_dense, tail_dense_evidence = dense.build_dense_features(CUTOFF, tail_keys, "cuda")
    parquet(tail_dense, TAIL_DENSE)
    select = [
        "c.customer_id", "c.article_id AS challenger_article_id",
        "v.article_id AS victim_article_id", "b.champion_rank::INTEGER AS champion_rank",
        *[f"c.{name}::FLOAT AS {name}" for name in relation.USER],
    ]
    for name in [*relation.ITEM, *dense.DENSE_FEATURES]:
        c_table = "cd" if name in dense.DENSE_FEATURES else "c"
        v_table = "vd" if name in dense.DENSE_FEATURES else "v"
        select += [f"{c_table}.{name}::FLOAT AS c_{name}", f"{v_table}.{name}::FLOAT AS v_{name}",
                   f"({c_table}.{name}-{v_table}.{name})::FLOAT AS diff_{name}"]
    select += [f"(c.{name}=v.{name})::FLOAT AS same_{name}" for name in relation.CATEGORIES]
    select += [f"b.{name}::FLOAT AS v_{name}" for name in relation.VICTIM]
    with connection("6GB") as db:
        db.execute(f"""COPY (SELECT {','.join(select)}
            FROM (SELECT * FROM read_parquet('{sql(MIND_FEATURES)}')) c
            JOIN read_parquet('{sql(MIND_DENSE)}') cd USING(customer_id,article_id)
            JOIN (SELECT * FROM read_parquet('{sql(MIND_ORDER)}') WHERE mind_rank<=50) w USING(customer_id,article_id)
            JOIN read_parquet('{sql(WV3_BASELINE)}') b ON b.customer_id=c.customer_id AND b.champion_rank BETWEEN 8 AND 12
            JOIN read_parquet('{sql(BASE_FEATURES)}') v ON v.customer_id=b.customer_id AND v.article_id=b.article_id
            JOIN read_parquet('{sql(TAIL_DENSE)}') vd ON vd.customer_id=v.customer_id AND vd.article_id=v.article_id
            ORDER BY c.customer_id,b.champion_rank,c.article_id)
            TO '{sql(PAIR_FEATURES)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 250000)""")
        stats = db.execute(f"""SELECT count(*),count(DISTINCT(customer_id,challenger_article_id,victim_article_id)),
            count(DISTINCT customer_id),min(champion_rank),max(champion_rank)
            FROM read_parquet('{sql(PAIR_FEATURES)}')""").fetchone()
        winners = db.execute(f"SELECT count(*) FROM read_parquet('{sql(MIND_ORDER)}') WHERE mind_rank<=50").fetchone()[0]
    result = {"cutoff": CUTOFF, "rows": int(stats[0]), "unique_rows": int(stats[1]),
              "winner_users_after_interest_and_rank_gates": int(stats[2]),
              "winner_rows_expected": int(winners), "victim_rank_range": [int(stats[3]), int(stats[4])],
              "relation_features": relation.FEATURES["dense_primary"], "tail_dense": tail_dense_evidence,
              "optimization": "P017 interest-count and P018 raw-MIND-rank gates pushed before pair expansion; exact frozen actions unchanged",
              "labels_read": False, "seconds": time.perf_counter() - started}
    if result["rows"] != result["unique_rows"] or result["rows"] != 5 * result["winner_rows_expected"] or result["victim_rank_range"] != [8, 12]:
        raise AssertionError("winner-by-five-victim pair product failed")
    write(marker, result)
    progress(f"P005/P006 pair features complete: {result['rows']:,} rows")
    return result


def freeze_p018_actions() -> dict[str, Any]:
    if ACTION_MANIFEST.is_file() and P018_ACTIONS.is_file():
        saved = read(ACTION_MANIFEST)
        if saved.get("actions_frozen") is not True or saved.get("cutoff") != CUTOFF:
            raise ValueError("invalid final action manifest")
        return saved
    if any(path.exists() for path in (ACTION_MANIFEST, P005_ACTIONS, P006_ACTIONS, P018_ACTIONS)):
        raise FileExistsError("preserve incomplete final action set")
    started = time.perf_counter(); pairs_meta = build_pair_features(); resource_gate()
    p005_meta = read(P005_META); relation_meta = read(RELATION_META)
    purchase_model = lgb.Booster(model_file=str(P005_MODEL))
    relation_model = lgb.Booster(model_file=str(RELATION_MODEL))
    rel_features = relation.FEATURES["dense_primary"]
    if purchase_model.feature_name() != candidate.FEATURES or relation_model.feature_name() != rel_features:
        raise AssertionError("frozen P005 or relation model schema drift")
    if p005_meta.get("training") != ["2020-01-22", "2020-03-18", "2020-06-24"]:
        raise AssertionError("unexpected P005 temporal provenance")
    if relation_meta.get("training") != ["2020-01-22", "2020-03-18", "2020-06-24"]:
        raise AssertionError("unexpected relation-model temporal provenance")
    with connection("5GB") as db:
        frame = db.execute(f"SELECT {','.join(relation.KEYS + rel_features)} FROM read_parquet('{sql(PAIR_FEATURES)}') ORDER BY customer_id,champion_rank").fetchdf()
        winner_scores = db.execute(f"SELECT customer_id,article_id AS challenger_article_id,score candidate_purchase_score FROM read_parquet('{sql(MIND_ORDER)}') WHERE mind_rank<=50").fetchdf()
    matrix = frame[rel_features].to_numpy(np.float32); matrix[~np.isfinite(matrix)] = np.nan
    probabilities = relation_model.predict(matrix, num_threads=4)
    if probabilities.shape != (len(frame), 3) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("invalid P005 relation probabilities")
    relation_scores = frame[relation.KEYS].copy()
    relation_scores[relation.PROBS] = probabilities
    p005 = relation.select_actions(relation_scores)
    parquet(p005, P005_ACTIONS)
    eligible = frame[frame.v_user_item_events_12w == 0].copy()
    victim_columns = [name if name in relation.USER else "v_" + name for name in candidate.FEATURES]
    victim_matrix = eligible[victim_columns].to_numpy(np.float32); victim_matrix[~np.isfinite(victim_matrix)] = np.nan
    p006_scores = eligible[relation.KEYS].merge(winner_scores, on=["customer_id", "challenger_article_id"], validate="many_to_one")
    p006_scores["victim_purchase_score"] = purchase_model.predict(victim_matrix, num_threads=4)
    p006 = shared.select(p006_scores[shared.SCORE_COLUMNS])
    parquet(p006, P006_ACTIONS)
    action_keys = relation.KEYS
    p018 = p005.merge(p006[action_keys], on=action_keys, how="inner", validate="one_to_one")
    p018 = p018.merge(
        winner_scores.rename(columns={"candidate_purchase_score": "p005_candidate_purchase_score"}),
        on=["customer_id", "challenger_article_id"], validate="one_to_one",
    ).sort_values("customer_id", kind="mergesort").reset_index(drop=True)
    parquet(p018, P018_ACTIONS)
    forbidden = {"target", "truth_count", "challenger_target", "victim_target", "actual_delta"}
    if forbidden & set(p018.columns) or p018.duplicated("customer_id").any() or not p018.champion_rank.between(8, 12).all():
        raise AssertionError("P018 frozen actions violate identity, rank or label-firewall constraints")
    manifest = {
        "schema": "mind-warm-final001-action-manifest-v1", "cutoff": CUTOFF,
        "created_at": datetime.now(timezone.utc).isoformat(), "actions_frozen": True,
        "labels_read": False, "final_week_status": "actions_frozen_labels_unopened",
        "p005_actions": len(p005), "p006_actions": len(p006), "p018_actions": len(p018),
        "p018_users": int(p018.customer_id.nunique()), "maximum_actions_per_user": 1,
        "victim_rank_range": ([int(p018.champion_rank.min()), int(p018.champion_rank.max())] if len(p018) else None),
        "forbidden_action_columns_absent": not bool(forbidden & set(p018.columns)),
        "p009_exact_intersection": True, "p017_interest_count": 3,
        "p018_raw_mind_rank_max": 50,
        "early_gate_optimization": pairs_meta["optimization"],
        "models_reused_without_refit": [str(P005_MODEL), str(RELATION_MODEL), str(WV3_POINT), str(WV3_LAMBDA)],
        "seconds": time.perf_counter() - started,
    }
    write(ACTION_MANIFEST, manifest)
    progress(f"P018 actions irrevocably frozen before labels: {len(p018):,}")
    return manifest


def prepare() -> dict[str, Any]:
    if PREPARED.is_file():
        saved = read(PREPARED)
        if saved.get("status") == "complete_actions_frozen" and read(ACTION_MANIFEST).get("actions_frozen"):
            return saved
    started = time.perf_counter()
    result: dict[str, Any] = {"schema": "mind-warm-final001-prepared-v1", "cutoff": CUTOFF,
                              "status": "running", "labels_read": False,
                              "started_at": datetime.now(timezone.utc).isoformat()}
    write(PREPARED, result)
    try:
        result["preflight"] = check_inputs(); write(PREPARED, result)
        source = build_item2vec_source(); result["item2vec"] = {"rows": source["audit"]["rows"]}; write(PREPARED, result)
        result["expanded"] = build_expanded_candidates(source); write(PREPARED, result)
        result["base_features"] = build_base_features(result["expanded"]); write(PREPARED, result)
        result["wv2_ranks"] = build_ranks(); write(PREPARED, result)
        result["wv3_741"] = build_wv3_baseline(); write(PREPARED, result)
        result["mind_retrieval"] = build_mind_retrieval(); write(PREPARED, result)
        result["actions"] = freeze_p018_actions()
        result["status"] = "complete_actions_frozen"
    except Exception as error:
        result["status"] = "engineering_stopped_before_label_read"
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        result["seconds"] = time.perf_counter() - started
        write(PREPARED, result)
    return result


def ap_from_matrix(target: np.ndarray, truth_count: np.ndarray) -> tuple[float, np.ndarray]:
    if target.ndim != 2 or target.shape[1] != 12 or truth_count.shape != (len(target),):
        raise ValueError("AP@12 expects one twelve-item list and one truth count per user")
    precision = np.cumsum(target, axis=1) / np.arange(1, 13, dtype=np.float64)
    per_user = (precision * target).sum(axis=1) / np.maximum(np.minimum(truth_count, 12), 1)
    return float(per_user.mean()), per_user


def independent_group_ap(frame: pd.DataFrame, users: pd.Index) -> float:
    ordered = frame.sort_values(["customer_id", "champion_rank", "article_id"], kind="mergesort").copy()
    ordered["hit_before"] = ordered.groupby("customer_id", sort=False).target.cumsum()
    ordered["term"] = ordered.target * ordered.hit_before / ordered.champion_rank
    truth = ordered.groupby("customer_id", sort=False).truth_count.first().clip(upper=12).clip(lower=1)
    values = ordered.groupby("customer_id", sort=False).term.sum() / truth
    return float(values.reindex(users, fill_value=0.0).mean())


def render_final(result: dict[str, Any]) -> str:
    gate = result["promotion_gate"]
    outcome = "通过" if gate["passed"] else ("证据不足" if result["delta_MAP@12"] == 0 else "未通过")
    return "\n".join([
        "# MIND-WARM-FINAL-001：P018 最终周一次性时间确认", "",
        "## 结论", "",
        f"最终确认门槛：**{outcome}**。冻结 WV3-741 的 MAP@12 为 `{result['baseline_MAP@12']:.12f}`，",
        f"P018 为 `{result['P018_MAP@12']:.12f}`，绝对增量 `{result['delta_MAP@12']:+.12f}`。",
        "无论结果如何，本周结果不用于修改兴趣数、MIND 原始名次门、模型、候选预算或换位位置。", "",
        "## 术语与证据边界", "",
        "- MAP@12（行业通用离线排序指标）：对每名用户前12个推荐按命中位置累计精度，再除以该用户 `min(真实购买商品数,12)`，最后对固定68,984名用户等权平均。",
        "- WV3-741（本项目冻结 Warm 基线）：两个历史监督 LightGBM 排序分数先各自转成用户内名次，再以固定 RRF60 等权融合；最多把第13—50名中的两件商品与第8—12名换位，第1—7名受保护。",
        "- P018（本项目冻结准入策略）：P005 和 P006 必须对同一用户、同一 MIND 挑战商品、同一被替换商品和同一位置同时同意；用户必须有3个活跃兴趣，且挑战商品原始 MIND 名次不超过50。每用户最多一次替换。",
        "- 策略盲测（本项目证据分类）：P018 规则在未读取该周逐商品标签时冻结；但固定用户队列来自该周有购买的用户，历史报告也已含总体统计，所以不是完全数据盲测。",
        "- 有益、伤害与中性动作（本项目动作审计）：一次替换使该用户 AP@12 分别上升、下降或不变；三者分母均是冻结动作用户，不是全部用户。", "",
        "## 冻结与标签防火墙", "",
        f"- 标签打开前已冻结 `{result['actions']['p018_actions']:,}` 条 P018 动作；动作文件不含标签或标签派生字段。",
        "- 所有行为特征、Item2Vec、BPR 和 MIND 训练只使用 `t_dat < 2020-09-16`；标签区间仅为 `[2020-09-16, 2020-09-23)`。",
        "- 首次打开标签后，评测 SQL 因保留字别名缺少显式 `AS` 而在 MAP 计算前停止；随后只修复评测器语法，并对同一冻结动作文件重跑，没有产生或比较第二套策略。",
        "- 为节省本机计算，将已经冻结的 P017 三兴趣门和 P018 原始 MIND Top50 门提前到昂贵的尾部特征计算之前；这只删除最终必然不会行动的用户，不改变任何可保留动作。", "",
        "## 完整用户结果", "",
        "| 指标 | WV3-741 | P018 | 差值 |", "|---|---:|---:|---:|",
        f"| MAP@12 | {result['baseline_MAP@12']:.12f} | {result['P018_MAP@12']:.12f} | {result['delta_MAP@12']:+.12f} |", "",
        "## 动作结果", "",
        "| 冻结动作用户 | 有益 | 伤害 | 中性 |", "|---:|---:|---:|---:|",
        f"| {result['action_outcomes']['users']:,} | {result['action_outcomes']['beneficial']:,} | {result['action_outcomes']['harmful']:,} | {result['action_outcomes']['neutral']:,} |", "",
        f"两种独立 AP@12 实现的最大误差为 `{result['verification']['maximum_map_error']:.3g}`；第1—7名变化数为 `{result['verification']['protected_head_changes']}`，",
        f"最终每用户12件且无重复的异常用户数为 `{result['verification']['invalid_top12_users']}`。", "",
        "## 决策", "",
        ("P018 获得一次独立时间确认，可晋升为新的 Warm 时间确认基线。" if gate["passed"] else
         "P018 不晋升；继续保留 WV3-741 作为有效开发基线，并按预注册停止，不对最终周做补救调参。"), "",
    ])


def evaluate() -> dict[str, Any]:
    if METRICS.is_file() and FINAL_REPORT.is_file():
        saved = read(METRICS)
        if saved.get("status") in {"confirmed_promoted", "rejected_baseline_retained", "inconclusive_baseline_retained"}:
            return saved
    prepared = read(PREPARED) if PREPARED.is_file() else {}
    actions_meta = read(ACTION_MANIFEST) if ACTION_MANIFEST.is_file() else {}
    if prepared.get("status") != "complete_actions_frozen" or actions_meta.get("actions_frozen") is not True or actions_meta.get("labels_read") is not False:
        raise RuntimeError("final labels remain closed until the complete action manifest is frozen")
    started = time.perf_counter(); resource_gate()
    with connection("5GB") as db:
        db.execute(f"CREATE TEMP TABLE cohort AS SELECT DISTINCT customer_id FROM read_parquet('{sql(BASE100)}')")
        db.execute(f"""CREATE TEMP TABLE truth AS SELECT DISTINCT t.customer_id,t.article_id
            FROM read_parquet('{sql(TX)}') t SEMI JOIN cohort USING(customer_id)
            WHERE t.t_dat>=DATE '{CUTOFF}' AND t.t_dat<DATE '{END}'""")
        db.execute("CREATE TEMP TABLE truth_n AS SELECT customer_id,count(*)::INTEGER truth_count FROM truth GROUP BY 1")
        db.execute(f"""CREATE TEMP TABLE base AS SELECT b.customer_id,b.article_id,b.champion_rank,
            (t.article_id IS NOT NULL)::UTINYINT AS target,n.truth_count
            FROM read_parquet('{sql(WV3_BASELINE)}') b LEFT JOIN truth t USING(customer_id,article_id)
            JOIN truth_n n USING(customer_id) WHERE b.champion_rank<=12""")
        base = db.execute("SELECT * FROM base ORDER BY customer_id,champion_rank,article_id").fetchdf()
        actions = db.execute(f"SELECT * FROM read_parquet('{sql(P018_ACTIONS)}') ORDER BY customer_id").fetchdf()
        db.register("actions", actions)
        final = db.execute("""SELECT b.customer_id,coalesce(a.challenger_article_id,b.article_id) article_id,
            b.champion_rank,(t.article_id IS NOT NULL)::UTINYINT AS target,b.truth_count
            FROM base b LEFT JOIN actions a USING(customer_id,champion_rank)
            LEFT JOIN truth t ON t.customer_id=b.customer_id AND t.article_id=coalesce(a.challenger_article_id,b.article_id)
            ORDER BY b.customer_id,b.champion_rank,article_id""").fetchdf()
        truth_stats = db.execute("SELECT count(*),count(DISTINCT customer_id) FROM truth").fetchone()
    users = pd.Index(base.customer_id.drop_duplicates())
    if len(users) != TOTAL_USERS or len(base) != TOTAL_USERS * 12 or len(final) != len(base):
        raise AssertionError("final evaluation did not retain the complete fixed cohort")
    truth_count = base.groupby("customer_id", sort=False).truth_count.first().reindex(users).to_numpy(np.int32)
    base_target = base.target.to_numpy(np.uint8).reshape(TOTAL_USERS, 12)
    final_target = final.target.to_numpy(np.uint8).reshape(TOTAL_USERS, 12)
    baseline_map, baseline_user = ap_from_matrix(base_target, truth_count)
    final_map, final_user = ap_from_matrix(final_target, truth_count)
    independent_base = independent_group_ap(base, users)
    independent_final = independent_group_ap(final, users)
    maximum_error = max(abs(baseline_map - independent_base), abs(final_map - independent_final))
    invalid = int((final.groupby("customer_id").size() != 12).sum() + final.duplicated(["customer_id", "article_id"]).groupby(final.customer_id).any().sum())
    protected = int((base[base.champion_rank <= 7].article_id.to_numpy() != final[final.champion_rank <= 7].article_id.to_numpy()).sum())
    indexer = users.get_indexer(actions.customer_id)
    if np.any(indexer < 0):
        raise AssertionError("a frozen action user is outside the evaluation cohort")
    action_delta = final_user[indexer] - baseline_user[indexer]
    beneficial = action_delta > 1e-15; harmful = action_delta < -1e-15
    delta = final_map - baseline_map
    passed = bool(delta > 0 and maximum_error <= 1e-12 and invalid == 0 and protected == 0)
    if delta > 0:
        status = "confirmed_promoted" if passed else "engineering_verification_failed"
    elif delta == 0:
        status = "inconclusive_baseline_retained"
    else:
        status = "rejected_baseline_retained"
    result = {
        "schema": "mind-warm-final001-metrics-v1", "experiment_id": "MIND-WARM-FINAL-001",
        "status": status, "evidence_classification": "policy-blind temporal confirmation; not pristine dataset-blind",
        "cutoff": CUTOFF, "end_exclusive": END, "total_users": TOTAL_USERS,
        "truth_pairs": int(truth_stats[0]), "truth_users": int(truth_stats[1]),
        "baseline_MAP@12": baseline_map, "P018_MAP@12": final_map, "delta_MAP@12": delta,
        "actions": actions_meta,
        "action_outcomes": {"users": len(actions), "beneficial": int(beneficial.sum()),
                            "harmful": int(harmful.sum()), "neutral": int((~(beneficial | harmful)).sum()),
                            "gross_positive_MAP": float(action_delta[beneficial].sum() / TOTAL_USERS),
                            "gross_negative_MAP": float(action_delta[harmful].sum() / TOTAL_USERS)},
        "promotion_gate": {"passed": passed, "strictly_positive_delta": delta > 0,
                           "no_rescue_or_second_variant": True},
        "verification": {"independent_baseline_MAP@12": independent_base,
                         "independent_P018_MAP@12": independent_final,
                         "maximum_map_error": maximum_error, "invalid_top12_users": invalid,
                         "protected_head_changes": protected, "exactly_one_final_exposure": True},
        "no_post_result_tuning": True, "seconds": time.perf_counter() - started,
        "evaluation_engineering_retry": {"count": 1, "reason": "DuckDB target alias required explicit AS",
                                         "actions_or_policy_changed": False, "first_attempt_produced_metrics": False},
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if maximum_error > 1e-12 or invalid or protected:
        result["status"] = "engineering_verification_failed"
        write(METRICS, result)
        raise AssertionError("independent AP or served-list invariants failed")
    write(METRICS, result)
    FINAL_REPORT.write_text(render_final(result), encoding="utf-8")
    actions_meta.update(labels_read=True, final_week_status="evaluated_once", evaluation_metrics=str(METRICS),
                        evaluated_at=result["completed_at"])
    write(ACTION_MANIFEST, actions_meta)
    progress(f"FINAL one-shot result: WV3-741={baseline_map:.12f}, P018={final_map:.12f}, delta={delta:+.12f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "run", "preflight"))
    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(native(check_inputs()), ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        print(json.dumps(native(prepare()), ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        print(json.dumps(native(evaluate()), ensure_ascii=False, indent=2))
    else:
        prepare(); print(json.dumps(native(evaluate()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
