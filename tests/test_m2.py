import json
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from hm_recsys.m1 import SOURCES
from hm_recsys.m2 import (
    M2Config,
    _create_static_dimensions,
    build_point_in_time_dataset,
    evaluate_predictions,
    validate_candidate_artifact,
)


def _candidate_frame() -> pd.DataFrame:
    rows = []
    for article_id, rank in (("0000000001", 1), ("0000000002", 2)):
        row = {
            "customer_id": "u1",
            "article_id": article_id,
            "candidate_rank": rank,
            "fused_score": 0.2 / rank,
            "source_count": 1,
            "sources": "recent_popularity",
        }
        for source in SOURCES:
            present = int(source == "recent_popularity")
            row[f"{source}_present"] = present
            row[f"{source}_rank"] = rank if present else None
            row[f"{source}_score"] = 1.0 / rank if present else None
            row[f"{source}_rrf_contribution"] = 1.0 / (60 + rank) if present else None
        rows.append(row)
    return pd.DataFrame(rows)


class M2Tests(unittest.TestCase):
    def test_candidate_manifest_fails_closed_on_artifact_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate = root / "candidate.parquet"
            connection = duckdb.connect()
            try:
                frame = _candidate_frame()
                connection.register("frame", frame)
                connection.execute(
                    f"COPY frame TO '{candidate.as_posix()}' (FORMAT PARQUET)"
                )
            finally:
                connection.close()
            manifest = root / "manifest.json"
            payload = {
                "schema_version": "m1.5-wide-v1",
                "cutoff": "2020-09-01",
                "catalog_protocol": "optimistic_all_articles",
                "sample_rate": 1.0,
                "sampled_user_fingerprint": "abc",
                "config": {"fusion_profile": "collaborative", "candidate_k": 2},
                "wide_evidence": {"rows": 2},
                "artifacts": {
                    "candidate_features": str(candidate.resolve()),
                    "candidate_features_bytes": candidate.stat().st_size,
                },
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            identity = validate_candidate_artifact(candidate, manifest, 2)
            self.assertEqual(identity["cutoff"], "2020-09-01")
            payload["artifacts"]["candidate_features"] = str(root / "other.parquet")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not point"):
                validate_candidate_artifact(candidate, manifest, 2)

    def test_point_in_time_features_exclude_target_week_behavior(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidate_path = root / "candidates.parquet"
            output_path = root / "features.parquet"
            connection = duckdb.connect()
            try:
                candidate = _candidate_frame()
                connection.register("candidate_frame", candidate)
                connection.execute(
                    f"COPY candidate_frame TO '{candidate_path.as_posix()}' (FORMAT PARQUET)"
                )
                connection.execute(
                    """
                    CREATE TABLE transactions(
                        t_dat DATE, customer_id VARCHAR, article_id VARCHAR,
                        price DOUBLE, sales_channel_id UTINYINT
                    );
                    INSERT INTO transactions VALUES
                    ('2020-08-30', 'u1', '0000000001', 0.1, 2),
                    ('2020-09-02', 'u1', '0000000002', 0.2, 1),
                    ('2020-09-03', 'u1', '0000000003', 9.9, 1);
                    CREATE TABLE articles(
                        article_id VARCHAR, product_code VARCHAR, product_type_no VARCHAR,
                        garment_group_no VARCHAR, department_no VARCHAR,
                        index_group_no VARCHAR, perceived_colour_master_id VARCHAR
                    );
                    INSERT INTO articles VALUES
                    ('0000000001','1','1','1','1','1','1'),
                    ('0000000002','2','2','2','2','2','2'),
                    ('0000000003','3','3','3','3','3','3');
                    CREATE TABLE customers(customer_id VARCHAR, age VARCHAR);
                    INSERT INTO customers VALUES ('u1', '25');
                    """
                )
                _create_static_dimensions(connection)
                identity = {
                    "cutoff": "2020-09-01",
                    "declared_rows": 2,
                }
                evidence = build_point_in_time_dataset(
                    connection,
                    candidate_path,
                    identity,
                    output_path,
                    M2Config(candidate_k=2, metric_k=2),
                )
                rows = connection.execute(
                    f"""
                    SELECT article_id, user_history_events_12w, user_avg_price_12w, target
                    FROM read_parquet('{output_path.as_posix()}') ORDER BY article_id
                    """
                ).fetchall()
            finally:
                connection.close()
            self.assertEqual(rows[0], ("0000000001", 1, 0.1, 0))
            self.assertEqual(rows[1], ("0000000002", 1, 0.1, 1))
            self.assertEqual(evidence["latest_behavior_date"], "2020-08-30")
            self.assertEqual(evidence["truth_pairs"], 2)
            self.assertEqual(evidence["positives"], 1)

    def test_evaluation_uses_hm_map_denominator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = root / "evaluation.duckdb"
            transactions = root / "transactions.parquet"
            connection = duckdb.connect(str(database))
            try:
                connection.execute(
                    """
                    CREATE TABLE predictions(
                        customer_id VARCHAR, article_id VARCHAR, candidate_rank BIGINT,
                        target UTINYINT, user_history_events_12w BIGINT,
                        score_retrieval DOUBLE, score_full DOUBLE
                    );
                    INSERT INTO predictions VALUES
                    ('u1','a1',1,0,3,0.1,0.1),
                    ('u1','a2',2,1,3,0.9,0.9),
                    ('u2','b1',1,1,0,0.9,0.9),
                    ('u2','b2',2,0,0,0.1,0.1);
                    CREATE TABLE tx(t_dat DATE, customer_id VARCHAR, article_id VARCHAR,
                                    price DOUBLE, sales_channel_id UTINYINT);
                    INSERT INTO tx VALUES
                    ('2020-08-31','u1','a1',0.1,1),
                    ('2020-09-02','u1','a2',0.1,1),
                    ('2020-09-02','u2','b1',0.1,1);
                    """
                )
                connection.execute(
                    f"COPY tx TO '{transactions.as_posix()}' (FORMAT PARQUET)"
                )
            finally:
                connection.close()
            result = evaluate_predictions(database, transactions, "2020-09-01", 2)
            self.assertAlmostEqual(
                result["orderings"]["rrf"]["segments"]["overall"]["map@2"], 0.75
            )
            self.assertAlmostEqual(
                result["orderings"]["lightgbm_full"]["segments"]["overall"]["map@2"], 1.0
            )


if __name__ == "__main__":
    unittest.main()
