import json
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from hm_recsys.m1 import M1Config, run_m1
from hm_recsys.m15 import (
    CacheValidationError,
    _canonicalize_source,
    _create_collaborative_fusion,
    run_m15,
    validate_cache_manifest,
)


def _write_synthetic_raw(raw: Path) -> None:
    raw.mkdir()
    pd.DataFrame(
        [
            ("2020-08-30", "u1", "0000000001", 0.1, 2),
            ("2020-08-30", "u1", "0000000001", 0.1, 2),
            ("2020-08-30", "u1", "0000000002", 0.2, 2),
            ("2020-08-31", "u2", "0000000001", 0.1, 1),
            ("2020-08-31", "u2", "0000000003", 0.3, 1),
            ("2020-09-01", "u1", "0000000003", 0.3, 2),
            ("2020-09-02", "u3", "0000000001", 0.1, 1),
            ("2020-09-03", "u1", "0000000004", 0.4, 2),
        ],
        columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
    ).to_csv(raw / "transactions_train.csv", index=False)
    pd.DataFrame(
        [
            ("0000000001", "p1", "t1", "g1", "c1", "i1"),
            ("0000000002", "p1", "t1", "g1", "c2", "i1"),
            ("0000000003", "p2", "t2", "g2", "c1", "i1"),
            ("0000000004", "p2", "t2", "g2", "c2", "i1"),
            ("0000000005", "p3", "t3", "g3", "c3", "i2"),
        ],
        columns=[
            "article_id",
            "product_code",
            "product_type_no",
            "garment_group_no",
            "perceived_colour_master_id",
            "index_group_no",
        ],
    ).to_csv(raw / "articles.csv", index=False)
    pd.DataFrame(
        [("u1", 25), ("u2", 27), ("u3", None)],
        columns=["customer_id", "age"],
    ).to_csv(raw / "customers.csv", index=False)
    pd.DataFrame(
        [("u1", ""), ("u2", ""), ("u3", "")],
        columns=["customer_id", "prediction"],
    ).to_csv(raw / "sample_submission.csv", index=False)


def _config(protocol: str = "optimistic_all_articles") -> M1Config:
    return M1Config(
        cutoff="2020-09-01",
        sample_rate=1.0,
        min_covisit_count=1,
        source_k=10,
        candidate_k=10,
        covisit_neighbor_k=10,
        fusion_profile="collaborative",
        evaluation_mode="final",
        catalog_protocol=protocol,
    )


class M15Tests(unittest.TestCase):
    def test_canonical_duplicate_rule_is_min_rank_then_max_score(self):
        connection = duckdb.connect()
        try:
            connection.execute(
                """
                CREATE TEMP TABLE m1_repurchase(
                    customer_id VARCHAR, article_id VARCHAR, source VARCHAR,
                    source_rank BIGINT, source_score DOUBLE
                )
                """
            )
            connection.execute(
                """
                INSERT INTO m1_repurchase VALUES
                ('u1', '0000000001', 'repurchase', 2, 99.0),
                ('u1', '0000000001', 'repurchase', 1, 1.0),
                ('u1', '0000000001', 'repurchase', 1, 2.0),
                ('u1', '0000000002', 'repurchase', 3, 3.0)
                """
            )
            _canonicalize_source(connection, "repurchase")
            rows = connection.execute(
                """
                SELECT article_id, source_rank, source_score
                FROM m1_repurchase ORDER BY article_id
                """
            ).fetchall()
            self.assertEqual(
                rows,
                [("0000000001", 1, 2.0), ("0000000002", 3, 3.0)],
            )
        finally:
            connection.close()

    def test_fusion_is_stable_under_source_row_reordering(self):
        connection = duckdb.connect()
        try:
            rows = [
                ("u1", "0000000001", "repurchase", 1, 1.0),
                ("u1", "0000000001", "recent_popularity", 2, 1.0),
                ("u1", "0000000001", "product_family", 3, 1.0),
                ("u1", "0000000002", "product_family", 3, 1.0),
                ("u1", "0000000002", "recent_popularity", 2, 1.0),
                ("u1", "0000000002", "repurchase", 1, 1.0),
            ]
            connection.execute(
                """
                CREATE TEMP TABLE m1_candidates_long(
                    customer_id VARCHAR, article_id VARCHAR, source VARCHAR,
                    source_rank BIGINT, source_score DOUBLE
                )
                """
            )
            connection.executemany(
                "INSERT INTO m1_candidates_long VALUES (?, ?, ?, ?, ?)", rows
            )
            _create_collaborative_fusion(connection, _config())
            first = connection.execute(
                "SELECT article_id, candidate_rank, fused_score FROM m1_candidates ORDER BY candidate_rank"
            ).fetchall()
            connection.execute("DELETE FROM m1_candidates_long")
            connection.executemany(
                "INSERT INTO m1_candidates_long VALUES (?, ?, ?, ?, ?)",
                list(reversed(rows)),
            )
            _create_collaborative_fusion(connection, _config())
            second = connection.execute(
                "SELECT article_id, candidate_rank, fused_score FROM m1_candidates ORDER BY candidate_rank"
            ).fetchall()
            self.assertEqual(first, second)
            self.assertEqual([row[0] for row in first], ["0000000001", "0000000002"])
            self.assertEqual(first[0][2], first[1][2])
        finally:
            connection.close()

    def test_manifest_mismatches_fail_closed(self):
        base = {
            "cutoff": "2020-09-01",
            "catalog_protocol": "optimistic_all_articles",
            "sampled_user_fingerprint": "abc",
            "source_parameters": {"source_k": 100},
        }
        for key, value in (
            ("cutoff", "2020-09-02"),
            ("catalog_protocol", "strict"),
            ("sampled_user_fingerprint", "def"),
            ("source_parameters", {"source_k": 99}),
        ):
            expected = dict(base)
            expected[key] = value
            with self.assertRaises(CacheValidationError):
                validate_cache_manifest(base, expected, label=key)

    def test_optimistic_build_and_require_cache_reconstruction_are_exact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            _write_synthetic_raw(raw)
            config = _config()
            baseline = run_m1(
                raw_dir=raw,
                work_dir=root / "work",
                output_dir=root / "m1-report",
                artifact_dir=root / "m1-artifact",
                config=config,
            )
            first = run_m15(
                raw_dir=raw,
                work_dir=root / "work",
                output_dir=root / "m15-report-1",
                artifact_dir=root / "m15-artifact-1",
                cache_dir=root / "cache",
                config=config,
                cache_mode="auto",
                parity_reference=root / "m1-artifact" / "candidates.parquet",
                baseline_metrics=root / "m1-report" / "metrics.json",
                run_id="synthetic-build",
            )
            self.assertEqual(first["cache"]["target"]["status"], "built")
            self.assertEqual(first["parity"]["candidates"]["status"], "passed")
            self.assertEqual(first["parity"]["metrics"]["status"], "passed")
            self.assertEqual(
                first["truth_pairs"]["warm"] + first["truth_pairs"]["cold"],
                first["truth_pairs"]["overall"],
            )
            self.assertLess(first["leakage_audit"]["latest_behavior_date"], first["cutoff"])
            self.assertEqual(first["candidate_rows"], baseline["candidate_rows"])
            self.assertEqual(
                first["wide_evidence"]["duplicate_user_item_rows"], 0
            )
            self.assertEqual(
                first["wide_evidence"]["out_of_range_rank_rows"], 0
            )
            self.assertTrue(
                all(
                    value == 0
                    for value in first["wide_evidence"][
                        "source_null_semantics_violations"
                    ].values()
                )
            )
            self.assertGreater(first["wide_evidence"]["multi_source_rows"], 0)
            self.assertGreater(first["wide_evidence"]["leading_zero_article_rows"], 0)
            for source in first["cache"]["target"]["source_rows"]:
                manifest = (
                    root
                    / "cache"
                    / "target"
                    / first["cache"]["target"]["key"]
                    / "sources"
                    / f"{source}.manifest.json"
                )
                self.assertTrue(manifest.is_file())
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                self.assertEqual(payload["stats"]["rows"], first["cache"]["target"]["source_rows"][source])

            second = run_m15(
                raw_dir=raw,
                work_dir=root / "work",
                output_dir=root / "m15-report-2",
                artifact_dir=root / "m15-artifact-2",
                cache_dir=root / "cache",
                config=config,
                cache_mode="require",
                parity_reference=root / "m15-artifact-1" / "candidates.parquet",
                baseline_metrics=root / "m15-report-1" / "metrics.json",
                run_id="synthetic-reuse",
            )
            self.assertEqual(second["cache"]["target"]["status"], "hit")
            self.assertTrue(second["cache"]["target_cache_skipped_source_generation"])
            self.assertEqual(second["parity"]["candidates"]["status"], "passed")
            self.assertEqual(second["parity"]["metrics"]["status"], "passed")
            self.assertTrue((root / "m15-artifact-2" / "candidate_features.parquet").is_file())

            with self.assertRaises(FileExistsError):
                run_m15(
                    raw_dir=raw,
                    work_dir=root / "work",
                    output_dir=root / "m15-report-2",
                    artifact_dir=root / "new-artifact",
                    cache_dir=root / "cache",
                    config=config,
                )

    def test_strict_protocol_excludes_cold_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            _write_synthetic_raw(raw)
            result = run_m15(
                raw_dir=raw,
                work_dir=root / "work",
                output_dir=root / "report",
                artifact_dir=root / "artifact",
                cache_dir=root / "cache",
                config=_config("strict"),
                run_id="synthetic-strict",
            )
            self.assertIsNone(result["metrics"]["overall"])
            self.assertIsNotNone(result["metrics"]["warm"])
            self.assertIsNone(result["metrics"]["cold"])
            for audit in result["leakage_audit"]["source_catalog_audit"].values():
                self.assertEqual(audit["cold_candidate_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
