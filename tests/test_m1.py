import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from hm_recsys.m1 import M1Config, run_m1


class M1Tests(unittest.TestCase):
    def test_m1_smoke_keeps_temporal_boundary_and_writes_evidence(self):
        transactions = pd.DataFrame(
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
        )
        articles = pd.DataFrame(
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
        )
        customers = pd.DataFrame(
            [("u1", 25), ("u2", 27), ("u3", None)],
            columns=["customer_id", "age"],
        )
        submission = pd.DataFrame(
            [("u1", ""), ("u2", ""), ("u3", "")],
            columns=["customer_id", "prediction"],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            transactions.to_csv(raw / "transactions_train.csv", index=False)
            articles.to_csv(raw / "articles.csv", index=False)
            customers.to_csv(raw / "customers.csv", index=False)
            submission.to_csv(raw / "sample_submission.csv", index=False)

            result = run_m1(
                raw_dir=raw,
                work_dir=root / "work",
                output_dir=root / "reports",
                artifact_dir=root / "artifacts",
                config=M1Config(
                    cutoff="2020-09-01",
                    sample_rate=1.0,
                    min_covisit_count=1,
                    source_k=10,
                    candidate_k=10,
                    covisit_neighbor_k=10,
                ),
            )

            self.assertEqual(result["population"]["sampled_validation_users"], 2)
            self.assertEqual(result["catalog_protocol"], "optimistic_all_articles")
            self.assertEqual(result["eligible_catalog_size"], 5)
            self.assertEqual(result["truth_pairs"], {"overall": 3, "warm": 2, "cold": 1})
            self.assertEqual(result["catalog_unreachable_truth_pairs"], 0)
            self.assertIsNotNone(result["metrics"]["overall"])
            self.assertIsNotNone(result["metrics"]["warm"])
            self.assertIsNotNone(result["metrics"]["cold"])
            self.assertEqual(
                result["truth_pairs"]["warm"] + result["truth_pairs"]["cold"],
                result["truth_pairs"]["overall"],
            )
            self.assertFalse(result["leakage_audit"]["validation_used_for_catalog"])
            self.assertLess(
                result["leakage_audit"]["latest_behavior_date"], result["cutoff"]
            )
            for source in (
                "repurchase",
                "recent_popularity",
                "user_day_covisit",
                "age_popularity",
            ):
                self.assertEqual(
                    result["leakage_audit"]["source_catalog_audit"][source][
                        "cold_candidate_pairs"
                    ],
                    0,
                )
            self.assertEqual(result["config"]["duplicate_policy"], "keep_all_events")
            self.assertIn("attribute_content", result["source_metrics"])
            self.assertIn("activity_12w", result["segments"])
            self.assertIn("item_popularity", result["segments"])
            self.assertTrue((root / "reports" / "metrics.json").is_file())
            self.assertTrue((root / "reports" / "M1_REPORT.md").is_file())
            self.assertTrue((root / "artifacts" / "candidates.parquet").is_file())

            artifact_connection = duckdb.connect()
            try:
                article_id, logical_type = artifact_connection.execute(
                    f"SELECT article_id, typeof(article_id) FROM read_parquet('{(root / 'artifacts' / 'candidates.parquet').as_posix()}') LIMIT 1"
                ).fetchone()
            finally:
                artifact_connection.close()
            self.assertEqual(logical_type, "VARCHAR")
            self.assertTrue(article_id.startswith("0"))

            strict_result = run_m1(
                raw_dir=raw,
                work_dir=root / "work",
                output_dir=root / "reports-strict",
                artifact_dir=root / "artifacts-strict",
                config=M1Config(
                    cutoff="2020-09-01",
                    sample_rate=1.0,
                    min_covisit_count=1,
                    source_k=10,
                    candidate_k=10,
                    covisit_neighbor_k=10,
                    catalog_protocol="strict",
                ),
            )
            self.assertEqual(strict_result["eligible_catalog_size"], 3)
            self.assertEqual(strict_result["truth_pairs"], result["truth_pairs"])
            self.assertEqual(strict_result["catalog_unreachable_truth_pairs"], 1)
            self.assertIsNone(strict_result["metrics"]["overall"])
            self.assertIsNotNone(strict_result["metrics"]["warm"])
            self.assertIsNone(strict_result["metrics"]["cold"])
            self.assertEqual(
                strict_result["population"]["sampled_user_fingerprint"],
                result["population"]["sampled_user_fingerprint"],
            )
            for source_audit in strict_result["leakage_audit"][
                "source_catalog_audit"
            ].values():
                self.assertEqual(source_audit["cold_candidate_pairs"], 0)

            final_result = run_m1(
                raw_dir=raw,
                work_dir=root / "work",
                output_dir=root / "reports-final",
                artifact_dir=root / "artifacts-final",
                config=M1Config(
                    cutoff="2020-09-01",
                    sample_rate=1.0,
                    min_covisit_count=1,
                    source_k=10,
                    candidate_k=10,
                    covisit_neighbor_k=10,
                    fusion_profile="collaborative",
                    evaluation_mode="final",
                ),
            )
            self.assertEqual(
                list(final_result["fusion_comparison"]), ["rrf_collaborative"]
            )
            self.assertEqual(final_result["leave_one_out"], {})


if __name__ == "__main__":
    unittest.main()
