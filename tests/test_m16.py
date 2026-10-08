import json
import tempfile
import unittest
from pathlib import Path

import duckdb

from hm_recsys.m1 import SOURCES
from hm_recsys.m16 import (
    _direct_wide_sql,
    _external_wide_parity,
    _external_wide_parity_sql,
    _reference_groupby_wide_sql,
    _source_manifests,
    _validate_reference_manifest,
    _wide_parity,
)


class M16Tests(unittest.TestCase):
    def test_direct_wide_is_exactly_equal_to_groupby_wide(self):
        connection = duckdb.connect()
        try:
            connection.execute(
                """
                CREATE TEMP TABLE m16_reference(
                    customer_id VARCHAR, article_id VARCHAR,
                    candidate_rank BIGINT, fused_score DOUBLE,
                    source_count BIGINT, sources VARCHAR
                )
                """
            )
            connection.execute(
                """
                INSERT INTO m16_reference VALUES
                ('u1', '0000000001', 1, 0.2, 2, 'attribute_content,repurchase'),
                ('u1', '0000000002', 2, 0.1, 1, 'recent_popularity')
                """
            )
            for source in SOURCES:
                connection.execute(
                    f"""
                    CREATE TEMP TABLE m16_src_{source}(
                        customer_id VARCHAR, article_id VARCHAR,
                        source VARCHAR, source_rank BIGINT, source_score DOUBLE
                    )
                    """
                )
            connection.execute(
                "INSERT INTO m16_src_repurchase VALUES "
                "('u1', '0000000001', 'repurchase', 1, 3.0)"
            )
            connection.execute(
                "INSERT INTO m16_src_recent_popularity VALUES "
                "('u1', '0000000002', 'recent_popularity', 2, 1.0)"
            )
            connection.execute(
                "INSERT INTO m16_src_attribute_content VALUES "
                "('u1', '0000000001', 'attribute_content', 5, 0.5)"
            )
            union = " UNION ALL ".join(
                f"SELECT * FROM m16_src_{source}" for source in SOURCES
            )
            connection.execute(
                f"CREATE TEMP TABLE m16_long AS {union}"
            )
            connection.execute(_reference_groupby_wide_sql("collaborative", 60))
            connection.execute(_direct_wide_sql("collaborative", 60))
            self.assertEqual(_wide_parity(connection)["feature_mismatch_rows"], 0)

            with tempfile.TemporaryDirectory() as temp_dir:
                reference = Path(temp_dir) / "reference.parquet"
                escaped = str(reference).replace("'", "''")
                connection.execute(
                    f"COPY m16_direct_wide TO '{escaped}' (FORMAT PARQUET)"
                )
                connection.execute(_external_wide_parity_sql(reference))
                self.assertEqual(
                    _external_wide_parity(connection),
                    {
                        "actual_minus_reference_rows": 0,
                        "reference_minus_actual_rows": 0,
                    },
                )
        finally:
            connection.close()

    def test_reference_manifest_identity_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates = root / "candidates.parquet"
            connection = duckdb.connect()
            try:
                escaped = str(candidates).replace("'", "''")
                connection.execute(
                    f"COPY (SELECT 'u1' customer_id, 'a1' article_id, "
                    "1::BIGINT candidate_rank, 0.1::DOUBLE fused_score, "
                    "1::BIGINT source_count, 'repurchase' sources) "
                    f"TO '{escaped}' (FORMAT PARQUET)"
                )
            finally:
                connection.close()
            manifest = root / "manifest.json"
            payload = {
                "schema_version": "m1.5-cache-v1",
                "scope": "fusion_reference",
                "target_key": "target-a",
                "fusion_profile": "collaborative",
                "rrf_constant": 60,
                "candidate_k": 100,
                "artifact": str(candidates.resolve()),
                "artifact_bytes": candidates.stat().st_size,
                "stats": {"rows": 1, "unique_user_item_rows": 1},
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            validated = _validate_reference_manifest(
                candidates, manifest, "target-a", "collaborative", 60, 100
            )
            self.assertEqual(validated["target_key"], "target-a")
            with self.assertRaisesRegex(ValueError, "target_key"):
                _validate_reference_manifest(
                    candidates, manifest, "target-b", "collaborative", 60, 100
                )

    def test_manifest_declaring_duplicate_user_items_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir)
            (source_dir / "manifest.json").write_text(
                json.dumps(
                    {"schema_version": "m1.5-cache-v1", "scope": "target_sources"}
                ),
                encoding="utf-8",
            )
            parquet = source_dir / "repurchase.parquet"
            connection = duckdb.connect()
            try:
                escaped = str(parquet).replace("'", "''")
                connection.execute(
                    f"COPY (SELECT 'u1' customer_id, 'a1' article_id) "
                    f"TO '{escaped}' (FORMAT PARQUET)"
                )
            finally:
                connection.close()
            (source_dir / "repurchase.manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "m1.5-cache-v1",
                        "source": "repurchase",
                        "artifact_bytes": parquet.stat().st_size,
                        "stats": {"rows": 2, "unique_user_item_rows": 1},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not unique by user-item"):
                _source_manifests(source_dir)


if __name__ == "__main__":
    unittest.main()
