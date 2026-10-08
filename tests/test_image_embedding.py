import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hm_recsys.image_embedding import (
    _full_item_records,
    _item_manifest_sha256,
    _load_audit_contract,
    _relative_image_name,
    select_all_image_paths,
    select_image_paths,
)


class ImageEmbeddingTests(unittest.TestCase):
    def test_selection_is_deterministic_by_relative_filename_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            images = Path(temp_dir) / "images"
            paths = [
                images / "010" / "0100000001.jpg",
                images / "020" / "0200000002.jpg",
                images / "030" / "0300000003.jpg",
            ]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bytes([index]))

            expected = sorted(
                paths,
                key=lambda path: hashlib.sha256(
                    _relative_image_name(path, images).encode("utf-8")
                ).digest(),
            )[:2]
            first = select_image_paths(images, 2)
            second = select_image_paths(images, 2)
            self.assertEqual(first, expected)
            self.assertEqual(second, expected)

    def test_audit_contract_binds_v2_file_hash_and_image_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            payload = {
                "schema_version": "m2.3-image-audit-v2",
                "status": "completed",
                "mapping": {
                    "valid_image_files": 3,
                    "manifest_sha256": "manifest-hash",
                },
                "article_coverage": {"coverage_rate": 0.99},
            }
            metrics_path.write_text(
                json.dumps(payload), encoding="utf-8"
            )
            _, contract = _load_audit_contract(metrics_path, 3)
            self.assertEqual(contract["valid_image_files"], 3)
            self.assertEqual(
                contract["source_manifest_sha256"], "manifest-hash"
            )
            self.assertEqual(len(contract["sha256"]), 64)
            with self.assertRaisesRegex(ValueError, "image count mismatch"):
                _load_audit_contract(metrics_path, 2)

    def test_full_selection_and_manifest_are_stable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            images = Path(temp_dir) / "images"
            paths = [
                images / "020" / "0200000002.jpg",
                images / "010" / "0100000001.jpg",
            ]
            for index, path in enumerate(paths):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(bytes([index + 1]))

            selected = select_all_image_paths(images)
            self.assertEqual(selected, [paths[1], paths[0]])
            records = _full_item_records(selected, images)
            first_hash = _item_manifest_sha256(records)
            self.assertEqual(records[0]["row_index"], 0)
            self.assertEqual(records[0]["article_id"], "0100000001")
            self.assertEqual(len(first_hash), 64)
            self.assertEqual(first_hash, _item_manifest_sha256(records))

            paths[0].write_bytes(b"changed-size")
            changed_records = _full_item_records(selected, images)
            self.assertNotEqual(
                first_hash, _item_manifest_sha256(changed_records)
            )
    def test_selection_rejects_zero_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            images = Path(temp_dir) / "images"
            images.mkdir()
            with self.assertRaisesRegex(ValueError, "at least 1"):
                select_image_paths(images, 0)


if __name__ == "__main__":
    unittest.main()
