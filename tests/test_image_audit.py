import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from PIL import Image

from hm_recsys.image_audit import _parse_image_name, run_image_audit


def _jpeg_bytes(colour: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (16, 20), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


class ImageAuditTests(unittest.TestCase):
    def test_image_path_contract(self):
        self.assertEqual(
            _parse_image_name("images/010/0108775015.jpg"),
            ("0108775015", None),
        )
        self.assertEqual(
            _parse_image_name("images/999/0108775015.jpg"),
            (None, "prefix_mismatch"),
        )
        self.assertEqual(
            _parse_image_name("other/0108775015.jpg"),
            (None, "path_pattern"),
        )

    def test_archive_mapping_decode_and_temporal_coverage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            pd.DataFrame(
                {
                    "article_id": [
                        "0108775015",
                        "0108775044",
                        "0999999999",
                    ]
                }
            ).to_csv(raw / "articles.csv", index=False)
            pd.DataFrame(
                [
                    ("2020-07-01", "u1", "0108775015"),
                    ("2020-07-22", "u1", "0108775015"),
                    ("2020-07-22", "u2", "0108775044"),
                    ("2020-07-22", "u3", "0999999999"),
                ],
                columns=["t_dat", "customer_id", "article_id"],
            ).to_csv(raw / "transactions_train.csv", index=False)
            archive = raw / "competition.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr(
                    "images/010/0108775015.jpg", _jpeg_bytes((255, 0, 0))
                )
                zipped.writestr(
                    "images/010/0108775044.jpg", _jpeg_bytes((0, 255, 0))
                )
                zipped.writestr(
                    "images/011/0111111111.jpg", _jpeg_bytes((255, 0, 0))
                )
                zipped.writestr(
                    "images/999/0108775015.jpg", _jpeg_bytes((0, 0, 255))
                )

            output = root / "reports" / "image-audit"
            metrics = run_image_audit(
                raw_dir=raw,
                output_dir=output,
                source="archive",
                archive_path=archive,
                verify_sample=10,
                cutoffs=("2020-07-22",),
            )

            self.assertEqual(metrics["mapping"]["valid_image_files"], 3)
            self.assertEqual(
                metrics["mapping"]["invalid_path_counts"]["prefix_mismatch"], 1
            )
            self.assertEqual(metrics["article_coverage"]["unique_article_ids"], 3)
            self.assertEqual(metrics["article_coverage"]["covered_article_ids"], 2)
            self.assertEqual(metrics["decode_sample"]["decoded"], 3)
            self.assertEqual(
                metrics["mapping"]["exact_duplicate_verification"][
                    "exact_sha256_groups"
                ],
                1,
            )
            window = metrics["transaction_windows"][0]
            self.assertEqual(window["cutoff"], "2020-07-22")
            self.assertEqual(window["truth_pairs"], 3)
            self.assertEqual(window["image_truth_pairs"], 2)
            self.assertEqual(window["warm_truth_pairs"], 1)
            self.assertEqual(window["cold_truth_pairs"], 2)
            self.assertEqual(window["image_cold_truth_pairs"], 1)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "M2_3_IMAGE_AUDIT.md").is_file())

    def test_existing_output_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "existing"
            output.mkdir()
            (output / "metrics.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_image_audit(
                    raw_dir=root,
                    output_dir=output,
                    verify_sample=0,
                    cutoffs=("2020-07-22",),
                )


if __name__ == "__main__":
    unittest.main()
