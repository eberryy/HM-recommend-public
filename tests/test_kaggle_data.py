import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from hm_recsys.kaggle_data import (
    _run_download_command,
    detect_auth_source,
    extract_competition_archive,
    extract_image_archive,
    get_data_status,
)


class KaggleDataTests(unittest.TestCase):
    def test_token_auth_is_detected_without_exposing_value(self):
        clean_env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"}
        }
        clean_env["KAGGLE_API_TOKEN"] = "secret-value"
        with patch.dict(os.environ, clean_env, clear=True):
            self.assertEqual(detect_auth_source(), "KAGGLE_API_TOKEN")

    def test_status_detects_downloaded_transactions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw"
            raw_dir.mkdir()
            (raw_dir / "transactions_train.csv").write_text(
                "t_dat,customer_id,article_id\n", encoding="utf-8"
            )
            status = get_data_status(raw_dir, Path(temp_dir) / "missing.env")
            self.assertTrue(status.ready)
            self.assertFalse(status.tabular_ready)
            self.assertTrue(status.tabular_files["transactions_train.csv"])
            self.assertFalse(status.tabular_files["articles.csv"])

    def test_safe_archive_extracts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "data.zip"
            destination = root / "raw"
            destination.mkdir()
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("transactions_train.csv", "header\n")
            extract_competition_archive(archive, destination)
            self.assertTrue((destination / "transactions_train.csv").is_file())

    def test_archive_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "unsafe.zip"
            destination = root / "raw"
            destination.mkdir()
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("../secret.txt", "nope")
            with self.assertRaises(ValueError):
                extract_competition_archive(archive, destination)


    @patch("hm_recsys.kaggle_data.time.sleep")
    @patch("hm_recsys.kaggle_data.subprocess.run")
    def test_download_wrapper_retries_and_preserves_command(
        self, run_mock, sleep_mock
    ):
        command = ["kaggle", "competitions", "download", "competition"]
        run_mock.side_effect = [
            subprocess.CalledProcessError(1, command),
            None,
        ]
        _run_download_command(command, env={"TOKEN": "secret"}, attempts=3)
        self.assertEqual(run_mock.call_count, 2)
        self.assertEqual(run_mock.call_args_list[0], run_mock.call_args_list[1])
        sleep_mock.assert_called_once_with(1)

    def test_download_wrapper_rejects_zero_attempts(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            _run_download_command(["kaggle"], env={}, attempts=0)
    def test_image_only_extraction_is_resumable_and_does_not_touch_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "competition.zip"
            destination = root / "raw"
            destination.mkdir()
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("articles.csv", "article_id" + chr(10))
                zipped.writestr("images/010/0108775015.jpg", b"image-bytes")

            first = extract_image_archive(archive, destination)
            self.assertEqual(first["image_members"], 1)
            self.assertEqual(first["extracted_files"], 1)
            self.assertEqual(first["skipped_files"], 0)
            self.assertFalse((destination / "articles.csv").exists())
            image = destination / "images" / "010" / "0108775015.jpg"
            self.assertEqual(image.read_bytes(), b"image-bytes")

            second = extract_image_archive(archive, destination)
            self.assertEqual(second["extracted_files"], 0)
            self.assertEqual(second["skipped_files"], 1)

            image.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                extract_image_archive(archive, destination)

    def test_image_only_extraction_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / "unsafe-images.zip"
            destination = root / "raw"
            destination.mkdir()
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("images/../../secret.jpg", b"nope")
            with self.assertRaisesRegex(ValueError, "unsafe image archive member"):
                extract_image_archive(archive, destination)
if __name__ == "__main__":
    unittest.main()
