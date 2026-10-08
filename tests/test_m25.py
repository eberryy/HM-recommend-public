import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hm_recsys.m25 import _load_embedding_contract, _stable_topk


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M25Tests(unittest.TestCase):
    def test_stable_topk_sorts_score_then_index(self):
        indices = np.array([7, 2, 5, 3], dtype=np.int64)
        scores = np.array([0.8, 0.9, 0.8, 0.7], dtype=np.float32)
        selected, selected_scores, ambiguous = _stable_topk(indices, scores, 3)
        self.assertEqual(selected.tolist(), [2, 5, 7])
        np.testing.assert_allclose(selected_scores, [0.9, 0.8, 0.8])
        self.assertFalse(ambiguous)

    def test_stable_topk_detects_buffer_boundary_tie(self):
        indices = np.array([4, 3, 2, 1], dtype=np.int64)
        scores = np.array([0.9, 0.8, 0.8, 0.8], dtype=np.float32)
        _, _, ambiguous = _stable_topk(indices, scores, 2)
        self.assertTrue(ambiguous)

    def test_embedding_contract_binds_arrays_and_items(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embedding_path = root / "embeddings.npy"
            items_path = root / "items.csv"
            metrics_path = root / "metrics.json"
            np.save(
                embedding_path,
                np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float16),
                allow_pickle=False,
            )
            pd.DataFrame(
                {
                    "row_index": [0, 1],
                    "article_id": ["0000000001", "0000000002"],
                }
            ).to_csv(items_path, index=False)
            payload = {
                "schema_version": "m2.4-fashionclip-full-v1",
                "status": "completed",
                "contract_sha256": "source-contract",
                "embeddings": {
                    "rows": 2,
                    "valid_rows": 2,
                    "failed_rows": 0,
                    "dimensions": 2,
                    "all_valid_finite": True,
                    "artifact": str(embedding_path),
                    "artifact_bytes": embedding_path.stat().st_size,
                    "artifact_sha256": _sha(embedding_path),
                    "items_artifact": str(items_path),
                    "items_sha256": _sha(items_path),
                },
            }
            metrics_path.write_text(json.dumps(payload), encoding="utf-8")
            contract = _load_embedding_contract(metrics_path)
            self.assertEqual(contract["rows"], 2)
            self.assertEqual(contract["dimensions"], 2)
            items_path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                _load_embedding_contract(metrics_path)


if __name__ == "__main__":
    unittest.main()
