import tempfile
import unittest
from pathlib import Path

import pandas as pd

from hm_recsys.baseline import build_ground_truth, generate_baseline
from hm_recsys.data import load_temporal_split


class PipelineTests(unittest.TestCase):
    def test_time_split_and_baseline(self):
        rows = [
            ("2020-08-01", "u1", 10),
            ("2020-08-25", "u1", 11),
            ("2020-08-26", "u2", 12),
            ("2020-09-01", "u1", 11),
            ("2020-09-02", "u2", 12),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transactions.csv"
            pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id"]).to_csv(
                path, index=False
            )
            split = load_temporal_split(
                path,
                cutoff="2020-09-01",
                history_weeks=8,
                sample_rate=1.0,
                chunksize=2,
            )
        self.assertTrue((split.history["t_dat"] < split.cutoff).all())
        self.assertTrue((split.validation["t_dat"] >= split.cutoff).all())
        truth = build_ground_truth(split.validation)
        output = generate_baseline(split.history, sorted(truth), split.cutoff)
        self.assertIn(11, output.candidates["u1"])
        self.assertIn(12, output.candidates["u2"])


if __name__ == "__main__":
    unittest.main()
