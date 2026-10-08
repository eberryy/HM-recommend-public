from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import duckdb
import pandas as pd

from hm_recsys import mind_warm_ranking as ranking


class MindWarmOutputRootTests(unittest.TestCase):
    @staticmethod
    def write_parquet(frame: pd.DataFrame, path: Path) -> None:
        with duckdb.connect() as con:
            con.from_df(frame).write_parquet(str(path))

    def test_default_paths_remain_under_original_experiment(self) -> None:
        old = ranking._paths("2019-12-25")
        explicit = ranking._paths("2019-12-25", output_root=ranking.ARTIFACT_ROOT)
        self.assertEqual(old, explicit)
        self.assertTrue(all(path.parent == ranking.ARTIFACT_ROOT / "2019-12-25" for path in old))

    def test_new_candidate_and_feature_caches_cannot_reuse_or_overwrite_old_root(self) -> None:
        cutoff = "2019-12-25"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_root, new_root, raw = root / "001", root / "002", root / "raw"
            raw.mkdir()
            old_folder = old_root / cutoff
            old_folder.mkdir(parents=True)
            sentinels = {
                "augmented-candidates.parquet": b"old candidate evidence",
                "AUGMENTED.json": b"old candidate metadata",
                "features.parquet": b"old feature evidence",
                "FEATURES.json": b"old feature metadata",
            }
            for name, content in sentinels.items():
                (old_folder / name).write_bytes(content)
            article_ids = [f"{index:010d}" for index in range(1, 102)]
            base = root / "base.parquet"
            self.write_parquet(pd.DataFrame({
                "customer_id": ["u1"] * 100,
                "article_id": article_ids[:100],
                "candidate_rank": range(1, 101),
            }), base)
            mind = root / "mind.parquet"
            self.write_parquet(pd.DataFrame({
                "customer_id": ["u1", "u1"],
                "article_id": [article_ids[0], article_ids[-1]],
                "mind_rank": [1, 2],
                "mind_score": [0.8, 0.6],
                "best_interest": [1, 2],
                "interest_count": [3, 3],
            }), mind)
            pd.DataFrame({
                "article_id": article_ids,
                "product_code": range(101),
                "product_type_no": [1] * 101,
                "garment_group_no": [1] * 101,
                "department_no": [1] * 101,
                "index_group_no": [1] * 101,
                "perceived_colour_master_id": [1] * 101,
            }).to_csv(raw / "articles.csv", index=False)
            pd.DataFrame({"customer_id": ["u1"], "age": [22]}).to_csv(raw / "customers.csv", index=False)
            tx = root / "transactions.parquet"
            self.write_parquet(pd.DataFrame({
                "customer_id": ["u1", "u1"],
                "article_id": [article_ids[0], article_ids[-1]],
                "t_dat": [pd.Timestamp("2019-12-24").date(), pd.Timestamp(cutoff).date()],
                "price": [0.02, 0.03],
                "sales_channel_id": [2, 2],
            }), tx)
            with (
                mock.patch.object(ranking, "ARTIFACT_ROOT", old_root),
                mock.patch.object(ranking, "RAW", raw),
                mock.patch.object(ranking, "TX", tx),
                mock.patch.object(ranking, "_base_candidates", return_value=base),
                mock.patch.object(ranking, "ensure_mind_candidates", return_value=(mind, {"model": "frozen-model"})) as ensure,
            ):
                path, evidence = ranking.build_features(cutoff, "cpu", output_root=new_root)
                self.assertEqual(path, new_root / cutoff / "features.parquet")
                self.assertEqual(evidence["rows"], 101)
                self.assertEqual(evidence["mind_only_positive_rows"], 1)
                self.assertEqual(evidence["latest_behavior_date"], "2019-12-24")
                ensure.assert_called_once_with(cutoff, "cpu")
                with duckdb.connect() as con:
                    feature = con.read_parquet(str(path)).df()
                self.assertEqual(int(feature.loc[feature.article_id == article_ids[-1], "item_events_12w"].iloc[0]), 0)
                marker = json.loads((new_root / cutoff / "FEATURES.json").read_text(encoding="utf-8"))
                self.assertEqual(Path(marker["candidate_source"]), new_root / cutoff / "augmented-candidates.parquet")
                # Replay must use only the isolated cache and cannot trigger retrieval.
                again, saved = ranking.build_features(cutoff, "cpu", output_root=new_root)
                self.assertEqual(again, path)
                self.assertEqual(saved, evidence)
                ensure.assert_called_once()
            for name, content in sentinels.items():
                self.assertEqual((old_folder / name).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
