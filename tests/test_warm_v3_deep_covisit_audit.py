from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[1] / "src" / "hm_recsys" / "warm_v3_deep_covisit_audit.py"


class DeepCovisitAuditSourceTest(unittest.TestCase):
    def test_temporal_and_candidate_boundaries(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("t_dat<DATE '{cutoff}'", text)
        self.assertIn("WHERE p.article_id IS NULL", text)
        self.assertIn("NEIGHBOR_K = 200", text)
        self.assertIn("SOURCE_K = 200", text)

    def test_final_week_is_closed(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"final_week": "2020-09-16 not_run"', text)


if __name__ == "__main__":
    unittest.main()
