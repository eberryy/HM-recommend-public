from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[1] / "src" / "hm_recsys" / "warm_v3_pop_covisit_consensus_audit.py"


class PopCovisitConsensusAuditTest(unittest.TestCase):
    def test_consensus_is_intersection_and_antijoin(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("FROM covisit c JOIN popular p USING(article_id)", text)
        self.assertIn("WHERE f.article_id IS NULL", text)

    def test_final_week_is_closed(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"final_week": "2020-09-16 not_run"', text)


if __name__ == "__main__":
    unittest.main()
