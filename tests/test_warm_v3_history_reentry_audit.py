from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[1] / "src" / "hm_recsys" / "warm_v3_history_reentry_audit.py"


class HistoryReentryAuditSourceTest(unittest.TestCase):
    def test_candidates_are_pre_cutoff_and_anti_joined(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("WHERE t_dat<DATE '{cutoff}'", text)
        self.assertIn("WHERE c.article_id IS NULL", text)

    def test_outer_and_final_week_are_closed(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn('"outer": "not_run"', text)
        self.assertIn('"final_week": "2020-09-16 not_run"', text)


if __name__ == "__main__":
    unittest.main()
