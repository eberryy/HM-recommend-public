from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hm_recsys.warm_v3_continuation_finalize import architecture_text, family_summary


class ContinuationFinalizeTest(unittest.TestCase):
    def test_architecture_names_corrected_champion_components(self) -> None:
        text = architecture_text()
        self.assertIn("WV3-661", text)
        self.assertIn("WV3-680", text)
        self.assertIn("无标签最多两次尾部换位", text)
        self.assertNotIn("WV3-691", text)

    def test_family_summary_preserves_rejections(self) -> None:
        identifiers = {item for row in family_summary() for item in row["experiments"]}
        self.assertTrue({"WV3-720", "WV3-721", "WV3-741", "WV3-761", "WV3-810"}.issubset(identifiers))


if __name__ == "__main__":
    unittest.main()
