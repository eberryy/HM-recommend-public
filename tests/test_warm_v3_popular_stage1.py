from pathlib import Path
import unittest


SOURCE = Path(__file__).parents[1] / "src" / "hm_recsys" / "warm_v3_popular_stage1.py"


class PopularStage1SourceTest(unittest.TestCase):
    def test_article_dimension_uses_hm_article_id_column(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("SELECT article_id kpr", text)
        self.assertNotIn("SELECT article kpr", text)

    def test_stage1_selection_is_label_free(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        selection = text.split("def evaluate", 1)[1].split("def render", 1)[0]
        order_clause = selection.split("sort_values(", 1)[1].split(").groupby", 1)[0]
        self.assertNotIn("target_flag", order_clause)
        self.assertIn("FEATURES", text)
        feature_block = text.split("FEATURES = [", 1)[1].split("]", 1)[0]
        self.assertNotIn("target_flag", feature_block)


if __name__ == "__main__":
    unittest.main()
