import unittest

import pandas as pd

from hm_recsys.m2 import _prepare_frame, build_category_maps


class M2EncodingTests(unittest.TestCase):
    def test_categories_are_fit_on_train_and_unknown_maps_to_zero(self):
        train = pd.DataFrame(
            {
                "age_bucket": [4, 5, None],
                "article_product_type_no": [10, 20, 10],
                "article_garment_group_no": [1, 2, 1],
                "article_department_no": [100, 300, 100],
                "article_index_group_no": [1, 2, 1],
                "article_colour_master_id": [5, 7, 5],
            }
        )
        mappings = build_category_maps(train)
        validation = train.iloc[[0]].copy()
        validation.loc[:, "article_department_no"] = 999
        validation.loc[:, "age_bucket"] = float("nan")
        encoded = _prepare_frame(
            validation,
            ["age_bucket", "article_department_no"],
            mappings,
        )
        self.assertEqual(encoded["age_bucket"].iloc[0], 0)
        self.assertEqual(encoded["article_department_no"].iloc[0], 0)
        self.assertEqual(sorted(mappings["article_department_no"].values()), [1, 2])


if __name__ == "__main__":
    unittest.main()
