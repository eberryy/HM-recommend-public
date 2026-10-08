import numpy as np
import pandas as pd

from hm_recsys import mind_warm_final_confirmation as final


def test_wv3_selector_is_positive_greedy_and_disjoint():
    rows = []
    for rank in range(1, 51):
        score = 1.0 - rank / 100.0
        if rank == 13:
            score = 2.0
        if rank == 14:
            score = 1.9
        rows.append(("u", f"i{rank:02d}", rank, score))
    frame = pd.DataFrame(rows, columns=["customer_id", "article_id", "rf", "ranking_score"])
    selected = final.choose_wv3_swaps(frame)
    assert len(selected) == 2
    assert set(selected.challenger_article_id) == {"i13", "i14"}
    assert selected.victim_article_id.nunique() == 2
    assert selected.victim_rank.between(8, 12).all()
    assert (selected.score_difference > 0).all()


def test_ap_at_12_exact_example():
    labels = np.zeros((2, 12), dtype=np.uint8)
    labels[0, [0, 2]] = 1
    truth = np.array([2, 3], dtype=np.int32)
    value, per_user = final.ap_from_matrix(labels, truth)
    expected0 = (1.0 + 2.0 / 3.0) / 2.0
    assert np.allclose(per_user, [expected0, 0.0])
    assert np.isclose(value, expected0 / 2.0)


def test_wv3_feature_contract_is_99_unique_columns():
    assert len(final.WV3_FEATURES) == 99
    assert len(set(final.WV3_FEATURES)) == 99
    assert "target" not in final.WV3_FEATURES
