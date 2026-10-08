import numpy as np
import pandas as pd
import unittest
from scipy.sparse import csr_matrix

from hm_recsys.warm_v3_graph import (
    PARAMS, latent_factors, match_scores, normalized_relation, propagate, source_model)


def test_normalized_graph_matches_dense_bipartite_operator():
    r = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
    u = np.arange(8, dtype=np.float32).reshape(2, 4)/10
    i = np.arange(12, dtype=np.float32).reshape(3, 4)/10
    gu, gi, _ = propagate(csr_matrix(r), u, i)
    s = r / np.sqrt(r.sum(1)[:, None]*r.sum(0)[None, :])
    a = np.block([[np.zeros((2, 2)), s], [s.T, np.zeros((3, 3))]])
    e = np.concatenate([u, i])
    expected = (e + a@e + a@a@e)/3
    np.testing.assert_allclose(np.concatenate([gu, gi]), expected, atol=2e-7)


def test_graph_has_two_hop_effect_and_fixed_depth():
    r = csr_matrix(np.array([[1, 1], [0, 1]], np.float32))
    u = np.array([[1.], [0.]], np.float32)
    i = np.zeros((2, 1), np.float32)
    gu, _, _ = propagate(r, u, i)
    assert gu[1, 0] > 0  # another user reaches this user through a shared item
    with unittest.TestCase().assertRaises(ValueError):
        propagate(r, u, i, layers=3)


def test_binary_degree_not_transaction_frequency():
    with unittest.TestCase().assertRaises(ValueError):
        normalized_relation(csr_matrix([[2., 0.], [0., 1.]]))
    s, du, di = normalized_relation(csr_matrix([[1., 0.], [0., 0.]]))
    assert np.isfinite(s.data).all()
    assert list(du) == [1, 0] and list(di) == [1, 0]


def test_bias_coordinate_excluded_before_propagation():
    u = np.ones((2, 101), np.float32)
    i = np.ones((3, 101), np.float32)
    i[:, -1] = 999.
    ul, il = latent_factors(u, i)
    assert ul.shape == (2, 100) and il.shape == (3, 100)
    assert il.max() == 1
    u[:, -1] = 2
    with unittest.TestCase().assertRaises(ValueError):
        latent_factors(u, i)


def test_unknown_pairs_remain_missing_not_false_zero():
    frame = pd.DataFrame({'customer_id': ['u', 'x', 'u'], 'article_id': ['i', 'i', 'z']})
    score, missing = match_scores(frame, pd.Index(['u']), pd.Index(['i']),
                                  np.array([[2., 3.]]), np.array([[4., 5.]]))
    assert score[0] == 23 and np.isnan(score[1:]).all()
    np.testing.assert_equal(missing, [0, 1, 1])


def test_final_week_forbidden_and_graph_is_not_train_lightgcn():
    with unittest.TestCase().assertRaises(ValueError):
        source_model('2020-09-16')
    assert PARAMS['graph_training_steps'] == 0
    assert PARAMS['bias_propagated'] is False


class GraphTests(unittest.TestCase):
    pass


for _name, _function in list(globals().items()):
    if _name.startswith('test_') and callable(_function):
        setattr(GraphTests, _name, staticmethod(_function))
del _name, _function


if __name__ == '__main__':
    unittest.main()
