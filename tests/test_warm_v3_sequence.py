import numpy as np
import unittest
import torch

from hm_recsys.warm_v3_sequence import BasketGRU, contexts, guard, next_basket_samples, pool_basket


def test_day_basket_permutation_invariance_and_duplicate_preservation():
    vectors = np.array([[1., 0.], [0., 1.]], np.float32)
    assert np.array_equal(pool_basket([0, 1, 0], vectors), pool_basket([1, 0, 0], vectors))
    assert not np.allclose(pool_basket([0, 1, 0], vectors), pool_basket([0, 1], vectors))


def test_final_week_forbidden():
    assert str(guard('2020-07-22')) == '2020-07-22'
    with unittest.TestCase().assertRaises(ValueError): guard('2020-09-16')
    with unittest.TestCase().assertRaises(ValueError): guard('2020-09-17')


def test_context_excludes_target_day_and_other_users():
    data = {'starts': np.array([0, 0, 2]), 'dates': np.array([10, 12, 15]),
            'basket_vectors': np.ones((3, 4), np.float32),
            'gaps': np.array([0., 2., 0.]), 'counts': np.ones(3, np.float32)}
    values = contexts(data, np.array([1, 2]), np.array([14, 20]))
    assert values[3].tolist() == [2, 1]
    assert values[4].tolist() == [2., 5.]
    with unittest.TestCase().assertRaises(AssertionError): contexts(data, [1], np.array([12]))


def test_gru_padding_has_no_effect_and_learns():
    torch.manual_seed(1); model = BasketGRU(dim=4, hidden=4)
    values = [torch.randn(2, 8, 4), torch.ones(2, 8), torch.ones(2, 8),
              torch.tensor([2, 3]), torch.tensor([1., 4.])]
    before = model(*values)
    altered = [v.clone() for v in values]
    altered[0][0, 2:] = 999.; altered[0][1, 3:] = -999.
    assert torch.allclose(before, model(*altered))
    loss = before[:, 0].sum(); loss.backward()
    assert any(v.grad is not None and v.grad.abs().sum() > 0 for v in model.parameters())


def test_all_target_basket_items_masked_from_negatives():
    data = {'target_ids': np.array([1, 2, 3]), 'target_offsets': np.array([0, 3])}
    pos, neg, excluded = next_basket_samples(data, [0], np.random.default_rng(5), 5)
    assert pos[0] in [1, 2, 3]
    assert np.array_equal(excluded[0], np.isin(neg, [1, 2, 3]))


class SequenceTests(unittest.TestCase):
    test_pool = staticmethod(test_day_basket_permutation_invariance_and_duplicate_preservation)
    test_guard = staticmethod(test_final_week_forbidden)
    test_context = staticmethod(test_context_excludes_target_day_and_other_users)
    test_model = staticmethod(test_gru_padding_has_no_effect_and_learns)
    test_negative_mask = staticmethod(test_all_target_basket_items_masked_from_negatives)


if __name__ == '__main__':
    unittest.main()
