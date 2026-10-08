"""Synthetic AP-weight, identity, time, neutrality and context regressions."""
import unittest
from contextlib import contextmanager
from pathlib import Path
import tempfile
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import torch

from hm_recsys import warm_v3_multistage as multi


def brute_ap(labels, truth_count, k=12):
    y = np.asarray(labels)[:k]
    return float(np.sum(y*np.cumsum(y)/np.arange(1, len(y)+1))/min(truth_count, k))


def fixture():
    rows = []
    for user, active in [('a', 5), ('b', 3), ('c', 0)]:
        for rank in range(1, 101):
            rows.append({'customer_id': user, 'article_id': rank, 'candidate_rank': rank,
                         'target': int(rank in [10, 70]), 'truth_count': 8,
                         'user_history_events_12w': active, 'rf': 101-rank})
    keys = pd.DataFrame(rows)
    rng = np.random.default_rng(9)
    data = {'x': rng.normal(size=(300, 21)).astype(np.float32),
            'target': keys.target.to_numpy(np.uint8),
            'truth_count': keys.truth_count.to_numpy(np.int32),
            'active': (keys.user_history_events_12w.to_numpy() > 0).astype(np.uint8),
            'offset': np.array([0, 100, 200, 300])}
    return data, keys, {'features': ['f'+str(i) for i in range(21)]}


class MultistageTests(unittest.TestCase):
    def test_ap_weights_all_swaps_equal_bruteforce(self):
        rng = np.random.default_rng(7)
        labels = (rng.random((9, 50)) < .19).astype(np.uint8)
        labels[0] = 0
        labels[1] = 1
        labels[2] = 0
        labels[2, 49] = 1
        truth = labels.sum(axis=1)+rng.integers(1, 24, size=9)
        scores = rng.normal(size=(9, 50))
        scores[3] = 0  # stable ties are resolved by the existing Stage1 list order.
        weights = multi.delta_ap_weights(torch.from_numpy(labels), torch.from_numpy(scores),
                                         torch.from_numpy(truth)).numpy()
        for b in range(9):
            order = np.argsort(-scores[b], kind='stable')
            original = labels[b, order]
            before = brute_ap(original, truth[b])
            for i in range(50):
                for j in range(50):
                    if labels[b, i] != 1 or labels[b, j] != 0:
                        self.assertEqual(weights[b, i, j], 0)
                        continue
                    swapped = labels[b].copy()
                    swapped[i], swapped[j] = swapped[j], swapped[i]
                    expected = abs(brute_ap(swapped[order], truth[b])-before)
                    self.assertAlmostEqual(weights[b, i, j], expected, places=13)

    def test_ap_uses_full_truth_and_zero_after12(self):
        labels = torch.zeros(2, 50, dtype=torch.uint8)
        labels[:, [0, 19]] = 1
        scores = -torch.arange(50, dtype=torch.float64)[None, :].repeat(2, 1)
        weights = multi.delta_ap_weights(labels, scores, torch.tensor([2, 20]))
        torch.testing.assert_close(weights[0], weights[1]*6, atol=1e-15, rtol=0)
        self.assertEqual(float(weights[0, 19, 29]), 0.)
        with self.assertRaises(ValueError):
            multi.delta_ap_weights(labels, scores, torch.tensor([1, 20]))

    def test_zero_initialized_model_preserves_exact_order_and_bound(self):
        model = multi.Top50Residual()
        x = torch.randn(4, 50, 21)
        rank = torch.arange(1, 51).repeat(4, 1)
        scores, residual = model(x, rank)
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=0, rtol=0)
        actual = multi.residual_scores(rank.numpy(), residual.detach().numpy())
        np.testing.assert_array_equal(np.argsort(-actual, axis=1, kind='stable'),
                                      np.tile(np.arange(50), (4, 1)))
        with torch.no_grad():
            model.output[-1].weight.fill_(100)
        _, residual = model(x, rank)
        self.assertTrue(torch.all(torch.abs(residual) <= 2))

    def test_context_permutation_equivariance_and_gradients(self):
        torch.manual_seed(5)
        model = multi.Top50Residual()
        with torch.no_grad():
            model.output[-1].weight.fill_(.01)
        x = torch.randn(2, 50, 21)
        rank = torch.arange(1, 51).repeat(2, 1)
        permutation = torch.randperm(50)
        a, r = model(x, rank)
        b, rr = model(x[:, permutation], rank[:, permutation])
        torch.testing.assert_close(a[:, permutation], b, atol=1e-6, rtol=1e-6)
        labels = torch.zeros(2, 50, dtype=torch.uint8)
        labels[:, [4, 29]] = 1
        loss = multi.ap_pair_loss(a, r, labels, torch.tensor([4, 19]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.item[0].weight.grad.abs().sum()), 0)

    def test_groups_not_mixed_and_no_positive_regularization_only(self):
        scores = torch.randn(2, 50, requires_grad=True)
        residual = torch.full((2, 50), .2, requires_grad=True)
        labels = torch.zeros(2, 50, dtype=torch.uint8)
        loss = multi.ap_pair_loss(scores, residual, labels, torch.tensor([2, 3]))
        self.assertAlmostEqual(float(loss.detach()), .001*.2**2, places=10)
        loss.backward()
        torch.testing.assert_close(scores.grad, torch.zeros_like(scores), atol=0, rtol=0)
        self.assertTrue(torch.all(residual.grad > 0))
        y = torch.zeros(2, 50, dtype=torch.uint8)
        y[0, 0] = 1
        w = multi.delta_ap_weights(y, scores.detach(), torch.tensor([2, 3]))
        self.assertEqual(float(w[1].sum()), 0)

    def test_top50_label_free_selection_and_original_array_identity(self):
        data, keys, info = fixture()
        a = multi.top50_from_arrays(data, keys, info)
        self.assertEqual(a['x'].shape, (2, 50, 21))
        self.assertEqual(a['customer_id'].tolist(), ['a', 'b'])
        self.assertEqual(a['article_id'][0].tolist(), list(range(100, 50, -1)))
        self.assertTrue(np.all(a['truth_count'] == 8))
        changed = {**data, 'target': 1-data['target'], 'truth_count': np.full(300, 101)}
        altered = keys.copy()
        altered['target'] = changed['target']
        altered['truth_count'] = changed['truth_count']
        b = multi.top50_from_arrays(changed, altered, info)
        np.testing.assert_array_equal(a['x'], b['x'])
        np.testing.assert_array_equal(a['article_id'], b['article_id'])
        self.assertFalse(np.array_equal(a['target'], b['target']))

    def test_missing_rank_and_wrong_array_alignment_fail_closed(self):
        data, keys, info = fixture()
        bad = keys.copy()
        bad.loc[99, 'rf'] = 51
        with self.assertRaises(ValueError):
            multi.top50_from_arrays(data, bad, info)
        bad = keys.copy()
        bad.loc[0, 'target'] = 1-bad.loc[0, 'target']
        with self.assertRaises(ValueError):
            multi.top50_from_arrays(data, bad, info)
        with self.assertRaises(ValueError):
            multi.top50_from_arrays(data, keys, {'features': ['target']+info['features'][1:]})

    def test_only_strictly_prior2019outer_supervision(self):
        meta = [{'year': 2019, 'role': 'outer', 'cutoff': '2019-12-18'}]
        multi.validate_training_metadata(meta, '2019-12-25')
        for bad in [
            [{'year': 2019, 'role': 'outer', 'cutoff': '2019-12-19'}],
            [{'year': 2019, 'role': 'inner', 'cutoff': '2019-11-20'}],
            [{'year': 2020, 'role': 'outer', 'cutoff': '2020-01-22'}],
            meta+meta,
        ]:
            with self.assertRaises(ValueError):
                multi.validate_training_metadata(bad, '2019-12-25')
        with self.assertRaises(ValueError):
            multi.validate_training_metadata(meta, '2020-09-16')

    def test_full_neutral_ranks_and_inactive_fallback_and_no_admission(self):
        data, keys, info = fixture()
        info['source_identity'] = {'scope': 'synthetic'}
        top = multi.top50_from_arrays(data, keys, info)
        stage = np.where(keys.user_history_events_12w > 0, keys.rf, keys.candidate_rank)
        baseline = np.mean([brute_ap(g.target.to_numpy()[np.argsort(stage[g.index])], 8)
                            for _, g in keys.groupby('customer_id', sort=False)])
        meta = {'cutoff': '2019-12-25', 'total_users': 3,
                'baseline_map_population_component': baseline}
        @contextmanager
        def connection():
            with duckdb.connect() as con:
                yield con
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(multi, 'connection', connection), \
             patch.object(multi, 'top50_data', lambda m: (top, info, keys.copy())):
            result = multi.score(meta, multi.Top50Residual(), np.zeros(21), np.ones(21), Path(tmp)/'neutral')
            with duckdb.connect() as con:
                actual = con.execute("SELECT * FROM read_parquet(?) ORDER BY customer_id,candidate_rank",
                                     [result['rank_path']]).fetchdf()
            np.testing.assert_array_equal(actual.final_rank.to_numpy(), stage)
            self.assertAlmostEqual(result['population_delta'], 0., places=15)
            class Nonneutral(torch.nn.Module):
                def forward(self, x, rank):
                    residual = torch.where(rank == 50, 2., -2.)
                    return -torch.log(rank)+residual, residual
            shifted = multi.score(meta, Nonneutral(), np.zeros(21), np.ones(21), Path(tmp)/'shifted')
            with duckdb.connect() as con:
                changed = con.execute('SELECT * FROM read_parquet(?)', [shifted['rank_path']]).fetchdf()
            self.assertTrue((changed.loc[changed.final_rank <= 12, 'stage1_rank'] <= 50).all())
            inactive = changed.user_history_events_12w == 0
            np.testing.assert_array_equal(changed.loc[inactive, 'final_rank'], changed.loc[inactive, 'candidate_rank'])


if __name__ == '__main__':
    unittest.main()
