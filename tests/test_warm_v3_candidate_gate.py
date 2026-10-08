"""Candidate gate mechanics and source-integrity regressions; no real outer data."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import torch

from hm_recsys import warm_v3_candidate_gate as gate
from hm_recsys.warm_v2_rank_fusion import final_rank_sql
from hm_recsys.warm_v3_gate import safe_training


def write_frame(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        con.register('f', frame)
        con.execute("COPY f TO '"+str(path).replace("'", "''")+"' (FORMAT PARQUET)")


@contextmanager
def connection():
    with duckdb.connect() as con:
        yield con


class Fixture:
    def __init__(self, root, missing_counterpart=False):
        self.root = root
        rows = []
        for user, events in [('u0', 6), ('u1', 0)]:
            for k in range(1, 21):
                rows.append({'customer_id': user, 'article_id': k, 'candidate_rank': k,
                    'target': int(k in (2, 18)), 'truth_count': 3,
                    'user_history_events_12w': events, 'user_unique_items_12w': min(events, 3),
                    'user_days_since_last_purchase': 5, 'r0': k, 'r1': 21-k, 'rf': k,
                    'latent': k/10., 'missing': 0})
        self.ranks = pd.DataFrame(rows)
        self.base = self.ranks[['customer_id', 'article_id']].copy()
        for f in gate.PAIR_FEATURES:
            self.base[f] = 1 if 'present' in f or 'is_new' in f else .2
        if missing_counterpart:
            self.base = self.base.iloc[1:].copy()
        self.rank_path = root/'ranks.parquet'
        self.base_path = root/'base.parquet'
        write_frame(self.ranks, self.rank_path)
        write_frame(self.base, self.base_path)
        self.meta = {'cutoff': '2019-11-20', 'year': 2019,
            'ranks_path': str(self.rank_path), 'total_users': 2}

    @contextmanager
    def patched(self):
        fixture = self
        class Engine:
            def __init__(self, year):
                self.year = year
                self.history = {'feature_cache': {'2019-11-20': {'artifact': {
                    'path': str(fixture.base_path), 'bytes': fixture.base_path.stat().st_size,
                    'sha256': 'fixture-source-contract'}}}}
            def base_path(self, cutoff):
                return fixture.base_path
        with patch.object(gate, 'ART', self.root/'artifacts'), \
             patch.object(gate, 'Engine', Engine), \
             patch.object(gate, 'connection', connection), \
             patch.object(gate, 'save_parquet', write_frame):
            yield


class CandidateGateTests(unittest.TestCase):
    def test_zero_initialized_gate_is_neutral_and_bounded(self):
        torch.manual_seed(1)
        model = gate.Gate(21)
        x = torch.randn(32, 21)
        scores = torch.rand(32, 2)
        values, alpha = model(x, scores)
        torch.testing.assert_close(alpha, torch.full((32,), .5), rtol=0, atol=0)
        torch.testing.assert_close(values, scores.mean(1), rtol=0, atol=1e-7)
        self.assertTrue(torch.all(values >= scores.min(1).values-1e-7))
        self.assertTrue(torch.all(values <= scores.max(1).values+1e-7))

    def test_gate_gradients_and_equal_expert_boundary(self):
        model = gate.Gate(3)
        x = torch.ones(4, 3)
        scores = torch.tensor([[.1, .9], [.2, .8], [.3, .7], [.4, .6]])
        value, alpha = model(x, scores)
        (-value.sum()).backward()
        self.assertTrue(torch.isfinite(model.net[-1].bias.grad).all())
        self.assertGreater(float(model.net[-1].bias.grad.abs().sum()), 0)
        self.assertEqual(float(model.net[0].weight.grad.abs().sum()), 0)
        # The first layer being zero-gradient at initialization follows from the
        # deliberately zero final weight; after one update it becomes trainable.
        with torch.no_grad():
            model.net[-1].weight.add_(.1)
        model.zero_grad()
        value, _ = model(x, scores)
        (-value.sum()).backward()
        self.assertGreater(float(model.net[0].weight.grad.abs().sum()), 0)
        equal = torch.full((4, 2), .4)
        v, _ = model(x, equal)
        torch.testing.assert_close(v, torch.full((4,), .4), atol=1e-7, rtol=0)

    def test_neutral_preserves_original_rrf_ties(self):
        # Original two-expert RRF gives an exact double tie. Independently rounded
        # float32 reciprocal scores previously reversed these candidate ranks.
        r0 = np.array([228, 260])
        r1 = np.array([132, 120])
        exact = 1/(60+r0)+1/(60+r1)
        self.assertEqual(exact[0], exact[1])
        actual = gate.blend_scores(r0, r1, np.full(2, .5))
        # This regression documents the production neutral policy, not merely a
        # tolerance on scores: a tie must use original candidate_rank, then item.
        self.assertEqual(np.argsort(-actual, kind='stable').tolist(), [0, 1])
        np.testing.assert_array_equal(actual, exact)

    def test_final_blend_matches_expert_endpoints_and_score_bounds(self):
        r0 = np.array([1, 2, 50, 150, 300])
        r1 = np.array([100, 2, 250, 3, 20])
        for alpha in [np.zeros(5), np.ones(5), np.array([.1, .3, .5, .7, .9])]:
            actual = gate.blend_scores(r0, r1, alpha)
            expected = 2*((1-alpha)/(60+r0)+alpha/(60+r1))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=7e-18)
            low = 2*np.minimum(1/(60+r0), 1/(60+r1))
            high = 2*np.maximum(1/(60+r0), 1/(60+r1))
            self.assertTrue(np.all(actual >= low-1e-15))
            self.assertTrue(np.all(actual <= high+1e-15))

    def test_final_ranking_keeps_inactive_fallback(self):
        f = pd.DataFrame({'customer_id': ['active']*20+['inactive']*20,
            'article_id': list(range(1, 21))*2, 'candidate_rank': list(range(1, 21))*2,
            'user_history_events_12w': [3]*20+[0]*20, 'score': list(range(1, 21))*2})
        with duckdb.connect() as con:
            con.register('f', f)
            out = con.execute(f'SELECT *,{final_rank_sql()} final_rank FROM f ORDER BY customer_id,final_rank').fetchdf()
        self.assertEqual(out[out.customer_id == 'active'].head(12).article_id.tolist(), list(range(20, 8, -1)))
        self.assertEqual(out[out.customer_id == 'inactive'].head(12).article_id.tolist(), list(range(1, 13)))

    def test_labels_are_targets_never_model_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Fixture(Path(tmp)/'first')
            second = Fixture(Path(tmp)/'second')
            second.ranks['target'] = 1-second.ranks.target
            second.ranks['truth_count'] = 99
            write_frame(second.ranks, second.rank_path)
            with first.patched():
                a, info = gate.arrays(first.meta)
            with second.patched():
                b, _ = gate.arrays(second.meta)
            self.assertEqual(len(info['features']), 21)
            self.assertNotIn('target', info['features'])
            self.assertNotIn('truth_count', info['features'])
            np.testing.assert_array_equal(a['x'], b['x'])
            np.testing.assert_array_equal(a['scores'], b['scores'])
            self.assertFalse(np.array_equal(a['target'], b['target']))

    def test_missing_candidate_counterpart_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp), missing_counterpart=True)
            with f.patched(), self.assertRaises((AssertionError, ValueError, RuntimeError)):
                gate.arrays(f.meta)

    def test_cached_arrays_reject_different_rank_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            other = f.root/'different_ranks.parquet'
            modified = f.ranks.copy()
            modified['r0'] = 21-modified.r0
            write_frame(modified, other)
            with f.patched():
                gate.arrays(f.meta)
                changed = {**f.meta, 'ranks_path': str(other)}
                with self.assertRaises((AssertionError, ValueError, RuntimeError)):
                    gate.arrays(changed)

    def test_pair_sampling_stays_inside_user_and_cutoff_groups(self):
        def data(seed):
            rng = np.random.default_rng(seed)
            n = 3*70
            targets = np.zeros(n, np.uint8)
            targets[[1, 5, 71, 75, 141, 145]] = 1
            return {'x': rng.normal(size=(n, 3)).astype(np.float32),
                'scores': rng.uniform(.1, .9, size=(n, 2)).astype(np.float32),
                'target': targets, 'offset': np.array([0, 70, 140, 210]),
                'hard': np.tile(np.r_[np.ones(50), np.zeros(20)], 3).astype(np.uint8),
                'truth_count': np.full(n, 3, np.int32),
                'active': np.r_[np.ones(140), np.zeros(70)].astype(np.uint8)}
        values = [data(1), data(2)]
        meta = [{'cutoff': '2019-02-20'}, {'cutoff': '2019-05-22'}]
        captured = []
        original_array = np.array
        def array(value, *args, **kwargs):
            if isinstance(value, list) and value and isinstance(value[0], tuple) and len(value[0]) == 3:
                captured.extend(value)
            return original_array(value, *args, **kwargs)
        def arrays(m):
            d = values[0 if m['cutoff'] == '2019-02-20' else 1]
            return d, {'features': ['a', 'b', 'c']}
        with patch.object(gate, 'arrays', arrays), patch.object(gate, 'budget', lambda *a, **k: 1), \
             patch.dict(gate.PARAMS, {'epochs': 1}), patch.object(gate.np, 'array', array):
            _, _, _, report = gate.fit(meta)
        self.assertEqual(len(captured), 8*32)
        for p, n, weight in captured:
            self.assertEqual(p//210, n//210)
            self.assertEqual((p % 210)//70, (n % 210)//70)
            d = values[p//210]
            self.assertEqual(d['target'][p % 210], 1)
            self.assertEqual(d['target'][n % 210], 0)
            self.assertEqual(d['active'][p % 210], 1)
            self.assertGreater(weight, 0)
        for p in {p for p, _, _ in captured}:
            negatives = [n for pp, n, _ in captured if pp == p]
            self.assertEqual(len(negatives), len(set(negatives)))
        self.assertEqual(report['pairs'], len(captured))

    def test_metadata_training_excludes_future_or_overlapping_labels(self):
        meta = [{'cutoff': '2019-11-20'}, {'cutoff': '2019-12-18'}, {'cutoff': '2019-12-19'}]
        selected = safe_training(meta, '2019-12-25')
        self.assertEqual([m['cutoff'] for m in selected], ['2019-11-20', '2019-12-18'])


if __name__ == '__main__':
    unittest.main()
