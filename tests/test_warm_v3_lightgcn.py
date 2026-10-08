import unittest
import numpy as np
from scipy.sparse import csr_matrix
import torch
from hm_recsys.warm_v3_lightgcn import (FORMAL_PARAMS, PARAMS, LightGCN, authorized_cutoffs,
    normalized_relation, sample_triplets, source_metadata, sparse_tensor)


class LightGCNTests(unittest.TestCase):
    def test_formal_resource_revision_preserves_pilot_and_other_parameters(self):
        self.assertEqual(PARAMS['formal_updates_if_authorized'], 200)
        self.assertEqual(FORMAL_PARAMS['updates'], 1000)
        for key in ('dimensions', 'layers', 'learning_rate', 'ego_l2', 'triplets_per_update', 'seed'):
            self.assertEqual(FORMAL_PARAMS[key], PARAMS[key])

    def graph(self):
        return csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], np.float32))

    def test_symmetric_normalization(self):
        s = normalized_relation(self.graph()).toarray()
        np.testing.assert_allclose(s, [[1/np.sqrt(2), .5, 0], [0, .5, 1/np.sqrt(2)]], rtol=1e-6)

    def test_sparse_dense_forward_and_gradient_parity(self):
        torch.manual_seed(4); model = LightGCN(2, 3, 4, 2)
        s = normalized_relation(self.graph()); a = sparse_tensor(s, 'cpu'); at = sparse_tensor(s.T.tocsr(), 'cpu')
        u, i = model.propagate(a, at)
        sparse_loss = u.square().sum() + i.square().sum()
        sparse_grad = torch.autograd.grad(sparse_loss, (model.user, model.item))
        dense = torch.from_numpy(s.toarray()); du, di = model.user, model.item
        us, ins = [du], [di]
        for _ in range(2):
            du, di = dense @ di, dense.T @ du; us.append(du); ins.append(di)
        du, di = sum(us)/3, sum(ins)/3
        self.assertTrue(torch.allclose(u, du)); self.assertTrue(torch.allclose(i, di))
        dense_grad = torch.autograd.grad(du.square().sum() + di.square().sum(), (model.user, model.item))
        for x, y in zip(sparse_grad, dense_grad): self.assertTrue(torch.allclose(x, y, atol=1e-6))

    def test_negative_sampler_excludes_all_known_positive_edges(self):
        m = self.graph(); u, p, n, rejected = sample_triplets(m, 1024, np.random.default_rng(4))
        self.assertTrue(np.all(np.asarray(m[u, p]).ravel() == 1))
        self.assertTrue(np.all(np.asarray(m[u, n]).ravel() == 0))
        self.assertGreater(rejected, 0)

    def test_actual_bpr_gradient_update(self):
        torch.manual_seed(3); m = self.graph(); s = normalized_relation(m)
        model = LightGCN(2, 3, 4, 2); before = model.user.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        u, p, n, _ = sample_triplets(m, 32, np.random.default_rng(7))
        args = [torch.from_numpy(v) for v in (u, p, n)]
        loss, *_ = model.loss(sparse_tensor(s, 'cpu'), sparse_tensor(s.T.tocsr(), 'cpu'), *args)
        loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(before, model.user))
        self.assertTrue(torch.isfinite(model.user).all())

    @unittest.skipUnless(__import__("os").environ.get("HM_RESEARCH_ASSETS") == "1", "requires withheld historical contracts/checkpoints and original research branch")
    def test_explicit_outer_cutoff_extension_is_exact_and_parameter_frozen(self):
        allowed = authorized_cutoffs()
        self.assertIn('2020-03-18', allowed)
        self.assertIn('2020-08-19', allowed)
        self.assertNotIn('2020-09-16', allowed)
        for cutoff in ('2020-03-18', '2020-08-19'):
            path, meta = source_metadata(cutoff)
            self.assertIn('bpr-match-v1', str(path))
            self.assertEqual(meta['cutoff'], cutoff)


if __name__ == '__main__': unittest.main()
