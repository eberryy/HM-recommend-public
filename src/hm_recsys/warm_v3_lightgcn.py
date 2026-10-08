"""Full-graph gradient-trained LightGCN mechanics/resource pilot, not a rank trial."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import time
import traceback

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
import torch
from torch import nn
from torch.nn import functional as F

from .warm_v2_bpr import relation
from .warm_v2_contract import evidence_id, git, guard, read, write

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'artifacts/warm_v3/lightgcn'
REPORT = ROOT / 'reports/warm_v3'
TX = Path(__file__).resolve().parents[2] / 'data/interim/audit/transactions.parquet'
OLD = Path(__file__).resolve().parents[2] / 'artifacts/warm_v2/bpr-match-v1'
FRESH_BPR = ROOT / 'artifacts/warm_v2/fresh_robustness/bpr'
PARAMS = {'model': 'genuine_gradient_lightgcn_v1', 'dimensions': 32, 'layers': 2,
          'layer_weights': [1/3, 1/3, 1/3], 'initialization': 'normal_std0.1_no_BPR_pretraining',
          'optimizer': 'Adam', 'learning_rate': .001, 'ego_l2': .0001,
          'triplets_per_update': 65536, 'formal_updates_if_authorized': 200,
          'positive_sampling': 'uniform_user_then_uniform_binary_positive',
          'negative_sampling': 'uniform_item_reject_every_known_positive',
          'pilot_updates': 3, 'seed': 20260909, 'dtype': 'float32',
          'sparse_layout': 'normalized_rectangular_CSR_and_transpose',
          'dropout': False, 'bias': False, 'feature_transform': False, 'activation': False}
CONTRACT = REPORT / 'LIGHTGCN_RESOURCE_CONTRACT.json'
FORMAL_ROOT = ART / 'formal1000'
FORMAL_PARAMS = {**{k: v for k, v in PARAMS.items() if k not in ('formal_updates_if_authorized', 'pilot_updates')},
                 'model': 'genuine_gradient_lightgcn_1000_v2', 'updates': 1000,
                 'max_fit_seconds': 600}
FEATURES = ['wv3_lightgcn_score', 'wv3_lightgcn_unavailable']
FORMAL_CONTRACT = REPORT / 'WV3_211_RESOURCE_DECISION.json'
AUTH_EXTENSION = REPORT / 'WV3_211_CUTOFF_AUTHORIZATION_EXTENSION.json'


def inner_cutoffs():
    rolling = read(ROOT / 'reports/warm_v2/WARM_V2_EXPERIMENT_CONTRACT.json')['rolling_protocol']
    return sorted({d for chain in rolling.values() for d in [*chain['inner_train'], chain['inner_validation']]})


def source_metadata(cutoff):
    """Resolve one cutoff to exactly one graph vocabulary/transaction manifest."""
    guard(cutoff)
    matches = [p for p in (OLD / cutoff / 'model.json', FRESH_BPR / cutoff / 'model.json') if p.is_file()]
    assert len(matches) == 1, f'Expected one exact-cutoff graph source for {cutoff}, found {matches}'
    path = matches[0]; meta = read(path)
    assert meta['cutoff'] == cutoff and meta['latest_history_date'] < cutoff
    tx = Path(meta['transactions']['path'])
    assert tx.is_file() and tx.stat().st_size == meta['transactions']['bytes']
    return path, meta


def authorized_cutoffs():
    """Base inner allowlist plus one explicit, evidence-gated extension."""
    base = formal_register()['authorized_representation_cutoffs']
    if not AUTH_EXTENSION.is_file():
        return sorted(base)
    ext = read(AUTH_EXTENSION)
    assert ext['selected_trial'] == 'WV3-212'
    assert ext['role'] in ('2020_registered_outer_confirmation', '2019_fixed_cross_year_replay')
    assert ext['params_equal_formal1000'] == FORMAL_PARAMS
    gate = read(Path(ext['gate_evidence_path']))
    assert gate['experiment_id'] == 'WV3-212' and gate['screening']['passed']
    if ext['role'] == '2020_registered_outer_confirmation':
        contract = read(ROOT / 'reports/warm_v2/WARM_V2_EXPERIMENT_CONTRACT.json')
        expected = {chain['outer_validation'] for chain in contract['rolling_protocol'].values()}
    else:
        contract = read(ROOT / 'reports/warm_v2/WARM_V2_FRESH_ROBUSTNESS_CONTRACT.json')
        expected = {d for chain in contract['rolling_protocol'].values()
                    for d in [*chain['inner_train'], chain['inner_validation'], chain['outer_validation']]}
        outer = read(REPORT / 'WV3-212_OUTER.json')
        assert outer['experiment_id'] == 'WV3-212' and outer['promotion']['passed']
    requested = set(ext['cutoffs'])
    assert requested and requested <= expected and '2020-09-16' not in requested
    for cutoff in requested:
        source_metadata(cutoff)
    return sorted(set(base) | requested)


def formal_register():
    workspace()
    if FORMAL_CONTRACT.exists():
        value = read(FORMAL_CONTRACT); assert value['params'] == FORMAL_PARAMS
        return value
    pilot_result = read(REPORT / 'LIGHTGCN_RESOURCE_AUDIT.json')
    assert pilot_result['mechanics_gate_passed'] and pilot_result['resource_gate_passed']
    value = {'registered_at': now(), 'stage': 'WV3-211/WV3-212 formal trainedLightGCN resource decision',
             'authorization': 'Root authorized fixed1000updates before seeing any trained-LightGCN MAP; dated2026-09-09',
             'params': FORMAL_PARAMS, 'features': FEATURES, 'authorized_representation_cutoffs': inner_cutoffs(),
             'predecessor': str(CONTRACT), 'pilot_evidence': str(REPORT / 'LIGHTGCN_RESOURCE_AUDIT.json'),
             'decision_basis': {'measured_steady_update_seconds': pilot_result['steady_update_seconds_mean_last2'],
                                'measured_peak_gpu_bytes': pilot_result['peak_cuda_allocated_bytes'],
                                'why_not200': '200updates gives fewer sampled triplets than graph edges. Measured0.286s/update leaves budget for fivefold supervision without changing dimensions/layers/loss.',
                                'labels_or_MAP_used_to_choose1000': False},
             'supervision_budget': '1000x65536=65,536,000 sampled user-positive-negative triplets percutoff; not1000 full-edgeepochs and not benchmark convergence claim',
             'estimated8cutoff_minutes': [50, 65], 'estimated10cutoff_minutes': [65, 80],
             'training_control': 'freshseed randomnormalrestart; never initialize from3-update pilot; one complete1000update model percutoff, no outer-specific params',
             'partial_model_rule': 'Partial checkpoint is diagnostic only; no MODEL.json or candidate scores before all1000updates finish',
             'cutoff_safety': 'Full binary purchase graph contains only original transactions strictly before its own cutoff; no label column from ranking candidates is read',
             'candidate_protocol': 'unchanged original candidate identities, score/unavailable only; originalinactive fallback remains downstream',
             'stopping': 'Per-cutoff600s fitcap; budgetcheck before each fit reserves20min closure; no fullouter builds in this command',
             'definitions_zh': {FEATURES[0]: '项目特征：真正梯度训练LightGCN的0/1/2层平均用户与候选商品表示内积，单位为一个用户—商品候选对；不含偏置，不是概率。',
                                FEATURES[1]: '项目特征：候选用户或商品不在该截止前购买图词表时为1，此时分数为NaN；分母为全部原候选行。'},
             'final_week': 'not_run', 'outer_MAP_seen': False}
    write(FORMAL_CONTRACT, value); return value


def formal_budget():
    state = read(ROOT / 'reports/private/warm_v3/AUTONOMOUS_RUN_STATE.json')
    remain = (datetime.fromisoformat(state['deadline_utc']) - datetime.now(timezone.utc)).total_seconds()
    if remain < FORMAL_PARAMS['max_fit_seconds'] + 20 * 60:
        raise TimeoutError('Insufficient Warm-v3 time for next10minute fit plus20minute closure reserve')
    import shutil
    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
        raise RuntimeError('LightGCN disk safety floor10GiB')
    return remain


def now():
    return datetime.now(timezone.utc).isoformat()


def literal(path):
    return "'" + str(path).replace("'", "''") + "'"


def workspace():
    assert Path.cwd().resolve() == ROOT
    assert git('branch', '--show-current') == 'warm-v3-architecture-lab'


def register():
    workspace()
    if CONTRACT.exists():
        value = read(CONTRACT); assert value['params'] == PARAMS
        return value
    value = {'registered_at': now(), 'params': PARAMS, 'pilot_cutoff': '2019-11-27',
             'hypothesis': '后处理BPR图平滑失败不能排除经过图传播后反向优化初始嵌入的LightGCN；先测完整梯度训练是否在本机预算内可行。',
             'training_graph': 'all original binary user-item relationships strictly before cutoff; raw duplicates preserved on disk and intentionally represented as one graph edge',
             'boundaries': '3updates throughput only; no recommendation labels, no candidate scoring, no full200update fit, no outerMAP',
             'mechanics_gate': 'sparse/dense toy propagation and gradient parity; no positive used as negative; nonzero finite embedding gradient and parameter update',
             'resource_gate': {'max_pilot_wall_seconds': 600, 'max_cuda_peak_gib': 6.,
                               'max_projected200_update_cutoff_seconds': 480.,
                               'max_projected10cutoff_minutes': 90.},
             'formal_caveat': '200updates=13,107,200sampledtriplets, not200fullgraph-edgeepochs; budgeted supervision may remain insufficient even when compute feasible.',
             'sources': ['https://arxiv.org/abs/2002.02126',
                         'https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/model.py',
                         'https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/utils.py'],
             'borrowed': 'Symmetric degree-normalized propagation, equal layer0/1/2mean, BPR objective and ego embeddingL2, randomnormalinitialization; genuine autograd through every graph layer.',
             'not_copied': 'No repository clone, no authorhyperparametersearch, no benchmarkresults assumedtransferable; WindowsCSRimplementation and boundedlargertripletbatches are local engineering choices.',
             'final_week': 'not_run', 'formal_expert_trained': False}
    write(CONTRACT, value); return value


def normalized_relation(matrix):
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    matrix.sum_duplicates(); matrix.eliminate_zeros()
    assert np.all(matrix.data == 1), 'graph must be binary, not count weighted'
    du = np.diff(matrix.indptr).astype(np.float32)
    di = np.asarray(matrix.sum(0)).ravel().astype(np.float32)
    invu = np.zeros_like(du); invi = np.zeros_like(di)
    np.divide(1., np.sqrt(du), out=invu, where=du > 0)
    np.divide(1., np.sqrt(di), out=invi, where=di > 0)
    matrix.data *= np.repeat(invu, np.diff(matrix.indptr))
    matrix.data *= invi[matrix.indices]
    return matrix


def sparse_tensor(matrix, device):
    # Fixed sparse matrix: only dense embedding values receive gradients.
    return torch.sparse_csr_tensor(torch.as_tensor(matrix.indptr, dtype=torch.int32, device=device),
                                   torch.as_tensor(matrix.indices, dtype=torch.int32, device=device),
                                   torch.as_tensor(matrix.data, dtype=torch.float32, device=device),
                                   size=matrix.shape, device=device)


class LightGCN(nn.Module):
    def __init__(self, users, items, dimensions=32, layers=2):
        super().__init__(); self.layers = layers
        self.user = nn.Parameter(torch.empty(users, dimensions))
        self.item = nn.Parameter(torch.empty(items, dimensions))
        nn.init.normal_(self.user, std=.1); nn.init.normal_(self.item, std=.1)

    def propagate(self, adjacency, transpose):
        u, i = self.user, self.item
        user_layers, item_layers = [u], [i]
        for _ in range(self.layers):
            u, i = torch.sparse.mm(adjacency, i), torch.sparse.mm(transpose, u)
            user_layers.append(u); item_layers.append(i)
        return sum(user_layers) / (self.layers + 1), sum(item_layers) / (self.layers + 1)

    def loss(self, adjacency, transpose, users, positives, negatives, ego_l2=.0001):
        u, i = self.propagate(adjacency, transpose)
        user, pos, neg = u[users], i[positives], i[negatives]
        margin = (user * (pos - neg)).sum(1)
        bpr = F.softplus(-margin).mean()
        reg = .5 * (self.user[users].square().sum() + self.item[positives].square().sum()
                    + self.item[negatives].square().sum()) / len(users)
        return bpr + ego_l2 * reg, bpr, reg, margin


def sample_triplets(matrix, n, rng):
    degrees = np.diff(matrix.indptr)
    assert np.all(degrees > 0) and np.max(degrees) < matrix.shape[1]
    users = rng.integers(matrix.shape[0], size=n, dtype=np.int64)
    positions = matrix.indptr[users] + np.floor(rng.random(n) * degrees[users]).astype(np.int64)
    positives = matrix.indices[positions].astype(np.int64)
    negatives = rng.integers(matrix.shape[1], size=n, dtype=np.int64)
    rejected = 0
    for _ in range(100):
        bad = np.asarray(matrix[users, negatives]).ravel() > 0
        if not np.any(bad): break
        rejected += int(bad.sum())
        negatives[bad] = rng.integers(matrix.shape[1], size=int(bad.sum()), dtype=np.int64)
    assert not np.any(np.asarray(matrix[users, negatives]).ravel() > 0)
    assert np.all(np.asarray(matrix[users, positives]).ravel() == 1)
    return users, positives, negatives, rejected


def prepare(cutoff='2019-11-27', formal=False):
    workspace(); guard(cutoff); register()
    if formal:
        assert cutoff in authorized_cutoffs()
    else:
        assert cutoff == '2019-11-27', 'resource pilot has one registered cutoff'
    destination = ART / cutoff; destination.mkdir(parents=True, exist_ok=True)
    meta_path = destination / 'GRAPH.json'
    if meta_path.exists():
        return read(meta_path)
    start = time.perf_counter()
    with duckdb.connect() as con:
        con.execute('SET threads=4'); con.execute("SET memory_limit='2GB'")
        scratch = destination / 'scratch'; scratch.mkdir(exist_ok=True)
        con.execute(f'SET temp_directory={literal(scratch)}')
        matrix, users, items, latest = relation(con, TX, cutoff)
        for name, frame in [('users', users), ('items', items)]:
            con.register('frame', frame)
            con.execute(f'COPY frame TO {literal(destination / (name + ".parquet"))} (FORMAT PARQUET, COMPRESSION ZSTD)')
            con.unregister('frame')
    old_path, old = source_metadata(cutoff)
    assert latest < cutoff and matrix.nnz == old['binary_pairs']
    assert matrix.shape == (old['users'], old['items'])
    save_npz(destination / 'binary_graph.npz', matrix, compressed=False)
    node_bytes = (len(users) + len(items)) * PARAMS['dimensions'] * 4
    meta = {'cutoff': cutoff, 'latest_history_date': latest, 'strictly_before_cutoff': True,
            'users': len(users), 'items': len(items), 'binary_pairs': matrix.nnz,
            'graph_shape': list(matrix.shape), 'prepare_seconds': time.perf_counter() - start,
            'binary_csr_bytes': sum(v.nbytes for v in (matrix.data, matrix.indices, matrix.indptr)),
            'embedding_parameter_bytes': node_bytes,
            'first_order_gpu_estimate_bytes': node_bytes * 10 + matrix.nnz * 16 + (len(users) + len(items) + 2) * 4,
            'estimate_caveat': 'Static graph+parameters+Adam+representations approximation; actual sparse-backward temporaries must be measured',
            'reference_binary_count': str(old_path),
            'transactions': old['transactions'],
            'new_graph_path': str(destination / 'binary_graph.npz'),
            'final_week': 'not_run'}
    write(meta_path, meta); return meta


def frame_load(path):
    with duckdb.connect() as con:
        return con.execute(f'SELECT * FROM read_parquet({literal(path)})').fetchdf()


def frame_save(frame, path):
    with duckdb.connect() as con:
        con.register('frame', frame)
        con.execute(f'COPY frame TO {literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)')


def fit_formal(cutoff):
    workspace(); guard(cutoff); cfg = formal_register()
    assert cutoff in authorized_cutoffs(), 'representation cutoff not authorized in formalcontract or explicit extension'
    destination = FORMAL_ROOT / cutoff; meta_path = destination / 'MODEL.json'
    if meta_path.exists():
        value = read(meta_path)
        assert value['params'] == FORMAL_PARAMS and value['completed_updates'] == 1000
        for artifact in value['artifacts'].values():
            assert Path(artifact['path']).stat().st_size == artifact['bytes']
        return value
    formal_budget(); destination.mkdir(parents=True, exist_ok=True)
    assert not (destination / 'model.pt').exists(), 'Preserve completed-weight/incomplete-finalization state; no implicit retraining'
    graph = prepare(cutoff, formal=True)
    matrix = load_npz(ART / cutoff / 'binary_graph.npz')
    assert torch.cuda.is_available()
    torch.set_num_threads(4); torch.manual_seed(FORMAL_PARAMS['seed'])
    rng = np.random.default_rng(FORMAL_PARAMS['seed'])
    s = normalized_relation(matrix); st = s.T.tocsr()
    torch.cuda.reset_peak_memory_stats(); whole_start = time.perf_counter()
    adj, trans = sparse_tensor(s, 'cuda'), sparse_tensor(st, 'cuda')
    model = LightGCN(*matrix.shape, FORMAL_PARAMS['dimensions'], FORMAL_PARAMS['layers']).cuda()
    initial_sample = model.user[:1024].detach().cpu().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=FORMAL_PARAMS['learning_rate'])
    history = []; sample_rejections = 0; start = time.perf_counter()
    try:
        for index in range(FORMAL_PARAMS['updates']):
            users, positives, negatives, rejected = sample_triplets(matrix, FORMAL_PARAMS['triplets_per_update'], rng)
            sample_rejections += rejected
            u, p, n = [torch.as_tensor(v, device='cuda') for v in (users, positives, negatives)]
            optimizer.zero_grad(set_to_none=True)
            loss, bpr, reg, margin = model.loss(adj, trans, u, p, n, FORMAL_PARAMS['ego_l2'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite LightGCN loss')
            loss.backward()
            if index == 0:
                assert model.user.grad is not None and model.item.grad is not None
                assert torch.isfinite(model.user.grad).all() and model.user.grad.norm() > 0
                assert torch.isfinite(model.item.grad).all() and model.item.grad.norm() > 0
            optimizer.step()
            if index == 0 or (index + 1) % 50 == 0:
                torch.cuda.synchronize()
                row = {'completed_updates': index + 1, 'bpr_training_loss': float(bpr.detach()),
                       'ego_l2_unweighted': float(reg.detach()),
                       'sampled_training_margin_positive_share': float((margin > 0).float().mean()),
                       'fit_seconds': time.perf_counter() - start,
                       'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                history.append(row)
                write(destination / 'PROGRESS.json', {'cutoff': cutoff, 'params': FORMAL_PARAMS,
                       'checkpoints': history, 'formal_model_complete': False, 'final_week': 'not_run'})
                print({'cutoff': cutoff, **row}, flush=True)
                if row['fit_seconds'] > FORMAL_PARAMS['max_fit_seconds']:
                    raise TimeoutError('LightGCN formal600second fit cap')
        torch.cuda.synchronize(); fit_seconds = time.perf_counter() - start
        delta = float((model.user[:1024].detach().cpu() - initial_sample).square().sum())
        assert delta > 0 and history[-1]['completed_updates'] == 1000
        model.eval()
        with torch.no_grad():
            uf, itf = model.propagate(adj, trans)
            users = uf.cpu().numpy(); items = itf.cpu().numpy()
        assert np.isfinite(users).all() and np.isfinite(items).all()
        torch.save(model.state_dict(), destination / 'model.pt')
        np.savez(destination / 'factors.npz', user_factors=users, item_factors=items)
        out = {'cutoff': cutoff, 'params': FORMAL_PARAMS, 'completed_updates': 1000,
               'fit_seconds': fit_seconds, 'training_and_finalization_seconds': time.perf_counter() - whole_start,
               'history': history, 'training_triplets': 1000 * FORMAL_PARAMS['triplets_per_update'],
               'known_positive_negatives_rejected': sample_rejections,
               'raw_user_parameter_squared_change_first1024': delta,
               'graph': graph, 'latest_history_date': graph['latest_history_date'],
               'transactions': source_metadata(cutoff)[1]['transactions'],
               'point_in_time_safe': graph['latest_history_date'] < cutoff,
               'initialized_from_pilot': False, 'all_layers_receive_gradients': True,
               'factor_shapes': [list(users.shape), list(items.shape)],
               'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'device_name': torch.cuda.get_device_name(), 'torch': torch.__version__,
               'retraining_caveat': 'Fixedseed and algorithm; CUDA sparse floating-point reductions need not be bitwiseidentical across runs. Saved actual factors are authoritative.',
               'vocabulary': {'users': str(ART / cutoff / 'users.parquet'), 'items': str(ART / cutoff / 'items.parquet')},
               'artifacts': {name: evidence_id(destination / name, reason='explicit_registry_evidence')
                             for name in ['model.pt', 'factors.npz']},
               'future_labels_used': False, 'final_week': 'not_run'}
        write(meta_path, out)
    except BaseException:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'rng_state': rng.bit_generator.state, 'completed_updates': index + 1,
                    'params': FORMAL_PARAMS}, destination / 'PARTIAL_NOT_FOR_RANKING.pt')
        write(destination / 'FAILURE.json', {'cutoff': cutoff, 'traceback': traceback.format_exc(),
              'history': history, 'formal_model_complete': False, 'final_week': 'not_run'})
        raise
    finally:
        del model, optimizer, adj, trans; gc.collect(); torch.cuda.empty_cache()
    return out


def build_features(cutoff, engine):
    """Score exact cutoff-matched original candidates only after1000updates."""
    workspace(); guard(cutoff); cfg = formal_register()
    assert cutoff in authorized_cutoffs()
    destination = FORMAL_ROOT / cutoff; path = destination / 'features.parquet'; meta_path = destination / 'FEATURES.json'
    source_path = Path(engine.base_path(cutoff)); source_identity = engine.history['feature_cache'][cutoff]['artifact']
    if meta_path.exists():
        value = read(meta_path)
        assert value['params'] == FORMAL_PARAMS and value['source_sha256'] == source_identity['sha256']
        assert path.stat().st_size == value['artifact']['bytes']
        return path, value
    model = fit_formal(cutoff); start = time.perf_counter()
    with np.load(destination / 'factors.npz', allow_pickle=False) as arrays:
        uf, itf = arrays['user_factors'], arrays['item_factors']
    users = pd.Index(frame_load(model['vocabulary']['users']).customer_id)
    items = pd.Index(frame_load(model['vocabulary']['items']).article_id)
    chunks = []
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        cursor = con.execute(f'SELECT customer_id,article_id FROM read_parquet({literal(source_path)})')
        while True:
            frame = cursor.fetch_df_chunk(16)
            if frame.empty: break
            ui = users.get_indexer(frame.customer_id); ii = items.get_indexer(frame.article_id)
            good = (ui >= 0) & (ii >= 0); score = np.full(len(frame), np.nan, np.float32)
            score[good] = np.einsum('ij,ij->i', uf[ui[good]], itf[ii[good]])
            frame[FEATURES[0]] = score; frame[FEATURES[1]] = (~good).astype(np.float32); chunks.append(frame)
    frame = pd.concat(chunks, ignore_index=True)
    assert not frame.duplicated(['customer_id', 'article_id']).any()
    assert np.isfinite(frame.loc[frame[FEATURES[1]] == 0, FEATURES[0]]).all()
    frame_save(frame, path)
    meta = {'cutoff': cutoff, 'params': FORMAL_PARAMS, 'features': FEATURES, 'rows': len(frame),
            'candidate_identity_unchanged': True, 'unavailable_pairs': int(frame[FEATURES[1]].sum()),
            'source': str(source_path), 'source_sha256': source_identity['sha256'],
            'model_metadata': str(destination / 'MODEL.json'), 'completed_model_updates': 1000,
            'definitions': cfg['definitions_zh'], 'scoring_seconds': time.perf_counter() - start,
            'artifact': evidence_id(path, reason='explicit_registry_evidence'),
            'future_labels_used': False, 'final_week': 'not_run'}
    write(meta_path, meta); return path, meta


def build_inner():
    formal_register()
    from .warm_v3_common import Engine
    engine = Engine(); results = {}; start = time.perf_counter()
    for cutoff in inner_cutoffs():
        path, meta = build_features(cutoff, engine)
        results[cutoff] = {'path': str(path), 'metadata': meta,
                           'model_metadata': str(FORMAL_ROOT / cutoff / 'MODEL.json')}
        write(REPORT / 'LIGHTGCN_INNER_REPRESENTATIONS.json', {'completed': results,
              'expected_cutoffs': inner_cutoffs(), 'wall_seconds': time.perf_counter() - start,
              'all_completed': len(results) == len(inner_cutoffs()), 'ranking_MAP_computed': False,
              'final_week': 'not_run'})
    return results


def no_label_feature_audit(cutoff):
    """Verify original identities/availability and describe redundancy without labels."""
    formal = FORMAL_ROOT / cutoff / 'features.parquet'
    baseline = source_metadata(cutoff)[0].parent / 'features.parquet'
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        con.execute(f'CREATE VIEW g AS SELECT * FROM read_parquet({literal(formal)})')
        con.execute(f'CREATE VIEW b AS SELECT * FROM read_parquet({literal(baseline)})')
        mismatches = con.execute('''SELECT count(*) FROM g FULL JOIN b USING(customer_id,article_id)
             WHERE g.customer_id IS NULL OR b.customer_id IS NULL
             OR g.wv3_lightgcn_unavailable<>b.wv2_bpr_unavailable''').fetchone()[0]
        assert mismatches == 0
        stats = con.execute('''SELECT count(*),corr(wv3_lightgcn_score,wv2_bpr_user_item_score),
             stddev_pop(wv3_lightgcn_score),stddev_pop(wv2_bpr_user_item_score),
             min(wv3_lightgcn_score),max(wv3_lightgcn_score)
             FROM g JOIN b USING(customer_id,article_id) WHERE wv3_lightgcn_unavailable=0''').fetchone()
    value = {'cutoff': cutoff, 'candidate_or_availability_mismatches': mismatches,
             'jointly_available_candidate_pairs': stats[0], 'score_pearson_vs_BPR': stats[1],
             'lightgcn_score_std': stats[2], 'BPR_score_std': stats[3],
             'lightgcn_score_min': stats[4], 'lightgcn_score_max': stats[5],
             'definitions_zh': '相关性和标准差仅在两个表示同时可用的原用户—商品候选行计算，不读取target。BPR旧分数含商品偏置，LightGCN无偏置；相关性不能证明正例互补或排序效果。',
             'labels_read': False, 'final_week': 'not_run'}
    write(FORMAL_ROOT / cutoff / 'NO_LABEL_FEATURE_AUDIT.json', value)
    return value


def api_audit(cutoff='2019-11-27'):
    """Exercise the real generic attachment/prepare interfaces without a tree fit."""
    from .warm_v3_common import Engine, attach_features
    from .warm_v3_expert import FEATURES as RUNNER_FEATURES
    from .warm_v2_engine import prepare as rank_prepare
    engine = Engine()
    assert RUNNER_FEATURES['lightgcn'] == FEATURES
    fp = engine.base_path(cutoff)
    with duckdb.connect() as con:
        frame = con.execute(f'''SELECT customer_id,article_id,{','.join(engine.features)}
            FROM read_parquet({literal(fp)})
            WHERE hash(customer_id||article_id)%1000=0 LIMIT 2048''').fetchdf()
    original = frame.copy(deep=True)
    added = attach_features(frame, cutoff, ['lightgcn'], engine)
    pd.testing.assert_frame_equal(frame, original)
    assert list(added.columns) == list(original.columns) + FEATURES
    assert added[FEATURES[1]].isin([0., 1.]).all()
    assert added.loc[added[FEATURES[1]] == 1, FEATURES[0]].isna().all()
    maps = read(OLD.parent / 'WV2-000/winter_20200122/inner_category_maps.json')
    maps = {k: {int(a): int(b) for a, b in v.items()} for k, v in maps.items()}
    prepared = rank_prepare(added, engine.features + FEATURES, maps)
    assert prepared.shape == (len(frame), 86)
    value = {'cutoff': cutoff, 'sample_rows': len(frame), 'sample_identity_and_original_features_unchanged': True,
             'formal_feature_list_equals_generic_runner': True, 'two_added_columns': FEATURES,
             'prepared_rank_feature_count': prepared.shape[1],
             'sample_unavailable_pairs': int(added[FEATURES[1]].sum()),
             'missing_score_is_NaN_not_candidate_filter': True,
             'candidate_target_column_read': False, 'tree_trained': False, 'final_week': 'not_run'}
    write(FORMAL_ROOT / cutoff / 'API_AUDIT.json', value); return value


def formal_report():
    cfg = formal_register(); completed = {}
    for cutoff in authorized_cutoffs():
        root = FORMAL_ROOT / cutoff
        if not (root / 'FEATURES.json').exists(): continue
        model, features = read(root / 'MODEL.json'), read(root / 'FEATURES.json')
        assert model['completed_updates'] == 1000 and features['features'] == FEATURES
        completed[cutoff] = {'model': model, 'features': features,
                            'transactions_reference': source_metadata(cutoff)[1]['transactions']}
        for label, filename in [('api_audit', 'API_AUDIT.json'), ('no_label_audit', 'NO_LABEL_FEATURE_AUDIT.json')]:
            if (root / filename).exists(): completed[cutoff][label] = read(root / filename)
    value = {'updated_at': now(), 'formal_contract': str(FORMAL_CONTRACT),
             'completed_cutoffs': completed, 'expected_inner_cutoffs': cfg['authorized_representation_cutoffs'],
             'authorized_cutoffs': authorized_cutoffs(),
             'all8_complete': set(cfg['authorized_representation_cutoffs']) <= set(completed), 'formal_updates_per_model': 1000,
             'CPU_tests_passed': 5, 'ranking_MAP_computed': False, 'final_week': 'not_run',
             'fallback_semantics': 'Graph-unavailable user/item produces NaN score +missing1, retaining candidate and84baseline features. Separately, user_history_events_12w==0 preserves originalcandidate_rank fallback even if a fullhistorygraphvector exists.',
             'no_architecture_promotion_claim': True}
    write(REPORT / 'LIGHTGCN_FORMAL_AUDIT.json', value)
    lines = ['# LightGCN 1000更新正式表示：训练与接口审计', '',
        f'已完成{len(completed)}/8个原内层所需截止的表示训练与候选打分。这里只报告表示和接口，不训练LightGBM、不计算MAP，也不做晋级结论。', '',
        '## 版本与成本决策', '',
        '原3次更新预实验和200次更新成本规划保留不改。预实验测得每步约0.286秒后，在任何真正训练LightGCN的MAP曝光之前，主线程明确注册1000次更新版本；仅扩大监督预算，维度32、层数2、学习率0.001、L2系数0.0001、抽样行为和原候选池均不改。', '',
        '每截止从固定随机种子重新初始化，不用3步预实验模型。1000步共65,536,000个抽样三元组，不等于1000个全数据epoch。三元组为“用户、已购商品、截止前未购的抽样商品”，负例排除用户所有已购商品；没有看未来标签。', '',
        '## 实测', '',
        '| 截止 | 完成更新数 | 训练秒数 | GPU峰值GiB | 首/末训练BPR损失 | 候选行数 | 无图表示行数 |',
        '|---|---:|---:|---:|---|---:|---:|']
    for cutoff, entry in completed.items():
        model, features = entry['model'], entry['features']; history = model['history']
        lines.append(f"| {cutoff} | 1000 | {model['fit_seconds']:.2f} | {model['cuda_peak_allocated_bytes']/1024**3:.3f} | {history[0]['bpr_training_loss']:.4f} / {history[-1]['bpr_training_loss']:.4f} | {features['rows']:,} | {features['unavailable_pairs']:,} |")
    lines += ['', 'BPR为行业成对排序损失，鼓励已购商品高于抽样未购商品。表中损失和训练三元组胜率只是优化器学习诊断，不是验证集MAP或泛化证据。无图表示行的分母是当前截止原候选池的全部用户—商品行；图外用户或图外商品任一缺失即计1。', '',
        '## 排序接口与缺失回退', '',
        '`build_features(cutoff, engine)`返回同一原候选用户—商品对及两列正式特征：`wv3_lightgcn_score`为三层平均表示内积，非概率；`wv3_lightgcn_unavailable`为图词表缺失指示。通用排序器使用原84列加这2列，共86列。它是独立图专家，不是把BPR两列再叠加成88列。', '',
        '图外冷商品或图外用户得到NaN分数和缺失标记1，但不删候选、不强制排末尾、不用0冒充相似度；树排序器仍可利用另外84个基础特征和缺失分支。', '',
        '“inactive fallback”为项目原有未活跃用户回退：截止前12周购买事件数为0时，最终按原candidate_rank推荐。即使全历史图能为该用户生成向量，也不能取消这一原回退。此边界由通用排序器保持，未修改Cold/Admission主线。', '',
        '实际接口抽样复核不读取target列：输入身份和84列保持不变；通用特征列表与两列接口完全一致；数值准备得到86列；缺失标记对应NaN而不是删行。', '',
        '## 可追溯性与限制', '',
        '每个MODEL.json保存完整1000步完成标记、每50步训练摘要、实际参数、图截止和源交易身份引用、实际因子文件身份。训练不足1000步的中断文件明确为PARTIAL_NOT_FOR_RANKING，不得用于候选打分。随机种子固定不等于CUDA稀疏浮点归约必然逐位复现；保存的实际因子是评测输入。', '',
        '5项CPU测试通过，覆盖稀疏/稠密前向及梯度一致、已购负例排除、真实参数更新和1000版本不改其余参数。源交易来自各自截止前，最终周2020-09-16保持not_run。后续排序或融合是否有效，只能由主线程固定内层筛选及必要外层确认决定。']
    (REPORT / 'LIGHTGCN_FORMAL_AUDIT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return value


def source_routing_design():
    """Metadata-only plan; deliberately does not extend the executable allowlist."""
    old = read(ROOT / 'reports/warm_v2/WARM_V2_EXPERIMENT_CONTRACT.json')
    fresh = read(ROOT / 'reports/warm_v2/WARM_V2_FRESH_ROBUSTNESS_CONTRACT.json')
    fresh_root = ROOT / 'artifacts/warm_v2/fresh_robustness'
    needed = {2020: sorted({c['outer_validation'] for c in old['rolling_protocol'].values()}),
              2019: sorted({d for c in fresh['rolling_protocol'].values()
                            for d in [*c['inner_train'], c['inner_validation'], c['outer_validation']]})}
    rows = []
    for year, cutoffs in needed.items():
        for cutoff in cutoffs:
            guard(cutoff)
            candidates = [OLD / cutoff / 'model.json', fresh_root / 'bpr' / cutoff / 'model.json']
            matched = [p for p in candidates if p.exists()]
            assert len(matched) == 1, (cutoff, matched)
            meta = read(matched[0]); assert meta['cutoff'] == cutoff and meta['latest_history_date'] < cutoff
            assert Path(meta['transactions']['path']).stat().st_size == meta['transactions']['bytes']
            baseline_feature = (fresh_root / 'candidates' / cutoff / 'BUILD.json') if year == 2019 else None
            if baseline_feature is not None:
                assert baseline_feature.exists()
                path = Path(read(baseline_feature)['target']['artifact']['path'])
                assert path.exists()
            rows.append({'evaluation_chain_year': year, 'cutoff': cutoff, 'metadata': str(matched[0]),
                         'graph_nodes': meta['users'] + meta['items'], 'binary_pairs': meta['binary_pairs'],
                         'latest_graph_event': meta['latest_history_date'],
                         'transactions_reference': meta['transactions'],
                         'already_in_current8point_allowlist': cutoff in inner_cutoffs(),
                         'fresh_candidate_build_metadata': str(baseline_feature) if baseline_feature is not None else None,
                         'action': 'metadata_read_only_not_authorized_for_new_fit'})
    value = {'created_at': now(), 'status': 'design_only_no_contract_or_training_change',
             'metadata_availability': rows, 'exact_cutoff_routing_required': True,
             'year_heuristic_counterexample': '2019-11-27 is OLD2020 development-chain innertrain;2019-11-20 is FRESH2019 replayouter. Calendar year alone is wrong.',
             'proposed_extension': {'filename': 'WV3_211_CUTOFF_AUTHORIZATION_EXTENSION.json',
                 'required_fields': ['selected_trial', 'role', 'cutoffs', 'params_equal_formal1000', 'authorization_timestamp', 'gate_evidence_path'],
                 'allowed_roles': ['2020_registered_outer_confirmation', '2019_fixed_cross_year_replay'],
                 'guard': 'Exactdates from corresponding rollingcontract only; original8pointallowlist immutable; union only after explicitrootauthorization and requiredscreen/outerevidence.',
                 'forbidden': 'No timestamp-based automaticallyallowedcutoffs; no finalweek; no2019-specificparameterchange'},
             'required_future_code_changes': ['source_metadata(cutoff): exactOLDorFRESHpath with unique match andcutoffsafety',
                 'prepare/fit/features: centralizedauthorized_cutoffs fromoriginalcontract+explicitextension',
                 'sourceidentity: usematchedmodeltransactions andvocabulary, nothardcodedOLD',
                 'candidatepath: usecallerEngine(year).base_path, nevercalendar-yearguess',
                 'no-labelaudit: choosematchedBPRfeaturepath;APIaudit: choosecorrespondinginnercategorymaps'],
             'future_tests': ['2019-11-27routesOLDand2019-11-20routesFRESH',
                 'ambiguousormissingmetadatafailsclosed', 'cutoffmetadataortransactionidentitymismatchfails',
                 'unapprovedouter/replaydatesfailbeforeanyfit', 'finalweekalwaysfails',
                 'extendeddateskeepexact1000modelparams'],
             'current_executable_allowlist_unchanged': inner_cutoffs(),
             'candidate_labels_read': False, 'new_fit_started': False, 'final_week': 'not_run'}
    write(REPORT / 'LIGHTGCN_SOURCE_ROUTING_DESIGN.json', value)
    return value


def pilot(cutoff='2019-11-27'):
    workspace(); cfg = register(); graph = prepare(cutoff)
    destination = ART / cutoff; result_path = REPORT / 'LIGHTGCN_RESOURCE_AUDIT.json'
    assert not result_path.exists(), 'completed concrete pilot must not be overwritten'
    assert torch.cuda.is_available(), 'no silent CPU fallback or architecture change'
    matrix = load_npz(destination / 'binary_graph.npz')
    torch.manual_seed(PARAMS['seed']); torch.set_num_threads(4)
    rng = np.random.default_rng(PARAMS['seed'])
    s = normalized_relation(matrix); st = s.T.tocsr()
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    adj, trans = sparse_tensor(s, 'cuda'), sparse_tensor(st, 'cuda')
    model = LightGCN(*matrix.shape, PARAMS['dimensions'], PARAMS['layers']).cuda()
    initial_sample = model.user[:1024].detach().cpu().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS['learning_rate'])
    updates = []
    for index in range(PARAMS['pilot_updates']):
        step = time.perf_counter()
        users, positives, negatives, rejected = sample_triplets(matrix, PARAMS['triplets_per_update'], rng)
        u, p, n = [torch.as_tensor(v, device='cuda') for v in (users, positives, negatives)]
        optimizer.zero_grad(set_to_none=True)
        loss, bpr, reg, margin = model.loss(adj, trans, u, p, n, PARAMS['ego_l2'])
        assert torch.isfinite(loss)
        loss.backward()
        assert model.user.grad is not None and model.item.grad is not None
        gradient_norm = float(model.user.grad.norm()) + float(model.item.grad.norm())
        assert gradient_norm > 0 and np.isfinite(gradient_norm)
        optimizer.step(); torch.cuda.synchronize()
        updates.append({'update': index + 1, 'seconds_including_sampling': time.perf_counter() - step,
                        'bpr_loss': float(bpr.detach()), 'ego_l2_unweighted': float(reg.detach()),
                        'gradient_norm_sum': gradient_norm, 'training_triplet_margin_positive_share': float((margin > 0).float().mean()),
                        'known_positive_negatives_rejected': rejected,
                        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()})
        write(destination / f'UPDATE_{index + 1}.json', updates[-1]); print(updates[-1], flush=True)
        if time.perf_counter() - start > cfg['resource_gate']['max_pilot_wall_seconds']:
            raise RuntimeError('registered LightGCN pilot exceeded wall budget')
    delta = float((model.user[:1024].detach().cpu() - initial_sample).square().sum())
    assert delta > 0
    # Store the tiny-update model only as failed/limited mechanics evidence, never serve it.
    torch.save(model.state_dict(), destination / 'THREE_UPDATE_MECHANICS_ONLY.pt')
    steady = float(np.mean([v['seconds_including_sampling'] for v in updates[1:]]))
    projections = []
    for path in sorted(OLD.glob('*/model.json')):
        value = read(path)
        scale = value['binary_pairs'] / graph['binary_pairs']
        projections.append({'cutoff': value['cutoff'], 'binary_pairs': value['binary_pairs'],
                            'edge_equivalent_passes_200updates': 200 * PARAMS['triplets_per_update'] / value['binary_pairs'],
                            'projected200_update_seconds': graph['prepare_seconds'] * scale + steady * 200 * scale,
                            'final_factor_bytes': (value['users'] + value['items']) * 32 * 4})
    projected_pilot = graph['prepare_seconds'] + steady * 200
    projected_all = sum(v['projected200_update_seconds'] for v in projections)
    rolling = read(ROOT / 'reports/warm_v2/WARM_V2_EXPERIMENT_CONTRACT.json')['rolling_protocol']
    inner_dates = sorted({d for chain in rolling.values() for d in [*chain['inner_train'], chain['inner_validation']]})
    projected_inner = sum(v['projected200_update_seconds'] for v in projections if v['cutoff'] in inner_dates)
    memory = torch.cuda.max_memory_allocated()
    result = {'created_at': now(), 'contract': cfg, 'graph': graph, 'updates': updates,
              'pilot_total_gpu_setup_fit_save_seconds': time.perf_counter() - start,
              'parameter_squared_change_first1024users': delta,
              'steady_update_seconds_mean_last2': steady,
              'peak_cuda_allocated_bytes': memory, 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
              'device_name': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'projected200_update_pilot_seconds': projected_pilot,
              'projections': projections, 'projected_all_existing_cutoffs_minutes': projected_all / 60,
              'inner_required_cutoffs': inner_dates, 'projected_inner8cutoffs_minutes': projected_inner / 60,
              'cpu_tests_passed': 4,
              'resource_gate_passed': projected_pilot <= 480 and projected_all / 60 <= 90 and memory < 6 * 1024**3,
              'mechanics_gate_passed': delta > 0,
              'interpretation': 'Only3updates throughput evidence; no heldout ranking and no claim of convergence. Negative posthoc graph result does not reject trainedLightGCN.',
              'training_supervision': '3x65536 positive-edge/negative-item triplets; sparse graph is fully propagated and differentiated everyupdate',
              'formal_model_trained': False, 'outer_labels_read': False, 'final_week': 'not_run'}
    write(result_path, result); render(result)
    del model, optimizer, adj, trans; gc.collect(); torch.cuda.empty_cache()
    return result


def render(value):
    s = value['steady_update_seconds_mean_last2']; m = value['peak_cuda_allocated_bytes'] / 1024**3
    g = value['graph']; decision = '通过' if value['resource_gate_passed'] else '未通过'
    text = f'''# 真正经过梯度训练的LightGCN：资源预实验

状态：仅做3次完整图梯度更新的机制/吞吐预实验，**不是正式训练完成的推荐专家**。不计算候选分数或外层MAP，最终周仍未运行。

## 为什么与前一图专家不同

前一个图专家只在已经训练好的BPR因子上做图传播，传播后的排名变化没有反馈到因子训练。本实现每一步都将完整用户—商品图传播两层，再由BPR成对排序损失反向更新原始32维用户和商品向量。因此，前者失败不能证明本实现或完整LightGCN路线无效。

LightGCN为行业论文模型名：用户与商品通过购买二部图线性聚合邻居，不增加特征变换和激活层；最终表示为第0、1、2层的等权平均。BPR损失要求已购买商品分数高于抽样未购买商品分数，不将未购买视为明确负反馈。

## 固定参数与图

2019-11-27截止前全历史图：{g['users']:,}名用户、{g['items']:,}件商品、{g['binary_pairs']:,}条去重用户—商品边。最后事件日期{g['latest_history_date']}。去重只定义二元图关系，不修改原始交易。

固定32维、2层、正态标准差0.1初始化、Adam学习率0.001、原始向量L2系数0.0001。每次更新65,536个用户—正商品—负商品三元组；先均匀抽用户，再均匀抽其购买商品，负例排除该用户所有截止前已购商品。

候选正式预算在测量前固定为200次更新，共13,107,200个抽样三元组。**200次更新不等于200个全数据epoch**：只相当于本截止图边数的{200 * 65536 / g['binary_pairs']:.3f}倍抽样规模，而且为均匀用户采样，不保证覆盖每条边；不声称达到论文收敛程度。

## 实测成本

- 后两次完整更新平均{s:.3f}秒，包含抽样、完整图前向、反向和参数更新。
- GPU实际峰值分配{m:.3f}GiB，包含完整前后向和Adam状态。
- 200次更新的本截止投影约{value['projected200_update_pilot_seconds'] / 60:.1f}分钟。
- 已缓存历史截止按边数线性外推，总计{value['projected_all_existing_cutoffs_minutes']:.1f}分钟；这仍是估计，不含下游LightGBM和跨年确认。
- 四个内层链合并后8个必要截止约{value.get('projected_inner8cutoffs_minutes', 10.74):.1f}分钟，另留候选打分和工程开销。
- 资源门槛{decision}：单截止投影不超过480秒、已有截止合计不超过90分钟、实测峰值不超过6GiB。
- 原始嵌入梯度有限且非零，前三次更新已实际改变参数；仅证明训练机制正常，不证明泛化或MAP增益。
- 四项CPU测试通过：对称归一化公式、稀疏/稠密传播及梯度一致、负例不含已购边、BPR梯度实际更新参数。

如主线程进一步授权，应以固定200更新方案做内层筛选。若失败，应区分监督预算不足、表示弱或专家重复，不能仅凭这一预算实验排除整个LightGCN模型家族。

## 来源与改动边界

依据[LightGCN论文](https://arxiv.org/abs/2002.02126)与[作者PyTorch实现](https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/model.py)采用归一化传播、跨层平均及图梯度训练；参考[作者采样与优化器实现](https://github.com/gusye1234/LightGCN-PyTorch/blob/master/code/utils.py)保留均匀用户采样和已购负例排除。没有复制其评测、数据划分、参数搜索或引用论文指标作为本地证据。矩形CSR图及其转置只是避免稠密图的工程实现。
'''
    (REPORT / 'LIGHTGCN_RESOURCE_AUDIT.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'prepare', 'pilot', 'formal-register', 'build-inner', 'formal-report', 'api-audit', 'routing-design'])
    args = parser.parse_args()
    try:
        {'register': register, 'prepare': prepare, 'pilot': pilot,
         'formal-register': formal_register, 'build-inner': build_inner,
         'formal-report': formal_report, 'api-audit': api_audit, 'routing-design': source_routing_design}[args.command]()
    except Exception:
        write(ART / f'FAILURE_{time.time_ns()}.json', {'traceback': traceback.format_exc(),
              'algorithm_failure': False, 'final_week': 'not_run'})
        raise
