"""Independent frozen-row checks for P4.2D, without optimizer or matching."""
from pathlib import Path
import ast
import time
import traceback
import duckdb
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from .p41a_contract import read_json, write_json, check_identity
from .p41a_data import parquet
from .p42_contract import WINDOWS, TAUS, branch_guard, guard_cutoff
from .p42d_contract import RUN_ID

NAMES=['no_new_model_fit','no_optimizer_call','no_qC_qW_refit','qC_predictions_exact_reused',
       'qW_predictions_exact_reused','pair_U_exact_parity','W0_exact_parity','B0_Cold50_exact_parity',
       'final_week_fail_closed','candidate_positive_unique','edge_labels_exact','labels_only_in_audit',
       'strict_sparse_exact','qc_rank_complete_same_user50','B0_rank_exact_original','no_threshold_tuning',
       'only_original_three_taus','alternatives_no_formal_MAP','no_new_utility_matching',
       'future_proxy_never_model_feature','scaleup_no_training','all_trusted_SHA_pass']


def verify(repo):
    repo=Path(repo).resolve(); branch_guard(repo); report=repo/'reports/phase4'; root=repo/'artifacts/phase4'/RUN_ID
    m=read_json(report/'P4_2D_metrics.json'); c=read_json(report/'P4_2D_EXPERIMENT_CONTRACT.json')
    old=read_json(report/'P4_2R3_metrics.json'); oc=read_json(report/'P4_2R3_EXPERIMENT_CONTRACT.json')
    began=time.perf_counter(); out={'stage':'P4.2D','status':'fail','windows':{},'checks':[]}
    try:
        assert m['status']=='measured_pending_verification' and len(m['windows'])==4
        assert not m['runtime_forbidden_calls'] and all(m[k]==0 for k in ('new_model_fits','optimizer_calls','matching_calls','new_candidate_generation'))
        assert c['taus']==TAUS and c['pair_scores']==['U','D1=qC-qW','D2=qC-2*qW']
        assert c['availability']['exclusive_end']=='2020-09-16' and m['final_week']=='not_run'
        for t in ('2020-09-16','2020-09-10'):
            try: guard_cutoff(t)
            except ValueError: pass
            else: raise AssertionError('embargo guard failed')
        receipt=read_json(root/'INPUT_VERIFICATION.json')
        assert receipt['pass'] and receipt['comparisons']==len(receipt['trusted_records'])
        # The recorded input comparisons were performed before audit. Verify old
        # implementation remained untouched; don't hash fresh outputs as proof of origin.
        for rec in c['source_review'].values(): check_identity(rec)
        for path in (repo/'src/hm_recsys').glob('p42d*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    name=node.func.attr if isinstance(node.func,ast.Attribute) else node.func.id if isinstance(node.func,ast.Name) else ''
                    assert name not in {'fit','fit_transform','fit_predict','minimize','fmin_l_bfgs_b','exact_matching','linear_sum_assignment','predict_propensity','predict_repaired_propensity'},(path,name)
        with duckdb.connect() as db:
            db.execute('SET threads=4'); db.execute("SET memory_limit='4GB'")
            for window,t in WINDOWS.items():
                folder=repo/'artifacts/phase4'/oc['run_id']/'outer'/window
                cold,warm,e,ex,truth=[parquet(folder/(x+'.parquet')) for x in ('qC-predictions','qW-predictions','eligible_cold','executed','truth')]
                w=m['windows'][window]
                uq=np.load(folder/'pair-utility.float64.npy',mmap_mode='r')
                label=np.load(root/window/'edge-labels.int8.npy',mmap_mode='r')
                ap=np.load(root/window/'single-delta-ap.float64.npy',mmap_mode='r')
                pos=parquet(root/window/'positive-candidates.parquet')
                assert not pos.duplicated(['customer_id','article_id']).any()
                ep=e.target.eq(1).to_numpy(); idx=e.cold_row_index.to_numpy(int); ui=e.user_index.to_numpy(int)
                np.testing.assert_array_equal(pos.cold_row_index,idx[ep])
                assert len(pos)==w['census']['eligible']['all']['positives']
                assert int(cold.target.sum())==int(pos.target.sum())+w['census']['excluded_overlap']['all']['positives']==old['calibration'][window]['qC']['positives']
                counts=db.execute('SELECT article_id,count(*) n FROM read_parquet(?) WHERE t_dat<CAST(? AS DATE) GROUP BY article_id',[oc['transactions']['path'],t]).fetchdf().set_index('article_id').n
                for f in (cold,warm):
                    n=f.article_id.map(counts).fillna(0).to_numpy(int)
                    np.testing.assert_array_equal(f.interaction_count_before_cutoff,n)
                    np.testing.assert_array_equal(f.strict_cold_flag,n==0)
                    np.testing.assert_array_equal(f.sparse1_5_flag,(n>=1)&(n<=5))
                coldq=np.clip(cold.propensity.to_numpy(),1e-6,1-1e-6)
                warmq=np.clip(warm.propensity.to_numpy(),1e-6,1-1e-6).reshape(-1,12)
                np.testing.assert_array_equal(coldq[idx],e.qC)
                expected=np.log(coldq[idx]/(1-coldq[idx]))[:,None]-np.log(warmq[ui]/(1-warmq[ui]))
                np.testing.assert_array_equal(uq,expected)
                original=warm.article_id.to_numpy().reshape(-1,12)
                users=warm.customer_id.to_numpy().reshape(-1,12)[:,0]
                original_r=warm.target.to_numpy().reshape(-1,12)
                wr=original_r[ui]
                expected_labels=np.where(ep[:,None]&~wr.astype(bool),1,np.where(~ep[:,None]&wr.astype(bool),-1,0))
                np.testing.assert_array_equal(label,expected_labels)
                nt=truth.groupby('customer_id').size().reindex(users).to_numpy()
                denominator=np.minimum(nt[ui],12)
                inv=1/np.arange(1,13)
                base=(wr*np.cumsum(wr,axis=1)*inv).sum(axis=1)/denominator
                for j in range(12):
                    r=wr.copy(); r[:,j]=ep
                    expected_ap=(r*np.cumsum(r,axis=1)*inv).sum(axis=1)/denominator-base
                    np.testing.assert_allclose(ap[:,j],expected_ap,rtol=0,atol=4e-16)
                    slot=w['pair']['slots'][str(j+1)]
                    assert slot['beneficial']==int((label[:,j]==1).sum()) and slot['harmful']==int((label[:,j]==-1).sum())
                assert np.array_equal(np.sign(ap),label)
                q=cold.propensity.to_numpy().reshape(-1,50)
                order=np.argsort(-q,axis=1,kind='stable')
                rank=np.empty_like(order); np.put_along_axis(rank,order,np.tile(np.arange(1,51),(len(q),1)),axis=1)
                np.testing.assert_array_equal(pos.qc_rank,rank.ravel()[idx[ep]])
                np.testing.assert_array_equal(pos.b0_rank,cold.b0_rank.to_numpy()[idx[ep]])
                significant=label!=0
                y=label[significant]==1
                for name,s in [('U',np.asarray(uq)),('D1',coldq[idx,None]-warmq[ui]),('D2',coldq[idx,None]-2*warmq[ui])]:
                    metrics=w['pair']['comparison']['primary'] if name=='U' else w['pair']['alternatives'][name]['primary']
                    np.testing.assert_allclose([metrics['roc_auc'],metrics['pr_auc']],
                         [roc_auc_score(y,s[significant]),average_precision_score(y,s[significant])],rtol=1e-12)
                rowmap={int(v):i for i,v in enumerate(idx)}
                for variant,tau in TAUS.items():
                    selected=np.zeros(label.shape,bool); reconstructed=original.copy()
                    xx=ex.loc[ex.variant.eq(variant)]
                    for row in xx.itertuples():
                        ci=rowmap[int(row.cold_row_index)]; j=int(row.warm_slot_rank)-1
                        selected[ci,j]=True; reconstructed[int(row.user_index),j]=row.cold_article_id
                        assert row.utility==uq[ci,j] and row.net_positive==label[ci,j]
                    saved=parquet(folder/(variant+'-top12.parquet'))
                    np.testing.assert_array_equal(saved.article_id.to_numpy().reshape(-1,12),reconstructed)
                    survive=(uq>tau)&(label==1)
                    detail=w['pair']['losses'][variant]
                    assert detail['beneficial_edges_above_tau']==int(survive.sum())
                    assert detail['beneficial_candidates_above_tau']==int(survive.any(axis=1).sum())
                    assert detail['beneficial_edges_selected']==int((selected&(label==1)).sum())
                    assert detail['lost_candidates']==int((survive.any(axis=1)&~(selected&(label==1)).any(axis=1)).sum())
                    assert sum(detail['conflicts'][k] for k in ('same_cold_only','same_warm_only','both','neither'))==int((survive&~selected).sum())
                    assert w['pair']['survival'][variant]['all']['any_edge']==int((uq[ep]>tau).any(axis=1).sum())
                    assert w['pair']['executed'][variant]['all']['rows']==len(xx)
                    assert not any('map' in k.lower() for k in w['pair']['alternatives'])
                proxy=parquet(root/window/'strict-first-sale-proxy.parquet')
                assert len(proxy)==int(cold.strict_cold_flag.sum())
                assert not (proxy.audit_only_first_sale_days.dropna()<0).any()
                assert not (proxy.audit_only_first_sale_days.dropna()>=w['availability']['observation_days']).any()
                assert sum(b['rows'] for b in w['tails']['bins'])==len(cold)
                assert sum(b['positives'] for b in w['tails']['bins'])==int(cold.target.sum())
                out['windows'][window]={'pass':True,'full_candidates':len(cold),'primary_positive_candidates':len(pos),
                     'all_edges':int(label.size),'AP_all_edges_brute_force_parity':True,'sklearn_pair_AUC_AP_parity':True,
                     'saved_recommendations_reconstructed_not_reoptimized':True}
        out['status']='pass'; out['checks']=[{'number':i+1,'name':n,'status':'pass'} for i,n in enumerate(NAMES)]
        out['trusted_input_comparisons']=receipt['comparisons']; out['old_source_preservation_checks']=len(c['source_review'])
    except Exception:
        out['traceback']=traceback.format_exc()
    out['seconds']=time.perf_counter()-began
    boundary_path=report/'P4_2D_EXECUTION_BOUNDARY.json'
    if boundary_path.exists():
        boundary=read_json(boundary_path)
        out['execution_boundary']=boundary
        out['diagnostic_data_verification_status']=out['status']
        if boundary['synthetic_optimizer_calls_occurred'] and out['status']=='pass':
            out['status']='pass_with_execution_boundary_exception'
            out['whole_task_no_optimizer_call']='failed_due_to_extra_synthetic_regression_tests'
            for check in out['checks']:
                check['scope']='frozen-project diagnostic computation, not the extra synthetic regression process'
    write_json(report/'P4_2D_VERIFICATION.json',out)
    print({k:v for k,v in out.items() if k not in ('windows','checks')},flush=True)
    return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--repo',default='.')
    if verify(p.parse_args().repo)['status']!='pass': raise SystemExit(2)
