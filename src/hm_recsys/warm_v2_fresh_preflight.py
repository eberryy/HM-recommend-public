"""Read-only input checks and preregistration for frozen fresh temporal tests."""
from pathlib import Path
from datetime import date,timedelta
import json
import shutil
import subprocess
import duckdb
from .warm_v2_contract import now,read,write,evidence_id
from .warm_v2_bpr import PARAMS
from .m29 import ITEM2VEC_CONFIG

ROOT=Path(__file__).resolve().parents[2]
SHARED=Path(__file__).resolve().parents[2]
CHAINS=[('2018-12-26','2019-01-23','2019-02-20'),('2019-03-27','2019-04-24','2019-05-22'),
    ('2019-06-26','2019-07-24','2019-08-21'),('2019-09-25','2019-10-23','2019-11-20')]


def main():
    assert Path.cwd().resolve()==ROOT
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='warm-v2-fresh-robustness'
    contract_path=ROOT/'reports/warm_v2/WARM_V2_FRESH_ROBUSTNESS_CONTRACT.json'
    if contract_path.exists():
        print(json.dumps(read(contract_path)['preflight'],ensure_ascii=False,indent=2));return read(contract_path)
    with duckdb.connect() as con:
        con.execute('SET threads=8')
        source=SHARED/'data/interim/audit/transactions.parquet'
        first=con.execute(f"SELECT min(t_dat),max(t_dat),count(*) FROM read_parquet('{source.as_posix()}') WHERE t_dat < DATE '2018-12-26'").fetchone()
        days=con.execute(f"SELECT count(DISTINCT t_dat) FROM read_parquet('{source.as_posix()}') WHERE t_dat>=DATE '2018-10-03' AND t_dat<DATE '2018-12-26'").fetchone()[0]
        support=con.execute(f"SELECT count(DISTINCT customer_id),count(DISTINCT article_id) FROM read_parquet('{source.as_posix()}') WHERE t_dat<DATE '2019-11-20'").fetchone()
    boundary=date.fromisoformat(CHAINS[0][0])-timedelta(weeks=12)
    registry=json.loads((ROOT/'reports/warm_v2/WARM_EXPERIMENT_REGISTRY.json').read_text(encoding='utf-8'))
    exposure={t['experiment_id']:list(t.get('per_window_MAP',{})) for t in registry['trials'] if t.get('mean_MAP') is not None}
    forbidden={d for _,_,d in CHAINS}
    compact={d.replace('-','') for d in forbidden}
    seen=[(trial,w) for trial,windows in exposure.items() for w in windows if any(d in w for d in forbidden|compact)]
    result={'raw_start_before_first_training':str(first[0]),'last_observed_before_first_training':str(first[1]),
        'rows_before_first_training':first[2],'required_history_start':boundary.isoformat(),'history_pass':boundary>=first[0],
        'prior_outer_metric_windows':exposure,'fresh_previously_used_as_outer':seen,'freshness_pass':not seen,
        'distinct_days_in_required_history':days,
        'free_disk_gib':shutil.disk_usage(ROOT).free/1024**3,'final_week':'not_run'}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    assert result['history_pass'] and result['freshness_pass'] and days==84
    old=read(ROOT/'reports/warm_v2/WARM_V2_EXPERIMENT_CONTRACT.json')
    m=read(ROOT/'reports/m3_3/m3-3-v1-cross-season-adaptive-seasonal/metrics.json')
    parameter_sets={role:[d[role]['anchor']['parameters'] for d in m['development'].values()] for role in ('inner_models','outer_models')}
    for values in parameter_sets.values():assert all(v==values[0] for v in values)
    # Migration/shared-asset reuse: compare against trusted committed M3.3 identities once.
    expected=[m['inputs']['transactions'],m['prerequisite_cache']['target_features']['2020-01-22']['inputs']['articles']]
    identities=[]
    for item in expected:
        actual=evidence_id(item['path'],reason='compare_frozen_m33')
        assert actual['sha256']==item['sha256'] and actual['bytes']==item['bytes']
        identities.append({**actual,'trusted_reference':'committed M3.3 metrics inputs','match':True})
    baseline_manifest=read(m['inputs']['baseline_windows']['2019-11-27']['manifest']['path'])
    import implicit,lightgbm,torch
    assert implicit.__version__=='0.7.3'
    result['cuda_available']=torch.cuda.is_available()
    assert result['cuda_available'], 'Original Item2Vec CUDA neighbor protocol unavailable; do not substitute algorithm'
    c={'registered_at':now(),'stage':'Fresh Robustness Confirmation; no search','branch':'warm-v2-fresh-robustness',
        'warm_start_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'main_HEAD_at_registration':subprocess.check_output(['git','rev-parse','main'],text=True).strip(),
        'workspace':str(ROOT),'shared_assets_read_only':str(SHARED),'artifact_root':'artifacts/warm_v2/fresh_robustness',
        'rolling_protocol':{f'fresh_{o.replace("-","")}':{'inner_train':[t],'inner_validation':v,'outer_train':[t,v],'outer_validation':o} for t,v,o in CHAINS},
        'systems':['FRESH-000','FRESH-501','FRESH-601'],'frozen_feature_columns':old['frozen_feature_columns'],
        'extra_features':['wv2_bpr_user_item_score','wv2_bpr_unavailable'],
        'lightgbm_inner_params':parameter_sets['inner_models'][0],'lightgbm_outer_params':parameter_sets['outer_models'][0],
        'config':old['config'],'sampling':old['training']['sampling'],'sampling_seed':old['training']['sampling_seed'],
        'candidate_source_config':baseline_manifest['config'],'item2vec_config':ITEM2VEC_CONFIG,'bpr_params':PARAMS,
        'fusion':{'weights':[1,1],'constant':60,'formula':'1/(60+base_rank)+1/(60+bpr_rank)','fits':0},
        'fallback':'history_events_12w==0 => exact original candidate_rank; otherwise score desc, candidate_rank,article_id',
        'truth':'labels DISTINCT purchase pairs in [cutoff,cutoff+7d); features/training relations strictly BEFORE cutoff; complete truth denominator min(ntruth,12); original deterministic10% truth-user population',
        'labels_clarification':'Future7day labels are necessary for supervised targets/evaluation, never retrieval or feature construction. Attachment phrase labels before cutoff cannot literally define future-purchase supervision.',
        'gates':{'robust_pass':'mean delta>=0.000500 AND nondegrade>=3 AND worst>=-0.000300',
            'strong_pass':'robust_pass AND all4 deltas>0','weak_generalization':'mean delta>0 but robust gate fails','fresh_robustness_failed':'mean delta<=0'},
        'inner_complementarity':'diagnostic only; does not gate execution or change frozen fusion',
        'preflight':result,'trusted_input_identity':identities,
        'resources':{'estimate_hours':[2,4],'cpu_threads':8,'gpu':'original Item2Vec exact-neighbor stage only; BPR CPU',
            'candidate_features_cache_estimate_gib':[4,12],'bpr_factor_bytes_upper_bound':12*(support[0]+support[1])*101*4,
            'disk_floor_gib':10,'per_cutoff_candidate_review_seconds':1200,'per_cutoff_bpr_review_seconds':900,
            'cost_stop':'preserve completed checkpoints and diagnose engineering; never alter model params or gate'},
        'versions':{'implicit':implicit.__version__,'lightgbm':lightgbm.__version__,'torch':torch.__version__},
        'read_evidence_name_resolution':'WV2-601_CONTRACT.json in prompt resolves to existing WV2_601_CONTRACT.json; no semantic substitution',
        'no_training_before_this_contract':True,'no_search':True,'final_week':'not_run','merge':False,'integration':False,
        'completion':'pass or fail => complete report and STOP for human review; no automatic rescue experiments'}
    write(contract_path,c)
    print('Registered fresh contract before model construction',flush=True)
    return c


if __name__=='__main__':main()
