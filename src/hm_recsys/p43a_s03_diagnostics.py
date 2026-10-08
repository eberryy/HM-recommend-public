"""Read existing C scores and D restart prefix; no new fits or policies."""
from pathlib import Path
import numpy as np
from lightgbm import Booster
from .p41a_contract import read_json,write_json
from .p43a_verify import WINDOWS


def run(repo):
    repo=Path(repo).resolve(); root=repo/'artifacts/phase4/p4-3a-v1-map-cashout-hash10-tournament'
    rows=[]
    for w in WINDOWS:
        fit=read_json(root/'models/C-count'/w/'FIT_RESULT.json')
        folder=root/'evaluation/C-oracle'/w
        p=np.load(folder/'count-score.npy',mmap_mode='r'); k=np.load(folder/'policy/Khat.npy')
        rows.append(dict(window=w,users=len(k),training_rows=fit['rows'],
            training_class_counts=fit['class_counts'],predicted_K_counts={str(v):int((k==v).sum()) for v in range(4)},
            nonzero_class_score_quantiles=dict(zip(['min','median','p90','p99','max'],
                map(float,np.quantile(p[:,1:].sum(axis=1),[0,.5,.9,.99,1])))),
            users_nonzero_class_beats_zero=int((p[:,1:].max(axis=1)>p[:,0]).sum())))
    archived=root/'session-history/s03-20260912T143544/partial-fit-D-L63-D-1-M200-spring/partial-model.txt'
    actual=root/'models/D-L63-D-1-M200/spring_20200318/model.txt'
    old=Booster(model_file=str(archived)).dump_model()['tree_info']
    new=Booster(model_file=str(actual)).dump_model()['tree_info']
    assert len(old)==406 and len(new)==500 and old==new[:406]
    result=dict(stage='P4.3A',status='completed_read_only',new_fits=0,new_policy_evaluations=0,
        C_count=dict(definition='One row per development user-window; nonzero score is sum of model classes1..3, not calibrated purchase propensity. Training class counts are historical action-user observations.',windows=rows),
        D_restart_prefix=dict(status='pass',model='D-L63-D-1-M200',window='spring_20200318',
            previous_trees=406,completed_trees=500,all_prefix_tree_values_exact=True),final_week='not_run')
    write_json(repo/'reports/phase4/P4_3A_SESSION_03_DIAGNOSTICS.json',result)
    return result


if __name__=='__main__':print(run('.'))
