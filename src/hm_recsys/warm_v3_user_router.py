"""WV3-371 recent-inner cross-fitted user utility router for WV2-601 versus WV3-332."""
from __future__ import annotations
import hashlib
import time
import numpy as np
import pandas as pd
from . import warm_v3_common as common
from .warm_v2_contract import read,write
from .warm_v2_engine import connection,literal
from .warm_v3_expert import screening_gate

TRIAL='WV3-371';ALPHA=10.
FEATURES=['log_history_events','log_candidate_count','normalized_rank_distance','top12_overlap',
    'joint_top12_from_baseline_top20','joint_top12_from_baseline_top50','joint_score_std',
    'joint_score_rank12_minus13','rank_correlation']


def geometry(path):
    with connection() as con:
        frame=con.execute(f'''SELECT customer_id,ln(1+max(user_history_events_12w)) log_history_events,
            ln(1+count(*)) log_candidate_count,avg(abs(rj-baseline_rank))*1.0/greatest(count(*)-1,1) normalized_rank_distance,
            count(*) FILTER(WHERE rj<=12 AND baseline_rank<=12)/12.0 top12_overlap,
            count(*) FILTER(WHERE rj<=12 AND baseline_rank<=20)/12.0 joint_top12_from_baseline_top20,
            count(*) FILTER(WHERE rj<=12 AND baseline_rank<=50)/12.0 joint_top12_from_baseline_top50,
            stddev_pop(score_joint) joint_score_std,
            max(score_joint) FILTER(WHERE rj=12)-max(score_joint) FILTER(WHERE rj=13) joint_score_rank12_minus13,
            corr(rj,baseline_rank) rank_correlation
            FROM read_parquet({literal(path)}) GROUP BY customer_id ORDER BY customer_id''').fetchdf()
    frame[FEATURES]=frame[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0.)
    return frame


def fit_ridge(frame,target):
    x=frame[FEATURES].to_numpy(np.float64);y=np.asarray(target,np.float64)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1.;z=(x-mean)/scale
    design=np.column_stack([np.ones(len(z)),z]);penalty=np.eye(design.shape[1])*ALPHA;penalty[0,0]=0
    coefficients=np.linalg.solve(design.T@design+penalty,design.T@y)
    return {'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':coefficients.tolist(),'alpha':ALPHA,'features':FEATURES}


def predict(frame,model):
    assert model['features']==FEATURES and model['alpha']==ALPHA
    x=frame[FEATURES].to_numpy(np.float64);z=(x-np.asarray(model['mean']))/np.asarray(model['scale'])
    return np.column_stack([np.ones(len(z)),z])@np.asarray(model['coefficients'])


def fold(customer):return int(hashlib.sha256(str(customer).encode()).hexdigest()[-2:],16)%2


def sources(window,stage):
    root=common.ART/'WV3-331'/window
    if stage=='inner':return root/'inner_joint_ranks.parquet',root/'inner_user_ap.parquet'
    return root/'outer_multiview_ranks.parquet',root/'outer_user_ap.parquet'


def screen_window(window,total_users):
    root=common.ART/TRIAL/window/'inner';output=root/'REVIEW.json'
    if output.is_file():return read(output)
    root.mkdir(parents=True,exist_ok=True);started=time.perf_counter();ranks,ap_path=sources(window,'inner')
    frame=geometry(ranks)
    with connection() as con:aps=con.execute(f'SELECT * FROM read_parquet({literal(ap_path)}) ORDER BY customer_id').fetchdf()
    frame=frame.merge(aps,on='customer_id',validate='one_to_one');target=frame.ap_fusion-frame.ap_baseline
    pred=np.zeros(len(frame));models=[];folds=np.array([fold(v) for v in frame.customer_id])
    for held in (0,1):
        train=folds!=held;model=fit_ridge(frame.loc[train],target[train]);pred[~train]=predict(frame.loc[~train],model);models.append(model)
    selected=pred>0;delta=np.where(selected,target,0.);baseline=float(frame.ap_baseline.sum()/total_users)
    full_model=fit_ridge(frame,target);write(root/'MODEL.json',{'window':window,'fit_population':'all inner covered-active users',
        'model':full_model,'threshold':0.,'target':'ap_fusion-ap_baseline','outer_labels_used':False,'final_week':'not_run'})
    result={'window':window,'role':'two_fold_user_OOF_inner','included_users':len(frame),'total_users':total_users,
        'baseline_MAP':baseline,'MAP':float((frame.ap_baseline+delta).sum()/total_users),
        'population_delta':float(delta.sum()/total_users),'selected_users':int(selected.sum()),
        'selected_share':float(selected.mean()),'selected_positive_utility_users':int(((target>0)&selected).sum()),
        'selected_negative_utility_users':int(((target<0)&selected).sum()),'mean_predicted_utility':float(pred.mean()),
        'fold_models':models,'full_inner_model':str(root/'MODEL.json'),'candidate_pool_changed':False,
        'runtime_seconds':time.perf_counter()-started,'final_week':'not_run'}
    write(output,result);print({'router_inner':window,'delta':result['population_delta'],'selected':result['selected_users']},flush=True);return result


def screen():
    common.setup();common.budget(5);entry=next(t for t in read(common.REGISTRY)['trials'] if t['experiment_id']==TRIAL)
    assert entry['decision']=='preregistered' and entry['outer_exposures']==0
    engine=common.Engine(2020);started=time.perf_counter();rows={}
    for window,p in engine.contract['rolling_protocol'].items():
        meta=read(common.ART/'gate_data'/f"2020_inner_{p['inner_validation']}"/'DATA.json')
        rows[window]=screen_window(window,meta['total_users'])
        assert abs(rows[window]['baseline_MAP']-meta['baseline_map_population_component'])<1e-12
    gate=screening_gate(v['population_delta'] for v in rows.values())
    result={'experiment_id':TRIAL,'architecture':'recent_inner_crossfit_user_utility_router','windows':rows,
        'screening':gate,'features':FEATURES,'ridge_alpha':ALPHA,'threshold':0.,'new_representation_training':False,
        'candidate_pool_changed':False,'runtime_seconds':time.perf_counter()-started,'no_registry_mutation':True,'final_week':'not_run'}
    write(common.REPORT/(TRIAL+'_SCREEN.json'),result);print(gate,flush=True);return result


def confirm():
    common.setup();common.budget(10);entry=next(t for t in read(common.REGISTRY)['trials'] if t['experiment_id']==TRIAL)
    assert entry['outer_exposures']==1 and read(common.REPORT/(TRIAL+'_SCREEN.json'))['screening']['passed']
    destination=common.REPORT/(TRIAL+'_OUTER.json')
    if destination.is_file():return read(destination)
    base=read(common.ROOT/'reports/warm_v2/WV2-601_OUTER.json')['per_window_MAP'];rows={};started=time.perf_counter()
    for window in common.contracts()['rolling_protocol']:
        ranks,ap_path=sources(window,'outer');frame=geometry(ranks)
        with connection() as con:aps=con.execute(f'SELECT * FROM read_parquet({literal(ap_path)}) ORDER BY customer_id').fetchdf()
        frame=frame.merge(aps,on='customer_id',validate='one_to_one');model=read(common.ART/TRIAL/window/'inner'/'MODEL.json')['model']
        pred=predict(frame,model);selected=pred>0;target=frame.ap_fusion-frame.ap_baseline;delta=np.where(selected,target,0.)
        assert abs(float(frame.ap_baseline.mean())-base[window])<1e-12
        rows[window]={'window':window,'role':'heldout_outer','total_users':len(frame),'baseline_MAP':base[window],
            'MAP':float((frame.ap_baseline+delta).mean()),'population_delta':float(delta.mean()),
            'selected_users':int(selected.sum()),'selected_share':float(selected.mean()),
            'selected_positive_utility_users':int(((target>0)&selected).sum()),
            'selected_negative_utility_users':int(((target<0)&selected).sum()),
            'model_source':str(common.ART/TRIAL/window/'inner'/'MODEL.json'),
            'outer_labels_used_for_routing':False,'candidate_pool_changed':False,'final_week':'not_run'}
        (common.ART/TRIAL/window/'outer').mkdir(parents=True,exist_ok=True)
        write(common.ART/TRIAL/window/'outer'/'REVIEW.json',rows[window])
        print({'router_outer':window,'delta':rows[window]['population_delta'],'selected':rows[window]['selected_users']},flush=True)
    result={'experiment_id':TRIAL,'architecture':'recent_inner_crossfit_user_utility_router','windows':rows,
        **common.summary({w:r['MAP'] for w,r in rows.items()},2020),'features':FEATURES,'ridge_alpha':ALPHA,
        'threshold':0.,'candidate_pool_changed':False,'runtime_seconds':time.perf_counter()-started,
        'no_registry_mutation':True,'final_week':'not_run'}
    write(destination,result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['screen','confirm'],nargs='?',default='screen')
    globals()[parser.parse_args().command]()
