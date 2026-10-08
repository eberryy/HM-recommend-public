"""Fixed-pool full-negative/listwise factorial; all old Warm evidence preserved."""
from __future__ import annotations

import argparse
import time
import traceback

import numpy as np

from .warm_v2_contract import ARTIFACT, REPORT, assert_branch, evidence_id, now, read, record, register, write
from .warm_v2_engine import Engine, connection, literal
from .warm_v2_lab import log, spec
from .warm_v2_rethink import contract


def full_relation(source,sample):
    return f"SELECT f.* FROM read_parquet({literal(source)}) f SEMI JOIN (SELECT DISTINCT customer_id FROM read_parquet({literal(sample)})) s USING(customer_id)"


class SupervisionEngine(Engine):
    def __init__(self,c,trial):
        super().__init__(c)
        assert trial['sampling'] in ('distribution_30x','full_positive_groups')
        assert trial['objective'] in ('lambdarank','rank_xendcg')
        self.trial=trial
        self.parameter_overrides={'objective':trial['objective']}
        if trial['objective']=='rank_xendcg':
            self.parameter_overrides['objective_seed']=5

    def cached_data(self,cutoff,role):
        if role!='sample' or self.trial['sampling']=='distribution_30x':
            return super().cached_data(cutoff,role)
        sample,reference=super().cached_data(cutoff,'sample')
        path=ARTIFACT/'full-negative-v1'/cutoff/'sample.parquet';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():
            return path,read(meta)
        path.parent.mkdir(parents=True,exist_ok=True)
        relation=full_relation(self.base_path(cutoff),sample)
        columns=list(dict.fromkeys(['customer_id','article_id',*self.features,'target','user_history_events_12w']))
        with connection() as con:
            con.execute(f'CREATE VIEW full_groups AS {relation}')
            row=con.execute('SELECT count(*),count(DISTINCT customer_id),sum(target),count(DISTINCT(customer_id,article_id)) FROM full_groups').fetchone()
            minimum,maximum=con.execute('SELECT min(n),max(n) FROM(SELECT count(*) n FROM full_groups GROUP BY customer_id)').fetchone()
            assert row[1]==reference['groups'] and row[2]==reference['positive_rows'] and row[0]==row[3]
            assert row[0]/reference['rows']<=8 and maximum<=300
            con.execute(f"COPY (SELECT {','.join(columns)} FROM full_groups ORDER BY customer_id,candidate_rank,article_id) TO {literal(path)} (FORMAT PARQUET,COMPRESSION ZSTD)")
        result={**reference,'role':'full_same_positive_groups','rows':row[0],'groups':row[1],'positive_rows':int(row[2]),
            'min_group_rows':minimum,'max_group_rows':maximum,'sampled_reference_rows':reference['rows'],
            'sampling_change':'all original candidates for exactly the same positive-containing training users; no positive injection and no zero-positive users added',
            'artifact':evidence_id(path,reason='explicit_registry_evidence')}
        write(meta,result)
        return path,result


def screen(trial):
    assert_branch()
    trial_id=trial['id'];output=REPORT/f'{trial_id}_SCREEN.json'
    if output.exists():return read(output)
    cfg=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json')
    engine=SupervisionEngine(cfg,trial)
    entry=spec(engine,trial_id,[],[],'historical_screening',trial['hypothesis'])
    entry.update(research_stage='WV2.4',supervision_spec=trial,sampling_protocol=trial['sampling'],
        training_protocol={**entry['training_protocol'],'sampling':trial['sampling'],'objective':trial['objective'],
            'note':'All other historical parameters and training/validation cutoffs unchanged; XE_NDCG does not use LambdaRank pair truncation.'})
    register(entry)
    baseline=read(REPORT/'WARM_V1_REPRODUCTION.json');results={};start=time.perf_counter()
    for w in cfg['rolling_protocol']:
        print(f'{trial_id} {w} historical fit: {trial}',flush=True)
        path=ARTIFACT/trial_id/w/'inner_metrics.json'
        r=read(path) if path.exists() else engine.train_inner(w,trial_id,[],[])
        assert r['runtime_seconds']<900 and (r.get('process_peak_working_set_bytes') or 0)<12*1024**3
        r['population_delta_vs_v1']=(r['inner_covered_active_map']-baseline['windows'][w]['inner']['inner_covered_active_map'])*r['inner_population_weight']
        results[w]=r
        print(f"{trial_id} {w}: {r['population_delta_vs_v1']:+.8f}, rounds={r['best_iteration']}",flush=True)
    deltas=[r['population_delta_vs_v1'] for r in results.values()]
    passed=np.mean(deltas)>=.0001 and sum(d>0 for d in deltas)>=3 and min(deltas)>=-.0005
    summary={'mean_population_delta':float(np.mean(deltas)),'positive_windows':sum(d>0 for d in deltas),'worst_population_delta':min(deltas),'passed':bool(passed)}
    result={'experiment_id':trial_id,'supervision_spec':trial,'families':[],'windows':results,'screening':summary,
        'runtime_seconds':time.perf_counter()-start,'outer_windows':'not_run','final_week':'not_run'}
    write(output,result)
    record(trial_id,{'status':'screened','screening':summary,'decision':'retain_for_ablation' if passed else 'reject','result_path':str(output),'runtime_seconds':result['runtime_seconds']})
    log()
    return result


def confirm_best():
    c=contract()
    screened=[read(REPORT/f"{t['id']}_SCREEN.json") for t in c['trials']]
    passing=[r for r in screened if r['screening']['passed']]
    if not passing:
        write(REPORT/'WV2_300_SELECTION.json',{'decision':'no_qualified_arm','final_week':'not_run'})
        return
    best=sorted(passing,key=lambda r:(-r['screening']['mean_population_delta'],r['experiment_id']))[0]
    trial_id=best['experiment_id'];output=REPORT/f'{trial_id}_OUTER.json'
    if output.exists():return read(output)
    registry=read(REPORT/'WARM_EXPERIMENT_REGISTRY.json')
    entry=next(t for t in registry['trials'] if t['experiment_id']==trial_id)
    if not entry['outer_confirmations']:
        assert registry['outer_confirmation_counts'].get('supervision_factorial',0)==0
        entry.update(outer_confirmations=1,status='outer_preregistered',outer_registered_at=now())
        registry['outer_confirmation_counts']['supervision_factorial']=1
        write(REPORT/'WARM_EXPERIMENT_REGISTRY.json',registry)
        write(REPORT/'WV2_300_SELECTION.json',{'selected':trial_id,'selected_at':now(),'reason':'highest passing inner mean; no outer selection','final_week':'not_run'})
    cfg=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json');engine=SupervisionEngine(cfg,best['supervision_spec'])
    baseline=read(REPORT/'WARM_V1_REPRODUCTION.json');results={};start=time.perf_counter()
    for w in cfg['rolling_protocol']:
        print(f'{trial_id} {w} formal fixed-pool confirmation',flush=True)
        path=ARTIFACT/trial_id/w/'outer_metrics.json'
        results[w]=read(path) if path.exists() else engine.train_outer(w,trial_id,[],[],best['windows'][w])
    values={w:r['evaluation']['map@12'] for w,r in results.items()}
    deltas={w:v-baseline['per_window_MAP'][w] for w,v in values.items()}
    mean=float(np.mean(list(values.values())));delta=mean-baseline['mean_MAP']
    champion=next(t for t in registry['trials'] if t['experiment_id']==registry['current_champion'])
    stable=sum(d>=0 for d in deltas.values())>=3 and min(deltas.values())>=-.0003
    decision='milestone' if (stable and delta>=.0005) or delta>=.001 else 'new_champion' if stable and mean>champion['mean_MAP'] else 'reject'
    result={'experiment_id':trial_id,'supervision_spec':best['supervision_spec'],'families':[],'windows':results,
        'per_window_MAP':values,'per_window_delta':deltas,'mean_MAP':mean,'delta_vs_Warm_v1':delta,
        'delta_vs_current_champion':mean-champion['mean_MAP'],'non_degrade_windows':sum(d>=0 for d in deltas.values()),
        'worst_window_delta':min(deltas.values()),'stable':stable,'decision':decision,'runtime_seconds':time.perf_counter()-start,
        'candidate_pool_changed':False,'training_protocol_changed':True,'final_week':'not_run'}
    write(output,result)
    record(trial_id,{'status':'completed','result_path':str(output),**{k:v for k,v in result.items() if k not in ('windows','families')}})
    if stable and mean>champion['mean_MAP']:
        registry=read(REPORT/'WARM_EXPERIMENT_REGISTRY.json');registry['current_champion']=trial_id
        write(REPORT/'WARM_EXPERIMENT_REGISTRY.json',registry)
    log()
    print({k:result[k] for k in ('mean_MAP','per_window_delta','decision')},flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['screen-all','confirm-best'])
    a=p.parse_args()
    try:
        if a.command=='screen-all':
            for t in contract()['trials']:screen(t)
        else:confirm_best()
    except Exception:
        write(REPORT/f'FAILURE_RESEARCH_{time.time_ns()}.json',{'timestamp':now(),'traceback':traceback.format_exc(),'final_week':'not_run'})
        raise


if __name__=='__main__':main()
