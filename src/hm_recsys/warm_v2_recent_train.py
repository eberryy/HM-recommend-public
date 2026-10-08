"""Registered recent-target comparison with frozen validation pools and ranking."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date,timedelta
import time
import numpy as np

from .warm_v2_contract import ARTIFACT,REPORT,assert_branch,guard,now,read,write,register,record
from .warm_v2_engine import Engine
from .warm_v2_lab import log,spec
from .warm_v2_recent_data import ROOT,build

TRIAL='WV2-401'


def recent_protocol(original):
    result=deepcopy(original)
    for p in result.values():
        for key in ('inner_train','outer_train'):
            p[key]=[(date.fromisoformat(d)+timedelta(days=21)).isoformat() for d in p[key]]
        for key,vkey in (('inner_train','inner_validation'),('outer_train','outer_validation')):
            for d in p[key]:
                guard(d)
                assert date.fromisoformat(d)+timedelta(days=7)<=date.fromisoformat(p[vkey])
    return result


def contract():
    path=REPORT/'WV2_401_CONTRACT.json'
    if path.exists():return read(path)
    audit=read(REPORT/'WV2_400_FRESHNESS_AUDIT.json');assert audit['audit_gate_passed']
    c=deepcopy(read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json'))
    c.update(stage='WV2.5 recent target labels',created_at_utc=now(),experiment_id=TRIAL)
    c['rolling_protocol']=recent_protocol(c['rolling_protocol'])
    c['training']['rounds']='same one inner/two outer target weeks shifted21days toward validation; independent inner early stopping'
    c['candidate_contract']='Validation: exact frozen M3.3 pairs. New training dates: same six-sourceTop100+Item2Vec<=200, same hash10% users/full-history globals; no candidate budget or feature changes.'
    c['stage_budget']={'inner_trials':1,'outer_confirmations':1,
        'compute':'one cutoff build pilot5-15min; use measured cost before other builds; rank fits5-15min total; pause batch and diagnose if build>900sec or free disk<5GiB',
        'failure_path':'No more variants of label recency driven by outer. If rejected, audit learned user-item collaborative matching.'}
    c['controlled']='Only target-week freshness changes: weekly count, sampling rule, feature formulas, ranker params and evaluation all fixed. Different calendar weeks naturally have different users/events; not a perfectly equal-row causal contrast.'
    c['final_week']='not_run';write(path,c)
    return c


class RecentEngine(Engine):
    def cached_data(self,cutoff,role):
        if cutoff not in self.history['feature_cache']:
            result=build(cutoff)
            self.history['feature_cache'][cutoff]=result['target']
            # Check each new build before launching more work, not an algorithm gate.
            review_path=REPORT/'WV2_401_RESOURCE_REVIEW.json'
            allowed=read(review_path).get('completed_cache_reuse_only',[]) if review_path.exists() else []
            if (result['wall_seconds']>900 and cutoff not in allowed) or result['disk_free_gib']<5:
                raise RuntimeError('new-cutoff resource gate: retain completed checkpoint and diagnose before batch continuation')
        return super().cached_data(cutoff,role)


def screen():
    assert_branch();output=REPORT/f'{TRIAL}_SCREEN.json'
    if output.exists():return read(output)
    c=contract();e=RecentEngine(c)
    entry=spec(e,TRIAL,[],[],'historical_screening','训练周数量不变，将标签移近21天，检验需求关系过时是否限制泛化')
    entry.update(research_stage='WV2.5',rolling_protocol=c['rolling_protocol'])
    register(entry)
    baseline=read(REPORT/'WARM_V1_REPRODUCTION.json');results={};start=time.perf_counter()
    for w in c['rolling_protocol']:
        print(f'{TRIAL} {w}: recent-target historical fit',flush=True)
        path=ARTIFACT/TRIAL/w/'inner_metrics.json'
        r=read(path) if path.exists() else e.train_inner(w,TRIAL,[],[])
        r['population_delta_vs_v1']=(r['inner_covered_active_map']-baseline['windows'][w]['inner']['inner_covered_active_map'])*r['inner_population_weight']
        results[w]=r
        print(f'{w}: delta={r["population_delta_vs_v1"]:+.8f}',flush=True)
    ds=[r['population_delta_vs_v1'] for r in results.values()]
    passed=np.mean(ds)>=.0001 and sum(d>0 for d in ds)>=3 and min(ds)>=-.0005
    summary={'mean_population_delta':float(np.mean(ds)),'positive_windows':sum(d>0 for d in ds),'worst_population_delta':min(ds),'passed':bool(passed)}
    result={'experiment_id':TRIAL,'families':[],'windows':results,'screening':summary,
        'runtime_seconds':time.perf_counter()-start,'outer_windows':'not_run','final_week':'not_run'}
    write(output,result);record(TRIAL,{'status':'screened','screening':summary,'decision':'retain_for_ablation' if passed else 'reject','result_path':str(output),'runtime_seconds':result['runtime_seconds']});log()
    return result


def confirm():
    s=read(REPORT/f'{TRIAL}_SCREEN.json');assert s['screening']['passed']
    output=REPORT/f'{TRIAL}_OUTER.json'
    if output.exists():return read(output)
    c=contract();e=RecentEngine(c);registry=read(REPORT/'WARM_EXPERIMENT_REGISTRY.json')
    entry=next(t for t in registry['trials'] if t['experiment_id']==TRIAL)
    if not entry['outer_confirmations']:
        assert registry['outer_confirmation_counts'].get('recent_target_labels',0)==0
        entry.update(outer_confirmations=1,status='outer_preregistered',outer_registered_at=now())
        registry['outer_confirmation_counts']['recent_target_labels']=1;write(REPORT/'WARM_EXPERIMENT_REGISTRY.json',registry)
    baseline=read(REPORT/'WARM_V1_REPRODUCTION.json');results={};start=time.perf_counter()
    for w in c['rolling_protocol']:
        print(f'{TRIAL} {w}: formal confirmation',flush=True)
        path=ARTIFACT/TRIAL/w/'outer_metrics.json'
        results[w]=read(path) if path.exists() else e.train_outer(w,TRIAL,[],[],s['windows'][w])
    values={w:r['evaluation']['map@12'] for w,r in results.items()}
    ds={w:v-baseline['per_window_MAP'][w] for w,v in values.items()}
    mean=float(np.mean(list(values.values())));delta=mean-baseline['mean_MAP']
    champion=next(t for t in registry['trials'] if t['experiment_id']==registry['current_champion'])
    stable=sum(d>=0 for d in ds.values())>=3 and min(ds.values())>=-.0003
    decision='milestone' if (stable and delta>=.0005) or delta>=.001 else 'new_champion' if stable and mean>champion['mean_MAP'] else 'reject'
    result={'experiment_id':TRIAL,'families':[],'windows':results,'per_window_MAP':values,'per_window_delta':ds,
        'mean_MAP':mean,'delta_vs_Warm_v1':delta,'delta_vs_current_champion':mean-champion['mean_MAP'],
        'non_degrade_windows':sum(d>=0 for d in ds.values()),'worst_window_delta':min(ds.values()),
        'stable':stable,'decision':decision,'runtime_seconds':time.perf_counter()-start,
        'candidate_pool_changed':False,'training_protocol_changed':True,'final_week':'not_run'}
    write(output,result);record(TRIAL,{'status':'completed','result_path':str(output),**{k:v for k,v in result.items() if k not in ('windows','families')}})
    if stable and mean>champion['mean_MAP']:
        registry=read(REPORT/'WARM_EXPERIMENT_REGISTRY.json');registry['current_champion']=TRIAL;write(REPORT/'WARM_EXPERIMENT_REGISTRY.json',registry)
    log();print({k:result[k] for k in ('mean_MAP','per_window_delta','decision')},flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['register','screen','confirm']);a=p.parse_args()
    {'register':contract,'screen':screen,'confirm':confirm}[a.command]()
