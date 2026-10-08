"""Warm-v3 isolated paths, temporal contracts, and unchanged ranker machinery."""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime,timezone,timedelta
from functools import lru_cache
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from .warm_v2_contract import read,write,now,git,guard,evidence_id
from . import warm_v2_engine as we

ROOT=Path(__file__).resolve().parents[2]
SHARED=Path(__file__).resolve().parents[2]
ART=ROOT/'artifacts/warm_v3'
REPORT=ROOT/'reports/warm_v3'
PRIVATE=ROOT/'reports/private/warm_v3'
OLD=SHARED/'artifacts/warm_v2'
FRESH=ROOT/'artifacts/warm_v2/fresh_robustness'
REGISTRY=REPORT/'WARM_V3_EXPERIMENT_REGISTRY.json'
TX=SHARED/'data/interim/audit/transactions.parquet'

def budget(expected_minutes=0,closure=False):
    states=[]
    for path in PRIVATE.glob('AUTONOMOUS_RUN_STATE*.json'):
        candidate=read(path)
        if 'deadline_utc' in candidate:
            states.append(candidate)
    if not states:
        raise FileNotFoundError('No Warm-v3 autonomous run state')
    state=max(states,key=lambda value:datetime.fromisoformat(value['start_time_utc'].replace('Z','+00:00')))
    remaining=(datetime.fromisoformat(state['deadline_utc'])-datetime.now(timezone.utc)).total_seconds()
    if not closure and (remaining<=0 or remaining<expected_minutes*60+600):
        raise TimeoutError(f'Warm-v3 time budget: {remaining:.0f}s remain, expected {expected_minutes}min plus closure reserve')
    if shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('disk safety floor10GiB')
    return remaining

def setup():
    assert Path.cwd().resolve()==ROOT
    assert git('branch','--show-current')=='warm-v3-architecture-lab'
    assert git('merge-base','41fe308','HEAD')==git('rev-parse','41fe308')
    we.ARTIFACT=ART
    from . import warm_v2_features as wf
    wf.attach_features=attach_features
    return read(ROOT/'reports/warm_v2/WARM_V2_EXPERIMENT_CONTRACT.json')

def contracts(year=2020):
    return read(ROOT/('reports/warm_v2/WARM_V2_EXPERIMENT_CONTRACT.json' if year==2020 else 'reports/warm_v2/WARM_V2_FRESH_ROBUSTNESS_CONTRACT.json'))

def bpr_path(cutoff):
    guard(cutoff)
    root=(FRESH/'bpr'/cutoff) if (FRESH/'bpr'/cutoff/'features.json').exists() else OLD/'bpr-match-v1'/cutoff
    m=read(root/'features.json');assert m['cutoff']==cutoff and m['pit_safe'] and not m['future_labels_used']
    assert m['latest_history_date']<cutoff
    path=root/'features.parquet';assert path.is_file() and path.stat().st_size==m['artifact']['bytes']
    return path,m

def signal_path(family,cutoff,engine):
    if hasattr(engine,'signal_path'):
        result=engine.signal_path(family,cutoff)
        if result is not None:return result
    if family=='bpr_match':return bpr_path(cutoff)
    if family=='graph':
        from .warm_v3_graph import build_features
    elif family=='sequence':
        from .warm_v3_sequence import build_features
    elif family=='lightgcn':
        from .warm_v3_lightgcn import build_features
    elif family=='userknn':
        from .warm_v3_userknn import build_features
    else:raise ValueError(family)
    return build_features(cutoff,engine)

@lru_cache(maxsize=2)
def indexed(path):
    return we.load_parquet(path).set_index(['customer_id','article_id'],verify_integrity=True)

def attach_features(frame,cutoff,families,engine):
    added={}
    for family in families:
        path,meta=signal_path(family,cutoff,engine)
        f=indexed(str(path));keys=pd.MultiIndex.from_frame(frame[['customer_id','article_id']])
        assert keys.isin(f.index).all(),'new signal lost candidate identities'
        v=f.reindex(keys)
        for column in meta['features']:
            assert column not in added and column not in frame
            added[column]=v[column].to_numpy(np.float32)
    return pd.concat([frame,pd.DataFrame(added,index=frame.index)],axis=1)

class Engine(we.Engine):
    def __init__(self,year=2020):
        setup();super().__init__(contracts(year));self.transactions=str(TX)
        self.year=year
        if year==2019:
            template=next(iter(self.history['development'].values()))
            self.history['development']={w:deepcopy(template) for w in self.contract['rolling_protocol']}
        for cutoff_dir in (FRESH/'candidates').iterdir():
            if (cutoff_dir/'BUILD.json').exists():
                self.history['feature_cache'][cutoff_dir.name]=read(cutoff_dir/'BUILD.json')['target']

    def base_path(self,cutoff):
        guard(cutoff)
        path=Path(self.history['feature_cache'][cutoff]['artifact']['path'])
        assert path.is_file()
        return path

    def cached_data(self,cutoff,role):
        # Existing selected training samples/covered-active validation are exact
        # original assets. New signal joins never alter their labels or membership.
        for root in (FRESH,OLD):
            p=root/'frozen_data'/cutoff/f'{role}.parquet'
            if p.exists() and p.with_suffix('.json').exists():return p,read(p.with_suffix('.json'))
        return super().cached_data(cutoff,role)

def register(trial,family,hypothesis,**fields):
    budget();state=read(REGISTRY) if REGISTRY.exists() else {
        'baseline':'WV2-601','branch':'warm-v3-architecture-lab','start_head':'41fe30853c25d76794263b792d9b9bc811ad4c24',
        'current_champion':'WV2-601','outer_exposure_count':0,'trials':[],'final_week':'not_run'}
    assert not any(t['experiment_id']==trial for t in state['trials'])
    entry={'experiment_id':trial,'parent':'WV2-601','architecture_family':family,'hypothesis':hypothesis,
        'registered_at':now(),'decision':'preregistered','candidate_protocol':'unchanged six-sourceTop100+up to200Item2Vec-only',
        'training_protocol':'original30:1/LambdaRank/dates unless explicitly overridden','inner_evidence':None,
        'outer_MAP_by_window':None,'mean_MAP':None,'delta_vs_WV2_601':None,'nondegrade_windows':None,
        'worst_delta':None,'runtime':None,'artifact_paths':[],'outer_exposures':0,**fields}
    state['trials'].append(entry);write(REGISTRY,state);return entry

def update(trial,**fields):
    state=read(REGISTRY);entry=next(t for t in state['trials'] if t['experiment_id']==trial)
    entry.update(fields);write(REGISTRY,state)

def expose(trial):
    budget(5);state=read(REGISTRY);entry=next(t for t in state['trials'] if t['experiment_id']==trial)
    assert entry['outer_exposures']==0,'an outer-exposed concrete variant is closed'
    entry.update(outer_exposures=1,outer_exposed_at=now());state['outer_exposure_count']+=1;write(REGISTRY,state)

def log(stage,observation,hypothesis,audit,result,decision,next_action,alternatives=None,experiment=None,reflection=None):
    p=PRIVATE/'AUTONOMOUS_DECISION_LOG.md'
    text=p.read_text(encoding='utf-8')
    alternatives=alternatives or '保持已冻结基线作为回退；只推进有内层证据、资源有界的不同机制，不用外层结果微调同一方案。'
    experiment=experiment or f'本节点实验与冻结条件见对应注册表及审计：{audit}'
    reflection=reflection or decision
    text+=f'\n## {now()} — {stage}\n\n### Observation\n{observation}\n\n### Hypothesis\n{hypothesis}\n\n### Alternatives considered\n{alternatives}\n\n### Audit / Preflight\n{audit}\n\n### Experiment\n{experiment}\n\n### Result\n{result}\n\n### Reflection\n{reflection}\n\n### Decision\n{decision}\n\n### Next action\n{next_action}\n'
    p.write_text(text,encoding='utf-8')

def summary(maps,year=2020):
    if year==2020:base=read(ROOT/'reports/warm_v2/WV2-601_OUTER.json')['per_window_MAP']
    else:base={w:r['systems']['FRESH-601']['map@12'] for w,r in read(ROOT/'reports/warm_v2/WARM_V2_FRESH_ROBUSTNESS.json')['windows'].items()}
    assert set(maps)==set(base)
    delta={w:maps[w]-base[w] for w in maps};mean=float(np.mean(list(maps.values())))
    avg=float(np.mean(list(delta.values())));non=sum(d>=0 for d in delta.values());worst=min(delta.values())
    return {'per_window_MAP':maps,'per_window_delta':delta,'mean_MAP':mean,'delta_vs_WV2_601':avg,
        'nondegrade_windows':non,'worst_delta':worst,'stable':avg>0 and non>=3 and worst>=-.0005,
        'target_hit':year==2020 and mean>=.03 and non>=3 and worst>=-.0005,
        'robustness_pass':year==2019 and avg>=-.0003 and non>=2,'final_week':'not_run'}
