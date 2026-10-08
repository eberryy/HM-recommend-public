"""Warm-v2 registered baseline, historical screening and periodic confirmation CLI."""
from __future__ import annotations

import argparse
import time
import traceback

import numpy as np

from .warm_v2_contract import ARTIFACT, REPORT, assert_branch, now, preregister, read, record, register, write
from .warm_v2_engine import Engine


def log():
    registry = read(REPORT/"WARM_EXPERIMENT_REGISTRY.json")
    # Link already-recorded evidence IDs; do not rehash continuous local artifacts.
    def collect(value, found):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                found[(value['path'],value['sha256'])]=value
            for child in value.values():
                collect(child,found)
        elif isinstance(value,list):
            for child in value:
                collect(child,found)
    for trial in registry['trials']:
        found={}
        collect(trial.get('additional_artifact_evidence',[]),found)
        peaks=[]
        timings={}
        for result_path in [trial.get('result_path'),str(REPORT/f"{trial['experiment_id']}_SCREEN.json")]:
            if result_path:
                from pathlib import Path
                if Path(result_path).exists():
                    result=read(result_path)
                    collect(result,found)
                    if 'runtime_seconds' in result:
                        timing_role='historical_screening' if str(result_path).endswith('_SCREEN.json') else 'baseline_reproduction' if trial['experiment_id']=='WV2-000' else 'formal_confirmation'
                        timings[timing_role]=result['runtime_seconds']
                    for window in result.get('windows',{}).values():
                        for metrics in [window,window.get('inner',{}),window.get('outer',{})]:
                            if metrics.get('process_peak_working_set_bytes') is not None:
                                peaks.append(metrics['process_peak_working_set_bytes'])
        trial['artifact_evidence']=list(found.values())
        trial['artifact_paths']=sorted({v['path'] for v in found.values()})
        trial['runtime_breakdown_seconds']=timings
        trial['runtime_scope']='sum of recorded invocation timers, NOT end-to-end stage wall time; resumed invocations may omit earlier work and cached builds. See research cost audit for complete component scopes.'
        if timings:
            trial['runtime_seconds']=sum(timings.values())
        trial.setdefault('peak_memory_bytes',None)
        trial.setdefault('peak_memory_status','not_instrumented; resource budget is a target, not a measured peak')
        if peaks:
            trial['peak_memory_bytes']=max(peaks)
            trial['peak_memory_status']='maximum measured process-lifetime working set; includes prior fits in the same batch'
        for key in ('mean_MAP','delta_vs_Warm_v1','delta_vs_current_champion','non_degrade_windows','worst_window_delta'):
            trial.setdefault(key,None)
    write(REPORT/'WARM_EXPERIMENT_REGISTRY.json',registry)
    lines = ["# Warm-v2 自主实验日志", "", "只记录本分支 Warm 性能工程，不改 Cold 主线。", "",
        "术语：inner 为历史内层筛选窗口；outer 为四个冻结正式确认窗口。MAP@12 的分母包含全部原始真值用户；内层筛选记录的 population delta 是可覆盖活跃用户AP增量乘其占完整用户的比例。未做外层确认的数字标为未运行，不能当成正式成绩。champion 是通过稳定性条件的当前最好方案，milestone 是需人工审核的大阶段结果。", "",
        f"当前冠军：`{registry['current_champion']}`；最终周：`not_run`。", ""]
    for trial in registry["trials"]:
        lines += [f"## {trial['experiment_id']} — {trial['hypothesis']}", "",
            f"- 登记时间：{trial['timestamp']}；父实验：{trial['parent_experiment']}；代码提交：`{trial['git_commit']}`。",
            f"- 状态：{trial['status']}；特征组：{trial['feature_bundle']}；特征数：{trial['feature_count']}。",
            "- 候选身份、验证标签口径与无近期历史用户回退固定；采样、训练日期、损失或上游模型变动以本实验专用合同为准，不能将所有试验都解释为原训练流程。",
            f"- 模型参数覆盖：{trial.get('parameter_overrides',{})}。"]
        if "screening" in trial:
            s = trial['screening']
            lines += [f"- 历史内层平均完整用户贡献增量：{s['mean_population_delta']:+.8f}；正增窗口{s['positive_windows']}；最差{s['worst_population_delta']:+.8f}；筛选通过：{s['passed']}。"]
        if trial.get('realized_feature_layout'):
            lines += [f"- 实际结构：{trial['realized_feature_layout']}。"]
        if trial.get("mean_MAP") is not None:
            lines += [f"- 四窗 MAP：{trial['per_window_MAP']}",
                f"- 均值{trial['mean_MAP']:.8f}；相对Warm-v1增量{trial['delta_vs_Warm_v1']:+.8f}；相对当时冠军增量{trial['delta_vs_current_champion']:+.8f}；不退化{trial['non_degrade_windows']}窗；最差{trial['worst_window_delta']:+.8f}。"]
        else:
            lines += ["- 正式四窗 MAP：未运行；不从历史内层分数外推正式收益。"]
        lines += [f"- 决定：{trial.get('decision','pending')}；证据：`{trial.get('result_path','尚在计算')}`。", ""]
    (REPORT/"WARM_EXPERIMENT_LOG.md").write_text("\n".join(lines).rstrip()+"\n",encoding="utf-8")


def spec(engine, trial, families, extra, role, hypothesis, parent="WV2-000"):
    return {"experiment_id": trial, "parent_experiment": parent, "hypothesis": hypothesis,
        "candidate_contract": engine.contract['candidate_contract'], "feature_bundle": families,
        "feature_count": len(engine.features)+len(extra), "feature_columns": engine.features+extra,
        "training_protocol": engine.contract['training'], "model_params": engine.contract['config'],
        "sampling_protocol": engine.contract['training']['sampling'], "role": role,
        "parameter_overrides":engine.parameter_overrides,
        "per_window_MAP": {w: None for w in engine.contract['rolling_protocol']}, "outer_confirmations": 0}


def baseline():
    if (REPORT/"WARM_V1_REPRODUCTION.json").exists():
        return read(REPORT/"WARM_V1_REPRODUCTION.json")
    start = time.perf_counter()
    c = preregister()
    engine = Engine(c)
    register(spec(engine,"WV2-000",[],[],"baseline_reproduction","重新训练并逐候选复现冻结Warm-v1，不只引用旧MAP",parent="M3.3/Warm-v1"))
    engine.verify_historical_inputs()
    results = {}
    for window in c["rolling_protocol"]:
        print(f"WV2-000 {window}: inner/outer reproduction",flush=True)
        root = ARTIFACT/"WV2-000"/window
        inner = read(root/"inner_metrics.json") if (root/"inner_metrics.json").exists() else engine.train_inner(window,"WV2-000",[],[])
        outer = read(root/"outer_metrics.json") if (root/"outer_metrics.json").exists() else engine.train_outer(window,"WV2-000",[],[],inner)
        results[window] = {"inner": inner, "outer": outer}
        print(f"WV2-000 {window}: parity pass; MAP={outer['evaluation']['map@12']:.9f}",flush=True)
    values = {w:r['outer']['evaluation']['map@12'] for w,r in results.items()}
    result = {"experiment_id": "WV2-000", "status": "baseline_exact_reproduction_passed", "windows": results,
        "per_window_MAP": values, "mean_MAP": float(np.mean(list(values.values()))), "runtime_seconds": time.perf_counter()-start,
        "candidate_pool_changed": False,"training_protocol_changed": False,"final_week": "not_run"}
    path = REPORT/"WARM_V1_REPRODUCTION.json"
    write(path,result)
    record("WV2-000",{"status": "completed", "decision": "retain_for_ablation", "result_path": str(path),
        "per_window_MAP": values,"mean_MAP":result['mean_MAP'],"delta_vs_Warm_v1":0.0,"delta_vs_current_champion":0.0,
        "non_degrade_windows":4,"worst_window_delta":0.0,"runtime_seconds":result['runtime_seconds'],
        "baseline_parity_passed":True})
    log()
    return result


def screen(trial, families, hypothesis, parent="WV2-000", extra_columns=None, parameter_overrides=None):
    assert_branch()
    if (REPORT/f"{trial}_SCREEN.json").exists():
        return read(REPORT/f"{trial}_SCREEN.json")
    base = read(REPORT/"WARM_V1_REPRODUCTION.json")
    assert base["status"] == "baseline_exact_reproduction_passed"
    c = read(REPORT/"WARM_V2_EXPERIMENT_CONTRACT.json")
    engine = Engine(c,parameter_overrides)
    from .warm_v2_features import columns_for
    extra = columns_for(families) if extra_columns is None else list(extra_columns)
    assert len(extra)==len(set(extra)) and set(extra)<=set(columns_for(families))
    register(spec(engine,trial,families,extra,"historical_screening",hypothesis,parent))
    start = time.perf_counter()
    results = {}
    for window in c['rolling_protocol']:
        root = ARTIFACT/trial/window
        print(f"{trial} historical screening: {window}",flush=True)
        result = read(root/"inner_metrics.json") if (root/"inner_metrics.json").exists() else engine.train_inner(window,trial,families,extra)
        delta = (result['inner_covered_active_map']-base['windows'][window]['inner']['inner_covered_active_map'])*result['inner_population_weight']
        result['population_delta_vs_v1'] = delta
        results[window] = result
        print(f"{trial} {window}: historical population delta={delta:+.8f}",flush=True)
    deltas = [v['population_delta_vs_v1'] for v in results.values()]
    gate = c['screening']
    summary = {"mean_population_delta":float(np.mean(deltas)),"positive_windows":sum(d>0 for d in deltas),
        "worst_population_delta":min(deltas),"passed":float(np.mean(deltas))>=gate['mean_population_delta_min'] and sum(d>0 for d in deltas)>=gate['positive_windows_min'] and min(deltas)>=gate['worst_population_delta_min']}
    result = {"experiment_id":trial,"families":families,"windows":results,"screening":summary,"runtime_seconds":time.perf_counter()-start,
        "outer_windows": "not_run","final_week":"not_run"}
    path = REPORT/f"{trial}_SCREEN.json"
    write(path,result)
    record(trial,{"status":"screened","screening":summary,"runtime_seconds":result['runtime_seconds'],"decision":"retain_for_ablation" if summary['passed'] else "reject","result_path":str(path)})
    log()
    return result


def confirm(trial):
    assert_branch()
    if trial=='WV2-502':
        raise ValueError('WV2-502 is a preregistered inner-only attribution diagnostic, never an outer candidate')
    if (REPORT/f"{trial}_OUTER.json").exists():
        return read(REPORT/f"{trial}_OUTER.json")
    screening = read(REPORT/f"{trial}_SCREEN.json")
    assert screening['screening']['passed'], "outer confirmation requires historical screening"
    c = read(REPORT/"WARM_V2_EXPERIMENT_CONTRACT.json")
    from .warm_v2_features import columns_for
    families = screening['families']
    registry = read(REPORT/"WARM_EXPERIMENT_REGISTRY.json")
    entry = next(t for t in registry['trials'] if t['experiment_id']==trial)
    overrides=entry.get('parameter_overrides',{})
    engine = Engine(c,overrides)
    extra = entry['feature_columns'][len(engine.features):]
    assert entry['feature_columns'][:len(engine.features)]==engine.features
    assert set(extra)<=set(columns_for(families))
    if not entry['outer_confirmations']:
        counts = registry['outer_confirmation_counts']
        if overrides:
            polish=read(REPORT/'WARM_RANKER_POLISH_CONTRACT.json')
            assert polish['status']=='activated_after_feature_freeze'
            assert trial in polish['trial_ids'] and counts.get('ranker_polish',0)<1
            counts['ranker_polish']=counts.get('ranker_polish',0)+1
        elif entry.get('research_stage')=='WV2.6':
            learned=read(REPORT/'WV2_501_CONTRACT.json')
            assert trial==learned['trial'] and families==['bpr_match']
            assert counts.get('bpr_match',0)<learned['outer_confirmations_max']
        else:
            assert sum(t['outer_confirmations'] for t in registry['trials'] if not t.get('parameter_overrides')) < c['screening']['outer_trial_limit_first_feature_phase']
        for family in families:
            assert counts.get(family,0) < c['screening']['family_outer_confirmation_limit']
            counts[family] = counts.get(family,0)+1
        entry.update(outer_confirmations=1,status="outer_preregistered",outer_registered_at=now())
        write(REPORT/"WARM_EXPERIMENT_REGISTRY.json",registry)
    start = time.perf_counter()
    results = {}
    for window in c['rolling_protocol']:
        print(f"{trial} formal outer confirmation: {window}",flush=True)
        root = ARTIFACT/trial/window
        results[window] = read(root/"outer_metrics.json") if (root/"outer_metrics.json").exists() else engine.train_outer(window,trial,families,extra,screening['windows'][window])
    base = read(REPORT/"WARM_V1_REPRODUCTION.json")
    values = {w:r['evaluation']['map@12'] for w,r in results.items()}
    deltas = {w:v-base['per_window_MAP'][w] for w,v in values.items()}
    mean = float(np.mean(list(values.values())))
    delta_mean = mean-base['mean_MAP']
    champion = next(t for t in registry['trials'] if t['experiment_id']==registry['current_champion'])
    delta_champ = mean-champion['mean_MAP']
    stable = sum(v>=0 for v in deltas.values())>=3 and min(deltas.values())>=-.0003
    decision = "milestone" if (delta_mean>=.0005 and stable) or delta_mean>=.001 else "new_champion" if delta_champ>0 and stable else "reject"
    result = {"experiment_id":trial,"windows":results,"families":families,"per_window_MAP":values,"mean_MAP":mean,
        "delta_vs_Warm_v1":delta_mean,"delta_vs_current_champion":delta_champ,"non_degrade_windows":sum(v>=0 for v in deltas.values()),
        "worst_window_delta":min(deltas.values()),"per_window_delta":deltas,"decision":decision,"stable":stable,
        "runtime_seconds":time.perf_counter()-start,"final_week":"not_run","candidate_pool_changed":False,
        "training_protocol_changed":bool(overrides),"parameter_overrides":overrides,
        "training_protocol_changed_scope":"LightGBM ranker loop only; upstream feature-model fitting separately recorded",
        "upstream_feature_model_training_added":'bpr_match' in families,
        "training_change_scope":"num_leaves only; objective, sampling, temporal splits and inner early stopping unchanged" if overrides else "none"}
    path = REPORT/f"{trial}_OUTER.json"
    write(path,result)
    record(trial,{"status":"completed","result_path":str(path),**{k:v for k,v in result.items() if k not in ('windows','families')}})
    if stable and delta_champ>0:
        registry = read(REPORT/"WARM_EXPERIMENT_REGISTRY.json")
        registry['current_champion']=trial
        write(REPORT/"WARM_EXPERIMENT_REGISTRY.json",registry)
    log()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['baseline','screen','confirm','screen-first-batch'])
    p.add_argument('--trial')
    p.add_argument('--families',nargs='+',default=[])
    p.add_argument('--hypothesis',default='preregistered feature-family screening')
    a=p.parse_args()
    try:
        if a.command=='screen-first-batch':
            c=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json')
            for i,h in enumerate(c['initial_hypotheses'],start=101):
                screen(f'WV2-{i}',[h['family']],h['hypothesis'])
            result={'status':'first_batch_historical_screening_complete'}
        else:
            result=baseline() if a.command=='baseline' else screen(a.trial,a.families,a.hypothesis) if a.command=='screen' else confirm(a.trial)
        print({k:v for k,v in result.items() if k in ('experiment_id','status','screening','mean_MAP','delta_vs_Warm_v1','decision')},flush=True)
    except Exception:
        write(REPORT/f"FAILURE_{time.time_ns()}.json",{"timestamp":now(),"command":vars(a),"traceback":traceback.format_exc(),"final_week":"not_run"})
        raise


if __name__=='__main__':
    main()
