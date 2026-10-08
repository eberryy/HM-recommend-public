"""One bounded capacity diagnostic after freezing the feature champion."""
from __future__ import annotations

import argparse

from .warm_v2_contract import REPORT, now, read, write
from .warm_v2_lab import screen


def register_polish():
    path=REPORT/'WARM_RANKER_POLISH_CONTRACT.json'
    if path.exists():
        return read(path)
    result={'registered_at':now(),'status':'conditional_plan_not_activated',
        'activation':'All WV2-101..110 historical trials completed; perform any justified remaining feature confirmation first; no feature milestone reached. Freeze the then-current feature champion, including Warm-v1 if all new families rejected.',
        'evidence':'Seven distinct completed feature families have not yielded a stable promoted bundle; eighth candidate-context family pending. Item lifecycle ablation failed. Need distinguish limited tree capacity/regularization from a claim that candidates are exhausted.',
        'change':'num_leaves only: existing31 versus15 versus63; no objective/model-family/sampling/candidate changes; original min_data_in_leaf100, lr0.05, max200 and inner patience20 retained.',
        'trial_ids':['WV2-201','WV2-202'],
        'trials':[{'id':'WV2-201','num_leaves':15},{'id':'WV2-202','num_leaves':63}],
        'selection':'Two historical screens only. Same original screening gate. At most one formal four-window confirmation, highest passing historical mean population delta, tie by id. Never choose leaves from outer results.',
        'limits':'Separate ranker-polish stage: maximum one additional outer confirmation beyond original feature-stage maximum3. No subsequent depth, seed, learning-rate or leaf sweeps.',
        'gate':'Original stable champion and milestone thresholds unchanged.',
        'cost_estimate':'8 CPU historical fits about5-12minutes; selected formal confirmation about3-6minutes; noGPU.',
        'stop':'If no milestone after this bounded stage and eight distinct feature families plus ablations: evidence-backed plateau report; no autonomous retrieval expansion.',
        'fallback':'Retain frozen feature champion. Do not infer candidate ceiling from feature failures alone.',
        'final_week':'not_run'}
    write(path,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['register','run'])
    a=p.parse_args()
    c=register_polish()
    if a.command=='register':
        return
    registry=read(REPORT/'WARM_EXPERIMENT_REGISTRY.json')
    for n in range(101,111):
        assert (REPORT/f'WV2-{n}_SCREEN.json').exists()
    assert not any(t.get('decision')=='milestone' for t in registry['trials'])
    champion=next(t for t in registry['trials'] if t['experiment_id']==registry['current_champion'])
    if c['status']!='activated_after_feature_freeze':
        c.update(status='activated_after_feature_freeze',activated_at=now(),frozen_feature_champion=champion['experiment_id'],
            frozen_feature_columns=champion['feature_columns'],frozen_feature_bundle=champion['feature_bundle'])
        write(REPORT/'WARM_RANKER_POLISH_CONTRACT.json',c)
    for trial in c['trials']:
        screen(trial['id'],c['frozen_feature_bundle'],
            f"冻结{c['frozen_feature_champion']}特征，只把每树最多叶子数从31改为{trial['num_leaves']}，区分更强正则化与更高容量的历史预测效果。",
            parent=c['frozen_feature_champion'],extra_columns=c['frozen_feature_columns'][84:],parameter_overrides={'num_leaves':trial['num_leaves']})


if __name__=='__main__':
    main()
