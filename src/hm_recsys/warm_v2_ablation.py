"""Registered bundle-level ablations, never selected using outer feature gain."""
from __future__ import annotations

import argparse

from .warm_v2_contract import REPORT, now, read, write
from .warm_v2_features import columns_for
from .warm_v2_lab import screen


def contract():
    path=REPORT/'WARM_ITEM_ABLATION_CONTRACT.json'
    if path.exists():
        return read(path)
    all_columns=columns_for(['item_demand'])
    # Partition by semantics fixed in the first feature specification, not fitted gain.
    recent=all_columns[:14]
    lifetime=all_columns[14:]
    assert len(recent)==14 and len(lifetime)==6 and set(recent).isdisjoint(lifetime)
    result={'registered_at':now(),'status':'registered_before_ablation_fits',
        'parent':'WV2-101','reason':'Separate recent-demand information from all-history lifecycle, keeping candidate pool and ranker fixed.',
        'trials':[
            {'id':'WV2-105','columns':recent,'hypothesis':'只保留新增20列中的14列近期需求统计，移除6列全历史生命周期；检验生命周期是否提供额外价值。'},
            {'id':'WV2-106','columns':lifetime,'hypothesis':'只保留6列全历史生命周期，移除14列近期需求统计；检验短期需求是否为必要信息。'}],
        'selection':'Historical screening only. At most one ablation receives outer confirmation: choose highest historical population mean delta among passing ablations, tie by trial id. Count against original item_demand family limit.',
        'limits':'No per-window choices; no parameter tuning; no new features; no gate changes.',
        'interpretation':'Refitted bundle ablations estimate protocol-level predictive value, not causal effects of individual correlated features.',
        'final_week':'not_run'}
    write(path,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['register','run'])
    a=p.parse_args()
    c=contract()
    if a.command=='run':
        for trial in c['trials']:
            screen(trial['id'],['item_demand'],trial['hypothesis'],parent=c['parent'],extra_columns=trial['columns'])


if __name__=='__main__':
    main()
