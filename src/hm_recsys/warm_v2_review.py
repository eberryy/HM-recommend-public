"""Read-only identity and per-user ranking review of completed Warm trials."""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np

from .metrics import apk
from .warm_v2_contract import ARTIFACT, REPORT, assert_branch, guard, now, read, write
from .warm_v2_engine import literal


def review(trial):
    assert_branch()
    result=read(REPORT/f'{trial}_OUTER.json')
    base=read(REPORT/'WARM_V1_REPRODUCTION.json')
    c=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json')
    m=read(c['historical_metrics'])
    windows={}
    for window,p in c['rolling_protocol'].items():
        cutoff=guard(p['outer_validation'])
        with duckdb.connect(str(ARTIFACT/trial/window/'evaluation.duckdb'),read_only=True) as con:
            con.execute(f"ATTACH {literal((ARTIFACT/'WV2-000'/window/'evaluation.duckdb').resolve())} AS base (READ_ONLY)")
            identity=con.execute('''SELECT count(*) total,
                count(*) FILTER(WHERE n.customer_id IS NULL OR b.customer_id IS NULL) missing,
                count(*) FILTER(WHERE n.candidate_rank<>b.candidate_rank OR n.target<>b.target OR n.user_history_events_12w<>b.user_history_events_12w) mismatch
                FROM predictions n FULL OUTER JOIN base.predictions b USING(customer_id,article_id)''').fetchone()
            assert identity[0]==result['windows'][window]['evaluation']['candidate_rows'] and identity[1:]==(0,0)
            duplicates=con.execute('SELECT count(*)-count(DISTINCT (customer_id,article_id)) FROM predictions').fetchone()[0]
            assert duplicates==0
            fallback=con.execute('''SELECT count(*) FROM top12 n FULL OUTER JOIN base.top12 b USING(customer_id,article_id,final_rank)
                WHERE (n.user_history_events_12w=0 OR b.user_history_events_12w=0)
                AND (n.customer_id IS NULL OR b.customer_id IS NULL)''').fetchone()[0]
            assert fallback==0
            truth=con.execute(f"""SELECT customer_id,article_id FROM read_parquet({literal(m['inputs']['transactions']['path'])})
                WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY
                AND customer_id IN(SELECT DISTINCT customer_id FROM predictions)
                GROUP BY customer_id,article_id ORDER BY customer_id,article_id""").fetchdf()
            n=con.execute('SELECT * FROM top12 ORDER BY customer_id,final_rank').fetchdf()
            b=con.execute('SELECT * FROM base.top12 ORDER BY customer_id,final_rank').fetchdf()
            npred={u:g.article_id.tolist() for u,g in n.groupby('customer_id',sort=False)}
            bpred={u:g.article_id.tolist() for u,g in b.groupby('customer_id',sort=False)}
            tsets={u:g.article_id.tolist() for u,g in truth.groupby('customer_id',sort=False)}
            assert set(npred)==set(bpred)==set(tsets)
            users=sorted(tsets)
            nap=np.array([apk(tsets[u],npred[u]) for u in users])
            bap=np.array([apk(tsets[u],bpred[u]) for u in users])
            delta=nap-bap
            assert abs(nap.mean()-result['per_window_MAP'][window])<1e-12
            assert abs(bap.mean()-base['per_window_MAP'][window])<1e-12
            both=result['windows'][window]['evaluation'],base['windows'][window]['outer']['evaluation']
            for key in ('candidate_recall','candidate_oracle_map@12','candidate_rows','users','truth_pairs'):
                assert both[0][key]==both[1][key]
            pos=delta[delta>1e-15];neg=delta[delta< -1e-15]
            new_features=[v for v in result['windows'][window]['feature_importance'] if v['feature'].startswith('wv2_')]
            windows[window]={'cutoff':cutoff,'candidate_identity':{'joined_pairs':identity[0],'missing_pairs':identity[1],
                'rank_label_history_mismatches':identity[2],'duplicate_pairs':duplicates},'inactive_top12_mismatches':fallback,
                'MAP_recomputed':float(nap.mean()),'delta_recomputed':float(delta.mean()),'users':len(users),
                'users_improved':len(pos),'users_harmed':len(neg),'users_unchanged':len(users)-len(pos)-len(neg),
                'gross_positive_MAP_contribution':float(pos.sum()/len(users)),'gross_negative_MAP_contribution':float(neg.sum()/len(users)),
                'paired_delta_standard_error_descriptive':float(delta.std(ddof=1)/np.sqrt(len(delta))),
                'new_features_used':sorted(new_features,key=lambda x:-x['gain']),
                'candidate_recall':both[0]['candidate_recall'],'candidate_oracle_map@12':both[0]['candidate_oracle_map@12'],
                'top12_pairs_changed':int(con.execute('SELECT count(*) FROM top12 n ANTI JOIN base.top12 b USING(customer_id,article_id)').fetchone()[0])}
    output={'experiment_id':trial,'reviewed_at':now(),'status':'passed','windows':windows,
        'interpretation':'Paired standard error is descriptive, ignores multiple trial selection and cross-window dependence. Feature gain is model usage, not causal contribution.',
        'checks':'Independent pair-set equality, target/rank/history equality, no duplicates, exact inactive fallback, exact full-truth MAP recomputation, unchanged candidate recall/oracle.',
        'final_week':'not_run'}
    write(REPORT/f'{trial}_REVIEW.json',output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial')
    args=parser.parse_args()
    print(review(args.trial)['status'])
