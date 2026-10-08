"""Inner-only diagnostic of supervised target freshness; no model fitting."""
from __future__ import annotations

from datetime import date, timedelta
import time

from .warm_v2_contract import REPORT, assert_branch, guard, now, read, write
from .warm_v2_engine import Engine, connection, literal


def week_before(cutoff, weeks):
    guard(cutoff)
    assert weeks >= 1
    return (date.fromisoformat(cutoff)-timedelta(weeks=weeks)).isoformat()


def audit():
    assert_branch()
    start=time.perf_counter()
    contract=read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json')
    engine=Engine(contract)
    # Registered before reading diagnostic outcomes. This is an opportunity gate,
    # never evidence that fresh labels will improve a trained ranker.
    registration={'created_at':now(),'stage':'WV2.5 supervision freshness audit',
        'ages_weeks':[1,2,4,8], 'labels':'four historical inner validations only',
        'gate':'recent1wk vs old4wk: truth-pair-weighted global Top1000 item coverage increases in >=3/4 inner windows; user-item repeat coverage also reported, not a selection knob',
        'next_if_pass':'cost one new cutoff with exactly frozen candidate builders; pre-register recent-target training comparison before fitting',
        'next_if_fail':'audit learned BPR user-item matching rather than more time features',
        'cost':'CPU8, raw parquet aggregation, expected 1-3min; no models/GPU',
        'final_week':'not_run'}
    rp=REPORT/'WV2_400_AUDIT_CONTRACT.json'
    if not rp.exists(): write(rp,registration)
    results={}
    with connection() as con:
        con.execute(f'CREATE VIEW tx AS SELECT * FROM read_parquet({literal(engine.transactions)})')
        for window,p in contract['rolling_protocol'].items():
            cutoff=p['inner_validation'];guard(cutoff)
            con.execute(f"""CREATE OR REPLACE TEMP TABLE truth AS
                SELECT DISTINCT t.customer_id,t.article_id FROM tx t
                SEMI JOIN (SELECT DISTINCT customer_id FROM read_parquet({literal(engine.base_path(cutoff))})) u USING(customer_id)
                WHERE t_dat>=DATE '{cutoff}' AND t_dat<DATE '{cutoff}'+INTERVAL 7 DAY""")
            count,users=con.execute('SELECT count(*),count(DISTINCT customer_id) FROM truth').fetchone()
            ages={}
            for age in (1,2,4,8):
                begin=week_before(cutoff,age)
                con.execute(f"""CREATE OR REPLACE TEMP TABLE past AS SELECT customer_id,article_id
                    FROM tx WHERE t_dat>=DATE '{begin}' AND t_dat<DATE '{begin}'+INTERVAL 7 DAY""")
                con.execute('''CREATE OR REPLACE TEMP TABLE pop AS SELECT article_id,count(*) events,
                    row_number() OVER(ORDER BY count(*) DESC,article_id) popularity_rank FROM past GROUP BY article_id''')
                nitem,top100,top1000=con.execute('''SELECT count(*) FILTER(WHERE p.article_id IS NOT NULL),
                    count(*) FILTER(WHERE p.popularity_rank<=100),count(*) FILTER(WHERE p.popularity_rank<=1000)
                    FROM truth t LEFT JOIN pop p USING(article_id)''').fetchone()
                repeat=con.execute('SELECT count(*) FROM truth SEMI JOIN past USING(customer_id,article_id)').fetchone()[0]
                events=con.execute('SELECT count(*) FROM past').fetchone()[0]
                ages[str(age)]={'target_week_start':begin,'historical_events':events,
                    'truth_item_seen_pairs':nitem,'truth_item_seen_fraction':nitem/count,
                    'global_top100_truth_pairs':top100,'global_top100_truth_fraction':top100/count,
                    'global_top1000_truth_pairs':top1000,'global_top1000_truth_fraction':top1000/count,
                    'same_user_same_item_truth_pairs':repeat,'same_user_same_item_truth_fraction':repeat/count}
            results[window]={'inner_cutoff':cutoff,'truth_pairs':count,'truth_users':users,'ages':ages,
                'recent_minus_old_global_top1000':ages['1']['global_top1000_truth_fraction']-ages['4']['global_top1000_truth_fraction'],
                'recent_minus_old_same_user_item':ages['1']['same_user_same_item_truth_fraction']-ages['4']['same_user_same_item_truth_fraction']}
    positive=sum(r['recent_minus_old_global_top1000']>0 for r in results.values())
    out={'status':'measured','created_at':now(),'windows':results,'positive_windows':positive,
        'audit_gate_passed':positive>=3,'final_week':'not_run','outer_labels_used':False,
        'runtime_seconds':time.perf_counter()-start,
        'definitions':{'truth_pairs':'每个固定历史验证周、原10%验证用户的去重用户—商品购买对；所有覆盖率均以本窗全部这些真值对为分母。',
            'global_top1000':'行业常见销量排行：对应历史7天全部用户的商品事件数Top1000，重复事件保留；统计这些商品覆盖多少验证真值对，不是1000件新召回试验。',
            'same_user_same_item':'项目诊断：验证真值用户—商品对在相应历史7天也发生购买；仅度量复购时间分布。'},
        'limits':['Descriptive predictive freshness is not a trained-model gain or causal mechanism.',
            'Historical weeks differ in purchase volume; report event counts. A later training comparison must keep weekly count and sampling rule fixed before scaling supervision.',
            'No outer score is used to choose this diagnostic.']}
    write(REPORT/'WV2_400_FRESHNESS_AUDIT.json',out)
    print({w:{k:v[k] for k in ('recent_minus_old_global_top1000','recent_minus_old_same_user_item')} for w,v in results.items()},flush=True)
    return out


if __name__=='__main__':
    audit()
