"""Read-only fixed Top50 supervision/headroom audit; no training or new OUTER."""
from __future__ import annotations
from datetime import date, timedelta
import time
import numpy as np
from . import warm_v3_common as common
from .warm_v2_contract import read, write, now
from .warm_v2_engine import literal
from .warm_v3_pool import connection
from .warm_v2_rank_fusion import ap_table

PRIOR = ('2019-02-20','2019-05-22','2019-08-21','2019-11-20')
INNER = ('2019-12-25','2020-02-19','2020-05-27','2020-07-22')
K = 50


def safe_prior(prior, target):
    return date.fromisoformat(prior)+timedelta(days=7) <= date.fromisoformat(target)


def pair_counts(pos_head, pos_total, n_head=12, n_total=50):
    if not 0 <= pos_head <= n_head <= n_total or not pos_head <= pos_total <= n_total or pos_total-pos_head > n_total-n_head:
        raise ValueError('Invalid pair counts')
    total = pos_total*(n_total-pos_total)
    sensitive = pos_head*(n_total-pos_total)+(pos_total-pos_head)*(n_head-pos_head)
    return total,sensitive


def audit_one(cutoff, role):
    year = 2019 if role=='prior_outer_meta' else 2020
    folder = common.ART/'gate_data'/f'{year}_{"outer" if year==2019 else "inner"}_{cutoff}'
    meta = read(folder/'DATA.json')
    assert meta['cutoff']==cutoff and meta['final_week']=='not_run'
    if year==2019:
        check=meta['independent_source_check']
        assert all(v==0 for v in check.values())
    n = meta['total_users']
    with connection() as con:
        con.execute(f'''CREATE TEMP TABLE raw AS SELECT *,ap_rf stage1_rank FROM read_parquet({literal(meta['ranks_path'])})''')
        con.execute('CREATE TEMP TABLE active AS SELECT * FROM raw WHERE user_history_events_12w>0')
        assert con.execute('SELECT count(*)-count(DISTINCT (customer_id,article_id)) FROM active').fetchone()[0]==0
        con.execute('''CREATE TEMP TABLE groups AS SELECT customer_id,max(truth_count) truth_count,count(*) full_rows,
            sum(target) full_pos,count(*) FILTER(WHERE stage1_rank<=50) n50,
            count(*) FILTER(WHERE stage1_rank<=12) n12,
            sum(target) FILTER(WHERE stage1_rank<=50) p50,sum(target) FILTER(WHERE stage1_rank<=12) p12
            FROM active GROUP BY customer_id''')
        frame=con.execute('SELECT * FROM groups ORDER BY customer_id').fetchdf()
        ap=ap_table(con,'active','stage1_rank')
        top_users=frame.p50.gt(0)
        covered=frame.full_pos.gt(0)
        if year==2020:
            assert covered.all()  # original INNER scores cover active candidate-positive groups only
        all_pairs=sum(int(p)*(int(size)-int(p)) for p,size in zip(frame.p50,frame.n50))
        sensitive=sum(pair_counts(int(a),int(b),int(c),int(d))[1] for a,b,c,d in zip(frame.p12,frame.p50,frame.n12,frame.n50))
        # Negative currently above positive; only negatives in Top12 can produce positive AP@12 swap change.
        improvement_pairs=con.execute('''SELECT count(*) FROM active p JOIN active q USING(customer_id)
            WHERE p.target=1 AND q.target=0 AND p.stage1_rank<=50 AND q.stage1_rank<=12
            AND p.stage1_rank>q.stage1_rank''').fetchone()[0]
        oracle50=float((np.minimum(frame.p50,12)/np.minimum(frame.truth_count,12)).sum()/n)
        oracle12=float((np.minimum(frame.p12,12)/np.minimum(frame.truth_count,12)).sum()/n)
        baseline=float(ap.ap.sum()/n)
        count={'full_user_denominator':n,'observed_active_scored_users':len(frame),
            'candidate_covered_active_users':int(covered.sum()),'top50_positive_users':int(top_users.sum()),
            'top50_positive_pairs':int(frame.p50.sum()),'top12_positive_pairs':int(frame.p12.sum()),
            'full_pool_positive_pairs':int(frame.full_pos.sum()),
            'top50_positive_pair_retention':float(frame.p50.sum()/frame.full_pos.sum()),
            'positive_pairs_rank13_50':int((frame.p50-frame.p12).sum()),
            'users_with_positive_rank13_50':int((frame.p50>frame.p12).sum()),
            'head50_rows_among_candidate_covered_users':int(frame.loc[covered,'n50'].sum()),
            'head50_density_among_candidate_covered_users':float(frame.p50.sum()/frame.loc[covered,'n50'].sum()),
            'wide_pool_density_among_candidate_covered_users':float(frame.full_pos.sum()/frame.loc[covered,'full_rows'].sum()),
            'head50_training_rows_positive_head_users':int(frame.loc[top_users,'n50'].sum()),
            'head50_training_density_positive_head_users':float(frame.p50.sum()/frame.loc[top_users,'n50'].sum()),
            'positive_negative_pairs_within50':all_pairs,'ap12_sensitive_pairs':sensitive,
            'ap12_sensitive_pair_fraction':sensitive/all_pairs if all_pairs else 0,
            'positive_AP12_swap_pairs':improvement_pairs,
            'zero_AP12_pair_fraction':1-sensitive/all_pairs if all_pairs else 0,
            'baseline_active_MAP_component':baseline,'oracle_MAP_within12':oracle12,'oracle_MAP_within50':oracle50,
            'top50_oracle_minus_stage1_MAP':oracle50-baseline,'top50_oracle_minus_top12_oracle':oracle50-oracle12}
    return {'cutoff':cutoff,'role':role,'source_metadata':str(folder/'DATA.json'),'ranks_path':meta['ranks_path'],
        'population_note':'2019 outer artifacts include all active users; 2020 INNER only covered-active. Comparable density uses candidate-covered-active groups; all MAP differences use full original cohort.',
        'statistics':count,'final_week':'not_run'}


def run():
    common.setup()
    common.budget(2)
    started=time.perf_counter()
    rows=[audit_one(c,'prior_outer_meta') for c in PRIOR]+[audit_one(c,'inner_screen') for c in INNER]
    temporal={c:[p for p in PRIOR if safe_prior(p,c)] for c in INNER}
    assert all(len(v)==4 for v in temporal.values())
    prior=[r['statistics'] for r in rows if r['role']=='prior_outer_meta']
    output={'created_at':now(),'role':'read_only_head50_headroom','fixed_K':K,'new_training':0,'new_outer':0,
        'windows':rows,'safe_prior_meta_by_inner_target':temporal,
        'prior_totals':{key:sum(r[key] for r in prior) for key in
            ('top50_positive_pairs','top50_positive_users','head50_training_rows_positive_head_users',
             'positive_negative_pairs_within50','ap12_sensitive_pairs','positive_AP12_swap_pairs')},
        'runtime_seconds':time.perf_counter()-started,'final_week':'not_run',
        'selection_caveat':'2019 outer expert predictions were held out from expert training, but reused as meta supervision are not a new independent test. Original2020 INNER checkpoint used its labels for early stopping; screen only.',
        'automatic_model_authorization':False}
    write(common.REPORT/'MULTISTAGE_HEADROOM_AUDIT.json',output)
    print({'prior_totals':output['prior_totals'],'seconds':output['runtime_seconds']},flush=True)
    return output


if __name__=='__main__':
    run()
