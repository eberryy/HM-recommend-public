"""Prerequisite/funnel audit for a learned user-item matching feature, no fitting."""
from __future__ import annotations

import time
import lightgbm as lgb
from .warm_v2_contract import ARTIFACT,REPORT,assert_branch,guard,now,read,write
from .warm_v2_engine import Engine,connection,literal,load_parquet,prepare


def audit():
    assert_branch();start=time.perf_counter()
    e=Engine(read(REPORT/'WARM_V2_EXPERIMENT_CONTRACT.json'))
    registration={'registered_at':now(),'stage':'WV2.6 learned user-item prerequisite audit',
        'hypothesis':'Existing item-item/source/hierarchy summaries omit directly learned user-ID/item-ID collaborative matching; inspect supported non-repeat positives missed belowTop12.',
        'scope':'four historical inner validations only; no new fits and no outer diagnosis',
        'support_rule':'user>=5 distinct past items and item>=5 distinct past purchasing users, all strictlybeforecutoff',
        'opportunity_gate':'each inner window has>=100 supported non-repeat positive candidate pairs belowbaselineTop12, and these account for>=20% of all missed positive candidate pairs in>=3windows',
        'not_proof':'Support and missed positives are prerequisites only, never a guarantee that BPR learns their ranking.',
        'sources':['https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129','https://arxiv.org/abs/1205.2618','https://benfred.github.io/implicit/api/models/cpu/bpr.html'],
        'final_week':'not_run'}
    path=REPORT/'WV2_500_AUDIT_CONTRACT.json'
    if not path.exists():write(path,registration)
    windows={}
    for w,p in e.contract['rolling_protocol'].items():
        cutoff=p['inner_validation'];guard(cutoff)
        fp,meta=e.cached_data(cutoff,'inner');frame=load_parquet(fp)
        maps=read(ARTIFACT/'WV2-000'/w/'inner_category_maps.json')
        maps={k:{int(a):int(b) for a,b in v.items()} for k,v in maps.items()}
        model=lgb.Booster(model_file=str(ARTIFACT/'WV2-000'/w/'inner_model.txt'))
        frame['score']=model.predict(prepare(frame,e.features,maps))
        kept=frame[['customer_id','article_id','candidate_rank','target','score']]
        with connection() as con:
            con.register('score_frame',kept)
            con.execute('CREATE TEMP TABLE scored AS SELECT *,row_number() OVER(PARTITION BY customer_id ORDER BY score DESC,candidate_rank,article_id) predicted_rank FROM score_frame')
            con.execute(f"CREATE VIEW history AS SELECT customer_id,article_id FROM read_parquet({literal(e.transactions)}) WHERE t_dat<DATE '{cutoff}'")
            con.execute('CREATE TEMP TABLE hu AS SELECT customer_id,count(DISTINCT article_id) past_items FROM history SEMI JOIN (SELECT DISTINCT customer_id FROM scored) u USING(customer_id) GROUP BY customer_id')
            con.execute('CREATE TEMP TABLE hi AS SELECT article_id,count(DISTINCT customer_id) past_users FROM history GROUP BY article_id')
            con.execute('''CREATE TEMP TABLE a AS SELECT s.*,coalesce(u.past_items,0)>=5 AND coalesce(i.past_users,0)>=5 supported,
                EXISTS(SELECT 1 FROM history h WHERE h.customer_id=s.customer_id AND h.article_id=s.article_id) repeated
                FROM scored s LEFT JOIN hu u USING(customer_id) LEFT JOIN hi i USING(article_id)''')
            n,pos,miss,eligible,nonrepeat=con.execute('''SELECT count(*),sum(target),
                count(*) FILTER(WHERE target=1 AND predicted_rank>12),
                count(*) FILTER(WHERE target=1 AND predicted_rank>12 AND supported AND NOT repeated),
                count(*) FILTER(WHERE target=1 AND NOT repeated) FROM a''').fetchone()
            users,items,pairs=con.execute('SELECT count(DISTINCT customer_id),count(DISTINCT article_id),count(DISTINCT(customer_id,article_id)) FROM history').fetchone()
        windows[w]={'cutoff':cutoff,'evaluated_covered_active_candidate_rows':n,'positive_candidate_pairs':pos,
            'missed_below12_positive_pairs':miss,'supported_nonrepeat_missed_pairs':eligible,
            'supported_nonrepeat_share_of_missed':eligible/miss,'all_nonrepeat_positive_pairs':nonrepeat,
            'all_history_users':users,'all_history_items':items,'all_history_binary_pairs':pairs,
            'csr_and_100dim_float32_factor_lower_bound_bytes':pairs*8+(users+1)*4+(users+items)*101*4}
        print(w,windows[w],flush=True)
        del frame,kept,model
    passed=min(v['supported_nonrepeat_missed_pairs'] for v in windows.values())>=100 and sum(v['supported_nonrepeat_share_of_missed']>=.2 for v in windows.values())>=3
    out={'created_at':now(),'status':'measured','windows':windows,'opportunity_gate_passed':passed,
        'runtime_seconds':time.perf_counter()-start,'final_week':'not_run','outer_labels_used':False,
        'denominator':'Covered-active inner dataset = original validation users with past12w activity and >=1 positive in the frozen candidate pool. Not the full MAP denominator; used only to locate recoverable candidates.',
        'interpretation':'Only an opportunity audit; BPR must be separately pre-registered, trained strictlybeforeeachcutoff and confirmed by same MAP gates. Memory estimate excludes optimizer/buffers/library overhead.'}
    write(REPORT/'WV2_500_BPR_AUDIT.json',out);return out


if __name__=='__main__':audit()
