"""Read-only bounded within-user expert mixing headroom; never an executable policy."""
import numpy as np
from .warm_v3_common import *
from .warm_v2_engine import connection,literal
from .warm_v2_rank_fusion import rank_sql,ap_table

def main():
    setup();budget(2);source=read(REPORT/'WV3_100_GATE_ORACLE_AUDIT.json');windows={}
    for m in source['inner']:
        with connection() as con:
            con.execute(f'''CREATE TEMP TABLE x AS SELECT *,
                CASE WHEN target=1 THEN greatest(1.0/(60+r0),1.0/(60+r1))
                ELSE least(1.0/(60+r0),1.0/(60+r1)) END bounded_oracle FROM read_parquet({literal(m['ranks_path'])})''')
            con.execute(f'CREATE TEMP TABLE ranked AS SELECT *,{rank_sql("bounded_oracle")} oracle_rank FROM x')
            candidate=float(ap_table(con,'ranked','oracle_rank').ap.sum()/m['total_users'])
            counts=con.execute('''WITH per_user AS (SELECT customer_id,
                count(*) FILTER(WHERE target=1 AND r0<r1) e0_better,
                count(*) FILTER(WHERE target=1 AND r1<r0) e1_better,sum(target) positives FROM x GROUP BY customer_id)
                SELECT count(*),count(*) FILTER(WHERE e0_better>0 AND e1_better>0),
                    count(*) FILTER(WHERE positives>=2) FROM per_user''').fetchone()
        user_oracle=m['baseline_map_population_component']+m['oracle_headroom_population_MAP']
        windows[m['window']]={'cutoff':m['cutoff'],'eligible_users':counts[0],'conflicting_positive_expert_users':counts[1],
            'users_with_two_or_more_candidate_positives':counts[2],
            'bounded_candidate_oracle_population_component':candidate,'user_choice_oracle_population_component':user_oracle,
            'extra_headroom_over_user_choice':candidate-user_oracle,
            'headroom_over_frozen_RRF':candidate-m['baseline_map_population_component']}
    ds=[m['extra_headroom_over_user_choice'] for m in windows.values()]
    out={'created_at':now(),'windows':windows,'mean_extra_headroom':float(np.mean(ds)),
        'mechanism_gate':bool(np.mean(ds)>=.001 and sum(d>=.0005 for d in ds)>=3),
        'definition':'项目只读不可实现上限：每个正例选择两专家中更高的倒数名次分数，每个负例选择较低者；仍受两专家分数上下界约束。需要未来标签，严禁当推理规则。',
        'comparison_caution':'候选级边界上限超过整用户专家选择，不等于门控特征能够识别正确选择；内层专家检查点选取偏差仍存在。',
        'supervision_units':'冲突用户指同一用户至少两个已召回真值商品，分别在不同专家中名次更靠前；不是一个候选有冲突标签。',
        'final_week':'not_run'}
    write(REPORT/'WV3_110_CANDIDATE_GATE_AUDIT.json',out);print(out,flush=True)

if __name__=='__main__':main()
