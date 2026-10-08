import unittest
import duckdb
import pandas as pd
from hm_recsys.warm_v3_consensus_admission import build_admission_ranks


class ConsensusAdmissionTests(unittest.TestCase):
    def test_one_unanimous_swap_and_complete_permutation(self):
        rows=[]
        for rank in range(1,16):
            rows.append({'customer_id':'u','article_id':rank,'candidate_rank':rank,'target':0,
                'truth_count':1,'user_history_events_12w':3,'baseline_rank':rank,
                'bpr_rank':rank,'sequence_rank':rank,'lightgcn_rank':rank,
                'bpr_unavailable':0,'sequence_unavailable':0,'lightgcn_unavailable':0})
        frame=pd.DataFrame(rows)
        # Candidate13 unanimously beats baseline item12, but no earlier Top12 item.
        for col in ('bpr_rank','sequence_rank','lightgcn_rank'):
            frame[col]=frame[col].astype(float)
            frame.loc[frame.article_id==13,col]=11.5
        with duckdb.connect() as con:
            con.register('signals',frame);table=build_admission_ranks(con)
            out=con.execute(f'SELECT article_id,final_rank FROM {table} ORDER BY final_rank').fetchall()
        self.assertEqual(dict(out)[13],12)
        self.assertEqual(dict(out)[12],13)
        self.assertEqual(sorted(r for _,r in out),list(range(1,16)))

    def test_no_swap_without_unanimity(self):
        frame=pd.DataFrame([{'customer_id':'u','article_id':i,'candidate_rank':i,'target':0,
            'truth_count':1,'user_history_events_12w':3,'baseline_rank':i,'bpr_rank':i,
            'sequence_rank':i,'lightgcn_rank':i,'bpr_unavailable':0,
            'sequence_unavailable':0,'lightgcn_unavailable':0} for i in range(1,14)])
        with duckdb.connect() as con:
            con.register('signals',frame);table=build_admission_ranks(con)
            swaps=con.execute('SELECT count(*) FROM chosen_swap').fetchone()[0]
            changed=con.execute(f'SELECT count(*) FROM {table} WHERE final_rank<>baseline_rank').fetchone()[0]
        self.assertEqual((swaps,changed),(0,0))


if __name__=='__main__':unittest.main()
