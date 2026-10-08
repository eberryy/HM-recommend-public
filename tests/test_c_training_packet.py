"""Read-only checks of the prepared C packet; no model fitting."""
import json
from pathlib import Path
import unittest
import duckdb

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'artifacts/c_recent_first_layer/v1-prep2/training_packet'


@unittest.skipUnless((PACK/'PACKET.json').exists(), 'Local private training packet unavailable')
class PacketTests(unittest.TestCase):
    def test_time_and_model_contract(self):
        m=json.loads((PACK/'PACKET.json').read_text(encoding='utf-8'))
        self.assertEqual(m['feature_cutoff'],m['label_start'])
        self.assertEqual(m['label_end_exclusive'],m['inference_cutoff'])
        self.assertEqual(m['feature_evidence']['latest_history_date'],'2020-09-15')
        self.assertEqual(m['feature_reconstruction_mismatches'],0)
        self.assertEqual([m['models'][k]['rounds'] for k in ['wv2_base','wv2_bpr']],[67,7])
        self.assertEqual(m['models']['wv2_bpr']['features'],m['models']['wv2_base']['features']+
                         ['wv2_bpr_user_item_score','wv2_bpr_unavailable'])

    def test_rows_groups_and_bpr_missing(self):
        m=json.loads((PACK/'PACKET.json').read_text(encoding='utf-8'))
        with duckdb.connect() as db:
            p=str(PACK/'training.parquet').replace("'","''")
            db.execute(f"CREATE VIEW data AS SELECT * FROM read_parquet('{p}')")
            got=db.execute('SELECT count(*),count(DISTINCT customer_id),sum(target),count(DISTINCT(customer_id,article_id)) FROM data').fetchone()
            self.assertEqual(got,(m['sample_rows'],m['sample_groups'],m['positive_rows'],m['sample_rows']))
            self.assertEqual(db.execute('''SELECT count(*) FROM (SELECT customer_id,count(*) n,sum(target) pos FROM data GROUP BY customer_id)
                WHERE pos<=0 OR n-pos>30*pos''').fetchone()[0],0)
            self.assertEqual(db.execute('''SELECT count(*) FROM data WHERE
                (wv2_bpr_unavailable=1 AND wv2_bpr_user_item_score IS NOT NULL)
                OR (wv2_bpr_unavailable=0 AND (wv2_bpr_user_item_score IS NULL OR NOT isfinite(wv2_bpr_user_item_score)))''').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT count(*) FROM data WHERE hash(customer_id)%1000000>=100000').fetchone()[0],0)


if __name__=='__main__':unittest.main()
