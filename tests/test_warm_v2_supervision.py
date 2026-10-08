import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

from hm_recsys.warm_v2_engine import save_parquet
from hm_recsys.warm_v2_supervision import full_relation


class SupervisionTests(unittest.TestCase):
    def test_full_negative_keeps_users_and_positives_and_restores_competitors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=pd.DataFrame({'customer_id':['u','u','u','v','v'],'article_id':['1','2','3','1','4'],
                'target':[1,0,0,0,0],'candidate_rank':[3,1,2,1,2]})
            sampled=source.iloc[[0,2]]
            save_parquet(source,root/'source.parquet');save_parquet(sampled,root/'sample.parquet')
            with duckdb.connect() as con:
                restored=con.execute(full_relation(root/'source.parquet',root/'sample.parquet')).fetchdf()
            self.assertEqual(set(restored.customer_id),{'u'})
            self.assertEqual(set(restored.article_id),{'1','2','3'})
            self.assertEqual(restored.target.sum(),sampled.target.sum())
            self.assertEqual(len(restored),3)


if __name__=='__main__':unittest.main()
