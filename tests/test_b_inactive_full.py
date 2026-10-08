import importlib.util
from pathlib import Path
import tempfile
import unittest
import duckdb
import pandas as pd

spec = importlib.util.spec_from_file_location('b_full', Path(__file__).resolve().parents[1] / 'scripts/run_b_inactive_full.py')
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)


def parquet(frame, path):
    with duckdb.connect() as db:
        db.register('frame', frame)
        db.execute('COPY frame TO ? (FORMAT PARQUET)', [str(path)])


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        items = [f'{i:010d}' for i in range(30)]
        self.items = items
        pd.DataFrame({'article_id': items}).to_csv(self.root/'articles.csv', index=False)
        pd.DataFrame({'customer_id': ['a', 'b', 'c'], 'prediction': [' '.join(items[:12])]*3}).to_csv(self.root/'original.csv', index=False)
        pd.DataFrame({'customer_id': ['c', 'b', 'a']}).to_csv(self.root/'sample.csv', index=False)
        parquet(pd.DataFrame({'customer_id': ['b', 'c']}), self.root/'inactive.parquet')
        self.rows = pd.DataFrame([(u, i, rank) for u in ['b', 'c'] for rank, i in enumerate(items[12:24], 1)],
                                 columns=['customer_id', 'article_id', 'output_rank'])

    def merge(self):
        parquet(self.rows, self.root/'rank.parquet')
        with duckdb.connect() as db:
            return full.merge_submission(db, self.root/'original.csv', self.root/'sample.csv',
                 self.root/'articles.csv', self.root/'inactive.parquet', [self.root/'rank.parquet'],
                 self.root/'out.csv', total=3, inactive_count=2)

    def test_merge_preserves_active_and_replaces_inactive(self):
        result = self.merge()
        self.assertEqual(result['active_changed_users'], 0)
        self.assertEqual(result['inactive_changed_users'], 2)
        got = pd.read_csv(self.root/'out.csv').set_index('customer_id')
        self.assertEqual(got.loc['a','prediction'], ' '.join(self.items[:12]))
        self.assertEqual(got.loc['b','prediction'], ' '.join(self.items[12:24]))

    def test_reject_duplicate_items(self):
        self.rows.loc[0, 'article_id'] = self.rows.loc[1, 'article_id']
        with self.assertRaisesRegex(RuntimeError, 'Invalid or duplicated'):
            self.merge()

    def test_reject_wrong_population(self):
        self.rows.loc[self.rows.customer_id == 'c', 'customer_id'] = 'd'
        with self.assertRaisesRegex(RuntimeError, 'membership differs'):
            self.merge()

    def test_reject_unknown_article(self):
        self.rows.loc[0, 'article_id'] = '9999999999'
        with self.assertRaisesRegex(RuntimeError, 'outside catalog'):
            self.merge()

    def test_completed_batch_check(self):
        directory = self.root/'attempt'; directory.mkdir()
        rows = pd.DataFrame([('b', str(i), i+1) for i in range(100)], columns=['customer_id','article_id','output_rank'])
        parquet(rows, directory/'pilot-ranking.parquet')
        full.write(directory/'RESULT.json', {'status':'completed','users':1,'candidate_rows':100,
            'contract':{'stage':'B-INACTIVE-CPU-BATCH-v1','offset':0,'cutoff':'2020-09-23'},
            'original_top12_reconstruction_mismatches':0,'active_score_max_abs_error':0,
            'independent_rank_replay':'passed'})
        with duckdb.connect() as db:
            self.assertEqual(full.check_batch(db,directory,0,1,self.root/'inactive.parquet'),directory/'pilot-ranking.parquet')


if __name__ == '__main__':
    unittest.main()
