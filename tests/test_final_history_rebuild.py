import unittest
from pathlib import Path
import json

from hm_recsys.final_history_rebuild import CONTRACT, end, read


class FrozenHistoryTests(unittest.TestCase):
    def test_date_exclusive_end(self):
        self.assertEqual(end('2019-12-25'), '2020-01-01')

    @unittest.skipUnless(CONTRACT.exists(), 'Local ignored history contract required')
    def test_all_model_selection_and_fit_labels_precede_targets(self):
        for cutoff, item in read(CONTRACT)['mappings'].items():
            self.assertLessEqual(item['origin_cutoff'], cutoff)
            for model in item['models']:
                self.assertLess(end(model['selection_cutoff']), cutoff)
                self.assertTrue(all(end(t) < cutoff for t in model['training_cutoffs']))

    @unittest.skipUnless(CONTRACT.exists(), 'Local ignored history contract required')
    def test_frozen_base_feature_contract(self):
        mappings = read(CONTRACT)['mappings']
        reference = mappings['2020-01-22']['models']
        for row in mappings.values():
            for model, ref in zip(row['models'], reference):
                self.assertEqual(model['features'], ref['features'])
                self.assertEqual(model['parameters'], ref['parameters'])
                self.assertNotIn('target', model['features'])
                self.assertNotIn('truth_count', model['features'])

    @unittest.skipUnless(CONTRACT.exists(), 'Local ignored history contract required')
    def test_population_dates_and_final_week_firewall(self):
        c = read(CONTRACT)
        self.assertEqual(len(c['historical_dates']), 8)
        self.assertEqual(len(c['mappings']), 9)
        self.assertNotIn('2020-09-16', c['mappings'])
        self.assertFalse(c['model_training_allowed'])
        self.assertTrue(all(v['base_exists'] and v['bpr_exists'] for v in c['mappings'].values()))


if __name__ == '__main__':
    unittest.main()
