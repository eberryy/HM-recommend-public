import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd

from hm_recsys.m2 import _prepare_frame
from hm_recsys.warm_v2_contract import guard
from hm_recsys.warm_v2_engine import Engine, prepare, save_parquet
from hm_recsys.warm_v2_features import SOURCES, SPEC, _base_views, _item, _rhythm, _affinity, columns_for, source_features
from hm_recsys.warm_v2_extra_features import EXTRA_SPEC, dimensions, category, price, hierarchy
from hm_recsys.warm_v2_candidate_context import SIGNALS, compute


class WarmV2Tests(unittest.TestCase):
    def test_polish_only_preregistered_capacity_values(self):
        with patch('hm_recsys.warm_v2_engine.read',return_value={'inputs':{'transactions':{'path':'unused'}}}):
            c={'frozen_feature_columns':[]}
            self.assertEqual(Engine(c).parameter_overrides,{})
            for leaves in (15,63):
                self.assertEqual(Engine(c,{'num_leaves':leaves}).parameter_overrides,{'num_leaves':leaves})
            for invalid in ({'num_leaves':127},{'learning_rate':.1}):
                with self.assertRaises(AssertionError):
                    Engine(c,invalid)

    def test_candidate_context_full_pool_ties_and_label_invariance(self):
        frame=pd.DataFrame({'customer_id':['u','u','u','v'],'article_id':['01','02','03','04'],'target':[0,1,0,1]})
        for signal in SIGNALS:
            frame[signal]=[10,5,0,0]
        with duckdb.connect() as con:
            con.register('pool',frame)
            first=compute(con,'pool')
            frame['target']=1-frame.target
            con.unregister('pool');con.register('pool',frame)
            second=compute(con,'pool')
        pd.testing.assert_frame_equal(first,second,check_exact=True)
        self.assertEqual(first.wv2_context_item_events_7d_rank_fraction.iloc[1],.5)
        self.assertAlmostEqual(first.wv2_context_item_events_7d_strength_share.iloc[0],2/3)
        self.assertTrue(np.isnan(first.wv2_context_item_events_7d_rank_fraction.iloc[3]))
        self.assertEqual(first.wv2_context_item_events_7d_strength_share.iloc[3],0)

    def test_preparation_exact_value_dtype_order(self):
        frame = pd.DataFrame({'article_product_type_no':[1,2,np.nan,10], 'item_events_7d':[0,1,np.nan,7], 'candidate_rank':[1,2,3,4]})
        features=['candidate_rank','article_product_type_no','item_events_7d']
        maps={'article_product_type_no':{1:2,2:1}}
        pd.testing.assert_frame_equal(prepare(frame,features,maps),_prepare_frame(frame,features,maps),check_exact=True)

    def test_final_week_and_overlap_rejected(self):
        for cutoff in ('2020-09-16','2020-09-17','2020-09-10'):
            with self.assertRaises(ValueError):
                guard(cutoff)
        self.assertEqual(guard('2020-08-19'),'2020-08-19')

    def test_sources_missing_not_artificial_agreement(self):
        f=pd.DataFrame({'repurchase_present':[0,1],'repurchase_rank':[np.nan,2]})
        for source in SOURCES[1:]:
            f[source+'_present']=[0,0]
            f[source+'_rank']=[np.nan,np.nan]
        f['item2vec_present']=[1,1]; f['item2vec_cosine']=[.2,.3]; f['source_count']=[0,1]
        out=source_features(f)
        self.assertTrue(np.isnan(out['wv2_source_best_rank'][0]))
        self.assertTrue(np.isnan(out['wv2_source_second_rank'][1]))
        self.assertEqual(out['wv2_source_top10_support'][0],0)
        self.assertEqual(out['wv2_source_top10_support'][1],1)
        self.assertEqual(out['wv2_source_best_strength_share'][1],1)

    def test_specs_unique_no_window_router_or_label(self):
        names=columns_for(list(SPEC))
        self.assertEqual(len(names),len(set(names)))
        self.assertFalse(any('target' in s or 'outer_window' in s or 'season' in s for s in names))
        self.assertEqual(len(names),65)

    def test_raw_history_future_excluded_and_daily_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            raw=pd.DataFrame({'customer_id':['u']*5,'article_id':['01']*5,
                't_dat':[date(2020,2,2),date(2020,2,8),date(2020,2,9),date(2020,2,10),date(2020,2,11)],
                'price':[1.,2.,3.,100.,1000.],'sales_channel_id':[1,2,2,1,1]})
            base=pd.DataFrame({'customer_id':['u'],'article_id':['01']})
            save_parquet(raw,root/'transactions.parquet'); save_parquet(base,root/'base.parquet')
            engine=SimpleNamespace(transactions=str(root/'transactions.parquet'),base_path=lambda cutoff:root/'base.parquet')
            with duckdb.connect() as con:
                _base_views(con,'2020-02-10',engine)
                item=_item(con,'2020-02-10').iloc[0]
                rhythm=_rhythm(con,'2020-02-10').iloc[0]
            self.assertEqual(item.wv2_item_events_all,3)
            self.assertEqual(item.wv2_item_events_1d,1)
            self.assertEqual(item.wv2_item_events_3d,2)
            self.assertEqual(item.wv2_item_observed_age,8)
            self.assertEqual(item.wv2_item_peak_age_28d,1)
            self.assertEqual(rhythm.wv2_user_price_median_28d,2)
            self.assertEqual(rhythm.wv2_user_gap_median_84d,3.5)
            self.assertEqual(rhythm.wv2_user_last_gap_84d,1)

    def test_affinity_cutoff_raw_duplicates_and_article_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cutoff='2020-02-10'
            raw=pd.DataFrame({'customer_id':['u']*4,'article_id':['01','01','02','02'],
                't_dat':[date(2020,2,8),date(2020,2,8),date(2020,2,10),date(2020,2,11)]})
            base=pd.DataFrame({'customer_id':['u','u'],'article_id':['01','02'],'candidate_rank':[1,2],
                'user_item_events_12w':[2,0],'user_item_days_since_last_purchase':[2,np.nan],
                'user_product_code_events_12w':[2,0],'user_product_code_days_since':[2,np.nan],
                'user_product_type_events_12w':[2,0],'item_events_7d':[2,0],'item2vec_cosine':[.5,.2]})
            dims=pd.DataFrame({'article_id':['01','02'],'product_code':[10,20],'product_type_no':[1,2],
                'department_no':[3,4],'garment_group_no':[5,6],'perceived_colour_master_id':[7,8]})
            dims.to_csv(root/'articles.csv',index=False)
            save_parquet(raw,root/'transactions.parquet');save_parquet(base,root/'base.parquet')
            engine=SimpleNamespace(transactions=str(root/'transactions.parquet'),base_path=lambda c:root/'base.parquet',
                history={'prerequisite_cache':{'target_features':{cutoff:{'inputs':{'articles':{'path':str(root/'articles.csv')}}}}}})
            with duckdb.connect() as con:
                _base_views(con,cutoff,engine)
                result=_affinity(con,cutoff,engine)
            self.assertEqual(result.article_id.tolist(),['01','02'])
            self.assertEqual(result.wv2_user_item_events_7d.tolist(),[2,0])
            self.assertEqual(result.wv2_user_item_days_84d.tolist(),[1,0])
            self.assertEqual(result.wv2_same_last_type.tolist(),[1,0])
            self.assertEqual(result.wv2_type_recent_share_7d.tolist(),[1,0])

    def test_second_batch_pit_and_missing_price(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cutoff='2020-02-10'
            raw=pd.DataFrame({'customer_id':['u']*4,'article_id':['01','01','02','03'],
                't_dat':[date(2020,2,8),date(2020,2,9),date(2020,2,9),date(2020,2,10)],'price':[1.,3.,4.,1000.]})
            base=pd.DataFrame({'customer_id':['u']*3,'article_id':['01','02','03'],'candidate_rank':[1,2,3]})
            dims=pd.DataFrame({'article_id':['01','02','03'],'product_type_no':[1,1,2],'department_no':[1,1,2],
                'garment_group_no':[1,1,2],'section_no':[1,1,2],'colour_group_code':[1,2,3],
                'graphical_appearance_no':[1,2,3],'index_code':['A','A','B']})
            dims.to_csv(root/'articles.csv',index=False)
            save_parquet(raw,root/'transactions.parquet');save_parquet(base,root/'base.parquet')
            engine=SimpleNamespace(transactions=str(root/'transactions.parquet'),base_path=lambda c:root/'base.parquet',
                history={'prerequisite_cache':{'target_features':{cutoff:{'inputs':{'articles':{'path':str(root/'articles.csv')}}}}}})
            with duckdb.connect() as con:
                _base_views(con,cutoff,engine);dimensions(con,cutoff,engine)
                cat=category(con,cutoff).set_index('article_id')
                pri=price(con,cutoff).set_index('article_id')
                hier=hierarchy(con,cutoff).set_index('article_id')
            self.assertAlmostEqual(cat.loc['01','wv2_type_item_share_7d'],2/3)
            self.assertEqual(cat.loc['01','wv2_type_item_rank_fraction_7d'],0)
            self.assertEqual(cat.loc['02','wv2_type_item_rank_fraction_7d'],1)
            self.assertEqual(cat.loc['03','wv2_type_item_share_7d'],0)
            self.assertEqual(pri.loc['01','wv2_price_item_median_28d'],2)
            self.assertEqual(pri.loc['01','wv2_price_user_type_median_28d'],3)
            self.assertTrue(np.isnan(pri.loc['03','wv2_price_item_user_type_standard_gap']))
            self.assertEqual(hier.loc['01','wv2_fine_section_events_28d'],3)
            self.assertEqual(hier.loc['03','wv2_fine_section_events_28d'],0)
            self.assertTrue(np.isnan(hier.loc['03','wv2_fine_section_days_since']))
            for family,frame in [('category_competition',cat),('price_context',pri),('fine_hierarchy',hier)]:
                self.assertTrue(set(columns_for([family]))<=set(frame.columns))
            self.assertEqual(len(columns_for(list(EXTRA_SPEC))),43)


if __name__=='__main__':
    unittest.main()
