"""Read-only qC schema compatibility, no research fits or source modification."""
import unittest
import numpy as np
import pandas as pd
from hm_recsys.p43a_data import attach_frozen_cold_rank


class ColdSchemaTests(unittest.TestCase):
    def setUp(self):
        self.features=pd.DataFrame(dict(customer_id=['u']*50,article_id=[str(i) for i in range(50)],
            normalized=np.arange(50,dtype=float)/49,available=np.ones(50,dtype=np.uint8)))
        self.selection=self.features[['customer_id','article_id']].copy()
        self.selection['b0_rank']=np.arange(1,51)
        self.spec=dict(numeric=['normalized'],binary=['available'])

    def test_preserves_inputs_and_recovers_rank_by_identity_not_row_position(self):
        out=attach_frozen_cold_rank(self.features,self.selection.iloc[::-1],self.spec)
        pd.testing.assert_frame_equal(out[self.features.columns],self.features)
        np.testing.assert_array_equal(out.b0_rank,np.arange(1,51))

    def test_rejects_missing_feature_population_duplicate_and_bad_rank(self):
        with self.assertRaises(ValueError):
            attach_frozen_cold_rank(self.features.drop(columns='normalized'),self.selection,self.spec)
        with self.assertRaises(ValueError):
            attach_frozen_cold_rank(self.features.iloc[:-1],self.selection,self.spec)
        with self.assertRaises(ValueError):
            attach_frozen_cold_rank(self.features,pd.concat([self.selection,self.selection.iloc[:1]]),self.spec)
        bad=self.selection.copy();bad.loc[0,'b0_rank']=2
        with self.assertRaises(ValueError):
            attach_frozen_cold_rank(self.features,bad,self.spec)


if __name__=='__main__':
    unittest.main()
