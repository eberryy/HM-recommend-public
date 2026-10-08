import unittest
import numpy as np
import pandas as pd
from hm_recsys.warm_v3_user_router import FEATURES,fit_ridge,fold,predict


class UserRouterTests(unittest.TestCase):
    def test_ridge_is_finite_and_deterministic(self):
        frame=pd.DataFrame({name:np.arange(20,dtype=float)+(i%3) for i,name in enumerate(FEATURES)})
        target=np.linspace(-.1,.1,20);model=fit_ridge(frame,target);a=predict(frame,model);b=predict(frame,model)
        self.assertTrue(np.isfinite(a).all());np.testing.assert_array_equal(a,b)

    def test_customer_fold_is_stable_and_binary(self):
        self.assertEqual(fold('abc'),fold('abc'));self.assertIn(fold('abc'),(0,1))


if __name__=='__main__':unittest.main()
