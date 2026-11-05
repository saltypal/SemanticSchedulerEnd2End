import unittest
import numpy as np
from experiments.full_medic_labeling.balancing import resample_fit_fold

class BalanceTests(unittest.TestCase):
    def test_oversample_preserves_rows_and_balances(self):
        y=np.repeat(np.arange(4),[20,4,8,3]);ix=resample_fit_fold(y,'random_oversampling')
        self.assertEqual(set(ix),set(range(len(y))))
        self.assertEqual(np.bincount(y[ix]).tolist(),[20]*4)
    def test_undersampling_no_duplicates_or_invented_rows(self):
        y=np.repeat(np.arange(4),[20,4,8,3]);ix=resample_fit_fold(y,'moderate_undersampling')
        self.assertEqual(len(ix),len(set(ix)))
        self.assertTrue(set(ix)<=set(range(len(y))))
        self.assertEqual(set(y[ix]),set(range(4)))
    def test_reproducibility_and_missing_class(self):
        y=np.repeat(np.arange(4),[20,4,8,3])
        np.testing.assert_array_equal(resample_fit_fold(y,'random_oversampling'),resample_fit_fold(y,'random_oversampling'))
        with self.assertRaises(ValueError):resample_fit_fold([0,1,2],'random_oversampling')

if __name__=='__main__':unittest.main()
