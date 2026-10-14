import unittest
import numpy as np
import pandas as pd
from ocean_clean.data import parse_ndbc, hourly, quality_report
from ocean_clean.benchmark import split, mask_blocks, features, evaluate


class PipelineTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(2000)
        self.frame = pd.DataFrame({'WTMP': 20 + np.sin(t/24), 'ATMP': 18 + np.sin(t/24)},
            index=pd.date_range('2020-01-01', periods=len(t), freq='h', tz='UTC', name='time'))

    def test_ndbc_sentinels_and_minutes(self):
        f = parse_ndbc('#YY MM DD hh mm WDIR WSPD WTMP\n#yr mo dy hr mn deg m/s degC\n2024 01 01 00 00 99 99.0 999.0\n2024 01 01 00 30 180 3.0 20.0\n')
        self.assertEqual(f.WDIR.iloc[0], 99)
        self.assertTrue(np.isnan(f.WSPD.iloc[0]))
        self.assertTrue(np.isnan(f.WTMP.iloc[0]))
        self.assertEqual(hourly(f).WTMP.iloc[0], 20)

    def test_mask_reproducibility_and_no_target_leakage(self):
        hidden, mask, n = mask_blocks(self.frame, 'WTMP', 24, 5, 3, 'sensor')
        again, m2, _ = mask_blocks(self.frame, 'WTMP', 24, 5, 3, 'sensor')
        np.testing.assert_array_equal(mask, m2)
        self.assertEqual(mask.sum(), 24*n)
        self.assertTrue(hidden.loc[mask,'WTMP'].isna().all())
        changed = self.frame.copy()
        changed.loc[mask,'WTMP'] = 9999
        changed.loc[mask,'WTMP'] = np.nan
        pd.testing.assert_frame_equal(features(hidden,'WTMP'), features(changed,'WTMP'))
        station, sm, _ = mask_blocks(self.frame,'WTMP',24,5,3,'station')
        self.assertTrue(station.loc[sm].isna().all().all())

    def test_chronological_split_and_metrics(self):
        a,b,c = split(self.frame)
        self.assertLess(a.index.max(), b.index.min())
        self.assertLess(b.index.max(), c.index.min())
        metrics, pred = evaluate(self.frame, lengths=(6,), blocks=4)
        self.assertEqual(len(metrics), 6)
        self.assertTrue(np.isfinite(metrics[['mae','rmse']].values).all())
        self.assertEqual(set(pred.split), {'validation','test'})

    def test_quality_missing_hours(self):
        f = self.frame.drop(self.frame.index[3:7])
        self.assertEqual(quality_report(hourly(f))['variables']['WTMP']['longest_missing_hours'],4)


if __name__ == '__main__':
    unittest.main()
