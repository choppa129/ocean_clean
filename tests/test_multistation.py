import unittest
import numpy as np
import pandas as pd
from ocean_clean.multistation import corrupt,make_features,training_examples

class MultiStationTests(unittest.TestCase):
    def setUp(self):
        self.idx=pd.date_range('2022-01-01',periods=1000,freq='h',tz='UTC')
        self.local=pd.DataFrame({'WTMP':np.sin(np.arange(1000)/24)+20,'ATMP':22.},index=self.idx)
        self.neighbors=pd.DataFrame({'n41010_WTMP':19.},index=self.idx)

    def test_station_outage_preserves_neighbor_but_hides_local(self):
        x,mask,n=corrupt(self.local,self.neighbors,24,5,42,'station')
        self.assertTrue(x.loc[mask,'ATMP'].isna().all())
        self.assertTrue(x.loc[mask,'n41010_WTMP'].notna().all())
        self.assertEqual(mask.sum(),24*n)

    def test_hidden_labels_cannot_change_features(self):
        x,mask,_=corrupt(self.local,self.neighbors,6,5,42,'sensor')
        modified=self.local.copy()
        modified.loc[mask,'WTMP']=1000
        modified.loc[mask,'WTMP']=np.nan
        pd.testing.assert_frame_equal(x,make_features(modified,self.neighbors))

    def test_no_future_neighbor_fill(self):
        neighbor=self.neighbors.copy()
        neighbor.iloc[100:200]=np.nan
        x=make_features(self.local,neighbor)
        self.assertTrue(x.iloc[100:200].n41010_WTMP.isna().all())
        self.assertEqual(x.iloc[100:200].n41010_WTMP_available.sum(),0)

    def test_training_examples_use_only_training_year(self):
        x,y,counts=training_examples({2022:(self.local,self.neighbors)},1,'sensor')
        self.assertTrue((x.index.year==2022).all())
        self.assertTrue((y.index.year==2022).all())
        self.assertGreater(counts['augmented_rows'],20)
        self.assertLessEqual(counts['unique_target_hours'],counts['augmented_rows'])
