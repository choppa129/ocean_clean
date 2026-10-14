"""Offline residual learning with year-separated, availability-aware neighbors."""
import numpy as np
import pandas as pd
from .benchmark import mask_blocks

LOCAL = ['WTMP','ATMP','DEWP','WSPD','GST','PRES','WVHT']


def make_features(local, neighbors):
    target = local.WTMP
    linear = target.interpolate(method='time', limit_area='inside')
    x = local.drop(columns=['WTMP']).copy()
    x['linear'] = linear
    x['left'] = target.ffill()
    x['right'] = target.bfill()
    positions = pd.Series(np.arange(len(local)), index=local.index, dtype=float)
    anchors = positions.where(target.notna())
    x['since_left'] = positions - anchors.ffill()
    x['until_right'] = anchors.bfill() - positions
    x['gap_fraction'] = x.since_left / (x.since_left + x.until_right).replace(0,np.nan)
    x['season_sin'] = np.sin(2*np.pi*local.index.dayofyear/365.25)
    x['season_cos'] = np.cos(2*np.pi*local.index.dayofyear/365.25)
    for name in neighbors:
        values = neighbors[name].reindex(local.index)
        x[name] = values
        x[name+'_available'] = values.notna().astype(float)
        # Relative evolution since observed local anchors. All hidden local
        # targets were removed before computing these features.
        at_anchors = values.where(target.notna())
        x[name+'_departure'] = values-at_anchors.interpolate(method='time',limit_area='inside')
    return x


def corrupt(local, neighbors, length, blocks, seed, scenario):
    hidden, mask, count = mask_blocks(local,'WTMP',length,blocks,seed,scenario)
    # Neighbor data stays available in a LOCAL whole-station outage.
    return make_features(hidden,neighbors), mask, count


def training_examples(years, length, scenario):
    xs, ys, times = [], [], []
    counts=[]
    for year,(local,neighbors) in years.items():
        for repeat in range(4):
            x,mask,n = corrupt(local,neighbors,length,150,1000+year+repeat,scenario)
            xs.append(x.loc[mask]);ys.append(local.WTMP.loc[mask]-x.linear.loc[mask])
            times.extend(local.index[mask]);counts.append(n)
    x,y=pd.concat(xs),pd.concat(ys)
    # Repeated corruptions augment context, not independent observations.
    return x,y,{'augmented_rows':len(y),'unique_target_hours':len(set(times)),
                'blocks_per_round':counts}


def score(truth,pred):
    error=np.asarray(pred)-np.asarray(truth)
    if not np.isfinite(error).all():
        raise ValueError('Nonfinite predictions or labels')
    return {'mae':float(np.abs(error).mean()),'rmse':float(np.sqrt((error**2).mean()))}
