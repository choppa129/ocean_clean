"""Chronological offline benchmark. Hold out whole blocks before any features."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def split(frame):
    n = len(frame)
    if n < 100:
        raise ValueError('At least 100 hourly rows required')
    a, b = int(n * .6), int(n * .8)
    return frame.iloc[:a].copy(), frame.iloc[a:b].copy(), frame.iloc[b:].copy()


def mask_blocks(frame, target, length, blocks, seed, scenario):
    if length < 1 or blocks < 1:
        raise ValueError('length and blocks must be positive')
    if scenario not in {'sensor', 'station'}:
        raise ValueError('scenario must be sensor or station')
    rng = np.random.default_rng(seed)
    observed = frame[target].notna().to_numpy()
    mask = np.zeros(len(frame), dtype=bool)
    occupied = mask.copy()
    count = 0
    for start in rng.permutation(max(0, len(frame) - length - 1)) + 1:
        stop = start + length
        # Require observed anchors, and keep separate blocks from sharing anchors.
        if observed[start-1:stop+1].all() and not occupied[start-1:stop+1].any():
            mask[start:stop] = True
            occupied[start-1:stop+1] = True
            count += 1
            if count == blocks:
                break
    if count == 0:
        raise ValueError(f'No eligible {length}-hour block; choose more complete data')
    corrupted = frame.copy()
    if scenario == 'sensor':
        corrupted.loc[mask, target] = np.nan
    else:
        corrupted.loc[mask, :] = np.nan
    return corrupted, mask, count


def features(frame, target):
    x = frame.drop(columns=[target]).copy()
    # Masked target and every other hidden value are excluded before interpolation.
    x['target_linear'] = frame[target].interpolate(method='time', limit_area='inside')
    x['hour_sin'] = np.sin(2 * np.pi * frame.index.hour / 24)
    x['hour_cos'] = np.cos(2 * np.pi * frame.index.hour / 24)
    x['year_sin'] = np.sin(2 * np.pi * frame.index.dayofyear / 365.25)
    x['year_cos'] = np.cos(2 * np.pi * frame.index.dayofyear / 365.25)
    return x


def evaluate(frame, target='WTMP', lengths=(1,6,24,72), blocks=20, seed=42, scenario='sensor'):
    if target not in frame:
        raise ValueError(f'Missing target {target}')
    train, validation, test = split(frame)
    rows, predictions = [], []
    for length in lengths:
        tr, train_mask, _ = mask_blocks(train, target, length, blocks, seed, scenario)
        model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                    min_samples_leaf=5, random_state=seed)
        model.fit(features(tr, target).loc[train_mask], train.loc[train_mask, target])
        for split_name, part in [('validation', validation), ('test', test)]:
            hidden, mask, actual_blocks = mask_blocks(part, target, length, blocks, seed+1, scenario)
            x = features(hidden, target)
            truth = part.loc[mask, target].to_numpy()
            methods = {'linear': x.loc[mask, 'target_linear'].to_numpy(),
                       'train_median': np.full(mask.sum(), train[target].median()),
                       'gradient_boosting': model.predict(x.loc[mask])}
            for name, pred in methods.items():
                error = pred - truth
                rows.append(dict(split=split_name, scenario=scenario, gap_hours=length,
                                 model=name, blocks=actual_blocks, points=len(truth),
                                 mae=float(np.mean(np.abs(error))), rmse=float(np.sqrt(np.mean(error**2)))))
                predictions.append(pd.DataFrame({'time': part.index[mask], 'truth': truth,
                    'prediction': pred, 'model': name, 'gap_hours': length, 'split': split_name,
                    'scenario': scenario}))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)
