"""Frozen-checkpoint diagnostics; no training and no test-year access.

Run from repo root: PYTHONPATH=src python scripts/diagnose_transformer.py
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ocean_clean.data import load_csv, hourly
from ocean_clean.transformer import MaskedWindows, OceanTransformer, training_stats

torch.set_num_threads(2)
root = Path('reports/transformer/checkpoint')
out = Path('reports/transformer_diagnostics'); out.mkdir(parents=True, exist_ok=True)
cfg = json.loads((root/'config.json').read_text())
norm = np.load(root/'normalization.npz'); mean, std = norm['mean'], norm['std']
sources = {}
def load(year):
    index = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', freq='h', inclusive='left', tz='UTC')
    arrays = []
    for station in cfg['stations']:
        path = Path(f'data/raw/{station}h{year}.csv')
        sources[f'{station}/{year}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        arrays.append(hourly(load_csv(path)).reindex(index).reindex(columns=cfg['variables']).to_numpy(dtype='float32'))
    return np.stack(arrays, axis=1)
train = [load(y) for y in cfg['train_years']]
a = load(2024)
check_mean, check_std = training_stats(train)
assert np.allclose(mean, check_mean) and np.allclose(std, check_std)
assert all(sources[k] == cfg['source_csv_sha256'][k] for k in sources)
ds = MaskedWindows([a], mean, std, samples=cfg['validation_samples'], length=cfg['window_hours'], seed=cfg['validation_seed'])
v, m, truth, selected = [torch.stack(x) for x in zip(*ds)]
y = truth.numpy()*std[0,0]+mean[0,0]; mask = selected.numpy()
assert (m[:,:,0,0][selected] == 0).all() and (v[:,:,0,0][selected] == 0).all()
model = OceanTransformer(len(cfg['variables']), length=cfg['window_hours'], width=cfg['width'], layers=cfg['layers'])
model.load_state_dict(torch.load(root/'best.pt', map_location='cpu', weights_only=True)); model.eval()
def predict(values, observed):
    with torch.no_grad():
        return torch.cat([model(values[i:i+16], observed[i:i+16]) for i in range(0,len(values),16)]).numpy()*std[0,0]+mean[0,0]
preds = {'original': predict(v,m)}
for variant in ['neighbor_removed','neighbor_wtmp_removed','neighbor_time_shuffled','neighbor_window_swapped','neighbor_wtmp_plus1','neighbor_wtmp_minus1','target_context_plus1','target_context_minus1','target_wtmp_removed','target_observed_time_shuffled']:
    vv, mm = v.clone(), m.clone()
    if variant == 'neighbor_removed': vv[:,:,1,:]=0; mm[:,:,1,:]=0
    elif variant == 'neighbor_wtmp_removed': vv[:,:,1,0]=0; mm[:,:,1,0]=0
    elif variant == 'neighbor_time_shuffled':
        rng=np.random.default_rng(2026)
        for i in range(len(v)):
            order=rng.permutation(v.shape[1]); vv[i,:,1,:]=v[i,order,1,:]; mm[i,:,1,:]=m[i,order,1,:]
    elif variant == 'neighbor_window_swapped':
        # Cyclic permutation has no fixed points; deliberately destroys pairing.
        vv[:,:,1,:]=v.roll(1,0)[:,:,1,:]; mm[:,:,1,:]=m.roll(1,0)[:,:,1,:]
    elif variant == 'target_observed_time_shuffled':
        rng=np.random.default_rng(2026)
        for i in range(len(v)):
            ids=np.flatnonzero(m[i,:,0,0].numpy()); vv[i,ids,0,0]=v[i,rng.permutation(ids),0,0]
    elif variant == 'target_wtmp_removed': vv[:,:,0,0]=0; mm[:,:,0,0]=0
    else:
        station=1 if variant.startswith('neighbor') else 0
        sign=1 if variant.endswith('plus1') else -1
        vv[:,:,station,0] += sign*mm[:,:,station,0]/std[station,0]
    assert torch.equal(mm[:,:,0,0][selected], m[:,:,0,0][selected])
    assert torch.equal(vv[:,:,0,0][selected], v[:,:,0,0][selected])
    preds[variant]=predict(vv,mm)
rows=[]
for variant,pred in preds.items():
    for i,(_,start,offset,gap) in enumerate(ds.examples):
        use=mask[i]; err=pred[i,use]-y[i,use]; change=pred[i,use]-preds['original'][i,use]
        rows.append(dict(variant=variant,window_id=i,gap_hours=gap,month=int((pd.Timestamp('2024-01-01')+pd.Timedelta(hours=start+offset)).month),mae=float(abs(err).mean()),bias=float(err.mean()),mse=float((err**2).mean()),prediction_change=float(change.mean()),absolute_prediction_change=float(abs(change).mean()),truth_mean=float(y[i,use].mean()),pred_mean=float(pred[i,use].mean())))
f=pd.DataFrame(rows); f.to_csv(out/'window_metrics.csv',index=False)
summary=f.groupby(['variant','gap_hours'])[['mae','bias','mse','prediction_change','absolute_prediction_change']].mean().reset_index();summary['rmse']=np.sqrt(summary.pop('mse'));summary.to_csv(out/'summary.csv',index=False)
base=f[f.variant=='original'];base.groupby('month')[['mae','bias']].agg(['mean','count']).to_csv(out/'monthly_bias.csv')
example_id=json.loads(Path('reports/paired_comparison/protocol.json').read_text())['example_window_id']
example=f[f.window_id==example_id]; example.to_csv(out/'example_metrics.csv',index=False)
_,start,offset,gap=ds.examples[example_id]
ex=pd.DataFrame({'hour':np.arange(cfg['window_hours']),'hidden':mask[example_id], 'target_wtmp':a[start:start+cfg['window_hours'],0,0], 'neighbor_wtmp':a[start:start+cfg['window_hours'],1,0], **{k:p[example_id] for k,p in preds.items()}});ex.to_csv(out/'example_predictions.csv',index=False)
# Same sampled training windows as fitting (first 256); eval mode for comparable errors.
training_ds=MaskedWindows(train,mean,std,samples=256,length=cfg['window_hours'],seed=cfg['seed'])
tv,tm,ty,ts=[torch.stack(x) for x in zip(*training_ds)]
tp=predict(tv,tm);ty=ty.numpy()*std[0,0]+mean[0,0]
train_rows=[]
for i in range(len(training_ds)):
    e=(tp[i]-ty[i])[ts[i].numpy()];train_rows.append({'gap_hours':int(ts[i].sum()),'mae':float(abs(e).mean()),'bias':float(e.mean())})
pd.DataFrame(train_rows).groupby('gap_hours').mean().to_csv(out/'training_fit.csv')
protocol={'checkpoint_sha256':hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),'source_sha256':sources,'normalization_matches_training':True,'masked_target_hidden':True,'validation_windows':len(ds),'validation_seed':cfg['validation_seed'],'shuffle_seed':2026,'example_window_id':example_id,'no_retraining':True,'test_year_read':False,'limitations':['2024 validation already used for checkpoint selection','Windows overlap; do not interpret as independent samples','Input perturbations may be out of distribution; not a retrained single-station comparison','Gap-specific means weight each equal-length window equally; macro averages weight gaps equally']}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
print(summary.to_string(index=False));print('\nEXAMPLE\n',example[['variant','mae','bias','prediction_change']].to_string(index=False));print('\nTRAIN\n',pd.DataFrame(train_rows).groupby('gap_hours').mean());print('\nMONTH\n',base.groupby('month')[['mae','bias']].mean())

availability=[]
for i,(_,start,offset,gap) in enumerate(ds.examples):
    for station,name in enumerate(cfg['stations']):
        for j,var in enumerate(cfg['variables']):
            availability.append({'window_id':i,'station':name,'variable':var,'observed_fraction':float(m[i,:,station,j].mean())})
pd.DataFrame(availability).to_csv(out/'availability.csv',index=False)
print('EXAMPLE AVAILABILITY',pd.DataFrame(availability).query('window_id == @example_id').to_string(index=False))
print('EXAMPLE TEMPERATURES',ex.loc[ex.hidden,['target_wtmp','neighbor_wtmp','original']].mean().to_dict())
print('TRAIN MEAN',mean.tolist())
# Isolate windows with at least 90% observed neighbor water temperatures.
av = pd.DataFrame(availability)
eligible = av[(av.station == cfg['stations'][1]) & (av.variable == 'WTMP') & (av.observed_fraction >= .9)].window_id
subset = f[f.window_id.isin(eligible)].groupby('variant')[['mae','bias','absolute_prediction_change']].mean()
subset.to_csv(out/'neighbor_available_subset.csv')
macro = summary.groupby('variant')[['mae','bias','absolute_prediction_change']].mean()
macro.to_csv(out/'macro_summary.csv')
def table(frame):
    frame=frame.reset_index()
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'
report = '''# Frozen Transformer: bias and neighbor-use diagnosis

Run `PYTHONPATH=src python scripts/diagnose_transformer.py` from the repository root
with the deep-learning dependencies installed and the six 2022–2024 source CSVs available.
No retraining, checkpoint changes, or 2025 test access. All errors are in degrees Celsius.

## Findings

1. **The illustrated outage has no neighbor observations at all.** In the entire
128-hour example window, all four channels at station 41010 are missing. Removing
or perturbing observed neighbor temperatures therefore has exactly zero effect.
The hidden target mean is 21.9354°C and the model mean is 22.4212°C (bias +0.4858°C).
This case cannot demonstrate benefit from real neighbor measurements. The network
still receives the neighbor's learned embedding and missing-value mask.
2. **Time ordering is weakly used in the present checkpoint.** Shuffling observed
local water temperatures among their observed timestamps (holding the gap and masks
fixed) changes 24-hour predictions by only 0.0419°C on average, with MAE 0.7095°C
versus 0.7209°C originally. This suggests reliance on broad window-level conditions
rather than precise local evolution. It is evidence from an intervention, not proof
of one unique training failure. In the example this shuffle reduces bias to +0.3633°C.
3. **Neighbor values influence the network, but a net accuracy benefit is not established.**
Raising observed neighbor water temperatures by 1°C raises 24-hour predictions by
0.0778°C on average; raising local observed water temperatures by 1°C raises them
by 0.6183°C. Swapping complete neighbor windows worsens errors, but removing the
neighbor improves errors on these validation windows. Using an input does not mean
using it optimally. These are frozen-input interventions; they do not establish how
well a separately trained single-station model would perform.
4. **Bias varies by season rather than being one constant offset.** Mean window
bias in February is +0.9479°C; May is -0.4954°C and December -0.5380°C. Training-window
MAE is lower than validation MAE, so generalization is also a concern. The training
subset is the first 256 originally sampled fitting windows, not an unseen split.
5. **Basic data-path checks passed.** Source hashes match the training manifest;
normalization is exactly reproduced from 2022–2023; masked target values and their
observation flags remain zero in every intervention. There is no evidence here of
a normalization mismatch. This is not an exhaustive software audit.

## All 256 validation windows: macro averages over the four gap lengths

'''+table(macro)+'''
## Windows with at least 90% observed neighbor water temperature

'''+f'{len(eligible)} of {len(ds)} windows qualify. This table averages windows; the gap mix may differ.\n\n'+table(subset)+'''
## Training fit, evaluated with dropout disabled

'''+table(pd.DataFrame(train_rows).groupby('gap_hours').mean())+'''
## Likely contributors and next controlled experiment

The current network predicts absolute standardized temperature without a local
endpoint anchor; its learned positional embedding and temporal attention have not
produced strong sensitivity to local temperature ordering. The loss averages over
hidden points, so a 72-hour gap contributes far more labels than a 1-hour gap.
Masks are fixed across epochs, and training lasts 15 epochs. These are design facts
and plausible contributors, not experimentally isolated causes of the warm bias.
A missing station still produces learned tokens and participates in spatial attention.

Next, compare models trained with and without neighbor inputs under the same seeds
and masking schedule; explicitly gate unavailable stations. Test window-relative
normalization and balanced per-window loss one at a time. Use validation for these
choices. Do not claim improvement or inspect the test year during diagnosis.

Windows overlap, so these are descriptive paired results without independence-based
significance claims. Perturbations can create inputs unlike training data. 2024 was
already used for checkpoint selection. Detailed per-gap, per-window, monthly,
availability and example outputs are included next to this report.
'''
(out/'README.md').write_text(report)
print('NEIGHBOR AVAILABLE',len(eligible));print(subset.to_string())
