"""Paired comparison on original validation windows; illustrative case chosen by index."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ocean_clean.data import load_csv,hourly
from ocean_clean.transformer import MaskedWindows,OceanTransformer

torch.set_num_threads(2)
root=Path('reports/transformer/checkpoint');out=Path('reports/paired_comparison');out.mkdir(parents=True,exist_ok=True)
config=json.loads((root/'config.json').read_text());norm=np.load(root/'normalization.npz');mean,std=norm['mean'],norm['std']
index=pd.date_range('2024-01-01','2025-01-01',freq='h',inclusive='left',tz='UTC')
a=np.stack([hourly(load_csv(f'data/raw/{s}h2024.csv')).reindex(index).reindex(columns=config['variables']).to_numpy(dtype='float32') for s in config['stations']],axis=1)
ds=MaskedWindows([a],mean,std,samples=config['validation_samples'],seed=config['validation_seed'])
model=OceanTransformer(len(config['variables']),width=config['width'],layers=config['layers'])
model.load_state_dict(torch.load(root/'best.pt',map_location='cpu',weights_only=True));model.eval()
rows=[];point_rows=[];example=None;excluded=0
with torch.no_grad():
 for i,(values,observed,truth,mask) in enumerate(ds):
  year,start,offset,gap=ds.examples[i]
  times=index[start:start+config['window_hours']]
  actual=a[start:start+config['window_hours'],0,0].copy()
  hidden=actual.copy();hidden[mask.numpy()]=np.nan
  # Both methods receive the exact same target window and artificial gap.
  baseline=pd.Series(hidden,index=times).interpolate(method='time',limit_area='inside').to_numpy()
  pred=model(values[None],observed[None]).numpy()[0]*std[0,0]+mean[0,0]
  chosen=mask.numpy()
  if not np.isfinite(baseline[chosen]).all():excluded+=1;continue
  for name,estimate in [('transformer',pred),('linear',baseline)]:
   error=estimate[chosen]-actual[chosen]
   rows.append({'window_id':i,'gap_hours':gap,'model':name,'mae':float(np.abs(error).mean()),'rmse':float(np.sqrt((error**2).mean()))})
   point_rows.extend({'window_id':i,'time':str(t),'gap_hours':gap,'model':name,'truth':float(y),'prediction':float(p)} for t,y,p in zip(times[chosen],actual[chosen],estimate[chosen]))
  # First index-ordered 24-hour gap with visible immediate endpoints; not best performance.
  if example is None and gap==24 and np.isfinite(actual[offset-1]) and np.isfinite(actual[offset+gap]):
   example=pd.DataFrame({'time':times,'truth':actual,'observed':hidden,'is_hidden':chosen,'transformer':pred,'linear':baseline})
   example_id=i
metrics=pd.DataFrame(rows);points=pd.DataFrame(point_rows)
metrics.to_csv(out/'window_metrics.csv',index=False)
summary=points.assign(abs_error=lambda x:abs(x.prediction-x.truth),sq_error=lambda x:(x.prediction-x.truth)**2).groupby(['gap_hours','model']).agg(mae=('abs_error','mean'),mse=('sq_error','mean'),points=('truth','size')).reset_index()
summary['rmse']=np.sqrt(summary.pop('mse'));summary.to_csv(out/'summary.csv',index=False)
example.to_csv(out/'example.csv',index=False)
(out/'protocol.json').write_text(json.dumps({'year':2024,'split':'validation (already used for checkpoint selection)',
 'samples_requested':len(ds),'windows_used':len(metrics)//2,'excluded_no_interpolation_support':excluded,
 'example_window_id':example_id,'example_selection':'first index-ordered 24h window with finite immediate endpoints',
 'checkpoint_sha256':hashlib.sha256((root/'best.pt').read_bytes()).hexdigest(),
 'context_hours':128,'mask_seed':config['validation_seed'],
 'comparison':'same hidden labels and target context; Transformer additionally uses weather and neighbor channels',
 'no_retraining':True},indent=2))
plt.style.use('dark_background');fig,(ax,bx)=plt.subplots(2,1,figsize=(12,7),gridspec_kw={'height_ratios':[2,1]},sharex=True)
fig.patch.set_facecolor('#081722')
for panel in [ax,bx]:panel.set_facecolor('#081722');panel.grid(alpha=.12);panel.spines[['top','right']].set_visible(False)
h=example.is_hidden;t=pd.to_datetime(example.time)
for panel in [ax,bx]:panel.axvspan(t[h].iloc[0],t[h].iloc[-1]+pd.Timedelta(hours=1),color='#ffba70',alpha=.1)
ax.plot(t,example.truth,color='#3dd6c6',linewidth=2,label='Actual water temperature (hidden answer inside shaded area)')
ax.plot(t[h],example.loc[h,'transformer'],color='#ffba70',linewidth=2,label='Transformer prediction')
ax.plot(t[h],example.loc[h,'linear'],color='#a7b4d4',linestyle='--',linewidth=2,label='Linear baseline')
ax.set_ylabel('Water temperature (°C)');ax.legend(loc='upper left',fontsize=8)
for name,color in [('transformer','#ffba70'),('linear','#a7b4d4')]:bx.plot(t[h],abs(example.loc[h,name]-example.loc[h,'truth']),label=name,color=color)
bx.set_ylabel('Absolute error (°C)');bx.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M',tz=t.dt.tz));bx.set_xlabel('UTC · 2024 validation data')
fig.suptitle('What did the model guess during a 24-hour outage?',fontsize=17,x=.08,ha='left')
fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'example.svg');fig.savefig(out/'example.png',dpi=140)
selected=metrics[metrics.window_id==example_id]
print('EXAMPLE',example_id,str(t[h].iloc[0]),str(t[h].iloc[-1]));print(selected.to_string(index=False));print(summary.to_string(index=False))
def markdown_table(frame):
 return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---'] * len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)) + '\n'

text='''# A paired, visible reconstruction example

![One 24-hour gap](example.svg)

The shaded region contains 24 hourly water-temperature measurements deliberately
hidden from both methods. Green is the actual answer; orange is the Transformer;
the dashed line is interpolation. The lower panel shows absolute error.
The example is the first eligible 24-hour window in the fixed validation sequence,
not a case chosen by model performance.

## Example errors

'''+markdown_table(selected)+'''
## Paired comparison across the original validation windows

Both methods use the same 128-hour target context and score exactly the same
hidden values. The Transformer also sees available weather and neighbor channels;
the linear baseline only uses target water temperature. No retraining occurred.
These are 2024 validation data already used to choose the checkpoint, not a new
independent test. Overlapping windows are not independent samples.

'''+markdown_table(summary)+'''

The current Transformer is less accurate than the simple baseline in this paired
comparison. This confirms an implementation that runs, not a performance advance.
The next modeling decision should be based on observed error behavior and validation
experiments; do not claim that model complexity implies better reconstruction.

Reproduce: `python scripts/compare_transformer.py` (install the deep and dashboard
extras, and download both 2024 source CSVs first).
'''
(out/'README.md').write_text(text)
