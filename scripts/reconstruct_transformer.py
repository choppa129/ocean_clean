"""Reconstruct genuinely missing target values in one 128-hour CSV window."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ocean_clean.data import load_csv,hourly
from ocean_clean.transformer import OceanTransformer
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True,help='Directory with best.pt, config.json and normalization.npz')
p.add_argument('--target-csv',required=True)
p.add_argument('--neighbor-csv',required=True)
p.add_argument('--start',required=True,help='UTC window start, e.g. 2024-01-01T00:00:00Z')
p.add_argument('--out',default='outputs/reconstruction.csv')
a=p.parse_args();folder=Path(a.checkpoint);config=json.loads((folder/'config.json').read_text())
norm=np.load(folder/'normalization.npz');mean,std=norm['mean'],norm['std']
index=pd.date_range(pd.to_datetime(a.start,utc=True),periods=config['window_hours'],freq='h')
frames=[hourly(load_csv(path)).reindex(index).reindex(columns=config['variables']) for path in [a.target_csv,a.neighbor_csv]]
values=np.stack([f.to_numpy(dtype='float32') for f in frames],axis=1)
observed=np.isfinite(values)
x=np.where(observed,(values-mean)/std,0).astype('float32')
model=OceanTransformer(len(config['variables']),stations=len(config['stations']),length=config['window_hours'],width=config['width'],layers=config['layers'])
model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True));model.eval()
with torch.no_grad():pred=model(torch.from_numpy(x[None]),torch.from_numpy(observed[None].astype('float32'))).numpy()[0]*std[0,0]+mean[0,0]
original=values[:,0,0]
result=pd.DataFrame({'time':index,'observed_WTMP':original,'reconstructed_WTMP':np.where(np.isfinite(original),original,pred),'was_missing':~np.isfinite(original)})
Path(a.out).parent.mkdir(parents=True,exist_ok=True);result.to_csv(a.out,index=False)
print(f'Wrote {a.out}; reconstructed {int((~np.isfinite(original)).sum())} missing hours. No ground-truth accuracy available for real gaps.')
