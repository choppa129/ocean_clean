"""Train on 2022–23, select by 2024 validation. Never read the 2025 test."""
import argparse,json,random,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from ocean_clean.data import load_csv,hourly
from ocean_clean.transformer import MaskedWindows,OceanTransformer,training_stats

p=argparse.ArgumentParser()
p.add_argument('--epochs',type=int,default=3)
p.add_argument('--samples',type=int,default=512)
p.add_argument('--validation-samples',type=int,default=128)
p.add_argument('--batch-size',type=int,default=16)
p.add_argument('--width',type=int,default=32)
p.add_argument('--layers',type=int,default=2)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--out',default='outputs/transformer_pilot')
a=p.parse_args()
if a.epochs<1 or a.samples<4 or a.validation_samples<4:raise ValueError('Positive epochs and at least 4 samples required')
random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
variables=['WTMP','ATMP','WSPD','PRES'];stations=['41009','41010'];length=128
sources={}
def load(year):
 idx=pd.date_range(f'{year}-01-01',f'{year+1}-01-01',freq='h',inclusive='left',tz='UTC')
 arrays=[]
 for station in stations:
  path=Path(f'data/raw/{station}h{year}.csv')
  f=hourly(load_csv(path)).reindex(idx).reindex(columns=variables)
  arrays.append(f.to_numpy(dtype='float32'))
  sources[f'{station}/{year}']=hashlib.sha256(path.read_bytes()).hexdigest()
 return np.stack(arrays,axis=1)
train=[load(2022),load(2023)];validation=[load(2024)]
mean,std=training_stats(train)
trainset=MaskedWindows(train,mean,std,samples=a.samples,seed=a.seed)
valset=MaskedWindows(validation,mean,std,samples=a.validation_samples,seed=777)
trainloader=DataLoader(trainset,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed))
valloader=DataLoader(valset,batch_size=a.batch_size)
model=OceanTransformer(len(variables),width=a.width,layers=a.layers).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
config={**vars(a),'device':str(device),'variables':variables,'stations':stations,'window_hours':length,
 'train_years':[2022,2023],'validation_year':2024,'test_evaluated':False,
 'parameters':sum(p.numel() for p in model.parameters()),'torch':torch.__version__,
 'source_csv_sha256':sources,'inputs':'standardized observed values, observed masks, position and station embeddings; no interpolation',
 'normalization':'mean/std from training years only','validation_seed':777,
 'masking':'sensor-only; observed target gap; windows may overlap; fixed masks across epochs'}
(out/'config.json').write_text(json.dumps(config,indent=2))
np.savez(out/'normalization.npz',mean=mean,std=std)
best=float('inf');history=[]
for epoch in range(1,a.epochs+1):
 model.train();total=0.;count=0
 for values,observed,truth,mask in trainloader:
  values,observed,truth,mask=[x.to(device) for x in [values,observed,truth,mask]]
  prediction=model(values,observed)
  loss=((prediction[mask]-truth[mask])**2).mean()
  optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
  total+=loss.item()*mask.sum().item();count+=mask.sum().item()
 model.eval();errors={k:[] for k in [1,6,24,72]}
 with torch.no_grad():
  for values,observed,truth,mask in valloader:
   pred=model(values.to(device),observed.to(device)).cpu()
   for i in range(len(pred)):
    gap=int(mask[i].sum());errors[gap].extend(((pred[i][mask[i]]-truth[i][mask[i]])*std[0,0]).tolist())
 metrics={str(k):{'mae':float(np.abs(v).mean()),'rmse':float(np.sqrt(np.mean(np.square(v)))),'points':len(v)} for k,v in errors.items() if v}
 macro=float(np.mean([v['mae'] for v in metrics.values()]))
 history.append({'epoch':epoch,'train_masked_mse_normalized':total/count,'validation_macro_mae_C':macro,'by_gap':metrics})
 if macro<best:
  best=macro;torch.save(model.state_dict(),out/'best.pt');(out/'best_validation.json').write_text(json.dumps(history[-1],indent=2))
 print(json.dumps(history[-1]),flush=True)
(out/'history.json').write_text(json.dumps(history,indent=2))
print(f'Weights: {out}/best.pt; parameters: {config["parameters"]}; device: {device}',flush=True)
