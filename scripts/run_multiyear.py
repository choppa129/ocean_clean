"""Run validation first; test requires a saved protocol and model selection."""
import argparse,hashlib,json,pickle,platform
from pathlib import Path
from importlib.metadata import version
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from ocean_clean.data import download_thredds,hourly,quality_report
from ocean_clean.multistation import LOCAL,corrupt,training_examples,score

p=argparse.ArgumentParser()
p.add_argument('stage',choices=['validate','test'])
args=p.parse_args()
out=Path('reports/multiyear');out.mkdir(parents=True,exist_ok=True)
cache=Path('outputs/multiyear');cache.mkdir(parents=True,exist_ok=True)
protocol={'target_station':'41009','neighbors':['41010'],
 'train_years':[2022,2023],'validation_year':2024,'test_year':2025,
 'gap_hours':[1,6,24,72],'seeds':[42,43,44],'evaluation_blocks':40,
 'training_rounds_per_year':4,'training_blocks_requested_per_round':150,
 'task':'offline reconstruction; future gap endpoints allowed',
 'models':['linear','residual_local','residual_neighbors'],
 'hyperparameters':{'max_iter':150,'max_leaf_nodes':15,'min_samples_leaf':20,
                   'learning_rate':0.05,'l2_regularization':1.0,'early_stopping':False,'random_state':42},
 'selection':'per scenario/gap lowest mean validation MAE; linear included',
 'notes':'Same fixed models evaluated on 3 masks; seed spread is not an independent-data confidence interval.'}

def load(year):
 frames={}
 for station in [protocol['target_station']]+protocol['neighbors']:
  f=hourly(download_thredds(station,year,'data/raw'))
  idx=pd.date_range(f'{year}-01-01',f'{year+1}-01-01',freq='h',inclusive='left',tz='UTC',name='time')
  frames[station]=f.reindex(idx)
 local=frames['41009'].reindex(columns=LOCAL)
 neighbors=pd.DataFrame({f'n{s}_WTMP':frames[s].WTMP for s in protocol['neighbors']},index=local.index)
 return local,neighbors,frames

if args.stage=='validate':
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 train={};quality={};manifest={}
 for year in protocol['train_years']+[2024]:
  local,neighbors,frames=load(year)
  if year in protocol['train_years']:train[year]=(local,neighbors)
  else:val=(local,neighbors)
  for station,f in frames.items():
   quality[f'{station}/{year}']=quality_report(f)
   manifest[f'{station}/{year}']=json.loads(Path(f'data/raw/{station}h{year}.json').read_text())
 rows=[];train_counts=[];bundles={};selection={}
 for scenario in ['sensor','station']:
  for length in protocol['gap_hours']:
   x,y,counts=training_examples(train,length,scenario)
   train_counts.append(dict(scenario=scenario,gap_hours=length,**counts))
   columns={ 'residual_local':[c for c in x if not c.startswith('n410')],
             'residual_neighbors':list(x.columns)}
   models={}
   for name,cols in columns.items():
    model=HistGradientBoostingRegressor(**protocol['hyperparameters'])
    model.fit(x[cols],y);models[name]=model
   bundles[(scenario,length)]={'models':models,'columns':columns}
   for seed in protocol['seeds']:
    vx,mask,n=corrupt(*val,length,40,seed,scenario)
    truth=val[0].WTMP.loc[mask];base=vx.linear.loc[mask]
    for name in protocol['models']:
     pred=base if name=='linear' else base+models[name].predict(vx.loc[mask,columns[name]])
     rows.append(dict(split='validation',scenario=scenario,gap_hours=length,seed=seed,
                      model=name,blocks=n,points=int(mask.sum()),**score(truth,pred)))
   subset=pd.DataFrame(rows)
   means=subset[(subset.scenario==scenario)&(subset.gap_hours==length)].groupby('model').mae.mean()
   selection[f'{scenario}/{length}']=means.idxmin()
   print(scenario,length,counts['augmented_rows'],selection[f'{scenario}/{length}'],flush=True)
 pd.DataFrame(rows).to_csv(out/'validation_metrics.csv',index=False)
 (out/'selection.json').write_text(json.dumps(selection,indent=2))
 (out/'training_counts.json').write_text(json.dumps(train_counts,indent=2))
 (out/'quality_train_validation.json').write_text(json.dumps(quality,indent=2))
 (out/'sources_train_validation.json').write_text(json.dumps(manifest,indent=2))
 with (cache/'models.pkl').open('wb') as f:pickle.dump(bundles,f)
 (out/'environment.json').write_text(json.dumps({'python':platform.python_version(),
   'packages':{n:version(n) for n in ['numpy','pandas','scipy','scikit-learn']},
   'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()},indent=2))
else:
 saved=json.loads((out/'protocol.json').read_text())
 if saved!=protocol:raise ValueError('Protocol changed since validation')
 selection=json.loads((out/'selection.json').read_text())
 # Only load locally produced models; never unpickle external model files.
 with (cache/'models.pkl').open('rb') as f:bundles=pickle.load(f)
 local,neighbors,frames=load(2025)
 rows=[]
 for scenario in ['sensor','station']:
  for length in protocol['gap_hours']:
   b=bundles[(scenario,length)]
   for seed in protocol['seeds']:
    x,mask,n=corrupt(local,neighbors,length,40,seed,scenario)
    truth=local.WTMP.loc[mask];base=x.linear.loc[mask]
    for name in protocol['models']:
     pred=base if name=='linear' else base+b['models'][name].predict(x.loc[mask,b['columns'][name]])
     rows.append(dict(split='test',scenario=scenario,gap_hours=length,seed=seed,model=name,
         selected_on_validation=name==selection[f'{scenario}/{length}'],blocks=n,points=int(mask.sum()),**score(truth,pred)))
 pd.DataFrame(rows).to_csv(out/'test_metrics.csv',index=False)
 (out/'quality_test.json').write_text(json.dumps({s:quality_report(f) for s,f in frames.items()},indent=2))
 (out/'sources_test.json').write_text(json.dumps({s:json.loads(Path(f'data/raw/{s}h2025.json').read_text()) for s in frames},indent=2))
 print(pd.DataFrame(rows).groupby(['scenario','gap_hours','model']).mae.mean().to_string())
