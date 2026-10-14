"""Reproduce a fixed, multi-seed first experiment; no test-set tuning."""
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
import pandas as pd
from ocean_clean.data import download_thredds, hourly, quality_report
from ocean_clean.benchmark import evaluate, split

out = Path('reports/first_experiment')
out.mkdir(parents=True, exist_ok=True)
Path('outputs').mkdir(exist_ok=True)
frame = hourly(download_thredds('41009', 2024, 'data/raw'))
(out / 'quality.json').write_text(json.dumps(quality_report(frame), indent=2))
parts = split(frame)
metadata = {'station': '41009', 'year': 2024, 'target': 'WTMP', 'unit': 'degrees Celsius',
 'seeds': [42, 43, 44], 'requested_blocks_per_split': 20, 'gap_hours': [1,6,24,72],
 'mode': 'offline; both gap endpoints allowed', 'training': 'separate model per gap length and seed',
 'splits': {name: {'start': str(part.index.min()), 'end': str(part.index.max()), 'rows':len(part)}
            for name,part in zip(['train','validation','test'],parts)},
 'versions': {name: version(name) for name in ['numpy','pandas','scipy','scikit-learn']},
 'python': platform.python_version(),
 'source': json.loads(Path('data/raw/41009h2024.json').read_text()),
 'csv_sha256': hashlib.sha256(Path('data/raw/41009h2024.csv').read_bytes()).hexdigest()}
(out / 'manifest.json').write_text(json.dumps(metadata,indent=2))
metrics, predictions = [], []
for scenario in ['sensor','station']:
 for seed in metadata['seeds']:
  m,p = evaluate(frame, blocks=20, seed=seed, scenario=scenario)
  m['seed'],p['seed'] = seed,seed
  metrics.append(m); predictions.append(p)
  print(f'Finished {scenario} seed={seed}',flush=True)
metrics=pd.concat(metrics,ignore_index=True)
metrics.to_csv(out/'metrics.csv',index=False)
predictions=pd.concat(predictions,ignore_index=True)
predictions.to_csv('outputs/first_predictions.csv',index=False)
summary=metrics.groupby(['split','scenario','gap_hours','model']).agg(
 mae_mean=('mae','mean'),mae_seed_sd=('mae','std'),rmse_mean=('rmse','mean'),
 min_blocks=('blocks','min'),max_blocks=('blocks','max'),min_points=('points','min')).reset_index()
summary.to_csv(out/'summary.csv',index=False)
# One predeclared example, not selected for appearance or model performance.
predictions[(predictions.seed==42)&(predictions.split=='test')&(predictions.gap_hours==24)].to_csv(out/'example_predictions.csv',index=False)
print(summary[summary.split=='test'].to_string(index=False))
