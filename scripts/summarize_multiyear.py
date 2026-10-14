"""Summarize a completed fixed-protocol run without selecting on test results."""
import json
from pathlib import Path
import pandas as pd
p=Path('reports/multiyear')
v=pd.read_csv(p/'validation_metrics.csv');t=pd.read_csv(p/'test_metrics.csv')
s=pd.concat([v,t]).groupby(['split','scenario','gap_hours','model']).agg(mae_mean=('mae','mean'),mae_seed_sd=('mae','std'),rmse_mean=('rmse','mean'),min_blocks=('blocks','min'),max_blocks=('blocks','max')).reset_index()
s.to_csv(p/'summary.csv',index=False)
sel=json.loads((p/'selection.json').read_text())
selected=t[t.selected_on_validation].groupby(['scenario','gap_hours','model']).mae.mean().reset_index()
selected.to_csv(p/'selected_test_metrics.csv',index=False)
rows=[]
for gap in [1,6,24,72]:
 r=s[(s.split=='test')&(s.scenario=='sensor')&(s.gap_hours==gap)].set_index('model')
 rows.append(f"| {gap} | {r.loc['linear','mae_mean']:.5f} | {r.loc['residual_local','mae_mean']:.5f} | {r.loc['residual_neighbors','mae_mean']:.5f} | {sel[f'sensor/{gap}']} |")
q=json.loads((p/'quality_train_validation.json').read_text())
q.update({f'{k}/2025':v for k,v in json.loads((p/'quality_test.json').read_text()).items()})
coverage='\n'.join(f"| {k} | {v['variables']['WTMP']['missing_fraction']*100:.2f}% |" for k,v in sorted(q.items()))
report='''# Expanded training + neighboring buoy + independent-year evaluation

## Result

The neighbor residual model improves on the local residual model at all four sensor
outage lengths in the 2025 test, but **does not beat linear interpolation**.
The model selected using 2024 validation was linear for seven of eight scenario/gap
combinations, and neighbor residual learning for sensor-only 24-hour gaps. That one
learned selection did not carry its validation advantage into the independent year.
We retain this negative finding; no model or hyperparameter was changed after test.

## Data separation

- Target: NDBC 41009; neighbor: 41010 (an offshore station, approximately 173 km away
  using 2024 archive coordinates; not an interchangeable colocated sensor).
- Training: 2022–2023. Validation: 2024 (already inspected in the first pilot).
- Independent test: 2025. Its files were downloaded and coverage checked before
  modeling, but its target values did not enter fitting or model selection.
- Protocol, models and validation selection were frozen before test execution.
- Candidate 41012 returned 404 for all requested years; 41013 was excluded as more
  distant. Station selection was based on geography and archive availability,
  not test performance. This version uses exactly one neighbor.

All sources are official NOAA THREDDS archives. Year-wide UTC hourly reindexing
makes partial-year neighbor outages explicit rather than silently dropping them.

| Station/year | Missing hourly water temperature |
|---|---:|
'''+coverage+'''

## Expanded training

Four independently masked training copies per year; up to 150 disjoint blocks in
each copy. Training rows per model are 1,200 / 7,200 / 28,800 / 43,344 for
1 / 6 / 24 / 72-hour gaps. Unique target hours are 1,171 / 5,965 / 14,043 / 15,100.
Repeated masks augment context; they do not create new independent measurements.
The two scenarios use the same training target positions for a controlled comparison.

The learned target is the residual relative to linear interpolation, rather than
absolute water temperature. Local context includes observed endpoint temperatures,
distance to each endpoint, season and available local scalar variables. Neighbor
features add observed water temperature, an availability flag and departure from
its interpolated values at visible local-target anchors. Missing neighbor values
remain missing; their ground truth is never filled in from hidden data.

## Independent 2025 test: sensor-only outage MAE (degrees C)

Means over three evaluation masks; fixed fitted models, not three model-training seeds.

| Gap hours | Linear | Local residual | Neighbor residual | Validation-selected model |
|---:|---:|---:|---:|---|
'''+ '\n'.join(rows)+'''

`summary.csv` includes both scenarios and seed standard deviations. The validation
winner is recorded in `selection.json`; `selected_test_metrics.csv` reports those
choices without retrospective selection of a test winner. All ablations were
predeclared. Test blocks request 40 per gap/seed; actual counts are recorded.

## What this establishes—and what it does not

The pipeline now supports larger training sets, missing neighbor observations,
and a separate-year test. Neighbor features help some learned predictions, but
there is no demonstrated improvement over the strongest baseline. Missing neighbor
coverage and differences between coastal and offshore water may contribute; this
experiment does not isolate those causes. Residual learning and sample expansion
were changed together, so their separate effects cannot be inferred from comparison
to the first pilot.

This is offline reconstruction with future endpoints, not forecasting. Artificial
gaps require complete reference interiors and observed anchors, so real outages
and endpoint gaps can be harder. Hourly means do not require six valid raw samples.
Only one target and one neighbor are modeled. Averages across overlapping masks
are not independent-data confidence intervals. A single held-out year is limited
evidence. 2025 is now an evaluated test set, not a pristine holdout for further tuning.

## Reproduce

```bash
pip install -e .
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/run_multiyear.py validate
# This freezes selection.json and local fitted models before reading test targets.
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/run_multiyear.py test
python scripts/summarize_multiyear.py
python -m unittest discover -s tests -v
```

Models are local trusted pickle files in ignored `outputs/multiyear`; never load
untrusted pickle files. Source checksums, software versions, sample counts,
protocol, quality reports, validation selection and full metrics are committed.
'''
(p/'README.md').write_text(report)
