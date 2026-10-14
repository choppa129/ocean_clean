"""Render committed metrics; plotting is optional for benchmark reproduction."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('reports/first_experiment')
df=pd.read_csv(out/'summary.csv')
fig, axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,scenario in zip(axes,['sensor','station']):
 for model,label in [('linear','Linear interpolation'),('gradient_boosting','Gradient boosting'),('train_median','Training median')]:
  d=df[(df.split=='test')&(df.scenario==scenario)&(df.model==model)]
  ax.errorbar(range(4),d.mae_mean,yerr=d.mae_seed_sd,marker='o',capsize=3,label=label)
 ax.set_xticks(range(4),['1','6','24','72'])
 ax.set_xlabel('Artificial gap length (hours)')
 ax.set_title('Water-temperature sensor outage' if scenario=='sensor' else 'Whole-station outage')
 ax.grid(alpha=.2)
axes[0].set_ylabel('Test MAE (degrees C)')
axes[1].legend(fontsize=8)
fig.suptitle('NDBC 41009 / 2024 — offline reconstruction; 3 masking seeds')
fig.text(.5,.01,'Error bars: seed SD, not confidence intervals. Same single-year chronological split.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,1])
fig.savefig(out/'test_mae.svg')
fig.savefig('outputs/test_mae.png',dpi=160)
