"""Generate portfolio figures directly from committed experiment metrics."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'reports/multiyear/summary.csv')
plt.style.use('dark_background')
fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
fig.patch.set_facecolor('#081722')
for ax,scenario,title in zip(axes,['sensor','station'],['Water-temperature sensor outage','Entire target buoy outage']):
 ax.set_facecolor('#081722')
 for model,label,color in [('linear','Linear interpolation','#3dd6c6'),('residual_local','Local residual model','#8196b2'),('residual_neighbors','Neighbor residual model','#ffba70')]:
  r=d[(d.split=='test')&(d.scenario==scenario)&(d.model==model)]
  ax.plot(range(4),r.mae_mean,'o-',color=color,linewidth=2.4,label=label)
 ax.set_xticks(range(4),['1 h','6 h','24 h','72 h'])
 ax.set_title(title,fontsize=12,pad=15);ax.grid(alpha=.1)
 ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Mean absolute error (°C) · lower is better')
axes[1].legend(frameon=False,fontsize=8)
fig.suptitle('2025 / Independent-year evaluation',x=.075,ha='left',fontsize=20,fontweight='bold')
fig.text(.075,.02,'Trained on 2022–2023 · Selected on 2024 · Three artificial-gap masks · Offline reconstruction',fontsize=10,color='#aec6d2')
fig.tight_layout(rect=[0,.06,1,.94])
fig.savefig(ROOT/'assets/holdout_results.svg',facecolor=fig.get_facecolor())
fig.savefig(ROOT/'outputs/holdout_results.png',dpi=150,facecolor=fig.get_facecolor())
