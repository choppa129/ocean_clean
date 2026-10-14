"""Preserve small v1 weights and render learning curves from the completed run."""
import json,shutil,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
source=Path('outputs/transformer_v1');out=Path('reports/transformer')
ckpt=out/'checkpoint';ckpt.mkdir(parents=True,exist_ok=True)
for name in ['best.pt','normalization.npz','config.json']:
 shutil.copy2(source/name,ckpt/name)
for name in ['history.json','best_validation.json']:
 shutil.copy2(source/name,out/name)
history=json.loads((source/'history.json').read_text())
best=json.loads((source/'best_validation.json').read_text())
(out/'checkpoint_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ckpt.iterdir()},indent=2))
plt.style.use('dark_background');fig,ax=plt.subplots(figsize=(8,4))
fig.patch.set_facecolor('#081722');ax.set_facecolor('#081722')
ax.plot([r['epoch'] for r in history],[r['validation_macro_mae_C'] for r in history],color='#3dd6c6',marker='o')
ax.scatter([best['epoch']],[best['validation_macro_mae_C']],color='#ffba70',s=80,zorder=3,label='Validation-selected checkpoint')
ax.set_xlabel('Epoch');ax.set_ylabel('2024 validation macro MAE (°C)');ax.set_title('Direct Masked Transformer · first CPU training run')
ax.grid(alpha=.15);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'learning_curve.svg')
report='\n## Recorded v1 run\n\n![Validation learning curve](learning_curve.svg)\n\n'
report+=f"15 CPU epochs; 2,048 training windows and 256 validation windows. Best epoch: **{best['epoch']}**. Validation macro MAE: **{best['validation_macro_mae_C']:.4f} °C**.\n\n"
report+='| Hidden hours | Validation MAE (°C) |\n|---:|---:|\n'
for gap,m in best['by_gap'].items():report+=f"| {gap} | {m['mae']:.4f} |\n"
report+='\nThese are validation-selected development metrics, not independent test results.\n'
p=out/'README.md';text=p.read_text().split('\n## Recorded v1 run')[0];p.write_text(text+report)
