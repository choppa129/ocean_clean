"""Read-only explorer of committed OceanClean experiment results."""
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title='OceanClean | Experiment Explorer',page_icon='🌊',layout='wide')
st.markdown('### 🌊 OCEANCLEAN')
st.title('Recover the signal. Keep the evidence.')
st.caption('REAL NOAA OBSERVATIONS · OFFLINE RECONSTRUCTION · INDEPENDENT-YEAR EVALUATION')
summary=pd.read_csv(ROOT/'reports/multiyear/summary.csv')
with st.sidebar:
 st.header('Experiment controls')
 split=st.selectbox('Evaluation split',['test','validation'],format_func=lambda x:'2025 · Independent test' if x=='test' else '2024 · Validation')
 scenario=st.selectbox('Outage scenario',['sensor','station'],format_func=lambda x:'Water-temperature sensor' if x=='sensor' else 'Entire target buoy')
 metric=st.radio('Error metric',['mae_mean','rmse_mean'],format_func=lambda x:'MAE (°C)' if x=='mae_mean' else 'RMSE (°C)')
 gap=st.select_slider('Gap duration (hours)',options=[1,6,24,72],value=24)
 st.divider()
 st.caption('Training: 2022–2023\n\nTarget buoy: 41009\n\nNeighbor buoy: 41010\n\nThree evaluation masks; one fixed fitted model per scenario/gap.')
d=summary[(summary.split==split)&(summary.scenario==scenario)]
selected=d[d.gap_hours==gap].set_index('model')
labels={'linear':'Linear interpolation','residual_local':'Local residual model','residual_neighbors':'Neighbor residual model'}
a,b,c=st.columns(3)
for cell,name in zip([a,b,c],labels):
 cell.metric(labels[name],f"{selected.loc[name,metric]:.4f} °C")
st.info('Published finding: adding neighbor data improves several learned-model predictions, but linear interpolation remains strongest across the evaluated 2025 scenario/gap combinations.')
left,right=st.columns([2,1])
with left:
 st.subheader('Error across outage lengths')
 plt.style.use('dark_background')
 fig,ax=plt.subplots(figsize=(9,4.5));fig.patch.set_facecolor('#081722');ax.set_facecolor('#081722')
 for name,color in zip(labels,['#3dd6c6','#8196b2','#ffba70']):
  row=d[d.model==name]
  ax.plot(range(4),row[metric],marker='o',linewidth=2.3,label=labels[name],color=color)
 ax.set_xticks(range(4),['1 h','6 h','24 h','72 h']);ax.set_ylabel('Error (°C) · lower is better')
 ax.grid(alpha=.12);ax.legend(frameon=False,fontsize=8);fig.tight_layout()
 st.pyplot(fig);plt.close(fig)
with right:
 st.subheader('Read this result correctly')
 st.markdown('''**Future endpoints are allowed.** This is retrospective gap repair, not forecasting.

**Real data, artificial outages.** Ground truth exists because measured segments were deliberately hidden.

**No test-driven tuning.** Models were selected using 2024 before evaluating 2025.

**Partial neighbor coverage.** The pipeline keeps absent observations missing.''')
st.subheader('Inspect the numbers')
st.dataframe(d.rename(columns={'mae_seed_sd':'MAE SD across masks (not CI)'}),hide_index=True,width='stretch')
st.download_button('Download this comparison · CSV',d.to_csv(index=False),'ocean_clean_comparison.csv','text/csv')
with st.expander('Protocol, limitations & reproducibility'):
 st.markdown((ROOT/'reports/multiyear/README.md').read_text())
 st.code('pip install -e ".[dashboard]"\npython -m streamlit run app/dashboard.py',language='bash')
st.caption('OceanClean · An auditable ML engineering project. No claims of research novelty or operational sensor-fault detection.')
