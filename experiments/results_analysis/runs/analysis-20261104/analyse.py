import json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).parent
EXT=Path('C:/Users/satya/Documents/SemanticSchedulerArtifacts/paired-20261102/historical_state')
sources=[]
def read(p):
 sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return json.loads(p.read_text())
reports=list((ROOT/'experiments/master_study/runs/paired-launch-20261102-205933/checks/paired-check-20261102T220621Z').rglob('REPORT.json'))
models={}
for p in reports:
 for k,v in read(p)['statuses'].items():
  if v.get('status')=='evaluated_development_only':
   name=k.split('/')[0]
   if name not in models:models[name]={'penalty':-v['validation_score'],'seconds':v['seconds']}
b=read(next((EXT/'windows_v1_output_probe').rglob('baseline_summary.json')))
spi=-b['spi_fifo']['score'];fifo=-b['fifo']['score']
rank=[]
for name,v in models.items():rank.append({'method':name,**v,'improvement_vs_spi_percent':100*(spi-v['penalty'])/spi})
rank.sort(key=lambda x:x['penalty'])
with (OUT/'model_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rank[0]));w.writeheader();w.writerows(rank)
classes=[]
for policy,v in b.items():
 for tier in ['Critical','High','Medium','Low']:
  c=[e['by_reference_tier'][tier] for e in v['episodes']];n=sum(x['offered'] for x in c);sel=sum(x['selected'] for x in c);fu=sum(x['followup_count'] for x in c);elig=sum(x['eligible'] for x in c)
  classes.append({'policy':policy,'class':tier,'offered':n,'selected':sel,'unselected':n-sel,'mean_selected_wait':sum(x['mean_selected_wait']*x['selected'] for x in c if x['selected'])/sel if sel else None,'restricted_mean_wait':sum(x['restricted_mean_wait']*x['followup_count'] for x in c if x['followup_count'])/fu if fu else None,'within_budget':sum(x['within_budget']*x['eligible'] for x in c if x['eligible'])/elig if elig else None})
with (OUT/'baseline_class_tradeoffs.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(classes[0]));w.writeheader();w.writerows(classes)
rate=read(ROOT/'experiments/stream_scheduler/runs/gap-repair-20261102T221315Z/rate_cpu_full/results.json')
traces={}
for x in rate:traces.setdefault(x['trace'],{})[x['policy']]=x
paired=[]
for tid,d in traces.items():
 for pol in ['round_robin','spi_fifo','spi_novelty']:
  if pol not in d:continue
  a=d['fifo'];v=d[pol]
  paired.append({'trace':tid,'policy':pol,'load':a['realized_service_load'],'burst':a['burst_factor'],'ues':a['ues'],'delta_JD':v['metrics']['J_D']-a['metrics']['J_D'],'delta_wait':v['restricted_mean_wait']-a['restricted_mean_wait']})
with (OUT/'rate_paired_effects.csv').open('w',newline='') as f:
 if paired:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
curves={}
for p in (EXT/'windows_v1_verified').rglob('learning.json'):
 d=read(p)
 if d and 'steps' in d[0]:curves[p.parent.parent.name]=d
fig,ax=plt.subplots(figsize=(9,5));names=[x['method'] for x in rank];vals=[x['penalty'] for x in rank]
ax.barh(names[::-1],vals[::-1],color='#416784');ax.axvline(spi,color='#b14d35',ls='--',label='SPI first');ax.set_xlabel('Development waiting penalty, lower is better');ax.legend();fig.tight_layout();fig.savefig(OUT/'model_comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
for name,c in curves.items():ax.plot([x['steps'] for x in c],[-x['validation'] for x in c],marker='o',label=name)
ax.axhline(spi,color='black',ls='--',label='SPI first');ax.set_xlabel('Training decisions');ax.set_ylabel('Validation penalty');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'rl_curves.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
for pol in sorted(set(x['policy'] for x in rate)):
 z=[x for x in rate if x['policy']==pol];ax.scatter([x['realized_service_load'] for x in z],[x['metrics']['J_D'] for x in z],s=15,alpha=.65,label=pol)
ax.set_xlabel('Realized service load');ax.set_ylabel('Waiting penalty');ax.legend();fig.tight_layout();fig.savefig(OUT/'load_response.png',dpi=160);plt.close(fig)
lines=['# Advanced development result analysis','', '## Scope','This is read only analysis of saved development results. No training or final test access occurred. Time values are simulated ms under the agreed unit convention. No statistical significance claim is made from queue repetitions.','', '## Strong baseline changes the conclusion',f'FIFO penalty is {fifo:.6f}. SPI first is {spi:.6f}. SPI first reduces penalty by {100*(fifo-spi)/fifo:.2f}% relative to FIFO.',f'LambdaMART reduces penalty by {100*(spi-models["lambdamart"]["penalty"])/spi:.2f}% relative to SPI first. Only {sum(x["penalty"]<spi for x in rank)} of the 14 learned methods beat SPI first numerically. Small differences remain unconfirmed without paired per episode learned outcomes.','', '| Method | Penalty | Improvement over SPI first percent |','|---|---:|---:|']
for x in rank:lines.append(f'| {x["method"]} | {x["penalty"]:.4f} | {x["improvement_vs_spi_percent"]:.2f} |')
lines+=['','## Class trade offs in the matched baseline cohort','Counts pool the same 15 validation episodes. Restricted waits weight each eligible follow up observation. Selected mean excludes unselected messages and must be read with their count.','', '| Policy | Class | Offered | Unselected | Selected mean wait | Restricted wait | Within budget |','|---|---|---:|---:|---:|---:|---:|']
for x in classes:
 if x['policy'] in ['fifo','spi_fifo','spi_novelty']:
  lines.append(f'| {x["policy"]} | {x["class"]} | {x["offered"]} | {x["unselected"]} | {x["mean_selected_wait"]:.3f} | {x["restricted_mean_wait"]:.3f} | {x["within_budget"]:.3f} |')
lines+=['','## RL learning dynamics']
for name,c in curves.items():
 best=max(c,key=lambda x:x['validation']);lines.append(f'- {name}: best observed checkpoint at {best["steps"]} decisions, penalty {-best["validation"]:.4f}; final checkpoint penalty {-c[-1]["validation"]:.4f}. Observed checkpoints do not prove convergence.')
lines+=['','## Rate study paired effects','Effects below compare policies on identical saved traces. Wins mean strictly lower waiting penalty. These traces are workload variations from one seed, not independent emergency events.']
for pol in sorted(set(x['policy'] for x in paired)):
 z=[x for x in paired if x['policy']==pol];delta=np.array([x['delta_JD'] for x in z]);lines.append(f'- {pol}: {len(z)} paired traces; mean delta versus FIFO {delta.mean():.5f}; median {np.median(delta):.5f}; wins {sum(delta < -1e-12)}, ties {sum(abs(delta)<=1e-12)}, losses {sum(delta > 1e-12)}.')
lines+=['','## Interpretation and next experiments','The implemented target depends on reference priority and waiting. The current comparison cannot establish a distinct benefit from novelty or all 93 scene categories. Within class marginal costs often tie, leaving no training signal for a finer semantic order.','Keep LambdaMART, RankNet and delay regression as the principal learned candidates. Retain SPI first as the strong comparator. Attention and temporal complexity are not justified by the present aggregate scores.','Do not infer that replay prioritization is universally harmful. Its poorer result may reflect training distribution, optimization settings or reward target properties. Saved learning curves establish instability or regression only in this run.','Run matched no scene, compact scene and full scene ablations. Add SPI only learned models to distinguish better priority estimation from extra semantic reasoning. Keep the final test closed until these decisions are fixed.','Per episode learned outcomes, critical safeguards, measured inference profiles and feature ablations are not present in this retrieved evidence subset. Therefore no learned class benefit, confidence interval, Pareto deployment winner or feature importance is invented.','Training status seconds can include validation and concurrent resource contention. They are not pure training throughput measurements.','', '## Figures','![Learned comparison](model_comparison.png)','![RL curves](rl_curves.png)','![Load response](load_response.png)']
(OUT/'ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
(OUT/'manifest.json').write_text(json.dumps({'sources':sources,'models':len(models),'rate_rows':len(rate),'rate_traces':len(traces),'final_test_opened':False},indent=2))
print('\n'.join(lines[:24]));print('POLICIES',sorted(set(x['policy'] for x in rate)));print('OUTPUT',OUT)
