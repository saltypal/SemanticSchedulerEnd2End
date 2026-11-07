from pathlib import Path
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

R=Path(__file__).resolve().parent;ROOT=R.parents[3];F=R/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
C={'Traditional':'#72849a','Semantic rule':'#15958e','Learned':'#425f9c'}
bpath=next(Path('C:/Users/satya/Documents/SemanticSchedulerArtifacts/paired-20261102/historical_state/windows_v1_output_probe').rglob('baseline_summary.json'))
b=json.loads(bpath.read_text())
labels={'fifo':'FIFO','round_robin':'Round robin','random':'Random','spi_fifo':'SPI-first','relevance':'Relevance-first','novelty':'Novelty-first','spi_novelty':'SPI then novelty','predicted_edf':'Predicted EDF','lambdamart':'LambdaMART','delay_huber':'Delay regression','ranknet':'RankNet','deepsets_ranknet':'Deep Sets RankNet','listmle':'Tie-aware listwise','temporal_deepsets_ranknet':'Temporal Deep Sets','set_transformer_ranknet':'Attention set ranker','mlp_choice':'MLP choice','ranknet_warmstart_ppo':'Ranker + PPO','priority_classifier':'Priority classifier','linear_choice':'Linear choice','masked_ppo':'PPO','masked_double_dqn':'Double DQN','td_per':'Double DQN + replay priority'}
rows=[{'id':k,'name':labels[k],'family':'Traditional' if k in ['fifo','round_robin','random'] else 'Semantic rule','penalty':-v['score']} for k,v in b.items()]
for x in csv.DictReader((ROOT/'experiments/results_analysis/runs/analysis-20261104/model_comparison.csv').open()):rows.append({'id':x['method'],'name':labels[x['method']],'family':'Learned','penalty':float(x['penalty'])})
rows.sort(key=lambda x:x['penalty']);(R/'comparison.json').write_text(json.dumps({'rows':rows,'baseline_path':str(bpath)},indent=2))
def save(fig,name):fig.savefig(F/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(F/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def box(ax,x,y,w,h,t,color='#eaf0f6',size=11):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015,rounding_size=.025',fc=color,ec='#57718c',lw=1.2));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=size)
def arrow(ax,a,z):ax.add_patch(FancyArrowPatch(a,z,arrowstyle='-|>',mutation_scale=15,lw=1.3,color='#425469'))

fig,ax=plt.subplots(figsize=(11,5.5));ax.set_xlim(0,11);ax.set_ylim(0,5.5);ax.axis('off')
box(ax,.1,2.4,1.35,.9,'Original\nimage');box(ax,2,3.6,2,.9,'Parameter extractor\nsmall scheduling clues',color='#ddf2ed');box(ax,2,1.2,2,.9,'Image encoder\nreconstruction data');arrow(ax,(1.45,2.95),(2,4));arrow(ax,(1.45,2.7),(2,1.65))
box(ax,4.8,2.4,2,.9,'Complete message\nHEADER + PAYLOAD');arrow(ax,(4,4),(4.8,3.1));arrow(ax,(4,1.65),(4.8,2.6))
box(ax,7.55,3.6,2.8,.9,'Receiver queue\ncomplete messages wait');arrow(ax,(6.8,3),(7.6,3.9))
box(ax,7.55,2.1,2.8,.9,'Scheduler reads headers\nand chooses ONE message',color='#ddf2ed');arrow(ax,(8.95,3.6),(8.95,3))
box(ax,7.55,.6,2.8,.9,'Decoder reads selected payload\nand reconstructs the image');arrow(ax,(8.95,2.1),(8.95,1.5))
ax.text(.1,.15,'Header = information used to choose. Payload = information used to reconstruct.',fontsize=12,fontweight='bold');save(fig,'architecture')

fig,ax=plt.subplots(figsize=(10,3.8));ax.set_xlim(-.2,5.4);ax.set_ylim(-.8,2.4);ax.axis('off')
for y,title,order in [(1.45,'FIFO',['Routine','High','Critical']),(.1,'Priority order',['Critical','High','Routine'])]:
 ax.text(-.1,y+.65,title,fontsize=12,fontweight='bold')
 for i,t in enumerate(order):box(ax,i*1.8,y,1.5,.5,t,color={'Critical':'#f6dcd7','High':'#f8edce','Routine':'#e2ecf4'}[t]);
 for i in range(2):arrow(ax,(i*1.8+1.5,y+.25),((i+1)*1.8,y+.25))
ax.text(0,-.55,'Illustration only. The same three images can be processed in a different order.',fontsize=11);save(fig,'ordering')

fig,ax=plt.subplots(figsize=(10,3.6));ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,3.6)
for i,(t,col) in enumerate([('Raw image\nreal photograph','#eaf0f6'),('Offline teacher\nimage + rubric','#ddf2ed'),('Saved label\nLow / Medium /\nHigh / Critical','#eaf0f6'),('Train small model\nencoder + classifier','#eaf0f6')]):
 box(ax,.1+i*2.55,1.8,2,.95,t,col)
 if i<3:arrow(ax,(2.1+i*2.55,2.27),(2.65+i*2.55,2.27))
box(ax,2.7,.1,4.8,.8,'At runtime: image → small SPI model → scheduling header',color='#ddf2ed');ax.text(.1,3.25,'Training uses the teacher. Runtime scheduling does not.',weight='bold',fontsize=13);save(fig,'label_pipeline')

fig,ax=plt.subplots(figsize=(9,4.5));w=np.linspace(0,20,301)
for name,d,c in [('Critical',2,'#c05545'),('High',4,'#c18b24'),('Medium',8,'#15958e'),('Low',16,'#425f9c')]:ax.plot(w,w/d+np.maximum(w/d-1,0),label=f'{name}: target wait {d} ms',lw=2.3,color=c);ax.scatter([d],[1],color=c,s=45)
ax.set(xlabel='Queue waiting time (ms)',ylabel='Waiting penalty, lower is better',xlim=(0,20),ylim=(0,12));ax.axhline(1,ls=':',color='#999999');ax.legend();ax.grid(alpha=.15);save(fig,'budget_curves')

fig,ax=plt.subplots(figsize=(10,4));ax.set_ylim(-1,2);ax.set_xlim(-.5,7.5);ax.axis('off')
for x,t in [(0,'Arrives\n0'),(2,'Processing starts\n2'),(5,'Processing ends\n5')]:ax.axvline(x,ymin=.3,ymax=.8,color='#61758b',lw=1.2);ax.text(x,1.6,t,ha='center')
ax.annotate('',xy=(2,.6),xytext=(0,.6),arrowprops={'arrowstyle':'<->','color':'#15958e','lw':2});ax.text(1,.85,'Waiting = 2',ha='center',color='#137b76',weight='bold')
ax.annotate('',xy=(5,.6),xytext=(2,.6),arrowprops={'arrowstyle':'<->','color':'#425f9c','lw':2});ax.text(3.5,.85,'Service = 3',ha='center',color='#425f9c',weight='bold')
ax.annotate('',xy=(5,-.35),xytext=(0,-.35),arrowprops={'arrowstyle':'<->','color':'#516070','lw':1.5});ax.text(2.5,-.7,'Arrival to completion = 5',ha='center');save(fig,'time_definitions')

fig,ax=plt.subplots(figsize=(10,9));y=np.arange(len(rows));ax.barh(y,[r['penalty'] for r in rows],color=[C[r['family']] for r in rows]);ax.set_yticks(y,[r['name'] for r in rows]);ax.invert_yaxis();ax.set_xlim(0,7);ax.set_xlabel('Average waiting penalty, lower is better')
for i,r in enumerate(rows):ax.text(r['penalty']+.045,i,f"{r['penalty']:.3f}",va='center',fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=v,label=k) for k,v in C.items()],loc='lower right');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);save(fig,'all_methods')

fig,ax=plt.subplots(figsize=(9,4));selected=['fifo','round_robin','spi_fifo','lambdamart'];z=[next(r for r in rows if r['id']==k) for k in selected];ax.bar([r['name'] for r in z],[r['penalty'] for r in z],color=[C[r['family']] for r in z]);ax.set_ylim(0,7);ax.set_ylabel('Average waiting penalty')
for i,r in enumerate(z):ax.text(i,r['penalty']+.1,f"{r['penalty']:.3f}",ha='center',weight='bold')
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);save(fig,'key_comparison')

cl=list(csv.DictReader((ROOT/'experiments/results_analysis/runs/analysis-20261104/baseline_class_tradeoffs.csv').open()))
fig,(a,z)=plt.subplots(1,2,figsize=(10,4.1));tiers=['Critical','High','Medium','Low'];x=np.arange(4)
for j,(pol,label,color) in enumerate([('fifo','FIFO','#72849a'),('spi_fifo','SPI-first','#15958e'),('spi_novelty','SPI + novelty','#425f9c')]):
 vals=[next(r for r in cl if r['policy']==pol and r['class']==t) for t in tiers]
 a.bar(x+(j-1)*.25,[float(r['within_budget'])*100 for r in vals],width=.25,label=label,color=color)
 z.bar(x+(j-1)*.25,[int(r['unselected']) for r in vals],width=.25,label=label,color=color)
for q in [a,z]:q.set_xticks(x,tiers);q.grid(axis='y',alpha=.15);q.set_axisbelow(True)
a.set_ylabel('Selected within target wait, percent');a.set_ylim(0,105);a.legend(fontsize=9);z.set_ylabel('Images still unselected at end');save(fig,'class_tradeoffs')

fig,ax=plt.subplots(figsize=(8.5,4));ax.bar(['Low','Medium','High','Critical'],[79.49,36,69.54,73.33],color=['#72849a','#15958e','#425f9c','#c05545']);ax.set_ylim(0,105);ax.set_ylabel('Correctly recognized within class, percent');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
for i,v in enumerate([79.49,36,69.54,73.33]):ax.text(i,v+2,f'{v:.2f}%',ha='center')
save(fig,'spi_recall')
fig,ax=plt.subplots(figsize=(8.5,3.7));ax.barh(['CrisisMMD','Open Images','Places365'],[4000,1000,1000],color=['#425f9c','#15958e','#72849a']);ax.set_xlabel('Images');ax.set_xlim(0,4500)
for i,v in enumerate([4000,1000,1000]):ax.text(v+50,i,str(v),va='center')
save(fig,'image_sources')
print('Created',len(list(F.glob('*.png'))),'figures from recorded data or labeled illustrations')
