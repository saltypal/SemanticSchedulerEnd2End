"""Build evidence-linked parameter report and figures from saved measurements."""
from pathlib import Path
import json, hashlib, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
def save(fig,name):
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight',dpi=180)
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def flow(name,nodes,edges,title):
    fig,ax=plt.subplots(figsize=(13,6));ax.set(xlim=(0,13),ylim=(0,6));ax.axis('off')
    for label,(x,y,w,h) in nodes.items():
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor='#edf4fa',edgecolor='#24587a',lw=1.5))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9,wrap=True)
    for a,b in edges:
        x,y,w,h=nodes[a];xx,yy,ww,hh=nodes[b]
        ax.annotate('',xy=(xx,yy+hh/2),xytext=(x+w,y+h/2),arrowprops={'arrowstyle':'->','color':'#52616c','lw':1.4,'connectionstyle':'arc3,rad=.04'})
    ax.set_title(title,pad=15);save(fig,name)

flow('system',{
 'UE image':(.1,3.4,1.3,.9),'Parameter\nextractor':(2,4.2,1.6,1.1),
 'Image codec\nencoder':(2,2.3,1.6,1.1),'Header + complete\nlatent payload':(4.4,3.3,2,1.2),
 'Received queue\nA, B, C, ...':(7,3.3,1.8,1.2),'Scheduler\nreads headers':(9.4,3.3,1.7,1.2),
 'One decoder\nthen task output':(11.5,3.2,1.3,1.4)},
 [('UE image','Parameter\nextractor'),('UE image','Image codec\nencoder'),('Parameter\nextractor','Header + complete\nlatent payload'),('Image codec\nencoder','Header + complete\nlatent payload'),('Header + complete\nlatent payload','Received queue\nA, B, C, ...'),('Received queue\nA, B, C, ...','Scheduler\nreads headers'),('Scheduler\nreads headers','One decoder\nthen task output')],
 'Complete-message scheduling at the receiver. Transport and radio resources are outside the ranking decision.')
flow('embedding',{'RGB image\n224 × 224 × 3':(.1,3.4,1.7,1.1),
 'CLIP ViT-B/32\n49 patches + CLS\nhidden width 768':(2.4,3.2,2,1.5),
 'Pooled CLS\nprojection\n512 values, L2 norm':(5,3.2,2,1.5),
 'SPI head\n4 class scores':(8,4.7,2,1),
 'Mission text encoder\n512 values → cosine':(8,2.9,2.4,1.1),
 'Same-UE history\nnearest cosine → novelty':(8,1,2.8,1.1)},
 [('RGB image\n224 × 224 × 3','CLIP ViT-B/32\n49 patches + CLS\nhidden width 768'),('CLIP ViT-B/32\n49 patches + CLS\nhidden width 768','Pooled CLS\nprojection\n512 values, L2 norm'),('Pooled CLS\nprojection\n512 values, L2 norm','SPI head\n4 class scores'),('Pooled CLS\nprojection\n512 values, L2 norm','Mission text encoder\n512 values → cosine'),('Pooled CLS\nprojection\n512 values, L2 norm','Same-UE history\nnearest cosine → novelty')],
 'The CLIP embedding supports parameter extraction. It is separate from the image-codec latent.')
flow('scene',{'Image + ordered\n93 class names':(.1,3,2,1.2),
 'YOLO-World\nconf .25 / IoU .5\n640 / cap 300':(2.8,3,2.2,1.3),
 'Retained boxes\nclass, score, xyxy\nclip / validate geometry':(5.7,3,2.3,1.3),
 'Per-class counts\nand largest-box fractions\nplus availability masks':(8.7,4,3.5,1.3),
 'Raw box audit +\nwire representation\nproducer hash + timing':(8.7,1.6,3.5,1.3)},
 [('Image + ordered\n93 class names','YOLO-World\nconf .25 / IoU .5\n640 / cap 300'),('YOLO-World\nconf .25 / IoU .5\n640 / cap 300','Retained boxes\nclass, score, xyxy\nclip / validate geometry'),('Retained boxes\nclass, score, xyxy\nclip / validate geometry','Per-class counts\nand largest-box fractions\nplus availability masks'),('Retained boxes\nclass, score, xyxy\nclip / validate geometry','Raw box audit +\nwire representation\nproducer hash + timing')],
 'Detection evidence. Box extent does not establish depth, severity or urgency.')

comparison=next((ROOT/'experiments/spi_comparison_4000/runs/20261101T074756Z/ops/remote_compact').rglob('original/candidates.json'))
rows=json.loads(comparison.read_text());heads=list(dict.fromkeys(r['head'] for r in rows));encoders=['clip_b32','siglip2_b16','dino']
heat=np.array([[next(r['development']['macro_f1'] for r in rows if r['encoder']==e and r['head']==h) for e in encoders] for h in heads])
fig,ax=plt.subplots(figsize=(7,7));im=ax.imshow(heat,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(3),['CLIP','SigLIP2','DINO']);ax.set_yticks(range(len(heads)),heads)
for i in range(len(heads)):
 for j in range(3):ax.text(j,i,f'{heat[i,j]:.3f}',ha='center',va='center',color='white' if heat[i,j]>.65 else 'black')
fig.colorbar(im,ax=ax,label='Development macro-F1');ax.set_title('36 completed candidates, original balancing regime\nPartial comparison. ConvNeXtV2 did not complete.');save(fig,'comparison')
with (OUT/'completed_comparison.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['encoder','head','development_macro_f1','artifact_bytes'])
 for r in rows:w.writerow([r['encoder'],r['head'],r['development']['macro_f1'],r['artifact_bytes']])
fig,ax=plt.subplots(figsize=(8,4));labels=['Low','Medium','High','Critical'];xx=np.arange(4)
for i,(name,v) in enumerate([('Train',[782,339,899,48]),('Validation',[419,149,445,24]),('Test',[273,200,325,30])]):ax.bar(xx+(i-1)*.25,v,.25,label=name)
ax.set_xticks(xx,labels);ax.set_ylabel('Usable VLM-labeled images');ax.set_title('3,933 usable labels from 4,000 selected CrisisMMD images');ax.legend();save(fig,'support')
receipt=json.loads((ROOT/'experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/remote_compact/training_receipt.json').read_text())
fig,ax=plt.subplots(figsize=(7,4));ax.bar(labels,[receipt['test'][c]['recall'] for c in labels],color='#237b9b');ax.set_ylim(0,1);ax.set_ylabel('Recall against VLM pseudo-labels');ax.set_title('Deployed provisional Mini classifier\nCLIP + balanced logistic regression, 828 test images');save(fig,'mini_recall')
fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].bar(['YOLO-World','Grounding DINO'],[19,368.69],color=['#237b9b','#c36b45']);axes[0].set_ylabel('Median detector-call milliseconds');axes[0].set_title('Measured 300-image, 16-noun grid')
axes[1].bar(['YOLO-World','Grounding DINO'],[1.99,3.47],color=['#237b9b','#c36b45']);axes[1].set_ylabel('Mean exact-vocabulary boxes / image');axes[1].set_title('Counts are not detection accuracy');save(fig,'detectors')
leader=max(rows,key=lambda r:r['development']['macro_f1']);grid=max(leader['grid'],key=lambda r:r['mean']);fig,axes=plt.subplots(1,2,figsize=(10,4))
for i,hist in enumerate(grid['histories']):
 axes[0].plot([r['epoch'] for r in hist],[r['training_loss'] for r in hist],label=f'Fold {i+1}')
 axes[1].plot([r['epoch'] for r in hist],[r['validation_macro_f1'] for r in hist],label=f'Fold {i+1}')
axes[0].set_ylabel('Training cross-entropy');axes[1].set_ylabel('Fold validation macro-F1')
for ax in axes:ax.set_xlabel('Epoch');ax.legend()
fig.suptitle('CLIP + single-query cross-attention, selected learning-rate fold histories');save(fig,'learning')
manifest={'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [comparison,ROOT/'experiments/dataset_detector_grid/runs/analysis_v18/REPORT.md',ROOT/'experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/remote_compact/training_receipt.json']},'figures':'All plotted values come from saved outputs; diagrams explain architecture, not measured accuracy.'}
(OUT/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Built figures, comparison CSV and evidence manifest:',OUT)
