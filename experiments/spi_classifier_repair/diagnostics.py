"""Measured training-size and optimization diagnostics on recovered development data."""
from pathlib import Path
import json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score,balanced_accuracy_score,ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits
from parameters.spi import prepare,LEVELS
from parameters.spi_model_selection import candidate_fit,predict,mlp_fit,stopping_split
from parameters.artifacts import write_json
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/parameter_finalization/runs/notebook9_20261028/submission_v13_features/retrieved_version13/experiments/parameter_finalization/runs/notebook9_20261028_v13/data_run'

def main():
 out=ROOT/'experiments/spi_classifier_repair/runs'/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True)
 x,ids,ix,ys,identity=prepare(BASE/'spi_training/features.npz',BASE/'spi_training/labels.json')
 labels=json.loads((BASE/'spi_training/labels.json').read_text());lookup={r['image_id']:r for r in labels}
 xt=x[ix['train']];yt=np.array(ys['train']);groups=np.array([lookup[ids[i]]['group'] for i in ix['train']]);rows=[]
 with threadpool_limits(limits=1):
  for fold,(tr,va) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=17).split(xt,yt,groups)):
   for seed in (17,29,43):
    ordered=np.random.default_rng(seed).permutation(np.unique(groups[tr]));chosen=[]
    for fraction in (.25,.5,.75,1.):
     n=max(1,int(np.ceil(len(ordered)*fraction)))
     subset=tr[np.isin(groups[tr],ordered[:n])]
     if set(yt[subset])!=set(range(4)):continue
     assert not set(groups[subset]) & set(groups[va])
     model,_=candidate_fit({'family':'linear','C':1.,'balanced':True},xt[subset],yt[subset],seed)
     train=predict(model,'linear',xt[subset]).argmax(1);valid=predict(model,'linear',xt[va]).argmax(1)
     rows.append({'fold':fold,'seed':seed,'fraction':fraction,'training_rows':len(subset),'training_groups':n,
      'train_macro_f1':float(f1_score(yt[subset],train,average='macro',labels=range(4),zero_division=0)),
      'cv_macro_f1':float(f1_score(yt[va],valid,average='macro',labels=range(4),zero_division=0)),
      'cv_balanced_accuracy':float(balanced_accuracy_score(yt[va],valid))})
   print('Learning curve fold',fold+1,'/3',flush=True)
  history=[];fit,stop=stopping_split(yt,groups,117)
  mlp_fit(xt[fit],yt[fit],True,17,xt[stop],yt[stop],epochs=50,history=history)
 write_json(out/'learning_curve_rows.json',rows);write_json(out/'mlp_epoch_history.json',history)
 summaries=[]
 for fraction in (.25,.5,.75,1.):
  r=[v for v in rows if v['fraction']==fraction]
  summaries.append({'fraction':fraction,'mean_rows':float(np.mean([v['training_rows'] for v in r])),
    **{key:float(np.mean([v[key] for v in r])) for key in ('train_macro_f1','cv_macro_f1','cv_balanced_accuracy')}})
 fig,ax=plt.subplots(figsize=(7,4))
 for key,title in [('train_macro_f1','Training'),('cv_macro_f1','Grouped CV')]:
  ax.plot([r['mean_rows'] for r in summaries],[r[key] for r in summaries],'o-',label=title)
 ax.set(xlabel='Training images within each fold',ylabel='Macro-F1',ylim=(0,1),title='Frozen CLIP + balanced linear head; training-only groups');ax.legend();fig.tight_layout();fig.savefig(out/'learning_curve.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for split in ('train','validation'):
  axes[0].plot([r['epoch'] for r in history],[r[split+'_ce'] for r in history],label=split)
  axes[1].plot([r['epoch'] for r in history],[r[split+'_accuracy'] for r in history],label=split)
 axes[0].set(ylabel='Unweighted cross-entropy',xlabel='Epoch');axes[1].set(ylabel='Accuracy',xlabel='Epoch',ylim=(0,1))
 for ax in axes:ax.legend()
 fig.suptitle('MLP: group-separated internal training/early-stop split; not the 96-row validation set');fig.tight_layout();fig.savefig(out/'mlp_training_curves.png',dpi=160);plt.close(fig)
 report=json.loads((BASE/'spi_classifier/metrics/validation.json').read_text());cm=np.array(report['student']['confusion_true_rows_predicted_columns'])
 fig,ax=plt.subplots(figsize=(6,5));ConfusionMatrixDisplay(cm,display_labels=LEVELS).plot(ax=ax,colorbar=False);ax.set_title('Saved v13 student; development validation n=96');fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=160);plt.close(fig)
 write_json(out/'summary.json',{'learning_curve':summaries,'epoch_count':len(history),'data_role':'development; 96-row validation not used in learning curves','reference':'Qwen pseudo-label agreement','feature_signature':identity['feature_signature']})
 print('OUTPUT',out,flush=True);print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
