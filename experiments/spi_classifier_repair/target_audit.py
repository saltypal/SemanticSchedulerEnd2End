"""Audit target/feature consistency before another training or labeling campaign."""
from pathlib import Path
import json,zipfile,io,time
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from parameters.spi import prepare,SPIClassifier,metrics,LEVELS
from parameters.artifacts import write_json
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/parameter_finalization/runs/notebook9_20261028/submission_v13_features/retrieved_version13/experiments/parameter_finalization/runs/notebook9_20261028_v13/data_run'
def main():
 out=ROOT/'experiments/spi_classifier_repair/runs'/('target_audit_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
 labels=json.loads((BASE/'spi_training/labels.json').read_text());lookup={r['image_id']:r for r in labels}
 x,ids,ix,y,identity=prepare(BASE/'spi_training/features.npz',BASE/'spi_training/labels.json')
 model=SPIClassifier.load(BASE/'spi_classifier/models/spi.json')
 report={'class_source':{split:{source:{level:sum(r['split']==split and r['source']==source and r['level']==level for r in labels) for level in LEVELS} for source in sorted({r['source'] for r in labels})} for split in ('train','validation')},
  'saved_head_fit_metrics':{split:metrics(model.predict_proba(x[ix[split]],identity['feature_signature']),np.array(y[split])) for split in ('train','validation')}}
 similarity=x@x.T;np.fill_diagonal(similarity,-2)
 pairs=[]
 for i in range(len(ids)):
  for j in range(i+1,len(ids)):
   if lookup[ids[i]]['level']!=lookup[ids[j]]['level']:
    pairs.append({'first':ids[i],'second':ids[j],'cosine':float(similarity[i,j]),'levels':[lookup[ids[i]]['level'],lookup[ids[j]]['level']],'splits':[lookup[ids[i]]['split'],lookup[ids[j]]['split']]})
 pairs.sort(key=lambda r:-r['cosine']);report['nearest_conflicting_labels']=pairs[:30]
 report['near_pair_counts']={str(t):{'all_pairs':int(np.sum(np.triu(similarity>=t,1))),'different_labels':sum(r['cosine']>=t for r in pairs)} for t in (.9,.95,.98)}
 write_json(out/'audit.json',report)
 panel=json.loads((ROOT/'parameters/results/spi_inputs_v2_20261027b/panel.json').read_text());panel={r['image_id']:r for r in panel}
 with zipfile.ZipFile(ROOT/'parameters/results/spi_inputs_v2_20261027b/images.zip') as z:
  fig,axes=plt.subplots(6,2,figsize=(10,18))
  for row,entry in enumerate(pairs[:6]):
   for col,key in enumerate(('first','second')):
    image=Image.open(io.BytesIO(z.read(panel[entry[key]]['image_path']))).convert('RGB');axes[row,col].imshow(image);axes[row,col].axis('off');axes[row,col].set_title(f"{entry['levels'][col]} | {entry['splits'][col]} | cos={entry['cosine']:.3f}")
  fig.suptitle('Most similar CLIP pairs with different teacher labels; not proof labels are wrong');fig.tight_layout();fig.savefig(out/'conflicting_pairs.png',dpi=120);plt.close(fig)
 print('AUDIT',out);print(json.dumps({'class_source':report['class_source'],'near_pairs':report['near_pair_counts'],'train_f1':report['saved_head_fit_metrics']['train']['macro_f1']},indent=2))
if __name__=='__main__':main()
