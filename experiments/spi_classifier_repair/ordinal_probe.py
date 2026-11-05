"""Small ordered cumulative-logit test; four classes, fixed CLIP features, no GPU."""
import json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from parameters.spi import prepare,metrics,LEVELS
from parameters.artifacts import write_json
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/parameter_finalization/runs/notebook9_20261028/submission_v13_features/retrieved_version13/experiments/parameter_finalization/runs/notebook9_20261028_v13/data_run'

def numpy_predict(x,artifact):
 score=x@np.array(artifact['w']);b=np.array(artifact['thresholds']);z=np.clip(score[:,None]-b,-70,70);q=1/(1+np.exp(-z))
 return np.column_stack((1-q[:,0],q[:,0]-q[:,1],q[:,1]-q[:,2],q[:,2]))

def fit(x,y,C,steps=1000):
 torch.set_num_threads(1);torch.manual_seed(17)
 x=torch.tensor(x,dtype=torch.float32);y=torch.tensor(y,dtype=torch.long)
 w=torch.nn.Parameter(torch.zeros(x.shape[1]));raw=torch.nn.Parameter(torch.tensor([-1.1,.695,.695]))
 optimizer=torch.optim.Adam([w,raw],lr=.05)
 counts=torch.bincount(y,minlength=4)
 if torch.any(counts==0):raise ValueError('missing training class')
 weights=len(y)/(4*counts.float());history=[]
 def forward():
  thresholds=torch.cat((raw[:1],raw[:1]+torch.cumsum(torch.nn.functional.softplus(raw[1:]),0)))
  q=torch.sigmoid((x@w)[:,None]-thresholds)
  p=torch.stack((1-q[:,0],q[:,0]-q[:,1],q[:,1]-q[:,2],q[:,2]),1)
  return p,thresholds
 for step in range(steps):
  optimizer.zero_grad();p,b=forward();loss=torch.nn.functional.nll_loss(p.clamp_min(1e-7).log(),y,weight=weights)+w.square().sum()/(2*C*len(y))
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite ordinal loss')
  loss.backward();optimizer.step()
  if step%100==0:history.append({'step':step,'loss':float(loss.detach())})
 p,b=forward();artifact={'schema':'experimental_ordered_logit_v1','levels':list(LEVELS),'w':w.detach().tolist(),'thresholds':b.detach().tolist(),'C':C,'steps':steps,'runtime_promotion_allowed':False}
 actual=numpy_predict(x.numpy(),artifact)
 if not np.allclose(actual,p.detach().numpy(),atol=1e-5):raise ValueError('ordinal NumPy export differs')
 return artifact,history

def main():
 out=ROOT/'experiments/spi_classifier_repair/runs'/('ordinal_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
 x,ids,ix,ys,identity=prepare(BASE/'spi_training/features.npz',BASE/'spi_training/labels.json');labels=json.loads((BASE/'spi_training/labels.json').read_text());lookup={r['image_id']:r for r in labels};xt=x[ix['train']];yt=np.array(ys['train']);groups=np.array([lookup[ids[i]]['group'] for i in ix['train']]);candidates=[]
 folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=17).split(xt,yt,groups))
 for C in (.1,1.,10.):
  scores=[]
  for tr,va in folds:
   assert not set(groups[tr])&set(groups[va]);model,_=fit(xt[tr],yt[tr],C)
   scores.append(float(f1_score(yt[va],numpy_predict(xt[va],model).argmax(1),labels=range(4),average='macro',zero_division=0)))
  candidates.append({'C':C,'fold_macro_f1':scores,'mean_macro_f1':float(np.mean(scores))});print(candidates[-1],flush=True)
 selected=max(candidates,key=lambda r:(r['mean_macro_f1'],-r['C']))
 artifact,history=fit(xt,yt,selected['C']);artifact.update(feature_signature=identity['feature_signature'],rubric_sha256=identity['rubric_sha256'])
 report=metrics(numpy_predict(x[ix['validation']],artifact),np.array(ys['validation']))
 write_json(out/'candidates.json',candidates);write_json(out/'model.json',artifact);write_json(out/'history.json',history)
 write_json(out/'validation.json',{'student':report,'selected':selected,'scope':'exposed development teacher agreement; no final test','runtime_promotion_allowed':False})
 print('ORDINAL_OUTPUT',out,flush=True);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
