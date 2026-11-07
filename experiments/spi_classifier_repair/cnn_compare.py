"""Bounded EfficientNet-B1 image-only SPI experiment; no runtime promotion."""
from pathlib import Path
import time,copy,json,hashlib,random
import numpy as np
from PIL import Image,ImageOps
from parameters.artifacts import write_json
from parameters.spi import LEVELS,metrics
from parameters.spi_model_selection import stopping_split
from parameters.dataset_pipeline import validate_records

def _run(records,labels,output,device='cuda:0',seconds=1500,max_epochs=20):
 import torch,torchvision
 from torchvision import transforms
 from torch.utils.data import Dataset,DataLoader
 from sklearn.metrics import f1_score
 started=time.monotonic();out=Path(output)
 if not out.is_dir():raise ValueError('run must own its output directory')
 torch.hub.set_dir(str(Path(__file__).resolve().parents[2]/'models/torchhub'))
 random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.set_num_threads(2)
 validate_records(records)
 from parameters.spi_direct_priority import PROMPT
 from parameters.contracts import digest
 root=Path(__file__).resolve().parents[2]
 teacher=json.loads((root/'parameters/config/final_labeling_teacher.json').read_text())
 if len({r['image_id'] for r in labels})!=len(labels):raise ValueError('duplicate label identity')
 if {r['image_id'] for r in labels}!={r['image_id'] for r in records}:raise ValueError('image/label cohort differs')
 for r in labels:
  if (r.get('teacher_model_id'),r.get('teacher_revision'))!=(teacher['model_id'],teacher['revision']):raise ValueError('teacher identity differs')
  if r.get('rubric_sha256')!=digest(PROMPT) or r.get('label_source')!='vlm_pseudo' or r.get('label_unit')!='whole_image':raise ValueError('label protocol differs')
  if r.get('available') is True and r.get('level') not in LEVELS:raise ValueError('invalid available level')
 lookup={r['image_id']:r for r in labels}
 rows=[r for r in records if lookup[r['image_id']].get('available') is True]
 for r in rows:
  label=lookup[r['image_id']]
  if any(r[k]!=label[k] for k in ('image_sha256','group','split')):raise ValueError('label identity mismatch')
 training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
 y=np.array([LEVELS.index(lookup[r['image_id']]['level']) for r in training]);groups=np.array([r['group'] for r in training])
 split=stopping_split(y,groups,117)
 if split is None:raise ValueError('insufficient groups for inner early stopping')
 fit,stop=split
 if set(groups[fit])&set(groups[stop]):raise ValueError('group leakage')
 write_json(out/'config.json',{'model':'torchvision.efficientnet_b1','weights':'IMAGENET1K_V2','seed':17,'max_epochs':max_epochs,
  'seconds':seconds,'preprocessing':'RGB aspect-preserving bicubic pad240 grey127; ImageNet normalization; no cropping or flips',
  'inner_fit_ids':[training[i]['image_id'] for i in fit],'inner_stop_ids':[training[i]['image_id'] for i in stop],
  'outer_validation_role':'exposed development, not final test','reference':'Qwen pseudo-label agreement',
  'batchnorm_policy':'running statistics frozen; affine parameters in unfrozen blocks remain trainable','torch':torch.__version__,'torchvision':torchvision.__version__,'device':device})
 class Images(Dataset):
  def __init__(self,subset):self.rows=subset
  def __len__(self):return len(self.rows)
  def __getitem__(self,i):
   r=self.rows[i]
   with Image.open(r['image_path']) as im: image=ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),(240,240),method=Image.Resampling.BICUBIC,color=(127,127,127))
   tensor=transforms.functional.to_tensor(image)
   tensor=transforms.functional.normalize(tensor,[.485,.456,.406],[.229,.224,.225])
   return tensor,LEVELS.index(lookup[r['image_id']]['level'])
 def loader(subset,shuffle=False):return DataLoader(Images(subset),batch_size=16,shuffle=shuffle,num_workers=0)
 def build():
  torch.manual_seed(17)
  model=torchvision.models.efficientnet_b1(weights=torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V2)
  model.classifier[1]=torch.nn.Linear(model.classifier[1].in_features,4)
  weights_file=Path(torch.hub.get_dir())/'checkpoints/efficientnet_b1-c27df63c.pth'
  write_json(out/'backbone_identity.json',{'weights_enum':'IMAGENET1K_V2','weights_sha256':hashlib.sha256(weights_file.read_bytes()).hexdigest(),'pretraining':'ImageNet; not urgency supervision'})
  return model.to(device)
 def budget():
  if time.monotonic()-started>seconds:raise TimeoutError('bounded CNN experiment time budget reached')
 def evaluate(model,subset):
  model.eval();prob=[];truth=[];losses=[]
  with torch.inference_mode():
   for xb,yb in loader(subset):
    budget();xb=xb.to(device);yb=yb.to(device);logits=model(xb)
    losses.extend(torch.nn.functional.cross_entropy(logits,yb,reduction='none').cpu().tolist());prob.extend(logits.softmax(1).cpu().tolist());truth.extend(yb.cpu().tolist())
  report=metrics(np.array(prob),np.array(truth));report['loss']=float(np.mean(losses));return report,prob
 def fit_model(model,subset,epochs,early_subset=None):
  counts=np.bincount([LEVELS.index(lookup[r['image_id']]['level']) for r in subset],minlength=4)
  if np.any(counts==0):raise ValueError('missing fitting class')
  weights=torch.tensor(len(subset)/(4*counts),device=device,dtype=torch.float32)
  for p in model.parameters():p.requires_grad=False
  for p in model.classifier.parameters():p.requires_grad=True
  optimizer=torch.optim.AdamW(model.classifier.parameters(),lr=1e-3,weight_decay=1e-4)
  history=[];best=(-1,float('-inf'));best_epoch=1;stale=0
  for epoch in range(epochs):
   budget()
   if epoch==3:
    for p in model.features[6:].parameters():p.requires_grad=True
    optimizer=torch.optim.AdamW([{'params':model.features[6:].parameters(),'lr':1e-5},{'params':model.classifier.parameters(),'lr':1e-4}],weight_decay=1e-4)
   model.train()
   for m in model.modules():
    if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
   for xb,yb in loader(subset,True):
    budget();optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(xb.to(device)),yb.to(device),weight=weights)
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
    loss.backward();optimizer.step()
   train,_=evaluate(model,subset);entry={'epoch':epoch+1,'train':train,'elapsed_seconds':time.monotonic()-started}
   if early_subset is not None:
    valid,_=evaluate(model,early_subset);entry['inner_validation']=valid;key=(valid['macro_f1'],-valid['loss'])
    if key>best:best=key;best_epoch=epoch+1;stale=0
    else:stale+=1
   history.append(entry);write_json(out/('selection_history.json' if early_subset is not None else 'refit_history.json'),history)
   print('CNN epoch',epoch+1,'trainF1',train['macro_f1'],'innerF1',entry.get('inner_validation',{}).get('macro_f1'),flush=True)
   if early_subset is not None and stale>=6 and epoch>=3:break
  return best_epoch,history
 try:
  model=build();fit_rows=[training[i] for i in fit];stop_rows=[training[i] for i in stop]
  selected,history=fit_model(model,fit_rows,max_epochs,stop_rows)
  del model;torch.cuda.empty_cache()
  model=build();fit_model(model,training,selected)
  report,prob=evaluate(model,validation)
  torch.save({'state_dict':model.cpu().state_dict(),'model':'efficientnet_b1','class_order':list(LEVELS),'selected_epochs':selected},out/'candidate.pt')
  write_json(out/'validation.json',{'student':report,'selected_epochs':selected,'runtime_promotion_allowed':False,'claim':'development teacher agreement only'})
  write_json(out/'predictions.json',[{'image_id':r['image_id'],'truth':lookup[r['image_id']]['level'],'probabilities':p} for r,p in zip(validation,prob)])
  write_json(out/'status.json',{'status':'complete','elapsed_seconds':time.monotonic()-started,'all_class_recall_gate':all(report['per_class'][k]['recall']>=.7 for k in LEVELS),'runtime_promotion_allowed':False})
 finally:
  if 'model' in locals():del model
  if torch.cuda.is_available():torch.cuda.empty_cache()
 return out


def run(records,labels,output,device='cuda:0',seconds=1500,max_epochs=20):
    """Persist failure evidence even when preflight/splitting/model loading fails."""
    import traceback
    if not 1<=seconds<=1500 or not 1<=max_epochs<=20:raise ValueError('experiment budget exceeds approved bounds')
    Path(output).mkdir(parents=True,exist_ok=False)  # Outside handler: never overwrite a prior run.
    try:
        result=_run(records,labels,output,device,seconds,max_epochs)
        plot_run(result)
        return result
    except BaseException as error:
        path=Path(output);path.mkdir(parents=True,exist_ok=True)
        write_json(path/'status.json',{'status':'failed','error':type(error).__name__,
            'reason':str(error),'runtime_promotion_allowed':False})
        (path/'traceback.txt').write_text(traceback.format_exc())
        raise


def plot_run(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    output=Path(output)
    history=json.loads((output/'selection_history.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for key,label in [('train','Fitting groups'),('inner_validation','Early-stop groups')]:
        for ax,metric in zip(axes,('loss','accuracy','macro_f1')):
            ax.plot([r['epoch'] for r in history],[r[key][metric] for r in history],label=label)
            ax.set(xlabel='Epoch',ylabel=metric);ax.legend()
    fig.suptitle('EfficientNet-B1: training-only groups; outer 96 not used here')
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for name in LEVELS:ax.plot([r['epoch'] for r in history],[r['inner_validation']['per_class'][name]['recall'] for r in history],label=name)
    ax.set(xlabel='Epoch',ylabel='Inner recall',ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(output/'recall_curves.png',dpi=160);plt.close(fig)
    report=json.loads((output/'validation.json').read_text())['student']
    fig,ax=plt.subplots(figsize=(6,5));ConfusionMatrixDisplay(np.array(report['confusion_true_rows_predicted_columns']),display_labels=LEVELS).plot(ax=ax,colorbar=False)
    ax.set_title('EfficientNet-B1: exposed development validation')
    fig.tight_layout();fig.savefig(output/'confusion_matrix.png',dpi=160);plt.close(fig)
