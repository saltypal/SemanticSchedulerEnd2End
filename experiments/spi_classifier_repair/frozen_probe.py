"""Frozen local SigLIP probe against cached CLIP; same training-only head selection."""
from pathlib import Path
import hashlib,io,json,time
import numpy as np
from PIL import Image,ImageOps
import torch
from transformers import AutoModel,AutoImageProcessor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from parameters.spi import prepare,metrics
from parameters.spi_model_selection import candidate_fit,predict
from parameters.artifacts import write_json
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'experiments/parameter_finalization/runs/notebook9_20261028/submission_v13_features/retrieved_version13/experiments/parameter_finalization/runs/notebook9_20261028_v13/data_run'

def main():
 import zipfile,importlib.metadata as md
 out=ROOT/'experiments/spi_classifier_repair/runs'/('frozen_siglip_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True)
 start=time.monotonic();torch.set_num_threads(2)
 checkpoint=ROOT/'models/priority_classifiers/siglip'
 processor=AutoImageProcessor.from_pretrained(checkpoint,local_files_only=True)
 model=AutoModel.from_pretrained(checkpoint,local_files_only=True).eval()
 panel=json.loads((ROOT/'parameters/results/spi_inputs_v2_20261027b/panel.json').read_text());vectors=[];timings=[]
 with zipfile.ZipFile(ROOT/'parameters/results/spi_inputs_v2_20261027b/images.zip') as archive:
  for start_index in range(0,len(panel),8):
   images=[];batch=panel[start_index:start_index+8]
   for row in batch:
    raw=archive.read(row['image_path'])
    if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('image identity differs')
    with Image.open(io.BytesIO(raw)) as im:images.append(ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'),(224,224),method=Image.Resampling.BICUBIC,color=(127,127,127)))
   inputs=processor(images=images,return_tensors='pt',do_resize=False)
   begin=time.perf_counter()
   with torch.inference_mode():
    feature=model.get_image_features(pixel_values=inputs['pixel_values'])
    if not torch.is_tensor(feature):feature=feature.pooler_output
    feature=torch.nn.functional.normalize(feature.float(),dim=-1)
   timings.append({'start_index':start_index,'images':len(batch),'forward_seconds':time.perf_counter()-begin,'device':'cpu','first_batch':start_index==0})
   value=feature.numpy()
   if value.shape[0]!=len(batch) or not np.isfinite(value).all():raise ValueError('invalid embedding batch')
   vectors.extend(value)
   print('SigLIP frozen features',len(vectors),'/',len(panel),flush=True)
 del model
 np.savez_compressed(out/'features.npz',embeddings=np.array(vectors),image_ids=np.array([r['image_id'] for r in panel]),image_sha256=np.array([r['image_sha256'] for r in panel]),feature_signature=np.array('experimental_siglip_fullframe224_local'))
 write_json(out/'timings.json',timings)
 labels=json.loads((BASE/'spi_training/labels.json').read_text());lookup={r['image_id']:r for r in labels};results={}
 with threadpool_limits(limits=1):
  for name,path in [('clip',BASE/'spi_training/features.npz'),('siglip',out/'features.npz')]:
   x,ids,ix,ys,identity=prepare(path,BASE/'spi_training/labels.json');xt=x[ix['train']];yt=np.array(ys['train']);groups=np.array([lookup[ids[i]]['group'] for i in ix['train']]);grid=[]
   for C in (.1,1.,10.):
    scores=[]
    for tr,va in StratifiedGroupKFold(3,shuffle=True,random_state=17).split(xt,yt,groups):
     assert not set(groups[tr])&set(groups[va]);head,_=candidate_fit({'family':'linear','C':C,'balanced':True},xt[tr],yt[tr],17)
     scores.append(float(f1_score(yt[va],predict(head,'linear',xt[va]).argmax(1),labels=range(4),average='macro',zero_division=0)))
    grid.append({'C':C,'fold_macro_f1':scores,'mean_macro_f1':float(np.mean(scores))})
   selected=max(grid,key=lambda r:(r['mean_macro_f1'],-r['C']));head,_=candidate_fit({'family':'linear','C':selected['C'],'balanced':True},xt,yt,17)
   results[name]={'grid':grid,'selected':selected,'student':metrics(predict(head,'linear',x[ix['validation']]),np.array(ys['validation']))}
   write_json(out/'comparison.json',results)
 write_json(out/'status.json',{'status':'complete','elapsed_seconds':time.monotonic()-start,'runtime_promotion_allowed':False,'reference':'exposed-development teacher agreement','packages':{k:md.version(k) for k in ('torch','transformers','scikit-learn')},'model_path':str(checkpoint),'downloaded_models':False})
 print('OUTPUT',out,flush=True)
 for name,r in results.items():print(name,r['student']['macro_f1'],{k:v['recall'] for k,v in r['student']['per_class'].items()},flush=True)
if __name__=='__main__':main()
