"""Frozen image-only encoders: features and CPU/GPU inference resource profiles."""
from pathlib import Path
import argparse,hashlib,json,io,time,zipfile,platform,importlib.metadata
import numpy as np
import torch
from PIL import Image,ImageOps
from parameters.artifacts import write_json
from experiments.spi_classifier_repair.head_matrix import MemorySample

ENCODERS={
 'clip_b32':('openai/clip-vit-base-patch32','clip'),
 'siglip_b16':('google/siglip-base-patch16-224','siglip'),
 'siglip2_b16':('google/siglip2-base-patch16-224','siglip'),
 'dinov2_small':('facebook/dinov2-small','cls'),
 'dinov2_base':('facebook/dinov2-base','cls'),
 'ijepa_h14':('facebook/ijepa_vith14_1k','mean'),
 'mobilenet_v3_small':(None,'mobile'),
 'efficientnet_b0':(None,'efficient')}

class ImageEncoder(torch.nn.Module):
 def __init__(self,core,kind,projection=None):
  super().__init__();self.core=core;self.kind=kind;self.projection=projection
 def forward(self,x):
  if self.kind in ('mobile','efficient'):
   result=self.core(x).mean((2,3))
  else:
   output=self.core(pixel_values=x)
   if self.kind in ('clip','siglip'):result=output.pooler_output
   elif self.kind=='cls':result=output.last_hidden_state[:,0]
   else:result=output.last_hidden_state.mean(1)
   if self.projection is not None:result=self.projection(result)
  return torch.nn.functional.normalize(result.float(),dim=-1)

def load(name,revisions):
 import torchvision
 repo,kind=ENCODERS[name]
 identity={'name':name,'repo':repo,'kind':kind,'dtype':'float32','input_shape':[1,3,224,224]}
 if repo:
  from huggingface_hub import snapshot_download
  from transformers import AutoModel,AutoImageProcessor
  revision=revisions[name];identity['revision']=revision
  path=Path(snapshot_download(repo,revision=revision,allow_patterns=['config.json','preprocessor_config.json','processor_config.json','*.safetensors','*.safetensors.index.json']))
  processor=AutoImageProcessor.from_pretrained(path,local_files_only=True,use_fast=False)
  complete=AutoModel.from_pretrained(path,local_files_only=True).eval()
  if kind in ('clip','siglip'):
   model=ImageEncoder(complete.vision_model,kind,complete.visual_projection if kind=='clip' else None)
  else:model=ImageEncoder(complete,kind)
  del complete
  identity['download_checkpoint_bytes']=sum(p.stat().st_size for p in path.glob('*.safetensors'))
  identity['processor']=processor.to_dict()
  def preprocess(images):
   kwargs={'do_resize':False}
   if hasattr(processor,'do_center_crop'):kwargs['do_center_crop']=False
   return processor(images=images,return_tensors='pt',**kwargs)['pixel_values']
 else:
  from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights,efficientnet_b0,EfficientNet_B0_Weights
  if kind=='mobile':weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1;complete=mobilenet_v3_small(weights=weights)
  else:weights=EfficientNet_B0_Weights.IMAGENET1K_V1;complete=efficientnet_b0(weights=weights)
  model=ImageEncoder(complete.features,kind);del complete
  identity.update(weights_enum=str(weights),weight_url=weights.url)
  weight_path=Path(torch.hub.get_dir())/'checkpoints'/Path(weights.url).name
  h=hashlib.sha256()
  with weight_path.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  identity.update(weights_sha256=h.hexdigest(),download_checkpoint_bytes=weight_path.stat().st_size)
  def preprocess(images):
   from torchvision.transforms import functional as F
   return torch.stack([F.normalize(F.to_tensor(im),[.485,.456,.406],[.229,.224,.225]) for im in images])
 identity['parameter_count']=sum(p.numel() for p in model.parameters())
 identity['parameter_bytes']=sum(p.numel()*p.element_size() for p in model.parameters())
 identity['buffer_bytes']=sum(p.numel()*p.element_size() for p in model.buffers())
 identity['feature_policy']='fullframe RGB EXIF-upright bicubic pad224 grey127; L2 normalized'
 return model.eval(),preprocess,identity

def pad(image):return ImageOps.pad(ImageOps.exif_transpose(image).convert('RGB'),(224,224),method=Image.Resampling.BICUBIC,color=(127,127,127))
def sync(device):
 if str(device).startswith('cuda'):torch.cuda.synchronize(device)

def profile(model,x,device,repeats=20):
 model=model.to(device);x=x.to(device)
 with torch.inference_mode():
  for _ in range(5):model(x)
  sync(device)
  if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
  times=[]
  with MemorySample() as memory:
   for _ in range(repeats):
    sync(device);start=time.perf_counter_ns();model(x);sync(device);times.append((time.perf_counter_ns()-start)/1e6)
 result={'device':str(device),'p50_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),'raw_ms':times,
  'batch':1,'warmup':5,'scope':'encoder tensor forward only; exclude preprocessing/load/transfer','process_peak_rss_bytes':memory.peak}
 if str(device).startswith('cuda'):result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
 return result

def worker(name,input_root,output,revisions,seconds=500):
 torch.set_num_threads(1);out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
 from parameters.spi_campaign import verify_inputs
 rows,input_receipt=verify_inputs(input_root);write_json(out/'input_receipt.json',input_receipt)
 def budget():
  if time.monotonic()-started>seconds:raise TimeoutError('per-encoder budget reached')
 try:
  model,preprocess,identity=load(name,revisions);write_json(out/'encoder_identity.json',identity)
  load_seconds=time.monotonic()-started
  archive=Path(input_root)/'images.zip';features=[];costs=[]
  zipped=zipfile.ZipFile(archive) if archive.exists() else None
  def read(row):
   raw=zipped.read(row['image_path']) if zipped else (Path(input_root)/'images'/row['image_path']).read_bytes()
   if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('source image hash differs')
   with Image.open(io.BytesIO(raw)) as im:return pad(im)
  example=preprocess([read(rows[0])]);budget()
  gpu=profile(model,example,'cuda:0');budget()
  model=model.to('cuda:0')
  with torch.inference_mode():
   for start in range(0,len(rows),8 if name!='ijepa_h14' else 2):
    budget();batch=rows[start:start+(8 if name!='ijepa_h14' else 2)]
    t=time.perf_counter();tensor=preprocess([read(row) for row in batch]);prep=time.perf_counter()-t
    sync('cuda:0');t=time.perf_counter();tensor=tensor.to('cuda:0');value=model(tensor).cpu().numpy();sync('cuda:0')
    costs.append({'start':start,'count':len(batch),'load_prepare_seconds':prep,'forward_transfer_seconds':time.perf_counter()-t})
    if len(value)!=len(batch) or not np.isfinite(value).all():raise ValueError('invalid encoder features')
    features.extend(value);print(name,len(features),'/',len(rows),flush=True)
  if zipped:zipped.close()
  identity['feature_dimension']=np.array(features).shape[1]
  signature=hashlib.sha256(json.dumps(identity,sort_keys=True,default=str).encode()).hexdigest()
  np.savez_compressed(out/'features.npz',embeddings=np.array(features,np.float32),image_ids=np.array([r['image_id'] for r in rows]),image_sha256=np.array([r['image_sha256'] for r in rows]),feature_signature=np.array(signature))
  write_json(out/'extraction_timings.json',costs)
  model=model.cpu();torch.cuda.empty_cache();budget();cpu=profile(model,example,'cpu',repeats=10)
  from torch.profiler import profile as torch_profile,ProfilerActivity
  with torch.inference_mode(),torch_profile(activities=[ProfilerActivity.CPU],with_flops=True) as trace:model(example)
  events=trace.key_averages();flops=sum(e.flops for e in events)
  flop_record={'supported_operator_flops':int(flops) if flops else None,'convention':'2 FLOPs/multiply-add where profiler supports it','status':'partial supported matrix/conv operator count; not complete model FLOPs','ops':[e.key for e in events if 'scaled_dot_product' in e.key or 'attention' in e.key]}
  write_json(out/'profile.json',{'identity':identity,'cpu':cpu,'gpu':gpu,'flops':flop_record,'load_and_verify_seconds':load_seconds,'environment':{'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'packages':{p:importlib.metadata.version(p) for p in ('torch','torchvision','transformers','numpy')},'gpu':torch.cuda.get_device_name(0)},'runtime_promotion_allowed':False})
  write_json(out/'status.json',{'status':'complete','seconds':time.monotonic()-started})
 except BaseException as error:
  import traceback
  write_json(out/'status.json',{'status':'failed','error':type(error).__name__,'reason':str(error),'seconds':time.monotonic()-started});(out/'traceback.txt').write_text(traceback.format_exc());raise

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--name',choices=ENCODERS,required=True);parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);parser.add_argument('--revisions',required=True);args=parser.parse_args();worker(args.name,args.input,args.output,json.loads(Path(args.revisions).read_text()))
if __name__=='__main__':main()
