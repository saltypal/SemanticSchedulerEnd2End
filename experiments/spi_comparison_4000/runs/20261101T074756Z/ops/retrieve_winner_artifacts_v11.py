import hashlib,json,os,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
ops=Path(r'D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End\experiments\spi_comparison_4000\runs\20261101T074756Z\ops')
dest=ops/'remote_best_dev_candidate';dest.mkdir(parents=True,exist_ok=True)
owner='satyapaladugu';slug='crisismmd1000-spi-labeling';version=11
remote=[
 'experiments/spi_comparison_4000/runs/20261101T075007Z/comparison/original/clip_b32/single_query_cross_attention.pt',
 'experiments/spi_comparison_4000/runs/20261101T075007Z/comparison/features/clip_b32/identity.json',
]
api=KaggleApi();api.authenticate(); files=[]
for rel in remote:
 req=ApiDownloadKernelOutputRequest();req.owner_slug=owner;req.kernel_slug=slug;req.version_number=version;req.file_path=rel
 try:
  with api.build_kaggle_client() as client: response=client.kernels.kernels_api_client.download_kernel_output(req)
  url=response.url
  head=requests.head(url,timeout=(20,45),allow_redirects=True);head.raise_for_status()
  size=int(head.headers.get('content-length') or 0)
  if size<=0 or size>5*1024**3: raise RuntimeError(f'invalid declared artifact size={size}')
  free=shutil.disk_usage(dest).free
  if free<=size+1024**3: raise RuntimeError(f'insufficient disk reserve free={free}, size={size}')
  target=dest/Path(rel).name;partial=target.with_suffix(target.suffix+'.partial');h=hashlib.sha256();count=0
  with requests.get(url,stream=True,timeout=(30,120)) as r:
   r.raise_for_status()
   with partial.open('wb') as f:
    for chunk in r.iter_content(1024*1024):
     if not chunk: continue
     count+=len(chunk)
     if count>size: raise RuntimeError('download exceeded declared size')
     h.update(chunk);f.write(chunk)
    f.flush();os.fsync(f.fileno())
  if count!=size: raise RuntimeError(f'incomplete download {count}/{size}')
  partial.replace(target)
  files.append({'remote_path':rel,'local_path':str(target),'bytes':count,'sha256':h.hexdigest().upper(),'status':'retrieved'})
 except Exception as e:
  files.append({'remote_path':rel,'status':'unavailable','error_type':type(e).__name__,'error':str(e)})
result={'checked_utc':datetime.now(timezone.utc).isoformat(),'kernel':f'{owner}/{slug}/{version}','runtime_path':'experiments/spi_comparison_4000/runs/20261101T075007Z','selection_basis':'highest completed original-condition development macro-F1 row from the 36 available candidates; provisional only; locked test remains closed','files':files,'scope':'two exact named files only; no feature banks, checkpoints, image files, or full archive downloaded'}
(ops/'winner_artifact_retrieval_receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
if any(f['status']!='retrieved' for f in files):sys.exit(2)
