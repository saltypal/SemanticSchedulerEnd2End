from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import nbformat
from kaggle.api.kaggle_api_extended import KaggleApi
ops=Path(__file__).resolve().parent
stage=ops/'staging_v4'; nbfile=stage/'Image_Pool_Caching.ipynb'; slug='satyapaladuguplus/semantic-image-pool-scene-cache-20261030'
expected='DD51805844DC2B4B67D613F790795F6D8AEE202BB01C8FC0F1C4C15D553DADC2'
b=nbfile.read_bytes(); sha=hashlib.sha256(b).hexdigest().upper(); nb=nbformat.read(nbfile,as_version=4); nbformat.validate(nb)
setup=next(c for c in nb.cells if c.cell_type=='code'); compile(setup.source,'<scene-only-v4-setup>','exec')
md=json.loads((stage/'kernel-metadata.json').read_text(encoding='utf-8'))
if sha!=expected or len(b)!=909932 or len(nb.cells)!=36: raise SystemExit('staging-v4 integrity check failed')
if md.get('id')!=slug or md.get('is_private') is not True: raise SystemExit('Kaggle identity/privacy mismatch')
if "os.environ['IMAGE_POOL_SCENE_ONLY']='1'" not in setup.source or "if SCENE_ONLY:" not in setup.source: raise SystemExit('scene-only flag/dependency guard missing')
api=KaggleApi(); api.authenticate()
try:
 response=api.kernels_push(str(stage),timeout='24000')
 rec={'kernel':slug,'kernel_id':response.kernel_id,'version_number':response.version_number,'ref':response.ref,'url':response.url,'error':response.error,'invalid_tags':response.invalid_tags,'invalid_dataset_sources':response.invalid_dataset_sources,'invalid_competition_sources':response.invalid_competition_sources,'invalid_kernel_sources':response.invalid_kernel_sources,'invalid_model_sources':response.invalid_model_sources,'notebook_sha256':sha,'notebook_bytes':len(b),'cell_count':len(nb.cells),'scene_only':True,'timeout_seconds':24000,'submitted_utc':datetime.now(timezone.utc).isoformat()}
 (ops/'submission_v4_sdk_receipt.json').write_text(json.dumps(rec,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(rec,indent=2))
except Exception as e:
 resp=getattr(e,'response',None)
 rec={'kernel':slug,'error_type':type(e).__name__,'status_code':getattr(resp,'status_code',None),'response_body':getattr(resp,'text',None),'notebook_sha256':sha,'notebook_bytes':len(b),'failed_utc':datetime.now(timezone.utc).isoformat()}
 (ops/'submission_v4_sdk_error.json').write_text(json.dumps(rec,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(rec,indent=2)); raise
