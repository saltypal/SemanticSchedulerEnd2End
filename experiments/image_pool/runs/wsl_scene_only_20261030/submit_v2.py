import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import nbformat
from kaggle.api.kaggle_api_extended import KaggleApi
ops=Path(__file__).resolve().parent; stage=ops/'staging_v2'; notebook=stage/'Image_Pool_Caching.ipynb'; metadata=stage/'kernel-metadata.json'
expected='90246C36723F83D644B5265A305A58570A11017BD91966BCC45A2688B67AFB9A'; slug='satyapaladuguplus/semantic-image-pool-scene-cache-20261030'
b=notebook.read_bytes(); actual=hashlib.sha256(b).hexdigest().upper(); n=nbformat.read(notebook,as_version=4); nbformat.validate(n); first=next(c for c in n.cells if c.cell_type=='code'); compile(first.source,'<setup>','exec'); md=json.loads(metadata.read_text(encoding='utf-8'))
if actual!=expected or len(b)!=339986 or len(n.cells)!=36: raise SystemExit('staged v2 changed after source-ready check')
if md.get('id')!=slug or not md.get('is_private') or not first.source.startswith("import os,sys,time,json,hashlib,subprocess\nos.environ['IMAGE_POOL_SCENE_ONLY']='1'\n"): raise SystemExit('v2 scene-only identity/setup check failed')
api=KaggleApi();api.authenticate(); response=api.kernels_push(str(stage),timeout='24000')
receipt={'kernel':slug,'kernel_id':response.kernel_id,'version_number':response.version_number,'ref':response.ref,'url':response.url,'error':response.error,'invalid_tags':response.invalid_tags,'invalid_dataset_sources':response.invalid_dataset_sources,'invalid_competition_sources':response.invalid_competition_sources,'invalid_kernel_sources':response.invalid_kernel_sources,'invalid_model_sources':response.invalid_model_sources,'notebook_sha256':actual,'notebook_bytes':len(b),'cell_count':len(n.cells),'scene_only':True,'timeout_seconds':24000,'submitted_utc':datetime.now(timezone.utc).isoformat()}
(ops/'submission_v2_sdk_success.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
