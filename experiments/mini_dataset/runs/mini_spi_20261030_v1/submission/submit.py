from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import nbformat
from kaggle.api.kaggle_api_extended import KaggleApi
stage=Path(__file__).resolve().parent; nbfile=stage/"Mini_Dataset.ipynb"; slug="satyapaladuguplus/semantic-image-pool-scene-cache-20261030"
b=nbfile.read_bytes(); sha=hashlib.sha256(b).hexdigest().upper(); nb=nbformat.read(nbfile,as_version=4); nbformat.validate(nb)
setup=next(c for c in nb.cells if c.cell_type=="code"); compile(setup.source,"<mini-setup>","exec")
md=json.loads((stage/"kernel-metadata.json").read_text(encoding="utf-8"))
if md.get("id")!=slug or md.get("is_private") is not True: raise SystemExit("identity/privacy mismatch")
if md.get("dataset_sources")!=["satyapaladuguplus/mini-spi-training-labels"]: raise SystemExit("managed inputs mismatch")
if len(nb.cells)!=38 or len(b)!=916276 or sha!="0F3C647689F5017B0E9FB60F0376D43B278B4075A88B761FE2F5F0C69F27DB4B": raise SystemExit("canonical source changed after staging")
if "SCENE_ONLY=False" not in "".join(nb.cells[2].source): raise SystemExit("unexpected scene-only mode")
api=KaggleApi();api.authenticate()
try:
 response=api.kernels_push(str(stage),timeout="24000")
 rec={"kernel":slug,"kernel_id":response.kernel_id,"version_number":response.version_number,"ref":response.ref,"url":response.url,"error":response.error,"invalid_tags":response.invalid_tags,"invalid_dataset_sources":response.invalid_dataset_sources,"invalid_competition_sources":response.invalid_competition_sources,"invalid_kernel_sources":response.invalid_kernel_sources,"notebook_sha256":sha,"notebook_bytes":len(b),"cell_count":len(nb.cells),"dataset_sources":md["dataset_sources"],"timeout_seconds":24000,"submitted_utc":datetime.now(timezone.utc).isoformat()}
 (stage/"submission_receipt.json").write_text(json.dumps(rec,indent=2)+"\n",encoding="utf-8");print(json.dumps(rec,indent=2))
except Exception as e:
 resp=getattr(e,"response",None);rec={"kernel":slug,"error_type":type(e).__name__,"status_code":getattr(resp,"status_code",None),"response_body":getattr(resp,"text",None),"notebook_sha256":sha,"notebook_bytes":len(b),"failed_utc":datetime.now(timezone.utc).isoformat()}
 (stage/"submission_error.json").write_text(json.dumps(rec,indent=2)+"\n",encoding="utf-8");print(json.dumps(rec,indent=2));raise
