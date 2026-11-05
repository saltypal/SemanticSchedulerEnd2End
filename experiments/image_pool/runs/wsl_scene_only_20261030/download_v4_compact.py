import datetime,hashlib,json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
owner="satyapaladuguplus";slug="semantic-image-pool-scene-cache-20261030";version=4
run="experiments/image_pool/runs/20261030T105832Z/"
files=["ACCEPTANCE.json","artifact_manifest.json","source_status.json","source.json","environment.json","config.json","scene_producer.json","scene/progress.json","scene/summary.json","scene/scene_records.jsonl","scene/scene_cache.sqlite","scene/scheduler_arrays.npz","scene/cache_identity.json","scene/registry.json","scene/environment.json","scene/artifact_manifest.json","scene/failures.json","data/download_budget.json","data/reference_records.json","handoff/image_parameters.jsonl"]
base=Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/image_pool/runs/wsl_scene_only_20261030/retrieved_v4_targeted")
api=KaggleApi();api.authenticate();records=[];total=0
with api.build_kaggle_client() as c:
 for rel in files:
  remote=run+rel;req=ApiDownloadKernelOutputRequest();req.owner_slug=owner;req.kernel_slug=slug;req.version_number=version;req.file_path=remote
  redir=c.kernels.kernels_api_client.download_kernel_output(req);r=requests.get(redir.url,timeout=45);r.raise_for_status();payload=r.content
  if len(payload)>50_000_000 or total+len(payload)>100_000_000:raise RuntimeError("selected retrieval exceeds size bound")
  out=base/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(payload);total+=len(payload)
  records.append({"remote_path":remote,"local_path":str(out),"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest().upper(),"http_status":r.status_code})
receipt={"owner":owner,"kernel":slug,"version":version,"runtime_path":run,"retrieved_files":records,"file_count":len(records),"retrieved_bytes":total,"excluded_images_and_weights":True,"observed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
(base/"retrieval_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8");print(json.dumps(receipt,indent=2))
