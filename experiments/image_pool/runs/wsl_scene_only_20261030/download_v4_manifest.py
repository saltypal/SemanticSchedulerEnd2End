import hashlib,json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
owner="satyapaladuguplus";slug="semantic-image-pool-scene-cache-20261030";version=4
run="experiments/image_pool/runs/20261030T105832Z/";remote=run+"artifact_manifest.json"
api=KaggleApi();api.authenticate();req=ApiDownloadKernelOutputRequest();req.owner_slug=owner;req.kernel_slug=slug;req.version_number=version;req.file_path=remote
with api.build_kaggle_client() as c:redir=c.kernels.kernels_api_client.download_kernel_output(req)
r=requests.get(redir.url,timeout=30);r.raise_for_status();payload=r.content
if len(payload)>1000000:raise SystemExit("manifest exceeds 1MB bound")
out=Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/image_pool/runs/wsl_scene_only_20261030/retrieved_v4_targeted/artifact_manifest.json");out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(payload)
rec={"version":version,"remote_path":remote,"local_path":str(out),"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest().upper(),"http_status":r.status_code}
print(json.dumps(rec))
