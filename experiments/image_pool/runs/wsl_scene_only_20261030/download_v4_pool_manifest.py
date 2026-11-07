import hashlib,json,requests
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
api=KaggleApi();api.authenticate();q=ApiDownloadKernelOutputRequest();q.owner_slug="satyapaladuguplus";q.kernel_slug="semantic-image-pool-scene-cache-20261030";q.version_number=4;q.file_path="experiments/image_pool/runs/20261030T105832Z/pool_manifest.json"
with api.build_kaggle_client() as c:redir=c.kernels.kernels_api_client.download_kernel_output(q)
r=requests.get(redir.url,timeout=45);r.raise_for_status();dst=Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/image_pool/runs/wsl_scene_only_20261030/retrieved_v4_targeted/pool_manifest.json");dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(r.content);print(json.dumps({"version":4,"remote_path":q.file_path,"http_status":r.status_code,"bytes":len(r.content),"sha256":hashlib.sha256(r.content).hexdigest().upper()}))
