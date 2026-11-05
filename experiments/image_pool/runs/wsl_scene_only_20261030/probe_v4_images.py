import json,requests
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
owner="satyapaladuguplus";slug="semantic-image-pool-scene-cache-20261030";version=4;run="experiments/image_pool/runs/20261030T105832Z/"
req=ApiDownloadKernelOutputRequest();req.owner_slug=owner;req.kernel_slug=slug;req.version_number=version;req.file_path=run+"data/input_records.json"
api=KaggleApi();api.authenticate()
with api.build_kaggle_client() as c:redir=c.kernels.kernels_api_client.download_kernel_output(req)
r=requests.get(redir.url,timeout=45);r.raise_for_status();data=r.json()
root=Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/image_pool/runs/wsl_scene_only_20261030/retrieved_v4_targeted/data");root.mkdir(parents=True,exist_ok=True);(root/"input_records.json").write_bytes(r.content)
rows=data if isinstance(data,list) else data.get("records",[])
results=[]
for row in rows[:5]:
 p=str(row.get("image_path",""));rel=p.replace("/kaggle/working/","").lstrip("/")
 q=ApiDownloadKernelOutputRequest();q.owner_slug=owner;q.kernel_slug=slug;q.version_number=version;q.file_path=rel
 try:
  with api.build_kaggle_client() as c2:rr=c2.kernels.kernels_api_client.download_kernel_output(q)
  h=requests.head(rr.url,timeout=20,allow_redirects=True);results.append({"image_id":row.get("image_id"),"path":rel,"status":h.status_code,"content_length":h.headers.get("content-length")})
 except Exception as e:results.append({"image_id":row.get("image_id"),"path":rel,"error_type":type(e).__name__,"status_code":getattr(getattr(e,"response",None),"status_code",None)})
print(json.dumps({"version":4,"records":len(rows),"input_records_bytes":len(r.content),"samples":results},indent=2))
