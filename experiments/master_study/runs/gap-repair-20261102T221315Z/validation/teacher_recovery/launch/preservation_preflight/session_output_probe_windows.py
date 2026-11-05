import hashlib, json, re
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
root = Path(r"C:\Users\satya\Documents\SemanticSchedulerArtifacts\paired-20261102\historical_state\windows_v1_verified")
root.mkdir(parents=True, exist_ok=True)
api=KaggleApi(); api.authenticate(); token=None; pages=0; chosen=None; listed=0
with api.build_kaggle_client() as client:
 while pages<20:
  req=ApiListKernelSessionOutputRequest(); req.user_name="satyapaladugu"; req.kernel_slug="semantic-decoder-scheduler-rl"; api._set_paging(req,200,token)
  response=client.kernels.kernels_api_client.list_kernel_session_output(req); pages+=1
  for item in (response.files or []):
   listed+=1; name=str(item.file_name)
   if ("windows-20261102T154257Z" in name and "/jobs/double_dqn/D/models/" in name and name.endswith("/model.json")):
    chosen=item; break
  if chosen is not None: break
  token=response.next_page_token
  if not token: break
if chosen is None: raise SystemExit(f"No model.json for exact Windows run; pages={pages} files_seen={listed}")
name=str(chosen.file_name); url=chosen.url
head=requests.head(url,allow_redirects=True,timeout=30)
meta={"status":head.status_code,"content_length":head.headers.get("Content-Length"),"content_type":head.headers.get("Content-Type"),"etag_present":bool(head.headers.get("ETag"))}; head.close()
if head.status_code not in (200, 404): raise SystemExit(f"model.json HEAD status={head.status_code}")
limit=10*1024*1024
if meta["content_length"] and int(meta["content_length"])>limit: raise SystemExit(f"portable model.json unexpectedly large: {meta['content_length']}")
resp=requests.get(url,stream=True,allow_redirects=True,timeout=60); resp.raise_for_status()
path=root.joinpath(*name.split("/")); path.parent.mkdir(parents=True,exist_ok=True)
h=hashlib.sha256(); total=0
with path.open("wb") as f:
 for chunk in resp.iter_content(chunk_size=65536):
  if not chunk: continue
  total+=len(chunk)
  if total>limit: raise SystemExit("download exceeded 10 MiB safety limit")
  h.update(chunk); f.write(chunk)
resp.close()
parsed=json.loads(path.read_text(encoding="utf-8"))
result={"kernel":"satyapaladugu/semantic-decoder-scheduler-rl/1","run_id":"windows-20261102T154257Z","listed_files_seen_until_selection":listed,"pages_scanned":pages,"file_name":name,"saved_path":str(path),"head":meta,"actual_bytes":total,"sha256":h.hexdigest(),"json_top_level_type":type(parsed).__name__,"json_keys":sorted(parsed.keys()) if isinstance(parsed,dict) else None,"urls_redacted":True}
(root/"probe_record.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2))

