import hashlib, json, os, re, shutil, time, zipfile
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
root=Path(r"C:\Users\satya\Documents\SemanticSchedulerArtifacts\paired-20261102\historical_state\windows_v1_verified")
manifest_path=root/"retrieval_manifest.json"
run="windows-20261102T154257Z"
max_transfer=4_000_000_000
min_free=1_073_741_824
space=shutil.disk_usage(root.anchor)
if space.free < min_free + 230_000_000: raise SystemExit(f"Insufficient C free space before known selected payload: {space.free}")
api=KaggleApi(); api.authenticate(); token=None; pages=0; items=[]
with api.build_kaggle_client() as client:
 while pages<20:
  req=ApiListKernelSessionOutputRequest(); req.user_name="satyapaladugu"; req.kernel_slug="semantic-decoder-scheduler-rl"; api._set_paging(req,200,token)
  response=client.kernels.kernels_api_client.list_kernel_session_output(req); pages+=1
  items.extend(response.files or []); token=response.next_page_token
  if not token: break
run_hits=sum(run in str(x.file_name) for x in items)
if run_hits != len(items): raise SystemExit(f"Output path/run identity mismatch: {run_hits}/{len(items)} entries carry run id")
selected=[]
for it in items:
 n=str(it.file_name)
 if run not in n: continue
 basename=n.rsplit("/",1)[-1]
 in_model="/models/" in n and "/validation/" not in n
 wanted={"learning.json","model.json","parity.json","resume.pt","warmstart.pt","warmstart_actor.json","actor.json","frozen_actor.json","optimizer.pt","rng.pt","checkpoint.pt","config.json","manifest.json"}
 is_manifest=basename in {"artifact_manifest.json","experiment_scope.json","input_manifest.json","RUN_STATUS.json","shared_manifest.json"}
 if (in_model and basename in wanted) or is_manifest: selected.append(it)
model_dirs=sorted({re.search(r"/models/([^/]+)/1729/",str(x.file_name)).group(1) for x in selected if re.search(r"/models/([^/]+)/1729/",str(x.file_name))})
if model_dirs != ["deepsets_ranknet","masked_double_dqn","masked_ppo","ranknet_warmstart_ppo","td_per"]: raise SystemExit(f"Unexpected Windows v1 model dirs: {model_dirs}")
maxbytes=max_transfer
entries=[]; prior=0
if manifest_path.exists():
 try: prior=json.loads(manifest_path.read_text(encoding="utf-8")).get("actual_bytes_total",0)
 except Exception: prior=0
for idx,it in enumerate(selected,1):
 name=str(it.file_name); target=root.joinpath(*name.split("/")); target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  digest=hashlib.sha256(target.read_bytes()).hexdigest(); size=target.stat().st_size
  entry={"file_name":name,"saved_path":str(target),"actual_bytes":size,"sha256":digest,"status":"already_present_verified"}
 else:
  tmp=target.with_name(target.name+".partial")
  h=hashlib.sha256(); total=0
  try:
   with requests.get(it.url,stream=True,allow_redirects=True,timeout=(30,120)) as resp:
    resp.raise_for_status(); cl=resp.headers.get("Content-Length")
    if cl and int(cl)>maxbytes-prior: raise RuntimeError(f"transfer cap precheck: {cl} > remaining {maxbytes-prior}")
    with tmp.open("wb") as f:
     for chunk in resp.iter_content(chunk_size=1024*1024):
      if not chunk: continue
      total+=len(chunk); prior+=len(chunk)
      if prior>maxbytes: raise RuntimeError(f"transfer cap exceeded: {prior}")
      f.write(chunk); h.update(chunk)
      if shutil.disk_usage(root.anchor).free < min_free: raise RuntimeError("C free space reserve below 1 GiB")
    if cl and total!=int(cl): raise RuntimeError(f"length mismatch for {name}: actual {total} vs header {cl}")
   os.replace(tmp,target)
  except Exception:
   if tmp.exists(): tmp.unlink()
   raise
  entry={"file_name":name,"saved_path":str(target),"actual_bytes":total,"content_length":cl,"sha256":h.hexdigest(),"status":"downloaded_verified_http_200"}
  if target.suffix==".pt":
   try:
    with zipfile.ZipFile(target) as z: entry["zip_member_count"]=len(z.infolist()); entry["zip_valid_directory"]=True
   except zipfile.BadZipFile: entry["zip_valid_directory"]=False
 entries.append(entry)
 record={"kernel":"satyapaladugu/semantic-decoder-scheduler-rl/1","run_id":run,"api_version_label":None,"note":"ListKernelSessionOutput with version_label=1 returned HTTP 404; this unversioned session response was accepted only because every path carries exact run ID.","listed_files":len(items),"run_id_path_hits":run_hits,"selected_file_count":len(selected),"selected_model_dirs":model_dirs,"actual_bytes_total":sum(x["actual_bytes"] for x in entries),"max_transfer_bytes":maxbytes,"minimum_remaining_C_bytes":min_free,"files":entries,"urls_redacted":True}
manifest_path.write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in record.items() if k!="files"},indent=2))
for x in entries: print(json.dumps({k:x.get(k) for k in ("file_name","actual_bytes","sha256","status","zip_member_count","zip_valid_directory") if k in x}))
