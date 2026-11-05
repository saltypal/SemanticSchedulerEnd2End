import json
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
base=Path("/mnt/c/Users/satya/Documents/SemanticSchedulerArtifacts/paired-20261102/historical_state/wsl_v12_versioned")
base.mkdir(parents=True,exist_ok=True)
api=KaggleApi(); api.authenticate(); token=None; pages=0; items=[]
with api.build_kaggle_client() as client:
 while pages<20:
  req=ApiListKernelSessionOutputRequest(); req.user_name="satyapaladuguplus"; req.kernel_slug="semantic-decoder-scheduler-20261030"; req.version_label="12"; api._set_paging(req,200,token)
  response=client.kernels.kernels_api_client.list_kernel_session_output(req); pages+=1
  items.extend(response.files or []); token=response.next_page_token
  if not token: break
run="wsl-20261102T162419Z"
run_hits=sum(run in str(x.file_name) for x in items)
selected=[]
for it in items:
 n=str(it.file_name)
 if run not in n: continue
 basename=n.rsplit("/",1)[-1]
 in_model="/models/" in n and "/validation/" not in n
 wanted={"learning.json","model.json","parity.json","resume.pt","warmstart.pt","warmstart_actor.json","actor.json","frozen_actor.json","optimizer.pt","rng.pt","checkpoint.pt","config.json","manifest.json"}
 is_manifest=basename in {"artifact_manifest.json","experiment_scope.json","input_manifest.json","RUN_STATUS.json","shared_manifest.json"}
 if (in_model and basename in wanted) or is_manifest: selected.append(it)
results=[]
for it in selected:
 n=str(it.file_name); rec={"file_name":n,"status":None,"content_length":None,"content_type":None}
 try:
  resp=requests.get(it.url,stream=True,allow_redirects=True,timeout=45)
  rec["status"]=resp.status_code; rec["content_length"]=resp.headers.get("Content-Length"); rec["content_type"]=resp.headers.get("Content-Type"); resp.close()
 except requests.RequestException as exc: rec["error"]=type(exc).__name__
 results.append(rec)
summary={"kernel":"satyapaladuguplus/semantic-decoder-scheduler-20261030/12","requested_version_label":"12","run_id":run,"pages":pages,"listed_files":len(items),"run_id_path_hits":run_hits,"selected_file_count":len(selected),"selected_content_length_sum":sum(int(x["content_length"]) for x in results if x.get("content_length")),"all_status_200":all(x.get("status")==200 for x in results),"files":results,"urls_redacted":True}
(base/"selected_content_headers.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k!="files"},indent=2))
for x in results: print(json.dumps({k:x.get(k) for k in ("file_name","status","content_length","content_type","error") if k in x}))
