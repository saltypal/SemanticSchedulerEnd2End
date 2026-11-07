import json, re, time
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
base=Path(r"C:\Users\satya\Documents\SemanticSchedulerArtifacts\paired-20261102\historical_state\windows_v1_verified")
api=KaggleApi(); api.authenticate(); token=None; pages=0; items=[]
with api.build_kaggle_client() as client:
 while pages<20:
  req=ApiListKernelSessionOutputRequest(); req.user_name="satyapaladugu"; req.kernel_slug="semantic-decoder-scheduler-rl"; api._set_paging(req,200,token)
  response=client.kernels.kernels_api_client.list_kernel_session_output(req); pages+=1
  items.extend(response.files or []); token=response.next_page_token
  if not token: break
run="windows-20261102T154257Z"
selected=[]
for it in items:
 n=str(it.file_name)
 if run not in n: continue
 if "/models/" in n and "/validation/" not in n and n.rsplit("/",1)[-1] in {"learning.json","model.json","parity.json","resume.pt","warmstart.pt"}:
  selected.append(it)
 elif n.endswith(("/artifact_manifest.json","/experiment_scope.json","/input_manifest.json","/RUN_STATUS.json","/shared/shared_manifest.json")):
  selected.append(it)
results=[]
for it in selected:
 n=str(it.file_name)
 rec={"file_name":n,"status":None,"content_length":None,"content_type":None}
 try:
  resp=requests.get(it.url,stream=True,allow_redirects=True,timeout=45)
  rec["status"]=resp.status_code; rec["content_length"]=resp.headers.get("Content-Length"); rec["content_type"]=resp.headers.get("Content-Type")
  resp.close()
 except requests.RequestException as exc:
  rec["error"]=type(exc).__name__
 results.append(rec)
summary={"kernel":"satyapaladugu/semantic-decoder-scheduler-rl/1","run_id":run,"pages":pages,"listed_files":len(items),"selected_file_count":len(selected),"selected_content_length_sum":sum(int(x["content_length"]) for x in results if x.get("content_length")),"all_status_200":all(x.get("status")==200 for x in results),"files":results,"urls_redacted":True}
(base/"selected_content_headers.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k!="files"},indent=2))
for x in results: print(json.dumps({k:x.get(k) for k in ("file_name","status","content_length","content_type","error") if k in x}))
