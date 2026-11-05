import hashlib, json, os
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
root=Path("/mnt/c/Users/satya/Documents/SemanticSchedulerArtifacts/paired-20261102/teacher_v13_records")
root.mkdir(parents=True,exist_ok=True)
run="teacher_recovery-20261102T224440Z"
required={
 f"experiments/scheduler_pair/runs/{run}/RUN_STATUS.json",f"experiments/scheduler_pair/runs/{run}/experiment_scope.json",f"experiments/scheduler_pair/runs/{run}/input_manifest.json",f"experiments/scheduler_pair/runs/{run}/run_config.json",f"experiments/scheduler_pair/runs/{run}/logs/prepare_teacher.log",f"experiments/scheduler_pair/runs/{run}/logs/teacher.log",f"experiments/scheduler_pair/runs/{run}/reference_preflight/preflight.json",f"experiments/scheduler_pair/runs/{run}/sessions/20261102T224440Z/artifact_index.json",f"experiments/scheduler_pair/runs/{run}/sessions/20261102T224440Z/cells.json",f"experiments/scheduler_pair/runs/{run}/sessions/20261102T224440Z/environment.json",f"experiments/scheduler_pair/runs/{run}/teacher/provenance.json",f"experiments/scheduler_pair/runs/{run}/teacher/qualification.json",f"experiments/scheduler_pair/runs/{run}/teacher/responses.jsonl",f"experiments/scheduler_pair/runs/{run}/teacher_prepared/requests.jsonl"}
api=KaggleApi(); api.authenticate(); token=None; pages=0; items=[]
with api.build_kaggle_client() as client:
 while pages<20:
  req=ApiListKernelSessionOutputRequest(); req.user_name="satyapaladuguplus"; req.kernel_slug="semantic-decoder-scheduler-20261030"; api._set_paging(req,200,token)
  response=client.kernels.kernels_api_client.list_kernel_session_output(req); pages+=1; items.extend(response.files or []); token=response.next_page_token
  if not token: break
run_hits=[str(x.file_name) for x in items if run in str(x.file_name)]
if len(run_hits)!=len(items): raise SystemExit(f"Output paths are not uniquely this run: {len(run_hits)}/{len(items)}")
by_name={str(x.file_name):x for x in items}; missing=sorted(required-set(by_name))
if missing: raise SystemExit("Required evidence missing: "+", ".join(missing))
entries=[]; total=0
for name in sorted(required):
 item=by_name[name]; target=root.joinpath(*name.split("/")); target.parent.mkdir(parents=True,exist_ok=True); tmp=target.with_name(target.name+".partial"); h=hashlib.sha256(); size=0
 with requests.get(item.url,stream=True,allow_redirects=True,timeout=(30,120)) as resp:
  resp.raise_for_status(); cl=resp.headers.get("Content-Length")
  if cl and int(cl)>20_000_000: raise SystemExit(f"Selected file exceeds limit: {name} bytes={cl}")
  with tmp.open("wb") as f:
   for chunk in resp.iter_content(chunk_size=65536):
    if not chunk: continue
    size+=len(chunk); total+=len(chunk)
    if total>20_000_000: raise SystemExit("Evidence cap exceeded")
    f.write(chunk); h.update(chunk)
  if cl and size!=int(cl): raise SystemExit(f"Length mismatch {name}: {size} != {cl}")
 os.replace(tmp,target); entries.append({"file_name":name,"saved_path":str(target),"actual_bytes":size,"content_length":cl,"sha256":h.hexdigest(),"status":"http_200_streamed_verified"})
manifest={"kernel":"satyapaladuguplus/semantic-decoder-scheduler-20261030","run_id":run,"version":13,"requested_version_label":None,"run_paths_verified":len(run_hits),"output_file_count":len(items),"selected_count":len(entries),"actual_bytes_total":total,"files":entries,"urls_redacted":True}
(root/"retrieval_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8"); print(json.dumps(manifest,indent=2))
