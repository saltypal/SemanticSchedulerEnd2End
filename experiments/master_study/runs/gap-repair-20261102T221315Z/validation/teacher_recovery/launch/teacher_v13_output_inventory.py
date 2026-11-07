import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
out=Path("/mnt/c/Users/satya/Documents/SemanticSchedulerArtifacts/paired-20261102/teacher_v13_output_inventory.json")
api=KaggleApi(); api.authenticate(); token=None; pages=0; items=[]
with api.build_kaggle_client() as client:
 while pages<20:
  req=ApiListKernelSessionOutputRequest(); req.user_name="satyapaladuguplus"; req.kernel_slug="semantic-decoder-scheduler-20261030"; api._set_paging(req,200,token)
  response=client.kernels.kernels_api_client.list_kernel_session_output(req); pages+=1; items.extend(response.files or []); token=response.next_page_token
  if not token: break
run="teacher_recovery-20261102T224440Z"
names=[str(x.file_name) for x in items if run in str(x.file_name)]
result={"kernel":"satyapaladuguplus/semantic-decoder-scheduler-20261030","requested_version_label":None,"pages":pages,"listed_files":len(items),"run_id":run,"run_id_path_hits":len(names),"paths":names,"urls_redacted":True}
out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2))
