import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
root=Path(r"C:\Users\satya\Documents\SemanticSchedulerArtifacts\paired-20261102\historical_state\windows_v1_verified")
api=KaggleApi(); api.authenticate(); token=None; pages=0; items=[]
with api.build_kaggle_client() as client:
 while pages<20:
  req=ApiListKernelSessionOutputRequest(); req.user_name="satyapaladugu"; req.kernel_slug="semantic-decoder-scheduler-rl"; api._set_paging(req,200,token)
  response=client.kernels.kernels_api_client.list_kernel_session_output(req); pages+=1
  items.extend(response.files or []); token=response.next_page_token
  if not token: break
paths=[str(x.file_name) for x in items if "windows-20261102T154257Z" in str(x.file_name) and "/jobs/" in str(x.file_name) and "/models/" in str(x.file_name)]
(root/"model_file_names.json").write_text(json.dumps({"run_id":"windows-20261102T154257Z","pages":pages,"files":len(items),"model_paths":paths,"urls_redacted":True},indent=2)+"\n",encoding="utf-8")
print(json.dumps({"pages":pages,"files":len(items),"model_path_count":len(paths),"model_paths":paths},indent=2))
