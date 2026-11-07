import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
out=Path(r"C:\Users\satya\Documents\SemanticSchedulerArtifacts\paired-20261102\historical_state\windows_v1_verified\version_label_probe.json")
api=KaggleApi(); api.authenticate(); token=None; pages=0; n=0; names=[]
try:
 with api.build_kaggle_client() as client:
  while pages<20:
   req=ApiListKernelSessionOutputRequest(); req.user_name="satyapaladugu"; req.kernel_slug="semantic-decoder-scheduler-rl"; req.version_label="1"; api._set_paging(req,200,token)
   response=client.kernels.kernels_api_client.list_kernel_session_output(req); pages+=1
   for item in response.files or []: n+=1; names.append(str(item.file_name))
   token=response.next_page_token
   if not token: break
 result={"version_label":"1","pages":pages,"count":n,"run_id_hits":sum("windows-20261102T154257Z" in x for x in names),"paths":names[:8],"status":"ok","urls_redacted":True}
except Exception as exc:
 result={"version_label":"1","error_type":type(exc).__name__,"error":str(exc).split("url:")[0].strip(),"urls_redacted":True}
out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2))
