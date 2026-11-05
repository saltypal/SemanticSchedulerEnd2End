import hashlib,json,os,re,shutil,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

OWNER="satyapaladuguplus"; SLUG="semantic-image-pool-scene-cache-20261030"; VERSION=5
KERNEL=f"{OWNER}/{SLUG}/{VERSION}"
RUN=Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/mini_dataset/runs/mini_spi_20261030_v1")
OPS=RUN/"ops"; DOWNLOADS=RUN/"downloads"; OPS.mkdir(parents=True,exist_ok=True); DOWNLOADS.mkdir(parents=True,exist_ok=True)
STATE=OPS/"monitor_state.json"; LOG=OPS/"monitor.log"; MAX_HOURS=30; POLL_SECONDS=60; GIB=1024**3

def write_json(path,data):
 tmp=path.with_suffix(path.suffix+".partial"); tmp.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8"); tmp.replace(path)
def log(message):
 stamp=datetime.now(timezone.utc).isoformat(); line=f"{stamp} {message}"; print(line,flush=True)
 with LOG.open("a",encoding="utf-8") as f:f.write(line+"\n");f.flush();os.fsync(f.fileno())
def status(api):
 r=api.kernels_status(KERNEL)
 return getattr(r.status,"name",str(r.status))

def cli_output(pattern,path):
 cmd=[sys.executable,"-m","kaggle","kernels","output",KERNEL,"--path",str(path),"--file-pattern",pattern,"--page-size","200"]
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=600)
 with LOG.open("a",encoding="utf-8") as f:
  f.write("\n[OUTPUT COMMAND] "+" ".join(cmd)+"\n"+p.stdout+"\n[STDERR]\n"+p.stderr+"\n");f.flush();os.fsync(f.fileno())
 if p.returncode:raise RuntimeError(f"kernels output returned {p.returncode}; see monitor.log")
 return p.stdout

def retrieve_failure_log():
 p=subprocess.run([sys.executable,"-m","kaggle","kernels","logs",KERNEL],capture_output=True,text=True,timeout=240)
 (OPS/"terminal_kernel_log.txt").write_text(p.stdout+"\n[stderr]\n"+p.stderr,encoding="utf-8")
 write_json(OPS/"terminal_log_receipt.json",{"kernel":KERNEL,"returncode":p.returncode,"bytes":len(p.stdout.encode()),"saved":str(OPS/"terminal_kernel_log.txt")})

def download_archive(api,receipt):
 if receipt.get("images")!=6000:raise RuntimeError(f"archive receipt image count unexpected: {receipt.get('images')}")
 remote=str(receipt.get("path","")); marker="/kaggle/working/"
 if marker not in remote:raise RuntimeError("archive receipt path is not under /kaggle/working")
 remote_rel=remote.split(marker,1)[1]
 expected=int(receipt.get("bytes",-1)); expected_sha=str(receipt.get("sha256","" )).upper()
 if expected<=0 or not re.fullmatch(r"[0-9A-F]{64}",expected_sha):raise RuntimeError("invalid archive receipt size/checksum")
 usage=shutil.disk_usage(DOWNLOADS)
 if usage.free<=expected+GIB:raise RuntimeError(f"insufficient disk reserve: free={usage.free}, archive={expected}, required reserve={GIB}")
 if expected>5*GIB:raise RuntimeError(f"archive exceeds 5 GiB safety cap: {expected}")
 req=ApiDownloadKernelOutputRequest();req.owner_slug=OWNER;req.kernel_slug=SLUG;req.version_number=VERSION;req.file_path=remote_rel
 with api.build_kaggle_client() as client:redir=client.kernels.kernels_api_client.download_kernel_output(req)
 partial=DOWNLOADS/"Mini_dataset_6000.tar.partial"; target=DOWNLOADS/"Mini_dataset_6000.tar"
 h=hashlib.sha256(); count=0
 with requests.get(redir.url,stream=True,timeout=(30,180)) as response:
  response.raise_for_status()
  with partial.open("wb") as out:
   for chunk in response.iter_content(8*1024*1024):
    if not chunk:continue
    count+=len(chunk)
    if count>expected:raise RuntimeError("download exceeded receipt byte count")
    out.write(chunk);h.update(chunk)
   out.flush();os.fsync(out.fileno())
 actual=h.hexdigest().upper()
 if count!=expected or actual!=expected_sha:raise RuntimeError(f"archive checksum mismatch: bytes={count}/{expected}, sha256={actual}/{expected_sha}")
 partial.replace(target)
 write_json(DOWNLOADS/"Mini_dataset_6000_receipt.json",receipt)
 write_json(DOWNLOADS/"download_verification.json",{"kernel":KERNEL,"version":VERSION,"remote_path":remote_rel,"local_path":str(target),"bytes":count,"sha256":actual,"receipt_match":True,"free_bytes_before":usage.free,"verified_utc":datetime.now(timezone.utc).isoformat()})
 return target,count,actual

def main():
 start=time.time(); deadline=start+MAX_HOURS*3600; api=KaggleApi();api.authenticate()
 write_json(STATE,{"kernel":KERNEL,"version":VERSION,"pid":os.getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),"status":"monitoring","downloads":str(DOWNLOADS)})
 log(f"MONITOR START kernel={KERNEL} pid={os.getpid()} poll={POLL_SECONDS}s max_hours={MAX_HOURS}")
 terminal=None; last=None; errors=0
 while time.time()<deadline:
  try:
   current=status(api);errors=0
   if current!=last:
    log(f"STATUS {current}"); last=current
    write_json(STATE,{"kernel":KERNEL,"version":VERSION,"pid":os.getpid(),"status":current,"updated_utc":datetime.now(timezone.utc).isoformat(),"downloads":str(DOWNLOADS)})
   if current.upper() in ("COMPLETE","ERROR","CANCELLED","CANCELED"):
    terminal=current;break
  except Exception as e:
   errors+=1; log(f"STATUS ERROR {type(e).__name__} attempt={errors}")
   write_json(STATE,{"kernel":KERNEL,"version":VERSION,"pid":os.getpid(),"status":"status_poll_error","error_type":type(e).__name__,"attempt":errors,"updated_utc":datetime.now(timezone.utc).isoformat()})
   if errors>=8:raise
   time.sleep(min(60,2**errors));continue
  time.sleep(POLL_SECONDS)
 if terminal is None:
  write_json(STATE,{"kernel":KERNEL,"version":VERSION,"pid":os.getpid(),"status":"monitor_deadline_exceeded","deadline_hours":MAX_HOURS,"updated_utc":datetime.now(timezone.utc).isoformat()});log("MONITOR DEADLINE EXCEEDED; job left untouched");return
 if terminal.upper()!="COMPLETE":
  log(f"TERMINAL FAILURE status={terminal}; retrieving exact-version logs")
  retrieve_failure_log()
  write_json(STATE,{"kernel":KERNEL,"version":VERSION,"pid":os.getpid(),"status":terminal,"terminal_log":str(OPS/"terminal_kernel_log.txt"),"updated_utc":datetime.now(timezone.utc).isoformat()});return
 log("TERMINAL COMPLETE; retrieve compact receipts and verify gates")
 meta=OPS/"remote_compact"
 pattern=r"(Mini_dataset_6000_receipt\.json|ACCEPTANCE\.json|artifact_manifest\.json|training_receipt\.json|progress\.json|summary\.json|environment\.json|source_status\.json|logs/.*\.(json|log))$"
 candidates=[]; acceptance=[]; training=[]
 for publish_attempt in range(1,6):
  cli_output(pattern,meta)
  candidates=list(meta.rglob("Mini_dataset_6000_receipt.json"))
  acceptance=list(meta.rglob("ACCEPTANCE.json")); training=list(meta.rglob("training_receipt.json"))
  if len(candidates)==len(acceptance)==len(training)==1:break
  log(f"PUBLISH RACE attempt={publish_attempt}/5 receipts={len(candidates)} acceptance={len(acceptance)} training={len(training)}")
  if publish_attempt<5:time.sleep(20)
 if len(candidates)!=1:raise RuntimeError(f"expected one archive receipt after publish retries, found {len(candidates)}")
 receipt=json.loads(candidates[0].read_text(encoding="utf-8"))
 # Save compact ACCEPTANCE and training receipt locations as retrieved by the version-pinned CLI.
 if len(acceptance)!=1:raise RuntimeError(f"expected one acceptance record after publish retries, found {len(acceptance)}")
 acc=json.loads(acceptance[0].read_text(encoding="utf-8"))
 if acc.get("acquired_images")!=6000 or not acc.get("parameters_complete"):raise RuntimeError(f"remote acceptance gate failed: {acc}")
 if len(training)!=1:raise RuntimeError(f"expected one provisional training receipt after publish retries, found {len(training)}")
 downloaded=download_archive(api,receipt)
 # Keep exactly the selected compact evidence beside the image archive.
 evidence=DOWNLOADS/"remote_compact";evidence.mkdir(exist_ok=True)
 shutil.copy2(acceptance[0],evidence/"ACCEPTANCE.json");shutil.copy2(training[0],evidence/"training_receipt.json");shutil.copy2(candidates[0],evidence/"Mini_dataset_6000_receipt.json")
 write_json(STATE,{"kernel":KERNEL,"version":VERSION,"pid":os.getpid(),"status":"complete_and_archive_verified","archive":str(downloaded[0]),"archive_bytes":downloaded[1],"archive_sha256":downloaded[2],"updated_utc":datetime.now(timezone.utc).isoformat()})
 log(f"VERIFIED ARCHIVE path={downloaded[0]} bytes={downloaded[1]} sha256={downloaded[2]}")

if __name__=="__main__":
 try:main()
 except Exception as e:
  traceback.print_exc()
  try:write_json(STATE,{"kernel":KERNEL,"version":VERSION,"pid":os.getpid(),"status":"monitor_or_retrieval_error","error_type":type(e).__name__,"error":str(e),"updated_utc":datetime.now(timezone.utc).isoformat()});log(f"MONITOR ERROR {type(e).__name__}: {e}")
  except Exception:pass
  sys.exit(1)
