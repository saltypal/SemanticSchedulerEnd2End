from pathlib import Path
import sys,json,hashlib,datetime,traceback
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8',errors='replace')
if hasattr(sys.stderr,'reconfigure'): sys.stderr.reconfigure(encoding='utf-8',errors='replace')
from kaggle.api.kaggle_api_extended import KaggleApi
root=Path(__file__).resolve().parent
out=root/'retrieved_version9';out.mkdir(parents=True,exist_ok=True)
api=KaggleApi();api.authenticate();token=None;page=0;retrieved=[];events=root/'retrieval_pages.jsonl'
try:
 while True:
  files,next_token=api.kernels_output('satyapaladugu/betterscheduler-spi-20261027finalc',str(out),quiet=True,page_token=token,page_size=200)
  page+=1;retrieved.extend(str(x) for x in files)
  row={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'version':9,'page':page,'files_returned':len(files),'next_page':bool(next_token),'files':[str(x) for x in files]}
  with events.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=True)+'\n')
  print('RETRIEVED_PAGE',page,'FILES',len(files),'MORE',bool(next_token),flush=True)
  if not next_token:break
  token=next_token
except Exception as e:
 row={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'version':9,'page':page+1,'event':'retrieval_error','type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc().splitlines()}
 with events.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=True)+'\n')
 print('RETRIEVAL_ERROR',type(e).__name__,str(e),flush=True)
 raise
files=[p for p in out.rglob('*') if p.is_file()]
receipt={'version':9,'remote_kernel':'satyapaladugu/betterscheduler-spi-20261027finalc','retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'output_root':str(out),'files_on_disk':len(files),'bytes_on_disk':sum(p.stat().st_size for p in files),'pages_completed':page,'listed_files_this_pass':len(retrieved),'sha256':{p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
(out/'retrieval_inventory.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print('RETRIEVAL_COMPLETE',json.dumps({k:receipt[k] for k in ('files_on_disk','bytes_on_disk','pages_completed','listed_files_this_pass')},ensure_ascii=True),flush=True)
