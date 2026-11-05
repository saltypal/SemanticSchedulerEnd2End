import subprocess,time,json,datetime,hashlib
from pathlib import Path
root=Path.cwd();out=root/"experiments/master_study/runs/launch-20261101/v9";cli="/home/Satya/.venvs/semantic-kaggle/bin/kaggle";slug="satyapaladuguplus/semantic-decoder-scheduler-20261030"
with (out/"live.log").open("w") as f:
 logs=subprocess.Popen([cli,"kernels","logs",slug,"--follow"],stdout=f,stderr=subprocess.STDOUT)
 seen=False
 try:
  for iteration in range(2880):
   status=subprocess.run([cli,"kernels","status",slug],capture_output=True,text=True,timeout=45)
   text=status.stdout+status.stderr
   with (out/"status.jsonl").open("a") as h:h.write(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":text})+"\n")
   if "RUNNING" in text:seen=True
   if seen and any(x in text for x in ("COMPLETE","ERROR","CANCEL")):
    result=subprocess.run([cli,"kernels","output",slug,"--path",str(out/"retrieved"),"--file-pattern",".*[.]json$","--force"],capture_output=True,text=True,timeout=600)
    (out/"retrieval.log").write_text(result.stdout+result.stderr)
    (out/"retrieved_checksums.json").write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/"retrieved").rglob("*.json")},indent=2))
    break
   time.sleep(60)
 finally:
  logs.terminate()
