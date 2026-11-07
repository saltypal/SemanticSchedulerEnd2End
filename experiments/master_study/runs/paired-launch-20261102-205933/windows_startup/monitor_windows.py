import datetime as dt
import json
import pathlib
import subprocess
import time
root = pathlib.Path(r"D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End\experiments\master_study\runs\paired-launch-20261102-205933\windows_startup")
cli = r"C:\Users\satya\AppData\Roaming\Python\Python314\Scripts\kaggle.exe"
kernel = "satyapaladugu/semantic-decoder-scheduler-rl/1"
start = time.monotonic(); max_seconds = 15*60; interval = 120; poll = 0
def call(args):
    try:
        p = subprocess.run([cli, *args, kernel], capture_output=True, text=True, timeout=75)
        return {"exit_code":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"timed_out":False}
    except subprocess.TimeoutExpired as e:
        dec=lambda v: v.decode(errors="replace") if isinstance(v,bytes) else (v or "")
        return {"exit_code":None,"stdout":dec(e.stdout),"stderr":dec(e.stderr),"timed_out":True}
while time.monotonic()-start < max_seconds:
    poll += 1; now = dt.datetime.now(dt.timezone.utc).isoformat()
    status=call(["kernels","status"]); logs=call(["kernels","logs"])
    prefix=root/f"poll-{poll:02d}"
    prefix.with_suffix(".status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
    prefix.with_suffix(".logs.txt").write_text(logs["stdout"]+("\n[stderr]\n"+logs["stderr"] if logs["stderr"] else ""),encoding="utf-8")
    state={"utc":now,"kernel":kernel,"poll":poll,"elapsed_seconds":round(time.monotonic()-start,1),"status":status,"logs_exit_code":logs["exit_code"],"logs_timed_out":logs["timed_out"],"logs_chars":len(logs["stdout"]),"logs_empty":not bool(logs["stdout"].strip())}
    with (root/"polls.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(state,separators=(",",":"))+"\n")
    text=logs["stdout"].lower(); status_text=(status["stdout"]+status["stderr"]).strip()
    terms=["starting","initializing","installing","gpu","t4","cuda","training","epoch","step","dataset","hash","pytest","passed","failed","traceback","error"]
    signal={"utc":now,"poll":poll,"status":status_text,"log_chars":len(logs["stdout"]),"signal_terms":[term for term in terms if term in text],"logs_exit_code":logs["exit_code"],"logs_timed_out":logs["timed_out"]}
    with (root/"signals.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(signal,separators=(",",":"))+"\n")
    if status["exit_code"] not in (0,None) or logs["exit_code"] not in (0,None) or status["timed_out"] or logs["timed_out"]: print(f"poll {poll}: API/CLI issue; persisted status and logs",flush=True)
    elif status_text and any(term in status_text.lower() for term in ["complete","error","failed","cancel"]): print(f"poll {poll}: terminal status {status_text!r}; persisted logs",flush=True)
    elif logs["stdout"].strip(): print(f"poll {poll}: log output available ({len(logs['stdout'])} chars); persisted",flush=True)
    else: print(f"poll {poll}: status={status_text!r}; logs empty",flush=True)
    remain=max_seconds-(time.monotonic()-start)
    if remain>0: time.sleep(min(interval,remain))
summary={"kernel":kernel,"poll_count":poll,"duration_seconds":round(time.monotonic()-start,1),"artifact_dir":str(root)}
(root/"monitor_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print("monitor window finished",json.dumps(summary),flush=True)
