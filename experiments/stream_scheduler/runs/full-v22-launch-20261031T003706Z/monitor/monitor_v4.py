"""Finite v4 Kaggle monitor with terminal inventory and bounded retrieval."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, re, shutil, subprocess, time

ROOT = Path("/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End")
RUN = ROOT / "experiments/stream_scheduler/runs/full-v22-launch-20261031T003706Z"
MON = RUN / "monitor"
KAGGLE = "/home/Satya/.venvs/semantic-kaggle/bin/kaggle"
KERNEL = "satyapaladuguplus/semantic-decoder-scheduler-20261030/4"
SUBMITTED = datetime(2026, 10, 31, 0, 38, 37, tzinfo=timezone.utc)
MAX_SECONDS = 3 * 60 * 60
POLL_SECONDS = 60
TERMINAL = {"COMPLETE", "ERROR", "FAILED", "CANCELLED", "CANCELLED_BY_USER", "CANCEL_ACKNOWLEDGED"}
PICK = re.compile(r"(^|/)(REPORT\.md|REPORT\.json|.*manifest.*\.json|.*summary.*\.json|.*ledger.*\.json|.*status.*\.json|.*environment.*\.json|.*audit.*\.json|.*\.log|.*\.ipynb|.*\.pt|.*\.npz|.*\.json)$", re.I)
MON.mkdir(parents=True, exist_ok=True)
(MON / "monitor_pid.txt").write_text(str(os.getpid()))
start_utc = datetime.now(timezone.utc)
deadline = time.monotonic() + MAX_SECONDS
history = MON / "v4_status_history.jsonl"
progress_path = MON / "v4_progress.jsonl"
stream_path = MON / "remote_worker_stream.log"
snapshot_path = MON / "latest_worker_logs.txt"
stream_file = stream_path.open("a", buffering=1)
stream_proc = subprocess.Popen([KAGGLE, "kernels", "logs", "--follow", KERNEL.rsplit("/", 1)[0]],
                               stdout=stream_file, stderr=subprocess.STDOUT, start_new_session=True)

def append(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()

def run(args, timeout=120):
    return subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout)

def status_now():
    p = run(["kernels", "status", KERNEL], 90)
    s = p.stdout + " " + p.stderr
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", s)
    return (m.group(1) if m else "UNKNOWN"), p

def progress_signature():
    parts=[]
    for p in (snapshot_path, stream_path):
        if p.exists():
            try: parts.extend(p.read_text(errors="replace").splitlines())
            except OSError: pass
    parts=list(dict.fromkeys(parts))
    vector=None
    for line in parts:
        m=re.search(r"MISSING_CLIP\s+(\d+)\s+(\d+)", line)
        if m: vector=f"{m.group(1)}/{m.group(2)}"
    d_updates=[line.strip() for line in parts if "[D_suite] TRAIN " in line or "[D_suite] EPOCH " in line]
    teacher=[line.strip() for line in parts if "[Qwen] TEACHER" in line or "TEACHER control" in line]
    stage="waiting for worker"
    for line in reversed(parts):
        if "[D_suite] TRAIN " in line or "[D_suite] EPOCH " in line: stage="D optimizer"
        elif "[D_suite] BASELINE " in line: stage="D baselines"
        elif "[D_suite] START REAL DATA SUITE" in line: stage="D data/episode setup"
        elif "[device_preflight]" in line: stage="device preflight"
        elif "[prepare_teacher] DONE" in line: stage="teacher requests prepared"
        elif "[complete_vectors] MISSING_CLIP" in line:
            stage="vector fill complete" if vector and vector.split("/")[0]==vector.split("/")[1] else "vector fill"
            break
        if stage != "waiting for worker": break
    return {"stage":stage,"vector_fill":vector,"d_training_latest":d_updates[-1] if d_updates else None,
            "d_training_log_events":len(d_updates),"qwen_teacher_response_lines":len(teacher),
            "qwen_latest":teacher[-1] if teacher else None}

def write_progress(sig):
    append(progress_path, {"utc":datetime.now(timezone.utc).isoformat(),**sig})

def quota():
    p=run(["quota"],60)
    (MON/"v4_quota_terminal.txt").write_text(p.stdout+"\nSTDERR\n"+p.stderr)
    return p.stdout.strip()

def parse_pages():
    pages=[]; token=None
    for idx in range(30):
        args=["kernels","files",KERNEL,"--format","json","--page-size","200"]
        if token: args += ["--page-token",token]
        p=run(args,120)
        (MON/f"v4_files_page_{idx:02d}.txt").write_text(p.stdout+"\nSTDERR\n"+p.stderr)
        if p.returncode: raise RuntimeError("kernel files listing failed: "+p.stderr[-1000:])
        pos=p.stdout.find("[")
        if pos<0: raise RuntimeError("output listing lacks JSON array")
        rows=json.loads(p.stdout[pos:])
        token_match=re.search(r"Next Page Token = (.+)",p.stdout[:pos])
        pages.append({"index":idx,"rows":rows,"token":token_match.group(1).strip() if token_match else None})
        if not pages[-1]["token"]: break
        token=pages[-1]["token"]
    else: raise RuntimeError("output listing exceeded 30 pages")
    return pages

def date_utc(value):
    try: return datetime.strptime(value,"%I:%M %p, %A %d %B %Y UTC").replace(tzinfo=timezone.utc)
    except Exception: return None

def inventory_retrieve(final_status):
    pages=parse_pages()
    rows=[r for pg in pages for r in pg["rows"]]
    fresh=[];unknown=[]
    for r in rows:
        d=date_utc(str(r.get("creationDate","")))
        if d is None: unknown.append(r)
        elif d>SUBMITTED: fresh.append(r)
    selected=[r for r in fresh if PICK.search(str(r.get("name","")))]
    free=shutil.disk_usage("/mnt/d").free
    cap=min(4_000_000_000,max(0,free-1_000_000_000))
    bytes_selected=sum(int(r.get("size") or 0) for r in selected)
    out={"utc":datetime.now(timezone.utc).isoformat(),"kernel":KERNEL,"terminal_status":final_status,
         "file_count":len(rows),"fresh_after_submission":len(fresh),"unknown_date_count":len(unknown),
         "fresh_selected_count":len(selected),"fresh_selected_bytes":bytes_selected,"disk_free_before":free,
         "download_cap_bytes":cap,"files":rows,"selected_names":[r.get("name") for r in selected],
         "output_page":"https://www.kaggle.com/code/satyapaladuguplus/semantic-decoder-scheduler-20261030/output"}
    if final_status in {"CANCELLED","CANCELLED_BY_USER","CANCEL_ACKNOWLEDGED"}:
        out["retrieval"]="SKIPPED: cancelled run output may be stale"
    elif not fresh:
        out["retrieval"]="SKIPPED: output listing contains no files dated after v4 submission"
    elif not selected:
        out["retrieval"]="SKIPPED: no selected evidence files"
    elif bytes_selected>cap:
        out["retrieval"]="SKIPPED: selected files exceed conservative disk cap; use output page link"
    else:
        target=RUN/"retrieved/version4"
        target.mkdir(parents=True,exist_ok=True)
        pattern=PICK.pattern
        results=[]
        for pg in pages:
            if not pg["rows"]: continue
            names={r.get("name") for r in pg["rows"]}
            if not any(r.get("name") in names for r in selected): continue
            args=["kernels","output",KERNEL,"--path",str(target),"--file-pattern",pattern,"--page-size","200"]
            if pg["index"]>0:
                args += ["--page-token",pages[pg["index"]-1]["token"]]
            p=run(args,600)
            results.append({"page":pg["index"],"exit":p.returncode,"stdout":p.stdout,"stderr":p.stderr})
            if p.returncode: break
        actual=[]
        for f in target.rglob("*"):
            if f.is_file():
                h=hashlib.sha256(f.read_bytes()).hexdigest()
                actual.append({"path":str(f.relative_to(target)),"bytes":f.stat().st_size,"sha256":h})
        out.update(retrieval="COMPLETE" if results and all(x["exit"]==0 for x in results) else "PARTIAL_OR_FAILED",
                   retrieval_results=results,retrieved_files=actual,
                   disk_free_after=shutil.disk_usage("/mnt/d").free)
        (target/"retrieval_manifest.json").write_text(json.dumps(actual,indent=2))
    (MON/"v4_output_inventory.json").write_text(json.dumps(out,indent=2))
    return out

append(history,{"utc":start_utc.isoformat(),"event":"MONITOR_STARTED","pid":os.getpid(),"max_seconds":MAX_SECONDS,"poll_seconds":POLL_SECONDS,"kernel":KERNEL})
last_status=None; last_sig=None; final=None
try:
    while time.monotonic()<deadline:
        status,p=status_now()
        if status!=last_status:
            rec={"utc":datetime.now(timezone.utc).isoformat(),"status":status,"exit":p.returncode,
                 "stdout":p.stdout.strip(),"stderr":p.stderr.strip()}
            append(history,rec);print("STATUS",status,rec["utc"],flush=True);last_status=status
        sig=progress_signature()
        if sig!=last_sig:
            write_progress(sig);print("PROGRESS",json.dumps(sig,sort_keys=True),flush=True);last_sig=sig
        if status in TERMINAL:
            final=status
            append(history,{"utc":datetime.now(timezone.utc).isoformat(),"event":"TERMINAL","status":status})
            break
        time.sleep(POLL_SECONDS)
    if final is None:
        append(history,{"utc":datetime.now(timezone.utc).isoformat(),"event":"MONITOR_DEADLINE","remote_job_left_untouched":True})
        final="MONITOR_DEADLINE"
    if stream_proc.poll() is None:
        stream_proc.terminate()
        try: stream_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            stream_proc.kill();stream_proc.wait(timeout=5)
    stream_file.close()
    result={"final_status":final,"final_utc":datetime.now(timezone.utc).isoformat(),"budget_quota":quota(),
            "progress":progress_signature(),"monitor_pid":os.getpid()}
    if final in TERMINAL:
        try: result["output_inventory"]=inventory_retrieve(final)
        except Exception as exc:
            result["output_inventory_error"]=type(exc).__name__+": "+str(exc)
            append(MON/"v4_monitor_errors.jsonl",{"utc":datetime.now(timezone.utc).isoformat(),"error":result["output_inventory_error"]})
    (MON/"v4_monitor_final.json").write_text(json.dumps(result,indent=2))
    report=["# V4 Kaggle monitor final", "", f"- Status: {final}", f"- Checked UTC: {result['final_utc']}",
            f"- Latest progress: {json.dumps(result['progress'],sort_keys=True)}",
            f"- Quota snapshot: {result['budget_quota'].replace(chr(10),' | ')}",
            f"- Output inventory: see v4_output_inventory.json; retrieved artifacts, if any, are checksummed in retrieved/version4/retrieval_manifest.json."]
    (MON/"v4_monitor_report.md").write_text("\n".join(report)+"\n")
    print("MONITOR_FINAL",json.dumps({"status":final,"progress":result["progress"]},sort_keys=True),flush=True)
except BaseException as exc:
    append(MON/"v4_monitor_errors.jsonl",{"utc":datetime.now(timezone.utc).isoformat(),"error":type(exc).__name__+": "+str(exc)})
    raise


