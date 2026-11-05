"""Bound an experiment process group and preserve its streamed logs and exit status."""
import argparse,datetime,json,os,signal,subprocess,sys,threading,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument("--seconds",type=float,required=True);p.add_argument("--log-dir",type=Path,required=True);p.add_argument("command",nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==["--"] else a.command
    if not command or a.seconds<=0:p.error("positive seconds and command required")
    a.log_dir.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    def forward():
        with (a.log_dir/"driver.log").open("a",buffering=1) as f:
            for line in proc.stdout:f.write(line);print(line,end="",flush=True)
    thread=threading.Thread(target=forward,daemon=True);thread.start();timed_out=False
    try:
        proc.wait(timeout=a.seconds)
    except subprocess.TimeoutExpired:
        timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    finally:
        thread.join(timeout=5)
        receipt={"command":command,"returncode":proc.returncode,"timed_out":timed_out,"elapsed_seconds":time.monotonic()-start,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"scope":"Process completion is not scientific success; inspect method ledger"}
        (a.log_dir/"process_status.json").write_text(json.dumps(receipt,indent=2))
    return 124 if timed_out else proc.returncode
if __name__=="__main__":sys.exit(main())
