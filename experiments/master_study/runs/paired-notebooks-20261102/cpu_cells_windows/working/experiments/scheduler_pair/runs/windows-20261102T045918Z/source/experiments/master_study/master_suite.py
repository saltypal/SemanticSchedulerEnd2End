"""Master execution controller; all requested learners accounted for, failures preserved."""
import argparse,csv,json,os,subprocess,sys,time,traceback,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/"experiments/stream_scheduler"
SEEDS="1729"

def main():
    p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--seconds",type=float,default=None)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();ledger={};lock=threading.Lock()
    (a.output/"logs").mkdir()
    declared=("linear_choice","mlp_choice","delay_huber","ranknet","lambdamart","priority_classifier",
              "deepsets_ranknet","set_transformer_ranknet","listmle","masked_ppo","masked_double_dqn",
              "td_per","ranknet_warmstart_ppo","temporal_deepsets_ranknet","rl_vlm_f","vlm_replay")
    pending={arm+"/"+seed:{"status":"not_started"} for arm in declared for seed in SEEDS.split(",")}
    for arm in ("linear_choice","mlp_choice","ranknet","deepsets_ranknet","set_transformer_ranknet","temporal_deepsets_ranknet","visionprefer"):
        for seed in SEEDS.split(","):pending["P/"+arm+"/"+seed]={"status":"not_started"}
    (a.output/"method_ledger.json").write_text(json.dumps(pending,indent=2))

    def record():
        with lock:
            (a.output/"stages.json").write_text(json.dumps(ledger,indent=2))
            current=dict(pending)
            for folder in ("D","P","feedback"):
                path=a.output/folder/"status.json"
                if not path.exists():continue
                try:data=json.loads(path.read_text())
                except json.JSONDecodeError:continue # Child may be replacing its small status file; next update retries.
                for key,value in data.get("arms",data).items():
                    current[("P/" if folder=="P" else "")+key]=value
            (a.output/"method_ledger.json").write_text(json.dumps(current,indent=2))
    def run(name,script,args,limit=None):
        remaining=float("inf") if a.seconds is None else a.seconds-(time.monotonic()-start)
        if remaining<60:
            ledger[name]={"status":"not_started_budget"};record();return False
        command=[sys.executable,"-u",str(script),*map(str,args)]
        ledger[name]={"status":"running","command":command};record()
        print("MASTER_START",name,command,flush=True)
        with (a.output/"logs"/(name+".log")).open("w",buffering=1) as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            import threading,signal
            def capture():
                for line in proc.stdout:
                    log.write(line);print("["+name+"]",line,end="",flush=True)
                    if any(word in line for word in ("TRAIN ","EPOCH ","FEEDBACK_RL","P_STUDENT")):record()
            thread=threading.Thread(target=capture);thread.start()
            try:code=proc.wait(timeout=None if a.seconds is None and limit is None else max(1,min(remaining-20,limit or remaining-20)))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
            thread.join()
        ledger[name]={"status":"process_finished" if code==0 else "partial_budget" if code==124 else "failed",
                      "exit":code,"command":command};record()
        return code==0
    config={"seeds":SEEDS,"epochs":50,"rl_actions_per_arm_seed":20000,"global_seconds":a.seconds,
            "reference_scope":"development-exposed silver references; independent final evaluation blocked",
            "incident_status":"blocked pending supported real scene manifest","teacher_calls":216,
            "scope":"full declared method attempts; individual label/resource gates remain mandatory"}
    (a.output/"config.json").write_text(json.dumps(config,indent=2))
    if not run("input_preflight",SCRIPTS/"input_preflight.py",["--archive",a.archive,"--output",a.output/"input_preflight"]):raise SystemExit(2)
    if not run("vectors",SCRIPTS/"complete_vectors.py",["--archive",a.archive,"--output",a.output/"vectors","--device","cuda:0"]):raise SystemExit(2)
    ext=a.output/"vectors/vectors.npz"
    if not run("prepare_teacher",SCRIPTS/"prepare_teacher.py",["--archive",a.archive,"--output",a.output/"prepared","--mission-text","--vector-extension",ext]):raise SystemExit(2)
    # All D models, all seeds. No old600-second per-model cap.
    d_thread=threading.Thread(target=lambda:run("D",SCRIPTS/"train_real_suite.py",["--archive",a.archive,"--output",a.output/"D","--seeds",SEEDS,"--epochs",50,"--rl-steps",20000,
        "--training-episodes",64,"--samples",512,"--validation-traces",15,
        "--device","cuda:0","--mission-vectors",a.output/"prepared/data/mission_vectors.npy","--vector-extension",ext,
        "--stream-mode","mixed","--stream-duration",200,"--stream-burst-factor",4,"--require-sources","crisismmd,open_images,places365"]))
    d_thread.start()
    teacher_ok=run("teacher",SCRIPTS/"qwen_teacher.py",["--requests",a.output/"prepared/requests.jsonl","--output",a.output/"teacher",
        "--revision","cc594898137f460bfe9f0759e9844b3ce807cfb5","--device","cuda:1","--max-calls",216])
    if teacher_ok:
        run("P",SCRIPTS/"train_teacher_rankers.py",["--prepared",a.output/"prepared","--teacher-output",a.output/"teacher","--output",a.output/"P",
            "--device","cuda:1","--seeds",SEEDS,"--epochs",50])
        run("feedback",SCRIPTS/"train_feedback_rl.py",["--archive",a.archive,"--prepared",a.output/"prepared","--teacher-output",a.output/"teacher",
            "--output",a.output/"feedback","--device","cuda:1","--seeds",SEEDS,"--steps",20000,
            "--mission-vectors",a.output/"prepared/data/mission_vectors.npy","--vector-extension",ext])
    else:
        ledger["P"]={"status":"blocked_teacher"};ledger["feedback"]={"status":"blocked_teacher"};record()
    d_thread.join()
    if a.seconds is None or a.seconds-(time.monotonic()-start)>90:
        run("rate_study",SCRIPTS/"ue_rate_study.py",["--archive",a.archive,"--output",a.output/"rate_study","--vector-extension",ext,
            "--mission-vectors",a.output/"prepared/data/mission_vectors.npy","--ues","12,50,60,100","--seeds",SEEDS])
    # Construct expected identities before summarizing; no disappearance of unexecuted arms.
    sys.path.insert(0,str(ROOT))
    import importlib.util
    sp=importlib.util.spec_from_file_location("boot",SCRIPTS/"runtime_bootstrap.py");boot=importlib.util.module_from_spec(sp);sp.loader.exec_module(boot);boot.install(ROOT)
    from Scheduler.training.semantic_learning_models import ARMS
    methods={arm+"/"+seed:{"status":"not_started_budget"} for arm in ARMS if arm!="visionprefer" for seed in SEEDS.split(",")}
    for arm in ("linear_choice","mlp_choice","ranknet","deepsets_ranknet","set_transformer_ranknet","temporal_deepsets_ranknet","visionprefer"):
        for seed in SEEDS.split(","):methods["P/"+arm+"/"+seed]={"status":"not_started_budget"}
    for folder in ("D","P","feedback"):
        status=a.output/folder/"status.json"
        if status.exists():
            data=json.loads(status.read_text());data=data.get("arms",data)
            for key,value in data.items():
                if folder=="D" and key.startswith("visionprefer/"):continue
                methods[(folder+"/" if folder=="P" else "")+key]=value
    methods["final_independent_evaluation"]={"status":"blocked_data","reason":"No fresh independent final urgency cohort supplied"}
    methods["real_incident_evaluation"]={"status":"blocked_data","reason":"No supported real incident manifest supplied"}
    methods["adaptive_second_feedback_round"]={"status":"not_implemented","reason":"Existing feedback runner performs one frozen block only"}
    (a.output/"method_ledger.json").write_text(json.dumps(methods,indent=2))
    rows=[{"method":key,**{k:v for k,v in value.items() if not isinstance(v,(dict,list))}} for key,value in methods.items()]
    fields=sorted({key for row in rows for key in row})
    with (a.output/"comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (a.output/"REPORT.md").write_text("# Master execution\n\nDevelopment/silver scope only. Inspect method_ledger.json and comparison.csv. Process exit0 is not full study completion.\n\n"+
        "\n".join("- "+key+": "+value.get("status","unknown") for key,value in methods.items()))
    print("MASTER_FINISHED",a.output,flush=True)

if __name__=="__main__":main()
