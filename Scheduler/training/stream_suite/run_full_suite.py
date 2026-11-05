"""One bounded supervisor: preparation, D suite, Qwen, P students and AI-feedback RL."""
import argparse,datetime,json,os,subprocess,sys,threading,time,traceback
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
REVISION="cc594898137f460bfe9f0759e9844b3ce807cfb5"
def summarize_execution(ledger,receipts):
    """Process success and scientific completion are distinct, durable outcomes."""
    normalized={}
    for key,row in ledger.items():
        row=dict(row)
        if row.get("status") in ("running","planned","not_started"):
            row={**row,"status":"interrupted","reason":"no terminal method outcome recorded"}
        normalized[key]=row
    counts=dict(Counter(row.get("status","unknown") for row in normalized.values()))
    failures={name:r["exit"] for name,r in receipts.items() if r.get("exit")!=0}
    incomplete=any(row.get("status") not in ("evaluated_development_only","evaluated_silver_validation") for row in normalized.values())
    outcome={"status":"partial" if incomplete or failures else "completed",
             "method_status_counts":counts,"child_failures":failures,
             "scientific_superiority_established":False}
    return normalized,outcome


def main():
    p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--seconds",type=float,default=8400)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/"logs").mkdir(exist_ok=True);start=time.monotonic();jobs={}
    def launch(name,script,args):
        command=[sys.executable,"-u",str(ROOT/"Scheduler/training/stream_suite"/script)]+list(map(str,args))
        log=(a.output/"logs"/(name+".log")).open("w",buffering=1)
        environment=dict(os.environ)
        environment.update({name:"1" for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")})
        proc=subprocess.Popen(command,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,cwd=ROOT)
        def pump():
            for line in proc.stdout:log.write(line);print("["+name+"] "+line,end="",flush=True)
            log.close()
        t=threading.Thread(target=pump,daemon=True);t.start();jobs[name]={"process":proc,"thread":t,"command":command}
        return proc
    def wait(proc,limit):
        try:return proc.wait(timeout=max(1,min(limit,a.seconds-(time.monotonic()-start))))
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            return 124
    import torch
    count=torch.cuda.device_count()
    if not count:raise RuntimeError("full GPU execution requested but managed runtime exposes no CUDA")
    teacher_device="cuda:1" if count>1 else "cuda:0"
    manifest={"scope":"bounded v2.2 full-method implementation/feasibility; not convergence or independent urgency","started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"seconds":a.seconds,"gpu_count":count,"torch":torch.__version__,"teacher_revision":REVISION,"first_seed":1729,"additional_seeds":"retained; not_started_budget unless separately executed","models_required":17}
    (a.output/"config.json").write_text(json.dumps(manifest,indent=2))
    extra=a.output/"vector_completion"
    code=wait(launch("complete_vectors","complete_vectors.py",["--archive",a.archive,"--output",extra,"--device","cuda:0"]),900)
    extension=extra/"vectors.npz"
    if code!=0 or not extension.exists():
        (a.output/"method_ledger.json").write_text(json.dumps({"input_preparation":{"status":"blocked_input","reason":"Required complete vector preparation failed; no partial-input training launched","exit":code}},indent=2))
        raise RuntimeError("Required embedding completion failed; refuse silently degraded parameter study")
    extension_args=["--vector-extension",extension]
    prepared=a.output/"teacher_prepared"
    prepared_code=wait(launch("prepare_teacher","prepare_teacher.py",["--archive",a.archive,"--output",prepared,"--mission-text"]+extension_args),900)
    if prepared_code!=0:
        (a.output/"method_ledger.json").write_text(json.dumps({"input_preparation":{"status":"blocked_input","reason":"Required mission-aware teacher preparation failed","exit":prepared_code}},indent=2))
        raise RuntimeError("Required mission-aware input preparation failed")
    device="cuda:0"
    if prepared_code==0:
        timing=a.output/"device_preflight.json"
        timing_code=wait(launch("device_preflight","preflight_device.py",["--prepared",prepared,"--output",timing]),120)
        if timing_code==0:device=json.loads(timing.read_text())["chosen_neural_device"]
    # One GPU: teacher owns it while D learners use measured/explicit CPU fallback.
    d_device="cpu" if count==1 else device
    d=launch("D_suite","train_real_suite.py",["--archive",a.archive,"--output",a.output/"D","--epochs",50,"--seeds","1729","--training-episodes",64,"--samples",512,"--model-seconds",600,"--rl-steps",20000,"--deadline-seconds",max(1,a.seconds-(time.monotonic()-start)-1200),"--device",d_device,"--mission-text","--dynamic-ues","--burst"]+extension_args)
    if prepared_code==0:
        q=launch("Qwen","qwen_teacher.py",["--requests",prepared/"requests.jsonl","--output",a.output/"teacher","--revision",REVISION,"--device",teacher_device,"--max-calls",216,"--deadline-seconds",4200])
        qcode=wait(q,4500)
        if (a.output/"teacher/responses.jsonl").exists():
            pjob=launch("P_students","train_teacher_rankers.py",["--prepared",prepared,"--teacher-output",a.output/"teacher","--output",a.output/"P","--device",teacher_device,"--seeds","1729","--epochs",50,"--model-seconds",120])
            feedback=launch("AI_feedback_RL","train_feedback_rl.py",["--archive",a.archive,"--prepared",prepared,"--teacher-output",a.output/"teacher","--output",a.output/"AI_feedback","--device",teacher_device,"--seeds","1729","--steps",20000,"--model-seconds",600,"--mission-vectors",prepared/"data/mission_vectors.npy"]+extension_args)
            wait(pjob,1200);wait(feedback,1500)
    wait(d,max(1,a.seconds-(time.monotonic()-start)-120))
    if prepared_code==0 and a.seconds-(time.monotonic()-start)>180:
        profiler=launch("profile_exports","profile_exports.py",["--models",a.output,"--prepared",prepared,"--output",a.output/"systems_profile.json","--seconds",min(300,a.seconds-(time.monotonic()-start)-120)])
        wait(profiler,360)
    for name,job in jobs.items():
        proc=job["process"]
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        job["thread"].join(timeout=5)
    receipts={name:{"command":job["command"],"exit":job["process"].poll()} for name,job in jobs.items()}
    ledger={}
    arms=("linear_choice","mlp_choice","delay_huber","ranknet","lambdamart","priority_classifier","deepsets_ranknet","set_transformer_ranknet","listmle","temporal_deepsets_ranknet","masked_ppo","masked_double_dqn","td_per","ranknet_warmstart_ppo","rl_vlm_f","vlm_replay","visionprefer")
    for seed in (1729,2718,31415):
        for arm in arms:ledger[arm+"/"+str(seed)]={"status":"not_started_budget","reason":"first-seed breadth before additional seeds"}
    d_status=a.output/"D/status.json"
    if d_status.exists():ledger.update(json.loads(d_status.read_text()).get("arms",{}))
    for folder in ("P","AI_feedback"):
        status=a.output/folder/"status.json"
        if status.exists():
            entries=json.loads(status.read_text())
            for key,value in entries.items():
                if key.startswith(("visionprefer/","rl_vlm_f/","vlm_replay/")):ledger[key]=value
                else:ledger[folder+"/"+key]=value
    for arm in ("rl_vlm_f","vlm_replay","visionprefer"):
        key=arm+"/1729"
        if ledger[key]["status"] in ("planned","blocked_labels","not_started_budget"):
            ledger[key]={"status":"blocked_labels","reason":"No qualified trained feedback artifact; inspect Qwen, P_students and AI_feedback_RL logs and qualification"}
    ledger,outcome=summarize_execution(ledger,receipts)
    (a.output/"run_outcome.json").write_text(json.dumps(outcome,indent=2))
    (a.output/"method_ledger.json").write_text(json.dumps(ledger,indent=2))
    (a.output/"execution.json").write_text(json.dumps(receipts,indent=2))
    (a.output/"REPORT.md").write_text("# Bounded complete-method execution\\n\\nInspect D/status.json, P/status.json and AI_feedback/status.json. Teacher qualification failures and budget partials are not completed training.\\n")
    print("SUPERVISOR_FINISHED",json.dumps(outcome),flush=True)
if __name__=="__main__":main()
