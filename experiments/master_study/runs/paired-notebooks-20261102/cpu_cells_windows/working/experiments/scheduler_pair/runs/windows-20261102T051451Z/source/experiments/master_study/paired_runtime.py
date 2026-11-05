"""Shared orchestration for the two account notebooks; no training occurs on import."""
import hashlib,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/"experiments/stream_scheduler"
SUPERVISED=("linear_choice","mlp_choice","delay_huber","ranknet","lambdamart","priority_classifier",
            "deepsets_ranknet","set_transformer_ranknet","listmle","temporal_deepsets_ranknet")
WINDOWS0=("masked_double_dqn",)
WINDOWS1=("deepsets_ranknet","masked_ppo","td_per","ranknet_warmstart_ppo")
P_ARMS=("linear_choice","mlp_choice","ranknet","deepsets_ranknet","set_transformer_ranknet","temporal_deepsets_ranknet","visionprefer")

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False));temp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

class PairRun:
    def __init__(self,config):
        self.c=config;self.root=Path(config["run"]);self.root.mkdir(parents=True,exist_ok=True)
        self.stop=float(config["training_stop_monotonic"]);self.final=float(config["final_stop_monotonic"])
        self.lock=threading.Lock();self.stages={}
        self.expected=self.assignments()
        if not (self.root/"method_ledger.json").exists():
            write(self.root/"method_ledger.json",{job+"/"+arm+"/1729":{"status":"not_started"}
                for job,arms in self.expected.items() for arm in arms})
    def assignments(self):
        if self.c["profile"]=="windows":return {"double_dqn":WINDOWS0,"ppo_per":WINDOWS1}
        return {"supervised":SUPERVISED,"teacher_students":P_ARMS,"feedback":("rl_vlm_f","vlm_replay")}
    def command(self,name,script,args,training=True):
        budget=(self.stop if training else self.final)-time.monotonic()
        if budget<120:
            self.stages[name]={"status":"not_started_platform_reserve"};self.save();return False
        command=[sys.executable,"-u",str(script),*map(str,args)]
        self.stages[name]={"status":"running","command":command};self.save()
        print("START",name,"remaining_seconds",round(budget),flush=True)
        log=self.root/"logs"/(name+".log");log.parent.mkdir(parents=True,exist_ok=True)
        start=time.monotonic()
        with log.open("a",buffering=1) as f:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            def capture():
                for line in proc.stdout:
                    f.write(line);print("["+name+"]",line,end="",flush=True)
            thread=threading.Thread(target=capture);thread.start()
            try:exit_code=proc.wait(timeout=max(1,budget+60 if training else budget))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                exit_code=124
            thread.join()
        self.stages[name]={"status":"finished" if exit_code==0 else "partial_platform_budget" if exit_code==124 else "failed",
                           "exit":exit_code,"seconds":time.monotonic()-start,"command":command}
        self.save();return exit_code==0
    def save(self):
        with self.lock:write(self.root/"stages.json",dict(self.stages))
    def d_job(self,name,arms,device):
        out=self.root/"jobs"/name/"D"
        args=["--archive",self.c["archive"],"--output",out,"--seeds","1729","--epochs",50,"--rl-steps",20000,
              "--training-episodes",64,"--samples",512,"--validation-traces",15,"--arms",",".join(arms),
              "--device",device,"--mission-vectors",self.c["mission_vectors"],"--vector-extension",self.c["vectors"],
              "--stream-mode","mixed","--stream-duration",200,"--stream-burst-factor",4,
              "--objective","D","--require-sources","crisismmd",
              "--deadline-seconds",max(1,self.stop-time.monotonic()-90)]
        if (out/"config.json").exists():args+=["--resume"]
        return self.command(name,SCRIPTS/"train_real_suite.py",args)
    def teacher_jobs(self):
        prepared=self.root/"teacher_prepared"
        if not (prepared/"requests.jsonl").exists():
            if not self.command("prepare_teacher",SCRIPTS/"prepare_teacher.py",[
                "--archive",self.c["archive"],"--output",prepared,"--mission-vectors",self.c["mission_vectors"],
                "--vector-extension",self.c["vectors"]]):return
        teacher=self.root/"teacher"
        ok=self.command("teacher",SCRIPTS/"qwen_teacher.py",["--requests",prepared/"requests.jsonl","--output",teacher,
              "--revision","cc594898137f460bfe9f0759e9844b3ce807cfb5","--device","cuda:1","--max-calls",216,
              "--deadline-seconds",max(1,self.stop-time.monotonic()-90)])
        if not ok:
            write(self.root/"teacher_dependents.json",{"status":"blocked_teacher","methods":list(P_ARMS)+["rl_vlm_f","vlm_replay"]})
            return
        students=self.root/"jobs/teacher_students/P"
        args=["--prepared",prepared,"--teacher-output",teacher,"--output",students,"--device","cuda:1",
              "--seeds","1729","--epochs",50,"--model-seconds",max(1,(self.stop-time.monotonic()-120)/21)]
        if (students/"status.json").exists():args+=["--resume"]
        self.command("P",SCRIPTS/"train_teacher_rankers.py",args)
        feedback=self.root/"jobs/feedback/feedback"
        args=["--archive",self.c["archive"],"--prepared",prepared,"--teacher-output",teacher,
              "--output",feedback,"--device","cuda:1","--seeds","1729","--steps",20000,
              "--mission-vectors",self.c["mission_vectors"],"--vector-extension",self.c["vectors"],
              "--model-seconds",max(1,(self.stop-time.monotonic()-120)/2)]
        if (feedback/"status.json").exists():args+=["--resume"]
        self.command("feedback",SCRIPTS/"train_feedback_rl.py",args)
    def execute(self):
        write(self.root/"experiment_scope.json",{
          "reference":"CrisisMMD exposed/silver priority references only; not independently validated urgency",
          "ordinary_images":"cached, but excluded from D until independently assessable",
          "mission_E":"blocked unless independent mission utility manifest supplied; not silently replaced by D",
          "seed":1729,"epochs_max":50,"rl_actions_max":20000,
          "selected_codec":"SQ-GAN digital bytes, frozen; no codec training",
          "service_time":"synthetic1unit for scheduler learning; real SQ-GAN timing separate",
          "windows_deepsets":"initialization prerequisite and frozen PPO comparator; separate from WSL trained candidate"})
        with ThreadPoolExecutor(max_workers=2) as pool:
            if self.c["profile"]=="wsl":
                futures=[pool.submit(self.d_job,"supervised",SUPERVISED,"cuda:0"),pool.submit(self.teacher_jobs)]
            else:
                futures=[pool.submit(self.d_job,"double_dqn",WINDOWS0,"cuda:0"),pool.submit(self.d_job,"ppo_per",WINDOWS1,"cuda:1")]
            for f in futures:
                try:f.result()
                except Exception as e:
                    write(self.root/("controller_error_"+str(time.time_ns())+".json"),{"error":repr(e)})
                    self.report();raise
        if self.c["profile"]=="wsl":
            rate=self.root/"rate_study"
            if rate.exists() and not (rate/"REPORT.md").exists():
                self.stages["rate_study"]={"status":"preserved_partial_rate_outputs","reason":"not a learner checkpoint; inspect prior partial outputs before rerun"};self.save()
            elif not (rate/"REPORT.md").exists():
                self.command("rate_study",SCRIPTS/"ue_rate_study.py",["--archive",self.c["archive"],"--output",rate,
                    "--vector-extension",self.c["vectors"],"--mission-vectors",self.c["mission_vectors"],
                    "--ues","12,50,60,100","--seeds","1729"])
        self.report()
    def report(self):
        snapshots=next(iter((self.root/"jobs").glob("*/D/training_snapshots_1729.pickle")),None)
        for parent in sorted((self.root/"jobs").glob("*")):
            exports=list(parent.rglob("model.json"))
            profile_file=self.root/"profiles"/(parent.name+".json")
            prior=json.loads(profile_file.read_text()) if profile_file.exists() else {}
            identities={str(p.relative_to(parent)):json.loads(p.read_text()).get("sha256") for p in exports}
            unchanged=identities and all(prior.get(k,{}).get("model_sha256")==v for k,v in identities.items())
            if snapshots and exports and not unchanged:
                self.command("profile_"+parent.name,SCRIPTS/"profile_exports.py",
                    ["--models",parent,"--snapshots",snapshots,"--output",profile_file,
                     "--seconds",min(300,max(1,self.final-time.monotonic()-300))],training=False)
            self.command("report_"+parent.name,ROOT/"experiments/master_study/summarize_saved.py",
                         ["--results",parent,"--output",self.root/"reports"/parent.name,"--methods",",".join(self.expected.get(parent.name,()))],training=False)
        ledger={job+"/"+arm+"/1729":{"status":"not_started","family":"P" if job=="teacher_students" else "feedback" if job=="feedback" else "D"}
                for job,arms in self.expected.items() for arm in arms}
        for path in (self.root/"jobs").glob("*/*/status.json"):
            family=path.parent.name;job=path.parent.parent.name
            obj=json.loads(path.read_text());obj=obj.get("arms",obj)
            assigned=set(SUPERVISED if job=="supervised" else WINDOWS0 if job=="double_dqn" else WINDOWS1 if job=="ppo_per" else P_ARMS if family=="P" else ("rl_vlm_f","vlm_replay"))
            for key,row in obj.items():
                if key.rsplit("/",1)[0] in assigned:
                    ledger[job+"/"+key]={**row,"family":family}
        if (self.root/"teacher_dependents.json").exists():
            for key in ledger:
                if key.startswith(("teacher_students/","feedback/")):ledger[key]["status"]="blocked_teacher"

        write(self.root/"method_ledger.json",ledger)
        incomplete=[key for key,row in ledger.items() if not row.get("status","").startswith("evaluated")]
        write(self.root/"RUN_STATUS.json",{"status":"complete_method_attempts" if not incomplete and not any(v.get("exit",0)!=0 for v in self.stages.values()) else "partial_or_blocked",
             "incomplete":incomplete,"stage_failures":[k for k,v in self.stages.items() if v.get("exit",0)!=0],
             "scientific_final_evaluation":False})
        print("REPORTS_SAVED",self.root,flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Offline recovery report only; never launches training")
    parser.add_argument("--run",type=Path,required=True)
    args=parser.parse_args()
    saved=json.loads((args.run/"run_config.json").read_text())
    PairRun({"profile":saved["profile"],"run":str(args.run),
             "training_stop_monotonic":time.monotonic(),"final_stop_monotonic":time.monotonic()+1800}).report()
