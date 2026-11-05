"""Targeted teacher recovery: never schedules the completed D/annotation RL arms."""
import json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from experiments.master_study.paired_runtime import PairRun,P_ARMS,SCRIPTS,write

class TeacherRecovery(PairRun):
    def assignments(self):
        return {"teacher_students":P_ARMS,"feedback":("rl_vlm_f","vlm_replay")}

    def execute(self):
        write(self.root/"experiment_scope.json",{
            "purpose":"repair teacher protocol and run only nine previously blocked methods",
            "preserved":"ten supervised D methods and four annotation RL methods are not retrained",
            "seed":1729,"epochs_max":50,"rl_actions_max":20000,
            "teacher":"same immutable Qwen2.5-VL-7B FP16 balanced across two GPUs; labeled panels and explicit aspect rubric",
            "gates":"unchanged32 visual controls,>=8 pilot contexts and80%orientation agreement;16/8strict support",
            "references":"teacher preferences are silver labels, not independent emergency truth",
            "query_design":"first8 train contexts fixed for repair comparison; remaining candidate/prefix contrasts use runtime features only; held-out preference accuracy is contrast-conditioned",
            "aspect_rule":"equal mean of >=2 co-available dimensions including mission usefulness, no unknown imputation",
            "codec":"already validated frozen SQ-GAN v12; no codec load or training in this recovery",
            "rate_study":"separate full CPU WSL run under experiments/stream_scheduler/runs/gap-repair-20261102T221315Z/rate_cpu_full"})
        prepared=self.root/"teacher_prepared"
        if not (prepared/"requests.jsonl").exists():
            if not self.command("prepare_teacher",SCRIPTS/"prepare_teacher.py",[
                "--archive",self.c["archive"],"--output",prepared,
                "--mission-vectors",self.c["mission_vectors"],"--vector-extension",self.c["vectors"]]):
                self.block_dependents("request preparation failed");self.report();return
        else:
            manifest=json.loads((prepared/"manifest.json").read_text())
            if manifest.get("teacher_protocol")!="labeled_teacher_panels_v1":
                raise ValueError("Recovery request cache uses another teacher protocol")
        teacher=self.root/"teacher"
        if not self.command("teacher",SCRIPTS/"qwen_teacher.py",[
            "--requests",prepared/"requests.jsonl","--output",teacher,
            "--revision","cc594898137f460bfe9f0759e9844b3ce807cfb5",
            "--device","cuda:0","--precision","fp16-balanced","--max-calls",216,
            "--deadline-seconds",max(1,self.stop-time.monotonic()-300)]):
            self.block_dependents("teacher qualification or execution failed");self.report();return
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(self.students,prepared,teacher),pool.submit(self.feedback,prepared,teacher)]
            for future in futures:future.result()
        self.report()

    def block_dependents(self,reason):
        write(self.root/"teacher_dependents.json",{"status":"blocked_teacher","reason":reason,
             "methods":list(P_ARMS)+["rl_vlm_f","vlm_replay"]})

    def students(self,prepared,teacher):
        args=[
            "--prepared",prepared,"--teacher-output",teacher,
            "--output",self.root/"jobs/teacher_students/P","--device","cuda:0",
            "--seeds","1729","--epochs",50,
            "--model-seconds",max(1,(self.stop-time.monotonic()-300)/7)]
        if (self.root/"jobs/teacher_students/P/status.json").exists():args+=["--resume"]
        return self.command("P",SCRIPTS/"train_teacher_rankers.py",args)

    def feedback(self,prepared,teacher):
        args=[
            "--archive",self.c["archive"],"--prepared",prepared,"--teacher-output",teacher,
            "--output",self.root/"jobs/feedback/feedback","--device","cuda:1",
            "--seeds","1729","--steps",20000,
            "--mission-vectors",self.c["mission_vectors"],"--vector-extension",self.c["vectors"],
            "--model-seconds",max(1,(self.stop-time.monotonic()-300)/2)]
        if (self.root/"jobs/feedback/feedback/status.json").exists():args+=["--resume"]
        return self.command("feedback",SCRIPTS/"train_feedback_rl.py",args)

    def report(self):
        prepared=self.root/"teacher_prepared"
        for job in ("teacher_students","feedback"):
            models=self.root/"jobs"/job
            profile=self.root/"profiles"/(job+".json")
            if (prepared/"contexts.pickle").exists() and any(models.rglob("model.json")) and not profile.exists():
                self.command("profile_"+job,SCRIPTS/"profile_exports.py",[
                    "--models",models,"--prepared",prepared,"--output",profile,
                    "--seconds",min(300,max(1,self.final-time.monotonic()-300))],training=False)
        super().report()
