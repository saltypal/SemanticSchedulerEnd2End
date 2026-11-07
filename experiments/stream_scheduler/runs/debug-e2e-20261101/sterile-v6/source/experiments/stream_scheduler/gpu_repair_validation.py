"""Hardware repair checks, separate from full-budget scientific model training."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time
import subprocess,sys,os
import traceback

ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location("boot",Path(__file__).with_name("runtime_bootstrap.py"))
boot=importlib.util.module_from_spec(sp);sp.loader.exec_module(boot);boot.install(ROOT)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cached-inputs",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=1000)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    import numpy as np
    import torch
    from Scheduler.tests.test_scene_learning_system import fixture
    from Evaluation.semantic_learning_env import Environment
    from Scheduler.semantic_learning_policy import NumpyRanker,fit_normalizer
    from Scheduler.training.semantic_learning_models import KINDS,Sample,fit_supervised,fit_lambdamart
    from Scheduler.training.semantic_learning_rl import train_dqn,train_ppo
    from Scheduler.training.preference_reward import FrozenReward
    from experiments.stream_scheduler.semantic_suite_data import load_mission_vectors
    from experiments.stream_scheduler.teacher_pipeline import prepare,qualify
    from experiments.stream_scheduler.qwen_teacher import run,MODEL_ID
    torch.set_num_threads(1)
    if torch.cuda.device_count()<2:raise RuntimeError("two CUDA devices required by this bounded validation profile")
    ledger={}
    environment=dict(os.environ);environment["PYTHONPATH"]=str(ROOT)+os.pathsep+str(ROOT/"src")
    channel=subprocess.run([sys.executable,"-m","pytest","-q",str(ROOT/"Test/tests/test_channel_mapping.py")],env=environment,capture_output=True,text=True)
    (args.output/"cuda_channel_tests.log").write_text(channel.stdout+channel.stderr)
    ledger["cuda_channel_regression"]={"status":"passed" if channel.returncode==0 else "failed","exit":channel.returncode}
    def save():
        (args.output/"validation.json").write_text(json.dumps(ledger,indent=2))
    episode=fixture();pre=fit_normalizer([episode])
    snapshot=Environment(episode).snapshot()
    sample=Sample(snapshot,Environment(episode).target(snapshot),np.arange(8)%4,"gpu-fixture")
    def validate(artifact):
        scores=NumpyRanker(artifact).scores(snapshot)
        return float(scores[0]-scores[-1]),{"scope":"synthetic fixture score; not semantic performance"}
    for arm in KINDS:
        try:
            teacher_sample=Sample(snapshot,np.zeros(8),np.full(8,-1),"gpu-teacher-fixture",
                                 ((0,7,1.),),"synthetic_teacher_fixture")
            artifact,history=fit_supervised(
                [teacher_sample if arm=="visionprefer" else sample],pre,arm,args.output/arm,
                validate,epochs=2,task="P" if arm=="visionprefer" else "D",device="cuda:0")
            ledger[arm]={"status":"passed","updates_checked":True,"device":"cuda:0"}
        except Exception:
            ledger[arm]={"status":"failed","traceback":traceback.format_exc()}
        save();print("CUDA_CHECK",arm,ledger[arm]["status"],flush=True)
    try:
        fit_lambdamart([sample]*5,pre,args.output/"lambdamart")
        ledger["lambdamart"]={"status":"passed","device":"cpu","scope":"tree native/export parity"}
    except Exception:ledger["lambdamart"]={"status":"failed","traceback":traceback.format_exc()}
    save()
    for arm in ("masked_double_dqn","td_per","rl_vlm_f","vlm_replay"):
        try:
            reward=None;guidance=None
            if arm=="rl_vlm_f":reward=FrozenReward(torch.nn.Linear(1172,1),pre,"gpu-fixture","cuda:0")
            if arm=="vlm_replay":guidance=lambda state,token,nxt:1.5 if token%2 else .5
            artifact,history=train_dqn(
                lambda index:Environment(episode,"P" if reward else "D"),pre,args.output/arm,validate,
                arm=arm,steps=504,device="cuda:0",reward_model=reward,guidance=guidance,
                feedback_version="gpu-fixture" if reward or guidance else None,training_id="gpu-fixture")
            ledger[arm]={"status":"passed","device":"cuda:0","decisions":history[-1]["steps"],
                         "updates":history[-1]["updates"],"scope":"synthetic mechanism check"}
        except Exception:ledger[arm]={"status":"failed","traceback":traceback.format_exc()}
        save();print("CUDA_CHECK",arm,ledger[arm]["status"],flush=True)
    for arm in ("masked_ppo","ranknet_warmstart_ppo"):
        try:
            warm=None
            if arm=="ranknet_warmstart_ppo":
                warm=torch.load(args.output/"deepsets_ranknet/resume.pt",weights_only=False,map_location="cpu")["best_state"]
            artifact,history=train_ppo(lambda index:Environment(episode),pre,args.output/arm,validate,
                steps=64,warmstart=warm,device="cuda:0",training_id="gpu-fixture")
            ledger[arm]={"status":"passed","device":"cuda:0","decisions":history[-1]["steps"],
                         "updates":history[-1]["updates"]}
        except Exception:ledger[arm]={"status":"failed","traceback":traceback.format_exc()}
        save();print("CUDA_CHECK",arm,ledger[arm]["status"],flush=True)
    # Reuse checksum-verified v4 producers; there is no vision/SPI re-extraction.
    prepared=args.output/"teacher_prepared"
    try:
        extension=args.cached_inputs/"vectors.npz"
        if not extension.exists():
            # Kaggle caps notebook source at1MB, so do not embed the5MB vector archive.
            # Recompute only missing pinned CLIP vectors; never rerun SPI/detectors.
            remaining=args.seconds-(time.monotonic()-started)
            if remaining<240:raise RuntimeError("insufficient budget for missing-vector completion")
            completion=args.output/"vector_completion"
            subprocess.run([sys.executable,"-u",str(ROOT/"experiments/stream_scheduler/complete_vectors.py"),
                            "--archive",str(args.archive),"--output",str(completion),"--device","cuda:0"],
                           check=True,timeout=min(300,remaining-180))
            extension=completion/"vectors.npz"
        prepare(args.archive,prepared,
                text_vectors=load_mission_vectors(args.cached_inputs/"mission_vectors.npy"),
                vector_extension=extension)
        requests=[json.loads(line) for line in (prepared/"requests.jsonl").read_text().splitlines()]
        selected=[row for row in requests if row["task"]=="control"]
        for task,number in (("candidate",4),("prefix",2),("replay",1)):
            selected.extend([row for row in requests if row["task"]==task][:number])
        subset=prepared/"validation_requests.jsonl"
        subset.write_text("\n".join(map(json.dumps,selected))+"\n")
        remaining=args.seconds-(time.monotonic()-started)
        if remaining<180:raise RuntimeError("insufficient hardware-check budget for teacher load; not launched")
        run(subset,args.output/"teacher","cc594898137f460bfe9f0759e9844b3ce807cfb5",
            device="cuda:1",max_calls=48,deadline_seconds=min(900,remaining-30))
        responses=[json.loads(line) for line in (args.output/"teacher/responses.jsonl").read_text().splitlines()]
        gate=qualify(subset,args.output/"teacher/responses.jsonl")
        counts=Counter(row["task"] for row in responses if row["status"]=="parsed")
        ledger["teacher"]={"status":"passed" if len(responses)==len(selected) and all(row["status"]=="parsed" for row in responses) and gate["qualified"] else "failed_or_partial",
                           "model":MODEL_ID,"requested":len(selected),"responses":len(responses),
                           "parsed_by_task":dict(counts),"qualification":gate,
                           "scope":"schema/GPU and narrow controls only; not enough labels for training or emergency-priority validation"}
    except Exception:ledger["teacher"]={"status":"failed","traceback":traceback.format_exc()}
    save()
    summary={"scope":"hardware repair validation; not full training, convergence or final evaluation",
             "seconds":time.monotonic()-started,"status_counts":dict(Counter(row["status"] for row in ledger.values())),
             "cuda_devices":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    (args.output/"REPORT.json").write_text(json.dumps(summary,indent=2))
    print("GPU_REPAIR_CHECK_FINISHED",json.dumps(summary),flush=True)
    return int(any(row["status"]!="passed" for row in ledger.values()))


if __name__=="__main__":raise SystemExit(main())
