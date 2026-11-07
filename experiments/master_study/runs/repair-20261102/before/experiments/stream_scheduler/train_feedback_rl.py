"""Qualified AI-feedback reward and replay experiments on the shared environment."""
import argparse,hashlib,importlib.util,json,pickle,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location("boot",Path(__file__).with_name("runtime_bootstrap.py"));boot=importlib.util.module_from_spec(sp);sp.loader.exec_module(boot);boot.install(ROOT)
import numpy as np
import torch
from Scheduler.training.semantic_learning_models import mlp
from Scheduler.training.semantic_learning_rl import train_dqn
from Scheduler.training.preference_reward import fit_reward,integral,FrozenReward
from Scheduler.semantic_learning_policy import NumpyRanker,arrays
from Evaluation.semantic_learning_env import Environment,make_episode
from experiments.stream_scheduler.semantic_suite_data import load_pool,split_records
from experiments.stream_scheduler.teacher_pipeline import qualify
from experiments.stream_scheduler.train_real_suite import evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument("--archive",type=Path,required=True);p.add_argument("--prepared",type=Path,required=True);p.add_argument("--teacher-output",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cpu");p.add_argument("--steps",type=int,default=20000);p.add_argument("--model-seconds",type=float,default=None);p.add_argument("--seeds",default="1729,2718,31415");p.add_argument("--mission-vectors",type=Path)
    p.add_argument("--vector-extension",type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    gate=qualify(a.prepared/"requests.jsonl",a.teacher_output/"responses.jsonl")
    if not gate["qualified"]:
        blocked={arm+"/"+seed:{"status":"blocked_labels","reason":"teacher control qualification failed","gate":gate} for seed in a.seeds.split(",") for arm in ("rl_vlm_f","vlm_replay")}
        (a.output/"status.json").write_text(json.dumps(blocked,indent=2));return
    responses={r["id"]:r for r in map(json.loads,(a.teacher_output/"responses.jsonl").read_text().splitlines())}
    with (a.prepared/"prefix_contexts.pickle").open("rb") as f:cache=pickle.load(f)
    pre=cache["preprocessing"];pairs={"train":[],"validation":[]}
    opposite={"A":"B","B":"A","tie":"tie"}
    for key,record in cache["prefixes"].items():
        x=responses.get(key+"-0",{}).get("parsed",{}).get("choice");y=responses.get(key+"-1",{}).get("parsed",{}).get("choice")
        if x not in opposite or y!=opposite[x]:continue
        pairs[record["role"]].append((record["left"],record["right"],{"A":1.,"B":0.,"tie":.5}[x]))
    strict=lambda rows:[r for r in rows if r[2]!=.5]
    ledger={arm+"/"+seed:{"status":"not_started"} for seed in a.seeds.split(",") for arm in ("rl_vlm_f","vlm_replay")}
    (a.output/"status.json").write_text(json.dumps(ledger,indent=2))
    rows,features,vectors,profile,audit=load_pool(a.archive,a.output/"data",a.vector_extension)
    cohort=split_records(rows,"train",True)
    hold={r["group"] for r in cohort if int(hashlib.sha256(r["group"].encode()).hexdigest()[:8],16)%5==0}
    train=[r for r in cohort if r["group"] not in hold];val=[r for r in cohort if r["group"] in hold]
    text=np.load(a.mission_vectors) if a.mission_vectors else None
    def episode(role,seed):
        records=train if role=="train" else val
        return make_episode(records,features,vectors,text,count=min(96,len(records)),ues=(4,12,50,60,100)[seed%5],load=(.5,.9,1.2)[seed%3],seed=seed)
    validation=[episode("validation",70000+j) for j in range(15)]
    def validate(model):return evaluate(NumpyRanker(model),validation)
    version=hashlib.sha256((a.teacher_output/"responses.jsonl").read_bytes()).hexdigest()
    reward=None;guide=None
    if len(strict(pairs["train"]))>=16 and len(strict(pairs["validation"]))>=8:
        torch.manual_seed(1729)
        model,history=fit_reward(mlp(1172,1),pairs["train"],device=a.device)
        expected=[];pred=[]
        with torch.no_grad():
            for left,right,t in strict(pairs["validation"]):
                ra=sum(integral(model,path,a.device) for path in left);rb=sum(integral(model,path,a.device) for path in right)
                expected.append(t);pred.append(float(ra>rb) if ra!=rb else -1.)
        accuracy=float(np.mean(np.array(expected)==pred));majority=max(np.mean(expected),1-np.mean(expected))
        (a.output/"reward_validation.json").write_text(json.dumps({"accuracy":accuracy,"majority":float(majority),"train_strict":len(strict(pairs["train"])),"validation_strict":len(expected),"loss":history},indent=2))
        torch.save({"state":model.state_dict(),"preprocessing":pre,"feedback_version":version},a.output/"reward.pt")
        if accuracy>=.75 and accuracy>majority:reward=FrozenReward(model,pre,version,a.device)
    x=[];y=[]
    for j,(key,sample) in enumerate(cache["replay"].items()):
        label=responses.get("replay-"+str(j),{}).get("parsed",{}).get("value")
        if label not in ("low","neutral","high"):continue
        x.append(arrays(sample.snapshot,pre,False)["summary"][0]);y.append({"low":.5,"neutral":1.,"high":1.5}[label])
    if len(x)>=8:
        x=np.asarray(x,dtype=float);y=np.asarray(y);center=x.mean(0);scale=np.maximum(x.std(0),1.);z=(x-center)/scale
        bias=float(y.mean());coef=z.T@np.linalg.solve(z@z.T+np.eye(len(z)),y-bias)
        np.savez(a.output/"replay_guide.npz",center=center,scale=scale,coef=coef,bias=bias)
        def guide(state,token,nxt):
            index=next(j for j,m in enumerate(state.ready) if m.token==token)
            v=arrays(state,pre,False)["summary"][index];value=float(((v-center)/scale)@coef+bias)
            return .5 if value<.75 else 1.5 if value>1.25 else 1.
    for seed in map(int,a.seeds.split(",")):
        for arm in ("rl_vlm_f","vlm_replay"):
            key=arm+"/"+str(seed);dest=a.output/arm/str(seed)
            if (arm=="rl_vlm_f" and reward is None) or (arm=="vlm_replay" and guide is None):
                ledger[key]={"status":"blocked_labels","reason":"insufficient qualified prefix support/reward validation or replay guidance"};continue
            try:
                factory=lambda j:Environment(episode("train",seed*10000+j),"P" if arm=="rl_vlm_f" else "D")
                artifact,history=train_dqn(factory,pre,dest,validate,arm=arm,steps=a.steps,seed=seed,seconds=a.model_seconds,device=a.device,reward_model=reward if arm=="rl_vlm_f" else None,guidance=guide if arm=="vlm_replay" else None,feedback_version=version,training_id=version)
                ledger[key]={"status":"evaluated_development_only" if history[-1]["steps"]>=a.steps else "partial_budget","completed_decisions":history[-1]["steps"],"validation":validate(artifact)[0]}
            except Exception as e:
                dest.mkdir(parents=True,exist_ok=True);(dest/"error.txt").write_text(traceback.format_exc());ledger[key]={"status":"failed","error":str(e)}
            (a.output/"status.json").write_text(json.dumps(ledger,indent=2));print("FEEDBACK_RL",key,ledger[key],flush=True)
    (a.output/"status.json").write_text(json.dumps(ledger,indent=2))
if __name__=="__main__":main()
