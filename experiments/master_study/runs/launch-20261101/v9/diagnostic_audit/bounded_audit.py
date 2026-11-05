"""Read-only v9 diagnosis; generated diagnostics only, no training."""
from pathlib import Path
import sys,json,pickle,hashlib,collections,itertools
import numpy as np
ROOT=Path.cwd()
sys.path.insert(0,str(ROOT))
import importlib.util
spec=importlib.util.spec_from_file_location("_boot",ROOT/"experiments/stream_scheduler/runtime_bootstrap.py")
boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.install(ROOT)
from Evaluation.semantic_learning_env import Environment,Episode,Message,Snapshot,Rule
from Scheduler.semantic_learning_policy import NumpyRanker,fit_normalizer,arrays
from Scheduler.training.semantic_learning_models import supervised_loss,Scorer
from experiments.stream_scheduler.teacher_pipeline import qualify
import torch
torch.set_num_threads(1)
RUN=ROOT/"experiments/master_study/runs/launch-20261101/v9"
B=RUN/"retrieved/experiments/stream_scheduler/runs/master-study-20261101T062547Z/results"
OUT=RUN/"diagnostic_audit";OUT.mkdir(exist_ok=True)
report={}
path=B/"D/training_snapshots_1729.pickle"
if path.exists():
 with path.open("rb") as f:samples=pickle.load(f)
 now=np.array([s.snapshot.now for s in samples]);sizes=np.array([len(s.snapshot.ready) for s in samples])
 report["training_support"]={"n":len(samples),"episodes":len(set(s.episode_id for s in samples)),
  "decision_time_quantiles":dict(zip(["min","q25","q50","q75","max"],np.quantile(now,[0,.25,.5,.75,1]).tolist())),
  "queue_size_quantiles":np.quantile(sizes,[0,.5,.9,1]).tolist(),
  "before_burst_80":float(np.mean(now<80)),"during_burst_80_120":float(np.mean((now>=80)&(now<120))),
  "after_burst_120":float(np.mean(now>=120)),
  "episode_counts":dict(collections.Counter(s.episode_id for s in samples)),
  "tied_top_fraction":float(np.mean([np.sum(s.target==np.max(s.target))>1 for s in samples])),
  "fewer_than_3_distinct_targets":float(np.mean([len(set(s.target))<3 for s in samples]))}
 with torch.no_grad():
  no_gradient=sum(supervised_loss(torch.zeros(len(s.target)),s,"listmle") is None for s in samples)
 report["training_support"]["listmle_no_gradient_samples"]=no_gradient
 x=np.concatenate([np.array([m.features for m in s.snapshot.ready]) for s in samples])
 report["training_support"]["candidate_rows"]=len(x)
 report["training_support"]["scene_value_fields_varying"]=int(np.sum(np.ptp(x[:,:186],axis=0)>0))
 report["training_support"]["novelty_available_fraction"]=float(np.mean(x[:,385]))
 report["training_support"]["relevance_available_fraction"]=float(np.mean(x[:,384]))
 report["training_support"]["spi_available_fraction"]=float(np.mean(x[:,380]))
 report["training_support"]["novelty_cold_fraction"]=float(np.mean(x[:,386]))
# Compare saved per-trace metrics, avoiding repeated random seeds as independent groups.
baseline=json.loads((B/"D/baseline_summary.json").read_text())
comparison={}
def aggregate(episodes):
 out={"episodes":len(episodes),"mean_J_D":float(np.mean([e["J_D"] for e in episodes]))}
 for tier in ("Critical","High","Medium","Low"):
  rows=[e["by_reference_tier"][tier] for e in episodes]
  selected=sum(r["selected"] for r in rows);offered=sum(r["offered"] for r in rows)
  eligible=sum(r["eligible"] for r in rows)
  out[tier]={"offered":offered,"selected":selected,"unselected":offered-selected,
    "pooled_selected_wait":sum((r["mean_selected_wait"] or 0)*r["selected"] for r in rows)/selected if selected else None,
    "within_budget":sum((r["within_budget"] or 0)*r["eligible"] for r in rows)/eligible if eligible else None}
 return out
for name,v in baseline.items():comparison[name]=aggregate(v["episodes"])
for name in ("ranknet","deepsets_ranknet","masked_ppo","masked_double_dqn","delay_huber","priority_classifier"):
 ps=sorted((B/"D/models"/name/"1729/validation").glob("episode_*.json"))
 if ps:comparison[name]=aggregate([json.loads(p.read_text())["metrics"] for p in ps])
report["same_workload_results"]=comparison
report["ranking_probes"]={}
for p in B.rglob("ranking_probes.json"):
 if "1729" in str(p) or "baselines" in str(p):
  report["ranking_probes"][str(p.relative_to(B))]=json.loads(p.read_text())
# Teacher correctness control gate and qualification yield breakdown.
req=B/"prepared/requests.jsonl";res=B/"teacher/responses.jsonl"
if req.exists() and res.exists():
 report["teacher_gate"]=qualify(req,res)
 queries={q["id"]:q for q in map(json.loads,req.read_text().splitlines())}
 answers={a["id"]:a for a in map(json.loads,res.read_text().splitlines())}
 summary={}
 opp={"A":"B","B":"A","tie":"tie","uncertain":"uncertain"}
 for task in ("candidate","prefix"):
  for role in ("train","validation"):
   first=[q for q in queries.values() if q["task"]==task and q["split"]==role and not q["swapped"]]
   counts=collections.Counter()
   for q in first:
    ident=q["context_id"];a=answers.get(ident+"-0",{});b=answers.get(ident+"-1",{})
    x=a.get("parsed",{}).get("choice");y=b.get("parsed",{}).get("choice")
    counts["contexts"]+=1
    counts["first_choice_"+str(x)]+=1;counts["second_choice_"+str(y)]+=1
    if a.get("status")!="parsed" or b.get("status")!="parsed":counts["invalid_orientation_pair"]+=1
    elif x not in opp or y!=opp[x]:counts["orientation_disagreement"]+=1
    elif x=="uncertain":counts["consistent_uncertain"]+=1
    elif x=="tie":counts["consistent_tie"]+=1
    else:counts["accepted_strict"]+=1
   summary[task+"/"+role]=dict(counts)
 report["teacher_yield"]=summary
# Numerical counterexamples using actual imports.
x=np.zeros(387);x[375]=1;x[380:384]=1
x2=x.copy();x2[378]=.9;x2[379]=.9;x2[0]=7;x2[384:386]=1
msgs=(Message(0,0,0,0,0,x),Message(1,1,0,0,1,x2))
ep=Episode(msgs,(1,1),("same-fact","same-fact"),(1.,1.),(False,False),4.,{})
a=Environment(ep);target=a.target().tolist()
a.step(0);a.step(1);cost_a=a.cost(ep.horizon)
b=Environment(ep);b.step(1);b.step(0);cost_b=b.cost(ep.horizon)
report["objective_semantic_indifference"]={"target":target,"cost_order_A_B":cost_a,"cost_order_B_A":cost_b,
 "explanation":"same SPI and arrival; different scene, relevance, novelty. Exact-repeat fact identities also do not affect D."}
fail=Episode((msgs[0],),(1,),("f",),(1.,),(True,),4.,{})
e=Environment(fail);e.step(0)
report["failed_completion_history"]={"processed_history_size":len(e.processed),"failure_count":e.metrics()["failed"],
 "fact_marked_covered":"f" in e.covered}
pre=fit_normalizer([ep]);one=Environment(Episode((msgs[0],),(1,),("a",),(1.,),(False,),4.,{})).snapshot()
three_msgs=tuple(Message(i,i,0,0,i,x) for i in range(3))
three=Environment(Episode(three_msgs,(1,)*3,("a","b","c"),(1.,)*3,(False,)*3,4.,{})).snapshot()
from Scheduler.training.semantic_learning_rl import ActorCritic
from Scheduler.training.semantic_learning_models import tensors
torch.manual_seed(1729);model=ActorCritic()
with torch.no_grad():v1=float(model(tensors(one,pre))[1]);v3=float(model(tensors(three,pre))[1])
report["ppo_critic_queue_count_alias"]={"value_one":v1,"value_three":v3}
# Oracle marginal rule as diagnostic only: received SPI for fair rule, actual reference for privileged bound.
def marginal_cost_order(tiers,waits,order):
 from Evaluation.semantic_learning_env import ell,BUDGETS
 return sum(float(ell(waits[i]+k,BUDGETS[tiers[i]])) for k,i in enumerate(order))
tiers=[1,0,0,3];waits=[4,1,1,0]
orders=list(itertools.permutations(range(4)))
best=min(orders,key=lambda z:marginal_cost_order(tiers,waits,z))
from Evaluation.semantic_learning_env import ell,BUDGETS
remaining=list(range(4));greedy=[]
for step in range(4):
 i=max(remaining,key=lambda i:float(ell(waits[i]+step+1,BUDGETS[tiers[i]])-ell(waits[i]+step,BUDGETS[tiers[i]])))
 greedy.append(i);remaining.remove(i)
report["immediate_teacher_not_global_optimal"]={"greedy":greedy,"cost_greedy":marginal_cost_order(tiers,waits,greedy),
 "exact":list(best),"cost_exact":marginal_cost_order(tiers,waits,best)}
report["source_hashes"]={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
 [ROOT/"Evaluation/semantic_learning_env.py",ROOT/"Evaluation/ue_scenarios.py",
 ROOT/"Scheduler/training/semantic_learning_models.py",ROOT/"Scheduler/training/semantic_learning_rl.py",
 ROOT/"Scheduler/semantic_learning_policy.py",ROOT/"experiments/stream_scheduler/train_real_suite.py"]}
(OUT/"diagnostics.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ("ranking_probes","same_workload_results")},indent=2))
print("Saved",OUT/"diagnostics.json")


# Complete the audit provenance and negative teacher-guidance check.
responses=[json.loads(x) for x in res.read_text().splitlines()]
guide_path=B/"feedback/replay_guide.npz"
if guide_path.exists():
 with np.load(guide_path) as g:
  report["neutral_replay_guidance"]={"ratings":dict(collections.Counter(r["parsed"]["value"] for r in responses if r["task"]=="replay")),
   "bias":float(g["bias"]),"coefficient_norm":float(np.linalg.norm(g["coef"])),
   "consequence":"constant VLM multiplier1; replay is50% uniform +50% TD-prioritized"}
manifest=json.loads((B.parent/"manifest.json").read_text())["source_hashes"]
report["source_hashes_match_v9"]={k:v==manifest.get(k) for k,v in report["source_hashes"].items()}
import tarfile
archive=ROOT/"experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/Mini_dataset_6000.tar"
with tarfile.open(archive) as tar:
 all_rows=[json.loads(line) for line in tar.extractfile("Mini_dataset/handoff/image_parameters.jsonl") if line.strip()]
 labels=json.load(tar.extractfile("Mini_dataset/student/training_records.json"))
labels_by_hash={}
for label in labels:labels_by_hash.setdefault(label["image_sha256"],set()).add(label["level"])
splits=json.loads((B/"D/development_split.json").read_text())
report["reference_data_coverage"]={"known_reference_images":dict(collections.Counter(row["dataset"] for row in all_rows if len(labels_by_hash.get(row["image_sha256"],set()))==1))}
for role in ("train","validation"):
 eligible=set(splits[role+"_ids"])
 report["reference_data_coverage"]["scheduler_"+role]=dict(collections.Counter(row["dataset"] for row in all_rows if row["image_id"] in eligible))
(OUT/"diagnostics.json").write_text(json.dumps(report,indent=2))
