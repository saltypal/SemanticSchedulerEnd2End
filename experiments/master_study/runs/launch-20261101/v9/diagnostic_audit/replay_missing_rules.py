"""Fixed v9 workload replay: two missing rules, no training or tuning."""
from pathlib import Path
import sys,json,importlib.util
import numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location("_boot",ROOT/"experiments/stream_scheduler/runtime_bootstrap.py")
boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.install(ROOT)
from experiments.stream_scheduler.semantic_suite_data import load_pool,load_mission_vectors
from Evaluation.ue_scenarios import StreamConfig,build_stream
from Evaluation.semantic_learning_env import Rule,ell,BUDGETS
from experiments.stream_scheduler.train_real_suite import evaluate
RUN=ROOT/"experiments/master_study/runs/launch-20261101/v9"
B=RUN/"retrieved/experiments/stream_scheduler/runs/master-study-20261101T062547Z/results"
OUT=RUN/"diagnostic_audit";OUT.mkdir(exist_ok=True)
archive=ROOT/"experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/Mini_dataset_6000.tar"
rows,features,vectors,profile,audit=load_pool(archive,OUT/"data")
split=json.loads((B/"D/development_split.json").read_text())
ids=set(split["validation_ids"]);val=[r for r in rows if r["image_id"] in ids]
text=None # These diagnostic rules use only SPI and age; R/N are unused.
episodes=[build_stream(val,features,vectors,text,StreamConfig(mode="mixed",ues=u,load=l,seed=70000+i,duration=200.,burst_factor=4.))
 for i,(u,l) in enumerate((u,l) for u in (4,12,50,60,100) for l in (.5,.9,1.2))]
saved=json.loads((B/"D/baseline_summary.json").read_text());result={}
for name in ("fifo","spi_fifo"):
 score,met=evaluate(Rule(name),episodes)
 if abs(score-saved[name]["score"])>1e-10:raise ValueError("frozen workload reproduction mismatch "+name)
 result[name]={"score":score,"episodes":met}
print("Reproduced original FIFO and SPI-first exactly.",flush=True)
class Risk:
 def select(self,s):
  def risk(m):
   p=m.features[374:378]
   w=s.now-m.arrival
   return float(p@(ell(w+1,BUDGETS)-ell(w,BUDGETS)))
  return max(s.ready,key=risk).token
score,met=evaluate(Risk(),episodes);result["received_SPI_marginal_risk"]={"score":score,"episodes":met}
print("Received SPI marginal-risk score",score,flush=True)
# Explicitly privileged diagnostics only, not fair runtime competitors.
from Evaluation.semantic_learning_env import Environment
for name in ("reference_SPI_first","reference_marginal_risk"):
 metrics=[]
 for ep in episodes:
  env=Environment(ep)
  while not env.done:
   st=env.snapshot()
   if name=="reference_SPI_first":
    token=max(st.ready,key=lambda m:-ep.reference_tiers[env.index[m.token]]).token
   else:token=st.ready[int(np.argmax(env.target(st)))].token
   env.step(token)
  metrics.append(env.metrics())
 result[name]={"score":-float(np.mean([m["J_D"] for m in metrics])),"episodes":metrics,"privileged":True}
 print(name,result[name]["score"],flush=True)
(OUT/"missing_baseline_replay.json").write_text(json.dumps(result,indent=2))

