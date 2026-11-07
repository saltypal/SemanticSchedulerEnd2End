from pathlib import Path
import json,sys,importlib.util,numpy as np
root=Path.cwd();sys.path.insert(0,str(root))
sp=importlib.util.spec_from_file_location("boot",root/"experiments/stream_scheduler/runtime_bootstrap.py")
boot=importlib.util.module_from_spec(sp);sp.loader.exec_module(boot);boot.install(root)
from experiments.stream_scheduler.semantic_suite_data import load_pool,development_cohorts
from experiments.stream_scheduler.train_real_suite import collect
from Evaluation.ue_scenarios import build_stream,StreamConfig
from experiments.stream_scheduler.teacher_pipeline import preference_reliability
from Scheduler.training.semantic_learning_models import supervised_loss
import torch
torch.set_num_threads(1)
out=root/"experiments/master_study/runs/repair-20261102/pilot";out.mkdir(exist_ok=True)
archive=root/"experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/Mini_dataset_6000.tar"
rows,features,vectors,profile,audit=load_pool(archive,out/"data")
train,val=development_cohorts(rows)
print("Loaded existing cache; no vision extraction. Building64 complete training trajectories.",flush=True)
episodes=[build_stream(train,features,vectors,None,StreamConfig(mode="mixed",ues=(4,12,50,60,100)[j%5],load=(.5,.9,1.2)[j%3],seed=172900+j,duration=200.,burst_factor=4)) for j in range(64)]
samples=collect(episodes,512)
now=np.array([s.snapshot.now for s in samples]);qs=np.array([len(s.snapshot.ready) for s in samples])
report={"scope":"CPU cache/coverage pilot; no parameter extraction, fitting or new teacher inference",
 "samples":len(samples),"phase_counts":[int(np.sum(now<80)),int(np.sum((now>=80)&(now<120))),int(np.sum(now>=120))],
 "max_queue":int(qs.max()),"median_queue":float(np.median(qs)),
 "listwise_supported":sum(supervised_loss(torch.zeros(len(s.target)),s,"listmle") is not None for s in samples),
 "soft_SPI_rows":int(sum(np.count_nonzero(v[374:378])>1 for v in features.values())),
 "source_counts_training":{source:sum(r["dataset"]==source for r in train) for source in audit["source_counts"]}}
assert report["phase_counts"][2]>0 and report["max_queue"]>11
base=root/"experiments/master_study/runs/launch-20261101/v9/retrieved/experiments/stream_scheduler/runs/master-study-20261101T062547Z/results"
report["old_teacher_consistency"]=preference_reliability(base/"prepared/requests.jsonl",base/"teacher/responses.jsonl")
assert not report["old_teacher_consistency"]["qualified"]
report["external_blockers"]=["Open Images/Places365 priority references absent","new mission utility manifest absent","new visual preference correctness not measured"]
(out/"pilot.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)

