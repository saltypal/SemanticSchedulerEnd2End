from pathlib import Path
import json, time
from experiments.stream_scheduler.semantic_suite_data import load_pool,load_mission_vectors
from Evaluation.semantic_learning_env import make_episode,Environment,Rule
root=Path.cwd()
run=root/"experiments/stream_scheduler/runs/debug-e2e-20261101"
selected=root/"experiments/stream_scheduler/runs/full-v22-launch-20261031T003706Z/retrieved_selected"
extension=next(selected.rglob("vectors.npz"));mission=next(selected.rglob("mission_vectors.npy"))
archive=root/"experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/Mini_dataset_6000.tar"
rows,features,vectors,profile,audit=load_pool(archive,run/"large_stream_input",extension)
text=load_mission_vectors(mission)
results=[]
for ues in (50,60,100):
    started=time.monotonic()
    ep=make_episode(rows,features,vectors,text,count=50*ues,ues=ues,load=1.2,
                    seed=98000+ues,churn=True,burst=True,repeat="none")
    for message in ep.messages:
        assert any(start<=message.arrival<=end and session==message.session
                   for start,end,session in ep.metadata["ue_active_intervals"][message.source])
    for kind in ("fifo","round_robin"):
        env=Environment(ep,"P");policy=Rule(kind);peak=0
        while not env.done:
            state=env.snapshot();peak=max(peak,len(state.ready));env.step(policy.select(state))
        metrics=env.metrics()
        assert metrics["selected"]+metrics["queued_at_horizon"]==len(ep.messages)
        assert metrics["completed"]+metrics["in_flight"]==metrics["selected"]
        results.append({"ues":ues,"offered":len(ep.messages),"policy":kind,"peak_ready":peak,
                        "selected":metrics["selected"],"queued_at_horizon":metrics["queued_at_horizon"],
                        "mission_relevance_available":ep.metadata["mission_relevance_available"],
                        "novelty_available":ep.metadata["novelty_available"],
                        "scope":"MIXED_IID timing/KB/lifecycle stress only; no urgency training/evaluation",
                        "elapsed_from_generation_seconds":time.monotonic()-started})
        (run/"large_stream_checks.json").write_text(json.dumps(results,indent=2))
        print("LARGE_STREAM",results[-1],flush=True)
