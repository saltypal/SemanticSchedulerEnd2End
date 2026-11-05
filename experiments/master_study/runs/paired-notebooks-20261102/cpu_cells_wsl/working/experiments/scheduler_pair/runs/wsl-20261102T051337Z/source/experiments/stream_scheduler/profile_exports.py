"""CPU deployment timing and partial operator FLOPs from actual exported weights."""
import argparse,importlib.util,json,pickle,time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location("boot",Path(__file__).with_name("runtime_bootstrap.py"));b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b);b.install(ROOT)
import numpy as np
import torch
from threadpoolctl import threadpool_info,threadpool_limits
from Scheduler.semantic_learning_policy import NumpyRanker
from Scheduler.training.semantic_learning_models import Scorer,tensors
def main():
    p=argparse.ArgumentParser();p.add_argument("--models",type=Path,required=True);group=p.add_mutually_exclusive_group(required=True);group.add_argument("--prepared",type=Path);group.add_argument("--snapshots",type=Path);p.add_argument("--output",type=Path,required=True);p.add_argument("--seconds",type=float,default=300);a=p.parse_args()
    if a.snapshots:
        with a.snapshots.open("rb") as f:base=pickle.load(f)[0].snapshot
    else:
        with (a.prepared/"contexts.pickle").open("rb") as f:base=next(iter(pickle.load(f)["samples"].values())).snapshot
    a.output.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();results={};torch.set_num_threads(1);thread_guard=threadpool_limits(limits=1)
    for path in a.models.rglob("model.json"):
        if time.monotonic()-start>a.seconds:break
        try:
            artifact=json.loads(path.read_text());policy=NumpyRanker(artifact);row={"model_sha256":artifact["sha256"],"artifact_json_bytes":path.stat().st_size,"native_threadpools":threadpool_info(),"torch_threads":torch.get_num_threads(),"weight_numpy_bytes":sum(v.nbytes for v in policy.w.values()),"timing":[],"scope":"synthetic queue-shape probes using real cached headers; CPU NumPy deployment, one thread, 5 warmups and20 repeats; no decoder/channel"}
            model=None
            if artifact["kind"]!="trees":
                model=Scorer(artifact["kind"]);model.load_state_dict({k:torch.tensor(v) for k,v in artifact["weights"].items()});model.eval()
                row["parameters"]=sum(p.numel() for p in model.parameters())
            for size in (4,16,64,256):
                ready=tuple(replace(base.ready[j%2],token=100000+j) for j in range(size))
                snap=replace(base,ready=ready,ue_windows=tuple(base.ue_windows[j%2] for j in range(size)),waiting_tokens=frozenset(m.token for m in ready))
                times=[]
                for repeat in range(25):
                    t=time.perf_counter_ns();policy.select(snap);elapsed=(time.perf_counter_ns()-t)/1e6
                    if repeat>=5:times.append(elapsed)
                measured={"queue":size,"context_and_select_p50_ms":float(np.median(times)),"context_and_select_p95_ms":float(np.quantile(times,.95))}
                if model is not None:
                    inputs=tensors(snap,artifact["preprocessing"],kind=model.kind)
                    with torch.inference_mode():
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:model(inputs)
                    measured["supported_operator_flops"]=sum(e.flops for e in prof.key_averages())
                    measured["flops_scope"]="Torch profiler supported matmul/convolution operators,2ops/MAC; not total FLOPs; normalization/activation/pooling and context construction not fully counted"
                row["timing"].append(measured)
            results[str(path.relative_to(a.models))]=row
        except Exception as error:results[str(path)]={"status":"unavailable_or_failed","error":str(error)}
        a.output.write_text(json.dumps(results,indent=2));print("PROFILE",path,flush=True)
if __name__=="__main__":main()
