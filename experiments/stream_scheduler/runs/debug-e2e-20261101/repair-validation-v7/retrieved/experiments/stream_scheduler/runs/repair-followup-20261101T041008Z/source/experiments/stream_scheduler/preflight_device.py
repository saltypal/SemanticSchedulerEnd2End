"""Measure a representative Deep Sets training update before choosing device."""
import importlib.util,json,pickle,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location("boot",Path(__file__).with_name("runtime_bootstrap.py"));b=importlib.util.module_from_spec(s);s.loader.exec_module(b);b.install(ROOT)
import torch
from dataclasses import replace
from Scheduler.training.semantic_learning_models import Scorer,tensors
def run(prepared,output,gpu="cuda:0"):
    with (Path(prepared)/"contexts.pickle").open("rb") as f:cache=pickle.load(f)
    snap=next(iter(cache["samples"].values())).snapshot;ready=tuple(replace(snap.ready[j%2],token=100000+j) for j in range(64))
    snap=replace(snap,ready=ready,ue_windows=tuple(snap.ue_windows[j%2] for j in range(64)),waiting_tokens=frozenset(m.token for m in ready))
    result={};torch.set_num_threads(1)
    for device in ("cpu",gpu):
        if device!="cpu" and not torch.cuda.is_available():continue
        model=Scorer("deepsets").to(device);opt=torch.optim.Adam(model.parameters(),lr=3e-4);items=[tensors(snap,cache["preprocessing"],device=device,kind="deepsets") for _ in range(16)]
        times=[]
        for j in range(15):
            if device!="cpu":torch.cuda.synchronize()
            start=time.perf_counter()
            opt.zero_grad();loss=torch.cat(model.forward_batch(items)).square().mean();loss.backward();opt.step()
            if device!="cpu":torch.cuda.synchronize()
            if j>=5:times.append((time.perf_counter()-start)*1000)
        result[device]={"median_update_ms":sorted(times)[len(times)//2],"batch_states":16,"candidates_per_state":64,"scope":"prepared tensors, forward/backward/optimizer; preparation/transfer excluded"}
    choice=min(result,key=lambda d:result[d]["median_update_ms"])
    row={"measured":result,"chosen_neural_device":choice,"torch":torch.__version__,"cuda_available":torch.cuda.is_available()}
    Path(output).write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True);return choice
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--prepared",required=True);p.add_argument("--output",required=True);a=p.parse_args();run(a.prepared,a.output)
