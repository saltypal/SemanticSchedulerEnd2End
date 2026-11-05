"""Duration-aware post-dispatch preference reward; images/labels excluded at inference."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch.nn import functional as F
from Evaluation.semantic_learning_env import Snapshot
from Scheduler.semantic_learning_policy import arrays,normalized

@dataclass(frozen=True)
class RatePath:
    vectors:tuple
    durations:tuple

def transition_path(s,chosen,env,pre,end=None):
    end=env.now if end is None else float(end)
    if end<s.now:raise ValueError("negative transition duration")
    selected=next(m for m in s.ready if m.token==chosen)
    event=next(e for e in reversed(env.events) if e["token"]==chosen)
    finish=event["finish"];start=s.now
    arriving=[m for m in env.episode.messages if start<m.arrival<=end]
    boundaries={start,end}
    boundaries.update(m.arrival for m in arriving)
    if start<finish<end:boundaries.add(finish)
    boundaries.update(k*.25 for k in range(math.floor(start/.25)+1,math.ceil(end/.25)) if start<k*.25<end)
    boundaries=sorted(boundaries);vectors=[];durations=[]
    for left,right in zip(boundaries,boundaries[1:]):
        waiting=tuple(m for m in s.ready if m.token!=chosen)+tuple(m for m in arriving if m.arrival<=left)
        if not waiting:continue  # Zero rate while no work waits, including idle gaps.
        history=list(s.processed)
        if finish<=left:history.append((selected,finish))
        snap=Snapshot(waiting,tuple(history[-32:]),tuple(() for _ in waiting),(),left,s.attempts+int(finish<=left),frozenset(m.token for m in waiting),(chosen,) if left<finish else ())
        a=arrays(snap,pre,False)
        selected_token=np.concatenate([normalized((selected,),pre)[0],[np.log1p(left-selected.arrival),2. if left<finish else 1.]])
        # Reuse the exact masked queue/history summaries; selected report is excluded from queue mean.
        vector=a["summary"][0].copy();vector[:389]=selected_token;vector[-2:]=0.
        vectors.append(tuple(map(float,vector)));durations.append(float(right-left))
    return RatePath(tuple(vectors),tuple(durations))

def integral(model,path,device):
    if not path.vectors:return sum((p.sum()*0 for p in model.parameters()))
    x=torch.tensor(path.vectors,dtype=torch.float32,device=device)
    dt=torch.tensor(path.durations,dtype=torch.float32,device=device)
    return (-torch.sigmoid(model(x).flatten())*dt).sum()

def fit_reward(model,pairs,*,epochs=50,device="cpu"):
    # pairs contains (equal-time path-prefix A, path-prefix B, preference A).
    if not pairs:raise ValueError("no qualified prefix preferences")
    model=model.to(device);opt=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-4)
    history=[]
    for epoch in range(epochs):
        losses=[]
        for left,right,target in pairs:
            ra=sum(integral(model,p,device) for p in left);rb=sum(integral(model,p,device) for p in right)
            loss=F.binary_cross_entropy_with_logits(ra-rb,torch.tensor(float(target),device=device))
            if not torch.isfinite(loss):raise ValueError("nonfinite preference reward loss")
            opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model,history

class FrozenReward:
    def __init__(self,model,pre,version,device="cpu"):
        if not version:raise ValueError("frozen reward version required")
        self.model=model.to(device).eval();self.pre=pre;self.version=version;self.device=device
    def path(self,s,chosen,nxt,env):return transition_path(s,chosen,env,self.pre)
    def value(self,path):
        with torch.no_grad():return float(integral(self.model,path,self.device))
    def __call__(self,s,chosen,nxt,env):return self.value(self.path(s,chosen,nxt,env))

def relabel_replay(replay,reward):
    result=[]
    for row in replay:
        if len(row)!=6:raise ValueError("immutable reward path missing from replay")
        s,a,old,ns,end,path=row
        result.append((s,a,reward.value(path),ns,end,path))
    return result
