from pathlib import Path
import hashlib, json, pickle, sys
import numpy as np
import torch
root=Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End');sys.path.insert(0,str(root))
import project_compat; project_compat.install()
from Scheduler.training.semantic_learning_models import Scorer, tensors
from Scheduler.training.preference_reward import transition_path, integral
from Evaluation.semantic_learning_env import Message, Episode, Environment
run=root/'experiments/stream_scheduler/runs/implementation-v22-checks-20261030T235934Z/batched_reward_checks_20261031'
source=root/'experiments/stream_scheduler/runs/v22-real-integration-20261031T000527Z/results'
samples=pickle.load((source/'training_snapshots_1729.pickle').open('rb'));pre=json.loads((source/'training_1729.json').read_text())['preprocessing']
selected={}
for sample in samples:
 s=sample.snapshot;key=(len(s.ready),len(s.processed))
 if key not in selected and len(s.ready)>0:selected[key]=s
shape_rows=[s for s in selected.values() if len(s.processed)>0][:3]
if len(shape_rows)<2:shape_rows=list(selected.values())[:3]
if len({(len(s.ready),len(s.processed)) for s in shape_rows})<2:raise AssertionError('fixture lacks variable ready/history cardinalities')
items=[tensors(s,pre,kind='deepsets') for s in shape_rows];model=Scorer('deepsets').float().eval()
with torch.no_grad():ref=[model(x) for x in items];batch=model.forward_batch(items)
rawdiff=max(float((x-y).abs().max()) for x,y in zip(ref,batch))
a=Scorer('deepsets').float();a.load_state_dict(model.state_dict());a.train();b=Scorer('deepsets').float();b.load_state_dict(model.state_dict());b.train()
loss_a=torch.cat([a(x) for x in items]).square().mean();loss_a.backward();loss_b=torch.cat(b.forward_batch(items)).square().mean();loss_b.backward()
grad_diffs={}
for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
 if na!=nb:raise AssertionError('parameter order mismatch')
 grad_diffs[na]=float((pa.grad-pb.grad).abs().max())
gradmax=max(grad_diffs.values())
if rawdiff>1e-5 or gradmax>1e-5:raise AssertionError(f'parity exceeds tolerance raw={rawdiff} grad={gradmax}')
class Zero(torch.nn.Module):
 def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.tensor(0.))
 def forward(self,x):return x[:,0]*0+self.p
profile=pre['profile_identity']['sha256']
def msg(token,arr,seq):return Message(token,token,0,arr,seq,np.zeros(387,dtype=np.float32),profile_id=profile)
def episode(arrivals,services,horizon):
 ms=tuple(msg(i,t,i) for i,t in enumerate(arrivals));n=len(ms)
 return Episode(ms,tuple([-1]*n),tuple(range(n)),tuple(services),tuple([False]*n),horizon,{})
ep=episode([0.,0.],[2.,1.],4.);env=Environment(ep,'P');s=env.snapshot();token=s.ready[0].token;env.step(token);p=transition_path(s,token,env,pre,end=2.);val=float(integral(Zero(),p,'cpu').detach())
if abs(val+1.)>1e-6:raise AssertionError(f'continuous wait expected -1, got {val}')
ep0=episode([0.],[2.],4.);env0=Environment(ep0,'P');s0=env0.snapshot();token0=s0.ready[0].token;env0.step(token0);p0=transition_path(s0,token0,env0,pre,end=2.);val0=float(integral(Zero(),p0,'cpu').detach())
if p0.durations or abs(val0)>1e-7:raise AssertionError(f'no-wait path should be empty/zero: {p0.durations} {val0}')
epm=episode([0.,1.,2.25],[2.,1.,1.],4.);envm=Environment(epm,'P');sm=envm.snapshot();chosen=sm.ready[0].token;envm.step(chosen);pm=transition_path(sm,chosen,envm,pre,end=2.5)
expected=-.5*((2.-1.)+(2.25-2.)+(2.5-2.25));actual=float(integral(Zero(),pm,'cpu').detach())
if abs(actual-expected)>1e-6:raise AssertionError(f'mid-arrival expected {expected}, got {actual}; dt={pm.durations}')
if abs(sum(pm.durations)-1.5)>1e-9:raise AssertionError(f'selected job may be counted in waiting: {pm.durations}')
if len(pm.vectors)<2 or np.allclose(pm.vectors[0],pm.vectors[-1]):raise AssertionError('expected distinct pre/post-finish contexts showing history admission')
result={'raw_score_max_abs_diff':rawdiff,'gradient_max_abs_diff':gradmax,'tolerance_float32':1e-5,'loss_independent':float(loss_a.detach()),'loss_batch':float(loss_b.detach()),'gradient_diffs':grad_diffs,'queue_shapes':[{'ready':len(s.ready),'processed':len(s.processed)} for s in shape_rows],'reward':{'continuous_wait_elapsed2':val,'no_wait':val0,'mid_arrival_integral':actual,'mid_arrival_expected':expected,'mid_arrival_durations':pm.durations,'postfinish_context_diff_max':float(np.max(np.abs(np.asarray(pm.vectors[0])-np.asarray(pm.vectors[-1]))))}}
(run/'results.json').write_text(json.dumps(result,indent=2));files=['Scheduler/training/semantic_learning_models.py','Scheduler/training/semantic_learning_rl.py','Scheduler/training/preference_reward.py'];hashes={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files};(run/'source_hashes.json').write_text(json.dumps(hashes,indent=2));print(json.dumps(result,indent=2));print('SOURCE_HASHES',json.dumps(hashes,sort_keys=True))
