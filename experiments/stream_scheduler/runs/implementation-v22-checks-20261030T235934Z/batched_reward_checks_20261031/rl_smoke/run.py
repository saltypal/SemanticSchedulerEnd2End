import importlib.util,json,sys,hashlib
from pathlib import Path
import numpy as np, torch
root=Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End');spec=importlib.util.spec_from_file_location('boot',root/'experiments/stream_scheduler/runtime_bootstrap.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.install(root)
from Evaluation.semantic_learning_env import Message,Episode,Environment
from Scheduler.training.semantic_learning_rl import train_dqn
from Scheduler.semantic_learning_policy import fit_normalizer,NumpyRanker
out=root/'experiments/stream_scheduler/runs/implementation-v22-checks-20261030T235934Z/batched_reward_checks_20261031/rl_smoke'
x=np.zeros(387,dtype=np.float32);x[0]=1;x[186:279]=1;x[279:372]=1;x[372]=1;x[374]=1;x[380:384]=1
msgs=tuple(Message(i,i%4,1,float(i//4),i,x.copy()) for i in range(16));ep=Episode(msgs,tuple(i%4 for i in range(16)),tuple(str(i) for i in range(16)),(1.,)*16,(False,)*16,30.,{})
pre=fit_normalizer([ep]);factory=lambda i:Environment(ep)
def validate(a):
 env=factory(0);pol=NumpyRanker(a)
 while not env.done:env.step(pol.select(env.snapshot()))
 return -env.metrics()['J_D'],{}
source=['Scheduler/training/semantic_learning_models.py','Scheduler/training/semantic_learning_rl.py','Scheduler/training/preference_reward.py'];hashes={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in source};results={}
for arm in ('masked_double_dqn','td_per'):
 a,h=train_dqn(factory,pre,out/arm,validate,arm=arm,steps=504,seed=1729)
 state=torch.load(out/arm/'resume.pt',map_location='cpu',weights_only=False);replay=state['replay'];assert replay and all(len(row)==6 for row in replay) and all(row[5] is None for row in replay)
 results[arm]={'status':'PASS','steps':state['step'],'updates':state['updates'],'replay_rows':len(replay),'all_rows_six_tuple':True,'optional_path_is_none_without_reward_model':True,'validation_history':h}
(out/'source_hashes.json').write_text(json.dumps(hashes,indent=2));(out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2));print('HASHES',json.dumps(hashes,sort_keys=True))
