import importlib.util,json,pickle,sys
from pathlib import Path
root=Path(__file__).resolve().parents[5]
spec=importlib.util.spec_from_file_location('boot',root/'experiments/stream_scheduler/runtime_bootstrap.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.install(root)
import torch
from Evaluation.semantic_learning_env import Message,Episode,Environment
from Scheduler.training.semantic_learning_models import Scorer,export,parity
from Scheduler.training.semantic_learning_rl import train_dqn,train_ppo
import numpy as np
out=Path(__file__).parent/'rl_smoke_results';out.mkdir(exist_ok=True)
x=np.zeros(387,dtype=np.float32);x[0]=1;x[186:279]=1;x[279:372]=1;x[372]=1;x[374]=1;x[380:384]=1
msgs=tuple(Message(i,i%4,1,float(i//4),i,x.copy()) for i in range(16))
ep=Episode(msgs,tuple(i%4 for i in range(16)),tuple(str(i) for i in range(16)),(1.,)*16,(False,)*16,30.,{})
from Scheduler.semantic_learning_policy import fit_normalizer,NumpyRanker
pre=fit_normalizer([ep]);factory=lambda i:Environment(ep)
def validate(a):
 env=factory(0);pol=NumpyRanker(a)
 while not env.done:env.step(pol.select(env.snapshot()))
 return -env.metrics()['J_D'],{}
results={}
for arm in ('masked_double_dqn','td_per'):
 try:
  a,h=train_dqn(factory,pre,out/arm,validate,arm=arm,steps=504,seed=1729)
  results[arm]={'status':'PASS','history':h}
 except Exception as error:
  import traceback;(out/(arm+'_error.txt')).write_text(traceback.format_exc());results[arm]={'status':'FAIL','error':str(error)}
try:
 a,h=train_ppo(factory,pre,out/'ppo',validate,steps=64,seed=1729);results['masked_ppo']={'status':'PASS','history':h}
except Exception as error:
 import traceback;(out/'ppo_error.txt').write_text(traceback.format_exc());results['masked_ppo']={'status':'FAIL','error':str(error)}
(out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2),flush=True)
