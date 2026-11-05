import importlib.util,sys,json
from pathlib import Path
import numpy as np
root=Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End');sys.path.insert(0,str(root));spec=importlib.util.spec_from_file_location('boot',root/'experiments/stream_scheduler/runtime_bootstrap.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.install(root)
from Evaluation.semantic_learning_env import Message,Episode,Environment
from Scheduler.training.preference_reward import transition_path,relabel_replay
pre=json.loads((root/'experiments/stream_scheduler/runs/v22-real-integration-20261031T000527Z/results/training_1729.json').read_text())['preprocessing'];profile=pre['profile_identity']['sha256'];x=np.zeros(387,np.float32)
ms=(Message(0,0,0,0.,0,x,profile_id=profile),Message(1,1,0,0.,1,x,profile_id=profile))
ep=Episode(ms,(-1,-1),(0,1),(1.,1.),(False,False),4.,{});env=Environment(ep,'P');s=env.snapshot();chosen=s.ready[0].token;nxt,_,done=env.step(chosen);path=transition_path(s,chosen,env,pre,end=1.)
assert path.vectors and len(path.durations)>0
row=(s,0,0.,nxt,done,path)
class R:
 def value(self,p):assert p is path;return -0.5*sum(p.durations)
rebuilt=relabel_replay([row],R())
assert len(rebuilt[0])==6 and rebuilt[0][5] is path and rebuilt[0][2]==-0.5*sum(path.durations)
(root/'experiments/stream_scheduler/runs/implementation-v22-checks-20261030T235934Z/batched_reward_checks_20261031/replay_path_check.json').write_text(json.dumps({'status':'PASS','tuple_length':len(rebuilt[0]),'path_preserved_by_identity':rebuilt[0][5] is path,'duration_sum':sum(path.durations),'relabelled_reward':rebuilt[0][2]},indent=2));print('PASS',len(rebuilt[0]),rebuilt[0][2])
