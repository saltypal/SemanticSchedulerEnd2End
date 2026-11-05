from pathlib import Path
import hashlib, importlib.util, json, sys
from types import SimpleNamespace
import numpy as np
root=Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End');sys.path.insert(0,str(root))
spec=importlib.util.spec_from_file_location('boot',root/'experiments/stream_scheduler/runtime_bootstrap.py');boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.install(root)
from experiments.stream_scheduler.semantic_suite_data import load_pool
from Evaluation.semantic_learning_env import Environment,Message,Episode,Snapshot,Rule,make_episode
from experiments.stream_scheduler.train_real_suite import evaluate_probes
out=root/'experiments/stream_scheduler/runs/v22-dynamic-ndcg-validation-20261031T001846Z'
archive=root/'experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/Mini_dataset_6000.tar'
records,features,embeddings,profile,audit=load_pool(archive,out/'data')
source_hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['Evaluation/semantic_learning_env.py','experiments/stream_scheduler/train_real_suite.py','experiments/stream_scheduler/semantic_suite_data.py']}
result={'source_hashes':source_hashes,'dataset_cache':{'records':len(records),'cached_embeddings':len(embeddings),'profile_sha256':profile['sha256']},'population_checks':[]}
for ues,count,seed in ((50,256,50150),(60,384,60160),(100,512,100100)):
 ep=make_episode(records,features,embeddings,None,count=count,ues=ues,load=1.2,seed=seed,repeat='none',failure_rate=.17,variable_service=True,churn=True,burst=True)
 intervals=ep.metadata['ue_active_intervals'];assert ep.metadata['ue_count']==ues and len(ep.messages)==count and len(ep.metadata['profiles'])==ues
 invalid=[]
 for m in ep.messages:
  valid=any(sess==m.session and start<=m.arrival<end for start,end,sess in intervals[m.source])
  if not valid:invalid.append({'token':m.token,'source':m.source,'session':m.session,'arrival':m.arrival,'intervals':intervals[m.source]})
 assert not invalid,invalid[:3]
 reset_sources=[src for src,its in intervals.items() if len(its)==2 and its[0][2]!=its[1][2]]
 reset_checks=0
 for src in reset_sources:
  session=intervals[src][1][2];group=sorted((m for m in ep.messages if m.source==src and m.session==session),key=lambda m:(m.arrival,m.sequence))
  if group:
   first=group[0]
   if embeddings.get(next((r['image_sha256'] for r in records if r['image_id']==ep.metadata['assets'][first.token]),'')) is not None:
    assert first.features[386]==1.,(ues,src,session,first.features[386]);assert first.features[385]==0.,(ues,src,session,first.features[385]);reset_checks+=1
 assert reset_checks>0, f'no observable reset cold-start case for {ues}'
 # Verify cold-start and novelty masks per source/session against ordered causal observations.
 lookup={r['image_id']:r for r in records};by_session={}
 for m in ep.messages:
  key=(m.source,m.session);prior=by_session.setdefault(key,[]);record=lookup[ep.metadata['assets'][m.token]];has_vector=record['image_sha256'] in embeddings
  if has_vector:
   expected_cold=(len(prior)==0)
   assert float(m.features[386])==float(expected_cold),(ues,key,m.token,m.features[386],expected_cold)
   comparable=any(v is not None for v in prior)
   assert bool(m.features[385])==bool(comparable and not expected_cold),(ues,key,m.token,m.features[385],comparable,expected_cold)
  prior.append(embeddings.get(record['image_sha256']))
 # Environment service at first message from B crosses source A's leave; A's received messages must stay queued.
 counts={src:sum(m.source==src and any(sess==m.session and start<=m.arrival<end for start,end,sess in intervals[src]) for m in ep.messages) for src in intervals}
 target=next((src for src,n in counts.items() if n>=2 and any(m.source==src and m.arrival==0 for m in ep.messages)),None)
 initial_sources={m.source for m in ep.messages if m.arrival==0};other=next((s for s in initial_sources if s!=target),None)
 assert target is not None and other is not None,(ues,target,other)
 target_leave=max(end for start,end,sess in intervals[target]);first_other=next(m for m in ep.messages if m.source==other and m.arrival==0)
 service=list(ep.service);service[first_other.token]=target_leave+2.
 delayed=Episode(ep.messages,ep.reference_tiers,ep.fact_ids,tuple(service),ep.failures,max(ep.horizon,target_leave+5),ep.metadata)
 env=Environment(delayed,'P');before=env.snapshot();env.step(first_other.token);after=env.snapshot()
 queued_target=[m for m in after.ready if m.source==target and m.token!=first_other.token]
 assert env.now>target_leave and queued_target,(ues,target,target_leave,env.now,len(queued_target))
 # Full run: every offer is selected/attempted, remains queued, or has not arrived at horizon.
 def run(policy):
  e=Environment(ep,'P');pol=Rule(policy,seed=seed+7)
  while not e.done:e.step(pol.select(e.snapshot()))
  met=e.metrics();assert met['selected']+met['queued_at_horizon']+met['not_arrived']==count,(ues,met)
  assert len(e.events)==met['selected']
  by_token={z['token']:z for z in e.events}
  for tok,ev in by_token.items():assert ev['failed']==(bool(ep.failures[tok]) if ev['completed'] else None),(ues,tok,ev,ep.failures[tok])
  return e,by_token,met
 fifo,fifo_events,fifo_metrics=run('fifo');rand,rand_events,rand_metrics=run('random')
 # Recreate identical episode stream. Service/failure realizations are attached to message identity, not policy order.
 ep2=make_episode(records,features,embeddings,None,count=count,ues=ues,load=1.2,seed=seed,repeat='none',failure_rate=.17,variable_service=True,churn=True,burst=True)
 assert ep.metadata==ep2.metadata and ep.service==ep2.service and ep.failures==ep2.failures
 assert all(a.token==b.token and a.source==b.source and a.session==b.session and a.arrival==b.arrival and np.array_equal(a.features,b.features) for a,b in zip(ep.messages,ep2.messages))
 result['population_checks'].append({'ues':ues,'offered_messages':count,'churn_and_burst':True,'availability_membership':'pass; session and half-open interval match','reset_cold_start_cases':reset_checks,'queue_survives_departure':{'source':target,'leave':target_leave,'checked_at':env.now,'retained_ready':len(queued_target)},'fifo_accounting':fifo_metrics,'random_accounting':rand_metrics,'shared_failure_map':True,'deterministic_replay':True})
# Hand-computed nDCG probes: ideal [3,2,1] -> 1; reverse must score below1.
features0=np.zeros(387,dtype=np.float32)
msgs=tuple(Message(i,i,0,0.,i,features0) for i in range(3));ep_probe=Episode(msgs,(0,1,2),(0,1,2),(1.,1.,1.),(False,)*3,8.,{});snap=Environment(ep_probe,'P').snapshot()
class Probe:
 def __init__(self,target,reverse=False):self.target=np.asarray(target,dtype=float);self.reverse=reverse
 def select(self,s):
  if not s.ready:return None
  return (min if self.reverse else max)(s.ready,key=lambda m:self.target[m.token]).token
ideal=evaluate_probes(Probe([3,2,1]),[SimpleNamespace(snapshot=snap,target=np.array([3,2,1]))])
reverse=evaluate_probes(Probe([3,2,1],True),[SimpleNamespace(snapshot=snap,target=np.array([3,2,1]))])
assert all(abs(ideal['ndcg'][str(k)]['mean']-1.)<1e-12 for k in (1,3,5)),ideal
assert reverse['ndcg']['3']['mean']<1.,reverse
empty=Snapshot((),(),(),(),0.,0.,frozenset(),());one=Environment(Episode((Message(0,0,0,0.,0,features0),),(0,),(0,),(1.,),(False,),8.,{}),'P').snapshot()
class First:
 def select(self,s):return s.ready[0].token if s.ready else None
counts=evaluate_probes(First(),[SimpleNamespace(snapshot=empty,target=np.array([])),SimpleNamespace(snapshot=one,target=np.array([0.])),SimpleNamespace(snapshot=snap,target=np.array([0.,0.,0.]))])
assert counts['singletons']==2 and counts['candidate_counts']==[0,1,3] and counts['zero_ideal_metric_cells']==3,counts
result['ndcg']={'ideal':ideal,'reversed':reverse,'empty_singleton_zero_ideal':counts,'empty_and_singleton_reporting_separately':False,'observation':'evaluate_probes increments one combined singletons counter for n=0 and n=1; candidate_counts preserves individual sizes.'}
(out/'validation_results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2,allow_nan=False))
