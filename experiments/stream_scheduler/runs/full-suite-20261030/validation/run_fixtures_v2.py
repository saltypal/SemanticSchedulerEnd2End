import hashlib,json,math,sys,time,traceback
from pathlib import Path
import numpy as np
R=Path('/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End'); O=R/'experiments/stream_scheduler/runs/full-suite-20261030/validation'
sys.path.insert(0,str(R/'experiments/stream_scheduler')); import runtime_bootstrap; runtime_bootstrap.install(R)
from Evaluation.semantic_learning_env import Message,Episode,Environment,Snapshot,ell
from Scheduler.semantic_learning_policy import fit_normalizer,arrays,NumpyRanker
from Scheduler.training import semantic_learning_models as lm
import torch
SRC=['Evaluation/semantic_learning_env.py','Scheduler/semantic_learning_policy.py','Scheduler/training/semantic_learning_models.py','experiments/stream_scheduler/runtime_bootstrap.py']
def sh():return {p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in SRC}
def msg(i,a=0,z=0,features=None,source=0,session=0,sequence=None):
 if features is None:
  features=np.random.default_rng(9000+i).uniform(-1,1,387).astype(np.float32);features[374:378]=0;features[374+z]=1;features[378:380]=0;features[380:387]=0;features[384]=1;features[386]=1
 return Message(i,source,session,a,i if sequence is None else sequence,features)
def ep(ms,z=None,svc=None,fail=None,h=20,facts=None):
 n=len(ms);return Episode(tuple(ms),tuple(z or [0]*n),tuple(facts or [f'f{i}' for i in range(n)]),tuple(svc or [1.]*n),tuple(fail or [False]*n),float(h),{})
def cost():
 ms=[msg(0,0,0),msg(1,.5,1),msg(2,.75,2)];e=Environment(ep(ms,[0,1,2],[1.5,2.,1.],[False]*3,2))
 s,r,_=e.step(e.snapshot().ready[0].token); rs=[r];assert 1 in [m.token for m in s.ready]
 s,r,d=e.step(s.ready[0].token);rs.append(r);assert d and e.now==2
 c=e.cost(2);want=ell(0,2)+ell(1,4)+ell(1.25,8)
 assert math.isclose(c,-100*sum(rs),abs_tol=1e-8) and math.isclose(c,want,abs_tol=1e-8),(c,rs,want)
 assert e.selected=={0:0.,1:1.5} and 2 not in e.selected and e.events[-1]['observed_until']==2.
 return {'terminal_cost':c,'reward_telescoping':-100*sum(rs),'selected':e.selected,'unselected_wait_cost':float(ell(1.25,8))}
def target():
 a,b=msg(0,0,0),msg(1,0,3);sa=Snapshot((a,),(),((),),(),1.5,0);sb=Snapshot((b,),(),((),),(),3.,0)
 x=Environment(ep([a],[0]),'D').target(sa)[0];y=Environment(ep([b],[3]),'D').target(sb)[0]
 assert x==.75 and y==.0625,(x,y)
 return {'A':float(x),'B':float(y)}
def lifecycle():
 a,b=msg(0),msg(1,5);e=Environment(ep([a,b],[0,0],[.5,1.],[True,False],10))
 try:e.step(999);raise AssertionError('illegal accepted')
 except ValueError:pass
 s,_,_=e.step(0);assert s.attempts==1 and s.processed[-1][0].token==0 and e.events[-1]['failed'] is True
 try:e.step(0);raise AssertionError('retry accepted')
 except ValueError:pass
 assert e.now==5 and [m.token for m in e.snapshot().ready]==[1]
 assert not set(Snapshot.__dataclass_fields__).intersection({'reference_tiers','fact_ids','service','failures','future_arrays','labels'})
 try:Message(9,0,0,0,0,np.zeros(386,dtype=np.float32));raise AssertionError('386 accepted')
 except ValueError:pass
 return {'illegal_choice_rejected':True,'retry_rejected':True,'failed_attempt_in_history':True,'future_labels_not_snapshot_fields':True,'features':387}
def fixture(seed=17):
 rng=np.random.default_rng(seed);ms=[]
 for i in range(7):
  x=rng.normal(0,.3,387).astype(np.float32);x[374:378]=0;x[374+i%4]=1;x[378:380]=rng.uniform(-1,1,2);x[380:387]=0;x[384]=float(i%2==0);x[385]=float(i%3!=0);x[386]=float(i%3==0);ms.append(msg(i,float(i//4)*.25,i%4,x,i%3))
 e=ep(ms,[0,1,2,3,1,2,3]);s=Snapshot(tuple(ms[:4]),tuple((m,m.arrival+.4) for m in ms[4:6]),(tuple(ms[:3]),tuple(ms[1:4]),tuple(ms[2:5]),tuple(ms[3:6])),(tuple(ms[:3]),tuple(ms[2:5])),1.75,2)
 return e,s,fit_normalizer([e])
def parity():
 e,s,pre=fixture(); ar=arrays(s,pre);assert ar['ready'].shape==(4,389) and ar['summary'].shape==(4,1172)
 rows=[]
 for k in ('linear','mlp','classifier','deepsets','attention','temporal'):
  torch.manual_seed(121+len(rows));m=lm.Scorer(k);a=lm.export(m,pre,'fixture',{});p=lm.parity(m,a,[s]);n=NumpyRanker(a);base=n.scores(s)
  ix=[2,0,3,1];sp=Snapshot(tuple(s.ready[i] for i in ix),s.processed,tuple(s.ue_windows[i] for i in ix),s.active_windows,s.now,s.attempts);vals=n.scores(sp);back=np.empty_like(vals)
  for j,i in enumerate(ix):back[i]=vals[j]
  assert np.allclose(base,back,atol=2e-5,rtol=2e-5),(k,base,back)
  def cl(m):return Message(m.token+1000,m.source+50,m.session+9,m.arrival,m.sequence+100,m.features,m.header_bytes)
  sr=Snapshot(tuple(cl(m) for m in s.ready),tuple((cl(m),t) for m,t in s.processed),tuple(tuple(cl(m) for m in w) for w in s.ue_windows),tuple(tuple(cl(m) for m in w) for w in s.active_windows),s.now,s.attempts)
  assert np.allclose(base,n.scores(sr),atol=2e-5,rtol=2e-5),k
  rows.append({'kind':k,'max_abs':p['maximum_absolute_error'],'permute_ok':True,'relabel_ok':True})
 torch.manual_seed(404);a=lm.export(lm.Scorer('temporal'),pre,'fixture',{});n=NumpyRanker(a);rev=Snapshot(s.ready,s.processed,tuple(tuple(reversed(w)) for w in s.ue_windows),s.active_windows,s.now,s.attempts);delta=float(np.max(np.abs(n.scores(s)-n.scores(rev))));assert delta>1e-7,delta
 return {'models':rows,'temporal_order_delta':delta,'array_shapes':{k:list(v.shape) for k,v in ar.items()}}
def gradients():
 q=torch.tensor([.2,-.1],requires_grad=True);sample=lm.Sample(None,np.array([1.,0.]),np.array([0,1]),'p',pairs=((0,1,1.),));loss=lm.supervised_loss(q,sample,'ranknet');loss.backward();assert q.grad[0]<0 and q.grad[1]>0,q.grad
 eq=torch.tensor([0.,0.],requires_grad=True);t=lm.supervised_loss(eq,lm.Sample(None,np.array([0.,0.]),np.array([0,0]),'t',pairs=((0,1,.5),)),'ranknet');t.backward();assert torch.allclose(eq.grad,torch.zeros(2),atol=1e-7)
 e,s,pre=fixture();m=lm.Scorer('classifier')
 for p in m.parameters():torch.nn.init.zeros_(p)
 cw=np.array([1.,2.,4.,8.]);a=lm.export(m,pre,'classifier',{},class_weights=cw);n=NumpyRanker(a);got=n.scores(s);pr=(1/cw)/(1/cw).sum();z=np.array([s.now-m.arrival for m in s.ready])[:,None];d=np.array([2.,4.,8.,16.])[None,:]
 c=lambda w:w/d+np.maximum(w/d-1,0);want=np.sum(pr[None,:]*(c(z+1)-c(z)),axis=1);assert np.allclose(got,want,atol=1e-12),(got,want)
 return {'strict_grad':q.grad.tolist(),'tie_grad':eq.grad.tolist(),'inverse_cw_probabilities':pr.tolist(),'numpy_correction_match':True}
def lambdamart():
 import lightgbm as lgb
 cap={}
 class B:
  def dump_model(self):return {'tree_info':[{'tree_structure':{'leaf_value':0.}}]}
  def save_model(self,p):Path(p).write_text('test fake tree\n')
 class F:
  def __init__(self,**kw):pass
  def fit(self,x,y,group):cap.update(y=np.asarray(y).tolist(),g=list(group))
  @property
  def booster_(self):return B()
  def predict(self,x):return np.zeros(len(x))
 old=lgb.LGBMRanker;lgb.LGBMRanker=F
 try:
  e,s,pre=fixture();y=np.array([0.,.1,.5,.9,1.,.2,.7]);lm.fit_lambdamart([lm.Sample(s,y,np.zeros(7,dtype=int),'q')],pre,O/'lambdamart_fake_export')
 finally:lgb.LGBMRanker=old
 want=np.floor(255*y+.5).astype(int).tolist();assert cap['y']==want and cap['g']==[7],(cap,want)
 art=json.loads((O/'lambdamart_fake_export/model.json').read_text());assert NumpyRanker(art).scores(s).shape==(4,)
 return {'float':y.tolist(),'grade':want,'groups':cap['g'],'real_lgb_training':False}
tests=[('D_cost_midservice_telescoping_unselected_horizon',cost),('target_A_B',target),('illegal_retry_failure_history_future_labels_387',lifecycle),('six_torch_numpy_parity_candidate_permutation_identity_relabel_temporal_order',parity),('strict_pair_gradient_numpy_class_weight_correction',gradients),('lambdamart_float_grades',lambdamart)]
out={'scope':'CPU fixtures only; no real data, training, inference, GPU or cloud','python':sys.version,'checks':[]}
for name,fn in tests:
 before=sh();t=time.perf_counter()
 try:v=fn();status='PASS';err=None
 except BaseException:status='FAIL';v=None;err=traceback.format_exc()
 after=sh();row={'name':name,'status':status,'seconds':time.perf_counter()-t,'source_sha256_before':before,'source_sha256_after':after,'source_changed_during_check':before!=after,'result':v,'error':err};out['checks'].append(row)
 print(json.dumps({'name':name,'status':status,'seconds':row['seconds'],'source_changed':row['source_changed_during_check'],'result':v,'error':err},allow_nan=False),flush=True)
 (O/'fixture_validation_v2.json').write_text(json.dumps(out,indent=2,allow_nan=False))
 if status=='FAIL':sys.exit(1)
out['finish_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());(O/'fixture_validation_v2.json').write_text(json.dumps(out,indent=2,allow_nan=False));print('ALL_PASS',len(tests),flush=True)
