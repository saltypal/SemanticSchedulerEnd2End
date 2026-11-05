import json, os, math
from pathlib import Path
import numpy as np
import torch
import lightgbm as lgb
from Scheduler.scene_policy import ScenePolicy, SceneCandidate, SceneContext
from Scheduler.training.scene_learners import tensor
root=Path(os.environ['PILOT_AUDIT_ROOT']); check=Path(os.environ['PILOT_AUDIT_CHECK']); issues=[]
status=json.loads((root/'status.json').read_text()); learners=json.loads((root/'learner_status.json').read_text())
trained=sorted(k for k,v in learners.items() if v.get('status')=='fixture_trained')
blocked=sorted(k for k,v in learners.items() if v.get('status')=='blocked')
metrics=sorted(p for p in (root/'metrics').glob('*_[0-2].json') if 'shared_' not in p.name)
policies=sorted({p.stem.rsplit('_',1)[0] for p in metrics})
report={'run_status':status,'trained_arms':trained,'blocked_arms':blocked,'policies':policies,'metric_records':len(metrics),'issues':[]}
if status.get('status')!='complete_with_blocked_teacher_arms' or status.get('episodes')!=3: issues.append('pilot status/episode mismatch')
if len(trained)!=8: issues.append(f'expected 8 trained arms, got {len(trained)}')
if len(blocked)!=3 or set(blocked)!={'rl_vlm_f','vlm_replay','visionprefer'}: issues.append('expected 3 designated teacher arms blocked')
if len(policies)!=14 or len(metrics)!=42: issues.append('expected 14 policies and 42 policy-episode metrics')
counts={'accounting':0,'rewards':0,'decision_files':0,'decision_rows':0,'logs_match_selected':0}
for mp in metrics:
    m=json.loads(mp.read_text()); a=m.get('accounting',{}); components=sum(a.get(k,0) for k in ('succeeded','failed','in_flight_at_horizon','queued_at_horizon','not_arrived'))
    if a.get('offered')!=components: issues.append(f'{mp.name}: accounting mismatch')
    counts['accounting']+=1; j=m.get('J_time'); r=m.get('reward_sum'); w=m.get('offered_weight')
    if j is None:
        if r!=0 or w!=0: issues.append(f'{mp.name}: unknown truth is assigned reward/weight')
    elif not math.isclose(float(r),float(j),rel_tol=1e-5,abs_tol=1e-8): issues.append(f'{mp.name}: reward_sum != J_time')
    counts['rewards']+=1; ep=mp.stem.rsplit('_',1)[1]; name=mp.stem.rsplit('_',1)[0]
    lp=root/'logs'/f'episode_{ep}'/f'{name}.jsonl'
    if not lp.exists(): issues.append(f'missing log {lp.relative_to(root)}'); continue
    rows=[json.loads(line) for line in lp.read_text().splitlines() if line.strip()]
    counts['decision_files']+=1; counts['decision_rows']+=len(rows)
    if len(rows)!=a.get('selected'): issues.append(f'{lp.relative_to(root)} row count != selected')
    else: counts['logs_match_selected']+=1
report['stream_checks']=counts
history={}
for arm in trained:
    p=root/'models'/arm/'learning.json'; value=json.loads(p.read_text()) if p.exists() else None
    n=len(value) if isinstance(value,(list,dict)) else int(bool(value))
    history[arm]={'present':p.exists(),'entries':n}
    if arm=='lambdamart':
        if not p.exists() or n!=0: issues.append('LambdaMART history format changed unexpectedly; audit tree training separately')
    elif not p.exists() or n==0: issues.append(f'{arm}: missing nonempty optimization history')
report['optimization_histories']=history
portable=[]; checkpoints=[]
for arm in trained:
    artifact=json.loads((root/'models'/arm/'portable.json').read_text()); ScenePolicy(artifact); portable.append(arm)
    state=torch.load(root/'models'/arm/'resume.pt',map_location='cpu',weights_only=False)
    def cpu_only(x):
        if isinstance(x,torch.Tensor): return x.device.type=='cpu'
        if isinstance(x,dict): return all(cpu_only(k) and cpu_only(v) for k,v in x.items())
        if isinstance(x,(tuple,list)): return all(cpu_only(v) for v in x)
        return True
    if not cpu_only(state): issues.append(f'{arm}: checkpoint includes non-CPU tensor')
    checkpoints.append(arm)
report['portable_reloaded']=portable; report['checkpoints_reloaded_cpu']=checkpoints
# LambdaMART uses a fitted booster history, not an epoch-loss series.
arm='lambdamart'; state=torch.load(root/'models'/arm/'resume.pt',map_location='cpu',weights_only=False)
text=state.get('booster_text') if isinstance(state,dict) else None
if not isinstance(text,str) or not text.strip(): issues.append('LambdaMART checkpoint missing booster_text')
booster=lgb.Booster(model_str=text) if text else None
portable_artifact=json.loads((root/'models'/arm/'portable.json').read_text()); policy=ScenePolicy(portable_artifact)
dump=booster.dump_model() if booster else {}; tree_info=dump.get('tree_info',[])
leaf_values=[]
def visit(node):
    if 'leaf_value' in node: leaf_values.append(float(node['leaf_value']))
    else: visit(node['left_child']);visit(node['right_child'])
for tree in portable_artifact.get('model',{}).get('trees',[]): visit(tree)
if not tree_info or not portable_artifact.get('model',{}).get('trees'): issues.append('LambdaMART has zero fitted/exported trees')
if not leaf_values or not any(abs(v)>1e-12 for v in leaf_values): issues.append('LambdaMART exported leaf values are trivial/empty')
examples=json.loads((root/'data'/'training_examples.json').read_text());pre=state.get('preprocessing',{})
comparisons=0; max_abs=0.0
for item in examples:
    d=item['context']
    def candidates(rows): return tuple(SceneCandidate(**row) for row in rows)
    ctx=SceneContext(ready_candidates=candidates(d['ready_candidates']),successful_header_history=candidates(d['successful_header_history']),attempts_completed=d['attempts_completed'],admitted_count=d['admitted_count'],now=d['now'],horizon=d['horizon'],registry_hash=d['registry_hash'])
    if not ctx.ready_candidates: continue
    x=tensor(ctx,pre).numpy(); expected=np.asarray(booster.predict(x)); actual=policy.scores(ctx)
    if expected.shape!=actual.shape: issues.append('LambdaMART portable prediction shape mismatch');break
    delta=float(np.max(np.abs(expected-actual))) if expected.size else 0.0; max_abs=max(max_abs,delta); comparisons+=int(expected.size)
    if not np.allclose(expected,actual,rtol=1e-6,atol=1e-7): issues.append('LambdaMART portable/booster prediction parity failed');break
if comparisons==0: issues.append('no saved training-example candidate predictions compared')
tree={'status':'verified' if not any('LambdaMART' in x for x in issues) else 'failed','run_id':root.name,'history_note':'No per-epoch loss is applicable; fit_lambdamart returns an empty history and stores the fitted Booster text in resume.pt.','checkpoint':'models/lambdamart/resume.pt','booster_text_present':bool(text and text.strip()),'num_trees':len(tree_info),'portable_tree_count':len(portable_artifact.get('model',{}).get('trees',[])),'leaf_count':len(leaf_values),'nontrivial_leaf_count':sum(abs(v)>1e-12 for v in leaf_values),'prediction_rows_compared':comparisons,'max_abs_prediction_difference':max_abs,'saved_examples':'data/training_examples.json','source_sha256':json.loads((root/'manifest.json').read_text()).get('source_sha256',{}).get('Scheduler/training/scene_learners.py')}
(check/'tree_fitting.json').write_text(json.dumps(tree,indent=2))
report['tree_fitting']=tree; report['issues']=issues
(check/'pilot_artifact_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if issues: raise SystemExit(1)
