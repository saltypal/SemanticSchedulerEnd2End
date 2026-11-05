import json, os, math
from pathlib import Path
import torch
from Scheduler.scene_policy import ScenePolicy
root=Path(os.environ['PILOT_AUDIT_ROOT']); issues=[]; summary={}
status=json.loads((root/'status.json').read_text()); summary['status']=status
learners=json.loads((root/'learner_status.json').read_text())
trained=sorted(k for k,v in learners.items() if v.get('status')=='fixture_trained')
blocked=sorted(k for k,v in learners.items() if v.get('status')=='blocked')
summary['trained_arms']=trained; summary['blocked_arms']=blocked
if len(trained)!=8: issues.append(f'expected 8 trained arms, got {len(trained)}')
if len(blocked)!=3: issues.append(f'expected 3 blocked teacher arms, got {len(blocked)}')
if set(blocked)!={'rl_vlm_f','vlm_replay','visionprefer'}: issues.append('blocked arms differ from declared teacher arms')
metric_paths=sorted(p for p in (root/'metrics').glob('*_[0-2].json') if 'shared_probes' not in p.name)
policy_names=sorted({p.stem.rsplit('_',1)[0] for p in metric_paths})
summary['metric_files']=len(metric_paths); summary['policies']=policy_names
if len(metric_paths)!=42 or len(policy_names)!=14: issues.append(f'expected 42 metric records/14 policies, got {len(metric_paths)}/{len(policy_names)}')
checks={'accounting_records':0,'reward_records':0,'decision_logs':0,'decision_rows':0,'log_count_matches_selected':0}
for mp in metric_paths:
    m=json.loads(mp.read_text()); checks['accounting_records']+=1
    a=m.get('accounting',{}); offered=a.get('offered'); components=sum(a.get(k,0) for k in ('succeeded','failed','in_flight_at_horizon','queued_at_horizon','not_arrived'))
    if offered!=components: issues.append(f'{mp.name}: accounting {offered} != component sum {components}')
    j=m.get('J_time'); r=m.get('reward_sum'); w=m.get('offered_weight')
    if j is None:
        if r != 0 or w != 0: issues.append(f'{mp.name}: unknown truth has reward={r}, offered_weight={w}')
    elif not math.isclose(float(r),float(j),rel_tol=1e-5,abs_tol=1e-8): issues.append(f'{mp.name}: reward_sum {r} != J_time {j}')
    checks['reward_records']+=1
    ep=mp.stem.rsplit('_',1)[1]; name=mp.stem.rsplit('_',1)[0]; lp=root/'logs'/f'episode_{ep}'/f'{name}.jsonl'
    if not lp.exists(): issues.append(f'missing per-decision log {lp.relative_to(root)}'); continue
    rows=[json.loads(line) for line in lp.read_text().splitlines() if line.strip()]
    checks['decision_logs']+=1; checks['decision_rows']+=len(rows)
    if len(rows)!=a.get('selected'): issues.append(f'{lp.relative_to(root)}: {len(rows)} rows != selected {a.get("selected")}')
    else: checks['log_count_matches_selected']+=1
summary['checks']=checks
hist={}
for arm in trained:
    p=root/'models'/arm/'learning.json'; val=json.loads(p.read_text()) if p.exists() else None
    n=len(val) if isinstance(val,(list,dict)) else (1 if val else 0)
    hist[arm]={'present':p.exists(),'entries':n}
    if not p.exists() or n==0: issues.append(f'{arm}: empty or missing learning history')
summary['optimization_histories']=hist
portable_loaded=[]; checkpoints_loaded=[]
for arm in trained:
    artifact=json.loads((root/'models'/arm/'portable.json').read_text()); ScenePolicy(artifact); portable_loaded.append(arm)
    state=torch.load(root/'models'/arm/'resume.pt',map_location='cpu',weights_only=False)
    def cpu_only(x):
        if isinstance(x,torch.Tensor): return x.device.type=='cpu'
        if isinstance(x,dict): return all(cpu_only(k) and cpu_only(v) for k,v in x.items())
        if isinstance(x,(list,tuple)): return all(cpu_only(v) for v in x)
        return True
    if not cpu_only(state): issues.append(f'{arm}: checkpoint contains non-CPU tensors')
    checkpoints_loaded.append(arm)
summary['portable_reloaded']=portable_loaded; summary['checkpoints_reloaded_cpu']=checkpoints_loaded
summary['issues']=issues
(Path(os.environ['PILOT_AUDIT_CHECK'])/'pilot_artifact_audit.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if issues: raise SystemExit(1)
