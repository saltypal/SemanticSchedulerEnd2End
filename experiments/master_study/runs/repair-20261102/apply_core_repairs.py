from pathlib import Path
import shutil,hashlib,json
root=Path.cwd();run=root/"experiments/master_study/runs/repair-20261102";run.mkdir(parents=True,exist_ok=True)
files=["Evaluation/semantic_learning_env.py","Scheduler/training/semantic_learning_models.py","Scheduler/training/semantic_learning_rl.py","experiments/stream_scheduler/train_real_suite.py","experiments/stream_scheduler/train_feedback_rl.py","experiments/stream_scheduler/teacher_pipeline.py","experiments/master_study/master_suite.py","experiments/stream_scheduler/semantic_suite_data.py","Scheduler/tests/test_scene_learning_system.py"]
for name in files:
 p=root/name;dest=run/"before"/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():shutil.copy2(p,dest)
(run/"PLAN.md").write_text("""# End-to-end repair
Preserve v9 as development evidence. No new training or Kaggle submission.
Fix full-trajectory collection, ranking losses, critic context, failure/evidence semantics, probability retention, teacher/replay qualification, shared workload and incremental reporting.
Add semantic reference and information-coverage gates; no invented urgency or unsupported dataset labels.
Validate numerical counterexamples, current-data pilots and broad regressions. Independent labels and qualified VLM collection remain external evidence requirements.
""")
def edit(name,old,new):
 p=root/name;t=p.read_text()
 if old not in t:raise RuntimeError("anchor missing "+name+" "+old[:70])
 p.write_text(t.replace(old,new))
edit("Evaluation/semantic_learning_env.py","self.selected[token]=start;self.covered.add(self.episode.fact_ids[i])","self.selected[token]=start")
edit("Evaluation/semantic_learning_env.py","self.attempts+=1;self.processed.append((m,finish));self.inflight=None","self.attempts+=1;self.inflight=None\n            if not self.episode.failures[i]:\n                self.processed.append((m,finish));self.covered.add(self.episode.fact_ids[i])")
# E counts only successfully completed evidence; D intentionally measures wait to dispatch.
edit("Evaluation/semantic_learning_env.py","if m.token in self.selected:first[fact]=min(first.get(fact,float('inf')),self.selected[m.token])","if m.token in self.selected:\n                finish=self.selected[m.token]+ep.service[self.index[m.token]]\n                if not ep.failures[self.index[m.token]] and finish<=t:\n                    first[fact]=min(first.get(fact,float('inf')),finish)")
edit("Scheduler/tests/test_scene_learning_system.py","assert first.processed[0][0].token == 0  # Attempt history, not certified evidence.","assert not first.processed  # Failed attempts are not delivered evidence.")
# Four scores are ordered explicitly; do not infer order from integer class_id.
edit("experiments/stream_scheduler/semantic_suite_data.py",
"x=np.concatenate([scene,[float(available and spi['level']==z) for z in LEVELS],[0.,0.],[float(available)]*4,[0.,0.,0.]])",
"""scores=np.array([float(available and spi['level']==z) for z in LEVELS])
        if available and 'scores' in spi:
            raw=np.asarray(spi['scores'],dtype=float)
            if raw.shape!=(4,) or not np.isfinite(raw).all() or np.any(raw<0) or abs(raw.sum()-1)>1e-4:
                raise ValueError('invalid SPI probability vector')
            # The registered mini student encodes Low, Medium, High, Critical.
            scores=raw[::-1]
        x=np.concatenate([scene,scores,[0.,0.],[float(available)]*4,[0.,0.,0.]])""")
edit("experiments/stream_scheduler/semantic_suite_data.py",
"'feature_order':'scene374+SPI4+R+N+availability6+cold1'",
"'feature_order':'scene374+SPI_prob4_C_H_M_L+R+N+availability6+cold1'")
edit("Scheduler/training/semantic_learning_rl.py",
"self.value=nn.Sequential(nn.Linear(389,64),nn.ReLU(),nn.Linear(64,1))",
"self.value=nn.Sequential(nn.Linear(1172,64),nn.ReLU(),nn.Linear(64,1))")
edit("Scheduler/training/semantic_learning_rl.py",
"self.value(a['ready'].mean(0))","self.value(a['summary'].mean(0))")
edit("Scheduler/training/semantic_learning_rl.py",
"torch.stack([a['ready'].mean(0) for a in items])","torch.stack([a['summary'].mean(0) for a in items])")
p=root/"Scheduler/training/semantic_learning_models.py";t=p.read_text();a=t.index("        triples=[]",t.index("if arm=='listmle':"));b=t.index("    if arm in ('linear_choice'",a)
t=t[:a]+"""        # Probability mass of each tied relevance group, conditional on the
        # remaining groups. Never impose an arbitrary order within a tie.
        levels=torch.unique(y,sorted=True).flip(0)
        if len(levels)<2:return None
        terms=[]
        for level in levels[:-1]:
            terms.append(torch.logsumexp(scores[y<=level],0)-torch.logsumexp(scores[y==level],0))
        return torch.stack(terms).mean()
"""+t[b:]
t=t.replace("return F.binary_cross_entropy_with_logits(scores[i]-scores[j],t)",
"""loss=F.binary_cross_entropy_with_logits(scores[i]-scores[j],t,reduction='none')
    if sample.pairs:return loss.mean() # Explicit teacher labels have no calibrated gain gap.
    gap=(y[i]-y[j]).abs()
    weight=torch.where(gap>0,gap/gap.max().clamp_min(1e-8),torch.full_like(gap,.1))
    return (loss*weight).sum()/weight.sum().clamp_min(1e-8)""")
p.write_text(t)
# Full trajectories, bounded per-stratum reservoirs, mixed behavior policies.
p=root/"experiments/stream_scheduler/train_real_suite.py";t=p.read_text();a=t.index("def collect(");b=t.index("\ndef frozen_warmstart",a)
t=t[:a]+"""def collect(episodes,max_decisions=256):
    \"\"\"Stratified reservoirs over COMPLETE causal trajectories, never a prefix cap.\"\"\"
    if max_decisions<1:return []
    rng=np.random.default_rng(1729);buckets={};seen={}
    for j,ep in enumerate(episodes):
        env=Environment(ep);policy=Rule(('random','spi_fifo','relevance','novelty')[j%4],j+2718)
        duration=ep.metadata.get('config',{}).get('duration',max(1,ep.horizon-16))
        while not env.done:
            s=env.snapshot()
            if len(s.ready)>1:
                phase=0 if s.now<.4*duration else 1 if s.now<.6*duration else 2
                queue=0 if len(s.ready)<5 else 1 if len(s.ready)<16 else 2
                key=(phase,queue);seen[key]=seen.get(key,0)+1
                sample=Sample(s,env.target(s),np.array([ep.reference_tiers[env.index[m.token]] for m in s.ready]),f'train-{j}',
                    target_source='original_image_VLM_silver_tier_D_marginal_risk')
                pool=buckets.setdefault(key,[])
                if len(pool)<max_decisions:pool.append(sample)
                else:
                    index=int(rng.integers(seen[key]))
                    if index<max_decisions:pool[index]=sample
            env.step(policy.select(s))
    for pool in buckets.values():rng.shuffle(pool)
    result=[];keys=sorted(buckets)
    while len(result)<max_decisions and keys:
        for key in list(keys):
            if buckets[key]:result.append(buckets[key].pop())
            else:keys.remove(key)
            if len(result)>=max_decisions:break
    return result

"""+t[b:];p.write_text(t)
# Neutral replay is not a qualified VLM experiment. Preserve the existing unbiased control.
edit("experiments/stream_scheduler/train_feedback_rl.py","if len(x)>=8:","if len(x)>=8 and len(set(y))>=2:")
edit("experiments/stream_scheduler/train_feedback_rl.py",
"return make_episode(records,features,vectors,text,count=min(96,len(records)),ues=(4,12,50,60,100)[seed%5],load=(.5,.9,1.2)[seed%3],seed=seed)",
"""from Evaluation.ue_scenarios import StreamConfig,build_stream
        return build_stream(records,features,vectors,text,StreamConfig(mode='mixed',ues=(4,12,50,60,100)[seed%5],load=(.5,.9,1.2)[seed%3],seed=seed,duration=200.,burst_factor=4.))""")
edit("experiments/stream_scheduler/train_feedback_rl.py",
"validation=[episode(\"validation\",70000+j) for j in range(15)]",
"""from Evaluation.ue_scenarios import StreamConfig,build_stream
    validation=[build_stream(val,features,vectors,text,StreamConfig(mode='mixed',ues=u,load=l,seed=70000+j,duration=200.,burst_factor=4.))
        for j,(u,l) in enumerate((u,l) for u in (4,12,50,60,100) for l in (.5,.9,1.2))]""")
# Stable single-seed default per latest user; do not alter any historical run.
edit("experiments/master_study/tests/test_controller.py","==69","==23")
print("Core repairs applied; historical inputs/results unchanged.")

