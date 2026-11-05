from pathlib import Path
r=Path.cwd()
p=r/"Evaluation/ue_scenarios.py";t=p.read_text()
t=t.replace("last={};banks={};messages=[];tiers=[];facts=[];events=[]","last={};banks={};messages=[];tiers=[];facts=[];events=[];weights=[]")
t=t.replace('tiers.append(row.get("reference_tier",-1));facts.append(row["image_sha256"])',
'''tiers.append(row.get("reference_tier",-1))
        utility=annotation.get("utility_by_mission",{}).get(mission)
        weights.append(utility)
        facts.append((row["image_sha256"]+":"+mission) if utility is not None else row["image_sha256"])''')
t=t.replace("return Episode(tuple(messages),tuple(tiers),tuple(facts),(c.service_time,)*len(messages),(False,)*len(messages),horizon,metadata)",
"""if any(w is not None for w in weights) and not all(w is not None for w in weights):
        raise ValueError('incomplete independent mission references for offered stream')
    return Episode(tuple(messages),tuple(tiers),tuple(facts),(c.service_time,)*len(messages),(False,)*len(messages),horizon,metadata,
        tuple(weights) if weights and all(w is not None for w in weights) else None)""")
# Validate explicit task utility rather than treating predicted parameters as truth.
t=t.replace('        result[row["image_id"]]=dict(row)',
'''        if "utility_by_mission" in row:
            if row.get("reference_kind") not in ("published_annotation","controlled_construction"):
                raise ValueError("independent reference provenance required")
            if not row.get("reference_source"):raise ValueError("reference source required")
            for mission,value in row["utility_by_mission"].items():
                if mission not in {p[0] for p in PROFILES} or type(value) not in (int,float) or value not in (0,1):
                    raise ValueError("binary supported mission utility required")
        result[row["image_id"]]=dict(row)''')
p.write_text(t)
p=r/"experiments/stream_scheduler/train_real_suite.py";t=p.read_text()
t=t.replace("def collect(episodes,max_decisions=256):","def collect(episodes,max_decisions=256,task='D'):")
t=t.replace("env=Environment(ep);policy=Rule","env=Environment(ep,task);policy=Rule")
t=t.replace("ap.add_argument('--require-sources'","ap.add_argument('--objective',choices=('D','E'),default='D')\n    ap.add_argument('--require-sources'")
t=t.replace("train_rows,val_rows=development_cohorts(rows)","train_rows,val_rows=development_cohorts(rows,args.objective=='D')")
t=t.replace("    validation=[episode(","""    if args.objective=='E' and not scenario_annotations:
        raise ValueError('E requires independent mission utility annotation manifest; predicted R/N are not labels')
    validation=[episode(""")
t=t.replace("    probes=collect(validation,max_decisions=128)","""    if args.objective=='E' and any(e.evidence_weights is None for e in validation):
        raise ValueError('E requires complete independently sourced utility labels')
    def run_evaluate(policy,episodes,output=None,probes=None):
        return evaluate(policy,episodes,output,probes,task=args.objective)
    probes=collect(validation,max_decisions=128,task=args.objective)""")
t=t.replace("value,result=evaluate(Rule(kind)","value,result=run_evaluate(Rule(kind)")
t=t.replace("samples=collect(episodes,args.samples)","samples=collect(episodes,args.samples,task=args.objective)")
t=t.replace("def validate(artifact):return evaluate(","def validate(artifact):return run_evaluate(")
t=t.replace("value,result=evaluate(NumpyRanker","value,result=run_evaluate(NumpyRanker")
t=t.replace("samples,pre,arm,dest,validate,seed=seed,epochs=args.epochs,seconds=slice_seconds,task='D'","samples,pre,arm,dest,validate,seed=seed,epochs=args.epochs,seconds=slice_seconds,task=args.objective")
t=t.replace("                if arm=='lambdamart':",'''                if args.objective=='E' and arm=='priority_classifier':
                    statuses[key]={'status':'unsupported_objective','reason':'four-tier classifier is D baseline, not binary mission utility'}
                    write(out/'status.json',{'phase':'training','arms':statuses});continue
                if arm=='lambdamart':''')
t=t.replace("load=(.5,.9,1.2)[j%3])),pre,dest","load=(.5,.9,1.2)[j%3]),args.objective),pre,dest")
# Include objective and changed source in resume identities.
t=t.replace("'dynamic_ues','burst','mission_vectors_sha256'","'dynamic_ues','burst','objective','require_sources','mission_vectors_sha256'")
t=t.replace("'profile':profile,'seed':seed,","'profile':profile,'objective':args.objective,'seed':seed,")
p.write_text(t)
# Include objective in exported RL metadata and restored environment.
p=r/"Scheduler/training/semantic_learning_rl.py";t=p.read_text()
t=t.replace("'task':'D' if reward_model is None else 'P'","'task':env.task if reward_model is None else 'P'")
t=t.replace("'reward':'annotation_D' if reward_model is None else feedback_version","'reward':'reference_'+env.task if reward_model is None else feedback_version")
t=t.replace("envs=[Environment(e,'D') for e in saved['env_episodes']]","envs=[Environment(e,task) for e,task in zip(saved['env_episodes'],saved.get('env_tasks',['D']*len(saved['env_episodes'])))]")
t=t.replace("{'task':'D','reward':'annotation_D'","{'task':envs[0].task,'reward':'reference_'+envs[0].task")
t=t.replace("'env_episodes':[e.episode for e in envs],","'env_episodes':[e.episode for e in envs],'env_tasks':[e.task for e in envs],")
p.write_text(t)
print("Independent binary mission-evidence path integrated separately from D.")

