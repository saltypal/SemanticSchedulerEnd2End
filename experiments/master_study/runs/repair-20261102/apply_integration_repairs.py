from pathlib import Path
root=Path.cwd()
p=root/"experiments/stream_scheduler/semantic_suite_data.py"
with p.open("a") as f:f.write('''

def development_cohorts(rows,require_reference=True):
    """One grouped role assignment shared by every training driver and teacher."""
    eligible=[r for r in rows if r["split"]=="train" and (not require_reference or r["reference_tier"]>=0)]
    held={r["group"] for r in eligible if int(hashlib.sha256(r["group"].encode()).hexdigest()[:8],16)%5==0}
    return ([r for r in eligible if r["group"] not in held],[r for r in eligible if r["group"] in held])

def cohort_coverage(rows):
    return {source:{"offered":sum(r["dataset"]==source for r in rows),
                    "assessable":sum(r["dataset"]==source and r["reference_tier"]>=0 for r in rows)}
            for source in sorted({r["dataset"] for r in rows})}
''')
p=root/"experiments/stream_scheduler/train_real_suite.py";t=p.read_text()
t=t.replace("from experiments.stream_scheduler.semantic_suite_data import load_pool,","from experiments.stream_scheduler.semantic_suite_data import development_cohorts,cohort_coverage,load_pool,")
t=t.replace("ap.add_argument('--scenario-manifest',type=Path)","ap.add_argument('--scenario-manifest',type=Path)\n    ap.add_argument('--require-sources',default='',help='Fail before training when a requested source lacks independent references')")
a=t.index("    train_rows=split_records(");b=t.index("    if len(train_rows)<48",a)
t=t[:a]+'''    train_rows,val_rows=development_cohorts(rows)
    coverage=cohort_coverage(rows)
    write(out/'reference_coverage.json',coverage)
    missing=[source for source in args.require_sources.split(',') if source and not coverage.get(source,{}).get('assessable',0)]
    if missing:raise ValueError('requested source lacks reference labels: '+','.join(missing))
    write(out/'development_split.json',{'scope':'scheduler-only grouped development; exposed producers; not final evaluation',
        'train_ids':[r['image_id'] for r in train_rows],'validation_ids':[r['image_id'] for r in val_rows],
        'group_disjoint':not ({r['group'] for r in train_rows}&{r['group'] for r in val_rows}),
        'vision_producer_exposure':'known/possible; not independent emergency evaluation'})
''' +t[b:]
t=t.replace("write(out/f'training_{seed}.json',",
"""write(out/f'coverage_{seed}.json',{'snapshots':len(samples),'max_queue':max((len(s.snapshot.ready) for s in samples),default=0),
            'phase_counts':{phase:sum((0 if s.snapshot.now<.4*args.stream_duration else 1 if s.snapshot.now<.6*args.stream_duration else 2)==phase for s in samples) for phase in range(3)},
            'source_counts':{source:sum(r['dataset']==source for r in train_rows) for source in coverage},
            'supervision':'D reference class/age only; not mission/evidence preference'})
        write(out/f'training_{seed}.json',""")
p.write_text(t)
p=root/"experiments/stream_scheduler/train_feedback_rl.py";t=p.read_text()
t=t.replace("from experiments.stream_scheduler.semantic_suite_data import load_pool,","from experiments.stream_scheduler.semantic_suite_data import development_cohorts,load_pool,")
a=t.index('    cohort=split_records(');b=t.index('    text=np.load',a)
t=t[:a]+'    train,val=development_cohorts(rows)\n'+t[b:];p.write_text(t)
# Teacher roles must also respect scheduler holdout, not just original image split.
p=root/"experiments/stream_scheduler/teacher_pipeline.py";t=p.read_text()
t=t.replace("from experiments.stream_scheduler.semantic_suite_data import load_pool","from experiments.stream_scheduler.semantic_suite_data import development_cohorts,load_pool")
t=t.replace('    control_ids={r["image_id"]',
'''    train_roles,validation_roles=development_cohorts(rows,False)
    role_by_id={r["image_id"]:"train" for r in train_roles}
    role_by_id.update({r["image_id"]:"validation" for r in validation_roles})
    rows=[dict(r,split=role_by_id[r["image_id"]]) for r in rows if r["image_id"] in role_by_id]
    control_ids={r["image_id"]''')
p.write_text(t)
# Write pending-method identities before the first subprocess, even if killed later.
p=root/"experiments/master_study/master_suite.py";t=p.read_text()
anchor='    (a.output/"logs").mkdir()'
t=t.replace(anchor,anchor+'''
    declared=("linear_choice","mlp_choice","delay_huber","ranknet","lambdamart","priority_classifier",
              "deepsets_ranknet","set_transformer_ranknet","listmle","masked_ppo","masked_double_dqn",
              "td_per","ranknet_warmstart_ppo","temporal_deepsets_ranknet","rl_vlm_f","vlm_replay")
    pending={arm+"/"+seed:{"status":"not_started"} for arm in declared for seed in SEEDS.split(",")}
    for arm in ("linear_choice","mlp_choice","ranknet","deepsets_ranknet","set_transformer_ranknet","temporal_deepsets_ranknet","visionprefer"):
        for seed in SEEDS.split(","):pending["P/"+arm+"/"+seed]={"status":"not_started"}
    (a.output/"method_ledger.json").write_text(json.dumps(pending,indent=2))
''')
t=t.replace('"--stream-mode","mixed","--stream-duration",200,"--stream-burst-factor",4]',
'"--stream-mode","mixed","--stream-duration",200,"--stream-burst-factor",4,"--require-sources","crisismmd,open_images,places365"]')
p.write_text(t)
print("Shared roles, source gates, coverage reports and early ledger integrated.")

