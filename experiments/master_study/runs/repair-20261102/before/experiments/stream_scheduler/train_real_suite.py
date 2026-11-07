"""Observable real-data D suite; architecture training never uses SPI as its target."""
import argparse
import copy
from dataclasses import replace
import importlib.util
import json
import hashlib
import pickle
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('_suite_boot',Path(__file__).with_name('runtime_bootstrap.py'))
boot=importlib.util.module_from_spec(spec);spec.loader.exec_module(boot);boot.install(ROOT)
from Evaluation.semantic_learning_env import Environment,Rule,make_episode
from Scheduler.semantic_learning_policy import NumpyRanker,fit_normalizer
from Scheduler.training.semantic_learning_models import Sample,fit_supervised,fit_lambdamart,ARMS
from Scheduler.training.semantic_learning_rl import train_dqn,train_ppo
from experiments.stream_scheduler.semantic_suite_data import load_pool,split_records,mission_vectors,load_mission_vectors,file_hash


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def evaluate(policy,episodes,output=None,probes=None):
    results=[]
    for j,episode in enumerate(episodes):
        env=Environment(episode,'D');latency=[]
        while not env.done:
            s=env.snapshot();start=time.perf_counter_ns();token=policy.select(s);latency.append((time.perf_counter_ns()-start)/1e6)
            env.step(token)
        row=env.metrics();row['selection_ms']={'p50':float(np.median(latency)),'p95':float(np.quantile(latency,.95)),'n':len(latency)}
        row['reference_scope']='mixed-protocol image VLM silver tiers; not independent emergency correctness'
        results.append(row)
        if output:
            write(output/f'episode_{j}.json',{'metadata':episode.metadata,'metrics':row,'events':env.events})
    if probes is not None and output:
        ranking=evaluate_probes(policy,probes)
        write(output/'ranking_probes.json',ranking)
    return -float(np.mean([r['J_D'] for r in results])),results


def evaluate_probes(policy,probes):
    values={k:[] for k in (1,3,5)};singletons=empty=zero_ideal=0;counts=[]
    for sample in probes:
        snapshot=sample.snapshot;n=len(snapshot.ready);counts.append(n)
        if n==0:empty+=1;continue
        if n==1:singletons+=1;continue
        remaining=list(range(n));order=[];reference=np.asarray(sample.target,dtype=float)
        clone=copy.deepcopy(policy)
        for rank in range(min(5,n)):
            view=replace(snapshot,ready=tuple(snapshot.ready[j] for j in remaining),ue_windows=tuple(snapshot.ue_windows[j] for j in remaining))
            token=clone.select(view);j=next(j for j in remaining if snapshot.ready[j].token==token)
            order.append(j);remaining.remove(j)
        ideal=np.sort(reference)[::-1]
        for k in values:
            q=min(k,n);discount=np.log2(np.arange(q)+2)
            denominator=float(np.sum(ideal[:q]/discount))
            if denominator<=0:zero_ideal+=1;continue
            values[k].append(float(np.sum(reference[order[:q]]/discount)/denominator))
    return {'definition':'common reference snapshots; recursively select up to5 without advancing time/history; linear D marginal-risk gains','ndcg':{str(k):{'mean':float(np.mean(v)) if v else None,'defined':len(v)} for k,v in values.items()},'singletons':singletons,'empty':empty,'zero_ideal_metric_cells':zero_ideal,'candidate_counts':counts}



def collect(episodes,max_decisions=256):
    """Interleave causal trajectories so the sample cap cannot omit late cohorts."""
    out=[]
    states=[(Environment(episode),Rule('random',j+2718),j)
            for j,episode in enumerate(episodes)]
    while len(out)<max_decisions:
        progressed=False
        for env,policy,j in states:
            if env.done:continue
            progressed=True
            s=env.snapshot();risk=env.target(s)
            tiers=np.array([env.episode.reference_tiers[env.index[m.token]] for m in s.ready])
            if len(s.ready)>1:
                out.append(Sample(s,risk,tiers,f'train-{j}',
                    target_source='original_image_VLM_silver_tier_D_marginal_risk'))
            env.step(policy.select(s))
            if len(out)>=max_decisions:return out
        if not progressed:break
    return out


def frozen_warmstart(source, destination):
    """Bind PPO to its original pretrained actor even if that producer improves."""
    import torch
    destination=Path(destination)
    if destination.exists():
        record=torch.load(destination,map_location="cpu",weights_only=False)
        actual=hashlib.sha256(b"".join(key.encode()+value.detach().cpu().numpy().tobytes() for key,value in sorted(record["state"].items()))).hexdigest()
        if actual!=record["sha256"]:raise ValueError("frozen warmstart checksum mismatch")
        return record["state"]
    source=Path(source)
    if not source.exists():raise ValueError("validated compatible Deep Sets warmstart checkpoint missing")
    state=torch.load(source,map_location="cpu",weights_only=False)["best_state"]
    checksum=hashlib.sha256(b"".join(key.encode()+value.detach().cpu().numpy().tobytes() for key,value in sorted(state.items()))).hexdigest()
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state":state,"sha256":checksum,"source":str(source)},destination)
    return state


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--seeds',default='1729,2718,31415')
    ap.add_argument('--training-episodes',type=int,default=64);ap.add_argument('--samples',type=int,default=512)
    ap.add_argument('--model-seconds',type=float,default=None);ap.add_argument('--validation-traces',type=int,default=15);ap.add_argument('--arms',default='all');ap.add_argument('--mission-text',action='store_true');ap.add_argument('--device',default='cpu');ap.add_argument('--rl-steps',type=int,default=20000);ap.add_argument('--deadline-seconds',type=float,default=None);ap.add_argument('--resume',action='store_true');ap.add_argument('--dynamic-ues',action='store_true');ap.add_argument('--burst',action='store_true')
    ap.add_argument('--vector-extension',type=Path)
    ap.add_argument('--mission-vectors',type=Path)
    ap.add_argument('--stream-mode',choices=('legacy','mixed','mission','incident'),default='legacy')
    ap.add_argument('--scenario-manifest',type=Path)
    ap.add_argument('--stream-duration',type=float,default=200.)
    ap.add_argument('--stream-burst-factor',type=float,default=1.)
    args=ap.parse_args();deadline=float("inf") if args.deadline_seconds is None else time.monotonic()+args.deadline_seconds;args.output.mkdir(parents=True,exist_ok=args.resume);out=args.output
    import torch
    torch.set_num_threads(1)
    config=vars(args).copy();config['archive']=str(args.archive);config['output']=str(out);config['vector_extension']=str(args.vector_extension) if args.vector_extension else None;config['mission_vectors']=str(args.mission_vectors) if args.mission_vectors else None;config['scenario_manifest']=str(args.scenario_manifest) if args.scenario_manifest else None;config['scenario_manifest_sha256']=file_hash(args.scenario_manifest) if args.scenario_manifest else None;config['mission_vectors_sha256']=file_hash(args.mission_vectors) if args.mission_vectors else None
    if args.resume:
        prior=json.loads((out/'config.json').read_text())
        for key in ('archive','seeds','training_episodes','samples','rl_steps','mission_text','arms','dynamic_ues','burst','mission_vectors_sha256','scenario_manifest_sha256','stream_mode','stream_duration','stream_burst_factor'):
            if prior[key]!=config[key]:raise ValueError('run resume configuration mismatch: '+key)
    else:write(out/'config.json',config)
    print('START REAL DATA SUITE',config,flush=True)
    rows,features,vectors,profile,audit=load_pool(args.archive,out/'data',args.vector_extension)
    text=load_mission_vectors(args.mission_vectors) if args.mission_vectors else mission_vectors(out/'data') if args.mission_text else None
    if text is None:print('Mission cosine unavailable in local no-download test; causal novelty computed from cached compatible vectors',flush=True)
    train_rows=split_records(rows,'train',True);val_rows=split_records(rows,'validation',True)
    if len(val_rows)<48:
        # The pool's independently held-out images have no matching reference tiers.
        # Use a new SCHEDULER-only grouped development split inside exposed train rows.
        # This is explicitly not an independent SPI/vision/final evaluation.
        original=train_rows
        holdout={r['group'] for r in original if int(hashlib.sha256(r['group'].encode()).hexdigest()[:8],16)%5==0}
        train_rows=[r for r in original if r['group'] not in holdout];val_rows=[r for r in original if r['group'] in holdout]
        write(out/'development_split.json',{'scope':'scheduler-only grouped development validation within exposed source-train; final source test unopened','train_ids':[r['image_id'] for r in train_rows],'validation_ids':[r['image_id'] for r in val_rows],'group_disjoint':True,'vision_producer_exposure':'known/possible; not independent emergency evaluation'})
    if len(train_rows)<48 or len(val_rows)<48:raise ValueError('insufficient grouped silver-reference train/validation cohort')
    scenario_annotations=[json.loads(line) for line in args.scenario_manifest.read_text().splitlines() if line.strip()] if args.scenario_manifest else []
    def episode(role,seed,ues=4,load=.9):
        cohort=train_rows if role=='train' else val_rows
        if args.stream_mode!='legacy':
            from Evaluation.ue_scenarios import StreamConfig,build_stream
            ids={r['image_id'] for r in cohort}
            annotations=[a for a in scenario_annotations if a['image_id'] in ids]
            return build_stream(cohort,features,vectors,text,StreamConfig(mode=args.stream_mode,ues=ues,load=load,seed=seed,duration=args.stream_duration,burst_factor=args.stream_burst_factor),annotations)
        return make_episode(cohort,features,vectors,text,count=min({4:48,12:96,50:256,60:384,100:512}[ues],len(cohort)),ues=ues,load=load,seed=seed,repeat='none',churn=args.dynamic_ues,burst=args.burst)
    validation=[episode('validation',70000+i,ues,load) for i,(ues,load) in enumerate((u,l) for u in (4,12,50,60,100) for l in (.5,.9,1.2))]
    validation=validation[:args.validation_traces]
    probes=collect(validation,max_decisions=128)
    with (out/'ranking_probes.pickle').open('wb') as f:pickle.dump(probes,f)
    baselines={}
    for kind in ('fifo','round_robin','random','spi_fifo','relevance','novelty','spi_novelty','predicted_edf'):
        print('BASELINE',kind,flush=True);value,result=evaluate(Rule(kind),validation,out/'baselines'/kind,probes)
        baselines[kind]={'score':value,'episodes':result}
    write(out/'baseline_summary.json',baselines)
    statuses={a+'/'+seed:{'status':'planned'} for seed in args.seeds.split(',') for a in ARMS};write(out/'status.json',{'phase':'supervised_training','arms':statuses})
    arms=tuple(arm for arm in ARMS if arm not in ('rl_vlm_f','vlm_replay','visionprefer'))
    if args.arms!='all':arms=tuple(a for a in arms if a in args.arms.split(','))
    for arm in ('rl_vlm_f','vlm_replay','visionprefer'):
        for seed in args.seeds.split(','):
            statuses[arm+'/'+seed]={'status':'blocked_labels','reason':'Teacher-dependent arm is executed by the separate qualified-feedback drivers'}
    all_seeds=[int(x) for x in args.seeds.split(',')]
    for seed_index,seed in enumerate(all_seeds):
        episodes=[episode('train',seed*100+j,ues=(4,12,50,60,100)[j%5],load=(.5,.9,1.2)[j%3]) for j in range(args.training_episodes)]
        pre=fit_normalizer(episodes);samples=collect(episodes,args.samples)
        warm_dest=out/'models'/'ranknet_warmstart_ppo'/str(seed)
        if args.resume and (warm_dest/'resume.pt').exists():
            frozen_warmstart(out/'models'/'deepsets_ranknet'/str(seed)/'resume.pt',warm_dest/'warmstart.pt')
        write(out/f'training_{seed}.json',{'offered_episodes':len(episodes),'optimization_decisions':len(samples),'preprocessing':pre,'labels':'separate original image VLM tiers, not SPI predictions','final_test_opened':False})
        # Persist reference/observation snapshot exactly; never infer resumability from weights alone.
        with (out/f'training_snapshots_{seed}.pickle').open('wb') as f:pickle.dump(samples,f)
        for arm_index,arm in enumerate(arms):
            key=f'{arm}/{seed}';dest=out/'models'/arm/str(seed)
            remaining=deadline-time.monotonic()
            if remaining<=0:
                statuses[key]={'status':'partial_budget','reason':'global deadline before start'}
                write(out/'status.json',{'phase':'training','arms':statuses});continue
            slice_seconds=None if args.model_seconds is None and args.deadline_seconds is None else min(args.model_seconds or float('inf'),max(0.,remaining-120)/max(1,(len(all_seeds)-seed_index-1)*len(arms)+len(arms)-arm_index))
            if slice_seconds is not None and slice_seconds<=0:
                statuses[key]={'status':'partial_budget','reason':'reserved retrieval margin reached'};continue
            statuses[key]={'status':'running','device':args.device}
            write(out/'status.json',{'phase':'training','arms':statuses})
            print('TRAIN',key,flush=True);start=time.perf_counter()
            try:
                def validate(artifact):return evaluate(NumpyRanker(artifact),validation)
                if arm in ('rl_vlm_f','vlm_replay','visionprefer'):
                    statuses[key]={'status':'blocked_labels','reason':'Qualified contextual teacher artifacts not supplied; SPI pseudo-labels cannot substitute'}
                    write(out/'status.json',{'phase':'training','arms':statuses});continue
                if arm=='lambdamart':artifact=fit_lambdamart(samples,pre,dest,seed)
                elif arm in ('masked_double_dqn','td_per'):
                    artifact,history=train_dqn(lambda j:Environment(episode('train',seed*10000+j,ues=(4,12,50,60,100)[j%5],load=(.5,.9,1.2)[j%3])),pre,dest,validate,arm=arm,seed=seed,steps=args.rl_steps,seconds=slice_seconds,device=args.device,resume=(dest/'resume.pt') if args.resume and (dest/'resume.pt').exists() else None,training_id=hashlib.sha256(json.dumps({'profile':profile,'seed':seed,'train_groups':sorted({r['group'] for r in train_rows}),'episodes':args.training_episodes,'dynamic_ues':args.dynamic_ues,'burst':args.burst,'environment_sha256':hashlib.sha256((ROOT/'Evaluation/semantic_learning_env.py').read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest())
                elif arm in ('masked_ppo','ranknet_warmstart_ppo'):
                    warm=None
                    if arm=='ranknet_warmstart_ppo':
                        checkpoint=out/'models'/'deepsets_ranknet'/str(seed)/'resume.pt'
                        warm=frozen_warmstart(checkpoint,dest/'warmstart.pt')
                    artifact,history=train_ppo(lambda j:Environment(episode('train',seed*10000+j,ues=(4,12,50,60,100)[j%5],load=(.5,.9,1.2)[j%3])),pre,dest,validate,seed=seed,steps=args.rl_steps,seconds=slice_seconds,device=args.device,warmstart=warm,resume=(dest/'resume.pt') if args.resume and (dest/'resume.pt').exists() else None,training_id=hashlib.sha256(json.dumps({'profile':profile,'seed':seed,'train_groups':sorted({r['group'] for r in train_rows}),'episodes':args.training_episodes,'dynamic_ues':args.dynamic_ues,'burst':args.burst,'environment_sha256':hashlib.sha256((ROOT/'Evaluation/semantic_learning_env.py').read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest())
                else:artifact,history=fit_supervised(samples,pre,arm,dest,validate,seed=seed,epochs=args.epochs,seconds=slice_seconds,task='D',device=args.device,resume=(dest/'resume.pt') if args.resume and (dest/'resume.pt').exists() else None)
                value,result=evaluate(NumpyRanker(artifact),validation,dest/'validation',probes)
                statuses[key]={'status':('partial_budget' if artifact.get('source',{}).get('stop')=='partial_budget' or (arm in ('masked_ppo','masked_double_dqn','td_per','ranknet_warmstart_ppo') and (history[-1].get('steps',args.rl_steps) if history else 0)<args.rl_steps) else 'evaluated_development_only'),'seconds':time.perf_counter()-start,'validation_score':value,'model_sha256':artifact['sha256'],'episodes':len(episodes),'samples':len(samples),'final_test_opened':False}
            except Exception as error:
                dest.mkdir(parents=True,exist_ok=True);(dest/'error.txt').write_text(traceback.format_exc())
                statuses[key]={'status':'failed','error':str(error),'seconds':time.perf_counter()-start}
                print('FAILED',key,repr(error),flush=True)
            write(out/'status.json',{'phase':'supervised_training','arms':statuses})
    write(out/'status.json',{'phase':'finished_bounded_suite','arms':statuses,'complete_requested_method_set':args.arms=='all','teacher_arms_delegated':True})
    write(out/'REPORT.json',{'scope':'real-cache silver-D supervised and RL suite; no independent urgency claim','statuses':statuses,
        'input_profile':profile,'data_audit':audit,'final_test_opened':False,'P_teacher':'blocked until qualified contextual labels are collected','RL':'included; inspect per-arm ledger',
        'UE_mode':{'dynamic_ues':args.dynamic_ues,'burst':args.burst,'scope':'synthetic mixed-image streams; not real video chronology'}})
    write(out/'artifact_manifest.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})
    print('FINISHED bounded method attempts; consult ledger',out,flush=True)


if __name__=='__main__':main()
