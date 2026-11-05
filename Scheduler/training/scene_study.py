"""Bounded suite runner. Split roles, label origins and failures remain explicit."""
from pathlib import Path
import argparse
import csv
import json
import math
import time
import hashlib
from collections import OrderedDict
import numpy as np
from Scheduler.scene_policy import ScenePolicy,canonical
from Scheduler.training.scene_corpus import CausalCorpus,collect_examples
from Scheduler.training.scene_learners import ARMS,TEACHER_ARMS,fit_supervised,train_dqn,train_ppo
from Evaluation.scene_environment import run_policy,transition_fingerprint
from Scheduler.training.scene_prepare import write_json
from Scheduler.training.scene_evaluation import build_choice_probes,evaluate_choices,summarize_rollouts,choice_probe_identity


def serial_context(context):
    from dataclasses import asdict
    return asdict(context)


def restore_context(row):
    from Scheduler.scene_policy import SceneContext,SceneCandidate
    return SceneContext(tuple(SceneCandidate(**c) for c in row['ready_candidates']),
        tuple(SceneCandidate(**c) for c in row['successful_header_history']),
        row['attempts_completed'],row['admitted_count'],row['now'],row['horizon'],row['registry_hash'])


def teacher_requests(factory,count,output,role='train'):
    """Export causal decision requests; no reference annotations enter payload."""
    requests=[]
    for i in range(count):
        env=factory(i)
        # Initial snapshot has exactly eight ready candidates by protocol.
        ctx=env.observation();candidates=list(ctx.ready_candidates)
        if len(candidates)<2:continue
        selected=candidates[:2]
        by_id={m.message_id:m for m in env.episode.messages}
        row={'request_id':role+'-'+str(i),'role':role,'mission':'building_damage','time':ctx.now,
             'context':serial_context(ctx),'candidate_indices':[0,1],
             'images':[{'id':label,'image_id':by_id[c.message_id].payload_ref} for label,c in zip(('A','B'),selected)],
             'history_summary':'No successfully delivered evidence in this trace yet.',
             'scope':'causal source-image teacher proposal; no future candidates'}
        requests.append(row)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text('\n'.join(canonical(r) for r in requests)+'\n',encoding='utf-8')
    return requests


def feedback_requests(factory,count,output,role='train',policy=None):
    """Equal-time prefixes and replay clips from causal training/validation states.

    Image IDs are teacher-side joins only. Neither facts nor target gains are
    exposed. Four constant-service completions give the same elapsed endpoint.
    """
    requests=[]
    for i in range(count):
        sides=[]
        for side in range(2):
            env=factory(i);chooser=policy if side==0 and policy is not None else ScenePolicy(
                kind='fifo' if side==0 else 'random',seed=72000+i)
            contexts=[];indices=[];durations=[];images=[];times=[]
            by_id={m.message_id:m for m in env.episode.messages}
            while not env.finished and len(contexts)<4:
                ctx=env.observation();decision=chooser.select(ctx)
                identity=decision.message_id
                index=next(j for j,c in enumerate(ctx.ready_candidates) if c.message_id==identity)
                contexts.append(serial_context(ctx));indices.append(index)
                transition=env.step(identity);durations.append(transition.duration)
                images.append(by_id[identity].payload_ref);times.append(ctx.now+transition.duration)
            sides.append((contexts,indices,durations,images,times,transition_fingerprint(transition)))
        if any(len(s[0])!=4 for s in sides):continue
        if abs(sum(sides[0][2])-sum(sides[1][2]))>1e-9:continue
        left,right=sides
        requests.append({'kind':'prefix','request_id':role+'-prefix-'+str(i),'group':role+'-prefix-'+str(i),
            'role':role,'left':left[0],'right':right[0],'indices_left':left[1],'indices_right':right[1],
            'durations_left':left[2],'durations_right':right[2],
            'images_left':left[3],'images_right':right[3],'times_left':left[4],'times_right':right[4],
            'endpoint':sum(left[2]),'scope':'matched elapsed-time prefixes; visible source evidence only'})
        if role=='train' and i<24:
            requests.append({'kind':'replay','request_id':'replay-'+str(i),'role':role,
                'transition_key':left[5],
                'images':left[3],'times':left[4],'endpoint':sum(left[2]),
                'scope':'teacher training-value heuristic; not annotation reward'})
    Path(output).write_text('\n'.join(canonical(r) for r in requests)+'\n',encoding='utf-8')
    return requests


def evaluate(corpus,role,policy,seed=41000,traces=32):
    values=[];details=[]
    for index in range(traces):
        pattern=('no_repeat','local_repeat','cross_repeat')[index%3]
        load=('light','congested','burst')[(index//3)%3]
        key=(role,seed+index,pattern,load)
        if not hasattr(corpus,'evaluation_cache'):corpus.evaluation_cache={}
        if key not in corpus.evaluation_cache:
            env=corpus.make(role,seed+index,pattern=pattern,load=load)
            corpus.evaluation_cache[key]=(env.episode,env.registry,env.mission,env.specifications)
        from Evaluation.scene_environment import SceneEnvironment
        env=SceneEnvironment(*corpus.evaluation_cache[key])
        result=run_policy(env,policy);result.update(trace=index,pattern=pattern,load=load)
        details.append(result)
        if result['J_time'] is not None:values.append(result['J_time'])
    return float(np.mean(values)) if values else None,details


def run_study(data,upstream,output,*,epochs=50,steps=20000,train_traces=32,eval_traces=32,
              algorithms=ARMS,seconds=10800,open_pilot=False,teacher_file=None,resume_from=None,schedule_steps=None,
              collection_policy='random_damage_v1',training_stream_mode='fresh',
              q_value_mode='bounded_nonnegative_v1',rl_patience=3):
    import torch
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if training_stream_mode not in ('fresh','fixed'):raise ValueError('unknown training stream mode')
    if min(epochs,steps,train_traces,eval_traces,seconds)<=0:raise ValueError('positive experiment budgets required')
    start=time.perf_counter();corpus=CausalCorpus(data,upstream)
    experiment_id=hashlib.sha256((Path(data)/'records.json').read_bytes()+
        (Path(data)/'features.jsonl').read_bytes()+(Path(upstream)/'upstream_manifest.json').read_bytes()+
        canonical({'protocol':'scene_suite_v2','collection_policy':collection_policy,
                   'training_stream_mode':training_stream_mode,'q_value_mode':q_value_mode}).encode()).hexdigest()
    write_json(output/'config.json',{'data':str(data),'upstream':str(upstream),'epochs':epochs,'steps':steps,
        'train_traces':train_traces,'eval_traces':eval_traces,'algorithms':algorithms,'seconds':seconds,
        'open_pilot':open_pilot,'seed':1729,'timing_scope':'synthetic d0=1','one_seed_feasibility':True,
        'experiment_id':experiment_id,'resume_from':str(resume_from) if resume_from else None,
        'schedule_steps':schedule_steps or steps,'protocol':'scene_suite_v2','collection_policy':collection_policy,
        'training_stream_mode':training_stream_mode,'q_value_mode':q_value_mode,'rl_patience':rl_patience,
        'workload_macro':'raw trace mean retained; equal nine-cell macro also reported'})
    if resume_from:
        previous=json.loads((Path(resume_from)/'config.json').read_text())
        if previous.get('experiment_id')!=experiment_id or previous['train_traces']!=train_traces:
            raise ValueError('resume corpus/source/trace configuration mismatch')
    cached=OrderedDict()
    def factory(i):
        key=i%train_traces if training_stream_mode=='fixed' else i
        if key not in cached:
            env=corpus.make('train',10000+key,pattern=('no_repeat','local_repeat','cross_repeat')[key%3],
                            load=('light','congested','burst')[(key//3)%3])
            cached[key]=(env.episode,env.registry,env.mission,env.specifications)
            if len(cached)>64:cached.popitem(last=False)
        from Evaluation.scene_environment import SceneEnvironment
        return SceneEnvironment(*cached[key])
    print('Building annotation-backed training decisions',flush=True)
    examples=collect_examples(factory,train_traces,collection_policy=collection_policy)
    # No images/labels in these snapshots; targets persisted in separate rows.
    with (output/'training_decisions.jsonl').open('w',encoding='utf-8') as f:
        for e in examples:f.write(canonical({'context':serial_context(e.context),'gains':e.gains,
            'episode':e.episode_id,'origin':e.label_origin,'collection_policy':e.collection_policy})+'\n')
    teacher_requests(factory,min(24,train_traces),output/'teacher_requests_train.jsonl')
    val_factory=lambda i:corpus.make('validation',41000+i,pattern=('no_repeat','local_repeat','cross_repeat')[i%3])
    teacher_requests(val_factory,16,output/'teacher_requests_validation.jsonl',role='validation')
    feedback_requests(factory,min(24,train_traces),output/'feedback_requests_train.jsonl')
    feedback_requests(val_factory,16,output/'feedback_requests_validation.jsonl',role='validation')
    allowed_requests={}
    for request_path in output.glob('*requests*.jsonl'):
        for line in request_path.read_text().splitlines():
            if line.strip():
                row=json.loads(line);allowed_requests[row['request_id']]=row
    data_sha256=hashlib.sha256((Path(data)/'records.json').read_bytes()).hexdigest()
    write_json(output/'teacher_request_manifest.json',{'data_sha256':data_sha256,
        'requests':{key:hashlib.sha256(canonical(row).encode()).hexdigest() for key,row in allowed_requests.items()}})
    rows=[]
    print('Building shared-state validation choice probes',flush=True)
    probes=build_choice_probes(corpus,traces=eval_traces)
    write_json(output/'choice_probe_manifest.json',{'role':'validation','sha256':choice_probe_identity(probes),
        'snapshots':len(probes),'nontrivial':sum(len(set(e.gains))>1 for e in probes),
        'reference_policy':'seeded_random_v1','uses_evaluator_truth_for_targets_only':True})
    def validation(policy):
        value,_=evaluate(corpus,'validation',policy,traces=eval_traces)
        return value if value is not None else -1.
    def training_probe(policy):
        value,_=evaluate(corpus,'train',policy,seed=62000,traces=min(8,eval_traces))
        return value if value is not None else -1.
    references=('fifo','rr','random','damage_probability','damage_repeat_guard','context_alignment','appearance_novelty','concept_novelty')
    for name in references+tuple(algorithms):
        arm=output/'algorithms'/name;arm.mkdir(parents=True,exist_ok=False)
        begun=time.perf_counter();print('START',name,flush=True)
        record={'algorithm':name,'status':'running','scope':'real image parameters; synthetic service timing'}
        def save_progress(state,history):
            temporary=arm/'resume.tmp'
            torch.save(state,temporary);temporary.replace(arm/'resume.pt')
            write_json(arm/'learning_curve.json',history)
            print('CHECKPOINT',name,state.get('step',state.get('epoch')),flush=True)
        deadline=time.monotonic()+max(0,seconds-(time.perf_counter()-start))
        try:
            if time.perf_counter()-start>=seconds:raise TimeoutError('suite wall-clock cap; preserve previous arms')
            if name in TEACHER_ARMS and teacher_file is None:
                record.update(status='blocked',reason='task-qualified teacher responses/reward model are not supplied')
                write_json(arm/'status.json',record);rows.append(record);print('BLOCKED',name,flush=True);continue
            checkpoint_path=Path(resume_from)/'algorithms'/name/'resume.pt' if resume_from else None
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False) if checkpoint_path and checkpoint_path.exists() else None
            if name in TEACHER_ARMS:
                from Scheduler.training.scene_feedback import train_teacher_arm
                artifact,history,state=train_teacher_arm(name,Path(teacher_file),examples,factory,steps,epochs,validation,
                    save_progress=save_progress,deadline=deadline,experiment_id=experiment_id,
                    checkpoint=checkpoint,schedule_steps=schedule_steps,expected_data_sha256=data_sha256,
                    allowed_requests=allowed_requests)
                policy=ScenePolicy(artifact)
            elif name in ('fifo','rr','random'):
                policy=ScenePolicy(kind=name,seed=1729);artifact=None;history=[];state=None
            elif name=='damage_repeat_guard':
                fields=[i for i,f in enumerate(corpus.registry.fields) if f.startswith(('claim/','detection/'))]
                policy=ScenePolicy(kind='repeat_guard',
                    field=corpus.registry.fields.index('claim/building.affected_or_greater/score'),signature_fields=fields)
                artifact=None;history=[];state=None
            elif name in references:
                field={'damage_probability':'claim/building.affected_or_greater/score',
                    'context_alignment':'measurement/context_alignment',
                    'appearance_novelty':'measurement/novelty/appearance_distance',
                    'concept_novelty':'measurement/novelty/new_predicted_concept_fraction'}[name]
                policy=ScenePolicy(kind='single',field=corpus.registry.fields.index(field));artifact=None;history=[];state=None
            elif name=='masked_ppo':
                artifact,history,state=train_ppo(factory,examples,steps=steps,validation_score=validation,
                    save_progress=save_progress,deadline=deadline,experiment_id=experiment_id,checkpoint=checkpoint,
                    validation_patience=rl_patience,diagnostic_score=training_probe);policy=ScenePolicy(artifact)
            elif name in ('masked_double_dqn','td_per'):
                artifact,history,state=train_dqn(factory,examples,algorithm=name,steps=steps,validation_score=validation,
                    save_progress=save_progress,deadline=deadline,experiment_id=experiment_id,checkpoint=checkpoint,
                    schedule_steps=schedule_steps,value_mode=q_value_mode,validation_patience=rl_patience,
                    diagnostic_score=training_probe);policy=ScenePolicy(artifact)
            else:
                artifact,history,state=fit_supervised(examples,name,epochs=epochs,validation_score=validation,
                    save_progress=save_progress,deadline=deadline,checkpoint=checkpoint);policy=ScenePolicy(artifact)
            if artifact is not None:write_json(arm/'model.json',artifact)
            if state is not None:torch.save(state,arm/'resume.pt')
            write_json(arm/'learning_curve.json',history)
            val,details=evaluate(corpus,'validation',policy,traces=eval_traces)
            write_json(arm/'validation.json',details)
            interpretation=summarize_rollouts(details)
            choice=evaluate_choices(policy,probes)
            write_json(arm/'evaluation_summary.json',interpretation)
            write_json(arm/'shared_state_choices.json',choice)
            record.update(status='complete',validation_J=val,seconds=time.perf_counter()-begun)
            record.update(validation_useful_at_10=interpretation['distinct_useful_at_completions']['10']['mean'],
                validation_capped_delay=interpretation['mean_first_exposure_delay_capped']['mean'],
                validation_shared_choice_accuracy=choice['tie_aware_top1_accuracy'])
            if state:
                record['training_progress']=state.get('step',state.get('epoch'))
                record['training_stop_reason']=state.get('stop_reason','epoch_selection')
            if artifact:record['active_model_inputs']=len(artifact.get('feature_indices',artifact['mean']))
            if name=='vlm_replay':
                record['teacher_replay_matches']=state.get('teacher_replay_matches',{})
                if not record['teacher_replay_matches'].get('unique_matched'):
                    record.update(status='partial',reason='no exact teacher replay matches; neutral fallback only')
            measured=[r['seconds'] for trace in details for r in trace['selection_latency']]
            record['selection_p50_seconds']=float(np.median(measured)) if measured else None
            record['selection_p95_seconds']=float(np.quantile(measured,.95)) if measured else None
            record['artifact_bytes']=len(canonical(artifact).encode()) if artifact else 0
            if open_pilot:
                value,details=evaluate(corpus,'pilot_eval',policy,seed=51000,traces=eval_traces)
                write_json(arm/'pilot_eval.json',details);record['pilot_J']=value
        except Exception as error:
            import traceback
            record.update(status='partial' if isinstance(error,TimeoutError) else 'failed',error=type(error).__name__,reason=str(error),seconds=time.perf_counter()-begun)
            (arm/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(arm/'status.json',record);rows.append(record);write_json(output/'progress.json',rows)
        print('END',name,record['status'],'seconds',round(time.perf_counter()-begun,1),flush=True)
    write_json(output/'summary.json',rows)
    columns=['algorithm','status','validation_J','validation_useful_at_10','validation_capped_delay',
        'validation_shared_choice_accuracy','training_progress','training_stop_reason','pilot_J','seconds','reason']
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    (output/'REPORT.md').write_text('# Scene scheduler feasibility study\n\n'+
        '\n'.join(f"- {r['algorithm']}: {r['status']}; validation J={r.get('validation_J','unavailable')}; {r.get('reason','')}" for r in rows)+
        '\n\nOne training seed, simulated service timing. No natural cross-view identity, operational urgency or superiority claim.\n',encoding='utf-8')
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--upstream',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=20000);p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--train-traces',type=int,default=32);p.add_argument('--eval-traces',type=int,default=32)
    p.add_argument('--seconds',type=int,default=10800);p.add_argument('--algorithms',nargs='+',choices=ARMS,default=ARMS)
    p.add_argument('--open-pilot',action='store_true');p.add_argument('--teacher-file')
    p.add_argument('--resume-from');p.add_argument('--schedule-steps',type=int)
    p.add_argument('--collection-policy',choices=('random_damage_v1','random_oracle_v1'),default='random_damage_v1')
    p.add_argument('--training-stream-mode',choices=('fresh','fixed'),default='fresh')
    p.add_argument('--q-value-mode',choices=('bounded_nonnegative_v1','unbounded'),default='bounded_nonnegative_v1')
    p.add_argument('--rl-patience',type=int,default=3)
    a=p.parse_args();run_study(a.data,a.upstream,a.output,steps=a.steps,epochs=a.epochs,train_traces=a.train_traces,
        eval_traces=a.eval_traces,seconds=a.seconds,algorithms=a.algorithms,open_pilot=a.open_pilot,teacher_file=a.teacher_file,
        resume_from=a.resume_from,schedule_steps=a.schedule_steps,collection_policy=a.collection_policy,
        training_stream_mode=a.training_stream_mode,q_value_mode=a.q_value_mode,rl_patience=a.rl_patience)

if __name__=='__main__':main()
