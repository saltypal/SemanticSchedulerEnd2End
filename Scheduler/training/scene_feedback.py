"""Join qualified teacher feedback to parameter-only learning records."""
from pathlib import Path
import json
import hashlib
from dataclasses import replace
import numpy as np
from Scheduler.training.scene_learners import Example,fit_supervised,train_dqn
from Scheduler.training.scene_teacher import aspect_preferences,normalized_pair,fit_reward_model,LearnedPrefixReward,PrefixPair
from Evaluation.scene_environment import transition_fingerprint


def validate_feedback(bundle,arm=None):
    if bundle.get('schema')!='scene_teacher_feedback_v1':raise ValueError('unknown teacher feedback schema')
    if not bundle.get('gate',{}).get('passed'):raise ValueError('task-specific teacher gate failed')
    if bundle.get('fixture',False):raise ValueError('fixture teacher feedback cannot train a measured arm')
    if arm:
        gate={'visionprefer':'aspects','rl_vlm_f':'prefix','vlm_replay':'replay'}[arm]
        if not bundle.get('gates',{}).get(gate,{}).get('passed'):
            raise ValueError('teacher task gate missing/failed: '+gate)
    return bundle


def train_teacher_arm(name,path,examples,factory,steps,epochs,validation,*,save_progress=None,deadline=None,experiment_id=None,
                      checkpoint=None,schedule_steps=None,expected_data_sha256=None,allowed_requests=None):
    from Scheduler.training.scene_study import restore_context
    bundle=validate_feedback(json.loads(Path(path).read_text()),name)
    if not expected_data_sha256 or bundle.get('data_sha256')!=expected_data_sha256:
        raise ValueError('teacher feedback cohort identity mismatch')
    if allowed_requests is None:raise ValueError('pinned teacher request manifest required')
    for category in ('image_pairs','prefix_pairs','replay'):
        for row in bundle.get(category,[]):
            expected=allowed_requests.get(row.get('request_id'))
            if expected is None or any(row.get(key)!=value for key,value in expected.items()):
                raise ValueError('teacher request/context/image/role differs from pinned manifest')
            if row.get('group',row['request_id'])!=row['request_id']:
                raise ValueError('teacher group differs from request identity')
    if name=='visionprefer':
        data=[]
        for row in bundle.get('image_pairs',[]):
            if row['role']!='train':continue
            preference=normalized_pair(row['forward']['parsed'],row['reverse']['parsed'])
            if preference not in ('A','B'):continue
            context=restore_context(row['context'])
            indices=row['candidate_indices']
            ratings=aspect_preferences(row['forward']['parsed'],('A','B'))
            reverse=aspect_preferences(row['reverse']['parsed'],('A','B'))
            mapped={(1-i,1-j,y) for i,j,y in reverse}
            pairs=tuple((indices[i],indices[j],y) for i,j,y in ratings if (i,j,y) in mapped)
            if pairs:data.append(Example(context,tuple(0. for _ in context.ready_candidates),row['group'],
                                         'qwen_aspect_preference',pairs))
        if len(data)<16:raise ValueError('fewer than16 accepted teacher image preferences')
        return fit_supervised(data,'visionprefer',epochs=epochs,validation_score=validation,
                              save_progress=save_progress,deadline=deadline,checkpoint=checkpoint)
    if name=='rl_vlm_f':
        pairs=[]
        for row in bundle.get('prefix_pairs',[]):
            pref=normalized_pair(row['forward']['parsed'],row['reverse']['parsed'])
            if pref not in ('A','B'):continue
            p=PrefixPair(tuple(restore_context(c) for c in row['left']),tuple(restore_context(c) for c in row['right']),
                tuple(row['durations_left']),tuple(row['durations_right']),float(pref=='A'),row['group'],
                indices_left=tuple(row['indices_left']),indices_right=tuple(row['indices_right']))
            pairs.append((row['role'],p))
        train=[p for role,p in pairs if role=='train'];val=[p for role,p in pairs if role=='validation']
        artifact,report,_=fit_reward_model(train,val,epochs=epochs)
        if not report['promoted']:raise ValueError('held-out preference reward gate failed')
        # Every stored reward is computed from the frozen model at learning time.
        def persist(state,history):
            state['reward_model_artifact']=artifact;state['reward_model_report']=report
            if save_progress:save_progress(state,history)
        result,history,state=train_dqn(factory,examples,algorithm=name,steps=steps,
            reward_model=LearnedPrefixReward(artifact),validation_score=validation,
            feedback_version=hashlib.sha256(Path(path).read_bytes()).hexdigest(),experiment_id=experiment_id,
            save_progress=persist,deadline=deadline,checkpoint=checkpoint,schedule_steps=schedule_steps)
        state['reward_model_artifact']=artifact;state['reward_model_report']=report
        return result,history,state
    if name=='vlm_replay':
        ratings={r['transition_key']:r['training_value'] for r in bundle.get('replay',[]) if r.get('valid')}
        if not ratings:raise ValueError('no accepted semantic replay feedback')
        matched=set();offered=[0]
        def semantic(t):
            key=transition_fingerprint(t)
            offered[0]+=1
            if key in ratings:matched.add(key)
            return {0:.5,1:1.,2:2.}.get(ratings.get(key),1.)
        artifact,history,state=train_dqn(factory,examples,algorithm=name,steps=steps,semantic_scorer=semantic,validation_score=validation,
            feedback_version=hashlib.sha256(Path(path).read_bytes()).hexdigest(),experiment_id=experiment_id,
            save_progress=save_progress,deadline=deadline,checkpoint=checkpoint,schedule_steps=schedule_steps)
        state['teacher_replay_matches']={'unique_matched':len(matched),'offered_transitions':offered[0],
                                         'available_ratings':len(ratings)}
        return artifact,history,state
    raise ValueError('unsupported teacher arm')
