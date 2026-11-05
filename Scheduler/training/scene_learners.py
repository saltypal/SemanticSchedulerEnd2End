"""Training-only choice, preference, tree, PPO and replay learners.

Runtime uses exported NumPy weights. Independent references stay in Evaluation.
"""
from dataclasses import dataclass
from collections import Counter
import copy
import math
import random
import time
import hashlib
from dataclasses import asdict
import numpy as np
from Scheduler.scene_policy import ScenePolicy,context_matrix,seal_artifact,tie_key,canonical

ARMS=('linear_choice','mlp_choice','ranknet','lambdamart','priority_classifier',
      'masked_ppo','masked_double_dqn','td_per','rl_vlm_f','vlm_replay','visionprefer')
TEACHER_ARMS={'rl_vlm_f','vlm_replay','visionprefer'}


def training_identity(examples,experiment_id=None):
    h=hashlib.sha256()
    for e in examples:
        h.update(canonical({'context':asdict(e.context),'gains':e.gains,
            'preferences':e.preferences,'origin':e.label_origin,'collection_policy':e.collection_policy}).encode())
    return {'training_records_sha256':h.hexdigest(),'episode_manifest':experiment_id or 'fixture-unspecified-stream'}


@dataclass(frozen=True)
class Example:
    context:object
    gains:tuple
    episode_id:str
    label_origin:str='annotation'
    # Preferences carry candidate index pairs and probability first is preferred.
    preferences:tuple=()
    collection_policy:str='unspecified'


def seed_everything(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def preprocessing(examples):
    if not examples:raise ValueError('training examples required')
    registry=examples[0].context.registry_hash
    if any(e.context.registry_hash!=registry for e in examples):raise ValueError('mixed registries')
    d=len(examples[0].context.ready_candidates[0].values)
    medians=[];distance_scale=[]
    unique={(e.episode_id,c.message_id):c for e in examples for c in e.context.ready_candidates}
    for i in range(d):
        v=[c.values[i] for c in unique.values() if c.available[i]]
        medians.append(float(np.median(v)) if v else 0.)
        deviation=float(np.std(v)) if v else 0.
        distance_scale.append(deviation if deviation>1e-8 else 1.)
    counts=Counter(e.episode_id for e in examples)
    xs=[];ws=[]
    for e in examples:
        x=context_matrix(e.context,medians,distance_scale)
        xs.append(x);ws.extend([1/(counts[e.episode_id]*max(1,len(x)))]*len(x))
    x=np.concatenate(xs);w=np.asarray(ws)
    mean=np.average(x,axis=0,weights=w);variance=np.average((x-mean)**2,axis=0,weights=w)
    scale=np.where(variance<1e-12,1,np.sqrt(variance))
    active=np.flatnonzero(variance>=1e-12).tolist()
    if not active:raise ValueError('no varying training features')
    return dict(medians=medians,mean=mean.tolist(),scale=scale.tolist(),registry_hash=registry,
        distance_scale=distance_scale,feature_indices=active,preprocessing_version='training_scaled_active_v2')


def input_width(pre):
    return len(pre.get('feature_indices',pre['mean']))


def make_network(p,linear=False):
    from torch import nn
    if linear:
        layer=nn.Linear(p,1)
        nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        return nn.Sequential(layer)
    return nn.Sequential(nn.Linear(p,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))


def export_network(model,pre,algorithm,provenance=None,output_transform='identity'):
    from torch import nn
    layers=[{'weight':m.weight.detach().cpu().numpy().tolist(),'bias':m.bias.detach().cpu().numpy().tolist()}
            for m in model if isinstance(m,nn.Linear)]
    return seal_artifact(dict(schema='scene_policy_v2' if 'feature_indices' in pre else 'scene_policy_v1',algorithm=algorithm,**pre,
        model={'kind':'dense','layers':layers,'output_transform':output_transform},provenance=provenance or {}))


def tensor(context,pre):
    import torch
    x=context_matrix(context,pre['medians'],pre.get('distance_scale'))
    x=(x-np.asarray(pre['mean']))/np.asarray(pre['scale'])
    if 'feature_indices' in pre:x=x[:,pre['feature_indices']]
    return torch.as_tensor(x,dtype=torch.float32)


def sampled_pairs(example,limit=32,seed=1729):
    """Uniform bounded sampling after canonicalization, independent of queue order."""
    gains=example.gains
    pairs=example.preferences or tuple((i,j,1.) for i in range(len(gains))
                                      for j in range(len(gains)) if gains[i]>gains[j])
    rows=example.context.ready_candidates
    pairs=sorted(pairs,key=lambda p:(rows[p[0]].message_id,rows[p[1]].message_id,p[2]))
    if len(pairs)>limit:
        identity=canonical({'seed':seed,'episode':example.episode_id,'ids':sorted(c.message_id for c in rows)})
        qualified=int(hashlib.sha256(identity.encode()).hexdigest()[:16],16)
        indices=np.random.default_rng(qualified).choice(len(pairs),limit,replace=False)
        pairs=[pairs[int(i)] for i in sorted(indices)]
    return tuple(pairs)


def loss_for(scores,example,kind,positive_weight=None,pair_seed=1729):
    import torch
    from torch.nn import functional as F
    gain=torch.as_tensor(example.gains,dtype=scores.dtype)
    if gain.shape!=scores.shape or not torch.isfinite(gain).all() or torch.any(gain<0):
        raise ValueError('invalid candidate gains')
    if kind=='priority_classifier':
        labels=(gain>0).to(scores.dtype)
        return F.binary_cross_entropy_with_logits(scores,labels,pos_weight=positive_weight)
    if kind in ('ranknet','visionprefer'):
        pairs=sampled_pairs(example,seed=pair_seed) if kind!='visionprefer' or example.preferences else ()
        if not pairs:return None
        return torch.stack([F.binary_cross_entropy_with_logits(scores[i]-scores[j],
                            scores.new_tensor(y)) for i,j,y in pairs]).mean()
    if len(gain)<2 or gain.max()<=0:return None
    target=(gain==gain.max()).to(scores.dtype);target=target/target.sum()
    return -(target*F.log_softmax(scores,dim=0)).sum()


def fit_supervised(examples,kind,*,epochs=50,seed=1729,validation_score=None,fixture=False,
                   checkpoint=None,save_progress=None,deadline=None):
    import torch
    if kind not in ('linear_choice','mlp_choice','ranknet','priority_classifier','visionprefer','lambdamart'):
        raise ValueError('unknown supervised arm')
    if kind=='visionprefer' and not fixture and any(e.label_origin!='qwen_aspect_preference' for e in examples):
        raise ValueError('VisionPrefer adaptation requires aspect-derived teacher preferences')
    if kind=='visionprefer' and not any(e.preferences for e in examples):
        raise ValueError('VisionPrefer requires explicit teacher pairs; annotation fallback forbidden')
    seed_everything(seed);pre=preprocessing(examples)
    if kind=='lambdamart':return fit_lambdamart(examples,pre,seed)
    model=make_network(input_width(pre),kind=='linear_choice')
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    counts=Counter(e.episode_id for e in examples)
    rng=np.random.default_rng(seed);history=[];best=-float('inf');best_state=None;stale=0
    start_epoch=0
    if checkpoint:
        if checkpoint['preprocessing']!=pre or checkpoint['kind']!=kind:
            raise ValueError('checkpoint preprocessing or algorithm mismatch')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['torch_rng']);rng.bit_generator.state=checkpoint['numpy_rng']
        start_epoch=checkpoint['epoch'];history=list(checkpoint['history'])
        best=checkpoint['best'];best_state=checkpoint['best_state'];stale=checkpoint['stale']
    positive=sum(g>0 for e in examples for g in e.gains)
    negative=sum(g==0 for e in examples for g in e.gains)
    positive_weight=torch.tensor(negative/positive if positive and negative else 1.,dtype=torch.float32)
    state=None
    for epoch in range(start_epoch,epochs):
        optimizer.zero_grad();losses=[];updates=0
        order=rng.permutation(len(examples))
        for start in range(0,len(order),64):
            weighted=[]
            for i in order[start:start+64]:
                e=examples[i];scores=model(tensor(e.context,pre)).flatten()
                loss=loss_for(scores,e,kind,positive_weight,pair_seed=seed)
                if loss is not None:weighted.append((loss/counts[e.episode_id],1/counts[e.episode_id]))
            if not weighted:continue
            value=torch.stack([row[0] for row in weighted]).sum()/sum(row[1] for row in weighted)
            optimizer.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step();losses.append(float(value.detach()));updates+=1
        if not updates:raise ValueError('no informative training targets')
        artifact=export_network(model,pre,kind,{'seed':seed,'fixture':fixture})
        score=float(validation_score(ScenePolicy(artifact))) if validation_score else -float(np.mean(losses))
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation_score':score})
        if score>best+1e-12:best=score;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        state={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
            'preprocessing':pre,'torch_rng':torch.get_rng_state(),'numpy_rng':copy.deepcopy(rng.bit_generator.state),
            'epoch':epoch+1,'kind':kind,'history':list(history),'best':best,
            'best_state':copy.deepcopy(best_state),'stale':stale}
        if save_progress:save_progress(state,history)
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('supervised training deadline; checkpoint preserved')
        if stale>=8:break
    if best_state is None:raise ValueError('no completed training epoch')
    model.load_state_dict(best_state)
    artifact=export_network(model,pre,kind,{'seed':seed,'fixture':fixture,'history':history})
    # Boundary parity check uses the actual training matrix, not only zeros.
    with torch.no_grad():expected=model(tensor(examples[0].context,pre)).flatten().numpy()
    np.testing.assert_allclose(ScenePolicy(artifact).scores(examples[0].context),expected,rtol=1e-5,atol=1e-6)
    return artifact,history,state or checkpoint


def fit_lambdamart(examples,pre,seed):
    import lightgbm as lgb
    xs=[];ys=[];groups=[]
    for e in examples:
        if len(e.gains)<2 or len(set(e.gains))<2:continue
        if any(g<0 or int(g)!=g for g in e.gains):raise ValueError('LambdaMART requires integer gain grades')
        xs.append(tensor(e.context,pre).numpy());ys.extend(e.gains);groups.append(len(e.gains))
    if not xs:raise ValueError('no graded ranking groups')
    x=np.concatenate(xs);grade=np.asarray(ys,int)
    model=lgb.LGBMRanker(objective='lambdarank',n_estimators=100,learning_rate=.05,max_depth=3,
        num_leaves=7,min_child_samples=10,label_gain=list(range(int(grade.max())+1)),
        random_state=seed,n_jobs=1,verbosity=-1).fit(x,grade,group=groups)
    trees=[t['tree_structure'] for t in model.booster_.dump_model()['tree_info']]
    artifact=seal_artifact(dict(schema='scene_policy_v2',algorithm='lambdamart',**pre,
        model={'kind':'trees','trees':trees},provenance={'seed':seed,'label_gain':'linear','groups':groups}))
    # Check all informative groups, including when the first example was skipped.
    active=pre.get('feature_indices',list(range(len(pre['mean']))))
    raw=np.tile(pre['mean'],(len(x),1))
    raw[:,active]=x*np.asarray(pre['scale'])[active]+np.asarray(pre['mean'])[active]
    actual=ScenePolicy(artifact)._predict(raw)
    np.testing.assert_allclose(actual,model.booster_.predict(x),rtol=1e-6,atol=1e-7)
    return artifact,[],{'booster_text':model.booster_.model_to_string(),'preprocessing':pre}


class Replay:
    """Uniform/PER/semantic mixtures with actual-mixture importance correction."""
    def __init__(self,capacity=50000,seed=0):
        self.capacity=capacity;self.rows=[];self.td=[];self.semantic=[];self.cursor=0
        self.rng=np.random.default_rng(seed)
    def add(self,transition,semantic=1.):
        priority=max(self.td,default=1.)
        if len(self.rows)<self.capacity:
            self.rows.append(transition);self.td.append(priority);self.semantic.append(semantic)
        else:
            self.rows[self.cursor]=transition;self.td[self.cursor]=priority;self.semantic[self.cursor]=semantic
        self.cursor=(self.cursor+1)%self.capacity
    def probabilities(self,mode):
        n=len(self.rows)
        if n==0:raise ValueError('empty replay')
        if mode=='uniform':return np.full(n,1/n)
        p=(np.abs(self.td)+1e-6)**.6
        if mode=='vlm':
            if not np.isfinite(self.semantic).all() or np.any(np.asarray(self.semantic)<0):raise ValueError('invalid VLM priority')
            p=p*np.clip(self.semantic,.1,3.)
        p=p/p.sum()
        return .5/n+.5*p if mode=='vlm' else p
    def sample(self,batch,mode,beta):
        p=self.probabilities(mode);indices=self.rng.choice(len(p),size=batch,p=p)
        # Normalize by the entire distribution, not a random batch maximum.
        # At beta=1 this preserves uniform expectation up to one fixed scale.
        w=(len(p)*p[indices])**(-beta)/((len(p)*p.min())**(-beta))
        return indices,[self.rows[i] for i in indices],w


def candidate_q(model,context,pre,value_mode='unbounded'):
    import torch
    values=model(tensor(context,pre)).flatten()
    if value_mode=='bounded_nonnegative_v1':
        return max(0.,1-context.now/context.horizon)*torch.sigmoid(values)
    if value_mode!='unbounded':raise ValueError('unknown Q value mode')
    return values


def dqn_target(transition,online,target,pre,reward=None,value_mode='unbounded'):
    import torch
    r=transition.reward if reward is None else reward(transition)
    with torch.no_grad():
        result=torch.tensor(r,dtype=torch.float32)
        if not torch.isfinite(result):raise ValueError('nonfinite training reward')
        if not transition.terminal:
            if not transition.next_observation.ready_candidates:raise ValueError('nonterminal transition without legal actions')
            action=candidate_q(online,transition.next_observation,pre,value_mode).argmax()
            result=result+candidate_q(target,transition.next_observation,pre,value_mode)[action]
        if value_mode=='bounded_nonnegative_v1':
            bound=max(0.,1-transition.observation.now/transition.observation.horizon)
            if not 0<=r<=bound+1e-6:raise ValueError('bounded Q requires the declared nonnegative coverage reward')
            result=result.clamp(0,bound)
        return result


def train_dqn(env_factory,examples,*,algorithm='masked_double_dqn',steps=20000,seed=1729,
              reward_model=None,semantic_scorer=None,validation_score=None,checkpoint=None,
              save_progress=None,deadline=None,schedule_steps=None,experiment_id=None,feedback_version=None,
               allow_feedback_update=False,value_mode='unbounded',validation_patience=None,
               diagnostic_score=None,eval_interval=2000):
    import torch
    from torch.nn import functional as F
    if algorithm=='rl_vlm_f' and reward_model is None:raise ValueError('learned reward required')
    if algorithm=='vlm_replay' and semantic_scorer is None:raise ValueError('registered VLM feedback required')
    if algorithm in TEACHER_ARMS and not feedback_version:raise ValueError('teacher feedback version required')
    if value_mode not in ('unbounded','bounded_nonnegative_v1'):raise ValueError('unknown Q value mode')
    if reward_model is not None and value_mode!='unbounded':raise ValueError('learned teacher reward cannot use annotation Q bounds')
    if validation_patience is not None and (type(validation_patience) is not int or validation_patience<1):
        raise ValueError('validation patience must be positive')
    if type(eval_interval) is not int or eval_interval<1:raise ValueError('positive evaluation interval required')
    schedule_steps=schedule_steps or steps
    if steps>schedule_steps:raise ValueError('training exceeds declared exploration schedule')
    experiment_signature=dict(training_identity(examples,experiment_id),algorithm=algorithm,
                  schedule_steps=schedule_steps,feedback_version=feedback_version,seed=seed,
                  value_mode=value_mode,validation_patience=validation_patience,eval_interval=eval_interval)
    seed_everything(seed);pre=preprocessing(examples);p=input_width(pre)
    online=make_network(p);target=copy.deepcopy(online)
    optimizer=torch.optim.Adam(online.parameters(),lr=3e-4)
    replay=Replay(seed=seed);rng=np.random.default_rng(seed);episode_index=0
    env=env_factory(episode_index);episode_actions=[];start_step=0
    if checkpoint:
        previous_identity=copy.deepcopy(checkpoint.get('identity'))
        if allow_feedback_update and algorithm in TEACHER_ARMS and previous_identity:
            previous_identity['feedback_version']=feedback_version
        if previous_identity!=experiment_signature:raise ValueError('checkpoint experiment/algorithm/schedule identity mismatch')
        if checkpoint['preprocessing']!=pre:raise ValueError('checkpoint preprocessing mismatch')
        online.load_state_dict(checkpoint['online']);target.load_state_dict(checkpoint['target'])
        optimizer.load_state_dict(checkpoint['optimizer']);replay=checkpoint['replay']
        if semantic_scorer is not None:
            replay.semantic=[semantic_scorer(t) for t in replay.rows]
        rng.bit_generator.state=checkpoint['rng'];torch.set_rng_state(checkpoint['torch_rng'])
        episode_index=checkpoint['episode_index'];env=env_factory(episode_index)
        episode_actions=list(checkpoint['episode_actions'])
        for identity in episode_actions:env.step(identity)
        start_step=checkpoint['step']
    losses=[];learning=list(checkpoint.get('learning',[])) if checkpoint else []
    best_artifact=checkpoint.get('best_artifact') if checkpoint else None
    best=checkpoint.get('best',-float('inf')) if checkpoint else -float('inf')
    stale=checkpoint.get('stale',0) if checkpoint else 0
    if checkpoint and validation_patience is not None and stale>=validation_patience:
        return best_artifact,learning,{**checkpoint,'stop_reason':'validation_patience'}
    diagnostics=[];actual_step=start_step
    transform='remaining_sigmoid' if value_mode=='bounded_nonnegative_v1' else 'identity'
    mode='vlm' if algorithm=='vlm_replay' else ('per' if algorithm=='td_per' else 'uniform')
    for step in range(start_step,steps):
        if env.finished:
            episode_index+=1;env=env_factory(episode_index);episode_actions=[]
        context=env.observation();x=tensor(context,pre)
        epsilon=1-.95*min(1,step/max(1,schedule_steps/2))
        if rng.random()<epsilon:index=int(rng.integers(len(x)))
        else:
            with torch.no_grad():index=int(candidate_q(online,context,pre,value_mode).argmax())
        identity=context.ready_candidates[index].message_id
        t=env.step(identity);episode_actions.append(identity)
        replay.add(t,semantic_scorer(t) if semantic_scorer else 1.)
        if len(replay.rows)>=min(500,max(4,schedule_steps//4)) and step%4==0:
            indices,rows,weights=replay.sample(min(64,len(replay.rows)),mode,.4+.6*step/max(1,schedule_steps))
            predictions=torch.stack([candidate_q(online,t.observation,pre,value_mode)[t.action_index] for t in rows])
            targets=torch.stack([dqn_target(t,online,target,pre,reward_model,value_mode) for t in rows])
            errors=predictions-targets
            loss=(F.smooth_l1_loss(predictions,targets,reduction='none')*torch.tensor(weights,dtype=torch.float32)).mean()
            optimizer.zero_grad();loss.backward();grad=torch.nn.utils.clip_grad_norm_(online.parameters(),10);optimizer.step()
            for i,error in zip(indices,errors.detach().abs().numpy()):replay.td[i]=float(error)
            losses.append(float(loss.detach()))
            q=predictions.detach().numpy();ys=targets.detach().numpy()
            bounds=np.asarray([max(0.,1-t.observation.now/t.observation.horizon) for t in rows])
            diagnostics.append({'q_mean':float(q.mean()),'q_max':float(q.max()),
                'q_bound_violation_fraction':float(np.mean((q<0)|(q>bounds+1e-6))),
                'target_mean':float(ys.mean()),'target_max':float(ys.max()),
                'unweighted_td_mse':float(np.mean((q-ys)**2)),
                'mean_importance_weight':float(np.mean(weights)),'gradient_norm':float(grad)})
            diagnostics=diagnostics[-100:]
        actual_step=step+1
        if (step+1)%500==0:target.load_state_dict(online.state_dict())
        if (step+1)%eval_interval==0 or step+1==steps:
            artifact=export_network(online,pre,algorithm,{'seed':seed,'steps':step+1,'training_identity':experiment_signature},transform)
            value=float(validation_score(ScenePolicy(artifact))) if validation_score else 0.
            entry={'step':step+1,'validation_J':value,'loss':float(np.mean(losses[-100:])) if losses else None,
                'train_probe_J':float(diagnostic_score(ScenePolicy(artifact))) if diagnostic_score else None,
                'diagnostics':{k:float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]} if diagnostics else {}}
            learning.append(entry)
            if value>best:best=value;best_artifact=artifact;stale=0
            else:stale+=1
            if save_progress:
                save_progress({'online':online.state_dict(),'target':target.state_dict(),'optimizer':optimizer.state_dict(),
                    'replay':replay,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                    'episode_index':episode_index,'episode_actions':list(episode_actions),'step':step+1,
                    'preprocessing':pre,'identity':experiment_signature,'best':best,'best_artifact':best_artifact,
                    'learning':list(learning),'stale':stale},learning)
            if validation_patience is not None and stale>=validation_patience:break
        if deadline is not None and time.monotonic()>=deadline:
            if save_progress:
                save_progress({'online':online.state_dict(),'target':target.state_dict(),'optimizer':optimizer.state_dict(),
                    'replay':replay,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                    'episode_index':episode_index,'episode_actions':list(episode_actions),'step':step+1,
                    'preprocessing':pre,'identity':experiment_signature,'best':best,'best_artifact':best_artifact,
                    'learning':list(learning),'stale':stale},learning)
            raise TimeoutError('DQN training deadline; checkpoint preserved')
    state={'online':online.state_dict(),'target':target.state_dict(),'optimizer':optimizer.state_dict(),
           'replay':replay,'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
            'episode_index':episode_index,'episode_actions':episode_actions,'step':actual_step,'preprocessing':pre,
            'identity':experiment_signature,'best':best,'best_artifact':best_artifact,'learning':learning,'stale':stale,
            'stop_reason':'validation_patience' if actual_step<steps else 'decision_budget'}
    return best_artifact,learning,state


def train_ppo(env_factory,examples,*,steps=20000,seed=1729,validation_score=None,checkpoint=None,
              save_progress=None,deadline=None,experiment_id=None,rollout_steps=256,
              validation_patience=None,diagnostic_score=None,eval_interval=2000):
    import torch
    from torch.distributions import Categorical
    seed_everything(seed);pre=preprocessing(examples);p=input_width(pre)
    if type(rollout_steps) is not int or rollout_steps<4:raise ValueError('rollout needs at least four transitions')
    if validation_patience is not None and (type(validation_patience) is not int or validation_patience<1):
        raise ValueError('validation patience must be positive')
    if type(eval_interval) is not int or eval_interval<1:raise ValueError('positive evaluation interval required')
    experiment_signature=dict(training_identity(examples,experiment_id),algorithm='masked_ppo',seed=seed,
        rollout_steps=rollout_steps,validation_patience=validation_patience,eval_interval=eval_interval)
    actor=make_network(p);critic=make_network(p)
    optimizer=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=3e-4)
    episode_ids=list(range(4));actions=[[] for _ in range(4)]
    environments=[env_factory(i) for i in episode_ids];next_episode=4;done_steps=0;learning=[]
    if checkpoint:
        if checkpoint.get('identity')!=experiment_signature:raise ValueError('checkpoint experiment identity mismatch')
        if checkpoint['preprocessing']!=pre:raise ValueError('checkpoint preprocessing mismatch')
        actor.load_state_dict(checkpoint['actor']);critic.load_state_dict(checkpoint['critic'])
        optimizer.load_state_dict(checkpoint['optimizer']);done_steps=checkpoint['step']
        next_episode=checkpoint['next_episode'];episode_ids=checkpoint['episode_ids'];actions=checkpoint['actions']
        environments=[env_factory(i) for i in episode_ids]
        for env,ids in zip(environments,actions):
            for identity in ids:env.step(identity)
        torch.set_rng_state(checkpoint['torch_rng'])
    best=checkpoint.get('best',-float('inf')) if checkpoint else -float('inf')
    best_artifact=checkpoint.get('best_artifact') if checkpoint else None
    stale=checkpoint.get('stale',0) if checkpoint else 0
    if checkpoint:learning=list(checkpoint.get('learning',[]))
    state=checkpoint
    if checkpoint and validation_patience is not None and stale>=validation_patience:
        return best_artifact,learning,{**checkpoint,'stop_reason':'validation_patience'}
    while done_steps<steps:
        rows=[]
        for k in range(min(rollout_steps,steps-done_steps)):
            slot=k%4;env=environments[slot]
            if env.finished:
                episode_ids[slot]=next_episode;env=env_factory(next_episode);next_episode+=1
                environments[slot]=env;actions[slot]=[]
            x=tensor(env.observation(),pre)
            with torch.no_grad():
                dist=Categorical(logits=actor(x).flatten());a=dist.sample();v=critic(x.mean(0,keepdim=True)).flatten()[0]
            identity=env.observation().ready_candidates[int(a)].message_id
            t=env.step(identity);actions[slot].append(identity)
            with torch.no_grad():
                nx=tensor(t.next_observation,pre)
                nv=0. if t.terminal else float(critic(nx.mean(0,keepdim=True)).flatten()[0])
            rows.append({'slot':slot,'x':x,'a':int(a),'old':dist.log_prob(a).detach(),
                         'v':float(v),'nv':nv,'r':t.reward,'done':t.terminal})
        carry={i:0. for i in range(4)}
        for row in reversed(rows):
            delta=row['r']+row['nv']-row['v']
            gae=delta+(.95*carry[row['slot']] if not row['done'] else 0.)
            row['adv']=gae;row['return']=gae+row['v'];carry[row['slot']]=gae
        adv=np.asarray([r['adv'] for r in rows]);adv=(adv-adv.mean())/(adv.std()+1e-8)
        for r,a in zip(rows,adv):r['adv']=float(a)
        diagnostics=[]
        for _ in range(4):
            order=torch.randperm(len(rows)).tolist()
            for start in range(0,len(rows),64):
                losses=[]
                for index in order[start:start+64]:
                    row=rows[index]
                    dist=Categorical(logits=actor(row['x']).flatten())
                    logratio=dist.log_prob(torch.tensor(row['a']))-row['old'];ratio=logratio.exp()
                    policy=-torch.minimum(ratio*row['adv'],ratio.clamp(.8,1.2)*row['adv'])
                    value=critic(row['x'].mean(0,keepdim=True)).flatten()[0]
                    losses.append(policy+.5*(value-row['return'])**2-.01*dist.entropy())
                    diagnostics.append({'entropy':float(dist.entropy().detach()),
                        'approx_kl':float(((ratio-1)-logratio).detach()),
                        'clip_fraction':float((torch.abs(ratio-1)>.2).detach()),
                        'value_mse':float(((value-row['return'])**2).detach())})
                loss=torch.stack(losses).mean();optimizer.zero_grad();loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(list(actor.parameters())+list(critic.parameters()),.5);optimizer.step()
        done_steps+=len(rows)
        artifact=export_network(actor,pre,'masked_ppo',{'seed':seed,'steps':done_steps,'training_identity':experiment_signature})
        validate=done_steps==steps or done_steps//eval_interval>(done_steps-len(rows))//eval_interval
        value=float(validation_score(ScenePolicy(artifact))) if validation_score and validate else None
        with torch.no_grad():
            targets=np.asarray([r['return'] for r in rows]);predicted=np.asarray([
                float(critic(r['x'].mean(0,keepdim=True)).flatten()[0]) for r in rows])
        diagnostic={k:float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]}
        diagnostic['gradient_norm_last_update']=float(grad)
        diagnostic['value_explained_variance']=float(1-np.var(targets-predicted)/np.var(targets)) if np.var(targets)>1e-12 else None
        learning.append({'step':done_steps,'validation_J':value,'loss':float(loss.detach()),'diagnostics':diagnostic,
            'train_probe_J':float(diagnostic_score(ScenePolicy(artifact))) if diagnostic_score and validate else None})
        if value is not None:
            if value>best:best=value;best_artifact=artifact;stale=0
            else:stale+=1
        if not validation_score:best_artifact=artifact
        state={'actor':actor.state_dict(),'critic':critic.state_dict(),'optimizer':optimizer.state_dict(),
            'preprocessing':pre,'step':done_steps,'torch_rng':torch.get_rng_state(),
            'episode_ids':list(episode_ids),'actions':copy.deepcopy(actions),'next_episode':next_episode,
            'identity':experiment_signature,'best':best,'best_artifact':best_artifact,'learning':list(learning),'stale':stale,
            'stop_reason':'decision_budget' if done_steps==steps else 'training'}
        if save_progress:save_progress(state,learning)
        if validation_patience is not None and stale>=validation_patience:
            state['stop_reason']='validation_patience';break
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('PPO training deadline; checkpoint preserved')
    return best_artifact,learning,state
