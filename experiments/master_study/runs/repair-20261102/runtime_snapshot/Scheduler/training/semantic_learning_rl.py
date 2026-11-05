"""Masked episodic PPO and Double-DQN variants for the exact D environment."""
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from Evaluation.semantic_learning_env import Environment
from Scheduler.semantic_learning_policy import digest
from Scheduler.training.semantic_learning_models import Scorer,tensors,seed_all,export,parity


def train_dqn(factory,pre,output,validate,*,arm='masked_double_dqn',steps=20000,seed=1729,reward_model=None,guidance=None,feedback_version=None,seconds=None,device='cpu',resume=None,training_id=None):
    if arm not in ('masked_double_dqn','td_per','rl_vlm_f','vlm_replay'):raise ValueError('unknown Q variant')
    if arm=='rl_vlm_f' and (reward_model is None or not feedback_version):raise ValueError('qualified frozen preference reward required')
    if arm=='vlm_replay' and (guidance is None or not feedback_version):raise ValueError('qualified replay teacher guidance required')
    seed_all(seed);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    online=Scorer('deepsets').to(device);target=copy.deepcopy(online);opt=torch.optim.Adam(online.parameters(),lr=3e-4)
    replay=[];priorities=[];guides=[];rng=np.random.default_rng(seed);ep=0;env=factory(ep);history=[];updates=0
    best=-float('inf');best_artifact=None;start=time.perf_counter();actions=[];done_steps=0
    exploration_steps=max(1,steps//2);schedule_steps=steps
    signature=digest({'arm':arm,'pre':pre,'seed':seed,'feedback_version':feedback_version,'training_id':training_id})
    if resume:
        if not training_id:raise ValueError('resume requires explicit immutable training stream identity')
        saved=torch.load(resume,map_location=device,weights_only=False)
        if saved.get('signature')!=signature:raise ValueError('Q resume identity mismatch')
        exploration_steps=saved.get('exploration_steps',10000);schedule_steps=saved.get('schedule_steps',steps)
        online.load_state_dict(saved['online']);target.load_state_dict(saved['target']);opt.load_state_dict(saved['optimizer'])
        replay=saved['replay'];priorities=saved['priorities'];guides=saved['guidance'];ep=saved['episode_number']
        env=Environment(saved['episode'],saved.get('objective','D'));actions=list(saved['episode_actions'])
        for token in actions:env.step(token)
        rng.bit_generator.state=saved['rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        if device.type=='cuda' and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        done_steps=saved['step'];updates=saved['updates'];history=saved['history'];best_artifact=saved['best_artifact'];best=saved['best']
    for step in range(done_steps,steps):
        if seconds is not None and time.perf_counter()-start>=seconds:break
        if env.done:ep+=1;env=factory(ep);actions=[]
        s=env.snapshot();a=tensors(s,pre,device=device,kind='deepsets')
        epsilon=1-.95*min(1,step/exploration_steps)
        with torch.no_grad():q=online(a)
        index=int(rng.integers(len(s.ready))) if rng.random()<epsilon else int(q.argmax())
        token=s.ready[index].token;nxt,r,terminal=env.step(token)
        if reward_model is not None:r=float(reward_model(s,token,nxt,env))
        if not np.isfinite(r):raise ValueError('nonfinite reward')
        g=1. if guidance is None else float(guidance(s,token,nxt))
        if g not in (.5,1.,1.5):raise ValueError('invalid bounded teacher replay multiplier')
        path=reward_model.path(s,token,nxt,env) if reward_model is not None and hasattr(reward_model,'path') else None
        replay.append((s,index,r,nxt,terminal,path));priorities.append(max(priorities,default=1.));guides.append(g)
        if len(replay)>50000:replay.pop(0);priorities.pop(0);guides.pop(0)
        actions.append(token);done_steps=step+1
        if len(replay)>=500 and step%4==0:
            n=len(replay);p=np.asarray(priorities,dtype=float)
            if arm=='masked_double_dqn' or arm=='rl_vlm_f':prob=np.full(n,1/n)
            elif arm=='td_per':prob=p/p.sum()
            else:
                z=p*np.asarray(guides);prob=.5/n+.5*z/z.sum()
            ids=rng.choice(n,size=64,replace=True,p=prob);beta=.4+.6*min(1,step/max(1,schedule_steps))
            weights=(n*prob[ids])**(-beta);weights/=weights.max();losses=[];tds=[]
            chosen=[replay[j] for j in ids]
            current=online.forward_batch([tensors(row[0],pre,device=device,kind='deepsets') for row in chosen])
            pred=torch.stack([scores[row[1]] for scores,row in zip(current,chosen)])
            with torch.no_grad():
                value=torch.tensor([row[2] for row in chosen],dtype=torch.float32,device=device)
                live=[k for k,row in enumerate(chosen) if not row[4]]
                if live:
                    next_inputs=[tensors(chosen[k][3],pre,device=device,kind='deepsets') for k in live]
                    next_online=online.forward_batch(next_inputs);next_target=target.forward_batch(next_inputs)
                    value[torch.tensor(live,device=device)]+=torch.stack([q[int(o.argmax())] for o,q in zip(next_online,next_target)])
            delta=pred-value;tds=delta.detach().abs().cpu().tolist()
            losses=F.huber_loss(pred,value,delta=1.,reduction='none')*torch.as_tensor(weights,device=device,dtype=pred.dtype)
            opt.zero_grad();loss=losses.mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite TD loss')
            loss.backward();nn.utils.clip_grad_norm_(online.parameters(),10.);opt.step();updates+=1
            for j,td in zip(ids,tds):priorities[j]=(td+1e-6)**.6
        if (step+1)%500==0:target.load_state_dict(online.state_dict())
        if (step+1)%2000==0 or step==steps-1 or (seconds is not None and time.perf_counter()-start>=seconds):
            artifact=export(online,pre,arm,{'task':env.task if reward_model is None else 'P','reward':'reference_'+env.task if reward_model is None else feedback_version,'seed':seed,'steps':step+1})
            val,details=validate(artifact)
            if updates and val>best:best=val;best_artifact=artifact
            history.append({'steps':step+1,'updates':updates,'validation':val,'seconds':time.perf_counter()-start})
            state={'online':online.state_dict(),'target':target.state_dict(),'optimizer':opt.state_dict(),'replay':replay,'priorities':priorities,'guidance':guides,
                'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                'signature':signature,'exploration_steps':exploration_steps,'schedule_steps':schedule_steps,'best':best,'episode_number':ep,'objective':getattr(env,'objective',getattr(env,'task','D')),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'step':step+1,'episode':env.episode,'episode_actions':actions,'preprocessing':pre,'feedback_version':feedback_version,'arm':arm,
                'history':history,'updates':updates,'best_artifact':best_artifact}
            torch.save(state,out/'resume.pt');(out/'learning.json').write_text(json.dumps(history,indent=2))
            print('RL',arm,seed,step+1,'updates',updates,'validation',val,flush=True)
    if updates==0 or best_artifact is None:raise ValueError('no updated validated Q checkpoint; partial run not trained')
    restored=Scorer('deepsets').to(device)
    restored.load_state_dict({k:torch.as_tensor(v,device=device) for k,v in best_artifact['weights'].items()})
    check=parity(restored,best_artifact,[row[0] for row in replay[:3]])
    (out/'parity.json').write_text(json.dumps(check))
    (out/'model.json').write_text(json.dumps(best_artifact));return best_artifact,history


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__();self.actor=Scorer('deepsets');self.value=nn.Sequential(nn.Linear(1172,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,a):return self.actor(a),self.value(a['summary'].mean(0)).flatten()[0]
    def forward_batch(self,items):
        logits=self.actor.forward_batch(items)
        values=self.value(torch.stack([a['summary'].mean(0) for a in items])).flatten()
        return logits,values


def train_ppo(factory,pre,output,validate,*,steps=20000,seed=1729,warmstart=None,seconds=None,device='cpu',resume=None,training_id=None):
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    seed_all(seed);out=Path(output);out.mkdir(parents=True,exist_ok=True);model=ActorCritic().to(device)
    if warmstart is not None:model.actor.load_state_dict(warmstart)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4);envs=[factory(i) for i in range(4)];eps=list(range(4));episode_actions=[[] for _ in envs]
    history=[];count=0;updates=0;best=-float('inf');best_artifact=None;start=time.perf_counter()
    signature=digest({'arm':'ppo','pre':pre,'seed':seed,'warmstart_sha256':None if warmstart is None else digest({k:v.detach().cpu().numpy().tolist() for k,v in warmstart.items()}),'training_id':training_id})
    if resume:
        if not training_id:raise ValueError('resume requires explicit immutable training stream identity')
        saved=torch.load(resume,map_location=device,weights_only=False)
        if saved.get('signature')!=signature:raise ValueError('PPO resume identity mismatch')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        eps=saved['episode_numbers'];episode_actions=saved['episode_actions']
        envs=[Environment(e,task) for e,task in zip(saved['env_episodes'],saved.get('env_tasks',['D']*len(saved['env_episodes'])))]
        for env,actions in zip(envs,episode_actions):
            for token in actions:env.step(token)
        count=saved['count'];updates=saved['updates'];history=saved['history'];best_artifact=saved['best_artifact'];best=saved['best']
        torch.set_rng_state(saved['torch_rng'].cpu());np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
        if device.type=='cuda' and saved.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    rollouts=[[] for _ in envs]
    while count<steps:
        if seconds is not None and time.perf_counter()-start>=seconds:break
        rollouts=[[] for _ in envs]
        for t in range(min(256,steps-count)):
            k=t%4;env=envs[k]
            if env.done:eps[k]+=4;envs[k]=env=factory(eps[k]);episode_actions[k]=[]
            s=env.snapshot();a=tensors(s,pre,device=device,kind='deepsets')
            with torch.no_grad():
                logits,v=model(a);dist=torch.distributions.Categorical(logits=logits);act=dist.sample();lp=dist.log_prob(act)
            nxt,r,end=env.step(s.ready[int(act)].token);episode_actions[k].append(s.ready[int(act)].token)
            rollouts[k].append([s,int(act),float(lp),float(v),r,end]);count+=1
        data=[]
        for k,rows in enumerate(rollouts):
            if not rows:continue
            with torch.no_grad():next_v=0. if envs[k].done else float(model(tensors(envs[k].snapshot(),pre,device=device,kind='deepsets'))[1])
            advantage=0.;prepared=[]
            for s,act,lp,value,reward,end in reversed(rows):
                delta=reward+(0 if end else next_v)-value
                advantage=delta+(0 if end else .95*advantage);ret=advantage+value
                prepared.append((s,act,lp,advantage,ret));next_v=value
            data.extend(reversed(prepared))
        observations=[tensors(row[0],pre,device=device,kind='deepsets') for row in data]
        adv=np.array([r[3] for r in data]);adv=(adv-adv.mean())/(adv.std()+1e-8)
        for epoch in range(4):
            order=np.random.permutation(len(data))
            for offset in range(0,len(order),64):
                losses=[]
                ids=order[offset:offset+64]
                batch_logits,batch_values=model.forward_batch([observations[j] for j in ids])
                for j,logits,value in zip(ids,batch_logits,batch_values):
                    s,act,lp,_,ret=data[j]
                    dist=torch.distributions.Categorical(logits=logits);ratio=torch.exp(dist.log_prob(torch.tensor(act,device=device))-lp)
                    aa=float(adv[j]);actor=-torch.minimum(ratio*aa,torch.clamp(ratio,.8,1.2)*aa)
                    losses.append(actor+.5*(value-ret)**2-.01*dist.entropy())
                opt.zero_grad();loss=torch.stack(losses).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite PPO loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step();updates+=1
        if count%2000<256 or count==steps or (seconds is not None and time.perf_counter()-start>=seconds):
            artifact=export(model.actor,pre,'ranknet_warmstart_ppo' if warmstart is not None else 'masked_ppo',{'task':envs[0].task,'reward':'reference_'+envs[0].task,'seed':seed,'steps':count})
            val,details=validate(artifact)
            if val>best:best=val;best_artifact=artifact
            history.append({'steps':count,'updates':updates,'validation':val,'seconds':time.perf_counter()-start})
            torch.save({'signature':signature,'best':best,'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'model':model.state_dict(),'optimizer':opt.state_dict(),'rollouts':rollouts,'env_episodes':[e.episode for e in envs],'env_tasks':[e.task for e in envs],
                'episode_actions':episode_actions,'episode_numbers':eps,'count':count,'updates':updates,'torch_rng':torch.get_rng_state(),
                'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'preprocessing':pre,'history':history,'best_artifact':best_artifact},out/'resume.pt')
            (out/'learning.json').write_text(json.dumps(history,indent=2));print('PPO',count,'updates',updates,'validation',val,flush=True)
    if best_artifact is None:raise ValueError('no validated PPO checkpoint')
    restored=Scorer('deepsets').to(device)
    restored.load_state_dict({k:torch.as_tensor(v,device=device) for k,v in best_artifact['weights'].items()})
    probes=[row[0] for rows in rollouts for row in rows][:3] if count else []
    if not probes:probes=[Environment(env.episode,env.task).snapshot() for env in envs][:3]
    (out/'parity.json').write_text(json.dumps(parity(restored,best_artifact,probes)))
    (out/'model.json').write_text(json.dumps(best_artifact));return best_artifact,history
