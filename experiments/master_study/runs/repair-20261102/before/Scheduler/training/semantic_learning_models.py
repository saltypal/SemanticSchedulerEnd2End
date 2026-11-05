"""Torch training implementations; deployed inference stays in NumPy."""
from collections import Counter
from dataclasses import dataclass
import copy
import itertools
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from Scheduler.semantic_learning_policy import arrays,digest,NumpyRanker

ARMS=('linear_choice','mlp_choice','delay_huber','ranknet','lambdamart','priority_classifier',
'deepsets_ranknet','set_transformer_ranknet','listmle','masked_ppo','masked_double_dqn','td_per',
'ranknet_warmstart_ppo','rl_vlm_f','vlm_replay','visionprefer','temporal_deepsets_ranknet')
KINDS={'linear_choice':'linear','mlp_choice':'mlp','delay_huber':'mlp','ranknet':'mlp',
'priority_classifier':'classifier','deepsets_ranknet':'deepsets','set_transformer_ranknet':'attention',
'listmle':'deepsets','temporal_deepsets_ranknet':'temporal','visionprefer':'deepsets'}


def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def mlp(p,out):return nn.Sequential(nn.Linear(p,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,out))


class Scorer(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='linear':self.linear=nn.Linear(1172,1)
        elif kind in ('mlp','classifier'):self.head=mlp(1172,4 if kind=='classifier' else 1)
        elif kind=='attention':
            self.projection=nn.Linear(389,32)
            layer=nn.TransformerEncoderLayer(32,4,dim_feedforward=64,dropout=0.,batch_first=True)
            self.attention=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=mlp(426,1)
        elif kind in ('deepsets','temporal'):
            self.phi=nn.Sequential(nn.Linear(389,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
            if kind=='temporal':self.gru=nn.GRUCell(37,32)
            self.head=mlp(554 if kind=='temporal' else 490,1)
        else:raise ValueError('unknown scorer architecture')

    def history(self,tokens,mask):
        e=self.phi(tokens);h=torch.zeros((len(tokens),32),device=tokens.device)
        for j in range(3):
            status=tokens[:,j,388:389]
            z=torch.cat([e[:,j],tokens[:,j,387:388],(status==0).float(),(status==1).float(),(status==2).float(),mask[:,j:j+1].float()],dim=1)
            proposed=self.gru(z,h);h=torch.where(mask[:,j:j+1],proposed,h)
        return h

    def forward_batch(self,items):
        """Vectorized independent Deep Sets queues without padding or cross-queue pooling."""
        if not items:return []
        if self.kind not in ('deepsets','linear','mlp','classifier'):return [self(a) for a in items]
        lengths=[len(a['ready']) for a in items]
        if self.kind in ('linear','mlp','classifier'):
            x=torch.cat([a['summary'] for a in items])
            y=self.linear(x).flatten() if self.kind=='linear' else self.head(x)
            if self.kind=='mlp':y=y.flatten()
            return list(y.split(lengths))
        r=torch.cat([a['ready'] for a in items]);h=torch.cat([a['processed'] for a in items])
        counts=torch.tensor(lengths,device=r.device)
        ids=torch.repeat_interleave(torch.arange(len(items),device=r.device),counts)
        er=self.phi(r);qm=torch.zeros((len(items),32),device=r.device,dtype=er.dtype).index_add(0,ids,er)/counts[:,None].clamp_min(1)
        hcounts=torch.tensor([len(a['processed']) for a in items],device=r.device)
        hids=torch.repeat_interleave(torch.arange(len(items),device=r.device),hcounts)
        eh=self.phi(h);gm=torch.zeros((len(items),32),device=r.device,dtype=er.dtype).index_add(0,hids,eh)/hcounts[:,None].clamp_min(1)
        sizes=torch.stack([a['sizes'] for a in items])[ids]
        distances=torch.cat([a['distances'] for a in items])
        scores=self.head(torch.cat([r,er,qm[ids],gm[ids],sizes,distances],dim=1)).flatten()
        return list(scores.split(lengths))

    def forward(self,a,pooled_ready=None):
        r,h=a['ready'],a['processed'];n=len(r);sizes=a['sizes'].expand(n,-1)
        if self.kind=='linear':return self.linear(a['summary']).flatten()
        if self.kind in ('mlp','classifier'):
            y=self.head(a['summary']);return y if self.kind=='classifier' else y.flatten()
        if self.kind=='attention':
            tokens=torch.cat([r,h]).unsqueeze(0);z=self.attention(self.projection(tokens))[0,:n]
            return self.head(torch.cat([r,z,sizes,a['distances']],dim=1)).flatten()
        er=self.phi(r);eh=self.phi(h);qm=er.mean(0) if pooled_ready is None else pooled_ready
        gm=eh.mean(0) if len(eh) else torch.zeros(32,device=r.device)
        inputs=torch.cat([r,er,qm.expand(n,-1),gm.expand(n,-1),sizes,a['distances']],dim=1)
        if self.kind=='temporal':
            hh=self.history(a['ue'],a['ue_mask']);u=self.history(a['active'],a['active_mask'])
            up=u.mean(0) if len(u) else torch.zeros(32,device=r.device)
            inputs=torch.cat([inputs,hh,up.expand(n,-1)],dim=1)
        return self.head(inputs).flatten()


def tensors(s,pre,device='cpu',kind=None):
    a=arrays(s,pre,kind is None or kind=='temporal')
    return {k:torch.as_tensor(v,device=device) for k,v in a.items()}


def export(model,pre,arm,source,*,class_weights=None):
    model.eval()
    artifact={'schema':'semantic_scene_ranker_v1','kind':model.kind,'arm':arm,'preprocessing':pre,
        'input_profile':pre['profile_identity'],'objective_version':source.get('task','D')+'_v1','reward_version':source.get('reward','none_supervised'),
        'weights':{k:v.detach().cpu().numpy().tolist() for k,v in model.state_dict().items()},
        'source':source,'class_weights':class_weights.tolist() if class_weights is not None else None}
    artifact['sha256']=digest(artifact);return artifact


def parity(model,artifact,snapshots,tol=3e-4):
    portable=NumpyRanker(artifact);errors=[]
    model.eval()
    with torch.no_grad():
        for s in snapshots:
            y=model(tensors(s,artifact['preprocessing'],next(model.parameters()).device,model.kind)).detach().cpu().numpy()
            z=portable.raw(s)
            if not np.allclose(y,z,rtol=tol,atol=tol):raise ValueError('Torch/NumPy raw prediction mismatch '+str(np.max(np.abs(y-z))))
            errors.append(float(np.max(np.abs(y-z))))
    return {'maximum_absolute_error':max(errors,default=0.),'snapshots':len(errors),'tolerance':tol,'raw_logits_not_calibration':True}


@dataclass
class Sample:
    snapshot:object
    target:object
    tiers:object
    episode_id:str
    pairs:tuple=()
    target_source:str='reference_deferral_tier'


def pairs(sample,seed=1729):
    if sample.pairs:return sample.pairs
    y=sample.target;strict=[];ties=[]
    for i,j in itertools.combinations(range(len(y)),2):
        v=1. if y[i]>y[j] else 0. if y[i]<y[j] else .5
        (ties if v==.5 else strict).append((i,j,v))
    rng=random.Random(seed);rng.shuffle(strict);rng.shuffle(ties)
    result=strict[:24]+ties[:8]
    return tuple(result)


def supervised_loss(scores,sample,arm,class_weights=None,seed=1729):
    if arm=='priority_classifier':
        labels=torch.as_tensor(sample.tiers,dtype=torch.long,device=scores.device)
        return F.cross_entropy(scores,labels,weight=class_weights)
    y=torch.as_tensor(sample.target,dtype=scores.dtype,device=scores.device)
    if arm=='delay_huber':return F.huber_loss(scores,y,delta=1.)
    if arm=='listmle':
        triples=[]
        # Bounded deterministic sampling; avoid materializing Q choose3.
        rng=np.random.default_rng(seed)
        for _ in range(64):
            if len(y)<3:break
            ids=rng.choice(len(y),3,replace=False).tolist()
            if len({float(y[i]) for i in ids})<3:continue
            ids=sorted(ids,key=lambda i:-float(y[i]));z=scores[ids]
            triples.append(-z[0]+torch.logsumexp(z,0)-z[1]+torch.logsumexp(z[1:],0))
            if len(triples)==8:break
        return torch.stack(triples).mean() if triples else None
    if arm in ('linear_choice','mlp_choice'):
        if sample.pairs:
            if len(scores)!=2 or len(sample.pairs)!=1:raise ValueError('P choice requires matched two-candidate view')
            t=sample.pairs[0][2];target=torch.tensor([t,1-t],device=scores.device,dtype=scores.dtype)
        else:
            if len(y)<2 or y.max()==y.min():return None
            target=(y==y.max()).float();target=target/target.sum()
        return -(target*F.log_softmax(scores,dim=0)).sum()
    ps=pairs(sample,seed)
    if not ps:return None
    i=torch.tensor([p[0] for p in ps],device=scores.device);j=torch.tensor([p[1] for p in ps],device=scores.device)
    t=torch.tensor([p[2] for p in ps],device=scores.device,dtype=scores.dtype)
    return F.binary_cross_entropy_with_logits(scores[i]-scores[j],t)


def fit_supervised(samples,pre,arm,output,validate,*,seed=1729,epochs=50,patience=8,seconds=None,resume=None,task='D',device='cpu'):
    if arm not in KINDS:raise ValueError('not a supervised architecture')
    if not samples:raise ValueError('no supported training samples')
    seed_all(seed);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    model=Scorer(KINDS[arm]).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-4)
    cw=None
    if arm=='priority_classifier':
        counts=np.bincount(np.concatenate([s.tiers for s in samples]),minlength=4)
        if np.any(counts==0):raise ValueError('four-class training support missing')
        w=1/counts;w=w/w.mean();cw=torch.as_tensor(w,dtype=torch.float32,device=device)
    identity=hashlib.sha256()
    for sample in samples:
        snap=sample.snapshot
        identity.update(json.dumps([sample.episode_id,snap.now,snap.attempts,sample.target_source,sample.pairs,sorted(snap.waiting_tokens),list(snap.in_flight_tokens)],sort_keys=True).encode())
        identity.update(np.asarray(sample.target,dtype='<f8').tobytes());identity.update(np.asarray(sample.tiers,dtype='<i8').tobytes())
        groups=(snap.ready,tuple(m for m,t in snap.processed))+snap.ue_windows+snap.active_windows
        identity.update(json.dumps([t for m,t in snap.processed]).encode())
        for group in groups:
            identity.update(json.dumps([(m.token,m.source,m.session,m.arrival,m.sequence,m.profile_id) for m in group]).encode())
            for m in group:identity.update(np.asarray(m.features,dtype='<f4').tobytes())
    signature=digest({'arm':arm,'pre':pre,'seed':seed,'records_sha256':identity.hexdigest(),'task':task})
    history=[];best=-float('inf');best_state=None;stale=0;start_epoch=0;updates=0
    if resume:
        state=torch.load(resume,map_location=device,weights_only=False)
        if state['signature']!=signature:raise ValueError('resume signature mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        updates=state['updates'];history=state['history'];best=state['best'];best_state=state['best_state'];stale=state['stale'];start_epoch=state['epoch']+1
        torch.set_rng_state(state['torch_rng'].cpu());
        if device.type=='cuda' and state.get('cuda_rng') is not None:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    counts=Counter(s.episode_id for s in samples);weights=[len(samples)/(len(counts)*counts[s.episode_id]) for s in samples]
    # Cache only derived runtime observations; targets stay in separate Sample fields.
    data=[tensors(s.snapshot,pre,device=device,kind=model.kind) for s in samples]
    started=time.perf_counter();stop='max_epochs'
    for epoch in range(start_epoch,epochs):
        if seconds is not None and time.perf_counter()-started>=seconds:stop='partial_budget';break
        model.train();order=np.random.permutation(len(samples));losses=[]
        for offset in range(0,len(order),64):
            ids=order[offset:offset+64];batch=[]
            batch_scores=model.forward_batch([data[j] for j in ids])
            for j,scores in zip(ids,batch_scores):
                loss=supervised_loss(scores,samples[j],arm,cw,seed+int(j))
                if loss is not None:batch.append(loss*weights[j]);losses.append(float(loss.detach()))
            if batch:
                optimizer.zero_grad();loss=torch.stack(batch).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite supervised loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite gradients')
                if any(p.grad is not None and torch.any(p.grad.abs()>1e-12) for p in model.parameters()):optimizer.step();updates+=1
        if updates==0:raise ValueError('zero meaningful optimizer updates; no trained export')
        artifact=export(model,pre,arm,{'task':task,'seed':seed,'epoch':epoch,'target_sources':sorted({s.target_source for s in samples})},class_weights=cw.detach().cpu().numpy() if cw is not None else None)
        value,details=validate(artifact)
        if value>best+1e-12:best=value;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        history.append({'epoch':epoch,'loss':float(np.mean(losses)) if losses else None,'validation_score':float(value),'validation':details,'seconds':time.perf_counter()-started})
        state={'signature':signature,'updates':updates,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'best':best,'best_state':best_state,'stale':stale,'history':history,'torch_rng':torch.get_rng_state(),
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None,'device':str(device),'preprocessing':pre}
        torch.save(state,output/'resume.pt');(output/'learning.json').write_text(json.dumps(history,indent=2,allow_nan=False))
        print('EPOCH',arm,seed,epoch,'loss',history[-1]['loss'],'validation',value,flush=True)
        if stale>=patience:stop='validation_patience';break
    if best_state is None:raise ValueError('no validated optimization checkpoint')
    model.load_state_dict(best_state)
    artifact=export(model,pre,arm,{'task':task,'seed':seed,'target_sources':sorted({s.target_source for s in samples}),'stop':stop},class_weights=cw.detach().cpu().numpy() if cw is not None else None)
    check=parity(model,artifact,[s.snapshot for s in samples[:5]])
    (output/'model.json').write_text(json.dumps(artifact,allow_nan=False));(output/'parity.json').write_text(json.dumps(check))
    return artifact,history


def fit_lambdamart(samples,pre,output,seed=1729):
    import lightgbm as lgb
    xs=[];ys=[];groups=[]
    for sample in samples:
        if len(sample.target)<2 or len(set(sample.target))<2:continue
        xs.append(arrays(sample.snapshot,pre)['summary']);ys.extend(np.floor(255*np.asarray(sample.target)+.5).astype(int));groups.append(len(sample.target))
    if not groups:raise ValueError('no graded ranking support')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    model=lgb.LGBMRanker(objective='lambdarank',n_estimators=100,learning_rate=.05,max_depth=3,num_leaves=7,
        min_child_samples=10,label_gain=(np.arange(256)/255).tolist(),random_state=seed,verbosity=-1,n_jobs=1)
    x=np.concatenate(xs);model.fit(x,np.array(ys),group=groups)
    model.booster_.save_model(str(out/'lightgbm.txt'))
    artifact={'schema':'semantic_scene_ranker_v1','kind':'trees','arm':'lambdamart','preprocessing':pre,'input_profile':pre['profile_identity'],'objective_version':'D_v1','reward_version':'none_supervised',
        'trees':[t['tree_structure'] for t in model.booster_.dump_model()['tree_info']],
        'source':{'seed':seed,'target':'quantized D risk; linear gain','groups':groups}}
    artifact['sha256']=digest(artifact);check=NumpyRanker(artifact)
    pred=np.concatenate([check.raw(s.snapshot) for s in samples if len(s.target)>1 and len(set(s.target))>1])
    if not np.allclose(pred,model.predict(x),atol=1e-6):raise ValueError('tree export mismatch')
    out=Path(output);out.mkdir(parents=True,exist_ok=True);model.booster_.save_model(str(out/'lightgbm.txt'))
    (out/'model.json').write_text(json.dumps(artifact,allow_nan=False));return artifact
