"""Provenance-bearing preference/replay labels; no teacher truth fabrication."""
from dataclasses import dataclass
import json
import hashlib
import math
import numpy as np
from Scheduler.scene_policy import canonical,digest,ScenePolicy
from Scheduler.training.scene_learners import Example,make_network,preprocessing,tensor,export_network


def parse_choice(text,candidate_ids):
    from Evaluation.scene_queue import strict_json
    row=strict_json(text)
    if set(row)-{'preference','evidence','unknowns','aspects','training_value'}:
        raise ValueError('unexpected teacher keys')
    if row.get('preference') not in set(candidate_ids)|{'tie','uncertain'}:raise ValueError('foreign teacher selection')
    if not isinstance(row.get('evidence'),str) or not row['evidence'].strip():raise ValueError('supporting evidence required')
    if not isinstance(row.get('unknowns'),list) or not all(isinstance(x,str) for x in row['unknowns']):
        raise ValueError('unknowns list required')
    if 'training_value' in row and (type(row['training_value']) is not int or row['training_value'] not in (0,1,2)):
        raise ValueError('training_value must be integer 0, 1 or 2')
    return row


def control_gate(rows):
    """All denominators include invalid outputs; no filtering into success."""
    def rate(expected):
        subset=[r for r in rows if r['expected']==expected]
        return sum(r.get('valid',False) and r.get('observed')==expected for r in subset)/len(subset) if subset else None
    rates={k:rate(k) for k in ('A','B','tie','uncertain')}
    valid=sum(r.get('valid',False) for r in rows)/len(rows) if rows else 0.
    ba=(rates['A']+rates['B'])/2 if rates['A'] is not None and rates['B'] is not None else None
    passed=valid>=.95 and ba is not None and ba>=.8 and all(rates[k] is not None and rates[k]>=.75 for k in ('tie','uncertain'))
    return {'passed':passed,'valid_fraction':valid,'strict_balanced_accuracy':ba,'class_rates':rates,
            'count':len(rows),'scope':'task-specific controls, not general teacher correctness'}


def normalized_pair(forward,reverse):
    mapping={'A':'B','B':'A','tie':'tie','uncertain':'uncertain'}
    a=forward['preference'];b=mapping[reverse['preference']]
    return a if a==b else 'uncertain'


def aspect_preferences(response,candidates):
    aspects=response.get('aspects')
    keys={'mission_relevance','evidence_clarity','additional_information','condition_detail'}
    if not isinstance(aspects,dict) or set(aspects)!=set(candidates):raise ValueError('candidate aspect ratings required')
    means={}
    for candidate,ratings in aspects.items():
        if set(ratings)!=keys or any(type(v) is not int or not 1<=v<=5 for v in ratings.values()):
            raise ValueError('four integer aspect scores 1..5 required')
        means[candidate]=sum(ratings.values())/4
    return tuple((i,j,1.) for i,a in enumerate(candidates) for j,b in enumerate(candidates) if means[a]>means[b])


@dataclass(frozen=True)
class PrefixPair:
    left:tuple
    right:tuple
    durations_left:tuple
    durations_right:tuple
    preferred_left:float
    group:str
    origin:str='qwen_prefix_preference'
    indices_left:tuple=()
    indices_right:tuple=()

    def __post_init__(self):
        if not self.left or not self.right or len(self.left)!=len(self.durations_left) or len(self.right)!=len(self.durations_right):
            raise ValueError('prefix states/durations differ')
        if any(not math.isfinite(v) or v<0 for v in self.durations_left+self.durations_right):raise ValueError('invalid prefix duration')
        if not math.isclose(sum(self.durations_left),sum(self.durations_right),rel_tol=1e-9,abs_tol=1e-9):
            raise ValueError('teacher alternatives must end at the same elapsed time')
        if not 0<=self.preferred_left<=1:raise ValueError('invalid preference target')
        for contexts,indices in ((self.left,self.indices_left),(self.right,self.indices_right)):
            if indices and (len(indices)!=len(contexts) or any(type(i) is not int or not 0<=i<len(c.ready_candidates) for i,c in zip(indices,contexts))):
                raise ValueError('prefix action indices required for every decision')
            if not indices and any(len(c.ready_candidates)!=1 for c in contexts):
                raise ValueError('ambiguous prefix candidate selection')


class LearnedPrefixReward:
    """Reward model predicts per-state usefulness; training compares time integrals."""
    def __init__(self,artifact):self.policy=ScenePolicy(artifact);self.artifact=artifact
    def __call__(self,transition):
        scores=self.policy.scores(transition.observation)
        value=float(np.clip(scores[transition.action_index],-10,10))
        # The next decision may follow an idle interval after completion.
        remaining=max(transition.observation.horizon-transition.completion_time,0)/transition.observation.horizon
        return value*remaining if transition.successful else 0.


def fit_reward_model(training,validation,*,epochs=50,seed=1729,fixture=False):
    import torch
    from torch.nn import functional as F
    if len(training)<16 or len(validation)<8:raise ValueError('insufficient strict teacher prefix pairs')
    if {p.group for p in training}&{p.group for p in validation}:raise ValueError('reward train/validation group leakage')
    if not fixture and any(p.origin!='qwen_prefix_preference' for p in training+validation):
        raise ValueError('registered teacher-origin prefix pairs required')
    torch.manual_seed(seed)
    examples=[Example(c,tuple(0. for _ in c.ready_candidates),p.group) for p in training for c in p.left+p.right]
    from Scheduler.training.scene_learners import input_width
    pre=preprocessing(examples);model=make_network(input_width(pre))
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    def total(contexts,durations,indices):
        elapsed=0.;end=sum(durations);result=0.
        for n,(c,duration) in enumerate(zip(contexts,durations)):
            elapsed+=duration
            index=indices[n] if indices else 0
            result=result+model(tensor(c,pre)).flatten()[index]*max(end-elapsed,0)/max(end,1e-9)
        return result
    def logit(pair):
        return total(pair.left,pair.durations_left,pair.indices_left)-total(pair.right,pair.durations_right,pair.indices_right)
    losses=[]
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss=torch.stack([F.binary_cross_entropy_with_logits(logit(p),torch.tensor(p.preferred_left)) for p in training]).mean()
        loss.backward();optimizer.step();losses.append(float(loss.detach()))
    with torch.no_grad():
        correct=sum((float(logit(p))>0)==(p.preferred_left>.5) for p in validation)
    accuracy=correct/len(validation)
    majority=max(sum(p.preferred_left>.5 for p in validation),sum(p.preferred_left<.5 for p in validation))/len(validation)
    artifact=export_network(model,pre,'rl_vlm_reward',{'origin':'qwen_prefix_preference' if not fixture else 'fixture',
        'validation_accuracy':accuracy,'majority_accuracy':majority,'promoted':accuracy>=.75 and accuracy>majority})
    return artifact,{'validation_accuracy':accuracy,'majority_accuracy':majority,
                     'promoted':accuracy>=.75 and accuracy>majority,'loss':losses}, {'model':model.state_dict(),'optimizer':optimizer.state_dict()}


class TeacherLedger:
    def __init__(self,path):
        from pathlib import Path
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
    def append(self,*,request,response,model,prompt_version,input_hashes,status,origin='qwen_teacher'):
        row={'request':request,'response':response,'model':model,'prompt_version':prompt_version,
             'input_hashes':input_hashes,'status':status,'origin':origin}
        row['sha256']=digest(row)
        with self.path.open('a',encoding='utf-8') as f:f.write(canonical(row)+'\n')
        return row


class QwenChoiceTeacher:
    """Uses an already provisioned frozen Transformers backend; never reads queue."""
    def __init__(self,model,processor,model_identity,device):
        self.model=model;self.processor=processor;self.identity=model_identity;self.device=device
    def compare(self,images,mission,history_summary='',aspects=False):
        import torch,time
        from PIL import Image
        content=[];hashes=[]
        for identity,path in images:
            from pathlib import Path
            raw=Path(path).read_bytes();hashes.append(hashlib.sha256(raw).hexdigest())
            with Image.open(path) as source:
                image=source.convert('RGB');image.thumbnail((512,512))
                content.extend([{'type':'text','text':'Candidate '+identity},{'type':'image','image':image.copy()}])
        prompt=('Mission: '+mission+'\nPast successfully delivered evidence: '+history_summary+
            '\nChoose which candidate should be decoded next based only on visible evidence. '
            'Do not invent injury, occupancy, location, rescue urgency or future arrivals. '
            'Return JSON {"preference":"A" or "B" or "tie" or "uncertain",'
            '"evidence":"short visible supporting evidence","unknowns":[]}.')
        if aspects:prompt+=' Include aspects for A and B: mission_relevance, evidence_clarity, additional_information, condition_detail, each integer 1..5.'
        content.append({'type':'text','text':prompt})
        start=time.perf_counter()
        batch=self.processor.apply_chat_template([{'role':'user','content':content}],tokenize=True,
            add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to(self.device)
        with torch.inference_mode():
            output=self.model.generate(**batch,max_new_tokens=384,do_sample=False,logits_to_keep=1)
        text=self.processor.decode(output[0,batch['input_ids'].shape[1]:],skip_special_tokens=True)
        parse_error=None
        try:parsed=parse_choice(text,[i for i,_ in images])
        except (ValueError,KeyError) as error:parsed={'preference':'uncertain'};parse_error=str(error)
        return {'raw':text,'parsed':parsed,'parse_error':parse_error,'seconds':time.perf_counter()-start,
                'input_hashes':hashes,'enable_thinking':False,'model':self.identity,
                'image_count':len(images),'input_tokens':int(batch['input_ids'].shape[1]),
                'input_shapes':{k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')},
                'device':str(batch['input_ids'].device),
                'peak_allocated_bytes':int(torch.cuda.max_memory_allocated(self.device)) if str(self.device).startswith('cuda') else None}
