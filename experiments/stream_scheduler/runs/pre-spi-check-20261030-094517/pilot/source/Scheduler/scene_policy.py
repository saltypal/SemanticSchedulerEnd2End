"""Immutable numeric scene observations and framework-free selection/export."""
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import numpy as np
from Scheduler.decoder_scheduler import Decision


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class SceneCandidate:
    message_id: str
    source_id: str
    arrival_time: float
    sequence: int
    values: tuple
    available: tuple
    cold: tuple
    registry_hash: str

    def __post_init__(self):
        for name in ('values','available','cold'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        if not self.message_id or not self.registry_hash or not math.isfinite(self.arrival_time):
            raise ValueError('invalid candidate identity/time')
        if not len(self.values) == len(self.available) == len(self.cold):
            raise ValueError('scene feature lengths differ')
        if any(type(v) is not bool for v in self.available + self.cold):
            raise ValueError('scene availability must be boolean')
        if any(not math.isfinite(v) for v in self.values) or any(a and c for a,c in zip(self.available,self.cold)):
            raise ValueError('invalid numeric scene evidence')


@dataclass(frozen=True)
class SceneContext:
    ready_candidates: tuple
    successful_header_history: tuple
    attempts_completed: int
    admitted_count: int
    now: float
    horizon: float
    registry_hash: str

    def __post_init__(self):
        for name in ('ready_candidates','successful_header_history'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        if (not math.isfinite(self.now) or not math.isfinite(self.horizon)
                or self.now < 0 or self.horizon <= 0 or len(self.successful_header_history)>32):
            raise ValueError('invalid scene clock/history')
        if self.attempts_completed < 0 or self.admitted_count < self.attempts_completed:
            raise ValueError('invalid lifecycle counts')
        rows=self.ready_candidates+self.successful_header_history
        if any(not isinstance(c,SceneCandidate) or c.registry_hash!=self.registry_hash for c in rows):
            raise ValueError('incompatible scene feature registry')
        if rows and len({len(c.values) for c in rows})!=1:raise ValueError('inconsistent feature dimensions')
        if len({c.message_id for c in self.ready_candidates})!=len(self.ready_candidates):
            raise ValueError('duplicate candidate')


def tie_key(c):
    return c.arrival_time,c.sequence,c.message_id


def context_matrix(context, medians, distance_scale=None):
    """7*d+5 columns; no routing IDs, RGB, truth, future time or service cost."""
    d=len(medians)
    distance_scale=np.ones(d) if distance_scale is None else np.asarray(distance_scale,float)
    if distance_scale.shape!=(d,) or not np.isfinite(distance_scale).all() or np.any(distance_scale<=0):
        raise ValueError('invalid history-distance scales')
    def arrays(rows):
        return (np.asarray([c.values for c in rows],float).reshape(-1,d),
                np.asarray([c.available for c in rows],bool).reshape(-1,d))
    x,a=arrays(context.ready_candidates)
    h,ha=arrays(context.successful_header_history)
    if not len(x): return np.empty((0,7*d+5))
    medians=np.asarray(medians,float)
    def stats(v,mask):
        count=mask.sum(0)
        mean=np.divide(np.where(mask,v,0).sum(0),count,out=medians.copy(),where=count>0)
        return mean,count/max(1,len(v))
    qm,qa=stats(x,a); hm,hf=stats(h,ha)
    filled=np.where(a,x,medians)
    distances=[];comparable=[]
    # v1 defaults to raw distance. v2 supplies scales fitted on training only.
    for value,mask in zip(x,a):
        common=ha & mask
        counts=common.sum(1)
        if np.any(counts):
            delta=np.where(common,(h-value)/distance_scale,0)
            ds=np.sqrt(np.divide((delta*delta).sum(1),counts,out=np.zeros(len(h)),where=counts>0))
            distances.append(float(ds[counts>0].min()));comparable.append(1.)
        else: distances.append(0.);comparable.append(0.)
    cold=np.asarray([c.cold for c in context.ready_candidates],float)
    repeated=np.tile(np.concatenate([qm,qa,hm,hf]),(len(x),1))
    globals_=np.column_stack([distances,comparable,
        np.full(len(x),max(0.,1-context.now/context.horizon)),
        np.full(len(x),math.log1p(len(x))),np.full(len(x),math.log1p(context.attempts_completed))])
    result=np.concatenate([filled,a.astype(float),cold,repeated,globals_],axis=1)
    if not np.isfinite(result).all(): raise ValueError('nonfinite scene features')
    return result


def evidence_signature(candidate, fields):
    """Exact predicted-evidence repeat control; unknown evidence is not a match."""
    if not any(candidate.available[i] for i in fields):return None
    return tuple((candidate.available[i],candidate.values[i] if candidate.available[i] else None)
                 for i in fields)


class ScenePolicy:
    """All learned runtime inference is NumPy. Trees preserve <= semantics."""
    profile_id=None
    def __init__(self, artifact=None, *, kind='fifo', seed=0, field=None, signature_fields=(),defer_repeats=False):
        self.artifact=copy.deepcopy(artifact)
        self.kind=kind
        self.field=field
        self.signature_fields=tuple(signature_fields)
        self.defer_repeats=defer_repeats
        if type(defer_repeats) is not bool:raise ValueError('repeat filter must be boolean')
        if defer_repeats and (artifact is None or not self.signature_fields
            or any(type(i) is not int or i<0 for i in self.signature_fields)):
            raise ValueError('learned repeat filter requires a model and registered signature fields')
        if kind=='repeat_guard' and (type(field) is not int or field<0 or not self.signature_fields
            or any(type(i) is not int or i<0 for i in self.signature_fields)):
            raise ValueError('repeat guard requires registered score/signature fields')
        self.seed=seed
        self.reset()
        if artifact is not None:
            artifact=self.artifact
            unsigned={k:v for k,v in artifact.items() if k!='sha256'}
            if artifact.get('schema') not in ('scene_policy_v1','scene_policy_v2') or artifact.get('sha256')!=digest(unsigned):
                raise ValueError('corrupt scene policy artifact')
            self.kind=artifact['algorithm']
            d=len(artifact['medians']);p=7*d+5
            if defer_repeats and max(self.signature_fields)>=d:raise ValueError('signature field outside model registry')
            mean=np.asarray(artifact['mean']);scale=np.asarray(artifact['scale'])
            if mean.shape!=(p,) or scale.shape!=(p,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):
                raise ValueError('invalid scene preprocessing')
            if not np.isfinite(artifact['medians']).all():raise ValueError('invalid feature imputation')
            active=artifact.get('feature_indices',list(range(p)))
            if (not active or len(set(active))!=len(active)
                or any(type(i) is not int or not 0<=i<p for i in active)):
                raise ValueError('invalid active feature indices')
            distance=np.asarray(artifact.get('distance_scale',np.ones(d)),float)
            if distance.shape!=(d,) or not np.isfinite(distance).all() or np.any(distance<=0):
                raise ValueError('invalid artifact distance scales')
            if artifact['schema']=='scene_policy_v1' and ('feature_indices' in artifact or 'distance_scale' in artifact):
                raise ValueError('v2 preprocessing requires a v2 artifact')
            if artifact['schema']=='scene_policy_v2' and (artifact.get('preprocessing_version')!='training_scaled_active_v2'
                or 'feature_indices' not in artifact or 'distance_scale' not in artifact):
                raise ValueError('v2 preprocessing metadata required')
            model=artifact['model']
            if model.get('output_transform','identity') not in ('identity','remaining_sigmoid'):
                raise ValueError('unsupported model output transform')
            if model['kind']=='dense':
                width=len(active)
                if not model['layers']:raise ValueError('empty dense model')
                for layer in model['layers']:
                    weight=np.asarray(layer['weight']);bias=np.asarray(layer['bias'])
                    if weight.ndim!=2 or weight.shape[1]!=width or bias.shape!=(weight.shape[0],) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
                        raise ValueError('invalid dense layer')
                    width=weight.shape[0]
                if width!=1:raise ValueError('scalar output required')
            elif model['kind']=='trees':
                if model.get('output_transform','identity')!='identity':raise ValueError('tree output transform must be identity')
                if not model['trees']:raise ValueError('empty tree ensemble')
                for tree in model['trees']:
                    pending=[tree];seen=set()
                    while pending:
                        node=pending.pop()
                        if id(node) in seen:raise ValueError('cyclic/shared tree node')
                        seen.add(id(node))
                        if len(seen)>100000:raise ValueError('oversized tree')
                        if 'leaf_value' in node:
                            if not math.isfinite(node['leaf_value']):raise ValueError('invalid tree leaf')
                        else:
                            if (type(node.get('split_feature')) is not int or not 0<=node['split_feature']<len(active)
                                or node.get('decision_type','<=')!='<=' or not math.isfinite(node['threshold'])):
                                raise ValueError('invalid numeric tree split')
                            pending.extend([node['left_child'],node['right_child']])
            else:raise ValueError('unknown model kind')
            test=self._predict(np.zeros((1,p)))
            if test.shape!=(1,) or not np.isfinite(test).all(): raise ValueError('invalid model output')

    def reset(self):
        self.rng=np.random.default_rng(self.seed);self.last_source=None;self.fallback_count=0

    def _predict(self,x):
        a=self.artifact;m=a['model']
        z=(x-np.asarray(a['mean']))/np.asarray(a['scale'])
        if 'feature_indices' in a:z=z[:,a['feature_indices']]
        if m['kind']=='dense':
            for index,layer in enumerate(m['layers']):
                z=z@np.asarray(layer['weight']).T+np.asarray(layer['bias'])
                if index<len(m['layers'])-1: z=np.maximum(z,0)
            result=z.reshape(-1)
            if m.get('output_transform')=='remaining_sigmoid':
                result=np.clip(x[:,-3],0,1)/(1+np.exp(-np.clip(result,-60,60)))
            return result
        if m['kind']=='trees':
            def tree_value(node,row):
                while 'leaf_value' not in node:
                    feature=node['split_feature'];value=row[feature]
                    left=(value<=node['threshold'])
                    if np.isnan(value):left=node.get('default_left',True)
                    node=node['left_child'] if left else node['right_child']
                return node['leaf_value']
            return np.asarray([sum(tree_value(t,row) for t in m['trees']) for row in z])
        raise ValueError('unsupported exported scene model')

    def scores(self,context):
        if self.artifact is None: raise ValueError('policy has no model')
        if context.registry_hash!=self.artifact['registry_hash']: raise ValueError('feature registry mismatch')
        return self._predict(context_matrix(context,self.artifact['medians'],self.artifact.get('distance_scale')))

    def select(self,context):
        rows=context.ready_candidates
        if not rows:return None
        if self.artifact is not None:
            scores=self.scores(context)
            if scores.shape!=(len(rows),) or not np.isfinite(scores).all():
                self.fallback_count+=1
                c=min(rows,key=tie_key)
                return Decision(c.message_id,self.kind,0.,'nonfinite_model_output_fifo')
            seen={s for h in context.successful_header_history
                  if (s:=evidence_signature(h,self.signature_fields)) is not None} if self.defer_repeats else set()
            i=min(range(len(rows)),key=lambda j:(evidence_signature(rows[j],self.signature_fields) in seen
                if self.defer_repeats else False,-scores[j],tie_key(rows[j])))
            label=self.kind+'_repeat_filtered' if self.defer_repeats else self.kind
            return Decision(rows[i].message_id,label,float(scores[i]))
        ordered=sorted(rows,key=tie_key)
        if self.kind=='random': c=ordered[int(self.rng.integers(len(rows)))]
        elif self.kind=='rr':
            sources=sorted({c.source_id for c in rows})
            nexts=[s for s in sources if self.last_source is None or s>self.last_source]
            source=(nexts or sources)[0];self.last_source=source
            c=next(c for c in ordered if c.source_id==source)
        elif self.kind=='single':
            c=min(rows,key=lambda c:(-c.values[self.field] if c.available[self.field] else float('inf'),tie_key(c)))
        elif self.kind=='repeat_guard':
            if max((self.field,*self.signature_fields))>=len(rows[0].values):
                raise ValueError('repeat guard field outside registered observation')
            seen={s for h in context.successful_header_history
                  if (s:=evidence_signature(h,self.signature_fields)) is not None}
            c=min(rows,key=lambda c:(evidence_signature(c,self.signature_fields) in seen,
                -c.values[self.field] if c.available[self.field] else float('inf'),tie_key(c)))
        else:c=ordered[0]
        return Decision(c.message_id,self.kind,0.)


def seal_artifact(value):
    result=json.loads(canonical(value));result['sha256']=digest(result);return result
