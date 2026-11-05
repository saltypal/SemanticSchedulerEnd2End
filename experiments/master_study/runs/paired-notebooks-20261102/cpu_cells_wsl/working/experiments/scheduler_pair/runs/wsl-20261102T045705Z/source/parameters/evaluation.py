"""Independent observation facts and header-only policy replay, ideal-source arm."""
from dataclasses import dataclass
import itertools
import math
import numpy as np
from Evaluation.decoder_semantic_benchmark import coverage_metrics, marginal_gain
from .contracts import GROUPS, flatten_bundle


@dataclass(frozen=True)
class Observation:
    token: str
    source: int
    bundle: object
    historical_i: float | None = None


@dataclass(frozen=True)
class Episode:
    observations: tuple
    facts: dict
    weights: dict
    mission: str
    arm: str
    group_ids: tuple
    seed: int


def observation_facts(record,mission):
    """Native image observations do not imply distinct physical assets."""
    label=record['labels'].get(mission)
    return set() if label != 1 else {mission+'|'+record['parent_id']+'|native_claim'}


def build_episode(records,bundles,mission,seed,arm='native',jobs=32):
    eligible=[r for r in records if r['labels'].get(mission) in (0,1) and r['image_id'] in bundles]
    if arm=='relevant':eligible=[r for r in eligible if r['labels'][mission]==1]
    minimum=jobs//2 if arm=='exact_copy' else jobs
    if len(eligible)<minimum:return None
    rng=np.random.default_rng(seed)
    selected=[eligible[i] for i in rng.choice(len(eligible),size=minimum,replace=False)]
    if arm=='exact_copy':selected=selected+selected
    selected=[selected[i] for i in rng.permutation(len(selected))]
    observations=[];facts={}
    if hasattr(bundles,'begin'):bundles.begin()
    for sequence,record in enumerate(selected):
        token=str(sequence)+':'+record['image_id']
        bundle=bundles.for_observation(record,sequence,token) if hasattr(bundles,'for_observation') else bundles[record['image_id']]
        observations.append(Observation(token,sequence%4,bundle,record.get('historical_i',{}).get(mission)))
        facts[token]=observation_facts(record,mission)
    weights={fact:1. for values in facts.values() for fact in values}
    return Episode(tuple(observations),facts,weights,mission,arm,tuple(sorted({r['group'] for r in selected})),seed)


class FeatureEncoder:
    """No image, ID, source label, native truth or provenance enters numeric rows."""
    def __init__(self,fields,medians=None):
        self.fields=tuple(fields)
        self.medians=np.full(len(fields),.5) if medians is None else np.asarray(medians,float)
        if self.medians.shape!=(len(fields),) or not np.isfinite(self.medians).all():raise ValueError('invalid imputation')

    @classmethod
    def fit(cls,decisions,groups):
        allowed=set(groups)
        rows=[flatten_bundle(o.bundle,groups) for d in decisions for o in d['ready']]
        fields=sorted({k for row in rows for k in row if k.split('.')[0] in allowed})
        medians=[float(np.median([r[k] for r in rows if k in r])) for k in fields]
        if not fields:raise ValueError('no available fields in this configuration')
        return cls(fields,medians)

    def matrix(self,ready,history,attempts,total):
        def arrays(observations):
            raw=[flatten_bundle(o.bundle) for o in observations]
            masks=np.array([[field in row for field in self.fields] for row in raw],bool).reshape(len(raw),len(self.fields))
            vals=np.array([[row.get(field,self.medians[i]) for i,field in enumerate(self.fields)] for row in raw],float).reshape(len(raw),len(self.fields))
            return vals,masks
        hv,hm=arrays(history);cv,cm=arrays(ready)
        counts=hm.sum(axis=0)
        means=np.divide((hv*hm).sum(axis=0),counts,out=self.medians.copy(),where=counts>0)
        fractions=counts/max(1,len(history));result=[]
        for values,mask in zip(cv,cm):
            comparable=hm&mask;common=comparable.sum(axis=1)
            distance=np.divide((np.abs(hv-values)*comparable).sum(axis=1),common,out=np.ones(len(history)),where=common>0)
            usable=np.sort(distance[common>0])
            signature=any(np.array_equal(mask,hmask) and np.allclose(values[mask],hval[mask],rtol=0,atol=1/255) for hval,hmask in zip(hv,hm))
            result.append(np.concatenate([values,mask.astype(float),means,fractions,
                [usable[0] if len(usable) else 0.,usable[:3].mean() if len(usable) else 0.,float(len(usable)>0),float(signature),float(bool(history)),attempts/max(1,total),len(ready)/max(1,total)]]))
        return np.asarray(result,float).reshape(len(ready),4*len(self.fields)+7)


def scalar(observation,name,default=.5):
    result=observation.bundle.parameters.get(name)
    if result is None or not result.available:return default
    if isinstance(result.value,dict):return float(result.value.get('distance',default))
    return float(result.value)


def replay(episode,policy,cutoff=10):
    pending=list(episode.observations);history=[];order=[];trace=[];last_source=-1
    rng=np.random.default_rng(episode.seed)
    if policy=='historical_I' and any(o.historical_i is None for o in pending):return None
    if policy in ('relevance','novelty','incident_filter'):
        field='novelty' if policy=='novelty' else 'relevance'
        if not any(o.bundle.parameters.get(field) is not None and
                   o.bundle.parameters[field].available for o in pending):
            return None
    for step in range(len(pending)):
        scores=None
        if isinstance(policy,str):
            if policy=='fifo':index=0
            elif policy=='random':index=int(rng.integers(len(pending)))
            elif policy=='round_robin':
                sources=sorted({o.source for o in pending});next_sources=[s for s in sources if s>last_source]
                source=next_sources[0] if next_sources else sources[0]
                index=next(i for i,o in enumerate(pending) if o.source==source);last_source=source
            elif policy in ('relevance','novelty','historical_I','incident_filter'):
                scores=np.array([o.historical_i if policy=='historical_I' else scalar(o,'novelty' if policy=='novelty' else 'relevance') for o in pending])
                if policy=='incident_filter':scores=(scores>=.625).astype(float)
                index=int(np.argmax(scores))
            else:raise ValueError('unknown policy')
        else:
            scores=policy.scores(pending,history,step,len(episode.observations))
            if not np.isfinite(scores).all():raise ValueError('nonfinite learned policy output')
            index=int(np.argmax(scores))
        selected=pending.pop(index);order.append(selected.token);history.append(selected)
        trace.append({'step':step+1,'selected':selected.token,'score':None if scores is None else float(scores[index]),'successful_headers':len(history)})
    metrics=coverage_metrics(order,episode.facts,episode.weights,k=min(cutoff,len(order)))
    return {'metrics':metrics,'order':order,'trace':trace,'mission':episode.mission,'arm':episode.arm,
            'groups':list(episode.group_ids),'scope':'ideal source evidence; no actual decoder inference'}


def training_decisions(episodes,cutoff=10):
    output=[]
    for episode in episodes:
        # Fixed random/relevance histories; native annotations produce targets only.
        for kind in ('random','relevance'):
            result=replay(episode,kind,cutoff)
            delivered=[];by_id={o.token:o for o in episode.observations}
            for step,token in enumerate(result['order'][:cutoff]):
                ready=tuple(o for o in episode.observations if o.token not in delivered)
                targets=[marginal_gain(o.token,delivered,episode.facts,episode.weights) for o in ready]
                if all(value is not None for value in targets):
                    output.append({'ready':ready,'history':tuple(by_id[t] for t in delivered),'attempts':step,
                                   'total':len(episode.observations),'targets':targets,'weight':1/(2*cutoff)})
                delivered.append(token)
    return output


def grouped_split(records,folds=3,seed=7):
    groups=sorted({r['group'] for r in records})
    if len(groups)<folds:raise ValueError('insufficient independent groups')
    rng=np.random.default_rng(seed);rng.shuffle(groups)
    assignment={group:i%folds for i,group in enumerate(groups)}
    return [([r for r in records if assignment[r['group']]!=fold],
             [r for r in records if assignment[r['group']]==fold]) for fold in range(folds)]


def feature_subsets(groups):
    return [tuple(c) for n in range(1,len(groups)+1) for c in itertools.combinations(groups,n)]
