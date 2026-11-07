"""Independent shared-state choice probes and interpretable rollout summaries."""
from collections import defaultdict
import hashlib
import numpy as np
from Scheduler.scene_policy import ScenePolicy,canonical
from Scheduler.training.scene_learners import Example


def build_choice_probes(corpus,*,role='validation',traces=32,seed=41000):
    """Reference random trajectories; each method later receives identical states."""
    rows=[]
    for i in range(traces):
        env=corpus.make(role,seed+i,pattern=('no_repeat','local_repeat','cross_repeat')[i%3],
                        load=('light','congested','burst')[(i//3)%3])
        reference=ScenePolicy(kind='random',seed=73000+i)
        while not env.finished:
            context=env.observation()
            rows.append(Example(context,env.marginal_gains(),str(i),collection_policy='shared_random_probe_v1'))
            env.step(reference.select(context).message_id)
    return rows


def choice_probe_identity(probes):
    from dataclasses import asdict
    h=hashlib.sha256()
    for p in probes:
        h.update(canonical({'context':asdict(p.context),'gains':p.gains,'episode':p.episode_id}).encode())
    return h.hexdigest()


def evaluate_choices(policy,probes):
    """Ties supported; all-equal/singleton states excluded from preference accuracy."""
    correct=0;regrets=[];by_episode=defaultdict(list);excluded=0
    for p in probes:
        if len(set(p.gains))<2:
            excluded+=1;continue
        # Independent probes do not form the tested policy's own trajectory.
        policy.reset()
        if policy.kind=='rr' and p.context.successful_header_history:
            policy.last_source=p.context.successful_header_history[-1].source_id
        elif policy.kind=='random':
            identity=canonical({'seed':policy.seed,'episode':p.episode_id,
                'ready':sorted(c.message_id for c in p.context.ready_candidates)})
            policy.rng=np.random.default_rng(int(hashlib.sha256(identity.encode()).hexdigest()[:16],16))
        chosen=policy.select(p.context)
        i=next(j for j,c in enumerate(p.context.ready_candidates) if c.message_id==chosen.message_id)
        regret=max(p.gains)-p.gains[i]
        regrets.append(regret);correct+=int(regret==0);by_episode[p.episode_id].append(regret)
    n=len(regrets)
    return {'scope':'Identical random-reference states for every policy; not policy-visited-state accuracy',
        'stateful_probe_semantics':'Independent probes; RR follows last successful reference source; random uses a snapshot-qualified seed',
        'probe_sha256':choice_probe_identity(probes),'eligible':n,'excluded_all_tie_or_singleton':excluded,
        'tie_aware_top1_accuracy':correct/n if n else None,
        'mean_selection_regret':float(np.mean(regrets)) if n else None,
        'mean_episode_regret':float(np.mean([np.mean(v) for v in by_episode.values()])) if by_episode else None}


def summarize_rollouts(rows):
    def average(key,subset):
        values=[r[key] for r in subset if r.get(key) is not None]
        return {'mean':float(np.mean(values)) if values else None,'defined':len(values),'undefined':len(subset)-len(values)}
    result={k:average(k,rows) for k in ('J_time','coverage_at_horizon','mean_first_exposure_delay_capped',
                                     'undelivered_message_fraction','redundant_weight_fraction')}
    for metric in ('coverage_at_completions','distinct_useful_at_completions'):
        result[metric]={str(k):average('value',[{'value':r[metric][str(k)]} for r in rows]) for k in (1,5,10,20)}
    cells=defaultdict(list)
    for r in rows:cells[(r['load'],r['pattern'])].append(r)
    defined_cells=[average('J_time',v)['mean'] for v in cells.values()
                   if average('J_time',v)['mean'] is not None]
    result['equal_cell_macro_J']=float(np.mean(defined_cells)) if defined_cells else None
    result['by_workload_cell']={'/'.join(k):{'count':len(v),'J':average('J_time',v),
        'delay':average('mean_first_exposure_delay_capped',v)} for k,v in sorted(cells.items())}
    result['by_official_source_split']={}
    for source in sorted({s for r in rows for s in r['outcomes_by_official_source_split']}):
        subset=[r['outcomes_by_official_source_split'][source] for r in rows if source in r['outcomes_by_official_source_split']]
        result['by_official_source_split'][source]={'J':average('J_time',subset),'coverage':average('coverage',subset),
            'warning':'Repeated images across traces; counts are not independent groups or disasters.'}
    return result
