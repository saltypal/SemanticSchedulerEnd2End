"""Event-driven evaluator-owned scene environment. Truth never enters context."""
from dataclasses import dataclass,asdict,field
import math
import time
from Scheduler.decoder_scheduler import Decision
from Scheduler.scene_policy import SceneContext,digest
from Evaluation.decoder_service import DecoderService
from Evaluation.scene_queue import SceneQueue,context_builder


class ActionPolicy:
    profile_id=None
    def reset(self):self.action=None
    def select(self,context):
        return None if not context.ready_candidates else Decision(self.action,'environment_action',0.)


@dataclass(frozen=True)
class SceneEpisode:
    messages:tuple
    facts:dict
    durations:dict
    horizon:float
    weights:dict
    failures:frozenset=frozenset()
    scope:str='source_observation_control'
    fact_metadata:dict=field(default_factory=dict)

    def __post_init__(self):
        ids={m.message_id for m in self.messages}
        if len(ids)!=len(self.messages) or set(self.facts)!=ids or set(self.durations)!=ids:
            raise ValueError('episode joins/identities differ')
        if not math.isfinite(self.horizon) or self.horizon<=0:raise ValueError('invalid horizon')
        if any(not math.isfinite(v) or v<=0 for v in self.durations.values()):raise ValueError('positive finite service durations required')
        offered=set().union(*self.facts.values()) if self.facts else set()
        if set(self.weights)!=offered or any(not math.isfinite(w) or w<0 for w in self.weights.values()):
            raise ValueError('invalid offered weights')
        if not self.failures<=ids:raise ValueError('foreign failed identity')
        if not set(self.fact_metadata)<=offered:raise ValueError('foreign evaluator fact metadata')


@dataclass(frozen=True)
class Transition:
    observation:SceneContext
    action_id:str
    action_index:int
    reward:float
    next_observation:SceneContext
    duration:float
    terminal:bool
    successful:bool
    completion_time:float


def transition_fingerprint(transition):
    """Teacher-side join; bind the entire causal decision and observed outcome."""
    row=asdict(transition);row.pop('reward')
    return digest(row)


class SceneEnvironment:
    def __init__(self,episode,registry,mission,specifications):
        self.episode=episode;self.registry=registry;self.mission=mission
        self.specifications=dict(specifications)
        self.reset()

    def reset(self):
        self.action_policy=ActionPolicy()
        queue=SceneQueue(self.action_policy,self.registry,self.mission,self.specifications)
        self.controller=DecoderService(self.action_policy,queue=queue,
                                       context_builder=context_builder(self.registry,self.episode.horizon))
        self.pending=sorted(self.episode.messages,key=lambda m:(m.arrival_time,m.sequence,m.message_id))
        self.now=0.;self.covered=set();self.events=[];self.transitions=[]
        self.total=sum(self.episode.weights.values());self.finished=False
        self._advance_to_ready()
        return self.observation()

    def _admit(self):
        while self.pending and self.pending[0].arrival_time<=self.now and self.pending[0].arrival_time<=self.episode.horizon:
            self.controller.enqueue(self.pending.pop(0))

    def _advance_to_ready(self):
        self._admit()
        if not len(self.controller) and self.pending:
            self.now=min(self.pending[0].arrival_time,self.episode.horizon);self._admit()
        self.finished=self.now>=self.episode.horizon or (not len(self.controller) and not self.pending)

    def observation(self):
        candidates=() if self.finished else self.controller.queue.ready_snapshot(self.now)
        return self.controller.context_builder(candidates,self.controller.history,
            self.controller._attempts,self.controller._admitted,self.now)

    def marginal_gains(self):
        # Training/evaluator-only entry point. No call from runtime feature code.
        return tuple(sum(self.episode.weights[f] for f in self.episode.facts[c.message_id]-self.covered)
                     for c in self.observation().ready_candidates)

    def step(self,action_id):
        if self.finished:raise RuntimeError('episode already terminated')
        before=self.observation();ids=[c.message_id for c in before.ready_candidates]
        if action_id not in ids:raise ValueError('illegal action identity')
        self.action_policy.action=action_id
        job=self.controller.dispatch(self.now)
        start=self.now;finish=start+self.episode.durations[action_id]
        self.now=min(finish,self.episode.horizon);self._admit()
        success=finish<=self.episode.horizon and action_id not in self.episode.failures
        gain=0.
        if finish<=self.episode.horizon:
            self.controller.complete(job.message.message_id,'succeeded' if success else 'failed',finish)
            if success:
                new=self.episode.facts[action_id]-self.covered
                gain=sum(self.episode.weights[f] for f in new);self.covered.update(new)
        self.events.append({'message_id':action_id,'start':start,'finish':finish,'success':success,
                            'unfinished':finish>self.episode.horizon,'new_weight':gain})
        reward=gain/self.total*max(self.episode.horizon-finish,0)/self.episode.horizon if self.total else 0.
        self._advance_to_ready()
        transition=Transition(before,action_id,ids.index(action_id),reward,self.observation(),
                              self.now-start,self.finished,success,finish)
        self.transitions.append(transition)
        return transition

    def metrics(self):
        # Independent area integration, not summing training rewards.
        covered=set();last=0.;area=0.;curve=[];new_weights=[];repeated=0.
        selected_weight=0.;delays=[];first_completions={}
        arrivals={f:min(m.arrival_time for m in self.episode.messages if f in self.episode.facts[m.message_id]) for f in self.episode.weights}
        for e in self.events:
            t=min(e['finish'],self.episode.horizon)
            area+=sum(self.episode.weights[f] for f in covered)*(t-last);last=t
            if e['success']:
                current=self.episode.facts[e['message_id']]
                new=current-covered
                selected_weight+=sum(self.episode.weights[f] for f in current)
                repeated+=sum(self.episode.weights[f] for f in current&covered)
                delays.extend(t-arrivals[f] for f in new)
                first_completions.update({f:t for f in new})
                covered.update(new)
                new_weights.append(sum(self.episode.weights[f] for f in new))
            curve.append({'time':t,'covered_weight':sum(self.episode.weights[f] for f in covered)})
        area+=sum(self.episode.weights[f] for f in covered)*(self.episode.horizon-last)
        weight=sum(self.episode.weights[f] for f in covered)
        completed_curve=[point for event,point in zip(self.events,curve) if not event['unfinished']]
        budget_weights={str(k):completed_curve[k-1]['covered_weight'] if len(completed_curve)>=k else None
                        for k in (1,5,10,20)}
        capped_delay=sum(w*max(0.,first_completions.get(f,self.episode.horizon)-min(arrivals[f],self.episode.horizon))
                         for f,w in self.episode.weights.items())
        strata={}
        for source in sorted({self.episode.fact_metadata.get(f,{}).get('official_source_split','unknown') for f in self.episode.weights}):
            fs=[f for f in self.episode.weights
                if self.episode.fact_metadata.get(f,{}).get('official_source_split','unknown')==source]
            total=sum(self.episode.weights[f] for f in fs)
            strata[source]={'offered_fact_count':len(fs),'offered_weight':total,
                'coverage':sum(self.episode.weights[f] for f in fs if f in covered)/total if total else None,
                'J_time':sum(self.episode.weights[f]*(self.episode.horizon-first_completions.get(f,self.episode.horizon))
                             for f in fs)/(total*self.episode.horizon) if total else None}
        return {'J_time':area/(self.total*self.episode.horizon) if self.total else None,
                'coverage_at_horizon':weight/self.total if self.total else None,
                'offered_weight':self.total,'covered_weight':weight,'useful_weight_deferred':self.total-weight,
                'redundant_weight_fraction':repeated/selected_weight if selected_weight else None,
                'exposure_delays':delays,'undelivered_fact_count':len(self.episode.weights)-len(covered),
                'failures':sum(not e['success'] and not e['unfinished'] for e in self.events),
                'unfinished':sum(e['unfinished'] for e in self.events),'attempts':len(self.events),
                'reward_sum':sum(t.reward for t in self.transitions),'curve':curve,'scope':self.episode.scope,
                'distinct_useful_at_completions':budget_weights,
                'coverage_at_completions':{k:w/self.total if w is not None and self.total else None for k,w in budget_weights.items()},
                'mean_first_exposure_delay_capped':capped_delay/self.total if self.total else None,
                'first_useful_completion_time':min((t for f,t in first_completions.items()
                    if self.episode.weights[f]>0),default=None),
                'outcomes_by_official_source_split':strata,
                'delay_scope':'First successful distinct-fact exposure; undelivered facts capped at horizon',
                'coverage_at_fractions':{str(fraction):sum(self.episode.weights[f] for f in set().union(
                    *(self.episode.facts[e['message_id']] for e in self.events
                      if e['success'] and e['finish']<=fraction*self.episode.horizon)))/self.total if self.total else None
                    for fraction in (.25,.5,.75,1.)},
                'waiting_times':[e['start']-next(m.arrival_time for m in self.episode.messages if m.message_id==e['message_id']) for e in self.events],
                'undelivered_message_fraction':1-sum(e['success'] for e in self.events)/len(self.episode.messages) if self.episode.messages else None,
                'header_bytes_offered':sum(len(m.header_bytes) for m in self.episode.messages),
                'payload_bytes':'opaque reference; measured only in actual codec replay'}


def run_policy(environment,policy):
    environment.reset();policy.reset()
    latencies=[]
    while not environment.finished:
        start=time.perf_counter()
        decision=policy.select(environment.observation())
        latencies.append({'queue_size':len(environment.observation().ready_candidates),'seconds':time.perf_counter()-start})
        if decision is None:raise RuntimeError('policy idle with ready candidates')
        environment.step(decision.message_id)
    result=environment.metrics();result['selection_latency']=latencies
    result['fallback_count']=getattr(policy,'fallback_count',0)
    result['latency_scope']='CPU selection plus context construction; simulated time excludes this overhead'
    return result
