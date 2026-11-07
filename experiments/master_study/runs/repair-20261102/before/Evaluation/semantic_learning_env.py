"""One-shot semantic scheduling environment, exact D cost and causal runtime snapshots.
Source references/embeddings never enter Snapshot. Unit service is synthetic.
"""
from dataclasses import dataclass
from collections import deque
import copy
import math
import numpy as np

LEVELS=('Critical','High','Medium','Low')
BUDGETS=np.array([2.,4.,8.,16.])
PROFILES=(
('phone','General-purpose phone. Share user-selected everyday observations without assuming an emergency.'),
('incident_phone','Citizen incident reporting phone. Document visible incidents and useful changes for local response.'),
('damage_drone','Aerial structural inspection drone. Identify visible building damage and newly exposed structural evidence.'),
('person_drone','Person-search drone. Locate visible people and evidence relevant to search without assuming distress.'),
('flood_drone','Flood and evacuation drone. Observe inundation and passable or blocked evacuation routes.'),
('fire_drone','Wildfire observation drone. Observe visible smoke and flame and supported changes near infrastructure.'),
('road_drone','Road and bridge drone. Inspect transport structures and visible obstruction or damage.'),
('car','Ordinary connected-car camera. Observe immediate road conditions and travel obstacles.'),
('response_car','Emergency vehicle camera. Report response access and visible incident developments.'),
('traffic_cctv','Fixed intersection CCTV. Observe traffic flow, collisions and obstructions.'),
('public_cctv','Fixed public-space CCTV. Observe visible incidents and crowd changes without inferring danger from occupancy.'),
('water_camera','Fixed river and underpass camera. Observe water and route conditions without inferring physical depth.'),
('robot','Industrial inspection robot. Report visible equipment and area conditions on its inspection route.'),
('farm_drone','Agricultural survey drone. Observe crop and field conditions and supported changes.'),
('maintenance','Infrastructure maintenance camera. Document inspection findings, repair evidence and relevant changes.'))


def ell(wait,budget):
    waits=np.asarray(wait,dtype=float);budgets=np.asarray(budget,dtype=float)
    if not np.isfinite(waits).all() or not np.isfinite(budgets).all() or np.any(budgets<=0):
        raise ValueError("finite waits and positive finite delay budgets required")
    z=np.maximum(0.,waits)/budgets
    return z+np.maximum(z-1,0.)


def readonly(x):
    a=np.asarray(x,dtype=np.float32)
    return np.frombuffer(a.tobytes(),dtype=np.float32).reshape(a.shape)


@dataclass(frozen=True)
class Message:
    token:int
    source:int
    session:int
    arrival:float
    sequence:int
    features:object
    header_bytes:bytes=b''
    profile_id:str='fixture'

    def __post_init__(self):
        if self.token<0 or self.source<0 or self.session<0 or self.sequence<0 or not math.isfinite(self.arrival) or self.arrival<0:
            raise ValueError('invalid runtime identity/time')
        x=readonly(self.features)
        if x.shape!=(387,) or not np.isfinite(x).all():raise ValueError('387 finite received features required')
        object.__setattr__(self,'features',x)


@dataclass(frozen=True)
class Snapshot:
    ready:tuple
    processed:tuple # (Message, attempt completion time)
    ue_windows:tuple # tuple(Message,...), candidate aligned, current candidate excluded
    active_windows:tuple # one recent window per observed source session
    now:float
    attempts:int
    waiting_tokens:object=None
    in_flight_tokens:tuple=()

    @property
    def profile_id(self):
        rows=list(self.ready)+[m for m,t in self.processed]+[m for w in self.ue_windows+self.active_windows for m in w]
        ids={m.profile_id for m in rows}
        if len(ids)>1:raise ValueError('mixed runtime feature producer profiles')
        return next(iter(ids),'fixture')

    def __post_init__(self):
        if len(self.ready)!=len(self.ue_windows) or len(self.processed)>32 or not math.isfinite(self.now) or self.now<0 or self.attempts<0:
            raise ValueError('invalid context membership/history')
        if self.waiting_tokens is None:object.__setattr__(self,'waiting_tokens',frozenset(m.token for m in self.ready))
        else:object.__setattr__(self,'waiting_tokens',frozenset(self.waiting_tokens))
        all_messages=list(self.ready)+[m for m,t in self.processed]
        all_messages += [m for w in self.ue_windows+self.active_windows for m in w]
        if any(m.arrival>self.now for m in all_messages):raise ValueError('future message leaked')
        if any(not math.isfinite(t) or t<m.arrival or t>self.now for m,t in self.processed):
            raise ValueError('invalid processed completion time')
        if len({m.token for m,t in self.processed})!=len(self.processed) or {m.token for m,t in self.processed}&{m.token for m in self.ready}:
            raise ValueError('processed and ready identities must be disjoint and unique')
        if len({m.token for m in self.ready})!=len(self.ready):raise ValueError('duplicate ready identity')
        if any(m.token not in self.waiting_tokens for m in self.ready):raise ValueError('ready lifecycle status mismatch')
        _=self.profile_id

    def pair_view(self,ids):
        positions=[next(i for i,m in enumerate(self.ready) if m.token==x) for x in ids]
        sources={(self.ready[i].source,self.ready[i].session) for i in positions}
        windows=tuple(w for w in self.active_windows if w and (w[-1].source,w[-1].session) in sources)
        views=tuple(self.ue_windows[i] for i in positions)
        visible={self.ready[i].token for i in positions}|{m.token for w in views+windows for m in w}|{m.token for m,t in self.processed[-2:]}
        return Snapshot(tuple(self.ready[i] for i in positions),self.processed[-2:],views,windows,self.now,self.attempts,
            self.waiting_tokens & visible,tuple(t for t in self.in_flight_tokens if t in visible))


@dataclass(frozen=True)
class Episode:
    messages:tuple
    reference_tiers:tuple # evaluator only, -1 unknown
    fact_ids:tuple # exact-image correspondence only; NOT physical assets
    service:tuple
    failures:tuple
    horizon:float
    metadata:dict

    def __post_init__(self):
        n=len(self.messages)
        if not all(len(x)==n for x in (self.reference_tiers,self.fact_ids,self.service,self.failures)):
            raise ValueError('episode arrays differ')
        if len({m.token for m in self.messages})!=n or any(not np.isfinite(d) or d<=0 for d in self.service):
            raise ValueError('invalid identity or service')
        if not math.isfinite(self.horizon) or self.horizon<=0 or any(m.arrival>=self.horizon for m in self.messages):raise ValueError('invalid horizon')
        if any(t not in (-1,0,1,2,3) for t in self.reference_tiers):raise ValueError('invalid reference tier')


class Environment:
    """Single lifecycle owner for the new received-scene objective version.

    step selects one READY job, advances nonpreemptively, admits concurrent arrivals,
    commits processed headers after completion and never retries. Training references
    are separate from every runtime Snapshot. This does not call an actual codec.
    """
    def __init__(self,episode,task='D'):
        if task not in ('D','E','P'):raise ValueError('unknown objective')
        if task=='D' and any(z not in range(4) for z in episode.reference_tiers):
            raise ValueError('D training requires assessable reference tiers for every offer')
        self.episode,self.task=episode,task
        self.reset()

    def reset(self):
        self.now=0.;self.done=False;self.attempts=0;self.events=[];self.arrived=0;self.inflight=None;self.rate_path=[]
        self.ready={};self.rings={};self.processed=deque(maxlen=32);self.selected={};self.covered=set()
        self.ordered=sorted(self.episode.messages,key=lambda m:(m.arrival,m.sequence,m.token))
        self.index={m.token:i for i,m in enumerate(self.episode.messages)}
        self._admit();self._advance_idle()
        return self.snapshot() if not self.done else None

    def _admit(self):
        while self.arrived<len(self.ordered) and self.ordered[self.arrived].arrival<=self.now:
            m=self.ordered[self.arrived];self.arrived+=1;self.ready[m.token]=m
            key=(m.source,m.session)
            self.rings.setdefault(key,deque(maxlen=4)).append(m)

    def _advance_idle(self):
        if not self.ready and self.arrived<len(self.ordered) and self.ordered[self.arrived].arrival<self.episode.horizon:
            self.now=self.ordered[self.arrived].arrival;self._admit()
        if self.now>=self.episode.horizon or not self.ready and self.arrived==len(self.ordered):
            self.now=self.episode.horizon;self.done=True

    def snapshot(self):
        ready=tuple(sorted(self.ready.values(),key=lambda m:(m.arrival,m.sequence,m.token)))
        windows=tuple(tuple(x for x in self.rings[(m.source,m.session)] if x.token!=m.token)[-3:] for m in ready)
        active=tuple(tuple(w)[-3:] for key,w in sorted(self.rings.items()) if w)
        return Snapshot(ready,tuple(self.processed),windows,active,self.now,self.attempts,
            frozenset(self.ready),() if self.inflight is None else (self.inflight.token,))

    def cost(self,t):
        ep=self.episode
        if self.task=='P':return 0.
        if self.task=='D':
            waits=[max(0.,min(t,self.selected.get(m.token,float('inf')))-m.arrival) for m in ep.messages]
            return float(ell(waits,BUDGETS[np.asarray(ep.reference_tiers)]).sum())
        offered={}
        for m,fact in zip(ep.messages,ep.fact_ids):offered[fact]=min(offered.get(fact,float('inf')),m.arrival)
        first={}
        for m,fact in zip(ep.messages,ep.fact_ids):
            if m.token in self.selected:first[fact]=min(first.get(fact,float('inf')),self.selected[m.token])
        return float(sum(max(0.,min(t,first.get(f,float('inf')))-a) for f,a in offered.items()))

    def target(self,snapshot=None):
        s=snapshot or self.snapshot()
        if self.task=='P':raise ValueError('P requires actual teacher labels, not synthetic targets')
        if self.task=='E':
            return np.array([float(self.episode.fact_ids[self.index[m.token]] not in self.covered) for m in s.ready])
        z=np.array([self.episode.reference_tiers[self.index[m.token]] for m in s.ready])
        waits=np.array([s.now-m.arrival for m in s.ready]);d=BUDGETS[z]
        return ell(waits+1,d)-ell(waits,d)

    def step(self,token):
        if self.done or token not in self.ready:raise ValueError('selection must be currently ready')
        before=self.cost(self.now);m=self.ready.pop(token);i=self.index[token];start=self.now
        self.inflight=m
        self.selected[token]=start;self.covered.add(self.episode.fact_ids[i])
        finish=start+self.episode.service[i];complete=finish<=self.episode.horizon
        self.events.append({'token':token,'source':m.source,'arrival':m.arrival,'dispatch':start,
            'finish':finish,'observed_until':min(finish,self.episode.horizon),'wait':start-m.arrival,
            'completed':complete,'failed':bool(self.episode.failures[i]) if complete else None,
            'reference_tier':self.episode.reference_tiers[i]})
        self.now=min(finish,self.episode.horizon);self._admit()
        if complete:
            self.attempts+=1;self.processed.append((m,finish));self.inflight=None
        self._advance_idle()
        reward=-(self.cost(self.now)-before)/100.
        return (None if self.done else self.snapshot()),reward,self.done

    def metrics(self):
        ep=self.episode;selected={e['token']:e for e in self.events};classes={}
        for z,name in enumerate(LEVELS):
            jobs=[m for m,r in zip(ep.messages,ep.reference_tiers) if r==z]
            picked=[selected[m.token] for m in jobs if m.token in selected]
            follow=[m for m in jobs if m.arrival+16<=ep.horizon]
            eligible=[m for m in jobs if m.arrival+BUDGETS[z]<=ep.horizon]
            classes[name]={'offered':len(jobs),'selected':len(picked),'unselected':len(jobs)-len(picked),
                'mean_selected_wait':float(np.mean([e['wait'] for e in picked])) if picked else None,
                'median_selected_wait':float(np.median([e['wait'] for e in picked])) if picked else None,
                'p95_selected_wait':float(np.quantile([e['wait'] for e in picked],.95)) if len(picked)>=20 else None,
                'restricted_mean_wait':float(np.mean([min(selected.get(m.token,{}).get('wait',float('inf')),16.) for m in follow])) if follow else None,
                'followup_count':len(follow),'eligible':len(eligible),
                'within_budget':sum(m.token in selected and selected[m.token]['wait']<=BUDGETS[z] for m in eligible)/len(eligible) if eligible else None}
        return {'J_D':self.cost(ep.horizon)/len(ep.messages) if self.task=='D' and ep.messages else None,
            'L_E':self.cost(ep.horizon) if self.task=='E' else None,'offered':len(ep.messages),
            'selected':len(selected),'queued_at_horizon':len(self.ready),'not_arrived':len(ep.messages)-self.arrived,
            'completed':sum(e['completed'] for e in self.events),
            'failed':sum(bool(e['failed']) for e in self.events),'in_flight':sum(not e['completed'] for e in self.events),
            'by_reference_tier':classes}


class Rule:
    def __init__(self,kind,seed=1729):self.kind=kind;self.rng=np.random.default_rng(seed);self.last=-1
    def select(self,s):
        if not s.ready:return None
        if self.kind=='random':return s.ready[int(self.rng.integers(len(s.ready)))].token
        if self.kind=='round_robin':
            sources=sorted({m.source for m in s.ready});source=next((x for x in sources if x>self.last),sources[0]);self.last=source
            return next(m.token for m in s.ready if m.source==source)
        if self.kind=='fifo':return s.ready[0].token
        if self.kind in ('spi_fifo','spi_novelty','predicted_edf') and any(not m.features[380] for m in s.ready):return s.ready[0].token
        def score(m):
            x=m.features
            if self.kind in ('spi_fifo','spi_novelty','predicted_edf'):
                z=int(np.argmax(x[374:378])) if x[380] else 4
                if self.kind=='predicted_edf':return -(m.arrival+(BUDGETS[z] if z<4 else 16.))
                return (4-z)+(float(x[379])*.1 if self.kind=='spi_novelty' and x[385] else 0.)
            if self.kind=='relevance':return float(x[378]) if x[384] else -float('inf')
            if self.kind=='novelty':return float(x[379]) if x[385] else -float('inf')
            raise ValueError('unknown baseline')
        if self.kind in ('relevance','novelty') and any(not m.features[384 if self.kind=='relevance' else 385] for m in s.ready):return s.ready[0].token
        return max(s.ready,key=lambda m:score(m)).token


def received_features(base,vector,past,mission_vector=None):
    """Shared causal extraction from cached evidence; caller commits after query."""
    x=np.array(base,dtype=float,copy=True)
    if vector is None:
        x[378:380]=0;x[384:387]=0
        return x
    v=np.asarray(vector);cold=not past;comparable=[z for z in past if z is not None]
    rel=float(np.clip(v@mission_vector,-1,1)) if mission_vector is not None else None
    nov=float((1-np.clip(max(v@z for z in comparable),-1,1))/2) if comparable else None
    x[378]=round((rel+1)*127.5)/127.5-1 if rel is not None else 0.;x[384]=rel is not None
    x[379]=0. if cold or nov is None else round(nov*255)/255;x[385]=not cold and nov is not None;x[386]=cold
    return x


def make_episode(records,features,embeddings,text_vectors,*,count=48,ues=4,load=.9,seed=1729,repeat='none',failure_rate=0.,variable_service=False,churn=False,burst=False):
    rng=np.random.default_rng(seed)
    if count<1 or ues<1 or not math.isfinite(load) or load<=0 or not records:raise ValueError('invalid offered workload')
    if repeat not in ('none','local','cross'):raise ValueError('unknown repeat construction')
    if not math.isfinite(failure_rate) or not 0<=failure_rate<=1:raise ValueError('failure probability must lie in0..1')
    base=count if repeat=='none' else max(1,round(count*2/3))
    if base>len(records):raise ValueError('insufficient distinct observations for requested no-repeat base')
    ids=rng.choice(len(records),size=base,replace=False).tolist()
    sources=rng.integers(ues,size=base).tolist()
    if repeat!='none':
        if repeat=='cross' and ues<2:raise ValueError('cross-source repeats require2UEs')
        for pos in rng.integers(base,size=count-base):
            ids.append(ids[pos]);sources.append(sources[pos] if repeat=='local' else (sources[pos]+1)%ues)
    perm=rng.permutation(count);ids=[ids[i] for i in perm];sources=np.asarray([sources[i] for i in perm])
    window=count/load;arrival=np.sort(rng.uniform(0,window,size=count));arrival[:min(8,count)]=0
    availability={};sessions=np.full(count,seed,dtype=int)
    if churn:
        for src in range(ues):
            join=0. if src in set(map(int,sources[:min(8,count)])) or src<2 or rng.random()<.6 else float(rng.uniform(.05,.3)*window)
            leave=float(rng.uniform(.8,1.)*window)
            if rng.random()<.5:
                pause=float(rng.uniform(.35,.45)*window);resume=float(rng.uniform(.5,.65)*window)
                reset=bool(rng.random()<.5)
                availability[src]=[(join,pause,seed),(resume,leave,seed+1 if reset else seed)]
            else:availability[src]=[(join,leave,seed)]
        for k,src in enumerate(sources):
            intervals=availability[int(src)]
            lengths=np.array([end-start for start,end,_ in intervals]);part=int(rng.choice(len(intervals),p=lengths/lengths.sum()))
            start,end,session=intervals[part];arrival[k]=rng.uniform(start,end);sessions[k]=session
        initial=[src for src,intervals in availability.items() if intervals[0][0]==0.]
        for k in range(min(8,count)):
            arrival[k]=0.;sessions[k]=seed
    if burst:
        # Concentrate70% at two burst anchors; retain UE availability by projecting into its active intervals.
        chosen=rng.choice(np.arange(min(8,count),count),size=round(.7*max(0,count-min(8,count))),replace=False)
        for k in chosen:
            t=float(rng.choice((0.,.5))*window+rng.uniform(0,.1)*window)
            if churn:
                intervals=availability[int(sources[k])]
                valid=[(max(t,start),end,session) for start,end,session in intervals if end>=t]
                if valid:arrival[k],end,sessions[k]=valid[0]
                else:arrival[k]=intervals[-1][1];sessions[k]=intervals[-1][2]
            else:arrival[k]=t
    if churn or burst:
        order=np.argsort(arrival,kind='stable');arrival=arrival[order];sources=sources[order];sessions=sessions[order];ids=[ids[k] for k in order]
    profiles=rng.integers(len(PROFILES),size=ues)
    banks={};messages=[];tiers=[];facts=[];services=[];failures=[];assets={};rel_avail=nov_avail=0
    for k,(idx,a,src) in enumerate(zip(ids,arrival,sources)):
        row=records[idx];x=np.array(features[row['image_id']],dtype=float,copy=True);v=embeddings.get(row['image_sha256'])
        past=banks.setdefault((int(src),int(sessions[k])),deque(maxlen=128))
        x=received_features(x,v,past,None if text_vectors is None else text_vectors[int(profiles[src])])
        rel_avail+=bool(x[384]);nov_avail+=bool(x[385])
        past.append(v)
        messages.append(Message(k,int(src),int(sessions[k]),float(a),k,x,profile_id=row.get('_profile_id','fixture')))
        tiers.append(row.get('reference_tier',-1));facts.append(row['image_sha256']);assets[k]=row['image_id']
        services.append(float(rng.lognormal(-.125,.5)) if variable_service else 1.)
        failures.append(bool(rng.random()<failure_rate))
    return Episode(tuple(messages),tuple(tiers),tuple(facts),tuple(services),tuple(failures),float(window+32),
        {'seed':seed,'ue_count':ues,'churn':churn,'burst':burst,'ue_active_intervals':availability,'capture_metadata_scope':'synthetic times; unrelated IID-image geography unavailable',
         'capture_metadata':{m.token:{'capture_time':max(0.,m.arrival-.2),'extraction_ready_time':max(0.,m.arrival-.1),'receiver_arrival_time':m.arrival,'timestamp_status':'simulated','latitude':None,'longitude':None,'location_status':'unavailable_for_unrelated_IID_images'} for m in messages},'profiles':profiles.tolist(),'missions':[list(p) for p in PROFILES],
         'mode':'MIXED_IID; artificial chronology; exact-copy correspondence only','assets':assets,
         'input_profile':records[0].get('_profile_identity',{'schema':'fixture','sha256':'fixture'}),
         'mission_relevance_available':int(rel_avail),'novelty_available':int(nov_avail),'sender_kb':'pre-offer FIFO128, cosine query then commit; policy-independent'})
