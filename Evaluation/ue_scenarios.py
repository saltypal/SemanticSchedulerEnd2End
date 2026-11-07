"""Mission-conditioned UE offers and controlled incident streams on the shared decoder environment."""
from dataclasses import dataclass
from collections import deque
import math
import numpy as np
from Evaluation.semantic_learning_env import Episode, Message, PROFILES, received_features

KINDS=("routine","exact_repeat","additional_view","condition_change","distractor")

@dataclass(frozen=True)
class StreamConfig:
    mode: str="mixed"
    ues: int=12
    duration: float=200.
    service_time: float=1.
    load: float=.9
    burst_factor: float=1.
    mission_fraction: float=.8
    repeat_probability: float=.2
    capture_to_arrival: float=.1
    tail: float=16.
    seed: int=1729
    time_unit: str="synthetic_time_unit"

    def __post_init__(self):
        if self.mode not in ("mixed","mission","incident"):raise ValueError("unknown stream mode")
        if type(self.ues) is not int or self.ues<1:raise ValueError("positive UE count required")
        for name in ("duration","service_time","load","burst_factor"):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError("positive finite "+name)
        if self.burst_factor<1:raise ValueError("burst factor must be at least1")
        for name in ("mission_fraction","repeat_probability"):
            if not math.isfinite(getattr(self,name)) or not 0<=getattr(self,name)<=1:raise ValueError("probability outside0..1")
        if any(not math.isfinite(getattr(self,n)) or getattr(self,n)<0 for n in ("capture_to_arrival","tail")):raise ValueError("invalid timing")


def validate_assets(records, annotations):
    """Annotation manifest is evaluator/scenario state, never an actor observation."""
    known={r["image_id"] for r in records}
    result={}
    for row in annotations:
        if row["image_id"] not in known or row["image_id"] in result:raise ValueError("unknown/duplicate scenario image")
        kind=row["kind"]
        if kind not in KINDS or kind=="exact_repeat":raise ValueError("repeats reference existing source observations")
        if not row.get("provenance"):raise ValueError("scenario evidence provenance required")
        if any(m not in {p[0] for p in PROFILES} for m in row.get("missions",[])):raise ValueError("unknown mission")
        if kind in ("additional_view","condition_change"):
            if not row.get("incident_id") or not row.get("sequence_id") or not row.get("evidence_id"):
                raise ValueError("related/change observations require incident/sequence/evidence identities")
            if not math.isfinite(row.get("sequence_order",float("nan"))):raise ValueError("ordered sequence required")
            if kind=="condition_change" and not row.get("condition"):raise ValueError("condition annotation required")
        if "utility_by_mission" in row:
            if row.get("reference_kind") not in ("published_annotation","controlled_construction"):
                raise ValueError("independent reference provenance required")
            if not row.get("reference_source"):raise ValueError("reference source required")
            for mission,value in row["utility_by_mission"].items():
                if mission not in {p[0] for p in PROFILES} or type(value) not in (int,float) or value not in (0,1):
                    raise ValueError("binary supported mission utility required")
        result[row["image_id"]]=dict(row)
    return result


def build_stream(records, features, embeddings, mission_vectors, config, annotations=()):
    if not records:raise ValueError("empty image pool")
    if len({r["image_id"] for r in records})!=len(records):raise ValueError("duplicate image identities")
    c=config
    timing,content,profiles_rng=[np.random.default_rng(s) for s in np.random.SeedSequence(c.seed).spawn(3)]
    assets=validate_assets(records,annotations)
    if c.mode=="incident" and not any(a["kind"]=="condition_change" for a in assets.values()):
        raise ValueError("incident mode requires supported ordered change annotations; do not infer from similarity")
    profile_ids=profiles_rng.integers(len(PROFILES),size=c.ues)
    pools={}
    for p in set(profile_ids):
        matched=[r for r in records if PROFILES[p][0] in assets.get(r["image_id"],{}).get("missions",[])]
        if c.mode!="mixed" and not matched:raise ValueError("missing mission pool: "+PROFILES[p][0])
        pools[int(p)]=matched
    # Two-level rate process, normalized to preserve expected total load.
    # Demand increases during the middle20% without changing mean offered load.
    norm=.8+.2*c.burst_factor
    segments=((0.,.4*c.duration,1/norm),(.4*c.duration,.6*c.duration,c.burst_factor/norm),
              (.6*c.duration,c.duration,1/norm))
    offers=[]
    intervals={}
    for ue in range(c.ues):
        # A UE may join late, pause, then resume. Pending received jobs survive.
        start=0. if ue<2 or timing.random()<.7 else float(timing.uniform(0,.2*c.duration))
        pause=.3*c.duration;resume=.35*c.duration
        windows=[(start,pause),(resume,c.duration)] if timing.random()<.3 else [(start,c.duration)]
        intervals[ue]=windows
        for left,right,multiplier in segments:
            for a,b in windows:
                lo=max(left,a);hi=min(right,b)
                if hi<=lo:continue
                count=int(timing.poisson(c.load/(c.ues*c.service_time)*multiplier*(hi-lo)))
                offers.extend((float(t),ue) for t in timing.uniform(lo,hi,size=count))
    offers.sort()
    last={};banks={};messages=[];tiers=[];facts=[];events=[];weights=[]
    # All incident UEs share one explicitly annotated scenario, never a guessed cross-view match.
    incident_ids=sorted({a["incident_id"] for a in assets.values() if a.get("incident_id")})
    selected_incident=incident_ids[int(content.integers(len(incident_ids)))] if incident_ids else None
    for token,(capture,ue) in enumerate(offers):
        p=int(profile_ids[ue]);mission=PROFILES[p][0]
        phase="routine" if capture<.4*c.duration else "incident" if capture<.6*c.duration else "followup"
        pool=pools[p] if c.mode!="mixed" and content.random()<c.mission_fraction else records
        kind="routine";annotation={}
        if c.mode=="incident" and phase!="routine" and content.random()<c.mission_fraction:
            incident=selected_incident
            all_sequence=[a for a in assets.values() if a.get("incident_id")==incident and a["kind"] in ("additional_view","condition_change")]
            orders=sorted({a["sequence_order"] for a in all_sequence})
            progress=min(.999999,max(0.,(capture/c.duration-.4)/.6))
            current_order=orders[min(int(progress*len(orders)),len(orders)-1)]
            sequence=[r for r in pools[p] if assets.get(r["image_id"],{}).get("incident_id")==incident
                      and assets[r["image_id"]]["kind"] in ("additional_view","condition_change")
                      and assets[r["image_id"]]["sequence_order"]==current_order]
            if not sequence:raise ValueError("mission lacks support for incident stage: "+mission)
            row=sequence[int(content.integers(len(sequence)))];annotation=assets[row["image_id"]];kind=annotation["kind"]
        else:
            if c.mode=="incident":
                pool=[r for r in pool if assets.get(r["image_id"],{}).get("kind")=="routine" or assets.get(r["image_id"],{}).get("kind")=="distractor"]
                if not pool:raise ValueError("routine pre-incident observations required")
            row=pool[int(content.integers(len(pool)))];annotation=assets.get(row["image_id"],{});kind=annotation.get("kind","routine")
        # Repetition cannot revive an obsolete incident condition after a declared change.
        if ue in last and content.random()<c.repeat_probability and (c.mode!="incident" or row["image_id"]==last[ue]["image_id"]):
            row=last[ue];annotation=assets.get(row["image_id"],{});kind="exact_repeat"
        if ue in last and row["image_sha256"]==last[ue]["image_sha256"]:kind="exact_repeat"
        last[ue]=row
        past=banks.setdefault(ue,deque(maxlen=128))
        vector=embeddings.get(row["image_sha256"])
        x=received_features(features[row["image_id"]],vector,past,None if mission_vectors is None else mission_vectors[p])
        past.append(vector)
        arrival=capture+c.capture_to_arrival
        messages.append(Message(token,ue,0,arrival,token,x,profile_id=row.get("_profile_id","fixture")))
        tiers.append(row.get("reference_tier",-1))
        utility=annotation.get("utility_by_mission",{}).get(mission)
        weights.append(utility)
        facts.append((row["image_sha256"]+":"+mission) if utility is not None else row["image_sha256"])
        events.append({"token":token,"image_id":row["image_id"],"ue":ue,"mission":mission,"capture":capture,
                       "arrival":arrival,"phase":phase,"kind":kind,"annotation":annotation})
    horizon=c.duration+c.capture_to_arrival+c.tail
    metadata={"schema":"ue_scenario_v1","mode":c.mode,"incident_id":selected_incident,"config":vars(c),"profiles":profile_ids.tolist(),
              "missions":[list(p) for p in PROFILES],"ue_active_intervals":intervals,"offered_trace":events,
              "input_profile":records[0].get("_profile_identity",{"schema":"fixture","sha256":"fixture"}),
              "expected_aggregate_rate_if_all_active":c.load/c.service_time,
              "expected_per_ue_rate_if_active":c.load/(c.ues*c.service_time),
              "realized_rate":len(messages)/c.duration,"offered_service_load":len(messages)*c.service_time/c.duration,
              "scope":"simulated capture/arrival times; scene linkage only from declared manifest; no physical mobility/GPS inferred"}
    if any(w is not None for w in weights) and not all(w is not None for w in weights):
        raise ValueError('incomplete independent mission references for offered stream')
    return Episode(tuple(messages),tuple(tiers),tuple(facts),(c.service_time,)*len(messages),(False,)*len(messages),horizon,metadata,
        tuple(weights) if weights and all(w is not None for w in weights) else None)
