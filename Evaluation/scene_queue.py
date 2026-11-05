"""Scene JSON ingress. Frozen schema; no dependency on a mutable upstream tree."""
import json
import math
from pathlib import Path
from dataclasses import dataclass
from Scheduler.scene_policy import SceneCandidate,SceneContext,canonical,digest
from Evaluation.decoder_queue import ReceivedMessage

SCHEMA_PATH=Path(__file__).resolve().parents[1]/'Archived/owners_before_simplification_20261027/SemParam/schemas/semantic_scene_evidence_v2.schema.json'


def strict_json(raw):
    def pairs(rows):
        result={}
        for k,v in rows:
            if k in result: raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def validate_scene(document):
    from jsonschema import Draft202012Validator
    Draft202012Validator(strict_json(SCHEMA_PATH.read_text())).validate(document)
    forbidden={'native_labels','evaluator_facts','expected','audit','split','published_annotation_reference'}
    def visit(v):
        if isinstance(v,dict):
            if forbidden & v.keys(): raise ValueError('evaluator data in runtime document')
            for child in v.values():visit(child)
        elif isinstance(v,list):
            for child in v:visit(child)
        elif isinstance(v,float) and not math.isfinite(v):raise ValueError('nonfinite scene value')
    visit(document)
    claims=document['scene']['claims'];names=[c['name'] for c in claims]
    if len(set(names))!=len(names):raise ValueError('duplicate claim')
    producers=document['provenance']['producers']
    def refs(value):
        if isinstance(value,dict):
            if 'producer_ref' in value and value['producer_ref'] not in producers:raise ValueError('dangling evidence producer')
            for child in value.values():refs(child)
        elif isinstance(value,list):
            for child in value:refs(child)
    refs(document)
    for c in claims:
        if c['producer_ref'] not in producers:raise ValueError('unknown claim producer')
        s=c.get('score')
        if s and s['kind']=='positive_class_sigmoid' and c['status']=='available':
            if c['value'] is not (s['value']>s['threshold']):raise ValueError('claim threshold differs')
    return document


@dataclass(frozen=True)
class FeatureRegistry:
    claims:tuple
    categories:tuple=()
    similarities:tuple=()

    @property
    def fields(self):
        names=[f'claim/{n}/{s}' for n in self.claims for s in ('value','score')]
        names += ['measurement/context_alignment','measurement/novelty/appearance_distance',
            'measurement/novelty/new_predicted_concept_fraction','measurement/concept_rarity',
            'measurement/complexity','measurement/geographic']
        names += [f'measurement/spatial/{n}' for n in ('target_detections','all_detections','largest_target_box_area_fraction','target_box_union_fraction')]
        names += [f'temporal/{n}' for n in ('age_ms','allowed_delay_ms','completion_slack_ms')]
        names += [f'detection/{n}/{s}' for n in self.categories for s in ('detection_count','largest_box_area_fraction','box_union_fraction','max_detection_score')]
        names += [f'similarity/{n}' for n in self.similarities]
        names += ['diagnostic/ontology_conflict','diagnostic/producer_disagreement']
        return tuple(names)

    @property
    def sha256(self):return digest({'schema':'scene_numeric_v1','fields':self.fields})

    def extract(self,doc):
        data={}
        def put(name,value,status='available'):
            if status=='available':
                if type(value) not in (bool,int,float) or not math.isfinite(value):raise ValueError('invalid numeric feature')
                data[name]=(float(value),True,False)
            else:data[name]=(0.,False,status=='cold_start')
        for c in doc['scene']['claims']:
            put('claim/'+c['name']+'/value',c['value'],c['status'])
            if c.get('score') is not None:put('claim/'+c['name']+'/score',c['score']['value'],c['status'])
        for name in ('context_alignment','novelty','concept_rarity','spatial','complexity','geographic'):
            row=doc['measurements'][name];v=row['value']
            sub=('appearance_distance','new_predicted_concept_fraction') if name=='novelty' else (
                ('target_detections','all_detections','largest_target_box_area_fraction','target_box_union_fraction') if name=='spatial' else ())
            if sub:
                for key in sub:
                    val=v.get(key) if isinstance(v,dict) else None
                    status=row['status'] if row['status']!='available' else ('available' if val is not None else 'unavailable')
                    put(f'measurement/{name}/{key}',val,status)
            else:put('measurement/'+name,v,row['status'])
        temporal=doc['measurements']['temporal']
        for key in ('age_ms','allowed_delay_ms','completion_slack_ms'):
            val=temporal.get(key);put('temporal/'+key,val,'available' if val is not None and temporal.get('available') else 'unavailable')
        detections=doc['scene']['detections']
        for c in detections['value'] or []:
            for key in ('detection_count','largest_box_area_fraction','box_union_fraction','max_detection_score'):
                put(f'detection/{c["category"]}/{key}',c[key])
        signals=doc['similarity_signals']
        for row in signals['value'] or []:put('similarity/'+row['name'],row['value'],signals['status'])
        for kind in ('ontology_conflict','producer_disagreement'):
            put('diagnostic/'+kind,sum(r['kind']==kind for r in doc['diagnostics']))
        rows=[data.get(n,(0.,False,False)) for n in self.fields]
        return tuple(r[0] for r in rows),tuple(r[1] for r in rows),tuple(r[2] for r in rows)


class SceneQueue:
    def __init__(self,policy,registry,mission,accepted_specifications):
        self.policy=policy;self.registry=registry;self.mission=tuple(mission)
        if not isinstance(accepted_specifications,dict):raise ValueError('specification-to-producer fingerprint map required')
        self.specifications=dict(accepted_specifications)
        if not self.specifications:raise ValueError('producer specification allowlist required')
        self.reset()

    def __len__(self):return len(self.jobs)

    def reset(self):
        self.jobs={};self.seen=set();self.clock=0.;self.unknown_predicates={};self.policy.reset()

    def enqueue(self,message):
        if not isinstance(message,ReceivedMessage) or not message.complete:raise ValueError('complete message required')
        if message.message_id in self.seen:raise ValueError('duplicate full transport identity')
        doc=validate_scene(strict_json(message.header_bytes))
        if doc['observation']['source_id']!=message.source_id:raise ValueError('source mismatch')
        if (doc['mission']['id'],doc['mission']['version'])!=self.mission:raise ValueError('mission mismatch')
        if doc['provenance']['extraction_specification'] not in self.specifications:raise ValueError('unregistered producer specification')
        if digest(doc['provenance']['producers'])!=self.specifications[doc['provenance']['extraction_specification']]:
            raise ValueError('producer identity/settings differ from frozen specification')
        unknown={'claim/'+c['name'] for c in doc['scene']['claims'] if c['name'] not in self.registry.claims}
        unknown|={'category/'+c['category'] for c in doc['scene']['detections']['value'] or [] if c['category'] not in self.registry.categories}
        unknown|={'similarity/'+c['name'] for c in doc['similarity_signals']['value'] or [] if c['name'] not in self.registry.similarities}
        for name in unknown:self.unknown_predicates[name]=self.unknown_predicates.get(name,0)+1
        values,available,cold=self.registry.extract(doc)
        candidate=SceneCandidate(message.message_id,message.source_id,message.arrival_time,message.sequence,
                                 values,available,cold,self.registry.sha256)
        self.jobs[message.message_id]=(message,candidate);self.seen.add(message.message_id)

    def ready_snapshot(self,now,trusted_utc=None):
        if not math.isfinite(now) or now<self.clock:raise ValueError('nonmonotonic scene clock')
        self.clock=now
        return tuple(c for m,c in self.jobs.values() if m.arrival_time<=now)

    def take(self,selected_id,snapshot):
        if selected_id not in {c.message_id for c in snapshot} or selected_id not in self.jobs:
            raise RuntimeError('stale/foreign scene selection')
        return self.jobs.pop(selected_id)[0]


def context_builder(registry,horizon):
    return lambda ready,history,attempts,admitted,now:SceneContext(
        tuple(ready),tuple(history),attempts,admitted,now,horizon,registry.sha256)
