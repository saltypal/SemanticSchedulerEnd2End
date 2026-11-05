"""Frozen upstream ingestion, causal stream construction and reference joins."""
from pathlib import Path
import copy
import hashlib
import importlib
import json
import sys
import numpy as np
from Scheduler.scene_policy import ScenePolicy,canonical,digest
from Evaluation.scene_queue import FeatureRegistry,validate_scene
from Evaluation.decoder_queue import ReceivedMessage
from Evaluation.scene_environment import SceneEpisode,SceneEnvironment
from Scheduler.training.scene_learners import Example

CLAIMS=('bridge.present','building.affected_or_greater','building.flooded','building.minor_or_greater',
        'building.present','debris.present','flooding.present','road.damaged','road.present',
        'tree.damaged','tree.present','water.present')
CATEGORIES=('building','road','person','car','truck','vehicle','debris','tree','water','bridge',
            'chair','couch','dining table','television','refrigerator')


def default_registry():
    # Explicit versioned upstream-vocabulary allowlist; never fit field names on evaluation data.
    return FeatureRegistry(CLAIMS,CATEGORIES,())


def load_frozen_upstream(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'upstream_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        p=root/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('upstream snapshot checksum differs')
    package=str(root)
    if package not in sys.path:sys.path.insert(0,package)
    module=importlib.import_module('parameters.scene_json')
    if not Path(module.__file__).resolve().is_relative_to(root):raise ValueError('mutable upstream import rejected')
    return module


def assert_roles(rows):
    ownership={}
    for row in rows:
        for kind,key in (('parent',row['parent_id']),('group',row['group']),('image',row['image_id'])):
            if (kind,key) in ownership and ownership[kind,key]!=row['role']:raise ValueError('split leakage')
            ownership[kind,key]=row['role']


def split_groups(rows,seed=1729):
    """Group first; apportion 60/20/20 with deterministic largest remainders."""
    by_group={}
    for row in rows:
        by_group.setdefault(row['group'],[]).append(row)
    strata={0:[],1:[],2:[]}
    for group,members in sorted(by_group.items()):
        labels={r['stratum'] for r in members}
        if len(labels)!=1:continue
        strata[next(iter(labels))].append(group)
    rng=np.random.default_rng(seed);selected=[]
    total=sum(map(len,strata.values()))
    def quotas(fraction,capacity):
        result={s:min(capacity[s],int(len(gs)*fraction)) for s,gs in strata.items()}
        remainder=round(total*fraction)-sum(result.values())
        order=sorted(strata,key=lambda s:(-(len(strata[s])*fraction-result[s]),s))
        for s in order:
            if remainder and result[s]<capacity[s]:result[s]+=1;remainder-=1
        if remainder:raise ValueError('insufficient independent groups for role allocation')
        return result
    training=quotas(.6,{s:len(gs) for s,gs in strata.items()})
    validation=quotas(.2,{s:len(gs)-training[s] for s,gs in strata.items()})
    for stratum,groups in strata.items():
        rng.shuffle(groups)
        ntrain=training[stratum];nval=validation[stratum]
        for i,g in enumerate(groups):
            role='train' if i<ntrain else ('validation' if i<ntrain+nval else 'pilot_eval')
            selected.extend([{**r,'role':role} for r in by_group[g]])
    assert_roles(selected)
    return selected


class CausalCorpus:
    def __init__(self,root,upstream_root):
        self.root=Path(root);self.module=load_frozen_upstream(upstream_root)
        self.rows=json.loads((self.root/'records.json').read_text())
        assert_roles(self.rows)
        feature_path=self.root/'features.jsonl'
        features=[json.loads(line) for line in feature_path.read_text().splitlines() if line.strip()]
        from parameters.extractor import VisualEvidence
        self.evidence={r['image_id']:VisualEvidence.from_dict(r['evidence']) for r in features}
        from parameters.ue_profiles import load_ue_profiles
        missions,_=load_ue_profiles();self.mission=missions['building_damage']
        self.ontology=json.loads((Path(upstream_root)/'parameters/config/scene_ontology_ladi_v2.json').read_text())
        self.registry=default_registry()

    def make(self,role,seed,*,pattern='no_repeat',load='congested',variable_service=False):
        from parameters.pilot_parameters import PilotMemory,PilotParameterExtractor
        pool=[r for r in self.rows if r['role']==role and r['image_id'] in self.evidence]
        if not pool:raise ValueError('no usable role cohort')
        n=48 if pattern=='no_repeat' else 32
        if len({r['parent_id'] for r in pool})<n:raise ValueError('insufficient independent parents for 48-message trace')
        rng=np.random.default_rng(seed);chosen=[pool[int(i)] for i in rng.choice(len(pool),n,replace=False)]
        offered=[(r,i%4) for i,r in enumerate(chosen)]
        if pattern!='no_repeat':
            for i in rng.choice(n,16,replace=False):
                source=int(i)%4 if pattern=='local_repeat' else (int(i)+1)%4
                offered.append((chosen[int(i)],source))
        order=rng.permutation(len(offered));offered=[offered[int(i)] for i in order]
        window={'light':80.,'congested':32.,'burst':20.}[load]
        arrivals=[0.]*8+sorted(rng.uniform(0,window,40).tolist())
        memories={i:PilotMemory('ue'+str(i),self.mission.mission_id) for i in range(4)}
        extractor=PilotParameterExtractor(provider=None)
        messages=[];facts={};durations={};specifications={};by_sender={i:0 for i in range(4)}
        for k,((row,source),arrival) in enumerate(zip(offered,arrivals)):
            evidence=self.evidence[row['image_id']];token='trace-'+str(seed)+'-'+str(k)
            metadata={'source_id':'ue'+str(source),'image_token':token}
            bundle=extractor.extract_evidence(evidence,self.mission,metadata,memories[source])
            document,_=self.module.build_scene_document(bundle,evidence,self.mission,
                source_id=metadata['source_id'],image_id=row['image_id'],ontology=self.ontology)
            validate_scene(document)
            message=ReceivedMessage(metadata['source_id'],str(seed),k,canonical(document).encode(),
                row['image_id'],True,float(arrival),k)
            memories[source].commit(evidence,by_sender[source],{'image_id':row['image_id']});by_sender[source]+=1
            messages.append(message);specifications[bundle.specification_sha256]=digest(document['provenance']['producers'])
            facts[message.message_id]={row['image_id']} if row['damage_positive'] else set()
            if variable_service:
                qualified=int(hashlib.sha256((row['image_id']+str(seed)+str(k)).encode()).hexdigest()[:16],16)
                durations[message.message_id]=float(np.random.default_rng(qualified).lognormal(-.125,.5))
            else:durations[message.message_id]=1.
        weights={f:1. for fs in facts.values() for f in fs}
        metadata={row['image_id']:{'official_source_split':row.get('source_split','unknown')}
                  for row in chosen if row['image_id'] in weights}
        ep=SceneEpisode(tuple(messages),facts,durations,window+16.,weights,
            scope='LADI positive source-observation; exact-copy control',fact_metadata=metadata)
        return SceneEnvironment(ep,self.registry,(self.mission.mission_id,self.mission.version),specifications)


def collect_examples(factory,count,seed=1729,collection_policy='random_damage_v1'):
    if collection_policy not in ('random_damage_v1','random_oracle_v1'):
        raise ValueError('unknown demonstration collection policy')
    rng=np.random.default_rng(seed);rows=[]
    for index in range(count):
        env=factory(index)
        damage=env.registry.fields.index('claim/building.affected_or_greater/score') if collection_policy=='random_damage_v1' else None
        predicted=ScenePolicy(kind='single',field=damage)
        while not env.finished:
            ctx=env.observation();gains=env.marginal_gains()
            rows.append(Example(ctx,gains,str(index),collection_policy=collection_policy))
            if index%2==0:choice=int(rng.integers(len(gains)))
            elif collection_policy=='random_oracle_v1':choice=int(np.argmax(gains))
            else:
                identity=predicted.select(ctx).message_id
                choice=next(i for i,c in enumerate(ctx.ready_candidates) if c.message_id==identity)
            env.step(ctx.ready_candidates[choice].message_id)
    return rows
