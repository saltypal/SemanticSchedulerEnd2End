"""Small/large grouped experiments; gold facts never enter policy observations."""
from dataclasses import replace
from pathlib import Path
import csv
import itertools
import json
import math
import time
import numpy as np

from .artifacts import write_json
from .contracts import MISSIONS,GROUPS,MethodSpecification,ParameterValue,flatten_bundle,canonical,digest
from .extractor import SemanticParameterExtractor,SenderMemory
from .evaluation import build_episode,replay,training_decisions,grouped_split,feature_subsets
from .learning import ClaimHeads,fit_scorer
from .serialization import HeaderCodec


class BundleFactory:
    def __init__(self,evidence,mission,specification):
        self.evidence,self.mission,self.specification=evidence,mission,specification
        self.extractor=SemanticParameterExtractor(specification);self.begin()
    def __contains__(self,key):return key in self.evidence
    def begin(self):
        self.memories={i:SenderMemory(str(i),self.mission.mission_id,self.specification.kb_capacity) for i in range(4)}
    def for_observation(self,record,sequence,token):
        source=sequence%4;memory=self.memories[source];evidence=self.evidence[record['image_id']]
        metadata={'image_token':token,'source_id':str(source),'sequence':sequence}
        bundle=self.extractor.extract_evidence(evidence,self.mission,metadata,memory)
        canonical(bundle.to_dict()) # The analysis header is serializable before advancing sender state.
        memory.commit(evidence,sequence)
        return bundle


def episodes_for(records,evidence,mission,specification,count=6,seed=7):
    result=[];missing=[]
    for arm in ('native','relevant','exact_copy'):
        for repetition in range(count):
            factory=BundleFactory(evidence,MISSIONS[mission],specification)
            episode=build_episode(records,factory,mission,seed+repetition,arm)
            if episode is None:
                missing.append({'mission':mission,'arm':arm,'reason':'insufficient independently annotated parents for 32 jobs'})
                break
            result.append(episode)
    return result,missing


def score_episodes(episodes,policy):
    rows=[]
    for episode in episodes:
        result=replay(episode,policy)
        if result is not None:rows.append(result)
    valid=[r['metrics']['A_K'] for r in rows if r['metrics']['A_K'] is not None]
    # Average arms equally so one readily sampled arm does not dominate.
    arms={arm:[r['metrics']['A_K'] for r in rows if r['arm']==arm and r['metrics']['A_K'] is not None]
          for arm in ('native','relevant','exact_copy')}
    defined=[np.mean(v) for v in arms.values() if v]
    return {'mean_A10':None if not defined else float(np.mean(defined)),'defined_episodes':len(valid),
            'total_episodes':len(episodes),'arm_means':{k:None if not v else float(np.mean(v)) for k,v in arms.items()},'rows':rows}


def availability_report(episodes):
    records=[]
    for episode in episodes:
        for observation in episode.observations:
            for group,value in observation.bundle.parameters.items():
                records.append((group,value.available,value.method))
    return {group:{'available_fraction':float(np.mean([v for g,v,_ in records if g==group])),
                   'methods':sorted({m for g,_,m in records if g==group})} for group in sorted({r[0] for r in records})}


def choose_configuration(rows):
    valid=[r for r in rows if r.get('mean_A10') is not None and math.isfinite(r['mean_A10'])]
    if not valid:return None
    best=max(valid,key=lambda r:r['mean_A10'])
    tolerance=max(.005,best.get('standard_error',0.))
    near=[r for r in valid if r['mean_A10']>=best['mean_A10']-tolerance]
    return min(near,key=lambda r:(len(r['groups']),r.get('producer_cost_seconds',0.),r.get('header_bytes',0),-r.get('alpha',0),tuple(r['groups'])))


def local_study(records,evidence,output,budget_callback=lambda:None,episodes=4,backbone='clip'):
    """One grouped development holdout; explicitly provisional, not nested evidence."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    groups=sorted({r['group'] for r in records});rng=np.random.default_rng(7);rng.shuffle(groups)
    # Keep at least 32 native parents where possible for the bounded cohort.
    validation_groups=set(groups[:max(1,int(.4*len(groups)))])
    train=[r for r in records if r['group'] not in validation_groups]
    validation=[r for r in records if r['group'] in validation_groups]
    heads=ClaimHeads().fit(train,evidence)
    predicted=heads.apply(evidence)
    write_json(output/'claim_heads.json',heads.to_dict())
    split={'scope':'development exposed / provisional only','train_groups':sorted({r['group'] for r in train}),
           'validation_groups':sorted(validation_groups),'fresh_final_evaluated':False}
    write_json(output/'split.json',split)
    summary={};all_grid=[];missing=[];artifacts={}
    for mission in ('building_damage','road_damage','affected_people'):
        budget_callback();mission_summary={};best=None;best_policy=None
        # Limited pilot methods. The full method comparison occurs inside grouped cloud selection.
        for relevance,novelty in itertools.product(('similarity','claim_probability'),('embedding','semantic_delta')):
            budget_callback();spec=MethodSpecification(backbone=backbone,relevance=relevance,novelty=novelty,
                producer_revisions={'claim_heads_sha256':digest(heads.to_dict())})
            train_episodes,train_missing=episodes_for(train,predicted,mission,spec,count=episodes,seed=7)
            validation_episodes,val_missing=episodes_for(validation,predicted,mission,spec,count=episodes,seed=1007)
            missing.extend(train_missing+val_missing)
            decisions=training_decisions(train_episodes)
            if not decisions or not validation_episodes:continue
            available=[group for group in GROUPS if any(group==key.split('.')[0] for d in decisions for o in d['ready'] for key in flatten_bundle(o.bundle))]
            subsets=list(dict.fromkeys([('relevance',),('novelty',),('relevance','novelty'),tuple(available)]))
            method_key=relevance+'__'+novelty
            for subset in subsets:
                if not set(subset)<=set(available):continue
                for alpha in (.1,1.,10.):
                    budget_callback()
                    try:policy,fit=fit_scorer(decisions,subset,alpha=alpha,provenance={'scope':'provisional single grouped development split','mission':mission})
                    except ValueError:continue
                    scored=score_episodes(validation_episodes,policy)
                    codec=HeaderCodec(policy.encoder.fields,spec.sha256)
                    row={'mission':mission,'method':method_key,'groups':list(subset),'alpha':alpha,
                         'mean_A10':scored['mean_A10'],'header_bytes':codec.describe()['header_bytes'],**fit}
                    all_grid.append(row);artifacts[(mission,method_key,tuple(subset),alpha)]=(policy,spec,validation_episodes)
            # Baselines use identical episodes within this extraction variant.
            baseline={name:score_episodes(validation_episodes,name) for name in ('fifo','round_robin','random','relevance','novelty','incident_filter','historical_I')}
            mission_summary[method_key]={'baselines':{k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in baseline.items()},
                                        'availability':availability_report(validation_episodes)}
        chosen=choose_configuration([r for r in all_grid if r['mission']==mission])
        if chosen:
            policy,spec,selected_episodes=artifacts[(mission,chosen['method'],tuple(chosen['groups']),chosen['alpha'])]
            scored=score_episodes(selected_episodes,policy)
            spec=replace(spec,enabled_groups=tuple(chosen['groups']),feature_names=policy.encoder.fields)
            projected=[]
            for ep in selected_episodes:
                observations=[]
                for o in ep.observations:
                    parameters={k:(v if k in spec.enabled_groups or k=='reliability' else ParameterValue(None,False,'disabled','disabled by specification')) for k,v in o.bundle.parameters.items()}
                    observations.append(replace(o,bundle=replace(o.bundle,parameters=parameters,specification_sha256=spec.sha256)))
                projected.append(replace(ep,observations=tuple(observations)))
            selected_episodes=projected
            codec=HeaderCodec(policy.encoder.fields,spec.sha256)
            write_json(output/(mission+'_scorer.json'),policy.artifact)
            write_json(output/(mission+'_specification.json'),spec.to_dict())
            write_json(output/(mission+'_trace.json'),scored['rows'])
            wire=codec.encode(selected_episodes[0].observations[0].bundle)
            # Same value registry; the finalized spec digest is a registry identifier, not a label input.
            codec.decode(wire)
            write_json(output/(mission+'_header.json'),codec.describe())
            quantization={}
            from .cloud_study import project_episode
            for bits in (8,16):
                qspec=replace(spec,quantization_bits=bits)
                candidate=HeaderCodec(policy.encoder.fields,qspec.sha256,bits)
                errors=[]
                projected=[project_episode(ep,qspec) for ep in selected_episodes]
                transmitted=[project_episode(ep,qspec,candidate) for ep in selected_episodes]
                for ep in projected:
                    for o in ep.observations:
                        flat=flatten_bundle(o.bundle);decoded=candidate.decode(candidate.encode(o.bundle))['values']
                        errors.extend(abs(flat[k]-v) for k,v in decoded.items())
                quantized_score=score_episodes(transmitted,policy)
                quantization[str(bits)]={'header_bytes':candidate.describe()['header_bytes'],'max_value_error':max(errors,default=0.),
                    'mean_A10':quantized_score['mean_A10'],'difference_from_full':None if scored['mean_A10'] is None or quantized_score['mean_A10'] is None else quantized_score['mean_A10']-scored['mean_A10'],
                    'orders_changed':sum(replay(a,policy)['order']!=replay(b,policy)['order'] for a,b in zip(projected,transmitted))}
            projection_score=score_episodes(selected_episodes,policy)['mean_A10']
            if projection_score!=scored['mean_A10']:raise RuntimeError('field projection changed selected policy')
            summary[mission]={'selected':chosen,'selected_validation':{k:v for k,v in scored.items() if k!='rows'},
                              'header':codec.describe(),'quantization':quantization,'methods':mission_summary}
        else:summary[mission]={'selected':None,'reason':'insufficient annotated parents/classes for this provisional split','methods':mission_summary}
    write_json(output/'grid.json',all_grid);write_json(output/'missing_cells.json',missing)
    write_json(output/'summary.json',summary)
    return {'summary':summary,'grid_rows':len(all_grid),'missing_cells':missing,'split':split,'heads':heads.to_dict()}
