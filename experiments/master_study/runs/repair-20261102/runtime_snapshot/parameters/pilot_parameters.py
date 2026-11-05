"""Annotation-free producer summaries and causal UE history for the new pilot."""
from collections import Counter
from copy import deepcopy
import math
import numpy as np
from .contracts import ParameterValue,canonical,digest
from .extractor import SemanticParameterExtractor,SenderMemory
from .image_claims import image_claim_predictions


ALIASES={'tv':'television','tvmonitor':'television','sofa':'couch','diningtable':'dining table',
         'fridge':'refrigerator','automobile':'car','people':'person','buildings':'building',
         'roads':'road','cars':'car','trucks':'truck','buses':'bus'}
VEHICLES={'car','truck','bus','motorcycle','bicycle','van'}
VOCABULARY=('building','road','person','car','truck','bus','motorcycle','bicycle',
            'debris','chair','couch','dining table','television','refrigerator','tree','water')


def context_similarity_key(mission):
    return 'mission_context:'+mission.version


def canonical_label(label):
    value=str(label).strip().lower().rstrip('.')
    return ALIASES.get(value,value)


def matches_target(label,targets):
    label=canonical_label(label);targets={canonical_label(t) for t in targets}
    return label in targets or ('vehicle' in targets and label in VEHICLES)


def predicted_concepts(evidence):
    rows=detections(evidence);claims=image_claim_predictions(evidence)
    if rows is None and not claims:return None
    concepts={r['label'] for r in rows or []}
    concepts.update(key.removesuffix('.present') if key.endswith('.present') else 'condition:'+key
                    for key,row in claims.items() if row['value'])
    return concepts


def concept_signature(evidence):
    """Only compare histories with the same available producers and ontology."""
    rows=detections(evidence);claims=image_claim_predictions(evidence)
    if rows is None and not claims:return None
    detector=evidence.provenance.get('detector_provenance',{})
    image_producer=evidence.provenance.get('image_claim_producer',{})
    return digest({'detector_available':rows is not None,
        'detector':{k:detector.get(k) for k in ('weights_sha256','vocabulary_sha256','settings_sha256','conf_threshold')},
        'image_producer':{k:image_producer.get(k) for k in ('model_id','revision','weights_sha256','preprocessing_sha256')},
        'claim_methods':{key:{k:row.get(k) for k in ('method','threshold')} for key,row in claims.items()}})


def detections(evidence,threshold=None):
    if evidence.statuses.get('detector')!='ok' or evidence.width<=0 or evidence.height<=0:return None
    if threshold is None:threshold=evidence.provenance.get('detector_provenance',{}).get('conf_threshold',.25)
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<threshold<=1:
        raise ValueError('invalid effective detector confidence threshold')
    output=[]
    for row in evidence.detections:
        try:
            score=float(row['score']);box=list(map(float,row['box']))
            if len(box)!=4 or not all(math.isfinite(v) for v in box+[score]) or not 0<=score<=1 or score<threshold:continue
            x0,x1=np.clip([box[0],box[2]],0,evidence.width)
            y0,y1=np.clip([box[1],box[3]],0,evidence.height)
            if x1>x0 and y1>y0:
                output.append({'label':canonical_label(row['label']),'score':score,
                    'box':[float(x0),float(y0),float(x1),float(y1)]})
        except (KeyError,TypeError,ValueError):continue
    return output


def union_area(boxes):
    """Exact axis-aligned rectangle union in input coordinate units."""
    if not boxes:return 0.
    xs=sorted({b[i] for b in boxes for i in (0,2)});area=0.
    for left,right in zip(xs,xs[1:]):
        intervals=sorted((b[1],b[3]) for b in boxes if b[0]<right and b[2]>left)
        total=0.;end=None
        for lower,upper in intervals:
            if end is None or lower>end:total+=upper-lower;end=upper
            elif upper>end:total+=upper-end;end=upper
        area+=(right-left)*total
    return float(area)


class PilotMemory(SenderMemory):
    """One owner; sidecars contain only past observation references/predictions."""
    def commit(self,evidence,sequence,observation=None):
        observation={} if observation is None else observation
        allowed={'observation_id','image_id','image_path','mission_version'}
        if set(observation)-allowed:raise ValueError('unexpected field in observation sidecar')
        if any(not isinstance(v,str) or not v for v in observation.values()):raise ValueError('observation references must be nonempty strings')
        concepts=predicted_concepts(evidence)
        sidecar={'observation':deepcopy(observation),'concepts_available':concepts is not None,
                 'concepts':sorted(concepts or []),'concept_signature':concept_signature(evidence),
                 'producer':deepcopy(evidence.provenance)}
        canonical(sidecar) # Fail before mutating the KB.
        super().commit(evidence,sequence)
        self.entries[-1].update(sidecar)

    def history(self):
        return [{'observation':deepcopy(r.get('observation',{})),
                 'concepts':list(r.get('concepts',[])),
                 'concepts_available':r.get('concepts_available',False),
                 'concept_signature':r.get('concept_signature')} for r in self.entries]


class PilotParameterExtractor(SemanticParameterExtractor):
    """Reuse existing R/time/geofence; explicitly version the new concept methods."""
    def extract_context_relevance(self,evidence,mission):
        if self.specification.relevance=='claim_probability':
            return super().extract_context_relevance(evidence,mission)
        key=context_similarity_key(mission)
        if key not in evidence.similarities:
            # Historical cached evidence contains condition clauses only.
            return super().extract_context_relevance(evidence,mission)
        cosine=float(evidence.similarities[key])
        if not math.isfinite(cosine) or not -1.00001<=cosine<=1.00001:
            return self.missing('operator_paragraph_cosine_v1','invalid operator-paragraph cosine')
        return ParameterValue(float((np.clip(cosine,-1,1)+1)/2),True,
            'operator_paragraph_cosine_v1',supporting={'text_source':'operator_paragraph',
                'paragraph_sha256':digest(mission.paragraph),'raw_cosine':cosine,'is_probability':False})

    def extract_novelty(self,evidence,mission,memory):
        result=super().extract_novelty(evidence,mission,memory)
        concepts=predicted_concepts(evidence)
        if result.available and concepts is not None:
            valid=[r for r in memory.entries if r.get('concepts_available') and r.get('concept_signature')==concept_signature(evidence)]
            seen=set().union(*(set(r['concepts']) for r in valid)) if valid else set()
            value=dict(result.value)
            comparable=bool(valid) or not memory.entries
            value['new_concept_fraction']=len(concepts-seen)/len(concepts) if concepts and comparable else None
            return ParameterValue(value,True,'appearance_and_predicted_concept_delta_v2',
                supporting={**result.supporting,'asset_identity_verified':False,
                            'concept_history_count':len(valid),'concept_delta_not_new_asset_truth':True,
                            'concept_delta_available':bool(concepts) and comparable,
                            'concept_delta_reason':None if concepts and comparable else
                                ('no comparable concept-producer history' if not comparable else 'no positive predicted concept')})
        return result

    def extract_scarcity(self,evidence,mission,memory):
        concepts=predicted_concepts(evidence)
        if concepts is None:return self.missing('concept_bernoulli_surprisal_v2','image concepts and detector unavailable')
        if not concepts:return self.missing('concept_bernoulli_surprisal_v1','no predicted positive concept')
        prior=[r for r in memory.entries if r.get('concepts_available') and r.get('concept_signature')==concept_signature(evidence)]
        if memory.entries and not prior:
            return self.missing('concept_bernoulli_surprisal_v2','no comparable concept-producer history')
        count=len(prior)
        probabilities={c:(sum(c in r['concepts'] for r in prior)+1)/(count+2) for c in sorted(concepts)}
        raw={c:-math.log(p) for c,p in probabilities.items()}
        return ParameterValue(float(np.mean(list(raw.values()))/math.log(count+2)),True,
            'concept_bernoulli_surprisal_v2',supporting={'reference':'causal predicted UE observations',
                'concept_sources':'positive detections and optional native image claims',
                'valid_history_count':count,'probabilities':probabilities,'surprisal_nats':raw,
                'cold_start':count==0,'not_importance':True})

    def extract_spatial_evidence(self,evidence,mission):
        rows=detections(evidence)
        if rows is None:return self.missing('canonical_target_box_extent_v1','detector unavailable')
        boxes=[r['box'] for r in rows if matches_target(r['label'],mission.target_entities)]
        pixels=evidence.width*evidence.height
        areas=[(b[2]-b[0])*(b[3]-b[1])/pixels for b in boxes]
        return ParameterValue({'largest_area':max(areas,default=0.),
            'union_area':min(1.,union_area(boxes)/pixels),'target_count':min(len(boxes),100)/100},True,
            'canonical_target_box_extent_v1',supporting={'raw_target_count':len(boxes),
                 'physical_distance_available':False,'extent_kind':'bounding boxes,not masks'})

    def extract_scene_complexity(self,evidence,mission):
        rows=detections(evidence)
        if rows is None:return self.missing('task_count_coverage_v1','detector unavailable')
        counts=Counter(r['label'] for r in rows);n=len(rows)
        targets=[r['box'] for r in rows if matches_target(r['label'],mission.target_entities)]
        entropy=0. if n==0 or len(counts)<=1 else -sum((v/n)*math.log(v/n) for v in counts.values())/math.log(len(counts))
        return ParameterValue({'target_count_scaled':min(len(targets),100)/100,
            'target_box_coverage':min(1.,union_area(targets)/(evidence.width*evidence.height)),
            'class_diversity':entropy,'all_count_scaled':min(n,100)/100},True,'task_count_coverage_v1',
            supporting={'raw_count':n,'raw_target_count':len(targets),'class_counts':dict(counts),
                        'not_semantic_depth_truth':True})

    def extract_scene_evidence(self,evidence):
        base=super().extract_scene_evidence(evidence);value=dict(base.value or {})
        rows=detections(evidence)
        if rows is not None:
            present={r['label'] for r in rows}
            # Detection supports a positive observation; no box does not prove absence.
            value['entity_presence']={c:True for c in present}
            searched=[canonical_label(c) for c in evidence.provenance.get('detector_provenance',{}).get('vocabulary',VOCABULARY)]
            value['detector_observation']={c:'detected' if c in present else 'not_detected' for c in searched}
        claims=image_claim_predictions(evidence)
        if claims:
            presence=value.setdefault('entity_presence',{})
            value['producer_conflicts']=[{'fact':key,'detector_observed':True,'image_classifier_value':False}
                for key,row in claims.items() if key.endswith('.present') and not row['value']
                and key.removesuffix('.present') in presence]
            value['presence_fusion_policy']='native_classifier_priority; conflicts retained, no silent OR fusion'
            presence.update({key.removesuffix('.present'):row['value'] for key,row in claims.items() if key.endswith('.present')})
            value['image_claim_predictions']=claims
            value['condition_predictions']={key:row['value'] for key,row in claims.items() if not key.endswith('.present')}
        contrasts=evidence.provenance.get('condition_contrasts',{})
        if contrasts:
            value['condition_contrast_evidence']={key:{**row,'is_native_label_probability':False,
                'margin':float(row['positive_similarity'])-float(row['negative_similarity'])} for key,row in contrasts.items()}
        if not value:return self.missing('structured_scene_candidate_v1','no predicted scene evidence')
        return ParameterValue(value,True,'structured_scene_candidate_v2',supporting={
            'detector_non_detection_is_not_absence':True,'raw_CLIP_contrasts_are_not_binary_native_claims':True})

    def extract_reliability(self,evidence):
        rows=detections(evidence)
        claims=image_claim_predictions(evidence)
        if rows is None and not claims:return self.missing('per_claim_and_detection_scores_v2','claim and detector producers unavailable')
        scores={}
        for row in rows or []:scores[row['label']]=max(scores.get(row['label'],0.),row['score'])
        return ParameterValue({'detection_scores':scores,'image_claim_scores':claims,'calibrated':False,
            'detector_available':rows is not None},True,'per_claim_and_detection_scores_v2',supporting={'not_image_importance':True})
