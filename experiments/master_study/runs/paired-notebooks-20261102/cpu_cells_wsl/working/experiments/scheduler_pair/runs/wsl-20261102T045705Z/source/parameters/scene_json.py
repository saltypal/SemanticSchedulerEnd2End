"""Domain-independent scene evidence JSON; no labels, priority or VLM verdicts."""
from copy import deepcopy
import math
from .contracts import ParameterBundle, ParameterValue, MissionSpec, canonical, digest
from .extractor import VisualEvidence
from .image_claims import image_claim_predictions
from .pilot_parameters import canonical_label, detections, matches_target, union_area

SCHEMA = 'semantic_scene_evidence_v2'


def restore_bundle(value):
    """Rehydrate an existing runtime bundle, never an evaluator case envelope."""
    data = deepcopy(value)
    data['parameters'] = {key: ParameterValue(**row) for key, row in data['parameters'].items()}
    return ParameterBundle(**data)


def datum(value, method, *, status='available', reason=None, **details):
    if status not in ('available', 'unavailable', 'cold_start'):
        raise ValueError('unknown evidence status')
    if status == 'available' and value is None:
        raise ValueError('available evidence requires a value')
    if status != 'available' and (value is not None or not reason):
        raise ValueError('unavailable/cold-start evidence requires null value and reason')
    return dict(value=value, status=status, reason=reason, method=method, **details)


def score(value, kind, **details):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError('score must be a finite number')
    return dict(value=float(value), kind=kind, calibrated=False, **details)


def _claim(name, value, producer, evidence_score=None):
    subject, separator, predicate = name.rpartition('.')
    return dict(name=name, subject=dict(kind='category' if separator else 'scene',
                name=subject if separator else 'scene'), predicate=predicate if separator else name,
                **datum(value, 'producer_decision', producer_ref=producer), score=evidence_score)


def implication_conflicts(claims, rules):
    """Flag caller-declared ontology conflicts; never rewrite model predictions."""
    by_name = {row['name']: row for row in claims}
    result = []
    for rule in rules:
        if set(rule) != {'id', 'if', 'then'} or not isinstance(rule['id'], str):
            raise ValueError('rules require id, if and then')
        for side in ('if', 'then'):
            if (not isinstance(rule[side], (list, tuple)) or len(rule[side]) != 2
                    or not isinstance(rule[side][0], str) or type(rule[side][1]) is not bool):
                raise ValueError('rule sides require [claim name, boolean]')
        premise, consequence = (by_name.get(rule[side][0]) for side in ('if', 'then'))
        if (premise and consequence and premise['status'] == consequence['status'] == 'available'
                and premise['value'] is rule['if'][1] and consequence['value'] is not rule['then'][1]):
            result.append(dict(kind='ontology_conflict', rule_id=rule['id'],
                               claims=[premise['name'], consequence['name']], predictions_modified=False))
    return result


def build_scene_document(bundle, evidence, mission, *, source_id, image_id=None,
                         ontology=None):
    """Return (reviewable document, audit sidecar) from runtime predictions only.

    The caller supplies causal, pre-commit evidence and a matching mission. Models
    remain domain-specific; this schema does not promise universal recognition.
    No native-reference mapping or dataset annotation is accepted by this API.
    """
    if not isinstance(bundle, ParameterBundle) or not isinstance(evidence, VisualEvidence):
        raise TypeError('typed runtime bundle and visual evidence required')
    if not isinstance(mission, MissionSpec) or bundle.mission_version != mission.version:
        raise ValueError('bundle and mission identity differ')
    if not isinstance(source_id, str) or not source_id:
        raise ValueError('source identity required for causal KB interpretation')
    if evidence.width <= 0 or evidence.height <= 0:
        raise ValueError('positive original image dimensions required')
    ontology = deepcopy(ontology or {'id': 'unspecified', 'rules': []})
    producers = {}
    def register(kind, settings):
        producers[kind] = dict(kind=kind, settings_sha256=digest(settings), settings=deepcopy(settings))
        return kind
    image_source = register('image_text_encoder', {k: v for k, v in evidence.provenance.items()
        if k not in ('detector_provenance', 'condition_contrasts', 'image_claim_predictions', 'image_claim_producer')})
    detector = evidence.provenance.get('detector_provenance', {})
    detector_ref = register('object_detector', detector)
    classifier_ref = register('image_classifier', evidence.provenance.get('image_claim_producer', {}))
    claims = []
    for name, row in sorted(image_claim_predictions(evidence).items()):
        claim = _claim(name, row['value'], classifier_ref,
            score(row['score'], 'positive_class_sigmoid', threshold=row['threshold'],
                  decision_operator='>', positive_class=name))
        claim.update(method=row['method'], source_label=row.get('native_label'))
        claims.append(claim)
    observed = detections(evidence)
    # Only positive detections add claims; a missing box never creates false.
    existing = {row['name'] for row in claims}
    for label in sorted({row['label'] for row in observed or []}):
        name = label + '.present'
        if name not in existing:
            claims.append(_claim(name, True, detector_ref,
                score(max(row['score'] for row in observed if row['label'] == label), 'detector_score')))
    diagnostics = implication_conflicts(claims, ontology['rules'])
    for claim in claims:
        if claim['value'] is False and any(obj['label'] == claim['subject']['name'] for obj in observed or []) and claim['predicate'] == 'present':
            diagnostics.append(dict(kind='producer_disagreement', claim=claim['name'],
                                    detector_observed=True, classifier_value=False, predictions_modified=False))

    def parameter(name):
        return bundle.parameters.get(name, ParameterValue(None, False, 'not_requested', 'producer not requested'))
    def unavailable(p):
        return datum(None, p.method, status='unavailable', reason=p.reason or 'evidence unavailable')
    relevance = parameter('relevance')
    if relevance.available:
        cosine = relevance.supporting.get('raw_cosine')
        if cosine is None and 'cosine' in relevance.method:
            cosine = 2*float(relevance.value)-1
        relevance_result = datum(cosine if cosine is not None else relevance.value, relevance.method,
            units='cosine' if cosine is not None else 'producer_score', producer_ref=image_source,
            calibrated=False, scope='declared mission; not importance')
    else: relevance_result = unavailable(relevance)
    novelty = parameter('novelty')
    if novelty.available:
        cold = bool(novelty.value.get('cold_start'))
        novelty_result = datum(None if cold else dict(
            appearance_distance=novelty.value['distance'],
            new_predicted_concept_fraction=novelty.value.get('new_concept_fraction')),
            novelty.method, status='cold_start' if cold else 'available',
            reason='no prior comparable observations' if cold else None,
            kb_version=bundle.kb_version, comparable_history=novelty.supporting.get('comparable_history',
                novelty.supporting.get('history_count')), producer_ref=image_source,
            reference='sender observation history before current commit', new_asset_verified=False)
    else: novelty_result = unavailable(novelty)
    rarity = parameter('rarity')
    if rarity.available:
        support = rarity.supporting
        cold = bool(support.get('cold_start')) or support.get('valid_history_count') == 0
        rarity_result = datum(None if cold else rarity.value, rarity.method,
            status='cold_start' if cold else 'available',
            reason='no comparable concept-frequency history' if cold else None,
            units='normalized_mean_surprisal', history_count=support.get('valid_history_count'),
            reference=support.get('reference', 'sender predicted observations'),
            reference_probabilities=support.get('probabilities', {}), not_importance=True)
    else: rarity_result = unavailable(rarity)

    spatial = parameter('spatial')
    complexity = parameter('complexity')
    # Recompute exact raw counts from the saved accepted boxes; do not undo a
    # capped normalized count by multiplying it and inventing a raw count.
    targets = [row['box'] for row in observed or [] if matches_target(row['label'], mission.target_entities)]
    geometric = datum(dict(target_detections=len(targets), all_detections=len(observed or []),
        largest_target_box_area_fraction=max(((b[2]-b[0])*(b[3]-b[1])/(evidence.width*evidence.height) for b in targets), default=0.),
        target_box_union_fraction=union_area(targets)/(evidence.width*evidence.height)),
        'accepted_box_geometry_v2', producer_ref=detector_ref, targets=list(mission.target_entities),
        object_count_verified=False, physical_distance_available=False) if observed is not None else unavailable(spatial)
    diversity = datum(complexity.value.get('class_diversity'), complexity.method,
        units='normalized_detected_class_entropy', producer_ref=detector_ref,
        semantic_depth_verified=False) if complexity.available and 'class_diversity' in complexity.value else unavailable(complexity)
    signals = [dict(name=name, value=float(value), units='cosine')
        for name, value in sorted(evidence.similarities.items())]
    mapped_cosine = evidence.provenance.get('backbone') in ('clip', 'siglip')
    signals.extend(dict(name='scene:' + name, value=2*float(value)-1 if mapped_cosine else float(value),
        units='cosine' if mapped_cosine else 'producer_score') for name, value in sorted(evidence.scene_scores.items()))
    detected_categories = []
    for category in sorted({row['label'] for row in observed or []}):
        rows = [row for row in observed if row['label'] == category]
        boxes = [row['box'] for row in rows]
        detected_categories.append(dict(category=category, detection_count=len(rows),
            largest_box_area_fraction=max((b[2]-b[0])*(b[3]-b[1]) for b in boxes)/(evidence.width*evidence.height),
            box_union_fraction=union_area(boxes)/(evidence.width*evidence.height),
            max_detection_score=max(row['score'] for row in rows)))
    document = dict(schema=SCHEMA,
        observation=dict(id=bundle.image_token, image_id=image_id, source_id=source_id,
                         width_px=evidence.width, height_px=evidence.height),
        mission=dict(id=mission.mission_id, version=mission.version, context=mission.paragraph,
                     observable_clauses=[dict(name=k, text=v) for k, v in mission.conditions],
                     targets=list(mission.target_entities), unsupported_clauses=list(mission.unsupported_clauses)),
        scene=dict(claims=claims,
            detections=datum(detected_categories if observed is not None else None, 'accepted_detector_category_summary_v2',
                status='available' if observed is not None else 'unavailable',
                reason=None if observed is not None else 'object detector unavailable',
                producer_ref=detector_ref, searched_categories=[canonical_label(v) for v in detector.get('vocabulary', [])],
                non_detection_is_absence=False, instance_boxes_location='audit.raw_visual_evidence.detections'),
            relations=datum(None, 'not_implemented', status='unavailable', reason='no relationship producer enabled')),
        measurements=dict(context_alignment=relevance_result, novelty=novelty_result,
                          concept_rarity=rarity_result, spatial=geometric, complexity=diversity,
                          temporal=deepcopy(bundle.temporal), geographic=datum(parameter('geographic').value,
                              parameter('geographic').method, status='available' if parameter('geographic').available else 'unavailable',
                              reason=parameter('geographic').reason)),
        similarity_signals=datum(signals if signals else None, 'raw_image_text_similarities_v2',
            status='available' if signals else 'unavailable', reason=None if signals else 'no similarity producer output',
            calibrated=False, producer_ref=image_source, not_probabilities=True), diagnostics=diagnostics,
        provenance=dict(extraction_specification=bundle.specification_sha256, ontology=ontology, producers=producers))
    audit = dict(schema='semantic_scene_audit_v2', document_sha256=digest(document),
                 raw_bundle=bundle.to_dict(), raw_visual_evidence=evidence.to_dict(),
                 conversion='typed runtime evidence only; no predictions repaired; cold-start sentinels withheld')
    canonical(document); canonical(audit)
    return document, audit


def extract_document(extractor, image, mission, metadata, memory, *, ontology=None):
    """One visual pass, then a causal pre-commit document. Caller commits on success."""
    if extractor.provider is None: raise RuntimeError('visual producer required')
    evidence = extractor.provider.extract_batch([image], [mission])[0]
    bundle = extractor.extract_evidence(evidence, mission, metadata, memory)
    document, audit = build_scene_document(bundle, evidence, mission,
        source_id=metadata['source_id'], image_id=metadata.get('image_id'), ontology=ontology)
    validate_document(document)
    return document, audit


def validate_document(document):
    """Validate shape plus cross-field invariants without a neural dependency."""
    import json
    from pathlib import Path
    from jsonschema import Draft202012Validator
    schema = json.loads((Path(__file__).parent/'schemas/semantic_scene_evidence_v2.schema.json').read_text())
    Draft202012Validator(schema).validate(document)
    canonical(document)
    forbidden = {'native_labels', 'published_annotation_reference', 'expected', 'audit', 'evaluator_facts', 'split'}
    def walk(value):
        if isinstance(value, dict):
            if forbidden & set(value): raise ValueError('evaluator fields cannot enter runtime document')
            if 'producer_ref' in value and value['producer_ref'] not in document['provenance']['producers']:
                raise ValueError('dangling evidence producer')
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(document)
    producers = document['provenance']['producers']
    names = [row['name'] for row in document['scene']['claims']]
    if len(names) != len(set(names)): raise ValueError('claim names must be unique')
    for claim in document['scene']['claims']:
        if claim['producer_ref'] not in producers: raise ValueError('dangling claim producer')
        evidence_score = claim['score']
        if evidence_score is not None and evidence_score['kind'] in ('positive_class_sigmoid', 'detector_score'):
            if not 0 <= evidence_score['value'] <= 1: raise ValueError('invalid bounded producer score')
        if evidence_score is not None and evidence_score['kind'] == 'positive_class_sigmoid' and claim['status'] == 'available':
            if type(claim['value']) is not bool or claim['value'] != (evidence_score['value'] > evidence_score['threshold']):
                raise ValueError('claim value differs from declared threshold decision')
    for obj in document['scene']['detections']['value'] or []:
        if not 0 < obj['largest_box_area_fraction'] <= obj['box_union_fraction'] <= 1:
            raise ValueError('invalid category box area fractions')
    return document
