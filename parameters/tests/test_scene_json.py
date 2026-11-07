from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from parameters.contracts import MissionSpec, canonical
from parameters.extractor import VisualEvidence
from parameters.pilot_parameters import PilotMemory, PilotParameterExtractor, context_similarity_key
from parameters.pilot_producers import PilotVisionProducer
from parameters.scene_json import build_scene_document, extract_document, validate_document


def fixture(domain='machine', *, boxes=(), predictions=None):
    mission = MissionSpec('inspect-'+domain, 'Inspect the visible '+domain+'.',
        ((domain+'.condition', 'a photograph of a '+domain),), (domain,))
    evidence = VisualEvidence(embedding=(1., 0.), width=100, height=100,
        similarities={context_similarity_key(mission): .2}, scene_scores={'indoor': .6},
        detections=tuple(boxes), statuses={'detector': 'ok'},
        provenance={'backbone': 'clip', 'model_id': 'fixture-clip',
            'detector_provenance': {'conf_threshold': .05, 'vocabulary': [domain]},
            'image_claim_predictions': predictions or {}})
    memory = PilotMemory('sender', mission.mission_id)
    extractor = PilotParameterExtractor()
    bundle = extractor.extract_evidence(evidence, mission, {'source_id': 'sender', 'image_token': 'one'}, memory)
    return mission, evidence, bundle, memory, extractor


def prediction(value):
    return dict(value=value, score=.9 if value else .1, threshold=.5,
                method='fixture_image_classifier', calibrated=False)


@pytest.mark.parametrize('domain', ['machine', 'cup', 'animal'])
def test_schema_is_domain_independent_and_no_detection_does_not_assert_absence(domain):
    mission, evidence, bundle, _, _ = fixture(domain)
    document, audit = build_scene_document(bundle, evidence, mission, source_id='sender')
    validate_document(document)
    assert document['scene']['claims'] == []
    assert document['scene']['detections']['value'] == []
    assert document['scene']['detections']['status'] == 'available'
    assert document['scene']['relations']['status'] == 'unavailable'
    assert document['measurements']['context_alignment']['value'] == pytest.approx(.2)
    assert audit['raw_bundle'] == bundle.to_dict()
    assert document['measurements']['novelty']['value'] is None
    assert document['measurements']['novelty']['status'] == 'cold_start'
    assert document['similarity_signals']['calibrated'] is False


def test_explicit_false_is_available_and_history_cold_start_is_not_maximum_novelty():
    mission, evidence, bundle, memory, extractor = fixture(predictions={
        'machine.present': prediction(False), 'machine.damaged': prediction(True)})
    ontology = {'id': 'fixture-logic', 'rules': [
        {'id': 'damaged-implies-present', 'if': ['machine.damaged', True], 'then': ['machine.present', True]}]}
    document, _ = build_scene_document(bundle, evidence, mission, source_id='sender', ontology=ontology)
    validate_document(document)
    negative = next(row for row in document['scene']['claims'] if row['name'] == 'machine.present')
    assert negative['value'] is False and negative['status'] == 'available'
    assert negative['score']['kind'] == 'positive_class_sigmoid' and negative['score']['value'] == .1
    assert document['diagnostics'][0]['kind'] == 'ontology_conflict'
    assert not document['diagnostics'][0]['predictions_modified']
    assert document['measurements']['concept_rarity']['status'] == 'cold_start'
    memory.commit(evidence, 0)
    repeated = extractor.extract_evidence(evidence, mission, {'source_id': 'sender', 'image_token': 'two'}, memory)
    doc, _ = build_scene_document(repeated, evidence, mission, source_id='sender', ontology=ontology)
    assert doc['measurements']['novelty']['status'] == 'available'
    assert doc['measurements']['novelty']['value']['appearance_distance'] == 0
    assert doc['measurements']['concept_rarity']['status'] == 'available'


def test_raw_counts_and_producer_disagreement_are_retained_without_silent_fusion():
    boxes = [dict(label='machine', score=.1, box=[0, 0, 50, 50])]*150
    mission, evidence, bundle, _, _ = fixture(boxes=boxes, predictions={'machine.present': prediction(False)})
    document, _ = build_scene_document(bundle, evidence, mission, source_id='sender')
    validate_document(document)
    assert document['measurements']['spatial']['value']['target_detections'] == 150
    assert document['measurements']['spatial']['value']['target_box_union_fraction'] == .25
    assert document['scene']['detections']['value'][0]['detection_count'] == 150
    assert document['scene']['claims'][0]['value'] is False
    assert document['diagnostics'][0]['kind'] == 'producer_disagreement'


def test_validator_rejects_masks_bad_geometry_nan_duplicate_and_evaluator_content():
    mission, evidence, bundle, _, _ = fixture(boxes=[dict(label='machine', score=.8, box=[1, 1, 20, 20])])
    document, _ = build_scene_document(bundle, evidence, mission, source_id='sender')
    for change in (
        lambda d: d['scene']['claims'][0].update(value='unknown'),
        lambda d: d['scene']['claims'][0].update(status='unavailable'),
        lambda d: d['scene']['claims'].append(deepcopy(d['scene']['claims'][0])),
        lambda d: d['scene']['detections']['value'][0].update(box_union_fraction=.0001),
        lambda d: d['measurements']['context_alignment'].update(value=float('nan')),
        lambda d: d['scene']['claims'][0].update(producer_ref='unknown-producer'),
        lambda d: d['provenance']['producers'].update(native_labels={'truth': True}),
    ):
        altered = deepcopy(document); change(altered)
        with pytest.raises(Exception): validate_document(altered)


def test_single_visual_pass_and_failed_export_do_not_commit_memory():
    mission, evidence, _, memory, _ = fixture()
    calls = []
    class Provider:
        def extract_batch(self, images, missions):
            calls.append((images, missions)); return [evidence]
    extractor = PilotParameterExtractor(provider=Provider())
    doc, audit = extract_document(extractor, object(), mission, {'source_id': 'sender', 'image_token': 'one'}, memory)
    assert len(calls) == 1 and memory.version == 0
    with pytest.raises(ValueError):
        extract_document(extractor, object(), mission, {'source_id': 'wrong', 'image_token': 'two'}, memory)
    assert memory.version == 0
    assert 'audit' not in doc and audit['schema'] != doc['schema']


def test_custom_detector_vocabulary_is_validated_and_does_not_claim_other_categories():
    producer = PilotVisionProducer('unused', {'custom': object()}, device='cpu', vocabulary=['cup', 'plate'])
    assert producer.vocabulary == ('cup', 'plate')
    for vocabulary in ([], ['car', 'cars'], [''], [None]):
        with pytest.raises(ValueError):
            PilotVisionProducer('unused', {'custom': object()}, device='cpu', vocabulary=vocabulary)
    mission, evidence, _, _, _ = fixture('cup')
    evidence.provenance['detector_provenance']['vocabulary'] = ['cup', 'plate']
    scene = PilotParameterExtractor().extract_scene_evidence(evidence)
    assert set(scene.value['detector_observation']) == {'cup', 'plate'}
    assert scene.value['detector_observation']['cup'] == 'not_detected'
    evidence.provenance['detector_provenance']['vocabulary'] = ['cars']
    evidence.detections = ({'label': 'car', 'score': .8, 'box': [0,0,50,50]},)
    scene = PilotParameterExtractor().extract_scene_evidence(evidence)
    assert scene.value['detector_observation'] == {'car': 'detected'}


def test_schema_generation_matches_checked_in_definition(tmp_path):
    from parameters.build_scene_schema import build
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'schemas/semantic_scene_evidence_v2.schema.json'
    before = path.read_bytes()
    generated = build(tmp_path/'schema.json')
    assert generated.read_bytes() == before
