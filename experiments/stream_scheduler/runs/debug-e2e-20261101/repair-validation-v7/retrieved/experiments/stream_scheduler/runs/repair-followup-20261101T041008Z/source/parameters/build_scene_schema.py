"""Generate the versioned, domain-independent evidence JSON Schema."""
import json
from pathlib import Path


def build(output=None):
    text = {'type': 'string', 'minLength': 1}
    nullable_text = {'type': ['string', 'null']}
    datum = {
        'type': 'object', 'required': ['value', 'status', 'reason', 'method'],
        'properties': {'value': {}, 'status': {'enum': ['available', 'unavailable', 'cold_start']},
                       'reason': nullable_text, 'method': text, 'producer_ref': text},
        'allOf': [
            {'if': {'properties': {'status': {'const': 'available'}}},
             'then': {'properties': {'value': {'not': {'type': 'null'}}, 'reason': {'type': 'null'}}},
             'else': {'properties': {'value': {'type': 'null'}, 'reason': text}}}],
        'additionalProperties': True}
    claim = {'allOf': [{'$ref': '#/$defs/datum'}], 'type': 'object',
        'required': ['name', 'subject', 'predicate', 'producer_ref', 'score'],
        'properties': {'value': {'type': ['boolean', 'null']}, 'name': text, 'subject': {'type': 'object', 'required': ['kind', 'name'],
                        'properties': {'kind': text, 'name': text}, 'additionalProperties': False},
                       'predicate': text, 'score': {'anyOf': [{'$ref': '#/$defs/score'}, {'type': 'null'}]}}}
    schema = {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        '$id': 'urn:better-scheduler:semantic-scene-evidence:2',
        'title': 'Domain-independent image evidence, not a universal recognition guarantee',
        'type': 'object', 'additionalProperties': False,
        'required': ['schema', 'observation', 'mission', 'scene', 'measurements',
                     'similarity_signals', 'diagnostics', 'provenance'],
        '$defs': {'datum': datum, 'claim': claim, 'score': {
            'type': 'object', 'required': ['value', 'kind', 'calibrated'],
            'properties': {'value': {'type': 'number'}, 'kind': text, 'calibrated': {'type': 'boolean'}}}},
        'properties': {
            'schema': {'const': 'semantic_scene_evidence_v2'},
            'observation': {'type': 'object', 'required': ['id', 'image_id', 'source_id', 'width_px', 'height_px'],
                'additionalProperties': False, 'properties': {
                    'id': text, 'image_id': nullable_text, 'source_id': text,
                    'width_px': {'type': 'integer', 'minimum': 1}, 'height_px': {'type': 'integer', 'minimum': 1}}},
            'mission': {'type': 'object', 'required': ['id', 'version', 'context', 'observable_clauses', 'targets', 'unsupported_clauses'],
                'additionalProperties': False, 'properties': {
                    'id': text, 'version': text, 'context': text,
                    'observable_clauses': {'type': 'array', 'items': {'type': 'object', 'required': ['name', 'text'],
                        'additionalProperties': False, 'properties': {'name': text, 'text': text}}},
                    'targets': {'type': 'array', 'items': text}, 'unsupported_clauses': {'type': 'array', 'items': text}}},
            'scene': {'type': 'object', 'required': ['claims', 'detections', 'relations'],
                'additionalProperties': False, 'properties': {
                    'claims': {'type': 'array', 'items': {'$ref': '#/$defs/claim'}},
                    'detections': {'allOf': [{'$ref': '#/$defs/datum'}], 'properties': {
                        'non_detection_is_absence': {'const': False},
                        'value': {'type': ['array', 'null'], 'items': {'type': 'object',
                            'required': ['category', 'detection_count', 'largest_box_area_fraction', 'box_union_fraction', 'max_detection_score'],
                            'additionalProperties': False, 'properties': {
                                'category': text, 'detection_count': {'type': 'integer', 'minimum': 1},
                                'largest_box_area_fraction': {'type': 'number', 'minimum': 0, 'maximum': 1},
                                'box_union_fraction': {'type': 'number', 'minimum': 0, 'maximum': 1},
                                'max_detection_score': {'type': 'number', 'minimum': 0, 'maximum': 1}}}}}},
                    'relations': {'$ref': '#/$defs/datum'}}},
            'measurements': {'type': 'object', 'additionalProperties': False,
                'required': ['context_alignment', 'novelty', 'concept_rarity', 'spatial', 'complexity', 'temporal', 'geographic'],
                'properties': {**{name: {'$ref': '#/$defs/datum'} for name in (
                    'context_alignment', 'novelty', 'concept_rarity', 'spatial', 'complexity', 'geographic')},
                    'temporal': {'type': 'object'}}},
            'similarity_signals': {'allOf': [{'$ref': '#/$defs/datum'}],
                'required': ['calibrated', 'producer_ref', 'not_probabilities'],
                'properties': {'calibrated': {'const': False}, 'not_probabilities': {'const': True},
                    'value': {'type': ['array', 'null'], 'items': {'type': 'object',
                        'required': ['name', 'units', 'value'], 'additionalProperties': False,
                        'properties': {'name': text, 'value': {'type': 'number'}, 'units': text}}}}},
            'diagnostics': {'type': 'array', 'items': {'type': 'object', 'required': ['kind', 'predictions_modified'],
                'properties': {'kind': text, 'predictions_modified': {'const': False}}}},
            'provenance': {'type': 'object', 'required': ['extraction_specification', 'ontology', 'producers'],
                'additionalProperties': False, 'properties': {
                    'extraction_specification': text,
                    'ontology': {'type': 'object', 'required': ['id', 'rules'],
                        'properties': {'id': text, 'rules': {'type': 'array'}}},
                    'producers': {'type': 'object', 'additionalProperties': {
                        'type': 'object', 'required': ['kind', 'settings'],
                        'properties': {'kind': text, 'settings': {'type': 'object'}}}}}}}}
    path = Path(output) if output is not None else Path(__file__).parent/'schemas/semantic_scene_evidence_v2.schema.json'
    path.write_text(json.dumps(schema, indent=2) + '\n', encoding='utf-8', newline='\n')
    return path


if __name__ == '__main__': print(build())
