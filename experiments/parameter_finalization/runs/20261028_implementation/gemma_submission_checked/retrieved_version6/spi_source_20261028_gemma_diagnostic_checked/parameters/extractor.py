"""Pure evidence calculations; one provider forward can serve several parameters."""
from collections import deque, Counter
from dataclasses import dataclass, field
import math
import numpy as np

from .contracts import MethodSpecification, MissionSpec, ParameterBundle, ParameterValue, canonical,digest


def unit(vector):
    if vector is None:
        return None
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        return None
    norm = np.linalg.norm(vector)
    return None if norm == 0 else vector/norm


def embedding_signature(evidence):
    return digest({key:evidence.provenance.get(key) for key in (
        'backbone','model_id','config_sha256','weights_sha256','preprocessing','processor_sha256')})


@dataclass
class VisualEvidence:
    embedding: object = None
    similarities: dict = field(default_factory=dict)
    probabilities: dict = field(default_factory=dict)
    scene_scores: dict = field(default_factory=dict)
    detections: tuple = ()
    width: int = 0
    height: int = 0
    statuses: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value):
        if set(value) - set(cls.__dataclass_fields__):
            raise ValueError('unknown fields in runtime visual evidence')
        return cls(**value)

    def to_dict(self):
        return dict(embedding=None if self.embedding is None else np.asarray(self.embedding).tolist(),
                    similarities=self.similarities, probabilities=self.probabilities,
                    scene_scores=self.scene_scores, detections=list(self.detections),
                    width=self.width, height=self.height, statuses=self.statuses,
                    provenance=self.provenance)


class SenderMemory:
    """Causal sender/mission state. Queries never mutate it."""
    def __init__(self, source_id, mission_id, capacity=64):
        if not source_id or not mission_id or type(capacity) is not int or capacity <= 0:
            raise ValueError('source, mission and positive capacity required')
        self.source_id, self.mission_id, self.capacity = source_id, mission_id, capacity
        self.entries = deque(maxlen=capacity)
        self.version = 0
        self.last_sequence = -1

    def commit(self, evidence, sequence):
        if type(sequence) is not int or sequence <= self.last_sequence:
            raise ValueError('sender observation sequence must strictly increase')
        embedding = unit(evidence.embedding)
        self.entries.append({'embedding': None if embedding is None else embedding.copy(),
                             'embedding_signature':embedding_signature(evidence),
                             'probabilities': dict(evidence.probabilities)})
        self.version += 1
        self.last_sequence = sequence


class SemanticParameterExtractor:
    """Extract a bundle; explicit commit follows successful header serialization.

    `extract` and `extract_batch` do not modify memories. `transmit_stream` first
    batches visual inference, then serializes/commits each observation in input
    sender order. This avoids GPU completion order changing the knowledge base.
    """
    def __init__(self, specification=None, provider=None):
        self.specification = specification or MethodSpecification()
        self.provider = provider

    @staticmethod
    def missing(method, reason):
        return ParameterValue(None, False, method, reason)

    def extract(self, image, mission, metadata, memory):
        if self.provider is None:
            raise RuntimeError('an image provider is required; use extract_evidence for cached predictions')
        evidence = self.provider.extract_batch([image], [mission])[0]
        return self.extract_evidence(evidence, mission, metadata, memory)

    def extract_batch(self, images, missions, metadata, memories):
        if not (len(images) == len(missions) == len(metadata) == len(memories)):
            raise ValueError('batch inputs differ in length')
        if self.provider is None:
            raise RuntimeError('image provider required')
        evidence = self.provider.extract_batch(images, missions)
        return [self.extract_evidence(e, m, meta, kb) for e, m, meta, kb in zip(evidence, missions, metadata, memories)]

    def transmit_stream(self, images, missions, metadata, memories, serialize):
        if not (len(images)==len(missions)==len(metadata)==len(memories)):
            raise ValueError('stream batch inputs differ in length')
        evidence = self.provider.extract_batch(images, missions)
        output = []
        for observation, mission, meta, memory in zip(evidence, missions, metadata, memories):
            bundle = self.extract_evidence(observation, mission, meta, memory)
            wire = serialize(bundle)  # A failed serializer must not advance the sender KB.
            if not isinstance(wire,bytes):raise TypeError('serializer must return bytes before KB commit')
            memory.commit(observation, meta['sequence'])
            output.append((bundle, wire))
        return output

    def extract_evidence(self, evidence, mission, metadata, memory):
        if not isinstance(evidence, VisualEvidence) or not isinstance(mission, MissionSpec):
            raise TypeError('runtime VisualEvidence and MissionSpec required')
        if memory.mission_id != mission.mission_id or metadata.get('source_id') != memory.source_id:
            raise ValueError('sender memory identity differs from observation')
        parameters = {
            'relevance': self.extract_context_relevance(evidence, mission),
            'novelty': self.extract_novelty(evidence, mission, memory),
            'rarity': self.extract_scarcity(evidence, mission, memory),
            'scene': self.extract_scene_evidence(evidence),
            'spatial': self.extract_spatial_evidence(evidence, mission),
            'complexity': self.extract_scene_complexity(evidence, mission),
            'geographic': self.extract_geographic_context(mission, metadata),
            'reliability': self.extract_reliability(evidence),
        }
        for group in set(parameters)-set(self.specification.enabled_groups)-{'reliability'}:
            parameters[group] = self.missing('disabled', 'disabled by specification')
        bundle = ParameterBundle(str(metadata['image_token']), mission.mission_id, mission.version,
                                 self.specification.sha256, memory.version, parameters,
                                 self.extract_temporal_metadata(mission, metadata), evidence.provenance)
        canonical(bundle.to_dict())
        return bundle

    def extract_context_relevance(self, evidence, mission):
        if self.specification.relevance == 'claim_probability':
            values = [evidence.probabilities[c] for c, _ in mission.conditions if c in evidence.probabilities]
            method = 'max_supported_claim_probability'
        else:
            values = [evidence.similarities[c] for c, _ in mission.conditions if c in evidence.similarities]
            values = [(float(v)+1)/2 for v in values]
            method = 'max_condition_cosine_mapped'
        values = [float(v) for v in values if math.isfinite(v) and 0 <= v <= 1]
        if not values:
            return self.missing(method, 'no supported mission condition prediction')
        return ParameterValue(max(values), True, method, supporting={'conditions_used': len(values), 'conditions_requested': len(mission.conditions), 'is_probability': self.specification.relevance == 'claim_probability'})

    def extract_novelty(self, evidence, mission, memory):
        embedding = unit(evidence.embedding)
        if embedding is None:
            return self.missing(self.specification.novelty, 'invalid or unavailable embedding')
        if not memory.entries:
            return ParameterValue({'distance': 1., 'cold_start': 1.}, True, self.specification.novelty,
                                  supporting={'history_count': 0, 'interpretation': 'cold start, not proof of new event'})
        previous = [r for r in memory.entries if r['embedding'] is not None and r['embedding'].shape == embedding.shape
                    and r.get('embedding_signature')==embedding_signature(evidence)]
        if not previous:
            return self.missing(self.specification.novelty, 'no comparable KB embeddings')
        similarities = np.array([float(embedding @ r['embedding']) for r in previous])
        result = {'distance': float((1-np.clip(similarities.max(), -1, 1))/2), 'cold_start': 0.}
        if self.specification.novelty == 'semantic_delta':
            neighbors = [previous[i] for i in np.argsort(-similarities, kind='stable')[:self.specification.neighbor_count]]
            for claim, _ in mission.conditions:
                prior = [row['probabilities'][claim] for row in neighbors if claim in row['probabilities']]
                if claim in evidence.probabilities and prior:
                    result['added_' + claim] = max(0., float(evidence.probabilities[claim])-max(prior))
        return ParameterValue(result, True, self.specification.novelty,
                              supporting={'comparable_history': len(previous), 'history_count': len(memory.entries), 'identity_verified': False})

    def extract_scarcity(self, evidence, mission, memory):
        concepts = [claim for claim, _ in mission.conditions if evidence.probabilities.get(claim, -1) >= .5]
        if not concepts:
            return self.missing('sender_smoothed_surprisal', 'no supported positive eligible concept')
        values = []
        for concept in concepts:
            prior = [row['probabilities'][concept] for row in memory.entries if concept in row['probabilities']]
            probability = (sum(p >= .5 for p in prior)+1)/(len(prior)+2)
            values.append(-math.log(max(.01, probability))/-math.log(.01))
        return ParameterValue(float(np.mean(values)), True, 'sender_smoothed_surprisal',
                              supporting={'history_count': len(memory.entries), 'concept_count': len(concepts), 'reference': 'past predicted sender observations; not operational importance'})

    def extract_scene_evidence(self, evidence):
        values = {str(k): float(v) for k, v in evidence.scene_scores.items() if math.isfinite(v) and 0 <= v <= 1}
        values.update({'claim_' + str(k): float(v) for k, v in evidence.probabilities.items() if math.isfinite(v) and 0 <= v <= 1})
        return ParameterValue(values, True, 'shared_feature_multilabel_or_prompt_scores') if values else self.missing('scene', 'scene/event predictions unavailable')

    @staticmethod
    def _detections(evidence):
        if evidence.statuses.get('detector') != 'ok' or evidence.width <= 0 or evidence.height <= 0:
            return None
        valid = []
        for item in evidence.detections:
            try:
                x0,y0,x1,y1 = map(float,item['box']); score=float(item['score'])
                if not all(math.isfinite(x) for x in (x0,y0,x1,y1,score)) or score < .35:
                    continue
                x0,x1 = np.clip([x0,x1],0,evidence.width);y0,y1 = np.clip([y0,y1],0,evidence.height)
                if x1>x0 and y1>y0:
                    valid.append((str(item['label']),float((x1-x0)*(y1-y0)/(evidence.width*evidence.height))))
            except (KeyError,ValueError,TypeError):
                continue
        return valid[:100]

    def extract_spatial_evidence(self, evidence, mission):
        detections = self._detections(evidence)
        if detections is None:
            return self.missing('target_area', 'localized producer unavailable')
        areas = [area for label,area in detections if label in mission.target_entities]
        return ParameterValue({'largest_area': max(areas,default=0.), 'target_count': min(1.,len(areas)/100)}, True,
                              'target_area', supporting={'physical_distance_available': False, 'raw_target_count': len(areas)})

    def extract_scene_complexity(self, evidence, mission):
        detections = self._detections(evidence)
        if detections is None:
            return self.missing('count_entropy_diagnostic', 'localized producer unavailable')
        counts = Counter(label for label,_ in detections); n=sum(counts.values())
        entropy = 0. if n == 0 or len(counts) <= 1 else -sum((v/n)*math.log(v/n) for v in counts.values())/math.log(len(counts))
        relevant = sum(label in mission.target_entities for label,_ in detections)
        return ParameterValue({'density': math.log1p(n)/math.log(101), 'diversity': entropy,
                               'target_density': math.log1p(relevant)/math.log(101)}, True,
                              'count_entropy_diagnostic', supporting={'raw_count': n, 'not_a_priority_label': True})

    def extract_reliability(self, evidence):
        calibration = self.specification.calibration
        claims = {key: {'probability': float(value), 'calibrated': isinstance(calibration.get(key),dict) and calibration[key].get('available') is True}
                  for key,value in evidence.probabilities.items() if math.isfinite(value) and 0 <= value <= 1}
        return ParameterValue(claims, True, 'claim_specific_probability_and_calibration') if claims else self.missing('claim_reliability', 'no claim probability; cosine is not a calibrated probability')

    def extract_geographic_context(self, mission, metadata):
        coordinates = metadata.get('coordinates')
        if mission.region is None or metadata.get('location_trusted') is not True or coordinates is None:
            return self.missing('trusted_geofence', 'trusted coordinates and mission region required')
        if (len(coordinates)!=2 or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in coordinates)
                or not -90 <= coordinates[0] <= 90 or not -180 <= coordinates[1] <= 180):
            return self.missing('trusted_geofence', 'invalid coordinates')
        south,west,north,east=mission.region
        return ParameterValue(float(south<=coordinates[0]<=north and west<=coordinates[1]<=east),True,'trusted_geofence')

    def extract_temporal_metadata(self, mission, metadata):
        output = {'capture_time_ms': None, 'allowed_delay_ms': mission.allowed_delay_ms,
                  'age_ms': None, 'deadline_ms': None, 'completion_slack_ms': None,
                  'available': False, 'reason': 'trusted capture/current timestamps required'}
        captured, now = metadata.get('capture_time_ms'), metadata.get('now_ms')
        if (metadata.get('timestamp_trusted') is not True or type(captured) is not int or type(now) is not int
                or captured <= 0 or now < captured):
            return output
        output.update(capture_time_ms=captured,age_ms=now-captured,available=True,reason=None)
        if mission.allowed_delay_ms is not None:
            deadline=captured+mission.allowed_delay_ms;output['deadline_ms']=deadline
            service=metadata.get('expected_decode_ms')
            if isinstance(service,(float,int)) and not isinstance(service,bool) and math.isfinite(service) and service>=0:
                output['completion_slack_ms']=deadline-now-service
        return output
