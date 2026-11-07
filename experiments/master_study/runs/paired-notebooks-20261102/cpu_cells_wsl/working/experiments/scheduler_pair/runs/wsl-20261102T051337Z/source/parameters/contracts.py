"""Runtime contracts deliberately exclude dataset annotations and evaluator facts."""
from dataclasses import dataclass, field, asdict
from typing import Any
import hashlib
import json
import math

GROUPS = ('relevance', 'novelty', 'rarity', 'scene', 'spatial', 'complexity', 'geographic')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def finite_tree(value):
    if isinstance(value, dict):
        return all(isinstance(k, str) and finite_tree(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return value is None or isinstance(value, (str, bool, int))


@dataclass(frozen=True)
class MissionSpec:
    mission_id: str
    paragraph: str
    conditions: tuple[tuple[str, str], ...]
    target_entities: tuple[str, ...] = ()
    allowed_delay_ms: int | None = None
    region: tuple[float, float, float, float] | None = None
    unsupported_clauses: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'conditions', tuple(tuple(x) for x in self.conditions))
        object.__setattr__(self, 'target_entities', tuple(self.target_entities))
        if not self.mission_id or not self.paragraph or not self.conditions:
            raise ValueError('mission identity, paragraph and explicit observable conditions required')
        if (any(len(c) != 2 or not all(isinstance(v, str) and v for v in c) for c in self.conditions)
                or len({c[0] for c in self.conditions}) != len(self.conditions)):
            raise ValueError('conditions require unique claim names and nonempty text')
        if self.allowed_delay_ms is not None and (type(self.allowed_delay_ms) is not int or self.allowed_delay_ms <= 0):
            raise ValueError('allowed delay must be a positive integer in milliseconds')
        if self.region is not None:
            if (len(self.region) != 4 or not all(math.isfinite(v) for v in self.region)
                    or not -90 <= self.region[0] <= self.region[2] <= 90
                    or not -180 <= self.region[1] <= self.region[3] <= 180):
                raise ValueError('region requires min latitude/longitude and max latitude/longitude')

    @property
    def version(self):
        return digest(asdict(self))


MISSIONS = {
    'building_damage': MissionSpec('building_damage', 'Identify visible damage to buildings.',
                                 (('building_damage', 'a photograph of damaged buildings'),), ('building',)),
    'road_damage': MissionSpec('road_damage', 'Identify visible road damage or blockage.',
                             (('road_damage', 'a photograph of a damaged or blocked road'),), ('road',)),
    'affected_people': MissionSpec('affected_people', 'Find images showing affected, injured or dead people.',
                                 (('affected_people', 'a photograph showing people affected by a disaster'),), ('person',)),
    'structural_flooding': MissionSpec('structural_flooding', 'Identify buildings affected by flooding.',
                                     (('structural_flooding', 'a photograph of flooded buildings'),), ('building',)),
}


@dataclass(frozen=True)
class MethodSpecification:
    backbone: str = 'clip'
    relevance: str = 'similarity'
    novelty: str = 'embedding'
    enabled_groups: tuple[str, ...] = GROUPS
    kb_capacity: int = 64
    neighbor_count: int = 3
    detector: str | None = None
    quantization_bits: int = 8
    preprocessing: str = 'aspect_preserving_pad'
    calibration: dict = field(default_factory=dict)
    producer_revisions: dict = field(default_factory=dict)
    feature_names: tuple[str, ...] = ()
    dependency_graph: dict = field(default_factory=lambda: {
        'relevance': ['image_text_encoder'], 'novelty': ['image_text_encoder', 'sender_kb'],
        'rarity': ['predicted_concepts', 'sender_kb'], 'scene': ['image_text_encoder'],
        'spatial': ['detector'], 'complexity': ['detector'], 'geographic': ['trusted_metadata']})

    def __post_init__(self):
        object.__setattr__(self, 'enabled_groups', tuple(self.enabled_groups))
        object.__setattr__(self, 'feature_names', tuple(self.feature_names))
        if self.backbone not in ('clip', 'siglip', 'fixture'):
            raise ValueError('unsupported backbone')
        if self.relevance not in ('similarity', 'claim_probability') or self.novelty not in ('embedding', 'semantic_delta'):
            raise ValueError('unsupported extraction method')
        if len(set(self.enabled_groups)) != len(self.enabled_groups) or not set(self.enabled_groups) <= set(GROUPS):
            raise ValueError('invalid or repeated feature group')
        if self.kb_capacity <= 0 or self.neighbor_count <= 0 or self.quantization_bits not in (8, 16):
            raise ValueError('invalid extraction limits')
        if not finite_tree(asdict(self)):
            raise ValueError('nonfinite specification')

    @property
    def sha256(self):
        return digest(asdict(self))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ParameterValue:
    value: Any
    available: bool
    method: str
    reason: str | None = None
    supporting: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.available) is not bool or not self.method or not finite_tree(self.value) or not finite_tree(self.supporting):
            raise ValueError('invalid parameter result')
        if self.available == (self.value is None):
            raise ValueError('available results need a value; unavailable results must be None')
        if not self.available and not self.reason:
            raise ValueError('missing evidence requires a reason')


@dataclass(frozen=True)
class ParameterBundle:
    image_token: str
    mission_id: str
    mission_version: str
    specification_sha256: str
    kb_version: int
    parameters: dict[str, ParameterValue]
    temporal: dict
    producer: dict
    schema: str = 'better_scheduler_evidence_v1'

    def __post_init__(self):
        if not self.image_token or not self.mission_id or self.kb_version < 0:
            raise ValueError('invalid bundle identity')
        if not set(self.parameters) <= set(GROUPS) | {'reliability'}:
            raise ValueError('unknown parameter group')
        if not all(isinstance(v, ParameterValue) for v in self.parameters.values()):
            raise TypeError('typed parameter results required')
        if not finite_tree(self.temporal) or not finite_tree(self.producer):
            raise ValueError('invalid metadata')

    def to_dict(self):
        return asdict(self)


def flatten_bundle(bundle, groups=GROUPS):
    """Only normalized numeric predictions enter a policy, never IDs/provenance."""
    output = {}
    def visit(path, value):
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                visit(path + '.' + key, item)
        elif isinstance(value, (int, float, bool)):
            number = float(value)
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError('policy values must be normalized to [0,1]: ' + path)
            output[path] = number
    for group in groups:
        result = bundle.parameters.get(group)
        if result is not None and result.available:
            visit(group, result.value)
    return output
