"""V3 header-only decoder policies and validated NumPy inference.

No image model, payload, labels, queue mutation, or neural framework is imported.
Arrival eligibility and freshness updates belong to Evaluation.
"""
from dataclasses import dataclass
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PARAMETER_NAMES = ('R', 'S', 'Q', 'I', 'F', 'P', 'C', 'N')
FEATURE_NAMES = PARAMETER_NAMES + tuple(f'{name}_available' for name in PARAMETER_NAMES)
ARTIFACT_SCHEMA = 'decoder_scheduler_v3'


@dataclass(frozen=True)
class SchedulerCandidate:
    message_id: str
    source_id: str
    arrival_time: float
    sequence: int
    profile_id: int
    values: tuple
    available: tuple

    def __post_init__(self):
        if not self.message_id or not self.source_id:
            raise ValueError('message and source identities are required')
        if not math.isfinite(self.arrival_time) or self.arrival_time < 0 or self.sequence < 0:
            raise ValueError('invalid arrival or sequence')
        if len(self.values) != 8 or len(self.available) != 8:
            raise ValueError('exactly eight fields and validity flags required')
        if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in self.values):
            raise ValueError('parameter values must be finite in [0,1]')
        if any(x not in (False, True) for x in self.available):
            raise ValueError('availability must contain boolean flags')
        object.__setattr__(self, 'values', tuple(float(x) for x in self.values))
        object.__setattr__(self, 'available', tuple(bool(x) for x in self.available))

    @classmethod
    def from_header(cls, message_id, source_id, arrival_time, sequence, header):
        return cls(message_id, source_id, arrival_time, sequence, header.profile_id,
                   header.values, header.available)


@dataclass(frozen=True)
class Decision:
    message_id: str
    policy: str
    score: float | None
    fallback: str | None = None


def tie_key(candidate):
    return candidate.arrival_time, candidate.sequence, candidate.message_id


def feature_matrix(candidates, medians):
    medians = np.asarray(medians, dtype=float)
    if medians.shape != (8,) or not np.isfinite(medians).all():
        raise ValueError('eight finite imputation values required')
    if not candidates:
        return np.empty((0, 16), dtype=float)
    values = np.asarray([row.values for row in candidates], dtype=float)
    masks = np.asarray([row.available for row in candidates], dtype=bool)
    return np.concatenate([np.where(masks, values, medians), masks.astype(float)], axis=1)


def weights_digest(weights, biases):
    data = json.dumps({'weights': weights, 'biases': biases}, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


class SemanticScheduler:
    KINDS = {'fifo', 'round_robin', 'random', 'importance', 'heuristic',
             'linear', 'mlp', 'classifier', 'regressor', 'lightgbm'}

    def __init__(self, kind='fifo', profile_id=None, medians=None, weights=(), biases=(),
                 lambda_n=0., seed=0, booster=None, disabled_fields=()):
        if kind not in self.KINDS:
            raise ValueError(f'unknown scheduler kind: {kind}')
        self.kind, self.profile_id = kind, profile_id
        self.medians = np.asarray([.5]*8 if medians is None else medians, dtype=float)
        feature_matrix([], self.medians)
        self.weights = [np.asarray(x, dtype=float) for x in weights]
        self.biases = [np.asarray(x, dtype=float) for x in biases]
        self.lambda_n, self.seed, self.booster = float(lambda_n), int(seed), booster
        self.disabled_fields = tuple(disabled_fields)
        if any(index not in range(8) for index in self.disabled_fields):
            raise ValueError('invalid ablated field index')
        self._rr_after = None
        if not math.isfinite(self.lambda_n) or self.lambda_n < 0:
            raise ValueError('invalid novelty weight')
        if kind in {'linear', 'mlp', 'classifier', 'regressor'}:
            self._validate_network()
        if kind == 'lightgbm' and booster is None:
            raise ValueError('tree artifact requires a loaded booster')

    def _validate_network(self):
        if not self.weights or len(self.weights) != len(self.biases):
            raise ValueError('weights and biases must describe a network')
        size = 16
        for weight, bias in zip(self.weights, self.biases):
            if weight.ndim == 1:
                if self.kind != 'linear':
                    raise ValueError('network weights must have input/output dimensions')
                weight.shape = (weight.size, 1)
            if (weight.ndim != 2 or weight.shape[0] != size or bias.shape not in
                    {(weight.shape[1],), ()} or not np.isfinite(weight).all() or
                    not np.isfinite(bias).all()):
                raise ValueError('incompatible or nonfinite network arrays')
            size = weight.shape[1]
        if size != (3 if self.kind == 'classifier' else 1):
            raise ValueError('unexpected network output dimension')

    def scores(self, candidates):
        matrix = feature_matrix(candidates, self.medians)
        for index in self.disabled_fields:
            matrix[:,index] = self.medians[index]
            matrix[:,index+8] = 0
        if self.kind in {'importance', 'heuristic'}:
            importance = np.asarray([c.values[3] if c.available[3] else self.medians[3] for c in candidates])
            if self.kind == 'importance':
                return importance
            novelty = np.asarray([c.values[7] if c.available[7] else self.medians[7] for c in candidates])
            return importance * (1 + self.lambda_n * novelty)
        if self.kind == 'random':
            return np.asarray([int.from_bytes(hashlib.sha256(f'{self.seed}|{c.message_id}'.encode()).digest()[:8], 'big') / 2**64 for c in candidates])
        if self.kind == 'lightgbm':
            return np.asarray(self.booster.predict(matrix), dtype=float)
        if self.kind in {'linear', 'mlp', 'classifier', 'regressor'}:
            values = matrix
            for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
                values = values @ weight + bias
                if index + 1 < len(self.weights):
                    values = np.maximum(values, 0)
            if self.kind == 'classifier':
                probabilities = np.exp(values - values.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                return probabilities @ np.array([0., 1/3, 1.])
            return values[:, 0]
        return np.zeros(len(candidates))

    def rank(self, candidates):
        candidates = tuple(candidates)
        if not candidates:
            return []
        if len({c.message_id for c in candidates}) != len(candidates):
            raise ValueError('duplicate message identity')
        profiles = {c.profile_id for c in candidates}
        if len(profiles) != 1 or (self.profile_id is not None and profiles != {self.profile_id}):
            raise ValueError('incompatible task profiles in decoder queue')
        ordered = sorted(candidates, key=tie_key)
        if self.kind not in {'fifo', 'round_robin', 'random'} and not any(any(c.available) for c in candidates):
            return [Decision(c.message_id, self.kind, None, 'all_semantics_unavailable') for c in ordered]
        if self.kind == 'fifo':
            return [Decision(c.message_id, self.kind, None) for c in ordered]
        if self.kind == 'round_robin':
            queues = defaultdict(deque)
            for candidate in ordered:
                queues[candidate.source_id].append(candidate)
            sources = sorted(queues)
            if self._rr_after in sources:
                offset = (sources.index(self._rr_after) + 1) % len(sources)
                sources = sources[offset:] + sources[:offset]
            ranked = []
            while any(queues.values()):
                for source in sources:
                    if queues[source]:
                        ranked.append(Decision(queues[source].popleft().message_id, self.kind, None))
            return ranked
        scores = self.scores(candidates)
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            return [Decision(c.message_id, self.kind, None, 'nonfinite_model_output') for c in ordered]
        indices = sorted(range(len(candidates)), key=lambda i: (-scores[i], tie_key(candidates[i])))
        return [Decision(candidates[i].message_id, self.kind, float(scores[i])) for i in indices]

    def select(self, candidates):
        candidates = tuple(candidates)
        ranked = self.rank(candidates)
        if not ranked:
            return None
        if self.kind == 'round_robin':
            self._rr_after = next(c.source_id for c in candidates if c.message_id == ranked[0].message_id)
        return ranked[0]

    def reset(self):
        self._rr_after = None


def load_scheduler(artifact_directory, profile_registry=None):
    root = Path(artifact_directory)
    with (root / 'model.json').open(encoding='utf-8') as stream:
        metadata = json.load(stream)
    if metadata.get('schema') != ARTIFACT_SCHEMA or metadata.get('header_version') != 3:
        raise ValueError('incompatible scheduler artifact schema/header')
    if tuple(metadata.get('feature_names', ())) != FEATURE_NAMES:
        raise ValueError('artifact feature order mismatch')
    profile = metadata['profile_id']
    if profile_registry is not None and profile not in profile_registry:
        raise ValueError('artifact profile missing from registry')
    weights, biases = metadata.get('weights', []), metadata.get('biases', [])
    if metadata.get('weights_sha256') != weights_digest(weights, biases):
        raise ValueError('scheduler weight checksum mismatch')
    kind, booster = metadata['kind'], None
    if kind == 'lightgbm':
        tree = root / metadata['tree_file']
        if tree.resolve().parent != root.resolve():
            raise ValueError('tree file must be inside artifact directory')
        if hashlib.sha256(tree.read_bytes()).hexdigest() != metadata.get('tree_sha256'):
            raise ValueError('tree checksum mismatch')
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(tree))
    return SemanticScheduler(kind, profile, metadata['medians'], weights, biases,
                             metadata.get('lambda_n', 0), metadata.get('seed', 0), booster,
                             metadata.get('disabled_fields', ()))
