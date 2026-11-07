"""Immutable header-only context and portable NumPy policy inference."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from Scheduler.decoder_scheduler import Decision, SchedulerCandidate, SemanticScheduler, tie_key

SCHEMA = 'decoder_context_v1'
OBSERVATION_PROFILE = 'v3_headers_history32'
FEATURE_NAMES = tuple([f'candidate_{i}' for i in range(16)] +
                      [f'history_mean_{i}' for i in range(8)] +
                      [f'history_valid_fraction_{i}' for i in range(8)] +
                      ['nearest_distance', 'nearest3_mean_distance', 'comparison_available',
                       'signature_match', 'history_present', 'attempt_fraction', 'remaining_fraction'])


@dataclass(frozen=True)
class DecisionContext:
    ready_candidates: tuple
    successful_header_history: tuple
    attempts_completed: int
    initial_candidate_count: int
    task_profile: int
    observation_profile: str = OBSERVATION_PROFILE

    def __post_init__(self):
        for name in ('ready_candidates', 'successful_header_history'):
            rows = tuple(getattr(self, name))
            if any(not isinstance(row, SchedulerCandidate) for row in rows):
                raise TypeError('context requires immutable SchedulerCandidate headers')
            if any(row.profile_id != self.task_profile for row in rows):
                raise ValueError('context contains incompatible task profiles')
            object.__setattr__(self, name, rows)
        if len({c.message_id for c in self.ready_candidates}) != len(self.ready_candidates):
            raise ValueError('duplicate ready candidate identity')
        if (type(self.attempts_completed) is not int or type(self.initial_candidate_count) is not int
                or self.attempts_completed < 0 or self.initial_candidate_count < 0
                or self.attempts_completed > self.initial_candidate_count):
            raise ValueError('invalid episode progress')
        if len(self.successful_header_history) > 32:
            raise ValueError('history exceeds observation profile limit')
        if self.observation_profile != OBSERVATION_PROFILE:
            raise ValueError('unsupported observation profile')


class StaticPolicyAdapter:
    """Reuse existing policies without copying their rules or queue state."""
    def __init__(self, policy):
        self.policy = policy
        self.profile_id = policy.profile_id

    def select(self, context):
        return self.policy.select(context.ready_candidates)

    def reset(self):
        self.policy.reset()


def context_matrix(context, medians, fields=tuple(range(8))):
    """39 features; excluded fields disappear from ALL derived relations."""
    medians = np.asarray(medians, dtype=float)
    if medians.shape != (8,) or not np.isfinite(medians).all():
        raise ValueError('eight finite medians required')
    fields = tuple(fields)
    if len(set(fields)) != len(fields) or any(type(i) is not int or i not in range(8) for i in fields):
        raise ValueError('invalid field selection')
    enabled = np.zeros(8, dtype=bool)
    enabled[list(fields)] = True
    history = context.successful_header_history
    h_values = np.asarray([c.values for c in history], float).reshape(-1, 8)
    h_masks = np.asarray([c.available for c in history], bool).reshape(-1, 8) & enabled
    counts = h_masks.sum(axis=0)
    h_mean = np.divide((h_values * h_masks).sum(axis=0), counts,
                       out=medians.copy(), where=counts > 0)
    h_mean[~enabled] = 0.
    h_fraction = counts / max(1, len(history))
    output = []
    for candidate in context.ready_candidates:
        values = np.asarray(candidate.values)
        mask = np.asarray(candidate.available) & enabled
        candidate_values = np.where(mask, values, medians)
        candidate_values[~enabled] = 0.
        coavailable = h_masks & mask
        common = coavailable.sum(axis=1)
        distances = np.divide((np.abs(h_values-values) * coavailable).sum(axis=1), common,
                              out=np.ones(len(history)), where=common > 0)
        usable = np.sort(distances[common > 0])
        signature_match = any(np.array_equal(mask, hm) and np.any(mask)
                              and np.array_equal(values[mask], hv[mask])
                              for hv, hm in zip(h_values, h_masks))
        progress = [float(usable[0]) if usable.size else 0.,
                    float(usable[:3].mean()) if usable.size else 0., float(usable.size > 0),
                    float(signature_match), float(bool(history)),
                    context.attempts_completed/max(1, context.initial_candidate_count),
                    len(context.ready_candidates)/max(1, context.initial_candidate_count)]
        output.append(np.concatenate([candidate_values, mask.astype(float), h_mean, h_fraction, progress]))
    return np.asarray(output, dtype=float).reshape(-1, len(FEATURE_NAMES))


def artifact_digest(data):
    clean = {key: value for key, value in data.items() if key != 'sha256'}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def tree_predict(tree, matrix):
    arrays = {name: np.asarray(tree[name]) for name in ('left', 'right', 'feature', 'threshold', 'value')}
    size = len(arrays['left'])
    if size == 0 or any(array.shape != (size,) for array in arrays.values()):
        raise ValueError('invalid exported tree shape')
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError('nonfinite exported tree')
    predictions = []
    for row in np.asarray(matrix, dtype=np.float32):
        node = 0
        for _ in range(size + 1):
            left, right = int(arrays['left'][node]), int(arrays['right'][node])
            if left == right == -1:
                predictions.append(float(arrays['value'][node]))
                break
            feature = int(arrays['feature'][node])
            if not 0 <= feature < matrix.shape[1] or not 0 <= left < size or not 0 <= right < size:
                raise ValueError('invalid tree node index')
            node = left if row[feature] <= arrays['threshold'][node] else right
        else:
            raise ValueError('cycle in exported tree')
    return np.asarray(predictions)


def validate_tree(tree, feature_count):
    """Validate every branch, including branches not hit by a dummy input."""
    arrays = {name: np.asarray(tree[name], dtype=float)
              for name in ('left', 'right', 'feature', 'threshold', 'value')}
    size = len(arrays['left'])
    if not 0 < size <= 100000 or any(a.shape != (size,) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('invalid exported tree arrays')
    for name in ('left', 'right', 'feature'):
        if not np.equal(arrays[name], np.floor(arrays[name])).all():
            raise ValueError('tree indices must be exact integers')
    incoming = np.zeros(size, dtype=int)
    edges = []
    for node in range(size):
        left, right = int(arrays['left'][node]), int(arrays['right'][node])
        if left == right == -1:
            edges.append(())
            continue
        if (not 0 <= left < size or not 0 <= right < size or left == right
                or not 0 <= arrays['feature'][node] < feature_count):
            raise ValueError('invalid exported tree branch')
        incoming[left] += 1
        incoming[right] += 1
        edges.append((left, right))
    if incoming[0] != 0 or np.any(incoming[1:] != 1):
        raise ValueError('tree must have one root and single-parent branches')
    visited, pending = set(), [0]
    while pending:
        node = pending.pop()
        if node in visited:
            raise ValueError('cycle in exported tree')
        visited.add(node)
        pending.extend(edges[node])
    if len(visited) != size:
        raise ValueError('unreachable exported tree nodes')


class ContextualPolicy:
    def __init__(self, artifact):
        task_ids = artifact.get('task_ids', [])
        if (not isinstance(task_ids, list) or len(set(task_ids)) != len(task_ids)
                or any(type(value) is not int or not 0 < value < 65536 for value in task_ids)):
            raise ValueError('invalid known-task conditioning')
        self.feature_names = FEATURE_NAMES + tuple(f'task_{value}' for value in task_ids)
        if (artifact.get('schema') != SCHEMA or artifact.get('observation_profile') != OBSERVATION_PROFILE
                or artifact.get('sha256') != artifact_digest(artifact)
                or tuple(artifact.get('feature_names', ())) != self.feature_names):
            raise ValueError('incompatible or corrupted contextual artifact')
        self.artifact = json.loads(json.dumps(artifact, allow_nan=False))
        self.profile_id = artifact.get('profile_id')
        self.fallback_count = 0
        self._fallback = SemanticScheduler('importance', self.profile_id, artifact['medians'])
        fields = artifact.get('fields')
        if (not isinstance(fields, list) or len(set(fields)) != len(fields)
                or any(type(index) is not int or index not in range(8) for index in fields)):
            raise ValueError('invalid artifact field selection')
        if np.any(np.asarray(artifact['medians']) < 0) or np.any(np.asarray(artifact['medians']) > 1):
            raise ValueError('artifact medians must be in [0,1]')
        if self.profile_id is not None and (type(self.profile_id) is not int or not 0 < self.profile_id < 65536):
            raise ValueError('invalid artifact profile identity')
        if artifact['model'].get('kind') == 'boosted_trees':
            if not artifact['model'].get('trees'):
                raise ValueError('empty boosted model')
            for tree in artifact['model']['trees']:
                validate_tree(tree, len(self.feature_names))
        # Eager validation before the service accepts jobs.
        dummy = np.zeros((1, len(self.feature_names)))
        self._predict(dummy)

    def _predict(self, matrix):
        model = self.artifact['model']
        mean, scale = np.asarray(model['mean']), np.asarray(model['scale'])
        if (mean.shape != (len(self.feature_names),) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError('invalid model preprocessing')
        matrix = (matrix - mean)/scale
        if model['kind'] == 'linear':
            coefficient = np.asarray(model['coefficient'])
            if coefficient.shape != mean.shape or not np.isfinite(coefficient).all() or not math.isfinite(model['intercept']):
                raise ValueError('invalid linear model')
            return matrix @ coefficient + model['intercept']
        if model['kind'] == 'boosted_trees':
            result = np.full(len(matrix), model['initial'], dtype=float)
            if not math.isfinite(model['initial']) or not math.isfinite(model['learning_rate']):
                raise ValueError('nonfinite boosted model')
            for tree in model['trees']:
                result += model['learning_rate'] * tree_predict(tree, matrix)
            return result
        raise ValueError('unsupported model kind')

    def scores(self, context):
        if self.profile_id is not None and context.task_profile != self.profile_id:
            raise ValueError('policy and task profiles differ')
        matrix = context_matrix(context, self.artifact['medians'], tuple(self.artifact['fields']))
        if self.artifact.get('pointwise', False):
            matrix[:, 16:] = 0.
        if self.artifact.get('masks_only', False):
            matrix[:, :8] = 0.
            matrix[:, 16:] = 0.
        task_ids = self.artifact.get('task_ids', [])
        if task_ids:
            task_features = np.tile([float(context.task_profile == value) for value in task_ids], (len(matrix), 1))
            matrix = np.concatenate([matrix, task_features], axis=1)
        return self._predict(matrix)

    def select(self, context):
        if not context.ready_candidates:
            return None
        scores = self.scores(context)
        if scores.shape != (len(context.ready_candidates),) or not np.isfinite(scores).all():
            self.fallback_count += 1
            fallback = self._fallback.select(context.ready_candidates)
            return Decision(fallback.message_id, 'contextual', fallback.score, 'nonfinite_model_output')
        index = min(range(len(scores)), key=lambda i: (-scores[i], tie_key(context.ready_candidates[i])))
        return Decision(context.ready_candidates[index].message_id, 'contextual', float(scores[index]))

    def reset(self):
        self.fallback_count = 0
        self._fallback.reset()


def load_contextual_policy(path):
    return ContextualPolicy(json.loads(Path(path).read_text(encoding='utf-8')))


class OverlapPenaltyPolicy:
    """Standalone utility discounted by predicted directional delivered overlap."""
    def __init__(self, utility_policy, overlap_policy, strength=1.):
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError('overlap penalty must be in [0,1]')
        self.utility = utility_policy
        self.overlap = overlap_policy
        self.strength = strength
        self.profile_id = utility_policy.profile_id

    def select(self, context):
        if not context.ready_candidates:
            return None
        utility = np.clip(self.utility.scores(context), 0., 1.)
        penalty = np.zeros(len(utility))
        for i, candidate in enumerate(context.ready_candidates):
            for header in context.successful_header_history:
                pair = DecisionContext((candidate,), (header,), 1, 2, context.task_profile)
                penalty[i] = max(penalty[i], float(np.clip(self.overlap.scores(pair)[0], 0., 1.)))
        scores = utility * (1 - self.strength*penalty)
        if not np.isfinite(scores).all():
            fallback = self.utility._fallback.select(context.ready_candidates)
            return Decision(fallback.message_id, 'overlap_penalty', fallback.score, 'nonfinite_model_output')
        index = min(range(len(scores)), key=lambda i: (-scores[i], tie_key(context.ready_candidates[i])))
        return Decision(context.ready_candidates[index].message_id, 'overlap_penalty', float(scores[index]))

    def reset(self):
        self.utility.reset()
        self.overlap.reset()
