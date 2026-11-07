"""Training-only utilities and portable exports; runtime policy sees no truth."""
from dataclasses import dataclass
import numpy as np

from Scheduler.contextual_decoder_scheduler import (
    DecisionContext, ContextualPolicy, context_matrix, FEATURE_NAMES, SCHEMA,
    OBSERVATION_PROFILE, artifact_digest,
)


@dataclass(frozen=True)
class TrainingDecision:
    context: DecisionContext
    targets: tuple
    weight: float = 1.


def matrices(decisions, medians, fields, pointwise=False, masks_only=False, task_ids=()):
    xs, ys, ws = [], [], []
    for decision in decisions:
        matrix = context_matrix(decision.context, medians, fields)
        if len(decision.targets) != len(matrix):
            raise ValueError('training target and candidate count differ')
        if pointwise:
            matrix[:, 16:] = 0.
        if masks_only:
            matrix[:, :8] = 0.
            matrix[:, 16:] = 0.
        if task_ids:
            task = np.tile([float(decision.context.task_profile == value) for value in task_ids], (len(matrix), 1))
            matrix = np.concatenate([matrix, task], axis=1)
        xs.append(matrix)
        ys.extend(decision.targets)
        # Each decision contributes its allocated episode mass regardless of candidate count.
        ws.extend([decision.weight/max(1, len(matrix))]*len(matrix))
    if not xs or sum(map(len, xs)) == 0:
        raise ValueError('no usable supervised decisions')
    x, y, w = np.concatenate(xs), np.asarray(ys, float), np.asarray(ws, float)
    if (not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all()
            or np.any(w <= 0) or np.any(y < 0) or np.any(y > 1)):
        raise ValueError('invalid training arrays')
    return x, y, w


def fit_medians(decisions):
    result = []
    # Header values come only from the provided training decisions.
    for index in range(8):
        values = [c.values[index] for d in decisions for c in d.context.ready_candidates if c.available[index]]
        result.append(float(np.median(values)) if values else .5)
    return result


def export_tree(tree):
    return dict(left=tree.children_left.tolist(), right=tree.children_right.tolist(),
                feature=tree.feature.tolist(), threshold=tree.threshold.tolist(),
                value=tree.value[:, 0, 0].tolist())


def fit_policy(decisions, fields=tuple(range(8)), kind='linear', alpha=1., pointwise=False,
               masks_only=False, task_ids=(), tree_depth=2, min_leaf=5, learning_rate=.05,
               seed=0, provenance=None):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    medians = fit_medians(decisions)
    x, y, weight = matrices(decisions, medians, fields, pointwise, masks_only, task_ids)
    mean = np.average(x, axis=0, weights=weight)
    variance = np.average((x-mean)**2, axis=0, weights=weight)
    scale = np.where((np.ptp(x, axis=0) == 0) | (variance == 0), 1., np.sqrt(variance))
    z = (x-mean)/scale
    if kind == 'linear':
        model = Ridge(alpha=alpha).fit(z, y, sample_weight=weight)
        exported = dict(kind='linear', coefficient=model.coef_.tolist(), intercept=float(model.intercept_))
    elif kind == 'boosted_trees':
        model = GradientBoostingRegressor(n_estimators=50, max_depth=tree_depth,
                                         min_samples_leaf=min_leaf, learning_rate=learning_rate,
                                         loss='squared_error', random_state=seed)
        model.fit(z, y, sample_weight=weight)
        exported = dict(kind='boosted_trees', initial=float(model.init_.constant_[0, 0]),
                        learning_rate=float(model.learning_rate),
                        trees=[export_tree(estimator[0].tree_) for estimator in model.estimators_])
    else:
        raise ValueError('unsupported training family')
    exported.update(mean=mean.tolist(), scale=scale.tolist())
    data = dict(schema=SCHEMA, observation_profile=OBSERVATION_PROFILE,
                feature_names=list(FEATURE_NAMES)+[f'task_{value}' for value in task_ids],
                profile_id=None, task_ids=list(task_ids), fields=list(fields), medians=medians,
                pointwise=pointwise, masks_only=masks_only, model=exported,
                training_scope='caller_supplied_training_groups_only', provenance=provenance or {})
    data['sha256'] = artifact_digest(data)
    policy = ContextualPolicy(data)
    expected = model.predict(z)
    actual = policy._predict(x)
    if not np.allclose(expected, actual, rtol=1e-7, atol=1e-9):
        raise RuntimeError('portable export differs from trained estimator')
    return policy


def subset_candidates(limit=None):
    subsets = [tuple(i for i in range(8) if bits & (1 << i)) for bits in range(1, 256)]
    if limit is not None:
        if not 1 <= limit <= 255:
            raise ValueError('subset limit must be in [1,255]')
        subsets = subsets[:limit]
    return tuple(dict.fromkeys(subsets+[(3,), tuple(range(8))]))


def choose_config(rows):
    defined = [row for row in rows if row['mean_A_K'] is not None]
    if not defined:
        raise ValueError('selection has no defined validation objective')
    best = max(row['mean_A_K'] for row in defined)
    tied = [row for row in defined if row['mean_A_K'] >= best-1e-12]
    return min(tied, key=lambda row: (len(row['fields']), -row['alpha'], tuple(row['fields'])))
