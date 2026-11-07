from dataclasses import replace
import numpy as np
import pytest

from Scheduler.contextual_decoder_scheduler import (
    DecisionContext, ContextualPolicy, context_matrix, artifact_digest,
    FEATURE_NAMES, SCHEMA, OBSERVATION_PROFILE,
)
from Scheduler.decoder_scheduler import SchedulerCandidate


def candidate(name, value=.5, mask=True):
    return SchedulerCandidate(name, 'ue', 0., int(name), 1001, (value,)*8, (mask,)*8)


def artifact():
    coefficients = [0.]*len(FEATURE_NAMES)
    coefficients[3] = 1.
    result = dict(schema=SCHEMA, observation_profile=OBSERVATION_PROFILE,
                  feature_names=list(FEATURE_NAMES), profile_id=1001, fields=list(range(8)), medians=[.5]*8,
                  model=dict(kind='linear', mean=[0.]*len(FEATURE_NAMES), scale=[1.]*len(FEATURE_NAMES),
                             coefficient=coefficients, intercept=0.))
    result['sha256'] = artifact_digest(result)
    return result


def test_context_features_hand_reference_and_permutation():
    a, b, h = candidate('1', .2), candidate('2', .8), candidate('3', .2)
    ctx = DecisionContext((a, b), (h,), 1, 3, 1001)
    x = context_matrix(ctx, [.5]*8)
    assert x.shape == (2, 39)
    assert np.allclose(x[:, 16:24], .2)
    assert x[:, 32].tolist() == pytest.approx([0., .6])
    assert x[:, 35].tolist() == [1., 0.]
    policy = ContextualPolicy(artifact())
    scores = policy.scores(ctx)
    reversed_ctx = replace(ctx, ready_candidates=(b, a))
    assert np.allclose(policy.scores(reversed_ctx), scores[::-1])
    assert policy.select(ctx).message_id == '2'


def test_excluded_fields_cannot_leak_through_history_or_distance():
    ctx = DecisionContext((candidate('1'),), (candidate('2', .2),), 1, 2, 1001)
    x = context_matrix(ctx, [.5]*8, (3,))
    changed = replace(ctx, ready_candidates=(replace(ctx.ready_candidates[0], values=(.9, .9, .9, .5, .9, .9, .9, .9)),),
                      successful_header_history=(replace(ctx.successful_header_history[0], values=(.8, .8, .8, .2, .8, .8, .8, .8)),))
    assert np.array_equal(x, context_matrix(changed, [.5]*8, (3,)))
    assert context_matrix(replace(ctx, successful_header_history=()), [.5]*8).shape == (1, 39)
    assert np.isfinite(context_matrix(DecisionContext((), (), 0, 0, 1001), [.5]*8)).all()


def test_artifact_failure_and_nonfinite_fallback():
    corrupted = artifact()
    corrupted['model']['intercept'] = 2.
    with pytest.raises(ValueError):
        ContextualPolicy(corrupted)
    policy = ContextualPolicy(artifact())
    policy._predict = lambda x: np.full(len(x), np.nan)
    ctx = DecisionContext((candidate('1', .2), candidate('2', .8)), (), 0, 2, 1001)
    decision = policy.select(ctx)
    assert decision.message_id == '2' and decision.fallback == 'nonfinite_model_output'
    assert policy.fallback_count == 1
    assert not hasattr(ctx, 'labels') and not hasattr(ctx.ready_candidates[0], 'payload_ref')


def test_unvisited_malformed_branches_and_fractional_indices_fail_before_admission():
    data = artifact()
    data['fields'] = [8]
    data['sha256'] = artifact_digest(data)
    with pytest.raises(ValueError, match='field'):
        ContextualPolicy(data)
    for right in (99, 1.5):
        data = artifact()
        data['model'].update(kind='boosted_trees', initial=0., learning_rate=.1,
                             trees=[dict(left=[1, -1, -1], right=[right, -1, -1],
                                         feature=[3, -2, -2], threshold=[.5, -2., -2.], value=[0., 1., 2.])])
        data['sha256'] = artifact_digest(data)
        with pytest.raises(ValueError):
            ContextualPolicy(data)
