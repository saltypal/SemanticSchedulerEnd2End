"""Known-direction learning, export and reload check, not just an import test."""
from Scheduler.scene_policy import SceneCandidate, SceneContext, ScenePolicy
from Scheduler.training.scene_learners import Example, fit_supervised
from Evaluation.stream_simulator import REGISTRY_HASH
import pytest


@pytest.mark.parametrize('kind', ['linear_choice', 'ranknet'])
def test_pairwise_learning_prefers_supported_candidate_and_reloads(kind):
    a = SceneCandidate('a', 'source', 0., 0, (1.,0.,0.,0.,.8,.9,1.,1.,.2,.2),
                       (True,)*10, (False,)*10, REGISTRY_HASH)
    b = SceneCandidate('b', 'other', 0., 1, (0.,0.,0.,1.,-.2,0.,1.,1.,.1,.1),
                       (True,)*10, (False,)*10, REGISTRY_HASH)
    context = SceneContext((a,b), (), 0, 2, 0., 10., REGISTRY_HASH)
    example = Example(context, (1.,0.), 'known-pair', 'synthetic_fixture', ((0,1,1.),))
    artifact, history, state = fit_supervised([example], kind, epochs=20, fixture=True)
    policy = ScenePolicy(artifact)
    assert policy.scores(context)[0] > policy.scores(context)[1]
    assert policy.select(context).message_id == 'a'
    assert history[-1]['loss'] < history[0]['loss']
    assert state['optimizer']['state']
    reloaded, _, resumed = fit_supervised([example], kind, epochs=21,
                                         fixture=True, checkpoint=state)
    assert resumed['epoch'] == 21
    assert ScenePolicy(reloaded).select(context).message_id == 'a'
