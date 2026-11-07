"""Contract and causal regressions for the external-parameter stream path."""
from dataclasses import replace
import json
import sys
import numpy as np
import pytest

from Evaluation.stream_simulator import StreamConfig, StreamEnvironment, build_stream, compact_values, ParameterQueue
from Evaluation.stream_io import load_offered_stream
from Scheduler.scene_policy import ScenePolicy
from Scheduler.training.stream_pilot import evaluate


def stream(**kwargs): return build_stream(StreamConfig(seed=5, messages=18, capture_window=3., **kwargs))


def test_seeded_outcomes_and_no_classifier_import():
    before = 'parameters.spi' in sys.modules
    episode, manifest = stream(max_ues=100, failure_probability=.5)
    other, second = stream(max_ues=100, failure_probability=.5)
    assert manifest == second and episode.failures == other.failures
    assert len(manifest['ues']) == 100
    assert ('parameters.spi' in sys.modules) == before
    assert episode.horizon == 3. + .15 + 10.
    assert {p['profile_id'] for p in manifest['ues']}


def test_policy_independent_inputs_and_accounting():
    e, m = stream(max_ues=12, failure_probability=.4)
    a = evaluate(ScenePolicy(), e, m)
    b = evaluate(ScenePolicy(kind='random', seed=2), e, m)
    assert a['input_stream_sha256'] == b['input_stream_sha256']
    assert a['accounting']['offered'] == 18
    assert np.isclose(a['reward_sum'], a['J_time'])
    assert np.isclose(b['reward_sum'], b['J_time'])


def test_failure_never_enters_success_history_and_busy_dispatch():
    e, m = stream(failure_probability=1.)
    env = StreamEnvironment(e)
    while not env.finished:
        context = env.observation()
        env.step(context.ready_candidates[0].message_id)
        assert not env.controller.history
    assert not env.covered
    assert env.metrics()['J_time'] == 0.


def test_ingress_missing_zero_cold_start_and_no_truth():
    e, _ = stream()
    row = json.loads(e.messages[0].header_bytes)
    row.update(spi=None, spi_available=False, novelty=0., novelty_cold_start=False)
    values, available, cold = compact_values(row)
    assert not any(available[:4]) and available[5] and values[5] == 0
    row['novelty_cold_start'] = True
    _, available, cold = compact_values(row)
    assert cold[5] and not available[5]
    row['reference_priority'] = 0
    with pytest.raises(ValueError): compact_values(row)


def test_detector_cross_field_contract_and_zero_utility():
    e, _ = stream()
    row = json.loads(e.messages[0].header_bytes)
    row['scene'] = dict(instances=0, categories=0, largest_box_fraction=None, union_box_coverage=0.)
    assert compact_values(row)[1][9]
    for invalid in (dict(instances=0,categories=1,largest_box_fraction=None,union_box_coverage=0.),
                    dict(instances=1,categories=0,largest_box_fraction=.1,union_box_coverage=.2),
                    dict(instances=0,categories=0,largest_box_fraction=0.,union_box_coverage=0.),
                    dict(instances=1,categories=1,largest_box_fraction=.3,union_box_coverage=.2)):
        with pytest.raises(ValueError): compact_values({**row, 'scene':invalid})
    env = StreamEnvironment(replace(e, weights={f:0. for f in e.weights}))
    while not env.finished: env.step(env.observation().ready_candidates[0].message_id)
    assert env.metrics()['first_useful_completion_time'] is None


def test_controller_lifecycle_and_arrival_during_service():
    e, _ = stream()
    env = StreamEnvironment(e)
    env.action_policy.action = env.observation().ready_candidates[0].message_id
    job = env.controller.dispatch(env.now)
    assert env.controller.dispatch(env.now) is None
    with pytest.raises(ValueError): env.controller.complete('foreign', 'succeeded', env.now+1)
    env.controller.complete(job.message.message_id, 'failed', env.now+1)
    with pytest.raises(ValueError): env.controller.complete(job.message.message_id, 'failed', env.now+1)
    env = StreamEnvironment(e)
    # A long service admits later arrivals before the next selection.
    first = env.observation().ready_candidates[0].message_id
    env.episode = replace(e, durations={**e.durations, first: 2.})
    env.step(first)
    assert all(c.arrival_time <= env.now for c in env.observation().ready_candidates)


def test_no_backlog_invariance():
    e, m = stream()
    messages = tuple(replace(msg, arrival_time=i*3.) for i,msg in enumerate(e.messages))
    records = [{**r, 'arrival': i*3.} for i,r in enumerate(m['records'])]
    # Use unique report IDs to make follow-up simple in this control.
    from Evaluation.scene_environment import SceneEpisode
    facts = {msg.message_id: frozenset([msg.message_id]) for msg in messages}
    e = SceneEpisode(messages, facts, e.durations, 18*3.+10, {msg.message_id: 1. for msg in messages})
    m = {**m, 'records': [{**r,'evaluator':{**r['evaluator'],'fact_id':r['message_id']}} for r in records]}
    a = evaluate(ScenePolicy(), e, m)
    b = evaluate(ScenePolicy(kind='random', seed=2), e, m)
    assert a['events_sha256'] == b['events_sha256']


def test_external_stream_without_references_does_not_invent_truth(tmp_path):
    e, _ = stream()
    msg = e.messages[0]
    p = tmp_path/'input.jsonl'
    p.write_text(json.dumps({'schema':'stream_received_v1','source_id':msg.source_id,
        'session_id':msg.session_id,'message_token':msg.message_token,'sequence':msg.sequence,
        'arrival_time':msg.arrival_time,'parameters':json.loads(msg.header_bytes),'payload_ref':'opaque'})+'\n')
    episode, manifest = load_offered_stream(p)
    result = evaluate(ScenePolicy(), episode, manifest)
    assert result['J_time'] is None
    assert result['priority']['unknown_reference_messages'] == 1
    _, longer = load_offered_stream(p, observation_end=120.)
    assert manifest['sha256'] != longer['sha256']


@pytest.mark.parametrize('change', [{'max_ues':0}, {'max_ues':101}, {'failure_probability':float('nan')},
                                    {'service_mean':0}, {'messages':0}])
def test_bad_configuration_rejected(change):
    with pytest.raises(ValueError): StreamConfig(**change)
