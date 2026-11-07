"""End-to-end numerical and portable-runtime regression tests (CPU, no vision downloads)."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch

from Evaluation.semantic_learning_env import (
    BUDGETS, Episode, Environment, Message, Rule, Snapshot, ell, make_episode)
from Scheduler.semantic_learning_policy import NumpyRanker, arrays, fit_normalizer
from Scheduler.training.semantic_learning_models import (
    KINDS, Sample, Scorer, export, fit_lambdamart, fit_supervised, parity, tensors)
from Scheduler.training.semantic_learning_rl import ActorCritic, train_dqn, train_ppo
from Scheduler.training.preference_reward import (
    FrozenReward, RatePath, fit_reward, integral, relabel_replay, transition_path)


def fixture(count=8, seed=7):
    rng = np.random.default_rng(seed)
    messages = []
    for i in range(count):
        x = np.zeros(387, dtype=np.float32)
        x[:186] = rng.random(186)
        x[186:374] = 1
        x[374 + i % 4] = 1
        x[378:380] = rng.random(2)
        x[380:386] = 1
        messages.append(Message(i, i % 4, 0, 0., i, x))
    return Episode(tuple(messages), tuple(i % 4 for i in range(count)),
                   tuple("fact-" + str(i // 2) for i in range(count)),
                   (1.,) * count, (False,) * count, 20., {})


@pytest.fixture
def episode():
    return fixture()


@pytest.fixture
def pre(episode):
    return fit_normalizer([episode])


def reference(artifact, snapshot):
    scores = NumpyRanker(artifact).scores(snapshot)
    return float(scores[0] - scores[-1]), []


@pytest.mark.parametrize("kind", ["linear", "mlp", "classifier", "deepsets", "attention", "temporal"])
def test_export_and_permutation_equivariance(kind, episode, pre):
    torch.manual_seed(3)
    model = Scorer(kind).eval()
    snapshot = Environment(episode).snapshot()
    artifact = export(model, pre, kind, {"task": "D"},
                      class_weights=np.ones(4) if kind == "classifier" else None)
    assert parity(model, artifact, [snapshot])["maximum_absolute_error"] < 3e-4
    portable = NumpyRanker(artifact)
    before = portable.raw(snapshot)
    permutation = np.arange(len(snapshot.ready))[::-1]
    changed = replace(snapshot, ready=tuple(snapshot.ready[i] for i in permutation),
                      ue_windows=tuple(snapshot.ue_windows[i] for i in permutation))
    np.testing.assert_allclose(portable.raw(changed), before[permutation], rtol=3e-4, atol=3e-4)
    assert portable.select(replace(snapshot, ready=(), ue_windows=())) is None


@pytest.mark.parametrize("kind", ["linear", "mlp", "classifier", "deepsets"])
def test_independent_batch_scores_and_gradients(kind, episode, pre):
    torch.manual_seed(2)
    model = Scorer(kind)
    one = Environment(episode).snapshot()
    two = replace(one, ready=one.ready[:3], ue_windows=one.ue_windows[:3])
    inputs = [tensors(state, pre, kind=kind) for state in (one, two)]
    separate = [model(item) for item in inputs]
    sum(value.sum() for value in separate).backward()
    expected = {key: value.grad.clone() for key, value in model.named_parameters()}
    model.zero_grad()
    batched = model.forward_batch(inputs)
    for actual, wanted in zip(batched, separate):
        torch.testing.assert_close(actual, wanted, atol=2e-6, rtol=2e-6)
    sum(value.sum() for value in batched).backward()
    for key, value in model.named_parameters():
        torch.testing.assert_close(value.grad, expected[key], atol=2e-5, rtol=2e-5)


def test_actor_critic_batch_preserves_state_pooling(episode, pre):
    model = ActorCritic()
    snapshot = Environment(episode).snapshot()
    other = replace(snapshot, ready=snapshot.ready[:2], ue_windows=snapshot.ue_windows[:2])
    inputs = [tensors(state, pre, kind="deepsets") for state in (snapshot, other)]
    logits, values = model.forward_batch(inputs)
    for index, item in enumerate(inputs):
        expected_logits, expected_value = model(item)
        torch.testing.assert_close(logits[index], expected_logits)
        torch.testing.assert_close(values[index], expected_value)


@pytest.mark.parametrize("kind", ["fifo", "round_robin", "random", "spi_fifo",
                                 "relevance", "novelty", "spi_novelty", "predicted_edf"])
def test_policy_replay_and_exact_delay_reward(kind, episode):
    env = Environment(episode)
    policy = Rule(kind)
    reward = 0.
    while not env.done:
        state = env.snapshot()
        selected = policy.select(state)
        assert selected in {message.token for message in state.ready}
        _, value, _ = env.step(selected)
        reward += value
    assert reward == pytest.approx(-env.cost(episode.horizon) / 100.)
    metrics = env.metrics()
    assert metrics["selected"] + metrics["queued_at_horizon"] == metrics["offered"]
    assert metrics["completed"] + metrics["in_flight"] == metrics["selected"]


def test_empty_episode_and_nonfinite_inputs(episode):
    empty = Episode((), (), (), (), (), 2., {})
    assert Environment(empty).done
    assert Environment(empty).metrics()["J_D"] is None
    for value in (float("nan"), float("inf"), -1.):
        with pytest.raises(ValueError):
            replace(episode, horizon=value)
    snapshot = Environment(episode).snapshot()
    with pytest.raises(ValueError):
        replace(snapshot, now=float("nan"))
    with pytest.raises(ValueError):
        replace(snapshot, processed=((episode.messages[0], 99.),))
    with pytest.raises(ValueError):
        ell(1., 0.)


def test_arrivals_during_service_failure_no_retry_and_horizon(episode):
    messages = (episode.messages[0],
                replace(episode.messages[1], arrival=.5),
                replace(episode.messages[2], arrival=1.5))
    ep = Episode(messages, (0, 1, 2), ("a", "b", "c"),
                 (1., 5., 1.), (True, False, False), 2., {})
    env = Environment(ep)
    first, _, done = env.step(0)
    assert not done and [message.token for message in first.ready] == [1]
    assert not first.processed  # Failed attempts are not delivered evidence.
    last, _, done = env.step(1)
    assert done and last is None
    assert env.metrics()["failed"] == 1
    assert env.metrics()["in_flight"] == 1
    assert 2 in env.ready and 0 not in env.ready
    with pytest.raises(ValueError):
        env.step(0)


def test_runtime_observation_excludes_reference_and_future(episode, pre):
    env = Environment(episode)
    snapshot = env.snapshot()
    first = arrays(snapshot, pre)
    altered = replace(episode, reference_tiers=tuple(reversed(episode.reference_tiers)),
                      fact_ids=tuple("different" for _ in episode.messages),
                      horizon=1000.)
    second = arrays(Environment(altered).snapshot(), pre)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    with pytest.raises(ValueError):
        replace(snapshot, ready=(replace(snapshot.ready[0], arrival=1.),), ue_windows=((),))


@pytest.mark.parametrize("arm", list(KINDS))
def test_every_supervised_learner_updates_reloads_and_exports(arm, episode, pre, tmp_path):
    snapshot = Environment(episode).snapshot()
    target = Environment(episode).target(snapshot)
    samples = [Sample(snapshot, target, np.arange(8) % 4, "episode-0")]
    output = tmp_path / arm
    artifact, history = fit_supervised(
        samples, pre, arm, output, lambda model: reference(model, snapshot),
        epochs=3, patience=8, seconds=30.)
    assert history and json.loads((output / "parity.json").read_text())["snapshots"] == 1
    checkpoint = torch.load(output / "resume.pt", weights_only=False)
    assert checkpoint["updates"] > 0 and checkpoint["optimizer"]["state"]
    loaded = NumpyRanker(json.loads((output / "model.json").read_text()))
    assert loaded.select(snapshot) == NumpyRanker(artifact).select(snapshot)
    _, more = fit_supervised(
        samples, pre, arm, output, lambda model: reference(model, snapshot),
        epochs=4, resume=output / "resume.pt", seconds=30.)
    assert more[-1]["epoch"] == 3


def test_lambdamart_export_has_real_native_parity(episode, pre, tmp_path):
    snapshot = Environment(episode).snapshot()
    samples = [Sample(snapshot, Environment(episode).target(snapshot), np.arange(8) % 4, str(i))
               for i in range(5)]
    artifact = fit_lambdamart(samples, pre, tmp_path)
    assert (tmp_path / "lightgbm.txt").exists()
    assert np.isfinite(NumpyRanker(artifact).scores(snapshot)).all()


@pytest.mark.parametrize("arm", ["masked_double_dqn", "td_per", "rl_vlm_f", "vlm_replay"])
def test_every_dqn_arm_updates_checkpoint_resumes_and_exports(arm, episode, pre, tmp_path):
    snapshot = Environment(episode).snapshot()
    reward = None
    guide = None
    if arm == "rl_vlm_f":
        reward_model = torch.nn.Linear(1172, 1)
        reward = FrozenReward(reward_model, pre, "fixture-v1")
    if arm == "vlm_replay":
        guide = lambda state, token, nxt: 1.5 if token % 2 else .5
    factory = lambda index: Environment(episode, "P" if reward else "D")
    artifact, history = train_dqn(
        factory, pre, tmp_path, lambda model: reference(model, snapshot),
        arm=arm, steps=504, reward_model=reward, guidance=guide,
        feedback_version="fixture-v1" if reward or guide else None, training_id="fixture-v1")
    assert history[-1]["updates"] > 0 and history[-1]["steps"] == 504
    assert (tmp_path / "parity.json").exists()
    saved = torch.load(tmp_path / "resume.pt", weights_only=False)
    assert saved["exploration_steps"] == 252
    _, resumed = train_dqn(
        factory, pre, tmp_path, lambda model: reference(model, snapshot),
        arm=arm, steps=508, reward_model=reward, guidance=guide,
        feedback_version="fixture-v1" if reward or guide else None,
        resume=tmp_path / "resume.pt", training_id="fixture-v1")
    assert resumed[-1]["steps"] == 508 and resumed[-1]["updates"] > history[-1]["updates"]


@pytest.mark.parametrize("warm", [False, True])
def test_ppo_update_resume_and_export(warm, episode, pre, tmp_path):
    snapshot = Environment(episode).snapshot()
    initial = Scorer("deepsets").state_dict() if warm else None
    artifact, history = train_ppo(
        lambda index: Environment(episode), pre, tmp_path,
        lambda model: reference(model, snapshot), steps=256,
        warmstart=initial, training_id="fixture-v1")
    _, resumed = train_ppo(
        lambda index: Environment(episode), pre, tmp_path,
        lambda model: reference(model, snapshot), steps=512, warmstart=initial,
        resume=tmp_path / "resume.pt", training_id="fixture-v1")
    assert resumed[-1]["steps"] == 512
    assert resumed[-1]["updates"] > history[-1]["updates"]
    assert (tmp_path / "parity.json").exists()


def test_duration_reward_excludes_selected_and_relabels(episode, pre):
    snapshot = Environment(episode).snapshot()
    env = Environment(episode, "P")
    nxt, _, terminal = env.step(0)
    path = transition_path(snapshot, 0, env, pre)
    assert sum(path.durations) == pytest.approx(1.)
    model = torch.nn.Linear(1172, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    reward = FrozenReward(model, pre, "fixture")
    assert reward.value(path) == pytest.approx(-.5)
    rows = [(snapshot, 0, 999., nxt, terminal, path)]
    assert relabel_replay(rows, reward)[0][2] == pytest.approx(-.5)
    assert rows[0][2] == 999.
    with pytest.raises(ValueError):
        relabel_replay([rows[0][:-1]], reward)


def test_reward_model_learns_strict_prefix_preference():
    model = torch.nn.Linear(1172, 1)
    with torch.no_grad():
        model.weight.zero_();model.bias.zero_()
    left = RatePath((tuple([1.] + [0.] * 1171),), (1.,))
    right = RatePath((tuple([-1.] + [0.] * 1171),), (1.,))
    trained, history = fit_reward(model, [([left], [right], 1.)], epochs=30)
    assert history[-1] < history[0]
    assert float(integral(trained, left, "cpu").detach()) > float(integral(trained, right, "cpu").detach())


@pytest.mark.parametrize("repeat", ["local", "cross"])
def test_churn_preserves_declared_repeat_source_contract(repeat):
    records=[];features={};vectors={}
    base=fixture()
    for i in range(64):
        ident="image-"+str(i)
        records.append({"image_id":ident,"image_sha256":"hash-"+str(i),"reference_tier":i%4})
        features[ident]=base.messages[i%8].features
        vector=np.zeros(512);vector[i%512]=1.;vectors["hash-"+str(i)]=vector
    episode=make_episode(records,features,vectors,None,count=48,ues=12,seed=0,
                         repeat=repeat,churn=True,burst=True)
    sources={}
    for message,fact in zip(episode.messages,episode.fact_ids):
        sources.setdefault(fact,[]).append(message.source)
        active=episode.metadata["ue_active_intervals"][message.source]
        assert any(start<=message.arrival<=end and message.session==session
                   for start,end,session in active)
    repeated=[value for value in sources.values() if len(value)>1]
    assert repeated
    if repeat=="local":assert all(len(set(value))==1 for value in repeated)
    else:assert all(len(set(value))>=2 for value in repeated)


def test_warmstarted_policy_keeps_original_actor_after_source_improves(tmp_path):
    from experiments.stream_scheduler.train_real_suite import frozen_warmstart
    source=tmp_path/"source.pt";frozen=tmp_path/"ppo"/"warmstart.pt"
    first=Scorer("deepsets").state_dict()
    torch.save({"best_state":first},source)
    expected=frozen_warmstart(source,frozen)
    improved={name:value+1 for name,value in first.items()}
    torch.save({"best_state":improved},source)
    actual=frozen_warmstart(source,frozen)
    for name in expected:
        torch.testing.assert_close(expected[name],actual[name])
    state=torch.load(frozen,weights_only=False);state["sha256"]="bad"
    torch.save(state,frozen)
    with pytest.raises(ValueError):
        frozen_warmstart(source,frozen)


@pytest.mark.parametrize("fault", ["shape", "scale", "weights"])
def test_invalid_artifact_fails_before_dispatch(fault, episode, pre):
    from Scheduler.semantic_learning_policy import digest
    import copy
    artifact=export(Scorer("mlp"),pre,"ranknet",{"task":"D"})
    artifact=copy.deepcopy(artifact)
    if fault=="shape":artifact["weights"]["head.0.weight"]=[[1.]]
    if fault=="scale":artifact["preprocessing"]["scale"][0]=0.
    if fault=="weights":artifact["weights"]["unexpected"]=[1.]
    artifact.pop("sha256");artifact["sha256"]=digest(artifact)
    with pytest.raises(ValueError):
        NumpyRanker(artifact)


def test_no_backlog_policies_have_identical_dispatches(episode):
    messages=tuple(replace(message,arrival=2.*i) for i,message in enumerate(episode.messages))
    episode=replace(episode,messages=messages,horizon=20.)
    orders=[]
    for kind in ("fifo","round_robin","random","spi_fifo","novelty"):
        env=Environment(episode);rule=Rule(kind)
        while not env.done:
            env.step(rule.select(env.snapshot()))
        orders.append([(event["token"],event["dispatch"]) for event in env.events])
    assert all(order==orders[0] for order in orders)
