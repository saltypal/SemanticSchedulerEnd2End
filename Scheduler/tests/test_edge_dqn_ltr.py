"""DQN, reward attribution and learning-to-rank on synthetic data."""
import numpy as np
import pytest
import torch
from scipy.stats import kendalltau

from Evaluation.edge.contracts import Header, Job, LatencyProfile, Truth, ValueSpec
from Evaluation.edge.edge_config import load_config
from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.receiver import Receiver
from Scheduler.edge import registry
from Scheduler.training.edge import dqn, ltr
from Scheduler.training.edge.reward_attribution import LoggingPolicy, attribute_rewards
from Scheduler.training.edge.state_features import GLOBAL_DIM, JOB_DIM

SPEC = ValueSpec.from_config(load_config("sim"))
LATENCY = LatencyProfile({1: [0.8, 1.0, 1.2], 2: [1.2, 1.4, 1.6], 4: [2.0, 2.2, 2.4]})
ONE_HOT = [(0.9, 0.05, 0.03, 0.02), (0.05, 0.85, 0.05, 0.05), (0.03, 0.07, 0.8, 0.1), (0.02, 0.03, 0.05, 0.9)]


def make_jobs(n=60, seed=0, rate=1.5):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(1.0 / rate, size=n))
    jobs = []
    for i, a in enumerate(t):
        tier = int(rng.choice(4, p=[0.1, 0.2, 0.3, 0.4]))
        h = Header(capture_time=float(a), arrival_time=float(a), payload_bytes=500, mission=0, codec_id=1,
                   m_index=2, tier_probs=ONE_HOT[tier], relevance=0.5, sem_key=int(rng.integers(0, 2**62)),
                   perc_key=int(rng.integers(0, 2**62)), sts=(0.1,) * 13)
        jobs.append(Job(i, h, Truth(tier=tier, fidelity=int(rng.random() < 0.8), dup_group=None,
                                    event="e", source="crisismmd")))
    return jobs


def make_env(ep=0):
    return EdgeEnv(make_jobs(60, ep), LATENCY, SPEC, workers=1, max_batch=4, receiver=Receiver(), seed=ep)


@pytest.mark.parametrize("n", [1, 2, 7, 30])
def test_qnet_shapes(n):
    q = dqn.make_qnet(16)(torch.randn(n, JOB_DIM), torch.randn(GLOBAL_DIM))
    assert q.shape == (n,)


def test_replay_buffer_capacity_and_sample():
    buf = dqn.ReplayBuffer(3, seed=1)
    for i in range(5):
        buf.push(i)
    assert len(buf) == 3 and sorted(buf.items) == [2, 3, 4]
    assert len(buf.sample(10)) == 3


def test_target_network_sync():
    a, b = dqn.make_qnet(8), dqn.make_qnet(8)
    assert not torch.equal(a.net[0].weight, b.net[0].weight)
    dqn.sync_target(a, b)
    assert torch.equal(a.net[0].weight, b.net[0].weight)


def test_anneal_beta():
    assert dqn.anneal_beta(0, 1.0, 10) == 1.0
    assert dqn.anneal_beta(10, 1.0, 10) == 0.5
    assert dqn.anneal_beta(99, 2.0, None) == 2.0


def test_margin_loss_makes_teacher_action_preferred():
    torch.manual_seed(0)
    net = dqn.make_qnet(16)
    jx, g = torch.randn(6, JOB_DIM), torch.randn(GLOBAL_DIM)
    teacher = (3,)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(100):
        loss = dqn.large_margin_loss(net(jx, g), teacher, 0.8)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert int(net(jx, g).argmax()) == 3


def test_double_dqn_loss_with_variable_queues():
    online, target = dqn.make_qnet(8), dqn.make_qnet(8)
    mk = lambda n, done: dqn.Transition(np.random.rand(n, JOB_DIM), np.zeros(GLOBAL_DIM), (0,), (1,), 1.0,
                                        np.random.rand(0 if done else 3, JOB_DIM), np.zeros(GLOBAL_DIM), done, (0,))
    loss = dqn.dqn_loss(online, target, [mk(1, True), mk(5, False), mk(30, False)], 0.9, margin_weight=0.5)
    loss.backward()
    assert torch.isfinite(loss)


def test_dqn_policy_valid_in_env_and_records():
    model = dqn.make_qnet(8)
    pol = dqn.DQNPolicy(model, epsilon=0.3, teacher=registry.build("fifo"), record=True, seed=2)
    res = make_env().run(pol)
    assert pol.records and any(r["status"] == "decoded" for r in res.outcomes)
    assert all(len(r.batch) == len(r.idx) for r in pol.records)
    rew = attribute_rewards([r.batch for r in pol.records], res.outcomes, SPEC)
    assert len(rew) == len(pol.records) and rew.sum() >= 0


def test_attribute_rewards_penalty_and_value_fn():
    rows = [{"id": 1, "status": "decoded", "value": 2.0, "tier": 0, "delay": SPEC.hard_deadline(0) + 1},
            {"id": 2, "status": "decoded", "value": 1.0, "tier": 1, "delay": 0.1},
            {"id": 3, "status": "purged", "value": 0.0, "tier": 0, "delay": None}]
    lam = (5.0, 1.0, 0.0, 0.0)
    assert attribute_rewards([(1, 2), (3,)], rows, SPEC, lam).tolist() == [-2.0, 0.0]
    assert attribute_rewards([(2,)], rows, SPEC, value_fn=lambda r: 10.0).tolist() == [10.0]


def test_train_dqn_runs_and_learns_history():
    model, hist = dqn.train_dqn(make_env, 3, teacher=registry.build("fifo"), hidden=8, margin_weight=0.5,
                                updates_per_episode=3, batch_size=8, seed=0)
    assert len(hist) == 3 and hist[-1]["loss"] is not None
    pol = dqn.DQNPolicy(model)
    assert make_env(9).run(pol).outcomes


def _episodes():
    pol = LoggingPolicy(registry.build("fifo"))
    res = make_env().run(pol)
    return [(pol.records, res.outcomes)]


def test_ranking_dataset_shape():
    data = ltr.build_ranking_dataset(_episodes())
    assert data["X"].shape == (data["group"].sum(), JOB_DIM + GLOBAL_DIM)
    assert data["y"].shape == (data["X"].shape[0],) and data["y"].max() <= 4 and data["y"].min() >= 0


def _toy(n_groups=20, g=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_groups * g, 5))
    y = np.clip(np.rint(2 + X[:, 0] * 1.5), 0, 4).astype(int)
    return {"X": X, "y": y, "group": np.full(n_groups, g)}, X[:, 0]


def test_ranknet_learns_separable_ordering():
    data, truth = _toy()
    model = ltr.train_ranknet(data, epochs=300)
    tau = kendalltau(model.score(data["X"]), truth)[0]
    assert tau > 0.8


def test_tree_ranker_and_save_load(tmp_path):
    data, truth = _toy()
    tr = ltr.train_lambdamart(data, n_estimators=30, prefer_lightgbm=False)
    assert tr.backend == "sklearn_pointwise"
    assert kendalltau(tr.score(data["X"]), truth)[0] > 0.6
    p = tmp_path / "t.pkl"
    tr.save(p)
    np.testing.assert_allclose(ltr.TreeRanker.load(p).score(data["X"]), tr.score(data["X"]))
    rn = ltr.train_ranknet(data, epochs=20)
    rn.save(tmp_path / "r.pt")
    np.testing.assert_allclose(ltr.RankNet.load(tmp_path / "r.pt").score(data["X"]), rn.score(data["X"]), atol=1e-6)


def test_ltr_policy_valid_in_env():
    data = ltr.build_ranking_dataset(_episodes())
    rn = ltr.train_ranknet(data, epochs=5)
    assert make_env(4).run(ltr.LTRPolicy(rn)).outcomes


def test_listwise_distill_loss_decreases_when_scores_match_teacher():
    torch.manual_seed(0)
    teacher = torch.randn(8)
    probs = torch.softmax(torch.randn(8, 4), dim=1)
    lam = [3.0, 1.0, 0.0, 0.0]
    scores = torch.randn(8, requires_grad=True)
    start = float(ltr.listwise_distill_loss(scores, teacher, lam, probs))
    opt = torch.optim.Adam([scores], lr=0.1)
    for _ in range(100):
        loss = ltr.listwise_distill_loss(scores, teacher, lam, probs)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < start
    matched = float(ltr.listwise_distill_loss(teacher, teacher, [0, 0, 0, 0], probs))
    assert matched < float(ltr.listwise_distill_loss(-teacher, teacher, [0, 0, 0, 0], probs))
