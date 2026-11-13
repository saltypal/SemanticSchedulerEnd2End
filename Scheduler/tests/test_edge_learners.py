"""Learner modules: shapes, policy validity, schedules, dual update, BC, PPO, DAgger, save/load, rewards."""
import numpy as np
import torch

from Evaluation.edge.env import EdgeEnv
from Evaluation.edge.receiver import Receiver
from Scheduler.edge.sv_dpp import SVDPP
from Scheduler.tests.test_edge_integration import LATENCY, SPEC, make_jobs
from Scheduler.training.edge import policy_net as pn
from Scheduler.training.edge import rollout, scope, trainers
from Scheduler.training.edge.state_features import GLOBAL_DIM, JOB_DIM


def factory(seed):
    return EdgeEnv(make_jobs(24, seed, 2.0), LATENCY, SPEC, workers=1, max_batch=4, receiver=Receiver(), seed=seed)


def test_forward_shapes_variable_queue():
    m = pn.SetPolicy()
    for n in range(1, 31):
        lg, v = m(torch.randn(1, n, JOB_DIM), torch.randn(1, GLOBAL_DIM))
        assert lg.shape == (1, n) and v.shape == (1,)


def test_learned_policy_valid_for_env():
    for mode in ("greedy", "sample"):
        pol = pn.LearnedPolicy(pn.SetPolicy(), mode)
        res = factory(1).run(pol)
        assert pol.decisions and all(d["chosen"] for d in pol.decisions)
        assert all(len(d["ids"]) in LATENCY.batch_sizes for d in pol.decisions)
        assert not any(r["status"] == "purged" for r in res.outcomes)


def test_imitation_weight_decays():
    w = [scope.imitation_weight(n, 1.0, 100) for n in (0, 100, 1000)]
    assert w[0] == 1.0 and w[0] > w[1] > w[2] > 0


def test_dual_update():
    new = scope.dual_update([0.5, 0.0], [0.5, 0.99], [0.9, 0.9], 0.1)
    assert new[0] > 0.5 and new[1] == 0.0
    assert scope.dual_update([0.5], [1.0], [0.9], 0.1)[0] < 0.5


def test_bc_reduces_cross_entropy():
    _, hist = trainers.train_bc(factory, SVDPP(), episodes=1, epochs=15)
    assert hist[0]["bc_loss_last"] < hist[0]["bc_loss_first"]


def test_ppo_and_scope_run_with_valid_probs():
    _, hist = trainers.train_ppo(factory, iters=2, episodes=1)
    assert len(hist) == 2
    model, hist = trainers.train_scope_ppo(factory, SVDPP(), iters=2, episodes=1)
    assert len(hist[0]["lambdas"]) == 4
    # F4: the teacher weight adapts (TGRL-style) within [0.05, 1] x beta0 instead of a fixed decay
    assert all(0.0 < h["beta"] <= 1.0 for h in hist) and hist[1]["beta"] != hist[0]["beta"]
    lg, _ = model(torch.randn(1, 5, JOB_DIM), torch.randn(1, GLOBAL_DIM))
    p = torch.softmax(lg, -1)
    assert torch.isfinite(p).all() and abs(float(p.sum()) - 1) < 1e-5


def test_dagger_aggregates():
    _, hist = trainers.train_dagger(factory, SVDPP(), iters=3, episodes=1, epochs=5)
    n = [h["n_samples"] for h in hist]
    assert n[0] < n[1] < n[2]


def test_save_load_roundtrip(tmp_path):
    m = pn.SetPolicy()
    pn.save_model(m, tmp_path / "m.pt")
    m2 = pn.load_model(tmp_path / "m.pt")
    x, g = torch.randn(1, 6, JOB_DIM), torch.randn(1, GLOBAL_DIM)
    assert torch.equal(m(x, g)[0], m2(x, g)[0])


def _row(i, tier, status, delay=None, value=0.0):
    return dict(id=i, tier=tier, status=status, delay=delay, value=value, measured=True)


def test_reward_attribution():
    h = SPEC.hard_deadline(0)
    decisions = [dict(ids=[1, 2]), dict(ids=[3]), dict(ids=[4])]
    rows = [_row(1, 0, "decoded", 1.0, 2.0), _row(2, 0, "decoded", h + 1, 0.0),
            _row(3, 1, "decoded", 1.0, 1.5), _row(4, 2, "expired"), _row(5, 3, "expired")]
    r = rollout.decision_rewards(decisions, rows, SPEC, [10.0, 0, 0, 0])
    assert np.allclose(r, [2.0 - 10.0, 1.5, 0.0])
    j = rollout.tier_success_rates(rows, SPEC)
    assert j[0] == 0.5 and j[1] == 1.0 and j[2] == 0.0 and j[3] == 0.0
