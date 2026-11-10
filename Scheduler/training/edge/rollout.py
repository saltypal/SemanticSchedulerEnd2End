"""Run one episode with a ``LearnedPolicy`` and compute per-decision rewards from the RunResult rows.

Reward of a decision = sum of ``value`` of its decoded jobs - sum_k lambda_k * (true-tier-k jobs of the
decision that finished after H_k). Jobs that expired unserved belong to no decision; their misses count
only in the episode-level per-tier success rates J_k (used by the dual update). J_k = fraction of
tier-k jobs (measured rows, all rows if none measured) decoded within the hard deadline H_k.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    decisions: list
    rewards: np.ndarray
    tier_success: np.ndarray
    result: object


def decision_rewards(decisions, rows, spec, lambdas):
    by_id = {r["id"]: r for r in rows}
    out = []
    for d in decisions:
        total = 0.0
        for jid in d["ids"]:
            r = by_id.get(jid)
            if r is None or r["status"] != "decoded":
                continue
            total += r["value"]
            if r["delay"] > spec.hard_deadline(r["tier"]):
                total -= lambdas[r["tier"]]
        out.append(total)
    return np.asarray(out, dtype=float)


def tier_success_rates(rows, spec):
    """Per-tier timely-success J_k; a tier without jobs reports its target eta_k (no dual pressure)."""
    j = np.array(spec.eta, dtype=float)
    for k in range(len(j)):
        rk = [r for r in rows if r["tier"] == k]
        if rk:
            j[k] = np.mean([r["status"] == "decoded" and r["delay"] <= spec.hard_deadline(k) for r in rk])
    return j


def collect_episode(env_factory, learned_policy, teacher=None, seed=0, lambdas=None):
    env = env_factory(seed)
    if teacher is not None:
        learned_policy.teacher = teacher
    result = env.run(learned_policy)
    spec = env.spec
    lam = np.zeros(len(spec.weights)) if lambdas is None else np.asarray(lambdas, dtype=float)
    decisions = learned_policy.decisions
    rewards = decision_rewards(decisions, result.outcomes, spec, lam)
    rows = result.measured() or result.outcomes
    return Episode(decisions, rewards, tier_success_rates(rows, spec), result)
