"""Standard-library mathematical reference; no training or runtime integration.

Run from any directory:
    python reference_metric_and_loss.py --report reference_checks.json

Facts are evaluator/training annotations, never runtime header features.
Sorting and pair weights are detached in the surrogate reference.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

TOL = 1e-12


def offered_weight(facts, weights, known=frozenset()):
    offered = set().union(*facts.values()) - set(known)
    for fact in offered:
        if fact not in weights:
            raise ValueError("Missing fact weight")
        if not math.isfinite(weights[fact]) or weights[fact] < 0:
            raise ValueError("Weights must be finite and nonnegative")
    return math.fsum(weights[f] for f in offered)


def validate(order, facts, k):
    if not 1 <= k <= len(order):
        raise ValueError("Require 1 <= K <= N")
    if len(order) != len(set(order)) or set(order) != set(facts):
        raise ValueError("Order must be a permutation of all offered messages")


def coverage_curve(order, facts, weights, k, known=frozenset()):
    validate(order, facts, k)
    total = offered_weight(facts, weights, known)
    if total == 0:
        return None
    seen = set(known)
    mass = 0.0
    curve = []
    for item in order[:k]:
        added = set(facts[item]) - seen
        mass += math.fsum(weights[f] for f in added)
        seen.update(facts[item])
        curve.append(mass / total)
    return curve


def early_coverage(order, facts, weights, k, known=frozenset()):
    curve = coverage_curve(order, facts, weights, k, known)
    return None if curve is None else math.fsum(curve) / k


def first_rank_value(rank, k):
    return (k - rank + 1) / k if rank <= k else 0.0


def swap_delta_full(order, facts, weights, k, a, b, known=frozenset()):
    before = early_coverage(order, facts, weights, k, known)
    if before is None:
        return None
    changed = list(order)
    changed[a], changed[b] = changed[b], changed[a]
    return early_coverage(changed, facts, weights, k, known) - before


def first_two_ranks(order, facts):
    ranks = {}
    for r, item in enumerate(order, 1):
        for fact in facts[item]:
            support = ranks.setdefault(fact, [])
            if len(support) < 2:
                support.append(r)
    return ranks


def swap_delta_fast(order, facts, weights, k, a, b, known=frozenset(),
                    cached=None):
    total = offered_weight(facts, weights, known)
    if total == 0:
        return None
    ranks = first_two_ranks(order, facts) if cached is None else cached
    left, right = order[a], order[b]
    effects = []
    for fact in (set(facts[left]) ^ set(facts[right])) - set(known):
        old_rank, new_rank = ((a + 1, b + 1) if fact in facts[left]
                              else (b + 1, a + 1))
        support = ranks[fact]
        first = support[0]
        other = ((support[1] if len(support) > 1 else math.inf)
                 if first == old_rank else first)
        after = min(other, new_rank)
        effects.append(weights[fact] *
                       (first_rank_value(after, k) -
                        first_rank_value(first, k)))
    return math.fsum(effects) / total


@dataclass(frozen=True)
class Pair:
    earlier: str
    later: str
    target: int
    weight: float


def detached_pairs(scores, arrival_order, facts, weights, k, known=frozenset()):
    # Queue order, not lexicographic message ID, resolves equal scores.
    tie = {item: r for r, item in enumerate(arrival_order)}
    order = sorted(arrival_order, key=lambda item: (-scores[item], tie[item]))
    validate(order, facts, k)
    pairs = []
    cached = first_two_ranks(order, facts)
    for a, b in itertools.combinations(range(len(order)), 2):
        delta = swap_delta_fast(order, facts, weights, k, a, b, known, cached)
        if delta is None or abs(delta) <= TOL:
            continue
        pairs.append(Pair(order[a], order[b], int(delta < 0), abs(delta)))
    return order, pairs


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def softplus(value):
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def frozen_pair_loss_and_gradient(scores, pairs):
    """Only scores vary; pair membership, labels and weights remain frozen."""
    z = math.fsum(p.weight for p in pairs)
    if z == 0:
        return None, {item: 0.0 for item in scores}
    loss = []
    grad = {item: 0.0 for item in scores}
    for p in pairs:
        delta = scores[p.earlier] - scores[p.later]
        weight = p.weight / z
        loss.append(weight * (softplus(delta) - p.target * delta))
        direction = weight * (sigmoid(delta) - p.target)
        grad[p.earlier] += direction
        grad[p.later] -= direction
    return math.fsum(loss), grad


def check_close(a, b, tolerance=1e-10):
    if not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError((a, b))


def run_checks():
    facts = {"A": {"X"}, "B": {"X"}, "C": {"Y"}, "D": {"Z"}}
    weights = {"X": 3, "Y": 3, "Z": 1}
    arrival = list(facts)
    check_close(early_coverage(arrival, facts, weights, 3), 12 / 21)
    check_close(early_coverage(["A", "C", "D", "B"], facts, weights, 3), 16 / 21)
    check_close(swap_delta_full(arrival, facts, weights, 3, 1, 2), 1 / 7)
    check_close(swap_delta_full(arrival, facts, weights, 3, 1, 3), 2 / 21)
    optimum = max(early_coverage(p, facts, weights, 3)
                  for p in itertools.permutations(arrival))
    check_close(optimum, 16 / 21)

    rng = random.Random(20261022)
    compared = 0
    gradient_cases = 0
    for n in range(2, 9):
        for case in range(40):
            names = [f"item-{i}" for i in range(n)]
            pool = [f"fact-{f}" for f in range(7)]
            generated = {i: {f for f in pool if rng.random() < 0.4}
                         for i in names}
            w = {f: rng.choice([0.0, 1.0, 3.0]) for f in pool}
            known = {f for f in pool if rng.random() < 0.15}
            rng.shuffle(names)
            k = rng.randint(1, n)
            scores = {i: rng.uniform(-3, 3) for i in names}
            order, pairs = detached_pairs(scores, names, generated, w, k, known)
            cache = first_two_ranks(order, generated)
            for a, b in itertools.combinations(range(n), 2):
                slow = swap_delta_full(order, generated, w, k, a, b, known)
                fast = swap_delta_fast(order, generated, w, k, a, b, known, cache)
                if slow is None:
                    assert fast is None
                else:
                    check_close(slow, fast)
                compared += 1
            loss, grad = frozen_pair_loss_and_gradient(scores, pairs)
            if loss is not None:
                check_close(sum(grad.values()), 0.0)
                # Freeze targets/weights during finite differences.
                for item in names:
                    hi, lo = dict(scores), dict(scores)
                    eps = 1e-6
                    hi[item] += eps
                    lo[item] -= eps
                    upper = frozen_pair_loss_and_gradient(hi, pairs)[0]
                    lower = frozen_pair_loss_and_gradient(lo, pairs)[0]
                    check_close((upper - lower) / (2 * eps), grad[item], 1e-8)
                    gradient_cases += 1
                # Pair target must favor the metric-improving relation.
                for p in pairs:
                    a, b = order.index(p.earlier), order.index(p.later)
                    delta = swap_delta_full(order, generated, w, k, a, b, known)
                    assert p.target == int(delta < 0)

    zero = {"A": set(), "B": set()}
    assert early_coverage(["A", "B"], zero, {}, 1) is None
    _, pairs = detached_pairs({"A": 1, "B": 0}, ["A", "B"], zero, {}, 1)
    assert pairs == []
    duplicate = {"A": {"X"}, "B": {"X"}}
    # Nonzero task utility, but every permutation has identical prefix utility.
    _, pairs = detached_pairs({"A": 1, "B": 0}, ["A", "B"],
                              duplicate, {"X": 3}, 2)
    assert pairs == []
    assert early_coverage(["A", "B"], duplicate, {"X": 3}, 2) == 1
    assert early_coverage(["A", "B"], duplicate, {"X": 3}, 2, {"X"}) is None
    for value in [-1000, -1, 0, 1, 1000]:
        assert math.isfinite(softplus(value))
        assert 0 <= sigmoid(value) <= 1
    try:
        early_coverage(["A", "A"], duplicate, {"X": 3}, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate message IDs accepted")
    return {
        "status": "passed",
        "evidence_scope": "synthetic_mathematical_reference_only",
        "training_performed": False,
        "checkpoint_inference_performed": False,
        "fixture": {"initial_A3": 12 / 21, "improved_A3": 16 / 21,
                    "exact_fixture_optimum_A3": optimum,
                    "swap_B_C": 1 / 7, "swap_B_D": 2 / 21},
        "random_cases": 280,
        "fast_vs_full_pair_deltas": compared,
        "finite_difference_score_coordinates": gradient_cases,
        "edge_cases": ["zero_offered_weight", "nonzero_utility_neutral_swaps",
                       "initial_known_facts", "duplicate_message_ids",
                       "extreme_score_stability", "gradient_conservation"]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run_checks()
    report["reference_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    protocol_path = Path(__file__).with_name("benchmark_protocol_v1.json")
    report["protocol_sha256"] = hashlib.sha256(
        protocol_path.read_bytes()).hexdigest()
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n",
                               encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
