"""Evaluator-only weighted fact coverage. Never imported by runtime policies."""
from functools import lru_cache
import math


def _validate(job_facts, weights):
    offered = set().union(*(set(facts) for facts in job_facts.values())) if job_facts else set()
    if not offered <= set(weights):
        raise ValueError('missing offered fact weight')
    if any(not math.isfinite(weights[fact]) or weights[fact] < 0 for fact in offered):
        raise ValueError('fact weights must be finite and nonnegative')
    return offered, math.fsum(weights[fact] for fact in offered)


def coverage_metrics(order, job_facts, weights, k=10, succeeded=None, recovered=None, critical=()):
    """Source-offer denominator is fixed even after decoder failure/loss.

    `recovered` is evaluator-only. Unsupported recovered facts receive no credit.
    Missing recovered outcomes are rejected, never replaced by source truth.
    """
    offered, total = _validate(job_facts, weights)
    order = tuple(order)
    if (not 1 <= k <= len(order) or len(set(order)) != len(order)
            or not set(order) <= set(job_facts)):
        raise ValueError('unique known jobs and a valid cutoff required')
    if succeeded is not None and any(job not in succeeded for job in order[:k]):
        raise ValueError('missing execution status')
    if recovered is not None and any(job not in recovered for job in order[:k]):
        raise ValueError('missing recovered-fact assessment')
    critical = set(critical)
    if not critical <= offered:
        raise ValueError('critical facts must be offered source facts')
    covered, curve, increments = set(), [], []
    repeated_weight, raw_weight, unsupported, failures, avoidable = 0., 0., 0, 0, 0
    pending = set(job_facts)
    for job in order[:k]:
        before = math.fsum(weights[f] for f in covered)
        source = set(job_facts[job])
        successful = succeeded is None or succeeded[job]
        claims = source if recovered is None else set(recovered[job])
        if successful:
            unsupported += len(claims - source)
            useful = claims & source
            repeated_weight += math.fsum(weights[f] for f in useful & covered)
            raw_weight += math.fsum(weights[f] for f in useful)
        else:
            failures += 1
            useful = set()
        new = useful - covered
        if not new and any(set(job_facts[other]) - covered for other in pending if other != job):
            avoidable += 1
        covered.update(useful)
        pending.remove(job)
        after = math.fsum(weights[f] for f in covered)
        curve.append(None if total == 0 else after/total)
        increments.append(None if total == 0 else (after-before)/total)
    return dict(A_K=None if total == 0 else math.fsum(curve)/k,
                coverage_at_k=curve[-1], curve=curve, marginal_gains=increments,
                critical_recall=None if not critical else len(covered & critical)/len(critical),
                weighted_redundancy=None if raw_weight == 0 else repeated_weight/raw_weight,
                avoidable_zero_gain=avoidable, failures=failures, unsupported_claims=unsupported,
                offered_fact_count=len(offered), offered_weight=total,
                covered_facts=sorted(covered))


def marginal_gain(job, delivered, job_facts, weights):
    _, total = _validate(job_facts, weights)
    if total == 0:
        return None
    seen = set().union(*(set(job_facts[j]) for j in delivered)) if delivered else set()
    return math.fsum(weights[f] for f in set(job_facts[job])-seen)/total


def exact_sequence(job_facts, weights, k):
    """Evaluator-only subset DP, bounded to twelve jobs."""
    ids = tuple(job_facts)
    _, total = _validate(job_facts, weights)
    if not 1 <= k <= len(ids) <= 12:
        raise ValueError('exact reference requires 1 <= K <= N <= 12')
    if total == 0:
        return dict(order=(), A_K=None)

    @lru_cache(None)
    def coverage(bits):
        seen = set().union(*(set(job_facts[ids[i]]) for i in range(len(ids)) if bits & (1 << i)))
        return math.fsum(weights[f] for f in seen)/total

    @lru_cache(None)
    def solve(bits):
        if bits.bit_count() == k:
            return 0., ()
        best = None
        for i, job in enumerate(ids):
            if bits & (1 << i):
                continue
            future, sequence = solve(bits | (1 << i))
            candidate = (coverage(bits | (1 << i))/k + future, (job,)+sequence)
            if best is None or candidate[0] > best[0] + 1e-12:
                best = candidate
        return best

    score, order = solve(0)
    return dict(order=order, A_K=score)


def oracle_greedy(job_facts, weights):
    _validate(job_facts, weights)
    remaining, seen, order = list(job_facts), set(), []
    while remaining:
        job = max(remaining, key=lambda j: math.fsum(weights[f] for f in set(job_facts[j])-seen))
        order.append(job)
        seen.update(job_facts[job])
        remaining.remove(job)
    return order
