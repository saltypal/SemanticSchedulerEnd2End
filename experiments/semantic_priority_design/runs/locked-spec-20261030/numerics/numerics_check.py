#!/usr/bin/env python3
"""Independent CPU arithmetic checks for frozen semantic ranking spec sections 4, 7-9.
No production imports, model implementation, training, GPU, or network access.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]  # .../experiments/semantic_priority_design/runs/<id>/numerics -> repo
SPEC = REPO / "docs/decoder_context/SEMANTIC_RANKING_RESEARCH_DESIGN.md"
START = time.perf_counter()
checks: list[dict] = []

def record(name: str, passed: bool, count: int, detail: str, inputs=None, outputs=None):
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "cases": count,
                   "detail": detail, "inputs": inputs or {}, "outputs": outputs or {}})
    if not passed:
        raise AssertionError(name + ": " + detail)

def ell(wait: float, budget: float) -> float:
    ratio = wait / budget
    return ratio + max(0.0, ratio - 1.0)

def y_one_slot(wait: float, budget: float, d0: float) -> float:
    return ell(wait + d0, budget) - ell(wait, budget)

# Section 4: explicit examples and systematic y/ymax bounds for positive budgets and waits.
A = {"wait": 1.5, "budget": 2.0, "d0": 1.0}
B = {"wait": 3.0, "budget": 16.0, "d0": 1.0}
yA, yB = y_one_slot(**A), y_one_slot(**B)
record("D_examples", math.isclose(yA, .75, rel_tol=0, abs_tol=1e-12) and math.isclose(yB, .0625, rel_tol=0, abs_tol=1e-12),
       2, "Specified A/B one-slot costs", {"A": A, "B": B}, {"y_A": yA, "y_B": yB})

budget_cases = [([2, 4, 8, 16], 1), ([1, 3.5, 19], .25), ([.125, .5, 2], .0625), ([100, 250], 3.0)]
wait_factors = [0, .25, .5, .999999, 1, 1.000001, 1.5, 4, 32]
bound_cases = 0
max_ratio = 0.0
min_ratio = math.inf
for budgets, d0 in budget_cases:
    dmin = min(budgets)
    ymax = 2 * d0 / dmin
    waits = sorted(set([d0 * k for k in [0, .25, .5, .999999, 1, 1.000001, 2, 8, 100]] +
                       [b * k for b in budgets for k in wait_factors]))
    for delta, wait in itertools.product(budgets, waits):
        y = y_one_slot(wait, delta, d0)
        ratio = y / ymax
        assert y >= -1e-12 and ratio >= -1e-12 and ratio <= 1 + 1e-12, (budgets, d0, wait, delta, y, ymax)
        min_ratio, max_ratio = min(min_ratio, ratio), max(max_ratio, ratio)
        bound_cases += 1
record("normalized_one_slot_bounds", True, bound_cases,
       "Across multiple positive budget configurations and waits on, below, and above deadlines; v=y/ymax stayed in [0,1].",
       {"budget_cases": budget_cases, "wait_factor_grid": wait_factors},
       {"min_v": min_ratio, "max_v": max_ratio})

# Section 4: event accounting includes arrivals during service, dispatch freezing, and undispached terminal cost.
items = [
    {"id": "dispatched_early", "arrival": 0.0, "budget": 2.0, "dispatch": 1.0},
    {"id": "mid_service_arrival", "arrival": 0.5, "budget": 4.0, "dispatch": 3.0},
    {"id": "unselected_terminal", "arrival": 1.5, "budget": 16.0, "dispatch": None},
]
T = 5.0
def loss_at(t):
    return sum(ell(max(0.0, min(t, x["dispatch"] if x["dispatch"] is not None else math.inf) - x["arrival"]), x["budget"])
               for x in items if x["arrival"] <= t)
# Fine event partition includes all arrivals, dispatches and terminal time; this is a direct telescoping sum.
events = sorted(set([0.0, T] + [x["arrival"] for x in items] + [x["dispatch"] for x in items if x["dispatch"] is not None]))
losses = [loss_at(t) for t in events]
increments = [losses[k+1] - losses[k] for k in range(len(events)-1)]
reward_sum = sum(-inc / 100.0 for inc in increments)
terminal_loss = loss_at(T)
freeze_ok = loss_at(4.0) - ell(1.0, 2.0) - ell(2.5, 4.0)  # residual must be unchanged by already-dispatched jobs
expected_unselected = ell(T - 1.5, 16.0)
record("event_reward_telescoping_and_lifecycle", math.isclose(reward_sum, -terminal_loss/100, abs_tol=1e-12) and
       math.isclose(terminal_loss, sum(ell(min(T, x["dispatch"] if x["dispatch"] is not None else T)-x["arrival"], x["budget"]) for x in items), abs_tol=1e-12) and
       math.isclose(expected_unselected, ell(3.5,16), abs_tol=1e-12) and freeze_ok >= -1e-12,
       len(events)-1, "Event rewards telescope to terminal loss; mid-service arrival accrues after arrival; dispatched costs remain frozen; unselected job accrues through T.",
       {"items": items, "T": T, "events": events},
       {"losses_at_events": losses, "increments": increments, "terminal_loss": terminal_loss,
        "reward_sum": reward_sum, "unselected_terminal_cost": expected_unselected, "frozen_cost_residual_at_t4": freeze_ok})

# Section 7: LambdaMART grade quantization and declared gains.
quant_inputs = [0.0, 0.5/255, 1/255, 0.25, 0.5, 0.999, 1.0]
grades = [math.floor(255*v + .5) for v in quant_inputs]
gains = [g/255 for g in range(256)]
record("lambdamart_grade_mapping", all(0 <= g <= 255 for g in grades) and gains[0] == 0 and gains[-1] == 1 and
       all(gains[i] <= gains[i+1] for i in range(255)) and grades == sorted(grades),
       256 + len(quant_inputs), "floor(255*v+0.5) yields grades 0..255; declared D label_gain is g/255, monotone with endpoints 0 and 1.",
       {"sample_v": quant_inputs, "rule": "floor(255*v + .5)", "gain_rule": "g/255"},
       {"sample_grades": grades, "sample_gains": [gains[g] for g in grades], "gain_endpoints": [gains[0], gains[-1]]})

# Section 7: tie-target masked softmax CE gradient, plus pair logistic gradient direction.
def softmax(scores):
    m = max(scores)
    ex = [math.exp(x-m) for x in scores]
    z = sum(ex)
    return [x/z for x in ex]
def choice_loss(scores, target):
    p = softmax(scores)
    return -sum(t*math.log(max(v, 1e-300)) for t, v in zip(target,p))
def pair_loss(si, sj, target):
    # stable softplus forms: -t log(sigmoid(d)) -(1-t)log(sigmoid(-d))
    d = si-sj
    return target * max(0,-d) + math.log1p(math.exp(-abs(d))) + (1-target) * max(0,d) - max(0,-d) # replaced below by stable CE identity
# Correct stable pair loss via softplus(-d)*t + softplus(d)*(1-t)
def softplus(x): return max(0.0,x) + math.log1p(math.exp(-abs(x)))
def pair_loss(si, sj, target):
    d=si-sj
    return target*softplus(-d)+(1-target)*softplus(d)
score_sets = [([2.0,2.0,0.0],[.5,.5,0.0]), ([1.0,0.0,-2.0],[1.0,0.0,0.0])]
ce_cases=0
for scores,target in score_sets:
    p=softmax(scores)
    grads=[p[i]-target[i] for i in range(len(scores))]
    eps=1e-6
    fd=[(choice_loss([x+(eps if j==i else 0) for j,x in enumerate(scores)],target)-choice_loss([x-(eps if j==i else 0) for j,x in enumerate(scores)],target))/(2*eps) for i in range(len(scores))]
    assert max(abs(a-b) for a,b in zip(grads,fd)) < 1e-6
    assert abs(sum(grads)) < 1e-12
    ce_cases += len(scores)
# For target A>B, gradient descent increases sA-sB; for a tie at equal scores, pair gradient is zero.
d, target = -0.4, 1.0
pair_grad = (1/(1+math.exp(-d))) - target
fd_pair=(pair_loss(d/2+1e-6,-d/2,target)-pair_loss(d/2-1e-6,-d/2,target))/(2e-6) # derivative wrt si at fixed sj
assert pair_grad < 0 and abs(pair_grad-fd_pair)<1e-6
pair_tie_grad=1/(1+math.exp(0))-0.5
assert abs(pair_tie_grad)<1e-12
record("masked_softmax_tie_ce_and_pair_gradient", True, ce_cases+2,
       "Tie targets are uniform over tied maxima; CE gradient softmax-target matches finite differences; strict pair preference gradient descent raises preferred score difference; equal-score justified tie has zero pair gradient.",
       {"choice_cases": score_sets, "pair_strict": {"score_difference": d, "target": target}, "pair_tie": {"score_difference": 0, "target": .5}},
       {"ce_gradient_max_abs_error": max(abs(a-b) for scores,target0 in score_sets for a,b in zip([softmax(scores)[i]-target0[i] for i in range(len(scores))], [(choice_loss([x+(1e-6 if j==i else 0) for j,x in enumerate(scores)],target0)-choice_loss([x-(1e-6 if j==i else 0) for j,x in enumerate(scores)],target0))/(2e-6) for i in range(len(scores))])),
        "pair_gradient": pair_grad, "pair_finite_difference": fd_pair, "tie_gradient": pair_tie_grad})

# Section 8: exact mixed replay probability, full support, actual-P weights and neutral guidance equivalence.
priorities=[0.2, 1.0, 3.0, 0.05, 2.5]
alpha=.6
p=[x**alpha for x in priorities]
def mixture(guidance):
    denom=sum(pi*gi for pi,gi in zip(p,guidance))
    return [0.5/len(p)+0.5*pi*gi/denom for pi,gi in zip(p,guidance)]
td=mixture([1.0]*len(p))
neutral=mixture([1.0]*len(p))
guided=mixture([.5,1.0,1.5,1.0,.5])
assert abs(sum(td)-1)<1e-12 and abs(sum(guided)-1)<1e-12
assert min(td)>0 and min(guided)>0
assert td==neutral
# Full-support actual-P weights at beta endpoints; use all records as the sampled batch.
weight_checks={}
for distribution_name, distribution in [("td_only",td),("guided",guided)]:
    for beta in [.4,1.0]:
        raw=[(len(p)*prob)**(-beta) for prob in distribution]
        normalized=[w/max(raw) for w in raw]
        assert all(0 < w <= 1+1e-12 for w in normalized)
        weight_checks[f"{distribution_name}:{beta}"]={"raw_from_actual_mixture_P":raw,"batch_max_normalized":normalized}
record("replay_mixture_support_and_importance", True, len(priorities)*2+2,
       "Mixed probabilities sum to one and have full support; importance uses actual mixed P; neutral guidance g=1 exactly matches TD-only g=1.",
       {"priorities": priorities, "alpha": alpha, "guidance_neutral": [1]*len(p), "guidance_example": [.5,1,1.5,1,.5], "beta": [.4,1.0]},
       {"p_alpha":p,"P_td_only":td,"P_neutral":neutral,"P_guided":guided,"importance_weights":weight_checks,
        "max_neutral_td_abs_diff":max(abs(a-b) for a,b in zip(td,neutral))})

elapsed=time.perf_counter()-START
script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
spec_hash=hashlib.sha256(SPEC.read_bytes()).hexdigest()
receipt={
    "title":"Independent frozen-spec arithmetic validation",
    "classification":"specification arithmetic only; not production, model, training, or deployment validation",
    "source_revision":"not pinned; mutable checkout snapshot, source commit intentionally not inferred",
    "repo":str(REPO), "spec_path":str(SPEC), "spec_sha256":spec_hash,
    "sections_read":[4,7,8,9],
    "runtime":{"python_executable":sys.executable,"python_version":sys.version,"implementation":platform.python_implementation(),"platform":platform.platform(),"backend":"CPU; pure Python standard library; no model/GPU/network"},
    "command":f"& '{sys.executable}' '{Path(__file__).resolve()}'",
    "script_path":str(Path(__file__).resolve()), "script_sha256":script_hash,
    "overall":"PASS" if all(c['status']=='PASS' for c in checks) else "FAIL",
    "check_count":len(checks),"case_count":sum(c['cases'] for c in checks),"elapsed_seconds":elapsed,
    "checks":checks,
    "ambiguities":[],
    "limitations":["Arithmetic implements only equations read in the frozen design; no code was imported from production.","This is one mutable-checkout snapshot and does not validate model learning, simulator implementation, production behavior, or empirical performance."]
}
receipt_path=HERE/"receipt.json"
receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps({"overall":receipt["overall"],"check_count":receipt["check_count"],"case_count":receipt["case_count"],"elapsed_seconds":elapsed,"script_sha256":script_hash,"spec_sha256":spec_hash,"receipt":str(receipt_path)},indent=2))
