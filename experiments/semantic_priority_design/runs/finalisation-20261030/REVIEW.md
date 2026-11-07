# Bounded design assessment and disposition

30 October 2026. Assessment-only review of the final canonical design and the
tiny objective script. No production tests, code changes or model experiments
were performed by the assessor.

Accepted mathematical scope: with equal service, fixed known weights, full drain
and no deliberate idling, exchanging an earlier lower-weight choice with an
already-ready higher-weight job preserves feasibility and cannot increase
weighted waiting. Future arrivals do not invalidate this exchange. This is not
an optimality claim for missing information, changing utility, unequal service or
arbitrary idling.

Accepted design boundaries: ordinal preference loss versus cardinal waiting-cost
conventions; one shot/one attempt; causal transmitter novelty; disabled receiver-
success-history learning in the proposed core; historical behavior preserved;
RL controls retained without presuming planning headroom.

## Findings and main-thread actions

| Finding | Impact | Disposition |
|---|---|---|
| Unknown reference priority had no explicit cost-training rule | Could silently turn unassessed traffic into zero-cost work | Accepted and corrected in section 4: unknowns remain queued; unsupported comparisons are skipped; primary reference-cost RL requires all offered weights known; mixed-reference cost is explicitly partial |
| Numerical script does not calculate nDCG | Could overstate metric validation | Accepted and clarified in section 12: same-class wait is numerically checked, equal-grade nDCG invariance is analytical |

No blocking A-B design contradiction was identified. This is a bounded assessment,
not empirical confirmation of semantic correctness or a final whole-project audit.

The numerical run separately passed 6,561 exhaustive four-job instances and 100
five-job fractional-arrival cases. See ../objective-check-20261030-013021/metrics.json.
Original script SHA256: 0C1CF2F4A6B679083BD05DEFDA0F4C17A0A09E9FC210F368576DA8DA582DD621.
