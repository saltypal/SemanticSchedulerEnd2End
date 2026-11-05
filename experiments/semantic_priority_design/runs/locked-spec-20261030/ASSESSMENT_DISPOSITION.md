# Frozen-specification assessment disposition

Protocol SCHED-20261030-v1, 30 October 2026. Main owns the canonical source.

A bounded assessment-only review found three consequential issues. Main corrected
them and obtained a focused static recheck:

1. Class-balanced CE tilts output probabilities. Section 7 now divides by stored
   training class weights and renormalizes before expected-risk inference;
   correction is explicitly not a calibration guarantee.
2. An empty queue with future arrivals is nonterminal. Section 8 now defines
   transitions from dispatch decision to next nonempty ready decision or terminal
   horizon, including automatic idle intervals with corresponding reward.
3. A pre-dispatch state-only reward rate omits the selected action's immediate
   effect and leaves partition-dependent timing. Section 9 now defines
   post-dispatch pending/in-flight context, a fixed global d0/4 integration grid
   with actual events, empty-wait rate zero, and immutable event/header replay.
   Artificial transition subdivision reuses the same physical sampled trajectory.

Focused recheck: no residual blocking contradiction in those sections. This is
not executable validation of the future learned-reward implementation.

Luna's independent numerical receipt reports six groups/620 cases passing for D
cost/targets, normalization, grade quantization, choice/pair loss direction and
replay probabilities/corrections. See numerics/receipt.json. These arithmetic
checks do not validate model convergence, real labels, production integration,
the future learned-reward trajectory integrator, or empirical superiority.

No production model, parameter extractor, codec, GPU/Kaggle execution or training
was modified/launched during this specification freeze. The numerical script is
an experiment-local check, not a second implementation of the scheduler.
