# Semantic value, learned deadlines, and independent evaluation

26 October 2026. This note answers the user's proposal to train a small model
to assign an image a deadline/allowable delay, used during training and evaluation.
It records inspected literature and a recommendation; it does not add invented
deadlines to the existing dataset or change the locked pilot protocol.

**Subsequent concrete decision:** the next study targets new severe-building
facts (published major-damage/destroyed annotations), measured by distinct severe
fact recall after ten completed decoder attempts. This supplies a meaning-based
reference without an invented time label. The immediate build/access/parameter
sufficiency gates are now recorded at the top of
`SCENE_SUITE_IMPLEMENTATION.md`. Existing LADI experiments retain their original
positive-image scope. Real xBD assets and a new cohort are not yet validated.

## The distinction that determines the experiment

There are three different quantities:

1. **Application value:** what delivering evidence changes for the receiver's task.
2. **Predicted priority:** what a small model estimates from permitted parameters.
3. **Evaluation:** whether selections improve an outcome defined independently
   of that prediction.

A model can estimate priority or allowable delay. Its output can also serve as
weak supervision for a student scheduler. A frozen teacher is not automatically
data leakage, but agreement with its generated deadlines establishes compliance
with its convention. To claim emergency usefulness, validate the convention
against application requirements or independent annotations/outcomes.

If the predicted deadline is only used offline, it is a training label or
evaluation proxy, rather than an externally imposed runtime deadline. It still
needs a defined target. Offline VLM inference belongs in teacher/training cost;
it consumes a job's operational time budget only when the VLM actually lies on
that job's live processing path. An absolute capture-to-result deadline already
accounts for past elapsed processing time; do not subtract it a second time.

## What the inspected papers actually do

These are original-paper method readings, including objectives and evaluation
sections. Paper-specific task definitions are not universal emergency values.

| Paper | How importance or reward is defined | Independent evaluation | Transfer boundary |
|---|---|---|---|
| [Chen et al., Semantic-aware Transmission Scheduling: a Monotonicity-driven DRL Approach, 2023](https://arxiv.org/html/2305.13706v2) | Section II-B, Eqs. (6)-(8): semantic penalty is a nondecreasing age-dependent task cost; the remote-estimation example uses the trace of estimation-error covariance. RL reward is negative summed cost. | Estimation MSE follows from the simulated physical-process/Kalman model. No image-priority training labels or deadline classifier supplies the reward. | Radio/sensor scheduling with known process dynamics. Its monotonicity result does not justify arbitrary image-priority weights. |
| [Liu et al., Data-Importance Aware User Scheduling for Communication-Efficient Edge Machine Learning, 2019 preprint](https://arxiv.org/html/1910.02214) | Sections II-C, III, IV: selects learning samples using classifier uncertainty and channel reliability; SVM boundary distance and CNN predictive entropy are selection proxies. | Section V: held-out MNIST classification accuracy under sample/transmission budgets, supported by digit labels. | It schedules data acquisition for learning. Uncertainty can be useful for learning but is not a validated rescue-urgency score. |
| [Xie et al., Task-Oriented Multi-User Semantic Communications, 2021 preprint / 2022 journal](https://arxiv.org/abs/2112.10255) | Sections IV-V, Eqs. (9), (13): class-based metric learning for image retrieval; answer-label cross entropy for visual question answering. | Section VI-A: Recall@1 for retrieval and answer accuracy for VQA; published class/answer labels supply truth. | It designs communication transceivers rather than a ready-message decoder queue. The transferable principle is task success as an outcome. |
| [Bui et al., On-board Change Detection for Resource-efficient Earth Observation with LEO Satellites, 2023](https://arxiv.org/html/2305.10119) | Sections II-C/D, III-B: important units are true changed pixels. A learned change detector selects content under a missed-change constraint. | Annotated change maps support ROC/AUC and changed-pixel coverage; resource and reconstruction measures are separate. | A good analogue for preserving independently labeled useful evidence. Pixel change is not emergency severity, and the actor accesses images. |
| [Lusvarghi et al., The Search for Relevance, 2025 preprint](https://arxiv.org/abs/2508.07394) | Sections V-VI, Eqs. (4)-(5): receiver-dependent relevance; already-known information has zero added value. Simulation assigns low value zero and high values uniformly in [0.5,1], with context correlations and estimator error. | High-relevance coverage, semantic-value sums, low-relevance content and resource use, under that declared utility model. | Conceptually close to relevance plus receiver novelty. Assigned values are simulated assumptions, not independently validated collision reduction or human urgency. |
| [Maatouk et al., The Age of Incorrect Information: an Enabler of Semantics-Empowered Communication, 2020 preprint](https://arxiv.org/html/2012.13214) | Section II, Eqs. (4)-(9): combines receiver/source state mismatch with a penalty for how long it persists. A correct receiver can incur zero penalty despite old information. | True source state and the declared mismatch/time functions determine cost in modeled applications. | AoII remains partly temporal and requires persistent state/error truth. Independent photos with unknown asset correspondence do not directly supply it. |

The literature therefore uses several foundations beyond throughput: task error,
task accuracy, retrieval success, missed-change risk, distinct relevant evidence,
or an explicitly declared utility model. Time may be a constraint or part of
the cost; it does not supply the meaning of an image on its own.

No official implementation was located in the bounded source search for the
Liu, Bui or Lusvarghi papers. Their methods were inspected; implementation
fidelity was not reproduced. This is a focused evidence review, not an
exhaustive search or a claim that learned deadlines never appear in literature.

## A common formulation we can use

Let H be the successfully exposed evidence and F(H) the receiver's task value.
The marginal value of selecting candidate i is

`gain_i(H) = F(H union {i}) - F(H)`.

Examples of F include negative estimation error, correct task answers, or
coverage of independently annotated relevant facts. Application loss gives the
equivalent form `loss_before - loss_after`. This is the proposed organizing
principle for our project, not a new theorem or a claim of a novel algorithm.

For the current available LADI references the defensible narrow construction is:

`gain_i(H) = 1[published damage-positive label] * 1[exact-copy identity not exposed]`.

It is deliberately binary. Positive, previously unseen observations tie; an
exact repeat earns zero additional evidence. This supplies training targets
without asking a VLM to invent ground truth. Labels and copy identities remain
outside runtime observations. Different viewpoints of one building remain
distinct source observations until independent asset correspondence exists.

The small model estimates this gain from BetterScheduler parameters and causal
receiver history; a choice/ranking/classification loss can train that estimate.
It does not need to invent a globally correct total order when several choices
have equal value.

## Recommended measurements for this problem

1. **Primary semantic outcome:** independently defined useful evidence recovered
   under a fixed decoding budget or fixed elapsed-time horizon. Report the
   coverage curve rather than relying solely on one opaque area score.
2. **Priority diagnostic:** on identical nontrivial queue snapshots, measure
   supported-best selection and regret `max(gain) - selected_gain`. Exclude
   all-tie/singleton states from accuracy and report their counts. This tests
   one current choice without requiring a permanent queue ordering.
3. **Redundancy:** repeat completions and distinct new relevant evidence,
   including local/cross-sender controls. Novel irrelevant content still has
   zero task gain.
4. **Timing/feasibility:** capped evidence-exposure delay, unfinished fraction,
   selection latency and training/teacher cost. J is an aligned early-evidence
   summary for the current protocol, with an independent integration check.

If the application later supplies meaningful urgency deadlines, a valid primary
outcome is **relevant new evidence exposed before its reference deadline**.
For example, normalize `sum_f w_f * 1[first_exposure_f <= deadline_f]` by
offered reference value. The deadlines/weights must come from the task contract
or a validated annotation/rubric, not merely the tested scheduler's prediction.
The current data provide no defensible physical deadline in seconds.

Classification accuracy needs task labels, nDCG needs relevance judgments, and
AoII needs source-state/mismatch truth. Choosing their formulas does not create
those references. CLIP similarity is likewise not an independent definition of
operational importance.

## How to keep the deadline-classifier idea honestly

- **Teacher-student study:** a frozen image/VLM teacher assigns urgency or delay
  targets; train the parameter-only student. Report agreement and teacher-error
  stress, then evaluate independent task outcomes. Teacher-only success remains
  weak-supervision agreement.
- **Declared simulation model:** state a category-to-delay utility rule before
  policy comparison. This can be legitimate controlled research, like the
  explicitly simulated relevance values above. It demonstrates behavior under
  the rule; it does not establish real emergency urgency.
- **Operational deadline prediction:** learn from external deadline/urgency
  references, validate prediction independently, then test the complete scheduler
  under its prediction errors. This requires suitable references not presently
  available in LADI.

For our first study, retain independent positive-evidence references, receiver
novelty and the shared-choice diagnostic. Keep deadline learning as a separately
defined experiment. A deadline predictor would then have a stated target and
an independent check, rather than becoming both the rule and its own proof.
