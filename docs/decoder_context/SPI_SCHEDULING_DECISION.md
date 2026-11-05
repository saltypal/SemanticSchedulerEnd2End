# Four-level semantic priority: evidence, policy, training and evaluation

Decision date: 26 October 2026. This is the consolidated literature-backed
decision for the user's four SPI classes and TREC-IS suggestion. It refines the
previous severe-building discovery proposal; it does not relabel earlier results
or claim that a new image-priority benchmark has already been established.

## 1. Decision and system boundary

Use a **nonpreemptive priority-first decoder scheduler**. At each completion,
select one whole ready message from the highest available predicted SPI class.
Compare FIFO within that class with receiver-context semantic selection within
the same class. This is a current-queue choice, not a permanent order over future
arrivals. The decoder continues its in-flight image while new messages arrive.

SPI order is Critical, High, Medium, Normal. Integers 0,1,2,3 encode order only;
they do not imply evenly spaced utility or deadlines. This strict precedence is
an explicit application choice for the first study, not a consequence of the
ordinal labels alone. Under sustained critical load it can starve lower classes;
measure this instead of silently adding an unvalidated aging constant.

Primary evaluation becomes **completion delay of independently reference-labeled
Critical reports, including missed reports**, accompanied by per-class exposure
and missed fractions. This makes time meaningful by conditioning on actual
semantic importance. Novelty is evaluated using independently established repeat
groups. Keep completion-budget class coverage as a supporting semantic outcome.

No image-specific deadline is needed for this first comparison. A predeclared
evaluation observation budget is not an operational response deadline. The
classifier predicts urgency, and the scheduler is evaluated against separate
reference urgency; neither defines its own success labels.

All policies get identical predicted information, arrivals, payload outcomes,
queue capacity and service realizations. An oracle-label arm is a privileged
diagnostic. There are no radio grants, retransmissions, image/VLM actors or codec
changes. BetterScheduler owns the eventual upstream SPI field and producer.

## 2. What TREC-IS establishes

The [official task instructions](https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2020/participate.html)
use **Critical, High, Medium, Low** and separate information-type categories.
Priority describes the importance of showing a post to a response officer. Our
display name Normal can alias its Low class; retain the original label and
source metadata. Do not silently present Normal as TREC's original terminology.

The [official image dataset page](https://trecis.github.io/datasets.html) lists
312,546 images spanning 122 events. Therefore TREC-IS is not text-only. However,
the independently provided priority target is **post priority**. The
[NIST 2021 overview, Section 3.6](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=934456)
describes separate image-content tagging; this is not independently verified
four-class urgency assigned to each image with its text hidden.

The [2019 overview's labeling section](https://eprints.gla.ac.uk/210955/1/210955.pdf)
describes assessors viewing rendered posts with embedded multimedia. Its usual
one-assessor-per-event process is not evidence of multi-annotator consensus.
Do not assume that staffing detail applies unchanged to every later edition.

An inspected official preliminary 2021-A JSON file, pinned to repository revision
`f480bde06dbd7bdd64bd84595cef79bc556c7666`, contains 6,876 records: Critical 281,
High 1,087, Medium 1,262, Low 4,246. Fields are `postID`, `postCategories`,
`postPriority`. It has no per-image priority or numerical deadline. These are
counts for that preliminary file, not full-dataset counts. A reproducible bounded
preflight is `scripts/audit_trecis_reference.py`.

That preflight has now executed successfully. It retrieved 858,433 bytes with
SHA-256 `6db60d2f282d7822ab29ebf1e350db106458496b9e3b707809e1f37e9a09736a`.
Raw source and receipt are preserved under
`results/scene_suite/spi_reference_20261026/trecis_preflight/`. Predicting Low for
every record in this file would score 61.75% accuracy and zero Critical recall;
this arithmetic illustrates the weakness of overall accuracy, not a trained
classifier result. No images or tweet text were fetched.
This inspected file is a schema/development reference, not a new locked final
evaluation set. Its preliminary release status and original event role remain
recorded; do not train on official test events or declare new queue seeds to be
an independent test split.

The [official download instructions](https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2020/data.html)
distinguish events 1-75 for training and 76-122 for testing. Their historical
download client needs contact/institution details; ongoing service availability
was not verified. The image archive, image-to-post join, complete media access,
and media-specific redistribution terms remain separate access checks. A
repository documentation/code license alone does not settle third-party media
rights.

**Data decision:** retain TREC-IS as the strongest currently identified
human-reference candidate for report urgency. Before training an image-only
SPI producer, verify one event's image/label joins and whether its labels support
that target. Using associated post priority is a legitimate explicitly named
weak-label/proxy study; it does not establish image-only urgency. Missing labels
remain unknown. Do not replace them with predicted Normal. All media from a
post, exact/near duplicates and event groups must remain within one split.

No reviewers are assumed available. If post-to-image evidence cannot justify an
image-only target, stop the image-urgency claim. Continue the independently
annotated severe-building track for image evidence and use TREC-IS for separate
reference-priority policy tests. A blocked join does not justify guessed labels.

## 3. Inspected scheduling and evaluation methods

The following are method/evaluation readings, not abstract-only recommendations.
They solve different systems; the transfer limits are part of the decision.

| Original source | Scheduling and source of meaning | Evaluation and definition of time | Consequence for this system |
|---|---|---|---|
| [Hua et al., Semantic Scheduling for LLM Inference, 2025 preprint](https://arxiv.org/html/2506.12204v1), Sections 3.2-4.2; Eqs. 1-2 | Predicts urgency with a small model, then prioritizes urgency and estimated remaining computation; includes preemption, batching and cache management. Simulated priorities are distinct from its annotated EMS experiment. | Reports delay by urgency; its Eq. 1 uses finish minus arrival, and experiments normalize by generated tokens. Tests predictor errors and predictor latency. Author code includes strict priority/FIFO (PJF). No classifier-created per-request deadlines are needed for this objective. | Closest compute-queue analogue. Adopt explicit class-specific outcomes and the priority/FIFO baseline. We cannot import its preemption/speedups or token normalization into whole-image decoding. Its terminology calls finish-minus-arrival waiting; we use precise queue-wait and completion-delay names. |
| [Klaue et al., Semantic-aware Link Layer Scheduling of MPEG-4 Video Streams, 2003](https://www.jamesgross.org/wp-content/uploads/2016/01/aswn03.pdf), Sections 4.2, 5.1-5.4 | Strict I-before-P-before-B packet queues exploit codec dependencies. Class drop budgets are D,3D/4,D/2. Meaning comes from known video structure. | Uses remaining access-point budgets 100/175/250 ms; authors describe base timing/weights as experience-based/guessed. Assesses received/sent frame quality through PSNR-to-MOS, and supported users meeting a worsened-frame criterion. | Demonstrates semantic classes plus simple queue rules. Guessed video deadlines are declared model assumptions, not a source of emergency urgency truth. Radio dropping/dependencies differ from our queue. |
| [Zou and Subramaniam, On the Value of Information in Status Update Systems, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9030661/), Sections 2-4 | Compares FCFS, waiting replacement and service-preemption disciplines under modeled initial values and lifetimes. No independently labeled emergency classifier. | Age is time since generation; completion delay is waiting plus service. Constant-until-expiry or linearly declining value is evaluated at completion; dropped updates contribute zero. | Provides an explicit value-versus-time model when lifetimes have a declared basis. It does not establish our deadlines or require a learned actor. We adopt counting missed work, not excluding it from reported outcomes. |
| [Chen et al., Semantic-aware Transmission Scheduling: a Monotonicity-driven DRL Approach, 2023](https://arxiv.org/html/2305.13706v2), Section II-B/III; Eqs. 6-8 | Actor-critic scheduling minimizes age-dependent application cost; example cost is estimation-error covariance trace from a physical-process model. | Reports remote-estimation MSE and convergence/training costs. Age-dependent reward is calculated from the model; urgency labels/deadline teachers are absent. | RL reward must have an independent application definition. The age/channel monotonicity theorem does not apply automatically to arbitrary image parameters or SPI categories. |
| [Lusvarghi et al., The Search for Relevance, 2025 preprint](https://arxiv.org/abs/2508.07394), Sections V-VI | Selects receiver-relevant object content; already-known information adds zero value. Relevance weights and estimator errors are explicitly simulated. | Evaluates relevant-content coverage, semantic value and resources rather than only bit rate. No independently validated image-urgency deadline is produced. | Receiver novelty belongs in the semantic comparison. The numerical relevance model supports scoped simulation, not a claim of validated emergency importance. |
| [Official TREC-IS 2021 evaluator](https://github.com/trecis/trecis.github.io/blob/f480bde06dbd7bdd64bd84595cef79bc556c7666/eval/TRECIS-EvaluationNotebook.v3.0.ipynb) | Human post priorities provide reference labels for classification and ranking. Official evaluator maps Low/Medium/High/Critical to .25/.5/.75/1. | Priority error/classification metrics and nDCG@100. Its DCG uses exponential ordinal gains and logarithmic rank discount with ideal per-event normalization and randomized ties. This evaluates rank, not wall-clock decoder completion. | Reproduce official static metrics for compatibility, but add causal queue replay and actual class-specific completion outcomes. Official numerical gains are an evaluator convention, not automatically our cross-class utility weights. |
| [TREC-IS 2019 Accumulated Alert Worth specification](https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2019/TREC_IS_Metrics.pdf), Sections 4.1-4.5 | High/Critical reference posts merit alerts. The metric rewards correct alerts and penalizes missed alerts and excessive false alerts, also considering information types. | A declared threshold .7 controls alerts; false-alert sequences model lost user trust. This is alerting-service utility, not a measured response deadline. | Avoid rewarding every model-declared Critical item. False promotion and missed Critical messages both require independent reference accounting. We do not relabel our metric AAW without implementing its full definition. |

Exact-phrase searches for Semantic Priority Index, its hyphenated form and
scheduling combinations did not establish a standard index/formula for this
problem. We can use SPI as the project name for the four-class prediction, but
should not imply that this particular index is already a standardized literature
metric. This is bounded negative search evidence, not a claim of universal absence.

## 4. Chosen runtime policy and strong comparisons

The first policy is deliberately interpretable:

```text
When the decoder is free:
    snapshot complete, currently ready messages
    choose the highest predicted SPI class present
    within that class:
        baseline: oldest arrival (FIFO)
        semantic candidate: greatest predicted receiver-added evidence
                            then FIFO for ties
    dispatch one whole message; wait for completion
    update successful history only on success
    reconsider the then-current queue
```

The semantic candidate uses the full original queue/history to calculate scores
before selecting. Transmitter-KB novelty is retained as an input but is not
equivalent to receiver-added value. A different sender may have supplied the
same information already. Unknown comparisons are not declared duplicates.

Exact transport duplicates remain a queue responsibility. Weak header similarity
must not suppress a Critical report across SPI classes without a separately
validated duplicate rule. For this first SPI policy, semantic scoring only
breaks ties within the highest class. If a repeat-aware approach trades across
classes later, that changes the declared priority contract and needs its own
comparison.

Necessary baselines are FIFO, source RR, seeded random, SPI/FIFO, SPI plus
receiver-novelty rule, and SPI plus learned within-class scorer. Add priority/
shortest-estimated-service as a diagnostic only when service estimates exist.
Under equal decode cost, SJF adds no real distinction. EDF applies only when
an independently specified operational deadline exists. Keep a gold-SPI oracle
arm to isolate extraction error; never present it as an actor with the same
information.

With correct known classes and no independently meaningful within-class
distinctions, SPI/FIFO already solves the basic priority-choice rule. A learned
scheduler must improve receiver-added evidence or another declared outcome to
justify itself. We must accept a strong simple baseline winning.

There is no finite waiting guarantee for low classes under unbounded higher-class
load. Report undelivered fractions and overload explicitly. Do not claim starvation
prevention because an age feature exists. If a minimum service share is later a
requirement, define and evaluate that trade-off separately.

## 5. Training labels and losses

### SPI producer

Maintain two fields in separate stores: `predicted_spi` in permitted runtime
observations, and `reference_spi` in the training/evaluator reference store.
The upstream producer predicts four probabilities plus availability/provenance.
Its first supervised target is the independently annotated class. Use standard
four-class cross entropy; any class balancing is fitted only on training groups
and recorded. Report per-class precision/recall, macro F1, confusion and especially
Critical-to-Normal under-triage. Calibrate probabilities on development groups
before calling them calibrated confidence.

TREC post labels supervise a report-priority experiment. They do not automatically
supervise an image-only urgency claim. No classifier can turn an unsupported
label transfer into independent truth.

### Within-class semantic scorer

For known reference repeat groups, independently label whether each report adds
previously unexposed evidence after successful receiver history. Train the
existing pairwise scorer on strict added-value differences within class:

`L_pair = mean log(1 + exp(-(score_preferred - score_other)))`.

Preserve ties and unknown correspondence; neither becomes a fabricated negative.
Use independent annotation histories for targets and permitted headers/history
for features. If TREC lacks correspondence labels, use controlled exact-copy
repetition and state that scope. Do not assign natural-view identities using
the same embedding whose novelty we are evaluating.

Cross-class precedence is a policy rule in this first design. Training RankNet
to rediscover that same rule would not demonstrate a new scheduling capability.
RL remains a later mechanism experiment; it is unnecessary just to choose the
highest known class. No additional expensive RL sweep follows from adding SPI.

## 6. Evaluation: semantics-conditioned timing, not arbitrary deadline prediction

For a complete message, define:

`a_i = receiver ingress time`, `s_i = dispatch/start`, `c_i = successful completion`.

Queue waiting is `s_i-a_i`; decoder service is `c_i-s_i`; receiver completion
delay is `c_i-a_i`. Source age needs a trusted capture timestamp and includes
processing/transmission before ingress. If a VLM is only an offline teacher,
its latency is training cost. If it lies on the live path, its processing and
queue time must be included once in the relevant ingress/end-to-end measurement.

### Primary outcome

For distinct reference Critical reports, use restricted/capped mean completion
delay at a predeclared observation budget tau:

`D_C(tau) = (1/N_C) sum_{f:reference(f)=Critical} min(c_f-a_f, tau)`.

For unexposed reports, the contribution is tau. Use first known-reference
successful exposure and earliest known-reference offered arrival for an
independently identified repeat group. Unknown-class copies receive no reference
credit and are counted separately, rather than silently acquiring class labels.
Full follow-up through `a_f+tau` is required; otherwise mark the result censored
or reject the fixed-budget evaluation. Failed attempts consume service and
produce no exposure. No-Critical episodes give undefined Critical metrics with
counts, not a manufactured zero delay.

The first simulated budget is `tau=10*d0`, where `d0=1` is the declared synthetic
decode-time unit; also show the delay/recall curve at 1,2,5,10 units. This is an
evaluation window, not a clinically/operationally validated deadline. Actual
seconds require the eventual measured digital decoder service profile.

For three equal-cost reports already ready, the mean completion delay across
all three is two units under every permutation (completion times 1,2,3). A
Critical report can nevertheless complete at one or three units. Therefore
overall mean delay is insufficient for this priority objective; conditioning
outcomes on independent urgency labels is essential.

Always pair D_C with Critical exposure within tau and the undelivered fraction:
a completion exactly at tau and a missed report can have the same capped delay.
Report analogous High, Medium and Normal outcomes to reveal trade-offs. Do not
average all classes into one number that can conceal harm to Critical reports.

### Supporting semantic outcomes

- Distinct reference-Critical recall after ten completed decoder attempts,
  with failed attempts consuming opportunities; report High and all classes too.
- True-Critical exposure precision among the first completions, false Critical
  promotions, and controlled-repeat exposure. Count reference groups once.
- Reference-priority inversion at each decision: did the selected message have
  lower true priority than another then-ready message? This uses independent
  truth and is distinct from compliance with the actor's predicted SPI.
- SPI prediction quality separate from scheduling quality; gold-SPI diagnostic
  versus predicted-SPI replay separates producer and policy errors.
- p50/p95 selection latency versus queue size, extraction overhead, failures,
  schema bytes and memory. Timing includes declared warmup/hardware/synchronization.

Do not label a lower-class job already in service as an avoidable inversion when
a Critical report arrives later. Nonpreemption imposes a blocking floor equal to
the remaining service time of that job; policy comparisons need identical service
realizations. Simulator time must include any live overhead claimed in latency.

Use held-out events and matched arrival/service traces. Queue randomizations do
not create independent events. Preserve natural priority imbalance; also use
explicit class-balanced diagnostic cohorts and overload tests. Report both, not
a favorable rebalanced result as natural prevalence. Tune only on development
events and freeze the final comparison before opening new evaluation events.

## 7. Implementation and next gates

Additive reference components in this increment:

- `Scheduler/spi_policy.py`: four ordinal values and selection function over
  immutable ready candidates plus predicted-priority associations. No images,
  ground-truth labels or queue mutation. Missing SPI triggers documented FIFO;
  missing secondary scores trigger class-local FIFO; available zero is valid.
- `INFRA/spi_evaluation.py`: independently scoped references, exact-repeat units,
  successful completion outcomes, per-class capped delay/coverage and decision
  inversion. It does not obtain class labels from the policy.
- `scripts/audit_trecis_reference.py`: <=1 MB pinned official annotation preflight.
  It saves provenance/schema/counts, not a manufactured image training cohort.
- `INFRA/tests/test_spi_contract.py`: hand-computed priority, repeat, failed-job,
  under-triage and nonpreemptive-blocking cases.

These components do not yet modify BetterScheduler's Scene JSON or train an SPI
producer. That upstream interface remains owned by the other workstream. The
new selection/reference core is explicit groundwork, not a claim of integrated
real-image SPI performance.

Next, verify one event image manifest, join actual media to assessed posts,
preserve one-to-many mappings/missing media, and decide whether the result is an
image-only reference or a report-priority proxy. Then freeze the producer/SPI
schema and event groups. Only after this gate should the producer and within-class
scorer be trained. Existing xBD severe-building discovery remains the independent
image-fact benchmark rather than being mislabeled as emergency urgency.

The smallest policy tests are: a late Critical arrival behind a running Normal
job; many new Critical jobs starving Normal; Critical misclassified as Normal;
false Critical noise; repeated Critical reports; and equal/variable service.
Test these before expanding policies or GPU training. No new image downloads,
teacher inference, codec work or final evaluation are implied by the label
metadata preflight.

## 8. Checks actually completed

The new numeric SPI cases plus the existing focused lifecycle/learning/teacher
fixtures passed **56 tests in 26.72 seconds**, with no source changes during
execution. Checks cover strict ordinal parsing, Low->Normal alias, semantic
tie-breaking, explicit missingness, independent reference class membership,
failed service, repeated reports, insufficient follow-up, unknown-copy credit,
single-server overlap rejection and unavoidable nonpreemptive blocking.

A read-only Medium assessment found three initial edge-case defects (unknown
copy recall inconsistent with exposure, overlapping service acceptance, and
repeat follow-up checked before aggregation). Main corrected them and added
regressions before the successful run. The contract is standalone and has not
been evaluated as a trained image-SPI scheduler.

Test evidence: `results/scene_suite/spi_reference_20261026/checks/`. Its wrapper
start/end UTC fields were accidentally stamped after completion; the timing note
preserves that defect and reports valid elapsed durations (pytest 26.72 seconds,
wrapper 28.628 seconds). Do not infer valid wall timestamps from that receipt.

Public image-folder access was opened via the official dataset link, but the
browser exposed no usable listing. This does not establish inaccessible media;
archive contents, joins and image-specific labels remain unverified.
