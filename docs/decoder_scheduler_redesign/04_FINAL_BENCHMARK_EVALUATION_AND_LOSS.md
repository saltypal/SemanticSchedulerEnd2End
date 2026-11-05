# Final protocol v1: benchmarking, evaluation and ranking loss

**Date:** 22 October 2026  
**Status:** Finalized methodological contract for the next implementation. No training or measured scheduler improvement is claimed.  
**Authority:** This document refines and supersedes earlier choices of primary ranking loss and headline metric in documents 01 and 02. The eight parameter definitions in [03](03_FINAL_PARAMETER_EXTRACTION_GUIDE.md) are unchanged.  
**Machine-readable contract:** [benchmark_protocol_v1.json](benchmark_protocol_v1.json)

**VLM integration:** [Guide 05](05_VLM_INTEGRATION_AND_TASK_CONDITIONING.md) maps Qwen to task-conditioned parameter production, annotation assistance and auxiliary judging. It preserves this protocol's independent evaluation labels and transmitter-side novelty boundary.

## 1. Decisions we are freezing

| Component | Final choice |
|---|---|
| System boundary | Already-received [ParameterHeader][Complete latent] |
| Runtime action | Select one ready latent for one nonpreemptive decoder |
| Runtime features | R,S,Q,I,F,P,C,N and eight validity bits; no latent/image inspection |
| Knowledge-base novelty | Computed by the transmitter before sending, frozen in the header |
| Main ranker | Shared MLP, 16 -> 32 -> 16 -> 1, ReLU hidden layers |
| Primary training supervision | Independently reviewed task facts and their application weights |
| Main loss | Coverage-swap-weighted pairwise logistic loss, plus L2 regularization |
| Primary benchmark | Static batches, N=32, four logical transmitters, equal decode opportunities |
| Primary cutoff | K=10 |
| Primary metric | Mean weighted distinct-fact coverage over prefixes 1..10 |
| Supporting metrics | Coverage@10, high-priority fact recall, importance nDCG, real decoded-fact time curves |
| Model selection | Validation primary metric; held-out test stays locked |
| Statistical unit | Independent event/collection group, not reused queue draws |
| Core controls | No KB information leakage; same headers, arrivals and outcomes across policies |

We retain all eight finalized fields. Scarcity, complexity and prominence need not have uniformly positive effects on priority. Reliability is semantic-claim correctness, not a radio-link probability.

The main loss does not regress a hand-designed weighted sum of those fields. It learns from independent task evidence.

Finalized inputs do not imply eight guaranteed positive feature contributions. Under this fixed damage-review utility, freshness or complexity may add no predictive value after other evidence. A real-time age-sensitive utility requires independent task justification and a separately versioned target, not multiplying the current labels by the model's own freshness score. Show such null effects honestly.

## 2. Exactly what “good scheduling” means

For this study:

> A good schedule makes independently supported, important and distinct task evidence available within the early decoding prefix.

An important fact is credited when first supplied. Repeated views receive no additional **distinct-fact** credit unless they reveal another supported fact or a genuinely changed condition.

This does not make repeated images universally worthless. They can provide corroboration or better visual quality. Those benefits are outside the core binary-fact utility unless the task rubric explicitly defines and validates a richer utility before testing.

A high-importance observation that is already known to its transmitter can still be the receiver's first evidence of that fact. The training/evaluation target therefore uses actual annotated overlap in the offered queue, not the transmitter's predicted novelty as ground truth.

### Why this changes the earlier generic RankNet plan

Ordinary per-image relevance grades can rank several images of the same important fact at the top. Standard item nDCG can call that perfect.

We retain the RankNet-style scorer and logistic surrogate, but weight its learning signals by **changes in early unique-evidence utility**. This connects the loss to the intended benchmark.

This follows established ideas from metric-sensitive learning-to-rank and diversity evaluation; it is not a claim to have invented pairwise logistic loss or redundancy-aware evaluation. [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/), [LambdaRank overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/), [novelty/diversity evaluation](https://plg.uwaterloo.ca/~gvcormac/novelty.pdf).

## 3. Ground truth: facts and task weights

### 3.1 First task

Keep the existing proposed **building-damage evidence review** task. It is a review/retrieval experiment, not a validated emergency response or life-safety policy.

A task fact is a supported assertion about an identifiable target within an episode, for example:

- Event E / site S / building A / affected condition / episode t.
- Event E / site S / building B / minor-or-greater damage / episode t.

Two views of the same assertion share its fact ID. A new asset, additional supported condition or changed episode can receive a different fact ID.

Do not use an entire disaster event as one fact. Do not treat the same damage class across different buildings as one fact. Do not count a generic parent assertion and its specific child as independent damage facts.

### 3.2 Fact utility

Use the initial reviewed task rubric:

| Fact type | Weight |
|---|---:|
| Outside-task content or unsupported assertion | 0; not a useful task fact |
| Supported affected condition below minor-or-greater threshold | 1 |
| Supported minor-or-greater damage evidence | 3 |

Weights are application choices. They are not probabilities or universal consequence ratios. Test sensitivity using the fixed alternatives (1,2) and (1,5), without choosing the test-winning ratio.

Each fact has one adjudicated weight per task/episode. Do not assign different weights to the same fact merely because one source image has a different predicted score. Uncertain severity is explicitly reviewed; unresolved cases are excluded from the fully assessed primary cohort and counted.

Set standalone source utility U_i=sum of weights of facts in image i. This can exceed 3 when an image supports several distinct assets/facts. Do not confuse it with the header's normalized I.

### 3.3 Annotation process

Two reviewers independently identify supported facts, correspondence, severity/weight and assessability. A third adjudicator or documented consensus resolves the primary factual record. Preserve original disagreement.

Reviewers do not see predicted parameters, scheduler scores or policy order. Matching aids may propose candidate image correspondences, but a retrieved match is not itself truth. Record how correspondence candidates were found and audit missed matches.

Use available location/time/site evidence to support identity. If two images cannot confidently be matched, mark the relationship unknown rather than guessing “new” or “duplicate.”

Freeze fact granularity and the rubric after the development pilot and before opening the final test. Report agreement for fact presence, correspondence and grade separately.

### 3.4 What cannot supply these labels

Public image classes alone, CLIP similarity, the transmitter novelty field, a VLM-only judgment, or the previous scheduling formula cannot establish independent fact correspondence.

A LADI-only class-priority result remains a proxy experiment until the reviewed correspondence layer exists.

## 4. Data and split protocol

Use LADI v2/v2a as the initial source pool and public importance control. Its class labels support the chosen damage task; they do not supply all correspondence, geometry, timestamp or reliability targets. Pin the release and inspect actual metadata. [Official dataset/model overview](https://github.com/LADI-Dataset/ladi-overview)

### Fixed predictor route

1. Use a verified train-only reference predictor, not a deployment model trained across official splits.
2. Partition official validation by event/near-duplicate groups into three disjoint roles:
   - A: ranker training and development annotation.
   - B: parameter calibration/imputation preparation.
   - C: model and hyperparameter selection.
3. Reserve the official test for final evaluation.
4. Remove verified train/test image leakage without moving test images into training. Document exclusions.
5. If more rank-training examples are needed later, use grouped out-of-fold predictor outputs, not artificially perfect in-sample predictions.

Groups take precedence over exact numerical split fractions. Save the role manifest before fitting. Test-group labels do not choose prompts, models, normalizers, scarcity rules or hyperparameters.

### Initial reviewed-data budget

Target 600 reviewed images as an initial build:
- 300 for rank training from A, including the 120-image development pilot.
- 100 for validation from C.
- 200 from the locked official test cohort.

This is a work allocation, not evidence that 600 images provide journal-level power. Whole correspondence groups remain together; if grouping prevents these counts, preserve grouping and publish actual counts.

The test subset is selected with preregistered metadata/assessability criteria and fixed seeds, not by model success. Report selection coverage, excluded events and unassessable relationships.

### Missing parameters

Keep a field unavailable when its required evidence is absent. In particular, receive time is not capture time, raw confidence is not calibrated reliability, and box size is not metres.

The eight-field architecture can run with masks. It cannot claim all-eight empirical validation when some producers are missing. Synthetic ages are a separate sensitivity study, never measured freshness.

## 5. Transmitter KB and workload preparation

Build headers before comparing schedulers.

### Knowledge-base order

For each source/task episode:
1. Start with the declared local observation-memory state.
2. Extract the current image's supported evidence.
3. Compute N and S from the prior snapshot.
4. Serialize the eight values, masks and provenance.
5. Record the current observation in the transmitter memory.

No self-comparison, future observation, evaluator fact ID or receiver completion is used to calculate N.

The primary workload starts local episode memory empty and then processes its observation sequence causally. A nonempty prehistory or shared transmitter KB is a separate recorded experiment.

If real capture order is known, preserve it. Otherwise use a seeded **controlled observation order** and disclose that it is simulated acquisition order. Do not call it measured camera history.

### Primary static queue

- N=32 distinct received message IDs.
- Four logical transmitters, eight messages each.
- One common task and compatible parameter profile.
- All messages ready at decoder time zero.
- One decode opportunity per message; no preemption, dropping or retries.
- Feature snapshot frozen at the common decision epoch.
- N and S stay fixed throughout receiver scheduling.

Use actual source identities where available. Otherwise logical transmitter assignment is a reproducible workload simulation, not a claim that four real radios or devices were measured.

The primary queue-bank seed is 20261022. Generate source assignment once per queue and persist it. If verified incident/site metadata is unavailable, primary nearest-cosine retrieval stays local to the source/task history; a context-filtered novelty result requires legitimately available metadata and a separately identified producer profile.

Within each eligible event/collection group, draw 32 fixed queues without replacement within each queue. Reuse across queues is permitted and disclosed. A group needs at least 32 fully assessed eligible messages; smaller groups appear in a separate small-queue experiment.

Natural queues are sampled without balancing on model outputs or selecting good policy outcomes. The same queue bank and header bytes are shared by all methods.

### Redundancy and mismatch challenges

Report the natural workload first. Add separately labeled stress cases:
- No-overlap control.
- Different views of the same facts.
- Additional facts within one incident.
- Similar-looking different incidents.
- Small important targets and large benign clutter.
- Local-memory disagreement across transmitters.
- Empty, stale and explicitly shared transmitter knowledge.

For reviewed queues, quantify redundancy as:

\[
D=1-\frac{\sum_{f\in\cup_iF_i}w_f}{\sum_i U_i}.
\]

This is repeated fact mass, not a percentage of duplicate files. Report bins [0,0.25), [0.25,0.50), [0.50,0.75), [0.75,1). The all-zero denominator is undefined.

Construct additional challenge queues in a bin only when verified real examples support it. Exact-byte copies are a separate protocol sanity test. Do not fabricate different-view correspondence.

## 6. The exact primary metric

Let F_i be the reviewed fact set of image i. Let K_0 be the fixed application-known fact set; the primary benchmark uses K_0 empty. This is an evaluator reference, not a new receiver-KB runtime input.

Let:

\[
\mathcal F=(\cup_i F_i)\setminus K_0,\quad
W=\sum_{f\in\mathcal F}w_f.
\]

For order pi, the weighted distinct-fact coverage after k completions is:

\[
C_k(\pi)=
\frac{\sum_{f\in(\cup_{r=1}^{k}F_{\pi(r)})\setminus K_0}w_f}{W}.
\]

The final primary metric is:

\[
\boxed{
A_K(\pi)=\frac1K\sum_{k=1}^{K} C_k(\pi)
},\qquad K=10.
\]

Call it **early weighted fact coverage**. It rewards both which facts appear and how early their first appearance occurs.

Equivalent first-appearance form:

\[
A_K(\pi)=\frac1W\sum_{f\in\mathcal F}w_f\,b_K(r_f),
\quad
b_K(r)=
\begin{cases}
(K-r+1)/K,&r\le K,\\
0,&r>K.
\end{cases}
\]

r_f is the first rank supporting fact f. Missing facts have rank infinity.

A repeat receives no additional coverage credit. A first important fact receives full credit regardless of its transmitter-predicted novelty.

This normalization uses all offered fact weight, not an unavailable exact optimum. Some queues cannot achieve 1 with only K decodes. Do not divide by a greedy reference and call the result an exact normalized optimum.

For W=0, report undefined and the number of such queues. They still count in operational reporting. The discrete prefix average is not identical to continuous wall-clock AUC; report the latter separately.

## 7. Final loss function

### 7.1 Model inputs and scores

For each candidate:

\[
x_i=[R_i,S_i,Q_i,I_i,F_i,P_i,C_i,N_i,m_{i1},\ldots,m_{i8}],
\qquad s_i=f_\theta(x_i).
\]

The m values are validity bits. Impute unavailable scalars from frozen development-training statistics. Message IDs, KB versions, hidden facts and source labels are not learned numeric inputs.

The main MLP has 1,089 parameters with the proposed 16->32->16->1 layout. No encoder or decoder loss is changed.

### 7.2 Construct a semantic learning signal

For a training queue:

1. Compute scores and their current deterministic ordering pi.
2. Calculate A_K(pi) from the independent fact labels.
3. For each unordered item pair, swap their positions, giving pi^(ij).
4. Compute:

\[
\Delta_{ij}=A_K(\pi^{(ij)})-A_K(\pi).
\]

Fix i to be the earlier-ranked item and j the later-ranked item in the current order.

- If Delta>0, putting j before i improves the target, so t_ij=0.
- If Delta<0, the current i-before-j relation is preferable under this swap, so t_ij=1.
- If Delta=0 within the declared numerical tolerance, omit that pair.

Use w_ij=abs(Delta_ij). Large utility-changing mistakes get more influence than inconsequential swaps.

These are **queue-dependent, metric-derived targets**, not permanent global labels saying one image is always more important than another.

### 7.3 Pairwise logistic surrogate

Let delta_ij=s_i-s_j and Z_q=sum of nonzero pair weights in queue q. Then:

\[
L_q=
\frac{\sum_{ij}w_{ij}
[\operatorname{softplus}(\delta_{ij})-t_{ij}\delta_{ij}]}
{Z_q}.
\]

For a minibatch of informative queues:

\[
\boxed{
L(\theta)=\frac1{|\mathcal B|}
\sum_{q\in\mathcal B}L_q
+\rho\sum_{\theta_a\in\theta}\theta_a^2
}.
\]

Use stable softplus and stop gradients through sorting, pair targets, Delta, w and Z. Gradients flow through the scores only.

This is a **metric-weighted pairwise surrogate**, not a differentiable exact coverage objective and not a global-optimality guarantee. A swap's effect is computed exactly; the gradient is still a local surrogate.

For W=0, exclude the queue from this loss and record it. If W>0 but Z=0, all tested swaps are neutral: exclude it from the loss but retain it in evaluation. Do not convert zero-Delta pairs into human-judged ties.

If an entire minibatch is uninformative, skip the optimizer step. Avoid numerical division by an arbitrary replacement denominator.

Primary tolerance is 1e-12 in the float64 metric reference. Use the same agreed numerical semantics in optimized code.

All informative pairs within a queue are included; the queue, not each pair, has equal weight in a minibatch. If a future implementation samples pairs, it must specify the sampling/reweighting change as a new objective version.

### 7.4 Gradient direction

Holding weights and targets fixed:

\[
\frac{\partial L_q}{\partial s_i}
\mathrel{+}=\frac{w_{ij}}{Z_q}
[\sigma(\delta_{ij})-t_{ij}],
\quad
\frac{\partial L_q}{\partial s_j}
\mathrel{-}=\frac{w_{ij}}{Z_q}
[\sigma(\delta_{ij})-t_{ij}].
\]

Thus a useful j ranked below an unhelpful repeat i is pushed upward.

The diagonal pair Hessian contribution is:

\[
h_{ij}=\frac{w_{ij}}{Z_q}
\sigma(\delta_{ij})[1-\sigma(\delta_{ij})].
\]

Add this to both item diagonals for a boosted-tree implementation. This diagonal treatment omits cross-score Hessian terms, as an explicit optimization approximation. It is not a claim that the full Hessian is diagonal.

### 7.5 Exact efficient swap computation

Do not recompute every full prefix from scratch in the implementation.

For each fact cache its first and second supporting ranks. A swap only affects facts in the symmetric difference F_i triangle F_j. Facts supported by both swapped images retain the same support positions.

For each affected fact:
1. Find the earliest supporting rank excluding its moved carrier.
2. Insert the carrier at its swapped position.
3. Recalculate the first rank.
4. Accumulate w_f*[b_K(new rank)-b_K(old rank)]/W.

This gives the same Delta as full recomputation. Compare both methods exhaustively on small fixtures before optimizing.

With N=32 there are only 496 unordered pairs. Cost depends on fact-set sizes, not latent dimensions. The optimized reference uses training/evaluation labels; it never becomes a runtime KB query.

### 7.6 Why no auxiliary weighted parameter loss

Do not add a loss encouraging all eight parameters to be large. They are inputs, not eight independent objectives.

Do not train against I*N or a formula over the header. The target is annotated task evidence, which is independent of the producer's predictions.

Do not introduce a second importance/coverage tradeoff coefficient in the core loss. Fact weights already encode the declared task priorities. Ordinary importance RankNet remains an ablation.

### 7.7 Observable-information limitation

Coverage supervision can identify genuine overlap during training, but the runtime model sees only eight scalar summaries. It cannot always distinguish two messages with identical headers or infer unseen cross-source fact identity.

Consequently:
- The method optimizes expected performance within this restricted scorer family.
- Contradictory pair preferences across contexts are possible.
- A low training loss or powerful model cannot guarantee perfect redundancy handling.
- Report aliasing and the gap to privileged fact-aware references.

Training uses more informative labels than runtime features, as supervised learning normally does. It does not license claims that runtime receives those labels or maintains receiver novelty.

## 8. Training procedure and model selection

### Frozen initial setup

| Setting | Value |
|---|---|
| Architecture | 16->32->16->1, ReLU hidden layers |
| Dropout | 0 initially |
| Optimizer | Adam, learning rate 0.001, optimizer weight_decay=0 |
| Explicit L2 rho | Validation grid: 1e-5, 1e-4, 1e-3 |
| Batch | 16 informative queues |
| Train list size / loss cutoff | 32 / 10 |
| Training queue generation | 256 seeded training queues per epoch, from A only |
| Max epochs | 200 |
| Early stopping | 20 epochs without improved mean validation A_10 |
| Gradient clipping | Global norm 5 |
| Selection seed | 0, fixed before selection |
| Final robustness seeds | 0,1,2,3,4; never choose the test-best seed |
| Checkpoint selection | Best validation A_10; ties use earlier epoch, then smaller rho |

The explicit L2 term is not also applied through Adam weight decay. Save coefficients and optimizer configuration.

Fit score imputation on A only. Fit parameter calibration on B, then freeze it before creating the ranker features. Apply the actual uint8 serialization and parsing before fitting/evaluating the scheduler.

Sample train queues only from group-compatible A records. Keep the validation/test queue banks and their source-assignment seeds fixed. Do not update the transmitter KB according to minibatch order or model actions.

Use train-queue sampling seed 20261022+epoch, with epochs numbered from zero, common across model initialization seeds. Persist each epoch's queue manifest. Initialization seed changes model initialization and optimizer stochasticity, not the underlying transmitter histories.

Primary reported deployment seed is 0, fixed in advance. The other seeds report stability; their best test outcome is never substituted. Show each seed and mean/spread.

This grid is a starting contract, not a claim that the values are optimized. If the pilot requires a protocol change, version and justify it before releasing the final test.

### Selection endpoint

Select the model using validation A_10, not the lowest training loss. All evaluation uses the same primary metric definition.

Select the strongest non-main comparator on validation and freeze its identity before test. Do not select the test-winning baseline after seeing the main result.

All architecture comparisons state both input features and training objective. A neural coverage loss versus a tree trained on plain nDCG does not isolate architecture.

## 9. Required baselines and fair comparisons

| Arm | Training target/objective | Purpose |
|---|---|---|
| FIFO | None | Arrival/queue reference |
| Random | Fixed independent seeds | Sanity control |
| Round robin | Source grouping only | Source-fairness control |
| Importance only | Header I | Strong direct semantic baseline |
| I*(1+lambda*N) | lambda grid 0,0.25,0.5,1 on validation | Transparent KB-assisted formula |
| Priority classifier | Independent task-review/evidence category | Coarse learned baseline |
| Pointwise regression | Standalone source utility U_i | Scalar supervised baseline |
| Linear ordinary RankNet | Pairs with unequal standalone U | Capacity/control baseline |
| Neural ordinary RankNet | Same ordinary pair target | Isolates the benefit of the coverage loss |
| Linear coverage-weighted | Exact same coverage loss as main | Tests whether nonlinear architecture is necessary |
| Neural coverage-weighted | Final main loss | Main candidate |
| Ordinary LambdaMART | Standalone integer U with direct configured label gains | Strong standard tabular ranker |
| Coverage-gradient boosted trees | Same detached coverage pair gradients/Hessian approximation | Matched-objective tree competitor |

For ordinary RankNet, omit equal-U pairs unless independently adjudicated tie/preferences exist. Equality in a coarse grade is not automatically a semantic tie.

For LambdaMART, standalone U is an integer sum of reviewed weights. Configure identity label gains over the training/validation label range; do not inherit exponential gains. Evaluate test metrics with our independent evaluator rather than altering the model's training gain range using test labels.

A matched-objective tree arm prevents attributing an objective improvement to neural architecture alone. LightGBM supports custom objectives; implement query grouping, gradient signs and Hessian handling carefully. [Official API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)

For the tree arms, use a predeclared maximum of three configurations, for example num_leaves 7/15/31 with learning rate 0.05, at most 500 rounds and validation early stopping 30. Save actual budgets and runtime. Do not call this equal compute merely because the number of settings matches the neural grid.

SJF is a diagnostic only when independent decode-cost predictions exist. EDF is used only with real deadlines. DQN/PPO/bandits and listwise queue-interaction architectures remain outside the core experiment.

### Privileged references

- True standalone-utility sorting: additive-priority diagnostic, not a coverage optimum.
- Greedy true marginal evidence selection: fact-aware diagnostic, not generally an exact optimum.
- Exact enumeration for N<=8: reference optimum for the specified A_K and fixed outcomes.

These use labels unavailable to deployable policies. Mark them privileged. Never normalize a result by a nonoptimal greedy score and call the ratio a true optimality fraction.

## 10. Supporting evaluation and decoder validation

| Measure | Definition/use |
|---|---|
| A_10 | Primary: mean weighted distinct-fact coverage over prefixes 1..10 |
| C_10 | How much distinct useful evidence is covered after ten completions |
| High-priority fact recall@10 | Fraction of offered weight-3 facts covered; denominator zero is undefined |
| Standalone-utility nDCG@10 | Item-priority diagnostic, using direct U gains |
| First-evidence rank/time | First accepted output for each reviewed fact |
| Continuous verified AUC | Distinct retained facts over actual completion time |
| Tail waits | Median/p95/max for high-priority items/facts |
| Header/policy overhead | Measured parsing/scoring time, memory and bytes |
| Parameter validity | Separate accuracy/calibration/provenance for each of the eight inputs |

nDCG can decrease when the scheduler replaces an important repeat with a distinct lower-weight fact. This is a documented objective tradeoff, not automatically a failed schedule.

### Source-potential versus verified output

Primary development training uses source fact sets F_i. That measures the potential usefulness of choosing a source image.

For fixed decoder replay, independently determine accepted retained facts Fhat_i in the actual reconstruction. Use Fhat for the verified prefix numerator, but retain the source-offered denominator W. This prevents poor reconstruction from shrinking the benchmark's available evidence and inflating its score.

Credit only independently supported source-consistent facts. Hallucinated facts never increase coverage. Unknown output evidence remains unknown; do not assign perfect retention.

Use a separate validated task evaluator and a reviewed reconstruction subset. The same model that generated the header is not independent proof that its reconstruction preserved the claim.

### Actual time

For each fact let a_f be the earliest complete arrival of an image supporting it, and C_f its first accepted retained-evidence completion. Set C_f=infinity if unresolved.

\[
U(t)=\sum_f w_f\,1[C_f\le t],\quad
AUC_T=\frac{\int_0^T U(t)\,dt}
{\sum_{f:a_f<T}w_f(T-a_f)}.
\]

The common horizon and denominator are policy-independent. Failures remain unresolved. Report failures and missing assessments separately.

Freeze received latent, channel outcome, checkpoint, shape, precision, device and per-image seeds before comparison. Decode one message at a time. No new channel draw occurs after selection.

## 11. Mandatory ablations

1. Each of the eight parameters removed individually.
2. Both KB-derived fields N/S removed.
3. Ordinary versus coverage-weighted loss with the same MLP.
4. Linear versus MLP versus tree with the coverage objective.
5. Local versus explicitly shared transmitter KB.
6. Cosine-only versus independently validated context/evidence-aware novelty.
7. Float32 versus actual uint8 header values.
8. Missing-field, corrupt-header and wrong-clock cases.
9. Redundancy bins and similar-looking different incidents.
10. One/two/four/eight logical transmitter counts.
11. Queue sizes 16/32/64 and K sensitivity 5/10/20 where K<=N.
12. Frozen-static versus freshness-updated/streaming replay.

For feature removal, clear the field's bit and retrain with the same budget; do not merely set a valid value to zero. A model trained with all fields can also receive test-time missingness as a separate robustness test.

Synthetic timing is labeled synthetic. Shared-KB comparisons include their synchronization assumptions/cost. Neither is quietly inserted into the main local-KB system.

## 12. Statistical evaluation and acceptance of a claim

Within each event/collection group, average its fixed queue results. The main reported score is the unweighted mean over independent groups, so a large event does not dominate merely by contributing more repeated queues.

Compare policies with paired group differences. Use 10,000 group-bootstrap resamples, seed 20261022, for 95% intervals. Preserve each group's queue bank within a resample. Queue seeds and repeated views are not independent samples.

With few independent groups, explicitly report that confidence/transfer evidence is weak. Do not manufacture narrow intervals by treating 32 queues per event as 32 independent events.

Primary statistical comparison: main seed-0 model versus the validation-selected strongest comparator on A_10. Secondary model comparisons are exploratory or use a declared multiplicity correction; do not present every uncorrected p-value as confirmatory.

The frozen comparison pool includes the matched-objective tree and linear models. Also report main versus importance-only and the same MLP with ordinary RankNet loss as explanatory secondary comparisons.

A convincing result requires:
- Positive held-out improvement with its paired interval against the frozen strong comparator.
- Effect size and absolute metric values, not only statistical significance.
- Weight-3 fact recall and error cases reported alongside the primary result.
- Seed/domain/memory robustness, without cherry-picking.
- Verified reconstruction evidence before an end-to-end usefulness claim.

A nonnegative point estimate for high-priority recall is not proof of no harm. Report its uncertainty; use noninferiority wording only if a margin was agreed before testing and supported statistically.

No effect-size threshold or noninferiority margin is selected after seeing test results. Final 600-image precision may be inadequate; increase independent groups under a versioned protocol rather than recycle a repeatedly inspected test.

## 13. Numerical example

Four received images have these reviewed facts:

| Image | Facts | Standalone utility |
|---|---|---:|
| A | X, weight 3 | 3 |
| B | Same X | 3 |
| C | Y, weight 3 | 3 |
| D | Z, weight 1 | 1 |

W=7 and K=3 for this illustration.

Order A,B,C,D:

- Coverage after each completion: 3/7, 3/7, 6/7.
- A_3=12/21, approximately 0.5714.

Order A,C,D,B:

- Coverage: 3/7, 6/7, 7/7.
- A_3=16/21, approximately 0.7619.

Both schedules eventually decode all four images. Importance-only nDCG@3 prefers the first order because its first three standalone utilities are all 3. This is why item relevance is a diagnostic rather than the headline objective.

Starting from A,B,C,D, swapping B and C gives A,C,B,D with A_3=15/21. Delta=3/21=1/7>0. Thus t_BC=0 and w_BC=1/7.

If s_B-s_C=1, the unweighted pair loss is softplus(1), approximately 1.31326, with derivative sigmoid(1), approximately 0.73106. Gradient descent lowers B relative to C. Its weighted contribution is normalized with the other pairs in that queue.

Swapping B and D instead produces A,D,C,B with A_3=14/21, a smaller improvement 2/21.

These are arithmetic fixtures, not model results or evidence that the eight fields identify real overlaps reliably.

## 14. Implementation and acceptance checks

Deliver in this order:

1. Reviewed fact fixtures, coverage and primary metric reference.
2. Exact swaps, pair signs, loss normalization and score gradients.
3. Sealed splits, causal transmitter histories and actual header parsing.
4. Importance, pointwise and linear fits.
5. Ordinary and coverage-weighted neural scorers.
6. Standard and matched-coverage tree competitors.
7. Locked queue/group evaluation and paired intervals.
8. Fixed decoder replay with independently accepted retained facts.
9. Streaming, source mismatch and external replication.

Do not launch a GPU campaign before small-queue reference checks pass.

Required checks:

- Repeated facts receive no additional credit.
- Distinct supported facts increase coverage.
- A receiver's first fact receives credit even if transmitter novelty is zero.
- Same class at different assets is not automatically duplicate.
- Changed conditions follow the frozen episode/fact rubric.
- W=0 and Z=0 have explicit, different handling.
- Efficient and full-recomputation swap deltas agree.
- Score gradients match finite differences with detached weights/targets.
- Score gradients sum to zero before regularization.
- Hallucinated, failed or unassessed output facts receive no invented credit.
- Hidden test labels do not alter runtime features or schedules.
- IDs and KB versions do not become ordinal model features.
- Exported model inference matches offline inference.
- A no-backlog workload offers no ordering advantage.
- Finite workloads drain without deleting low-score jobs.
- Repeated queues are not counted as independent events.

Reference checks establish mathematical/software consistency, not generalization.

## 15. Required artifacts

Save the resolved protocol and hash; task/profile definitions; source commit and dirty state; split roles; causal transmitter traces and KB digests; serialized headers and message bindings; reviewer facts/disagreement; fixed train/validation/test queue manifests; model-selection report; exported model; per-queue and per-event metrics; paired bootstrap samples; completion traces; retained-fact evidence; hardware/runtime records; and a claims ledger.

Keep training/evaluator facts physically separate from runtime manifests. Save score order, masks, imputation values, fact weights, all seeds and exact objective version.

Reference arithmetic outputs are documentation checks, not measured research results.

## 16. Scope of the claim

If supported, claim:

> Under a fixed, already-received image-latent workload, an eight-field transmitter-header ranker improves early task-weighted distinct-evidence availability relative to strong same-input baselines.

Extend this to reconstructed usefulness only when accepted output facts support it.

This protocol does not establish guaranteed losslessness, spectral-efficiency gain, perfect disaster identity from cosine, exact receiver novelty from local sender history, global coverage optimization, or universal preference for high scarcity/complexity/prominence.

If the matched tree or linear model wins, report that. If headers are insufficient, quantify the failure rather than silently adding embeddings or receiver KB access.

## 17. Primary references

- [RankNet, 2005](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/): established pairwise ranking.
- [RankNet/LambdaRank/LambdaMART overview, 2010](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/): metric-sensitive learning signals and neural/tree relationships.
- [Novelty and Diversity in Information Retrieval Evaluation, 2008](https://plg.uwaterloo.ca/~gvcormac/novelty.pdf): redundancy-aware evaluation.
- [LADI official overview](https://github.com/LADI-Dataset/ladi-overview): source data and predictor distinctions.
- [LightGBM ranker API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html): custom objective interface.

The linear prefix discount, fact-weight rubric, queue design and loss normalization are our specified adaptation. These references do not validate this exact scheduler or establish novelty for its combination.

## 18. Executable mathematical reference

The standard-library [reference script](reference_metric_and_loss.py) implements the metric, exact and efficient swaps, detached pair construction, normalized logistic loss and score gradients. It does not import runtime Scheduler code, train a model, load images or run a decoder.

From the repository root:

~~~powershell
.\.venv\Scripts\python.exe docs\decoder_scheduler_redesign\reference_metric_and_loss.py --report docs\decoder_scheduler_redesign\reference_checks.json
~~~

The saved [check report](reference_checks.json) records 280 randomized synthetic cases, 3,360 fast-versus-full swap comparisons and 1,359 finite-difference score-coordinate checks, plus zero-utility, neutral-swap, initial-known-fact, duplicate-ID and extreme-score cases. All passed in this documentation run.

This establishes consistency of the reference mathematics only. Runtime model export, real dataset annotations, trained generalization and reconstruction usefulness remain implementation/evaluation gates.

