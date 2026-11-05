# Final research design: semantic priority learning and decoder scheduling

Revised 30 October 2026. This is the canonical plan. It supersedes conflicting earlier proposals about retries, successful-delivery history and a mandatory RL-first design. Previous versions are preserved under experiments/semantic_priority_design/runs/. This revision does not implement the new objective or retrain a model.

## 1. Decision and research question

Build a parameter-only learned semantic scorer, initially a small RankNet, which selects the highest-scoring complete message currently ready for one nonpreemptive decoder. Use independent reference-class waiting outcomes together with supported semantic-choice evaluation. Retain RL/RLAIF as comparison arms, not as an assumed necessary or superior policy.

The primary question is: can SPI, mission relevance, transmitter novelty and spatial evidence together infer which message deserves service better than predicted SPI alone?

A second, separately gated question is: does any demonstrated action-dependent effect justify sequential planning beyond a learned immediate scorer?

Delay supplies operational pressure. References and preferences define which images deserve that pressure. A weighted delay formula cannot establish its own semantic labels, and a VLM cannot independently certify its own training judgments.

## 2. Fixed system boundary

Complete [parameter record][opaque payload] messages arrive once, wait in a receiver buffer, and receive one service attempt. Arrivals continue while the decoder is busy. The scheduler selects exactly one ready message when free, with no learned WAIT, preemption, resource grant, retry, retransmission or encoder redesign.

~~~mermaid
flowchart LR
    A[Ordered UE observations] --> B[Frozen extractor and sender KB]
    B --> C[Complete messages arrive once]
    C --> D[Receiver buffer]
    D --> E[Parameter-only scorer]
    E --> F[One nonpreemptive service attempt]
    F --> G[Processed; no retry]
    G --> D
    E --> H[Independent evaluator]
    F --> H
~~~

All complete messages are assumed to reach this receiver. A visually poor reconstruction is an outcome, not a request for another attempt. Do not equate poor visual quality with a detectable runtime exception.

The core study does not use reconstruction-success-dependent history. Maintain processed-message events for lifecycle/accounting only. Disable receiver-history-derived learned features for the first model comparison through a versioned observation profile. Preserve legacy controller/history behavior for historical callers; do not silently change old experiment semantics.

Transmitter KB novelty remains causal and enabled. Its history follows the offered source sequence and is independent of receiver selection. Optional receiver-aware redundancy is a later information/utility study, not a prerequisite and not forbidden merely because transmission is one-shot.

Actual codec work remains paused. Simulated service slots establish queue contention without claiming physical decoding latency.

## 3. Meaning and permitted inputs

Freeze the actual upstream parameter export; BetterScheduler/parameters owns its meanings and producer choices. Do not train or redesign the SPI classifier here.

The four input families are predicted SPI, context relevance, transmitter novelty and supported spatial evidence, plus availability/conflict/cold-start flags. Preserve available zero/false versus missing. Do not invent Medium support, subclass probabilities or fields absent from the frozen schema.

Routing identities, filenames, hashes, images, embeddings, payload contents, annotations, reference priorities, future arrivals and hidden outcomes cannot enter semantic scoring.

Define a receiver purpose separately from UE acquisition missions. Start the annotated pilot with the existing building_damage mission and a fixed building-damage-response objective. Routine and unrelated observations compete for the same receiver. A rescue drone does not get priority merely because of its device type; an ordinary phone may provide important evidence.

Different devices can both have high local mission relevance without equal receiver usefulness. First validate whether the existing exported features express that distinction. Arbitrary unseen mission understanding is not established by one scalar relevance value. Record representation collisions instead of demanding that a larger network recover absent information.

Sender novelty is not proof that a message is new to the receiver, and low novelty does not automatically erase urgency. Spatial size/count is evidence, not a direct measure of harm or physical distance. Do not impose monotonic novelty/relevance constraints without a valid task-specific argument.

The base scorer uses candidate semantic fields and masks. Receiver-history features are disabled; queue waiting time is logged rather than assumed to require an aging feature. Queue-context and learned pooling are controlled extensions if their information has a stated purpose. The common policy API and legal candidate set remain unchanged.

## 4. What defines importance

Keep reference sources explicit rather than calling every target ground truth:

| Reference | Valid use | Boundary |
|---|---|---|
| Published assessed priority | Reference-class timing and coarse ranking | Requires valid image/input correspondence |
| Published object/condition labels | Supported task relevance and evidence | Building damage does not establish rescue urgency |
| Declared controlled construction | Exact repeats, object overlap or changed supported conditions | Does not establish natural cross-view correspondence |
| Qualified Qwen preference | Mission-conditioned weak/silver ordering | Teacher agreement is not independent emergency correctness |

TREC-IS remains a possible report-priority source. Verify media-to-post linkage and whether the judgment is supported by the image before transferring priority to image-only evaluation. Unknown remains unknown.

Unknown-reference observations remain in the offered queue. They do not become Normal, zero-utility negatives or zero-cost jobs by default. Skip unsupported supervised comparisons. For the primary reference-cost RL experiment, use episodes with a valid reference weight for every offered job; begin with fully specified controlled fixtures and block that real-data arm if label support is insufficient. Mixed-reference replay may report explicitly assessable-cohort cost and its coverage, but must not present it as total semantic waiting cost or silently train a complete-reference objective from it.

For current VLM-labeled SPI collections, label the priority reference as silver unless independently supported. They can support a distillation pilot; they cannot alone prove correct real-world emergency triage. No new human reviewers are assumed.

Group incidents/events where known, parent images, derivatives and cross-source near duplicates before splitting. Scheduler training should preferentially use out-of-fold SPI predictions. Reusing SPI-training images for scheduler development is disclosed upstream exposure, not a fresh independent final evaluation.

Preserve the previously proposed 240-parent-group feasibility target and 144/48/48 roles only if valid data/grouping supports them. Previously exposed evaluation groups remain development data. Do not invent class balance, statistical power or missing priority labels.

## 5. Main learning method

Use the existing shared 64/32 ReLU MLP with one scalar output. All candidates pass through the same scorer. Select the maximum with arrival, sequence and full routing identity as deterministic tie-breakers.

For a strict reference preference A over B:

P(A before B) = sigmoid(score_A - score_B)

Minimize pairwise binary cross-entropy; use stable logits. Genuine ties can target 0.5. Uncertain/unanswerable comparisons are excluded from that loss, not relabeled as ties. Normalize contributions by decision and episode rather than counting every pair equally across queue sizes.

A preference ranker does not need cardinal harm weights or an expected delay label per image. It also does not acquire calibrated harm weights merely by outputting a scalar.

Initially use annotation-supported preferences where the relation is valid and separately identified Qwen preferences for the weak-supervision arm. Match supervision across architecture comparisons. A method receiving extra teacher labels has a different supervision budget and cannot be advertised as an architecture-only improvement.

Do not silently reuse old marginal-fact-gain targets as new priority labels. Current stream_pilot.collect obtains targets from env.marginal_gains; those belong to the legacy distinct-evidence experiment.

Reuse existing optimizer/checkpoint/export code. Keep the approved small supervised defaults (Adam 3e-4, up to 50 epochs, validation every epoch, patience eight). Use a predeclared validation decision rule considering timing and semantic choice, not training loss or a post-hoc composite score.

## 6. Qualified offline Qwen teacher

Qwen3.5-9B remains a candidate, not a measured winner. Pin model revision, quantization, processor, prompt and runtime. Reuse the registered direct Transformers setup; this plan adds no Ollama/vLLM service.

For each pair show the fixed receiver objective, the relevant UE mission descriptions, real candidate images and sufficient causal sender-sequence evidence when novelty is judged. Previous captures do not imply previous receiver decoding. Hide student scores, producer SPI predictions and evaluator labels to reduce anchoring.

Ask which candidate should be selected first, considering supported urgency, receiver usefulness, mission relevance, genuinely additional information and relevant spatial evidence. Return A, B, tie or uncertain, concise visible justification and unknowns. This is a declared decision rubric, not a discovered universal law.

Cache image/mission/history hashes, raw responses, parsed labels and exclusions. Sample pairs within and across SPI classes and include disagreement cases. Do not label every possible pair in every 100-UE queue.

Before use, test schema validity, swapped orientation, duplicates, obvious task contrasts, unanswerable pairs and unsupported claims. Retain the prior pilot gates (95% schema validity, 0.80 strict-preference balanced accuracy, 0.75 tie/unanswerable handling) only on controls with defensible expected answers. Report counts and uncertainty; these engineering gates do not validate all emergency judgments.

Retain the existing maximum 216 teacher-call feasibility budget, including controls and reserved trajectory/replay feedback. If strict pairs or valid controls are insufficient, label the arm insufficient/blocked rather than claiming convergence. No new GPU allocation or teacher run is launched by this design.

Audit cycles and header/context collisions. Keep difficult evaluation cases even when their distinctions are absent from permitted inputs. A second VLM is a diagnostic, not independent factual truth.

## 7. Delay objective and its exact scope

For complete receiver arrival a_i and dispatch b_i, queue wait is W_i = b_i - a_i. Completion/capture-to-completion delay is separate. No per-image correct delay or invented operational deadline is required.

Report waiting by independent reference class and additionally by supported semantic distinctions. Do not group by the policy's own predicted classes and call that independent validation.

Use weighted waiting as an explicit cost convention and RL control objective:

J_T = integral from 0 to T of sum_{i arrived but undispatched at t} w_i dt
    = sum_{i:a_i<=T} w_i * (min(b_i,T)-a_i)

Unselected jobs use b_i = infinity. A job leaves the waiting set at dispatch regardless of later reconstruction quality. Include arrivals inside service intervals and the terminal interval.

For the cost-control experiment, predeclare Critical/High/Medium/Normal weights 8/4/2/1 and sensitivity conventions 4/3/2/1 and 16/4/2/1. These encode alternative study preferences, not measured clinical harm. Do not tune them to make a policy win. The main pairwise ranker does not require these weights for training.

For RL use negative interval cost, discount 1 over the finite episode. Reward sum must equal -J_T on an independent integration. The old completion-coverage reward stays versioned and is not renamed.

Linear waiting cost accumulates with time, but its marginal rate for a given fixed-weight image stays constant. It does not by itself create an increasing aging bonus or guarantee freedom from starvation. Do not introduce quadratic costs or deadlines merely to make RL necessary.

### Why a strong simple policy is essential

Under one work-conserving nonpreemptive server, equal service durations, fixed known positive weights and full draining, choosing the largest-weight ready job minimizes weighted waiting. At the first deviation, swapping that ready high-weight job with the lower-weight job scheduled first preserves feasibility and service-start times and cannot increase cost. Repeat the exchange. The argument does not cover intentional idling, unequal service, changing utility or missing information.

Therefore the learning opportunity in the core study is inferring the appropriate priority from imperfect parameters. It is not discovering a difficult schedule after perfect priorities have already been supplied. SPI-first need not be optimal when SPI is predicted incorrectly or omits the supported task preference.

For equal-service static queues, unweighted average waiting is invariant to order. Same-class swapping can also leave class delay and SPI-only nDCG unchanged. Do not interpret such invariances as learner failure.

Do not require Critical mean wait < High < Medium < Normal in every streaming trace. A Critical image arriving during service can wait while a previously available High image had zero wait. Use matched-trace changes, offered/selected counts and blocking explanations.

## 8. Evaluation and choosing a result

Keep two co-primary axes rather than one fabricated universal score:

1. Operational priority handling: reference-class waiting and important messages not selected.
2. Semantic discrimination: supported next-choice quality and within-class useful-evidence distinctions.

For each reference class report offered/assessable/selected counts, mean/median wait and adequately sampled tails, and unselected fraction. Include all selected attempts, irrespective of reconstruction outcome.

Use a predeclared 10-service-slot follow-up window for the first controlled timing comparison: report restricted mean min(W_i,10*d0) and selected-within-window. Restrict this analysis to arrivals with full follow-up, including unselected jobs at the cap. Report the excluded boundary cohort separately. This window is an evaluation convention, not an image deadline. J_T uses all arrived jobs' observed waiting through T and is a different quantity.

Report nDCG@1/@3/@5 on common ready snapshots when independently graded references exist; identify silver graded results separately. Pairwise preferences alone do not define cardinal nDCG gains: report held-out pairwise agreement and tie/abstention counts instead. SPI grades cannot establish improvements among equal-SPI images.

Supported exact-repeat/controlled-fact experiments assess information value separately. Do not infer physical identity or urgency from cosine similarity alone.

Report starvation/low-priority unselected fractions and absolute delays; never reward enlargement of the Critical-versus-Normal delay gap by harming Normal traffic.

Profile context/feature construction plus selection and separately model forward time: hardware, backend, precision, warmup, queue size, sample count, p50/p95 where justified, memory and counted MACs/FLOPs. Teacher labeling cost is offline training cost. It is not subtracted from an invented image deadline.

Do not claim a universal winner from one scalar. Use validation to identify timing/semantic trade-offs, then freeze one selection rule before final evaluation. Require evidence of a useful semantic gain with acceptable prespecified timing/starvation effects to promote a complex policy. Report Pareto trade-offs if no single model dominates.

## 9. Algorithm registry and escalation

Retain all requested implementations and distinguish algorithm from label/reward source:

| Arm | Training mechanism | Role |
|---|---|---|
| Linear choice | Masked choice cross-entropy | Simple learned comparison |
| MLP choice | Shared nonlinear choice score | Tests nonlinear benefit |
| RankNet | Pairwise cross-entropy | Main semantic preference candidate |
| LambdaMART | Grouped graded ranking | Strong tabular comparison when grades exist |
| Priority classifier | Context-appropriate independent reference category | Coarse learned comparison; not an SPI extractor redesign |
| Masked PPO | Finite-horizon reference waiting cost | RL control |
| Masked Double DQN | Same cost, uniform replay | RL control |
| Double DQN + TD-PER | Same model/reward, different replay | Replay ablation |
| RL-VLM-F-inspired | Preference-learned reward with DDQN | Conditional RLAIF study |
| VLM-guided replay | VLM changes sampling, not independent reward | Conditional replay study |
| VisionPrefer-inspired | Aspect-based weak preferences | Conditional supervision comparison |

References are FIFO, round robin, seeded random, SPI-first and individual relevance/novelty policies. Same candidate inputs, traces and tie rules apply.

All eight non-teacher arms have historical fixture integrations, not proof of real-data quality. The three teacher arms still need qualified labels. No method is promoted because it is newer or more complex.

Deep Sets pooling, attention and ranker-warmstarted PPO remain optional comparisons. Keep existing portable NumPy inference and training-side frameworks. Export/reload parity, legal masks and true optimizer/replay resume remain acceptance checks.

PPO/DDQN can be trained as controls in the core study. Do not expect them to beat the full-information optimum. Meaningful planning claims require a demonstrated sequential effect: for example unequal service with legitimate observable forecasts, or supported action-dependent receiver evidence utility. Random arrivals/population size alone do not establish such headroom under equal service and fixed known weights.

Before expanding RL, compare an exact tiny reference, same-information greedy and short lookahead under the proposed extension. Keep retrospective future-aware references privileged.

For RLAIF, collect feasible same-start/same-stream/equal-elapsed-time selection prefixes with dispatch timing and enough candidate context to reveal neglected important messages. Learn a duration-aware reward from permitted observations; do not call image-pair labels trajectory supervision. Freeze reward versions per block and relabel replay after changes. Correct VLM-guided replay using actual sampling probabilities and retain uniform support. PPO uses fresh rollouts.

Retain the previous 20,000-decision one-seed feasibility cap for existing RL arms, not a convergence claim. Additional training requires label validity, useful validation progress and a measured compute estimate. No broad hyperparameter sweep is needed now.

## 10. Common dynamic simulator

Reuse the existing event/controller infrastructure. Small cases 4/12/32 UEs precede 50/60/100 and within-episode join/pause/resume/leave. Report active population and queue depth separately; keep already-received messages when a UE leaves.

Keep 15 project profile templates: ordinary smartphone; incident-reporting phone; structural-damage drone; person-search drone; flood/evacuation drone; wildfire drone; bridge/road inspection drone; connected-car camera; emergency-vehicle camera; traffic CCTV; public-space CCTV; river/underpass camera; industrial robot; agricultural drone; maintenance camera. Each has a device description and mission paragraph. They are project choices informed by application families, not standardized 6G device classes.

Use incident-compatible profile mixtures and coherent ordered observations. Preserve real sequences where available; report controlled repeats/crops and synthetic chronology separately. Random unrelated images plus random GPS are not a field-realistic incident.

Capture metadata includes source/session/sequence, capture time and coherent coordinates with accuracy/coordinate system. Orientation/altitude remain supported or unknown. Mark observed/derived/simulated/unknown. Keep capture, extraction-ready, complete receiver arrival, dispatch and finish distinct on a declared clock.

Main service is d0=1 synthetic slot, not measured milliseconds. Retain 48-message, four-UE trace controls, then vary stream lengths and population. Use initial backlog and light/congested/burst workloads, common message-qualified randomness and a predeclared endpoint. Vary UE count at fixed total load separately from varying total load.

All messages arrive complete and receive at most one attempt. The main experiment has no quality-dependent scheduling state. Optional service-duration/outcome perturbations are separate stress settings, never retries. Random visual degradation affects only separately evaluated reconstruction outcomes unless a new observation/objective is explicitly approved. Decoder exceptions are recorded processing errors; neither errors nor poor images automatically reenter the queue.

For any stress comparison generate durations/outcomes before policy replay and reuse by message identity. Never redraw by selection order or guarantee Critical success.

## 11. Implementation sequence and evidence gates

| Increment | Build/reuse | Required evidence |
|---|---|---|
| A | Versioned core observation profile; no receiver-history learning | No labels/IDs/future leakage; legacy callers unchanged |
| B | Dispatch-based wait evaluator and reference-cost reward | Exact numerical integration, fractional arrivals, horizon/idle/one-attempt accounting |
| C | Reference and preference records | Source scope, grouping, unknowns, producer exposure and teacher controls |
| D | Linear/MLP/RankNet/LambdaMART comparisons | Matched supervision, known-direction learning, portable parity, common replay |
| E | Existing PPO/DDQN/PER controls | Same objective/inputs, legal actions, replay/checkpoint integrity |
| F | Qualified teacher-dependent arms | Held-out reward/preference validity, query-budget/cost logs and independent outcomes |
| G | Scaling/transfer/report | 15 profiles, changing UE count/load, held-out groups, measured runtime and bounded claims |

The next code increment is A-B, not a rebuild. Add the new evaluator/target configuration alongside the old completion-coverage experiment; preserve historical source/results. No SPI classifier or codec implementation.

Before real training, test no double dispatch, no future selection, arrivals while busy, immutable snapshots, duplicate/foreign completions, no automatic retry, correct reset, full message accounting and semantic-feature permutation invariance. Keep fixed/unknown reference distinctions and poisoning/split-isolation checks.

CPU correctness first. Reuse existing project environment and Kaggle notebook identities. The prior six aggregate GPU-hour ceiling is not a fresh allocation. Any eventual run needs quota/preflight, measured cost, logs/checkpoints and verified retrieval through the existing artifact map. This finalisation launches no cloud jobs.

Save source/input/config/environment identity, resolved seeds/traces, per-decision events, class/semantic metrics, model checkpoints, exports, failures, plots and a concise report under experiments/<experiment>/runs/<unique-id>/. Preserve partial and negative outcomes.

## 12. Research evidence and bounded numerical checks

Original mechanisms inspected:

| Source | Mechanism/assumption | Decision here |
|---|---|---|
| [Monotonicity-driven scheduling, 2023](https://arxiv.org/html/2305.13706v2) | DDPG with application-specified costs; remote-estimation MSE example | Borrow explicit objective discipline, not its urgency labels or monotonicity theorem |
| [Task-oriented AoI, 2024](https://arxiv.org/abs/2411.00319) | Relevant-target update aging; binary task truth and perfectly accurate receiver classifier assumed | Supports task-relative utility; its request/transmission and perfect-verifier assumptions are not ours |
| [Efficient LLM Scheduling by Learning to Rank, 2024](https://arxiv.org/abs/2408.15792) | Rank measured output lengths; evaluate ranking and serving outcomes | Supports separating ranking from operational metrics; do not transfer its length objective/preemption |
| [RankNet/LambdaMART overview, 2010](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) | Pairwise scoring versus metric-oriented boosted trees | Match label type to algorithm; avoid invented grades |
| [RL-VLM-F, 2024](https://arxiv.org/abs/2402.03681) | VLM preferences train a reward; policy is evaluated using external task progress | RLAIF is a distinct weak-supervision arm, not an independent evaluator |
| [B-Pref, 2021](https://arxiv.org/abs/2111.03026) | Known-reward synthetic teachers with noise, errors, ties and skips; independent return evaluation | Controlled teacher-error stress and independent evaluation; synthetic truth is not real urgency |
| [PEBBLE, 2021](https://proceedings.mlr.press/v139/lee21i.html) | Preference reward learning and replay relabeling | Version learned rewards and replay consistently |

The optional [Deep Sets](https://arxiv.org/abs/1703.06114) comparison concerns permutation-safe pooling, not guaranteed semantic understanding. [TREC-IS results](https://trec.nist.gov/pubs/trec29/appendices/incident/TREC_IS_2020_A_Results.pdf) support priority-centric nDCG, not an automatic image-urgency mapping. [ITU M.2160](https://www.itu.int/rec/R-REC-M.2160-0-202311-I/en) supplies application context, not the project's exact 15 profiles. Qwen qualification is task-specific despite [official vision-model support](https://huggingface.co/Qwen/Qwen3.5-9B).

New CPU mathematical experiment: experiments/semantic_priority_design/check_objective.py. Verified run: experiments/semantic_priority_design/runs/objective-check-20261030-013021/. It checks 6,561 exhaustive four-job arrival/weight instances and 100 five-job fractional-arrival cases against exact legal schedules: zero greedy-minus-exact gap. It also checks 24-order unweighted-wait invariance, same-class waiting-cost indistinguishability, finite-horizon waiting integration and a correct-policy nonmonotone class-delay example. Equal-grade nDCG invariance is an analytical observation; the script does not compute nDCG. These tiny checks support only the stated mathematical scope; they validate neither production changes nor emergency semantics.

## 13. Current status and stop rules

Inspected current production paths: Evaluation/scene_environment.py still rewards successful distinct-fact coverage after completion; Scheduler/training/stream_pilot.py still derives supervised targets from marginal fact gains. Those are coherent legacy choices, but they do not implement the new dispatch-priority objective. STREAM_BUILD_STATUS.md preserves the earlier fixture evidence.

Stop model escalation if references are invalid, permitted inputs cannot express the distinction, or teacher feedback fails qualification. If SPI-first wins under perfect class-only costs, that is expected. If it also wins on noisy real inputs and supported semantic comparisons, report the result rather than changing the benchmark to rescue the neural model.

A journal contribution remains a hypothesis: reliable priority inference from compact semantic parameters, with independent outcome validation, weak-supervision reliability, and realistic controlled scaling. No novelty, broad emergency correctness or lives-saved claim follows from algorithm count or synthetic pilots.

