# Simplified semantic-aware image scheduler plan

Implementation checkpoint: see STREAM_BUILD_STATUS.md for the classifier-independent
numeric ingress, dynamic synthetic pilot, eight fixture-trained learners and
15 passing focused checks. Real image/teacher/codec validity remains separate.

Revised 29 October 2026. This replaces the earlier research design as the current plan. The previous document is preserved under experiments/semantic_priority_design/runs/20261029_simplified_plan/. This is a design revision: no simulator implementation, teacher qualification, training or measured performance is claimed.

## 1. The project in one sentence

Given complete image messages already received from multiple UEs, learn which available message should be decoded next, using SPI, context relevance, novelty and spatial evidence.

The system has four logical components: workload generator, existing parameter extractor, scheduler and evaluator. One shared implementation serves every policy. Keep the current Codec/, Scheduler/, parameters/ and Evaluation/ owners; put new experimental runs in experiments/. Codec work remains paused. There are no resource grants, retries, new codebooks or preemption.

## 2. The complete flow

~~~mermaid
flowchart TD
    A[Scenario and UE profiles] --> B[Ordered images and capture metadata]
    B --> C[Existing parameter extractor and sender KB]
    C --> D[Complete header and opaque payload messages]
    D --> E[Receiver buffer]
    E --> F[Select one ready message]
    F --> G[Simulated decoder service]
    G --> H[Success or failure]
    H --> I[Successful delivered-header history]
    I --> F
    H --> J[Independent metrics]
    H --> E
~~~

Arrivals continue during service. A free decoder selects one message; a busy decoder finishes its current message. Success updates delivered history. Failure consumes time, exposes no image, and causes no automatic retry.

Routing identifiers are not learned semantic features. Payload bytes/images, evaluator labels and future arrivals stay outside the runtime policy. The student uses parameters and permitted causal queue/history information only.

## 3. What enters the scheduler

| Input | Purpose |
|---|---|
| Predicted SPI | Coarse urgency; planned Critical/High/Medium/Normal |
| Context relevance | Relationship to this UE's device description and mission |
| Novelty | Change relative to that transmitter's previous observations |
| Spatial evidence | Supported entities, prominence and spatial relationships |
| Availability/status flags | Distinguish missing, available zero/false and KB cold start |
| Permitted timing and history | Current wait, ready queue and successful delivered-header summaries |

SPI is one feature, not a hard gate for the learned models. SPI-first is a baseline. Device identity alone must never imply importance.

The mission paragraph is used by extraction and the offline VLM teacher. Its meaning reaches the initial student through the mission-relative parameters. Do not add a language model to the runtime scheduler. Arbitrary unseen mission interpretation is not yet established.

Consume the actual final extractor contract before implementation: fields, units, ranges, missingness, probabilities and whether SPI is global or mission-conditional. Do not invent subclasses or new inputs. Current README says Medium is unresolved and the 35-byte header is not yet integrated; resolve those gaps rather than pretending four-class support already works. Measure actual serialized bytes after freezing the interface.

Sender KB updates follow all offered observations in causal order, independently of receiver policy. Successful receiver history is separate: transmitter novelty does not prove receiver novelty across UEs.

## 4. Build one realistic episode generator

An episode is one incident/environment, a changing UE population and a time-ordered set of image observations. It is not a random collection of unrelated photos.

Support small 4/12/32-UE cases for debugging, then 50/60/100 UEs and population changes during an episode. UEs may join, leave, pause and resume. Stream lengths and frame rates vary. A UE leaving does not delete its already received messages. Define session restart and KB reset explicitly.

Generate scenario-compatible random profile mixtures. Assign devices coherent locations/trajectories, and use real chronological image sequences where available. Nearby devices may report the same incident. Use supported exact repeats and annotated condition changes; do not infer physical identity merely from similarity.

Separate real sequence replay, controlled repeated/cropped imagery and synthetic chronology in the results. A random image stream with simulated GPS is a constructed scenario, not measured field behavior.

### Fifteen profile templates

Each has two short texts: device description and mission. Randomize sensible operating details without changing its semantic definition.

| # | Device/profile | Mission |
|---|---|---|
| 1 | General-purpose smartphone | Neutral everyday photo sharing |
| 2 | Citizen incident-reporting phone | Visible incident hazards and updates |
| 3 | Structural-damage drone | Building-condition and damage evidence |
| 4 | Person-search drone | Visible people and relevant surroundings |
| 5 | Flood/evacuation drone | Water coverage and route access |
| 6 | Wildfire observation drone | Visible fire/smoke and exposed infrastructure |
| 7 | Bridge/road inspection drone | Structural/access-condition observations |
| 8 | Ordinary connected-car camera | Routine road conditions and obstacles |
| 9 | Emergency-response vehicle camera | Approach-route hazards and incident updates |
| 10 | Traffic CCTV | Intersection obstruction and traffic incidents |
| 11 | Public-space/building CCTV | Visible access and safety changes |
| 12 | River/underpass camera | Water coverage and access changes |
| 13 | Industrial inspection robot | Visible equipment and facility changes |
| 14 | Agricultural drone | Field inspection and visible crop changes |
| 15 | Maintenance field camera | Utility/infrastructure condition and repair evidence |

These are our profile templates informed by published application families, not fifteen standardized 6G device classes. References: [ITU IMT-2030](https://www.itu.int/en/ITU-R/study-groups/rsg5/rwp5d/imt-2030/pages/default.aspx) and [IETF RFC 9450](https://datatracker.ietf.org/doc/html/rfc9450). An ordinary phone can capture Critical evidence; a rescue drone can transmit a useless repeat.

### Capture metadata

Generate capture timestamp, source/session/sequence and coherent location with coordinate system and accuracy. Add orientation/altitude only where supported. Tag each field as observed, derived, simulated or unknown. Preserve real source metadata separately from assigned simulation metadata.

Keep capture, extraction-ready, send-ready, complete receiver arrival, dispatch and finish distinct. Define one monotonic simulation clock and a reproducible wall-clock mapping. Under the common-clock assumption these stages must remain causally ordered. Clock errors are a separate stress case.

### Arrivals and service

Start with three load settings: light, congested and overload, plus an incident burst. Parameterize load relative to measured or explicitly synthetic service time; report realized rates/queue sizes. Vary stream length instead of fixing every episode to the same count.

Use constant service time first to isolate selection quality, then message-qualified random service durations. Until the digital codec supplies measurements, label timing synthetic and use normalized service units. Do not turn assumed times into measured milliseconds.

Hold total load constant when testing population effects; separately test increasing total load as more UEs arrive. Otherwise UE count and congestion are confounded.

## 5. Random success and failure

Generate random outcomes reproducibly, before policy replay. For each message save a hidden uniform draw u_i and a declared failure probability p_i. Service fails when u_i < p_i. The actor cannot see that draw or the future result.

Every policy receives the same offered messages, arrival times, durations and hidden outcomes. Index randomness by stable message/session identity, not by the order in which a policy happens to select jobs.

Use:
- All-success control to isolate scheduling.
- Independent failures, initially illustrative p=0.05, with p=0/0.01/0.10 stress settings.
- A separate correlated-outage case where multiple messages are affected by one shared hidden condition.

These probabilities are experimental assumptions, not measured wireless/codec reliability. Correlation mechanisms need an explicit model. Do not correlate failures with SPI without a separate justified stress experiment.

For this study, failure means an unsuccessful service attempt after a complete message arrived. It is not automatically radio packet loss. Transport loss before receiver arrival, UE disappearance, service failure and a horizon-unfinished job are different states. Never silently discard offered messages from evaluation.

Do not guarantee that Critical jobs succeed or redraw outcomes until a method looks good. Failure consumes duration and adds no successful-history entry. A failure stress experiment cannot prove real codec error behavior.

## 6. Train a small neural ranker first

Use one shared MLP scorer with candidate parameters plus simple current-queue and successful-history summaries. Start with hidden widths 64 and 32 and one score output. Choose the maximum score, with deterministic arrival-order tie-breaking. No permanent queue sort is necessary.

This simplifies the earlier Deep Sets-first architecture. Learned set pooling and attention remain later architecture comparisons if simple summaries demonstrably lose useful information. Queue summaries and scores must be permutation-safe and explicitly handle empty history/missing fields.

Train with pairwise preferences:
P(A before B) = sigmoid(score_A - score_B).
Minimize binary cross-entropy against accepted preferences. Genuine ties may use 0.5; uncertain comparisons are excluded, not converted to ties. Normalize by decision so large queues do not dominate through pair count.

One small scorer serves any queue length by scoring each candidate with shared weights. More parameters do not guarantee better semantics. Diagnose cases where identical available features need different teacher decisions before enlarging the model.

## 7. The VLM teacher

Qwen3.5-9B is the leading candidate. First pin and test its exact revision, processor, quantization and backend. Test single images, pairs and planned prefix inputs for memory, speed, valid outputs and judgment reliability.

For offline labels show the teacher:
1. Mission/device paragraph.
2. Actual candidate images.
3. Necessary causal delivered-history context.
4. A request to choose A, B, tie or uncertain, supported by visible evidence.

Initially hide extractor scores, reference labels and scheduler predictions to reduce anchoring. Cache raw responses and parsed labels with image, prompt, model and history hashes. The same candidate pair under different missions/history can require different labels.

Use controls for order reversal, duplicates, clear mission contrasts, missing information, unanswerable scenes, misleading overlays and unsupported claims. A second VLM is a disagreement diagnostic, not human truth. Freeze qualification criteria before labeling and report denominators/uncertainty. If the teacher fails, its training arm remains blocked; annotation-based work can continue.

Qwen does not run on every live queue update. Sample training snapshots from FIFO/SPI/random behavior first, label offline, train the student, then optionally label a bounded set of new student-generated training cases. Never update from final evaluation groups.

Teacher-to-small-ranker distillation is supported by [RankGPT](https://arxiv.org/abs/2304.09542), but its retrieval results are not emergency-scheduling evidence.

## 8. Keep the algorithms, stage the work

One simulator, feature adapter, split manifest and evaluator serve all rows.

| Stage | Comparisons | Purpose |
|---|---|---|
| First study | FIFO, RR, random, SPI-first, relevance-only, novelty-only; linear choice, MLP choice, MLP RankNet and LambdaMART | Establish what information and learning contribute |
| Sequential study | Masked PPO and Masked Double DQN; DDQN + TD-PER | Compare temporal optimization and replay |
| VLM/RL study | Ranker-warmstarted PPO; VLM-learned reward + DDQN; VLM-guided replay | Separate initialization, reward learning and replay guidance |
| Conditional extensions | Priority classifier, Deep Sets, Set Transformer, listwise preferences, VisionPrefer-inspired aspects, FSP-inspired scoring or IPL | Run only when they address a measured limitation |

No requested family is discarded. Existing implementations are reused where correct. Do not build every extension before the first working comparison.

LambdaMART needs actual graded labels grouped by decision; pair-only preferences are not automatically grades. Compare architectures on matched supervision. Compare replay methods with the same reward/architecture. Do not advertise loss variants as independent new algorithms.

The simpler main ranker is a deliberate starting choice, not a claim that it beats the earlier contextual architecture.

## 9. What is the training truth?

Keep three label sources distinct:
- Published assessed priority: reference class, if its input scope matches.
- Published image/object/condition annotations: supported task facts.
- VLM preferences: weak/silver fine-grained supervision.

TREC-IS is a report-priority candidate; validate media/post joins and whether priority is justified by imagery before calling it image urgency. Building-damage annotations alone do not establish rescue urgency. Similar images alone do not establish natural cross-view fact identity.

Reuse the SPI collection, but keep event/parent/near-duplicate groups disjoint across producer fitting, scheduler fitting and evaluation. Prefer out-of-fold SPI predictions for scheduler training. Disclose upstream producer exposure if it cannot be excluded. VLM-generated RGB and VLM-labeled real imagery are different data sources.

An expected numerical delay for every image is unnecessary. We need a supported importance reference, then measure the delay under each policy. A classifier-produced deadline is a prediction, not an operational requirement.

VLM labels can train the model. They cannot independently certify the model when the same VLM also supplied the SPI labels and final judgments.

## 10. RL and RLAIF without extra machinery

**29 October priority-waiting clarification:** the next primary operational objective is timely selection of independently important messages. Use an explicitly versioned priority-weighted queue-wait objective when aligning RL to it: integrate reference-weighted arrived-but-not-yet-dispatched messages over elapsed time, counting a message as selected at dispatch regardless of later decode success. Predeclare the utility convention; ordinal classes alone do not define cardinal harm weights. Never train on delay-gap size alone, because delaying Normal images could game it. Preference-trained rankers can retain their pairwise loss and be measured by the same queue-wait evaluator.

The existing local pilot instead optimizes early successful useful-evidence exposure with a completion-based reward. Preserve that version and its results as auxiliary evidence; it has NOT been changed or retrained by this design clarification. The dispatch-based objective needs numerical references and a separate experiment before claims of alignment.

For pure task-evidence experiments use unit fact weights. Any SPI-to-cost weights are an explicit convention with sensitivity analysis, not a learned law of emergency harm. Delay by reference class remains separately reported.

Account for arrivals within service intervals when integrating cost. Validate the reward against an independent tiny-trace calculation. Do not supply references, future draws or future arrivals to the actor. Variable-time transitions and bounded observations require honest semi-Markov/partial-observation scope.

For RLAIF, Qwen compares two recorded prefixes from the same start and same elapsed endpoint, with timing visible. Learn a small duration-aware preference reward model, then train the same masked RL architecture. Freeze reward versions within blocks; relabel replay after updates. Evaluate externally rather than by the learned reward itself. This adapts the mechanism in [RL-VLM-F](https://arxiv.org/abs/2402.03681), not its original robotics task.

For VLM-guided replay retain the independent reward and change only sampling. Preserve uniform support and correct using actual sampling probabilities. PPO uses fresh rollouts, not a stale DQN buffer.

Do not assume RL must beat a ranker. In simple equal-duration, perfectly graded distinct-job cases, a priority rule may already optimize the chosen cost.

## 11. A small, defensible metric set

| Metric | Question answered |
|---|---|
| Critical/High queue waiting time, by independent reference | Were important messages selected promptly? |
| Selected-within-budget and never-selected fractions | Did short waits hide unselected important images? |
| nDCG@1/@3/@5 on shared ready snapshots | Were higher-reference-grade candidates scored earlier? |
| Supported distinct evidence and repeats | Did novelty expose additional useful information? |
| Selector milliseconds, memory and FLOPs | Is the policy practical? |

**Current decision:** prioritize selection quality plus class-conditioned queue waiting; decoder completion delay is supporting information at this stage. In the main controlled comparison every image occupies one equal service slot d0. This retains realistic queue contention without requiring a physical decoder benchmark. Report waiting in service-slot units W/d0 until measured timing is available. Variable service and random failures remain separate stress settings.

For message i, a_i is complete receiver arrival and b_i is dispatch. Queue waiting is W_i=b_i-a_i. Completion delay is finish-a_i; capture-to-completion is separate. Evaluate SPI groups using independent reference classes, never the policy's own class guesses. Primary timing units are messages/images; supported distinct-fact exposure is a separate semantic analysis.

Desired population-level behavior under contention: mean W_Critical < mean W_High < mean W_Medium < mean W_Normal, with Critical improved relative to FIFO and the SPI baseline also shown. This is not a universal per-image or per-episode law: idle queues yield equal zero waits, different arrival phases cause unavoidable blocking, class counts/load affect means, and a nonpreemptive server cannot interrupt a running image. Do not invent a universal gap or force low-priority delays to make separation look large.

For each reference class report offered/selected counts, actual selected-message mean/median/p95 wait, never-selected fraction, and capped mean wait over ALL assessable messages: min(W_i,tau), with never-selected messages contributing tau. Require full follow-up. Also report selected-within-tau. Tau is an evaluation window, not an image-specific operational deadline. A message dispatched before the horizon counts as selected even if decoding fails or remains in flight at the horizon; its scheduling wait is known. Successful exposure/misses and failures remain separate outcomes.

For a controlled same-arrival, equal-service sanity case, select Critical before High before Medium before Normal. For streaming data compare matched arrival/service traces across policies and report per-class changes, not only a class-ordering Boolean. A priority-delay gap such as mean(W_High)-mean(W_Critical) is descriptive only: it can grow by making High worse. Inspect absolute waits and lower-class starvation alongside nDCG.

nDCG uses independent grades. SPI-only nDCG cannot distinguish equal-SPI images. Use supported task grades or explicitly named silver preferences for within-class distinctions. Uncertain labels and zero ideal gain remain undefined. Shared ranking probes must have identical candidates/history across policies; never rank unarrived images.

Class mean delay can miss within-class ordering improvements. That is why delay, evidence and choice quality are reported together.

Literature checked for this clarification: [NPTEL nonpreemptive priority queues](https://archive.nptel.ac.in/content/storage2/courses/117103017/module9/lec2/3.html) explicitly defines waiting before service and residual-service blocking. [Semantic Scheduling for LLM Inference](https://arxiv.org/html/2506.12204v1) uses urgency-conditioned evaluation but its arrival-to-completion quantity is named waiting time, and its preemption/batching setting differs. [TREC-IS 2020 overview](https://trec.nist.gov/pubs/trec29/papers/OVERVIEW.IS.pdf) reports priority-centric nDCG, supporting a separate ranking-quality axis; report priority is not automatically image-only truth.

**Implementation boundary:** dispatch/arrival events are already logged. Existing priority output includes delivered-only mean wait and completion-based capped metrics, which do not implement the new all-selected/all-offered queue-wait analysis. This clarification updates the plan only; new metrics, reward version and reruns are still required. Old pilot outcomes are not relabeled.

## 12. Six result tables rather than a giant factorial study

1. Overall policies: Critical/High queue wait, selection/misses, ranking/evidence metrics, runtime; completion/exposure auxiliary.
2. Every reference class: offered/selected counts, wait distributions, never-selected fraction and separate decode success/failure outcomes.
3. Same-SPI comparisons: different relevance, novelty and supported spatial evidence.
4. Workloads: UE count/churn, load, burst, service and independent/correlated failures.
5. Learning ablations: parameter families, architecture and supervision/reward/replay mechanisms.
6. Costs and transfer: teacher/student compute, held-out mission/profile/event results and limitations.

Run a small common core, then targeted stresses. Show natural imbalance and balanced diagnostics separately. Use identical traces across policies, validation-only model selection and locked final groups. Queue seeds are not independent disasters; uncertainty is grouped accordingly.

## 13. Inference milliseconds and FLOPs

Measure extractor, teacher and student separately. Report hardware, precision, batch size, warmup, synchronization, candidate/history count and input size. Include complete feature/context-building plus selection latency, with model-forward timing separate.

Measure across actual queue sizes; 100 active UEs can generate much larger queues. Report p50/p95 only with adequate samples and count fallbacks/peak memory.

State MAC/FLOP convention, operator coverage and any uncounted operations. Quantized/VLM FLOPs can be incomplete estimates; report images/tokens and prefill/decode scope. FLOPs do not imply milliseconds.

Offline teacher annotation is training cost. Live extraction overhead belongs in capture-to-completion timing once. Until codec timing is measured, synthetic service must remain labeled synthetic.

## 14. Build sequence and acceptance

| Step | Deliverable | Check before proceeding |
|---|---|---|
| 1 | Freeze actual parameter/mission contract and reference scope | Resolve four-class mapping, missingness, metadata and splits |
| 2 | One event-driven simulator with fixed policies | Tiny reference orders, busy/arrival/failure behavior and accounting |
| 3 | Fifteen profiles, ordered streams, metadata and UE churn | Coherent streams, deterministic seeds and valid clocks/locations |
| 4 | Qwen qualification and cached preference dataset | Valid labels, reliability controls and measured cost |
| 5 | First-stage ranker comparisons | Learning direction, export parity and common replay |
| 6 | RL and VLM variants in the same pipeline | Legal masks, reward-integral agreement and checkpoint/replay integrity |
| 7 | 50/60/100-UE studies and final grouped comparison | Runtime/quality scaling and preserved evaluation boundary |

Core simulator checks:
- No double dispatch or future-message selection.
- Arrivals during decoding are handled at the next decision.
- Invalid/duplicate completions are rejected.
- Sender KB is policy independent; successful receiver history is causal.
- Joins/leaves/resets do not lose accounting.
- All offered messages end queued, in flight, completed-success, completed-failure or explicitly excluded by a recorded rule.
- Horizon-unfinished work is recorded.
- Same seeds reproduce messages, durations and outcomes across policies.
- No-backlog invariance and tiny analytical cases pass.
- Input permutation, missingness and evaluator/identifier leakage checks pass.

Use CPU correctness first, then bounded Kaggle execution after preflight. Reuse current environments, existing notebook identities and parameters/KAGGLE_RUNS.md. Preserve cell logs, executed notebooks, source/data/config identity, checkpoints, metrics and retrieval evidence.

New runs go under experiments/<experiment>/runs/<run-id>/. Historical artifacts stay in place. The prior six aggregate GPU-hour ceiling is not a fresh allocation; project cost from timings before launch. No GPU or codec execution is started by this replan.

## 15. What changed and what happens next

The earlier plan introduced too many policy variants and architecture branches before validating the basic experiment. We now start with one small ranker, simple context summaries, one simulator and six result tables. Advanced methods remain controlled later comparisons.

Next: obtain the actual final parameter examples/definitions, resolve SPI class support and identify which reference labels are genuinely available. Then implement the simulator vertical slice. No new weights, deadlines, header fields or claimed simulator realism will be invented to bypass those inputs.

This revision changes documentation only. It does not claim new code, tested Qwen reliability, support for 100 UEs, measured FLOPs or trained-model superiority.

