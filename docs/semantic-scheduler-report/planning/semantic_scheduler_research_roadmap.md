# Semantic and urgency driven scheduler: research and engineering roadmap

This proposal is grounded in a source review of `saltypal/SemanticSchedulerEnd2End` at commit `edad7d07c44fd3006e4359b8c32b6dbe143818d7`. It proposes future work; none of the proposed algorithms or benchmarks has been implemented or evaluated here. No new model inference result or literature novelty claim is made. The current checkout has no `.venv/bin/python`, so the earlier onboarding test results are historical evidence rather than a fresh test run.

## 1. Intended contribution and scope

Use the existing emergency/fire task as the initial application. Define success as delivering correct, useful emergency information before it loses operational value under constrained multi-source communication. The objective is receiver task success and timely risk reduction. Reconstruction fidelity, throughput, and resource efficiency remain supporting measurements.

A candidate research question is: does a scheduler using calibrated event importance, remaining deadline slack, and receiver knowledge deliver more actionable critical information on time than urgency-only, semantic-only, and existing class-priority policies, after accounting for extraction, communication, and knowledge-maintenance costs?

The candidate contribution is causal, acknowledgement-aware estimation of the marginal value of another observation about an evolving event, combined with protected criticality tiers and deadline feasibility. Establish novelty through a literature matrix covering deadline scheduling, least-slack scheduling, AoI/value-of-information scheduling, task-oriented semantic communication, redundancy-aware scheduling, and knowledge-assisted resource allocation. Fixed weighted sums, CLIP similarity, graph storage, and EDF alone do not establish novelty.

Freeze the system boundary: begin with an application/edge scheduler deciding which UE image jobs use a constrained link. A radio MAC/RAN scheduler needs additional interfaces and a validated radio/resource model. State which level is implemented and avoid calling an application queue a deployed radio scheduler.

## 2. Repository evidence and immediate gaps

| Area | Existing behavior | Required work |
|---|---|---|
| Scheduling | `Scheduler/policies.py` has FIFO, round robin, fixed class priority, and EDF within class | Semantic policy, global EDF and least-slack baselines, uncertainty treatment, overload policy |
| Input | `INFRA/cli.py` always creates six jobs from `Scheduler/fixtures.py` | Manifest-driven real observations, measured costs, reproducible arrival traces |
| Contract | `INFRA/contracts.py:Job` carries class, deadline, arrival, and resource demand | Immutable semantic evidence, confidence/availability, event/task identity, provenance, remaining work and delivery state |
| Semantics | CLIP-based relevance, concept scores and hand-configured importance | Independent event labels, operational criticality rubric, calibration, open-set and ambiguous-case handling |
| Urgency | `SemParam/extractors.py` uses a supplied deadline | Domain-defined deadlines; end-to-end remaining slack and infeasible-deadline outcomes |
| Confidence | Maximum CLIP softmax score is called reliability | Separate semantic uncertainty from delivery probability; evaluate calibrated risk and abstention |
| Novelty | One minus nearest historical embedding similarity | Event-level information gain; avoid treating visually different pictures as automatically useful |
| Knowledge | Local in-memory NetworkX descriptors, updated after each extraction | Separate sender observations from usable receiver knowledge; ACK feedback, causal state, expiry, corrections and event identity |
| Simulation | All selected jobs finish at slot end; demands are synthetic | Service over multiple slots, measured payload cost, transport outcome, ACK delay, retries and complete timing |
| Evaluation | Completion, deadlines, mean wait, runtime and per-UE completions | Receiver task quality, unique event recall, false alerts, delay tails, overhead, fairness and uncertainty |
| Semantic header | Versioned logical sidecar exists | Include its real transport cost, integrity, attribution and delivery behavior |
| Runtime | CLI-oriented research components | Durable queue, idempotence, backpressure, recovery, health, monitoring, bounded inference and secure ingress |

CLIP concept scores are a softmax across configured prompts. Multiple hazards can coexist, and an irrelevant image still receives a highest-scoring prompt. Validate multilabel recognition, negative/unknown cases, and out-of-distribution behavior. Similarity and maximum softmax are useful proxies, but their operational meaning requires calibration against labels. Image semantics can suggest a hazard; response deadlines and consequences require an explicit application policy.

The existing extraction AoI is the age of the current observation since capture. Receiver information age requires tracking the newest useful information actually received for each event/source. Freshness is derived from age, so weighting both independently can double count the same evidence.

## 3. End-to-end design

At the UE/source, extract a compact semantic descriptor and operational metadata. At the edge, validate the descriptor, combine it with a versioned task policy and available knowledge, then schedule. Transmit the selected payload, decode at the receiver, run the receiver task, and return receipt/quality feedback. Charge descriptor upload, payload, feedback, model processing and database work to the appropriate resource and timing budgets.

The edge must have an authorized way to obtain semantics before admitting the full payload. In early experiments this can be a trace of source-side descriptors; label it as such. Measure source compute and descriptor transmission. An experiment that gives the edge original full-resolution images for free would miss the bottleneck being studied.

Keep ML inference in SemParam/EncDec. Scheduler consumes immutable typed evidence and policy settings. INFRA owns time, queues, transmissions, receiver state updates and persistence. Preserve the project's model-free Scheduler boundary.

For job i at time t, define remaining slack as:

`slack_i(t) = deadline_i - t - estimated_remaining_service_i(t)`

Remaining service includes the stages still ahead of the job: extraction where applicable, encoding, queued/allocated transmission, decoding, receiver inference, and required feedback. Record capture-to-decision and capture-to-useful-reception separately. Jobs with negative predicted slack need an explicit outcome or a smaller feasible representation; wasting a slot on an impossible full-resolution transfer should be visible.

Use lower class integers for higher operational criticality, matching existing contracts. A recommended initial order is:

1. Validated operational criticality tier.
2. Urgency band determined by slack and the domain's response window.
3. Expected useful task benefit within the protected tier and band.
4. Exact slack, then stable arrival/job-ID tie breaking.

Resource feasibility is a hard constraint. Resource cost or favorable channel quality cannot silently promote routine observations above feasible critical ones. An uncertain possible emergency requires a documented cautious handling policy; low confidence should not automatically demote a dangerous case.

Use urgency bands to make the precedence explicit and allow semantic benefit to influence decisions within comparable time pressure. Validate band thresholds on development data. Also report a strict criticality-then-slack variant as an ablation. The chosen precedence is an application decision, not a universal optimal ordering.

Handle critical payloads larger than one slot with remaining-work accounting, fragmentation or a separately validated thumbnail/alert representation. The current skip-if-too-large rule can otherwise starve exactly the job the system intends to protect. Define preemption boundaries and whether partial work can be reused.

## 4. Static semantic policy first

Here, static means that the semantic rubric, learned model, and per-job semantic valuation are fixed during a run. Time and arrivals can still evolve, and slack must be recomputed. If all jobs are known at time zero, call that an offline batch track separately.

Implement a transparent policy using event criticality, urgency band and a calibrated fixed task value. Use the same transport representation for every policy in the first comparison. Begin with externally annotated semantic tiers as an oracle diagnostic, then evaluate predicted tiers separately. Ground truth must not enter the ordinary predicted-policy inputs.

Implement manifest-driven jobs rather than attaching inferred labels to the six synthetic jobs. Preserve those six jobs as contract regression fixtures.

Static acceptance evidence:

- Changing a valid semantic criticality label changes ordering under a matched queue.
- Clock advancement changes urgency and slack as specified.
- Input shuffling preserves deterministic ties.
- Missing evidence, uncertain critical cases, expiration, oversized payloads and overload have explicit outcomes.
- A receiver processes reconstructed/transmitted data, and evaluation reports correct event recognition by deadline.
- Baselines and the proposed policy see the same observations, information available at decision time, link budget and realized exogenous channel trace.

## 5. Dynamic knowledge-aware policy second

Maintain two logical stores: what a source has observed, and what the receiver has successfully obtained and can use. Recording an image during extraction does not demonstrate successful remote delivery. ACKs establish receipt; receiver task feedback establishes how useful the evidence was. Model delayed/lost feedback explicitly and version the edge's estimate of receiver knowledge.

An event record should include stable event identity, location/source association where available, state/severity estimates and uncertainty, newest useful observation timestamp, receiver delivery evidence, unresolved action needs, retention/expiry rules and task/model versions. Log attempted delivery separately from useful knowledge.

Recompute predicted marginal task value at decision time:

`delta_value_i(t) = expected task utility after usable receipt of i - current expected task utility`

The estimate uses only state available at t. It can account for redundancy, a newly worsened event, corroboration from a second UE, receiver uncertainty and missing evidence. Estimate it from a versioned rule or trained model; evaluate actual benefit using independent labels and receiver outcomes.

Use this marginal value in the protected criticality/urgency ordering. Re-score queued jobs on relevant acknowledgements, event-state changes and timing updates. Bound query and inference time. When feedback is stale or the KB is unavailable, use the validated static policy with visible state-staleness metadata.

Demonstrate four causal cases: a repeated unchanged fire view after useful delivery becomes less valuable; an escalating fire remains valuable; an independently corroborating view may remain valuable; an undelivered emergency must not be suppressed as already known. A similarly embedded background scene is not proof that an emergency was resolved.

Start with an inspectable local store and chronological state transitions. Select a distributed/vector/graph backend only after profiling retrieval correctness, state size and latency. The current NetworkX graph provides concept associations and brute-force descriptor similarity, rather than operational event reasoning or cross-UE receiver knowledge.

## 6. Codec methodology and demonstration choice

| Method | Recommended role | Evidence and limitation |
|---|---|---|
| JPEG | First integrated payload codec and conventional comparator | Straightforward actual byte cost; existing benchmark has no wireless delivery chain by default |
| JPEG + framed LDPC/QPSK | Conventional communication baseline | Existing Sionna-based reference code needs runtime validation and scheduler integration |
| SwinJSCC | First neural JSCC candidate | Existing adapter and historical one-image saved-checkpoint smoke; checkpoint/source absent from the inspected cloud checkout |
| DeepJSCC | Additional neural comparator | Production checkpoint is external; documented training/benchmark SNR mismatch must be retained in provenance |
| NTSCC | Later rate-adaptive research comparator | External production checkpoint and rate/environment contracts need real validation |
| Communication/semantic codebooks | Controlled efficiency ablations | Demonstration artifacts do not establish generalization; count codebook sync/version/header costs |
| RAW | Reference ceiling under lossless delivery | Not a communication-efficient default |

No method is established as best by the current evidence. Use JPEG to get a complete scheduler demonstrator quickly. Add SwinJSCC once a compatible real checkpoint passes smoke and held-out task tests. Determine the better codec from on-time task success under equal resource/time/energy assumptions, not PSNR alone.

Keep codec fixed for the first scheduler ablation, then run a codec-by-policy matrix. Fair digital/JSCC comparisons must specify channel uses, transmit power, channel conditions, coding/framing/padding, headers and feedback. Never equate floating-point tensor memory bytes with digital radio payload bits. Match training conditions where possible and clearly identify unmatched checkpoints. Preserve a fixed receiver model and report clean-image performance as a reference. Any domain fine-tuning uses training data only.

For a physical testbed, distinguish a deployable packet transport from a simulated neural-symbol channel. A JSCC simulation with high task scores does not by itself establish a radio implementation.

## 7. Proposed benchmark: Timely Semantic Event Delivery v1

This is a proposed project benchmark name and protocol, not an established dataset or accepted community benchmark. Its defining task is delivering unique, correctly interpreted emergency event evidence by the operational deadline.

Use three tracks:

1. **Controlled scheduling:** oracle semantic metadata and fixed payload/service traces to isolate policy mechanics.
2. **Integrated static semantics:** predicted semantic metadata, actual codec costs and a receiver task; fixed semantic valuation and no adapting receiver knowledge.
3. **Causal dynamic knowledge:** chronological multi-UE event sequences, shared receiver state, delivery feedback, repeated observations and event changes.

AIDER is a candidate source for static disaster categories after label suitability is checked. Class names alone do not supply severity, deadlines, event progression or real timestamped multi-UE traces. Kodak remains useful for reconstruction checks. Build or obtain separately annotated event sequences for the dynamic track. Explicitly mark replay-generated arrivals, urgency windows and channel conditions as scenario assumptions; distinguish replay from field measurements.

The observation manifest should contain immutable image/video-frame identity/hash, event ID, UE/source ID, split, capture time, arrival time, operational deadline, independently annotated event labels and severity, annotation confidence/adjudication, task identity and provenance. Maintain separate fields for policy predictions. Ground-truth annotations are evaluator/oracle inputs only. Avoid semantic labels in filenames.

The execution trace should contain semantic model/version/output and availability, predicted value and uncertainty, KB snapshot/version, decision/rationale, encoded operating point, measured payload and overhead, allocation/remaining work, channel trace identifier, transmission/decoding outcome, receiver prediction, ACK time and observed task utility.

Have at least two annotators apply an explicit operational rubric; adjudicate disagreements and report agreement. Freeze the test split and thresholds after development. Split by event/scene/source and time, rather than randomly separating near-duplicate frames. Codebook fitting, semantic calibration and utility-model training must exclude test events. Replay KBs start from defined causal warm-up history and receive no future observations.

Use paired episode comparisons under the same exogenous traffic/channel realizations. Actions can change delivery and receiver state; permit those causal differences. Use independent events/episodes as uncertainty units. Repeated frames and additional random seeds do not create additional independent real-world events.

Run benign load, congestion, bursts, tight deadlines, fluctuating channels, heterogeneous UEs, unequal payload sizes, irrelevant arrivals, duplicates, event escalation, ambiguous semantics, missing timestamps, delayed/lost feedback and KB outage. Include a critical job that cannot fit one slot, and a stream in which repeated critical arrivals cause starvation. Define expiry and overload policies before seeing results.

Required baselines are FIFO, RR, global EDF, least slack, existing fixed class priority, existing class-plus-EDF, semantic-only, urgency-only, and the fixed semantic/urgency rule. Add an applicable value-of-information/AoI baseline with matched information access. Use a clairvoyant/offline optimizer on small instances as a labeled bound; it cannot be treated as an ordinary online competitor.

Primary outcomes:

- Critical-event recall by deadline at a predeclared false-alert operating point, with per-criticality results.
- Event detection/action delay distribution, especially p95/p99 and critical misses.
- Independently judged unique event task utility received on time. Count duplicate views only when they improve task evidence.

Supporting outcomes are false alerts, per-UE/per-class starvation and timely service, receiver information age, bytes/channel uses, useful value per resource, energy, extraction/codec/KB/scheduler latency, queue growth, retries, header and feedback overhead, CPU/GPU/memory demand and reconstruction PSNR/SSIM/LPIPS where available. Mean wait over completed jobs must not hide dropped jobs.

For an optional aggregate utility, define `J = sum_e(weight_e * on_time_quality_e) / sum_e(weight_e)`, where events, operational weights and bounded quality scoring are independently specified before testing. Report class-specific results alongside J so numerous routine events cannot hide critical failures. Evaluate false alerts separately; avoid rewarding a policy simply for reproducing its own score formula.

Ablate semantics, urgency, receiver knowledge, acknowledgement gating, change detection, calibration, protected tiers, codec adaptation and codebooks. Report both oracle and predicted semantics to separate model failure from scheduling failure. Include all computation and communication costs; compare a cached-feature policy trace separately from a measured full runtime.

Report paired effect sizes with confidence intervals clustered by event/episode. Start with a pilot to estimate variance and choose independent test-event count, workload points and repetitions. Preserve negative findings and baseline wins; do not tune on test results. A journal claim should match the observed scope and show a benefit across meaningful load regimes or precisely explain where it fails.

## 8. Publication and production gates

A publishable study needs a supported novelty claim, formal objective/constraints and information assumptions, causal algorithm, strong comparable baselines, independent labels, leakage-safe benchmark, uncertainty/effect sizes, overhead accounting and reproducible artifacts. An interpretable deterministic policy is a legitimate starting point; RL is optional and must earn its complexity. Include time/space complexity and an optimal small-instance comparison or applicable analytical result.

Production readiness needs a chosen deployment boundary and quantitative latency, availability, critical-miss and capacity targets. Add bounded queue admission/backpressure, remaining-work service, durable state, restart/replay recovery, idempotent job/ACK handling, deduplication, clock alignment, authenticated source identity and trusted criticality metadata, bounded inference/retrieval, schema/model/KB versioning, health/metrics/decision logs, static fallback, model rollback and a staging/shadow rollout. Verify data/model drift and uncertainty handling on the intended population.

Measure p99 complete-path latency and resource capacity on target hardware. Exercise burst load, restart mid-transmission, duplicated/lost/out-of-order messages, missing or stale KB, GPU/inference timeout, malformed or inflated urgency, transport failures and stale ACKs. Specify emergency behavior during degraded operation. A hardware/GPU result or simulator unit test alone cannot establish service reliability. NS-3 or another validated radio simulator is warranted when the paper makes radio/system-level claims; a physical trial is needed for a physical deployment claim.

## 9. Implementation sequence and acceptance gates

1. **Specify the task and benchmark.** Deliver `task_policy.yaml`, manifest schema, annotation rubric, primary metrics, split protocol, information timing and baseline list. Agree on criticality versus urgency precedence. Gate: ambiguous/unknown/deadline cases are explicitly scoreable and causal replay is defined.
2. **Connect one static end-to-end slice.** SemParam produces evidence; INFRA creates real jobs and measured costs; Scheduler makes a transparent semantic/urgency decision; EncDec uses JPEG; INFRA records receiver task output. Gate: a matched congested episode proves decisions respond to semantics and urgency, with complete timing/resource provenance.
3. **Build the evaluation harness.** Add global EDF/least slack and semantic/urgency ablations, paired episodes and independent event metrics. Gate: a frozen held-out pilot can be replayed and failures are visible.
4. **Add causal receiver knowledge.** Implement ACK/usefulness feedback, event state and bounded re-scoring. Gate: repeats, escalation, corroboration and failed-delivery cases produce the intended different decisions without future-state access.
5. **Add neural communication.** Restore compatible SwinJSCC assets, then additional codecs as warranted. Gate: real held-out codec/receiver evaluation and matched resource accounting; no availability-only inference claim.
6. **Complete the study.** Freeze code/config/data versions, expand independent test events based on pilot variance, run ablations and uncertainty analysis, build the literature comparison and write only supported claims.
7. **Harden for the chosen deployment.** Meet explicit service and critical-delivery objectives on the target runtime, exercise failures, then shadow/canary operation and rollback.

The immediate deliverable is step 1 followed by the JPEG-based integrated static slice. Adaptive knowledge-aware scheduling becomes meaningful once receiver delivery and task feedback exist. Jointly learning a scheduler, codec, semantic classifier and codebook at the outset would make attribution and debugging difficult; stage those experiments so each component's contribution is measurable.

## 10. Identified semantic LLM scheduling prior work

Hua et al., *Semantic Scheduling for LLM Inference* (2025), arXiv:2506.12204, is directly relevant prior work: https://arxiv.org/abs/2506.12204 . Its author implementation at https://github.com/Wenyueh/latency_optimization_with_priority_constraints combines semantic class with estimated remaining work, serving-stage constraints and cache management. Add a semantic-class-plus-shortest-remaining-service baseline and prediction-error/overhead sweeps. Distinguish computational KV cache from application receiver knowledge. Full source-grounded mapping and review limits are saved in [llm_semantic_scheduling_methodology.md](llm_semantic_scheduling_methodology.md). The arXiv PDF was blocked by the egress proxy; no numerical paper results or proofs were verified.

## 11. Additional related work and requirements

See [related_work_and_scheduler_upgrades.md](related_work_and_scheduler_upgrades.md) for a targeted comparison of decision-aware multi-UAV scheduling, push/pull anomaly communication, incorrect receiver information, image communication with feedback, decoder-side information, adaptive rates and knowledge-based retransmission.

The closest inspected research implementation already combines receiver belief, marginal decision value, freshness and rate selection. Revise the novelty investigation accordingly: a generic KB plus semantic priority is insufficient. Investigate calibrated real-image evidence, explicit operational deadlines, useful receiver-task outcomes and measured computation/control costs.

Add these items to the staged work: a small emergency-alert representation; event-aware receiver beliefs and incorrectness metrics; useful-delivery probability; representation selection; costed active evidence requests; dependence-aware corroboration; supersession/expiry; feasible large-job handling; work-conserving fallback; complete signalling accounting; and bounded online execution/recovery. Treat each as a separately tested mechanism. Compare weighted freshness/AoII and value-based baselines as well as EDF/least slack, and retain the initial fixed-codec static integration as the first implementation gate.
