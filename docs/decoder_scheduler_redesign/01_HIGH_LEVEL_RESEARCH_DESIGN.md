# Receiver-side semantic latent scheduling: high-level research design

**Date:** 22 October 2026  
**Status:** Proposed design and evaluation plan; no new training, checkpoint or performance result is claimed.  
**Final methodological authority:** [Protocol 04](04_FINAL_BENCHMARK_EVALUATION_AND_LOSS.md) fixes the benchmark, primary metric and main loss. Earlier generic pairwise-loss formulas below describe the baseline; use protocol 04's coverage-weighted variant for the main training run.  
**Companions:** [Low-level implementation and evaluation specification](02_LOW_LEVEL_IMPLEMENTATION_SPEC.md) · [Final parameter extraction guide](03_FINAL_PARAMETER_EXTRACTION_GUIDE.md)

## 1. The exact system we are building

Complete messages arrive as **[ParameterHeader][Payload]**. Each payload is one complete image latent. A single unchanged decoder processes one latent at a time. The receiver scheduler reads the header and chooses which complete latent to decode next.

**The transmitter maintains the knowledge base, computes novelty before sending, and places that value in the header.** The receiver scheduler does not infer novelty from images, inspect latents, reconstruct candidates in advance, or query a receiver knowledge base to refresh their scores.

The objective is to bring **important and informative image evidence** to the application earlier. Throughput, radio grants and bandwidth allocation are not objectives of this scheduler. An item of low novelty can still be extremely important.

The encoder/decoder architecture stays fixed. We specify the meaning and provenance required of the supplied header, rather than redesigning the image encoder. Semantic-parameter codebooks are excluded. **Knowledge-base novelty remains a core part of the project.**

~~~mermaid
flowchart LR
    subgraph TX["Transmitter: supplied metadata, existing encoder"]
        X["Current image"] --> A["Shared extraction: R, S, Q, I, F, P, C"]
        X --> N["Novelty estimator"]
        KB["Transmitter knowledge base: prior observations"] --> N
        N --> U["Update local observation KB after scoring"]
        A --> H["Header: eight separate parameters + KB version"]
        N --> H
        X --> ENC["Unchanged encoder"]
        ENC --> Z["Complete latent"]
    end
    H --> M["Already-received [Header][Payload]"]
    Z --> M
    M --> P["Parse actual header bytes"]
    P --> Q["Receiver ready queue"]
    Q --> S["Semantic ranking policy"]
    S --> D["One unchanged decoder"]
    D --> O["Image / task evidence"]
    O --> E["Independent evaluation"]
    GT["Hidden annotations and fact coverage"] --> E
~~~

**Where decisions happen:** the transmitter describes a message; the receiver schedules decoding. INFRA owns arrival, queue and completion state. Scheduler owns only the selection rule.

## 2. Why the earlier design is hard to validate

The repository has useful modular contracts, whole-latent processing, outcome traces and header serialization. Several incompatible objectives and parameter generations make the research question unclear. The latest user-selected eight-parameter contract is the authority for this proposed path.

| Finding from inspected source | Consequence | Proposed correction |
|---|---|---|
| A predicted/supplied class gate precedes semantic scoring. | A lower-class candidate cannot win regardless of evidence; gate compliance is not semantic validation. | Compare class priority as a baseline; remove the hard gate from the new semantic policy. |
| The heuristic multiplies relevance, novelty, importance and urgency with undocumented 0.25 floors and divides by service cost. | Units and arbitrary interactions can determine results. | Use explicit importance and novelty meanings, strong simple baselines and independent ranking labels. |
| Current novelty reads a source embedding against receiver history. | It differs from the requested transmitter-computed, serialized novelty. | Preserve legacy behavior separately; new policy sees only transmitter-header novelty. |
| Service duration is resource demand divided by a rate. | Radio demand is being used as a substitute for decoder computation time. | Equal decode opportunities first; measured decoder duration in later replay. |
| Some candidates are built from structured fields instead of parsed bytes. | A header-only claim can accidentally use information never transmitted. | Construct the policy view from actual serialized bytes. |
| Eleven-family plans, seven-family plans and a five-value V2 header coexist. | There is no single minimal contract. | One finalized eight-parameter contract with shared extraction. |
| Annotation-derived impact/urgency and proxies appear alongside predictions. | Hidden ground truth can become runtime input. | Separate prediction, annotation and outcome artifacts physically. |
| Existing nDCG gives all reference classes positive gain. | Task-irrelevant images cannot receive zero utility under that mapping. | Declare task gains, including zero where justified. |
| A finite reconstruction can count as successful processing. | Numerical completion does not show that useful evidence survived. | Log completion, valid output and retained task evidence separately. |
| AWGN can be applied after a job is selected. | The new boundary already assumes full payload reception; order-dependent noise can confound policies. | Freeze received latents and channel outcomes before scheduling. |

These are source-audit findings, not newly measured model failures. Detailed anchors and source fingerprints are in the companion. Preserve legacy code and the existing dirty working tree.

## 3. The finalized semantic input

The user-selected set is **context relevance, scarcity, reliability, task importance, freshness, spatial prominence and scene complexity**, with the explicitly retained **transmitter knowledge-base novelty** as the eighth parameter.

| Field | Meaning | Producer |
|---|---|---|
| Context relevance R | Alignment with the active task | Shared image/task embeddings or calibrated relevance head |
| Scarcity S | Rarity of supported concepts in a declared prior population | Transmitter KB concept counts |
| Reliability Q | Probability that a specified reported semantic claim is correct | Independently calibrated correctness estimator |
| Task importance I | Expected consequence/review gain under a declared task rubric | Evidence probabilities and task gain mapping |
| Freshness F | Age since capture under a task-specific decay rule | Trusted capture time; receiver may update deterministically |
| Spatial prominence P | Image-plane size of supported task-relevant objects | Detector boxes, optionally reviewed masks |
| Scene complexity C | Supported semantic count and diversity | Deduplicated detections and fixed ontology |
| Novelty N | Difference/additional evidence relative to prior transmitter knowledge | Transmitter KB retrieval and evidence comparison |

All eight remain separate inputs. Availability masks distinguish unavailable values from actual zero. Bounded similarities, geometric fractions, engineering proxies and calibrated probabilities retain their different meanings in the producer profile.

The complete formulas, numerical examples, candidate models, validation targets and failure cases are in the [final parameter extraction guide](03_FINAL_PARAMETER_EXTRACTION_GUIDE.md).

**Do not interpret every large value as an instruction to decode first.** A rare but irrelevant object can have high S; clutter can have high C; a large benign building can have high P. A small critical target can still have high I. The ranking model must learn useful interactions from independent task judgments.

**Physical closeness is not bounding-box size.** The primary P measures image-area prominence. Actual proximity requires validated geometric/range evidence; it is not fabricated from pixel area. Freshness likewise requires capture-time provenance and cannot be inferred from a photograph or receive time.

For the initial building-damage review control, importance uses four evidence states from LADI's v2a labels: no building; building without affected evidence; affected below minor; minor-or-greater. Proposed gains are (0,0,1,3), and I is expected gain divided by 3. This is a damage-review proxy, not life-safety urgency. Human review is required for the stronger importance claim.

We simplify **implementation**, by sharing one embedding, one detector, one small task head and one knowledge store. We do not simplify away the finalized parameters.

## 4. Transmitter knowledge-base novelty

### What the knowledge base stores

The first implementation can use a small structured evidence memory:

| Stored item | Purpose |
|---|---|
| Normalized image embedding | Retrieve visually related prior observations. |
| Task/profile and extractor version | Keep comparisons meaningful and reproducible. |
| Source identity and capture time, if available | Define local history and causal order. |
| Incident/site/asset metadata, when actually known | Restrict comparisons to a defensible context. |
| Observation/provenance identifier | Inspect matches and prevent self-comparison. |
| Optional reviewed facts/relations | Test a richer evidence interpretation later. |

A complete knowledge graph is not required. An embedding memory is an operational starting point; it does not establish that the system understands disaster identity or logical facts.

### Cosine similarity: a useful first implementation

For a current normalized embedding e_i and prior normalized entries e_j:

\[
c_{ij}=e_i^\top e_j,\qquad
c_i^{\max}=\max_{j\in\mathcal K_{TX}^{<i}}c_{ij}.
\]

One baseline, matching the repository's existing cosine-distance convention, is:

\[
N_i^{cos}=\frac{1-c_i^{\max}}{2}.
\]

This maps cosine [-1,1] to novelty [1,0]. It is a distance-derived score, **not a calibrated probability**.

Example: maximum similarity 0.96 gives N=0.02; similarity 0.30 gives N=0.35. The latter is more visually unusual relative to that memory. It does not prove a new disaster.

If the declared reference KB is genuinely empty, set novelty to 1 by convention: there is no prior evidence in that scope. If the KB or embedding is unavailable, mark novelty unavailable. Those cases are different.

Compute the score **before** adding the current observation. Otherwise its self-similarity is 1 and it appears non-novel. After scoring, the transmitter may record that observation according to its declared observation-memory rule.

### Make context explicit before adding a bigger model

When verified incident/site metadata exists, retrieve related observations in that context before interpreting visual distance. A separate incident can be new to this transmitter even if its images resemble a known incident.

When metadata is absent, report “visual novelty relative to local task history.” Do not claim reliable new-incident detection. Do not invent event IDs from image categories.

Two photographs with the same damage class can describe different assets. Two photographs of the same asset can contain complementary facts. Category equality, image-ID inequality and a cosine threshold cannot settle those questions by themselves.

A richer version can compare detected entities, attributes and relations with stored evidence, but it must outperform the simple cosine baseline on independently labeled examples before it is justified.

### A limitation that we must retain

\[
\text{new to transmitter A}\ne\text{new to receiver or transmitter B}.
\]

A local transmitter can have observed a scene that the receiver has never decoded. Two transmitters can each mark the same incident as novel. A frozen header also cannot react when a queued duplicate is decoded first.

Therefore this design studies **transmitter-KB-assisted decoder prioritization**, not perfect receiver-marginal information scheduling. Shared transmitter knowledge is an explicit separately costed experiment, not an assumed capability. No receiver feedback, acknowledgments or resource grants are introduced into the core system.

## 5. The header contract

Propose a separate **32-byte V3 header**, keeping existing V2 unchanged:

| Information | Role |
|---|---|
| Version and eight validity bits | Parse and distinguish unavailable values from zero |
| Task/producer profile ID | Define all score meanings, methods and normalizations |
| Message token | Bind header to the existing envelope; never a learned feature |
| Transmitter KB snapshot version | Reproduce novelty/scarcity reference; never a numeric priority |
| Capture timestamp | Support freshness when clock provenance is valid |
| Eight uint8 scalar values | R, S, Q, I, F, P, C, N, separately |
| CRC32 | Byte-corruption detection, with its limitations |

Eight bytes carry the eight semantic scalars. The low-level specification defines exact offsets, score order, quantization, fallbacks and timestamp rules.

Source/session identity comes from the existing envelope. Registry and run artifacts identify actual model, vocabulary, clock and KB snapshots; an integer version alone is insufficient provenance across sources.

Boxes, embeddings, masks, matched KB images and explanations stay in the transmitter audit record. The receiver scheduler does not receive them in the primary study.

Do not add a separate learned urgency class, semantic-parameter codebook or radio-grant field. Real deadlines are a separately justified extension; their absence does not justify inferring them from an image.

## 6. Ranking: neural RankNet candidate, strong simple competitors

The input is x_i=(R,S,Q,I,F,P,C,N), with eight availability bits. The main candidate is a **small shared neural RankNet-style scorer**. It scores each candidate with the same function, so it supports different queue sizes without padding the whole queue into fixed action slots.

A starting architecture is 16 inputs -> 32 ReLU units -> 16 ReLU units -> 1 score: 1,089 trainable parameters. This is a proposed starting budget, not an optimized architecture.

Train on independent preferences: which image should be decoded first for the declared task and a fixed, documented prior-information context?

\[
L=\frac1Z\sum_{ij}w_{ij}
[\operatorname{softplus}(s_i-s_j)-t_{ij}(s_i-s_j)]
+\rho\|\theta\|_2^2.
\]

t=1 favors i, t=0 favors j and t=0.5 denotes an explicitly judged tie. Reviewer disagreement is retained separately. This is established pairwise learning-to-rank, not a new loss invention. [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)

**Do not multiply all eight parameters.** Reliability need not be an importance multiplier; scarcity, novelty and complexity need not deserve uniformly positive weight. Separate inputs permit testing these relationships.

Required competitors:

- FIFO, seeded random and coarse category priority.
- Importance-only ordering.
- A transparent importance/novelty baseline I(1+lambda*N), lambda selected on validation.
- Priority classifier and pointwise regression on the same legal fields.
- Linear pairwise ranking.
- LambdaMART as a strong tabular learning-to-rank alternative.
- The current formula as a labeled legacy/extra-input diagnostic when appropriate.

LambdaMART is not declared the empirically strongest method before testing. Neural ranking is the main candidate from the proposed study table, not a guaranteed winner. If the simple model matches it, report that result. [Microsoft ranking overview](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)

Static ordering freezes the feature snapshot at the common decision time. Streaming or freshness-updated dispatch recomputes only the allowed time-derived F among complete arrived messages. Transmitter novelty remains fixed. Low-score images are not dropped; finite workloads drain.

A shared per-item scorer cannot observe overlap absent from the header. It does not become a set-level redundancy optimizer merely because it is neural. A listwise/queue-interaction model would be a separately tested extension with a declared information budget.

For known additive weights and equal service times, descending weight is already optimal for weighted completion time. That classical limit does not establish optimal distinct-fact coverage. Ordinary sorting or adding RL is not a new theorem. [Smith, 1956](https://doi.org/10.1002/nav.3800030106)

## 7. The benchmark must answer four separate questions

1. **Parameter validity:** do the supplied measurements and predictions match their declared meanings, including independently supported importance?
2. **Novelty:** does transmitter novelty distinguish repeated evidence from useful additions?
3. **Scheduling:** do the eight header values bring valuable, distinct evidence earlier?
4. **Decoded usefulness:** do actual reconstructions preserve the evidence credited by the benchmark?

Success at one question does not answer the others.

### Dataset layer

Use LADI v2, pinned to its v2a configuration, for an initial damage-evidence proxy benchmark. Its published official split is 8,030 train / 892 validation / 1,041 test, with later-year testing. The authors distinguish train-only reference classifiers from deployment models trained on all splits; use a verified reference artifact for official-split tests. [LADI paper](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/papers/Scheele_LADI_v2_Multi-label_Dataset_and_Classifiers_for_Low-Altitude_Disaster_Imagery_CVPRW_2025_paper.pdf)

LADI image labels also do not automatically provide capture-time provenance, physical range, calibrated reliability or exact object counts. Inspect the pinned schema; keep unsupported values unavailable or use a separately labeled metadata-rich subset. Synthetic ages are sensitivity tests, not measured freshness.

**Public image-class labels do not provide semantic-redundancy truth.** They cannot establish that two photographs show the same building, incident fact or changed condition. Novelty evaluation requires an additional reviewed subset or a dataset with verified correspondence.

Retain public-label priority results as a proxy. Do not call them evidence of successful knowledge-base reasoning.

### Novelty annotation layer: core, not an optional future extension

Begin with a 120-image feasibility pilot and two independent reviewers. Plan an initial budget around 600 images, adjusting after disagreement and group/event coverage are known. These counts are planning allocations, not statistical sufficiency guarantees.

For selected pairs or small history-plus-candidate groups, label:

- Same or different incident/site, where assessable.
- Same known fact, corroboration, additional fact, changed condition or unassessable.
- Relative task usefulness and whether the additional view materially helps.

Reviewers see source images and a fixed description of prior evidence, not predicted scores. Distinguish transmitter-history judgments from receiver-prefix utility. If experts are unavailable, report a non-expert feasibility study with appropriately limited claims.

Use real different-view examples for semantic claims. Byte-identical repeats are a necessary sanity test, but cannot alone demonstrate understanding of novelty.

### Primary experiment

Static queues contain 32 complete messages, all available at time zero; primary prefix length K=10. Every message receives one equal-cost decode opportunity. The finalized primary endpoint on the reviewed novelty subset is **mean weighted distinct-fact coverage over prefixes 1..10**, as specified in protocol 04. Coverage@10 remains a supporting endpoint.

Importance nDCG@10, human pair preference, top-priority recall and importance-weighted completion are secondary diagnostics. nDCG can favor repeated high-severity images, so it cannot be the only headline metric.

Evaluate:

- No redundancy, as a control.
- Verified repeats and alternative views of the same evidence.
- Same incident with genuinely additional evidence.
- Different incidents with similar visual appearance.
- Local transmitter KB disagreement and stale memories.
- Multiple transmitters with overlapping observations.

Measure natural frequency when known; report constructed challenge mixtures separately.

### Actual decoder replay

Freeze received latent, channel outcome, codec, shape and seed before comparing policies. Decode selected payloads with the unchanged checkpoint.

A source annotation is not a reconstruction score. Measure retained task evidence on outputs using an independently validated evaluator and reviewed examples. Finite output tensors, PSNR or CLIP similarity alone do not establish preservation of the specific damage fact.

## 8. Metrics that match the objective

| Metric | What it tells us | Limitation |
|---|---|---|
| Mean weighted distinct-fact coverage over prefixes 1..K | Primary: how early independently annotated useful facts appear, without duplicate credit. | Needs verified fact grouping and a declared granularity. |
| Weighted distinct-fact coverage@K | Supporting: how much evidence is covered at the cutoff. | Does not distinguish orders with equal final prefix coverage. |
| Distinct-evidence coverage over time / AUC | How early useful facts become available. | Source-only curves do not prove decoded retention. |
| Time to first usable evidence per fact | Delay until an accepted output establishes each fact. | Failed or unobserved outputs remain unresolved. |
| Importance nDCG@K | Whether important images appear early. | Can reward several images of the same fact. |
| Pairwise preference agreement | Agreement with independent task judgments. | Task/rubric dependent; report disagreement. |
| Novelty retrieval/classification metrics | Whether matches and novel facts are recognized. | Cosine is uncalibrated unless a mapping is evaluated. |
| Per-importance p95/max wait | Harm to important images scored incorrectly. | Mean delay can conceal bad tails. |
| Header and policy cost | Bytes, memory, query time and dispatch overhead. | Separate transmitter extraction/KB cost from receiver scheduling. |

The low-level document gives equations, toy examples, uncertainty methods and undefined-case handling.

## 9. Required comparisons

| Policy or ablation | Question |
|---|---|
| FIFO and seeded random | Is semantic information useful at all? |
| Coarse predicted-category FIFO | Is a simple class already sufficient? |
| Importance only | What is gained without KB novelty? |
| Novelty only | Does novelty alone mistakenly prioritize irrelevant images? |
| Naive importance-times-novelty | Does suppressing known critical evidence cause harm? |
| Validation-tuned bounded bonus | Strong transparent importance-plus-novelty baseline. |
| Priority classifier / pointwise / linear pairwise | Do simpler models explain the gain? |
| Neural RankNet / LambdaMART | Which same-input ranker generalizes best? |
| Cosine memory versus context-filtered memory | Does context improve novelty validity? |
| Local versus explicitly shared transmitter KB | How much does cross-source mismatch matter? |
| Tiny exhaustive fact-coverage reference | How far is a restricted scheduler from the true small-queue optimum? |

All deployable policies receive the same bytes, messages, task, arrivals and fixed outcomes. The privileged fact-coverage reference is evaluator-only. The old enriched heuristic is a separate extra-input diagnostic unless its inputs are genuinely reproduced.

For static equal-cost additive importance, true-grade sorting supplies an importance upper bound. It is not a unique-fact oracle.

## 10. Candidate contribution and honest limitations

A plausible research contribution is a validated **transmitter-knowledge-assisted, compact-header decoder scheduler** with:

- Explicit separation of intrinsic importance, local novelty and receiver usefulness.
- An eight-parameter serialized interface with an observable-information audit and feature-validity tests.
- A redundancy-aware benchmark with independent fact and preference annotations.
- Measured value and failure cases for cosine versus contextual knowledge retrieval.
- Fixed-decoder evidence showing earlier usable reconstructions.
- Local/shared-memory, staleness, cross-source and header-information ablations.

“Using a KB,” “adding a score” and “using RankNet” alone are not a novelty claim. Related learning-to-rank, content-aware worklists, information coverage and semantic computation scheduling need a focused literature comparison. Journal acceptance cannot be promised before the evidence exists.

Related receiver computation work includes [CAPS-TDPC, 2026 preprint](https://arxiv.org/abs/2604.11849). A related processing-priority study is [AI radiology worklist prioritization](https://arxiv.org/abs/2001.08625). These differ in task and system boundary; we do not claim the surrounding field is unexplored.

Falsifiable hypotheses:

- H1: importance improves early task value over FIFO.
- H2: transmitter-KB novelty adds distinct-evidence coverage beyond importance alone.
- H3: context makes novelty more valid than visual distance alone.
- H4: neural RankNet improves independent preferences beyond linear ranking, the tuned simple baseline and LambdaMART.
- H5: the eight-scalar header retains sufficient information; richer evidence is justified only by demonstrable failures.
- H6: prefix gains survive actual reconstruction and do not hide unacceptable important-image delays.

The learned policy receives all eight fields, with one-feature removal and correlated-feature ablations. The role of each field is a testable claim.

A null H2 is possible: local novelty may be too weakly related to what the receiver needs. Report that result rather than silently changing to a receiver-side novelty system.

## 11. Build sequence and stop rules

| Milestone | Build | Required evidence | Revise if |
|---|---|---|---|
| M1: boundary and contract | V3 parser, complete-message queue, dummy decoder, pure policies. | Deterministic round-trip and arithmetic checks; no hidden inputs. | Policy can see images, labels or future outcomes. |
| M2: parameter validation | Shared extractors, eight-field validity, official splits and metadata audit. | Per-field accuracy/provenance and importance-only controls. | Values are invented, test-derived or mislabeled probabilities. |
| M3: transmitter KB, mandatory | Causal local histories, cosine baseline, context retrieval and reviewed groups. | Novelty validity plus distinct-fact coverage beyond importance-only. | Only identical-file deduplication works, or sender novelty harms the task. |
| M4: independent ranking | Human preference pilot, neural RankNet and required competitors. | Held-out comparisons, feature ablations and LambdaMART benchmark. | Labels are inconsistent or header aliasing dominates. |
| M5: real decoder | Fixed received latent replay. | Measured timing and independently verified useful evidence. | Improvement exists only in source-label sorting. |
| M6: streaming and replication | Same headers, finite arrivals, external profile. | Tail waits, multiple-source stress and domain limitations. | Gains rely on cherry-picked traces or invented identities. |

Knowledge-base novelty belongs in M3 and in the main research claim. A full knowledge graph, MoE, RL agent or new encoder is not required to complete it.

Include a no-backlog control: when exactly one message is ready, all work-conserving policies choose it. This scheduler matters only when a decoding backlog creates a choice.

## 12. Deliverables, figures and provenance

Produce:

1. Boundary diagram and serialized header layout.
2. Visual examples separating repeat, new view, new fact and new incident.
3. Transmitter KB query matches with provenance and false matches.
4. Importance nDCG alongside distinct-fact coverage curves.
5. Local/shared-KB and stale-memory comparisons.
6. Human agreement and cosine-novelty validity.
7. Real-decoder time-to-useful-evidence and failure examples.
8. Header sufficiency, runtime and important-image tail waits.

Label figures as illustrative, label-derived simulation, human annotation or actual checkpoint execution. The original poster's projected spectral-efficiency gains are not measured outcomes of this revised system.

Inspected source: HEAD 032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0 plus the pre-existing dirty working tree. No runtime source, model, dataset or historical result was changed for these design documents. The finalized extraction guide is the authority for the eight parameter definitions.

Primary references supporting background, not validating our proposed method:

- [LADI v2, CVPR EarthVision 2025](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Scheele_LADI_v2_Multi-label_Dataset_and_Classifiers_for_Low-Altitude_Disaster_Imagery_CVPRW_2025_paper.html), [official overview](https://github.com/LADI-Dataset/ladi-overview).
- [CrisisMMD dataset paper](https://arxiv.org/abs/1805.00713), [official dataset card](https://huggingface.co/datasets/QCRI/CrisisMMD/blob/main/README.md), for later external replication.
- [RankNet, 2005](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/).
- [Graded ranking evaluation, 2002](https://doi.org/10.1145/582415.582418).
- [Probability calibration, 2017](https://arxiv.org/abs/1706.04599).
- [Smith's weighted-completion scheduling, 1956](https://doi.org/10.1002/nav.3800030106).
- [AI worklist prioritization, 2020 preprint](https://arxiv.org/abs/2001.08625).
- [CAPS-TDPC, 2026 preprint](https://arxiv.org/abs/2604.11849).
