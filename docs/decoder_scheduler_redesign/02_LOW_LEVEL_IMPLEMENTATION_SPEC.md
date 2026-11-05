# Receiver semantic scheduler: implementation and evaluation specification

**Date:** 22 October 2026  
**Status:** Design specification. Proposed modules, layouts, algorithms and acceptance checks are not implemented runtime behavior. No trained-model or performance result is claimed.  
**Final methodological authority:** [Protocol 04](04_FINAL_BENCHMARK_EVALUATION_AND_LOSS.md) supersedes this document's earlier generic main-loss, annotation-budget, queue-bank and primary-endpoint choices. Header and parameter definitions remain as specified here and in guide 03.  
**Companions:** [High-level research design](01_HIGH_LEVEL_RESEARCH_DESIGN.md) · [Finalized parameter extraction guide](03_FINAL_PARAMETER_EXTRACTION_GUIDE.md)

## 1. Scope and authoritative choices

The transport has already delivered complete **[ParameterHeader][Payload]** messages. A payload is one complete latent for a fixed image decoder. One nonpreemptive decoder processes one selected latent at a time.

The scheduler's only action is to return the ID of one ready message. It does not grant resources, schedule uplink transmission, partition latents, retransmit, inspect pending tensors or redesign the encoder.

The finalized inputs are:

1. Context relevance R.
2. Scarcity S.
3. Reliability Q.
4. Task importance I.
5. Freshness F.
6. Spatial prominence P.
7. Scene complexity C.
8. Transmitter-KB novelty N.

**Novelty is computed at the transmitter before sending.** It is not recomputed against receiver decode history. Scarcity also uses a declared transmitter/reference population. Freshness may be updated deterministically from a trusted capture timestamp.

All fields remain separate. We do not multiply them into a mandatory Semantic Priority Index. The main candidate is a small neural RankNet-style scorer, with linear ranking, pointwise models and LambdaMART as required competitors.

Historical eleven-family/seven-family specs and V2 serialization remain unchanged. This design is a new proposed receiver path, not an enacted migration.

## 2. Information boundary and ownership

| Information | Owner | Policy access |
|---|---|---|
| Parsed eight scores and availability mask | SemParam/parser | Yes |
| Profile meaning, task and freshness rule | Application/profile registry | Yes |
| Message ID and ready timestamp | INFRA | Identity, eligibility and tie breaking |
| Capture timestamp with valid clock status | Envelope/header/INFRA | Deterministic freshness only |
| Raw header bytes | INFRA/parser | Parser validates; policy gets parsed view |
| KB snapshot version and source/session ID | Provenance/runtime | Not numeric learned features in the primary study |
| Payload, source image, embedding, detections and masks | Transmitter or INFRA/EncDec | No preselection access |
| Hidden grade, fact set, event/asset truth, preferences | Evaluator | No |
| Future arrivals/outcomes/runtime | Evaluator/simulator | No |
| Decoded reconstruction | Output consumer/evaluator | Not an input to primary ranking |
| Payload bytes or nominal link rate | Diagnostic logging | Not a primary semantic-ranking feature |

Only complete messages are eligible. Define a_i as the time the entire header and payload are available.

Preserve canonical owners:

- **SemParam:** extraction definitions, transmitter KB interfaces, headers, profile registry and offline fitting.
- **Scheduler:** pure policies over immutable candidates and exported model coefficients.
- **INFRA:** clock, lifecycle, queue mutation, source identities, replay, outcome persistence.
- **EncDec:** unchanged encoder/decoder and a thin received-latent decoder adapter.

Scheduler imports no image codec or neural framework. The small model's inference can use exported weights and a pure-Python forward pass. Offline training is a separate SemParam/INFRA activity.

## 3. Parameter producer contract

Full formulas, examples and component references are in the [extraction guide](03_FINAL_PARAMETER_EXTRACTION_GUIDE.md). This section fixes runtime meaning.

| Index/bit | Name | Registered initial meaning | Special requirement |
|---:|---|---|---|
| 0 | context_relevance | Normalized image/task cosine, or a separately identified calibrated relevance producer | Similarity is not a probability |
| 1 | scarcity | Smoothed prior semantic-concept rarity | Causal reference counts, declared sampling population |
| 2 | reliability | Calibrated probability of correctness for the specified task-determining claim | Claim and calibration target must be explicit |
| 3 | task_importance | Normalized expected task-review gain | Fixed task rubric; independent labels |
| 4 | freshness | exp(-capture_age/tau) | Trusted capture time, units and clock assumptions |
| 5 | spatial_prominence | Maximum supported task-relevant image-area fraction | Box/mask basis identified; not physical distance |
| 6 | scene_complexity | Fixed mixture of supported count saturation and class diversity | Vocabulary and detector settings frozen |
| 7 | novelty | Transmitter prior-KB cosine-distance proxy initially | Before insertion; local/reference scope and KB version |

A profile maps these names to score kinds, model/vocabulary hashes, task gain rubric, calibration artifacts, supported domains, timing rule, KB namespace and normalization constants. Two producers with different meanings need distinct profile entries.

Use one compatible task/producer profile per primary queue. Equal numeric ranges do not make different tasks' gains comparable. Cross-task conditioning and unseen-task generalization require a separate design and test.

Every scalar is bounded to [0,1] for the wire interface. This common range does not make the meanings interchangeable. Preserve raw scores and units in an offline audit record.

### Initial importance profile

Use building_damage_review_v1 with LADI v2's **v2a** configuration:

- y=0: no building label.
- y=1: building, no affected-or-greater evidence.
- y=2: affected-or-greater below minor-or-greater.
- y=3: minor-or-greater.

Proposed gains g=(0,0,1,3), g_max=3, and I=(p_2+3p_3)/3.

Use buildings_any, buildings_affected_or_greater and buildings_minor_or_greater only after schema verification. Missing or contradictory annotations are flagged. Evidence state 3 has highest gain here; legacy scheduler priority class 0 means highest priority in a different namespace.

For nested predictor outputs b,a,m, require 1>=b>=a>=m>=0. If needed, fit/freeze a documented projection rule, then p=(1-b,b-a,a-m,m). Projection constructs a distribution; it does not establish calibration.

### Unknown is not zero

Examples:

- Valid detector, no relevant target: P=0.
- Detector failed: P unavailable.
- Empty declared KB: N=1 by explicit first-observation convention.
- Unavailable/incompatible KB: N unavailable.
- Missing capture timestamp: F unavailable, not “fresh on arrival.”
- Confidence not calibrated: Q is unavailable under a probability-profile contract, or explicitly a different proxy profile.

Record unavailable reasons in producer/run audit artifacts.

## 4. Transmitter knowledge-base lifecycle

### Stored state

For each source/task scope maintain prior:

- normalized embeddings and extractor identity;
- supported concept-presence counts;
- observation IDs and capture/observation order;
- available incident/site/asset metadata;
- snapshot version and deterministic retention rule;
- optional independently supported fact/relationship annotations in enriched experiments.

A vector store plus typed metadata is an initial operational KB. A delivery log alone is not a full semantic KB; a vector database alone does not establish logical understanding.

### Causal order

~~~text
receive a new source observation at the transmitter
  freeze prior KB snapshot K_TX(version)
  extract supported embedding, detections and task evidence
  compute novelty against the prior snapshot
  compute scarcity using prior concept counts
  produce all eight values with validity and provenance
  serialize header bound to the unchanged encoder's latent
  record the observation in transmitter memory using the declared rule
  increment the KB snapshot version
~~~

Do not insert the current embedding before its novelty query. Do not query future observations. Do not condition transmitter observation-memory updates on receiver scheduling decisions.

This memory records what the transmitter has observed. It is not a guarantee of what the receiver has decoded. If a separate experiment uses “previously sent” rather than “previously observed” memory, name it separately and declare send/failure semantics.

### Cosine baseline and context

For unit embeddings:

\[
N_i^{cos}=(1-\max_{j\in K_{TX}^{<i}}e_i^\top e_j)/2.
\]

It is a visual-distance proxy. It does not prove a new incident or additional task evidence.

Use verified incident/site metadata when available for context-aware retrieval. Do not infer incident identity merely from a disaster category. Different views of one site can be redundant or complementary; similar images can be from different events.

Store top matches and similarity scores for review. A later evidence-comparison model must be validated against reviewed novel/redundant facts, not only nearest-neighbor appearance.

### Scarcity reference

Each supported concept contributes at most once per observation to n_k. With M prior observations:

\[
\hat p_k=(n_k+\alpha)/(M+2\alpha).
\]

Use fixed smoothing, probability floor and aggregation from the extraction guide. Record whether sampling counts raw observations, deduplicated episodes or a frozen background set. Do not describe local observation frequency as global disaster frequency.

### Multi-source and memory limitations

Source/session identity disambiguates KB versions. A version number is not a priority feature. Empty, evicted, stale and incompatible memories are distinct statuses.

The primary system uses local transmitter memory. Shared memory is a separately specified comparison with explicit communication/synchronization cost. No receiver novelty refresh or hidden feedback is assumed.

Precompute header inputs from one policy-independent source-observation trace. All scheduling policies see exactly those bytes. A different policy cannot change the transmitter's prior history retroactively.

## 5. Proposed V3 wire format: 32 bytes

Keep the existing 24-byte V2 implementation unchanged.

Network byte order; proposed struct:

~~~text
>BBHIIQ8BI
~~~

| Offset | Bytes | Field | Meaning |
|---:|---:|---|---|
| 0 | 1 | version | 3 |
| 1 | 1 | valid_mask | Bits 0..7 match R,S,Q,I,F,P,C,N |
| 2 | 2 | profile_id | Registered task/producer semantics |
| 4 | 4 | message_token | Bound to source/session envelope |
| 8 | 4 | tx_kb_version | Prior KB snapshot for N/S provenance |
| 12 | 8 | captured_at_ms | UTC epoch milliseconds, only trusted when validated |
| 20 | 8 | scores_u8[8] | R,S,Q,I,F,P,C,N |
| 28 | 4 | crc32 | CRC32 of preceding 28 bytes |
| **Total** | **32** | | **Eight semantic scalar bytes** |

The profile and source/session audit record identify KB namespace, extractor and snapshot digest. tx_kb_version alone is not globally unique. Do not feed identifiers, CRC or versions as continuous features.

### Quantization

For an available x in [0,1]:

\[
q=\lfloor255x+0.5\rfloor,\quad \hat x=q/255.
\]

Validate bounds before packing. Absolute rounding error is at most 1/(2*255). Near ties can invert. Compare float32, uint8 and a separately versioned uint16 diagnostic; do not assume order is unchanged.

An unavailable field has its bit cleared and byte zero. This byte is not interpreted as measured zero. Validate profile, length, CRC and field availability before constructing a candidate.

### Missing or corrupt input

A corrupt header supplies no trusted semantic values; the trusted queue/task configuration selects the permitted fallback profile. Do not silently discard an otherwise eligible latent or call it unimportant.

For learned models, impute each unavailable field using a frozen training-only prior/median and supply all eight validity bits. Fit imputation on the correct training partition. If a field has no usable training values, explicitly define its neutral fallback and report it unvalidated; an all-feature empirical claim is then unsupported.

Construct the model input in the registered order: eight imputed scores followed by eight 0/1 validity indicators. Validate tau>0, n_ref>0, K>=2, alpha>0 and 0<p_min<1 in applicable profiles before a run starts.

For importance-only or formula comparators, use the same declared importance prior when I is unavailable; document how missing N is handled. Do not compare raw learned scores with a fallback taken from a different score scale.

Malformed timing beyond allowed clock skew makes F unavailable. An unknown profile is not guessed from an image name or test label.

CRC32 is not authentication and cannot establish semantic truth or zero undetected-error probability.

### Freshness update

The transmitted F is a producer snapshot. In dynamic mode, INFRA recomputes:

\[
F_i(t)=\exp[-\max(0,t-t_i^{capture})/\tau_T]
\]

only when timestamp and clock status are valid. Simulation time is mapped to a declared UTC origin. Large future timestamps are invalid; small permitted skew is logged.

Static ranking experiments freeze F at the common decision epoch. Freshness-updated dispatch is a separate mode, even for static arrivals. Novelty, scarcity and other image-derived scores remain fixed.

## 6. Planned code contracts

| Proposed location | Responsibility |
|---|---|
| SemParam/features/receiver_parameters.py | Eight-field definitions and evidence aggregation |
| SemParam/history/transmitter.py | Causal local KB snapshots/counts; no receiver-history substitution |
| SemParam/headers/decoder_priority.py | V3 encoding/parsing/validation |
| SemParam/receiver_profiles.py | Score meanings, task, calibration and timing registry |
| SemParam/learning/receiver_ranker.py | Offline fitting/export; not imported by Scheduler |
| Scheduler/receiver/policies.py | Pure selection from immutable candidates |
| Scheduler/receiver/scorers.py | Exported linear/MLP/tree inference without neural-framework imports |
| INFRA/receiver_decode/contracts.py | Public and private message views |
| INFRA/receiver_decode/simulation.py | Lifecycle, time, arrivals, freshness and dispatch |
| INFRA/receiver_decode/evaluation.py | Hidden-label and fact-coverage metrics |
| INFRA/receiver_decode/experiment.py | Manifests, common traces and provenance |
| EncDec/integration/receiver_decode.py | Fixed decoder on selected already-received latent |

These are design locations, not files implemented by this document.

~~~python
@dataclass(frozen=True)
class ReceivedMessage:                  # INFRA-private
    message_id: str
    header_bytes: bytes
    payload_ref: str
    complete_arrival: float
    source_session: str
    codec_profile: str

@dataclass(frozen=True)
class SchedulerCandidate:               # policy-visible
    message_id: str
    scores: tuple[float, ...]            # eight imputed/valid values
    valid_mask: int
    complete_arrival: float
    queue_sequence: int
    header_status: str

@dataclass(frozen=True)
class ParameterProfile:
    profile_id: int
    ordered_names: tuple[str, ...]
    imputation_values: tuple[float, ...]
    freshness_tau_seconds: float
    profile_hash: str

@dataclass(frozen=True)
class DecodeOutcome:                    # evaluator-only
    message_id: str
    start: float
    finish: float
    decoder_completed: bool
    reconstruction_valid: bool
    task_retention: float | None
    evaluator_profile: str | None
~~~

Implementation must enforce vector length, finite values, legal profile and explicit enums. The clock/timestamp/provenance fields remain INFRA metadata unless specifically exposed as a declared policy input.

Never put images, payload references, embeddings, hidden grades or true fact IDs into SchedulerCandidate.

## 7. Scheduler and model architecture

### Main candidate

Input per item: eight scores plus eight binary validity indicators. Initial architecture:

~~~text
16 -> Linear(32) -> ReLU -> Linear(16) -> ReLU -> Linear(1)
~~~

Trainable parameter count:

\[
16(32)+32+32(16)+16+16(1)+1=1089.
\]

Use the same scorer for each ready item. Select maximum score, then complete-arrival time, queue sequence and identity only as a final deterministic tie breaker. Identity is never a learned feature.

A neural score is a relative ranking value, not automatically a probability or calibrated utility. Start with modest regularization and grouped model selection; a small annotation set may favor linear models.

Runtime MLP inference is a pure deterministic matrix-vector/ReLU calculation over exported validated weights. Offline training frameworks stay outside Scheduler. A tree export can likewise support a pure tree evaluator.

### Dispatch

~~~text
on complete message arrival:
  validate and parse the actual header
  store private payload and public candidate separately

when the decoder becomes idle:
  collect eligible immutable candidates
  update only permitted freshness values for this decision time
  apply frozen imputation and score model
  select exactly one candidate
  INFRA removes it from the ready queue
  INFRA gives the unchanged decoder that private payload
  log completion/failure and actual duration
  repeat
~~~

No preemption, payload inspection, hidden class gate, radio service, automatic retries or low-score dropping.

If scores are frozen, static sorting costs O(N log N) after O(NP) model evaluation for P fixed model work. With freshness-dependent scores, a direct scan costs O(NP) per dispatch. For small queues, measure this before adding heap-refresh machinery.

All finite workloads drain. A reporting horizon censors observation, not permission to remove inconvenient jobs.

### Information limit

A per-item scorer with fixed N cannot determine untransmitted pairwise overlap. Two distinct messages can have identical eight-field headers but different meanings. Queue size flexibility is not set-level understanding.

Keep a scalar-header aliasing audit. Rich embeddings or controller context from the algorithm-comparison table are not silently added to this finalized primary header. A later expanded-input study requires its own declared budget.

## 8. Learning targets and loss

### Independent targets

- Public y_i: visible-evidence class.
- Public g(y_i): declared review-importance proxy.
- Human z_i: task-review grade with a written rubric.
- Human t_ij: preference/tie between two candidate images under a fixed prior-information context.
- Hidden F_i: reviewed task facts in the source image.
- Hidden B_TX,i: prior transmitter evidence used for independent novelty annotation.
- Predicted x_i: actual serialized eight-field input.

Do not train to imitate the old heuristic, a product of the eight fields or its own predicted importance and call agreement semantic understanding.

### Pointwise comparator

For a common independent grade z:

\[
L_{point}=\frac1n\sum_i(q_\theta(x_i)-z_i)^2+\rho\|\theta\|_2^2.
\]

Public-proxy and human-grade training are separately labeled runs.

### Pairwise RankNet loss

The formula below is the ordinary-pairwise comparator. The finalized main run uses protocol 04's exact coverage-swap targets, weights and query normalization, retaining the same logistic surrogate.

For s_i=f_theta(x_i), delta=s_i-s_j and t in {0,0.5,1}:

\[
L_{pair}=\frac1Z\sum_{(i,j)}w_{ij}
[\operatorname{softplus}(\delta_{ij})-t_{ij}\delta_{ij}]
+\rho\|\theta\|_2^2.
\]

Use stable softplus, primary w=1, Z=sum w. The derivative with respect to delta is sigmoid(delta)-t.

An explicit judged tie is not the same as reviewer disagreement. Preserve individual votes, assessability and rationales. Regularization/early stopping are chosen on validation.

Use identical legal inputs for linear pairwise and neural pairwise models. A shared bias cancels in pairwise score differences; omit/fix it in a linear model or acknowledge that regularization anchors it.

### Pair construction and context

Split images/events/duplicate groups before constructing pairs. Do not randomly split pairs while their images cross training and test. Cap participation per image and include ties, difficult critical/benign confusions, novelty/importance conflicts and small-object cases.

Reviewers see images, task and a fixed prior-evidence description, not headers or policy outputs. For transmitter-novelty annotations, show the actual earlier transmitter context. For decode-priority judgments, use the declared receiver-initial context.

The primary scorer cannot adapt to hidden dynamic receiver beliefs. Do not train preferences on changing hidden receiver states and pretend those states are observable.

### LambdaMART competitor

Use the same image/query splits and eight fields/masks. It requires query groups and graded labels; collect those alongside pair preferences rather than pretending arbitrary pairs are list grades.

Declare relevance encoding and gain mapping. If an implementation defaults to exponential label gains while our metric uses direct gains, configure or report that difference. Tune under a comparable validation budget.

LambdaMART is a strong tabular competitor, not an assumed winner. The neural model is the proposed method; selection follows held-out evidence.

For example, LightGBM exposes configurable label gains for its ranking objective. Pin the implementation version and explicitly register the gain list rather than inheriting incompatible defaults. [Official ranking parameter documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#label_gain)

### Why RL remains deferred

The initial task has supervised labels and a clear dispatch boundary. Pairwise/list ranking can be tested without reward engineering or exploration.

For static equal-duration additive weights, descending known gain is already optimal for weighted completion. For unique facts, utility depends on overlap and this proof no longer applies. RL cannot repair missing overlap information absent from the header.

A sequential extension requires an observable state, validated outcome signals and strong greedy/lookahead baselines. Do not introduce DQN/PPO to replace unresolved parameter validation.

## 9. Dataset, metadata and annotation protocol

### Public importance control

Pin LADI v2 revision and **v2a** schema. Published train/validation/test counts are 8,030/892/1,041. The later-year official test provides a useful domain-shift check.

Use the train-only **reference** predictor for official-split experiments. Deployment/main variants trained on all splits would leak the official test. Pin checkpoint and supported ontology.

For a small pilot with a fixed reference predictor, split official validation into group-aware rank-training A, calibration B and model-selection C. Freeze before official testing. The authors' train/validation data share incident distributions; do not claim the entire official protocol is event-disjoint.

If more training data are needed, use grouped out-of-fold predictions rather than in-sample predictions from a supervised extractor. Never tune rankers, calibration or scarcity reference statistics on future test labels.

### Real metadata gaps

LADI class labels do not supply all eight targets. Verify timestamp, site correspondence, boxes and capture ordering from the actual release; do not assume they exist.

Options:

- Keep unsupported fields unavailable for the public-label control.
- Use a metadata-rich held-out subset with verified provenance.
- Add independently reviewed boxes/counts, facts and novelty judgments.
- Use synthetic capture times only in labeled sensitivity experiments.

An experiment with absent F or uncalibrated Q is not an empirical validation of all eight parameters. Finalized parameter definitions do not justify fabricated values.

### Core human/novelty pilot

Plan 120 images with two independent reviewers first. A later budget around 600 images is a starting allocation, not a sufficient-sample guarantee. Determine size from event/group coverage, disagreement and uncertainty.

Collect:

- task relevance and importance grade;
- pair preferences and explicit ties;
- supported concepts, count/box audit subset;
- fact IDs at a frozen granularity;
- repeat/new-view/new-fact/changed-condition judgments;
- incident/site identity only where assessable;
- reliability claim truth and assessability.

Use qualified review for domain-level claims; otherwise describe a non-expert feasibility pilot. Automated VLM labels alone are not independent expert truth.

### Causal test histories and leakage

Within a held-out episode, earlier test observations may populate the transmitter runtime KB before later observations. That is causal test-time memory, not fitting on test labels. Log the order and exclude future items.

Train/test duplication is leakage. Repeated evidence **within** the held-out scheduling workload is intentional and essential to the novelty benchmark. Do not deduplicate away all semantic redundancy.

A public event ID is not a building/fact ID. Same class does not mean same fact. Exact-byte repeats are a sanity test, not the sole evidence for semantic novelty.

### External replication

CrisisMMD can supply a separate damage-review profile, with its own labels and missing-label handling. Its domain, task and terms differ. A separately trained predictor is external replication, not zero-shot transfer.

The finalized parameter definitions carry over only where their underlying observations/metadata can be obtained and validated.

### Physically separate manifests

~~~text
received_messages.csv -- runtime only
  message_id, source_session, complete_arrival, header_path,
  received_payload_ref, codec_profile

transmitter_audit.jsonl -- producer/research provenance, not policy input
  observation_id, source_session, prior_kb_version, prior_kb_digest,
  extractor_hashes, raw_scores, valid_reasons, top_matches,
  concept_counts_reference, capture_time_provenance

evaluation_labels.jsonl -- evaluator only
  message_id, split, event_or_source_group, independent_grade,
  fact_ids, fact_weights_profile, preferences, novelty_review,
  annotation_source, assessable

decode_evidence.jsonl -- evaluator only
  message_id, payload_hash, decoder_hash, completion_status,
  measured_decode_ms, accepted_retained_facts, evidence_assessable,
  evaluator_profile, fixed_seed
~~~

The runtime loader never imports the evaluation-label table. Ground-truth fact/incident identities are not copied into the transmitter header; only legitimately available source metadata may guide its KB retrieval.


## 10. Experiments and algorithm comparisons

### Static primary protocol

Use N=32 complete messages, all ready at time zero, K=10 and one equal decode opportunity per message. Secondary queue sizes 16 and 64 test scale. Freeze static feature values at the common decision epoch.

Use a fixed queue bank, for example 256 draws after locking the held-out cohort. Repeated draws sharing images/events are not independent scientific samples.

Prefer within-event queues where verified event identity and enough images exist. Declare exclusions and smaller-event variants. Mixed-event queues are a separate workload. Use event/source groups for uncertainty and preserve causal transmitter prehistory even when decoder arrival order is shuffled.

Do not invent building IDs from event IDs. Distinct-view redundancy requires reviewed correspondence.

### Algorithm table and roles

| Method | Role and legal interpretation |
|---|---|
| FIFO | Essential deterministic reference |
| Seeded random | Sanity control |
| Round robin | Supporting source-fairness baseline; source grouping comes from declared envelope metadata |
| SJF | Diagnostic only with independently profiled/estimated decoder cost; never size divided by radio rate |
| EDF | Deadline extension only when real external deadlines exist |
| Existing fixed formula | Legacy baseline; identify extra inputs and arbitrary floors |
| Importance only | Strong semantic control using I |
| Novelty only | Diagnostic for novel-but-irrelevant errors |
| I*N and bounded I*(1+lambda*N) | Transparent novelty interactions; lambda tuned on validation, not test |
| Priority classifier | Required learned baseline on the same eight fields/masks |
| Pointwise regression | Required continuous-grade baseline |
| Linear pairwise | Strong low-capacity ranking baseline |
| Neural RankNet-style | Main candidate; eight scalars and availability, not undeclared embeddings |
| LambdaMART | Strong tabular ranker competitor with graded query supervision |
| Listwise neural | Deferred; requires list/set objective and annotation protocol |
| Contextual bandit | Deferred; selected-only feedback/exploration is not the initial supervised problem |
| DQN/Double DQN | Sequential-objective extension only after state/outcome validation |
| PPO/actor-critic | Deferred policy-learning extension, not required for initial ordering |

All deployable primary methods use the same legal scalar-header information. Extra-context baselines are labeled supporting/diagnostic, not used to imply that the same-input neural model had that context.

The phrase “strong competitor” is a design assessment; actual performance ranking follows experiments.

### Novelty workloads

Include:

- No overlap as a control.
- Exact repeats, as an arithmetic/identity sanity check.
- Different photographs repeating the same fact.
- Different views revealing additional facts.
- Changed conditions at one site.
- Different incidents with similar appearance.
- Rare but irrelevant content.
- Small critical targets and large benign/cluttered scenes.
- Stale/local/shared-transmitter memories and cross-source duplicates.

Constructed challenge mixtures are labeled as such. Report natural prevalence when it is known.

### Streaming and real decoder replay

Streaming uses identical finite arrival traces. At equal timestamps process completion, then all arrivals, then dispatch. No future arrivals are visible; no artificial idling. Recompute permitted freshness and drain after the last arrival.

A common maximum-wait guard is a separate comparison. It can reduce starvation but cannot guarantee deadlines under overload.

Freeze the received payload, channel result and image-specific seed before policy execution. Never apply fresh AWGN after selection. Warm up on separate non-test inputs; keep checkpoint, device, precision, image shape and batch size=1 fixed.

Measured median per-image duration supports trace replay. Actual wall-clock execution is separate evidence with variability. A finite output tensor is not semantic retention.

## 11. Evaluation equations

Use independent hidden labels. Do not evaluate by agreement with the scheduler's own score.

### 11.1 Coverage@K: supporting endpoint for the finalized protocol

Protocol 04's mean of this coverage over prefixes 1..K is the finalized primary metric. This subsection defines its constituent coverage curve.

Let F_i be the reviewed task facts supported by source image i. Each fact f has a preregistered nonnegative weight w_f. Define a fixed receiver-initial known set K_RX,0; the core experiment uses the empty set.

\[
F_{offered}=(\bigcup_i F_i)\setminus K_{RX,0},
\quad
Coverage@K=
\frac{\sum_{f\in(\cup_{r\le K}F_{\pi(r)})\setminus K_{RX,0}}w_f}
{\sum_{f\in F_{offered}}w_f}.
\]

This is **source-evidence potential coverage** when source facts are used. It is not verified reconstruction utility.

For actual decoded evidence replace F_i by independently accepted facts retained in that output, counting completion/acceptance only. Unknown reconstruction evidence stays unknown; do not set it equal to source facts.

The denominator is all offered useful facts, not an asserted achievable optimum at K. A perfect K-prefix can have coverage below 1 when K cannot cover everything.

Freeze fact granularity: for example incident/site/asset/property/state/episode. Do not count both a parent claim and its more specific child as two independent facts unless the rubric explicitly values both. Different angles are not automatically different facts. Corroboration can have human-review value even without a new fact; binary coverage alone does not measure that value.

With zero total fact weight, report undefined and counts. Missing correspondence is unassessable, not “no facts.”

### 11.2 Importance nDCG and related diagnostics

For independent item gain g_i:

\[
DCG@K=\sum_{r=1}^K\frac{g_{\pi(r)}}{\log_2(r+1)},\quad
nDCG@K=DCG@K/IDCG@K.
\]

Use direct declared gains; do not exponentiate again unless explicitly defined. IDCG=0 is undefined. Label this importance ordering, not novelty reduction.

Also report:

- independent top-priority recall@K;
- human pairwise agreement, with ties/disagreement separate;
- severity-weighted inversions;
- per-grade wait median/p95/max;
- importance-weighted completion sum g_i(C_i-a_i)/sum g_i for complete workloads.

These item-additive measures can reward repeated evidence. They complement rather than replace distinct-fact coverage.

### 11.3 Time to first usable fact and coverage AUC

Let a_f be the earliest complete arrival of an offered message supporting fact f. Let C_f be the first completion whose accepted reconstruction supports f. If no such output exists, C_f=infinity.

\[
U(t)=\sum_{f\in F_{offered}}w_f\mathbf1[C_f\le t].
\]

At fixed horizon T:

\[
AUC_T=
\frac{\int_0^T U(t)\,dt}
{\sum_{f:a_f<T}w_f(T-a_f)}.
\]

The denominator is policy-independent potential availability. Failed or unprocessed evidence remains unresolved. Only include a cohort with the necessary independently checked outcome evidence for a fully verified metric.

The corresponding unresolved-evidence area is:

\[
L_T=
\frac{\sum_{f:a_f<T}w_f[\min(C_f,T)-a_f]}
{\sum_{f:a_f<T}w_f}.
\]

For fully available verified facts, higher normalized AUC and lower unresolved time express earlier evidence. Do not discard failures, overflow or censoring to improve them. Report denominators and undefined cases.

Source-potential versions use source fact support at completion and are explicitly labeled simulated/proxy evidence.

### 11.4 Novelty validity

Compare N against independently judged additional evidence relative to the actual prior transmitter context.

Report AUROC/AUPRC for a frozen novel/redundant definition, repeat/new-view/new-fact confusion, incident-identity errors and nearest-match audit examples. Include prevalence.

Cosine distance is not a calibrated probability. Brier/NLL for novelty are meaningful only after fitting and testing a probability mapping. Thresholds are selected on validation, never from the final test examples.

Report how novelty validity differs from receiver prefix utility, especially under cross-source and stale-memory mismatch.

### 11.5 Per-parameter validity and cost

| Field | Required check |
|---|---|
| R | Task relevance labels, domain/task changes, small-object relevance |
| S | Causal count reproduction, rare false detections, reference-population bias |
| Q | Claim-correctness Brier/NLL, calibration diagram, class/domain slices |
| I | Independent grade/preference validity and calibration where applicable |
| F | Clock/timestamp provenance, missing-time counts, decay sensitivity |
| P | Box/mask area error; small-critical versus large-benign cases; no unsupported distance claims |
| C | Count/diversity audit, threshold/vocabulary sensitivity, clutter |
| N | Reviewed redundancy/additional-evidence validity and local/shared-memory errors |

Measure transmitter embedding, detector, task head, KB query/update and header cost separately. Measure receiver parsing, scoring, sorting and decoder time separately. Report peak memory only if measured. Published GPU latency is not our implementation cost.

## 12. Numerical verification examples

These are invented fixtures, not experimental results.

### Eight-feature arithmetic

The extraction guide's worked example has:
R=0.70, S≈0.655, Q=0.80 (hypothetical calibrated output),
I=0.75, F≈0.6065, P=0.04, C≈0.7245, N=0.02.

A raw task cosine of 0.40 becomes R=0.70; a nearest-KB cosine of 0.96 becomes N=0.02. The same embedding family supports different comparisons. Neither is automatically a probability.

### Why importance nDCG alone misses redundancy

Three images A,B,C all have independent item gain 3 and I=1:

- A supports fact X, weight 3.
- B repeats fact X, weight 3.
- C supports distinct fact Y, weight 3.

All arrive at time zero and each decode takes one unit. Assume accepted outputs retain the facts solely for this arithmetic fixture.

FIFO A,B,C:

- coverage@2=3/6=0.5.
- weighted mean first-fact completion=(3*1+3*3)/6=2.
- normalized AUC at T=3 is 6/18=1/3.

Order A,C,B:

- coverage@2=6/6=1.
- weighted mean first-fact completion=(3*1+3*2)/6=1.5.
- normalized AUC is 9/18=0.5.

Both have importance nDCG@3=1 and both finish three messages. The distinction is early unique evidence.

If illustrative transmitter N values are A=1, B=0.01 and C=0.40, a bounded bonus with lambda=1 produces A,C,B. This does not prove that real transmitter scores identify those facts reliably; that is the novelty experiment.

### Pairwise loss arithmetic

If a preferred item i has score 0 and item j has score 1, delta=-1 and t=1:

\[
L_{pair}=\operatorname{softplus}(-1)+1\approx1.31326,
\quad
\partial L/\partial\delta\approx-0.73106.
\]

Gradient descent increases the preferred score relative to the other. This verifies the loss direction; it does not establish meaningful supervision.

## 13. Header sufficiency and oracle diagnostics

Test:

- Importance-only and I/N-only subsets.
- All eight fields with masks.
- Leave-one-field-out and joint removal of correlated families.
- Float32 versus uint8 and separately versioned higher precision.
- A preregistered richer evidence/embedding diagnostic, never silently added.
- Metadata-missing, corrupt-header and domain-shift cases.

Group identical semantic bytes/masks while ignoring message IDs and versions. Inspect conflicting grades, preferences and fact sets. No ranker can distinguish identical observations without another legal input.

Theoretical references:

- True-grade sorting: privileged upper bound for static equal-cost additive importance only.
- Exact permutation enumeration for N<=8: evaluator-only reference for the specified fact-coverage/AUC objective and known outcomes.
- Greedy true marginal fact coverage for larger queues: privileged diagnostic, not automatically an exact upper bound.

Do not compare the neural scorer to a deliberately weak formula and ignore a tuned linear/tree competitor.

## 14. Statistical protocol and ablations

Predeclare mean weighted distinct-fact coverage over prefixes 1..10 on the reviewed novelty cohort as the main endpoint, following protocol 04. Public-label nDCG@10 is an importance proxy endpoint. Human-preference and actual-decoder results have explicit cohorts and provenance; report disagreement between endpoints.

Mandatory ablations:

- Remove each finalized parameter separately.
- Remove both KB fields N/S.
- R versus I; Q versus expected importance; P versus C.
- Cosine-only versus context-filtered novelty.
- Local versus explicitly shared transmitter memory.
- Memory capacity, eviction, time window and stale histories.
- Prior-count smoothing and observation/episode sampling for scarcity.
- Raw confidence proxy versus validated calibration.
- Missing/incorrect timestamps and freshness time constant.
- Detector thresholds, spatial box/mask basis and complexity normalization.
- Linear versus neural versus LambdaMART under comparable tuning budgets.
- Frozen-static versus freshness-updated and streaming modes.
- Cross-source duplicates and uncommon-but-irrelevant scenes.

Pair comparisons on the same jobs, headers, prehistories and decoder outcomes. Bootstrap independent event/source/evidence groups, not repeated queue seeds as if independent. Save the resampled units. With few groups, show group-wise results and wide intervals.

Use 95% intervals, paired effect sizes and natural/challenge results separately. Do not choose the best seed. Fit normalizers, calibration, rarity references and hyperparameters on training/validation roles only.

If no independent novelty correspondence exists, label results as importance-only/visual-proxy evidence and withhold a semantic-novelty performance claim.

## 15. Configuration proposal

~~~yaml
experiment:
  boundary: fully_received_latents
  task_profile: building_damage_review_v1
  mode: static_frozen_snapshot
  queue_size: 32
  primary_k: 10
  header_version: 3
  header_bytes: 32
  decoder: frozen_existing_decoder
  payload_inspection_by_policy: false
  receiver_novelty_refresh: false

parameters:
  order: [context_relevance, scarcity, reliability, task_importance,
          freshness, spatial_prominence, scene_complexity, novelty]
  transmit_separately: true
  validity_mask_bits: 8
  quantization: uint8_round_half_up
  missing_values: training_only_imputation_plus_mask

transmitter_knowledge:
  scope: local_source_task_observation_history
  novelty_query: prior_snapshot_before_insertion
  initial_method: nearest_cosine_distance
  scarcity_counts: supported_concept_presence_per_observation
  use_future_observations: false
  shared_knowledge: false

ranking:
  main_candidate: neural_pairwise
  objective: protocol_04_coverage_swap_weighted_pairwise_logistic
  mlp_layers: [16, 32, 16, 1]
  required_learned_competitors:
    [priority_classifier, pointwise, linear_pairwise, lambdamart]
  tie_break: [complete_arrival, queue_sequence]
  hidden_labels_in_input: false

service:
  primary_model: one_decode_opportunity_per_message
  batch_size: 1
  preemption: false
  retries: false
  drop_low_scores: false

evaluation:
  novelty_primary: mean_weighted_distinct_source_fact_coverage_prefixes_1_to_10
  importance_control: ndcg_at_10
  verified_utility_only_when_measured: true
  bootstrap_unit: event_source_or_evidence_group
  confidence_level: 0.95
~~~

These keys describe proposed functionality; they are not a runnable command for the current codebase. Actual-decoder and freshness-updated experiments use separate resolved profiles.

## 16. Acceptance gates

| Check | Required result |
|---|---|
| Header layout | Exactly 32 bytes, network order, correct offsets and CRC coverage |
| Field presence | Eight distinct validity bits; unavailable never interpreted as measured zero |
| Quantization | Bounds and round-trip error verified; near-tie inversions reported |
| Profile semantics | Score kinds, ontology, gains, time units and KB scope validated |
| Identity binding | Source/session/token pairing verified; duplicate tokens rejected |
| Transmitter causality | Novelty and scarcity use the prior snapshot, never current/future observations |
| Empty versus unavailable KB | Different documented behavior |
| Cross-source versions | Same numeric KB version from two sources is not the same snapshot |
| Novelty location | Changing receiver completion history alone does not alter received N |
| Freshness | Capture age used; unknown/future-invalid timestamps remain unavailable |
| Geometry | Box scaling/letterbox reversal checked; area never labeled metres |
| Complexity | Duplicate detections removed; count/diversity normalization reproducible |
| Reliability | Calibration target explicit; raw detector score not called probability without evidence |
| Observability | No payload, embedding, hidden grade, fact truth or future result in candidate |
| Hidden-label perturbation | Schedule unchanged when evaluator labels alone change |
| Identifier permutation | Semantic scores unchanged by renaming IDs |
| Model export | Exported linear/MLP/tree inference agrees with offline model within declared tolerance |
| Queue size | Shared scorer handles different candidate counts |
| Static/dynamic distinction | Frozen features versus allowed freshness updates tested separately |
| No backlog | All work-conserving policies select the sole ready item; no claimed scheduling gain |
| No useful facts | Coverage undefined with counts, not an artificial perfect score |
| Redundancy example | Repeated fact credited once; numerical example matches |
| At-most-once decode | Selected complete latent opened and decoded once |
| Failure/censoring | No invented semantic score; unresolved facts remain unresolved |
| Comparator fairness | Same traces, bytes, task and fixed outcomes |
| Actual checkpoint | Selected already-received latent decoded by the specified frozen checkpoint |
| Semantic outcome | Independent retention evidence before verified-utility claims |

Functional fixtures validate code behavior. They do not validate parameter accuracy, meaningful importance, a trained ranker or a journal contribution.

## 17. Artifacts, milestones and claims ledger

Save:

- resolved configuration, profile definitions and score order;
- model/vocabulary/calibration/decoder digests;
- exact data revision, metadata provenance and split roles;
- causal transmitter observation trace and KB snapshot digests;
- scarcity counts and novelty nearest matches;
- actual serialized headers and validity reasons;
- independent grades, fact correspondence and reviewer disagreement, evaluator-only;
- pair/query construction manifests and model-selection budget;
- exported models and export-equivalence report;
- full scheduling and completion trace;
- per-field validity, prefix/time metrics, tail waits and group uncertainty;
- actual hardware/runtime records and unmeasured-cost fields;
- illustrative/simulated/annotated/trained/checkpoint-measured claim labels.

Suggested build order:

1. Header/parser and deterministic queue fixtures.
2. Real extraction and per-field validity/missingness audit.
3. Mandatory causal transmitter KB and reviewed novelty subset.
4. Linear/pointwise baselines, neural RankNet and LambdaMART.
5. Fixed-latent decoder replay and retained-evidence evaluation.
6. Streaming, source mismatch and external replication.

Report a null result if a field or neural ranker adds no reliable value. Do not remove a finalized parameter silently; show its ablation and revise only through an explicit research decision.

Potential paper framing: **Transmitter-Knowledge-Assisted Semantic Prioritization of Received Image Latents**. The title and novelty claims depend on actual results and a focused prior-work audit.

The documents define what to implement and test. They do not claim that any proposed model has been trained, that all eight features are currently available, or that the project already has measured gains.


## 18. Current-source audit anchors

The following anchors describe the inspected implementation, not the proposed files.

| Current file | Anchor | Finding |
|---|---:|---|
| Scheduler/latent/policies.py | 24-30, 52 onward | Strict predicted-class gate followed by multiplicative heuristic |
| INFRA/latent/contracts.py | 19, 72, 142-162 | Current fields, median priority conversion and resource-rate service |
| INFRA/latent/simulation.py | 76, 93-100, 126 | Source-embedding novelty, selection then transport, confirmed-history update |
| INFRA/latent/experiment.py | 21 onward | Structured batch fields rather than new V3-only observations |
| INFRA/latent/evaluation.py | 38, 44 onward | Positive gain for all classes; task-quality evidence is optional |
| SemParam/headers/semantic.py | 53, 98 onward | Existing distinct 24-byte V2 header |
| SemParam/features/latent.py | 11 onward | Cosine novelty proxy and unavailable importance prediction |
| SemParam/features/scheduler.py | 191 onward | Annotation/proxy provenance for several legacy fields |
| EncDec/integration/latent_transport.py | 64-70 onward | Channel application after selection; finite output is not semantic validity |
| docs/FINAL_PARAMETER_SPECIFICATION.md | 7 onward | Eleven-family proposed contract |
| docs/PARAMETER_SPECIFICATION_V2.md | 17 onward | Seven-family proposed contract |

The repository's existing dirty work is preserved. The new documents do not enact a migration, retrain a model or replace policy code. Source fingerprints are appended below so a later implementation can detect drift.

## 19. Primary references and boundaries

- [LADI v2 paper](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/html/Scheele_LADI_v2_Multi-label_Dataset_and_Classifiers_for_Low-Altitude_Disaster_Imagery_CVPRW_2025_paper.html) and [dataset card](https://huggingface.co/datasets/MITLL/LADI-v2-dataset/blob/main/README.md): evidence data and split provenance, not universal urgency labels.
- [CrisisMMD official card](https://huggingface.co/datasets/QCRI/CrisisMMD/blob/main/README.md): independent domain and label availability.
- [RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/): established pairwise ranking.
- [Calibration](https://arxiv.org/abs/1706.04599): separates model score from calibrated probability.
- [Graded gain evaluation](https://doi.org/10.1145/582415.582418): nDCG lineage.
- [Smith's scheduling work](https://doi.org/10.1002/nav.3800030106): classical comparator and limit on algorithm novelty.
- [AI worklist prioritization](https://arxiv.org/abs/2001.08625): related processing-priority problem and false-negative/tail-wait limitations.
- [Semantics-aware differentiated allocation](https://arxiv.org/abs/2605.09095): related computation-allocation preprint, not the same whole-image benchmark.
- [CAPS-TDPC, 2026 preprint](https://arxiv.org/abs/2604.11849): related diffusion/channel scheduling and receiver compensation; different from a fixed already-received decoder worklist.

The proposed protocol, gain vector, annotation allocations, header layout, experimental matrix and acceptance gates are our design choices. They are not claimed as experimentally validated by these references.


## 20. Inspected source fingerprints

Recorded 22 October 2026. SHA-256 hashes identify the inspected working-tree files; HEAD alone does not describe existing uncommitted work.

| Source | SHA-256 |
|---|---|
| Scheduler/latent/policies.py | ca5b017f1fe671392cdce10d194ff9ab35407f777af2e8946b738fc7d7c0c751 |
| INFRA/latent/contracts.py | 7d27d83a1556b7cffaeaaa631a2ee80ee4392e854ffcd5858084f977ce8a1faa |
| INFRA/latent/simulation.py | c054a3cb3358b236363e5ef05be94725688dc343e0cac8c074e06299768ae7a3 |
| INFRA/latent/experiment.py | 6b8d7e87b3f48685e02790626ff3cb0f1d3d049dd230a6d47542e695e758baa9 |
| INFRA/latent/evaluation.py | 21949c2122c2009e7b9e5a479ecd94994f9ed6b6455af21393ec494673bcd4d0 |
| SemParam/headers/semantic.py | 2f8d9440fb432a140a8cffdf5c95cb2276e67609d7a927570fe432375030d614 |
| SemParam/features/latent.py | e2f16101477407a836c06e37628eb8e714e35e0642d203c22a4955247b8f832e |
| SemParam/features/scheduler.py | 625aa4fbf771061d82bde9f7353f8868660f92657e0dd707d1b97a1296e383ba |
| EncDec/integration/latent_transport.py | 52a1ba3437f7169ab164cc6b3e81891da3d27b78072959a6e96d08fc3281a5de |
| docs/FINAL_PARAMETER_SPECIFICATION.md | ea5471743b8e068f467c21e3a45a154aeb269059b9c2ebfb9ebc544fc4b27f2f |
| docs/PARAMETER_SPECIFICATION_V2.md | 5b9e7d74de540b9ccbed1fb8bec966c3a5381fdc07130d81cdf3465d79f95b88 |
