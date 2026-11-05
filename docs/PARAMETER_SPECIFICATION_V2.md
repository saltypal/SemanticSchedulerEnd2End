# Final Implementable Parameter Specification

**Project:** Semantic-Aware Latent Scheduler

**Date:** 28 September 2026

**Status:** final design contract for implementation; empirical validation is still required

**Scope:** scheduling complete image latents produced by SwinJSCC or its communication-codebook path

This specification replaces the earlier eleven-parameter proposal for the next implementation. The design uses seven parameter families. It keeps the information that can affect scheduling, removes overlapping or currently unmeasurable fields, and assigns an exact meaning to every transmitted value.

The scheduler receives the parameters separately. The extractor does not calculate one weighted Semantic Priority Index. A learned scheduler may discover interactions among the parameters only after its training objective and labels are defined.

## 1. Final parameter set

| ID | Parameter family | Header output | Type | Primary question |
|---|---|---|---|---|
| P1 | Semantic evidence | Eight fixed concept slots | Small structured vector | What setting, entities, relations and events are supported? |
| P2 | Context relevance | `context_similarity` | Float | How closely does the image match the active task? |
| P3 | Delivered-history novelty | `novelty_distance` | Float | How different is it from evidence already delivered for this task? |
| P4 | Visible extent | `extent_fraction` | Float | How much of the relevant visible target is affected? |
| P5 | Task impact | `impact_level` | Ordinal integer | If the supported condition is true, how consequential is it for this task? |
| P6 | Urgency | `urgency_level` | Ordinal integer | How rapidly does the usefulness of this evidence decay with delay? |
| P7 | Evidence sufficiency | `sufficiency_probability` | Calibrated probability | Is enough visual evidence present to answer the task reliably? |

The seven families are not seven arbitrary floating-point numbers. P1 contains categorical concept identities and their evidence strengths. P2–P7 are separate scalars with declared semantics. A validity mask accompanies all optional fields.

## 2. Core design decisions

### 2.1 Progressive understanding

The extraction process is:

```text
image + task
    -> broad scene hypotheses
    -> entity detection
    -> geometric or learned relations
    -> event hypotheses refined by the scene and entities
    -> task-, history- and policy-dependent parameters
    -> fixed semantic header + complete image latent
```

This is refinement, not an irreversible chain. A wrong scene prediction must not prevent detection of an unexpected entity. Scene, entity, relation and event candidates are inferred from shared image evidence and can correct one another.

A single still image usually cannot prove causality or temporal progression. For example, it may support “flames overlap a vehicle region,” but not “a building fire spread to the vehicle.” Observations, inferences and unknowns remain distinguishable in the audit record.

### 2.2 Values have different mathematical meanings

- A category ID names a concept. ID 30 is not more important than ID 2.
- A cosine is embedding alignment. A cosine of 0.40 is not 40% relevance.
- A fraction has a declared numerator and denominator.
- An ordinal level states an ordering. Impact 3 is not three times impact 1.
- A probability is used only after calibration against independently labelled data.
- Missing evidence is unavailable, not zero.

Float32 is the reference continuous representation. More decimal places do not create more semantic precision. Header quantisation is a later experiment.

## 3. Required task profile

Task-dependent parameters require a versioned `TaskProfile` selected before evaluation. It contains:

- `task_id` and `task_version`;
- a task description or visual question;
- allowed scene, entity, event and relation concepts;
- the applicable extent target, effect and measurement basis;
- the task-impact rubric;
- the urgency rubric and any trusted timing rules;
- the evidence-sufficiency annotation question;
- the eligible delivered-history scope;
- optionally, the target state, possible actions and loss function used for significance evaluation.

Dataset classes may be used to construct tasks, such as “find evidence of road obstruction.” The task must not be chosen from the ground-truth label of each evaluation image. The same image must be testable under different tasks.

## 4. P1 — Semantic evidence vector

### Meaning

P1 compactly describes what is visible and how the visible concepts relate. It combines four tightly connected kinds of evidence:

1. scene or setting;
2. entities;
3. spatial relations;
4. event or condition hypotheses.

These are combined into one parameter family because they form one scene-understanding process. Their identities are not collapsed into one opaque score.

### Fixed header representation

Use eight fixed slots:

| Slots | Family | Example concepts |
|---|---|---|
| 0–1 | Scene | residential exterior, highway, indoor room, landscape |
| 2–3 | Entity | person, building, vehicle, smoke |
| 4–5 | Event | ordinary traffic, visible fire, collision, road obstruction |
| 6–7 | Relation | vehicle on road, smoke overlaps building, person near affected region |

Each slot contains:

```text
concept_id: uint16
evidence_score: float32
score_kind: uint8 enum
valid: bool
```

`score_kind` is one of `cosine_similarity`, `detector_score`, `geometry_measure`, or `calibrated_probability`. The scheduler must know the kind because scores produced by different mechanisms are not directly comparable.

Two candidates are retained per family. This prevents one uncertain top prediction from destroying alternatives while keeping a fixed input size. The complete candidate list, boxes, masks and instance bindings stay in an audit artifact. The header records an omitted-candidate count for each family.

The two-slot budget is an initial engineering choice. It must be tested against four-slot and larger alternatives. It is retained only if important task evidence is rarely omitted.

### Concept catalogue

Concept definitions live in a versioned catalogue, not in Python conditionals. Each entry contains:

- numeric ID;
- family;
- canonical name and description;
- parent concept, where useful;
- positive examples and explicit exclusions;
- allowed task profiles;
- extraction method and model version;
- score meaning;
- support status.

The catalogue can begin with 30–60 reviewed concepts and grow without changing the header schema. Broad settings and events remain separate. “Residential exterior” is a scene; “building visibly burning” is an event. Combinations are represented by multiple concepts rather than creating a new class for every combination.

### Extraction

1. Compute one shared image embedding.
2. Compare it with cached scene and event prompt embeddings.
3. Run one configured entity detector.
4. Generate eligible geometric relations from boxes or masks.
5. Use a trained relation/event head only for predicates that geometry cannot establish.
6. Select the strongest supported candidates per family using deterministic ties.

The first implementation uses the repository's CLIP-style embedding interface and YOLO-World as the detector candidate. Places365 is the scene-classification comparator. Florence-2 is an alternative unified model to benchmark, not an additional mandatory dependency.

Segmentation is selective. Bounding boxes are enough for detection and rough geometry. A mask model runs only when containment, overlap or extent requires a boundary. Segment Anything produces masks but does not by itself supply trustworthy semantic labels.

### Spatial relations

Geometry can support predicates such as:

- left/right/above/below;
- overlaps;
- contains/inside;
- near in image coordinates;
- on-road when the road region is supported.

Every predicate defines its exact measurement and validation-selected threshold. For example:

```text
containment(A, B) = area(A intersect B) / area(A)
```

Image-plane proximity is not physical distance. Geometry alone cannot emit semantic predicates such as `trapped_in`, `injured`, `caused_by`, or `spreading_to`.

### Scheduler effect

P1 gives the policy grounded, interpretable evidence rather than a premature emergency class. A task-aware scheduler can learn that the combination of a building, visible fire and a nearby person matters differently from an isolated landscape or ordinary portrait.

### Paper support

- [Concept Bottleneck Models](https://proceedings.mlr.press/v119/koh20a.html): predictions through human-understandable intermediate concepts.
- [Label-Free Concept Bottleneck Models](https://arxiv.org/abs/2304.06129): reducing concept-annotation burden with pretrained representations.
- [Visual Genome](https://arxiv.org/abs/1602.07332): objects, attributes and relationship graphs.
- [YOLO-World](https://arxiv.org/abs/2401.17270): open-vocabulary object detection.
- [Places365](https://github.com/CSAILVision/places365): pretrained scene recognition and scene vocabulary.
- [Segment Anything](https://arxiv.org/abs/2304.02643): prompted mask generation.
- [Florence-2](https://arxiv.org/abs/2311.06242): one promptable model across several visual tasks.
- [Transmit What You Need](https://arxiv.org/abs/2412.13646): task-adaptive structured visual evidence.
- [Winoground](https://arxiv.org/abs/2204.03162): evidence that compositional relations require explicit evaluation.

### Validation

Measure scene/event precision and recall, unknown rejection, entity localization, relation precision, slot omission rate, extraction latency and peak memory. Include counterexamples where the same objects occur in different relationships.

## 5. P2 — Context relevance

### Header field

```text
context_similarity: float32 in [-1, 1]
```

### Calculation

For unit-normalised image embedding `z` and task-text embedding `t`:

```text
context_similarity = dot(z, t)
```

The text embedding is cached. The image embedding is reused from P1. No K-means clustering is required to define relevance; a cluster describes data similarity but does not define the user's goal.

### Scheduler effect

The same Eiffel Tower photograph can be relevant for landmark identification and irrelevant for road-obstruction monitoring. The value lets the scheduler condition its decision on the active task.

### Paper support

- [CLIP](https://arxiv.org/abs/2103.00020): shared image/text embedding and zero-shot matching.
- [Towards Goal-Oriented Semantic Communications](https://arxiv.org/abs/2304.00848): communication value depends on the receiver's goal.

### Limits and validation

This is raw similarity, not a probability or final importance. Validate matched versus mismatched task ranking, prompt sensitivity, task swaps on the same image and visually similar but task-irrelevant negatives. Missing task or embedding failure makes the field unavailable.

## 6. P3 — Delivered-history novelty

### Header fields

```text
novelty_distance: float32 in [0, 1]
history_version: uint64
history_count: uint32
novelty_method: categorical ID
```

### Calculation

The eligible history contains only observations successfully delivered for the configured task scope. Let `s` be the maximum cosine similarity between the current embedding and an eligible delivered embedding:

```text
novelty_distance = (1 - s) / 2
```

The transform maps cosine `1` to distance `0` and cosine `-1` to distance `1`. It does not convert similarity into a probability of new information. For explanation only, a hypothetical `s = 0.94` produces `0.03`.

If no eligible history exists, novelty is unavailable and `history_count = 0`. It is not assigned an invented zero or one.

### Knowledge base

Use two distinct stores:

- a versioned concept catalogue for shared meanings;
- a delivered-observation store for hashes, embeddings, supported concepts, timestamps, task IDs and delivery outcomes.

Query history before recording the candidate. Extraction alone must not make an image non-novel. INFRA updates delivered history only after successful delivery. Novelty is refreshed in a decision-time sidecar when history changes; the original header stays immutable.

### Scheduler effect

P3 helps deprioritise repeated views when they provide no new task evidence. It remains separate from impact because a duplicate can depict a serious event while adding little new information.

### Paper support

- [Knowledge Base-based Semantic Image Transmission Using CLIP](https://arxiv.org/abs/2504.01053): receiver knowledge-base retrieval and semantic matching.
- [DINOv2](https://arxiv.org/abs/2304.07193): alternative general visual descriptors for a later retrieval ablation.

The exact distance formula and delivered-history lifecycle are project definitions.

### Limits and validation

Appearance change is not necessarily new knowledge. Similar-looking images can contain a new fact. Test exact duplicates, near duplicates, viewpoint changes, small new task-relevant evidence, failed deliveries and two redundant images waiting in the same queue. Fact-level novelty is a later method requiring reliable entity/incident association.

## 7. P4 — Visible extent of effect

### Header fields

```text
extent_fraction: float32 in [0, 1]
extent_basis_id: categorical ID
```

### Calculation

For a declared target mask and effect mask:

```text
extent_fraction = area(target_mask intersect effect_mask) / area(target_mask)
```

The denominator must be nonzero and the target/effect pair must be applicable. An instance-count basis may instead use:

```text
affected supported instances / visible supported instances
```

Mask overlap and instance count receive different basis IDs and are never exchanged silently.

### Scheduler effect

P4 distinguishes a local visible condition from a widespread one. It does not determine impact by itself: a small affected region may contain the decisive evidence.

### Paper support

- [Segment Anything](https://arxiv.org/abs/2304.02643): transferable prompted masks.
- [LADI v2](https://arxiv.org/abs/2406.02780) and [xBD](https://arxiv.org/abs/1911.09296): domain precedents for visible affected structures and damage annotations.

### Limits and validation

The field measures visible image coverage, not physical area, hidden damage, financial loss or seriousness. A verified absence may be zero; an unsupported measurement is unavailable. Validate mask quality and extent error against annotated examples, including occlusion, perspective and small targets.

## 8. P5 — Task impact

### Header field

```text
impact_level: uint8 ordinal in {0, 1, 2, 3}
```

### Definition

Task impact answers: if the supported situation is true, how consequential is it for this task?

| Level | Meaning |
|---|---|
| 0 | Negligible consequence |
| 1 | Limited consequence |
| 2 | Substantial consequence |
| 3 | Major consequence |

The task profile supplies concrete examples and boundaries. The levels are ordered but not equally spaced.

### Extraction

The first implementation uses a reviewed, versioned rule table over supported semantic concepts and task context. This makes the prototype inspectable and supplies labels for checking the end-to-end system. The rule table cannot convert unsupported predictions into facts.

After sufficient independently reviewed labels exist, train an ordinal classifier using P1 evidence and the task representation. Record whether each value is human-provided, rule-derived or model-predicted.

### Scheduler effect

P5 supplies consequence without mixing it with interpretation confidence, novelty or visible size. A serious repeated image may have high impact and low novelty.

### Paper support

- [Towards Goal-Oriented Semantic Communications](https://arxiv.org/abs/2304.00848): goal-specific performance and loss.
- [Automated Triaging of Adult Chest Radiographs](https://pmc.ncbi.nlm.nih.gov/articles/PMC6438359/): application-defined image triage and queue evaluation.

Medical categories are methodological precedent and are not transferred to generic images.

### Validation

Use blinded human review, inter-rater agreement and ordinal error. Include small high-impact evidence, high-confidence low-impact content, ambiguous potentially consequential evidence and task swaps. A scheduler trained to reproduce a rule table demonstrates rule imitation, not discovered significance.

## 9. P6 — Urgency

### Header field

```text
urgency_level: uint8 ordinal in {0, 1, 2, 3}
```

| Level | Meaning |
|---|---|
| 0 | Delay-tolerant |
| 1 | Routine |
| 2 | Time-sensitive |
| 3 | Immediate under the declared task policy |

### Definition

Urgency answers how rapidly the usefulness or avoidable consequence changes when delivery is delayed. It is not a second name for task impact.

### Extraction

Use the task profile, trusted capture metadata and a reviewed event/task mapping. Pixels alone do not prove that an old photograph depicts a current emergency.

If the application provides a deadline, INFRA retains it and computes at scheduling time:

```text
slack_ms = deadline_ms - now_ms - estimated_service_ms
```

`slack_ms`, arrival time, age and service cost are dynamic scheduler-state fields, not replacements for semantic urgency. Unknown deadlines and unknown urgency remain unavailable.

### Scheduler effect

P6 lets the scheduler distinguish important archival evidence from information whose value decays quickly. Queue age and maximum-wait safeguards remain necessary to prevent starvation.

### Paper support

- [Smart Chest X-ray Worklist Prioritization](https://arxiv.org/abs/2001.08625): waiting-time effects and safeguards against long waits.
- [Significance-Driven Semantic Communication](https://arxiv.org/abs/2608.28441): task value together with temporal and resource allocation considerations.

The four urgency levels and task mappings are project definitions.

### Validation

Test high-impact delay-tolerant versus lower-impact time-sensitive examples, stale images, missing timestamps, negative slack, p95/max waits and starvation. Synthetic deadlines must be labelled as simulation settings.

## 10. P7 — Task evidence sufficiency

### Header field

```text
sufficiency_probability: float32 in [0, 1]
sufficiency_basis: categorical ID
```

### Definition

The source-image target is:

> Probability that the visible image contains enough evidence for a reviewer following the task rubric to provide a supported answer.

This is not the probability of a particular event. It is not generic image quality. It is also not automatically the probability that one selected VQA model answers correctly. A supported answer can be “no”; target absence does not imply insufficient evidence.

### Extraction

1. Create answerable, unanswerable and ambiguous labels for explicit tasks/questions.
2. Treat disagreement according to a documented policy.
3. Train a small binary head using frozen image features, task features and optionally P1 evidence.
4. Calibrate on a separate validation split.
5. Evaluate on a held-out test split.

Until this head exists and is calibrated, the field is unavailable. Do not substitute CLIP similarity, caption confidence, blur, bitrate or latent width under the same name.

### Scheduler effect

P7 distinguishes an image that contains usable task evidence from one where the target is hidden, cropped or too degraded. Low sufficiency does not automatically mean drop; it can coexist with uncertain but potentially consequential evidence.

### Codec-aware extension

A later, separately calibrated field may estimate reconstructed-image sufficiency for a specified SwinJSCC rate and channel condition. Train it using actual decoded outputs. SwinJSCC latent width or transmitted channel count does not identify how many meaningful facts survive. Actual future reconstruction results cannot be used as current scheduling inputs.

### Paper support

- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599): calibration of predictive probabilities.
- [GQA](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.pdf): grounded, compositional visual question answering.
- [SwinJSCC](https://arxiv.org/abs/2308.09361): rate- and channel-adaptive JSCC representation; it does not provide task sufficiency automatically.

### Validation

Report discrimination, Brier score, reliability diagrams, calibration error, false-low-sufficiency cases, task switching and controlled degradation. Source-image and reconstructed-image sufficiency use different basis IDs and calibration evidence.

## 11. Interpretation confidence and evidence quality

These are not separate mandatory parameter families in v2.

- Interpretation confidence is attached to the specific P1 claim through its `evidence_score` and `score_kind`. A single global confidence would hide which claim is uncertain.
- Blur, exposure and detector diagnostics remain in the audit artifact. They may be useful debugging signals but do not directly state task evidence quality.
- P7 replaces generic evidence quality only after it is trained and calibrated against an explicit answerability target.

This removes duplication while preserving the information where it has a precise meaning.

## 12. Role of the significance paper

The significance paper remains central to the scientific argument, but significance is not a mandatory header field in this version.

[Significance-Driven Semantic Communication](https://arxiv.org/html/2608.28441v2#S3.SS1) defines the conditional value of a sampled observation as the difference between the loss of the old action and the loss of the observation-updated action, both evaluated under the updated posterior:

```text
v(x; H, T) = sum_y p(y | H, x, T) *
             [L(y, a_old) - L(y, a_new)]
```

This is used in our project as:

- an optional task-specific teacher target;
- an evaluation of whether earlier delivery improves receiver decisions;
- a principled link between semantic evidence and scheduler reward.

It is not a universal image-importance score. It requires a target state, actions, task loss, belief model and delivered history. We are adapting its decision-theoretic metric; we are not claiming to reproduce the paper's Meta-VIB transceiver or sensor-allocation system.

Illustrative arithmetic: for binary state and unit error loss, if evidence changes the belief from 0.20 to 0.80, the old and new decisions have losses 0.80 and 0.20 under the updated belief. The sampled significance is 0.60 loss units. This example is not a project measurement.

## 13. Logical header contract

```text
SemanticHeaderV2
  schema_version
  catalogue_version
  task_id
  task_version
  observation_id
  extraction_manifest_id

  semantic_slots[8]
    concept_id
    evidence_score
    score_kind
    valid

  omitted_count_by_family[4]

  context_similarity
  novelty_distance
  history_version
  history_count
  novelty_method
  extent_fraction
  extent_basis_id
  impact_level
  urgency_level
  sufficiency_probability
  sufficiency_basis

  field_validity_mask
  unavailable_reason_by_field
```

Start with explicit JSON or equivalent typed records for inspection. Do not claim a header byte size until a binary serialization is specified and measured. Categorical IDs must be represented categorically when passed into a learned policy, not treated as continuous magnitudes.

The immutable source header retains extraction-time values. A decision-time sidecar contains refreshed novelty, history version, queue age, deadline slack and service cost. The audit artifact retains full candidates, boxes, masks, prompts, raw scores, model hashes, configuration and timing.

## 14. Module ownership

- **SemParam:** image extraction, concept catalogue, task heads, parameter provenance and knowledge queries.
- **EncDec:** SwinJSCC, communication codebook, rate/channel configuration and reconstruction.
- **INFRA:** arrivals, clock, budgets, delivery acknowledgement, mutable delivered history and immutable scheduler snapshots.
- **Scheduler:** consumes typed snapshots and returns a job ID. It does not import image models, codecs or neural frameworks.

Image-space extraction runs alongside encoding. An off-the-shelf image model cannot read an arbitrary SwinJSCC latent. A latent-only semantic extractor requires a separately trained and validated adapter through `INFRA.features.FeatureProvider`.

The current `SemParam/pipeline.py` records knowledge immediately after extraction. That record cannot represent successfully delivered history. Implementation must separate an observation cache from a delivery-confirmed knowledge store before P3 is considered correct.

## 15. Implementation order

1. Define and review the task profile, concept catalogue and typed header.
2. Implement shared embedding, detector output and P1 slot selection.
3. Implement P2 relevance and P3 delivery-confirmed novelty.
4. Add deterministic relations and selective P4 masks.
5. Implement reviewed P5/P6 rule tables with explicit provenance.
6. Build and calibrate P7 only after answerability labels exist.
7. Integrate immutable headers and dynamic sidecars with Scheduler/INFRA.
8. Evaluate source-image and actual decoded-image task outcomes.

## 16. Minimum validation before claiming the system works

- Use incident/source/near-duplicate group splits.
- Separate training, calibration and test sets.
- Run actual checkpoint inference; do not infer model quality from architecture alone.
- Report accuracy, calibration, coverage, unavailable rate, latency and memory.
- Compare eight semantic slots with larger capacity and measure omitted important evidence.
- Verify novelty updates only after successful delivery.
- Evaluate identical images under different tasks.
- Compare FIFO, declared priority, deadline scheduling where valid, and learned scheduling under identical arrivals/resources.
- Measure receiver task accuracy or task loss over time, timely useful deliveries, p95/max wait, misses/drops, extraction cost and header overhead.
- Report human/oracle parameters separately from model-extracted parameters.
- Run ablations: P1 only; add P2; add P3; add P4; add P5/P6; add P7.

A small demonstration validates data flow and scheduling mechanics, not generalisation. No paper establishes this complete seven-family header as a universal standard. Each field earns its place only if its measured scheduling benefit justifies its extraction and serialization cost.

## 17. Final commitment

The implementable system is a fixed, versioned concept header plus six independent task/history values. It uses shared perception, selective segmentation, explicit task policy and delivery-confirmed memory. It keeps missing information honest and retains the significance paper as the decision-theoretic evaluation foundation.

The full implementation target contains all seven families. The immediate scheduler experiment deliberately excludes P1 and P4 from the RL state. Its state uses P2, P3, P5, P6 and P7, with reviewed annotations or task rules supplying values that do not yet have validated automatic extractors. This separation prevents a scheduling demonstration from being misreported as an extraction result.

## 18. Open-source model plan for semantic evidence

This section specifies realistic open-source choices for P1. These models are not all loaded together. The implementation exposes interchangeable backends and measures quality, latency and memory before selecting one profile.

### 18.1 Recommended first full profile

The recommended modular profile is:

```text
OpenCLIP image encoder
    -> scene and event prompt similarities
    -> shared embedding for P2 and P3

YOLO-World-S
    -> configured entity labels, boxes and detector scores

deterministic geometry
    -> supported spatial relations from boxes

SAM 2, invoked only when required
    -> masks for containment and extent

small calibrated task heads, added after labels exist
    -> event refinement, task impact and evidence sufficiency
```

This profile is preferred initially because each output can be inspected, ablated and replaced independently. It also reuses one image embedding across scene/event matching, context relevance and novelty.

### 18.2 Available model matrix

| Subtask | Primary model | Alternative | Output used by this project | Practical decision |
|---|---|---|---|---|
| Shared image/text embedding | [OpenCLIP](https://github.com/mlfoundations/open_clip), starting with a pinned ViT-B/32 checkpoint | SigLIP 2 after a controlled comparison | Unit-normalised image and text vectors; cosine similarities | Use the existing repository embedding abstraction. Do not run both in production. |
| Broad scene recognition | [Places365 ResNet-18](https://github.com/CSAILVision/places365) | OpenCLIP prompt matching | Top scene IDs and scores | Benchmark Places365 against prompts. Map its 365 native classes into our smaller catalogue without retraining IDs. |
| Open-vocabulary entity detection | [YOLO-World](https://github.com/AILab-CVC/YOLO-World), small checkpoint first | [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | Entity label, box and detector score | Prefer YOLO-World for the fast path; use Grounding DINO as the accuracy/grounding comparator. |
| Prompted segmentation | [SAM 2](https://github.com/facebookresearch/sam2) or original [SAM](https://github.com/facebookresearch/segment-anything) | A task-specific segmentation head after labels exist | Mask and mask-quality diagnostics | Prompt with validated detector boxes. Run only for eligible relations or extent. |
| Unified detection/caption/grounding experiment | [Florence-2-base-ft](https://huggingface.co/microsoft/Florence-2-base-ft) | Florence-2-large-ft if hardware permits | Parsed detections, region descriptions and grounding outputs | Treat as an alternative end-to-end backend. Do not mix free-form captions directly into the numeric header. |
| Learned scene graph / semantic relations | [OpenPSG](https://github.com/franciszzj/OpenPSG) or [Microsoft Scene Graph Benchmark](https://github.com/microsoft/scene_graph_benchmark) | A small project-specific pairwise relation head | Subject-predicate-object candidates | Research comparator only after geometric relations work. These stacks are heavier and domain-sensitive. |
| Event hypothesis | OpenCLIP event prompts followed by a small multi-label head on frozen features | Florence-2 structured prompt, validated on the target domain | Event IDs plus cosine or calibrated probability | Prompt scores bootstrap the catalogue. A probability is emitted only after supervised training and calibration. |

Pin the exact repository revision, checkpoint hash, preprocessing, vocabulary and licence in the run manifest. Model code being public does not imply every checkpoint has identical licensing terms. YOLO-World's official repository and its dependencies need particular licence review before redistribution.

### 18.3 Scene classification in depth

Use a two-level catalogue:

```text
level 1: indoor, outdoor-natural, outdoor-built, road/transport
level 2: residential street, highway, bridge, forest, coast,
         office, hospital room, industrial site, landmark exterior, ...
```

Places365 produces native place categories. A versioned mapping converts selected native categories into our level-1 and level-2 scene concepts. OpenCLIP independently scores project prompts. The two outputs are evidence sources; they are not averaged without validation.

Implementation sequence:

1. Select 30–60 scene concepts with clear definitions and an `unknown` class.
2. Map relevant Places365 labels into those concepts.
3. Write two to four text prompts per concept for OpenCLIP.
4. Cache normalised text embeddings.
5. Run both methods on a labelled development subset.
6. Select the backend using top-k recall, calibration/rejection behaviour, latency and domain errors.
7. Retain the top two supported scene candidates in P1.

Do not define `cars plus disaster` as a scene. Represent it as a road scene, vehicle entity and event hypothesis. This prevents combinatorial growth while retaining the combination.

### 18.4 Entity detection and part understanding

The task profile supplies the detector vocabulary, for example:

```text
person, face, vehicle, car, truck, building, road, bridge,
flame, smoke, water, debris, traffic signal, animal
```

YOLO-World is run once with this fixed vocabulary. Non-maximum suppression and score thresholds are fixed from validation data. Keep all supported instances in the audit artifact, but transmit only the two selected entity concepts in P1.

Parts require an explicit design. A detector may support `face`, `wheel` or `door`, but it does not automatically know that a particular wheel belongs to a particular car. Part-of relations require containment/overlap plus a compatible parent-child catalogue entry. SAM/SAM 2 can improve the boundary, but semantic ownership must still be validated.

Grounding DINO is valuable when the target is described by a phrase, such as “blocked exit” or “person beside the damaged vehicle.” A grounded box remains a model hypothesis, not proof of injury, danger or causality.

### 18.5 Spatial relations in depth

Implement relations in three tiers:

**Tier A — deterministic geometry:** left/right, above/below, overlap, containment and normalised image-plane separation. These are cheap and testable.

**Tier B — mask geometry:** inside, covers and on-region where box geometry is insufficient. Detector boxes prompt SAM 2; masks are associated back to detected instances.

**Tier C — learned semantic relations:** holding, riding, emerging-from or interacting-with. Use OpenPSG/a scene-graph model only after a labelled relation set and a measured need exist.

Every relation candidate retains the subject instance ID, object instance ID, predicate, source method, raw measurement and threshold version. The compact header contains the class-level relation concept and score; the audit record preserves the grounding.

No tier emits causal claims. Temporal causal or chain-reaction reasoning requires multiple observations, tracking and a separate temporal model.

### 18.6 Event hypotheses in depth

An event is inferred from the whole image plus grounded concepts:

```text
event features = shared visual embedding
               + scene candidates
               + entity presence/count features
               + selected relation features
```

Start with OpenCLIP prompt similarities for a bounded event catalogue. Examples include ordinary traffic, visible fire, collision evidence, flooding evidence, visible structural damage and road obstruction. Prompt scores retain `cosine_similarity` provenance.

Once labelled examples exist, train a small multi-label MLP or linear head over frozen features. Use a multi-label target because fire and obstruction may coexist. Calibrate each supported event or use a documented multiclass calibration method. The event head does not output delivery priority.

Florence-2 or another open vision-language model may generate structured descriptions during dataset review. Its output must be parsed, validated and stored as model-assisted annotation. It must not silently become ground truth.

### 18.7 Backend interface

Each model adapter implements an interface equivalent to:

```text
extract(image, task_profile) -> ModelEvidence

ModelEvidence:
  image_embedding
  scene_candidates[]
  entity_instances[]
  relation_candidates[]
  event_candidates[]
  optional_masks[]
  provenance
  timings
```

The semantic-header builder consumes `ModelEvidence`. The scheduler never calls these models. This preserves the SemParam/Scheduler boundary and allows mock evidence in tests.

## 19. Tomorrow's scheduler profile

### 19.1 Exact scope

Tomorrow's RL state excludes:

- P1 semantic evidence vector;
- P4 visible extent.

It uses only:

```text
P2 context relevance
P3 delivered-history novelty
P5 task impact
P6 urgency
P7 task evidence sufficiency
```

Queue age, arrival time, service demand, deadline/slack where available and a validity mask remain environmental state. They are necessary to make a scheduling decision and are not semantic-image parameters.

### 19.2 Provenance of tomorrow's five values

| Parameter | Tomorrow's source | Claim allowed |
|---|---|---|
| P2 context relevance | Actual OpenCLIP cosine between image and predefined task text | Measured model similarity, after the checkpoint runs |
| P3 novelty | Actual embedding distance to delivery-confirmed task history | Measured representation novelty under the configured method |
| P5 task impact | Reviewed dataset annotation or deterministic task rubric | Human/rule-derived task consequence, not automatic visual inference |
| P6 urgency | Reviewed task policy, capture metadata and optional declared deadline | Policy-derived delay sensitivity, not urgency discovered from pixels |
| P7 sufficiency | Reviewed answerable/unanswerable annotation for the task | Human/oracle sufficiency label; not a calibrated model probability |

For tomorrow, P7 is stored numerically as 0 or 1 with `score_kind=human_label` and `sufficiency_basis=source_image_oracle`. It must not be called a calibrated probability in plots. Ambiguous cases are unavailable or retain an explicit reviewer-agreement value outside the RL input.

This profile validates whether a scheduler can use meaningful semantic parameters. It does not validate automatic task-impact, urgency or sufficiency extraction. That claim requires the later full profile.

### 19.3 Knowledge base for P3

Use a small exact-search store first; FAISS is unnecessary for tens or hundreds of images.

Logical tables:

```text
observations
  observation_id, image_hash, embedding_ref, task_id, captured_at

deliveries
  delivery_id, observation_id, task_id, delivered_at,
  success, history_version

semantic_facts_later
  observation_id, concept_id, relation_binding, confidence
```

The novelty query selects only successful deliveries from the applicable task/history scope. It runs before the selected job is inserted. After successful service, INFRA appends the delivery and increments `history_version`. Remaining jobs are rescored or have a refreshed sidecar generated.

Unit tests must cover empty history, exact duplicate, failed delivery, task isolation, query-before-record and two queued duplicates.

### 19.4 Fixed RL observation

For a bounded queue of `N` candidates, each row contains:

```text
[context_similarity,
 novelty_distance,
 impact_level / 3,
 urgency_level / 3,
 sufficiency_label,
 queue_age_normalised,
 service_demand_normalised,
 slack_normalised,
 validity bits]
```

The environment also supplies an action mask. Empty padded rows cannot be selected. Stable job IDs, not row positions, identify jobs. Randomise or consistently sort candidate rows during training so accidental input order does not become the policy.

The action selects one whole latent job. INFRA, not the policy, advances time, completes service, handles arrivals/deadlines and updates delivered history.

### 19.5 Significance-driven reward and evaluation

The core ideology from [Significance-Driven Semantic Communication](https://arxiv.org/html/2608.28441v2#S3.SS1) is followed directly: useful communication is measured by improvement in the receiver's task decision, not by visual rarity or a handcrafted sum of features.

For task `T`, history `H_t`, selected observation `x_t`, target state `Y`, actions `A` and loss `L`:

```text
a_old = argmin_a E[L(Y,a) | H_t, T]
a_new = argmin_a E[L(Y,a) | H_t, x_t, T]

significance_t = E[L(Y,a_old) - L(Y,a_new) | H_t, x_t, T]
```

Both actions are evaluated under the same updated belief, matching equation 3 of the paper. Exact optimal actions make the value nonnegative.

Use `significance_t` as the primary delivery reward when a valid belief/loss model exists. Resource feasibility, drops and deadlines remain environment constraints/outcomes. Do not replace the significance term with a weighted sum of the five input features.

For tomorrow, choose one explicitly finite task where significance can be calculated and audited, such as deciding whether the current route is blocked. Define:

- binary or small finite target state;
- receiver prior;
- permitted actions;
- a fixed loss table;
- how a delivered annotated image updates the receiver belief.

If this belief model cannot be completed and checked in time, use supervised/oracle per-image utility as a temporary simulator target and label it accordingly. Do not call it equation-3 significance.

Primary metrics are cumulative delivered significance versus simulated time and final receiver task loss. Secondary metrics are timely useful deliveries, per-impact/urgency wait, p95/max delay, deadline misses, drops, throughput and starvation. This makes the paper's idea visible in the results rather than merely cited in the introduction.

### 19.6 RL algorithm and baselines

Use a basic DQN only for a bounded discrete queue and whole-job selection. The network consumes the flattened fixed-size queue matrix plus masks, or a small shared per-job encoder followed by masked action scores. The shared encoder is preferable because the same job representation is applied to every row.

Training requirements:

- replay buffer;
- target network;
- masked epsilon-greedy actions;
- deterministic seeds;
- episode-level train/validation/test traces;
- saved configuration and model checkpoint;
- no test-scene or near-duplicate leakage into training.

Required baselines:

1. FIFO;
2. strict annotated priority;
3. earliest deadline first, only where deadlines are independently defined;
4. greedy current significance, when the exact significance oracle exists;
5. DQN using the five semantic inputs and environmental state.

Greedy significance is important: if DQN cannot outperform or trade off better than this baseline under future arrivals and resource limits, RL has not demonstrated additional value.

## 20. Detailed implementation plan

### Phase A — Freeze contracts and experiment meaning

**Goal:** make every number reproducible before writing models.

1. Add versioned `TaskProfile`, `ConceptCatalogue` and `SemanticHeaderV2` schemas.
2. Add `value`, `value_kind`, `availability`, `source`, `method_version` and `evidence_ref` to each parameter record.
3. Define tomorrow's task, target state, actions and loss table.
4. Define the history scope and delivery-success event.
5. Reject unknown task/catalogue versions during configuration validation.

**Acceptance:** schema round-trip tests; categorical IDs are not normalised as floats; unavailable values survive serialization; invalid versions fail before execution.

### Phase B — Build the labelled scheduling fixture

**Goal:** create a small, auditable dataset for the five-input experiment.

1. Select images and assign immutable image/source/incident-group IDs.
2. Select task text before labelling each image's relevance.
3. Record human/rule-derived impact, urgency and sufficiency with rationales.
4. Record ambiguity and reviewer identity/version.
5. Keep related incidents and near duplicates in the same split.
6. Define arrival traces, payload/service demand, deadlines and buffer policy separately from image labels.

**Acceptance:** every image has provenance; every parameter has a source; no label is inferred from a filename; split leakage audit passes.

### Phase C — Implement P2 and P3

**Goal:** obtain the two actually model-computed parameters for tomorrow.

1. Pin an OpenCLIP checkpoint and preprocessing.
2. Add deterministic task-text prompt files and cached text embeddings.
3. Store unit-normalised image embeddings with checkpoint/config hashes.
4. Calculate raw P2 cosine.
5. Split observation caching from successful-delivery history.
6. Implement exact nearest-neighbour P3 and history versioning.

**Acceptance:** repeated runs match within declared tolerance; task swap changes P2 as expected on fixtures; an exact duplicate becomes near-zero novelty only after successful delivery; failed delivery does not change history.

### Phase D — Supply P5, P6 and P7 for tomorrow

**Goal:** provide honest semantic inputs without claiming unavailable automatic models.

1. Implement a versioned impact rubric and urgency policy as data files.
2. Load reviewed values through `provided_parameters` or a dedicated labelled-fixture provider.
3. Load sufficiency as an oracle binary label with explicit basis.
4. Preserve human/rule provenance through the parameter bundle and result files.
5. Reject missing required labels or pass them with validity bits according to the experiment configuration.

**Acceptance:** no field is labelled `model_predicted`; plots distinguish extracted and annotated inputs; manual examples reproduce the rubric.

### Phase E — Implement the scheduling environment and DQN

**Goal:** schedule whole latent jobs using the five-input profile.

1. Extend immutable snapshots through INFRA contracts without importing SemParam into Scheduler.
2. Implement bounded queue encoding, padding and action masks.
3. Keep all queue/time/history mutation inside INFRA.
4. Implement the finite task belief and equation-3 significance calculation in the experiment/evaluation layer.
5. Implement baselines before DQN.
6. Implement DQN with saved seeds, replay, target network and validation checkpoints.

**Acceptance:** no job is served before arrival; no invalid action is selected; one server never overlaps jobs; higher reward is traceable to task-loss reduction; accounting balances completed, dropped and unfinished jobs.

### Phase F — Build the standalone notebook

**Goal:** produce the professor-facing result without hiding logic inside the notebook.

The notebook imports canonical modules and contains:

1. environment and commit information;
2. task/parameter definitions;
3. dataset thumbnails with annotations;
4. extraction/loading of the five inputs;
5. baseline and DQN execution;
6. before/after image order for every policy;
7. cumulative significance/task-loss plots;
8. wait, throughput, miss, drop and starvation metrics;
9. a clear limitations section distinguishing oracle and model-derived inputs;
10. saved outputs outside the disposable runtime.

**Acceptance:** clean restart reproduces the run from a pinned commit/config; plots use the same traces across policies; failures and unfinished jobs remain visible.

### Phase G — Implement full P1 semantic evidence

**Goal:** replace annotated visual understanding with measured open-source perception.

1. Create catalogue and prompt files.
2. Implement OpenCLIP and Places365 scene backends.
3. Implement YOLO-World entity adapter and Grounding DINO comparator.
4. Implement deterministic geometry and audit bindings.
5. Add selective SAM 2 masks.
6. Add the fixed eight-slot selector.
7. Benchmark Florence-2 as an alternative backend.

**Acceptance:** held-out scene/entity/relation/event measurements, slot omission study, actual latency/memory, unknown handling and checkpoint provenance.

### Phase H — Implement P4 and learned task heads

**Goal:** add extent and replace oracle task fields where data supports it.

1. Label applicable target/effect masks and calculate P4.
2. Train an ordinal P5 head on reviewed impact labels.
3. Keep P6 primarily application/policy-defined; train it only if temporal outcome labels justify doing so.
4. Create answerability labels and train/calibrate P7.
5. Compare learned values with oracle values in scheduler ablations.

**Acceptance:** disjoint training/calibration/test splits; calibration evidence; mask/extent errors; provenance differentiates rule, human and model values.

### Phase I — SwinJSCC integration

**Goal:** schedule real codec payloads and measure receiver outcomes.

1. Pair each immutable semantic header with its encoded payload ID.
2. Use actual payload cost in the correct unit: native complex channel uses or digital bits, never tensor storage bytes.
3. Schedule before the constrained transmission point.
4. Decode selected jobs with the real checkpoint.
5. Measure receiver task outcome on decoded images.
6. Train a separate reconstructed-sufficiency predictor only from actual codec/channel experiments.

**Acceptance:** real checkpoint inference, matched source/decoded task evaluation, no future reconstruction leakage and complete run provenance.

### Phase J — Ablation and final research claim

Run:

- environmental state only;
- P2 only;
- P2 + P3;
- add P5;
- add P6;
- add P7;
- later add P1 and P4;
- oracle parameters versus extracted parameters;
- significance reward versus declared class-priority objective.

Retain a parameter only when its task benefit justifies extraction latency and header cost. The final claim must state which values were measured, annotated, rule-derived or predicted.

## 21. Significance-paper traceability across the five-input profile

| Parameter | Relationship to sampled significance |
|---|---|
| Context relevance | Evidence unrelated to task `T` should produce little posterior/action change, but similarity alone is not significance. |
| Delivered novelty | History `H` is explicit in the posterior. Redundant evidence normally changes the receiver belief less, but representation distance alone is not significance. |
| Task impact | The task loss `L` encodes the consequence of wrong decisions. Impact is a coarse interpretable description of that consequence. |
| Urgency | The usefulness of delivering a sample can depend on when its decision can still affect the task. Timing enters state and evaluation rather than being hidden inside visual importance. |
| Evidence sufficiency | An answerable observation can produce a meaningful belief update. Sufficiency does not determine whether the update changes the optimal action. |

The five parameters are predictors or explanatory factors. The significance value is the task-decision outcome used for evaluation/reward when its assumptions are satisfied. Keeping these roles separate prevents circular labels and makes the paper's ideology operational.
