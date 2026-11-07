# Semantic parameters for whole image latent scheduling

Presentation reference and implementation specification — 28 September 2026

## Decision

Keep eleven parameter families: scene category, spatial relations, event hypotheses, extent of effect, context relevance, novelty, task impact, urgency, evidence quality, interpretation confidence, and decision significance. Each transmitted field is a single numeric value. Some families require several individually named scalars to preserve their meaning. This is a proposed contract, not measured extraction results.

Rename the earlier **task significance** field to **task impact**. Impact describes the consequence of the situation under the task. **Decision significance** describes the benefit of receiving this observation. A serious event can have high impact while a redundant photograph of it has low additional decision value.

The target is a generic whole-image latent scheduler with explicit application context. The same architecture supports ordinary photos, landmarks, portraits, infrastructure, or hazards through a versioned task vocabulary. Generic architecture does not imply that one model reliably understands every possible event. The scheduler receives separate scalars and no handcrafted weighted SPI. It chooses which complete SwinJSCC or communication-codebook payload is served next.

## What a value actually means

A numeric value can be a **category code**, **physical or geometric measurement**, **similarity**, **ordinal rating**, **probability**, or **expected loss reduction**. These types must not be mixed. A scene ID of 7 is not more important than a scene ID of 2; a CLIP similarity of 0.31 is not a 31 percent probability; a visible affected fraction of 0.25 is not 25 percent damage severity.

Every field has a definition, units, vocabulary/model version and availability flag. Missing is unavailable, not zero. The header contains scalars; detailed boxes, masks, embeddings and explanations stay in an audit artifact. Categorical codes are encoded categorically for learned policies, not normalized as continuous scores. The numerical examples below are invented illustrations, not outputs from model inference.

**How to read an example:** First identify its source. A category number comes from our vocabulary table. A fraction comes from counted pixels or instances divided by a declared denominator. A cosine comes from model embeddings; its magnitude has no intrinsic priority interpretation. A probability requires calibration against labeled data. A task impact level comes from a human-defined rubric. Decision significance comes from a task-specific loss calculation. No example number in this document is a measured project result, and none is an automatic priority threshold.

## Parameter overview

| Family | Header field examples | Output and illustrative meaning | Main motivation |
|---|---|---|---|
| 1 Scene category | scene_id | Integer 7 = residential exterior in our vocabulary | R1, R2 |
| 2 Spatial relations | subject_id, relation_id, object_id | Integers 4, 2, 9 = smoke region overlaps building region | R1, R3, R4 |
| 3 Event hypotheses | event_id, event_match | Integer 3 = possible fire; cosine 0.31 = image/prompt similarity | R2, R5, R6 |
| 4 Extent of effect | affected_fraction, extent_basis_id | 0.25, 1 = 25 percent of a supported target mask overlaps the effect mask | R7, R8 |
| 5 Context relevance | context_similarity | Cosine 0.34 between image and supplied task text | R2, R9 |
| 6 Novelty | novelty_distance | Hypothetical cosine 0.94 gives (1−0.94)/2 = 0.03 | R10, R11 |
| 7 Task impact | impact_level | Ordinal 0–3; 3 = major consequence under a documented task rubric | R9, R12 |
| 8 Urgency | deadline_ms | Absolute deadline; scheduler derives 400 ms remaining slack | R12, R13 |
| 9 Evidence quality | blur_variance, exposure_bad_fraction | 120 variance units; 0.08 of pixels outside fixed exposure thresholds | R14 motivates separating quality from uncertainty; statistics are engineering choices |
| 10 Interpretation confidence | event_confidence | Calibrated 0.80 for the specified event claim | R14 |
| 11 Decision significance | decision_value | 0.60 normalized loss units avoided in the worked example below | R15 |

The invented values above have these exact origins:

| Example | Where the number comes from | What we may conclude |
|---|---|---|
| scene_id=7 or event_id=3 | We assign an integer to each category in a fixed vocabulary. | The code names a category; its numeric size means nothing. |
| affected_fraction=0.25 | For example, 2,500 supported affected target pixels / 10,000 visible target pixels. | One quarter of the visible target overlaps the validated effect mask. |
| context_similarity=0.34 | Hypothetical dot product of unit-normalized image and task embeddings. | Relative model alignment within a controlled comparison; no probability or universal cutoff. |
| novelty_distance=0.03 | Hypothetical closest delivered-image cosine s=0.94; (1−s)/2=0.03. | Small distance in this embedding space; no proof that the information is useless. |
| exposure_bad_fraction=0.08 | For example, 800 pixels outside configured exposure thresholds / 10,000 image pixels. | Eight percent of pixels cross those thresholds. |
| blur_variance=120 | Hypothetical sample variance of the configured Laplacian filter response. | A raw sharpness proxy; whether 120 is good requires a validated comparison. |
| event_confidence=0.80 | Would have to come from a trained event predictor calibrated on held-out labeled images. | On a matching data distribution, predictions around 0.80 should be correct around 80 percent of the time. We have not established that calibration here. |
| decision_value=0.60 | Expected loss 0.80 for the old decision minus 0.20 for the revised decision under the same updated belief. | The decision improves by 0.60 units **under the chosen task and loss**. |

**Why the novelty transformation?** Cosine similarity between normalized vectors lies in [-1,1]. The transform (1−s)/2 maps identical vectors (s=1) to distance 0 and the mathematical opposite (s=−1) to distance 1. That makes a bounded input convenient for a scheduler. This transform is our engineering convention; R10/R11 motivate retrieval and embeddings, not this exact formula. Real cosine values must come from actual model inference, and their distribution must be checked on the target dataset. A hypothetical 0.94 was chosen solely to make the arithmetic visible.

## 1 Scene category

**Question:** What kind of setting is visible?

**Extraction:** Run the existing CLIP image encoder once and compare its embedding with cached text embeddings for a fixed, versioned scene vocabulary. Select the strongest supported category; apply a validation-selected reject threshold/margin so unsupported scenes become unknown. A supervised scene head can replace prompt matching if validation shows a need.

**Values:** scene_id is an unsigned categorical integer, with 0 reserved for unknown. Store scene_match separately as a cosine in [-1,1] if the policy uses match strength. For example, scene_id=7 means residential exterior only because vocabulary version 1 assigns that meaning.

**Why useful:** Scene context helps distinguish visually similar evidence in different settings. It does not establish importance by itself. A landmark scene can matter more than a street scene under a landmark-identification task.

**Motivation:** R1 motivates task-dependent visual representations; R2 supplies image/text matching. Our taxonomy, threshold and scalar encoding are project design choices.

## 2 Spatial relations

**Question:** Which visible entities are related, and how?

**Extraction:** Detect supported entities with one small detector. Compute image-plane overlap, containment and relative position from boxes. Introduce masks only when the required relation depends on shape. Use a learned relation head only for predicates that geometry cannot justify.

**Values:** Send subject_id, relation_id and object_id as three separate integer scalars. A bounded vocabulary can encode smoke / overlaps / building. These identify entity categories, not persistent real-world identities. Initially retain one task-relevant relation, chosen by a declared rule and deterministic ties; send relation_overflow_count when additional relations are omitted. Keep all instances and boxes locally. More relation slots require an explicit schema revision.

**Why useful:** Objects alone cannot distinguish water beside a road from water over a road. However, overlapping boxes do not prove flooding, a trapped person, injury or causal involvement. A relation inferred only from boxes must retain that method provenance.

**Motivation:** R1 and R3 motivate relationships; R4/R5 provide localization. They do not validate our particular relation selection rule.

## 3 Event hypotheses

**Question:** What might be happening, or what condition might be present?

**Extraction:** Reuse CLIP against versioned event descriptions initially. Evaluate on independently labeled mixed images; train a small multi-label head on frozen features if prompt matching is insufficient. The default runtime emits one dominant hypothesis. Preserve additional hypotheses locally and expose event_overflow_count. A later schema can add independent per-event probability fields where multi-event omissions harm scheduling.

**Values:** event_id is a categorical integer; event_match is a cosine [-1,1]. Possible fire and collision scene are hypotheses, not Critical/Emergency classes. Unknown is valid. Ordinary activity may be an event category if visible; motion direction, consciousness and intent are unavailable from an unsupported still image.

**Why useful:** Separates setting from occurrence: residential exterior is a scene; apparent fire is an event.

**Motivation:** R2 supplies the efficient matching mechanism. R6 offers event-classification precedent and R7 supplies multi-label condition evidence. Their disaster labels do not define a generic priority order.

## 4 Extent of effect

**Question:** How much of the relevant visible target is affected by the supported condition?

**Extraction:** For a validated target/effect pair, calculate area(target mask intersect effect mask) / area(target mask). Require a nonempty, sufficiently visible target mask. A count-based definition is possible for supported instances: affected targets / visible targets, but must have a different extent_basis_id. Never silently substitute box area or whole-image area for affected target area.

**Values:** affected_fraction is a float [0,1]; extent_basis_id identifies mask-overlap or instance-count semantics. For example, 0.25 means one quarter of the visible target region overlaps the effect region. It does not measure hidden damage, financial loss or physical area in square meters. Segmentation provides boundaries, not the truth of the effect claim.

**Why useful:** Distinguishes a localized observation from a widespread visible effect. Ordinary photos without an applicable effect receive unavailable, rather than a fictional zero-severity value.

**Motivation:** R7 and R8 motivate labeled condition/extent evidence and segmentation. This exact scalar is our operational definition. Optional segmentation runs only for applicable supported events.

## 5 Context relevance

**Question:** How closely does the image match the current application request?

**Extraction:** Compute cosine similarity between normalized CLIP image and task-text embeddings. Cache task embeddings. Use a specific supplied goal such as identify this landmark or inspect visible road blockage. Do not derive the task from the image itself.

**Values:** context_similarity is a float [-1,1], with larger values indicating stronger alignment under that model/prompt. Example 0.34 is a similarity, not 34 percent relevance. If a [0,1] storage range is necessary, (cosine+1)/2 is only an affine representation; it does not calibrate relevance.

**Why useful:** The Eiffel Tower can score more strongly for landmark identification than road-blockage assessment. K-means may index observations, but a cluster does not define the user's goal.

**Motivation:** R2 provides the method; R9 motivates goal/context dependence. Validate task switching and prompt sensitivity.

## 6 Novelty

**Question:** How different is this evidence from what was successfully delivered for this task?

**Extraction:** Use existing image embeddings and a task-scoped delivered-history index. Let s be the maximum cosine similarity to an eligible reference; novelty_distance=(1-s)/2. Exact image hashes independently identify byte-identical duplicates. Query before recording. Update delivered memory only on successful delivery and refresh this decision-time field when history changes.

**Values:** Float [0,1]. For the hypothetical closest-match cosine s=0.94, novelty_distance=(1−0.94)/2=0.03. This is representation distance, not a probability of new information or a priority level. The number 0.94 is invented for explanation. No reference history means unavailable plus history_count=0. The scheduler can recognize this as a first observation. Never insert zero or one and silently call it measured novelty.

**Why useful:** Avoids treating repeated imagery as equally informative. Lighting/viewpoint changes can increase distance without adding useful facts; therefore this field remains separate from relevance and decision significance. Fact novelty is an extension requiring validated event identity and fact matching, not the default generic field.

**Motivation:** R10 motivates receiver KB retrieval; R11 motivates reusable visual descriptors. Start with CLIP to avoid a second backbone; compare DINOv2 only if retrieval errors justify it.

## 7 Task impact

**Question:** If the hypothesized situation is real, how consequential is it for this task?

**Extraction:** Define a task rubric with independent impact labels. Train a small ordinal head using scene/event/relation/extent evidence and task context, or use a transparent task rulebook for the first labeled prototype. Store whether a result is human-supplied, rule-derived or predicted. Use consequence conditional on the situation, leaving probability of the situation in confidence.

**Values:** impact_level is ordinal: 0 negligible, 1 limited, 2 substantial, 3 major task consequence. These are declared rubric categories; a 3 is not mathematically three times a 1. An unsupported task has no impact estimate. A general-purpose model cannot infer an arbitrary application's losses without that definition.

**Why useful:** A serious condition remains serious even after several photographs arrive. Its next photograph may nevertheless have little new decision value. This field therefore differs from novelty and significance.

**Motivation:** R9 motivates task consequences; R12 demonstrates application-specific triage. Neither validates this generic four-level impact rubric. The rubric requires human review and independent test labels.

## 8 Urgency

**Question:** How long can delivery wait before it loses its intended usefulness?

**Extraction:** Application or task policy supplies an absolute deadline. At each decision, INFRA computes slack_ms=deadline_ms-now_ms-estimated_service_ms. Keep timestamps and cost separate. If only a task-specific validity duration is available, derive a deadline from a trustworthy capture timestamp and record that derivation.

**Values:** deadline_ms is a timestamp; dynamic slack_ms is a signed number of milliseconds. A slack of 400 means the estimated service may start within 400 ms and still finish on time. Negative slack means lateness is already expected. No declared timing requirement means unavailable.

**Why useful:** High consequence and short remaining time are different properties. A historic photograph of a fire cannot establish a current emergency or a deadline from pixels alone.

**Motivation:** R12/R13 motivate urgency-aware queues and waiting-time protection. The timing definition is our operational choice. A learned delay-cost curve is a later alternative requiring temporal outcome labels.

## 9 Evidence quality

**Question:** Is the image visually usable enough to support the interpretation?

**Extraction:** Use two cheap independent measurements on standardized image preprocessing: variance of the grayscale Laplacian and fraction of pixels below/above fixed exposure thresholds. Pin resize, grayscale conversion, Laplacian kernel, intensity scale and thresholds in configuration.

**Values:** blur_variance is nonnegative in squared intensity-response units; exposure_bad_fraction is [0,1]. Illustratively 120 and 0.08 mean a measured sharpness proxy of 120 and 8 percent of pixels outside the chosen exposure range. No universal good/bad threshold applies across scenes. Textureless clear images can have low Laplacian variance.

**Why useful:** Gives the policy observations about blur/exposure without inventing a universal image quality probability. Poor quality does not mean low importance. A later task-answerability head may estimate usable evidence, but needs explicit labels and calibration.

**Motivation:** R14 supports keeping model confidence distinct from correctness; these low-cost image statistics are our engineering choices, not formulas proposed by the cited semantic papers.

## 10 Interpretation confidence

**Question:** How well supported is the named event interpretation?

**Extraction:** Calibrate a supervised event classifier on a disjoint validation split using an appropriate method such as temperature scaling. Send probability of the selected event, explicitly tied to event_id and the class vocabulary. Keep raw prompt similarity as event_match until a validated probability estimator exists. Unknown/unsupported scenes need rejection and an unavailable confidence where calibration is not applicable.

**Values:** event_confidence is [0,1]. A calibrated 0.80 means similarly scored predictions should be correct about 80 percent of the time on the relevant evaluation distribution. It is not 80 percent urgency, image quality or overall trustworthiness. Relation and scene confidence, if needed, must be separate named fields rather than one global confidence covering everything.

**Why useful:** Allows strong and weak interpretations to be treated differently. Check reliability diagrams, Brier score, calibration error and domain shift. Low confidence in possible high-impact evidence is not an automatic drop instruction.

**Motivation:** R14 provides the calibration foundation. Calibration on one domain does not guarantee calibration on every generic photo.

## 11 Decision significance

**Question:** How much does this observation improve the receiver's decision?

**Paper definition:** R15, Section III-A equation 3, evaluates both the old optimal action and the observation-updated optimal action under the same posterior. Their expected-loss difference is nonnegative for exact optimal decisions. It is not simply prior Bayes risk minus this sample's posterior Bayes risk.

**Our adaptation:** Define task T, delivered history H, target state Y, possible actions and a fixed task loss L. Estimate beliefs before and after candidate image evidence x. Let a_old minimize expected loss before x and a_new minimize it afterward. Then:

decision_value = sum over y of P(y | H,x,T) × [L(y,a_old) − L(y,a_new)].

**Values:** Nonnegative float in the task's declared loss units. With bounded loss [0,1], the ideal value lies in [0,1]. It is not a probability. Tasks with different loss scales are not directly comparable without a declared shared normalization. Approximation/calibration failures must be reported; never hide them by calling every predicted value exact information value.

**Extraction:** Fit a small calibrated task-state model; evaluate the finite action/loss table before and after evidence. Train an optional lightweight value predictor against these independently specified targets. For SwinJSCC, estimate beliefs from delivered reconstruction during evaluation, and validate a sender-side predictor for the configured rate/channel before using its estimate at runtime. Do not use future actual reconstruction outcomes as scheduling inputs.

**Dependencies:** This field is unavailable until the task, loss, belief model, actions and history are defined. It is not computable by feeding arbitrary pixels to an off-the-shelf detector. All other fields remain separate policy inputs.

### Worked value example for presentation

This is a deliberately simplified nonmedical binary task: decide whether a route is blocked. Actions are report clear or report blocked. A wrong answer costs 1 and a correct answer costs 0. Before an image, the estimated blockage probability is 0.20, so the old decision is clear. After image evidence it is 0.80, so the updated decision is blocked.

Under the updated belief, the old decision has expected loss 0.80 and the updated decision has expected loss 0.20. Thus decision_value=0.60 loss units. The prior and posterior minimum risks are both 0.20; subtracting them would incorrectly give zero for this per-sample definition.

Now consider a duplicate that leaves the belief and optimal decision unchanged. It can still depict an important blocked route, but its decision_value is zero for this particular loss and action set. Under another loss, such as probabilistic prediction loss, additional certainty can have value even without changing a hard class. The loss definition is therefore part of the parameter's meaning.

## Supporting fields and normalization

Keep task_id, task_version, schema_version, image/job ID, capture time, codec/rate identity, payload cost and cost unit as metadata. Add detected_object_count and detected_person_count as optional scalar evidence. They are outputs of supported detection, not verified counts of all entities in the world. They help preserve object information beyond the selected relation.

All families are generic, but some values will be unavailable for a given image. A portrait can have scene, relevance, novelty and quality without an applicable effect extent. A first observation has no measured history distance. A photograph without an application task cannot have task significance by definition.

Normalize continuous model inputs using training-split statistics or fixed declared transformations, retaining physical units in stored artifacts. Do not min–max normalize arbitrary categorical IDs. Do not use DBMS normalization as a numerical operation: its useful analogy here is avoiding redundant meanings and preserving relationships among fields. Never multiply relevance and novelty into one input; let the policy learn their interaction.

## Efficient execution and architecture

Image + application context → shared CLIP embedding + one small detector → scalar measurements and hypothesis → delivered-history lookup → optional calibrated task heads → versioned scalar header paired with whole SwinJSCC latent → scheduler selection → transmission/decode → delivered-history update.

Run masks only for a supported extent question. Avoid mandatory depth, full scene graphs, VLM captioning and multiple embedding backbones. Benchmark quality/latency on our actual hardware. Scene/event/relevance text embeddings are cached. A lightweight local history index is enough initially. A structured scene artifact may support explanation without being sent to the scheduler.

SemParam owns extraction and prediction; EncDec owns SwinJSCC and communication VQ; INFRA owns state, service, dynamic slack and delivery notifications; Scheduler consumes immutable scalar snapshots and selects job IDs. Codec features enter through the existing FeatureProvider. Off-the-shelf CLIP/YOLO cannot directly read arbitrary SwinJSCC latents. A latent-only extractor requires a separately trained and validated adapter.

Novelty and decision value depend on history and must be tagged with history_version. In the centralized prototype, INFRA can refresh them through SemParam before policy evaluation. A frozen transmitted header is not silently rewritten; refreshed values belong to a decision-time sidecar. A distributed deployment would need an explicit feedback/update mechanism, outside the present scope.

## What changed from the supplied presentation and note

The existing PDF survey records relevant pages 87–95, 113–131, 143 and 174–175. This specification uses that prior page review and its source mapping; PDF text extraction was not available in the current environment for a fresh full-page audit.

- Preserve the PDF's independent parameters and the note's separation of evidence, task consequence, knowledge and reliability.
- Generalize hazard/damage into event hypotheses and supported extent; disaster datasets are optional domain evidence, not the universal taxonomy.
- Replace a hand-weighted task importance/SPI with separately labeled task impact and the explicitly defined decision-value calculation.
- Use successfully delivered history for novelty; replace cross-UE scarcity with image-level redundancy in current scope.
- Keep raw latent storage, digital payload bits and native JSCC channel uses distinct. No RB field is needed here.
- Leave SRS, next-layer gain and quality deficit out of this whole-latent header. The current codec does not thereby acquire a validated progressive-layer representation.
- Treat the PDF's unverified titles as unresolved citations, not implementation evidence. Specifically, the full method of Feature Importance-Aware Task-Oriented Semantic Transmission and Optimization was not inspected; the XR MAC paper had program/abstract evidence only; the exact Knowledge-Based Edge and Access Scheduling citation remained unresolved in the existing survey.

## Validation and presentation sequence

Validate scene/event labels and detector evidence on held-out images; relation claims against annotated relations; extent against masks where applicable; relevance with matched/mismatched tasks; novelty using duplicates and genuinely different images; impact against independent task rubrics; urgency through hand-calculated deadlines; confidence with calibration plots; decision value using enumerated toy decisions and held-out task outcomes. Split by source/event/near-duplicate group. No priority labels or future deliveries may leak into extraction.

For a presentation, use six slides: (1) same image, different task; (2) eleven families and output types; (3) shared-model extraction flow; (4) one illustrative scalar header with missing fields visible; (5) the blocked-route 0.60 decision-value example; (6) before/after whole-latent scheduling and the tests needed to validate it. Explain that Critical/Emergency/Moderate/Normal are scheduler/predictor outputs under a rubric, not additional visual event categories.

## References and exactly what they support

R1. [Transmit What You Need: Task-Adaptive Semantic Communications for Visual Information](https://arxiv.org/abs/2412.13646). Objects, relations and task-dependent representations; adapted as evidence design, not copied as a priority model.

R2. [Learning Transferable Visual Models From Natural Language Supervision — CLIP](https://arxiv.org/abs/2103.00020). Image/text embeddings and zero-shot classification; direct extraction foundation.

R3. [Visual Genome](https://arxiv.org/abs/1602.07332). Structured object/attribute/relation annotations; motivates relation schema.

R4. [Grounding DINO](https://arxiv.org/abs/2303.05499). Text-conditioned localization; detector comparator.

R5. [YOLO-World](https://arxiv.org/abs/2401.17270). Efficient open-vocabulary detection; candidate runtime detector, speed to be measured locally.

R6. [EmergencyNet](https://arxiv.org/abs/2104.14006). Lightweight emergency event classification; domain precedent rather than universal priority.

R7. [LADI v2](https://arxiv.org/abs/2406.02780), [CrisisMMD](https://arxiv.org/abs/1805.00713), [xBD](https://arxiv.org/abs/1911.09296). Condition, informativeness and damage supervision with domain-specific applicability. Their labels are not equivalent to generic scheduling priority.

R8. [Segment Anything](https://arxiv.org/abs/2304.02643). Prompted masks; geometric support for extent when the target/effect labels are independently established.

R9. [Towards Goal-Oriented Semantic Communications: New Metrics, Framework, and Open Challenges](https://arxiv.org/abs/2304.00848). Goal/context-sensitive evaluation; motivates task definition and distinctions among relevance and consequence.

R10. [Knowledge Base-based Semantic Image Transmission Using CLIP](https://arxiv.org/abs/2504.01053). Receiver KB retrieval; inspires history matching, not a verified novelty formula.

R11. [DINOv2](https://arxiv.org/abs/2304.07193). Alternative reusable visual descriptors; optional retrieval ablation.

R12. [Automated Triaging of Adult Chest Radiographs](https://pmc.ncbi.nlm.nih.gov/articles/PMC6438359/). Application-defined image triage; adapt the evidence/label/queue separation, not medical class semantics.

R13. [Smart Chest X-ray Worklist Prioritization](https://arxiv.org/abs/2001.08625). Waiting-time and false-negative effects; motivates tail-delay evaluation and maximum-wait safeguards.

R14. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599). Calibration methods and score/probability distinction. Does not supply the chosen blur/exposure statistics.

R15. [Significance-Driven Semantic Communication](https://arxiv.org/html/2608.28441v2#S3.SS1). Equation 3 defines conditional value of sampled information. The task/history-conditioned image estimator proposed here is our adaptation; the paper's transceiver and scheduling experiments do not validate our full system.

R16. [SwinJSCC](https://arxiv.org/abs/2308.09361). Encoder/decoder and channel/rate adaptation; supplies representation, not intrinsic task importance.

R17. [QuizRank](https://arxiv.org/abs/2509.15059), [Visual Jev](https://arxiv.org/abs/2609.25845), [Conditional Visual Evidence Utility](https://arxiv.org/abs/2608.28316). Optional task-question assessment, shared VLM execution and state-dependent evidence motivation. Not required dependencies or proof that a learned policy will outperform baselines.

R18. [Semantic-aware Transmission Scheduling](https://arxiv.org/abs/2305.13706). Explicit state/action/objective motivation for later learned scheduling; not a generic visual parameter extractor.

The local PHASE1_LITERATURE_SURVEY.md retains additional PDF references and unresolved-reference notes. Papers above motivate methods or concepts individually; no source establishes this complete eleven-family contract as a universal validated standard.
