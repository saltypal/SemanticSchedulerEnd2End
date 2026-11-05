# Literature survey: Semantic-Aware AI Scheduler, Phase 1

Research checkpoint: 28 September 2026. Status: design evidence and recommendations; no new extraction model, scheduler, or experiment results are claimed.

## Scope and research method

The target is an image/latent scheduler using separate evidence fields, explicit task context, and the existing SwinJSCC pipeline. Phase 1 selects whole image jobs; device scheduling and ORAN control procedures remain outside the scope. The user selected mixed images with explicit task context and content-based labels on curated disaster/traffic scenes for the first demonstration.

This is a focused engineering survey, not a systematic review or a claim of exhaustive coverage. Searches used alphaXiv discovery and PDF queries, followed by primary papers, author repositories, and dataset documentation. Coverage includes semantic communications, image triage, disaster assessment, detection and grounding, scene graphs, representation learning, uncertainty, and scheduling. Paper mechanisms are distinguished below from proposed adaptations. Recent preprints are candidate evidence, not established deployment results. No reported performance number is treated as a result for this project.

Local context reviewed: the 181-page *Semantic Aware Scheduler Presentation.pdf*, existing repository architecture, scalar header, extraction code, and history interface. Relevant presentation pages include 87–95, 113–131, 143, and 174–175. The current branch is `semantic-aware-ai-scheduler`. Existing uncommitted changes are preserved.

## 1. Findings that should drive the design

1. **Importance depends on the task.** A damaged bridge can matter for route planning; a crowded road can matter for traffic control. Image/text similarity supplies evidence of relevance, not the downstream benefit of delivery.
2. **Keep event, severity, confidence, and priority separate.** “Flood” is an event class; “severe structural damage” is a condition; “Critical” is a delivery decision under a task and policy.
3. **No surveyed source establishes a universal four-class priority dataset for mixed disaster/traffic images.** We should create a documented manual demonstration subset rather than rename existing event classes as priorities.
4. **Spatial and relational evidence is useful, but model outputs need limits.** Boxes, masks, and predicted relations can expose who or what is affected. Box proximity alone does not establish physical danger, injury, or distance.
5. **A feature vector does not remove the need to define the scheduling objective.** A learned policy still needs training targets, a reward, or explicit constraints. We can avoid a handcrafted SPI without pretending that priorities emerge without preferences.
6. **Priority classification and scheduling are different problems.** Classification estimates a property of one image. Scheduling selects an eligible image using queue state, service demand, time, and possibly delivered history.

These are synthesis and design recommendations, not claims that one paper implements our complete system.

## 2. Direct evidence for image prioritization

| Reference | What it actually does | How we should use it |
|---|---|---|
| Annarumma et al., 2019, [Automated Triaging of Adult Chest Radiographs with Deep Artificial Neural Networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC6438359/) | Uses report-derived radiological findings to develop critical/urgent/nonurgent/normal image triage and simulates reporting prioritization. | Adapt the separation of image evidence, expert urgency mapping, and queue evaluation. Clinical categories and models do not transfer directly to disaster scenes. |
| Baltruschat et al., 2020, [Smart Chest X-ray Worklist Prioritization using Artificial Intelligence](https://arxiv.org/abs/2001.08625) | Simulates FIFO and AI prioritization, including a maximum-wait safeguard. Average delay can improve while worst-case delay worsens because of false negatives. | Include per-class tail delay, maximum wait, and priority-with-maximum-wait as a baseline. A mean-only plot would hide an important failure. |
| Ghulam et al., 2026 preprint, [Cross-modal triage network](https://arxiv.org/abs/2609.04357) | Uses images and clinical indications with report-derived severity tiers. Its small expert audit shows a substantial gap between agreement with proxy labels and expert judgments. | Treat automatically generated priority labels as hypotheses needing human validation. This recent preprint is a cautionary example, not a disaster-triage benchmark. |

**Answer to the user's priority-literature question:** explicit multi-level image triage is well motivated by these medical worklist studies. Their transferable contribution is the experimental methodology. They do not validate a four-level emergency ranking for arbitrary images.

## 3. Dataset choices and their actual labels

| Dataset/source | Native supervision | Fit and limitation |
|---|---|---|
| [CrisisMMD](https://arxiv.org/abs/1805.00713), [official dataset](https://crisisnlp.qcri.org/crisismmd) | Informativeness, humanitarian categories, and damage severity for applicable imagery from seven disaster events; image and text annotations are separate. | Strong candidate for task relevance and humanitarian evidence. Damage severity is not available for every image and is not a scheduling label. Keep unknown/not-applicable states. Avoid using tweet text accidentally in an image-only experiment. |
| [LADI v2](https://arxiv.org/abs/2406.02780), [author dataset](https://huggingface.co/datasets/MITLL/LADI-v2-dataset) | Aerial infrastructure/environment labels, including roads, road damage, buildings, building damage, debris, flooding, and affected structures. | Useful for infrastructure assessment and multi-label extraction. Its aerial viewpoint is a domain shift from road-level photographs. Labels indicate visible conditions, not urgency. |
| [xBD](https://arxiv.org/abs/1911.09296), [official baseline](https://github.com/DIUx-xView/xView2_baseline) | Paired pre/post-disaster satellite imagery, building polygons, and building damage levels. | Strong later choice for change and structural damage features. It cannot provide individual human urgency from building damage alone. Requires suitable paired imagery. |
| [AIDER](https://github.com/ckyrkou/AIDER) | Fire/smoke, flood, collapsed building/rubble, traffic accident, normal. | Practical candidate for the small mixed demonstration because it includes traffic accidents. Event class must remain separate from our manually assigned priority. Review source/licensing conditions; the repository describes academic/noncommercial use. |
| [V-TIDB](https://arxiv.org/abs/2312.10737) | Traffic videos with accident and contextual annotations, including contact severity and temporal attributes. | Useful for a future temporal traffic extension. Reaction, predictability, and motion-based evidence cannot simply be copied onto isolated frames. |

**Recommended demonstration:** curate a small balanced disaster/traffic subset, initially around 24–40 images, subject to available image rights and coverage. AIDER can supply event diversity; CrisisMMD can broaden humanitarian content. This is our proposed experimental subset, not a published benchmark.

Every image needs an ID, original source, event/group identifier, task description, manual class, visible-evidence rationale, and ambiguity flag. Do not label every fire Critical or every normal-looking scene Normal. Use the explicit task to decide whether the visible condition requires immediate, rapid, routine, or background delivery. For the demonstration, provisionally define the order **Critical > Emergency > Moderate > Normal** and document that Critical outranks Emergency; these names have no universal standardized ordering.

Prefer a second human review and record disagreements. Keep near-duplicates and frames from the same incident together in any later train/test split. A tiny curated subset demonstrates scheduling; it is insufficient evidence that a learned model generalizes.

## 4. Extraction literature mapped to candidate header evidence

| Evidence family | Primary reference and contribution | Proposed adaptation and limit |
|---|---|---|
| Task-conditioned objects and layout | Park & Yoon, [Transmit What You Need](https://arxiv.org/abs/2412.13646): task-adaptive semantic communication using objects, relationships, and layout. | Core conceptual anchor: retain task-dependent structured evidence. Its semantic payload design is not a whole-image priority scheduler. |
| Efficient open-vocabulary detection | [YOLO-World](https://arxiv.org/abs/2401.17270): text-guided object vocabulary and detections. | Candidate extractor for class, box, score, count, and region coverage. Benchmark on our scenes; a general detector is not automatically a reliable injury/fire/damage detector. |
| Text-conditioned grounding | [Grounding DINO](https://arxiv.org/abs/2303.05499): grounding language expressions in image regions. | Comparator for targeted task prompts and uncommon objects. A grounded region does not prove complex states such as “person trapped.” |
| Object relationships | [Visual Genome](https://arxiv.org/abs/1602.07332): objects, attributes, and relationships with vocabulary grounding. | Adopt a scene-graph schema. Keep deterministic 2D geometry distinct from learned semantic relationships and attach provenance/confidence. |
| Shared semantic knowledge | [Scene Graph-Aided Probabilistic Semantic Communication for Image Transmission](https://arxiv.org/abs/2507.11913), 2025 preprint: shared statistical graphs support relation/entity reconstruction. | Adapt relation representations and context-conditioned novelty. Common co-occurrences are priors, not evidence that an event occurred; rare emergencies can violate them. |
| Task relevance | [CLIP](https://arxiv.org/abs/2103.00020): shared image/text embeddings. [SigLIP 2](https://arxiv.org/abs/2502.14786): improved multilingual and dense representation capabilities. | Compare separate similarities to task descriptions or concepts. Similarity is a proxy for relevance, not calibrated urgency or value of information. SigLIP 2 is a candidate, not a proven winner on our dataset. |
| Visual novelty and retrieval | [DINOv2](https://arxiv.org/abs/2304.07193), [DINOv3](https://arxiv.org/abs/2508.10104): transferable visual representations. | Compare embedding distance to appropriate history. A visually novel image can be irrelevant; embeddings should not become an implicit priority score. |
| Region extent | [Segment Anything](https://arxiv.org/abs/2304.02643): promptable segmentation. | Optional masks and area fractions for relevant regions. A mask supplies shape, not event meaning or importance. |
| Depth cues | [Depth Anything V2](https://arxiv.org/abs/2406.09414): monocular depth representations. | Optional relative-depth evidence after validation. Do not claim metric distance or time-to-collision from an uncalibrated still image. |
| Uncertainty | [Probabilistic Object Detection](https://arxiv.org/abs/1811.10800): evaluates semantic and spatial uncertainty. [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599): confidence calibration. | Preserve detector scores and missingness; calibrate on held-out domain data. Class entropy requires a real label distribution, and spatial uncertainty requires an appropriate model. |
| Paired-image change | [IP2CP](https://arxiv.org/abs/2310.17801): damage assessment from paired pre/post imagery. | Optional change evidence when a registered reference exists. An isolated image does not provide measured change. |
| Task-dependent attention | [TDSal](https://arxiv.org/abs/2607.09827), 2026 preprint: task-conditioned saliency. | Possible later comparator for task attention. Predicted human gaze is not emergency importance. |

The first extractor should be a small, testable combination: one detector/grounder, one image-text embedding model, explicit 2D geometry, and a history comparison. Additional segmentation, depth, or relation models should enter through ablation evidence rather than being mandatory dependencies from day one.

### Knowledge base: three distinct roles

1. **Task ontology:** relevant entities and relations, label synonyms, and task-specific interpretation rules. [ConceptNet](https://arxiv.org/abs/1612.03975) can offer vocabulary and commonsense candidates, but generic knowledge must not manufacture visual facts.
2. **Observation memory:** image embeddings, observed objects/relations, timestamps, and source provenance. This supports retrieval and novelty.
3. **Delivered-information memory:** records what successfully reached the receiver. This supports redundancy relative to receiver knowledge.

[Knowledge Base-based Semantic Image Transmission Using CLIP](https://arxiv.org/abs/2504.01053) uses embedding-based retrieval from a receiver knowledge base. Its category-consistency setting does not establish faithful reconstruction of the particular transmitted emergency scene. We can adapt the retrieval-memory concept without replacing SwinJSCC or treating a same-category image as a successful scene reconstruction.

A graph database does not reason merely because it stores edges. Reasoning needs explicit rules, retrieval procedures, or learned models. For Phase 1, a small curated ontology and auditable history are preferable to an unvalidated elaborate knowledge graph.

## 5. Proposed evidence contract for discussion

This is a design proposal, not an implemented header specification. Avoid collapsing these fields into one SPI.

| Group | Example fields | Source and handling |
|---|---|---|
| Identity and provenance | image/job ID, schema version, task ID, extractor versions | Supplied metadata; needed for traceability. |
| Objects | vocabulary ID, normalized box, detector score, optional mask extent | Image models; bounded list with explicit truncation information. |
| Relations | subject ID, predicate ID, object ID, score, method | Geometry or learned model; distinguish the two. |
| Task evidence | named concept similarities, relevant entity counts, domain damage labels where validated | Separate features; no arbitrary weighted aggregation. |
| Uncertainty and quality | missing-value mask, available calibrated scores, blur/exposure indicators if useful | Confidence and image quality are distinct. Low confidence is not an automatic reason to discard a potentially critical image. |
| History evidence | reference/history version, nearest-match similarity, relevant new entities/relations | Depends on history, not pixels alone. Record when and against what it was computed. |
| Timing and service metadata | capture/arrival time, deadline, encoded payload demand, codec/rate ID | Supplied or codec-derived, not extracted visually. |

Dynamic waiting time, deadline slack, remaining service, and current queue composition belong in scheduler state. Frozen header novelty can become stale as other images are delivered; either refresh the decision-time feature or explicitly evaluate the frozen approximation. Compute novelty before inserting the current image into the comparison history. Update delivered history only after successful delivery.

The existing seven-scalar TLV header is insufficient for arbitrary object/relation lists without a versioned extension. Before implementation, decide what is serialized, what remains local, list bounds, quantization, missing-value representation, and actual byte overhead. Sending every intermediate embedding is not justified merely because it is available.

## 6. Scheduling literature and the learned-policy question

| Reference | Actual mechanism | Relevance to our project |
|---|---|---|
| [Semantic-aware Transmission Scheduling](https://arxiv.org/abs/2305.13706), 2023 | Uses task/estimation-related costs, age/channel state, and learned transmission scheduling. | Adapt the explicit state/action/objective formulation. Its sensor/channel allocation setting does not validate image urgency features. |
| [Goal-oriented communication: a tensor-based framework](https://arxiv.org/abs/2304.00848) | Models goals, context, and sender/receiver state in evaluating communication consequences. | Strong basis for separating age, relevance, novelty, and actual task value. True value needs a specified consequence of delivery. |
| [DeepRM: Resource Management with Deep Reinforcement Learning](https://www.microsoft.com/en-us/research/publication/resource-management-deep-reinforcement-learning/), 2016 | Learns resource scheduling from state and an explicit objective. | Methodological precedent for job selection; not an image-triage model or evidence DQN must outperform heuristics. |
| [Learning to Compress and Transmit](https://arxiv.org/abs/2605.10095), 2026 preprint | Uses SwinJSCC and learned rate control under link/queue conditions. | Useful future connection between payload demand, rate, and queueing. Rate control is distinct from semantic ordering of images. |
| [Utility-Aware Progressive Inference over UDP Packet Blocks for Emergency Communications](https://arxiv.org/abs/2605.09933), 2026 preprint | Studies task utility of blocks and progressive inference. | Later inspiration for measuring task impact through ablation. Its packet/block setting and weighted objective are not our proposed whole-image scheduler. |

For a later DQN, a candidate state consists of each eligible job's evidence plus dynamic timing/service fields and task context. An action selects a job, with invalid-action masking. Variable queue sizes need a defined bounded representation or a set-based policy; arbitrary input order should not determine behavior. The transition models time, arrivals, completion, expiration, and history updates.

The unresolved part is the **objective**. Standard DQN estimates scalar discounted returns. Passing multiple features into it does not specify which delays or misses matter. A class-based first experiment can use a declared objective with constraints or a lexicographic evaluation, then investigate an appropriate constrained or multi-objective learning method. We should not introduce an arbitrary weighted feature sum or reward without discussing its meaning.

A classifier can estimate priority from image evidence and task context, followed by a transparent scheduler. It is not a substitute for scheduling. [I-JEPA](https://arxiv.org/abs/2301.08243) is representation learning, not an out-of-the-box queue policy. The user's “JEV” remains unresolved and is not assumed to mean JEPA.

## 7. Monday baseline: research-backed experiment specification

No extractor or RL model is required to validate the scheduler. Feed the manually assigned classes directly into policies; this isolates whether queue logic does what it claims.

| Policy | Purpose | Important condition |
|---|---|---|
| FIFO | Arrival-order control | Stable deterministic tie-breaking. |
| Strict priority | Tests that eligible higher-priority images start first | Nonpreemptive service: cannot interrupt a job already in service. |
| Earliest deadline first | Deadline-oriented comparator | Use independently specified deadlines; do not make it artificially identical to priority. Classical EDF optimality claims have assumptions that do not automatically hold here. |
| Class round robin | Shows a class-sharing trade-off | Cycles nonempty priority classes; this is not UE scheduling. |
| Priority with maximum-wait promotion | Tests protection against starvation | Inspired by worklist-triage safeguards; document promotion threshold and tie rules. |

Use two complementary views: a static arrangement illustration and a discrete-event streaming experiment. For every policy, show the same input thumbnails, output service order, class labels, and a timeline. Mark dropped or unfinished images separately rather than silently excluding them from the after-panel.

First, all jobs arrive at zero with equal service and an unlimited buffer. This makes order easy to explain; all work-conserving policies should have the same total throughput if all jobs complete. Then introduce streaming arrivals, overload, deadlines, and a separate finite-buffer scenario. Vary service demand only in a clearly identified experiment. Keep arrival trace, demand, buffer rules, and horizon fixed across policies.

Report per-class waiting time, completion delay, median/p95/maximum wait, served and timely-served counts, timely throughput, deadline misses, and drops by cause. Explain the fairness target: within-class equity and starvation are different from equal service across classes. A Jain index across priority classes alone is not a measure of scheduling correctness.

Acceptance checks: strict priority has no higher-priority *eligible* job waiting when a lower-priority one begins service; no job is served before arrival; one server does not overlap service; accounting balances arrivals with completed/dropped/unfinished jobs; deterministic ties reproduce; overload does not hide unserved jobs in delay averages. Report simulated durations as simulated, not measured radio performance.

## 8. Corrections and unresolved references in the presentation

| Presentation point | Required correction or evidence status |
|---|---|
| Pages 174–175: latent followed by off-the-shelf image models | CLIP/YOLO expects its supported image input representation, not an arbitrary SwinJSCC latent. A latent-based extractor requires a trained adapter/head. Initial image-space extraction can run alongside encoding. |
| Communication and semantic codebooks | VQ representatives used for compression do not inherently carry urgency labels. A task prototype bank is a different object requiring supervision/validation. |
| Page 87: similarity as VoI | Similarity can estimate relevance. Value of information concerns improvement in an explicit downstream decision/outcome. |
| Image age versus AoI | Time since a queued image's capture is image age. Receiver AoI depends on the freshest successfully delivered update for a defined process. Arbitrary unrelated mixed images do not automatically form one AoI process. |
| Weighted novelty/context formulas | These are design hypotheses and conflict with the user's current preference for separate fields. They are not adopted. |
| Page 130: box size/proximity | Image geometry alone cannot establish physical distance, motion, or urgency. |
| Page 131: YOLO to REBEL | REBEL extracts relations from text; it is not directly a visual relationship model. A caption-to-text-relation pipeline would add its own errors and needs evaluation. |
| “Feature Importance-Aware Task-Oriented Semantic Transmission and Optimization” | Bibliographic identity supported by the [IEEE ComSoc contents digest](https://www.comsoc.org/system/files/2024-08/Publications_Contents_Digest_2024_Aug.pdf); full method not independently inspected. Do not substitute a similarly titled rate-control paper. |
| “Semantic-Aware MAC Scheduling for XR Traffic in 6G Networks” | Title and abstract supported by the [official ITNAC 2025 program](https://itnac.org.au/sites/default/files/2025/ITNAC_2025_Program_Final.pdf). Abstract describes frame prediction and semantic/network evidence; full experimental details remain unverified. |
| “AI-Driven Task Scheduling in Edge and Access Networks: A Knowledge-Based Approach” | Exact bibliographic/full-text match remains unresolved. Detailed mechanisms attributed to it in the slides are not used as verified evidence. |

## 9. Architecture constraint that affects the bandwidth claim

[SwinJSCC](https://arxiv.org/abs/2308.09361) is the codec anchor; the communication codebook is a separate project component. Keep native JSCC channel uses separate from packetized codebook bits and in-memory tensor storage.

The image/latent selection decision must precede the constrained transmission if it is intended to control uplink traffic. If every full latent has already traversed that link before scheduling, the policy only changes downstream processing order. Even scheduling before the link does not reduce total transmitted payload when every image is eventually sent unchanged. Compression, filtering/dropping, or rate selection can reduce traffic; reordering primarily changes timeliness. Count header serialization and transmission overhead in later bandwidth evaluation.

## 10. Proposed next checkpoint and evidence gaps

The next discussion should settle the extraction contract: task descriptions, event/priority rubric, first detector and embedding candidates, object/relation bounds, uncertainty handling, and observation versus delivered-history semantics. This precedes implementation.

Suggested progression is (1) standalone manually labeled scheduler notebook, (2) independently validated image evidence extraction, (3) transparent priority prediction plus the same scheduling harness, (4) learned queue selection only after its objective and training data are defined, and (5) value-based decisions and device/ORAN extensions later.

Unknowns remain: domain-specific extraction accuracy, model latency on available hardware, header cost, inter-annotator priority agreement, training-set size, and the best policy under representative arrivals. No literature result resolves these for our system. They require controlled measurements and ablations: metadata only; add objects; add relations; add task evidence; add delivered-history evidence. Compare performance and extraction/header cost together.

Research deliverable complete for this checkpoint: source-linked candidate methods, dataset label distinctions, presentation corrections, baseline evaluation design, and unresolved decisions. Implementation and empirical validation remain pending by design.
