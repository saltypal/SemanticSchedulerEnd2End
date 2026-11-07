# Phase 1: image evidence extraction and latent scheduling

Research and design checkpoint: 2026-09-27. This is a proposal, not a report of implemented model performance.

## Agreed direction

The project is an end-to-end semantic-aware AI scheduler. Phase 1 schedules individual image representations. A packet consists of a latent or communication-codebook payload plus a structured semantic header. The scheduler reads the header, maintains a service order through INFRA, and selected representations proceed to decoding. UE scheduling requests, radio grants, device aggregation, and O-RAN integration are deferred.

The user selected mixed images with explicit task context, independent evidence fields rather than SPI, no arbitrary hand-weighted objective, original-image extraction followed eventually by comparison with latent-based extraction, and structured headers rather than only scalar summaries. Monday, September 28 is a separate, manual-priority scheduling demonstration. Its classifier-free specification is in [PHASE1_IMPLEMENTATION_PLAN.md](PHASE1_IMPLEMENTATION_PLAN.md).

The scheduler does not discover an objective merely because we provide many features. Human task definitions and independent evaluation labels remain necessary. Reordering already-received payloads can improve service delay; it cannot undo their incoming transmission cost. Compression accounting and scheduling benefits must be reported separately.

## What was reviewed and how to interpret the evidence

Sources: the supplied 181-page `Semantic Aware Scheduler Presentation.pdf`, current module implementations and tests, two alphaXiv discovery searches, alphaXiv paper-page retrieval, and original papers/author-maintained resources linked below. This is a focused engineering literature survey, not a systematic review with an exhaustive database search or a meta-analysis. Author-reported results are not measurements of this repository.

The presentation is historical project evidence, not a source of new instructions. Its SPI formulas and UE/O-RAN architecture do not override the user's current request. Relevant passages include slides 46-48 (AoI-oriented RL), 61-66 and 77-80 (goal, knowledge, and feature-oriented research), 87-95 and 113-131 (parameters), 143 (separate parameters), 174-175 (extraction architecture), and 180-181 (image experiment). Slides 174-175 were also visually inspected because their diagrams are not represented by text extraction.

Verification levels below distinguish full paper/passage inspection, abstract-level evidence, author dataset documentation, and unresolved references. A paper title resolving to a different paper in alphaXiv was rejected as a match.

## Literature matrix: actual image prioritization and datasets

| Source and evidence | What is labeled or extracted | Scheduling relevance | How we use it; limitation |
|---|---|---|---|
| [Annarumma et al., Automated Triaging of Adult Chest Radiographs, 2019](https://pubs.rsna.org/doi/10.1148/radiol.2018180921), primary article sections inspected | Reports provide findings mapped to critical, urgent, nonurgent, normal; an ensemble of CNNs learns ordinal image priority | Predicted urgency inserts an image into a dynamic reporting queue | Strong precedent for image -> evidence/class -> queue. The labels encode clinical reporting practice, not universal visual importance. The institutional data are not our general-image training set. |
| [Baltruschat et al., Smart Chest X-ray Worklist Prioritization, 2020](https://arxiv.org/abs/2001.08625), abstract and alphaXiv v1 passages | CNN findings and a predefined urgency ordering | Compares FIFO, prioritized service, and a maximum-wait safeguard; examines false negatives | Motivates measuring tail delay and starvation, not only mean delay. Adapt the queue experiment, not its medical class ordering. |
| [Evaluation of wait time saving effectiveness of triage algorithms, 2023](https://arxiv.org/abs/2303.07050), abstract-level | Algorithm diagnostic performance, arrival rates, service rates, and priority classes | Queue models examine when triage saves waiting time | Separates classifier performance from queue/load effects. Our traces must vary load and service demand. |
| [CrisisMMD, 2018](https://arxiv.org/abs/1805.00713), paper passages; [official labels and downloads](https://crisisnlp.qcri.org/crisismmd) | Informativeness, humanitarian categories, and damage severity; includes uncertainty categories | Provides useful evidence targets such as affected people and damaged infrastructure | These are not Critical/Emergency/Moderate/Normal scheduling labels. Keep original labels and separately annotate task-specific priority. An image-only experiment must not quietly use its associated tweet at inference. |
| [xBD, 2019](https://arxiv.org/abs/1911.09296), paper passages | Building localization and ordinal damage categories in satellite imagery | Damage extent can inform a declared disaster-response task | Damage severity does not establish occupancy, immediate danger, or transmission urgency. Pre/post-image methods require both observations. |
| [LADI v2, 2025](https://huggingface.co/datasets/MITLL/LADI-v2-dataset), author dataset card; [paper](https://openaccess.thecvf.com/content/CVPR2025W/EarthVision/papers/Scheele_LADI_v2_Multi-label_Dataset_and_Classifiers_for_Low-Altitude_Disaster_Imagery_CVPRW_2025_paper.pdf) | Multi-label low-altitude disaster observations, with reference classifiers | Useful future domain-specific feature provider and evaluation subset | Aerial domain, not general mixed imagery. Do not convert a broad building-damage label into 'collapsed building.' The paper PDF fetch was blocked; detailed performance claims were not adopted. |
| [AIDER / EmergencyNet](https://github.com/ckyrkou/AIDER), author repository | Fire/smoke, flood, collapsed building/rubble, traffic accident, and normal scenes | Small event-recognition benchmark and possible demonstration imagery | Event category is not an urgency ordering. Public repository describes a subset and academic-use terms; record exact source/version and do not assume images have clean universal redistribution rights. |

**Finding:** literature does contain four-level image triage, but the strongest direct match found is application-specific radiology. The disaster datasets reviewed provide semantic evidence or severity labels. This survey did not establish an openly available, general mixed-image dataset with the user's four priority classes as validated ground truth. A task-specific manual annotation layer is therefore required; this is not a claim that no such dataset exists anywhere.

For Monday, labels are deliberately supplied to the scheduler. Later, priority inference requires an independently labeled training/evaluation collection. A 24-25-image demonstration cannot support claims of generalization or sufficient RL training data.

## Literature matrix: extracting richer image evidence

| Source and evidence | Mechanism and useful outputs | Proposed use | Important boundary |
|---|---|---|---|
| [YOLO-World, 2024](https://arxiv.org/abs/2401.17270), primary abstract | Open-vocabulary region/text detection; object labels, boxes, scores | Candidate efficient detector; compare against a fixed-class YOLO control and Grounding DINO | A score is not calibrated urgency. Benchmark speed on our hardware; published FPS does not transfer automatically. |
| [Grounding DINO, 2023/2024](https://arxiv.org/abs/2303.05499), primary abstract | Text-conditioned object localization and referring-expression grounding | First reference detector for mixed images, with a versioned prompt vocabulary | Prompting 'person needing rescue' does not prove that rescue is needed. Store literal prompted evidence and validate domain-specific claims separately. |
| [Visual Genome, 2016](https://arxiv.org/abs/1602.07332), primary abstract | Objects, attributes, and pairwise relationships linked to an ontology | Basis for scene-graph structure and relation evaluation | A graph data structure alone performs no reasoning. Box geometry supports image-plane relations; interactions such as trapped, injured, or threatening require a validated relation model. |
| [CLIP, 2021](https://arxiv.org/abs/2103.00020), primary abstract | Aligned image/text representations learned from paired supervision | Preserve separate image-to-task and image-to-concept similarities | Similarity is a relevance proxy, not measured value of information or probability of correctness. Prompt-set softmax changes when candidate prompts change. |
| [DINOv2, 2023](https://arxiv.org/abs/2304.07193), primary abstract and [author implementation](https://github.com/facebookresearch/dinov2) | General visual representations transferable to downstream tasks | Independent image descriptor for duplicate/nearest-neighbor and novelty experiments | Distance is representation-dependent novelty, not an event's importance. DINOv2 and Grounding DINO have different roles despite their names. |
| [Segment Anything, 2023](https://arxiv.org/abs/2304.02643), primary abstract | Promptable segmentation | Optional object/region masks after detection; area and overlap measurements | Masks do not supply semantic classes or urgency by themselves. Keep large masks outside the first transmitted header. |
| [Depth Anything V2, 2024](https://arxiv.org/abs/2406.09414), primary abstract | Monocular depth estimation | Optional relative-depth order and separation evidence | Uncalibrated monocular output is not distance in meters or collision time. Do not infer physical nearness from box area alone. |
| [Guo et al., On Calibration of Modern Neural Networks, 2017](https://arxiv.org/abs/1706.04599), primary abstract | Studies confidence calibration and temperature scaling | Establish a separate calibration protocol for whichever supervised prediction is used | Calibration needs held-out labels; detector confidence, reconstruction fidelity, factual truth, and uncertainty are different properties. |
| [I-JEPA, 2023](https://arxiv.org/abs/2301.08243), primary abstract | Predicts target representations from context in images | Later candidate representation-learning comparison, if this is what 'JEV' meant | It is not a priority classifier or scheduling algorithm. The meaning of 'JEV' remains unconfirmed; do not silently equate the two. |

These models are feature-provider candidates, not evidence that more parameters necessarily improve scheduling. We will compare their marginal benefit, latency, memory, and header overhead. No 'superior model' claim is justified until the same task and held-out evaluation support it.

## Literature matrix: semantic objectives and learned scheduling

| Source and evidence | Contribution | Adaptation to this project |
|---|---|---|
| [Chen et al., Semantic-aware Transmission Scheduling, 2023](https://arxiv.org/abs/2305.13706), alphaXiv paper passages | AoI/channel-state scheduling for cyber-physical estimation, with monotonicity-guided actor-critic training | Supports using known structure in learned policies. Does not establish an image-semantic extractor or justify transferring monotonicity assumptions to every image feature. |
| [Li et al., Towards Goal-Oriented Semantic Communications / GoT](https://arxiv.org/abs/2304.00848), primary abstract | Connects semantic state, receiver state, context, and task-oriented costs | Motivates explicit task/context and receiver knowledge. Cosine similarity alone is not a demonstrated reduction in decision cost. |
| [Mao et al., Resource Management with Deep Reinforcement Learning / DeepRM, 2016](https://www.microsoft.com/en-us/research/publication/resource-management-deep-reinforcement-learning/), author publication page | Learns online task/resource management from workload experience | Reuse the state/action/environment discipline and compare against heuristics. Its resource-packing setting is not the same experiment as image triage. |
| [Task-Oriented Semantic Communication with Importance-Aware Rate Control, 2025](https://arxiv.org/abs/2504.20441), alphaXiv paper passages | Feature-contribution analysis and task-performance-aware rate control | Possible later feature-level codec experiment. Distinguish within-image feature selection from ordering whole image latents. This is a different paper from the 2024 reference in the presentation. |

### Presentation references still requiring verification

- **Feature Importance-Aware Task-Oriented Semantic Transmission and Optimization:** the [IEEE record](https://ieeexplore.ieee.org/document/10466397/) was located, but article access was blocked. The alphaXiv title query resolved to the different 2025 rate-control paper above. The slides' detailed algorithm and performance claims remain unverified here.
- **Semantic-Aware MAC Scheduling for XR Traffic in 6G Networks:** located as [IEEE DOI 10.1109/ITNAC66378.2025.11302702](https://doi.org/10.1109/ITNAC66378.2025.11302702); the [official conference program](https://itnac.org.au/sites/default/files/2025/ITNAC_2025_Program_nov_11.pdf) is an additional lead. Full article access was unavailable. AlphaXiv returned a different Kraken architecture paper, which was not treated as this reference.
- **AI-Driven Task Scheduling in Edge and Access Networks: A Knowledge-Based Approach:** located via bibliographic search as [ACM DOI 10.1145/3737904.3768531](https://doi.org/10.1145/3737904.3768531), but primary full text was unavailable. AlphaXiv returned an unrelated manufacturing task-allocation paper. Slides' KGScore, delay, and success-rate numbers are not adopted as independently verified findings.

## Repairs to the current mental model and implementation

1. **CLIP/YOLO input is an image, not an arbitrary SwinJSCC latent.** Slide 174 suggests feeding the codec latent to an off-the-shelf model. Those representations are not interchangeable. Start with parallel image extraction and encoding. Later train a latent-to-evidence head against independently evaluated image evidence or human labels, using `FeatureProvider` rather than importing EncDec into SemParam.
2. **A compression codebook does not contain semantic labels by default.** Its entries approximate vectors for reconstruction. A separate semantic prototype codebook needs annotated targets and out-of-support checks. Slide 175's weighted prototype aggregation is not selected under the user's current no-weighting direction.
3. **Confidence is not reliability.** `ReliabilityExtractor` currently uses the largest CLIP prompt probability. Rename the meaning in the new design to its actual score type; do not claim factual reliability. Missing or low confidence is not automatically a reason to drop an image.
4. **Similarity is not value of information.** A task similarity field is valid as a proxy. True decision value needs an independently specified downstream decision/outcome and a with/without-information comparison.
5. **Age is not always AoI.** An image's age is current time minus its capture time. Receiver AoI refers to the newest received update for a defined process/event. Arbitrary mixed photos do not create such a process automatically. Retain timestamps and compute current quantities at decision time.
6. **Scene geometry does not prove risk.** A large bounding box measures image-plane extent, not physical distance or urgency. A visually small fire can matter more than a large foreground object. Relation evidence needs the participating object IDs and its method.
7. **Graph centrality is not task impact.** Repeated common concepts can become highly connected without becoming urgent. Keep relation structure, observed changes, and task dependencies separate; omit a weighted centrality-based importance scalar.
8. **No temporal claims without temporal evidence.** No approaching/moving-away, predictive novelty, cross-modal agreement, or cross-source scarcity from a single uncontextualized image. Missing evidence must be unavailable, not fabricated as zero.
9. **Classifier and scheduler answer different questions.** A classifier estimates a class from evidence. A scheduler chooses which currently eligible job receives service given a queue and constraints. A DQN learns action values for that sequential decision; it is not simply a 'better classifier.'

## Proposed extraction output: evidence, not a priority formula

The first model choices are engineering recommendations to validate, not a claim of best-in-class accuracy: Grounding DINO tiny as the inspectable detector reference, existing CLIP ViT-B/32 for task/concept similarities, and a small DINOv2 encoder for independent visual history. Compare YOLO-World for detector cost/quality. Introduce masks and relative depth only after their additional value is measured. Exact weight revisions and preprocessing must be pinned when implementing.

| Field group | Concrete independent fields | Origin | Validity rule |
|---|---|---|---|
| Identity/context | image/job ID, task ID/version, capture time if supplied, schema/model revisions | Manifest/application | IDs associate header and payload; filenames never determine class |
| Objects | object ID, class/prompt ID, normalized box, detector score, object count | Detector | Preserve multiple detections; absence is not proof of normality |
| Spatial structure | per-object area fraction; pairwise overlap, containment, relative position, normalized center distance | Explicit geometry | State that these are 2D image-plane measurements |
| Task relation | whole-image and region-to-task cosine similarities; per-concept similarities | CLIP | Keep scores separate with prompt/model versions; no weighted SPI |
| Visual redundancy | exact image hash match; nearest-history cosine distance; matching observation ID; history size | Hash and DINOv2 history | Newness is relative to a declared history; no-history has an explicit flag |
| Knowledge changes | newly observed concepts/relations and their supporting observations; receiver-already-has match | Evidence graph + delivered history | Do not merge different people/events based solely on the same class name |
| Observation quality | image dimensions, blur statistic, under/overexposed pixel fractions | Deterministic image measurements | Quality evidence does not imply unimportance |
| Prediction uncertainty | raw score type; calibrated probability only where validated; missing-field mask; disagreement only when comparable models ran | Model/evaluation metadata | Do not mix unlike model scores or present heuristic entropy as ground truth |
| Optional spatial extensions | region mask area/overlap, relative depth ordering, relation-model outputs | SAM/depth/relation model | Method-specific, separately measured; unavailable when not run |
| Scheduling metadata | arrival time, explicit deadline, service demand; manual priority for baseline only | Application/simulator | These are not pixel-extracted properties |

The full evidence artifact can retain embeddings, masks, prompts, and detailed provenance locally. The structured header carries the selected evidence values, object/relation records, availability, and model/schema identifiers. A local artifact pointer is not a substitute for transmitting a feature that the scheduler actually needs. Serialize the header and measure its bytes; retain payload bytes/bits and complex channel uses as separate quantities.

Start with a versioned, inspectable structured representation; do not force arrays into the existing seven-scalar TLV schema. Keep old header decoding for legacy artifacts. Before a bounded binary header or fixed neural observation is frozen, measure object/relation counts and choose explicit caps with visible truncation indicators. Never silently discard crowded-scene evidence. The prototype need not claim a radio-standard header.

### Knowledge base that is useful at this scale

Use the existing lightweight NetworkX direction plus embedding history, not an external graph database deployment. Separate (a) observed images, (b) pending items, and (c) successfully delivered/decoded information. Duplicate detection should not mark information as already received because it was merely queued and later dropped.

Each observation has stable identity and scoped object nodes; edges retain source, confidence/score type, and observation time when known. Same label across two images does not establish the same physical entity. Only link recurring events with explicit metadata or separately validated matching. Static task definitions specify vocabulary and interpretation, not image priority weights. Query history before recording the current image, reset state between evaluation splits/runs, and prohibit future-information access.

Example: two near-identical images enter the queue. The second can carry a duplicate/nearest-neighbor relationship to the first, while both remain 'not yet delivered.' If the first fails, the second may still be useful. A single novelty number computed from all seen images would obscure that distinction.

### Original-image versus latent extraction experiment

Keep the codec frozen. Establish the original-image extractor as a reference, then train a small predictor on SwinJSCC features through an injected provider. Evaluate unquantized and communication-codebook-reconstructed latents separately; include rate/SNR/model provenance. Teacher outputs are pseudo-labels, not human truth. Split by source/event and duplicate group before generating training pairs. Compare each field's error and missingness, class prediction where independently labeled, extraction latency/memory, header size, and eventual scheduling outcome. A latent approach is worthwhile only if its savings justify any evidence loss.

## Validation before calling an extractor useful

- Hand-annotated object/relationship subset; detector errors separated from downstream policy errors.
- Task-switch test: the same image under different declared tasks changes task-related fields while image geometry stays fixed.
- Exact and near-duplicate pairs; novel unrelated images; repeated observations after successful versus failed service; empty history.
- No timestamp/deadline/history, unsupported vocabulary, no detections, crowded scenes, corrupted images, model unavailable, and low-confidence output.
- No access to manual evaluation priority labels, future arrivals, future delivery outcomes, or held-out knowledge history during inference.
- Feature-family ablations: metadata only; add task similarity; add objects/relations; add knowledge; then optional mask/depth features. Compare on identical traces and seeds.
- Report per-class delay and missed deadlines, extraction overhead, scheduler runtime, header bytes, and uncertainty. Better extraction accuracy need not imply better queue outcomes.

## What remains a later design decision

The baseline is specified separately and has no RL. The learned-policy stage still needs a sufficiently sized independently labeled dataset, an agreed primary objective and constraints, the allowed observation/history, queue-size handling, and a training/evaluation protocol. A candidate is an ordinal supervised classifier followed by the same queue policies, then a masked DQN over eligible job choices. It must not be trained merely to imitate a fixed priority sorter and advertised as outperforming it.

For RL, keep all parameters as inputs and define a task objective explicitly. Because arbitrary weighted objectives were rejected, select a primary outcome with constraints or an ordered objective after reviewing baseline trade-offs. Standard scalar DQN cannot be promised to optimize an unspecified vector objective. Value-of-information scheduling and the possible JEV/JEPA direction follow only after class-based validation.

## Status

Research/design completed to this checkpoint; no detector, classifier, latent-extraction head, or RL model was trained or benchmarked in this work. Existing codec files were inspected only; file presence is not checkpoint-inference validation. The original pilot was archived with a byte-preserving manifest and a local Git recovery branch. The Monday notebook and new extraction pipeline remain implementation milestones.
