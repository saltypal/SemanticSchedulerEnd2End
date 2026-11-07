# Semantic Aware AI Scheduler

Satya Srinivas Paladugu [BL.AI.U4AID24063]

Nandini Padavala [BL.SC.U4CSE24133]

Jahnvi KV [BL.SC.U4ECE24034]



## Abstract

We study how a receiver should choose the next complete image message for a single decoder when multiple user devices produce observations with different importance, mission relevance and repetition. Each message pairs a scheduling header with a complete codec payload. A lightweight policy reads the received parameters and causal queue history before dispatch. It does not reconstruct every image to decide which one deserves reconstruction.

The work includes semantic parameter design, detector comparisons, offline VLM labeling, frozen-encoder SPI classification, a 6000-image development cache, dynamic multi-device simulation and development evaluation of fourteen learned scheduling methods. The SPI pipeline has test macro-F1 0.59997 against pseudo-labels. The incomplete broader encoder comparison has a development leader at 0.6336517065, a different model and evaluation role. The scheduler analysis reports LambdaMART penalty 3.2183 against SPI-first 3.374083 and FIFO 5.822023. These are development measurements under a simulated waiting objective, not verified emergency effectiveness.

# I. Computational constraints and evaluation scope

The intended training and validation program remains constrained by the available compute, memory, storage and execution budgets. Teacher inference, repeated image-feature extraction, token-feature banks, classifier searches and scheduler interaction training compete for practical resources. The parameter comparison stopped during ConvNeXtV2 extraction after storage exhaustion. Earlier runs also encountered parsing, token-mask and split-preparation faults. Those are distinct causes and should not all be described as insufficient GPU arithmetic.

Some scheduler models have trained successfully. Fourteen learned methods completed development evaluation in the completed development runs. The bounded RL budgets and observed checkpoint regressions do not establish convergence, repeated-seed stability or a final deployment winner. Compute limits constrain the breadth and repetition of the evidence, but do not explain away weak results. Independent urgency references, appropriate objectives and matched ablations are also missing or limited. The protected final scheduler test remains unopened in the recorded evaluation. The SPI test cohort has already been exposed in earlier development and is not a newly untouched confirmation set.

## A. Project goal and research questions

Our goal is to reduce the waiting of important and mission-useful observations at a receiver bottleneck without losing track of messages that remain unselected. We seek an ordering policy over complete received messages. We do not claim that changing decoder order saves already transmitted bytes, allocates the radio, guarantees exact image recovery or detects every emergency.

The first question is whether semantic inputs improve dispatch over FIFO, round robin and simple SPI-first selection. The second is whether mission relevance, appearance novelty and scene arrays add measurable value beyond SPI. The third is whether learned context or temporal history earns its additional inference and training cost. The fourth is whether earlier selection exposes independently supported useful evidence rather than only matching a model-generated priority label.

# II. Semantic parameters and SPI development

## A. Project goal and the compute limitation

The receiver accepts complete image messages. Each message contains a parameter header and an encoded image payload. Several messages may wait for the same decoder. The scheduler chooses which complete payload to decode next so the operator receives useful, mission-relevant evidence early.

The project therefore concerns the order in which the receiver exposes information. It does not allocate radio resources. Changing the decode order does not reduce the bytes that have already arrived. A useful result must show better early evidence, lower waiting time for independently identified important observations, or another declared semantic objective. Throughput alone cannot establish that result.

DeepJSCC provides background for learned image transmission. It maps images to channel representations and reconstructs them after a noisy channel. Its image-transmission objective is different from our receiver queue decision. See Bourtsoulatze, Kurka and Gündüz, Deep Joint Source-Channel Coding for Wireless Image Transmission, 2019 [1].

**Compute and completion disclosure.** Our experiments have struggled with model-loading cost, repeated image extraction, sequential labeling, feature storage and training orchestration. The parameter comparison also failed on preprocessing, split reconstruction and disk exhaustion. These are documented engineering failures. They are not evidence that the scheduler objective is impossible or that a model has poor accuracy. The wider scheduler experiments compete for the same practical GPU resources, but this report does not assign them an invented training duration or claim that compute is their only bottleneck. The available evidence supports a completed parameter pipeline and an incomplete broader model comparison.

The two-T4 Kaggle setup limited the teacher candidates we could test. A quantized 7B or 9B VLM was feasible within that workflow. Model weights are only part of GPU memory use. Activations, image tokens, generation state and framework buffers also consume memory. Quantization reduces weight storage; it does not remove inference cost. A 72B candidate was outside the approved practical comparison. Parameter count alone cannot prove a model fits or predicts correctly.

## B. Complete message, header and latent payload

![Header and payload system](D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/parameter_extraction_report/figures/system.png)

The diagram describes the receiver boundary. The transmitter has two jobs. The image codec creates the representation needed for reconstruction. The parameter extractor creates the small observations used for scheduling. The receiver first obtains a complete message, validates its metadata and payload compatibility, then places it in the ready queue. One decoder processes one selected payload at a time.

For example, messages A, B and C can arrive before the decoder is free. Their arrival order can be A, B, C while their decode order is C, A, B. The scheduler reads the available header and queue context before making this decision. It must not read evaluator labels or reconstruct every image first to decide which image deserves reconstruction.

The payload format depends on the codec. A digital semantic codec can serialize quantized latent values, codebook indices or entropy-coded bytes. A learned analog JSCC system can instead produce complex channel symbols. In that case, received latent samples can form an internal decoder job after reception. Such samples do not become an ordinary packet merely because we call them a payload. The transport and codec adapter must define dtype, shape, model version and framing. Our message interface abstracts this distinction; it does not resolve physical-layer implementation by itself.

One complete scheduler job also need not equal one IP packet or one OFDM symbol. A large latent may require lower-layer framing before the receiver assembles it. Our experiment's one-shot condition means no application-level retry is introduced by this scheduling mechanism. It does not demonstrate lossless delivery, one-symbol transmission or arbitrary compression.

## C. Why the scheduler needs parameters

An opaque latent is difficult to interpret before its decoder runs. A source identifier, file size or arrival time does not tell the scheduler whether the observation matches the mission, repeats earlier observations or depicts an urgent problem. Parameters provide a common input contract across codecs and image sizes.

The parameters are a compact description of selected evidence, not a complete representation of the image. An image of one person in a bedroom and an image of one person trapped near a vehicle may share counts and box fractions. This is why geometry alone is inadequate. SPI, mission relevance and history provide different information, and their benefit must be tested rather than assumed.

| Parameter group | Question it addresses | Why we retain it | Literature link and actual scope |
|---|---|---|---|
| SPI | What intervention need does the image appear to depict? | Gives the scheduler an image-level assessment beyond object counts | CrisisMMD and MEDIC motivate crisis-image tasks. Neither gives us the four-class SPI truth used here. Our urgency rubric is project-defined. |
| Context relevance | Does this image match this UE's assigned mission? | A severe-looking image can be irrelevant to the operator's task | CLIP supports image-text comparison. It does not establish emergency importance. |
| Transmitter novelty | How different is this image from comparable prior observations at this UE? | Helps distinguish repetition from appearance change | Relevance and diversity objectives motivate considering history. Our cosine-distance implementation is an engineering proxy, not a proof of new facts. |
| Capture metadata | When was the image captured, and what delay did the operator permit? | Supports age, deadline and slack calculations without a fabricated decay curve | Age-of-information research supports freshness as distinct from throughput. It does not determine our mission's allowed delay. |
| Scene evidence | Which configured entities were detected, how many, and at what image extent? | Preserves distinctions hidden by one scene summary scalar | YOLO-World and Grounding DINO support text-conditioned detection. They do not justify treating large boxes as urgent. |
| Availability and producer identity | Was the measurement actually produced, and by which configuration? | Prevents failure, missing vocabulary and observed zero from collapsing together | This is a measurement and software contract, rather than a new semantic theory. |

We removed rarity as a primary field. A rare concept in a small local history need not be urgent or useful. It also overlaps with novelty unless its reference population and purpose are defined. We did not replace it with arbitrary importance weights.

The exact joint design is our proposal. The papers motivate components; no cited paper proves that these fields are sufficient for semantic scheduling. The correct sufficiency test is whether the available fields distinguish the decisions the task requires, including counterexamples where similar headers correspond to different priorities.

## D. Image to embedding, and what the embedding buys us

![Image to CLIP representation](D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/parameter_extraction_report/figures/embedding.png)

An image is first oriented using EXIF metadata, converted to RGB, resized while preserving aspect ratio and padded to 224 × 224 with gray value 127. The CLIP processor applies its native normalization. We record this preprocessing because a different crop or normalization changes the representation.

For the retained CLIP ViT-B/32, a 224 × 224 image yields a 7 × 7 patch grid. There are 49 image patch tokens plus a class token. The transformer uses hidden width 768. The pooled image representation passes through the visual projection to produce 512 values. L2 normalization gives a unit vector z.

The text encoder maps a mission description into the same 512-dimensional comparison space. For unit vectors z and t, cosine similarity is zᵀt. A cosine of 0.4 is a similarity score, not a 40 percent probability. These mechanics follow Radford et al., Learning Transferable Visual Models From Natural Language Supervision, 2021 [2].

The vector helps because one image forward can support mission similarity, nearest-history lookup and an SPI head. We can cache frozen features and compare small heads without running the vision encoder for every training epoch. A 512-value float32 vector uses 2,048 bytes in storage. That does not mean we transmit all 2,048 bytes. The scheduler can receive derived scalar fields while the transmitter keeps the embedding.

A vector is not universally better than the image. Pooling discards spatial detail; CLIP can miss small objects, text, counts and fine-grained conditions. Its pretrained associations can also create misleading similarities. A fixed vector head offers cheap training and inference, while token attention can preserve more spatial information at greater storage and training cost. We compare both for this reason.

The CLIP vector is also not the reconstruction codec's latent. CLIP was not trained here to reconstruct the original pixels. We use it to extract scheduling evidence. An image codec has its own objective and representation.

## E. Context relevance and the UE paragraph

Literature basis [12].

Each UE receives an operator-defined mission. A building-assessment drone might have this paragraph: "Document visible damage to buildings for post-event infrastructure assessment. Do not infer occupancy, structural safety, cause, or rescue urgency." A household-inventory camera can have a different paragraph about furniture and appliances. Device type is provenance, not a priority weight.

We keep the paragraph as the mission specification and derive a supported observable description for embedding. The current UE registry stores both `paragraph` and `conditions`. Its documented extraction path embeds the explicit condition text, rather than silently assuming the entire long paragraph enters CLIP unchanged. A building mission uses text such as "a photograph documenting visible damage to buildings." That distinction matters for reproducibility.

For a single condition, relevance is R = zᵀt, with values in [-1, 1]. Suppose z = [0.6, 0.8] and t = [1, 0]. Both have unit norm and R = 0.6. If a second mission has t = [0, 1], the same image gets R = 0.8. Relevance changes with the mission even when the image and its SPI do not change.

For a mission with several observable conditions, the aggregation rule must be declared and versioned. A maximum across clauses means that one matching clause is enough; an average asks a different question. Long mission paragraphs also need a token-length policy. Negations and constraints in natural language are not guaranteed to behave as logical rules in CLIP. We therefore retain the original paragraph, the actual encoded text and the producer signature.

Mission-text encoding is a reusable setup cost. It should not be charged as a new image inference on every frame unless the mission actually changes. A changed mission invalidates cached relevance, but not the image-dependent embedding or scene measurements.

## F. Novelty and the transmitter knowledge base

The transmitter keeps a causal history for each UE and mission. It queries history before committing the new observation. The scheduler cannot treat a record as previously observed merely because it will appear later in an offline dataset.

For a comparable unit embedding z and history H, the implemented appearance novelty is N = [1 - max(zᵀh, h in H)] / 2. A repeat with maximum similarity 1 gives N = 0. A maximum similarity 0.8 gives N = 0.1. A maximum similarity -0.2 gives N = 0.6. The formula maps the cosine range to [0, 1].

This is appearance novelty relative to one UE's stored observations. It does not establish a newly discovered building, a newly damaged building or additional evidence from another viewpoint. A different angle can change the embedding without revealing anything useful. A small but important new crack can leave global embeddings almost unchanged. Cross-UE repetition also escapes separate UE histories unless a later mechanism shares information.

The current `SenderMemory` and UE-memory factory default to 64 FIFO entries. Earlier planning proposed 128. The executable default is 64 unless the caller supplies a different capacity. At 512 float32 values, the embedding component uses 128 KiB for 64 records or 256 KiB for 128, before Python and provenance overhead. Exact lookup over such a small history is a reasonable baseline. An approximate nearest-neighbor service is not automatically an improvement.

Cold start is explicit. The minimal extractor uses N = 1 with a `cold_start` flag when a valid embedding exists and history is empty. That value is a convention, not measured novelty. If history exists but no embedding has a compatible dimension and producer signature, novelty is unavailable. It must not silently become either zero or maximum novelty.

The operation order is query, produce the bundle, serialize successfully, then commit to the transmitter KB. This sender KB is different from receiver successful-decode history. A receiver can separately track what it has successfully delivered. Failure and success semantics must remain explicit at that boundary.

## G. Capture metadata and freshness

We retain a trusted capture timestamp and an operator-supplied allowed delay. At decision time t, image age is t minus capture time. A deadline is capture time plus allowed delay. Slack is deadline minus t. These quantities answer different questions from visual severity.

For example, an image captured at time 100 seconds with a permitted delay of 20 seconds has deadline 120 seconds. At time 115 seconds, age is 15 seconds and slack is 5 seconds. No learned urgency interpretation is needed to calculate them.

Kaul, Yates and Gruteser, Real-Time Status: How Often Should One Update?, 2012 [3] motivates treating information age as its own communication quantity. Classical age of information at a receiver refers to the generation time of its freshest delivered update. The age of an individual queued image is related but is not automatically that receiver process.

Many source images lack verified capture timestamps. Download time and social-post time must not silently replace capture time. Missing or untrusted timestamps remain unavailable. We do not invent an exponential freshness decay or claim a universal delay budget. A scheduler can use declared age or slack alongside semantic evidence.

## H. Scene measurements and why we retained YOLO-World

Literature basis [4] and [5].

![Scene extraction](D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/parameter_extraction_report/figures/scene.png)

YOLO-World accepts an ordered vocabulary through the implementation's class configuration. We freeze that dictionary, checkpoint and preprocessing, then extract category labels, confidence scores and bounding boxes. The detector does not receive a free-form prompt asking it to explain urgency.

The current dictionary contains the 80 COCO categories plus five object extensions, building, bridge, utility pole, traffic sign and animal. Eight region terms complete the 93 slots: road, water, debris, smoke, flame, snow, ground and grass. This is a configured vocabulary, not all possible visual concepts. It does not guarantee reliable recognition of every listed hazard. Generic animal and traffic-sign entries can overlap with specific categories, so summed counts cannot be described as unique physical entities.

For class c, the count is the number of retained class-c detections. Largest-box fraction is max[(x₂-x₁)(y₂-y₁)/(WH)] over that class's retained boxes. A box 100 pixels wide and 80 pixels high in a 400 × 200 image has fraction 8,000/80,000 = 0.1. This measures rectangular image extent. It does not measure segmented object area or physical distance.

The class-specific arrays preserve more meaning than global counts. Two images can both contain four detections while one has four chairs and the other has people and vehicles. The scheduler can learn different associations from those vectors, provided its training targets support them. It must not use the rule that more people, more boxes or larger boxes always means higher priority.

We distinguish object-count semantics from region extent. The array contract has count availability for 85 object slots and does not pretend that several water boxes are several independent floods. A successful zero detection differs from a failed detector and from a category outside the vocabulary. The cache also records saturation when the detection cap may have truncated results.

The historical compact scene summary contained total instances, category count, largest box fraction and union coverage. Exact union coverage avoids double-counting overlapping rectangles. The expanded cache adds per-class arrays and retains raw boxes for auditing. We do not compute a relationship knowledge graph, depth map, monocular distance estimate, REBEL relation graph or graph entropy in this final cached pipeline. The earlier nodes-plus-edges complexity proposal was not validated.

### Detector comparison and the actual selection basis

Cheng et al., YOLO-World, CVPR 2024 [4] motivates efficient open-vocabulary detection. Liu et al., Grounding DINO [5] motivates grounded text-conditioned detection. Neither paper gives an urgency score or proves the superiority of our chosen deployment settings.

Our measured grid used the same 300 MEDIC images and a fixed 16-noun vocabulary. Twenty YOLO settings completed. Thirty-four of 56 Grounding DINO settings produced outputs, but the last of those had only 204 images and was excluded from complete-setting comparisons. The study did not contain reference bounding boxes.

![Detector comparison](D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/parameter_extraction_report/figures/detectors.png)

| Operating point | Median call time | Mean exact-vocabulary boxes | Images with zero such detections |
|---|---:|---:|---:|
| YOLO confidence .25, IoU .5, size 640 | 19.00 ms | 1.99 | 40.0% |
| Grounding DINO box .35, text .25, IoU .5, size 640 | 368.69 ms | 3.47 | 13.7% |
| Grounding DINO box .35, text .35, IoU .3, size 640 | 369.64 ms | 3.52 | 12.7% |

YOLO provided lower detector-call latency and fixed canonical category labels in this run. Grounding DINO produced more exact-vocabulary boxes and fewer empty images, but also phrases that required explicit vocabulary filtering. More boxes do not establish better recall, and fewer boxes do not establish better precision. We chose YOLO as the lower-cost engineering baseline, not the proven accuracy winner.

The measured latency includes detector-call processing and synchronization, excludes file and model loading, and includes observed cold first calls. It is not a warmed CPU/GPU end-to-end comparison. The 93-class cache is a later configuration. The 16-noun experiment does not establish its accuracy or identical latency.

The retained YOLO settings are confidence 0.25, NMS IoU 0.5, input size 640 and maximum 300 detections. On the 300-image grid, caps 100, 300 and 600 retained identical detections; the maximum observed was 26 boxes. Size 960 increased latency and box count without annotated evidence that those boxes were more accurate. These findings justify retaining a baseline while reporting uncertainty, not declaring a universal optimum.

## I. Header size, arrays and the scheduler input

The repository contains several protocol generations. The historical minimal core is 28 bytes. The scalar scene extension makes 35 bytes. The later 93-class scene-array block is 369 bytes on its own. It must not be described as fitting inside the old 35-byte format.

The expanded scene block carries 93 uint16 count slots, with the eight region counts marked unavailable, 93 uint8 area fractions, availability masks and producer/vocabulary identification fields. The learned scene feature representation has 374 components: 93 transformed count slots, 93 area slots, two sets of 93 masks and two status values. Masked region counts remain unavailable. Log1p(count) controls numeric scale without assigning importance weights.

Combining the scene block with another core requires a declared combined schema and its own measured byte count. JSON audit records, embedding storage, wire bytes and codec payload bytes are different quantities. The payload size still depends on the chosen image codec and operating point. The scene experiment does not establish that size.

The classifier label index and wire scheduling-class index also differ. The student classifier uses Low, Medium, High, Critical as indices 0, 1, 2, 3. The historical minimal wire class uses Critical, High, Medium, Low as 0, 1, 2, 3. Use label-aware conversion. Copying integers directly reverses their meaning.

The scheduler should receive the declared runtime fields and availability masks. Audit references, teacher targets, source-dataset identity and split membership must stay outside learned priority inputs unless an explicitly justified experiment says otherwise. Availability is useful, but a model can learn dataset or failure shortcuts from it. Missingness-matched controls are therefore relevant.

## J. SPI target and the choice of four classes

Dataset context [6] and [7]. The four ordinal classes are our project rubric, not a label taxonomy supplied by those papers.

SPI is our image-level prediction of depicted intervention need. It is not a validated measurement of an operator's real action urgency. The image may show severe damage after everyone has left. It may omit a person who needs immediate help just outside the frame. Context, time and mission remain necessary.

| Level | Proposed annotation meaning | Boundary to preserve |
|---|---|---|
| Low | No visible problem requiring intervention, or routine/low-impact content | Low does not prove that the wider situation is safe |
| Medium | A visible problem that warrants attention without clear immediate severe danger | Do not force every nontrivial incident into High |
| High | Serious visible damage or disruption supporting prompt attention | Severity alone does not establish an immediate life threat |
| Critical | Clear visible evidence supporting immediate severe intervention need | Do not invent hidden victims, imminent collapse or a dam failure |

These are four ordinal bins chosen for the project. They offer more distinction than binary incident detection while keeping annotation and a small classifier manageable. We do not claim that a cited dataset establishes these exact four universal urgency levels. Their separability and correctness require evaluation. A failure to learn Medium cannot be repaired by renaming examples to balance the histogram.

### Which images we used and why

The final labeling cohort selected 4,000 CrisisMMD images. The usable classifier cohort contains 3,933 available pseudo-labels and excludes 67 unavailable assessments. CrisisMMD offers incident imagery with text and humanitarian annotations, which makes it useful for studying crisis-related image parameters. Alam et al., CrisisMMD, 2018 [6] describes the multimodal dataset. 7 [7] offers broader disaster-image tasks and includes material overlapping CrisisMMD. They must not be counted as independent sources without overlap checks.

Earlier development used a 300-image LADI/MEDIC cohort, 220 LADI-v2 and 80 MEDIC images. Native labels described objects, damage, informativeness or humanitarian categories. They were not direct SPI urgency truth. TREC-IS post-level priority was discussed, but we did not convert its text priority into verified image-only urgency labels for this classifier.

We restricted SPI fitting to a crisis-image cohort because that was the available labeling and task context. This is a scope restriction, not evidence of generalization. The image dataset later adds 1,000 Open Images and 1,000 Places365 images to the 4,000 CrisisMMD images. Those extra sources broaden scheduler-development inputs. Applying the crisis-trained SPI classifier to them does not establish accuracy on routine scenes, screenshots, insurance claims or arbitrary phone photographs.

![SPI class support](D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/parameter_extraction_report/figures/support.png)

| Class | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Low | 782 | 419 | 273 | 1,474 |
| Medium | 339 | 149 | 200 | 688 |
| High | 899 | 445 | 325 | 1,669 |
| Critical | 48 | 24 | 30 | 102 |
| Total | 2,068 | 1,037 | 828 | 3,933 |

Class support improved compared with the earlier 1,000-image experiment, where Medium had only two training examples. Class balancing cannot manufacture independent evidence or correct wrong labels. The 102 Critical examples also limit confidence in per-class estimates.

## K. VLM labeling, comparisons and hallucination

Teacher architecture reference [10]. Its capabilities motivate the tool, not the truth of each generated urgency label.

The teacher runs offline. It sees the image and a task-specific system prompt, then returns the declared four-class assessment. We keep the raw output, parse status, image hash, teacher version, prompt version and generation settings. Runtime SPI inference uses the small student, rather than a VLM call for every scheduler decision.

This is supervised learning from AI-generated labels. It is not automatically reinforcement learning from AI feedback. RLAIF requires a reinforcement-learning procedure that uses AI feedback; training a classifier from generated class labels does not supply that mechanism.

The retrieved comparison folders contain outputs for Qwen2.5-VL-7B, Qwen3-VL-8B, Qwen3.5-9B, InternVL3-8B and the configured google/gemma-4-E4B-it candidate. Their presence establishes that diagnostic outputs were saved. It does not establish a complete balanced benchmark or that all model/version/prompt pairs passed the same gate. A model name containing effective parameter count also need not equal total stored parameters. Our resource reasoning must use measured memory and runtime, not the name alone.

The adopted 4,000-image label file mixes 2,540 earlier checkpoint assessments and 1,460 continuation assessments. The continuation used pinned Qwen2.5-VL-7B-Instruct, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, with thinking disabled. This is not a single-teacher, single-prompt labeling run. The teacher protocol must remain attached to each image. Differences between protocols can confound class distributions and model learning.

The prompt defines visible evidence, the four levels and output structure. It asks the model to avoid unsupported causes, hidden victims and invented imminent events. Thinking-disabled generation reduced the risk of long reasoning consuming the output budget before valid JSON appeared. Earlier thinking-enabled attempts often ran to their token/time limits without a final parseable answer. A strict output parser distinguishes a valid class assessment from a failed execution. It does not establish that a valid answer is true.

**Hallucination remains an issue.** Earlier teacher diagnostics included unsupported dam or bridge claims. Removing explanations does not remove incorrect labels. Distillation can copy those errors into the student. We therefore cannot write that hallucination would not be an issue. We can write that offline labeling keeps VLM latency out of runtime, retains auditable evidence and permits bounded reviews before labels are reused.

Useful controls include unchanged-image repeatability, contradicted-caption tests, images with no visible emergency, clearly distinct intervention levels, unsupported-detail rejection and abstention/failure accounting. These controls must not become the same examples used to tune every prompt and then be advertised as a fresh final test. Native annotations can verify supported object/damage claims, but cannot validate invented urgency levels by themselves. Independent urgency review remains unresolved.

The teacher's job must also stay separate from the proposed JSON audit. Asking a VLM whether an image agrees with a parameter JSON is a model-assisted consistency check. A yes answer does not prove that the JSON captures all decision-relevant information. Omitted critical facts require their own checks, and numeric predictions need interpretable definitions before a VLM can assess them.

## L. Encoders and classifiers we compared

The four encoder candidates test distinct representations. CLIP offers an image-text space reusable for relevance. SigLIP2 supplies another pretrained visual-language representation. DINO supplies self-supervised visual features. ConvNeXtV2 supplies a convolutional representation. Their role is frozen feature extraction in this study, not full encoder fine-tuning.

Feature compatibility is strict. A head fitted on a SigLIP2 representation must receive the same encoder, revision, projection and preprocessing at inference. CLIP vectors cannot replace it just because both are vectors. The complete inference pipeline still accepts a raw image, then runs the matching encoder and head.

| Classifier family | Why compare it | Main limitation |
|---|---|---|
| Logistic regression | Cheap linear decision boundaries and a useful transparent baseline | Cannot model nonlinear feature interactions |
| Ridge classifier | Regularized linear reference | Scores are not calibrated probabilities |
| Linear SVM | Another regularized linear boundary | Decision scores do not equal probabilities |
| RBF-SVM | Nonlinear boundaries without a large neural head | Support-vector storage and inference can grow |
| kNN | Tests whether nearby feature vectors share the label | Retains fitting data and has query cost |
| Random Forest / Extra Trees | Nonlinear tabular baselines | Dense pretrained embeddings may not favor axis-aligned splits |
| Histogram boosting | Small boosted-tree alternative | Needs enough label support and tuning |
| MLP | Nonlinear learning from pooled vectors | May overfit a small, imbalanced label set |
| Mean-MLP | Pools token features before classification | Loses token-specific interactions |
| Gated attention MIL | Learns a weighted aggregation over image tokens | Attention weights are not verified object explanations |
| Single-query cross-attention | Learns a query that aggregates spatial tokens for the class decision | More computation and masking complexity |

Nine vector families plus three token families give 12 heads per encoder and 48 nominal pairs. The class-weighted setting excludes unsupported class-weighted kNN, leaving 44 supported pairs in that regime. The other regimes include original, random oversampling and moderate undersampling. Oversampling duplicates training rows only; it must not alter validation/test or claim new independent examples.

Linear and tree estimators do not share a 15-epoch limit. The saved setup uses up to 1,000 logistic solver iterations, 10,000 for Linear SVM, 100 trees for forests, 50 boosting iterations and up to 100 sklearn-MLP iterations. Neural token heads allow up to 30 epochs with patience five on fold validation macro-F1. Final neural fitting uses a training-fold-derived epoch count. An observed 15 epochs can be a selected duration, not a universal training cap.

## M. Results and the distinction between a deployed model and a partial leader

### Completed SPI classifier

The SPI extraction pipeline trained CLIP plus class-balanced logistic regression using 2,068 training rows. It compared C = 0.1, 1 and 10 using the 1,037 validation rows. C = 10 won. The model then received one evaluation on the saved 828-image test split.

| Metric | Result |
|---|---:|
| Validation macro-F1 | 0.60245 |
| Test macro-F1 | 0.59997 |
| Test accuracy | 64.86% |
| Low recall | 79.49% |
| Medium recall | 36.00% |
| High recall | 69.54% |
| Critical recall | 73.33% |

![SPI classifier recalls](D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/parameter_extraction_report/figures/mini_recall.png)

This is the selected model actually used to populate the 6,000-image image cache. It was selected within a small logistic-regression experiment, not declared the winner across every encoder and head. The four-output linear head has 4 × 512 coefficients plus four intercepts, 2,052 scalar parameters. The encoder dominates its compute cost.

The dataset archive contains all 6,000 images, all 6,000 available SPI predictions and all 6,000 scene records. Its predicted distribution is Low 3,189, Medium 1,053, High 1,539 and Critical 219. That is a distribution of model outputs, not independent truth. The archive SHA-256 is `0F8C6186B09DCF6AB87911DF642612919377157BB1585082BED1C93B1506FE2C` and its size is 941,352,960 bytes.

### Full comparison, partial results

![Completed comparison](D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/parameter_extraction_report/figures/comparison.png)

The earlier 1,000-image study completed 188 supported configurations. Its development-selected SigLIP2 plus class-weighted RBF-SVM had test macro-F1 0.5388 and zero Medium recall. Every one of its 48 selected pair results also had zero Medium recall. That result motivated addressing label support rather than assuming a larger head would solve the problem.

The 4,000-image continuation labeled the cohort but failed the full matrix when SigLIP2 encountered an image with no unpadded token centers. A subsequent run failed because the setup regenerated development splits and then compared them with the saved assignments. We corrected preparation to restore the recorded splits directly.

Version 11 progressed further, but Kaggle storage filled after about 97 minutes. The original-balancing regime saved all 12 heads for CLIP, all 12 for SigLIP2 and all 12 for DINO. ConvNeXtV2 failed during feature extraction. The other balancing regimes, full final selection and full final test evaluation did not complete.

The development leader among those 36 saved candidates is CLIP plus single-query cross-attention, macro-F1 0.6336517065. It is a leader in a partial comparison. Its test result is not supplied here because the full final evaluation did not run. The per-candidate measurements are in `completed_comparison.csv` alongside this report.

![Saved training histories](D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/parameter_extraction_report/figures/learning.png)

These are the actual saved training-fold histories for the leader's selected learning-rate configuration. The loss and validation curves are separate quantities. Their purpose is to show training behavior and early stopping, not to replace independent evaluation. We did not draw a fabricated smooth curve or assign the leader the SPI classifier's test score.

The dataset evaluation has already exposed the native test cohort to development. Later experiments can keep its labels out of fitting and selection, but cannot describe that same cohort as a newly untouched final test. A fresh independent cohort is needed for confirmatory claims.

## N. Costs, FLOPs and the practical bottleneck

The completed dataset profile measured raw-image-to-SPI inference with the same ten development images. CPU used three repetitions per image after ten warmups. GPU used ten repetitions per image after ten warmups, with synchronization.

| Scope | CPU | GPU |
|---|---:|---:|
| SPI pipeline p50 | 104.74 ms | 28.56 ms |
| SPI pipeline p95 | 112.46 ms | 38.84 ms |
| Measured calls | 30 | 100 |

These timings include the CLIP-plus-head SPI path. They are not the complete deployed header extractor with detector, mission lookup, KB updates and serialization. Host/backend details remain in the archived profile records. The older detector timings use a different measurement scope, so adding their p95 values cannot produce a valid full-pipeline p95.

We count two floating-point operations per multiply-accumulate when that convention applies. Neural profiling reports supported operators and unsupported operations separately. Quantization does not make arithmetic zero. Scikit-learn iteration counts and wall time are not FLOPs. Unsupported FLOPpy estimators remain unmeasured or partial. Encoder extraction, head search, teacher labeling and deployment cost must be reported separately.

Storage was a demonstrated bottleneck. Global vectors are small, but token arrays are much larger. A single CLIP token tensor with 49 tokens and width 768 requires 150,528 float32 bytes per image before compression. At 3,933 images that is about 565 MiB. Higher-resolution token grids and multiple checkpoint copies multiply storage. Feature shards, consolidated banks, duplicated all-feature checkpoints, downloaded encoder weights and images can fill the runtime even when the classifier heads themselves are small.

The recovery should persist one verified feature bank per encoder, avoid redundant full-bank copies, reserve space before extraction and package successful heads and metrics incrementally. A failed final notebook cell must not be the only route to saving a trained model. These are concrete changes justified by the disk failure. We do not attribute that failure to poor model architecture or claim that the failed run established a final winner.

## O. What the parameter experiment establishes, and what remains to test

The completed pipeline establishes software extraction and cache coverage for the acquired dataset pool. It establishes a classifier's agreement with pseudo-labels. It establishes observed detector operating behavior and the runtime costs measured under their stated scopes.

It does not establish universal scene understanding, independently verified emergency urgency, complete cross-view novelty or a scheduling gain caused by every parameter. A successful serialization test establishes compatible bytes, not semantic correctness. A VLM consistency check establishes a model's judgment, not ground truth.

The parameter validation should ask distinct questions. Does each measurement match its definition? Does it remain available across sources? Does it add information beyond SPI and relevance? Do identical facts with different appearance fool novelty? Can different facts produce similar headers? Does quantization change rank decisions? Does a mask reveal dataset identity rather than useful semantics?

Useful scheduler comparisons hold the message pool, arrivals, decode costs and failures fixed while varying the permitted parameters. Compare SPI alone, relevance alone, novelty alone, scalar scenes, class-specific scenes and combinations. Train/validation separation must include duplicates, parent images and temporal sequences. Report policy effects with and without transmission/extraction overhead. Independent evaluator facts must stay outside the runtime header.

Delay by class is useful but not sufficient by itself. A policy could lower predicted-Critical delay by trusting a biased classifier. Independent semantic coverage and missed-useful-evidence measurements are needed to assess whether the earlier outputs actually help the mission. A10 can measure average normalized distinct-evidence coverage over the first ten decode opportunities when defensible fact correspondence exists. It must not count repeated views as distinct facts merely because their file IDs differ.

The immediate deliverable is therefore a reusable, auditable parameter dataset and a declared model. Completing the broader model comparison and independently validating urgency remain separate tasks. The report preserves those boundaries so the scheduler paper can state what the evidence supports.

## P. Mechanisms and additional primary references

Carbonell and Goldstein, MMR, 1998 [8] combines query relevance and dissimilarity from already selected documents. This motivates keeping context relevance and redundancy separate. Our image-history novelty is an adaptation, not the complete MMR algorithm.

Maatouk, Assaad and Ephremides, The Age of Incorrect Information [9] combines content mismatch with its duration. It motivates considering content, receiver knowledge and time jointly. Its source/channel model and threshold-policy optimality do not establish optimality for our image scheduler.

Qwen2.5-VL Technical Report, 2025 [10] describes a visual encoder, spatial merger and language decoder with structured-output capability. That supports automated labeling, but does not validate our urgency taxonomy or project labels. MEDIC's paper [11] describes four annotation tasks, not four urgency classes.

Representation references are CLIP, 2021 [12], SigLIP 2, 2025 [13], DINOv2, 2023 [14], and ConvNeXt V2, 2023 [15]. They motivate different pretrained feature families. Their published benchmark rankings do not predict our SPI winner. Read the actual DINO checkpoint/fallback from the encoder identity, rather than treating the short name `dino` as an exact model specification.

Ilse, Tomczak and Welling, Attention-based Deep Multiple Instance Learning, 2018 [16] motivates learned aggregation over instance features. Our token heads apply that idea to image patches. Attention weights are not verified emergency localizations.

### What the classifiers calculate

The logistic head computes a = Wz + b. The input z has 512 components, W has shape 4 × 512, and b has four values. Softmax normalizes the four logits. Class-weighted cross-entropy changes the influence of training examples, not their independent information content. Scores remain uncalibrated unless separately calibrated.

A token head receives a matrix H with one row per patch. Mean-MLP averages the rows. Gated attention MIL learns normalized weights over valid rows and forms a weighted sum. Single-query cross-attention learns a query q, keys K and values V, then computes softmax(qKᵀ / sqrt(d))V before classification. The padding mask excludes invalid image regions. The empty-mask correction retains patches intersecting content when no patch center lies inside it. This is a handling correction, not evidence of improved accuracy.

A linear head tests whether the representation already separates labels. RBF-SVM, trees and MLP test nonlinear decision functions. Token attention tests whether global pooling hides useful spatial distinctions. This is why the comparison includes several mechanisms rather than only bigger heads.

For each class, precision = TP / (TP + FP), recall = TP / (TP + FN), and F1 = 2PR / (P + R). Macro-F1 averages the four class F1 values equally. Accuracy averages correctness across images and can conceal minority-class failure. The dataset test's Medium recall of 0.36 corresponds to 72 correct predictions among 200 pseudo-labeled Medium images.

### Numerical results for all 36 completed original-setting pairs

These are development macro-F1 values, not final test scores.

| Head | CLIP | SigLIP2 | DINO |
|---|---:|---:|---:|
| logistic | 0.4398 | 0.4401 | 0.5317 |
| ridge | 0.4517 | 0.4624 | 0.5249 |
| linear_svm | 0.4862 | 0.5490 | 0.5783 |
| rbf_svm | 0.6141 | 0.6003 | 0.5796 |
| knn | 0.5832 | 0.6031 | 0.5686 |
| random_forest | 0.4193 | 0.4250 | 0.4130 |
| extra_trees | 0.4386 | 0.4424 | 0.4364 |
| hist_boost | 0.5386 | 0.5320 | 0.5092 |
| mlp | 0.5896 | 0.6152 | 0.5835 |
| mean_mlp | 0.6120 | 0.6031 | 0.5592 |
| gated_abmil | 0.6079 | 0.5717 | 0.5661 |
| single_query_cross_attention | 0.6337 | 0.6263 | 0.5710 |


# III. Receiver policy and scheduler training

## A. System boundary and policy

Each offered message contains semantic parameters and a complete opaque digital payload. Only complete messages become ready. The scheduler reads parameters and permitted context. It does not inspect images or latent contents.

At decision time t, the scorer evaluates every ready candidate i using s_i = f(candidate parameters, queue context, permitted history). It selects the candidate with the largest score. Candidate identity routes the selection but must not become a learned semantic feature.

Processing is nonpreemptive. Messages can arrive while the server works. They become candidates at the next decision. A new urgent arrival does not interrupt the current job. The scheduler takes a fresh snapshot after processing ends, rather than committing to a permanent sorted queue.

| Step | Receiver behavior |
| --- | --- |
| 1 | Admit complete messages whose arrival time has been reached |
| 2 | Build an immutable ready queue and permitted history snapshot |
| 3 | Score the candidates with the selected policy |
| 4 | Dispatch exactly one ready message |
| 5 | Advance the simulated server while admitting arrivals |
| 6 | Record waiting time and processing outcome |
| 7 | Update permitted history and select again |

### One shot operation and history

Each message receives one attempt and no automatic retry. Successful history describes evidence processed earlier at the receiver. It does not imply a retransmission protocol. For an all success control, every completed job enters that history. Outcome stress tests can assign saved failures that consume service time without adding successful evidence.

The current environment retains up to 32 successful processed headers. Temporal windows contain up to three recent observations. The history consists of parameter records rather than reference labels or evaluator fact identities.

### Delay is the objective rather than the policy

A policy chooses the next candidate. Delay measures the operational effect of that choice. A ranker, a tree and an RL agent can all optimize the same waiting objective through different learning procedures. This separation lets us compare learning mechanisms without changing what success means.

## B. Parameters and transport

| Input | What it contributes |
| --- | --- |
| SPI | Four uncalibrated producer scores for Critical High Medium and Low |
| Context relevance | Numeric alignment between the image and its UE mission |
| Novelty | Difference relative to the transmitter knowledge base before committing this observation |
| Scene evidence | Retained detections normalized largest box areas and availability |
| Status | Missingness cold start and detector status |

The received vector contains 387 entries. Scene evidence contributes 374. Four SPI scores, relevance, novelty, six further availability indicators and one cold start indicator supply the remaining thirteen. Context builders derive queue and history representations from those entries.

### Scene vocabulary

All 93 categories enter the current training representation. They include the 80 COCO objects, five extra objects and eight region categories. The extra objects are building, bridge, utility pole, traffic sign and animal. The regions are road, water, debris, smoke, flame, snow, ground and grass.

Counts apply to 85 object categories. Region counts remain unavailable. Area means the largest detected bounding rectangle divided by image area. It is not segmentation coverage or physical distance. A person detection does not establish distress. A building detection does not establish damage. Many objects do not establish urgency.

Novelty follows a causal transmitter sequence. Query the existing knowledge base, calculate novelty, serialize parameters, then commit the observation. The same offered stream determines transmitter state across policies. The current observation cannot enter its own comparison history.

Mission relevance is an extracted numeric input. The current student does not read the mission paragraph. High novelty can describe an unimportant new photograph. A model must interpret novelty with priority and mission evidence.

### Communication cost

| Scene representation | Bytes |
| --- | --- |
| Original dense | 369 |
| Compact dense | 307 |
| Compact sparse | 29 plus four times the active category count |
| Ten active categories in sparse form | 69 |

Compact encoding preserves the stored measurements and masks under a preinstalled producer and vocabulary profile. These values cover the scene block only. SPI relevance novelty routing and timing add overhead. One combined binary envelope size has not been established.

We have not measured which categories are necessary. A smaller vocabulary requires a retrained comparison between no scene input, a compact subset and all 93 categories on identical splits and streams.

## C. Waiting objective and training references

Queue waiting time is dispatch time minus arrival time. Completion delay adds service duration. Our implemented D objective measures waiting until dispatch. A Low image entering an empty queue can legitimately wait less than a Critical image entering a burst. The goal concerns matched workload results rather than a strict class mean ordering in every episode.

| Reference class | Budget in simulated ms |
| --- | --- |
| Critical | 2 |
| High | 4 |
| Medium | 8 |
| Low | 16 |

These budgets are experimental assumptions. They are not published emergency response deadlines. The constant service control takes one simulated ms per job. Historical files retain their original simulation unit representation.

### Penalty

The penalty is ell(w,b) = w/b + max(w/b - 1, 0). Waiting is w and the class budget is b. The slope doubles after the budget. For Critical images, waits of 1, 2 and 4 ms produce penalties of 0.5, 1 and 3. A Low image waiting 4 ms contributes 0.25.

The episode score J_D is the mean penalty across offered messages. Lower is better. An unselected message contributes elapsed waiting up to the observation horizon. The score is dimensionless and must not be presented as a mean number of milliseconds.

### Supervision

The reference target is the extra penalty caused by waiting one additional service interval. It equals ell(w + 1,b) minus ell(w,b). Larger targets identify candidates whose postponement is more costly under the stated objective.

Choice models learn the best target candidates. Regression predicts target magnitudes. Pairwise losses learn their relative order. Training references remain separate from predicted runtime inputs.

The completed target depends on reference priority and waiting time. It does not independently reward every within class mission or novelty distinction. Scene evidence may improve reference priority prediction. Current results alone cannot prove a broader understanding of usefulness.

### RL reward

The environment computes reward as the negative change in total accumulated waiting cost divided by 100. The factor scales the reward. It does not change the preference for lower waiting cost. The selected message stops accumulating queue waiting at dispatch while other messages continue waiting.

## D. Conventional policies and learned scoring

| Policy | Selection rule |
| --- | --- |
| FIFO | Earliest arrival |
| Round robin | Rotate among sources |
| Seeded random | Random ready candidate |
| SPI FIFO | Highest predicted class then arrival |
| Relevance | Highest context alignment |
| Novelty | Highest novelty |
| SPI novelty | Priority then novelty |
| Predicted EDF | Earliest arrival plus predicted class budget |

These controls separate arrival order fairness chance and simple semantic signals. Predicted EDF uses assumed budgets derived from predicted classes. It does not represent real application deadlines.

### Linear and MLP choice

Linear choice learns a shared linear score from contextual features. MLP choice uses hidden widths 64 and 32 with ReLU activations. Both apply softmax over ready candidates and choice cross entropy. Best target ties receive equal target probability. Completely tied snapshots do not impose an arbitrary choice label.

### Delay regression and priority classifier

Delay Huber regression predicts the marginal penalty target with Huber loss and delta equal to one. This tests learning target magnitudes. The completed priority classifier uses four class cross entropy. It is not the earlier proposed binary useful next classifier.

### RankNet

RankNet predicts pair preference with sigmoid of the score difference. Binary cross entropy trains that preference. Our implementation weights strict pairs by normalized target gap, gives smaller weight to ties and bounds sampled pairs per snapshot. This adapts the neural pairwise mechanism in Burges et al [17].

### LambdaMART and listwise ranking

LambdaMART uses boosted trees to learn nonlinear tabular interactions. It is a strong comparison for numeric parameter inputs and follows the ranking family described by Burges [18]. The listmle arm uses a tie aware strict lower surrogate that penalizes lower target candidates outranking higher ones. It is a listwise adaptation rather than the original permutation likelihood objective [21].

## E. Set and temporal models

### Deep Sets RankNet

A shared encoder maps candidates to learned vectors. Mean pooling summarizes the queue. Another mean summarizes successful processed history. The scoring head combines the candidate, its encoding, pooled context, sizes and comparable distances. RankNet supplies the training objective.

The representation supports variable queue sizes and does not assign meaning to arbitrary array position. Candidate permutation must preserve identity associated scores. Deep Sets provides the shared encoder and invariant pooling foundation [19]. Our implementation uses mean pooling.

### Attention based set ranker

The attention model uses a 32 dimensional projection and two Transformer encoder layers with four heads. Candidates can exchange information before scoring. Pairwise loss trains selection order. The registry calls this set_transformer_ranknet. It is an attention based adaptation rather than a literal reproduction of every module in Lee et al [20].

### Temporal Deep Sets

A GRU encodes up to three recent observations in the temporal windows. The model combines those short ordered histories with candidate and queue context. This tests whether ordering adds value beyond pooled history. It does not establish long term event tracking or natural cross view matching.

### How scene information can contribute

SPI supplies a coarse priority assessment. Relevance identifies mission alignment. Novelty compares current observations with earlier transmitter observations. Detected categories and areas supply content context. The scorer can learn combinations but cannot recover information absent from those parameters.

For two High candidates, zero relevance could mean an off mission image, an extractor error or important evidence outside the device mission. A large object count alone cannot resolve those cases. The supervision must distinguish them. The current delay target is narrower than a full operational judgment of emergency usefulness.

Architecture and supervision are separate choices. Deep Sets defines representation. RankNet defines a loss. Attention defines interactions. RL learns through transitions and reward. A larger model cannot repair a target that omits the desired distinction.

## F. Reinforcement learning and execution

### Masked Double DQN

The online network selects the next legal candidate and the target network evaluates it for the TD target. Masks exclude messages outside the ready queue. Double Q learning separates action selection and evaluation to reduce action value overestimation [22]. Candidate conditioned scoring adapts the mechanism to a changing queue.

### Prioritized replay

TD prioritized replay samples stored transitions according to prediction error. Importance weights account for changed sampling probability. Replay priority controls learning samples rather than operational message urgency [23].

### Masked PPO

PPO learns an actor over ready candidates and a value estimate. The legal action mask prevents invalid selections. Its clipped objective limits the effect of large probability ratio changes during updates [24].

### Ranker warmstarted PPO

A trained ranker initializes the PPO actor. PPO then updates it using delay rewards. The comparison asks whether supervised initialization helps within the same bounded interaction budget. This is our hybrid training adaptation.

| Component | Completed scope |
| --- | --- |
| Supervised methods | 64 development episodes and 512 sampled decision snapshots per method |
| Epoch limit | 50 with patience eight |
| RL budget | 20000 action decisions per method |
| Model seed | 1729 |
| Validation | 15 traces across UE counts and loads |
| Managed runtime | Python 3.13.15 Torch 2.11.0 and two T4 GPUs per paired notebook |

A decision is one selection rather than an episode. Cached evidence avoids image producer retraining. The paired runs lasted about 2 hours 56 minutes and 2 hours 46 minutes. These are execution totals for their assigned computation. The budgets do not establish convergence or stability across seeds.

## G. Data and dynamic simulation

| Dataset | Images | Role |
| --- | --- | --- |
| CrisisMMD | 4000 | Disaster imagery |
| Open Images | 1000 | Everyday object diversity |
| Places365 | 1000 | Scene diversity |

The collection has 6000 images and 5952 raw unique images or cached embedding entries. Full pool split counts are 5376 training, 417 validation and 207 test. The assessable delay reference cohort has 3915 records. The final test remains unopened.

CrisisMMD supports humanitarian image analysis [6]. Its published labels do not automatically supply four class scheduling urgency. Current priority references are silver labels rather than independent operational urgency truth. The everyday sources broaden inputs but do not supply assessable urgency labels for the completed D comparison.

### Fifteen UE profiles

The device and mission templates are general phone, citizen incident phone, structural inspection drone, person search drone, flood drone, wildfire drone, road inspection drone, connected car camera, emergency vehicle camera, traffic CCTV, public space CCTV, river or underpass camera, industrial robot, agricultural drone and maintenance camera.

These are project templates rather than fifteen official 6G device classes. A normal phone can capture important evidence. A disaster drone can send a routine repeat. The device mission supplies context rather than automatic importance.

### Traffic and stream construction

Development validation includes 4, 12, 50, 60 and 100 UEs. The rate study includes 12, 50, 60 and 100. Fixed total load separates population from aggregate demand. Fixed per UE rates allow demand to grow with population. Burst settings change concentration. Joining and pausing alter achieved traffic, so intended and realized rates both matter.

The simulator saves arrivals, service durations and potential outcomes before comparison. Every policy receives the same realization. The transmitter knowledge base follows the offered sequence independently of receiver choices. Mixed image sampling supplies diversity. Mission sampling uses image text similarity. Neither establishes natural incident correspondence.

Capture time, extraction readiness, receiver arrival, dispatch and completion are distinct. Generated metadata must be marked simulated. Realistic stream construction requires supported sequences rather than unrelated images assigned invented physical correspondence.

## H. Available development results

The following fourteen methods completed development training and evaluation. Values are dimensionless J_D penalties. Lower is better. They are not mean waiting milliseconds.

| Model | Validation penalty |
| --- | --- |
| LambdaMART | 3.2183 |
| Delay Huber regression | 3.3573 |
| RankNet | 3.3642 |
| Deep Sets RankNet | 3.3819 |
| Tie aware listwise | 3.3859 |
| Temporal Deep Sets | 3.4480 |
| Attention based set ranker | 3.4520 |
| MLP choice | 3.5675 |
| Ranker warmstarted PPO | 3.6375 |
| Priority classifier | 3.8271 |
| Linear choice | 4.3392 |
| Masked PPO | 4.5935 |
| Masked Double DQN | 4.7099 |
| Double DQN with TD replay | 5.1757 |

LambdaMART has the lowest penalty among these learned models. Several supervised methods outperform the completed RL variants under the configured budget. Attention and short temporal history did not improve on the best simpler methods in this table. Warmstarted PPO performed better than the other completed RL methods.

The matched SPI-first comparison is reported in the subsequent results analysis. Aggregate improvements require class-specific safeguards and independent confirmation. One seed cannot establish stable superiority. Aggregate penalty alone cannot identify which class benefited or lost.

### Completed rate study

The CPU rate experiment produced 320 rows from 80 traces and four policies. It crossed population, fixed total or per UE traffic, burst and sampling settings. Policies were FIFO, round robin, SPI first and SPI novelty. It completed in about 44 seconds. Metrics were finite and event and class accounting conserved messages. This establishes a reproducible workload comparison rather than final scientific evaluation.

## I. Evaluation and evidence boundaries

Class tables must show offered selected and unselected counts, mean and median selected waiting, restricted mean wait and fraction selected within budget. p95 requires at least twenty selected observations in the implementation. Selected only means need unselected counts beside them.

Restricted waits use a 16 ms cap for messages with sufficient follow up. A message remaining unselected contributes the cap. This prevents undelivered images disappearing from the delay analysis. Snapshot nDCG uses declared gains and fixed queue context. It is separate from rollout performance.

Evidence metrics use declared fact identities. Exact copies can share an identity and receive no extra distinct credit. This construction does not establish that natural camera views show the same building.

### Validation requirements

Check empty queues, arrivals during processing, busy dispatch, duplicate and foreign completions, reset, nonfinite outputs and horizon censoring. Verify penalty arithmetic independently. Compare exported predictions and selected identities with the training backend. Keep reference labels, future arrivals and evaluator identities outside runtime observations.

### Remaining comparisons

Merge baseline and learned results on identical traces. Report each priority class with misses. Retrain no scene, compact scene and full scene variants on matched groups. Measure policy latency and context construction separately with hardware queue size warmup and counting convention. FLOPs and measured latency are different quantities.

Final uncertainty must respect independent source or duplicate groups. Queue seeds alone are not independent disasters. Freeze model and comparison choices before opening the final test. Separate broad image pool coverage from the narrower assessable priority cohort.

### Local evidence

The environment is Evaluation/semantic_learning_env.py. Learning models are Scheduler/training/semantic_learning_models.py. The vocabulary is parameters/scene_vocabulary.py. Transport is Evaluation/compact_scene_transport.py.

Development evidence is under experiments/master_study/runs/paired-launch-20261102-205933/checks/paired-check-20261102T220621Z. Rate outputs are under experiments/stream_scheduler/runs/gap-repair-20261102T221315Z/rate_cpu_full. The existing Kaggle map is parameters/KAGGLE_RUNS.md.

# IV. Updated results and strong baseline analysis

## A. Updated analysis of the completed development runs

## B. Scope
This is read only analysis of saved development results. No training or final test access occurred. Time values are simulated ms under the agreed unit convention. No statistical significance claim is made from queue repetitions.

## C. Strong baseline changes the conclusion
FIFO penalty is 5.822023. SPI first is 3.374083. SPI first reduces penalty by 42.05% relative to FIFO.
LambdaMART reduces penalty by 4.62% relative to SPI first. Only 3 of the 14 learned methods beat SPI first numerically. Small differences remain unconfirmed without paired per episode learned outcomes.

| Method | Penalty | Improvement over SPI first percent |
|---|---:|---:|
| lambdamart | 3.2183 | 4.62 |
| delay_huber | 3.3573 | 0.50 |
| ranknet | 3.3642 | 0.29 |
| deepsets_ranknet | 3.3819 | -0.23 |
| listmle | 3.3859 | -0.35 |
| temporal_deepsets_ranknet | 3.4480 | -2.19 |
| set_transformer_ranknet | 3.4520 | -2.31 |
| mlp_choice | 3.5675 | -5.73 |
| ranknet_warmstart_ppo | 3.6375 | -7.81 |
| priority_classifier | 3.8271 | -13.43 |
| linear_choice | 4.3392 | -28.60 |
| masked_ppo | 4.5935 | -36.14 |
| masked_double_dqn | 4.7099 | -39.59 |
| td_per | 5.1757 | -53.40 |

## D. Class trade offs in the matched baseline cohort
Counts pool the same 15 validation episodes. Restricted waits weight each eligible follow up observation. Selected mean excludes unselected messages and must be read with their count.

| Policy | Class | Offered | Unselected | Selected mean wait | Restricted wait | Within budget |
|---|---|---:|---:|---:|---:|---:|
| fifo | Critical | 75 | 9 | 21.441 | 10.066 | 0.173 |
| fifo | High | 1087 | 92 | 22.099 | 9.916 | 0.333 |
| fifo | Medium | 435 | 39 | 20.884 | 9.583 | 0.425 |
| fifo | Low | 846 | 69 | 19.282 | 9.555 | 0.528 |
| spi_fifo | Critical | 75 | 0 | 2.929 | 1.684 | 0.893 |
| spi_fifo | High | 1087 | 12 | 8.314 | 4.830 | 0.636 |
| spi_fifo | Medium | 435 | 16 | 15.005 | 6.487 | 0.634 |
| spi_fifo | Low | 846 | 181 | 34.582 | 10.679 | 0.394 |
| spi_novelty | Critical | 75 | 0 | 3.102 | 1.237 | 0.947 |
| spi_novelty | High | 1087 | 10 | 8.356 | 3.979 | 0.744 |
| spi_novelty | Medium | 435 | 15 | 14.347 | 5.494 | 0.690 |
| spi_novelty | Low | 846 | 184 | 30.883 | 10.075 | 0.433 |

## E. RL learning dynamics
- td_per: best observed checkpoint at 16000 decisions, penalty 5.1757; final checkpoint penalty 5.3716. Observed checkpoints do not prove convergence.
- ranknet_warmstart_ppo: best observed checkpoint at 2048 decisions, penalty 3.6375; final checkpoint penalty 4.1661. Observed checkpoints do not prove convergence.
- masked_ppo: best observed checkpoint at 20000 decisions, penalty 4.5935; final checkpoint penalty 4.5935. Observed checkpoints do not prove convergence.
- masked_double_dqn: best observed checkpoint at 10000 decisions, penalty 4.7099; final checkpoint penalty 5.4125. Observed checkpoints do not prove convergence.

## F. Rate study paired effects
Effects below compare policies on identical saved traces. Wins mean strictly lower waiting penalty. These traces are workload variations from one seed, not independent emergency events.
- round_robin: 80 paired traces; mean delta versus FIFO -0.02059; median 0.00547; wins 32, ties 5, losses 43.
- spi_fifo: 80 paired traces; mean delta versus FIFO -1.70285; median -0.97357; wins 74, ties 6, losses 0.
- spi_novelty: 80 paired traces; mean delta versus FIFO -1.62261; median -0.93560; wins 74, ties 6, losses 0.

## G. Interpretation and next experiments
The implemented target depends on reference priority and waiting. The current comparison cannot establish a distinct benefit from novelty or all 93 scene categories. Within class marginal costs often tie, leaving no training signal for a finer semantic order.
Keep LambdaMART, RankNet and delay regression as the principal learned candidates. Retain SPI first as the strong comparator. Attention and temporal complexity are not justified by the present aggregate scores.
Do not infer that replay prioritization is universally harmful. Its poorer result may reflect training distribution, optimization settings or reward target properties. Saved learning curves establish instability or regression only in this run.
Run matched no scene, compact scene and full scene ablations. Add SPI only learned models to distinguish better priority estimation from extra semantic reasoning. Keep the final test closed until these decisions are fixed.
Per episode learned outcomes, critical safeguards, measured inference profiles and feature ablations are not present in this retrieved evidence subset. Therefore no learned class benefit, confidence interval, Pareto deployment winner or feature importance is invented.
Training status seconds can include validation and concurrent resource contention. They are not pure training throughput measurements.

## H. Figures
![Learned comparison](D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/results_analysis/runs/analysis-20261104/model_comparison.png)
![RL curves](D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/results_analysis/runs/analysis-20261104/rl_curves.png)
![Load response](D:/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End/experiments/results_analysis/runs/analysis-20261104/load_response.png)

# V. Literature decisions limitations and next experiments

## A. Interpreting the current evidence

The current D objective rewards reference-priority-sensitive waiting. It does not independently reward every novel fact, mission distinction or important visual relation. Within a class, marginal costs often tie. A richer network may receive little training signal for distinctions its designer hopes it will learn. Under this target, the strong SPI-first baseline is expected to be competitive. The results support testing target sufficiency before enlarging the architecture.

Only three of fourteen learned methods beat SPI-first numerically in the aggregate analysis. LambdaMART's 4.62 percent advantage is an observed development difference, not a statistically confirmed advantage across independent incidents. Temporal Deep Sets is 2.19 percent worse than SPI-first here. This does not prove temporal information is useless. It means the tested history, objective, optimization and budget did not establish its benefit.

## B. What is inspired by papers and what we chose

| Decision | Literature mechanism | Project choice and validation boundary |
|---|---|---|
| Header and image payload | [1] supplies learned image transmission background | Our complete-job receiver interface and scheduling boundary are project choices. Analog JSCC symbols require a compatible adapter. |
| Mission relevance | [12] aligns image and text representations | Mission wording, aggregation and cosine interpretation are our choices. No urgency probability follows. |
| Novelty | Relevance-diversity retrieval and [9] motivate history and content dependence | Per-UE cosine distance is an appearance proxy. We do not implement the AoII optimal policy or claim fact identity. |
| Scene evidence | [4], [5] localize text-specified categories | The 93 terms, masks and geometry fields are our interface. Detection is not distress or severity. |
| Four SPI classes | [6], [7] motivate incident-image analysis | Four intervention tiers, abstention and class boundaries are project-defined. Native categories are not urgency truth. |
| Offline pseudo-labeling | [10] supports multimodal generation | Mixed teacher protocols and prompts need per-row provenance. Parser success does not validate a label. |
| Token aggregation | [16] motivates learned attention over instances | Patch attention is a classifier mechanism, not a validated emergency explanation. |
| Contextual ranking | [17], [18] learn ordering | Our waiting-derived targets, tie weighting and sample limits adapt those methods. |
| Variable candidate sets | [19], [20] motivate set processing | Our mean pooling and encoder-only attention paths are adaptations. |
| Sequential learning | [22], [23], [24] supply update mechanisms | Legal masks, queue transitions and waiting rewards are project-specific. Replay priority is not message urgency. |

## C. Next experiments and acceptance

First preserve the completed artifacts and freeze the exact source, feature registry, split identities and teacher provenance. Complete the resource recovery without regenerating labels or resetting useful results. Store each feature bank once, reserve disk space and checkpoint useful heads and metrics before the notebook's final cell.

Then compare no-scene, compact-scene and full-scene inputs, with SPI-only learned models as a control. Retrain on identical groups and evaluate on identical offered traces. Remove excluded fields from every derived history feature as well as the candidate input. A masked candidate field that survives in a pooled context would invalidate the ablation.

Independently audit the SPI rubric and ambiguous class boundaries. Expand representative training material only when its labels and source groups are defensible. Generalization needs routine and unfamiliar scenes with independently assessed urgency, not merely model predictions on broader source images. Keep unknown references unknown and distinguish label unavailability from Low.

Keep the protected scheduler final test closed until model and comparison decisions freeze. Use paired per-episode outcomes for learned comparisons and report class-specific misses. A queue-seed repeat is not an independent disaster. Explain the experimental unit before applying uncertainty estimates.

Measure extraction, serialization, queue-context construction, inference and decoding separately with matching hardware and workloads. Charge the full pipeline when claiming deployment cost. Do not add unrelated p95 timings and call the sum a measured p95. Report average waiting, restricted waiting, unselected counts and class safeguards alongside the aggregate penalty.

For the separate P study, qualify image-and-context teacher preferences before fitting the temporal ranker. For E, use independently supported fact correspondence. Training a D model does not establish a completed P preference experiment or E distinct-evidence result.

## D. Conclusion

We have built a parameter-producing and receiver-scheduling research pipeline, completed a SPI model and broad cache, and obtained bounded development results for fourteen schedulers. Simple semantic prioritization is already a strong control. The results support retaining LambdaMART, RankNet and delay regression for focused follow-up while testing whether richer semantic inputs and history earn their cost. Independent urgency validation, complete parameter comparison, repeated scheduler evidence and final confirmatory evaluation remain unfinished.


# References

[1] [Bourtsoulatze, Kurka and Gündüz, Deep Joint Source-Channel Coding for Wireless Image Transmission, 2019](https://arxiv.org/abs/1809.01733)

[2] [Radford et al., Learning Transferable Visual Models From Natural Language Supervision, 2021](https://arxiv.org/abs/2103.00020)

[3] [Kaul, Yates and Gruteser, Real-Time Status: How Often Should One Update?, 2012](https://www.winlab.rutgers.edu/~gruteser/papers/sanjitnew.pdf)

[4] [Cheng et al., YOLO-World, CVPR 2024](https://arxiv.org/abs/2401.17270)

[5] [Liu et al., Grounding DINO](https://arxiv.org/abs/2303.05499)

[6] [Alam et al., CrisisMMD, 2018](https://arxiv.org/abs/1805.00713)

[7] [MEDIC](https://crisisnlp.qcri.org/medic/)

[8] [Carbonell and Goldstein, MMR, 1998](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)

[9] [Maatouk, Assaad and Ephremides, The Age of Incorrect Information](https://arxiv.org/html/2012.13214v3)

[10] [Qwen2.5-VL Technical Report, 2025](https://arxiv.org/html/2502.13923v1)

[11] [MEDIC's paper](https://arxiv.org/html/2108.12828v4)

[12] [CLIP, 2021](https://proceedings.mlr.press/v139/radford21a.html)

[13] [SigLIP 2, 2025](https://arxiv.org/abs/2502.14786)

[14] [DINOv2, 2023](https://arxiv.org/abs/2304.07193)

[15] [ConvNeXt V2, 2023](https://arxiv.org/abs/2301.00808)

[16] [Ilse, Tomczak and Welling, Attention-based Deep Multiple Instance Learning, 2018](https://proceedings.mlr.press/v80/ilse18a.html)

[17] [Burges et al. Learning to Rank using Gradient Descent. ICML 2005.](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)

[18] [Burges. From RankNet to LambdaRank to LambdaMART An Overview. 2010.](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)

[19] [Zaheer et al. Deep Sets. NeurIPS 2017.](https://arxiv.org/abs/1703.06114)

[20] [Lee et al. Set Transformer. ICML 2019.](https://proceedings.mlr.press/v97/lee19d.html)

[21] [Xia et al. Listwise Approach to Learning to Rank Theory and Algorithm. ICML 2008.](https://www.microsoft.com/en-us/research/publication/listwise-approach-to-learning-to-rank-theory-and-algorithm/)

[22] [van Hasselt, Guez and Silver. Deep Reinforcement Learning with Double Q-learning. AAAI 2016.](https://arxiv.org/abs/1509.06461)

[23] [Schaul et al. Prioritized Experience Replay. ICLR 2016.](https://arxiv.org/abs/1511.05952)

[24] [Schulman et al. Proximal Policy Optimization Algorithms. 2017.](https://arxiv.org/abs/1707.06347)

