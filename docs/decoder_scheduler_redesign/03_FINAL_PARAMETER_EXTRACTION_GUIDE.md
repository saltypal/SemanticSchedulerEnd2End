# Finalized semantic parameters: extraction and validation guide

**Date:** 22 October 2026  
**Authority:** Latest user-selected parameter set, plus the explicitly retained transmitter-side knowledge-base novelty.  
**Status:** Implementable design, not measured extraction accuracy or implemented runtime behavior.

**Core parameters:** context relevance, scarcity, reliability, task importance, freshness, spatial prominence, scene complexity and novelty.

**Companions:** [High-level design](01_HIGH_LEVEL_RESEARCH_DESIGN.md) · [Runtime and benchmark specification](02_LOW_LEVEL_IMPLEMENTATION_SPEC.md)

**Final training/evaluation method:** [Protocol 04](04_FINAL_BENCHMARK_EVALUATION_AND_LOSS.md). All eight parameter meanings in this guide remain unchanged.

**Candidate VLM producer:** [Guide 05](05_VLM_INTEGRATION_AND_TASK_CONDITIONING.md) describes Qwen relevance/evidence labeling, transmitter retrieval comparisons, student training and independent evaluation safeguards.

## 1. Keep eight meanings; share the extraction work

Do not run eight independent large models. Reuse one image embedding, one detector, a small task head, source timestamps and one transmitter knowledge store.

~~~mermaid
flowchart TD
    X["Image + task + available source metadata"] --> V["Shared image embedding"]
    X --> D["One detector: classes, boxes, scores"]
    X --> T["Capture time and clock status"]
    V --> R["Context relevance"]
    V --> N["Novelty against prior transmitter KB"]
    D --> I["Task importance: task rubric + evidence"]
    D --> P["Spatial prominence: supported target area"]
    D --> C["Scene complexity: count + diversity"]
    D --> Q["Reliability: calibrated task-claim correctness"]
    K["Prior transmitter KB: embeddings, concept counts, time, context"] --> N
    K --> S["Scarcity: prior concept frequency"]
    D --> S
    T --> F["Freshness"]
    R --> H["Eight scalars + validity mask + provenance"]
    N --> H
    I --> H
    P --> H
    C --> H
    Q --> H
    S --> H
    F --> H
    H --> W["[ParameterHeader][Complete latent]"]
    W --> SCH["Receiver ranking policy"]
    SCH --> DEC["One existing decoder"]
~~~

All image-dependent extraction takes place at the transmitter. The receiver parses scalar metadata. It can update freshness from a trusted capture timestamp and current time; it does not recompute novelty or inspect pending latents.

The two KB-derived quantities use different views of the same prior memory:

- **Novelty:** similarity to particular previous observations or evidence.
- **Scarcity:** how uncommon a concept is in a declared reference population.

A common event at a new location can be novel without containing rare classes. A rare object photographed repeatedly can be scarce but non-novel.

## 2. Parameter contract at a glance

| Parameter | Question | Initial implementation | Output kind | Independent validation |
|---|---|---|---|---|
| Context relevance R | Does this serve the active task? | Image–task embedding alignment; small supervised task-relevance head when labels exist | Bounded similarity, or separately calibrated probability | Task-relevance labels and retrieval/ranking quality |
| Scarcity S | How uncommon is the supported content in prior observations? | Smoothed concept occurrence counts in transmitter KB | Bounded rarity statistic | Frequency audit, rare-class slice, sampling-bias stress |
| Reliability Q | How trustworthy is the reported task evidence? | Calibrated correctness model for a specified task-determining claim | Probability only after calibration | Correct/incorrect claim labels, Brier/NLL/reliability plots |
| Task importance I | How consequential is this evidence for the declared task? | Expected task gain from evidence probabilities and a fixed reviewed rubric | Normalized expected review gain | Independent importance grades/preferences |
| Freshness F | How old is the observation, in the relevant time scale? | Trusted capture timestamp and declared decay constant | Deterministic time-derived score | Timestamp/clock audit and age-response checks |
| Spatial prominence P | How much of the image is occupied by a relevant supported entity? | Maximum relevant bounding-box area fraction; masks as an optional comparator | Image-plane area fraction | Reviewed boxes/masks and tiny-important-object errors |
| Scene complexity C | How many distinct supported entities and concepts are present? | Count saturation plus class-diversity entropy | Declared complexity proxy | Counts/diversity labels, detector-error and clutter slices |
| Novelty N | What is new relative to this transmitter's prior KB? | Nearest embedding cosine distance, then context/fact-aware comparisons | Visual-distance proxy initially | Reviewed repeat/new-view/new-fact/new-incident cases |

Being bounded to [0,1] does not make these values interchangeable probabilities. The task/profile registry records the score meaning, producer and normalization.

## 3. Context relevance

### Meaning

Relevance is alignment with a declared task, not severity. An image of an intact building can be relevant to a building survey even when its damage-review importance is low.

The task is selected before looking at the hidden label of a test image. Examples: building-damage assessment, road obstruction review or person-location review. One image can be relevant for one task and irrelevant for another.

### Obtain it

Compute unit-normalized image embedding e_i and a cached task-text embedding v_T. A CLIP-style baseline uses:

\[
r_i=e_i^\top v_T,\qquad
R_i^{sim}=(r_i+1)/2.
\]

CLIP aligns image and text representations; this motivates the baseline, not a claim that cosine is a calibrated relevance probability. [CLIP, 2021](https://proceedings.mlr.press/v139/radford21a.html)

Use a fixed prompt template and vocabulary; log the raw cosine. Multiple task prompts can be compared, but a softmax over a small prompt list is not automatically a probability of real-world relevance.

For a fixed application with labels, train a small head on the same frozen image embedding to estimate P(relevant | image,task). Calibrate on separate data. Do not mix similarity and probability producers under the same profile ID.

### Example

Cosine 0.40 maps to R=0.70. That means normalized model alignment, not “70% relevant,” unless calibration has established that interpretation.

### Failure modes and validation

Generic prompt matches can ignore small critical regions. Compare whole-image relevance with evidence from relevant detected regions in a later ablation. Validate with human task relevance, AUROC/AUPRC when binary labels exist, and errors for small or occluded targets.

## 4. Scarcity

### Meaning

Scarcity is rarity in a declared reference population. Here it means **semantic-content scarcity**, not remaining radio resources, queue slots or processor capacity.

The reference is the transmitter's causal observation history or a frozen training-only background population. These are distinct experiments. A local source's observation frequency is not the frequency of disasters in the world.

### Obtain it

For each vocabulary concept k, store:

- M: number of eligible prior observations;
- n_k: number containing supported concept k, counting it at most once per observation.

With Bernoulli smoothing alpha>0:

\[
\hat p_k=\frac{n_k+\alpha}{M+2\alpha}.
\]

Use a fixed lower probability floor p_min and define:

\[
s_k=
\frac{-\log(\max(\hat p_k,p_{min}))}
{-\log(p_{min})}.
\]

The result is in [0,1]. Average over the unique, supported, task-eligible concepts in the image, with a declared weighting rule:

\[
S_i=\frac{\sum_{k\in A_i}a_k s_k}{\sum_{k\in A_i}a_k}.
\]

The first baseline uses a_k=1 after a validation-selected detection threshold. Confidence-weighted aggregation is a separate ablation. If no eligible concept is supported, S is unavailable; do not label detector silence “extremely rare.”

### Example

M=100, n_k=4 and alpha=1 give p_k=5/102≈0.0490. With p_min=0.01, s_k≈0.655. This number describes rarity under that history, not task consequence.

### Failure modes and validation

A rare false detection can receive a high rarity score. Use supported concepts, maintain uncertainty and audit rare detections. Repeated frames can bias counts; compare raw observation counts with a predefined episode/site sampling rule. Never compute test rarity using future observations or final full-test frequencies.

Rare concepts need not be useful. The scheduler receives scarcity separately from importance. The surprisal formula is an engineering rarity feature; it is not measured compressed payload length or information-theoretic semantic value.

## 5. Reliability

### Meaning

For this receiver-after-arrival scheduler, define reliability as:

> Probability that the transmitter's specified task-determining semantic claim is correct.

It is **not** a second SNR field, a guarantee of latent reconstruction quality, CRC success, source reputation or pixel sharpness. Those are different quantities.

Choose the claim explicitly, for example “this image contains minor-or-greater building damage.” If the producer reports several claims, declare which claim Q refers to; do not silently average confidence for unrelated objects.

### Obtain it

Start with the task predictor's logits, margin or confidence. Fit a held-out calibration mapping against independent claim-correctness labels. Temperature scaling is a first comparator; a small correctness predictor can incorporate blur/exposure or detector agreement if it improves independent validation.

For a predicted claim h_i and truth h_i*, define the calibration target:

\[
b_i=\mathbf 1[h_i=h_i^*],\qquad
Q_i\approx P(b_i=1\mid\text{producer outputs}).
\]

A raw maximum softmax or detector confidence is initially a proxy, not established reliability. [Calibration, 2017](https://arxiv.org/abs/1706.04599)

### Example

If a validated calibration model outputs Q=0.80, matching-distribution cases near that value should be correct about 80% of the time. This property must be measured; it is not obtained by renaming a score.

### Failure modes and validation

Report Brier score, NLL, calibration plots, class-wise correctness and domain shift. A model can be confidently wrong. The header reliability model does not prove that the eventual decoder output preserves the claim.

Do not automatically multiply I by Q: if I is already an expected gain over calibrated class probabilities, another multiplication can double-count predictive uncertainty. Keep both inputs for the learned ranker and test whether Q adds information.

## 6. Task importance

### Meaning

Importance reflects the task's consequence or review value. It needs a task rubric. Pixels alone do not define a universal ordering of social or operational importance.

For a first reproducible building-damage profile, use a visible-evidence proxy rather than claiming life-safety urgency.

### Obtain it

Let y denote a supported evidence category, p_ic its predicted probability and g_c the application's declared gain. Then:

\[
I_i=\frac{\sum_c g_c p_{ic}}{g_{max}}.
\]

Use a gain range [0,g_max]. If the utility has a different scale, register its normalization explicitly.

Example for LADI v2a: categories no building, building without affected evidence, affected below minor, and minor-or-greater; illustrative gains (0,0,1,3). For p=(0.05,0.10,0.15,0.70):

\[
I=(0.15+3(0.70))/3=0.75.
\]

LADI provides visible-content labels, not emergency response priority or per-building correspondence. The gain mapping is our proposed review-task preference. [LADI v2](https://github.com/LADI-Dataset/ladi-overview)

### Better supervision

Collect task-review grades and independent pair preferences. A small task head can then predict that reviewed importance instead of a label-derived proxy. Keep public-proxy and human-importance results separate.

An object detector identifies candidate evidence; it does not establish damage severity, injury or causality without a corresponding trained, validated task model.

### Failure modes and validation

Importance and relevance overlap but are not identical. Scarcity and novelty do not determine consequence. A small relevant object may dominate the importance judgment despite low spatial prominence. Measure these counterexamples and feature dependence.

## 7. Freshness

### Meaning

Freshness measures age since capture, not time since the message reached the receiver. Queue age is waiting time; an old photograph can have just arrived.

### Obtain it

Use a trusted capture timestamp t_capture and current time t:

\[
a_i(t)=\max(0,t-t_i^{capture}),\qquad
F_i(t)=\exp(-a_i(t)/\tau_T).
\]

The time constant tau_T is fixed by the task or selected using validation with independent time-value judgments. Its units must match age. It is not a universal learned property of all images.

The transmitter can place an initial F in the header. The receiver can update it using capture time and the same registered rule. This is deterministic metadata processing; novelty remains fixed.

### Example

Age 30 seconds, tau=60 seconds:

\[
F=e^{-30/60}\approx0.607.
\]

After another 30 seconds it is approximately 0.368.

### Failure modes and validation

EXIF timestamps can be missing, edited, local-time-only or unsynchronized. Unknown capture time makes freshness unavailable. Do not substitute receive time and pretend the image was newly captured.

If clocks have bounded error, retain the uncertainty and test its effect. A future capture time beyond the permitted skew is invalid. For static illustrations, freeze F at the common decision time; for dynamic queues, recompute before dispatch.

Historical observations can still be important; a fixed decay is a task assumption to validate. Freshness is not the same as an externally specified hard deadline.

## 8. Spatial prominence

### Meaning

Prominence measures how much of the image a supported relevant entity occupies. **Physical proximity is a separate geometric question.**

A large distant building can fill more pixels than a nearby small object. Zoom, focal length, object size and viewpoint all affect image area.

### Obtain size prominence

Run one detector, retain boxes, and deduplicate overlapping predictions using the declared detector/NMS settings. For each supported task-relevant object o:

\[
A_o=\frac{(x_2-x_1)(y_2-y_1)}{WH},\qquad
P_i=\max_{o\in O_i^{task}} A_o.
\]

Coordinates refer to the original image after undoing any letterboxing. Clip boxes to image bounds. With a mask, replace box area with mask pixels and register a distinct measurement basis.

A box is an approximation and includes background. Use one basis per profile; do not mix mask and box measurements without labeling it.

If the detector is valid and finds no supported task-relevant object, P=0. If the detector failed or the target vocabulary is unsupported, P is unavailable.

### Example

A 1000×800 image with one relevant 200×160 box gives:

\[
P=32000/800000=0.04.
\]

A relevant 500×400 box gives P=0.25.

### Obtain “how close?” only when evidence supports it

Options include calibrated depth/range sensors, trusted geometric metadata, or an independently validated metric-depth model for the actual domain. Relative monocular depth can supply relative near/far structure, but it cannot automatically provide reliable metres.

Depth Anything V2 distinguishes general relative-depth models and models adapted with metric labels. That is a model distinction we must preserve. [Official implementation](https://github.com/DepthAnything/Depth-Anything-V2)

For the core scalar P use image-area prominence. Log any proximity estimate separately and add an explicitly named optional proximity field only after its units and accuracy are validated. Do not silently encode a mixture of area and distance under one number.

### Importance is learned separately

Nearby or visually dominant targets can matter under some tasks. They do not always deserve higher priority. Include examples with tiny critical targets and large benign objects in ranking evaluation.

## 9. Scene complexity

### Meaning

Complexity describes supported semantic density and diversity. It is not JPEG file size, pixel entropy, image resolution, novelty or guaranteed actionable information.

Many similar cars and a scene with people, damage, vehicles and obstructions have different count/diversity patterns.

### Obtain it

After deduplication and fixed detection thresholds, let n be supported instance count, n_k counts per class and K the fixed vocabulary size. Use:

\[
C_{count}=\min\left(1,\frac{\log(1+n)}{\log(1+n_{ref})}\right).
\]

For n>0, pi_k=n_k/n and:

\[
H=-\sum_{k:\pi_k>0}\pi_k\log\pi_k,\qquad
C_{div}=H/\log K.
\]

For no supported detections, set both components to zero only when the detector is functioning in its supported domain. Otherwise mark unavailable.

The initial engineering proxy is:

\[
C_i=0.5C_{count}+0.5C_{div}.
\]

Fix K, n_ref and the mixture before test evaluation. These coefficients define a complexity measurement, not an overall semantic priority formula. Preserve count and diversity separately in the audit record.

Require K>=2 and n_ref>0. Taxonomy/model changes require a new profile and renewed validation.

### Example

For n=4, class counts (2,1,1), K=4 and n_ref=9:

- C_count=log(5)/log(10)≈0.699.
- H≈1.040; C_div=1.040/log(4)=0.750.
- C≈0.7245.

Four objects of the same class have the same count component and diversity zero. Ten false detections do not constitute ten verified facts.

### Failure modes and validation

A detector's vocabulary, threshold and duplicate suppression strongly affect complexity. Validate counts, diversity and human simple/complex comparisons on a reviewed subset. Compare relevant versus irrelevant clutter. Relations can enrich complexity later, but counting unsupported scene-graph edges is not reliable information measurement.

## 10. Novelty from the transmitter knowledge base

### Definition and computation

Use the transmitter's observation history **before** the current image, within the declared task/context scope.

For unit-normalized current embedding e_i and prior entries e_j:

\[
c_i=\max_{j\in K_{TX}^{<i}}e_i^\top e_j,\qquad
N_i^{cos}=(1-c_i)/2.
\]

A genuinely empty declared KB yields N=1 by convention. An unavailable KB, invalid embedding or incompatible extractor version yields unavailable N.

Compute novelty before insertion. Then add the current observation according to the declared observation-memory rule. Receiving or decoding that image is not required for a transmitter observation KB to know it.

### Example

Nearest cosine 0.96 gives N=0.02; nearest cosine 0.30 gives N=0.35.

Both are embedding-distance statistics. A universal “new disaster if N>0.3” rule is unsupported until separately validated.

### Context and same-disaster/different-angle cases

With verified incident/site/time metadata, compare relevant prior observations in that scope. Without it, say visual novelty relative to local history.

Annotate and test:

1. Same image or nearly identical view.
2. Different angle with no additional task evidence.
3. Different angle revealing additional damage or another affected asset.
4. Similar-looking imagery from a different incident.
5. Changed condition at the same site.
6. Unassessable identity or evidence.

Do not force cases 2 and 3 into the same class. Identical damage categories do not establish identical facts. Cosine can fail on each boundary.

### Upgrade path

Retrieve top-k matches and compare supported entities, attributes and relations with known evidence. Use a lightweight trained “adds task evidence” model only if independently labeled data show that it improves the cosine baseline. A vector database or knowledge graph provides storage/retrieval structure; it does not solve evidence comparison automatically.

A 2025 KB-based CLIP transmission paper uses a receiver-side retrieval KB and category-consistency evaluation. It motivates embedding retrieval, but does not validate this transmitter novelty estimator or disaster identity. [Paper](https://arxiv.org/abs/2504.01053), pp. 1–3.

### Multiple transmitters

Novel to sender A can already be known to sender B or already available at the receiver. Local memories and fixed headers cannot resolve all cross-source redundancy. Compare local versus explicitly shared transmitter memory, recording synchronization assumptions and cost. Do not silently add receiver-based novelty.

## 11. Joint worked example

Illustrative numbers only; no model inference is claimed.

| Field | Example origin | Value |
|---|---|---:|
| R | Raw task cosine 0.40, registered normalized-similarity profile | 0.700 |
| S | M=100, n=4, alpha=1, p_min=0.01 | 0.655 |
| Q | Hypothetical calibrated claim-correctness model | 0.800 |
| I | Expected gain from p=(0.05,0.10,0.15,0.70), gains=(0,0,1,3) | 0.750 |
| F | Capture age 30 s, tau=60 s | 0.607 |
| P | Maximum relevant box area 200×160 in a 1000×800 image | 0.040 |
| C | Four detections with class counts (2,1,1), K=4, n_ref=9 | 0.724 |
| N | Nearest prior transmitter-KB cosine 0.96 | 0.020 |

The scheduler receives the values separately, plus availability and metadata. It does not multiply all eight values.

A low N does not erase the high I. A large P or C does not guarantee more importance. The learned ranker must earn those interactions against independent task judgments.

## 12. Shared models, implementation and cost

| Component | Core use | Optional alternative or extension |
|---|---|---|
| Existing CLIP-compatible image/text embedding | R and N; supports a small task head | Another frozen embedding only after matched evaluation |
| One detector, candidate YOLO-World | Objects/boxes for S, P, C and task evidence | Grounding DINO comparator; not an additional mandatory pass |
| Small task-specific head | I and claim outputs for Q | Public reference classifier when compatible and leakage-free |
| Calibration artifact | Q and probability-based R/I | Compare calibration methods on held-out labels |
| Plain vector store and concept counters | Transmitter N and S | FAISS index when profiling justifies it |
| Capture metadata | F | Clock-error and missing-time stress |
| Selective segmentation | More accurate P in an ablation | Avoid mandatory segmentation for every image |
| Validated range/depth | Optional proximity | Core P remains image-area prominence |

YOLO-World supports open-vocabulary detection; its published hardware speed is not a latency claim for our machine. [YOLO-World, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html)

Measure embedding, detector, head, KB query, counter update and serialization time separately at the transmitter. Measure parsing, ranking and decoder time separately at the receiver.

For M stored d-dimensional float32 vectors, embedding storage alone is 4Md bytes; 10,000×512 takes 20.48 MB, excluding metadata and index overhead. Exact cosine retrieval costs O(Md) per query. Approximate search requires recall and novelty-error checks, not just speed measurements.

## 13. Validation gates and frozen choices

Before calling the parameter system usable:

1. Pin the task, vocabulary, score kinds, model revisions and ontology.
2. Establish real supported detections and a labeled task head; prompt matching remains a baseline.
3. Keep predicted values physically separate from hidden labels.
4. Build causal KB snapshots with no self/future comparison.
5. Fix scarcity population, sampling unit, smoothing and memory eviction.
6. Calibrate reliability on independent labels and report unavailable values.
7. Validate capture-time provenance; do not invent freshness for timestamp-free datasets.
8. Validate boxes/counts and distinguish size from physical distance.
9. Review repeat/new-view/new-fact cases and measure novelty errors.
10. Train/evaluate the scheduler with actual serialized fields and masks.
11. Run single-feature removal, correlated-feature and missing-value tests.
12. Report semantic prefix quality and real decoded evidence, not only feature-extraction plausibility.

Features can be finalized conceptually while particular producers remain unvalidated. Every run records the actual producer, score kind, valid status, model/KB version and evidence supporting each reported number.

## 14. Literature role and scope

The formulas for scarcity normalization, prominence aggregation, complexity mixture, header size and experimental protocol are **our proposed engineering definitions**. The cited papers support components; they do not establish that this exact eight-parameter combination improves scheduling.

- [CLIP](https://proceedings.mlr.press/v139/radford21a.html): image/text embeddings.
- [YOLO-World](https://openaccess.thecvf.com/content/CVPR2024/html/Cheng_YOLO-World_Real-Time_Open-Vocabulary_Object_Detection_CVPR_2024_paper.html): one open-vocabulary detector candidate.
- [Probability calibration](https://arxiv.org/abs/1706.04599): confidence versus empirical correctness.
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2): relative/metric-depth distinction.
- [LADI v2](https://github.com/LADI-Dataset/ladi-overview): public visible-evidence labels and reference/deployment model distinction.
- [Knowledge-Base based Semantic Image Transmission Using CLIP](https://arxiv.org/abs/2504.01053): receiver retrieval, not a transmitter-novelty validation.
- [Scene Graph-Aided Probabilistic Semantic Communication for Image Transmission](https://arxiv.org/abs/2507.11913): entity/relation and shared-knowledge compression direction; not validation of our scene-complexity priority scalar.
