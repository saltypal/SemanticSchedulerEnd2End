# Scene measurements: implementation and validation boundary

27 October 2026. The governing parameter list is the first section of
`../PLAN.md`. This document specifies the scene increment; it does not launch it.
The previous claim-vector proposal below is historical and superseded.

## Selected measurements

Use ONE existing YOLO-World (`yolov8s-worldv2.pt`) and a frozen, versioned object
vocabulary. SPI/relevance/novelty continue using shared CLIP. No exclusive scene
classifier or fleet of scene-specific detectors is introduced.

For retained, canonicalized detections with boxes clipped to the oriented input
image of width W and height H:

| Output | Definition | Meaning |
|---|---|---|
| instance_count | Number of retained boxes | Predicted instances, not verified objects/facts |
| category_count | Number of distinct canonical category IDs | Five people and two cars produce two categories |
| largest_box_fraction | max((x2-x1)*(y2-y1))/(W*H) | Largest predicted rectangular extent |
| union_box_coverage | area(union of retained boxes)/(W*H) | Rectangular coverage, overlapping pixels once |

These are box measurements, not segmentation areas, physical distance, emergency
severity or a complete description of semantic depth. Do not sum box areas as
coverage. Two boxes of area0.20 and0.15 overlapping by0.05 cover0.30, not0.35.
Category identifiers do not receive fixed importance weights.

Illustrative output, NOT measured results:

```json
{
  "instance_count": 7,
  "category_count": 2,
  "largest_box_fraction": 0.20,
  "union_box_coverage": 0.45
}
```

Local audit records retain names and per-category counts (e.g. person5,car2),
boxes, scores, vocabulary and checkpoint/settings identity. These records can
explain a measurement without transmitting the entire detection list. Aggregates
alone cannot distinguish seven chairs from seven people: test that ambiguity.
Transmit a category-summary extension only if that information demonstrably adds
value. No universal descriptive sufficiency is claimed for four numbers.

## Validity and extraction settings

- Successful zero detections: instance_count=0, category_count=0,
  union_box_coverage=0, largest_box_fraction unavailable. This states what the
  detector returned, not verified object absence or low importance.
- Failure, invalid image, unsupported applicability: all four unavailable with
  a failure reason in the audit record. Missing outputs never become zeros.
- Reject nonfinite/degenerate boxes, clip coordinates and drop zero-area boxes.
  Convert coordinates back from model resize/padding before geometry. Use the
  actual EXIF-oriented image dimensions and a declared category alias map.
- Existing settings are the reference, not an optimized guarantee: confidence
  0.25, NMS IoU0.5, image size640, max detections300, with the actual agnostic-NMS
  option and all settings captured from `model_settings.DetectorSettings`.
  Record truncation/saturation when max detections is reached. Never infer that
 300 is the number of real objects or fit an integer encoding that silently wraps.
- Freeze the exact vocabulary and weights hashes before extraction. Counts are
  comparable only under compatible vocabularies/filtering/resolution. Reuse
  `pilot_producers.py`; geometry aggregation should be a small pure function.
- Screenshots remain valid inputs to SPI/relevance/novelty. Detector limitations
  stay visible; do not add OCR or condition classifiers merely to hide a failure.

## Build and test order

1. Finish the SPI label-support diagnosis without altering scene code or labels.
2. Add a pure aggregation function over existing detector output and tests:
   empty/failing producer; identical/nested/disjoint/partly overlapping boxes;
   clipping; invalid coordinates; aliases; duplicate filtering; saturation.
   Tests must distinguish detector non-detection from technical failure.
3. Run a bounded development subset through the existing detector. Compare
   object count/category count and box measurements against compatible published
   box annotations. Image-level labels cannot validate localization or counts.
4. Use grouped frozen splits and matched queues to compare core versus core+
   counts, core+areas, and core+all four. Remove ablated features from derived
   history summaries too. Independent task evidence evaluates scheduling; the
   model's own SPI or Qwen agreement is not ground truth.
5. Report detection precision/recall and count/area error where annotation scope
   permits; report unsupported cells explicitly. Report incremental A10/coverage,
   misses, redundancy and extraction/policy cost in the later scheduler study.
6. Freeze versioned serialization only after range, availability, quantization
   and overflow checks. The existing core remains28bytes. Combined header size
   is not yet measured or implemented; the prior16-claim/4-byte proposal is not
   the selected measurement format.

These measurements are selected engineering candidates. Independent accuracy and
incremental usefulness determine retention; a failed candidate is reported and
removed rather than renamed. Planning adds no downloads, extraction job or GPU
run. The references below remain mechanism/context sources, not proof that these
measurements are sufficient or confer scheduling importance.

<details>
<summary>Historical 26 October claim-vector proposal, superseded above</summary>

# Scene evidence extension: YOLO-World first

26 October 2026. User-selected direction: YOLO-World for scene evidence. Planning only; the current SPI campaign is unchanged. No new model download, extraction job or scheduler experiment is launched by this document.

## 1. Purpose and decision

Retain the four core outputs: SPI, relevance, causal UE novelty and capture metadata. Investigate ONE additional structured evidence vector explaining what the image visibly contains. No rarity or new hand-weighted importance formula.

Use one shared YOLO-World detector with a declared vocabulary. Cache detections once and reuse them for presence, image extent and count diagnostics. Do not keep separate detectors for each scene. Do not route the image through an exclusive disaster/normal/person classifier first: a scene may contain roads, people and flooding together.

The image remains the atomic scheduled job. The scheduler receives serialized predictions, not RGB, boxes or native annotations. Image-level scene evidence can explain task contents and support an evidence-diversity experiment. It does not identify real-world assets across views.

## 2. What the detector actually produces

```text
Image + declared object vocabulary
                 ↓
           One YOLO-World
                 ↓
{category, bounding box, detector score} for each retained detection
                 ↓
Presence claims + optional extent/count measurements
                 ↓
Small evidence vector with explicit unavailable fields
                 ↓
Compare core four outputs versus core + evidence
```

YOLO-World is an open-vocabulary OBJECT detector. Text vocabulary specifies the objects being sought; it does not certify arbitrary event reasoning. Keep these outputs distinct:

| Candidate evidence | Producer | Valid interpretation | Invalid shortcut |
|---|---|---|---|
| person/vehicle/building/road present | YOLO-World detections | A retained detector prediction of a named entity | Person detected → urgent/distressed |
| relative image extent | Geometry of retained boxes | Largest/union box area divided by image area | Large box → nearby in metres or more severe |
| object/category count | Retained detections | Detector-dependent count | Number of real semantic facts or importance |
| building damaged/road blocked | Explicitly validated condition method | Condition prediction at its tested label scope | Building box plus debris elsewhere → this building damaged |
| scene type | Optional later image-level head | Multilabel place/incident prediction | Renaming object detections into a verified scene class |
| screenshot content | Profile-specific evidence, if supported | Readable/labelled screen facts under a tested method | Empty YOLO detections → empty/unimportant scene |

No detections does not automatically mean absence. For the first presence-only detector output, unmatched categories are **unavailable**, not confirmed false. A separately calibrated image-level negative decision can support false where validation warrants it. Missing annotation likewise is not a negative label.

## 3. Vocabulary and applicability

One shared broad object vocabulary supports overlapping content. The initial exact vocabulary and checkpoint/settings are recorded in a configuration before extraction, using the existing YOLO-World integration. Begin with existing supported entities (person, vehicle, building, road, bridge, tree, debris, water); retain existing canonical aliases consistently. Detection quality for each is measured rather than assumed.

Keep a small maximum of16 mission-relevant claims in each shared evidence profile. A profile defines claim meaning, permitted producer, units and dictionary version. This is a dictionary of observable claims, not hidden priority weights. The same detector can serve several profiles. A mission selects which outputs are useful; it does not alter the fixed SPI rubric.

Examples of candidate profiles:

- Aerial infrastructure: building/road/bridge/tree/water/debris presence. Add building-affected or road-damage predictions only from the existing validated aerial condition producer or a separately validated head.
- Ground/traffic: people/vehicle/bicycle/road presence, with optional box extents. Driving-object importance references remain separate from damage/urgency labels.
- Screen/document: detector applicability may be unavailable. OCR or a supported image/text method is a conditional extension if screenshot failures matter; do not add it merely to claim universal coverage.

The same schema works across profiles; the available claims differ. No claim that one detector covers every domain follows from the interface.

## 4. Extraction experiment

Reuse the current implementation in `pilot_producers.py` with `enable_detector=True`, `DetectorSettings`, canonical aliases and pinned weights. Preserve the SPI `minimal_vision.py` preprocessing and signature; detector inputs/settings have their own receipt. Do not silently replace the SPI image representation when adding boxes.

Reuse the existing settings as the fixed reference. Compare only one predeclared lower-confidence setting and, if small objects explain misses, one larger input-size setting on development data. Record confidence, NMS IoU, image size, maximum detections, vocabulary/checkpoint hashes and filtering. Select thresholds by independent annotations, never by Qwen approval. No large grid or model fleet.

Detector confidence is a model score, not a calibrated probability of urgency or condition correctness. Confidence controls whether a predicted detection is retained; its calibration and coverage must be measured.

## 5. Evidence and examples

Illustrative diagnostic JSON, not measured output:

```json
{
  "profile": "aerial_infrastructure_v1",
  "claims": {
    "road.present": true,
    "vehicle.present": true,
    "person.present": null,
    "road.damaged": null
  },
  "measurements": {
    "vehicle_count": 2,
    "largest_vehicle_box_fraction": 0.06
  }
}
```

`person.present: null` means no supported decision, not no people. `road.damaged: null` prevents the object detector from inventing a condition. Count/extent are diagnostics initially; retain them in the transmitted extension only if they add demonstrated value.

Another view may show a new category but still depict the same asset. Call this **additional predicted category evidence**, not verified new information about the world. Exact copies are a control; natural cross-view asset novelty requires correspondence labels.

## 6. Validation in two steps

### A. Can it describe the image accurately?

Use published native labels at their actual scope: existing LADI presence/damage annotations, MEDIC image-level native tasks, accessible object-box annotations for people/vehicles and Incidents1M where legitimate access exists. MEDIC includes CrisisMMD; source names do not imply independence. Screen cohorts need screen labels, not disaster-label remapping. Keep all exposed cohorts development-only.

Measure per-claim precision/recall/F1, supported annotation counts, unknown rate, false assertions and contradictions. Object localization requires box/mask truth; image-level labels do not validate box quality or counts. Separate trusted explicit negatives from unlabeled categories.

Inspect matched cases: intact versus damaged infrastructure; road versus blocked road; people present versus distress only where labelled; ordinary images and hard visual confusions; screens with meaningful content; empty detector output; small/occluded objects. Publish supported cells and gaps, not invented labels.

Use Qwen optionally for claim-level diagnostic review: image + mission + candidate claims + compatible reference annotations, one verdict per claim plus visible evidence and inability-to-assess. Qwen agreement is a secondary model measure. Do not train the extractor to maximize its judge's approval.

### B. Does it add useful scheduling information?

Compare the same policies and queues under:

1. Core SPI + relevance + novelty + capture.
2. Core + presence evidence.
3. Core + validated condition evidence, where supported.
4. Core + extent/count, only on cohorts with valid measurements.

Use identical training splits/budgets and causal sender histories. Fit preprocessing/thresholds/heads inside grouped training roles. Ablate the added candidate outputs AND every derived history/context feature; otherwise the ablation leaks the removed information.

The main outcome remains A10 of independently annotated task evidence, with coverage@10, redundant selections, misses, extraction cost, policy latency, unavailable counts and header bytes. For image-level categories this measures distinct observed categories/annotated observations, not distinct real-world buildings. Controlled object correspondence and natural cross-view studies remain separate.

SPI from the same teacher is not the evaluator's ground truth. Conditioning a diagnostic on similar predicted SPI is descriptive and must not select final examples by their favourable outcomes. Freeze episode construction before comparing policies.

## 7. Compact extension and retention decision

Default candidate:16 registered binary claims, encoded as a16-bit known mask plus16-bit value mask =4 additional bytes. A known bit of0 means unavailable; a known bit of1 disambiguates true/false. The existing shared profile ID must resolve the exact dictionary/version. An extended-header version is required; do not change the current28-byte parser silently. If profile identity cannot resolve the dictionary, include its explicit identifier and account for the bytes.

Thus28+4=32bytes is a **candidate**, not implemented or finalized. Boxes, prose, producer hashes and full audit JSON stay in local audit records. Count/extent would add separately declared bytes if retained. No codebook or dense semantic descriptor is introduced here.

Retain the extension only if independent labels support its claims and grouped evaluation shows useful incremental evidence delivery relative to its extraction/byte cost. Use the established minimum0.01A10 improvement and a grouped interval supporting positive gain for a superiority claim; otherwise report descriptive findings and keep the core design. Insufficient groups do not become statistically independent through more queue seeds.

## 8. Build order and stop rules

1. Finish current SPI results independently; do not interrupt that run.
2. Freeze object vocabulary/profile claim list and reuse existing detector baseline/configuration.
3. Extract once on bounded existing development images, inspect claim-level errors and unsupported fields.
4. Run the small fixed settings comparison only for observed detection failures.
5. Add condition prediction only when object presence cannot express a demonstrated distinction and native labels can supervise it.
6. Run the matched core-versus-extension experiment; then decide transmission fields.

No new dataset download or GPU job is authorized by this planning step. Begin implementation only under the user's subsequent build instruction. Core SPI/relevance/novelty/capture functions and their historical results remain stable.

## Literature and precise inspiration

- [YOLO-World, CVPR2024](https://arxiv.org/html/2401.17270): text-conditioned object detection and offline vocabulary; supports one flexible detector. Does not establish universal condition/urgency recognition.
- [Incidents1M,2022](https://arxiv.org/html/2201.04236): overlapping incident/place labels and explicit category-positive/negative/unknown distinctions. Supports multilabel evidence and missing-label handling rather than exclusive scene routing.
- [LADI official overview](https://github.com/ladi-dataset/ladi-overview): infrastructure/hazard observation labels for post-disaster imagery. Supports a native aerial evaluation cohort, not general-purpose urgency.
- [MEDIC official schema](https://crisisnlp.qcri.org/medic/): image-level disaster,information,humanitarian and damage labels. Its targets remain distinct from SPI.
- [SA-OOSC,2025](https://arxiv.org/html/2509.07436): separates detection from contextual importance assignment and checks importance against human driving annotations. Supports the distinction between observing an entity and deciding its task value; it does not validate our whole-latent scheduler.

All retention choices above are proposed project decisions. No cited paper proves this particular16-claim representation sufficient or optimal.

</details>
