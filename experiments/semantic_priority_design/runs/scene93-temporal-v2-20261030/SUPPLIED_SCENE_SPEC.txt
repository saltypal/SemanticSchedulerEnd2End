# Final scene-array caching specification, version 2

Status: implementation and readable notebook built; software tests and fixture execution passed.
Actual 4,000-image inference, expanded-vocabulary accuracy and CPU/GPU costs are not yet measured.
This is one fixed implementation, not a settings search or a replacement SPI experiment.

## Scope and ownership

Canonical repository: `SemanticSchedulerEnd2End`. Shared extraction/serialization contract:
`parameters/scene_arrays.py`. Cache/profiling runner: `experiments/scene_array_cache/cache.py`.
Notebook: `experiments/scene_array_cache/Scene_Array_Caching.ipynb`.
Existing SPI labeling, scheduler policies, historical results and the 35-byte header remain untouched.

## Frozen detector and dictionary

Use `yolov8s-worldv2.pt`, SHA256
`9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`,
Ultralytics 8.3.237, FP32, confidence 0.25, class-aware NMS IoU 0.5,
input size 640, maximum 300 detections, no test-time augmentation.
These are the measured provisional settings retained from the earlier 16-noun study;
they are fixed operating settings here, not proven optimal for the new dictionary.
Pinned OpenAI CLIP source for YOLO class-text setup:
`a1d071733d7111c9c014f024669f959182114e33`. Actual text-feature SHA is recorded.
Class text encoding is one setup cost, not a repeated image-encoder forward.

The exact ordered dictionary is in `parameters/scene_arrays.py` and exported `registry.json`:

| IDs | Type | Names |
|---|---|---|
| 0â€“79 | Countable | Complete standard COCO80, preserving original order/names; see COCO_ALL_80_CLASSES.json |
| 80â€“84 | Countable extensions | building, bridge, utility pole, traffic sign, animal |
| 85â€“92 | Regions | road, water, debris, smoke, flame, snow, ground, grass |

The full expanded registry is `ALL_93_CLASSES.json`. Standard COCO80 includes specific animal categories and `stop sign`; broad extension prompts such as `animal`/`traffic sign` can overlap them. Do not sum channels as unique physical assets. Counts are class-specific retained detections.

Call `set_classes()` explicitly with this dictionary. Do not use default COCO names or
change order between UEs. Categories are detector hypotheses; especially hazard/region
accuracy is unverified. This does not cover every situation, relationship, screenshot or condition.
Zero detections never imply a harmless image. No scene classifier or knowledge graph is added.

## Measurement definitions

For each class: count of retained detections (85 countable classes only), and largest
clipped XYXY rectangle area divided by the EXIF-upright frame area (all 93 classes).
Count masks are false for regions. Area zero after a successful call means no retained
box, not verified absence. Failure makes every measurement unavailable. Outside-dictionary
queries fail explicitly; they do not create a zero-valued slot. A detection-cap flag warns
that the retained evidence may be truncated. Invalid boxes are recorded in audit evidence.

No scalar complexity, urgency, distance, severity, or relationship is inferred from box size/counts.
Per-class union coverage is not included; raw boxes allow a later separately versioned method.

## Exact runtime representation

The standalone scene block is **369 bytes / 2,952 bits**:

| Field | Bytes |
|---|---:|
| Version + status/saturation flags | 2 |
| Full vocabulary/schema SHA256 | 32 |
| Full detector/producer SHA256 | 32 |
| 93 uint16 count slots | 186 |
| 93 uint8 largest-area slots | 93 |
| Count mask + area mask | 24 |
| Total | 369 |

Region count storage is canonical zero with count mask clear. Area rounding has maximum
error 0.5/255. Producer identity includes weights, encoded text features, preprocessing,
configuration, code, package versions and backend. Mismatches are rejected, not guessed compatible.
This block excludes routing metadata, SPI, relevance, novelty and codec payload; do not call it
the final whole-message header or the historical 35-byte header.

The scheduler feature vector has **374 entries**, in fixed order:
`log1p(counts)[93], areas[93], count_masks[93], area_masks[93], detector_ok, saturated`.
Export float and wire-derived versions separately. Train on the received representation,
join by image ID and hash, and preserve grouped splits. The ranker needs an updated input
dimension and retraining; historical weights expecting fewer inputs are incompatible.
No manual class-importance weights are introduced.

## Notebook execution sequence

1. Print actual interpreter/packages/hardware, fixed source receipt and output paths.
2. Materialize readable, hash-verified source under this run, never over the working repository.
3. Export dictionary, configuration, input IDs and known raw-image hashes.
4. Verify numerical count/area/mask/quantization contracts.
5. Load one model, encode vocabulary, verify ten real development images, warm up ten calls.
6. Cache all 4,000 existing CrisisMMD selections; commit each image, export every 25.
7. Release bulk model. Profile CPU and assigned GPU separately on the same ten development images.
8. Export ID-aligned arrays, scene blocks, failures, distributions and annotated examples.
9. Save acceptance status, logs, source/data identities and actual artifact checksums.

## Data, cache identity and recovery

Use the existing 4,000 selected CrisisMMD records at pinned source revision
`10a5626ba112ab9c50369c473a2062bcf205d708`. Known hashes from the 2,540-record
checkpoint are checked. Remaining raw hashes are measured on acquisition.
No SPI targets, native annotations, mission paragraphs or future histories enter detection.
Source split/group identities are carried as provenance, not used as model inputs.

Only cloud execution downloads source images, individually with bounded retries and 20MiB/image cap.
Local real execution uses `SCENE_ARRAY_IMAGE_ROOT` and never bulk-downloads missing images.
`SCENE_ARRAY_RESUME_DIR` points to an existing run. Its source/model/input identity must match.
Successful cached rows are hash-verified, including current raw-image bytes. Failed rows are
retried explicitly (`retry_failed=True`), with prior attempts retained. Cache hits are not fresh
inference measurements. An incompatible cache stops the run. The default session budget is
14,400 seconds; partial coverage is a checkpoint, not completion. No full dataset archive is needed.

## Cost and quality evidence

Per fresh image: fetch/hash verification separately; image load/EXIF/RGB; combined detector
preprocessing/forward/postprocessing/transfer; array calculation; serialization/validation; total.
CPU/GPU controlled profiles: ten fixed training images, ten warmups, 3 CPU repetitions/image
and 10 GPU repetitions/image (30/100 calls), CUDA synchronization, separate memory pass,
parameter count, weight bytes, model-object load and vocabulary setup cost.
Throughput is ten sequential batch-one full extractions, explicitly not vectorized batch throughput.
Report process RSS, CUDA allocated/reserved peaks and hardware. OS/download caches may be warm.
Supported-operator forward FLOPs use two FLOPs/MAC; list uncounted operations, preserve failures,
mark every incomplete count partial. This scene-only notebook runs no VLM or training workload.

Counts/coverage/availability are diagnostics, not detector accuracy. No expanded-vocabulary
mAP, semantic scene completeness, or scheduler benefit is claimed without corresponding evaluation.

## Artifacts and handoff

Runtime: `/kaggle/working/experiments/scene_array_cache/runs/<run-id>/`.
Local: the matching canonical `experiments/scene_array_cache/runs/<run-id>/`.

- `scene_cache.sqlite`: committed rows and preserved failed attempts.
- `scene_records.jsonl`: source IDs/hashes, masks, float arrays, raw boxes, statuses and timings.
- `scheduler_arrays.npz`: ID-aligned 374-feature float/wire matrices and success flags.
- `registry.json`, `producer.json`, `cache_identity.json`, source/environment/input manifests.
- `profiles/`, `logs/`, `figures/`, `validation.json`, `ACCEPTANCE.json`, `artifact_manifest.json`.

Preserve the executed notebook exposed by Kaggle. Retrieve results, not images or model weights,
to the laptop by default. A local path is not proof a remote result was retrieved; verify checksums.
The prepared notebook is not yet submitted. Reuse an established parameter/detector notebook
identity as a new version after checking it is idle; do not replace the active SPI job.

## Completed local validation

Seven regression checks passed. Medium assessment confirmed original identity/recovery/profile
issues were fixed. The complete notebook executed in fresh-kernel fixture mode on 12 generated
software test images, with no detector calls or network downloads. Source LF normalization was
fixed for Windows/Kaggle portability. Actual GPU inference and the expanded vocabulary remain
unexecuted. The fixture output is software evidence only and is not training data.

References: official YOLO-World API https://docs.ultralytics.com/models/yolo-world/ ;
earlier measured operating report `experiments/dataset_detector_grid/runs/analysis_v18/REPORT.md`.

## Vocabulary expansion receipt
User requested every standard YOLO class. This revision includes all80COCOcategories plus13projectextensions. YOLO-World has no finite universal all-classes list;93 is the configured vocabulary, not its total capability. Expanded26cellnotebookfixture execution passed,7regressiontests passed. Old32classschema/170byteblock preserved under runs/schema_v1_32_preserved_*/. New369byteblock is schema v2 and rejects old vocabulary hashes. No actual93class inference or accuracy measurement yet.
