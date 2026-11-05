# Shared image pool and parameter cache

One canonical codebase: `SemanticSchedulerEnd2End`. No repository restructuring is needed.
Notebook: `experiments/image_pool/Image_Pool_Caching.ipynb`. This replaces the queued
scene-only acquisition entry point, reusing `parameters/scene_arrays.py` and
`experiments/scene_array_cache/cache.py`; it does not replace SPI training or scheduler policies.

## Fixed scope

Cache source-qualified image-dependent scene evidence and predictions from the **completed
4,000-image training study's validation-selected SPI student**. Preserve its limitations.
Neither a source category nor a VLM teacher label is substituted for the student's prediction.
Mission relevance, context centroids, causal sender KB queries and novelty stay in the simulator.
No timestamps, GPS coordinates or natural temporal correspondence are fabricated in this cache.

Use small absolute caps, not 10–20% of large corpora:

| Source | Maximum | Acquisition / role |
|---|---:|---|
| COCO | 200 | Official small test2017 image-info ZIP and individual public images |
| Open Images V7 | 200 | Official validation image CSV and individual ID-addressed public objects |
| Places365 | 200 | Official val256 TAR bounded to64MiB via HTTP Range, streaming selected members only |
| CrisisMMD | 200 | Existing pinned4,000selection, deterministic subset, individual source images |
| BDD100K | 5clips ×10frames | Accessible managed frames only; ordered clip identity required |
| ImageNet | 50 | Authorized managed selection; no login or license bypass |
| DoTA | 5clips ×10frames | Accessible managed frames only; no full YouTube source-video fetch |
| UCF-Crime | 5clips ×10frames | Accessible managed frames; source/video IDs retained |
| xBD / xView2 | 50 | Authorized managed images/annotations, no full archive |
| RescueNet | 50 | Managed selection; author/source licensing discrepancy unresolved |
| FloodNet | 50 | Managed selection; scheduler-task transfer only, excluded from fitting/selection |

Total cap1,150 selected images/frames. New-data transfer cap2GiB including metadata and failed
downloads, excluding separately measured model weights. Every image response is capped20MiB;
metadata64MiB. Cloud acquisition alone is enabled. No media or model downloads are executed on the
laptop by the real notebook. Fixture mode makes no network requests.

These caps are ceilings, not guarantees. Sources with inaccessible media or missing managed manifests
are reported `input_required`, `unavailable` or `partial`. They are not silently replaced with unrelated
images and are not counted as completed sources. The pool can cache accessible sources while
preserving the missing-source report.

## Source and annotation cautions

COCO's selected test2017 images do not come with accessible object truth from the image-info file.
Open Images has substantial annotations, but this first acquisition obtains image information only;
the cache must not claim those annotations were joined. Original source URL/license/attribution
metadata is retained where supplied. Places acquisition is a deterministic convenience sample from a
bounded TAR prefix, **not** uniform sampling over Places365. Record individual image hashes and
archive ETag/member identity; no whole-TAR MD5 claim follows from a partial download.

For managed sources provide JSONL rows with `source_id`, `image_path`, `source_revision`,
`source_split`, `source_page`, optional `image_sha256`/native annotations. Temporal sources also need
actual `clip_id`, integer `frame_index`, and `source_video_id` when known. The sampler selects at most
five clips and ten evenly spaced available frames each, preserving order. An isolated driving
keyframe does not become a temporal sequence. No complete video is downloaded automatically.

Dataset and native labels stay in a separate reference record, not runtime model input. xBD and
ImageNet access-confirmation metadata must be supplied for those inputs. The author/source access
audit is recorded in `SOURCES.md`; it is not an assertion of unrestricted redistribution rights.

## Grouping and splitting

Source-qualified IDs plus raw-image SHA256 prevent positional joins. Connect all frames from one
parent video (or clip where video identity is absent), parent-image derivatives identified by the
manifest, and exact byte copies across datasets before assigning group roles. FloodNet-connected
groups stay transfer-only. Known parameter-training-exposed CrisisMMD observations stay development
training, not a fresh final evaluation. Other groups receive a deterministic70/20/10 assignment;
native source partitions are separately retained as provenance. Do not call these random image
groups event-held-out or near-duplicate-safe: near copies and unknown cross-view identities are
explicitly unaudited. Frozen vision-pretraining exposure is unknown.

## Frozen SPI student

Set `IMAGE_POOL_TRAINING_ROOT` to the mounted completed4,000run. Verify:

1. Cohort manifest contains4,000unique IDs; progress confirms4,000stored records and full coverage.
2. All4balance studies and all48distinct encoder/head pairs are complete.
3. Comparison fitting-record IDs/hashes equal the available valid teacher-label IDs/hashes for this
   exact cohort. A stale1,000comparison next to a new progress file fails.
4. Select the best development macroF1; ties use deterministic encoder/head/balance ordering.
   Test performance is never used for selection.
5. Copy/hash the selected head and exact encoder identity into a small bundle. Check trained class
   ordering, frozen checkpoint hashes, architecture source and processor identity at inference.

Image→matching frozen encoder→matching vector/token head→Low/Medium/High/Critical prediction.
Do not supply CLIP embeddings to a SigLIP head. SVM margins are labeled decision scores, never
probabilities. Neural softmax/predict_proba remain uncalibrated unless a calibration artifact exists.
This is an experimental image-priority predictor, **not independent urgency ground truth**. Weak
Medium recall and cross-domain distribution shift remain material limitations. Failed predictions
remain unavailable; no image-source name determines its priority.

## Scene and cache outputs

93configured categories, per-class retained counts (85objectclasses only), largest rectangular
areas, count/area masks, detector failure/cap flags, raw boxes/scores, oriented dimensions,
source/image/producer/vocabulary hashes. Reuse the existing369-byte scene-block and374-feature
wire/full-precision matrices. Their header excludes SPI/routing and codec payload bytes.

Separate scene/SPI SQLite caches bind model/code/input identities, verify bytes and support explicit
failed-attempt retry. JSONL scheduler exports join by ID/hash. No evaluator truth is joined into runtime
observations. Image source and group metadata are provenance, not permission to use those fields as
learned priority shortcuts. Simulator observation code must still enforce its allowed feature boundary.

Per-image measured inference times are retained; cache hits are not new timing samples. CPU/GPU
profiles use ten fixed development images, ten warmups,30CPU/100GPU calls, synchronization, separate
memory passes and partial supported neural FLOPs. SPI and scene costs are reported as separate
scopes; do not sum their p95s as a combined pipeline p95. No VLM inference, classifier fitting or
mission/KB computation is performed in this notebook.

## Artifact layout and execution

```
experiments/image_pool/
  Image_Pool_Caching.ipynb
  config.json, sources.py, student.py, runtime.py, test_pool.py
  PLAN.md, SOURCES.md, source_manifest.json
  runs/<run-id>/
    data/                         # cloud-only selected assets and source reports
    pool_manifest.json
    source_status.json
    references.json
    student/                      # frozen selected head + encoder identity
    scene/                        # shared scene cache, arrays, audit, profiles
    spi/                          # student predictions, independent cache
    handoff/image_parameters.jsonl
    handoff/reference_records.json
    spi_profiles/, logs/, figures/
    ACCEPTANCE.json, artifact_manifest.json
```

Remote root `/kaggle/working/experiments/image_pool/runs/<id>/`; local retrieval mirrors this path
under the canonical repository. Keep cloud source images there. Retrieve compact records/profiles/
manifests and model head first; image retrieval to the laptop is a separate explicit choice.

The prepared notebook reuses the established idle parameter/detector notebook identity after the
completed training source can be mounted. It must not interrupt the active SPI-labeling notebook.
No new remote notebook identity or cloud execution is created by this implementation alone.
`IMAGE_POOL_RESUME_DIR` selects a matching run; code/model/input identity changes reject caches.

## Acceptance and current evidence

Twelve software tests pass: cloud-only network gating, clip selection/order, exact/transfer groups,
SPI validation-only selection, stale fitting-cohort rejection, runtime/reference isolation, and scene
codec/cache invariants. The notebook has executed fully on12softwarefixtures, with no real producer
inference or media acquisition. Actual new-source access, GPU inference, profile measurements,
near-duplicate audit and cross-domain model accuracy remain unexecuted/unestablished.

## Approved scene-only execution — 2026-10-30
The user approved running acquisition and scene caching on WSL account satyapaladuguplus while Windows SPI training continues. Set IMAGE_POOL_SCENE_ONLY=1. SPI remains unavailable/pending; no historical student is substituted. The completed scene cache and image hashes are retained for later SPI enrichment. Source and transfer caps remain unchanged. Final acceptance requires every acquired scene record to succeed; combined parameter completeness remains false until SPI is added.
Validation: all 36 cells executed in scene-only software-fixture mode; 12 existing focused tests passed. This does not establish real GPU inference or detector accuracy.

## Expanded pool — 2026-10-30
User approved 6000 target records: all exact 4000 selected CrisisMMD IDs plus 1000 Open Images and 1000 Places365. Scene-only cloud acquisition proceeds while Windows SPI training is untouched. Keep 2 GiB transfer cap; report shortages rather than claiming 6000 unique images. Places bounded TAR prefix enlarged to 256 MiB; remains convenience sampling. Deduplicate source-qualified identities and image hashes before downstream split construction. Previous 600 records are preserved; cross-run reuse requires verified identities and producer compatibility, not in-place mutation of a different cache.
