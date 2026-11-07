# Notebook9: Full dataset Labeling and parameter caching

## Active structure and functions

Keep Codec/,Scheduler/,parameters/,Evaluation/ owners unchanged. Production
functions live in parameters; this experiment directory holds notebook/tests/runs.
No source fork or project copy. Historical outputs are untouched.

| Function/component | Input -> output | Important boundary |
|---|---|---|
| validate_records | Image/group/source manifest -> validated records | Exact image hashes, unique safe IDs, group/content split isolation |
| load_selected_teacher | Accepted model/revision/prompt receipt -> reviewer | Immutable revision pinned before load; no guessed winner |
| label_dataset | Images + selected reviewer -> labels.json + raw requests | Image-only direct4class pseudo-labels; evaluator facts/post text excluded |
| export_spi_training | Bound features.npz + labels.json -> trainer bundle | Calls existing SPI prepare validator; no fitting or invented classes |
| require_spi_ready | Four-class model + quality receipt -> classifier | Artifact identity/support/recall/export/review gates; synthetic receipts blocked by default |
| cache_parameters | Images + UE missions + one detector + readySPI -> cache | Shared image evidence, separate causal perUE/split histories; serialize before KB commit |
| YOLOWorldSceneProvider OR GroundingDINOSceneProvider | Image -> SceneMeasurements | Counts/categories/box extents; neither measures urgency/distance |
| profile_forward | Neural module + example input -> partial operator FLOPs | Supported-operator count, omissions explicit; never complete total |

## Notebook stages

1. Print actual source/interpreter/packages/hardware and unique output directory.
2. Execute8-image synthetic fixture through shared extraction/header code.
3. Require reviewed real config (RUN_REAL isFalse by default).
4. Label with notebook8-reviewed teacher; export verified image-only SPI dataset.
5. Fit via existing grouped SPI trainer as a separate reviewed step; install
   accepted readiness receipt only after all gates pass. Do not autofit/promote.
6. Cache with readySPI+CLIP+exactlyone detector; six existing UE profiles.
7. Full-source inventory/resource gate; no implicit dataset downloads.

## Real records and artifacts

Each record requires image_id,image_path,image_sha256,group,split,source.
Optionalcapture_time_ms/timestamp_trusted remain unknown when absent. Source
annotations and associated post text stay in evaluator records. MEDIC/CrisisMMD
shared images cannot cross splits or count as independent sources. TREC-IS post
priority is not an image-only target.

Output manifest binds ordered identities, mission versions, model/readinesshashes.
Image shards contain static measurements, per-call timing, header and causal UE
records. features.npz contains image_ids,image_sha256,embeddings,feature_signature.
Training bundle contains these features plus labels.json, schema-validated by the
actual SPI training loader. All real labels remain VLM pseudo-labels. Incomplete
runs retain status/checksums; completed_images counts persisted shards only.

## Validation and honest limits

Dummy notebook executes in a fresh kernel using canonical project.venv. Tests
check hashes/splits, unavailablecontext preservation, fixture/readinessboundary,
35byteheaders, independenthistories, persisted-workstatus and actual trainer
schema. Synthetic receipt/data are explicitly marked and never a real release.
No real teacher selection, four-class student,300extraction, detectorbenchmark or
full-dataset run follows from these tests.

Timing currently reports combined measured scopes (e.g.visual extraction includes
preprocessing and firstmission setup); first-call flags are explicit. No claim
of warmed steady-state latency. The optional profiler is partial and not wired
as full perimage modelFLOPs; uninstrumented counts remainnull. Automatic resume,
fully audited source adapters, warmed fullprofiling and fullcorpus staging remain
pending. Existingoutput is refused rather than silently resumed.

Notebook9 is prepared locally, NOT a newly created Kaggle notebook identity.
Remote run is gated on8selection, source packaging, SPI readiness, datasetaccess,
<=20GB curatedclouddata/eightaggregateGPUh estimates, and completed300review.
Kaggle ops continue versioning the existing stable notebook where applicable.


## Final user decision: teacher and detector contract

Teacher is now Qwen/Qwen3.5-9B, revisionc202236235762e1c871ad0ccb60c8ee5ba337b9a,
accepted explicitly by the user in parameters/config/final_labeling_teacher.json.
This supersedes the earlier pendingteacherchoice. No independent urgencyquality
claim is made; known warnings remain recorded. Normaldisplay/unknownavailability
policy is unchanged for processing failures.

YOLOWorld is the default spatialproducer; GroundingDINO is an optional replacement
via make_spatial_provider. Both provide instances,categories,largest_box_fraction,
union_box_coverage. No combineddetections, cropassistedVLM, area-to-urgency rule,
newSPIparameterorheaderchange. YOLOconf.25/IoU.5/imgsz640/max300. DINOthreshold.35/
textthreshold.25/class-awareNMS.5/max300; its categories are retainedtextphrases,
not necessarily identical semanticclasses toYOLO. DINOrealinferenceunverified.

Notebook9 now contains explicit label300 -> trainingbundle -> groupedSPIhead
training -> validation/cache-readiness -> cache300 cells. train_spi_for_cache
reuses existing selection/export; supports linear andtinyMLP, fouroutputs in
Low/Medium/High/Critical order. Qualitychecks require actual classsupport,
perclassvalidationrecall.70andexportparity. No forcedmissingclasses, no historical
threeclassfallback. Readyreceipt permits teacher-agreement caches under explicit
userteacherchoice while retaining independent_urgency_validated=false and known
teacherlimitations. The historical installedreleasepointer is not modified.

RUN_REAL remainsFalse. Do NOT submit or executeNotebook9 untiluserrequests.
Onlyunit/integrationtests were executed, including CPU SPI fitting on synthetic
separablefeatures and actual NumPyexportparity. 15tests passed15.13s. This is not
SPItraining on real300images. Existing input300ZIP materializes only whenreal
labelingcellisrun. Exampleconfig includes a detectorcheckpoint placeholder that
must be set to verifiedweights; no automaticdownload is introduced.
