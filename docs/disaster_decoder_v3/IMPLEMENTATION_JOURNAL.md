# Implementation journal

Latest completion is recorded at the end; earlier entries are chronological checkpoints, not current status.

All entries distinguish code inspection, synthetic checks, planned work, and measured model results. Earlier status statements are chronological and are superseded by later checkpoints below; Stage B results are now recorded in the 2026-10-23 entry.

## 2026-10-22 — kickoff / Stage 1 checkpoint

- Scope: additive receiver-side ranking study for fully arrived digital latent jobs; no changes to legacy scheduler behavior are intended.
- Process correction: three parallel coding workers were stopped immediately after the user clarified that implementation must be sequential in the main thread. Current ownership: main thread implements sequentially; short audits and medium testing/Kaggle support are separate roles; documentation is maintained here. This is a process record, not evidence of completed implementation.
- Verified implementation: `SemParam/decoder_parameters.py` contains the eight-field parameter bundle/extractor and V3 32-byte digital header. Its tests reportedly passed 11/11 in 1.25 s. These are synthetic math/contract checks only; they do not establish extraction quality, model performance, or training.
- Planned data lock: dataset revision `5f2dbfe8c466d32edafd1bab847ec5252309acdb`; validation/test only are downloaded. Validation A/B/C split 50/20/30 using grouped separation; test remains locked. State here records the approved protocol, not independent confirmation of all downloads or grouped artifacts.
- Label limitations: controlled observation coverage is not verified fact coverage. Exact repeats can share IDs; different images are not proven to depict physically distinct incidents. Qwen-generated annotations are silver labels and must remain analytically separate.
- Remaining: extraction/cache, annotation, actual ranker fitting, classical/linear/tree baselines, receiver replay, and external Kaggle work. No model-2 implementation or training is confirmed. `INFRA/learning/decoder_v3_extract.py` is a partial draft and has not run.
- Timing: started about 10:15 IST 22 Oct; 24-hour deadline about 10:15 IST 23 Oct. Target depends on external allocation. Future checkpoints must be appended with actual evidence; do not infer progress from this plan.

## 2026-10-22 — Stage 1 audit and regression checkpoint

- Sol-high audit identified calibration loss being optimized before probability projection, incomplete concept logits being treated as negative observations, permissive/string clock-trust values, malformed logits able to fail unsafely, and embedding norm overflow.
- Main-thread fixes: calibration grid now evaluates post-projection BCE; semantic history requires complete eligible-concept support; timestamp trust/numeric inputs are validated safely; embedding normalization is stable for large finite vectors; native-label `S` threshold and calibrated `I` handling were corrected.
- Added three regression-test groups covering these cases. Actual command `python -m pytest SemParam/tests/test_decoder_parameters.py -q`: **14 passed in 1.33 s**.
- Header layout/CRC/availability-mask audit passed. This remains unit/contract evidence only: no Kaggle run, model inference, extraction cache, or training has occurred.

## 2026-10-22 — Stage 2 scheduler module

- Main thread implemented `Scheduler/decoder_scheduler.py`: NumPy MLP/linear/classifier/regressor model loading with checksummed `model.json`, optional LightGBM loading, plus FIFO, round-robin, random, and importance-heuristic policies. Candidate features are the 16-value pure-header representation; identity ranking is not learned. Selection uses deterministic tie-breaking and does not mutate the queue.
- Added `Scheduler/tests/test_decoder_scheduler.py` (8 tests). Combined command `python -m pytest SemParam/tests/test_decoder_parameters.py Scheduler/tests/test_decoder_scheduler.py -q`: **22 passed in 2.85 s**. Tests also confirm importing this path does not load Torch.
- No actual model training has occurred. Next implementation work is the frozen extractor and repair of the partial data loader; neither is yet verified here.

## 2026-10-22 — extraction failure and simplified recovery

- This checkpoint supersedes the earlier extraction-status statements above. The first Kaggle extraction saved 1,192 unique image feature/label pairs before a same-run duplicate bookkeeping error. Those files were recovered; they are not scheduler results.
- The next notebook confirmed two Tesla T4 GPUs but stopped before workers or Ollama ran: two exact-byte image identities have contradictory native labels. No new inference FLOP measurements or Qwen execution were obtained in that run.
- Main-thread repair: retain all original manifest rows and labels, mark every copy of those two identities `excluded_conflicting_labels`, and write `label_conflicts.json`. Identical copies with consistent labels are extracted once. No majority voting, invented labels, or semantic identity inference was introduced. Existing grouped splits remain unchanged for unaffected observations.
- A dry check against the real failed-run manifest found 1,934 eligible unique images, 1,191 compatible cached records, 743 remaining images, and zero remaining eligible cache/label conflicts. The excluded records remain in the historical artifacts; no dataset files were deleted.
- Extraction tests: 7 passed. Combined parameter/header/scheduler/extraction/coverage/replay checks: 37 passed in 22.48 seconds. A two-epoch fixture-based neural training/export/load smoke test also completed; its score is a software check, not measured disaster-data performance.
- Rebuilt extraction notebook submitted as version 2 of private `hmmmkk/disaster-eight-field-ollama`; initial status QUEUED. This is not evidence of successful execution. Download target for this attempt: `results/disaster_decoder_v3/20261022_stageA_v3`.
- Runtime extraction remains supported by composing `FrozenImageExtractor.extract` with `ParameterExtractor.extract` and query-before-commit transmitter memory. Offline scheduler experiments reuse only frozen image-model outputs; task/history-dependent header values are generated causally for each episode. The ranker is trained on serialized-and-parsed headers, not images or evaluator labels.
- Remaining: full extraction completion, measured FLOP profiles, Ollama/Qwen runtime verification, complete real-data scheduler fitting/comparison, additional integration checks, and final reports. No superiority or codec reconstruction claim is supported yet.
- Follow-up training checks: all 11 study tests passed in 33.58 seconds, including two-epoch train/export/load checks for coverage, ordinary pairwise, classifier and regressor objectives, and three-round checks for standard/custom tree objectives.
- A separate real-feature smoke run completed the same six training/export/load paths using recovered image features from A and C only, excluding the two contradictory identities. No official test rows were used. Checkpoints, model exports, histories, and `smoke_status.json` are saved under `results/disaster_decoder_v3/20261022_real_feature_smoke`. This tiny smoke experiment is not the full comparison and does not establish model superiority.

## 2026-10-23 — Stage A extraction audit and training handoff

- Stage A extraction artifacts under `results/disaster_decoder_v3/20261022_stageA_v3` completed; `artifact_audit.json` reports `passed`. The audit matched 1,934 unique feature IDs and labels, finite values, 512-dimensional embeddings, disjoint role groups, pinned producer/source hashes, and exclusion of both conflicting IDs. All three frozen image models succeeded on all 1,934 retained IDs.
- Counts: source validation/test rows were 890/1,049. The manifest had 1,939 rows and 1,936 unique IDs before four conflicting-label rows (two exact image IDs) were excluded. Retained role counts by unique image ID are A=445, B=178, C=267, test=1,044. The extraction summary's test count of 1,045 is pre-deduplication; it must not be reported as the unique test count. Consistent exact duplicates were extracted once.
- The artifact audit confirms eight inference-cost profiles (four per Tesla T4) and 32 schema-valid Qwen proposals. Registered-operation inference estimates: CLIP 8.81762304 GFLOPs, LADI reference 32.6971392 GFLOPs, and DINO 665.607–746.230 GFLOPs depending on input shape. These are not full hardware FLOPs. Qwen full-model FLOPs were not measured; partial analytic text estimates are not comparable. Qwen annotations remain silver-only.
- Provenance: extraction was uploaded from commit `02580b0921265ac0f0c30177d6ea60e7f8e6e368` with a dirty working tree recorded. `source_manifest.json` pins source-file SHA-256 values and uploaded ZIP digest; use these with environment and artifact hashes for reproduction rather than treating the commit alone as sufficient.
- Training protocol clarification: all methods must use the same fixed 128 A-role queries per task, not a fresh set of 128 each epoch. Protocol is 13 methods (12 baselines plus main CAHR), 60 epochs maximum, patience 10; L2 grid `[1e-5, 1e-4, 1e-3]`; tree leaves `[7, 15, 31]`, max 200 iterations, learning rate .05. Finish all 13 on core seed 0 before seed 1/2 for main and strongest baseline, subject to explicit deterministic/tree exemptions. Nine retrained ablations remain as specified by the training protocol.
- Private Kaggle kernel `hmmmkk/disaster-eight-field-scheduler`, version 1, is RUNNING. No completed epoch or final benchmark result has been verified. Full performance, final comparison, locked-test evaluation, additional seeds, ablations, and final replay remain pending. A strong-baseline short-Sol audit identified no other default blocker. A resumed checkpoint matched uninterrupted execution; the strict published-cache gate was tested. Neither fact establishes full training.
- Validation boundary: the reported 45 tests passed before the latest report/metric additions; selected resume and three-task end-to-end fixture checks passed afterward. Two-epoch fixture and short real-feature A/C checks remain smoke tests, not study results. Do not claim journal readiness, superiority, a real codec benefit, real MAC deadlines, or life-safety impact.
- Documentation checkpoint: `RESEARCH_REPORT.md`, `SIMPLE_REPORT.md`, and `ARTIFACT_INDEX.md` are drafts that identify verified extraction evidence and leave full-benchmark conclusions pending. Update them after actual training completion and review; preserve the exact A/B/C/test role semantics and silver-label boundary.

## 2026-10-23 — Stage B benchmark completion and comparison download

- Private Kaggle kernel `hmmmkk/disaster-eight-field-scheduler`, version 1, completed; recorded elapsed time is 470.9174869 seconds. Downloaded artifacts are under `results/disaster_decoder_v3/20261023_stageB_v1/`; published comparison is `monitor_snapshot/study/comparison.csv` and the broader study tree is `full/study/`.
- The comparison CSV has 39 rows (all 13 methods for each of three tasks), 960 episodes per task, and 24/10/32 disjoint source pools for building/road/flooding. Undefined-A10 counts are 376/752/96 respectively. The task-level A10 table, all methods, and paired comparisons are recorded in `RESEARCH_REPORT.md`. Pools—not generated episodes—are the statistical unit.
- C-selected candidates were `coverage_tree` for building, `regressor` for road, and `coverage_tree` for structural flooding. Selection records are separate from held-out descriptive comparison results. The A10 values are mixed; `importance` exceeds CAHR for road and flooding, and paired source-pool CIs for CAHR minus the C-selected method include zero for building and flooding. Do not claim a universal win.
- Paired CAHR-minus-C-selected results: building -0.014396 (95% CI [-0.036446, 0.006614], 24 pools); road +0.057567 ([0.000000, 0.159633], 10 pools); flooding +0.026629 ([-0.018008, 0.073700], 32 pools). These generated source pools are not independent disasters.
- The job's `status.json` reports all three task studies completed, `codec_evaluation=false`; `data_audit.json` records `test_used_for_selection=false`, cache hashes, and role counts. Stage B includes C selections, seeds, nine ablations, and abstract-tick streaming outputs. Full artifact/prediction integrity auditing is still pending, so completion does not itself certify every output.
- `INFRA/learning/decoder_v3_study.py` source hash reportedly matches the published source. A combined 49-test check passed; three receiver-queue wrapper tests also passed. Three newer analysis tests remain to be included in the complete 52-test run. `INFRA/decoder_queue.py` is an integration wrapper validating complete jobs, header CRC/token/profile, arrival constraints, and F-only updates; it does not alter model training.
- The final executed notebook with captured outputs has not yet been published. The intended resume is to reuse primary checkpoints without test retuning, capture actual outputs, and verify exact agreement with the downloaded 39-row comparison. Do not report notebook-output verification as complete before that parity check.
- Graphify AST-only refresh was attempted but could not run because the `graphify` module is unavailable. No package was installed and no architecture change was made.
- Documentation now records the completed comparison but keeps full artifact audit, full test run, and final notebook parity as open verification gates. No journal-readiness, real codec, MAC deadline, or life-safety claim is supported.

## 2026-10-23 — final prototype completion and report correction

- This entry supersedes the pending verification statements above. The primary Kaggle study completed in 470.9174869 seconds. All 480 selected files were downloaded and hashed, all seven original source hashes matched, and all 12 gzip traces parsed to EOF with zero invalid rows (1,205,010 rows total). Generated pycache files were excluded. A transient transfer/path-guard failure was recovered and retained in the transfer ledger.
- Post-analysis reloaded every selected-per-method exported model and reproduced all 37,440 model/episode orders from serialized headers. Each task offered 32 source pools/960 episodes; only 24/10/32 building/road/flood pools contributed defined positive-offer A10. Zero-utility episodes were recorded, not assigned artificial scores.
- All three task comparisons, building-only seed-1/2 runs, all nine retrained building ablations and finite streaming replay passed audit. No-backlog waiting was zero. Replay verified no future selections, no overlapping service and no dropped/duplicated jobs. Actual codec bytes/inference were not evaluated.
- Main-thread receiver integration added `INFRA/decoder_queue.py` with completeness, CRC/token/profile binding, duplicate-admission and arrival checks. Only F can change from a trusted receiver clock. The policy never receives payload references or evaluator labels. Runtime usage is documented separately.
- The combined current test command passed **53 tests in 48.45 seconds**. Checkpoint resume matched uninterrupted training; the actual building CAHR checkpoint also resumed against the real pinned inputs without refitting/tuning.
- Final Kaggle notebook version 1 captured actual execution outputs. A visual audit found that the original Figure 1 episode average differed from primary source-pool weighting. This was corrected in reporting only: repetitions -> strata -> pools, and a fixed comparator list including the C-selected policy replaced test-best curve selection. Models, predictions, quality metrics and selection choices were unchanged. A regression test verifies plot area agrees with primary A10.
- Final notebook version 2 completed in 297.4164031 seconds. Its 39 quality-metric rows matched the original primary comparison within 1e-12; timing columns were measured again and are not expected to match. Its three executed cells contain a 39-data-row HTML table, two embedded PNGs matching downloaded files byte-for-byte, and no errors. Notebook SHA-256: `2cb2fae87b14d37c52b37f065d2af872bfb34907d9b7b5d7ba49319ae36a04e0`.
- The executed notebook was copied to `Scheduler/notebooks/05_disaster_decoder_v3.ipynb`. From-source template generation uses a separate `_source.ipynb` name so it cannot overwrite the delivered executed notebook. Original runs and corrected reporting artifacts remain separately preserved.
- Results are mixed: CAHR A10 was .7634/.5301/.6317 for building/road/flooding, versus importance-only .7580/.5953/.6331. C selected coverage trees/regressor/coverage trees, and test scores did not replace those choices. No universal or statistically robust CAHR superiority is claimed. Building seed variation and ablations are reported, including the unavailable-F zero-effect control.
- Short Sol-high audits found and helped correct a skipped strongest-tree seed path and misleading report denominators/building-only auxiliary scope. The final reports distinguish real image evidence from constructed queues and exclude codec, MAC deadline, life-safety and journal-readiness claims. Qwen's 32 proposals remain silver; complete Qwen VLM FLOPs were not measured.
- The original 24-hour target was not met. Scope cleanup was additive, with a current project-map entry and isolated handoff package; no legacy files, datasets or teammate codec work were removed, and no Git commit/push occurred. Graphify refresh remained unavailable because its package was absent.

## 23 October follow-up: original comparison and Qwen repair

- Inspecting the historical 32 valid Qwen JSON outputs found universal abstention
  across all three tasks. The prior format-only success description was corrected;
  these outputs never supplied scheduler training targets.
- Main-thread implementation added a bounded development-only Qwen diagnostic,
  explicit structured schema, absence-versus-unassessability rubric, independent
  prose vision checks and synthetic transport control. A separate private Kaggle
  version completed; all 27 published files/logs were downloaded under
  `results/disaster_decoder_v3/20261023_qwen_repair_v1/`.
- Qwen correctly described the known red-square/blue-circle control. Two of four
  role-A images became assessable per task; one native building-damage-positive
  observation still received zero evidence grade. No positive road/flood cases
  were in this tiny pilot. The VLM is not accepted as a priority-label generator.
  Native labels and calibrated supervised extraction remain the primary route.
  The combined prompt/schema/resolution changes do not isolate the failure cause.
- `scripts/compare_original_schedulers.py` successfully ran the original eight
  supported methods on identical headers. Terminal default is ordinary neural
  RankNet, with CAHR explicitly optional. SJF needs actual costs, EDF needs valid
  deadlines, and the original deferred methods remain untrained. No historical
  model, selection or held-out score was changed. The walkthrough's repeat-first
  order is not alone proof of a semantic error; that requires a reference objective.
- Combined regression checks passed **56 tests in 31.11 seconds**. No Gemini key,
  paid call, Git push or expanded scheduler training was used. The immutable first
  handoff ZIP predates these follow-up additions; current source/docs and the new
  result directory are authoritative for this checkpoint.
