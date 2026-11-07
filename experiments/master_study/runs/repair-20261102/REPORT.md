# Scheduler repair report — 2 November 2026

## Outcome

Confirmed implementation faults were repaired in the canonical End2End source. Existing v9 models, datasets, notebooks/results and earlier failure logs are preserved. No new real-data training, vision extraction, teacher inference or Kaggle job was launched.

This is a validated code repair and corrected experiment pipeline. It is not evidence that newly repaired models outperform SPI-first: the saved models were trained before these repairs. Missing independent labels and unqualified VLM judgments remain explicit scientific blockers.

## Repairs mapped to the audit

| Fault | Implemented correction | Validation / remaining limit |
|---|---|---|
| D cannot teach mission/evidence preference | Preserved D; separate E training/evaluation path accepts explicit evaluator-only binary mission utility. Every E factory rejects missing independent reference weights; source/provenance is required. | Independent-reference fixtures pass. Real utility manifest remains unavailable; no predicted R/N is used as truth. |
| Six-thousand pool misrepresented as multi-domain training | Shared cohort assignment, exact-hash dedup, per-source reference coverage, optional required-source gate. Master general study checks required sources before GPU/model loading. | Real-cache preflight exits2 as intended because Open Images/Places365 priority references are absent. CrisisMMD train/validation remain3099/780. |
| Prefix-only training samples | Complete causal trajectories, stratified bounded reservoirs across phase and queue size, mixed behavior policies. | Same-size512 cache pilot now has106 pre-burst,204 burst,202 post-burst samples; max queue93 versus11 before. |
| ListMLE drops two-level queues | Tie-aware per-item strict-lower listwise surrogate; every higher-grade candidate is trained against lower grades without arbitrary tie ordering. Adaptation is identified in model metadata. | Gradient-direction, ties and bad-high-item counterexample tests pass.463/512 new pilot states provide listwise signal. |
| Uniform pair importance | RankNet weights reference pairs by gain gap, with explicit low-weight ties; teacher preference labels retain their separate loss semantics. | Unequal-gap regression passes. No claims of a measured performance gain. |
| PPO critic aliases queue counts | Critic now consumes1172-dimensional queue/history summary, including counts. Batch/single paths agree. E resume preserves its objective. | Queue-count discriminability and broad learning/export/resume tests pass. |
| Failed jobs credited as useful evidence | Successful processed pool and coverage commit only after successful completion; incomplete/failed attempts do not commit. Temporal windows encode failed status3. | Failure, incomplete, completion-time credit and no-retry regressions pass. D still deliberately measures wait to dispatch. |
| Cached uncertainty discarded | Retain/reorder all four registered SPI producer scores. Record producer identity, score order and uncalibrated status in profile binding. | All6000 cached rows retain soft scores in real-cache pilot. Calibration is not claimed; revised profile must not silently reuse old artifacts. |
| Simple controls mistaken for scheduling qualification | Added actual swapped-preference reliability gate and early stop after8 completed pair contexts when unreliable. Prompt distinguishes current candidates from historical context. Audits write to new output directories, not historical inputs. | Old v9 teacher agreement17.5% is refused. Actual improved VLM accuracy remains unmeasured. |
| Neutral guide called informative VLM | Nonconstant replay support required; saved support report explains refusal. Prefix reward fitting also requires actual preference consistency. | Existing neutral guide is labeled inactive in offline report. Informative replay still requires new valid labels and a matched ablation. |
| Feedback/D mismatched traces | Feedback uses shared grouped roles and mixed-stream validation with the same UE/load/seed definitions as D. | Shared-role isolation fixtures pass. Teacher supervision remains weak, not independent gold. |
| Timeout prevented reporting | Method ledger exists before training and refreshes from child status. Fatal preflight failures propagate nonzero exit. Standalone saved-run reporter works without training. | Controller fixture runs all8 stages. Saved v9 inventory and delay CSV regenerated independently. |
| Aggregate hides Critical regressions | Offline reporter emits explicit Critical waiting/miss safeguards against matched SPI-first metrics. Missing/unmatched evidence fails the guard instead of passing. | This is an engineering safeguard on development data, not a statistical final-test certificate. |

## Validation actually executed

- Final broad local suite:321 passed,3 skipped in255.81seconds.
- Explicit roots: parameters/tests, Scheduler/tests, Test/tests, experiments/stream_scheduler/tests, experiments/master_study/tests.
- Isolated packaged snapshot:13 passed in6.58seconds, with parent pytest configuration disabled.
- Prepared notebook:5 code cells compile;170 packaged source files,537224bytes at preparation. No upload/submission.
- Real-cache coverage pilot:64 complete episode trajectories,512 sampled states; no parameter model fitting/inference.
- Real-cache general-study preflight: expected exit2, recorded as successful refusal of unsupported source coverage.
- Old teacher qualification pilot: actual candidate preference consistency17.5%, correctly rejected.
- Offline v9 inventory and shared D comparison regenerated without requiring master completion.

Final broad execution used the existing WSL environment: Python3.14.4, pytest9.2.0, Torch2.15 CPU. These differ from older project dependency declarations. Tests establish behavior on this observed environment, not validation of every declared/Kaggle environment.

Skipped: one two-GPU CUDA test and two exported SwinJSCC tests requiring external model files. Codec work remains outside the user's requested scheduler repair. No GPU performance, new convergence, physical timing, or independent final accuracy was established.

Earlier failed validation and pilot attempts remain preserved. The new real-cache pilot exposed and led to correction of a score-vector variable shadowing bug; final repeat completed successfully. Synthetic teacher fixtures were updated to supply the newly required explicit context/swap metadata; the historic-teacher test now expects rejection on actual preference inconsistency. Tests were not changed to accept invalid teacher labels.

## Source and artifact locations

- before/: original touched sources preserved before core edits.
- validation/: first broad run and actual failure traces.
- validation-final/: final broad tests, environment metadata and source hashes.
- packaged-tests.log: isolated source-capsule validation.
- pilot/pilot.json and pilot-verified.log: measured coverage, probabilities and old-teacher rejection.
- input_preflight/: per-source label coverage and refusal reason.
- input_preflight_exit.json: expected-failure acceptance record.
- offline_report/: inventory.csv, delay_comparison.csv, critical_safeguards.json, summary.json and REPORT.md.
- notebook_validation.json: prepared notebook identity, compilation scope, submitted=false.
- runtime_snapshot/: disposable pinned execution capsule, not a second development codebase.

## Use and recovery

Canonical notebook remains experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb at the existing Kaggle identity. It runs CPU reference checks before expensive setup and refuses unsupported general-study labels. It includes packaged scientific regressions; this packaged subset is not represented as the entire local test suite.

For a saved run, use experiments/master_study/summarize_saved.py --results <saved-results-root> --output <new-report-directory>. This does not train.

A scoped D experiment may explicitly require only supported sources. A general multi-domain result must satisfy the broader reference gate. An E experiment requires a scenario manifest whose records include provenance, reference_kind, reference_source and utility_by_mission with independently justified binary values. The code does not infer those labels from runtime parameters.

Historical unit-weight E is retained only as a labeled exact-fact control. New mission training factories reject absent independent weights. E facts are mission-qualified, and duplicates with conflicting fact utility are rejected.

## Scientific requirements that code cannot manufacture

1. Independently supported priority labels for the ordinary-image sources.
2. A valid mission-utility/correspondence manifest for the intended semantic study.
3. Reliable teacher preferences and nonconstant supported replay ratings.
4. Information-sufficiency checks for H0: scalar relevance and counts/areas cannot express every relationship or identify every mission.
5. A supported natural sequence cohort for natural temporal-change claims.
6. Independent final evaluation, calibration and targeted ablations.

These are now visible requirements rather than silent assumptions. They have not been fabricated or reported as completed. The completed legacy models remain useful development controls; the repaired protocol must be evaluated before any improved-performance claim.


## Subsequent skip review
The historical two-device channel test incorrectly placed both shards on GPU0. Corrected second shard to GPU1 and asserted both device indices. CPU checks:2 passed,1 two-CUDA-device skip. Corrected cross-device behavior remains unexecuted on this CPU-only environment. Prepared notebook refreshed; not submitted. The earlier321-pass broad result predates this test-only correction.
