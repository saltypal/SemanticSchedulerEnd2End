# Notebook version policy after reorganization

Local BetterScheduler paths are now parameters/. Historical paths, slugs,
version IDs and checksums below are retained as provenance. Update the existing
owner/slug with a new version for future runs; do not create a fresh notebook
identity for each edit. No notebook was submitted during this reorganization.

# BetterScheduler Kaggle artifact map
## SPI teacher-label and student development lab — candidate b (version 1 ERROR)

- Candidate `staging/spi_20261026b/`, private kernel `satyapaladugu/betterscheduler-spi-20261026b`, source archive SHA-256 `13790a342e83e1b2d10462fdb9e0e35a74b7f90bf014d0aaa33587f07cb00078`; source snapshot commit `1b45ab60c9e72ec5b426ff4488a92ab0c4fd043d`. Manifest contains 90 files plus the separately declared `SOURCE_MANIFEST.json`; 87 non-stub source files match live files and the 3 intentional isolated namespace stubs match the exact builder literal and frozen hash. Notebook nbformat validation and all 4 code-cell compiles pass; embedded archive matches.
- Candidate b input fix accepts either original ZIP or Kaggle's observed expanded layout and verifies panel/features plus every image SHA before model download; the campaign repeats verification before use. If archive is expanded, the original archive hash is unavailable by platform design; exact member hashes are checked. Unit tests cover both layouts and corrupted image rejection. This keeps rubric, model, and data unchanged.
- Private attached input `satyapaladugu/betterscheduler-spi-inputs-20261026a`, version 1, status ready, 183 files: 180 SHA-named PNGs under `images/images/` and `features.npz`, `input_manifest.json`, `panel.json`; no literal `images.zip`. Remote PNG names match all 180 panel hashes and advertised sizes match local ZIP members. Remote inventory cannot expose the platform-expanded original archive hash. No reupload performed.
- Pre-submit checks: 10 SPI engineering tests passed (5.534 s; not urgency validation); candidate archive, notebook schema/compilation, metadata, input contract and exact v1 expanded inventory passed. Receipts: `results/validation/spi_20261026b_20261026T153916Z/` and `results/ops/spi_20261026b_prelaunch/`. Kaggle target search returned Not found before launch. Account quota showed 27.46 GPU-hours remaining; the two most recent visible Scene Suite kernels were COMPLETE. No unrelated jobs were stopped.
- Submitted once using Kaggle CLI 2.2.4: `kaggle kernels push -p staging/spi_20261026b -t 3600 --accelerator NvidiaTeslaT4` from `BetterScheduler/`. Requested machine `NvidiaTeslaT4`, server timeout 3,600 s, maximum aggregate allocation 2 GPU-hours; notebook requires exactly two visible GPUs, runs Qwen3.5-9B NF4 on GPU1 with thinking off. Submission receipt/status and live cell evidence follow below.
- Retrieved runtime outputs are under `results/ops/spi_20261026b_prelaunch/monitor/retrieved/`. The recovery ZIP and 28 listed checksums verified; no student checkpoint exists because the teacher gate stopped before fitting.

- Terminal result: Kaggle kernel version 1 ended `ERROR` after cell 2 ran 344.59 s (setup cell 1 ran 49.30 s); its experiment `status.json` is `incomplete`, 10 images attempted, `offline_teacher` failed its 8-of-10 parseable-response preflight after 175.31 s. All 10 generations finished (43–50 tokens each, 5.19–5.89 s); all failed strict SPI parsing because the model returned only `evidence`, `level`, and `reason`, omitting required boolean `available`. These are parser failures, not valid abstentions; stored `available=false` is failure fallback. No usable pseudo-labels and no student fit were produced. Thinking-disabled template receipt and finite-logit preflight passed for the first 10. No retry or source change was made.
- Runtime: Kaggle Python 3.13.15, Torch 2.11.0+cu128, Transformers 5.3.0, two Tesla T4 devices. Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` loaded NF4 on `cuda:1` (358 Linear4bit modules); model shard hashes are in the saved first-row provenance. The verifier passed all 180 expanded image hashes before model download and again in the campaign. Account GPU quota after this run: 2.66 h used / 27.34 h remaining; no unrelated jobs were stopped.
- Retrieved evidence: `results/ops/spi_20261026b_prelaunch/monitor/retrieved/`. It contains `cell_evidence/executed.ipynb`, three cell logs/status (cells 0/1 complete, cell 2 failed), runtime/device metadata, campaign status/progress, `spi_labels.json`, all 10 raw response and failure JSON pairs, recovery ZIP, and the Kaggle kernel log. Recovery ZIP SHA-256 `9c1f0c9d832566831250aaba05d9d9b15f86cc156c4845dc2e5fc12038eefc34`; all 28 checksum-manifest entries match, with `checksums.json` as the expected extra ZIP member. Inventory and integrity receipt: `results/ops/spi_20261026b_prelaunch/retrieved_inventory.json`. Exact first raw response and failure are under `results/ops/spi_20261026b_prelaunch/monitor/retrieved/results/spi/20261026b/raw_labels/01adbd5612ed4272802ad95437bab3c6e929d0b89d0f2def8a76baead7c03f7d.txt` and `.failure.json`.
## SPI teacher-label and student development lab — candidate 20261026a superseded; 20261026b ERROR

- Status: tests, cached input preparation, input integrity, and notebook package preflight passed. The private input dataset is **READY, version 1**; candidate a was blocked by Kaggle's automatic ZIP expansion and never submitted. Candidate b verified the expanded input and was submitted once, then stopped at the 10-response parser preflight; see the 20261026b section above.
- Purpose: one bounded model-assisted image SPI labelling run and CPU SPI-head fit on a development-exposed cache. Qwen labels are pseudo-labels, not independent human urgency truth. Native LADI/MEDIC annotations and source labels are excluded from the teacher input.
- Input package: `results/spi_inputs_20261026a/`, 180 512x512-or-smaller RGB PNG derivatives (52,925,486 packaged image bytes), `features.npz` (180x512 finite normalized CLIP vectors), and `panel.json`; 110 train / 70 validation images, 94 / 62 disjoint source groups. Input manifest schema `spi_inputs_v1`; content hashes and image joins pass in `results/validation/spi_20261026T152205Z/package_verification.json`. The package is below the 250,000,000-byte cap and contains no model weights or native labels.
- Private input: `satyapaladugu/betterscheduler-spi-inputs-20261026a`, version 1, status `ready`, created privately with Kaggle CLI 2.2.4 (no `--public`). Metadata file: `results/spi_inputs_20261026a/dataset-metadata.json`, `other` license entry, LADI/MEDIC source notice. The first create invocation using a forward-slash relative `-p` path failed locally before remote creation because Kaggle's Windows resumable-upload state path retained the `/` separators. After authenticated listing confirmed the target absent, the one lead-authorized correction ran from the dataset folder with `kaggle datasets create -p .`; logs and receipts are `results/ops/spi_20261026a_20261026T152708Z/dataset_create.log`, `dataset_create_attempt_01.json`, and `dataset_create_attempt_02.json`.
- Remote dataset inventory is persisted at `results/ops/spi_20261026a_20261026T152708Z/dataset_remote_inventory.json`: 183 listed files, including 180 PNGs under `images/images/<sha256>.png` plus `features.npz`, `input_manifest.json`, and `panel.json`. All 180 remote image basenames match the panel's image SHA names and reported sizes match the packaged PNG sizes. Kaggle expanded the uploaded `images.zip`; the literal `images.zip` is absent. Candidate `spi_20261026a` expected the literal ZIP and was never submitted. Candidate `spi_20261026b` accepts this exact expanded layout and verified every image SHA-256. Keep private dataset version 1; no reupload or reconstruction.
- Frozen notebook candidate: `staging/spi_20261026a/`, private kernel `satyapaladugu/betterscheduler-spi-20261026a`, attached input above, `NvidiaTeslaT4`, server timeout 3,600 s, maximum aggregate allocation 2 GPU-hours, thinking off. Runtime asserts exactly two visible GPUs, then runs Qwen3.5-9B NF4 on GPU1; cached CLIP features avoid another encoder pass. Source commit recorded in package manifest: `1b45ab60c9e72ec5b426ff4488a92ab0c4fd043d`; `source.zip` SHA-256 `ba279221cd2af8a8af0da3aabb5e4b1fbb14c22ca5d09a0dbcb43a589de63fd7`. Preflight verifies 87 source files byte-identically against live files and the three intentional namespace stubs against the builder's exact declared bytes. Notebook schema and code compilation pass. Receipt: `results/ops/spi_20261026a_20261026T152708Z/candidate_preflight_final.json`.
- Initial candidate a local checks: 9 tests passed; candidate b re-ran the expanded-layout suite with 10 tests passed in 5.534 s. Engineering fixtures only, not real urgency validation. Candidate b log and runtime/source-hash receipt are under `results/validation/spi_20261026b_20261026T153916Z/`.
- Candidate `staging/spi_20261026a` remains private and was never submitted. Candidate `spi_20261026b` contains the expanded-layout reader with per-image SHA checks; rubric, label definition, and model definition stay unchanged. See the submission and monitored result in the candidate b section above.
- Expected outputs from the blocked candidate are `/kaggle/working/results/spi/20261026a/`, `/kaggle/working/cell_evidence/`, `/kaggle/working/runtime.json`, `/kaggle/working/spi_checksums.json`, `/kaggle/working/spi_recovery.zip`, and the executed notebook/cell logs exposed by Kaggle. Candidate a was superseded before submission; no outputs exist for it. Candidate b output persistence and actual runtime evidence will be recorded separately.


## Review-only follow-up 20261026i — terminal result

- Kaggle version 1 is **COMPLETE**: https://www.kaggle.com/code/satyapaladugu/betterscheduler-qwen9b-review-20261026i. Experiment status is `complete`, 30/30 reviews, worker return code 0; four notebook cells completed. The metadata ID differs from Kaggle's canonical title-derived URL slug; status and artifacts were retrieved through the returned URL slug.
- Results: 24/24 controls yielded valid outputs and 16/24 joint correctness (consistency balanced accuracy 0.7083; sufficiency balanced accuracy 0.7500). Six actual JSONs yielded valid outputs and 4/6 joint correctness; the two misses were explicit negative claims falsely treated as missing evidence. All 30 responses have `enable_thinking=false` and `thinking_disabled_verified=true`; none hit the token cap. Main's independent report: [`docs/SEMANTIC_VALIDATION_I_RESULTS.md`](docs/SEMANTIC_VALIDATION_I_RESULTS.md).
- Runtime: Qwen3.5-9B NF4/FP16 compute on GPU1 of two Tesla T4s, Transformers 5.3.0 / CUDA 12.8. Peak measured GPU1 allocation 11.811 GiB; review worker 617.17 seconds.
- Retrieval: `results/ops/20261025T2356Z/review_only_package/retrieved/` includes 12 images, 30 raw response texts, review JSONL/metrics, the executed notebook and four cell logs. All run artifacts verify **69/69** against `artifact_checksums.json`. The downloaded remote source manifest exactly matches the submitted archive manifest; all **69/69** source files verify inside the source ZIP (SHA-256 `6cd5e8abc3d99113eff24b53a7b601bf0bc08db6687eae2e5caf08828b01a566`). The selected retrieval filter did not separately download the source `.py` tree; no second download was needed.
- Integrity receipts: `results/ops/20261025T2356Z/review_only_package/terminal_audit.json`, `terminal_checksums_verified.json`, and `source_manifest_archive_verified.json`; concise operational report: `results/ops/20261025T2356Z/review_only_package/FINAL_REPORT.md`.
- The status monitor saw COMPLETE at 00:14:01 UTC, then hit a Windows `charmap` error while the SDK rendered the download response. Its retry handling repeatedly attempted terminal retrieval. I stopped only that local monitor process, retained downloaded files and verified both manifests locally without another download. Kaggle execution was unaffected.


## Balanced semantic validation — 20261026h3 (Kaggle COMPLETE; experiment PARTIAL)

- This is one bounded corrective attempt after h2a failed at 9B environment bootstrap and Qwen4B stopped after three controls under its conservative time gate. Earlier h, h2a, and staged h2 artifacts remain preserved; no retry of those immutable notebooks.
- Frozen snapshot `8947b399c6d0c178fe4cf870aa8a4700a960cf35`; the full suite was reported by main Sol as 80 passed in 12.07 s. h3 package is `staging/20261026h3/`, tag `20261026h3`, source ZIP SHA-256 `4963e4baf2cf051fb7632881a7d0ab199e3697704446321a463c059f75b2c2d0` (175,500 bytes). All 66 source entries match the frozen live snapshot except three intentional namespace stubs; notebook embeds the exact ZIP and all four code cells compile.
- h3 keeps the 12-image extraction and selects a preregistered budget subset of two parent images per native stratum, yielding 24 controls plus 6 actual parameter cases for each model (30/model; 60 total). The full extraction and full evaluator-only cases remain saved separately in the run. Both reviewers use the same selected cases and thinking remains off. This is a small budget feasibility pilot, not a powered accuracy comparison.
- Configuration pins Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` / Transformers 4.57.6 and Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` / Transformers 5.3.0. The corrected 9B setup uses a venv without pip, installs through the supported base-interpreter `pip --python <target>` mechanism, then checks required imports before either model review worker downloads VLM weights. The prior public metadata check verified these pinned revisions and the 9B chat template; no local weights were downloaded.
- Caps: 1,200 server seconds, 1,050 internal seconds, two T4 GPUs. Kaggle quota preflight at `2026-10-25T22:56:15Z` showed 27.90 GPU-hours remaining, refresh 31 October. Recent relevant kernel statuses were checked; all were terminal. h3 target was not listed before submission. Cumulative conservative wall-equivalent accounting across h, h2a, and h3 is 1,605.783 seconds on two GPUs, below the 1,800-second cap equivalent to one aggregate GPU-hour. Detailed preflight and package evidence: `results/ops/20261025T2256Z/semantic_validation_h3/`.
- Submitted once with Kaggle CLI 2.2.4; version 1 accepted at `2026-10-25T22:59:16Z`. URL: https://www.kaggle.com/code/satyapaladugu/betterscheduler-semantic-validation-20261026h3. Kaggle reports COMPLETE at `2026-10-25T23:11:48Z`; the experiment's own status is PARTIAL after 604.89 internal seconds. The exact push command used `-t 1200 --accelerator NvidiaTeslaT4`; submission receipt is `results/ops/20261025T2256Z/semantic_validation_h3/submission_receipt.json`.
- Extraction completed for 12 panel images, 15 observations, 48 controls, and 12 full actual-review cases; the budget-selected cohort was six parents (two/stratum), 24 controls plus six actual cases per reviewer. Seven extracted parameter families were available on all 15 observations; the native classifier was correct on 24/24 comparable facts. The 9B isolated environment install/import gate passed before either reviewer loaded weights. Each `model.json` records `enable_thinking=false`; each separate `processor_preflight.json` records `thinking_disabled_verified=true`. Both used 192-token generation and the 262,144-pixel cap on `cuda:1` across two Tesla T4s.
- Each reviewer had 30 planned cases: 24 controls plus six actual parameter cases. Qwen3-VL-4B/FP16/SDPA (revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, Transformers 4.57.6) completed 24/30 total cases, all controls, then failed on its 25th attempted case (`actual-0`) with CUDA OOM while requesting another 3.08 GiB (2.13 GiB free). It produced no valid actual-case review. Qwen3.5-9B/NF4 with FP16 compute/SDPA (revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, Transformers 5.3.0) loaded successfully but stopped after its three-control technical gate: projected remaining time 450.01 s exceeded 441.78 s available. It completed 3/30 total cases, all controls, and produced zero actual-case reviews. The run supports no quality comparison between reviewers and no scheduler-performance claim; no retry was launched. See `docs/SEMANTIC_VALIDATION_H3_RESULTS.md` for the independent metric recomputation and semantic analysis.
- The original terminal receipt records a `charmap` error while rendering output during retrieval. The UTF-8, `force=false` correction retrieved 105 artifact paths into `monitor/retrieved/`; 83/83 global and 82/82 run checksum entries verify with no missing or mismatched files. It contains the executed notebook, four cell logs, both model partial outputs, all 12 images, extraction artifacts, and worker logs. The original receipt is preserved; corrected evidence is in `retrieval_verification.json`. Full bounded result summary: `results/ops/20261025T2256Z/semantic_validation_h3/FINAL_REPORT.md`.

## Corrective balanced semantic validation — 20261026h2a (Kaggle COMPLETE; experiment FAILED during 9B setup)

- The preceding h notebook completed at the platform level but failed before the source panel, due to the materializer rejecting `Run`'s empty pre-created `data/` directory. The original h source/logs remain preserved. Its conservative push-to-terminal upper bound, including queue time, is 146.484 seconds; this is below the 200-second gate for the approved 1,600-second corrective cap.
- An intermediate h2 bundle was built and validated before a reporting-only summary fix; it was not submitted and remains preserved under `staging/20261026h2/`. The corrected h2a bundle is built separately under `staging/20261026h2a/` and uses the frozen summary that reports actual artifact availability.
- h2a validation: 79 tests passed (11.59 s), focused driver/cap tests 4 passed (2.76 s), four notebook code cells compile, 66 source files match the manifest and live snapshot except the three intentional namespace stubs, and the embedded ZIP matches `source.zip`. ZIP SHA-256: `80482613a5c5c08cc5b26509c08a2ed5e9964137595100fa92e018e4455b2f0a`. Kernel metadata is private, internet-enabled, T4, and has no upstream/private dataset sources.
- The h2a caps are 1,600 server seconds, 1,450 internal seconds, and 0.888889 aggregate GPU-hours (two GPUs at the full server cap), within the original one-GPU-hour allowance. It retains the same 12-image, three-stratum LADI panel, model revisions, Transformers pins, 192-token setting, and thinking-off configuration. Public model and dataset revisions and row labels were verified metadata-only immediately before h2a; no local model weights or image bytes were downloaded.
- Pre-launch preflight: Kaggle account quota 27.96 GPU-hours remaining; h is COMPLETE; h2a slug was absent; latest 11 account kernels checked were terminal. Receipts: `results/ops/20261025T2238Z/semantic_validation_h2a/`; h2 remains preserved and was not submitted.
- Submitted once with Kaggle CLI 2.2.4; version 1 was accepted. URL: https://www.kaggle.com/code/satyapaladugu/betterscheduler-semantic-validation-20261026h2a. Kaggle terminal status is COMPLETE at `2026-10-25T22:45:59Z`; the experiment's own `status.json` is FAILED after 156.62 seconds. The notebook completed its reporting cells after capturing the failure, so notebook COMPLETE is not experiment success.
- Panel and extraction completed: 12 official LADI test images, 15 causal observations, 48 controls, and 12 actual reviewer cases. Seven parameter families were available on all 15 observations; the native classifier was correct on 24/24 comparable facts; repeat-novelty and sender-cold-start checks passed. Extraction used CLIP, YOLO-Worldv2-small, and the native LADI classifier on GPU0 (`cuda:0`); Qwen4B was on GPU1 (`cuda:1`). Hardware: two Tesla T4s, Torch 2.11.0+cu128, CUDA 12.8.
- Qwen4B completed processor preflight, then only 3/60 planned reviews (all controls) before its technical time gate stopped it. It produced 0/12 actual-parameter reviews. The partial three-control metrics do not support a model-quality conclusion. Qwen3.5-9B did not load or review anything: isolated environment creation failed because `/kaggle/temp/qwen35_env/bin/python3 -m ensurepip --upgrade --default-pip` returned nonzero. The captured traceback does not establish the underlying cause. No retry or follow-on run was launched.
- Retrieval root: `results/ops/20261025T2238Z/semantic_validation_h2a/monitor/retrieved/`; run root: `monitor/retrieved/results/semantic_validation/20261026h2a/`. The executed notebook, four cell logs, stage logs, partial review outputs, extraction artifacts, and all 12 images are present. The original terminal receipt is preserved at `monitor/terminal_receipt.json`; its initial checksum check used the wrong `20261026h` base. Corrected verification in `checksum_correction.json` derives the run root from the staged tag and verifies 43/43 global plus 42/42 run entries, with zero missing/mismatched files and no second download. See `results/ops/20261025T2238Z/semantic_validation_h2a/FINAL_REPORT.md` for scope and interpretation.

## Balanced semantic validation — 20261026h (Kaggle COMPLETE; experiment FAILED before extraction)

- Private target: `satyapaladugu/betterscheduler-semantic-validation-20261026h`; target slug was absent before submission. Staged package: `staging/20261026h/`; source ZIP SHA-256 `665d0ac86f479593881623ff9dd1dc391ba2b37d762169504084ea058e661b19` (175,302 bytes), embedded notebook archive identical, 66 source-manifest entries. All packaged source files match the live snapshot; the only deliberate namespace-stub differences are `INFRA/__init__.py`, `INFRA/learning/__init__.py`, and `Scheduler/__init__.py`.
- Preflight passed: full BetterScheduler suite 79 passed (11.74 s), affected driver tests 4 passed (2.56 s), all four code cells compile. Public metadata only confirms the two exact pinned model revisions and the LADI test revision; no weights or image bytes were downloaded locally. The Qwen3.5-9B repository includes `chat_template.jinja`. All 12 source rows' native labels and dataset revision were rechecked; the notebook refreshes image URLs at runtime and verifies image hashes.
- Panel: 12 new-to-this-study official LADI test-source images (4 absent, 4 present-unaffected, 4 affected); 15 causal observations across three UE histories, 48 constructed controls and 12 actual parameter JSONs per reviewer. This is a development pilot, not a locked final evaluation. No prepare-kernel/private-data dependency.
- Configuration: GPU0 shared feature/claim extraction; GPU1 sequential Qwen3-VL-4B FP16 (Transformers 4.57.6) then Qwen3.5-9B NF4 (isolated Transformers 5.3 environment at `/kaggle/temp/qwen35_env`); thinking disabled. 9B technical/runtime/memory gate may leave the run partial. Do not interpret a partial result as a completed comparison.
- Resource gate passed: Kaggle CLI 2.2.4 account quota 27.98 GPU-hours remaining, refresh 31 October 2026; latest 11 account runs are terminal. This run is capped at 1,800 server seconds, 1,450 internal seconds, and 1 aggregate GPU-hour. Launch command: `kaggle kernels push -p BetterScheduler/staging/20261026h -t 1800 --accelerator NvidiaTeslaT4`.
- Submitted once with Kaggle CLI 2.2.4; version 1 was accepted, observed RUNNING at `2026-10-25T22:33:18Z`, then COMPLETE at `2026-10-25T22:35:23Z`. Notebook URL: https://www.kaggle.com/code/satyapaladugu/betterscheduler-semantic-validation-20261026h. Submission receipt: `results/ops/20261025T2228Z/semantic_validation_h/submission_receipt.json`.
- Experiment `status.json` is `failed` at the first stage, `00_new_test_source_panel`: `semantic_validation_run.py:22` attempts `mkdir(..., exist_ok=False)` on `/kaggle/working/results/semantic_validation/20261026h/data`, which already exists. The driver caught this failure and the notebook summary cell still completed, so four completed top-level cells do not indicate experiment success. No image download, feature extraction, parameter generation, weight load, inference, isolated 9B environment, or review ran. Setup verified two T4s and the 4B processor-only, thinking-off preflight passed.
- A bounded monitor (PID 31096; 60-second status polls, 300-second log snapshots, 2,400-second maximum) recorded under `results/ops/20261025T2228Z/semantic_validation_h/monitor/`. Its first output retrieval failed with a Windows `charmap` UnicodeEncodeError. One UTF-8-mode retrieval correction (no notebook rerun) retrieved the executed notebook, four cell logs/status, failure trace, and receipts. Both manifests verify: global 6/6 and run 5/5, zero missing/mismatched entries, using nested base `monitor/retrieved/results/semantic_validation/20261026h/`. No image files were present because the source-panel stage failed before materialization. Full report: `results/ops/20261025T2228Z/semantic_validation_h/FINAL_REPORT.md`; correction receipt: `retrieval_attempt_2.json`. No retry/follow-on was launched.

## Reviewer repair follow-up — 20261025g (version 1 COMPLETE)

- Private target `satyapaladugu/betterscheduler-reviewer-repair-20261025g`; existing staged package preserved without overwrite after a builder collision. Package hash: `3dccefbbc1bab1e5f25271133b7a8315b92445cc596b545d7d341dff688ca307`; notebook embeds the same ZIP. Frozen source manifest has 56 entries; all archive hashes verified. All live production/config files match; the three namespace `__init__.py` files are intentional isolated-package stubs.
- Checks: 68 tests passed; four notebook code cells and four embedded source cells compile; isolated imports of reviewer follow-up and Qwen reviewer pass with model libraries stubbed. Current frozen processor path and 56-group exclusion/six-case cohort configuration match the package.
- Fixed Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, GPU1, FP16/SDPA, greedy 192 tokens, 512-side contain and 262,144 effective image pixels. v2 versus explicit-v4 prompt: six previously exposed cases with and without images (24 calls) plus 24 fresh controls per prompt (48), 72 total. The 56 prior groups are excluded. No extraction, scheduler training or automatic winner selection.
- Metadata audit found four of eight fresh groups annotated building-absent and four affected; no intact-positive building group. The study cannot estimate intact-building specificity.
- Terminal result: Kaggle reports COMPLETE at `2026-10-25T17:08:39Z`; the run's own status is complete. All four notebook code cells completed and all six review conditions produced 72 records in 478.72 internal seconds. 70 parsed successfully; two v2 diagnostic `control-4-complete` records (image and text-only) failed because the model returned an empty `reason`. No fresh-control response failed parsing and no generation hit the 192-token limit. There were zero passing fresh configurations and no selected configuration. This is not a scheduler result or evidence that a prompt setting is superior; main Sol is independently recomputing the six metric rows.
- Runtime evidence: two Tesla T4s, Torch 2.11.0+cu128, CUDA 12.8; Qwen inference and all model parameters were on `cuda:1` (GPU0 intentionally idle). All six provenance records name Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, FP16/SDPA, greedy 192-token generation, and 262,144 effective image pixels. The explicit processor size was shortest edge 65,536 and longest edge 262,144. No `I/O operation on closed file` or `Tee.write` error was found in retrieved logs/notebook output.
- Environment warning: cell 1 reports Hugging Face Hub 0.36.2 below the declared minimums of preinstalled Gradio 6.26.0 and Diffusers 0.40.0. The review run completed; those packages were not used in the inference path.
- Retrieval/integrity: terminal receipt, executed notebook (226,159 bytes), four cell logs/status, raw reviews/failures, and run artifacts are under `results/ops/20261025T165514Z/reviewer_followup_package/monitor/`. All 109 entries in the run checksum manifest verify with zero missing/mismatched files when resolved against `monitor/retrieved/results/reviewer_followup/20261025g/`. Full operational audit: `monitor/FINAL_REPORT.md`; machine-readable audit: `monitor/terminal_audit.json`.
- Cohort limitation: the eight fresh groups comprise four building-absent and four affected groups, with no intact-positive building group. This cohort cannot estimate intact-building specificity, and reviewer judgments are not independent visual ground truth. Preserve the failed records and do not retry the immutable run.

## Latest bounded model grid — 20261025f (COMPLETE)

- Terminal verification: Kaggle COMPLETE and run status complete; 1266.58 internal seconds. All 12 YOLO, 8 CLIP (160 rows), and 8 Qwen (192 reviews) settings finished. Zero Qwen configurations passed. All 240 manifest-listed run artifacts match SHA-256 when resolved against `monitor/retrieved/results/model_grid/20261025f/`; the monitor receipt's zero verified count used the wrong manifest base. Exact outcomes and limitations: `results/ops/20261025T160614Z/grid_campaign_package/GRID_RESULT_REVIEW.md`. Earlier RUNNING bullets below are historical observations.

- Private notebook, version 1: https://www.kaggle.com/code/satyapaladugu/betterscheduler-model-grid-20261025f. Submitted with Kaggle CLI 2.2.4; exact command: `kaggle kernels push -p BetterScheduler/staging/20261025f -t 3600 --accelerator NvidiaTeslaT4`. Submission receipt: `results/ops/20261025T160614Z/grid_campaign_package/submission/push.log`.
- Package: source ZIP SHA-256 `40e59670657722b7f4c67ce90255c55ddf01d6de51c8019484e698a06956a9b3`; notebook embeds the identical archive. Manifest describes 12 detector, 8 CLIP, and 8 Qwen settings; no scheduler training or automatic winner selection. Upstream source: completed `satyapaladugu/betterscheduler-prepare-20261025b`. Limits: 3,000 internal s, 3,600 server s, maximum 2 aggregate GPU-hours.
- Models/design: Ultralytics YOLO-World `yolov8s-worldv2.pt` with 16 labels; four forward settings (640/1024 px × NMS IoU 0.5/0.7) cache raw boxes at confidence 0.05, then replay confidence 0.05/0.15/0.25 for 12 detector rows. CLIP `openai/clip-vit-base-patch32` at revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` compares 8 preprocessing/text/prompt configurations. Qwen3-VL-4B-Instruct at revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` evaluates 8 prompt/token/pixel settings ×24 controls (192 calls maximum). Its 24 controls use 8 groups and exclude all 48 listed c/e groups. YOLO and CLIP use the first 20 hash-sorted LADI image IDs in the prepared cohort; exploratory only.
- Validation before push: 62 tests passed; package metadata/private/T4/upstream/caps checked; four code cells and four embedded source strings compiled; 53 archive manifest files verified; two model-grid modules imported from an isolated extracted archive with inference packages stubbed. Receipts: `results/ops/20261025T160614Z/grid_campaign_package/`.
- Historical UI checkpoint (superseded by terminal result above): at a 702.2-second snapshot, YOLO and CLIP had completed and Qwen was 3/8 configurations. The run subsequently completed all stages; see `GRID_RESULT_REVIEW.md`.
- The historical UI snapshot showed `ValueError: I/O operation on closed file` when Transformers warnings reached the stage logger. It did not stop inference. A local `Tee.write` fix and regression tests are included in later packages; the immutable f package retained the logging issue.
- Terminal retrieval saved status/config/progress/checksum JSON, JSONL outputs, text reviewer responses, logs and executed notebook. All 240 entries in the run checksum manifest verified when resolved under `monitor/retrieved/results/model_grid/20261025f/`; see `monitor/checksum_audit_runroot.json`.
- Account quota at preflight: 16,285.463 seconds remaining (4.524 hours), zero reserved; quota refresh 31 October UTC. Preflight receipt: `results/ops/20261025T1604Z/grid_campaign_preflight/`.


## Latest annotation pilot status — 20261025e

- Private notebook: https://www.kaggle.com/code/satyapaladugu/betterscheduler-annotation-pilot-20261025e (version 1); submitted 25 October 2026 at 19:19:42 IST from frozen `staging/20261025e`. Source archive SHA-256: `36f0d55bbd73f89f3a7b515a3cf8fdd0460df943579b9c1a5d7e73ebf7a0b577`.
- Terminal status: **COMPLETE**, verified by Kaggle CLI at 19:49:58 IST. First observed RUNNING at 19:24:20 IST; exact worker start time is not exposed. `status.json` records 1,321.4 internal seconds and 0.7341 allocated GPU-hours.
- Limits: server timeout 3,600 s, internal cap 3,000 s, maximum 2 aggregate GPU-hours. No cancellation or retry occurred.
- All five experiment stages completed: data/reference preflight 8.08 s; reviewer/vision preflight 86.08 s; 72 annotation-assisted controls 374.23 s; GPU0 producer 18.55 s; parameter/history/review stage 834.44 s. It processed 44 parent images, 36 UE observations, and 72 binary parameter reviews. Provenance records CLIP and YOLO-World on cuda:0 and Qwen3-VL-4B on cuda:1. No scheduler was trained and no codec/channel transmission was tested.
- The engineering control gate passed: 70/72 valid responses (97.22%); consistency balanced accuracy 0.8958; representation-sufficiency balanced accuracy 0.875; omission rejection 22/24 (91.67%). This is annotation-assisted reviewer assessment on constructed controls, not independent visual-recognition accuracy.
- Parameter results remain weak: structured view had 13/36 consistent reviews (36.1%), 4/36 sufficient (11.1%), and 50/126 direct known-fact comparisons correct (39.7%). Compact view had 7/35 consistent (20.0%), 0/35 sufficient, and 4/18 comparisons correct (22.2%). Structured building-present recall was 0/22 positives and road-present recall 0/24 positives. Two control reviews failed schema validation because the reviewer returned an empty reason; one compact parameter review was truncated JSON. Do not claim the representation is reliable.
- Outputs retrieved under `results/cloud_annotation_pilot/20261025e/`: executed notebook, four cell logs/status, all stage logs, controls/truth/exposure, model provenance, feature evidence, parameters, representations, histories, references, control/parameter reviews, and 149 raw reviewer files. All 176 files in `artifact_checksums.json` matched SHA-256; no missing or mismatched manifest-listed files. Review audit, retrieval receipts, and integrity report are in `results/ops/20261025T141900Z/terminal_e/`. The separate local CPU CLIP preflight and 42-test checkpoint are in `results/ops/20261025T140300Z/local_validation/`.

## Current annotation-assisted pilot — 20261025c

- Private notebook: https://www.kaggle.com/code/satyapaladugu/betterscheduler-annotation-pilot-20261025c
- Version1 submitted successfully; API verified RUNNING after submission.
- Launch: `kaggle kernels push -p BetterScheduler/staging/20261025c -t 3600 --accelerator NvidiaTeslaT4`.
- Immutable package: `staging/20261025c/{notebook.ipynb,source.zip,pilot_manifest.json,kernel-metadata.json}`.
- Source HEAD8947b399c6d0c178fe4cf870aa8a4700a960cf35 plus per-file hashes of dirty working sources;41 packaged files verified. No repository reset/commit.
- Input: completed private prepare20261025b output; selected images only, no new image-dataset download.
- Managed Kaggle environment; pinned dependencies installed in active kernel. Actual GPU/model capability remains a runtime gate.
- Two T4 requested;GPU0producer/GPU1reviewer, no DataParallel. Server3600s bounds setup+execution to at most2 allocated GPU-hours;internal work3000s.
- 72 building annotation-assisted controls before producer loading;at most36 UE observations,each compact/structured. No scheduler training or old autochain.
- Expected saved outputs:`/kaggle/working/results/annotation_pilot/20261025c/`, `cell_evidence/`, `dependency_receipt.json`, `artifact_checksums.json`.
- Expected local retrieval:`results/cloud_annotation_pilot/20261025c/`. Not yet retrieved or verified; RUNNING is not a scientific result.
- Status:`kaggle kernels status satyapaladugu/betterscheduler-annotation-pilot-20261025c`.
- Retrieval after termination:`kaggle kernels output satyapaladugu/betterscheduler-annotation-pilot-20261025c -p BetterScheduler/results/cloud_annotation_pilot/20261025c`.
- Never relaunch blindly. Check actual status,source/config and partial artifacts first. No automatic downstream stage or retry.
- Bounded local monitor PID31824 (`staging/20261025c/monitor.ps1`); launch receipt `staging/20261025c/launch_receipt.json`. Polls every60s,maximum65checks; retrieves outputs on terminal status and hashes local files. It cannot notify through the chat and stops if the laptop/process stops. Remote server timeout remains independent. Retrieval success still requires inspection of retrieval.log and actual artifacts.
- Local suite39passed5.42s,44Python source hashes stable;receipt `results/ops/20261025T114950Z/annotation_pilot_validation/`.
- Luna hit usage limit after local tests;main Sol performed package verification/submission. Medium found no additional confirmed launch blocker after explicit server timeout verified. Limitations recorded in JOURNAL.md.
- Terminal result COMPLETE; outputs retrieved and checked. Reviewer produced72/72 valid JSON responses, but gate FAILED: sufficiency balanced accuracy0.6354, consistency balanced accuracy0.6771, omission rejection0/24. Producer not loaded; no extracted-parameter result. Full trace includes per-case output and executed notebook under the retrieved results path. Do not rerun this unchanged design.

Account verified: `satyapaladugu`. Older `hmmmkk` private notebooks are inaccessible
and are not dependencies.

## VLM-pilot stop check and next design

`results/ops/20261025T105121Z/stop_check/` verifies prepare COMPLETE, no confirmed
running BetterScheduler job, and unavailable/unlisted extract/select slugs. No
cancellation was needed or sent; the automatic local launcher remains stopped. No
replacement notebook was pushed. The proposed GPU0 producer/GPU1 Qwen4B reviewer
pilot, benchmark sources, parameter methods, and go/no-go controls are in
`docs/VLM_PILOT_PROPOSAL.md`. Await methodology review requested by the user before
extractor implementation/launch. Preserve existing data outputs.

## Active campaign — 25 October2026,14:42 IST

### Latest verified prepare receipt — 25 October2026

The prepared notebook `satyapaladugu/betterscheduler-prepare-20261025b` now
reports **COMPLETE**. Its `status.json` records 7,555 merged rows and
14,392,072,243 downloaded bytes; of the 20,000,000,000-byte campaign cap,
5,607,927,757 bytes remain by subtraction. Quota at the latest check was 29.12
GPU-hours. This prepare stage is configured CPU-only.

Small metadata outputs were retrieved by exact output file path through
`ApiDownloadKernelOutputRequest.file_path`, avoiding the earlier paginated
`kernels output` HTTP 429. No image files were downloaded locally. Receipts and
SHA-256 hashes are in
`results/ops/20261025T110200Z/annotation_pilot/`; see
`PREPARED_RUN_REPORT.md` for source/task counts, label mappings, path resolution,
conflicts and evidence limits. The retrieved `records.json` has 4,793 LADI and
2,762 MEDIC rows. Its relative image paths are `data/ladi/images/<sha256>.img`
and `data/medic/images/<sha256>.img`, rooted at the prepared run directory.
The extraction helper can resolve the mounted source by finding its unique
`data/records.json`.

The source availability request counts (4,800 LADI/2,800 MEDIC) differ from the
merged retained row counts. MEDIC is partial because the source call recorded an
HTTP500. The run reports no verified event grouping; three conflicting
affected-people labels remain preserved in the conflict artifact. Treat these
as the latest verified input facts. Earlier checkpoint text below records what
was known at that time and is retained as history.

No CLIP, YOLO-World, or Qwen model download or inference job has been launched
for the annotation-assisted pilot. The forthcoming first pilot is capped at
3,600 server seconds (two GPUs, at most two aggregate GPU-hours), with CLIP plus
YOLO-World on GPU0 and Qwen4B on GPU1. Its frozen package, not this prepare run,
must pass model-placement and runtime preflight before launch.

**Latest verified update (parameter-representation replan):** API reports prepare
COMPLETE. Partial local receipts in `results/cloud_prepare_receipts/20261025b/`
show all four notebook cells complete,an exposure audit,and the LADI20-record
annotation preflight. Output retrieval then hit HTTP429; actual full parent counts,
source availability and download bytes await remaining receipts. No new extraction
or selection stage has been launched. Automatic downstream watcher remains stopped.
The active plan/protocol now require direct parameter/scene-representation tests
before scheduler sweeps; do not resume the old uploaded extraction chain.

**Downstream launch hold:** the local watcher PID41916 was verified by command
and package path, then stopped during the user-requested extraction replan.
See operations/HOLD.json. Remote preparation remains unchanged; do not resume
the old automatic chain. The revised producer/parameter-validation plan is in
PLAN.md. YOLO-World was not part of this uploaded snapshot; the optional local
Grounding DINO adapter was also absent from cloud execution.

- Package:`staging/20261025b`; source archive SHA/file manifest in campaign.json.
- Prepare version1 successfully pushed; API verified RUNNING:
  https://www.kaggle.com/code/satyapaladugu/betterscheduler-prepare-20261025b
- Target:24seed17 sampled official LADI Arrow shards (~4,800parents),plus5,200
  MEDIC images from randomized100-row pages across the official training split.
  Downloads limited20GB. Actual acquired counts remain pending.
- Exposure ledger:2,434 known historical/pilot entries,zero missing local images.
  Content/source identities and near-pixel matches propagate through components.
  This covers declared local development exposure,not unknown external pretraining.
-24focused tests passed; isolated source.zip imports and every cloud cell compiled.
  Dual-GPU execution has not yet occurred; runtime parity/ownership remain gates.
- Local bounded watcher PID41916;receipt:`staging/20261025b/operations/launch.json`.
  Poll120s,24hour total cap. Keep host awake. The remote current stage survives
  local exit; automatic downstream submission requires this local watcher.
- Next slugs:`satyapaladugu/betterscheduler-extract-20261025b` and
  `satyapaladugu/betterscheduler-select-20261025b`. Not launched at this checkpoint.
  Upstream dependencies are unique campaign notebooks,never historical mutable runs.
- Extract launch timeout14,400s caps twoGPU allocation at8aggregatehours;
  internal work cap7.5aggregatehours leaves setup margin. Actual DP and bothdevice
  ownership/parity required. Selection uses CPU,21,600s cap. No automatic retry.
- Watcher retrieves saved status/exposure/parity receipts before downstream launch.
  Output roots:`/kaggle/working/results/{prepare,extract,select}/20261025b`.
  Local retrieval:`staging/20261025b/operations/{stage}`. Persistence/retrieval not
  yet verified because prepare remains running. Cell evidence and partial feature
  shards save incrementally. Complete source/metrics remain in saved Kaggle outputs.

Resume watcher only after inspecting watch_status.json,watch_failure.json and actual
Kaggle statuses; never blindly relaunch a completed/downstream stage. It is an
operational stage chain,not a scientific validity certificate. The selection
notebook records freeze and locked-final sections separately; a distinct final
notebook remains a packaging improvement rather than an executed claim.

## Current launch gates

The laptop run and corrected cache replay are complete (see PILOT_REPORT.md).
Cloud modules exist but have not been remotely tested. A verified historical
exposure/near-duplicate exclusion ledger and complete paired comparator support
are required before claiming fresh final selection. The API still returned
RUNNING for the previous Mamba notebook in five bounded checks after the user
reported stopping it manually. No session ID was guessed and no duplicate GPU
campaign was launched. Resolve available allocation before the GPU stage.

| Notebook | Purpose | Inputs | Status / outputs |
|---|---|---|---|
| better-scheduler-prepare | CPU source/schema/group audit | Versioned source + official accessible image labels | Planned; new immutable data manifest |
| better-scheduler-extract | Verified dual-GPU shared vision extraction | Prepared manifest/images; pinned model source | Planned; feature shards, producer provenance, ownership/parity evidence |
| better-scheduler-select | Grouped field/method/scorer selection | Validated prediction cache + isolated labels | Planned; full grid, fold artifacts, provisional selected spec |
| better-scheduler-final | Locked eligible fresh-group replay | Frozen spec/scorer and final group manifest | Planned; paired grouped results and bounded claims |

Expected runtime root: `/kaggle/working/BetterScheduler/results/<stage>/<run-id>/`.
Each stage must save executed notebook/cell logs, source/environment/configuration,
feature/model checkpoints, metrics/figures and hashes. Published/saved output and
local retrieval paths must be verified and added here; a working path alone is
not evidence of durable recovery. No final results are implied by this map.


## Review-only follow-up 20261026i — prelaunch record

- Kernel: [`betterscheduler-review-20261026i`](https://www.kaggle.com/code/satyapaladugu/betterscheduler-review-20261026i), private, NvidiaTeslaT4, upstream `satyapaladugu/betterscheduler-semantic-validation-20261026h3`.
- Purpose: Qwen3.5-9B-only reviewer follow-up on the exact same 30 selected cases (24 controls and 6 actual review cases); no feature extraction or 4B rerun. Thinking disabled; max 192 new tokens; image cap 262,144 pixels.
- Runtime limits: Kaggle server 1,800 seconds; internal 1,650 seconds; maximum allocated GPU time 1 hour. Fresh prelaunch quota check at 2026-10-26 00:00:50 UTC: 27.704 GPU-hours remaining; all nine prior BetterScheduler campaign kernels terminal; h3 upstream COMPLETE.
- Validation: End2End `.venv` Python 3.12.14 full suite 83 passed before final manifest synchronization; final targeted builder tests 3 passed. Four notebook code cells compile. Both `review_manifest.json` and `pilot_manifest.json` declare matching limits, case count, thinking setting and upstream. Source archive has 69 manifest entries with no archive hash mismatches; three package namespace `__init__.py` files are intentional isolated stubs. Archive SHA-256: `6cd5e8abc3d99113eff24b53a7b601bf0bc08db6687eae2e5caf08828b01a566`.
- Staging: `staging/20261026i_final/`. Operations receipts: `results/ops/20261025T2356Z/review_only_package/`.
- Expected outputs: executed notebook, cell logs, run status/config, raw reviewer JSONL, metrics and checksum manifests under the Kaggle run output. Retrieval is pending; expected is not verified until fetched.
- Current status at this entry: packaged and validated; not yet submitted.

### 20261026i submission receipt (00:02 UTC)

- Push returned version 1 and `RUNNING` for Kaggle URL slug `satyapaladugu/betterscheduler-qwen9b-review-20261026i`: https://www.kaggle.com/code/satyapaladugu/betterscheduler-qwen9b-review-20261026i
- Kaggle's canonical URL slug differs from `kernel-metadata.json` id (`betterscheduler-review-20261026i`); the metadata-id status query was denied, while the exact returned URL slug reports RUNNING. Treat the returned URL as canonical for monitoring and outputs.
- Push command/receipt: `results/ops/20261025T2356Z/review_only_package/push_20261026i.json`; final package verification: `package_validation_final_corrected.json`; actual status sample: `initial_status_20261026i.json`. Terminal output retrieval is pending.

## Scene Suite data-only cohort — 20261026b (prelaunch)

- Private Kaggle notebook: `satyapaladugu/scene-suite-data-20261026b` (title `Scene Suite data 20261026b`). Expected URL slug: `https://www.kaggle.com/code/satyapaladugu/scene-suite-data-20261026b`; after push, use the canonical URL returned by Kaggle if its title-derived slug differs.
- Frozen package: `SemanticSchedulerBuild/staging/scene_suite/data_20261026b/`; source ZIP SHA-256 `667947710aa12c1b8dada85b20b19be133e1d9da38206f7bb072fd62d18c593d`. All 191 source-manifest entries and all 26 upstream manifest entries were hash-verified. The only archive member outside `source_manifest.json` is `upstream/upstream_manifest.json`. Source snapshot commit `8947b399c6d0c178fe4cf870aa8a4700a960cf35`; dirty state is captured in the manifest.
- Purpose: CPU-only official LADI-v2 test-source preparation. Target exactly 80 new parent images in each of three native building strata (240 total). Runtime receipt marks `partial` and raises if target counts are not exactly `[80,80,80]`. Downloaded image bytes are capped at 2,000,000,000. Dataset revision and row indices are recorded at run time; no event-held-out claim is made.
- The data stage records source native annotations in `records.json` as reference truth. It performs no model inference, teacher review, scheduler training, or policy evaluation; downstream runtime documents are created from prediction evidence and exclude evaluator labels.
- Resource cap: 1,800 server seconds, CPU only (`enable_gpu=false`; no accelerator flag). GPU-time allocation for this stage is zero. Current prelaunch Kaggle GPU quota: 27.51 hours remaining, refresh `2026-10-31T00:00:00Z`; quota is rechecked at submission.
- Expected remote output root: `/kaggle/working/results/scene_suite/20261026b/`, including prepared images, `data/records.json`, `progress.json`, `data_receipt.json`, stage stdout/stderr, environment/source provenance, artifact checksums, and executed notebook/cell evidence as exposed by Kaggle. These are expected until retrieved and verified.
- Expected local retrieval root: `SemanticSchedulerBuild/results/scene_suite/ops/20261026b/data/`. The sole planned submission command is `kaggle kernels push -p staging/scene_suite/data_20261026b -t 1800` from the Build checkout. Current status: package preflight passed; not yet submitted. Do not rerun automatically.

### 20261026b submission receipt

- Version 1 was accepted once. Returned canonical URL: https://www.kaggle.com/code/satyapaladugu/scene-suite-data-20261026b. Initial status: RUNNING. The returned URL slug is authoritative for subsequent status and output retrieval.
- Push receipt and exact command: `SemanticSchedulerBuild/results/scene_suite/ops/20261026b/push_output.log` and `push_command.txt`. No retry or other remote job was launched.
- Terminal status, executed notebook/cell logs, `data_receipt.json`, outputs/checksums, and local retrieval verification are pending.

### 20261026b terminal result — ERROR before notebook execution

- Kaggle version 1 reached platform status **ERROR** at `2026-10-26T03:26:51Z`. Papermill failed before the first cell with `ValueError: No kernel name found in notebook and no override provided`; the submitted notebook metadata is `{}` and has no `kernelspec.name`.
- No data preparation cell ran: zero LADI rows/images were downloaded, no model or scheduler code ran, and GPU use was zero. This is a packaging/runtime bootstrap failure, not a cohort result.
- One terminal retrieval succeeded but returned only `scene-suite-data-20261026b.log` (3,205 bytes; SHA-256 `41cf8122eff4aa1ba184ebc444fa8fcae19104bae02cf8957286352cea9819ad`). The subsequent remote file listing was empty. No executed notebook, cell logs, `data_receipt.json`, checksums, or data files persisted/retrieved.
- Local receipt: `SemanticSchedulerBuild/results/scene_suite/ops/20261026b/terminal_receipt.json`; platform log and retrieval records are in the same `ops/20261026b/` folder. No automatic retry was launched. A new package must include a valid notebook kernelspec and pass preflight before any resubmission.


## Scene Suite data-only cohort — 20261026c (diagnosed CPU retry; prelaunch)

- This is one authorized retry of data-only run `20261026b`, which failed before cell execution because the uploaded notebook had no `kernelspec.name`. The corrected notebook now declares `kernelspec.name=python3`, `language=python`, and language metadata; `nbformat.validate` and compilation of all three code cells pass. No data/model/scheduler code change is part of the correction.
- Private Kaggle notebook: `satyapaladugu/scene-suite-data-20261026c` (title `Scene Suite data 20261026c`). Expected URL: `https://www.kaggle.com/code/satyapaladugu/scene-suite-data-20261026c`; use the returned canonical URL if rewritten.
- Frozen package: `SemanticSchedulerBuild/staging/scene_suite/data_20261026c/`; source ZIP SHA-256 `ac8156e4ec9e2d098ec4f0d105a169513f24f7a73c6b3c31f9506cdfe102920e`. All 191 source manifest entries and 26 frozen-upstream entries verify. The only ZIP member outside the source file manifest is `upstream/upstream_manifest.json`. Commit snapshot `8947b399c6d0c178fe4cf870aa8a4700a960cf35` with dirty-state provenance recorded.
- Stage: official `MITLL/LADI-v2-dataset` test-split cohort preparation only, exact target 80 independent images per native stratum (240 total); partial cohort is an explicit failure. Source image download cap is 2,000,000,000 bytes. `records.json` retains source labels for later evaluator joins; runtime scene documents are separately produced from prediction evidence. No models, teacher calls, training, or scheduler scoring in this stage.
- Resource cap: 1,800 server seconds, private, internet-enabled, CPU-only (`enable_gpu=false`, no accelerator flag). Fresh quota preflight: 27.51 GPU-hours remaining, refresh `2026-10-31T00:00:00Z`; the prior b run and latest review kernel were terminal. Target c was absent before submission.
- Expected remote output root: `/kaggle/working/results/scene_suite/20261026c/`; expected local retrieval: `SemanticSchedulerBuild/results/scene_suite/ops/20261026c/data/`. Expected artifacts include actual prepared images/records, strict target counts and source revision receipt, environment/provenance, cell/stage logs, checksums, and executed notebook; verify what Kaggle actually persists.
- Sole planned push: `kaggle kernels push -p staging/scene_suite/data_20261026c -t 1800` from SemanticSchedulerBuild. Prelaunch package check passed; not yet submitted. No automatic retry beyond this diagnosed attempt.

### 20261026c submission receipt

- Kaggle accepted version 1 once. Returned canonical URL: https://www.kaggle.com/code/satyapaladugu/scene-suite-data-20261026c. Initial status: RUNNING. Use this returned URL slug for monitoring/output retrieval.
- Push command and response are saved under `SemanticSchedulerBuild/results/scene_suite/ops/20261026c/`. No automatic retry was started.
- Terminal experiment status and actual persisted/retrieved outputs are pending.

### 20261026c terminal result and retrieval

- Kaggle version 1 reached terminal `ERROR`; the notebook bootstrap succeeded and the CPU data stage executed, but the strict cohort gate correctly raised `RuntimeError: partial cohort: target is 80 independent parents per stratum; preserve observed rows`. Retrieved `status.json` records `failed`, stage `data`, and 125.07 s. Kaggle's platform log reports the failed cell at In[2] and runtime about 158 s. This is a cohort insufficiency result, not a source-download/API failure.
- Verified receipt/progress: counts `[80,80,42]` against `[80,80,80]`, source revision `5f2dbfe8c466d32edafd1bab847ec5252309acdb`, 68,289,033 bytes fetched, and zero row download errors. Retrieved `records.json` contains 202 unique records (strata 80/80/42; roles train 121, validation 40, pilot_eval 41). The 202 local image files all match record IDs and their SHA-256 filename hashes; 62,419,415 bytes persisted locally.
- Read-only source API check at `results/scene_suite/ops/20261026c/source_rows_api_check.json` confirms the same dataset revision and only 1,049 rows in the test split. The remaining rows 1000–1048 contain just two valid damaged-building labels, so scanning the tail can raise the third count to at most 44; the 80/80/80 test-split target is unattainable. No GPU, teacher/model inference, training, or scheduler scoring occurred. Do not retry this cohort unchanged.
- Initial `kernels output` retrieval was still in progress when first inspected; it completed with exit 0. Final retrieved root: `SemanticSchedulerBuild/results/scene_suite/ops/20261026c/data/`; the exact-pattern retrieval for receipt, progress, records, stage logs, status, and platform log is under `.../ops/20261026c/metadata_retry/`. File hashes and record/image reconciliation were checked. This was not a timeout or API-pagination loss.
- The platform log says nbconvert wrote `__notebook__.ipynb`, but no executed notebook was exposed by the output retrieval/file-pattern check. Stage stdout/stderr and platform logs are preserved; per-cell JSONL logs/checksum inventory were not retrieved. Notebook-output capture remains incomplete for this failed run.

## Scene Suite data-only source-mix cohort — 20261026e (prelaunch)

- User-approved correction after 20261026c found the LADI-v2 test split cannot supply 80 damaged-building parents. This package uses the official `test` source first, capped at 40 parents per each of the three strata, then fills each cumulative target to 80 from official `train`. The frozen vision producer may have trained on the added source-train images; the receipt records this exposure and actual source counts per stratum. Do not claim vision-model holdout for these rows.
- Target: 240 independent source-image parents, 80 per native binary stratum. Data records retain source labels, source split/row/revision, and group identity for evaluator joins. The feature writer emits only `{image_id, prediction evidence}`; runtime scene validation rejects labels/split/evaluator fields. Grouped 60/20/20 allocation was checked on 240 singleton groups: train 144, validation 48, pilot evaluation 48 (48/16/16 per stratum). Dataset event IDs are unavailable, so no event-held-out claim.
- Frozen package: `SemanticSchedulerBuild/staging/scene_suite/data_20261026e/`; source ZIP SHA-256 `3b745c3109c522a2593a289294f3f2d0debad785a2da629a8df5a86ce6d78cef`. All 192 manifest entries match; the only ZIP entry outside the source manifest is `upstream/upstream_manifest.json`. The notebook embeds the byte-identical archive and manifest, validates under nbformat, and all code cells compile. The staged exclusion list has 211 unique IDs and matches the current union from BetterScheduler config/local JSON plus Build `results/decoder_context` JSON (zero missing/stale IDs). Snapshot commit: `8947b399c6d0c178fe4cf870aa8a4700a960cf35`; dirty snapshot is captured in the manifest.
- Private Kaggle notebook: `satyapaladugu/scene-suite-data-20261026e`; GPU disabled, internet enabled, no attached datasets. CPU-only limit 1,800 server seconds; source download cap remains 2,000,000,000 bytes. Fresh quota at preflight: 27.51 GPU-hours remaining, refresh `2026-10-31T00:00:00Z`; GPU use for this run is zero. Latest relevant kernels c=ERROR, i/h3/h2a/h=COMPLETE; no active relevant remote run.
- Expected output root: `/kaggle/working/results/scene_suite/20261026e/`. The stage writes source manifest/environment, receipt/records/images, progress, stage stdout/stderr, terminal status, and artifact checksums in `finally`; it creates one recovery archive at `/kaggle/working/results/scene_suite/20261026e-artifacts.zip` before raising on a partial cohort. Expected local retrieval: `SemanticSchedulerBuild/results/scene_suite/ops/20261026e/`. Prefer the recovery ZIP, then retrieve exact status/logs/notebook outputs. These files remain expected until verified.
- Planned single push: `kaggle kernels push -p staging/scene_suite/data_20261026e -t 1800` from SemanticSchedulerBuild. No accelerator flag. Do not retry automatically.

### 20261026e submission receipt

- Kaggle accepted version 1 once. Returned canonical URL: https://www.kaggle.com/code/satyapaladugu/scene-suite-data-20261026e. Initial status: RUNNING. No accelerator flag was passed.
- Exact command, output, and exit code are recorded at `SemanticSchedulerBuild/results/scene_suite/ops/20261026e/push_command.txt`, `push_output.log`, and `push_exit.txt`. The bounded monitor completed; see the terminal result and retrieval section below.

### 20261026e terminal result and retrieval

- Kaggle version 1 reached terminal `ERROR`. Notebook setup succeeded under managed Python 3.13.15 / PyTorch 2.11.0+cpu (`cuda=false`, zero GPUs). The data stage failed after 0.49 s on the first LADI rows API request with `HTTPError: HTTP Error 502: Bad Gateway`; no row receipt, source images, or data records were created. This is an upstream HTTP failure before cohort selection, not a package/bootstrap or cohort-gate result. No same-package retry was launched; the separately authorized reasoned follow-up package and preflight are recorded below. GPU time was zero.
- The preferred recovery artifact was retrieved first: `SemanticSchedulerBuild/results/scene_suite/ops/20261026e/recovery/results/scene_suite/20261026e-artifacts.zip` (20,504 bytes; SHA-256 `2cbe3b426472c15be422b7df1a498bade553ae48514da67dada2bdda544a193b`). Its six files are `environment.json`, `source_manifest.json`, `status.json`, `stage_stdout.log`, `stage_stderr.log`, and `artifact_checksums.json`; all five listed file checksums validate and the recovered source manifest matches the staged manifest.
- Exact metadata retrieval completed with exit 0 under `.../ops/20261026e/metadata/`, including the Kaggle platform log. No executed `.ipynb` was exposed by output retrieval. No records/images/receipt were created, so no source counts or exposure-by-stratum are available for this run. `recovery_verification.json` records the artifact checks.

## Scene Suite data-only source-mix recovery — 20261026f (preflight; not submitted)

- This immutable follow-up contains the approved bounded GET retry for the `20261026e` HTTP 502: at most three attempts for idempotent requests, retrying HTTP 429/500/502/503/504 and URL/timeout failures with 1 s then 3 s waits; permanent non-listed HTTP errors fail immediately. The 2,000,000,000-byte cap, source-split protocol, 80-per-stratum target, train-exposure disclosure, and 40-per-stratum test phase are unchanged for the fixed 80 target.
- Frozen package: `SemanticSchedulerBuild/staging/scene_suite/data_20261026f/`; archive SHA-256 `3bce5614e20769d70b36cbde380e18b73a93dcdb115475bc9d3879084c060634`. All 192 manifest files and embedded archive hashes verify; notebook validates and code cells compile. Exclusions remain 211 IDs matching BetterScheduler config/results and Build decoder-context records. Labels/provenance stay in `records.json`; feature rows contain only image ID and prediction evidence; runtime scene schema rejects label/split fields. Evidence: `SemanticSchedulerBuild/results/scene_suite/ops/20261026f/preflight/package_audit.json`.
- The first read-only preflight at 2026-10-26 03:56 UTC returned dataset metadata HTTP 200 and test rows HTTP 200, but train rows HTTP 500. One delayed read-only train-row recheck at 03:58 UTC also returned HTTP 500. Because the required test/train source endpoints were not both available, the authorized CPU launch was withheld; no Kaggle version was pushed and GPU use was zero. Exact checks are in `.../ops/20261026f/preflight/hf_endpoint_preflight.json`.
- After package freeze, the current working tree changed only `INFRA/learning/scene_prepare.py` in the compared manifest set: the test-phase expression became `min((n+1)//2,40)` rather than frozen `min(n,40)`. For target 80 both evaluate to 40. The frozen approved package was not modified or launched.

## Scene Suite frozen feature extraction — 20261026k (prelaunch)

- Purpose: run the single bounded GPU feature-extraction stage for the locked 240-image Scene Suite cohort. Encoder/decoder work and all training remain out of scope.
- Private input dataset: `satyapaladugu/scene-suite-cohort-20261026g`, private, version 1, status `ready`. Remote inventory contains 244 files; every staged file and size matches, plus the upload manifest. Retrieved `records.json` SHA-256 matches local `d47fdd35f7c2747bc9804696e1f9d7882433ed079812863bc60bd8966c8685fb`. No dataset reupload planned.
- Dataset source exposure: cohort includes official training-source images to meet the positive-class target. This is disclosed; frozen vision-producer training exposure is not treated as scheduler-held-out evidence.
- Frozen package: `SemanticSchedulerBuild/staging/scene_suite/extract_20261026k/`, archive SHA-256 `91dcd9ccfb3fe150066d5f698f2535b79f9e780d6f07a500645beab663775bb5`. All 194 manifest files match the archive; all Python files and notebook code compile. Embedded SIGALRM cutoff is 2,640 seconds; Kaggle server cap is 2,700 seconds. GPU is enabled and requested accelerator is `NvidiaTeslaT4`. Attached input is the private cohort above.
- Kaggle preflight 2026-10-26: CLI 2.2.4 authenticated; quota reports 27.51 GPU-hours remaining (refresh 2026-10-31T00:00:00). Planned push is one version with `-t 2700 --accelerator NvidiaTeslaT4`; conservative reservation is 1.5 aggregate GPU-hours for two allocated devices over the full cap. Actual device allocation must be taken from runtime evidence, not inferred from the request.
- Expected output root: `/kaggle/working/results/scene_suite/20261026k`; retrieve with `kaggle kernels output satyapaladugu/scene-suite-extract-20261026k -p results/scene_suite/ops/20261026k/retrieved/`. Verify extraction receipt, feature count, finite values, unique source IDs, checksums, logs, and actual runtime GPU metadata before any downstream training decision.
- Prelaunch artifact: `SemanticSchedulerBuild/results/scene_suite/ops/20261026k/package_audit.json`. Submission is not yet recorded; fill the actual returned URL/version, status, observed hardware, timings, and verified retrieval after the single push. Do not auto-retry.

- Kaggle accepted version 1 at 2026-10-26T06:49:28.9301951Z: https://www.kaggle.com/code/satyapaladugu/scene-suite-extract-20261026k. Push exit 0; initial status RUNNING. Requested NvidiaTeslaT4, 2,700-second timeout; actual runtime devices pending. Receipt: SemanticSchedulerBuild/results/scene_suite/ops/20261026k/submission_receipt.json.

### 20261026k terminal result and verified extraction

- Kaggle version 1 reached platform `COMPLETE`; experiment status is `complete`. Stage wall time was 141.524 s; feature extraction took 96.947 s. Managed runtime was Python 3.13.15, PyTorch 2.11.0+cu128, CUDA true, and two Tesla T4 devices. The producer used `cuda:0`; checkpoint `yolov8s-worldv2.pt`, weights SHA-256 `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792`, vocabulary/settings hashes are recorded in `producer_receipt.json`.
- Verified artifact bundle: `SemanticSchedulerBuild/results/scene_suite/ops/20261026k/verified_bundle/`. Its 254 declared artifact checksums pass; all 240 source records join one-to-one with feature evidence. The causal corpus check created 48-message cross-repeat traces for train and validation, each with 8 ready messages and 99 features; schema and producer references pass. The pilot evaluation was not opened. Detailed validation is in `results/scene_suite/ops/20261026k/`.
- Kaggle quota after completion: 27.46 GPU-hours remaining versus 27.51 before. The rounded quota delta is 0.05 h; wall-clock reservation estimate from 192 s and two T4s is 0.107 aggregate GPU-hours. Treat the former as rounded account reporting and the latter as a conservative runtime estimate, not exact billed usage.

## Scene Suite bounded CPU learner pilot — 20261026l (prelaunch)

- Purpose: one CPU-only feasibility run for annotation-supervised learner methods on the verified k extraction. Qwen-dependent arms remain blocked without teacher labels. No encoder/decoder integration or GPU is included.
- Frozen package: `SemanticSchedulerBuild/staging/scene_suite/train_20261026l/`, archive SHA-256 `e2fa30c2ceea02b6d6b5f2ccc4ce5ca1b6b3ec14294eb83b615ede0d96224aed`. All 194 manifest files match and Python/notebook code compiles. Kernel metadata disables GPU, attaches upstream kernel `satyapaladugu/scene-suite-extract-20261026k`, and requests 10,800 server seconds; worker deadline is 10,740 seconds. Settings are one seed, up to 50 epochs/20,000 RL decisions, 32 traces; expected scope is eight annotation learners plus seven references.
- Preflight: current quota 27.46 GPU-hours; upstream extraction kernel is COMPLETE and its published output bundle is verified. Kaggle CLI 2.2.4. No existing `scene-suite-train-20261026l` kernel was found. Planned push omits accelerator: `kaggle kernels push -p <absolute train_20261026l path> -t 10800`.
- Expected outputs: `/kaggle/working/results/scene_suite/train_20261026l`; retrieve under `results/scene_suite/ops/20261026l/retrieved/`. Acceptance requires checking the run's own status, preserved partial artifacts, learner statuses, logs, and verified artifact hashes; one training seed is feasibility evidence only.

- Kaggle accepted version 1 at 2026-10-26T07:00:05.2742575Z: https://www.kaggle.com/code/satyapaladugu/scene-suite-train-20261026l. Push exit 0; initial status RUNNING; no accelerator was requested. Receipt: SemanticSchedulerBuild/results/scene_suite/ops/20261026k/20261026l/submission_receipt.json.
- Full `kernels output` retrieval completed at `results/scene_suite/ops/20261026k/retrieved/`. CLI exit was 1 after download completion because PowerShell's Windows charmap could not print a Unicode checkmark; the downloaded ZIP is 380,462,301 bytes (SHA-256 `7b61d2b361309ee0beeba72b6fdf49eaf8d01610a864ef6f2b733487241bb561`), has 255 members, and all 254 declared artifact checksums validate with zero missing/mismatched members. No retry was needed.

### 20261026l terminal result and local audit

- Kaggle version 1 reached platform `COMPLETE`; run `status.json` reports stage `train`, status `complete`, 5,310.603 s. Environment: managed Python 3.13.15, PyTorch 2.11.0+cpu, CUDA false, zero GPUs. The notebook's `open_pilot=false`; no pilot-evaluation data was opened.
- Retrieved archive: `SemanticSchedulerBuild/results/scene_suite/ops/retrieved_train_20261026l/results/scene_suite/20261026l-artifacts.zip`. Local audit verified all 80 declared artifact checksums and exact frozen source hashes. Run outputs include comparison CSV, configuration, training/validation feedback requests, metrics, logs, and checkpoints in the ZIP.
- Eight annotation-supervised learners completed: linear choice, MLP choice, RankNet, LambdaMART, priority classifier, masked PPO, masked Double DQN, and TD-PER. The three Qwen-dependent arms (`rl_vlm_f`, `vlm_replay`, `visionprefer`) are blocked because teacher labels were not available. Seven reference policies completed. One seed and 32 validation traces make this a feasibility pilot only.
- Validation replay (32 traces, validation only) mean integrated coverage J: damage-probability 0.701215; damage-header repeat guard 0.713482; LambdaMART 0.718637; PPO 0.715551; Double DQN 0.646381; TD-PER 0.663264; privileged greedy diagnostic 0.729226. The replay uses each policy's own trajectory; it is not a shared-state policy-choice comparison and is not evidence of statistical superiority.
- Local audit command used the `SemanticSchedulerEnd2End/.venv` Python executable: `python scripts/audit_scene_suite_run.py --archive results/scene_suite/ops/retrieved_train_20261026l/results/scene_suite/20261026l-artifacts.zip --output results/scene_suite/audit_20261026l --data results/scene_suite/ops/20261026k/verified_bundle/data --upstream results/scene_suite/ops/20261026k/retrieved/scene_source/upstream --replay-traces 32`. It ran CPU-only in 192.935 s, exit 0. Audit outputs: `results/scene_suite/audit_20261026l/archive_audit.json`, `validation_replay.json`, `rl_learning_curves.png`, `validation_scores.png`; logs and execution receipt are in `results/scene_suite/ops/audit_20261026l_preflight/`.
- Data diagnostic: 1,428 training snapshots; 753 all-zero-gain decisions and 345 singleton decisions (these categories can overlap, so do not sum them); 603 nontrivial decisions; 11,178 zero-gain candidate rows and 1,858 positive-gain rows. Twenty of 99 feature fields were never available; 34 were constant or unavailable. These counts motivate further data/information analysis, not a claim that 77% of decisions lacked choice.
- The local audit run has no remaining monitor process. The only audit issue was an initial invocation against an already-created empty output directory; that failed before computation, was preserved in `results/scene_suite/ops/audit_20261026l_preflight/`, then the requested audit ran successfully to a fresh output directory. No code, tests, protocols, or submitted runs were changed during closeout.


## SPI corrected response contract — candidate c

- Source `staging/spi_20261026c`, ZIP SHA-256 `1c827673ec6a6ce8f47939d9f2135b78f80e097b806ecf83f308a5cc5b1b1ef2`; preflight and11targeted tests passed. Same private180image inputv1; no reupload.
- Luna reached a usage limit after saving preflight; lead completed the already-authorized single submission. Receipt: `results/ops/spi_20261026c_prelaunch/submission_receipt.json`.
- Kernel: https://www.kaggle.com/code/satyapaladugu/betterscheduler-spi-20261026c . Server2700s; internal2400s; <=1.5aggregateGPUhours. Actual startup/status pending verification. No further automatic retry.


## 27 October 2026 — SPI candidate c verified closeout and parameter finalization

- Kernel `satyapaladugu/betterscheduler-spi-20261026c` version1 is ERROR;
  recovered experiment status is incomplete. Offline labelling completed all180
  responses in1034.366seconds. Labels:82Low,0Medium,73High,25Critical.
  Train support49/0/46/15; validation33/0/27/10. All180 are available pseudo-labels.
- CPU fitting stopped before optimization at the explicit all-class-support
  guard. No exported student model is present. This is a target-support failure,
  not the earlier response-parser failure. Teacher label correctness and the
  explanation for absent Medium remain unresolved.
- Recovery archive SHA-256:
  `6c621553fbd9d28a1a4e5ad49dada368eb8ffe71e6fd20b85fd59bebffd7fdee`;
 144970bytes. All 189 declared members match their hashes; no missing
  or mismatched members. Verified receipt:
  `results/ops/spi_20261026c_postrun/verification.json`.
- Selected logs/runtime/status and recovery archive retrieved to
  `results/ops/spi_20261026c_postrun/retrieved/`. CLI returned1 after downloading
  due Windows charmap printing; local integrity verification succeeded. No full
  dataset, weights download, duplicate retrieval or new Kaggle submission.
- Canonical PLAN.md now fixes SPI/relevance/novelty, capture time/allowed delay,
  and four YOLO measurements: instances, categories, largest-box fraction and
  union-box coverage. Scene aggregation/wire extension are planned, not integrated.
  Existing core header stays28bytes; no combined byte claim. Prior16-claim mask
  proposal is superseded; category/box details stay local audit evidence initially.
- Next SPI action: audit rubric evidence/cohort support versus teacher category
  collapse; do not fabricate Medium labels, weaken guards or repeat the same run.
  No additional GPU job is launched by this planning update. No production code
  or tests changed; the checks here are artifact integrity/counts and document
  alignment, not classifier accuracy or detector validation.


### Candidate c downstream local repair — 27 October2026

No additional Kaggle run. Existing labels/features fed explicit Low/High/Critical
CPU head: `results/spi_three_class_20261027/models/spi.json`. Teacher agreement
62.86% on70development validation images; Critical2/10. Model works technically
but urgency reliability remains unestablished. Tests13passed; real-image CLI
passed. Receipt `results/validation/spi_three_class_20261027/ops_summary.json`.
Medium remains unsupported; original labels and failed four-class artifacts
preserved. Detailed diagnosis `docs/SPI_MEDIUM_DIAGNOSIS.md`.

## 27 October 2026 — final parameter campaign, prelaunch

- Target private notebook `satyapaladugu/betterscheduler-spi-20261027finalb`,
  input dataset `satyapaladugu/betterscheduler-spiv2-inputs-20261027b` (READY,
  300 development-exposed images). New source snapshot SHA-256
  `89ae47ceaae1e9cdf56f31d4bd09e6219c9ce0f83c48f6b6ce7d0a525edc136a`.
- Purpose: 12 authored text controls; Qwen3.5-9B VLM v2 pseudo-labels on GPU1;
  YOLO-World scene measurements on GPU0 after the text gate; grouped four-class
  CPU comparison if support gate passes; optional model-assisted review controls.
- Bounds: exactly two T4, 3,600 server seconds / <=2 aggregate GPU-hours;
  no paid API. Cached CLIP image features; no native annotations given to the
  label teacher. New-only groups are sealed before labels but remain exposed
  development data. No inference from dataset name to urgency.
- Expected output `/kaggle/working/results/spi/20261027finalb/`, cell evidence
  `/kaggle/working/cell_evidence/`, checksums and recovery ZIP; local retrieval
  `results/ops/spi_final_20261027b/retrieved/`. These are expected until verified.
  Notebook records actual interpreter, versions, device ownership and logs.
- Stage source `staging/spi_final_20261027b`; all98 archived source files/hash
  entries checked, code cells compiled, input dataset READY. Notebook has not
  yet been submitted. Source is a dirty working snapshot, not commit alone.
- One attempted dataset create was rejected before transfer due subtitle length;
  corrected create produced a private dataset. The earlier candidate sampler
  erroneously excluded minor-or-greater samples; its package is preserved as a
  diagnostic and was not uploaded. Corrected data contains14 such candidate
  rows,43 affected and63 other; labels are not generated from these strata.

### Submission receipt — final parameter campaign

Version1 accepted once at2026-10-27T01:02:44Z; platform status RUNNING at
first check. Actual notebook URL: https://www.kaggle.com/code/satyapaladugu/betterscheduler-spi-20261027finalb .
Receipt: `results/ops/spi_final_20261027b/submission_receipt.json`.
No other Kaggle run was stopped or changed. Terminal results and retrieval remain
pending until separately verified.

### Terminal outcome — final parameter campaign

Platform ERROR; run `status.json`: incomplete, reason text rubric control gate
failed. Text control accuracy9/12: Low3/3, Medium0/3, High3/3, Critical3/3.
All three Medium descriptions were returned as Low. No image labels, student
fit, YOLO worker or VLM parameter audit ran. Runtime: two T4s, transformers5.3,
torch2.11.0+cu128; control stage173.524s, full cell270.791s.

Recovered archive `results/ops/spi_final_20261027b/retrieved/results/spi/20261027finalb/spi_recovery.zip`,
SHA256 `0803679e613612e33ef7f5270fec804bfb6862496e563240b23d521921586eb4`.
All22 declared hashes verified; no mismatches. Cell logs, executed notebook and
runtime retrieved alongside it. CLI returned1 while printing an unsupported
Windows character after files were downloaded; integrity was verified locally.

Diagnosis: the submitted text control appended an instruction to assess a
hypothetical description, but also retained the image-only rubric and a shared
runner prefix asking for a candidate and reference. The inconsistent prompt
limits interpretation of the Medium0/3 result. Local code now accepts a custom
text-control prefix and substitutes the text-control rubric context;29 tests pass.
The corrected VLM prompt was NOT rerun; this fix is unvalidated against Qwen.
No automatic retry or extra GPU allocation occurred. Original run and labels are
preserved. The current three-class fallback remains explicitly weak.

### Corrected VLM preflight and independent YOLO retry — planned 27 October

User explicitly requested continuing the planned parameter experiment after the
failed run. One new private notebook is prepared:
`satyapaladugu/betterscheduler-spi-20261027finalc`, input dataset
`satyapaladugu/betterscheduler-spiv2-inputs-20261027b` (verified READY).
Snapshot `staging/spi_final_20261027c/source.zip`, SHA-256
`5d4a84927dd4854ccd5aee28dc7763922ec4e058f529d1181134a0ecaa029096`;
102 archived files verified, code cells compiled. Existing 300 inputs are
all development-exposed. Two T4s, server max3600s, aggregate GPU max2hours.

Fixes: the text-only control now receives a description-specific prefix and
rule instead of an image-only/candidate-reference instruction; visual preflight
still receives actual image and original image-only rubric. GPU0 YOLO now runs
independently of the VLM text gate, and has up to240 additional seconds to
finish if the teacher gate fails. This does not weaken the Medium0/3 gate.

Expected runtime output `/kaggle/working/results/spi/20261027finalc/` and
`/kaggle/working/cell_evidence/`; retrieve to
`results/ops/spi_final_20261027c/retrieved/`. No output or class support claimed
before verification. Preserve previous failed run, cache and dirty checkout.

### Corrected VLM and independent-scene retry — submission

User authorized continuing the previously planned parameter work. Version1
`satyapaladugu/betterscheduler-spi-20261027finalc` was accepted and reported
RUNNING. Submission receipt and source hash:
`results/ops/spi_final_20261027c/submission_receipt.json`. No other job was
stopped or overwritten. A failed teacher gate no longer prevents GPU0 detector
measurements; it still prevents pseudo-labelling/training. If teacher gating
fails, the detector gets a bounded240 seconds to finish before cleanup.
All new observations remain development-exposed.

### Corrected retry terminal outcome — 27 October

Version1 `satyapaladugu/betterscheduler-spi-20261027finalc` ended ERROR because
corrected text-control gate returned9/12: Low3/3 Medium0/3 High3/3 Critical3/3.
No image SPI labels/training/reviewer calls. Independent GPU0 YOLO completed
300/300 inputs, 202 zero-detection images, 241 total detections (153 person,
38 car,25 truck,1 building,0 road); this is detector output, not ground truth.
Scene stage41.165s. All24 archive members match hashes, ZIP CRC verified;
recovered files at `results/ops/spi_final_20261027c/verified/`, cell logs and
notebook in `retrieved/`. Cohort is previously exposed; no accuracy claim.

## 27 October 2026 — four-class SPI execution preflight and local checks

- Existing private notebook `satyapaladugu/betterscheduler-spi-20261027finalc`
  remains the canonical identity, current version 1, remote status ERROR. Kaggle
  CLI pull confirms the identity and attached input; pulled read-only source and
  metadata are preserved under `experiments/spi_four_class/runs/ops_preflight_20261027/remote_notebook/`.
- Existing private input `satyapaladugu/betterscheduler-spiv2-inputs-20261027b`
  is READY, current version 1. Latest inventory: 300 PNG images plus
  `features.npz`, `input_manifest.json`, `panel.json`, and `selection_audit.json`
  (304 files total). No data or model weights were downloaded locally.
- Authenticated Kaggle CLI 2.2.4 access verified; account GPU quota was 3.24 h
  used / 26.76 h remaining of 30 h. No submission or GPU run was started.
- Focused local checks: 51 passed in 47.17 s; six modules compiled and
  `parameters.spi_protocol` / `parameters.spi_pipeline` imported. These are
  engineering checks, not SPI urgency accuracy evidence. Source hashes changed
  for `parameters/spi_pipeline.py` and `parameters/spi_teachers.py` during the
  test window, so the pass is not pinned to one immutable snapshot and needs a
  clean rerun after those files stabilize.
- Receipts: `experiments/spi_four_class/runs/ops_preflight_20261027/` and
  `experiments/spi_four_class/runs/local_checks_20261027/`.

## 27 October 2026 — four-class SPI evidence v3a prelaunch

- Existing private kernel `satyapaladugu/betterscheduler-spi-20261027finalc`,
  current version 1 / prior status ERROR; submit the frozen package as version 2.
  Existing private input `satyapaladugu/betterscheduler-spiv2-inputs-20261027b`
  is READY, current version 1. No new Kaggle identity or input version.
- Frozen stage: `experiments/spi_four_class/runs/20261027_spi_evidence_v3a/submission_final/`;
  source ZIP SHA-256 `792e8b28c141beb33d5bed984125d22caf92d4684456d271a5b19cd8b8586604`.
  Archive and notebook embedded source match. 105 canonical files match current
  sources; the three `Evaluation` / `Scheduler` namespace initializers match the
  documented isolated stub literal. `project_compat.py` and new SPI modules are
  present. Notebook schema and code cells compile; package identity/input,
  runtime receipt copy, 7,200-second timeout and 4 aggregate GPU-hour cap pass.
- Local focused engineering checks: 61 passed in 48.10 s; `parameters/tests/test_spi_notebook.py`
  is absent and recorded separately. All 11 `parameters/spi*.py` sources and the
  notebook builder compile. No source hash drift during the test window. These
  checks do not validate urgency accuracy.
- Data: 300 development-exposed images, 204 train / 96 validation, no test rows.
  Expected Kaggle output root `/kaggle/working/experiments/spi_four_class/runs/20261027_spi_evidence_v3a/`,
  with `cell_evidence/`, `campaign/`, `retrieval_checksums.json`,
  `run_recovery.zip`, executed notebook and cell logs. Runtime receipt is written
  to cell evidence and copied into the campaign run path. GPU quota before submit:
  26.76 h remaining. No local model weights downloaded.
- Prelaunch record: `experiments/spi_four_class/runs/ops_preflight_20261027/submission_preflight.json`.
  Authorized command: `kaggle kernels push -p experiments/spi_four_class/runs/20261027_spi_evidence_v3a/submission_final -t 7200 --accelerator NvidiaTeslaT4`.

### Submission receipt and live run — SPI evidence v3a

- Kaggle accepted the frozen source ZIP (SHA-256 unchanged) as notebook version 2 on the existing private kernel object, numeric ID `141089219`. The submitted `kernel-metadata.json` title did not slug to its prior ref; Kaggle warned and renamed the ref to `satyapaladugu/betterscheduler-spi-four-class-evidence-pipeline`. The previous ref `satyapaladugu/betterscheduler-spi-20261027finalc` now returns 404. API metadata confirms the numeric ID is unchanged, current version is 2, and the same private input dataset remains attached. This is a slug/title rename of the existing object, not a second notebook object. Restore the original title/slug in a later justified version after this healthy run ends; do not interrupt or replace the current run.
- Version 2 status: RUNNING after QUEUED (status transition at 2026-10-27 15:42:04 UTC). Submitted with 7,200-second server timeout and T4 accelerator; package configuration caps aggregate GPU allocation at 4 hours. No other job was stopped. Kaggle CLI emitted a successful version-2 receipt; status and monitor receipts are in `experiments/spi_four_class/runs/ops_preflight_20261027/`.
- Run ID `20261027_spi_evidence_v3a`; expected runtime output root is `/kaggle/working/experiments/spi_four_class/runs/20261027_spi_evidence_v3a/`. Expected `cell_evidence/`, its `runtime.json`, `campaign/` with the copied runtime receipt/status/report, `retrieval_checksums.json`, and `run_recovery.zip` are not yet verified. Kaggle `kernels logs` returned no cell text at the first RUNNING check. Retrieve outputs after terminal status using the current ref, preserve version 2, and verify recovery ZIP/member checksums before local use.
- All 300 images are development-exposed (204 train / 96 validation; no test rows). Any validation agreement is teacher agreement on exposed development data, not a fresh final result. Gemma remains deferred; no claim of that candidate's support or execution.
- Follow-up after the submitted v2 snapshot: all 136 existing tests in
  `parameters/tests/` passed in 56.33 s with zero source-hash drift; this validates
  the current local checkout only and does not modify the submitted notebook.
  Receipt: `experiments/spi_four_class/runs/broad_parameter_checks_20261027/`.


### v2 terminal evidence and v3b retry prelaunch

- Version 2 (`20261027_spi_evidence_v3a`) ended ERROR in cell 2 after the
  10-image visual-transport preflight; cell 3 did not run. `campaign/status.json`
  is incomplete, zero labels and no four-class model. The root cause was strict
  parsing of complete JSON replies inside Markdown fences: all 10 were initially
  marked `parse_failed`. The current parser accepts only the exact complete
  fence wrapper; offline replay parsed all 10 with claims/evidence unchanged
  (8 Low, 1 Medium, 1 Critical, 0 High), no invalid response or nested-claim
  contradiction, and no model/prompt/class change. Request receipts confirm one
  image per request, image-bearing rendered text, only the instruction payload,
  evidence protocol, and thinking OFF. Generic legacy provenance
  `mode=annotation_assisted_visible_json` remains ambiguous metadata; actual
  request payloads show no annotations, and raw provenance was preserved.
- Retrieval was pinned to current-ref `/2`. 186 files retrieved; campaign direct
  files and all 45 checksum entries match; recovery ZIP SHA-256
  `386323ba5ca009377bbbcb33d7da026a113185d912c75edc03d8c79bef36f90d`, CRC
  valid, all 45 members match, and 108 source files match the runtime manifest.
  Kaggle CLI returned 1 after files arrived while printing a Windows charmap
  error; local integrity passed. Runtime confirms Python 3.13.15, two Tesla T4,
  CUDA 12.8. Cell logs and executed notebook were retrieved. Quota after v2:
  3.35 h used / 26.65 h remaining. Receipts and extracted recovery are under
  `experiments/spi_four_class/runs/20261027_spi_evidence_v3a/retrieved_version2/`.
- Affected parser tests passed 26/26 (`test_pipeline.py` and `test_spi.py`),
  with no source drift. Frozen v3b package SHA-256
  `bf06d6779e0bfcdf9f89c78218b60cebcfe22d996ae4fda999bbd77aaec3c85e`; embedded
  archive matches, 105 canonical source hashes match and three documented
  namespace stubs match their exact builder literal. Notebook schema and cells
  pass. Metadata targets numeric kernel ID `141089219`, uses the current alias
  for update, and restores title `BetterScheduler SPI 20261027finalc` so the
  original ref should return. Input remains private v1 READY. Prior v2 is
  terminal ERROR; v3b is authorized as version 3 on the same object, with
  7,200 seconds and a 4 aggregate GPU-hour cap. Prelaunch receipt:
  `experiments/spi_four_class/runs/ops_preflight_20261027/v3b_prelaunch.json`.

### Submission and monitoring receipt — SPI evidence v3b

- Kaggle accepted v3b as version 3 of numeric kernel object `141089219`. The
  original ref `satyapaladugu/betterscheduler-spi-20261027finalc` is restored;
  API metadata confirms current version 3, private state and the same v1 input.
  The status endpoint reported QUEUED at 2026-10-27 16:08:33 UTC. No second
  notebook object was created and no other run was stopped.
- Submitted source ZIP SHA-256:
  `bf06d6779e0bfcdf9f89c78218b60cebcfe22d996ae4fda999bbd77aaec3c85e`.
  Timeout 7,200 seconds; `NvidiaTeslaT4`; aggregate GPU-hour cap 4. The prior
  v2 run remains preserved as ERROR and its evidence/replay are unchanged.
- v3b run ID `20261027_spi_evidence_v3b`; Kaggle recorded COMPLETE at
  2026-10-27 17:02:34 UTC. Retrieval was pinned to original ref `/3`; recovery ZIP
  and Kaggle log are in
  `experiments/spi_four_class/runs/20261027_spi_evidence_v3b/retrieved_version3/`.
  The recovery bundle has 895 entries and a valid CRC. All 885 campaign checksum
  entries match; runtime source hashes match 79/79; the staged `source.zip` hash
  matches the submitted SHA above. Detailed receipt:
  `retrieved_version3/integrity_verification.json`.
- Run environment: managed Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8,
  two Tesla T4 GPUs. Campaign cell 2 completed in 2,898.97 seconds. Cell 3's
  captured `cell_status.json` says `running` and its per-cell log contains only
  the start line, while the Kaggle kernel log records the recovery export and
  `CELL END 3`, and Kaggle reports COMPLETE; retain this capture discrepancy.
- Actual result: `campaign/status.json` is `four_class_support_failed`, with no
  four-class student and no independent urgency validation. Support gate fails
  Medium (4 train / 1 validation) and High (8 / 6); Low (109 / 50) and Critical
  (80 / 39) pass its engineering minimum. The 300 labels are development-exposed
  teacher labels (train 204 / validation 96; full counts Low 159, Medium 5,
  High 14, Critical 119). Selected Qwen 3.5 evidence protocol passes authored
  text controls (48, macro-F1 1.0) and its 80-image visual panel is 80/80 valid;
  these are not image urgency accuracy. Direct protocol fails text controls
  (valid fraction 0.0833, macro-F1 0.1385). No student metrics were available;
  `analyze_run.py` reports no training and `runtime_promotion_allowed=false`.
  Analysis is in `retrieved_version3/analysis_recomputed/`.
- Submission receipt: `experiments/spi_four_class/runs/ops_preflight_20261027/v3b_submission_receipt.txt`;
  current status/identity receipt: `v3_postsubmit_status.txt`,
  `v3_postsubmit_identity.json` and `v3b_submission_runtime_status.json`.
- During v3 RUNNING, one UTF-8 `kernels logs` check returned only the CLI
  version warning; no cell text was exposed. Local C6 parity check passed on one
  cached train image using only `models/priority_classifiers/clip`: exact cached
  feature signature, cosine 1.0, max-absolute difference `1.49e-8` against `1e-4`
  threshold. No weights were downloaded and no feature cache was changed. Receipt:
  `experiments/spi_four_class/runs/local_feature_parity_20261027/feature_parity.json`.
- Local-only `analyze_run.py` smoke on checksum-verified v2 `recovery_extracted`
  passed: it reports incomplete/zero labels/no student, no training and
  `runtime_promotion_allowed=false`; no old three-class substitution. Receipt:
  `experiments/spi_four_class/runs/20261027_spi_evidence_v3a/retrieved_version2/analysis_v1/`.

### SPI visual grounding diagnostic — version 4 prelaunch

- Purpose: diagnose the v3 visual-grounding failure on a frozen 24-image enriched
  development panel (8 prior Critical, 5 Medium, 3 High, 8 Low; 24 distinct
  groups). Compare Qwen 3.5-9B and Qwen3-VL-8B with identical tightened evidence
  prompt; run 48 authored text controls and 24 images per teacher. No bulk
  labeling, student training, policy training, independent urgency claim or
  automatic promotion. Sampling labels/findings stay evaluator-only; request
  payloads are image plus instruction (text controls use hypothetical_scene).
- Existing notebook `satyapaladugu/betterscheduler-spi-20261027finalc`, numeric
  ID `141089219`, current version 3 COMPLETE; same attached input
  `satyapaladugu/betterscheduler-spiv2-inputs-20261027b` v1. No new identity or
  dataset. Local canonical notebook source copied to
  `experiments/spi_four_class/notebooks/spi_pipeline.ipynb`; prior canonical
  notebook preserved at `experiments/spi_four_class/runs/run20261028_spi_grounding_v1/canonical_notebook_before_v4.ipynb`.
- Snapshot: `experiments/spi_four_class/runs/run20261028_spi_grounding_v1/submission/`;
  source ZIP SHA-256 `5f385fbefde265f390d1520026e583d3cfbb37a2a4c1cb4f51d93da92ff9a3d1`.
  All 110 manifest entries match; notebook schema/cells, embedded archive, code
  compilation, same kernel/input, T4, 3,600-second server limit and 2 aggregate
  GPU-hour cap pass. Focused checks: 18 passed in 4.77 seconds, no source hash
  drift (`.../freeze_checks/`). Package receipt:
  `experiments/spi_four_class/runs/run20261028_spi_grounding_v1/package_validation.json`.
- Prelaunch: v3 status COMPLETE; live quota 25.82 GPU-hours remaining. Quota
  moved from 3.35h used before v3 to 4.18h now (rounded delta 0.83h); the new
  2 GPU-hour cap leaves 23.82h at worst. Prelaunch record:
  `experiments/spi_four_class/runs/run20261028_spi_grounding_v1/prelaunch/prelaunch_receipt.json`.
  Authorized submission target: version 4 on the same notebook, timeout 3,600s,
  `NvidiaTeslaT4`. Expected output root
  `/kaggle/working/experiments/spi_four_class/runs/run20261028_spi_grounding_v1/`.

### SPI grounding diagnostic submission — v4 live status

- Kaggle accepted version 4 on the existing numeric kernel object `141089219`;
  original ref/title restored. Submit command used timeout 3,600s and
  `NvidiaTeslaT4`; CLI exit 0. Package hash remains
  `5f385fbefde265f390d1520026e583d3cfbb37a2a4c1cb4f51d93da92ff9a3d1`.
  Submission receipt: `experiments/spi_four_class/runs/run20261028_spi_grounding_v1/submission_receipt.json`.
- Initial remote status is RUNNING. Bounded 35-second monitor (100-minute
  deadline; never cancels) writes transitions to
  `experiments/spi_four_class/runs/run20261028_spi_grounding_v1/status_changes.jsonl`.
  Retrieve explicit `/4` recovery ZIP after terminal status; verify source and
  completed cell status before interpreting results.

### SPI grounding diagnostic v4 terminal result

- Version 4 ended ERROR at 2026-10-27 21:57:46 UTC. Pinned `/4` retrieval
  fetched the partial campaign recovery ZIP and the executed notebook; the outer
  `run_recovery.zip` was not produced because cell 3 never ran. The recovery ZIP
  SHA-256 is `169440f0db5636b6f07c296ca797773d4761494e40b3331498a9106113ad996b`;
  all 10 internal campaign checksums match, 79 runtime source hashes match, and
  source ZIP is the approved `5f385fbefde265f390d1520026e583d3cfbb37a2a4c1cb4f51d93da92ff9a3d1`.
- Failure: first Qwen 3.5 text-control call raised
  `ValueError: supply images for visual review, or explicitly request the text-only diagnostic`.
  The diagnostic called `review_structured` with no image but omitted its
  `text_only_diagnostic=True` flag. Campaign stage 01 completed (62.77 s), model
  load completed on GPU1 (207.68 s), and text stage failed immediately (0.004 s).
  No text control response or visual response completed; Qwen3-VL was not loaded.
- The recovered executed notebook has cells 0/1 complete, cell 2 errored, cell 3
  not executed. No `cell_status.json`/per-cell logs were included; this capture
  limit is preserved. Runtime: Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8,
  two T4s, thinking OFF. Legacy reviewer metadata says
  `mode=annotation_assisted_visible_json`; no request receipt exists because the
  failure precedes inference. The attempted text payload was hypothetical-scene
  only; it was rejected before an emitted model request.
- This run made no accuracy, training, or promotion result. `status.json` is
  incomplete and both training/promotion flags are false. Rounded quota moved
  from 4.18h used before run to 4.27h after (0.09h displayed delta). Verified
  artifacts are under `experiments/spi_four_class/runs/run20261028_spi_grounding_v1/retrieved_version4/`;
  `partial_integrity.json` and `cell_capture_summary.json` summarize checks.


### SPI grounding diagnostic corrective run — v5 prelaunch

- Version 4 failed before inference. Exact cause: `assess()` omitted the
  reviewer API's `text_only_diagnostic=True` argument for text controls. No VLM
  response, visual comparison, training or promotion occurred. Partial artifacts
  and checksum audit are preserved under
  `experiments/spi_four_class/runs/run20261028_spi_grounding_v1/retrieved_version4/`.
- Added the explicit mode forwarding and regression coverage against the actual
  `QwenAnnotationReviewer` method contract. Focused tests: 19 passed in 10.40s.
  Frozen diagnostic package SHA-256:
  `11c88ee29e3389be4c36d75d48ae72d174239984c0eec3f7ee32d220e1a90210`.
  All source manifest, code cell, archive embedding, source identity, panel,
  thinking OFF, and 3,600 second/2 aggregate GPU hour cap checks pass.
- Quota before v5: 4.27 GPU hours used, 25.73 hours remaining (2026-10-31
  refresh). v4 is confirmed terminal ERROR. Same private notebook numeric ID
  141089219; original reference/title; existing private input v1. No second
  notebook identity and no other job cancellation.
- Run `run20261028_spi_grounding_v2`; local submission package and canonical
  notebook source are under `experiments/spi_four_class/runs/run20261028_spi_grounding_v2/` and
  `experiments/spi_four_class/notebooks/spi_pipeline.ipynb`. A pre-v5 notebook
  copy and prelaunch receipt are preserved in that run directory. Submit only
  the validated package, then record the actual returned version and terminal
  status before interpretation. Expected remote output is not evidence.

- Submitted v5 to the same notebook. Kaggle CLI exit 0; returned version 5 and
  RUNNING for numeric ID 141089219. Current ref and input are unchanged. The
  frozen source hash is `11c88ee29e3389be4c36d75d48ae72d174239984c0eec3f7ee32d220e1a90210`.
  Submission receipt: `experiments/spi_four_class/runs/run20261028_spi_grounding_v2/submission_receipt.json`.
  Monitor is bounded to the declared 3,600 second server cap; retrieve explicit
  `/5` artifacts only after terminal status. No result is claimed yet.

### SPI grounding diagnostic v5 terminal retrieval

- Rechecked on 2026-10-28: same Kaggle notebook `satyapaladugu/betterscheduler-spi-20261027finalc`, numeric ID `141089219`, version 5 is **COMPLETE**. No new notebook identity was created and no remote job was submitted or cancelled in this ops pass.
- The retrieved run was preserved under `experiments/spi_four_class/runs/run20261028_spi_grounding_v2/retrieved_version5/retrieval_capture/`. The copied retrieval manifest contains 317 file entries; all 317 SHA-256 values verify. Outer `run_recovery.zip` SHA-256: `63b74a999b8c9fc6b2447f10facc77ba26aa81ac2b5a990021c835185a8e8fbc`. An already-expanded campaign tree remains alongside it under `retrieved_version5/experiments/spi_four_class/runs/run20261028_spi_grounding_v2/campaign/`.
- The diagnostic is **not SPI model readiness**. Qwen3.5-9B: 48/48 text controls, macro-F1 1.0 (text-policy conformance only); 24/24 valid visual outputs, 22 available, 7 changed from v3. Qwen3-VL-8B: 48/48 text controls, macro-F1 0.9590 (same limited scope); 24/24 valid visual outputs, 14 available, 14 changed from v3. No independent image-urgency reference exists, so visual accuracy is undefined. Training and promotion remain prohibited. These outputs do not clear the four-class SPI readiness gate.
- `kaggle quota` at the same check reported GPU 4.63h used / 25.37h remaining of 30h, refresh `2026-10-31T00:00:00Z`; TPU 0h used / 20h remaining. Current run/output status and checksum evidence are recorded in `experiments/spi_four_class/runs/run20261028_spi_grounding_v2/v5_terminal_receipt.json`.
- Next Kaggle action is gated on the main thread's Gemma adapter/package and explicit readiness. Do not submit until that package is ready; do not start full-image caching until SPI readiness passes.

### Gemma SPI diagnostic v6 — submitted on existing notebook

- Version 6 was submitted to the existing private notebook ID `141089219` (`satyapaladugu/betterscheduler-spi-20261027finalc`); CLI returned success and immediate Kaggle status is RUNNING. No new notebook identity was created.
- Package/run: `experiments/parameter_finalization/runs/20261028_implementation/gemma_submission_checked/`; run ID `20261028_gemma_diagnostic_checked`; input `satyapaladugu/betterscheduler-spiv2-inputs-20261027b`. Frozen source ZIP SHA-256 `db20d35e3e21f4a0c0a35dfcff4925c6c24adfe9f00f89bdc201ee9a010e9751`; checked notebook SHA-256 `75135135320acd54ff9b59e7a13da5dc277105dac33aedb5ec99d448fa4f9c6f`.
- Pre-submit checks: ZIP matches its 112-file source manifest; embedded ZIP is byte-identical to external ZIP; all four notebook code cells compile and notebook is nbformat 4.5; only Gemma4-E4B is selected; diagnostic is fixed to 48 text controls and 24 visual cases, with training/promotion disabled. Three current-tree differences are the exact documented isolated namespace stubs; the other 109 packaged files match current source. Previous canonical notebook preserved as `canonical_notebook_before_v6.ipynb`; canonical source now matches the submitted notebook.
- Runtime limits: 3,600 seconds, 2 aggregate GPU-hours, NvidiaTeslaT4, thinking OFF; Transformers 5.5.0 Gemma NF4/FP16 on GPU1 as recorded in the checked package. Do not start training or bulk image parameter caching from this diagnostic. Immediate failure should be inspected and reported before any retry; do not cancel a healthy run.
- Submission receipt: `experiments/parameter_finalization/runs/20261028_implementation/gemma_submission_checked/v6_submission_receipt.json`. Status history: `v6_status_changes.jsonl`. Expected output: `/kaggle/working/experiments/spi_four_class/runs/20261028_gemma_diagnostic_checked/`; retrieve explicit version 6 outputs only after a terminal status and verify recovery ZIP/checksums/raw responses.

### Gemma all-300 diagnostic v7 — submitted on existing notebook

- Version 7 was submitted to the existing private notebook ID `141089219` (`satyapaladugu/betterscheduler-spi-20261027finalc`); Kaggle accepted it and immediate status is RUNNING. No new notebook identity was created.
- Run/package: `experiments/parameter_finalization/runs/20261028_implementation/gemma_all300_submission/`, run ID `20261028_gemma_all300`; source ZIP SHA-256 `2b5344794746b18143e408b80af6fa30444bd7209c8a2bb4eaa2637a7ca54daf`; checked notebook SHA-256 `fdcf31d3edf8a859f604155f35d4d4e31188e5ed538f4c650302c32a2ccd4401`.
- Pre-submit checks: 113 source files match the embedded manifest; embedded ZIP equals external ZIP; 110 non-stub files match current source and the three isolated namespace stubs match the documented bytes; notebook is nbformat 4.5 and every code cell compiles. Driver imports `experiments.spi_four_class.gemma_300.execute`, uses one Gemma teacher, and targets 300 images without repeating text controls or the old 24-image panel. Training, promotion, and bulk parameter caching remain disabled; unknowns must remain unknown.
- Runtime/input caps: 3,600 seconds, 2 aggregate GPU-hours, NvidiaTeslaT4, same pinned 300-image input, Transformers 5.5.0, Gemma NF4/FP16 on GPU1, thinking OFF. Expected output root: `/kaggle/working/experiments/spi_four_class/runs/20261028_gemma_all300/`.
- V6 timing projection: 24-image visual stage 114.25s (4.761s/image), median per-call 4.586s, plus 150.68s model load. Simple linear estimate for 300 image calls plus load is about 1,579s before data validation and export overhead; this is only a projection, not measured 300-image runtime.
- Receipt: `v7_submission_receipt.json`; status history: `v7_status_changes.jsonl`. Monitor this version and retrieve explicit `/7` outputs only after terminal status, verifying raw outputs, recovery archive and checksums. Do not train, promote, cache parameters, or retry blindly.

### Gemma SPI diagnostic v6 — terminal result and verified retrieval

- Kaggle v6 is COMPLETE. Runtime confirms Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 5.5.0, CUDA 12.8, two T4s, and thinking OFF. Gemma4-E4B-it revision `ee0ef6023621cff504d758262d4e04895a5af4a2`, NF4/double-quant FP16, loaded on GPU1; model load metadata reports 81.15s, while the complete load stage took 150.68s.
- Text controls: 48/48 valid, macro-F1 `0.9372`; recalls Low `1.000`, Medium `0.917`, High `0.833`, Critical `1.000`. These are authored text-policy controls, not image urgency accuracy.
- Visual diagnostic: all 24 calls valid, 1 available (Low), 23 abstentions; every output changed from the v3 label. There is no independent image-urgency reference, so accuracy is undefined. No training or promotion was allowed; this result does not establish SPI readiness.
- Visual stage elapsed `114.25s` for 24 images (`4.761s/image` including stage overhead); per-call median `4.586s`, mean `4.701s`, p90 `4.960s`. Linear projection: 300 image calls ≈`1,428s`, plus the full model-load stage ≈`151s`, total ≈`1,579s` before validation/export overhead. This is a planning estimate, not measured 300-image runtime.
- Explicit `/6` retrieval verified: 239/239 top-level retrieval hashes; 230/230 campaign checksums against downloaded files and recovery ZIP members; both run recovery ZIP CRC checks passed. One initial file fetch returned HTTP 502 HTML; one targeted retrieval of that exact file succeeded and its correct checksum now passes. The original error response is preserved in `retrieval_errors/`.
- Receipts and raw evidence: `experiments/parameter_finalization/runs/20261028_implementation/gemma_submission_checked/v6_submission_receipt.json` and `retrieved_version6/`. No run retry or model substitution occurred.

### Under-10B VLM direct-class comparison v8 — submitted

- Version 8 was submitted after v7 reached COMPLETE, to the existing private notebook ID `141089219` (`satyapaladugu/betterscheduler-spi-20261027finalc`). Kaggle returned version 8 successfully; the immediate status check was recorded as RUNNING. No new notebook identity was created.
- Package: `experiments/parameter_finalization/runs/20261028_implementation/v8_comparison_checked/`, run `20261028_under10b_comparison_v8_checked`; source SHA-256 `36c5ec35ed09d6a7621d598fc39eeffae727beb73106a840d2fcbad19df7dcb7`; notebook SHA-256 `443938978aa04b30d5a6f0ad2b149e6d5dff6b93280b0e977e4cf372f07757f1`.
- The notebook calls `experiments.spi_four_class.compare_vlms.execute`. Its fixed candidate set is Gemma4-E4B, Qwen3.5-9B, Qwen3-VL-8B, Qwen2.5-VL-7B, and InternVL3-8B HF. It uses the 24-image held-out-from-training diagnostic panel plus 48 authored text controls, the direct four-class prompt, sequential one-teacher-at-a-time execution, and per-candidate failure capture. It does not train/promote a model, cache parameters, or automatically select a winner.
- Pre-submit validation: 116/116 archive manifest entries match; all 113 non-stub source files match the current repository; three namespace stubs match documented bytes; embedded ZIP matches source ZIP; notebook is nbformat 4.5 and code cells compile. Metadata targets the existing ID/slug and existing input. Runtime cap 7,200s / 4 aggregate GPU-hours.
- Previous canonical notebook copy: `canonical_notebook_before_v8.ipynb`. Submission receipt: `v8_submission_receipt.json`; status history: `v8_status_changes.jsonl`. Monitor v8 and retrieve explicit `/8` outputs only after terminal status; verify each candidate’s logs/results/checksums. Expected output is not evidence.

### Gemma all-300 v7 terminal outcome (retrieval scope noted)

- V7 finished COMPLETE. Stage times: Gemma load `147.19s`; 300-image labeling `1,587.04s` (about `5.29s/image` including extraction-loop overhead). All 300 image assessment records report completed calls.
- Retrieved label file result: `19 Low`, `281 unavailable/None`; no Medium, High, or Critical outputs. This is a clear failure to produce a usable four-class SPI labeling set. It is not an accuracy result: there is no independent image-urgency reference. Preserve all 281 unknowns; do not convert them to Low/Normal for SPI fitting. No student training, promotion, or parameter cache occurred.
- Gemma revision `ee0ef6023621cff504d758262d4e04895a5af4a2`, Transformers `5.5.0`, NF4/FP16 on GPU1, thinking OFF. Stage logs and labels are in `experiments/parameter_finalization/runs/20261028_implementation/gemma_all300_submission/retrieved_version7/`.
- Retrieval limitation: the Kaggle run's `campaign/checksums.json` lists 909 payload files. The local output retrieval currently has 219 matching files, zero corrupt files, and 690 not retrieved; all 300 labels plus stage logs are present. This is an incomplete raw-request artifact retrieval, not missing model output. The notebook output files endpoint pagination did not return additional matching files through the CLI attempt; retain this limitation and do not claim all raw requests were locally verified.
- Version 8 (the five-model fixed-panel comparison) was submitted after v7 terminal status; see its own receipt/status history. Account quota at submission remained within the planned 8-GPU-hour study budget based on attributed version deltas; older untracked account use remains unattributed.

### Under-10B VLM direct-class comparison v8 — complete and verified

- Kaggle version 8 is COMPLETE on the existing notebook ID `141089219`. All five candidates completed their 24 visual-panel cases and 48 authored text controls. The run made no model selection, SPI training, promotion, or bulk parameter cache.
- Full recovery bundle: `experiments/parameter_finalization/runs/20261028_implementation/v8_comparison_checked/retrieved_version8/run_recovery.zip` (SHA-256 `21bb4ff008e0ae49ea85f92f13fd050f61a7a997dcf37e8953ec2e0eb3c736eb`); expanded verified outputs: `.../retrieved_version8/recovery_expanded/`. ZIP CRC passes. All 982 recovery-manifest file hashes and all 973 campaign-manifest file hashes verify, including the five raw 24-image label JSON files, 240 text-control records, individual request/output captures, model provenance, cell logs, executed notebook and status/runtime.
- Runtime: Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 5.5.0, CUDA 12.8, two Tesla T4 GPUs; sequential single-model use on GPU1; thinking OFF. Prompt SHA-256 `d82cf5e618264170488cd92ff0cba5da04c9861aebc21f03bbeac687d5b992a9`; 24-image panel SHA-256 `634d3f5eee9477a7f54fc54fd64d424699fa16124d376d131cae8327a4c74e1d`.
- Results below are descriptive model behavior on this fixed panel, not image-urgency accuracy (there is no independent image-urgency reference). All five produced 24 valid visual responses:

| Model | Visual class counts (Critical / High / Medium / Low) | Text-control macro-F1 | Median image-call time | Model footprint |
|---|---:|---:|---:|---:|
| Gemma 4 E4B | 0 / 0 / 0 / 24 | 0.6455 | 0.722 s | 9.257 GB |
| Qwen 3.5 9B | 2 / 8 / 5 / 9 | 0.9164 | 1.125 s | 7.765 GB |
| Qwen 3 VL 8B | 1 / 8 / 0 / 15 | 0.6931 | 0.842 s | 6.258 GB |
| Qwen 2.5 VL 7B | 6 / 4 / 5 / 9 | 0.5953 | 0.830 s | 5.785 GB |
| InternVL3 8B HF | 7 / 0 / 1 / 16 | 0.7371 | 3.152 s | 5.607 GB |

- Qwen 3.5 is the only candidate passing the authored text-control criterion (class recalls: Low 1.000, Medium 0.833, High 0.833, Critical 1.000). This does not make it an image-urgency winner. Gemma classified every visual-panel case Low; Qwen 3 VL emitted no Medium; Qwen 2.5 and InternVL3 also failed at least one class recall. Keep this as evidence of prompt/model behavior and class coverage gaps; do not train on these labels as though they were ground truth.
- Stage logs total 1,171.62 seconds. FLOPs were not instrumented. Candidate IDs, revisions, load timings, memory, per-case outputs and all failure/recovery evidence are in `v8_verified_receipt.json` and the expanded recovery tree. No accuracy-based winner claim is supported.

### Notebook9 full 300-image parameter finalization v9 — terminal preflight failure

- Target remains the existing private kernel `satyapaladugu/betterscheduler-spi-20261027finalc`, ID `141089219`; expected new version 9. Attached input is the existing READY dataset `satyapaladugu/betterscheduler-spiv2-inputs-20261027b` (300 development-exposed images, feature cache and panel; Kaggle expands `images.zip` to `images/images/<sha>.png`). No new notebook or dataset identity was created.
- Staged package: `experiments/parameter_finalization/runs/notebook9_20261028/submission_finalcandidate/`. Source ZIP SHA-256 `199fd16e47f4bf0bb40075ecdf78e467ceaa12e2ac0d44860fd5a44505f09dcc`; notebook SHA-256 `d2c25558c9902b18bf47e95d7f2e261e662ee61dd52758373051a291b0521269`; separate runtime-config SHA-256 `954df05f24610823e651ee33665f75527ce254711b79d99f0d27aaa10c74bb57`. The dirty working-tree snapshot records HEAD `032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0` plus hashes for all 381 archived source/config files.
- Staged runtime config alone enables `PARAMETER_RUN_REAL=1`; canonical notebook source remains unchanged. Bootstrap verifies embedded ZIP/config and extracted source hashes. Two-T4 assertion occurs before model downloads; the reviewed cap is 7,200 server seconds / at most 4 aggregate GPU-hours. Downloaded HF CLIP, YOLO-World and OpenAI CLIP assets are pinned by revision/hash and bounded size/time. The input manifest and a fresh canonical-wrapper CLIP feature signature, then real YOLO-World 16-class first-image inference, must pass before teacher labeling. SPI fit/cache remain readiness-gated. No full-corpus work is enabled.
- Local package checks passed: ZIP CRC; 381/381 source hashes; syntax compile of archived Python; notebook schema and both notebook code cells compile; expected kernel ID/input metadata; pinned YOLO/CLIP dependency declarations; run-cap and fail-closed preflight markers. A one-image local CPU invocation of canonical `MinimalVisionProvider.extract_batch` matched the input feature signature exactly (`3f526c96ba934fdc42657a77be22908459060da741676c067575645822faf1de`).
- Corrected lean package after the rejected oversized candidate: `experiments/parameter_finalization/runs/notebook9_20261028/submission_lean/`. The source ZIP excludes `parameters/results/**`, `parameters/tests/**`, and `__pycache__`; it contains 115 production/config/schema sources, with no local model weights. The HF CLIP tokenizer files are fetched from the same pinned revision with size and SHA-256 checks. ZIP SHA-256 `560920182dedba6bdb3e81c00e3f2fb7df130389d5e8b945040631934da73c7c`; notebook SHA-256 `ad1faa6e08903de9995940e5b09f3bba78c42c80e13be93f9a0644a24a5bbc3a`; notebook size 470,538 B versus prior v8 922,411 B. ZIP/source hashes, archived Python syntax, notebook schema/cells and existing metadata identity/input/cap checks pass locally. Status before launch: prepared; main reviewed and cleared this corrected candidate.
- Accepted launch: Kaggle API returned version 9 at `2026-10-28T04:42:55Z` for the existing kernel; no invalid tags, datasets or sources. The accepted submission receipt is `experiments/parameter_finalization/runs/notebook9_20261028/submission_lean/launch_receipt.json`. Version 9 reached `KernelWorkerStatus.ERROR` at `2026-10-28T04:45:02Z`. Runtime logged Python 3.13.15, Torch 2.11.0+cu128, CUDA 12.8, and two Tesla T4 GPUs. Dependency setup completed; the real pre-label asset check failed after 69.6 s at `RuntimeError: pinned HF CLIP tokenizer asset size/hash mismatch: tokenizer.json`. No teacher label, SPI fit or parameter cache stage ran. Captured Kaggle execution log: `experiments/parameter_finalization/runs/notebook9_20261028/submission_lean/monitor_logs/kaggle_logs_latest.txt` (41,057 B).
- Kaggle account quota before v9 was 5.66 GPU-hours used / 24.34 remaining; after the 69.6 s preflight failure it reported 5.69 used / 24.31 remaining (refresh `2026-10-31T00:00:00Z`). Kernel v8 is COMPLETE. The version-9 push was attempted after main-thread clearance: attempt 1 ended with Windows `ConnectionAbortedError(10053)`; the one authorized retry returned HTTP 400 from Kaggle `SaveKernel`. Post-attempt pull still matches the previous v8 source SHA-256 `52e4662df10ebc5656f38b83f9c4e1403cf947e3ffd163051a34c378c04dbc9d`, so the first oversized candidate was not accepted and no run launched from it. Attempt receipts are in `experiments/parameter_finalization/runs/notebook9_20261028/submission_finalcandidate/submission_attempts.jsonl`; preserve the package and investigate the API rejection before any further push. CLI stderr did not expose the HTTP response body. Read-only comparison shows the v9 notebook payload is 2,883,734 B (largest cell 2,861,077 chars) versus v8 at 922,411 B; the ZIP builder also included historical `parameters/results/**` modules. This is a likely request-size cause, not a confirmed Kaggle error explanation. Details: `experiments/parameter_finalization/runs/notebook9_20261028/submission_finalcandidate/rejection_diagnosis.json`. The v9 output API listed 135 files; all 135 were retrieved locally (one extra local CLI log file is also present). All five entries in the run `retrieval_checksums.json` match, all 115 extracted source files match `SOURCE_MANIFEST.json`, and `run_recovery.zip` is present (19,426 B; ZIP CRC passes). Retrieval audit: `experiments/parameter_finalization/runs/notebook9_20261028/submission_lean/retrieval_receipt.json`; retrieved outputs: `experiments/parameter_finalization/runs/notebook9_20261028/submission_lean/retrieved_version9/`. A current read-only check at `2026-10-28T05:33:19Z` confirms v9 remains ERROR (latest run time `04:42:57Z`) and all 135 listed outputs remain locally present; latest logs SHA-256 `374d3ea3df8693b7fbf4916d00815e0bdfd79b6611819a73a10d8d73830f7ba0` are saved in `submission_lean/monitor_logs/kaggle_logs_v9_live_20261028T053319Z.txt`. Main independently identified the exact failure: the staged notebook tokenizer.json expected SHA literal omitted digits `411`; the correct hash at the pinned HF revision is `b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842`, matching the local file. The embedded literal caused the fail-closed preflight. No further push or run retry was made.

### Notebook9 parameter finalization v10 — corrected tokenizer asset pins

- Submitted 2026-10-28 at 05:35:11Z to the existing private kernel `satyapaladugu/betterscheduler-spi-20261027finalc` (ID `141089219`). Kaggle accepted version 10; immediate `kaggle kernels status` returned `KernelWorkerStatus.RUNNING`. This reuses the existing identity. No retry or cancellation was issued.
- Candidate package: `experiments/parameter_finalization/runs/notebook9_20261028/submission_v10_fixed/`. Notebook SHA-256 `07b2c6f42b80c165d9a74fdd28ac732b735ab13a6cd6c3e9529a6769574fe180`; source ZIP SHA-256 `d912b15e624f8a2381419e2348600131587d59f5e2cd23bdc22ceb2d49bed053`; runtime config SHA-256 `954df05f24610823e651ee33665f75527ce254711b79d99f0d27aaa10c74bb57`. Independent pre-submit checks: 115/115 manifest source hashes match, ZIP CRC passes, notebook schema valid. Main’s builder now derives runtime CLIP asset hashes/sizes from local assets to avoid duplicate literals. Run receipt: `experiments/parameter_finalization/runs/notebook9_20261028/submission_v10_receipt.json`.
- Runtime limits: 7,200 server seconds and 4 aggregate GPU-hours; expected two Tesla T4s with actual runtime preflight. Input remains `satyapaladugu/betterscheduler-spiv2-inputs-20261027b`. Expected output root `/kaggle/working/experiments/parameter_finalization/runs/notebook9_20261028_v10/`, including incremental cell logs/status and executed notebook, plus `/kaggle/working/run_recovery.zip`. No v10 logs or output artifacts have been retrieved yet; monitor the existing version and retrieve only after terminal status. Version 9 remains preserved with its verified pre-label tokenizer hash failure and retrieved recovery artifacts.

- Follow-up check at 2026-10-28T05:35:40Z: still RUNNING; Kaggle logs endpoint returned no cell output yet. v10 source asset manifest pins the exact pinned-revision CLIP tokenizer sizes and 64-character hashes, including tokenizer.json 556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842; this now matches the independently verified local/HF asset receipt.

#### v10 terminal outcome and retrieved evidence

- Latest version-specific check: version 10 is `KernelWorkerStatus.ERROR`. The captured execution log is 53,069 B, SHA-256 `cd1c48dcb0172b684e3a3777eaecf285b02b39ee942c177da202a325ef905aa3`, at `experiments/parameter_finalization/runs/notebook9_20261028/submission_v10_fixed/monitor_logs/kaggle_logs_v10_terminal.txt`. Runtime was Python 3.13.15, Torch 2.11.0+cu128, CUDA 12.8, two Tesla T4s. Dependency installation completed. All three HF tokenizer assets passed actual-size and SHA checks; the YOLO-WORLD weight and OpenAI CLIP weight hashes also passed. Cells 0–3 completed; cell 4 failed in `verify_inputs` after about 84 seconds with `FileNotFoundError` for `/kaggle/input/betterscheduler-spiv2-inputs-20261027b/panel.json`. No teacher labels, SPI training or cache ran.
- Kaggle dataset API independently lists 304 files in the selected input: root `panel.json`, `input_manifest.json`, `features.npz`, `selection_audit.json`, and 300 `images/images/*.png`. The runtime path therefore did not expose the listed root layout; investigate the mounted directory before any next version. Do not assume the API listing proves runtime mount layout.
- Retrieved 19 files into `experiments/parameter_finalization/runs/notebook9_20261028/submission_v10_fixed/retrieved_version10/`. The run recovery manifest lists 10 payload files; all 10 local hashes match. The executed notebook and per-cell logs/status are present. Kaggle output retrieval did not expose `run_recovery.zip`; record this as unavailable, not verified. The local retrieval command stalled after downloading the files and was stopped locally; remote kernel was already terminal and was not cancelled. Full receipt: `experiments/parameter_finalization/runs/notebook9_20261028/submission_v10_receipt.json`.

### Notebook9 parameter finalization v11 — mounted-input discovery

- Submitted 2026-10-28 at 05:43Z to existing private kernel `satyapaladugu/betterscheduler-spi-20261027finalc` (ID `141089219`). Kaggle accepted version 11; immediate status `KernelWorkerStatus.RUNNING`. No new identity, retry, or cancellation.
- Candidate `experiments/parameter_finalization/runs/notebook9_20261028/submission_v11_mountfix/`. Notebook SHA-256 `36119afd88231c2eeb5dd30a1d13dd7e8b7f67497f2c9c71e4c0c58ef1a218c9`; source ZIP SHA-256 `4451f64938baf2da326ddd78e819b96333c841a531dcceb8cc44458e94042cea`; config embedded in notebook SHA-256 `954df05f24610823e651ee33665f75527ce254711b79d99f0d27aaa10c74bb57`. Independent checks: 115/115 source-manifest hashes, ZIP CRC, notebook schema, and canonical code-cell equality passed. Source discovers the exact input root from manifest SHA and verifies required panel/features; rejects missing, wrong, ambiguous or incomplete mounts. RUN_REAL remains enabled only by deployed wrapper config; readiness gates are unchanged.
- Runtime cap: 7,200 seconds and 4 aggregate GPU-hours. v9 and v10 used approximately 0.085 aggregate GPU-hours combined based on logged two-T4 execution times (69.7 s + 83.8 s). Expected output `/kaggle/working/experiments/parameter_finalization/runs/notebook9_20261028_v11/`, recovery `/kaggle/working/run_recovery.zip`. Immediate status/log poll is RUNNING with no cell logs yet. Continue bounded monitoring; retrieve and verify only after terminal state. Receipt: `experiments/parameter_finalization/runs/notebook9_20261028/submission_v11_receipt.json`.

- Quota recheck at 2026-10-28T06:26Z via authenticated Kaggle `quota_view()`: API returned GPU `timeUsed="20724.555.0s"`, `totalTimeAllowed="21600s"`, refresh `2026-10-31T00:00Z` (interpreted provisionally as about 875 seconds / 0.243 GPU-hours remaining). This differs from the earlier CLI quota record of 24.31 GPU-hours remaining after v9; preserve both observations and treat the newer lower API reading as the operative caution until reconciled. Version 11 was already accepted and RUNNING before this check; do not cancel it. Prior v9+v10 observed aggregate use is approximately 0.085 GPU-hours. Version 6 and 8 logged cell time imply conservative two-device upper estimates of 0.310 and 0.696 GPU-hours; v7 exact runtime capture is missing, though its package cap was 2 aggregate GPU-hours.
- Bounded monitor script: `experiments/parameter_finalization/runs/notebook9_20261028/monitor_v11.ps1`, local monitor artifacts under `experiments/parameter_finalization/runs/notebook9_20261028/monitor_v11/`; polls version-specific status/logs for up to three hours, retrieves only at terminal status, and never cancels or resubmits.

### Notebook9 parameter finalization v12 — output stream compatibility fix

- Submitted 2026-10-28 at 06:28Z to existing private kernel `satyapaladugu/betterscheduler-spi-20261027finalc` (ID `141089219`). Kaggle accepted version 12; immediate status `KernelWorkerStatus.RUNNING`; startup logs were empty. No new identity, retry, or cancellation.
- Package `experiments/parameter_finalization/runs/notebook9_20261028/submission_v12_streamfix/`. Notebook SHA-256 `d6bb9f0e686d181a883b0b516a93f66f73dd74c0ed916eae11698a64fbafd3e9`; source ZIP SHA-256 `78596d05e9d6e5cb2595d764da1b54addd116924f142bab85abc8c58dec4fd8f`; config hash `954df05f24610823e651ee33665f75527ce254711b79d99f0d27aaa10c74bb57`. Independent checks: 115/115 source hashes, ZIP CRC, notebook schema, canonical cell equality, expected tokenizer sizes/hashes, and deployed real-run setting passed. `Tee.isatty()` and delegated stream metadata are present in `parameters/artifacts.py`.
- Runtime cap: 7,200 seconds / 4 aggregate GPU-hours. Current authenticated Kaggle CLI quota view: 5.76h used, 24.24h remaining of 30h; refresh 2026-10-31T00:00Z. Conservative prior campaign upper estimate (including missing exact v7 runtime) is ~3.18 aggregate hours; with v12 full cap it remains below the existing 8h campaign ceiling. Expected output `/kaggle/working/experiments/parameter_finalization/runs/notebook9_20261028_v12/`, recovery `/kaggle/working/run_recovery.zip`. Receipt `experiments/parameter_finalization/runs/notebook9_20261028/submission_v12_receipt.json`; monitor status/logs and retrieve only at terminal state.

#### v11 terminal outcome and verified recovery entries

- Version 11 reached `KernelWorkerStatus.ERROR` at 2026-10-28T06:26:23Z after about 155 seconds. The runtime discovered the expected mount at `/kaggle/input/datasets/satyapaladugu/betterscheduler-spiv2-inputs-20261027b` and selected its manifest by exact SHA-256 `b633e716aa20a36d7f5581dce49bffebc260516ba5ecf78a1ec0ee83ddd92be0`. Two T4s and model asset hashes passed. Cell 4 failed during CLIP `AutoModel.from_pretrained` before feature signature equality: `AttributeError: 'Tee' object has no attribute 'isatty'` from Transformers loading-report formatting. Cells 0–3 completed; no Qwen labels, SPI training or parameter cache ran.
- Full 60,213 B log SHA-256 `1f8c20a12593c4cb398fd88b9c0e906b0ec723ccc3929cecf0289c363dc0235d`: `experiments/parameter_finalization/runs/notebook9_20261028/monitor_v11/kaggle_logs_v11_latest.txt`. Retrieved 19 files; all 10 entries in the recovery checksum manifest match; executed notebook and per-cell logs are present. Kaggle did not expose `run_recovery.zip`. Full receipt `experiments/parameter_finalization/runs/notebook9_20261028/v11_terminal_receipt.json`. The local Kaggle output helper stalled after these files were saved and was stopped locally; remote run was already terminal and not cancelled.

#### v12 terminal outcome and verified artifacts

- Version 12 reached `KernelWorkerStatus.ERROR` at 2026-10-28T06:31:52Z after about 123 seconds. Runtime: Python 3.13.15, Torch 2.11.0+cu128, CUDA 12.8, two Tesla T4s. Asset/dependency checks passed; input discovery selected `/kaggle/input/datasets/satyapaladugu/betterscheduler-spiv2-inputs-20261027b` by manifest SHA `b633e716aa20a36d7f5581dce49bffebc260516ba5ecf78a1ec0ee83ddd92be0`; Kaggle expanded input layout and verified 300 image member hashes. Fresh CLIP feature signature `105485efb3f14d80514f39138c3310714dfdbee850291a8f4e76551df5e5f065` differed from the input cache signature `3f526c96ba934fdc42657a77be22908459060da741676c067575645822faf1de`, so the fail-closed preflight stopped before detector warmup, teacher labels, SPI training, or parameter cache.
- Captured full 49,592 B log SHA-256 `7b22f2c8bbaad5eef1f2414da45871380fde34e2a0769972d76f68b5f25cd783` at `experiments/parameter_finalization/runs/notebook9_20261028/monitor_v12/kaggle_logs_v12_terminal.txt`. Retrieved 19 files; all 10 recovery-manifest entries match their hashes, and the executed notebook/per-cell logs are present. Kaggle did not expose `run_recovery.zip`; the local output helper stalled after all these files were saved and was stopped locally, not remotely. Full receipt: `experiments/parameter_finalization/runs/notebook9_20261028/v12_terminal_receipt.json`.
- Quota after v12: 5.79h GPU used / 24.21h remaining of 30h; refresh 2026-10-31T00:00Z. v12 logged about 123s on two T4s, conservatively about 0.068 aggregate GPU-hours.

### Notebook9 parameter finalization v13 — runtime feature-space alignment

- Submitted to the existing private kernel `satyapaladugu/betterscheduler-spi-20261027finalc` (ID `141089219`); Kaggle accepted version 13 and immediate status was `KernelWorkerStatus.RUNNING`. Same identity, no retry/cancellation.
- Candidate `experiments/parameter_finalization/runs/notebook9_20261028/submission_v13_features/`. Notebook SHA-256 `dd090b3c2876220005d2d3959b58604f35fd5cc988f2949612f63c2f118f87fe`; source ZIP SHA-256 `3636d9502ec663ca507ca62f65862f1ac9177c0b723c166eab25cf75cd31b12e`. Independent checks: 115/115 source hashes, ZIP CRC, notebook schema, canonical cells, embedded config, and pinned asset sizes/hashes pass. When cached features differ from the active runtime producer, the new gate re-extracts all 300 into a separate runtime feature path with provenance; it preserves the original cache and checks the fresh signature before detector warmup, labels, or training. Readiness gates unchanged.
- Runtime cap: 7,200 seconds / 4 aggregate GPU-hours. Last verified Kaggle CLI quota after v12: 5.79h used /24.21h remaining of 30h, refresh Oct31. Expected outputs `/kaggle/working/experiments/parameter_finalization/runs/notebook9_20261028_v13/`, recovery `/kaggle/working/run_recovery.zip`. Immediate status is RUNNING and logs are not available yet. Bounded monitor `experiments/parameter_finalization/runs/notebook9_20261028/monitor_v13.ps1`; artifacts under `monitor_v13/`. Submission receipt `experiments/parameter_finalization/runs/notebook9_20261028/submission_v13_receipt.json`.

#### v13 terminal outcome, verified recovery, and readiness stop

- Version 13 is terminal `KernelWorkerStatus.ERROR` on the existing identity; verified with `kaggle kernels status satyapaladugu/betterscheduler-spi-20261027finalc`. No remote job remains active and none was cancelled. The bounded local monitor ended after retrieving terminal artifacts.
- The executed run used Python 3.13.15, Torch 2.11.0+cu128, CUDA 12.8, and two Tesla T4 GPUs. Timed notebook cells: setup 66.04 s, model preflight 25.39 s, teacher labeling 593.55 s, training/release cell 3.51 s; cell setup/log totals are recorded in the verified `cell_evidence/cell_status.json`. This corresponds to about 688.5 s of timed cell wall time; with both GPUs allocated throughout those timed stages, the conservative upper bound is 0.382 aggregate GPU-hours. It is not a direct provider billing measurement.
- The first targeted file retrieval returned HTTP 502 text for `cell_evidence/cell_2.log` and `data_run/runtime_features/provenance.json`. A subsequent bounded fetch of `run_recovery.zip` succeeded. Archive `experiments/parameter_finalization/runs/notebook9_20261028/submission_v13_features/recovery_fetch/run_recovery.zip` is 1,942,678 bytes, SHA-256 `9ea5535e0f256316e9a412102b6a404729c3001910365e1682b37a769dcb529f`. ZIP CRC passed; all 632/632 manifest payload hashes match, with no missing, unexpected, or unsafe paths. Verified extraction: `experiments/parameter_finalization/runs/notebook9_20261028/submission_v13_features/recovery_expanded_verified/`. This restores the two 502'd files and all teacher labels/status, features/provenance, model, metrics, and cell logs. Key per-file hashes are in `experiments/parameter_finalization/runs/notebook9_20261028/v13_terminal_ops_receipt.json`.
- Runtime feature extraction completed for all 300 images and produced signature `105485efb3f14d80514f39138c3310714dfdbee850291a8f4e76551df5e5f065`; the original input feature cache was preserved. Qwen3.5 pseudo-labeling completed for all 300: Low 131, Medium 32, High 116, Critical 21. Labels are VLM pseudo-labels, not independent human urgency truth.
- Grouped selection and fit completed; selected classifier is linear C=1, class balanced. The support gate passed. Validation on 96 rows: accuracy 0.5104, balanced accuracy 0.5738, macro-F1 0.4637; per-class recall Low 0.535, Medium 0.625, High 0.421, Critical 0.714. The 0.70-per-class recall gate therefore failed for Low, Medium, and High. Main's saved decision receipt is `experiments/parameter_finalization/runs/notebook9_20261028/recovered_readiness_v13_complete/decision.json`; it rejects readiness and cache promotion. No test rows were evaluated.
- The terminal runtime error followed fitting: `ValueError: release checkpoint must be persisted inside the canonical project` from `parameters/spi_release.py` during `candidate_release`. The run output under `/kaggle/working/experiments/...` was outside the helper's canonical project root. This is a packaging/path guard bug, distinct from the independent failed recall gate. No cache was promoted. The correction is recorded in the canonical notebook source; v13 remains immutable evidence.
- `selection.json` was not emitted; the selected configuration and grouped-fold scores are in `data_run/spi_classifier/metrics/validation.json`, and the complete candidate comparison is `data_run/spi_classifier/metrics/candidates.json`. No relabeling, refit, v14 submission, or further GPU job was run after the readiness decision.
- Full log: `experiments/parameter_finalization/runs/notebook9_20261028/monitor_v13/kaggle_logs_v13_terminal.txt` (185,329 bytes, SHA-256 `ff5887e1fb62e4fd7ab7a83982a3dd1a5455b7847912d1228add79d4d094b3e6`). Receipt: `experiments/parameter_finalization/runs/notebook9_20261028/v13_terminal_ops_receipt.json`.

### Notebook9 v14 — EfficientNet-B1 comparison (submitted; terminal import failure)

- The first candidate was submitted once and accepted as version 14 of the existing private kernel `satyapaladugu/betterscheduler-spi-20261027finalc` (ID `141089219`). It terminated during package import; details and the preserved log hash are below.
- Canonical notebook: `experiments/spi_classifier_repair/notebooks/notebook9_efficientnet_b1_comparison.ipynb`. Candidate package: `experiments/spi_classifier_repair/runs/notebook9_20261028_v14/submission_v14_cnn_compare/`. Notebook SHA-256 `c62eaa6bdfa5a15b3fe7274a0cb5a4dbb584a31ac3b6c9c7f28e5e88e8608ac3`; source ZIP SHA-256 `d16dd423b13c63681d728313410dd61da90831a36879ef6b4a8ab02e76eef5ba` (327,870 B; CRC and all 91 manifested source/data hashes pass). Exact v13 pseudo-label file SHA-256 `1f5b3d78124d79c6fd43bb87ba2e7e94a0c18852bc45ca020aa550b40c7c75a5`; required input manifest SHA-256 `b633e716aa20a36d7f5581dce49bffebc260516ba5ecf78a1ec0ee83ddd92be0`.
- Purpose: compare one EfficientNet-B1 ImageNet transfer candidate against the saved frozen-CLIP-feature baseline, using the same 300 images and exact v13 labels; no VLM calls, relabeling, parameter cache, release, or runtime promotion. Two Tesla T4s are required and asserted; training uses only `cuda:0`. Keep Kaggle's installed Torch/TorchVision pair unchanged and record versions; install pinned scikit-learn 1.7.2 and nbformat 5.10.4 into the active kernel with a logged 180 s timeout and `--no-deps`, before importing scikit-learn. Outer validation remains exposed development data, evaluated once after grouped train-only early stopping and refit; no test set is opened.
- Run bounds: intended Kaggle push timeout 1,800 s and 1,500 s model-stage budget, capped at 20 epochs, maximum one aggregate GPU-hour. Expected remote run root `/kaggle/working/experiments/spi_classifier_repair/runs/notebook9_20261028_v14/`; incremental `cell_evidence/`, `efficientnet_b1/selection_history.json`, `refit_history.json`, `validation.json`, `predictions.json`, `candidate.pt`, and `/kaggle/working/run_recovery.zip`.
- Static staging validation is recorded in `experiments/spi_classifier_repair/runs/notebook9_20261028_v14/STAGING_VALIDATION.json`; package and metadata are under the candidate package directory. Rebuild/refresh the package and both notebook copies if reviewed source changes before launch. After terminal execution, retrieve and checksum the recovery archive before reporting comparison results.



#### v14 package refresh r2 — exclusive output ownership and hard caps

- This refreshed candidate supersedes the first staged package above. The previous package remains preserved at `experiments/spi_classifier_repair/runs/notebook9_20261028_v14/submission_v14_cnn_compare/`; refreshed package is `experiments/spi_classifier_repair/runs/notebook9_20261028_v14/submission_v14_cnn_compare_r2/`. Canonical notebook was refreshed at `experiments/spi_classifier_repair/notebooks/notebook9_efficientnet_b1_comparison.ipynb`.
- Comparison module snapshot SHA-256 `e046d0abb525223953156fe158a3a9e466cb4e22a7fc18852b74ebf5725f2d07`; notebook SHA-256 `e033131715f88a489e9f3e54f86f03655f64285aa0e61e3255e9264183ac8a4e`; source ZIP SHA-256 `56b621f4cac2931ad9944377fb59efb79a591ff8a0762ece76c69cd596eb9888` (328,031 B; all 91 source/data hashes and CRC pass). V13 label and input manifest hashes are unchanged.
- Source now creates its output directory exclusively before entering the failure handler and checks ownership before writing. The module rejects budgets above 1,500 seconds or 20 epochs. Notebook uses the approved 1,500-second / 20-epoch setting. Setup logs active-kernel installs for scikit-learn 1.7.2 and nbformat 5.10.4 with 180-second subprocess timeouts before importing scikit-learn; it does not change Torch/TorchVision. Input receipt uses `write_json`.
- Refreshed static validation: `experiments/spi_classifier_repair/runs/notebook9_20261028_v14/STAGING_VALIDATION_r2.json`. Notebook schema/stage AST, install order, receipt serialization, exclusive output behavior, budget bounds, embedded source hash and 91/91 source hashes pass. Still not submitted; waiting for final Medium review and main launch clearance.



#### v14 terminal package-import failure

- Kaggle accepted v14 on the existing identity, then reached `KernelWorkerStatus.ERROR`. The exact exception was `ModuleNotFoundError: No module named 'project_compat'`: importing `parameters.cloud_notebook` executes `parameters/__init__.py`, which imports root `project_compat.py`. The uploaded 91-file source archive omitted that root module. Failure occurred before cell recorder, dependency installation, input discovery, image access, model-weight download, teacher calls, or training. No recovery ZIP was produced; the output endpoint returned the kernel log only.
- Preserved log: `experiments/spi_classifier_repair/runs/notebook9_20261028_v14/monitor_v14/betterscheduler-spi-20261027finalc.log` (8,409 B, SHA-256 `d43d945504b0651720a08644282d33adf1bc4ebf321b851016bb38063d9d3af9`). Receipt: `experiments/spi_classifier_repair/runs/notebook9_20261028_v14/v14_terminal_receipt.json`. Kernel is terminal; no cancellation or retry.
- The launch log reports about 15.84 s until terminal completion. With two T4s assigned, that is a conservative upper bound of 0.0088 aggregate GPU-hours; actual provider billing was not exposed.

### Notebook9 v15 — corrected package staged, not submitted

- Corrected package: `experiments/spi_classifier_repair/runs/notebook9_20261028_v15/submission_v15_cnn_compare/`; it adds the exact root `project_compat.py` to the immutable source manifest. It preserves the v14 package and changes no production source. Existing notebook identity is unchanged; final Medium review cleared the source, and submission awaits main launch clearance.
- Canonical notebook `experiments/spi_classifier_repair/notebooks/notebook9_efficientnet_b1_comparison.ipynb` SHA-256 `a919ae42c89cd4410e728a0509f7fb0ee41d6953cbc66b30aba36d3a1840de06`; source ZIP SHA-256 `ad58e74c5703be282accc8dfa9358fea7a1e0008ef78277cc573b1f122fdadcf` (329,008 B; 92/92 manifest hashes and CRC pass). Embedded v13 labels and input manifest hashes remain pinned.
- Exact-archive isolation check passed using the project `.venv` interpreter, `python -I`, a temporary working directory, and only extracted ZIP on `sys.path`; it imported `parameters.cloud_notebook`, `parameters.dataset_pipeline`, and the packaged CNN module without invoking weights/model code. Static validation: `experiments/spi_classifier_repair/runs/notebook9_20261028_v15/STAGING_VALIDATION_v15.json`.
- Next version run cap is 1,740 s overall (`-t 1740`) and 1,500 s model-stage / 20 epochs on `cuda:0`. Including v14's 15.84-second two-device startup upper bound, the campaign bound is 0.9755 aggregate GPU-hours. Do not relabel or call the teacher; use the embedded verified v13 labels.

#### Notebook9 v15 submitted — isolated imports fixed

- Submitted once to the existing private identity `satyapaladugu/betterscheduler-spi-20261027finalc` (ID `141089219`) with Kaggle timeout 1,740 seconds. Kaggle accepted version 15; first status check at 2026-10-28T07:41:37Z was `KernelWorkerStatus.RUNNING`. No other identity/version was created.
- Submitted notebook SHA-256 `a919ae42c89cd4410e728a0509f7fb0ee41d6953cbc66b30aba36d3a1840de06`; source ZIP SHA-256 `ad58e74c5703be282accc8dfa9358fea7a1e0008ef78277cc573b1f122fdadcf`. The 92-file archive includes `project_compat.py`, and the exact archive isolation import check passed before push.
- Terminal status: `KernelWorkerStatus.COMPLETE` at 2026-10-28T07:58:57Z. The monitored CLI retrieval hit a Windows encoding error; a bounded quiet UTF-8 artifact request retrieved the run log and a range-capable fetch retrieved the recovery ZIP. Verified archive: `experiments/spi_classifier_repair/runs/notebook9_20261028_v15/monitor_v15/recovery_resumable/run_recovery.zip` (24,750,017 B, SHA-256 `9c51d2641ddbac805efeed6e5ae694bacdc7290e1b6f8fb5bf0bbf5e7e8f2634`); ZIP CRC passed and all 20/20 retrieval-manifest hashes matched. Full extracted artifacts: `experiments/spi_classifier_repair/runs/notebook9_20261028_v15/monitor_v15/recovery_resumable/extracted/`; verification receipt: `.../monitor_v15/recovery_resumable/retrieval_verify.json`.
- Consolidated terminal receipt with version, environment, input, metrics, cell timings, and SHA-256 values: `experiments/spi_classifier_repair/runs/notebook9_20261028_v15/v15_terminal_receipt.json` (SHA-256 `41e67ebf74698ef8fbf70358bf6076ce9f915a35b36bf66ef02562419b0f9ae7`). The recovery includes the executed notebook/cell evidence, 300-image input receipt, validation and selection/refit histories, predictions, plots, candidate checkpoint, and backbone identity.
- All three instrumented cells completed without runtime failure; the CNN comparison cell took 95.12 s (candidate status records 94.08 s). Environment was Python 3.13.15, Torch 2.11.0+cu128, TorchVision 0.26.0+cu128, scikit-learn 1.7.2, two Tesla T4s assigned and training on `cuda:0`. Input manifest SHA-256 `b633e716aa20a36d7f5581dce49bffebc260516ba5ecf78a1ec0ee83ddd92be0`, 300 images; embedded v13 teacher labels SHA-256 `1f5b3d78124d79c6fd43bb87ba2e7e94a0c18852bc45ca020aa550b40c7c75a`; no feature cache used and no relabeling/teacher inference performed.
- EfficientNet-B1 selected epoch 6. Exposed development validation (`n=96`): accuracy 0.5417, balanced accuracy 0.4852, macro-F1 0.4462, Brier 0.63584, log loss 1.16620, RPS 0.16640, ordinal MAE 0.67708. Per-class recalls: Low 24/43 (0.5581), Medium 3/8 (0.3750), High 22/38 (0.5789), Critical 3/7 (0.4286). There were 12 high/critical underpredictions; zero Critical-to-Low/Medium errors. `all_class_recall_gate=false` and `runtime_promotion_allowed=false`; claim is development teacher agreement only. Metrics/status: `.../extracted/efficientnet_b1/validation.json` and `status.json`; selection/refit histories, predictions, plots, and checkpoint are alongside them. Backbone weights enum `IMAGENET1K_V2`, SHA-256 `c27df63ce6eb17ef8bcea58922fd3a254cba910c720f41ee89d64d99fb7a4ddf`.
- Campaign upper bound remains 0.9755 aggregate GPU-hours including v14 startup; no further run is authorized or needed for this comparison. Preserve v15 artifacts and do not promote this candidate.

### Notebook9 full-source dataset download test — CPU-only prelaunch

- Reuses existing private identity `satyapaladugu/betterscheduler-spi-20261027finalc` (ID `141089219`); no new notebook identity. Pre-submit status check on 2026-10-28 returned `KernelWorkerStatus.COMPLETE`. Kaggle CLI 2.2.4 authenticated successfully; quota then was GPU 6.04h used / 23.96h remaining of 30h, TPU 0h used / 20h remaining, refresh `2026-10-31T00:00:00Z`.
- User-authorized purpose: attempt the complete published LADI-v2 `data/` file set plus official MEDIC and CrisisMMD archives wholly in Kaggle, with no extraction, GPU use, model calls, or laptop bulk retrieval. TREC-IS linked images are outside the full-download claim. Input remains `satyapaladugu/betterscheduler-spiv2-inputs-20261027b` at the previously verified mount; the Kaggle API inventory is 304 files / 90,251,657 bytes (300 PNGs, 3 JSON, 1 NPZ), manifest SHA-256 `b633e716aa20a36d7f5581dce49bffebc260516ba5ecf78a1ec0ee83ddd92be0`.
- Canonical notebook: `experiments/dataset_detector_grid/notebooks/notebook9_dataset_download_test.ipynb`; exact embedded source: `experiments/dataset_detector_grid/download_test.py`, SHA-256 `06f19ff2598c0de8b9d865923c933db6be82836d554449c3302ea209a79e6e30`. Staging package and static validation: `experiments/dataset_detector_grid/runs/notebook9_download_test_20261028/submission_cpu_download/`; notebook SHA-256 `4439ebe77b07ecfc1d8623cf8fc9edf44b73afa34c1bc27f52e2f57de7c3ebb8`. nbformat validation and both code-cell compiles passed. Metadata preserves same identity/input and Internet access, sets `enable_gpu=false`, removes `machine_shape`, and declares 3,600-second timeout / 0 GPU-hour cap.
- Kaggle CLI/API does not report free `/kaggle/working` space. The notebook records live free space and enforces a 3,000,000,000-byte reserve before/during downloads; local D: had 12,558,630,912 bytes free, which is not evidence of Kaggle capacity. Preflight receipt: `experiments/dataset_detector_grid/runs/ops_preflight_20261028T125709Z/receipt.json`.
- Expected cloud output: `/kaggle/working/experiments/dataset_detector_grid/runs/<UTC run id>/`, including source inventory, incremental download status, per-file SHA-256/byte counts, stdout log, provenance and summary. Expected result remains unmeasured. Do not submit unless an immediate status check confirms the existing kernel is idle/terminal; after submit, record actual returned version/status before monitoring and retrieve only small receipts/logs, never the downloaded image archives.

#### Notebook9 full-source download test — version 16 submitted

- Kaggle accepted version 16 on the existing private identity `satyapaladugu/betterscheduler-spi-20261027finalc` (ID `141089219`) with `-t 3600`; immediate `kaggle kernels status` returned `KernelWorkerStatus.RUNNING`. GPU is disabled in metadata and no accelerator was requested. No new identity or cancellation.
- Submitted notebook SHA-256 `4439ebe77b07ecfc1d8623cf8fc9edf44b73afa34c1bc27f52e2f57de7c3ebb8`; embedded exact source SHA-256 `06f19ff2598c0de8b9d865923c933db6be82836d554449c3302ea209a79e6e30`. Submission receipt: `experiments/dataset_detector_grid/runs/notebook9_download_test_20261028/submission_cpu_download/submission_receipt.json`.
- Expected outputs remain under `/kaggle/working/experiments/dataset_detector_grid/runs/<UTC run id>/`. Monitor v16 to terminal and retrieve only small status/log/summary receipts; do not download bulk dataset contents. Full success and output persistence are not established by submission or RUNNING status.
- Monitor update at approximately 2026-10-28T14:03Z (about one hour after push): explicit v16 status still `KernelWorkerStatus.RUNNING`. The normal `/16` log endpoint is empty; live log streaming returned HTTP 404. Kaggle's `/16` output index reports 135 generic existing files but no names matching this download test (`inventory.json`, `downloads.json`, `provenance.json`, `stdout.log`, summaries, or failure receipt); no output files were downloaded. GPU quota remains 6.04h used / 23.96h remaining, consistent with the CPU-only configuration. Download progress, disk free space and failure state remain unverified. Preserve this active status and do not submit a new version or cancel it.


## MEDIC-only300 detector grid — prelaunch
Supersedes cancelled full-download v16. Existing notebook9 identity, twoT4s,20YOLO/56DINO configurations, individualMEDIC300images only with1GBimage transfercap. Canonical notebook experiments/dataset_detector_grid/notebooks/notebook9_medic300_detector_grid.ipynb; stage runs/medic300_grid_submission. nbformat/codecompile/isolatedsourceimports passed. Run timeout6000s; modelworkers4800s each. No teacher/SPItraining/fullarchives. Results are development sensitivity/cost, no mAP or bestaccuracyclaim withoutboxtruth. Logs/environment/sourcehashes and per-setting rows persist incrementally under /kaggle/working/experiments/dataset_detector_grid/runs.

Kaggle accepted version17 via main-thread CLI push, timeout6000s. Next status verified separately; no success claim from submission.

Version17 ERROR beforefirstcell: papermill No kernel name found in notebook. No downloadsorinference. v18correctskernelspec python3/language_info only; preservedv17stage. Static kernelpresence/schema/codecompile checks pass.

- Ops refresh: v18 explicitly remains `KernelWorkerStatus.RUNNING` (Kaggle CLI/API authentication recovered). `kernels logs .../18` returns no execution text. The 135-file output index is stale v15 content: sampled `cell_0.log`, `cell_status.json`, `executed.ipynb`, `input_receipt.json`, and `status.json` all have remote last-modified time 2026-10-28 07:40Z. No v18 MEDIC acquisition receipt, outputs, progress, or exact failure are currently accessible; no bulk files were retrieved. Receipt: `experiments/dataset_detector_grid/runs/monitor_v18_recovery/status_receipt.json`. Continue watching exact version 18 only.


## MEDIC SPI training — prelaunch 2026-10-29 IST
Existing notebook9 identity, exactprivateMEDIC300 snapshot satyapaladugu/semanticscheduler-medic300-v18; originalrecordssha1b0feb7ec30c3d9fb033d9df8456daf37e56b5941f35e40aeb684d02e98845ae. Stage experiments/spi_classifier_repair/runs/medic_training_20261028T214102Z; source/code/kernel checks,26localtests andisolatedfullpipelineimports passed. Fourencoders36vectorpairs+6tokenpairs, Qwen9B image-onlylabels, resumableindividualstages, calibration/imageprovider/wirecheck/profile/report, gatedcache. TwoT4s,4500snotebooktimeout upper2.5aggregateGPUh. Optionalfinetuning/CORNnotexecutedinthisfrozen-featurecomparison. Noindependenturgencyclaim;readyrequiresdiagnosticreceiptandmetricgates.

#### MEDIC SPI training — version 19 monitor update (2026-10-28T21:44Z)

- Main submitted version 19 on the existing private notebook identity `satyapaladugu/betterscheduler-spi-20261027finalc` (ID `141089219`) from `experiments/spi_classifier_repair/runs/medic_training_20261028T214102Z/`. Staged notebook SHA-256 `fd05914c739291b9186a0ac436dcbd890d7af5b035d316fae2675cad3bad5cef`; source SHA-256 in validation manifest `6c5758f5a9da0b4b433e1ae6eda07557187a84b885bd493d154fadd874b0b364`. Submission acceptance was reported by main and exact-version status independently verified `KernelWorkerStatus.RUNNING` at 2026-10-28T21:44Z.
- Input ref is `satyapaladugu/semanticscheduler-medic300-v18`; records SHA-256 `1b0feb7ec30c3d9fb033d9df8456daf37e56b5941f35e40aeb684d02e98845ae`. Current dataset file listing has 303 files: 300 `images/<sha>.img`, plus root `records.json`, `input_manifest.json`, `README.md`, matching the flattened archive layout. Kaggle dataset-list metadata omits the current version number; v1 is preserved, and the current listing timestamp/layout matches the intended v2 update. Private status was verified when v1 was created; no visibility change was requested.
- At monitor time, `kaggle kernels logs .../19` returned an empty response and no run artifacts were visible. This does not establish cell start or training progress. No output files retrieved. Next action: monitor version 19 only and retrieve compact logs/receipts/metrics after they appear; preserve partial outputs on terminal failure.


## Version19 failure and resumable version20 correction
V19ERROR after~959s,24/42candidatescomplete: CLIP+ConvNeXtloadinghitcustomnotebookTee missingisatty. Sharedartifacts.Teealreadycompatible; modelrecipeunchanged. Mainfixedonlynotebookwrapper, exactstagedregression+compile/schema pass, sourceZIPidentical19. V20stage experiments/spi_classifier_repair/runs/medic_training_streamfix_20261028T220317Z. Exactresume inputprivate satyapaladugu/semanticscheduler-spi-resume-v19 READY,checkpoint.bin316883530B sha4532dbeaf9f4ec7a3bd98da4842f6ec0c98fc0591cba5dddc260f78223933a5b,1395members. Fullv19archive875993157B shaa8b803ab0ba44323b391ecc3a898f5c1678655b367d129c7cd9e04fe914780c9 verified1557members. Redundantfeaturefilesomittedonlyafterreceipt/hashproof. Labels/source/modelcacheidentitiespreserved; no relabelingrequiredifruntimeversionsmatch.

#### MEDIC SPI training — version 19 terminal and resume input (verified 2026-10-28T22:16Z)

- Version 19 on `satyapaladugu/betterscheduler-spi-20261027finalc` terminated `ERROR` at 2026-10-28T21:58:53Z after about 959 seconds in the campaign cell. Setup, dependency installation, and source verification passed on two Tesla T4s. Loading `clip_b32` and `convnextv2_tiny` failed because Transformers 5.5.0 calls `sys.stdout.isatty()` and the notebook's Tee did not implement it. The failure receipts show 24/42 classifier candidates complete and 18 missing; the test gate stayed closed. This is a stream-wrapper fault, not evidence of bad model weights or a completed comparison. Evidence: `experiments/spi_classifier_repair/runs/monitor_v19/kaggle_output_v19/experiments/spi_classifier_repair/runs/medic_spi_20261028T214226Z/campaign/comparison/failures.json`, `coverage.json`, `failure.txt`, and `experiments/spi_classifier_repair/runs/monitor_v19/recovery_inspection/betterscheduler-spi-20261027finalc.log`.
- The original v19 `spi_recovery.zip` was retrieved after matching the source manifest and output log to run `medic_spi_20261028T214226Z`: 875,993,157 bytes, SHA-256 `a8b803ab0ba44323b391ecc3a898f5c1678655b367d129c7cd9e04fe914780c9`. All 1,557 members (875,539,407 uncompressed bytes) matched manifest sizes and SHA-256. The original archive remains at `experiments/spi_classifier_repair/runs/monitor_v19/recovery_inspection/spi_recovery.zip`.
- Curated `checkpoint.bin` is under `experiments/spi_classifier_repair/data/semanticscheduler-spi-resume-v19/`; it contains 1,395 verified `campaign/...` entries and no images. It retains label/raw/candidate caches and completed all-feature data, receipts, and identities. We omitted 152 duplicate DINO/SigLIP feature shards and two duplicate output `features.npz` files only after complete aggregate receipts and matching hashes. Eight non-campaign runtime files remain in the unchanged original archive and are excluded from the resume ZIP because restore expects the `campaign/` prefix. Curated ZIP size 316,883,530 bytes; SHA-256 `4532dbeaf9f4ec7a3bd98da4842f6ec0c98fc0591cba5dddc260f78223933a5b`.
- Canonical MEDIC records SHA-256 is `1b0feb7ec30c3d9fb033d9df8456daf37e56b5941f35e40aeb684d02e98845ae`, confirmed by v19 `input_receipt.json`; the run's serialized `data/records.json` has changed runtime paths/schema and is intentionally excluded. The private MEDIC 300 dataset remains attached separately.
- Private Kaggle managed input is ready: `satyapaladugu/semanticscheduler-spi-resume-v19`, dataset ID 12932005, version 1, `isPrivate=true`, license CC-BY-NC-SA-4.0. It has five root files: `checkpoint.bin`, `resume_index.json`, `resume_manifest.json`, `source_recovery_manifest.json`, and `README.md`; the `.bin` archive is not auto-expanded. Mount path: `/kaggle/input/semanticscheduler-spi-resume-v19/checkpoint.bin`. Downloaded remote `resume_index.json` SHA-256 matches the local copy; it contains the archive and records hashes above.

#### MEDIC SPI training — version 20 submitted and monitoring

- Main submitted v20 on the same notebook identity from `experiments/spi_classifier_repair/runs/medic_training_streamfix_20261028T220317Z/`. Notebook SHA-256 `0e0ecd835562f9bc64867c437628666fb6c1523c5dd560148cf9379ca1554fc4`; intended embedded source fingerprint `6c5758f5a9da0b4b433e1ae6eda07557187a84b885bd493d154fadd874b0b364`. Stage records only a Tee logging-wrapper fix and says the source ZIP is byte-identical to v19. Metadata attaches the canonical MEDIC 300 input and private resume input; runtime cap 3,000 seconds on two T4s. Main reports acceptance; exact `/20` status check confirmed `KernelWorkerStatus.RUNNING` at 2026-10-28T22:15Z. Logs were empty at that initial poll, so no cache restores, reused labels, completed encoders, or model keys are claimed yet.
- Ops monitor polls only v20 every 60 seconds with a 60-minute local cap. Status and changed log snapshots go to `experiments/spi_classifier_repair/runs/monitor_v20/`. Record actual runtime run ID, recovery/cache hits, 300-label reuse, unexpected model-loading keys, terminal state, and verified outputs when observed. Do not count v19 outputs as v20.

#### MEDIC SPI training — version 20 terminal results (verified 2026-10-28T22:28Z)

- Exact notebook version 20 terminated `COMPLETE`; runtime run ID `medic_spi_20261028T221515Z`, elapsed 335.94s, source fingerprint verified `6c5758f5a9da0b4b433e1ae6eda07557187a84b885bd493d154fadd874b0b364`, Python 3.13.15, Torch 2.11.0+cu128, two Tesla T4s. The run restored `/kaggle/input/semanticscheduler-spi-resume-v19/checkpoint.bin`. The local PowerShell monitor could not print a long Unicode log under Windows `charmap`; direct UTF-8 Kaggle API retrieval verified and preserved the full log.
- `campaign/labeling/teacher_identity.json` says `all_labels_reused=true`. `labels.json` contains 300 rows and is byte-identical by SHA to v19; all 600 image-label checkpoint JSON files and 600 raw request/response files also match v19 path/hash exactly. `campaign/splits.json` reports 180/60/60 train/validation/test, seed 17, parent/exact/dHash groups, known-exposure exclusion true; event independence remains unverified. The run iterated over 300 label rows in logs, but the persisted label/cache artifacts are unchanged from v19.
- All 42/42 classifier candidates completed. Hash comparison against the v19 recovery manifest confirms all 18 cached vector candidates and all 6 token candidates were reused unchanged; v20 added the remaining 18 vector candidates. The selected development pipeline is CLIP-B/32 + random forest: macro-F1 0.592, accuracy 0.817, balanced accuracy 0.563, worst-class recall 0. Development and held-out values measure agreement with teacher labels only.
- The 60-row held-out result is macro-F1 0.398, accuracy 0.783, balanced accuracy 0.400, worst-class recall 0; 10 High/Critical underpredictions, including 2 Critical-to-Low/Medium. Readiness blocks runtime promotion: all-class support false, per-class recall false, independent diagnostic review false; export parity and runtime integration passed. Validation Critical support is 2 (train 26); final-test Critical support is 3. `campaign/final_parameter_cache/status.json` is `not_started_readiness_failed`, with zero records. No independent human-urgency validity is established.
- CUDA profile p50/p95 is 82.26/104.67 ms and CPU p50/p95 is 533.06/565.74 ms for the complete SPI image encoder/head + CLIP relevance + YOLO scene measurements + header serialization, batch 1, 10 warmups. Wire checks passed on 10 fixture cases with 35-byte headers; this remains a fixture probe and does not change runtime promotion.
- `campaign/comparison/failures.json` is stale: its SHA-256 is identical to v19’s two Tee/isatty failures while v20 coverage is complete. Treat this file as inherited historical evidence, not current v20 failures. Main has a local fix for future runs; v20 remains unchanged.
- Model-load logs show repeated CLIPModel `UNEXPECTED` keys `vision_model.embeddings.position_ids` and `text_model.embeddings.position_ids`, plus ConvNextV2Model keys `classifier.bias` and `classifier.weight`; no model-load exception occurred. Full exact reports are in `experiments/spi_classifier_repair/runs/monitor_v20/stderr_v20.log` for review.
- Retrieved and SHA-verified 51 compact files (5,001,318 bytes) against the v20 recovery manifest, including reports, split/records/label receipts, readiness/development/held-out metrics, coverage, calibration, profiles, curves, and the selected 330,873-byte `campaign/release/bundle/model.joblib`. No image files, feature arrays, or `spi_recovery.zip` were retrieved. The v20 output index has no executed `.ipynb`; no executed notebook copy was available.
- Durable local evidence: `experiments/spi_classifier_repair/runs/monitor_v20/run_summary_v20.json`, `terminal_receipt.json`, `artifact_manifest_v20.json`, `logs_raw_v20.json`, `stdout_v20.log`, `stderr_v20.log`, and compact outputs under `experiments/spi_classifier_repair/runs/monitor_v20/compact_outputs_v20/`. Main owns any follow-up run or notebook version.

- Medium assessment of v20 artifacts confirms 300 unique IDs and disjoint split groups; the fresh test excludes known development exposure, but all 60 test images still have disclosed prior detector-study exposure. The frozen pipeline was evaluated once with no automatic retuning; the holdout is now exposed. The unexpected CLIP `position_ids` and ConvNeXt `classifier.weight`/`classifier.bias` keys are expected buffers/task-head weights for the loaded backbones, not missing backbone weights. Assessment only; no edits or new execution.

## Full MEDIC labeling — bounded 12-image preflight v1

- New user-approved notebook identity: `satyapaladugu/medic-full-spi-labeling` (private). Stage: `experiments/full_medic_labeling/runs/preflight_v1/`; notebook `notebook.ipynb`, SHA-256 `7a0304e5413040dae55c80354c5c038190c53ca1e4d2cb368e17929e9bfc608b`. Metadata enables internet and one Tesla T4 GPU. Local nbformat validation and Python-cell compilation passed; no local GPU execution. Account authentication works; prior exact-slug status query was not found/denied, so this is intended as the initial version.
- Purpose and bound: process 12 real MEDIC images only, with Qwen3.5-9B NF4 on GPU 1 (thinking disabled, evidence-JSON output), CLIP and YOLO on GPU 0; save labels and embeddings incrementally. This is a prompt/runtime/cache preflight only; it does not establish full-corpus coverage or training readiness.
- Planned push: `kaggle kernels push -p experiments/full_medic_labeling/runs/preflight_v1 -t 1800`; verify returned version and actual run state before monitoring. Expected runtime ceiling is 1,800 s on one T4 (at most 0.5 aggregate GPU-hours). Capture status/logs and compact 12-row outputs under `experiments/full_medic_labeling/runs/monitor_preflight_v1/`; do not retrieve images or model weights. This entry is prelaunch; version/run ID/status/artifacts remain pending until verified.
- Launch verified: Kaggle accepted notebook version 1 at `https://www.kaggle.com/code/satyapaladugu/medic-full-spi-labeling`; immediate status was `KernelWorkerStatus.RUNNING`. No runtime run ID or cell progress was exposed at first poll; initial logs contain only the CLI version warning. Bounded monitor artifacts are in `experiments/full_medic_labeling/runs/monitor_preflight_v1/` (`status_history.jsonl`, timestamped changed-log snapshots, and monitor script); final state and compact outputs pending.

- Preflight v1 terminal verification: version 1 is `KernelWorkerStatus.COMPLETE`; runtime path identifies `preflight_20261028T234518Z`, log elapsed upper bound 395.56 s. Setup log reports Python 3.13.15 and Torch 2.11.0+cu128; the notebook asserted two CUDA devices and attached machine metadata was Tesla T4. All 12/12 VLM responses were complete, valid JSON; observed labels were High 10 and Low 2. Summary explicitly records `full_dataset_labeled=false`, `training_started=false`, and FLOPs uninstrumented. This validates only the small prompt/model/device/cache preflight, not full-corpus labeling, label validity, or training readiness. Downloaded only `labels.json` (52,001 B) and `summary.json` (217 B); did not download `clip_embeddings.npy`, YOLO weights, or images. Exact log, receipt and compact outputs are under `experiments/full_medic_labeling/runs/monitor_preflight_v1/`. Initial PowerShell log capture hit a Windows `charmap` checkmark encoding error; terminal state was rechecked and the log retrieved via UTF-8. No retry or new notebook version was submitted.

#### Full MEDIC labeling — full-corpus version 2 launched

- Main staged and submitted version 2 on the approved new identity `satyapaladugu/medic-full-spi-labeling`; stage `experiments/full_medic_labeling/runs/main_v2/`, notebook SHA-256 `ADC3B1267FDBDAA28156246C4474F9108B5560FA8FD0E02A9B3A80FD5B75B759`. Kaggle accepted version 2; exact version status checked as `KernelWorkerStatus.RUNNING`. Metadata assigns a Tesla T4, GPU, and internet. Notebook timeout is 12,000 seconds (maximum 6.67 aggregate GPU-hours); inner labeling budget is 11,200 seconds with checkpoint reserve. The run targets the pinned 71,198-row corpus, shardwise streaming with incremental labels/embeddings and managed checkpoint resume. This stage does not integrate classifier training; no training claim.
- Monitor only version 2. Store status/log snapshots and compact version-specific outputs in `experiments/full_medic_labeling/runs/monitor_v2/`; record actual runtime path/counts before making coverage claims. On error preserve evidence and escalate to main; no blind retry. Outputs are expected but not yet verified. Do not retrieve images or model weights.
- Stop requested on 2026-10-29 after the revised seven-hour campaign ceiling made the current 12,000-second two-T4 run unacceptable. Stop limitation: installed Kaggle CLI has no `kernels stop` command; the SDK cancel method requires a numeric `kernel_session_id`, which the accessible status/log/list endpoints did not expose. Latest exact v2 status remained `RUNNING`, logs 0 B. Do not claim stopped. Kaggle UI session link for immediate manual Stop: `https://www.kaggle.com/code/satyapaladugu/medic-full-spi-labeling/2`. Ops continues bounded status checks and will preserve terminal evidence. No replacement job submitted.
- Cancellation update: the latest log snapshot for v2 identifies runtime run path `/kaggle/working/experiments/full_medic_labeling/runs/main_20261029T004955Z`. It reports one `dev-00000-of-00004` shard downloaded (364,358,020 B) and `MAIN PROGRESS 50 new images seconds 609.0`; no complete-corpus claim. At 2026-10-29T01:08Z the exact status became `KernelWorkerStatus.CANCEL_ACKNOWLEDGED`; monitor remains active until a final terminal state is confirmed. At this check the output listing had no downloadable files and the live log endpoint returned 0 B; preserved the latest nonempty snapshot locally. No images/embeddings/model weights were retrieved.

#### CrisisMMD 1,000-image run — version 3 accepted under verified slug

- Main submitted `experiments/full_medic_labeling/runs/crisismmd1000_v3/`; Kaggle version 3 status is `KernelWorkerStatus.RUNNING`. Local submitted notebook SHA-256 `A13863DB6FC898927138B877C2F9EABC7A9950828DA293F6D77DB2A59FF21B58`. Kaggle account listing resolves the notebook identity to `satyapaladugu/crisismmd1000-spi-labeling` (title `CrisisMMD1000 SPI Labeling`); `/3` is RUNNING at this exact slug. The earlier metadata slug `medic-full-spi-labeling` no longer appears in the owned listing. The submitted metadata is preserved as `kernel-metadata-v3-submitted.json`; working `kernel-metadata.json` is updated to the verified slug for subsequent maintenance.
- This run targets the 1,000-row CrisisMMD fixed selection only. Notebook scope explicitly excludes classifier training. Do not describe models as training; monitor actual labels/cache progress and preserve compact outputs only.
- v2 cancellation is not yet verified terminal: it previously showed `CANCEL_ACKNOWLEDGED` under the prior slug, but after the notebook identity re-slugged, exact status `/2` under the verified new slug reports RUNNING, as does `/3`. This overlap is under investigation; retain both version/run records separately and do not claim v2 fully stopped.
- Monitor `/3` only in `experiments/full_medic_labeling/runs/monitor_v3/`. Outputs/artifacts pending verified discovery; no image, feature-array, or weight downloads.
- Clarification: Kaggle status queries for version labels `/2` and `/3` both returned `RUNNING` after the notebook slug changed; these version-label queries may alias the current kernel session. This is not evidence of two simultaneous runs. Ops monitors only verified `/3`; v2 cancellation acknowledgment from the earlier log is retained, with no inference of a parallel active v2 process.
- Live log limitation: `kaggle kernels logs -f satyapaladugu/crisismmd1000-spi-labeling/3` returned HTTP 404 from the CLI stream endpoint. Standard logs currently expose only the local CLI version warning; rely on exact status checks until Kaggle publishes the run log/output inventory.
- A bounded `kernels output` probe requesting only `summary.json` and `labels.json` returned no files while v3 is running; `monitor_v3/output_probe/` is empty. No output artifact or image/feature/weight was retrieved.

#### CrisisMMD 1,000-image v4 initial CLI submission attempt — superseded

- Authorized target remains the existing private identity `satyapaladugu/crisismmd1000-spi-labeling`; v3 was checked before submission and reported `KernelWorkerStatus.COMPLETE`. The v4 stage is `experiments/full_medic_labeling/runs/crisismmd1000_v4_complete/`, with metadata still naming that same identity and enabling GPU/internet on `NvidiaTeslaT4`. Staged notebook SHA-256 is `FDAE8867E9FB92282FADAD26F5FD7513789A198F37780AD74324DEF79F7DA901` (5,356,458 B). The checkout already contained unrelated working-tree edits, which were preserved.
- Two authorized `kaggle kernels push -p <stage> -t 24000` attempts failed locally with `ConnectionAbortedError(10053)` and exit code 1. Logs: `experiments/full_medic_labeling/runs/crisismmd1000_v4_complete/submission_v4.log` and `submission_v4_retry.log`. No successful push response or new version number was returned.
- After both attempts, Kaggle status for the base identity remained `KernelWorkerStatus.COMPLETE`. The exact `/4` status query also reported COMPLETE, but must not be treated as proof of a v4 run: the post-retry pull of the account's current source returned the earlier 654,492 B notebook, SHA-256 `A45915C976D4C8BCBC43D3F314BD950A56E161627E8DECD823DB9F12259EB0F2`, which differs from staged v4. This indicates the current published source is still the earlier version; Kaggle did not confirm a new v4 submission. Remote verification artifacts and logs are under `experiments/full_medic_labeling/runs/crisismmd1000_v4_complete/verify_remote_v4/`.
- No v4 execution, run ID, outputs, or persistence/retrieval were verified. Do not infer runtime completion or start another retry from the ambiguous `/4` status. Main must decide next action; no additional push was attempted.

#### CrisisMMD 1,000-image v4 — accepted, terminal error

- Existing private identity: `satyapaladugu/crisismmd1000-spi-labeling`; accepted Kaggle version 4 at 2026-10-29 06:40:06 UTC, kernel ID `141416127`. Version link: https://www.kaggle.com/code/satyapaladugu/crisismmd1000-spi-labeling/4. Immediate base and exact `/4` status checks both returned `KernelWorkerStatus.RUNNING`; notebook listing lastRunTime was 2026-10-29 06:40:06.317.
- Staged source: `experiments/full_medic_labeling/runs/crisismmd1000_v4_complete/notebook.ipynb`, 662,368 bytes, 26 cells, SHA-256 `CDA913B1C1A90E933D244F353EB12C7571FACB3759ECEF6D18DE51D2537F4AEC`. Kaggle SDK `KaggleApi.kernels_push(stage, timeout="24000")` returned version 4. Source receipt: `submission_v4_source_provenance.json`; accepted-version receipt: `submission_v4_sdk_success.json`. Local source revision was `032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0`; the checkout had unrelated working-tree edits. Kaggle source pull previously returned HTTP 403, so byte-for-byte re-download is unavailable; acceptance response associates this source hash with version 4.
- Purpose: run the fixed CrisisMMD 1,000-image labeling pipeline with full teacher cost/FLOP profiling; preserve compact ID/hash/class receipts while prior full raw records remain under v3. Metadata declares a private notebook, internet enabled, GPU enabled, `NvidiaTeslaT4`; attached dataset, kernel, and model source lists are empty. The runtime's actual device and data-source behavior remain to be confirmed from setup logs.
- Execution limit: server timeout 24,000 seconds. Exact `/4` status changed from RUNNING to `KernelWorkerStatus.ERROR` at 2026-10-29 06:43:33 UTC; no run ID is exposed by the available status/list commands. Runtime path is `/kaggle/working/experiments/full_medic_labeling/runs/main_20261029T064052Z`. Setup log reports Python 3.13.15, Torch 2.11.0+cu128, and CUDA device count 2. The verified 1,000-label cache was restored and the GPU1 teacher load was skipped, so two visible devices are not evidence of dual-GPU teacher execution. The new teacher cost profile did not run.
- First two CLI pushes failed with host-side WinError 10053; a later CLI attempt returned HTTP 400 at `SaveKernel`. After the notebook was reduced to the current size, the single SDK submission succeeded. Prior failure evidence remains in `submission_v4.log`, `submission_v4_retry.log`, and `submission_v4_20261029T063508Z.log`; the successful SDK receipt supersedes the earlier not-accepted state.
- Failure: notebook cell `In [24]` failed in `crisis_study.prepare` while downloading a pinned CrisisMMD image from Hugging Face; the redirected `urllib` request returned HTTP 503 Service Unavailable after about 179 seconds. The exact image URL is not included in the log. This is a remote asset-fetch failure; no automatic retry or source fix was made. Earlier pip dependency warnings did not prevent setup completion.
- Persistence/retrieval: Kaggle output retrieval returned only `kaggle_v4_output/crisismmd1000-spi-labeling.log` (28,866 B, SHA-256 `4F69D261751EB16E323798E2ECB18C26C83AE19203129B3175AFA9D42A212CAC`). No labels, metrics, status JSON, or cost-profile artifact was published/retrieved. The full captured log is `kaggle_v4_logs_20261029T064333Z.txt` (28,868 B); status history and monitor receipt are `kaggle_v4_status_history.jsonl` and `kaggle_v4_monitor_receipt.json`. No raw images, feature arrays, or model weights were retrieved.

#### CrisisMMD 1,000-image v5 — accepted and running

- Following v4's Hugging Face image-fetch HTTP 503 after 226 verified image downloads, the staged downloader was updated by the main thread with five bounded retries, exponential backoff, and cache-busting for CDN redirects. The existing private identity was reused; v4 logs and outputs were preserved.
- Kaggle SDK accepted version 5 at 2026-10-29 11:47:32 UTC for `satyapaladugu/crisismmd1000-spi-labeling` (kernel ID `141416127`), timeout 24,000 seconds. Exact `/5` status verified `KernelWorkerStatus.RUNNING`; account listing lastRunTime was 2026-10-29 11:47:34.247. Link: https://www.kaggle.com/code/satyapaladugu/crisismmd1000-spi-labeling/5.
- Stage: `experiments/full_medic_labeling/runs/crisismmd1000_v4_complete/`; notebook is 663,009 bytes, 26 cells, SHA-256 `BB9AFBD6795EFDCC54C20239B43B7108DDBEB62694A2206D748E77B0019682DF`. Source and acceptance receipts: `submission_v5_source_provenance.json` and `submission_v5_sdk_success.json`. Metadata targets the same private identity, Tesla T4 GPU, and internet.
- Runtime setup/device state, cell logs, cost-profile outputs, completion state, and output persistence are pending. No v5 run ID or runtime artifacts have yet been verified. Monitor only `/5`; do not infer completion from accepted submission.

#### CrisisMMD 1,000-image v5 terminal outcome and v6 launch

- Version 5 terminated with `KernelWorkerStatus.ERROR`. The retrieved `class_weighted` failure receipt contains 32 `FileNotFoundError` records: the output directory for each encoder comparison was missing when the shared feature loader attempted to write. This is the inspected subset of failure evidence; other balance-mode failures have not yet been inspected. Preserve v5 outputs under their existing version-specific artifacts.
- Main corrected the notebook to create each encoder output directory before comparison. The staged v6 notebook is `experiments/full_medic_labeling/runs/crisismmd1000_v4_complete/notebook.ipynb`, 663,072 bytes, 26 cells, SHA-256 `8AD229ADB25034DDB0673EC5A26D88C8AD8E33CF5A3D714BB106E3FF3DF97C85`; 5 focused tests passed before submission (per main's report).
- Kaggle SDK accepted version 6 of the existing private identity at 2026-10-29 15:45:20 UTC (kernel ID `141416127`), with timeout 24,000 seconds. Exact `/6` status verified `KernelWorkerStatus.RUNNING`; account listing lastRunTime was 2026-10-29 15:45:22.190. Link: https://www.kaggle.com/code/satyapaladugu/crisismmd1000-spi-labeling/6.
- Source/acceptance receipts: `submission_v6_source_provenance.json` and `submission_v6_sdk_success.json`. Version-specific output retrieval is under `experiments/full_medic_labeling/runs/v6_results/`; the consolidated `artifact_manifest.json` records 399 files and 1,343,810,542 actual bytes with SHA-256 checksums, below the 5 GiB retrieval cap. It includes the executed `kaggle_export_v6/__notebook__.ipynb` (26 cells; 24 executed code cells, 20 with captured outputs), `__results__.html`, labels and SQLite cache, actual records/split protocol/exclusions, four feature banks, 188 direct classifiers, 48 token-head learning-curve JSONs, 96 complete-pipeline profiles, and per-balance metrics/failures. Images, raw encoder weights, and duplicate checkpoint trees were excluded. The executed notebook export preserves code cells, but Kaggle GetKernel returned HTTP 403 for the exact accepted upload-source bytes; no byte-identical source file was recovered. `scheduler_handoff/image_records.jsonl` now contains v6 split/group fields joined by image ID plus SHA-256 for 983 rows (462 train, 224 validation, 297 test); 17 unmatched rows remain null with explicit status. The pre-join file and receipt are preserved and checksummed. The current `/6` status alias may reflect v7, so no historical v6 status is inferred from it. Do not infer cache resume beyond the retrieved matching artifacts.

#### CrisisMMD 4,000-image campaign v7 — partial checkpoint, terminal ERROR

- This authorized campaign preserves the 1,000 prior seed labels and targets 3,000 new teacher labels. It uses GPU1 for VLM and GPU0 for CLIP/YOLO, then the four encoder families, classifier heads, balancing, and cost profiling. Labeling is checkpointed every 25 new images. The 19,000-second label budget stops before training if fewer than 4,000 identity-matched rows exist. This is a resumable session plan, not evidence that the whole campaign has completed.
- First push attempt was rejected with HTTP 400 because the 1,294,729-byte notebook exceeded Kaggle's documented-in-response source-size limit of 1 MB; error receipt `experiments/full_medic_labeling/runs/crisismmd4000/submission_v7_sdk_error.json`. Main reduced the staged source to 798,997 bytes, retaining tuple seed labels and runtime identity dictionaries; no data/artifact source was attached for this fresh session.
- Kaggle SDK then accepted version 7 of the existing private identity at 2026-10-29 17:39:22 UTC (kernel ID `141416127`), timeout 24,000 seconds. Exact `/7` status verified `KernelWorkerStatus.RUNNING`; account listing lastRunTime was 2026-10-29 17:39:25.070. Link: https://www.kaggle.com/code/satyapaladugu/crisismmd1000-spi-labeling/7.
- Submitted notebook: `experiments/full_medic_labeling/runs/crisismmd4000/notebook.ipynb`, 798,997 bytes, 26 cells, SHA-256 `FBDA2E0A7292E6263FAC5A150B0ABDE00B9D33B6650F6979E9FB7C35D0F82E2A`. Acceptance/source receipts: `submission_v7_retry_sdk_success.json`, `submission_v7_retry_source_provenance.json`. The exact `/7` terminal evidence follows below.
- Bounded monitor: `experiments/full_medic_labeling/runs/crisismmd4000/monitor_v7.py`; PID 16624 reached terminal `/7` status `ERROR` on 2026-10-29 23:01 UTC and exited after saving the terminal receipt, log, identity, and progress. Progress: 2,540/4,000 rows, 1,540 new this session, 19,000.65 s elapsed, `training_started=false`. The captured traceback is the intended `SystemExit: Checkpoint complete; more labeling remains` at the 19,000-second label budget, not a model-training failure. Preserved evidence: `kaggle_v7_monitor_receipt.json`, `kaggle_v7_logs_20261029T230113Z.txt`, and `kaggle_v7_progress_identity/`; no labels, SQLite DB, images, embeddings, checkpoints, or weights were retrieved by the monitor.

#### CrisisMMD 4,000-image thinking-enabled continuation v8 — cancellation acknowledged

- v8 continues from the v7 integrity-checked label cache. It restores the 2,540 matching label receipts inline by ID/hash/class/availability, then targets the remaining 1,460 images with four-class prompt protocol v3 and thinking enabled; the prior v1 labels remain a distinct protocol cohort. GPU1 runs Qwen with 2,048-token and 120-second call caps; GPU0 runs CLIP/YOLO. The four encoder families, 48 balance/head comparisons, and profiling start only after all 4,000 rows are available. The new session stops at its 19,000-second labeling cap if coverage is incomplete.
- Kaggle SDK accepted version 8 on the existing private identity `satyapaladugu/crisismmd1000-spi-labeling` (kernel ID `141416127`), timeout 24,000 seconds. Exact `/8` initially verified `RUNNING`, then user-confirmed cancellation was verified as `CANCEL_ACKNOWLEDGED`; link: https://www.kaggle.com/code/satyapaladugu/crisismmd1000-spi-labeling/8. No new notebook identity or managed input source was attached.
- Source: `experiments/full_medic_labeling/runs/crisismmd4000_thinking/notebook.ipynb`, 903,803 bytes, 26 cells, SHA-256 `21B36FD98FE41FC44D6CB9D48E2590109B9FD9CBA4AF500D17B3D44AC71BD92A`. Receipts: `experiments/full_medic_labeling/runs/crisismmd4000_thinking_ops/submission_v8_source_provenance.json` and `submission_v8_sdk_success.json`.
- v8 cancellation snapshot: `progress.json` reports 2,565/4,000 rows, 25 processed this session, 2,769.09 seconds elapsed, and no training. Eight successful labels from this v3-thinking protocol remain archived remotely and were not reused in v9. Minimal identity/progress retrieval and hashes are recorded in `v8_cancelled_progress_identity_receipt.json`; local monitor PID 5048 was stopped after `CANCEL_ACKNOWLEDGED` and its stop receipt is `monitor_v8_stop_receipt.json`.

#### CrisisMMD 4,000-image Qwen2.5-VL continuation v9 — accepted and running

- User-selected model: pinned `Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, no thinking, same v3 rubric. The model-specific Qwen2_5_VL processor/class uses NF4 on GPU1 with 384-token and 45-second call caps; CLIP/YOLO run on GPU0. v9 restores 2,540 integrity-checked prior label receipts and retries all remaining 1,460 records, including failures from v8; the eight successful v8 labels are archived but not reused. A 10-real-image validity gate requires at least 8 valid labels before bulk labeling. Four encoders, all heads/balance comparisons, and cost profiling start only after all 4,000 rows are valid.
- Kaggle SDK accepted version 9 on the same private identity `satyapaladugu/crisismmd1000-spi-labeling` (kernel ID `141416127`), timeout 24,000 seconds. Exact `/9` terminal status is `KernelWorkerStatus.ERROR`; link: https://www.kaggle.com/code/satyapaladugu/crisismmd1000-spi-labeling/9.
- Source: `experiments/full_medic_labeling/runs/crisismmd4000_qwen25/notebook.ipynb`, 901,968 bytes, 26 cells, SHA-256 `C44C2D075B23900F784C6B1CDD363CA35E57448EA78A1213819C281AFA704C4D`. Receipts: `experiments/full_medic_labeling/runs/crisismmd4000_qwen25_ops/submission_v9_source_provenance.json` and `submission_v9_sdk_success.json`.
- The original persisted monitor became stale after its first RUNNING record. A direct Windows SDK query later confirmed `/9` ERROR. The full sanitized SDK kernel log is preserved at `experiments/full_medic_labeling/runs/crisismmd4000_qwen25_ops/v9_compact/full_kernel_log.txt` (3,424,266 bytes; SHA-256 `1F769C60B5FB2A72C5CB9014A46F18D2A3FBEF0F0D1DD5817C76D8C6D26D347`).
- Terminal evidence: label `progress.json` is 4,000/4,000, with 1,460 new rows this session, 13,361.27 s elapsed, and `training_started=false`. Label distribution: Low 1,474, Medium 688, High 1,669, Critical 102, unavailable 67; the 2,540 v1 checkpoint labels and 1,460 v3 Qwen2.5 labels remain distinguishable. `comparison/support.json` reports 2,068 train, 1,037 validation, 828 test available rows.
- All four balance matrices failed completeness. `original`, `random_oversampling`, and `moderate_undersampling` completed 36/48 pairs; `class_weighted` completed 33/44 (KNN omitted by design). Each failed on SigLIP2 feature extraction: `ValueError: image has no unpadded token centers; unsupported aspect ratio` at `medic_campaign.py:206`, reached through the crisis feature loader. The remaining SigLIP2 pairs are consequently missing. Partial candidate metrics are preserved, but final pair selection, locked-test evaluation, and full-pipeline profiling were not reached. Retrieved support, study status, coverage, candidate, failure, records, split, exclusions, labels and progress files are indexed by local `v9_compact/artifact_manifest.json` (28 files, 17,486,407 bytes; no images or weights).




- Per-image label preservation: version-9 `labels.jsonl` and selected cohort were joined on all 4,000 unique IDs; 3,933 available IDs join exactly to comparison records with zero hash or level mismatches. Raw 4,000-row receipts (including 67 unavailable) and separate 3,933-row usable export with provenance-only split references are under `experiments/full_medic_labeling/runs/crisismmd4000_qwen25_ops/v9_compact/per_image_labels/`; checksums/counts are in `label_export_manifest.json`. Split references are not runtime feature inputs.
- v9 has no frozen or promoted model because all four balance matrices failed coverage. The available prior selected artifact remains the local v6 SigLIP2 base patch16 + class-weighted RBF-SVM at `experiments/full_medic_labeling/runs/v6_results/experiments/full_medic_labeling/runs/main_20261029T154606Z/comparison/class_weighted/siglip2_b16/rbf_svm.joblib`; its test Medium recall was zero, so it is development-only. Summary: `experiments/full_medic_labeling/runs/crisismmd4000_qwen25_ops/v9_compact/model_availability_summary.json`.


## Prepared: class-specific scene-array cache v1 (not submitted)
Localnotebook `experiments/scene_array_cache/Scene_Array_Caching.ipynb`; SHA256 `39c0dae00701e6a5d7399be5604539b8cb86a3dc7edaa6988831c4ac66b54b57`; bytes 647103. Preparedmetadatareuses `satyapaladugu/betterscheduler-spi-20261027finalc` (establishedparameter/detectorstudyidentity), no newnotebookidentity. No remoteversion/runID assigned. ActiveSPIidentityisnotchanged. Expectedruntime `/kaggle/working/experiments/scene_array_cache/runs/<run-id>/`; retrieveinto matchinglocalexperimentruns. Inputs existing4000CrisisMMDselection pinned10a5626ba112ab9c50369c473a2062bcf205d708; noVLMlabelsusedbydetector. Expectedartifacts SQLite,sceneJSONL,schedulerNPZ,registry/producer/input/source/environment identities, profiles, logs, validation, figures andchecksums. InitialrealrunmustverifyassignedGPU, tenimagepreflight andactualartifactretrieval. Sourcebuildreads canonicalmodules; environmentpins innotebook. Resume `SCENE_ARRAY_RESUME_DIR` requiresexactcacheidentity; no bulkimages/weightsretrievaltolaptopdefault. No cloudcost/performanceclaimedfromlocalfixturevalidation.


### Prepared scene cache superseded by vocabularyv2 (not submitted)
Same experiments/scene_array_cache/Scene_Array_Caching.ipynb now26cells withcompleteCOCO80+13extensions=93classes. Fullclasslists COCO_ALL_80_CLASSES.json andALL_93_CLASSES.json. Scene369bytes,features374. Previous32classsources/spec/notebook preserved inruns/schema_v1_32_preserved_*/.7tests andfreshkernelfixtureexecution passed; no93classrealinference. No activeSPIjobchanged.


### Scene cache queued locally
All93classes retained; class_groups.json defines seven diagnosticgroups, noimportanceweights. Notebookexportscontractforlatergroupablations; no policytraininghere. QUEUE.json pending/not submitted. ActiveSPIunchanged.


## Diverse image-pool scene-only cache — v1-v3 results; v4 running

- Canonical notebook: `experiments/image_pool/Image_Pool_Caching.ipynb`, 36 cells, 909,608 bytes, SHA-256 `D84B8127BC7D1DB5D58D32E54897CCB76CE4AFCD8F1C9AF116B6D3B1993B2629`. Scene-only is injected only into staged copies; the canonical default remains full-pipeline. Main reports 12 focused tests passed; these do not establish real inference accuracy.
- Approved scene-only identity: `satyapaladuguplus/semantic-image-pool-scene-cache-20261030`, kernel ID `141684654`, URL https://www.kaggle.com/code/satyapaladuguplus/semantic-image-pool-scene-cache-20261030. Windows SPI v9 remains a separate, untouched run; SPI outputs remain pending until the completed 4,000-image model is available.
- Expanded target for v4: exact 4,000-row pinned CrisisMMD cohort, 1,000 Open Images, 1,000 Places365. Configured new-download cap remains 2 GiB. Missing sources are reported explicitly.
- Version 1 failed setup with `NameError` because the scene-only environment setting preceded `import os`; logs are in `experiments/image_pool/runs/wsl_scene_only_20261030/`.
- Version 2 failed setup while pip attempted to build the pinned OpenAI CLIP Git dependency; no acquisition or inference ran. Captured log SHA-256 `840072BCF4BC02A1E5B179BF61AD9BF356B65AC5BE9CDD6487CFD85EDB98FB6F`.
- Version 3 completed: 600 images acquired (200 each Open Images, Places365, CrisisMMD), and 600/600 scene records succeeded. Acceptance is `scene_only_spi_pending`; SPI and accuracy are unmeasured. Retrieved artifacts are under `experiments/image_pool/runs/wsl_scene_only_20261030/retrieved_v3/experiments/image_pool/runs/20261030T091952Z/`, 10,089,420 bytes excluding images and weights. The v3 600-row cache is preserved but not reused for v4: cache identity covers the complete input set and cache hits verify image bytes, so the expanded identity cannot be appended safely by the current implementation.
- Version 4 stage: `experiments/image_pool/runs/wsl_scene_only_20261030/staging_v4/Image_Pool_Caching.ipynb`, 909,932 bytes, SHA-256 `DD51805844DC2B4B67D613F790795F6D8AEE202BB01C8FC0F1C4C15D553DADC2`. Setup cell compiled and nbformat validated. Inspection confirmed scene-only setup fix and embedded caps 4,000/1,000/1,000 plus the exact 4,000 CrisisMMD rows. SDK accepted version 4; exact `/4` terminal status is `COMPLETE`. Acceptance reports all 6,000 images acquired, all requested sources available, SPI pending and accuracy unmeasured; source counts are exactly 4,000 CrisisMMD, 1,000 Open Images, 1,000 Places365. Scene progress and `scene_records.jsonl` both confirm 6,000/6,000 successful, zero failed. Download budget used 824,760,124 / 2,147,483,648 bytes. Selected version-specific metadata/cache outputs (scene SQLite, JSONL, scheduler arrays, identities, references and handoff) are at `experiments/image_pool/runs/wsl_scene_only_20261030/retrieved_v4_targeted/`, 91,239,377 bytes; no images or weights retrieved. Receipt `retrieval_receipt.json`. The monitor’s first broad output retrieval exited 1 after two files; targeted exact version 4 downloads succeeded.

## Mini provisional SPI enrichment — version 5 complete (2026-10-30)

- Existing private WSL identity `satyapaladuguplus/semantic-image-pool-scene-cache-20261030` (kernel ID `141684654`), version 5 ([Kaggle notebook](https://www.kaggle.com/code/satyapaladuguplus/semantic-image-pool-scene-cache-20261030/5)); exact status `KernelWorkerStatus.COMPLETE`. Runtime used Kaggle-managed Python with GPU and internet enabled. The source was the 916,276-byte, 38-cell `experiments/mini_dataset/Mini_Dataset.ipynb`, SHA-256 `0F3C647689F5017B0E9FB60F0376D43B278B4075A88B761FE2F5F0C69F27DB4B`, source commit `032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0`. SDK acceptance receipt: `experiments/mini_dataset/runs/mini_spi_20261030_v1/submission/submission_receipt.json`.
- Private managed input `satyapaladuguplus/mini-spi-training-labels` is ready. It contains 3,933 usable labels joined on image ID and SHA-256, all 4,000 raw label receipts (including 67 unavailable), comparison split references, and the exact v4 6,000-row `pool_manifest.json` plus expanded `scene/` cache. Payload was 80,004,468 bytes; checksums are in `experiments/mini_dataset/runs/mini_spi_20261030_v1/training_input/input_manifest.json`. v4 did not persist raw image files, so v5 reacquired the exact 6,000-image cohort under the 2 GiB transfer cap: CrisisMMD 4,000, Open Images 1,000, Places365 1,000; no sources missing.
- Version-5 run ID `20261030T130620Z`. Acceptance reports 6,000 acquired images, all requested parameters complete and no missing sources. The existing 6,000-scene cache is complete (6,000/6,000); handoff completion is 6,000/6,000. The archive contains 6,000 unique image files (6,136 tar members total), all 6,000 SPI rows available and unique, with exact input-ID and image-SHA coverage (zero missing, extra, or mismatched records). Prediction distribution: Low 3,189; Medium 1,053; High 1,539; Critical 219. Prediction JSONL SHA-256 `949e7b0a57d5451baba248a3b309d9c13dcf5646cfd495a062531f7e7b5bd7ef`.
- The provisional classifier is CLIP global embeddings with balanced logistic regression, C=10 selected using validation macro-F1. Train/validation/test counts are 2,068/1,037/828. Development macro-F1 is 0.602; held-out test macro-F1 is 0.600 and accuracy 0.649 against mixed-protocol VLM pseudo-labels. This measures pseudo-label agreement only; independent urgency validity and SPI accuracy on the 6,000-image pool are unmeasured. The result is provisional and does not alter full study promotion gates.
- Profiling scope is raw image to SPI (no scene inference or simulator calculations), 10 warmups: CUDA p50/p95 28.56/38.84 ms over 100 calls; CPU 104.74/112.46 ms over 30 calls. FLOPs are partial, with unsupported operators omitted.
- Remote archive `/kaggle/working/experiments/image_pool/runs/Mini_dataset_6000.tar` was retrieved to `experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/Mini_dataset_6000.tar`, 941,352,960 bytes, SHA-256 `0F8C6186B09DCF6AB87911DF642612919377157BB1585082BED1C93B1506FE2C`; SHA/size match the remote receipt and the tar listing contains exactly 6,000 image files. Receipt, verification and compact summary are in the same `downloads/` directory. Version-specific compact logs, acceptance, training receipt and artifact manifest are preserved under `experiments/mini_dataset/runs/mini_spi_20261030_v1/ops/remote_compact/`; monitor evidence under `ops/`.


## Full CrisisMMD SPI comparison — version 10 running (2026-10-30)

- Uses the existing private Windows Kaggle identity `satyapaladugu/crisismmd1000-spi-labeling` (kernel ID `141416127`), version 10; exact `/10` status is `KernelWorkerStatus.RUNNING`. SDK acceptance receipt is `experiments/spi_comparison_4000/runs/20261030T140904Z/submission/submission_receipt.json`; the staged 17-cell notebook is 152,718 bytes, SHA-256 `75A110567F38734A6617947EBBF0E35EE863AEB99EC03CD8247E8F9C8D7BCFFA`, source commit `032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0`. Run URL: https://www.kaggle.com/code/satyapaladugu/crisismmd1000-spi-labeling/10.
- Private managed input `satyapaladugu/spi-4000-training-input` is ready. It preserves v9 `labels.jsonl` (4,000 receipts), `selected_manifest.json` (4,000 rows), and `records.json` (3,933 available comparison rows), plus identity/progress/protocol/distribution receipts. The 3,933 available IDs equal the selected-label subset; SHA and label levels match. Label protocols remain distinguished (2,540 v1 checkpoint and 1,460 v3 Qwen2.5 no-thinking); 67 label receipts are unavailable and excluded from training. Input hashes/counts: `experiments/spi_comparison_4000/runs/20261030T140904Z/training_input/input_manifest.json`.
- Scope is training-only, no VLM relabeling: fresh feature extraction because v9 feature banks were unavailable/incompatible. Compare four encoders, twelve heads, and four balancing regimes with grouped development selection; locked-test use follows the staged notebook's explicit comparison protocol. Runtime is Kaggle-managed Python on `NvidiaTeslaT4` with internet enabled; setup pins Transformers 5.5.0, Ultralytics 8.3.237, floppy-tracker 0.1.0, and sentencepiece 0.2.2.
- Remote run output is expected under `/kaggle/working/experiments/spi_comparison_4000/runs/<run-id>/`, with stage logs, `study_status.json`, support/split/coverage/candidates/selection/test/cost receipts and `SPI_comparison_results.tar`. Ops monitor PID 6224 polls exact v10 every 60 seconds within a 24,000-second cap; monitor/logs/terminal receipt are under `experiments/spi_comparison_4000/runs/20261030T140904Z/ops/`. On completion retrieve compact study evidence and the packaged comparison result archive (which excludes image, feature, and checkpoint directories); do not retrieve all feature banks. Select the winner by development macro-F1 only, then preserve its matching classifier head and fetch only the selected encoder snapshot at the v10-recorded Hugging Face revision; verify its weight hashes against the exact remote encoder identity and retain processor/config/class-order/preprocessing/provenance files. On failure preserve full v10 traceback and study status. No rerun is authorized from this entry.




## 2026-10-30 WSL scheduler component launch — version1 RUNNING

User explicitly instructed START via the WSL Kaggle account. Verified account satyapaladuguplus; no eligible existing scheduler-training notebook among16owned identities. Prepared first private scheduler identity satyapaladuguplus/semantic-decoder-scheduler-20261030; canonical notebook experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb. source_manifest.json pins notebook/source hashes. CPU-only,GPU0hours,servercap1200seconds,drivercap900seconds;managedKagglePython standardlibrary/noinstalls. Attach own source kernel satyapaladuguplus/semantic-image-pool-scene-cache-20261030 (v5archiveSHA0f8c6186b09dcf6ab87911df642612919377157bb1585082bed1c93b1506fe2c). Parameter jobs untouched.

Purpose:6000exact compact wire roundtrips and12baseline rows across3fixed-population dynamic-arrival MIXED_IID traces. Channels retained93. No extraction/SPI/VLM inference or neural/RL training; no invented urgencylabels; J_D/nDCGundefined. Relevance/novelty belong to streaming stage, not blockers for these baselines. Expected runtime /kaggle/working/experiments/stream_scheduler/runs/compact-real-cache-<UTC>/ with source,cell_status/logs,environment/input/manifests,profilemapping,decisions/metrics,REPORT,recoveryZIP. Expected local retrieval experiments/stream_scheduler/runs/kaggle-compact-real-cache-20261030/. Kaggle CLI 2.2.4 accepted version1; exact /1 status was RUNNING on first check. CPU only (GPU=false, internet=false), timeout1200s, no accelerator flag. Six notebook cells comprise five computational stages: 01_environment, 02_source_snapshot, 03_input_and_regression, 04_transport_and_dynamic_replay, 05_report_and_recovery. Prepush exact hashes: notebook cd5b94f4f7cf723f6c10832988110952abd0a5fb8cb38e47bb958193c2d81386; embedded source ZIP 35fa19e75d939ebdab535feb083d6774c44d5baa66ba782db6a25cd5ff9590d1; driver 59f88f17df96ec23809548dfebd8d4822ca953e7a34f30aaa6cef8fe78ae2b4d; transport 15a7f1e4aac1435761de5ee53f8cc245c223fb057d281f588928d999fb95caf9; focused test c13149ed9f7c0e33cf3c39a40a1b6c7cae6e11d524c014998668a84cd50fcc70. All five code cells compile; synthetic 512-job horizon timestamp check passes. Push/status logs: experiments/stream_scheduler/runs/kaggle-launch-preflight-20261030/; initial expected output retrieval: experiments/stream_scheduler/runs/kaggle-compact-real-cache-20261030/. Current account quota14.51 GPU-hours remaining; this run uses0 and stays under the six aggregate GPU-hour program ceiling. Exact /1 status COMPLETE at2026-10-30T15:38:18Z. Remote runtime root /kaggle/working/experiments/stream_scheduler/runs/compact-real-cache-20261030T153724Z; all five computation stages completed. Recovered 6,000 exact roundtrips and12 rows across48/256/512 offered messages; observed compact scene bytes mean34.451333,p9545,max73; offered accounting holds, J_D/nDCG null, no training/inference, zero GPU. Recovery ZIP SHA-256 d82a13ae813751bec52b1aaf9899671f841a542c73d9c89c6ee3afc650a4b26b matches remote kernel log and local bytes. Retrieved ZIP, six stage/driver logs, kernel log, and retrieval_verification.json under experiments/stream_scheduler/runs/kaggle-compact-real-cache-20261030/. ZIP has50 members and no6000-image input tar; 27/28 internal artifact hashes match, sole cell_status.json mismatch because final cell-status write follows artifact-manifest hash. Kaggle output listing exposed no executed notebook; all five stage records say complete. The 941MB input archive was not retrieved.


## 2026-10-30 WSL scheduler real-data training — version2 RUNNING

Existing satyapaladuguplus/semantic-decoder-scheduler-20261030 identity; CPU private notebook version2 accepted and exactstatusRUNNING. WSLCLI push -t10800; noaccelerator or installs; imagepoolsourcev5 attached. Source notebook experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb and source_manifest.json. New code Evaluation/semantic_learning_env.py, Scheduler/semantic_learning_policy.py, Scheduler/training/semantic_learning_models.py and experiments/stream_scheduler/{runtime_bootstrap,semantic_suite_data,train_real_suite}.py. Sixreal-data local optimizer+export smoke arms passed2epochs,32decisions,1validationtrace:linear,MLP,RankNet,DeepSets,SetTransformer,temporalGRU. Shared D/lifecycle target/parity fixtures passed; Medium findings fixed.

Training config:10supervised arms, seeds1729/2718/31415,max50epochs,patience8,64offered training episodes/seed,512optimization snapshots,15validationtraces,modelwallcap600seconds,overalldrivercap10000seconds. Original image VLMsilver reference tiers joinedseparatelyfromruntimeSPI;3915observations match labels,3879unique known-reference source-train rows; originalvalidation/testhave0matchingtiers. Explicit inner grouped scheduler-development holdout is used; notindependentSPI/urgencytest. Originalsourcefinaltestunopened. Cached compatibleCLIPvectors supplycausalnovelty; missiontextcosinenotcomputedinthisCPUincrement. Baseline relevancepolicy fallsbackFIFO whenunavailable; no fabricatedsemanticvalues.

This version is supervisedDtraining, not all17suitecompletion or freshcontextualQwen training. RLmoduleswrittenseparatelyandvalidationpending. SixGPUhourceilingunchanged (CPUversionrequests0GPU). Five computational stages,capturelogs/checkpoints/models/learning/decision traces andchecksums under /kaggle/working/experiments/stream_scheduler/runs/compact-real-cache-<UTC>/. AutomatedfiniteWSLmonitor/expectedretrieval:experiments/stream_scheduler/runs/full-suite-20261030/ops and retrieved_v2/. Outputs notyetverified; no convergence/superiorityclaim.


### Training v2 output visibility inspection
User-supplied log ends atTRAINCOMMAND; exactv2status checkedRUNNING. Actualnotebook redirects childstdout/stderr todriver.log and waits beforeprinting, so visiblecelloutputcannotshowepochprogresslive. This is anobservabilitygap,not evidenceoftrainingfailure. Activev2notinterruptedorrelaunched. Acceptedv2source preserved at experiments/stream_scheduler/runs/full-suite-20261030/submission/accepted_training_v2.ipynb. Canonical notebook now contains a next-version thread-based stdouttee, tested withrealchildSTART/EPOCHoutput, cellscompile. Corrected stale900secondmanifestfield toactual10000seconddrivercap. Local nextversionsource NOTsubmitted; don'tconfuseitschecksumwithacceptedv2. Record experiments/stream_scheduler/runs/full-suite-20261030/ops/log_forwarding_fix.json.


## Scheduler full v2.2 launch 20261031T002242Z

- State: version 3 accepted and then manually stopped by the user. Kaggle now reports CANCEL_ACKNOWLEDGED at 2026-10-31 00:36:27 UTC. Stop is confirmed; no v4 relaunch was made by this operations task.
- Existing identity: satyapaladuguplus/semantic-decoder-scheduler-20261030.
- Canonical source: experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb.
- Local launch record: experiments/stream_scheduler/runs/full-v22-launch-20261031T002242Z/launch_config.json.
- Inputs: existing6000 mini-cache source kernel; exact archive SHA guarded.
- Requested runtime: managed GPU, internet for pinned official models; active-kernel Transformers4.57.6/bitsandbytes0.49.2; managed Torch retained.
- Expected remote output: /kaggle/working/experiments/stream_scheduler/runs/full-v22-<UTC>/, including source/data/teacher/P/D/AI-feedback checkpoints, logs, executed notebook if exposed, status/manifest/metrics.
- Expected local retrieval: this launch record/retrieved/. Retrieval and notebook version are tracked in the version-3 submission/monitor record below.
- One-seed breadth run within six aggregate GPU-hours; additional seeds not claimed. Teacher failures remain blocked; no actual codec.

### Version 3 launch acceptance and monitoring

- Kaggle CLI 2.2.4 accepted version 3 under the existing private identity. Exact command: kaggle kernels push -p experiments/stream_scheduler/runs/full-v22-launch-20261031T002242Z/submission -t 9900 --accelerator NvidiaTeslaT4. Acceptance stdout and receipt are in experiments/stream_scheduler/runs/full-v22-launch-20261031T002242Z/submission/. No new notebook identity was created.
- Pre-submit static audit passed: 113/113 canonical source hashes match the frozen manifest and embedded ZIP; all 113 source files and five notebook code cells compile; metadata confirms private, GPU and internet enabled. Notebook SHA-256 4265d0e356aee766e0d3268e7851fbe19cf9565f4e7c2dde3422e84a3fcc9819. Audit: pre_submit_audit.json.
- Immediately before push, the previous latest version was COMPLETE; quota was 0/30 GPU-hours used, 30 remaining (refresh 2026-11-07 UTC). Version 3 status was RUNNING on the first exact-version check.
- Requested runtime: T4, server cap 9,900 seconds; notebook measured driver cap is at most 8,400 seconds. Runtime reports two Tesla T4s and cuda:0; the driver maximum therefore projects 4.67 aggregate GPU-hours, within the six-hour ceiling. Final quota delta remains pending.
- Current local disk had 6.2 GB free at launch. Monitor inventories remote output and checks selected file sizes/free space before retrieval; it is bounded to 2 hours 50 minutes from monitor start and polls exact version 3. Monitor script/logs: experiments/stream_scheduler/runs/full-v22-launch-20261031T002242Z/monitor.py and monitor/.
- Runtime reported complete_vectors failure because project_compat was absent from the packaged source set. Row/hash audit: 3,887/5,952 unique image hashes have cached vectors; exactly 2,065 of 6,000 rows lack vectors (1,441 train, 417 validation, 207 test). Their relevance, novelty and cold-start availability masks are all false. This invalidates the intended full-parameter study and triggered the stop request. A corrected isolated helper has since passed (see full-v22-launch-20261031T002242Z/isolated_helper_fix_validation.json); no v4 relaunch is authorized yet.
- Do not treat Kaggle COMPLETE as success. Require stage/driver exit, runtime GPU count, actual aggregate quota delta, method ledger, and verified retrieved artifacts. The output file list initially reflected prior-version artifacts while the new version was starting; exact-version artifacts will be inventoried after termination. Initial runtime observation: experiments/stream_scheduler/runs/full-v22-launch-20261031T002242Z/monitor/first_worker_observation.json; streamed logs: monitor/follow_worker_logs.txt.

### Version 3 stop lookup status

- Checked at 2026-10-31 00:32:32 UTC: Kaggle status reports KernelWorkerStatus.RUNNING.
- Authenticated Kaggle SDK GetKernelSessionStatus returns only status and failure message; GetKernel metadata exposes notebook ID/current version but no session ID. The CLI status implementation ignores the version suffix. Therefore version number 3 is not a verified kernel session ID.
- One earlier cancel request used guessed ID 3 and was denied 403; this does not establish a permission limitation and is not repeated. No actual session ID was available from supported GET metadata, so no valid cancellation request has been sent. Do not infer a stopped/cancelled state. The user manually stopped version 3. At 00:36:27 UTC the notebook-level status reported CANCEL_ACKNOWLEDGED. Exact user click time is unavailable.
- Conservative aggregate GPU use from submission at 00:24 UTC through the last status check is at most 24.9 aggregate GPU-minutes (0.415 GPU-hours) from submission 00:24:00 UTC through 00:36:27 UTC, assuming both allocated T4s count for the entire interval. Kaggle quota after cancellation reports 0.18 GPU-hours used (rounded to 0.01 hours); starting quota was 0.00 hours.

## Scheduler full v2.2 launch 20261031T003706Z

- State: prepared, not yet submitted.
- Existing identity: satyapaladuguplus/semantic-decoder-scheduler-20261030.
- Canonical source: experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb.
- Local launch record: experiments/stream_scheduler/runs/full-v22-launch-20261031T003706Z/launch_config.json.
- Inputs: existing6000 mini-cache source kernel; exact archive SHA guarded.
- Requested runtime: managed GPU, internet for pinned official models; active-kernel Transformers4.57.6/bitsandbytes0.49.2; managed Torch retained.
- Expected remote output: /kaggle/working/experiments/stream_scheduler/runs/full-v22-<UTC>/, including source/data/teacher/P/D/AI-feedback checkpoints, logs, executed notebook if exposed, status/manifest/metrics.
- Expected local retrieval: this launch record/retrieved/. Retrieval and notebook version unverified until operations record confirms.
- One-seed breadth run within six aggregate GPU-hours; additional seeds not claimed. Teacher failures remain blocked; no actual codec.

- Terminal receipt and quota/output inventory snapshot: experiments/stream_scheduler/runs/full-v22-launch-20261031T002242Z/monitor/v3_terminal_check.json, v3_quota_after_cancel.txt, and v3_output_inventory_raw.txt. The listing only exposed previous-run artifacts dated 2026-10-30 16:31 UTC, so none were retrieved as v3 outputs.
## Scheduler full v2.2 corrected package, version 4, 20261031T003706Z

- State: accepted 2026-10-31 00:38:37 UTC; notebook status RUNNING at 00:39:45 UTC. At 00:52:52 UTC logs show vector fill MISSING_CLIP 2065/2065, teacher requests prepared, D suite consuming the vector_extension and training mlp_choice/1729 after linear_choice epochs 0-6. Qwen control outputs and train-pair outputs are visible; numeric control qualification has not been retrieved.
- Existing notebook identity reused: satyapaladuguplus/semantic-decoder-scheduler-20261030; Kaggle version 4.
- Canonical notebook: experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb; SHA-256 3d80562e9a04c91a531fc09b08f9b63a5776fae53c71ad1b2640bfd59a2e7f1c.
- Exact command: kaggle kernels push -p experiments/stream_scheduler/kaggle -t 9300 --accelerator NvidiaTeslaT4. CLI 2.2.4 exit 0.
- Pre-submit audit: 114/114 source hashes match; all 114 Python files and five notebook code cells compile; embedded ZIP contains exactly the 114 manifest files with matching hashes; metadata is private with GPU and internet enabled. Audits: experiments/stream_scheduler/runs/full-v22-launch-20261031T003706Z/v4_static_audit.json and v4_embedded_bundle_audit.json.
- Resource cap: Kaggle timeout 9,300 seconds; notebook internal cap 8,400 seconds; two T4 maximum 4.667 aggregate GPU-hours. Include prior v3 conservative 0.415 hours in the six-hour program cap. Kaggle quota after v3 stop displayed 0.18h used, rounded.
- Run records and logs: experiments/stream_scheduler/runs/full-v22-launch-20261031T003706Z/; first worker log capture is monitor/first_worker_logs.txt. Do not treat RUNNING status as evidence that GPU allocation, vector extension, teacher or training started.
- Persistent finite monitor: WSL PID 5544, script experiments/stream_scheduler/runs/full-v22-launch-20261031T003706Z/monitor/monitor_v4.py; hidden WSL parent PID 28476. It polls to terminal for at most three hours, streams worker logs, records status/progress, inventories outputs only at terminal, filters outputs by post-submission timestamps, checks free space, and checksums retrieved files. Current monitor files: monitor_console.log, v4_status_history.jsonl, v4_progress.jsonl, remote_worker_stream.log.



## Scheduler v2.2 repair validation, expected Kaggle version 5 (2026-11-01)

- State before submission: ready to submit; latest version of the existing notebook was checked COMPLETE at 2026-11-01 03:01 UTC. The existing private identity `satyapaladuguplus/semantic-decoder-scheduler-20261030` is reused.
- Mode: `repair_validation`, not full retraining. It checks all 17 synthetic supervised/RL method mechanisms, three CUDA channel math tests, and 39 real-image Qwen output-schema/control requests (32 controls, 4 candidate, 2 prefix, 1 replay). It performs no CLIP/SPI extraction and consumes the v4 verified vector and mission-vector artifacts.
- Canonical notebook: `experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb`; SHA-256 `dc893aada315ba73951d11b1807864cfdb0cc79f25f2fdefcaea5289bd00f202`.
- Source manifest: `experiments/stream_scheduler/runs/debug-e2e-20261101/repair_notebook_manifest.json`; 135 source/data bundle members. Source hashes match the local manifest and embedded ZIP hashes match, all five notebook code cells compile, and auxiliary input hashes match v4 `artifact_manifest.json`.
- Exact command: `kaggle kernels push -p experiments/stream_scheduler/kaggle -t 1200 --accelerator NvidiaTeslaT4`. Hard cap 1,200 wall seconds; at most two T4s = 0.667 aggregate GPU-hours. Prior conservative v3+v4 use is 3.987 aggregate hours; projected total is 4.654/6.0 hours.
- Launch config/receipt/logs: `experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v5/`. Expected remote output is `/kaggle/working/experiments/stream_scheduler/runs/repair-validation-<UTC>/`; only validation report, raw teacher outputs, driver logs and essential checksums will be selectively retrieved under this run record. No model weights are retrieved.
- Pre-submit audit passed at 2026-11-01 03:03 UTC: 131 source files hash-matched, 130 Python files compile, 135/135 embedded members match, four retrieved auxiliary files match, notebook hash/code cells and Kaggle metadata verified. Audit: experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v5/pre_submit_audit.json. Read-only account quota was 1.9354/30 GPU-hours used; project-specific conservative budget had about 2.013 aggregate GPU-hours remaining after v3+v4.
- Post-submit version, actual run path/status, measured duration/quota delta, outputs and hashes remain pending; Kaggle COMPLETE alone will not count as validation success.

### Scheduler v2.2 repair validation v5 submission correction (2026-11-01 03:08 UTC)

- The first SaveKernel attempt was rejected before execution with HTTP 400: “The kernel source must be less than 1 megabytes in size.” It created no version/run and consumed no GPU. Exact response: experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v5/diagnostic_push_response.json.
- The canonical notebook was repackaged without the large cached vector NPZ; the GPU run regenerates only the missing 2,065 pinned CLIP vectors from the attached source archive. Current notebook is 525,682 bytes, SHA-256 97d50ab85685c57fcff51cdf34398e8afe02da20337d61f1931fe42a2aef339c. The 133-member embedded bundle (131 source files, 2 mission-vector inputs) matched its manifest; all 130 embedded Python files parsed. Preflight: experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v5/preflight_repair_v5.json.
- Corrected submission to the same existing private identity was accepted as Kaggle version 5, kernel ID 141790587, at 2026-11-01 03:08:21 UTC; SaveKernel returned HTTP 200. Acceptance receipt: experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v5/diagnostic_push_response.json. No other version or notebook identity was created.
- The run is bounded to 1,200 wall seconds (at most 0.667 aggregate GPU-hours on two T4s), repair-validation only. Runtime reports two Tesla T4s, Torch CUDA 2.11.0+cu128. Stages 01–03 completed; 6,000-image archive mounted, compact transport regression passed, and 2,065 missing CLIP vectors regenerated. Pinned teacher is Qwen/Qwen2.5-VL-7B-Instruct revision cc594898137f460bfe9f0759e9844b3ce807cfb5 (NF4_double_FP16), not Qwen3.5; first two strict control responses parsed. These are progress observations, not terminal results.
- Terminal state: Kaggle version 5 COMPLETE. Remote run repair-validation-20261101T030831Z; driver ran 974.14s and exited 1 under the bounded repair-validation command. Runtime confirmed two Tesla T4s. For project-budget accounting, reserve the full 1,200s × 2 T4 cap = 0.667 aggregate GPU-hours (rather than converting CPU/GPU cell time into a misleading efficiency claim); using the prior 3.987h reserved estimate, project total is 4.654h/6.0h with 1.346h remaining. Account quota after run displayed 2.22/30 GPU-hours (rounded).
- Results: 17/17 learner mechanisms passed (16 CUDA arms plus LambdaMART CPU native/export parity). Teacher was partial: 34/39 responses parsed (32 controls, 2 candidate); all 32 controls were schema-valid and the narrow strict/tie/unknown control accuracy was 1.0. This tests formatting and visible-person/identical/blank controls only, not emergency priority. No prefix or replay requests completed; no RLAIF/replay learner labels were promoted. The CUDA channel regression failed at test collection because source/Channels/__init__.py imports Evaluation.Channels, whose awgn module requires the absent packaged Evaluation.Artifacts module. Exact traceback: retrieved results/cuda_channel_tests.log. Fix is pending and requires a bounded follow-up validation; this run is not an all-green suite.
- Version-specific selected output retrieval completed under experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v5/retrieved/experiments/stream_scheduler/runs/repair-validation-20261101T030831Z/. Twelve retrieved outputs (including REPORT.md, validation.json, CUDA traceback, raw teacher responses, qualification, provenance, and all 39 prepared teacher requests) match the remote artifact_manifest SHA-256 with zero mismatches. The teacher provenance pins Qwen/Qwen2.5-VL-7B-Instruct revision cc594898137f460bfe9f0759e9844b3ce807cfb5, NF4_double_FP16; no weights were retrieved. Verification record: experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v5/retrieval_verification.json. Kaggle run logs: experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v5/kaggle-live.log.
- The full local release harness that includes the later teacher cache tests passed 297 tests with 3 documented skips; waiting reference checks passed and source_drift=0. See experiments/stream_scheduler/runs/debug-e2e-20261101/validation-release/.

### Scheduler v2.2 technical follow-up v6 (2026-11-01; pre-submit)

- Purpose is limited to the two diagnosed v5 gaps: run the three CUDA channel tests using the added Evaluation/Artifacts, Interfaces, and Registries source closure; make five remaining Qwen technical-format queries (train-1-0/1, prefix-train-0-0/1, replay-0) using the cached teacher path and image bytes recovered from the attached 6,000-image archive. No vector/SPI/parameter re-extraction, learner retraining, 32-control repeat, or teacher-label promotion. v5 controls remain format/control evidence only.
- Reuse existing private kernel satyapaladuguplus/semantic-decoder-scheduler-20261030. Canonical notebook experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb SHA-256 d74699398f4eb77d8e56ada978958be3df72435f8c7c4583bdc8fc555c32742b (515,615 bytes, below the 1 MB source limit). Frozen manifest experiments/stream_scheduler/runs/debug-e2e-20261101/repair_notebook_manifest.json lists 162 source files and 3 verified v5 JSON inputs (165 embedded bundle members); local/embedded hashes and 161 Python AST parses passed in experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v6/preflight_repair_v6.json. The initial sterile check in experiments/stream_scheduler/runs/debug-e2e-20261101/sterile-v6/channel_bundle_preflight.log was not actually isolated: project-root imports masked missing Evaluation.errors.py, later exposed in Kaggle.
- Teacher pinned to Qwen/Qwen2.5-VL-7B-Instruct revision cc594898137f460bfe9f0759e9844b3ce807cfb5, NF4_double_FP16; v6 must compare model revision, quantization, output schema/version and all safetensor hashes against v5 provenance. CPU tokenizer factory tests verified equivalent allowed-token sets. New responses are technical schema smoke checks only, not training labels or policy-performance evidence.
- Exact submission: kaggle kernels push -p experiments/stream_scheduler/kaggle -t 900 --accelerator NvidiaTeslaT4. Accepted as Kaggle version 6 at 2026-11-01 03:33:58 UTC, same existing identity/kernel ID 141790587, SaveKernel HTTP 200. Receipt: experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v6/submission_receipt.json. Notebook SHA d74699398f4eb77d8e56ada978958be3df72435f8c7c4583bdc8fc555c32742b, 515,615 bytes. Hard cap 900s, at most 0.500 aggregate GPU-hours; prior conservative total 4.654h, projected maximum 5.154h/6.0h.
- Pre-submit gate passed at 2026-11-01: exact canonical notebook/metadata hashes, 165-member source/data bundle, all 161 embedded Python files parsed, and five technical query IDs plus prior teacher provenance verified in experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v6/preflight_repair_v6.json. Same existing kernel status was COMPLETE immediately before the one authorized push.
- Terminal state: version 6 COMPLETE; remote run repair-followup-20261101T033409Z, driver exit 1, targeted stage 411.66s. Runtime used two Tesla T4s. Teacher-format check passed 5/5 requests (candidate×2, prefix×2, replay×1), with exact v5 model/revision/NF4/output provenance verified; no training or label promotion occurred. Logs, response JSONL, provenance, qualification inputs, validation and channel traceback are retrieved under experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v6/retrieved/experiments/stream_scheduler/runs/repair-followup-20261101T033409Z/.
- CUDA channel test still failed during collection, exit 2: Evaluation.Channels.rician imports Evaluation.errors, which is not in the 165-file package. The earlier absent Evaluation.Artifacts dependency was fixed, exposing the next missing transitive module. This is a source-closure issue; the CUDA tests never executed. The reported local sterile run was contaminated by the canonical repository on sys.path, so it was not evidence that the Kaggle package closure was complete. Exact traceback is results/cuda_channel_tests.log in the retrieved run.
- Selective retrieval verified 11 downloaded artifacts against the remote artifact_manifest (225 entries total), with 0 hash mismatches. Verification: experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v6/retrieval_verification.json. Account quota displays 2.34/30 GPU-hours used (rounded). Reserved program total is 5.654/6.0 aggregate GPU-hours after the full v6 cap; 0.346h remains under the conservative budget.


## Version7: isolated CUDA channel closure check

Same notebook identity; accepted version7. Source includes Evaluation/errors.py and runs pytest with -c /dev/null, explicit rootdir/confcutdir to prevent parent repository fallback. Local extracted-bundle preflight:2passed/1CUDA skip. Hard cap300seconds, at most0.167aggregate GPU-hours; even prior conservative5.654 reserve remains below6. No teacher/learner retraining. Run logs/receipt: experiments/stream_scheduler/runs/debug-e2e-20261101/repair-validation-v7/. Terminal result pending.

- V7 terminal evidence: CUDA enabled, two devices, channel regression3passed in2.23seconds, exit0. Retrieved validation/JUnit/exit hashes match manifest; verification at debug-e2e-20261101/repair-validation-v7/verification.json. No model training or teacher calls.


## Master development run — version8
Accepted on existing identity; cap10800wall seconds/twoT4s (6aggregateGPUh new execution cap). Three seeds,50epochs/20kRLactions configured. Source+receipt+finite monitor under experiments/master_study/runs/launch-20261101/. Local preflight308passed/3skipped before allocation-only controller edits. Independent final references absent; adaptive second feedback round remains unimplemented and visible. Do not call this the completed journal study. Terminal/payload/actualdevice verification pending.


## Version9: corrected controller, no custom training wall limit
Version8 failed BEFORE training: inner import threading shadowed threading.Lock (UnboundLocalError). Preserved launch/live.log is failure evidence, not training. Removed the shadowing import and added a subprocess execution regression. Controller/teacher checks21passed. User explicitly revoked the three-hour cap. Version9 accepted on existing identity; CLI push has no -t override, notebook calls master directly with no run_guarded/time budget, per-model wrapper time limits removed. Kaggle platform limits remain. Monitoring/receipts at experiments/master_study/runs/launch-20261101/v9/ and submission-v9.log. Independent labels and second-feedback-round implementation remain honest outstanding study gaps, not successful results.


## Single-trial steering
User requested one trial per method. Canonical master SEEDS now1729; next notebook snapshot refreshed/compiled. Running version9 remains its frozen three-seed configuration. No new version submitted and no active training interrupted. Record: experiments/master_study/runs/single-trial-20261101/config_change.json.


### Scheduler master v9: platform timeout recovery (2026-11-02 IST)

API state CANCEL_ACKNOWLEDGED; user-visible timeout 43213.6 s, exit 137. Published output reported 951.33 MB. Recovery ongoing under experiments/master_study/runs/launch-20261101/v9/retrieved/; see TIMEOUT_RECOVERY.md in that run. No new training submitted. Final recovered master log reaches TD-PER seed 2718 at 4000 decisions. Full study incomplete.


### Scheduler repair preparation — 2 November 2026

Existing notebook source updated and compiled locally; not submitted. 321 local tests pass, 3 environment/codec skips; isolated source-capsule13 tests pass. New CPU preflight refuses unsupported general-study reference coverage. v9 outputs/models retained. See experiments/master_study/runs/repair-20261102/REPORT.md and notebook_validation.json for source snapshot, environment and limits.


### Two-account scheduler notebooks — submitted 2 November 2026
WSL active local source:experiments/master_study/paired_kaggle/wsl/Scheduler_WSL_Ranking_VLM.ipynb, existing owner/slug satyapaladuguplus/semantic-decoder-scheduler-20261030. Version12 run `wsl-20261102T162419Z` passed the real SQ-GAN input/hash gate, decoded six packets and started supervised training. Version11 failed at sidecar materialization; version10 failed at the earlier kernel-source preflight.
Windows local source:experiments/master_study/paired_kaggle/windows/Scheduler_Windows_RL.ipynb, actual owner/slug satyapaladugu/semantic-decoder-scheduler-rl, version1 RUNNING.
Both use the same source/cache snapshot and one seed1729; learner window10.5h, finalization11.5h, observed platform maximum12h.
WSL runs supervised/ranking+conditional teacher methods; Windows runs four RL methods with local Deep Sets initialization. SQ-GAN is frozen, verified separately; no Swin default.
Input readiness:Windows private dataset contains mini archive and shared tar. WSL private inputs include shared tar and the flat selected SQ-GAN bundle; remote inventory has 29/29 expected names/sizes, including `last.pt` exact size. Remote file content hashes have not been downloaded and independently verified.
Expected outputs:/kaggle/working/experiments/scheduler_pair/runs/<profile>-<UTC>/ with jobs, sessions, checkpoints, rawteacher, profiles, reports. WSL run logs are not yet available; Windows startup logs are retrieved under the paired launch run.
Canonical build/readme:experiments/master_study/paired_kaggle/README.md. Notebook/source hashes:paired_kaggle/notebook_manifest.json. Local checks:experiments/master_study/runs/paired-notebooks-20261102/.

### Two-account scheduler run — submission receipts (2 November 2026)

- WSL account `satyapaladuguplus`: version 10 accepted on existing identity `semantic-decoder-scheduler-20261030`; observed RUNNING. Private input dataset: `satyapaladuguplus/semantic-scheduler-paired-inputs-20261102` (shared `.tar.bin`). Kaggle rejected the historical `semantic-notebook-2` kernel source in push version10; the full selected SQ-GAN bundle is being uploaded as a separate private dataset. Version10 may fail its codec preflight; do not count it as a complete run.
- Windows account `satyapaladugu`: version1 accepted, observed RUNNING, actual identity `semantic-decoder-scheduler-rl` (Kaggle derived the slug from title; local metadata corrected). Private dataset version contains `Mini_dataset_6000.tar.bin` (941,352,960 B) and `Scheduler_shared_cache_20261102.tar.bin` (6,696,960 B). Kaggle expanded the earlier `.tar` upload; replacement `.tar.bin` paths were verified before submission.
- Both source capsules match `1aea11bce37676b35629b4520800a6e4b452b393fa81939c812a2defd38168a0`. Notebook hashes and API push messages: `experiments/master_study/runs/paired-launch-20261102-205933/launch_status.json`; ongoing upload logs are stored beside it. No training-completion or result claim until terminal outputs are retrieved and checked.

### Paired scheduler startup update — 2 November 2026

- WSL version10 is terminal ERROR at the SQ-GAN preflight because Kaggle rejected the `semantic-notebook-2` kernel source; no WSL learners trained. A private bundle dataset was created but must not be attached: its remote file listing shows zero-byte files, truncated `last.pt` (1,717,108,736 B vs expected 2,341,438,033 B), and no `SQ-GAN/` root. The bundle dataset ID was removed from local WSL metadata while the source integrity discrepancy is investigated. Remote listing and local staging checks are preserved in `experiments/master_study/runs/paired-launch-20261102-205933/`.
- Windows version1 is running at `satyapaladugu/semantic-decoder-scheduler-rl`, remote run `windows-20261102T154257Z`. Verified two Tesla T4s, archive/cache preflight, 13 tests, Deep Sets epoch23, and Double DQN 2,000 decisions/375 updates. These are early training checks, not completion or final results. Logs and status: `experiments/master_study/runs/paired-launch-20261102-205933/windows_startup/`.
- Full submission/current state: `experiments/master_study/runs/paired-launch-20261102-205933/launch_status.json`; paired notebook README now reflects actual status. Do not claim the paired suite complete until Windows terminal outputs are retrieved and the WSL SQ-GAN package is rebuilt, hash-verified, and rerun.

### Paired scheduler corrected WSL submission — version11 (2 November 2026)

- Existing private identity reused: `satyapaladuguplus/semantic-decoder-scheduler-20261030`; version11 run `wsl-20261102T161938Z` failed cell06 before training because a sidecar record object was combined directly with a `Path`. Version12 was accepted (push exit0), notebook SHA256 `0e5d6e58ed451f1c9e382fef4a68ee069cf898d6cde69abd026c62a4335bf61f`; run `wsl-20261102T162419Z` now trains. Cell05 passed 13 tests; cell06 materialized all 27 SQ-GAN files and encoded/decoded six 464-byte payloads, with 0.392–0.527s decode times. Cell07 started supervised methods on `cuda:0`, teacher-request preparation, and reached `linear_choice/1729` epoch0 (loss 2.206066, validation -5.683251); Qwen checkpoint download completed. No completion or superiority claim yet. Source capsule SHA256: `1aea11bce37676b35629b4520800a6e4b452b393fa81939c812a2defd38168a0`.
- Attached private SQ-GAN bundle dataset: `satyapaladuguplus/semantic-scheduler-sqgan-bundle-20261102`. Flat upload uses a sidecar mapping to rebuild the original bundle paths inside Kaggle working storage. Remote API inventory has 29/29 expected filenames and byte sizes; `f015_last.pt` is 2,341,438,033 bytes. Remote content SHA256 was not downloaded/rechecked. The original 27-file bundle on the workstation independently passed its manifest audit. The earlier corrupt remote dataset version is preserved but not attached.
- Version11 push receipt: `experiments/master_study/runs/paired-launch-20261102-205933/push-wsl-v11.log` and `.exit`; exact remote inventory: `sqgan-remote-files-v2.json` and `sqgan_remote_verification.json`. Flat upload progress/capture limitation: `upload-wsl.log`; launch state: `launch_status.json`.
- Windows version1 remains on its actual identity `satyapaladugu/semantic-decoder-scheduler-rl`; early startup/training evidence is in `windows_startup/`. The WSL run must show bundle materialization, adapter verification, real codec preflight and learner progress before being described as training.

### Paired scheduler run completion check — 3 November 2026

- WSL v12 (`wsl-20261102T162419Z`) and Windows v1 (`windows-20261102T154257Z`) both report Kaggle API `COMPLETE`. Runtime from captured logs: 10,572 s (~2 h 56 m) WSL and 9,933 s (~2 h 46 m) Windows; neither hit the 12-hour limit.
- WSL result status is `partial_or_blocked`: ten supervised/ranking methods are recorded `evaluated_development_only` (64 episodes, 512 samples each; holdout unopened). Nine teacher-dependent methods are blocked. Qwen pilot had 8 parsed comparisons, 4 consistent, agreement 0.50, qualification false. Rate study failed on missing `Evaluation.ue_scenarios` import. No scientific final evaluation is claimed.
- Windows `RUN_STATUS.json` is `complete_method_attempts` with no incomplete methods or stage failures. Its assigned four RL methods reached 20,000 decisions/steps; Deep Sets initialization completed 50 epochs. Windows final test was not opened.
- Across both accounts this is 14 unique non-teacher methods (10 supervised + 4 RL); Windows Deep Sets initialization duplicates the WSL Deep Sets arm. Nine VLM-dependent methods were blocked, not trained.
- Current status evidence is under `experiments/master_study/runs/paired-launch-20261102-205933/checks/paired-check-20261102T220621Z/`. `CHECK_REPORT.md` is the concise summary; per-version full logs and only small status/report/ledger artifacts are saved there. No weights, images, or archives were retrieved. The checked notebook/run manifest and canonical paired README are updated.
### Teacher-only recovery WSL — version 13 (3 November 2026)

- Reused private kernel satyapaladuguplus/semantic-decoder-scheduler-20261030. v12 reported COMPLETE before launch. Kaggle CLI push exit 0 accepted version 13; latest status is RUNNING. Remote run ID from live logs: teacher_recovery-20261102T224440Z.
- Local notebook experiments/master_study/gap_recovery/WSL_Teacher_Recovery.ipynb SHA-256 2fdea70f7df416957dadd91cd77ce5ac0bc0ca91f80a4726388e99edf830d30c; embedded source capsule SHA-256 a824d45cb146d850e709a67d29ded1aa2fa8dd3ca1db8248a82a4548fa3291e8. All 182 capsule members match the manifest. Kaggle serializes the remote notebook under the kernel slug; the pulled latest source preserves the teacher_recovery profile and exact capsule hash.
- Private metadata attaches only semantic-scheduler-paired-inputs-20261102 and semantic-image-pool-scene-cache-20261030; no SQ-GAN bundle. Runtime confirmed Python 3.13.15, Torch 2.11.0+cu128 and two Tesla T4 devices.
- Preflight passed: master study 28, teacher/supervisor 20 and scene learning 48 tests. Notebook nbformat/AST checks, 182 source-file hash checks and the three-code-cell CPU smoke passed. Source hash diff is empty.
- Live cell05 finished with 38 packaged regression/SQ01 checks passed; two local tokenizer/retrieved-evidence checks were explicitly skipped because those optional artifacts were not installed locally. Cell07 began prepare_teacher; no teacher qualification, 32-control/8-context swap pilot result, or first-inference result is claimed yet.
- Scope is only nine teacher-dependent arms; the 14 completed non-teacher methods are not rerun. Scientific controls, 0.8 swap consistency and minimum 16/8 support gates remain unchanged. Runtime caps are 10.5 h training, 11.5 h finalization and 12 h platform limit.
- Expected Kaggle output root: /kaggle/working/experiments/scheduler_pair/runs/teacher_recovery-20261102T224440Z/. Retrieve the executed notebook, per-cell logs/errors, run config/input manifest, teacher panels, first inference, checkpoints and metrics when available; verify local persistence.
- Launch and preflight logs, metadata pulls, live log stream and run record: experiments/master_study/runs/gap-repair-20261102T221315Z/validation/teacher_recovery/launch/. Live stdout capture is live_logs.stdout.log.
- Kaggle quota was not exposed by the installed CLI and remains unverified. This is a running remote job, not a completed teacher qualification or final scientific result.


### 3 November 2026: targeted FP16 recovery v14
Existing satyapaladuguplus/semantic-decoder-scheduler-20261030, version14 submitted and API RUNNING. Local notebook experiments/master_study/gap_recovery/WSL_Teacher_Recovery.ipynb. Source cc9655ede5cf2bfcad5a5611efddc7390f73d49229c2db488236850f3a82fe57. Same teacher revision, cases, panels and thresholds; FP16 balanced over2T4 rather than NF4.27 regression tests passed. Teacher qualification and downstream nine learner outcomes pending; no completed14retraining. Receipts experiments/master_study/runs/gap-repair-20261102T221315Z/fp16-push.log and fp16_launch_status.json.

Version14 outcome verified: COMPLETE platform, partial_or_blocked research. Visual controls strict/tie/unknown100%; orientation1/8=12.5%, teacherunqualified. No nine dependent models trained. Evidence experiments/master_study/runs/gap-repair-20261102T221315Z/v14-check-logs.txt. FP16 did not fix scheduling-choice reliability; original14 and full320row rate study remain completed.

## Mainstream source promotion 5 November 2026

Canonical notebook sources now live under `Notebooks/SchedulerSuite/paired/wsl`, `Notebooks/SchedulerSuite/paired/windows` and `Notebooks/SchedulerSuite/recovery`. Existing owner slugs, published versions, run IDs and retrieved outputs are unchanged. Builders are in `Scheduler/training/study`. These rebuilt sources have not been submitted. Historical notebooks remain at their recorded paths.

## Trial 2 Nemotron single-photo follow-up — 10 Nov 2026
- Existing private notebook: `hmmkk20/semantic-scheduler-trial2-vlm-smoke`, version 14 submitted; status was `QUEUED` at last check.
- Purpose: same 15 LADI calibration photos and same single-photo grade/extent/affected/evidence rubric as Qwen v13; two sweeps; pinned Nemotron revision `437f4e28b989cc2d9a16b6767cc930cdf48797ff`; one Tesla T4, NF4/FP16.
- Source: `%TEMP%/trial2-vlm-nemotron-single-v14/pilot.py`; local Python compilation passed. Expected output: `/kaggle/working/reviewer_extent_nemotron.json`. Results pending; no completion or inference claim until downloaded and hash-verified.

### Trial 2 Nemotron v14 completion — 10 Nov 2026
- Version14 status: COMPLETE. Downloaded the exact output to `%TEMP%/trial2-vlm-nemotron-single-output-v14/` and archived the submitted source, raw response JSON, log, and offline analysis under `experiments/trial2_vlm_review_nemotron_v14/`.
- Raw receipt SHA-256: `3bd6f90d3c5f1e0b8f7ba87bcb822c36653eeffcb514b2068ba5cb9ea5a4d159`. Model: `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1`, revision `437f4e28b989cc2d9a16b6767cc930cdf48797ff`; Tesla T4; 225 NF4 layers; peak allocated memory 9,150,445,568 bytes.
- Two reversed sweeps produced 30 valid structured raw responses; all 15 photos repeated exact grade, extent, affected list, and evidence. The submitted online parser rejected the outputs due to an over-escaped regex; corrected offline parsing validated the preserved raw JSON. Median call latency 9.31s.
- Of Qwen's 16 calibration ties, Nemotron has 5 strict grade differences: 4 unsupported preferences for `c05293ad` (based on buildings/parking/roads, not visible disaster evidence), 1 unadjudicated possible damaged-roof preference for `446d99e4`; 11 ties. Count zero new strict training labels. No RankNet training.
- Exact grade agreement with Qwen: 9/15 for v12, 10/14 for v13 (one malformed Qwen photo excluded). Archived source SHA-256: `3e0b172246228643864a5cbb3c25ee99a8f95824e7aac94e336d6550cf54d082`; offline analysis SHA-256: `c52f6013e60299e6f5fcc790a83879daa72401fe419d9c6ed7bfaa5039141c78`.
