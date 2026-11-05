# C1 assessment

Outcome: lifecycle/header boundaries look consistent in the inspected paths, but exported-artifact validation is incomplete. Resolve the two confirmed findings below before claiming that malformed contextual artifacts fail before job admission.

## Scope and identity

Assessment only; no source, test, configuration or plan changes. Baseline HEAD: `2f16785a16dd9c58ce12c98b502087cc212b279b`; assessed working-tree additions are pinned by SHA-256 in `results/decoder_context/c1_assessment/reproductions.json`. Files inspected: `INFRA/decoder_queue.py`, `INFRA/decoder_service.py`, `Scheduler/decoder_scheduler.py`, `Scheduler/contextual_decoder_scheduler.py`, `SemParam/decoder_parameters.py`, `SemParam/decoder_context_profiles.py`, and the two new service/context scheduler test files. Limited static inspection of the current benchmark/dataset adapters was contextual only; those are still being developed by the main owner.

Acceptance contracts came from the parent handoff and `docs/decoder_context/RESEARCH_PLAN.md` sections 7 and 11: opaque complete 32-byte-header jobs; one nonpreemptive in-flight job; exact monotonic completion; success-only bounded header history; no automatic retry; stable payload-free context; existing baseline/profile behavior; field exclusion from derived features; validated NumPy export; counted nonfinite fallback to direct I.

## Confirmed findings

### C1-01 — High: load-time validation can accept a model that crashes on real candidates

Locations: `Scheduler/contextual_decoder_scheduler.py:144-146`, `:110-130`, `:172`.

The eager check evaluates only one all-zero dummy row. A three-node tree can have a valid left leaf and an invalid internal node on its right branch. Construction follows the left branch and succeeds; a real candidate whose first transformed feature exceeds 0.5 enters the right branch and raises `ValueError: invalid tree node index`. Likewise, an otherwise valid artifact with `fields=[8]` passes construction, then raises `ValueError: invalid field selection` on its first score request. These are correctly checksummed artifacts, so checksum verification does not detect structural invalidity.

Impact: violates validation-before-admission and can halt a decoder service after valid jobs have been admitted. The job remains queued because service removal follows policy selection; this reproducer does not establish job loss. Nonfinite-output fallback does not handle these exceptions, nor should it conceal a malformed export that should have failed at load time.

Smallest correction: explicitly validate field selection and all tree nodes at construction; verify array shapes/types, finite numerics, exact integer child/feature indices, leaf/internal-node structure, valid indices, and root-reachable acyclicity independently of a chosen feature row. Keep the existing dummy numerical check as an additional check, not the structural validator. Reject missing/invalid artifact configuration before returning the policy.

Evidence: `invalid_fields` and `unvisited_branch` in `results/decoder_context/c1_assessment/reproductions.json`. Both constructors were accepted and both failed on one valid candidate.

### C1-02 — Medium: fractional exported indices silently change model semantics

Locations: `Scheduler/contextual_decoder_scheduler.py:121`, `:125`.

Child and feature arrays are converted with `int(...)` without an integrality check. A correctly checksummed tree with children 1.8 and 2.8 and feature 0.8 is accepted and interpreted as children 1 and 2 and feature 0. This is silent truncation of malformed artifact data rather than a clear rejection.

Impact: exported numerical behavior can differ from the serialized model's stated structure; a failed export may produce plausible scores and evade detection.

Smallest correction: require exact integer-valued child and feature arrays before conversion, in the same full-tree validator needed for C1-01. Test fractional values that would truncate into valid ranges.

Evidence: `fractional_index` in `reproductions.json`: construction accepted; score was 2.0.

## Coverage assessment and limits

- Static inspection supports 32-byte/CRC/profile/token validation at ingress, opaque payload ownership inside queue/service, one in-flight job, exact completion identity, monotonic service clock, failed completion exclusion from history, no automatic retry, and tuple snapshots of frozen header candidates. These are inspection conclusions plus the existing test assertions, not a new broad execution certification.
- Existing service tests exercise busy dispatch, foreign/duplicate completion, backward completion/dispatch time, future arrival, busy reset, failed history, and registry injection. Context tests cover a simple hand-reference distance/signature case, candidate permutation, one-field exclusion, empty context, checksum failure, and injected nonfinite linear output fallback with its counter.
- The inspected new tests do not exercise exported tree inference at all, full artifact structural validation, 32-to-33 successful-history rollover, or an actual contextual-policy/service lifecycle together. Add focused cases for these meaningful gaps; broad duplicate tests are unnecessary.
- The nonfinite fallback test monkeypatches `_predict`. It supports selection/counter behavior for a returned NaN array, not exporter equivalence or real arithmetic-overflow behavior. Independent exported linear/tree equivalence on representative finite rows remains a necessary milestone check when the training exporter is ready.
- No additional confirmed scientific or integration defect was established in the bounded static inspection of benchmark/dataset adapters. No dataset cohort, training quality, generalization, transport authentication, real codec, concurrency safety, or hardware behavior was tested by this checkpoint.

## Reproduction

Only the three targeted numerical reproductions were executed, using `C:\Python314\python.exe` (Python 3.14; full version recorded in JSON), NumPy 2.4.3, CPU; one candidate and one three-node tree per tree case. Execution finished in under one second of process wall time. No broad test suite was run; the parent's reported 32 passing focused tests are not independently rerun here.

From repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python results/decoder_context/c1_assessment/reproduce.py
```

Artifacts: `results/decoder_context/c1_assessment/reproduce.py`, `results/decoder_context/c1_assessment/reproductions.json`.

## Bounded correction recheck

C1-01 and C1-02 are closed for the reported triggers. The constructor now checks selected fields explicitly and validates every tree node before dummy inference. Full graph validation checks finite arrays, exact integral indices, legal branches, a single root/single parent and traversal of all nodes. The original invalid-field, unvisited-invalid-branch and fractional-index examples now all fail during construction. The new regression test contains invalid fields, an out-of-range child and a fractional child; the broader validator also checks fractional features.

The new `INFRA/learning/decoder_context_training.py` export was assessed with independent sklearn refits and runtime `policy.scores(context)` on separate generated contexts. Six configurations passed: linear and 50-tree boosting, each in full-context, pointwise and masks-only mode, with fields `(0,3,7)` and task IDs `(2001,2002)`. Each configuration trained on 24 decisions / 72 candidate rows and was compared on 12 separate decisions / 36 rows; 41 features included the two task indicators. Maximum absolute error was `5.551115123125783e-17` for linear and zero for trees. Thus the runtime feature masking/task conditioning and exported arithmetic agree with the sklearn reference on this synthetic scale. This establishes export parity, not predictive quality or dataset generalization. The tree float32 conversion matches sklearn's input convention for these checks.

No new blocking issue was established in this bounded recheck. No broad tests or production edits were performed. Explicit exact-threshold and nextafter boundary probes, real-dataset generalization and caller-enforced training/validation group isolation remain outside this check; the exporter labels its scope but cannot itself establish caller provenance.

Current code hashes, Python/NumPy/sklearn identities, closure outcomes, seed and numerical results are in `results/decoder_context/c1_assessment/recheck.json` (sklearn 1.9.0, CPU, data seed 1729). These checks ran in approximately five seconds. The initial reproducer writes a fixed output path: rerunning it overwrote the initial hash record. `reproductions.json` now explicitly preserves the initial observed outcomes reconstructed from the original tool output, and states that the initial hashes are unavailable; use `recheck.json` for the current assessed snapshot. This supersedes the initial report's claim that the initial working snapshot remains hash-pinned.

## Final bounded real-codec integration assessment

Outcome: no critical blocker established in the inspected five-image integration. This is a completion-controller/codec integration demonstration, not evidence of a scheduling improvement, independent semantic preservation, losslessness, or production readiness.

Inspected `INFRA/decoder_codec_driver.py`, `SemParam/decoder_native_classifier.py`, `INFRA/learning/decoder_codec_semantic_smoke.py`, their two integration test files, the called received-cache decoder, and Swin's `reconstruct_received` boundary. The three new module hashes match the operation records under `results/decoder_context/implementation_ops/20261024T083335Z`:

- driver: `C95CEABCA3A364EFD37E04E876A54E97CAB6AFEFAC9C90B7A65F00E304535D4D`
- classifier: `07205B08E4FA57FDC233CC0EF3B3775713CD61CE8EA36D7381BF3BD991730437`
- semantic smoke: `DAC41BE6C12CAFCC43BA25E78099E6FAA0C6109DC78D2B2D220B3F743584B5A7`

Information boundaries: the local classifier derives logits from source/reference images. `make_headers` consumes only those `ImageEvidence` objects, extracts/serializes before committing source memory, and supplies no native annotation or reconstruction assessment to the policy. For this producer, embeddings, relevance, calibrated Q, trusted time and detections are absent: R/Q/F/P/C/N remain explicitly unavailable; S is present only when supported eligible concepts exist; I is predicted image evidence. Native labels are read in the assessment loop and are not supplied to extraction or queue/service. Header policies receive immutable candidate snapshots, with payload records remaining inside the service. Cached payloads also contain the source latent for adapter packaging, but the inspected Swin `reconstruct_received` replaces the tensor with `transmission.received` before decoding; no clean-latent reconstruction bypass was found.

Execution/failure boundary: the driver admits all five complete manifest records with validated 32-byte headers, then dispatches/decodes/completes one job at a time. A decode exception produces one failed completion and no automatic retry; existing service logic excludes it from successful history. The fixture test exercises terminal failures, common payload hashes and different FIFO/I orders. It does not establish recovery after process termination or independent failure isolation for filesystem exhaustion. Successful decoder timing is synchronized inside `decode_cached`; the controller clock adds dispatch plus inference only. Payload loading, hashing, PSNR calculation, PNG output and successful artifact I/O are excluded; the failed path instead includes elapsed error/I/O time, as the driver documents. Therefore cumulative completion values are a stated timing model, not measured end-to-end receiver latency, and their cross-policy differences are not a defensible speedup claim.

Read existing artifacts only; no inference or broad tests were rerun. `partial_headers/status.json` records five items, profile 2001 and classifier checkpoint `57288c631557ec80d9d50e4f4b7f5b11aeb1c3fd9905fc3c6c6d59e7cbf45213`. Operation records identify source classification in the existing `.venv`, and strict real-codec replay in `.venv-models` on CPU. The cached 10 dB/rate-96 payloads were generated earlier and replayed without re-encoding/re-sampling. All FIFO/I/learned summaries contain five successes and zero failures. The learned order equals direct I on this cohort. Cumulative dispatch-plus-inference endpoints were approximately 56.937, 56.922 and 59.825 seconds respectively; these are descriptive run values only. All reconstructed `exact_pixels` flags are false.

`partial_headers/assessment.json` reports four model-positive reconstructions out of five native-positive source examples; this is frozen-classifier assessment on a small exposed cohort, not independent fact truth or demonstrated native-fact retention. The supplied assessment images originate from the earlier decode pass; the replay inventories record identical per-image reconstruction hashes across policies, supporting reuse for this integration. No claim about real semantic novelty or model generalization follows from the exported contextual model running successfully.

Evidence paths: `partial_headers.json`, `partial_headers/headers.json`, `partial_headers/assessment.json`, `actual_policy_replay.json`, `actual_policy_replay/summary.json`, and `actual_policy_replay_review.json` under the operation root above. Luna's artifact/run verification remains the execution authority for the full run inventory. Remaining limits include no real-codec failure injection, no process-resume test, no independent human fact assessment, no fresh holdout cohort and no repeated timing experiment. The current semantic-smoke CLI also performs reconstruction assessment before writing final headers; this couples offline artifact availability to assessment success, although it does not leak labels into header values. A future standalone producer should separate these stages if independent header availability becomes a requirement.
