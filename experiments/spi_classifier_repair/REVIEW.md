# Bounded classifier-repair source review

Assessment only, 2026-10-28. Reviewed `cnn_compare.py`, `diagnostics.py`, and optional history changes in `parameters/spi_model_selection.py`; no GPU/model execution, downloads, training, or production edits. **Final affected-change clearance: all original findings and the output-preservation follow-up are resolved. No consequential source blocker remains within this bounded review.** No change to the approved experiment is proposed.

## Final affected-change recheck

Pinned whole-image teacher/rubric and exact unique label-cohort validation are now explicit. The outer run handler persists failure status and traceback, and the inner split explicitly rejects insufficient group support. BatchNorm policy now accurately states frozen running statistics with trainable affine parameters in unfrozen blocks, resolving the ambiguity. Added plotting reads saved inner histories and exposed-development results without adding a new evaluation or affecting model selection. Main reports a real 24-image CPU one-epoch smoke and four regression tests passing, including weight invariance under instrumentation; these are reported checks, not independently repeated here. GPU execution remains unverified.

**RESOLVED follow-up preservation finding (`cnn_compare.py:run` exception handler):** when `_run` rejects an already-existing output directory via `mkdir(exist_ok=False)`, outer `run` catches that error and rewrites `status.json` and `traceback.txt` inside the existing run. A retry/path collision can therefore replace a successful run status with failure. Check output nonexistence before entering the handler, or track successful acquisition/creation of the output directory and write failure artifacts only if this invocation owns it. Preserve existing runs byte-for-byte on collision. Add a focused existing-output regression assertion. Final recheck confirms output creation now occurs outside the exception handler, so collision rejection cannot rewrite an existing run. _run requires an already-owned directory.

Final recheck confirms 1..20 epochs and 1..1500 seconds are enforced before output creation. Invalid bounds leave no output directory. Main reports the added collision-preservation and invalid-budget regression checks passing; reviewer did not repeat execution.

## Original findings (resolved)


1. **P2: label identity and duplicate ownership are insufficiently guarded (`cnn_compare.py:19-26`).** A dict comprehension silently overwrites duplicate label IDs. Available rows are accepted after checking image hash/group/split, without enforcing whole-image pseudo-label source, selected Qwen teacher/revision, or one expected rubric. Thus wrong-protocol/native/remapped labels with matching image metadata would pass. Validate exact expected record/label ID ownership and duplicate counts; enforce the accepted existing teacher label schema and identity (or bind a verified immutable labels-file hash with equivalent schema checks), and persist that identity in config. This is a robustness/provenance fault, not evidence current saved labels are contaminated. Existing canonical `prepare` contracts can supply reusable schema/group checks where suitable.

2. **P2: failure has no persisted terminal status (`cnn_compare.py:85-98`).** Training uses try/finally without except. Timeout, OOM, failed pretrained load, or evaluation failure leaves histories when available but no failed status/reason; initial identity/split errors occur outside that guard. Wrap the run lifecycle in error recording with failed stage, elapsed time, exception/traceback and runtime_promotion_allowed=false, then rethrow and clean up. Persist available epoch histories/checkpoint evidence without promising killed-kernel recovery. Notebook recorder capture complements, but does not replace, the run's terminal status.

3. **P2: requested BN freezing is incomplete (`cnn_compare.py:65-71`).** Calling BatchNorm.eval freezes running statistics, but unfreezing all parameters in features[6:] also enables BatchNorm affine weights/biases in those blocks. If the approved BN-frozen contract means fully frozen BN, explicitly keep their parameters requires_grad=false after unfreezing and exclude frozen parameters from the optimizer. If only running-stat freezing was intended, name that narrower behavior in config rather than claiming all BN is frozen.

Also add inexpensive argument guards for 1<=max_epochs<=20 and 0<seconds<=1800. Currently max_epochs=0 silently selects epoch 1 and performs a one-epoch refit; larger caller values can exceed the advertised epoch ceiling. Defaults are within budget. The code uses one explicit model device, so 1800 wall seconds on one GPU stays below the one aggregate GPU-hour envelope, subject to external launcher/download overhead; no measured cost was established here. Check budget around pretrained loading and terminal serialization, and let the launcher enforce its outer deadline.

## Correct behavior by inspection

- The inner split uses only training rows/groups; `stopping_split` checks class support and split group separation. The selected epoch is maximum inner macro-F1 with lower loss as tie-break; patience is six epochs. The 96-row exposed development split is evaluated after the full-204 refit only, not used to choose epochs.
- Selection training starts with head-only warmup, then features[6:] plus head after three epochs. Refitting rebuilds ImageNet-initialized model and resets Torch seed before training the selected epoch count; it does not load inner-selection weights. Fit class weights are recomputed from that fit subset, and full-refit weights use only the 204 training labels.
- The source never automatically promotes the CNN; saved status/report mark runtime_promotion_allowed=false. Images are padded without crop/flip and use declared normalization. Outcomes remain teacher agreement on development-exposed data, with no human accuracy or fresh final-test claim.
- Diagnostics learning curves score held-out training CV groups; subset sampling uses only each fold's training groups and fit-only class weighting. Different seeds vary subsets, not independent dataset replications; the full-size subsets repeat the same deterministic linear fit. Do not interpret these repeated rows as independent performance evidence.
- Optional MLP history collects eval-mode CE/accuracy and does not change loss optimization or early-stopping criterion; the extra deterministic evaluation consumes no dropout RNG. Internal stop curves are accurately distinguished from the 96-row development validation.

## Limits

This is source-only validation. Label-group completeness, actual GPU backend, runtime/dependency/weights identity, package inclusion, checkpoint recovery, wall-time enforcement, and real CNN quality need the staged preflight/run evidence. Keep the existing v13 baseline unchanged and record source/input hashes in the launch manifest. No VLM rerun is needed or authorized by this comparison.

## Final reviewed source identity

`cnn_compare.py` SHA256: `e046d0abb525223953156fe158a3a9e466cb4e22a7fc18852b74ebf5725f2d07`. GPU execution, packaged-source identity, and remote resource enforcement remain operational validation; source clearance does not establish CNN quality or promotion readiness.
