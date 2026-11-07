# Classifier-independent stream build: local status

30 October consolidated master plan: SEMANTIC_RANKING_RESEARCH_DESIGN.md now
retains contextual Deep Sets RankNet, processed-header history, and RL/VLM arms
as the full project. The first simple scorer is a baseline. Versioned image and
distinct-fact first-service delay costs are specified; they are not implemented
or trained by this documentation revision. The earlier design notes below are
historical and must not override the consolidated plan.

30 October design finalisation: the canonical plan now fixes one-shot arrival and
one attempt, no quality-driven retry/rescheduling, and disables receiver-success-
history learning in the planned core observation profile. This is not yet a
production change. The new bounded objective experiment under
experiments/semantic_priority_design/runs/objective-check-20261030-013021/
checks tiny scheduling mathematics only; it does not rerun the learners or validate
the proposed new wait evaluator. The historical implementation below is preserved.

Later design clarification, 29 October: primary scheduling evaluation is moving
to reference-class queue wait (arrival to dispatch) plus nDCG, with equal service
slots in the main comparison. The plan records this separately. Existing pilot
reward and class cap metrics remain completion-based; the new all-selected wait
evaluation is not claimed implemented or rerun here.

29 October 2026. Canonical codebase: SemanticSchedulerEnd2End.

## Implemented vertical slice

- Compact parameter-export ingress with availability/cold-start preservation,
  four supplied SPI classes or explicit unavailable SPI, and current global
  detector count/box invariants. No SPI prediction/training/import.
- Fifteen device/mission templates, population join/pause/resume/leave schedules,
  variable sender assignments/capture times, coherent simulated trajectories,
  causal sender novelty and fixed hidden message outcomes/durations.
- External runtime JSONL and separate optional reference JSONL; unknown truth
  remains unknown. Predeclared endpoint is part of the external run identity.
- Existing exact-once nonpreemptive service controller and causal successful
  history reused through numeric queue/context injection.
- Existing eight non-teacher learning arms integrated with source snapshots,
  checkpoints, portable runtime inference, validation and evaluation replay.
- Common fixed validation/evaluation probes, independent priority/timing and
  area metrics, outcome/accounting assertions and per-decision JSONL logs.
- External replay CLI for supplied parameter records without any classifier.

Sources: Evaluation/stream_simulator.py, Evaluation/stream_io.py,
Scheduler/training/stream_pilot.py. The focused fix in
Evaluation/scene_environment.py ensures zero-utility observations do not count
as first useful completion. No parameter-team or codec source was changed.

## Corrected local pilot actually executed

Run: experiments/stream_scheduler/runs/stream-pilot-20261029-220835/.

- CPU only, about 19 seconds for the bounded run.
- Three evaluation episodes, UE pools 4/60/100 and message counts 32/48/64.
- Three separate training worlds and separate validation/evaluation worlds.
- Three supervised epochs; 64 decisions per RL arm. This is a short integration
  pilot, not convergence training.
- Eight fixture-trained learners: linear choice, MLP choice, RankNet,
  LambdaMART, priority classifier, Masked PPO, Masked Double DQN and TD-PER.
- Six references plus those eight policies evaluated: 14 policies, 42
  policy/episode traces and 742 per-decision log rows.
- All eight portable policies and locally generated checkpoints reloaded on CPU.
- Accounting and reward-versus-independent-integration checks passed.
- Three real-data teacher arms are explicitly blocked pending qualified labels:
  RL-VLM-F adaptation, VLM-guided replay and VisionPrefer adaptation.

The run has immutable source copies and unchanged source hashes during execution.
A later external-loader provenance correction does not change these synthetic
pilot results; the saved source snapshot identifies the tested code exactly.

The initial artifact audit treated an empty LambdaMART learning.json as a failed
history check. The existing tree fitting function does not emit epoch losses;
verify its fitted trees/booster and prediction parity instead of fabricating
loss history. The final operational check records that distinction separately.

Final audit passed: 100 fitted/exported trees, 683 nontrivial leaves, and 744
saved-example prediction comparisons with maximum absolute error 2.665e-15.
Final receipt:
experiments/stream_scheduler/runs/stream-final-check-20261029-221141/FINAL_CHECKS.md.

## Verification and review

The initial focused test failure was an attempted numeric comparison against an
undefined semantic metric for an unlabeled stream. It was fixed without creating
labels. The corrected simulator suite passed 13 cases before the successful run.
Additional known-direction and actual optimizer-resume checks are in
Test/tests/test_stream_learning.py. Final receipts live under
experiments/stream_scheduler/runs/stream-check-*/.

Final executed checks: **15 passed** (13 simulator cases and two learning/resume
cases). The external one-record, no-reference replay CLI also exited 0 with
undefined J_time and zero offered reference weight. Final receipts/audits are in
experiments/stream_scheduler/runs/stream-final-check-20261029-221141/.

Bounded assessment also identified and main corrected: invalid synthetic detector
cross-fields, zero-utility useful-completion accounting, future-dependent horizon,
validation/evaluation probe mixing, and external horizon missing from provenance.
The full repository suite was not run; focused checks do not establish global
migration correctness.

An earlier run, stream-pilot-20261029-220639, is INVALIDATED and preserved because
its fixture generation violated the detector contract and code changed while it
ran. Do not compare or publish its scores.

## What this does not establish

This is a synthetic world/parameter fixture, not real camera imagery, real
mission-relative semantic extraction or an independently validated emergency
benchmark. Reference relevance and priority are constructed world variables.
Locations and timing are simulated. A 100-UE pool does not mean 100 independent
live cameras or 100 continuously backlogged senders.

The adapter consumes compact JSON exports, not the binary 35-byte header. The
payload is an opaque synthetic reference. No image reconstruction, GPU, Kaggle
submission, SPI model, Qwen inference or actual radio failure was run.

Dense student compute is a matrix-only MAC/FLOP estimate. Measured CPU selection
latency includes context build/select, but is not charged to synthetic service
time. Teacher/extractor timing and FLOPs remain unmeasured. No policy superiority
claim follows from these short fixture runs.

Conditional Deep Sets/attention, PPO warmstart, real image-stream adapters and
qualified teacher-dependent training remain outstanding. Existing teacher source
is preserved; an unqualified synthetic preference is not renamed Qwen feedback.

## Reproduce and hand off

See experiments/stream_scheduler/README.md for local commands, output layout and
external runtime/reference schemas. Use a new run directory; do not overwrite.

Next integration input: versioned compact exports plus image/profile/mission
provenance and separate assessable labels. The SPI classifier can remain external
or unavailable. Real teacher work requires a qualified backend and actual images;
it cannot be meaningfully completed from synthetic scalars alone.


## WSL execution checkpoint — 30 October 2026

User requires all subsequent execution/source edits through Ubuntu-26.04 user Satya. Same canonical repository at /mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticSchedulerEnd2End; no parallel source tree. Real mini-cache preflight completed in experiments/stream_scheduler/runs/mini-wsl-ingress-20261030-v2/REPORT.md. Verified6000 image files (5952 unique hashes), scene-wire records and387-entry received-feature export. This is cache ingress evidence, not trained scheduling or semantic superiority. Relevance/novelty remain unavailable, contextual preferences missing; SPI predictions are not reference urgency. Proposed sparse scene byte accounting is measured-from-cache arithmetic, not implemented transport. Training and the frozen D/P temporal integration remain outstanding.
