# Local decoder scheduling system without an SPI classifier

This experiment reuses Scheduler/training/scene_learners.py and the existing
single-decoder completion controller. It does not train/import the active SPI
classifier, modify parameters/, run a codec, or download/load Qwen.

## Components

- Evaluation/stream_simulator.py: fifteen UE profiles, seeded synthetic offered
  streams, capture/position metadata, churn schedules, fixed random outcomes,
  external compact-parameter validation and lifecycle integration.
- Evaluation/stream_io.py: external runtime JSONL plus a separate optional
  reference JSONL; no guessed labels when references are absent.
- Scheduler/training/stream_pilot.py: collection, eight fixture learner updates,
  portable models, checkpoint saves, common rollout/probe evaluation and logs.
- Existing Scheduler/training/scene_teacher*.py: teacher backend, feedback and
  reward-model machinery. Three teacher-dependent registry arms stay blocked
  for real-data use until qualified VLM feedback is supplied.

The numeric adapter matches the currently exported compact v2 fields. Its scene
fields are global detector measurements, not task-filtered counts. It is JSON
interchange, not binary 35-byte-header integration. SPI can be supplied externally
or explicitly unavailable. The generator supplies noisy synthetic classes only;
it is not an SPI classifier.

## Local pilot

From the canonical SemanticSchedulerEnd2End repository root:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv/Scripts/python.exe -m Scheduler.training.stream_pilot --output experiments/stream_scheduler/runs/<unique-run-id> --episodes 3 --epochs 3 --rl-steps 64
```

Use the existing project environment. No new installation is required by this
entrypoint. It records the actual interpreter, packages, CPU platform, source
snapshot/hashes, config and dirty state. Three evaluation fixtures use UE pools
4/60/100, with respectively 32/48/64 offered messages. These are not 100
simultaneously streaming real cameras. Profiles and activity intervals are
generated; coherent real RGB chronology and mission interpretation are not
validated by synthetic scalar evidence.

Training uses three separate synthetic worlds, and model selection uses a fourth
validation world. Evaluation worlds and shared ranking probes have different
world identities. Tiny training budgets establish integration and optimization
updates, not convergence or generalization.

## Outputs

Each new run saves:

- manifest.json and source/: exact code/environment identity.
- data/: full generated streams, hidden outcomes/references, metadata and all
  training/validation/evaluation decision examples.
- profiles.json: actual fifteen device/mission templates.
- run.log, status.json, learner_status.json: progress, failures and blocked arms.
- models/<arm>/: learning.json, trusted-local resume.pt and portable.json.
- logs/episode_<n>/<policy>.jsonl: every decision observation, chosen identity,
  score/fallback, service outcome/reward, timing and next-ready IDs.
- metrics/: independent area/priority/delay/accounting and shared ranking probes.
- comparison.json and REPORT.md: result table with scope and limitations.

Outer process stdout/stderr are also captured by the check runner when run via
the operational delegate. Within normal manual execution Python logs exceptions
to stderr; preserve the terminal output if the process fails. Incremental model
and learner files survive later stages failing, but a killed process can still
lose its most recent buffered output.

Failures are fixed per message across policies; no automatic retry. Successful
history excludes failed jobs. Horizon-unfinished work and queued work are counted
separately. Runtime models never receive the evaluator facts or future outcomes.
Priority caps are evaluation budgets, not real emergency deadlines. Unknown
reference truth yields undefined semantic metrics.

## Real parameter handoff

Use Evaluation.stream_io.load_offered_stream(runtime_path, reference_path=None,
observation_end=..., delay_cap=...). Each runtime JSONL record contains:

```json
{
  "schema": "stream_received_v1",
  "source_id": "ue-1",
  "session_id": "session-1",
  "message_token": 1,
  "sequence": 1,
  "arrival_time": 1.5,
  "payload_ref": "opaque-reference",
  "parameters": {
    "schema": "better_minimal_parameters_v2",
    "spi": null,
    "spi_available": false,
    "relevance": 0.2,
    "novelty": 0.4,
    "novelty_cold_start": false,
    "capture_time_ms": 1793232000000,
    "allowed_delay_ms": null,
    "scene": {
      "instances": 0,
      "categories": 0,
      "largest_box_fraction": null,
      "union_box_coverage": 0
    }
  }
}
```

An optional separate reference record has exactly message_id, fact_id, priority
(0 Critical through 3 Normal), and nonnegative weight. message_id is the canonical
JSON string [source_id,session_id,message_token]. All joins are checked. References
must have independently audited provenance before scientific claims. The loader
does not invent a dataset split or urgency labels from predictions. It uses a
predeclared horizon and explicitly synthetic service/failures.

Replay external records from the repository root, for example:

```powershell
.venv/Scripts/python.exe -m experiments.stream_scheduler.replay_external --runtime <runtime.jsonl> --references <references.jsonl> --output experiments/stream_scheduler/runs/<unique-id> --observation-end 60 --policy spi
```

Omit references to perform an unlabeled lifecycle replay with null semantic
metrics. Use --model <portable.json> instead of a rule for a matching registered
artifact. This command never fits parameters or predicts SPI.

## Current limitations and next work

The consolidated master plan is
`docs/decoder_context/SEMANTIC_RANKING_RESEARCH_DESIGN.md`. The existing pilot
below remains a legacy completion-coverage regression. It does not implement
the planned image/fact dispatch-wait objectives or the new Deep Sets scorer.

`check_waiting_spec.py --output <unique-directory>` provides independent
arithmetic fixtures for the planned waiting cost, including unfinished units,
mid-service releases, exact repeats and a greedy counterexample. Passing this
experiment validates its specification examples only, not production wiring.

The first pilot is synthetic plumbing, not real-image semantic validation.
Actual capture-to-header extraction, real source images, Qwen qualification,
independent fine-grained image references and physical decoder integration are
not established by it. Teacher-dependent models are implemented in the reused
registry but not claimed trained here. Conditional Deep Sets/attention and PPO
warmstart remain later plan items, not secretly substituted implementations.
No measured teacher/extractor FLOPs or latency is fabricated. Student dense
matrix FLOPs are explicitly scoped MAC estimates, and selection milliseconds
are separate measured CPU values.

The invalidated early pilot stream-pilot-20261029-220639 is preserved with its
failure context; use only the subsequent corrected-source run identified in the
implementation status. Never copy its scores into a final comparison.


## v2.2 complete-method implementation and bounded execution

Canonical design: docs/decoder_context/SEMANTIC_RANKING_RESEARCH_DESIGN.md.
The source of truth remains this repository; packaged Kaggle sources are pinned runtime snapshots.

- train_real_suite.py now invokes the ten supervised and four annotation/silver-D RL methods, with explicit device, resume, decision and wall-time options. Three feedback methods retain visible label-dependent statuses.
- prepare_teacher.py / teacher_pipeline.py construct independently specified person-presence, duplicate and blank controls plus causal candidate/prefix/replay requests. Person controls do not certify emergency-priority correctness.
- qwen_teacher.py implements the pinned Qwen2.5-VL-7B NF4 backend, content/revision-bound response cache and qualification gate; no VLM is in live policy selection.
- train_teacher_rankers.py consumes accepted swapped preferences for P students and aspect preferences for VisionPrefer.
- train_feedback_rl.py fits and qualifies a duration-aware reward model, or a bounded replay guide, and invokes the same DDQN learner.
- run_full_suite.py supervises those paths; run_guarded.py supplies process-group timeout and streamed logs. Each method's actual status, not notebook COMPLETE, defines progress.
- complete_vectors.py fills only missing compatible CLIP vectors; required input preparation fails closed. Existing cache/classifier/detector outputs are not replaced.
- profile_exports.py measures exported CPU selection and partial supported-operator FLOPs. Its current thread statement pins Torch only; native NumPy/BLAS attribution needs verification before publication.

Local evidence includes tree threshold parity repair,8focused tests, DDQN/PER/PPO resume equivalence, batched score/gradient parity, reward integrals, dynamic UE/session behavior and common-probe NDCG. The tiny real-cache pilot evaluated14methods and explicitly blocked three without teacher labels. These are not convergence or independent urgency results.

First remote v3 was stopped after a packaged producer-import dependency prevented complete embedding preparation. Corrected v4 is tracked under runs/full-v22-launch-20261031T003706Z and parameters/KAGGLE_RUNS.md. Never mix its model/source/artifacts with v3. Actual remote learning/qualification must be verified from logs and retrieved artifacts.

Known remaining protocol work must stay visible: adaptive second-round query generation/reward-update orchestration is not yet integrated (the current feedback collection is one fixed offline batch); natural-video and physical-object correspondence require qualified sources; extra seeds, channel-subset studies and broader robustness comparisons remain budget-dependent. No codebook, grant, retry or actual codec path was added. Existing synthetic-slot timing is not measured end-to-end radio latency.


## End-to-end repair validation (2026-11-01)

Run the complete active local test setup in the existing WSL environment from the canonical repository:

```bash
/home/Satya/.venvs/semantic-scheduler/bin/python experiments/stream_scheduler/validate_system.py
```

The runner executes `parameters/tests`, `Scheduler/tests`, `Test/tests` and
`experiments/stream_scheduler/tests`, plus the independent waiting-cost reference.
It saves exact commands, interpreter/package versions, source hashes, JUnit,
per-stage logs and source-drift checks in one unique run directory. It does not
load vision weights or silently turn skipped codec/CUDA tests into passes.

Repair evidence and decisions: `runs/debug-e2e-20261101/`. Its real-cache pilot
uses checksum-verified cached image/mission vectors and runs all14 non-teacher
methods at fixture/pilot scale. Teacher distillation is tested with explicitly
synthetic preferences and separately replayed historical rejected labels.

The canonical Kaggle notebook currently selects **targeted technical follow-up**,
clearly printed in its title/configuration. The preserved version5 already ran
all17 optimization mechanisms and32 narrow Qwen controls; the follow-up fixes
the channel dependency closure and finishes five remaining format queries.
This is not the20,000-decision scientific training run.
The full supervisor remains `run_full_suite.py`; do not interpret the validation
notebook as completing that full training objective. Original v4 notebook and
all results remain preserved under the repair run. GPU inference results must
be retrieved and checked before promoting the teacher output fix.

The teacher runtime dependency set is `requirements-teacher.txt`. Grammar
constraints enforce output structure; task-specific qualification, swapped
preferences, abstentions and independent evaluation remain required.
