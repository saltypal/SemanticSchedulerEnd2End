# Current project map

## Classifier-independent local stream system

`docs/decoder_context/STREAM_BUILD_STATUS.md` records the new external-parameter
ingress, synthetic dynamic-UE pilot, actual checks and remaining real-data gates.
Shared source is `Evaluation/stream_simulator.py`, `Evaluation/stream_io.py` and
`Scheduler/training/stream_pilot.py`. Commands and artifact layout are in
`experiments/stream_scheduler/README.md`. No SPI classifier or producer was changed.

| Active owner | Responsibility | First files |
|---|---|---|
| Codec | Image encoding/decoding, latent adapters, codec training | runtime.py, Models/, integration/ |
| Scheduler | All existing policy implementations and scheduler training | ALGORITHMS.md, decoder_scheduler.py, scene_policy.py, training/ |
| parameters | Current BetterScheduler implementation and parameter research | README.md, minimal_parameters.py, spi.py, scene_measurements.py |
| Evaluation | Queue lifecycle, simulated execution, benchmark definitions and metrics | decoder_queue.py, decoder_service.py, decoder_semantic_benchmark.py, codec_metrics/ |

## Exact relocation

- Original `INFRA` -> `Archived/owners_before_simplification_20261027/INFRA`.
- Original `EncDec` -> `Archived/owners_before_simplification_20261027/EncDec`.
- Original `SemParam` -> `Archived/owners_before_simplification_20261027/SemParam`.
- Active codec source reused under `Codec`; reconstruction metrics moved into
  `Evaluation/codec_metrics`.
- Active infrastructure source reused under `Evaluation`; its learning package
  moved into `Scheduler/training`.
- Whole `BetterScheduler` directory moved to `parameters`, including saved
  results, notebooks, experiments, configuration and research notes.
- Existing `Scheduler` algorithms retained, with imports pointing at new owners.
- Old `src/Evaluation` compatibility package archived to avoid colliding with
  the new Evaluation package.

Some active older code still uses archived SemParam implementations. These are
explicit transitional dependencies, not a newly finished parameter system.
`project_compat.py` resolves historical module names after activation.

No notebook cell contents or saved outputs were rewritten. Historical notebooks
remain where preserved: root Notebooks, Scheduler/notebooks,
parameters/notebooks and the archived owner notebook directories. New notebooks
belong with their experiment. Runtime caches, datasets and model weights remain
in place.

## Validation boundary

The user explicitly requested no tests or experiments during this restructure.
Only filesystem moves/copies, source/configuration edits and inventory reads
were performed. No claim of passing imports or runtime integration is made.
The old full-suite failures and header-integration gap are not fixed by moving
files. Migration hashes are in
`experiments/codebase_simplification/runs/20261027_layout/migration.json`.
