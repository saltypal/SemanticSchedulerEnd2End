# Semantic image decoder scheduler

One codebase. Four active owners.

```text
SemanticSchedulerEnd2End/
  Codec/          image encoders, decoders and latent adapters
  Scheduler/      scheduling algorithms and training implementations
  parameters/     SPI, relevance, novelty, capture and scene extraction
  Evaluation/     receiver queue, decoder service, benchmarks and metrics
  experiments/    new shared experiments and run records
  Notebooks/      preserved team notebooks
  Archived/       old INFRA, EncDec and SemParam implementations
  data/           existing datasets (preserved)
  models/         existing checkpoints (preserved)
  results/        historical results (preserved)
```

## Flow

Image + UE mission + sender history -> parameters.
Image -> Codec encoder -> complete latent payload.
Received [parameter header][complete latent] -> Evaluation receiver queue ->
Scheduler selects one job -> Codec decoder -> Evaluation records outcome.

The current 35-byte parameter header is not yet integrated with the receiver
queue. The existing SPI head supports Low, High and Critical; Medium remains
unresolved. Parameter extraction will be revised next. Reorganization does not
change these research limitations.

## Where to work

- `parameters/minimal_parameters.py`: current extractor and compact header.
- `parameters/run_parameters.ps1`: parameter launcher; relocated model path.
- `parameters/PLAN.md`: existing parameter research decision history.
- `Scheduler/ALGORITHMS.md`: algorithm implementations and their requirements.
- `Scheduler/training/`: RankNet, trees, PPO/DQN and other training code.
- `Evaluation/decoder_queue.py` and `decoder_service.py`: receiver lifecycle.
- `Evaluation/decoder_semantic_benchmark.py`: distinct-evidence coverage metrics.
- `Evaluation/codec_metrics/`: image/codec reconstruction metrics.
- `Codec/Models/`: existing codec implementations.

## Execution policy for this change

No tests, inference, training, dependency installs or Kaggle jobs were run.
Runtime/import compatibility of this layout remains unverified. Historical test
results describe the old layout only. Existing virtual-environment console
scripts may require reinstalling the editable package before future use.

All original notebooks and outputs were preserved. Old notebook imports can be
resolved by explicitly calling `import project_compat; project_compat.install()`
from a setup cell at the project root. This compatibility path has not been run.
New code must use the four canonical owners directly.

## Kaggle

Update the existing owner/slug with a new notebook version. Do not make a new
notebook identity for every experiment revision. Historical slugs and run IDs in
`parameters/KAGGLE_RUNS.md` remain unchanged; relocated files are under
`parameters/`. No remote notebook was submitted or stopped during this change.

See `PROJECT_MAP.md` and `experiments/codebase_simplification/PLAN.md`.

## Four-class SPI execution

The active plan is parameters/PLAN.md and the notebook/run map is
parameters/KAGGLE_RUNS.md. The parameter launcher now rejects a selected
three-class fallback by default; historical comparison needs explicit
-AllowHistoricalThreeClass. Local source includes this safeguard while the
current remote version remains the exact frozen snapshot recorded in its run.
Do not infer completed training from a running notebook or local regression tests.
