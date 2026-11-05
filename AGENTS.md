# Canonical project ownership

The current explicit user-approved layout replaces earlier owner names.

- Codec/: encoders, decoders and latent adapters.
- Scheduler/: scheduling algorithms and their training code.
- parameters/: current BetterScheduler extraction and parameter research.
- Evaluation/: queue execution, benchmarking and metrics.
- Archived/: preserved INFRA, EncDec and SemParam originals. Do not edit them.

Read README.md and PROJECT_MAP.md first. Preserve user/teammate work, model
weights, datasets, notebook outputs and historical results. New code uses the
four active owners. Historical import support is in project_compat.py.

For this reorganization the user explicitly requested no tests, experiments or
model inference. Do not claim runtime validation from file operations.

Use existing Kaggle owner/slug and publish notebook versions, not a new notebook
identity per run. Preserve all notebooks, outputs, version/run IDs and the
existing parameters/KAGGLE_RUNS.md record. Obtain explicit scope before creating
a genuinely different remote notebook purpose.
