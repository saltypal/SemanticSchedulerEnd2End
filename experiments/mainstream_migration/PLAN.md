# Mainstream scheduler promotion

Approved on 5 November 2026. Promote reusable scheduler drivers and orchestration
without changing model behavior, datasets, objective, recorded metrics or remote runs.

## Ownership

- `Scheduler/training/stream_suite`: data adapters, training drivers, qualification and profiling utilities.
- `Scheduler/training/study`: paired orchestration, builders and reports.
- `Evaluation`: existing simulator, queue and metric implementation.
- `Scheduler`: existing models and deployed inference.
- `Notebooks/SchedulerSuite`: active notebook sources and build metadata.
- `experiments`: previous notebooks, compatibility entry points, historical results and this migration evidence.

## Compatibility

Old Python imports alias the canonical module object so callers and monkeypatches
share state. Old file commands execute the canonical source through `runpy`.
Every promoted module resolves the same project root. Source-only modules with
unchanged AST preserve their original bytes where possible. Notebook capsules
include all promoted sources. Existing account metadata is copied before building.

## Checks

Run the existing focused scheduler and orchestration tests, new import/CLI/capsule
regressions and fresh-process command checks. Compile promoted sources and notebook
code cells. Check canonical imports do not depend on old implementation locations.
Inspect capsule hashes and account identities. No training, model inference,
dataset download or Kaggle submission is part of this migration.

## Preservation

Historical run trees and model files are not moved. Windows Git may falsely report
long-path artifacts as deleted without `core.longpaths=true`. Do not stage them.
Commit only reviewed migration paths. Preserve other users' changes.

The first regression run had 105 passes and two path-related test failures. One
test imported another test through a mechanically replaced package name. Another
hashed the compatibility wrapper instead of the actual renderer. Correct their
locations without weakening their assertions, then rerun.
