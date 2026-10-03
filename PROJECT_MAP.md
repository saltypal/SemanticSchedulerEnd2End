# Project map

Read this before changing the repository. Current goal: **schedule complete image latent payloads**, with static batches as the main working path.

## Owners

| Owner | Canonical responsibility | Entry points |
|---|---|---|
| INFRA | Contracts, lifecycle, transport, persistence, configuration and learning orchestration | `latent/cli.py`, `cli.py`, `configuration.py`, `learning/` |
| EncDec | Models, communication codebooks, reconstruction and codec benchmarks | `runtime.py`, `Models/`, `Codebooks/`, `Evaluation/`, `integration/` |
| SemParam | Extraction, semantic codebooks, knowledge and serialization | `features/`, `history/`, `headers/`, `pipeline.py`, `runtime.py` |
| Scheduler | Policies over immutable snapshots | `latent/policies.py`, `baselines/policies.py`, `semantic/` |

Policies select jobs; only INFRA changes time/lifecycle/history. Lower class integers mean higher priority. Do not infer classes from filenames. Scheduler imports no codec or neural framework. Image-only SemParam methods need no codec checkpoint; codec-dependent methods receive `INFRA.features.FeatureProvider`.

## Main whole-latent path

- `INFRA/latent/contracts.py`: frozen jobs, candidates, snapshots, delivery evidence and service configuration. No UE field.
- `INFRA/latent/simulation.py`: whole-payload service, arrivals, deadlines, censoring, live/cached feedback and receiver history.
- `INFRA/latent/evaluation.py`: reference ranking and confirmed task-quality evidence remain separate.
- `INFRA/latent/experiment.py`: input validation, cache adaptation, logging and persistence.
- `Scheduler/latent/policies.py`: six static policies.
- `SemParam/features/latent.py`: image/task relevance and confirmed receiver novelty.
- `EncDec/integration/scheduler_replay.py`: tensor caching and saved-order replay.
- `EncDec/integration/latent_transport.py`: load-once live codec callback.

```powershell
python -m INFRA.latent.cli --batch Scheduler/config/static_latent_demo.json
```

Transport returns confirmed reconstruction evidence. Source embeddings are not substituted for receiver embeddings. See `docs/STATIC_LATENT_SCHEDULER.md` for methodology and evidence limits.

## Earlier experiments and compatibility

`INFRA/scheduling/contracts.py` and `simulation.py` retain the original slotted fixture. `semantic_contracts.py`, `streams.py`, `requests.py` and `event_simulation.py` retain the earlier five-UE P1-P7 experiment. `Scheduler/semantic/` retains the semantic/DQN policy interface; `INFRA/learning/` retains classifier/DQN training and optional priority-match objectives. Refactoring changes neither existing observation dimensions nor experimental rules.

`SemParam/features/scheduler.py`, `history/delivered.py` and `headers/semantic.py` own V2 features/history/header contracts. P1/P4 remain in the scheduling envelope; V2 headers serialize P2/P3/P5/P6/P7. `INFRA/data/` owns LADI manifests, feature caches and upload bundles. `INFRA/experiments/semantic_scheduler_experiment.py` is the earlier training/evaluation orchestrator.

Old flat paths, `src/` and `semcom_benchmark/` are identity-preserving compatibility aliases. Edit canonical packages. Preserve old serialized class references. Supplied reference classes are evaluator-only except declared privileged baselines. The existing dirty `learning/dqn_training.py` work was preserved during cleanup.

## Assets and notebooks

Team DeepJSCC/NTSCC benchmark code lives in `EncDec/Research/`. Pinned snapshots stay in `references/`, Git LFS identities in `models/`, shared images in `data/`, and historical outcomes in `results/`. Never change these as a refactoring side effect.

The main whole-latent guide is scheduler notebook 04. Scheduler notebooks 01/02 retain the earlier training/evaluation workflow. Teammate-local parameter/static notebooks remain intact. Root `Notebooks/` and module `notebooks/historical/` are reference material. Remote `8073b75`'s LADI notebook schedules JPEG chunks, not the canonical learned latents.

## Validation and provenance

Run owning-module tests, affected integration checks and full regressions. Keep imports lazy, reject invalid configuration before execution and fail unsupported accelerators explicitly. Utilities belong to their group under `scripts/`; flat utility paths are compatibility launchers.

New outputs use configured `results/<module>/<run-id>`, retaining config, commit, dirty state, source digest, environment and measurement outcomes. Keep storage bytes, bits and complex channel uses distinct. Missing evidence stays unavailable; failed reconstructions have no quality score. Hardware comparisons require matching provenance.

Refresh Graphify with `python scripts/refresh_graph.py --semantic docs/architecture_semantics.json`. It excludes compatibility aliases and records exact working-source hashes; HEAD alone does not describe a dirty tree. Do not claim checkpoint inference, training convergence or research novelty from fixture tests.

See `docs/CODEBASE_CLEANUP.md`, `docs/MIGRATION_MAP.md`, `docs/NOTEBOOK_MIGRATION.md` and `docs/README.md` for migration and navigation.
