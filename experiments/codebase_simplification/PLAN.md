# Approved codebase simplification

Objective: replace active INFRA/EncDec/SemParam/BetterScheduler ownership with
Codec/Scheduler/parameters/Evaluation within SemanticSchedulerEnd2End.

User directly authorized the move and explicitly prohibited tests/experiments.

Completed filesystem/source work:
1. Move exact original owners to Archived/owners_before_simplification_20261027.
2. Reuse codec and infrastructure source under Codec and Evaluation.
3. Move all BetterScheduler material to parameters without rerunning extraction.
4. Move scheduler training from infrastructure into Scheduler/training.
5. Keep all notebooks/results, update imports, packaging, launcher and model path.
6. Document active owners and algorithm inventory; retain historical Kaggle IDs.

Manifest: runs/20261027_layout/migration.json. It records archived file hashes,
source edits and absence of runtime validation. Archived pre_edit/ contains
pre-rewrite active sources. Do not rerun reorganize.py: it intentionally refuses
existing run/archive destinations.

Not performed: tests, import checks, training, parameter redesign, codec runs,
Kaggle operations, remote pushes, new-header queue integration.

Remaining: revise SPI parameters, define one policy input adapter, repair
historical missing modules/artifacts if needed, then validate under new approval.
