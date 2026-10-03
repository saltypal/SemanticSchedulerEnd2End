# Codebase cleanup: 2026-10-02

The project keeps four major owners: INFRA, EncDec, SemParam and Scheduler.
Complete image latent scheduling is the primary documented workflow.

## Changes

- Grouped 29 existing Python implementations into explicit lifecycle, data,
  policy, feature, history and header packages; moved the static CLI into INFRA.
- Grouped 36 remaining Python utilities by purpose. In total, 66 source/utility
  paths received canonical destinations and thin compatibility aliases.
- Updated 13 canonical consumers to import their actual owner packages.
  Existing dirty DQN work was preserved; it can still use the old imports.
- Added a `latent-scheduler` console command and package discovery for utility
  packages. Included JSON configuration fixtures in package data.
- Replaced outdated root architecture documentation, added package/document
  indexes, and recorded this complete migration map.
- Ignored generated downloads, Kaggle staging and scratch/browser artifacts.
  Nothing in those directories was deleted or moved.
- Made SemParam public exports lazy while preserving API/class identity, so
  cached static scheduling does not eagerly import the earlier extraction pipeline.
- Excluded compatibility aliases from Graphify source extraction to avoid
  indexing the same implementation twice. Exact source hashes remain recorded.

## Preserved behavior and material

Old imports resolve to the same module and class objects as their canonical
counterparts. Old pickle class references remain resolvable. Old flat script
commands remain usable, including from an unrelated working directory.
Aliases contain no independent policy, extraction, simulation or codec logic.

Datasets, model assets, pinned source, historical results, downloaded runtime
snapshots, root notebooks and teammate-local notebooks stay at their original
paths. Scheduler policy rules, feature meanings, DQN dimensions and checkpoint
identities were not changed by restructuring. No training, dataset upload or
remote push is part of this cleanup.

Original bytes, hashes and consumer backups were captured under
`results/cleanup/20261002T175037Z/`. A Windows line-ending issue in initially
written destination files was corrected from those originals. The relocation
and import edits do not claim model performance or research novelty.

## Canonical commands

```powershell
python -m INFRA.latent.cli --batch Scheduler/config/static_latent_demo.json
python -m scripts.notebooks.build_static_latent_notebook
python -m pytest -q -p no:cacheprovider
python scripts/refresh_graph.py --semantic docs/architecture_semantics.json
```

The previous `python scripts/run_static_latent_scheduler.py` command is a
compatibility launcher. Do not add new code to flat aliases or to `src/`.
Final checks: **147 tests passed, 3 skipped** (CUDA/checkpoint prerequisites);
all static notebook cells executed; the built wheel ran its own static source
in an isolated directory. Cached static scheduling also passed with optional
site-packages disabled. Removed 86 generated cache directories and preserved
108 recorded protected files with zero unexpected changes. Full evidence is
in `results/cleanup/20261002T175037Z/validation.json`.

## Migration map

| Previous path (compatibility) | Canonical implementation |
|---|---|
| `INFRA/contracts.py` | `INFRA/scheduling/contracts.py` |
| `INFRA/semantic_contracts.py` | `INFRA/scheduling/semantic_contracts.py` |
| `INFRA/scheduling_requests.py` | `INFRA/scheduling/requests.py` |
| `INFRA/semantic_streams.py` | `INFRA/scheduling/streams.py` |
| `INFRA/simulation.py` | `INFRA/scheduling/simulation.py` |
| `INFRA/event_simulation.py` | `INFRA/scheduling/event_simulation.py` |
| `INFRA/semantic_evaluator.py` | `INFRA/evaluation/semantic.py` |
| `INFRA/feature_cache.py` | `INFRA/data/feature_cache.py` |
| `INFRA/ladi_manifest.py` | `INFRA/data/ladi_manifest.py` |
| `INFRA/kaggle_dataset_bundle.py` | `INFRA/data/kaggle_bundle.py` |
| `INFRA/latent_contracts.py` | `INFRA/latent/contracts.py` |
| `INFRA/latent_simulation.py` | `INFRA/latent/simulation.py` |
| `INFRA/latent_evaluation.py` | `INFRA/latent/evaluation.py` |
| `INFRA/experiments/static_latent_experiment.py` | `INFRA/latent/experiment.py` |
| `scripts/run_static_latent_scheduler.py` | `INFRA/latent/cli.py` |
| `Scheduler/policies.py` | `Scheduler/baselines/policies.py` |
| `Scheduler/fixtures.py` | `Scheduler/baselines/fixtures.py` |
| `Scheduler/semantic_policies.py` | `Scheduler/semantic/policies.py` |
| `Scheduler/feature_adapter.py` | `Scheduler/semantic/features.py` |
| `Scheduler/policy_registry.py` | `Scheduler/semantic/registry.py` |
| `Scheduler/latent_policies.py` | `Scheduler/latent/policies.py` |
| `SemParam/clip_backbone.py` | `SemParam/features/clip.py` |
| `SemParam/extractors.py` | `SemParam/features/extractors.py` |
| `SemParam/observation_builder.py` | `SemParam/features/observation.py` |
| `SemParam/scheduler_features.py` | `SemParam/features/scheduler.py` |
| `SemParam/ladi_prediction_provider.py` | `SemParam/features/ladi_prediction.py` |
| `SemParam/latent_features.py` | `SemParam/features/latent.py` |
| `SemParam/delivered_history.py` | `SemParam/history/delivered.py` |
| `SemParam/semantic_header.py` | `SemParam/headers/semantic.py` |
| `SemParam/packet_builder.py` | `SemParam/headers/packet.py` |
| `scripts/bootstrap_swinjscc.py` | `scripts/codecs/bootstrap_swinjscc.py` |
| `scripts/download_kodak24.py` | `scripts/codecs/download_kodak24.py` |
| `scripts/register_deepjscc.py` | `scripts/codecs/register_deepjscc.py` |
| `scripts/render_kodak_benchmark_table.py` | `scripts/codecs/render_kodak_benchmark_table.py` |
| `scripts/run_communication_benchmark.py` | `scripts/codecs/run_communication_benchmark.py` |
| `scripts/run_deepjscc.py` | `scripts/codecs/run_deepjscc.py` |
| `scripts/run_ntscc_benchmark.py` | `scripts/codecs/run_ntscc_benchmark.py` |
| `scripts/test_deepjscc_registry.py` | `scripts/codecs/test_deepjscc_registry.py` |
| `scripts/build_ladi_review_page.py` | `scripts/data/build_ladi_review_page.py` |
| `scripts/curate_ladi260.py` | `scripts/data/curate_ladi260.py` |
| `scripts/curate_ladi60.py` | `scripts/data/curate_ladi60.py` |
| `scripts/freeze_ladi60_prototype.py` | `scripts/data/freeze_ladi60_prototype.py` |
| `scripts/package_ladi260_dataset.py` | `scripts/data/package_ladi260_dataset.py` |
| `scripts/package_ladi_review_dataset.py` | `scripts/data/package_ladi_review_dataset.py` |
| `scripts/build_downloaded_run_review.py` | `scripts/notebooks/build_downloaded_run_review.py` |
| `scripts/build_parameter_extraction_demo_notebook.py` | `scripts/notebooks/build_parameter_extraction_demo_notebook.py` |
| `scripts/build_semantic_scheduler_notebooks.py` | `scripts/notebooks/build_semantic_scheduler_notebooks.py` |
| `scripts/build_stage1a_notebook.py` | `scripts/notebooks/build_stage1a_notebook.py` |
| `scripts/build_stage1b_notebooks.py` | `scripts/notebooks/build_stage1b_notebooks.py` |
| `scripts/build_static_latent_notebook.py` | `scripts/notebooks/build_static_latent_notebook.py` |
| `scripts/build_scheduler_explanation_doc.py` | `scripts/reports/build_scheduler_explanation_doc.py` |
| `scripts/plot_dynamic_priority_matches.py` | `scripts/reports/plot_dynamic_priority_matches.py` |
| `scripts/plot_stage1a_four_way_comparison.py` | `scripts/reports/plot_stage1a_four_way_comparison.py` |
| `scripts/plot_stage1a_snr_psnr_ssim_cbr.py` | `scripts/reports/plot_stage1a_snr_psnr_ssim_cbr.py` |
| `scripts/plot_static_dynamic_scheduler_comparison.py` | `scripts/reports/plot_static_dynamic_scheduler_comparison.py` |
| `scripts/prepare_semantic_kaggle.py` | `scripts/kaggle/prepare_semantic_kaggle.py` |
| `scripts/prepare_static_latent_kaggle.py` | `scripts/kaggle/prepare_static_latent_kaggle.py` |
| `scripts/package_semantic_payload_costs.py` | `scripts/kaggle/package_semantic_payload_costs.py` |
| `scripts/package_semantic_train_output.py` | `scripts/kaggle/package_semantic_train_output.py` |
| `scripts/evaluate_downloaded_scheduler.py` | `scripts/scheduler/evaluate_downloaded_scheduler.py` |
| `scripts/evaluate_dynamic_overload_frozen_dqn.py` | `scripts/scheduler/evaluate_dynamic_overload_frozen_dqn.py` |
| `scripts/evaluate_fixed_decision_matches.py` | `scripts/scheduler/evaluate_fixed_decision_matches.py` |
| `scripts/replay_static_latent_schedule.py` | `scripts/scheduler/replay_static_latent_schedule.py` |
| `scripts/train_class_priority_dqn.py` | `scripts/scheduler/train_class_priority_dqn.py` |
| `scripts/train_parameter_codebook.py` | `scripts/semparam/train_parameter_codebook.py` |
| `scripts/refresh_graph.py` | `scripts/maintenance/refresh_graph.py` |
