# Notebook migration

Active module notebooks are top-level files under each module/notebooks. Historical copies are reference-only and retain original outputs.

On 2026-09-27 the superseded `Scheduler/notebooks/01_compare_image_classifiers.ipynb` and `Scheduler/notebooks/02_compare_traditional_schedulers.ipynb` were moved byte-for-byte into `Archived/image_scheduling_pilot_20260927/`, preserving their original relative paths. Their pilot-only `SemParam/priority_models.py` and `scripts/download_priority_models.py` were archived alongside them. `manifest.json` records SHA256 identities; commit `45738cb` and local branch `codex/archive-image-scheduling-20260927` preserve the original executable layout. No active canonical module imports these pilot adapters.

The current Track-A branch adds two canonical notebooks:
`Scheduler/notebooks/01_train_semantic_scheduler.ipynb` and
`Scheduler/notebooks/02_evaluate_all_policies.ipynb`. They import reusable
module code, clean-clone the latest remote branch, and require a prototype-fixed
LADI manifest. Notebook 1 includes all policy evaluation cells after its
two-hour capped DQN training; notebook 2 supports later independent reruns.
`scripts/build_semantic_scheduler_notebooks.py` regenerates them;
`scripts/prepare_semantic_kaggle.py` stages private Kaggle CLI copies and
metadata. See `SEMANTIC_SCHEDULER_RUNBOOK.md` for commands and data gates.

- `Notebooks\Kodak24_All_Model_Benchmark.ipynb` → `EncDec\notebooks\historical\Kodak24_All_Model_Benchmark.ipynb` (historical; original preserved)
- `Notebooks\Nandini_DeepJSCC_NTSCC_Integration.ipynb` → `EncDec\notebooks\historical\Nandini_DeepJSCC_NTSCC_Integration.ipynb` (historical; original preserved)
- `Notebooks\notebookbf0f6f4129.ipynb` → `EncDec\notebooks\historical\notebookbf0f6f4129.ipynb` (historical; original preserved)
- `Notebooks\Stage1A\01_stage1a_complete_kaggle.ipynb` → `EncDec\notebooks\historical\Stage1A\01_stage1a_complete_kaggle.ipynb` (historical; original preserved)
- `Notebooks\Stage1A\SwinJSCC_Practical_Inspection.ipynb` → `EncDec\notebooks\historical\Stage1A\SwinJSCC_Practical_Inspection.ipynb` (historical; original preserved)
- `Notebooks\Stage1A\SwinJSCC_Practical_Inspection_Local.ipynb` → `EncDec\notebooks\historical\Stage1A\SwinJSCC_Practical_Inspection_Local.ipynb` (historical; original preserved)
- `Notebooks\Stage1B\02_transformer_vs_codebook_semantic_header.ipynb` → `SemParam\notebooks\historical\Stage1B\02_transformer_vs_codebook_semantic_header.ipynb` (historical; original preserved)
- `Notebooks\Stage1C\01_train_parameter_codebook_local.ipynb` → `SemParam\notebooks\historical\Stage1C\01_train_parameter_codebook_local.ipynb` (historical; original preserved)
- `Notebooks\SwinJSCC_DIV2K_Kodak_T4x2_WITH_CODEBOOK.ipynb` → `EncDec\notebooks\historical\SwinJSCC_DIV2K_Kodak_T4x2_WITH_CODEBOOK.ipynb` (historical; original preserved)
- `Notebooks\SwinJSCC_Kaggle_T4x2_DataParallel_Training.ipynb` → `EncDec\notebooks\historical\SwinJSCC_Kaggle_T4x2_DataParallel_Training.ipynb` (historical; original preserved)
- `Notebooks\SwinJSCC_Official_DIV2K_Kodak_Kaggle_T4x2_DataParallel_FINAL.ipynb` → `EncDec\notebooks\historical\SwinJSCC_Official_DIV2K_Kodak_Kaggle_T4x2_DataParallel_FINAL.ipynb` (historical; original preserved)
# Optional class-order DQN experiment

The teammate's `8073b75` notebook remains at
`Notebooks/04_LADI_v2_End_to_End_Semantic_Channel_Scheduling.ipynb` as an
upstream JPEG/chunk reference. Its classifier and simulated channel experiment
are separate from the canonical whole-latent notebook. See
`docs/MERGE_8073B75.md`; no automatic relocation or algorithm replacement was
performed during the merge.

The device-independent static path is
`Scheduler/notebooks/04_static_latent_scheduler.ipynb`. It imports canonical
modules, resets only a disposable cloud checkout, persists outputs separately,
and compares complete latent payloads. The teammate's LADI static notebook is
preserved as reference; its rubric-derived order is not a deployment input.

`scripts/train_class_priority_dqn.py` is a separate local entry point for a
prescribed-class ordering objective. It reuses the cached 260-image features
and the same 559-input DQN runtime, but selects checkpoints by validation
priority-pair match rate instead of semantic-utility AUC. The normal training
notebook and its existing checkpoints retain their original objective. Do not
compare the two checkpoint objectives as if they were the same experiment.


# Canonical package cleanup (2026-10-02)

The static notebook generator now imports INFRA.latent and Scheduler.latent.
Existing teammate/static/extraction and earlier training notebooks remain intact;
old imports resolve through identity-preserving aliases. See CODEBASE_CLEANUP.md.
