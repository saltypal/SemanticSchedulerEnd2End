# Notebook migration

Active module notebooks are top-level files under each module/notebooks. Historical copies are reference-only and retain original outputs.

On 2026-09-27 the superseded `Scheduler/notebooks/01_compare_image_classifiers.ipynb` and `Scheduler/notebooks/02_compare_traditional_schedulers.ipynb` were moved byte-for-byte into `Archived/image_scheduling_pilot_20260927/`, preserving their original relative paths. Their pilot-only `SemParam/priority_models.py` and `scripts/download_priority_models.py` were archived alongside them. `manifest.json` records SHA256 identities; commit `45738cb` and local branch `codex/archive-image-scheduling-20260927` preserve the original executable layout. No active canonical module imports these pilot adapters.

The current Track-A branch adds two canonical notebooks:
`Scheduler/notebooks/01_train_semantic_scheduler.ipynb` and
`Scheduler/notebooks/02_evaluate_all_policies.ipynb`. They import reusable
module code, clean-clone the latest remote branch, and require a reviewed LADI
manifest. `scripts/build_semantic_scheduler_notebooks.py` regenerates them;
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
