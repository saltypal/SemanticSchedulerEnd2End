# v15 EfficientNet-B1 comparison package (staged only)

- Canonical notebook: `experiments\spi_classifier_repair\notebooks\notebook9_efficientnet_b1_comparison.ipynb`
- Existing Kaggle identity: `satyapaladugu/betterscheduler-spi-20261027finalc` (ID `141089219`); intended next version 15, not submitted.
- Scope: image-only EfficientNet-B1 transfer-learning comparison using the verified v13 fixed 300-image cohort and exact v13 labels. No teacher calls, cache promotion, parameter release, or runtime promotion.
- Runtime: existing Kaggle two-T4 allocation required and asserted; training on `cuda:0`; preinstalled Torch/TorchVision pair is recorded and never upgraded. The notebook installs `scikit-learn==1.7.2` and `nbformat==5.10.4` into the active kernel (`--no-deps`, 180 s timeout) before importing scikit-learn.
- Budgets: Kaggle timeout 1,740 s; one GPU training budget 1,500 s; maximum aggregate 1 GPU-hour including the brief v14 startup; max 20 epochs.
- Input manifest SHA-256: `b633e716aa20a36d7f5581dce49bffebc260516ba5ecf78a1ec0ee83ddd92be0`. Label source SHA-256: `1f5b3d78124d79c6fd43bb87ba2e7e94a0c18852bc45ca020aa550b40c7c75a5`.
- Package ZIP SHA-256: `ad58e74c5703be282accc8dfa9358fea7a1e0008ef78277cc573b1f122fdadcf`; bytes 329008; 92 hashed source/data entries.
- Expected remote outputs: `/kaggle/working/experiments/spi_classifier_repair/runs/notebook9_20261028_v15/`, incremental `cell_evidence/`, `efficientnet_b1/selection_history.json`, `refit_history.json`, `validation.json`, `predictions.json`, `candidate.pt`, and `/kaggle/working/run_recovery.zip`.
- This v15 package includes the root `project_compat.py` dependency omitted from v14. The v14 package remains preserved in `experiments/spi_classifier_repair/runs/notebook9_20261028_v14/submission_v14_cnn_compare_r2/`. Submission is pending main clearance after the final Medium review. This package has not been submitted.


- Includes root `project_compat.py`, required by `parameters/__init__.py`; isolated package import validation is required before launch.
