# v14 EfficientNet-B1 comparison package (staged only)

- Canonical notebook: `experiments\spi_classifier_repair\notebooks\notebook9_efficientnet_b1_comparison.ipynb`
- Existing Kaggle identity: `satyapaladugu/betterscheduler-spi-20261027finalc` (ID `141089219`); intended next version 14, not submitted.
- Scope: image-only EfficientNet-B1 transfer-learning comparison using the verified v13 fixed 300-image cohort and exact v13 labels. No teacher calls, cache promotion, parameter release, or runtime promotion.
- Runtime: existing Kaggle two-T4 allocation required and asserted; training on `cuda:0`; preinstalled Torch/TorchVision pair is recorded and never upgraded. The notebook installs `scikit-learn==1.7.2` and `nbformat==5.10.4` into the active kernel (`--no-deps`, 180 s timeout) before importing scikit-learn.
- Budgets: Kaggle timeout 1,800 s; one GPU training budget 1,500 s; maximum aggregate 1 GPU-hour; max 20 epochs.
- Input manifest SHA-256: `b633e716aa20a36d7f5581dce49bffebc260516ba5ecf78a1ec0ee83ddd92be0`. Label source SHA-256: `1f5b3d78124d79c6fd43bb87ba2e7e94a0c18852bc45ca020aa550b40c7c75a5`.
- Package ZIP SHA-256: `d16dd423b13c63681d728313410dd61da90831a36879ef6b4a8ab02e76eef5ba`; bytes 327870; 91 hashed source/data entries.
- Expected remote outputs: `/kaggle/working/experiments/spi_classifier_repair/runs/notebook9_20261028_v14/`, incremental `cell_evidence/`, `efficientnet_b1/selection_history.json`, `refit_history.json`, `validation.json`, `predictions.json`, `candidate.pt`, and `/kaggle/working/run_recovery.zip`.
- Submission is blocked pending Medium review and main launch clearance.

