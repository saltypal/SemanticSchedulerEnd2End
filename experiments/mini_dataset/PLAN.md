# Mini dataset: provisional SPI enrichment

User authorized 2026-10-30. Target exactly 6000 acquired image records (4000 CrisisMMD, 1000 Open Images, 1000 Places365); source exact duplicates remain grouped. Canonical notebook Mini_Dataset.ipynb built by build_notebook.py imports shared image-pool producers.

Use 3933 available VLM labels from Windows v9; exclude 67 unavailable labels from fitting. CLIP global embeddings avoid token-center mask failures. Train balanced logistic regression on existing training groups, choose C in 0.1/1/10 using validation macro-F1, smaller C breaks ties. Evaluate existing test once; do not refit against its results. Full original study completeness gate remains unchanged; this is an explicitly authorized provisional alternative.

Runtime: acquire exact pool in Kaggle, copy verified previous scene cache from a managed input, verify image IDs/hashes at join, extract CLIP and train, infer all6000 SPI, profile CPU/GPU, package image files and all run evidence. Save encoder identity, model/receipt, image-to-prediction mapping, source/duplicate groups and portable image paths. Scores are uncalibrated; no independent urgency validation. Relevance and causal novelty remain simulator-generated.

Validation performed: local software fixture covers three hyperparameters and all split roles, no group overlap; training export and notebook Python-cell compilation pass. Not remote model accuracy evidence.

Cloud transfer cap 2GiB for reacquisition; no image/model downloads to laptop during setup. Download completed Mini archive only after checksum and disk-space verification; preserve Kaggle outputs if local capacity is insufficient.
