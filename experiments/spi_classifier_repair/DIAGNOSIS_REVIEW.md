# Bounded SPI failure diagnosis

2026-10-28. Read-only investigation of the actual v13 label/feature cohort and v15 CNN outputs; no production changes or teacher calls. Checkout HEAD `032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0`, with concurrent uncommitted work. Conclusions bind to recovered artifacts rather than a clean HEAD snapshot.

## Finding

**No confirmed teacher-image mismatch, masking failure, cache identity error, class-order error, or optimization collapse was found.** Poor held-out teacher agreement is confirmed; its causal decomposition into representation limits, training support, teacher boundary inconsistency and domain shift remains unresolved. An evidence-based repair cannot honestly be described as fixing a demonstrated data-corruption bug.

## Discriminating check actually executed

Used project `.venv/Scripts/python.exe`, NumPy, Pillow and the existing `parameters.spi.prepare`/`SPIClassifier`. Joined all 300 recovered records/labels to the immutable local input `parameters/results/spi_inputs_v2_20261027b/images.zip` and panel. For **every image**, verified:

- Source ZIP image bytes hash equals label/record `image_sha256`; record ID, group, split and source match label.
- Exactly one request image; its hash and supplied dimensions equal the actual source bytes/dimensions. Execution receipts equal request receipts and case ID equals image ID.
- Request system prompt equals current `spi_direct_priority.PROMPT`; payload contains only the fixed urgency instruction; rendered template includes the image token; pixel tensor is nonempty.
- Raw response is exactly the corresponding class word after case normalization; saved parsed label agrees.
- Grid-derived area equals recorded processed pixels.

Results: **300/300 pass; zero mismatches; zero nontrivial EXIF orientations.** Thus an untransposed EXIF difference between teacher and students does not explain this cohort. Teacher-label JSON and training-label JSON are exactly equal. Their SHA256 is `1f5b3d78124d79c6fd43bb87ba2e7e94a0c18852bc45ca020aa550b40c7c75a5`. Runtime feature NPZ and training feature NPZ have array-identical embeddings, IDs, content hashes and feature signature. Recomputed saved-head development macro-F1 is `0.46370801725683597`; training macro-F1 is `0.6636545064055952`. All 96 v15 prediction IDs are unique, are validation IDs and have matching reference classes.

This rejects the concrete hypothesis that labels were accidentally assigned to different/masked image inputs at the audited boundary. Pixel tensor hashes and nonempty inputs establish supplied visual content, not a guarantee that Qwen semantically attended to it correctly. No independent replay of teacher tensor preprocessing/inference was performed.

## Mechanisms and competing causes

1. **Confirmed evidence limitation: teacher agreement is the objective actually measured.** `dataset_pipeline.label_dataset` passes one image and instruction only, with the forced four-class visible-scene rubric. No native labels or student predictions enter that request. Qwen semantic errors and inconsistent class boundaries remain possible; one-word responses contain no evidence explanation to audit. No independent urgency annotations are supplied here. This is not a code bug.
2. **Confirmed information difference; causal impact untested.** Teacher processed image areas are 65,536–262,144 pixels; CLIP receives a 224-square aspect-padded representation and CNN a 240-square aspect-padded representation. Full frames are preserved by student preprocessing; padding reduces effective scene pixels especially for wide/tall images. Teacher grid rounding produces a maximum relative aspect-ratio discrepancy of 10.94% in this cohort. There is no evidence here that `max_pixels` means side length, causes catastrophic distortion, or corrupts labels. Changing it and relabeling would change the target rather than establish a repair.
3. **Confirmed limited class support and generalization gap; exact cause unresolved.** Train support Low/Medium/High/Critical is 88/24/78/14; validation 43/8/38/7. CNN inner fitting has 43/12/42/8; early stopping 45/12/36/6. At selected epoch 6, inner-fit macro-F1 is .777 versus inner-stop .497; continued training through epoch 12 gives .791 versus .465. Reinitialized refit on all 204 reaches .679 at epoch 6. Loss decreases and fitting works; this is not a collapsed optimizer. A selected epoch from 105 fitting rows is not evidence that a 204-row refit is fully optimized.
4. **Possible objective mismatch, not confirmed defect.** Both methods optimize nominal class cross-entropy/logistic likelihood while evaluating macro-F1/per-class recall and the operational urgency classes are ordered. Class weights partly address imbalance. An ordered loss is a legitimate bounded candidate; it is not a demonstrated root-cause fix and must beat the same grouped-CV baseline before outer re-evaluation.
5. **Parser concern rejected.** `parse_recovery_used=True` on uppercase class words reflects the explicit title-case canonicalization (`CRITICAL` -> `Critical`), not fallback label creation. Strict one-word matching passed all 300 raw files.

## Justified next action

Keep both failed candidates and gate unchanged. Do not relabel or patch teacher input based on an image-mismatch hypothesis: the cheapest direct check rejected it. For the currently proposed ordered classifier, compare on training-only grouped folds and identical features/labels, with complexity/regularization chosen there. Treat any gain as an empirical candidate improvement, not proof that ordering caused the original failure. If no training-only improvement emerges, the next distinct evidence question is class-boundary/visible-cue consistency; inspect a small fixed image panel independently before funding more labeling or changing architecture. Additional training groups should address minority support and source diversity, but this investigation does not establish a sample count or guarantee a .70 gate.

## Evidence locations and coverage

Base `experiments/parameter_finalization/runs/notebook9_20261028/submission_v13_features/recovery_expanded_verified/data_run/`: `input_images/records.json`, `teacher_labels/{labels.json,status.json,raw/}`, `runtime_features/{features.npz,provenance.json}`, `spi_training/{labels.json,features.npz,training_manifest.json}`, `spi_classifier/{models/spi.json,metrics/validation.json}`.

CNN base `experiments/spi_classifier_repair/runs/notebook9_20261028_v15/monitor_v15/recovery_resumable/extracted/efficientnet_b1/`: `config.json`, `predictions.json`, `selection_history.json`, `refit_history.json`.

Source paths traced: `parameters/{dataset_pipeline.py,spi_direct_priority.py,qwen_reviewer.py,qwen35_reviewer.py,spi_teachers.py,minimal_vision.py,vision.py,spi.py,spi_model_selection.py}`, `experiments/spi_classifier_repair/cnn_compare.py`. This review did not retrain, repeat Qwen, establish human correctness, independently verify a complete archive manifest, or audit every upstream dataset-grouping decision. No general literature search was needed to resolve the bounded image-identity question.
