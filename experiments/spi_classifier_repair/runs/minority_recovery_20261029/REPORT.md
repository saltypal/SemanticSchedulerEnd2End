# SPI minority-class recovery and teacher-label audit

Outcome: the bounded diagnosis and corrective training experiment are complete. The classifier remains unready; no production SPI, final parameter cache or new cloud run was forced. All work is in the canonical SemanticSchedulerEnd2End repository.

## What was executed

- Exact v20 CLIP features reused after recovery-manifest SHA verification; no new image dataset or model downloaded.
- Blinded AI review of48 training images,12 per original teacher class. Teacher labels/native annotations were withheld from the reviewer manifest. Initial AI review and corrections are preserved. Eleven originals were reopened; other cases used contact sheets, so fine-detail inspection is incomplete.
- Nested grouped3x3 training-only comparison of balanced logistic, a random-forest control and a fold-local prior-adjustment hypothesis. Train180 only; existing development60 scored after freezing the training-only choice. No former-test target interpreted by selection/evaluation.
- Real image-to-class runtime parity on10 training images; actual605,247,071-byte CLIP .bin hash verified against the original pinned upstream LFS SHA.
- Local executed notebook with per-stage logs, figures and source/input checks;13 focused tests passed, including held-out-label poisoning, group isolation, hand-computed prior adjustment and strict teacher-candidate parsing.

## Training results (not fresh final performance)

| Procedure | Training nested OOF macro-F1 | Medium/Critical OOF recall | Development macro-F1 | Medium/Critical development recall |
|---|---:|---|---:|---|
| balanced_logistic | 0.6708 | 0.294/0.731 | 0.4534 | 0.200/0.500 |
| rf_control | 0.5647 | 0.000/0.731 | 0.4470 | 0.000/0.500 |
| rf_prior_adjustment | 0.6300 | 0.412/0.808 | 0.4470 | 0.000/0.500 |

Balanced logistic (C10) was selected from training evidence. Its winning OOF value is family-selection-exposed. Development is historically exposed and has only5 Medium/2 Critical targets. No statistically supported generalization improvement is claimed.

The final prior-adjusted forest chose tau0 and leaf2: its fitted artifact/predictions equal the RF control. Its larger nested OOF value measures a tuner that sometimes selected tau0.5 inside folds, not a better final corrected forest. Reject the correction as a demonstrated final-model improvement. This control also uses a different seed/local library from v20; it is not an exact replay of the previous forest.

## Label findings

| Original teacher class | Reviewed | Outside AI-review plausible range |
|---|---:|---:|
| Low | 12 | 0 |
| Medium | 12 | 3 |
| High | 12 | 3 |
| Critical | 12 | 6 |

These are balanced-panel AI disagreement counts, not teacher-error rates or human urgency truth. The original teacher labeled a person carrying a rifle Critical despite no clearly visible immediate harm (A27). Other flagged cases involve ambiguous prone-person or aerial-damage/flooding evidence. Such cases need resolved evidence, not confidence alone.

The reviewer also made mistakes: A32 initially confused a phone-call screen with a threat; A33 initially missed actual checkout-error text. The latter is plausibly a nonurgent Medium problem, not automatically Low because it is a screenshot. Main spotted these issues; all corrections and prior review are retained. Neither AI agreement nor one reviewer clears the independent reliability gate. No source labels were rewritten.

## Implemented fixes and bounded candidate

1. Added a selector that prevents a zero pooled-OOF-recall candidate silently winning on macro-F1 alone. This is an anti-collapse check, not the original0.70 readiness gate.
2. Added strict train-only panel selection, blinded evidence review, immutable join key and no-automatic-relabeling results.
3. Implemented an evidence-first teacher candidate in experiments/spi_classifier_repair/evidence_priority.py. It requires actual observations, separates severity from immediacy, handles text/UI evidence and preserves insufficient evidence as unavailable. Its parser rejects internally unsupported Critical JSON. It has NOT been run as a VLM, accepted as the teacher or used to change labels. A hallucinated cue could still pass schema validation; image-level verification is required.
4. Exported an unpromoted four-class research bundle using the existing image runtime. Ten training-image probability comparisons have maximum error1.37334e-6, under1e-4. No global inference/generalization guarantee follows. Temperature1 is uncalibrated in this bundle. Production loading rejects the false readiness receipt.

## Cost and scope

All recovery fitting uses cached features on CPU. Measured selected-head-only median0.1184ms, p950.2427ms, batch1,100 calls after10 warmups, threadpool1. This excludes image encoder/preprocessing, scene detector, KB and header. It is NOT comparable to the previous Kaggle82ms/533ms whole-pipeline figures. Neural dense-dot arithmetic proxy is4096 operations per logistic prediction, using2FLOPs/MAC; solver/tree/full inference FLOPs remain uninstrumented, not zero. The10-image parity run is functional validation, not a latency benchmark.

## Decision and next required evidence

Keep all four classes, current parameters and35-byte header. Do not add rarity, new scene engines, larger encoders or another broad grid in response to this failure. Retain the compact balanced-logistic research baseline, but keep readiness false.

The remaining work is label evidence and independent class support. Validate the evidence-first candidate on resolved image-level cases before relabeling; preserve original and adjudicated labels separately. More copies, oversampling or redistributing the current300 do not create independent evidence. Current validation needs at least3 additional independently supported Critical groups merely to meet the old minimum5; five is an engineering gate, not a statistical guarantee of0.70 recall. A new final study needs adequate per-class independent groups chosen before fitting, and a sampling/uncertainty calculation based on the intended claim. No such fresh test currently exists.

No new corpus download/cloud labeling was launched in this increment. A follow-on MEDIC-only acquisition/label audit must be explicitly bounded and reserve fresh groups; do not reopen the old test for tuning then call it fresh. Until then this is a tested research prototype, not a finalized urgency classifier.

## Artifacts and reproduction

- SPI_minority_and_label_audit.ipynb — executed locally; notebook_execution/logs, metrics and figures.
- training/results.json, summary.json, manifest.json and selection_frozen.json — training/selection evidence.
- training/bundle/ — research model, pinned encoder identity, false readiness receipt and checksums.
- label_audit/ — blind panels/manifest, withheld key, preserved review versions/corrections and diagnostic join.
- REVIEW.md — Sol Medium assessment; no blocking selection/leakage bug confirmed.
- source_manifest.json and source_before/ — preserved relevant source identity.

Rerun tests: .venv/Scripts/python.exe -m unittest experiments.spi_classifier_repair.test_minority_recovery experiments.spi_classifier_repair.test_medic_campaign experiments.spi_classifier_repair.test_resume_and_runtime -q

Refit only with a new output directory: .venv/Scripts/python.exe -m experiments.spi_classifier_repair.minority_recovery --features <verified v20 CLIP features.npz> --records <v20 records.json> --labels <v20 labels.json> --output <new output>. Notebook rendering reuses saved results; rerendering creates a fresh notebook execution directory while protecting prior logs/outputs.

## Source rationale

[Menon et al., Long-tail learning via logit adjustment, ICLR2021](https://research.google/pubs/long-tail-learning-via-logit-adjustment/) motivates investigating prior-based correction; this forest application is a bounded hypothesis, not a transfer of the paper's guarantees. [Official RandomForestClassifier documentation](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.RandomForestClassifier.html) documents balanced class weighting and probability averaging. Existing weighted training was not assumed to guarantee nonzero minority argmax predictions.
