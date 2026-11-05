# EfficientNet-B1 v15 result review

**Decision supported: reject this trained candidate as a replacement, retain both experiments as evidence, and do not promote or create a post-SPI parameter cache.** No further tuning is justified by this bounded result review. This rejects the particular tested configuration under the agreed gate, not CNNs or EfficientNet as a class.

## Evidence and comparison

Inspected the actual structured Kaggle log at `runs/notebook9_20261028_v15/monitor_v15/decision_artifacts_retry_quiet/betterscheduler-spi-20261027finalc.log` and currently extracted `efficientnet_b1/{validation,status}.json` under `monitor_v15/recovery_resumable/extracted`. These agree with the provided result summary. The recovered status records complete, selected training ran with no automatic promotion, and the saved run elapsed time is 94.0755 seconds; that is run elapsed time, not measured aggregate GPU utilization or total notebook/download time. Full archive checksum verification was still pending in the handoff; this assessment does not certify complete recovery.

The existing v13 frozen-CLIP balanced-linear baseline and v15 CNN were scored against the same exposed-development labels (n=96, support 43/8/38/7). The handoff binds both to the existing 300 teacher labels, SHA256 beginning `1f5b` and ending `c75a5`, with 204 training rows and 96 development-validation rows; no VLM rerun is reported. No independent label-hash re-audit was performed for this result review.

| Metric | CLIP linear v13 | EfficientNet-B1 v15 |
|---|---:|---:|
| Accuracy | 0.510417 | 0.541667 |
| Macro-F1 | 0.463708 | 0.446224 |
| Balanced accuracy | 0.573806 | 0.485165 |
| Low recall | 23/43 = 0.534884 | 24/43 = 0.558140 |
| Medium recall | 5/8 = 0.625000 | 3/8 = 0.375000 |
| High recall | 16/38 = 0.421053 | 22/38 = 0.578947 |
| Critical recall | 5/7 = 0.714286 | 3/7 = 0.428571 |

CNN selected epoch count is six, using only an inner group-separated split of the training data before refitting all 204 training examples. Its higher aggregate accuracy comes with lower macro-F1 and balanced accuracy and lower Medium/Critical recall. All four CNN recall values fail the >=0.70 each-class gate. The CLIP baseline already failed that gate for three classes; it also remains unready. Neither student is approved for production. Do not interpret the small observed macro-F1 difference as a statistically established universal ranking; no appropriate paired uncertainty analysis or independent final evaluation is claimed.

Both models show zero Critical -> Low/Medium errors on seven Critical examples, while the CNN predicts four of those seven as High. The absence of the most severe errors in this small set does not override the failed recall gate or establish safety. CNN Critical precision is 0.1875; its predictions are uncalibrated teacher-agreement scores, not validated urgency probabilities.

## Learning-curve interpretation

Inspected `runs/20261028_123917/summary.json`: local frozen-feature linear learning curves increase grouped-CV macro-F1 from 0.419971 at mean 35.625 training images to 0.517510 at 136 images, while training macro-F1 decreases from 0.867843 to 0.699234. This supports a limited hypothesis that additional representative training data may help this local fitting procedure. It does not demonstrate a causal explanation for the CNN outcome, prove that more data will clear the release threshold, or quantify a cross-environment improvement over v13.

The local diagnostics used a different scikit-learn version (reported 1.9 vs cloud 1.7); equivalent software/objective behavior has not been demonstrated. Therefore the local absolute CV value must not be advertised as a new improvement over cloud CV 0.436856. Within the local curve the fitting pipeline is shared, making its qualitative size trend more interpretable. Fold/subset outcomes are correlated, and full-size subsets across sampling seeds repeat deterministic fits; their rows are not independent replications. Do not extrapolate a sample count needed for a target score.

## Claim and delivery limits

These are image-only Qwen pseudo-label agreement results on development-exposed LADI/MEDIC data, not human urgency accuracy, calibrated probabilities, held-out final-test generalization, or validation of online scheduling. Source/group completeness and pretraining overlap are not independently certified here. Archive recovery should finish with verified checksums and required model/metric/history/plot/input identity files before delivery claims complete preservation. Keep the failed quality gate and no-promotion flags explicit regardless of successful execution or archive recovery. No new tuning, model run, production edits, or additional experiments were performed by reviewer.


Main subsequent retrieval update: full version15 recovery ZIP passed CRC and all20 manifest hashes, including plots/histories/checkpoint; see v15_terminal_receipt.json. The earlier pending archive verification is now resolved. No promotion or new run.
