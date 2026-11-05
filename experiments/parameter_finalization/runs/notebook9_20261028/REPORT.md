# Notebook 9: execution recovery and 300-image outcome

## Outcome
All 300 image-only Qwen3.5-9B pseudo-labels and runtime CLIP features were produced. Grouped training completed with four output classes. The student is **not ready**: three validation class recalls fail the unchanged 0.70 threshold. No parameter cache or full-corpus job was started. No further GPU run is justified by the current result.

## Results
Teacher-label counts: Low 131, Medium 32, High 116, Critical 21. Training 204 and validation 96, with grouped separation. These are teacher assessments, not independent human urgency labels.

| Class | Validation support | Recall |
|---|---:|---:|
| Low |43|0.5349|
| Medium |8|0.6250|
| High |38|0.4211|
| Critical |7|0.7143|

Selected model: balanced logistic linear head,C=1. Macro-F1 .4637, accuracy .5104. Existing grouped comparison included six linear settings and two MLP settings; selected head had best mean groupedCV macro-F1 .4369. More tuning on the exposed96 validation rows would not constitute fresh validation.

## Execution faults corrected
- v9: mistyped 61-character tokenizer checksum. Expected mapping now generated from local verified bytes and referenced once; exact emitted-code regression added.
- v10: assumed Kaggle input mount. Exact-manifest discovery now finds nested mounted snapshot, rejects ambiguity/wrong data.
- v11: logging Tee lacked standard console-stream methods. Noninteractive isatty and stream-metadata delegation added and tested.
- v12: cached local/cloud producer signatures differ. Re-extract all 300 with actual runtime producer; original cache untouched. New training and parameter extraction must share actual feature identity.
- v13: release artifact-root guard assumed source checkout while outputs live under experiment root. Notebook now passes its declared OUTPUT root; regression exercises outputroot separate from configroot. This final patch was locally tested, not a new GPU replay.

## Evidence and scope
Read `recovered_readiness_v13_complete/decision.json` and `cache_readiness.json`; `runtime_promotion_allowed=false`. Same saved v13 artifact/validation was used to finish the receipt without refitting or relabeling. `MEDIUM_FEATURE_REVIEW.md` and `FINAL_RESULTS_REVIEW.md` record independent assessments. `parameters/KAGGLE_RUNS.md` is the version/artifact map. See ops retrieval receipt for actual file completeness; missing downloads or HTTP error pages must not be counted as verified artifacts. FLOPs remain not instrumented, not zero.

## Next evidence needed
The observed failure is now a classifier/representation generalization limitation against a noisy teacher target, not missing classes or an execution blocker. Preserve this development cohort and all negative results. Before another training campaign, define a bounded representation/label-quality hypothesis and new grouped evaluation data. Do not force class labels, reduce the gate, or claim independent urgency accuracy.

## Final recovery verification
The targeted v13 recovery archive contains all 632 manifest-listed files with matching hashes and valid ZIP CRC. This resolves the initial HTTP502 provenance/raw-output retrieval gap. Complete labels, runtime feature provenance, trained model, validation/support metrics and cell evidence are preserved. No teacher or training rerun was required for this recovery.
