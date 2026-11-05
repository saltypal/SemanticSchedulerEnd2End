# Bounded minority-recovery assessment

Assessment only, 2026-10-29. Reviewed `minority_recovery.py`, `spi_label_audit.py`, `test_minority_recovery.py`, training manifest/results/selection/summary, and audit provenance/current review/preserved v1/corrections/results. No production edits, additional labels, old-test evaluation or model fitting performed by this reviewer. Training script SHA256 matches the run manifest (`b63012e30e579abbd202ffd088cccbb8cf07c726c7fef1f8f0e0f70c3fc83b18`); all three saved model hashes match their result records.

## Result and selection integrity

No blocking implementation bug or confirmed selection leakage was found in this increment. Hyperparameters use inner grouped folds; outer grouped folds evaluate each family's training procedure. Saved outer train/evaluation image IDs are disjoint. Family selection reads only the 180 training rows and their outer predictions, and writes `selection_frozen.json` before computing validation scores. No selector receives validation targets. Former test targets are ignored when constructing either development role, and neither fitting nor scoring uses them. The loader does deserialize the source labels JSON; “excluded from target interpretation/use” is more literal than “never parsed.” The supplied poisoning test covers ignored former-test level values and availability flags.

Balanced logistic, C=10, was selected. Training nested OOF macro-F1 is 0.6708, with recall [0.9362, 0.2941, 0.6977, 0.7308] for Low/Medium/High/Critical. Validation descriptive macro-F1 is 0.4534, with recall [0.9394, 0.2, 0.4, 0.5]. Minority collapse is reduced within this exploratory training procedure, but these results do not establish recovery or readiness. Validation Critical support remains two. All promotion flags correctly remain false; there is no unexposed final evaluation.

The selected family's outer OOF score is **family-selection-exposed**, because it also chooses the winning family. It is not an independent final performance estimate. Validation is also historically development-exposed, even though this script did not select against it.

## Interpretation cautions

1. **The final RF correction was rejected by training selection.** Both `rf_control` and `rf_prior_adjustment` ultimately choose leaf=2, tau=0; their saved model hashes and validation predictions are identical. The correction family's larger nested OOF score (0.6300 versus 0.5647) evaluates its tuning procedure: two outer folds choose tau=0.5 and one chooses tau=0. It does not demonstrate benefit from the final saved prior correction, which is zero. The RF control is a newly fit seed-29/local-library control, not an exact replay of v20's seed-17 forest and selected leaf=5.
2. Prior division is explicitly a hypothesis on an already class-balanced forest, not a justified calibrated posterior transformation. The code and report should retain that distinction. Positive pooled OOF recall is an anti-collapse selection rule, not the 0.70 release criterion or evidence that each fold/minority scenario is reliable.
3. Timing is local head-only CPU prediction, and encoder features are reused. No fresh encoder inference, complete-pipeline timing or training FLOPs was measured here. The execution manifest reports the local interpreter and package versions; do not compare its head times directly with Kaggle full-pipeline times.

## Teacher audit

The panel contains 48 distinct training images, 12 per teacher class. The review manifest hides teacher labels; the key and audit analysis remain separate. Corrected AI review hashes match `audit_results.json`; v1 and thirteen correction records are preserved. Initial semantic and visual mistakes, including A32/A33, show why this is a fallible diagnostic. Original-image reinspection covered eleven cases; two further corrections used previous panels/descriptions, as disclosed. Artifact structure alone does not independently prove that the reviewer never accessed the withheld key.

Outside the AI reviewer's plausible range: Low 0/12, Medium 3/12, High 3/12, Critical 6/12. These class-balanced, AI-versus-AI disagreement counts are not representative teacher error rates, human urgency truth or reliability clearance. `automatic_relabeling=false` and `independent_urgency_validated=false` are appropriate; no path from audit analysis changes source labels or recovery targets. Preserve these limitations in the pending notebook/report and label the exported model an unpromoted experimental artifact.

The four reported tests were inspected, not independently rerun. They cover prior-adjustment algebra, grouped support/disjointness, former-test target poisoning/identity mismatch, and training-only audit/no automatic relabeling. They do not establish scientific generalization or teacher validity.
