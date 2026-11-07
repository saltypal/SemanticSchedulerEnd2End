# Notebook9 v13 final results assessment

**Recommendation: stop this run, preserve its trained artifacts, record failed readiness, and do not create the post-SPI parameter cache. Do not expand model search merely to clear the observed validation gate.** The saved results establish a completed four-class teacher-agreement experiment with an inadequate student for the specified release criterion. They do not establish human urgency accuracy or a successful full parameter-cache run.

## Evidence inspected

Local evidence root: `submission_v13_features/retrieved_version13/experiments/parameter_finalization/runs/notebook9_20261028_v13/data_run/`, relative to this review directory. Inspected `spi_classifier/metrics/{validation,candidates,support_gate}.json`, `spi_classifier/{status.json,models/spi.json}`, `spi_training/{labels.json,training_manifest.json}`, `producer_preflight.json`, and byte hashes of both saved feature NPZ files. No new training, label generation, or GPU/model execution performed by reviewer.

- Saved training labels: 300/300 available, all Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, one rubric; 204 training and 96 validation. Sources are LADI 220 and MEDIC 80. Direct inspection found no cross-split group identity. This does not independently certify event-group completeness or duplicate detection.
- All four classes pass the declared engineering row/group support gate: training counts 88/24/78/14 and validation counts 43/8/38/7, in Low/Medium/High/Critical order. Adequate minimum support is not evidence of sufficient statistical precision.
- Eight fixed configurations were evaluated by grouped training CV: six linear and two MLP. Selected balanced linear C=1 has mean macro-F1 0.436856. The balanced/unbalanced MLP values are 0.411877/0.405960. These results support retaining the existing selected configuration for this experiment, not assuming a larger head solves it.
- Saved validation teacher agreement: accuracy 0.510417, balanced accuracy 0.573806, macro-F1 0.463708. Each-class recall is Low 23/43=0.534884, Medium 5/8=0.625000, High 16/38=0.421053, Critical 5/7=0.714286. The >=0.70 every-class recall gate FAILS for Low, Medium, and High. Passing Critical recall on seven examples is fragile and does not offset those failures; its precision is 0.263158. Fourteen High/Critical labels were underpredicted, including 12 High -> Low/Medium and 2 Critical -> High.
- Export parity report maximum absolute difference is 8.60862e-8, within the 1e-4 release threshold. The classifier has four real outputs and its saved status is trained. Support/export success does not imply release readiness.
- Training-prior macro-F1 is 0.154676, so the student improves this simple baseline while remaining insufficient under the approved release gate. The baseline accuracy is 0.447917; improvement alone does not justify lowering thresholds.
- Runtime and training feature NPZ files are byte-identical. Producer preflight, training manifest, and model agree on fresh feature signature `105485efb3f14d80514f39138c3310714dfdbee850291a8f4e76551df5e5f065`. No old/new feature numerical equivalence is claimed.

## Outcome and gates

The remote execution's candidate-release path exception is a distinct orchestration defect: source-root containment was used for an output stored in a separate writable artifact root. The supplied correction to use the artifact root, and reported regression checks, do not change any saved predictions or metrics. A recovered release receipt should be explicitly labeled as post-run local recovery from the saved model/report; it must have `runtime_promotion_allowed=false`, with export/support checks passing and per-class recall failing. Do not describe the original remote run as having completed readiness or parameter caching. Producing a failed receipt needs no new labels or fitting.

This remains development-exposed teacher agreement on the same fixed cohort. There is no final test sample (`test` count zero), independent human urgency reference, source-generalization result, calibrated urgency probability, or proof of physical urgency from image area. The one-image YOLO preflight returning zero detections is a successful available measurement, not a full-corpus detector benchmark.

## Retrieval limitation requiring repair

At review time `runtime_features/provenance.json` contains an HTML 502 Server Error body rather than JSON. Original `teacher_labels/labels.json` and `teacher_labels/status.json` are absent from this local retrieval, although the complete training-label copy is available and valid at `spi_training/labels.json`. Repair/retrieve the bad provenance and any required raw teacher/cell evidence before claiming complete recovery. This is a local retrieval integrity gap, not evidence the remote feature extraction or labeling failed. Preserve the bad response separately if useful; do not invent missing producer details. The saved manifest/preflight/model/feature identities support the qualified result above.

## Further work boundary

No additional search or cloud rerun is justified merely by this gate failure. First close artifact recovery and report the negative result. If future improvement is requested, first inspect errors and class/source/group patterns against a small independently reviewed image-only reference to distinguish label ambiguity, frozen-feature limitations, and head fitting. Propose one discriminating bounded experiment from that evidence and preserve this run as its baseline. The current validation set has now been inspected; further choices using it are additional development, not a fresh final evaluation. Reserve a genuinely separate evaluation set for a later claim. Do not waive the gate, relabel classes to manufacture support, silently promote the model, or claim an untested stronger model will fix it.

## Subsequent recovery qualification

Operations subsequently reported targeted run_recovery.zip recovered and verified by CRC plus SHA against all 632/632 manifest entries, including the real runtime-feature provenance and original teacher-label files. The earlier local 502/missing-file retrieval gap is therefore resolved in that verified archive; the scientific no-cache verdict is unchanged. This qualification records the operations verification report rather than a new independent archive re-audit.
