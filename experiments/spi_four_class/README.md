# Four-class SPI execution

Production modules are in parameters/. This folder owns the experiment notebook,
fixtures and run artifacts. No codec or scheduling policy is trained here.

## Pipeline

1. Verify existing image/feature hashes and development groups.
2. Load one pinned teacher on GPU1 with thinking OFF and verify actual images.
3. Compare direct and evidence-first prompts on 48 authored text controls.
4. Compare passing protocols on paired 80-image development panels; save raw
   prompts, outputs, disagreements, validity and timing. No natural urgency gold
   is inferred from this comparison.
5. Freeze the protocol and pseudo-label the remaining development images.
6. Require all four classes and group support; otherwise stop without a model.
7. Compare six logistic and two MLP configurations with grouped selection and
   fit-fold-only stopping; export the selected classifier and parity evidence.
8. Retrieve and verify run outputs. The release guard must reject a model whose
   Medium or other class validation is weak; old checkpoints are preserved.

## Model and status interpretation

Low/Medium/High/Critical are exactly four outputs. Unknown assessment is a status,
not a fifth label. Evidence claims distinguish nonurgent action from no action.
An old three-class artifact cannot satisfy four-class support. Model output
probabilities are uncalibrated unless a separate calibration procedure exists.

status.json distinguishes teacher_gate_failed, four_class_support_failed,
incomplete, and trained_development_model. Notebook COMPLETE alone is not success.
Teacher agreement is not independent human urgency accuracy. Existing 300 inputs
are development-exposed; they are not a fresh final test.

## Frozen submissions and recovery

runs/20261027_spi_evidence_v3a/submission_final/ contains notebook.ipynb,
source.zip, SOURCE_MANIFEST through the archive, PACKAGE.json and kernel metadata.
Source SHA256: 792e8b28c141beb33d5bed984125d22caf92d4684456d271a5b19cd8b8586604.
Canonical notebook: notebooks/spi_pipeline.ipynb.
Existing remote identity: satyapaladugu/betterscheduler-spi-20261027finalc.
Use a NEW VERSION of that identity. No new slug per run.

Run map and actual version/status: ../../parameters/KAGGLE_RUNS.md.
Expected remote output: /kaggle/working/experiments/spi_four_class/runs/
20261027_spi_evidence_v3a/, plus /kaggle/working/run_recovery.zip.
Local verified retrieval belongs below that same run folder. Never store model
weights/credentials in logs, and never report expected files as retrieved files.

## Useful commands after retrieval

From the repository .venv, evaluate the saved model or generate a release candidate
using shared parameters.spi_infer and parameters.spi_release modules. The latter
requires --model, --validation, --current-manifest and a NEW --output path, preserves
the current manifest, and does not overwrite the selected runtime model.

The local fixture tests are explicitly engineering references. Operational logs,
commands, source hashes and results are in runs/local_checks_20261027_stable/.
The first mutable-source run is preserved separately and is not final acceptance.

## Corrected v3b submission

The v3a notebook ended ERROR before labeling/training because the new evidence
parser rejected otherwise valid JSON inside Markdown fences. Verified recovery
and offline replay show ten valid claims with wrapper-only normalization, no
semantic edits. The affected regression suite passed26 tests.

Corrected package: runs/20261027_spi_evidence_v3b/submission/.
Source SHA256: bf06d6779e0bfcdf9f89c78218b60cebcfe22d996ae4fda999bbd77aaec3c85e.
Kernel numeric identity remains141089219. Submission addresses its current
alias with the original title to restore the original URL through versioning.
Read parameters/KAGGLE_RUNS.md for ACTUAL version/status rather than inferring
execution or success from this prepared-package record.
