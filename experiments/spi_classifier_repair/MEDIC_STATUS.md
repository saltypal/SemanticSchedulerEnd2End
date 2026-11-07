# MEDIC parameter implementation status

Kaggle notebook9 version20 COMPLETE, run medic_spi_20261028T221515Z. Retrieved summary evidence: runs/monitor_v20/compact_outputs_v20/campaign/. Exact immutable source SHA256: 6c5758f5a9da0b4b433e1ae6eda07557187a84b885bd493d154fadd874b0b364.

All42 planned frozen-encoder/head candidates completed on300 MEDIC images. Actual encoders: CLIP B32, SigLIP2 Base16, DINOv2 Small (declared fallback), ConvNeXtV2 Tiny. Qwen image-only pseudo-labels were reused from verified checkpoints; no fresh VLM inference needed in version20.

Development-selected CLIP+RandomForest: macroF1 .5921. Locked test: macroF1 .3979, accuracy .7833, recalls Low1.0/Medium0/High.6/Critical0. Predeclared CLIP-logistic baseline test macroF1 .5461. Selection failed to generalize; do not replace the winner posthoc using the opened test. Validation Critical support2; test Medium2/Critical3. No accepted independent teacher diagnostic. Runtime promotion FALSE. Bulk final parameter cache correctly NOT STARTED (0records).

Pipeline software checks passed: export parity0, actual-output35-byte serialization integration. Measured full extraction p50/p95: T4 GPU82.26/104.67ms; CPU533.06/565.74ms. Batch1, GPU100/CPU30 timed repetitions; CPU-stage3threads, GPU-stage1thread. Partial supported-forward count about35.075GFLOPs, NOT total FLOPs; unsupported ops/sklearn/Python excluded. RSS is sampled shared-process absolute usage, not model-only memory.

Evidence defects: restored failures.json contains historical v19 Tee failures despite42/42complete. CLIP identity only inventoried safetensors although .bin loaded, falsely reporting0checkpoint bytes and empty hashes. Main corrected both locally afterv20; historical evidence remains untouched. These fixes are not rerun-quality evidence and do not improve classification metrics. Immutable revision remains recorded, but v20 checksum coverage was incomplete.

Local checks after fixes:8focused tests passed; git diff --check passed. Real cached CLIP and SigLIP1 image/adapter shape checks passed locally (SigLIP1 branch check is not SigLIP2 quality evidence). Saved-result confusion, calibration/recall and MLP training plots rendered under retrieved campaign/figures/; original attention-fold curves preserved remotely/retrieved by ops.

Next decision: the300-image comparison is complete but the SPI classifier is not ready. A follow-on study needs independent teacher/label review and adequate Medium/Critical group support; freeze new development/final boundaries before new selection. Do not run another blind architecture sweep or force readiness. No scheduler/codec changes. Optional encoder fine-tuning/CORN not executed. No independent urgency accuracy or event generalization claim.


## Follow-on completed: minority recovery and label audit

See runs/minority_recovery_20261029/REPORT.md and the executed SPI_minority_and_label_audit.ipynb. No new cloud job or images. Blinded48-training-image AI diagnostic and nested3x3 train-only correction completed. Selected balanced logistic C10: training OOF macroF1 .6708 (selection-exposed), descriptive development .4534; Medium development recall .20. Original data labels unchanged. Research bundle exported with10-image runtime parity1.37334e-6 and actual605247071-byte CLIP binary checked against upstream pinned LFS SHA. Evidence-first teacher candidate implemented/parser-tested, not run/accepted.13focusedtests pass. Medium assessment found no blocking selection leakage. Promotion and bulk parameter cache still blocked; no fresh final test exists.
