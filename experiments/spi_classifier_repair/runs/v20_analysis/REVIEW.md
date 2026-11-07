# v20 assessment

Assessment only; no production edits, retuning, additional labels, model execution or remote jobs. Run: Kaggle v20, `medic_spi_20261028T221515Z`. Executed source archive: `medic_training_streamfix_20261028T220317Z/source.zip`, SHA256 `6c5758f5a9da0b4b433e1ae6eda07557187a84b885bd493d154fadd874b0b364`. Release/deployment/parameter source matches that archive; the live campaign file has subsequently changed.

## Outcome

The frozen-representation comparison completed 42/42 candidates. CLIP + random forest was selected using development macro-F1, not the test. Its failure to meet release gates is an unready-model result, not evidence that v20 crashed. Promotion is correctly false.

| Evidence | Development | Locked test |
| --- | --- | --- |
| Macro-F1 | 0.5921 | 0.3979 |
| Accuracy | 0.8167 | 0.7833 |
| Class support: Low/Medium/High/Critical | 33/5/20/2 | 35/2/20/3 |
| Recall in the same order | 1/0/0.75/0.5 | 1/0/0.6/0 |

Development Critical support fails the configured minimum; Medium recall is zero. Test Medium/Critical support and recall also fail. No accepted diagnostic receipt was supplied. The locked-test CLIP-logistic baseline has macro-F1 0.5461; selected-minus-baseline difference is -0.1482, with recorded group-bootstrap 95% interval [-0.3047, 0.0271]. This does not establish improvement. These are Qwen pseudo-label agreement results; event independence and human urgency accuracy are unverified.

## Contracts and validation

- Inspected selection/CV/calibration paths keep hyperparameter fitting and temperature fitting in training data, and family selection in development. The executed release code freezes calibration, verifies parity/integration and profiles before test evaluation; no automatic test-based retuning follows. The baseline is predeclared in the executed source. This holdout is now exposed and must not become a new tuning set while retaining a fresh-test claim.
- Receipt parity error is 0 on the checked ten development images, comparing cached features with the image runtime. Both controlled 35-byte wire probes pass; observed probe classes are Low/High/Critical. This is finite-sample Python-artifact/runtime evidence, not universal export equivalence or physical deployment validation.
- `cache_ready` returns immediately on the observed false promotion receipt and writes zero-record readiness-failed status. The lead separately confirmed the cache status reports zero records; this reviewer inspected the gate/control flow but did not independently receive that status file. Full records/splits/labels were not present in this review's compact retrieval, so direct split-disjointness rechecking remains a retrieval limitation.
- GPU pipeline p50/p95: 82.26/104.67 ms, 100 samples; CPU: 533.06/565.74 ms, 30 samples. Scope is SPI + separate CLIP relevance + YOLO + header/query/commit with a seeded FIFO128 engineering history. Reported Torch threads differ (1 versus 3); keep this and hardware attached to comparisons. RSS is explicitly sampled absolute shared-process memory. The approximately 35.075 billion supported FLOPs exclude sklearn/Python/image processing and unsupported operators (96 GPU-stage/85 CPU-stage entries); they are not full pipeline FLOPs. Setup-load timing occurs in the existing process/cache context, not a clean-host deployment benchmark.

## Confirmed implementation findings

1. **P2: stale failure report on resume.** The executed `compare` initializes a fresh failure list but only writes `failures.json` on exceptions. Consequently restored v19 Tee errors survive beside v20's complete 42/42 coverage. Preserve historical failures separately and always write current-run failures, including an empty list. This changes interpretation, not v20 model results; no rerun is needed to distinguish the stale errors.
2. **P2: incomplete weight checksums and size.** Selected CLIP identity records `weights_sha256={}` and `download_checkpoint_bytes=0` despite 87,849,216 vision parameters. `load_encoder` permits `.bin` weights but hashes/counts only `.safetensors`. The immutable repository revision remains useful provenance, but this checksum comparison is vacuous for the actual binary checkpoint and zero is an incorrect size claim. Hash/count the actual weight files, including `.bin` and relevant indexes; use unavailable rather than zero if not established.

The lead reports both P2 issues have since been corrected locally, with eight focused checks passing. Those changes do not alter immutable v20 artifacts or its measured results; this reviewer did not independently rerun those checks.

No further blocking implementation/scientific fault was confirmed within this bounded review. Do not promote the model or reinterpret sparse classes/failed gates as successful four-class readiness. Refinement and natural semantic-novelty accuracy remain untested.

Inspected retrieved artifacts under `experiments/spi_classifier_repair/runs/v20_analysis/experiments/spi_classifier_repair/runs/medic_spi_20261028T221515Z/campaign/`: `REPORT.md`, comparison candidates/coverage/failures, release readiness/development/locked_test/profiles. This review does not claim full artifact retrieval or independent retraining reproducibility.
