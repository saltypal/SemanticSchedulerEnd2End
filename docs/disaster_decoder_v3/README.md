# Disaster decoder V3 study

This additive study evaluates ordering fully arrived digital latent jobs at a receiver using compact task-conditioned evidence. It does not change transmission, decode quality, partial-arrival handling, or the legacy scheduler. It is not a deployed decoder, real wireless result, or evidence of improved reconstruction.

## Current checkpoint — 23 October 2026

The eight-field contract, frozen extractors, 32-byte header, header-only rankers, receiver ingress/dispatch, training and static/streaming benchmark are implemented. Stage A extraction and Stage B training completed on Kaggle. All 480 primary artifacts passed package checks, and 37,440 exported-model episode orders were reproduced. The final version-2 notebook resumed the completed study, matched its 39 quality rows and saved actual captured outputs, including the full table and two embedded figures. The combined local suite passed 53 tests.

The pinned dataset revision is `5f2dbfe8c466d32edafd1bab847ec5252309acdb`. Downloaded source counts are 890 validation and 1,049 test rows. Two exact-image IDs have contradictory native labels (four source rows); all four rows are excluded and preserved in conflict records. Consistent exact duplicates are extracted once. The audited unique-ID set has 1,934 rows: A=445, B=178, C=267, test=1,044. The extraction summary's test count of 1,045 is a pre-deduplication role-row count and is not the unique test count.

The Stage A audit reports 1,934 finite feature rows and matching labels, 512-dimensional embeddings, three image-model outputs per retained ID, eight device cost profiles (four per Tesla T4), and 32 schema-valid Qwen proposals. Source/model snapshots and artifact hashes match the audit. These are extraction and annotation checks, not ranker-quality results.

Registered-operation inference counts were measured as CLIP 8.81762304 GFLOPs, LADI 32.6971392 GFLOPs, and DINO 665.607–746.230 GFLOPs depending on input shape. These counts cover registered operations, not all hardware work. No comparable full-model Qwen FLOP count was measured; partial analytic text estimates are not comparable. Qwen output is silver annotation only. LADI is a reference signal, not ground truth.

The benchmark compared 13 methods (12 baselines plus CAHR), using the same fixed 128 A-only queries per task and at most 60 epochs. Outcomes were mixed and did not establish CAHR superiority. C-selected candidates were coverage trees for building/flooding and a regressor for roads; test results did not change those choices. Building-only auxiliary experiments included nine retrained ablations and two additional seeds. Codec execution and journal-stage validation remain outside the completed prototype.

## Reports and artifacts

- [Executed notebook](../../Scheduler/notebooks/05_disaster_decoder_v3.ipynb) — actual Kaggle outputs, 39-row table and two figures.
- [Start here](HANDOFF.md) and [runtime API usage](RUNTIME_USAGE.md).
- [Research report](RESEARCH_REPORT.md) — methods, measured comparison, audits, costs and limitations.
- [Simple report](SIMPLE_REPORT.md) — plain-language report with the same evidence limits.
- [Artifact index](ARTIFACT_INDEX.md) — evidence files, provenance, and use restrictions.
- [Technical specification](TECHNICAL_SPEC.md), [plain-language guide](PLAIN_LANGUAGE_GUIDE.md), and [machine-readable protocol](PROTOCOL.json).
- [Implementation journal](IMPLEMENTATION_JOURNAL.md) — chronological evidence checkpoints.

Do not claim journal readiness, universal superiority, real codec benefits, MAC deadlines, or life-safety value from this prototype.
