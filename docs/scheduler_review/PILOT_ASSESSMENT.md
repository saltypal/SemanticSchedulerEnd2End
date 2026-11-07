# Pilot assessment checkpoint

Assessment only, 2026-10-24. No production, test, notebook, dependency or plan edits. No full suite or long run executed by this reviewer.

## Reviewed snapshot and scope

- `INFRA/learning/decoder_review_pilot.py` SHA256 `d8032071ffe11b85636bf700f5e08c1ee3501f498dc5a05f8c4e63d6ec9603a7`.
- `INFRA/tests/test_decoder_review_pilot.py` SHA256 `547ee0c5771d5d203ed6abf75abd8a07fc246bdf1f187880ed33625fb53c54a6`.
- Inspected the called cache, calibration, episode, imputation, scoring and aggregation contracts in `decoder_v3_study.py`, `decoder_parameter_audit.py`, `decoder_scheduler.py` and `decoder_parameters.py`.
- Targeted diagnostic: existing `..\SemanticSchedulerEnd2End\.venv\Scripts\python.exe`, CPU, two synthetic aggregation rows; approximately 6 seconds including startup/import. Evidence: `results/scheduler_review/checkpoint_checks/aggregation_support.json`.

## Outcome

No blocking correctness issue found in the inspected execution paths. Selection takes A records only and splits complete duplicate groups. B calibration is fixed independently of the A folds. C episode construction/scoring follows selection; test truth is confined to the stated cache integrity read. Medians/weighted means/scales and regression fit use training episodes only. Each included field brings its own availability column; Ridge sample weights give each observed image ID total weight one across training occurrences. Selected and control alphas are selected independently on the same inner grid.

The singleton exact optimum is valid for this fixed nonnegative weighted-ID, equal-cost, successful-decode objective: one copy of each positive-weight ID in descending weight simultaneously maximizes every prefix. This supports A10 optimality, including zero-weight tails; it supplies no planning claim for multi-fact, uncertain failure, unequal-cost or different-view scenes. The code keeps its synthetic multi-fact counterexample separate. All policies share each task's C episode bank. Defined pool means, rather than raw episode counts, weight fold aggregation. For the full grid there are 255 x 3 x 3 = 2,295 inner fits per task, 6,885 across three tasks, plus nine final fits. Controls already occur in the full grid and add no inner fits.

## Findings and limitations

1. **Low, confirmed reporting gap — `decoder_review_pilot.py:336-338`.** Aggregation skips undefined values independently per metric, while `undefined_episodes` records A10 only. An episode with positive weight but no weight-3 ID has defined A10 and undefined high_recall10. Thus A10 and high-recall may use different episode/pool support without denominator counts in the summary. The two-row diagnostic demonstrates A10 support of two pools versus high-recall support of one pool, while the listed A10 undefined count is zero. The metric values are mathematically consistent; their support is underreported. Minimal correction: report defined/undefined episode counts and defined pool counts per metric (and per stratum where reported), or explicitly disclose these denominators in the downstream report. Raw per-episode CSV retains enough information to recover them. This does not invalidate A10 selection or require interrupting a healthy run.

2. **Claim limitation, not a detected implementation bug — controls and test coverage.** This restricted pilot contains I/all-field, reversed-I, shuffled-vector, signature and oracle controls. It does not separately establish masks-only, missingness-matched, decoder-failure, heterogeneous-cost or realistic multi-fact performance. The six inspected tests exercise the exhaustive tiny optimum/reference, failure-reward identity, role poison, group partition, mechanics/header masks, fixture run and constant-column/C-poison paths. They do not independently assert the weighted matrix/target numerics, full-grid execution or recovery from an interrupted run. These are boundaries on claims; static inspection found the weighting and bundle indexing correct. Full-run empirical evidence belongs to the separately running validation.

## Operational observations

The run records interpreter/package versions, Git revision/dirty state, five source hashes and input-file hashes, refuses nonempty output reuse, writes progress and partial task results, and captures Python exceptions. A task interrupted during its grid has progress/provenance but no completed grid until the search returns. Restart/resume is not implemented, so this is a fresh-output development run rather than a resumable training system. Calibration objects are reproducible from the hashed cache/source but are not themselves serialized in this pilot's summary. Neither issue is a blocker for the bounded CPU experiment.
