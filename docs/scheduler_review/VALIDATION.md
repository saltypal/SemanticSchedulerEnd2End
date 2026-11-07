# Scheduler review validation

## Run identity and scope

- Run: `20261024T071700Z`; repository `SemanticSchedulerParameterAudit`; source baseline `2f16785a16dd9c58ce12c98b502087cc212b279b`.
- The checkout was sparse and dirty before execution: the extraction `features.jsonl` materialization was modified and the parameter audit, review pilot, tests, and review documents were untracked. The frozen pilot source/test hashes matched the implementation owner's supplied hashes before and after execution. All eight existing test files also retained their pre-run SHA-256 hashes. The validation did not edit source or tests.
- Environment: project interpreter `D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End\.venv\Scripts\python.exe`, Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1, pytest 9.1.1, torch 2.14.0. CPU only; the installed torch build has no CUDA support and no CUDA device was visible.
- This is cached-feature replay. No fresh image extraction, detector/CLIP inference, codec execution, GPU work, or Kaggle run occurred. The pilot fits CPU Ridge models over the saved extraction features. The local run directory was enumerated and its JSON/CSV outputs were reopened for validation; remote/Kaggle persistence was not exercised.

## Checks run

Existing approved tests:

```powershell
D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End\.venv\Scripts\python.exe -m pytest SemParam/tests/test_decoder_parameters.py SemParam/tests/test_decoder_costs.py Scheduler/tests/test_decoder_scheduler.py INFRA/tests/test_decoder_queue.py INFRA/tests/test_decoder_v3_extract.py INFRA/tests/test_decoder_v3_study.py INFRA/tests/test_decoder_v3_analysis.py INFRA/tests/test_decoder_parameter_audit.py --junitxml <run>/existing_tests.junit.xml -q
```

Result: **58 passed, 3 skipped, 24 warnings** in 25.43 seconds. The three skips require the absent `lightgbm` package; the 24 warnings are scikit-learn `invalid value encountered in sqrt` warnings from the parameter audit tests. No dependency was installed.

The frozen pilot tests passed **6/6** in 5.88 seconds. CLI `--help` confirmed the full-run options. One initial launch was rejected before cache processing because the run harness put its start manifest inside the pilot output directory, which the module requires to be empty. That failure and return code are retained in `results/scheduler_review/20261024T071700Z/full_pilot/`. The clean retry wrote to a new output path and completed successfully.

Full pilot command (CPU thread limits set to one):

```powershell
D:\Bunker\BaseCamp\WirelessCommunication\SemanticCommunication\SemanticSchedulerEnd2End\.venv\Scripts\python.exe -m INFRA.learning.decoder_review_pilot --cache results/disaster_decoder_v3/20261022_stageA_v3/extraction --output results/scheduler_review/20261024T071700Z/full_pilot_result --seed 20261024 --folds 3 --train-episodes 24 --repeats 1
```

It completed with return code 0 in 80.084 seconds. The run used 255 nonempty field subsets, 3 alpha values, and 3 grouped folds per task: 765 grid candidates and 2,295 inner fit rows per task, 6,885 inner fits total, plus 3 final refits per task (9). There are 40 cached C episodes per task and 14 evaluated model/baseline rows per episode. Fold memberships were checked against A-eligible cache groups and had no train/held overlap. Selection choices were independently recomputed from the stored inner-fold scores, including the 1e-12 tie rule; C labels were not used for selection.
A-fold winners varied by task. Building choices were `[3,6,7]` at alpha 10, `[1,3,7]` at 0.1, and `[1,2,3,7]` at 0.1; road choices were `[1,3,6]`, `[1,3]`, and `[3]` (all alpha 10); flooding choices were `[0,3,5,6,7]` at 10, `[1,2,3,5]` at 1, and `[3,5,6,7]` at 1. Road fold scores ranged from 0.4469 to 0.9300. The near-best sets also had diffuse field inclusion for structural flooding. These are selection-instability diagnostics, not independent final scores.

The independent output audit passed **210 checks**: JSON parsing and finite numeric values (intentional JSON nulls remain), source/cache hash stability, grid/fit counts, fold membership, weighted fold-score to grid-score agreement, selected and control grid choices, baseline presence, equal-pool aggregate recomputation from raw episode CSV, oracle A10 identity, and parameter/header/queue fixtures. Per-metric defined and undefined episode and pool counts are recorded in `results/scheduler_review/20261024T071700Z/metric_support.json` because the pilot summary's `undefined_episodes` field covers A10 only.

## Measured comparison

The selected policy is compared on C, which the run marks as **historically exposed development**. These values are descriptive only; this is not nested final evaluation. `A10` is the mean cumulative normalized utility over the first ten positions.

| Task | FIFO A10 | RR A10 | Existing importance I A10 | Selected A10 | Selected minus I |
|---|---:|---:|---:|---:|---:|
| building_damage | 0.3058 | 0.3081 | 0.7034 | 0.7037 | +0.0003 |
| road_damage | 0.1657 | 0.2153 | 0.7197 | 0.6120 | -0.1077 |
| structural_flooding | 0.1590 | 0.2100 | 0.7856 | 0.7950 | +0.0094 |

The selected feature sets were `[0, 3]`, `[1, 3, 6]`, and `[0, 3, 5, 6, 7]`, respectively (alpha 10 for all three). On this exposed development comparison, selection is nearly tied with `I` for building damage, improves A10 slightly for structural flooding, and falls materially below `I` for road damage. The results do not support a general improvement claim. The `I-only` control exactly matches `I` in all three task aggregates; `all-fields` is also saved. `reversed-I` and `shuffled-semantic-vector` are included as negative controls, and both are far below `I`/selected for A10 in the saved summary.
Paired differences were also bootstrapped over the same useful C pools (mean repetitions within stratum, then equal strata within each pool; 1,000 resamples, seed 20261024). These intervals are exploratory because C is historically exposed development data; they do not select a final winner.

| Task | Pools | Selected − I, delta [95% bootstrap interval] | Pools | Selected − all-fields, delta [95% bootstrap interval] |
|---|---:|---:|---:|---:|
| building_damage | 8 | +0.00031 [−0.00500, +0.00744] | 8 | +0.00029 [−0.00554, +0.00548] |
| road_damage | 5 | −0.10767 [−0.36400, +0.11900] | 5 | +0.03333 [0.00000, +0.10000] |
| structural_flooding | 8 | +0.00940 [−0.00947, +0.03125] | 8 | +0.00041 [−0.00919, +0.01311] |


The closed-form singleton optimum and `oracle-unit-greedy` A10 agree for all 40 C episodes in each task (maximum absolute difference 0.0). The saved artifacts do not contain each episode's candidate permutation, so an independent rank-by-rank manual replay was unavailable; the independent numerical comparison was at A10 and pool-aggregate level. Oracle rows are upper-bound references, not deployable policies.

Support is important when reading the table: per model, A10 and coverage are undefined for 1/40 building episodes, 19/40 road episodes, and 4/40 structural-flooding episodes. High-recall is undefined for 10/40, 19/40, and 18/40 episodes, respectively; redundancy is defined for all 40. The raw episode CSV retains blanks for undefined values, and aggregate calculations were independently recomputed using the defined per-pool rows.

## Contract and scale checks

Across all three task fixture records, the saved mechanics confirm `N=1` for empty transmitter history, `N=0` for an exact same embedding, and `N=0.5` for an orthogonal embedding; the cold-start/orthogonal values are the expected fixture outputs. The 8-bit header half-LSB bound, confident-absence `Q`/`I` behavior, and no-backlog invariance checks also passed. For C evidence, freshness `F` was unavailable in all 1,280 rows per task and had no occupied wire levels, consistent with untrusted timestamps remaining unavailable.

Near-best subset inclusion is not feature importance. Unavailable F, and unavailable road Q, create functionally equivalent grid configurations; their inclusion in half of a near-best collection can simply reflect enumerating both versions. Fold-specific choices vary and do not establish stable universal parameters.

## Additional header-to-queue behavioral replay

The main thread completed this bounded check after the validation delegate became unavailable. The delegate had drafted `queue_replay.py`; main inspection corrected two script-only issues before execution: transport tokens are source-local, whereas candidate sequence is the randomized receiver tie order, and the Git helper belongs to the audit module. Production modules and frozen pilot source remained unchanged.

Run from the repository root:

```powershell
..\SemanticSchedulerEnd2End\.venv\Scripts\python.exe results/scheduler_review/20261024T071700Z/queue_replay.py
```

The replay regenerated one defined C episode per stratum/task, totaling 15 episodes. It passed 45 policy-episode checks for FIFO, RR and I: serialized headers entered `DecoderQueue` with opaque payload references, each 32-job queue drained, all returned orders matched the benchmark, and independently calculated A10 matched the saved CSV. Traces preserve input headers, orders and dispatch records in `behavioral_replay/queue_replay.json`; `queue_replay_console.log` preserves execution output. Source hashes were unchanged before/after. This tests dispatch behavior, not actual decoder execution or completion-state ownership.

The missingness fixture also reproduced RR's existing fallback: with all semantics unavailable it serves source order `[u1,u1,u2,u2]` as indices `[0,1,2,3]`; making one zero-valued field available restores RR order `[0,2,1,3]`. This must be accounted for in missingness experiments; it does not explain current natural-cache rankings where semantic fields are present. No production fix was applied during this review.

A separate direct `freshness` plus `ParameterHeader` round-trip check used the default `tau_ms=30000`. The uint8 F value rounds to zero beyond `30*ln(510) = 187.0323` seconds: age 180 s produced byte 1, while 188 s, 300 s, and 3,600 s produced byte 0 with the availability bit still true. A receiver with a trusted capture timestamp may recompute float F, so this is specifically a wire-quantization/default-timescale concern, not a claim that every receiver always sees zero. An untrusted timestamp remained unavailable and serialized with capture timestamp zero.

## Artifacts

Run evidence is under `results/scheduler_review/20261024T071700Z/`:

- `preflight.json`, `environment.txt`, and source-hash snapshots record the checkout/runtime and test/pilot hashes.
- `existing_tests.log`, `existing_tests.command.txt`, `existing_tests.timing.txt`, and `existing_tests.junit.xml` preserve the existing-test run.
- `pilot_tests.log`, `pilot_tests.command.txt`, `pilot_tests.timing.txt`, and `pilot_tests.junit.xml` preserve pilot unit tests.
- `full_pilot_result/` contains run provenance/cache hashes, final progress state, summaries, per-episode CSV, per-task grids, fold scores, fold memberships, selections, and analytic optima. `full_pilot_stdout_stderr.log` preserves progress output.
- `full_pilot_independent_validation.json` and `independent_validation.log` record the 210-check audit; `metric_support.json` contains per-model/per-metric defined and undefined support counts. `development_paired_comparisons.json` contains the exploratory paired C intervals. `all_json_audit.json` records that all 32 run JSON files parsed with 20,487 finite numeric values and 26 intentional nulls.
- `parameter_scale_checks.json` and `parameter_scale_check.log` record the independent F-scale round trip. `final_manifest.json` and `git_status_after.txt` record final source identity and checkout state.

No test expectation, implementation, dependency, objective, or remote artifact was changed by this validation run.
