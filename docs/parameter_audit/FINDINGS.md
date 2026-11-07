# Parameter audit: measured findings and next decision

**24 October 2026 — local CPU experiment completed.** Sol Light implemented the diagnostic; Luna High executed the tests, cache checks and real-data runs. The main thread reviewed the design/code and interpreted the evidence; a bounded Medium assessment identified the empty-calibration-partition bug that was corrected before final delivery.

**Recommendation:** reevaluate what the parameters represent and which task-specific information they add before investing in RL. Preserve the KB and existing header while doing this. The results support a knowledge-target mismatch and mixed incremental feature value; they do not establish that every parameter is wrong or that one new subset is best.

## 1. What was actually executed

| Item | Verified scope |
|---|---|
| Baseline | `new-scheduler`, commit `2f16785a16dd9c58ce12c98b502087cc212b279b`, isolated checkout |
| Cache | 1,934 original records; exact feature object SHA-256 verified and original manifest checks passed |
| Experiment roles | A=445 records for fitting, B=178 for original Q calibration, C=267 for diagnostics; test excluded before experiment label/model use |
| Effective training | 64 constructed queues per task; 432 original observation units represented in each task's training queues |
| Evaluation | 80 queues per task: eight disjoint source-image pools, five strata, two repetitions; K=10 |
| Policies/probes | Direct I, FIFO, signature deferral, two privileged references, and 18 feature variants in each of Ridge and histogram boosting |
| Final result size | 9,840 per-episode rows and 279 paired model comparisons |
| Final run | 29.0 seconds, CPU only; Python 3.12.14, NumPy 2.5.3, scikit-learn 1.9.1, SciPy 1.18.1 |
| Tests | Nine focused tests passed in 8.31 seconds |
| Undefined cases | 49 distinct task/episode cases had zero offered useful utility; their A10 remains undefined |

The complete persisted cache was parsed for integrity, including test records. Test records were then filtered out before calibration, task utility assessment and probe fitting/evaluation. The protocol distinguishes these operations.

C was already used historically for development/model selection. It is **development diagnostic data**, not a fresh publication holdout. Useful paired coverage pools were building=8, road=5, flooding=8. The other road pools offer no positive utility under this rubric. Generated queues are not independent events.

## 2. More parameters did not provide a uniform improvement

All values below are measured development A10, using the unchanged quantized headers. Ridge uses a fixed linear probe; boosting uses a fixed small nonlinear probe. Their labels are native standalone utility, not independent real-world fact correspondence.

| Task | Direct importance I | Ridge: I only | Ridge: all eight | Boosting: I only | Boosting: all eight |
|---|---:|---:|---:|---:|---:|
| Building damage | 0.6817 | 0.6817 | 0.6764 | 0.6735 | 0.6743 |
| Road damage | 0.7245 | 0.7245 | 0.8200 | 0.6497 | 0.7237 |
| Structural flooding | 0.7735 | 0.7735 | 0.7865 | 0.7703 | 0.7838 |

The road Ridge result shows that additional fields can help in this development sample. Building damage shows that adding them can also fail to help. The difference between learned-I and direct-I results also shows why a gain over a learned baseline alone need not beat the practical baseline.

Inspect all prespecified additions/removals and paired intervals in [model_comparisons.csv](../../results/parameter_audit/20261024T010201_real_final/audit_output/model_comparisons.csv). Small group counts, one fixed fitting seed and many exploratory comparisons prevent a defensible universal field ranking. No model or parameter subset was selected as a new deployment winner using these results.

## 3. N is a sender-history score, not receiver knowledge

For each queue, an evaluator simulated a random five-message completed prefix. Remaining messages were labeled receiver-new or receiver-seen by original observation ID. The policy never received these IDs as semantic features.

| Task | N versus receiver-new ROC AUC |
|---|---:|
| Building damage | 0.5151 |
| Road damage | 0.4965 |
| Structural flooding | 0.5218 |

These descriptive AUCs are close to chance in this specific controlled-observation protocol. They establish a practical warning about using the current N as a proxy for new receiver evidence. They do not test whether N correctly measures similarity to its own transmitter's history, whether it recognizes real new disasters, or whether it helps some other task.

**Keep transmitter novelty and the KB. Change the interpretation:** “unfamiliar to this sender's stored representations” is the supported definition. A low N can still be the receiver's first useful report. A high N can repeat evidence already delivered by another sender.

Per-stratum results and class counts are in [novelty_diagnostics.json](../../results/parameter_audit/20261024T010201_real_final/audit_output/novelty_diagnostics.json). No-repeat strata have only receiver-new observations, so their AUC is correctly undefined.

## 4. Some parameters cannot be validated on this cache

- **F:** unavailable for every candidate occurrence in all tasks. There is no measured freshness effect to learn here. This supports changing the data/claim, not declaring freshness inherently useless.
- **Q for roads:** unavailable because B has only six positive critical-claim examples, below the existing calibration gate. Building/flooding have ten each. The threshold is a protocol rule, not proof of calibrated reliability.
- **R:** spans only 14, 14 and 15 occupied eight-bit levels across the three tasks. This identifies limited dynamic range after the present cosine mapping/quantization. It does not prove that additional bits or a rescaling would improve decisions.
- **P/C:** current outputs are detector area and count/entropy proxies. Their correlation or predictive value cannot establish physical proximity, urgency, causal importance or a general definition of scene complexity. Those constructs still require independent annotation.

Q's confidence must remain separate from importance: confident absence of damage can produce high Q and low I. No universal “higher value means higher priority” rule is appropriate for all eight fields.

## 5. The copy-based benchmark has a useful shortcut control

A header-only heuristic defers an already-served quantized R/Q/I/P/C signature while other signatures remain, then eventually serves everything. It uses neither original image IDs nor evaluator labels for decisions.

| Task | Direct I | Signature deferral | Paired difference | Exploratory pool-bootstrap interval |
|---|---:|---:|---:|---|
| Building damage | 0.6817 | 0.7621 | +0.0803 | [0.0563, 0.1056] |
| Road damage | 0.7245 | 0.7585 | +0.0340 | [0.0128, 0.0595] |
| Structural flooding | 0.7735 | 0.8249 | +0.0514 | [0.0268, 0.0751] |

Exact copies reuse deterministic extractor outputs, so the heuristic exploits the benchmark's construction. Distinct original images can also collide. The audit records those collisions and false deferrals in [header_collisions.json](../../results/parameter_audit/20261024T010201_real_final/audit_output/header_collisions.json).

This is an important baseline for this benchmark. It is not verified semantic duplicate recognition across different camera angles, assets or disaster episodes. Do not promote it into the production policy on this evidence alone.

## 6. What to do next

1. **Preserve this audit as development evidence.** Keep the full add/drop comparisons, failure history and source/data hashes.
2. **Review the parameter definitions against independent targets.** Start with N's distinction between same fact/new view/changed condition, Q's specific claim correctness, and P/C's task relevance. Use blinded image/fact annotations; do not ask the same extractor to grade itself.
3. **Use a genuinely semantic development cohort.** Include multiple views of identified assets/conditions and similar-looking different sites, with documented correspondence. Exact copies stay a separate control. Reserve new unexposed groups for final evaluation.
4. **Test a smaller task-specific feature profile against the full profile.** Choose candidates from this exploratory evidence, freeze them before the new evaluation, and include extractor cost. Do not force eight positive contributions or remove the KB.
5. **Only then test the contextual scorer and RL.** The key gate is whether permitted observations predict marginal new evidence. More complex policies cannot guarantee distinctions erased by the header.

The human construct/correspondence review has not been completed in this automated audit. Actual decoded-image semantics and real codec service times also remain outside the measured scope.

## 7. Implementation, checks and reproduction

- [Experiment protocol](PROTOCOL.md).
- [Audit implementation](../../INFRA/learning/decoder_parameter_audit.py).
- [Focused tests](../../INFRA/tests/test_decoder_parameter_audit.py).
- [Full generated report](../../results/parameter_audit/20261024T010201_real_final/audit_output/REPORT.md).
- [Final run validation](../../results/parameter_audit/20261024T010201_real_final/post_run_validation.json).
- [Final test log](../../results/parameter_audit/20261024T010134_validation_final/targeted_pytest.log).

From the isolated checkout, use the existing environment and invoke the module:

```powershell
$auditPython = '..\SemanticSchedulerEnd2End\.venv\Scripts\python.exe'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $auditPython -m pytest INFRA/tests/test_decoder_parameter_audit.py -q
& $auditPython -m INFRA.learning.decoder_parameter_audit --cache results/disaster_decoder_v3/20261022_stageA_v3/extraction --output results/parameter_audit/NEW_RUN --seed 20261024 --repeats 2 --train-episodes 64 --bootstrap 1000
```

Use a new output directory for each run. An existing nonempty output is rejected. A direct `python INFRA/learning/decoder_parameter_audit.py` invocation fails package import; use `-m` from the repository root. The failed initial invocation was preserved separately before any real audit stage ran.

The nine tests cover missing-versus-zero handling, undefined/constant fields, collisions, valid selection order, exact small coverage, pool weighting, feature families, poisoned-test-label exclusion, empty calibration support, and actual A-only scaler/sample-weight instrumentation. Several checks share test functions; this is not a claim of nine independent experiments.

The final run's JSON and numerical CSV outputs are finite. Its 9,840 episode rows and 279 comparisons match the earlier completed run exactly after the empty-B correction. Scikit-learn emitted 27 weighted-scaler square-root warnings: targeted reproduction identified constant mask columns with tiny negative variance from roundoff (about -2.159e-44), unit scale and finite predictions. The warning and diagnosis are retained; no blind suppression or numerical model change was applied.

The isolated Git worktree reports the feature cache as modified because its verified 25,664,555-byte LFS content was materialized from the pointer. Its content matches the expected source-object hash; the original repository/data were not rewritten. New implementation files remain uncommitted and nothing was pushed.
