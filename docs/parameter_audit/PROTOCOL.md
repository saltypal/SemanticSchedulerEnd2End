# Empirical audit of the eight semantic scheduling parameters

Date: 24 October 2026. Status: authorized implementation/validation protocol. This document records the experiment design before examining new audit results. It is a focused increment of the scheduler's revised plan, not a new encoder or radio architecture.

## Decision to make

Determine whether poor learned scheduling is predominantly consistent with weak/incorrect feature measurement, redundant or insufficient header information, limited training data/model fit, or a benchmark that offers little benefit from contextual ordering. An ablation cannot establish those causes by itself; triangulate the checks below.

Retain the existing implementation and 32-byte header throughout the audit. Treat R,S,Q,I,F,P,C,N as hypotheses about useful information. Do not invent a result requiring all eight to help, and do not delete a feature because one small proxy dataset fails to show an advantage.

## Three questions, three kinds of evidence

| Question | Measurement | What a positive result establishes |
|---|---|---|
| Does the parameter measure its stated concept? | Independent reference annotations/calibration, units and evidence provenance | Construct validity within the assessed domain; not automatically scheduling value |
| Does it add predictive value? | Matched-model add-one/drop-one probes, direct-I control, masks-only control, paired early-coverage differences | Conditional usefulness on this controlled workload; not a causal or universal effect |
| Does the representation preserve necessary distinctions? | Header collision audit, controlled overlap counterexample, signature heuristic and privileged references | Identifiable information limitations and benchmark shortcuts; not their universal prevalence |

These questions must not be collapsed into correlation with the training target. For example, image area can be measured correctly and still be useless for a given task. Conversely, a biased proxy can predict a dataset label without measuring its advertised meaning.

## Per-parameter interpretation and reference requirements

| Field | Current measurement | Cheap diagnostic now | Independent evidence required before a semantic claim |
|---|---|---|---|
| R: context relevance | Mapped CLIP image/task cosine | Range, available levels, association with native task labels, add-on to I | Blinded relevance judgments against one fixed task, including visually similar irrelevant examples and relevant small targets |
| S: scarcity | Local eligible-concept history counts/self-information | Availability, count dependence, relation to N, add-on/drop-out and masked controls | Observed local concept counts under a specified history; distinguish rarity from consequence and receiver need |
| Q: reliability | Calibrated confidence in the critical binary claim, including confident absence | B-fitted calibration, C confidence-versus-correctness bins/Brier/ECE, class support | Correctness of the actual claim; high confidence in an absence is not high disaster importance. Gates alone do not prove calibration |
| I: task importance | Frozen LADI probabilities mapped to profile utility | Utility prediction error/class prevalence, direct-I ordering and learned-I calibration control | Independent task-consequence rubric; native label utility remains a damage-review proxy, not emergency triage truth |
| F: freshness | Trusted capture time through a declared decay | Availability and clock provenance; unavailable means untestable | Actual capture timestamps and a task-specific reason age changes utility. Synthetic time validates code only |
| P: spatial prominence | Largest retained target-box area / image area | Detection availability, zeros, saturation, class specificity and incremental value | Independently checked target boxes/areas. Image area cannot establish metres, proximity, severity or urgency |
| C: scene complexity | Normalized detector count plus class entropy | Count/entropy dynamic range, correlation with P/I, incremental value | Reviewed object presence/count/relationships under a fixed complexity definition. Dense irrelevant clutter need not be valuable evidence |
| N: transmitter novelty | One minus maximum embedding similarity, rescaled | Causal KB checks, dynamic range, relation to sender history and receiver-new controlled observations | Same-asset/same-condition versus changed-condition correspondence and distinct-but-similar scenes. Cosine distance alone does not establish a new disaster |

Novelty stays at the transmitter. Receiver-new labels in the audit are evaluator-only checks of the sender/receiver knowledge mismatch, not a new runtime parameter.

## Baseline, data and compute boundary

- Source baseline: `new-scheduler`, commit `2f16785a16dd9c58ce12c98b502087cc212b279b`.
- Work in the isolated sibling checkout `SemanticSchedulerParameterAudit`; preserve the dirty original checkout.
- Use existing frozen-feature cache `results/disaster_decoder_v3/20261022_stageA_v3/extraction/`.
- Verify feature SHA-256, schema, native-label consistency, duplicate groups and source cache manifest before analysis.
- **A fits probes and preprocessing. B fits only the original parameter calibration. C is development diagnostic evaluation. The old test role is not used by probes, calibration assessment or scheduler comparison.** Parsing the complete persisted cache for integrity is disclosed separately from using its labels in an experiment.
- C has previously served model selection; do not call it untouched, final or unbiased publication holdout. Future final claims require genuinely unexposed groups, not merely new queue seeds.
- CPU-only, existing project Python environment, no new dependencies, image extraction, VLM calls, GPU training or external publication.
- Initial configuration: seed 20261024, 64 A training queues per task, two C repetitions per fixed stratum, 1,000 paired pool bootstrap draws. Record actual counts and elapsed time. These are pilot work allocations, not statistical power guarantees.

## Experiment A: measurement coverage and information content

Report each field's available count/fraction, distinct quantized values, available-value histogram, endpoint mass, zero fraction, median and spread. Missing values are not observations of zero. Report candidate-exposure statistics separately from unique source-image counts because the queue generator reuses images and changes history-based fields.

Report pairwise correlation only where both fields are available, with pair counts; constant/no-data correlations are undefined, never silently zero. Correlation is descriptive. Do not infer redundancy solely from high correlation, since nonlinear interactions or different missingness can matter.

Assess Q confidence against correctness using only C for the diagnostic and B for its disclosed fitted calibration. Road Q may remain unavailable; F is expected to remain unavailable naturally. Unavailable or constant fields are marked **not empirically testable here**, rather than useful/useless by decree.

## Experiment B: conditional predictive value under matched capacity

Fit two small, fixed probe families:

1. Ridge regression, alpha=1, with A-only scaling.
2. Histogram gradient boosting regression, fixed 80 iterations, seven leaves, L2=1, no automatic validation split/early stopping.

Target: native standalone utility divided by three. This is a pointwise label proxy; it does not claim independently reviewed real-world marginal semantic utility. Weight each training source image's total contribution equally across its repeated candidate exposures. Use A-only imputation medians, and input a selected parameter's value **and** availability bit together.

For each family fit:

- I only.
- All eight.
- I plus each of the seven other fields, one at a time.
- All eight minus each field, one at a time.
- Availability bits only, as a dataset/missingness diagnostic.

Also evaluate direct I and FIFO. Learned-I versus direct-I separates recalibration/model effects from extra-field value. Cross-family consistency is supporting evidence; it does not prove necessity. Rank with raw probe outputs under the declared stable tie rule; clip predictions only for bounded utility-error reporting. No C-based hyperparameter sweep or field subset selection is part of this pilot.

For feature j and model family m, report:

\[
\Delta_j^{add}=A_{10}(m[I,j])-A_{10}(m[I]),
\qquad
\Delta_j^{drop}=A_{10}(m[all])-A_{10}(m[all\setminus j]).
\]

An all-feature model beating a differently sized/nonlinear I model would confound feature value with model design; matching probe families reduces that problem. These remain predictive ablations, not causal interventions on image meaning.

## Experiment C: header aliasing and duplicate shortcuts

Compare exact 16-feature tuples and a fixed R/Q/I/P/C signature that excludes sender history and freshness. Include profile, masks and quantized available values; disable the copy heuristic when fewer than two signature fields are available.

Distinguish:

- repeated observations of the same original image;
- distinct originals sharing a signature;
- distinct originals with different evaluator utility sharing a signature.

A signature equality is not proof of semantic identity. Evaluate a header-only heuristic that defers already-served signatures while other signatures remain, orders otherwise by I, and eventually services every message. Evaluator IDs may measure false deferral but never enter the decision function.

Compare all results per no-repeat, local25, cross25, local50 and cross50 strata. Improvement solely on exact copies supports a controlled-copy optimization, not different-view semantic understanding.

Include a numerical same-observation/different-hidden-overlap example. Under balanced hidden worlds with opposite correct next choices and identical permitted observations, a policy cannot exceed 50% expected correct choice. This is an illustrative information-bound calculation, not a measured real-world prevalence claim.

## Experiment D: novelty target mismatch and oracle headroom

Under a seeded random completion prefix, label each remaining observation receiver-new or receiver-seen using original image IDs. Measure N's association with this controlled target, report both class counts and undefined AUCs, and stratify by repeat mechanism. A threshold of .5 may illustrate mismatches but is not tuned or interpreted as calibrated novelty probability.

Separately compare evaluator-only true-item-utility ordering and greedy unique-observation ordering. The first estimates useful headroom from better utility knowledge; the second additionally uses overlap identities. Neither is a deployed baseline. On this controlled benchmark a source image is an observation unit, not a verified physical fact.

## Aggregation, uncertainty and decision rules

Keep A10 as mean weighted distinct-observation coverage over completions 1–10. Zero-positive-utility episodes are undefined and counted. Never assign them perfect coverage or silently omit their prevalence.

Average repetitions within each stratum, then strata within source pool. Pair policies on the same pools for deltas and confidence intervals. These pools are disjoint original-image groups, **not independent events**. C's few positive road pools may make estimates highly uncertain. Keep per-task/per-stratum outputs; a macro is a declared descriptive summary.

Report all prespecified add/drop comparisons. No multiplicity-adjusted confirmatory inference is claimed. A feature's apparent best exploratory result is not a final selection rule.

| Evidence pattern | Allowed next decision |
|---|---|
| Missing or nearly constant field | Defer claims until measurement/data improve; retain its mask in the current wire contract |
| Consistent incremental gain with plausible construct validity | Retain as a promising feature and test on untouched groups |
| Little incremental gain, wide interval | Inconclusive; no forced removal or new architecture |
| No gain with precise practically small effects and high producer cost | Candidate for a simpler future profile, after task/generalization review |
| Weak utility prediction even with richer probe family | Investigate label/measurement information and data shift before RL |
| Same-header collisions with opposing desired choices | Record unresolved information; more ranking capacity cannot guarantee resolution |
| Signature baseline solves copy cases but fails meaningful different views | Replace the benchmark's claim, not its labels; prioritize reviewed correspondence |
| Context/sequential advantage is small after better utility prediction | Retain simpler scoring; do not add RL merely for novelty |

Keep an independent construct-review pilot pending. Use blinded tasks/rubrics and image IDs without predicted parameter values; target error cases plus a separately sampled representative subset so a challenge set is not misreported as prevalence. Actual images/metadata and two reviewers are needed before claiming those checks have been completed.

## Ownership, outputs and acceptance

Sol Light owns new audit implementation and focused test source. Luna High owns execution, logs, environment/artifact evidence and bounded validation; it does not modify expected outcomes to pass. The main thread owns scientific interpretation, integration review and any approved corrections.

Expected outputs: `summary.json`, `feature_stats.json`, `correlation.json`, `model_comparisons.csv`, `per_episode.csv`, `header_collisions.json`, `novelty_diagnostics.json`, `REPORT.md`, resolved configuration, code/data provenance and captured run/test logs. Expected files become verified artifacts only after execution checks.

Acceptance checks include missing-versus-zero behavior, A-only preprocessing, absence of test-role label use, weighted training exposures, valid output permutations, exact tiny coverage reference, paired pool aggregation, signature collisions, zero-utility/constant-field handling, reproducible configuration, and strict finite JSON. Record source hashes because the experiment adds uncommitted files to a pinned base.

No extractor/header replacement, production scheduler change, RL training, model superiority or real emergency validity is authorized or established by a successful audit command.
