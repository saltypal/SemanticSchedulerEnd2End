# Critical review of the semantic decoder scheduler

**24 October 2026.** This is an evidence report supporting the [canonical execution plan](../../../SemanticSchedulerEnd2End/docs/decoder_scheduler_redesign/07_MASTER_EXECUTION_PLAN.md). It records faults, tests and decisions; it is not a second implementation roadmap.

## 1. Honest assessment

**The receiver-ordering problem is coherent. The present results do not establish a general semantic-aware scheduler or journal-level novelty.** The code and prior experiments are useful progress: they provide reproducible extraction, serialized metadata, ranking baselines and a tested controlled benchmark. The weak point is the link between the benchmark's labels and the intended semantic decision.

The current study measures priority under native LADI image labels and exact-copy redundancy. It does not establish urgency, real-world fact overlap, cross-event generalization, actual latent decoding, or improved disaster decisions. Those limitations should determine the next experiment. Adding reinforcement learning, attention or extra datasets before addressing them could produce a larger study with the same unresolved validity problem.

The recommended direction is **task-conditioned utility estimation with explicit delivered-evidence context**, evaluated against calibrated simple policies. A contextual tree/linear model with transparent overlap summaries should precede a neural set model. Independent semantic correspondence and an actual decoder replay are the important new evidence.

The previous master plan nominated a neural set scorer and specified an optional descriptor layout before proving that the available evidence could support the target distinction. That ordering was premature. The revised plan makes the semantic benchmark and transparent contextual baseline decisive; architecture complexity follows measured need.

## 2. FIFO and round robin are not winning the primary metric

These are **historical saved test results**, freshly read from baseline commit `2f16785a16dd9c58ce12c98b502087cc212b279b`, path `results/disaster_decoder_v3/20261023_stageB_v1/full/study/comparison.csv`. They are not results from the new pilot. The table is the unweighted macro mean of the three reported task values.

| Policy | A10, higher is better | Critical-unit recall@10, higher is better | Unweighted repeat fraction, lower is better |
|---|---:|---:|---:|
| FIFO | 0.2503 | 0.4036 | 0.0842 |
| Round robin | 0.2500 | 0.4267 | 0.0776 |
| Direct importance I | **0.6622** | **0.8379** | 0.2386 |
| Neural coverage ranker | 0.6417 | 0.7538 | 0.1870 |
| Ordinary neural RankNet | 0.6244 | 0.7814 | 0.1968 |
| Coverage-gradient tree | 0.6471 | 0.7732 | 0.2316 |

FIFO/RR have fewer repeats partly because they select more unimportant items. A low duplicate fraction is not sufficient semantic success. The valid criticism is that additional learned policies have not consistently improved on direct I. Each metric also has its own defined-case denominator; no-high-importance and zero-useful-utility episodes must be reported separately.

Even if FIFO genuinely wins on a future semantic endpoint, do not change the endpoint to force a semantic model to win. Possible explanations include inverted/miscalibrated importance, distribution shift, hidden source bias, little useful priority variation, no queue backlog, insufficient runtime information, or an actually poor learned policy. Each calls for a different control.

## 3. Fault and evidence ledger

| Finding | Classification and evidence | Consequence | Smallest corrective action |
|---|---|---|---|
| “General semantic importance” is currently a narrow native-label rubric | **Confirmed scope limitation.** `decoder_v3_study.native_utility` and `decoder_parameters.PROFILES` use the same LADI label structure/weights that define the I producer | Predicting a native label is valid, but does not validate action urgency or generic semantics | Declare tasks, independently review useful evidence and importance, test other domains under the same rubric |
| Different images are treated as different evidence units | **Confirmed benchmark limitation.** `make_episode` assigns `units=row['image_id']` | Two views of one building receive separate credit unless they are the same original; a similar-looking different building is not a tested hard negative | Build a verified asset/condition correspondence panel; keep exact copies as a distinct control |
| Current oracle planning headroom is zero | **Mathematical consequence, checked numerically in pilot.** One nonnegative weighted unit per job, equal service opportunities, deterministic success | Sophisticated sequence optimization is not required when unit identity/weight are known | Use exact distinct-unit greedy as the oracle; defer RL until a richer valid problem has measurable planning headroom |
| Some finalized parameters lack evidence or have ambiguous meanings | **Confirmed.** F unavailable; road Q lacks calibration support; P is area, C is detector count/entropy; N is sender-history distance | Eight implemented fields do not imply eight useful independent semantic signals | Validate construct, availability, incremental value, stability and shared extraction cost separately |
| Default freshness scale is unsuitable for some plausible deployments | **Controlled numeric finding.** tau=30 s and uint8 quantization make wire F zero after approximately 187.03 s | Older valid photos become indistinguishable in the transmitted F value | Derive the scale from the task/time objective; use the existing trusted timestamp where available; validate temporal use separately |
| A new sender's first image gets N=1 | **Confirmed intended definition, interpretive risk.** Empty causal KB returns 1 | Cold start is confounded with strong novelty if a policy interprets N as absolute new-event probability | Preserve N, expose/handle its existing KB-version context and test cold-start versus established senders |
| Current scoring does not inspect delivered evidence | **Confirmed architecture limitation.** `SemanticScheduler.scores` evaluates each candidate independently | A context-sensitive label/loss cannot recover a relationship unavailable to the scorer | Compare explicit overlap features and contextual scoring with identical information |
| Three task-specific models do not establish one general policy | **Confirmed scope gap.** Profiles and producer prompts are fixed; per-task selections are separate | Cross-task reuse and unseen-task claims remain untested | Compare a shared conditioned policy, a shared scorer without explicit task ID, and validation-selected task specialists |
| Existing final test has influenced subsequent design | **Confirmed development exposure.** Results have been inspected repeatedly | It cannot serve as an untouched final test of this revision | Use new locked groups for final evaluation; old data remain development evidence |
| End-to-end codec integration is not established by the scheduler demo | **Confirmed.** Pinned `scripts/demo_decoder_scheduler.py` explicitly uses observation references and `codec_executed=False` | Queue draining does not prove a latent was reconstructed or its facts preserved | Connect one existing fixed codec/profile and measure recovered evidence and completion times |
| Runtime lifecycle is caller-owned but completion contract is not demonstrated here | **Integration evidence gap.** Queue dispatch removes the job; decoder outcome belongs to the caller | History could be updated at dispatch or on a failed decode unless the caller contract is enforced | One in-flight owner, exact-once matching completion, success-only delivered history |
| RR becomes FIFO when all semantic fields are unavailable | **Confirmed boundary behavior.** The all-semantics-unavailable fallback occurs before the RR branch in `SemanticScheduler.rank` | A missingness stress test labeled RR may actually test FIFO; current natural-cache results are unaffected because some fields are available | Separate semantics-independent baseline behavior from learned-policy fallback before new missingness experiments |
| Generalized event/site independence is unknown | **Confirmed metadata gap.** Current extracted manifest has null event/capture fields | Near-duplicate grouping is weaker than unseen-site/event validation | Recover upstream provenance; group across dataset names, not just inside each source |

This review did not find evidence that the basic coverage implementation should be discarded. The principal problem is what its current units stand for. A reproducible but limited benchmark is more useful when its limitation is explicit.

### Exact optimum in the current benchmark

If each unique observation u has weight w_u, its first scheduled occurrence at rank t contributes `w_u * (K-t+1)/(K*W)` when t<=K. A repeated occurrence contributes zero. Moving a positive-weight unseen unit before a repeat never decreases the objective. Swapping two unseen units so the larger weight is earlier never decreases it either. Therefore one occurrence of each unit in descending weight, then repeats, is optimal.

This theorem assumes the current equal-cost, all-arrived, successful-decoding, singleton-unit setting. It does not establish the optimality of predicted I, a signature matcher, or a learned overlap estimate.

For actual sets of nonnegative weighted facts, union coverage is monotone submodular. Exact marginal greedy has a classic cardinality guarantee, which can be applied to each prefix and summed for A_K. That supports testing greedy policies seriously; it does not transfer automatically to arbitrary neural gain predictions. See the [original submodular maximization result](https://thibaut.horel.org/submodularity/papers/nemhauser1978.pdf) and the detailed deduction in [research evidence](RESEARCH_EVIDENCE.md).

## 4. What a substantially semantic comparison should contain

Keep three evidence tiers, with independent result tables:

1. **Native utility:** image-level damage/relevance labels; tests basic task-conditioned ordering.
2. **Controlled overlap:** exact copies and clearly labeled known-parent crops/transformations; tests algorithm mechanics. Augmentations are not natural different-camera-view evidence.
3. **Reviewed semantic correspondence:** same asset/same condition, new view adding a fact, changed condition, and similar-looking different assets. This tests the actual proposed semantic contribution.

Define a visible-evidence fact using event, asset, condition and condition episode. Fix the parent/child credit rule so one verbose annotation cannot create ten times the utility. Visible image evidence is not automatically verified physical truth. Contradictory observations, corroboration and confidence improvement require a distinct objective if the application values them; the initial fact-union objective gives repeated confirmation no extra credit.

Primary evaluation remains early weighted distinct-evidence coverage. Supporting measurements should include critical-fact recall, weighted redundant selections, useful evidence incorrectly deferred, false/unsupported claims after actual decoding, time to first critical evidence, and per-group failures. A CLIP similarity score or a VLM's preference alone cannot validate these outcomes.

Use paired traces with the same offered images, task, headers, decode outcomes and compute opportunity budget. Include:

- Original metadata, permuted metadata and reversed importance controls.
- No backlog, equal-importance/no-overlap and deliberately meaningful-overlap regimes.
- Source assignments independent of labels, plus a separately declared correlated-source stress test.
- Different queue sizes and K values after the base case is correct; an arbitrary K=10 is a controlled budget, not a measured emergency-response deadline.
- Semantic label/rubric sensitivity without selecting whichever rubric favors the proposed method.
- Outcome review blinded to scheduler name and parameter values.

Source-preserving random queues represent the sampled source cohort, not a validated real disaster ingress distribution. Natural and enriched challenge samples need separate reporting. Do not call more generated queue repetitions more independent incidents.

## 5. Dataset portfolio and generalization claims

Use a small portfolio with complementary annotation capabilities, not an indiscriminate union of images.

| Cohort | What it adds | What it does not supply |
|---|---|---|
| [LADI v2](https://arxiv.org/pdf/2406.02780) | Existing aerial baseline and infrastructure/damage labels | A validated emergency priority scale or guaranteed cross-view asset identity |
| [MEDIC](https://crisisnlp.qcri.org/medic/) **or a clearly tracked CrisisMMD subset** | Ground/social imagery, informativeness, humanitarian/damage labels, hard negatives | Independent asset/fact correspondence; MEDIC and CrisisMMD are overlapping sources |
| [RescueNet](https://arxiv.org/html/2202.12361v4) | UAV scene labels with building damage and blocked/clear roads | Multiple independent disasters; it is a Hurricane Michael cohort |
| [FloodNet](https://arxiv.org/pdf/2012.02951) as an alternative first UAV cohort | Flooded roads/buildings and scene/VQA labels | Cross-view tracking or broad event generalization; it is a Hurricane Harvey cohort |
| [xBD/xView2](https://github.com/DIUx-xView/xView2_baseline), later if required | Satellite building polygons and pre/post damage | An independent second dataset for each name, or same-condition repeated views by default |

A feasible development allocation is **360 images: 120 LADI, 120 MEDIC, 120 RescueNet**; substitute FloodNet when flood access is the chosen second task. This is a proposed allocation, not downloaded/annotated data or a sufficient final sample size. First inspect 20 image/pair candidates to measure provenance and correspondence yield. Aim for a 60–100-image verified correspondence panel inside the allocation only if real matches exist. Reserve final evaluation groups separately.

Apply the same two reviewed profiles across source domains: building-damage evidence and road-obstruction/flood-access evidence. A MEDIC infrastructure label alone cannot supply the second profile's truth. Keep absent, irrelevant and uncertain states. Frozen aerial producers may require training-only calibration/adaptation on ground imagery; report adaptation separately from zero-shot transfer.

Group by actual event/site/flight and connected near-duplicate/correspondence components across source names. Harvey and Michael recur in different public datasets. Dataset-held-out therefore does not necessarily mean event-held-out. The extracted LADI cache lost event/time provenance even though upstream sources may contain metadata; recovering it is a concrete task, not an invitation to invent it.

Define “general” as a ladder:

- **G1:** unseen sites/groups under a known task and domain.
- **G2:** another source domain under the same reviewed task.
- **G3:** one reusable policy for multiple declared tasks, tested on unseen groups.
- **G4:** a held-out task with compatible runtime task representation and no target-task tuning.

Target G1–G3 first. Do not claim G4 because separate models work on three LADI profiles. Shared task-conditioned producer outputs already carry task information, so an “unconditioned” scorer means no explicit task ID; it is not necessarily task-agnostic.

## 6. Better policy comparison

The strongest immediate alternatives are:

1. **Direct I and calibrated standalone utility:** linear, small boosted-tree and pointwise neural models. The same information/tuning allowance applies.
2. **Utility plus calibrated overlap penalty:** an interpretable MMR-style baseline, with an independently learned directional overlap target. [Original MMR](https://aclanthology.org/X98-1025.pdf).
3. **Contextual tabular marginal-gain prediction:** candidate values/masks plus explicit distances to delivered headers, signature-match flag, maximum calibrated overlap and history count. These features are computed solely from permitted runtime information. A small tree or linear interaction model is the first learned contextual candidate.
4. **Deep Sets contextual prediction:** test whether learned summaries add benefit beyond the same contextual information and transparent relationships.
5. **Two-step lookahead:** use the same predicted gain model; isolate planning benefit from better observations or a larger model.
6. **PPO only after a planning gate:** compare against both learned greedy and cheap lookahead using the same observation contract. The current singleton benchmark cannot supply that gate.

The optional H1 descriptor needs pointwise and contextual baselines within H1. Beating H0 by exposing extra information is a useful information-cost result, not proof of a superior scheduling algorithm. Do not interpret visual similarity as fact identity.

For an eventual equal-wall-time experiment with unequal service costs, include predicted semantic gain per unit service cost. In the simpler independent-job/all-served setting, descending weight/processing-time is Smith's rule for minimizing weighted completion time; arbitrary truncated horizons and overlapping facts need separate analysis. [Original scheduling paper](https://onlinelibrary.wiley.com/doi/10.1002/nav.3800030106). This remains semantic utility delivered early, not throughput maximization.

## 7. Empirical parameter selection and grid search

Assess a parameter on five axes: its intended construct, availability/calibration, incremental scheduling value, stability under group/domain shift, and removable extraction cost. Correlation with labels alone does not resolve these questions.

Treat each scalar and its validity mask as one inclusion bundle. Keep the eight-field wire contract and sender KB while testing subsets. An ablation of N from a scorer does not delete the KB. If future design requires N in every candidate scorer, constrain to the 128 N-containing subsets and retain no-N only as a diagnostic.

The current pilot exhaustively searches 255 subsets with Ridge alpha {0.1,1,10}, using three A-only group folds: **6,885 inner fits across three tasks**, plus nine final refits for selected/I-only/all-fields controls. B-only calibration is a fixed separate input. C is exposed development evaluation after selection. This is an intentionally bounded one-family search, not a final journal estimate.

Unavailable fields create functionally duplicate configurations. The 6,885 fits are not 6,885 independent scientific experiments. In particular, F appears in roughly half of some near-best subset collections simply because adding its constant unavailable input makes no difference. Near-best inclusion frequency is not feature importance; inspect availability and functional equivalence first.

For the new cohort, use nested grouped evaluation of the **whole selection procedure**. The inner fold fits preprocessing, any producer calibration/adaptation, descriptor projection, overlap estimator, feature subset and policy hyperparameters. Outer groups assess the selected procedure. Do not pick a subset using outer results and reuse them as its unbiased score. Final locked groups are used once after procedure freeze. [Original analysis of selection bias](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf).

Suggested bounded search:

| Stage | Search | Selection rule |
|---|---|---|
| Cheap subset screen | All 255 bundles; Ridge alpha 0.1/1/10 | Inner grouped A_K; numerical ties prefer fewer fields and stronger regularization |
| Nonlinear check | Prespecified I/full plus a small inner-selected shortlist; tree leaves 3/7/15, L2 0/1/10, fixed training-round budget | Same inner folds; no outer-driven shortlist |
| Context strength | Overlap penalty 0/0.25/0.5/1 or a calibrated gain model | Inner semantic endpoint with critical-evidence guardrail |
| Neural comparison | Few prespecified widths/regularization values | Same information, split and bounded tuning allowance |
| Robustness | Grouped permutation, missingness-matched controls, cold-start KB, source/domain shift | Report sensitivity and subgroup failures, not a new winner chosen on final data |

Do not grid-search gold fact weights, evaluation metrics or dataset composition to make the new policy win. Those define the task and should be justified independently. Do not run every subset against every architecture and every producer setting. Shared extraction dependencies matter: removing R does not save CLIP cost if N/H1 still uses it; removing P/C may remove a detector only if no other retained field needs it.

Report fold selection stability, near-best subsets, effect sizes, per-metric denominator support and cost/utility trade-offs. “Best parameters” means best under a declared task, source, observation contract and selection procedure, with uncertainty.

## 8. Pilot and validation record

The additive implementation is [decoder_review_pilot.py](../../INFRA/learning/decoder_review_pilot.py), with [focused reference tests](../../INFRA/tests/test_decoder_review_pilot.py). Existing production modules remain unchanged. Sol Low wrote the bounded harness; a Medium checkpoint assessed isolation/metrics; Luna High independently executed the authorized tests and full grid. The main review traced the source contracts and interprets the findings.

Use [VALIDATION.md](VALIDATION.md) for exact run paths, hashes, commands, checks, elapsed times and exclusions. [PILOT_ASSESSMENT.md](PILOT_ASSESSMENT.md) records the assessment-only review. Per-metric support must be derived from raw rows because the pilot's summary lists undefined A10 episodes but not every metric's denominator.

### New measured development results

The full pilot completed in **80.084 seconds on CPU**. Existing component tests: **58 passed, 3 skipped** because LightGBM was unavailable; the six new pilot tests passed. The existing audit tests emitted 24 scikit-learn scaler warnings; they remain recorded, and the new pilot explicitly handles constant columns. A separate output audit checked 210 consistency conditions. Counts of checks are not counts of independent scientific experiments.

| Task | FIFO A10 | RR A10 | Direct I | Training-fold-selected subset | Selected A10 | Selected minus I, exploratory 95% pool interval |
|---|---:|---:|---:|---|---:|---|
| Building damage | 0.3058 | 0.3081 | 0.7034 | R, I | 0.7037 | +0.00031 [−0.00500, +0.00744] |
| Road damage | 0.1657 | 0.2153 | 0.7197 | S, I, C | 0.6120 | −0.10767 [−0.36400, +0.11900] |
| Structural flooding | 0.1590 | 0.2100 | 0.7856 | R, I, P, C, N | 0.7950 | +0.00940 [−0.00947, +0.03125] |

All selected regularization values were alpha=10. Selection used A folds; C was scored afterward. These numbers differ from the historical table because this is a different development workload with one repetition per stratum. C remains historically exposed. There are only 8/5/8 useful comparison pools, not thousands of independent events. The I-only learned control reproduced direct I ordering.

**Conclusion:** the grid does not establish a superior universal subset. It nearly ties for buildings, degrades road performance descriptively and gives an uncertain small flooding improvement. No selected subset replaces the practical direct-I reference on this evidence.

Fold selections were unstable. For example, building folds preferred I/C/N, then S/I/N, then S/Q/I/N; the aggregate choice was R/I. This supports more independent groups and better construct validation, not a larger hyperparameter search on the same exposed data.

The exact singleton oracle matched the analytical optimum with zero numerical difference. Negative controls were substantially below I. No-backlog service was policy-invariant in the controlled fixture. These results support the correctness/sensitivity of the basic priority benchmark while confirming its narrow semantic scope.

For building damage, reversed I scored 0.0010 and shuffled semantic vectors scored 0.2531 against I's 0.7034. The benchmark is therefore sensitive to useful ordering information; poor parameter/model transfer cannot be explained by asserting that every input is semantically identical. Signature deferral scored 0.7727 / 0.7550 / 0.8316 across tasks, but its advantage remains an exact-copy shortcut until cross-view correspondence is tested.

A further serialized-header → receiver queue → policy replay passed **45 policy-episode checks** across 15 regenerated C episodes. Orders and A10 matched the saved benchmark; opaque payload references were used and no codec executed. The main thread completed this check after the validation delegate became unavailable. The replay also confirmed RR's all-semantics-missing FIFO fallback, which is documented for correction before missingness studies.

Per-policy A10 is undefined for 1/40 building, 19/40 road and 4/40 flooding episodes. Critical-unit recall is undefined for 10/40, 19/40 and 18/40 respectively. The complete denominator report is [metric_support.json](../../results/scheduler_review/20261024T071700Z/metric_support.json); the paired effects are in [development_paired_comparisons.json](../../results/scheduler_review/20261024T071700Z/development_paired_comparisons.json).

No fresh image-model inference, new dataset annotation, actual latent decoding or emergency deployment validation is established by these CPU tests. Cached real-image features are real evidence; exact-repeat queues and numerical fixtures remain constructed experiments.

## 9. Scope coverage and remaining work

Inspected execution path: pinned extraction manifest and split logic → cached producer outputs → causal KB/parameters → serialized header → policy input/scoring → ingress/dispatch → study target and coverage metric → feature fitting/selection → persisted results. The pinned terminal demo was inspected through Git because the audit checkout is sparse. Its explicit no-codec scope was verified.

Fresh producer execution, actual image labels/correspondence, full codec inference, long-running queue service, event-level holdout validity and field deployment remain untested. A passing test suite cannot supply those missing experiments.

The next useful work is a small independently reviewed cross-domain/correspondence cohort alongside completion-aware replay. The first learned comparison should be transparent contextual gain prediction versus direct I and calibrated utility/overlap baselines. Keep the neural and RL candidates conditional on what those comparisons reveal.
