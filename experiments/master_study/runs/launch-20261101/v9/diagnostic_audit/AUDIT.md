# Version 9 scientific and implementation audit

Date: 2 November 2026 (IST). Scope: saved Kaggle v9 scheduler study. No training, model selection sweep, or new Kaggle execution was launched for this audit. Production code was not edited.

## Judgment

The run establishes that many learners can optimize and export models. It does not yet establish a generally semantic scheduler. The main experiment optimizes reference priority and queue waiting time, while the intended contribution also requires mission relevance, additional information, scene evidence, and causal history. Several mechanisms were implemented without the supervision and workload needed to demonstrate their value.

SPI-first winning is a meaningful result, not proof the benchmark is broken. We must improve validity and learning coverage, and accept that the simple policy may remain strongest. We must not revise labels, workloads, or metrics merely to obtain a learned-model win.

The important correction to the earlier inventory is that VLM-guided replay completed optimization, but its VLM guidance was entirely neutral. It is a completed TD-based mixed-replay run, not evidence that informative VLM feedback helped.

## Evidence and boundaries

Inspected: main orchestrator and configuration; input adapter; stream generator; environment and rules; supervised models/losses; portable inference; PPO/DQN training; teacher query generation and qualification; replay/reward training; saved statuses, predictions, histories, and reference split.

The six core source files used in numerical checks match their hashes in the v9 manifest. The checkout is dirty; no repository mutation or reset was performed. Diagnostic results are in diagnostics.json, missing_baseline_replay.json, validation_queue_support.json, and the two audit scripts in this directory.

At least one seed completed for all 14 ordinary learners and the replay-feedback configuration. Teacher-supervised rankers, VisionPrefer, and learned-reward RL were blocked. Final independent evaluation, the post-training rate study, and the adaptive second teacher round were not completed. Saved status files contain stale running/planned entries because the platform killed the process; they must not be interpreted as live jobs.

## 1. Confirmed objective mismatch: the semantic inputs have no direct semantic target

Evidence: Evaluation/semantic_learning_env.py:166-187, 203; experiments/stream_scheduler/train_real_suite.py:69-87.

The primary cost is:

    J_D = mean_i ell(W_i, delta_reference_i)
    ell(W, delta) = W/delta + max(W/delta - 1, 0)
    delta = [2, 4, 8, 16]

For supervised training, the label is the one-unit increase in this cost if a candidate waits longer. Only reference class and age determine it. Mission relevance, novelty, scene counts/areas, and delivered correspondence do not change the target for equal-class/equal-age reports.

Imported numerical check: two High reports with the same arrival receive [0.25, 0.25], despite different relevance, novelty and scene measurements. Both orders have identical total cost 0.25. Assigning the same fact identity does not change D.

Why it matters: the models may use scene information to correct predicted SPI, but they are not being taught that new useful evidence deserves earlier service than repeated evidence. Adding a history encoder or attention does not supply the missing target.

Fix: retain D as the priority-delay experiment. Add a separately specified mission/additional-evidence experiment with published task labels and declared exact-copy or controlled correspondence. Compare within-SPI, matched-age decisions. Keep the label construction separate from predicted runtime parameters. Do not make the evaluation utility a weighted copy of those same predicted parameters.

Acceptance: controlled cases in which changing independently supported mission/evidence value changes the target, and changing irrelevant object counts does not. Report class delays and misses alongside evidence outcomes. No arbitrary image-specific deadline invented by a VLM is needed.

## 2. Confirmed dataset filtering: the six-thousand-image pool was not the scheduler training cohort

Evidence: semantic_suite_data.py:37-39, 65-68, 126-132; train_real_suite.py:132-140; archived handoff and training_records joined by exact image hash.

The archive contains 4,000 CrisisMMD, 1,000 Open Images and 1,000 Places365 records. Only 3,915 CrisisMMD records have unambiguous reference priority labels. The scheduler development split contains 3,099 training and 780 validation images, all CrisisMMD.

Why it matters: ordinary-image diversity in the archive did not become supervised/RL priority-training diversity. The teacher queries also use a different cohort construction from D. General everyday/emergency or diverse-mission claims are unsupported.

Fix: publish a source/task/reference coverage matrix before sampling. Admit broader datasets only where labels support the intended target, or identify the labels as qualified silver supervision. Never silently turn an unknown ordinary-image priority into Low. Keep event/parent/near-duplicate groups separate. Keep the present CrisisMMD result as a legitimate scoped development result.

Acceptance: report actual eligible records and class/source counts after every filter; no unavailable task/source cell is approximated. Independent final groups remain separate from vision/SPI and scheduler fitting.

## 3. Confirmed training-state coverage failure

Evidence: train_real_suite.py:69-87, 168-175; recovered training_snapshots_1729.pickle.

The first seed has only 512 supervised decision snapshots:
- 62 of 64 episodes contributed samples.
- Median queue size: 2; maximum: 11.
- 76.37% of snapshots precede the burst at time 80.
- 23.63% are in the early burst.
- None are after time 120; latest sampled time is 98.37.
- 48.86% of candidate rows have novelty cold-start status.

The full FIFO validation trajectories reach queue sizes up to 93. The common nDCG probes cover queues only 2-8.

Why it matters: the model is fitted mostly to early small queues and then evaluated on late congestion and large backlogs. Multiple epochs recycle these same states. Temporal and queue-set models are especially disadvantaged because much of the intended history regime is absent.

Fix: traverse complete training trajectories, then reservoir/stratify by phase, queue size, load, priority conflicts, novelty availability, repeats, and source history. Include histories from random, simple semantic rules, and the learned policy within training groups. This is data collection, not a demand for another architecture or more random seeds.

Acceptance: saved coverage tables must demonstrate representation of late burst/post-burst and large queues; compare top-action regret by queue size and phase. DAgger provides a primary-source precedent for training on states induced by the learner, not just one behavior policy: https://proceedings.mlr.press/v15/ross11a.html .

## 4. Confirmed teacher reliability failure on the actual task

Evidence: teacher_pipeline.py:79-103, 145-186; saved requests.jsonl and responses.jsonl.

All 32 simple controls passed. Those controls assess visible-person presence, identical images, and blank-image abstention. They do not establish reliable scheduling comparisons with several missions and historical images.

Actual orientation audit:
- Candidate training: 3 of 28 strict preferences survive; 25 fail swapped-choice agreement.
- Candidate validation: 4 of 12 survive; 8 fail agreement.
- Prefix training: 2 of 28 survive; 26 disagree.
- Prefix validation: no strict supported preference survives; one orientation is invalid.
- Every original-orientation prefix chooses B, and most swapped versions still choose B.

Raw outputs include generic descriptions of a single image, copied mission text, and aspect ratings unrelated to a stable pairwise decision. This audit does not identify model capacity versus prompting, role binding, quantization or generation constraints as the sole cause.

Fix: test image/role/prompt binding deterministically with constructed requests. Separate factual extraction from preference choice and remove simultaneous aspect-rating obligations initially. Start with two current images under a clear mission, add history only after consistency survives. Add controls involving actual scheduling trade-offs, meaningful history, changed mission and swapped presentation. Diagnose raw outputs before changing model size. Keep abstentions and ties; do not loosen gates to manufacture labels.

Acceptance: report validity, supported preference accuracy, swap agreement, unsupported claims and accepted label yield on actual task cases. Passing schema validity is insufficient. RL-VLM-F learns reward from task-related visual preferences; it does not establish that any syntactically valid VLM judgments are usable: https://arxiv.org/html/2402.03681v2 .

## 5. Confirmed inactive VLM replay mechanism

Evidence: train_feedback_rl.py:60-72; semantic_learning_rl.py:51-64; responses and replay_guide.npz.

All 24 replay labels are neutral. The fitted guide has bias 1.0 and coefficient norm exactly zero. Every transition receives multiplier 1.

Therefore the actual sampling law is:

    0.5 * uniform + 0.5 * TD-prioritized

It has no differentiated VLM term. Furthermore, the feedback runner uses legacy make_episode rather than D's mixed-stream generator, so its scores are not directly comparable with D.

A separate mechanism limitation: the guide is fit using the first candidate's header for a two-step experience-level teacher rating. That is not a full state-action-next-state representation, even if future ratings become nonconstant.

Fix: relabel this run as neutral-guide mixed replay. Compare it to an identical no-VLM 50/50 replay control on the same trace factory, preprocessing and budgets. A guide should have measured nonconstant support and a validated mapping from the rated experience to its features. Preserve unknown/neutral cases rather than forcing diversity.

Acceptance: show nonconstant accepted guidance and measured changes to replay selection, plus an ablation with the exact same mixture but no teacher. Neutral guidance remains a valid negative result, not a reason to alter the teacher thresholds after seeing outcomes.

## 6. Confirmed supervised loss limitations and missing baseline

Evidence: semantic_learning_models.py:131-172; semantic_learning_env.py:180-187.

ListMLE ignores triples unless all three gains differ. In the actual 512 snapshots, 393 (76.76%) produce no ListMLE gradient. RankNet assigns the same unit pair weight to tiny and large gain gaps; it also pushes equal gains toward equal scores. A delay-only target can thus erase useful within-class semantic distinctions even if the inputs contain them.

The marginal-risk teacher is locally correct for unit service but not globally optimal. Imported exhaustive example:
- (budget, initial wait) = [(4,4), (2,1), (2,1), (16,0)].
- FIFO-tied marginal greedy [0,1,2,3] costs 4.1875.
- Exact order [1,2,0,3] costs 3.6875.

Fix: use a tie-aware listwise loss or supported strict pairwise preferences; add normalized gain-gap weighting or explicit decision regret where consistent with the target. For planning labels, use small exact examples or limited lookahead rather than describing immediate risk as global ground truth. If ties are genuinely unassessed, do not invent a semantic order.

Executed missing-rule replay:
- Original FIFO and SPI-first were reproduced exactly on all 15 validation traces.
- Received-SPI marginal-risk rule: cost 3.374323.
- SPI-first: 3.374083.
- Privileged reference-SPI-first diagnostic: 2.795158.
- Privileged reference marginal-risk diagnostic: 2.820034.

Thus adding this particular aging rule does not improve the present workload. Better reference-class knowledge shows headroom; it is privileged and not a deployable competitor. These replay rules ignore R/N, so unavailable vector extensions did not affect their decisions; no learned policy was reevaluated with altered features.

## 7. Confirmed loss of SPI uncertainty and limits of H0

Evidence: semantic_suite_data.py:60-62; raw archive SPI record.

The cached extractor supplies four uncalibrated class scores. The scheduler adapter discards them and constructs a hard one-hot class. It cannot distinguish a confident High prediction from a nearly tied High/Critical prediction using SPI alone.

The receiver also does not receive the UE mission text/category as a learned input: only a scalar CLIP relevance score carries the mission. A scalar match does not identify which detected entities are mission-relevant. Counts and aggregate areas describe presence and prominence, not all relationships or emergency situations.

These are information limitations, not proof that a larger network would recover missing facts. The exact same available H0 can correspond to different image-level teacher preferences.

Fix within H0: retain calibrated predicted class probabilities if the agreed parameter contract permits the already cached scores; otherwise learn/calibrate reference-class risk from the existing hard class and scene fields. Use only permitted registered mission context, never routing IDs as a shortcut. If the receiver is not allowed mission identity/context, explicitly bound the mission-generalization claim. Improve task-specific parameter producers or align supervision to observable H0 rather than silently introducing images/embeddings.

Acceptance: confidence calibration on training/validation roles, conditional class-error analysis, H0 matched/near-matched preference consistency, and SPI-only versus SPI+scene versus full-H0 ablations. Do not feed evaluation labels into the actor.

## 8. Confirmed PPO critic representation limitation; RL optimization is not diagnosed as a sign bug

Evidence: semantic_learning_rl.py:103-108, 145-165.

The actor receives queue size and history. The critic sees only the mean ready token. One and three identical tokens produce exactly the same value estimate in the imported numerical check, despite different aggregate outstanding costs. It also omits explicit history.

The RL reward sign, gamma=1 and telescope were independently checked: accumulated rewards match negative terminal total cost /100, including idle intervals. All action outputs correspond to ready candidates; no invalid-action masking failure was found. Poor scores do not justify claiming that the RL equations are reversed.

Other limitations: total-cost reward scales with episode size while evaluation averages per-report cost; the fixed 20,000-decision schedule is not convergence evidence; PPO's entropy coefficient and reward scale may impede a strong deterministic priority policy. These are hypotheses until checked.

Fix: provide the critic with pooled queue/history features and counts that its actor already observes. Normalize consistently with the chosen task objective and document episode weighting. Test a frozen supervised initialization before RL and preserve it as a no-RL comparator. Plot returns, action regret, entropy, value errors, TD statistics and class-delay curves rather than only a scalar best checkpoint.

Acceptance: queue-count distinguishability and known-value fixtures pass; validation gains must exceed the frozen starting policy on matched traces before attributing value to RL. Reproducibility concerns and sensitivity in RL are well documented: https://arxiv.org/abs/1709.06560 .

## 9. Confirmed simulator scope is narrower than the intended temporal study

Evidence: ue_scenarios.py:63-136; v9 D/config.json.

The executed mode is mixed IID images with generated timestamps, injected same-UE exact repeats, constant unit service and no failures. It includes late joins and fixed pause/resume windows, but does not establish natural incident progression, mobility/geographic causality, condition-change understanding, permanent departures, or session-reset effects in this mode.

At 100 UEs, fixed aggregate load produces only 0.99-2.58 offered images per configured UE across the validation traces. Increasing UE count does not automatically generate long temporal histories. The 50/60/100 UE tests are mostly many sparse senders rather than many persistent streams.

Fix: keep this as a valid mixed-queue control. Add explicitly supported sequence/incident manifests separately. Distinguish fixed-total-load scaling from fixed-per-UE streaming load and from fixed-images-per-UE history tests. Preserve policy-independent arrivals/outcomes. Do not claim physical realism or new camera-view correspondence from random images and GPS.

Acceptance: phase/history/repeat/condition provenance, source activity, load and queue-growth checks. Longer temporal histories require a deliberately designed workload; simply adding more UE IDs does not supply them.

## 10. Confirmed failure/history contract problem, not exercised by v9

Evidence: semantic_learning_env.py:194-202.

The environment marks a fact covered at dispatch and appends a completed header to processed history even if failure=True. Imported failure fixture reports one failure, processed_history_size=1, fact_marked_covered=True.

D measures waiting until dispatch and can legitimately freeze that cost after an attempt; this does not justify calling attempted evidence successfully delivered. In E, coverage/targets depend on selected facts rather than verified successful exposure.

Fix: distinguish attempted history from successful-evidence history explicitly. Keep failures and no-retry semantics. Commit useful/delivered fact state only at the relevant declared successful outcome. Preserve the D waiting metric if the study intentionally measures time to service.

Acceptance: success, failure, horizon-incomplete and duplicate-fact fixtures. The bug does not explain v9 results because executed streams have no failures, but it invalidates future failure/evidence claims unless corrected.

## 11. Evaluation and operational gaps

The aggregate D score is not a critical-delay guarantee. Across validation traces, reference counts are Critical 75, High 1,087, Medium 435 and Low 846. An average can favor many common-class improvements while harming the 75 critical observations.

First-seed selected-critical mean wait:
- SPI-first 2.928626; 75/75 selected; 89.33% within the declared synthetic critical budget.
- RankNet 8.048626; 75/75 selected; 60.00% within budget.
- Deep Sets 6.692157; 74/75 selected; 62.67% within budget.
- PPO 10.829778; 70/75 selected; 66.67% within budget.

These are synthetic time units and silver reference priorities. Delivered/selected-only means must remain accompanied by misses.

The budgets 2/4/8/16 are design assumptions, not measured emergency deadlines. nDCG uses the same marginal-risk reference and mostly tiny queues; it is not an independent semantic-quality certification. Final held-out urgency labels are absent. The actor does not pay its measured scheduling overhead in the simulation; measured selection timing and synthetic queue time remain different quantities. No actual codec was requested for this block.

Operationally, the sequential all-seed master exceeded the platform's 12-hour duration before post-training aggregation. The live log stream failed several hours before termination. Preserved checkpoints were useful, but the final ledger/report/rate study was unreachable in that execution. Many learner statuses are valid; the overall run is partial.

Fix: use one completed seed per method now, as requested. Perform offline evaluation/reporting from those frozen checkpoints. Preserve run-level status incrementally and make evaluation separable from training, so timeout does not prevent result assembly. For future runs, account for platform limits and checkpoint recovery; no new training is authorized by this audit. Add actual selection overhead only in a separately declared timing study.

## Recommended repair order and rejection tests

1. Correct reporting immediately: 15 optimization-complete methods, but zero informative VLM replay guidance; blocked teacher rankers/reward models; no general final evaluation.
2. Retain the current D benchmark and strongest SPI-first baseline. Do not replace them to force learned superiority.
3. Decide exactly which semantic distinctions are independently assessable under H0, and define their reference targets alongside priority-delay protection.
4. Repair training-state collection across complete trajectories and expose source/task coverage. This directly addresses a measured support gap.
5. Repair teacher query reliability and inspect accepted-label yield before training another teacher-dependent model.
6. Repair ListMLE's support and PPO critic state. Run bounded known-reference checks before any expensive retraining.
7. Evaluate the existing frozen models on matched workloads and per-class outcomes, then do the planned H0 ablations. A new model is justified only by a remaining measured limitation.

Candidate retained architecture: the existing small RankNet/MLP or tree model. A residual score around a transparent SPI/age risk estimate can reduce the burden of relearning basic priorities, but its benefits and critical-delay behavior must be tested, not assumed. No hard SPI gating, new image-bearing H1 header, larger VLM, extra RL variant or encoder redesign is required by this diagnosis.

If the objective remains only priority-weighted waiting, SPI-first may be the correct practical result. If the contribution is semantic context and additional information, those must be present in the references, workload and measurements before claiming success.

## Reproduction and remaining omissions

Run bounded_audit.py with the existing WSL semantic-scheduler interpreter. It reads trusted project-generated snapshots/checkpoints and writes only diagnostics. Run replay_missing_rules.py with OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 for deterministic CPU replay. Inputs and original source hashes are captured in diagnostics.json and the v9 manifest.

No new vision extraction, VLM inference, parameter fitting, scheduler training or Kaggle submission was performed. Full production/test regression was not rerun because production was unchanged. Portable export checks from training exist, but this audit did not independently revalidate every model. Final-independent outcomes, natural cross-view facts, real codec latency, real emergency deadlines, statistical population superiority and deployment readiness remain unestablished.
