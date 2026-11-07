# Locked specification: semantic-aware decoder scheduling

**Protocol ID: SCHED-20261030-v1. Date: 30 October 2026.**
This is the single executable study specification. It replaces conflicting earlier design alternatives. The source before this freeze is preserved under experiments/semantic_priority_design/runs/locked-spec-20261030/. Algorithm, target and metric choices below are fixed. Missing qualified labels/assets cause an explicit blocked result, not a new model-selection discussion or fabricated truth. This document specifies work; it does not claim that the new objectives/models have been implemented or trained.

## 1. Fixed objective, system and roles

Select one complete ready message for one nonpreemptive decoder. New messages arrive during service. Each received message gets one attempt; poor reconstruction remains poor. No radio grants, retransmissions, retry action, preemption, encoder redesign or live VLM inference.

**Primary study D:** minimize importance-sensitive waiting and lateness under declared simulator delay budgets. The proposed policy family is contextual Deep Sets scoring. Its RankNet-trained member is the main supervised candidate; same-information PPO/DDQN are sequential comparisons.
**Required semantic study E:** minimize first-service delay of distinct supported task evidence on annotation-backed overlap data.
**Teacher-supervision study P:** compare accepted VLM preferences/aspect supervision with reference-derived supervision.

These are named tasks, not interchangeable metrics. Every result records task_id, input_profile, architecture, target_source, reward_source, seed and model hash. D supplies the primary checkpoint/leaderboard criterion. E/P establish what the D cost alone cannot show about novelty and semantic selection.

Ownership: parameters/ owns extraction, SPI, vocabulary, transmitter KB and serialization; Scheduler/ owns inference and training modules; Evaluation/ owns lifecycle, simulator and references; Codec/ is paused; experiments/ contains scripts/notebooks and unique run artifacts. No second codebase.

~~~mermaid
flowchart TD
 A[Fixed UE missions and ordered source observations] --> B[Frozen extractor and causal sender KB]
 B --> C[Complete parameter record and opaque payload]
 C --> D[Receiver buffer]
 D --> E[Immutable ready snapshot and processed headers]
 E --> F[Shared neural scorer]
 F --> G[Select one ready message]
 G --> H[One nonpreemptive attempt]
 H --> I[Processed-header log; no retry]
 I --> E
 J[Separate references, budgets and annotated facts] --> K[Training targets and evaluator]
 G --> K
 Q[Offline Qwen, actual images and causal context] --> R[Versioned weak labels]
 R --> K
~~~

## 2. Input contracts: two fixed profiles

The user's explicit H0-only restriction remains the primary runtime boundary. The later proposed class arrays are a fixed additional comparison, not assumed already implemented or silently inserted into H0.

### 2.1 H0 primary

Ten numerical values: four supplied SPI one-hot values (Critical, High, Medium, Low), raw mission relevance, transmitter novelty, global retained detection count, distinct category count, largest-box fraction and union-box fraction. Add explicit availability masks and novelty cold-start. Unknown SPI is not Low. Normalize display Normal to Low only when SPI availability is true.

No images, latent contents, raw embeddings, annotations, reference budgets/tiers, fact IDs, filenames, source IDs, future arrivals/outcomes or remaining episode horizon enter learned features. Routing identity is used only for legal selection, exclusion of self in comparisons, duplicate handling and deterministic ties.

Runtime timing is receiver arrival/current wait and processed-header age. UE mission paragraphs affect the upstream relevance producer and offline teacher. Arbitrary language understanding is not inferred from a mission ID. Start with fixed receiver purpose: building-damage situational awareness.

### 2.2 SCENE_ARRAY16 comparison

Append two fixed-length arrays to H0: retained box counts per class and largest normalized box fraction per class. Freeze the existing producer vocabulary and order:

0 building; 1 road; 2 person; 3 car; 4 truck; 5 bus; 6 motorcycle; 7 bicycle; 8 debris; 9 chair; 10 couch; 11 dining table; 12 television; 13 refrigerator; 14 tree; 15 water.

Counts describe retained detector boxes, not physical water/road/debris instances. Largest area is rectangular image extent, not distance or segmentation. Count zero means no retained detection, not proven physical absence; absent-class area zero is padding with count/presence semantics. Detector failure leaves values unavailable. Record vocabulary hash, detector status and cap/saturation flags. Do not infer region counts as urgency.

Use one pinned YOLO-World checkpoint from the producer release, conf=.25, IoU=.5, imgsz=640, max_det=300, agnostic_nms=False. No threshold search or expanded vocabulary in this study. The previous 300-image grid established behavior/cost, not an accuracy optimum. A different vocabulary is outside this frozen release.

The existing parser must reject SCENE_ARRAY16 until its versioned schema adapter is implemented. The parameter workstream supplies the arrays; this plan does not alter its producer. Report blocked_input if absent. Keep comparison cohorts identical across profiles. Different-dimensional models are trained/exported separately.

Primary interchange is canonical JSON with actual UTF-8 byte counts. Do not call this the historical 35-byte wire header. A proposed uint16 count/uint8 area encoding would add 48 array bytes for K=16, but binary quantization is not implemented by this specification. Payload bytes are measured only when actual payloads exist; synthetic references have no claimed image-payload size.

### 2.3 Processing and preprocessing

Sender KB uses the upstream release's causal per-UE/mission history (current experiment FIFO128). Query, serialize, then commit each offered observation; receiver decisions never change sender KB. Recompute history-dependent parameters when the offered sequence changes. Reset across splits/sessions according to explicit metadata.

Preprocessing is fitted on training groups only: log1p counts, retain bounded fractions/cosine, median imputation for missing values and training mean/std standardization; constant std becomes one. Append masks and cold-start flags. Do not standardize masks into guessed values. Audit all-missing field groups. Models may ignore uninformative fields; forced scene weights are prohibited.

## 3. Reference labels and leakage boundary

Keep runtime evidence and evaluator/reference files separate, joinable by full message identity outside scoring.

1. Published valid priority labels: reference tier only where image/input correspondence is defensible.
2. Published object/condition annotations: supported task facts, not invented urgency.
3. Qwen labels: silver mission-relative acceptable-deferral tiers, pair preferences, aspects, prefixes or replay guidance.

Native MEDIC labels do not establish four-class emergency urgency. TREC-IS labels describe posts; image-to-post support must be established before image-only priority use. MEDIC/CrisisMMD and xBD/xView2 are overlapping naming/source relationships, not independent domains.

Unknown references remain unknown. They stay in mixed replay but cannot become Low, zero-cost or negative examples. D training requires all offered jobs' reference tiers known for that episode. Mixed-reference evaluation reports assessable-cohort coverage and costs explicitly; it is not a total-objective score. E requires declared fact correspondence. Teacher-only evaluations establish agreement under weak supervision, not verified emergency effectiveness.

Reference tier z is judged from actual image, receiver purpose, UE mission and any causal source context used by the rubric. It is not copied from the runtime predicted SPI. Cache keys include image, mission, receiver purpose, source-prefix/context and label-version hashes. Never reuse a context-dependent label for an incompatible stream.

Group exact/near duplicates, parent derivatives, annotated assets/conditions and available event/site identity before splitting. Use 60/20/20 group train/validation/test with seed1729; retain whole groups and report actual sizes. Use existing official source roles where required. Existing 300-image MEDIC and historical exposed panels are development-exposed; they do not become a fresh final test by reshuffling. Prefer out-of-fold SPI predictions; disclose frozen producer exposure separately.

Dataset roles are fixed: LADI and MEDIC/CrisisMMD for compatibility/priority/task evidence; xBD controlled building overlap for E; RescueNet source transfer; FloodNet structural-flooding scheduler-task transfer, withheld from fitting and selection. An unavailable source is marked unavailable, not replaced with guessed labels. Start source access with 20 records and retain the 2GB new-source-image download cap. Prior 240-group 144/48/48 counts are a feasibility target, not confirmed assets or a power calculation.

## 4. Primary D objective: declared budgets, exact training targets

All primary time is in d0=1 synthetic service slot until a digital codec supplies measured service. Arrival a_i is complete receiver arrival; dispatch b_i; wait W_i=b_i-a_i. Completion/capture-to-completion are supporting distinct measurements.

Map reference acceptable-deferral tiers to positive budgets:
Critical/Immediate=2d0; High/Prompt=4d0; Medium/Deferrable=8d0; Low/Routine=16d0.
Deadline is a_i+delta_i. This is an experimental requirement, not a real harm threshold inferred by Qwen. Delay sensitivity runs multiply all budgets by .5 or2; these settings are fixed, not tuned to make a winner.

Define:
ell(W,delta) = W/delta + max(0,W/delta-1)
W_i(t) = max(0,min(t,b_i)-a_i), with b_i=infinity until dispatched
L_D(t) = sum over a_i<=t of ell(W_i(t),delta_i)

Accrued cost freezes at dispatch; it is not erased. Late and unselected jobs continue accumulating cost. Dispatch at its deadline meets the budget. There is no minimum waiting time requirement. Qwen runtime is offline annotation cost, never a subtraction from delta.

Primary reported cost is J_D=L_D(T)/N_assessed, dimensionless, on matched complete-reference episodes. Record L_D, N and eligible arrival count. Also report per-tier waits and misses; a single average cannot establish correct urgency handling.

The supervised immediate-delay target is:
y_i = ell(W_i+d0,delta_i)-ell(W_i,delta_i)
ymax=2d0/delta_min for the active budget configuration
v_i=y_i/ymax in [0,1].

y_i is the exact cost of one-slot deferral under the convention, not a globally optimal action value. Use Huber regression or ordering losses on this target. Scalar importance cannot be read from H0 by a hand formula; the network predicts the reference-derived target using its inputs.

For RL, event reward is:
r_k = -(L_D(t_next)-L_D(t_k))/100.
Reward scale100 is fixed and independent of offered future jobs. Gamma=1. Include mid-service arrivals, idle intervals and the terminal interval. Reward sum equals -L_D(T)/100. The actor never receives true delta or z.

Do not claim universal Critical mean < High < Medium < Low across arbitrary streaming arrivals. Blocking and arrival phase can reverse class means. With equal service and fixed known linear weights, highest-weight-ready-first is optimal among work-conserving full-drain schedules; this is a useful diagnostic, not proof that a learned imperfect-input model is unnecessary or that the nonlinear budget objective has the same theorem.

## 5. Required E and P objectives

E uses independently supported fact-condition sets E_i. A changed condition is a different reference unit. Start with exact copies and xBD annotated controlled crops (half-image windows, quarter-image stride plus full image; credit only >=95% enclosed source polygon). Keep parent/groups within splits. Similar appearance does not establish natural cross-view identity.

For each offered fact f, r_f is first containing complete arrival, t_f first containing dispatch:
L_E(T)=sum_f w_f*(min(t_f,T)-r_f)/d0.
Use w_f=1 for supported useful building facts; do not turn damage categories into undocumented rescue severity. Count each fact once. Actual reconstructed meaning is not verified by this source-reference calculation.

Current E target y_i=sum of currently uncovered supported w_f in candidate i; its next-slot cost reduction is proportional to y_i. Unknown correspondence is not non-overlap. RL uses -Delta L_E/100. The actor sees headers/context, never E_i or f. This is a separate target/reward version from D.

P contains accepted Qwen A/B/tie preferences under the declared purpose and causal context. No invented continuous utility or complete ranking is inferred from arbitrary disconnected pairs. P loss and teacher agreement are weak-supervision results. D checkpoint selection and independent E outcomes are still reported wherever valid.

Core task schedule: all feasible learned arms below run D; Deep Sets RankNet, PPO and DDQN also run E when assets are valid; Deep Sets RankNet-P and VisionPrefer compare weak semantic supervision. This is the fixed comparison set, not a giant task-by-model sweep.

## 6. One shared observation architecture

For p value fields, immutable semantic vector x has standardized values, p masks and novelty-cold-start. Thus semantic width is 2p+1 (H0 p=10; ARRAY16 p=42). Add log1p(wait/d0) at decision time. Historical tokens use log1p(age-since-processing/d0) and a history-role bit; this age is not a new source parameter.

Keep the most recent32 processed headers, oldest removed at capacity. Update after an attempt completes; do not assert reconstructed facts or inspect quality. Failed exceptions and poor visual outcomes do not create retries. Log status outside semantic scoring. Empty-history and availability indicators are explicit.

Summary feature input for scalar models contains candidate x/wait, masked queue/history means, availability fractions, log1p queue/history size, empty-history flag, minimum comparable history distance and its validity flag. Distances use training-scaled co-available fields; no comparison gives zero plus invalid, not observed similarity. No self-distance is used. Routing IDs only associate records.

MLP scorer: shared64/32 ReLU layers, one scalar, no dropout/batchnorm. Linear scorer: one affine scalar.

Deep Sets: shared phi with32/32 ReLU layers encodes header/time/role tokens into32 dimensions. Mean-pool ready and historical embeddings separately; concatenate candidate raw features/embedding, ready mean, history mean, set sizes and empty flags into64/32 ReLU head. Mean plus set size preserves cardinality information. No list position encodings. Candidate input permutation must permute associated outputs, not alter identity-associated scores.

Set Transformer comparator: token dimension32, two self-attention blocks, four heads, feedforward64, layernorm, dropout0; ready/history role bits and padding masks; candidate token outputs feed the same64/32 score head. No routing/array-position embedding and no silent candidate truncation. Full attention cost is quadratic; measure actual observed queue lengths.

All learned policies select argmax among ready messages at deployment. Training PPO samples a masked categorical distribution. Ties: arrival, sequence, full source/session/token. Empty queue waits only for a real future arrival. Busy decoder cannot dispatch. Invalid artifacts fail before service; unexpected nonfinite scores use FIFO and increment a fallback counter.

## 7. Locked learned-method registry and losses

All existing IDs remain recognizable; objective/profile versions distinguish new experiments from historical fixture models. Old binary fact-gain classifiers or completion-coverage checkpoints are never relabeled as newly trained methods.

For decision s with q candidates, let y_i be D risk (or valid E gain in the named E arm). Best-tie target p_i=1/|B| for i in B=argmax y, zero otherwise.
Choice CE: L_choice=-sum_i p_i log softmax(score)_i.
Skip singleton, all-equal and unsupported choice snapshots; retain them for lifecycle/evaluation.
For pair preference t_ij: L_pair= -t log sigmoid(s_i-s_j) -(1-t)log sigmoid(s_j-s_i).
Strict comparisons use1/0, justified ties.5; uncertainty skipped. Max32 deterministic sampled pairs/decision, with nonzero tie budget. Average within decision then episode. Do not let large queues dominate by pair count.

| ID | Model | Frozen supervision/loss |
|---|---|---|
| linear_choice | Shared linear summary scorer | L_choice |
| mlp_choice | Shared MLP summary scorer | L_choice |
| delay_huber | Shared MLP summary regressor | Huber(predicted v_i,v_i), delta=1 |
| ranknet | Shared MLP summary scorer | L_pair from exact y comparisons |
| lambdamart | LightGBM grouped LambdaMART | Pair logistic lambdas weighted by abs Delta NDCG; explicit linear label gains |
| priority_classifier | MLP four-class reference-deferral head | Class-balanced multiclass CE on z; inference scores expected y_i over predicted tier probabilities and current wait |
| deepsets_ranknet | Deep Sets scorer | L_pair; main D member, separate E/P label versions |
| set_transformer_ranknet | Attention set scorer | Same L_pair and sampled pairs as Deep Sets |
| listmle | Deep Sets scorer | ListMLE on deterministically sampled strict three-candidate reference orders |
| masked_ppo | Deep Sets actor and pooled critic | Clipped PPO + value MSE - entropy |
| masked_double_dqn | Deep Sets candidate-conditioned Q | Uniform-replay Double-DQN Huber TD loss |
| td_per | Same DDQN | Same TD loss; mixed TD-PER sampling/importance correction |
| ranknet_warmstart_ppo | D-trained Deep Sets actor initialization | Same PPO loss/reward; optimizer/critic fresh |
| rl_vlm_f | Same DDQN architecture | Double-DQN TD loss using preference-learned reward, not D reward |
| vlm_replay | Same DDQN architecture | D reward fixed, guided sampling only |
| visionprefer | Deep Sets scorer | L_pair from four-aspect average teacher preferences |

LambdaMART quantization: g_i=floor(255*v_i+.5), integer0..255, label_gain[g]=g/255. This is an explicit approximation of continuous D risk. E uses integer uncovered unit-fact gains and label_gain[g]=g. No grades from arbitrary P pairs. Preserve quantization ties; do not silently replace the D target with SPI class.

LambdaMART settings:100 trees, learning_rate=.05, max_depth3, num_leaves7, min_child_samples10. Pass decision-group boundaries; do not shuffle rows across groups.

Classifier balancing weights w_z are inverse-frequency from training only, normalized to mean1. A weighted-CE softmax q_z is a tilted distribution, not the original-tier posterior. Before expected-risk inference use p_z=(q_z/w_z)/sum_c(q_c/w_c). This correction is not a guarantee of calibration; report held-out Brier/log loss. No class present in validation/test is used to fit weights. A training-absent required class blocks a complete four-class claim; never synthesize its support.

ListMLE loss for strict order pi on three supported distinct y values:
L_list=-sum_k [s_pi(k)-logsumexp_{j>=k} s_pi(j)].
Sample at most8 triples/decision. Skip ties/unknowns; do not arbitrarily break them. The VLM pair cache is not asserted to provide complete lists.

Supervised defaults: Adam lr3e-4, weight_decay1e-4, max50 epochs, batch up to64 decision groups, patience8 validation checks. Torch initialization/trace RNGs separated. Targets v are normalized by the fixed analytic ymax; this does not use test statistics. All neural scores other than classifier probabilities are relative scores, not calibrated urgency probabilities.

## 8. RL objectives and update details

PPO: rho_k=pi_theta(a_k|s_k)/pi_old(a_k|s_k).
L_actor=-mean min(rho*A,clip(rho,.8,1.2)*A).
L_PPO=L_actor + .5*mean(V-G)^2 - .01*entropy(pi).
Use GAE gamma1/lambda.95, detached advantages/returns, normalize advantages per rollout, gradient norm.5, Adam3e-4. Four CPU envs,256 aggregate transitions/rollout, four update epochs, batch64. Final partial rollout uses actual masks and observations.

Distinguish genuine episode termination (no bootstrap) from rollout/time-limit truncation while an episode continues (bootstrap value). No bootstrap beyond the declared finite-horizon objective T. The continuing-stream deployment claim is not established by a finite-horizon pilot.

Replay transitions run from one dispatch decision to the next nonempty ready decision, accumulating reward through service and any automatic idle gap; if none occurs before the horizon, end at terminal T. An empty queue with a future arrival is not a learned WAIT or a terminal bootstrap state. Use the same boundaries for PPO and learned-reward replay. If all work finishes earlier, jump to T with zero additional waiting cost.

Double DQN:
a_star=argmax_legal Q_online(s_next,a);
Y=r + (1-terminal)*Q_target(s_next,a_star);
L_TD=mean Huber(Q_online(s,a)-stopgrad(Y)), delta1.
Both argmax and evaluation respect next-state membership. Terminal empty next set contributes zero. Store current/next immutable candidate sets and snapshot-associated selected identity/index; an index does not identify a message in another state.

DDQN defaults: replay50,000; warmup500 decisions; batch64; update every4 decisions; target copy every500 decisions; Adam3e-4; gradient norm10. Epsilon1 to .05 over first10,000 decisions, then .05. Replay sampling and model initialization have separate RNGs.

For TD-PER, p_j=(abs(TD_j)+1e-6)^.6. Both td_per and vlm_replay use:
P(j)=.5/N + .5*p_j*g_j/sum_l(p_l*g_l).
TD-only g_j=1. VLM g_j is .5/1/1.5 for low/neutral/high training-value; missing/uncertain=1.
Importance weight=(N*P(j))^(-beta), beta.4 to1 over20k decisions; normalize by the maximum in the sampled batch. Use actual mixture P, not only the guided component. Recompute sampled TD priorities after learning. Guidance concerns training value, not a runtime image priority feature. PPO does not use stale replay.

Twenty thousand action decisions per RL arm per study seed is a fixed cap. Validate every2,000; no unreported extra training to rescue a method. Warmstarted PPO starts with the D ranker's phi/score weights only, a fresh critic/optimizer and the same20k RL decision cap; report supervised pretraining cost separately.

## 9. VLM teacher, reward learning and adapted methods

Backend: Qwen/Qwen3.5-9B, the existing registered NF4 setup, direct Transformers subprocess, thinking disabled. Resolve/store exact checkpoint revision, processor, prompt, quantization and environment hashes before labels. No model/provider switching or prompt hunting on validation/final results. A technical incompatibility blocks the run until a diagnosed correction is recorded.

Teacher sees actual RGB images, declared receiver purpose, UE missions, trusted/simulated metadata clearly distinguished, and bounded causal context. Hide predicted SPI/H0 values, scheduler outputs and reference facts. It must return structured JSON with choice/tie/uncertain, concise visible evidence and unknowns. Never request invented numeric real-world deadlines or use self-rated confidence as validated correctness.

Deferral tier labels provide D budgets; source-prefix-dependent tiers require matching context cache keys. Pair labels supply P. Teacher costs include every call/reversal/failed parse; no free uncapped retries.

**Fixed216-call maximum allocation:**

| Purpose | Calls |
|---|---:|
| Independently specified controls |32|
| Tier labeling, up to2 current context-records/call |24|
| Candidate pairs,32 unique pairs in both orientations |64|
| Equal-time prefixes,32 unique pairs in both orientations |64|
| Replay training-value clips |24|
| Diagnosed technical recovery reserve, no prompt search |8|

This allows at most48 newly tier-labeled context-records absent reusable compatible labels. It is a feasibility label budget, not evidence of a sufficiently powered journal cohort. Reuse independently compatible cached labels only with verified provenance. Never pretend48 contexts or repeated queues equal240 independent images. Candidate/prefix pairs use24 training and8 held-out validation groups before collection; retain rejected/uncertain counts. Candidate responses also include aspect ratings, avoiding a separate unlimited aspect-label campaign.

Largest planned teacher request is preflighted before collection. At most2 current images plus bounded necessary causal context per tier call; resize/image-token settings are pinned from the working registered backend and recorded. If planned context exceeds the measured envelope, skip/mark insufficient rather than silently omit relevant history or exceed resources.

Controls require independently defensible expected answers. Gates: >=95% valid schema, >=.80 balanced strict-preference accuracy, >=.75 tie and unanswerable handling; report denominators and uncertainty. Accepted oriented pairs must agree after swapping; disagreement is uncertain, not a forced majority. Unsupported factual assertions invalidate the affected label where independently detectable. These gates do not certify real triage accuracy.

**RL-VLM-F-inspired reward.** Prefix comparisons start from the same state/offered trace and end at the same elapsed time; show dispatch times and enough candidate context to expose neglected important messages. Do not compare final collages or reward more pictures automatically. Only causal observations appear.

Reward model uses permitted header/context information and the chosen candidate, never reference labels. Its state is explicitly post-dispatch: remove the selected header from the waiting set, retain it as in-flight context, update processed-header history only at completion, and admit subsequent messages at their actual arrival times. A bounded rate c_psi(s_post,a)=-sigmoid(h_psi(s_post,a)) models cost while waiting work remains; force rate zero when no job is waiting, including idle gaps. No direct VLM scalar is treated as calibrated reward.

Integrate this rate with one fixed left-sample rule on a global grid of d0/4 anchored at episode time0, additionally split at actual arrival/dispatch/completion/horizon events. Freeze the sampled rate within each resulting microinterval and sum rate*duration. Artificially partitioning a stored transition does not create a new sample; reuse the same physical-event/grid path. This is a declared numerical integration approximation, not an exact analytic learned-cost integral. Preference probability=sigmoid(R_A-R_B) for equal-time prefix integrals; optimize pair BCE, weight_decay1e-4 and Adam3e-4. The action/remaining-queue distinction is needed even for one-decision prefix feedback.

Store immutable header-cache references and event-stream slices sufficient to reconstruct all reward microstates and ages, rather than relying only on pre/post decision snapshots or a mutable queue. Relabeling uses exactly the same grid, event boundaries and reward version. Do not require full copied queues at every tick; deterministic event replay from pinned immutable data is sufficient. Report integration/feature-building cost separately from the actor.

Fit/reward-update blocks are frozen while DDQN learns. After a reward-model change, relabel every replay transition using stored snapshots/durations before resuming. At least16 strict train and8 held-out strict prefix pairs, >=.75 held-out accuracy and improvement over constant-majority are engineering requirements. Insufficient feedback marks rl_vlm_f blocked_labels, not successful RLAIF.

**VisionPrefer-inspired arm.** Adapt only its multi-aspect preference-construction mechanism. Rate mission usefulness, visible evidence clarity, additional information under the shown context, and condition specificity on integers0..4; allow uncertain. Average four supported aspects, derive strict preferences/ties, train Deep Sets L_pair. Equal averaging is an explicit weak-supervision convention. No aesthetics or text-to-image generator tuning is introduced. This is not reproduction of the original VisionPrefer benchmark.

Round0: controls, compatible tier labels and initial candidate/prefix labels. Train initial models. Round1: remaining training-only prefix/replay requests from their experience, within the same216 cap. The actor never waits for Qwen. Missing replay guidance stays neutral. Rejected teacher feedback does not stop reference-supported methods.

## 10. Fixed baselines and fair comparisons

Untrained: FIFO; round robin by source; seeded random; predicted-SPI then FIFO; relevance-only; novelty-only; lexicographic SPI then novelty; predicted-SPI-budget EDF. If a required field for a scalar rule is missing in any ready candidate, that rule falls back to FIFO for that decision and logs the fallback. RR/random do not consult semantic availability.

Learned classifier plug-in greedy is required: compute expected one-slot y under its predicted reference-tier distribution, then select the largest. It separates learning urgency from sophisticated scheduling. EDF uses predicted budget and absolute arrival+budget, not true reference delta.

Privileged diagnostics: true-budget EDF (not claimed optimal); exact tiny legal-sequence optimum; reference marginal-fact greedy. None is a fair deployed competitor. A greedy schedule is not a lower bound. No WAIT or future observations are added to deployed actors.

D core comparison uses all16 feasible learned arms and the baseline set. P/teacher arms differ in label/reward source by design. Architecture comparisons match reference targets/pair budgets. Replay comparisons match actor/reward/training steps and the uniform-mixture setting. Warmstart comparisons count pretraining and RL decisions. Never call multiple losses on the same architecture independent new algorithmic discoveries.

## 11. Data streams, UE profiles and randomized metadata

Use the existing15 profile definitions, with fixed device-description/mission hashes:
ordinary phone; incident phone; structural drone; person-search drone; flood/evacuation drone; wildfire drone; road/bridge drone; ordinary car; emergency vehicle; traffic CCTV; public-space CCTV; river/underpass camera; industrial robot; agricultural drone; maintenance camera.
These are project templates, not standardized15 6G device classes. Device identity is never semantic priority.

Episodes are ordered incident observations with profile-compatible background traffic, not random unrelated image bags. Keep real sequence replay, controlled correspondence/crops, and synthetic chronology labeled separately. Repeated images retain parent/group identity; changed conditions require annotation support.

Fixed workload axes:
- populations4/12/50/100 and within-episode join/pause/resume/leave;
- training32–512 messages total, variable per-source lengths;
- required scale cases50 images per UE (2,500 and5,000 messages for50/100 UEs);
- service d0=1 in primary; separate lognormal(mean1, log-std.5) stress;
- offered-load ratios .5/.9/1.2 versus nominal one-job-per-slot capacity;
- initial backlog min(8,N), plus a separate all-at-once contention control;
- chronological uniform arrivals or two incident bursts containing70% of offered messages in20% of the arrival-window duration.

Compute arrival window from offered count/load and report realized rates, backlog and active counts. Hold total load fixed when changing UE count; population and congestion are different experiments. At fixed arrival window, separately report load growth. Episode endpoint is fixed before policy replay, arrival-window end plus32d0; actor excludes remaining horizon. All methods use identical message-qualified arrivals/durations/outcomes.

Timestamp chain: capture -> extraction-ready -> complete receiver arrival -> dispatch -> finish. Synthetic clock, coordinate system and accuracy are explicit. Stationary cameras do not wander. Generated phone/robot/drone/car trajectories follow bounded synthetic paths (illustrative speed caps2/2/15/20 m/s), with metadata marked simulated. No geographic truth is inferred from assigned coordinates.

A UE leaving does not delete received messages; session restart and sender-KB reset are explicit. Reset clears queue/in-flight/policy rotations/receiver history. Account for every offered job as not-arrived, queued, in-flight, processed or explicitly excluded.

Core has all normally completed attempts. A separate fixed5% processing-exception stress and .01/.10 sensitivities use hidden identity-qualified draws, not selection order. No retries. At unchanged durations and no outcome features, outcome flags alone must not alter selection-cost trajectories. Poor reconstruction is not automatically an exception or observed semantic failure.

Perform demand/capacity checks before interpreting missed budgets: only three jobs can start by time2 under unit service at times0,1,2. Impossible simultaneous deadlines remain documented overload, not proof a learner is defective.

## 12. Evaluation metrics, exact definitions

**D primary:** macro-average J_D across matched episodes/load strata; lower is better. Pair with independently reference-tier-conditioned timing/misses. Do not claim urgent performance from an unweighted mean across all images.

For each reference tier:
- offered, assessed, selected and never-selected counts;
- selected-message mean/median wait and p95 only with >=20 selected samples;
- restricted mean min(W_i,16d0) over messages with full16d0 follow-up, unselected contributing16d0;
- within-budget fraction and miss rate on messages whose arrival+delta<=T; separately report boundary/censored messages;
- decoder completion/turnaround supporting metrics, explicitly separate from queue wait.

Unselected jobs remain in L_D(T) with observed waiting through T. A job dispatched by deadline meets the selection budget even if its service finishes later. Do not call insufficient-follow-up messages deadline misses. No assertion of an unconditional class-mean ordering.

**Ranking:** common ready/history probe snapshots for all models, generated with fixed random/SPI behavior independently of validation. D risk nDCG@1/3/5 uses linear gain y_i/log2(rank+1), ideal sorted y; singleton/all-zero ideal cases are reported separately/undefined as appropriate. Class-grade nDCG is a separate coarse table. Pair agreement is evaluated only on supported labeled pairs; ties/uncertainties have distinct counts. Kendall tau-b only when a graded reference order exists. Pair labels do not automatically yield nDCG.

**E:** mean capped first-service fact delay, distinct supported facts selected by time/budget, unserved facts and repeated-reference selections. Source facts selected are not proof of reconstructed evidence quality. No natural cross-view claims from exact copies/crops.

**Teacher:** control performance, schema validity, swap disagreement, accepted-label yield, ties, abstentions, unsupported claims, calls, images/tokens, seconds/call and memory. Report independent task outcomes separately from teacher agreement. B-Pref-style controlled error rates0/.1/.3 include random flips and systematic subgroup errors, introduced separately on training labels with clean evaluation references.

**Systems:** context-build+select milliseconds and forward-only time separately; warmed hardware/backend/precision/thread count; p50/p95 with samples; peak memory; model parameters and supported-operator FLOPs. Count2 operations/MAC. Include pooling/attention scope and missing operators; teacher/quantized FLOPs unavailable is not zero. Profile queue sizes4/16/64/256 and actual observed large-backlog sizes with no truncation. CPU profiling budget5 minutes/model; partial timing scope is recorded.

**Transport:** actual parameter JSON bytes, candidate profile/vocabulary hashes and serialization latency. No synthetic payload-size or radio-spectrum improvement claim. Selector overhead is measured; synthetic slot results are not milliseconds. Report a separate overhead-charged sensitivity once a measured selector-to-service scale exists.

## 13. Model selection, splits, seeds and statistics

Training seeds1729/2718/31415. Separate RNG streams for model, training traces, validation, test and service outcomes. The correctness run is fixed2–3 episodes,3 epochs,64 RL decisions/arm,1 seed; it is never merged with study-performance results.

Validation set:12 matched traces covering populations4/12/50/100 and three loads; message counts48/96/256/512 by population. Same validation trace IDs across arms. Validate supervised models each epoch and RL each2,000 decisions.

Checkpoint rule for each arm: lowest validation macro J_D. Ties within1e-6: higher assessable Critical within-budget fraction if defined, then fewer parameters, then registry ID. No test results enter selection. P-only and E-specific checkpoints select their named validation objective (P accepted-pair accuracy; E cost); D results are still reported where assessable. Learned-reward RL is selected by independent D validation, never by its own learned reward. Missing independent selection references block that claim.

Promotion rule across selected D policies: a learned policy is eligible as the recommended operational candidate only if, on the same validation traces, its Critical within-budget fraction is no lower and its Critical restricted mean wait no higher than predicted-SPI-FIFO. Among eligible policies choose the lowest macro J_D with the tie rule above. If none qualify, retain the baseline and report the negative result; do not change the threshold or hide failed arms. If Critical references/follow-up are absent, a Critical-service recommendation is undefined, not passed. This is a fixed sample-based engineering guard, not a statistical guarantee or a promise of feasible deadlines during overload. Final test results report all preregistered methods even if they fail promotion.

Final evaluation uses locked test groups once after selection, on matched workloads. Bootstrap1,000 paired parent/duplicate groups and regenerate matched traces; queue seeds are not independent incidents. Report effect sizes/intervals, per-source failures, counts and all chosen seeds. Source-held-out and event-held-out are separate. FloodNet transfer is scheduler-unseen task only; vision-model exposure is disclosed.

No hyperparameter search in this fixed study. Budget/common-scale sensitivities are prespecified, not used to pick a favorite budget. If no model dominates timing and semantic outcomes, report the trade-off rather than invent a post-hoc overall winner. One-seed or tiny-label results remain feasibility evidence.

## 14. Acceptance checks and artifacts

Required behavior checks:
1. Full-identity serialization/uniqueness, parser finite/range/version checks.
2. Busy/empty/future arrivals, arrivals during service, foreign/duplicate completion rejection.
3. No retry; reset correctness; queued messages survive UE departure.
4. Zero versus missing/cold start; detector failure/cap and class-order mismatch.
5. Candidate permutation equivariance and routing/evaluator/future-information isolation.
6. Mid-service/terminal/idle cost integral versus direct accrued costs, including unknown references rejected and unselected jobs.
7. One-slot target arithmetic, pair-gradient sign, choice tie targets, classifier expected-risk inference and LambdaMART grade quantization.
8. Legal DDQN next-action selection/evaluation, genuine terminal versus PPO rollout truncation.
9. PER mixture sums/support/importance correction; constant-guidance parity with TD-PER.
10. Variable-duration learned-rate reward and equal-time preference invariance to decision segmentation.
11. Training-group-only preprocessing and held-out-label poisoning checks.
12. Actual optimizer/RNG/replay resume; weights-only is labeled weights-only.
13. Torch/LightGBM versus portable NumPy predictions and selected identities, including tree threshold boundaries.
14. Outcome-only stress invariance at fixed service/no outcome actor input.
15. Artifact checksums, model/profile/target compatibility and rejection of legacy weights under new semantics.

Do not run the same smoke repeatedly without new changes. Existing focused suites remain in Test/tests. New research checks/benchmarks and their results stay in experiments/. No data/parameter-team sources are modified by scheduler work.

Each run contains manifest.json, config.json, environment/hardware/source hashes, runtime/reference manifests, raw teacher responses, targets, per-decision/per-trace/per-group metrics, learning curves, model/checkpoint/export files, logs/stdout/stderr/errors, figures, validation.json and REPORT.md. Preserve failures/partial runs and executed notebooks with cell outputs. A completed process is not proof of semantic validity.

## 15. Fixed execution order, resources and implementation status

Execution order is dependency-driven, not a new architecture-selection stage:
1. Freeze current source/input artifacts, preserve dirty working tree and old results.
2. Implement D/E evaluators and typed targets beside the legacy completion objective.
3. Implement processed-header adapter and exact observation profiles.
4. Run acceptance/numerical checks and the fixed2–3-episode correctness run.
5. Connect qualified reference/teacher records; blocked arms retain explicit statuses.
6. Train registered models under fixed defaults; validate/export/resume.
7. Run locked comparisons, scaling, ablations and report.

Main Sol owns code/integration and the canonical record. Medium assesses completed consequential slices without edits. Luna High runs pinned tests, operational monitoring and authorized Kaggle actions. Agents do not reset/pull/move a live checkout. No permanent supervisor or redundant full surveys.

Reuse the project .venv locally; training frameworks stay outside deployed Scheduler inference. Portable artifacts include preprocessing, schema/profile/vocabulary hashes, architecture, objective/label/reward source and weights. Deep Sets/attention exporters require parity before deployment; existing dense exporters are not assumed sufficient.

CPU first. Six aggregate GPU-hours remains the prior execution ceiling, not a new quota claim:3h extraction/teacher,2h justified accelerator work,1h diagnosed contingency; actual codec work remains paused. Count allocated idle GPUs and loading. Recheck current access/quota before any submission. Reuse established Kaggle notebook identities and parameters/KAGGLE_RUNS.md; no healthy run is interrupted to add logs. Preflight hardware and ten-image/call cost before expensive work. A cap stops execution with a checkpoint/partial status; it does not authorize silent model or label substitution.

The current legacy pilot pre-spi-check-20261030-094517 records15 focused passes, eight fixture-trained learners, three blocked teacher arms and11 waiting-spec arithmetic checks. It does not validate these new D/E losses, Deep Sets/attention, scene arrays or model superiority. The finite J_time values refer to the legacy completion-coverage objective.

Final specification status: model registry, targets, losses, defaults, teacher allocation and metrics are fixed. Execution blockers are qualified labels, source assets, export implementation and measured resource availability. Record these as blocked_input/blocked_labels/partial_budget/failed_validation; do not relabel them as completed work.

## 16. Literature basis and limits

The implementation uses established mechanisms, adapted to this queue:
- [RankNet, 2005](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf): shared scorer and pairwise logistic loss.
- [Deep Sets, 2017](https://arxiv.org/abs/1703.06114): shared set encoding and invariant pooling.
- [Set Transformer, 2019](https://proceedings.mlr.press/v97/lee19d.html): attention-based set interaction.
- [LambdaMART overview, 2010](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf): metric-weighted ranking gradients and boosted trees.
- [PPO, 2017](https://arxiv.org/abs/1707.06347): clipped policy update.
- [Double DQN, 2015/2016](https://arxiv.org/abs/1509.06461): separate action selection/target evaluation.
- [Prioritized Experience Replay, 2015/2016](https://arxiv.org/abs/1511.05952): TD priorities and importance correction.
- [DeepRM, 2016](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2020_2021/papers/mao_HOTNETS_2016.pdf) and [Decima, 2019](https://arxiv.org/abs/1810.01963): objective-driven time rewards and streaming-scheduling training; their resource models differ.
- [Semantic-aware Transmission Scheduling, 2023](https://arxiv.org/abs/2305.13706): application cost specified before DRL; its estimation costs/monotonicity theorem do not define our image importance.
- [Semantic Scheduling for LLM Inference, 2025](https://arxiv.org/abs/2506.12204): urgency-conditioned timing, with different batching/preemption assumptions.
- [RankGPT, 2023](https://arxiv.org/abs/2304.09542): teacher-to-ranker distillation; passage ranking is not image triage.
- [RL-VLM-F, 2024](https://arxiv.org/abs/2402.03681) and [PEBBLE, 2021](https://proceedings.mlr.press/v139/lee21i.html): preference reward learning and replay relabeling; our duration-aware receiver adaptation is explicitly different.
- [VisionPrefer paper, NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/9421261e06f1a63a352b068f1ac90609-Paper-Conference.pdf): multi-aspect AI feedback and pairwise reward fitting; our aspects/tasks differ.
- [B-Pref, 2021](https://arxiv.org/abs/2111.03026): independent returns and controlled teacher imperfections.
- [Min-sum Set Cover, 2003](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2003-21.pdf): first-cover delay; no direct online approximation guarantee inherited.
- [Novelty and Diversity in IR Evaluation, 2008](https://cormack.uwaterloo.ca/novelty.pdf): independent nugget-level redundancy evaluation.
- [YOLO-World documentation](https://docs.ultralytics.com/models/yolo-world/): custom vocabulary support, not verified detection of every hazard.

The budget mapping, piecewise D penalty, profile templates, aspect rubric and numerical defaults are declared design conventions. Literature does not prove these are universally best, clinically safe, novel, converged or journal-ready. The scientific claim must remain tied to the executed sources, labels, operating regime and checks.

