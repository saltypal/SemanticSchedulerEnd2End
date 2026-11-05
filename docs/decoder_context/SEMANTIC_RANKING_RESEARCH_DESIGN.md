# Locked specification: semantic-aware decoder scheduling

**Protocol ID: SCHED-20261031-v2.2. Date: 31 October 2026.**

**Final plan.** All seventeen methods, P/D/E studies, original losses and reference boundaries are retained. Section20 finalizes execution, action budgets, delay interpretation and acceptance. Section9 changes the initial teacher candidate to Qwen2.5-VL-7B-Instruct with NF4 loading. Historical results retain their original backend/protocol. This finalization does not claim new implementation or training. The preceding document is preserved with checksums in experiments/semantic_priority_design/runs/.
This revision modifies the existing frozen plan; it does not discard its models, objectives, reference boundaries or negative results. The user has now explicitly adopted the final scene-array v2 contract and asked for diverse streams plus a VLM-supervised temporal scheduler as the first experiment. Those decisions supersede the former H0-only primary input, optional ARRAY16 comparison and D-first execution order. D/E losses, the sixteen previous learned methods, baselines, RL/RLAIF arms and resource ceilings remain. The v1 document and the supplied final cache specification are preserved under experiments/semantic_priority_design/runs/scene93-temporal-v2-20261030/.

Revision v2.1 adds the user-requested compact transmission/field-selection study in Section18. The full93-class extraction cache is retained; the selected transmitted profile can be smaller. This is a design specification, not a claim that new policies, broad datasets, teacher reliability or real inference have been validated. Qualified labels/assets and compute still determine whether an execution is complete, partial or blocked. No production code, parameter producer, SPI model, codec or remote job is changed by this document revision.

## 1. Fixed system, revised main experiment and retained studies

Select one complete ready message for one nonpreemptive decoder. New messages arrive during service. Each message receives one attempt; poor reconstruction remains poor. No radio grants, retransmissions, retry action, preemption, encoder redesign or live VLM inference.

**First experiment P: VLM-only supervision of a parameter-only temporal ranker.** Qwen sees actual candidate images, each UE mission, a fixed receiver objective and bounded causal image histories offline. The student sees only received SPI, relevance, novelty, scene-array v2, timing and header history. Its main loss is accepted teacher pairwise preference loss; no delay target, published priority or manually weighted semantic formula is secretly added to that arm. VLM-only describes the source of scheduler supervision, not a large VLM running on every dispatch or new VLM fine-tuning.

**Retained D study:** minimize importance-sensitive waiting/lateness under the existing declared simulator budgets. Delay-trained regression, choice, classifier, neural ranking and RL comparisons remain as specified below. D is the operational cost criterion on cohorts with valid references; P is the first training experiment, not a substitute definition of delay.

**Retained E study:** minimize first-service delay of distinct independently supported facts. It is required for claims about useful novelty/redundancy. A shuffled broad pool cannot manufacture physical correspondence.

The main proposed model is a shared temporal encoder plus Deep Sets contextual scoring, trained with RankNet-style Qwen preferences. Static Deep Sets and the former summary MLP remain direct controls. All sixteen former methods are retained; the temporal model is the seventeenth architecture/registry entry. A model's D/P/E variants are label/reward experiments, not separate discoveries.

Every artifact records task_id, input_profile, architecture, target_source, reward_source, seed and source/model hashes. The broad receiver objective is fixed for the first revised experiment: multi-source visual situational awareness, giving earlier service to supported time-sensitive incidents and mission-useful additional evidence while monitoring neglect of routine messages. It replaces building-damage-only receiver purpose; the original building-damage cohort remains a separately reported restricted comparison. UE mission relevance is evidence, not permission for device identity to dictate priority. A neutral phone can show a critical incident; a disaster drone can send a routine repeat.

Ownership remains parameters/ for extraction/SPI/vocabulary/sender KB/serialization; Scheduler/ for inference/training; Evaluation/ for lifecycle/simulator/references; Codec/ paused. All experiments and new outputs stay under the canonical repository's experiments/ hierarchy.

~~~mermaid
flowchart TD
 A[Broad image manifests and real sequence sources] --> B[15 UE missions; causal offered streams]
 B --> C[Frozen parameter producers and sender KB]
 C --> D[Complete parameter record and opaque payload]
 D --> E[Receiver buffer; arrivals continue]
 E --> F[Current candidates; per-UE header windows; processed history]
 F --> G[Shared temporal encoder and set scorer]
 G --> H[Select one; one nonpreemptive attempt]
 H --> I[Processed-header log; no retry]
 I --> F
 J[Offline RGB and corresponding causal histories] --> Q[Qualified Qwen preferences]
 Q --> K[Train parameter-only student]
 K --> G
 R[Separate references and fixed D/E evaluators] --> S[Delay; ranking; evidence; compute report]
 H --> S
~~~

## 2. Final scene-array v2 input and transport boundary

### 2.1 Full-cache and full-information reference input: SCENE93_V2

The authoritative producer contract is parameters/scene_arrays.py plus parameters/scene_vocabulary.py and the supplied final caching specification v2. This replaces ARRAY16 as the full-information reference input. The smaller transmitted profile will be selected by Section18; the extraction/cache dictionary remains unchanged. H0 remains an ablation/control only where genuinely available; no invented global union coverage is reconstructed from largest rectangles.

The ordered vocabulary is complete COCO80 IDs0..79, countable extensions building/bridge/utility pole/traffic sign/animal IDs80..84, and regions road/water/debris/smoke/flame/snow/ground/grass IDs85..92. Use the exact exported registry, not a separately typed list. There are85 countable classes and8 region classes, with93 stored slots in each array. Region count slots are canonical zero with count masks clear.

For each class retain detection count where applicable and largest clipped XYXY rectangle divided by the EXIF-upright image area. Count/area masks, detector_ok and saturated preserve availability. No retained box does not prove absence or safety. Rectangular area does not establish distance, severity, condition, relationship or segmentation. Animal/specific-animal and traffic-sign/stop-sign channels can overlap; they must not be summed as unique assets. Region areas are detector hypotheses, not measured physical extent.

Frozen upstream detector: yolov8s-worldv2.pt SHA256 9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792; Ultralytics8.3.237; FP32; conf.25; class-aware NMS IoU.5; imgsz640; max_det300; no augmentation. Explicit set_classes() uses the93-class registry. CLIP setup source is a1d071733d7111c9c014f024669f959182114e33; actual text-feature/producer hashes must match. Class text encoding is one setup cost. The scheduler does not retune or replace this producer.

The fixed scene feature block is374 entries in this order:
log1p(counts)[93], largest_area[93], count_masks[93], area_masks[93], detector_ok, saturated.
Join by image/raw hash and producer identity outside inference. Features are derived by deserializing the received scene bytes, not by using higher-precision sidecars that the actor would never receive. Float arrays are an audit comparison only. A failed detector produces unavailable measurements, not an apparently empty scene.

Add four received SPI one-hot entries, one relevance and one novelty scalar, their six availability masks, and novelty-cold-start. Unknown SPI has four zero values with all four masks clear; known SPI has exactly one active value and all four masks set. Thus the base semantic student vector is374+6+6+1=387 entries. SPI confidence is not silently added when the finalized record contains only a class. Normal aliases Low only when available. New parameters require an explicit version, not dynamic field discovery.

Current wait, history time gaps and status flags are derived receiver context, not new detector outputs. Never put images, raw boxes, embeddings, payload contents, producer hashes, source labels, annotations, true delay budgets, fact identities, future arrivals/outcomes or remaining episode horizon into learned features. Full routing identity groups history and routes selection but is not a learned UE ID embedding.

### 2.2 Exact bytes and compatibility

The standalone scene block is369 bytes =2,952 bits:
2 version/status +32 vocabulary SHA256 +32 producer SHA256 +186 uint16 count slots +93 uint8 area slots +24 mask bytes.
Area rounding error is at most.5/255. Vocabulary/producer/order mismatches reject before inference. Mask padding, illegal region counts, invalid flags, length and incompatible models must reject. The current historical35-byte header is not declared integrated with this block. 35+369=404 bytes is merely arithmetic until a real composite wire envelope exists. Actual total parameters equal serialized routing/envelope plus SPI/relevance/novelty plus this block; report measured bytes and avoid adding the scene block twice when also exporting JSON. Codec payload bytes remain unknown while Codec is paused.

The374-feature block and387-entry augmented vector have different meanings from byte length; tensor float32 memory is not radio payload size. Temporal context is receiver memory, not automatically transmitted on every message. Profile/version/hash metadata belongs in artifacts and compatibility checks, never predictive features. All previous H0/ARRAY16 weights are dimensionally incompatible and require distinct artifacts/retraining.

### 2.3 Causal preprocessing and caching

The existing4,000-record CrisisMMD cache selection remains an upstream extraction workload, not evidence of broad coverage or completed93-class inference. Its supplied report states software tests/fixture notebook passed; actual expanded-vocabulary accuracy, all4,000 detections and CPU/GPU costs remain unmeasured. Reuse successful rows only after raw-image, scene-registry and producer hashes match. Do not restart extraction or mutate that notebook from this scheduler revision.

Image-only scene arrays can be cached once and reused under new compatible missions. Mission relevance and transmitter novelty cannot be reused across changed missions/source-prefix histories. Sender KB follows the producer's causal per-UE/mission contract (current experiment FIFO128): query -> serialize -> commit each offered observation; reset across incompatible sessions/splits. Receiver policy must not alter this sender stream. A broader image pool does not authorize replacing the upstream KB algorithm.

The scene producer has already applied log1p to its returned count features. Apply it exactly once. Fit missing-value imputation and standardization on training groups only. Constant standard deviation becomes one; masks/status flags remain explicit. Registered always-inapplicable region counts have zero storage/masks, not invented negatives. Freeze feature order and preprocessing per artifact; no dynamic vocabulary expansion from evaluation data.

## 3. Reference labels and leakage boundary

Keep runtime evidence and evaluator/reference files separate, joinable by full message identity outside scoring.

1. Published valid priority labels: reference tier only where image/input correspondence is defensible.
2. Published object/condition annotations: supported task facts, not invented urgency.
3. Qwen labels: silver mission-relative acceptable-deferral tiers, pair preferences, aspects, prefixes or replay guidance.

Native MEDIC labels do not establish four-class emergency urgency. TREC-IS labels describe posts; image-to-post support must be established before image-only priority use. MEDIC/CrisisMMD and xBD/xView2 are overlapping naming/source relationships, not independent domains.

Unknown references remain unknown. They stay in mixed replay but cannot become Low, zero-cost or negative examples. D training requires all offered jobs' reference tiers known for that episode. Mixed-reference evaluation reports assessable-cohort coverage and costs explicitly; it is not a total-objective score. E requires declared fact correspondence. Teacher-only evaluations establish agreement under weak supervision, not verified emergency effectiveness.

Reference tier z is judged from actual image, receiver purpose, UE mission and any causal source context used by the rubric. It is not copied from the runtime predicted SPI. Cache keys include image, mission, receiver purpose, source-prefix/context and label-version hashes. Never reuse a context-dependent label for an incompatible stream.

Group exact/near duplicates, parent derivatives, annotated assets/conditions and available event/site identity before splitting. Use 60/20/20 group train/validation/test with seed1729; retain whole groups and report actual sizes. Use existing official source roles where required. Existing 300-image MEDIC and historical exposed panels are development-exposed; they do not become a fresh final test by reshuffling. Prefer out-of-fold SPI predictions; disclose frozen producer exposure separately.

The original dataset roles remain: LADI and MEDIC/CrisisMMD for compatibility and supported task labels, xBD controlled building overlap for E, RescueNet source transfer, and FloodNet structural-flooding scheduler-task transfer withheld from fitting/selection. Broader everyday/sequence sources are added in Section11; dataset category, anomaly or caption labels do not become emergency priority. An unavailable source remains unavailable, not replaced with guessed labels. Audit20 records/source and retain the2GB new-source-image download cap. Prior240-group144/48/48 counts remain a restricted-cohort feasibility target, not a cap on a logical image manifest or confirmed new assets. Materializing or teacher-labeling a huge pool is not free.

The new main P arm uses only Qwen preferences for scheduler optimization. Native annotations serve independent controls/evaluation, and train the separately named D/E comparator arms where valid. Do not mix them into P and still call it VLM-only. Reusing images from SPI training is allowed only in development with exposure disclosed; those parent/duplicate groups cannot serve as fresh independent final scheduler evaluation. No source's missing annotation is converted to Low.

The teacher's images are privileged training information. There is no guarantee that their distinctions survive into387 parameter features. Audit exact collisions of received features AND permitted history/timing with inconsistent preferences. For repeated identical binary inputs, the empirical majority-label accuracy is an identifiable ceiling on that sample, not a proof about near-neighbor cases. Near-collision disagreement is an ambiguity diagnostic only. If rescue versus recreation or a screenshot fault is absent from all supplied signals, report an information limitation rather than claiming a smarter ranker recovers missing visual relations.

## 4. Primary D objective: declared budgets, exact training targets

All primary time is in d0=1 synthetic service slot until a digital codec supplies measured service. Arrival a_i is complete receiver arrival; dispatch b_i; wait W_i=b_i-a_i. Completion/capture-to-completion are supporting distinct measurements.

Map reference acceptable-deferral tiers to positive budgets:
Critical/Immediate=2d0; High/Prompt=4d0; Medium/Deferrable=8d0; Low/Routine=16d0.
Deadline is a_i+delta_i. This is an experimental requirement, not a real harm threshold inferred by Qwen. Delay sensitivity runs multiply all budgets by .5 or2; these settings are fixed, not tuned to make a winner.

Define:
ell(W,delta) = W/delta + max(0,W/delta-1)
W_i(t) = max(0,min(t,b_i)-a_i), with b_i=infinity until dispatched
L_D(t) = sum over a_i<=t of ell(W_i(t),delta_i)

Accrued cost freezes at dispatch; it is not erased. Jobs still waiting, including late and never-selected jobs, continue accumulating cost until dispatch or the horizon. Dispatch at its deadline meets the budget. There is no minimum waiting time requirement. Qwen runtime is offline annotation cost, never a subtraction from delta.

Primary reported cost is J_D=L_D(T)/N_assessed, dimensionless, on matched complete-reference episodes. Record L_D, N and eligible arrival count. Also report per-tier waits and misses; a single average cannot establish correct urgency handling.

The supervised immediate-delay target is:
y_i = ell(W_i+d0,delta_i)-ell(W_i,delta_i)
ymax=2d0/delta_min for the active budget configuration
v_i=y_i/ymax in [0,1].

y_i is the exact cost of one-slot deferral under the convention, not a globally optimal action value. Use Huber regression or ordering losses on this target. The finalized parameter record is not assigned importance by a hand formula; the network predicts the reference-derived target using its inputs.

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

Core task schedule: first run static Deep Sets-P and temporal Deep Sets-P against the fixed untrained references on the same real teacher labels; train a summary MLP-P on those same pairs as the small distillation control. The temporal P arm is the new main candidate. The original sixteen feasible arms remain in D, and original Deep Sets/PPO/DDQN also run E where assets are valid. VisionPrefer remains a separately named weak-supervision comparison. Temporal-D is a matched mechanism comparison only on a valid D cohort; this avoids a full architecture-by-task factorial. No new urgency labels or rewards are inferred merely because the image pool is larger.

## 6. Shared set scoring and causal temporal extension

### 6.1 Base models and shared information

The full-reference semantic vector x is387 entries from Section2. A selected profile retaining K complete count/area class groups has4K+15 entries before timing/context; unequal count/area subsets use their exact versioned registry dimension. Losslessly sparse full93 transmission reconstructs the same387 entries. Append log1p(current receiver wait/d0) to current candidate tokens. Processed headers carry age-since-attempt and a history-role bit. Keep the most recent32 processed headers globally. Completion records an attempted job/header, not verified reconstructed meaning; no retry or semantic-success oracle is added.

Summary models retain the previous candidate, masked queue/history means, availability fractions, log1p sizes, empty-history flag and minimum comparable history distance with validity indicator. Only co-available standardized values enter distance; do not take distances over IDs or mix masks as image meaning. Exclude self. No comparable fields means missing comparison, not observed equality. A summary deliberately compresses legal history; its information loss is tested rather than hidden.

MLP remains64/32 ReLU, scalar output, no dropout/batchnorm. Linear remains one affine score. Static Deep Sets retains phi32/32 ReLU, separate ready and processed mean pools, candidate skip features, set sizes/empty flags, and rho64/32 ReLU. Set Transformer remains two blocks, dimension32, four heads, FF64, layernorm, dropout0 and legal masks. All models receive the same eligible messages and frozen producer; static/temporal context representations are an explicit factor.

### 6.2 Main temporal ranker

Use one shared GRU, not one trained model per UE. At decision t, construct H_i(t) from the three most recent complete receiver arrivals of candidate i's source/session before t, excluding i itself. Retain a four-record recent-arrival ring per UE so excluding a recent candidate still permits up to three records; an older candidate uses the latest three. Mark each history item as waiting/processed/in-flight using current lifecycle state. This prevents 'has arrived at receiver' from becoming 'already decoded'. Source capture order is used only when trusted and consistent; otherwise order by receiver arrival/sequence and record temporal_order_source. The same ordering rule feeds teacher and student. All history records have arrived by t; no future source frame appears.

Take the selected three records in that declared order. Project each x through shared phi into32 dimensions, append log1p(time gap/d0), lifecycle status and gap/availability indicators, then encode with one GRU hidden width32. Reset the GRU to zero for each bounded window: hidden state cannot secretly retain older images after a split/session reset. Empty history has a zero state and explicit flag. Full routing IDs are dictionary lookup keys only, never numeric/embedding inputs. Retain independent source/session rings so churn and session reuse cannot cross-contaminate history.

Let h_i=GRU(H_i), e_i=phi(x_i), q=mean_ready(e), g=mean_processed(phi(x)), and u=mean over active UE history encodings. Construct u from one canonical recent three-arrival window per active UE, not once per queued candidate; otherwise a chatty UE is over-counted. A shared rho64/32 scorer returns:
s_i=rho(e_i, x_i, h_i, q, g, u, wait_i, sizes, masks).
The active-UE pool uses received evidence from active UE sessions, not simulator-only future captures. A UE with no received evidence has an explicit empty token. Active-state joins/departures are lifecycle metadata; if a departure signal is unavailable, keep the last session until its declared timeout and log this rule.

The architecture has no fixed UE slots or fixed candidate output layer. Parameters are shared for N=4,50,60,100 and other positive counts. Set means/sizes are permutation-invariant across UEs/candidates, while history order is intentionally meaningful inside each UE. No claim of infinite-scale constant cost follows: storage is O(4N +32 +Q) headers and queue inference is linear in Q for fixed hidden width/history length, with an O(N) UE pool. Cached encodings must be invalidated when history membership, age/status inputs or model weights change. Simply reusing a state after those inputs change would be incorrect.

Use history_length=3 for the frozen first experiment; do not search16/32/128 windows. It permits a bounded actual-image teacher request. The existing32-item global processed history remains. A separately identified zero-history ablation removes temporal features and uses the static set model; a header-window mean control distinguishes temporal order from merely more records. A flat ordered-window MLP control receives the same three header records/gaps/status masks and padding, avoiding an architecture comparison against a deliberately deprived input.

### 6.3 Selection, export and causality

Deploy argmax over currently ready candidates, with ties by arrival/sequence/full identity. Busy means no dispatch. Training PPO samples only legal candidates. Empty ready queues wait for actual arrivals, never a learned WAIT. Invalid model artifacts fail before service; unexpected nonfinite scores invoke FIFO plus counter.

Export the full387-feature registry, transforms, shared phi, exact GRU equations/gate order/bias convention, pooling and rho. PyTorch and NumPy GRU formulations must agree at gate boundaries, empty windows and mixed window lengths; do not assume the existing dense exporter implements recurrence. Include activation/gate FLOPs and context rebuilding in measured cost. Pending/history state and source rings are part of resumable environment checkpoints; weights-only is not a full resume.

## 7. Locked learned-method registry and losses

All sixteen existing IDs remain recognizable; objective/profile versions distinguish new experiments from historical fixture models. Old binary fact-gain classifiers or completion-coverage checkpoints are never relabeled as newly trained methods.

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
| deepsets_ranknet | Static Deep Sets scorer | L_pair; retained D model and matched P/E controls |
| set_transformer_ranknet | Attention set scorer | Same L_pair and sampled pairs as Deep Sets |
| listmle | Deep Sets scorer | ListMLE on deterministically sampled strict three-candidate reference orders |
| masked_ppo | Deep Sets actor and pooled critic | Clipped PPO + value MSE - entropy |
| masked_double_dqn | Deep Sets candidate-conditioned Q | Uniform-replay Double-DQN Huber TD loss |
| td_per | Same DDQN | Same TD loss; mixed TD-PER sampling/importance correction |
| ranknet_warmstart_ppo | D-trained Deep Sets actor initialization | Same PPO loss/reward; optimizer/critic fresh |
| rl_vlm_f | Same DDQN architecture | Double-DQN TD loss using preference-learned reward, not D reward |
| vlm_replay | Same DDQN architecture | D reward fixed, guided sampling only |
| visionprefer | Deep Sets scorer | L_pair from four-aspect average teacher preferences |
| temporal_deepsets_ranknet | Shared GRU plus Deep Sets scorer | L_pair from qualified Qwen P labels; new main arm; matched D version only on valid references |

For the first P comparison, linear-P and summary-MLP-P are the existing linear_choice/mlp_choice architectures with task_id=P and a two-candidate teacher target p_A=t, p_B=1-t. Their two-way choice cross-entropy is exactly the RankNet binary preference loss on s_A-s_B, including t=.5 ties. This preserves their choice-loss definition while matching the accepted pair labels; do not reuse D-best-tie targets in the P arm. Static/temporal RankNet-P directly use L_pair. Priority-classifier labels still require four-tier supervision; LambdaMART/ListMLE still require their supported grades/orders, not guessed targets from disconnected pairs.

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

Backend: Qwen/Qwen2.5-VL-7B-Instruct with bitsandbytes four-bit NF4, double quantization and FP16 compute, direct Transformers subprocess and SDPA. NF4 is weight compression, not a model family or reliability guarantee. Pin the official model/processor revision before labels and record weights, prompt, quantization and environment hashes. Verify an implementation-pinned runtime; do not claim an untested dependency version works. Use deterministic instruction generation without Qwen3.5-specific thinking/DeltaNet settings. Verify the explicitly assigned GPU; do not inherit the old worker's mandatory two-GPU/GPU1 checks.

Qwen3-VL-8B-Instruct is the documented alternative, not an automatically launched second teacher. A switch requires a recorded reason, backend-specific qualification and accounting within the unchanged216-call/compute ceilings. If it cannot fit, record blocked_labels or blocked_input rather than reuse another model's qualification. InternVL3.5-8B is a research alternative, not an added mandatory run. No model/prompt hunting on final results; maturity does not establish emergency-judgment correctness.

Teacher sees actual RGB images, declared receiver purpose, UE mission paragraphs, trusted/simulated metadata clearly distinguished, current receiver wait and bounded causal context. Hide received SPI/relevance/novelty/scene scores, scheduler predictions and independent reference facts. Show up to two current candidate images, their H_i windows of up to three source images each, and up to two most recently processed images: at most ten unique RGB images/request. Reuse a displayed current image by reference if it also occurs in another window. Every image has neutral role/time/status labels; never concatenate them into an unmarked collage that destroys chronology. Teacher context is bounded, not a claim of complete receiver knowledge.

At deployed replay the student retains the32-header processed context and the full ready/active-UE pools, while Qwen never receives an unbounded queue. A processed image is shown as source evidence from an attempted job, not as a verified decoder reconstruction. For the first P optimizer and held-out teacher-agreement probes, construct an explicit PAIR_VIEW: queue pool q contains only the two displayed candidates, processed pool g only the two displayed processed records, and UE pool u only the displayed candidate-source histories. Build both teacher and student views from the same immutable snapshot; do not let hidden extra UEs/candidates enter those pooled inputs. This removes the false claim that merely capping g makes full contexts equal. Static/temporal/flat-history controls use the same projection; removing history is an intentional information ablation. At deployment the same shared scorer receives the full legal queue, and the expansion from paired training contexts to large queues is a generalization test, not already validated behavior. Report pair-view agreement and full-queue D/E outcomes separately. Teacher-local pairs do not certify a full-queue optimal choice.

Upstream novelty may summarize longer sender history; it remains an allowed producer feature. The offline teacher is not granted unseen future or unsent source frames, and must abstain when the displayed history cannot justify a claimed novelty distinction. The RGB-versus-parameter representation difference remains privileged distillation, even after visible context membership is matched. It must return structured JSON with choice/tie/uncertain, concise visible evidence and unknowns. Never request invented numeric real-world deadlines or use self-rated confidence as validated correctness.

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

This allows at most48 newly tier-labeled context-records absent reusable compatible labels. It is a feasibility label budget, not evidence of a sufficiently powered journal cohort. Reuse independently compatible cached labels only with verified provenance. Never pretend48 contexts or repeated queues equal240 independent images. Candidate pairs use16 training,8 validation and8 locked pilot-test context groups before collection. Prefix pairs retain24 training and8 validation groups; they are reward-model feasibility labels and supply no separate teacher test claim. Group allocation applies to source/video/incident components, not just pair-row IDs. If overlapping pair components prevent these roles, retain components and mark missing-role support rather than split the same source history across roles; retain rejected/uncertain counts. Candidate responses also include aspect ratings, avoiding a separate unlimited aspect-label campaign.

Largest planned teacher request (up to ten unique images) is preflighted before collection at the actual registered image-token settings. The one/two/ten-image checks and reversals consume the declared control/recovery budgets, not uncapped extra inference. At most2 current images plus bounded necessary causal context per tier call; resize/image-token settings are pinned from the working registered backend and recorded. If planned context exceeds the measured envelope, skip/mark insufficient rather than silently omit relevant history or exceed resources.

Controls require independently defensible expected answers. Gates: >=95% valid schema, >=.80 balanced strict-preference accuracy, >=.75 tie and unanswerable handling; report denominators and uncertainty. Accepted oriented pairs must agree after swapping; disagreement is uncertain, not a forced majority. Unsupported factual assertions invalidate the affected label where independently detectable. These gates do not certify real triage accuracy.

**RL-VLM-F-inspired reward.** Prefix comparisons start from the same state/offered trace and end at the same elapsed time; show dispatch times and enough candidate context to expose neglected important messages. Do not compare final collages or reward more pictures automatically. Only causal observations appear.

Reward model uses permitted header/context information and the chosen candidate, never reference labels. Its state is explicitly post-dispatch: remove the selected header from the waiting set, retain it as in-flight context, update processed-header history only at completion, and admit subsequent messages at their actual arrival times. A bounded rate c_psi(s_post,a)=-sigmoid(h_psi(s_post,a)) models cost while waiting work remains; force rate zero when no job is waiting, including idle gaps. No direct VLM scalar is treated as calibrated reward.

Integrate this rate with one fixed left-sample rule on a global grid of d0/4 anchored at episode time0, additionally split at actual arrival/dispatch/completion/horizon events. Freeze the sampled rate within each resulting microinterval and sum rate*duration. Artificially partitioning a stored transition does not create a new sample; reuse the same physical-event/grid path. This is a declared numerical integration approximation, not an exact analytic learned-cost integral. Preference probability=sigmoid(R_A-R_B) for equal-time prefix integrals; optimize pair BCE, weight_decay1e-4 and Adam3e-4. The action/remaining-queue distinction is needed even for one-decision prefix feedback.

Store immutable header-cache references and event-stream slices sufficient to reconstruct all reward microstates and ages, rather than relying only on pre/post decision snapshots or a mutable queue. Relabeling uses exactly the same grid, event boundaries and reward version. Do not require full copied queues at every tick; deterministic event replay from pinned immutable data is sufficient. Report integration/feature-building cost separately from the actor.

Fit/reward-update blocks are frozen while DDQN learns. After a reward-model change, relabel every replay transition using stored snapshots/durations before resuming. At least16 strict train and8 held-out strict prefix pairs, >=.75 held-out accuracy and improvement over constant-majority are engineering requirements. Insufficient feedback marks rl_vlm_f blocked_labels, not successful RLAIF.

**VisionPrefer-inspired arm.** Adapt only its multi-aspect preference-construction mechanism. Rate mission usefulness, visible evidence clarity, additional information under the shown context, and condition specificity on integers0..4; allow uncertain. Average four supported aspects, derive strict preferences/ties, train Deep Sets L_pair. Equal averaging is an explicit weak-supervision convention. No aesthetics or text-to-image generator tuning is introduced. This is not reproduction of the original VisionPrefer benchmark.

First P execution uses the control, tier and candidate-pair portions only, with unused prefix/replay call slots reserved for the retained RL studies. It does not spend an additional216 calls per model. All static, temporal, linear/MLP P models and VisionPrefer reuse the same eligible pair/context records. There are only32 unique candidate pairs in this first budget: it is a feasibility experiment, not enough evidence for a broad journal ranking claim. A large image manifest does not multiply teacher labels. Qualified existing labels may be reused only with matching task/mission/history/schema provenance. Further collection requires a separately budgeted run.

The main P learner uses L_pair only, optimizer weight decay as specified, and lambda_D=lambda_E=0. Qwen pair output is A/B/tie/uncertain, plus concise supported evidence and named change/additional-information/mission factors. An explicit tie maps to target.5, uncertainty is skipped. Do not convert self-reported confidence or explanation length into a label weight. Every queried comparison belongs to an actual causal ready snapshot or a declared common probe; its pair label is not an assertion that one candidate is globally best among100 unseen alternatives. Train P on PAIR_VIEW contexts as defined above. Validate deployed full-queue next-choice behavior separately by D/E on supported cohorts and through qualified small-set choices where separately labeled. Do not claim pair accuracy proves globally consistent selection. D/E learners continue to train on full simulator contexts under their own independent targets.

Temporal labeling is identical except that the ordered H_i image windows are visible. There is no extra no-history teacher labeling pass in this216-call run: the32 swapped candidate pairs already consume64 calls. Student no-history, flat-history and mean-history controls reuse those same accepted labels without extra inference. Reversing A/B preserves each UE window and roles. History-order perturbation is a student diagnostic on real ordered streams; arbitrary time reversal is not automatically a physically valid labeled trajectory.

Round0: controls, compatible tier labels and initial candidate/prefix labels. Train initial models. Round1: remaining training-only prefix/replay requests from their experience, within the same216 cap. The actor never waits for Qwen. Missing replay guidance stays neutral. Rejected teacher feedback does not stop reference-supported methods.

## 10. Fixed baselines and fair comparisons

Untrained: FIFO; round robin by source; seeded random; predicted-SPI then FIFO; relevance-only; novelty-only; lexicographic SPI then novelty; predicted-SPI-budget EDF. If a required field for a scalar rule is missing in any ready candidate, that rule falls back to FIFO for that decision and logs the fallback. RR/random do not consult semantic availability.

Learned classifier plug-in greedy is required: compute expected one-slot y under its predicted reference-tier distribution, then select the largest. It separates learning urgency from sophisticated scheduling. EDF uses predicted budget and absolute arrival+budget, not true reference delta.

Privileged diagnostics: true-budget EDF (not claimed optimal); exact tiny legal-sequence optimum; reference marginal-fact greedy. None is a fair deployed competitor. A greedy schedule is not a lower bound. No WAIT or future observations are added to deployed actors.

The first P comparison contains FIFO/RR/random, SPI-FIFO, relevance-only, novelty-only, SPI-then-novelty and predicted-budget EDF; linear-P, summary-MLP-P, static-Deep-Sets-P and temporal-Deep-Sets-P train on identical accepted pairs. The flat-history MLP and mean-history controls isolate representation. None of the rule baselines is trained on hidden reference priority. Delay-trained neural/ranking models are reported as target-source comparisons when ready, not misrepresented as P learners.

D core comparison retains all16 former feasible learned arms and the baseline set; the temporal-D mechanism control is additional. P/teacher arms differ in label/reward source by design. Architecture comparisons match reference targets/pair budgets. Replay comparisons match actor/reward/training steps and the uniform-mixture setting. Warmstart comparisons count pretraining and RL decisions. Never call multiple losses on the same architecture independent new algorithmic discoveries.

## 11. Broad image pools, dynamic UE missions and stream construction

### 11.1 Source roles: broad manifest, selective acquisition

Use diverse established sources, not an indiscriminate web scrape. The selected core additions are COCO and Open Images for everyday mixed scenes, plus BDD100K for real temporal streams when an authorized image/clip subset is accessible. ImageNet and Places365 add object/scene breadth when existing assets or access permit. Their category labels support sampling and detector checks, not scheduler priority. Screenshots need an explicit separate UI source; a laptop detector class cannot prove an operational error.

| Source and primary reference | Added value | Use/limit |
|---|---|---|
| [COCO](https://cocodataset.org/) | People, objects, indoor/outdoor clutter, food/sports/furniture with annotations | Core everyday pool; no urgency labels or temporal order |
| [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html) | Diverse scenes, labels/boxes and selected relationships | Core broad pool; published relationships may test a missing signal, never leak into actor |
| [ImageNet](https://www.image-net.org/download-images.php) | Wide object-category coverage | Optional sampled slice; official image access may require login; not natural UE sequences |
| [Places365](https://github.com/CSAILVision/places365) | Indoor/outdoor scene diversity | Optional scene slice; category breadth is not emergency evidence; archive access preflight required |
| [BDD100K](https://github.com/bdd100k/bdd100k) | Actual ordered driving videos with varied conditions | Core temporal source if accessible; single annotated keyframes alone are not a sequence |
| [DoTA](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly) | Traffic-anomaly temporal boundaries and clips | Targeted temporal stress; anomaly is not a calibrated four-tier priority; released frame archive is large |
| [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) | Normal/anomalous surveillance episodes | Optional CCTV temporal stress; coarse training video labels do not label every frame |
| [CrisisMMD](https://crisisnlp.qcri.org/crisismmd) / MEDIC | Existing disaster observations and compatibility assets | Preserve pinned4,000 selection and known exposure; not general93-class coverage |
| [xBD/xView2](https://github.com/DIUx-xView/xView2_baseline) | Annotated buildings/conditions and controlled overlap | Original independent E arm; not two independent datasets |
| [RescueNet](https://github.com/BinaLab/RescueNet-A-High-Resolution-UAV-Semantic-Segmentation-Dataset) | Aerial disaster scene/source transfer | Preserve supported label mapping and source-held-out report |
| [FloodNet](https://github.com/BinaLab/FloodNet-Supervised_v1.0) | Structural-flooding task transfer | Still withheld from scheduler fitting/selection for the original transfer claim |

These are evaluated source choices, not a claim of downloaded assets. Official ImageNet image downloads are access-controlled; several Places/BDD host requests did not resolve during this inspection, although their author repositories describe the sources. DoTA's author release describes55GB of extracted frames, exceeding this block's2GB acquisition cap. Do not download a giant archive to obtain a handful of frames or silently use an unverified mirror. Preflight schema, image bytes, licenses/terms, annotation validity and source identity for20 records before making a source active. The BDD paper/toolkit source supports the temporal choice; it does not establish that a particular local archive is available.

Keep one source manifest scalable to large pools: source/version, original identifier, content hash, parent/video/event group, official split, annotation provenance, acquisition status and license/attribution metadata. Selective caching, not the manifest length, determines compute/storage. Target an additional2,000 static parent images for feasibility (800COCO,600Open Images,400Places365,200ImageNet), plus at most120 real clips with up to8 extracted frames each if small authorized assets fit the same2GB new-source cap. Existing disaster rows are additional; intended totals are not counts of materialized independent groups. Access failure leaves the quota unmet and source unavailable; no silent redistribution or claim of completion. These caps are workload choices, not power calculations.

Source metadata and grouping remain outside learned features. Audit duplicates across all collections before splitting, including disaster copies reproduced in general-image sources. Split whole video/incident/parent components60/20/20 before selecting frames or episode construction. An official held-out source split is not casually repurposed as training. Final family/held-out-task tests cannot be used to adjust the source mixture.

### 11.2 Three named stream families

**MIXED_IID (40% of first training traces):** independently sample diverse images into queues. This directly tests the user's ordinary-photo/traffic/disaster/food/person-near-fire mix. It is a valid synthetic mixed-workload stress, but assigned ordering/location is not a real trajectory. Mark capture geography unknown or simulated; do not describe random unrelated pictures as repeated views of a place.

**CAUSAL_SEQUENCE (40%):** assign real source clips to compatible UE missions, preserve frame order, and interleave multiple UEs by receiver arrival time. Sample up to8 source frames per clip using declared source timestamps; arrival jitter never exposes a frame before its own capture/extraction/transport readiness. Repeated adjacent frames, changed appearance and changed condition are distinct cases. Only annotations/declared exact-copy construction establish shared facts or actual change; appearance difference alone does not.

**MISSION_CONTEXT (20%):** use eligible parent images under different mission paragraphs in separate grouped traces, recomputing relevance/novelty. Include high-SPI/low-relevance, low-SPI/clear incident, high-novelty irrelevant content and crowded-routine versus sparse-critical comparisons. These are challenge strata, not automatic preference labels. If the evidence does not support the contrast, Qwen may tie/abstain. Keep matched image derivatives in one split. The common receiver objective stays fixed; UE mission changes do not create fifteen unrelated definitions of global priority.

Sample sources by the frozen per-family manifest and report realized proportions. Do not label all COCO images routine or all CrisisMMD images urgent. Include source-matched positives/negatives and source-held-out results; a learned source shortcut can survive removal of source IDs. Never route all critical images to one UE type. Keep cold starts, local exact repeats, cross-UE repeats, long unserved histories and all-empty detector scenes. Exact-copy parent identity is evaluator-only. Random pool diversity and temporally meaningful novelty are separate mechanisms and separate reported strata.

### 11.3 Fifteen retained UE templates

Each template supplies device description, mission paragraph, capture-rate/trajectory metadata and availability. These are project templates informed by IMT-2030 use-case families, not fifteen standardized6G classes. Device type never grants priority.

| Device | Fixed mission intent |
|---|---|
| General smartphone | Share user-selected visual observations; no presumption of emergency |
| Citizen incident phone | Report visible incidents and changes useful to local response |
| Structural survey drone | Inspect buildings for visible damage and newly exposed structural evidence |
| Person-search drone | Locate visible people and evidence relevant to a search task |
| Flood/evacuation drone | Observe inundation and passable/blocked evacuation routes |
| Wildfire drone | Monitor visible smoke/flame and supported changes in fire conditions |
| Road/bridge drone | Inspect transport structures and visible obstruction/damage |
| Ordinary vehicle camera | Record immediate road scene and obstacles relevant to travel |
| Emergency vehicle camera | Report response access and visible incident developments |
| Traffic-intersection CCTV | Observe flow, collisions and obstructions at a stationary intersection |
| Public-space CCTV | Observe visible public-space incidents and crowd changes |
| River/underpass camera | Monitor water/route conditions from a fixed viewpoint |
| Industrial inspection robot | Report equipment/area conditions relevant to its inspection mission |
| Agricultural drone | Observe crop/field conditions and supported changes |
| Maintenance field camera | Document infrastructure inspection findings and relevant change |

The complete actual paragraphs are versioned extractor inputs, not labels generated afresh for each scheduler. All learning arms use the same paragraphs/parameters. Because the student receives relevance rather than raw mission language, this study tests the selected task-relative parameterization; it does not establish arbitrary language-instructed scheduling. If a changed mission yields identical permitted features but requires opposite behavior, record an information collision, not an invented hidden mission embedding.

### 11.4 Workload, timing and outcomes retained

Populations4/12/50/60/100 with join/pause/resume/leave; variable per-source message counts. Train on32..512 offered messages/trace and test scaling with50 images/UE (2,500/3,000/5,000 for50/60/100). One hundred active UEs need not mean100 queued messages. Population, load, source distribution and history length are independent declared axes, not one giant factorial.

Retain service d0=1 primary; lognormal(mean1,log-std.5) stress; offered loads.5/.9/1.2; initial backlog min(8,N); separate all-at-once control; uniform arrivals or two bursts containing70% of traffic in20% of the arrival window. Compute window from offered count/load and report realized backlog/rates. Hold total load fixed for population comparisons; report load-growth experiments separately. T=arrival-window end+32d0 is fixed before replay and hidden from actor.

Capture -> extraction-ready -> complete arrival -> dispatch -> finish are distinct timestamps. Use real source timestamps where known, otherwise clearly simulated times with uncertainty. Stationary cameras do not move; bounded synthetic speed caps2/2/15/20m/s for phone/robot/drone/car remain illustrative. Unrelated internet photos receive no fabricated claim of coherent physical GPS. Missing capture time/location stays unavailable; inference runtime is measured rather than silently subtracted from a deadline.

UE departure does not delete received jobs. Session resets clear the correct rings/KB and never merge identities. Every offered message is accounted as not-arrived, queued, in-flight, processed or explicitly excluded. Hidden identity-qualified outcomes/service realizations are generated once and shared across policies. All-success core;5% exception stress and1/10% sensitivities; unchanged durations/outcome-hidden policies must have unchanged selection-cost trajectories. No retry and no reconstructed-quality feedback. Only three unit-duration jobs can start by time2 at0,1,2: infeasible simultaneous budgets are recorded overload, not a fabricated training failure.

## 12. Evaluation metrics, exact definitions

**P first-experiment report:** held-out qualified pair agreement, strict/tie/uncertain counts and per-family/mission/source breakdown. A teacher-labeled pilot test is silver agreement, not independently verified triage. Report D timing and E evidence where valid on the same replay; absent references mean those cells are unavailable, not a zero score. Compare temporal versus static/flat-history/mean-history under identical labels before attributing gains to temporal reasoning.

**D operational criterion retained:** macro-average J_D across matched episodes/load strata; lower is better. Pair with independently reference-tier-conditioned timing/misses. Do not claim urgent performance from an unweighted mean across all images.

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

**Transport:** standalone369-byte scene block plus measured complete parameter/envelope bytes, candidate profile/vocabulary hashes and serialization latency. Report JSON and binary alternatives separately rather than double-counting both as transmitted. Required scene-value ablations are SPI/relevance/novelty only, those fields plus93-class arrays, and those fields plus temporal history. Class-family ablations remove corresponding values AND masks/history-derived features, not a manual channel-importance formula. Count/area/mask-only and detector-unavailable controls test shortcuts. The full-information reference retains all93 channels. A separately versioned compact model may use the train/validation-selected subset from Section18; this changes its transmitted feature profile, not the cache vocabulary. Untransmitted classes are excluded by profile, never reported as observed absence. No synthetic payload-size or radio-spectrum improvement claim. Selector overhead is measured; synthetic slot results are not milliseconds. Report a separate overhead-charged sensitivity once a measured selector-to-service scale exists.

## 13. Model selection, splits, seeds and statistics

Training seeds1729/2718/31415. Separate RNG streams for model, training traces, validation, test and service outcomes. The correctness run is fixed2–3 episodes,3 epochs,64 RL decisions/arm,1 seed; it is never merged with study-performance results.

Validation set:15 matched traces covering populations4/12/50/60/100 and three loads; message counts48/96/256/384/512 by population (15 matched traces). Same validation trace IDs across arms. Validate supervised models each epoch and RL each2,000 decisions.

Checkpoint rule for each arm: lowest validation macro J_D. Ties within1e-6: higher assessable Critical within-budget fraction if defined, then fewer parameters, then registry ID. No test results enter selection. P-only and E-specific checkpoints select their named validation objective (P accepted-pair accuracy, ties by lower held-out pair BCE then fewer parameters; E cost); D results are still reported where assessable. Learned-reward RL is selected by independent D validation, never by its own learned reward. Missing independent selection references block that claim.

Promotion rule across selected D policies: a learned policy is eligible as the recommended operational candidate only if, on the same validation traces, its Critical within-budget fraction is no lower and its Critical restricted mean wait no higher than predicted-SPI-FIFO. Among eligible policies choose the lowest macro J_D with the tie rule above. If none qualify, retain the baseline and report the negative result; do not change the threshold or hide failed arms. If Critical references/follow-up are absent, a Critical-service recommendation is undefined, not passed. This is a fixed sample-based engineering guard, not a statistical guarantee or a promise of feasible deadlines during overload. Final test results report all preregistered methods even if they fail promotion.

Final evaluation uses locked test groups once after selection, on matched workloads. Bootstrap1,000 paired parent/duplicate groups and regenerate matched traces; queue seeds are not independent incidents. Report effect sizes/intervals, per-source failures, counts and all chosen seeds. Source-held-out and event-held-out are separate. FloodNet transfer is scheduler-unseen task only; vision-model exposure is disclosed.

No architecture/optimizer hyperparameter search in this fixed study. The explicitly user-authorized bounded transmitted-field selection in Section18 is a separate exception; it does not authorize an unrestricted sweep. Budget/common-scale sensitivities are prespecified, not used to pick a favorite budget. If no model dominates timing and semantic outcomes, report the trade-off rather than invent a post-hoc overall winner. One-seed or tiny-label results remain feasibility evidence.

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
16. Scene v2 exact369-byte roundtrip,93 order,374 scene features/387 semantic entries, region mask semantics, wire-derived area quantization and one-time log1p.
17. Temporal ring/window construction, lifecycle status, source/session isolation, history membership at decision time and no future images in teacher views.
18. UE/candidate permutation invariance with history order preserved; intentional history-order changes can change temporal output without introducing array-position IDs.
19. Static/flat-history/mean-history versus GRU same-context controls, NumPy GRU parity and reset/checkpoint continuation.
20. Broad-pool source/clip grouping, MIXED_IID versus CAUSAL_SEQUENCE provenance, mission-dependent cache recomputation and scene-cache reuse checks.
21. Teacher request unique-image cap, swap consistency with histories intact, pair split component isolation and final silver-test access log.
22. Exact feature/history collisions and source-predictability diagnostics; no claim that93 classes recover unencoded visual relationships.

Do not run the same smoke repeatedly without new changes. Existing focused suites remain in Test/tests. New research checks/benchmarks and their results stay in experiments/. No data/parameter-team sources are modified by scheduler work.

Each run contains manifest.json, config.json, environment/hardware/source hashes, runtime/reference manifests, raw teacher responses, targets, per-decision/per-trace/per-group metrics, learning curves, model/checkpoint/export files, logs/stdout/stderr/errors, figures, validation.json and REPORT.md. Preserve failures/partial runs and executed notebooks with cell outputs. A completed process is not proof of semantic validity.

## 15. Fixed execution order, resources and implementation status

Execution order now implements the user's VLM-first request without discarding the rest:
1. Freeze supplied scene-v2 source/artifacts, preserve dirty working tree and old results; consume upstream outputs without launching/replacing its cache job.
2. Implement the387-entry adapter, per-UE windows, same-context teacher records and broad source/trace manifests. Dataset access audits are selective and within the existing cap.
3. Implement retained D/E evaluators/typed labels beside the legacy completion objective so every trained model has a valid operational report where references exist.
4. Run meaningful scene/lifecycle/temporal/leakage/numerical/export tests and the fixed2–3-episode correctness run. A synthetic label fixture validates plumbing only.
5. Qualify Qwen on the exact ten-image request envelope, then collect bounded P preferences. Failure blocks real VLM training; do not silently substitute synthetic preferences as real labels.
6. Train the static/MLP/temporal P comparison first under identical labels; validation/export/resume and independent D/E outcomes where assessable. No SPI classifier or codec work is included.
7. Run the retained delay-trained/classifier/tree and RL/RLAIF/replay arms under their exact original losses and budget. Prefix/replay feedback stays within the same216-call allocation.
8. Run the locked scaling/ablation/transfer comparisons and report all partial, unavailable and negative results. No claim of learned superiority follows from a2–3-episode software pilot.

Main Sol owns code/integration and the canonical record. Medium assesses completed consequential slices without edits. Luna High runs pinned tests, operational monitoring and authorized Kaggle actions. Agents do not reset/pull/move a live checkout. No permanent supervisor or redundant full surveys.

Reuse the project .venv locally; training frameworks stay outside deployed Scheduler inference. Portable artifacts include preprocessing, schema/profile/vocabulary hashes, architecture, objective/label/reward source and weights. Deep Sets/attention/GRU exporters require parity before deployment; existing dense exporters are not assumed sufficient.

CPU first. Six aggregate GPU-hours remains the prior execution ceiling, not a new quota claim:3h extraction/teacher,2h justified accelerator work,1h diagnosed contingency; actual codec work remains paused. Count allocated idle GPUs and loading. Recheck current access/quota before any submission. Reuse established Kaggle notebook identities and parameters/KAGGLE_RUNS.md; no healthy run is interrupted to add logs. Preflight hardware and ten-image/call cost before expensive work. A cap stops execution with a checkpoint/partial status; it does not authorize silent model or label substitution.

The current legacy pilot pre-spi-check-20261030-094517 records15 focused passes, eight fixture-trained learners, three blocked teacher arms and11 waiting-spec arithmetic checks. It does not validate these new D/E losses, Deep Sets/attention, scene arrays or model superiority. The finite J_time values refer to the legacy completion-coverage objective.

Revised specification status: SCENE93_V2 is the full-cache/full-information reference and Section18 selects the compact deployment profile; VLM-supervised temporal P is the first candidate; all sixteen previous models, exact D/E objectives, budgets and controls remain. The new temporal registry entry makes seventeen architecture IDs, with explicitly named target-source variants. Model registry, targets, losses, defaults, teacher allocation and metrics are fixed. Execution blockers are qualified labels, source assets, export implementation and measured resource availability. Record these as blocked_input/blocked_labels/partial_budget/failed_validation; do not relabel them as completed work.

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
- [GRU / RNN Encoder-Decoder, 2014](https://aclanthology.org/D14-1179/): shared recurrent encoding of ordered observations. Its language experiment does not validate our temporal scheduler.
- [Unifying distillation and privileged information, ICLR2016](https://leon.bottou.org/publications/pdf/iclr-2016.pdf): teacher-only representations during learning versus student inputs at inference. It motivates the RGB-teacher/header-student boundary, not a guarantee that omitted visual distinctions are recoverable.
- [COCO paper](https://arxiv.org/abs/1405.0312), [Open Images description](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), [Places365 author repository](https://github.com/CSAILVision/places365), [official ImageNet access](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php), [BDD100K author repository](https://github.com/bdd100k/bdd100k), [DoTA author release](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly): source diversity and sequence/annotation roles, not independent scheduler-priority truth.
- [Qwen2.5-VL-7B-Instruct official model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct): documented multi-image/structured-output capabilities; task-specific qualification remains required.
- [Official AWQ alternative](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ): an alternative deployment, not the selected NF4 configuration.
- [Qwen3-VL-8B-Instruct official model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct): alternative candidate, not verified scheduler-task superiority.
- [ITU IMT-2030](https://www.itu.int/en/ITU-R/study-groups/rsg5/rwp5d/imt-2030/pages/default.aspx): broad use-case context, not an official list of these fifteen UE templates.

The budget mapping, piecewise D penalty, profile templates, aspect rubric and numerical defaults are declared design conventions. Literature does not prove these are universally best, clinically safe, novel, converged or journal-ready. The scientific claim must remain tied to the executed sources, labels, operating regime and checks.

## 17. Change record and acceptance evidence for this revision

Retained from v1: one-shot complete-message nonpreemptive service; all sixteen method IDs and their losses; exact D waiting/lateness equations and E fact-delay objective; no invented real-world deadlines; qualified weak teacher labels;216 total calls; same-policy traces; seeds; train-only preprocessing; independent references; failures/misses; NumPy export/resume; paired-group evaluation; local-first execution and six aggregate GPU-hour ceiling; encoder/SPI ownership boundaries.

Changed by explicit user instruction: primary H0/optional16-class input becomes finalized93-class scene v2; P temporal distillation runs before the retained D/RL suite; broad IID pools are added as legitimate synthetic workload alongside real causal streams;50/60/100 UE scales and fifteen missions remain; bounded per-UE image histories teach the parameter-only temporal scorer. The source/vocabulary is not dismissed because some channels lack disaster relevance.

New or strengthened validity checks: exact387-entry adapter; wire-derived features; unknown detector evidence; per-UE GRU/session invariance; teacher/student history correspondence; broad-source and video grouping; source shortcuts; context-dependent cache keys; explicit silver-test pair role; exact-input ambiguity; measured context-building cost and complete parameter-byte accounting.

This revision is accepted only as a documented protocol update after source/contract inspection and focused assessment. No actual detector inference, teacher inference, learner training, codec work, dataset download or Kaggle launch is claimed. The preserved legacy pilot remains evidence only for its original completion-coverage objective. The next authorized implementation uses this revision; historical outputs retain their original protocol IDs.

## 18. Compact transmitted parameters with the full extraction cache retained

### 18.1 Decision and scope

The user explicitly requested a smaller header selected empirically during scheduler training. Keep all93-class counts, areas, masks, statuses, raw/audit evidence, hashes and existing scalar parameters in the extraction cache. Do not rerun extraction or discard cache columns to make a smaller wire record. A projection/serializer chooses what is transmitted; training consumes the projected received representation. A compact profile requires a versioned registry/model, not relabeling old387-input weights.

Keep the old369-byte scene-array serializer and results intact. It is a full self-describing scene block, not an established370-byte whole-message header. Whether it is expensive depends on the real payload and transport framing, which remain unmeasured. The controlled objective here is fewer transmitted parameter bytes with preserved measured scheduling behavior. Reducing transmitted fields does not reduce the current detector's extraction cost while full caching remains required.

### 18.2 Save bytes without removing scene evidence

The current369 bytes include64 bytes of registry/producer hashes. In a new COMPACT_SCENE_V3 profile, pre-provision an immutable uint16 profile-ID mapping to the full registry, producer, quantization and class-order manifest at both endpoints. Keep full hashes in cache/model/run artifacts. A profile ID is a lookup key, not a truncated hash or an authentication mechanism. Unknown/mismatched IDs reject; no guessing and no resource-grant/retry/online negotiation is introduced. Account for provisioning bytes separately and amortize them only over the actual messages sharing that mapping. Standalone records without the configured mapping retain the self-describing format.

Use a4-byte prefix: version1, status/mode flags1, profile ID2. Full93 dense data then cost4+186+93+24=307 bytes. This preserves the current uint16 counts, uint8 areas, all93 count/area mask bits and detector/saturation status exactly after the original quantization. It saves62 bytes/message when the mapping is already installed.

A sparse full-information alternative retains the24 mask bytes and encodes only classes whose received count or quantized area is nonzero. Store entry_count1, then sorted unique entries of class_id1/count_uint16_2/area_uint8_1 =4 bytes each. Size is29+4L bytes for L entries, including the4-byte prefix. Omitted slots reconstruct zero VALUES under the separately transmitted masks; they do not imply physical absence or unavailability. Successful zero detections and unavailable zeros remain different. Reject duplicate/unknown IDs, noncanonical entries, nonzero region counts, incompatible flags, missing bytes and excess entries. Preserve class-aware overlap semantics.

Choose dense307 versus sparse29+4L per message using the explicit mode bit, whichever is shorter. Examples are arithmetic scenarios, not observed detector sparsity:

| Representation | Scene bytes | Information retained |
|---|---:|---|
| Existing self-describing scene v2 |369|All93 channels, full hashes per record|
| Configured compact dense93 |307|Same received scene measurements|
| Configured compact sparse, L=10 |69|Same received scene measurements|
| Configured compact sparse, L=20 |109|Same received scene measurements|
| Configured compact sparse, L=30 |149|Same received scene measurements|

The sparse mode preserves all available v2 MEASUREMENT VALUES, including already-quantized areas; it is not lossless relative to original floating-point box geometry. Its reconstructed feature vector and model decisions must equal dense v2 exactly when metadata resolves to the same producer. Full93 models need no semantic retraining for this lossless transport transform. Measure actual L distributions, bytes, parsing time and fallback to dense on cached real images before claiming a typical header size. Implementation checkpoint,30October2026: Evaluation/compact_scene_transport.py now implements this format alongside the unchanged upstream v2 serializer. Six focused protocol tests and6000real cached exact-byte roundtrips passed in WSL. Field selection and semantic-quality claims remain unmeasured.

### 18.3 Empirically select a smaller semantic profile

Use the temporal main model as the selection reference, not all17 architectures multiplied by every subset. Keep SPI/relevance/novelty and their availability/cold-start fields in the first deployment profile; evaluate their ablations but do not remove them from the interface in this first compaction study. Select scene class groups before exploring separate count-only/area-only fields.

1. Fit the full-information reference with the existing targets/loss, grouped splits and compute budget. If it has no useful held-out behavior or the producer has inadequate coverage, do not interpret its feature importance as a reliable selection criterion.
2. On grouped development folds only, screen each class by removing its count, area and corresponding availability channels from candidate, queue, UE-history and processed-history calculations together. Rebuild all derived summaries/distances. Perturbing one input while leaking the same field through history is invalid. This masked/full-model screen is a reliance diagnostic, not evidence a retrained compact model needs that class.
3. Check overlapping/correlated groups jointly, especially animal plus specific animals and traffic sign plus stop sign. Low standalone importance does not establish uselessness. Do not rank by detector frequency, neural weight magnitude, raw attention or tree impurity alone. Rare emergency evidence is reported separately from common everyday cases. See [official permutation-importance limitations](https://scikit-learn.org/stable/modules/permutation_importance.html).
4. Build fixed nested candidate subsets of K=8,16,24,32 scene class groups plus the full93 reference. Ranking uses only the development screen; ties use registry ID. Use three grouped folds within the training partition when group/label support permits. The outer validation set chooses among the resulting fixed profiles; the final test stays unopened. A fold missing required mission/reference support is explicitly insufficient, not replaced by queue seeds.
5. Retrain each compact candidate from scratch under the same architecture/loss/defaults/seeds and teacher records. Temporal history is projected through that same candidate profile. Quantize/deserialize before fitting; no unavailable full-cache sidecars enter scoring. Only after a winning class subset is chosen, compare count-only versus area-only removal for that subset as a bounded additional development comparison, not an unrestricted186-field search.
6. Freeze the profile, transforms and model before final evaluation. Retrain other retained algorithm comparisons on that same profile for a fair compact-suite table; keep the full-reference table separate. Do not choose a different favorable header for each algorithm in the main comparison.

For K complete count/area pairs, an intentionally simple compact dense scene profile costs
B_scene(K)=4+3K+2*ceil(K/8) bytes.
The profile registry defines class membership/order, so per-entry class IDs are unnecessary. Region count slots remain zero/inapplicable if included. Example sizes: K8=30, K16=56, K24=82, K32=108 bytes. Those are scene-only proposed binary sizes, not complete headers and not measured implementation outputs. A later asymmetric count/area profile has exact cost4+2C+A+ceil(C/8)+ceil(A/8), where C/A are configured count/area slots; no unsupported scalar-size saving is presumed.

### 18.4 Selection objective, preservation guards and insufficient labels

For every candidate, report actual mean/p95/max transmitted scene and total-parameter bytes, then paired scheduling differences against full-information and SPI-FIFO references. Compare lossless sparse93 as well: a small class subset is not automatically the smallest observed packet if the full scene is naturally sparse.

Freeze the initial engineering tolerance before results: choose the smallest mean-byte profile whose validation P preference agreement is no more than.01 below full-reference agreement. Where independently assessable D exists, also require J_D no more than1.02 times full-reference J_D (if full cost is zero, require zero), Critical within-budget fraction no lower and Critical restricted mean wait no higher than the full reference. Preserve the existing SPI-FIFO promotion guard too. Report High/Medium/Low, source/mission groups and E coverage where available; no broader semantic-retention claim follows when those reference cells are missing. These are sample-based engineering tolerances, not statistical guarantees. Print actual denominators; eight validation pairs cannot resolve a one-percentage-point difference.

If evidence cannot support a trustworthy subset decision, retain all93 logical channels and use the lossless configured/sparse transport candidate after exact-parity validation. Do not permanently erase a category because of32 teacher pairs or one disaster-heavy cohort. Critical rare-channel support, source shifts and supervised performance are constraints on claims; cache completeness makes later reevaluation possible without detector reruns.

No new task utility, model loss, reward or real-world deadline is invented for compaction. Existing P/D/E supervision stays unchanged. Compression selection uses training/development only; test labels never decide the subset. All selection fits/profiles remain inside the existing time/GPU ceiling; if the budget cannot finish, save partial evidence and do not call the header finalized by training.

### 18.5 Required acceptance and ownership

parameters/ retains producer/cache ownership. Scheduler training owns the selected feature registry; its projection removes fields from all context paths. A new compact serializer/receiver adapter must be agreed at that owner boundary, kept alongside v2, and pass exact reconstruction/parity for lossless dense/sparse forms, missing/zero/cold-start semantics, profile mismatch rejection, ordering, corruption/range checks, class-subset artifact compatibility and full lifecycle replay. The existing receiver history is rebuilt from received compact records, never enriched from the offline full cache. Scene-byte budgets do not include opaque payloads; encoder work remains paused.

Artifacts add selection_manifest.json, grouped screen results, every fixed subset and retraining run, profile-ID/full-hash mapping, byte distributions, quality/latency Pareto tables, failed guards and the locked profile. No detector/cache fields, historical models,17 algorithm IDs or prior research results are removed by this revision. Current status,30October2026: compact serialization is implemented and locally verified; actual field importance and performance-preserving subset selection remain unmeasured. All93logical channels are retained. Dynamic-arrival baseline replay is a separately scoped component, not the temporalP/neural/RL training study.


## 19. WSL/Kaggle execution checkpoint — 30 October 2026

User requires WSL Kaggle account execution. Actual local user Satya on Ubuntu-26.04; authenticated Kaggle account satyapaladuguplus. No existing eligible scheduler notebook on that account, so user-authorized START created private identity satyapaladuguplus/semantic-decoder-scheduler-20261030. Canonical source experiments/stream_scheduler/kaggle/Semantic_Decoder_Scheduler.ipynb; version1 accepted and platform reachedCOMPLETE. Remote experiment results/retrieval are recorded separately in parameters/KAGGLE_RUNS.md and are not inferred from platform status alone. CPU-only, no GPU use, installs, detector/classifier/VLM calls or neural training in this component.

Implemented Evaluation/compact_scene_transport.py and experiments/stream_scheduler/mini_kaggle_run.py. Local WSL checks:6protocol tests;6000real cached exact encode/decode roundtrips;12FIFO/RR/random/input-SPI comparisons across4/50/100fixed UE populations and continuous arrivals. Source after review preserves actual unit-service finish and separately observed_until for horizon censoring. This slice does not yet implement UE lifecycle churn or receiver-controller integration for the new scene adapter.

Current YOLO selection: all80COCO classes plus building/bridge/utility pole/traffic sign/animal and road/water/debris/smoke/flame/snow/ground/grass =93logical channels. Input pool shows85channels with nonzero cached measurements; this is detector coverage, not accuracy or importance. Actual implemented compact scene blocks on the local6000cache average34.4513bytes,p9545,max73 using the configured profile. Costs exclude profile provisioning, scalar parameters, routing and codec payload. No permanent class removal has been trained.

Relevance and novelty are intentionally streaming/mission-dependent, not missing-cache blockers. This baseline component does not require them, and does not substitute fake values or SPI predictions as truth. Qualified contextual preferences and the neural/RL portions of this plan remain the subsequent implementation work.


## 20. Final complete-suite execution agreement (v2.2)

### 20.1 Full scope and evidence ledger

Retain all17 Section7 methods, all untrained baselines, P teacher distillation, D waiting-cost and E supported-evidence studies. Main P remains temporal Deep Sets RankNet; independently running D while labels are prepared does not replace P. Models/variants retain separate task, target-source and reward-source identities. The earlier sections' exact architecture/loss/default settings remain normative.

One machine-readable ledger must reconcile planned versus actual methods at launch and completion. Each entry records implementation, validation, execution, device, decisions/updates, checkpoint/export, evaluation and local retrieval. States: planned, implemented, validated, running, trained, evaluated, retrieved, partial_budget, blocked_input, blocked_labels, failed_validation, failed_runtime. Meaningful updates plus a validated artifact are required for trained. A registry entry, fixture pass or Kaggle COMPLETE is not full-suite completion or convergence.

Never silently drop methods, reduce context, fabricate labels, change objectives or expand resource ceilings. Preserve isolated failures and continue independent eligible work. Full scientific completion requires valid supervision and adequate training/evaluation; a deadline-bounded attempt is not equivalent.

### 20.2 Decisions, episodes and training caps

One RL training decision is one legal selection from the nonempty current ready queue when the decoder is free. Busy intervals, empty-queue advances, Qwen calls, validation selections and replay updates do not count.

The cap is20,000 training action decisions PER RL arm PER study seed. It is not20,000 epochs or messages in one queue. An episode selecting100 jobs contributes100 decisions; roughly200 such episodes yield20,000. Actual counts depend on horizons/unfinished work. PPO's four environments share the aggregate cap and256-transition rollout count.

Validate every2,000 training decisions and on the final supported checkpoint. Warmstarted PPO has the same RL cap and separately reported supervised pretraining. Retain seeds1729/2718/31415 and supervised max50epochs/patience8. First-seed breadth precedes additional seeds/sweeps. Resume interrupted training from full state, not newly initialized weights.

### 20.3 Delay definitions and learning

a_i=complete receiver arrival; b_i=dispatch/start; f_i=finish.
W_i=b_i-a_i is QUEUE WAIT, the main operational concern.
D_i=f_i-b_i is processing duration.
R_i=f_i-a_i=W_i+D_i is receiver turnaround.
Capture-to-finish is separate and requires usable capture metadata.

These are simulator-clock measurements, not VLM estimates. Primary d0=1 is synthetic service; variable-service stress uses identity-qualified durations shared across policies. Synthetic units are never milliseconds. Measure actual context-building/selection ms separately. Offline Qwen cost is never subtracted from deadlines. Charge measured selector overhead only through an explicitly measured service-scale sensitivity.

Reference-tier budgets remain Critical2/High4/Medium8/Low16 d0. They are declared experimental waiting budgets, not medically validated deadlines. Predicted SPI remains an input, not reference truth. Retain fixed budget-multiplier sensitivities0.5 and2.

Section4 remains exact:
ell(W,delta)=W/delta+max(0,W/delta-1).
At W=4, Critical costs3 and Low costs0.25.
Cost starts at arrival and freezes rather than disappears at dispatch; never-selected jobs accrue through T. Service after dispatch adds no queue-wait cost. Exceptions consume their attempt; no retry.

D target v_i=[ell(W_i+d0,delta_i)-ell(W_i,delta_i)]/ymax.
D reward r_k=-[L_D(t_next)-L_D(t_k)]/100, gamma1.
J_D=L_D(T)/N_assessed, lower is better.
Include mid-service arrivals, idle and terminal intervals. Report tier waits, misses, censoring and unselected counts; do not optimize delivered-only unweighted averages or promise unconditional class-mean ordering.

P learns qualified pair preferences without secretly mixing D losses. RLAIF uses Section9's integrated learned reward. VLM replay changes sampling while retaining D reward. The same delay metrics may evaluate every method where references exist, without requiring identical training supervision.

### 20.4 Complete runner, repairs and device paths

Extend the canonical runner with objective/device/resume/seed/decision-budget/deadline. Keep one source of truth and stable Kaggle identity. Persist atomic progress/checkpoints, resolved configuration and streamed notebook/persistent logs.

Repair LambdaMART by reproducing the saved mismatch and comparing trees, feature precision, thresholds, missing/default directions and aggregation. Save the native booster before parity checking. Diagnose before patching; do not weaken tolerance/change labels to hide a bug.

Implement real RL resume: networks/target/critic, optimizers, replay/priorities/probabilities, schedules, RNGs, episode/action position and best validated state. Learned reward requires immutable event/grid reconstruction, frozen reward blocks and full replay relabeling after updates.

Propagate device through models, tensors, losses, classifier weights, restored optimizer and CUDA RNG. Verify placement. CPU handles simulation/baselines/trees. Benchmark representative neural training updates end-to-end on CPU/GPU before placement. GPU metadata alone does not establish CUDA execution.

If measured per-snapshot loops bottleneck, use length-bucketed masked padded batches. Padding cannot affect pooling/attention/actions/losses/bootstrap. Never truncate legal candidates to fit timing. Retain NumPy deployment and native/export score/selection parity including tree thresholds and GRU/empty histories.

### 20.5 Data, streaming and all retained methods

Reuse6000 cached image scene/SPI outputs and all93 logical channels. Preserve compact transport and Section18 field-selection research. No SPI retraining or cached detector reruns. Complete missing compatible image embeddings only as needed and fifteen pinned mission embeddings. Relevance/novelty are causal mission/stream values; query KB before commit, never invent static scores.

Use64 training traces/seed cycling offered counts32/96/256/384/512; retain Section13's15 validation population/load cases and message counts. Retain50images/UE scaling for50/60/100 UEs. Implement join/leave/pause/resume/session resets and fifteen mission profiles. Received jobs survive departure; identities/device types do not grant priority. Capture/readiness/arrival/dispatch/finish remain distinct.

All-success is primary; identity-seeded exception stresses share outcomes/durations/arrivals across policies without retries. Current data support mixed IID and exact copies, not real video chronology. Implement the sequence adapter and use accessible verified clips within the source cap. Missing qualified clips/object annotations make those DATA arms unavailable, not evidence that shuffled photos are real sequences. Temporal architecture/synthetic-history tests still run with honest scope.

Keep parent/duplicate/sequence groups isolated and exposure disclosed. New queue seeds do not refresh final independence. Unknown references stay unknown and cannot be guessed Low for D.

The runner must include:
linear_choice, mlp_choice, delay_huber, ranknet, lambdamart,
priority_classifier, deepsets_ranknet, set_transformer_ranknet, listmle,
temporal_deepsets_ranknet, masked_ppo, masked_double_dqn, td_per,
ranknet_warmstart_ppo, rl_vlm_f, vlm_replay, visionprefer.

Teacher: Section9's revised Qwen2.5-VL, unchanged216-call allocation/gates/roles. Match visible teacher/student PAIR_VIEW context and preflight maximum context. P linear/MLP/compatible rankers share accepted preferences; VisionPrefer reuses aspect fields. LambdaMART/ListMLE require supported grades/orders, not fabricated complete lists. Classifier requires four-tier support. RLAIF requires qualified prefix/reward validation; missing replay guidance is neutral.

Prepare P/teacher work alongside CPU tasks and independently supported RL. Reserve budget for teacher-dependent first attempts before extra seeds consume it. Failed qualification is a demonstrated blocker, not an excuse to skip implementation/collection attempts or claim success.

### 20.6 Four-hour execution and resources

The continuation question had no answer: default hard stop. Resolve one absolute execution deadline, using any earlier user deadline or otherwise execution-start+4h. Never reset it on relaunch. If a prior deadline has expired, disclose that rather than silently extending. Documentation finalization is not a new training run.

Keep six aggregate GPU-hours including loading and idle allocation. Two T4s consume this in three wall-clock hours; their VRAM is not a shared pool. Verify explicit teacher/learner devices and isolated dependency processes. Quota does not guarantee allocation.

Checkpoints, not timing promises:
- First30min: freeze source, reproduce failures, validate contracts and measure feasibility.
- By60min: submit complete instrumented runner if critical gates pass; expose blockers immediately.
- Remaining window: first-seed breadth, bounded teacher rounds, then extra seeds/targeted ablations.
- Final20min: avoid expensive new work; checkpoint, retrieve, verify and report.
- Deadline or GPU ceiling: stop safely, retain partial state.

After preflight reserve measured teacher/retrieval overhead and equally divide first-pass training time across remaining eligible method/study entries. Require a meaningful update/checkpoint/validation slice. After all eligible entries have had an attempt, refill incomplete entries in manifest order, then additional seeds. If a minimum slice cannot fit, record not_started/partial_budget, never claim all methods ran. No extra seed consumes another method's reserved first attempt.

Use finite monitoring and bounded diagnosed retries; no blind relaunches, new notebook identity per run, account switches or unauthorized resource expansion.

### 20.7 Acceptance and artifacts

Verify lifecycle, mid-service arrivals, no retry, independent D/reward arithmetic, group/label/ID/future isolation, missingness/cold-start, permutation/source-renaming invariance, teacher/student context matching, portable/CPU/GPU parity, meaningful updates/real resume, legal masks, actual-mixture PER correction, reward-grid partition consistency, replay relabeling and failure preservation.

Compare all baselines under identical offered traces/information. Report tier waits/misses/censoring, J_D, supported graded nDCG1/3/5, silver pair agreement separately, distinct supported evidence/observations, repeats and population/load/source/mission strata. Preserve unknown/undefined counts. No nDCG from disconnected pairs or natural asset matching from cosine alone.

Profile context+selection ms separately from forward-only with hardware/precision/batch/thread/warmup/counts. Report memory, parameters and supported FLOPs at2ops/MAC; unavailable teacher/quantized FLOPs are not zero. Retain scalar-only/scene-added/temporal-added and transport/subset comparisons. Deadline-limited work stays visibly partial rather than disappearing.

Persist one run's manifests/config/environment/code/data/split hashes, stage/cell logs, raw teacher outputs, checkpoints/exports, per-decision/trace metrics, figures, validation and REPORT. Update parameters/KAGGLE_RUNS.md. Verify local retrieval checksums; distinguish expected and found outputs.

Current evidence: CPU supervised v2 is retrieved with27 trained-validation entries (nine architectures x three seeds), while LambdaMART failed parity on allthree. That runner did not execute RL or qualified contextual teacher training. Temporal D is not qualified P or natural sequence validation. Its original final test remains unopened.

This revision changes documentation only. No production code, parameter producer, model weights, training or Kaggle job is changed. Later implementation must satisfy the full contract rather than silently reducing it.
