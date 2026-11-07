# RL and VLM-assisted training for online disaster decoder selection

25 October 2026. Focused primary-source review, proposed research design and
small mathematical reference checks. No RL training, new VLM annotation, GPU
experiment or asynchronous runtime integration was executed for this report.
This is an RL research extension to the canonical build record, not a replacement
implementation plan or a claim that RL has passed its existing evidence gate.

## Recommendation and scope

Use an event-driven, nonpreemptive, single-decoder environment. At each free-
decoder event choose one complete ready message, continue admitting arrivals
during service, and reconsider the updated candidate set after completion.
The deployed actor receives parameters/masks and permitted causal history only.
Images, annotations, fact identities and future arrivals do not enter the actor.

First establish whether decisions have useful sequential consequences beyond a
strong contextual greedy policy. Train a shared candidate-scoring Masked PPO
baseline if this gate passes. For the most informative VLM-in-loop experiment,
compare a replay-based Masked Double DQN with uniform replay, TD-prioritized replay
and mixed VLM-guided replay, using exactly the same annotation-derived reward.
Keep VLM-learned reward and supervised choice warmstart as separate comparisons.

This choice separates three questions: does sequential learning help; does the
VLM supply useful training information; and is changing the reward necessary?
No algorithm is a measured winner for this project. A recent publication is not
automatically a better algorithm for the actual action space and information.

Only disaster/emergency image evidence is in the current application scope.
Begin with aerial disaster assessment supported by existing labels and models.
Rescue urgency and life-safety value are not inferred from building damage alone.

## What would make RL useful?

A model choosing the item with highest immediate importance is a contextual
selector. If nothing about a choice changes subsequent utility, a supervised
scorer or contextual bandit may suffice. RL becomes relevant when a choice
changes the evidence history, consumes variable decoder time, affects deadline
opportunities, or interacts with later arrivals and repeated evidence.

Dynamic arrivals alone do not establish that RL will improve decisions. The
policy cannot know an unannounced future arrival. It can learn workload
statistics from training, but these assumptions must be tested under shift.

Distinguish physical environment state from actor observation: the header is a
compressed observation of image content, and a bounded history need not be a
sufficient statistic. This is naturally partially observed and semi-Markov in
elapsed service time. Do not assert a fully observed MDP merely by naming the
parameter vector state. A recurrent encoder is conditional on evidence that
recoverable history matters; recurrence cannot restore information never sent.

## Inspected literature and transferable mechanisms

This is a targeted survey through 25 October 2026, not an exhaustive newest-paper
or novelty certification. Performance figures in other environments do not
predict decoder-scheduler gains. The following records identify actual methods
and relevant limits rather than combining reported headline scores.

### Decima, SIGCOMM 2019

[Learning Scheduling Algorithms for Data Processing Clusters](https://web.mit.edu/decima/content/sigcomm-2019.pdf).
Inspected scheduling formulation, event-time reward and arrival-variance sections.
Learns graph-based scheduling with stochastic arrivals; reward accounts for
elapsed wall-clock time and active jobs. Transfer event-driven training and
paired workload control, not executor allocation or its completion-time objective.
Our messages have no job-dependency DAG that would alone justify its graph model.

### Learning to Dispatch, NeurIPS 2020

[Original paper](https://arxiv.org/abs/2010.12367).
Inspected Sections 4.1–4.3, reward telescoping and candidate-policy architecture.
Uses graph state embeddings and PPO to select eligible operations with a
size-agnostic policy. Its reward differences telescope to negative makespan.
Useful precedent for candidate selection and objective-consistent rewards; job-
shop precedence and machine allocation are absent from our single-decoder task.

### Semantic-aware Transmission Scheduling, 2023

[Monotonicity-driven DRL](https://arxiv.org/abs/2305.13706).
Previously inspected system model, MDP reward and critic objective. Known remote-
estimation dynamics give an AoI-dependent error cost; DDPG uses structural
monotonicity in critic learning. It supplies no universal image-importance label.
Do not enforce monotonicity in every proposed semantic field without justification.

### RL-VLM-F, ICML 2024

[Original methods](https://arxiv.org/html/2402.03681v4),
[venue](https://proceedings.mlr.press/v235/wang24bn.html).
Inspected Sections 3–5, 6.3, appendices and reference code. VLM preferences over
image pairs train a Bradley–Terry visual reward model; SAC learns from that
reward and replay rewards are recomputed after reward updates. Published image
query sampling is random. Actor numerical state and reward RGB inputs differ,
so privileged visual supervision is already established. Teacher mistakes remain.
Replay relabeling is reward recomputation, not correction of teacher annotations.

### FuRL, ICML 2024

[Original methods](https://arxiv.org/html/2406.00645v2).
Inspected Sections 3 and 5 and Algorithm 1. Aligns VLM embedding rewards with
sparse success signals; relay RL helps escape misleading visual reward optima.
It does not use generative pairwise annotation. Transfer the need to check
semantic-reward alignment. Later frames in a successful robot trajectory being
better is not a valid general assumption about a disaster-camera sequence.

### ERL-VLM, ICML 2025

[Original methods](https://arxiv.org/html/2506.12822v1),
[venue](https://proceedings.mlr.press/v267/luu25a.html).
Inspected Sections 3–5 and appendices. Uses absolute VLM ratings of images or
trajectory segments; stratified training addresses rating imbalance and an MAE
probability objective reduces noise sensitivity. Reward updates alternate with
policy learning. A useful alternative when pair annotation is costly; robustness
to some noise does not remove systematic teacher bias or establish our urgency.

### Offline-LD, Machine Learning 2025

[Original paper](https://link.springer.com/content/pdf/10.1007/s10994-025-06826-w.pdf),
[code](https://github.com/jesserem/Offline-LD).
Inspected Section 4 and dataset construction. Adds action-masked QR-DQN and
discrete SAC with Conservative Q-Learning to learn from CP-generated schedules.
Useful if only logged transitions are available; CP schedules and makespan
rewards are domain-specific. Logged next-choice labels alone are not a complete
offline RL transition dataset, and a single behavior policy may give poor coverage.

### CoRe, ICML 2026

[Original methods](https://arxiv.org/html/2607.01721v1),
[project](https://core-2026.github.io/).
Inspected Section 3, 4.1 and Appendix B. Combines LLM-refined reward code with a
learned residual from VLM video preferences and frame-importance feedback.
This is close prior art for hybrid formal/visual rewards and temporal feedback.
Its robotics state/goal-image assumptions differ. A weighted base-plus-residual
reward generally changes the objective; it is not policy-invariant shaping.

### VLM-Guided Experience Replay, 2026 preprint

[Revision 2, 27 September 2026](https://arxiv.org/html/2602.01915v2).
Inspected Section 3 and implementation/ablation appendices. An asynchronous VLM
scores past clips; VLM and TD information prioritize replay mixed with uniform
sampling. The task reward is unchanged, and one teacher can serve many simulated
environments. Its VLM arm disables importance correction in reported settings;
our corrected-sampling variant would be an adaptation, not exact reproduction.
This is the closest recent precedent for the proposed training-data-adviser arm.

### Policy-Invariant Reward Shaping from LLM Feedback, August 2026 preprint

[Original paper](https://arxiv.org/abs/2608.18008).
Inspected full text, potential construction and theory/validation scope. Applies
bounded LLM potentials to an augmented MDP; empirical evidence includes small
MDP and MiniGrid pipeline checks with remaining integration limitations.
The underlying shaping principle is established by
[Ng, Harada and Russell, 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
It motivates a removable guidance arm, not claims of better learning or reliable
LLM judgments. Our finite-horizon conditions are specified separately below.

### EARS, September 2026 preprint

[Specifying Reward Functions for RL Without Environment Sampling](https://arxiv.org/abs/2609.15544).
Inspected full text, feature construction, imagined trajectories and comparisons.
Uses LLM-mediated reward features and preference learning over imagined feature
trajectories. Useful for clarifying intended tradeoffs before simulation; it
still assumes a preference source and does not verify imagined visual facts.
Parameter redesign is a separate workstream, so do not add this to the first build.

### Foundational learning kernels

[PPO, 2017](https://arxiv.org/pdf/1707.06347) supplies the on-policy clipped
policy-update baseline; it is not an LLM-specific training method. Masking and
candidate-set encoding adapt it to our actions. Distributional QR-DQN is a
conditional extension if return variability matters. Continuous-action SAC/DDPG
should not be copied directly for selecting one discrete message. Offline-LD's
discrete masked SAC is a relevant alternative when that additional complexity
addresses an observed limitation.

## Proposed environment contract

### Decisions and observations

One action selects one complete message from the ready buffer. Mask padded,
incomplete, incompatible and otherwise ineligible candidates. Do not use source
or message IDs as learned semantic features. IDs route actions to messages.
An empty ready buffer advances to the next arrival; there is no learned wait
action in the first work-conserving study. A busy decoder has no selection action.

Observations contain candidate parameters and availability masks, permitted
time/age/deadline metadata, successful-header history summaries and declared
mission/context. Completion failure is observable but never adds delivered
evidence. Future arrivals and predicted service times are not silently supplied
as truth. A cost estimate, if used by a policy, must be available identically to
all relevant baselines and fitted without held-out exposure.

Shared network: encode each candidate with one small MLP; pool candidate
embeddings using mean/max plus count; combine each embedding with pooled queue
and causal history context; output a scalar action logit or Q value. Padding is
masked. This handles variable queue sizes without a class for each message ID.
Do not add graph edges unless relations observable in the header justify them.

The physical event loop owns the receiver buffer. A separate decoder worker
returns completion events while arrivals continue. Training can use many
simulated environments on CPU. Qwen runs in one asynchronous/offline labeling
worker initially. This does not require one Qwen instance per UE.

### Two buffers with different purposes

Receiver buffer: complete messages waiting to be decoded.
Training replay buffer: past tuples (observation, action, reward, next observation,
elapsed duration, terminal flag) used to update an off-policy learner.

VLM-guided experience replay prioritizes the second buffer. It does not let Qwen
look inside queued latents during deployed operation.

## Reward and primary endpoint

For a fixed offered stream and declared observation horizon T, define the usable
fact set and nonnegative mission weights from independent annotations. Let W be
the total offered usable fact weight in that stream. Count a repeated fact once;
a condition change requires its own supported fact identity. Let D(t) contain
facts exposed by successful completions by elapsed time t.

    C(t) = sum_{f in D(t)} w_f / W
    J_time = (1/T) integral_0^T C(t) dt

This measures how early distinct useful evidence appears. W is fixed across
policies; its use by the training evaluator does not put future labels into actor
observations. Zero-W traces are undefined for normalized coverage, counted
separately, and do not silently become perfect or zero-score observations.

At decoder-free decision time t_k, advance through service to t_{k+1}, clipped at
T, while admitting arrivals. Attribute the area for that interval to its action:

    r_k = (1/T) integral_{t_k}^{t_{k+1}} C(t) dt

With successful evidence credited only at completion and no intervening
completion, this is C(t_k) * (t_{k+1}-t_k)/T. Changes at a single endpoint have
zero measure. Include the terminal and any idle tail intervals so all episode
rewards sum to J_time. Idle periods have no chosen action but their reward must
not disappear. Use gamma=1 for this finite-horizon objective; include remaining
time in observations. Constant per-decision discounting would introduce a
different preference over action count rather than elapsed time.

An equivalent useful check is
J_time = sum_{f delivered} w_f * (T - first_delivery_time_f)_+ / (T*W).
Training labels may describe ideal source facts first. Actual reconstructed
evidence is a separate outcome, using independently supported or explicitly
model-assessed reconstruction facts. Never call a successful decode proof that
every annotated fact survived compression.

For a static equal-opportunity control, retain A_K=(1/K)sum_{k=1}^K C_k, with
reward C_k/K when transition k completes. Unweighted incremental coverage sums
to final coverage and loses the early-delivery objective. Do not mix opportunity
and elapsed-time experiments into one headline metric.

Do not add throughput, fairness, arbitrary decay or teacher-confidence bonuses
to the primary reward merely because they are measurable. Declare meaningful
constraints or separate outcomes where required. No case-specific urgency
weights should be invented from image appearance by the VLM.

## Where Qwen can enter training

### A. Choice demonstrations

Qwen sees a mission, current eligible images and causal past evidence; it
proposes a next choice. Save strict preference, ties, uncertainty and factual
evidence separately. Train a small actor with masked choice cross-entropy, then
optionally warmstart RL. This is behavior cloning/distillation, not RL by itself.
Use annotation-derived demonstrations when computable. The same warmstart must
be provided to ablations claiming a benefit from later RL guidance.

### B. Learned preference reward, a competing weak-supervision arm

Qwen compares two alternative outcome prefixes from the same initial state,
mission and equal elapsed-time endpoint. Show per-image completion times; a
longer contact sheet must not be implicitly preferred for containing more images.
Train a reward model with preference cross-entropy, then optimize it using RL.
This follows RL-VLM-F/CoRe principles. It changes the target to learned teacher
preference unless anchored by independently defined outcomes. Keep true task
returns and predicted rewards separate to detect reward exploitation.

Freeze the learned reward during each rollout/update block. If it is changed in
off-policy learning, recompute stored rewards consistently; if using on-policy
PPO, avoid mixing changed rewards with stale advantages. Teacher labels themselves
are not corrected merely by recomputing rewards. Reward weighting does not have
a universally justified gold/silver mixing ratio; select inside development data.

### C. VLM-guided replay, the preferred recent mechanism to investigate

Train Masked Double DQN on the fixed task reward. Asynchronously ask Qwen about
past decision clips to prioritize semantically consequential experiences. Keep
original rewards unchanged. Mix uniform and VLM-prioritized sampling so falsely
low-scored experiences still get revisited. Compare to TD-error PER first.

One explicit adaptation is:

    q(i) = (1-rho)/N + rho*q_vlm(i), 0 <= rho < 1
    importance_weight(i) = (N*q(i))^(-beta)

Use the actual mixture probability in any correction, not the prioritized branch
alone. A bounded rho and positive floors retain sampling support; they do not
prove finite-sample improvement or remove function-approximation bias. Calibrate
rho/beta within training/validation roles. If Qwen times out or rejects a case,
keep provisional/uniform replay and never block inference or simulator progress.

This uses an off-policy replay learner. Ordinary PPO should not be trained from
arbitrary stale replay transitions. Vectorized PPO rollouts and asynchronous
teacher work can coexist, but that is a different mechanism from replay reuse.

### D. Potential-based shaping, a conditional alternative

For a frozen bounded progress potential Phi, an exactly finite terminated
episode with gamma=1, and Phi(terminal)=0, use:

    r'_k = r_k + eta*(Phi(s_{k+1}) - Phi(s_k))
    sum_k r'_k = sum_k r_k - eta*Phi(s_0)

The added return is action-independent for a fixed initial state. Therefore this
construction preserves episode return ordering, even if the potential is wrong.
Zero terminal potential is required at the declared horizon as well as natural
termination. Do not clip only shaped rewards, change discount convention, update
the potential inconsistently mid-episode, or omit final boundary contributions.
If the potential changes between phases, handle rollout/replay versioning.

This does not verify the base objective, guarantee useful guidance, guarantee
convergence or fix missing actor information. An exactly transformed critic can
also leave advantages unchanged; practical gains depend on approximation and
optimization and may be absent with an already dense coverage reward. Test
random, reversed and poor potentials, and remove this arm unless it helps under
matched compute. Prefer it to unrestricted additive teacher reward only when
preserving the existing objective is the intended mechanism.

## Proposed experiment sequence and stop rules

1. Validate causal events and rewards with hand traces, failures, idle intervals,
   arrivals during service and horizon clipping. Existing lifecycle ownership
   remains the integration boundary; do not imply production thread safety.
2. Measure immediate-versus-planning headroom on small annotated multi-fact traces.
   Compare same-information contextual greedy and short lookahead. Exact
   fact/future-aware optimization is a privileged diagnostic upper reference,
   not a fair deployed baseline. If there is no useful headroom, retain the
   supervised selector rather than force RL into the project.
3. Train supervised linear/small-MLP choice models and Masked PPO on the same
   observation contract. Include an identical architecture's one-step/short-
   horizon control to separate context encoding from long-term RL learning.
4. Train Masked Double DQN with uniform and TD-prioritized replay. Start with
   at most 100,000 action decisions per RL feasibility arm as a workload cap,
   not a power calculation or a promise of adequate convergence. Record actual
   training time and learning curves before extending budgets.
5. Run a bounded teacher preflight, initially up to 200 decision clips/pairs plus
   reversed-order checks when relevant. Inspect parsing, availability, latency,
   factual errors on annotation-supported cases and group diversity. This is
   a feasibility cap, not a claimed statistical sample requirement.
6. Add mixed VLM-guided replay only after the uniform/PER learners work. Keep
   rewards fixed and compare teacher budgets including a zero-call arm. Learned
   teacher reward and BC warmstart are separate controlled alternatives. Do not
   simultaneously introduce every reward/curriculum/architecture mechanism.
7. Test teacher-error stress, header bottlenecks, new events and service-time
   shifts. Lock final selection before new independent groups are opened.
8. Demonstrate real payload decoding and measured end-to-end timing on a bounded
   replay. Cache policy-independent received latents/channel noise. Training can
   use profiled service distributions first, clearly labeled simulated timing.

Stop escalation for invalid facts, insufficient independent groups, inadequate
observations, teacher cost without useful benefit, or no improvement over the
same-information supervised/contextual policy. No healthy expensive run needs
interruption to retrofit instrumentation; instrument before launch.

## Comparators and ablations

### Essential comparisons

- FIFO, RR, seeded random; single permitted semantic fields and the existing
  direct-I reference; SJF as a cost-oriented diagnostic only.
- Strong supervised/contextual greedy with the same observable inputs.
- Masked PPO using independent task reward.
- Masked Double DQN: uniform replay, TD-PER, mixed VLM replay.
- Conditional BC-warmstart plus RL; compare against BC-only with the same labels.
- Conditional preference-reward-only RL inspired by RL-VLM-F; optional hybrid
  additive reward or PBRS as separate mechanisms, not a combined default.

Same actor observation and history rules apply throughout. Equal model width
does not make on/off-policy training cost equal: report both interaction-matched
and compute-matched comparisons, including teacher inference. Tune each method
fairly within development groups; preserve unsuccessful configurations.

### Mechanism and robustness tests

- No backlog: all work-conserving policies select the sole ready item.
- History disabled, candidate order permuted, fields/masks ablated jointly.
- Exact repeats versus new supported facts versus changed conditions.
- Random teacher, reversed teacher priorities and shuffled semantic information.
- Injected independent and class/systematic teacher errors separately. Proposed
  initial flip levels are 0, 10 and 30 percent for stress testing, not measured
  Qwen error rates; preserve natural model errors as their own experiment.
- Queue bursts and UE arrival imbalance; service-time variability; heavy load;
  header missingness; annotation uncertainty; new events and camera domains.
- Uniform versus disagreement-selected teacher queries under the same call
  budget; known method ingredients are not themselves a novelty claim.
- Compare source-ideal supervision with available reconstruction-assessed cases.
- Teacher sees pixels, actor sees headers: quantify identical-observation target
  conflicts before interpreting more training as an information remedy.

Do not turn all axes into a giant full-factorial sweep. Start with the mechanism
comparison; add one targeted stress test per unresolved failure explanation.

## Metrics and definitions

### Semantic outcomes

Primary: J_time above, on the same offered trace and fixed observation horizon.
Report coverage at declared times and within-source critical-fact recall where
criticality is supported independently. Repeated evidence gets no additional
credit merely because it occupies another image or another UE's message.

Define useful exposure delay for a fact f as first successful delivery time
minus first eligible complete-message arrival that could expose f. This separates
scheduler opportunity from unknown capture/network delays. Report capture-to-
delivery delay separately only with trustworthy capture times.

For supported deadlines, report timely critical recall:

    sum_f w_f * 1[first_delivery_f <= deadline_f] / sum_f w_f

Deadline convention (capture-relative, arrival-relative or absolute) must be
declared. Never fabricate clinical/rescue deadlines from image appearance.
For undelivered facts report misses/censoring; delivered-only p95 delay is biased
and must not be presented without the accompanying delivery fraction.

Weighted redundant selection rate: duplicate fact-weight exposure divided by
total selected annotated fact weight, where a duplicate means already exposed
at that completion. Record unknown/unassessable selected evidence separately.
Report useful evidence deferred beyond horizon, decoder failures and unfinished
jobs. Zero-denominator metrics remain undefined with explicit counts.

### Decision and teacher metrics

Next-choice regret relative to an annotation-supported immediate-gain reference
is diagnostic, not a substitute for sequential return. Teacher agreement on
independent strict pairs is correct comparable choices / assessable strict pairs;
also report ties, abstentions, malformed outputs and unsupported factual claims.
Calibrate confidence against independent targets before interpreting it as error
probability. Agreement between models or reversed presentations is consistency,
not verified truth. Reward-model pair accuracy measures imitation quality.

Reward exploitation gap: predicted return rises while independent task return
does not. Plot these separately instead of subtracting incompatible scales.
Measure missingness/calibration by source and task; rare critical cases may be
hidden by overall accuracy. Teacher rationales support audit, not automatic proof.

### Learning efficiency

Plot held-out-development J_time versus action decisions and versus total elapsed
training time, including Qwen time. Record time/steps to a predeclared useful
target; unreached targets are censored/failures, not silently omitted. Report
learning-curve area on a fixed budget, final locked performance, seed variability,
teacher calls/images/tokens, accepted-label yield and unique source groups.
Repeated states/pairs are not independent data collection.

### Systems cost

Policy inference p50/p95 by queue size; decode p50/p95 by codec/input profile;
teacher latency and backlog; feature extraction/preparation; received-to-output
and capture-to-output delay where defined; peak RAM/VRAM and model artifact size.
Separate cold start from warm inference, batch size and GPU synchronization.
If teacher and decoder share a device, measure actual contention rather than
claiming asynchronous execution removes the hardware cost. Teacher calls are
zero on the deployed path for the proposed design.

Offered load lambda*E[service] matters for whether an unbounded queue grows even
when throughput is not the objective. Use the same admission/capacity/failure
rules for all policies, and do not drop difficult jobs only for one method.
Report starvation and per-UE delay as diagnostics; equal source service does not
by itself establish equal semantic value.

### Scope and uncertainty

Split parent/event/site/duplicate groups before extraction adaptation, queue
generation and model selection. Protect final events. Pair policies on identical
offered traces and received payloads. Bootstrap independent groups rather than
counting thousands of queue seeds as thousands of independent disasters.
Report paired effects, intervals and per-event failures. Training seeds assess
optimization variability, not domain replication. Preserve parameter-selection
boundaries from the main plan and report development-exposed data as such.

PSNR/SSIM/LPIPS describe codec reconstruction, not scheduling correctness. Keep
codec/SNR/rate fixed during the first scheduler comparison. We make no new
lossless, compression, radio-throughput or lives-saved claim from this study.

## A defensible contribution hypothesis

Working question: Can training-only VLM guidance improve sample efficiency and
early semantic evidence delivery under limited headers, causal novelty/history
and variable decoder durations, without changing the declared task reward or
adding deployed VLM latency?

The known loop, masked actions, replay prioritization, active preferences, PBRS
and supervised warmstart are not new inventions. A useful contribution could
be a demonstrated mechanism and failure map for this receiver setting: when
semantic replay helps versus TD-PER, how teacher errors affect it, and where
header aliasing makes a stronger model ineffective. New terminology alone is
not novelty. Closest broader scheduling/active-RL work remains a coverage gap;
no journal novelty claim is established by this targeted review.

## Mathematical check executed

Artifact: [JSON reference receipt](../../results/decoder_context/research_checks/20261025T143345Z/rl_reward_reference.json).

Six permutations of three equal one-second jobs with fact sets:
A={1,2,3,4,5,6}, B={1,2,3,7,8}, C={4,5,6,9,10}, T=3 seconds.
Immediate maximum-gain A-first gives J_time=0.4666666667. B,C,A or C,B,A gives
0.5. This constructed full-fact example proves that myopic gain can miss the
time-coverage optimum; it does not prove the headers reveal the required overlap.

For all six permutations, three frozen potential settings (zero, misleading,
large incorrect) with zero terminal potential satisfied the telescoping identity
and preserved the maximizing sequence set within floating-point tolerance.
These are numerical identity checks, not RL performance or learning-speed tests.

## Next deliverable if implementation proceeds

Build one event simulator adapter and one portable candidate-scoring observation
contract, validate the reward trace, then implement the supervised/PPO comparison.
Proceed to Double DQN/PER/VLM replay only after the base learners are sound.
Teacher deployment reuses an explicit registered Qwen backend; one worker starts
the pilot, and no four-model replication is assumed necessary. All new training
runs must record source hashes, data groups, reward/teacher versions, environment,
hardware, decisions, metrics and failures in separate run directories. Preserve
the existing controller, codec, checkpoints and historical results.
