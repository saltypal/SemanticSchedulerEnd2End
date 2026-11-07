# Master plan: contextual semantic decoder scheduling

30 October 2026. Canonical plan for SemanticSchedulerEnd2End. This consolidates the complete contextual project and its smaller first implementation. Earlier versions are preserved under experiments/semantic_priority_design/runs/consolidated-master-20261030-042328/. This document changes the design record; it does not claim new training or implementation.

## 1. Final direction and research question

Build a small contextual neural ranker that selects one complete, currently available message for one nonpreemptive decoder. It learns from SPI, UE-mission relevance, transmitter novelty and spatial evidence together, using permitted queue context and processed-header history.

The objective is early service of important, mission-useful information. Assess both:

1. How long independently important images wait.
2. How long distinct useful evidence waits before receiving service.

The main candidate is **Deep Sets contextual scoring trained with RankNet preferences**. A shared 64/32 MLP is the simple baseline and first integration step. RL, RLAIF and VLM supervision remain study arms. They must be compared with a competent contextual ranker.

References establish what matters; elapsed time establishes the consequence of postponing it. Neither predicted SPI nor a delay formula establishes its own semantic correctness.

## 2. Literature foundation: what each source actually supports

| Original source and inspected mechanism | Decision for our study | Transfer limit |
|---|---|---|
| [Semantic-aware Transmission Scheduling, 2023, Sections III–VI](https://arxiv.org/html/2305.13706v2): application cost g(AoI), negative summed cost reward, channel assignment and monotonicity-guided DDPG | Define the application consequence before training RL | Its radio assignment, estimation-error objective and monotonicity theorem do not directly apply to our decoder queue |
| [Semantic Scheduling for LLM Inference, 2025, Sections 3–4](https://arxiv.org/html/2506.12204v1): response-time objective subject to urgency constraints, predicted urgency and remaining computation | Semantic classes and latency can be evaluated together | Its batching/preemption differ; inspected predictor code simulates priorities and does not establish a reproducible real emergency classifier |
| [Efficient LLM Scheduling by Learning to Rank, 2024, Sections 4–5](https://arxiv.org/html/2408.15792v1): measured generated-length labels, ListMLE training and serving-latency evaluation | Keep ranking loss and actual scheduling outcomes distinct; targets need provenance | Its target is job length, not semantic importance; replacing that target requires new supervision |
| [The Age of Incorrect Information](https://arxiv.org/html/2012.13214): source–receiver mismatch combined with duration, modeled state transitions and update constraints | Information content and elapsed time belong in an application-dependent cost | No known Markov mismatch process or receiver correctness oracle exists in our image queue |
| [Approximating Min-sum Set Cover, 2003 report, Sections 1–2](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2003-21.pdf): minimize summed first-cover times; equivalently sum remaining uncovered elements over steps | Give distinct information a first-service delay; repeated information earns no repeated coverage | Static unit-cost theorem; no inherited guarantee for our live arrivals, partial observations or arbitrary weights |
| [Novelty and Diversity in IR Evaluation, 2008, Sections 4–6](https://cormack.uwaterloo.ca/novelty.pdf): information nuggets and redundancy-sensitive alpha-nDCG | Plain nDCG cannot establish novelty benefit; evidence units need independent support | Retrieval nuggets are an analogy, not automatically physical-building identities |
| [RankNet, 2005](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf): shared scalar scorer and pairwise probabilistic loss | Learn relative priority without inventing an exact correct delay for every image | Ordinal scores are not calibrated harm |
| [Deep Sets, 2017, Sections 2–3](https://arxiv.org/html/1703.06114v3): shared encoding and invariant pooling | Variable-size queue/history representation independent of array order | Pooling may miss distinctions; compare against simple summaries |
| [Set Transformer, 2019](https://proceedings.mlr.press/v97/lee19d.html): attention between set elements | Strong interaction-based architecture comparison | Additional cost needs measured benefit |
| [RankGPT, 2023, Section 4.2](https://arxiv.org/html/2304.09542v3): teacher permutations distilled with RankNet | Offline teacher preferences can train a fast student | Our image/mission application is an adaptation of passage ranking |
| [RL-VLM-F, 2024, Sections 4–6](https://arxiv.org/html/2402.03681v3): visual preferences train reward, then preference-based RL | Retain a separate RLAIF arm | Its task-progress setting differs; VLM errors remain possible |
| [PEBBLE, 2021](https://proceedings.mlr.press/v139/lee21i.html): preference reward learning and replay relabeling | Relabel replay when the reward model changes | Stale rewards invalidate the intended continuing experiment |
| [B-Pref, 2021](https://arxiv.org/abs/2111.03026): controlled teacher errors and independent task evaluation | Test noise, bias and uncertainty; separate teacher agreement from success | Injected errors do not estimate Qwen's actual error rate |

This evidence supports the components and formulation. It does not establish novelty, superiority, universal emergency weights or realistic simulation by itself. The minimum-latency connection is a testable adaptation, not a new theorem.

## 3. End-to-end system

~~~mermaid
flowchart TD
 A[Ordered UE observations and mission paragraphs] --> B[Existing extractor and causal sender KB]
 B --> C[Complete parameter record plus opaque payload]
 C --> D[Receiver buffer]
 D --> E[Immutable ready-set snapshot]
 H[Processed-header history] --> E
 E --> F[Small contextual scorer selects one message]
 F --> G[One nonpreemptive processing attempt]
 G --> H
 G --> D
 R[Independent references and frozen offered trace] --> V[Delay, evidence, ranking and runtime evaluator]
 F --> V
 G --> V
 T[Offline Qwen: images, missions, causal context] --> L[Versioned preferences or reward model]
 L --> M[Training]
 M --> F
~~~

Arrivals continue during decoding. A free decoder selects exactly one ready job. The scheduler need not maintain a permanent queue order. No grants, preemption, learned WAIT, automatic retries or retransmissions are introduced.

The main experiment assumes complete messages arrive and service completes normally. Equal synthetic durations isolate selection. Bad reconstruction stays bad; it does not imply a detectable failure. Optional exceptions and variable service are separate stress tests.

**Processed-header history is retained.** It means these messages finished processing, not that their reconstructed contents were independently verified. Evaluator facts never enter runtime history. Preserve historical successful-history APIs with a versioned adapter; do not rewrite old experiment semantics.

Actual codec work remains paused until the user supplies the digital-payload codec.

## 4. Runtime information and upstream ownership

Freeze the actual parameters/ export, producer settings and schema. This work consumes SPI; it does not implement or retrain the SPI classifier.

| Family | Meaning | Restriction |
|---|---|---|
| SPI and supplied status/uncertainty | Predicted intervention need | Soft evidence; no hard predicted-class gate |
| Context relevance | Alignment with UE mission | High relevance to routine acquisition is not automatically high receiver urgency |
| Sender novelty | Difference from causal sender KB | Not proof of receiver-wide novelty |
| Spatial evidence | Supported detection counts/extent/diversity | Not physical distance, damage or urgency |
| Masks/conflicts/cold start | Evidence availability | Zero/false/missing/cold start remain distinct |
| Trusted timing and mission metadata | Causal context | No invented real timestamps or mission meaning from IDs |

The active spatial note exports global retained detections, distinct categories, largest-box fraction and union coverage. Do not silently substitute historical mission-filtered definitions.

**H0-only decision, 30 October:** the user rejected the proposed H1 extension. Use the current SPI, relevance, transmitter novelty and four global scene measurements with their masks; do not add category tokens, object boxes, scene graphs, images or raw embeddings. The earlier [scene-evidence literature review](../../experiments/semantic_priority_design/runs/scene-evidence-review-20261030/RESEARCH.md) remains research history, not authorization to change the parameter contract. Learn nonlinear interactions among the existing fields with contextual Deep Sets RankNet. In particular, neither low relevance nor high object count is a hard priority rule. Processed-header history supplies permitted statistical context, not verified scene identities or decoded facts. H0's loss of object types is a known information limit to report, not a reason to abandon joint learned scoring or silently extend the inputs.

Initial receiver purpose: building-damage situational awareness. UE paragraphs condition upstream relevance and offline teacher judgments. Any runtime mission identifier must follow an explicit supplied registry; a relevance scalar does not establish arbitrary unseen-language mission understanding.

Exclude images, raw embeddings, payload contents, filenames, hashes, source IDs, reference priorities/facts, future arrivals and hidden outcomes from learned features. Routing IDs remain outside scoring for dispatch and duplicate detection.

Start with 32 processed headers and report truncation. Compare no-history versus history. Neural encodings of permitted numbers are allowed. Headers that cannot distinguish two important cases impose an information limit; a larger network cannot recover missing facts.

## 5. Ground truth and supervision

Maintain three distinct reference layers:

1. Published object/condition annotations: supported task evidence.
2. Independently assessed priority where image correspondence is valid: class timing.
3. Qualified Qwen preferences/grades: explicitly silver supervision.

No new reviewers are assumed. VLM-generated SPI and preference labels can support distillation; they cannot independently establish emergency correctness.

[TREC-IS](https://www.dcs.gla.ac.uk/~richardm/TREC_IS/2020/participate.html) assesses posts. Audit media linkage and image-supported interpretation before transferring priority to images.

Start with accessible LADI and MEDIC/CrisisMMD records. Use xBD for controlled annotated building overlap if assets are available; RescueNet is an additional building domain. FloodNet structural flooding stays a separate scheduler-unseen-task study. MEDIC/CrisisMMD overlap and xBD/xView2 naming do not create independent domains.

Unknown labels remain unknown, never Normal or zero utility by default. Keep unknown traffic in mixed replay, skip unsupported training comparisons, and report assessable coverage. Reference-cost RL requires complete-reference controlled episodes or a clearly restricted fully assessed cohort.

Group parent images, derivatives, exact/near duplicates and available events/sites before splitting. Exposed evaluation material becomes development data. Prefer out-of-fold SPI predictions for scheduler training; disclose upstream producer exposure separately.

The earlier 240-group 144/48/48 cohort is a feasibility target conditional on access/grouping. Final sample size follows independent-group variability and a useful effect, not the number of generated queues.

## 6. Delay: definitions, objective and interpretation

Record capture h_i, extraction-ready e_i, complete receiver arrival a_i, dispatch b_i and completion c_i.

- Queue wait W_i = b_i-a_i: primary selection timing.
- Service S_i = c_i-b_i.
- Receiver turnaround = c_i-a_i.
- Capture-to-completion additionally includes upstream work.

Dispatch isolates the current selection question. Completion is reported alongside it. First dispatch is called **first service**, not reconstructed evidence availability. A real codec later requires measured completion outcomes. Qwen inference is an offline cost; do not subtract it from invented deadlines.

### 6.1 Reference-class image timing

For each independent priority class report offered/assessed/selected counts, restricted mean wait, selected-within-window fraction, conditional median/tail delay and unselected fraction.

Initially tau=10*d0. For arrivals with full follow-up compute mean min(W_i,tau); unselected jobs contribute tau. Report boundary arrivals separately. This window is an evaluation convention, not an operational deadline.

Do not demand Critical mean < High mean in every trace: a High image may start before a Critical image arrives. Compare policies on identical streams and explain blocking.

### 6.2 Unified reference-unit waiting cost

Define a reference unit u, its release time r_u, nonnegative reference weight w_u, and first-service time t_u >= r_u. Unserved units have t_u=infinity. For fixed horizon T:

L_T = sum_{u:r_u<=T} w_u [min(t_u,T)-r_u]
    = integral_0^T pending_reference_weight(t) dt.

Only released units enter. Report L_T and L_T/sum_{u:r_u<=T}(w_u) in time units when the denominator is positive.

Two explicit configurations make the formulation useful:

| Cost configuration | Unit/release/service | Scientific role |
|---|---|---|
| message_priority | Image / receiver arrival / dispatch | Importance-weighted image waiting control |
| evidence_priority | Supported mission-useful fact-condition / first containing arrival / first containing dispatch | Contextual distinct-evidence delay |

These share an evaluator mechanism but are different tasks. Do not silently switch targets. The full contextual study uses evidence_priority where correspondence is supported; image-priority timing remains a required companion measurement.

For class-cost controls use 8/4/2/1, with 4/3/2/1 and 16/4/2/1 sensitivity. These are study preferences, not medical harm estimates. For building-only labels start with unit-weight useful facts; building damage alone does not invent urgency. RankNet needs preferences, not these cardinal weights.

Count repeated facts once; supported changed conditions are new units. Start with exact-copy correspondence and annotated controlled crops. Similar embeddings alone do not establish natural cross-view building identity.

### 6.3 Why both timing views are required

Three High images A,B,C arrive together. A/B contain the same fact; C contains another. Equal service gives waits 0,1,2 in either A,B,C or A,C,B, so High mean wait remains 1. SPI-only nDCG cannot distinguish them.

Distinct-fact first-service delays are 0 and 2 for A,B,C, versus 0 and 1 for A,C,B. Evidence delay detects additional information earlier without pretending class delay changed.

With fixed known image weights and equal service/full drain, highest-weight-ready-first minimizes weighted image waiting. This is a strong control. The learning problem is inferring useful priority from imperfect parameters.

Overlapping facts make marginal value depend on prior choices. Greedy is a strong baseline, not generally optimal. Hand-worked example: A={1..6}; B={1,2,3,7,8}; C={4,5,6,9,10}. First-dispatch delay sums are 6 for greedy A,B,C and 5 for B,C,A. This illustrates possible sequential headroom; it is not an executed production result.

For fixed all-ready units, integrated uncovered weight and summed first-service delay are equivalent. Old time-average coverage was not mathematically meaningless. Renaming it cannot manufacture improvement. Release times, unknowns and dispatch/completion endpoints must match before comparing versions.

### 6.4 Proposed controlled delay-budget experiment

This is an additional objective configuration for the existing H0 scorer, not a new header, architecture, validated SLA or replacement of evidence_priority. The user proposed offline Qwen-assigned maximum delays. Accept the controlled experiment; do not call arbitrary VLM milliseconds operational ground truth.

Qwen provides an independently judged mission-relative service-urgency/acceptable-deferral tier from the image and permitted causal source context. Do not instruct it to copy the predicted SPI header. Published priority references take precedence where their scope is valid; Qwen-only labels remain silver. Freeze labels before policy comparison and separate training/evaluation groups.

The experimenter maps tiers to declared positive waiting budgets delta_i. An initial pilot convention can be 2/4/8/16 synthetic service slots, with common-scale sensitivity; it is not measured emergency tolerance. Absolute dispatch deadline is a_i + delta_i. These labels stay evaluator/trainer-only: the actor still receives H0 and permitted queue timestamps, not reference budgets or tier labels. An EDF comparator using true budgets is privileged; a fair deployed EDF comparator uses budgets predicted from its permitted inputs.

For a bounded candidate cost use ell(W,delta) = W/delta + max(0, W/delta - 1). This is a proposed engineering convention: waiting always costs something, and its marginal cost doubles after the budget. It avoids treating a message as worthless once it is late. Compare against normalized waiting without the late term. Do not introduce another severity multiplier without a separately justified experiment.

With fixed service d0, a computable immediate target for candidate i is y_i = ell(W_i+d0,delta_i)-ell(W_i,delta_i), the extra cost of deferring it one slot. Train the existing scorer by target regression or RankNet strict preferences derived from y. Label this delay-cost supervision; it is not the old marginal-fact target or a globally optimal action label. The target is exact under the chosen budget convention, while the semantic validity of the budget assignment remains an evidence question. Larger values should score earlier.

For the same-scorer PPO arm define L(t) as the sum of accrued ell costs over arrived messages, freezing each message's accrued cost at dispatch rather than removing it. Reward over an event interval is -(L(t_next)-L(t)). Include arrivals within service, accumulated waiting of unselected jobs, and terminal intervals. Thus undispatched jobs cannot disappear from the loss. Finite-horizon costs do not by themselves prove freedom from starvation in indefinite overload. Preserve one attempt and no quality-driven retries.

Compare the delay-trained ranker, same-input PPO, current preference-trained ranker, predicted-SPI-first, FIFO/RR and same-information predicted-budget EDF/greedy risk. Distinguish objective/supervision comparisons from architecture comparisons. Report per-tier waiting distributions, deadline-miss/within-budget fractions with adequate follow-up, unselected/censored counts, ordinary priority nDCG and supported evidence outcomes separately. Deadlines affect neither receiver arrivals nor actual service durations.

Run a demand/capacity preflight: for example, with unit service only three jobs can start at times 0,1,2; eight simultaneously ready jobs with maximum wait two cannot all meet the budget. Report infeasible/overloaded regimes instead of blaming the learner or guaranteeing compliance. Fifty images per UE is a workload size, not a feasible arrival rate. Qwen latency is offline annotation cost and is not subtracted from a deadline.

Literature inspected for this arm: [DeepRM, 2016, Sections 3.2-3.4](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2020_2021/papers/mao_HOTNETS_2016.pdf) turns elapsed-time system costs into RL rewards; [Decima, 2019, Section 5.3](https://arxiv.org/abs/1810.01963) addresses continuous arrivals, input-driven variance and termination gaming; [Semantic Scheduling for LLM Inference, 2025](https://arxiv.org/abs/2506.12204) evaluates semantic urgency-conditioned latency with priority constraints and preemption. None establishes real per-image deadlines from VLM guesses or superiority of the particular penalty above. The deadline-cost experiment is specified here, not implemented or trained.

## 7. Main model and training

Encode each candidate's permitted numbers/masks with a shared network phi. Pool candidate encodings into queue context q and processed encodings into history h. Mean pooling plus counts and empty-history indicators is the first Deep Sets implementation.

score_i = rho(x_i, phi(x_i), q, h, permitted timing).
Select the largest legal score.

Start with a 32-dimensional element encoding and 64/32 ReLU scoring head. Test permutation equivariance of identity-associated scores. Compare against the same head with simple summaries and against a candidate-only MLP.

The deployed rule is argmax over ready candidates, recomputed whenever the decoder becomes free. Use FIFO/sequence/full identity only to break equal-score ties, not as learned semantic features. No hard SPI gate, relevance threshold, fixed monotonic object-count bonus, or forced nonzero scene weight is imposed. Inputs are fitted/normalized on training groups only; handle count scale and missingness explicitly. Train on mission/context-conditioned accepted preferences, with enough same-SPI and conflicting-feature cases to teach the interactions. Do not generate the target from a hand-weighted H0 formula and claim the model discovered importance. Show scene ablations and matched-information comparisons; heavy use means demonstrated conditional value, not compulsory influence when the scene evidence is uninformative. RL and RLAIF receive the same H0 observation boundary as the ranker.

For preference i over j:
P(i preferred to j)=sigmoid(score_i-score_j).

Use stable binary cross-entropy. Strict comparisons use 1/0; justified ties may use .5; uncertainty is excluded. Normalize by decision and episode; initially sample at most 32 pairs/decision.

Annotation-supported current marginal fact gain supplies controlled preferences. This is myopic supervision, not a globally optimal action label. Qwen preferences supply a separate weak-supervision arm. Include same-SPI and cross-SPI pairs. Spatial extent alone cannot justify urgency.

Initial defaults: Adam 3e-4, up to 64 decision groups/batch, 50 epochs maximum, patience eight. Select on validation rollouts and supported choice quality, not training loss alone.

Generate histories from fixed random/rule mixtures first; later policy-generated histories remain inside training groups. Teacher sees corresponding source images offline, while the student sees only headers. Use the same causal context window and measure representation ambiguity.

## 8. Algorithms retained

| Family | Methods | Supervision/purpose |
|---|---|---|
| Basic | FIFO, RR, seeded random | Arrival/fairness/chance |
| Semantic rules | SPI-FIFO, relevance-only, novelty-only, explicit SPI/novelty rule | Transparent controls |
| Simple learned | Linear choice, MLP choice, useful-next classifier | Choice CE / useful-increment classification |
| Ranking | MLP RankNet, LambdaMART | Preferences / supported graded decision groups |
| Main contextual | Deep Sets RankNet | Same labels with learned queue/history context |
| Attention comparison | Small Set Transformer | Same supervision, stronger interactions |
| RL | Masked Double DQN, TD-PER Double DQN, Masked PPO | Integrated reference cost |
| Hybrid | Ranker-initialized PPO | Initialization benefit |
| RLAIF | RL-VLM-F-inspired reward + DDQN | Preference-learned reward |
| Replay guidance | VLM-guided DDQN replay | Reference reward fixed; sampling changes |
| Aspect supervision | VisionPrefer-inspired scorer | Explicit weak aspect preferences |

Do not manufacture integer grades from arbitrary pairs for LambdaMART. Use supported grades and explicit label_gain. Match label budgets for architecture comparisons and architecture for supervision comparisons.

Use privileged fact-greedy policies as diagnostic baselines and exact tiny-instance optimization as cost lower bounds. Greedy is not a bound on achievable minimum cost. These privileged methods are not deployable same-information competitors. Deployed comparisons receive the same actor observation contract and legal ready set.

## 9. RL and teacher-feedback mechanics

Store immutable current/next candidate sets, snapshot-associated action identity, elapsed duration, reward and terminal masks. No index is assumed to identify the same message in a different queue.

Reference reward is minus the exact interval integral of pending weight. Include mid-service arrivals, idle time and the final censored interval. Discount=1 for the finite episode. Per-decision pending count without elapsed duration is wrong for variable intervals. Independently verify reward sum=-L_T.

Initial RL feasibility cap: 20,000 decisions/arm, validation every 2,000 on fixed traces. One seed establishes feasibility only.

Reuse starting settings: PPO four CPU envs, rollout256, epochs4, batch64, lr3e-4, clip.2, GAE.95, entropy.01, value.5, gradient norm.5. DDQN replay50k, warmup500, batch64, update every4, target sync500, lr3e-4, Huber loss, norm10, epsilon1 to .05 over half budget. PER alpha.6/beta.4 to1.

Use candidate-conditioned networks and masks for variable sets. Historical fixed25-slot/five-UE adapters cannot serve the 100-UE experiment. No silent truncation; mask online argmax and target evaluation.

RLAIF compares outcome prefixes from the same starting situation at the same elapsed endpoint. Qwen sees missions, selected images, timestamps and necessary history, with model predictions/reference labels hidden. A preference reward model uses causal runtime summaries; freeze it during each RL block and relabel all replay after reward updates.

VLM replay retains reference reward and uses q(i)=.5/N+.5*q_guided(i), correcting with actual mixture probability. Unknown guidance is neutral. PPO does not consume stale off-policy replay.

## 10. Qwen qualification

[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) is the initial candidate. Pin model revision, quantization, prompt, processor and environment. Use existing direct Transformers processes; no Ollama service is required.

Teacher input: receiver objective, relevant UE missions, actual RGB candidates, bounded causal history and neutral IDs. Hide SPI predictions/student scores. Output A/B/tie/uncertain, visible reasons and unknowns.

Preflight one/two images and largest planned request; measure latency/memory. Controls: duplicates, clear task contrasts, temporal uncertainty, orientation swaps, unsupported claims and cycles.

Keep the 216-call feasibility cap: 32 controls; up to40 candidate pairs queried both ways; up to40 equal-time prefix pairs both ways; up to24 replay clips. Partition training and held-out validation before collection.

Engineering gates: 95% valid schema; .80 strict-preference balanced accuracy; .75 tie/unanswerable handling on defensible controls. Report denominators/uncertainty. These do not certify broad triage accuracy.

At least16 strict train pairs and8 held-out pairs are preliminary engineering minima, not statistical power. Reward-model agreement must reach .75 and beat constant baselines before learned-reward RL. Failed/insufficient teacher arms remain blocked.

Test 0/10/30% independent and systematic teacher error on controlled labels. Teacher agreement and independent task outcomes remain separate.

## 11. Dynamic simulation and 15 UE templates

Extend one existing event simulator: exact small fixtures, then4/12/32 UEs, then50/60/100. Active UEs, offered messages and backlog are different quantities.

| UE template | Mission description |
|---|---|
| General smartphone | Everyday observations without a dedicated emergency task |
| Citizen reporting phone | Visible incidents, affected people and local access |
| Structural drone | Supported building conditions and damage changes |
| Person-search drone | Visible people and supported assistance needs |
| Flood-route drone | Inundation and evacuation-route accessibility |
| Wildfire drone | Visible fire/smoke and threatened areas |
| Road/bridge drone | Infrastructure defects and access obstruction |
| Connected-car camera | Driving conditions and encountered hazards |
| Response vehicle | Scene/access evidence during response travel |
| Intersection CCTV | Traffic conflicts, obstruction and incidents |
| Building CCTV | Public-space conditions within a fixed view |
| River/underpass camera | Visible water/inundation and access |
| Industrial robot | Equipment/site inspection under its assigned task |
| Agricultural drone | Crop/field survey and visible off-mission incidents |
| Maintenance camera | Asset condition and supported field defects |

Generate full mission paragraphs from frozen templates with parameters for actual area/task; save exact text. These are study profiles informed by [ITU M.2160](https://www.itu.int/rec/R-REC-M.2160-0-202311-I/en) and [RFC9450](https://datatracker.ietf.org/doc/html/rfc9450), not standardized 15-class 6G equipment categories.

Simulate joins/leaves, pauses/resumptions, variable capture rates and incident bursts. Already received jobs remain after UE departure. Device identity never determines priority.

Use compatible source pools and coherent incident sequences. Label real ordered streams, controlled copies/crops and shuffled stress tests separately. Random images do not establish temporal novelty.

Preserve real metadata. Generated timestamps/GPS/uncertainty/trajectories/orientation are simulated and internally coherent. Do not claim arbitrary real photos were captured at invented coordinates.

Sender processing order: query pre-commit KB, compute history-dependent parameters, serialize, commit sender observation, then offer after extraction/transport delay. Offered stream determines sender KB independently of receiver policy.

Initial aggregate loads rho=.3/.7/1.1 with rho=lambda*d0. Fixed aggregate-load population scaling is separate from fixed per-UE-rate scaling. Add repeat/burst scenarios selectively, not a giant factorial.

Main service uses d0 and all normal completions. Optional variable durations/exceptions are generated once per message/incident and shared across policies. No selection-order-dependent redraws. Exception consumes time, produces no verified evidence and gets no retry.

## 12. Evaluation, comparisons and selection

Six mandatory tables:

1. Reference-class waits and counts, including unscheduled and unknown-reference traffic.
2. Distinct-evidence first-service delay, coverage and repeat selections.
3. nDCG@1/3/5 where grades exist; preference agreement/ties/uncertainty otherwise.
4. Population, load, burst, missingness and source/event/mission transfer.
5. Feature-family, receiver-history, architecture, reward and teacher ablations.
6. Runtime ms, FLOPs, memory, model size, training/teacher cost and actual transport bytes.

Use fixed policy-independent snapshot probes alongside on-policy probes. Different policies visit different queues; do not ignore that difficulty difference. nDCG uses a frozen grade-to-gain map and same-set IDCG; all-zero IDCG is undefined. Pair preferences alone do not provide cardinal gains. SPI-only nDCG does not prove within-class novelty performance.

Alpha-nDCG is optional only with real nugget labels. Evidence delay/coverage are the interpretable primary semantic measures.

Select task-specific champions: message cost for message_priority, evidence cost for evidence_priority, with class-delay/starvation outcomes always alongside. Preserve Pareto trade-offs. Set practical tolerances from development-only variability before locked evaluation; until then report feasibility rather than promotion.

Use locked groups once after selection; paired uncertainty resamples independent source groups, not queue seeds. Event-held-out needs verified event IDs. Repeat chosen model/baseline training sufficiently to expose observed variation within budget.

Profile context construction plus selection separately from model forward; declare hardware/backend/precision/batch/warmup, queue sizes, sample counts and MAC/FLOP convention. Synthetic d0 is not measured milliseconds. Runtime-aware replay includes measured selection overhead.

Measure actual serialized record bytes. Historical35-byte,32-byte and Scene JSON contracts are different. Payload size remains unknown until a codec/payload is supplied.

## 13. Build ownership and contracts

One canonical repository. parameters/ extracts; Scheduler/ scores; Scheduler/training/ trains/exports; Evaluation/ owns lifecycle/benchmarks; Codec/ is deferred. New experiments live in experiments/stream_scheduler/.

Extend Evaluation/stream_simulator.py, stream_io.py, scene_environment.py, decoder_service.py and Scheduler/training/stream_pilot.py/scene_learners.py. Keep old objective versions reproducible.

Runtime ingress: versioned parameter bytes, producer version, routing identity, arrival/sequence, permitted mission association and complete opaque payload reference.

References: split/group/provenance, assessed priority/status and supported facts, separate from ingress.

Context: immutable ready candidates, processed headers, current time and permitted metadata. Decision: full selected identity, scores, snapshot/model/profile hashes, fallback status.

Controller owns queue mutation and one in-flight job. Check finite monotonic clocks, complete payloads and duplicate identities. Tie-break arrival/sequence/full identity. Invalid model artifact fails startup; unexpected nonfinite score triggers logged FIFO fallback.

## 14. Build–test–integrate sequence

| Stage | Deliverable | Acceptance |
|---|---|---|
| A Freeze | Source/schema/data-exposure manifest | Actual field/mask/producer compatibility; no SPI changes |
| B Objective | Versioned image/fact delay evaluators | Independent interval-integral and hand-worked cases |
| C Lifecycle/context | Processed-history adapter and variable ready sets | Exact-once lifecycle, permutation/leakage/reset checks |
| D First vertical slice | Rules+linear/MLP/RankNet,2–3 episodes | Meaningful learning, export/reload/resume and independent replay |
| E Contextual model | Deep Sets RankNet and controlled overlap | Matched labels, no-history/summary ablations, exact tiny references |
| F Real labels | Source audit and qualified Qwen | Provenance/splits, raw responses and honest silver boundaries |
| G Suite | Trees/classifier, RL/PER, RLAIF/replay, attention/warmstart | Shared observations/budgets; blocked arms explicit |
| H Scale | Dynamic50–100 UEs and cost profiling | No truncation; measured overhead and targeted stresses |
| I Locked study | Frozen champions and final groups | Paired evidence, claims and reproduction artifacts |
| J Codec later | Supplied digital payload path | Actual bytes, service and completion measurements |

Numerical tests: reward/area equality; mid-service arrivals; terminal/idle intervals; unknown/zero-useful cases; exact-repeat suppression; equal-weight invariance; restricted greedy optimum; multi-fact counterexamples.

Isolation tests: evaluator-label poisoning, IDs/future/outcomes absent, candidate permutations, source-group separation and training-only preprocessing.

Learning tests: known pair gradient direction, actual optimization update, checkpoint optimizer/RNG/replay resume, NumPy parity and legal target masks. A successful import or arbitrary decreasing loss is insufficient.

## 15. Resources, logs and Kaggle

Reuse local environment and CPU first. No GPU execution is launched by this plan. Recheck access/quota and established notebook identity before remote submission. Reuse notebook versions; update parameters/KAGGLE_RUNS.md.

Retain six aggregate GPU-hours as the initial approved-run ceiling when execution is authorized/preflighted. Count loading/allocated idle devices. Earlier allocation:3h extraction/teacher,2h justified accelerator work,1h diagnosed contingency. Codec allocation remains held while paused; no automatic broader sweep.

Audit metadata and20 records before new downloads; preserve initial2GB source-image cap. Cache image evidence once; recompute causal sender novelty for changed streams.

Every computational cell logs purpose/config/inputs/destination, progress and completion timing. Preserve executed notebook and incremental stdout/stderr/errors plus source/environment/hardware/data/split versions.

Each unique run stores manifest/config, source hashes, runtime/reference traces, raw teacher outputs, targets, checkpoints, portable models, per-decision/per-group metrics, figures, validation and report. Verify representative remote retrieval/checksums. Platform COMPLETE alone is insufficient.

## 16. Current evidence, immediate next action and claims

Existing CPU pilot stream-pilot-20261029-220835 integrated8 non-teacher learners,14 policies and742 decisions. Three supervised epochs/64 RL decisions established fixture feasibility, not convergence.15 focused checks and export audit were recorded. Its reward remains successful completion coverage; it does not implement the newly planned dispatch-cost objectives. Three teacher arms remain blocked.

The earlier objective-check-20261030-013021 tested6,561 restricted tiny fixed-weight cases and100 fractional-arrival cases. It did not validate real semantics, nDCG or this contextual system.

**Next increment: A–D.** Freeze actual upstream records; implement reference-unit timing and processed-header context beside legacy behavior; run2–3 complete episodes with rules and small learners. Real teacher training follows reference/qualification gates. The broader contextual project remains intact.

Intended contribution: an audited parameter-only contextual scheduler with semantic supervision and explicit delay/evidence trade-offs under dynamic streams. Novelty, general emergency effectiveness and superiority remain hypotheses.

Stop semantic claims when labels fail. Investigate missing information before adding models. Retain negative RL/teacher results. If no independent image urgency exists, report task-evidence or silver-supervision findings honestly.

Maintain this one plan. Record changes with evidence and affected results. Never silently replace dispatch with completion, image delay with fact coverage, teacher agreement with truth, or synthetic timing with deployment performance.

