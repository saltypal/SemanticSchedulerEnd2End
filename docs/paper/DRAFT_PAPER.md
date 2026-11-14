# Semantic-Aware Decoder Scheduling at the O-RAN Edge: Stateless Semantic Headers, Receiver-Side Novelty, and Semantic-Coverage Learning for One-Shot Generative Image Communication

*Living draft. IEEE journal structure in single-column Markdown. Target: IEEE TCCN / TWC / IoT-J.*
*Methods synced to FINAL_PLAN_V3 v3.1, decisions D1-D20 of `docs/CHECKLIST.md` and fix plan `docs/reviews/FIX_PLAN_2026-11-13.md` (2026-11-14). Status: **no final results yet**.
Every `[RESULT PENDING: ...]` marks a number that has not been computed; nothing in brackets is a result.
The only measured numbers in this draft are dataset counts, the SPI validation report, and the batched decode-latency
measurement of codec v1 (Sec. VII-B). Cross-references: `docs/reviews/FINAL_PLAN_V3.md`, `docs/CHECKLIST.md`,
`docs/design/SEMANTIC_SCHEDULER_DESIGN.md`, `parameters/JOURNAL.md`.*

Authors: [names], [affiliations]

---

## Abstract
Generative semantic codecs such as SQ-GAN compress an image into a few hundred bytes, but they move
the cost from the channel to the receiver. Decoding is slow, and during a disaster surge an edge
decoder pool cannot keep up with arriving payloads. We study **which received image should be
decoded next, and which should never be decoded**, when every UE transmits once, keeps no history,
and attaches only a fixed 64-byte header. We make four contributions:

1. **A stateless semantic header.** It carries a calibrated urgency posterior (SPI), mission relevance, a
   64-bit semantic key, a 64-bit perceptual key, and a codec-aware *Semantic Token Summary* (STS) derived for
   free from SQ-GAN's own semantic mask. The receiver fuses SPI and STS into one tier belief that every
   scheduling policy reads.
2. **Receiver-side novelty.** The scheduler detects cross-UE redundancy from the two keys, so no UE state is needed.
3. **A timely-semantic-value objective with per-tier timeliness targets.** Value is v = w * g * zeta
   (tier weight, deadline curve, duplicate credit); the headline metric is Critical Timely Recall at the
   deadline (CTR@D). It is solved by a drift-plus-penalty scheduler (SV-DPP), by constrained learners trained
   with the SCOPE recipe, and by **SemCover**, a proposed semantic-coverage scheduler that selects the next
   decode batch as a conditional determinantal point process / facility-location cover of urgent meaning.
4. **Evaluation against human labels.** CrisisMMD human damage annotations mixed with everyday imagery,
   event-disjoint splits, a fine-tuned SQ-GAN codec, and a line-up of classical, 5G-derived, index-based,
   extended and learned schedulers.

[RESULT PENDING: headline sentence, e.g. CTR@D of the best baseline vs. SV-DPP / SemCover at rho = 1.2]

**Index Terms:** semantic communication, goal-oriented scheduling, O-RAN, edge inference, generative
image coding, Lyapunov optimization, determinantal point process, submodular coverage, constrained reinforcement
learning, disaster response.

---

## I. Introduction
**Motivation.** 6G envisions semantic and goal-oriented communication [1]-[4]. Generative codecs
reach extreme compression by letting the receiver *synthesize* detail [5]-[8], [37]. This creates a new
bottleneck: **receiver compute**. In a disaster, thousands of phones, drones and vehicles upload
images within minutes [9]. A generative decoder at the edge cannot decode them all in time, and
not all of them matter. Selfies, memes and retweeted duplicates compete with images of collapsed
buildings and injured people.

**Gap.** Existing semantic resource allocation schedules *radio* resources and assumes importance is
known [10]-[13], [38]-[40]. Queueing-aware SemCom controls the semantic rate of one stream [14].
Edge-inference serving schedules DNN requests by deadline [15]-[17], [85] but is blind to image content.
Disaster-imagery works classify urgency [9], [18] without connecting it to the communication system.
To our knowledge, no work schedules **generative semantic decoding** with:

- a compact header that travels with the payload;
- no per-UE state;
- a selection rule that values *new meaning* rather than jobs in isolation.

**Contributions.**
1. System model and problem: decoder-side semantic scheduling at the O-RAN edge, posed as a constrained
   problem on timely value (Sec. III).
2. A 64-byte stateless semantic header, and codec-aware STS derived from SQ-GAN's masking, fused with SPI at the
   receiver (Sec. IV).
3. Receiver-side semantic novelty from a SimHash semantic key and a pHash perceptual key (Sec. V).
4. SV-DPP, a renewal-frame completion-aware Lyapunov scheduler with per-tier debt queues (extending [60], [14]);
   SemCover, a semantic-coverage scheduler with a (1 - 1/e) per-decision guarantee for its facility-location
   variant; and SCOPE, a constrained, teacher-guided training recipe (composed from [20]-[24], [63]-[68]) (Sec. VI).
5. A human-labelled, event-disjoint benchmark with trace-driven simulation, a fine-tuned codec, measured batched
   decode latency, and a live 2xT4 check (Sec. VII-VIII).

*We adapt DPP and PPO-Lagrangian. The novelty is the problem, the header, the receiver-side novelty, the
semantic-coverage view of decoder scheduling and the evaluation.*

---

## II. Related Work
**A. Semantic and goal-oriented communication.**
- Foundations and surveys: [1]-[4], [41]; the AoI -> semantics-of-information -> goal-oriented line is traced in [89]
  (semantics-empowered communication and value of information: [93]).
- DeepJSCC [5] and Transformer / SwinJSCC codecs [6].
- Generative SemCom [7], [37], [42], including GAN-based and packet-erasure-robust variants [43], [44].
- SQ-GAN [8]: semantic-conditioned masked vector quantisation with runtime masking fractions.
- User- and query-oriented semantic coding with large vision-language models [45], [46].
- These works optimise the codec. We take the codec as given and schedule its decoding.

**B. Semantic resource allocation and semantic scheduling.**
- Importance-aware RA for task-oriented SemCom [10]; multi-view semantic RA for edge AI [11].
- Semantic spectral efficiency [12]; QoE-driven OFDMA SemCom [38].
- Content-preference feedback [39]; goal-oriented utility optimisation [40].
- Age of Semantic Importance, which couples freshness and semantic loss and is minimised with DQN scheduling [80];
  monotonicity-driven DRL for semantic-aware scheduling [81]; constrained-MDP / Lagrangian-duality semantic-aware
  scheduling [82].
- All of these allocate radio resources and assume importance is known at the scheduler.

**C. Timeliness.**
- Age of Information [25]; semantic and version AoI [26], [27]; Age of Incorrect Information [28].
- Time-utility functions in real-time scheduling [29]. Our deadline curve g(.) follows the TUF view.

**D. Queueing and index policies.**
- Generalised c-mu rule [30]; Whittle index for convex delay costs with abandonment [31].
- Lyapunov drift-plus-penalty [19]; its SemCom use with a virtual semantic-error queue [14].
- Wireless QoS schedulers PF / M-LWDF / EXP [32], [33]; we adapt them to a decoder queue.

**E. Edge inference serving.**
- Deadline-aware multi-DNN GPU serving [15]; priority- and interference-aware serving [16]; RL for
  hard/soft real-time inference [17]; pricing priority under excess demand [34].
- Progressive edge inference under deadlines [47].
- BCEdge [85], maximum-entropy RL for adaptive batching under SLOs, is the closest learning-based edge-serving
  baseline; a BCEdge-style SAC batching baseline is added next to the serving baselines.

**F. Redundancy, deduplication and coverage.**
- LSH [35], SimHash [36]; CLIP-based semantic deduplication [48]; copy detection [71].
- Redundancy-aware uplink for overlapping fields of view [49]; Tri-Select [86] filters redundant crowdsensing
  images (disaster response as a use case) and notes that centralised filtering struggles at the edge. SemCover does
  redundancy-aware selection header-only, with no decoding.
- Diversity through DPPs [91], [92], fast greedy MAP inference [90]; facility-location coverage combined with
  query relevance is monotone submodular, which gives the greedy (1 - 1/e) guarantee [94] and is used in SAVER [83]
  and MASCOT [84].
- DPP-based quality-diversity RL post-training [87] and differentiable DPP learning (Wasserstein DPP) [88].

**G. Disaster imagery and VLMs.**
- CrisisMMD [9], MEDIC [18], VLM disaster benchmarks [50], [51]. O-RAN architecture [52].

**H. Learning to schedule.**
- LambdaMART [53], PPO [54], constrained RL [20]-[22], imitation with a queryable teacher [63]-[68].
- Human-in-the-loop RL for SemCom latency [55].

<!-- TODO-verify: all B.7 citations ([80]-[89]) are preprints or proceedings found 2026-11-13; publication status not verified. [87] DQO (2025) has no arXiv id in the design note. [91]-[94] are from memory (design B.5 says check before citing). -->

---

## III. System Model and Problem Formulation
**Fig. 1 [PLACEHOLDER]: system architecture.**
- UEs run SegFormer [56] -> SQ-GAN encoder + SAMM -> CLIP [57] -> 64-B header.
- One-shot transmission over an internet-like pipe.
- O-Cloud edge site runs: header parser -> receiver memory (delivered-image hash ring) -> shared queue ->
  scheduler -> c decoder workers (one GPU each, exclusive mode, batch B) -> responder application.
- The near-RT RIC xApp manages policy parameters (tier weights, targets eta_k, V, mission registry) over A1/E2 [52].

**A. Traffic.**
- U UEs, each with a mission profile from a registry (4 missions in the experiments: general, flood_storm,
  wildfire, earthquake; Appendix A lists the wider draft registry).
- Everyday images arrive Poisson (MMPP optional).
- CrisisMMD images arrive in their **real tweet chronology**, time-compressed by a factor kappa to reach the target
  load rho = lambda E[S_B] / (B c). Tweet post time is a proxy for capture time (jitter sensitivity row).
- Each image is sent once as `[h_i][payload_i]`. A loss probability p_loss in [0, 0.05] is a sensitivity row.
- Sampling is without replacement: a realisation replays each test image at most once; realisations differ by
  permutation, kappa and seed.

**B. Receiver queue.**
- A single shared queue with c identical decoder workers; decode time is sampled from the **empirical** latency
  distribution L(codec, B) measured per batch size (Sec. VII-B). c >= 4 is labelled idealised (no host contention).
- The scheduler observes only headers. It sees neither UE identity nor payload content.
- Actions at each worker-free epoch: choose a batch, defer, or purge.

**C. Value.** An image i finished at time t yields

  v_i(t) = w(k_i) * g_{k_i}(t - tau_i) * zeta(dup_i),

where k_i is the true tier (Critical / High / Medium / Low), w = (8, 4, 2, 1) (swept), g_k is a time-utility curve
equal to 1 up to D_k and decaying linearly to 0 at the hard deadline H_k = 2 D_k, tau_i is capture time, and zeta = 1
for the first delivered copy and 0 for later copies of non-Critical images; **zeta_C = 0.25 for later Critical copies**
(corroboration value). **Decoded-image fidelity and the u-gated metrics of earlier drafts are dropped (D5):** the
value is not multiplied by a measured task-correctness indicator.

**D. Problem.** Maximise long-run average value subject to per-tier timeliness targets

  P(decoded before H_k | true tier k) >= eta_k,  eta = (0.95, 0.85, 0.60, 0.20).

A necessary feasibility condition is sum_k eta_k lambda_k E[S_B]/(B c) < 1; we claim constraint satisfaction only
inside the feasible (rho, c) region. Only header quantities are causal; true tier and duplicate group are used for
evaluation and for the training reward on training events only.

**Table I [PLACEHOLDER]: notation.**

---

## IV. Stateless Semantic Header and Codec-Aware Descriptors
**A. Header layout (64 B, FINAL_PLAN_V3 Sec. 4).**

| Off | B | Field |
|---:|---:|---|
| 0 | 1 | version (4 b) + flags (4 b: SPI / STS / semantic key / perceptual key valid) |
| 1 | 4 | message id |
| 5 | 4 | capture time (ms mod 2^32) |
| 9 | 3 | payload length |
| 12 | 1 | codec id (4 b) + m index (4 b) |
| 13 | 2 | mission profile id |
| 15 | 8 | SPI posterior (4 x u16) |
| 23 | 2 | mission relevance (u16) |
| 25 | 8 | semantic key: 64-bit SimHash of the CLIP embedding (secret per-deployment seed) |
| 33 | 7 | STS: 13 x 4-bit nibbles: 7-class kept-token histogram (7), retention ratios (4), centroid + spread (2); 14th nibble zero pad |
| 40 | 8 | perceptual key: 64-bit pHash (DCT, no model) |
| 48 | 2 | CRC-16 over bytes 0-47 |
| 50 | 14 | reserved / extension |

Bit-depths were set by a decision-flip-rate test on validation data (S6, < 0.5% decision changes versus full
precision): SPI and relevance need 16 bits to keep decision flips under 0.5% (8 bits: 8.2% / 7.9%); STS needs only
4 bits (0 flips at 4 bits, with and without fusion). Version 1 (8-bit fields) is no longer produced. Codec v1 changes the STS class map, so it sets `version = 2` and the receiver uses
the matching STS profile. The predicted decode utility u-hat of earlier drafts is not in the header and is not used.

**B. Semantic Priority Indicator (SPI).** CLIP ViT-B/32 embedding -> class-balanced multinomial logistic regression
(C = 1) -> temperature scaling (T = 0.95 on validation), trained on human-derived tiers including everyday images;
training-row posteriors are 5-fold out-of-fold so the simulator never sees over-confident train posteriors.
Selected from a grid of five encoders x five classifier families by validation macro-F1 and Critical recall
(Table V); CLIP-B/32 is kept because the same embedding already yields the semantic key and relevance.

**C. Mission relevance.** r_i = (cos(CLIP(x_i), CLIP(text_pi)) + 1)/2, with each mission text the mean embedding of
three prompts, pre-shared via the registry. CLIP cosines are compressed: values span 0.52-0.68 (about 40 of 256 levels at 8 bits; about 10,500 of 65,536
at the 16-bit field); the scaling is fixed in the bit-depth test.

**D. Scene descriptors: from detectors to the codec's own mask.**
1. A YOLO / COCO scene block averaged 34.45 B (dense maximum 307 B); removing COCO classes cut it to 8.16 B and the
   full header from 110.45 to 84.16 B (earlier iteration).
2. Open-vocabulary detection (Grounding-DINO) is flexible but adds a second network on the UE.
3. **STS.** SQ-GAN tiles the image into 16x32 tokens and SAMM keeps the top m fraction [8]. The 20-class map is
   majority-voted onto the grid; over the **kept** tokens we compute a 7-superclass histogram, retention ratios for
   person / vehicle / building / road, and the centroid and spread of kept objects. Kept tokens are transmitted
   and dropped tokens are synthesised, so STS measures how much task-relevant content is *real*.

**Algorithm 1 [PLACEHOLDER]: STS extraction.**

**E. Receiver fusion of SPI and STS (D19).** In plan v3 STS reached the analytic policies only through the fidelity
predictor, which D5 removed. The receiver therefore fits a class-balanced multinomial logistic model on
[log SPI (4), STS (13)] on the train split, temperature-calibrated on validation, and produces one tier belief
q = Fuse(SPI, STS). **Every policy** reads q where it read SPI; for SV-DPP this changes E[v_i] = sum_k q_ik w_k g_k(.) zeta
and the debt arrivals A_k = sum_i q_ik. If the STS field is masked (flag bit off) the belief falls back to SPI, which
is how the header ablation measures the value of STS.

**Table II [RESULT PENDING: header ablation (SPI, +relevance, +keys, +STS/fusion, equal-bytes thumbnail baseline, genie), CTR@D per policy with and without STS].**
**Table III [RESULT PENDING: SPI vs. Fuse(SPI, STS): macro-F1 and Critical recall on validation].**

---

## V. Receiver-Side Semantic Novelty
**A. SimHash semantic key.** b_j = 1[r_j^T e > 0], j = 1..64, with random hyperplanes R shared via the registry.
P(b_j != b'_j) = theta/pi [36], so the Hamming distance d estimates the angle: cos(theta) ~ cos(pi d / 64).

**B. Perceptual key.** A 64-bit pHash (about 7 ms on the UE) detects near-copies independently of the CLIP geometry.

**C. Duplicate model and soft dedup.** P(dup) is a logistic model on (semantic Hamming, perceptual Hamming),
fitted on mined, rule-labelled pairs (D14: duplicate iff pHash <= 8 or SimHash <= 1; 372 candidate pairs from
train + validation, of which 116 pHash <= 8, 74 SimHash <= 1, 54 same-tweet; mining was necessary because uniform
random pairs gave only 3 real copies in 353). The paper states that duplicate labels follow this rule. The receiver
memory holds only *delivered* images. Dedup is soft (E[v] already includes zeta); a hard purge happens only if the
matching copy was delivered and P(dup) > 0.95, or if the job is past H_k. Never purge a likely-Critical image unless
the semantic key is near-identical (d <= 3).

**Fig. 4 [RESULT PENDING]:** duplicate-detection ROC per event and vs. hash length (32 / 64 / 128 / 256 bits).

---

## VI. Scheduling Policies
**A. Policy line-up.** FINAL_PLAN_V3 Sec. 8 representatives plus the extended set and learners:
- *v3 line-up:* FIFO, Random; RR over missions; EDF, TUF best-effort (Locke/DASA); SPI-first + EDF, priority + aging;
  M-LWDF (representative of delay-weighted QoS schedulers); gen-c-mu/h with abandonment and Whittle with
  abandonment; EdgeServing stability-score lookahead (adapted); completion-aware DPP [60] (adapted) and per-tier DPP
  without Delta-G; MPC / Monte-Carlo rollout; genie-aided SV-DPP and an offline clairvoyant MILP bound on short traces.
  Redundant rules (e.g. EXP/DAPF = M-LWDF here, LLF = EDF) are implemented and shown equivalent numerically in an
  appendix.
- *13 extended policies (D10):* e.g. D-over, HVDF, EDF-triage, LCFS-priority, DRR over tiers, MaxWeight, MaxAge,
  Whittle-AoI, CVaR-SPI, LinUCB (full list in `Scheduler/ALGORITHMS.md`).
- *Learners:* LambdaMART, RankNet, DQN, PPO, BC-only, DAgger, and the SCOPE instances.
- *Proposed:* SV-DPP, SCOPE-PPO / DQN / LTR, **SemCover**.
- *BCEdge-style batching baseline* [85]: entropy-regularised actor-critic over the batch size (implemented).
- *Batching controls (semantics vs batching):* every ranking policy and learner chooses its batch size with one shared
  renewal-ratio rule [61]: B* = argmax_B sum_{i in top-B} sum_k q_ik w_k P(S_B <= H_k - a_i) / E[S_B], over its own order.
  `spi_edf_ratio` (SPI-first-EDF order + the rule) and `edf_ratio` (header-blind EDF order + the rule with a uniform
  tier prior) isolate the value of the header from the value of adaptive batching.
- SCOPE-AIF (RLAIF) is parked by the user and is not part of this paper's result set.

**B. SV-DPP (FINAL_PLAN_V3 Sec. 6).** SV-DPP extends completion-aware Lyapunov scheduling [60] and semantic DPP [14]
to decoder-side scheduling with noisy headers.

*Debt queues.* One per tier, Z_k <- max(0, Z_k + eta_k A_k - T_k), with A_k and T_k soft counts of predicted arrivals
and predicted on-time decodes (computed from the fused belief q).

*Score.* s_i = V * dE[v_i] + sum_k Z_k q_ik dG_i, where dG_i is the change in on-time probability from decoding now
instead of deferring, from the empirical latency CDF and a first-pass wait estimate. Ties are broken by E[v_i], then
arrival time.

*Batching and dedup.* Batching follows the renewal-frame ratio rule [61] (maximise sum of scores / E[S_B], every
item evaluated at S_B). Dedup is soft with a delivered-only memory (Sec. V).

*Relevance factor (D18).* E[v] is multiplied by 1 + beta (r_tilde - 0.5), r_tilde the header relevance; beta is tuned
on validation, and beta = 0 recovers plan v3.

*Two specification fixes found during implementation.* (i) Deferring the top-ranked job looked free; the defer-wait
estimate is now at least one service round. (ii) The all-or-nothing tie rule is replaced by a per-job tie tolerance
`tie_tol_frac`, a parameter added beyond plan v3. V and the purge threshold are tuned on validation; beta (D18) and
`tie_tol_frac` are tuned on a 3 x 3 validation grid [RESULT PENDING]. Final numbers come from the full validation and
test sweeps.

*Claims.* Debt stability is verified empirically (no O(1/V) theorem for non-stationary traces). In the degenerate
case SV-DPP reduces to expected-weight priority. Proposition [PLACEHOLDER]: relation to gen-c-mu/h [62] and to Whittle
with abandonment [31] for c = 1.

**Algorithm 2 [PLACEHOLDER]: SV-DPP.**

**C. SemCover: scheduling meaning, not jobs (proposed).** *Status: implemented (`Scheduler/edge/semcover.py`,
`Scheduler/training/edge/semcover_train.py`); results [RESULT PENDING]. Implementation notes: (i) cos(pi d/64) and the
p-hash indicator are not PSD in general, so the kernel is projected onto the PSD cone (eigenvalue clipping), normalised
to a unit diagonal and given a 1e-3 ridge; (ii) a time-decayed memory is modelled as noisy observations of delivered
scenes, S_QQ - S_QM A (A S_MM A + I - A^2)^-1 A S_MQ with A = diag(exp(-age/tau_k)), tau_k = 2 H_k (32 most recent
deliveries), because decaying urgency alone cancels in DPP conditioning; (iii) the RL stage samples items
sequentially in proportion to their marginal gains (greedy MAP is the mode), which has a tractable log-probability but
is not an exact k-DPP sample; (iv) urgency is monotone in the Critical/High belief and deadline pressure with the
context embedding fixed; the learned net is a residual on the analytic prior. Fallback if it does not beat SCOPE-LTR / SV-DPP on validation: reported
as a semantic-diversity variant with its ablations; SV-DPP stays the main proposal.*

*Philosophy.* The value of decoding an image is not a property of the image alone: it is how urgent its meaning is
for the mission, times how much *new* meaning it adds given what the edge already decoded and what else is in the
batch. Two photos of the same collapsed bridge are worth less together than a bridge and a flooded hospital.

*Semantic space visible from the header.* The SimHash key gives an approximate CLIP angle between any two images,
queued or delivered, without decoding; STS gives scene composition; the fused belief q (Sec. IV-E) gives tier belief;
relevance and mission id give mission context; pHash flags near-copies.

*Model.*
- Urgency q_i: a set-attention encoder over the queue (reusing the SetPolicy encoder) from [fused belief, STS,
  relevance, mission embedding, slack/age], so urgency is judged in context. The urgency network is constrained to be
  monotone non-decreasing in the Critical/High belief and in deadline pressure, following [81].
- Similarity kernel S_ij: a learned non-negative mix of cos(pi d_sem/64), STS-histogram similarity and a pHash
  near-copy indicator; positive semidefinite by construction.
- Selection: a conditional DPP over the queue, L = diag(q) S diag(q). The next batch is the greedy MAP of det(L_B)
  conditioned on the receiver memory (Schur complement) with exact incremental-Cholesky greedy MAP [90]. This trades
  urgency against semantic redundancy inside the batch and against what was already decoded. Memory is time-decayed,
  with weight exp(-age/tau_k) per tier, so a scene delivered long ago no longer suppresses a new similar image
  (freshness in the spirit of [80]). Critical corroboration (zeta_C = 0.25) is kept by a tier-dependent kernel
  floor for Critical-Critical pairs.
- Deadline feasibility: jobs that cannot finish before H_k are masked, and q_i is scaled by the on-time probability
  G_i(now), the same quantity SV-DPP uses.
- Two coverage objectives are offered: log-det (DPP, best diversity, not monotone) and mission-relevance
  facility-location coverage, which is monotone submodular, so greedy selection is within (1 - 1/e) of the best batch
  per decision [94], [83], [84]. The theory point is this guarantee for the facility-location variant only.
- "DPP" here is *determinantal point process*, not drift-plus-penalty; the paper spells both out.

*Learning (SCOPE recipe, semantic objective; train split only, tuned on validation, 5 seeds).*
1. Supervised stage: the DPP log-likelihood of high-value batches from simulator rollouts of SV-DPP and the genie
   (value = simulator's own timely value with true tier and duplicate group) trains q and the kernel weights.
2. Fusion head: cross-entropy of the fused belief versus true tier, sharing the encoder.
3. Constrained RL fine-tune: sample batches from the k-DPP policy (tractable log-probability), reward
   r = value - sum_k lambda_k miss_k with dual updates, and annealed distillation toward SV-DPP early. A Wasserstein-DPP
   relaxation [88] is the fallback if the policy gradient is noisy; DPP-based quality-diversity training is supported by [87].

*Ablations:* kernel off (S = I, an urgency ranker), memory conditioning off, STS off, memory decay off, monotonicity off,
each loss term off. Extra metric beside CTR@D: distinct semantic clusters (true duplicate groups) of Critical images
delivered on time. *Claimed narrowly:* decoder-queue scheduling as conditional DPP selection in a header-only semantic
space, with soft geometry-based redundancy handling and a learned kernel; no claim of optimality for the log-det variant.

**D. SCOPE: a training recipe for learners.** The objective is
min_theta max_lambda -J_v + sum_k lambda_k (eta_k - J_k) + beta(n) L_imit(theta; pi_SV-DPP).
- Constraint costs enter only through returns; teacher and student share the four tier constraints.
- Imitation queries the analytic teacher at the states the learner visits [63]-[66]. The imitation weight is
  **adaptive** (TGRL-style [96]): each iteration the greedy student and the teacher run the same episode; the weight
  shrinks (x0.9, floor 0.05) when the student matches or beats the teacher and grows (/0.9, cap 1) otherwise. A fixed
  decay schedule let the student drift below the teacher once imitation faded. Claimed novelty is only the
  composition and its application.
- Instances: SCOPE-PPO (set-attention actor), SCOPE-DQN (DQfD-style margin loss [68]; replay buffer, optimizer and
  target network persist across outer iterations; best checkpoint kept on held-out training episodes), SCOPE-LTR
  (listwise distillation). All ranking learners share the batch-size rule of Sec. VI-A and rank jobs that cannot meet
  any likely hard deadline last (a queueing analogue of the work-conserving parametrisation of [97]). Plain PPO is
  warm-started by behaviour cloning of SV-DPP (PPO-BC [101]); value targets use scaled returns so the value loss
  does not swamp the clipped policy gradient.
- *Training protocol:* train split only; bursty MMPP arrivals at rate 4.5 /s on c = 2 workers, B <= 8, 150 iterations x
  13 episodes (about 1,950 episodes); two independent seeds for proposed learners (each seed has its own episode
  stream), one for baselines. Following [98], learners are evaluated on both steady and bursty load, since RL tends to
  help only on the latter.
- A proof-of-value experiment compares SCOPE with the teacher, BC-only, DAgger and plain RL at equal budgets in regimes
  where the teacher is misspecified (wrong sigma_W, noisy SPI, MMPP bursts). SCOPE is claimed only if it beats the
  teacher there. SemCover is trained with this recipe but is not a SCOPE variant.

---

## VII. Experimental Setup
**A. Data, labels and splits.**
- Pool (D1-D3, dataset A `scheduler-dataset-v1-images`, 27,168 images): CrisisMMD v2 [9] 18,082 images with human
  annotations; Places365 [58] validation sample 6,086 (17 per class, minus 7 damage-lookalike classes); COCO people
  3,000 [95]. All non-CrisisMMD images are Low (D3).
- CrisisMMD tier counts before dedup and the D16 fix: Critical 2,483, High 2,192, Medium 5,589, Low 7,818.
- Tier mapping (D16, aligned with FINAL_PLAN_V3 Sec. 3): severe damage -> Critical; mild -> High; little_or_none and
  informative -> Medium; injured / dead / missing people -> Critical (override); affected individuals -> Medium (High
  only in the `affected_high` sensitivity mapping); not informative -> Low. Only 3,526 of 18,082 CrisisMMD images carry
  a damage label, so humanitarian labels fill in for the rest (infrastructure and vehicle damage -> High). Label
  noise: 1,032 humanitarian-vs-informative conflicts, 236 informative flags overridden.
- Event-disjoint splits (D4): train Harvey, Irma, Iraq-Iran, Sri Lanka; **validation California wildfires** (Sri Lanka
  had only 71 Critical, below the gate of 100); test Maria, Mexico earthquake. Everyday images split 60/10/30 by a
  stable hash. Cross-split near-duplicates are removed: a validation/test image is dropped if pHash <= 8 or SimHash
  <= 1 against train (124 dropped; looser rules would drop hundreds of distinct Critical test images). The pHash
  histogram shows a gap after distance 8; the SimHash histogram is smooth because crisis photos are semantically close.

**Table IV: final tier counts per split (v3 mapping, after dedup; `docs/CHECKLIST.md` S2b).**

| Split | Critical | High | Medium | Low | Total |
|---|---:|---:|---:|---:|---:|
| train | 1,218 | 1,047 | 3,653 | 10,056 | 15,974 |
| validation | 499 | 139 | 429 | 1,418 | 2,485 |
| test | 760 | 492 | 1,892 | 5,441 | 8,585 |

Gates pass (test Critical 760 >= 150; validation Critical 499 >= 100).

**Table V: SPI classifier grid on validation (class-balanced; selection on val; 15,974 train, 2,501 val before dedup).**
Excerpt of `parameters/JOURNAL.md` (2026-11-11).

| Candidate | macro-F1 | Critical recall | ECE |
|---|---:|---:|---:|
| SigLIP2 logistic C=1 | 0.589 | 0.750 | 0.023 |
| SigLIP2 logistic C=0.1 (grid pick) | 0.587 | 0.754 | 0.023 |
| **CLIP-B/32 logistic C=1 (kept)** | 0.577 | **0.830** | 0.021 |
| CLIP-B/32 logistic C=10 | 0.570 | 0.774 | 0.025 |
| CLIP-B/32 logistic C=0.1 | 0.566 | 0.824 | 0.018 |
| DINOv2-base logistic C=1 | 0.564 | 0.730 | 0.021 |
| CLIP-B/32 MLP 128 | 0.560 | 0.680 | 0.025 |
| DINOv2-small linear SVM C=0.1 | 0.548 | 0.502 (fails gate) | 0.029 |

**Table VI: final SPI head on validation (2,485 images; test not scored).**

| Metric | Value |
|---|---|
| macro-F1 | 0.567 |
| Critical recall / precision | 0.818 / 0.683 (gate 0.6) |
| High recall / precision | 0.079 / 0.141 |
| Medium recall / precision | 0.534 / 0.545 |
| Low recall / precision | 0.875 / 0.893 |
| ECE (calibrated) | 0.024 |
| CrisisMMD-only macro-F1 / Critical recall | 0.519 / 0.818 |
| Everyday images predicted Low (COCO / Places365) | 98.6% / 98.7% |
| False Critical on everyday val, without / with normals | 1.1% / 0.5% |
| Train out-of-fold macro-F1 / Critical recall | 0.678 / 0.745 |

High is the confused middle tier (reported as a limitation); the scheduler consumes the posterior, not the argmax,
and class imbalance is handled by a class-balanced loss, calibration, a Critical-recall gate and a Critical-share
sensitivity row (D13). Mean UE-side cost per image on a T4: CLIP-B/32 12.2 ms, SigLIP2 18.1 ms, DINOv2-base 24.5 ms,
pHash 6.9 ms, SimHash 0.3 ms. The simulator table has 27,044 rows (one per image: true and sensitivity tier,
calibrated SPI, relevance per mission, keys, 760 duplicate groups over 1,965 images, tweet time for CrisisMMD).

**B. Codec v1 and decode latency.**
- *Fine-tune (D7-D9).* SQ-GAN [8] is fine-tuned from the authors' official Cityscapes component checkpoints on 10,434
  train-split images (5,540 smaller than 256x512 skipped, which removes Places365), with a 20-class ADE20K-merged
  semantic map. Batch size 2 (one image per T4; batch 4 and 8 ran out of memory); about 4,700 steps (about 0.9 epoch)
  in an 8 h cap, DataParallel on 2xT4. Less than one epoch was possible, so v1 is a light adaptation.
  No v0 pilot results are used (D17). A full v1 pass over all images is being completed with a resume run
  [RESULT PENDING: STS, payload bytes at m = 0.25 / 0.55, kept-token maps, LPIPS and SegFormer mIoU of
  reconstruction vs. original (D6 gate)].
- *Batched decode latency.* True batched decode of codec v1 on one T4 (m = 0.25, 40 samples each), median:

| Batch B | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Latency S_B (s) | 0.74 | 1.46 | 2.88 | 3.86 |
| S_B / (B S_1) | 1.00 | 0.99 | 0.97 | 0.65 |

  Batching helps only at B = 8; for B <= 4 it gives essentially no speed-up. An earlier pipeline emulated batches
  sequentially; this was corrected. The batching decision in SV-DPP and the c-worker simulator therefore use the
  measured per-B empirical distributions, and batching is not assumed to be beneficial at small B.

**C. Simulator.**
- Trace-driven, event-driven, c in {1, 2, 4, 6, 8}, rho swept; 10% warm-up, 10 realisations, latency sampled from the
  measured distribution (P95 only inside policies).
- Live validation on 2xT4 with c = 2, about 1,000 images, three policies: KS test of simulated vs. live latency and the
  CTR@D difference with CIs; the claim is limited to c = 2.
- Sensitivity rows: `affected_high` tier mapping, Critical share, w x0.5 / x2, loss 0-5%, capture-time jitter.

**D. Metrics.**
- Headline: **CTR@D** (Critical Timely Recall at deadline D_C): fraction of human-labelled Critical images decoded
  before the Critical deadline.
- Normalised timely value v = w g zeta; Critical-correct per decoder-second; per-tier deadline success against eta_k;
  p95 sojourn; Low starvation; redundant decodes; false purges of distinct Critical images; decision time (us);
  header bytes and overhead 64/(64 + payload); distinct Critical clusters delivered (for SemCover).
- Bootstrap CIs over images and events.

---

## VIII. Results
All entries below are placeholders; no sweep has been run.

*Internal preliminary checks (not results, not for the paper; 300-500 validation images, c = 2, rho = 1.2, one
realisation, no CIs; recorded only to guide the sweeps):* with the shared batch rule, SPI-first ordering reached CTR@D
0.44-0.50 while the header-blind order reached 0.00, so the header carries the ordering signal; SCOPE-PPO (2 seeds) 0.51-0.52,
LambdaMART 0.51, SV-DPP 0.49, SCOPE-LTR 0.43-0.45, SCOPE-DQN 0.28-0.33, DQN 0.32. A quick SV-DPP header drop-out gave the
same CTR@D with and without the header (0.49); the decisive full header proofs with CIs are pending.
- **Fig. 5 [RESULT PENDING]:** CTR@D vs. load for all policies.
- **Table VII [RESULT PENDING: main CTR@D table at rho = 1.2, c = 2, validation-tuned, test-evaluated].**
- **Fig. 6 [RESULT PENDING]:** normalised value vs. load; Pareto front of CTR@D vs. Low starvation.
- **Table II (above) [RESULT PENDING]:** header ablation with and without STS fusion; equal-bytes baseline.
- **Fig. 7 [RESULT PENDING]:** decision-flip rate vs. bit-depth (SPI, relevance, STS).
- **Fig. 8 [RESULT PENDING]:** SCOPE learning curves vs. teacher, BC, DAgger, plain RL in misspecified regimes.
- **Table VIII [RESULT PENDING]:** SemCover vs. SV-DPP vs. SCOPE-LTR, with its ablations.
- **Table IX [RESULT PENDING]:** SPI with vs. without everyday images (false-Critical rate).
- **Table X [RESULT PENDING]:** simulator vs. live 2xT4 (latency KS, CTR@D delta).
- **Fig. 9 [RESULT PENDING]:** sensitivity (weights, deadlines, tier mapping, Critical share, loss, jitter).
- **Table XI [RESULT PENDING]:** beta (D18) tuned on validation; SV-DPP tie-tolerance sensitivity.

---

## IX. Discussion and Limitations
- Codec v1 had under one epoch of fine-tuning, and SQ-GAN is out of distribution on disaster photos; STS inherits the
  20-class ADE20K-merged map, which has no fire or rubble class.
- Fidelity of the reconstruction is no longer part of the objective (D5); codec quality is only gated (D6: LPIPS and
  SegFormer mIoU of reconstruction against the original, on validation).
- Tweet timing is a proxy for field traces; the tier mapping is a design choice (sensitivity provided); duplicate labels
  follow a rule (D14); relevance values are narrow (0.52-0.68).
- SPI High recall is weak (0.08 on California); only Critical recall is gated.
- Batching helps only at B = 8 on a T4; conclusions depend on this latency profile.
- DPP guarantees hold on time-averages, not per image; the (1 - 1/e) guarantee applies per decision to the
  facility-location variant only. c >= 4 is idealised.
- The Qwen label audit and SCOPE-AIF (RLAIF) were parked and are not evaluated.
- The uplink is not modelled: images reach the edge queue at their arrival time without channel delay or loss (a loss
  sensitivity row only); the study is about decoder-side compute scheduling.
- DQN-family learners were unstable across seeds; they are reported as baselines.

## X. Conclusion
[PLACEHOLDER]

---

## References
*Verify every arXiv ID before submission. 26xx IDs come from the alphaXiv index and are summarised
from their abstracts.*

[1] D. Gündüz et al., "Beyond transmitting bits: Context, semantics, and task-oriented communications," IEEE JSAC, vol. 41, no. 1, 2023.
[2] E. C. Strinati and S. Barbarossa, "6G networks: Beyond Shannon towards semantic and goal-oriented communications," Computer Networks, 2021.
[3] E. Uysal et al., "Semantic communications in networked systems," IEEE Network, 2022.
[4] H. Xie, Z. Qin, G. Y. Li, B.-H. Juang, "Deep learning enabled semantic communication systems," IEEE TSP, 2021.
[5] E. Bourtsoulatze, D. B. Kurka, D. Gündüz, "Deep joint source-channel coding for wireless image transmission," IEEE TCCN, 2019.
[6] K. Yang et al., "SwinJSCC: Taming Swin Transformer for deep joint source-channel coding," IEEE TCCN, 2024.
[7] E. Erdemir, T.-Y. Tung, P. L. Dragotti, D. Gündüz, "Generative joint source-channel coding for semantic image transmission," IEEE JSAC, 2023.
[8] F. Pezone, S. Barbarossa, G. Caire, "SQ-GAN: Semantic image communications using masked vector quantization," IEEE TCCN (arXiv:2502.09520).
[9] F. Alam, F. Ofli, M. Imran, "CrisisMMD: Multimodal Twitter datasets from natural disasters," ICWSM, 2018.
[10] "Importance-aware resource allocation for collaborative task-oriented semantic communication," arXiv:2606.29052.
[11] "Multi-view semantic-aware resource allocation for edge AI," arXiv:2610.34532.
[12] L. Yan, Z. Qin, R. Zhang, Y. Li, G. Y. Li, "Resource allocation for text semantic communications," IEEE WCL, 2022.
[13] "Semantic-aware sub-band allocation for terahertz communications," arXiv:2608.30984.
[14] Y. E. Sagduyu, T. Erpek, "When semantic communication meets queueing: Cross-layer latency and task fidelity optimization," arXiv:2605.05514.
[15] "EdgeServing: Deadline-aware multi-DNN serving at the edge," arXiv:2605.05527.
[16] "Strait: Perceiving priority and interference in ML inference serving," arXiv:2604.28175.
[17] "Hard and soft real-time DNN inference scheduling on heterogeneous multi-accelerator systems via RL," arXiv:2610.29503.
[18] F. Alam et al., "MEDIC: A multi-task learning dataset for disaster image classification," arXiv:2108.12828.
[19] M. J. Neely, Stochastic Network Optimization with Application to Communication and Queueing Systems, Morgan & Claypool, 2010.
[20] A. Ray, J. Achiam, D. Amodei, "Benchmarking safe exploration in deep reinforcement learning," OpenAI, 2019.
[21] C. Tessler, D. J. Mankowitz, S. Mannor, "Reward constrained policy optimization," ICLR, 2019.
[22] A. Stooke, J. Achiam, P. Abbeel, "Responsive safety in RL by PID Lagrangian methods," ICML, 2020.
[23] Y. Bai et al., "Constitutional AI: Harmlessness from AI feedback," arXiv:2212.08073, 2022.
[24] H. Lee et al., "RLAIF: Scaling reinforcement learning from human feedback with AI feedback," arXiv:2309.00267, 2023.
[25] S. Kaul, R. Yates, M. Gruteser, "Real-time status: How often should one update?" IEEE INFOCOM, 2012.
[26] "Minimizing quantized semantic age of information in foundation-model-based semantic communications," arXiv:2606.31303.
[27] "Diffusion model-based RL for version age of information scheduling," arXiv:2601.18069.
[28] "Minimizing age of incorrect information in pull-based remote estimation of a Markov source," arXiv:2610.22265.
[29] C. D. Locke, "Best-effort decision making for real-time scheduling," PhD thesis, CMU, 1986.
[30] J. A. van Mieghem, "Dynamic scheduling with convex delay costs: The generalized cμ rule," Annals of Applied Probability, 1995.
[31] "Whittle index approach to multi-server scheduling with convex delay costs and impatient customers," arXiv:2609.19792.
[32] M. Andrews et al., "Providing quality of service over a shared wireless link," IEEE Commun. Mag., 2001.
[33] S. Shakkottai, A. Stolyar, "Scheduling for multiple flows sharing a time-varying channel: The exponential rule," AMS Translations, 2002.
[34] "Inference auctions," arXiv:2609.40070.
[35] P. Indyk, R. Motwani, "Approximate nearest neighbors: Towards removing the curse of dimensionality," STOC, 1998.
[36] M. S. Charikar, "Similarity estimation techniques from rounding algorithms," STOC, 2002.
[37] "Towards a joint task-oriented and generative semantic communication framework for 6G networks," arXiv:2606.31426.
[38] "QoE-driven hybrid resource allocation for multi-user OFDMA semantic communications," arXiv:2610.32077.
[39] "Optimization of collaborative semantic communication network performance with channel and content preference feedback," arXiv:2607.25011.
[40] "Reference-dependent utility optimization for goal-oriented communication systems," arXiv:2611.12480.
[41] "Foundation models for generalizable semantic and goal-oriented communication," arXiv:2609.07853.
[42] "Diffusion-aided bandwidth-efficient semantic communication with adaptive requests," arXiv:2510.26442.
[43] "GAN-based semantic communication for image transmission in IoV," arXiv:2608.27989.
[44] "Lightweight generative image semantic communication over packet erasure channels," arXiv:2609.06989.
[45] "Personalized digital semantic communication for image transmission with vision-language models," arXiv:2608.14260.
[46] "Generalized user-oriented image semantic coding empowered by large vision-language model," arXiv:2509.08913.
[47] "Split-gate pooled-evidence stochastic-rollout scheduling for timely progressive edge inference," arXiv:2608.08338.
[48] A. Abbas et al., "SemDeDup: Data-efficient learning at web-scale through semantic deduplication," arXiv:2303.09540, 2023.
[49] "Exploiting overlapping fields of view for redundancy-aware uplink transmission in vehicular 6G," arXiv:2606.31715.
[50] "DisasterVL: Evidence-based visual reasoning dataset and model for disaster scene understanding," arXiv:2611.12144.
[51] "DisasterInsight: A building-centric benchmark for evaluating VLMs in disaster response," arXiv:2601.18493.
[52] M. Polese et al., "Understanding O-RAN: Architecture, interfaces, algorithms, security, and research challenges," IEEE COMST, 2023.
[53] C. J. C. Burges, "From RankNet to LambdaRank to LambdaMART: An overview," Microsoft Research TR, 2010.
[54] J. Schulman et al., "Proximal policy optimization algorithms," arXiv:1707.06347, 2017.
[55] "Latency-aware human-in-the-loop reinforcement learning for semantic communications," arXiv:2602.15640.
[56] E. Xie et al., "SegFormer: Simple and efficient design for semantic segmentation with transformers," NeurIPS, 2021.
[57] A. Radford et al., "Learning transferable visual models from natural language supervision," ICML, 2021.
[58] B. Zhou et al., "Places: A 10 million image database for scene recognition," IEEE TPAMI, 2018.
[59] A. Kuznetsova et al., "The Open Images dataset V4," IJCV, 2020.

[60] B. Zhao, Q. Sun, N. Li, C.-L. I, "Lyapunov-based completion-aware scheduling for PDU set-based real-time XR traffic," arXiv:2608.05537.
[61] M. J. Neely, "Dynamic optimization and learning for renewal systems," IEEE TAC, 2013.
[62] Z. Long, N. Shimkin, H. Zhang, J. Zhang, "Dynamic scheduling of multiclass many-server queues with abandonment: The generalized c-mu/h rule," Operations Research, 2020.
[63] S. Ross, G. Gordon, J. A. Bagnell, "A reduction of imitation learning and structured prediction to no-regret online learning (DAgger)," AISTATS, 2011.
[64] S. Ross, J. A. Bagnell, "Reinforcement and imitation learning via interactive no-regret learning (AggreVaTe)," arXiv:1406.5979, 2014.
[65] W. Sun, J. A. Bagnell, B. Boots, "Truncated horizon policy search: Combining RL and imitation learning (THOR)," ICLR, 2018.
[66] C.-A. Cheng et al., "Fast policy learning through imitation and reinforcement (LOKI)," UAI, 2018.
[67] V. N. Vitale et al., "From prior-guided heuristics to deployable agents (MGA-RL)," arXiv:2609.03590.
[68] T. Hester et al., "Deep Q-learning from demonstrations," AAAI, 2018.
[69] J. Cao et al., "EdgeServing: Deadline-aware multi-DNN serving at the edge," arXiv:2605.05527 (cf. [15]).
[70] S. Aalto, "Whittle index approach to multi-server scheduling with convex delay costs and impatient customers," arXiv:2609.19792 (cf. [31]).
[71] E. Pizzi et al., "A self-supervised descriptor for image copy detection (SSCD)," CVPR, 2022.
[72] "Benchmarking pretrained vision embeddings for near- and duplicate detection in medical images," arXiv:2312.07273.
[73] "On the de-duplication of LAION-2B," arXiv:2303.12733.
[74] "Adversarial collision attacks on image hashing functions," arXiv:2011.09473.
[75] S. Gul et al., "PDU set-based QoS handling in 3GPP: Release 18 overview and future directions," IEEE Commun. Stand. Mag., 2025.
[76] FEMA, Preliminary Damage Assessment Guide, 2021; S. Benson et al., "Disaster triage: START, then SAVE," Prehosp. Disaster Med., 1996.
[77] M. Imran et al., "Processing social media messages in mass emergency: A survey," ACM CSUR, 2015.
[78] M. Oquab et al., "DINOv2: Learning robust visual features without supervision," TMLR, 2024.
[79] "TempoNet: Slack-quantized transformer-guided reinforcement scheduler," arXiv:2602.18109.
[80] Y. Chen, J. Gong, "Age of semantic importance (AoSI)," arXiv:2403.07386, 2024. <!-- TODO-verify: authors/title as listed in design B.7 (Chen & Gong); publication status unknown -->
[81] "Monotonicity-driven deep reinforcement learning for semantic-aware scheduling," arXiv:2305.13706, 2023. <!-- TODO-verify: title, authors -->
[82] "Constrained MDP with Lagrangian duality for semantic-aware scheduling," arXiv:2609.07563. <!-- TODO-verify: title, authors -->
[83] SAVER, arXiv:2605.20713. <!-- TODO-verify: full title, authors -->
[84] MASCOT, arXiv:2608.12532. <!-- TODO-verify: full title, authors -->
[85] BCEdge: SLO-aware DNN inference services with adaptive batching on edge platforms (maximum-entropy RL), arXiv:2305.01519 (IEEE journal version). <!-- TODO-verify: exact title, authors, venue -->
[86] Tri-Select, arXiv:2512.16469, 2025. <!-- TODO-verify: full title, authors -->
[87] DQO: DPP-based quality-diversity RL post-training, 2025. <!-- TODO-verify: DQO name/title/authors/venue; no arXiv id available -->
[88] Wasserstein DPP (differentiable DPP learning), arXiv:2011.09712. <!-- TODO-verify: title, authors -->
[89] Luo, Delfani, Salimnejad, Pappas, survey on AoI, semantics of information and goal-oriented communication, arXiv:2512.12758, 2025. <!-- TODO-verify: exact title -->
[90] L. Chen, G. Zhang, H. Zhou, "Fast greedy MAP inference for determinantal point process to improve recommendation diversity," NeurIPS, 2018, arXiv:1709.05135.
[91] A. Kulesza, B. Taskar, "Determinantal point processes for machine learning," Foundations and Trends in ML, 2012. <!-- TODO-verify: from memory -->
[92] A. Kulesza, B. Taskar, "k-DPPs: Fixed-size determinantal point processes," ICML, 2011. <!-- TODO-verify: from memory -->
[93] M. Kountouris, N. Pappas, "Semantics-empowered communication for networked intelligent systems," IEEE Commun. Mag., 2021. <!-- TODO-verify: from memory -->
[94] G. L. Nemhauser, L. A. Wolsey, M. L. Fisher, "An analysis of approximations for maximizing submodular set functions I," Math. Programming, 1978. <!-- TODO-verify: added by the paper writer as the (1-1/e) source; the design note cites it only via SAVER/MASCOT -->
[95] T.-Y. Lin et al., "Microsoft COCO: Common objects in context," ECCV, 2014. <!-- TODO-verify -->

[96] I. Shenfeld, Z.-W. Hong, A. Tamar, P. Agrawal, "TGRL: An algorithm for teacher guided reinforcement learning," ICML, 2023, arXiv:2307.03186.
[97] E. Che, J. Dong, H. Namkoong, "Differentiable discrete event simulation for queuing network control," arXiv:2409.03740, 2024.
[98] G. Zhang et al., "When does deep RL beat calibrated baselines? A benchmark study on adaptive resource control," arXiv:2605.26418, 2026.
[99] R. Sharifullin, "Adaptive inference batching using policy gradients," arXiv:2607.05272, 2026.
[100] D. Wang, Q. Zhang, L. Ying, "Zeroth-order policy optimization for scheduling in queueing networks," arXiv:2611.01186, 2026.
[101] J. G. Dai, M. Gluzman, "Queueing network controls via deep reinforcement learning," Stochastic Systems, vol. 12, no. 1, 2022.

[59] is superseded by [95] (COCO people replaced Open Images, D1); [78] (DINOv2 judge) is unused after D5 and can be dropped.

---

## Appendix A: Mission profile registry
See `docs/reviews/MISSION_PROFILES_DRAFT.md`: 26 draft profiles across UAV, V2X/ADAS, public safety,
CCTV/IoT, industrial, consumer/XR and NTN, mapped to 3GPP TS 22.125 / 22.186 / 22.261 / 22.263 /
22.104 / 22.179 and ITU-R M.2160. **Some clause references are unverified.** The experiments use four missions
(`config/edge/missions.yaml`: general, flood_storm, wildfire, earthquake; D15).

## Appendix B: Header field justification and bit-depth study
Field table and flip-rate methodology are in Sec. IV-A. SPI and relevance need 16 bits to keep decision flips under 0.5% (8 bits: 8.2% / 7.9%); STS needs only 4 bits (0 flips at 4 bits, with and without fusion).

## Appendix C: Tier mapping and sensitivity
Full rule table (Sec. VII-A), the `affected_high` alternative mapping (509 images move between main and alternative
mapping: 437 High -> Medium, 72 High -> Low), per-event tier counts [RESULT PENDING], Cohen's kappa vs. Qwen
[parked; not computed].

## Appendix D: Hyperparameters
SV-DPP (V, beta, tie_tol_frac, eta), SCOPE learners (network, learning rate, clip, beta schedule), SemCover
(kernel weights, tau_k, k) [RESULT PENDING].

## Appendix E: Changes from plan v3 (decisions D1-D19)
Plan v3 (`docs/reviews/FINAL_PLAN_V3.md`) is the base; where it differs from the table below, the table wins.

| # | Topic | Decision / change versus plan v3 |
|---|---|---|
| D1 | People photos | COCO people, 3,000 (v3: Open Images person subset) |
| D2 | Everyday scenes | Places365 validation, 17/class, minus 7 blacklisted classes |
| D3 | Non-CrisisMMD images | all Low |
| D4 | Validation event | California wildfires (Sri Lanka had 71 Critical); train adds Sri Lanka; test Maria, Mexico |
| D5 | Fidelity | u, the DINOv2 judge, u-hat and u-gated metrics dropped; v = w g zeta; headline CTR@D |
| D6 | Codec quality gate | LPIPS + SegFormer mIoU of reconstruction vs. original, on validation |
| D7 | SQ-GAN fine-tune | done by us, second Kaggle account |
| D8 | SQ-GAN order | fine-tune, then one full pass with v1 (header version 2) |
| D9 | Fine-tune data | train-split images only; images < 256x512 skipped (drops Places365) |
| D10 | Algorithms | v3 line-up plus 13 extended policies |
| D11 | Scheduler training | all learners in phase 2, in parallel notebooks |
| D12 | Accounts | data and main line on one account, SQ-GAN on the second |
| D13 | Class imbalance | class-balanced SPI loss, calibration, Critical-recall gate, natural mix in val/test, Critical-share sensitivity |
| D14 | Duplicate labels | rule-based (pHash <= 8 or SimHash <= 1) instead of team labelling |
| D15 | Relevance | precomputed per (image, mission) from 4 missions |
| D16 | Tier mapping | affected_individuals -> Medium (High only in the sensitivity mapping); humanitarian labels fill missing damage labels |
| D17 | SQ-GAN v0 | no v0 pilot or pass for results |
| D18 | Relevance in SV-DPP | E[v] x (1 + beta (r_tilde - 0.5)), beta tuned on validation (beta = 0 is plan v3) |
| D19 | STS use | receiver fuses SPI + STS into the tier belief all policies see; STS-masked header falls back to SPI |
| - | SV-DPP specification | defer wait >= one service round; per-job tie tolerance `tie_tol_frac` (beyond v3) |
| - | Batched latency | true batched decode measured; earlier sequential emulation corrected |
| - | New method | SemCover (proposed); SCOPE-AIF / Qwen audit parked |

## Appendix F: Prior design iterations (negative results)
- Old circular waiting-penalty benchmark: FIFO 5.822, SPI-first 3.374, LambdaMART 3.218, PPO 4.594,
  DDQN 4.710 (the benchmark was circular: a VLM tier defined both labels and cost).
- YOLO / Grounding-DINO trials; VLM-reviewer gating attempts.

These are reported for transparency.
