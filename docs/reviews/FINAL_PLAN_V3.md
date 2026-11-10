# Final End-to-End Plan v3: Semantic-Aware Decoder Scheduling at the O-RAN Edge

Status: **LOCKED v3.1** (2026-11-10), after the clean-context counter-review (resolution log in §11). Where §1 differs from §4–§11, §4–§11 win. It supersedes v2, v2.1 and the
explainer wherever they conflict; those files stay as history. **This is the file to send to
teammates.**

How to read it:
- §1 answers the review questions.
- §2–§9 are the locked design.
- §10 is the **chronological build order**: steps in sequence, no dates.

Tags: [V] verified in code or files; [C] to check in step S0; [D] our documented design choice;
[L] taken from literature (cited).

---

## 1. Answers to the review questions (2026-11-10)

**Q1. G/G/c: what is it, and is it from a paper?**
G/G/c is queueing notation:
- the first G is the arrival process (General);
- the second G is the service-time distribution (General);
- c is the number of servers.

Our servers are **decoder workers**. Two papers ground it:
- Aalto (2609.19792) [L]: a multi-class M/G/N+M queue, i.e. many servers with impatient customers
  (deadlines). It derives a Whittle index for that case and notes that the generalised cμ rule is
  heavy-traffic optimal for many servers (Mandelbaum & Stolyar 2004).
- EdgeServing (2605.05527) [L], which we follow for the GPU model:
  - each GPU serves **one batch at a time in exclusive (time-division) mode**, so latency is
    predictable;
  - latency is **profiled offline** per batch size and the P95 value is stored in a table;
  - arrivals are Poisson, with warm-up excluded.

Multi-server does **not** mean "no image is left out". With enough load, any c can be overloaded; the
scheduler decides who waits and who is purged.

**Q2. How will multi-decoder simulation actually run? Kaggle 2×T4? Several workers per T4?**
- **One worker = one GPU in exclusive mode** (EdgeServing's choice). We do *not* pack several
  decoders onto one T4: concurrent kernels make latency unpredictable (EdgeServing Sec. II).
- Step S3 measures, on a single T4: decode latency for batch B ∈ {1, 2, 4, 8} (P50 and P95),
  throughput, GFLOPs per image, and peak VRAM.
- The simulator then uses c ∈ {1, 2, 4, 6, 8} **identical** workers from that profile. c > 2 is
  simulated; it is never run live.
- **Live check:** c = 2 on Kaggle's 2×T4 in step S10 confirms that the simulator's c = 2 numbers
  match reality.
- **Compute reported in the paper:** ms per decode (P50/P95 per batch size), GFLOPs, VRAM, images/s
  per T4, and the scheduler's own decision time in µs on CPU.

**Q3. In Sagduyu & Erpek, where does "semantic error" come from? Why does N appear, if SQ-GAN's
latent size is fixed?**
- Their error: the receiver's built-in classifier head labels the received image. E_k = 1 when that
  label is wrong.
- Their N = number of channel symbols per image. SQ-GAN's **latent grid** is fixed (16×32 = 512
  tokens) [V]. Its analogue of N is the number of **kept** tokens, m × 512, which sets the payload
  size.
- In our setting **decode time hardly depends on m**: the decoder always rebuilds the full grid
  [C, measured in S3]. So, unlike their N, m does not trade decode delay against accuracy. That is
  why m is fixed per device profile and we schedule *which* image to decode instead.
- **What we borrow** is their constraint machinery: a virtual queue that enforces a semantic quality
  target with no retransmission.

**Q4. "Fading and outage model removed": what was it, and what if radio scheduling is added later?**
- Fading means the channel's strength changes randomly from packet to packet. Outage means a packet
  is lost when the channel is too weak for its size at that moment.
- It was removed because you fixed the model as "no radio scheduling; each UE has its own internet-like
  pipe". Only a 0–5% random-loss sensitivity row remains.
- If radio scheduling is added later, the problem becomes **two-stage**: the base-station MAC decides
  who uplinks, then our scheduler decides who is decoded. Our simulator's arrival stream would then
  come from a radio simulator (Sionna or ns-3). This is listed as an extension, and the code keeps an
  `ArrivalSource` interface so it can be plugged in.

**Q5. Literature for the tier mapping.** Cite:
- FEMA *Preliminary Damage Assessment Guide* (2021), with its Destroyed / Major / Minor / Affected
  categories;
- START triage: Benson, Koenig & Schultz, "Disaster triage: START, then SAVE", *Prehospital and
  Disaster Medicine* (1996);
- Imran et al., "Processing social media messages in mass emergency: A survey", *ACM CSUR* (2015),
  for informative vs. non-informative;
- Alam et al., CrisisMMD (ICWSM 2018), for the annotation scheme itself.

These are established references, but **check the exact bibliographic details before the paper
goes out**.

**Q6. The semantic hash: codebook bias ("trained on Indian images, fails on Chinese images"), and
the information hashing loses.**
- **SimHash has no learned codebook.** Its 64 directions are *random* and data-independent, fixed
  once by a seed. The hash itself cannot be biased toward a region. Any bias comes from the
  **embedding** it hashes: CLIP, trained on web-scale multilingual and multi-region data, is among the
  broadest available. We **measure** cross-event robustness: duplicate-detection ROC per CrisisMMD
  event (US hurricanes vs. Mexico, Iraq–Iran, Sri Lanka).
- **Hashing loses information.** We only need it to keep "same vs. different" correct near the
  threshold. We compensate four ways:
  1. **Measure** the loss: ROC vs. bits (32 / 64 / 128 / 256).
  2. **Two-key test (your idea, upgraded):** call it a duplicate only if the hash is close **and** the
     STS histograms are close. STS alone is too coarse (two *different* collapsed buildings both look
     like "70% building"). As a second key it cuts false purges.
  3. **Safety asymmetry:** never purge an image with high pCritical unless the hash is near-identical
     (d ≤ 3).
  4. **Learned-hash upgrade** (ITQ, an orthogonal rotation fitted on embeddings), as an ablation.
- **Newer literature:**
  - SSCD (Pizzi et al., CVPR 2022, 2202.10261): a descriptor built for copy detection, used as an
    ablation embedding (extra UE model);
  - embedding benchmarks for near-duplicate detection (2312.07273);
  - LAION-2B de-duplication (2303.12733);
  - collision-resistant fine-grained hashing (2605.18288);
  - adversarial hash collisions (2011.09473), a security caveat in the limitations;
  - SemDeDup (Abbas et al. 2023), SimHash (Charikar 2002), LSH (Indyk & Motwani 1998).

**Q7. "What if the UE has never seen a photo before?" Is this two-phase?**
- Yes, and we name it: **two-stage semantic scheduling**.
  - Stage 1 is **stateless UE-side annotation**: SPI, relevance, hash, STS.
  - Stage 2 is **stateful receiver-side contextualisation**: novelty against the receiver's memory,
    debt queues, and the scheduling decision.
- The UE never needs history. At receiver start-up the memory is empty, so everything counts as novel.
  That is the correct behaviour at the start of an event.

**Q8. Rethink of û (predicted decode utility). [Changed]**
- **u**, the *measured* "decoded image still says the right tier", is **needed for evaluation**. It is
  measured by an independent judge model: CLIP-L/14 or ConvNeXt, not the UE's CLIP-B/32 SPI.
  - Why a different backbone: if the same model judged the reconstructions, a GAN artefact that
    consistently fools *that* model would be scored correct. This is the "judge ≠ model under test"
    rule.
  - The judge is trained on human tiers from training events, and u is computed only on images whose
    *original* the judge classifies correctly.
- **û moves to the receiver**, out of the header. The receiver already has STS, SPI and the codec
  config, so a tiny receiver-side logistic model computes û. This saves a header byte and UE compute
  and gives the same information.
- û is an **ablation**, not a pillar: the policies are evaluated with and without it. If it does not
  help, it is dropped from the paper.

**Q9. Is SV-DPP's "risk / slack" predefined?**
- Deadlines D_k per tier are predefined [D].
- Slack is **computed live**: slack_i = D_k − age_i − S.
- **[Changed]** The ad-hoc risk term is replaced by a principled **completion-probability gain** ΔP_i
  (§6.2), following the completion-aware Lyapunov scheduler [L 2608.05537].

**Q10. Why a Lagrange multiplier in SC-PPO? Is it gradient descent?**
- A Lagrange multiplier is a **price** on breaking a constraint; it is not a descent direction.
- Reward = value − λ × (Critical misses).
- If the policy misses too many Critical images, λ goes up and misses become expensive. If it is
  safely under the target, λ goes down.
- "Dual ascent" is just that price-adjustment rule, run alongside training. It is the learning
  counterpart of SV-DPP's debt queue Z: same idea, one learned and one online.

**Q11. What is CTR@D?**
**Critical Timely Recall at deadline D**: of all images that the **human labels** say are Critical,
the fraction that were (1) decoded, (2) finished before the Critical deadline D_C, and (3) still judged
the correct tier after decoding. It is our headline metric, and it uses no chosen weights.

**Q12. "Invent a method: if SC-PPO works, apply it to DQN, RLAIF and ranking."**
Adopted as the **SCOPE training framework** (§7): one objective applied to every learner.
- It is built from existing parts: PPO-Lagrangian, and the GPR / MGA-RL demonstration-annealing idea
  [L 2609.03590].
- Its composition for semantic decoder scheduling, with an analytic SV-DPP teacher, is ours.
- Instances: SCOPE-PPO, SCOPE-DQN, SCOPE-LTR (ranking), SCOPE-AIF (RLAIF with Qwen pairwise
  feedback).

---

## 2. System model (locked)
- **UE (stateless):** SegFormer → SQ-GAN encoder + masker (fixed m per device profile) → CLIP-B/32
  (SPI, relevance, hash) → STS. It sends `[64 B header][payload]` once.
- **Network:** internet-like, with a per-UE pipe and delay. Optional 0–5% loss as a sensitivity row.
  An `ArrivalSource` interface allows a radio simulator later.
- **O-RAN edge site (O-Cloud):** header parser → receiver memory (hash ring) → shared queue →
  scheduler → c decoder workers (exclusive GPU, batch B) → judge/application.
  - The near-RT RIC xApp sets the policy parameters: tier weights, success targets η_k, V, and the
    mission registry.
  - The non-RT RIC pushes event-level intents over A1 [L Polese et al. 2023].
- **Task-awareness:** the codec itself is unchanged. The *decoding service* becomes task-aware: the
  receiver's task decides what decoder compute is spent on.
- **Standards hook:** 3GPP Rel-18 introduced PDU Set Importance and delay-budget fields
  [L Gül et al., IEEE Commun. Stand. Mag. 2025]. Our header is the semantic, application-layer
  counterpart.

## 3. Data and labels (locked)
- **Pool:** CrisisMMD v2 (7 events, human labels) + Places365 (minus a damage-lookalike blacklist)
  + people photos (Open Images person subset).
- **Tier mapping:**
  - severe → Critical;
  - mild → High;
  - little_or_none & informative → Medium;
  - injured / dead / missing people → Critical (life-safety override);
  - not-informative and all everyday images → Low.
  - Alternative mapping for sensitivity: affected_individuals → High.
- **Splits (event-disjoint):** train {Harvey, Irma, California wildfires, Iraq–Iran earthquake};
  val {Sri Lanka floods}; test {Maria, Mexico earthquake}. Everyday images are split 60/10/30 after
  hash dedup. Gate: ≥ 150 Critical in test [C].
- **Qwen labels:** κ agreement audit, plus the RLAIF feedback source.


## 4. Header: 64 B fixed (locked v3.1)
| Off | B | Field |
|---:|---:|---|
| 0 | 1 | version (4b) + flags (4b: SPI / STS / sem-hash / p-hash valid) |
| 1 | 4 | message id |
| 5 | 4 | capture time (ms mod 2^32) |
| 9 | 3 | payload length |
| 12 | 1 | codec id (4b) + m index (4b) |
| 13 | 2 | mission profile id |
| 15 | 4 | SPI posterior (4 x u8)* |
| 19 | 1 | mission relevance (u8)* |
| 20 | 8 | **semantic key**: 64-bit SimHash of the CLIP embedding (random hyperplanes; the seed is **secret and per-deployment**, distributed with the mission registry) |
| 28 | 13 | STS: 7-class kept-token histogram (7), retention ratios (4), centroid + spread (2)* |
| 41 | 8 | **perceptual key**: 64-bit pHash (DCT; no model, ~1 ms on UE) |
| 49 | 2 | CRC-16 over the header |
| 51 | 13 | reserved / extension |

\*Bit-depth set by a decision-flip-rate test on **val** (< 0.5% decision changes vs. full precision).

Version bump rule: codec v1 changes STS semantics (new class map), so it sets `version=2`. The receiver
uses the STS profile matching the version.

## 5. Objective, judge and metrics (locked v3.1)

**5.1 Fidelity judge (independent of CLIP).** A DINOv2 linear probe trained on human tiers from
**train events**.
- **u_i = 1[judge(reconstruction) = judge(original)]**, i.e. *fidelity to the original's meaning*.
  No images are dropped.
- A **human audit** of about 300 reconstructions (team, roughly 1–2 h) reports agreement between u and
  human judgement.

**5.2 Value.**

  v_i(t) = w(k_i) · u_i · g_k(t − τ_i) · ζ(dup_i)

- g_k is a time-utility function [Locke 1986]: 1 up to D_k, then linear to 0 at the **hard deadline
  H_k = 2D_k**. Every rule in the plan uses the same g and H.
- ζ(dup) = 1 for the first delivered copy, and 0 for later copies of Low, Medium or High.
  **ζ_C = 0.25 for later Critical copies** (corroboration value).
- w = (8, 4, 2, 1), swept.

**5.3 Per-tier targets on timeliness only (feasibility fixed).**
- Target: P(decoded before H_k | true tier k) ≥ η_k, with η = (0.95, 0.85, 0.60, 0.20).
- Fidelity u is **not** part of the constraint; it enters only through value.
- **Necessary feasibility condition:** Σ_k η_k λ_k E[S_B]/(B·c) < 1. The paper reports the feasible
  (ρ, c) region and only claims constraint satisfaction inside it.
- Predicted-vs-true timely-success calibration is reported per tier, because the debt queues run on
  predictions.

**5.4 Metrics.**
- **Headline:** CTR@D, reported **with and without the u-gate**.
- Per-tier timely success vs. η_k (true labels).
- Normalised value.
- Critical-correct per GPU-second.
- Breakdowns: by u-difficulty tercile (anti-gaming), by data source (anti-shortcut), and by event.
- Also: P95 sojourn, Low starvation, redundant decodes, **false purges / false deprioritisations of
  distinct Critical images (with CIs)**, decision time (µs), header bytes, decode ms / GFLOPs / VRAM.
- Stats: bootstrap over images/events; 10 realisations.

## 6. SV-DPP v3.1 (renewal-frame, ratio form)
**6.1 Debt queues (timeliness only, soft predicted counts).** At each decision epoch:

  Z_k ← max(0, Z_k + η_k·Â_k − T̂_k)

- Â_k = Σ p_ik over new arrivals.
- T̂_k = Σ p_ik over images finished before H_k.

**6.2 Soft on-time probability from the empirical latency CDF F_S** (not P95):
- G_i(now) = F_{S_B}(H_k − age_i).
- G_i(defer) = P(age_i + Ŵ_i + S_B ≤ H_k).
  - Ŵ_i comes from a **first-pass ranking** by E[v] (no fixed point needed): the work of the jobs
    ahead of i, plus expected higher-tier arrivals within i's remaining slack, divided by c.
  - σ_W combines the latency variance with the Poisson/MMPP arrival variance.
- ΔG_i = G_i(now) − G_i(defer).

**6.3 Consistently differential score:**

  s_i = V·ΔE[v_i] + Σ_k Z_k·p_ik·ΔG_i

- ΔE[v_i] = E[v_i | now] − E[v_i | defer], so both terms are "what this decision changes".
- Ties (both terms ≈ 0, i.e. nobody is at risk) are broken by E[v_i] and then by arrival time.

**6.4 Batching (renewal-frame ratio rule)** [Neely, "Dynamic optimization and learning for renewal
systems", IEEE TAC 2013].
- At a worker-free epoch, choose the batch size B and the batch set 𝓑 (top-B by s_i, with every item
  evaluated at finish time S_B) to maximise Σ_{i∈𝓑} s_i / E[S_B].
- Head-of-line misses inside a batch are visible to the rule because every item uses S_B.

**6.5 Soft deduplication** (replaces hard purge):
- P(dup) comes from a logistic model on (semantic-key Hamming, perceptual-key Hamming), fitted on
  **labelled duplicate pairs** (S2c).
- E[v] already includes ζ.
- A hard purge happens only when the matching copy has been **delivered** (the memory holds only
  delivered images) and P(dup) > 0.95, or when G_i(now) = 0 (past H_k).

**6.6 Claims (honest).**
- Bound: we state the renewal-system DPP conditions [Neely 2013] and **verify empirically** that the
  Z_k are mean-rate stable in the feasible region. There is no O(1/V) theorem for non-stationary
  tweet traces.
- Degenerate case (no deadline pressure, no duplicates, u ≡ 1): s reduces to **expected-weight
  priority**. That equals SPI-first only when SPI is confident.
- An analytic note (c = 1, Z = 0) relating the score to generalised cμ/h and to Whittle with
  abandonment goes in the paper as a proposition placeholder.
- Positioning: SV-DPP extends completion-aware DPP [2608.05537] with soft multi-class membership,
  semantic value, renewal-frame batching and redundancy. Both 2608.05537 (adapted) and "per-tier DPP
  without ΔG" are baselines.

## 7. SCOPE v3.1 (corrected)
**Formulation:**

  min_θ max_{λ ≥ 0}  −J_v(θ) + Σ_k λ_k (η_k − J_k(θ)) + β(n)·L_imit(θ; π_SV-DPP)

- The constraint costs enter **only through returns**: reward r = v − Σ_k λ_k c_k, where c_k = a
  tier-k miss. There is no separate double-counted term.
- λ_k is updated by dual ascent (or PID-Lagrangian [Stooke 2020]).
- **Teacher and student share the same four tier constraints.**

**Imitation:** queryable-teacher imitation of the DAgger / AggreVaTe / THOR / LOKI family (Ross 2011;
Ross & Bagnell 2014; Sun et al. 2018; Cheng et al. 2018) and DQfD (Hester 2018), with β(n) decaying in
live experience [MGA-RL 2609.03590]. **Claimed novelty is only the composition and its application.**

**Instances:**
- SCOPE-PPO: set-attention actor.
- SCOPE-DQN: per-job Q.
- SCOPE-LTR: listwise distillation + Lagrangian-reweighted gains.

**SCOPE-AIF** is reframed as **two arms**:
- (i) reward-only RLAIF: the Qwen Bradley–Terry reward model replaces v^true, everything else as in
  SCOPE.
- (ii) a **fully AI-feedback pipeline**: an SPI head and teacher built from Qwen labels, no human
  labels anywhere in training.

Both are evaluated on human labels. The question answered: how much of the human-label pipeline's
CTR@D does AI feedback recover?

**Minimal proof experiment (required):**
- Compare teacher vs. BC-only vs. DAgger vs. plain RL vs. SCOPE.
- Equal compute and tuning budgets; 5 seeds; learning curves.
- Run in **teacher-misspecified regimes**: MMPP bursts, wrong σ_W, noisy SPI with label noise
  injected on val.
- SCOPE is claimed only if it beats the teacher there.

## 8. Policy line-up (pruned to representatives; equivalences shown in an appendix)
At a decoder queue every job has the same service time and there is no channel diversity, so several
5G and real-time rules collapse into each other:
- M-LWDF, EXP and DAPF reduce to delay-weighted priority;
- PF over missions reduces to RR;
- LLF and LELF reduce to EDF;
- gen-cμ reduces to weight priority.

We implement them all once, **show the equivalence numerically in Appendix F** (so the comparison
stays broad), and keep these representatives in the main tables:

| Group | Main-table policies |
|---|---|
| Sanity | FIFO, Random |
| Fairness | RR over mission profiles |
| Real-time | EDF; TUF best-effort (Locke/DASA) |
| Priority | SPI-first + EDF; priority + aging |
| 5G-derived | M-LWDF (representative of delay-weighted QoS schedulers) |
| Queueing theory | gen-cμ/h with abandonment [Long et al. 2020]; Whittle with abandonment [Aalto 2609.19792] |
| Edge serving | EdgeServing stability-score lookahead (adapted) |
| Lyapunov | Completion-aware DPP [2608.05537] (adapted); per-tier DPP without ΔG |
| Planning | MPC / Monte-Carlo rollout using the simulator (strong, slow) |
| Learning (plain) | LambdaMART, RankNet, DQN, PPO, BC-only, DAgger |
| **Proposed** | **SV-DPP v3.1**, SCOPE-PPO, SCOPE-DQN, SCOPE-LTR, SCOPE-AIF (i) and (ii) |
| References | **Genie-aided SV-DPP** (true tier, u, dup); **offline clairvoyant MILP upper bound** on short traces (about 200 jobs) |

## 9. Simulator (locked v3.1)
- **Trace-driven, event-driven, sampled without replacement.** Each realisation replays every test
  image **at most once**, unless it is a true repost in the tweet data.
- Realisations differ by permutation of everyday-image arrivals, time compression κ, and seed.
  Artificial resampling duplicates are not allowed.
- Latency is **sampled from the empirical distribution** L(codec, B) (P95 is used only inside policies).
- c ∈ {1, 2, 4, 6, 8}; c ≥ 4 is labelled "idealised independent workers (no host contention)".
- Arrival model:
  - **κ and ρ calibrated and stated:** ρ = λ·E[S_B]/(B·c);
  - tweet post time is **not** capture time (limitation, plus a jitter sensitivity row);
  - Poisson for everyday images; MMPP option;
  - `ArrivalSource` interface kept for radio extensions.
- **Live validation** on 2×T4 with c = 2, about 1,000 images, 3 policies:
  - latency KS test (simulated vs. live);
  - per-policy CTR@D delta with CIs;
  - claim limited to c = 2.

## 10. Chronological build order (fixed v3.1; each step starts only when its inputs exist)
Rule: the codec checkpoint is a versioned input; every downstream table is tagged `codec=v0` or `v1`.

| Step | What | Needs | Produces | Gate / freeze |
|---|---|---|---|---|
| **S0** Verify & profile | CrisisMMD files, labels, tweet ids, per-event and per-tier counts (**check that Sri Lanka has ≥ 100 Critical, else val = California wildfires, removed from train**); Places365 blacklist; SQ-GAN v0 on T4: masker keys, latency **distribution** per B, GFLOPs, VRAM | — | `counts.json`, `sqgan_v0_profile.json` | ≥ 150 test Critical |
| **S1** Raw labels | Apply the tier mapping (no splits yet) | S0 | `labels_raw.parquet` | — |
| **S2** UE features | CLIP-B/32 embeddings → semantic key, relevance; pHash | S1 | `ue_features.npz` | — |
| **S2b** Dedup & splits | Cross-split near-duplicate removal (pHash + SimHash): any test/val image with a near-duplicate in train is dropped from test/val. Event-disjoint splits; everyday images split 60/10/30 | S2 | `splits_v1.json` | **Splits FROZEN** |
| **S2c** Dup-pair labels | About 400 candidate pairs (retweets + near matches) from train/val, labelled duplicate / distinct by the team | S2b | `dup_pairs.csv` | — |
| **S3** SPI | Retrain + calibrate (train events + normals); "no normals" ablation; κ vs. Qwen; per-source report | S2b | SPI head | Critical recall on **val** ≥ 0.6 |
| **S3b** Judge | DINOv2 linear probe on human tiers (train events) | S2b | judge | val accuracy reported |
| **S4** Codec v0 table | SegFormer + SQ-GAN v0: STS, payload bytes, decode, u (judge fidelity), LPIPS; human audit of u | S0, S2b, S3b | `codec_v0_table` | u not ≈ 0 → else JPEG ladder becomes primary |
| **S5** Receiver models | û model; P(dup) logistic model on `dup_pairs`; cross-event key ROC | S2c, S4 | `receiver_models_v0` | — |
| **S6** Header | Pack/unpack/CRC; bit-depth flip-rate test (val) | S3–S5 | `semantic_header64.py` | **Header v1 FROZEN** |
| **S7** Simulator + analytic policies | Env (§9); all analytic policies; SV-DPP; genie; MILP bound; sanity checks | S4–S6 | sim + policies | Tuned on val only |
| **S8** Learning | Plain learners, BC, DAgger, SCOPE-PPO/DQN/LTR; minimal proof experiment | S7 | checkpoints | — |
| **S9** RLAIF | Qwen pairwise + pointwise on **train** images (**collection may start right after S2b**); reward model; SCOPE-AIF (i) and (ii) | S2b (data), S8 (training) | AIF checkpoints | — |
| **S10** Freeze & test (v0) | Freeze everything; run sweeps and ablations on **test**; live 2×T4 validation | S7–S9 | results v0 | **Results v0 FROZEN** |
| **S11** Codec v1 | Teammate's SQ-GAN fine-tune (from S2b, train events only, ADE20K-merged 20 classes) → header `version=2` STS profile → re-run S4 → S5 → S7 (re-tune on val) → S8 (**retrain learners**) → S10. Also report "v0-trained policies on v1" as a robustness row | v1 checkpoint | results v1 | v1 adopted only if mean u rises on val |
| **S12** Paper | Update DRAFT_PAPER from frozen results | S10, S11 | manuscript | — |

**Parallel tracks (no collisions):**
- (a) SQ-GAN v1 training: starts after S2b, train images only.
- (b) Qwen RLAIF collection: starts after S2b, train images only.
- (c) Main line: S2c → S10.

**Leakage rules:**
- Test is touched only in S10 and S11.
- Every gate uses val.
- Cross-split near-duplicates are removed in S2b.

## 11. Counter-review resolution log (2026-11-10)
| Objection | Resolution |
|---|---|
| η infeasible (û only on the completion side) | Constraint now on timeliness only; feasibility region stated; calibration reported (§5.3, §6.1) |
| True success capped by SPI recall | Calibration study plus a genie-aided reference that quantifies the header gap (§5.3, §8) |
| CLIP judge correlated with CLIP SPI; biased drop | DINOv2 judge, u = fidelity to the original, no drop, human audit, CTR@D with and without the u-gate (§5.1, §5.4) |
| g vs. purge deadline mismatch | One hard deadline H_k = 2D_k everywhere (§5.2, §6) |
| û gaming / source shortcut | u-difficulty, source and event breakdowns (§5.4) |
| O(1/V) bound invalid | Renewal-frame ratio DPP; empirical stability; no unsupported theorem (§6.4, §6.6) |
| Mixed absolute/differential score; circular Ŵ; undefined σ_W; P95 not a probability | Fully differential score; first-pass ranking; σ_W defined; empirical CDF (§6.2–6.3) |
| Degenerate ≠ SPI-first | Claim corrected (§6.6) |
| Thin novelty vs. 2608.05537 | It is a baseline, plus a "no ΔG" ablation and an analytic note; positioning narrowed (§6.6, §8) |
| λ double-counted; teacher/student constraints misaligned | min-max with costs via returns; same four constraints (§7) |
| SCOPE is DAgger-like; capped by the teacher | Cited; novelty narrowed; minimal proof experiment in misspecified regimes (§7) |
| SCOPE-AIF leaks human labels | Two arms, incl. a fully AI-feedback pipeline (§7) |
| Two correlated semantic keys; double loss; public seed; no labelled pairs | Perceptual + semantic keys; delivered-only memory; secret seed; labelled pairs; soft dedup; Critical corroboration (§4, §6.5, S2c) |
| Resampling creates fake duplicates; P95 circularity; batch head-of-line; c = 8 contention; weak live check | Without replacement; empirical latency; batch rule evaluates S_B; c ≥ 4 idealised; KS test + CIs with 1,000 images (§9) |
| S3 test gate; missing judge; S1/S2 cycle; cross-split near-dups; numbering; single small val; dirty v1 swap | All fixed in the §10 table (S0 val check, S2b, S3b, S11 retrain + version bump) |
| Redundant policies; oracle not an upper bound; missing baselines | Representatives + equivalence appendix; genie-aided naming; MILP bound, MPC rollout, TUF/DASA, 2608.05537 added (§8) |
