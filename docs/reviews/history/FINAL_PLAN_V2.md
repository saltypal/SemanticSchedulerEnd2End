# Final Plan v2: Semantic-Aware Decoder Scheduling at the O-RAN Edge

> **v2.1 overrides (2026-11-10). See `PLAN_V2_EXPLAINED.md` for the reasoning.**
> - **Labels:** CrisisMMD damage severity: severe→Critical, mild→High, little_or_none + informative→Medium.
>   A life-safety override sends injured/dead/missing people to Critical. Not-informative and all
>   everyday images (Places365 minus a damage-lookalike blacklist, plus people photos)→Low.
>   This replaces §5.3.
> - **Network:** an internet-like pipe per UE. No radio scheduling and no fading/outage model; only an
>   optional 0–5% loss sensitivity row. This replaces the uplink text in §1.
> - **UE rate rule and semantic-map-only decode:** removed from the core. m is fixed per device profile;
>   the rate rule is one ablation row only. Map-only decode is future work.
> - **Header:** new layout in `PLAN_V2_EXPLAINED.md` §9. CRC-16 over the header only; mission id 16-bit;
>   payload length 24-bit; the learned descriptor moves to reserved bytes (ablation); 20 B reserved.
>   Bit-depths are chosen by a decision-flip-rate test. This replaces §4.
> - **SV-DPP score corrected:** `V·E[v_i] + Z·pC_i·risk_i`, with purge when E[v_i] = 0. This replaces §8.1.
>   SV-DPP and SC-PPO are **adaptations** (Neely DPP; PPO-Lagrangian), not inventions.
> - **Policies:** expanded to about 20, including adapted 5G logic (PF over mission profiles, M-LWDF,
>   EXP) and **SC-PPO-AIF (RLAIF)**. See explainer §11.4–11.7.
> - **Simulation:** trace-driven lookup plus one live end-to-end validation run.
>   SQ-GAN fine-tune with ADE20K-merged classes is a parallel teammate track.
> - **Schedule:** phases P0–P7, not 4 days (time constraint lifted).

Date: 2026-11-10. This **supersedes §6, §7, §9.2 and §10 of `Q1_REVIEW_AND_FINAL_PLAN.md`**. The
diagnosis in that file (§1–§5) still holds.

Why v2: the faculty clarified the system model. (1) The encoder sits on the UE and the decoder sits at
the O-RAN edge. (2) There is no scheduling-request/grant handshake: every packet is
`[header][payload]`, and it lands in one receiver queue. (3) The scheduler does not track UEs, and UEs
keep no history. v1 had turned the work into an uplink radio scheduler, which is the wrong problem. The
scheduled resource is the **semantic decoder at the edge**. That is a real bottleneck: SQ-GAN decode
takes 13.5 s per image on CPU (`native_sqgan_pilot.md`), and even on GPU a generative decoder serves
far fewer images per second than the radio can deliver during a disaster surge.

Tags used below: **[V]** = verified in code or files, **[C]** = check on Day 0, **[D]** = design
decision (ours, documented, swept for sensitivity).

---

## 0. What changes, at a glance

| Area | Old | New | Why |
|---|---|---|---|
| System model | Single server, unit service, i.i.d. images, uniform fake arrivals | Edge decoder pool (c GPU workers) with measured per-mode decode time; one-shot lossy uplink feeds the queue; Poisson/bursty arrivals; disaster arrivals in **real tweet chronology** | Queueing model from Sagduyu & Erpek (2605.05514) and edge-DNN serving work; realism |
| Where the scheduler runs | Unspecified receiver | Edge application co-located with the SQ-GAN decoder on O-Cloud; policy can be packaged as a near-RT RIC xApp (10 ms–1 s loop fits decode timescales) | Matches the agreed UE-encoder / O-RAN-decoder split |
| What the scheduler decides | Order only | **Decode next / defer / purge** (+ optional semantic-map-only decode) | This is how "the scheduler helps the decoder": compute is spent only on images worth decoding |
| Ground truth | Qwen pseudo-labels (circular) | **Human labels** (CrisisMMD) mapped to 4 tiers by a fixed rule; normal images = Low | Removes circularity |
| SPI classifier | CLIP head on VLM labels, never saw normal photos; Critical recall 2/10 | Retrained CLIP head on human tiers **plus** Places365/OpenImages normals, temperature-calibrated | Your point: real traffic includes selfies and ordinary photos |
| Novelty | Distance to the UE's own past images (needs UE history) | **Receiver-side redundancy**: 64-bit semantic hash in the header, compared at the scheduler with what it has already decoded or queued | UEs keep no history; the receiver legitimately does |
| Context relevance | CLIP(image, UE mission paragraph) | **Kept** (1 byte, computed on UE) | Needs no history; mission text is pre-shared |
| Scene descriptor | YOLO / COCO / Grounding-DINO block (8–34 B, never shown to help) | **SQ-GAN Semantic Token Summary (STS)**, 14 B, free by-product of encoding | Codec-aware; no second detector on the UE |
| Header | 72 B prefix + scene + CRC ≈ 84–110 B; variable | **Fixed 64 B** (12 B reserved) | Your 64-byte target |
| Objective | Class-weighted waiting penalty | **Timely semantic value**: tier weight × task-correct-after-decode × deadline curve × not-duplicate | Measures what the decoder delivers, beyond delay |
| Policies | 14 learners + RL that lose to a sort | 6 baselines, 1 analytic proposal (**SV-DPP**), 1 RL proposal (**SC-PPO**), 1 oracle | Each has a reason to exist |
| Datasets | LADI / MEDIC / CrisisMMD / Places / OpenImages, mixed and re-decided many times | **CrisisMMD v2 (disaster, human labels, 7 events) + Places365 + OpenImages (normal)**. LADI and MEDIC dropped | Smallest set that has human labels, events, timestamps and normals |

---

## 1. System model (what the paper's Fig. 1 shows)

```
 UE (phone / drone / car)                    One-shot uplink           O-RAN edge (O-Cloud GPU)
 ┌──────────────────────────────┐            (no HARQ, no SR/BSR)     ┌───────────────────────────────────────┐
 │ image x                      │                                     │ header parser (64 B) ──► Scheduler     │
 │ SegFormer → 20-class map s   │  [64 B header][SQ-GAN payload] ───► │ receiver queue (all UEs mixed)   │     │
 │ SQ-GAN encoder + SAMM (m_x)  │   lost w.p. PER(SNR, bytes)         │ redundancy memory (hashes)       ▼     │
 │ CLIP-B/32 → SPI, relevance,  │                                     │ c decoder workers: SQ-GAN decode       │
 │            64-bit hash       │                                     │   → task model / responder dashboard  │
 └──────────────────────────────┘                                     └───────────────────────────────────────┘
```

- **UE compute** [V shapes in `sqgan.py:60-75`]: SegFormer (needed by SQ-GAN anyway), SQ-GAN
  encoder and masker, CLIP-B/32. No YOLO and no VLM on the UE. The VLM is used **offline only**, as an
  audit.
- **UE rate rule** [D]: the UE picks its SQ-GAN masking fraction from its own SPI:
  Critical/High → m_x = 0.55, otherwise m_x = 0.25 (m_s fixed at 0.25). It needs no history. An
  ablation fixes m = 0.25 for everyone.
- **Uplink** [D]: block Rayleigh fading per packet. Lost if the instantaneous rate cannot carry the
  bytes within the slot budget, with no retransmission. Airtime = bytes / rate. The radio is **not**
  scheduled; it only thins and delays arrivals.
- **Receiver**: one FIFO-agnostic queue. The scheduler sees **only headers**, never UE identity, and
  acts whenever a decoder worker becomes free.
- **Decoder service time** [C]: measured on a Kaggle T4 for (a) full decode, (b) semantic-map-only
  decode (`decoder_semantic` alone; the code runs it first, then the image decoder conditioned on it
  [V `sqgan.py` decode path]).

**How the scheduler helps the decoder (the task-aware story).** The decoder pool is the scarce
resource. Without the header, every payload costs one full decode, whether it is a selfie, a
retweeted duplicate or a collapsed building. With the header, the scheduler:
1. decodes high-value, deadline-close images first;
2. **purges** payloads that are expired or exact semantic duplicates of something already decoded;
3. optionally gives a cheap semantic-map-only decode to images whose value is unclear (stretch goal).

The paper metric that captures this is **task-correct Critical images delivered per decoder-second**.

---

## 2. Parameters: final set

| Parameter | Computed where | How | Kept / changed / dropped |
|---|---|---|---|
| **SPI posterior** p = (pC, pH, pM, pL) | UE | CLIP-B/32 image embedding → retrained, calibrated linear head (§4) | Kept, retrained |
| **Context relevance** r | UE | cos(CLIP(image), CLIP(mission paragraph)), quantised 8-bit | Kept |
| **Semantic hash** h (64 bit) | UE | SimHash: sign(R·e) with a fixed shared random matrix R (64×512) over the CLIP embedding e | **New**. Replaces UE-history novelty |
| **Receiver-side novelty** n | Receiver | n = min Hamming(h, H_mem)/64 over a ring H_mem of the last 512 decoded+queued hashes; duplicate if n ≤ τ (τ calibrated on CrisisMMD retweet duplicates) | **Changed**: computed at the scheduler from headers, no UE history |
| **STS** (SQ-GAN Semantic Token Summary) | UE | §3 | **New**. Replaces YOLO |
| **Predicted decode utility** û | UE | tiny head on the SQ-GAN pre-mask grid (teammate hook), trained on measured u | **New** |
| Capture time, payload length, codec config | UE | — | Kept |
| Source id, session id, COCO block, YOLO counts/boxes, UE history fields | — | — | **Dropped** |

---

## 3. Scene descriptor: SQ-GAN Semantic Token Summary (STS)

**What SQ-GAN gives for free** [V `sqgan.py:69-70`]: `masker(encoder(x,s), s)['sample_index']`, the
positions of the image tokens it keeps on the 16×32 grid. The same holds for `masker_semantic`. The
20-class map `s` is already computed. Whether relevance scores α are also in the returned dict is **[C]**
(Day 0). STS does not depend on them.

**STS fields (14 B):**
| Field | Bytes | Definition | What it tells the scheduler |
|---|---|---|---|
| Kept-token class histogram | 7 | Downsample s to 16×32 (majority vote). Fraction of kept tokens per superclass: person, vehicle, building, road, vegetation, sky, other | *What the payload actually carries* |
| Retention ratios | 4 | For person / vehicle / building / road: kept tokens of that class ÷ all tokens of that class | *How much of the important content survives at this m* |
| Spatial | 2 | Centroid of kept person+vehicle tokens (4b x, 4b y) + spread (entropy, 8b) | Concentrated object vs. diffuse scene |
| û | 1 | Predicted probability that the decoded image is task-correct | Whether decoding is worth the decoder time |

**Why STS is better than YOLO here:**
1. YOLO says what is in the image. STS says what is in the **payload**. The decoder can only show
   what was kept, so STS is the honest descriptor for a decoder-side scheduler.
2. It is free: no second network on the UE, unlike YOLO or Grounding-DINO.
3. It is ~14 B fixed instead of a variable 8–34 B sparse block.

**What STS cannot do (we say this in the paper).** The map is Cityscapes 20-class. It has no
rubble, flood, fire or "damaged building" class. STS therefore does **not** detect urgency; urgency comes
from SPI. STS adds content type, compressibility and decode-worthiness.

**How we prove it replaces YOLO** (one table): header variants
`{SPI only, SPI+YOLO (cached features), SPI+STS, SPI+teammate learned 8-B descriptor, SPI+STS+û, oracle}`
under the same policy. Metrics: CTR@D (§6) and Spearman ρ between û and measured u.

---

## 4. Header: fixed 64 bytes (big-endian)

| Offset | Size | Field |
|---:|---:|---|
| 0 | 1 | version (4b) \| flags (4b: SPI valid, STS valid, hash valid, learned-desc valid) |
| 1 | 4 | message id (random 32-bit from the UE; bookkeeping only, never scored) |
| 5 | 4 | capture time (ms, mod 2³²) |
| 9 | 2 | payload length (bytes) |
| 11 | 1 | codec config: m_x index (4b) \| m_s index (4b) |
| 12 | 1 | mission profile id (pre-shared paragraph) |
| 13 | 4 | SPI posterior, 4 × uint8 |
| 17 | 1 | context relevance, uint8 |
| 18 | 8 | 64-bit semantic hash |
| 26 | 14 | STS (§3) |
| 40 | 8 | learned SQ-GAN descriptor (teammate `scene_descriptor.py`, 8 × uint8; zero if unused) |
| 48 | 12 | reserved (zero) |
| 60 | 4 | CRC32 over header + payload |
| | **64** | |

Against the old format: the full trial header was 110.45 B on average, and 84.16 B after the COCO
removal (`work_record.md`). The new header is a fixed 64 B and still has 12 B spare. The implementation
template is the teammate's `parameters/spi_scene_packet.py` (versioned struct + CRC + profile
check) [V].

---

## 5. Dataset, labels and SPI retraining

### 5.1 Final pool [D]
| Source | Role | Size target | Labels |
|---|---|---|---|
| **CrisisMMD v2.0** (7 events: Harvey, Irma, Maria, California wildfires, Mexico earthquake, Iraq–Iran earthquake, Sri Lanka floods) | Disaster traffic | all labelled images (~18k [C]) | **Human**: informative, humanitarian category, damage severity |
| **Places365** (val split) | Everyday scenes | 6,000 | Low by rule |
| **Open Images** (person/portrait-heavy subset) | Selfies, people photos | 3,000 | Low by rule |

Why this set:
1. Human labels give ground truth that is independent of the VLM.
2. CrisisMMD comes from tweets, so each image has a tweet id. Twitter ids encode the post time, which
   gives **real arrival chronology, real bursts and real retweet duplicates** [C: confirm ids present].
3. Its seven events allow **event-disjoint** splits.
4. It is street- and phone-level imagery, which is closer to Cityscapes than aerial LADI.
5. You already used it for SPI.

LADI is dropped (aerial; out of distribution for a Cityscapes-trained codec). MEDIC is dropped for the
4-day scope (its download was never completed, per `KAGGLE_RUNS.md`, and it largely overlaps
CrisisMMD).

### 5.2 Splits [D]
- **Train events:** Harvey, Irma, California wildfires, Iraq–Iran earthquake.
- **Val:** Sri Lanka floods.
- **Test:** Maria + Mexico earthquake.
- Normals: random 60/10/30, after near-duplicate removal by hash.

This is changeable on Day 0 once per-event counts are known. Rule: test must hold ≥ 150 Critical.

### 5.3 Tier mapping from human labels [D, published; one alternative reported]
Precedence top-down: the first rule that matches wins.
| Tier | Rule |
|---|---|
| Critical | humanitarian ∈ {injured_or_dead_people, missing_or_found_people} **or** damage_severity = severe |
| High | humanitarian ∈ {affected_individuals, infrastructure_and_utility_damage, vehicle_damage} **or** damage_severity = mild |
| Medium | informative and humanitarian ∈ {rescue_volunteering_or_donation_effort, other_relevant_information} |
| Low | not informative / not_humanitarian, **and all Places365 / OpenImages images** |

Alternative mapping for sensitivity: affected_individuals → Critical.

**Role of Qwen labels:** audit only. Report Cohen's κ between Qwen's existing labels (3,933 cached) and
the human-rule tiers on overlapping hashes. No new VLM GPU time is needed.

### 5.4 SPI retraining (fast, about 1 GPU-hour)
1. Extract CLIP-B/32 image embeddings for the whole pool once (T4, < 1 h). Cache them, then compute the
   64-bit hashes from the same embeddings.
2. Train a multinomial logistic regression with class-balanced weights on train events + train
   normals. Choose C on val.
3. Apply **temperature scaling** on val. The policies use posteriors, so calibration matters.
4. Report on the test events: macro-F1, per-class recall (**Critical recall is the headline**),
   confusion matrix, ECE.
5. Ablation answering your concern: the same head trained **without** normals, tested on a mix with
   normals. We expect a false-Critical flood on selfies; this justifies the mixed pool.

Optional if Day 1 has slack: CLIP-L/14 head (larger UE cost; report as an upper reference).

---

## 6. Objective and evaluation metrics (beyond delay)

**Timely semantic value** of decoding job i, finished at time t:

  v_i(t) = w_{k_i} · u_i · g_{k_i}(t − a_i) · (1 − dup_i)

- k_i = **true** tier (human labels); weights w = (8, 4, 2, 1) for (C, H, M, L) [D, swept].
- u_i ∈ {0, 1}: task-correct after decode. An independent receiver model (CLIP-L/14 linear probe on
  human tiers, a **different backbone** from the UE's SPI) gives the same tier on the decoded image as on
  the original. Computed only on images whose original is classified correctly. LPIPS and CLIP-cosine
  are reported alongside.
- g_k: deadline curve. 1 up to the soft deadline D_k, linear decay to 0 at 2D_k.
  D = (5, 15, 60, 300) s [D, swept ×0.5 / ×2].
- dup_i = 1 if a near-duplicate (same hash cluster) was already delivered. Only the first copy counts.

**Primary metrics:**
1. **CTR@D (Critical Timely Recall):** the share of true-Critical images that are decoded,
   task-correct and within D_C. This is the headline. It does **not** depend on the chosen weights.
2. **Normalised timely value:** Σv / Σv(oracle).
3. **Decoder efficiency:** task-correct timely Critical deliveries per decoder-GPU-second.

**Secondary metrics:** per-tier within-deadline rate; p50/p95 sojourn per tier; Low-tier starvation
(share of Low images never decoded); redundant decodes per 100 decodes; decoder utilisation;
scheduler decision time (µs) and header bytes; the legacy waiting penalty J_D (with true tiers) for
continuity with the old results.

**Statistics:** test events only. ≥ 10 independent arrival realisations per configuration. Paired
comparisons on identical traces. 95% CIs by bootstrap over **images/events**, not seeds.

---

## 7. Scheduler simulator: review and v2 spec

### 7.1 Is the current simulator good?
| Aspect | `Evaluation/semantic_learning_env.py` (main) | Teammate staged replay (`scheduler_codec_replay.py`) | Literature practice | v2 |
|---|---|---|---|---|
| Service time | 1.0 (or lognormal toy) | **Measured per image** ✔ | Measured / deterministic per model (2605.05514 deterministic; EdgeServing measured) | Measured T4 time per decode mode |
| Servers | 1 | 1 | Multi-server (M/G/N + M in 2609.19792; multi-GPU serving) | c ∈ {1, 2, 4} |
| Arrivals | uniform over a window, first 8 at t=0, i.i.d. images | synthetic loads 0.6/0.9/1.2 + bursts ✔ | Poisson baseline + bursty (MMPP) | Poisson for normals; **tweet chronology** (time-compressed) for disaster images; MMPP burst option |
| Episode length | 48 messages | 24 offers (6 images × 4) | Long-run steady state | ≥ 20k arrivals, 10% warm-up discarded |
| Channel / loss | none | ideal delivery | One-shot, no retransmission (2605.05514) | Rayleigh outage + airtime |
| Pipeline stages | none | capture/sender/arrival/queue/decode/decision ✔ | — | Reuse teammate stages |
| Deadlines / abandonment | budget only inside the cost | fixed deadlines ✔ | Impatient customers (2609.19792); deadline-aware dropping (EdgeServing) | Deadline curve + purge action |
| Outcome after decode | none | reader on decoded image (failed) | Task accuracy vs. rate (SQ-GAN; 2605.05514) | u_i lookup from measured reconstructions |
| Redundancy | exact copies only | — | Redundancy-aware uplink (2606.31715) | Hash-based, real retweet duplicates |
| UE identity / history | used (rings per source) | — | Not available under our model | **Removed** from the scheduler's view |

**Verdict:** main's simulator is not usable for the paper (unit service, i.i.d. images, 48-job
episodes, label-weighted cost). The teammate's replay fixed the most important engineering gap,
measured per-stage timing. It is our base, but it had no outcome model, no channel and tiny cohorts.
v2 = teammate's staged timing + the rows above.

### 7.2 v2 file layout (new code, canonical owners)
- `Evaluation/edge_decoder_env.py`: event-driven simulator (arrivals, uplink outage, queue, c workers,
  purge, outcome lookup, metrics). NumPy only, vectorised.
- `Evaluation/semantic_header64.py`: 64-B pack/unpack/CRC (adapted from `spi_scene_packet.py`).
- `Scheduler/edge_policies.py`: all baselines + SV-DPP + oracle.
- `Scheduler/training/sc_ppo.py`: RL.
- `parameters/sts_extract.py`: STS + hash + û extraction (extends `sqgan_prefeatures.py` hook).

---

## 8. Scheduling policies (final)

All policies receive the **same header information**. They differ only in how they use it. Whenever
a decoder worker frees, the policy picks one queued job.

| Id | Policy | Rule | Why it is in the paper |
|---|---|---|---|
| B0 | FIFO | earliest arrival | No-information floor |
| B1 | EDF | smallest deadline of argmax-SPI tier minus age | Classical real-time baseline |
| B2 | SPI-first | strict priority by argmax SPI, FIFO within tier | The old strong baseline |
| B3 | SPI-first + EDF | strict priority, EDF within tier | Stronger version of B2 |
| B4 | Expected cost-of-delay (cμ) index | index_i = [E v̂_i(t+S) − E v̂_i(t+2S)] / S, expectation over the SPI posterior, û, novelty | Generalised cμ: the classical near-optimal rule for convex class-dependent costs (van Mieghem; Whittle extension 2609.19792). This explains *why* old RL lost |
| B5 | LambdaMART | retrained on train events, target = true marginal value | Best old learned method, now evaluated fairly |
| **P1** | **SV-DPP** (Semantic-Value Drift-Plus-Penalty) | see below | **Proposed, analytic** |
| **P2** | **SC-PPO** (Semantic Constrained PPO) | see below | **Proposed, learned** (your own RL, reward, loss) |
| O | Oracle | P1 with true tier and true u | Upper bound; the gap shows how much a better header would buy |

### 8.1 P1: SV-DPP (adapted from Sagduyu & Erpek, 2605.05514)
Their paper puts a virtual queue on a long-term semantic-error constraint and selects the action that
minimises drift-plus-penalty. Our constraint is on missed Critical deadlines.

- Virtual queue, updated at each decision epoch:
  Z ← max(Z + M̂_C − ε·Â_C, 0). M̂_C = predicted-Critical jobs (pC ≥ 0.5) whose deadline expired
  undecoded since the last epoch; Â_C = predicted-Critical arrivals. Fully causal, with ε = 0.05 target.
- Score for each queued job i:
  **score_i = V·ΔV_i + Z·pC_i·urg_i**, with ΔV_i the B4 cost-of-delay index and
  urg_i = 1/(1 + slack_i/S).
- Pick argmax. **Purge** any job with E v̂_i(t+S) = 0 (expired or duplicate).
- Properties: O(queue) per decision (µs). The standard DPP result is an O(1/V) optimality gap with an
  O(V) virtual-queue backlog, which holds the *time-average* Critical-miss rate at ε. We state this
  honestly; it is not a per-image guarantee.
- Sanity theorem/check: with ideal channel, unit service, no duplicates and Z ≡ 0, SV-DPP reduces to
  cμ ≈ SPI-first. This **reproduces the old result as a special case**, which is a strong narrative
  line.

### 8.2 P2: SC-PPO (time-boxed: ≤ 6 GPU-h)
- **Why RL can win now, when it could not before:** the old reward was a function of the same label the
  sort used. Now the training reward uses the **true** outcome (human tier, measured u, duplicates),
  available in simulation on train events only. The header is noisy (SPI is miscalibrated on new
  events), so an index that trusts SPI is no longer optimal. RL can learn when to distrust SPI, using
  STS, û, relevance and novelty. Bursts and multi-worker interactions also break the index's
  optimality.
- **Observation:** a set of per-job vectors x_i = [p(4), r, n, STS(14), û, age_i/D_k (4), slack_i] plus
  globals g = [queue length, Z, busy workers, recent arrival rate].
- **Network:** shared per-job MLP φ (2 × 64) → attention pooling → logit_i = ψ(φ_i, pool, g).
  Softmax over queued jobs. Permutation-invariant and handles any queue length.
- **Reward** per decision: r = v_i^true(t_finish) − λ·(true-Critical deadline misses since the last
  decision).
- **Loss:**
  L = L_clip(PPO) + 0.5·L_value − 0.01·H(π) + β·KL(π ‖ π_SV-DPP).
  - β starts at 1.0 and anneals to 0 over the first 20% of updates (warm start from the analytic policy).
  - Lagrange multiplier: λ ← max(0, λ + η·(CMR_batch − ε)) once per update (primal-dual constrained RL).
- **Purge** uses the same rule as P1, so both are judged on ordering.
- **Rule:** if SC-PPO does not beat SV-DPP on CTR@D with separated CIs, we report it in one paragraph
  as "analytic index near-optimal" and keep SV-DPP as the main method.

### 8.3 Ablations (each one a single table row)
- **Header:** SPI only / +relevance / +novelty / +STS / +û / YOLO instead of STS / oracle.
- **Purge:** on / off for B4 and P1.
- **UE rate rule:** SPI-adaptive m vs. fixed m = 0.25.
- **SPI trained** with vs. without normals.
- **Deadline / weight tables:** ×0.5, ×2.
- **Sweeps:** load ρ ∈ {0.6, 0.8, 1.0, 1.2, 1.5}; workers c ∈ {1, 2, 4}; disaster share
  π_d ∈ {0.2, 0.5, 0.8}.

---

## 9. Four-day execution plan (with gates)

Accounts: **A** = Windows Kaggle CLI, **B** = WSL Kaggle CLI. Reuse existing notebook slugs as new
versions, per AGENTS.md. "You" and "Teammate" are suggested owners.

### Day 0 (today, 3 h): verify before building
| # | Task | Owner | Output | Gate |
|---|---|---|---|---|
| 0.1 | Confirm CrisisMMD v2 images + label TSVs + tweet ids reachable on Kaggle; per-event and per-tier counts under §5.3 | You (B) | `counts.json` | test events ≥ 150 Critical, else re-pick events |
| 0.2 | Places365-val and OpenImages subsets reachable | You (B) | file list | — |
| 0.3 | SQ-GAN on T4: 20 images → full-decode s/img, semantic-only s/img, encoder s/img; masker dict keys (scores?) | Teammate (A) | `sqgan_t4_bench.json` | full decode measured; if > 1 s/img, cap the profiled pool at 4k |
| 0.4 | Parity re-run (teammate's `prefeature_parity_smoke.py`) on T4 | Teammate (A) | receipt | identical payload |

### Day 1: features and labels (all GPU work)
| # | Task | Account | Detail | Gate |
|---|---|---|---|---|
| 1.1 | CLIP-B/32 embeddings + 64-bit hashes + relevance for the whole pool | B | single pass, cached `.npz` | — |
| 1.2 | SPI retrain + temperature scaling + test report (§5.4); κ vs. Qwen | local CPU | ~30 min | **Critical recall on test events ≥ 0.6**; else try CLIP-L/14 |
| 1.3 | SegFormer + SQ-GAN encode at m_x ∈ {0.25, 0.55} on the **test + val pools**; STS from `sample_index`; payload bytes; decode → reconstructions | A | extends `sqgan_prefeatures.py` hook | — |
| 1.4 | Receiver model (CLIP-L/14 probe on human tiers, train events) → u_i on reconstructions; LPIPS | B | — | **u not ≈ 0 everywhere, and higher at m=0.55.** If it fails: JPEG ladder becomes the codec and SQ-GAN is a secondary row (the paper still stands) |
| 1.5 | û head (teammate descriptor retargeted, MSE/BCE on u) | local | 8-B descriptor + û | Spearman ρ(û, u) reported |

### Day 2: simulator + policies
| # | Task | Owner |
|---|---|---|
| 2.1 | `semantic_header64.py` (pack/unpack/CRC, round-trip test) | Teammate |
| 2.2 | `edge_decoder_env.py` (§7) on top of the teammate's staged timing | You |
| 2.3 | B0–B5, P1, Oracle in `edge_policies.py`; sanity check that P1 reduces to SPI-first in the degenerate case | You |
| 2.4 | `sc_ppo.py`; start training on account A/B overnight (≤ 6 GPU-h) | Teammate |
| Gate | P1 beats B3 and B4 on CTR@D at ρ ≥ 1.0 with CI separation on **val** events. If not, inspect the header ablation first (likely SPI calibration) | |

### Day 3: experiments
Run the full sweep (§8.3) on **test events** once the policy set is frozen. Evaluate SC-PPO. Produce
figures: CTR@D vs. load; normalised value vs. load; decoder efficiency bar chart; header-ablation
table; per-tier p95 sojourn; Low starvation; CDF of Critical sojourn; decision-time table.

### Day 4: writing
System model (Fig. 1), problem as a constrained MDP, SV-DPP derivation and bound, SC-PPO, datasets and
labels, results, and limitations: Cityscapes-class STS, SQ-GAN out of distribution, tweet-time arrivals
as a proxy for field traces, tier rule as a design choice. Realistic target: a complete letter
(IEEE WCL/CL) or the TCCN-scope evidence base. A polished full journal draft needs about 1–2 more weeks.

---

## 10. Literature anchors (re-verify each 26xx ID before citing)
- Queueing + semantic rate + DPP + one-shot: Sagduyu & Erpek, 2605.05514.
- SQ-GAN (SAMM masking, m_x / m_s runtime knobs, Cityscapes): Pezone et al., 2502.09520 (IEEE TCCN).
- Whittle / cμ for convex delay costs with abandonment: 2609.19792.
- Deadline-aware edge DNN serving: EdgeServing 2605.05527; priority-aware inference serving: Strait
  2604.28175; RL for hard/soft real-time DNN inference: 2610.29503; priority under excess demand:
  Inference Auctions 2609.40070.
- Importance-aware RA for task-oriented SemCom: 2606.29052; semantic spectral efficiency: Yan, Qin et
  al., IEEE WCL 2022.
- Semantic / version AoI and AoII: 2606.31303, 2601.18069, 2610.22265.
- Redundancy-aware uplink: 2606.31715.
- Disaster imagery: CrisisMMD (Alam et al., ICWSM 2018), MEDIC 2108.12828.
