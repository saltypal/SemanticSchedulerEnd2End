# Plan v2 explained, with decisions v2.1

Companion to `FINAL_PLAN_V2.md`. It answers every question from the 2026-11-10 discussion in the order
they were asked. Where the discussion changed a decision, the change is marked **[v2.1]** and is
authoritative over `FINAL_PLAN_V2.md`. Tags: **[V]** = verified in code or files, **[C]** = to
check, **[D]** = our documented design choice.

---

## 1. The queueing model, in plain words

**Analogy: a hospital emergency desk.**
- Patients = images arriving from many UEs.
- Wristband = the 64-byte header.
- Triage nurse = the scheduler. The nurse reads only the wristband and never examines the patient
  (never decodes the payload).
- Doctors = decoder workers (GPUs running the SQ-GAN decoder). Doctors are few and each one takes time.
- The nurse decides who sees a doctor next, who can wait, and who should be sent home (purged).

**Formally.** Images arrive as a random process: Poisson for everyday photos, real tweet timing for
disaster photos. Each brings a header. They wait in **one shared queue**. There are **c decoder
workers**. When a worker becomes free, the scheduler picks one queued image. That worker is busy for
the image's decode time S (measured on a GPU), then picks again. In queueing notation this is a
**multi-class G/G/c queue with deadlines and dropping**:
- "multi-class": images differ in importance;
- "G/G": general arrival and service distributions;
- "c": number of servers;
- "deadlines and dropping": an image that waits too long loses its value and may be purged, which the
  literature calls "impatient customers" or "abandonment".

**What is new compared with the old simulator:**
| Old | New | Why it matters |
|---|---|---|
| Service time always 1 | Measured SQ-GAN decode time | Real generative decoders are slow; that is the bottleneck we schedule |
| One server | c ∈ {1, 2, 4} workers | An edge site has a few GPUs or model replicas |
| 48 images per episode | ≥ 20,000 arrivals, first 10% discarded as warm-up | Steady-state statistics, not start-up artefacts |
| Random unrelated images, fake uniform times | Real tweet chronology for disaster images (bursts, retweets) + Poisson everyday images | Realistic surges and duplicates |
| Only order is decided | decode / defer / purge | The scheduler protects the decoder from useless work |

**"Multi-server / multi-GPU serving".** c identical workers share one queue, like several doctors
sharing one waiting room. The edge-inference papers do the same for DNN requests:
- EdgeServing (2605.05527) schedules several DNN models sharing one GPU under deadlines.
- Strait (2604.28175) adds request priority and GPU interference to inference serving.

(Both are summarised from their abstracts; read the full papers before citing details.)

---

## 2. What the Sagduyu & Erpek paper does (2605.05514)

- **Data:** CIFAR-10 images (32×32, 10 classes).
- **Model:** a learned semantic autoencoder that sends an image over a Rayleigh-fading AWGN channel.
  The receiver produces two things: a reconstructed image and a class label.
- **Knob:** the latent dimension N, i.e. how many channel symbols per image. Larger N gives better
  accuracy but a longer service time. Accuracy saturates as N grows.
- **Queue:** a single-server queue with Poisson arrivals and deterministic service time S = N. There is
  no retransmission: each image is sent once.
- **Goal:** minimise average delay (or Age of Information) subject to long-run semantic error ≤ ε.
- **Method: Lyapunov drift-plus-penalty with a virtual queue.**
  - A virtual queue Z counts "error debt": Z ← max(Z + E_k − ε, 0), where E_k = 1 if image k was
    misclassified.
  - At every service start the controller picks N ∈ {5, 10, 15, 20} by minimising Q·N + V·Z·p̂_e(N).
  - Q = backlog, so a long queue pushes N down for speed.
  - Large debt Z pushes N up for accuracy. V trades delay against fidelity.
- **Result:** the adaptive N beats every fixed N in delay and AoI at the same error cap, and widens the
  range of loads the system can handle.

**What we borrow and change:**
- **Borrowed:** the queue framing, one-shot transmission, a semantic quality constraint enforced
  by a virtual queue, and the drift-plus-penalty decision rule.
- **Changed:**
  - Their decision is how many symbols to send. Ours is **which image to decode next, and whether
    to purge it**.
  - Their constraint is average classification error. Ours is the **rate of missed Critical
    deadlines**.
  - They have one class of images. We have four importance tiers, a noisy header and duplicates.

---

## 3. Where the scheduler runs, and O-RAN (for the paper's system section)

- **UE:** runs the encoder (SegFormer → SQ-GAN encoder) plus CLIP for SPI, relevance and hash. It
  sends `[header][payload]` once.
- **Network:** **[v2.1]** treated like the internet, as you said: each UE has its own pipe, and **no
  radio scheduling** is modelled. Packets arrive after a network delay. As a sensitivity row only, we
  drop a small fixed fraction (0–5%) to show that the one-shot assumption (no retransmission) does not
  change the ranking of policies. The fading and outage model in `FINAL_PLAN_V2.md` §1 is **removed**.
- **Edge site:** O-Cloud compute next to the O-RAN distributed unit / central unit (O-DU / O-CU).
  It hosts:
  1. the header parser;
  2. the receiver queue;
  3. the scheduler;
  4. the SQ-GAN decoder pool;
  5. the downstream task (responder dashboard or analytics).
- **O-RAN framing for the paper:**
  - The scheduler's *policy* (weights, ε, mission registry) is managed from the near-RT RIC as an
    xApp. Its control loop of 10 ms–1 s matches decode timescales.
  - The policy can be updated over the A1 interface from the non-RT RIC (for example, "a flood event
    is declared in region X: raise flood-mission weights").
  - Per-image decisions run inside the edge application itself.

  This is a clean split: RIC = slow policy, edge app = fast decisions.

**Is this "task-aware"?** Yes, in a specific sense. The receiver-side task (responders need Critical
evidence fast) shapes what the decoder spends compute on. The codec itself is unchanged. The
scheduler makes the *decoding service* task-aware. That sentence goes in the abstract.

---

## 4. Labels: your mapping, finalised **[v2.1]**

You proposed using CrisisMMD's damage severity directly. I agree. It is simpler, it needs no VLM, and
it is defensible. CrisisMMD's damage-severity values are **severe / mild / little_or_none** (with
"dont_know" in some versions [C]). Your mapping, plus two small safety rules:

| Our tier | Rule (top-down; first match wins) |
|---|---|
| **Critical** | damage = severe **or** humanitarian ∈ {injured_or_dead_people, missing_or_found_people} *(life-safety override)* |
| **High** | damage = mild |
| **Medium** | damage = little_or_none **and** informative (disaster-related, no visible damage: flooded street with no destruction, shelters, relief supplies) |
| **Low** | not informative (memes, screenshots, unrelated photos inside disaster tweets) **and all everyday images** (Places365, people photos) |

**Why the two extra rules.**
1. **Life-safety override.** An image of injured people with no building damage would otherwise fall
   to Medium. Every triage system ranks human casualties at the top.
2. **Not-informative → Low.** Many tweet images are memes or text screenshots. Calling them Medium
   would pollute the disaster tiers.

Both are one-line rules over **human** labels. **No one hand-labels anything**: the CrisisMMD
annotators already did it.

**How to justify the mapping (real frameworks; no invented story needed).** A reviewer will check,
so we justify with frameworks that exist:
- **FEMA Preliminary Damage Assessment** categories: Destroyed / Major / Minor / Affected. Severe
  and mild map naturally onto the urgent tiers.
- **START mass-casualty triage** (immediate / delayed / minor): life-threatening human cases first.
  That is the life-safety override.
- **Information priority in crisis informatics.** CrisisMMD itself separates *informative* from
  *not informative*, precisely because responders cannot use non-informative posts.

Paper wording: "We derive a four-tier urgency scale from CrisisMMD's human damage-severity and
humanitarian annotations, aligned with damage-assessment and triage practice, and assign all
non-disaster imagery to the lowest tier." Plus one sensitivity row with an alternative mapping:
affected_individuals → High.

**Everyday images (Low).**
- **Places365** is mostly scenes (kitchens, streets, beaches) with few selfies. Some categories
  *look* like damage: ruin, junkyard, landfill, construction_site, and possibly others [C, scan the
  category list]. We **exclude a small blacklist** so a "ruin" photo is not labelled Low while
  looking like rubble.
- **People / selfie photos:** an Open Images subset with person or face boxes. Optionally add a
  public selfie dataset [C, licence].
- These Low images are **in training too** (your point), so SPI learns that a selfie is not an
  emergency. The "SPI without normals" ablation shows what goes wrong otherwise.

**Size [C].** CrisisMMD v2.0 has about 18k images from 7 events. The archive is on the order of a few
GB. The local `Mini_dataset_6000.tar` already exists
(`experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/`), but the full set is easiest on
Kaggle. Decision: **Kaggle for the full set**; local only for the 6k mini set and analysis.

**Is the VLM still used?** Not for ground truth. The old Qwen labels (3,933) become an
**agreement audit** (Cohen's κ vs. the human-derived tier). The VLM comes back for RLAIF, as a source
of *training feedback* (§11).

---

## 5. Receiver-side novelty and the semantic hash (dedicated paper section)

**The problem.** UEs keep no history, and the scheduler ignores UE identity. But disasters
generate floods of **repeated content**: retweets, the same collapsed bridge shot by 20 people, a
drone filming the same roof twice. Decoding all of them wastes the scarce decoder. Only the
**receiver** knows what it has already decoded, so novelty must be measured *there*.

**The tool: SimHash, a locality-sensitive hash (Charikar 2002).**
1. The UE already computes a CLIP embedding e (512 numbers) for SPI.
2. Both sides share a fixed random matrix R of 64 rows × 512 columns, fixed once in the standard,
   like a codebook.
3. Hash bit j = 1 if (row j of R) · e > 0, else 0. That gives 64 bits = 8 bytes.

**Why 64 bits can say something useful.** Each bit asks "which side of a random plane is this
image on?" For two images whose CLIP embeddings are at angle θ, a bit differs with probability
exactly θ/π (Goemans–Williamson / Charikar). The Hamming distance d between two hashes is therefore
Binomial(64, θ/π).
- **Near-duplicates** (cos ≈ 0.95 → θ ≈ 18°): expected d ≈ 6.4, sd ≈ 2.4.
- **Unrelated photos** (cos ≈ 0.5 → θ ≈ 60°): expected d ≈ 21, sd ≈ 3.8.

Those two distributions barely overlap, so a threshold of around d ≤ 11 separates them well. The
cosine values are typical for CLIP but must be **calibrated on CrisisMMD** [C]. A 256-bit hash halves
the noise but costs 32 B. 64 bits is enough for duplicates. The paper reports the bits-vs-accuracy
curve (32/64/128/256) as one figure. The hash does **not** describe content. It answers one question
very cheaply: *have I already got something like this?*

**Receiver-side novelty algorithm:**
```
memory = ring buffer of (hash, value_delivered) for the last M = 512 decoded images
pending = hashes of images currently queued

on arrival of header h:
    d_dec = min Hamming(h, memory)           # closest thing already decoded
    d_q   = min Hamming(h, pending)          # closest thing waiting in the queue
    novelty n = d_dec / 64                   # 0 = seen before, ~0.33+ = new content
    if d_dec <= tau_dup:   mark duplicate -> value x 0 -> purge candidate
    elif d_q <= tau_dup:   keep only the better copy (higher SPI, then higher û); purge the other
    elif d_dec <= tau_sim: similar but not identical -> value x n_scaled (partial novelty)
```
Thresholds τ_dup and τ_sim are calibrated on CrisisMMD retweet and near-duplicate pairs (CrisisMMD
tweets include many reposted images [C]).

**Why this is a contribution (paper wording).** "Receiver-side semantic novelty without UE state: an
8-byte LSH of the sender's semantic embedding lets a stateless decoder-side scheduler detect
cross-UE redundancy and purge it before spending decoder compute."

**Related work to cite:**
- LSH: Indyk & Motwani 1998; Charikar 2002.
- Semantic deduplication with CLIP-like embeddings: SemDeDup, Abbas et al. 2023.
- Redundancy-aware uplink for overlapping fields of view: 2606.31715.

**Ablations:**
- hash vs. no novelty;
- hash bits 32/64/128/256;
- random SimHash vs. a learned projection (PCA-ITQ, Gong & Lazebnik 2011), as a stretch.

**Side benefit:** a hash reveals far less than an embedding, which helps privacy. This gets one
sentence in the paper, not a claim.

---

## 6. SQ-GAN in five steps, and the STS (Semantic Token Summary)

**How SQ-GAN works, simply** [V `sqgan.py` + paper 2502.09520]:
1. The image is resized to 256×512. **SegFormer** colours every pixel with one of 20 classes (road,
   building, person, car, vegetation, sky, …). This **semantic map s** is required input.
2. The **encoder** cuts the image into a grid of **16 × 32 = 512 tiles**, each 16×16 pixels. Every
   tile becomes a "token", a vector of 256 numbers. The map s is encoded the same way into its own 512
   tokens.
3. **SAMM** (the semantic adaptive mask) gives every tile an importance score, guided by the map
   (people and cars matter more than sky). It **keeps only the top fraction m**. For example, m = 0.25
   keeps 128 of 512 tiles.
4. Each kept token is replaced by the number of its closest entry in a **1,024-word codebook**
   (10 bits). The payload = which tiles were kept + their codebook numbers, for both image and map.
   That is how an image fits in about 464 B.
5. The **decoder** first rebuilds the semantic map, then rebuilds the image guided by that map. The
   dropped tiles are **filled in (hallucinated) by the GAN**.

**The key insight for scheduling.** The kept tiles are real information. The dropped tiles are the
GAN's imagination. If a person's tiles were dropped, the decoded image may show a plausible but
*invented* person, or none at all. The scheduler should know **how much of the important content is
real and how much will be imagined**. YOLO cannot tell you that. The codec's own mask can, for free.

**"Downsample s to 16×32 by majority vote".** Each tile covers 16×16 = 256 pixels of the map. The
tile's class is simply the most frequent class among those 256 pixels. Then every tile has one class
label, aligned with the token grid.

**STS pseudo-code (runs on the UE, after encoding; no extra network):**
```python
def sts(seg_map_20cls, kept_idx, m):           # seg_map: 256x512 ints; kept_idx: list of kept tile ids (0..511)
    tiles = majority_vote(seg_map_20cls, tile=16)   # -> 16x32 class per tile
    sup = SUPERCLASS[tiles]                       # 20 Cityscapes classes -> 7: person, vehicle, building, road, vegetation, sky, other
    kept = zeros(16*32, bool); kept[kept_idx] = True

    hist = [mean(sup[kept] == c) for c in range(7)]               # what the payload carries (7 bytes)
    retention = [sum(kept & (sup == c)) / max(1, sum(sup == c))    # share of each key class that is REAL, not imagined
                 for c in (PERSON, VEHICLE, BUILDING, ROAD)]       # (4 bytes)
    ys, xs = divmod(where(kept & isin(sup, [PERSON, VEHICLE])), 32)
    centroid = (mean(xs)/32, mean(ys)/16) if len(xs) else (NA, NA) # where the important objects are (1 byte, 4b+4b)
    spread = entropy_of_positions(kept_idx)                        # one object vs scattered scene (1 byte)
    u_hat = utility_model(hist, retention, spread, m)              # predicted decode utility (1 byte), see section 7
    return quantize_u8(hist + retention) + pack(centroid, spread, u_hat)   # 14 bytes total
```
Data flow: SQ-GAN's `masker(...)['sample_index']` gives `kept_idx` [V `sqgan.py:69`]. The map is
the SegFormer output, already computed. The teammate's forward-hook pattern
(`sqgan_prefeatures.py`) captures it in the same single encode pass.

**Why it is "codec-aware":** every STS number is computed **on the kept tiles**, which are exactly
what the decoder will reproduce faithfully. Change m, or change the codec, and STS changes with it.
YOLO features would stay identical, even though what reaches the receiver is completely different.

**Honest limitation:** the classes are Cityscapes (driving scenes). There is no water, fire or rubble
class, so STS describes *content type and reconstruction trustworthiness*, not urgency. Urgency comes
from SPI. Retraining SQ-GAN can fix the class list (§12).

**Paper section "Scene descriptors" (as you asked).** Tell the story in order:
1. **YOLO** (COCO 80 classes). Old 387-column scene vector; scene block averaged 34.45 B, max 307 B
   dense [V `work_record.md`].
2. **COCO removal.** Scene block 8.16 B, full header 110.45 → 84.16 B (−23.8%) [V].
3. **Grounding-DINO / open-vocabulary detection.** Prompt-able disaster classes, but a heavy second
   network on the UE. Earlier grid runs measured cost but had no box ground truth [V `KAGGLE_RUNS.md`].
4. **STS.** A fixed 14 B, about zero extra compute, codec-aware.

One table compares header bytes, UE compute (ms), and scheduler gain (CTR@D) for {none, YOLO,
Grounding-DINO, STS}. The YOLO features are already cached. Grounding-DINO features for the test pool
cost one T4 run [C, check whether cached].

---

## 7. Predicted decode utility (û), and how it differs from the learned descriptor

**u (measured, offline).** Decode an image, then ask an independent receiver classifier (a
different backbone from the UE's SPI) for its tier. If it gives the same tier as on the original
image, u = 1; otherwise u = 0. Plain meaning: *did the decoded image still say the right thing?*

**û (predicted, sent in the header).** The UE cannot decode its own image cheaply, so it estimates
the chance that the decoded image will still say the right thing, from things it already has
(STS, SPI, m). This is a small logistic regression trained offline on measured u. It is sent as
1 byte.

**What the scheduler does with it:**
- It does not spend a decoder slot on an image that will come out useless (low û, for instance
  when most of the person tiles were dropped).
- Between two duplicates, it decodes the one with higher û.

**Learned 8-B descriptor (teammate's `scene_descriptor.py`) vs. STS:**

| | STS | Learned descriptor |
|---|---|---|
| What | Hand-defined, interpretable numbers from the codec's kept tiles | A neural layer squeezes the encoder grid into 8 numbers |
| Needs training labels? | No | Yes (originally VLM pair rankings, which never stabilised) |
| Explainable in the paper? | Yes, every byte has a meaning | Partly |
| Role | **Core header field** | **Ablation only**: stored in the reserved bytes, retrained to predict u instead of rankings |

**[v2.1] decision:** the learned descriptor leaves the mandatory header (see §9).

---

## 8. The UE rate rule and "semantic-map-only decode" **[v2.1]: both removed from the core**

- **UE rate rule** (UE chooses masking fraction m from its own SPI). Under your network model each UE
  has its own pipe and bandwidth is not scarce, so this rule has no tradeoff to win. Decision:
  **fixed m per device profile** (default 0.25), recorded in the codec-config byte. A single
  ablation row ("Critical images sent at m = 0.55") stays, because it shows how STS and û react to m.
- **Semantic-map-only decode.** SQ-GAN's decoder runs in two stages: first it rebuilds the coloured
  segmentation map (where road, buildings and people are), then the photo. "Map-only" would stop
  after stage one: a cheap, rough layout for quick triage. With Cityscapes classes that layout cannot
  show damage, so its value is doubtful. Decision: **future work**. We only measure its time on
  Day 0, for one sentence in the discussion.

---

## 9. The 64-byte header: final layout and the reason for every byte **[v2.1]**

**How we choose bit-depths: by measurement, not taste.** The 2026-10-25 work found that **8-bit
quantisation sometimes changed rankings, while 16-bit preserved them** (`digest_codex_1025`).
Rule: for each numeric field, pick the **smallest bit-depth whose scheduling-decision flip rate is
below 0.5%** against full precision, measured on validation traces at 4, 6, 8 and 16 bits. The
layout below reserves space for 8-bit; if a field needs 16-bit, it takes reserved bytes.

| Off | Size | Field | Why it exists / why this size |
|---:|---:|---|---|
| 0 | 1 | version (4 bit) + flags (4 bit: SPI / STS / hash / extension valid) | Lets the format evolve; flags let a codec without STS (e.g. SwinJSCC, JPEG) still send a valid header |
| 1 | 4 | message id | Bookkeeping, duplicate-arrival detection and logging; never scored. 32 bits: collision odds are negligible within a decoder's window |
| 5 | 4 | capture time (ms, mod 2³²) | Age = now − capture; deadlines are measured from capture, not arrival. Wraps every ~49 days, which is fine |
| 9 | 3 | payload length | 24 bits up to 16 MB, so JPEG/BPG controls (tens of kB) also fit; 16 bits would cap at 64 kB |
| 12 | 1 | codec config: codec id (4 bit) + m index (4 bit) | The decoder must know **which codec and which m** produced the payload. It enables a mixed fleet (SQ-GAN, SwinJSCC, JPEG) and the experiment "header value with and without STS across codecs" |
| 13 | 2 | mission profile id | Index into a pre-shared registry of mission paragraphs (§10). 16 bits = 65,536 profiles, since "thousands" was the requirement |
| 15 | 4 | SPI posterior (C, H, M, L) × uint8 | Calibrated probabilities feed the expected-value score. 8-bit resolution 0.004 is well below typical calibration error. Bit-depth is confirmed by the flip-rate test |
| 19 | 1 | context relevance (uint8) | CLIP similarity to the mission text; one byte is plenty for a ranking feature (flip-rate tested) |
| 20 | 8 | semantic hash (64-bit SimHash) | Receiver-side novelty (§5); 64 bits separates duplicates from new content |
| 28 | 14 | STS (incl. û) | §6–§7 |
| 42 | 2 | CRC-16 over **the header only** | See below |
| 44 | 20 | reserved / extension | Room for 16-bit fields if the flip-rate test demands them, the learned descriptor (8 B, ablation), or a 128-bit hash. Fixed size → constant-time parsing, a standard protocol practice (IPv4/TCP reserved bits) |
| | **64** | | |

**Why keep a CRC at all, in semantic communication?** Because the **header drives decisions**.
One flipped bit in the SPI byte can promote a selfie to Critical or bury a casualty image. With no
retransmission, the receiver must at least *detect* a corrupted header and fall back (treat it as
"SPI unavailable" → default tier). **[v2.1] change:** CRC-16 over the header only (2 B, not CRC-32
over header + payload). Payload integrity is left to the transport (UDP checksum / link-layer CRC)
and to the SQ-GAN bitstream parser. That saves 2 B and is easier to defend.

---

## 10. Mission profiles

A Haiku agent is compiling a draft catalogue of 5G/6G device families from 3GPP documents (UAV,
V2X/ADAS, public-safety, industrial, consumer, NTN …), with sub-types such as disaster-recovery,
surveillance, cinematography, agriculture and inspection drones, and a mission paragraph each. Output:
`docs/reviews/MISSION_PROFILES_DRAFT.md`. We review it together. Then the registry gets
24–32 profiles, each with an expected traffic mix (mostly urgent vs. mostly everyday). The simulator
draws UE profiles from it.

---

## 11. Objective, metrics and policies

### 11.1 What the old objective was, and why it is gone
Old "class-weighted waiting penalty":
- Each image gets a budget from its tier (2/4/8/16 units).
- Penalty = wait/budget, with the slope doubled after the budget, summed over images.
- **Problems:**
  1. It only counts waiting. A decoded image that came out useless still scores perfectly.
  2. The tier comes from the same label the policy sorts on, which is circular.
  3. Its optimum is a known simple sort (the cμ rule), so learning cannot win.

### 11.2 Timely semantic value: definition and justification
For an image i decoded and finished at time t:

  **v_i = w(tier_i) × u_i × g(t − capture_i) × (1 − dup_i)**

| Factor | Meaning | Why it belongs | Inspiration |
|---|---|---|---|
| w(tier) | how much responders care (8/4/2/1) | Triage priority | Weighted scheduling; START triage ordering |
| u_i ∈ {0, 1} | decoded image still says the right thing | One-shot generative decoding can fail; a wrong image has no value | Sagduyu & Erpek's semantic-error constraint; task-oriented SemCom |
| g(·) | 1 until the soft deadline D, then linear decay to 0 at 2D | Information loses value with time | **Time-utility functions** in real-time scheduling (Jensen; Locke 1986); value/age of information (Kaul, Yates & Gruteser 2012) |
| 1 − dup | duplicate brings no new information | First copy counts | Value of information; redundancy-aware uplink (2606.31715) |

**Why multiply instead of add:** each factor is a *gate*. Late, wrong or duplicate means
worthless, whatever the tier. Weights and deadlines are design values swept ×0.5 / ×2. The
**headline metric CTR@D does not use the weights at all**, which answers "you chose the weights to
win".

### 11.3 Metrics (unchanged from v2 §6)
- **CTR@D (headline):** the share of human-labelled Critical images decoded on time *and* still
  correct.
- Normalised timely value vs. the oracle.
- Task-correct Critical images per decoder-second.
- Plus per-tier deadline rates, p95 waits, Low starvation, redundant-decode rate, decision time and
  header bytes.

### 11.4 Policy line-up **[v2.1]: expanded, grouped, every one defined**
Note on "normal 5G schedulers": PF, Max-C/I and EXP/PF schedule **radio resources by channel
quality**. Our scheduler has no channel-quality input (no radio scheduling, and no UE identity). We
therefore include their **delay- and fairness-based logic, adapted to decoder scheduling**, and say
so explicitly. A reviewer accepts "adapted", but not a scheduler that cannot see its own inputs.

| Group | Policy | Rule at the decoder |
|---|---|---|
| Classic queueing | FIFO; LIFO; Random | — |
| | Round Robin over mission profiles | Cycle through mission ids (the receiver knows mission, not UE) |
| | Deficit/Weighted Fair Queuing (DRR/WFQ) | Tier-weighted fair share |
| Real-time | EDF | Earliest deadline (predicted tier) |
| | LLF (least laxity first) | Smallest slack = deadline − age − S |
| Priority | Strict priority (SPI-first) | Highest predicted tier, FIFO within tier |
| | Priority + aging | Priority rises with waiting time (prevents Low starvation) |
| 5G-derived (adapted) | PF over mission profiles | value / (recent served value of that mission): fairness across missions |
| | M-LWDF (adapted) | −log(δ)/D × waiting time × expected value |
| | EXP rule (adapted) | Exponential delay-urgency weighting × expected value |
| Index theory | Generalised cμ / cost-of-delay index | Expected value lost by waiting one more decode ÷ S |
| | Whittle-type index with abandonment | Following 2609.19792 (convex delay costs + impatience) |
| Learning to rank | LambdaMART; RankNet | Retrained on train events, target = true value |
| RL (standard) | DQN; PPO (unconstrained) | Reward = timely value |
| **Proposed** | **SV-DPP** | §11.5 |
| **Proposed** | **SC-PPO** (constrained PPO) | §11.6 |
| **Proposed** | **SC-PPO-AIF** (RLAIF) | §11.7 |
| Bound | Oracle | SV-DPP with true tier and true u |

That is about 20 policies, all sharing the same header and the same purge rule, compared on identical
traces.

### 11.5 SV-DPP: taught slowly, with a corrected formula **[v2.1]**
**Honesty first:** drift-plus-penalty is Neely's Lyapunov method (2010), and Sagduyu & Erpek applied it
to semantic queues. **We did not invent DPP. We adapt it.** Our contribution is the formulation: what
counts as value, what the constraint is, and how the header feeds both. That is normal and
acceptable in a Q1 paper, *if we say it plainly*.

**The idea in one sentence.** Maximise delivered timely value, but keep a running **debt counter Z**
for missed Critical deadlines. The bigger the debt, the more the scheduler sacrifices value to save
images at risk of missing.

**Debt update** (after every decision, using predictions only, so it is causal):

  Z ← max(Z + (predicted-Critical images that just missed their deadline) − ε × (predicted-Critical arrivals), 0)

With ε = 0.05: we tolerate 5% Critical misses. Above that, debt grows. Below it, debt drains.

**Score of each queued image i when a decoder frees up:**

  **score_i = V · E[v_i] + Z · pC_i · risk_i**

- **E[v_i]** = expected timely value if decoded now = Σ_tier p_tier · w_tier · g_tier(age + S) ·
  û_i · (1 − dup_i). "How much good this decode does."
- **pC_i** = SPI's probability that i is Critical.
- **risk_i** = the chance that i misses its Critical deadline if we do *not* serve it now. Simple form:
  1 if slack_i < S × (queue position under the current order), else a smooth fall-off. "Is this the
  last chance to save it?"
- **V** = a knob. Large V favours value; small V favours constraint safety.

**Purge:** any image with E[v_i] = 0 (expired or duplicate) is dropped instead of decoded.

**Worked example** (S = 0.5 s, D_C = 5 s, D_H = 15 s, V = 1):
| Image | SPI | Age | û | E[v] (approx.) | risk | Z = 0 score | Z = 6 score |
|---|---|---|---|---:|---:|---:|---:|
| A: collapsed house | pC 0.9 | 2.0 s | 0.8 | 0.9·8·0.8 ≈ 5.8 | 0 (3 s slack) | **5.8** | 5.8 |
| B: flooded road | pH 0.8 | 10 s | 0.9 | 0.8·4·0.9 ≈ 2.9 | 0 | 2.9 | 2.9 |
| C: selfie | pL 0.95 | 30 s | 0.9 | ≈ 0.9 | 0 | 0.9 | 0.9 |
| D: blurry, maybe injured person | pC 0.6 | 4.4 s | 0.5 | 0.6·8·0.5 ≈ 2.4 | 1 (0.1 s slack) | 2.4 | **2.4 + 6·0.6·1 = 6.0** |

- **With no debt (Z = 0):** pick A, the highest expected value.
- **After recent Critical misses (Z = 6):** pick D first. D is the *last chance* for a possibly-Critical
  image. A still has 3 s of slack and goes next.

This is exactly what SPI-first cannot do, because it only sorts by tier.

**Special case.** With no duplicates, û = 1, and deadlines far away, the risk term is zero and
score = E[v]. That is a sort by expected tier weight ≈ SPI-first, so **the old result is reproduced
as a special case**.

**Optional second debt counter Y** for Low starvation (guaranteeing a minimum service share for
Low): the same mechanism, one more term. We decide after the first results.

### 11.6 SC-PPO (also adapted, not invented)
- **What it is:** PPO with a Lagrange multiplier for the Critical-miss constraint. This is the
  established "PPO-Lagrangian" family (Ray et al. 2019; Tessler et al., RCPO 2019; Stooke et al.,
  PID-Lagrangian 2020).
- **Our parts:**
  1. a set-based network that reads a variable-length queue of headers;
  2. a reward built from true outcomes;
  3. a warm start that imitates SV-DPP;
  4. one constraint aligned with SV-DPP's, so the two are directly comparable.

Full spec: `FINAL_PLAN_V2.md` §8.2.

**Why RL can now win.** SPI is noisy on new events. RL trains on *true* outcomes and can learn when
the header lies, for example "low SPI but STS shows many kept person tiles". The index cannot learn
that.

### 11.7 SC-PPO-AIF: RLAIF (RL from AI feedback)
**What RLAIF is:** instead of human labels, an AI judge provides the training signal (Bai et al. 2022,
Constitutional AI; Lee et al. 2023, "RLAIF"). Here, Qwen2.5-VL is the judge.

**Design:**
1. Collect AI feedback on **train-event** images:
   - **(a) Pointwise:** Qwen's tier probabilities from class-token log-probabilities. Stable, and
     reuses the 3,933 existing labels.
   - **(b) Pairwise**, on a few thousand pairs: "which image should a responder see first?", asked in
     **both orders**, keeping only consistent answers. This fixes the order bias the teammate hit
     (7/30 stable in v5).
2. Fit a small **reward model** r_φ(header) with a Bradley–Terry loss on the pairs, plus a
   cross-entropy loss on the pointwise labels.
3. Train SC-PPO with r_φ replacing the true-value reward.
4. **Evaluate on human labels**, so there is no circularity.

**Research question it answers:** *can VLM feedback replace human annotation when training semantic
schedulers, and how much CTR@D does that cost?*

**Cost:** about 3–5 T4-hours for the pairs.

---

## 12. SQ-GAN retraining: a parallel track for the teammate

**Should we retrain?**
- The current checkpoint is Cityscapes-trained; disaster photos are out of distribution, and the
  earlier reader test found 0/12 positives after decode.
- Retraining improves u (better reconstructions) and can fix the class list.

**How to split the work so nobody blocks anyone:**
- **Scheduler track (us):** builds everything on the **current** checkpoint. The simulator only reads
  a lookup table (payload bytes, decode time, STS, û, u per image), so a better codec later only means
  regenerating that table.
- **Codec track (teammate):** fine-tune from the existing checkpoint, not from scratch. The paper
  trained on Cityscapes for 200 + 100 epochs; fine-tuning is far cheaper. Train on CrisisMMD-train
  + Places365 with **new segmentation maps from an ADE20K SegFormer** (it has water/sea/river
  classes; check for smoke/fire [C]), merged into **20 project classes** so SQ-GAN's 20-channel input
  stays unchanged. This puts flood and water into STS.
- **Gate:** the fine-tuned model must raise mean u at m = 0.25 on val events. Otherwise we keep the
  original checkpoint.

---

## 13. Simulation: should SQ-GAN run inside every simulation?

**No: trace-driven simulation.** Decoding costs seconds per image; RL needs millions of decisions.
Standard practice in serving and scheduling research:
1. **Once, offline:** encode and decode every pool image on Kaggle and store, per image: header
   fields, payload bytes, measured decode time, u.
2. **Simulation:** replays arrivals and decisions using that table. Fast (millions of decisions on
   CPU), exactly repeatable, and every policy sees identical data.
3. **Validation run:** one small *live* end-to-end run (~500 images, real SQ-GAN decoding inside the
   loop, the 2–3 best policies). It shows the lookup simulation matches reality. Reviewers like this
   line: "trace-driven results validated by a live run".

---

## 14. Implementation plan

Time is no longer the constraint, so the plan below is in **phases**, not days. The order matters
more than the hours.

| Phase | What | Output | Gate |
|---|---|---|---|
| P0 Verify | CrisisMMD files, labels, tweet ids, per-event and per-tier counts; Places365 blacklist; SQ-GAN T4 timings (full decode, map-only), masker output keys | `counts.json`, `sqgan_t4_bench.json` | ≥ 150 Critical in test events |
| P1 Labels + SPI | Tier mapping (§4); CLIP embeddings; SPI retrain + calibration; κ vs. Qwen; "no normals" ablation | SPI report | Critical recall on test events ≥ 0.6 |
| P2 Codec table | SQ-GAN encode/decode for the pool; STS; hash; u from the independent receiver model; û model; bit-depth flip-rate test | lookup table + 64-B header spec | u varies and is not ≈ 0 everywhere |
| P2' (parallel) | Teammate: SQ-GAN fine-tune with ADE20K-merged classes | new checkpoint | raises mean u on val |
| P3 Simulator | `edge_decoder_env.py` on top of the teammate's staged timing; 64-B header codec; mission registry | simulator + tests | degenerate case reproduces SPI-first |
| P4 Policies | All ~20 policies; SV-DPP; oracle | `edge_policies.py` | SV-DPP > SPI-first + EDF on val |
| P5 Learning | LambdaMART/RankNet retrain; DQN/PPO; SC-PPO; RLAIF data + reward model + SC-PPO-AIF | checkpoints | — |
| P6 Experiments | Sweeps + ablations on **test events only**, after the policy set is frozen; live validation run | figures, tables | — |
| P7 Paper | Sections: system/O-RAN, problem, header, receiver novelty, STS, policies, results, limitations | manuscript | — |

---

## 15. Your other questions
- **"Did we invent SV-DPP and SC-PPO?"** No: both adapt established methods (DPP: Neely; PPO-Lagrangian:
  Ray, Tessler, Stooke). Our inventions are:
  1. the decoder-side semantic scheduling problem;
  2. the stateless 64-B semantic header;
  3. receiver-side LSH novelty;
  4. the codec-aware STS and û;
  5. the timely-semantic-value formulation with a Critical-miss constraint;
  6. the RLAIF comparison against human labels.

  The paper must say this in exactly those words. Reviewers reject overclaiming much faster than
  modest claims.
- **Swin-JSCC:** it is an analog JSCC codec, meaning channel symbols rather than bytes, so it fits our
  "internet-like" digital pipe awkwardly. Use JPEG/BPG as the digital comparison codecs, flagged as
  "STS unavailable". SwinJSCC is optional, using a quantised variant. The result to show: how much
  of the header's gain comes from STS, and therefore exists only with a token-masking codec like SQ-GAN.
- **Gender:** none. "It" or "they" are both fine.
- **One caution:** semantic communication is an established field going back to Shannon and Weaver's
  "levels of communication" (1949). Our claim should be the narrower one above. That is what makes
  it survive review.
