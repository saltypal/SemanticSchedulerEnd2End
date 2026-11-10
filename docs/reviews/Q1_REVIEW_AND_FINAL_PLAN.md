# Semantic-Aware AI Scheduler — Reviewer Report and Final Plan

Date: 2026-11-10. Scope: `main` (10283199), teammate branch `sem_schedluerAI_trial2` (41f74017),
`scheduler_improvement_trials`, `new-scheduler`, the 2026-10-25 Codex "Parameters" sessions and the
teammate's 11-07 → 11-09 sessions. Nothing was trained or run for this review; every number below is
quoted from a file in the repo or a chat log and is cited.

---

## 0. Verdict in one paragraph

As it stands this would be **rejected at any Q1 venue (TWC/JSAC/TCOM/TCCN/IoT-J)**, and not because of
polish. The central experiment is **circular**: the VLM assigns each image a tier. The metric then
penalises waiting by `budget[tier]`, using the same kind of tier. The "scheduler" is a single-server
queue with **unit service time and no radio channel**. Under those conditions, "sort by predicted tier"
(SPI-first) is close to optimal by construction. It is the classical generalised-cμ / Whittle result for
convex class-dependent delay costs. That is why 11 of 14 learned methods, and **all four RL agents**,
lose to a sort (`experiments/results_analysis/runs/analysis-20261104/ANALYSIS.md`). The project has
good parts: real SQ-GAN integration, a measured header, solid provenance discipline, and a large VLM
label pool. But the research question has been framed as a queue-ordering toy, not a **wireless**
scheduling problem. The fix is not more algorithms. The fix is to (1) put the channel and the semantic
rate back into the decision, (2) evaluate against **independent human labels**, and (3) measure
**task success after one-shot decoding**, not waiting. All of this fits in four days if YOLO, COCO,
novelty and reviewer-gating work are cut.

---

## 1. What exists (evidence map)

| Component | Where | State |
|---|---|---|
| Environment | `Evaluation/semantic_learning_env.py` | Single-server; `service=1.0` unless `variable_service`; arrivals `uniform(0,count/load)` over **i.i.d. images**; cost `ell(wait/BUDGETS[tier])`, `BUDGETS=[2,4,8,16]`; reward `-Δcost/100` |
| Labels | `parameters/KAGGLE_RUNS.md` | Qwen VLM pseudo-labels (Low/Medium/High/Critical). "Native LADI/MEDIC annotations… are excluded" (line 24) |
| SPI head | `parameters/PLAN.md:535` | CLIP + head; teacher agreement **44/70 = 62.9%** vs 47.1% prior; **Critical recall 2/10** |
| Leaderboard | `ANALYSIS.md` | FIFO 5.822, SPI-first 3.374, LambdaMART 3.218 (other run: 3.704), PPO 4.594, DDQN 4.710, TD-PER 5.176 |
| Class outcomes | same | SPI-first Critical within budget 0.893 vs FIFO 0.173; Low left unselected 181 vs 69 |
| Header | trial2 `work_record.md` | No-COCO cut full app header 110.45 → 84.16 B (−23.8%); trial-2 packet fixed at 36 B |
| SQ-GAN | `Codec/integration/sqgan_payload.py`; trial2 `native_sqgan_pilot.md` | Real fine-tuned bundle; **464 B** at m_x=m_s=0.25; CPU decode 13.5 s; reader 0/3 → 0/9 → 0/12 positives after decode |
| Teammate Trial 2 | `parameters/spi_scene_packet.py`, reviewer runs v1–v21 | 8-byte SQ-GAN encoder summary (mean+max pool → Linear 512→8) + RankNet; VLM reviewer never passed qualification (v5 7/30 stable; v12 14/30 clear winners) |

---

## 2. The Good (keep)

1. **SQ-GAN is the right codec choice.** Pezone, Barbarossa and Caire (IEEE TCCN, arXiv 2502.09520) give
   runtime-adjustable masking fractions `m_x, m_s ∈ [0.05,1]` **with one set of weights**. That is a
   free semantic MCS knob, and nobody in the scheduling literature exploits it yet.
2. **Integration is real.** Pinned checkpoint, hash checks, pixel-identical decode parity, and measured
   payload bytes. Most semantic-comm papers never get this far.
3. **A strong-baseline culture.** The team reported honestly that SPI-first beats almost every
   learned method. A reviewer will respect that, as long as the paper turns it into a contribution.
4. **Provenance and paired replays.** Same traces across policies, fixed seeds, held-out test kept
   closed. Keep the habit, but spend far less effort on it.
5. **Large image pool and VLM labels already paid for.** 6,000-record cache (CrisisMMD 4,000, OpenImages
   1,000, Places365 1,000), MEDIC-300 set, and Qwen labels.
6. **Header measured in bytes.** −23.8% after COCO removal is a small but real engineering result.

## 3. The Bad

1. **No wireless channel.** No SNR, fading, resource blocks, MCS, outage or uplink grant. For a
   "6G scheduler" paper this alone means desk rejection. A Samsung/Huawei reviewer will ask first:
   *what does the scheduler allocate?* Today the answer is "the order of decoding at the receiver".
2. **Delay is the only objective.** Waiting time is a proxy for value. In a one-shot system, what matters
   is *whether the received image still supports the decision, and whether it arrived in time*. That is
   task success × timeliness, not delay.
3. **Artificial chronology over i.i.d. images.** Uniform arrivals over random images, so "novelty",
   "history" and "redundancy" carry no signal (the env docstring admits this:
   `'mode':'MIXED_IID; artificial chronology'`). Any feature built on temporal context is noise by design.
4. **Statistics are thin.** 15 validation episodes; 75 Critical offers pooled; SQ-GAN pilots on 6/24
   images; LambdaMART's own score varies 3.22 vs 3.70 between runs. No confidence intervals.
5. **The algorithm zoo is "too shiny".** 14 learners (Set-Transformer, temporal DeepSets, ListMLE,
   TD-PER…) on a problem whose optimum is a sort. Each one adds a reviewer question and no insight.
6. **YOLO and scene features never showed value.** The analysis says so ("cannot establish a distinct
   benefit from novelty or all 93 scene categories"). COCO removal *changed* learned-policy orders but
   not SPI-first. The 387-dim feature vector is mostly dead weight.
7. **Process overhead swamped the research.** Gates, receipts and reviewer qualification v1→v21 (Nemotron,
   Qwen, rubrics) produced no scheduler result. The teammate spent about 2 days making a VLM agree with
   itself on 15 photos.

## 4. The Worst (fatal, must fix)

| # | Flaw | Why it is fatal | Evidence |
|---|---|---|---|
| W1 | **Circular evaluation.** Tier from VLM → policy sorts by predicted tier → metric weights by tier | The best possible policy is "predict the label well". RL learns nothing a classifier cannot. Any reviewer will spot it in one read | `cost()`/`target()` in env; ANALYSIS "within class marginal costs often tie" |
| W2 | **The objective is a solved problem.** Linear-then-doubled convex delay cost per class, single server, unit service | Generalised cμ rule (van Mieghem 1995); Whittle index for convex delay costs (arXiv 2609.19792). An index policy is near-optimal, so the RL loss is *expected*, not a finding | Leaderboard: PPO −36%, DDQN −40% vs SPI-first |
| W3 | **No independent ground truth.** Human labels from MEDIC/CrisisMMD were deliberately excluded. SPI Critical recall is 2/10 | "Critical within budget 89%" means "VLM-Critical within budget". It says nothing about true emergencies | `KAGGLE_RUNS.md:24`, `PLAN.md:535` |
| W4 | **One-shot reception is not evaluated.** Header-only scheduling; the payload is never judged after decode | The project's own reader found 0/12 positives after SQ-GAN at 464 B. If the decoded image is useless, scheduling it "on time" has zero value. The current metric hides this | trial2 `strong_reader_trial.md` |
| W5 | **SQ-GAN domain mismatch is unacknowledged.** SQ-GAN is trained on Cityscapes (driving, 256×512, INTERN-2.5 classes) | Disaster images are out of distribution. That probably explains the reader failures. It must be measured (rate–utility curves), not hidden | SQ-GAN paper §V; pilot results |
| W6 | **The RL reward is the metric delta.** `reward=-Δcost/100` with a sorting-optimal cost | There is no reason for an agent to beat the index, and no constraint handling for Critical safety | env `step()` |

---

## 5. Literature position (AlphaXIV survey, 2025–2026)

| Thread | Representative work | What we take | Gap we fill |
|---|---|---|---|
| Semantic rate control and queueing | Sagduyu & Erpek, *When SemCom Meets Queueing* (2605.05514): latent dim as action, drift-plus-penalty with virtual error queue, one-shot | Lyapunov DPP with virtual queue as the **analytic** proposed policy | Single user, CIFAR, no importance heterogeneity, no multi-UE channel scheduling |
| Importance-aware RA | 2606.29052 (importance-aware RA for task-oriented SemCom); 2610.34532 (multi-view semantic RA, CentraleSupélec) | Importance weights in the utility | Importance is assumed known; we use a **noisy, transmitted** importance (header) and quantify the header's value |
| Semantic AoI | QSAoI (2606.31303); VAoI diffusion-RL (2601.18069); AoII (2610.22265, 2608.13248) | Age **of critical information** as the timeliness metric | Not tied to image content or generative decoding |
| Index policies | Whittle for convex delay (2609.19792); RL-Whittle (2601.01179); RMAB with penalty constraints (2604.04101) | Whittle / cμ as the **strong analytic baseline** that explains why RL lost | — |
| SQ-GAN | 2502.09520 | m_x/m_s knob; SAMM relevance scores α_k | SQ-GAN never used SAMM output for **scheduling** |
| Disaster VLM triage | MEDIC (2108.12828); DisasterVL (2611.12144); DisasterInsight (2601.18493) | Human labels as ground truth; VLM as noisy sensor | No link to the radio |
| Redundancy-aware uplink | Overlapping FoV in vehicular 6G (2606.31715) | Optional: same-event redundancy | — |

**Novelty statement that survives review:** *a first scheduler that jointly decides who transmits and at
what generative semantic rate (SQ-GAN masking fraction), driven by a few-byte "semantic buffer status
report". The report carries the VLM posterior and an SQ-GAN mask summary that predicts each image's
rate–utility curve. The objective is constrained: maximise task-correct, deadline-met semantic value
under a Critical-miss constraint, evaluated against independent human labels.*

---

## 6. Final approach

### 6.1 System model (wireless, uplink, one-shot)
- K UEs (default 8; sweep 4–32) under one gNB. Slotted time with TTI τ = 1 ms (sweep 0.5 ms), B
  resource blocks per slot (default 25 at 180 kHz).
- Channel: 3GPP UMa path loss + log-normal shadowing + block Rayleigh fading per UE per slot. SNR maps
  to spectral efficiency through the 3GPP CQI/MCS table (or Shannon with a 3 dB gap).
- **One-shot:** an image's payload `b_i(m)` must be delivered within its allocation window. Otherwise it
  is an outage and the image is lost, with no HARQ. Optional realism: finite-blocklength error
  probability (Polyanskiy normal approximation), which suits a 6G short-packet audience.
- Each UE's queue holds images drawn from **event-coherent streams**: images from the same MEDIC or
  CrisisMMD disaster event per UE, so redundancy is real.

### 6.2 Header = "Semantic BSR" (≈ 8–12 bytes, sent in the scheduling request)
| Field | Bits | Source |
|---|---|---|
| VLM tier posterior (4 classes × 4 bits) | 16 | Qwen2.5-VL-7B class-token log-probs (reuse labels; add log-probs) |
| VLM confidence / entropy | 4 | same |
| SQ-GAN mask summary: SAMM relevance mass at m = 0.1/0.25/0.55 | 3 × 4 | SQ-GAN encoder only (cheap at sender) |
| Semantic-class histogram of top-k selected tokens (7 superclasses × 4 bits) | 28 | **Cityscapes-native only** (the bundle's SegFormer teacher is 20-class Cityscapes): person, building, road, vehicle, vegetation, sky, other. Water/fire do **not** exist in that label space |
| Payload size code for each m | 3 × 6 | SQ-GAN bitstream |
| Deadline class | 2 | mission |

Ablations: header {VLM only} vs {VLM + mask summary} vs {mask summary only} vs {oracle human label}.
This replaces YOLO/COCO completely. The mask summary has one job: **predicting how much task utility
an image loses when compressed** ("semantic compressibility"). That is measurable, and it is exactly
what a rate decision needs.

### 6.3 Ground truth and utility (breaks W1, W3, W4)
- **True tier** from MEDIC human labels via a fixed, published mapping:
  damage_severity × humanitarian/informativeness → {Critical, High, Medium, Low}. Do this for CrisisMMD
  as well. **The VLM never defines ground truth.**
- **Task utility** `u_i(m)`: does a frozen task model, applied to the *reconstructed* image, produce
  the same human damage/humanitarian label as on the original? Use a CLIP linear probe trained on
  MEDIC-train human labels; it is cheap. Also report LPIPS and CLIP-cosine as secondary measures.
  `u_i(m) ∈ [0,1]`, precomputed for m ∈ {0.05, 0.1, 0.25, 0.55, 1.0}.
- **Delivered value** of image i: `V_i = w(tier_true) · u_i(m_i) · 1{delivered ∧ t_i ≤ D_tier}`.

### 6.4 Objective and metrics (beyond delay)
Primary:
1. **Critical Semantic Recall @ deadline (CSR)**: the fraction of human-Critical images delivered,
   decoded *task-correctly*, and within their deadline. This is the headline number.
2. **Semantic value per resource block** (semantic spectral efficiency): ΣV / RBs used.
3. **Age of Critical Information (AoCI)** per event: mean and p95. Time since the last task-correct
   Critical image from that event.
Secondary: outage rate, Jain fairness across UEs, header overhead bits/image, scheduler decision time
(µs; must fit a TTI), Low-tier starvation.
Statistics: ≥ 30 seeds, event-disjoint train/val/test, paired bootstrap 95% CIs, Wilcoxon versus the
best baseline.

### 6.5 Policies (lean; every one has a reason to exist)
| Group | Policy | Why |
|---|---|---|
| Classical wireless | Round Robin, Max-C/I, Proportional Fair, M-LWDF, EXP/PF | A wireless reviewer will demand these; they are missing today |
| Priority / ranking | SPI-first (strict priority, fixed m), generalised cμ / Whittle index on VLM posterior, LambdaMART (existing) | The existing strong baselines, now judged fairly |
| **Proposed A (analytic)** | **SV-DPP**: Lyapunov drift-plus-penalty. Per slot it picks (UE, m, RBs) maximising `V·E[w·u(m)·P_succ(SNR,b(m))] − Q·RB-cost` and adds a virtual queue Z for the Critical-miss constraint (CSR ≥ 1−ε). Uses header posterior expectations | Provable constraint satisfaction, O(K·M) per slot, interpretable. Follows 2605.05514 and extends it to multi-UE plus importance plus generative rate |
| **Proposed B (learned)** | **SA-PPO-Lagrangian**: permutation-invariant actor (shared MLP per UE plus attention pooling). Action = UE selection + m; reward = ΣV − λ·Critical-miss; λ learned by dual ascent; warm-started by behaviour cloning of SV-DPP | Shows RL helps when the channel and arrivals are bursty or non-stationary; that is where myopic DPP is suboptimal |
| Oracle | Same as A with human tier and true u(m) | Upper bound; the gap shows how much a better header would buy |

**Our reward and loss, stated explicitly:**
- Reward per slot: `r_t = Σ_{i delivered in t} w(ĉ_i)·û_i(m_i) − λ·#(Critical deadlines expired in t) − β·RB_used/B`.
  During training, ĉ and û come from the header (causal). Evaluation uses human labels.
- Actor loss: PPO-clip + entropy, plus a Lagrangian term `λ·(J_C − ε)`, with `λ ← [λ + η(J_C − ε)]⁺`.
- Auxiliary loss (novel, cheap): a header-to-utility head trained with MSE on u(m) from the mask
  summary. This makes the "semantic compressibility" claim testable.

### 6.6 What to cut now
YOLO scene features, COCO block, novelty/history (unless event streams are used), Set-Transformer /
temporal DeepSets / ListMLE / TD-PER / DDQN, VLM-as-reviewer qualification (v1–v21), CRASAR/road readers,
the 387-dim feature vector. Keep their commits; drop them from the paper.

---

## 7. Four-day plan (training time included)

| Day | Kaggle account A (Windows CLI) | Kaggle account B (WSL CLI) | Local |
|---|---|---|---|
| **D1** | SQ-GAN profiling notebook (existing slug, new version): ~2,000 MEDIC test images, event-disjoint, + 500 CrisisMMD. Five m values → bytes, reconstructions, LPIPS, SAMM α mass, SSM token histogram. T4 is fast enough (~10k decodes) | CLIP linear probe on MEDIC-train **human** labels (original images); then score all reconstructions → `u_i(m)`. Add Qwen class log-probs on the 2,500 (reuse existing labels where hashes match) | Build `Evaluation/wireless_semantic_env.py`: vectorised numpy, lookup tables, 3GPP CQI table, one-shot outage. Implement RR/MaxCI/PF/M-LWDF/EXP-PF/SPI-first/cμ |
| **D2** | — | PPO-Lagrangian training (GPU or 4-CPU); BC warm-start from SV-DPP | SV-DPP + oracle; sanity check: with ideal channel and unit service, SV-DPP ≈ SPI-first. This reproduces the old result as a special case, which is a strong story |
| **D3** | Seeds × sweeps: load {0.6…1.4}, K {4,8,16,32}, mean SNR {0…20 dB}, VLM noise. Header ablations | — | Figures: CSR vs load, value/RB vs SNR, AoCI CDF, Pareto (CSR vs fairness), decision-time table |
| **D4** | — | — | Write: system model, problem (CMDP), SV-DPP derivation plus constraint bound, results, limitations (SQ-GAN OOD) |

Go/no-go checkpoints:
- **End of D1:** u(m) must vary across images *and* the mask summary must predict u(m) better than VLM
  tier alone (Spearman ρ). If not, drop the mask summary claim and keep rate adaptation with VLM.
- **End of D2:** SV-DPP must beat PF and SPI-first on CSR at ≥1 operating point with non-overlapping
  CIs. If PPO does not beat SV-DPP, report it honestly: "analytic index is near-optimal; RL only helps
  under bursts". That is a publishable negative result inside a positive paper.

---

## 8. Risks I will not hide
- **SQ-GAN out of distribution.** Utility at low m may be ~0 for most disaster images. Mitigation: that
  is exactly why m must be adapted. Also report a JPEG/BPG ladder as an alternative codec so the
  scheduler result does not depend on SQ-GAN's quality.
- **Human-label-to-tier mapping is a design choice.** Publish it and run a sensitivity check with one
  alternative mapping.
- **Four days fits a strong TCCN / IoT-J / TVT submission.** JSAC needs field traces or a testbed; say
  so in future work.
- **Medium class:** the README notes the SPI head lacks Medium. The VLM posterior covers four classes, so
  use that and drop the SPI head from the header.

---

## 9. Counter-review (fresh-context adversarial AE) and resolution

A separate reviewer with no shared context attacked sections 1–8. Accepted points **override** §6–§7.

| # | Objection | Verdict | Resolution |
|---|---|---|---|
| C1 | 25 RB × 180 kHz × 1 ms carries ≈560 B/TTI, so a 464 B SQ-GAN image fits in one TTI. The radio is never the bottleneck | **Accept (was fatal)** | Use a bandwidth-scarce disaster regime: damaged RAN with 1–6 RBs, or an NTN/NB-IoT-class link at tens of kb/s, or a UAV relay with a shared backhaul. Slots 10–100 ms, deadlines in **seconds**. Calibrate so offered load ≈ 0.6–1.4 × capacity at mid m. Add a JPEG/WebP ladder (10–100 kB) as the "no semantic codec" reference |
| C2a | Authors choose w and D, so a w·u maximiser wins by construction | **Accept** | Sweep w ratios and D tables. Report the region where rankings hold. Headline = **CSR@deadline**, which does not use w |
| C2b | Classical baselines get no header, so information and algorithm are confounded | **Accept** | Every baseline gets the same header: weighted PF, weighted M-LWDF, EDF, cμ. Claims are split into (i) value of header (ablation within one policy) and (ii) value of algorithm (same header) |
| C2c | Header→u(m) predictor can leak into the oracle | **Accept** | Event-disjoint train/test for the predictor. Verify that MEDIC's non-CrisisMMD sources carry event metadata; if not, use source-disjoint splits |
| C3 | Hand-made damage×humanitarian→tier mapping invites attack | **Accept** | Tier = MEDIC **native damage_severity** (severe / mild / little-none), plus one documented rule for Critical (severe ∧ people/rescue humanitarian label). Report one alternative mapping as sensitivity |
| C4 | SQ-GAN expects 1×3×256×512 input with a 20-class Cityscapes SegFormer teacher. SAMM access not verified. "Same label after decode" measures consistency, not correctness | **Accept** | Letterbox to 256×512 and report it. Cityscapes-only superclasses (fixed in §6.2). **Hour 1:** confirm SAMM α is exposed. Compute u(m) only on images whose original is correctly classified. Use two probes (CLIP-ViT and ConvNeXt) |
| C5 | "SR" is 1 bit on PUCCH; a BSR is a MAC CE needing a grant. A 7B VLM on a UE is not credible | **Accept** | Define the **Semantic-BSR as a new MAC CE**. Model SR→grant→BSR latency and count its bits in the RB budget. The VLM runs at a **UAV/edge aggregator** (sender-side MEC); handset variant = CLIP-tiny head. Report sender compute in a table |
| C6 | DPP with a virtual queue is textbook Neely. Time-average constraints are not per-image deadlines. Per slot it is MCKP ≈ cμ | **Accept** | SV-DPP becomes the *principled* proposed scheduler with an honest O(1/V) bound. Per-slot solve = greedy MCKP on `w·u(m)·P_succ/b(m)`. Novelty moves to: **measured value of a noisy transmitted semantic header + joint generative-rate selection under one-shot decoding**, evaluated against human labels |
| C7 | "First scheduler" claim will not survive; semantic spectral efficiency (Yan, Qin et al., WCL 2022) exists | **Accept** | Narrow the claim (below). Cite Yan/Qin. **Verify every 26xx arXiv ID before citing** (they came from the AlphaXIV index; re-fetch each) |
| C8 | Cut PPO-Lagrangian: likely null, a fifth RL loss | **Partial rebut** | The user requires an own RL method, reward and loss, so it **stays, time-boxed to ≤6 GPU-h on D3**. It runs only where myopic DPP is provably weak: Gilbert–Elliott correlated channel plus bursty arrivals. Warm start = behaviour cloning of SV-DPP. If it does not beat SV-DPP with CI separation, it is reported as a negative result in one paragraph. Never the headline |
| C9 | 30 seeds over the same 2,500 images are not independent | **Accept** | Bootstrap over **images/events**; seeds only for channel realisations |
| — | Missing baselines | **Accept** | Add: semantics-blind rate-adaptive PF (largest m the channel allows), lowest-m-first, EDF, header-weighted PF |
| — | Cheaper experiment with the same evidence | **Accept, promoted to D1** | **Static per-frame MCKP**: capacity C bytes, items (w, u(m), b(m)), compared across header variants. Runs in hours with no simulator. It is the first figure and the go/no-go |
| — | Header may not need SQ-GAN | **Accept as ablation** | Tiny CLIP regressor of u(m) from the original image versus SAMM-mass features. If CLIP matches, the header drops SQ-GAN features (cheaper sender) and SQ-GAN stays only as the codec. Report either outcome |
| — | MEDIC vs RescueNet/FloodNet | **Accept MEDIC** | MEDIC for the 4-day scope (labels, scale, existing pipeline, street-level views closer to Cityscapes). RescueNet = future work (needs an in-domain segmenter and SQ-GAN retraining) |
| — | Target venue | **Partial** | Four days yields complete *evidence* for a TCCN/IoT-J-scope paper. A polished full journal manuscript in the same four days is unrealistic. Plan: submit an IEEE WCL/CL letter now, extend to TCCN within about 3 weeks |

### 9.1 Locked claim (narrowed)
> Under a bandwidth-scarce one-shot uplink, a few-byte semantic MAC CE (VLM posterior + compressibility
> estimate) lets the gNB jointly pick *who transmits* and *which SQ-GAN masking fraction*. This raises
> Critical Semantic Recall@deadline against human labels over header-equipped PF/M-LWDF/EDF/cμ. We
> quantify how much of the oracle gap each header field closes.

### 9.2 Locked four-day schedule (replaces §7)
| When | Task | Gate |
|---|---|---|
| D1 h0–1 | Benchmark 20 images through SQ-GAN on T4: seconds/pass, SAMM α exposed? 256×512 letterbox | If >1 s/pass, cap at 1,000 images |
| D1 h1–4 | Profile N images × m ∈ {0.05, 0.1, 0.25, 0.55, 1.0} (acct A); CLIP + ConvNeXt probes on MEDIC-train human labels (acct B) | **Hour 4 go/no-go:** u(m) not ≈0 everywhere and increasing in m. If it fails, the JPEG ladder becomes the primary codec and SQ-GAN is a secondary row |
| D1 h4–10 | Static MCKP experiment across header variants (local CPU). Qwen log-probs on new images (1–2 GPU-h) | Does the header beat VLM-only on MCKP value? |
| D2 | `Evaluation/wireless_semantic_env.py`: scarce-RB regime, CQI table, one-shot outage, MAC-CE overhead, SR→grant delay. Baselines RR, Max-C/I, PF, wPF, wM-LWDF, EDF, cμ, SPI-first, blind rate-adaptive PF. SV-DPP + oracle | Sanity: ideal channel + fixed m ⇒ SV-DPP ≈ SPI-first (reproduces old result) |
| D3 | Sweeps (load, K ∈ {4, 8, 16}, SNR, w/D tables, VLM noise), header ablations. PPO-Lagrangian ≤6 GPU-h on GE-channel + bursts | Image-level bootstrap CIs |
| D4 | Figures + letter manuscript (system model, CMDP, SV-DPP + bound, results, limitations: SQ-GAN out of distribution, Cityscapes classes, synthetic arrivals) | — |

---

## 10. SQ-GAN semantic mask as the YOLO replacement (verified against code)

**What SQ-GAN already computes at the sender** (`SemanticLossless/Semantic_Lossless/src/sqgan.py:60-75`):
- a 20-class Cityscapes semantic map `s` (required input, 256×512);
- `masker(...)['sample_index']`: the positions of the image tokens kept on the 16×32 = 512 grid, at
  fraction m_x;
- `masker_semantic(...)['sample_index']`: the same for semantic-map tokens (m_s).
Whether the raw relevance *scores* are also in the returned dict is **not yet verified** (hour-1
check). The positions alone are enough for the header below.

**Header block "SQ-Mask" (≈8 B), replacing YOLO/COCO entirely:**
| Field | How | Bits |
|---|---|---|
| Kept-token class histogram | downsample `s` to 16×32; histogram of classes at `sample_index` for m=0.25, over 7 superclasses (person, building, road, vehicle, vegetation, sky, other) | 28 |
| Retention curve | fraction of person/vehicle/building tokens kept at m = 0.1 / 0.25 / 0.55 (run the masker 3× with the encoder output cached; masker is cheap) | 3 × 6 |
| Spatial spread | entropy of kept positions | 4 |
| (if scores exposed) relevance mass at m=0.25 | Σ top-k α / Σ α | 6 |

**Why this beats YOLO for this paper, and what it cannot do:**
- YOLO describes *what is in the image*. SQ-Mask describes *what the codec will keep*, and that is the
  quantity a joint who-transmits-and-at-what-rate decision needs. It costs ~0 extra sender compute,
  because the encoder and map are already run for the payload. YOLO is a second network.
- It **cannot detect urgency**: Cityscapes has no rubble, water, fire or "damaged building" class.
  Urgency stays with the VLM posterior. SQ-Mask's job is **compressibility** (how fast task utility
  drops as m shrinks).
- The claim "SQ-Mask replaces YOLO" becomes a measured ablation, not an assertion:
  `{none, YOLO-scene (cached), SQ-Mask, learned SQ-GAN descriptor (teammate), SQ-Mask+VLM, oracle}`.
  Metrics: Spearman ρ of predicted vs true u(m) slope, then CSR@deadline in the scheduler.

## 11. Reusable from teammate's Trial 2 work (`sem_schedluerAI_trial2`)

| Reuse directly | File | Use in final plan |
|---|---|---|
| SQ-GAN payload adapter with hash-pinned fine-tuned bundle | `Codec/integration/sqgan_payload.py` | Codec for every m level |
| Single-pass pre-mask feature capture (forward hook; payload bytes identical) | `Codec/integration/sqgan_prefeatures.py` | Extend the same hook pattern to capture `masker` output → SQ-Mask fields |
| Parity smoke test (identical 464 B payload, pixel-identical decode) | `Codec/integration/prefeature_parity_smoke.py`, `docs/scheduler_improvement_trials/trial2_prefeature_parity.json` | Hour-1 sanity check on Kaggle T4 |
| Pooled 8-value encoder descriptor (mean+max → Linear 512→8) | `Scheduler/training/scene_descriptor.py` | One ablation arm ("learned SQ-GAN descriptor"); retarget its loss from RankNet-on-reviewed-pairs to **MSE on u(m) slope** (labels now come from measured utility, so no VLM reviewer needed) |
| Versioned fixed-overhead packet + CRC (36 B) | `parameters/spi_scene_packet.py` | Template for the Semantic-BSR MAC CE (swap the 4 SPI bytes for the VLM posterior and the 8 scene bytes for SQ-Mask) |
| No-COCO transport + 6,000-record byte audit | `Evaluation/no_coco_scene_transport.py` | Byte-accounting precedent; cite the −23.8% |
| Measured-service staged simulator (capture/sender/arrival/queue/decode times) | `Evaluation/scheduler_codec_replay.py`, `docs/scheduler_improvement_trials/sqgan_staged_replay.md` | Reuse the timing stages inside `wireless_semantic_env.py` |
| Native SQ-GAN timings (464 B; 624 B at m_x=0.5; CPU decode 13.5 s) | `native_sqgan_pilot.md`, `native_sqgan_extension.md` | Payload ladder seed values; motivates GPU/edge decode |
| Working Qwen-on-T4 recipe (4-bit, SDPA, fixed image-token budget, 7.12 GB peak) | reviewer runs v8–v10 | Qwen log-prob pass on new MEDIC images |

| Do not reuse | Why |
|---|---|
| VLM reviewer qualification (v1–v21, Nemotron/Qwen pair grading) | Utility is now measured on reconstructions against human labels, so no reviewer gate is needed |
| LADI v2 road-damage readers, CRASAR | Different task; MEDIC is the dataset |
| RankNet-on-reviewed-pairs training | No stable labels were produced (best 14/30 clear pairs) |
