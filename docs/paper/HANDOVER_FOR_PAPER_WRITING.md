# Handover: writing the paper (for the teammate and Codex)

> **Update, 2026-11-15:** `DRAFT_PAPER.md` was rewritten from scratch. Section numbers and figure numbers in the
> task list below refer to the previous draft. Work from the current file, and use `docs/RESULTS_ALL.md` for every
> number. Test results are now complete for all learners except the live hardware check, which is not reported.

**Your job: write and polish the paper only. Do not run code, notebooks, training, sweeps or Kaggle. Do not
change any file outside `docs/paper/`.** Every number you write must come from a file listed in Section 9. If a
number is not in those files, do not write it.

The working draft is `docs/paper/DRAFT_PAPER.md` (IEEE journal structure, Markdown, Mermaid figures). Improve it; do
not start from scratch. Target venue: IEEE TCCN / IoT-J. Date of this handover: 2026-11-14.

---

## 1. The problem in one paragraph
Generative semantic codecs (here SQ-GAN) compress an image to a few hundred bytes, but decoding is slow (0.74 s for
one image on a T4 GPU). In a disaster, thousands of phones, drones and vehicles upload images at once, and an edge
decoder pool cannot decode them all in time. We schedule **which received image is decoded next** using only a
**64-byte semantic header** sent with each payload, with no per-UE state. Success is measured against **human labels**
(CrisisMMD): **CTR@D = fraction of truly Critical images decoded before their deadline**.

## 2. Claims of the paper (write these, in this order of emphasis)
1. **A stateless 64-byte semantic header** (urgency posterior SPI, mission relevance, SimHash and pHash keys, codec STS)
   lets an edge decoder pool triage images without per-UE state.
2. **Header-aware value-rate batching**: order by SPI urgency, choose the batch size B that maximises expected on-time
   value per second of decoding (renewal-ratio rule). Strong and robust.
3. **SV-DPP**: a Lyapunov drift-plus-penalty scheduler with per-tier debt queues, the principled generalisation of (2).
   Best on the full validation and full test splits.
4. **The header is what makes it work**: the same batching rule without the header delivers 0 Critical images in time
   under overload, with the header 0.78.
5. **Learned schedulers** (RankNet, LambdaMART, BC, SCOPE recipe) trained only through the header interface reach
   0.74-0.75, close to the hand-designed schedulers and +70% over EDF: the header is learnable.
6. Results hold on **held-out disaster events** (test: 0.80 vs 0.24 for EDF) and behind an **AWGN / Rayleigh uplink**.

Keep weak results (SemCover, SCOPE-DQN, DQN, BCEdge, Whittle, the LLM audit) in **Appendix G** only. Do not call them
proposed methods. Do not write "pending" or mention runs that are not in Section 9.

## 3. System (what was built)
- **UE side** (Fig. 1): SegFormer semantic map -> SQ-GAN v1 encoder with its semantic mask (SAMM) -> 464-byte payload
  at masking fraction m = 0.25; CLIP ViT-B/32 embedding -> SPI posterior (4 tiers), mission relevance, 64-bit SimHash;
  64-bit pHash; STS = 13 statistics of the kept codec tokens. All packed into the 64-byte header (Sec. IV, Fig. 2).
- **Header v2 layout** (bytes): 0 ver+flags | 1-4 message id | 5-8 capture time | 9-11 payload length | 12 codec+m |
  13-14 mission | 15-22 SPI 4 x u16 | 23-24 relevance u16 | 25-32 SimHash | 33-39 STS 13 x 4 bit | 40-47 pHash |
  48-49 CRC-16 | 50-63 reserved. Bit depths from a decision-flip test: SPI and relevance need 16 bits (8 bits flipped
  8.2% / 7.9% of SV-DPP decisions), STS needs 4 bits (0 flips).
- **Edge side**: header parser -> receiver fusion of SPI + STS into one tier belief -> duplicate probability against a
  memory of delivered images -> shared queue -> scheduler -> c decoder workers (one GPU each) -> responder.
- **Codec**: SQ-GAN (Pezone, Barbarossa, Caire, IEEE TCCN, arXiv 2502.09520) fine-tuned by us from the authors'
  Cityscapes checkpoints on 10,434 train-split images (batch 2, about 4,700 steps, 8 h on 2xT4, under one epoch).
  Payload 464 B per image; mean LPIPS of reconstructions 0.466. **SQ-GAN is source coding only and channel-agnostic**
  (the authors send its bits over any standard digital stack such as 3GPP); we therefore fine-tune without a channel
  and model the uplink separately (Sec. 6).
- **Measured batched decode latency** (codec v1, one T4, median): B = 1/2/4/8 -> 0.74 / 1.46 / 2.88 / 3.86 s. Batching
  helps only at B = 8. The simulator samples from these measured distributions.

## 4. Data, labels and splits
- Pool: CrisisMMD v2 (18,082 human-labelled tweets' images), Places365 validation sample (6,086), COCO people (3,000);
  27,168 images, 27,044 after cross-split near-duplicate removal (124 dropped: pHash <= 8 or SimHash <= 1 against train).
- Tiers (human labels mapped by rule): severe damage or injured/dead/missing people -> Critical; mild damage and
  infrastructure/vehicle damage -> High; little/no damage, informative, affected individuals -> Medium; not informative
  and all non-CrisisMMD images -> Low.
- **Event-disjoint splits**: train = Harvey, Irma, Iraq-Iran, Sri Lanka; **validation = California wildfires**;
  **test = Hurricane Maria + Mexico earthquake**. Counts: train C 1,218 / H 1,047 / M 3,653 / L 10,056; validation
  C 499 / H 139 / M 429 / L 1,418 (2,485); test C 760 / H 492 / M 1,892 / L 5,441 (8,585).
- SPI head (CLIP-B/32 logistic, class-balanced, temperature 0.95): validation macro-F1 0.567, **Critical recall 0.818**,
  ECE 0.024; High recall 0.079 (weak middle tier; a limitation). Everyday images predicted Low 98.6-98.7%.

## 5. Schedulers
- **Header-aware value-rate batching** (`spi_edf_ratio` in the code): order by SPI Critical probability then soft-deadline
  slack; move jobs that cannot meet any likely hard deadline to the back; choose
  B* = argmax_B [sum over top-B of sum_k q_k w_k P(S_B <= H_k - age)] / E[S_B] (Fig. 3).
- **Control without header** (`edf_ratio`): same batch rule, deadline order and uniform tier prior.
- **SV-DPP** (Fig. 4): score = V (E[v] now - E[v] after waiting) + sum_k Z_k q_k (on-time probability now - deferred);
  per-tier debt queues Z_k; per-job tie rule; renewal-ratio batching; soft dedup, purge only for delivered duplicates
  (P_dup > 0.99) or jobs past their hard deadline. Tuned on validation: **V = 0.1, purge 0.99, relevance weight
  beta = 1.0, tie tolerance 0.5**.
- **Baselines**: FIFO, EDF on SPI-predicted deadlines, SPI-first-EDF, gen-c-mu/h with abandonment, Whittle with
  abandonment, true-tier SV-DPP reference ("genie": SV-DPP with the true tier instead of SPI; a reference, not an
  upper bound).
- **Learners** (all see only header features): RankNet, LambdaMART, BC (imitation of SV-DPP), SCOPE-LTR, SCOPE-PPO,
  PPO-BC; also SCOPE-DQN, DQN, BCEdge, SemCover (appendix). **SCOPE** = constrained RL + imitation of the SV-DPP teacher
  with an **adaptive teacher weight** (TGRL-style: weight x0.9 when the student beats the teacher on the same episode,
  /0.9 otherwise) and dual updates of per-tier penalties (Fig. 5). All ranking learners use the shared value-rate batch
  rule and rank dead jobs last. Training: train split only, bursty MMPP arrivals at 4.5 /s on c = 2, B <= 8,
  150 iterations x 13 episodes, independent seeds.

## 6. Evaluation protocol
- Trace-driven event simulator (Figs. 7-10): CrisisMMD images arrive in tweet-time order compressed to the target load
  rho = lambda E[S_B] / (B c); everyday images Poisson-interleaved; 5 realisations; 95% bootstrap CIs.
- Grid: c in {1, 2, 4, 8} decoders, rho in {0.8, 1.2, 1.5}. Main operating point: **c = 2, rho = 1.2**.
- Policies never see the true tier; it is used only for scoring. Validation for tuning and selection; **test run once
  with frozen settings**.
- **Uplink model** (sensitivity, Table IX): narrowband 10 kHz uplink, 8 x (464 + 64) bits per image. AWGN: delay =
  bits / (B log2(1 + SNR)). Rayleigh block fading: fixed rate 0.7 B log2(1 + SNR); an attempt fails when the fade
  cannot carry it; up to 3 HARQ attempts of bits/R0 + 10 ms; lost after 3 failures. Mean delay / loss: AWGN 10 dB
  0.12 s / 0%; AWGN 0 dB 0.42 s / 0%; Rayleigh 20 dB 0.13 s / 1.1%; Rayleigh 10 dB 0.26 s / 4.6%; Rayleigh 0 dB
  0.95 s / 10.1%.
- Live 2xT4 check (Fig. 11) is implemented (same scheduler code, one SQ-GAN v1 decoder per GPU); only describe it if a
  result file appears in Section 9.

## 7. Results to report (all from Section 9 files)
**Validation, full split (Table VII), CTR@D at c = 2, rho = 1.2 [95% CI]:** SV-DPP tuned 0.82 [0.80, 0.83];
value-rate batching 0.78 [0.76, 0.79]; genie 0.79; gen-c-mu/h 0.47; EDF 0.44 [0.42, 0.47]; no-header control 0.00;
FIFO 0.00. Single decoder (c = 1, rho = 1.2): value-rate 0.81, SV-DPP 0.64, EDF 0.26 (3.1x). c = 8: 0.87-0.96 for all
deadline-aware policies.
**Test, full split (Table VIII), c = 2, rho = 1.2:** SV-DPP (tuned on validation) **0.80 [0.78, 0.81]**; genie 0.49;
SPI-first-EDF 0.48; EDF 0.24; FIFO 0.00. c = 1, rho = 1.2: SV-DPP 0.65 vs EDF 0.05.
**Learners, full validation (Table XII), c = 2, rho = 1.2:** RankNet 0.75 [0.73, 0.76], LambdaMART 0.74, BC 0.74,
SCOPE-LTR 0.68; c = 1: 0.72-0.73 vs EDF 0.26.
**SV-DPP tuning (Table XI, 1,000 validation images):** beta x tie grid, best beta 1.0 / tie 0.5 = 0.825; beta 0 / tie 0.5
= 0.787 (relevance adds value).
**Uplink (Table IX, 1,000 validation images, c = 2, rho = 1.2):** value-rate 0.60 ideal -> 0.43 Rayleigh 0 dB; EDF 0.30
-> 0.09; ranking unchanged.
Additional results that may appear later in `results/edge_final/kaggle_full/` (only add them if the file exists):
SCOPE-PPO / PPO full validation, learner tests, STS-fusion ablation, value-rate rule on test, live check.

## 8. Limitations to state (Sec. IX)
Codec fine-tuned under one epoch and out of domain (LPIPS 0.466); SPI High recall 0.08; label noise in CrisisMMD (the
audit found a logo and a sports collage labelled Critical and one image under two ids); no policy is perfectly
tier-ordered; tweet time as proxy for capture time; simulation-based results with measured latency; uplink is a
sensitivity model; SV-DPP parameters tuned on the validation event; DQN-family learners unstable (appendix).

## 9. Where the numbers are (read-only)
- `results/edge_final/kaggle_full/*.json`: full-split sweeps; each file has `cells` with `policy`, `workers`, `rho`,
  `ctr_ungated` (= CTR@D), `ci95.ctr_ungated`, `value`, `low_starvation`, `timely` (per-tier success),
  `decision_time_mean_us`. File names: `edge-sweep-<val|test>-<group>.json`; `-tuned` = tuned SV-DPP config.
- `results/edge_final/subsample/*.json`: 1,000 / 2,000-image subsets (learners, references); appendix only.
- `results/edge_final/link/*.json`: uplink sensitivity, one file per link setting.
- `results/edge_final/tuning/sv_dpp_beta_tie_grid.csv`: Table XI.
- `results/edge_final/audit/AUDIT_REPORT.md` + `*_metrics.json`: LLM audit (Claude Haiku, low effort); appendix only.
- Design and history: `docs/CHECKLIST.md` (decisions D1-D20), `docs/reviews/FINAL_PLAN_V3.md`,
  `docs/reviews/FIX_PLAN_2026-11-13.md`, `docs/reviews/PLAN_2026-11-14_TO_8PM.md`, `parameters/JOURNAL.md`,
  `docs/design/SEMANTIC_SCHEDULER_DESIGN.md`.
- Code (to describe methods, not to run): `Scheduler/edge/sv_dpp.py`, `Scheduler/edge/batching.py`
  (`ratio_batch_size`, `SpiEdfRatio`, `EdfRatio`), `Scheduler/training/edge/` (learners, SCOPE), `Evaluation/edge/`
  (simulator `env.py`, `run_sweep.py`, `regimes.py` incl. `apply_link`), `parameters/edge/header64.py`.

## 10. Writing tasks, in order
1. Read `DRAFT_PAPER.md` end to end. Fix any sentence left incomplete by edits (search the codec-quality paragraph
   in Sec. VII-B and the Table II/III lines in Sec. IV).
2. Tighten the abstract to <= 250 words around claims 1-6 and the headline numbers (0.82 / 0.78 vs 0.44 on validation;
   0.80 vs 0.24 on test).
3. Introduction: motivation, gap, contributions (header; value-rate batching; SV-DPP; SCOPE; evaluation). No SemCover.
4. Related work: keep, shorten; verify every arXiv id marked TODO-verify, drop any you cannot verify.
5. Method sections III-VI: keep the equations, align names (value-rate batching = `spi_edf_ratio`).
6. Results VIII: Tables VII, VIII, XI, IX, XII as in the draft; add one figure "CTR@D vs number of decoders at
   rho = 1.2" from Table VII (data in the JSON files).
7. Discussion/limitations (Section 8 above), conclusion, appendix G (weak variants + audit).
8. Figures: keep the 12 Mermaid diagrams as specifications; redraw them as clean vector figures.
9. Consistency pass: every number in the text appears in a Section 9 file; no "pending", no runs that do not exist;
   units and tier names consistent; references numbered in order of first use.
