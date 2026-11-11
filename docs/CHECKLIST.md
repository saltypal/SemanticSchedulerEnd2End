# Master plan and checklist (final replan, 2026-11-11, deadline Friday 2026-11-13)

**Source of truth:** `docs/reviews/FINAL_PLAN_V3.md` (system model §2, data §3, header §4, objective §5,
SV-DPP §6, SCOPE §7, policies §8, simulator §9, build order §10) **plus the decisions in the table below**.
Nothing else changes. If something is not in v3 or in this table, it is not done without asking the user.

Status: DONE / READY (code + unit tests, not run on real data) / TODO / BLOCKED.

## Paper claim
On real crisis images, an O-RAN edge decoder scheduler that sees only a 64-byte semantic header (SV-DPP)
delivers more Critical images before their deadline (CTR@D) than every baseline, close to the genie, and each
header field earns its bytes.

## Decisions locked after v3 (by the user)
| # | Topic | v3 said | Decided now |
|---|---|---|---|
| D1 | People photos | Open Images | **COCO people, 3000** (Kaggle `awsaf49/coco-2017-dataset`) |
| D2 | Everyday scenes | Places365 | Places365 **val, 17 per class**, minus 7 blacklisted classes (Kaggle `benjaminkz/places365`) |
| D3 | Non-CrisisMMD images | Low | all Low (unchanged) |
| D4 | Val event | Sri Lanka | **California wildfires** (Sri Lanka 71 Critical; v3 S0 rule). Train = Harvey, Irma, Iraq–Iran, Sri Lanka; test = Maria, Mexico |
| D5 | Fidelity u, judge (S3b), û model (S5) | in value and gated CTR@D | **dropped.** v = w·g·ζ. Headline = **CTR@D ungated** |
| D6 | Codec gates that used u | S4 "u not ≈ 0", S11 "mean u rises" | replaced by **LPIPS + SegFormer mIoU (recon vs original)** on val |
| D7 | SQ-GAN fine-tune | teammate (S11, later) | **we do it ourselves, FIRST**, on the second Kaggle account `satyapaladugu` |
| D8 | SQ-GAN passes | v0 pass (S4) then v1 re-run (S11) | v0 **pilot only** (plumbing + latency) -> fine-tune v1 -> **one full pass with v1** = the codec table. Header `version=2` (v1 STS profile). Small v0-vs-v1 quality table on a val subset |
| D9 | Fine-tune data | train events | CrisisMMD train events + COCO train portion; images smaller than 256x512 are skipped (so Places365 is not used for fine-tuning). `# DECISION:` rows in `config/edge/seg_classes.yaml` keep their current defaults |
| D10 | Scheduling algorithms | v3 §8 line-up | v3 line-up **plus 13 new algorithms** (list in P0). Implemented **first** |
| D11 | Scheduler training | S8/S9 | kept separate: **phase 2, all at once**, parallel notebooks on both accounts |
| D12 | Kaggle accounts | one | `satyapaladuguplus` (WSL CLI) = data + main line; `satyapaladugu` (Windows CLI) = SQ-GAN line; both for phase 2 |

## Rules
- Laptop: no datasets or large outputs. Small report files only, to `D:\Bunker\BaseCamp\_claude_scratch` (never C:).
  Pytest: `--basetemp D:\Bunker\BaseCamp\_claude_scratch\pyt`.
- Kaggle CLI: plus account `wsl -e /home/Satya/.venvs/semantic-kaggle/bin/kaggle ...` (from PowerShell);
  second account = native Windows `kaggle` (account `win` in `config/edge/kaggle.yaml`). On HTTP 429 wait ~30 s.
- Notebooks clone latest `main`, print the SHA, write `run_manifest.json`, keep logs clean.
- Commit to `main` only after the edge tests pass; no AI co-author lines; never edit `Archived/` or old notebooks.
- Leakage (v3): test events touched only in S10; every gate uses val; fine-tuning uses train images only.
- Every error -> a row in `docs/MISTAKES.md`. Every finished box -> ticked here with the number/result.

---

# PHASE 0 — code first (Wednesday)

## P0 New scheduling algorithms (code + unit tests, CPU; worker W-ALG)
Already in code (22 + bound): FIFO, Random, RR-mission, EDF, TUF/DASA, SPI-first+EDF, priority+aging, M-LWDF, LLF,
LELF, EXP, DAPF, PF-mission, gen-cμ, gen-cμ/h (abandonment), Whittle-abandon, EdgeServing, completion-aware DPP,
per-tier DPP without ΔG, MPC rollout, SV-DPP, genie SV-DPP, MILP clairvoyant bound.
Learners already in code: DQN, LTR, SCOPE (`Scheduler/training/edge/`). Missing: plain PPO / BC / DAgger / RankNet /
LambdaMART entries and RLAIF — checked and filled in P0b.
- [ ] D-over (Koren & Shasha 1995)
- [ ] HVDF (highest value density first)
- [ ] EDF with admission/triage (drop jobs that cannot meet H_k)
- [ ] LCFS preemptive-priority
- [ ] WFQ / Deficit Round Robin over tiers
- [ ] Max-Weight / backpressure over tier queues
- [ ] Max-age-first (AoI) and Whittle index for AoI
- [ ] Clipper AIMD adaptive batching (adapted)
- [ ] Clockwork-style deadline-aware batch planner (adapted)
- [ ] Shepherd/Nexus-style SLO batch packing (adapted)
- [ ] Risk-averse SPI priority (CVaR / lower confidence bound of expected weight)
- [ ] Contextual bandit priority (LinUCB / Thompson on header features, online)
- [ ] registered in `Scheduler/edge/registry.py` and a group in `config/edge/policies.yaml`; header-only (no truth leak)
- [ ] main tables keep one representative per group (v3 §8); the rest go to the appendix equivalence plot
## P0b Learner/RLAIF code audit (W-ALG, after P0)
- [ ] confirm `train_cli` can train: PPO, BC, DAgger, DQN, RankNet, LambdaMART, SCOPE-PPO/DQN/LTR; add missing ones
- [ ] RLAIF code for the edge simulator: Qwen pairwise/pointwise collector, Bradley–Terry reward model, SCOPE-AIF (i), (ii)
- [ ] metric: `ctr_ungated` is the headline; u-gated columns removed from main tables (D5)

## P0c SQ-GAN code fixes (worker W-SQ)
- [ ] 3 failing tests: `test_skip_when_nothing_mounted` (pilot_checks), `test_discover_root`,
      `test_obtain_source_pin_and_fallback` (sqgan_finetune: str + WindowsPath in pinned fetch)
- [ ] commit the SQ-GAN source/mount edits (`--only-sqgan`, `--require-sqgan`, `find_dataset`, pinned-clone fallback) after edge tests pass

---

# PHASE 1 — everything before scheduler training (Wednesday–Thursday)

## S0 Verify & profile
- [x] CrisisMMD v2, Places365, COCO built into dataset A (2.5 GB); notebook `dataset-semschedule` v2, all stages PASS
- [x] val event: Sri Lanka 71 Critical -> California (D4)
- [ ] read `counts.json`: test Critical >= 150, val Critical >= 100; record counts per split x tier here
- [ ] SQ-GAN v0 pilot on T4 (second account): bundle restore, strict load, masker keys, latency **distribution** per B,
      GFLOPs, VRAM -> `sqgan_v0_profile.json` (W-SQ)

## S1 Raw labels
- [x] tier mapping applied (dataset B, `config/edge/data.yaml`)

## S2 UE features
- [x] 5 encoders incl. CLIP-B/32, SimHash semantic key, relevance, pHash, SegFormer class maps (C1, 433 MB, 2 GPUs)

## S2b Dedup & splits -> SPLITS FROZEN (worker W-DATA)
- [ ] any val/test image with a near-duplicate (pHash + SimHash) in train is dropped from val/test (v3 rule)
- [ ] event-disjoint splits; everyday images 60/10/30 after hash dedup
- [ ] re-check gates after removal
- [ ] publish `splits_v1.json` + removal report (plus account dataset)

## S2c Duplicate pairs
- [ ] ~400 candidate pairs (retweets + near matches) from train/val -> `dup_pairs_candidates.csv` (W-DATA)
- [ ] TEAM labels duplicate/distinct (~1 h) -> `dup_pairs.csv`

## S3 SPI
- [x] grid 5 encoders x 5 classifier families ran (C_grid PASS)
- [ ] read grid; choose the UE classifier (v3 locks CLIP-B/32 on the UE; switch only if the grid shows a clear gain at
      UE cost); **gate: Critical recall on val >= 0.6**
- [ ] calibration on val; "no normals" ablation; per-source report
- [ ] κ agreement vs Qwen labels (uses the S9 collection)

## S11→now: SQ-GAN fine-tune v1 (worker W-SQ, second account) — D7/D9
- [ ] fine-tune pilot (tiny run): source fetch, data list, class map, trainer launch, checkpoint save
- [ ] full fine-tune on train images only; bundle published as `satyapaladugu/scheduler-sqgan-v1`; fill `config/edge/codec.yaml` (v1)
- [ ] adopt v1 only if LPIPS and recon mIoU improve over v0 on a val subset (D6)

## S4 Codec table (v1) (W-SQ, second account)
- [ ] full SQ-GAN pass with v1 over all images: STS (13), payload bytes at m 0.25/0.55, kept-token maps, decode latency
      per B, LPIPS, recon mIoU -> `codec_v1_table`
- [ ] v0-vs-v1 quality table on the val subset; masking-vs-YOLO evidence (kept tokens on crisis classes)
- [ ] if reconstructions are unusable -> JPEG ladder becomes the primary codec (v3 S4 rule)

## S5 Receiver models (W-DATA)
- [ ] P(dup) logistic model on (semantic-key Hamming, perceptual-key Hamming) from `dup_pairs.csv`; cross-event key ROC

## S6 Header -> HEADER FROZEN (W-DATA)
- [x] pack/unpack/CRC (`parameters/edge/header64.py`)
- [ ] bit-depth test on val: fewest bits with < 0.5% decision flips vs full precision; freeze `config/edge/header.yaml`

## Simulator inputs (W-DATA; needs S2b, S3, S4, S5)
- [ ] sim table per image: uid, split, event, true tier, calibrated SPI (out-of-fold for train), relevance, keys,
      STS, payload bytes, capture time; latency CDF L(v1, B); publish as dataset
- [ ] simulator smoke + sanity checks on a 200-row val slice (CPU)

## S9 data: Qwen RLAIF collection (after S2b; train images only; either account)
- [ ] Qwen pairwise + pointwise feedback; also gives the S3 κ labels; publish

## BAR (all true before phase 2)
- [ ] P0, P0b, P0c done; splits frozen; header frozen; codec_v1_table; sim table + latency CDF published;
      Qwen feedback published; full edge test suite green on `main`

---

# PHASE 2 — scheduler phase, all at once (Friday)
- [ ] S7 analytic policies (v3 line-up + P0) + SV-DPP tuned on val (V, w); genie; MILP bound on ~200-job traces
- [ ] S8 learners: LambdaMART, RankNet, DQN, PPO, BC, DAgger, SCOPE-PPO/DQN/LTR; minimal proof experiment
      (5 seeds, equal budgets, misspecified regimes: MMPP bursts, wrong σ_W, noisy SPI) — parallel notebooks, both accounts
- [ ] S9 training: reward model, SCOPE-AIF (i) and (ii), evaluated on human labels
- [ ] header proofs on val: field drop-out ablation; equal-bytes baseline (64 B thumbnail / embedding prefix);
      dedup precision/recall; gap to genie; header overhead 64/(64+payload)
- [ ] S10 freeze -> test sweeps (Maria + Mexico), c ∈ {1,2,4,6,8}, 10 realisations, bootstrap CIs; live 2×T4 check (c=2, KS test)
- [ ] S12 paper: update `docs/paper/DRAFT_PAPER.md` from frozen results

## Workers now
| Worker | Model | Scope | Files it may touch |
|---|---|---|---|
| W-SQ | Sonnet low | P0c, S0 SQ-GAN pilot, fine-tune v1, S4 full pass, all SQ-GAN Kaggle ops (second account) | `Codec/`, `config/edge/{sqgan_finetune,codec,feature_pass,seg_classes}.yaml`, `scripts/edge/`, `Notebooks/Edge/*sqgan*`, SQ-GAN tests |
| W-ALG | Sonnet low | P0, P0b | `Scheduler/`, `config/edge/policies.yaml`, `Evaluation/edge/{metrics,run_sweep}.py`, scheduler tests |
| W-DATA | next, after W-ALG or in parallel when usage allows | S0 counts, S2b, S2c, S3, S5, S6, sim inputs | `parameters/edge/`, `Evaluation/edge/{tables,receiver_models}.py`, new data notebook |

## User actions
- [ ] make A, B, C1 Kaggle datasets (notebook -> Save Version -> "New dataset from output"), and make them readable
      by the second account (public, or shared with `satyapaladugu`) — W-SQ needs them
- [ ] TEAM: label ~400 duplicate pairs
- [ ] approve merged-branch deletions and deleting `edge-labels-v1`
- [ ] later: move Claude temp/config to D:
