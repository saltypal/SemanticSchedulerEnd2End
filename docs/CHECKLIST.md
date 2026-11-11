# Master plan and checklist (replan 2026-11-11 evening; deadline Friday 2026-11-13)

**Source of truth:** `docs/reviews/FINAL_PLAN_V3.md` (system §2, data §3, header §4, objective §5, SV-DPP §6,
SCOPE §7, policies §8, simulator §9, build order §10) **plus the locked decisions D1–D13 below.** Anything not in
v3 or in this table is not done without asking the user. Status: DONE / READY (code + tests, not run) / TODO / BLOCKED.

## Paper claim
On real crisis images, an O-RAN edge decoder scheduler that sees only a 64-byte semantic header (SV-DPP) delivers
more Critical images before their deadline (CTR@D) than every baseline, close to the genie, and each header field
earns its bytes.

## Locked decisions (after v3)
| # | Topic | Decided |
|---|---|---|
| D1 | People photos | COCO people, 3000 (not Open Images) |
| D2 | Everyday scenes | Places365 val, 17/class, minus 7 blacklisted classes |
| D3 | Non-CrisisMMD images | all Low |
| D4 | Val event | California wildfires (Sri Lanka had 71 Critical; v3 S0 rule). Train = Harvey, Irma, Iraq–Iran, Sri Lanka; test = Maria, Mexico |
| D5 | Fidelity u / judge / û model | dropped. v = w·g·ζ. Headline = CTR@D (no u-gate) |
| D6 | Codec gates that used u | LPIPS + SegFormer mIoU of reconstruction vs original, on val |
| D7 | SQ-GAN fine-tune | we do it, on the second account `satyapaladugu` |
| D8 | SQ-GAN order | v0 pilot (plumbing + latency) -> fine-tune pilot -> full fine-tune -> one full pass with v1 = codec table (header version 2) |
| D9 | Fine-tune data | train-split images only; images < 256x512 skipped (drops Places365); `seg_classes.yaml` defaults kept |
| D10 | Algorithms | v3 line-up + 13 new algorithms (P0), implemented first |
| D11 | Scheduler training | phase 2, all at once, parallel notebooks on both accounts |
| D12 | Accounts | plus `satyapaladuguplus` (WSL CLI) = data + main line; second `satyapaladugu` (Windows CLI) = SQ-GAN line |
| D14 | Duplicate-pair labels (S2c) | rule-based, not human: duplicate iff pHash <= 8 or SimHash <= 1. The P(dup) model therefore mirrors that rule; the paper says so |
| D15 | Relevance | per mission profile: cosine(CLIP image, CLIP mission text) precomputed per (image, mission) as columns; the simulator picks the column of each UE's mission profile |
| D13 | Class imbalance | handled in the SPI head (class-balanced loss, already used by the grid) + calibration + Critical-recall gate; val/test keep the natural mix; Critical-share is a sweep sensitivity row. No resampling |

## Rules
- Small inputs only on the PC, under `D:\Bunker\BaseCamp\_claude_scratch` (B 32 MB and C1 433 MB are there; images are not).
  Pytest: `--basetemp D:\Bunker\BaseCamp\_claude_scratch\pyt...`; in PowerShell pass test files explicitly (no `*` expansion).
- Kaggle: plus = `wsl -e /home/Satya/.venvs/semantic-kaggle/bin/kaggle ...`; second = native Windows `kaggle`. HTTP 429 -> wait 30 s.
- Notebooks clone latest `main`, print the SHA, write `run_manifest.json`. Kaggle Secrets are read with
  `kaggle_secrets.UserSecretsClient` (they are NOT env vars) and only reach runs started from the web editor.
- Commit to `main` only after the edge tests pass; never edit `Archived/`. Errors -> `docs/MISTAKES.md`.
- Leakage: test events only in S10; every gate uses val; fine-tune reads train images only.

---

# Where we are (2026-11-11 evening)

## Datasets (all private, owner satyapaladuguplus)
| Id | Slug | Contents | State |
|---|---|---|---|
| A | `scheduler-dataset-v1-images` | 27,168 images + index (2.5 GB) | DONE |
| B | `scheduler-dataset-v1-labels` | labels, counts, gates; **v2 adds `splits_v1.json`, `split_report.json`, `dup_pairs_candidates.csv`** | DONE |
| C1 | `scheduler-dataset-v1-cache` | 5-encoder embeddings, SimHash, pHash, SPI grid | DONE |
| C2 | `scheduler-dataset-v1-sqgan` | STS, payload bytes, kept-token maps, latency, LPIPS, mIoU (codec v1) | TODO (S4) |
| TAB | `scheduler-dataset-v1-simtable` | per-image simulator table + latency CDF | TODO |

## Done
- [x] S0 counts + gates; S1 labels; S2 UE features (27,168/27,168, 0 failures)
- [x] **S2b splits frozen** (`parameters/edge/freeze_splits.py`): a val/test image within pHash ≤ 8 or SimHash ≤ 1 of any
      train image is dropped (thresholds from the measured histogram: pHash has a gap after 8, SimHash has none).
      124 dropped. Train 15,974 (C 1218) / val 2,485 (C 499) / test 8,585 (C 760). Gates pass. Train identical to the grid's.
- [x] S2c candidates: 372 pairs (116 pHash-close, 74 SimHash-close, 54 same-tweet, 230 boundary, 68 far), train+val only
- [x] S3 grid ran (class-balanced). Top macro-F1: SigLIP2 logistic 0.587 (Critical recall 0.754). **CLIP-B/32 logistic C=1:
      F1 0.577, Critical recall 0.830, ECE 0.021, 12 ms vs 18 ms.** v3 locks CLIP-B/32 on the UE and the grid shows no clear
      gain from SigLIP2 (−0.01 F1 but −0.08 Critical recall) -> **CLIP-B/32 stays** (also feeds the SimHash key: one embedding).

---

# PHASE 0 — code (Wed night–Thu morning)
## P0 New scheduling algorithms (worker W-ALG; code + unit tests, CPU)
Already in code: FIFO, Random, RR-mission, EDF, TUF/DASA, SPI-first+EDF, priority+aging, M-LWDF, LLF, LELF, EXP, DAPF,
PF-mission, gen-cμ, gen-cμ/h, Whittle-abandon, EdgeServing, completion-aware DPP, per-tier DPP w/o ΔG, MPC, SV-DPP, genie, MILP.
- [ ] D-over · HVDF · EDF with admission/triage · LCFS preemptive-priority · WFQ/DRR over tiers · Max-Weight/backpressure ·
      max-age-first (AoI) · Whittle index for AoI · Clipper AIMD batching · Clockwork-style batch planner ·
      Shepherd/Nexus-style SLO packing · risk-averse SPI priority (CVaR/LCB) · contextual-bandit priority (LinUCB/Thompson)
- [ ] registered + grouped in `config/edge/policies.yaml`; header-only (no truth leak); one representative per group in main tables
## P0b Learner audit (W-ALG)
- [ ] `train_cli` trains every v3 learner: today ppo / scope_ppo / bc / dagger; **add dqn, scope_dqn, ltr (LambdaMART, RankNet), scope_ltr**
- [ ] `run_sweep`: headline `ctr_ungated`; remove u-gated columns from main tables (D5)
## P0c SQ-GAN code (me) — DONE
- [x] 3 failing tests fixed: real bug in `obtain_source` (successful clone crashed: Path used as commit string);
      `discover_root` depth 5 test; source_root test no longer clones from the network (9 min -> 70 s)

# PHASE 1 — everything before scheduler training (Thu)
## SQ-GAN line (second account; worker W-SQ) — BLOCKED on sharing (see "You")
- [ ] S0 pilot `dataset-semschedule-sqgan --mode pilot`: bundle restore, strict load, masker, latency distribution per B, VRAM, GFLOPs
- [ ] fine-tune pilot (few steps) -> full fine-tune (≤ 10 h) -> bundle `satyapaladugu/scheduler-sqgan-v1`; `codec.yaml` v1
- [ ] v1 adopted only if LPIPS and recon mIoU improve over v0 on a val subset (D6)
- [ ] S4 full pass with v1 -> C2 + codec-quality table; JPEG ladder becomes primary only if recon is unusable (v3)
## S3 SPI finalize (me, local CPU from C1)
- [ ] CLIP-B/32 logistic C=1 + temperature on val; out-of-fold posteriors for train (5 folds); val/test posteriors
- [ ] "no normals" ablation; per-source report (anti-shortcut: everyday images are all Low); CrisisMMD-only Critical recall
- [ ] κ vs Qwen (after the Qwen notebook)
## S9 data: Qwen feedback (notebook `edge-qwen-feedback`, plus account, T4x2) — code exists (`Scheduler/training/edge/rlaif.py`)
- [ ] pointwise tiers on a train sample + pairwise (both orders) -> feedback dataset; also gives the S3 κ labels
## S5 receiver models (me) — needs YOUR labels in `dup_pairs.csv`
- [ ] P(dup) logistic on (SimHash Hamming, pHash Hamming); cross-event key ROC
## S6 header -> HEADER FROZEN (me)
- [ ] bit-depth test on val (< 0.5% SV-DPP decision flips vs full precision); freeze `config/edge/header.yaml`
## Simulator table (me; STS/payload placeholders until C2)
- [ ] per image: uid, split, event, tier, calibrated SPI, relevance, keys, STS, payload bytes, capture time; latency CDF
- [ ] smoke + sanity on a 200-row val slice; publish TAB

## BAR (all true before phase 2)
- [ ] P0, P0b done; splits + header frozen; SPI final; C2 (v1) done; TAB + latency CDF published; Qwen feedback published;
      full edge suite green on `main`

# PHASE 2 — scheduler phase, all at once (Fri)
- [ ] S7 tune on val: SV-DPP (V, w), analytic policies + P0; genie; MILP bound (~200-job traces)
- [ ] S8 learners: LambdaMART, RankNet, DQN, PPO, BC, DAgger, SCOPE-PPO/DQN/LTR; minimal proof experiment
      (5 seeds, equal budgets, misspecified regimes) — one notebook per learner, both accounts in parallel
- [ ] S9 training: reward model, SCOPE-AIF (i), (ii) — evaluated on human labels
- [ ] header proofs on val: field drop-out, equal-bytes baseline, dedup P/R, gap to genie, overhead 64/(64+payload)
- [ ] S10 freeze -> test sweeps (Maria + Mexico), c ∈ {1,2,4,6,8}, 10 realisations, bootstrap CIs; live 2×T4 (c=2, KS test)
- [ ] S12 paper update from frozen results

# Notebook inventory (`Notebooks/Edge/`)
| Notebook (slug) | Account | Step | Hardware | State |
|---|---|---|---|---|
| `dataset-semschedule` | plus | S0–S3 grid (A, B, C1) | T4x2 | DONE |
| `edge-publish-runs` | plus | publish `run/` parts as datasets | CPU | DONE |
| `dataset-semschedule-sqgan` | second | SQ-GAN pilot, then full v1 pass (C2) | T4x2 | READY, blocked |
| `edge-sqgan-finetune-v1` | second | codec v1 fine-tune | T4x2 | READY, blocked |
| `edge-qwen-feedback` | plus | S9 data + S3 κ labels | T4x2 | TODO (code exists) |
| `edge-scheduler-train` (one version per learner) | both | S8, S9 training | CPU/T4 | TODO (phase 2) |
| `edge-scheduler-sweeps` | both | S7 tuning, main comparison, ablations, S10 test | CPU | TODO (phase 2) |
| `edge-live-t4` | plus | S10 live 2×T4 check | T4x2 | TODO (phase 2) |
| `edge-tunnel` | either | backup SSH into a session (`scripts/edge/kaggle_tunnel_client.sh`) | any | READY |
Local CPU (no notebook): S2b (done), S3 finalize, S5, S6, sim table, all P0 code.

# You (blocking items)
- [ ] **Share with `satyapaladugu` (Viewer)**: dataset Settings -> Sharing for `scheduler-dataset-v1-images`, `-labels`, `-cache`,
      `semantic-scheduler-sqgan-bundle-20261102`, `semantic-lossless-private-sqgan-source`, `semantic-lossless-private-sqgan-assets`
      (today the second account gets 403 on all six). This unblocks the whole SQ-GAN line.
- [ ] Revoke the API token that was pasted in chat and in the old notebook cell (if not done yet)
- [ ] Later: approve merged-branch deletions and deleting `edge-labels-v1`
