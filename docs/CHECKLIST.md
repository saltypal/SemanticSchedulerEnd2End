# Master plan and checklist (finalized 2026-11-11 night; deadline Friday 2026-11-13)

**Source of truth:** `docs/reviews/FINAL_PLAN_V3.md` (system §2, data §3, header §4, objective §5, SV-DPP §6, SCOPE §7,
policies §8, simulator §9, build order §10) **plus the decisions D1–D17 below.** Anything else needs the user's OK.
Status: DONE / READY (code + tests, not run) / TODO / PARKED. Results go to `parameters/JOURNAL.md`, errors to `docs/MISTAKES.md`.

## Paper claim
On real crisis images, an O-RAN edge decoder scheduler that sees only a 64-byte semantic header (SV-DPP) delivers more
Critical images before their deadline (CTR@D) than every baseline, close to the genie, and each header field earns its bytes.

## Locked decisions (after v3)
| # | Topic | Decided |
|---|---|---|
| D1 | People photos | COCO people, 3000 (not Open Images) |
| D2 | Everyday scenes | Places365 val, 17/class, minus 7 blacklisted classes |
| D3 | Non-CrisisMMD images | all Low |
| D4 | Val event | California wildfires (Sri Lanka had 71 Critical). Train = Harvey, Irma, Iraq–Iran, Sri Lanka; test = Maria, Mexico |
| D5 | Fidelity u / judge / û | dropped. v = w·g·ζ (env + genie fixed). Headline = CTR@D |
| D6 | Codec quality gate | LPIPS + SegFormer mIoU of reconstruction vs original, on val |
| D7 | SQ-GAN fine-tune | we do it, second account `satyapaladugu` |
| D8 | SQ-GAN order | fine-tune -> one full pass with v1 = codec table (header version 2). **No v0 pilot / v0 pass (D17)** |
| D9 | Fine-tune data | train-split images only; < 256x512 skipped (drops Places365); `seg_classes.yaml` defaults |
| D10 | Algorithms | v3 line-up + 13 new (DONE) |
| D11 | Scheduler training | phase 2, all at once, parallel notebooks on both accounts |
| D12 | Accounts | plus `satyapaladuguplus` (WSL CLI) = data + main line; second `satyapaladugu` (Windows CLI) = SQ-GAN |
| D13 | Class imbalance | class-balanced SPI loss + calibration + Critical-recall gate; natural mix in val/test; Critical-share sensitivity row |
| D14 | Duplicate labels (S2c) | rule-based: duplicate iff pHash ≤ 8 or SimHash ≤ 1; P(dup) mirrors the rule; paper says so |
| D15 | Relevance | precomputed per (image, mission) from `config/edge/missions.yaml` (4 missions); simulator picks each UE's mission |
| D16 | Tier mapping | **aligned with v3 §3**: affected_individuals -> Medium (High only in the `affected_high` sensitivity mapping). Humanitarian labels fill in where no damage label exists (14,556 of 18,082 images) |
| D17 | SQ-GAN v0 | no v0 pilot / pass for results (user, 2026-11-11). The v0 pilot that already ran is ignored |
| D18 | Relevance in SV-DPP | E[v] x (1 + beta (r_tilde - 0.5)), r_tilde per-mission normalised; beta tuned on val (0 = plan v3) |
| D19 | STS use | receiver fuses SPI + STS (logistic on train, temperature on val) into the tier belief all policies see; STS-masked header falls back to SPI (header ablation measures STS value) |
| D20 | Header bit depths (S6) | SPI 4x16 bit, relevance 16 bit, STS 13x4 bit; still 64 B. 8-bit SPI/relevance flipped 8.2%/7.9% of SV-DPP decisions (> 0.5% bar) |
| PARKED | Qwen feedback, S3 κ audit, SCOPE-AIF (S9) | set aside by the user; code/notebook exist (`edge-qwen-feedback`) |

## Rules
- PC: small inputs only, under `D:\Bunker\BaseCamp\_claude_scratch` (B, C1, sim table). HF cache: `HF_HOME` on D:. Never C:.
  Pytest: `--basetemp D:\...\pyt*`, delete afterwards; in PowerShell pass test files explicitly (no `*` expansion).
- Worker worktrees: `git worktree add --no-checkout` + `MSYS_NO_PATHCONV=1 git sparse-checkout set --no-cone '/*' '!/experiments/'`
  (without MSYS_NO_PATHCONV Git Bash breaks the pattern and the worktree is 2.2 GB).
- Kaggle: plus = `wsl -e /home/Satya/.venvs/semantic-kaggle/bin/kaggle ...` (from PowerShell); second = Windows `kaggle`.
  Titles must slugify to the kernel id. Secrets are read with `UserSecretsClient` and only reach editor runs.
- Commit to `main` only with the edge suite green; never edit `Archived/`.
- Leakage: test only in S10; gates on val; fine-tune reads train only; SPI test scores are never computed before S10.

---

# Status

## Datasets (owner satyapaladuguplus)
| Id | Slug | Contents | State |
|---|---|---|---|
| A | `scheduler-dataset-v1-images` | 27,168 images + index (2.5 GB) | DONE (public) |
| B | `scheduler-dataset-v1-labels` | labels (v3 tiers), counts, gates, `splits_v1.json`, split report, dup candidates | DONE (public; v3 = tier fix) |
| C1 | `scheduler-dataset-v1-cache` | 5-encoder embeddings, SimHash, pHash, SPI grid | DONE (public) |
| TAB | `scheduler-dataset-v1-simtable` | `simtable_v1.jsonl` (27,044 rows), `spi_head.npz`, `spi_report.json`, `manifest.json` | DONE (**private** — make public like the others if you want) |
| C2 | `scheduler-dataset-v1-sqgan` | STS, payload bytes, kept-token maps, latency per batch, LPIPS, mIoU (codec v1) | TODO (after fine-tune) |
| V1 | `scheduler-sqgan-v1` (second account) | fine-tuned SQ-GAN bundle | TODO (fine-tune running) |

## Done
- [x] S0 counts/gates · S1 labels (v3 mapping, D16) · S2 UE features (27,168/27,168)
- [x] S2b splits frozen: 124 val/test copies of train dropped (pHash ≤ 8 or SimHash ≤ 1). Final tiers (v3 mapping):
      train C 1,218 / H 1,047 / M 3,653 / L 10,056 · val C 499 / H 139 / M 429 / L 1,418 · test C 760 / H 492 / M 1,892 / L 5,441
- [x] S2c: 372 mined candidate pairs; labels by rule (D14)
- [x] S3 SPI final (`parameters/edge/simtable.py`): CLIP-B/32 logistic C=1, class-balanced, T = 0.95 on val; train scores out-of-fold.
      Val: macro-F1 0.567, **Critical recall 0.818** (gate 0.6), ECE 0.024; CrisisMMD-only Critical recall 0.818;
      everyday images 98.7% Low; no-normals ablation: false Critical on everyday 1.1% without vs 0.5% with normals.
      **Weak spot: High** (val recall 0.08, precision 0.14) — report it; the scheduler sees the posterior, not the argmax.
- [x] Relevance (D15): 4 missions × 27,044 images; values 0.52–0.68 (narrow, see S6)
- [x] Sim table built + published (TAB): tiers, alt tiers, calibrated SPI, relevance by mission, keys, 760 duplicate groups
      (1,965 rows), tweet-time capture for CrisisMMD; STS = [] and payload = 464 B **placeholders until C2**
- [x] P0 13 new policies + P0b learners in `train_cli` (dqn, scope_dqn, ltr lambdamart/ranknet, scope_ltr) + ctr_ungated headline
- [x] SQ-GAN code fixes (source clone bug, tests); fine-tune notebook on the second account
- [~] **SQ-GAN fine-tune RUNNING**: https://www.kaggle.com/code/satyapaladugu/edge-sqgan-finetune-v1 (v1, ≤ 10 h)

---

# Left to implement (Qwen / SCOPE-AIF parked)
| # | Item | Plan step | Where | Needs |
|---|---|---|---|---|
| I1 | **Payload reader for v1:** `Codec/integration/sqgan_payload.py` must apply `Codec/edge/seg_map.project_logits` when the bundle has `seg_classes.yaml` (teacher 150 ADE classes -> codec 20) | S4 | Codec | nothing; **must be done before the v1 pass** |
| I2 | Publish fine-tune output as `satyapaladugu/scheduler-sqgan-v1` (kernel_sources helper like `edge-publish-runs`), `bundle_check --print-codec-yaml` -> `codec.yaml` v1 | S4 | scripts, config | fine-tune done |
| I3 | SQ-GAN pass notebook on v1 bundle, full mode: STS (13), payload bytes at m 0.25/0.55, kept-token maps, decode latency samples per batch size, LPIPS, recon mIoU | S4 | Codec, notebook | I1, I2, I4 |
| I4 | **Recon mIoU** (SegFormer on reconstruction vs original) in `Codec/edge/codec_table.py` — LPIPS exists, mIoU does not (D6 gate) | S4 | Codec | nothing |
| I5 | `simtable.fill_codec`: merge C2 (STS, payload bytes, payload_source="sqgan_v1") + write `latency_v1.json` (LatencyProfile) from C2 timings | sim inputs | parameters | code now; run after I3 |
| I6 | S5: fit P(dup) on D14-labelled pairs (`receiver_models.fit_dup_model`) + cross-event ROC; save receiver model | S5 | Evaluation | nothing |
| I7 | S6: bit-depth flip test on val (`header64` study) for SPI + relevance now (decide relevance scaling: raw (cos+1)/2 uses ~40 of 256 levels), STS after C2 -> freeze `header.yaml` | S6 | parameters | I6 for dup; STS after I5 |
| I8 | **Trace-driven arrivals** in `run_sweep` (v3 §9): CrisisMMD in tweet-time order compressed by κ to the target ρ; everyday images Poisson-interleaved; realisations = permutation + κ + seed; MMPP option; mission per UE stream | S7 prep | Evaluation | nothing (today sweeps use synthetic Poisson/MMPP only; `compress_trace` is unused) |
| I9 | **Header-proof runner**: field drop-out (mask SPI / relevance / keys / STS), equal-bytes baseline (64 B thumbnail / embedding prefix), dedup P/R, overhead 64/(64+payload) | phase 2 prep | Evaluation | nothing |
| I10 | SCOPE minimal-proof regimes: wrong σ_W, noisy SPI (label noise on val), MMPP bursts; equal budgets, 5 seeds, learning curves | S8 prep | Scheduler/Evaluation | nothing |
| I11 | Live 2×T4 check: decode ~1,000 images with v1 on 2 GPUs (c = 2), 3 policies, KS test sim vs live latency, CTR@D delta with CIs | S10 prep | Evaluation + notebook | I3 for real run; code now |
| I12 | Phase-2 notebooks: `edge-scheduler-train` (one version per learner), `edge-scheduler-sweeps` (tuning, comparison, ablations, test), `edge-live-t4` | phase 2 prep | scripts, Notebooks | inputs from TAB + C2 |
| I13 | Sensitivity rows in sweep config: `tier_alt` mapping, Critical share, w ×0.5/×2, network loss 0–5 %, capture-time jitter | phase 2 prep | config + Evaluation | nothing |

# Next steps (in order)
1. **Now, in parallel while the fine-tune runs (code + tests, CPU):**
   - W-CODEC (Sonnet low): I1, I4, I5 (code), I3 notebook prep.
   - W-SIM (Sonnet low): I8, I9, I10, I13.
   - W-NB (Sonnet low): I11 (code), I12.
   - Me: I6, I7 (SPI + relevance part), integration, journal, review of every worker diff before it lands on `main`.
2. **When the fine-tune ends:** read its log -> I2 -> run I3 (v1 full pass, second account) -> D6 gate (LPIPS + mIoU vs original)
   -> I5 fill the table + latency -> I7 STS bits -> **header frozen**.
3. **BAR:** splits + header frozen; SPI final; TAB with C2 filled; latency profile; I8–I13 done; edge suite green on `main`.
4. **Phase 2 (Friday), all at once:** S7 tuning on val; S8 learners (parallel notebooks, both accounts); header proofs;
   S10 frozen test sweeps + live 2×T4; S12 paper.

# Notebook inventory (`Notebooks/Edge/`)
| Notebook | Account | Purpose | State |
|---|---|---|---|
| `dataset-semschedule` | plus | A, B, C1 + SPI grid | DONE |
| `edge-publish-runs` | plus | notebook output -> datasets | DONE |
| `edge-sqgan-finetune-v1` | second | codec v1 fine-tune | RUNNING |
| `dataset-semschedule-sqgan` | second | SQ-GAN full pass with v1 (I3) | needs I1–I4 |
| `edge-scheduler-train` / `-sweeps` / `edge-live-t4` | both / both / plus | phase 2 | TODO (I11, I12) |
| `edge-qwen-feedback` | plus | Qwen feedback | PARKED |
| `edge-tunnel` | either | SSH backup | READY |

# You
- [ ] Make `scheduler-dataset-v1-simtable` public if you want it like the others (I was not allowed to publish publicly)
- [ ] Revoke the API token pasted earlier in chat / the old notebook cell (if not done)
- [ ] Later: approve merged-branch deletions and deleting `edge-labels-v1`


# Status update 2026-11-13 (Friday)
## Done since the last update
- [x] SQ-GAN v1 fine-tune (authors' Cityscapes checkpoints; 10,434 train images; batch 2 = 1/T4; ~4,700 steps, 8 h cap)
- [x] True batched decode + latency (`latency_v1_true.json`): B 1/2/4/8 = 0.74/1.46/2.88/3.86 s; batching helps only at B=8
- [x] SV-DPP fixes (defer-wait floor, per-job tie tolerance `tie_tol_frac`); smoke: ties EDF at rho 0.8, trails at 1.2
- [x] D18 relevance factor; D19 receiver SPI+STS fusion (`parameters/edge/sts_fusion.py`, `--sts-fusion`)
- [x] SemCover implemented (analytic `semcover` + learned `--algo semcover`); design `docs/design/SEMANTIC_SCHEDULER_DESIGN.md`
- [x] Report scripts (`python -m Evaluation.edge.report codec-quality|results|header-proofs`); paper draft updated
- [x] Trained learners evaluable in sweeps (`learned:<algo>:<ckpt>`); trace arrivals; header proofs; regimes; sensitivity rows
## Running
- [~] SQ-GAN v1 resume pass (`edge-sqgan-v1-resume`): 43/55 shards copied, last 12 processing (~2 h)
- [~] W-LAUNCH: phase-2 launcher across both accounts (dry-run default), SemCover in notebooks, BCEdge-style baseline
- [~] W-OVER: SV-DPP overload gap with true latency on val (1,000 rows)
## Next (in order)
1. Pass ends -> `report codec-quality` (D6 gate) -> `simtable fill-codec` (STS, payload) + true latency -> republish TAB
2. `python -m parameters.edge.sts_fusion` (SPI vs SPI+STS on val) -> bit-depth test -> header frozen = BAR
3. Phase 2 via `scripts/edge/launch_phase2.py --go`: tune -> train (11 learners x 5 seeds incl. SemCover, BCEdge)
   -> compare (+ semcover, learned specs, `--sts-fusion`) -> header proofs -> test (once) -> live 2xT4
4. `report results` / `report header-proofs` -> fill paper placeholders
## Accounts
- plus `satyapaladuguplus`: WSL CLI. second `satyapaladugu`: native Windows `kaggle` CLI (login works). Max 2 GPU batch sessions each.

## Locked inputs (2026-11-13)
- Sim table `simtable_v1.jsonl` sha256 prefix `bd679b9e0e13ae82` (27,044 rows, STS + payload filled, true latency `latency_v1.json`, Critical-safe `sts_fusion.json`) — dataset `satyapaladuguplus/scheduler-dataset-v1-simtable` latest version. Do not regenerate.
- Header layout v2 (D20). Splits `splits_v1.json` (dataset B).
