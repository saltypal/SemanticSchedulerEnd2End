# Implementation Guide: Semantic Decoder Scheduler (plan v3.1)

The design is in `docs/reviews/FINAL_PLAN_V3.md`. This file covers *how to build it*: who does what,
in which order, where code lives, and how Kaggle runs it. Every worker, human or model, reads this
first.

## 0. House rules (non-negotiable)
1. **Modular.** One responsibility per module.
   - No notebook-only logic: notebooks only *call* package code.
   - No copy-paste between modules.
   - Config lives in YAML under `config/edge/`, never hard-coded.
2. **GitHub is the source of truth.**
   - All work is committed and pushed to `main` of `saltypal/SemanticSchedulerEnd2End`, or to a short
     feature branch merged into `main`.
   - No `claude/…` branch names.
   - **No AI co-author or collaborator lines** in commits or PRs.
3. **Kaggle runs notebooks only.** Each notebook's first cell does a fresh
   `git clone --depth 1` (or `git pull`) of `main`, then `pip install -e .`, then calls a package entry
   point. Results are written to `/kaggle/working/runs/<step>/<run_id>/` and copied back to the repo by
   the ops worker.
4. **Reuse before writing.** Every new module states, in its docstring, what it reuses (§3).
5. **Leakage rules** (plan §10): test events are touched only in S10/S11, every gate uses val, and
   splits are frozen at S2b.
6. Preserve history: do not edit `Archived/`, old runs or old notebooks (AGENTS.md).

## 1. Worker roles (model routing)
| Role | Model / effort | Does | Does not |
|---|---|---|---|
| Architect | Opus (this chat) | Plan changes, interface decisions, research questions | Routine coding |
| Implementer | Sonnet, low | Writes modules and tests from the task cards in §4 | Change interfaces without the architect |
| Reviewer | Sonnet, medium | Reviews every PR against §0 and the card's acceptance tests; runs `pytest` | Write features |
| Ops / runner | Haiku, medium | Kaggle CLI (both accounts), notebook push, status polling, pulling outputs, running tests, logging in `parameters/KAGGLE_RUNS.md` | Edit package code |
| Teammate | Human | Parallel tracks in §5 | — |

Loop for every card: implement → test (ops) → review → merge to `main` → ops runs on Kaggle if
needed.

## 2. Target package layout (new code; canonical owners kept)
```
config/edge/                     # YAML: data.yaml, header.yaml, sim.yaml, policies.yaml, train.yaml
parameters/edge/                 # UE side (stateless)
  labels.py        tier mapping from CrisisMMD human labels (S1)
  features.py      CLIP-B/32 embeddings, relevance, SimHash, pHash (S2)
  splits.py        cross-split near-dup removal, event-disjoint splits (S2b)
  dup_pairs.py     candidate dup-pair export / labelled-pair loader (S2c)
  spi.py           SPI head train/calibrate/predict (S3)
  sts.py           Semantic Token Summary from SQ-GAN masker output (S4)
  header64.py      64-B pack/unpack/CRC-16, bit-depth study (S6)
Codec/edge/
  sqgan_profile.py latency distribution per batch, GFLOPs, VRAM (S0)
  codec_table.py   encode → STS/payload → decode → judge → u, LPIPS (S4)
  jpeg_ladder.py   fallback codec table (S4 gate)
Evaluation/edge/
  judge.py         DINOv2 linear-probe fidelity judge (S3b)
  receiver.py      receiver memory, P(dup) model, û model (S5)
  arrivals.py      ArrivalSource: tweet chronology, Poisson, MMPP (S7)
  env.py           event-driven trace simulator, c workers, batching (S7)
  metrics.py       CTR@D, per-tier success, value, false purges, bootstrap CIs (S7)
  milp_bound.py    offline clairvoyant bound on short traces (S7)
Scheduler/edge/
  base.py          Policy interface: select(state) -> (batch_ids, purge_ids)
  classic.py       FIFO, Random, RR-mission, EDF, TUF/DASA, priority+aging, M-LWDF, equivalence set
  theory.py        gen-cμ/h, Whittle-Aalto, completion-aware DPP (2608.05537), per-tier DPP
  edgeserving.py   stability-score lookahead (adapted)
  mpc.py           Monte-Carlo rollout
  sv_dpp.py        SV-DPP v3.1 (+ genie-aided variant)
Scheduler/training/edge/
  scope.py         SCOPE objective (Lagrangian + annealed imitation)
  ppo.py dqn.py ltr.py bc_dagger.py
  rlaif.py         Qwen feedback collection, Bradley–Terry reward model
Notebooks/Edge/    one thin notebook per Kaggle step (built from a template)
tests/edge/        unit tests per module + tiny fixtures
```

## 3. Reuse map (existing code to adopt, fixing what is broken)
| Need | Reuse | Action |
|---|---|---|
| SQ-GAN encode/decode | `Codec/integration/sqgan_payload.py` (main) | Use as is; wrap a batch-decode API for profiling |
| Masker output capture | `Codec/integration/sqgan_prefeatures.py` (branch `sem_schedluerAI_trial2`) | Cherry-pick; extend the hook to capture `masker()['sample_index']` and the 20-class map |
| Parity test | `Codec/integration/prefeature_parity_smoke.py` (teammate) | Cherry-pick into `tests/edge/` |
| Fixed header + CRC pattern | `parameters/spi_scene_packet.py` (teammate) | Template for `header64.py` (new 64-B layout) |
| Staged timing simulator | `Evaluation/scheduler_codec_replay.py`, `trial_scheduler_packet.py` (teammate) | Reuse stage timing in `env.py`; drop the per-source state |
| LambdaMART / RankNet / PPO / DQN machinery | `Scheduler/training/scene_learners.py` (main) | Reuse model and optimiser parts in `ltr.py`, `ppo.py`, `dqn.py`; new env interface |
| Old env | `Evaluation/semantic_learning_env.py` | **Do not reuse** for results (unit service, circular cost); keep for the legacy-metric row only |
| Kaggle notebook builder | `Scheduler/training/study/build_paired_notebooks.py`, `Notebooks/SchedulerSuite/snapshot_template.ipynb` | Adapt to a git-pull template (`Notebooks/Edge/_template.ipynb`) |
| Kaggle run log | `parameters/KAGGLE_RUNS.md` | Append; keep existing slugs and versions (AGENTS.md) |
| CLIP SPI pipeline | `parameters/minimal_parameters.py`, SPI comparison code | Reuse embedding extraction; retrain the head on new labels |
| Mini 6k cache | `experiments/mini_dataset/runs/mini_spi_20261030_v1/downloads/Mini_dataset_6000.tar` | Local fixture source for tests only |

**First merge (card C00):** cherry-pick the teammate files above from `origin/sem_schedluerAI_trial2`
into `main`, with tests passing, before anything else.

## 4. Task cards in build order (plan §10, expanded)
Each card lists: inputs → outputs → acceptance test. ⟂ marks a card that can run in parallel with the
previous one.

**Phase A: foundation (local, no GPU)**
- **C00 Merge teammate reusables.** → files from §3 on `main` → their tests pass.
- **C01 Package skeleton + config loader.** Create the dirs in §2, `config/edge/*.yaml`, and
  `common/edge_config.py`. → `pytest tests/edge -q` green on an empty suite.
- **C02 Kaggle notebook template.** `Notebooks/Edge/_template.ipynb` and `scripts/edge/build_notebook.py`:
  - cell 1: clone/pull `main`, print the commit SHA;
  - cell 2: `pip install -e .`;
  - cell 3: `python -m <entrypoint> --config ...`;
  - cell 4: write `run_manifest.json` (SHA, config hash, timings).
  → acceptance: build a notebook for a dummy entry point; ops pushes it to the existing slug and the log
  shows the SHA.
- **C03 Ops scripts (Haiku).** `scripts/edge/kaggle_ops.ps1` and `.sh`: push, status, wait, pull
  outputs, for **both** accounts. Selection by `-Account win|wsl`. → dry run lists kernels on both.

**Phase B: S0–S2c data and features**
- **C10 S0 data verification.** `parameters/edge/labels.py --verify` on Kaggle:
  - counts per event / tier / source;
  - tweet-id presence;
  - Places365 blacklist candidates.
  → `counts.json`; gate: ≥ 150 test Critical, Sri Lanka ≥ 100 Critical (else switch val).
- **C11 ⟂ S0 SQ-GAN profile.** `Codec/edge/sqgan_profile.py` on a T4:
  - masker dict keys;
  - latency samples for B ∈ {1, 2, 4, 8} (≥ 200 each);
  - GFLOPs (fvcore), peak VRAM, encoder ms.
  → `sqgan_v0_profile.json`.
- **C12 S1 labels.** Tier mapping (plan §3) + alternative mapping. → `labels_raw.parquet`; unit tests on
  synthetic label rows for every rule branch.
- **C13 S2 UE features.** CLIP-B/32 embeddings, relevance (mission registry), 64-bit SimHash (secret
  seed from config), pHash. → `ue_features.npz`; test: deterministic hashes for a fixture.
- **C14 S2b dedup + splits.** Cross-split near-dup removal; event-disjoint splits.
  → `splits_v1.json` (**frozen**); test: no test image within the thresholds of any train image.
- **C15 S2c dup-pair candidates.** Export about 400 pairs for team labelling, plus a loader. → `dup_pairs.csv`.

**Phase C: S3–S6 models and header**
- **C20 S3 SPI.** Class-balanced LR + temperature scaling; "no normals" ablation; κ vs. Qwen;
  per-source report. Gate: val Critical recall ≥ 0.6.
- **C21 ⟂ S3b judge.** DINOv2 linear probe on train events. → val accuracy.
- **C22 S4 codec v0 table.** Per image: STS (`sts.py`), payload bytes, decode, u = judge(recon) =
  judge(orig), LPIPS. Resumable, sharded over both accounts. Gate: u not ≈ 0, else run the JPEG ladder.
- **C23 S4 human audit sheet.** Export 300 (orig, recon) pairs → CSV for the team to label.
- **C24 S5 receiver models.** û logistic model; P(dup) logistic model; cross-event ROC.
- **C25 S6 header64.** Pack/unpack/CRC-16; bit-depth flip-rate study. → layout frozen; round-trip tests
  on 10k random headers.

**Phase D: S7 simulator and analytic policies (local CPU)**
- **C30 arrivals.py + env.py + metrics.py.**
  - Sampling without replacement; empirical latency sampling; c workers; batch actions; purge; delivered-only memory.
  - Tests: conservation (every arrival ends exactly once as decoded, purged, expired or in queue);
    determinism by seed; c = 1 hand-computed toy trace.
- **C31 classic.py + equivalence check.** Tests: toy traces where the known equivalences hold.
- **C32 theory.py, edgeserving.py, mpc.py.**
- **C33 sv_dpp.py** (+ genie). Tests: the degenerate case gives expected-weight priority; Z stable on a feasible toy; batch ratio rule.
- **C34 milp_bound.py** (OR-Tools / PuLP) on ≤ 200-job traces.
- **C35 Val tuning run.** All analytic policies on val; choose V, η and B rules. → `val_tuning.json`.

**Phase E: S8–S9 learning**
- **C40 Plain learners** (PPO, DQN, LambdaMART, RankNet) + BC + DAgger on the new env.
- **C41 SCOPE** (`scope.py`) + SCOPE-PPO / DQN / LTR; minimal proof experiment (plan §7) with equal budgets, 5 seeds.
- **C42 ⟂ RLAIF collection** (`rlaif.py collect`) on Kaggle: Qwen2.5-VL pairwise (both orders) +
  pointwise log-probs on train images. **Can start right after C14.**
- **C43 Reward model + SCOPE-AIF (i) and (ii).**

**Phase F: S10–S12**
- **C50 Freeze + test sweep** (plan §8.3 sweeps) on test events; bootstrap CIs; figures.
- **C51 Live validation** on 2×T4 (c = 2, 1,000 images, 3 policies, KS test).
- **C52 Codec v1 swap** (after the teammate delivers the checkpoint): re-run C22 → C24 → C35 → C40–C43 → C50 with `codec=v1`, header `version=2`.
- **C53 Paper sync** (`docs/paper/DRAFT_PAPER.md`).

## 5. Parallel work for the teammate (no collisions)
| When | Teammate task | Interface |
|---|---|---|
| Now | Review C00 cherry-picks (her code) and fix anything she knows is stale | PR to `main` |
| After C14 | **SQ-GAN v1 fine-tune**: ADE20K-SegFormer maps merged to 20 project classes; train events + Places365 train only; start from the v0 checkpoint | Delivers only a checkpoint + bundle manifest (same format as v0) |
| After C15 | Label about 400 duplicate pairs (with you) | `dup_pairs.csv` |
| After C23 | Human audit of 300 reconstructions (with you) | audit CSV |
| After C14 | Supervise the C42 RLAIF prompt design (she has the Qwen/Nemotron prompt experience) | prompt file in `config/edge/rlaif.yaml` |
| Anytime | Mission-profile registry review (`docs/reviews/MISSION_PROFILES_DRAFT.md` → `config/edge/missions.yaml`) | YAML |

## 6. Kaggle protocol
- Accounts:
  - **win** = Windows Kaggle CLI;
  - **wsl** = WSL Kaggle CLI.
  - Shard large jobs (C22, C42) across both, by image-id hash parity.
- Notebook identity: one slug per step (`edge-s0-profile`, `edge-s4-codec-table`, …). New runs are
  **new versions** of that slug. Record slug, version and SHA in `parameters/KAGGLE_RUNS.md`.
- Every notebook prints the commit SHA, and refuses to run if the working tree differs from `main`
  HEAD.
- Outputs are pulled by ops into `experiments/edge/<step>/<run_id>/`. Large binaries go through Git
  LFS or stay as Kaggle datasets; record their hashes.

## 7. Definition of done (every card)
- Code lives in its module (§2), and the config lives in YAML.
- Unit tests pass, and the reviewer approves.
- If a GPU is needed, the Kaggle run is recorded with its SHA.
- `FINAL_PLAN_V3.md` / `DRAFT_PAPER.md` are updated if a decision changed.
- A dated, factual entry is added to `parameters/JOURNAL.md`.
