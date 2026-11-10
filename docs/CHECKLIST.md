# Run checklist (check each box before the next step)

Rules for every step:
- Nothing is downloaded or stored on the laptop. All data work runs on Kaggle (internet on).
- Each notebook clones the latest `main` (the repo is public), prints the commit SHA, runs a package entry
  point, and writes a `run_manifest.json`. Outputs later steps need are published as **Kaggle datasets**.
- Kaggle CLI: **WSL**, account `satyapaladuguplus` (the Windows account is busy). New slugs are approved for
  this study; later runs are new versions of the same slug.
- Publishing from inside a notebook needs a Kaggle Secret that only you can add (notebook Add-ons ->
  Secrets: `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` + `KAGGLE_KEY`). Without it a notebook still finishes and
  tells you to use "Save Version" -> "New dataset from output" in the web UI.
- Design: `docs/reviews/FINAL_PLAN_V3.md`. Status: DONE / READY (code + unit tests, not yet run on Kaggle) /
  MISSING / UNVERIFIED (needs a real Kaggle run).

## The three datasets
| Dataset | Slug (owner satyapaladuguplus) | Contents | Built by |
|---|---|---|---|
| A | `scheduler-dataset-v1-images` | raw images only: CrisisMMD v2.0 (HF, pinned revision), Places365 val_256 scenes, Open Images people; `index.jsonl`, manifest, failures | notebook `edge-a-download-images` |
| B | `scheduler-dataset-v1-labels` | tier labels applied to A (rules in `config/edge/data.yaml`), counts, gates; no images | notebook `edge-b-labeling` |
| C | `scheduler-dataset-v1-cache` | embeddings (5 encoders), SimHash + pHash, SQ-GAN kept-token map, class grid, 13-value mask summary, pooled encoder grid, payload bytes, decoded JPEGs, per-stage timings, FLOPs, grid + selection results | notebook `edge-s2-feature-pass` |
Also existing on the account: `edge-labels-v1` (labels only, superseded by B; delete when you say so),
`semantic-scheduler-sqgan-bundle-20261102` (v0 SQ-GAN bundle, flat layout), `semantic-lossless-private-sqgan-source`,
`semantic-lossless-private-sqgan-assets`.

## 0. Housekeeping
- [x] Secret scan: no keys or tokens in tracked text. Large-file scan: no tracked file over 20 MB.
- [x] Repo is publicly readable (anonymous GitHub API returns 200).
- [x] Merge teammate Trial 2 branch into `main`; restore archived `SemParam`; line-ending rules.
- [ ] Delete merged branches (needs your OK; contained in `main`): remote `nandini-integration`,
  `scheduler_improvement_trials`, `sem_schedluerAI_trial2`; local `NewSetup`,
  `claude/scheduler-params-policy-3a647b`, `codex/archive-image-scheduling-20260927`,
  `semantic-awareAIscheduler`, `merge-trial2`. Kept: `checkpoint-20261105-before-lfs-repair` (1 unique commit).
- [ ] Decide whether personal paths in old docs matter for a public repo.
- [ ] Add the Kaggle Secret to the notebooks that publish (A, B, C).

## 1. PILOT FIRST (one GPU notebook, `edge-pilot`) -> read results -> fix -> only then the full runs
The pilot runs every stage on about 20 images per source and writes `PILOT_REPORT.md` with pass/fail per stage
and a table "unverified item -> check -> result". Stages: download sample (A) -> label it (B) -> feature pass
smoke (embeddings, hashes, SQ-GAN encode/decode, masks, timings, FLOPs) -> grid export.
Checks cover everything earlier reported as unverified:
- GPU/disk/library versions; DataParallel on 2 GPUs; flat v0 bundle restore and strict weight load; deep key scan
- real masker output (`sample_index`, kept-token count vs m x 512, whether scores are exposed), m mapping,
  pre-mask grid shape, pooled-feature size
- decode latency per batch size, VRAM, SQ-GAN FLOPs; compute-profile hooks on every real model
- all 5 embedders load and match their expected dimensions; DINOv2 judge path
- ADE20K teacher label names vs `seg_classes.yaml`; real-logit projection
- fine-tune plumbing: pinned-commit fetch, `.upstream/sqgan` extraction, real trainer pilot launch, 256x256
  image rejection; payload reader vs class-map gap
- dataset size extrapolation vs the 20 GB limit; memory at 16 workers; download throughput
- Kaggle publish round trip (small private test dataset), secrets, status parsing, zip mode with many files;
  `/kaggle/input` auto-discovery (only verifiable when A/B are attached as datasets)
- [ ] pilot code finished and reviewed (in progress)
- [ ] push pilot notebook to Kaggle (WSL) and run it
- [ ] read `PILOT_REPORT.md`; list every FAIL; fix in code; re-run the failing stages
- [ ] only when the pilot is clean: continue below

## 2. Full dataset builds (CPU, internet on)
- [ ] A: run `edge-a-download-images`; review `failures.json` and size vs 20 GB; publish A
- [ ] B: run `edge-b-labeling` on A; gates: test Critical >= 150 (765 in the local pass), validation >= 100
  (California 500); publish B
Labels recap: CrisisMMD v2.0 human labels -> Critical (severe damage, or injured/dead/missing people),
High (mild damage; affected individuals, infrastructure/utility damage, vehicle damage),
Medium (little/no damage; rescue/other relevant info when informative), Low (not informative; all everyday
images). Validation event = California wildfires (Sri Lanka had only 71 Critical); test = Maria + Mexico.

## 3. Cache + grid (C), GPU
- [ ] run `edge-s2-feature-pass` (sharded, resumable) on A + B + v0 bundle; review the grid comparison
  (5 encoders x 5 classifier families, gate: Critical recall >= 0.6 on validation); pick the UE encoder; publish C
- [ ] cross-split near-duplicate removal and `splits_v1.json` (frozen); label about 400 duplicate pairs

## 4. Teammate SQ-GAN fine-tune (codec v1)
READY as a scaffold, never run: `Notebooks/Edge/edge-sqgan-finetune-v1.ipynb`, hand-off in
`docs/SQGAN_FINETUNE_HANDOFF.md`, class map in `config/edge/seg_classes.yaml` (rows marked `# DECISION:` need her call).
Her output is a bundle dataset `satyapaladuguplus/scheduler-sqgan-v1`; we fill `config/edge/codec.yaml` (v1).
Known risks: Places365 images are 256x256 and the trainer rejects parents under 256x512 (policy decision:
drop or upscale); the fallback source dataset may lack SemanticLossless `src/`; `sqgan_payload.semantic_map` must
call `seg_map.project_logits` for bundles that ship `seg_classes.yaml`.
- [ ] pilot checks of the fine-tune plumbing pass
- [ ] hand her the train-image list after splits are frozen; she trains and uploads; fill `codec.yaml`
- [ ] re-run the feature pass with `codec=v1` (header version 2) and retrain learned policies

## 5. Everything after (core first; extras only if time)
- [ ] DINOv2 fidelity judge + 300-image human audit; receiver models; header v1 freeze (bit-depth test)
- [ ] simulator sweep on validation: FIFO, EDF, SPI-first+EDF, M-LWDF, Whittle, EdgeServing, completion-aware DPP,
  SV-DPP, one learner, genie reference; tune on validation only
- [ ] test-event sweeps (frozen policy set), live 2xT4 check, codec v1 re-run
- [ ] optional extras: MPC, MILP bound, SCOPE variants, DQN/LTR, RLAIF, JPEG ladder

## Status of code (all unit-tested on CPU, none run on Kaggle yet)
Simulator, ~24 policies, SV-DPP, UE-side modules, SPI head, grid, learners, dataset A/B builders, feature pass,
fine-tune scaffold, Kaggle tools (WSL + datasets), pilot orchestrator. Pilot checks are being added.
Two old tests still fail on Windows (`test_handoff_tools`, `test_paired_notebook_capsules...`): preserved
notebooks embed CRLF copies of sources; left untouched by design.
