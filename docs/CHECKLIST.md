# Run checklist (check each box before the next step)

Rule for every step: the notebook clones the latest `main`, prints the commit SHA, runs a package entry
point, writes outputs under `/kaggle/working/runs/<step>/`, and the outputs that later steps need are
saved as a **Kaggle dataset** (new version each time). Windows Kaggle CLI only, account `satyapaladugu`.
Design: `docs/reviews/FINAL_PLAN_V3.md`. Status words: READY = code + unit tests exist; MISSING = must be
built first; UNVERIFIED = needs a real run to confirm.

## 0. Housekeeping before the repo goes public
- [x] Secret scan of tracked text: no keys or tokens found (the 3 hits are code that *excludes* `kaggle.json`).
- [x] Large-file scan: no tracked file over 20 MB; full history pack is about 600 MB (fine for GitHub).
- [ ] Delete merged branches (needs your OK; all three are fully contained in `main`, so nothing is lost):
  `git push origin --delete nandini-integration scheduler_improvement_trials sem_schedluerAI_trial2`
  and locally `git branch -d NewSetup claude/scheduler-params-policy-3a647b codex/archive-image-scheduling-20260927 semantic-awareAIscheduler merge-trial2`.
  Kept on purpose: `checkpoint-20261105-before-lfs-repair` (1 commit not in `main`) and any branch checked out
  in a worktree.
- [ ] Decide if personal paths in old docs (e.g. `C:\Users\...`) matter for a public repo.

## 1. Re-label the datasets and save them on Kaggle
| What | Status |
|---|---|
| Tier mapping from CrisisMMD human labels (`parameters/edge/labels.py`, CLI, 20 tests) | READY. Real counts already computed locally |
| Normal-image lists: Places365 (minus damage-lookalike classes) and people photos | MISSING: `parameters/edge/normals.py` (acquire + list + blacklist) |
| Publish outputs as a Kaggle dataset (`kaggle datasets create/version`) | MISSING: `kaggle_ops.py` only handles kernels; add `dataset` subcommands |
| Notebook `edge-s1-labels` | MISSING: build with `scripts/edge/build_notebook.py` |

Outputs to save (dataset `edge-labels-v1`): `labels_raw.jsonl`, `counts.json`, the 9 pinned CrisisMMD
metadata JSON files, `normals_places365.jsonl`, `normals_people.jsonl`.
Gate: test events >= 150 Critical (765 now), validation event >= 100 Critical (California 500).
Frozen after step 2b (splits).
- [ ] relabel run done, dataset uploaded, counts reviewed together

## 2. SQ-GAN on the T4 (profile + masker check)
| What | Status |
|---|---|
| `Codec/edge/sqgan_profile.py` (latency per batch size, VRAM, GFLOPs, masker keys) | READY (fakes only) |
| SQ-GAN source + v0 bundle available to Kaggle as datasets | MISSING: source zip exists in the SemanticLossless repo (`Kaggle/assets/sqgan_source`); the v0 bundle lives in the other Kaggle account's notebook output and must be republished under `satyapaladugu` |
| Real masker output format (`sample_index`) | UNVERIFIED: first thing this run checks |
- [ ] publish `edge-sqgan-source` and `edge-sqgan-v0` datasets (fill `config/edge/codec.yaml`)
- [ ] run `edge-s0-profile`; review `sqgan_v0_profile.json`
Note: decode is single-image, so "batch B" is B sequential decodes in one timed window (documented in the code).

## 3. Semantic masks / STS for all images: does it wait for the teammate? **No.**
The plan is two passes with the same code (S4 with codec v0, S11 with codec v1):
1. Now, with the existing v0 checkpoint: produces the STS, payload bytes and fidelity table for the whole
   pool. This is also how we find bugs early.
2. After her fine-tune: re-run the same notebook with `codec=v1`; header `version=2`; learned policies are retrained.
Cost of v0 first: one extra pass. Benefit: the whole pipeline and the first results do not block on her.
- [ ] STS table v0 (needs steps 1, 2 and the splits)

## 4. Teammate's SQ-GAN fine-tune: where is her notebook, and where does the model file go?
- **There is no notebook for this in this repo.** The training code is in the separate SemanticLossless
  project (`Semantic_Lossless/src/sqgan_train.py`, notebooks under `Notebooks/project_notebooks/Training/`).
  The earlier fine-tune (bundle `20261028T130100Z-eea4f3aa`) came from a Kaggle notebook on the other account.
- **Her inputs:** `splits_v1.json` + `labels` from step 1 (train images only; never val/test), and a segmentation
  map source (ADE20K-trained SegFormer merged to 20 project classes so SQ-GAN's 20-channel input is unchanged).
- **Her output (a bundle folder):** `bundle.json` (algorithm `sqgan`, `pipeline_completed: true`, per-file
  `path`/`bytes`/`sha256`), `last.pt`, `assets/checkpoint`. `sqgan_payload.py` verifies every file against `bundle.json`.
- **Placeholders:** `config/edge/codec.yaml` now has `v0` and `v1` entries (dataset slug, bundle subdir,
  `last_pt_sha256`, rates). She uploads the bundle as Kaggle dataset `satyapaladugu/edge-sqgan-v1`, and we
  fill the two TODO fields. Nothing else in the code changes.
- [ ] hand her the train-image list after step 2b; she trains; she uploads the bundle; we fill `codec.yaml`

## 5. Re-train all embeddings + classifiers, select the best, save everything
| What | Status |
|---|---|
| CLIP-B/32 + logistic head, calibration, ablation without normals (`parameters/edge/spi.py`) | READY (one encoder only) |
| Old comparison notebook (`experiments/spi_comparison_4000`: 4 encoders x 12 classifiers) | NOT reusable as is: trained on VLM pseudo-labels, and run v11 crashed out of disk space |
| Multi-encoder grid on human tiers with resumable saves | MISSING: `parameters/edge/embedding_grid.py` + notebook |

What the new grid must do: extract embeddings per encoder (CLIP-B/32, SigLIP2, DINOv2, ConvNeXtV2), save each
as it finishes (and free memory/disk before the next), train the classifier families on train events
(+ everyday images), pick by validation macro-F1 with a Critical-recall gate (>= 0.6), calibrate, and save
every embedding, head, per-model report and the chosen model as a Kaggle dataset (`edge-spi-v1`).
Constraint from the header: the UE-side encoder must be light, so the report also lists cost per image.
- [ ] build grid + notebook, run, review the comparison table, pick the UE encoder

## 6. After steps 1-5 (in order)
- [ ] 2b dedup across splits and freeze `splits_v1.json`; label ~400 duplicate pairs
- [ ] DINOv2 fidelity judge (train events) and the 300-image human audit sheet
- [ ] codec table v0 (STS, payload bytes, fidelity) and receiver models (duplicate model, predicted fidelity)
- [ ] header v1 freeze (bit-depth flip-rate test), then tune policies on the validation event only
- [ ] first simulator sweep on validation (core policies only): FIFO, EDF, SPI-first+EDF, M-LWDF, Whittle, EdgeServing, completion-aware DPP, SV-DPP, one learner, genie
- [ ] test-event sweeps (only after everything is frozen), then live 2xT4 check, then codec v1 re-run
- [ ] optional extras only if time: MPC, MILP bound, SCOPE variants, DQN/LTR, RLAIF, JPEG ladder
