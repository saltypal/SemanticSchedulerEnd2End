# Run handoff (read this first if you are a new/smaller model thread)

## Goal
Build the three Kaggle datasets A (images), B (labels), C (cache + classifier grid) with ONE notebook,
`satyapaladuguplus/dataset-semschedule`, then move on to the next checklist steps (`docs/CHECKLIST.md`).
Order: PILOT mode first -> read the report -> fix failures -> only then FULL mode.

## State when this file was written (2026-11-11)
- Code is on GitHub `main` (latest commit `54a084cb`). All 364+ edge tests passed before the push.
- Notebook `dataset-semschedule` version 1 was pushed in PILOT mode and was RUNNING on Kaggle
  (URL https://www.kaggle.com/code/satyapaladuguplus/dataset-semschedule , GPU 2 x T4, six datasets attached).
- A watcher (Monitor task `bzx7g6gjp`, script scratchpad `watch_kernel.ps1`) prints one line
  `KERNEL_DONE: ...` when it finishes. It lasts 30 minutes; if it expires while the notebook still runs, re-arm
  it with the Monitor tool (same command: `powershell -NoProfile -File <scratchpad>/watch_kernel.ps1`).

## When the pilot finishes
1. Spawn ONE Haiku (low) agent to read the small report files (never download big outputs; laptop has no disk):
   `kaggle kernels output satyapaladuguplus/dataset-semschedule -p <scratchpad>/pilot2_out --file-pattern "<regex for PILOT_REPORT.md, pilot_report.json, run summary, *.log, *done.json, counts.json, summary.json>"`.
   It reports: stage table, the unverified-items table with every FAIL's error tail, GPU info, per-image ms for
   SQ-GAN encode/decode and each embedder, counts.json gates, whether both GPUs were used (part0/part1).
2. Decide with this rule. The pilot is CLEAN only if ALL are true:
   - stages `A`, `B`, `C_features`, `C_grid` are PASS;
   - the SQ-GAN items pass: `bundle_flat_restore`, `sqgan_loads_strict`, `real_masker_output`, `decode_latency_gpu`;
   - all five embedders load (`load_embedder_clip/siglip2/dinov2_small/dinov2_base/convnextv2`);
   - the `feature_pass_*` smoke-verify items pass, including `feature_pass_segmap_matches_grid`;
   - both GPUs were used by the feature pass.
   Ignore (not blockers): Kaggle publish checks that SKIP because no Kaggle Secret is attached, and
   `kaggle_input_discovery` SKIP.
3. If CLEAN -> go to "Full run". If NOT clean -> "Fix loop". Never start the full run on a failing pilot.

## Fix loop
- For each FAIL write a precise prompt (file, function, exact error, expected behaviour, tests to add) and give it to a
  Sonnet (low) implementer. Do not implement yourself. Add a row to `docs/MISTAKES.md` for every error.
- When the implementer reports: run the edge tests in the background
  (`python -m pytest -q -p no:cacheprovider Scheduler/tests/test_edge_*.py parameters/tests/test_edge_*.py Evaluation/tests/test_edge_*.py`,
  about 2-3 min) and only if green: commit, `git push origin HEAD:main` (the notebook clones main).
- Re-push the same notebook as a new version (Haiku low runs `python -m scripts.edge.kaggle_ops push --step dataset-semschedule`
  from the repo root with PYTHONPATH=.), watch it again, read the report again.

## Full run
- `python -m scripts.edge.make_run_all_notebook --mode full` -> commit -> push main -> Haiku low pushes the notebook (new version) ->
  watcher -> when done, a Haiku low agent reads only the small summary files. Expected 5-10 hours; the run is resumable by shard.
- A, B, C only become real Kaggle DATASETS if a Kaggle Secret (`KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME`+`KAGGLE_KEY`) is attached to the
  notebook (only the user can add it). Without it, tell the user: open the finished notebook -> Save Version -> "New dataset from output".

## Standing rules (from the user)
- Kaggle work: WSL CLI, account `satyapaladuguplus`, call it from PowerShell as
  `wsl /home/Satya/.venvs/semantic-kaggle/bin/kaggle <args>` (Git Bash mangles /home paths). Kaggle ops and monitoring -> Haiku (low).
- Implementation -> Sonnet (low). The main thread writes detailed prompts and reviews; it does not code or babysit.
- NEVER download datasets or large outputs to the laptop (no disk). Datasets: get from Kaggle first, other sources only if Kaggle lacks them.
- Notebook logs must be clean (the user reads them on the Kaggle website).
- Commit to `main` only after the edge tests pass. No AI co-author or "Generated with" lines in commits or PRs.
- Do not delete branches or kernels without the user's explicit approval (a list of merged branches is waiting in `docs/CHECKLIST.md`).
- Do not edit `Archived/`, old notebooks, or old results. Keep `docs/MISTAKES.md` and `docs/CHECKLIST.md` updated.
- Do not change scope (dataset sources are: CrisisMMD v2.0 + Places365 val sample + COCO people, all non-CrisisMMD images tier Low).
- If unsure or something unexpected happens, stop and tell the user plainly instead of improvising.

## After datasets A/B/C exist (next in `docs/CHECKLIST.md`)
Freeze splits (cross-split near-duplicate removal), duplicate-pair labelling, DINOv2 fidelity judge + audit, receiver models, header freeze,
simulator sweeps on validation, then test events (only when everything is frozen). Teammate's SQ-GAN fine-tune notebook
(`Notebooks/Edge/edge-sqgan-finetune-v1.ipynb`, hand-off `docs/SQGAN_FINETUNE_HANDOFF.md`) runs after A and B exist.

## STATUS UPDATE (2026-11-11, usage limit approaching)
- Pilot 2 result (single notebook): A, B, embeddings (all 5 encoders), hashes, 2-GPU split, class-map caching all WORK.
  Failures: (1) the SQ-GAN source dataset has only `.upstream/sqgan`, not SemanticLossless `src/` -> 26 SQ-GAN items fail;
  (2) C_grid crashed (pilot sample had only Low images); (3) SQ-GAN stage skipped silently; (4) hard-coded old mount paths.
  Datasets mount at `/kaggle/input/datasets/<owner>/<slug>/`. Full details: `docs/MISTAKES.md` rows 27-30.
- DECISION: split into two notebooks. Notebook 1 `dataset-semschedule` = A + B + embeddings/hashes + grid (NO SQ-GAN).
  Notebook 2 `dataset-semschedule-sqgan` = SQ-GAN pass only, reads notebook 1 output via kernel_sources.
  SQ-GAN code source = the SemanticLossless GitHub clone at the pinned sha (same repo as the user's local Semantic_Lossless);
  weights = the v0 bundle dataset (restores fine). v1 weights come later from the teammate's fine-tune.
- Worker Y (Sonnet low; run_all, notebook generator, stratified pilot sample, grid tolerance): was running; check its report/tests.
- Worker X (Sonnet low; SQ-GAN source discovery, find_dataset, --only-sqgan/--require-sqgan): STOPPED by the user mid-way.
  It left ~320 UNTESTED lines uncommitted in Codec/edge/{compute_profile,feature_pass,feature_pass_backends,pilot_checks,sqgan_finetune}.py.
  Do not commit them without running the edge tests. Either finish X's remaining tasks (A-E in this file's history / the brief in
  MISTAKES rows 27-29) with a new Sonnet low worker, or `git checkout -- Codec/` to drop them.
- Nothing is committed since `54a084cb`. This file and docs/MISTAKES.md are uncommitted edits.
- Next when usage allows: finish X, run the edge tests, commit+push, regenerate both notebooks (`python -m scripts.edge.make_run_all_notebook --which ab-embed|sqgan --mode pilot|full`),
  push notebook 1 in pilot mode via a Haiku low agent, read PILOT_REPORT.md, then full mode.
