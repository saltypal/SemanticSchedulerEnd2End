# Two-account scheduler notebooks — read before launch

Status (checked 3 November 2026): Both Kaggle runs are terminal COMPLETE. WSL version12 (`wsl-20261102T162419Z`) completed 10 supervised development-only evaluations; 9 teacher-dependent arms are blocked. WSL stage failures: Qwen preference qualification (8 comparisons, 4 consistent, agreement 0.50, not qualified) and the rate study import (`ModuleNotFoundError: Evaluation`). Windows version1 completed its assigned methods with no stage failures: four RL arms reached the configured 20,000 step/decision budget; Deep Sets initialization completed 50 epochs. Both reports explicitly do not establish scientific final evaluation. Details and small artifacts: `experiments/master_study/runs/paired-launch-20261102-205933/checks/paired-check-20261102T220621Z/CHECK_REPORT.md`.

## Notebooks and accounts

| Notebook | Launch account | GPU0 | GPU1 |
|---|---|---|---|
| wsl/Scheduler_WSL_Ranking_VLM.ipynb | satyapaladuguplus | ten supervised/ranking methods | Qwen labels, qualified P students and feedback RL |
| windows/Scheduler_Windows_RL.ipynb | satyapaladugu/semantic-decoder-scheduler-rl | Double DQN | Deep Sets initialization, PPO, TD-PER, warmstarted PPO |

Both execute in Kaggle Linux with T4x2. Windows/WSL identify the authenticated launch account, not the remote OS.

WSL methods:linear choice, MLP choice, Delay-Huber, priority classifier, RankNet, LambdaMART, Deep Sets RankNet, Set Transformer RankNet, tied-listwise variant, Temporal Deep Sets RankNet.
Conditional teacher methods:seven P students (linear/MLP/RankNet/Deep Sets/Set Transformer/Temporal Deep Sets/VisionPrefer), learned-reward RL and VLM-guided replay.
Windows' Deep Sets fit is an initialization prerequisite and frozen PPO comparator, not an independent extra seed. No cross-account completion dependency exists.

Eight reference policies are evaluated before learning:FIFO,RR,random,SPI-first,relevance,novelty,SPI+novelty,predicted EDF.
WSL also runs the baseline UE-rate study if training-window time remains.
Both profile exported actors on CPU for queue sizes4/16/64/256:selection milliseconds, weight/parameter size and supported-operator FLOPs. Partial operator FLOPs are not total FLOPs or a speed claim.

## Exactly what the run does

1. Initialize a unique run, timing budget and cell logging.
2. Extract and verify the SAME canonical source capsule in both notebooks.
3. Verify the mini archive and6.7MB shared producer/source cache.
4. Install pinned lightweight/model dependencies into the current kernel; retain managed Torch/CUDA.
5. Verify both CUDA allocations; run packaged scientific/controller/SQ01 packet checks.
6. Audit source/reference coverage. Run D only on assessable CrisisMMD silver-priority records.
7. WSL:strict-load the selected fine-tuned SQ-GAN, encode/decode one real image then five more, cache immutable digital packets and record timings. No codec training.
8. Execute assigned learners with the same split, seed, observed headers and validation trace definitions.
9. WSL:collect at most216 Qwen calls; stop an inconsistent scheduling teacher after an8-context paired pilot. P/reward/replay methods run only if their gates pass.
10. Save models, optimizer/RNG/episode state, portable exports, raw teacher answers, per-job logs and validation results.
11. Generate D comparisons, Critical-wait/miss safeguards, method statuses and plots.
12. Preserve partial work before the12hour platform limit.

This is a corrected scoped DEVELOPMENT study. It does not establish independent urgency truth, everyday-domain generalization, natural cross-view understanding or native SQ-GAN convergence.
Ordinary-source priority labels and independently supported mission-E references are missing. They are not silently invented.
A teacher failure is a recorded negative result; it does not authorize training on unreliable labels.

## Schedules and platform limit

- One fixed seed:1729.
- Supervised maximum:50epochs; early stopping after8 unimproved validation checks.
- RL maximum:20,000 action decisions per method; this is not20,000 episodes.
- Learner window ends at10.5hours from the first notebook cell.
- Reports/finalization target11.5hours, leaving30minutes before observed Kaggle12hour termination.
- Setup, downloads, codec checks and resume-copy time count against the same clock.
- Partial methods are labeled partial. Completion inside12hours is not guaranteed.
- Models checkpoint atomically. A parent watchdog is a last resort; it preserves the most recent completed checkpoint.
- Allocated-but-idle GPUs count against quota. Remaining quota/allocation must be checked immediately before launch.

## Inputs and access — essential

Both need:
1. Existing Mini_dataset_6000.tar, SHA2560f8c6186b09dcf6ab87911df642612919377157bb1585082bed1c93b1506fe2c.
2. inputs/Scheduler_shared_cache_20261102.tar, SHA2568bbea66e64e64817d03e23e6b5332f548771a54d425a2a698d75f62a7a5abb44.

The shared tar contains already-computed frozen image/mission embeddings and pinned SQ-GAN reference source. It contains no credentials and no scheduler weights.
No SPI/scene extraction or missing-CLIP computation is rerun. Context relevance and novelty are recomputed causally from mission and each UE's evolving history for every offered stream. Both accounts use identical cache bytes.

WSL versions10 and11 reuse `satyapaladuguplus/semantic-decoder-scheduler-20261030`. Version10's API push/run failed at codec preflight after Kaggle rejected `semantic-notebook-2`. Version11 removes that kernel source and references the completed image-pool kernel, private shared-cache dataset, and private flat SQ-GAN dataset `satyapaladuguplus/semantic-scheduler-sqgan-bundle-20261102`. The flat dataset version was uploaded with `--dir-mode skip`; its 29 remote filenames and sizes match the staged files, including `f015_last.pt` at 2,341,438,033 bytes. The sidecar restores the original bundle paths inside the notebook. Remote file content SHA256 was not re-downloaded; the notebook's own pinned-manifest and adapter checks remain the runtime verification gate. The earlier corrupt bundle upload remains preserved as failed evidence and is not attached.

Windows created version1 at the actual Kaggle identity `satyapaladugu/semantic-decoder-scheduler-rl` (Kaggle rewrote the requested dated slug from the title). Its private dataset version contains `Mini_dataset_6000.tar.bin` and `Scheduler_shared_cache_20261102.tar.bin`; both file sizes were verified. Dataset ID: `satyapaladugu/semantic-scheduler-paired-inputs-20261102`. The Windows account cannot read the plus account's private cache.

All datasets remain private. Kaggle unpacked the first `.tar` upload, so the notebook supports the original `.tar` or byte-identical `.tar.bin` form. The initial Windows dataset version is superseded by the `.tar.bin` version.

Metadata contains the attached input dataset IDs; consult each `kernel-metadata.json` and the launch receipt. Windows startup passed its two-T4, input, and 13-test checks. WSL version12 passed source/input/reference, packaged-test and real six-payload SQ-GAN decode gates; supervised training and teacher-request preparation have started.

## Recovery

Retrieve the whole run under:
/kaggle/working/experiments/scheduler_pair/runs/<profile>-<UTC>/

It contains source/input identities, sessions, cell logs, jobs, teacher data, checkpoints, reports, profiles and RUN_STATUS.
Each session has an incremental executed_notebook.ipynb with source/stdout/stderr; rich tables/figures are saved separately. Also retrieve Kaggle's executed notebook. Kernel death may lose the final native output buffer.

For same-protocol resume:
- Attach the previous RUN folder read-only.
- Set RESUME_INPUT to that exact folder.
- Keep the original input mount names.
- Use the SAME source capsule and verified input identities.
- The run copies into its original relative /kaggle/working path, preserving teacher request paths.
- Completed methods are not retrained; partial training restores optimizer/progress/RNG state where supported.
- Weights-only transfer is not called resume.
- A partially finished baseline rate study is preserved for inspection rather than automatically restarted.

For report-only recovery:
python -m experiments.master_study.paired_runtime --run <downloaded-RUN>
No learner is launched.

After retrieving BOTH runs:
python -m experiments.master_study.merge_paired_runs --wsl <WSL-RUN> --windows <WINDOWS-RUN> --output <NEW-MERGE-DIR>
It rejects different source/input/evaluator-trace identities. D cost and teacher preference agreement remain separate measurements.

## What has and has not been verified

Local:source/nbformat/AST checks, actual cache hashes, CPU first-three-cell execution in fresh processes, orchestration fixtures, numerical/export/resume tests, atomic-write interruption and missing-job failure accounting. The remote WSL SQ-GAN input inventory also matches all expected filenames and byte sizes; remote bytes were not re-downloaded for SHA256 verification.
Not established: independent urgency truth, teacher reliability, scientific final-test performance, broad-domain/generalization claims, or model superiority. Development training completed in both runs. The rate-study import and Qwen teacher qualification failures remain in the final reports; these require a code/data correction before any follow-up study.
The selected SQ-GAN was separately verified with real trained weights and real digital payloads. It remains lossy and its CRC detects errors rather than correcting them.

## Launch packaging correction
Kaggle dataset uploads unpack .tar files. Dataset inputs therefore use identical .tar.bin bytes; both notebooks accept either suffix and require the original content hashes. Original kernel-output tar inputs remain supported.



