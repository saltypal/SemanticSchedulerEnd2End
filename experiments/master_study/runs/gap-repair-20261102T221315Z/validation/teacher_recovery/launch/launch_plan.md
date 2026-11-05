# Teacher recovery launch plan

- Notebook: experiments/master_study/gap_recovery/WSL_Teacher_Recovery.ipynb
- Source revision: 032c23a3dafc7bfeb39a84ba55193b6b6f3a22a0
- Notebook SHA-256: 2fdea70f7df416957dadd91cd77ce5ac0bc0ca91f80a4726388e99edf830d30c
- Embedded capsule SHA-256: a824d45cb146d850e709a67d29ded1aa2fa8dd3ca1db8248a82a4548fa3291e8; all 182 embedded source files match the capsule manifest.
- Existing private kernel: satyapaladuguplus/semantic-decoder-scheduler-20261030; v12 is COMPLETE. Local metadata keeps only the paired private dataset and image-pool source kernel, with no SQ-GAN bundle.
- Preflight: master study 28/28, teacher/supervisor 20/20, scene learning 48/48; source hash diff is empty. Notebook nbformat and AST checks pass. Three setup/input cells passed CPU smoke using symlinked inputs; no training or weights download.
- Planned run: teacher-recovery-wsl-v13-20261102T2243Z; expected Kaggle version 13. Request NvidiaTeslaT4; first runtime gate must observe exactly two CUDA devices or refuse training.
- Scope: the nine teacher-dependent arms only. Preserve the 14 previously completed non-teacher arms. Keep 32 controls, 0.8 swap consistency, and minimum 16/8 train/validation support unchanged. Do not create incident labels.
- Caps: 50 epochs with patience 8; at most 20,000 decisions per feedback arm; stop training by 10.5 hours, finalize by 11.5 hours, platform cap 12 hours.
- Inputs: satyapaladuguplus/semantic-scheduler-paired-inputs-20261102 plus satyapaladuguplus/semantic-image-pool-scene-cache-20261030; reuse verified archive/cache identities.
- Expected runtime output root: /kaggle/working/experiments/scheduler_pair/runs/. Retrieve executed notebook, cell logs/errors, provenance, teacher requests/panels, first inference evidence, checkpoints, and metrics when available; verify local persistence.
- Kaggle quota: not exposed by installed CLI; unavailable in this check.
- Launch command, stdout, stderr, exit status, accepted version/run, and follow-on status evidence are recorded beside this plan.