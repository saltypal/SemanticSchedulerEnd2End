# Targeted gap repair — 3 November 2026

## Authorized outcome
Repair and execute the failed UE-rate study and nine teacher-dependent learners. Preserve all fourteen completed D/annotation-RL models and the closed final test.

## Evidence and decisions
- The rate script failed because direct-file execution did not install the canonical namespaces. It now uses the existing runtime bootstrap and ROOT-qualified provenance paths.
- A tiny outside-repository CLI pilot passed80rows. The full CPU/WSL grid passed320rows across80traces:12/50/60/100UEs, fixed-total/fixed-per-UE loads, two burst settings and mixed/mission modes, four reference policies. Accounting and finite-value checks passed. Incident mode remains unavailable without annotations.
- Old Qwen candidate responses sometimes bound current-choice labels to recent-history images. Current A/B now occupy large labeled panels; recent-window and successful receiver-history evidence are separately labeled. Prefix/replay images have visible Evidence IDs matching the schedules.
- Recent windows are observations available by decision time, not necessarily earlier than the candidate or successfully delivered. Failed and unknown status remain distinct.
- The old prompt instructed unknown aspect ratings, mechanically removing VisionPrefer labels. The corrected prompt anchors all four rating dimensions and permits genuine unknowns. Partial known ratings are retained; the explicitly adapted VisionPrefer target averages the same >=2known dimensions in both orientations, including mission usefulness. Unknowns are never filled with zero. Common-dimension and strict-four support are reported.
- First8training candidate contexts are fixed for a repair comparison. Later candidate and prefix query contrasts are selected using permitted SPI/relevance/novelty features, not evaluator labels. Validation preference agreement is contrast-conditioned, not an unbiased queue outcome.
- Fresh teacher cache/provenance binds layout renderer, processor settings, greedy decoding, model revision, output grammar and image/request hashes. Rejected old answers are preserved, not reused.
- All32visual controls,80%orientation agreement after at least8contexts,16strict train/8strict validation support and reward-model validation thresholds are retained. Prompt/layout correction is not proof the teacher now qualifies.
- Recovery runs only seven P students plus two feedback RL arms. P and feedback workers use separate GPUs after teacher collection;50epoch/20,000decision maxima and one seed1729 remain. No new image extraction, SPI training, codec training or completed-D retraining occurs.

## Artifacts
- Full rate run: ../../../stream_scheduler/runs/gap-repair-20261102T221315Z/rate_cpu_full/REPORT.md
- Full rate evidence: ../../../stream_scheduler/runs/gap-repair-20261102T221315Z/validation/full/
- Original failure audit: teacher_pair_audit.md
- Assessment: TEACHER_REPAIR_ASSESSMENT.md
- Actual-data visual preflight: panel_preflight/
- Recovery notebook: ../../gap_recovery/WSL_Teacher_Recovery.ipynb
- Notebook/source identities: ../../gap_recovery/manifest.json

## Pending acceptance
Fresh GPU startup, teacher controls and orientation qualification, qualified label support, nine learner outcomes and recovered artifacts must be measured. A failed teacher or insufficient support remains blocked; it is not relabeled as success.
