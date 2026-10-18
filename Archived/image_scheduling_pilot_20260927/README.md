# Archived image scheduling pilot

Archived on 2026-09-27 at the user's request. The four files in `manifest.json` retain their exact original bytes, including notebook outputs. They are historical reference material and are not intended to execute from these archive paths because their original imports and working-directory assumptions are preserved.

Recovery point: commit `45738cbe8730e614608eadb1e43d007476a1740a`, also named by local branch `codex/archive-image-scheduling-20260927`. That commit contains the original active file layout and the pre-existing `.gitignore` change. The former branch `semantic-awareAIscheduler` was not advanced by the snapshot. New work is on `semantic-aware-ai-scheduler`, the valid no-space form of the requested branch name.

Contents:

- Classifier comparison notebook using provisional LADI/task mappings.
- Traditional scheduler notebook using synthetic UE/channel metadata.
- Pilot image model adapters and their model download helper.

Reason for archival: Phase 1 now begins with manual image priorities, no UE abstractions, no classifier dependency in the scheduling baseline, and independently structured extraction evidence. The old scene-to-priority mapping is a hypothesis, not validated ground truth. The archive preserves it without endorsing it.

Existing EncDec, canonical extraction/policy implementations, checkpoints, downloaded models, datasets, and historical results were not moved. The `models/priority_classifiers/` ignore rule remains so downloaded weights do not accidentally enter Git. No remote push was performed by this archival step.

See `../../docs/PHASE1_EXTRACTION_RESEARCH.md` and `../../docs/PHASE1_IMPLEMENTATION_PLAN.md` for the replacement design. The original layout can be inspected with `git show 45738cb:SemParam/priority_models.py`; use a separate checkout of the recovery branch if you need to run the old notebooks without disturbing new work.
