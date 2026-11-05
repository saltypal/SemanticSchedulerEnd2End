# Scene93 temporal scheduling protocol revision

Date: 30 October 2026. Protocol: SCHED-20261030-v2.

Canonical document: docs/decoder_context/SEMANTIC_RANKING_RESEARCH_DESIGN.md
Final document SHA256: c60f10f4b792cb7c9065a6b24e1ad7531d5ad0a19814ff9a4ea2c29aa05e9752

## Delivered

- Revised the existing canonical plan, preserving v1 as MASTER.before.md and the supplied upstream specification as SUPPLIED_SCENE_SPEC.txt. No replacement codebase.
- Adopted final93-class scene v2:369-byte scene block,374 wire-derived scene features,387-entry augmented semantic vector. Historical headers/payloads remain distinct.
- Added VLM-only preference-supervised shared GRU plus Deep Sets as the first experiment. All sixteen former registry IDs, their D/E losses and the RL/RLAIF/replay methods remain;17 IDs now documented.
- Added broad source manifests and separate mixed-IID, real causal-sequence and mission-context trace families. Sources are researched choices, not downloaded assets.
- Specified receiver-causal three-record UE windows, bounded ten-image teacher requests, matched PAIR_VIEW training/probes, full-queue generalization evaluation and same-label temporal controls.
- Preserved216 teacher-call maximum, six aggregate GPU-hour ceiling, local-first execution, source grouping and independent/silver reference boundaries.

## Checks actually run

- Read actual parameters/scene_arrays.py, scene_vocabulary.py and the supplied final v2 specification.
- Static regression confirmed all sixteen old model IDs and exact D/E/PER equations retained; teacher-call allocation remains216.
- Luna High ran14 bounded checks /648 counted cases against the existing pure scene modules: dimensional arithmetic, masks/status, examples, quantization, byte/record roundtrip, malformed-block rejection, received-feature semantics and two-way choice/pair-loss equivalence. See validation/receipt.json and run.log.
- Producer source hashes were unchanged. No new scheduler adapter or temporal learner is validated by these checks.
- Initial assessment found two plan inconsistencies (teacher-visible context and an unallocated extra ablation). Main fixed both; a focused second review was unavailable because the selected model was at capacity. Main static checks verified the edits; no successful second-assessment claim.
- First local validation-harness parsing error is preserved in validation/first_attempt_failure.json; corrected harness passed.

## Not executed

No detector or VLM inference, model training, dataset acquisition, codec work, dependency installation or Kaggle operations. No claim of broad scene accuracy, temporal-policy correctness, convergence or superiority.

## Reproduction

From the canonical repository root with its existing .venv:

```powershell
& .\.venv\Scripts\python.exe experiments/semantic_priority_design/runs/scene93-temporal-v2-20261030/validation/validate_scene_contract.py
```

The optional revise_design.py deterministically regenerates the canonical document from the preserved v1 source. Do not rerun it over later user edits; the checked final document is already present. canonical.diff records the revision. freeze.json and manifest.json record source identity and scope.

The next implementation consumes this revised protocol, starting with actual scene ingress, causal header windows, the simulator and qualified pair labels. The first fixed teacher allocation has only32 unique candidate pairs (16train/8validation/8locked pilot test before abstentions), so it is feasibility-scale. A large image catalog is not a large qualified supervision set.
