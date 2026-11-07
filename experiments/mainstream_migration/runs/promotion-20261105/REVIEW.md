# Bounded mainstream promotion review

Outcome: no migration blocker found in the reviewed scope.

Reviewed mutable checkout based on `e9bf4e2f193f44ba274392706c0229d9fb003fac`, migration manifest `migration.json`, and all 31 promoted modules on 2026-11-05. Assessment only; no production, test, notebook or plan edits.

## Independent evidence

- Original SHA256 values for all 31 mappings match the exact Git HEAD bytes. Promoted SHA256 values match all current canonical files.
- Normalized AST comparison against HEAD originals: all 29 non-builder modules are identical after only namespace, repository-root depth and script-path substitutions. Only the two notebook builders require additional changes; inspected differences move active notebook/template/dependency paths and include canonical modules in source capsules. No numerical logic, model target, dataset selection or budget change found.
- Parsed all import statements in all 31 canonical modules, including both `import` and `from`: none import experimental implementations. Historical output paths and experimental regression-test references remain deliberately present.
- Independently decoded all three generated notebook capsules. Both paired capsules contain 208 entries; recovery contains 210. All 31 promoted modules match live canonical bytes exactly in every capsule. All included live Scheduler/Evaluation Python sources also match checkout bytes.
- Compared old and canonical metadata for both paired profiles and recovery: `id`, `dataset_sources`, `kernel_sources`, and `competition_sources` are unchanged.
- Inspected bootstrap root depth, compatibility wrappers, three console entrypoint functions, and builders' source-snapshot hashing/resume identity. Root depth resolves the same checkout; wrappers retain direct execution and alias canonical module objects.

## Other validation evidence

The lead reports 111 focused tests passed in 118.34 seconds (`tests-final.log`) and six extracted-capsule CLI checks passed (`capsule_smoke.json`). These results are supporting evidence supplied by the lead, not independently rerun here.

## Limits and non-blocking observations

- No GPU/model inference, dataset download, training, Kaggle submission or remote execution performed. Numerical equivalence is supported by normalized source equivalence and the focused regression suite, not a new training run.
- A broad import-identity probe stopped on the existing top-level argparse behavior of `prepare_teacher.py`; it performed no training/download. Arbitrary imports of every CLI module are not a supported general smoke check. This behavior predates promotion; representative alias identity is covered by the focused suite.
- Reviewed a mutable checkout, not a finalized commit. Recheck staged scope before commit, particularly historical long-path artifacts. No historical outputs were modified by this reviewer.
- The source-capsule builders retain their pre-existing template-based inventory; the checked generated capsules contain the full promoted runtime sources, but this review does not establish that every builder can itself rebuild notebooks inside an extracted runtime capsule.
