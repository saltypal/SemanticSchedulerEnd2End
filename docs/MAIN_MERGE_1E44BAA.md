# Main and NewSetup integration

The modular rebuild on `NewSetup` (`3253676`) was merged with the teammate's
`origin/main` consolidation (`1e44baa`). The merge retains both commit histories.
The resulting merge commit and publication status are recorded by Git; inspect
`git log --oneline --graph -5` and `git status -sb` for the current checkout.

## What was reconciled

| Incoming change | Final owner and treatment |
|---|---|
| DeepJSCC/NTSCC benchmark methods | `EncDec/Research/` owns the reusable implementation. `semcom_benchmark/` forwards historical notebook imports to it. |
| Repaired Kodak evaluator | `EncDec/Evaluation/repaired_notebook_models.py` owns the factory. `src/Evaluation/repaired_notebook_models.py` forwards the old import. The teammate's four methods and Kodak-excluding calibration logic remain. |
| Source snapshots | `references/DeepJSCC_checkpoint_source/` and `references/NTSCC_JSAC22_official/` remain versioned reference dependencies. |
| Model assets | Teammate's CIFAR DeepJSCC files, NTSCC calibration cache, and ADJSCC checkpoint remain in `models/` through Git LFS. Existing local Swin and codebook assets remain local and ignored. |
| Notebooks and runner scripts | Retained as historical experiment paths. New reusable work belongs to the owning module. |
| Root README | Modular architecture and run instructions remain primary; team model, asset and method details were integrated in section 14. |
| `ORGANIZATION.md` | Retained as the teammate's consolidation inventory and labeled historical where its ownership map differs. |

The merge had textual conflicts in `README.md` and the old `src/Evaluation`
evaluator. Each was reconciled through one canonical implementation instead of
maintaining two diverging copies. `.gitignore` now permits the teammate's tracked
LFS collection while excluding existing local Swin and codebook artifacts.

## Model availability and limits

The incoming `.pt` and `.pkl` files are Git LFS assets. A checkout with only
pointer files cannot load them. During the local `main` fast-forward, Git LFS
hydrated all 32 assets; their actual SHA-256 hashes match
`models/MANIFEST.sha256`. Other clones may still need `git lfs pull`. The two
production `.pth` checkpoints used by the repaired four-method comparison are
not in this Git repository: their expected locations and original hashes are
documented in `models/DeepJSCC/README.md` and `models/NTSCC/README.md`. The
ADJSCC checkpoint is archived without a compatible benchmark adapter.

Native DeepJSCC and NTSCC can each run alone through the configured EncDec
worker factory. The four-method comparison additionally needs both production
checkpoints, CIFAR-10 training data, the hydrated NTSCC calibration cache and
digital PHY dependencies. The NTSCC codebook uses only CIFAR records from that
cache; the cached Kodak records are excluded from fitting. The factory fails
early on missing files or unhydrated LFS pointers. No model quality result is
claimed until inference has run with those assets.

## Reproduce the integration checks

From the repository root in the compatible model environment:

```powershell
$env:STAGE1_MODEL_DIR = "models/SwinJSCC"
.\.venv-models\Scripts\python.exe -m pytest -q
python E2EMain.py --module EncDec --action availability
git lfs pull
```

The `git lfs pull` command is only needed for real LFS inference; it is not
required for contract tests. See `docs/VALIDATION_REPORT.md` for the measured
rebuild checks and the post-merge result. `PROJECT_MAP.md` records the current
ownership and boundaries.
