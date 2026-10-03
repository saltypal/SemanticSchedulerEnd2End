# Utilities

Run from the repository root: `python -m scripts.<group>.<name>`.
Old flat script paths remain compatibility launchers; add new logic to the
owning group, never to a launcher.

| Group | Purpose |
|---|---|
| `codecs` | Model setup and communication benchmarks |
| `data` | LADI curation, review and packaging |
| `notebooks` | Reproducible notebook generation |
| `reports` | Plots and document generation |
| `kaggle` | Kaggle staging and experiment packaging |
| `scheduler` | Saved-run evaluation, replay and legacy training utilities |
| `semparam` | Semantic codebook utilities |
| `maintenance` | Graph refresh and audited migration tooling |

Whole-latent execution belongs to `python -m INFRA.latent.cli`.
`maintenance/restructure_repository.py` records the one-time migration;
do not rerun its apply mode on an already migrated tree.
