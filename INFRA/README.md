# INFRA

Read `PROJECT_MAP.md` for module boundaries and `docs/REBUILD_GUIDE.md` for commands.
Implementation lives here; notebooks import it. Defaults are `config/defaults.yaml`.
From the repository root: `python E2EMain.py --module INFRA --action validate`.
Tests: `python -m pytest INFRA/tests -q`.
Outputs: configured results directory / `INFRA` / unique run ID.
Environment: install `environment/requirements.txt` from the repository root.

`latent/contracts.py`, `latent/simulation.py` and `latent/evaluation.py` own
device-independent whole-latent lifecycle and metrics. Static orchestration is
`experiments/static_latent_experiment.py`; see `docs/STATIC_LATENT_SCHEDULER.md`.

## Current package ownership

Canonical lifecycle packages: `latent/` for whole image latents, `scheduling/` for earlier slotted/UE experiments, `data/` for manifests/caches and `learning/` for training. Flat migrated modules are compatibility aliases. The static CLI is `python -m INFRA.latent.cli`.
