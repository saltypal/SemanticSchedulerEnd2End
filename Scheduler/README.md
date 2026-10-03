# Scheduler

Read `PROJECT_MAP.md` for module boundaries and `docs/REBUILD_GUIDE.md` for commands.
Implementation lives here; notebooks import it. Defaults are `config/defaults.yaml`.
From the repository root: `python E2EMain.py --module Scheduler --action validate`.
Tests: `python -m pytest Scheduler/tests -q`.
Outputs: configured results directory / `Scheduler` / unique run ID.
Environment: install `environment/requirements.txt` from the repository root.

Whole-latent policies live in `latent/policies.py`. They accept immutable
payload snapshots and never select UEs. See `docs/STATIC_LATENT_SCHEDULER.md`
and `notebooks/04_static_latent_scheduler.ipynb` for the static workflow.

## Current package ownership

Canonical policy packages: `latent/` for whole-latent decisions, `baselines/` for original fixtures and `semantic/` for the earlier semantic/DQN interface. Policies never mutate lifecycle or import codecs/neural models.
