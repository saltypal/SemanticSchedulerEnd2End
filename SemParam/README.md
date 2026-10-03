# SemParam

Read `PROJECT_MAP.md` for module boundaries and `docs/REBUILD_GUIDE.md` for commands.
Implementation lives here; notebooks import it. Defaults are `config/defaults.yaml`.
From the repository root: `python E2EMain.py --module SemParam --action validate`.
Tests: `python -m pytest SemParam/tests -q`.
Outputs: configured results directory / `SemParam` / unique run ID.
Environment: install `environment/requirements.txt` from the repository root.

`features/latent.py` supplies image/task relevance and confirmed-receiver
novelty for whole-latent scheduling. Importance and priority are never guessed
from evaluation labels; see `docs/STATIC_LATENT_SCHEDULER.md`.

## Current package ownership

Canonical extraction lives in `features/`; confirmed knowledge in `history/`; semantic serialization in `headers/`; semantic codebooks in `Codebooks/`. Flat migrated modules are compatibility aliases.
