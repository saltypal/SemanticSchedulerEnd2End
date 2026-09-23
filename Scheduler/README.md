# Scheduler

Read `PROJECT_MAP.md` for module boundaries and `docs/REBUILD_GUIDE.md` for commands.
Implementation lives here; notebooks import it. Defaults are `config/defaults.yaml`.
From the repository root: `python E2EMain.py --module Scheduler --action validate`.
Tests: `python -m pytest Scheduler/tests -q`.
Outputs: configured results directory / `Scheduler` / unique run ID.
Environment: install `environment/requirements.txt` from the repository root.
