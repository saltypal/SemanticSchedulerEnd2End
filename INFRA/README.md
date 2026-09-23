# INFRA

Read `PROJECT_MAP.md` for module boundaries and `docs/REBUILD_GUIDE.md` for commands.
Implementation lives here; notebooks import it. Defaults are `config/defaults.yaml`.
From the repository root: `python E2EMain.py --module INFRA --action validate`.
Tests: `python -m pytest INFRA/tests -q`.
Outputs: configured results directory / `INFRA` / unique run ID.
Environment: install `environment/requirements.txt` from the repository root.
