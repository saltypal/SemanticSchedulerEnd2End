# EncDec

Read `PROJECT_MAP.md` for module boundaries and `docs/REBUILD_GUIDE.md` for commands.
Implementation lives here; notebooks import it. Defaults are `config/defaults.yaml`.
From the repository root: `python E2EMain.py --module EncDec --action validate`.
Tests: `python -m pytest EncDec/tests -q`.
Outputs: configured results directory / `EncDec` / unique run ID.
Environment: install `environment/requirements.txt` from the repository root.

`integration/latent_transport.py` exposes a prepared checkpoint transport
callback for INFRA's whole-latent engine. It loads once and validates each
selected cached payload. Unit tests use an explicit fake decoder; actual
checkpoint inference must be validated separately.

The teammate's consolidated DeepJSCC/NTSCC benchmark code lives in `Research/`.
`semcom_benchmark/` and `src/Evaluation/repaired_notebook_models.py` preserve
historical notebook imports. The canonical factory is
`EncDec.Evaluation.repaired_notebook_models:build_team_benchmark_codecs`.
Configure it through `dependencies.deepjscc`, `dependencies.ntscc`, or
`dependencies.external_worker`; the first two run one native model each, while
`external_worker` can run the native and codebook methods together. The
checkpoint-compatible source snapshots are under `references/`, and the
production `.pth` files remain explicit external inputs. `models/` also has
Git LFS assets that must be hydrated before use. See `README.md` section 14
and `docs/MAIN_MERGE_1E44BAA.md` for setup and validation limits.
