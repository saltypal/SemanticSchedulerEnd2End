# Agent collaboration contract

Read `PROJECT_MAP.md` before changing this repository. Then read the owning module's README, defaults and tests. The user's explicit task takes precedence over these notes.

## Boundaries

- Implement codecs and communication codebooks in EncDec; extraction and semantic codebooks in SemParam; policies in Scheduler; shared contracts, channels, simulation and orchestration in INFRA.
- Policies receive immutable snapshots and budgets. Only INFRA mutates UE queues/clock/job lifecycle. Lower class integer means higher priority. Do not infer class labels from images or names.
- Do not import EncDec or neural frameworks from Scheduler. Image-only SemParam methods must run without a codec checkpoint. Inject `INFRA.features.FeatureProvider` for codec-dependent methods.
- `src/` is compatibility only. Edit canonical implementations, never introduce duplicate logic into wrappers. Preserve alias class identity.
- Do not change models, dataset files, historical results, pinned upstream source, or a teammate's uncommitted work as a side effect of refactoring.
- New results belong to configured results/module/run-id. Save resolved configuration, commit, dirty state, source digest, environment and measurement outcomes.
- Keep latent storage bytes, digital bits and complex channel uses distinct. Missing evidence stays unavailable; failed reconstruction has no quality score. Hardware/runtime comparisons require matching provenance.
- Root config validation must reject invalid options before execution. Keep imports lazy. Unsupported accelerators must fail explicitly.
- Active notebooks import module code and persist output outside disposable checkouts. `notebooks/historical/` and root `Notebooks/` are reference material.

## Change validation

Run owning-module tests first, then affected regressions and integration tests. For architecture changes update PROJECT_MAP and migration documentation. Refresh Graphify with `scripts/refresh_graph.py`; record exact source digest because an uncommitted working tree is not represented by HEAD alone.

Do not claim trained-model validation without actual checkpoint inference. Do not turn synthetic scheduler results into performance or novelty claims. Document external models that are unavailable.
