# Whole-latent lifecycle

`contracts.py` defines immutable jobs/snapshots/delivery evidence.
`simulation.py` owns arrivals, whole-job service, deadlines and receiver history.
`evaluation.py` reports ranking, deadline, resource and confirmed-evidence metrics.
`experiment.py` adapts caches and persists policy comparisons.
`cli.py` provides `python -m INFRA.latent.cli` and `latent-scheduler`.

Policies live in `Scheduler/latent`, extraction in `SemParam/features`, and
codec transport in `EncDec/integration`. See `docs/STATIC_LATENT_SCHEDULER.md`.
