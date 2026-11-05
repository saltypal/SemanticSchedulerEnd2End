# Scheduler

All existing scheduling policy implementations and scheduler training live here.
See ALGORITHMS.md for exact entry points and evidence boundaries.

- baselines/ and latent/: classical policies and historical whole-latent policies.
- decoder_scheduler.py: eight-field scalar-header policy family.
- contextual_decoder_scheduler.py: completed-history-aware policies.
- scene_policy.py: variable-feature policy inference and model artifacts.
- spi_policy.py: SPI class policy helpers.
- semantic/: older semantic/DQN policy family.
- training/: scheduler fitting, selection, PPO/DQN and teacher experiments.

These policy families have different contracts. The new parameter header still
requires a deliberate adapter; moving sources does not make them interchangeable.
Models require compatible trained artifacts. No tests or training ran in this
reorganization.
