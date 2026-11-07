# Whole-latent policies

`policies.py` provides FIFO, SJF, EDF, priority FIFO, priority EDF and semantic
selection. It accepts immutable latent snapshots and returns one whole job ID.
It imports no codec or neural framework and changes no lifecycle state.

Earlier fixture policies live in `Scheduler/baselines`; the previous semantic
and DQN policy interface lives in `Scheduler/semantic`.
