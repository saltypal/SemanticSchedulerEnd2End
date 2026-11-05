# Main scheduler suite

This is the canonical implementation of the completed parameter-based training
and evaluation system. It replaces implementation ownership under
`experiments/stream_scheduler`. Historical commands there forward to these modules.

- `train_real_suite.py`: conventional policies, supervised methods and annotation-reward RL.
- `semantic_suite_data.py`: cached observations, splits and causal parameter inputs.
- `ue_rate_study.py`: dynamic UE workload and rate comparisons.
- `profile_exports.py`: exported policy profiling.
- `train_teacher_rankers.py`, `train_feedback_rl.py`, `qwen_teacher.py`: retained optional supervision paths, with their existing qualification gates.
- `validate_system.py`: verification driver.

Run from the repository root in the existing scheduler Python environment:

```sh
python -m Scheduler.training.stream_suite.train_real_suite --help
python -m Scheduler.training.stream_suite.ue_rate_study --help
```

The receiver environment remains `Evaluation/semantic_learning_env.py`; deployed
NumPy inference remains `Scheduler/semantic_learning_policy.py`; model and RL
implementations remain directly under `Scheduler/training`.

Promotion changes code ownership and paths. It changes no objective, training
budget, model weights, dataset, evaluation split or runtime information boundary.
Runs and exploratory analyses remain under `experiments/`.
