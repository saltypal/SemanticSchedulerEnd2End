# Parameter handoff for the policy designer

The current spatial design is in
[parameters/docs/SPATIAL_PARAMETER_DECISION.md](../parameters/docs/SPATIAL_PARAMETER_DECISION.md).
Use task-relevant count and box extent conditionally alongside the core
parameters. The old aggregate outputs and the new contract are not interchangeable.
No scheduling algorithm was changed or run for this documentation decision.

# Scheduling implementations

Source inventory after relocation; none was trained or executed by this change.
Implemented source is not evidence of superiority or new-header compatibility.

| Family | Source | Requirements |
|---|---|---|
| FIFO, round robin, random | decoder_scheduler.py; scene_policy.py | Candidate queue |
| SJF, EDF, priority FIFO/EDF, aging | latent/policies.py; semantic/policies.py | Service cost/deadline/class where applicable |
| SPI class ordering | spi_policy.py | Available priority class |
| Semantic weighted/lexicographic/greedy | semantic/policies.py | Declared feature contract |
| Direct importance and hand-written heuristic | decoder_scheduler.py | Historical eight-field header |
| Linear, MLP, classifier, regressor | decoder_scheduler.py | Compatible exported model |
| Linear/contextual marginal-gain models | contextual_decoder_scheduler.py; training/decoder_context_training.py | Context/history and fitted artifact |
| Signature deferral/overlap penalty | contextual_decoder_scheduler.py; scene_policy.py | Permitted signatures/history |
| Linear/MLP choice, RankNet, priority classifier | training/scene_learners.py: fit_supervised | Independent examples/preferences |
| LambdaMART | training/scene_learners.py: fit_lambdamart | Grouped ranking targets and LightGBM |
| LightGBM inference | decoder_scheduler.py | Matching booster artifact |
| Masked Double DQN, TD/PER variants | training/scene_learners.py: train_dqn | Legal actions, environment, reward |
| Masked PPO | training/scene_learners.py: train_ppo | Legal actions, environment, reward |
| Historical DQN | semantic/policies.py; training/dqn_training.py | Matching observation/model contract |
| Teacher-feedback/replay experiments | training/scene_teacher*.py; scene_learners.py | Explicit teacher provenance and compatible environment |
| Privileged marginal-utility reference | semantic/policies.py | Evaluator-only facts; never a deployable policy |

`training/scene_learners.py` retains arms linear_choice, mlp_choice, ranknet,
lambdamart, priority_classifier, masked_ppo, masked_double_dqn, td_per,
rl_vlm_f, vlm_replay and visionprefer. These share machinery; their names do not
establish independent new algorithms.

Previously discussed contextual bandits, Deep Sets and explicit two-step
lookahead are not implemented as finalized policies here. They remain optional
research proposals. Existing neural/RankNet/PPO code is preserved rather than
inventing untrained substitutes. A universal adapter and final SPI-based
training/evaluation contract are still required before a fair common comparison.
