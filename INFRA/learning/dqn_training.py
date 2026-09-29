"""Small standard DQN training; Scheduler receives exported NumPy weights only."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from time import monotonic
from typing import Mapping, Sequence

import numpy as np

from INFRA.event_simulation import SemanticSchedulingEnvironment, run_episode
from INFRA.semantic_contracts import (
    EpisodeSpec, PolicyDecision, ResourceBudgetSemantic, SemanticCandidateSnapshot,
)
from INFRA.semantic_evaluator import normalized_utility_auc, priority_delivery_auc
from Scheduler.feature_adapter import MAX_CANDIDATES, STATE_DIM, model_state


@dataclass(frozen=True, slots=True)
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 10_000
    replay_warmup: int = 500
    target_update_steps: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_fraction: float = 0.7
    episodes: int = 300
    validate_every: int = 25
    max_decisions: int = 60
    gradient_clip_norm: float = 10.0
    max_seconds: float | None = 2400.0

    def __post_init__(self) -> None:
        if self.episodes <= 0 or self.batch_size <= 0 or self.replay_capacity < self.batch_size:
            raise ValueError("DQN episode and replay sizes must be positive and compatible")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Epsilon bounds must satisfy 0 <= end <= start <= 1")
        if not 0 < self.epsilon_fraction <= 1 or not 0 < self.gamma <= 1:
            raise ValueError("DQN epsilon fraction and discount are invalid")
        if self.max_seconds is not None and self.max_seconds <= 0:
            raise ValueError("DQN max_seconds must be positive or None")


def build_q_network():
    """Import PyTorch only when training is requested."""
    from torch import nn
    return nn.Sequential(
        nn.Linear(STATE_DIM, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, MAX_CANDIDATES),
    )


def _export_numpy(network, destination: Path) -> None:
    layers = (network[0], network[2], network[4])
    arrays = {}
    for index, layer in enumerate(layers, start=1):
        arrays[f"w{index}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{index}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(destination, **arrays)


def _epsilon(config: DQNConfig, episode_index: int) -> float:
    progress = min(1.0, episode_index / max(1.0, config.episodes * config.epsilon_fraction))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def _validate(network, episodes: Sequence[EpisodeSpec], budget: ResourceBudgetSemantic,
              device: str, *, enable_delivered_history: bool) -> float:
    import torch

    class GreedyPolicy:
        name = "dqn_validation"

        def reset(self, episode_metadata=None):
            pass

        def observe(self, outcome):
            pass

        def decide(self, snapshot: SemanticCandidateSnapshot,
                   budget: ResourceBudgetSemantic) -> PolicyDecision:
            state, mask = model_state(snapshot, budget)
            with torch.no_grad():
                values = network(torch.from_numpy(state).to(device)).cpu().numpy()
            chosen = int(np.argmax(np.where(mask, values, -np.inf)))
            return PolicyDecision(self.name, snapshot.now_ms, chosen,
                                  float(values[chosen]), "greedy validation Q")

    scores = []
    network.eval()
    for episode in episodes:
        result = run_episode(episode, GreedyPolicy(), budget,
                             enable_delivered_history=enable_delivered_history)
        value = normalized_utility_auc(
            result["utility_curve"], horizon_ms=result["horizon_ms"],
            maximum_utility=result["maximum_utility"])
        if value is not None:
            scores.append(value)
    network.train()
    return float(np.mean(scores)) if scores else 0.0


def train_dqn(
    train_episodes: Sequence[EpisodeSpec],
    validation_episodes: Sequence[EpisodeSpec],
    budget: ResourceBudgetSemantic,
    output_dir: str | Path,
    *, seed: int = 42,
    config: DQNConfig | None = None,
    provenance: Mapping[str, object] | None = None,
    device: str = "auto",
    enable_delivered_history: bool = True,
) -> dict[str, object]:
    """Train one seed and save best/last training and inference checkpoints."""
    import torch
    from torch import nn

    settings = config or DQNConfig()
    if len(train_episodes) < settings.episodes:
        raise ValueError("Provide at least config.episodes pre-generated training episodes")
    if not validation_episodes:
        raise ValueError("DQN selection needs fixed validation episodes")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random_source = random.Random(seed)
    online = build_q_network().to(device)
    target = build_q_network().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    use_data_parallel = device.startswith("cuda") and torch.cuda.device_count() > 1
    online_batches = nn.DataParallel(online) if use_data_parallel else online
    target_batches = nn.DataParallel(target) if use_data_parallel else target
    optimizer = torch.optim.Adam(online.parameters(), lr=settings.learning_rate)
    replay = deque(maxlen=settings.replay_capacity)
    optimization_steps = 0
    best_validation = float("-inf")
    history: list[dict[str, float | int]] = []
    action_histogram = [0] * MAX_CANDIDATES
    provenance_data = dict(provenance or {})
    provenance_data.setdefault("feature_schema", f"track-a/v2:{STATE_DIM}-64-32-25")
    provenance_data["dqn_data_parallel"] = use_data_parallel
    provenance_data["dqn_gpu_count"] = torch.cuda.device_count() if device.startswith("cuda") else 0
    training_started = monotonic()

    def save_checkpoint(stem: str, episode_index: int) -> None:
        checkpoint = {
            "online": online.state_dict(), "target": target.state_dict(),
            "optimizer": optimizer.state_dict(), "episode": episode_index,
            "optimization_steps": optimization_steps,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (),
            "python_rng_state": random_source.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "config": asdict(settings), "seed": seed,
            "provenance": provenance_data,
        }
        torch.save(checkpoint, output / f"{stem}.pt")
        _export_numpy(online, output / f"{stem}.npz")

    completed_episodes = 0
    stopped_by_time_budget = False
    for episode_index in range(settings.episodes):
        environment = SemanticSchedulingEnvironment(
            train_episodes[episode_index], budget,
            enable_delivered_history=enable_delivered_history)
        snapshot = environment.snapshot()
        episode_reward = 0.0
        losses: list[float] = []
        decisions = 0
        epsilon = _epsilon(settings, episode_index)
        while snapshot is not None and not environment.done:
            state, mask = model_state(snapshot, budget)
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                raise RuntimeError("Simulator exposed no eligible DQN action")
            if random_source.random() < epsilon:
                action = int(random_source.choice(valid.tolist()))
            else:
                with torch.no_grad():
                    values = online(torch.from_numpy(state).to(device)).cpu().numpy()
                action = int(np.argmax(np.where(mask, values, -np.inf)))
            next_snapshot, reward, done, _ = environment.step(action)
            if done or next_snapshot is None:
                next_state = np.zeros(STATE_DIM, dtype=np.float32)
                next_mask = np.zeros(MAX_CANDIDATES, dtype=np.bool_)
            else:
                next_state, next_mask = model_state(next_snapshot, budget)
            replay.append((state, action, reward, next_state, next_mask, done))
            action_histogram[action] += 1
            episode_reward += reward
            decisions += 1
            snapshot = next_snapshot
            if len(replay) >= max(settings.replay_warmup, settings.batch_size):
                batch = random_source.sample(replay, settings.batch_size)
                states = torch.from_numpy(np.stack([item[0] for item in batch])).to(device)
                actions = torch.tensor([item[1] for item in batch], dtype=torch.int64, device=device)
                rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
                next_states = torch.from_numpy(np.stack([item[3] for item in batch])).to(device)
                next_masks = torch.from_numpy(np.stack([item[4] for item in batch])).to(device)
                terminals = torch.tensor([item[5] for item in batch], dtype=torch.bool, device=device)
                current = online_batches(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_values = target_batches(next_states).masked_fill(~next_masks, -torch.inf)
                    maximum = next_values.max(dim=1).values
                    maximum = torch.where(terminals | ~next_masks.any(dim=1), 0.0, maximum)
                    expected = rewards + settings.gamma * maximum
                loss = nn.functional.smooth_l1_loss(current, expected)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), settings.gradient_clip_norm)
                optimizer.step()
                optimization_steps += 1
                if optimization_steps % settings.target_update_steps == 0:
                    target.load_state_dict(online.state_dict())
                losses.append(float(loss.item()))
            if decisions > settings.max_decisions:
                raise RuntimeError("Episode exceeded the configured decision cap")
        row: dict[str, float | int] = {
            "episode": episode_index + 1, "seed": seed,
            "reward": float(episode_reward), "epsilon": float(epsilon),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "decisions": decisions, "optimization_steps": optimization_steps,
            "invalid_action_count": 0,
        }
        episode_result = environment.result(policy_name="dqn_training")
        training_auc = priority_delivery_auc(episode_result)
        row["training_priority_delivery_auc"] = 0.0 if training_auc is None else training_auc
        if ((episode_index + 1) % settings.validate_every == 0
                or episode_index + 1 == settings.episodes):
            validation = _validate(online, validation_episodes, budget, device,
                                   enable_delivered_history=enable_delivered_history)
            row["validation_semantic_utility_auc"] = validation
            if validation > best_validation:
                best_validation = validation
                save_checkpoint(f"dqn_seed{seed}_best", episode_index + 1)
        history.append(row)
        completed_episodes = episode_index + 1
        if (settings.max_seconds is not None
                and monotonic() - training_started >= settings.max_seconds):
            if "validation_semantic_utility_auc" not in row:
                validation = _validate(online, validation_episodes, budget, device,
                                       enable_delivered_history=enable_delivered_history)
                row["validation_semantic_utility_auc"] = validation
                if validation > best_validation:
                    best_validation = validation
                    save_checkpoint(f"dqn_seed{seed}_best", completed_episodes)
            stopped_by_time_budget = True
            break
    save_checkpoint(f"dqn_seed{seed}_last", completed_episodes)
    serializable_history = [
        {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
         for key, value in row.items()} for row in history
    ]
    (output / f"dqn_seed{seed}_history.json").write_text(
        json.dumps(serializable_history, indent=2, allow_nan=False),
        encoding="utf-8")
    (output / f"dqn_seed{seed}_actions.json").write_text(
        json.dumps({"seed": seed, "action_index_counts": action_histogram}, indent=2),
        encoding="utf-8")
    manifest = {
        "seed": seed, "configured_episodes": settings.episodes,
        "completed_episodes": completed_episodes,
        "stopped_by_time_budget": stopped_by_time_budget,
        "elapsed_seconds": monotonic() - training_started,
        "best_validation_semantic_utility_auc": best_validation,
        "enable_delivered_history": enable_delivered_history,
        "config": asdict(settings), "provenance": provenance_data,
        "training_checkpoint": f"dqn_seed{seed}_best.pt",
        "numpy_inference_weights": f"dqn_seed{seed}_best.npz",
    }
    (output / f"dqn_seed{seed}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def train_dqn_seeds(
    train_episodes_by_seed: Mapping[int, Sequence[EpisodeSpec]],
    validation_episodes: Sequence[EpisodeSpec], budget: ResourceBudgetSemantic,
    output_dir: str | Path, *, config: DQNConfig | None = None,
    provenance: Mapping[str, object] | None = None,
    device: str = "auto",
    enable_delivered_history: bool = True,
) -> dict[int, dict[str, object]]:
    return {seed: train_dqn(episodes, validation_episodes, budget, output_dir,
                            seed=seed, config=config, provenance=provenance,
                            device=device,
                            enable_delivered_history=enable_delivered_history)
            for seed, episodes in train_episodes_by_seed.items()}
