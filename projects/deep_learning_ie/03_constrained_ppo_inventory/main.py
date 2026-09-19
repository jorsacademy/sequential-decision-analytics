from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


SEED = 23
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class Config:
    horizon: int = 60
    capacity: int = 20
    episodes: int = 300
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 6
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    shortage_target: float = 0.05
    lagrange_lr: float = 0.08


class ProductionInventoryEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(SEED)
        self.t = 0
        self.inventory = 0.0
        self.prev_demand = 10.0
        self.current_mean = 10.0

    def _mean_demand(self, t: int) -> float:
        seasonal = 3.0 * np.sin(2.0 * np.pi * t / 12.0)
        regime = 2.5 if (t // 20) % 2 else 0.0
        return max(2.0, 10.0 + seasonal + regime)

    def _state(self) -> np.ndarray:
        return np.array(
            [
                np.clip(self.inventory / 30.0, -2.0, 2.0),
                self.current_mean / 20.0,
                self.prev_demand / 20.0,
                self.cfg.capacity / 20.0,
            ],
            dtype=np.float32,
        )

    def reset(self) -> np.ndarray:
        self.t = 0
        self.inventory = 0.0
        self.prev_demand = 10.0
        self.current_mean = self._mean_demand(self.t)
        return self._state()

    def step(self, action: int) -> tuple[np.ndarray, float, float, bool]:
        production = float(np.clip(action, 0, self.cfg.capacity))
        demand = float(self.rng.poisson(self.current_mean))

        available = max(self.inventory, 0.0) + production
        shortage = max(demand - available, 0.0)
        shortage_rate = shortage / max(demand, 1.0)

        self.inventory = self.inventory + production - demand

        holding_cost = 0.4 * max(self.inventory, 0.0)
        backlog_cost = 2.8 * max(-self.inventory, 0.0)
        production_cost = 0.9 * production
        operating_cost = holding_cost + backlog_cost + production_cost

        self.prev_demand = demand
        self.t += 1
        done = self.t >= self.cfg.horizon
        self.current_mean = self._mean_demand(self.t)

        return self._state(), operating_cost, shortage_rate, done


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.policy = nn.Linear(64, n_actions)
        self.value = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.body(x)
        return self.policy(z), self.value(z).squeeze(-1)


def compute_gae(
    rewards: list[float],
    values: list[float],
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_adv = 0.0
    next_value = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        last_adv = delta + gamma * lam * last_adv
        advantages[t] = last_adv
        next_value = values[t]

    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def train() -> tuple[ActorCritic, Config, float]:
    cfg = Config()
    env = ProductionInventoryEnv(cfg)
    model = ActorCritic(state_dim=4, n_actions=cfg.capacity + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    lagrange_multiplier = 1.0

    for episode in range(cfg.episodes):
        states: list[np.ndarray] = []
        actions: list[int] = []
        old_log_probs: list[float] = []
        values: list[float] = []
        shaped_rewards: list[float] = []
        shortage_rates: list[float] = []

        state = env.reset()
        done = False

        while not done:
            state_t = torch.from_numpy(state).unsqueeze(0)
            with torch.no_grad():
                logits, value = model(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, cost, shortage_rate, done = env.step(int(action.item()))
            reward = -cost - lagrange_multiplier * shortage_rate

            states.append(state)
            actions.append(int(action.item()))
            old_log_probs.append(float(log_prob.item()))
            values.append(float(value.item()))
            shaped_rewards.append(float(reward))
            shortage_rates.append(float(shortage_rate))

            state = next_state

        advantages, returns = compute_gae(
            shaped_rewards,
            values,
            cfg.gamma,
            cfg.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.from_numpy(np.asarray(states, dtype=np.float32))
        actions_t = torch.tensor(actions, dtype=torch.long)
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32)
        advantages_t = torch.from_numpy(advantages)
        returns_t = torch.from_numpy(returns)

        for _ in range(cfg.ppo_epochs):
            logits, value_pred = model(states_t)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            unclipped = ratio * advantages_t
            clipped = torch.clamp(
                ratio,
                1.0 - cfg.clip_eps,
                1.0 + cfg.clip_eps,
            ) * advantages_t
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            value_loss = torch.mean((value_pred - returns_t) ** 2)
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        mean_shortage = float(np.mean(shortage_rates))
        lagrange_multiplier = max(
            0.0,
            lagrange_multiplier
            + cfg.lagrange_lr * (mean_shortage - cfg.shortage_target),
        )

        if (episode + 1) % 50 == 0:
            mean_cost = -float(np.mean(shaped_rewards))
            print(
                f"episode={episode + 1:03d} "
                f"mean_shaped_cost={mean_cost:.3f} "
                f"shortage_rate={mean_shortage:.3f} "
                f"lambda={lagrange_multiplier:.3f}"
            )

    return model, cfg, lagrange_multiplier


@torch.no_grad()
def evaluate(model: ActorCritic, cfg: Config, episodes: int = 40) -> None:
    costs = []
    shortage_rates = []

    for seed_offset in range(episodes):
        env = ProductionInventoryEnv(cfg)
        env.rng = np.random.default_rng(SEED + 1000 + seed_offset)
        state = env.reset()
        done = False
        episode_cost = 0.0
        episode_shortages = []

        while not done:
            logits, _ = model(torch.from_numpy(state).unsqueeze(0))
            action = int(torch.argmax(logits, dim=-1).item())
            state, cost, shortage_rate, done = env.step(action)
            episode_cost += cost
            episode_shortages.append(shortage_rate)

        costs.append(episode_cost)
        shortage_rates.append(np.mean(episode_shortages))

    print("\nEvaluation")
    print("-" * 48)
    print(f"Mean episode operating cost: {np.mean(costs):.3f}")
    print(f"Mean shortage rate:          {np.mean(shortage_rates):.3f}")
    print(f"Target shortage rate:        {cfg.shortage_target:.3f}")


def main() -> None:
    model, cfg, final_lambda = train()
    print(f"\nFinal Lagrange multiplier: {final_lambda:.3f}")
    evaluate(model, cfg)


if __name__ == "__main__":
    main()
