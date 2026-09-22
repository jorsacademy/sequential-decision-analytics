"""Finite-state lost-sales inventory MDP."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial

import mdptoolbox.mdp
import numpy as np


@dataclass(frozen=True)
class InventoryConfig:
    max_inventory: int = 8
    max_order: int = 8
    demand_rate: float = 2.0
    order_cost: float = 1.0
    holding_cost: float = 0.4
    lost_sales_cost: float = 4.0
    discount: float = 0.95

    def validate(self) -> None:
        if self.max_inventory < 1:
            raise ValueError("max_inventory must be positive")
        if self.max_order < 0:
            raise ValueError("max_order must be non-negative")
        if self.demand_rate <= 0:
            raise ValueError("demand_rate must be positive")
        if min(self.order_cost, self.holding_cost, self.lost_sales_cost) < 0:
            raise ValueError("costs must be non-negative")
        if not 0 < self.discount < 1:
            raise ValueError("discount must be between 0 and 1")


def _poisson_prob(k: int, rate: float) -> float:
    return exp(-rate) * rate**k / factorial(k)


def _demand_distribution(rate: float, cutoff: int) -> np.ndarray:
    probs = np.array([_poisson_prob(k, rate) for k in range(cutoff)], dtype=float)
    tail = max(0.0, 1.0 - probs.sum())
    return np.append(probs, tail)


def build_inventory_mdp(config: InventoryConfig) -> tuple[np.ndarray, np.ndarray]:
    """Build transition tensor P[A,S,S] and reward matrix R[S,A]."""
    config.validate()
    states = config.max_inventory + 1
    actions = config.max_order + 1
    cutoff = config.max_inventory + config.max_order + 1
    demand_probs = _demand_distribution(config.demand_rate, cutoff)

    transitions = np.zeros((actions, states, states), dtype=float)
    rewards = np.zeros((states, actions), dtype=float)

    for action in range(actions):
        for stock in range(states):
            post_order = min(config.max_inventory, stock + action)
            effective_order = post_order - stock
            # The tail bucket gives exact next-stock probabilities, but its
            # representative demand understates lost sales. Recover E[D]
            # exactly so simulation and DP use the same unbounded Poisson law.
            tail_excess = max(0.0, config.demand_rate - np.dot(
                np.arange(len(demand_probs)), demand_probs))
            expected_cost = (config.order_cost * effective_order
                             + config.lost_sales_cost * tail_excess)

            for demand, prob in enumerate(demand_probs):
                next_stock = max(post_order - demand, 0)
                lost_sales = max(demand - post_order, 0)
                transitions[action, stock, next_stock] += prob
                expected_cost += prob * (
                    config.holding_cost * next_stock
                    + config.lost_sales_cost * lost_sales
                )

            rewards[stock, action] = -expected_cost

    return transitions, rewards


def solve_inventory_mdp(
    config: InventoryConfig,
    algorithm: str = "value_iteration",
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Solve the inventory MDP and return policy and value function."""
    transitions, rewards = build_inventory_mdp(config)

    if algorithm == "value_iteration":
        solver = mdptoolbox.mdp.ValueIteration(transitions, rewards, config.discount)
    elif algorithm == "policy_iteration":
        solver = mdptoolbox.mdp.PolicyIteration(transitions, rewards, config.discount)
    else:
        raise ValueError("algorithm must be 'value_iteration' or 'policy_iteration'")

    solver.run()
    return tuple(int(a) for a in solver.policy), tuple(float(v) for v in solver.V)
