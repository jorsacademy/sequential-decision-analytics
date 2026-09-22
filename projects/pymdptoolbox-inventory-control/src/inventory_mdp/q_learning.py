"""Tabular Q-learning and paired evaluation on the repository's inventory MDP."""
from __future__ import annotations

import argparse
import json

import numpy as np

from .model import InventoryConfig, build_inventory_mdp, solve_inventory_mdp


def inventory_step(config: InventoryConfig, stock: int, action: int, demand: int):
    """Immediate replenishment, capacity clipping, then unbounded Poisson demand.

    Inputs are discrete nonnegative quantities. Charge only the delivered order;
    unmet demand is lost (no backlog). Returns next stock, reward, and units sold.
    """
    if not 0 <= stock <= config.max_inventory or not 0 <= action <= config.max_order or demand < 0:
        raise ValueError("stock, action or demand outside the model domain")
    available = min(config.max_inventory, stock + action)
    sold = min(available, demand)
    leftover = available - sold
    cost = (config.order_cost * (available - stock)
            + config.holding_cost * leftover
            + config.lost_sales_cost * (demand - sold))
    return leftover, -cost, sold


def train_q_learning(config: InventoryConfig, updates: int = 200_000, seed: int = 42):
    """Off-policy Q-learning from a generative simulator, without transition tables.

    Reset to a uniformly sampled state/action for each one-step experience so all
    pairs are explored. This is not on-policy trajectory training. Per-pair step
    sizes decay as visit_count**(-0.6). No terminal bootstrap is removed because
    the underlying objective is discounted infinite-horizon cost.
    """
    config.validate()
    if updates < 1:
        raise ValueError("updates must be positive")
    rng = np.random.default_rng(seed)
    q = np.zeros((config.max_inventory + 1, config.max_order + 1))
    visits = np.zeros_like(q, dtype=int)
    for _ in range(updates):
        stock = int(rng.integers(q.shape[0]))
        action = int(rng.integers(q.shape[1]))
        next_stock, reward, _ = inventory_step(config, stock, action, int(rng.poisson(config.demand_rate)))
        visits[stock, action] += 1
        alpha = visits[stock, action] ** -0.6
        q[stock, action] += alpha * (reward + config.discount * q[next_stock].max() - q[stock, action])
    return q.argmax(axis=1), q


def validate_policy(config, policy):
    policy = np.asarray(policy)
    if (policy.shape != (config.max_inventory + 1,)
            or not np.issubdtype(policy.dtype, np.integer)
            or np.any(policy < 0) or np.any(policy > config.max_order)):
        raise ValueError("policy must give one legal integer action per stock state")
    return policy


def discounted_cost(config: InventoryConfig, policy):
    """Exact policy evaluation, C = (I - gamma P_pi)^-1 (-R_pi)."""
    policy = validate_policy(config, policy)
    transitions, rewards = build_inventory_mdp(config)
    states = np.arange(len(policy))
    return np.linalg.solve(np.eye(len(policy)) - config.discount * transitions[policy, states],
                           -rewards[states, policy])


def evaluate_paired(config, policies, episodes=100, periods=200, seed=10_000):
    """Identical unseen demand paths and zero initial stock for every policy.

    Mean period cost is an operational KPI, not the discounted training objective.
    Demand-weighted fill rate is total fulfilled units / total requested units.
    """
    config.validate()
    if episodes < 2 or periods < 1:
        raise ValueError("need at least two episodes and one period")
    demand_paths = np.random.default_rng(seed).poisson(config.demand_rate, (episodes, periods))
    results = {}
    for name, policy in policies.items():
        policy = validate_policy(config, policy)
        costs, sold_total = [], 0
        for demands in demand_paths:
            stock, cost = 0, 0.0
            for demand in demands:
                stock, reward, sold = inventory_step(config, stock, int(policy[stock]), int(demand))
                cost -= reward
                sold_total += sold
            costs.append(cost / periods)
        results[name] = {"mean_period_cost": float(np.mean(costs)),
                         "period_cost_standard_error": float(np.std(costs, ddof=1) / np.sqrt(episodes)),
                         "fill_rate": float(sold_total / demand_paths.sum()) if demand_paths.sum() else 1.0,
                         "discounted_cost_from_empty": float(discounted_cost(config, policy)[0])}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    config = InventoryConfig()
    vi, _ = solve_inventory_mdp(config)
    pi, _ = solve_inventory_mdp(config, "policy_iteration")
    # Tune the base-stock level using the known model, never the evaluation paths.
    candidates = [np.minimum(config.max_order, np.maximum(0, level - np.arange(config.max_inventory + 1)))
                  for level in range(config.max_inventory + 1)]
    base_stock = min(candidates, key=lambda policy: discounted_cost(config, policy)[0])
    policies = {"value_iteration": vi, "policy_iteration": pi, "tuned_base_stock": base_stock}
    for seed in args.seeds:
        policies[f"q_learning_seed_{seed}"], _ = train_q_learning(config, args.updates, seed)
    print(json.dumps({"updates_per_training_seed": args.updates,
                      "evaluation_seed": 10_000, "episodes": 100, "periods": 200,
                      "policies": {k: np.asarray(v).tolist() for k, v in policies.items()},
                      "metrics": evaluate_paired(config, policies)}, indent=2))


if __name__ == "__main__":
    main()
