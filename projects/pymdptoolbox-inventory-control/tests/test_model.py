import numpy as np
import pytest

from inventory_mdp.model import InventoryConfig, build_inventory_mdp, solve_inventory_mdp


def test_transition_probabilities_are_valid():
    config = InventoryConfig(max_inventory=5, max_order=4, demand_rate=1.5)
    transitions, rewards = build_inventory_mdp(config)

    assert transitions.shape == (5, 6, 6)
    assert rewards.shape == (6, 5)
    assert np.all(transitions >= 0)
    np.testing.assert_allclose(transitions.sum(axis=2), 1.0)


def test_invalid_configs_are_rejected():
    cases = [
        InventoryConfig(max_inventory=0),
        InventoryConfig(max_order=-1),
        InventoryConfig(demand_rate=0),
        InventoryConfig(order_cost=-1),
        InventoryConfig(discount=1.0),
    ]
    for config in cases:
        with pytest.raises(ValueError):
            build_inventory_mdp(config)


def test_ordering_is_capped_by_storage_capacity():
    config = InventoryConfig(max_inventory=3, max_order=5, demand_rate=1.0)
    transitions, rewards = build_inventory_mdp(config)

    np.testing.assert_allclose(transitions[3, 0], transitions[5, 0])
    assert rewards[0, 3] == pytest.approx(rewards[0, 5])


def test_value_and_policy_iteration_agree():
    config = InventoryConfig()
    vi_policy, vi_values = solve_inventory_mdp(config, "value_iteration")
    pi_policy, pi_values = solve_inventory_mdp(config, "policy_iteration")

    assert vi_policy == pi_policy
    assert len(vi_policy) == config.max_inventory + 1
    assert all(0 <= action <= config.max_order for action in vi_policy)
    assert len(vi_values) == len(pi_values) == config.max_inventory + 1
    assert np.all(np.isfinite(vi_values))
    assert np.all(np.isfinite(pi_values))


def test_invalid_algorithm_is_rejected():
    with pytest.raises(ValueError, match="algorithm"):
        solve_inventory_mdp(InventoryConfig(), "unknown")
