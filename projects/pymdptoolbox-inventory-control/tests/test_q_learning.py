import sys
import json
import numpy as np
import pytest
from inventory_mdp.model import InventoryConfig, build_inventory_mdp, solve_inventory_mdp
from inventory_mdp.q_learning import inventory_step, train_q_learning, discounted_cost, evaluate_paired, main


def test_tail_reward_matches_unbounded_poisson_expectation():
    # At zero capacity delivery (action zero), every demand unit is lost.
    c = InventoryConfig(max_inventory=1, max_order=0, demand_rate=10)
    _, r = build_inventory_mdp(c)
    assert r[0, 0] == pytest.approx(-10 * c.lost_sales_cost)


def test_simulator_matches_transition_and_reward_by_enumeration():
    from math import exp, factorial
    c = InventoryConfig(max_inventory=3, max_order=4, demand_rate=2)
    p, r = build_inventory_mdp(c)
    for stock in range(4):
        for action in range(5):
            transition, reward = np.zeros(4), 0.0
            for d in range(40):
                probability = exp(-2) * 2**d / factorial(d)
                nxt, rew, _ = inventory_step(c, stock, action, d)
                transition[nxt] += probability
                reward += probability * rew
            np.testing.assert_allclose(transition, p[action, stock], atol=1e-12)
            assert reward == pytest.approx(r[stock, action])


def test_q_learning_recovers_small_optimal_policy():
    c = InventoryConfig(max_inventory=2, max_order=2, discount=0.7)
    policy, q = train_q_learning(c, updates=30_000, seed=11)
    optimal, _ = solve_inventory_mdp(c, 'policy_iteration')
    np.testing.assert_allclose(discounted_cost(c, policy), discounted_cost(c, optimal), atol=1e-7)
    policy2, q2 = train_q_learning(c, updates=30_000, seed=11)
    np.testing.assert_array_equal(q, q2)
    np.testing.assert_array_equal(policy, policy2)


def test_evaluation_is_paired_and_reproducible():
    c = InventoryConfig()
    p, _ = solve_inventory_mdp(c)
    result = evaluate_paired(c, {'a': p, 'b': p}, episodes=3, periods=20)
    assert result['a'] == result['b']
    assert result == evaluate_paired(c, {'a': p, 'b': p}, episodes=3, periods=20)
    assert 0 <= result['a']['fill_rate'] <= 1


@pytest.mark.parametrize('policy', [[0], [0.5]*9, [-1]*9, [9]*9])
def test_invalid_policy(policy):
    with pytest.raises(ValueError):
        discounted_cost(InventoryConfig(), policy)


def test_invalid_simulation_arguments():
    c = InventoryConfig()
    with pytest.raises(ValueError):
        inventory_step(c, 9, 0, 0)
    with pytest.raises(ValueError):
        train_q_learning(c, updates=0)
    with pytest.raises(ValueError):
        evaluate_paired(c, {}, episodes=1)


def test_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['compare', '--updates', '10', '--seeds', '1'])
    main()
    assert 'q_learning_seed_1' in json.loads(capsys.readouterr().out)['metrics']
