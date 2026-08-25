import numpy as np
import matplotlib.pyplot as plt

from epsilon_greedy_bandit import simulate_bandit
from ucb_bandit import simulate_ucb
from thompson_sampling_bandit import simulate_thompson_sampling


def run_comparison(
    true_probabilities,
    n_rounds=1000,
    n_replications=200,
    epsilon=0.1,
):
    """Compare average cumulative reward and regret across bandit policies."""
    true_probabilities = np.asarray(true_probabilities, dtype=float)
    optimal_mean = np.max(true_probabilities)

    policy_names = ["Epsilon-Greedy", "UCB", "Thompson Sampling"]
    reward_paths = {
        name: np.zeros((n_replications, n_rounds), dtype=float)
        for name in policy_names
    }

    for replication in range(n_replications):
        seed = 1000 + replication

        _, rewards_eps, _, _ = simulate_bandit(
            true_probabilities,
            n_rounds=n_rounds,
            epsilon=epsilon,
            seed=seed,
        )
        _, rewards_ucb, _ = simulate_ucb(
            true_probabilities,
            n_rounds=n_rounds,
            exploration_coefficient=2.0,
            seed=seed,
        )
        _, rewards_ts, _ = simulate_thompson_sampling(
            true_probabilities,
            n_rounds=n_rounds,
            seed=seed,
        )

        reward_paths["Epsilon-Greedy"][replication] = rewards_eps
        reward_paths["UCB"][replication] = rewards_ucb
        reward_paths["Thompson Sampling"][replication] = rewards_ts

    rounds = np.arange(1, n_rounds + 1)
    optimal_expected_reward = rounds * optimal_mean

    average_rewards = {
        name: paths.mean(axis=0)
        for name, paths in reward_paths.items()
    }
    average_regret = {
        name: optimal_expected_reward - avg_reward
        for name, avg_reward in average_rewards.items()
    }

    return average_rewards, average_regret


def plot_comparison(average_rewards, average_regret):
    """Plot average cumulative reward and pseudo-regret."""
    plt.figure(figsize=(10, 5))
    for name, values in average_rewards.items():
        plt.plot(np.arange(1, len(values) + 1), values, label=name)
    plt.title("Average Cumulative Reward Across Bandit Policies")
    plt.xlabel("Decision Round")
    plt.ylabel("Average Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    for name, values in average_regret.items():
        plt.plot(np.arange(1, len(values) + 1), values, label=name)
    plt.title("Average Pseudo-Regret Across Bandit Policies")
    plt.xlabel("Decision Round")
    plt.ylabel("Pseudo-Regret")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    true_probabilities = [0.3, 0.5, 0.7, 0.4]

    average_rewards, average_regret = run_comparison(
        true_probabilities=true_probabilities,
        n_rounds=1000,
        n_replications=200,
        epsilon=0.1,
    )

    plot_comparison(average_rewards, average_regret)

    print("Average reward after 1000 rounds:")
    for name, values in average_rewards.items():
        print(f"  {name}: {values[-1]:.2f}")

    print("Average pseudo-regret after 1000 rounds:")
    for name, values in average_regret.items():
        print(f"  {name}: {values[-1]:.2f}")


if __name__ == "__main__":
    main()
