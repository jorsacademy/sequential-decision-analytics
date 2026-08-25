import numpy as np
import matplotlib.pyplot as plt


class MultiArmedBandit:
    """Epsilon-greedy multi-armed bandit for sequential decision analytics."""

    def __init__(self, n_arms, epsilon=0.1, seed=None):
        if n_arms <= 0:
            raise ValueError("n_arms must be a positive integer.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")

        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms, dtype=int)
        self.value_estimates = np.zeros(n_arms, dtype=float)
        self.rng = np.random.default_rng(seed)

    def select_arm(self):
        """Select an arm using an epsilon-greedy policy."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))

        # Random tie-breaking avoids a systematic preference for lower-index arms
        # when multiple arms currently have the same estimated value.
        best_value = np.max(self.value_estimates)
        best_arms = np.flatnonzero(np.isclose(self.value_estimates, best_value))
        return int(self.rng.choice(best_arms))

    def update(self, arm, reward):
        """Update the sample-mean reward estimate for the selected arm."""
        if arm < 0 or arm >= self.n_arms:
            raise IndexError("arm index is out of range.")

        self.counts[arm] += 1
        n = self.counts[arm]

        # Incremental sample-mean update:
        # Q_n = Q_(n-1) + (R_n - Q_(n-1)) / n
        self.value_estimates[arm] += (
            reward - self.value_estimates[arm]
        ) / n


def simulate_bandit(
    true_probabilities,
    n_rounds=1000,
    epsilon=0.1,
    seed=42,
):
    """Simulate an epsilon-greedy multi-armed bandit process."""
    true_probabilities = np.asarray(true_probabilities, dtype=float)

    if true_probabilities.ndim != 1 or len(true_probabilities) == 0:
        raise ValueError("true_probabilities must be a non-empty 1D sequence.")
    if np.any((true_probabilities < 0.0) | (true_probabilities > 1.0)):
        raise ValueError("All reward probabilities must be between 0 and 1.")
    if n_rounds <= 0:
        raise ValueError("n_rounds must be a positive integer.")

    # Separate random generators keep the policy and environment reproducible
    # while preserving a clear conceptual distinction between decision-making
    # and stochastic feedback from the environment.
    policy_rng_seed = seed
    environment_rng = np.random.default_rng(None if seed is None else seed + 1)

    bandit = MultiArmedBandit(
        n_arms=len(true_probabilities),
        epsilon=epsilon,
        seed=policy_rng_seed,
    )

    cumulative_rewards = np.zeros(n_rounds, dtype=int)
    selected_arms = np.zeros(n_rounds, dtype=int)
    rewards = np.zeros(n_rounds, dtype=int)

    total_reward = 0

    for t in range(n_rounds):
        # 1. Decision: choose an action using current estimates.
        chosen_arm = bandit.select_arm()

        # 2. Observation: receive stochastic feedback from the environment.
        reward = environment_rng.binomial(1, true_probabilities[chosen_arm])

        # 3. Learning: update information for subsequent decisions.
        bandit.update(chosen_arm, reward)

        total_reward += reward
        selected_arms[t] = chosen_arm
        rewards[t] = reward
        cumulative_rewards[t] = total_reward

    return bandit, cumulative_rewards, selected_arms, rewards


def plot_results(
    bandit,
    cumulative_rewards,
    selected_arms,
    true_probabilities,
):
    """Visualize cumulative performance, decisions, and learned estimates."""
    true_probabilities = np.asarray(true_probabilities, dtype=float)
    arms = np.arange(len(true_probabilities))

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(cumulative_rewards) + 1), cumulative_rewards)
    plt.title("Cumulative Reward Over Time")
    plt.xlabel("Decision Round")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    selection_counts = np.bincount(
        selected_arms,
        minlength=len(true_probabilities),
    )

    plt.figure(figsize=(10, 5))
    plt.bar(arms, selection_counts)
    plt.title("Arm Selection Frequency")
    plt.xlabel("Arm")
    plt.ylabel("Number of Selections")
    plt.xticks(arms)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        arms - width / 2,
        true_probabilities,
        width,
        label="True Probability",
    )
    plt.bar(
        arms + width / 2,
        bandit.value_estimates,
        width,
        label="Estimated Probability",
    )
    plt.title("True vs Estimated Reward Probabilities")
    plt.xlabel("Arm")
    plt.ylabel("Reward Probability")
    plt.xticks(arms)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def main():
    true_probabilities = [0.3, 0.5, 0.7, 0.4]
    n_rounds = 1000
    epsilon = 0.1
    seed = 42

    bandit, cumulative_rewards, selected_arms, rewards = simulate_bandit(
        true_probabilities=true_probabilities,
        n_rounds=n_rounds,
        epsilon=epsilon,
        seed=seed,
    )

    plot_results(
        bandit=bandit,
        cumulative_rewards=cumulative_rewards,
        selected_arms=selected_arms,
        true_probabilities=true_probabilities,
    )

    print(f"Total reward: {cumulative_rewards[-1]}")
    print(
        "Estimated reward probabilities:",
        np.round(bandit.value_estimates, 3),
    )
    print("Arm selection counts:", bandit.counts)
    print(f"Observed average reward: {rewards.mean():.3f}")


if __name__ == "__main__":
    main()
