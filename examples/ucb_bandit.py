import numpy as np
import matplotlib.pyplot as plt


class UCBBandit:
    """Upper Confidence Bound (UCB1-style) bandit for sequential decisions."""

    def __init__(self, n_arms, exploration_coefficient=2.0, seed=None):
        if n_arms <= 0:
            raise ValueError("n_arms must be a positive integer.")
        if exploration_coefficient <= 0:
            raise ValueError("exploration_coefficient must be positive.")

        self.n_arms = n_arms
        self.exploration_coefficient = exploration_coefficient
        self.counts = np.zeros(n_arms, dtype=int)
        self.value_estimates = np.zeros(n_arms, dtype=float)
        self.rng = np.random.default_rng(seed)

    def select_arm(self, round_index):
        """Select an arm using an upper-confidence index."""
        untried = np.flatnonzero(self.counts == 0)
        if len(untried) > 0:
            return int(self.rng.choice(untried))

        confidence_bonus = np.sqrt(
            self.exploration_coefficient
            * np.log(round_index + 1)
            / self.counts
        )
        scores = self.value_estimates + confidence_bonus

        best_score = np.max(scores)
        best_arms = np.flatnonzero(np.isclose(scores, best_score))
        return int(self.rng.choice(best_arms))

    def update(self, arm, reward):
        """Update the sample-mean reward estimate of the selected arm."""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.value_estimates[arm] += (
            reward - self.value_estimates[arm]
        ) / n


def simulate_ucb(
    true_probabilities,
    n_rounds=1000,
    exploration_coefficient=2.0,
    seed=42,
):
    """Simulate UCB on a stationary Bernoulli multi-armed bandit."""
    true_probabilities = np.asarray(true_probabilities, dtype=float)

    if true_probabilities.ndim != 1 or len(true_probabilities) == 0:
        raise ValueError("true_probabilities must be a non-empty 1D sequence.")
    if np.any((true_probabilities < 0.0) | (true_probabilities > 1.0)):
        raise ValueError("All reward probabilities must be between 0 and 1.")
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive.")

    environment_rng = np.random.default_rng(None if seed is None else seed + 1)
    bandit = UCBBandit(
        n_arms=len(true_probabilities),
        exploration_coefficient=exploration_coefficient,
        seed=seed,
    )

    cumulative_rewards = np.zeros(n_rounds, dtype=int)
    selected_arms = np.zeros(n_rounds, dtype=int)
    total_reward = 0

    for t in range(n_rounds):
        # Decision: use current estimates plus an uncertainty bonus.
        arm = bandit.select_arm(t)

        # Observation: receive stochastic reward from the environment.
        reward = environment_rng.binomial(1, true_probabilities[arm])

        # Learning: update the estimate used by future decisions.
        bandit.update(arm, reward)

        total_reward += reward
        selected_arms[t] = arm
        cumulative_rewards[t] = total_reward

    return bandit, cumulative_rewards, selected_arms


def plot_results(bandit, cumulative_rewards, selected_arms):
    """Plot cumulative reward and arm-selection frequencies."""
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(cumulative_rewards) + 1), cumulative_rewards)
    plt.title("UCB: Cumulative Reward Over Time")
    plt.xlabel("Decision Round")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    arms = np.arange(bandit.n_arms)
    selection_counts = np.bincount(selected_arms, minlength=bandit.n_arms)

    plt.figure(figsize=(10, 5))
    plt.bar(arms, selection_counts)
    plt.title("UCB: Arm Selection Frequency")
    plt.xlabel("Arm")
    plt.ylabel("Number of Selections")
    plt.xticks(arms)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def main():
    true_probabilities = [0.3, 0.5, 0.7, 0.4]

    bandit, cumulative_rewards, selected_arms = simulate_ucb(
        true_probabilities=true_probabilities,
        n_rounds=1000,
        exploration_coefficient=2.0,
        seed=42,
    )

    plot_results(bandit, cumulative_rewards, selected_arms)

    print(f"Total reward: {cumulative_rewards[-1]}")
    print("Arm selection counts:", bandit.counts)
    print(
        "Estimated reward probabilities:",
        np.round(bandit.value_estimates, 3),
    )


if __name__ == "__main__":
    main()
