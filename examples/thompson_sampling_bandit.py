import numpy as np
import matplotlib.pyplot as plt


class ThompsonSamplingBandit:
    """Thompson sampling for Bernoulli multi-armed bandits."""

    def __init__(self, n_arms, seed=None):
        if n_arms <= 0:
            raise ValueError("n_arms must be a positive integer.")

        self.n_arms = n_arms
        self.alpha = np.ones(n_arms, dtype=float)
        self.beta = np.ones(n_arms, dtype=float)
        self.counts = np.zeros(n_arms, dtype=int)
        self.rng = np.random.default_rng(seed)

    def select_arm(self):
        """Sample one reward probability per arm and select the largest draw."""
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        """Update the Beta posterior after observing a Bernoulli reward."""
        if reward not in (0, 1):
            raise ValueError("Thompson sampling example expects binary rewards.")

        self.counts[arm] += 1
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward

    @property
    def posterior_means(self):
        return self.alpha / (self.alpha + self.beta)


def simulate_thompson_sampling(
    true_probabilities,
    n_rounds=1000,
    seed=42,
):
    """Simulate Thompson sampling on a Bernoulli multi-armed bandit."""
    true_probabilities = np.asarray(true_probabilities, dtype=float)

    if true_probabilities.ndim != 1 or len(true_probabilities) == 0:
        raise ValueError("true_probabilities must be a non-empty 1D sequence.")
    if np.any((true_probabilities < 0.0) | (true_probabilities > 1.0)):
        raise ValueError("All reward probabilities must be between 0 and 1.")
    if n_rounds <= 0:
        raise ValueError("n_rounds must be positive.")

    environment_rng = np.random.default_rng(None if seed is None else seed + 1)
    bandit = ThompsonSamplingBandit(len(true_probabilities), seed=seed)

    cumulative_rewards = np.zeros(n_rounds, dtype=int)
    selected_arms = np.zeros(n_rounds, dtype=int)
    total_reward = 0

    for t in range(n_rounds):
        # Decision: sample from current posterior beliefs.
        arm = bandit.select_arm()

        # Observation: receive stochastic Bernoulli feedback.
        reward = environment_rng.binomial(1, true_probabilities[arm])

        # Learning: update posterior beliefs for future decisions.
        bandit.update(arm, reward)

        total_reward += reward
        selected_arms[t] = arm
        cumulative_rewards[t] = total_reward

    return bandit, cumulative_rewards, selected_arms


def plot_results(bandit, cumulative_rewards, selected_arms, true_probabilities):
    """Plot cumulative reward, decisions, and posterior mean estimates."""
    true_probabilities = np.asarray(true_probabilities, dtype=float)
    arms = np.arange(bandit.n_arms)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(cumulative_rewards) + 1), cumulative_rewards)
    plt.title("Thompson Sampling: Cumulative Reward Over Time")
    plt.xlabel("Decision Round")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    selection_counts = np.bincount(selected_arms, minlength=bandit.n_arms)
    plt.figure(figsize=(10, 5))
    plt.bar(arms, selection_counts)
    plt.title("Thompson Sampling: Arm Selection Frequency")
    plt.xlabel("Arm")
    plt.ylabel("Number of Selections")
    plt.xticks(arms)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(arms - width / 2, true_probabilities, width, label="True Probability")
    plt.bar(arms + width / 2, bandit.posterior_means, width, label="Posterior Mean")
    plt.title("True vs Posterior Mean Reward Probabilities")
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

    bandit, cumulative_rewards, selected_arms = simulate_thompson_sampling(
        true_probabilities=true_probabilities,
        n_rounds=1000,
        seed=42,
    )

    plot_results(
        bandit,
        cumulative_rewards,
        selected_arms,
        true_probabilities,
    )

    print(f"Total reward: {cumulative_rewards[-1]}")
    print("Arm selection counts:", bandit.counts)
    print("Posterior mean probabilities:", np.round(bandit.posterior_means, 3))


if __name__ == "__main__":
    main()
