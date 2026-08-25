import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    """Small deterministic grid-world environment for Q-learning."""

    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }
    ACTION_NAMES = {0: "U", 1: "D", 2: "L", 3: "R"}

    def __init__(self, rows=4, cols=4, terminal_state=(3, 3), step_reward=-1.0):
        self.rows = rows
        self.cols = cols
        self.terminal_state = terminal_state
        self.step_reward = step_reward

    def step(self, state, action):
        """Apply an action and return next state, reward, and terminal flag."""
        if state == self.terminal_state:
            return state, 0.0, True

        dr, dc = self.ACTIONS[action]
        nr = min(max(state[0] + dr, 0), self.rows - 1)
        nc = min(max(state[1] + dc, 0), self.cols - 1)
        next_state = (nr, nc)
        done = next_state == self.terminal_state

        return next_state, self.step_reward, done

    def state_index(self, state):
        return state[0] * self.cols + state[1]


def epsilon_greedy(q_values, epsilon, rng):
    """Select an action using an epsilon-greedy policy with random tie-breaking."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))

    best_value = np.max(q_values)
    best_actions = np.flatnonzero(np.isclose(q_values, best_value))
    return int(rng.choice(best_actions))


def train_q_learning(
    env,
    n_episodes=1000,
    alpha=0.2,
    gamma=0.95,
    epsilon=0.1,
    seed=42,
    max_steps_per_episode=200,
):
    """Learn action values with tabular Q-learning."""
    rng = np.random.default_rng(seed)
    n_states = env.rows * env.cols
    n_actions = len(env.ACTIONS)
    q_table = np.zeros((n_states, n_actions), dtype=float)
    episode_returns = np.zeros(n_episodes, dtype=float)
    episode_lengths = np.zeros(n_episodes, dtype=int)

    for episode in range(n_episodes):
        state = (0, 0)
        total_reward = 0.0

        for step in range(max_steps_per_episode):
            state_idx = env.state_index(state)
            action = epsilon_greedy(q_table[state_idx], epsilon, rng)

            next_state, reward, done = env.step(state, action)
            next_idx = env.state_index(next_state)

            td_target = reward
            if not done:
                td_target += gamma * np.max(q_table[next_idx])

            td_error = td_target - q_table[state_idx, action]
            q_table[state_idx, action] += alpha * td_error

            total_reward += reward
            state = next_state

            if done:
                episode_lengths[episode] = step + 1
                break
        else:
            episode_lengths[episode] = max_steps_per_episode

        episode_returns[episode] = total_reward

    return q_table, episode_returns, episode_lengths


def extract_policy(env, q_table):
    """Extract a greedy policy from the learned Q-table."""
    policy = {}

    for r in range(env.rows):
        for c in range(env.cols):
            state = (r, c)
            if state == env.terminal_state:
                policy[state] = "T"
                continue

            state_idx = env.state_index(state)
            best_action = int(np.argmax(q_table[state_idx]))
            policy[state] = env.ACTION_NAMES[best_action]

    return policy


def moving_average(values, window=50):
    """Compute a moving average for visualization."""
    if window <= 1:
        return values.copy()
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_learning(episode_returns, episode_lengths, window=50):
    """Visualize Q-learning performance across episodes."""
    avg_returns = moving_average(episode_returns, window)
    x_returns = np.arange(window, len(episode_returns) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x_returns, avg_returns)
    plt.title(f"Q-Learning: Moving Average Return (Window = {window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    avg_lengths = moving_average(episode_lengths.astype(float), window)
    x_lengths = np.arange(window, len(episode_lengths) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x_lengths, avg_lengths)
    plt.title(f"Q-Learning: Moving Average Episode Length (Window = {window})")
    plt.xlabel("Episode")
    plt.ylabel("Average Number of Steps")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_policy(env, policy):
    """Print the learned greedy policy as a grid."""
    print("Learned greedy policy:")
    for r in range(env.rows):
        print("  ".join(policy[(r, c)] for c in range(env.cols)))


def main():
    env = GridWorld()

    q_table, episode_returns, episode_lengths = train_q_learning(
        env,
        n_episodes=1000,
        alpha=0.2,
        gamma=0.95,
        epsilon=0.1,
        seed=42,
    )

    policy = extract_policy(env, q_table)

    print_policy(env, policy)
    print(f"Mean episode length over final 100 episodes: {episode_lengths[-100:].mean():.2f}")
    print(f"Mean return over final 100 episodes: {episode_returns[-100:].mean():.2f}")

    plot_learning(episode_returns, episode_lengths)


if __name__ == "__main__":
    main()
