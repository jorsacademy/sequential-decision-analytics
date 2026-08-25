import numpy as np
import matplotlib.pyplot as plt


class GridWorldMDP:
    """Small deterministic grid-world Markov Decision Process."""

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
        self.states = [(r, c) for r in range(rows) for c in range(cols)]

    def transition(self, state, action):
        """Return next state and reward for a state-action pair."""
        if state == self.terminal_state:
            return state, 0.0

        dr, dc = self.ACTIONS[action]
        nr = min(max(state[0] + dr, 0), self.rows - 1)
        nc = min(max(state[1] + dc, 0), self.cols - 1)
        next_state = (nr, nc)

        return next_state, self.step_reward


def value_iteration(mdp, gamma=0.95, tolerance=1e-10, max_iterations=10000):
    """Solve the MDP with Bellman optimality updates."""
    values = {state: 0.0 for state in mdp.states}

    for iteration in range(max_iterations):
        delta = 0.0
        new_values = values.copy()

        for state in mdp.states:
            if state == mdp.terminal_state:
                new_values[state] = 0.0
                continue

            action_values = []
            for action in mdp.ACTIONS:
                next_state, reward = mdp.transition(state, action)
                action_values.append(reward + gamma * values[next_state])

            new_values[state] = max(action_values)
            delta = max(delta, abs(new_values[state] - values[state]))

        values = new_values

        if delta < tolerance:
            return values, iteration + 1

    return values, max_iterations


def extract_greedy_policy(mdp, values, gamma=0.95):
    """Extract an optimal greedy policy from a value function."""
    policy = {}

    for state in mdp.states:
        if state == mdp.terminal_state:
            policy[state] = "T"
            continue

        q_values = []
        for action in mdp.ACTIONS:
            next_state, reward = mdp.transition(state, action)
            q_values.append(reward + gamma * values[next_state])

        best_action = int(np.argmax(q_values))
        policy[state] = mdp.ACTION_NAMES[best_action]

    return policy


def plot_value_function(mdp, values):
    """Visualize the optimal state-value function."""
    value_matrix = np.array(
        [[values[(r, c)] for c in range(mdp.cols)] for r in range(mdp.rows)]
    )

    plt.figure(figsize=(7, 6))
    image = plt.imshow(value_matrix)
    plt.colorbar(image, label="Optimal State Value")
    plt.title("MDP Value Iteration: Optimal State Values")
    plt.xlabel("Column")
    plt.ylabel("Row")

    for r in range(mdp.rows):
        for c in range(mdp.cols):
            plt.text(c, r, f"{value_matrix[r, c]:.1f}", ha="center", va="center")

    plt.tight_layout()
    plt.show()


def print_policy(mdp, policy):
    """Print the policy as a grid."""
    print("Optimal policy:")
    for r in range(mdp.rows):
        print("  ".join(policy[(r, c)] for c in range(mdp.cols)))


def main():
    mdp = GridWorldMDP()
    gamma = 0.95

    values, iterations = value_iteration(mdp, gamma=gamma)
    policy = extract_greedy_policy(mdp, values, gamma=gamma)

    print(f"Value iteration converged in {iterations} iterations.")
    print_policy(mdp, policy)
    plot_value_function(mdp, values)


if __name__ == "__main__":
    main()
