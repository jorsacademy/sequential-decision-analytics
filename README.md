# Sequential Decision Analytics

Educational implementations of sequential decision-making algorithms for courses in Optimization, Operations Research, and Reinforcement Learning.

## Purpose

This repository provides compact and transparent examples of Sequential Decision Analytics (SDA). The code is designed for teaching rather than production deployment.

The examples emphasize a common sequential structure:

1. observe the current information or state,
2. select an action,
3. observe uncertain feedback,
4. update knowledge or move to a new state,
5. use the new information in subsequent decisions.

The repository begins with multi-armed bandits and then extends to stochastic control, Markov Decision Processes, dynamic programming, and model-free Reinforcement Learning. This makes it possible to connect learning, optimization, Operations Research, and Reinforcement Learning within one small teaching repository.

## Topics

- Static decision making under uncertainty
- Decision matrices and state-of-nature models
- Expected value and outcome risk
- Monte Carlo validation
- Expected Value of Perfect Information (EVPI)
- Probability sensitivity and decision stability
- Sequential decision making
- State, action, uncertainty, transition, and reward
- Exploration versus exploitation
- Multi-armed bandits
- Epsilon-greedy policies
- Upper Confidence Bound (UCB)
- Thompson Sampling
- Bayesian sequential learning
- Online action-value estimation
- Cumulative reward
- Regret and policy comparison
- Finite-horizon stochastic optimization
- Dynamic programming and backward induction
- Markov Decision Processes
- Bellman optimality equations
- Value iteration
- Tabular Q-learning
- Temporal-difference learning
- Model-based versus model-free decision making
- Policy evaluation by simulation

## Repository Structure

```text
sequential-decision-analytics/
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   └── SDA_CONCEPTS.md
├── tests/
│   └── test_static_decision_analysis.py
└── examples/
    ├── static_decision_under_uncertainty.py
    ├── epsilon_greedy_bandit.py
    ├── ucb_bandit.py
    ├── thompson_sampling_bandit.py
    ├── compare_bandit_policies.py
    ├── finite_horizon_inventory_control.py
    ├── mdp_value_iteration.py
    └── q_learning_gridworld.py
```

## Example 0: Static Decision Analysis Under Uncertainty

Before introducing sequential learning, this example establishes the one-shot decision-analysis baseline: a decision-maker chooses among alternatives while the state of nature is uncertain but its probability distribution is known.

The example computes:

- expected value for each alternative;
- probability-weighted outcome standard deviation;
- Monte Carlo outcome distributions;
- Expected Value of Perfect Information (EVPI);
- probability sensitivity while preserving a valid probability vector;
- decision stability across sensitivity scenarios.

The generic model supports both benefit maximization and cost minimization. A four-alternative investment example is included only as an illustrative payoff matrix; the core class is not finance-specific.

A key methodological distinction is made in the sensitivity output: the share of perturbed scenarios in which an alternative remains preferred is a **decision-stability diagnostic**, not a statistical confidence level.

Run:

```bash
python examples/static_decision_under_uncertainty.py
```

This example provides the static baseline for the rest of the repository. The later examples add repeated decisions, learning, state transitions, and dynamic value.

## Example 1: Epsilon-Greedy Multi-Armed Bandit

The epsilon-greedy example contains four Bernoulli reward processes with true reward probabilities:

```python
[0.3, 0.5, 0.7, 0.4]
```

The decision-maker does not use these probabilities directly. It learns estimated action values from observed rewards.

With probability `epsilon`, the policy explores by selecting a random arm. Otherwise, it exploits current information by selecting an arm with the highest estimated reward.

The selected arm is updated using the incremental sample mean:

```text
Q_n = Q_(n-1) + (R_n - Q_(n-1)) / n
```

Run:

```bash
python examples/epsilon_greedy_bandit.py
```

## Example 2: Upper Confidence Bound

The UCB example replaces random exploration with an explicit uncertainty bonus. An arm can be attractive because its estimated reward is high or because it has not yet been sampled enough.

```text
UCB(a) = Q(a) + sqrt(c * log(t) / N(a))
```

Run:

```bash
python examples/ucb_bandit.py
```

## Example 3: Thompson Sampling

The Thompson Sampling example provides a Bayesian approach to sequential learning. Each Bernoulli arm has a Beta posterior distribution. The policy samples one plausible reward probability from each posterior and selects the arm with the largest sampled value.

Run:

```bash
python examples/thompson_sampling_bandit.py
```

## Example 4: Comparing Bandit Policies

The comparison example evaluates epsilon-greedy, UCB, and Thompson Sampling over many independent replications. It plots average cumulative reward and average pseudo-regret.

Run:

```bash
python examples/compare_bandit_policies.py
```

## Example 5: Finite-Horizon Inventory Control

The inventory example introduces explicit state, action, uncertainty, transition, and downstream value.

The Bellman recursion is:

```text
V_t(s) = max_a E[C_t(s, a, W) + V_(t+1)(S')]
```

Run:

```bash
python examples/finite_horizon_inventory_control.py
```

## Example 6: Markov Decision Process and Value Iteration

The grid-world MDP introduces an infinite-horizon discounted control problem with a known transition model.

The decision-maker knows how each action changes the state and uses the Bellman optimality equation to compute the optimal value function:

```text
V*(s) = max_a [R(s,a) + gamma * V*(s')]
```

The code uses value iteration until the value function converges and then extracts a greedy optimal policy.

This example provides the bridge from finite-horizon dynamic programming to stationary Markov Decision Processes.

Run:

```bash
python examples/mdp_value_iteration.py
```

## Example 7: Q-Learning in the Same Grid World

The Q-learning example uses the same basic grid-world structure but removes the assumption that the transition model must be used by the learning algorithm.

The action-value update is:

```text
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

The agent learns by interacting with the environment, using epsilon-greedy exploration and temporal-difference updates.

This creates a direct teaching comparison:

- value iteration: model-based dynamic programming,
- Q-learning: model-free Reinforcement Learning.

Run:

```bash
python examples/q_learning_gridworld.py
```


## Deep Learning Project Suite for Industrial Engineering

A separate project suite extends the repository from classical sequential decision analytics into neural sequence models, graph neural networks, and deep reinforcement learning. The examples avoid image-processing and quality-control use cases and focus on forecasting, networks, and dynamic resource decisions.

The suite is under [`projects/deep_learning_ie/`](projects/deep_learning_ie/README.md):

- **GRU vs Transformer multi-horizon demand forecasting** — compares recurrent and attention-based sequence models using both predictive error and an asymmetric inventory-cost proxy.
- **Temporal GNN for supply-network delay propagation** — combines graph message passing with temporal recurrence to predict how congestion and disruptions propagate through a directed supply network.
- **Constrained PPO for production-inventory control** — demonstrates actor-critic deep RL, PPO clipping, generalized advantage estimation, and Lagrangian shortage control.

Install the additional dependency set with:

```bash
pip install -r projects/deep_learning_ie/requirements.txt
```

These projects are deliberately decision-oriented: model quality is connected to replenishment cost, network delay, or service constraints rather than treated as prediction accuracy alone.

## Suggested Teaching Sequence

A useful classroom sequence is:

0. static decision analysis: alternatives, states of nature, expected value, risk, EVPI, and sensitivity,
1. epsilon-greedy: basic sequential learning and exploration versus exploitation,
2. UCB: optimism under uncertainty,
3. Thompson Sampling: Bayesian exploration,
4. bandit comparison: cumulative reward, regret, and repeated experiments,
5. inventory control: state transitions, Bellman recursion, and finite-horizon dynamic programming,
6. MDP value iteration: stationary state-action models and Bellman optimality,
7. Q-learning: model-free temporal-difference learning.

This sequence provides a compact path from Sequential Decision Analytics to Operations Research and Reinforcement Learning.

## Model-Based and Model-Free Perspective

The examples can also be organized by what the decision-maker knows.

### Model-based methods

- static decision analysis under known state probabilities,
- finite-horizon inventory control,
- MDP value iteration.

These methods use an explicit model of transitions or uncertainty distributions when evaluating decisions.

### Learning-oriented or model-free methods

- epsilon-greedy bandits,
- UCB,
- Thompson Sampling,
- Q-learning.

These methods learn decision-relevant quantities from observations or interaction.

## Visualizations

The repository includes visualizations for:

- simulated outcome distributions for static decision alternatives,
- cumulative reward over decision rounds,
- arm selection frequencies,
- true versus estimated reward probabilities,
- posterior mean probabilities,
- comparisons of cumulative reward and regret across policies,
- finite-horizon optimal inventory policies,
- optimal MDP state values,
- Q-learning returns and episode lengths.

## Reproducibility

The examples use NumPy random generators with configurable random seeds. Fixed seeds are useful for classroom demonstrations because students can reproduce the same results. Changing or removing a seed can be used to study stochastic variability.

## Installation

```bash
pip install -r requirements.txt
```

## Testing

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the regression suite and executes the static decision example with a non-interactive Matplotlib backend.

## Educational Scope

The implementations deliberately favor mathematical transparency and instructional clarity over production-oriented software architecture. They are intended to expose the sequential decision logic rather than hide it behind specialized libraries.

## License

This repository is provided for educational, academic, and other non-commercial use only. Commercial use is not permitted. See the `LICENSE` file for the applicable terms.

Because commercial use is restricted, the license is not an OSI-approved open-source license.
