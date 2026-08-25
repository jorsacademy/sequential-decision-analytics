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

The repository begins with multi-armed bandits and then extends to a finite-horizon stochastic control problem. This makes it possible to connect learning, optimization, dynamic programming, Operations Research, and Reinforcement Learning within one small teaching repository.

## Topics

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
- Policy evaluation by simulation

## Repository Structure

```text
sequential-decision-analytics/
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   └── SDA_CONCEPTS.md
└── examples/
    ├── epsilon_greedy_bandit.py
    ├── ucb_bandit.py
    ├── thompson_sampling_bandit.py
    ├── compare_bandit_policies.py
    └── finite_horizon_inventory_control.py
```

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

This is the simplest example in the repository and is intended to introduce the decision-observation-learning cycle.

Run:

```bash
python examples/epsilon_greedy_bandit.py
```

## Example 2: Upper Confidence Bound

The UCB example replaces random exploration with an explicit uncertainty bonus. An arm can be attractive because its estimated reward is high or because it has not yet been sampled enough.

The index is based on the form:

```text
UCB(a) = Q(a) + sqrt(c * log(t) / N(a))
```

This example introduces optimism under uncertainty and shows a more structured alternative to epsilon-greedy exploration.

Run:

```bash
python examples/ucb_bandit.py
```

## Example 3: Thompson Sampling

The Thompson Sampling example provides a Bayesian approach to sequential learning. Each Bernoulli arm has a Beta posterior distribution. The policy samples one plausible reward probability from each posterior and selects the arm with the largest sampled value.

Observed successes and failures update the posterior distributions, so exploration emerges naturally from posterior uncertainty.

Run:

```bash
python examples/thompson_sampling_bandit.py
```

## Example 4: Comparing Bandit Policies

A single stochastic simulation can give a misleading impression of algorithm quality. The comparison example therefore evaluates epsilon-greedy, UCB, and Thompson Sampling over many independent replications.

It plots:

- average cumulative reward,
- average pseudo-regret.

Pseudo-regret measures performance relative to an oracle that always chooses the arm with the highest expected reward.

Run:

```bash
python examples/compare_bandit_policies.py
```

## Example 5: Finite-Horizon Inventory Control

The inventory example extends the repository beyond bandits and introduces a more explicit sequential optimization model.

The elements are:

- **State**: current inventory,
- **Action**: order quantity,
- **Uncertainty**: random demand,
- **Transition**: ending inventory,
- **Contribution**: sales revenue minus ordering, holding, and stockout costs,
- **Objective**: maximize expected total profit over a finite horizon.

The model is solved by backward induction using the Bellman recursion:

```text
V_t(s) = max_a E[C_t(s, a, W) + V_(t+1)(S')]
```

This example demonstrates that a sequential decision is evaluated not only by its immediate contribution but also by its effect on future states and future decisions.

Run:

```bash
python examples/finite_horizon_inventory_control.py
```

## Suggested Teaching Sequence

A useful classroom sequence is:

1. epsilon-greedy: basic sequential learning and exploration versus exploitation,
2. UCB: optimism under uncertainty,
3. Thompson Sampling: Bayesian exploration,
4. bandit comparison: cumulative reward, regret, and repeated experiments,
5. inventory control: state transitions, Bellman recursion, and dynamic programming.

The conceptual notes in `docs/SDA_CONCEPTS.md` summarize the mathematical role of each example.

## Visualizations

The repository includes visualizations for:

- cumulative reward over decision rounds,
- arm selection frequencies,
- true versus estimated reward probabilities,
- posterior mean probabilities,
- comparisons of cumulative reward and regret across policies,
- finite-horizon optimal inventory policies.

## Reproducibility

The examples use NumPy random generators with configurable random seeds. Fixed seeds are useful for classroom demonstrations because students can reproduce the same results. Changing or removing a seed can be used to study stochastic variability.

## Installation

```bash
pip install -r requirements.txt
```

## Educational Scope

The implementations deliberately favor mathematical transparency and instructional clarity over production-oriented software architecture. They are intended to expose the sequential decision logic rather than hide it behind specialized libraries.

## License

This repository is provided for educational, academic, and other non-commercial use only. Commercial use is not permitted. See the `LICENSE` file for the applicable terms.

Because commercial use is restricted, the license is not an OSI-approved open-source license.
