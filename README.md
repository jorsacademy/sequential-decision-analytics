# Sequential Decision Analytics

Educational implementations of sequential decision-making algorithms for courses in Optimization, Operations Research, and Reinforcement Learning.

## Purpose

This repository provides compact, transparent examples that illustrate the logic of sequential decision analytics. The examples are designed for teaching rather than for production deployment.

The initial example is an epsilon-greedy multi-armed bandit. It demonstrates how a decision-maker repeatedly:

1. selects an action using current information,
2. observes stochastic feedback,
3. updates action-value estimates, and
4. uses the updated information in later decisions.

This decision-observation-update cycle is the central sequential structure emphasized in the repository.

## Topics

- Sequential decision making
- Exploration versus exploitation
- Multi-armed bandits
- Epsilon-greedy policies
- Online learning
- Incremental action-value estimation
- Cumulative reward
- Learning from stochastic feedback

## Repository Structure

```text
sequential-decision-analytics/
├── README.md
├── LICENSE
├── requirements.txt
└── examples/
    └── epsilon_greedy_bandit.py
```

## Example: Epsilon-Greedy Multi-Armed Bandit

The example contains four Bernoulli reward processes with true reward probabilities:

```python
[0.3, 0.5, 0.7, 0.4]
```

The agent does not use these probabilities when selecting actions. Instead, it learns estimated action values from observed rewards.

With probability `epsilon`, the policy explores by selecting a random arm. Otherwise, it exploits current information by selecting the arm with the highest estimated reward.

The estimated value of a selected arm is updated with the incremental sample-mean formula:

```text
Q_n = Q_(n-1) + (R_n - Q_(n-1)) / n
```

This avoids storing the complete reward history and is a standard action-value update for stationary multi-armed bandit problems.

## Visualizations

The example generates three figures:

- cumulative reward over decision rounds,
- arm selection frequency,
- true versus estimated reward probabilities.

Together, these figures show both performance and learning behavior.

## Reproducibility

The example uses NumPy random generators and a configurable random seed. The default seed makes classroom demonstrations reproducible. Removing or changing the seed can be used to demonstrate stochastic variability across simulations.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Example

```bash
python examples/epsilon_greedy_bandit.py
```

## Educational Scope

This repository is intended for conceptual and instructional use. The examples deliberately favor clarity over large-scale software architecture or application-specific complexity.

## License

This repository is provided for educational, academic, and other non-commercial use only. Commercial use is not permitted. See the `LICENSE` file for the applicable terms.

This is not an OSI-approved open-source license because it restricts commercial use.
