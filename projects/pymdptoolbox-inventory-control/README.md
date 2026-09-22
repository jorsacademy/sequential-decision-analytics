# PyMDPToolbox Inventory Control

A finite-state stochastic inventory-control example solved with `pymdptoolbox` using both Value Iteration and Policy Iteration.

## Problem

The state is on-hand inventory from `0` to `max_inventory`. The action is an order quantity from `0` to `max_order`. Demand follows a Poisson distribution. Unmet demand is treated as lost sales and incurs a penalty; remaining stock incurs holding cost, and replenishment incurs order cost.

The model builds:

- transition probabilities `P[A, S, S]`
- expected rewards `R[S, A]`
- discounted infinite-horizon policies solved with PyMDPToolbox

## Installation

```bash
python -m pip install -e '.[dev]'
```

## Run

```bash
pymdptoolbox-inventory
```

The CLI solves the same MDP with both algorithms and prints the order quantity chosen in each inventory state.

## Test

```bash
pytest
```

Tests validate stochastic matrices, configuration constraints, storage-capacity behavior, Value Iteration / Policy Iteration agreement, CLI output, and a minimum 90% project coverage threshold.

## Compatibility note

The upstream `pymdptoolbox` package is mature but old: its latest PyPI release is `4.0b3` from 2015. This repository therefore treats modern Python compatibility as something to verify continuously rather than assume. GitHub Actions tests Python 3.10 through 3.14 on every push and pull request.

## Q-learning on the same MDP

The inventory example from `im_rl.ipynb` has been adapted to this repository's
**immediate-replenishment, lost-sales model**. It does not retain the notebook's
separate on-order state or lead-time assumptions. DP and Q-learning now use the
same storage cap, effective-order cost, Poisson demand and infinite-horizon
discounted objective. The DP reward includes the full expected Poisson tail
shortage cost, even though tail transitions all reach zero stock.

```bash
python -m inventory_mdp.q_learning --updates 200000 --seeds 42 43 44
```

Training uses uniformly sampled state/action resets and simulated demand with
per-pair learning rates `visits**(-0.6)`. This is off-policy Q-learning with a
resettable generative simulator, not an online agent following one trajectory.
It does not read the DP transition matrix or optimal policy during training.

The comparison includes Value Iteration, Policy Iteration and a base-stock level
tuned using the known model. All policies are evaluated from zero stock on the
same 100 held-out demand paths of 200 periods (seed 10000). Each training seed
is reported separately. Outputs include period cost, its standard error across
episodes, demand-weighted fill rate and **exact discounted cost from empty**.
The last metric matches the training objective; finite-horizon average cost is
an additional operational KPI. Finite training does not guarantee an optimal
policy, and ties can produce different but equally valuable actions.

Tests independently enumerate Poisson outcomes to reconcile simulator and DP,
check a small-instance learned policy against DP, and verify paired evaluation.

A reference run is saved in [`results/q_learning_comparison.json`](results/q_learning_comparison.json).
At the default settings, exact discounted cost from empty is 63.1847 for DP
and 63.5871–63.7899 for the three learned policies. This finite training run
approaches but does not beat the DP optimum; the result is not a general ranking
of algorithms. Equivalent overflow orders may differ in the reported policies.
