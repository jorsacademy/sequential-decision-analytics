# Project 3 — Constrained PPO for Production-Inventory Control

## Industrial-engineering question

How should a production system choose output dynamically when demand is uncertain, capacity is limited, inventory is costly, and excessive shortage is not acceptable?

This project uses **Proximal Policy Optimization (PPO)** with a simple **Lagrangian constraint penalty**.

## State

The policy observes:

- current net inventory;
- current demand signal;
- previous demand;
- current capacity.

## Action

The action is a discrete production quantity between zero and the current nominal capacity.

## Objective

The operating cost contains:

- production cost;
- holding cost;
- backlog/shortage cost.

A separate shortage-rate signal acts as a soft service constraint.

The Lagrange multiplier is updated during training:

```text
lambda <- max(0, lambda + step * (observed_shortage - target_shortage))
```

This makes the project more relevant than unconstrained toy RL: many industrial systems optimize cost subject to service, safety, capacity, or risk requirements.

## What to learn from this project

- actor-critic architecture;
- stochastic policies;
- discounted returns and generalized advantage estimation;
- PPO probability-ratio clipping;
- entropy regularization;
- Lagrangian constraint handling.

## Extensions

- Add setup costs and minimum lot sizes.
- Replace a single inventory location with a multi-echelon system.
- Use continuous actions with a Gaussian or Beta policy.
- Add forecast features from Project 1.
- Compare PPO against dynamic programming on a small state space.
- Compare constrained RL with model predictive control.
