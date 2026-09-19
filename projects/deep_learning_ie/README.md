# Deep Learning for Industrial Engineering: Project Suite

This suite focuses on deep learning methods that map naturally to industrial engineering decisions. It deliberately excludes image processing and quality-control examples.

The three projects are designed as a progression:

1. **GRU vs Transformer for multi-horizon demand forecasting**
   - Learn sequence modeling, recurrent gating, self-attention, positional information, and multi-step forecasting.
   - Evaluate models not only with forecast error but also with a downstream inventory-cost proxy.

2. **Temporal GNN for supply-network delay propagation**
   - Learn message passing on a directed network and temporal aggregation.
   - Model how local congestion or disruption propagates through supplier-customer dependencies.

3. **Constrained PPO for production-inventory control**
   - Learn actor-critic reinforcement learning, PPO clipping, generalized advantage estimation, and Lagrangian constraint handling.
   - Optimize sequential production decisions while controlling shortage risk.

## Why these three?

Industrial engineering problems are rarely pure prediction problems. The value of a model depends on how its output changes a decision.

- Sequence models are useful when the system state evolves over time and recent history matters.
- GNNs are useful when the structure of the system is a network and interactions are not exchangeable.
- Deep RL is useful when actions affect future states and the decision policy must adapt online.

A useful learning principle is:

> **baseline -> predictive model -> decision metric -> stress test**

Each project therefore includes a simple synthetic data-generating process, a neural model, and a decision-oriented evaluation target.

## Installation

From the repository root:

```bash
pip install -r projects/deep_learning_ie/requirements.txt
```

Run each project:

```bash
python projects/deep_learning_ie/01_sequence_forecasting/main.py
python projects/deep_learning_ie/02_temporal_gnn_supply_network/main.py
python projects/deep_learning_ie/03_constrained_ppo_inventory/main.py
```

These examples favor transparency over production engineering. They are intended to be read, modified, and extended.
