# Project 2 — Temporal GNN for Supply-Network Delay Propagation

## Industrial-engineering question

When congestion or disruption occurs at one supplier, how does it affect downstream lead times across the network?

A conventional tabular model treats rows as independent observations. A supply network is not independent: facilities are connected by material-flow relationships. A GNN explicitly uses those relationships.

This project combines:

- graph message passing across suppliers and plants;
- a GRU across time;
- node-level next-period lead-time prediction.

## Representation

Each node has time-varying features:

- local demand pressure,
- capacity utilization,
- current lead time,
- disruption indicator.

The graph adjacency matrix represents directed supply dependencies.

At every time step:

```text
node features -> graph convolution -> node embeddings
```

The node embeddings are then passed through a GRU over time:

```text
graph embeddings over time -> GRU -> next-period lead time
```

## Why this is useful for industrial engineers

This architecture is appropriate when the system state depends on both:

1. **where an observation sits in the network**, and
2. **how conditions evolved over time**.

Examples include:

- multi-tier supplier networks;
- intermodal transportation systems;
- production-routing networks;
- energy and utility networks;
- service networks with upstream/downstream dependencies.

## Extensions

- Replace the fixed adjacency matrix with learned edge weights.
- Add edge features such as transportation time or contractual capacity.
- Predict disruption probability and lead time jointly.
- Feed predictions into a robust sourcing or network-design model.
- Compare against node-wise GRU models that ignore network structure.
