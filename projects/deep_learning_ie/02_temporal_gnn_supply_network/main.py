from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


SEED = 11
torch.manual_seed(SEED)
np.random.seed(SEED)


@dataclass
class Config:
    nodes: int = 12
    periods: int = 700
    history: int = 12
    train_fraction: float = 0.8
    epochs: int = 80
    learning_rate: float = 2e-3


def make_graph(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    a = np.zeros((n, n), dtype=np.float32)

    # A sparse directed backbone plus a few cross-links.
    for i in range(n - 1):
        a[i, i + 1] = 1.0
    for i in range(n - 2):
        if rng.random() < 0.55:
            a[i, i + 2] = 1.0

    # Self loops stabilize message passing.
    a = a + np.eye(n, dtype=np.float32)

    # Row-normalize incoming aggregation after transpose in the layer.
    degree = a.sum(axis=1, keepdims=True)
    return a / np.maximum(degree, 1.0)


def simulate_network(
    adjacency: np.ndarray,
    periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    n = adjacency.shape[0]

    demand_pressure = rng.normal(0.0, 1.0, size=(periods, n)).astype(np.float32)
    disruptions = (rng.random((periods, n)) < 0.025).astype(np.float32)

    utilization = np.zeros((periods, n), dtype=np.float32)
    lead_time = np.zeros((periods, n), dtype=np.float32)
    utilization[0] = 0.65 + 0.05 * rng.normal(size=n)
    lead_time[0] = 2.0 + 0.2 * rng.normal(size=n)

    for t in range(1, periods):
        upstream_util = adjacency.T @ utilization[t - 1]
        upstream_delay = adjacency.T @ lead_time[t - 1]

        utilization[t] = (
            0.55 * utilization[t - 1]
            + 0.15 * upstream_util
            + 0.10 * demand_pressure[t]
            + 0.22 * disruptions[t]
            + rng.normal(0, 0.025, n)
        )
        utilization[t] = np.clip(utilization[t], 0.0, 1.4)

        lead_time[t] = (
            0.60 * lead_time[t - 1]
            + 0.18 * upstream_delay
            + 0.90 * np.maximum(utilization[t] - 0.85, 0.0)
            + 1.50 * disruptions[t]
            + rng.normal(0, 0.08, n)
        )
        lead_time[t] = np.maximum(lead_time[t], 0.5)

    features = np.stack(
        [
            demand_pressure,
            utilization,
            lead_time,
            disruptions,
        ],
        axis=-1,
    ).astype(np.float32)

    return features, lead_time


def build_windows(
    features: np.ndarray,
    target: np.ndarray,
    history: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for t in range(history, len(features)):
        xs.append(features[t - history : t])
        ys.append(target[t])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


class GraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # x: [batch, nodes, features]
        # Aggregate predecessor information with the fixed graph.
        aggregated = torch.einsum("ij,bjf->bif", a.T, x)
        return torch.relu(self.linear(aggregated))


class TemporalGNN(nn.Module):
    def __init__(self, in_features: int, graph_hidden: int = 32, temporal_hidden: int = 48):
        super().__init__()
        self.gconv = GraphConv(in_features, graph_hidden)
        self.gru = nn.GRU(graph_hidden, temporal_hidden, batch_first=True)
        self.head = nn.Linear(temporal_hidden, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, nodes, features]
        batch, time, nodes, _ = x.shape
        embeddings = []
        for t in range(time):
            embeddings.append(self.gconv(x[:, t], a))
        z = torch.stack(embeddings, dim=1)

        # Each node gets its own temporal sequence while sharing model weights.
        z = z.permute(0, 2, 1, 3).reshape(batch * nodes, time, -1)
        out, _ = self.gru(z)
        pred = self.head(out[:, -1]).reshape(batch, nodes)
        return pred


def main() -> None:
    cfg = Config()
    adjacency = make_graph(cfg.nodes)
    features, target = simulate_network(adjacency, cfg.periods)
    x, y = build_windows(features, target, cfg.history)

    split = int(len(x) * cfg.train_fraction)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    feat_mean = x_train.mean(axis=(0, 1, 2), keepdims=True)
    feat_std = x_train.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    x_train = (x_train - feat_mean) / feat_std
    x_test = (x_test - feat_mean) / feat_std

    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-6
    y_train_scaled = (y_train - y_mean) / y_std

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train_scaled.astype(np.float32))
    x_test_t = torch.from_numpy(x_test)
    a_t = torch.from_numpy(adjacency)

    model = TemporalGNN(in_features=x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(cfg.epochs):
        optimizer.zero_grad()
        pred = model(x_train_t, a_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_test_t, a_t).numpy()
    pred = pred_scaled * y_std + y_mean

    mae = float(np.mean(np.abs(pred - y_test)))
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))

    naive = x_test[:, -1, :, 2] * feat_std[..., 2].squeeze() + feat_mean[..., 2].squeeze()
    naive_mae = float(np.mean(np.abs(naive - y_test)))

    print("Temporal GNN supply-network forecasting")
    print("-" * 56)
    print(f"Test MAE:       {mae:.4f}")
    print(f"Test RMSE:      {rmse:.4f}")
    print(f"Naive MAE:      {naive_mae:.4f}")
    print(f"MAE improvement vs naive: {100.0 * (naive_mae - mae) / naive_mae:.2f}%")


if __name__ == "__main__":
    main()
